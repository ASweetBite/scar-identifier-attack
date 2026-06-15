import itertools
import math
import re
from typing import Dict, Any, List

import torch
import torch.nn.functional as F


class LightweightCandidateGenerator:
    def __init__(self, mlm_engine, analyzer, config, llm_client=None):
        # Initializes the lightweight candidate generator with an MLM engine and optional LLM client.
        self.mlm_engine = mlm_engine
        self.analyzer = analyzer
        self.config = config
        self.llm_client = llm_client
        cg_cfg = self.config.get('candidate_generation', {})
        stats_path = cg_cfg.get('naming_stats_path', 'naming_stats.json')
        from utils.scorer import StatisticalNamingScorer
        self.scorer = StatisticalNamingScorer(stats_path)

    @torch.no_grad()
    def _calculate_perplexity_batch(self, texts: List[str], batch_size: int = 4) -> List[float]:
        # Calculates perplexity scores for a batch of text inputs using the LLM.
        if not texts or not self.llm_client:
            return [0.0] * len(texts)

        tokenizer = getattr(self.llm_client, 'tokenizer', None)
        model = getattr(self.llm_client, 'model', None)
        if not tokenizer or not model: return [0.0] * len(texts)

        device = model.device
        ppls = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to(device)

            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            shift_mask = inputs["attention_mask"][..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size(0), shift_labels.size(1)) * shift_mask

            seq_lens = torch.clamp(shift_mask.sum(dim=1), min=1.0)
            seq_loss = loss.sum(dim=1) / seq_lens

            for val in seq_loss:
                try:
                    ppls.append(math.exp(val.item()))
                except OverflowError:
                    ppls.append(float('inf'))

            del inputs, outputs, loss
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return ppls

    def _get_dynamic_threshold(self, target_name: str, cand: str, base_threshold: float) -> float:
        # Adjusts the semantic threshold dynamically based on structural similarity between names.
        target_lower = target_name.lower()
        cand_lower = cand.lower()

        if cand_lower.endswith(f"_{target_lower}") or cand_lower.startswith(f"{target_lower}_"):
            return min(0.99, base_threshold + 0.05)

        if target_lower in cand_lower:
            return min(0.99, base_threshold + 0.03)

        import Levenshtein
        if Levenshtein.distance(target_lower, cand_lower) <= 2:
            return min(0.99, base_threshold + 0.07)

        target_parts, target_sep = self._split_identifier(target_name)
        cand_parts, cand_sep = self._split_identifier(cand)

        if len(target_parts) > 1 and len(target_parts) == len(cand_parts) and target_sep == cand_sep:
            identical_count = sum(1 for t, c in zip(target_parts, cand_parts) if t.lower() == c.lower())

            if identical_count > 0:
                overlap_ratio = identical_count / len(target_parts)

                if overlap_ratio >= 0.5:
                    base_penalty = 0.02
                    ratio_penalty = overlap_ratio * 0.06
                    penalty = base_penalty + ratio_penalty

                    if target_parts[-1].lower() != cand_parts[-1].lower():
                        penalty += 0.015

                    return min(0.99, base_threshold + penalty)

        return base_threshold

    def _get_mutation_pattern(self, target_name: str, cand_name: str) -> str:
        # Extracts the specific mutation pattern of target and candidate identifiers.
        t_parts, t_sep = self._split_identifier(target_name)
        c_parts, c_sep = self._split_identifier(cand_name)

        if len(t_parts) == len(c_parts) and t_sep == c_sep and len(t_parts) > 1:
            pattern = []
            identical_count = 0
            for t, c in zip(t_parts, c_parts):
                if t.lower() == c.lower():
                    pattern.append(t.lower())
                    identical_count += 1
                else:
                    pattern.append('*')

            if identical_count > 0 and identical_count < len(t_parts):
                sep_display = '_' if t_sep == '_' else ''
                return sep_display.join(pattern)

        return '*'

    def _detect_naming_style(self, name: str) -> str:
        # Identifies the naming convention style of the given variable or function.
        if not name:
            return 'unknown'
        core_name = name.strip('_')

        if not core_name:
            return 'unknown'
        if '_' in core_name:
            return 'SCREAMING_SNAKE' if core_name.isupper() else 'snake_case'
        if core_name.islower():
            return 'single_lower'
        if core_name.isupper():
            return 'single_upper'
        if core_name[0].islower():
            return 'camelCase'
        if core_name[0].isupper():
            return 'PascalCase'
        return 'unknown'

    def _matches_style(self, original_style: str, candidate: str) -> bool:
        # Determines if the candidate matches the style format of the original name.
        cand_style = self._detect_naming_style(candidate)
        if original_style in ('snake_case', 'camelCase', 'PascalCase') and cand_style == 'single_lower': return True
        if original_style == 'single_lower' and cand_style in ('snake_case', 'camelCase'): return True
        if original_style == 'single_upper' and cand_style == 'SCREAMING_SNAKE': return True
        return cand_style == original_style

    def _split_identifier(self, name: str):
        # Separates a multi-word identifier into individual lexical token components.
        if '_' in name:
            return name.split('_'), '_'
        else:
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
            if not parts or (len(parts) == 1 and parts[0] == name): return [name], ''
            return parts, 'camel'

    def _build_masked_string(self, parts: List[str], start: int, end: int, num_masks: int, style: str, mask_token: str,
                             target_name: str) -> str:
        # Synthesizes a code string with mask tokens inserted for MLM prediction.
        mask_list = [mask_token] * num_masks
        new_parts = parts[:start] + mask_list + parts[end:]

        if style == '_':
            return "_".join(new_parts)
        elif style == 'camel':
            res = []
            for j, p in enumerate(new_parts):
                if p == mask_token:
                    res.append(p)
                else:
                    res.append(p.lower() if j == 0 and target_name[0].islower() else p.capitalize())
            return "".join(res).replace(mask_token.capitalize(), mask_token)
        else:
            return mask_token

    def _extract_local_context_ast(self, code_bytes: bytes, target_start: int, target_end: int) -> tuple[str, str]:
        # Extracts neighboring syntax fragments surrounding the target variable from AST.
        from tree_sitter import Parser
        parser = Parser()
        parser.language = self.analyzer.language
        tree = parser.parse(code_bytes)
        node = tree.root_node.descendant_for_byte_range(target_start, target_end)

        if not node:
            line_start = code_bytes.rfind(b'\n', 0, target_start) + 1
            line_end = code_bytes.find(b'\n', target_end)
            if line_end == -1: line_end = len(code_bytes)
            return (code_bytes[line_start:target_start].decode("utf-8", errors="replace"),
                    code_bytes[target_end:line_end].decode("utf-8", errors="replace"))

        statement_node = node
        stop_parent_types = {'compound_statement', 'translation_unit', 'function_definition', 'for_statement',
                             'while_statement', 'if_statement'}

        while statement_node.parent and statement_node.parent.type not in stop_parent_types:
            statement_node = statement_node.parent

        stmt_start = statement_node.start_byte
        stmt_end = statement_node.end_byte
        local_prefix = code_bytes[stmt_start:target_start].decode("utf-8", errors="replace")
        local_suffix = code_bytes[target_end:stmt_end].decode("utf-8", errors="replace")
        return local_prefix, local_suffix

    def _find_best_context_occurrence(self, code_bytes: bytes, occurrences: List[dict]) -> int:
        # Selects the context occurrence that provides the richest syntactic environment.
        if len(occurrences) <= 1: return 0
        best_idx, max_score = 0, -1.0
        search_limit = min(len(occurrences), 10)

        for i in range(search_limit):
            occ = occurrences[i]
            local_prefix, local_suffix = self._extract_local_context_ast(code_bytes, occ['start'], occ['end'])
            score = len(local_prefix) + len(local_suffix)
            if '(' in local_suffix or ',' in local_suffix: score += 100
            if any(k in local_prefix for k in ['if ', 'while ', 'for ', 'return ']): score += 80
            if re.search(r'=\s*(0|NULL|nullptr|false|true|\{\})\s*;', local_suffix): score -= 150
            if score > max_score:
                max_score = score
                best_idx = i
        return best_idx

    def _get_model_logits_batched(self, cropped_codes: List[str]):
        # Runs MLM inference to predict masked token logits for a batch of code inputs.
        if not cropped_codes: return None, []
        inputs = self.mlm_engine.tokenizer(
            cropped_codes, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.mlm_engine.device)
        mask_token_id = self.mlm_engine.tokenizer.mask_token_id

        with torch.no_grad():
            batch_logits = self.mlm_engine.model(**inputs).logits

        batch_mask_indices = [(inputs.input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0] for i in
                              range(batch_logits.size(0))]
        return batch_logits, batch_mask_indices

    def _decode_words(self, mask_logits, top_k, allow_underscore=False, required_length=None):
        # Decodes token predictions from model logits into standard text representations.
        _, top_indices = torch.topk(mask_logits, top_k, dim=-1)
        words = []
        for idx in top_indices:
            w = self.mlm_engine.tokenizer.decode([idx]).strip().replace('Ġ', '').replace('##', '')
            if allow_underscore:
                w = re.sub(r'[^a-zA-Z0-9_]', '', w)
                if not w or (not w[0].isalpha() and w[0] != '_'): continue
            else:
                w = re.sub(r'[^a-zA-Z0-9]', '', w)
                if not w: continue
            if required_length is not None and len(w) != required_length: continue
            words.append(w)
        return words

    def _get_variable_token_embeddings(self, prefixes: List[str], var_names: List[str], suffixes: List[str],
                                       batch_size: int = 64) -> torch.Tensor:
        # Extracts contextual token embeddings representing variable semantics.
        all_embeddings = []
        tokenizer = self.mlm_engine.tokenizer
        full_texts = [p + v + s for p, v, s in zip(prefixes, var_names, suffixes)]
        device = self.mlm_engine.device
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[
            0] >= 8 else torch.float16
        self.mlm_engine.model.to(dtype)

        for i in range(0, len(full_texts), batch_size):
            batch_texts = full_texts[i: i + batch_size]
            batch_prefixes = prefixes[i: i + batch_size]
            batch_vars = var_names[i: i + batch_size]

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(
                device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype):
                outputs = self.mlm_engine.model.roberta(**inputs)
                last_hidden = outputs.last_hidden_state

            cached_p_tokens = {}
            for b_idx in range(len(batch_texts)):
                p_text = batch_prefixes[b_idx]
                if p_text not in cached_p_tokens:
                    cached_p_tokens[p_text] = tokenizer.encode(p_text, add_special_tokens=False)

                p_tokens = cached_p_tokens[p_text]
                pv_tokens = tokenizer.encode(p_text + batch_vars[b_idx], add_special_tokens=False)

                shared_len = sum(1 for pt, pvt in zip(p_tokens, pv_tokens) if pt == pvt)
                start_idx = min(shared_len + 1, 255)
                end_idx = min(max(start_idx + 1, len(pv_tokens) + 1), 256)

                pooled = last_hidden[b_idx, start_idx:end_idx, :].mean(dim=0)
                all_embeddings.append(pooled.to(torch.float32).cpu())

        return torch.stack(all_embeddings)

    def _is_trivial_change(self, target_name: str, cand: str) -> bool:
        # Identifies whether the candidate represents a trivial spelling change from original name.
        target_parts, _ = self._split_identifier(target_name)
        cand_parts, _ = self._split_identifier(cand)
        if len(target_parts) > 2 and len(cand_parts) > 0:
            identical_count = sum(1 for p1, p2 in zip(target_parts, cand_parts) if p1.lower() == p2.lower())
            change_ratio = 1.0 - (identical_count / max(len(target_parts), len(cand_parts)))
            return change_ratio <= 0.33
        return False

    def _verify_ast_single(self, cand: str, ctx: dict) -> str | None:
        # Validates AST compliance for a proposed identifier replacement.
        if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand):
            return None
        try:
            from utils.ast_tools import CodeTransformer
            CodeTransformer.validate_and_apply(ctx['code_bytes'], ctx['identifiers'], {ctx['target_name']: cand},
                                               analyzer=self.analyzer)
            return cand
        except Exception:
            return None

    def _verify_and_filter(self, candidate_list, quota, final_candidates, ctx):
        # Filters candidates using semantic similarity thresholds, syntactic heuristic scores, and AST renaming verification.
        base_threshold = ctx.get('semantic_threshold', 0.85)
        entity_type = ctx.get('entity_type', 'VARIABLE')

        base_cands = []
        for cand in candidate_list:
            if cand in ctx['keywords'] or cand == ctx['target_name']: continue
            if ctx['preserve_style'] and not self._matches_style(ctx['original_style'], cand): continue
            base_cands.append(cand)
        if not base_cands: return 0

        orig_emb = None
        if base_threshold > 0:
            orig_emb = self._get_variable_token_embeddings(
                [ctx['local_prefix']], [ctx['target_name']], [ctx['local_suffix']]
            ).to(self.mlm_engine.device)

        added = 0
        CHUNK_SIZE = max(50, quota * 2)
        target_name = ctx['target_name']
        target_parts, _ = self._split_identifier(target_name)
        return_type = ctx.get('return_type', None)

        for i in range(0, len(base_cands), CHUNK_SIZE):
            if added >= quota: break

            chunk = base_cands[i: i + CHUNK_SIZE]
            filtered_chunk, heuristic_bonuses = [], []

            for cand in chunk:
                bonus = 0.0
                if hasattr(self, 'scorer'):
                    cand_parts, _ = self._split_identifier(cand)
                    bonus = self.scorer.calculate_heuristic_score(cand_parts, entity_type, target_parts, return_type)
                if bonus <= -100: continue
                if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand): continue
                filtered_chunk.append(cand)
                heuristic_bonuses.append(bonus)

            if not filtered_chunk: continue

            semantically_valid = []
            if base_threshold > 0:
                prefixes = [ctx['local_prefix']] * len(filtered_chunk)
                suffixes = [ctx['local_suffix']] * len(filtered_chunk)
                cand_embs = self._get_variable_token_embeddings(prefixes, filtered_chunk, suffixes).to(
                    self.mlm_engine.device)
                sims = F.cosine_similarity(orig_emb, cand_embs)
                for cand, sim, bonus in zip(filtered_chunk, sims, heuristic_bonuses):
                    final_score = sim.item() + bonus
                    dynamic_threshold = self._get_dynamic_threshold(
                        target_name, cand, base_threshold
                    )

                    if final_score >= dynamic_threshold:
                        semantically_valid.append((cand, final_score))
            else:
                semantically_valid = [(cand, 1.0) for cand in filtered_chunk]

            for cand, final_score in semantically_valid:
                if added >= quota: break
                valid_cand = self._verify_ast_single(cand, ctx)
                if valid_cand and valid_cand not in final_candidates:
                    final_candidates.append(valid_cand)
                    added += 1

        return added

    def generate_candidates(self, batch_tasks: List[Dict[str, Any]], top_k_mlm: int = 40, top_n_keep: int = 20,
                            ) -> Dict[str, List[str]]:
        # Generates semantic replacement candidates using contextual masked language modeling.
        results = {task["target_name"]: [] for task in batch_tasks}
        mlm_tracking = []
        task_metadata = {}
        mask_token = self.mlm_engine.tokenizer.mask_token

        for task_idx, task in enumerate(batch_tasks):
            target_name = task["target_name"]

            slice_code_str = task["code_str"]
            slice_code_bytes = slice_code_str.encode("utf-8")

            slice_identifiers = self.analyzer.extract_identifiers(slice_code_bytes)

            if target_name not in slice_identifiers:
                continue

            best_occ_idx = self._find_best_context_occurrence(slice_code_bytes, slice_identifiers[target_name])
            target_info = slice_identifiers[target_name][best_occ_idx]

            leading_m = re.match(r'^_+', target_name)
            leading_us = leading_m.group(0) if leading_m else ""
            core_name = target_name[len(leading_us):] if leading_us else target_name

            entity_type = 'BOOLEAN_VAR' if core_name.startswith(('is_', 'has_', 'can_', 'should_')) else (
                'FUNCTION' if target_info.get('entity_type') == 'function' else 'VARIABLE')

            original_style = self._detect_naming_style(target_name)
            parts, style = self._split_identifier(core_name)

            prefix_bytes = slice_code_bytes[:target_info['start']]
            suffix_bytes = slice_code_bytes[target_info['end']:]

            local_prefix = prefix_bytes.decode("utf-8", errors="replace")
            local_suffix = suffix_bytes.decode("utf-8", errors="replace")

            task_metadata[task_idx] = {
                "target_name": target_name, "core_name": core_name, "leading_us": leading_us,
                "parts": parts, "style": style, "n_parts": len(parts),
                "entity_type": entity_type, "original_style": original_style,

                "full_code_str": task["full_code_str"],
                "full_code_bytes": task["full_code_str"].encode("utf-8"),
                "full_identifiers": task.get("full_identifiers", slice_identifiers),

                "local_prefix": local_prefix, "local_suffix": local_suffix,
                "raw_mlm_cands": []
            }

            MAX_CHAR_LIMIT = 2500

            prefix_str = local_prefix[-MAX_CHAR_LIMIT:] if len(local_prefix) > MAX_CHAR_LIMIT else local_prefix
            suffix_str = local_suffix[:MAX_CHAR_LIMIT] if len(local_suffix) > MAX_CHAR_LIMIT else local_suffix

            variations = []
            if len(parts) == 1:
                variations.extend([
                    {'expand_mode': 'none', 'num_masks': 1, 'masked_str': leading_us + mask_token},
                    {'expand_mode': 'prefix', 'num_masks': 1, 'masked_str': leading_us + f"{mask_token}_{core_name}"},
                    {'expand_mode': 'suffix', 'num_masks': 1, 'masked_str': leading_us + f"{core_name}_{mask_token}"}
                ])
            else:
                for i in range(len(parts)):
                    m_str = self._build_masked_string(parts, i, i + 1, 1, style, mask_token, core_name)
                    variations.append({'expand_mode': 'sub', 'start': i, 'end': i + 1, 'num_masks': 1,
                                       'masked_str': leading_us + m_str})
                if len(parts) >= 2:
                    for i in range(len(parts) - 1):
                        m_str = self._build_masked_string(parts, i, i + 2, 2, style, mask_token, core_name)
                        variations.append({'expand_mode': 'sub', 'start': i, 'end': i + 2, 'num_masks': 2,
                                           'masked_str': leading_us + m_str})

            for var in variations:
                mlm_tracking.append({"task_idx": task_idx, "cropped_code": prefix_str + var['masked_str'] + suffix_str,
                                     "variation_info": var})

        if not task_metadata: return results

        all_cropped_codes = [item["cropped_code"] for item in mlm_tracking]
        batch_logits, batch_mask_indices = self._get_model_logits_batched(all_cropped_codes)

        def _join_parts(new_parts, orig_name, st):
            if st == '_':
                return "_".join(new_parts)
            elif st == 'camel':
                return "".join(
                    p.lower() if j == 0 and orig_name[0].islower() else p.capitalize() for j, p in enumerate(new_parts))
            return "".join(new_parts)

        if batch_logits is not None:
            for i, track_info in enumerate(mlm_tracking):
                meta = task_metadata[track_info["task_idx"]]
                var_info = track_info["variation_info"]
                logits = batch_logits[i:i + 1]
                mask_indices = batch_mask_indices[i]
                num_masks = var_info.get('num_masks', 1)

                core_name = meta["core_name"]
                leading_us = meta["leading_us"]

                if len(mask_indices) < num_masks: continue

                if num_masks == 1:
                    words = self._decode_words(logits[0, mask_indices[0], :], top_k_mlm)
                    for w in words:
                        if var_info.get('expand_mode') == 'prefix':
                            meta["raw_mlm_cands"].append(f"{leading_us}{w}_{core_name}")
                        elif var_info.get('expand_mode') == 'suffix':
                            meta["raw_mlm_cands"].append(f"{leading_us}{core_name}_{w}")
                        else:
                            if meta["n_parts"] == 1:
                                meta["raw_mlm_cands"].append(f"{leading_us}{w}")
                            else:
                                joined = _join_parts(
                                    meta["parts"][:var_info['start']] + [w] + meta["parts"][var_info['end']:],
                                    core_name, meta["style"])
                                meta["raw_mlm_cands"].append(f"{leading_us}{joined}")
                elif num_masks == 2:
                    top_k_2holes = min(4, max(2, top_k_mlm // 4))
                    words1 = self._decode_words(logits[0, mask_indices[0], :], top_k_2holes)
                    words2 = self._decode_words(logits[0, mask_indices[1], :], top_k_2holes)
                    for w1, w2 in itertools.product(words1, words2):
                        joined = _join_parts(
                            meta["parts"][:var_info['start']] + [w1, w2] + meta["parts"][var_info['end']:],
                            core_name, meta["style"])
                        meta["raw_mlm_cands"].append(f"{leading_us}{joined}")

        for t_idx, meta in task_metadata.items():
            unique_mlm_cands = list(dict.fromkeys(meta["raw_mlm_cands"]))

            cg_cfg = self.config.get('candidate_generation', {})
            lw_cfg = cg_cfg.get('lightweight', {})


            ctx = {
                'code_bytes': meta["full_code_bytes"],
                'full_code_str': meta["full_code_str"],
                'target_name': meta["target_name"],
                'identifiers': meta["full_identifiers"],
                'keywords': self.analyzer.keywords,
                'original_style': meta["original_style"],
                'local_prefix': meta["local_prefix"],
                'local_suffix': meta["local_suffix"],

                'semantic_threshold': lw_cfg.get('semantic_threshold', 0.85),
                'preserve_style': cg_cfg.get('preserve_style', True),
                'ppl_max_ratio': cg_cfg.get('ppl_max_ratio', 1.2),
                'ppl_max_abs': cg_cfg.get('ppl_max_abs', 50.0),

                'entity_type': meta["entity_type"],
                'return_type': next(
                    (u['return_type'] for u in meta["full_identifiers"].get(meta["target_name"], []) if
                     u.get('return_type')), None),
            }

            final_candidates = []
            self._verify_and_filter(unique_mlm_cands, top_n_keep, final_candidates, ctx)
            results[meta["target_name"]] = final_candidates

        return results