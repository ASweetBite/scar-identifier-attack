import json
import re
from typing import Dict, Any, List

import torch
import torch.nn.functional as F


class HeavyWeightCandidateGenerator:
    def __init__(self, embedder, llm_client, analyzer, config):
        # Initializes the heavyweight candidate generator utilizing a deep-semantic LLM.
        self.embedder = embedder
        self.llm_client = llm_client
        self.analyzer = analyzer
        self.config = config
        cg_cfg = self.config.get('candidate_generation', {})
        stats_path = cg_cfg.get('naming_stats_path', 'naming_stats.json')
        from utils.scorer import StatisticalNamingScorer
        self.scorer = StatisticalNamingScorer(stats_path)

    def _detect_naming_style(self, name: str) -> str:
        # Detects the naming convention style of the given identifier name.
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
        # Verifies if the candidate identifier matches the expected original styling.
        cand_style = self._detect_naming_style(candidate)
        if original_style in ('snake_case', 'camelCase', 'PascalCase') and cand_style == 'single_lower': return True
        if original_style == 'single_lower' and cand_style in ('snake_case', 'camelCase'): return True
        if original_style == 'single_upper' and cand_style == 'SCREAMING_SNAKE': return True
        return cand_style == original_style

    def _split_identifier(self, name: str):
        # Splits snake_case or camelCase identifiers into individual token parts.
        if '_' in name:
            return name.split('_'), '_'
        else:
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
            if not parts or (len(parts) == 1 and parts[0] == name): return [name], ''
            return parts, 'camel'

    def _extract_local_context_ast(self, code_bytes: bytes, target_start: int, target_end: int) -> tuple[str, str]:
        # Extracts local prefix and suffix string context for an AST range.
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
        # Computes a score to select the best occurrence of a variable based on syntax complexity.
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

    def _get_variable_token_embeddings(self, prefixes: List[str], var_names: List[str], suffixes: List[str],
                                       batch_size: int = 64) -> torch.Tensor:
        # Generates token-level embeddings for variables within their syntax contexts.
        all_embeddings = []
        tokenizer = self.embedder.tokenizer
        full_texts = [p + v + s for p, v, s in zip(prefixes, var_names, suffixes)]

        device = self.embedder.device
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[
            0] >= 8 else torch.float16
        self.embedder.model.to(dtype)

        for i in range(0, len(full_texts), batch_size):
            batch_texts = full_texts[i: i + batch_size]
            batch_prefixes = prefixes[i: i + batch_size]
            batch_vars = var_names[i: i + batch_size]

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(
                device)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype):
                outputs = self.embedder.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]

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

    def _verify_ast_single(self, cand: str, ctx: dict) -> str | None:
        # Performs isolated AST syntax renaming check for a candidate name.
        if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand):
            return None
        try:
            from utils.ast_tools import CodeTransformer
            CodeTransformer.validate_and_apply(ctx['code_bytes'], ctx['identifiers'], {ctx['target_name']: cand},
                                               analyzer=self.analyzer)
            return cand
        except Exception:
            return None

    def _is_trivial_change(self, target_name: str, cand: str) -> bool:
        # Checks if the candidate is a trivial change from the target variable.
        target_parts, _ = self._split_identifier(target_name)
        cand_parts, _ = self._split_identifier(cand)
        if len(target_parts) > 2 and len(cand_parts) > 0:
            identical_count = sum(1 for p1, p2 in zip(target_parts, cand_parts) if p1.lower() == p2.lower())
            change_ratio = 1.0 - (identical_count / max(len(target_parts), len(cand_parts)))
            return change_ratio <= 0.33
        return False

    def _verify_and_filter(self, candidate_list, quota, final_candidates, ctx):
        # Filters candidates using semantic similarity thresholds, syntactic heuristic scores, and AST renaming verification.
        base_threshold = ctx.get('semantic_threshold', 0.85)
        entity_type = ctx.get('entity_type', 'VARIABLE')

        base_cands = []
        for cand in candidate_list:
            if cand in ctx['keywords'] or cand == ctx['target_name']:
                print(f"        🚫 [Filter | Keyword/Self] '{cand}'")
                continue
            if ctx['preserve_style'] and not self._matches_style(ctx['original_style'], cand):
                print(f"        🚫 [Filter | Style Clash] '{cand}' (Expected: {ctx['original_style']})")
                continue
            base_cands.append(cand)

        if not base_cands: return 0

        orig_emb = None
        if base_threshold > 0:
            orig_emb = self._get_variable_token_embeddings(
                [ctx['local_prefix']], [ctx['target_name']], [ctx['local_suffix']]
            ).to(self.embedder.device)

        added = 0
        CHUNK_SIZE = max(50, quota * 2)
        target_name = ctx['target_name']
        target_parts, _ = self._split_identifier(target_name)
        return_type = ctx.get('return_type', None)

        for i in range(0, len(base_cands), CHUNK_SIZE):
            if added >= quota: break

            chunk = base_cands[i: i + CHUNK_SIZE]
            filtered_chunk = []
            heuristic_bonuses = []

            for cand in chunk:
                bonus = 0.0
                if hasattr(self, 'scorer'):
                    cand_parts, _ = self._split_identifier(cand)
                    bonus = self.scorer.calculate_heuristic_score(
                        cand_parts, entity_type, target_parts=target_parts, return_type=return_type
                    )

                if bonus <= -100:
                    continue
                if not self.analyzer.can_rename_to(ctx['code_bytes'], ctx['target_name'], cand):
                    continue

                filtered_chunk.append(cand)
                heuristic_bonuses.append(bonus)

            if not filtered_chunk: continue

            semantically_valid = []
            if base_threshold > 0:
                prefixes = [ctx['local_prefix']] * len(filtered_chunk)
                suffixes = [ctx['local_suffix']] * len(filtered_chunk)

                cand_embs = self._get_variable_token_embeddings(prefixes, filtered_chunk, suffixes).to(
                    self.embedder.device)
                sims = F.cosine_similarity(orig_emb, cand_embs)

                for cand, sim, bonus in zip(filtered_chunk, sims, heuristic_bonuses):
                    final_score = sim.item() + bonus
                    if final_score >= base_threshold:
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

    def _build_llm_prompt(self, context_code: str, target_name: str, style: str, top_n: int, entity_type: str, n_parts: int) -> str:
        # Formulates a strict, JSON-enforced instructions prompt for candidate generation.
        if style == 'camelCase':
            ex_var = "dataBuffer"
            ex_bool = "'isReady', 'hasData'"
            ex_func = "'getData', 'updateState'"
            ex_short = "'shmInfo', 'memData', 'idx'"
        elif style == 'PascalCase':
            ex_var = "DataBuffer"
            ex_bool = "'IsReady', 'HasData'"
            ex_func = "'GetData', 'UpdateState'"
            ex_short = "'ShmInfo', 'MemData', 'Idx'"
        elif style == 'SCREAMING_SNAKE':
            ex_var = "DATA_BUFFER"
            ex_bool = "'IS_READY', 'HAS_DATA'"
            ex_func = "'GET_DATA', 'UPDATE_STATE'"
            ex_short = "'SHM_INFO', 'MEM_DATA', 'IDX'"
        else:
            ex_var = "data_buffer"
            ex_bool = "'is_ready', 'has_data'"
            ex_func = "'get_data', 'update_state'"
            ex_short = "'shm_info', 'mem_data', 'idx'"

        if entity_type == 'VARIABLE':
            entity_rule = f"Use NOUNS only (e.g., '{ex_var}'). NO verbs."
        elif entity_type == 'BOOLEAN_VAR':
            entity_rule = f"Use BOOLEAN prefixes (e.g., {ex_bool})."
        else:
            entity_rule = f"Use ACTION VERBS (e.g., {ex_func})."

        leading_us_rule = ""
        if target_name.startswith('_'):
            leading_us_rule = "\n- PRESERVE PREFIX: The original name starts with '_'. ALL your suggestions MUST start with '_'."

        if n_parts <= 2:
            max_allowed_parts = n_parts + 1
            strategy_instruction = f"""[Strategy: Short & Concise]
- MAX WORDS: {max_allowed_parts} words per name.
- EXAMPLES: {ex_short}
- Use common C/C++ abbreviations (ptr, buf, mem, val)."""
        else:
            strategy_instruction = """[Strategy: Semantic Refactoring]
- Provide professional synonyms matching the exact system logic.
- Keep the length similar to the original name."""

        return f"""You are an expert C/C++ developer. Suggest exactly {top_n} alternative names for `{target_name}`.

[Context Code]
{context_code}
{strategy_instruction}

[Strict Rules]
{entity_rule}
STYLE: Use {style} naming convention.{leading_us_rule}
NO generic names ("new_var", "temp").

[Task]
Output ONLY a JSON array containing EXACTLY {top_n} strings. Do not explain.
Example format for {top_n} items: ["name1", "name2", "name3", ...]

JSON
["""

    def generate_candidates(self, vulnerable_tasks: List[Dict[str, Any]], target_quota: int = 20) -> Dict[str, List[str]]:
        # Generates deep semantic naming candidates for target entities using LLM and vector constraints.
        results = {task["target_name"]: [] for task in vulnerable_tasks}

        llm_prompts = []
        task_metadata = {}

        for task_idx, task in enumerate(vulnerable_tasks):
            target_name = task["target_name"]

            slice_code_str = task["code_str"]
            slice_code_bytes = slice_code_str.encode("utf-8")

            slice_identifiers = self.analyzer.extract_identifiers(slice_code_bytes)
            if target_name not in slice_identifiers:
                continue

            best_occ_idx = self._find_best_context_occurrence(slice_code_bytes, slice_identifiers[target_name])
            target_info = slice_identifiers[target_name][best_occ_idx]

            raw_entity_type = target_info.get('entity_type', 'variable')
            entity_type = 'BOOLEAN_VAR' if target_name.startswith(('is_', 'has_', 'can_', 'should_')) else (
                'FUNCTION' if raw_entity_type == 'function' else 'VARIABLE')

            original_style = self._detect_naming_style(target_name)
            parts, style = self._split_identifier(target_name)

            prefix_bytes = slice_code_bytes[:target_info['start']]
            suffix_bytes = slice_code_bytes[target_info['end']:]
            local_prefix = prefix_bytes.decode("utf-8", errors="replace")
            local_suffix = suffix_bytes.decode("utf-8", errors="replace")

            MAX_CHAR_LIMIT = 2500
            prefix_str = local_prefix[-MAX_CHAR_LIMIT:] if len(local_prefix) > MAX_CHAR_LIMIT else local_prefix
            suffix_str = local_suffix[:MAX_CHAR_LIMIT] if len(local_suffix) > MAX_CHAR_LIMIT else local_suffix

            task_metadata[task_idx] = {
                "target_name": target_name, "parts": parts, "style": style, "n_parts": len(parts),
                "entity_type": entity_type, "original_style": original_style,

                "full_code_str": task["full_code_str"],
                "full_code_bytes": task["full_code_str"].encode("utf-8"),
                "full_identifiers": task.get("full_identifiers", slice_identifiers),

                "local_prefix": prefix_str, "local_suffix": suffix_str
            }

            prompt = self._build_llm_prompt(slice_code_str, target_name, original_style, target_quota * 2, entity_type,
                                            len(parts))
            llm_prompts.append(prompt)

        if not llm_prompts: return results

        try:
            llm_responses = self.llm_client.batch_chat(llm_prompts)
        except Exception as e:
            print(f"[!] LLM Batch Chat Failed: {e}")
            llm_responses = [""] * len(llm_prompts)

        for resp_idx, response in enumerate(llm_responses):
            meta = task_metadata[resp_idx]
            parsed_cands = []

            leading_m = re.match(r'^_+', meta["target_name"])
            leading_us = leading_m.group(0) if leading_m else ""

            if response and isinstance(response, str):
                clean_text = response.replace("```json", "").replace("```", "").strip()
                first_quote, last_quote = clean_text.find('"'), clean_text.rfind('"')

                if first_quote != -1 and last_quote != -1 and first_quote != last_quote:
                    patched_json = f"[{clean_text[first_quote:last_quote + 1]}]"
                    try:
                        parsed_cands = json.loads(patched_json)
                        if not isinstance(parsed_cands, list): parsed_cands = [str(parsed_cands)]
                    except Exception:
                        pass

                if not parsed_cands:
                    parsed_cands = re.findall(r'["\']([a-zA-Z0-9_]+)["\']', response)

            valid_cands, oversized_cands = [], []
            for c in parsed_cands:
                if isinstance(c, str) and c.strip():
                    clean_cand = c.strip()

                    if leading_us and not clean_cand.startswith(leading_us):
                        clean_cand = leading_us + clean_cand.lstrip('_')
                    elif not leading_us and clean_cand.startswith('_'):
                        clean_cand = clean_cand.lstrip('_')

                    if clean_cand in valid_cands or clean_cand in oversized_cands: continue

                    cand_parts_list, _ = self._split_identifier(clean_cand)
                    limit = meta["n_parts"] + 1 if meta["n_parts"] <= 2 else meta["n_parts"] + 2

                    if len(cand_parts_list) <= limit:
                        valid_cands.append(clean_cand)
                    else:
                        oversized_cands.append(clean_cand)

            min_threshold = int(target_quota * 0.8)
            if len(valid_cands) < min_threshold and oversized_cands:
                valid_cands.extend(oversized_cands[:min_threshold - len(valid_cands)])

            cg_cfg = self.config.get('candidate_generation', {})
            hw_cfg = cg_cfg.get('heavyweight', {})

            ctx = {
                'code_bytes': meta["full_code_bytes"],
                'full_code_str': meta["full_code_str"],
                'target_name': meta["target_name"],
                'identifiers': meta["full_identifiers"],
                'keywords': self.analyzer.keywords,
                'original_style': meta["original_style"],
                'local_prefix': meta["local_prefix"],
                'local_suffix': meta["local_suffix"],

                'semantic_threshold': hw_cfg.get('semantic_threshold', 0.85),
                'preserve_style': cg_cfg.get('preserve_style', True),

                'entity_type': meta["entity_type"],
                'return_type': next(
                    (u['return_type'] for u in meta["full_identifiers"].get(meta["target_name"], []) if
                     u.get('return_type')), None),
            }

            final_candidates = []
            self._verify_and_filter(valid_cands, target_quota, final_candidates, ctx)
            results[meta["target_name"]] = final_candidates

        return results