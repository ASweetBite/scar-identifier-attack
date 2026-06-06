import itertools
import math
import re
from typing import Dict, Any
from typing import List

import torch
import torch.nn.functional as F


class LightweightCandidateGenerator:
    def __init__(self, mlm_engine, analyzer, config, llm_client=None):
        """
        初始化轻量级生成器（依赖 MLM 进行掩码预测，同时引入 LLM 进行 PPL 自然度评估）
        :param mlm_engine: 用于掩码预测、提取 Token Embedding 和计算相似度的模型
        :param analyzer: 语法树与上下文分析器
        :param config: 全局配置字典
        :param llm_client: [新增] 传入 Qwen LLM 客户端，复用其计算 PPL (困惑度)
        """
        self.mlm_engine = mlm_engine
        self.analyzer = analyzer
        self.config = config
        self.llm_client = llm_client  # 保存 LLM 句柄用于 PPL 计算
        cg_cfg = self.config.get('candidate_generation', {})
        stats_path = cg_cfg.get('naming_stats_path', 'naming_stats.json')
        from utils.scorer import StatisticalNamingScorer
        self.scorer = StatisticalNamingScorer(stats_path)

    @torch.no_grad()
    def _calculate_perplexity_batch(self, texts: List[str], batch_size: int = 4) -> List[float]:
        """批量计算 PPL，避免闲置 GPU 算力"""
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
        target_lower = target_name.lower()
        cand_lower = cand.lower()

        # 1. 纯前缀/后缀附加
        if cand_lower.endswith(f"_{target_lower}") or cand_lower.startswith(f"{target_lower}_"):
            return min(0.99, base_threshold + 0.05)

        # 2. 词根完全包含但没有分隔符
        if target_lower in cand_lower:
            return min(0.99, base_threshold + 0.03)

        # 3. 极小编辑距离
        import Levenshtein
        if Levenshtein.distance(target_lower, cand_lower) <= 2:
            return min(0.99, base_threshold + 0.07)

        # ==========================================================
        # 4. [终极版] 多词根局部微调：引入基于长度的重叠率惩罚
        # ==========================================================
        target_parts, target_sep = self._split_identifier(target_name)
        cand_parts, cand_sep = self._split_identifier(cand)

        if len(target_parts) > 1 and len(target_parts) == len(cand_parts) and target_sep == cand_sep:
            identical_count = sum(1 for t, c in zip(target_parts, cand_parts) if t.lower() == c.lower())

            if identical_count > 0:
                # 计算重叠率 (例如 4/5 = 0.8)
                overlap_ratio = identical_count / len(target_parts)

                # 只有当重叠率超过一半时，才触发高度惩罚
                if overlap_ratio >= 0.5:
                    # 惩罚公式：基础惩罚 + (重叠率的缩放)
                    # 重叠率越高，惩罚越重
                    base_penalty = 0.02
                    ratio_penalty = overlap_ratio * 0.06  # 最高可贡献 0.06 的惩罚
                    penalty = base_penalty + ratio_penalty

                    # 尾部词根替换额外惩罚（核心名词替换）
                    if target_parts[-1].lower() != cand_parts[-1].lower():
                        penalty += 0.015

                    return min(0.99, base_threshold + penalty)

        # 5. 完全替换 / 无明显结构对齐
        return base_threshold

    def _get_mutation_pattern(self, target_name: str, cand_name: str) -> str:
        """
        提取变异模式。
        对于结构完全改变、单字母替换、或无重叠部分的候选词，返回 '*'，这些词永远不被信任。
        只有保留了部分结构的局部替换（如 'sys_ue' -> 'mem_ue' = '*_ue'）才返回特定模式。
        """
        t_parts, t_sep = self._split_identifier(target_name)
        c_parts, c_sep = self._split_identifier(cand_name)

        # 仅当：1. 词根数量相同 2. 拼接符相同 3. 原词不仅包含一个词根
        if len(t_parts) == len(c_parts) and t_sep == c_sep and len(t_parts) > 1:
            pattern = []
            identical_count = 0
            for t, c in zip(t_parts, c_parts):
                if t.lower() == c.lower():
                    pattern.append(t.lower())
                    identical_count += 1
                else:
                    pattern.append('*')

            # 必须至少保留了一个原始词根，不能是全盘替换（如 'old_val' -> 'new_var' = '*_*'）
            if identical_count > 0 and identical_count < len(t_parts):
                sep_display = '_' if t_sep == '_' else ''
                return sep_display.join(pattern)

        # 单词变量、长度改变、结构完全改变，统一归为不可信模式
        return '*'
    # =========================================================================
    # 基础工具与命名规范检测 (原样保留)
    # =========================================================================
    def _detect_naming_style(self, name: str) -> str:
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
        cand_style = self._detect_naming_style(candidate)
        if original_style in ('snake_case', 'camelCase', 'PascalCase') and cand_style == 'single_lower': return True
        if original_style == 'single_lower' and cand_style in ('snake_case', 'camelCase'): return True
        if original_style == 'single_upper' and cand_style == 'SCREAMING_SNAKE': return True
        return cand_style == original_style

    def _split_identifier(self, name: str):
        if '_' in name:
            return name.split('_'), '_'
        else:
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', name)
            if not parts or (len(parts) == 1 and parts[0] == name): return [name], ''
            return parts, 'camel'

    def _build_masked_string(self, parts: List[str], start: int, end: int, num_masks: int, style: str, mask_token: str,
                             target_name: str) -> str:
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

    # =========================================================================
    # 上下文提取与模型推理交互
    # =========================================================================
    def _extract_local_context_ast(self, code_bytes: bytes, target_start: int, target_end: int) -> tuple[str, str]:
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
        target_parts, _ = self._split_identifier(target_name)
        cand_parts, _ = self._split_identifier(cand)
        if len(target_parts) > 2 and len(cand_parts) > 0:
            identical_count = sum(1 for p1, p2 in zip(target_parts, cand_parts) if p1.lower() == p2.lower())
            change_ratio = 1.0 - (identical_count / max(len(target_parts), len(cand_parts)))
            return change_ratio <= 0.33
        return False

    def _verify_ast_single(self, cand: str, ctx: dict) -> str | None:
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
        base_threshold = ctx.get('semantic_threshold', 0.85)
        entity_type = ctx.get('entity_type', 'VARIABLE')

        # === 新增/修改：提取 PPL 开关 ===
        is_ppl_filter = ctx.get('is_ppl_filter', True)

        ppl_max_ratio = ctx.get('ppl_max_ratio', 1.2)
        ppl_max_abs = ctx.get('ppl_max_abs', 50.0)

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

        orig_context = ctx.get('full_code_str', ctx['local_prefix'] + ctx['target_name'] + ctx['local_suffix'])
        orig_ppl = None

        pattern_trust_cache = {}
        TRUST_TEST_REQ = 3
        TRUST_DIFF_MARGIN = 2.0

        added = 0
        CHUNK_SIZE = max(50, quota * 2)
        target_name = ctx['target_name']
        target_parts, _ = self._split_identifier(target_name)
        return_type = ctx.get('return_type', None)

        all_evaluated_cands = []

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

            eval_queue = []
            for cand, final_score in semantically_valid:
                valid_cand = self._verify_ast_single(cand, ctx)
                if valid_cand and valid_cand not in final_candidates:
                    pattern_key = self._get_mutation_pattern(target_name, valid_cand)
                    if pattern_key != '*':
                        pattern_trust_cache.setdefault(pattern_key,
                                                       {'tests': 0, 'passes': 0, 'max_diff': 0.0, 'trusted': False})

                    if 'full_code_str' in ctx and 'code_bytes' in ctx:
                        mod_bytes = bytearray(ctx['code_bytes'])
                        cand_bytes = valid_cand.encode('utf-8')
                        occurrences = sorted(ctx['identifiers'][ctx['target_name']], key=lambda x: x['start'],
                                             reverse=True)
                        for occ in occurrences: mod_bytes[occ['start']:occ['end']] = cand_bytes
                        adv_context = mod_bytes.decode('utf-8', errors='replace')
                    else:
                        adv_context = ctx['local_prefix'] + valid_cand + ctx['local_suffix']

                    eval_queue.append(
                        {'cand': valid_cand, 'score': final_score, 'pattern': pattern_key, 'context': adv_context})

            # === LLM 批量计算 (加入 PPL 开关限制) ===
            batch_contexts = []
            for item in eval_queue:
                pat = item['pattern']
                if pat == '*' or not pattern_trust_cache[pat]['trusted']:
                    batch_contexts.append(item['context'])

            # === 修改：如果关闭了 is_ppl_filter，直接跳过 LLM 推理计算 ===
            if is_ppl_filter and batch_contexts and hasattr(self, 'llm_client') and self.llm_client:
                cg_cfg = self.config.get('candidate_generation', {})
                ppl_batch_size = cg_cfg.get('ppl_batch_size', 4)
                if orig_ppl is None:
                    orig_ppl = self._calculate_perplexity_batch([orig_context], batch_size=1)[0]
                adv_ppls = self._calculate_perplexity_batch(batch_contexts, batch_size=ppl_batch_size)

                ppl_idx = 0
                for item in eval_queue:
                    pat = item['pattern']
                    if pat == '*' or not pattern_trust_cache[pat]['trusted']:
                        item['adv_ppl'] = adv_ppls[ppl_idx]
                        ppl_idx += 1

            # === 结果结算与缓存动态更新 ===
            for item in eval_queue:
                if added >= quota: break

                valid_cand, final_score, pattern_key = item['cand'], item['score'], item['pattern']

                # === 修改：关闭 PPL 时，不执行后面的逻辑，全部直接放行 ===
                if not is_ppl_filter or not hasattr(self, 'llm_client') or not self.llm_client:
                    final_candidates.append(valid_cand)
                    added += 1
                    # print(
                    #     f"        ✅ [Passed | {added}/{quota}] '{valid_cand}' (Score: {final_score:.3f}) | [PPL Check Disabled]")
                    continue

                is_trusted_now = False
                if pattern_key != '*':
                    is_trusted_now = pattern_trust_cache[pattern_key]['trusted']

                if is_trusted_now:
                    final_candidates.append(valid_cand)
                    added += 1
                    # print(
                    #     f"        ✅ [Passed | {added}/{quota}] '{valid_cand}' (Score: {final_score:.3f}) | [PPL: Cached Auto-Pass (Pattern '{pattern_key}')]")
                    continue

                adv_ppl = item['adv_ppl']
                ppl_diff = adv_ppl - orig_ppl
                ppl_ratio = (adv_ppl / orig_ppl) if orig_ppl > 0 else float('inf')

                # 全局评估池，用于存放所有走过 PPL 计算的候选词以备抢救
                all_evaluated_cands.append({
                    "cand": valid_cand, "score": final_score, "adv_ppl": adv_ppl,
                    "ppl_ratio": ppl_ratio, "ppl_diff": ppl_diff, "pattern": pattern_key
                })

                # [由于加入了 is_ppl_filter 拦截，以下逻辑只有在开启 PPL 时才会走到]
                if orig_ppl > 0 and adv_ppl != float('inf') and hasattr(self, 'ppl_stats'):
                    self.ppl_stats["total_evaluated"] += 1
                    self.ppl_stats["diff_distribution"].append(ppl_diff)
                    self.ppl_stats["ratio_distribution"].append(ppl_ratio)

                passed_ppl_check = True
                is_abs_over = adv_ppl > ppl_max_abs
                is_ratio_over = ppl_ratio > ppl_max_ratio

                if is_abs_over or is_ratio_over:
                    passed_ppl_check = False
                    if hasattr(self, 'ppl_stats'):
                        if is_abs_over and is_ratio_over:
                            self.ppl_stats["rejected_both"] += 1
                        elif is_abs_over:
                            self.ppl_stats["rejected_by_abs"] += 1
                        elif is_ratio_over:
                            self.ppl_stats["rejected_by_ratio"] += 1

                if passed_ppl_check:
                    final_candidates.append(valid_cand)
                    added += 1
                    # print(
                    #     f"        ✅ [Passed | {added}/{quota}] '{valid_cand}' (Score: {final_score:.3f}) | [PPL: {orig_ppl:.1f} -> {adv_ppl:.1f} (Ratio: {ppl_ratio:.2f}x, Diff: {ppl_diff:+.1f})]")

                    if pattern_key != '*':
                        p_info = pattern_trust_cache[pattern_key]
                        if not p_info['trusted']:
                            p_info['tests'] += 1
                            p_info['passes'] += 1
                            p_info['max_diff'] = max(p_info['max_diff'], ppl_diff)
                            if p_info['tests'] >= TRUST_TEST_REQ and p_info['passes'] == p_info['tests'] and p_info[
                                'max_diff'] <= TRUST_DIFF_MARGIN:
                                p_info['trusted'] = True
                                # print(
                                #     f"        💡 [Heuristic] Pattern '{pattern_key}' is now trusted (Max Diff: {p_info['max_diff']:+.1f}). Subsequent candidates will skip PPL check.")
                else:
                    if pattern_key != '*':
                        pattern_trust_cache[pattern_key]['tests'] += 1
                    print(
                        f"        🚫 [Filter | PPL Overload] '{valid_cand}' | [PPL: {orig_ppl:.1f} -> {adv_ppl:.1f} (Ratio: {ppl_ratio:.2f}x, Diff: {ppl_diff:+.1f})]")

        if is_ppl_filter and added == 0 and all_evaluated_cands:
            print(f"        ⚠️ [Fallback] 变量 '{target_name}' 的 PPL 过滤全军覆没。启动综合打分抢救机制。")
            for item in all_evaluated_cands:
                item['comp_score'] = item['score'] / max(1.0, item['ppl_ratio'])
            all_evaluated_cands.sort(key=lambda x: x['comp_score'], reverse=True)

            rescue_quota = min(3, len(all_evaluated_cands))
            for i in range(rescue_quota):
                item = all_evaluated_cands[i]
                valid_cand = item['cand']
                final_candidates.append(valid_cand)
                added += 1
                pat = item['pattern']
                if pat != '*' and pat in pattern_trust_cache:
                    pattern_trust_cache[pat]['tests'] += 1
                print(
                    f"        🚑 [Rescued | {added}/{quota}] '{valid_cand}' (Sim: {item['score']:.3f}, PPL Ratio: {item['ppl_ratio']:.2f}x) -> Comp Score: {item['comp_score']:.3f}")

        return added

    def generate_candidates(self, batch_tasks: List[Dict[str, Any]], top_k_mlm: int = 40, top_n_keep: int = 20,
                            is_ppl_filter: bool = False) -> Dict[
        str, List[str]]:
        """
        批量快速生成 MLM 候选词。
        """
        results = {task["target_name"]: [] for task in batch_tasks}
        mlm_tracking = []
        task_metadata = {}
        mask_token = self.mlm_engine.tokenizer.mask_token

        # 1. 任务解析与 MLM 变体构建
        for task_idx, task in enumerate(batch_tasks):
            target_name = task["target_name"]

            # ==========================================
            # A. 生成阶段：严格使用切片数据 (Code Slice)
            # ==========================================
            slice_code_str = task["code_str"]
            slice_code_bytes = slice_code_str.encode("utf-8")

            # 【核心修正】：仅解析切片代码的 AST，获取相对坐标
            slice_identifiers = self.analyzer.extract_identifiers(slice_code_bytes)

            # 如果切片后变量意外丢失（通常不会发生，容错防崩），跳过
            if target_name not in slice_identifiers:
                continue

            # 使用切片内的标识符获取最佳位置与坐标
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

                # 验证阶段 (PPL)：保存透传的全量数据
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

        # 3. 解析 MLM 输出
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

        # 4. 执行复杂度过滤系统
        for t_idx, meta in task_metadata.items():
            unique_mlm_cands = list(dict.fromkeys(meta["raw_mlm_cands"]))

            # 提取新版层级配置
            cg_cfg = self.config.get('candidate_generation', {})
            lw_cfg = cg_cfg.get('lightweight', {})

            # 使用配置覆盖函数默认参数
            actual_is_ppl_filter = cg_cfg.get('is_ppl_filter', is_ppl_filter)

            ctx = {
                'code_bytes': meta["full_code_bytes"],
                'full_code_str': meta["full_code_str"],
                'target_name': meta["target_name"],
                'identifiers': meta["full_identifiers"],
                'keywords': self.analyzer.keywords,
                'original_style': meta["original_style"],
                'local_prefix': meta["local_prefix"],
                'local_suffix': meta["local_suffix"],

                # 读取专属和共享配置
                'semantic_threshold': lw_cfg.get('semantic_threshold', 0.85),
                'preserve_style': cg_cfg.get('preserve_style', True),
                'is_ppl_filter': actual_is_ppl_filter,
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