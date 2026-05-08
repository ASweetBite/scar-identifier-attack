import json
import math
import re
from typing import Dict, Any
from typing import List

import torch
import torch.nn.functional as F


class HeavyWeightCandidateGenerator:
    def __init__(self, embedder, llm_client, analyzer, config):
        """
        初始化重量级候选词生成器 (仅依赖 LLM 进行深度语义改写)
        """
        self.embedder = embedder
        self.llm_client = llm_client
        self.analyzer = analyzer
        self.config = config
        stats_path = config.get('naming_stats_path', 'naming_stats.json')

        from utils.scorer import StatisticalNamingScorer
        self.scorer = StatisticalNamingScorer(stats_path)

    @torch.no_grad()
    def _calculate_perplexity_batch(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        利用批处理 (Batching) 并行计算多个代码片段的 PPL，极大提升运算速度。
        """
        if not texts:
            return []

        tokenizer = getattr(self.llm_client, 'tokenizer', None)
        model = getattr(self.llm_client, 'model', None)

        if not tokenizer or not model:
            raise AttributeError("无法从 llm_client 提取 tokenizer 或 model。")

        device = model.device
        ppls = []

        # 根据显存大小可适当调大 batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,  # 必须开启 padding 以支持 Batching
                truncation=True,
                max_length=1024
            ).to(device)

            # 前向传播 (不传 labels，手动计算 per-sequence loss 以避免 padding 污染)
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

            # CausalLM 需要错位 (Shift) 计算预测下一个 Token 的 Loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            shift_mask = inputs["attention_mask"][..., 1:].contiguous()

            # 计算所有 Token 的 Loss (不聚合)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size(0), shift_labels.size(1))

            # 使用 mask 过滤掉 padding 产生的无效 Loss
            loss = loss * shift_mask

            # 计算每个序列的有效 Token 平均 Loss
            seq_lens = shift_mask.sum(dim=1)
            # 防止除以 0 导致 NaN
            seq_lens = torch.clamp(seq_lens, min=1.0)
            seq_loss = loss.sum(dim=1) / seq_lens

            for val in seq_loss:
                try:
                    ppls.append(math.exp(val.item()))
                except OverflowError:
                    ppls.append(float('inf'))

            # 释放缓存
            del inputs, outputs, loss, shift_logits, shift_labels
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return ppls

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

    # =========================================================================
    # AST 上下文与向量相似度验证 (原样保留)
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

    def _get_variable_token_embeddings(self, prefixes: List[str], var_names: List[str], suffixes: List[str],
                                       batch_size: int = 64) -> torch.Tensor:
        """精准提取Token级变量语义向量（仅用于计算 LLM 候选词的相似度）"""
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
        target_parts, _ = self._split_identifier(target_name)
        cand_parts, _ = self._split_identifier(cand)
        if len(target_parts) > 2 and len(cand_parts) > 0:
            identical_count = sum(1 for p1, p2 in zip(target_parts, cand_parts) if p1.lower() == p2.lower())
            change_ratio = 1.0 - (identical_count / max(len(target_parts), len(cand_parts)))
            return change_ratio <= 0.33
        return False

    def _verify_and_filter(self, candidate_list, quota, final_candidates, ctx):
        base_threshold = ctx.get('semantic_threshold', 0.85)
        entity_type = ctx.get('entity_type', 'VARIABLE')

        # === 新增/修改：提取 PPL 开关 ===
        is_ppl_filter = ctx.get('is_ppl_filter', True)

        ppl_max_ratio = ctx.get('ppl_max_ratio', 1.2)
        ppl_max_abs = ctx.get('ppl_max_abs', 50.0)

        # 1. 预过滤：基础检查
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

        orig_context = ctx.get('full_code_str', ctx['local_prefix'] + ctx['target_name'] + ctx['local_suffix'])
        orig_ppl = None

        added = 0
        CHUNK_SIZE = max(50, quota * 2)
        target_name = ctx['target_name']
        target_parts, _ = self._split_identifier(target_name)
        return_type = ctx.get('return_type', None)

        # 全局评估池，用于存放所有走过 PPL 计算的候选词以备抢救
        all_evaluated_cands = []

        # 核心修改：以 Chunk 为单位循环
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

            # 2. 相似度打分
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
                        semantically_valid.append((cand, final_score, sim.item(), bonus))
                    else:
                        # print(f"        🚫 [Filter | Semantic] '{cand}' (Score: {final_score:.3f} < {base_threshold})")
                        pass
            else:
                semantically_valid = [(cand, 1.0, 1.0, 0.0) for cand in filtered_chunk]

            # 3. 收集该块内所有合法的上下文用于 LLM 批处理
            pending_ppl_eval = []

            for cand_data in semantically_valid:
                cand, final_score, sim_val, bonus_val = cand_data
                valid_cand = self._verify_ast_single(cand, ctx)

                if valid_cand and valid_cand not in final_candidates:
                    if 'full_code_str' in ctx and 'code_bytes' in ctx:
                        mod_bytes = bytearray(ctx['code_bytes'])
                        cand_bytes = valid_cand.encode('utf-8')
                        occurrences = sorted(ctx['identifiers'][ctx['target_name']], key=lambda x: x['start'],
                                             reverse=True)
                        for occ in occurrences:
                            mod_bytes[occ['start']:occ['end']] = cand_bytes
                        adv_context = mod_bytes.decode('utf-8', errors='replace')
                    else:
                        adv_context = ctx['local_prefix'] + valid_cand + ctx['local_suffix']

                    pending_ppl_eval.append({
                        "cand": valid_cand, "score": final_score,
                        "sim": sim_val, "bonus": bonus_val, "context": adv_context
                    })
                else:
                    if not valid_cand:
                        # print(f"        🚫 [Filter | Final AST] '{cand}' (Failed context insertion verify)")
                        pass

            if not pending_ppl_eval:
                continue

            # 4. LLM 批量计算 PPL 并执行过滤拦截
            adv_ppls = []
            # === 修改：如果开启了 PPL 过滤，才进行批量推理 ===
            if is_ppl_filter and hasattr(self, 'llm_client') and self.llm_client:
                if orig_ppl is None:
                    orig_ppl = self._calculate_perplexity_batch([orig_context], batch_size=1)[0]

                adv_contexts_batch = [item["context"] for item in pending_ppl_eval]

                # 动态读取 batch_size，兜底为 4
                ppl_batch_size = self.config.get('ppl_batch_size', 4) if hasattr(self, 'config') else 4
                adv_ppls = self._calculate_perplexity_batch(adv_contexts_batch, batch_size=ppl_batch_size)

            for idx, eval_data in enumerate(pending_ppl_eval):
                if added >= quota: break

                valid_cand = eval_data["cand"]

                # === 修改：关闭 PPL 时，直接放行 ===
                if not is_ppl_filter or not hasattr(self, 'llm_client') or not self.llm_client:
                    final_candidates.append(valid_cand)
                    added += 1
                    # print(
                    #     f"        ✅ [Passed | {added}/{quota}] '{valid_cand}' (Score: {eval_data['score']:.3f}) | [PPL Check Disabled]")
                    continue

                adv_ppl = adv_ppls[idx]
                ppl_ratio = (adv_ppl / orig_ppl) if orig_ppl > 0 else float('inf')
                ppl_diff = adv_ppl - orig_ppl

                # 将评估数据存入全局池备用
                all_evaluated_cands.append({
                    "cand": valid_cand, "score": eval_data["score"],
                    "adv_ppl": adv_ppl, "ppl_ratio": ppl_ratio, "ppl_diff": ppl_diff
                })

                # === 新增：PPL 统计探针写入 ===
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
                    #     f"        ✅ [Passed | {added}/{quota}] '{valid_cand}' (Score: {eval_data['score']:.3f}) | [PPL: {orig_ppl:.1f} -> {adv_ppl:.1f} (Ratio: {ppl_ratio:.2f}x, Diff: {ppl_diff:+.1f})]")
                else:
                    # print(
                        # f"        🚫 [Filter | PPL Overload] '{valid_cand}' | [PPL: {orig_ppl:.1f} -> {adv_ppl:.1f} (Ratio: {ppl_ratio:.2f}x, Diff: {ppl_diff:+.1f})]")
                    pass

        if is_ppl_filter and added == 0 and all_evaluated_cands:
            # print(f"        ⚠️ [Fallback] 变量 '{target_name}' 的 PPL 过滤全军覆没。启动综合打分抢救机制。")

            for item in all_evaluated_cands:
                item['comp_score'] = item['score'] / max(1.0, item['ppl_ratio'])

            all_evaluated_cands.sort(key=lambda x: x['comp_score'], reverse=True)

            rescue_quota = min(3, len(all_evaluated_cands))
            for i in range(rescue_quota):
                item = all_evaluated_cands[i]
                valid_cand = item['cand']
                final_candidates.append(valid_cand)
                added += 1

                # print(
                #     f"        🚑 [Rescued | {added}/{quota}] '{valid_cand}' (Sim: {item['score']:.3f}, PPL Ratio: {item['ppl_ratio']:.2f}x) -> Comp Score: {item['comp_score']:.3f}")

        return added


    def _build_llm_prompt(self, context_code: str, target_name: str, style: str, top_n: int, entity_type: str, n_parts: int) -> str:
        # 1. 根据传入的 style 动态生成匹配的正面示例
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
        else: # 默认 snake_case
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

        # === 新增：检测并生成前导下划线强制约束 ===
        # 即使是 1.5B 小模型，直接陈述正向命令（MUST PRESERVE）也能取得较好效果
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

        # 最后的 JSON\n[ 用于强制小模型直接输出数组，不要有任何寒暄
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

    def generate_candidates(self, vulnerable_tasks: List[Dict[str, Any]], target_quota: int = 20,
                            is_ppl_filter: bool = False) -> Dict[str, List[str]]:
        """
        专门为 RNNS 选出的薄弱点调用 LLM 深度生成。
        """
        results = {task["target_name"]: [] for task in vulnerable_tasks}

        llm_prompts = []
        task_metadata = {}

        for task_idx, task in enumerate(vulnerable_tasks):
            target_name = task["target_name"]

            # ==========================================
            # A. 生成阶段：严格使用切片数据 (Code Slice)
            # ==========================================
            slice_code_str = task["code_str"]
            slice_code_bytes = slice_code_str.encode("utf-8")

            # 【核心修正】：仅解析切片代码的 AST，获取切片内的相对坐标
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

            # 【核心修正】：废弃二次解析，直接截取字节获取前后缀 (如果 LLM 提示词需要依赖 local_prefix/suffix 的话)
            prefix_bytes = slice_code_bytes[:target_info['start']]
            suffix_bytes = slice_code_bytes[target_info['end']:]
            local_prefix = prefix_bytes.decode("utf-8", errors="replace")
            local_suffix = suffix_bytes.decode("utf-8", errors="replace")

            # 长度截断保护
            MAX_CHAR_LIMIT = 2500
            prefix_str = local_prefix[-MAX_CHAR_LIMIT:] if len(local_prefix) > MAX_CHAR_LIMIT else local_prefix
            suffix_str = local_suffix[:MAX_CHAR_LIMIT] if len(local_suffix) > MAX_CHAR_LIMIT else local_suffix

            task_metadata[task_idx] = {
                "target_name": target_name, "parts": parts, "style": style, "n_parts": len(parts),
                "entity_type": entity_type, "original_style": original_style,

                # ==========================================
                # B. 验证阶段 (PPL)：保存透传的全量数据
                # ==========================================
                "full_code_str": task["full_code_str"],
                "full_code_bytes": task["full_code_str"].encode("utf-8"),
                "full_identifiers": task.get("full_identifiers", slice_identifiers),

                "local_prefix": prefix_str, "local_suffix": suffix_str
            }

            # 【关键修改】：构建 LLM Prompt 时，严格传递 slice_code_str！
            # 这样 LLM 看到的只有折叠精简后的代码，大幅降低 Token 消耗并提高聚焦度
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

                    # 强制后处理校准，确保 100% 对齐前导下划线
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

            # ==========================================
            # 【核心修正】：交给 PPL 进行验证时，组装全量级 Context
            # ==========================================
            ctx = {
                'code_bytes': meta["full_code_bytes"],  # 用于 PPL 字节绝对替换
                'full_code_str': meta["full_code_str"],  # 用于作为 PPL 原始对比
                'target_name': meta["target_name"],
                'identifiers': meta["full_identifiers"],  # PPL 所需的全局坐标字典
                'keywords': self.analyzer.keywords,
                'original_style': meta["original_style"],
                'local_prefix': meta["local_prefix"],  # PPL 回退方案使用的局部语句
                'local_suffix': meta["local_suffix"],
                'semantic_threshold': self.config.get('semantic_threshold', 0.85) if hasattr(self, 'config') else 0.85,
                'preserve_style': self.config.get('preserve_style', True) if hasattr(self, 'config') else True,
                'entity_type': meta["entity_type"],
                'return_type': next(
                    (u['return_type'] for u in meta["full_identifiers"].get(meta["target_name"], []) if
                     u.get('return_type')), None),
                'is_ppl_filter': is_ppl_filter
            }

            final_candidates = []
            self._verify_and_filter(valid_cands, target_quota, final_candidates, ctx)
            results[meta["target_name"]] = final_candidates

        return results