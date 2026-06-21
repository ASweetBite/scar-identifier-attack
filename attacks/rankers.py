import gc
import random
import torch


class RNNS_Ranker:
    def __init__(self, model_zoo, target_model: str, rename_fn):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn

    def rank_variables(self, code, variables, subs_pool, reference_label,
                       test_sample_size=10, top_k=10, filter_short_vars=True,
                       guaranteed_head_size=2, history_cache=None):  # 🚀 [修复] 添加了 history_cache 参数

        # 初始化 Cache 兜底
        if history_cache is None:
            history_cache = {}

        oref_idx = 0 if reference_label == -1 else reference_label
        orig_prob = self.model_zoo.predict_label_conf(code, oref_idx, self.target_model)

        valid_vars = [v for v in variables if len(v) > 2] if filter_short_vars else variables

        # 提前初始化记录器，方便后续复用 Cache
        var_max_drop = {var: -float('inf') for var in valid_vars}
        var_best_cand = {var: var for var in valid_vars}

        mutation_tasks = []

        # 串行执行重命名，彻底规避 AST Parser 多线程崩溃风险
        for var in valid_vars:
            all_cands = subs_pool.get(var, [])
            if not all_cands:
                continue

            if len(all_cands) <= test_sample_size:
                candidates = all_cands
            else:
                # =========================================================
                # 🚀 [优化 1] 动态分层采样：确保头部(LLM)和尾部(MLM)获得均等的探查机会
                # =========================================================
                actual_head_size = min(len(all_cands), test_sample_size // 2)

                head_cands = all_cands[:actual_head_size]
                tail_pool = all_cands[actual_head_size:]
                sample_count = test_sample_size - len(head_cands)

                if len(tail_pool) >= sample_count:
                    tail_cands = random.sample(tail_pool, sample_count)
                else:
                    tail_cands = tail_pool

                candidates = head_cands + tail_cands

            for cand in candidates:
                if cand != var:
                    # =========================================================
                    # 🚀 [优化 2] 惰性求值：如果在本次样本中已经测过这个组合，直接复用分数！
                    # =========================================================
                    cache_key = (var, cand)
                    if cache_key in history_cache:
                        prob_drop = history_cache[cache_key]
                        if prob_drop > var_max_drop[var]:
                            var_max_drop[var] = prob_drop
                            var_best_cand[var] = cand
                    else:
                        # 只有没测过的全新组合，才去走真实的 AST 重命名
                        try:
                            renamed_code = self.rename_fn(code, {var: cand})
                            if renamed_code:
                                mutation_tasks.append((var, cand, renamed_code))
                        except Exception:
                            continue

        # =========================================================
        # 🚀 [优化 3] 极速短路：如果全命中 Cache（重排序阶段常见），直接返回结果，不再动用模型！
        # =========================================================
        if not mutation_tasks:
            valid_scores = [(var, score) for var, score in var_max_drop.items() if score != -float('inf')]
            sorted_all_vars_with_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)
            ranked_vars = [var for var, _ in sorted_all_vars_with_scores]
            score_dict = {var: score for var, score in sorted_all_vars_with_scores}
            best_seeds = {var: var_best_cand[var] for var in ranked_vars if var_best_cand[var] != var}
            return ranked_vars, score_dict, best_seeds

        codes_to_predict = [task[2] for task in mutation_tasks]
        all_probs = []
        BATCH_SIZE = 16

        # 分块推理，彻底杜绝 OOM
        for i in range(0, len(codes_to_predict), BATCH_SIZE):
            chunk = codes_to_predict[i:i + BATCH_SIZE]
            chunk_probs, _ = self.model_zoo.batch_predict(chunk, self.target_model)
            all_probs.extend(chunk_probs)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # =========================================================
        # 🚀 [优化 4] 将新计算的结果写入历史记录 Cache，并更新当前最高分
        # =========================================================
        for (var, cand, _), probs in zip(mutation_tasks, all_probs):
            prob_drop = orig_prob - probs[oref_idx]
            history_cache[(var, cand)] = prob_drop  # 写入全局 Cache

            if prob_drop > var_max_drop[var]:
                var_max_drop[var] = prob_drop
                var_best_cand[var] = cand

        valid_scores = [(var, score) for var, score in var_max_drop.items() if score != -float('inf')]
        sorted_all_vars_with_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)

        ranked_vars = [var for var, _ in sorted_all_vars_with_scores]
        score_dict = {var: score for var, score in sorted_all_vars_with_scores}

        best_seeds = {var: var_best_cand[var] for var in ranked_vars if var_best_cand[var] != var}

        return ranked_vars, score_dict, best_seeds