import math
import random

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


class GeneticAlgorithmOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        ga_cfg = config.get('genetic_algorithm', {}) if config else {}
        run_cfg = config.get('run_params', {}) if config else {}

        self.pop_size = ga_cfg.get('pop_size', 40)
        self.max_generations = run_cfg.get('iterations', 60)
        self.run_mode = run_cfg.get('run_mode', 'attack')

        self.stagnation_limit = ga_cfg.get('stagnation_threshold', 5)
        self.m_rate_min = ga_cfg.get('mutation_rate_min', 0.1)
        self.m_rate_max = ga_cfg.get('mutation_rate_max', 0.5)

    def _calculate_fitness(self, probs, original_pred):
        if self.mode != "binary":
            orig_idx = 0 if original_pred == -1 else original_pred
            orig_idx = min(orig_idx, len(probs) - 1 if isinstance(probs, (list, np.ndarray)) else 0)
            orig_prob = max(probs[orig_idx] if isinstance(probs, (list, np.ndarray)) else probs, 1e-9)
            return -orig_prob

        is_orig_vuln = (original_pred == 1)
        p_safe = float(probs[0])
        p_vuln = float(probs[1]) if len(probs) > 1 else 1.0 - p_safe

        orig_prob = max(p_vuln if is_orig_vuln else p_safe, 1e-9)
        target_prob = max(p_safe if is_orig_vuln else p_vuln, 1e-9)
        return math.log(target_prob) - math.log(orig_prob)

    def _get_target_prob(self, probs, original_pred):
        if self.mode != "binary":
            orig_idx = 0 if original_pred == -1 else original_pred
            orig_idx = min(orig_idx, len(probs) - 1 if isinstance(probs, (list, np.ndarray)) else 0)
            orig_prob = probs[orig_idx] if isinstance(probs, (list, np.ndarray)) else float(probs)
            return 1.0 - orig_prob

        is_orig_vuln = (original_pred == 1)
        p_safe = float(probs[0])
        p_vuln = float(probs[1]) if len(probs) > 1 else 1.0 - p_safe
        return p_safe if is_orig_vuln else p_vuln

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None, rnns_best_seed=None, all_vars=None):
        """Executes a genetic algorithm to find the optimal adversarial variable substitutions."""

        # 1. 确定基因全集与边缘基因
        if all_vars is None:
            all_vars = target_vars

        background_vars = [v for v in all_vars if v not in target_vars]

        # 2. 构建非对称变异概率表 (Asymmetric Mutation Probabilities)
        mutation_probs = {}
        if variable_scores and target_vars:
            scores = [variable_scores.get(v, 0) for v in target_vars]
            min_s, max_s = min(scores), max(scores)
            for var in target_vars:
                score = variable_scores.get(var, 0)
                if max_s > min_s:
                    mutation_probs[var] = self.m_rate_min + (self.m_rate_max - self.m_rate_min) * (
                            (score - min_s) / (max_s - min_s))
                else:
                    mutation_probs[var] = (self.m_rate_min + self.m_rate_max) / 2
        else:
            for v in target_vars: mutation_probs[v] = 0.3

        # 🌟 为边缘基因赋予极低的探索性变异概率（例如 3%）
        # 这使得 GA 偶尔能摸奖，但大部分算力依然集中在 target_vars 上
        for var in background_vars:
            mutation_probs[var] = 0.03

        def get_safe_choice(var, pool, current_val=None):
            choices = list(set(pool)) if pool else []
            if not choices:
                return var
            if current_val and len(choices) > 1 and current_val in choices:
                choices.remove(current_val)
            return random.choice(choices)

        fitness_cache = {}
        best_code, best_fitness, best_probs, best_pred = code, float('-inf'), None, original_pred
        stagnation_counter = 0

        # --- 初始化种群 (此时染色体长度为 len(all_vars)) ---
        population = [{var: var for var in all_vars}]  # 1. 保留完全不突变的原始基因

        # 2. 注入 RNNS 精英种子
        if rnns_best_seed:
            seed_ind = {var: rnns_best_seed.get(var, var) for var in all_vars}
            population.append(seed_ind)

        # 3. 填满剩余种群
        while len(population) < self.pop_size:
            ind = {}
            for v in all_vars:
                # 初始种群生成时，靶点高频突变，边缘基因低频突变
                if v in target_vars and random.random() < 0.8:
                    ind[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v])
                elif v in background_vars and random.random() < 0.1:
                    ind[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v])
                else:
                    ind[v] = v
            population.append(ind)

        print(f"\n--- 🧬 GA 初始化完成 (种群: {self.pop_size}, 核心基因: {len(target_vars)}, 边缘基因: {len(background_vars)}) ---")

        for gen in range(self.max_generations):
            evaluated = []
            codes_to_predict = []
            keys_to_predict = []

            previous_best_fitness = best_fitness

            # 收集评估
            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                cache_key = frozenset(rename_map.items())

                if cache_key not in fitness_cache:
                    try:
                        mutated_code = self.rename_fn(code, rename_map)
                        if mutated_code:
                            codes_to_predict.append(mutated_code)
                            keys_to_predict.append(cache_key)
                    except Exception:
                        fitness_cache[cache_key] = (float('-inf'), original_pred, code, None)

            # 批量黑盒查询
            if codes_to_predict:
                batch_probs, batch_preds = self.model_zoo.batch_predict(codes_to_predict, self.target_model)
                for i in range(len(codes_to_predict)):
                    probs = batch_probs[i]
                    pred = batch_preds[i]
                    fitness = self._calculate_fitness(probs, original_pred)
                    fitness_cache[keys_to_predict[i]] = (fitness, pred, codes_to_predict[i], probs)

            # 记录最优
            generation_best_fitness = float('-inf')
            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                cache_key = frozenset(rename_map.items())

                if cache_key in fitness_cache:
                    fitness, pred, mutated_code, probs = fitness_cache[cache_key]
                    if probs is not None:
                        evaluated.append((ind, fitness, pred, mutated_code, probs))

                        if fitness > generation_best_fitness:
                            generation_best_fitness = fitness

                        if fitness > best_fitness:
                            best_fitness, best_code, best_probs, best_pred = fitness, mutated_code, probs, pred
                            current_target_prob = self._get_target_prob(probs, original_pred)
                            print(f"  [Gen {gen + 1:02d}] 🌟 突破! 适应度: {fitness:.4f} | 目标概率: {current_target_prob:.2%} | 预测: {pred}")

                        if pred != original_pred and self.run_mode == "attack":
                            final_target_prob = self._get_target_prob(probs, original_pred)
                            print(f"\n🎉 攻击成功！在第 {gen + 1} 代突破防线。最终目标概率: {final_target_prob:.2%}")
                            return True, mutated_code, probs, pred

            if best_probs is not None:
                current_target_prob = self._get_target_prob(best_probs, original_pred)
                print(f"[Gen {gen + 1:02d}/{self.max_generations}] 历史最优适应度: {best_fitness:.4f} | 目标概率: {current_target_prob:.2%}")

            # --- 繁衍逻辑 (交叉与突变现在覆盖全基因段) ---
            unique_evaluated = []
            seen_genes = set()
            for ind_tuple in evaluated:
                gene_signature = frozenset(ind_tuple[0].items())
                if gene_signature not in seen_genes:
                    seen_genes.add(gene_signature)
                    unique_evaluated.append(ind_tuple)

            if best_fitness <= previous_best_fitness + 1e-6:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            unique_evaluated.sort(key=lambda x: x[1], reverse=True)

            # 停滞重启机制 (Restart)
            if stagnation_counter >= self.stagnation_limit:
                best_elite = unique_evaluated[0][0] if unique_evaluated else population[0]
                population = [best_elite]
                while len(population) < self.pop_size:
                    ind = {}
                    for v in all_vars:
                        if random.random() < (0.8 if v in target_vars else 0.1):
                            ind[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v])
                        else:
                            ind[v] = best_elite[v]
                    population.append(ind)
                stagnation_counter = 0
                continue

            num_elites = max(2, min(len(unique_evaluated), self.pop_size // 4))
            elites = [x[0] for x in unique_evaluated[:num_elites]]

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                if len(elites) >= 2:
                    p1, p2 = random.sample(elites, 2)
                else:
                    p1, p2 = elites[0], elites[0]

                # 交叉 (Crossover): 在全量变量上进行
                child = {v: (p1[v] if random.random() > 0.5 else p2[v]) for v in all_vars}

                # 突变 (Mutation): 依据动态/非对称概率字典进行触发
                for v in child:
                    if random.random() < mutation_probs.get(v, 0.03):
                        child[v] = get_safe_choice(v, subs_pool.get(v, [v]) + [v], current_val=child[v])

                new_pop.append(child)

            population = new_pop

        if best_probs is not None:
            final_target_prob = self._get_target_prob(best_probs, original_pred)
            print(f"\n⚠️ 攻击结束。未能改变模型预测。最终目标概率峰值: {final_target_prob:.2%}")

        return (best_pred != original_pred), best_code, best_probs, best_pred


class GreedyOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        run_cfg = config.get('run_params', {}) if config else {}
        self.run_mode = run_cfg.get('run_mode', 'attack')

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None):
        """Executes a sequential greedy search to apply variable substitutions and bypass model defenses."""
        if variable_scores:
            sorted_vars = sorted(target_vars, key=lambda v: variable_scores.get(v, 0), reverse=True)
        else:
            sorted_vars = target_vars

        current_code = code
        current_best_probs = None
        current_best_pred = original_pred
        overall_best_fitness = float('-inf')
        overall_best_code = code

        for var in sorted_vars:
            candidates = list(set(subs_pool.get(var, [])))
            if not candidates:
                continue

            codes_to_predict = []
            for cand in candidates:
                if cand == var:
                    continue
                try:
                    temp_code = self.rename_fn(current_code, {var: cand})
                    if temp_code:
                        codes_to_predict.append((cand, temp_code))
                except Exception:
                    continue

            if not codes_to_predict:
                continue

            candidate_strings = [item[1] for item in codes_to_predict]

            batch_probs, batch_preds = self.model_zoo.batch_predict(candidate_strings, self.target_model)

            best_var_fitness = float('-inf')
            best_var_code = None
            best_var_probs = None
            best_var_pred = None

            for i in range(len(codes_to_predict)):
                probs = batch_probs[i]
                pred = batch_preds[i]

                orig_idx = 0 if original_pred == -1 else original_pred

                if orig_idx >= len(probs):
                    orig_idx = len(probs) - 1

                orig_prob = max(probs[orig_idx], 1e-9)

                if self.mode == "binary":
                    target_idx = 1 if original_pred == -1 else 0

                    if target_idx >= len(probs):
                        target_idx = len(probs) - 1

                    target_prob = max(probs[target_idx], 1e-9)

                    fitness = math.log(target_prob) - math.log(orig_prob)
                else:
                    fitness = -orig_prob

                if fitness > best_var_fitness:
                    best_var_fitness = fitness
                    best_var_code = candidate_strings[i]
                    best_var_probs = probs
                    best_var_pred = pred

            if best_var_code and best_var_fitness > float('-inf'):
                current_code = best_var_code
                current_best_probs = best_var_probs
                current_best_pred = best_var_pred

                if best_var_fitness > overall_best_fitness:
                    overall_best_fitness = best_var_fitness
                    overall_best_code = best_var_code

                if current_best_pred != original_pred and self.run_mode == "attack":
                    verify_probs, verify_pred = self.model_zoo.predict(current_code, self.target_model)

                    if verify_pred != original_pred:
                        return True, current_code, verify_probs, verify_pred
                    else:
                        current_best_pred = verify_pred

        final_probs, final_pred = self.model_zoo.predict(overall_best_code, self.target_model)
        is_success = (final_pred != original_pred)

        return is_success, overall_best_code, final_probs, final_pred


import math
from typing import List, Dict, Tuple, Any


class BeamSearchOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        run_cfg = config.get('run_params', {}) if config else {}
        beam_cfg = config.get('beam_params', {}) if config else {}
        self.run_mode = run_cfg.get('run_mode', 'attack')

        self.beam_size = beam_cfg.get('beam_size', 3)
        self.cand_chunk_size = beam_cfg.get('cand_chunk_size', 10)

        # ==========================================================
        # Beam 早停策略：可配置
        #   none/disabled/off/false : 不早停，遍历当前变量全部候选
        #   dynamic                 : 保留原逻辑：候选 chunk 有显著提升且变量数充足才早停
        #   gain                    : 只要候选 chunk 有显著提升就早停
        #   patience                : 连续若干 chunk 没有显著提升才早停
        # ==========================================================
        self.early_stop_delta = beam_cfg.get('early_stop_delta', 0.3)
        self.early_stop_strategy = str(
            beam_cfg.get('beam_early_stop_strategy', beam_cfg.get('early_stop_strategy', 'dynamic'))
        ).lower()
        self.early_stop_patience = int(beam_cfg.get('beam_early_stop_patience', 2))
        self.early_stop_min_valid_vars = int(beam_cfg.get('beam_early_stop_min_valid_vars', 3))

        # ==========================================================
        # AST 合法性校验：默认开启。
        # 需要在 run(...) 中传入 analyzer；如果没有传入 analyzer，自动降级为只依赖 rename_fn。
        # ==========================================================
        self.enable_ast_check = bool(beam_cfg.get('beam_enable_ast_check', True))
        self._warned_missing_analyzer = False

    def _calculate_fitness(self, probs: List[float], original_pred: int) -> float:
        """
        计算适应度 (Fitness): 评估替换词对模型置信度的破坏程度。
        返回值越大，说明该替换词的攻击潜力越高。
        """
        orig_idx = 0 if original_pred == -1 else original_pred
        orig_idx = min(orig_idx, len(probs) - 1)
        orig_prob = max(probs[orig_idx], 1e-9)

        if self.mode == "binary":
            target_idx = 1 if orig_idx == 0 else 0
            target_idx = min(target_idx, len(probs) - 1)
            target_prob = max(probs[target_idx], 1e-9)
            return math.log(target_prob) - math.log(orig_prob)

        # 多分类 Margin Loss：防止概率扩散。
        other_probs = [p for i, p in enumerate(probs) if i != orig_idx]
        max_other_prob = max(other_probs) if other_probs else 1e-9
        max_other_prob = max(max_other_prob, 1e-9)
        return math.log(max_other_prob) - math.log(orig_prob)

    def _dedup_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _is_ast_legal_rename(self, curr_code: str, var: str, cand: str, analyzer: Any = None) -> bool:
        """
        在当前 beam 状态 curr_code 上校验 var -> cand 是否仍然 AST 合法。
        注意：候选生成阶段的合法性是在原始代码上验证的；Beam 组合替换后仍可能产生新冲突，
        所以这里必须基于 curr_code 重新解析。
        """
        if not self.enable_ast_check:
            return True

        if analyzer is None:
            if not self._warned_missing_analyzer:
                print("    [Warn] Beam AST check enabled but analyzer is None; fallback to rename_fn only.")
                self._warned_missing_analyzer = True
            return True

        try:
            code_bytes = curr_code.encode("utf-8")
            identifiers = analyzer.extract_identifiers(code_bytes)

            if var not in identifiers:
                return False
            if not analyzer.can_rename_to(code_bytes, var, cand):
                return False

            from utils.ast_tools import CodeTransformer
            CodeTransformer.validate_and_apply(
                code_bytes,
                identifiers,
                {var: cand},
                analyzer=analyzer,
            )
            return True
        except Exception:
            return False

    def _should_stop_after_chunk(self, chunk_best_fitness_gain: float, valid_var_count: int,
                                 bad_chunk_count: int) -> bool:
        strategy = self.early_stop_strategy

        if strategy in {"none", "disabled", "disable", "off", "false", "0"}:
            return False

        if strategy == "gain":
            return chunk_best_fitness_gain >= self.early_stop_delta

        if strategy == "patience":
            if valid_var_count < self.early_stop_min_valid_vars:
                return False
            return bad_chunk_count >= self.early_stop_patience

        # 默认 dynamic：兼容你原来的“有明显提升就停，但变量数太少时不停”。
        if valid_var_count >= self.early_stop_min_valid_vars:
            return chunk_best_fitness_gain >= self.early_stop_delta
        return False

    def run(self, code: str, original_pred: int, target_vars: List[str], subs_pool: Dict[str, List[str]],
            variable_scores: Dict[str, float] = None, analyzer: Any = None):

        if variable_scores:
            sorted_vars = sorted(target_vars, key=lambda v: variable_scores.get(v, 0), reverse=True)
        else:
            sorted_vars = target_vars

        query_cache = {}

        def _get_predictions(codes_to_predict: List[str]) -> Tuple[List[List[float]], List[int]]:
            if not codes_to_predict:
                return [], []

            codes_to_predict = self._dedup_keep_order(codes_to_predict)
            uncached_codes = [c for c in codes_to_predict if c not in query_cache]

            if uncached_codes:
                batch_probs, batch_preds = self.model_zoo.batch_predict(uncached_codes, self.target_model)
                for c, p, pred in zip(uncached_codes, batch_probs, batch_preds):
                    query_cache[c] = (p, pred)

            cached_results = [query_cache[c] for c in codes_to_predict]
            probs_list = [res[0] for res in cached_results]
            preds_list = [res[1] for res in cached_results]
            return probs_list, preds_list

        # 基线预测。
        init_probs, init_preds = _get_predictions([code])
        orig_probs = init_probs[0]
        orig_pred = init_preds[0]

        initial_fitness = self._calculate_fitness(orig_probs, original_pred)

        beam = [(initial_fitness, code, orig_probs, orig_pred)]
        overall_best_fitness = initial_fitness
        overall_best_code = code

        # 记录当前样本可用的有效变量数量。
        valid_var_count = len([v for v in sorted_vars if subs_pool.get(v, [])])

        for var in sorted_vars:
            candidates = self._dedup_keep_order(subs_pool.get(var, []))
            if not candidates:
                continue

            new_beam_candidates = []

            for curr_fitness, curr_code, curr_probs, curr_pred in beam:
                new_beam_candidates.append((curr_fitness, curr_code, curr_probs, curr_pred))
                bad_chunk_count = 0

                for i in range(0, len(candidates), self.cand_chunk_size):
                    cand_chunk = candidates[i:i + self.cand_chunk_size]

                    codes_to_predict = []
                    for cand in cand_chunk:
                        if cand == var:
                            continue

                        # 关键新增：在当前 beam 代码状态上做 AST 合法性校验。
                        if not self._is_ast_legal_rename(curr_code, var, cand, analyzer=analyzer):
                            continue

                        try:
                            temp_code = self.rename_fn(curr_code, {var: cand})
                            if temp_code and temp_code != curr_code:
                                codes_to_predict.append(temp_code)
                        except Exception:
                            continue

                    if not codes_to_predict:
                        # 当前 chunk 全部非法时，也算一个无收益 chunk，供 patience 策略使用。
                        if self.early_stop_strategy == "patience":
                            bad_chunk_count += 1
                            if self._should_stop_after_chunk(0.0, valid_var_count, bad_chunk_count):
                                break
                        continue

                    batch_probs, batch_preds = _get_predictions(codes_to_predict)
                    chunk_best_fitness_gain = 0.0

                    for probs, pred, temp_code in zip(batch_probs, batch_preds, codes_to_predict):
                        fitness = self._calculate_fitness(probs, original_pred)
                        fitness_gain = fitness - curr_fitness

                        if fitness_gain > chunk_best_fitness_gain:
                            chunk_best_fitness_gain = fitness_gain

                        if fitness > overall_best_fitness:
                            overall_best_fitness = fitness
                            overall_best_code = temp_code

                        # 全局熔断：发现翻转立即停止攻击并返回。
                        if pred != original_pred and self.run_mode == "attack":
                            verify_probs, verify_preds = _get_predictions([temp_code])
                            if verify_preds[0] != original_pred:
                                return True, temp_code, verify_probs[0], verify_preds[0]

                        new_beam_candidates.append((fitness, temp_code, probs, pred))

                    # 可配置早停策略。
                    if self.early_stop_strategy == "patience":
                        if chunk_best_fitness_gain < self.early_stop_delta:
                            bad_chunk_count += 1
                        else:
                            bad_chunk_count = 0

                    if self._should_stop_after_chunk(chunk_best_fitness_gain, valid_var_count, bad_chunk_count):
                        break

            unique_candidates = {}
            for state in new_beam_candidates:
                # 去重逻辑：保留相同代码下 fitness 最高的状态。
                if state[1] not in unique_candidates or state[0] > unique_candidates[state[1]][0]:
                    unique_candidates[state[1]] = state

            sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x[0], reverse=True)
            beam = sorted_candidates[:self.beam_size]

        final_probs_list, final_preds_list = _get_predictions([overall_best_code])
        final_probs = final_probs_list[0]
        final_pred = final_preds_list[0]

        is_success = (final_pred != original_pred)
        return is_success, overall_best_code, final_probs, final_pred



class BayesianOptimizer:
    def __init__(self, model_zoo, target_model, rename_fn, mode="binary", config=None):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.mode = mode

        run_cfg = config.get('run_params', {}) if config else {}
        bo_cfg = config.get('bayesian', {}) if config else {}

        self.run_mode = run_cfg.get('run_mode', 'attack')
        self.max_iters = run_cfg.get('iterations', 50)

        # BO 专属参数
        self.init_samples = bo_cfg.get('init_samples', 10)
        # 优化1：大幅扩大内部采样池，由于代理模型极快，2000次毫无压力
        self.acq_samples = bo_cfg.get('acq_samples', 2000)

        # 动态 Kappa 的初始值和下限设定
        self.kappa_start = 2.5
        self.kappa_end = 0.5

    def _calculate_fitness(self, probs, original_pred):
        # 多分类模式
        if self.mode != "binary":
            orig_idx = 0 if original_pred == -1 else original_pred
            orig_idx = min(orig_idx, len(probs) - 1 if isinstance(probs, (list, np.ndarray)) else 0)
            orig_prob = max(probs[orig_idx] if isinstance(probs, (list, np.ndarray)) else probs, 1e-9)
            return -orig_prob

        is_orig_vuln = (original_pred == 1)

        p_safe = float(probs[0])
        p_vuln = float(probs[1]) if len(probs) > 1 else 1.0 - p_safe

        orig_prob = max(p_vuln if is_orig_vuln else p_safe, 1e-9)
        target_prob = max(p_safe if is_orig_vuln else p_vuln, 1e-9)

        return math.log(target_prob) - math.log(orig_prob)

    def _get_target_prob(self, probs, original_pred):
        """专门用于在日志中打印我们试图拉高的目标概率"""
        if self.mode != "binary":
            orig_idx = 0 if original_pred == -1 else original_pred
            orig_idx = min(orig_idx, len(probs) - 1 if isinstance(probs, (list, np.ndarray)) else 0)
            orig_prob = probs[orig_idx] if isinstance(probs, (list, np.ndarray)) else float(probs)
            return 1.0 - orig_prob

        is_orig_vuln = (original_pred == 1)
        p_safe = float(probs[0])
        p_vuln = float(probs[1]) if len(probs) > 1 else 1.0 - p_safe

        return p_safe if is_orig_vuln else p_vuln

    def run(self, code, original_pred, target_vars, subs_pool, variable_scores=None, rnns_best_seed=None):
        """使用贝叶斯优化在离散空间中寻找最优替换组合"""

        valid_vars = [v for v in target_vars if subs_pool.get(v)]
        if not valid_vars:
            return False, code, None, original_pred

        num_vars = len(valid_vars)

        if variable_scores:
            weights = [variable_scores.get(v, 1.0) for v in valid_vars]
            total_weight = sum(weights)
            mutation_probs = [w / total_weight for w in weights]
        else:
            mutation_probs = None

        var_candidates = {}
        categories_for_encoder = []

        # 🌟 修复 2：强制截断候选空间 (针对 BO 的降维打击)
        # BO 绝对无法在 50 个候选词中收敛。我们强制只给它每个变量前 10 个最高质量的候选词 (主要涵盖 LLM 生成的优质词)
        BO_CANDIDATE_LIMIT = 10

        for var in valid_vars:
            # 取前 BO_CANDIDATE_LIMIT 个词，极大压缩 One-Hot 维度
            top_cands = subs_pool[var][:BO_CANDIDATE_LIMIT]
            cands = [var] + [c for c in set(top_cands) if c != var]
            var_candidates[var] = cands
            categories_for_encoder.append(np.arange(len(cands)))

        encoder = OneHotEncoder(categories=categories_for_encoder, sparse_output=False)
        dummy_data = np.zeros((1, num_vars), dtype=int)
        encoder.fit(dummy_data)

        def state_to_code(state_indices):
            rename_map = {}
            for i, var in enumerate(valid_vars):
                chosen_cand = var_candidates[var][state_indices[i]]
                if chosen_cand != var:
                    rename_map[var] = chosen_cand
            if not rename_map:
                return code
            try:
                return self.rename_fn(code, rename_map)
            except Exception:
                return None

        X_history, Y_history = [], []
        seen_states = set()
        best_code, best_fitness, best_probs, best_pred = code, float('-inf'), None, original_pred

        # --- 初始化阶段 ---
        initial_states = [np.zeros(num_vars, dtype=int)] # 原代码保留

        # 🌟 修复 3：将 RNNS_BEST_SEED 注入为初始状态
        if rnns_best_seed:
            seed_state = []
            for var in valid_vars:
                seed_word = rnns_best_seed.get(var, var)
                # 尝试在截断后的候选词库中找到该词的索引
                if seed_word in var_candidates[var]:
                    seed_state.append(var_candidates[var].index(seed_word))
                else:
                    seed_state.append(0)
            initial_states.append(np.array(seed_state))

        # 填充剩余的初始化样本
        while len(initial_states) < self.init_samples:
            state = [random.randint(0, len(var_candidates[var]) - 1) for var in valid_vars]
            initial_states.append(np.array(state))

        codes_to_predict = []
        valid_initial_states = []
        for state in initial_states:
            state_tuple = tuple(state)
            if state_tuple in seen_states: continue
            seen_states.add(state_tuple)

            mutated_code = state_to_code(state)
            if mutated_code:
                codes_to_predict.append(mutated_code)
                valid_initial_states.append(state)

        if codes_to_predict:
            batch_probs, batch_preds = self.model_zoo.batch_predict(codes_to_predict, self.target_model)
            for i in range(len(codes_to_predict)):
                fit = self._calculate_fitness(batch_probs[i], original_pred)
                X_history.append(valid_initial_states[i])
                Y_history.append(fit)

                if batch_preds[i] != original_pred and self.run_mode == "attack":
                    return True, codes_to_predict[i], batch_probs[i], batch_preds[i]

                if fit > best_fitness:
                    best_fitness, best_code, best_probs, best_pred = fit, codes_to_predict[i], batch_probs[i], batch_preds[i]

        total_bo_iters = max(1, self.max_iters - len(X_history))

        for iteration in range(total_bo_iters):
            if not X_history: break

            progress = iteration / total_bo_iters
            current_kappa = self.kappa_start - (self.kappa_start - self.kappa_end) * progress

            X_encoded = encoder.transform(X_history)

            # 🌟 修复 4：极其保守的随机森林参数
            # 因为样本极少，强制限制树的深度和最小叶子节点，防止对少数样本绝对过拟合
            rf = RandomForestRegressor(
                n_estimators=50,       # 减少树的数量，加速
                max_depth=5,           # 限制树深，强制代理模型学习全局趋势而非记住单个样本
                min_samples_split=4,   # 叶子分裂限制
                random_state=42
            )

            # 如果历史样本太少，加入微小的高斯噪声防止模型方差崩溃为0
            if len(X_encoded) < 10:
                rf.fit(X_encoded, np.array(Y_history) + np.random.normal(0, 1e-5, len(Y_history)))
            else:
                rf.fit(X_encoded, Y_history)

            candidate_states = []
            top_k_indices = np.argsort(Y_history)[-min(5, len(Y_history)):]

            for _ in range(self.acq_samples):
                if random.random() < 0.8:
                    base_state = X_history[random.choice(top_k_indices)].copy()
                    num_mutations = random.randint(1, min(max(2, num_vars // 3), num_vars))

                    if mutation_probs:
                        mutate_indices = np.random.choice(range(num_vars), size=num_mutations, replace=False, p=mutation_probs)
                    else:
                        mutate_indices = random.sample(range(num_vars), num_mutations)

                    for idx in mutate_indices:
                        base_state[idx] = random.randint(0, len(var_candidates[valid_vars[idx]]) - 1)
                    candidate_states.append(base_state)
                else:
                    candidate_states.append([random.randint(0, len(var_candidates[var]) - 1) for var in valid_vars])

            unseen_candidates = [s for s in candidate_states if tuple(s) not in seen_states]
            if not unseen_candidates: continue

            unseen_candidates_encoded = encoder.transform(unseen_candidates)

            # 计算 UCB
            tree_predictions = np.array([tree.predict(unseen_candidates_encoded) for tree in rf.estimators_])
            mean_preds = np.mean(tree_predictions, axis=0)

            # 🌟 修复 5：加入微小方差平滑，防止在前期树未充分生长时 std 崩溃为 0 导致纯贪婪
            std_preds = np.std(tree_predictions, axis=0) + 1e-6

            ucb_scores = mean_preds + current_kappa * std_preds
            best_candidate_idx = np.argmax(ucb_scores)
            chosen_state = unseen_candidates[best_candidate_idx]

            mutated_code = state_to_code(chosen_state)
            if not mutated_code:
                seen_states.add(tuple(chosen_state))
                continue

            probs, pred = self.model_zoo.predict(mutated_code, self.target_model)
            fitness = self._calculate_fitness(probs, original_pred)

            seen_states.add(tuple(chosen_state))
            X_history.append(chosen_state)
            Y_history.append(fitness)

            if pred != original_pred and self.run_mode == "attack":
                return True, mutated_code, probs, pred

            if fitness > best_fitness:
                best_fitness, best_code, best_probs, best_pred = fitness, mutated_code, probs, pred

        return (best_pred != original_pred), best_code, best_probs, best_pred