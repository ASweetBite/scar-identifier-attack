import gc
import json
import math
import os
import time
from typing import List, Dict

import torch

from attacks.optimizers import GeneticAlgorithmOptimizer, GreedyOptimizer, BeamSearchOptimizer, BayesianOptimizer
from attacks.rankers import RNNS_Ranker
from utils.model_zoo import ModelZooQueryTracker


class IRTGAttacker:
    def __init__(self, model_zoo, get_all_vars_fn, mlm_gen, llm_gen, rename_fn, mode: str, config: dict):
        self.model_zoo = ModelZooQueryTracker(model_zoo)

        self.model_names = self.model_zoo.model_names
        self.mode = mode
        self.config = config

        _global = config.get('global', {})
        run_params = config.get('run_params', {})
        irtg_config = config.get('irtg_attacker', {})
        hw_config = config.get('heavyweight_candidate', {})

        self.result_dir = _global.get('result_dir', "./results")
        self.top_k = irtg_config.get('top_k', 5)
        self.iterations = run_params.get('iterations', 10)
        self.run_mode = run_params.get('run_mode', 'attack')

        self.total_quota = hw_config.get('top_n_keep', 50)
        self.llm_target_quota = max(1, int(self.total_quota * 0.5))

        # LLM 相关流程参数：集中配置，避免在 attack() 中写死。
        self.llm_probe_quota = int(irtg_config.get('llm_probe_quota', run_params.get('llm_probe_quota', 2)))
        self.max_llm_enrich_attempts = int(
            irtg_config.get('max_llm_enrich_attempts', run_params.get('max_llm_enrich_attempts', 1))
        )
        # 可选：LLM 深度补强后，对目标变量做一次轻量 RNNS 重排。默认关闭，避免额外 query。
        self.rerank_after_llm_enrich = bool(
            irtg_config.get('rerank_after_llm_enrich', run_params.get('rerank_after_llm_enrich', False))
        )

        self.optimizer_type = str(run_params.get('algorithm', 'greedy')).lower()
        if self.optimizer_type not in ["greedy", "beam", "ga", "bo"]:
            raise ValueError(f"Unsupported algorithm: {self.optimizer_type}.")

        self.get_all_vars_fn = get_all_vars_fn
        self.mlm_gen = mlm_gen
        self.llm_gen = llm_gen
        self.rename_fn = rename_fn

    def _merge_candidate_pools(self, mlm_pool: dict, llm_pool: dict, final_quota: int = 20) -> dict:
        """
        按候选来源合并候选词池：
        1. 先放入 LLM 候选词，确保 LLM 生成结果不会因数量不足被丢失。
        2. 如果 LLM 候选数量不足，则用 MLM 候选词补齐。
        3. 候选词去重逻辑由上游 cache 的 set 去重负责，因此最终顺序不强制保序。
        """
        final_pool = {}
        all_vars = set(mlm_pool.keys()).union(set(llm_pool.keys()))

        for var in all_vars:
            llm_cands = llm_pool.get(var, [])
            mlm_cands = mlm_pool.get(var, [])

            merged_cands = list(llm_cands)

            if len(merged_cands) < final_quota:
                for cand in mlm_cands:
                    if cand not in merged_cands:
                        merged_cands.append(cand)
                        if len(merged_cands) >= final_quota:
                            break

            final_pool[var] = merged_cands[:final_quota]

        return final_pool

    def attack(self, dataset: List[Dict]):
        stats = {atk: {vic: {"total": 0, "fooled": 0, "success_queries": []} for vic in self.model_names} for atk in
                 self.model_names}
        storage_orig = {m: [] for m in self.model_names}
        storage_adv = {m: [] for m in self.model_names}

        total_valid_sample_time = 0.0
        valid_sample_count = 0
        model_time_stats = {m: 0.0 for m in self.model_names}
        model_valid_counts = {m: 0 for m in self.model_names}
        shared_prep_time = 0.0

        # 初始化 Ranker 和 Optimizer
        rankers = {m: RNNS_Ranker(self.model_zoo, m, self.rename_fn) for m in self.model_names}
        optimizers = {}
        for m in self.model_names:
            opt_kwargs = {"model_zoo": self.model_zoo, "target_model": m, "rename_fn": self.rename_fn,
                          "mode": self.mode, "config": self.config}
            if self.optimizer_type == "greedy":
                optimizers[m] = GreedyOptimizer(**opt_kwargs)
            elif self.optimizer_type == "beam":
                optimizers[m] = BeamSearchOptimizer(**opt_kwargs)
            elif self.optimizer_type == "ga":
                optimizers[m] = GeneticAlgorithmOptimizer(**opt_kwargs)
            elif self.optimizer_type == "bo":
                optimizers[m] = BayesianOptimizer(**opt_kwargs)

        for idx, sample in enumerate(dataset):
            t_sample_start = time.time()
            t_shared_start = time.time()

            code = sample["code"]
            ground_truth = sample.get("label")
            orig_predictions = {}
            has_correct_pred = False

            for m in self.model_names:
                probs, pred = self.model_zoo.predict(code, m)
                orig_predictions[m] = {"probs": probs, "pred": pred}
                if pred == ground_truth:
                    has_correct_pred = True

            if not has_correct_pred:
                print(f"[Sample {idx}] 所有模型初始预测均错误，跳过。")
                continue

            variables = self.get_all_vars_fn(code)
            if not variables: continue

            code_bytes = code.encode("utf-8")
            analyzer = self.mlm_gen.analyzer

            # === AST 解析 ===
            t_ast_start = time.time()
            full_identifiers = analyzer.extract_identifiers(code_bytes)
            batch_tasks = []

            for var in variables:
                if var not in full_identifiers: continue
                is_callable_or_class = all(
                    occ.get("entity_type") in ["function", "method", "class"] for occ in full_identifiers[var])

                if is_callable_or_class:
                    target_code_str = code
                else:
                    try:
                        target_code_str = analyzer.get_folded_code(code_bytes, var)
                    except Exception:
                        target_code_str = code

                batch_tasks.append({
                    "target_name": var,
                    "code_str": target_code_str,
                    "full_code_str": code,
                    "full_identifiers": full_identifiers
                })

            print(f"    [Time] AST Folding & Setup took {time.time() - t_ast_start:.2f}s")

            # 1. 预选阶段 - 全局 MLM 满载生成 (天然复用)
            print(f" -> Running FULL MLM Generation (Target: {self.total_quota} cands/var for ALL vars)...")
            t_mlm_start = time.time()
            mlm_full_pool = {}
            MAX_BATCH_SIZE = 4

            for i in range(0, len(batch_tasks), MAX_BATCH_SIZE):
                chunk = batch_tasks[i:i + MAX_BATCH_SIZE]
                try:
                    chunk_pool = self.mlm_gen.generate_candidates(
                        chunk, top_k_mlm=max(40, self.total_quota + 10), top_n_keep=self.total_quota,
                    )
                    mlm_full_pool.update(chunk_pool)
                finally:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"    [Time] Global MLM Generation took {time.time() - t_mlm_start:.2f}s")

            variables = [v for v in variables if mlm_full_pool.get(v)]
            if not variables: continue

            # === 初始化当前样本的 LLM 全局缓存 ===
            # 结构: { variable_name: [candidate_1, candidate_2, ...] }
            # 注意：候选词顺序代表质量优先级，因此必须保序去重。
            batch_tasks_by_var = {task["target_name"]: task for task in batch_tasks}
            sample_llm_cache = {v: [] for v in variables}
            deep_enrich_attempts = {v: 0 for v in variables}

            # 2. LLM 浅层探针 (Probe)
            llm_probe_quota = self.llm_probe_quota
            print(f" -> Running LLM Shallow Probe (Target: {llm_probe_quota} cands/var)...")
            t_probe_start = time.time()
            try:
                llm_probe_pool = self.llm_gen.generate_candidates(batch_tasks, target_quota=llm_probe_quota)
                for var, cands in llm_probe_pool.items():
                    if var in sample_llm_cache:
                        sample_llm_cache[var] = list(set(cands))
            finally:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"    [Time] LLM Probe Generation took {time.time() - t_probe_start:.2f}s")

            # 合并探测池用于 RNNS 分析
            rnns_eval_pool = self._merge_candidate_pools(mlm_full_pool, sample_llm_cache, final_quota=self.total_quota)

            current_shared_time = time.time() - t_shared_start
            sample_attacked_by_any = False

            # =====================================================================
            # 模型独立攻击循环
            # =====================================================================
            for atk_model in self.model_names:
                t_atk_model_start = time.time()

                orig_pred = orig_predictions[atk_model]["pred"]
                if orig_pred != ground_truth:
                    print(f"[{atk_model}] 初始预测错误，跳过该模型攻击流程。")
                    continue

                print(f"\n[{atk_model}] Optimizer={self.optimizer_type.upper()} ({self.run_mode} mode)")
                stats[atk_model][atk_model]["total"] += 1
                rnns_best_seed = None
                self.model_zoo.reset_counter()

                # 1. RNNS 显著性分析
                print(" -> Running RNNS Saliency Analysis...")
                t_rnns_start = time.time()
                top_k = max(self.top_k, int(len(variables) * 0.5))
                rnns_output = rankers[atk_model].rank_variables(
                    code=code, variables=variables.copy(), subs_pool=rnns_eval_pool,
                    reference_label=orig_pred, top_k=top_k
                )

                if len(rnns_output) == 3:
                    ranked_vars, all_scores, rnns_best_seed = rnns_output
                else:
                    ranked_vars, all_scores = rnns_output
                print(f"    [Time] RNNS Analysis took {time.time() - t_rnns_start:.2f}s")

                if self.optimizer_type in ["greedy", "beam"]:
                    target_vars = ranked_vars
                else:
                    target_vars = ranked_vars[:top_k]
                target_scores = {var: all_scores[var] for var in target_vars}

                t_enrich_start = time.time()
                tasks_to_generate = []

                for var in target_vars:
                    task = batch_tasks_by_var.get(var)
                    if not task:
                        continue

                    cached_cands = sample_llm_cache.get(var, [])
                    attempts = deep_enrich_attempts.get(var, 0)
                    if len(cached_cands) < self.llm_target_quota and attempts < self.max_llm_enrich_attempts:
                        tasks_to_generate.append(task)

                deep_enriched_this_round = False
                if tasks_to_generate:
                    print(
                        f" -> Cache Miss! Running LLM Deep Enrichment for {len(tasks_to_generate)} target vars...")
                    missed_vars = [t['target_name'] for t in tasks_to_generate]
                    print(f"    [Info] Variables sent to LLM: {missed_vars}")

                    try:
                        new_llm_pool = self.llm_gen.generate_candidates(
                            tasks_to_generate, target_quota=self.llm_target_quota
                        )

                        for var in missed_vars:
                            deep_enrich_attempts[var] = deep_enrich_attempts.get(var, 0) + 1

                        # 将新生成的词合并到全局缓存中：保序去重，保留 LLM 输出质量顺序。
                        for var, cands in new_llm_pool.items():
                            old_cands = sample_llm_cache.get(var, [])
                            merged = list(set(old_cands + list(cands or [])))

                            if len(merged) > len(old_cands):
                                deep_enriched_this_round = True

                            sample_llm_cache[var] = merged
                    finally:
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                else:
                    print(
                        " -> Cache Hit! All target variables have sufficient LLM candidates or reached max attempts.")

                print(
                    f"    [Time] Target Enrichment (Cache Check & Generation) took {time.time() - t_enrich_start:.2f}s")

                # 构建当前模型的最终候选池 (满载 MLM + 缓存 LLM)
                final_subs_pool = self._merge_candidate_pools(mlm_full_pool, sample_llm_cache, self.total_quota)

                # 可选：LLM 深度补强后用最终候选池对目标变量做轻量重排，避免 RNNS 只看浅层 probe。
                if self.rerank_after_llm_enrich and deep_enriched_this_round and target_vars:
                    print(" -> Re-running lightweight RNNS after LLM enrichment...")
                    t_rerank_start = time.time()
                    rerank_vars = target_vars[:top_k] if len(target_vars) > top_k else target_vars
                    rerank_output = rankers[atk_model].rank_variables(
                        code=code,
                        variables=rerank_vars.copy(),
                        subs_pool=final_subs_pool,
                        reference_label=orig_pred,
                        top_k=len(rerank_vars)
                    )
                    if len(rerank_output) == 3:
                        target_vars, rerank_scores, rnns_best_seed = rerank_output
                    else:
                        target_vars, rerank_scores = rerank_output
                    target_scores = {var: rerank_scores.get(var, all_scores.get(var, 0.0)) for var in target_vars}
                    print(f"    [Time] RNNS Re-rank took {time.time() - t_rerank_start:.2f}s")

                # 2. 优化器执行
                print(" -> Attack execution started...")
                t_opt_start = time.time()
                run_kwargs = {
                    "code": code, "original_pred": orig_pred,
                    "target_vars": target_vars, "subs_pool": final_subs_pool,
                    "variable_scores": target_scores
                }
                if self.optimizer_type == "ga":
                    if rnns_best_seed: run_kwargs["rnns_best_seed"] = rnns_best_seed
                    run_kwargs["all_vars"] = ranked_vars
                    run_kwargs["variable_scores"] = all_scores
                if self.optimizer_type == "bo":
                    run_kwargs["rnns_best_seed"] = rnns_best_seed

                is_success, adv_code, adv_probs, adv_pred = optimizers[atk_model].run(**run_kwargs)
                print(f"    [Time] Optimizer ({self.optimizer_type.upper()}) Run took {time.time() - t_opt_start:.2f}s")

                # 状态与耗时落盘
                queries_consumed = self.model_zoo.get_query_count()
                if is_success:
                    stats[atk_model][atk_model]["fooled"] += 1
                    stats[atk_model][atk_model]["success_queries"].append(queries_consumed)
                    storage_adv[atk_model].append(
                        {"original_code": code, "adversarial_code": adv_code, "label": ground_truth})
                    print(f"    ✅ Success | {orig_pred} -> {adv_pred} | Queries: {queries_consumed}")

                    # 迁移攻击评估...
                    for vic_model in self.model_names:
                        if vic_model == atk_model: continue
                        if orig_predictions[vic_model]["pred"] == ground_truth:
                            stats[atk_model][vic_model]["total"] += 1
                            _, vic_adv_pred = self.model_zoo.predict(adv_code, vic_model)
                            if vic_adv_pred != orig_predictions[vic_model]["pred"]:
                                stats[atk_model][vic_model]["fooled"] += 1
                else:
                    print(f"    ❌ Failed | Queries: {queries_consumed}")

                model_elapsed = time.time() - t_atk_model_start
                model_time_stats[atk_model] += model_elapsed
                model_valid_counts[atk_model] += 1
                sample_attacked_by_any = True

                print(f"    [Time] Total processing time for model '{atk_model}': {model_elapsed:.2f}s")

            sample_elapsed = time.time() - t_sample_start
            if sample_attacked_by_any:
                total_valid_sample_time += sample_elapsed
                valid_sample_count += 1
                shared_prep_time += current_shared_time

            print("-" * 50)
            print(f"\n[Time] Total elapsed time for Sample {idx}: {sample_elapsed:.2f}s\n" + "-" * 50)
        print("\n" + "=" * 50)
        print("🎯 FINAL ATTACK SUMMARY")
        print("=" * 50)
        asr_matrix = {}
        avg_queries = {}
        for atk_m in self.model_names:
            asr_matrix[atk_m] = {}
            import numpy as np
            success_queries = stats[atk_m][atk_m]["success_queries"]
            avg_q = round(np.mean(success_queries), 2) if success_queries else 0.0
            avg_queries[atk_m] = avg_q

            total_atk = stats[atk_m][atk_m]["total"]
            fooled_atk = stats[atk_m][atk_m]["fooled"]
            asr_atk = (fooled_atk / total_atk * 100) if total_atk > 0 else 0.0

            print(f"🛡️ Target Model: {atk_m.upper()}")
            print(f"   ► ASR (Attack Success Rate) : {asr_atk:.2f}% ({fooled_atk}/{total_atk})")
            print(f"   ► Avg. Queries (Success)    : {avg_q}")
            print("-" * 50)

            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                asr_matrix[atk_m][vic_m] = round(asr, 2)

        # === 修改 4：按模型分列显示的精确时间统计面板 ===
        print("\n" + "=" * 50)
        print("⏱️ TIME STATISTICS (Valid Samples Only)")
        print("=" * 50)
        avg_sample_time = (total_valid_sample_time / valid_sample_count) if valid_sample_count > 0 else 0.0
        avg_shared_time = (shared_prep_time / valid_sample_count) if valid_sample_count > 0 else 0.0

        print(f"   ► Valid Attacked Samples    : {valid_sample_count}")
        print(f"   ► Avg. Total Time / Sample  : {avg_sample_time:.2f}s")
        print(f"   ► Avg. Shared Prep Time     : {avg_shared_time:.2f}s (AST, Global MLM, LLM Probe)")
        print("-" * 50)
        print("   [Breakdown by Target Model (RNNS + LLM Enrich + Optimizer)]")

        for m in self.model_names:
            m_count = model_valid_counts[m]
            avg_m_time = (model_time_stats[m] / m_count) if m_count > 0 else 0.0
            print(f"     * {m.upper():<12} | Valid attacks: {m_count:<3} | Avg Time: {avg_m_time:.2f}s")

        print("=" * 50)
        self.save_results(storage_orig, storage_adv)

        return asr_matrix, avg_queries


    def save_results(self, storage_orig, storage_adv):
        """Saves original and adversarial samples to JSON files based on the configured result directory."""
        result_dir = self.result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for model in self.model_names:
            if self.run_mode == "dataset":
                if storage_orig[model]:
                    orig_filename = f"orig_dataset_{model}_{self.mode}.json"
                    orig_path = os.path.join(result_dir, orig_filename)
                    self._write_json(orig_path, storage_orig[model])

                if storage_adv[model]:
                    adv_filename = f"adv_dataset_{model}_{self.mode}.json"
                    adv_path = os.path.join(result_dir, adv_filename)
                    self._write_json(adv_path, storage_adv[model])
            else:
                if storage_adv[model]:
                    filename = f"adv_test_set_{model}_{self.mode}.json"
                    file_path = os.path.join(result_dir, filename)
                    self._write_json(file_path, storage_adv[model])

    def _write_json(self, filename, data):
        """Handles the standard JSON serialization and file writing process for sample results."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[INFO] Saved {len(data)} samples to: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save {filename}: {e}")

    def print_summary(self, stats):
        """Prints a formatted matrix displaying the Attack Success Rate (ASR) across all target and victim models."""
        print("\n" + "=" * 90)
        print("📊 FINAL CROSS-MODEL TRANSFERABILITY MATRIX (ASR %)")
        print("=" * 90)
        header = f"{'Attacker \\ Victim':<20} |"
        for m in self.model_names:
            header += f" {m:<13} |"
        print(header)
        print("-" * len(header))
        for atk_m in self.model_names:
            row = f"{atk_m:<20} |"
            for vic_m in self.model_names:
                total = stats[atk_m][vic_m]["total"]
                fooled = stats[atk_m][vic_m]["fooled"]
                asr = (fooled / total * 100) if total > 0 else 0.0
                row += f" {asr:>11.2f}% |"
            print(row)
        print("=" * 90 + "\n")