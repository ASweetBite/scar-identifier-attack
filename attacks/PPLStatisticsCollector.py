import time
import json
import numpy as np
from typing import List, Dict


class PPLStatisticsCollector:
    def __init__(self, get_all_vars_fn, mlm_gen, llm_gen, config: dict):
        """
        全量 PPL 统计收集器 (依靠拉大 Quota 防止早停)。
        复用生成器内部的 PPL 拦截器，收集最真实分布。
        """
        self.get_all_vars_fn = get_all_vars_fn
        self.mlm_gen = mlm_gen
        self.llm_gen = llm_gen
        self.config = config
        self.analyzer = mlm_gen.analyzer

        # 确保初始化时清空底层探针，防止受到之前的历史数据污染
        self._reset_generator_stats(self.mlm_gen)
        self._reset_generator_stats(self.llm_gen)

    def _reset_generator_stats(self, generator):
        generator.ppl_stats = {
            "total_evaluated": 0,
            "rejected_by_abs": 0,
            "rejected_by_ratio": 0,
            "rejected_both": 0,
            "diff_distribution": [],
            "ratio_distribution": []
        }

    def collect(self, dataset: List[Dict]):
        """
        执行统计流程的主函数：严格对齐主流程的切片与上下文构建逻辑
        """
        print("\n" + "=" * 50)
        print("🚀 启动 PPL 全量统计管道 (无限制 Quota，严格对齐主流程切片与数据隔离)")
        print("=" * 50)

        # 核心修改：将目标数量拉到极大，防止 _verify_and_filter 因为 added >= quota 而提早 break，
        # 从而保证所有的 Chunk 都会被送进 PPL 计算。
        HUGE_QUOTA = 99999

        for idx, sample in enumerate(dataset):
            t_start = time.time()
            code = sample["code"]

            variables = self.get_all_vars_fn(code)
            if not variables:
                continue

            code_bytes = code.encode("utf-8")

            # 【核心修正 1】：提取全量代码的标识符，作为 PPL 计算时的全局绝对坐标
            full_identifiers = self.analyzer.extract_identifiers(code_bytes)

            batch_tasks = []

            # =========================================================
            # 严格对齐主攻击流程的 AST 切片构建逻辑
            # =========================================================
            for var in variables:
                # 使用全量标识符进行存在性校验
                if var not in full_identifiers:
                    continue

                # 判断是否为全局可调用的实体（函数、方法、类）
                is_callable_or_class = all(
                    occ.get("entity_type") in ["function", "method", "class"] for occ in full_identifiers[var])

                if is_callable_or_class:
                    target_code_str = code
                else:
                    # 核心切片：对普通变量使用 AST 代码折叠 (Code Folding)
                    try:
                        target_code_str = self.analyzer.get_folded_code(code_bytes, var)
                    except Exception as e:
                        target_code_str = code  # 切片失败时的二次容错

                # 【核心修正 2】：打包任务时，实现生成数据与验证数据的严格隔离
                batch_tasks.append({
                    "target_name": var,
                    "code_str": target_code_str,  # [生成用] 结构折叠代码 (长度变短)
                    "full_code_str": code,  # [PPL用] 原始全量代码
                    "full_identifiers": full_identifiers  # [PPL用] 原始全量绝对坐标字典
                })
            # =========================================================

            if not batch_tasks:
                continue

            print(f"   [Sample {idx}] Generating & Evaluating candidates for {len(batch_tasks)} variables...")

            # MLM 生成全量评估
            self.mlm_gen.generate_candidates(
                batch_tasks,
                top_k_mlm=50,  # 探测深度
                top_n_keep=HUGE_QUOTA,  # 取消配额拦截，强制计算所有合法变体的 PPL
                is_ppl_filter=True
            )

            # LLM 生成全量评估
            self.llm_gen.generate_candidates(
                batch_tasks,
                target_quota=HUGE_QUOTA,  # 取消配额拦截
                is_ppl_filter=True
            )

            print(f"   [Sample {idx}] Completed in {time.time() - t_start:.2f}s.")

        self._print_and_save_report()

    def _print_and_save_report(self):
        """
        聚合底层探针数据并打印报表
        """
        print("\n" + "=" * 60)
        print("📊 EMPIRICAL PPL DISTRIBUTION REPORT (For Research Paper)")
        print("=" * 60)

        # 聚合 MLM 和 LLM 的统计数据
        mlm_stats = getattr(self.mlm_gen, 'ppl_stats', {})
        llm_stats = getattr(self.llm_gen, 'ppl_stats', {})

        total_evaluated = mlm_stats.get("total_evaluated", 0) + llm_stats.get("total_evaluated", 0)

        if total_evaluated == 0:
            print("   ► No candidates were evaluated. Please check if the pipeline is functioning.")
            return

        rejected_abs = mlm_stats.get("rejected_by_abs", 0) + llm_stats.get("rejected_by_abs", 0)
        rejected_ratio = mlm_stats.get("rejected_by_ratio", 0) + llm_stats.get("rejected_by_ratio", 0)
        rejected_both = mlm_stats.get("rejected_both", 0) + llm_stats.get("rejected_both", 0)

        rejected_total = rejected_abs + rejected_ratio + rejected_both
        rejection_rate = (rejected_total / total_evaluated) * 100

        all_diffs = np.array(mlm_stats.get("diff_distribution", []) + llm_stats.get("diff_distribution", []))
        all_ratios = np.array(mlm_stats.get("ratio_distribution", []) + llm_stats.get("ratio_distribution", []))

        print(f"   ► Total Candidates Evaluated: {total_evaluated}")
        print(f"   ► Total Filtered (Rejected) : {rejected_total} ({rejection_rate:.2f}%)")
        print(f"      - Exceeded Absolute Only : {rejected_abs}")
        print(f"      - Exceeded Ratio Only    : {rejected_ratio}")
        print(f"      - Exceeded Both          : {rejected_both}")

        if len(all_diffs) > 0:
            print("\n   ► [PPL Absolute Difference Distribution]")
            print(f"      - Mean                   : {np.mean(all_diffs):+.3f}")
            print(f"      - 50th Percentile (Median): {np.median(all_diffs):+.3f}")
            print(f"      - 75th Percentile        : {np.percentile(all_diffs, 75):+.3f}")
            print(f"      - 90th Percentile        : {np.percentile(all_diffs, 90):+.3f}")
            print(f"      - 99th Percentile        : {np.percentile(all_diffs, 99):+.3f}")
            print(f"      - Max                    : {np.max(all_diffs):+.3f}")

        if len(all_ratios) > 0:
            print("\n   ► [PPL Change Ratio Distribution]")
            print(f"      - Mean                   : {np.mean(all_ratios):.3f}x")
            print(f"      - 50th Percentile (Median): {np.median(all_ratios):.3f}x")
            print(f"      - 75th Percentile        : {np.percentile(all_ratios, 75):.3f}x")
            print(f"      - 90th Percentile        : {np.percentile(all_ratios, 90):.3f}x")
            print(f"      - 99th Percentile        : {np.percentile(all_ratios, 99):.3f}x")
            print(f"      - Max                    : {np.max(all_ratios):.3f}x")

        # 保存聚合后的结果
        aggregated_stats = {
            "total_evaluated": total_evaluated,
            "rejected_by_abs": rejected_abs,
            "rejected_by_ratio": rejected_ratio,
            "rejected_both": rejected_both,
            "diff_distribution": all_diffs.tolist(),
            "ratio_distribution": all_ratios.tolist()
        }

        export_path = self.config.get('global', {}).get('result_dir', './results') + "/ppl_empirical_statistics.json"
        import os
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated_stats, f, indent=4)
            print(f"\n   💾 Full aggregated dataset saved to: {export_path}")
            print("   💡 Tip: Use this JSON to plot CDF/PDF graphs for your paper!")
        except Exception as e:
            print(f"   [!] Failed to save stats: {e}")
        print("=" * 60)