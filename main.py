import argparse
import math
import os
import random

import numpy as np
import torch
import yaml

from attacks.HeavyWeightCandidateGenerator import HeavyWeightCandidateGenerator
from attacks.IRTGAttacker import IRTGAttacker
from attacks.LightWeightCandidateGenerator import LightweightCandidateGenerator
from attacks.NormalizationAttacker import NormalizationAttacker
from attacks.PPLStatisticsCollector import PPLStatisticsCollector
from attacks.RandomAttacker import RandomAttacker
from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from utils.dataset import DatasetLoader
from utils.llm_loader import LocalLLMClient
from utils.miner import NamingDataMiner
from utils.mlm_engine import MLMEngine
from utils.model_zoo import ModelZoo, CodeSmoother


def main(args, config):
    """Orchestrates the evaluation of model robustness against various renaming attacks."""

    analyzer = IdentifierAnalyzer(lang=config['analyzer']['lang'])
    if 'heavyweight_candidate' not in config:
        config['heavyweight_candidate'] = {}
    stats_path = config['heavyweight_candidate'].get('naming_stats_path', 'naming_stats.json')
    dataset_path = config['run_params']['dataset']

    # 1. 启发式命名数据挖掘
    if not os.path.exists(stats_path):
        print(f"\n[!] 启发式命名统计字典 '{stats_path}' 不存在。")
        print(f"[*] 正在启动离线数据挖掘程序 (基于数据集: {dataset_path})...")
        miner = NamingDataMiner(analyzer)
        miner.mine_parquet(dataset_path)
        miner.export_json(stats_path)
    else:
        print(f"\n[*] 发现已存在的命名统计字典: {stats_path}，跳过挖掘阶段。")

    config['heavyweight_candidate']['naming_stats_path'] = stats_path

    # =========================================================================
    # 2. 加载底层引擎
    # =========================================================================
    print("\n[*] Loading Engines and Models...")
    mlm_engine = MLMEngine(config['mlm_engine']['model_name'])

    llm_name = config.get('llm_generator_name', 'models/qwen2.5-1.5b-code')
    llm_client = LocalLLMClient(model_name=llm_name)

    # =========================================================================
    # 3. 实例化双引擎生成器 (接口对齐)
    # =========================================================================
    # 实例化轻量级生成器 (MLM)
    lightweight_generator = LightweightCandidateGenerator(
        mlm_engine=mlm_engine,
        analyzer=analyzer,
        config=config.get('lightweight_candidate', {}),
        llm_client = llm_client,
    )

    # 实例化重量级生成器 (LLM) - 注意参数名对齐了 embedder
    heavyweight_generator = HeavyWeightCandidateGenerator(
        embedder=mlm_engine,
        llm_client=llm_client,
        analyzer=analyzer,
        config=config['heavyweight_candidate']
    )

    # =========================================================================
    # 4. 初始化周边组件
    # =========================================================================
    smoother_cfg = config['smoother']
    smoother = CodeSmoother(smoother_cfg, candidate_generator=lightweight_generator)

    model_configs = config['model_zoo']
    model_zoo = ModelZoo(
        model_configs=model_configs,
        eval_mode=args.mode,
        config=config,
        smoother=smoother
    )
    transformer = CodeTransformer()

    def get_all_identifiers_fn(code_str: str) -> list:
        """Extracts all identifiers from the code except 'main'."""
        data = analyzer.extract_identifiers(code_str.encode("utf-8"))
        return [name for name in data.keys() if name != "main"]

    def rename_fn(code_str: str, renaming_map: dict) -> str:
        """Applies variable renaming using the code transformer."""
        code_bytes = code_str.encode("utf-8")
        ids = analyzer.extract_identifiers(code_bytes)
        return transformer.validate_and_apply(code_bytes, ids, renaming_map, analyzer=analyzer)

    # =========================================================================
    # 5. 实例化并启动全新架构的 IRTGAttacker
    # =========================================================================
    run_params = config['run_params']

    evaluator = IRTGAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        mlm_gen=lightweight_generator,      # <--- 注入轻量级生成器
        llm_gen=heavyweight_generator,      # <--- 注入重量级生成器
        rename_fn=rename_fn,
        mode=args.mode,
        config=config
    )

    loader = DatasetLoader()
    print(f"\n[*] Loading dataset in {args.mode} mode...")
    dataset = loader.load_parquet_dataset(
        filepath=run_params['dataset'],
        mode=args.mode,
        max_samples=run_params['samples'],
        label_map_path=run_params.get('label_map'),
        random_seed = run_params.get('random_seed', 42)
    )
    collector = PPLStatisticsCollector(
        get_all_vars_fn=get_all_identifiers_fn,
        mlm_gen=lightweight_generator,
        llm_gen=heavyweight_generator,
        config=config
    )

    # collector.collect(dataset)
    # 执行攻击
    asr_matrix_vrtg = evaluator.attack(dataset)

    # normalier = NormalizationAttacker(
    #     model_zoo=model_zoo,
    #     rename_fn=rename_fn,
    #     mode=args.mode
    # )
    # asr_matrix_norm = normalier.attack(dataset)
    #
    # print("\n" + "=" * 80)
    # print("🚀  RUNNING RANDOM RENAMING ATTACK")
    # print("=" * 80)
    #
    # random_attacker = RandomAttacker(
    #     model_zoo=model_zoo,
    #     get_all_vars_fn=get_all_identifiers_fn,
    #     get_subs_pool_fn=get_subs_pool_fn,
    #     rename_fn=rename_fn,
    #     mode=args.mode
    # )
    # random_attacker.set_analyzer(analyzer)
    # asr_matrix_random = random_attacker.attack(dataset)
    #
    # print("\n" + "=" * 80)
    # print("🏆  MODEL DEFENSE SCORES")
    #
    # W_VRTG = config['scoring_weights']['W_VRTG']
    # W_NORM = config['scoring_weights']['W_NORM']
    # W_RAND = config['scoring_weights']['W_RAND']
    #
    # print(
    #     f"Weight Distribution: VRTG({int(W_VRTG * 100)}%) + Normalization({int(W_NORM * 100)}%) + Random({int(W_RAND * 100)}%)")
    # print("=" * 80)
    #
    # model_names = model_zoo.model_names
    # defense_scores = {}
    #
    # for m in model_names:
    #     vrtg_self_asr = asr_matrix_vrtg.get(m, {}).get(m, 0.0)
    #     norm_self_asr = asr_matrix_norm.get(m, {}).get(m, 0.0)
    #     rand_self_asr = asr_matrix_random.get(m, {}).get(m, 0.0)
    #
    #     vrtg_def = 100 - vrtg_self_asr
    #     norm_def = 100 - norm_self_asr
    #     rand_def = 100 - rand_self_asr
    #
    #     total_score = (vrtg_def * W_VRTG) + (norm_def * W_NORM) + (rand_def * W_RAND)
    #
    #     defense_scores[m] = {
    #         "total": round(total_score, 2),
    #         "vrtg_asr": round(vrtg_self_asr, 2),
    #         "norm_asr": round(norm_self_asr, 2),
    #         "rand_asr": round(rand_self_asr, 2),
    #         "vrtg_def": round(vrtg_def, 2)
    #     }
    #
    # header = f"{'Target Model':<20} | {'VRTG ASR':<12} | {'Norm ASR':<12} | {'Rand ASR':<12} | {'OVERALL SCORE'}"
    # print(header)
    # print("-" * len(header))
    #
    # ranked_models = sorted(defense_scores.items(), key=lambda x: x[1]['total'], reverse=True)
    #
    # for model_name, data in ranked_models:
    #     print(
    #         f"{model_name:<20} | {data['vrtg_asr']:>10.2f}% | {data['norm_asr']:>10.2f}% | {data['rand_asr']:>10.2f}% | {data['total']:>13} / 100")
    #
    # print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial sample generation attack tool")

    parser.add_argument("--mode", type=str, choices=["binary", "multi"], default="binary",
                        help="Select run mode: binary or multi")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="System configuration file path (YAML format)")

    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        parser.error(f"❌ Configuration file not found: {args.config}. Please ensure the file exists!")

    run_params = config.get('run_params', {})
    if args.mode == "multi" and run_params.get('label_map') is None:
        parser.error(
            "❌ When --mode=multi, the run_params.label_map parameter must be provided in the configuration file")

    seed = config.get('global', {}).get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    main(args, config)