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
from attacks.PPLStatisticsCollector import PPLStatisticsCollector
from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from utils.dataset import DatasetLoader
from utils.llm_loader import LocalLLMClient
from utils.miner import NamingDataMiner
from utils.mlm_engine import MLMEngine
from utils.model_zoo import ModelZoo, CodeSmoother


def main(args, config):
    """Orchestrates the evaluation of model robustness against various renaming attacks."""

    # 读取全局语言
    lang = config['global'].get('lang', 'cpp')
    analyzer = IdentifierAnalyzer(lang=lang)

    # 路径映射调整
    cg_config = config.get('candidate_generation', {})
    stats_path = cg_config.get('naming_stats_path', 'naming_stats.json')
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

    # 确保保存到配置中，供后续对象使用
    if 'candidate_generation' not in config:
        config['candidate_generation'] = {}
    config['candidate_generation']['naming_stats_path'] = stats_path

    # =========================================================================
    # 2. 加载底层引擎
    # =========================================================================
    print("\n[*] Loading Engines and Models...")
    mlm_engine_name = config['models'].get('mlm_engine', 'microsoft/codebert-base-mlm')
    mlm_engine = MLMEngine(mlm_engine_name)

    llm_name = config['models'].get('llm_generator', 'models/qwen2.5-1.5b-code')
    llm_client = LocalLLMClient(model_name=llm_name)

    # =========================================================================
    # 3. 实例化双引擎生成器 (接口对齐)
    # =========================================================================
    # 传入全量 config，生成器内部按需解析
    lightweight_generator = LightweightCandidateGenerator(
        mlm_engine=mlm_engine,
        analyzer=analyzer,
        config=config,  # <--- 必须传完整的 config
        llm_client=llm_client,
    )

    heavyweight_generator = HeavyWeightCandidateGenerator(
        embedder=mlm_engine,
        llm_client=llm_client,
        analyzer=analyzer,
        config=config  # <--- 必须传完整的 config
    )

    # =========================================================================
    # 4. 初始化周边组件
    # =========================================================================
    smoother_cfg = config['smoother']
    smoother = CodeSmoother(smoother_cfg, candidate_generator=lightweight_generator)

    model_configs = config['models'].get('target_models', {})
    model_zoo = ModelZoo(
        model_configs=model_configs,
        eval_mode=args.mode,
        config=config,
        smoother=smoother
    )
    transformer = CodeTransformer()

    def get_all_identifiers_fn(code_str: str) -> list:
        data = analyzer.extract_identifiers(code_str.encode("utf-8"))
        return [name for name in data.keys() if name != "main"]

    def rename_fn(code_str: str, renaming_map: dict) -> str:
        code_bytes = code_str.encode("utf-8")
        ids = analyzer.extract_identifiers(code_bytes)
        return transformer.validate_and_apply(code_bytes, ids, renaming_map, analyzer=analyzer)

    # =========================================================================
    # 5. 实例化并启动全新架构的 IRTGAttacker
    # =========================================================================
    # 将算法和迭代次数等写回 run_params 供下游兼容（因为 IRTGAttacker 里可能读 run_params）
    config['run_params']['algorithm'] = config['attack'].get('algorithm', 'beam')
    config['run_params']['iterations'] = config['attack'].get('iterations', 25)

    # 补充 irtg_attacker 节点兼容 IRTGAttacker
    config['irtg_attacker'] = config['attack'].get('irtg', {})
    config['heavyweight_candidate'] = config['candidate_generation'].get('heavyweight', {})

    evaluator = IRTGAttacker(
        model_zoo=model_zoo,
        get_all_vars_fn=get_all_identifiers_fn,
        mlm_gen=lightweight_generator,
        llm_gen=heavyweight_generator,
        rename_fn=rename_fn,
        mode=args.mode,
        config=config
    )

    loader = DatasetLoader()
    print(f"\n[*] Loading dataset in {args.mode} mode...")
    run_params = config['run_params']
    dataset = loader.load_parquet_dataset(
        filepath=run_params['dataset'],
        mode=args.mode,
        max_samples=run_params['samples'],
        label_map_path=run_params.get('label_map'),
        random_seed=config['global'].get('random_seed', 42)
    )

    # collector = PPLStatisticsCollector(...)

    # 执行攻击
    asr_matrix_vrtg, avg_queries = evaluator.attack(dataset)

    # normalizer, random attacker... (按需开启)


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
        parser.error("❌ When --mode=multi, the label_map must be provided")

    seed = config.get('global', {}).get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    main(args, config)