import logging
import random
from typing import List, Dict, Tuple
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

from utils.ast_tools import IdentifierAnalyzer, CodeTransformer

logger = logging.getLogger(__name__)


class ModelZooQueryTracker:
    """
    黑盒查询拦截器：记录所有预测查询开销。
    """

    def __init__(self, model_zoo):
        self._model_zoo = model_zoo
        self._query_count = 0

    def reset_counter(self):
        self._query_count = 0

    def get_query_count(self):
        return self._query_count

    def predict(self, *args, **kwargs):
        self._query_count += 1
        return self._model_zoo.predict(*args, **kwargs)

    def batch_predict(self, codes, *args, **kwargs):
        self._query_count += len(codes)
        return self._model_zoo.batch_predict(codes, *args, **kwargs)

    def predict_label_conf(self, *args, **kwargs):
        self._query_count += 1
        return self._model_zoo.predict_label_conf(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model_zoo, name)


class CodeSmoother:
    def __init__(self, config: Dict, candidate_generator):
        """Initializes the smoother with Monte Carlo sampling parameters."""
        self.num_samples = config.get("num_samples", 50)
        self.variance_threshold = config.get("variance_threshold", 0.05)
        self.replace_prob = config.get("replace_prob", 0.5)
        self.batch_size = config.get("batch_size", 32)
        self.candidate_generator = candidate_generator
        # ✨ 修改点：使用 python 的解析器
        self.analyzer = IdentifierAnalyzer(lang="python")

    def generate_smoothed_samples(self, code: str, candidate_dict: dict = None, sensitive_vars: list = None) -> List[
        str]:
        """Generates batch Monte Carlo variants of the input code for randomized smoothing."""
        code_bytes = code.encode("utf-8")
        try:
            identifiers = self.analyzer.extract_identifiers(code_bytes)
        except Exception as e:
            logger.warning(f"AST parsing failed, returning original code: {e}")
            return [code] * self.num_samples

        if not identifiers:
            return [code] * self.num_samples

        samples = []
        for _ in range(self.num_samples):
            if sensitive_vars:
                targets = [v for v in identifiers if v in sensitive_vars and random.random() < self.replace_prob]
            else:
                targets = [v for v in identifiers if random.random() < self.replace_prob]

            if not targets:
                samples.append(code)
                continue

            rename_map = {}
            for t in targets:
                if candidate_dict and t in candidate_dict and candidate_dict[t]:
                    rename_map[t] = random.choice(candidate_dict[t])
                else:
                    cands = self.candidate_generator.get_random_replacement(code, [t])
                    if cands and t in cands:
                        rename_map[t] = cands[t]

            if not rename_map:
                samples.append(code)
            else:
                transformed = CodeTransformer.validate_and_apply(code_bytes, identifiers, rename_map, self.analyzer)
                samples.append(transformed if transformed else code)

        return samples


class ModelZoo:
    def __init__(self, model_configs: dict, eval_mode: str, config: dict):
        glob_cfg = config.get('global', {})
        run_cfg = config.get('run_params', {})

        self.device = torch.device(glob_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        # eval_mode 参数这里可以保留占位，但在作者溯源中我们固定走多分类
        self.max_seq_len = run_cfg.get('max_seq_len', 512)

        # ✨ 修改点：作者溯源是固定的 66 分类（或其他由外部传入的类别数）
        self.num_classes = run_cfg.get('num_classes', 66)
        print(f"[*] ModelZoo running in Authorship Attribution Mode (num_classes = {self.num_classes})")

        self.models = {}
        self.model_names = list(model_configs.keys())

        # 动态加载 DFG 特征提取器（适配 Python）
        self.analyzer = None
        if any("graphcodebert" in name.lower() for name in self.model_names):
            print("[*] Detected GraphCodeBERT in targets. Initializing Python DFG Extractor...")
            self.analyzer = IdentifierAnalyzer(lang="python")

        for name, path in model_configs.items():
            print(f"\n[*] Loading Model[{name}] from {path}...")

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[!] CRITICAL: Target path {path} not found for model '{name}'. Aborting init.")

            try:
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
                adapter_config_path = os.path.join(path, "adapter_config.json")

                if os.path.exists(adapter_config_path):
                    print(f"    |- Detected LoRA adapter. Parsing config...")
                    with open(adapter_config_path, 'r', encoding='utf-8') as f:
                        peft_config = json.load(f)

                    base_model_name = peft_config.get("base_model_name_or_path")

                    if not base_model_name or (not os.path.exists(base_model_name) and (
                            "/" in base_model_name or "\\" in base_model_name) and "microsoft" not in base_model_name):
                        name_lower = name.lower()
                        if "graphcodebert" in name_lower:
                            base_model_name = "microsoft/graphcodebert-base"
                        elif "unixcoder" in name_lower:
                            base_model_name = "microsoft/unixcoder-base"
                        else:
                            base_model_name = "microsoft/codebert-base"
                        print(f"    |- [*] Auto-fallback to HF Hub: {base_model_name}")

                    print(f"    |- Loading Standard HF Classifier Skeleton from: {base_model_name}")
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_name,
                        num_labels=self.num_classes,
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True
                    )

                    print(f"    |- Loading LoRA Adapters and Classifier from {path}...")
                    model = PeftModel.from_pretrained(base_model, path)
                    print("    |- ✅ LoRA weights and Custom Classifier successfully injected.")

                else:
                    print(f"    |- Loading standard HF classifier...")
                    model = AutoModelForSequenceClassification.from_pretrained(
                        path,
                        num_labels=self.num_classes,
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True
                    )
                    print("    |- ✅ Standard model loaded successfully.")

                model.to(self.device)
                model.eval()
                self.models[name] = {"type": "transformer", "tokenizer": tokenizer, "model": model}
                print(f"[+] Successfully loaded {name} to {self.device}")

            except Exception as e:
                import traceback
                print("\n" + "=" * 50)
                print(f"🚨 FAILED TO LOAD MODEL: {name}")
                traceback.print_exc()
                print("=" * 50 + "\n")
                raise RuntimeError(f"Failed to load model '{name}'. Execution halted.") from e

    def _encode_graphcodebert(self, code: str, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code_bytes = code.encode('utf-8')
        dfg_nodes, dfg_to_code_chars, dfg_to_dfg = self.analyzer.extract_dataflow(code_bytes)

        args_code_length = 384
        args_dfg_length = self.max_seq_len - args_code_length

        dfg_nodes = dfg_nodes[:args_dfg_length]
        dfg_to_code_chars = dfg_to_code_chars[:args_dfg_length]
        dfg_to_dfg = [[e for e in edges if e < args_dfg_length] for edges in dfg_to_dfg[:args_dfg_length]]

        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("[!] 必须加载 Fast 版本的 Tokenizer！")

        encoded = tokenizer(
            code,
            truncation=True,
            max_length=args_code_length,
            return_offsets_mapping=True
        )
        text_ids = encoded['input_ids']
        offsets = encoded['offset_mapping']
        text_len = len(text_ids)

        dfg_to_subwords = []
        for (start_char, end_char) in dfg_to_code_chars:
            subword_indices = []
            for idx, (o_start, o_end) in enumerate(offsets):
                if o_start == o_end: continue
                if o_start < end_char and o_end > start_char:
                    subword_indices.append(idx)
            dfg_to_subwords.append(subword_indices)

        input_ids = text_ids + [tokenizer.unk_token_id] * len(dfg_nodes)
        position_ids = [i + tokenizer.pad_token_id + 1 for i in range(text_len)] + [0] * len(dfg_nodes)

        pad_len = self.max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        position_ids += [tokenizer.pad_token_id] * pad_len

        attn_mask = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.bool_)
        attn_mask[:text_len, :text_len] = True

        for idx, token_id in enumerate(input_ids):
            if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                attn_mask[idx, :text_len + len(dfg_nodes)] = True
                attn_mask[:text_len + len(dfg_nodes), idx] = True

        for dfg_idx, subword_idxs in enumerate(dfg_to_subwords):
            matrix_dfg_idx = text_len + dfg_idx
            for sub_idx in subword_idxs:
                attn_mask[matrix_dfg_idx, sub_idx] = True
                attn_mask[sub_idx, matrix_dfg_idx] = True

        for dfg_idx, edges in enumerate(dfg_to_dfg):
            matrix_dfg_idx = text_len + dfg_idx
            for source_dfg_idx in edges:
                matrix_source_idx = text_len + source_dfg_idx
                attn_mask[matrix_dfg_idx, matrix_source_idx] = True
                attn_mask[matrix_source_idx, matrix_dfg_idx] = True

        float_mask = np.where(attn_mask, 0.0, -10000.0).astype(np.float32)
        float_mask_4d = np.expand_dims(float_mask, axis=(0, 1))

        return (
            torch.tensor([input_ids], dtype=torch.long),
            torch.tensor(float_mask_4d, dtype=torch.float32),
            torch.tensor([position_ids], dtype=torch.long)
        )

    def _encode_unixcoder(self, code: str, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = tokenizer.tokenize(code)
        tokens = tokens[:self.max_seq_len - 4]

        mode_token = "<encoder-only>"
        source_tokens = [tokenizer.bos_token, mode_token, tokenizer.eos_token] + tokens + [tokenizer.eos_token]
        input_ids = tokenizer.convert_tokens_to_ids(source_tokens)

        padding_length = self.max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask = [1] * (self.max_seq_len - padding_length) + [0] * padding_length

        return (
            torch.tensor([input_ids], dtype=torch.long),
            torch.tensor([attention_mask], dtype=torch.long)
        )

    # =========================================================================
    # 推断接口动态路由 (Prediction Dispatcher)
    # =========================================================================

    def predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
        m = self.models.get(target_model)
        if m is None:
            # 返回空概率向量和未知分类
            return [0.0] * self.num_classes, -1

        tokenizer = m["tokenizer"]
        model = m["model"]
        model_name_lower = target_model.lower()

        with torch.no_grad():
            if "graphcodebert" in model_name_lower:
                input_ids, attn_mask, position_ids = self._encode_graphcodebert(code, tokenizer)
                outputs = model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn_mask.to(self.device),
                    position_ids=position_ids.to(self.device)
                )

            elif "unixcoder" in model_name_lower:
                input_ids, attn_mask = self._encode_unixcoder(code, tokenizer)
                outputs = model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attn_mask.to(self.device)
                )

            else:
                inputs = tokenizer(
                    code, return_tensors="pt", truncation=True, max_length=self.max_seq_len, padding="max_length"
                ).to(self.device)
                outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy().tolist()
            pred_label = int(np.argmax(probs))

        return probs, pred_label

    def batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32) -> Tuple[
        List[List[float]], List[int]]:
        """安全的 Batch Predict: 兼容各种非标准架构的特征重组"""
        m = self.models.get(target_model)
        if m is None:
            return [[0.0] * self.num_classes] * len(codes), [-1] * len(codes)

        tokenizer = m["tokenizer"]
        model = m["model"]
        model_name_lower = target_model.lower()

        all_probs, all_preds = [], []

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]

            with torch.no_grad():
                if "graphcodebert" in model_name_lower:
                    b_input_ids, b_attn_mask, b_position_ids = [], [], []
                    for c in batch_codes:
                        i_ids, a_mask, p_ids = self._encode_graphcodebert(c, tokenizer)
                        b_input_ids.append(i_ids)
                        b_attn_mask.append(a_mask)
                        b_position_ids.append(p_ids)

                    outputs = model(
                        input_ids=torch.cat(b_input_ids, dim=0).to(self.device),
                        attention_mask=torch.cat(b_attn_mask, dim=0).to(self.device),
                        position_ids=torch.cat(b_position_ids, dim=0).to(self.device)
                    )

                elif "unixcoder" in model_name_lower:
                    b_input_ids, b_attn_mask = [], []
                    for c in batch_codes:
                        i_ids, a_mask = self._encode_unixcoder(c, tokenizer)
                        b_input_ids.append(i_ids)
                        b_attn_mask.append(a_mask)

                    outputs = model(
                        input_ids=torch.cat(b_input_ids, dim=0).to(self.device),
                        attention_mask=torch.cat(b_attn_mask, dim=0).to(self.device)
                    )

                else:
                    inputs = tokenizer(
                        batch_codes, return_tensors="pt", truncation=True, max_length=self.max_seq_len,
                        padding="max_length"
                    ).to(self.device)
                    outputs = model(**inputs)

                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                probs_list = probs.tolist() if probs.ndim == 2 else [probs.tolist()]
                preds_list = [int(np.argmax(p)) for p in probs_list]

                all_probs.extend(probs_list)
                all_preds.extend(preds_list)

        return all_probs, all_preds

    def predict_label_conf(self, code: str, label: int, target_model: str) -> float:
        probs, _ = self.predict(code, target_model)
        return probs[label] if label < len(probs) else 0.0