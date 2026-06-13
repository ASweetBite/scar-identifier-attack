import logging
import random
from typing import List, Dict

from utils.ast_tools import IdentifierAnalyzer, CodeTransformer

logger = logging.getLogger(__name__)

class ModelZooQueryTracker:
    """
    黑盒查询拦截器：利用代理模式透明地包装 ModelZoo。
    严格记录针对特定大模型的所有预测查询开销（包含单步预测和批处理预测）。
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
        # 将其他所有未重写的方法/属性（如 model_names）透明转发给底层的 model_zoo
        return getattr(self._model_zoo, name)

class CodeSmoother:
    def __init__(self, config: Dict, candidate_generator):
        """Initializes the smoother with Monte Carlo sampling parameters."""
        self.num_samples = config.get("num_samples", 50)
        self.variance_threshold = config.get("variance_threshold", 0.05)
        self.replace_prob = config.get("replace_prob", 0.5)
        self.batch_size = config.get("batch_size", 32)
        self.candidate_generator = candidate_generator
        self.analyzer = IdentifierAnalyzer()

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


import os
import json
import torch
import numpy as np
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

class ModelZoo:
    def __init__(self, model_configs: dict, eval_mode: str, config: dict):
        glob_cfg = config.get('global', {})
        run_cfg = config.get('run_params', {})

        self.device = torch.device(glob_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.eval_mode = eval_mode
        self.max_seq_len = run_cfg.get('max_seq_len', 512)

        if self.eval_mode == "binary":
            self.num_classes = 2
            print("[*] ModelZoo running in BINARY mode (Forcing num_classes = 2)")
        else:
            self.num_classes = run_cfg.get('num_classes', 16)
            print(f"[*] ModelZoo running in MULTI mode (num_classes = {self.num_classes})")

        self.models = {}
        self.model_names = list(model_configs.keys())

        # =====================================================================
        # 1. 动态加载 DFG 特征提取器
        # =====================================================================
        self.analyzer = None
        if any("graphcodebert" in name.lower() for name in self.model_names):
            print("[*] Detected GraphCodeBERT in targets. Initializing DFG Extractor...")
            self.analyzer = IdentifierAnalyzer(lang="cpp")

        for name, path in model_configs.items():
            print(f"\n[*] Loading Model[{name}] from {path}...")

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[!] CRITICAL: Target path {path} not found for model '{name}'. Aborting init.")

            try:
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                adapter_config_path = os.path.join(path, "adapter_config.json")

                if os.path.exists(adapter_config_path):
                    print(f"    |- Detected LoRA adapter. Parsing base model config...")
                    with open(adapter_config_path, 'r', encoding='utf-8') as f:
                        peft_config = json.load(f)

                    base_model_name = peft_config.get("base_model_name_or_path")
                    if not base_model_name:
                        raise ValueError(f"Missing 'base_model_name_or_path' in {adapter_config_path}")

                    print(f"    |- Loading base model skeleton from: {base_model_name}")
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_name,
                        num_labels=self.num_classes,
                        trust_remote_code=True
                    )

                    print(f"    |- Mounting LoRA weights and merging...")
                    model = PeftModel.from_pretrained(base_model, path)
                    model = model.merge_and_unload()
                else:
                    print(f"    |- Loading standard HF classifier skeleton...")
                    # 先加载骨架 (忽略默认抛出的前缀不匹配警告)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        path,
                        num_labels=self.num_classes,
                        trust_remote_code=True
                    )

                    # =========================================================
                    # 🌟 核心恢复：底层权重拦截、去前缀与 1D -> 2D 维度对齐
                    # =========================================================
                    weight_path = os.path.join(path, "pytorch_model.bin")
                    if not os.path.exists(weight_path):
                        weight_path = os.path.join(path, "model.bin")

                    if os.path.exists(weight_path):
                        print(f"    |- Intercepting raw weights from {os.path.basename(weight_path)}...")
                        raw_state_dict = torch.load(weight_path, map_location="cpu")

                        clean_state_dict = {}
                        has_custom_prefix = False

                        # A. 动态清洗 'encoder.' 前缀
                        for key, value in raw_state_dict.items():
                            if key.startswith("encoder."):
                                has_custom_prefix = True
                                clean_key = key[8:]  # 剥离 'encoder.'
                            else:
                                clean_key = key
                            clean_state_dict[clean_key] = value

                        if has_custom_prefix:
                            print("    |- 🛡️ Detected 'encoder.' prefix. Keys cleaned.")

                        # B. Math Magic: 1D Sigmoid 转 2D Softmax
                        if self.num_classes == 2 and "classifier.out_proj.weight" in clean_state_dict:
                            w = clean_state_dict["classifier.out_proj.weight"]
                            b = clean_state_dict.get("classifier.out_proj.bias")

                            if w.shape[0] == 1:
                                print("    |- 🔧 Math Magic: Converting 1D Sigmoid head to 2D Softmax head...")
                                zero_w = torch.zeros_like(w)
                                clean_state_dict["classifier.out_proj.weight"] = torch.cat([zero_w, w], dim=0)

                                if b is not None:
                                    zero_b = torch.zeros_like(b)
                                    clean_state_dict["classifier.out_proj.bias"] = torch.cat([zero_b, b], dim=0)

                        # C. 强制注入清洗后的权重
                        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

                        # 验证注入结果：必须过滤掉分类头正常初始化的少量差异
                        critical_missing = [k for k in missing if "classifier" not in k]
                        if len(critical_missing) == 0:
                            print("    |- ✅ Clean weights successfully injected. Matrix perfectly aligned.")
                        else:
                            print(f"    |- [!] Warning: {len(critical_missing)} critical keys are still missing!")
                    else:
                        print("    |- [!] No standard .bin weight file found. Relying strictly on HF auto-load.")
                    # =========================================================

                model.to(self.device)
                model.eval()
                self.models[name] = {"type": "transformer", "tokenizer": tokenizer, "model": model}
                print(f"[+] Successfully loaded {name} to {self.device}")

            except Exception as e:
                raise RuntimeError(f"Failed to load model '{name}' from path '{path}'. Execution halted.") from e

    # =========================================================================
    # 特征编码分发区 (Feature Encoding Dispatchers)
    # =========================================================================

    def _encode_graphcodebert(self, code: str, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code_bytes = code.encode('utf-8')
        dfg_nodes, dfg_to_code_chars, dfg_to_dfg = self.analyzer.extract_dataflow(code_bytes)

        # ====================================================================
        # 🌟 关键对齐：严格匹配训练脚本 GraphCodeBERTVulDataset 的硬编码长度
        # ====================================================================
        args_code_length = 384
        args_dfg_length = self.max_seq_len - args_code_length  # 128

        # 1. 截断图节点 (对齐训练集)
        dfg_nodes = dfg_nodes[:args_dfg_length]
        dfg_to_code_chars = dfg_to_code_chars[:args_dfg_length]
        dfg_to_dfg = [[e for e in edges if e < args_dfg_length] for edges in dfg_to_dfg[:args_dfg_length]]

        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("[!] 必须加载 Fast 版本的 Tokenizer！")

        # 2. 文本截断必须锁定 384 (对齐训练集)
        encoded = tokenizer(
            code,
            truncation=True,
            max_length=args_code_length,
            return_offsets_mapping=True
        )
        text_ids = encoded['input_ids']
        offsets = encoded['offset_mapping']
        text_len = len(text_ids)

        # 3. 构建 Subwords 映射
        dfg_to_subwords = []
        for (start_char, end_char) in dfg_to_code_chars:
            subword_indices = []
            for idx, (o_start, o_end) in enumerate(offsets):
                if o_start == o_end: continue
                if o_start < end_char and o_end > start_char:
                    subword_indices.append(idx)
            dfg_to_subwords.append(subword_indices)

        # 4. 组装 IDs
        input_ids = text_ids + [tokenizer.unk_token_id] * len(dfg_nodes)
        position_ids = [i + tokenizer.pad_token_id + 1 for i in range(text_len)] + [0] * len(dfg_nodes)

        # 5. Padding
        pad_len = self.max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        position_ids += [tokenizer.pad_token_id] * pad_len

        # 6. 构建 Attention Mask
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

        # 7. 转换 4D Float 掩码
        float_mask = np.where(attn_mask, 0.0, -10000.0).astype(np.float32)
        float_mask_4d = np.expand_dims(float_mask, axis=(0, 1))

        return (
            torch.tensor([input_ids], dtype=torch.long),
            torch.tensor(float_mask_4d, dtype=torch.float32),
            torch.tensor([position_ids], dtype=torch.long)
        )

    def _encode_unixcoder(self, code: str, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        """为 UniXcoder 构建特征：强制注入 <encoder-only> 控制符"""
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
            return [1.0, 0.0], -1

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

            if self.eval_mode == "binary" and pred_label == 0:
                pred_label = -1

        return probs, pred_label

    def batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32) -> Tuple[
        List[List[float]], List[int]]:
        """安全的 Batch Predict: 兼容各种非标准架构的特征重组"""
        m = self.models.get(target_model)
        if m is None:
            return [[1.0, 0.0]] * len(codes), [-1] * len(codes)

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

                if self.eval_mode == "binary":
                    preds_list = [1 if p == 1 else -1 for p in preds_list]

                all_probs.extend(probs_list)
                all_preds.extend(preds_list)

        return all_probs, all_preds

    def predict_label_conf(self, code: str, label: int, target_model: str) -> float:
        probs, _ = self.predict(code, target_model)
        return probs[label] if label < len(probs) else 0.0