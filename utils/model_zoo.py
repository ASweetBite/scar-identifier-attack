import logging
import os
import random
from collections import Counter
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.ast_tools import IdentifierAnalyzer, CodeTransformer
from utils.bert_loader import CodeBERTModelLoader

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


class ModelZoo:
    def __init__(self, model_configs: dict, eval_mode: str, config: dict, smoother=None):
        """Initializes the ModelZoo by batch loading target models and injecting the smoother."""
        glob_cfg = config.get('global', {})
        run_cfg = config.get('run_params', {})

        self.device = torch.device(glob_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.eval_mode = eval_mode
        self.num_classes = run_cfg.get('num_classes', 16)
        self.max_seq_len = run_cfg.get('max_seq_len', 512)
        self.use_majority_voting = run_cfg.get('use_majority_voting', False)

        self.models = {}
        self.model_names = list(model_configs.keys())
        self.smoother = smoother

        for name, path in model_configs.items():
            print(f"[*] Loading Model[{name}] from {path}...")
            if not os.path.exists(path):
                print(f"[!] Path {path} not found. Skipping {name}.")
                continue

            try:
                if os.path.exists(os.path.join(path, "dual_heads.pt")):
                    print(f"[*] Dual-head model detected, initializing loader...")
                    loader_cfg = {
                        "model": {
                            "model_name": path,
                            "max_seq_len": self.max_seq_len,
                            "device": str(self.device)
                        },
                        "data": {"num_classes": self.num_classes}
                    }
                    loader = CodeBERTModelLoader(loader_cfg)
                    model_obj, _ = loader.load_model()
                    self.models[name] = {"type": "dual_head", "model_obj": model_obj}
                else:
                    print(f"[*] Loading standard HF classifier from {path}...")
                    # 1. 直接拉取标准 tokenizer，保证绝对不出错
                    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", trust_remote_code=True)

                    # 2. 找到实际的权重文件
                    bin_path = os.path.join(path, "pytorch_model.bin")
                    if not os.path.exists(bin_path):
                        bin_path = os.path.join(path, "model.bin")

                    if not os.path.exists(bin_path):
                        raise FileNotFoundError(f"[!] Could not find any .bin file in {path}")

                    # 3. 强行读取字典到内存
                    print(f"[*] Reading raw state_dict from {bin_path}...")
                    raw_state_dict = torch.load(bin_path, map_location=self.device)

                    # 4. 🌟 绝对执行清洗：脱去 'encoder.' 帽子
                    clean_state_dict = {}
                    has_custom_prefix = False
                    for key, value in raw_state_dict.items():
                        if key.startswith("encoder."):
                            has_custom_prefix = True
                            clean_key = key[8:]  # 去掉 'encoder.'
                        else:
                            clean_key = key
                        clean_state_dict[clean_key] = value

                    if has_custom_prefix:
                        print("[*] 🛡️ Detected 'encoder.' custom prefix! Keys have been successfully cleaned.")

                    # =========================================================
                    # 🌟 4.5 核心魔法：将 1D Sigmoid 分类头动态等价转换为 2D Softmax
                    # =========================================================
                    num_labels = 2 if self.eval_mode == "binary" else self.num_classes
                    if num_labels == 2 and "classifier.out_proj.weight" in clean_state_dict:
                        w = clean_state_dict["classifier.out_proj.weight"]
                        b = clean_state_dict.get("classifier.out_proj.bias")

                        if w.shape[0] == 1:
                            print("[*] 🔧 Math Magic: Converting 1D Sigmoid head to 2D Softmax head...")
                            # 构造 [0, 原权重] 以匹配 Softmax 逻辑
                            zero_w = torch.zeros_like(w)
                            clean_state_dict["classifier.out_proj.weight"] = torch.cat([zero_w, w], dim=0)

                            if b is not None:
                                zero_b = torch.zeros_like(b)
                                clean_state_dict["classifier.out_proj.bias"] = torch.cat([zero_b, b], dim=0)
                    # =========================================================

                    # 5. 使用微软官方架构作为骨架
                    model = AutoModelForSequenceClassification.from_pretrained(
                        "microsoft/codebert-base",
                        num_labels=num_labels,
                        ignore_mismatched_sizes=True,
                        trust_remote_code=True
                    )

                    # 5.2 原生 PyTorch 强行注入清洗并转换后的权重！
                    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

                    if len(missing) > 0 or len(unexpected) > 0:
                        print(f"[*] Partial match. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                    else:
                        print("[*] ✅ Perfect Match! All custom weights mathematically aligned to HF standard!")

                    model = model.to(self.device)
                    model.eval()
                    self.models[name] = {"type": "standard", "tokenizer": tokenizer, "model": model}
                    print(f"[*] ✅ Successfully loaded {name} into ModelZoo!")

            except Exception as e:
                print(f"[!] Error completely failed to load {name}: {e}")

    def _base_predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
        """Performs raw single-inference logic without smoothing or voting."""
        m = self.models.get(target_model)
        if m is None: return [1.0, 0.0], -1

        if m["type"] == "dual_head":
            res = m["model_obj"].predict(code)
            p_vuln = float(res["f_det"])
            if p_vuln <= 0.5:
                return [1.0 - p_vuln, p_vuln], -1
            else:
                if self.eval_mode == "binary":
                    return [1.0 - p_vuln, p_vuln], 1
                else:
                    probs = res["f_cls"]
                    return probs.tolist(), int(np.argmax(probs))
        else:
            inputs = m["tokenizer"](
                code, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
            ).to(self.device)

            with torch.no_grad():
                outputs = m["model"](**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy().tolist()
                pred_label = int(np.argmax(probs))

                if self.eval_mode == "binary" and pred_label == 0:
                    pred_label = -1

            return probs, pred_label

    def _base_batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32) -> Tuple[
        List[List[float]], List[int]]:
        m = self.models.get(target_model)
        if m is None: return [[1.0, 0.0]] * len(codes), [-1] * len(codes)

        if m["type"] == "dual_head":
            res = m["model_obj"].batch_predict(codes, batch_size=batch_size)
            f_det = res["f_det"]  # 假设是 (B,) 的漏洞概率

            final_probs = []
            final_preds = []

            if self.eval_mode == "binary":
                for p in f_det:
                    p = float(p)
                    final_probs.append([1.0 - p, p])
                    final_preds.append(1 if p > 0.5 else -1)  # 🌟 修正为 -1
            else:
                f_cls = res["f_cls"]  # (B, num_classes)
                for p_det, p_cls in zip(f_det, f_cls):
                    p_det = float(p_det)
                    final_probs.append(p_cls.tolist())
                    # 🌟 只有检测头过关，才返回 CWE ID，否则返回 -1
                    final_preds.append(int(np.argmax(p_cls)) if p_det > 0.5 else -1)

            return final_probs, final_preds
        else:
            all_probs, all_preds = [], []
            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i:i + batch_size]
                inputs = m["tokenizer"](
                    batch_codes, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
                ).to(self.device)

                with torch.no_grad():
                    outputs = m["model"](**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                    probs = [probs.tolist()] if probs.ndim == 1 else probs.tolist()
                    preds = [int(np.argmax(p)) for p in probs]

                    if self.eval_mode == "binary":
                        preds = [1 if p == 1 else -1 for p in preds]

                    all_probs.extend(probs)
                    all_preds.extend(preds)
            return all_probs, all_preds

    def predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
        """Predicts the label for a single code snippet.
        Applies Monte Carlo smoothing and majority voting.
        If variance is high (adversarial attack detected), it silently canonicalizes the code
        and forces a standard definitive prediction.
        """
        if self.use_majority_voting and self.smoother:
            samples = self.smoother.generate_smoothed_samples(code)
            probs_list, preds_list = self._base_batch_predict(samples, target_model,
                                                              batch_size=self.smoother.batch_size)

            majority_class = Counter(preds_list).most_common(1)[0][0]

            majority_probs = [probs[majority_class] for probs in probs_list]
            variance = float(np.var(majority_probs, ddof=1)) if len(samples) > 1 else 0.0

            if variance > self.smoother.variance_threshold:
                # 触发防御：静默去噪并强制输出标准结果
                canonical_code = self.smoother.analyzer.canonicalize(code) if hasattr(self.smoother.analyzer,
                                                                                      'canonicalize') else code
                fallback_probs, fallback_pred = self._base_predict(canonical_code, target_model)
                # 直接返回整数标签，对外隐藏防御动作
                return fallback_probs, fallback_pred

            avg_probs = np.mean(probs_list, axis=0).tolist()
            return avg_probs, majority_class

        return self._base_predict(code, target_model)

    def batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32) -> Tuple[
        List[List[float]], List[int]]:
        """Predicts labels for a batch of code snippets using efficient flattened inference.
        Ensures the output is always a valid list of probabilities and integer labels,
        silently falling back to canonicalized code for high-variance samples.
        """
        if self.use_majority_voting and self.smoother:
            all_samples = []
            for code in codes:
                all_samples.extend(self.smoother.generate_smoothed_samples(code))

            all_probs, all_preds = self._base_batch_predict(all_samples, target_model, batch_size=batch_size)

            final_probs = [[] for _ in range(len(codes))]
            final_preds = [0 for _ in range(len(codes))]
            num_samples = self.smoother.num_samples

            fallback_indices = []
            fallback_codes = []

            for i in range(len(codes)):
                start_idx = i * num_samples
                end_idx = start_idx + num_samples

                group_preds = all_preds[start_idx:end_idx]
                group_probs = all_probs[start_idx:end_idx]

                majority_class = Counter(group_preds).most_common(1)[0][0]

                majority_probs = [probs[majority_class] for probs in group_probs]
                variance = float(np.var(majority_probs, ddof=1)) if num_samples > 1 else 0.0

                if variance > self.smoother.variance_threshold:
                    # 记录需要兜底的索引，准备二次预测
                    fallback_indices.append(i)
                    canonical_code = self.smoother.analyzer.canonicalize(codes[i]) if hasattr(self.smoother.analyzer,
                                                                                              'canonicalize') else \
                    codes[i]
                    fallback_codes.append(canonical_code)
                else:
                    final_preds[i] = majority_class
                    final_probs[i] = np.mean(group_probs, axis=0).tolist()

            # 对所有高方差样本进行批量的去噪预测
            if fallback_codes:
                fb_probs, fb_preds = self._base_batch_predict(fallback_codes, target_model, batch_size=batch_size)

                # 将去噪后的结果以标准格式（float和int）回填
                for idx, prob, pred in zip(fallback_indices, fb_probs, fb_preds):
                    final_preds[idx] = pred
                    final_probs[idx] = prob

            return final_probs, final_preds

        return self._base_batch_predict(codes, target_model, batch_size)
    # def predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
    #     """
    #     [纯标准化测试版]
    #     仅对输入代码进行 AST 规范化去噪，然后进行单次预测。
    #     用于测试模型在失去所有变量语义情况下的基础 F1 得分和原生防御力。
    #     """
    #     # 1. 强制进行代码规范化（抹除所有变量名/函数名）
    #     canonical_code = self.smoother.analyzer.canonicalize(code) if self.smoother and hasattr(self.smoother.analyzer,
    #                                                                                             'canonicalize') else code
    #
    #     # 2. 直接使用规范化后的代码进行单次预测（无平滑开销）
    #     probs, pred = self._base_predict(canonical_code, target_model)
    #
    #     return probs, pred
    #
    # def batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32) -> Tuple[
    #     List[List[float]], List[int]]:
    #     """
    #     [纯标准化测试版]
    #     对批量输入的代码进行 AST 规范化去噪，然后进行高效扁平化推理。
    #     """
    #     if self.smoother and hasattr(self.smoother.analyzer, 'canonicalize'):
    #         # 在 CPU 上极速批量处理规范化
    #         canonical_codes = [self.smoother.analyzer.canonicalize(code) for code in codes]
    #     else:
    #         canonical_codes = codes
    #
    #     # 直接将全部规范化代码送入 GPU 进行批处理推理
    #     return self._base_batch_predict(canonical_codes, target_model, batch_size=batch_size)

    def predict_label_conf(self, code: str, label: int, target_model: str) -> float:
        """Retrieves confidence for a specific label, inheriting the current prediction logic."""
        probs, _ = self.predict(code, target_model)
        return probs[label] if label < len(probs) else 0.0

    def predict_with_rejection(self, code: str, target_model: str, candidate_dict: dict = None,
                               sensitive_vars: list = None) -> Tuple[Union[int, str], float, float]:
        """Predicts a label with a rejection mechanism for high-variance adversarial samples."""
        if not self.smoother:
            raise ValueError("[!] Smoother not initialized. Provide smoother_config and generator in ModelZoo.")

        m = self.models.get(target_model)
        if m is None:
            return 0, 0.0, 0.0

        samples = self.smoother.generate_smoothed_samples(code, candidate_dict, sensitive_vars)
        N = len(samples)

        if m["type"] == "dual_head":
            res = m["model_obj"].batch_predict(samples, batch_size=self.smoother.batch_size)
            f_det, f_cls = res["f_det"], res["f_cls"]
            raw_probs = np.zeros((N, f_cls.shape[1] + 1))
            raw_probs[:, 0] = 1.0 - f_det
            raw_probs[:, 1:] = f_det[:, np.newaxis] * f_cls
        else:
            batch_probs, _ = self._base_batch_predict(samples, target_model, batch_size=self.smoother.batch_size)
            raw_probs = np.array(batch_probs)

        predictions = np.argmax(raw_probs, axis=1)
        majority_class, count = Counter(predictions.tolist()).most_common(1)[0]
        confidence = count / N
        variance = float(np.var(raw_probs[:, majority_class], ddof=1)) if N > 1 else 0.0

        if variance > self.smoother.variance_threshold:
            return "Reject_Adversarial", confidence, variance

        if self.eval_mode == "binary" and majority_class > 0:
            majority_class = 1

        return int(majority_class), confidence, variance