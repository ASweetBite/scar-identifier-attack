import logging
import os
import json
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

logger = logging.getLogger(__name__)


class ModelZooQueryTracker:
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


class ModelZoo:
    def __init__(self, model_configs: dict, eval_mode: str, config: dict):
        glob_cfg = config.get('global', {})
        run_cfg = config.get('run_params', {})

        self.device = torch.device(glob_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.eval_mode = eval_mode
        if self.eval_mode == "binary":
            self.num_classes = 2
            print("[*] ModelZoo running in BINARY mode (Forcing num_classes = 2)")
        else:
            self.num_classes = run_cfg.get('num_classes', 16)  # 从 config 读多分类的具体数量
            print(f"[*] ModelZoo running in MULTI mode (num_classes = {self.num_classes})")
        self.max_seq_len = run_cfg.get('max_seq_len', 512)

        self.models = {}
        self.model_names = list(model_configs.keys())

        for name, path in model_configs.items():
            print(f"[*] Loading Model[{name}] from {path}...")
            if not os.path.exists(path):
                print(f"[!] Path {path} not found. Skipping {name}.")
                continue

            try:
                # 尝试加载 Tokenizer，优先从当前目录加载
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

                # 核心变动：探测是否为 PEFT/LoRA 模型
                adapter_config_path = os.path.join(path, "adapter_config.json")

                if os.path.exists(adapter_config_path):
                    print(f"[*] Detected LoRA adapter. Parsing base model config...")
                    with open(adapter_config_path, 'r', encoding='utf-8') as f:
                        peft_config = json.load(f)

                    base_model_name = peft_config.get("base_model_name_or_path")
                    if not base_model_name:
                        raise ValueError(f"Missing 'base_model_name_or_path' in {adapter_config_path}")

                    print(f"[*] Loading base model skeleton from: {base_model_name}")
                    # 加载基础模型时必须显式传入 num_labels，防止使用默认的二分类输出
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_name,
                        num_labels=self.num_classes,
                        trust_remote_code=True
                    )

                    print(f"[*] Mounting LoRA weights and merging...")
                    model = PeftModel.from_pretrained(base_model, path)
                    # 推理阶段直接合并权重，显著提升前向传播速度并节省显存
                    model = model.merge_and_unload()
                else:
                    print(f"[*] Loading standard HF classifier...")
                    model = AutoModelForSequenceClassification.from_pretrained(
                        path,
                        num_labels=self.num_classes,
                        trust_remote_code=True
                    )

                model.to(self.device)
                model.eval()
                self.models[name] = {"type": "transformer", "tokenizer": tokenizer, "model": model}
                print(f"[+] Successfully loaded {name} to {self.device}")

            except Exception as e:
                print(f"[!] Error loading {name}: {e}")

    def predict(self, code: str, target_model: str) -> Tuple[List[float], int]:
        m = self.models.get(target_model)
        if m is None:
            return [1.0, 0.0], -1

        inputs = m["tokenizer"](
            code, return_tensors="pt", truncation=True, max_length=self.max_seq_len, padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = m["model"](**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy().tolist()
            pred_label = int(np.argmax(probs))

            if self.eval_mode == "binary" and pred_label == 0:
                pred_label = -1

        return probs, pred_label

    def batch_predict(self, codes: List[str], target_model: str, batch_size: int = 32) -> Tuple[
        List[List[float]], List[int]]:
        m = self.models.get(target_model)
        if m is None:
            return [[1.0, 0.0]] * len(codes), [-1] * len(codes)

        all_probs, all_preds = [], []
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]
            inputs = m["tokenizer"](
                batch_codes, return_tensors="pt", truncation=True, max_length=self.max_seq_len, padding="max_length"
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

    def predict_label_conf(self, code: str, label: int, target_model: str) -> float:
        probs, _ = self.predict(code, target_model)
        return probs[label] if label < len(probs) else 0.0