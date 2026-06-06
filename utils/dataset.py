import pandas as pd
import os
import json
from typing import List, Dict, Optional
from sklearn.preprocessing import LabelEncoder


class DatasetLoader:
    def __init__(self):
        """Initializes the dataset loader with label encoding and safe flag definitions."""
        self.label_encoder = LabelEncoder()
        self.label_map = {}
        self.mode = "binary"

        # 这里的 safe_flags 主要是为了兼容当 vul=1 但 cwe 却为空或标注不明时的容错处理
        self.safe_flags = [
            "", "none", "0", "safe", "nan", "null", "false",
            "<null>", "<na>"
        ]

    def load_parquet_dataset(self, filepath: str, mode: str = "binary", max_samples: int = None,
                             random_seed: int = 50, label_map_path: Optional[str] = None) -> List[Dict]:
        """Loads data from a Parquet file, applying sampling, label mapping, and data cleaning."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        self.mode = mode
        print(f"\n[*] Loading dataset in '{mode}' mode from {filepath}...")

        df = pd.read_parquet(filepath)

        # ✨ 修改点 1：根据 mode 动态校验列名
        if 'func' not in df.columns or 'vul' not in df.columns:
            raise ValueError("Parquet file must contain 'func' and 'vul' columns.")

        if self.mode == "multi" and 'cwe' not in df.columns:
            raise ValueError("In 'multi' mode, the Parquet file must also contain a 'cwe' column.")

        def _line_count(s):
            return len([l for l in str(s).splitlines() if l.strip()])

        initial_count = len(df)
        df = df[df["func"].apply(_line_count) > 1].copy()

        # ✨ 修改点 2：安全清洗。只有当 'cwe' 存在时才进行清洗操作
        if 'cwe' in df.columns:
            df["cwe"] = df["cwe"].fillna("").astype(str).str.strip()

        # 清洗并标准化 vul 列为整数（0 或 1）
        df["vul"] = pd.to_numeric(df["vul"], errors='coerce').fillna(0).astype(int)

        print(
            f"[*] Data cleaning: Filtered {initial_count - len(df)} single-line or empty codes, {len(df)} valid samples remaining.")

        processed_data = []

        if self.mode == "binary":
            # 二分类核心修改，完全根据 vul 字段来决定 label
            # vul == 0 -> 安全 (-1)；vul == 1 -> 漏洞 (1)
            df['label'] = df['vul'].apply(lambda x: -1 if x == 0 else 1)

            safe_df = df[df['label'] == -1]
            vuln_df = df[df['label'] == 1]

            print(f"[*] Original data distribution: Safe={len(safe_df)}, Vuln={len(vuln_df)}")

            if max_samples and max_samples < len(df):
                safe_needed = max_samples // 2
                vuln_needed = max_samples - safe_needed
            else:
                safe_needed = len(safe_df)
                vuln_needed = safe_needed

            safe_needed = min(safe_needed, len(safe_df))
            vuln_needed = min(vuln_needed, len(vuln_df), safe_needed)

            print(f"[*] Balanced sampling (1:2): Safe={safe_needed}, Vuln={vuln_needed} (Seed={random_seed})")

            safe_sampled = safe_df.sample(n=safe_needed, random_state=random_seed)
            vuln_sampled = vuln_df.sample(n=vuln_needed, random_state=random_seed)

            df = pd.concat([safe_sampled, vuln_sampled]).sample(frac=1, random_state=random_seed).reset_index(drop=True)

            self.label_map = {-1: "Safe", 1: "Vulnerable"}

        elif self.mode == "multi":
            # 多分类核心修改
            def determine_multi_label(row):
                if row['vul'] == 0:
                    return "Safe"
                else:
                    cwe_val = str(row['cwe']).strip()
                    if not cwe_val or cwe_val.lower() in getattr(self, 'safe_flags', []):
                        return "Unknown_CWE"
                    return cwe_val

            df['label_raw'] = df.apply(determine_multi_label, axis=1)

            if label_map_path and os.path.exists(label_map_path):
                print(f"[*] Loading existing label map from {label_map_path}...")
                with open(label_map_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)

                    if "id2label" in loaded_data:
                        raw_map = loaded_data["id2label"]
                    else:
                        raw_map = loaded_data

                    self.label_map = {int(k): v for k, v in raw_map.items()}

                self.label_map[-1] = "Safe"
                cwe_to_id = {v: k for k, v in self.label_map.items()}

                valid_mask = df['label_raw'].isin(cwe_to_id.keys())
                dropped_count = len(df) - valid_mask.sum()
                if dropped_count > 0:
                    print(f"[!] Warning: Dropped {dropped_count} samples due to unknown CWEs not in the label map.")
                    df = df[valid_mask].reset_index(drop=True)

                df['label'] = df['label_raw'].map(cwe_to_id)

            else:
                print("[*] Generating new label mapping from current dataset...")
                is_safe = df['label_raw'] == "Safe"
                vuln_cwes = df.loc[~is_safe, 'label_raw']

                if not vuln_cwes.empty:
                    self.label_encoder.fit(vuln_cwes)
                    self.label_map = {int(i): cls for i, cls in enumerate(self.label_encoder.classes_)}
                else:
                    self.label_map = {}

                self.label_map[-1] = "Safe"
                cwe_to_id = {v: k for k, v in self.label_map.items()}

                df['label'] = df['label_raw'].map(cwe_to_id)

                if label_map_path:
                    os.makedirs(os.path.dirname(label_map_path) or '.', exist_ok=True)
                    with open(label_map_path, 'w', encoding='utf-8') as f:
                        save_data = {
                            "id2label": self.label_map,
                            "label2id": {v: k for k, v in self.label_map.items()}
                        }
                        json.dump(save_data, f, indent=4, ensure_ascii=False)
                    print(f"[*] Saved new label map to {label_map_path}")

            if max_samples and max_samples < len(df):
                print(f"[*] Randomly sampling {max_samples} from {len(df)} multi-class samples (seed={random_seed})...")
                df = df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)
        else:
            raise ValueError("Mode must be 'binary' or 'multi'")

        # ✨ 修改点 3：在生成结果字典时使用 .get('cwe', '') 防止 KeyError
        for _, row in df.iterrows():
            processed_data.append({
                "code": row["func"],
                "label": int(row["label"]),
                "raw_cwe": row.get("cwe", ""),  # 如果没有 cwe 列，默认为空字符串
                "vul": row["vul"]
            })

        print(f"[*] Successfully processed {len(processed_data)} samples.")
        return processed_data

    def get_label_map(self) -> Dict:
        """Returns the dictionary mapping integer labels to their string classifications."""
        return self.label_map