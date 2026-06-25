import os
import pandas as pd
from typing import List, Dict, Optional
from sklearn.preprocessing import LabelEncoder

class DatasetLoader:
    def __init__(self):
        """Initializes the dataset loader for Authorship Attribution."""
        self.label_encoder = LabelEncoder()
        self.label_map = {}  # Format: {0: "alexamici", 1: "nwin", ...}

    def load_classes_txt(self, txt_path: str):
        """Parses the classes.txt file to build the label_map."""
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Classes text file not found: {txt_path}")

        print(f"[*] Loading author classes from {txt_path}...")
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    idx, author = parts
                    self.label_map[int(idx)] = author
        print(f"[*] Successfully loaded {len(self.label_map)} authors.")

    def load_jsonl_dataset(self, filepath: str, max_samples: int = None,
                           random_seed: int = 50, classes_txt_path: Optional[str] = None) -> List[Dict]:
        """Loads code authorship data directly from a .jsonl file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        print(f"\n[*] Loading authorship dataset from JSONL file: {filepath}...")

        # 1. 使用 Pandas 高效读取 JSONL 文件
        try:
            df = pd.read_json(filepath, lines=True)
        except ValueError as e:
            raise ValueError(f"Failed to parse JSONL file. Ensure it is valid JSONL format. Error: {e}")

        if df.empty:
            raise ValueError("The JSONL file is empty.")

        # 2. 统一列名 (兼容 code/func 和 label/author 键名)
        if 'text' in df.columns and 'code' not in df.columns:  # 新增
            df.rename(columns={'text': 'code'}, inplace=True)  # 新增
        if 'name' in df.columns and 'label' not in df.columns:  # 新增
            df.rename(columns={'name': 'label'}, inplace=True)  # 新增

        # 2. 统一列名 (兼容 code/func 和 label/author 键名)
        if 'func' in df.columns and 'code' not in df.columns:
            df.rename(columns={'func': 'code'}, inplace=True)
        if 'author' in df.columns and 'label' not in df.columns:
            df.rename(columns={'author': 'label'}, inplace=True)

        if 'code' not in df.columns or 'label' not in df.columns:
            raise ValueError("JSONL missing required keys: must contain ('code' or 'func') and ('label' or 'author').")

        # 3. 清理空行或过短的代码
        def _line_count(s):
            return len([l for l in str(s).splitlines() if l.strip()])

        initial_count = len(df)
        df = df[df["code"].apply(_line_count) > 1].copy()

        print(f"[*] Data cleaning: Filtered {initial_count - len(df)} single-line or empty codes, {len(df)} valid samples remaining.")

        # 4. 加载并映射 Label
        if classes_txt_path:
            self.load_classes_txt(classes_txt_path)
            # 如果数据集里的 label 是作者名字字符串，需要转换为 ID
            if pd.api.types.is_string_dtype(df['label']):
                author_to_id = {v: k for k, v in self.label_map.items()}
                df['label'] = df['label'].map(author_to_id)

                # 过滤掉不在 classes.txt 中的未知作者
                unknown_count = df['label'].isna().sum()
                if unknown_count > 0:
                    print(f"[!] Dropping {unknown_count} samples due to authors not found in classes.txt")
                df = df.dropna(subset=['label'])
                df['label'] = df['label'].astype(int)

        else:
            # 如果没有提供 classes.txt，并且数据集是字符串作者名，自动生成映射
            if pd.api.types.is_string_dtype(df['label']):
                print("[*] Automatically generating author label map...")

                # ---> 在这里加上这行代码 <---
                df['label'] = df['label'].astype(int)  # 强制转为数字，避免字典序Bug

                df['label'] = self.label_encoder.fit_transform(df['label'])
                self.label_map = {int(i): cls for i, cls in enumerate(self.label_encoder.classes_)}

        # 5. 采样处理
        if max_samples and max_samples < len(df):
            print(f"[*] Randomly sampling {max_samples} from {len(df)} samples (seed={random_seed})...")
            df = df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)

        # 6. 直接利用 Pandas 将 DataFrame 转换为 List[Dict] 格式返回
        processed_data = df[['code', 'label']].to_dict('records')

        print(f"[*] Successfully processed {len(processed_data)} samples for Authorship Attribution.")
        return processed_data

    def get_label_map(self) -> Dict:
        """Returns the dictionary mapping integer IDs to author names."""
        return self.label_map