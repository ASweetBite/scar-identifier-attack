import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import classification_report
from tqdm import tqdm
from datasets import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def evaluate_models(test_data_path, model_names):
    """加载测试数据集并评估模型，输出分类报告。"""
    print(f"[*] Loading test data from: {test_data_path}")
    df = pd.read_parquet(test_data_path)

    # --- 核心修改 1：直接使用现成的 'vul' 列作为标签 ---
    # 不需要再基于 cwe 去写 lambda 推断了，直接转换为整型即可
    df['label'] = df['vul'].astype(int)

    # --- 核心修改 2：防御性填充空值 ---
    # 确保 'func' 列全是字符串，防止分词器(tokenizer)遇到 None 报错
    df['func'] = df['func'].fillna("")

    TOTAL_SAMPLES = 2000

    safe_df = df[df['label'] == 0]
    vuln_df = df[df['label'] == 1]

    print(f"[*] Original data distribution: Safe={len(safe_df)}, Vuln={len(vuln_df)}")

    # 如果总量不够 2000，稍微做一下宽容处理，避免直接抛出错误中断程序
    if len(safe_df) + len(vuln_df) < TOTAL_SAMPLES:
        print(
            f"[!] Warning: Total data count ({len(safe_df) + len(vuln_df)}) is less than {TOTAL_SAMPLES}! Using all available data.")
        TOTAL_SAMPLES = len(safe_df) + len(vuln_df)

    target_safe = TOTAL_SAMPLES // 2
    target_vuln = TOTAL_SAMPLES - target_safe

    # 平衡采样的逻辑保持不变：尽量 1:1，不够的用另一半补齐
    if len(safe_df) < target_safe:
        target_safe = len(safe_df)
        target_vuln = TOTAL_SAMPLES - target_safe
    elif len(vuln_df) < target_vuln:
        target_vuln = len(vuln_df)
        target_safe = TOTAL_SAMPLES - target_vuln

    safe_sampled = safe_df.sample(n=target_safe, random_state=42)
    vuln_sampled = vuln_df.sample(n=target_vuln, random_state=42)

    df = pd.concat([safe_sampled, vuln_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(
        f"[*] Sampled distribution: Safe={len(df[df['label'] == 0])}, Vuln={len(df[df['label'] == 1])}, Total={len(df)}")

    test_ds = Dataset.from_pandas(df[['func', 'label']])
    y_true = df['label'].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}, Test sample size: {len(test_ds)}")

    for name in model_names:
        # 注意确认你的模型路径是否正确，如果是当前目录下的 models 文件夹
        model_path = f"./models/binary_diversevul_codebert_pure_dataset"
        if not os.path.exists(model_path):
            print(f"[!] Model {name} not found at {model_path}, skipping.")
            continue

        print(f"\n{'=' * 50}\n[*] Evaluating model: {name}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        model.eval()

        def tokenize(batch):
            """对代码片段进行分词截断。"""
            return tokenizer(batch['func'], truncation=True, max_length=512)

        tokenized_ds = test_ds.map(tokenize, batched=True)
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask"])

        dataloader = DataLoader(
            tokenized_ds,
            batch_size=32,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
            pin_memory=True
        )

        y_pred = []

        for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                y_pred.extend(preds)

        print(f"\n[+] {name} Evaluation Report:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=["Safe", "Vulnerable"],
            digits=4
        ))


if __name__ == "__main__":
    # 使用刚清洗好的 Parquet 文件跑测试
    evaluate_models("./data/test_binary.parquet", ["CodeBERT"])