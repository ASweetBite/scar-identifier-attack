import os
import re
import json
import random
import logging
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# 导入你写好的 SPT 混淆模块
from test_spt import obfuscate

# =============== 环境配置 ===============
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embedding(input_ids)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        logits = self.fc(last_hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


def augment_data(df: pd.DataFrame, needed_count: int, label_value: int) -> pd.DataFrame:
    funcs = df['func'].tolist()

    # 预过滤：跳过超过 3000 字符的超大样本
    funcs =[f for f in funcs if len(f) <= 3000]
    n_samples = len(funcs)

    if n_samples == 0:
        return pd.DataFrame({'func':[], 'label':[]})

    base_aug = needed_count // n_samples
    remainder = needed_count % n_samples

    aug_counts =[base_aug] * n_samples
    for idx in random.sample(range(n_samples), remainder):
        aug_counts[idx] += 1

    augmented_funcs =[]
    label_name = "漏洞(1)" if label_value == 1 else "安全(0)"
    print(f"[*] {label_name} 样本分配策略: 平均每个样本进行 {base_aug} 到 {base_aug + 1} 次混淆...")

    pbar = tqdm(total=needed_count, desc=f"Augmenting {label_name}")

    for code, target_count in zip(funcs, aug_counts):
        successful = 0
        attempts = 0
        while successful < target_count and attempts < target_count * 3:
            attempts += 1
            try:
                new_code = obfuscate(code)
                augmented_funcs.append(new_code)
                successful += 1
                pbar.update(1)
            except Exception:
                continue

    missing = needed_count - len(augmented_funcs)
    attempts = 0
    while missing > 0 and attempts < missing * 5:
        attempts += 1
        code = random.choice(funcs)
        try:
            new_code = obfuscate(code)
            augmented_funcs.append(new_code)
            missing -= 1
            pbar.update(1)
        except Exception:
            pass

    pbar.close()
    return pd.DataFrame({'func': augmented_funcs, 'label': label_value})


# =============== 🛠️ 核心数据准备逻辑 ===============

def prepare_dataset(parquet_path):
    print(f"[*] Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    df['label'] = df['vul'].astype(int)

    # 基础清洗：长度过滤
    max_char_length = 4000
    df = df[df['func'].str.len() <= max_char_length].copy()

    df_safe = df[df['label'] == 0]
    df_vul = df[df['label'] == 1]

    real_vul_count = len(df_vul)
    real_safe_count = len(df_safe)
    print(f"[*] 真实标签分布: 安全(0)={real_safe_count}, 漏洞(1)={real_vul_count}")

    # 设定总量目标 (每类 8 万，总共 16 万)
    target_count = 18945

    # ================= 实现完美的对称增强 =================

    # 步骤 A: 漏洞样本处理
    if real_vul_count < target_count:
        needed_vul = target_count - real_vul_count
        print(f"\n[*] 漏洞样本缺口: {needed_vul}，启动漏洞样本增强...")
        aug_df_vul = augment_data(df_vul, needed_vul, label_value=1)
        sampled_vul = pd.concat([df_vul, aug_df_vul], ignore_index=True)
        # 记录原生的数量，用于安全样本对齐
        vul_original_used = real_vul_count
        vul_aug_used = needed_vul
    else:
        sampled_vul = df_vul.sample(n=target_count, random_state=42)
        vul_original_used = target_count
        vul_aug_used = 0

    # 步骤 B: 安全样本处理 (强制使其结构与漏洞样本一模一样)
    print(f"\n[*] 为了防止捷径学习，安全样本将强制采用相同的 [原生:增强] 比例：{vul_original_used} : {vul_aug_used}")

    base_safe = df_safe.sample(n=vul_original_used, random_state=42)

    if vul_aug_used > 0:
        remaining_safe = df_safe.drop(base_safe.index)
        seed_safe = remaining_safe.sample(n=min(len(remaining_safe), 30000), random_state=42)
        print(f"[*] 启动安全样本增强程序...")
        aug_df_safe = augment_data(seed_safe, vul_aug_used, label_value=0)

        sampled_safe = pd.concat([base_safe, aug_df_safe], ignore_index=True)
    else:
        sampled_safe = base_safe

    # 最终合并并彻底打乱顺序
    df_final = pd.concat([sampled_safe, sampled_vul]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n[+] 训练集构建完成 (完美对称):")
    print(f"    - 安全样本(0): {vul_original_used} 原生 + {vul_aug_used} 混淆 = {len(sampled_safe)}")
    print(f"    - 漏洞样本(1): {vul_original_used} 原生 + {vul_aug_used} 混淆 = {len(sampled_vul)}")
    print(f"    - 总计: {len(df_final)}")

    return Dataset.from_pandas(df_final[['func', 'label']])


# =============== 训练执行 ===============

def train_models(dataset):
    # 📌 此处将 CodeT5 替换为了 UniXcoder
    models_to_train = {
        "GraphCodeBERT": {"path": "microsoft/graphcodebert-base", "type": "transformer"},
        "UniXcoder": {"path": "microsoft/unixcoder-base", "type": "transformer"},
    }

    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)

    for name, info in models_to_train.items():
        save_path = f"./models/binary_diversevul_{name.lower()}"
        if os.path.exists(save_path):
            print(f"[*] Model {name} already exists, skipping.")
            continue

        print(f"\n🚀 Preparing model: {name}")

        # 📌 核心修复：手动传入所有特殊的 token，强行用纯字符串覆盖掉云端 json 中引发类型异常的旧字典
        # 从而完美避开 Rust tokenizers 的列表类型检测错误！
        tokenizer = AutoTokenizer.from_pretrained(
            info["path"],
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            additional_special_tokens=[]
        )

        if info["type"] == "transformer":
            model = AutoModelForSequenceClassification.from_pretrained(
                info["path"],
                num_labels=2,
                trust_remote_code=True
            )
            model = get_peft_model(model, peft_config)
        else:
            model = BiLSTMClassifier(vocab_size=len(tokenizer), num_labels=2)

        def tokenize_func(examples):
            return tokenizer(examples["func"], truncation=True, max_length=512)

        tokenized_ds = dataset.map(tokenize_func, batched=True)

        if info["type"] == "transformer":
            tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        else:
            tokenized_ds.set_format("torch", columns=["input_ids", "label"])

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"./temp_{name}",
                per_device_train_batch_size=16,
                num_train_epochs=3,
                learning_rate=3e-4,
                save_strategy="epoch",
                report_to="none",
                fp16=torch.cuda.is_available()
            ),
            train_dataset=tokenized_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer) if info["type"] == "transformer" else None,
        )

        trainer.train()

        if info["type"] == "transformer":
            model.merge_and_unload().save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
            tokenizer.save_pretrained(save_path)

        print(f"[+] {name} training completed and saved to: {save_path}")


if __name__ == "__main__":
    if os.path.exists("data/diverse_vul.parquet"):
        ds = prepare_dataset("data/diverse_vul.parquet")
        train_models(ds)
    else:
        print("Dataset not found.")