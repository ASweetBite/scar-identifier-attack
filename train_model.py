import os
import gc
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import recall_score, precision_score, f1_score

# =============== 环境配置 ===============
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# =============== 数据集定义 ===============

class VulnerabilityDataset(Dataset):
    def __init__(self, tokenizer, parquet_path, max_len=512):
        self.examples = []
        logger.info(f"Loading dataset file at {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df['label'] = df['vul'].astype(int)

        # 基础截断过滤
        df = df[df['func'].str.len() <= 4000].copy()

        funcs = df['func'].tolist()
        labels = df['label'].tolist()

        for func, label in tqdm(zip(funcs, labels), total=len(funcs), desc=f"Tokenizing"):
            encoded = tokenizer(
                func,
                truncation=True,
                max_length=max_len,
                padding='max_length'
            )
            self.examples.append({
                "input_ids": encoded['input_ids'],
                "attention_mask": encoded['attention_mask'],
                "label": label
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        return (
            torch.tensor(example['input_ids'], dtype=torch.long),
            torch.tensor(example['attention_mask'], dtype=torch.long),
            torch.tensor(example['label'], dtype=torch.long)
        )


# =============== 评估与训练逻辑 ===============

def evaluate(model, eval_dataset, eval_batch_size, device):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    logits_list = []
    y_trues = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
        inputs = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            eval_loss += outputs.loss.mean().item()
            logits_list.append(outputs.logits.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

        nb_eval_steps += 1

    logits = np.concatenate(logits_list, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = np.argmax(logits, axis=-1)

    recall = recall_score(y_trues, y_preds, average='macro', zero_division=0)
    precision = precision_score(y_trues, y_preds, average='macro', zero_division=0)
    f1 = f1_score(y_trues, y_preds, average='macro', zero_division=0)

    return {
        "eval_loss": eval_loss / nb_eval_steps,
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
    }


def train(model_name, model, train_dataset, eval_dataset, cfg):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=cfg["train_batch_size"])

    t_total = len(train_dataloader) // cfg["gradient_accumulation_steps"] * cfg["num_epochs"]

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg["learning_rate"], eps=1e-8)
    warmup_steps = int(t_total * 0.1)  # 🌟 算出总步数的 10% 作为预热

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,  # 🌟 把这里的 0 换成 warmup_steps
        num_training_steps=t_total
    )
    logger.info(f"***** Running training for {model_name} *****")

    best_f1 = 0.0
    model.zero_grad()

    for epoch in range(int(cfg["num_epochs"])):
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(bar):
            model.train()
            inputs = batch[0].to(cfg["device"])
            attention_mask = batch[1].to(cfg["device"])
            labels = batch[2].to(cfg["device"])

            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if cfg["gradient_accumulation_steps"] > 1:
                loss = loss / cfg["gradient_accumulation_steps"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            bar.set_postfix({"loss": round(loss.item() * cfg["gradient_accumulation_steps"], 4)})

            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Epoch 结束，进行评估
        results = evaluate(model, eval_dataset, cfg["eval_batch_size"], cfg["device"])
        logger.info(
            f"  Epoch {epoch + 1} Results: F1: {results['eval_f1']:.4f} | Recall: {results['eval_recall']:.4f} | Prec: {results['eval_precision']:.4f}")

        if results['eval_f1'] > best_f1:
            best_f1 = results['eval_f1']
            output_dir = os.path.join(cfg["output_dir"], f"{model_name}_best_f1")
            os.makedirs(output_dir, exist_ok=True)

            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            logger.info(f"  [+] 新的最佳 F1 ({best_f1:.4f})! 模型已保存至 {output_dir}")


# =============== 主控流程 ===============

def main():
    # 📌 写死所有的超参数和数据路径，方便直接跑
    cfg = {
        "train_data_file": "data/diverse/train.parquet",  # 替换为你的真实路径
        "eval_data_file": "data/diverse/valid.parquet",  # 替换为你的真实路径
        "output_dir": "./models",
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-4,
        "num_epochs": 3,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "seed": 42
    }

    set_seed(cfg["seed"])

    # 📌 写死你要跑的三个核心模型
    models_to_train = {
        "CodeBERT": "microsoft/codebert-base"
        # "GraphCodeBERT": "microsoft/graphcodebert-base",
        # "UniXcoder": "microsoft/unixcoder-base"
    }

    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=["query", "value"],modules_to_save=["classifier"])

    for model_name, model_path in models_to_train.items():
        logger.info("\n" + "=" * 50)
        logger.info(f"🚀 初始化并开始训练: {model_name}")
        logger.info("=" * 50)

        # 1. 加载对应模型的 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, bos_token="<s>", eos_token="</s>", sep_token="</s>",
            cls_token="<s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
            additional_special_tokens=[]
        )

        # 2. 针对当前 Tokenizer 预处理数据 (因为每个模型的词表可能不同)
        train_dataset = VulnerabilityDataset(tokenizer, cfg["train_data_file"])
        eval_dataset = VulnerabilityDataset(tokenizer, cfg["eval_data_file"])

        # 3. 加载模型 & 注入 LoRA
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, trust_remote_code=True)
        model = get_peft_model(model, peft_config)
        model.to(cfg["device"])

        # 4. 执行训练
        train(model_name, model, train_dataset, eval_dataset, cfg)

        # 5. 保存 Tokenizer (因为只保存了 model 的 weight)
        tokenizer.save_pretrained(os.path.join(cfg["output_dir"], f"{model_name}_best_f1"))

        # 📌 核心防爆显存逻辑：清理当前模型的驻留内存
        del model, tokenizer, train_dataset, eval_dataset
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"✅ {model_name} 训练完毕，显存已清空。准备进入下一个模型。\n")


if __name__ == "__main__":
    main()