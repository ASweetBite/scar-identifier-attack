import os
import gc
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
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

from utils.ast_tools import IdentifierAnalyzer

# =============== Environment Configuration ===============
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


# =============== Dataset Definition (GraphCodeBERT 专属核心) ===============

class GraphCodeBERTVulDataset(Dataset):
    def __init__(self, tokenizer, parquet_path, max_len=512, lang="c"):
        self.examples = []
        self.max_len = max_len
        logger.info(f"Loading dataset file at {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df['label'] = df['vul'].astype(int)

        # 基础截断过滤
        df = df[df['func'].str.len() <= 4000].copy()

        funcs = df['func'].tolist()
        labels = df['label'].tolist()

        # 初始化 AST / DFG 提取器
        logger.info("Initializing IdentifierAnalyzer for DFG extraction...")
        self.analyzer = IdentifierAnalyzer(lang=lang)

        for func, label in tqdm(zip(funcs, labels), total=len(funcs), desc="Extracting DFG & Tokenizing"):
            try:
                # 1. 提取 DFG 特征
                code_bytes = func.encode('utf-8')
                dfg_nodes, dfg_to_code, dfg_to_dfg = self.analyzer.extract_dataflow(code_bytes)

                # 2. Tokenize 文本并预留空间
                text_tokens = tokenizer.tokenize(func)
                text_tokens = text_tokens[:max_len - 2 - len(dfg_nodes)]

                total_tokens = [tokenizer.cls_token] + text_tokens + [tokenizer.sep_token] + dfg_nodes
                input_ids = tokenizer.convert_tokens_to_ids(total_tokens)

                # 3. 构造位置编码 (Position IDs)
                text_len = len(text_tokens) + 2
                position_ids = [i + tokenizer.pad_token_id + 1 for i in range(text_len)]
                position_ids += [0 for _ in dfg_nodes]

                # 4. 构造 2D 注意力掩码
                seq_length = len(total_tokens)
                attn_mask = np.zeros((seq_length, seq_length), dtype=bool)

                # a. 文本互相可见
                attn_mask[:text_len, :text_len] = True

                # b. DFG 节点连通性
                for idx, edges in enumerate(dfg_to_dfg):
                    node_idx_in_matrix = text_len + idx
                    for source_idx in edges:
                        if source_idx < len(dfg_nodes):
                            source_idx_in_matrix = text_len + source_idx
                            attn_mask[node_idx_in_matrix, source_idx_in_matrix] = True
                            attn_mask[source_idx_in_matrix, node_idx_in_matrix] = True

                # 5. Padding 补齐
                pad_len = max_len - seq_length
                if pad_len > 0:
                    input_ids += [tokenizer.pad_token_id] * pad_len
                    position_ids += [tokenizer.pad_token_id] * pad_len

                    padded_attn_mask = np.zeros((max_len, max_len), dtype=bool)
                    padded_attn_mask[:seq_length, :seq_length] = attn_mask
                else:
                    input_ids = input_ids[:max_len]
                    position_ids = position_ids[:max_len]
                    padded_attn_mask = attn_mask[:max_len, :max_len]

                # 6. 🌟 核心修复：转换为 3D 浮点掩码 (1, max_len, max_len)
                # DataLoader 会自动将 batch 个 (1, L, L) 堆叠成 (B, 1, L, L)，完美契合 Hugging Face
                float_mask = np.where(padded_attn_mask, 0.0, -10000.0)
                float_mask_3d = np.expand_dims(float_mask, axis=0)

                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": float_mask_3d,
                    "position_ids": position_ids,
                    "label": label
                })

            except Exception as e:
                # 极少数 AST 解析彻底崩溃的代码直接丢弃
                continue

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        return (
            torch.tensor(example['input_ids'], dtype=torch.long),
            torch.tensor(example['attention_mask'], dtype=torch.float32),  # 注意是 float32
            torch.tensor(example['position_ids'], dtype=torch.long),  # 新增 position_ids
            torch.tensor(example['label'], dtype=torch.long)
        )


# =============== Evaluation & Training Logic ===============

def evaluate(model, eval_dataset, args):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    logits_list = []
    y_trues = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
        inputs = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        position_ids = batch[2].to(args.device)  # 解包 position_ids
        labels = batch[3].to(args.device)

        with torch.no_grad():
            # 必须传入 position_ids
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels
            )
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


def train(model, train_dataset, eval_dataset, args):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    logger.info(f"***** Running training for {args.model_name} *****")

    best_f1 = 0.0
    model.zero_grad()

    for epoch in range(int(args.num_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(bar):
            model.train()
            inputs = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            position_ids = batch[2].to(args.device)  # 解包 position_ids
            labels = batch[3].to(args.device)

            # 必须传入 position_ids
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels
            )
            loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            bar.set_postfix({"loss": round(loss.item() * args.gradient_accumulation_steps, 4)})

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Evaluate at the end of each epoch
        results = evaluate(model, eval_dataset, args)
        logger.info(
            f"  Epoch {epoch + 1} Results: F1: {results['eval_f1']:.4f} | Recall: {results['eval_recall']:.4f} | Prec: {results['eval_precision']:.4f}")

        if results['eval_f1'] > best_f1:
            best_f1 = results['eval_f1']
            output_dir = os.path.join(args.output_dir, f"{args.model_name}_best_f1")
            os.makedirs(output_dir, exist_ok=True)

            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            logger.info(f"  [+] New best F1 ({best_f1:.4f})! Model saved to {output_dir}")


# =============== Main Control Flow ===============

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for GraphCodeBERT Vulnerability Detection")

    # Core data and path parameters
    parser.add_argument("--train_data_file", type=str, required=True, help="Path to training Parquet file")
    parser.add_argument("--eval_data_file", type=str, required=True, help="Path to validation Parquet file")
    parser.add_argument("--output_dir", type=str, default="./models", help="Model output directory")

    parser.add_argument("--model_name", type=str, default="GraphCodeBERT",
                        help="Model alias (used for naming the output folder)")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/graphcodebert-base",
                        help="HuggingFace model path or local path")

    # Hyperparameter settings
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    logger.info("\n" + "=" * 50)
    logger.info(f"🚀 Initializing GraphCodeBERT structural training: {args.model_name_or_path}")
    logger.info("=" * 50)

    # 1. Load Tokenizer (移除不必要的 UniXcoder 特殊 Token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # 2. Preprocess Data with DFG Extraction
    # 假设语言为 C/C++。若是 Python，需在内部指定 lang="python"
    train_dataset = GraphCodeBERTVulDataset(tokenizer, args.train_data_file, lang="c")
    eval_dataset = GraphCodeBERTVulDataset(tokenizer, args.eval_data_file, lang="c")

    # 3. Load Model & Inject LoRA
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        trust_remote_code=True
    )
    model = get_peft_model(model, peft_config)
    model.to(args.device)

    # 4. Execute Training
    train(model, train_dataset, eval_dataset, args)

    # 5. Save Tokenizer
    tokenizer.save_pretrained(os.path.join(args.output_dir, f"{args.model_name}_best_f1"))

    # 6. Clean up Memory
    del model, tokenizer, train_dataset, eval_dataset
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"✅ {args.model_name} training completed, GPU memory cleared.\n")


if __name__ == "__main__":
    main()