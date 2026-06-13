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
    def __init__(self, tokenizer, parquet_path, max_len=512, lang="cpp"):
        self.examples = []
        self.args_code_length = 384
        self.args_dfg_length = max_len - self.args_code_length
        self.tokenizer = tokenizer

        # 必须确保是 Fast Tokenizer 才能开启 offsets_mapping
        assert tokenizer.is_fast, "Must use a FastTokenizer for offset mapping!"

        logger.info(f"Loading dataset file at {parquet_path}")

        df = pd.read_parquet(parquet_path)
        df['label'] = df['vul'].astype(int)
        df = df[df['func'].str.len() <= 4000].copy()

        funcs = df['func'].tolist()
        labels = df['label'].tolist()

        # =========================================================
        # 🌟 修复点 1：正确初始化我们重构后的 AST 分析器
        # =========================================================
        logger.info(f"Initializing Custom DFG Extractor for lang: {lang}...")
        self.analyzer = IdentifierAnalyzer(lang=lang)

        for func, label in tqdm(zip(funcs, labels), total=len(funcs), desc="Building GraphCodeBERT Features"):
            try:
                # 1. 提取图拓扑 (传入 bytes)
                code_bytes = func.encode('utf-8')
                dfg_nodes, dfg_to_code_chars, dfg_to_dfg = self.analyzer.extract_dataflow(code_bytes)

                # 2. 文本编码并提取 Subword 级别的位置映射
                encoded = tokenizer(
                    func,
                    truncation=True,
                    max_length=self.args_code_length,
                    return_offsets_mapping=True  # 🌟 神级功能：返回每个 Token 在原字符串中的起始结束位置
                )
                text_ids = encoded['input_ids']
                offsets = encoded['offset_mapping']
                text_len = len(text_ids)

                # 3. 将变量的 Char Offset 映射到被切碎的 Subword Indices
                dfg_to_subwords = []
                for (start_char, end_char) in dfg_to_code_chars:
                    subword_indices = []
                    for idx, (o_start, o_end) in enumerate(offsets):
                        if o_start == o_end: continue  # 过滤特殊 token 的空白映射
                        if o_start < end_char and o_end > start_char:  # 判断区间有交集
                            subword_indices.append(idx)
                    dfg_to_subwords.append(subword_indices)

                # 4. 根据 args_dfg_length 截断图谱
                dfg_nodes = dfg_nodes[:self.args_dfg_length]
                dfg_to_subwords = dfg_to_subwords[:self.args_dfg_length]
                # 切断指向被截断节点的边
                dfg_to_dfg = [[e for e in edges if e < self.args_dfg_length] for edges in
                              dfg_to_dfg[:self.args_dfg_length]]

                # 5. 组装输入序列 (完全符合微软官方：DFG 转 <unk>，Position 为 0)
                input_ids = text_ids + [tokenizer.unk_token_id] * len(dfg_nodes)
                position_ids = [i + tokenizer.pad_token_id + 1 for i in range(text_len)] + [0] * len(dfg_nodes)

                # 6. Padding
                pad_len = max_len - len(input_ids)
                input_ids += [tokenizer.pad_token_id] * pad_len
                position_ids += [tokenizer.pad_token_id] * pad_len

                # 7. 构建完美官方对齐的 2D 注意力矩阵
                attn_mask = np.zeros((max_len, max_len), dtype=np.bool_)

                # a. 代码 Token 相互可见
                attn_mask[:text_len, :text_len] = True

                # b. 🌟 CLS 和 SEP 全局视野 (避免梯度回传丢失)
                for idx, token_id in enumerate(input_ids):
                    if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                        attn_mask[idx, :text_len + len(dfg_nodes)] = True
                        attn_mask[:text_len + len(dfg_nodes), idx] = True

                # c. DFG 节点与对应的 Subword 互相可见
                for dfg_idx, subword_idxs in enumerate(dfg_to_subwords):
                    matrix_dfg_idx = text_len + dfg_idx
                    for sub_idx in subword_idxs:
                        attn_mask[matrix_dfg_idx, sub_idx] = True
                        attn_mask[sub_idx, matrix_dfg_idx] = True

                # d. DFG 节点之间基于数据流依赖可见
                for dfg_idx, edges in enumerate(dfg_to_dfg):
                    matrix_dfg_idx = text_len + dfg_idx
                    for source_dfg_idx in edges:
                        matrix_source_idx = text_len + source_dfg_idx
                        attn_mask[matrix_dfg_idx, matrix_source_idx] = True
                        attn_mask[matrix_source_idx, matrix_dfg_idx] = True

                # 8. 以极低内存消耗保存
                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attn_mask,  # np.bool_
                    "position_ids": position_ids,
                    "label": label
                })

            except Exception as e:
                import traceback
                print(f"\n[!] 🚨 提取特征时发生崩溃！")
                print(f"[-] 当前报错: {e}")
                traceback.print_exc()
                raise e

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]

        # 惰性浮点 3D 掩码转换，完美规避 OOM 和 HF Expand 报错
        bool_mask = example['attention_mask']
        float_mask = np.where(bool_mask, 0.0, -10000.0).astype(np.float32)
        float_mask_3d = np.expand_dims(float_mask, axis=0)

        return (
            torch.tensor(example['input_ids'], dtype=torch.long),
            torch.tensor(float_mask_3d, dtype=torch.float32),
            torch.tensor(example['position_ids'], dtype=torch.long),
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

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
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

    train_dataset = GraphCodeBERTVulDataset(tokenizer, args.train_data_file, lang="cpp")
    eval_dataset = GraphCodeBERTVulDataset(tokenizer, args.eval_data_file, lang="cpp")

    # 3. Load Model & Inject LoRA
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1,target_modules=["query", "value"])
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