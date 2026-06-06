# analyze_rename_stealthiness.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd
import matplotlib.pyplot as plt


KEYWORDS = set("""
auto break case char const continue default do double else enum extern float for goto if inline int long
register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while
_Bool _Complex _Imaginary

alignas alignof and and_eq asm bitand bitor bool catch class compl constexpr const_cast decltype
delete dynamic_cast explicit export false friend mutable namespace new noexcept not not_eq nullptr operator
or or_eq private protected public reinterpret_cast static_assert static_cast template this thread_local throw
true try typeid typename using virtual wchar_t xor xor_eq

abstract assert boolean byte extends final finally implements import instanceof interface native package
strictfp super synchronized throws transient

False None True as async await def del elif except from global in is lambda nonlocal pass raise with yield
""".split())


@dataclass
class Token:
    kind: str
    text: str
    start: int
    end: int


def is_identifier_start(ch: str) -> bool:
    return ch == "_" or ch.isalpha()


def is_identifier_part(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def tokenize_code(code: str):
    """
    A lightweight tokenizer for C/C++/Java/Python-like code.

    It ignores whitespace and comments, but keeps identifiers, keywords,
    literals, numbers, and operators as tokens.

    The goal is not full compilation-level parsing, but stable comparison
    between original_code and adversarial_code when the transformation is
    mainly identifier renaming.
    """
    tokens = []
    i = 0
    n = len(code)

    while i < n:
        ch = code[i]

        # Whitespace
        if ch.isspace():
            i += 1
            continue

        # Line comment: // ...
        if i + 1 < n and code[i:i + 2] == "//":
            j = i + 2
            while j < n and code[j] != "\n":
                j += 1
            i = j
            continue

        # Block comment: /* ... */
        if i + 1 < n and code[i:i + 2] == "/*":
            j = i + 2
            while j + 1 < n and code[j:j + 2] != "*/":
                j += 1
            i = min(j + 2, n)
            continue

        # String or char literal
        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            escaped = False
            while j < n:
                if escaped:
                    escaped = False
                elif code[j] == "\\":
                    escaped = True
                elif code[j] == quote:
                    j += 1
                    break
                j += 1
            tokens.append(Token("literal", code[i:j], i, j))
            i = j
            continue

        # Identifier or keyword
        if is_identifier_start(ch):
            j = i + 1
            while j < n and is_identifier_part(code[j]):
                j += 1
            text = code[i:j]
            kind = "keyword" if text in KEYWORDS else "id"
            tokens.append(Token(kind, text, i, j))
            i = j
            continue

        # Number
        if ch.isdigit():
            j = i + 1
            while j < n and (code[j].isalnum() or code[j] in "._xX"):
                j += 1
            tokens.append(Token("number", code[i:j], i, j))
            i = j
            continue

        # Operator / punctuation
        # Keep common multi-character operators together when possible.
        multi_ops = [
            ">>=", "<<=", "++", "--", "==", "!=", ">=", "<=", "&&", "||",
            "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "->", "::",
            "<<", ">>", "=>", "**"
        ]
        matched = None
        for op in multi_ops:
            if code.startswith(op, i):
                matched = op
                break

        if matched:
            tokens.append(Token("op", matched, i, i + len(matched)))
            i += len(matched)
        else:
            tokens.append(Token("op", ch, i, i + 1))
            i += 1

    return tokens


def split_identifier(name: str):
    """
    Split identifier into subwords.

    Examples:
        input_buffer -> ["input", "buffer"]
        inputBuffer  -> ["input", "buffer"]
        SSLContext   -> ["ssl", "context"]
        buf2Len      -> ["buf", "2", "len"]
    """
    name = name.strip("_")
    if not name:
        return []

    parts = re.split(r"[_\W]+", name)
    subwords = []

    pattern = re.compile(
        r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|"
        r"[A-Z]?[a-z]+|"
        r"[A-Z]+|"
        r"\d+"
    )

    for part in parts:
        if not part:
            continue
        found = pattern.findall(part)
        if found:
            subwords.extend([x.lower() for x in found])
        else:
            subwords.append(part.lower())

    return subwords


def edit_distance(a, b):
    """
    Levenshtein edit distance for strings or lists.
    """
    if a == b:
        return 0

    a = list(a)
    b = list(b)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    return previous[-1]


def normalized_edit_distance(a, b):
    denom = max(len(a), len(b), 1)
    return edit_distance(a, b) / denom


def jaccard_distance(a, b):
    sa = set(a)
    sb = set(b)

    if not sa and not sb:
        return 0.0

    return 1.0 - len(sa & sb) / max(len(sa | sb), 1)


def token_signature(token: Token):
    """
    For alignment, different identifiers are treated as the same abstract ID token.
    Non-identifier tokens are matched by exact text.
    """
    if token.kind == "id":
        return "<ID>"
    return f"{token.kind}:{token.text}"


def detect_identifier_changes(original_code: str, adversarial_code: str):
    """
    Detect identifier substitutions by token alignment.

    Main assumption:
        The adversarial code mainly differs from the original code by identifier renaming.

    Returns:
        changed_pairs: list of (old_identifier, new_identifier)
        warning: whether fallback alignment was used or suspicious mismatch occurred
    """
    original_tokens = tokenize_code(original_code)
    adversarial_tokens = tokenize_code(adversarial_code)

    changed_pairs = []
    warning = False

    # Fast path: same token length and non-id tokens remain identical.
    if len(original_tokens) == len(adversarial_tokens):
        non_id_ok = True

        for ot, at in zip(original_tokens, adversarial_tokens):
            if ot.kind == "id" and at.kind == "id":
                continue
            if ot.kind != at.kind or ot.text != at.text:
                non_id_ok = False
                break

        if non_id_ok:
            for ot, at in zip(original_tokens, adversarial_tokens):
                if ot.kind == "id" and at.kind == "id" and ot.text != at.text:
                    changed_pairs.append((ot.text, at.text))

            return changed_pairs, warning, original_tokens, adversarial_tokens

    # Fallback path: align by abstract token signature.
    warning = True

    original_sig = [token_signature(t) for t in original_tokens]
    adversarial_sig = [token_signature(t) for t in adversarial_tokens]

    matcher = SequenceMatcher(None, original_sig, adversarial_sig, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            o_block = original_tokens[i1:i2]
            a_block = adversarial_tokens[j1:j2]

            for ot, at in zip(o_block, a_block):
                if ot.kind == "id" and at.kind == "id" and ot.text != at.text:
                    changed_pairs.append((ot.text, at.text))

        elif tag == "replace":
            o_block = original_tokens[i1:i2]
            a_block = adversarial_tokens[j1:j2]

            if len(o_block) == len(a_block):
                for ot, at in zip(o_block, a_block):
                    if ot.kind == "id" and at.kind == "id" and ot.text != at.text:
                        changed_pairs.append((ot.text, at.text))

    return changed_pairs, warning, original_tokens, adversarial_tokens


def count_changed_lines(original_code: str, adversarial_code: str):
    original_lines = original_code.splitlines()
    adversarial_lines = adversarial_code.splitlines()

    matcher = SequenceMatcher(None, original_lines, adversarial_lines, autojunk=False)

    changed_original_lines = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            for idx in range(i1, i2):
                changed_original_lines.add(idx)

    return len(changed_original_lines), max(len(original_lines), 1)


def analyze_one_record(record, sample_id):
    original_code = record.get("original_code", "")
    adversarial_code = record.get("adversarial_code", "")

    if not isinstance(original_code, str) or not isinstance(adversarial_code, str):
        raise ValueError(f"sample {sample_id}: original_code/adversarial_code must be strings")

    changed_pairs, alignment_warning, original_tokens, adversarial_tokens = detect_identifier_changes(
        original_code,
        adversarial_code
    )

    pair_counter = Counter(changed_pairs)

    original_identifier_tokens = [t.text for t in original_tokens if t.kind == "id"]
    unique_original_identifiers = set(original_identifier_tokens)

    total_code_tokens = len(original_tokens)
    total_identifier_occurrences = len(original_identifier_tokens)
    total_unique_identifiers = len(unique_original_identifiers)

    changed_identifier_occurrences = sum(pair_counter.values())
    changed_unique_original_identifiers = len(set(old for old, _ in pair_counter.keys()))

    mapping_rows = []

    weighted_char_edit_sum = 0
    weighted_char_edit_ratio_sum = 0.0
    weighted_subword_edit_ratio_sum = 0.0
    weighted_subword_jaccard_sum = 0.0

    for (old, new), count in pair_counter.items():
        old_subwords = split_identifier(old)
        new_subwords = split_identifier(new)

        char_lev = edit_distance(old, new)
        char_norm = normalized_edit_distance(old, new)

        subword_lev = edit_distance(old_subwords, new_subwords)
        subword_norm = normalized_edit_distance(old_subwords, new_subwords)

        subword_jaccard = jaccard_distance(old_subwords, new_subwords)

        weighted_char_edit_sum += char_lev * count
        weighted_char_edit_ratio_sum += char_norm * count
        weighted_subword_edit_ratio_sum += subword_norm * count
        weighted_subword_jaccard_sum += subword_jaccard * count

        mapping_rows.append({
            "sample_id": sample_id,
            "old_identifier": old,
            "new_identifier": new,
            "occurrences": count,
            "old_subwords": " ".join(old_subwords),
            "new_subwords": " ".join(new_subwords),
            "char_edit_distance": char_lev,
            "char_edit_ratio": char_norm,
            "subword_edit_distance": subword_lev,
            "subword_edit_ratio": subword_norm,
            "subword_jaccard_distance": subword_jaccard,
        })

    denom_changed = max(changed_identifier_occurrences, 1)

    changed_lines, total_lines = count_changed_lines(original_code, adversarial_code)

    # A direct global character diff proxy.
    # This is not exact Levenshtein distance, but it is stable and efficient for long functions.
    sequence_char_diff_ratio = 1.0 - SequenceMatcher(
        None,
        original_code,
        adversarial_code,
        autojunk=False
    ).ratio()

    metrics = {
        "sample_id": sample_id,

        "total_code_tokens": total_code_tokens,
        "total_identifier_occurrences": total_identifier_occurrences,
        "total_unique_identifiers": total_unique_identifiers,

        "changed_identifier_occurrences": changed_identifier_occurrences,
        "changed_unique_original_identifiers": changed_unique_original_identifiers,

        # 修改标识符出现次数 / 全部代码 token
        "token_modification_ratio": changed_identifier_occurrences / max(total_code_tokens, 1),

        # 修改标识符出现次数 / 全部标识符出现次数
        "identifier_occurrence_modification_ratio": changed_identifier_occurrences / max(total_identifier_occurrences, 1),

        # 修改的唯一标识符数量 / 全部唯一标识符数量
        "unique_identifier_modification_ratio": changed_unique_original_identifiers / max(total_unique_identifiers, 1),

        # 标识符本身的编辑距离总和 / 整个函数字符数
        # 这个指标比较适合表示“变量名扰动相对于整个函数的幅度”
        "relative_identifier_edit_to_function_chars": weighted_char_edit_sum / max(len(original_code), 1),

        # 作为补充：原代码和对抗代码整体字符差异的 SequenceMatcher 近似值
        "sequence_char_diff_ratio": sequence_char_diff_ratio,

        # 改动行比例
        "changed_line_ratio": changed_lines / total_lines,

        # 单词间平均修改幅度，按出现次数加权
        "avg_char_edit_ratio_weighted": weighted_char_edit_ratio_sum / denom_changed,
        "avg_subword_edit_ratio_weighted": weighted_subword_edit_ratio_sum / denom_changed,
        "avg_subword_jaccard_distance_weighted": weighted_subword_jaccard_sum / denom_changed,

        "alignment_warning": alignment_warning,
    }

    return metrics, mapping_rows


def load_json_records(path):
    """
    Supports:
        1. JSON list:
            [
              {"original_code": "...", "adversarial_code": "..."},
              ...
            ]

        2. Single JSON object:
            {"original_code": "...", "adversarial_code": "..."}

        3. JSONL:
            {"original_code": "...", "adversarial_code": "..."}
            {"original_code": "...", "adversarial_code": "..."}
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "original_code" in data and "adversarial_code" in data:
                return [data]

            # Common wrapper cases, e.g. {"results": [...]}.
            for value in data.values():
                if isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        if "original_code" in value[0] and "adversarial_code" in value[0]:
                            return value

            raise ValueError("JSON object does not contain original_code/adversarial_code")
    except json.JSONDecodeError:
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
        return records


def save_histogram(df, column, out_path, title, xlabel):
    values = df[column].dropna()

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_top_rename_plot(mapping_df, out_path, top_k=20):
    if mapping_df.empty:
        return

    tmp = mapping_df.copy()
    tmp["rename_pair"] = tmp["old_identifier"] + " → " + tmp["new_identifier"]

    top_pairs = (
        tmp.groupby("rename_pair", as_index=False)["occurrences"]
        .sum()
        .sort_values("occurrences", ascending=False)
        .head(top_k)
    )

    if top_pairs.empty:
        return

    plt.figure(figsize=(10, max(5, 0.35 * len(top_pairs))))
    plt.barh(top_pairs["rename_pair"][::-1], top_pairs["occurrences"][::-1])
    plt.title(f"Top {len(top_pairs)} Identifier Renames")
    plt.xlabel("Occurrences")
    plt.ylabel("Rename pair")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


import os
import json
import argparse
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

# 导入你项目中的原生模块 (请根据实际路径微调)
from utils.ast_tools import IdentifierAnalyzer
from utils.llm_loader import LocalLLMClient
from utils.mlm_engine import MLMEngine
from attacks.LightWeightCandidateGenerator import LightweightCandidateGenerator

# 假设这些是你原本用来做常规分析的工具函数


def main():
    parser = argparse.ArgumentParser(
        description="Analyze identifier-level modification magnitude, Semantic Similarity, and PPL."
    )

    parser.add_argument("--input_json", help="Path to adversarial results JSON/JSONL file")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to system config (YAML)")
    parser.add_argument("--out", default="rename_stealthiness_stats", help="Output directory")
    parser.add_argument("--sample-id-field", default=None, help="Optional field used as sample_id")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K rename pairs to plot")

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # =========================================================================
    # 1. 初始化引擎与配置 (与攻击主程序严格对齐)
    # =========================================================================
    print(f"[*] Loading config from {args.config}...")
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        parser.error(f"❌ Configuration file not found: {args.config}")

    print("\n[*] Initializing Deep Learning Engines for Strict Offline Evaluation...")
    lang = config.get('global', {}).get('lang', 'cpp')
    analyzer = IdentifierAnalyzer(lang=lang)

    mlm_engine_name = config['models'].get('mlm_engine', 'microsoft/codebert-base-mlm')
    mlm_engine = MLMEngine(mlm_engine_name)

    llm_name = config['models'].get('llm_generator', 'models/qwen2.5-1.5b-code')
    llm_client = LocalLLMClient(model_name=llm_name)

    # 实例化 LightweightCandidateGenerator，为了复用它里面神级的辅助函数
    mlm_gen = LightweightCandidateGenerator(
        mlm_engine=mlm_engine,
        analyzer=analyzer,
        config=config,
        llm_client=llm_client,
    )

    # =========================================================================
    # 2. 读取并解析样本记录
    # =========================================================================
    print(f"\n[*] Loading records from {args.input_json}...")
    records = load_json_records(args.input_json)

    all_metrics = []
    all_mapping_rows = []

    # 用于批量计算的队列
    ppl_orig_codes = []
    ppl_adv_codes = []
    ppl_record_indices = []

    sim_prefixes = []
    sim_orig_vars = []
    sim_adv_vars = []
    sim_suffixes = []
    sim_record_indices = []

    print("[*] Performing structural analysis and context slicing...")
    for idx, record in enumerate(records):
        sample_id = record.get(args.sample_id_field, idx) if args.sample_id_field else idx

        # 原有的基础结构指标分析
        metrics, mapping_rows = analyze_one_record(record, sample_id)

        orig_code = record.get("original_code", "")
        adv_code = record.get("adversarial_code", "")
        is_success = record.get("is_success", False)

        if is_success and orig_code and adv_code:
            # --- 收集 PPL 数据 (全局代码上下文) ---
            ppl_orig_codes.append(orig_code)
            ppl_adv_codes.append(adv_code)
            ppl_record_indices.append(idx)

            # --- 收集 Semantic Similarity 数据 (AST 语句级切片上下文) ---
            orig_bytes = orig_code.encode("utf-8")
            identifiers = analyzer.extract_identifiers(orig_bytes)

            # 解析本次攻击替换了哪些变量 (从 mapping_rows 中提取对齐信息)
            # 解析本次攻击替换了哪些变量 (直接从 JSONL 原生的 replaced_names 提取，绝对安全)
            replaced_names = record.get("replaced_names", {})
            for old_var, new_var in replaced_names.items():

                # 确保提取出来的变量是个字符串，防范极端的脏数据
                old_var = str(old_var)
                new_var = str(new_var)

                if old_var in identifiers and old_var != new_var:
                    try:
                        # 严格复用 _find_best_context_occurrence 获取最佳语境位置
                        best_occ_idx = mlm_gen._find_best_context_occurrence(orig_bytes, identifiers[old_var])
                        target_info = identifiers[old_var][best_occ_idx]

                        # 严格复用 _extract_local_context_ast 获取精确语法切片
                        local_prefix, local_suffix = mlm_gen._extract_local_context_ast(
                            orig_bytes, target_info['start'], target_info['end']
                        )

                        sim_prefixes.append(local_prefix)
                        sim_orig_vars.append(old_var)
                        sim_adv_vars.append(new_var)
                        sim_suffixes.append(local_suffix)
                        sim_record_indices.append(idx)
                    except Exception as e:
                        print(f"        [!] 解析变量上下文出错 {old_var}->{new_var}: {e}")
                        continue
        else:
            # 失败的样本，深度学习指标赋 NaN
            metrics["semantic_similarity"] = np.nan
            metrics["orig_ppl"] = np.nan
            metrics["adv_ppl"] = np.nan
            metrics["ppl_ratio"] = np.nan
            metrics["ppl_diff"] = np.nan

        all_metrics.append(metrics)
        all_mapping_rows.extend(mapping_rows)

    # =========================================================================
    # 3. 严格一致的批量深度学习指标推理
    # =========================================================================

    # 3.1 PPL 批量计算 (复用 Generator 中的 _calculate_perplexity_batch)
    if ppl_orig_codes:
        ppl_bs = config.get('candidate_generation', {}).get('ppl_batch_size', 4)
        print(f"\n[*] Calculating Strict Perplexity for {len(ppl_orig_codes)} samples (Batch Size: {ppl_bs})...")

        batch_orig_ppls = mlm_gen._calculate_perplexity_batch(ppl_orig_codes, batch_size=ppl_bs)
        batch_adv_ppls = mlm_gen._calculate_perplexity_batch(ppl_adv_codes, batch_size=ppl_bs)

        for i, global_idx in enumerate(ppl_record_indices):
            o_ppl = batch_orig_ppls[i]
            a_ppl = batch_adv_ppls[i]
            all_metrics[global_idx]["orig_ppl"] = float(o_ppl)
            all_metrics[global_idx]["adv_ppl"] = float(a_ppl)
            all_metrics[global_idx]["ppl_diff"] = float(a_ppl - o_ppl)
            all_metrics[global_idx]["ppl_ratio"] = float(a_ppl / o_ppl) if o_ppl > 0 else float('inf')

    # 3.2 变量级语义相似度批量计算 (复用 Generator 中的 _get_variable_token_embeddings)
    if sim_prefixes:
        print(f"[*] Calculating Strict Variable-Level Similarity for {len(sim_prefixes)} rename instances...")

        # 提取原始代码中特定变量 Token 的 Embedding
        orig_embs = mlm_gen._get_variable_token_embeddings(
            sim_prefixes, sim_orig_vars, sim_suffixes, batch_size=64
        ).to(mlm_engine.device)

        # 提取对抗代码中新变量 Token 的 Embedding
        adv_embs = mlm_gen._get_variable_token_embeddings(
            sim_prefixes, sim_adv_vars, sim_suffixes, batch_size=64
        ).to(mlm_engine.device)

        # 进行严格的张量广播余弦相似度计算
        sims = F.cosine_similarity(orig_embs, adv_embs, dim=-1).cpu().numpy()

        # 如果一个样本被替换了多个变量，我们要对其相似度取平均
        sample_sim_accum = {}
        sample_sim_count = {}
        for i, global_idx in enumerate(sim_record_indices):
            sample_sim_accum[global_idx] = sample_sim_accum.get(global_idx, 0.0) + sims[i]
            sample_sim_count[global_idx] = sample_sim_count.get(global_idx, 0) + 1

        for global_idx in sample_sim_accum:
            avg_sim = sample_sim_accum[global_idx] / sample_sim_count[global_idx]
            all_metrics[global_idx]["semantic_similarity"] = float(avg_sim)

    # =========================================================================
    # 4. 结果汇总与保存
    # =========================================================================
    metrics_df = pd.DataFrame(all_metrics)
    mapping_df = pd.DataFrame(all_mapping_rows)

    metrics_csv = os.path.join(args.out, "sample_metrics.csv")
    mapping_csv = os.path.join(args.out, "identifier_renames.csv")
    summary_json = os.path.join(args.out, "summary.json")

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    mapping_df.to_csv(mapping_csv, index=False, encoding="utf-8-sig")

    summary = {
        "num_samples": int(len(metrics_df)),
        "num_successfully_changed_samples": int(
            (metrics_df["changed_identifier_occurrences"] > 0).sum()) if not metrics_df.empty else 0,
        "num_rename_pairs": int(len(mapping_df)),
    }

    numeric_columns = [
        "token_modification_ratio", "identifier_occurrence_modification_ratio",
        "unique_identifier_modification_ratio", "relative_identifier_edit_to_function_chars",
        "sequence_char_diff_ratio", "changed_line_ratio",
        "avg_char_edit_ratio_weighted", "avg_subword_edit_ratio_weighted", "avg_subword_jaccard_distance_weighted",
        "semantic_similarity", "orig_ppl", "adv_ppl", "ppl_ratio", "ppl_diff"
    ]

    for col in numeric_columns:
        if col in metrics_df.columns and not metrics_df.empty:
            valid_series = metrics_df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if not valid_series.empty:
                summary[col] = {
                    "mean": float(valid_series.mean()),
                    "median": float(valid_series.median()),
                    "min": float(valid_series.min()),
                    "max": float(valid_series.max()),
                }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not metrics_df.empty:
        # 原有的柱状图
        save_histogram(metrics_df, "token_modification_ratio",
                       os.path.join(args.out, "hist_token_modification_ratio.png"), "Token-level Modification Ratio",
                       "Ratio")

        # ✨ 新增：极其精准的变量级语义相似度图表
        if "semantic_similarity" in metrics_df.columns:
            save_histogram(
                metrics_df.dropna(subset=["semantic_similarity"]),
                "semantic_similarity",
                os.path.join(args.out, "hist_semantic_similarity.png"),
                "Variable-Level Semantic Similarity (CodeBERT Pool)",
                "Similarity Score (1.0 = identical)"
            )

        # ✨ 新增：PPL Ratio 劣化率直方图
        if "ppl_ratio" in metrics_df.columns:
            # 过滤掉极端异常值以便画图更美观 (比如 ratio > 10 的截断)
            plot_df = metrics_df.dropna(subset=["ppl_ratio"])
            plot_df = plot_df[plot_df["ppl_ratio"] < 5.0]
            save_histogram(
                plot_df,
                "ppl_ratio",
                os.path.join(args.out, "hist_ppl_ratio.png"),
                "Perplexity (PPL) Degradation Ratio",
                "Adversarial PPL / Original PPL (Close to 1.0 is better)"
            )

    save_top_rename_plot(mapping_df, os.path.join(args.out, "top_identifier_renames.png"), top_k=args.top_k)

    print(f"\n✅ Offline Stealthiness Evaluation Completed! Results saved to: {args.out}")


if __name__ == "__main__":
    main()