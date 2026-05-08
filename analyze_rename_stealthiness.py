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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze identifier-level modification magnitude between original_code and adversarial_code."
    )

    parser.add_argument("input_json", help="Path to JSON or JSONL file")
    parser.add_argument("--out", default="rename_stealthiness_stats", help="Output directory")
    parser.add_argument("--sample-id-field", default=None, help="Optional field used as sample_id")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K rename pairs to plot")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    records = load_json_records(args.input_json)

    all_metrics = []
    all_mapping_rows = []

    for idx, record in enumerate(records):
        if args.sample_id_field and args.sample_id_field in record:
            sample_id = record[args.sample_id_field]
        else:
            sample_id = idx

        metrics, mapping_rows = analyze_one_record(record, sample_id)

        all_metrics.append(metrics)
        all_mapping_rows.extend(mapping_rows)

    metrics_df = pd.DataFrame(all_metrics)
    mapping_df = pd.DataFrame(all_mapping_rows)

    metrics_csv = os.path.join(args.out, "sample_metrics.csv")
    mapping_csv = os.path.join(args.out, "identifier_renames.csv")
    summary_json = os.path.join(args.out, "summary.json")

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    mapping_df.to_csv(mapping_csv, index=False, encoding="utf-8-sig")

    summary = {
        "num_samples": int(len(metrics_df)),
        "num_successfully_changed_samples": int((metrics_df["changed_identifier_occurrences"] > 0).sum())
        if not metrics_df.empty else 0,
        "num_rename_pairs": int(len(mapping_df)),
    }

    numeric_columns = [
        "token_modification_ratio",
        "identifier_occurrence_modification_ratio",
        "unique_identifier_modification_ratio",
        "relative_identifier_edit_to_function_chars",
        "sequence_char_diff_ratio",
        "changed_line_ratio",
        "avg_char_edit_ratio_weighted",
        "avg_subword_edit_ratio_weighted",
        "avg_subword_jaccard_distance_weighted",
    ]

    for col in numeric_columns:
        if col in metrics_df.columns and not metrics_df.empty:
            summary[col] = {
                "mean": float(metrics_df[col].mean()),
                "median": float(metrics_df[col].median()),
                "min": float(metrics_df[col].min()),
                "max": float(metrics_df[col].max()),
            }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not metrics_df.empty:
        save_histogram(
            metrics_df,
            "token_modification_ratio",
            os.path.join(args.out, "hist_token_modification_ratio.png"),
            "Token-level Modification Ratio",
            "Changed identifier occurrences / all code tokens"
        )

        save_histogram(
            metrics_df,
            "identifier_occurrence_modification_ratio",
            os.path.join(args.out, "hist_identifier_occurrence_modification_ratio.png"),
            "Identifier Occurrence Modification Ratio",
            "Changed identifier occurrences / all identifier occurrences"
        )

        save_histogram(
            metrics_df,
            "unique_identifier_modification_ratio",
            os.path.join(args.out, "hist_unique_identifier_modification_ratio.png"),
            "Unique Identifier Modification Ratio",
            "Changed unique identifiers / all unique identifiers"
        )

        save_histogram(
            metrics_df,
            "relative_identifier_edit_to_function_chars",
            os.path.join(args.out, "hist_relative_identifier_edit_to_function_chars.png"),
            "Relative Identifier Edit Magnitude",
            "Identifier edit distance sum / function characters"
        )

        save_histogram(
            metrics_df,
            "avg_subword_edit_ratio_weighted",
            os.path.join(args.out, "hist_avg_subword_edit_ratio_weighted.png"),
            "Average Subword Edit Ratio",
            "Weighted average subword edit ratio"
        )

    save_top_rename_plot(
        mapping_df,
        os.path.join(args.out, "top_identifier_renames.png"),
        top_k=args.top_k
    )

    print(f"Done. Results saved to: {args.out}")
    print(f"- Per-sample metrics: {metrics_csv}")
    print(f"- Identifier rename details: {mapping_csv}")
    print(f"- Summary: {summary_json}")


if __name__ == "__main__":
    main()