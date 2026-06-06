# # import json
# # import pyarrow as pa
# # import pyarrow.parquet as pq
# #
# #
# # def clean_cwe(cwe_data):
# #     """
# #     清洗 CWE 数据。
# #     处理格式如: [{"element":[67, 87, 69, 45, 55, 56, 55]}] 或 ["CWE-787"]
# #     """
# #     if not cwe_data:
# #         return None
# #
# #     cleaned_list = []
# #
# #     # 如果是列表格式
# #     if isinstance(cwe_data, list):
# #         for item in cwe_data:
# #             # 处理用户提到的特殊格式: {"element": [67, 87, ...]}
# #             if isinstance(item, dict) and 'element' in item:
# #                 try:
# #                     # 将 ASCII 码转换回字符串
# #                     tag = "".join(chr(c) for c in item['element'])
# #                     cleaned_list.append(tag)
# #                 except:
# #                     continue
# #             # 处理标准格式: "CWE-787"
# #             elif isinstance(item, str):
# #                 cleaned_list.append(item)
# #
# #     # 将多个 CWE 用逗号连接成一个字符串返回，例如 "CWE-787,CWE-119"
# #     return ", ".join(cleaned_list) if cleaned_list else None
# #
# #
# # def robust_jsonl_to_parquet(input_file, output_file, chunksize=10000):
# #     print(f"开始处理数据集，正在从 {input_file} 写入 {output_file}...")
# #
# #     # 定义 Parquet Schema，确保 cwe 列定义为 string
# #     schema = pa.schema([
# #         ('func', pa.large_string()),
# #         ('func_after', pa.large_string()),
# #         ('vul', pa.int64()),
# #         ('cwe', pa.string()),  # 统一存为字符串
# #         ('project', pa.string())
# #     ])
# #
# #     writer = None
# #     records = []
# #     total_processed = 0
# #
# #     try:
# #         with open(input_file, 'r', encoding='utf-8') as f:
# #             for line_num, line in enumerate(f):
# #                 line = line.strip()
# #                 if not line:
# #                     continue
# #
# #                 try:
# #                     data = json.loads(line)
# #                 except json.JSONDecodeError:
# #                     continue
# #
# #                 # 核心逻辑：清洗 CWE 字段
# #                 raw_cwe = data.get('cwe', None)
# #                 final_cwe = clean_cwe(raw_cwe)
# #
# #                 # 提取并构建行数据
# #                 records.append({
# #                     'func': data.get('func', None),
# #                     'func_after': None,  # 严格存入空值
# #                     'vul': data.get('target', None),
# #                     'cwe': final_cwe,
# #                     'project': data.get('project', None)
# #                 })
# #
# #                 if len(records) >= chunksize:
# #                     table = pa.Table.from_pylist(records, schema=schema)
# #                     if writer is None:
# #                         writer = pq.ParquetWriter(output_file, schema)
# #                     writer.write_table(table)
# #                     total_processed += len(records)
# #                     print(f"已处理 {total_processed} 行...")
# #                     records.clear()
# #
# #             # 处理尾部数据
# #             if records:
# #                 table = pa.Table.from_pylist(records, schema=schema)
# #                 if writer is None:
# #                     writer = pq.ParquetWriter(output_file, schema)
# #                 writer.write_table(table)
# #                 total_processed += len(records)
# #
# #     except Exception as e:
# #         print(f"处理出错: {e}")
# #     finally:
# #         if writer:
# #             writer.close()
# #             print(f"处理完成！共写入 {total_processed} 行，保存至: {output_file}")
# #
# #
# # if __name__ == "__main__":
# #     input_jsonl = "data/diversevul_20230702.json"
# #     output_parquet = "data/diverse_vul.parquet"
# #     robust_jsonl_to_parquet(input_jsonl, output_parquet)
#
# import csv
# import sys
# import pyarrow as pa
# import pyarrow.parquet as pq
#
# # 突破 Python 默认的 CSV 字段长度限制，防止超长源代码导致报错
# # 使用一个极大的整数覆盖默认限制
# csv.field_size_limit(2147483647)
#
#
# def process_cpp_csv_to_parquet(input_file, output_file, chunksize=10000):
#     print(f"🚀 开始提取 C/C++ 数据集...")
#
#     # 定义输出结构
#     schema = pa.schema([
#         ('func', pa.large_string()),
#         ('func_after', pa.large_string()),
#         ('vul', pa.int64()),
#         ('cwe', pa.string()),
#         ('project', pa.string())
#     ])
#
#     # 允许的 C/C++ 语言标签（不区分大小写）
#     cpp_labels = {'c', 'cpp', 'c++', 'c/cpp', 'c/c++'}
#
#     writer = None
#     records = []
#     total_processed = 0
#     filtered_count = 0
#
#     try:
#         with open(input_file, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#
#             for row in reader:
#                 # 1. 提取 lang 并进行过滤 (转小写后匹配)
#                 lang = str(row.get('lang', '')).strip().lower()
#                 if lang not in cpp_labels:
#                     continue
#
#                 # 2. 字段转换与映射
#                 # func_before -> func
#                 # CWE ID -> cwe
#                 try:
#                     vul_val = int(row.get('vul')) if row.get('vul') is not None else 0
#                 except:
#                     vul_val = 0
#
#                 records.append({
#                     'func': row.get('func_before', None),
#                     'func_after': row.get('func_after', None),
#                     'vul': vul_val,
#                     'cwe': row.get('CWE ID', None),
#                     'project': row.get('project', None)
#                 })
#
#                 filtered_count += 1
#
#                 # 3. 分块写入
#                 if len(records) >= chunksize:
#                     table = pa.Table.from_pylist(records, schema=schema)
#                     if writer is None:
#                         writer = pq.ParquetWriter(output_file, schema)
#                     writer.write_table(table)
#                     total_processed += len(records)
#                     print(f"已提取 {total_processed} 条 C/C++ 样本...")
#                     records.clear()
#
#             # 处理剩余数据
#             if records:
#                 table = pa.Table.from_pylist(records, schema=schema)
#                 if writer is None:
#                     writer = pq.ParquetWriter(output_file, schema)
#                 writer.write_table(table)
#                 total_processed += len(records)
#
#     except Exception as e:
#         print(f"❌ 处理中断: {e}")
#     finally:
#         if writer:
#             writer.close()
#             print(f"✅ 处理完成！")
#             print(f"统计：从 CSV 中共提取出 {filtered_count} 条 C/C++ 代码样本。")
#             print(f"保存路径: {output_file}")
#
# # ----------------- 使用方式 -----------------
# if __name__ == "__main__":
#     # 替换为你实际的文件路径
#     input_csv = "data/MSR_data_cleaned.csv"
#     output_parquet = "data/bigvul_polars.parquet"
#
#     process_cpp_csv_to_parquet(input_csv, output_parquet)

# import pandas as pd
# from sklearn.model_selection import train_test_split
#
#
# def split_dataset_with_constraint(input_parquet, train_path, test_path, test_size=0.2, random_state=42):
#     """
#     拆分数据集，并确保特定样本强制进入训练集。
#
#     :param input_parquet: 输入的完整数据集路径
#     :param train_path: 训练集保存路径
#     :param test_path: 测试集保存路径
#     :param test_size: 测试集占比（针对可拆分的部分）
#     """
#     print(f"正在读取数据集: {input_parquet}")
#     df = pd.read_parquet(input_parquet)
#
#     # 1. 定义强制训练集的条件
#     # 条件：func_after 不为 None/NaN 且 不为空字符串，并且 vul == 1
#     force_train_mask = (
#             df['func_after'].notna() &
#             (df['func_after'].astype(str).str.strip() != "") &
#             (df['vul'] == 1)
#     )
#
#     # 2. 提取强制训练集样本和可自由拆分的样本
#     df_force_train = df[force_train_mask].copy()
#     df_splittable = df[~force_train_mask].copy()
#
#     print(f"统计:")
#     print(f" - 总样本数: {len(df)}")
#     print(f" - 强制进入训练集的样本数: {len(df_force_train)}")
#     print(f" - 参与随机拆分的样本数: {len(df_splittable)}")
#
#     # 3. 对可拆分部分进行常规拆分
#     if len(df_splittable) > 0:
#         train_part, test_part = train_test_split(
#             df_splittable,
#             test_size=test_size,
#             random_state=random_state,
#             stratify=df_splittable['vul'] if len(df_splittable['vul'].unique()) > 1 else None
#         )
#     else:
#         train_part = pd.DataFrame(columns=df.columns)
#         test_part = pd.DataFrame(columns=df.columns)
#
#     # 4. 合并强制样本到训练集
#     final_train = pd.concat([train_part, df_force_train], ignore_index=True)
#     final_test = test_part
#
#     # 打乱训练集顺序（因为强制样本是直接追加在后面的）
#     final_train = final_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
#
#     print(f"最终结果:")
#     print(f" - 最终训练集大小: {len(final_train)}")
#     print(f" - 最终测试集大小: {len(final_test)}")
#
#     # 5. 保存结果
#     final_train.to_parquet(train_path, index=False)
#     final_test.to_parquet(test_path, index=False)
#     print(f"✅ 数据集已保存至 {train_path} 和 {test_path}")
#
#
# # ----------------- 使用方式 -----------------
# if __name__ == "__main__":
#     # 假设你之前处理好的文件是这个
#     input_file = "data/cleaned_dataset.parquet"
#
#     split_dataset_with_constraint(
#         input_file,
#         train_path="data/train.parquet",
#         test_path="data/test.parquet",
#         test_size=0.2  # 20% 作为测试集
#     )

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# def process_large_json_array(input_file, output_file, chunksize=10000):
#     print(f"正在通过 ijson 流式处理: {input_file}")
#
#     schema = pa.schema([
#         ('func', pa.large_string()),
#         ('vul', pa.int64())
#     ])
#
#     writer = None
#
#     with open(input_file, 'rb') as f:
#         # ijson.items 会迭代数组中的每一个对象
#         # 'item' 表示数组里的每一个元素
#         objects = ijson.items(f, 'item')
#
#         chunk = []
#         total_processed = 0
#
#         for obj in objects:
#             # 提取并清洗数据
#             func = obj.get('func', "")
#             # 处理可能的 target -> vul 映射
#             vul = obj.get('vul', obj.get('target', 0))
#
#             chunk.append({'func': func, 'vul': vul})
#
#             if len(chunk) >= chunksize:
#                 df = pd.DataFrame(chunk)
#                 table = pa.Table.from_pandas(df, schema=schema)
#
#                 if writer is None:
#                     writer = pq.ParquetWriter(output_file, schema)
#
#                 writer.write_table(table)
#                 total_processed += len(chunk)
#                 print(f"已写入 {total_processed} 行...")
#                 chunk = []  # 清空缓冲区
#
#         # 处理最后剩余的数据
#         if chunk:
#             df = pd.DataFrame(chunk)
#             table = pa.Table.from_pandas(df, schema=schema)
#             if writer is None:
#                 writer = pq.ParquetWriter(output_file, schema)
#             writer.write_table(table)
#             total_processed += len(chunk)
#
#     if writer:
#         writer.close()
#     print(f"✅ 处理完成！总计: {total_processed} 行，保存至: {output_file}")
#
#
# if __name__ == "__main__":
#     process_large_json_array("data/function.json", "data/test_binary.parquet")

import pandas as pd
import os


def convert_jsonl_to_parquet(input_file: str, output_file: str):
    """
    将 JSONL 文件转换为 Parquet 格式，并过滤、重命名列。
    - 保留 'func' 和 'target' 列
    - 将 'target' 重命名为 'vul'
    """
    print(f"正在处理: {input_file} ...")

    # 1. 读取 JSONL 文件
    # lines=True 表示每一行都是一个独立的 JSON 对象
    df = pd.read_json(input_file, lines=True)

    # 2. 检查所需列是否存在
    if 'func' not in df.columns or 'target' not in df.columns:
        raise ValueError(f"文件 {input_file} 缺少必要的 'func' 或 'target' 列！当前列: {df.columns.tolist()}")

    # 3. 过滤并重命名列
    df = df[['func', 'target']]
    df = df.rename(columns={'target': 'vul'})

    # 4. 保存为 Parquet 格式 (使用 pyarrow 引擎)
    df.to_parquet(output_file, engine='pyarrow', index=False)

    print(f"✅ 转换完成！已保存为: {output_file}")
    print(f"   数据集大小: {len(df)} 条样本")
    print("-" * 50)


if __name__ == "__main__":
    # 假设你的数据集分别存放在以下路径，你可以根据实际情况修改
    dataset_splits = ["train", "valid", "test"]

    input_dir = "./data/alert"  # 你的 jsonl 文件夹路径
    output_dir = "./data/alert"  # 输出的 parquet 文件夹路径

    os.makedirs(output_dir, exist_ok=True)

    for split in dataset_splits:
        input_path = os.path.join(input_dir, f"{split}.jsonl")
        output_path = os.path.join(output_dir, f"{split}.parquet")

        if os.path.exists(input_path):
            convert_jsonl_to_parquet(input_path, output_path)
        else:
            print(f"⚠️ 未找到文件: {input_path}，跳过。")