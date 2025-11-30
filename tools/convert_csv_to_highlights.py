"""
CSV 到 Highlight JSON 转换工具

功能:
    将包含 highlight 时间戳的 CSV 文件转换为训练所需的 JSON 格式。

CSV 文件格式:
    - 必须包含 'start_time' 和 'end_time' 列。
    - 可以包含可选的 'type' 列。
    - 时间戳可以是 "HH:MM:SS" 格式或秒数。

示例 CSV (`highlights.csv`):
    start_time,end_time,type
    00:15:30,00:18:45,exciting_moment
    01:23:00,01:26:30,funny_moment
    5000,5025,skill_showcase

使用方法:
    python tools/convert_csv_to_highlights.py \
        --csv_file "path/to/highlights.csv" \
        --output_file "path/to/highlights.json"
"""

import csv
import json
import argparse
from pathlib import Path

def convert_csv_to_json(csv_path: str, output_path: str):
    """
    读取 CSV 文件并将其转换为 highlight JSON 格式。

    Args:
        csv_path (str): 输入的 CSV 文件路径。
        output_path (str): 输出的 JSON 文件路径。
    """
    highlights = []
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # 检查必需的列是否存在
            if 'start_time' not in reader.fieldnames or 'end_time' not in reader.fieldnames:
                raise ValueError("CSV 文件必须包含 'start_time' 和 'end_time' 列。")

            for row in reader:
                highlight_item = {
                    "start_time": row['start_time'],
                    "end_time": row['end_time']
                }
                # 如果存在 'type' 列，则添加它
                if 'type' in row and row['type']:
                    highlight_item['type'] = row['type']
                
                highlights.append(highlight_item)

        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(highlights, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"✅ 成功将 {csv_path} 转换为 {output_path}")
        print(f"   共转换了 {len(highlights)} 个 highlights。")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {csv_path}")
    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 CSV 文件转换为 Highlight JSON 格式。")
    parser.add_argument("--csv_file", required=True, help="输入的 CSV 文件路径。")
    parser.add_argument("--output_file", required=True, help="输出的 JSON 文件路径。")
    args = parser.parse_args()

    convert_csv_to_json(args.csv_file, args.output_file)