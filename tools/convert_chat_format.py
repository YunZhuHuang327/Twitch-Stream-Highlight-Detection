"""
聊天格式转换工具：将 TwitchDownloader 的聊天记录转换为 prepare_highlight_data.py 所需的格式

使用方法:
    # 单个文件转换
    python tools/convert_chat_format.py \
        --input_chat "D:/Highlight_groundtruth/分割/output/[10-18-25] ExtraEmily - Chat.json" \
        --emote_csv "D:/chapter-llama/emote_text.csv" \
        --output_chat "output/chat_converted.json"

    # 批量转换目录中所有 Chat.json 文件
    python tools/convert_chat_format.py \
        --input_dir "D:/Highlight_groundtruth/分割/output" \
        --emote_csv "D:/chapter-llama/emote_text.csv" \
        --output_dir "output/converted_chats"
"""

import json
import csv
import argparse
from pathlib import Path
import re
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')


def load_emote_map(csv_path):
    """
    從 CSV 文件加載表情貼映射表

    預期格式 (無表頭):
    exemClap
    exemClean
    ...

    返回: dict { "exemClap": "exemClap", ... }
    """
    emote_map = {}
    try:
        with open(csv_path, mode='r', encoding='utf-8') as infile:
            for line in infile:
                emote_text = line.strip()
                if emote_text:
                    # 將表情貼文字作為鍵和值
                    emote_map[emote_text] = emote_text

        print(f"✓ 加載了 {len(emote_map)} 個表情貼映射")
        return emote_map

    except FileNotFoundError:
        print(f"⚠️  警告: 未找到表情貼映射文件 {csv_path}")
        print(f"   將無法識別表情貼，所有 emotes 欄位將為空")
        return {}


def extract_emotes_from_fragments(fragments, emote_map):
    """
    從 TwitchDownloader 的 message.fragments 中提取表情貼

    TwitchDownloader 格式:
    "fragments": [
        {"text": "exemFlushed", "emoticon": {"emoticon_id": "emotesv2_..."}}
        {"text": "  incoming", "emoticon": null}
    ]

    返回: ["exemFlushed"]
    """
    emotes = []

    for fragment in fragments:
        text = fragment.get('text', '')
        emoticon = fragment.get('emoticon')

        # 如果 fragment 有 emoticon 字段，說明這是表情貼
        if emoticon is not None:
            emotes.append(text)
        # 也檢查 text 是否在表情貼映射表中
        elif text in emote_map:
            emotes.append(text)

    return emotes


def convert_chat(input_path, output_path, emote_csv_path):
    """
    轉換單個 TwitchDownloader 聊天記錄文件

    輸入格式 (TwitchDownloader):
    {
        "FileInfo": {...},
        "streamer": {...},
        "video": {...},
        "comments": [
            {
                "_id": "...",
                "created_at": "2025-10-17T19:18:55.995Z",
                "content_offset_seconds": 8,
                "commenter": {
                    "display_name": "ryancomebackhome",
                    ...
                },
                "message": {
                    "body": "ASSEMBLE",
                    "fragments": [
                        {"text": "ASSEMBLE", "emoticon": null}
                    ],
                    ...
                }
            },
            ...
        ]
    }

    輸出格式 (prepare_highlight_data.py 需要的格式):
    [
        {
            "timestamp": 8,
            "user": "ryancomebackhome",
            "message": "ASSEMBLE",
            "emotes": []
        },
        ...
    ]
    """
    print(f"\n🔄 轉換聊天文件: {Path(input_path).name}")

    # 1. 加載表情貼映射表
    emote_map = load_emote_map(emote_csv_path)

    # 2. 讀取原始聊天文件
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 錯誤: 無法解析 JSON 文件 {input_path}")
        print(f"   {e}")
        return False

    # 3. 檢查並提取 comments
    if isinstance(source_data, dict) and 'comments' in source_data:
        source_messages = source_data['comments']
        print(f"   檢測到 TwitchDownloader 格式 (包含 {len(source_messages)} 條消息)")
    elif isinstance(source_data, list):
        source_messages = source_data
        print(f"   檢測到列表格式 (包含 {len(source_messages)} 條消息)")
    else:
        print(f"❌ 錯誤: 無法識別的聊天記錄格式")
        return False

    # 4. 轉換每條消息
    converted_messages = []

    for i, msg in enumerate(source_messages):
        try:
            # 提取時間戳 (優先使用 content_offset_seconds)
            timestamp = msg.get('content_offset_seconds')
            if timestamp is None:
                # 如果沒有 content_offset_seconds，嘗試解析 created_at
                print(f"⚠️  警告: 消息 {i} 缺少 content_offset_seconds")
                timestamp = 0

            # 提取用戶名
            user = msg.get('commenter', {}).get('display_name', 'anonymous')

            # 提取消息文本
            message_body = msg.get('message', {}).get('body', '')

            # 提取表情貼
            fragments = msg.get('message', {}).get('fragments', [])
            emotes = extract_emotes_from_fragments(fragments, emote_map)

            # 創建新格式的消息
            new_msg = {
                "timestamp": timestamp,
                "user": user,
                "message": message_body,
                "emotes": emotes
            }
            converted_messages.append(new_msg)

        except Exception as e:
            print(f"⚠️  警告: 處理消息 {i} 時出錯: {e}")
            continue

    # 5. 按時間戳排序
    converted_messages.sort(key=lambda x: x['timestamp'])

    # 6. 寫入輸出文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_messages, f, indent=2, ensure_ascii=False)

    # 7. 統計信息
    total_emotes = sum(len(msg['emotes']) for msg in converted_messages)
    print(f"✅ 轉換完成!")
    print(f"   輸出文件: {output_path}")
    print(f"   消息總數: {len(converted_messages)}")
    print(f"   表情貼總數: {total_emotes}")

    return True


def batch_convert(input_dir, output_dir, emote_csv_path):
    """
    批量轉換目錄中所有的 Chat.json 文件
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有 Chat.json 文件（排除 Chat-cut.json）
    chat_files = []
    for file in input_dir.glob("*Chat.json"):
        if "Chat-cut.json" not in file.name:
            chat_files.append(file)

    if not chat_files:
        print(f"❌ 在 {input_dir} 中沒有找到任何 Chat.json 文件")
        return

    print(f"\n📁 找到 {len(chat_files)} 個聊天文件")
    print("="*60)

    success_count = 0
    fail_count = 0

    for i, chat_file in enumerate(chat_files, 1):
        print(f"\n[{i}/{len(chat_files)}]")

        # 生成輸出文件名（簡化文件名）
        output_filename = f"{chat_file.stem}_converted.json"
        output_path = output_dir / output_filename

        # 轉換文件
        if convert_chat(chat_file, output_path, emote_csv_path):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "="*60)
    print(f"📊 批量轉換完成!")
    print(f"   成功: {success_count} 個文件")
    print(f"   失敗: {fail_count} 個文件")
    print(f"   輸出目錄: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="將 TwitchDownloader 聊天記錄轉換為 prepare_highlight_data.py 所需的格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 單個文件轉換:
   python tools/convert_chat_format.py \\
       --input_chat "D:/Highlight_groundtruth/分割/output/[10-18-25] ExtraEmily - Chat.json" \\
       --emote_csv "D:/chapter-llama/emote_text.csv" \\
       --output_chat "output/chat_converted.json"

2. 批量轉換:
   python tools/convert_chat_format.py \\
       --input_dir "D:/Highlight_groundtruth/分割/output" \\
       --emote_csv "D:/chapter-llama/emote_text.csv" \\
       --output_dir "output/converted_chats"
        """
    )

    # 單個文件模式
    parser.add_argument('--input_chat',
                       help="單個輸入聊天記錄 JSON 文件路徑")
    parser.add_argument('--output_chat',
                       help="單個輸出 JSON 文件路徑")

    # 批量模式
    parser.add_argument('--input_dir',
                       help="包含多個 Chat.json 文件的輸入目錄")
    parser.add_argument('--output_dir',
                       help="批量轉換的輸出目錄")

    # 共用參數
    parser.add_argument('--emote_csv', required=True,
                       help="表情貼文字映射 CSV 文件路徑")

    args = parser.parse_args()

    # 驗證參數
    if args.input_chat and args.output_chat:
        # 單個文件模式
        convert_chat(args.input_chat, args.output_chat, args.emote_csv)

    elif args.input_dir and args.output_dir:
        # 批量模式
        batch_convert(args.input_dir, args.output_dir, args.emote_csv)

    else:
        print("❌ 錯誤: 請指定以下任一組合:")
        print("   1. --input_chat 和 --output_chat (單個文件)")
        print("   2. --input_dir 和 --output_dir (批量轉換)")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
