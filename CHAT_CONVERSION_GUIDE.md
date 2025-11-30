# 聊天格式轉換指南

這個工具將 TwitchDownloader 導出的聊天記錄轉換為 `prepare_highlight_data.py` 所需的格式。

## 文件說明

- **工具**: [tools/convert_chat_format.py](tools/convert_chat_format.py)
- **表情貼映射表**: [emote_text.csv](emote_text.csv)

## 輸入格式

TwitchDownloader 導出的 JSON 格式：

```json
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
```

## 輸出格式

`prepare_highlight_data.py` 需要的格式：

```json
[
  {
    "timestamp": 8,
    "user": "ryancomebackhome",
    "message": "ASSEMBLE",
    "emotes": []
  },
  {
    "timestamp": 14,
    "user": "Jolt_m",
    "message": "exemFlushed  incoming",
    "emotes": ["exemFlushed"]
  },
  ...
]
```

## 使用方法

### 1. 單個文件轉換

```bash
python tools/convert_chat_format.py \
    --input_chat "D:/Highlight_groundtruth/分割/output/[10-18-25] ExtraEmily - Chat.json" \
    --emote_csv "D:/chapter-llama/emote_text.csv" \
    --output_chat "output/chat_converted.json"
```

### 2. 批量轉換整個目錄

```bash
python tools/convert_chat_format.py \
    --input_dir "D:/Highlight_groundtruth/分割/output" \
    --emote_csv "D:/chapter-llama/emote_text.csv" \
    --output_dir "converted_chats"
```

這會自動找到目錄中所有的 `*Chat.json` 文件（排除 `*Chat-cut.json`），並轉換它們。

## 表情貼處理

工具會自動識別兩種類型的表情貼：

1. **TwitchDownloader 已標記的表情貼**：如果 `fragments` 中包含 `emoticon` 字段
2. **CSV 映射表中的表情貼**：如果文字匹配 `emote_text.csv` 中的項目

### emote_text.csv 格式

這是一個簡單的文本文件，每行一個表情貼名稱：

```
exemClap
exemClean
exemEat
exemEGALUL
exemNod
...
```

## 輸出統計信息

轉換完成後，工具會顯示：

- 轉換的消息總數
- 識別的表情貼總數
- 轉換成功/失敗的文件數量（批量模式）

## 範例輸出

```
🔄 轉換聊天文件: [10-18-25] ExtraEmily - Chat.json
✓ 加載了 56 個表情貼映射
   檢測到 TwitchDownloader 格式 (包含 89112 條消息)
✅ 轉換完成!
   輸出文件: converted_chats/chat_converted.json
   消息總數: 89112
   表情貼總數: 21330
```

## 常見問題

### Q: 如果沒有 emote_text.csv 會怎樣？

程式仍會運行，但只會識別 TwitchDownloader 已標記的表情貼。自定義頻道表情貼可能無法被識別。

### Q: 轉換後的文件可以直接用於 prepare_highlight_data.py 嗎？

是的！輸出格式完全符合 `prepare_highlight_data.py` 的要求。

### Q: 批量轉換會覆蓋已存在的文件嗎？

是的，如果輸出文件已存在會被覆蓋。請確保輸出目錄正確。

## 後續步驟

轉換完成後，使用轉換後的聊天文件配合 `prepare_highlight_data.py`：

```bash
python tools/prepare_highlight_data.py \
    --video_path "video.mp4" \
    --chat_file "converted_chats/chat_converted.json" \
    --highlights_file "highlights.json" \
    --output_dir "dataset/highlights"
```

更多詳情請參考：
- [HIGHLIGHT_USAGE_GUIDE.md](HIGHLIGHT_USAGE_GUIDE.md)
- [prepare_highlight_data.py 文檔](tools/prepare_highlight_data.py)
