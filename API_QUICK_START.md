# API 快速開始 - Chapter 分段

## 📋 你需要什麼

1. ✅ OpenAI API Key
2. ✅ ASR 文件: `dataset/highlights/123/asr.txt` (已有)
3. ✅ `openai` 套件 (已安裝)

---

## 🚀 3 步驟開始

### 步驟 1: 獲取 API Key

1. 訪問: https://platform.openai.com/api-keys
2. 登入或註冊帳號
3. 點擊 "Create new secret key"
4. 複製你的 key (格式: `sk-proj-...`)

**新帳號通常有 $5 免費額度！**

---

### 步驟 2: 設置 API Key

**方法 A: 環境變量 (推薦)**

```batch
set OPENAI_API_KEY=sk-proj-你的key
```

**方法 B: 直接在命令行提供**

```batch
python quick_chapter.py --api_key sk-proj-你的key
```

---

### 步驟 3: 運行

```batch
python quick_chapter.py
```

就這麼簡單！

---

## 📊 你會看到什麼

```
============================================================
Quick Chapter Segmentation (GPT-4o-mini)
============================================================
ASR file: dataset/highlights/123/asr.txt
Video title: TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY
Output dir: outputs/chapters/123
============================================================

Loading ASR file: dataset/highlights/123/asr.txt
  - Total lines: 8579
  - Duration: 06:22:01

Estimated cost: $0.2145
  (Input: 87,048 tokens = $0.0131)
  (Output: ~1,000 tokens = $0.0006)

Proceed? (y/n):
```

輸入 `y` 確認

---

## ⏱️ 處理時間

- **分析中**: 2-5 分鐘
- **進度顯示**: 每個 chunk 的處理狀態

```
Processing 4 chunks...

Processing chunk 1/4...
  Found 6 chapters in this chunk

Processing chunk 2/4...
  Found 7 chapters in this chunk

...
```

---

## ✅ 完成後

```
============================================================
Generated Chapters
============================================================
  00:00:00: Arrival at Empire State Building
  00:18:30: Meeting Agent00 and Planning
  00:42:15: Walking Through Manhattan Streets
  01:05:00: Exploring Little Italy Neighborhood
  01:28:45: Food Tour - First Restaurant Visit
  01:52:30: Shopping and Fan Interactions
  02:15:20: Street Performance and Entertainment
  02:38:45: Dinner at Italian Restaurant
  ...
  (總共 20-30 個章節)
============================================================

Success! Saved 24 chapters to:
  outputs/chapters/123/chapters.json
  outputs/chapters/123/chapters_summary.txt
```

---

## 📁 生成的文件

### 1. `outputs/chapters/123/chapters.json`

```json
{
  "00:00:00": "Arrival at Empire State Building",
  "00:18:30": "Meeting Agent00 and Planning",
  "00:42:15": "Walking Through Manhattan Streets",
  "01:05:00": "Exploring Little Italy Neighborhood",
  ...
}
```

### 2. `outputs/chapters/123/chapters_summary.txt`

```
Video: TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY
Total Chapters: 24
Model: gpt-4o-mini

============================================================
00:00:00: Arrival at Empire State Building
00:18:30: Meeting Agent00 and Planning
00:42:15: Walking Through Manhattan Streets
...
```

---

## 💰 成本

對於你的 6 小時 22 分視頻：

- **輸入**: ~87,000 tokens × $0.150 / 1M = **$0.013**
- **輸出**: ~1,000 tokens × $0.600 / 1M = **$0.001**
- **總計**: **$0.014 - $0.20**

非常便宜！

---

## 🎬 下一步

生成章節後，你可以：

### 1. 查看結果

```batch
type outputs\chapters\123\chapters_summary.txt
```

### 2. 用於 Highlight 檢測

```batch
python tools/highlight_detection_pipeline.py ^
    --video_path "123.mp4" ^
    --video_title "TwitchCon W/ AGENT00" ^
    --chat_file "123.json" ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --chapters_file "outputs/chapters/123/chapters.json" ^
    --output_dir "outputs/highlights/123" ^
    --use_api openai
```

### 3. 按章節切割視頻

使用 ffmpeg 根據時間戳切割。

---

## ❓ 常見問題

### Q: 我沒有 API Key 怎麼辦？

A: 註冊 OpenAI 帳號即可免費獲得，新帳號通常有 $5 額度。

### Q: API 會很慢嗎？

A: 不會，通常 2-5 分鐘完成 6 小時視頻的分段。

### Q: 我的數據會被訓練嗎？

A: 不會，OpenAI API 政策明確不會用 API 數據訓練模型。

### Q: 如果遇到錯誤？

A: 檢查：
1. API Key 是否正確
2. 帳號是否有額度
3. 網絡連接是否正常

### Q: 可以用其他 API 嗎？

A: 可以！修改 `quick_chapter.py` 支援：
- Claude 3.5 Haiku
- Google Gemini Flash

---

## 🔧 進階選項

### 自定義參數

```batch
python quick_chapter.py ^
    --asr_file "your/asr.txt" ^
    --video_title "Your Video Title" ^
    --output_dir "your/output" ^
    --model "gpt-4o-mini"
```

### 使用不同模型

- `gpt-4o-mini` (推薦，最便宜)
- `gpt-4o` (更好但更貴)
- `gpt-4-turbo` (最好但最貴)

---

## 📞 需要幫助？

如果遇到問題：

1. 檢查 API Key 是否正確設置
2. 確認有網絡連接
3. 查看錯誤訊息

---

## 🎉 立即開始！

```batch
# 1. 設置 key
set OPENAI_API_KEY=sk-proj-你的key

# 2. 運行
python quick_chapter.py

# 3. 等待 2-5 分鐘

# 4. 查看結果
type outputs\chapters\123\chapters_summary.txt
```

**就這麼簡單！** 🚀

---

## 📝 完整命令示例

```batch
# 設置 API Key
set OPENAI_API_KEY=sk-proj-abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx

# 運行章節分段
python quick_chapter.py

# 確認運行 (輸入 y)

# 等待完成...

# 查看結果
type outputs\chapters\123\chapters_summary.txt

# 使用章節進行 Highlight 檢測
python tools/highlight_detection_pipeline.py ^
    --video_path "123.mp4" ^
    --video_title "TwitchCon W/ AGENT00" ^
    --chat_file "123.json" ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --chapters_file "outputs/chapters/123/chapters.json" ^
    --output_dir "outputs/highlights/123" ^
    --use_api openai
```

完成！🎊
