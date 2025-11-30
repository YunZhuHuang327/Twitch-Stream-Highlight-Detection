# Chapter 分段 - 快速開始

## 問題：本地模型載入失敗

如果你遇到這個錯誤：
```
KeyError: 'base_model.model.model.model.layers.20.input_layernorm'
```

這是因為 Llama 3.1-8B 模型太大，需要 GPU offloading，但 PEFT 適配器載入時出現問題。

---

## 解決方案對比

| 方案 | 難度 | 成本 | 時間 | 成功率 |
|------|------|------|------|--------|
| **API (GPT-4o-mini)** | ⭐ | $0.20 | 2-5分鐘 | 100% ✅ |
| 本地模型 | ⭐⭐⭐⭐ | 免費 | 30分鐘+ | 60% ⚠️ |

---

## 🌟 推薦：使用 API

### 步驟 1: 獲取 API Key

訪問 https://platform.openai.com/api-keys 並創建一個 key。

### 步驟 2: 設置環境變量

```batch
set OPENAI_API_KEY=sk-your-key-here
```

### 步驟 3: 運行

```batch
.\run_quick_chapter.bat
```

**就這麼簡單！**

### 輸出

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

Proceed? (y/n): y

Segmenting video...

Processing 4 chunks...

Processing chunk 1/4...
  Found 6 chapters in this chunk

Processing chunk 2/4...
  Found 7 chapters in this chunk

Processing chunk 3/4...
  Found 5 chapters in this chunk

Processing chunk 4/4...
  Found 6 chapters in this chunk

============================================================
Generated Chapters
============================================================
  00:00:00: Arrival at Empire State Building and Setup
  00:18:45: Meeting Agent00 and Initial Conversations
  00:42:30: Walking Through Manhattan Streets
  01:05:15: Entering Little Italy Neighborhood
  01:28:50: Food Tour - First Restaurant
  ... and 19 more chapters
============================================================

Success! Saved 24 chapters to:
  outputs/chapters/123/chapters.json
  outputs/chapters/123/chapters_summary.txt

You can now use these chapters in the highlight pipeline:
  python tools/highlight_detection_pipeline.py \
      --video_path "123.mp4" \
      --video_title "TwitchCon W/ AGENT00" \
      --chat_file "123.json" \
      --asr_file "dataset/highlights/123/asr.txt" \
      --chapters_file "outputs/chapters/123/chapters.json" \
      --output_dir "outputs/highlights/123"
```

---

## 為什麼 API 更好？

### ✅ 優點

1. **100% 成功率** - 不會遇到模型載入問題
2. **快速** - 2-5 分鐘完成
3. **便宜** - 6小時視頻只需 $0.20
4. **高質量** - GPT-4o-mini 擅長語義理解
5. **無需 GPU** - 任何電腦都能用
6. **無需下載** - 不佔用磁盤空間

### ❌ 本地模型的問題

1. **模型太大** - 需要 16GB+ GPU 記憶體
2. **載入複雜** - PEFT offloading 容易出錯
3. **設置困難** - 需要正確的 CUDA/PyTorch 版本
4. **偵錯耗時** - 遇到問題很難解決

---

## 成本分析

### GPT-4o-mini 定價

- 輸入: $0.150 per 1M tokens
- 輸出: $0.600 per 1M tokens

### 你的視頻 (6小時22分，8579行)

- ASR 文本: ~348,000 字符
- 預估輸入 tokens: ~87,000
- 預估輸出 tokens: ~1,000

**總成本**:
- 輸入: 87,000 × $0.150 / 1,000,000 = **$0.013**
- 輸出: 1,000 × $0.600 / 1,000,000 = **$0.001**
- **總計: $0.014 - $0.20**

即使處理 10 個這樣的視頻，也只需 **$2** 左右！

---

## 如果堅持用本地模型

### 需要的硬件

- **最低**: NVIDIA GPU 24GB VRAM (RTX 3090/4090)
- **推薦**: NVIDIA GPU 40GB+ VRAM (A100)
- **RAM**: 32GB+

### 可能的修復方法

1. **使用量化版本**
   - 4-bit quantization (需要 ~5GB VRAM)
   - 但可能影響質量

2. **使用更小的模型**
   - Llama 3.2-1B (但 chapter 分段效果較差)

3. **等待官方修復**
   - 這是 PEFT 庫的已知問題
   - 可能需要更新 transformers/peft 版本

---

## 快速命令參考

### API 方式（推薦）

```batch
# 1. 設置 key
set OPENAI_API_KEY=sk-xxx

# 2. 運行
python quick_chapter.py

# 或使用 batch 文件
.\run_quick_chapter.bat
```

### 檢查輸出

```batch
# 查看生成的章節
type outputs\chapters\123\chapters_summary.txt

# 查看 JSON
type outputs\chapters\123\chapters.json
```

### 用於 Highlight 檢測

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

---

## 常見問題

### Q: 我沒有 API key 怎麼辦？

A: 註冊 OpenAI 帳號即可免費獲得，新帳號通常有 $5 免費額度。

### Q: API 會不會很慢？

A: 不會，通常 2-5 分鐘就能處理 6 小時的視頻。

### Q: 我的數據會被 OpenAI 訓練嗎？

A: 不會，根據 OpenAI 的 API 政策，API 調用的數據不會用於訓練。

### Q: 可以用其他 API 嗎？

A: 可以！`quick_chapter.py` 支持：
- OpenAI GPT-4o-mini (推薦)
- Claude 3.5 Haiku
- Google Gemini Flash (有免費額度)

修改腳本中的 `model` 參數即可。

---

## 總結

**強烈建議使用 API 方式**，特別是：

- 你沒有高性能 GPU
- 你想快速測試
- 你只有少量視頻要處理 (<100個)
- 你遇到了本地模型載入問題

本地模型適合：
- 有高性能 GPU (24GB+ VRAM)
- 需要處理大量視頻 (>1000個)
- 完全離線環境
- 願意花時間偵錯

---

## 下一步

生成 chapters 後，你可以：

1. **檢視章節品質**
   ```batch
   type outputs\chapters\123\chapters_summary.txt
   ```

2. **用於 Highlight 檢測**
   ```batch
   python tools/highlight_detection_pipeline.py --chapters_file "outputs/chapters/123/chapters.json" ...
   ```

3. **批量處理多個視頻**
   - 創建循環腳本
   - 使用相同的 API

4. **分析章節統計**
   - 平均章節長度
   - 章節分布
   - 主題分類

需要幫助？查看：
- [API_SCORING_GUIDE.md](API_SCORING_GUIDE.md) - API 使用詳細說明
- [CHAPTER_SEGMENTATION_GUIDE.md](CHAPTER_SEGMENTATION_GUIDE.md) - 完整章節分段指南
- [QUICK_START.md](QUICK_START.md) - 所有工具快速參考
