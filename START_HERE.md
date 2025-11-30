# 🚀 開始使用 Chapter-Llama

## 你遇到的問題總結

1. ✅ **編碼問題** - 已修復（使用英文版腳本）
2. ✅ **模型下載** - 已完成（都在 cache 裡）
3. ✅ **模型載入問題** - 已修復（使用 4-bit 量化）
4. ✅ **Conda 啟動問題** - 已繞過（直接使用環境）

## 🎯 立即開始

### 一鍵運行（推薦）

```batch
.\run_chapter_direct.bat
```

這個腳本會：
- ✅ 直接使用 chapter-llama 環境（繞過 conda）
- ✅ 使用 4-bit 量化（避免記憶體問題）
- ✅ 處理所有編碼問題
- ✅ 設置所有環境變量

### 預期輸出

```
============================================================
Chapter-Llama: Generate chapters from ASR text
============================================================

Using environment: chapter-llama
Using Python: C:\Users\Huang\Miniconda3\envs\chapter-llama\python.exe

ASR file: dataset/highlights/123/asr.txt
Output: outputs/chapters/123/

[OK] Loaded ASR file: dataset\highlights\123\asr.txt
  - Lines: 8579
  - Duration: 06:22:01

Prompt statistics:
  - Prompt length: 439 chars
  - Transcript length: 348191 chars
  - Total length: 348630 chars
  - Estimated tokens: ~87157

[INFO] Downloading model: asr-10k
File adapter_model.safetensors found in cache at: ...
File adapter_config.json found in cache at: ...
All files loaded successfully

Loading model:
  - Base model: meta-llama/Llama-3.1-8B-Instruct
  - PEFT model: ...
  - Quantization: 4bit

Generating chapters...
[這裡會顯示進度，需要 3-5 分鐘]

============================================================
Generated Chapters
============================================================
  00:00:00: Arrival at Empire State Building
  00:18:30: Meeting with Agent00
  00:42:15: Walking through Manhattan
  01:05:00: Exploring Little Italy
  ... (總共約 20-30 個章節)
============================================================

[SUCCESS] Results saved:
  - Chapters: outputs\chapters\123\chapters.json
  - Full output: outputs\chapters\123\output_text.txt
  - Total chapters: 24

DONE!
```

---

## 📊 處理時間預估

- **模型載入**: 1-2 分鐘
- **生成章節**: 3-5 分鐘
- **總計**: **5-7 分鐘**

---

## 🆚 兩種方式對比

你現在有兩種方式可以選擇：

### 方式 1: 本地模型（免費）

```batch
.\run_chapter_direct.bat
```

| 項目 | 詳情 |
|------|------|
| 時間 | 5-7 分鐘 |
| 成本 | **免費** |
| 需要 | GPU (4GB+ VRAM) |
| 品質 | 優秀 |

### 方式 2: API（簡單）

```batch
set OPENAI_API_KEY=sk-your-key
.\run_quick_chapter.bat
```

| 項目 | 詳情 |
|------|------|
| 時間 | 2-5 分鐘 |
| 成本 | **$0.20** |
| 需要 | 網絡連接 |
| 品質 | 非常優秀 |

---

## 📁 生成的文件

成功後，你會得到：

### 1. chapters.json
```json
{
  "00:00:00": "Arrival at Empire State Building",
  "00:18:30": "Meeting with Agent00 and Initial Planning",
  "00:42:15": "Walking Through Manhattan Streets",
  "01:05:00": "Exploring Little Italy Neighborhood",
  "01:28:45": "Food Tour - First Restaurant",
  "01:52:30": "Shopping and Fan Interactions",
  ...
}
```

### 2. chapters_summary.txt
```
Video: TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY
Total Chapters: 24
Model: asr-10k

============================================================
00:00:00: Arrival at Empire State Building
00:18:30: Meeting with Agent00 and Initial Planning
00:42:15: Walking Through Manhattan Streets
...
```

### 3. output_text.txt
完整的 LLM 輸出（包含章節生成過程）

---

## 🔍 檢查結果

```batch
# 查看生成的章節
type outputs\chapters\123\chapters_summary.txt

# 查看 JSON
type outputs\chapters\123\chapters.json
```

---

## 🎬 下一步

生成章節後，你可以：

### 1. 用於 Highlight 檢測

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

### 2. 按章節切割視頻

使用 ffmpeg 根據章節時間戳切割視頻。

### 3. 分析章節分布

- 平均章節長度
- 章節主題分類
- 場景轉換頻率

---

## 🐛 常見問題

### Q1: CUDA out of memory

**症狀**:
```
RuntimeError: CUDA out of memory
```

**解決**: GPU 記憶體不足，使用 API 代替
```batch
.\run_quick_chapter.bat
```

### Q2: bitsandbytes 錯誤

**症狀**:
```
ModuleNotFoundError: No module named 'bitsandbytes'
```

**解決**: 安裝 bitsandbytes
```batch
C:\Users\Huang\Miniconda3\envs\chapter-llama\python.exe -m pip install bitsandbytes
```

### Q3: 運行很慢

**正常**: 第一次載入模型較慢（1-2分鐘），這是正常的。

### Q4: 還是遇到編碼錯誤

**解決**: 確保使用 `run_chapter_direct.bat`，不是手動運行。

---

## 📚 相關文檔

| 文檔 | 用途 |
|------|------|
| **[FIXED_LOCAL_MODEL.md](FIXED_LOCAL_MODEL.md)** | 本地模型修復詳情 |
| **[FIX_CONDA.md](FIX_CONDA.md)** | Conda 問題解決方案 |
| [CHAPTER_QUICKSTART.md](CHAPTER_QUICKSTART.md) | API vs 本地模型對比 |
| [ENCODING_FIX_SUMMARY.md](ENCODING_FIX_SUMMARY.md) | 編碼問題總結 |

---

## ✅ 檢查清單

在運行之前，確認：

- [x] 你有 NVIDIA GPU（至少 4GB VRAM）
- [x] 模型已下載（在 HuggingFace cache）
- [x] chapter-llama 環境已安裝所有套件
- [x] ASR 文件存在：`dataset/highlights/123/asr.txt`

如果全部打勾，就可以運行了！

---

## 🎉 現在就試試

```batch
.\run_chapter_direct.bat
```

等待 5-7 分鐘，你就會得到完整的章節分段結果！

如果成功，請查看：
```
outputs\chapters\123\chapters.json
```

---

## 💡 建議

### 第一次使用
→ 使用**本地模型**測試（免費）

### 批量處理多個視頻
→ 使用**本地模型**（省錢）

### 追求速度
→ 使用 **API**（2-5 分鐘）

### 離線環境
→ 使用**本地模型**（唯一選擇）

---

## 需要幫助？

如果遇到問題：

1. 檢查 [FIXED_LOCAL_MODEL.md](FIXED_LOCAL_MODEL.md) - 模型問題
2. 檢查 [FIX_CONDA.md](FIX_CONDA.md) - Conda 問題
3. 檢查 [ENCODING_FIX_SUMMARY.md](ENCODING_FIX_SUMMARY.md) - 編碼問題

或直接使用 API（最簡單）：
```batch
.\run_quick_chapter.bat
```

---

**祝你好運！🚀**
