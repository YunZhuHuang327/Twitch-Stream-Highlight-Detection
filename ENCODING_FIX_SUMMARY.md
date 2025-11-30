# 編碼問題修正總結

## 問題

在 Windows 系統使用 Chapter-Llama 時，遇到 Unicode 編碼錯誤：

```
UnicodeEncodeError: 'cp950' codec can't encode character '\U0001f4d6' in position 0: illegal multibyte sequence
```

這是因為：
1. Windows 控制台默認使用 CP950/CP936 編碼（繁體中文/簡體中文）
2. Python print 語句中的 emoji 和某些中文字符無法編碼
3. 即使使用 `chcp 65001` 也可能無法完全解決

---

## 解決方案

### 方案 1: 使用英文版本腳本（推薦用於本地模型）

創建了 `chapter_from_asr_english.py`，**完全避免編碼問題**。

**關鍵改進**:

1. **所有訊息改為英文**
   ```python
   # 原版 (中文)
   print("✓ 加载 ASR 文件")
   print("📖 Chapter-Llama: 从 ASR 文本生成章节")

   # 英文版
   safe_print("[OK] Loaded ASR file")
   safe_print("Chapter-Llama: Generate chapters from ASR text")
   ```

2. **safe_print() 函數**
   ```python
   def safe_print(text):
       """Safe print to avoid encoding errors"""
       try:
           print(text)
       except (UnicodeEncodeError, UnicodeDecodeError):
           # If encoding error, remove unencodable characters
           safe_text = str(text).encode('ascii', 'ignore').decode('ascii')
           print(safe_text)
   ```

3. **自動設置編碼**
   ```python
   if sys.platform == 'win32':
       try:
           sys.stdout.reconfigure(encoding='utf-8')
           sys.stderr.reconfigure(encoding='utf-8')
       except:
           pass
       os.environ['PYTHONIOENCODING'] = 'utf-8'
       os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
   ```

### 方案 2: 使用 API (quick_chapter.py)

最簡單的方法，完全不需要處理本地模型。

---

## 使用方法對比

### 英文版本 (本地模型)

```batch
# 運行
.\run_chapter_english.bat

# 或手動
python chapter_from_asr_english.py ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --video_title "TwitchCon W/ AGENT00" ^
    --output_dir "outputs/chapters/123"
```

**優點**:
- ✅ 完全免費
- ✅ 無編碼問題
- ✅ 本地運行，無需網絡
- ✅ 功能完全相同

**缺點**:
- ❌ 需要下載模型 (~3GB)
- ❌ 需要 GPU
- ❌ 訊息是英文（但不影響輸出的章節標題）

### API 版本

```batch
# 設置 API Key
set OPENAI_API_KEY=sk-xxx

# 運行
.\run_quick_chapter.bat
```

**優點**:
- ✅ 最簡單
- ✅ 無需下載模型
- ✅ 無需 GPU
- ✅ 無編碼問題

**缺點**:
- ❌ 需要花錢 (~$0.20)
- ❌ 需要網絡連接

---

## 輸出對比

### 控制台輸出

**英文版本**:
```
============================================================
Chapter-Llama: Generate chapters from ASR text
============================================================
ASR file: dataset/highlights/123/asr.txt
Video title: TwitchCon W/ AGENT00
Output dir: outputs/chapters/123
============================================================

[OK] Loaded ASR file: dataset\highlights\123\asr.txt
  - Lines: 8579
  - Duration: 06:12:30

Prompt statistics:
  - Prompt length: 532 chars
  - Transcript length: 245678 chars
  - Total length: 246210 chars
  - Estimated tokens: ~61552

Loading model:
  - Base model: meta-llama/Llama-3.1-8B-Instruct
  - PEFT model: asr-10k

Generating chapters...

============================================================
Generated Chapters
============================================================
  00:00:00: Arrival at Empire State Building
  00:15:30: Meeting Agent00 and Planning
  00:45:00: Exploring Little Italy - Food Tour
  01:30:00: Shopping and Fan Interactions
  ...
============================================================

[SUCCESS] Results saved:
  - Chapters: outputs\chapters\123\chapters.json
  - Full output: outputs\chapters\123\output_text.txt
  - Total chapters: 24
```

### 輸出檔案 (完全相同)

`outputs/chapters/123/chapters.json`:
```json
{
  "00:00:00": "Arrival at Empire State Building",
  "00:15:30": "Meeting Agent00 and Planning",
  "00:45:00": "Exploring Little Italy - Food Tour",
  "01:30:00": "Shopping and Fan Interactions",
  ...
}
```

**重要**: 章節標題仍然可以是中文，因為這是 LLM 生成的，不受控制台編碼影響！

---

## 檔案對照表

| 檔案 | 用途 | 編碼問題 |
|------|------|---------|
| `chapter_from_asr.py` | 原版（中文訊息） | ⚠️ 可能有問題 |
| `chapter_from_asr_english.py` | 英文版本 | ✅ 無問題 |
| `quick_chapter.py` | API 版本 | ✅ 無問題 |
| `run_chapter_english.bat` | 英文版啟動器 | ✅ 無問題 |
| `run_quick_chapter.bat` | API 版啟動器 | ✅ 無問題 |

---

## 推薦使用

### 如果你有 OpenAI API Key:
→ 使用 `quick_chapter.py` (最簡單)

### 如果你想用本地模型:
→ 使用 `chapter_from_asr_english.py` (無編碼問題)

### 如果你堅持用中文訊息:
→ 使用 `chapter_from_asr.py` (可能遇到編碼錯誤)

---

## 測試建議

1. **先測試英文版本**:
   ```batch
   .\run_chapter_english.bat
   ```

2. **如果成功，確認輸出**:
   - 檢查 `outputs/chapters/123/chapters.json`
   - 章節標題應該正確
   - 控制台訊息都是英文

3. **如果還有問題**:
   - 檢查是否有 GPU
   - 檢查模型是否下載成功
   - 使用 `quick_chapter.py` 作為替代

---

## 技術細節

### 為什麼英文版本有效？

1. **移除所有 emoji**: `✓ ❌ 🔄` → `[OK] [ERROR] [INFO]`
2. **只使用 ASCII 字符**: 所有 print 訊息都是英文
3. **safe_print() 兜底**: 即使有錯誤，也會自動降級為 ASCII
4. **自動設置編碼**: 腳本啟動時自動配置 UTF-8

### 為什麼輸出仍然可以是中文？

- 輸出到**檔案**使用 `encoding='utf-8'`，不受控制台限制
- LLM 生成的章節標題寫入 JSON，不經過 `print()`
- 只有控制台訊息需要是英文

---

## 總結

| 方案 | 難度 | 成本 | 編碼問題 | 推薦度 |
|------|------|------|----------|--------|
| API 版本 | ⭐ | $0.20 | ✅ 無 | ⭐⭐⭐⭐⭐ |
| 英文版本 | ⭐⭐ | 免費 | ✅ 無 | ⭐⭐⭐⭐ |
| 原版中文 | ⭐⭐⭐ | 免費 | ⚠️ 可能有 | ⭐⭐ |

**最佳實踐**:
- 快速測試 → 用 API
- 批量處理 → 用英文版本
- 需要中文訊息 → 修改控制台設置（不推薦）
