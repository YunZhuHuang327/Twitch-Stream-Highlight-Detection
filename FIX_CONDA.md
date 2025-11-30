# 修復 Conda 啟動問題

## 問題

你遇到這個錯誤：
```
ModuleNotFoundError: No module named 'boltons'
```

這是因為 conda 的 base 環境損壞了，缺少 `boltons` 模組。

## 解決方案

### 方案 1: 直接使用 chapter-llama 環境（推薦，最簡單）

**不需要修復 conda**，直接使用環境的 Python：

```batch
.\run_chapter_direct.bat
```

這個腳本會直接調用：
```
C:\Users\Huang\Miniconda3\envs\chapter-llama\python.exe
```

完全繞過 conda activate！

---

### 方案 2: 修復 Conda Base 環境

如果你想修復 conda，執行以下步驟：

#### 步驟 1: 在 base 環境安裝 boltons

```batch
C:\Users\Huang\Miniconda3\python.exe -m pip install boltons
```

#### 步驟 2: 測試 conda

```batch
conda --version
```

#### 步驟 3: 如果還有問題，重新安裝 conda

```batch
C:\Users\Huang\Miniconda3\python.exe -m pip install --force-reinstall conda
```

---

### 方案 3: 使用 PowerShell 手動啟動

```powershell
# 設置環境路徑
$env:Path = "C:\Users\Huang\Miniconda3\envs\chapter-llama;" + $env:Path
$env:Path = "C:\Users\Huang\Miniconda3\envs\chapter-llama\Library\bin;" + $env:Path
$env:Path = "C:\Users\Huang\Miniconda3\envs\chapter-llama\Scripts;" + $env:Path

# 設置其他環境變量
$env:CONDA_DEFAULT_ENV = "chapter-llama"
$env:CONDA_PREFIX = "C:\Users\Huang\Miniconda3\envs\chapter-llama"

# 運行腳本
python chapter_from_asr_english.py --asr_file "dataset/highlights/123/asr.txt" --video_title "TwitchCon W/ AGENT00" --output_dir "outputs/chapters/123" --quantization 4bit
```

---

## 推薦方式對比

| 方案 | 難度 | 需要修復 Conda | 推薦度 |
|------|------|---------------|--------|
| **方案 1: run_chapter_direct.bat** | ⭐ | ❌ | ⭐⭐⭐⭐⭐ |
| 方案 2: 修復 Conda | ⭐⭐⭐ | ✅ | ⭐⭐ |
| 方案 3: PowerShell 手動 | ⭐⭐ | ❌ | ⭐⭐⭐ |

---

## 🎯 立即使用（推薦）

直接運行，不需要任何修復：

```batch
.\run_chapter_direct.bat
```

這個腳本會：
1. ✅ 直接使用 `C:\Users\Huang\Miniconda3\envs\chapter-llama\python.exe`
2. ✅ 設置所有必要的環境變量
3. ✅ 運行 chapter 分段腳本
4. ✅ 完全繞過損壞的 conda

---

## 驗證環境是否正確

檢查 chapter-llama 環境是否有所需的套件：

```batch
C:\Users\Huang\Miniconda3\envs\chapter-llama\python.exe -c "import torch; import transformers; import peft; print('All imports OK!')"
```

如果成功顯示 "All imports OK!"，就可以直接使用了！

---

## 為什麼會遇到這個問題？

1. **Conda base 環境損壞**
   - `boltons` 模組遺失
   - 可能是更新或安裝時出錯

2. **但 chapter-llama 環境是完好的**
   - 所有套件都在
   - 可以直接使用

3. **Conda activate 需要 base 環境**
   - 所以無法啟動
   - 但我們可以繞過它！

---

## 長期解決方案

### 選項 A: 保持現狀（推薦）

使用 `run_chapter_direct.bat`，不修復 conda。

**優點**:
- ✅ 立即可用
- ✅ 不需要折騰
- ✅ 更快（不需要 conda activate）

**缺點**:
- ❌ 每次都要用這個 batch 文件
- ❌ 其他需要 conda 的項目可能有問題

### 選項 B: 修復 Conda

修復 base 環境，恢復 conda activate 功能。

**優點**:
- ✅ 可以正常使用 conda activate
- ✅ 其他項目也能用

**缺點**:
- ❌ 需要時間修復
- ❌ 可能需要重裝 conda

---

## 🚀 快速開始

現在就試試：

```batch
.\run_chapter_direct.bat
```

如果成功，你會看到：
```
============================================================
Chapter-Llama: Generate chapters from ASR text
============================================================

Using environment: chapter-llama
Using Python: C:\Users\Huang\Miniconda3\envs\chapter-llama\python.exe

[OK] Loaded ASR file: dataset\highlights\123\asr.txt
  - Lines: 8579
  - Duration: 06:22:01

Loading model:
  - Base model: meta-llama/Llama-3.1-8B-Instruct
  - PEFT model: ...
  - Quantization: 4bit

Generating chapters...
```

---

## 常見問題

### Q: 為什麼不直接修復 conda？

A: 因為：
1. 修復可能很複雜
2. 直接使用環境更簡單
3. 效果完全一樣

### Q: 這樣會不會有問題？

A: 不會！直接使用環境的 Python 是完全合法且常見的做法。

### Q: 其他項目怎麼辦？

A: 也可以用同樣方式：
```batch
C:\Users\Huang\Miniconda3\envs\YOUR_ENV\python.exe your_script.py
```

### Q: 我還是想修復 conda

A: 可以，但建議先用 `run_chapter_direct.bat` 完成當前任務，之後再慢慢修復 conda。

---

## 總結

✅ **立即可用的解決方案**: `.\run_chapter_direct.bat`

這個方案：
- 完全繞過損壞的 conda
- 直接使用 chapter-llama 環境
- 所有功能正常
- 無需任何修復

**現在就試試吧！** 🎉
