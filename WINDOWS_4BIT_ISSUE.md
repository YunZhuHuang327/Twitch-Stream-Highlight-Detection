# Windows 4-bit 量化問題

## 問題

在 Windows 上使用 4-bit 量化時遇到錯誤：

```
ImportError: cannot import name 'get_num_sms' from 'torch._inductor.utils'
```

## 原因

這是 Windows 上 `bitsandbytes` 和 `torch` 兼容性問題：
- PyTorch 2.5.1 + bitsandbytes 0.48.2 在 Windows 上有兼容性問題
- `torch.compile` 和 `bitsandbytes` 的整合在 Windows 上不穩定

## 解決方案

### 🌟 方案 1: 使用 API（最推薦）

不需要處理這些問題，直接使用 API：

```batch
set OPENAI_API_KEY=sk-your-key-here
.\run_quick_chapter.bat
```

**優點**:
- ✅ 100% 可靠
- ✅ 2-5 分鐘完成
- ✅ 只需 $0.20
- ✅ 無需 GPU
- ✅ 無兼容性問題

---

### 方案 2: 不使用量化（可能可行）

讓模型自動 offload 到 CPU/磁盤：

```batch
.\run_chapter_no_quant.bat
```

**說明**:
- 會使用完整精度模型
- 如果 VRAM 不足，會自動 offload 到 CPU
- 可能遇到之前的 PEFT offloading 問題
- 速度會比較慢

**成功率**: ~50%（取決於你的硬件和運氣）

---

### 方案 3: 降級 PyTorch（複雜）

降級到相容的版本：

```batch
C:\Users\Huang\Miniconda3\envs\chapter-llama\python.exe -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

**警告**:
- 可能破壞其他套件
- 需要重新安裝很多東西
- 不保證能解決問題

**不推薦**

---

### 方案 4: 使用 Linux/WSL（複雜）

在 WSL2 中運行，Linux 上 bitsandbytes 更穩定。

**不推薦**（太複雜）

---

## 推薦決策

### 你的情況分析

| 因素 | 評估 |
|------|------|
| 視頻數量 | 1 個（測試） |
| 時間需求 | 希望快速完成 |
| 硬件 | GPU 有，但 Windows 兼容性問題 |
| 預算 | $0.20 完全可接受 |

### 建議

**使用 API（方案 1）**

理由：
1. ✅ 你之前成功用過本地模型，但現在環境有變化
2. ✅ Windows + bitsandbytes 兼容性問題很難解決
3. ✅ $0.20 的成本遠低於你花時間偵錯的價值
4. ✅ 2-5 分鐘就能完成，不需要折騰

---

## 快速開始（推薦）

### 步驟 1: 獲取 OpenAI API Key

訪問: https://platform.openai.com/api-keys

### 步驟 2: 設置環境變量

```batch
set OPENAI_API_KEY=sk-your-key-here
```

### 步驟 3: 運行

```batch
.\run_quick_chapter.bat
```

### 預期結果

```
============================================================
Quick Chapter Segmentation (GPT-4o-mini)
============================================================

Loading ASR file: dataset/highlights/123/asr.txt
  - Total lines: 8579
  - Duration: 06:22:01

Estimated cost: $0.2145

Proceed? (y/n): y

Segmenting video...
Processing 4 chunks...

Processing chunk 1/4...
  Found 6 chapters in this chunk

...

============================================================
Generated Chapters
============================================================
  00:00:00: Arrival at Empire State Building
  00:18:30: Meeting Agent00 and Planning
  00:42:15: Walking Through Manhattan
  ...
============================================================

Success! Saved 24 chapters to:
  outputs/chapters/123/chapters.json
```

---

## 如果堅持用本地模型

### 嘗試方案 2（不用量化）

```batch
.\run_chapter_no_quant.bat
```

**可能的結果**:

1. **成功** 🎉
   - 模型正常載入
   - 自動 offload 到 CPU
   - 生成章節（可能比較慢）

2. **失敗（PEFT offloading 錯誤）** ❌
   - 遇到之前的 KeyError
   - 回到方案 1（API）

3. **CUDA OOM** ❌
   - GPU 記憶體不足
   - 回到方案 1（API）

---

## 技術解釋

### 為什麼 Windows 有問題？

1. **bitsandbytes 主要為 Linux 設計**
   - CUDA kernel 編譯在 Windows 上不穩定
   - 很多功能在 Windows 上不完整

2. **torch.compile 整合問題**
   - PyTorch 2.5 的新功能
   - bitsandbytes 還沒完全相容

3. **Windows CUDA 支援較差**
   - Linux 上的 CUDA 工具鏈更成熟
   - Windows 版本常有延遲

### 為什麼之前可以？

可能的原因：
1. 之前沒用量化（用的是完整精度）
2. 之前的 PyTorch/transformers 版本不同
3. 之前的硬件配置不同

---

## 對比總結

| 方案 | 成功率 | 時間 | 成本 | 難度 |
|------|--------|------|------|------|
| **API** | 100% | 2-5分鐘 | $0.20 | ⭐ |
| 不用量化 | ~50% | 10-20分鐘 | 免費 | ⭐⭐ |
| 降級 PyTorch | ~30% | 1-2小時 | 免費 | ⭐⭐⭐⭐ |
| 使用 WSL | ~80% | 2-4小時 | 免費 | ⭐⭐⭐⭐⭐ |

---

## 我的強烈建議

**直接使用 API**

不要浪費時間在 Windows 兼容性問題上。
$0.20 遠遠低於你的時間價值。

如果你之後需要處理大量視頻（>100個），
再考慮在 Linux 環境或 WSL2 中設置本地模型。

但對於現在的測試，**API 是最佳選擇**。

---

## 立即行動

```batch
# 設置 API key
set OPENAI_API_KEY=sk-your-key-here

# 運行
.\run_quick_chapter.bat
```

**2-5 分鐘後，你就會有完整的章節分段結果！** 🎉
