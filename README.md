# 🎬 Twitch Stream Highlight Detection

基於 LLM 的 Twitch 直播精彩片段自動檢測系統

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)


## 📋 專案簡介

本專案將 [Chapter-Llama](https://arxiv.org/abs/2504.00072) 框架改造為 Twitch 直播精彩片段（Highlight）檢測系統。透過分析：
- 📝 語音轉錄文本（ASR）
- 💬 聊天室訊息與彈幕
- 🎯 觀眾互動峰值

自動識別長達 6-8 小時直播中的精彩時刻。

## 📄 論文

本專案的研究成果已整理成論文：

**[Multimodal Fusion for Highlight Detection in Lifestyle Live Streams](./Multimodal_Fusion_for_Highlight_Detection_in_Lifestyle_Live_Streams.pdf)**

### ✨ 主要功能

- **長影片處理**：支援 6-8 小時的長直播影片
- **多模態分析**：結合 ASR、聊天室資料進行綜合分析
- **滑動窗口**：高效處理長影片，避免 GPU 記憶體溢出
- **API 支援**：支援 OpenAI / Groq API 快速推理
- **本地模型**：支援 Llama 3.2-1B 本地微調與推理

## 🚀 快速開始

### 環境需求

- Python 3.10+
- CUDA 11.8+ (GPU 推理)
- 16GB+ RAM
- 8GB+ VRAM (本地模型)

### 安裝

```bash
# 克隆專案
git clone https://github.com/YunZhuHuang327/Twitch-Stream-Highlight-Detection.git
cd Twitch-Stream-Highlight-Detection

# 建立虛擬環境
conda create -n highlight python=3.10
conda activate highlight

# 安裝依賴
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 安裝專案
pip install -e .
```

### 設定 API 金鑰（可選）

```bash
# 使用 OpenAI API
export OPENAI_API_KEY=your-key-here

# 或使用 Groq API (更快且免費)
export GROQ_API_KEY=your-key-here
```

## 📖 使用方法

### 方法 1：使用 API（推薦新手）

```bash
# 使用 OpenAI GPT-4o-mini
python quick_chapter.py \
    --asr_file "path/to/asr.txt" \
    --video_title "直播標題" \
    --api openai

# 使用 Groq (免費且快速)
python quick_chapter.py \
    --asr_file "path/to/asr.txt" \
    --video_title "直播標題" \
    --api groq
```

### 方法 2：完整 Pipeline

```bash
python tools/highlight_detection_pipeline.py \
    --video_path "video.mp4" \
    --video_title "TwitchCon with Agent00" \
    --chat_file "chat.json" \
    --output_dir "outputs/highlights" \
    --top_k 20
```

### 方法 3：本地模型推理

```bash
# 下載模型
python download_llama_base.py

# 執行推理
python inference_highlight.py \
    --video_id "123" \
    --model_path "Llama-3.2-1B-Instruct"
```

## 📁 專案結構

```
📦 Twitch-Stream-Highlight-Detection/
├── 📂 configs/              # Hydra 配置檔
│   ├── 📂 data/             # 資料載入配置
│   ├── 📂 model/            # 模型配置
│   └── 📂 experiment/       # 實驗配置
├── 📂 src/                  # 核心原始碼
│   ├── 📂 data/             # 資料處理模組
│   │   ├── highlight_data.py    # Highlight 資料載入
│   │   ├── utils_asr.py         # ASR 處理工具
│   │   └── prompt.py            # Prompt 模板
│   ├── 📂 models/           # 模型定義
│   │   ├── llama_finetune.py    # Llama 微調
│   │   └── llama_inference.py   # Llama 推理
│   ├── 📂 test/             # 測試腳本
│   └── 📂 utils/            # 工具函數
├── 📂 tools/                # 實用工具
│   ├── highlight_detection_pipeline.py  # 完整 Pipeline
│   ├── score_highlights_v4.py           # Highlight 評分
│   ├── prepare_highlight_data.py        # 資料準備
│   └── convert_chat_format.py           # 聊天格式轉換
├── 📜 quick_chapter.py      # 快速 API 推理
├── 📜 inference.py          # 本地模型推理
├── 📜 train.py              # 模型訓練
└── 📜 requirements.txt      # 依賴套件
```

## 🎯 輸出格式

### Highlight 檢測結果

```json
{
  "video_id": "123",
  "highlights": [
    {
      "start_time": "00:15:30",
      "end_time": "00:18:45",
      "type": "exciting_moment",
      "description": "Team won the game",
      "score": 0.95,
      "chat_intensity": 0.92
    },
    {
      "start_time": "01:23:00",
      "end_time": "01:26:30",
      "type": "funny_moment",
      "description": "Streamer made a hilarious mistake",
      "score": 0.88,
      "chat_intensity": 0.85
    }
  ]
}
```

### Highlight 類型

| 類型 | 說明 |
|------|------|
| `exciting_moment` | 精彩時刻、勝利、成就 |
| `funny_moment` | 搞笑時刻、意外、笑話 |
| `emotional_moment` | 感人或戲劇性時刻 |
| `skill_showcase` | 高超技術展示 |
| `chat_peak` | 聊天室高峰時刻 |

## 🔧 進階配置

### 滑動窗口設定

針對不同長度的影片，調整 `configs/data/highlight.yaml`：

```yaml
# 2-3 小時影片（推薦）
window_token_size: 35000
window_overlap: 300  # 5 分鐘重疊

# 6-8 小時影片
window_token_size: 70000
window_overlap: 600  # 10 分鐘重疊
```

### 模型訓練

```bash
# 訓練 Highlight 檢測模型
python train.py \
    experiment=highlight \
    data=highlight \
    model=llama3.2_1B_highlight \
    trainer.max_epochs=10
```

## 📊 效能指標

| 模型 | Precision | Recall | F1 Score |
|------|-----------|--------|----------|
| GPT-4o-mini | 0.82 | 0.78 | 0.80 |
| Llama 3.2-1B (fine-tuned) | 0.85 | 0.81 | 0.83 |
| Llama 3.2-1B + Chat | 0.88 | 0.84 | 0.86 |


### 摘要

本研究提出了一個多模態融合框架，用於 Lifestyle 類型直播的精彩片段自動檢測。系統整合了：
- 🎤 **語音模態**：透過 ASR 分析主播語音內容
- 💬 **聊天模態**：分析觀眾互動模式與彈幕強度
- 🎬 **視覺模態**：結合 VLM 進行場景理解

實驗結果顯示，多模態融合方法相比單一模態有顯著提升。

## 🙏 致謝

本專案基於以下研究成果：

- [Chapter-Llama](https://github.com/lucas-ventura/chapter-llama) - CVPR 2025
- [Llama 3.2](https://github.com/meta-llama/llama) - Meta AI

```bibtex
@inproceedings{ventura2025chapter,
  title={Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs},
  author={Ventura, Lucas and Yang, Antoine and Schmid, Cordelia and Varol, G{\"u}l},
  booktitle={CVPR},
  year={2025}
}
```


## 📧 聯絡方式

如有問題或建議，歡迎提交 Issue 或 Pull Request。


