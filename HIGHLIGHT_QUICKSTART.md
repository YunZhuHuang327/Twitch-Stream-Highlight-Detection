# 🎯 Highlight Detection - 快速开始

将 Chapter-Llama 从章节生成改造为 Highlight 检测系统。

## 🚀 快速开始（5分钟）

### 1. 准备示例数据

```powershell
# 创建示例聊天数据和 ground truth
# 已提供示例文件在 examples/ 目录
Copy-Item examples\chat_example.json mydata\chat.json
Copy-Item examples\highlights_example.json mydata\highlights.json
```

### 2. 生成训练数据

```powershell
conda activate chapter-llama

python tools\prepare_highlight_data.py `
    --video_path "D:\your_video.mp4" `
    --chat_file "mydata\chat.json" `
    --highlights_file "mydata\highlights.json" `
    --output_dir "dataset\highlights"
```

### 3. 训练模型

```powershell
.\train_highlight.ps1
```

### 4. 运行推理

```powershell
# 编辑 run_inference_highlight.ps1 中的路径
.\run_inference_highlight.ps1
```

## 📚 文档

- **[HIGHLIGHT_DETECTION.md](HIGHLIGHT_DETECTION.md)** - 技术方案和架构
- **[HIGHLIGHT_USAGE_GUIDE.md](HIGHLIGHT_USAGE_GUIDE.md)** - 完整使用指南
- **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** - Windows 环境配置

## 📊 数据格式

### 聊天数据 (`chat.json`)

```json
[
  {
    "timestamp": "00:15:30",  // 或秒数: 930
    "user": "username",
    "message": "wow amazing!",
    "emotes": ["poggers"]  // 可选
  }
]
```

### Ground Truth (`highlights.json`)

```json
[
  {
    "start_time": "00:15:30",  // 或秒数: 930
    "end_time": "00:18:45",    // 或秒数: 1125
    "type": "exciting_moment",
    "description": "Team won the game"
  }
]
```

### 输出 Highlights

```json
[
  {
    "start_time": 930.0,
    "end_time": 1125.5,
    "start_time_str": "00:15:30",
    "end_time_str": "00:18:45",
    "duration": 195.5,
    "type": "exciting_moment",
    "description": "Team won the final round"
  }
]
```

## 🎬 Highlight 类型

- `exciting_moment`: 激动人心的时刻（游戏胜利、成就等）
- `funny_moment`: 搞笑时刻
- `emotional_moment`: 感人时刻
- `skill_showcase`: 技术展示
- `chat_peak`: 聊天高峰

## 🔧 关键参数

### 训练

- **Context Length**: 2048 (针对 16GB GPU 优化)
- **LoRA Rank**: 4
- **Batch Size**: 1 (gradient accumulation: 8)
- **Epochs**: 3

### 推理

- **Window Size**: 35,000 tokens (~2-3 小时视频)
- **Window Overlap**: 300 秒 (5 分钟)
- **Max Windows**: 100

## 📈 性能

**16GB RTX A4000**:
- 训练速度: ~500 samples/hour
- 推理速度: ~10 minutes/hour of video
- 内存使用: ~14GB

**6-8 小时视频**:
- ASR 提取: ~30-60 分钟
- Highlight 检测: ~60-90 分钟
- 预期检测: 10-30 个 highlights

## 🏗️ 项目结构

```
chapter-llama/
├── tools/
│   └── prepare_highlight_data.py      # 数据准备工具
├── src/
│   ├── data/
│   │   ├── highlight_data.py          # 数据加载
│   │   └── utils_highlights.py        # Prompt 生成
│   ├── test/
│   │   └── highlights_window.py       # 滑动窗口推理
│   └── utils/
│       └── metrics_highlight.py       # 评估指标
├── configs/
│   ├── data/highlight.yaml            # 数据配置
│   ├── model/llama3.2_1B_highlight.yaml  # 模型配置
│   └── experiment/highlight.yaml      # 实验配置
├── examples/
│   ├── chat_example.json              # 示例聊天数据
│   └── highlights_example.json        # 示例 highlights
├── inference_highlight.py             # 推理脚本
├── train_highlight.ps1                # 训练脚本
└── run_inference_highlight.ps1        # 推理运行脚本
```

## 📝 完整流程

### 1. 数据准备

```powershell
# 为每个视频准备数据
python tools\prepare_highlight_data.py `
    --video_path "video1.mp4" `
    --chat_file "chat1.json" `
    --highlights_file "highlights1.json" `
    --output_dir "dataset\highlights"

# 重复处理所有视频...

# 生成索引
python tools\prepare_highlight_data.py `
    --update_index `
    --output_dir "dataset\highlights"
```

### 2. 训练

```powershell
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
conda activate chapter-llama

python train.py `
    experiment=highlight `
    data=highlight `
    model=llama3.2_1B_highlight
```

### 3. 推理

```powershell
python inference_highlight.py "video.mp4" `
    --model "outputs\highlight\...\model_checkpoints" `
    --base_model "Llama-3.2-1B-Instruct" `
    --chat_file "chat.json" `
    --window_size 35000 `
    --overlap 300
```

### 4. 评估

```python
from src.utils.metrics_highlight import evaluate_video

evaluate_video(
    'outputs/inference/video_001/highlights.json',
    'dataset/highlights/video_001/highlights.json'
)
```

## 🎯 评估指标

- **Precision**: 预测准确性 (TP / (TP + FP))
- **Recall**: 检测覆盖率 (TP / (TP + FN))
- **F1 Score**: 综合指标
- **mAP**: 平均精度 (不同 IoU 阈值)
- **IoU**: 时间段重叠度

## 🔍 示例输出

```
🔍 开始检测 highlights (window_size=35000, overlap=300s)...
  Window 1: 00:00:00 - 02:30:00
    ✓ 检测到 3 个 highlights
  Window 2: 02:25:00 - 05:00:00
    ✓ 检测到 2 个 highlights
  Window 3: 04:55:00 - 07:30:00
    - 本窗口无 highlights

✅ 完成！共检测到 5 个 highlights
📊 合并重叠 highlights: 5 → 4

检测到 4 个 highlights:

  exciting_moment: 2
  funny_moment: 1
  skill_showcase: 1

详细结果:
  1. [00:15:30-00:18:45] (195s)
     exciting_moment: Team won the final round
  2. [00:45:20-00:47:30] (130s)
     funny_moment: Character fell through map
  3. [01:23:00-01:26:30] (210s)
     skill_showcase: Frame-perfect execution
  4. [02:15:45-02:17:00] (75s)
     exciting_moment: Comeback victory

结果已保存到: outputs/inference/video_001/highlights.json
```

## ❓ 常见问题

### Q: 没有聊天数据可以用吗？

A: 可以！仅用 ASR 也能工作：

```powershell
python inference_highlight.py video.mp4 --model ...
# 不提供 --chat_file 参数
```

### Q: GPU 内存不足？

A: 减小 context_length:

```yaml
# configs/model/llama3.2_1B_highlight.yaml
config_train:
  context_length: 1024  # 从 2048 降到 1024
```

### Q: 如何调整检测灵敏度？

A: 修改生成参数:

```python
# inference_highlight.py 中调整
temperature=0.5,  # 降低 = 更保守
top_p=0.9,
```

## 📞 获取帮助

- 技术细节: [HIGHLIGHT_DETECTION.md](HIGHLIGHT_DETECTION.md)
- 完整指南: [HIGHLIGHT_USAGE_GUIDE.md](HIGHLIGHT_USAGE_GUIDE.md)
- 环境配置: [WINDOWS_SETUP.md](WINDOWS_SETUP.md)

---

## 🎉 开始使用

```powershell
# 1. 准备数据
python tools\prepare_highlight_data.py --video_path "video.mp4" --chat_file "chat.json" --highlights_file "highlights.json" --output_dir "dataset\highlights"

# 2. 训练
.\train_highlight.ps1

# 3. 推理
.\run_inference_highlight.ps1
```

Good luck! 🚀
