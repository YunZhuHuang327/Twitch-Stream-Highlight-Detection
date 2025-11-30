# 🎯 Highlight 检测完整实施指南

## 📚 目录
1. [数据准备](#数据准备)
2. [训练模型](#训练模型)
3. [运行推理](#运行推理)
4. [评估结果](#评估结果)
5. [常见问题](#常见问题)

---

## 📊 数据准备

### Step 1: 准备聊天数据

创建 `chat.json` 文件，格式如下：

```json
[
  {
    "timestamp": "00:15:30",
    "user": "user123",
    "message": "wow amazing play!",
    "emotes": ["poggers", "fire"]
  },
  {
    "timestamp": 930,
    "user": "user456",
    "message": "笑死了哈哈哈",
    "emotes": ["lol"]
  }
]
```

**说明**:
- `timestamp`: 可以是 `"HH:MM:SS"` 格式或秒数（整数/浮点数）
- `user`: 用户名（可选）
- `message`: 聊天内容
- `emotes`: 表情/关键词列表（可选）

### Step 2: 准备 Ground Truth

创建 `highlights.json` 文件：

```json
[
  {
    "start_time": "00:15:30",
    "end_time": "00:18:45",
    "type": "exciting_moment",
    "description": "Team won the final round"
  },
  {
    "start_time": 4980,
    "end_time": 5190,
    "type": "funny_moment",
    "description": "Streamer made a hilarious mistake"
  }
]
```

**Highlight 类型**:
- `exciting_moment`: 激动人心的时刻
- `funny_moment`: 搞笑时刻
- `emotional_moment`: 感人时刻
- `skill_showcase`: 技术展示
- `chat_peak`: 聊天高峰

### Step 3: 生成训练数据

```powershell
# 为单个视频生成数据
python tools/prepare_highlight_data.py `
    --video_path "D:\videos\stream_001.mp4" `
    --chat_file "D:\data\chat_001.json" `
    --highlights_file "D:\data\highlights_001.json" `
    --output_dir "dataset/highlights"

# 处理多个视频
# 循环处理目录中的所有视频
$videos = Get-ChildItem "D:\videos\*.mp4"
foreach ($video in $videos) {
    $video_id = $video.BaseName
    python tools/prepare_highlight_data.py `
        --video_path $video.FullName `
        --chat_file "D:\data\chat_${video_id}.json" `
        --highlights_file "D:\data\highlights_${video_id}.json" `
        --output_dir "dataset/highlights"
}

# 生成数据集索引
python tools/prepare_highlight_data.py `
    --update_index `
    --output_dir "dataset/highlights"
```

**生成的目录结构**:
```
dataset/highlights/
├── index.json
└── video_001/
    ├── asr.txt          # ASR 转录
    ├── asr.json         # 带时间戳的 ASR
    ├── chat.json        # 聊天数据
    ├── highlights.json  # Ground truth
    ├── duration.txt     # 视频时长
    └── metadata.json    # 元数据
```

---

## 🚀 训练模型

### Step 1: 检查数据

```powershell
# 查看数据集索引
Get-Content dataset/highlights/index.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### Step 2: 修改配置（可选）

编辑 `configs/data/highlight.yaml`:

```yaml
# 调整这些参数
train_subset: train  # 训练集名称
val_subset: val      # 验证集名称
max_length: 2048     # 根据你的 GPU 调整
window_size: 35000   # Window token 大小
window_overlap: 300  # 重叠秒数
```

### Step 3: 开始训练

```powershell
# 使用提供的脚本
.\train_highlight.ps1

# 或手动运行
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
conda activate chapter-llama
python train.py experiment=highlight
```

**训练配置 (针对 16GB GPU 优化)**:
- Context length: 2048
- Batch size: 1
- Gradient accumulation: 8 (等效 batch size = 8)
- LoRA rank: 4
- BF16 混合精度
- Gradient checkpointing

**预期训练时间**:
- 1k 视频: ~4-6 小时
- 5k 视频: ~20-30 小时

### Step 4: 监控训练

```powershell
# 查看日志
Get-Content outputs/highlight/Llama-3.2-1B-Instruct-Highlight/highlight/highlight_detection/train/default/train.log -Tail 50 -Wait
```

---

## 🎬 运行推理

### 方法 1: 使用脚本（推荐）

编辑 `run_inference_highlight.ps1`:

```powershell
# 修改这些路径
$VIDEO_PATH = "D:\your_video.mp4"
$CHAT_FILE = "D:\your_chat.json"  # 可选
$MODEL_PATH = "D:\chapter-llama\outputs\highlight\...\model_checkpoints"
```

运行:
```powershell
.\run_inference_highlight.ps1
```

### 方法 2: 手动运行

```powershell
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
conda activate chapter-llama

# 有聊天数据
python inference_highlight.py "D:\video.mp4" `
    --model "outputs/highlight/.../model_checkpoints" `
    --base_model "D:\chapter-llama\Llama-3.2-1B-Instruct" `
    --chat_file "D:\chat.json" `
    --window_size 35000 `
    --overlap 300

# 无聊天数据（仅用 ASR）
python inference_highlight.py "D:\video.mp4" `
    --model "outputs/highlight/.../model_checkpoints" `
    --base_model "D:\chapter-llama\Llama-3.2-1B-Instruct" `
    --window_size 35000 `
    --overlap 300
```

### 参数说明

- `--model`: 训练好的模型路径
- `--base_model`: 基础模型路径（如果使用 LoRA adapter）
- `--chat_file`: 聊天数据文件（可选）
- `--window_size`: 窗口大小（tokens），默认 35000
- `--overlap`: 窗口重叠（秒），默认 300（5分钟）
- `--device`: 设备 (cuda/cpu)
- `--output`: 输出文件路径

### 输出格式

`outputs/inference/VIDEO_ID/highlights.json`:

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

---

## 📊 评估结果

### Step 1: 运行评估

```powershell
conda activate chapter-llama

# 评估单个视频
python -c "
from src.utils.metrics_highlight import evaluate_video
evaluate_video(
    'outputs/inference/video_001/highlights.json',
    'dataset/highlights/video_001/highlights.json'
)
"
```

### Step 2: 批量评估

创建 `evaluate_all.py`:

```python
from pathlib import Path
from src.utils.metrics_highlight import evaluate_video, calculate_all_metrics
import json

# 所有视频的结果
all_results = []

videos = Path("dataset/highlights").glob("*/metadata.json")
for metadata_file in videos:
    video_id = metadata_file.parent.name
    
    pred_file = f"outputs/inference/{video_id}/highlights.json"
    gt_file = f"dataset/highlights/{video_id}/highlights.json"
    
    if Path(pred_file).exists():
        print(f"\n评估 {video_id}...")
        results = evaluate_video(pred_file, gt_file)
        all_results.append({
            'video_id': video_id,
            'results': results
        })

# 保存结果
with open("evaluation_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n✅ 评估完成！结果保存到 evaluation_results.json")
```

运行:
```powershell
python evaluate_all.py
```

### 评估指标说明

**Precision (精确率)**:
- 预测的 highlights 中有多少是正确的
- 公式: TP / (TP + FP)

**Recall (召回率)**:
- 真实的 highlights 中有多少被检测到
- 公式: TP / (TP + FN)

**F1 Score**:
- Precision 和 Recall 的调和平均
- 公式: 2 × (Precision × Recall) / (Precision + Recall)

**mAP (mean Average Precision)**:
- 在不同 IoU 阈值下的平均精度
- 常用阈值: 0.3, 0.5, 0.7

**IoU (Intersection over Union)**:
- 预测时间段和真实时间段的重叠程度
- IoU > 0.5 通常认为是正确匹配

---

## 🔧 常见问题

### Q1: 聊天数据格式不对怎么办？

A: 转换你的聊天数据：

```python
# 示例：从 Twitch IRC 日志转换
def convert_twitch_logs(log_file):
    import re
    from datetime import datetime
    
    chat_data = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析格式: [HH:MM:SS] username: message
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\] (\w+): (.+)', line)
            if match:
                timestamp, user, message = match.groups()
                chat_data.append({
                    'timestamp': timestamp,
                    'user': user,
                    'message': message
                })
    
    import json
    with open('chat.json', 'w', encoding='utf-8') as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)
```

### Q2: GPU 内存不足

A: 调整配置：

```yaml
# configs/model/llama3.2_1B_highlight.yaml
config_train:
  context_length: 1024  # 从 2048 减小到 1024
  gradient_accumulation_steps: 16  # 增加到 16
```

或使用更小的窗口：

```powershell
python inference_highlight.py video.mp4 --window_size 20000
```

### Q3: 没有聊天数据怎么办？

A: 可以仅用 ASR 训练和推理：

```powershell
# 训练时禁用聊天
# 编辑 configs/data/highlight.yaml:
# include_chat: false

# 推理时不提供 --chat_file 参数
python inference_highlight.py video.mp4 --model ...
```

### Q4: 如何调整检测灵敏度？

A: 修改推理参数：

```python
# 在 inference_highlight.py 中调整 temperature
outputs = model.generate(
    temperature=0.5,  # 降低 = 更保守，减少误报
    top_p=0.9,
    # ...
)
```

或后处理时调整 IoU 阈值：

```python
# 更高的 IoU 阈值 = 更严格的匹配
merged = merge_overlapping_highlights(highlights, iou_threshold=0.7)
```

### Q5: 6-8 小时视频太长怎么办？

A: 增加窗口重叠或使用两阶段检测：

```powershell
# 方法 1: 增加重叠
python inference_highlight.py video.mp4 --overlap 600  # 10 分钟重叠

# 方法 2: 先粗检测再精细化（需要修改代码）
# 参考 HIGHLIGHT_DETECTION.md 中的 "多阶段检测"
```

### Q6: 如何添加自定义 highlight 类型？

A: 修改配置和 prompt：

```yaml
# configs/data/highlight.yaml
highlight_types:
  - exciting_moment
  - funny_moment
  - my_custom_type  # 添加自定义类型
```

```python
# src/data/utils_highlights.py
# 在 prompt 中添加描述
"""
Highlight types:
- exciting_moment: ...
- funny_moment: ...
- my_custom_type: Your description here
"""
```

### Q7: 训练数据太少怎么办？

A: 使用数据增强：

1. **时间偏移**: 轻微调整 highlight 边界
2. **负样本**: 添加明确不是 highlight 的片段
3. **Few-shot 学习**: 在 prompt 中添加示例
4. **预训练微调**: 先在更大的数据集上预训练

---

## 📈 性能优化建议

### 1. 批量推理

修改 `inference_highlight.py` 支持批处理：

```python
# 处理多个窗口时批量推理
def batch_inference(prompts, batch_size=4):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        # ... batch inference
    return results
```

### 2. 缓存 ASR

第一次提取后会自动缓存到 `outputs/inference/VIDEO_ID/`

### 3. 并行处理

```powershell
# 使用 PowerShell 并行处理多个视频
$videos = Get-ChildItem "D:\videos\*.mp4"
$videos | ForEach-Object -Parallel {
    python inference_highlight.py $_.FullName
} -ThrottleLimit 2  # 同时处理 2 个视频
```

---

## 🎓 进阶技巧

### 1. 融合多个信号

在 `src/test/highlights_window.py` 中添加更多特征：

```python
def get_multimodal_features(video_id, start_time, end_time):
    features = {
        'asr': get_asr_features(),
        'chat': get_chat_features(),
        'audio': get_audio_features(),  # 音量、音调
        'visual': get_visual_features()  # 场景变化
    }
    return features
```

### 2. 使用更大的模型

```powershell
# 切换到 Llama-3.1-8B （需要更多 GPU 内存）
python train.py `
    experiment=highlight `
    model=llama3.1_8B_highlight
```

### 3. 集成到视频播放器

生成 FFMPEG 兼容的章节文件：

```python
def export_to_ffmpeg_chapters(highlights, output_file):
    with open(output_file, 'w') as f:
        f.write(";FFMETADATA1\n")
        for i, hl in enumerate(highlights):
            f.write(f"[CHAPTER]\n")
            f.write(f"TIMEBASE=1/1000\n")
            f.write(f"START={int(hl['start_time'] * 1000)}\n")
            f.write(f"END={int(hl['end_time'] * 1000)}\n")
            f.write(f"title={hl['type']}: {hl['description']}\n")
```

---

## 📞 获取帮助

如果遇到问题：

1. 检查 `train.log` 日志
2. 使用 `nvidia-smi` 监控 GPU
3. 参考 `HIGHLIGHT_DETECTION.md` 技术文档
4. 查看 `WINDOWS_SETUP.md` 环境配置

---

## ✅ 快速检查清单

开始前确认：

- [ ] 视频文件存在且可访问
- [ ] 聊天数据格式正确（如果使用）
- [ ] Ground truth highlights 已准备
- [ ] Conda 环境已激活
- [ ] GPU 驱动和 CUDA 正常
- [ ] 至少 50GB 可用磁盘空间
- [ ] 16GB GPU 内存（训练）或 8GB（推理）

开始训练/推理！🚀
