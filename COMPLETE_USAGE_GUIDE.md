# 长视频 Highlight 检测完整使用指南

## 📚 目录
1. [快速开始](#快速开始)
2. [数据准备](#数据准备)
3. [训练流程](#训练流程)
4. [推理使用](#推理使用)
5. [常见问题](#常见问题)

---

## 🚀 快速开始

### 前置要求
- Python 3.8+
- NVIDIA GPU (推荐 16GB+ 显存)
- Conda 环境: `chapter-llama`

### 环境激活
```powershell
conda activate chapter-llama
```

---

## 📦 数据准备

### 第一步：组织你的数据

创建以下目录结构：

```
D:/streaming_data/          # 你的数据根目录
├── videos/                  # 原始视频文件
│   ├── stream_001.mp4      # 6-8 小时的直播录像
│   ├── stream_002.mp4
│   └── stream_003.mp4
├── chats/                   # 聊天室记录
│   ├── stream_001.json
│   ├── stream_002.json
│   └── stream_003.json
└── highlights/              # Highlight 标注
    ├── stream_001.json
    ├── stream_002.json
    └── stream_003.json
```

### 第二步：准备聊天数据格式

`chats/stream_001.json`:
```json
[
  {
    "timestamp": "00:15:30",
    "user": "username",
    "message": "wow amazing!",
    "emotes": ["poggers", "fire"]
  },
  {
    "timestamp": 930,
    "user": "another_user",
    "message": "LMAO"
  }
]
```

**注意**: 时间戳可以是 `"HH:MM:SS"` 格式或秒数

### 第三步：标注 Highlights

`highlights/stream_001.json`:
```json
[
  {
    "start_time": "00:15:30",
    "end_time": "00:18:45",
    "type": "exciting_moment"
  },
  {
    "start_time": "01:23:00",
    "end_time": "01:26:30",
    "type": "funny_moment"
  }
]
```

**Highlight 类型**:
- `exciting_moment`: 激动时刻（精彩操作、胜利）
- `funny_moment`: 搞笑时刻（失误、幽默）
- `skill_showcase`: 技术展示（高难度操作）
- `emotional_moment`: 情感时刻（感人瞬间）
- `chat_peak`: 聊天高峰（弹幕爆炸）

📖 详细标注指南请查看: [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)

### 第四步：处理数据（单个视频测试）

```powershell
# 测试单个短视频（不分段）
python tools/prepare_highlight_data.py `
    --video_path "test_data/v1.mp4" `
    --chat_file "test_data/chat_v1.json" `
    --highlights_file "test_data/highlights_v1.json" `
    --output_dir "dataset/highlights" `
    --simplify
```

### 第五步：批量处理长视频（分段模式）

```powershell
# 批量处理 6-8 小时长视频
python tools/batch_prepare_highlights.py `
    --data_dir "D:/streaming_data" `
    --output_dir "dataset/highlights" `
    --segment_mode `
    --segment_window 1800 `
    --segment_overlap 300 `
    --simplify
```

**参数说明**:
- `--segment_window 1800`: 每个片段 30 分钟（1800 秒）
- `--segment_overlap 300`: 片段重叠 5 分钟（300 秒）
- `--simplify`: 使用简化标注格式（只有时间+类型，无描述）

**处理结果**:
```
dataset/highlights/
├── stream_001_seg000/      # 第 1 个 30 分钟片段
│   ├── asr.txt
│   ├── asr.json
│   ├── chat.json
│   ├── highlights.json
│   ├── duration.txt
│   └── metadata.json
├── stream_001_seg001/      # 第 2 个片段（与上一个重叠 5 分钟）
│   └── ...
├── stream_001_seg002/
│   └── ...
...
└── stream_003_seg011/      # 6 小时视频约生成 12 个片段
    └── ...
```

### 第六步：创建训练/验证集划分

创建 `dataset/docs/subset_data/train.json`:
```json
[
  "stream_001_seg000",
  "stream_001_seg001",
  "stream_001_seg002",
  "stream_002_seg000",
  "stream_002_seg001"
]
```

创建 `dataset/docs/subset_data/val.json`:
```json
[
  "stream_003_seg000",
  "stream_003_seg001"
]
```

创建 `dataset/docs/chapters.json`:
```json
{
  "stream_001_seg000": {
    "title": "Stream 001 - Segment 0",
    "duration": 1800
  },
  "stream_001_seg001": {
    "title": "Stream 001 - Segment 1",
    "duration": 1800
  }
}
```

**快速生成脚本**:
```python
import json
from pathlib import Path

# 自动生成子集文件
highlight_dir = Path("dataset/highlights")
all_segments = sorted([d.name for d in highlight_dir.iterdir() if d.is_dir()])

# 80% 训练，20% 验证
split_idx = int(len(all_segments) * 0.8)
train_segments = all_segments[:split_idx]
val_segments = all_segments[split_idx:]

# 保存训练集
with open("dataset/docs/subset_data/train.json", "w") as f:
    json.dump(train_segments, f, indent=2)

# 保存验证集
with open("dataset/docs/subset_data/val.json", "w") as f:
    json.dump(val_segments, f, indent=2)

# 生成 chapters.json
chapters = {}
for seg in all_segments:
    metadata_file = highlight_dir / seg / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
            chapters[seg] = {
                "title": f"{meta['original_video']} - Segment {meta['segment_index']}",
                "duration": meta['duration']
            }

with open("dataset/docs/chapters.json", "w", encoding='utf-8') as f:
    json.dump(chapters, f, indent=2, ensure_ascii=False)

print(f"✅ 训练集: {len(train_segments)} 个片段")
print(f"✅ 验证集: {len(val_segments)} 个片段")
```

---

## 🎓 训练流程

### 准备训练配置

检查 `configs/data/highlight.yaml`:
```yaml
_target_: src.data.highlight_data.HighlightDataModule
vidc_dir: ${paths.vidc_dir}
train_subset: train
val_subset: val
prompt: null
subset: null
data_flags:
  include_chat: true
  simplify_format: true
```

### 训练命令（生产环境）

创建 `run_train_production.bat`:
```batch
@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
call conda activate chapter-llama

python train.py ^
  data=highlight ^
  paths.output_dir=outputs/highlight_production ^
  model.config_train.model_name=D:/chapter-llama/Llama-3.2-1B-Instruct ^
  model.config_train.num_epochs=5 ^
  model.config_train.batch_size_training=1 ^
  model.config_train.gradient_accumulation_steps=8 ^
  model.config_train.context_length=4096 ^
  model.config_train.use_peft=True ^
  model.config_train.r=8 ^
  model.config_train.lora_alpha=16 ^
  model.config_train.lr=5e-5 ^
  model.config_train.output_dir=outputs/highlight_production/model ^
  model.config_train.enable_fsdp=False ^
  model.config_train.run_validation=True > train_production.log 2>&1

echo Training completed. Check train_production.log
pause
```

### 执行训练

```powershell
cmd /c run_train_production.bat
```

### 监控训练

```powershell
# 实时查看日志
Get-Content train_production.log -Wait -Tail 50

# 查看关键指标
Get-Content train_production.log | Select-String -Pattern "epoch|loss|perplexity"
```

### 预期训练时间

| 数据量 | 片段数 | 训练时间（5 epochs） | 显存使用 |
|--------|--------|---------------------|----------|
| 10 个视频 | ~100 片段 | 2-4 小时 | 11-13 GB |
| 30 个视频 | ~300 片段 | 6-12 小时 | 11-14 GB |
| 100 个视频 | ~1000 片段 | 20-40 小时 | 12-15 GB |

---

## 🔮 推理使用

### 对新视频进行 Highlight 检测

```powershell
# 使用训练好的模型
python inference_highlight.py `
    --video_path "D:/new_streams/stream_new.mp4" `
    --chat_file "D:/new_streams/stream_new_chat.json" `
    --model "outputs/highlight_production/model" `
    --output "results/stream_new_highlights.json"
```

### 长视频自动使用 Sliding Window

对于 6-8 小时长视频，推理脚本会自动：
1. 将视频切分为 30 分钟窗口
2. 对每个窗口独立推理
3. 合并所有检测到的 highlights
4. 去除重复的 highlight

**输出格式** (`results/stream_new_highlights.json`):
```json
[
  {
    "start_time": "00:15:30",
    "end_time": "00:18:45",
    "type": "exciting_moment",
    "confidence": 0.92
  },
  {
    "start_time": "01:23:00",
    "end_time": "01:26:30",
    "type": "funny_moment",
    "confidence": 0.87
  }
]
```

### 批量推理

```python
# batch_inference.py
import subprocess
from pathlib import Path

videos_dir = Path("D:/new_streams/videos")
model_path = "outputs/highlight_production/model"

for video in videos_dir.glob("*.mp4"):
    chat_file = video.parent.parent / "chats" / f"{video.stem}.json"
    output_file = Path("results") / f"{video.stem}_highlights.json"
    
    cmd = [
        "python", "inference_highlight.py",
        "--video_path", str(video),
        "--chat_file", str(chat_file),
        "--model", model_path,
        "--output", str(output_file)
    ]
    
    print(f"处理: {video.name}")
    subprocess.run(cmd)
```

---

## ❓ 常见问题

### Q1: 我的视频是 6-8 小时，GPU 会 OOM 吗？

**A**: 不会。系统会自动使用分段处理：
- **训练时**: 数据准备阶段已经将长视频切分成 30 分钟片段
- **推理时**: 使用 sliding window，每次只处理 30 分钟
- **显存使用**: 每个片段约 11-13 GB（16GB GPU 足够）

### Q2: 需要多少个视频才能训练？

对于 6-8 小时长视频：
- **最少**: 10 个视频（~100 个片段） - 可以验证概念
- **推荐**: 30-50 个视频（~300-500 个片段） - 基本可用
- **最佳**: 100+ 个视频（~1000+ 个片段） - 良好性能

每个 6 小时视频会生成约 12 个训练片段。

### Q3: 标注工作量太大怎么办？

**优先级策略**:
1. 先标注 10 个视频验证可行性
2. 使用聊天高峰辅助标注（查看 `chat.json` 的 `peak_moments`）
3. 专注于明显的 highlight，不追求完美
4. 标注质量 > 数量

**时间估算**:
- 熟练后每个 6 小时视频约 30-60 分钟标注
- 10 个视频约 5-10 小时工作量

### Q4: 可以不要 description 字段吗？

**A**: 可以！已经默认使用简化格式。

标注文件只需要：
```json
{
  "start_time": "00:15:30",
  "end_time": "00:18:45",
  "type": "exciting_moment"
}
```

不需要 `description` 字段。

### Q5: 训练时出现错误怎么办？

**常见错误**:

1. **显存不足 (OOM)**:
   ```
   解决方案:
   - 减小 context_length (4096 → 2048)
   - 增加 gradient_accumulation_steps (8 → 16)
   - 启用 gradient checkpointing
   ```

2. **训练集为空**:
   ```
   检查:
   - dataset/docs/subset_data/train.json 是否存在
   - 里面的视频 ID 是否与 dataset/highlights/ 中的文件夹匹配
   ```

3. **Chat 文件找不到**:
   ```
   确保每个视频都有对应的 chat.json 文件
   ```

### Q6: 如何评估模型效果？

```python
# 使用评估脚本
python tools/evaluate_highlights.py \
    --ground_truth "test_data/highlights_ground_truth.json" \
    --predictions "results/predictions.json" \
    --metrics "all"
```

**评估指标**:
- **Temporal IoU**: Highlight 时间重叠度
- **Precision**: 预测的 highlight 有多少是对的
- **Recall**: 真实的 highlight 有多少被找到
- **F1 Score**: Precision 和 Recall 的调和平均
- **mAP**: 平均精度

---

## 📊 性能优化建议

### 训练优化

1. **使用 LoRA** (已在生产配置中启用):
   - 减少训练参数
   - 节省显存
   - 加快训练速度

2. **调整 context length**:
   - 如果片段较短: `context_length=2048`
   - 标准片段: `context_length=4096`
   - 长片段/复杂对话: `context_length=8192`

3. **Gradient Accumulation**:
   - 模拟更大的 batch size
   - `gradient_accumulation_steps=8` 相当于 `batch_size=8`

### 推理优化

1. **Window Size 调整**:
   ```python
   # 对于聊天密集的视频，使用较小窗口
   --window_size 1200  # 20 分钟
   
   # 对于聊天稀疏的视频，使用较大窗口
   --window_size 2400  # 40 分钟
   ```

2. **并行推理**:
   - 使用批量推理脚本
   - 同时处理多个视频
   - 利用多 GPU（如果有）

---

## 🎯 最佳实践

1. **从小开始**: 先用 1-2 个视频测试完整流程
2. **逐步扩展**: 验证可行后再增加数据量
3. **质量优先**: 10 个高质量标注 > 30 个低质量标注
4. **定期评估**: 每训练一定量后在验证集上测试
5. **版本管理**: 保存不同版本的模型，比较效果
6. **文档记录**: 记录每次训练的配置和结果

---

## 📞 获取帮助

查看详细文档:
- [LONG_VIDEO_FAQ.md](LONG_VIDEO_FAQ.md) - 长视频处理 FAQ
- [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) - 标注指南
- [PRODUCTION_TRAINING_GUIDE.md](PRODUCTION_TRAINING_GUIDE.md) - 生产环境配置

---

**祝训练顺利！🚀**
