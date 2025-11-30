# Highlight Detection 常见问题解答

## 问题 1: type 和 description 的作用及是否可选

### 当前用途：

在标注文件中，`type` 和 `description` 字段的用途：

1. **`type`**（类型标签）：
   - 分类 highlight 的种类（exciting_moment, funny_moment, emotional_moment 等）
   - 在训练时作为**输出目标**的一部分
   - 格式：`[START_TIME-END_TIME] TYPE: DESCRIPTION`
   
2. **`description`**（描述）：
   - 详细说明这个 highlight 为什么重要
   - 在训练时作为**输出目标**的一部分
   - 帮助模型学习什么样的内容算是 highlight

### 是否可以省略？

**可以，但需要修改代码。** 目前有两种简化方案：

#### 方案 A：只保留时间段（最简单）

如果你只需要模型输出时间段，不需要分类和描述：

**标注格式简化为：**
```json
[
  {
    "start_time": "00:15:30",
    "end_time": "00:18:45"
  },
  {
    "start_time": "00:45:20",
    "end_time": "00:47:30"
  }
]
```

**输出格式改为：**
```
[00:15:30-00:18:45]
[00:45:20-00:47:30]
```

**需要修改的代码：**
1. `src/data/highlight_data.py` 第 185 行附近
2. `src/data/utils_highlights.py` 的 prompt 模板

#### 方案 B：只保留类型，不要描述

**标注格式：**
```json
[
  {
    "start_time": "00:15:30",
    "end_time": "00:18:45",
    "type": "exciting_moment"
  }
]
```

**输出格式：**
```
[00:15:30-00:18:45] exciting_moment
```

### 建议：

**保留 type，省略 description**（方案 B）

原因：
- `type` 可以帮助你后续筛选不同类型的 highlight
- 标注时只需要选择类型，不需要写详细描述，工作量小很多
- 模型输出仍然有结构化信息，便于后处理

---

## 问题 2: 文件存放位置和输出位置

### 训练数据存放位置：

```
dataset/
├── highlights/              ← 所有 highlight 训练数据的根目录
│   ├── video_001/          ← 每个视频一个文件夹（用视频 ID 命名）
│   │   ├── asr.txt         ← 自动生成（运行 prepare_highlight_data.py）
│   │   ├── asr.json        ← 自动生成（带时间戳的 ASR）
│   │   ├── chat.json       ← **你需要准备**
│   │   ├── highlights.json ← **你需要准备**（ground truth）
│   │   ├── duration.txt    ← 自动生成
│   │   └── metadata.json   ← 自动生成
│   ├── video_002/
│   │   └── ...
│   └── video_003/
│       └── ...
└── docs/
    ├── subset_data/
    │   ├── train.json       ← 训练集视频 ID 列表
    │   └── val.json         ← 验证集视频 ID 列表
    └── chapters.json        ← 视频元数据（标题、时长）
```

### 你需要准备的原始文件：

**建议在项目外的某个位置存放原始数据：**

```
D:/
└── streaming_data/          ← 你的原始数据存放处
    ├── videos/              ← 原始视频文件
    │   ├── stream_2024_01_15.mp4
    │   ├── stream_2024_01_16.mp4
    │   └── ...
    ├── chats/               ← 聊天室记录（你需要从平台导出）
    │   ├── stream_2024_01_15.json
    │   ├── stream_2024_01_16.json
    │   └── ...
    └── highlights/          ← 标注的 highlight（你需要手工标注或用工具）
        ├── stream_2024_01_15.json
        ├── stream_2024_01_16.json
        └── ...
```

### 数据准备流程：

```powershell
# 对每个视频运行数据准备脚本
python tools/prepare_highlight_data.py `
    --video_path "D:/streaming_data/videos/stream_2024_01_15.mp4" `
    --chat_file "D:/streaming_data/chats/stream_2024_01_15.json" `
    --highlights_file "D:/streaming_data/highlights/stream_2024_01_15.json" `
    --output_dir "dataset/highlights"
```

这会自动创建 `dataset/highlights/stream_2024_01_15/` 并生成所有需要的文件。

### 训练输出位置：

```
outputs/
└── highlight_production/    ← 你在训练命令中指定的 output_dir
    ├── model/               ← 训练好的模型权重
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── ...
    ├── checkpoints/         ← 训练过程中的检查点
    │   ├── epoch_1/
    │   ├── epoch_2/
    │   └── ...
    └── logs/                ← 训练日志
        └── train.log
```

---

## 问题 3: 6-8 小时长视频的 GPU 处理问题

### ⚠️ 直接处理 6-8 小时视频会遇到的问题：

1. **显存不足**：
   - 当前测试用 179 秒视频，context_length=4096 就用了 11GB 显存
   - 6 小时 = 21,600 秒，约是测试视频的 **120 倍**
   - **无法一次性加载到 GPU**

2. **上下文长度限制**：
   - Llama 模型的最大 context length 通常是 4k-8k tokens
   - 6 小时的 ASR + 聊天数据可能产生 **50k-200k tokens**
   - **远超模型能处理的长度**

### ✅ 解决方案：Sliding Window（滑动窗口）

**好消息：项目已经实现了 sliding window！**

#### 工作原理：

```
6 小时视频 = [窗口1] [窗口2] [窗口3] ... [窗口N]
             ↓       ↓       ↓
           推理     推理     推理
             ↓       ↓       ↓
         highlight highlight highlight
                    ↓
            合并所有 highlight
```

每个窗口：
- 固定 token 数量（默认 35,000 tokens）
- 约对应 10-20 分钟的视频内容（取决于聊天密度）
- 窗口之间有重叠，避免遗漏边界的 highlight

#### 使用方法：

**推理（Inference）时使用 sliding window：**

```python
# 使用专门的 windowed inference 脚本
python inference_highlight.py \
    --video_path "D:/streaming_data/videos/long_stream_6hrs.mp4" \
    --chat_file "D:/streaming_data/chats/long_stream_6hrs.json" \
    --model "outputs/highlight_production/model" \
    --use_window  # 启用滑动窗口
```

或者直接使用：

```python
from src.test.highlights_window import get_highlights

highlights = get_highlights(
    video_path="long_stream.mp4",
    chat_file="long_stream_chat.json",
    model_path="outputs/highlight_production/model",
    window_token_size=35000  # 可调整
)
```

### 训练时如何处理长视频？

**关键：训练时不需要完整视频，只需要 highlight 片段！**

#### 策略 1：提取 Highlight 周围的上下文（推荐）

```python
# 伪代码示例
for each highlight in video:
    # 提取 highlight 前后各 5 分钟的内容
    context_start = max(0, highlight.start - 300)
    context_end = min(video_duration, highlight.end + 300)
    
    # 只用这个片段训练
    asr_segment = extract_asr(video, context_start, context_end)
    chat_segment = extract_chat(chat, context_start, context_end)
    
    # 创建训练样本
    training_sample = {
        "input": asr_segment + chat_segment,
        "target": highlight (adjusted timestamp)
    }
```

**修改 `tools/prepare_highlight_data.py` 实现这个策略：**

添加一个 `--segment_mode` 选项：
- 对每个 highlight，提取前后各 N 分钟的内容
- 生成多个训练样本，而不是一个完整视频的样本
- 这样可以将 6 小时视频切分成 10-20 个可训练的片段

#### 策略 2：使用固定窗口切分

```python
# 将 6 小时视频切分成 30 分钟的片段
for i in range(0, video_duration, 1800):  # 1800 秒 = 30 分钟
    segment_start = i
    segment_end = min(i + 1800, video_duration)
    
    # 找出这个片段中的 highlights
    segment_highlights = [h for h in highlights 
                         if segment_start <= h.start < segment_end]
    
    # 如果这个片段有 highlights，就用它训练
    if segment_highlights:
        create_training_sample(segment_start, segment_end, segment_highlights)
```

### 具体实现建议：

我可以为你修改 `prepare_highlight_data.py`，添加以下功能：

```python
python tools/prepare_highlight_data.py \
    --video_path "long_stream.mp4" \
    --chat_file "long_stream.json" \
    --highlights_file "long_stream_highlights.json" \
    --output_dir "dataset/highlights" \
    --segment_mode True \              # 启用分段模式
    --segment_window 1800 \             # 30 分钟窗口
    --segment_overlap 300               # 5 分钟重叠
```

这会将一个 6 小时视频切分成约 12-15 个训练样本，每个都可以被 GPU 处理。

---

## 问题 4: 需要多少视频作为训练数据？

### 计算考虑因素：

1. **视频时长**：6-8 小时/视频
2. **Highlight 密度**：每小时约有多少个 highlight？
3. **训练样本数**：需要多少个 highlight 样本？

### 估算：

假设：
- 每小时有 **3-5 个 highlight**
- 每个视频 6 小时 = **18-30 个 highlight/视频**
- 使用分段策略（每个 highlight 提取前后 10 分钟上下文）

#### 最小数据集（验证概念）：
```
视频数量: 10 部
总时长: 60-80 小时
Highlight 数: ~200 个
训练样本: ~200 个片段
```
✅ 可以验证模型是否学会了任务

#### 小规模生产（初步部署）：
```
视频数量: 30-50 部
总时长: 180-400 小时
Highlight 数: ~600-1000 个
训练样本: ~600-1000 个片段
```
✅ 可以得到基本可用的模型

#### 中等规模（推荐）：
```
视频数量: 100 部
总时长: 600-800 小时
Highlight 数: ~2000-3000 个
训练样本: ~2000-3000 个片段
```
✅ 可以得到较好的性能

#### 大规模（最佳性能）：
```
视频数量: 300+ 部
总时长: 1800+ 小时
Highlight 数: ~6000+ 个
训练样本: ~6000+ 个片段
```
✅ 可以得到接近商用水平的性能

### 对于 6-8 小时长视频的特殊建议：

**由于单个视频已经很长，你需要的视频数量可以相对较少！**

推荐策略：
```
1. 先准备 10 部视频 + 标注
2. 使用分段策略，生成 ~200 个训练样本
3. 训练并评估
4. 根据效果决定是否需要更多数据
5. 逐步增加到 30-50 部（约 500-1000 个样本）
```

### 数据质量 vs 数量：

**更重要的是标注质量！**

- ✅ 50 个精准标注的 highlight > 200 个粗糙标注
- ✅ 多样性：不同游戏/内容类型/聊天风格的视频
- ✅ 平衡性：各种类型的 highlight（exciting, funny, emotional 等）都要有
- ✅ 难度分布：既要有明显的 highlight，也要有不太明显的

### 标注工作量估算：

假设每个 6 小时视频需要标注 20 个 highlight：

```
标注一个 highlight：
- 找到开始/结束时间：2-3 分钟
- 选择类型：10 秒
- （可选）写描述：1-2 分钟

每个 highlight: ~3-5 分钟
每个视频（20 个 highlight）: ~60-100 分钟 = 1-1.5 小时

标注 10 个视频：10-15 小时
标注 30 个视频：30-45 小时
标注 100 个视频：100-150 小时
```

### 建议的开始方案：

```
Phase 1: 概念验证（1-2 周）
- 视频数量：5-10 部
- 专注于标注质量
- 训练并测试基础性能

Phase 2: 小规模部署（1 个月）
- 视频数量：30 部
- 调整模型和超参数
- 在验证集上评估

Phase 3: 扩展（持续）
- 根据实际效果决定是否继续添加数据
- 专注于模型表现不好的情况
- 可能需要 50-100 部视频
```

---

## 总结

1. **type 和 description**：可以简化，建议保留 type，省略 description
2. **文件位置**：
   - 原始数据：任意位置
   - 训练数据：`dataset/highlights/`
   - 输出模型：`outputs/highlight_production/`
3. **长视频处理**：
   - ✅ Sliding window 可以解决推理问题
   - ✅ 分段策略可以解决训练问题
   - 需要修改数据准备脚本
4. **数据量**：
   - 最少：10 部 6 小时视频（~200 个 highlight）
   - 推荐：30-50 部（~600-1000 个 highlight）
   - 理想：100 部（~2000 个 highlight）

### 下一步行动：

1. 我可以修改 `prepare_highlight_data.py` 支持分段模式
2. 简化标注格式（只保留 type，去掉 description）
3. 创建批量处理脚本，方便处理多个视频
4. 修改 inference 代码，确保 sliding window 正确处理长视频

**需要我现在就进行这些修改吗？**
