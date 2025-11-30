# 继续训练指南 - 加入聊天室资讯

## 概述

本指南说明如何在已训练的 Chapter-Llama 模型基础上，加入新的聊天室数据继续训练。

## 模型来源说明

### 当前模型状态

1. **Llama-3.2-1B-Instruct** (本地)
   - 位置: `D:\chapter-llama\Llama-3.2-1B-Instruct`
   - 用途: 当前训练使用的基础模型
   - 已训练的 LoRA 适配器: `outputs/chapterize/Llama-3.2-1B-Instruct/.../model_checkpoints/`

2. **Llama-3.1-8B-Instruct** (缓存)
   - 位置: `C:\Users\Huang\.cache\huggingface\hub\models--meta-llama--Llama-3.1-8B-Instruct`
   - 大小: ~15 GB
   - 来源: HuggingFace 自动下载（首次使用时）
   - 用途: 推理时指定 `meta-llama/Llama-3.1-8B-Instruct` 会自动使用此缓存

### 它们的关系

- **训练**: 使用 Llama-3.2-1B (1.2B 参数)
- **推理**: 之前错误地使用了 Llama-3.1-8B (8B 参数)
- **现在**: 已修复，推理使用与训练相同的模型

## 继续训练的方法

### 方法 1: 在 Llama-3.2-1B 基础上继续训练（推荐）

**优点**:
- 保留之前训练的知识
- 内存占用小（适合 16GB GPU）
- 训练速度快

**步骤**:

#### 1. 准备聊天室数据

```python
# 使用 prepare_chat_data.py
python prepare_chat_data.py

# 或手动创建数据文件
# dataset/docs/subset_data/chat_data.json
# dataset/docs/subset_data/asrs/asrs_chat_data.json
# dataset/docs/subset_data/chapters/chapters_chat_data.json
```

数据格式示例：

```json
// chat_data.json - 视频/会话 ID 列表
["chat_001", "chat_002", "chat_003"]

// asrs/asrs_chat_data.json - ASR 转录
{
  "chat_001": "00:00:10 UserA: Hello everyone\n00:00:15 UserB: Hi there\n...",
  "chat_002": "..."
}

// chapters/chapters_chat_data.json - 章节标注
{
  "chat_001": {
    "duration": 600,
    "chapters": [
      {"timestamp": "00:00:00", "title": "Introduction"},
      {"timestamp": "00:02:30", "title": "Main Discussion"}
    ]
  }
}
```

#### 2. 合并数据集

创建包含原始数据 + 新数据的子集：

```json
// dataset/docs/subset_data/sml1k_train+chat.json
// 合并原有的 sml1k_train 和新的 chat_data
[
  // 原始视频 IDs
  "vid1", "vid2", ...,
  // 新的聊天会话 IDs  
  "chat_001", "chat_002", ...
]
```

或者创建纯聊天数据子集用于增量训练。

#### 3. 运行继续训练

```powershell
# 使用脚本（推荐）
.\continue_training.ps1

# 或手动运行
conda activate chapter-llama
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
python train.py data=asr_continue model=llama3.2_1B
```

### 方法 2: 使用 Llama-3.1-8B 训练

**优点**:
- 更强大的基础模型
- 可能有更好的性能

**缺点**:
- 需要更多 GPU 内存
- 训练速度较慢
- 需要更小的 context_length (1024 tokens)

**步骤**:

```powershell
# 使用 8B 模型配置
python train.py model=llama3.1_8B_cached data=asr_continue
```

配置文件: `configs/model/llama3.1_8B_cached.yaml`

## LoRA 适配器管理

### 合并多个 LoRA 适配器

如果你想合并已训练的适配器和新训练的适配器：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "D:/chapter-llama/Llama-3.2-1B-Instruct"
)

# 加载第一个 LoRA 适配器
model = PeftModel.from_pretrained(
    base_model,
    "outputs/.../sml1k_train/.../model_checkpoints",
    adapter_name="original"
)

# 加载第二个 LoRA 适配器
model.load_adapter(
    "outputs/.../chat_data/.../model_checkpoints",
    adapter_name="chat"
)

# 合并适配器
model.add_weighted_adapter(
    adapters=["original", "chat"],
    weights=[0.7, 0.3],  # 可调整权重
    adapter_name="merged",
    combination_type="linear"
)

# 保存合并后的适配器
model.save_pretrained("outputs/merged_adapter")
```

### 增量训练（不合并）

更简单的方法是直接在新数据上训练，这会：
1. 创建新的 LoRA 适配器
2. 保持原有适配器不变
3. 推理时可以选择使用哪个适配器

## 内存优化建议

### 对于 16GB GPU

**Llama-3.2-1B**:
```yaml
context_length: 2048
r: 4
gradient_accumulation_steps: 8
```

**Llama-3.1-8B**:
```yaml
context_length: 1024  # 更小
r: 4
gradient_accumulation_steps: 16  # 更多累积
```

### 监控内存使用

```powershell
# 训练时监控 GPU
nvidia-smi -l 1  # 每秒更新
```

## 最佳实践

1. **小批量测试**: 先用少量聊天数据测试训练流程
2. **保存检查点**: 定期保存，避免训练中断损失
3. **版本管理**: 为不同的训练版本创建不同的输出目录
4. **评估**: 训练后在测试集上评估性能

## 推理时使用正确的模型

训练后，确保推理使用正确的模型和适配器：

```powershell
# 使用继续训练后的模型
.\run_inference.ps1 -VideoPath "video.mp4" `
  -ModelPath "outputs/chapterize/Llama-3.2-1B-Instruct/asr/default/sml1k_train+chat/default/model_checkpoints" `
  -BaseModel "D:\chapter-llama\Llama-3.2-1B-Instruct"
```

## 常见问题

### Q: 继续训练会覆盖原有知识吗？

A: 不会。LoRA 是增量训练，会在原有知识基础上学习。但如果新数据与原数据差异很大，可能需要调整学习率。

### Q: 可以同时使用多个 LoRA 适配器吗？

A: 可以。PEFT 支持加载和组合多个适配器。

### Q: 训练需要多长时间？

A: 取决于数据量。1k 样本在 RTX A4000 上约需 2-4 小时。

### Q: 如何选择使用哪个基础模型？

A: 
- **Llama-3.2-1B**: 适合快速迭代、资源受限
- **Llama-3.1-8B**: 适合追求最佳性能、有足够资源

## 相关文件

- `prepare_chat_data.py`: 数据准备工具
- `continue_training.ps1`: 继续训练脚本
- `configs/data/asr_continue.yaml`: 数据配置
- `configs/model/llama3.1_8B_cached.yaml`: 8B 模型配置

## 下一步

1. 准备聊天室数据
2. 运行数据准备脚本
3. 选择训练方法
4. 开始训练
5. 评估结果
6. 使用新模型进行推理
