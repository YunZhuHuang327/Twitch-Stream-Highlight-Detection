# Windows 使用指南 - Chapter-Llama

本文档总结了在 Windows 环境下使用 chapter-llama 遇到的问题和解决方案。

## 已解决的问题

### 1. 符号链接创建
**问题**: Windows 需要创建符号链接到 VidChapters 数据集。
**解决**: 首先创建 `dataset` 目录，然后创建符号链接：
```powershell
New-Item -ItemType Directory -Path "dataset"
New-Item -ItemType SymbolicLink -Path "dataset\VidChapters" -Target "path\to\actual\VidChapters"
```

### 2. 数据集下载
**问题**: 缺少训练所需的数据文件。
**解决**: 从 HuggingFace 下载所需数据：
```powershell
# 创建目录结构
New-Item -ItemType Directory -Path "dataset\docs\subset_data\asrs" -Force
New-Item -ItemType Directory -Path "dataset\docs\subset_data\chapters" -Force

# 下载数据文件（已完成）
# - dataset/docs/subset_data/sml1k_train.json
# - dataset/docs/subset_data/asrs/asrs_sml1k_train.json  
# - dataset/docs/subset_data/chapters/chapters_sml1k_train.json
```

### 3. DDP (Distributed Data Parallel) 问题
**问题**: PyTorch DDP 在 Windows 上不工作。
**解决**: 修改 `train.py` 使用 `strategy="auto"` 而不是 `"ddp"`：
```python
fabric = Fabric(accelerator="gpu", strategy="auto", devices=1, num_nodes=1)
```

### 4. llama_cookbook 导入问题
**问题**: `get_policies` 函数在新版本中已被移除。
**解决**: 在 `src/models/llama_finetune.py` 中添加了自定义的 `get_policies` 函数。

### 5. FSDP 配置问题
**问题**: 在 Windows 单 GPU 环境下不需要 FSDP。
**解决**: 在 `configs/model/llama3.1_8B.yaml` 中设置：
```yaml
enable_fsdp: False
```

### 6. bitsandbytes 兼容性问题
**问题**: bitsandbytes 与 Windows PyTorch 不兼容。
**解决**: 卸载 bitsandbytes（因为配置中 `quantization: Null`）：
```powershell
conda activate chapter-llama
pip uninstall bitsandbytes -y
```

### 7. GPU 内存不足问题
**问题**: 16GB GPU 无法加载原始配置（context_length=16384）。
**解决**: 优化配置以适应 16GB GPU：
```yaml
context_length: 2048  # 从 16384 减少到 2048
r: 4  # LoRA rank 从 8 减少到 4
gradient_accumulation_steps: 8  # 增加以补偿较小的 context
use_gradient_checkpointing: True  # 启用梯度检查点
```

### 8. OpenMP 库冲突
**问题**: libiomp5md.dll 初始化冲突。
**解决**: 设置环境变量：
```powershell
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
```

### 9. 推理时模型路径问题
**问题**: 推理脚本尝试从 HuggingFace 下载模型，路径格式错误。
**解决**: 修改 `inference.py` 支持本地模型路径。

## 使用方法

### 训练

使用提供的脚本进行训练（已配置所有必要的环境变量）：

```powershell
.\run_train.ps1
```

或手动运行：
```powershell
conda activate chapter-llama
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:128'
python train.py
```

### 推理

使用本地训练的模型进行推理：

```powershell
.\run_inference.ps1 -VideoPath "path\to\video.mp4"
```

或指定自定义模型路径：
```powershell
.\run_inference.ps1 -VideoPath "path\to\video.mp4" -ModelPath "path\to\model_checkpoints" -BaseModel "path\to\base_model"
```

或手动运行：
```powershell
conda activate chapter-llama
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
python inference.py video.mp4 --model "outputs\chapterize\Llama-3.2-1B-Instruct\asr\default\sml1k_train\default\model_checkpoints" --base_model "Llama-3.2-1B-Instruct"
```

## 训练配置说明

当前为 16GB GPU 优化的配置：

- **模型**: Llama-3.2-1B-Instruct
- **Context Length**: 2048 tokens
- **LoRA Rank**: 4
- **Batch Size**: 1
- **Gradient Accumulation**: 8 steps
- **Gradient Checkpointing**: 启用
- **训练数据**: sml1k_train (1k 视频)

## 文件说明

- `run_train.ps1`: 训练启动脚本（包含所有环境变量）
- `run_inference.ps1`: 推理启动脚本（支持本地模型）
- `train_windows.ps1`: 训练脚本（带配置说明）

## 注意事项

1. **GPU 内存**: 确保在训练前清理 GPU 内存
2. **模型路径**: 确保本地模型路径存在且正确
3. **Base Model**: 确保 Llama-3.2-1B-Instruct 已下载到本地
4. **数据集**: 确保 VidChapters 数据集正确设置
5. **Conda 环境**: 始终在 `chapter-llama` 环境中运行

## 故障排除

### GPU 内存不足
如果仍然遇到 OOM 错误，可以进一步减少 `context_length` 到 1024。

### 模型加载失败
确保路径使用正确的格式（Windows 路径或 HuggingFace ID）。

### ASR 处理失败
确保视频文件存在且格式正确（mp4, mkv 等）。

## 性能优化

当前配置针对 16GB GPU（NVIDIA RTX A4000）优化。如果有更大的 GPU：

- 可以增加 `context_length` (4096 或 8192)
- 可以增加 `r` (LoRA rank 到 8 或 16)
- 可以减少 `gradient_accumulation_steps`
- 可以禁用 `use_gradient_checkpointing` 以提高速度

## 下一步

训练完成后，模型将保存在：
```
outputs/chapterize/Llama-3.2-1B-Instruct/asr/default/sml1k_train/default/model_checkpoints/
```

使用该路径进行推理以生成视频章节。
