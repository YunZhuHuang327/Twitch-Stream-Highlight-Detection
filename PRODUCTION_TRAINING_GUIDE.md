# 从测试训练到生产训练的配置变更指南

本文档说明了从小规模测试训练切换到真正的生产训练时需要修改的配置。

## 📋 需要修改的配置项

### 1. 数据集配置

#### 当前测试配置（需要修改）：
```yaml
# configs/data/highlight.yaml
vidc_dir: ${paths.vidc_dir}  # 当前指向 ./dataset/
train_subset: train           # 只有 1 个视频 (v1)
val_subset: val               # 没有验证集
```

**测试用的子集文件：**
- `dataset/docs/subset_data/train.json` - 当前只包含 `["v1"]`

#### 生产配置（应该修改为）：
```yaml
# configs/data/highlight.yaml
vidc_dir: ${paths.vidc_dir}
train_subset: train           # 指向完整训练集
val_subset: val               # 指向验证集
```

**需要创建/更新：**
- `dataset/docs/subset_data/train.json` - 添加所有训练视频 ID
  ```json
  ["video1", "video2", "video3", ..., "video100"]
  ```
- `dataset/docs/subset_data/val.json` - 添加验证集视频 ID
  ```json
  ["val_video1", "val_video2", ..., "val_video20"]
  ```

### 2. 模型训练超参数

#### 当前测试配置（`run_train.bat`）：
```batch
python train.py data=highlight ^
  paths.output_dir=outputs/highlight_test ^
  model.config_train.model_name=D:/chapter-llama/Llama-3.2-1B-Instruct ^
  model.config_train.num_epochs=1 ^                          ⚠️ 只训练 1 轮
  model.config_train.batch_size_training=1 ^                 ⚠️ 最小 batch
  model.config_train.gradient_accumulation_steps=1 ^         ⚠️ 无梯度累积
  model.config_train.context_length=4096 ^                   ✅ 可保持
  model.config_train.use_peft=False ^                        ⚠️ 未使用 LoRA
  model.config_train.output_dir=outputs/highlight_test/model ^
  model.config_train.enable_fsdp=False
```

#### 生产配置（推荐）：
```batch
python train.py data=highlight ^
  paths.output_dir=outputs/highlight_production ^
  model.config_train.model_name=D:/chapter-llama/Llama-3.2-1B-Instruct ^
  model.config_train.num_epochs=3 ^                          ✅ 增加到 3-5 轮
  model.config_train.batch_size_training=1 ^                 ✅ 16GB GPU 保持为 1
  model.config_train.gradient_accumulation_steps=8 ^         ✅ 模拟更大的 batch
  model.config_train.context_length=4096 ^                   ✅ 保持
  model.config_train.use_peft=True ^                         ✅ 启用 LoRA 节省显存
  model.config_train.r=8 ^                                   ✅ LoRA rank
  model.config_train.lora_alpha=16 ^                         ✅ LoRA alpha
  model.config_train.output_dir=outputs/highlight_production/model ^
  model.config_train.enable_fsdp=False ^
  model.config_train.run_validation=True                     ✅ 启用验证
```

### 3. 优化器和学习率

#### 生产环境额外建议：
```batch
  model.config_train.lr=5e-5 ^                               ✅ 微调合适的学习率
  model.config_train.weight_decay=0.01 ^                     ✅ 权重衰减
  model.config_train.warmup_steps=100 ^                      ✅ 学习率预热（如果支持）
  model.config_train.lr_scheduler=cosine                     ✅ Cosine 学习率调度
```

### 4. 数据准备

#### 当前测试数据：
```
dataset/highlights/
└── v1/                         ⚠️ 只有 1 个视频
    ├── asr.txt
    ├── asr.json
    ├── chat.json
    ├── highlights.json
    ├── duration.txt
    └── metadata.json
```

#### 生产数据（需要准备）：
```
dataset/highlights/
├── video1/
│   ├── asr.txt
│   ├── asr.json
│   ├── chat.json
│   ├── highlights.json
│   ├── duration.txt
│   └── metadata.json
├── video2/
│   └── ...
├── video3/
│   └── ...
...
└── video100/
    └── ...
```

**使用数据准备脚本批量处理：**
```powershell
# 对每个视频运行
foreach ($video in Get-ChildItem "path/to/videos/*.mp4") {
    python tools/prepare_highlight_data.py `
        --video_path $video.FullName `
        --chat_file "path/to/chat_$($video.BaseName).json" `
        --highlights_file "path/to/highlights_$($video.BaseName).json" `
        --output_dir "dataset/highlights"
}
```

### 5. 监控和日志

#### 测试配置：
```yaml
use_wandb: False  ⚠️ 未启用 W&B 监控
```

#### 生产配置（推荐）：
```yaml
use_wandb: True   ✅ 启用 W&B 追踪训练
logger:
  wandb_config:
    project: highlight-detection
    entity: your-username
    tags: 
      - production
      - highlight
      - chat-augmented
```

### 6. 混合精度训练

#### 当前配置：
```yaml
fsdp_config.pure_bf16: True  # 在配置中，但 FSDP 未启用
```

#### 生产配置（16GB GPU 优化）：
```yaml
# 如果 RTX A4000 支持 BF16
model.config_train.use_fp16: False
model.config_train.use_bf16: True  # 通过 fsdp_config.pure_bf16 控制
```

### 7. 检查点保存

#### 生产配置建议：
```yaml
model.config_train.save_model: True
model.config_train.save_optimizer: True  # 保存优化器状态以便恢复
model.config_train.save_metrics: True
# 添加检查点保存策略
model.config_train.checkpoint_every_n_steps: 500  # 每 500 步保存一次
```

## 🔧 快速切换脚本

### 创建生产训练脚本 `run_train_production.bat`：

```batch
@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
call conda activate chapter-llama

REM 生产环境训练配置
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
  model.config_train.run_validation=True ^
  use_wandb=True > train_production_log.txt 2>&1

echo Training completed. Check train_production_log.txt for details.
pause
```

## 📊 预期资源需求

### 测试训练（当前）：
- 数据：1 个视频
- 时间：~1 分钟
- 显存：11 GB
- 总内存：1 GB

### 生产训练（预估）：
- 数据：100+ 个视频
- 时间：数小时到数天（取决于数据量）
- 显存：11-15 GB（使用 LoRA + gradient checkpointing）
- 总内存：2-4 GB

## ⚠️ 重要注意事项

1. **在启动生产训练前**：
   - 确保准备了足够的训练数据（建议 100+ 视频）
   - 创建验证集（建议 20-30 个视频）
   - 测试数据加载管线是否正常
   - 检查所有视频的 ASR 和 chat 数据质量

2. **训练过程中**：
   - 监控显存使用，避免 OOM
   - 定期检查验证集性能
   - 保存中间检查点

3. **如果遇到 OOM**：
   - 减小 `context_length` (4096 → 2048)
   - 增加 `gradient_accumulation_steps` (8 → 16)
   - 减小 LoRA `r` (8 → 4)
   - 启用更激进的 gradient checkpointing

## 📝 检查清单

训练前确认：
- [ ] 已准备 100+ 训练视频的数据
- [ ] 已创建 `train.json` 和 `val.json` 子集文件
- [ ] 已测试数据加载（运行几个 epoch 验证无错误）
- [ ] 已配置 W&B 或其他日志工具
- [ ] 已调整 `num_epochs` 到合适的值 (3-5)
- [ ] 已启用 LoRA (`use_peft=True`)
- [ ] 已启用验证 (`run_validation=True`)
- [ ] 已设置合适的输出目录
- [ ] 已备份重要配置文件

---

**总结**：当前测试配置可以成功运行，但仅用于验证管线。真正的训练需要：
1. 更多数据 (100+ 视频)
2. 更多训练轮数 (3-5 epochs)
3. 启用 LoRA 和其他优化
4. 启用验证和监控
