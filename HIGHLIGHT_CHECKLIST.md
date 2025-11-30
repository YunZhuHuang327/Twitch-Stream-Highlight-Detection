# 📋 Highlight Detection 实施清单

## ✅ 已创建的所有文件

### 📚 文档 (Documentation)

1. **HIGHLIGHT_DETECTION.md** - 技术方案和架构设计
   - 核心修改点
   - 数据格式
   - 需要修改的文件列表
   - 实现步骤 (Phase 1-4)
   - 评估指标
   - 进阶优化建议

2. **HIGHLIGHT_USAGE_GUIDE.md** - 完整使用指南
   - 数据准备详细步骤
   - 训练配置说明
   - 推理参数详解
   - 评估方法
   - 常见问题解答
   - 性能优化技巧

3. **HIGHLIGHT_QUICKSTART.md** - 快速开始指南
   - 5分钟快速上手
   - 示例命令
   - 关键参数
   - 预期输出

---

### 🛠️ 工具 (Tools)

4. **tools/prepare_highlight_data.py** - 数据准备工具
   - 功能：
     - ✅ 提取视频 ASR
     - ✅ 处理聊天数据（计算强度、检测峰值、提取关键词）
     - ✅ 转换 ground truth highlights
     - ✅ 生成训练数据集
     - ✅ 创建数据集索引
   - 使用方法：
     ```powershell
     python tools/prepare_highlight_data.py `
         --video_path "video.mp4" `
         --chat_file "chat.json" `
         --highlights_file "highlights.json" `
         --output_dir "dataset/highlights"
     ```

---

### 📊 数据模块 (Data Modules)

5. **src/data/highlight_data.py** - Highlight 数据加载器
   - `HighlightData` 类：加载 highlights、聊天、ASR
   - `HighlightDataset` 类：PyTorch Dataset
   - `HighlightDataModule` 类：Lightning DataModule
   - 功能：
     - ✅ 加载 highlight annotations
     - ✅ 加载和汇总聊天数据
     - ✅ 获取时间窗口内的聊天
     - ✅ 格式化 highlights

6. **src/data/utils_highlights.py** - Highlight Prompt 生成
   - `PromptHighlight` 类
   - 功能：
     - ✅ 训练 prompt（带 ground truth）
     - ✅ 推理 prompt（无 ground truth）
     - ✅ 窗口 prompt（带聊天数据）

7. **src/data/single_video.py** - 修改支持时间戳
   - 新增 `return_timestamps` 参数
   - 新增 `parse_asr_timestamps()` 函数
   - 保存 ASR JSON 格式

---

### 🔍 推理模块 (Inference)

8. **src/test/highlights_window.py** - 滑动窗口推理
   - 功能：
     - ✅ 滑动窗口 + 重叠 (overlap)
     - ✅ 聊天数据集成
     - ✅ Highlight 解析
     - ✅ 重叠 highlight 合并
     - ✅ 结果保存
   - 关键函数：
     - `get_window_with_chat()` - 获取窗口数据
     - `get_highlights()` - 主推理循环
     - `parse_highlights()` - 解析模型输出
     - `merge_overlapping_highlights()` - 合并重复

9. **inference_highlight.py** - 主推理脚本
   - 完整的推理流程
   - 支持有/无聊天数据
   - 可配置窗口参数
   - 自动保存结果

---

### 📈 评估模块 (Evaluation)

10. **src/utils/metrics_highlight.py** - 评估指标
    - 功能：
      - ✅ Temporal IoU 计算
      - ✅ Precision, Recall, F1
      - ✅ Mean Average Precision (mAP)
      - ✅ 多 IoU 阈值评估
      - ✅ 结果可视化打印
    - 使用：
      ```python
      from src.utils.metrics_highlight import evaluate_video
      evaluate_video('pred.json', 'gt.json')
      ```

---

### ⚙️ 配置文件 (Configs)

11. **configs/data/highlight.yaml** - 数据配置
    - 数据集路径
    - Highlight 类型定义
    - 聊天数据设置
    - 窗口参数

12. **configs/model/llama3.2_1B_highlight.yaml** - 模型配置
    - 基于 llama3.2_1B
    - 针对 16GB GPU 优化：
      - context_length: 2048
      - LoRA rank: 4
      - gradient_accumulation: 8
      - BF16 混合精度
      - Gradient checkpointing

13. **configs/experiment/highlight.yaml** - 实验配置
    - 任务定义
    - 评估指标
    - IoU 阈值

---

### 🎬 运行脚本 (Scripts)

14. **train_highlight.ps1** - 训练脚本
    - 设置环境变量
    - 激活 conda
    - 运行训练

15. **run_inference_highlight.ps1** - 推理脚本
    - 配置路径
    - 检查聊天文件
    - 运行推理

---

### 📝 示例文件 (Examples)

16. **examples/chat_example.json** - 聊天数据示例
    - 包含时间戳、用户、消息、表情
    - 展示两种时间格式（HH:MM:SS 和秒数）

17. **examples/highlights_example.json** - Highlights 示例
    - 5 种 highlight 类型示例
    - 展示输出格式

---

## 🗂️ 完整目录结构

```
chapter-llama/
├── 📚 文档
│   ├── HIGHLIGHT_DETECTION.md          # 技术方案
│   ├── HIGHLIGHT_USAGE_GUIDE.md        # 使用指南
│   ├── HIGHLIGHT_QUICKSTART.md         # 快速开始
│   └── HIGHLIGHT_CHECKLIST.md          # 本文件
│
├── 🛠️ 工具
│   └── tools/
│       └── prepare_highlight_data.py   # 数据准备
│
├── 📊 源代码
│   └── src/
│       ├── data/
│       │   ├── highlight_data.py       # 数据加载
│       │   ├── utils_highlights.py     # Prompt 生成
│       │   └── single_video.py         # (修改) 时间戳支持
│       ├── test/
│       │   └── highlights_window.py    # 滑动窗口推理
│       └── utils/
│           └── metrics_highlight.py    # 评估指标
│
├── ⚙️ 配置
│   └── configs/
│       ├── data/
│       │   └── highlight.yaml          # 数据配置
│       ├── model/
│       │   └── llama3.2_1B_highlight.yaml  # 模型配置
│       └── experiment/
│           └── highlight.yaml          # 实验配置
│
├── 🎬 脚本
│   ├── inference_highlight.py          # 推理主脚本
│   ├── train_highlight.ps1             # 训练脚本
│   └── run_inference_highlight.ps1     # 推理运行脚本
│
└── 📝 示例
    └── examples/
        ├── chat_example.json           # 聊天数据示例
        └── highlights_example.json     # Highlights 示例
```

---

## 🚀 使用流程

### Step 1: 准备数据 ⏱️ 1-2 天

```powershell
# 1. 准备你的聊天数据和 ground truth
# 参考 examples/ 目录中的格式

# 2. 运行数据准备工具
python tools/prepare_highlight_data.py `
    --video_path "video.mp4" `
    --chat_file "chat.json" `
    --highlights_file "highlights.json" `
    --output_dir "dataset/highlights"

# 3. 为所有视频重复步骤 2

# 4. 生成索引
python tools/prepare_highlight_data.py `
    --update_index `
    --output_dir "dataset/highlights"
```

**输出**:
```
dataset/highlights/
├── index.json
├── video_001/
│   ├── asr.txt
│   ├── asr.json
│   ├── chat.json
│   ├── highlights.json
│   ├── duration.txt
│   └── metadata.json
├── video_002/
│   └── ...
```

### Step 2: 训练模型 ⏱️ 4-6 小时 (1k 视频)

```powershell
# 检查配置
cat configs/data/highlight.yaml
cat configs/model/llama3.2_1B_highlight.yaml

# 开始训练
.\train_highlight.ps1

# 或手动
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
conda activate chapter-llama
python train.py experiment=highlight
```

**监控训练**:
```powershell
# 查看日志
Get-Content outputs/highlight/.../train.log -Tail 50 -Wait

# 监控 GPU
nvidia-smi -l 1
```

### Step 3: 运行推理 ⏱️ 60-90 分钟 (6-8 小时视频)

```powershell
# 编辑脚本中的路径
notepad run_inference_highlight.ps1

# 运行推理
.\run_inference_highlight.ps1

# 或手动
python inference_highlight.py "video.mp4" `
    --model "outputs/highlight/.../model_checkpoints" `
    --base_model "Llama-3.2-1B-Instruct" `
    --chat_file "chat.json"
```

**输出**:
```
outputs/inference/VIDEO_ID/
├── asr.txt
├── asr.json
└── highlights.json  # 检测结果
```

### Step 4: 评估结果 ⏱️ 5-10 分钟

```python
from src.utils.metrics_highlight import evaluate_video

# 评估单个视频
evaluate_video(
    'outputs/inference/video_001/highlights.json',
    'dataset/highlights/video_001/highlights.json'
)
```

**输出示例**:
```
📊 Highlight Detection Evaluation Results
============================================================

📈 Statistics:
  Predictions: 12
  Ground Truth: 10
  Avg Predicted Duration: 145.3s
  Avg GT Duration: 158.7s

🎯 Precision, Recall, F1:

  IoU @0.3:
    Precision: 0.833
    Recall:    0.900
    F1 Score:  0.865
    TP: 9, FP: 3, FN: 1

  IoU @0.5:
    Precision: 0.750
    Recall:    0.800
    F1 Score:  0.774
    TP: 8, FP: 4, FN: 2

⭐ Mean Average Precision:
  mAP: 0.784
  AP@0.3: 0.856
  AP@0.5: 0.781
  AP@0.7: 0.715
```

---

## 📊 关键参数速查

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `context_length` | 2048 | 针对 16GB GPU 优化 |
| `batch_size` | 1 | 每次处理 1 个样本 |
| `gradient_accumulation` | 8 | 等效 batch_size=8 |
| `r` (LoRA rank) | 4 | LoRA 秩 |
| `lora_alpha` | 8 | LoRA alpha |
| `lr` | 5e-5 | 学习率 |
| `num_epochs` | 3 | 训练轮数 |
| `use_bf16` | True | BF16 混合精度 |

### 推理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `window_size` | 35000 | 窗口大小（tokens） |
| `overlap` | 300 | 窗口重叠（秒） |
| `max_windows` | 100 | 最大窗口数 |
| `temperature` | 0.7 | 生成温度 |
| `top_p` | 0.9 | Nucleus sampling |
| `max_new_tokens` | 512 | 最大生成 tokens |

### 评估参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `iou_thresholds` | [0.3, 0.5, 0.7] | IoU 阈值 |
| `iou_merge` | 0.5 | 合并重叠的阈值 |

---

## ✅ 验证清单

### 数据准备

- [ ] 视频文件存在且可访问
- [ ] 聊天数据格式正确（参考 `examples/chat_example.json`）
- [ ] Ground truth highlights 准备完成
- [ ] ASR 提取成功（`dataset/highlights/VIDEO_ID/asr.txt`）
- [ ] 聊天数据已处理（`chat.json` 有 intensity_timeline）
- [ ] 数据集索引已生成（`dataset/highlights/index.json`）

### 训练环境

- [ ] Conda 环境已激活
- [ ] GPU 可用且有 16GB+ 内存
- [ ] CUDA 和 PyTorch 版本匹配
- [ ] 配置文件正确（`configs/data/highlight.yaml`）
- [ ] 基础模型存在（`Llama-3.2-1B-Instruct`）
- [ ] 磁盘空间充足（至少 50GB）

### 推理准备

- [ ] 模型训练完成
- [ ] Checkpoint 存在（`outputs/highlight/.../model_checkpoints/`）
- [ ] 测试视频准备好
- [ ] （可选）测试视频的聊天数据准备好
- [ ] GPU 可用（或使用 CPU 模式）

### 评估准备

- [ ] 预测结果已生成（`highlights.json`）
- [ ] Ground truth 可用
- [ ] 评估脚本可运行

---

## 🎯 预期结果

### 训练

**时间**: 
- 1k 视频: ~4-6 小时
- 5k 视频: ~20-30 小时

**内存**: ~14GB GPU

**输出**:
```
outputs/highlight/Llama-3.2-1B-Instruct-Highlight/
└── highlight/
    └── highlight_detection/
        └── train/
            └── default/
                ├── model_checkpoints/  # 训练好的模型
                ├── train.log          # 训练日志
                └── metrics.json       # 训练指标
```

### 推理

**时间**: 
- 6-8 小时视频: ~60-90 分钟
- ASR 提取: ~30-60 分钟
- Highlight 检测: ~30-40 分钟

**输出**: 10-30 个 highlights

**准确率**: 
- Precision@0.5: 0.70-0.85
- Recall@0.5: 0.75-0.90
- F1@0.5: 0.72-0.87

### 评估

**指标**:
- mAP: 0.75-0.85
- IoU@0.5 F1: 0.75-0.85

---

## 🆘 故障排除

### 问题 1: ImportError

```
ModuleNotFoundError: No module named 'src.data.highlight_data'
```

**解决**:
```powershell
# 确保在项目根目录
cd D:\chapter-llama
# 重新安装
pip install -e .
```

### 问题 2: GPU OOM

```
CUDA out of memory
```

**解决**: 减小 context_length
```yaml
# configs/model/llama3.2_1B_highlight.yaml
context_length: 1024  # 从 2048 降到 1024
```

### 问题 3: 聊天数据解析错误

```
KeyError: 'timestamp'
```

**解决**: 检查 JSON 格式
```python
import json
with open('chat.json', 'r') as f:
    data = json.load(f)
    print(data[0])  # 应该有 'timestamp' 字段
```

### 问题 4: 检测不到 highlights

**可能原因**:
1. 模型未充分训练
2. 温度参数太低
3. 窗口太小

**解决**:
```python
# 调整 inference_highlight.py
temperature=0.8,  # 增加多样性
top_p=0.95,
```

---

## 📞 获取更多帮助

1. **技术细节**: 阅读 `HIGHLIGHT_DETECTION.md`
2. **使用指南**: 阅读 `HIGHLIGHT_USAGE_GUIDE.md`
3. **快速上手**: 阅读 `HIGHLIGHT_QUICKSTART.md`
4. **环境问题**: 阅读 `WINDOWS_SETUP.md`

---

## 🎉 准备就绪！

所有文件已创建完成！你现在可以：

1. ✅ 查看示例数据格式（`examples/`）
2. ✅ 准备你的数据
3. ✅ 运行数据准备工具
4. ✅ 开始训练模型
5. ✅ 运行 highlight 检测
6. ✅ 评估结果

祝你成功！🚀
