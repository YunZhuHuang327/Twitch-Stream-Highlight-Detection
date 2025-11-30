# 长视频 Highlight 检测 - 快速参考

## 🎯 核心变更总结

所有修改已完成！现在系统支持：

1. ✂️ **长视频分段处理** - 自动将 6-8 小时视频切分为 30 分钟片段
2. 🏷️ **简化标注格式** - 只需要 `start_time`, `end_time`, `type`（无需 description）
3. 📦 **批量处理工具** - 一次处理多个视频
4. 📖 **完整文档** - 标注指南、FAQ、使用教程

---

## ⚡ 快速开始（3 步）

### 1️⃣ 准备数据结构

```
D:/streaming_data/
├── videos/
│   ├── stream_001.mp4    # 6-8 小时视频
│   ├── stream_002.mp4
│   └── stream_003.mp4
├── chats/
│   ├── stream_001.json   # 聊天记录
│   ├── stream_002.json
│   └── stream_003.json
└── highlights/
    ├── stream_001.json   # Highlight 标注（简化格式）
    ├── stream_002.json
    └── stream_003.json
```

### 2️⃣ 批量处理视频（分段模式）

```powershell
python tools/batch_prepare_highlights.py `
    --data_dir "D:/streaming_data" `
    --output_dir "dataset/highlights" `
    --segment_mode `
    --segment_window 1800 `
    --segment_overlap 300 `
    --simplify
```

**结果**: 每个 6 小时视频 → 约 12 个 30 分钟训练片段

### 3️⃣ 训练模型

```powershell
cmd /c run_train_production.bat
```

---

## 📝 简化标注格式示例

**你只需要标注这 3 个字段**:

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

**支持的 Highlight 类型**:
- `exciting_moment` - 激动时刻（精彩操作、胜利）
- `funny_moment` - 搞笑时刻（失误、幽默）
- `skill_showcase` - 技术展示（高难度操作）
- `emotional_moment` - 情感时刻（感人瞬间）
- `chat_peak` - 聊天高峰（弹幕爆炸）

---

## 🔧 新增工具

### 1. 分段数据准备工具

```powershell
# 单个长视频处理（自动分段）
python tools/prepare_highlight_data.py `
    --video_path "long_stream.mp4" `
    --chat_file "long_stream_chat.json" `
    --highlights_file "long_stream_highlights.json" `
    --segment_mode `
    --segment_window 1800 `
    --segment_overlap 300
```

### 2. 批量处理工具

```powershell
# 一次处理多个视频
python tools/batch_prepare_highlights.py `
    --data_dir "D:/streaming_data" `
    --segment_mode `
    --limit 10  # 只处理前 10 个（测试用）
```

### 3. 自动生成训练集划分

```python
# 运行快速脚本生成 train.json, val.json, chapters.json
python tools/generate_dataset_split.py
```

---

## 📚 完整文档

| 文档 | 用途 |
|------|------|
| [COMPLETE_USAGE_GUIDE.md](COMPLETE_USAGE_GUIDE.md) | 📖 完整使用教程（推荐从这里开始）|
| [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) | 🏷️ Highlight 标注指南和技巧 |
| [LONG_VIDEO_FAQ.md](LONG_VIDEO_FAQ.md) | ❓ 长视频处理常见问题 |
| [PRODUCTION_TRAINING_GUIDE.md](PRODUCTION_TRAINING_GUIDE.md) | 🎓 生产环境训练配置 |

---

## 🎯 关键参数说明

### 分段处理参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `segment_window` | 1800 | 每个片段长度（秒） | 1800 (30分钟) |
| `segment_overlap` | 300 | 片段重叠（秒） | 300 (5分钟) |

**为什么需要重叠？**
- 避免 highlight 被切分到两个片段边界
- 确保不遗漏重要时刻

### 训练参数（生产环境）

| 参数 | 测试值 | 生产值 | 说明 |
|------|--------|--------|------|
| `num_epochs` | 1 | 5 | 训练轮数 |
| `batch_size_training` | 1 | 1 | 单个样本（16GB GPU） |
| `gradient_accumulation_steps` | 1 | 8 | 模拟 batch=8 |
| `context_length` | 4096 | 4096 | 上下文长度 |
| `use_peft` | False | True | 启用 LoRA |
| `lr` | 0.0001 | 5e-5 | 学习率 |

---

## 📊 数据量建议

对于 6-8 小时长视频：

| 阶段 | 视频数 | 片段数 | 用途 |
|------|--------|--------|------|
| **概念验证** | 5-10 | ~60-120 | 验证流程可行 |
| **小规模** | 30 | ~360 | 初步可用 |
| **推荐** | 50-100 | ~600-1200 | 良好性能 |

**每个 6 小时视频 ≈ 12 个训练片段**

---

## 🚀 从测试到生产的变更

### 需要修改的配置

参考 [PRODUCTION_TRAINING_GUIDE.md](PRODUCTION_TRAINING_GUIDE.md)

**核心变更**:
1. 增加数据量（1 → 100+ 视频）
2. 增加训练轮数（1 → 5 epochs）
3. 启用 LoRA（`use_peft=True`）
4. 启用验证（`run_validation=True`）
5. 增加梯度累积（1 → 8）

---

## 💡 最佳实践

### ✅ 推荐做法

1. **从小开始**: 先用 1-2 个视频测试完整流程
2. **使用分段模式**: 对 6+ 小时视频启用 `--segment_mode`
3. **简化标注**: 使用 `--simplify`，只标注必要字段
4. **批量处理**: 使用 `batch_prepare_highlights.py`
5. **定期备份**: 保存标注数据和训练模型

### ❌ 避免做法

1. 不要直接训练完整 6 小时视频（会 OOM）
2. 不要忽略 `--segment_overlap`（会遗漏边界 highlight）
3. 不要追求完美标注（质量 > 数量）
4. 不要一次处理太多视频（先验证流程）

---

## 🔍 验证流程

### 处理后检查

```powershell
# 检查生成的片段
ls dataset/highlights/

# 查看某个片段的元数据
Get-Content dataset/highlights/stream_001_seg000/metadata.json | ConvertFrom-Json

# 统计总片段数
(Get-ChildItem dataset/highlights/ -Directory).Count
```

### 训练前检查

```powershell
# 确认训练集文件存在
Test-Path dataset/docs/subset_data/train.json
Test-Path dataset/docs/chapters.json

# 查看训练集包含多少片段
(Get-Content dataset/docs/subset_data/train.json | ConvertFrom-Json).Count
```

---

## 🆘 遇到问题？

### 常见错误

1. **找不到聊天/highlight 文件**
   ```
   确保文件名匹配: video.mp4 → video.json
   ```

2. **分段后片段太少**
   ```
   减小 segment_window 或检查视频时长
   ```

3. **训练时显存不足**
   ```
   减小 context_length: 4096 → 2048
   ```

4. **训练集为空**
   ```
   检查 train.json 中的 ID 是否与实际文件夹匹配
   ```

---

## 📞 完整文档索引

1. **开始使用** → [COMPLETE_USAGE_GUIDE.md](COMPLETE_USAGE_GUIDE.md)
2. **如何标注** → [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)
3. **常见问题** → [LONG_VIDEO_FAQ.md](LONG_VIDEO_FAQ.md)
4. **生产配置** → [PRODUCTION_TRAINING_GUIDE.md](PRODUCTION_TRAINING_GUIDE.md)
5. **原始 README** → [README.md](README.md)

---

**现在你可以开始处理 6-8 小时的长视频了！** 🎉

有任何问题，请查看相应的详细文档。
