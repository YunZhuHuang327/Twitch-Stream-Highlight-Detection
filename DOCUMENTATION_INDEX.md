# 📖 长视频 Highlight 检测 - 文档索引

## 🎯 我应该从哪里开始？

### 👉 **首次使用** → 从这里开始！
📄 [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - **5 分钟快速入门**
- 核心变更总结
- 3 步快速开始
- 简化标注格式示例
- 新增工具介绍

---

## 📚 完整文档列表

### 1️⃣ 使用指南

#### 🚀 [QUICK_REFERENCE.md](QUICK_REFERENCE.md) ⭐ 推荐起点
**用途**: 5 分钟快速了解所有变更和使用方法
**适合**: 第一次使用、快速查阅

**包含内容**:
- 核心变更总结
- 3 步快速开始
- 简化标注格式
- 关键参数说明
- 数据量建议
- 常见错误快速解决

---

#### 📖 [COMPLETE_USAGE_GUIDE.md](COMPLETE_USAGE_GUIDE.md)
**用途**: 完整的端到端使用教程
**适合**: 第一次完整操作流程

**包含内容**:
- 数据准备详细步骤
- 聊天数据格式
- Highlight 标注格式
- 单个/批量处理命令
- 训练配置和执行
- 推理使用方法
- 常见问题解答

**章节**:
1. 快速开始
2. 数据准备（6 个步骤）
3. 训练流程
4. 推理使用
5. 常见问题

---

### 2️⃣ 标注相关

#### 🏷️ [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)
**用途**: 详细的 Highlight 标注指南
**适合**: 准备标注数据时参考

**包含内容**:
- 简化标注格式说明
- 5 种 Highlight 类型详解
- 时间格式支持
- 标注工作流程（3 种方法）
- 标注质量建议
- 标注技巧
- 验证工具使用

**亮点**:
- 每种类型的详细说明和示例
- 时间精度建议
- 数量分布建议
- 聊天数据辅助标注

---

#### ✅ [examples/highlight_template_simple.json](examples/highlight_template_simple.json)
**用途**: 简化标注模板
**适合**: 快速创建新的标注文件

```json
[
  {
    "start_time": "00:00:00",
    "end_time": "00:00:00",
    "type": "exciting_moment"
  }
]
```

---

### 3️⃣ 问题解答

#### ❓ [LONG_VIDEO_FAQ.md](LONG_VIDEO_FAQ.md)
**用途**: 长视频处理常见问题详解
**适合**: 遇到具体问题时查阅

**包含内容**:
- Q1: type 和 description 的作用及是否可选
- Q2: 文件存放位置和输出位置
- Q3: 6-8 小时视频 GPU 处理问题
  - 分段处理详细说明
  - Sliding window 工作原理
  - 训练时如何处理长视频
- Q4: 需要多少视频训练
  - 不同阶段的数据量建议
  - 标注工作量估算

**亮点**:
- 详细的分段处理原理
- 具体的数据量建议
- 标注工作量估算

---

#### 🎓 [PRODUCTION_TRAINING_GUIDE.md](PRODUCTION_TRAINING_GUIDE.md)
**用途**: 从测试到生产的配置变更指南
**适合**: 准备生产环境训练时参考

**包含内容**:
- 需要修改的配置项（6 大类）
- 数据集配置变更
- 训练超参数调整
- 优化器和学习率设置
- 数据准备要求
- 监控和日志配置
- 快速切换脚本

---

### 4️⃣ 变更说明

#### 📝 [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
**用途**: 完整的修改总结报告
**适合**: 了解具体做了哪些修改

**包含内容**:
- 完成的 4 大功能
- 修改的文件列表
- 新增的工具和文档
- 处理效果预览
- 关键改进说明
- 代码质量改进
- 下一步建议

---

## 🔧 工具说明

### 数据处理工具

#### 1. `tools/prepare_highlight_data.py`
**功能**: 单个视频数据准备（支持分段模式）

**基础用法**:
```powershell
python tools/prepare_highlight_data.py `
    --video_path "video.mp4" `
    --chat_file "chat.json" `
    --highlights_file "highlights.json"
```

**分段模式**（长视频）:
```powershell
python tools/prepare_highlight_data.py `
    --video_path "long_stream.mp4" `
    --chat_file "chat.json" `
    --highlights_file "highlights.json" `
    --segment_mode `
    --segment_window 1800 `
    --segment_overlap 300
```

---

#### 2. `tools/batch_prepare_highlights.py` ⭐ 推荐
**功能**: 批量处理多个视频

**用法**:
```powershell
python tools/batch_prepare_highlights.py `
    --data_dir "D:/streaming_data" `
    --segment_mode `
    --simplify
```

**参数**:
- `--data_dir`: 数据根目录（包含 videos/, chats/, highlights/）
- `--segment_mode`: 启用分段模式（长视频）
- `--simplify`: 使用简化标注格式
- `--limit N`: 只处理前 N 个视频（测试用）

---

#### 3. `tools/generate_dataset_split.py`
**功能**: 自动生成训练/验证集划分

**用法**:
```powershell
python tools/generate_dataset_split.py
```

**输出**:
- `dataset/docs/subset_data/train.json`
- `dataset/docs/subset_data/val.json`
- `dataset/docs/chapters.json`

---

#### 4. `tools/validate_highlights.py`
**功能**: 验证 Highlight 标注文件格式

**验证单个文件**:
```powershell
python tools/validate_highlights.py highlights.json
```

**验证整个目录**:
```powershell
python tools/validate_highlights.py --dir D:/streaming_data/highlights -v
```

**检查项**:
- JSON 格式正确性
- 必需字段完整性
- 时间格式和有效性
- 类型字段有效性
- 时间重叠检测
- 时长合理性

---

## 🎯 常见任务快速导航

### 任务 1: 我是第一次使用，如何开始？
1. 阅读 [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (5 分钟)
2. 阅读 [COMPLETE_USAGE_GUIDE.md](COMPLETE_USAGE_GUIDE.md) 的"数据准备"部分
3. 准备 1-2 个测试视频的数据
4. 运行单个视频处理测试

---

### 任务 2: 如何标注 Highlight？
1. 阅读 [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)
2. 使用 [examples/highlight_template_simple.json](examples/highlight_template_simple.json) 作为模板
3. 标注完成后用 `tools/validate_highlights.py` 验证

---

### 任务 3: 如何处理 6-8 小时长视频？
1. 阅读 [LONG_VIDEO_FAQ.md](LONG_VIDEO_FAQ.md) 的 Q3
2. 使用 `--segment_mode` 参数
3. 查看 [COMPLETE_USAGE_GUIDE.md](COMPLETE_USAGE_GUIDE.md) 的"第五步"

---

### 任务 4: 如何批量处理多个视频？
1. 组织数据目录结构（见 [COMPLETE_USAGE_GUIDE.md](COMPLETE_USAGE_GUIDE.md) "第一步"）
2. 运行 `tools/batch_prepare_highlights.py`
3. 运行 `tools/generate_dataset_split.py`

---

### 任务 5: 如何从测试切换到生产训练？
1. 阅读 [PRODUCTION_TRAINING_GUIDE.md](PRODUCTION_TRAINING_GUIDE.md)
2. 修改配置参数
3. 准备更多训练数据
4. 运行 `run_train_production.bat`

---

### 任务 6: 遇到错误怎么办？
1. 查看 [COMPLETE_USAGE_GUIDE.md](COMPLETE_USAGE_GUIDE.md) 的"常见问题"
2. 查看 [LONG_VIDEO_FAQ.md](LONG_VIDEO_FAQ.md)
3. 查看 [QUICK_REFERENCE.md](QUICK_REFERENCE.md) 的"验证流程"

---

## 📊 文档特性对比

| 文档 | 长度 | 难度 | 用途 | 推荐阅读顺序 |
|------|------|------|------|-------------|
| QUICK_REFERENCE.md | 短 | ⭐ 简单 | 快速入门 | 1️⃣ 第一个读 |
| COMPLETE_USAGE_GUIDE.md | 长 | ⭐⭐ 中等 | 完整教程 | 2️⃣ 开始操作前 |
| ANNOTATION_GUIDE.md | 中 | ⭐⭐ 中等 | 标注指南 | 3️⃣ 标注数据时 |
| LONG_VIDEO_FAQ.md | 长 | ⭐⭐⭐ 详细 | 问题解答 | 遇到问题时 |
| PRODUCTION_TRAINING_GUIDE.md | 中 | ⭐⭐ 中等 | 配置优化 | 生产部署前 |
| CHANGES_SUMMARY.md | 长 | ⭐ 简单 | 变更记录 | 了解修改内容 |

---

## 🎓 推荐学习路径

### 路径 A: 快速上手（1-2 小时）
```
1. QUICK_REFERENCE.md (5 分钟)
   ↓
2. ANNOTATION_GUIDE.md - 标注模板部分 (10 分钟)
   ↓
3. 标注 1-2 个测试视频 (30-60 分钟)
   ↓
4. 运行单个视频处理测试 (10 分钟)
   ↓
5. 查看 COMPLETE_USAGE_GUIDE.md - 训练部分 (20 分钟)
```

### 路径 B: 完整学习（3-4 小时）
```
1. QUICK_REFERENCE.md (5 分钟)
   ↓
2. COMPLETE_USAGE_GUIDE.md 完整阅读 (40 分钟)
   ↓
3. ANNOTATION_GUIDE.md 完整阅读 (30 分钟)
   ↓
4. 标注 5-10 个视频 (2-3 小时)
   ↓
5. LONG_VIDEO_FAQ.md 浏览 (20 分钟)
   ↓
6. PRODUCTION_TRAINING_GUIDE.md (30 分钟)
```

### 路径 C: 问题驱动（按需查阅）
```
遇到具体问题时：
1. 先查 QUICK_REFERENCE.md 的"常见错误"
2. 再查 LONG_VIDEO_FAQ.md
3. 最后查 COMPLETE_USAGE_GUIDE.md 的"常见问题"
```

---

## 🔍 快速搜索

### 我想了解...

- **简化标注格式** → QUICK_REFERENCE.md, ANNOTATION_GUIDE.md
- **分段处理原理** → LONG_VIDEO_FAQ.md Q3, COMPLETE_USAGE_GUIDE.md
- **批量处理** → QUICK_REFERENCE.md, COMPLETE_USAGE_GUIDE.md 第五步
- **数据量建议** → LONG_VIDEO_FAQ.md Q4, QUICK_REFERENCE.md
- **训练配置** → PRODUCTION_TRAINING_GUIDE.md, COMPLETE_USAGE_GUIDE.md
- **标注技巧** → ANNOTATION_GUIDE.md
- **GPU 显存问题** → LONG_VIDEO_FAQ.md Q3
- **文件位置** → LONG_VIDEO_FAQ.md Q2, COMPLETE_USAGE_GUIDE.md

---

## 💡 使用建议

### ✅ 高效的学习方式
1. **先读 QUICK_REFERENCE.md**（5 分钟了解全貌）
2. **边做边学**（实践中遇到问题再查文档）
3. **书签重要页面**（ANNOTATION_GUIDE 的类型说明等）
4. **使用 Ctrl+F 搜索**（在长文档中快速定位）

### ❌ 避免的方式
1. 不要试图一次读完所有文档
2. 不要跳过 QUICK_REFERENCE.md
3. 不要在不理解的情况下直接运行命令
4. 不要忽略文档中的警告和注意事项

---

## 📞 获取更多帮助

如果文档没有解决你的问题：

1. **检查是否有更新的文档**
2. **查看工具的 `--help` 输出**:
   ```powershell
   python tools/batch_prepare_highlights.py --help
   python tools/validate_highlights.py --help
   ```
3. **查看训练日志**（可能有具体的错误信息）

---

**祝使用顺利！** 🚀

*最后更新: 2025-10-19*
