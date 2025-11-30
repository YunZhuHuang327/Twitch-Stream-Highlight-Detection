# Quick Start Guide - 快速开始

## 你想做什么？

### 1. 对长视频进行章节分段 (Chapter Segmentation)

使用 `quick_chapter.py` - 最简单的方法！

```batch
# 设置 API Key
set OPENAI_API_KEY=sk-your-key-here

# 运行
.\run_quick_chapter.bat
```

**输出**: `outputs/chapters/123/chapters.json`

**成本**: ~$0.20 for 6小时视频

**详细文档**: [CHAPTER_SEGMENTATION_GUIDE.md](CHAPTER_SEGMENTATION_GUIDE.md)

---

### 2. 提取聊天室特征 (Chat Feature Extraction)

使用 `extract_chat_features.py` - 分析 Twitch 聊天数据

```batch
python tools/extract_chat_features.py ^
    --chat_file "123.json" ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --output_dir "outputs/chat_features/123" ^
    --readable_transcript
```

**输出**:
- `readable_transcript.txt` - ASR + 聊天事件标签
- `chat_features_summary.json` - 统计数据

**详细文档**: [CHAT_FEATURES_GUIDE.md](CHAT_FEATURES_GUIDE.md)

---

### 3. 完整 Highlight 检测流程 (Complete Pipeline)

使用 `highlight_detection_pipeline.py` - 端到端检测精彩片段

```batch
python tools/highlight_detection_pipeline.py ^
    --video_path "123.mp4" ^
    --video_title "TwitchCon W/ AGENT00" ^
    --chat_file "123.json" ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --output_dir "outputs/highlights/123" ^
    --use_api openai ^
    --clip_duration 60
```

**输出**:
- `final_highlights.json` - Top-K 精彩片段（带开始/结束时间）
- `highlight_scores.json` - 所有片段的分数

**详细文档**:
- [HIGHLIGHT_USAGE_GUIDE.md](HIGHLIGHT_USAGE_GUIDE.md)
- [API_SCORING_GUIDE.md](API_SCORING_GUIDE.md)

---

## 推荐工作流程

### 快速测试（10分钟）

1. **章节分段**:
   ```batch
   set OPENAI_API_KEY=sk-xxx
   .\run_quick_chapter.bat
   ```

2. **提取聊天特征**:
   ```batch
   python tools/extract_chat_features.py --chat_file "123.json" --asr_file "dataset/highlights/123/asr.txt" --output_dir "outputs" --readable_transcript
   ```

3. **检测精彩片段**:
   ```batch
   python tools/highlight_detection_pipeline.py --video_path "123.mp4" --video_title "TwitchCon" --chat_file "123.json" --asr_file "dataset/highlights/123/asr.txt" --output_dir "outputs/highlights/123" --use_api openai
   ```

### 完整生产流程

参考: [PRODUCTION_TRAINING_GUIDE.md](PRODUCTION_TRAINING_GUIDE.md)

---

## 常用文档索引

| 文档 | 用途 |
|------|------|
| [CHAPTER_SEGMENTATION_GUIDE.md](CHAPTER_SEGMENTATION_GUIDE.md) | 章节分段详细教程 |
| [CHAT_FEATURES_GUIDE.md](CHAT_FEATURES_GUIDE.md) | 聊天特征提取说明 |
| [HIGHLIGHT_USAGE_GUIDE.md](HIGHLIGHT_USAGE_GUIDE.md) | Highlight 检测完整流程 |
| [API_SCORING_GUIDE.md](API_SCORING_GUIDE.md) | API 使用和成本对比 |
| [WINDOWS_SETUP.md](WINDOWS_SETUP.md) | Windows 环境配置 |
| [LONG_VIDEO_FAQ.md](LONG_VIDEO_FAQ.md) | 长视频处理常见问题 |

---

## 工具脚本索引

| 脚本 | 功能 | 位置 |
|------|------|------|
| `quick_chapter.py` | 章节分段 (API) | 根目录 |
| `chapter_from_asr.py` | 章节分段 (本地模型) | 根目录 |
| `extract_chat_features.py` | 聊天特征提取 | tools/ |
| `highlight_detection_pipeline.py` | 完整 Highlight 流程 | tools/ |
| `prepare_highlight_data.py` | 准备训练数据 | tools/ |
| `convert_chat_format.py` | 聊天格式转换 | tools/ |

---

## 快速命令参考

### 章节分段
```batch
# API 方式（推荐）
python quick_chapter.py --api_key sk-xxx

# 本地模型方式
python chapter_from_asr.py --asr_file "asr.txt" --video_title "Title" --output_dir "outputs"
```

### 聊天特征
```batch
# 生成可读 transcript
python tools/extract_chat_features.py --chat_file "123.json" --asr_file "asr.txt" --output_dir "outputs" --readable_transcript

# 提取完整特征
python tools/extract_chat_features.py --chat_file "123.json" --output_dir "outputs" --merge_into_asr
```

### Highlight 检测
```batch
# 使用 API 打分
python tools/highlight_detection_pipeline.py --video_path "video.mp4" --video_title "Title" --chat_file "chat.json" --asr_file "asr.txt" --output_dir "outputs" --use_api openai

# 使用规则打分
python tools/highlight_detection_pipeline.py --video_path "video.mp4" --video_title "Title" --chat_file "chat.json" --asr_file "asr.txt" --output_dir "outputs"
```

---

## 需要帮助？

1. **编码问题**: 查看 [CHAPTER_SEGMENTATION_GUIDE.md](CHAPTER_SEGMENTATION_GUIDE.md) 的故障排除部分
2. **API 使用**: 查看 [API_SCORING_GUIDE.md](API_SCORING_GUIDE.md)
3. **长视频处理**: 查看 [LONG_VIDEO_FAQ.md](LONG_VIDEO_FAQ.md)
4. **Windows 配置**: 查看 [WINDOWS_SETUP.md](WINDOWS_SETUP.md)

---

## 下一步

完成章节分段后，你可以：

1. 使用章节数据进行 Highlight 检测
2. 分析章节分布和时长
3. 使用 ffmpeg 按章节切割视频
4. 继续训练自己的 Highlight 检测模型

参考完整文档获取更多细节！
