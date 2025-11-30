# Chapter-Llama 分段使用指南

## 🎯 目标

使用 Chapter-Llama 对长视频的 ASR 文本进行语义分段。

---

## 📋 你有什么

- ✅ ASR 文本: `dataset/highlights/123/asr.txt`
- ✅ 格式: `HH:MM:SS: text`
- ✅ 时长: ~6小时
- ✅ 行数: ~8579 行

---

## 🚀 使用方法

### ⭐ 方法 1: 使用 `quick_chapter.py` + GPT-4o-mini（最简单！推荐）

这个方法完全避免了本地模型的编码问题，使用 OpenAI API。

#### 步骤 1: 设置 API Key

```batch
set OPENAI_API_KEY=sk-your-key-here
```

或者在命令行直接提供：`--api_key sk-xxx`

获取 API Key: https://platform.openai.com/api-keys

#### 步骤 2: 运行脚本

**选项 A: 使用 batch 文件**
```batch
.\run_quick_chapter.bat
```

**选项 B: 直接运行 Python**
```batch
python quick_chapter.py
```

**选项 C: 自定义参数**
```batch
python quick_chapter.py ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --video_title "TwitchCon W/ AGENT00" ^
    --output_dir "outputs/chapters/123" ^
    --api_key sk-xxx
```

#### 输出

- `outputs/chapters/123/chapters.json` - 章节时间戳和标题
- `outputs/chapters/123/chapters_summary.txt` - 可读格式摘要

#### 成本

- 对于 6 小时视频 (~8500 行 ASR)
- 预估: **$0.15 - $0.30**
- 处理时间: **2-5 分钟**

---

### 方法 2: 使用 `chapter_from_asr_english.py` (本地模型 - 無編碼問題！)

如果你已经下载了 Chapter-Llama 模型，可以使用这个**英文版本**，完全避免編碼問題。

#### 步骤 1: 直接運行 batch 文件

```batch
.\run_chapter_english.bat
```

**或者手動運行**:

```batch
python chapter_from_asr_english.py ^
    --asr_file "dataset/highlights/123/asr.txt" ^
    --video_title "TwitchCon W/ AGENT00" ^
    --output_dir "outputs/chapters/123"
```

#### 為什麼使用英文版本？

- ✅ **完全避免 Unicode 編碼錯誤**
- ✅ **所有輸出訊息都是英文**
- ✅ **自動處理 Windows 控制台編碼問題**
- ✅ **包含 safe_print() 函數防止崩潰**
- ✅ **自動設置 KMP_DUPLICATE_LIB_OK=TRUE**

#### 功能完全相同

`chapter_from_asr_english.py` 和 `chapter_from_asr.py` 功能完全一樣，只是：
- 所有 print 訊息改為英文
- 所有註釋改為英文
- 添加了更強大的編碼錯誤處理

---

### 方法 3: 使用 Chapter-Llama 原始 inference.py

如果你有完整的视频文件：

```bash
python inference.py 123.mp4 --model asr-10k
```

**输出**:
- `outputs/inference/123/chapters.json`
- `outputs/inference/123/output_text.txt`

---

### 方法 3: 手动准备数据（最简单测试）

#### 步骤 1: 格式转换

创建 `convert_asr_format.py`:

```python
import sys
import os

# 读取原始 ASR
with open('dataset/highlights/123/asr.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 转换为 Chapter-Llama 格式
output = []
for line in lines:
    line = line.strip()
    if ':' in line:
        parts = line.split(':', 3)
        if len(parts) >= 4:
            # HH:MM:SS: text -> [HH:MM:SS] text
            timestamp = f"{parts[0]}:{parts[1]}:{parts[2]}"
            text = parts[3].strip()
            output.append(f"[{timestamp}] {text}")

# 保存
os.makedirs('outputs/chapters/123', exist_ok=True)
with open('outputs/chapters/123/formatted_asr.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print(f"Converted {len(output)} lines")
print("Saved to: outputs/chapters/123/formatted_asr.txt")
```

运行:
```bash
python convert_asr_format.py
```

#### 步骤 2: 使用 Llama 进行分段

可以用任何 LLM（包括 API）来做分段。

---

## 🤖 使用 API 进行分段（最简单！）

如果 Chapter-Llama 模型下载或运行有问题，直接用 GPT-4 更简单：

### 使用 OpenAI API

创建 `chapter_with_api.py`:

```python
import json
from openai import OpenAI

# 读取 ASR
with open('dataset/highlights/123/asr.txt', 'r', encoding='utf-8') as f:
    asr_text = f.read()

# 准备 prompt
prompt = f'''You are a video chapter segmentation expert.

Video Title: "TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY"

Task: Analyze the following ASR transcript and identify semantic chapter boundaries.
For each chapter, provide:
1. Start timestamp
2. Descriptive chapter title

ASR Transcript:
{asr_text[:50000]}  # 只取前50K字符测试

Output format (JSON):
{{
  "00:00:00": "Chapter Title 1",
  "00:15:30": "Chapter Title 2",
  ...
}}

Only output the JSON, no other text.
'''

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

# 解析结果
chapters = json.loads(response.choices[0].message.content)

# 保存
with open('outputs/chapters/123/chapters.json', 'w', encoding='utf-8') as f:
    json.dump(chapters, f, indent=2, ensure_ascii=False)

print(f"Generated {len(chapters)} chapters")
for ts, title in list(chapters.items())[:5]:
    print(f"  {ts}: {title}")
```

**成本**: 对于6小时视频，约 $0.20

---

## 📊 预期输出

### `chapters.json`:

```json
{
  "00:00:00": "Arrival at Empire State Building",
  "00:15:30": "Meeting Agent00 and Planning",
  "00:45:00": "Exploring Little Italy - Food Tour",
  "01:30:00": "Shopping and Fan Interactions",
  "02:15:00": "Kayaking Incident Discussion",
  "03:00:00": "Dinner and Conversations",
  "04:30:00": "Q&A with Chat",
  "05:45:00": "Wrap Up and Goodbyes"
}
```

### 章节特点

- 通常 6-10 小时视频会生成 **20-50 个章节**
- 每个章节 **10-30 分钟**
- 基于语义转换，不是简单的时间切分

---

## 🔧 故障排除

### 问题 1: Unicode 编码错误

**症状**:
```
UnicodeEncodeError: 'cp950' codec can't encode character
```

**解决方案**:
```batch
chcp 65001
set PYTHONIOENCODING=utf-8
```

### 问题 2: OpenMP 错误

**症状**:
```
OMP: Error #15: Initializing libiomp5md.dll
```

**解决方案**:
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

### 问题 3: 模型下载失败

**解决方案**: 使用 API 代替
- GPT-4o-mini: 最便宜
- Claude Haiku: 最快
- Gemini Flash: 免费

---

## 🎯 推荐流程

### 如果你想快速测试:

1. **用 `quick_chapter.py` (GPT-4o-mini API)** ✅ 推荐！
   - **最简单** - 只需要 API key
   - **成本低** - $0.15-0.30 for 6小时视频
   - **快速** - 2-5分钟完成
   - **无编码问题** - 完全避免 Unicode 错误
   - **运行**: `.\run_quick_chapter.bat`

2. **用 Chapter-Llama 本地模型**
   - 免费但需要设置
   - 需要下载模型 (~3GB)
   - 需要 GPU
   - 可能遇到编码问题

### 快速开始（推荐）:

```batch
# 步骤 1: 设置 API Key
set OPENAI_API_KEY=sk-your-key-here

# 步骤 2: 运行
.\run_quick_chapter.bat
```

就这么简单！

### 旧版完整脚本示例（已被 quick_chapter.py 替代）:

```python
# quick_chapter.py
import json
import os
from openai import OpenAI

# API key from environment
api_key = os.getenv('OPENAI_API_KEY', 'your-key-here')
client = OpenAI(api_key=api_key)

# Read ASR
print("Reading ASR...")
with open('dataset/highlights/123/asr.txt', 'r', encoding='utf-8') as f:
    asr_lines = f.readlines()

# 分段处理长 ASR (GPT-4o-mini 有 128K token 限制)
# 对于 6 小时视频，可能需要分段
chunk_size = 100000  # 每次处理 100K 字符

# 只处理前 100K 字符做测试
asr_text = ''.join(asr_lines[:2000])  # 前 2000 行

prompt = f'''You are an expert in video chapter segmentation.

Video: "TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY"

Analyze this ASR transcript and create semantic chapters.

Guidelines:
- Look for topic changes, location changes, or activity changes
- Typical chapter length: 10-30 minutes
- Use descriptive titles (e.g., "Exploring Little Italy", not "Chapter 1")

ASR Transcript:
{asr_text}

Output JSON only (no other text):
{{
  "00:00:00": "Chapter title",
  "00:15:30": "Next chapter",
  ...
}}
'''

print("Calling GPT-4o-mini...")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

result = response.choices[0].message.content
print(f"\nResponse:\n{result}\n")

# Parse and save
try:
    chapters = json.loads(result)
    os.makedirs('outputs/chapters/123', exist_ok=True)
    with open('outputs/chapters/123/chapters.json', 'w', encoding='utf-8') as f:
        json.dump(chapters, f, indent=2, ensure_ascii=False)

    print(f"SUCCESS! Generated {len(chapters)} chapters:")
    for ts, title in chapters.items():
        print(f"  {ts}: {title}")

    print(f"\nSaved to: outputs/chapters/123/chapters.json")

except json.JSONDecodeError:
    print("Failed to parse JSON, saving raw response...")
    with open('outputs/chapters/123/raw_response.txt', 'w', encoding='utf-8') as f:
        f.write(result)
```

**运行**:
```bash
python quick_chapter.py
```

---

## 💡 下一步

生成 chapters 后，你可以：

1. **用于 Highlight Pipeline**:
   ```bash
   python tools/highlight_detection_pipeline.py \
       --video_path "123.mp4" \
       --video_title "TwitchCon" \
       --chat_file "123.json" \
       --output_dir "outputs/highlights/123" \
       --skip_chapters  # 使用已生成的 chapters
   ```

2. **分析章节分布**:
   ```python
   import json
   with open('outputs/chapters/123/chapters.json') as f:
       chapters = json.load(f)

   # 计算每个章节的时长
   timestamps = sorted([time_to_seconds(t) for t in chapters.keys()])
   for i in range(len(timestamps)-1):
       duration = timestamps[i+1] - timestamps[i]
       print(f"Chapter {i+1}: {duration//60} minutes")
   ```

3. **导出视频片段**:
   ```python
   import subprocess
   for i, (start, title) in enumerate(chapters.items()):
       # 计算结束时间（下一个章节或视频结尾）
       if i < len(chapters) - 1:
           end = list(chapters.keys())[i+1]
       else:
           end = "06:00:00"  # 视频结尾

       subprocess.run([
           'ffmpeg', '-i', '123.mp4',
           '-ss', start, '-to', end,
           '-c', 'copy',
           f'chapter_{i+1:02d}_{title.replace(" ", "_")}.mp4'
       ])
   ```

---

## 🎉 总结

**最简单的方法**：用 GPT-4o-mini API
- 复制上面的 `quick_chapter.py`
- 设置你的 `OPENAI_API_KEY`
- 运行
- 5分钟完成

**完全本地的方法**：用 Chapter-Llama
- 需要下载模型
- 需要 GPU
- 但完全免费

选择适合你的方法！🚀
