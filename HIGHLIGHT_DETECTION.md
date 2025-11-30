# Highlight 检测模型改造方案

## 📋 目标

将 Chapter-Llama 从章节生成模型改造为影片 Highlight 检测模型

**输入**:
- 6-8 小时的长视频
- 聊天室内容（实时弹幕/评论）
- Ground truth highlight 时间戳

**输出**:
- Highlight 时间段 (开始时间 - 结束时间)
- Highlight 描述/类型

---

## 🔧 核心修改点

### 1. Sliding Window 配置

**当前位置**: `src/test/vidchapters_window.py`

**当前设置**:
```python
window_token_size = 35_000  # tokens
# 约等于 2-3 小时的 ASR 文本（取决于说话速度）
```

**针对 6-8 小时视频的建议配置**:

```python
# 选项 A: 保持 35k tokens (推荐)
window_token_size = 35_000  # 每个窗口约 2-3 小时
window_overlap = 300  # 5 分钟重叠，避免边界处遗漏

# 选项 B: 更大的窗口（需要更多 GPU 内存）
window_token_size = 70_000  # 每个窗口约 4-6 小时
window_overlap = 600  # 10 分钟重叠

# 选项 C: 更小的窗口（更快，但需要更多窗口）
window_token_size = 20_000  # 每个窗口约 1-2 小时
window_overlap = 180  # 3 分钟重叠
```

### 2. 输出格式修改

**当前**: 章节 = `{timestamp: title}`
```json
{
  "00:00:00": "Introduction",
  "00:15:30": "Main Topic Discussion",
  "01:30:00": "Q&A Session"
}
```

**改为**: Highlight 段 = `{start_time-end_time: {type, description, chat_intensity}}`
```json
{
  "00:15:30-00:18:45": {
    "type": "exciting_moment",
    "description": "Team won the game",
    "chat_intensity": 0.95,
    "chat_keywords": ["wow", "amazing", "poggers"]
  },
  "01:23:00-01:26:30": {
    "type": "funny_moment", 
    "description": "Streamer made a hilarious mistake",
    "chat_intensity": 0.88,
    "chat_keywords": ["lol", "笑死", "haha"]
  }
}
```

### 3. Prompt 修改

**当前 Prompt** (`src/data/utils_asr.py`):
```python
prompt = f"""You are an expert at creating chapter timestamps for videos.
Given the transcript, create chapter timestamps with titles.
Duration: {duration}
Transcript:
"""
```

**改为 Highlight 检测 Prompt**:
```python
prompt = f"""You are an expert at detecting highlight moments in live streaming videos.
Given the video transcript and chat messages, identify exciting/funny/important moments.
Output format: [START_TIME-END_TIME] TYPE: DESCRIPTION

Highlight types:
- exciting_moment: Exciting gameplay, achievements, victories
- funny_moment: Humorous situations, jokes, accidents
- emotional_moment: Touching or dramatic moments
- skill_showcase: Impressive plays or techniques
- chat_peak: Moments with extremely high chat activity

Duration: {duration}
Transcript:
{transcript}

Chat Activity:
{chat_summary}

Highlights:
"""
```

---

## 📂 数据格式修改

### 原始数据格式

**当前**: VidChapters 格式
```json
{
  "video_id": "123",
  "chapters": [
    {"timestamp": "00:00:00", "title": "Intro"},
    {"timestamp": "00:15:30", "title": "Main Content"}
  ]
}
```

### 新的 Highlight 数据格式

```json
{
  "video_id": "stream_001",
  "duration": 28800,  // 8 hours in seconds
  "highlights": [
    {
      "start_time": "00:15:30",
      "end_time": "00:18:45",
      "type": "exciting_moment",
      "description": "Epic victory in final round",
      "chat_peak": true,
      "chat_messages_count": 450
    },
    {
      "start_time": "01:23:00",
      "end_time": "01:26:30",
      "type": "funny_moment",
      "description": "Unexpected game glitch",
      "chat_peak": true,
      "chat_messages_count": 320
    }
  ],
  "chat_data": {
    "total_messages": 15000,
    "peaks": [
      {"time": "00:17:00", "intensity": 0.95, "keywords": ["wow", "amazing"]},
      {"time": "01:24:30", "intensity": 0.88, "keywords": ["lol", "haha"]}
    ]
  }
}
```

---

## 🔄 需要修改的文件

### 1. 数据加载 - `src/data/chapters.py`

创建新的 `HighlightData` 类：

```python
class HighlightData(Chapters):
    """Load highlight annotations and chat data"""
    
    def load_highlights(self, video_id):
        """Load highlight segments for a video"""
        highlights_file = self.vidc_dir / "highlights" / f"{video_id}.json"
        return openf(highlights_file)
    
    def load_chat_data(self, video_id):
        """Load chat/comment data for a video"""
        chat_file = self.vidc_dir / "chat" / f"{video_id}.json"
        return openf(chat_file)
    
    def get_chat_summary(self, video_id, time_window=60):
        """
        Summarize chat activity in time windows
        
        Args:
            time_window: seconds to group chat messages
        
        Returns:
            dict: {timestamp: {count, keywords, intensity}}
        """
        pass
```

### 2. Prompt 生成 - `src/data/utils_highlights.py` (新建)

```python
class PromptHighlight:
    """Generate prompts for highlight detection"""
    
    def __init__(self, highlights_data, chat_data):
        self.highlights = highlights_data
        self.chat = chat_data
    
    def get_prompt_train(self, video_id):
        """Training prompt with ground truth highlights"""
        duration = self.highlights.get_duration(video_id, hms=True)
        transcript = self.highlights.get_asr(video_id)
        chat_summary = self.chat.get_chat_summary(video_id)
        ground_truth = self.highlights.get_highlights(video_id)
        
        prompt = f"""Detect highlights in this {duration} video.
Transcript: {transcript}
Chat: {chat_summary}
Highlights: {format_highlights(ground_truth)}
"""
        return prompt
    
    def get_prompt_test(self, video_id):
        """Inference prompt without ground truth"""
        pass
```

### 3. 窗口处理 - 修改 `src/test/vidchapters_window.py`

```python
def get_window_with_chat(
    prompt: str,
    transcript: str,
    chat_data: dict,
    tokenizer,
    start_time: float = 0,
    window_token_size: int = 35_000,
    window_overlap: int = 300,  # 5 minutes overlap
):
    """
    Get windowed transcript + chat data
    
    Args:
        window_overlap: seconds to overlap between windows
    """
    # ... existing code ...
    
    # Add chat data for this time window
    window_chat = get_chat_in_timerange(
        chat_data, 
        start_time, 
        start_time + window_duration
    )
    
    return prompt, windowed_transcript, window_chat, reached_end
```

### 4. 训练配置 - `configs/data/highlight.yaml` (新建)

```yaml
defaults:
  - _self_

prompt: highlight
_target_: src.data.highlight_data.HighlightDataModule

# Dataset settings
subset: stream_train  # your training set
vidc_dir: ./dataset/

# Highlight specific settings
highlight_types:
  - exciting_moment
  - funny_moment
  - emotional_moment
  - skill_showcase
  - chat_peak

# Chat integration
use_chat: true
chat_window: 60  # seconds to summarize chat
chat_weight: 0.3  # weight for chat signals

# Window settings (for long videos)
window_size: 35000  # tokens
window_overlap: 300  # seconds
```

### 5. 模型配置 - `configs/model/llama3.2_1B_highlight.yaml` (新建)

```yaml
defaults:
  - llama3.2_1B

model_name: Llama-3.2-1B-Instruct-Highlight

config_train:
  # ... existing settings ...
  
  # Highlight-specific
  task_type: highlight_detection
  output_format: segment  # instead of timestamp
  
  # For 6-8 hour videos
  context_length: 2048  # for 16GB GPU
  gradient_accumulation_steps: 8
  
  # Learning rate - might need adjustment
  lr: 5e-5  # slightly lower for fine-tuning on specific domain
```

---

## 🎯 实现步骤

### Phase 1: 数据准备 (1-2 days)

1. **转换 ground truth 数据**
   ```python
   # tools/prepare_highlight_data.py
   def convert_to_highlight_format(video_file, chat_file, highlight_timestamps):
       """Convert your data to the required format"""
       pass
   ```

2. **处理聊天室数据**
   ```python
   def process_chat_data(chat_messages):
       """
       Process chat to extract:
       - Message timestamps
       - Message content
       - Emote/keyword frequency
       - Activity intensity
       """
       pass
   ```

3. **创建训练数据集**
   - `dataset/highlights/stream_001/`
     - `asr.txt` - 视频转录
     - `chat.json` - 聊天数据
     - `highlights.json` - Ground truth
     - `duration.txt` - 视频长度

### Phase 2: 代码修改 (2-3 days)

1. **创建新的数据加载器**
   - `src/data/highlight_data.py`
   - `src/data/utils_highlights.py`

2. **修改窗口处理**
   - 添加 overlap 支持
   - 集成 chat 数据

3. **修改输出解析**
   - 解析时间段而非时间点
   - 添加类型和描述提取

### Phase 3: 训练和评估 (3-5 days)

1. **训练 Highlight 检测模型**
   ```powershell
   python train.py data=highlight model=llama3.2_1B_highlight
   ```

2. **评估指标**
   - Precision/Recall at different IoU thresholds
   - Mean Average Precision (mAP)
   - Temporal localization accuracy

### Phase 4: 优化 (ongoing)

1. **Chat 特征工程**
   - 实验不同的 chat 表示方法
   - 调整 chat 权重

2. **多模态融合** (可选)
   - 如果有视频帧，可以添加视觉特征
   - 融合音频特征（音量、音调变化）

---

## 📊 评估指标

```python
def calculate_highlight_metrics(pred_highlights, gt_highlights, iou_threshold=0.5):
    """
    Calculate:
    - Precision: 预测的 highlight 中有多少是正确的
    - Recall: 真实的 highlight 中有多少被检测到
    - F1 Score
    - mAP (mean Average Precision)
    """
    pass

def calculate_temporal_iou(pred_segment, gt_segment):
    """Calculate Intersection over Union for time segments"""
    pred_start, pred_end = pred_segment
    gt_start, gt_end = gt_segment
    
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    
    return intersection / union if union > 0 else 0
```

---

## 🚀 快速开始

### 1. 准备你的数据

```python
# prepare_my_highlight_data.py
from pathlib import Path
import json

def prepare_data(video_path, chat_file, highlight_timestamps):
    # 1. 提取 ASR
    from src.data.single_video import get_asr
    asr, duration = get_asr(video_path)
    
    # 2. 处理聊天数据
    chat_data = process_chat(chat_file)
    
    # 3. 格式化 highlights
    highlights = {
        "video_id": video_path.stem,
        "duration": duration,
        "highlights": highlight_timestamps,
        "chat_data": chat_data
    }
    
    # 4. 保存
    output_dir = Path(f"dataset/highlights/{video_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "asr.txt", "w") as f:
        f.write(asr)
    with open(output_dir / "highlights.json", "w") as f:
        json.dump(highlights, f, indent=2)
    
    return output_dir
```

### 2. 修改 Sliding Window

在 `src/test/vidchapters_window.py` 中添加：

```python
# 在 get_window 函数中添加 overlap
def get_window(
    prompt: str,
    transcript: str,
    tokenizer,
    start_time: float = 0,
    window_token_size: int = 35_000,
    overlap_seconds: int = 300,  # <-- 新增
):
    # ... existing code ...
    
    # Calculate next window start with overlap
    next_start = last_timestamp - overlap_seconds
    
    return prompt, windowed_transcript, reached_end, next_start
```

### 3. 运行训练

```powershell
# 设置环境变量
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

# 运行训练
conda activate chapter-llama
python train.py data=highlight model=llama3.2_1B_highlight
```

---

## 💡 进阶优化建议

### 1. 聊天强度作为额外信号

```python
def calculate_chat_intensity(messages, window_size=10):
    """
    Calculate chat message rate in sliding windows
    High intensity often correlates with highlights
    """
    intensity_timeline = []
    for i in range(0, len(messages), window_size):
        window = messages[i:i+window_size]
        intensity = len(window) / window_size
        intensity_timeline.append({
            'timestamp': window[0]['timestamp'],
            'intensity': intensity
        })
    return intensity_timeline
```

### 2. 关键词提取

```python
from collections import Counter

def extract_chat_keywords(messages, top_k=10):
    """Extract most frequent words/emotes during highlight moments"""
    words = []
    for msg in messages:
        words.extend(msg['content'].split())
    return Counter(words).most_common(top_k)
```

### 3. 多阶段检测

```python
# Stage 1: Coarse detection (大窗口，快速定位可能区域)
coarse_highlights = detect_highlights(
    window_size=70000,
    threshold=0.7
)

# Stage 2: Fine-grained detection (小窗口，精确边界)
fine_highlights = []
for coarse in coarse_highlights:
    fine = detect_highlights(
        window_size=20000,
        start_time=coarse['start'] - 60,
        end_time=coarse['end'] + 60,
        threshold=0.5
    )
    fine_highlights.extend(fine)
```

---

## 📝 下一步行动

1. **确认数据格式**: 检查你的 ground truth 格式
2. **准备聊天数据**: 确保有时间戳和内容
3. **选择窗口大小**: 根据 GPU 内存选择
4. **开始实现**: 按照 Phase 1-4 逐步进行

需要我帮你实现具体的某个部分吗？比如：
- 数据转换脚本
- 聊天数据处理
- 修改后的训练代码
- 评估指标实现
