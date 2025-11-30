"""
Highlight Detection Data Module

加载和处理 highlight 检测数据
"""

from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import lightning as L
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset


class HighlightData:
    """Load highlight annotations and chat data"""
    
    def __init__(self, vidc_dir: str, subset: str = "train"):
        """
        Args:
            vidc_dir: 数据集根目录 (e.g., "dataset")
            subset: 子集名称 (e.g., "train", "val", "test")
        """
        self.vidc_dir = Path(vidc_dir)
        self.subset = subset
        self.highlights_dir = self.vidc_dir / "highlights"
        
        # Load video IDs for this subset
        subset_file = self.vidc_dir / "docs" / "subset_data" / f"{subset}.json"
        if subset_file.exists():
            with open(subset_file, 'r') as f:
                self.video_ids = json.load(f)
        else:
            # Fallback: scan highlights directory
            self.video_ids = [d.name for d in self.highlights_dir.iterdir() if d.is_dir()]
    
    def get_asr(self, video_id: str) -> str:
        """Load ASR transcript for a video"""
        asr_file = self.highlights_dir / video_id / "asr.txt"
        if not asr_file.exists():
            return ""
        return asr_file.read_text(encoding='utf-8')
    
    def get_duration(self, video_id: str, hms: bool = False) -> str:
        """Get video duration"""
        duration_file = self.highlights_dir / video_id / "duration.txt"
        if not duration_file.exists():
            return "00:00:00" if hms else "0"
        
        duration_str = duration_file.read_text(encoding='utf-8').strip()
        if hms:
            return duration_str
        else:
            # Parse HH:MM:SS to seconds
            parts = duration_str.split(':')
            if len(parts) == 3:
                h, m, s = parts
                return str(int(h) * 3600 + int(m) * 60 + float(s))
            return duration_str
    
    def load_highlights(self, video_id: str) -> List[Dict]:
        """Load highlight segments for a video"""
        highlights_file = self.highlights_dir / video_id / "highlights.json"
        if not highlights_file.exists():
            return []
        with open(highlights_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_chat_data(self, video_id: str) -> Dict:
        """Load chat/comment data for a video"""
        chat_file = self.highlights_dir / video_id / "chat.json"
        if not chat_file.exists():
            return {"messages": [], "intensity_timeline": [], "peak_moments": []}
        with open(chat_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_chat_summary(self, video_id: str, max_length: int = 2000) -> str:
        """
        获取聊天数据摘要，用于添加到 prompt 中
        
        Args:
            video_id: 视频 ID
            max_length: 最大字符数
        
        Returns:
            格式化的聊天摘要字符串
        """
        chat_data = self.load_chat_data(video_id)
        
        if not chat_data.get('intensity_timeline'):
            return "No chat data available."
        
        # 格式化聊天强度时间线
        summary_lines = ["Chat Activity Timeline:"]
        
        # 只显示高强度时段
        timeline = chat_data['intensity_timeline']
        if timeline:
            avg_intensity = sum(t['intensity'] for t in timeline) / len(timeline)
            high_activity = [t for t in timeline if t['intensity'] > avg_intensity * 1.5]
            
            for entry in high_activity[:20]:  # 最多显示 20 个高峰
                summary_lines.append(
                    f"  [{entry['timestamp_str']}] {entry['message_count']} messages"
                )
        
        # 添加聊天峰值时刻
        if chat_data.get('peak_moments'):
            summary_lines.append("\nChat Peak Moments:")
            for peak in chat_data['peak_moments'][:10]:  # 最多 10 个峰值
                keywords = ', '.join(peak['keywords'][:5])
                summary_lines.append(
                    f"  [{peak['timestamp_str']}] Intensity: {peak['intensity']}, Keywords: {keywords}"
                )
        
        summary = '\n'.join(summary_lines)
        
        # 截断到最大长度
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    def get_chat_in_timerange(
        self, 
        video_id: str, 
        start_time: float, 
        end_time: float
    ) -> Dict:
        """
        获取指定时间范围内的聊天数据
        
        Args:
            video_id: 视频 ID
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
        
        Returns:
            该时间段的聊天数据摘要
        """
        chat_data = self.load_chat_data(video_id)
        
        # 过滤消息
        messages = [
            m for m in chat_data.get('messages', [])
            if start_time <= m['timestamp'] < end_time
        ]
        
        # 过滤强度数据
        intensity = [
            i for i in chat_data.get('intensity_timeline', [])
            if start_time <= i['timestamp'] < end_time
        ]
        
        # 过滤峰值
        peaks = [
            p for p in chat_data.get('peak_moments', [])
            if start_time <= p['timestamp'] < end_time
        ]
        
        return {
            'messages': messages,
            'intensity_timeline': intensity,
            'peak_moments': peaks,
            'total_messages': len(messages)
        }
    
    def format_highlights(self, highlights: List[Dict]) -> str:
        """
        将 highlights 格式化为训练用的字符串
        
        Args:
            highlights: highlight 列表
        
        Returns:
            格式化的字符串，例如:
            [00:15:30-00:18:45] exciting_moment: Team won the game
            [01:23:00-01:26:30] funny_moment: Unexpected game glitch
        """
        if not highlights:
            return "No highlights."
        
        lines = []
        for hl in highlights:
            line = f"[{hl['start_time_str']}-{hl['end_time_str']}] {hl['type']}"
            lines.append(line)
        
        return '\n'.join(lines)


class HighlightDataset(Dataset):
    """PyTorch Dataset for Highlight Detection"""
    
    def __init__(
        self,
        data: HighlightData,
        video_ids: List[str],
        tokenizer,
        max_length: int = 2048,
        include_chat: bool = True
    ):
        """
        Args:
            data: HighlightData instance
            video_ids: List of video IDs to include
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            include_chat: Whether to include chat data in prompt
        """
        self.data = data
        self.video_ids = video_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_chat = include_chat
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # 获取数据
        asr = self.data.get_asr(video_id)
        duration = self.data.get_duration(video_id, hms=True)
        highlights = self.data.load_highlights(video_id)
        
        # 构建 prompt
        prompt = self._build_prompt(video_id, asr, duration)
        
        # 构建 target（ground truth）
        target = self.data.format_highlights(highlights)
        
        # Tokenize
        full_text = f"{prompt}\n{target}"
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'video_id': video_id,
            'prompt': prompt,
            'target': target
        }
    
    def _build_prompt(self, video_id: str, asr: str, duration: str) -> str:
        """构建训练 prompt（简化格式：只输出时间+类型）"""
        prompt_parts = [
            "You are an expert at detecting highlight moments in live streaming videos.",
            "Given the video transcript and chat messages, identify exciting/funny/important moments.",
            "Output format: [START_TIME-END_TIME] TYPE",
            "",
            "Highlight types:",
            "- exciting_moment: Exciting gameplay, achievements, victories",
            "- funny_moment: Humorous situations, jokes, accidents",
            "- emotional_moment: Touching or dramatic moments",
            "- skill_showcase: Impressive plays or techniques",
            "- chat_peak: Moments with extremely high chat activity",
            "",
            f"Duration: {duration}",
            "",
            "Transcript:",
            asr[:5000],  # 限制 ASR 长度避免过长
        ]
        
        # 添加聊天数据
        if self.include_chat:
            chat_summary = self.data.get_chat_summary(video_id, max_length=1000)
            prompt_parts.extend([
                "",
                "Chat Activity:",
                chat_summary
            ])
        
        prompt_parts.extend([
            "",
            "Highlights:"
        ])
        
        return '\n'.join(prompt_parts)


class HighlightDataModule(L.LightningDataModule):
    """Lightning Data Module for Highlight Detection"""
    
    def __init__(
        self,
        vidc_dir: str = "dataset/highlights",
        train_subset: str = "train",
        val_subset: str = "val",
        batch_size: int = 1,
        max_length: int = 2048,
        include_chat: bool = True,
        num_workers: int = 0,
        # Additional config parameters (may be passed but not used)
        prompt: str = "highlight",
        subset: str = None,
        data_flags: str = "default",
        highlight_types: list = None,
        chat_window: int = 60,
        window_size: int = 35000,
        window_overlap: int = 300,
        **kwargs  # Catch any other unexpected parameters
    ):
        super().__init__()
        self.vidc_dir = vidc_dir
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.batch_size = batch_size
        self.max_length = max_length
        self.include_chat = include_chat
        self.num_workers = num_workers
        
        # Store additional config
        self.prompt = prompt
        self.subset = subset or train_subset
        self.data_flags = data_flags
        
        self.train_data = None
        self.val_data = None
        self.tokenizer = None
    
    def process(self, tokenize_dialog, tokenizer):
        """
        Process the dataset for training (used by llama_finetune.py)
        
        Args:
            tokenize_dialog: Function to tokenize dialogues
            tokenizer: Tokenizer to use
        
        Returns:
            Hugging Face Dataset with tokenized data
        """
        # Initialize data if not already done
        if self.train_data is None:
            self.train_data = HighlightData(self.vidc_dir, self.train_subset)
        
        # Get all video IDs
        video_ids = self._get_video_ids(self.train_subset)
        
        # Prepare dialogs in the format expected by tokenize_dialog
        dialogs = []
        valid_video_ids = []
        
        for vid_id in video_ids:
            try:
                # Get ASR and highlights
                asr = self.train_data.get_asr(vid_id)
                highlights = self.train_data.load_highlights(vid_id)
                duration = self.train_data.get_duration(vid_id, hms=True)
                
                if not asr:
                    continue
                
                # Build prompt
                prompt = self._build_prompt(vid_id, asr, duration)
                
                # Build target (ground truth)
                target = self.train_data.format_highlights(highlights)
                
                # Format as dialog (list of message dicts)
                dialog = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target}
                ]
                
                dialogs.append(dialog)
                valid_video_ids.append(vid_id)
            except Exception as e:
                print(f"Warning: Failed to process video {vid_id}: {e}")
                continue
        
        print(f"[INFO] Processed {len(dialogs)} training examples from {len(video_ids)} videos")
        
        # Create HF dataset
        dataset = HFDataset.from_dict({
            "dialog": dialogs,
            "vid_id": valid_video_ids
        })
        
        # Tokenize
        dataset = dataset.map(
            lambda x: tokenize_dialog(x["dialog"], tokenizer),
            remove_columns=["dialog", "vid_id"],
        )
        
        return dataset
    
    def _build_prompt(self, video_id: str, asr: str, duration: str) -> str:
        """构建训练 prompt"""
        prompt_parts = [
            "You are an expert at detecting highlight moments in live streaming videos.",
            "Given the video transcript and chat messages, identify exciting/funny/important moments.",
            "Output format: [START_TIME-END_TIME] TYPE: DESCRIPTION",
            "",
            "Highlight types:",
            "- exciting_moment: Exciting gameplay, achievements, victories",
            "- funny_moment: Humorous situations, jokes, accidents",
            "- emotional_moment: Touching or dramatic moments",
            "- skill_showcase: Impressive plays or techniques",
            "- chat_peak: Moments with extremely high chat activity",
            "",
            f"Duration: {duration}",
            "",
            "Transcript:",
            asr[:5000],  # 限制 ASR 长度避免过长
        ]
        
        # 添加聊天数据
        if self.include_chat:
            chat_summary = self.train_data.get_chat_summary(video_id, max_length=1000)
            prompt_parts.extend([
                "",
                "Chat Activity:",
                chat_summary
            ])
        
        prompt_parts.extend([
            "",
            "Highlights:"
        ])
        
        return '\n'.join(prompt_parts)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        # Load data
        self.train_data = HighlightData(self.vidc_dir, self.train_subset)
        self.val_data = HighlightData(self.vidc_dir, self.val_subset)
        
        # Get video IDs from index
        train_video_ids = self._get_video_ids(self.train_subset)
        val_video_ids = self._get_video_ids(self.val_subset)
        
        print(f"[INFO] Loaded {len(train_video_ids)} training videos")
        print(f"[INFO] Loaded {len(val_video_ids)} validation videos")
    
    def _get_video_ids(self, subset: str) -> List[str]:
        """从索引文件获取视频 ID 列表"""
        index_file = Path(self.vidc_dir) / "index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
                return [v['video_id'] for v in index['videos']]
        else:
            # 如果没有索引，扫描目录
            highlights_dir = Path(self.vidc_dir) / "highlights"
            return [d.name for d in highlights_dir.iterdir() if d.is_dir()]
    
    def train_dataloader(self):
        video_ids = self._get_video_ids(self.train_subset)
        dataset = HighlightDataset(
            self.train_data,
            video_ids,
            self.tokenizer,
            self.max_length,
            self.include_chat
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        video_ids = self._get_video_ids(self.val_subset)
        dataset = HighlightDataset(
            self.val_data,
            video_ids,
            self.tokenizer,
            self.max_length,
            self.include_chat
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
