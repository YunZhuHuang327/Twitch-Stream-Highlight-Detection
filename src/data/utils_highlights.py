"""
Prompt utilities for Highlight Detection
"""

from .highlight_data import HighlightData


class PromptHighlight:
    """Generate prompts for highlight detection"""
    
    def __init__(self, data: HighlightData, include_chat: bool = True):
        """
        Args:
            data: HighlightData instance
            include_chat: Whether to include chat data in prompts
        """
        self.data = data
        self.include_chat = include_chat
    
    def get_prompt_train(self, video_id: str, max_asr_length: int = 8000) -> str:
        """
        Generate training prompt with ground truth highlights
        
        Args:
            video_id: Video ID
            max_asr_length: Maximum ASR text length to include
        
        Returns:
            Full prompt including transcript, chat, and ground truth
        """
        # 获取基础数据
        asr = self.data.get_asr(video_id)
        duration = self.data.get_duration(video_id, hms=True)
        highlights = self.data.load_highlights(video_id)
        
        # 截断 ASR
        if len(asr) > max_asr_length:
            asr = asr[:max_asr_length] + "...[truncated]"
        
        # 构建 prompt
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
            f"Video Duration: {duration}",
            "",
            "Transcript:",
            asr,
        ]
        
        # 添加聊天数据
        if self.include_chat:
            chat_summary = self.data.get_chat_summary(video_id, max_length=1500)
            prompt_parts.extend([
                "",
                "Chat Activity:",
                chat_summary
            ])
        
        # 添加 ground truth
        formatted_highlights = self.data.format_highlights(highlights)
        prompt_parts.extend([
            "",
            "Highlights:",
            formatted_highlights
        ])
        
        return '\n'.join(prompt_parts)
    
    def get_prompt_test(self, video_id: str, max_asr_length: int = 8000) -> str:
        """
        Generate inference prompt without ground truth
        
        Args:
            video_id: Video ID
            max_asr_length: Maximum ASR text length
        
        Returns:
            Prompt for inference (without ground truth)
        """
        # 获取基础数据
        asr = self.data.get_asr(video_id)
        duration = self.data.get_duration(video_id, hms=True)
        
        # 截断 ASR
        if len(asr) > max_asr_length:
            asr = asr[:max_asr_length] + "...[truncated]"
        
        # 构建 prompt
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
            f"Video Duration: {duration}",
            "",
            "Transcript:",
            asr,
        ]
        
        # 添加聊天数据
        if self.include_chat:
            chat_summary = self.data.get_chat_summary(video_id, max_length=1500)
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
    
    def get_prompt_window(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
        transcript_window: str
    ) -> str:
        """
        Generate prompt for a time window (for long videos)
        
        Args:
            video_id: Video ID
            start_time: Window start time (seconds)
            end_time: Window end time (seconds)
            transcript_window: ASR text for this window
        
        Returns:
            Prompt for this specific time window
        """
        from datetime import timedelta
        
        def format_time(seconds):
            td = timedelta(seconds=int(seconds))
            hours = td.seconds // 3600
            minutes = (td.seconds % 3600) // 60
            secs = td.seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        start_str = format_time(start_time)
        end_str = format_time(end_time)
        
        # 构建 prompt
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
            f"Time Window: {start_str} to {end_str}",
            "",
            "Transcript:",
            transcript_window,
        ]
        
        # 添加该窗口的聊天数据
        if self.include_chat:
            chat_window = self.data.get_chat_in_timerange(video_id, start_time, end_time)
            if chat_window['total_messages'] > 0:
                chat_summary = self._format_chat_window(chat_window)
                prompt_parts.extend([
                    "",
                    "Chat Activity in this window:",
                    chat_summary
                ])
        
        prompt_parts.extend([
            "",
            "Highlights:"
        ])
        
        return '\n'.join(prompt_parts)
    
    def _format_chat_window(self, chat_window: dict, max_length: int = 800) -> str:
        """Format chat data for a specific time window"""
        lines = []
        
        # 聊天强度
        if chat_window['intensity_timeline']:
            lines.append(f"Total messages: {chat_window['total_messages']}")
            
            # 显示高强度时段
            timeline = chat_window['intensity_timeline']
            if timeline:
                avg = sum(t['intensity'] for t in timeline) / len(timeline)
                high = [t for t in timeline if t['intensity'] > avg * 1.2]
                
                if high:
                    lines.append("High activity periods:")
                    for entry in high[:5]:
                        lines.append(f"  [{entry['timestamp_str']}] {entry['message_count']} msgs")
        
        # 峰值时刻
        if chat_window['peak_moments']:
            lines.append("Peak moments:")
            for peak in chat_window['peak_moments'][:3]:
                keywords = ', '.join(peak['keywords'][:5])
                lines.append(f"  [{peak['timestamp_str']}] Keywords: {keywords}")
        
        summary = '\n'.join(lines)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
