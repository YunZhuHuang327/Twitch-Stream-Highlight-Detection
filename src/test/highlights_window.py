"""
Highlight Detection - Sliding Window with Chat Integration

Modified from vidchapters_window.py to support:
- Window overlap to avoid missing highlights at boundaries
- Chat data integration
- Highlight segment output instead of chapter timestamps
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.utils_highlights import PromptHighlight
from src.data.highlight_data import HighlightData


def get_window_with_chat(
    prompt: str,
    transcript: list,
    chat_data: dict,
    tokenizer,
    start_time: float = 0,
    window_token_size: int = 35_000,
    overlap_seconds: int = 300,  # 5 minutes overlap
) -> Tuple[str, str, str, bool, float]:
    """
    Get windowed transcript + chat data with overlap
    
    Args:
        prompt: Base prompt text
        transcript: List of transcript lines with timestamps
        chat_data: Chat data dictionary
        tokenizer: Tokenizer for counting tokens
        start_time: Start time in seconds
        window_token_size: Maximum tokens per window
        overlap_seconds: Seconds to overlap between windows
    
    Returns:
        (prompt, windowed_transcript, chat_summary, reached_end, next_start_time)
    """
    # 计算当前 prompt 的 token 数
    prompt_tokens = len(tokenizer.encode(prompt))
    remaining_tokens = window_token_size - prompt_tokens
    
    # 构建窗口化的转录文本
    windowed_lines = []
    current_tokens = 0
    last_timestamp = start_time
    reached_end = False
    
    for line in transcript:
        # 解析时间戳 (假设格式: [HH:MM:SS] text)
        if line.startswith('['):
            try:
                time_str = line[1:line.index(']')]
                parts = time_str.split(':')
                if len(parts) == 3:
                    h, m, s = map(float, parts)
                    line_time = h * 3600 + m * 60 + s
                else:
                    line_time = float(time_str)
                
                # 跳过早于 start_time 的行
                if line_time < start_time:
                    continue
                
                last_timestamp = line_time
            except (ValueError, IndexError):
                pass
        
        # 计算添加这行后的 token 数
        line_tokens = len(tokenizer.encode(line))
        
        if current_tokens + line_tokens > remaining_tokens:
            # 达到 token 限制
            break
        
        windowed_lines.append(line)
        current_tokens += line_tokens
    else:
        # 所有行都处理完了
        reached_end = True
    
    windowed_transcript = '\n'.join(windowed_lines)
    
    # 获取该窗口的聊天数据摘要
    window_end = last_timestamp
    chat_summary = get_chat_in_window(chat_data, start_time, window_end)
    
    # 计算下一个窗口的起始时间（带重叠）
    next_start = max(start_time, last_timestamp - overlap_seconds)
    
    return prompt, windowed_transcript, chat_summary, reached_end, next_start


def get_chat_in_window(chat_data: dict, start_time: float, end_time: float) -> str:
    """
    获取指定时间窗口内的聊天数据摘要
    
    Args:
        chat_data: Chat data dictionary from HighlightData
        start_time: Window start time (seconds)
        end_time: Window end time (seconds)
    
    Returns:
        Formatted chat summary string
    """
    from datetime import timedelta
    
    def format_time(seconds):
        td = timedelta(seconds=int(seconds))
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    # 过滤该时间段的数据
    messages = [
        m for m in chat_data.get('messages', [])
        if start_time <= m['timestamp'] < end_time
    ]
    
    intensity = [
        i for i in chat_data.get('intensity_timeline', [])
        if start_time <= i['timestamp'] < end_time
    ]
    
    peaks = [
        p for p in chat_data.get('peak_moments', [])
        if start_time <= p['timestamp'] < end_time
    ]
    
    # 格式化输出
    lines = []
    
    if messages:
        lines.append(f"Total messages in window: {len(messages)}")
    
    if intensity:
        avg = sum(i['intensity'] for i in intensity) / len(intensity)
        high_activity = [i for i in intensity if i['intensity'] > avg * 1.2]
        
        if high_activity:
            lines.append("High activity periods:")
            for entry in high_activity[:5]:
                lines.append(f"  [{entry['timestamp_str']}] {entry['message_count']} messages")
    
    if peaks:
        lines.append("Peak moments:")
        for peak in peaks[:3]:
            keywords = ', '.join(peak['keywords'][:5])
            lines.append(f"  [{peak['timestamp_str']}] Keywords: {keywords}")
    
    if not lines:
        return "No significant chat activity in this window."
    
    return '\n'.join(lines)


def get_highlights(
    inference,
    prompt: str,
    transcript: list,
    chat_data: dict,
    tokenizer,
    window_token_size: int = 35_000,
    overlap_seconds: int = 300,
    max_windows: int = 100,
) -> List[Dict]:
    """
    Detect highlights in a video using sliding windows
    
    Args:
        inference: Inference function (model)
        prompt: Base prompt
        transcript: Full transcript with timestamps
        chat_data: Chat data dictionary
        tokenizer: Tokenizer
        window_token_size: Tokens per window
        overlap_seconds: Overlap between windows
        max_windows: Maximum number of windows to process
    
    Returns:
        List of detected highlights
    """
    all_highlights = []
    start_time = 0
    reached_end = False
    window_count = 0
    
    print(f"🔍 开始检测 highlights (window_size={window_token_size}, overlap={overlap_seconds}s)...")
    
    while not reached_end and window_count < max_windows:
        window_count += 1
        
        # 获取当前窗口
        window_prompt, windowed_transcript, chat_summary, reached_end, next_start = get_window_with_chat(
            prompt=prompt,
            transcript=transcript,
            chat_data=chat_data,
            tokenizer=tokenizer,
            start_time=start_time,
            window_token_size=window_token_size,
            overlap_seconds=overlap_seconds
        )
        
        # 构建完整 prompt
        full_prompt = f"{window_prompt}\n\nTranscript:\n{windowed_transcript}\n\nChat Activity:\n{chat_summary}\n\nHighlights:"
        
        print(f"  Window {window_count}: {format_time(start_time)} - {format_time(next_start)}")
        
        # 运行推理
        try:
            response = inference(full_prompt)
            
            # 解析 highlights
            window_highlights = parse_highlights(response, start_time)
            
            if window_highlights:
                print(f"    ✓ 检测到 {len(window_highlights)} 个 highlights")
                all_highlights.extend(window_highlights)
            else:
                print(f"    - 本窗口无 highlights")
        
        except Exception as e:
            print(f"    ✗ 推理失败: {e}")
        
        # 移动到下一个窗口
        start_time = next_start
        
        # 防止无限循环
        if next_start == start_time:
            break
    
    print(f"\n✅ 完成！共检测到 {len(all_highlights)} 个 highlights")
    
    # 去重和合并重叠的 highlights
    merged_highlights = merge_overlapping_highlights(all_highlights)
    
    if len(merged_highlights) < len(all_highlights):
        print(f"📊 合并重叠 highlights: {len(all_highlights)} → {len(merged_highlights)}")
    
    return merged_highlights


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS"""
    from datetime import timedelta
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_time(time_str: str) -> float:
    """Parse HH:MM:SS to seconds"""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60 + s
    else:
        return float(time_str)


def parse_highlights(response: str, window_start: float = 0) -> List[Dict]:
    """
    Parse model output to extract highlights
    
    Expected format:
    [00:15:30-00:18:45] exciting_moment: Team won the game
    [01:23:00-01:26:30] funny_moment: Unexpected glitch
    
    Args:
        response: Model output text
        window_start: Start time of current window (for relative timestamps)
    
    Returns:
        List of highlight dictionaries
    """
    highlights = []
    
    for line in response.split('\n'):
        line = line.strip()
        
        # 查找时间段 [START-END]
        if not line.startswith('['):
            continue
        
        try:
            # 提取时间段
            time_part = line[1:line.index(']')]
            if '-' not in time_part:
                continue
            
            start_str, end_str = time_part.split('-')
            start_time = parse_time(start_str.strip())
            end_time = parse_time(end_str.strip())
            
            # 提取类型和描述
            rest = line[line.index(']')+1:].strip()
            
            if ':' in rest:
                type_str, description = rest.split(':', 1)
                hl_type = type_str.strip()
                hl_desc = description.strip()
            else:
                hl_type = "highlight"
                hl_desc = rest
            
            highlights.append({
                'start_time': start_time,
                'end_time': end_time,
                'start_time_str': format_time(start_time),
                'end_time_str': format_time(end_time),
                'duration': end_time - start_time,
                'type': hl_type,
                'description': hl_desc
            })
        
        except (ValueError, IndexError) as e:
            # 解析失败，跳过该行
            continue
    
    return highlights


def merge_overlapping_highlights(highlights: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    合并重叠的 highlights（来自不同窗口的重复检测）
    
    Args:
        highlights: List of highlight dictionaries
        iou_threshold: IoU threshold for merging (0-1)
    
    Returns:
        Merged highlights list
    """
    if not highlights:
        return []
    
    # 按开始时间排序
    sorted_highlights = sorted(highlights, key=lambda x: x['start_time'])
    
    merged = [sorted_highlights[0]]
    
    for current in sorted_highlights[1:]:
        last = merged[-1]
        
        # 计算 IoU (Intersection over Union)
        intersection_start = max(last['start_time'], current['start_time'])
        intersection_end = min(last['end_time'], current['end_time'])
        intersection = max(0, intersection_end - intersection_start)
        
        union_start = min(last['start_time'], current['start_time'])
        union_end = max(last['end_time'], current['end_time'])
        union = union_end - union_start
        
        iou = intersection / union if union > 0 else 0
        
        if iou > iou_threshold:
            # 合并：取并集
            merged[-1] = {
                'start_time': union_start,
                'end_time': union_end,
                'start_time_str': format_time(union_start),
                'end_time_str': format_time(union_end),
                'duration': union_end - union_start,
                'type': last['type'],  # 保留第一个的类型
                'description': last['description']  # 可以合并描述
            }
        else:
            # 不重叠，添加新的
            merged.append(current)
    
    return merged


def save_highlights(highlights: List[Dict], output_file: str):
    """Save highlights to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(highlights, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Highlights 已保存到: {output_file}")


# 主函数示例
def main():
    """示例：使用滑动窗口检测 highlights"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect highlights in long videos")
    parser.add_argument("--video_id", required=True, help="Video ID")
    parser.add_argument("--data_dir", default="dataset/highlights", help="Data directory")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--output", default="output/highlights.json", help="Output JSON file")
    parser.add_argument("--window_size", type=int, default=35000, help="Window token size")
    parser.add_argument("--overlap", type=int, default=300, help="Overlap in seconds")
    
    args = parser.parse_args()
    
    # 加载数据
    data = HighlightData(args.data_dir)
    prompt_gen = PromptHighlight(data, include_chat=True)
    
    # 获取转录和聊天数据
    transcript = data.get_asr(args.video_id)  # 需要带时间戳的格式
    chat_data = data.load_chat_data(args.video_id)
    
    # 加载模型和 tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    def inference(prompt):
        """简单的推理函数"""
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 基础 prompt
    base_prompt = """You are an expert at detecting highlight moments.
Output format: [START_TIME-END_TIME] TYPE: DESCRIPTION"""
    
    # 检测 highlights
    highlights = get_highlights(
        inference=inference,
        prompt=base_prompt,
        transcript=transcript.split('\n'),  # 需要列表格式
        chat_data=chat_data,
        tokenizer=tokenizer,
        window_token_size=args.window_size,
        overlap_seconds=args.overlap
    )
    
    # 保存结果
    save_highlights(highlights, args.output)
    
    # 打印摘要
    print(f"\n📊 检测摘要:")
    print(f"  总 highlights: {len(highlights)}")
    
    type_counts = {}
    for hl in highlights:
        type_counts[hl['type']] = type_counts.get(hl['type'], 0) + 1
    
    print(f"  类型分布:")
    for hl_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {hl_type}: {count}")


if __name__ == "__main__":
    main()
