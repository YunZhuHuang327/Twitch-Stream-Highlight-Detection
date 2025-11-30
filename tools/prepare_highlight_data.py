"""
数据准备工具：将你的视频、聊天数据和 ground truth 转换为训练格式

使用方法:
    # 基础用法（完整视频）
    python tools/prepare_highlight_data.py \
        --video_path "path/to/video.mp4" \
        --chat_file "path/to/chat.json" \
        --highlights_file "path/to/highlights.json" \
        --output_dir "dataset/highlights"
    
    # 长视频分段模式（推荐用于 6-8 小时视频）
    python tools/prepare_highlight_data.py \
        --video_path "path/to/long_stream.mp4" \
        --chat_file "path/to/long_stream_chat.json" \
        --highlights_file "path/to/long_stream_highlights.json" \
        --output_dir "dataset/highlights" \
        --segment_mode \
        --segment_window 1800 \
        --segment_overlap 300
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).parent.parent))

from src.data.single_video import get_asr


def parse_timestamp(ts_str):
    """
    解析时间戳字符串为秒数
    支持格式: "HH:MM:SS", "MM:SS", 或直接秒数
    """
    if isinstance(ts_str, (int, float)):
        return float(ts_str)
    
    parts = ts_str.split(':')
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60 + s
    else:
        return float(ts_str)


def format_timestamp(seconds):
    """将秒数转换为 HH:MM:SS 格式"""
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def process_chat_data(chat_file):
    """
    处理聊天室数据
    
    输入格式 (chat_file JSON):
    [
        {
            "timestamp": "00:15:30" or 930 (seconds),
            "user": "username",
            "message": "wow amazing!",
            "emotes": ["poggers", "fire"]  # optional
        },
        ...
    ]
    
    返回:
    {
        "messages": [...],
        "intensity_timeline": [{timestamp, intensity, message_count}],
        "peak_moments": [{timestamp, intensity, keywords}]
    }
    """
    with open(chat_file, 'r', encoding='utf-8') as f:
        raw_messages = json.load(f)
    
    # 转换时间戳为秒数
    messages = []
    for msg in raw_messages:
        timestamp_sec = parse_timestamp(msg['timestamp'])
        messages.append({
            'timestamp': timestamp_sec,
            'timestamp_str': format_timestamp(timestamp_sec),
            'user': msg.get('user', 'anonymous'),
            'message': msg.get('message', ''),
            'emotes': msg.get('emotes', [])
        })
    
    # 按时间排序
    messages.sort(key=lambda x: x['timestamp'])
    
    # 计算聊天强度（每分钟消息数）
    window_size = 60  # 60 秒窗口
    intensity_timeline = []
    peak_moments = []
    
    if messages:
        max_time = int(messages[-1]['timestamp'])
        for t in range(0, max_time, window_size):
            window_msgs = [m for m in messages if t <= m['timestamp'] < t + window_size]
            intensity = len(window_msgs)
            
            intensity_timeline.append({
                'timestamp': t,
                'timestamp_str': format_timestamp(t),
                'intensity': intensity,
                'message_count': len(window_msgs)
            })
            
            # 检测峰值（消息数 > 平均值的 2 倍）
            if len(intensity_timeline) > 10:
                avg_intensity = sum(x['intensity'] for x in intensity_timeline[-10:]) / 10
                if intensity > avg_intensity * 2:
                    # 提取关键词
                    keywords = extract_keywords(window_msgs)
                    peak_moments.append({
                        'timestamp': t,
                        'timestamp_str': format_timestamp(t),
                        'intensity': intensity,
                        'keywords': keywords
                    })
    
    return {
        'messages': messages,
        'intensity_timeline': intensity_timeline,
        'peak_moments': peak_moments,
        'total_messages': len(messages)
    }


def extract_keywords(messages, top_k=5):
    """从消息中提取最常见的关键词"""
    from collections import Counter
    import re
    
    words = []
    for msg in messages:
        # 简单的分词（可以改进）
        text = msg['message'].lower()
        # 提取英文单词和中文字符
        words.extend(re.findall(r'\b\w+\b', text))
        # 添加表情
        words.extend(msg.get('emotes', []))
    
    # 过滤常见停用词
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
    words = [w for w in words if w not in stopwords and len(w) > 1]
    
    return [word for word, count in Counter(words).most_common(top_k)]


def process_highlights(highlights_file, simplify=False):
    """
    处理 ground truth highlights
    
    输入格式 (highlights_file JSON):
    简化格式（推荐）:
    [
        {
            "start_time": "00:15:30" or 930,
            "end_time": "00:18:45" or 1125,
            "type": "exciting_moment"  # optional
        },
        ...
    ]
    
    完整格式:
    [
        {
            "start_time": "00:15:30" or 930,
            "end_time": "00:18:45" or 1125,
            "type": "exciting_moment",
            "description": "Team won the game"
        },
        ...
    ]
    """
    with open(highlights_file, 'r', encoding='utf-8') as f:
        raw_highlights = json.load(f)
    
    highlights = []
    for hl in raw_highlights:
        start_sec = parse_timestamp(hl['start_time'])
        end_sec = parse_timestamp(hl['end_time'])
        
        # 简化格式：只保留必要字段
        highlight_item = {
            'start_time': start_sec,
            'end_time': end_sec,
            'start_time_str': format_timestamp(start_sec),
            'end_time_str': format_timestamp(end_sec),
            'duration': end_sec - start_sec,
            'type': hl.get('type', 'highlight')
        }
        
        # 如果不简化，保留 description
        if not simplify and 'description' in hl:
            highlight_item['description'] = hl['description']
        
        highlights.append(highlight_item)
    
    return highlights


def extract_segment(asr_data, chat_data, start_time, end_time):
    """
    从完整数据中提取指定时间段的 ASR 和聊天数据
    
    Args:
        asr_data: 完整的 ASR 数据（带时间戳）
        chat_data: 完整的聊天数据
        start_time: 片段开始时间（秒）
        end_time: 片段结束时间（秒）
    
    Returns:
        segment_asr: 片段的 ASR 数据
        segment_chat: 片段的聊天数据
    """
    # 提取 ASR 片段
    segment_asr = []
    if isinstance(asr_data, list):
        for item in asr_data:
            if 'start' in item and 'end' in item:
                # 检查是否在时间范围内
                if item['end'] >= start_time and item['start'] <= end_time:
                    # 调整时间戳为相对于片段开始的时间
                    adjusted_item = item.copy()
                    adjusted_item['start'] = max(0, item['start'] - start_time)
                    adjusted_item['end'] = min(end_time - start_time, item['end'] - start_time)
                    segment_asr.append(adjusted_item)
    
    # 提取聊天片段
    segment_chat = {
        'messages': [],
        'intensity_timeline': [],
        'peak_moments': [],
        'total_messages': 0
    }
    
    if 'messages' in chat_data:
        for msg in chat_data['messages']:
            if start_time <= msg['timestamp'] <= end_time:
                # 调整时间戳为相对于片段开始的时间
                adjusted_msg = msg.copy()
                adjusted_msg['timestamp'] = msg['timestamp'] - start_time
                adjusted_msg['timestamp_str'] = format_timestamp(adjusted_msg['timestamp'])
                segment_chat['messages'].append(adjusted_msg)
        
        segment_chat['total_messages'] = len(segment_chat['messages'])
    
    return segment_asr, segment_chat


def create_segmented_data(video_path, chat_file, highlights_file, output_dir, 
                          segment_window=1800, segment_overlap=300):
    """
    创建分段的训练数据（用于长视频）
    
    Args:
        video_path: 视频文件路径
        chat_file: 聊天数据文件
        highlights_file: highlight 标注文件
        output_dir: 输出目录
        segment_window: 每个片段的长度（秒），默认 1800 = 30 分钟
        segment_overlap: 片段之间的重叠（秒），默认 300 = 5 分钟
    
    工作原理：
        6 小时视频 = [片段1: 0-30分] [片段2: 25-55分] [片段3: 50-80分] ...
        每个片段只包含该时间范围内的 ASR、聊天和 highlights
    """
    video_path = Path(video_path)
    video_id = video_path.stem
    
    print(f"\n🎬 处理长视频（分段模式）: {video_path.name}")
    print(f"   片段长度: {segment_window}秒 ({segment_window//60}分钟)")
    print(f"   片段重叠: {segment_overlap}秒 ({segment_overlap//60}分钟)")
    
    # 1. 提取 ASR
    print("\n📝 提取 ASR 转录...")
    asr_text, asr_with_timestamps, duration = get_asr(str(video_path), return_timestamps=True)

    print(f"   ✓ 视频时长: {format_timestamp(duration)} ({duration:.1f} 秒)")
    print(f"   ✓ ASR 文本长度: {len(asr_text)} 字符")
    
    # 2. 处理聊天数据
    print("\n💬 处理聊天数据...")
    chat_data = process_chat_data(chat_file)
    print(f"   ✓ 聊天消息数: {chat_data['total_messages']}")
    
    # 3. 处理 highlights（简化格式）
    print("\n⭐ 处理 Highlights...")
    all_highlights = process_highlights(highlights_file, simplify=True)
    print(f"   ✓ Highlights 数量: {len(all_highlights)}")
    
    # 4. 计算需要多少个片段
    step = segment_window - segment_overlap
    num_segments = max(1, int((duration - segment_overlap) / step) + 1)
    
    print(f"\n✂️ 将视频切分为 {num_segments} 个片段...")
    
    created_segments = 0
    total_segment_highlights = 0
    
    for i in range(num_segments):
        segment_start = i * step
        segment_end = min(segment_start + segment_window, duration)
        
        # 如果片段太短，跳过
        if segment_end - segment_start < 300:  # 至少 5 分钟
            continue
        
        # 找出这个片段中的 highlights
        segment_highlights = []
        for hl in all_highlights:
            # Highlight 的中心点在片段范围内
            hl_center = (hl['start_time'] + hl['end_time']) / 2
            if segment_start <= hl_center < segment_end:
                # 调整 highlight 时间为相对于片段开始的时间
                adjusted_hl = hl.copy()
                adjusted_hl['start_time'] = max(0, hl['start_time'] - segment_start)
                adjusted_hl['end_time'] = min(segment_end - segment_start, hl['end_time'] - segment_start)
                adjusted_hl['start_time_str'] = format_timestamp(adjusted_hl['start_time'])
                adjusted_hl['end_time_str'] = format_timestamp(adjusted_hl['end_time'])
                adjusted_hl['duration'] = adjusted_hl['end_time'] - adjusted_hl['start_time']
                segment_highlights.append(adjusted_hl)
        
        # 如果这个片段没有 highlights，跳过（可选）
        # if not segment_highlights:
        #     continue
        
        # 提取片段的 ASR 和聊天数据
        segment_asr, segment_chat = extract_segment(
            asr_with_timestamps, chat_data, segment_start, segment_end
        )
        
        # 生成片段 ASR 文本
        segment_asr_text = ""
        for item in segment_asr:
            segment_asr_text += item.get('text', '') + " "
        segment_asr_text = segment_asr_text.strip()
        
        # 创建片段输出目录
        segment_id = f"{video_id}_seg{i:03d}"
        segment_dir = Path(output_dir) / segment_id
        segment_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存片段数据
        # ASR 文本
        with open(segment_dir / "asr.txt", 'w', encoding='utf-8') as f:
            f.write(segment_asr_text)
        
        # ASR JSON（带时间戳）
        with open(segment_dir / "asr.json", 'w', encoding='utf-8') as f:
            json.dump(segment_asr, f, indent=2, ensure_ascii=False)
        
        # 聊天数据
        with open(segment_dir / "chat.json", 'w', encoding='utf-8') as f:
            json.dump(segment_chat, f, indent=2, ensure_ascii=False)
        
        # Highlights
        with open(segment_dir / "highlights.json", 'w', encoding='utf-8') as f:
            json.dump(segment_highlights, f, indent=2, ensure_ascii=False)
        
        # 片段时长
        segment_duration = segment_end - segment_start
        with open(segment_dir / "duration.txt", 'w', encoding='utf-8') as f:
            f.write(str(segment_duration))
        
        # 元数据
        metadata = {
            'video_id': segment_id,
            'original_video': video_id,
            'segment_index': i,
            'segment_start': segment_start,
            'segment_end': segment_end,
            'duration': segment_duration,
            'duration_str': format_timestamp(segment_duration),
            'num_highlights': len(segment_highlights),
            'num_chat_messages': len(segment_chat['messages']),
            'asr_length': len(segment_asr_text),
            'created_at': datetime.now().isoformat(),
            'source_video': str(video_path),
            'is_segment': True
        }
        
        with open(segment_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        created_segments += 1
        total_segment_highlights += len(segment_highlights)
        
        print(f"   ✓ 片段 {i+1}/{num_segments}: {segment_id}")
        print(f"      时间范围: {format_timestamp(segment_start)} - {format_timestamp(segment_end)}")
        print(f"      Highlights: {len(segment_highlights)}, 聊天消息: {len(segment_chat['messages'])}")
    
    print(f"\n✅ 分段完成!")
    print(f"   创建了 {created_segments} 个训练片段")
    print(f"   总 Highlights: {total_segment_highlights}")
    print(f"   输出目录: {output_dir}")
    
    return created_segments


def create_training_data(video_path, chat_file, highlights_file, output_dir, simplify=True):
    """
    创建完整的训练数据集（用于短视频或测试）
    
    输出目录结构:
    output_dir/
        video_id/
            asr.txt           - 视频转录
            asr.json          - 带时间戳的转录
            chat.json         - 聊天数据
            highlights.json   - Ground truth
            duration.txt      - 视频长度
            metadata.json     - 元数据
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    video_id = video_path.stem
    
    print(f"\n🎬 处理视频: {video_path.name}")
    
    # 创建输出目录
    video_dir = output_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 提取 ASR
    print("\n📝 提取 ASR 转录...")
    asr_text, asr_data, duration = get_asr(str(video_path), return_timestamps=True)

    print(f"   ✓ 视频时长: {format_timestamp(duration)} ({duration:.1f} 秒)")
    print(f"   ✓ ASR 文本长度: {len(asr_text)} 字符")
    
    with open(video_dir / "asr.txt", 'w', encoding='utf-8') as f:
        f.write(asr_text)
    
    with open(video_dir / "asr.json", 'w', encoding='utf-8') as f:
        json.dump(asr_data, f, indent=2, ensure_ascii=False)
    
    with open(video_dir / "duration.txt", 'w') as f:
        f.write(str(duration))
    
    # 2. 处理聊天数据
    print("\n💬 处理聊天数据...")
    chat_data = process_chat_data(chat_file)
    
    with open(video_dir / "chat.json", 'w', encoding='utf-8') as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ 共 {chat_data['total_messages']} 条消息")
    print(f"   ✓ 检测到 {len(chat_data['peak_moments'])} 个聊天高峰")
    
    # 3. 处理 highlights（使用简化格式）
    print("\n⭐ 处理 Highlights...")
    highlights = process_highlights(highlights_file, simplify=simplify)
    
    with open(video_dir / "highlights.json", 'w', encoding='utf-8') as f:
        json.dump(highlights, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ 共 {len(highlights)} 个 Highlights")
    for i, hl in enumerate(highlights, 1):
        if 'description' in hl and hl['description']:
            print(f"      {i}. [{hl['start_time_str']}-{hl['end_time_str']}] {hl['type']}: {hl['description'][:50]}")
        else:
            print(f"      {i}. [{hl['start_time_str']}-{hl['end_time_str']}] {hl['type']}")
    
    # 4. 创建元数据
    metadata = {
        'video_id': video_id,
        'video_path': str(video_path),
        'duration': duration,
        'duration_str': format_timestamp(duration),
        'num_highlights': len(highlights),
        'total_chat_messages': chat_data['total_messages'],
        'chat_peaks': len(chat_data['peak_moments']),
        'created_at': datetime.now().isoformat()
    }
    
    with open(video_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 数据准备完成！输出目录: {video_dir}")
    return video_dir


def create_dataset_index(output_dir):
    """
    创建数据集索引文件
    
    生成 dataset/highlights/index.json:
    {
        "videos": [
            {
                "video_id": "...",
                "duration": ...,
                "num_highlights": ...,
                ...
            }
        ],
        "total_videos": ...,
        "total_highlights": ...,
        "total_duration": ...
    }
    """
    output_dir = Path(output_dir)
    
    videos = []
    total_highlights = 0
    total_duration = 0
    
    for video_dir in output_dir.iterdir():
        if video_dir.is_dir():
            metadata_file = video_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    videos.append(metadata)
                    total_highlights += metadata['num_highlights']
                    total_duration += metadata['duration']
    
    index = {
        'videos': videos,
        'total_videos': len(videos),
        'total_highlights': total_highlights,
        'total_duration': total_duration,
        'total_duration_str': format_timestamp(total_duration),
        'updated_at': datetime.now().isoformat()
    }
    
    with open(output_dir / "index.json", 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 数据集索引已更新: {output_dir / 'index.json'}")
    print(f"   总视频数: {len(videos)}")
    print(f"   总 Highlights: {total_highlights}")
    print(f"   总时长: {format_timestamp(total_duration)}")


def main():
    parser = argparse.ArgumentParser(
        description="准备 Highlight 检测训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 基础模式（短视频或测试）:
   python tools/prepare_highlight_data.py \\
       --video_path "test_data/v1.mp4" \\
       --chat_file "test_data/chat_v1.json" \\
       --highlights_file "test_data/highlights_v1.json" \\
       --output_dir "dataset/highlights"

2. 分段模式（6-8 小时长视频）:
   python tools/prepare_highlight_data.py \\
       --video_path "D:/streaming_data/videos/long_stream.mp4" \\
       --chat_file "D:/streaming_data/chats/long_stream.json" \\
       --highlights_file "D:/streaming_data/highlights/long_stream.json" \\
       --output_dir "dataset/highlights" \\
       --segment_mode \\
       --segment_window 1800 \\
       --segment_overlap 300

3. 简化标注格式（只保留 type，去掉 description）:
   python tools/prepare_highlight_data.py \\
       --video_path "video.mp4" \\
       --chat_file "chat.json" \\
       --highlights_file "highlights.json" \\
       --simplify
        """
    )
    
    parser.add_argument("--video_path", required=True, 
                       help="视频文件路径")
    parser.add_argument("--chat_file", required=True, 
                       help="聊天数据 JSON 文件")
    parser.add_argument("--highlights_file", required=True, 
                       help="Ground truth highlights JSON 文件")
    parser.add_argument("--output_dir", default="dataset/highlights", 
                       help="输出目录（默认: dataset/highlights）")
    
    # 分段模式选项
    parser.add_argument("--segment_mode", action="store_true",
                       help="启用分段模式（用于长视频，如 6-8 小时）")
    parser.add_argument("--segment_window", type=int, default=1800,
                       help="每个片段的长度（秒），默认 1800 = 30 分钟")
    parser.add_argument("--segment_overlap", type=int, default=300,
                       help="片段之间的重叠（秒），默认 300 = 5 分钟")
    
    # 格式选项
    parser.add_argument("--simplify", action="store_true",
                       help="使用简化格式（只保留 type，去掉 description）")
    
    # 其他选项
    parser.add_argument("--update_index", action="store_true", 
                       help="处理完成后更新数据集索引")
    
    args = parser.parse_args()
    
    # 根据模式处理视频
    if args.segment_mode:
        print("\n" + "="*60)
        print("🔄 分段模式 - 适用于长视频（6-8 小时）")
        print("="*60)
        
        create_segmented_data(
            args.video_path,
            args.chat_file,
            args.highlights_file,
            args.output_dir,
            segment_window=args.segment_window,
            segment_overlap=args.segment_overlap
        )
    else:
        print("\n" + "="*60)
        print("📦 标准模式 - 适用于短视频或测试")
        print("="*60)
        
        create_training_data(
            args.video_path,
            args.chat_file,
            args.highlights_file,
            args.output_dir,
            simplify=args.simplify
        )
    
    # 更新索引
    if args.update_index:
        create_dataset_index(args.output_dir)


if __name__ == "__main__":
    main()
