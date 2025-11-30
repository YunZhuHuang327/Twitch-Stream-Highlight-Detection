"""
批量处理多个视频的 highlight 数据准备

使用方法:
    python tools/batch_prepare_highlights.py \
        --data_dir "D:/streaming_data" \
        --output_dir "dataset/highlights" \
        --segment_mode \
        --segment_window 1800
"""

import json
import argparse
from pathlib import Path
import subprocess
import sys

def find_video_files(data_dir):
    """
    查找数据目录中的所有视频文件
    
    预期目录结构:
    data_dir/
        videos/
            stream_001.mp4
            stream_002.mp4
        chats/
            stream_001.json
            stream_002.json
        highlights/
            stream_001.json
            stream_002.json
    """
    data_dir = Path(data_dir)
    video_dir = data_dir / "videos"
    chat_dir = data_dir / "chats"
    highlight_dir = data_dir / "highlights"
    
    if not video_dir.exists():
        print(f"❌ 视频目录不存在: {video_dir}")
        return []
    
    video_files = []
    for video_path in video_dir.glob("*.mp4"):
        video_name = video_path.stem
        chat_file = chat_dir / f"{video_name}.json"
        highlight_file = highlight_dir / f"{video_name}.json"
        
        # 检查是否有对应的聊天和 highlight 文件
        if chat_file.exists() and highlight_file.exists():
            video_files.append({
                'video': video_path,
                'chat': chat_file,
                'highlights': highlight_file,
                'name': video_name
            })
        else:
            print(f"⚠️ 跳过 {video_name}: ", end="")
            if not chat_file.exists():
                print(f"缺少聊天文件 {chat_file.name}", end=" ")
            if not highlight_file.exists():
                print(f"缺少 highlight 文件 {highlight_file.name}", end=" ")
            print()
    
    return video_files


def process_videos(video_files, output_dir, segment_mode=False, 
                   segment_window=1800, segment_overlap=300, simplify=True):
    """批量处理视频文件"""
    
    total = len(video_files)
    success_count = 0
    failed_count = 0
    failed_videos = []
    
    print(f"\n{'='*60}")
    print(f"开始批量处理 {total} 个视频")
    print(f"{'='*60}\n")
    
    for i, video_info in enumerate(video_files, 1):
        print(f"\n[{i}/{total}] 处理: {video_info['name']}")
        print("-" * 60)
        
        # 构建命令
        cmd = [
            sys.executable,  # Python 解释器
            "tools/prepare_highlight_data.py",
            "--video_path", str(video_info['video']),
            "--chat_file", str(video_info['chat']),
            "--highlights_file", str(video_info['highlights']),
            "--output_dir", str(output_dir)
        ]
        
        if segment_mode:
            cmd.extend([
                "--segment_mode",
                "--segment_window", str(segment_window),
                "--segment_overlap", str(segment_overlap)
            ])
        
        if simplify:
            cmd.append("--simplify")
        
        # 执行命令
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            success_count += 1
            print(f"✅ 完成: {video_info['name']}")
        except subprocess.CalledProcessError as e:
            failed_count += 1
            failed_videos.append(video_info['name'])
            print(f"❌ 失败: {video_info['name']}")
            print(f"   错误信息: {e}")
    
    # 最终统计
    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count}/{total}")
    print(f"❌ 失败: {failed_count}/{total}")
    
    if failed_videos:
        print(f"\n失败的视频:")
        for name in failed_videos:
            print(f"  - {name}")
    
    # 更新数据集索引
    if success_count > 0:
        print(f"\n📋 更新数据集索引...")
        try:
            from prepare_highlight_data import create_dataset_index
            create_dataset_index(output_dir)
        except Exception as e:
            print(f"⚠️ 索引更新失败: {e}")
            print("   可以手动运行: python tools/prepare_highlight_data.py --update_index")


def main():
    parser = argparse.ArgumentParser(
        description="批量准备多个视频的 Highlight 检测训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 批量处理短视频:
   python tools/batch_prepare_highlights.py \\
       --data_dir "D:/streaming_data" \\
       --output_dir "dataset/highlights"

2. 批量处理长视频（分段模式）:
   python tools/batch_prepare_highlights.py \\
       --data_dir "D:/streaming_data" \\
       --output_dir "dataset/highlights" \\
       --segment_mode \\
       --segment_window 1800 \\
       --segment_overlap 300

预期数据目录结构:
data_dir/
    videos/
        stream_001.mp4
        stream_002.mp4
    chats/
        stream_001.json
        stream_002.json
    highlights/
        stream_001.json
        stream_002.json
        """
    )
    
    parser.add_argument("--data_dir", required=True,
                       help="数据根目录（包含 videos/, chats/, highlights/ 子目录）")
    parser.add_argument("--output_dir", default="dataset/highlights",
                       help="输出目录（默认: dataset/highlights）")
    
    # 分段模式选项
    parser.add_argument("--segment_mode", action="store_true",
                       help="启用分段模式（用于长视频）")
    parser.add_argument("--segment_window", type=int, default=1800,
                       help="每个片段的长度（秒），默认 1800 = 30 分钟")
    parser.add_argument("--segment_overlap", type=int, default=300,
                       help="片段之间的重叠（秒），默认 300 = 5 分钟")
    
    # 格式选项
    parser.add_argument("--simplify", action="store_true", default=True,
                       help="使用简化格式（只保留 type，去掉 description）")
    
    # 过滤选项
    parser.add_argument("--limit", type=int,
                       help="只处理前 N 个视频（用于测试）")
    
    args = parser.parse_args()
    
    # 查找所有视频文件
    print(f"\n🔍 扫描数据目录: {args.data_dir}")
    video_files = find_video_files(args.data_dir)
    
    if not video_files:
        print("\n❌ 没有找到可处理的视频文件!")
        print("\n请确保目录结构如下:")
        print("data_dir/")
        print("  ├── videos/")
        print("  │   ├── stream_001.mp4")
        print("  │   └── stream_002.mp4")
        print("  ├── chats/")
        print("  │   ├── stream_001.json")
        print("  │   └── stream_002.json")
        print("  └── highlights/")
        print("      ├── stream_001.json")
        print("      └── stream_002.json")
        return
    
    print(f"✓ 找到 {len(video_files)} 个可处理的视频")
    
    # 限制处理数量（用于测试）
    if args.limit and args.limit < len(video_files):
        video_files = video_files[:args.limit]
        print(f"⚠️ 限制处理前 {args.limit} 个视频")
    
    # 显示将要处理的视频
    print("\n将要处理的视频:")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf['name']}")
    
    # 确认
    response = input(f"\n确认处理这 {len(video_files)} 个视频? (y/n): ")
    if response.lower() != 'y':
        print("取消处理")
        return
    
    # 开始处理
    process_videos(
        video_files,
        args.output_dir,
        segment_mode=args.segment_mode,
        segment_window=args.segment_window,
        segment_overlap=args.segment_overlap,
        simplify=args.simplify
    )


if __name__ == "__main__":
    main()
