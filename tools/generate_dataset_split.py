"""
自动生成训练/验证集划分文件

运行此脚本会自动创建:
- dataset/docs/subset_data/train.json
- dataset/docs/subset_data/val.json
- dataset/docs/chapters.json
"""

import json
from pathlib import Path


def generate_dataset_split(highlight_dir="dataset/highlights", 
                          train_ratio=0.8,
                          output_dir="dataset/docs"):
    """
    自动生成数据集划分文件
    
    Args:
        highlight_dir: highlight 数据目录
        train_ratio: 训练集比例（默认 80%）
        output_dir: 输出目录
    """
    highlight_dir = Path(highlight_dir)
    output_dir = Path(output_dir)
    
    print(f"\n🔍 扫描目录: {highlight_dir}")
    
    # 查找所有片段/视频
    all_segments = []
    for item in sorted(highlight_dir.iterdir()):
        if item.is_dir():
            # 检查是否有必要的文件
            metadata_file = item / "metadata.json"
            if metadata_file.exists():
                all_segments.append(item.name)
            else:
                print(f"⚠️ 跳过 {item.name}: 缺少 metadata.json")
    
    if not all_segments:
        print("\n❌ 没有找到有效的数据！")
        print("请确保已运行 prepare_highlight_data.py 或 batch_prepare_highlights.py")
        return
    
    print(f"✓ 找到 {len(all_segments)} 个有效数据")
    
    # 划分训练集和验证集
    split_idx = int(len(all_segments) * train_ratio)
    train_segments = all_segments[:split_idx]
    val_segments = all_segments[split_idx:]
    
    print(f"\n📊 数据划分:")
    print(f"  训练集: {len(train_segments)} 个片段 ({train_ratio*100:.0f}%)")
    print(f"  验证集: {len(val_segments)} 个片段 ({(1-train_ratio)*100:.0f}%)")
    
    # 创建输出目录
    subset_dir = output_dir / "subset_data"
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存训练集
    train_file = subset_dir / "train.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_segments, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 训练集已保存: {train_file}")
    
    # 保存验证集
    val_file = subset_dir / "val.json"
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_segments, f, indent=2, ensure_ascii=False)
    print(f"✅ 验证集已保存: {val_file}")
    
    # 生成 chapters.json
    print(f"\n📝 生成 chapters.json...")
    chapters = {}
    
    for seg in all_segments:
        metadata_file = highlight_dir / seg / "metadata.json"
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                
                # 检查是否是分段数据
                if meta.get('is_segment', False):
                    # 分段数据
                    title = f"{meta.get('original_video', seg)} - Segment {meta.get('segment_index', 0)}"
                else:
                    # 完整视频数据
                    title = meta.get('video_id', seg)
                
                chapters[seg] = {
                    "title": title,
                    "duration": meta.get('duration', 0)
                }
        except Exception as e:
            print(f"⚠️ 处理 {seg} 时出错: {e}")
            # 使用默认值
            chapters[seg] = {
                "title": seg,
                "duration": 0
            }
    
    chapters_file = output_dir / "chapters.json"
    with open(chapters_file, 'w', encoding='utf-8') as f:
        json.dump(chapters, f, indent=2, ensure_ascii=False)
    print(f"✅ Chapters 已保存: {chapters_file}")
    
    # 显示示例
    print(f"\n📋 训练集前 5 个:")
    for i, seg in enumerate(train_segments[:5], 1):
        print(f"  {i}. {seg}")
    if len(train_segments) > 5:
        print(f"  ...")
    
    print(f"\n📋 验证集前 5 个:")
    for i, seg in enumerate(val_segments[:5], 1):
        print(f"  {i}. {seg}")
    if len(val_segments) > 5:
        print(f"  ...")
    
    print(f"\n✅ 数据集划分完成!")
    print(f"\n下一步: 运行训练")
    print(f"  cmd /c run_train_production.bat")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="自动生成训练/验证集划分")
    parser.add_argument("--highlight_dir", default="dataset/highlights",
                       help="Highlight 数据目录（默认: dataset/highlights）")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="训练集比例（默认: 0.8 = 80%%）")
    parser.add_argument("--output_dir", default="dataset/docs",
                       help="输出目录（默认: dataset/docs）")
    
    args = parser.parse_args()
    
    generate_dataset_split(
        highlight_dir=args.highlight_dir,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
