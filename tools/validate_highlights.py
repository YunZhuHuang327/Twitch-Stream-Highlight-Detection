"""
验证 Highlight 标注文件格式

使用方法:
    python tools/validate_highlights.py highlights.json
    python tools/validate_highlights.py --dir D:/streaming_data/highlights
"""

import json
import argparse
from pathlib import Path
from datetime import timedelta


def parse_timestamp(ts):
    """解析时间戳"""
    if isinstance(ts, (int, float)):
        return float(ts)
    
    parts = ts.split(':')
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60 + s
    else:
        return float(ts)


def format_timestamp(seconds):
    """格式化时间戳"""
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def validate_highlight_file(file_path, verbose=False):
    """
    验证单个 highlight 标注文件
    
    返回: (is_valid, errors, warnings, stats)
    """
    errors = []
    warnings = []
    stats = {
        'total': 0,
        'types': {},
        'avg_duration': 0,
        'total_duration': 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            highlights = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"JSON 解析错误: {e}"], [], stats
    except Exception as e:
        return False, [f"文件读取错误: {e}"], [], stats
    
    if not isinstance(highlights, list):
        return False, ["根元素必须是数组"], [], stats
    
    if len(highlights) == 0:
        warnings.append("文件为空，没有 highlight")
    
    valid_types = {
        'exciting_moment', 'funny_moment', 'skill_showcase',
        'emotional_moment', 'chat_peak', 'highlight'
    }
    
    durations = []
    
    for i, hl in enumerate(highlights):
        stats['total'] += 1
        
        # 检查必需字段
        if 'start_time' not in hl:
            errors.append(f"Highlight {i+1}: 缺少 'start_time' 字段")
            continue
        
        if 'end_time' not in hl:
            errors.append(f"Highlight {i+1}: 缺少 'end_time' 字段")
            continue
        
        # 解析时间
        try:
            start = parse_timestamp(hl['start_time'])
            end = parse_timestamp(hl['end_time'])
        except Exception as e:
            errors.append(f"Highlight {i+1}: 时间格式错误 - {e}")
            continue
        
        # 检查时间有效性
        if start < 0:
            errors.append(f"Highlight {i+1}: start_time 不能为负数")
        
        if end < 0:
            errors.append(f"Highlight {i+1}: end_time 不能为负数")
        
        if start >= end:
            errors.append(f"Highlight {i+1}: start_time ({format_timestamp(start)}) 必须小于 end_time ({format_timestamp(end)})")
        
        duration = end - start
        durations.append(duration)
        stats['total_duration'] += duration
        
        # 检查时长是否合理
        if duration < 5:
            warnings.append(f"Highlight {i+1}: 时长太短 ({duration:.1f}秒)，建议至少 5 秒")
        
        if duration > 600:  # 10 分钟
            warnings.append(f"Highlight {i+1}: 时长较长 ({duration/60:.1f}分钟)，确认是否正确")
        
        # 检查 type 字段
        if 'type' not in hl:
            warnings.append(f"Highlight {i+1}: 缺少 'type' 字段（可选但推荐）")
        else:
            hl_type = hl['type']
            stats['types'][hl_type] = stats['types'].get(hl_type, 0) + 1
            
            if hl_type not in valid_types:
                warnings.append(
                    f"Highlight {i+1}: 未知的 type '{hl_type}'。"
                    f"推荐使用: {', '.join(valid_types)}"
                )
        
        # 检查是否有不推荐的字段
        if 'description' in hl:
            if verbose:
                warnings.append(f"Highlight {i+1}: 包含 'description' 字段（简化格式不需要）")
        
        if verbose:
            print(f"  ✓ Highlight {i+1}: [{format_timestamp(start)}-{format_timestamp(end)}] {hl.get('type', 'N/A')}")
    
    # 计算统计信息
    if durations:
        stats['avg_duration'] = sum(durations) / len(durations)
    
    # 检查时间重叠
    sorted_highlights = sorted(highlights, key=lambda x: parse_timestamp(x['start_time']))
    for i in range(len(sorted_highlights) - 1):
        try:
            curr_end = parse_timestamp(sorted_highlights[i]['end_time'])
            next_start = parse_timestamp(sorted_highlights[i+1]['start_time'])
            
            if curr_end > next_start:
                warnings.append(
                    f"时间重叠: Highlight {i+1} ({format_timestamp(curr_end)}) "
                    f"与 Highlight {i+2} ({format_timestamp(next_start)}) 重叠"
                )
        except:
            pass
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings, stats


def main():
    parser = argparse.ArgumentParser(description="验证 Highlight 标注文件格式")
    parser.add_argument("file", nargs='?', help="要验证的 JSON 文件")
    parser.add_argument("--dir", help="验证整个目录中的所有 JSON 文件")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    files_to_check = []
    
    if args.dir:
        # 验证目录
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"❌ 目录不存在: {dir_path}")
            return
        
        files_to_check = list(dir_path.glob("*.json"))
        if not files_to_check:
            print(f"❌ 目录中没有找到 JSON 文件: {dir_path}")
            return
    elif args.file:
        # 验证单个文件
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return
        files_to_check = [file_path]
    else:
        print("❌ 请指定要验证的文件或目录")
        print("用法:")
        print("  python tools/validate_highlights.py highlights.json")
        print("  python tools/validate_highlights.py --dir D:/streaming_data/highlights")
        return
    
    print(f"\n{'='*60}")
    print(f"验证 {len(files_to_check)} 个文件")
    print(f"{'='*60}\n")
    
    valid_count = 0
    total_highlights = 0
    all_types = {}
    
    for file_path in files_to_check:
        print(f"📄 {file_path.name}")
        
        is_valid, errors, warnings, stats = validate_highlight_file(file_path, args.verbose)
        
        if is_valid:
            print(f"  ✅ 有效")
            valid_count += 1
        else:
            print(f"  ❌ 无效")
        
        if errors:
            print(f"  错误 ({len(errors)}):")
            for err in errors:
                print(f"    ❌ {err}")
        
        if warnings:
            print(f"  警告 ({len(warnings)}):")
            for warn in warnings:
                print(f"    ⚠️ {warn}")
        
        # 统计信息
        print(f"  📊 统计:")
        print(f"    - Highlights: {stats['total']}")
        if stats['total'] > 0:
            print(f"    - 平均时长: {stats['avg_duration']:.1f} 秒")
            print(f"    - 总时长: {format_timestamp(stats['total_duration'])}")
            if stats['types']:
                print(f"    - 类型分布:")
                for t, count in sorted(stats['types'].items(), key=lambda x: -x[1]):
                    print(f"      • {t}: {count}")
                    all_types[t] = all_types.get(t, 0) + count
        
        total_highlights += stats['total']
        print()
    
    # 总结
    print(f"{'='*60}")
    print(f"验证完成")
    print(f"{'='*60}")
    print(f"✅ 有效文件: {valid_count}/{len(files_to_check)}")
    print(f"📊 总 Highlights: {total_highlights}")
    
    if all_types:
        print(f"\n类型分布（所有文件）:")
        for t, count in sorted(all_types.items(), key=lambda x: -x[1]):
            percentage = count / total_highlights * 100
            print(f"  • {t}: {count} ({percentage:.1f}%)")
    
    if valid_count == len(files_to_check):
        print(f"\n🎉 所有文件都通过验证！")
    else:
        print(f"\n⚠️ 有 {len(files_to_check) - valid_count} 个文件存在问题，请修正后重试")


if __name__ == "__main__":
    main()
