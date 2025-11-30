"""
Extract highlights from pre-scored highlight_scores.json with custom thresholds
Allows re-processing without re-scoring (saves API costs)
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_scored_windows(json_path: str) -> List[Dict]:
    """Load pre-scored highlight windows"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_adjacent_windows(
    scored_windows: List[Dict],
    min_score: int = 7,
    max_gap: int = 30
) -> List[Dict]:
    """
    Merge adjacent high-scoring windows into continuous highlight segments

    Args:
        scored_windows: List of scored windows (must be sorted by time)
        min_score: Minimum score to consider as highlight
        max_gap: Maximum gap in seconds to merge windows (default 30s)

    Returns:
        List of merged highlight segments
    """
    # Filter windows with high scores
    high_score_windows = [w for w in scored_windows if w['highlight_score'] >= min_score]

    if not high_score_windows:
        return []

    # Sort by start time
    high_score_windows.sort(key=lambda x: x['start_time'])

    # Merge adjacent windows
    merged_segments = []
    current_segment = {
        'start_time': high_score_windows[0]['start_time'],
        'end_time': high_score_windows[0]['end_time'],
        'scores': [high_score_windows[0]['highlight_score']],
        'reasonings': [high_score_windows[0]['reasoning']],
        'key_moments': high_score_windows[0].get('key_moments', []).copy()
    }

    for window in high_score_windows[1:]:
        gap = window['start_time'] - current_segment['end_time']

        # If gap is small enough, merge into current segment
        if gap <= max_gap:
            current_segment['end_time'] = window['end_time']
            current_segment['scores'].append(window['highlight_score'])
            current_segment['reasonings'].append(window['reasoning'])
            current_segment['key_moments'].extend(window.get('key_moments', []))
        else:
            # Save current segment and start new one
            merged_segments.append(current_segment)
            current_segment = {
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                'scores': [window['highlight_score']],
                'reasonings': [window['reasoning']],
                'key_moments': window.get('key_moments', []).copy()
            }

    # Don't forget the last segment
    merged_segments.append(current_segment)

    # Format merged segments
    formatted_segments = []
    for segment in merged_segments:
        duration = segment['end_time'] - segment['start_time']
        avg_score = sum(segment['scores']) / len(segment['scores'])
        max_score = max(segment['scores'])
        min_score_in_segment = min(segment['scores'])

        # Combine reasonings (deduplicate similar ones)
        unique_reasonings = []
        for r in segment['reasonings']:
            if r not in unique_reasonings:
                unique_reasonings.append(r)

        formatted_segments.append({
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'start_timestamp': format_timestamp(segment['start_time']),
            'end_timestamp': format_timestamp(segment['end_time']),
            'duration': duration,
            'duration_formatted': format_timestamp(duration),
            'avg_score': round(avg_score, 1),
            'max_score': max_score,
            'min_score': min_score_in_segment,
            'num_windows': len(segment['scores']),
            'reasoning': ' | '.join(unique_reasonings[:3]),  # Top 3 reasonings
            'all_reasonings': unique_reasonings,
            'key_moments': segment['key_moments']
        })

    return formatted_segments


def filter_by_duration(
    segments: List[Dict],
    min_duration: int = None,
    max_duration: int = None
) -> List[Dict]:
    """
    Filter segments by duration

    Args:
        segments: List of merged segments
        min_duration: Minimum duration in seconds (optional)
        max_duration: Maximum duration in seconds (optional)

    Returns:
        Filtered list of segments
    """
    filtered = segments.copy()

    if min_duration is not None:
        filtered = [s for s in filtered if s['duration'] >= min_duration]

    if max_duration is not None:
        filtered = [s for s in filtered if s['duration'] <= max_duration]

    return filtered


def get_top_highlights(
    segments: List[Dict],
    top_n: int = None,
    sort_by: str = 'max_score'
) -> List[Dict]:
    """
    Get top N highlights

    Args:
        segments: List of merged segments
        top_n: How many to return (None = all)
        sort_by: 'max_score', 'avg_score', 'duration', or 'score_duration'

    Returns:
        Sorted list of top highlights
    """
    if sort_by == 'max_score':
        segments.sort(key=lambda x: (x['max_score'], x['duration']), reverse=True)
    elif sort_by == 'avg_score':
        segments.sort(key=lambda x: (x['avg_score'], x['duration']), reverse=True)
    elif sort_by == 'duration':
        segments.sort(key=lambda x: x['duration'], reverse=True)
    elif sort_by == 'score_duration':
        # Combined metric: avg_score * log(duration)
        import math
        segments.sort(key=lambda x: x['avg_score'] * math.log(x['duration'] + 1), reverse=True)
    else:
        segments.sort(key=lambda x: (x['max_score'], x['duration']), reverse=True)

    if top_n is not None:
        return segments[:top_n]
    return segments


def save_results(
    segments: List[Dict],
    output_dir: Path,
    config: Dict
):
    """Save extracted highlights"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save segments
    segments_file = output_dir / "extracted_highlights.json"
    with open(segments_file, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved highlights: {segments_file}")

    # Save summary
    summary_file = output_dir / "extraction_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("HIGHLIGHT EXTRACTION SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write("Extraction Configuration:\n")
        f.write(f"  Min score: {config['min_score']}\n")
        f.write(f"  Max gap for merging: {config['max_gap']}s\n")
        if config.get('min_duration'):
            f.write(f"  Min duration: {config['min_duration']}s\n")
        if config.get('max_duration'):
            f.write(f"  Max duration: {config['max_duration']}s\n")
        f.write(f"  Sort by: {config['sort_by']}\n")
        if config.get('top_n'):
            f.write(f"  Top N: {config['top_n']}\n")

        f.write(f"\nResults:\n")
        f.write(f"  Total segments: {len(segments)}\n")

        if segments:
            total_duration = sum(s['duration'] for s in segments)
            avg_duration = total_duration / len(segments)
            avg_score = sum(s['avg_score'] for s in segments) / len(segments)

            f.write(f"  Total duration: {format_timestamp(total_duration)}\n")
            f.write(f"  Average duration per segment: {format_timestamp(avg_duration)}\n")
            f.write(f"  Average score: {avg_score:.1f}/10\n")

        # Score distribution
        f.write(f"\nScore distribution (by max_score):\n")
        for score in range(10, 0, -1):
            count = sum(1 for s in segments if s['max_score'] == score)
            if count > 0:
                bar = "█" * count
                f.write(f"  {score:2d}: {count:3d} {bar}\n")

        # Duration distribution
        f.write(f"\nDuration distribution:\n")
        duration_ranges = [
            (0, 30, "0-30s"),
            (30, 60, "30s-1m"),
            (60, 120, "1-2m"),
            (120, 180, "2-3m"),
            (180, 300, "3-5m"),
            (300, float('inf'), ">5m")
        ]
        for min_d, max_d, label in duration_ranges:
            count = sum(1 for s in segments if min_d <= s['duration'] < max_d)
            if count > 0:
                bar = "█" * count
                f.write(f"  {label:>8}: {count:3d} {bar}\n")

        f.write(f"\n" + "="*70 + "\n")
        f.write(f"EXTRACTED HIGHLIGHTS\n")
        f.write("="*70 + "\n\n")

        for i, highlight in enumerate(segments, 1):
            f.write(f"{i}. [{highlight['start_timestamp']} - {highlight['end_timestamp']}] "
                   f"Duration: {highlight['duration_formatted']}\n")
            f.write(f"   Score: Avg {highlight['avg_score']}/10 | "
                   f"Max {highlight['max_score']}/10 | "
                   f"Min {highlight['min_score']}/10\n")
            f.write(f"   Windows merged: {highlight['num_windows']}\n")
            f.write(f"   Reasoning: {highlight['reasoning']}\n")

            if highlight.get('key_moments'):
                f.write(f"   Key moments ({len(highlight['key_moments'])}):\n")
                for moment in highlight['key_moments'][:5]:  # Show top 5
                    f.write(f"     - {moment}\n")
                if len(highlight['key_moments']) > 5:
                    f.write(f"     ... and {len(highlight['key_moments']) - 5} more\n")

            f.write("\n")

    print(f"[OK] Saved summary: {summary_file}")

    # Save FFmpeg cut commands
    ffmpeg_script = output_dir / "cut_highlights.sh"
    ffmpeg_script_ps = output_dir / "cut_highlights.ps1"

    # Bash script
    with open(ffmpeg_script, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("# FFmpeg commands to cut highlight segments\n")
        f.write("# Usage: bash cut_highlights.sh <input_video>\n\n")
        f.write('VIDEO="$1"\n\n')
        f.write('if [ -z "$VIDEO" ]; then\n')
        f.write('  echo "Usage: bash cut_highlights.sh <input_video>"\n')
        f.write('  exit 1\n')
        f.write('fi\n\n')

        for i, highlight in enumerate(segments, 1):
            output_name = f"highlight_{i:02d}_{highlight['start_timestamp'].replace(':', '-')}.mp4"
            f.write(f"# Highlight {i}: {highlight['start_timestamp']} - {highlight['end_timestamp']} "
                   f"(Score: {highlight['max_score']}/10)\n")
            f.write(f'ffmpeg -ss {highlight["start_timestamp"]} -to {highlight["end_timestamp"]} '
                   f'-i "$VIDEO" -c copy "{output_name}"\n\n')

    # PowerShell script
    with open(ffmpeg_script_ps, 'w', encoding='utf-8') as f:
        f.write("# FFmpeg commands to cut highlight segments\n")
        f.write("# Usage: .\\cut_highlights.ps1 -Video <input_video>\n\n")
        f.write('param(\n')
        f.write('    [Parameter(Mandatory=$true)]\n')
        f.write('    [string]$Video\n')
        f.write(')\n\n')

        for i, highlight in enumerate(segments, 1):
            output_name = f"highlight_{i:02d}_{highlight['start_timestamp'].replace(':', '-')}.mp4"
            f.write(f"# Highlight {i}: {highlight['start_timestamp']} - {highlight['end_timestamp']} "
                   f"(Score: {highlight['max_score']}/10)\n")
            f.write(f'ffmpeg -ss {highlight["start_timestamp"]} -to {highlight["end_timestamp"]} '
                   f'-i $Video -c copy "{output_name}"\n\n')

    print(f"[OK] Saved FFmpeg scripts: {ffmpeg_script} and {ffmpeg_script_ps}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract highlights from pre-scored highlight_scores.json with custom thresholds'
    )
    parser.add_argument('--input', required=True,
                       help='Path to highlight_scores.json')
    parser.add_argument('--output', required=True,
                       help='Output directory for extracted highlights')
    parser.add_argument('--min-score', type=int, default=7,
                       help='Minimum score threshold (default: 7)')
    parser.add_argument('--max-gap', type=int, default=30,
                       help='Maximum gap in seconds to merge windows (default: 30)')
    parser.add_argument('--min-duration', type=int, default=None,
                       help='Minimum duration in seconds (optional)')
    parser.add_argument('--max-duration', type=int, default=None,
                       help='Maximum duration in seconds (optional)')
    parser.add_argument('--top-n', type=int, default=None,
                       help='Extract only top N highlights (optional, default: all)')
    parser.add_argument('--sort-by', default='max_score',
                       choices=['max_score', 'avg_score', 'duration', 'score_duration'],
                       help='Sort highlights by (default: max_score)')

    args = parser.parse_args()

    print("="*70)
    print("HIGHLIGHT EXTRACTION")
    print("="*70)

    # Load scored windows
    print(f"\n[1/5] Loading pre-scored windows...")
    scored_windows = load_scored_windows(args.input)
    print(f"  Loaded {len(scored_windows)} scored windows")

    # Get score statistics
    scores = [w['highlight_score'] for w in scored_windows]
    print(f"  Score range: {min(scores)} - {max(scores)}")
    print(f"  Average score: {sum(scores) / len(scores):.1f}")

    high_score_count = sum(1 for s in scores if s >= args.min_score)
    print(f"  Windows with score >= {args.min_score}: {high_score_count}")

    # Merge adjacent windows
    print(f"\n[2/5] Merging adjacent high-scoring windows...")
    print(f"  Min score: {args.min_score}")
    print(f"  Max gap: {args.max_gap}s")

    merged_segments = merge_adjacent_windows(
        scored_windows,
        min_score=args.min_score,
        max_gap=args.max_gap
    )
    print(f"  Merged into {len(merged_segments)} segments")

    if not merged_segments:
        print("\n[WARNING] No segments found with the specified criteria!")
        print("Try lowering --min-score or increasing --max-gap")
        return

    # Filter by duration
    print(f"\n[3/5] Filtering by duration...")
    if args.min_duration or args.max_duration:
        before_count = len(merged_segments)
        merged_segments = filter_by_duration(
            merged_segments,
            min_duration=args.min_duration,
            max_duration=args.max_duration
        )
        print(f"  Filtered: {before_count} → {len(merged_segments)} segments")
    else:
        print(f"  No duration filter applied")

    if not merged_segments:
        print("\n[WARNING] No segments remaining after duration filter!")
        return

    # Sort and select top N
    print(f"\n[4/5] Sorting and selecting highlights...")
    print(f"  Sort by: {args.sort_by}")

    top_highlights = get_top_highlights(
        merged_segments,
        top_n=args.top_n,
        sort_by=args.sort_by
    )

    if args.top_n:
        print(f"  Selected top {len(top_highlights)} highlights")
    else:
        print(f"  All {len(top_highlights)} segments included")

    total_duration = sum(h['duration'] for h in top_highlights)
    print(f"  Total duration: {format_timestamp(total_duration)}")

    # Save results
    print(f"\n[5/5] Saving results...")
    output_dir = Path(args.output)

    config = {
        'min_score': args.min_score,
        'max_gap': args.max_gap,
        'min_duration': args.min_duration,
        'max_duration': args.max_duration,
        'top_n': args.top_n,
        'sort_by': args.sort_by
    }

    save_results(top_highlights, output_dir, config)

    # Print summary
    print(f"\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"Extracted {len(top_highlights)} highlights")
    print(f"Total duration: {format_timestamp(total_duration)}")
    print(f"\nFiles saved to: {output_dir}")
    print(f"  - extracted_highlights.json")
    print(f"  - extraction_summary.txt")
    print(f"  - cut_highlights.sh (bash)")
    print(f"  - cut_highlights.ps1 (PowerShell)")
    print("="*70)


if __name__ == "__main__":
    main()
