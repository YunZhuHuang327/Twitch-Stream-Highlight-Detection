"""
Score highlight intensity for merged events using GPT-4o-mini
Uses sliding window approach to analyze events in temporal context
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from openai import OpenAI
import time


def parse_timestamp(timestamp_str: str) -> float:
    """Convert HH:MM:SS to seconds"""
    parts = timestamp_str.split(':')
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_merged_events(json_path: str) -> List[Dict]:
    """Load merged events JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_time_windows(
    events: List[Dict],
    window_size: int = 30,
    stride: int = 15
) -> List[Dict]:
    """
    Create sliding time windows over events

    Args:
        events: List of events sorted by timestamp
        window_size: Window size in seconds (default 30s)
        stride: How much to move window forward (default 15s for 50% overlap)

    Returns:
        List of windows, each containing:
        - start_time: Window start time in seconds
        - end_time: Window end time in seconds
        - events: Events in this window
    """
    if not events:
        return []

    start_sec = events[0]['timestamp_sec']
    end_sec = events[-1]['timestamp_sec']

    windows = []
    current_start = start_sec

    while current_start < end_sec:
        current_end = current_start + window_size

        # Get events in this window
        window_events = [
            e for e in events
            if current_start <= e['timestamp_sec'] < current_end
        ]

        if window_events:
            windows.append({
                'start_time': current_start,
                'end_time': current_end,
                'start_timestamp': format_timestamp(current_start),
                'end_timestamp': format_timestamp(current_end),
                'events': window_events
            })

        current_start += stride

    return windows


def format_window_for_prompt(window: Dict) -> str:
    """
    Format a time window into text for GPT prompt

    Args:
        window: Dict with start_time, end_time, events

    Returns:
        Formatted string with all events in the window
    """
    lines = []
    lines.append(f"Time Range: {window['start_timestamp']} - {window['end_timestamp']}")
    lines.append("")

    for event in window['events']:
        timestamp = event['timestamp']
        event_type = event['type']

        if event_type == 'ASR':
            text = event.get('text', '')
            lines.append(f"{timestamp} [ASR] {text}")

        elif event_type == 'CHAT_EVENT':
            label = event.get('event_label', '')
            text = event.get('text', '')
            if text:
                lines.append(f"{timestamp} [{label}] {text}")
            else:
                lines.append(f"{timestamp} [{label}]")

        elif event_type == 'VISUAL':
            description = event.get('description', '')
            lines.append(f"{timestamp} [VISUAL] {description}")

    return '\n'.join(lines)


def score_window_highlights(
    client: OpenAI,
    window: Dict,
    video_context: str = ""
) -> Dict:
    """
    Score highlight intensity for a time window using GPT-4o-mini

    Returns:
        Dict with:
        - start_time, end_time
        - highlight_score: 0-10 integer
        - reasoning: Why this score
        - key_moments: List of specific timestamps that are interesting
    """
    window_text = format_window_for_prompt(window)

    prompt = f"""You are an expert at identifying highlight moments in Twitch livestreams.

Video Context: {video_context}

Analyze the following 30-second segment and score its "highlight intensity" from 0-10:

{window_text}

Guidelines for scoring:
- 0-2: Boring, nothing interesting happening
- 3-4: Mildly interesting, some activity but not clip-worthy
- 5-6: Moderately interesting, potential for a decent clip
- 7-8: Very interesting, strong clip potential with excitement/reactions
- 9-10: AMAZING moment, definitely clip-worthy (big reactions, funny moments, intense gameplay, surprising events)

Consider:
- Chat activity spikes (CHAT_SPIKE_CLIP_MOMENT suggests excitement)
- Audience reactions and engagement
- Visual content (exciting scenes, funny moments)
- Speech content (jokes, reactions, interesting conversations)
- Unusual or surprising events

Respond in JSON format ONLY (no other text):
{{
  "highlight_score": <integer 0-10>,
  "reasoning": "<brief explanation of why this score>",
  "key_moments": ["HH:MM:SS description", ...]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        return {
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'start_timestamp': window['start_timestamp'],
            'end_timestamp': window['end_timestamp'],
            'highlight_score': result.get('highlight_score', 0),
            'reasoning': result.get('reasoning', ''),
            'key_moments': result.get('key_moments', []),
            'tokens_used': response.usage.total_tokens
        }

    except Exception as e:
        print(f"  [ERROR] Failed to score window {window['start_timestamp']}: {e}")
        return {
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'start_timestamp': window['start_timestamp'],
            'end_timestamp': window['end_timestamp'],
            'highlight_score': 0,
            'reasoning': f'Error: {str(e)}',
            'key_moments': [],
            'tokens_used': 0
        }


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
        List of merged highlight segments with:
        - start_time, end_time
        - duration
        - avg_score, max_score
        - reasoning (combined from all windows)
        - key_moments (combined)
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
        'key_moments': high_score_windows[0]['key_moments'].copy()
    }

    for window in high_score_windows[1:]:
        gap = window['start_time'] - current_segment['end_time']

        # If gap is small enough, merge into current segment
        if gap <= max_gap:
            current_segment['end_time'] = window['end_time']
            current_segment['scores'].append(window['highlight_score'])
            current_segment['reasonings'].append(window['reasoning'])
            current_segment['key_moments'].extend(window['key_moments'])
        else:
            # Save current segment and start new one
            merged_segments.append(current_segment)
            current_segment = {
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                'scores': [window['highlight_score']],
                'reasonings': [window['reasoning']],
                'key_moments': window['key_moments'].copy()
            }

    # Don't forget the last segment
    merged_segments.append(current_segment)

    # Format merged segments
    formatted_segments = []
    for segment in merged_segments:
        duration = segment['end_time'] - segment['start_time']
        avg_score = sum(segment['scores']) / len(segment['scores'])
        max_score = max(segment['scores'])

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
            'num_windows': len(segment['scores']),
            'reasoning': ' | '.join(unique_reasonings[:3]),  # Top 3 reasonings
            'key_moments': segment['key_moments']
        })

    return formatted_segments


def get_top_highlights(
    merged_segments: List[Dict],
    top_n: int = 20
) -> List[Dict]:
    """
    Get top N highlights sorted by max score and duration

    Args:
        merged_segments: List of merged highlight segments
        top_n: How many top highlights to return

    Returns:
        List of top highlights sorted by score
    """
    # Sort by max_score (primary) and duration (secondary)
    merged_segments.sort(key=lambda x: (x['max_score'], x['duration']), reverse=True)

    return merged_segments[:top_n]


def save_results(
    scored_windows: List[Dict],
    output_dir: Path,
    total_tokens: int,
    top_highlights: List[Dict],
    all_merged_segments: List[Dict] = None
):
    """Save scoring results"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all scored windows
    all_scores_file = output_dir / "highlight_scores.json"
    with open(all_scores_file, 'w', encoding='utf-8') as f:
        json.dump(scored_windows, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved all scores: {all_scores_file}")

    # Save all merged segments (if provided)
    if all_merged_segments:
        merged_segments_file = output_dir / "merged_segments.json"
        with open(merged_segments_file, 'w', encoding='utf-8') as f:
            json.dump(all_merged_segments, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved all merged segments: {merged_segments_file}")

    # Save top highlights
    top_highlights_file = output_dir / "top_highlights.json"
    with open(top_highlights_file, 'w', encoding='utf-8') as f:
        json.dump(top_highlights, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved top highlights: {top_highlights_file}")

    # Save summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("HIGHLIGHT SCORING SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total windows analyzed: {len(scored_windows)}\n")
        f.write(f"Total tokens used: {total_tokens:,}\n")

        # Cost estimation
        input_tokens = int(total_tokens * 0.85)  # Approximate input/output ratio
        output_tokens = total_tokens - input_tokens
        input_cost = input_tokens * 0.15 / 1_000_000
        output_cost = output_tokens * 0.60 / 1_000_000
        total_cost_usd = input_cost + output_cost
        total_cost_twd = total_cost_usd * 32

        f.write(f"\nEstimated cost:\n")
        f.write(f"  Input: {input_tokens:,} tokens × $0.15/1M = ${input_cost:.4f}\n")
        f.write(f"  Output: {output_tokens:,} tokens × $0.60/1M = ${output_cost:.4f}\n")
        f.write(f"  Total: ${total_cost_usd:.4f} USD (≈ NT$ {total_cost_twd:.2f})\n")

        # Score distribution
        f.write(f"\nScore distribution:\n")
        for score in range(11):
            count = sum(1 for w in scored_windows if w['highlight_score'] == score)
            if count > 0:
                bar = "█" * (count // 5 or 1)
                f.write(f"  {score:2d}: {count:4d} {bar}\n")

        f.write(f"\n" + "="*70 + "\n")
        f.write(f"TOP {len(top_highlights)} HIGHLIGHTS (MERGED SEGMENTS)\n")
        f.write("="*70 + "\n\n")

        for i, highlight in enumerate(top_highlights, 1):
            duration_str = highlight.get('duration_formatted', 'N/A')
            avg_score = highlight.get('avg_score', highlight.get('highlight_score', 0))
            max_score = highlight.get('max_score', highlight.get('highlight_score', 0))
            num_windows = highlight.get('num_windows', 1)

            f.write(f"{i}. [{highlight['start_timestamp']} - {highlight['end_timestamp']}] "
                   f"Duration: {duration_str}\n")
            f.write(f"   Score: Avg {avg_score}/10 | Max {max_score}/10 "
                   f"({num_windows} windows merged)\n")
            f.write(f"   Reasoning: {highlight['reasoning']}\n")
            if highlight.get('key_moments'):
                f.write(f"   Key moments:\n")
                for moment in highlight['key_moments'][:5]:  # Show top 5
                    f.write(f"     - {moment}\n")
            f.write("\n")

    print(f"[OK] Saved summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Score highlight intensity for merged events using GPT-4o-mini'
    )
    parser.add_argument('--input', required=True,
                       help='Path to merged_events.json')
    parser.add_argument('--output', required=True,
                       help='Output directory for results')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Window size in seconds (default: 30)')
    parser.add_argument('--stride', type=int, default=15,
                       help='Window stride in seconds (default: 15 for 50%% overlap)')
    parser.add_argument('--context', default="",
                       help='Video context description')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top highlights to extract (default: 20)')
    parser.add_argument('--min-score', type=int, default=7,
                       help='Minimum score for top highlights (default: 7)')

    args = parser.parse_args()

    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    print("="*70)
    print("HIGHLIGHT SCORING PIPELINE")
    print("="*70)

    # Load events
    print(f"\n[1/4] Loading merged events...")
    events = load_merged_events(args.input)
    print(f"  Loaded {len(events):,} events")

    if events:
        duration = events[-1]['timestamp_sec'] - events[0]['timestamp_sec']
        print(f"  Duration: {format_timestamp(duration)}")

    # Create time windows
    print(f"\n[2/4] Creating time windows...")
    print(f"  Window size: {args.window_size}s")
    print(f"  Stride: {args.stride}s")
    windows = create_time_windows(events, args.window_size, args.stride)
    print(f"  Created {len(windows):,} windows")

    # Estimate cost
    estimated_tokens = len(windows) * 800  # Average tokens per window
    estimated_cost_usd = estimated_tokens * 0.20 / 1_000_000
    estimated_cost_twd = estimated_cost_usd * 32
    print(f"\n  Estimated tokens: {estimated_tokens:,}")
    print(f"  Estimated cost: ${estimated_cost_usd:.4f} USD (≈ NT$ {estimated_cost_twd:.2f})")

    input("\nPress Enter to continue with scoring...")

    # Score windows
    print(f"\n[3/4] Scoring highlights...")
    scored_windows = []
    total_tokens = 0

    for i, window in enumerate(windows, 1):
        print(f"  [{i}/{len(windows)}] {window['start_timestamp']} - {window['end_timestamp']}", end=' ')

        result = score_window_highlights(client, window, args.context)
        scored_windows.append(result)
        total_tokens += result['tokens_used']

        print(f"→ Score: {result['highlight_score']}/10 (tokens: {result['tokens_used']})")

        # Rate limiting: ~500 requests per minute = ~120ms per request
        time.sleep(0.15)

    # Merge adjacent high-scoring windows
    print(f"\n[4/5] Merging adjacent high-scoring windows...")
    merged_segments = merge_adjacent_windows(scored_windows, args.min_score, max_gap=30)
    print(f"  Merged {sum(1 for w in scored_windows if w['highlight_score'] >= args.min_score)} "
          f"windows into {len(merged_segments)} continuous segments")

    # Get top highlights
    print(f"\n[5/5] Extracting top highlights...")
    top_highlights = get_top_highlights(merged_segments, args.top_n)
    print(f"  Selected top {len(top_highlights)} highlights")

    if top_highlights:
        total_duration = sum(h['duration'] for h in top_highlights)
        print(f"  Total highlight duration: {format_timestamp(total_duration)}")

    # Save results
    output_dir = Path(args.output)
    save_results(scored_windows, output_dir, total_tokens, top_highlights, merged_segments)

    # Final summary
    actual_cost_usd = total_tokens * 0.20 / 1_000_000
    actual_cost_twd = actual_cost_usd * 32

    print(f"\n" + "="*70)
    print(f"COMPLETE!")
    print(f"="*70)
    print(f"Total tokens used: {total_tokens:,}")
    print(f"Actual cost: ${actual_cost_usd:.4f} USD (≈ NT$ {actual_cost_twd:.2f})")
    print(f"\nResults saved to: {output_dir}")
    print(f"="*70)


if __name__ == "__main__":
    main()
