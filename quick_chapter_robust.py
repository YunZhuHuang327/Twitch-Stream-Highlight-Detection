"""
Robust chapter segmentation with retry logic and progress tracking

This script improves upon quick_chapter.py by:
1. Retrying failed chunks automatically
2. Saving progress to resume interrupted runs
3. Better error logging
4. Detecting missing time ranges
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import time


def time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to seconds"""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    return 0


def seconds_to_time(seconds: int) -> str:
    """Convert seconds to HH:MM:SS"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_asr_file(asr_file: str) -> Tuple[str, int]:
    """
    Load ASR file and return text + duration in seconds

    Supports two formats:
    1. Original ASR: HH:MM:SS: text
    2. Merged transcript: HH:MM:SS [ASR] text or HH:MM:SS [EVENT_LABEL]
    """
    with open(asr_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find video duration from last timestamp
    duration_sec = 0
    for line in reversed(lines):
        try:
            # Handle both formats
            timestamp_part = line.split()[0]

            # Remove colon suffix if present (original format: HH:MM:SS:)
            if timestamp_part.endswith(':'):
                timestamp_part = timestamp_part[:-1]

            if ':' in timestamp_part:
                parts = timestamp_part.split(':')
                if len(parts) >= 3:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = int(parts[2])
                    duration_sec = hours * 3600 + minutes * 60 + seconds
                    break
        except:
            continue

    asr_text = ''.join(lines)
    return asr_text, duration_sec


def chunk_asr_text(asr_text: str, max_lines: int = 1500, overlap: int = 150) -> List[Tuple[str, int, int]]:
    """
    Split ASR text into chunks with overlap

    Returns:
        List of (chunk_text, start_line, end_line)
    """
    lines = asr_text.split('\n')
    chunks = []
    step = max_lines - overlap

    for i in range(0, len(lines), step):
        chunk = '\n'.join(lines[i:i+max_lines])
        if chunk.strip():
            chunks.append((chunk, i, min(i+max_lines, len(lines))))

        if i + max_lines >= len(lines):
            break

    return chunks


def process_chunk_with_retry(
    client,
    model: str,
    video_title: str,
    chunk_text: str,
    chunk_idx: int,
    max_retries: int = 3
) -> Dict:
    """Process a single chunk with retry logic"""

    prompt = f"""You are an expert in video chapter segmentation.

Video Title: "{video_title}"

Task: Analyze the following ASR transcript and identify semantic chapter boundaries.

Guidelines:
- Look for topic changes, location changes, or activity transitions
- Typical chapter length: 10-30 minutes
- Use descriptive titles (e.g., "Exploring Little Italy", not "Chapter 1")
- Be concise and specific

ASR Transcript (with timestamps):
{chunk_text}

Output JSON only (no other text):
{{
  "HH:MM:SS": "Chapter title",
  "HH:MM:SS": "Next chapter",
  ...
}}

CRITICAL RULES - ABSOLUTELY MANDATORY:
1. COPY timestamps EXACTLY from the transcript above - DO NOT CHANGE THEM
2. If transcript shows "03:45:23", use "03:45:23" - NOT "03:45:00" or "00:00:00"
3. DO NOT start from 00:00:00 unless the transcript actually starts at 00:00:00
4. The timestamps in your output MUST match the time range in the transcript
5. Each chapter timestamp must be a line that EXISTS in the transcript above

WRONG Example:
Transcript contains "03:15:20 [ASR] Starting topic"
Output: {{"00:15:20": "Starting topic"}}  ← WRONG! Changed hour from 03 to 00

CORRECT Example:
Transcript contains "03:15:20 [ASR] Starting topic"
Output: {{"03:15:20": "Starting topic"}}  ← CORRECT! Exact match"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            result = response.choices[0].message.content.strip()

            # Try to extract JSON if wrapped in markdown
            if '```' in result:
                # Find JSON between code blocks
                parts = result.split('```')
                for part in parts:
                    part = part.strip()
                    if part.startswith('json'):
                        part = part[4:].strip()
                    if part.startswith('{') and part.endswith('}'):
                        result = part
                        break

            # Parse JSON
            chapters = json.loads(result)

            # Validate it's a dict
            if not isinstance(chapters, dict):
                raise ValueError("Response is not a dictionary")

            return chapters

        except json.JSONDecodeError as e:
            print(f"    Attempt {attempt+1}/{max_retries}: JSON parse error")
            if attempt < max_retries - 1:
                print(f"    Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"    Failed after {max_retries} attempts")
                print(f"    Raw response: {result[:300]}...")
                return {}

        except Exception as e:
            print(f"    Attempt {attempt+1}/{max_retries}: Error - {str(e)}")
            if attempt < max_retries - 1:
                print(f"    Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"    Failed after {max_retries} attempts")
                return {}

    return {}


def detect_missing_ranges(chapters: Dict, video_duration: int, gap_threshold: int = 600) -> List[Tuple[int, int]]:
    """
    Detect missing time ranges in chapters

    Args:
        chapters: Dict of {timestamp: title}
        video_duration: Total video duration in seconds
        gap_threshold: Minimum gap size to report (default 10 minutes)

    Returns:
        List of (start_sec, end_sec) for missing ranges
    """
    if not chapters:
        return [(0, video_duration)]

    # Convert timestamps to seconds and sort
    timestamps = sorted([time_to_seconds(ts) for ts in chapters.keys()])

    missing_ranges = []

    # Check gaps between consecutive chapters
    for i in range(len(timestamps) - 1):
        gap = timestamps[i+1] - timestamps[i]
        if gap > gap_threshold:
            missing_ranges.append((timestamps[i], timestamps[i+1]))

    return missing_ranges


def segment_with_openai(
    asr_text: str,
    video_title: str,
    api_key: str,
    video_duration: int,
    model: str = "gpt-4o-mini",
    progress_file: str = None
) -> Dict:
    """Segment video with OpenAI API using robust processing"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # Load existing progress if available
    all_chapters = {}
    if progress_file and Path(progress_file).exists():
        print(f"\n📂 Loading existing progress from {progress_file}")
        with open(progress_file, 'r', encoding='utf-8') as f:
            all_chapters = json.load(f)
        print(f"   Loaded {len(all_chapters)} existing chapters")

    # Create chunks
    chunks = chunk_asr_text(asr_text, max_lines=1500, overlap=150)

    print(f"\n🔄 Processing {len(chunks)} chunks with OpenAI...")
    print(f"   Model: {model}")
    print(f"   Chunk size: 1500 lines with 150 line overlap")
    print("-" * 60)

    failed_chunks = []

    for idx, (chunk_text, start_line, end_line) in enumerate(chunks):
        print(f"\n[{idx+1}/{len(chunks)}] Lines {start_line}-{end_line}")

        chunk_chapters = process_chunk_with_retry(
            client, model, video_title, chunk_text, idx
        )

        if chunk_chapters:
            # Merge chapters, track new vs updated timestamps
            new_count = 0
            updated_count = 0
            for timestamp, title in chunk_chapters.items():
                if timestamp not in all_chapters:
                    all_chapters[timestamp] = title
                    new_count += 1
                else:
                    # Update existing (from overlap region)
                    all_chapters[timestamp] = title
                    updated_count += 1

            print(f"  ✓ Found {len(chunk_chapters)} chapters ({new_count} new, {updated_count} updated from overlap)")

            # Save progress
            if progress_file:
                # Sort before saving
                sorted_progress = dict(sorted(all_chapters.items()))
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(sorted_progress, f, indent=2, ensure_ascii=False)
        else:
            print(f"  ✗ Failed to process chunk")
            failed_chunks.append((idx, start_line, end_line))

    # Report results
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Successful: {len(chunks) - len(failed_chunks)}")
    print(f"Failed: {len(failed_chunks)}")

    if failed_chunks:
        print(f"\n⚠️  Failed chunks:")
        for idx, start, end in failed_chunks:
            print(f"   Chunk {idx+1}: lines {start}-{end}")

    # Detect missing time ranges
    sorted_chapters = dict(sorted(all_chapters.items()))
    missing_ranges = detect_missing_ranges(sorted_chapters, video_duration)

    if missing_ranges:
        print(f"\n⚠️  Detected {len(missing_ranges)} missing time ranges:")
        for start_sec, end_sec in missing_ranges:
            duration_min = (end_sec - start_sec) // 60
            print(f"   {seconds_to_time(start_sec)} - {seconds_to_time(end_sec)} ({duration_min} minutes)")
        print(f"\n💡 Suggestion: Rerun the script to retry failed chunks")
    else:
        print(f"\n✓ No significant gaps detected!")

    print("=" * 60)

    return sorted_chapters


def main():
    parser = argparse.ArgumentParser(description="Robust chapter segmentation with retry logic")
    parser.add_argument('--api', choices=['openai'], default='openai', help='API to use')
    parser.add_argument('--asr_file', default='D:\chapter-llama\outputs\highlights\123\merged_transcript.txt', help='Path to ASR file')
    parser.add_argument('--api_key', help='API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to use')
    parser.add_argument('--output', default='outputs/chapters/123', help='Output directory')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not set!")
        return 1

    # Load ASR
    print("Loading ASR file...")
    asr_text, video_duration = load_asr_file(args.asr_file)

    num_lines = asr_text.count('\n')
    duration_str = seconds_to_time(video_duration)

    print(f"✓ Loaded ASR file")
    print(f"  Lines: {num_lines}")
    print(f"  Duration: {duration_str}")

    # Confirm
    if not args.yes:
        response = input("\nProceed? (y/n): ").lower()
        if response != 'y':
            print("Cancelled.")
            return 0

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_file = output_dir / "chapters_progress.json"

    # Segment
    chapters = segment_with_openai(
        asr_text,
        "TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY",
        api_key,
        video_duration,
        args.model,
        str(progress_file)
    )

    # Save final results
    chapters_file = output_dir / "chapters.json"
    with open(chapters_file, 'w', encoding='utf-8') as f:
        json.dump(chapters, f, indent=2, ensure_ascii=False)

    # Save summary
    summary_file = output_dir / "chapters_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Video: TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY\n")
        f.write(f"Total Chapters: {len(chapters)}\n")
        f.write(f"API: {args.api}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"\n{'='*60}\n")
        for timestamp, title in chapters.items():
            f.write(f"{timestamp}: {title}\n")

    print(f"\n✓ Saved {len(chapters)} chapters to:")
    print(f"  {chapters_file}")
    print(f"  {summary_file}")

    return 0


if __name__ == "__main__":
    exit(main())
