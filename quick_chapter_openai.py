"""
Quick Chapter Segmentation using GPT-4o-mini API

This script avoids the encoding issues of local Chapter-Llama execution
by using OpenAI's API instead.

Usage:
    python quick_chapter.py --api_key YOUR_API_KEY

Or set environment variable:
    set OPENAI_API_KEY=your-key-here
    python quick_chapter.py
"""

import json
import os
import argparse
from pathlib import Path
from openai import OpenAI


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_asr_file(asr_file):
    """Load ASR file and return text content"""
    print(f"Loading ASR file: {asr_file}")

    with open(asr_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  - Total lines: {len(lines)}")

    # Calculate duration from last line
    last_line = lines[-1].strip()
    if ':' in last_line:
        parts = last_line.split(':', 3)
        if len(parts) >= 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2].split(':')[0])
            duration_sec = hours * 3600 + minutes * 60 + seconds
            print(f"  - Duration: {format_timestamp(duration_sec)}")

    return ''.join(lines)


def chunk_asr_text(asr_text, max_lines=2000):
    """
    Split ASR text into chunks for API processing.
    GPT-4o-mini has 128K token limit, but we'll be conservative.
    """
    lines = asr_text.split('\n')
    chunks = []

    for i in range(0, len(lines), max_lines):
        chunk = '\n'.join(lines[i:i+max_lines])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def segment_with_api(asr_text, video_title, api_key, model="gpt-4o-mini"):
    """Use OpenAI API to segment the video"""

    client = OpenAI(api_key=api_key)

    # For very long videos, process in chunks
    chunks = chunk_asr_text(asr_text, max_lines=2000)
    print(f"\nProcessing {len(chunks)} chunks...")

    all_chapters = {}

    for idx, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {idx+1}/{len(chunks)}...")

        prompt = f"""You are an expert in video chapter segmentation.

Video Title: "{video_title}"

Task: Analyze the following ASR transcript and identify semantic chapter boundaries.

Guidelines:
- Look for topic changes, location changes, or activity transitions
- Typical chapter length: 10-30 minutes
- Use descriptive titles (e.g., "Exploring Little Italy", not "Chapter 1")
- Be concise and specific

ASR Transcript:
{chunk}

Output JSON only (no other text):
{{
  "00:00:00": "Chapter title",
  "00:15:30": "Next chapter",
  ...
}}

IMPORTANT: Only output valid JSON, nothing else."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            result = response.choices[0].message.content.strip()

            # Try to extract JSON if wrapped in markdown
            if result.startswith('```'):
                result = result.split('```')[1]
                if result.startswith('json'):
                    result = result[4:]
                result = result.strip()

            chunk_chapters = json.loads(result)
            all_chapters.update(chunk_chapters)

            print(f"  Found {len(chunk_chapters)} chapters in this chunk")

        except json.JSONDecodeError as e:
            print(f"  Warning: Failed to parse JSON from chunk {idx+1}")
            print(f"  Raw response: {result[:200]}...")
            continue
        except Exception as e:
            print(f"  Error processing chunk {idx+1}: {str(e)}")
            continue

    # Sort chapters by timestamp
    sorted_chapters = dict(sorted(all_chapters.items()))

    return sorted_chapters


def main():
    parser = argparse.ArgumentParser(
        description="Quick chapter segmentation using GPT-4o-mini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  python quick_chapter.py --api_key sk-xxx

  python quick_chapter.py --asr_file "path/to/asr.txt" --api_key sk-xxx

  set OPENAI_API_KEY=sk-xxx
  python quick_chapter.py
        """
    )

    parser.add_argument('--asr_file',
                       default='dataset/highlights/123/asr.txt',
                       help='Path to ASR text file')
    parser.add_argument('--video_title',
                       default='TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY',
                       help='Video title for context')
    parser.add_argument('--output_dir',
                       default='outputs/chapters/123',
                       help='Output directory')
    parser.add_argument('--api_key',
                       default=None,
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model',
                       default='gpt-4o-mini',
                       choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'],
                       help='OpenAI model to use')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required!")
        print("  Either set OPENAI_API_KEY environment variable")
        print("  Or use --api_key parameter")
        print("\nGet your API key at: https://platform.openai.com/api-keys")
        return 1

    # Check if ASR file exists
    asr_file = Path(args.asr_file)
    if not asr_file.exists():
        print(f"Error: ASR file not found: {asr_file}")
        return 1

    print("="*60)
    print("Quick Chapter Segmentation (GPT-4o-mini)")
    print("="*60)
    print(f"ASR file: {args.asr_file}")
    print(f"Video title: {args.video_title}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model: {args.model}")
    print("="*60)

    # Load ASR
    asr_text = load_asr_file(asr_file)

    # Estimate cost
    total_chars = len(asr_text)
    estimated_tokens = total_chars // 4
    if args.model == 'gpt-4o-mini':
        # $0.150 per 1M input tokens, $0.600 per 1M output tokens
        cost_input = (estimated_tokens / 1_000_000) * 0.150
        cost_output = (1000 / 1_000_000) * 0.600  # Assume 1000 output tokens
        total_cost = cost_input + cost_output
        print(f"\nEstimated cost: ${total_cost:.4f}")
        print(f"  (Input: {estimated_tokens:,} tokens = ${cost_input:.4f})")
        print(f"  (Output: ~1,000 tokens = ${cost_output:.4f})")

    # Ask for confirmation
    response = input("\nProceed? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return 0

    # Segment video
    print("\nSegmenting video...")
    chapters = segment_with_api(asr_text, args.video_title, api_key, args.model)

    if not chapters:
        print("\nError: No chapters generated!")
        return 1

    # Display results
    print("\n" + "="*60)
    print("Generated Chapters")
    print("="*60)
    for timestamp, title in list(chapters.items())[:10]:
        print(f"  {timestamp}: {title}")
    if len(chapters) > 10:
        print(f"  ... and {len(chapters)-10} more chapters")
    print("="*60)

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chapters_file = output_path / "chapters.json"
    with open(chapters_file, 'w', encoding='utf-8') as f:
        json.dump(chapters, f, indent=2, ensure_ascii=False)

    print(f"\nSuccess! Saved {len(chapters)} chapters to:")
    print(f"  {chapters_file}")

    # Save summary
    summary_file = output_path / "chapters_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Video: {args.video_title}\n")
        f.write(f"Total Chapters: {len(chapters)}\n")
        f.write(f"Model: {args.model}\n")
        f.write("\n" + "="*60 + "\n")
        for timestamp, title in chapters.items():
            f.write(f"{timestamp}: {title}\n")

    print(f"  {summary_file}")

    print("\nYou can now use these chapters in the highlight pipeline:")
    print(f"  python tools/highlight_detection_pipeline.py \\")
    print(f"      --video_path \"123.mp4\" \\")
    print(f"      --video_title \"{args.video_title}\" \\")
    print(f"      --chat_file \"123.json\" \\")
    print(f"      --asr_file \"{args.asr_file}\" \\")
    print(f"      --chapters_file \"{chapters_file}\" \\")
    print(f"      --output_dir \"outputs/highlights/123\"")

    return 0


if __name__ == "__main__":
    exit(main())
