"""
Quick Chapter Segmentation using Free LLM APIs

Supports:
- Groq (FREE, fast, Llama 3.1 70B) - RECOMMENDED
- OpenAI GPT-4o-mini (paid)

Usage:
    python quick_chapter.py --api groq --api_key YOUR_GROQ_KEY

Or set environment variable:
    set GROQ_API_KEY=your-key-here
    python quick_chapter.py --api groq

    
    python quick_chapter.py --api openai --api_key YOUR_KEY --yes

"""

import json
import os
import argparse
from pathlib import Path


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

    # Try to calculate duration from last line
    # Support both formats:
    # 1. Original: HH:MM:SS: text
    # 2. Merged: HH:MM:SS [ASR] text or HH:MM:SS [CHAT_xxx]
    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith('['):
            continue

        try:
            # Extract timestamp (first part before space or bracket)
            timestamp_part = line.split()[0]
            if ':' in timestamp_part:
                parts = timestamp_part.split(':')
                if len(parts) >= 3:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = int(parts[2])
                    duration_sec = hours * 3600 + minutes * 60 + seconds
                    print(f"  - Duration: {format_timestamp(duration_sec)}")
                    break
        except:
            continue

    return ''.join(lines)


def chunk_asr_text(asr_text, max_lines=400, overlap=80):
    """Split ASR text into chunks with overlap for API processing

    Args:
        asr_text: The full ASR transcript
        max_lines: Maximum lines per chunk (default: 400 to stay under 12K TPM limit)
        overlap: Number of lines to overlap between chunks (default: 80)

    Returns:
        List of text chunks with overlap
    """
    lines = asr_text.split('\n')
    chunks = []

    step = max_lines - overlap  # Move forward by (max_lines - overlap) each time

    for i in range(0, len(lines), step):
        chunk = '\n'.join(lines[i:i+max_lines])
        if chunk.strip():
            chunks.append(chunk)

        # Break if we've covered all lines
        if i + max_lines >= len(lines):
            break

    return chunks


def segment_with_groq(asr_text, video_title, api_key, model="llama-3.3-70b-versatile"):
    """Use Groq API to segment the video (FREE!)"""
    from groq import Groq

    client = Groq(api_key=api_key)

    chunks = chunk_asr_text(asr_text, max_lines=400, overlap=80)
    print(f"\nProcessing {len(chunks)} chunks with Groq...")
    print(f"Model: {model}")
    print(f"Chunk size: 400 lines with 80 line overlap")

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


def segment_with_openai(asr_text, video_title, api_key, model="gpt-4o-mini"):
    """Use OpenAI API to segment the video"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    chunks = chunk_asr_text(asr_text, max_lines=400, overlap=80)
    print(f"\nProcessing {len(chunks)} chunks with OpenAI...")
    print(f"Chunk size: 400 lines with 80 line overlap")

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
        description="Quick chapter segmentation using free/paid LLM APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Use Groq (FREE!)
  python quick_chapter.py --api groq --api_key gsk_xxx

  # Use OpenAI
  python quick_chapter.py --api openai --api_key sk-xxx

  # Or set environment variables:
  set GROQ_API_KEY=gsk-xxx
  python quick_chapter.py --api groq
        """
    )

    parser.add_argument('--api',
                       default='groq',
                       choices=['groq', 'openai'],
                       help='Which API to use (default: groq, FREE!)')
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
                       help='API key (or set GROQ_API_KEY/OPENAI_API_KEY env var)')
    parser.add_argument('--model',
                       default=None,
                       help='Model to use (default: llama-3.1-70b-versatile for Groq, gpt-4o-mini for OpenAI)')
    parser.add_argument('--yes', '-y',
                       action='store_true',
                       help='Skip confirmation prompt')

    args = parser.parse_args()

    # Get API key
    if args.api == 'groq':
        api_key = args.api_key or os.getenv('GROQ_API_KEY')
        default_model = 'llama-3.3-70b-versatile'  # Updated model
        api_name = "Groq (FREE!)"
    elif args.api == 'openai':
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        default_model = 'gpt-4o-mini'
        api_name = "OpenAI"

    if not api_key:
        print(f"Error: {args.api.upper()} API key required!")
        if args.api == 'groq':
            print("  Get your FREE key at: https://console.groq.com/keys")
            print("  Then set: set GROQ_API_KEY=your-key-here")
        else:
            print("  Get your key at: https://platform.openai.com/api-keys")
            print("  Then set: set OPENAI_API_KEY=your-key-here")
        return 1

    model = args.model or default_model

    # Check if ASR file exists
    asr_file = Path(args.asr_file)
    if not asr_file.exists():
        print(f"Error: ASR file not found: {asr_file}")
        return 1

    print("="*60)
    print(f"Quick Chapter Segmentation ({api_name})")
    print("="*60)
    print(f"API: {args.api}")
    print(f"Model: {model}")
    print(f"ASR file: {args.asr_file}")
    print(f"Video title: {args.video_title}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)

    # Load ASR
    asr_text = load_asr_file(asr_file)

    # Estimate cost/quota
    total_chars = len(asr_text)
    estimated_tokens = total_chars // 4

    if args.api == 'groq':
        print(f"\nEstimated tokens: ~{estimated_tokens:,}")
        print(f"Cost: FREE! (Groq offers free API)")
        print(f"Speed: Very fast (~10x faster than GPT-4o-mini)")
    else:
        cost_input = (estimated_tokens / 1_000_000) * 0.150
        cost_output = (1000 / 1_000_000) * 0.600
        total_cost = cost_input + cost_output
        print(f"\nEstimated cost: ${total_cost:.4f}")
        print(f"  (Input: {estimated_tokens:,} tokens = ${cost_input:.4f})")
        print(f"  (Output: ~1,000 tokens = ${cost_output:.4f})")

    # Ask for confirmation
    if not args.yes:
        response = input("\nProceed? (y/n): ").lower()
        if response != 'y':
            print("Cancelled.")
            return 0
    else:
        print("\nProceeding (--yes flag used)...")

    # Segment video
    print("\nSegmenting video...")
    if args.api == 'groq':
        chapters = segment_with_groq(asr_text, args.video_title, api_key, model)
    else:
        chapters = segment_with_openai(asr_text, args.video_title, api_key, model)

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
        f.write(f"API: {args.api}\n")
        f.write(f"Model: {model}\n")
        f.write("\n" + "="*60 + "\n")
        for timestamp, title in chapters.items():
            f.write(f"{timestamp}: {title}\n")

    print(f"  {summary_file}")

    print("\nYou can now use these chapters in the highlight pipeline!")

    return 0


if __name__ == "__main__":
    exit(main())
