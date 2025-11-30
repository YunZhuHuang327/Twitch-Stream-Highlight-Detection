"""
Chapter segmentation from existing ASR text using Chapter-Llama

Usage:
    python chapter_from_asr_english.py \
        --asr_file "dataset/highlights/123/asr.txt" \
        --video_title "TwitchCon W/ AGENT00" \
        --output_dir "outputs/chapters/123"
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == 'win32':
    # Set standard output to UTF-8
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass
    # Set environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.data.utils_asr import PromptASR
from src.models.llama_inference import LlamaInference
from src.test.vidchapters import get_chapters
from tools.download.models import download_model


def safe_print(text):
    """Safe print to avoid encoding errors"""
    try:
        print(text)
    except (UnicodeEncodeError, UnicodeDecodeError):
        # If encoding error, remove unencodable characters
        safe_text = str(text).encode('ascii', 'ignore').decode('ascii')
        print(safe_text)


class ASRChapters:
    """Mock Chapters interface using existing ASR text"""

    def __init__(self, asr_file: Path, video_title: str):
        self.asr_file = Path(asr_file)
        self.video_title = video_title
        self.video_ids = [self.asr_file.stem]  # Use filename as ID

        # Read ASR text
        with open(self.asr_file, 'r', encoding='utf-8') as f:
            self.asr_lines = f.readlines()

        # Convert to Chapter-Llama expected format
        self.asr_text = self._convert_asr_format()

        # Calculate video duration
        self.duration = self._calculate_duration()

        safe_print(f"[OK] Loaded ASR file: {self.asr_file}")
        safe_print(f"  - Lines: {len(self.asr_lines)}")
        safe_print(f"  - Duration: {self._format_duration(self.duration)}")

    def _convert_asr_format(self, max_lines=2000) -> str:
        """
        Convert HH:MM:SS: text format to Chapter-Llama expected format
        Limit to max_lines to avoid exceeding model context length

        Input:  00:00:00: Empire State is this one
        Output: [00:00:00] Empire State is this one
        """
        converted_lines = []
        for line in self.asr_lines[:max_lines]:  # Limit lines
            line = line.strip()
            if not line:
                continue

            # Split timestamp and text
            if ':' in line:
                parts = line.split(':', 3)
                if len(parts) >= 4:
                    # HH:MM:SS: text
                    timestamp = f"{parts[0]}:{parts[1]}:{parts[2]}"
                    text = parts[3].strip()
                    converted_lines.append(f"[{timestamp}] {text}")

        if len(self.asr_lines) > max_lines:
            safe_print(f"[WARNING] Transcript truncated from {len(self.asr_lines)} to {max_lines} lines")

        return '\n'.join(converted_lines) + '\n'

    def _calculate_duration(self) -> float:
        """Calculate video duration from last line"""
        if not self.asr_lines:
            return 0.0

        # Extract time from last line
        last_line = self.asr_lines[-1].strip()
        if ':' in last_line:
            parts = last_line.split(':', 3)
            if len(parts) >= 3:
                try:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = int(parts[2].split(':')[0])  # Handle extra colons
                    return float(hours * 3600 + minutes * 60 + seconds + 10)  # +10s buffer
                except:
                    pass

        return 3600.0  # Default 1 hour

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def __len__(self):
        return len(self.video_ids)

    def __iter__(self):
        return iter(self.video_ids)

    def __contains__(self, vid_id):
        return vid_id in self.video_ids

    def get_duration(self, vid_id, hms=False):
        if hms:
            return self._format_duration(self.duration)
        return self.duration

    def get_asr(self, vid_id):
        return self.asr_text


def main(
    asr_file: str,
    video_title: str,
    output_dir: str,
    model: str = "asr-10k",
    base_model: str = None,
    quantization: str = None
):
    """
    Generate chapters from existing ASR text

    Args:
        asr_file: Path to ASR text file
        video_title: Video title (for context)
        output_dir: Output directory
        model: Chapter-Llama model
        base_model: Base model path
    """
    safe_print("\n" + "="*60)
    safe_print("Chapter-Llama: Generate chapters from ASR text")
    safe_print("="*60)
    safe_print(f"ASR file: {asr_file}")
    safe_print(f"Video title: {video_title}")
    safe_print(f"Output dir: {output_dir}")
    safe_print("="*60 + "\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load ASR
    chapters = ASRChapters(asr_file, video_title)

    # Create prompt
    prompt_builder = PromptASR(chapters=chapters)
    vid_id = chapters.video_ids[0]
    prompt = prompt_builder.get_prompt_test(vid_id)
    transcript = chapters.get_asr(vid_id)
    full_prompt = prompt + transcript

    safe_print("\nPrompt statistics:")
    safe_print(f"  - Prompt length: {len(prompt)} chars")
    safe_print(f"  - Transcript length: {len(transcript)} chars")
    safe_print(f"  - Total length: {len(full_prompt)} chars")
    safe_print(f"  - Estimated tokens: ~{len(full_prompt) // 4}")

    # Check model path
    model_path_obj = Path(model)
    if model_path_obj.exists():
        safe_print(f"\n[OK] Using local model: {model}")
        model_path = str(model_path_obj)
    else:
        safe_print(f"\n[INFO] Downloading model: {model}")
        model_path = download_model(model)
        if model_path is None:
            safe_print("[ERROR] Model download failed")
            return

    # Use default base model
    if base_model is None:
        base_model = "meta-llama/Llama-3.1-8B-Instruct"

    safe_print(f"\nLoading model:")
    safe_print(f"  - Base model: {base_model}")
    safe_print(f"  - PEFT model: {model_path}")
    if quantization:
        safe_print(f"  - Quantization: {quantization}")

    # Create inference engine
    # Use quantization to reduce memory usage and avoid offloading issues
    inference = LlamaInference(
        ckpt_path=base_model,
        peft_model=model_path,
        quantization=quantization
    )

    # Generate chapters
    safe_print(f"\nGenerating chapters...")
    output_text, chapters_dict = get_chapters(
        inference,
        full_prompt,
        max_new_tokens=1024,
        do_sample=False,
        vid_id=vid_id,
    )

    # Debug output
    if chapters_dict is None:
        safe_print("\n[ERROR] No chapters generated!")
        safe_print("[DEBUG] Output text:")
        safe_print(str(output_text)[:1000])
        return

    # Display results
    safe_print("\n" + "="*60)
    safe_print("Generated Chapters")
    safe_print("="*60)
    for timestamp, title in chapters_dict.items():
        safe_print(f"  {timestamp}: {title}")
    safe_print("="*60)

    # Save results
    chapters_file = output_path / "chapters.json"
    with open(chapters_file, 'w', encoding='utf-8') as f:
        json.dump(chapters_dict, f, indent=2, ensure_ascii=False)

    output_text_file = output_path / "output_text.txt"
    with open(output_text_file, 'w', encoding='utf-8') as f:
        f.write(output_text)

    safe_print(f"\n[SUCCESS] Results saved:")
    safe_print(f"  - Chapters: {chapters_file}")
    safe_print(f"  - Full output: {output_text_file}")
    safe_print(f"  - Total chapters: {len(chapters_dict)}")

    safe_print("\n" + "="*60)
    safe_print("DONE!")
    safe_print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate chapters from existing ASR text using Chapter-Llama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

python chapter_from_asr_english.py \
    --asr_file "dataset/highlights/123/asr.txt" \
    --video_title "TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY" \
    --output_dir "outputs/chapters/123"

Using local model:
python chapter_from_asr_english.py \
    --asr_file "dataset/highlights/123/asr.txt" \
    --video_title "TwitchCon W/ AGENT00" \
    --output_dir "outputs/chapters/123" \
    --base_model "D:/chapter-llama/Llama-3.1-8B-Instruct"
        """
    )

    parser.add_argument('--asr_file', required=True,
                       help="Path to ASR text file (format: HH:MM:SS: text)")
    parser.add_argument('--video_title', required=True,
                       help="Video title (for LLM context)")
    parser.add_argument('--output_dir', required=True,
                       help="Output directory")
    parser.add_argument('--model', default='asr-10k',
                       help="Chapter-Llama model name or path (default: asr-10k)")
    parser.add_argument('--base_model', default=None,
                       help="Base model path (default: meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument('--quantization', default=None,
                       choices=[None, '4bit', '8bit'],
                       help="Quantization mode (4bit/8bit) to reduce memory usage")

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.asr_file).exists():
        safe_print(f"[ERROR] ASR file not found: {args.asr_file}")
        exit(1)

    main(
        asr_file=args.asr_file,
        video_title=args.video_title,
        output_dir=args.output_dir,
        model=args.model,
        base_model=args.base_model,
        quantization=args.quantization
    )
