"""
VLM (Vision-Language Model) Analyzer for Video Frames

Supports:
- GPT-4o-mini (OpenAI) - Fast, cheap, good quality
- GPT-4o (OpenAI) - Best quality
- LLaVA (Local) - Free, slower

Usage:
    from tools.vlm_analyzer import VLMAnalyzer

    analyzer = VLMAnalyzer(model="gpt-4o-mini", api_key="your-key")
    description = analyzer.analyze_frame("frame.jpg", timestamp="00:05:30", chapter_title="Gameplay")
"""

import base64
import subprocess
from pathlib import Path
from typing import Dict, Optional
import json


class VLMAnalyzer:
    """Analyze video frames using Vision-Language Models"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize VLM Analyzer

        Args:
            model: "gpt-4o-mini", "gpt-4o", or "llava"
            api_key: OpenAI API key (required for GPT models)
        """
        self.model = model
        self.api_key = api_key

        if model in ["gpt-4o-mini", "gpt-4o"] and not api_key:
            raise ValueError(f"API key required for {model}")

        # Initialize model-specific clients
        if model in ["gpt-4o-mini", "gpt-4o"]:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        elif model == "llava":
            self._init_llava()

    def _init_llava(self):
        """Initialize LLaVA model"""
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        print("Loading LLaVA model (this may take a while on first run)...")
        model_name = "llava-hf/llava-1.5-13b-hf"  # or "llava-1.5-7b-hf" for smaller

        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.llava_processor = AutoProcessor.from_pretrained(model_name)
        print(f"LLaVA loaded on device: {self.llava_model.device}")

    def extract_frame(
        self,
        video_path: str,
        timestamp_sec: float,
        output_path: str,
        width: int = 1024,
        height: int = 1024
    ) -> bool:
        """
        Extract a single frame from video

        Args:
            video_path: Path to video file
            timestamp_sec: Time in seconds
            output_path: Where to save frame
            width: Frame width
            height: Frame height

        Returns:
            True if successful
        """
        try:
            cmd = [
                'ffmpeg',
                '-ss', str(timestamp_sec),
                '-i', video_path,
                '-vframes', '1',
                '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
                '-q:v', '2',
                '-y',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frame at {timestamp_sec}s: {e}")
            return False

    def analyze_frame(
        self,
        image_path: str,
        timestamp: str,
        chapter_title: str,
        context: Optional[str] = None
    ) -> Dict:
        """
        Analyze a video frame

        Args:
            image_path: Path to frame image
            timestamp: Timestamp in HH:MM:SS format
            chapter_title: Title of current chapter
            context: Optional context about the video

        Returns:
            Dict with analysis results
        """
        if self.model in ["gpt-4o-mini", "gpt-4o"]:
            return self._analyze_with_gpt(image_path, timestamp, chapter_title, context)
        elif self.model == "llava":
            return self._analyze_with_llava(image_path, timestamp, chapter_title, context)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _analyze_with_gpt(
        self,
        image_path: str,
        timestamp: str,
        chapter_title: str,
        context: Optional[str]
    ) -> Dict:
        """Analyze frame using GPT-4o or GPT-4o-mini"""
        base64_image = self._encode_image(image_path)

        # Build prompt
        prompt_parts = [
            f"Analyze this frame from a Twitch livestream at timestamp {timestamp}.",
            f"Current chapter: \"{chapter_title}\"",
        ]

        if context:
            prompt_parts.append(f"Video context: {context}")

        prompt_parts.extend([
            "",
            "Provide a concise analysis covering:",
            "1. Main activity/scene (what's happening)",
            "2. People visible and their emotions/reactions",
            "3. On-screen text, UI elements, or graphics",
            "4. Overall atmosphere and energy level",
            "",
            "Also identify if this moment contains:",
            "- Humorous, emotional, or socially engaging moment",
            "- Funny/entertaining interaction",
            "- Surprising or unexpected event",
            "- High-energy celebration",
            "- Tense or dramatic moment",
            "",
            "Be specific and concise. Focus on what makes this moment notable."
        ])

        prompt = "\n".join(prompt_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # Use "high" for better quality but more expensive
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.3
            )

            description = response.choices[0].message.content.strip()

            return {
                "timestamp": timestamp,
                "chapter": chapter_title,
                "description": description,
                "model": self.model,
                "success": True
            }

        except Exception as e:
            print(f"Error analyzing frame with {self.model}: {e}")
            return {
                "timestamp": timestamp,
                "chapter": chapter_title,
                "description": "",
                "model": self.model,
                "success": False,
                "error": str(e)
            }

    def _analyze_with_llava(
        self,
        image_path: str,
        timestamp: str,
        chapter_title: str,
        context: Optional[str]
    ) -> Dict:
        """Analyze frame using LLaVA"""
        from PIL import Image
        import torch

        image = Image.open(image_path)

        # Build prompt
        prompt_parts = [
            f"Analyze this frame from a Twitch livestream at timestamp {timestamp}.",
            f"Current chapter: \"{chapter_title}\"",
        ]

        if context:
            prompt_parts.append(f"Video context: {context}")

        prompt_parts.extend([
            "",
            "Describe:",
            "1. What's happening in the scene",
            "2. People's emotions/reactions",
            "3. Any text or graphics visible",
            "4. Overall atmosphere",
            "",
            "Be concise but specific."
        ])

        prompt_text = "\n".join(prompt_parts)

        try:
            # Prepare conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image"},
                    ],
                },
            ]

            # Apply chat template
            prompt_formatted = self.llava_processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = self.llava_processor(
                images=image,
                text=prompt_formatted,
                return_tensors="pt"
            ).to(self.llava_model.device)

            # Generate response
            with torch.no_grad():
                output = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.2
                )

            # Decode response
            response = self.llava_processor.decode(output[0], skip_special_tokens=True)

            # Extract only assistant's response
            if "ASSISTANT:" in response:
                description = response.split("ASSISTANT:")[-1].strip()
            else:
                description = response

            return {
                "timestamp": timestamp,
                "chapter": chapter_title,
                "description": description,
                "model": self.model,
                "success": True
            }

        except Exception as e:
            print(f"Error analyzing frame with LLaVA: {e}")
            return {
                "timestamp": timestamp,
                "chapter": chapter_title,
                "description": "",
                "model": self.model,
                "success": False,
                "error": str(e)
            }

    def analyze_video_chapters(
        self,
        video_path: str,
        chapters: Dict[str, str],
        output_dir: str,
        frames_per_chapter: int = 5,
        video_context: Optional[str] = None
    ) -> Dict:
        """
        Analyze frames from video chapters

        Args:
            video_path: Path to video file
            chapters: Dict of {timestamp: title}
            output_dir: Where to save frames and results
            frames_per_chapter: How many frames to extract per chapter
            video_context: Optional context about the video

        Returns:
            Dict of {timestamp: analysis_result}
        """
        output_dir = Path(output_dir)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Get actual video duration
        video_duration = self._get_video_duration(video_path)
        print(f"Video duration: {self._seconds_to_time(video_duration)} ({video_duration:.2f}s)")

        results = {}
        chapter_times = sorted(chapters.keys())

        print(f"\nAnalyzing {len(chapter_times)} chapters...")
        print(f"Frames per chapter: {frames_per_chapter}")
        print(f"Model: {self.model}")
        print("-" * 60)

        for i, timestamp in enumerate(chapter_times):
            chapter_title = chapters[timestamp]
            print(f"\n[{i+1}/{len(chapter_times)}] {timestamp}: {chapter_title}")

            chapter_start = self._time_to_seconds(timestamp)

            # Get chapter duration
            if i < len(chapter_times) - 1:
                next_timestamp = chapter_times[i + 1]
                chapter_end = self._time_to_seconds(next_timestamp)
            else:
                # For last chapter, use video duration
                chapter_end = video_duration

            duration = chapter_end - chapter_start

            # Skip if chapter starts beyond video duration
            if chapter_start >= video_duration:
                print(f"  [SKIP] Chapter starts beyond video duration")
                continue

            # Extract frames evenly across chapter
            chapter_results = []
            for frame_idx in range(frames_per_chapter):
                offset = (duration / (frames_per_chapter + 1)) * (frame_idx + 1)
                frame_time = chapter_start + offset

                # Skip if frame time exceeds video duration
                if frame_time >= video_duration:
                    print(f"  [SKIP] Frame {frame_idx+1} at {frame_time:.2f}s exceeds video duration")
                    continue

                frame_timestamp = self._seconds_to_time(frame_time)

                # Extract frame
                frame_path = frames_dir / f"{timestamp.replace(':', '-')}_{frame_idx}.jpg"

                if not frame_path.exists():
                    success = self.extract_frame(video_path, frame_time, str(frame_path))
                    if not success:
                        continue

                # Analyze frame
                result = self.analyze_frame(
                    str(frame_path),
                    frame_timestamp,
                    chapter_title,
                    video_context
                )

                chapter_results.append(result)
                print(f"  Frame {frame_idx+1}/{frames_per_chapter}: {frame_timestamp}")

            results[timestamp] = chapter_results

        # Save results
        results_file = output_dir / "visual_analysis.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Analysis complete! Results saved to: {results_file}")

        return results

    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        """Get video duration in seconds using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Warning: Could not get video duration: {e}")
            return float('inf')  # Return infinity to skip duration checks

    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert HH:MM:SS to seconds"""
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(parts[0])

    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        """Convert seconds to HH:MM:SS"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze video frames with VLM")
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--chapters', required=True, help='Path to chapters.json')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--model', default='gpt-4o-mini', choices=['gpt-4o-mini', 'gpt-4o', 'llava'])
    parser.add_argument('--api_key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--frames_per_chapter', type=int, default=5)
    parser.add_argument('--context', help='Video context/description')

    args = parser.parse_args()

    # Load chapters
    with open(args.chapters, 'r', encoding='utf-8') as f:
        chapters = json.load(f)

    # Initialize analyzer
    import os
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    analyzer = VLMAnalyzer(model=args.model, api_key=api_key)

    # Analyze
    results = analyzer.analyze_video_chapters(
        args.video,
        chapters,
        args.output,
        args.frames_per_chapter,
        args.context
    )

    print(f"\n✓ Analyzed {len(results)} chapters")


if __name__ == "__main__":
    main()
