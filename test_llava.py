"""
Test LLaVA Vision-Language Model

This script demonstrates how to use LLaVA to analyze video frames.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import subprocess
import os

def extract_frame(video_path, timestamp, output_path):
    """Extract a single frame from video at given timestamp

    Args:
        video_path: Path to video file
        timestamp: Time in seconds
        output_path: Where to save the frame
    """
    cmd = [
        'ffmpeg',
        '-ss', str(timestamp),
        '-i', video_path,
        '-vframes', '1',
        '-q:v', '2',
        '-y',  # Overwrite output file
        output_path
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    print(f"Extracted frame at {timestamp}s -> {output_path}")


def analyze_frame_with_llava(image_path, prompt, model_name="llava-hf/llava-1.5-7b-hf"):
    """Analyze an image using LLaVA

    Args:
        image_path: Path to image file
        prompt: Question/instruction for the model
        model_name: HuggingFace model ID

    Returns:
        Model's response as string
    """
    print(f"\nLoading LLaVA model: {model_name}")
    print("This will download ~13GB on first run...")

    # Load model and processor
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Model loaded! Using device: {model.device}")

    # Load image
    image = Image.open(image_path)

    # Prepare prompt in LLaVA format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]

    # Apply chat template
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Process inputs
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)

    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.2
        )

    # Decode response
    response = processor.decode(output[0], skip_special_tokens=True)

    # Extract only the assistant's response (remove the prompt)
    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1].strip()

    return response


def main():
    # Example usage
    video_path = "123.mp4"
    timestamp = 60  # 1 minute into the video
    frame_path = "test_frame.jpg"

    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print("Please provide a valid video path.")
        return

    # Extract frame
    print("=" * 60)
    print("Step 1: Extract frame from video")
    print("=" * 60)
    extract_frame(video_path, timestamp, frame_path)

    # Analyze with LLaVA
    print("\n" + "=" * 60)
    print("Step 2: Analyze frame with LLaVA")
    print("=" * 60)

    prompt = """Analyze this frame from a Twitch livestream.

Describe:
1. What's happening in the scene
2. People's emotions/reactions (if visible)
3. Any text or graphics visible
4. Overall atmosphere

Be concise but specific."""

    response = analyze_frame_with_llava(frame_path, prompt)

    print("\n" + "=" * 60)
    print("LLaVA Response:")
    print("=" * 60)
    print(response)
    print("=" * 60)

    print(f"\nFrame saved to: {frame_path}")
    print("\nYou can now integrate this into your highlight detection pipeline!")


if __name__ == "__main__":
    main()
