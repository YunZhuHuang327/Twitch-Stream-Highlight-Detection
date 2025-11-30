"""
Highlight Detection Inference Script

使用训练好的模型在单个视频上检测 highlights
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from src.data.single_video import get_asr
from src.test.highlights_window import (
    get_highlights,
    save_highlights,
    parse_highlights
)


def load_chat_data(chat_file: str) -> dict:
    """加载聊天数据"""
    if not Path(chat_file).exists():
        print(f"⚠️ 聊天文件不存在: {chat_file}，将不使用聊天数据")
        return {"messages": [], "intensity_timeline": [], "peak_moments": []}
    
    with open(chat_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_inference_function(model, tokenizer, device='cuda'):
    """创建推理函数"""
    def inference(prompt: str, max_new_tokens: int = 512) -> str:
        """
        运行模型推理
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after prompt)
        prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        response = generated_text[prompt_length:].strip()
        
        return response
    
    return inference


def main():
    parser = argparse.ArgumentParser(description="Detect highlights in a video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--model", default="D:/chapter-llama/Llama-3.2-1B-Instruct", help="Path to trained model or HF model ID")
    parser.add_argument("--base_model", default=None, help="Base model if using adapter")
    parser.add_argument("--chat_file", default=None, help="Path to chat data JSON file (optional)")
    parser.add_argument("--output", default=None, help="Output JSON file (default: outputs/inference/VIDEO_ID/highlights.json)")
    parser.add_argument("--window_size", type=int, default=35000, help="Window token size")
    parser.add_argument("--overlap", type=int, default=300, help="Overlap in seconds")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    video_id = video_path.stem
    
    # 设置输出路径
    if args.output:
        output_file = args.output
    else:
        output_file = f"outputs/inference/{video_id}/highlights.json"
    
    print(f"\n{'='*60}")
    print(f"🎬 Highlight Detection")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Model: {args.model}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    # 1. 提取 ASR
    print("🎤 提取 ASR...")
    asr_text, asr_data, duration = get_asr(video_path, return_timestamps=True)
    print(f"   ✓ ASR 完成，时长: {duration:.1f}秒")
    
    # 2. 加载聊天数据（可选）
    chat_data = {}
    if args.chat_file:
        print(f"💬 加载聊天数据...")
        chat_data = load_chat_data(args.chat_file)
        print(f"   ✓ 共 {len(chat_data.get('messages', []))} 条消息")
    else:
        print("ℹ️  未提供聊天数据")
    
    # 3. 加载模型
    print(f"🤖 加载模型...")
    
    model_path = Path(args.model)
    if not model_path.exists():
        # Try as HuggingFace model ID
        model_path = args.model
    else:
        model_path = str(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model if args.base_model else model_path
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if args.device == 'cuda' else torch.float32,
        device_map=args.device if args.device == 'cuda' else None
    )
    
    if args.device == 'cpu':
        model = model.to('cpu')
    
    print(f"   ✓ 模型加载完成")
    
    # 4. 创建推理函数
    inference_fn = create_inference_function(model, tokenizer, args.device)
    
    # 5. 基础 prompt
    base_prompt = """You are an expert at detecting highlight moments in live streaming videos.
Given the video transcript and chat messages, identify exciting/funny/important moments.
Output format: [START_TIME-END_TIME] TYPE: DESCRIPTION

Highlight types:
- exciting_moment: Exciting gameplay, achievements, victories
- funny_moment: Humorous situations, jokes, accidents
- emotional_moment: Touching or dramatic moments
- skill_showcase: Impressive plays or techniques
- chat_peak: Moments with extremely high chat activity"""
    
    # 6. 检测 highlights
    print(f"\n🔍 开始检测 highlights...")
    
    # 转换 ASR 数据为带时间戳的文本行
    transcript_lines = []
    for entry in asr_data:
        line = f"[{entry['timestamp_str']}] {entry['text']}"
        transcript_lines.append(line)
    
    highlights = get_highlights(
        inference=inference_fn,
        prompt=base_prompt,
        transcript=transcript_lines,
        chat_data=chat_data,
        tokenizer=tokenizer,
        window_token_size=args.window_size,
        overlap_seconds=args.overlap,
        max_windows=100
    )
    
    # 7. 保存结果
    save_highlights(highlights, output_file)
    
    # 8. 打印摘要
    print(f"\n{'='*60}")
    print(f"✅ 检测完成！")
    print(f"{'='*60}")
    print(f"检测到 {len(highlights)} 个 highlights:")
    print()
    
    type_counts = {}
    for hl in highlights:
        type_counts[hl['type']] = type_counts.get(hl['type'], 0) + 1
    
    for hl_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {hl_type}: {count}")
    
    print(f"\n详细结果:")
    for i, hl in enumerate(highlights, 1):
        print(f"  {i}. [{hl['start_time_str']}-{hl['end_time_str']}] ({hl['duration']:.0f}s)")
        print(f"     {hl['type']}: {hl['description']}")
    
    print(f"\n结果已保存到: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
