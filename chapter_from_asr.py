"""
直接使用已有的 ASR 文本进行 Chapter 分段

使用方法:
    python chapter_from_asr.py \
        --asr_file "dataset/highlights/123/asr.txt" \
        --video_title "TwitchCon W/ AGENT00" \
        --output_dir "outputs/chapters/123"
"""

import argparse
import json
import sys
import os
from pathlib import Path

# 修正 Windows 控制台編碼問題
if sys.platform == 'win32':
    # 設置標準輸出為 UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    # 設置環境變量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.data.utils_asr import PromptASR
from src.models.llama_inference import LlamaInference
from src.test.vidchapters import get_chapters
from tools.download.models import download_model


def safe_print(text):
    """安全的 print，避免編碼錯誤"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 如果有編碼錯誤，移除無法編碼的字符
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        print(safe_text)


class ASRChapters:
    """模拟 Chapters 接口，使用已有的 ASR 文本"""

    def __init__(self, asr_file: Path, video_title: str):
        self.asr_file = Path(asr_file)
        self.video_title = video_title
        self.video_ids = [self.asr_file.stem]  # 使用文件名作为 ID

        # 读取 ASR 文本
        with open(self.asr_file, 'r', encoding='utf-8') as f:
            self.asr_lines = f.readlines()

        # 转换为 Chapter-Llama 期望的格式
        self.asr_text = self._convert_asr_format()

        # 计算视频长度
        self.duration = self._calculate_duration()

        safe_print(f"[OK] 加载 ASR 文件: {self.asr_file}")
        safe_print(f"  - 行数: {len(self.asr_lines)}")
        safe_print(f"  - 时长: {self._format_duration(self.duration)}")

    def _convert_asr_format(self) -> str:
        """
        将 HH:MM:SS: text 格式转换为 Chapter-Llama 期望的格式

        输入: 00:00:00: Empire State is this one
        输出: [00:00:00] Empire State is this one
        """
        converted_lines = []
        for line in self.asr_lines:
            line = line.strip()
            if not line:
                continue

            # 分割时间戳和文本
            if ':' in line:
                parts = line.split(':', 3)
                if len(parts) >= 4:
                    # HH:MM:SS: text
                    timestamp = f"{parts[0]}:{parts[1]}:{parts[2]}"
                    text = parts[3].strip()
                    converted_lines.append(f"[{timestamp}] {text}")

        return '\n'.join(converted_lines) + '\n'

    def _calculate_duration(self) -> float:
        """从最后一行计算视频时长"""
        if not self.asr_lines:
            return 0.0

        # 从最后一行提取时间
        last_line = self.asr_lines[-1].strip()
        if ':' in last_line:
            parts = last_line.split(':', 3)
            if len(parts) >= 3:
                try:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = int(parts[2].split(':')[0])  # 处理可能的额外冒号
                    return float(hours * 3600 + minutes * 60 + seconds + 10)  # 加10秒缓冲
                except:
                    pass

        return 3600.0  # 默认1小时

    def _format_duration(self, seconds: float) -> str:
        """格式化时长为 HH:MM:SS"""
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
    base_model: str = None
):
    """
    使用已有的 ASR 文本进行章节分段

    Args:
        asr_file: ASR 文本文件路径
        video_title: 视频标题（用于上下文）
        output_dir: 输出目录
        model: Chapter-Llama 模型
        base_model: 基础模型路径
    """
    print("\n" + "="*60)
    print("📖 Chapter-Llama: 从 ASR 文本生成章节")
    print("="*60)
    print(f"ASR 文件: {asr_file}")
    print(f"视频标题: {video_title}")
    print(f"输出目录: {output_dir}")
    print("="*60 + "\n")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载 ASR
    chapters = ASRChapters(asr_file, video_title)

    # 创建 prompt
    prompt_builder = PromptASR(chapters=chapters)
    vid_id = chapters.video_ids[0]
    prompt = prompt_builder.get_prompt_test(vid_id)
    transcript = chapters.get_asr(vid_id)
    full_prompt = prompt + transcript

    print("\n📝 Prompt 统计:")
    print(f"  - Prompt 长度: {len(prompt)} 字符")
    print(f"  - Transcript 长度: {len(transcript)} 字符")
    print(f"  - 总长度: {len(full_prompt)} 字符")
    print(f"  - 预估 tokens: ~{len(full_prompt) // 4}")

    # 检查模型路径
    model_path_obj = Path(model)
    if model_path_obj.exists():
        print(f"\n✓ 使用本地模型: {model}")
        model_path = str(model_path_obj)
    else:
        print(f"\n🔄 下载模型: {model}")
        model_path = download_model(model)
        if model_path is None:
            print(f"❌ 模型下载失败")
            return

    # 使用默认基础模型
    if base_model is None:
        base_model = "meta-llama/Llama-3.1-8B-Instruct"

    print(f"\n🤖 加载模型:")
    print(f"  - 基础模型: {base_model}")
    print(f"  - PEFT 模型: {model_path}")

    # 创建推理引擎
    inference = LlamaInference(
        ckpt_path=base_model,
        peft_model=model_path
    )

    # 生成章节
    print(f"\n🔄 生成章节中...")
    output_text, chapters_dict = get_chapters(
        inference,
        full_prompt,
        max_new_tokens=1024,
        do_sample=False,
        vid_id=vid_id,
    )

    # 显示结果
    print("\n" + "="*60)
    print("📚 生成的章节")
    print("="*60)
    for timestamp, title in chapters_dict.items():
        print(f"  {timestamp}: {title}")
    print("="*60)

    # 保存结果
    chapters_file = output_path / "chapters.json"
    with open(chapters_file, 'w', encoding='utf-8') as f:
        json.dump(chapters_dict, f, indent=2, ensure_ascii=False)

    output_text_file = output_path / "output_text.txt"
    with open(output_text_file, 'w', encoding='utf-8') as f:
        f.write(output_text)

    print(f"\n✅ 结果已保存:")
    print(f"  - 章节: {chapters_file}")
    print(f"  - 完整输出: {output_text_file}")
    print(f"  - 总章节数: {len(chapters_dict)}")

    print("\n" + "="*60)
    print("🎉 完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用已有的 ASR 文本进行章节分段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

python chapter_from_asr.py \
    --asr_file "dataset/highlights/123/asr.txt" \
    --video_title "TwitchCon W/ AGENT00 - EXPLORING LITTLE ITALY" \
    --output_dir "outputs/chapters/123"

使用本地模型:
python chapter_from_asr.py \
    --asr_file "dataset/highlights/123/asr.txt" \
    --video_title "TwitchCon W/ AGENT00" \
    --output_dir "outputs/chapters/123" \
    --base_model "D:/chapter-llama/Llama-3.1-8B-Instruct"
        """
    )

    parser.add_argument('--asr_file', required=True,
                       help="ASR 文本文件路径 (格式: HH:MM:SS: text)")
    parser.add_argument('--video_title', required=True,
                       help="视频标题（用于 LLM 理解上下文）")
    parser.add_argument('--output_dir', required=True,
                       help="输出目录")
    parser.add_argument('--model', default='asr-10k',
                       help="Chapter-Llama 模型名称或路径（默认: asr-10k）")
    parser.add_argument('--base_model', default=None,
                       help="基础模型路径（默认: meta-llama/Llama-3.1-8B-Instruct）")

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.asr_file).exists():
        print(f"❌ 错误: ASR 文件不存在: {args.asr_file}")
        exit(1)

    main(
        asr_file=args.asr_file,
        video_title=args.video_title,
        output_dir=args.output_dir,
        model=args.model,
        base_model=args.base_model
    )
