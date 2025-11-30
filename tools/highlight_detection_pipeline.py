"""
完整的 Highlight 检测管道

流程:
1. Chapter-Llama 将长视频切成语义片段
2. VLM 为每个片段生成视觉描述
3. 将 VLM 描述转换为事件标签
4. 合并 ASR + Chat + Visual 事件到 transcript
5. 使用 Llama 3.2-1B 为每个时间窗口打分
6. 根据分数筛选 highlight timestamps

使用方法:
    python tools/highlight_detection_pipeline.py \
        --video_path "video.mp4" \
        --video_title "TwitchCon with Agent00" \
        --chat_file "chat.json" \
        --output_dir "outputs/highlights" \
        --top_k 20
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')


class HighlightDetectionPipeline:
    """完整的 Highlight 检测管道"""

    def __init__(
        self,
        video_path: str,
        video_title: str,
        chat_file: str,
        output_dir: str,
        chapter_model: str = "asr-10k",
        vlm_model: str = "llava",
        scoring_model: str = "Llama-3.2-1B-Instruct",
    ):
        self.video_path = Path(video_path)
        self.video_title = video_title
        self.chat_file = Path(chat_file)
        self.output_dir = Path(output_dir)
        self.chapter_model = chapter_model
        self.vlm_model = vlm_model
        self.scoring_model = scoring_model

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("🎬 Highlight Detection Pipeline")
        print("="*60)
        print(f"Video: {self.video_path}")
        print(f"Title: {self.video_title}")
        print(f"Chat: {self.chat_file}")
        print(f"Output: {self.output_dir}")
        print("="*60 + "\n")

    def step1_chapter_segmentation(self) -> Dict:
        """
        步骤 1: 使用 Chapter-Llama 切分视频

        Returns:
            chapters: {timestamp: title}
        """
        print("\n📍 Step 1: Chapter Segmentation")
        print("-" * 60)

        # 检查是否已经有 chapters
        chapters_file = self.output_dir / "chapters.json"
        if chapters_file.exists():
            print(f"✓ 加载已有的 chapters: {chapters_file}")
            with open(chapters_file, 'r', encoding='utf-8') as f:
                chapters = json.load(f)
            print(f"✓ 找到 {len(chapters)} 个章节")
            return chapters

        print(f"🔄 运行 Chapter-Llama...")

        # 调用 Chapter-Llama
        try:
            from inference import main as chapter_main
            chapter_main(
                video_path=self.video_path,
                model=self.chapter_model
            )

            # 读取生成的 chapters
            inference_output = Path(f"outputs/inference/{self.video_path.stem}")
            chapters_src = inference_output / "chapters.json"

            if not chapters_src.exists():
                print(f"❌ 错误: 找不到 chapters 文件: {chapters_src}")
                return {}

            with open(chapters_src, 'r', encoding='utf-8') as f:
                chapters = json.load(f)

            # 保存到输出目录
            with open(chapters_file, 'w', encoding='utf-8') as f:
                json.dump(chapters, f, indent=2, ensure_ascii=False)

            print(f"✓ 生成 {len(chapters)} 个章节")
            return chapters

        except Exception as e:
            print(f"❌ Chapter-Llama 出错: {e}")
            return {}

    def step2_generate_visual_descriptions(self, chapters: Dict) -> Dict:
        """
        步骤 2: 使用 VLM 为每个片段生成视觉描述

        Args:
            chapters: {timestamp: title}

        Returns:
            visual_descriptions: {timestamp: description}
        """
        print("\n🎨 Step 2: Generate Visual Descriptions")
        print("-" * 60)

        visual_file = self.output_dir / "visual_descriptions.json"
        if visual_file.exists():
            print(f"✓ 加载已有的视觉描述: {visual_file}")
            with open(visual_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        print(f"🔄 使用 VLM 生成描述...")
        print(f"   模型: {self.vlm_model}")
        print(f"   章节数: {len(chapters)}")

        # TODO: 实现 VLM 调用
        # 这里需要根据你选择的 VLM (LLaVA, BLIP, etc.) 来实现
        # 目前先返回占位符

        visual_descriptions = {}
        chapter_times = sorted([self._time_to_seconds(t) for t in chapters.keys()])

        for i, timestamp_sec in enumerate(chapter_times):
            timestamp = self._seconds_to_time(timestamp_sec)

            # 占位符 - 实际应该调用 VLM
            visual_descriptions[timestamp] = {
                'description': f"[占位符] 视觉描述 for {chapters[timestamp]}",
                'objects': [],
                'actions': [],
                'emotions': []
            }

            if (i + 1) % 10 == 0:
                print(f"   进度: {i+1}/{len(chapter_times)}")

        # 保存
        with open(visual_file, 'w', encoding='utf-8') as f:
            json.dump(visual_descriptions, f, indent=2, ensure_ascii=False)

        print(f"✓ 生成 {len(visual_descriptions)} 个视觉描述")
        print(f"⚠️  注意: 当前使用占位符，需要实现实际的 VLM 调用")

        return visual_descriptions

    def step3_extract_visual_events(self, visual_descriptions: Dict) -> Dict:
        """
        步骤 3: 从 VLM 描述中提取事件标签

        类似于 chat features 的事件分类，但针对视觉内容

        Args:
            visual_descriptions: {timestamp: {description, objects, actions, emotions}}

        Returns:
            visual_events: {timestamp: [event_labels]}
        """
        print("\n🏷️  Step 3: Extract Visual Event Labels")
        print("-" * 60)

        events_file = self.output_dir / "visual_events.json"
        if events_file.exists():
            print(f"✓ 加载已有的视觉事件: {events_file}")
            with open(events_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        print(f"🔄 分析视觉事件...")

        visual_events = {}

        for timestamp, desc in visual_descriptions.items():
            events = []

            # 基于关键词的简单分类（实际可以更复杂）
            description = desc.get('description', '').lower()
            actions = desc.get('actions', [])
            emotions = desc.get('emotions', [])

            # 动作类事件
            if any(word in description for word in ['laugh', 'laughing', 'smile']):
                events.append('VISUAL_LAUGH')

            if any(word in description for word in ['jump', 'run', 'dance', 'move']):
                events.append('VISUAL_ACTION')

            if any(word in description for word in ['eating', 'eat', 'food', 'drink']):
                events.append('VISUAL_EATING')

            if any(word in description for word in ['talk', 'speaking', 'conversation']):
                events.append('VISUAL_TALKING')

            # 情绪类事件
            if any(word in description for word in ['excited', 'happy', 'celebration']):
                events.append('VISUAL_EXCITEMENT')

            if any(word in description for word in ['shocked', 'surprised', 'wow']):
                events.append('VISUAL_SURPRISE')

            # 场景类事件
            if any(word in description for word in ['crowd', 'people', 'audience']):
                events.append('VISUAL_SOCIAL')

            # 如果没有特别的事件
            if not events:
                events.append('VISUAL_NORMAL')

            visual_events[timestamp] = events

        # 保存
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(visual_events, f, indent=2, ensure_ascii=False)

        # 统计
        from collections import Counter
        event_counts = Counter()
        for events in visual_events.values():
            event_counts.update(events)

        print(f"✓ 提取 {len(visual_events)} 个时间点的视觉事件")
        print("\n📊 视觉事件分布:")
        for event, count in event_counts.most_common():
            print(f"   {event}: {count}")

        return visual_events

    def step4_merge_transcripts(
        self,
        visual_events: Dict,
        asr_file: str = None,
        readable_transcript: str = None
    ) -> str:
        """
        步骤 4: 合并 ASR + Chat Events + Visual Events

        Args:
            visual_events: {timestamp: [event_labels]}
            asr_file: ASR 文件路径（可选）
            readable_transcript: 已有的 readable_transcript.txt 路径（可选）

        Returns:
            merged_transcript_path: 合并后的 transcript 文件路径
        """
        print("\n🔀 Step 4: Merge All Transcripts")
        print("-" * 60)

        merged_file = self.output_dir / "merged_transcript.txt"

        # 如果指定了 readable_transcript，从它开始
        if readable_transcript and Path(readable_transcript).exists():
            print(f"✓ 加载 readable transcript: {readable_transcript}")
            with open(readable_transcript, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        elif asr_file and Path(asr_file).exists():
            print(f"✓ 加载 ASR 文件: {asr_file}")
            # 简单格式化 ASR
            lines = []
            with open(asr_file, 'r', encoding='utf-8') as f:
                for line in f:
                    lines.append(line)
        else:
            print("❌ 错误: 需要提供 readable_transcript 或 asr_file")
            return None

        # 插入视觉事件标签
        print(f"🔄 插入 {len(visual_events)} 个视觉事件...")

        merged_lines = []
        visual_timestamps = set(visual_events.keys())

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 提取时间戳
            if line.startswith('00:'):
                timestamp = line.split()[0]

                # 检查是否有对应的视觉事件
                if timestamp in visual_timestamps:
                    # 先插入视觉事件标签
                    for event in visual_events[timestamp]:
                        merged_lines.append(f"{timestamp} [{event}]")

            # 保留原始行
            merged_lines.append(line)

        # 保存
        with open(merged_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(merged_lines))

        print(f"✓ 合并后的 transcript: {merged_file}")
        print(f"   总行数: {len(merged_lines)}")

        return str(merged_file)

    def step5_score_highlights(
        self,
        merged_transcript: str,
        use_api: str = None,
        api_key: str = None
    ) -> Dict:
        """
        步骤 5: 使用 LLM 为每个时间段打分

        Args:
            merged_transcript: 合并后的 transcript 文件路径
            use_api: API 类型 ('openai', 'claude', 'gemini', None=规则系统)
            api_key: API 密钥

        Returns:
            scores: {timestamp: score}
        """
        print("\n🎯 Step 5: Score Highlight Intensity")
        print("-" * 60)

        scores_file = self.output_dir / "highlight_scores.json"
        if scores_file.exists():
            print(f"✓ 加载已有的分数: {scores_file}")
            with open(scores_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        if use_api:
            print(f"🔄 使用 {use_api} API 打分...")
            print(f"   视频标题: {self.video_title}")
            scores = self._api_based_scoring(
                merged_transcript,
                use_api,
                api_key
            )
        else:
            print(f"🔄 使用规则系统打分...")
            print(f"   模型: {self.scoring_model}")
            print(f"   视频标题: {self.video_title}")
            scores = self._rule_based_scoring(merged_transcript)
            print(f"⚠️  注意: 使用规则系统，建议用 --use_api 获得更好效果")

        # 保存
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        print(f"✓ 生成 {len(scores)} 个时间段的分数")
        print(f"   平均分: {sum(scores.values()) / len(scores):.1f}")
        print(f"   最高分: {max(scores.values()):.1f}")

        return scores

    def step6_select_highlights(
        self,
        scores: Dict,
        top_k: int = 20,
        clip_duration: int = 60
    ) -> List[Tuple]:
        """
        步骤 6: 根据分数筛选 highlight timestamps

        Args:
            scores: {timestamp: score}
            top_k: 选择前 K 个 highlights
            clip_duration: 每个 highlight 的持续时长（秒），默认 60 秒

        Returns:
            highlights: [(timestamp, score, end_timestamp), ...]
        """
        print("\n🌟 Step 6: Select Top Highlights")
        print("-" * 60)

        # 按分数排序
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 选择 top K
        highlights = []
        for i, (timestamp, score) in enumerate(sorted_scores[:top_k]):
            # 计算开始和结束时间
            start_sec = self._time_to_seconds(timestamp)
            end_sec = start_sec + clip_duration
            end_timestamp = self._seconds_to_time(end_sec)

            highlights.append((timestamp, score, end_timestamp))

        # 保存
        highlights_file = self.output_dir / "final_highlights.json"
        highlights_data = {
            'video_title': self.video_title,
            'total_scored': len(scores),
            'selected': top_k,
            'clip_duration_seconds': clip_duration,
            'highlights': [
                {
                    'rank': i + 1,
                    'start': ts_start,
                    'end': ts_end,
                    'duration': clip_duration,
                    'score': score
                }
                for i, (ts_start, score, ts_end) in enumerate(highlights)
            ]
        }

        with open(highlights_file, 'w', encoding='utf-8') as f:
            json.dump(highlights_data, f, indent=2, ensure_ascii=False)

        print(f"✓ 选出 Top-{top_k} Highlights")
        print(f"   每个片段时长: {clip_duration} 秒")
        print(f"\n📋 Top 10 Highlights:")
        for i, (start, score, end) in enumerate(highlights[:10]):
            print(f"   #{i+1:2d}. {start} - {end} (Score: {score:.1f})")

        print(f"\n✅ 结果已保存到: {highlights_file}")

        return highlights

    def _api_based_scoring(
        self,
        transcript_file: str,
        api_type: str,
        api_key: str
    ) -> Dict:
        """使用 API 进行 LLM 打分"""
        with open(transcript_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 按时间窗口分组 (5秒窗口)
        window_size = 5
        windows = {}
        current_window = None
        current_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 提取时间戳
            if line.startswith('00:'):
                parts = line.split()
                timestamp = parts[0]
                timestamp_sec = self._time_to_seconds(timestamp)
                window_start = (timestamp_sec // window_size) * window_size
                window_key = self._seconds_to_time(window_start)

                if current_window != window_key:
                    # 保存上一个窗口
                    if current_window and current_lines:
                        windows[current_window] = '\n'.join(current_lines)

                    current_window = window_key
                    current_lines = []

            current_lines.append(line)

        # 保存最后一个窗口
        if current_window and current_lines:
            windows[current_window] = '\n'.join(current_lines)

        print(f"   分组为 {len(windows)} 个窗口")

        # 调用 API 打分
        scores = {}

        if api_type == 'openai':
            scores = self._score_with_openai(windows, api_key)
        elif api_type == 'claude':
            scores = self._score_with_claude(windows, api_key)
        elif api_type == 'gemini':
            scores = self._score_with_gemini(windows, api_key)
        else:
            print(f"❌ 不支持的 API 类型: {api_type}")
            return {}

        return scores

    def _score_with_openai(self, windows: Dict, api_key: str) -> Dict:
        """使用 OpenAI API 打分"""
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ 错误: 请先安装 openai: pip install openai")
            return {}

        client = OpenAI(api_key=api_key)
        scores = {}
        total = len(windows)

        for i, (timestamp, segment) in enumerate(windows.items()):
            try:
                prompt = self._create_scoring_prompt(segment)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100
                )

                # 解析响应
                result = response.choices[0].message.content
                import json as json_module
                score_data = json_module.loads(result)
                scores[timestamp] = float(score_data.get('score', 50))

                if (i + 1) % 10 == 0:
                    print(f"   进度: {i+1}/{total}")

            except Exception as e:
                print(f"⚠️  警告: 时间 {timestamp} 打分失败: {e}")
                scores[timestamp] = 50.0

        return scores

    def _score_with_claude(self, windows: Dict, api_key: str) -> Dict:
        """使用 Claude API 打分"""
        try:
            import anthropic
        except ImportError:
            print("❌ 错误: 请先安装 anthropic: pip install anthropic")
            return {}

        client = anthropic.Anthropic(api_key=api_key)
        scores = {}
        total = len(windows)

        for i, (timestamp, segment) in enumerate(windows.items()):
            try:
                prompt = self._create_scoring_prompt(segment)

                message = client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=100,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )

                result = message.content[0].text
                import json as json_module
                score_data = json_module.loads(result)
                scores[timestamp] = float(score_data.get('score', 50))

                if (i + 1) % 10 == 0:
                    print(f"   进度: {i+1}/{total}")

            except Exception as e:
                print(f"⚠️  警告: 时间 {timestamp} 打分失败: {e}")
                scores[timestamp] = 50.0

        return scores

    def _score_with_gemini(self, windows: Dict, api_key: str) -> Dict:
        """使用 Gemini API 打分"""
        try:
            import google.generativeai as genai
        except ImportError:
            print("❌ 错误: 请先安装 google-generativeai: pip install google-generativeai")
            return {}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        scores = {}
        total = len(windows)

        for i, (timestamp, segment) in enumerate(windows.items()):
            try:
                prompt = self._create_scoring_prompt(segment)
                response = model.generate_content(prompt)
                result = response.text

                import json as json_module
                score_data = json_module.loads(result)
                scores[timestamp] = float(score_data.get('score', 50))

                if (i + 1) % 10 == 0:
                    print(f"   进度: {i+1}/{total}")

            except Exception as e:
                print(f"⚠️  警告: 时间 {timestamp} 打分失败: {e}")
                scores[timestamp] = 50.0

        return scores

    def _create_scoring_prompt(self, segment: str) -> str:
        """创建打分 prompt"""
        return f"""你是 Twitch 直播 Highlight 检测专家。

视频标题: "{self.video_title}"

任务: 分析以下 transcript 片段，评估 Highlight 强度 (0-100分)。

考虑因素:
1. ASR 内容的趣味性和情绪强度
2. 聊天室反应标签的数量和类型:
   - CHAT_SPIKE_HIGH: 高能量爆发 (+20分)
   - CHAT_SPIKE_LAUGH: 爆笑时刻 (+15分)
   - CHAT_SPIKE_CLIP_MOMENT: 值得剪辑 (+25分)
   - CHAT_SPIKE_LOVE: 温馨时刻 (+10分)
   - CHAT_SPIKE_EXCITEMENT: 兴奋 (+15分)
3. 视觉事件标签 (如果有):
   - VISUAL_LAUGH: 视觉笑声 (+15分)
   - VISUAL_ACTION: 动作场景 (+10分)
   - VISUAL_EXCITEMENT: 视觉兴奋 (+20分)
4. 多个事件标签叠加 → 更高分数

评分标准:
- 90-100: 极度精彩，必须收录
- 75-89: 非常精彩，强烈推荐
- 60-74: 有趣，值得考虑
- 40-59: 普通
- 0-39: 无聊

Transcript 片段:
{segment}

只输出 JSON，不要其他文字:
{{"score": 85, "reason": "简短原因"}}"""

    def _rule_based_scoring(self, transcript_file: str) -> Dict:
        """规则系统打分（占位符）"""
        with open(transcript_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        scores = {}
        current_timestamp = None
        current_score = 50.0  # 基础分

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 提取时间戳
            if line.startswith('00:'):
                parts = line.split()
                timestamp = parts[0]

                # 如果是新的时间戳，保存之前的分数
                if current_timestamp and current_timestamp != timestamp:
                    scores[current_timestamp] = current_score
                    current_score = 50.0  # 重置

                current_timestamp = timestamp

                # 根据事件标签调整分数
                if '[CHAT_SPIKE_HIGH]' in line:
                    current_score += 20
                if '[CHAT_SPIKE_LAUGH]' in line:
                    current_score += 15
                if '[CHAT_SPIKE_CLIP_MOMENT]' in line:
                    current_score += 25
                if '[VISUAL_LAUGH]' in line:
                    current_score += 15
                if '[VISUAL_ACTION]' in line:
                    current_score += 10
                if '[VISUAL_EXCITEMENT]' in line:
                    current_score += 20

                # 限制在 0-100
                current_score = min(100, max(0, current_score))

        # 保存最后一个
        if current_timestamp:
            scores[current_timestamp] = current_score

        return scores

    def _time_to_seconds(self, time_str: str) -> int:
        """HH:MM:SS 转秒"""
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    def _seconds_to_time(self, seconds: int) -> str:
        """秒转 HH:MM:SS"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def run_full_pipeline(
        self,
        top_k: int = 20,
        clip_duration: int = 60,
        skip_chapters: bool = False,
        skip_vlm: bool = False,
        use_api: str = None,
        api_key: str = None
    ):
        """运行完整的管道"""
        print("\n" + "🚀" * 30)
        print("开始完整的 Highlight 检测管道")
        print("🚀" * 30 + "\n")

        # Step 1: Chapter Segmentation
        if not skip_chapters:
            chapters = self.step1_chapter_segmentation()
            if not chapters:
                print("❌ 无法继续: Chapter segmentation 失败")
                return
        else:
            # 加载已有的 chapters
            chapters_file = self.output_dir / "chapters.json"
            if not chapters_file.exists():
                print("❌ 找不到 chapters 文件，无法跳过")
                return
            with open(chapters_file, 'r', encoding='utf-8') as f:
                chapters = json.load(f)

        # Step 2: Generate Visual Descriptions
        if not skip_vlm:
            visual_descriptions = self.step2_generate_visual_descriptions(chapters)
        else:
            visual_file = self.output_dir / "visual_descriptions.json"
            if not visual_file.exists():
                print("⚠️  跳过 VLM，使用空的视觉描述")
                visual_descriptions = {}
            else:
                with open(visual_file, 'r', encoding='utf-8') as f:
                    visual_descriptions = json.load(f)

        # Step 3: Extract Visual Events
        visual_events = self.step3_extract_visual_events(visual_descriptions)

        # Step 4: Merge Transcripts
        # 尝试找 readable_transcript
        readable_transcript = f"dataset/highlights/{self.video_path.stem}/readable_transcript.txt"
        if not Path(readable_transcript).exists():
            readable_transcript = None

        merged_transcript = self.step4_merge_transcripts(
            visual_events,
            readable_transcript=readable_transcript
        )

        if not merged_transcript:
            print("❌ 无法继续: Transcript merge 失败")
            return

        # Step 5: Score Highlights
        scores = self.step5_score_highlights(
            merged_transcript,
            use_api=use_api,
            api_key=api_key
        )

        # Step 6: Select Top Highlights
        highlights = self.step6_select_highlights(
            scores,
            top_k=top_k,
            clip_duration=clip_duration
        )

        print("\n" + "🎉" * 30)
        print("管道完成！")
        print("🎉" * 30 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="完整的 Highlight 检测管道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

python tools/highlight_detection_pipeline.py \
    --video_path "123.mp4" \
    --video_title "TwitchCon with Agent00" \
    --chat_file "123.json" \
    --output_dir "outputs/highlights/123" \
    --top_k 20
        """
    )

    parser.add_argument('--video_path', required=True,
                       help="视频文件路径")
    parser.add_argument('--video_title', required=True,
                       help="视频标题（用于 LLM 打分上下文）")
    parser.add_argument('--chat_file', required=True,
                       help="聊天记录 JSON 文件")
    parser.add_argument('--output_dir', required=True,
                       help="输出目录")
    parser.add_argument('--top_k', type=int, default=20,
                       help="选择前 K 个 highlights（默认 20）")
    parser.add_argument('--clip_duration', type=int, default=60,
                       help="每个 highlight 片段的时长（秒），默认 60")
    parser.add_argument('--skip_chapters', action='store_true',
                       help="跳过 chapter segmentation（使用已有的）")
    parser.add_argument('--skip_vlm', action='store_true',
                       help="跳过 VLM 描述生成")
    parser.add_argument('--use_api', type=str, choices=['openai', 'claude', 'gemini'],
                       help="使用 API 进行 LLM 打分 (openai/claude/gemini)")
    parser.add_argument('--api_key', type=str,
                       help="API 密钥 (或设置环境变量 OPENAI_API_KEY/ANTHROPIC_API_KEY/GOOGLE_API_KEY)")

    args = parser.parse_args()

    # 从环境变量获取 API key
    if args.use_api and not args.api_key:
        import os
        if args.use_api == 'openai':
            args.api_key = os.getenv('OPENAI_API_KEY')
        elif args.use_api == 'claude':
            args.api_key = os.getenv('ANTHROPIC_API_KEY')
        elif args.use_api == 'gemini':
            args.api_key = os.getenv('GOOGLE_API_KEY')

        if not args.api_key:
            print(f"❌ 错误: 请提供 API key 或设置环境变量")
            return

    # 创建管道
    pipeline = HighlightDetectionPipeline(
        video_path=args.video_path,
        video_title=args.video_title,
        chat_file=args.chat_file,
        output_dir=args.output_dir
    )

    # 运行管道
    pipeline.run_full_pipeline(
        top_k=args.top_k,
        clip_duration=args.clip_duration,
        skip_chapters=args.skip_chapters,
        skip_vlm=args.skip_vlm,
        use_api=args.use_api,
        api_key=args.api_key
    )


if __name__ == '__main__':
    main()
