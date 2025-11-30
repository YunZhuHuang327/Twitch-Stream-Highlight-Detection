import random
import re
from pathlib import Path

from transformers import AutoTokenizer

from src.data.chapters import hms_to_sec, sec_to_hms
from src.data.utils_captions_asr import ChaptersCaptionsASR, PromptCaptionsASR
from src.test.vidchapters_window import get_window
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PromptCaptionsASRWindow(PromptCaptionsASR):
    def __init__(
        self,
        ckpt_path,
        chapters: ChaptersCaptionsASR,
        merging_method="interleave",
        window_token_size=15_000,
    ):
        PromptCaptionsASR.__init__(self, chapters, merging_method)

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

        self.window_token_size = window_token_size

        self.id2window = {}

    def get_prompt_train(self, vid_id: str) -> str:
        if vid_id in self.id2window:
            return self.id2window[vid_id]["prompt"]

        self.get_window_transcript(vid_id=vid_id)
        return self.id2window[vid_id]["prompt"]

    def get_transcript(self, vid_id):
        if vid_id in self.id2window:
            return self.id2window[vid_id]["transcript"]

        self.get_window_transcript(vid_id)
        return self.id2window[vid_id]["transcript"]

    def get_window(self, transcript, prompt, start_time=0):
        prompt, transcript, _ = get_window(
            prompt=prompt,
            transcript=transcript,
            start_time=start_time,
            tokenizer=self.tokenizer,
            window_token_size=self.window_token_size,
        )
        return prompt, transcript

    def get_window_output(self, vid_id: str, start_time=0, end_time=None) -> str:
        vid_chapters = self.chapters.get_chapters(vid_id)
        if end_time is None:
            end_time = self.chapters.get_duration(vid_id)

        answers = []
        for chp_time, chp_title in vid_chapters.items():
            if chp_time >= start_time and chp_time <= end_time:
                chp_time = sec_to_hms(chp_time - start_time)
                answers.append(f"{chp_time} - {chp_title}")

        assert answers, "No chapters after start_time"
        return "\n".join(answers)

    def get_output(self, vid_id: str) -> str:
        if vid_id in self.id2window:
            return self.id2window[vid_id]["output"]

        self.get_window_transcript(vid_id)
        return self.id2window[vid_id]["output"]

    def get_window_transcript(self, vid_id):
        if vid_id in self.id2window:
            return self.id2window[vid_id]

        prompt = self.get_base_prompt(vid_id)
        caption = self.chapters.get_caption(vid_id)
        asr = self.chapters.get_asr(vid_id)
        transcript = self.merging_method(asr, caption)
        n_tokens = len(self.tokenizer.encode(transcript))
        if n_tokens <= self.window_token_size:
            output = self.get_window_output(vid_id)
            self.id2window[vid_id] = {
                "prompt": prompt,
                "transcript": transcript,
                "output": output,
            }
            return self.id2window[vid_id]

        start_times = self.chapters.get_timestamps(
            vid_id, zero_handling="add", duration_handling="remove"
        )
        # Weight earlier chapters more heavily by using exponential weights
        # This is because most of the videos fi
        weights = [0.8**i for i in range(len(start_times))]  # 1, 0.8, 0.64, 0.512, ...
        start_time = random.choices(start_times, weights=weights, k=1)[0]

        prompt, transcript = self.get_window(
            transcript=transcript,
            prompt=prompt,
            start_time=start_time,
        )

        last_line = transcript.split("\n")[-1]
        pattern = r"(\d{2}:[0-5]\d:[0-5]\d)\b"
        end_time = re.search(pattern, last_line).group(1)
        end_time = start_time + hms_to_sec(end_time)
        output = self.get_window_output(vid_id, start_time, end_time)

        self.id2window[vid_id] = {
            "prompt": prompt,
            "transcript": transcript,
            "output": output,
        }


if __name__ == "__main__":
    captions_dir = Path(
        "dataset/captions/HwwwH_MiniCPM-V-2/asr_s10k-2_train_preds+no-asr-10s/"
    )
    ckpt_path = "checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct/"
    chapters = ChaptersCaptionsASR(captions_dir=captions_dir, subset="s1k_train")
    prompt = PromptCaptionsASRWindow(
        ckpt_path=ckpt_path, chapters=chapters, window_token_size=1200
    )

    vid_id = chapters.sample()
    print(vid_id)
    print(prompt.get_prompt_train(vid_id), end="")
    print(prompt.get_transcript(vid_id))
    print("=" * 100)
    print(prompt.get_output(vid_id))
