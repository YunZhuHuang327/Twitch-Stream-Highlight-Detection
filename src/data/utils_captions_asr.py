import re
from datetime import datetime
from pathlib import Path

from src.data.utils_asr import ChaptersASR
from src.data.utils_captions import ChaptersCaptions, PromptCaptions
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ChaptersCaptionsASR(ChaptersCaptions, ChaptersASR):
    def __init__(self, captions_dir, vidc_dir: str = "dataset/", subset=""):
        ChaptersCaptions.__init__(
            self, captions_dir=captions_dir, vidc_dir=vidc_dir, subset=subset
        )
        ChaptersASR.__init__(self, vidc_dir=vidc_dir, subset=subset)

    def __contains__(self, vid_id):
        # It's enough to check if the video exists in the captions dir
        return ChaptersCaptions.__contains__(self, vid_id)


class PromptCaptionsASR(PromptCaptions):
    def __init__(
        self,
        chapters: ChaptersCaptionsASR,
        merging_method="interleave",
        add_data_prefix=True,
        add_end=False,
    ):
        PromptCaptions.__init__(self, chapters)

        assert merging_method in ["interleave", "captions+asr", "asr+captions"]
        if merging_method == "interleave":
            self.merging_method = interleave_asr_and_captions
        elif merging_method == "captions+asr":
            self.merging_method = concatenate_captions_and_asr
        elif merging_method == "asr+captions":
            self.merging_method = concatenate_asr_and_captions

        self.add_data_prefix = add_data_prefix
        self.add_end = add_end

    def get_task_prompt(self):
        return "use the provided captions and ASR transcript to identify distinct chapters based on content shifts.\n"

    def get_transcript(self, vid_id):
        caption = self.chapters.get_caption(vid_id)
        asr = self.chapters.get_asr(vid_id, add_end=self.add_end)
        return self.merging_method(asr, caption, add_data_prefix=self.add_data_prefix)


def parse_timestamp(timestamp):
    return datetime.strptime(timestamp, "%H:%M:%S")


def parse_input(input_str, prefix, suffix=""):
    lines = input_str.strip().split("\n")
    parsed = []
    for line in lines:
        match = re.match(r"(\d{2}:\d{2}:\d{2}):\s*(.*)", line)
        if match:
            timestamp, content = match.groups()
            parsed.append(
                (parse_timestamp(timestamp), f"{prefix}{timestamp}: {content}{suffix}")
            )
    return parsed


def interleave_asr_and_captions(asr_input, captions_input, add_data_prefix=True):
    prefix = "Caption " if add_data_prefix else ""
    captions_data = parse_input(captions_input, f"{prefix}")

    if asr_input is None:
        combined = captions_data
    else:
        prefix = "ASR " if add_data_prefix else ""
        asr_data = parse_input(asr_input, f"{prefix}")
        combined = sorted(asr_data + captions_data, key=lambda x: x[0])

    return "\n".join(item[1] for item in combined)


def concatenate_asr_and_captions(asr_input, captions_input, **kwargs):
    if asr_input is None:
        return captions_input
    prompt_a = "ASR transcript:\n" + asr_input + "\n\n"
    prompt_c = "Captions:\n" + captions_input
    return prompt_a + prompt_c


def concatenate_captions_and_asr(captions_input, asr_input, **kwargs):
    if captions_input is None:
        return asr_input
    prompt_c = "Captions:\n" + captions_input + "\n\n"
    prompt_a = "ASR transcript:\n" + asr_input
    return prompt_c + prompt_a


if __name__ == "__main__":
    captions_dir = Path(
        "dataset/captions/HwwwH_MiniCPM-V-2/asr_s10k-2_train_preds+no-asr-10s/"
    )
    chapters = ChaptersCaptionsASR(captions_dir=captions_dir, subset="s1k_train")
    # merging_method = "asr+captions"
    merging_method = "captions+asr"
    # merging_method = "interleave"
    prompt = PromptCaptionsASR(chapters=chapters, merging_method=merging_method)

    vid_id = chapters.sample()
    print(prompt.get_prompt_train(vid_id))
    print(prompt.get_transcript(vid_id))
    print(prompt.get_output(vid_id))
