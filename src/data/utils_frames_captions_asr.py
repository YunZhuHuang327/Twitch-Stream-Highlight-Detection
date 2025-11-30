import re

from src.data.utils_asr import ChaptersASR
from src.data.utils_captions import ChaptersCaptions
from src.data.utils_captions_asr import parse_input
from src.data.utils_frames import ChaptersFrames, PromptFrames
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ChaptersFramesCaptionsASR(ChaptersFrames, ChaptersCaptions, ChaptersASR):
    def __init__(
        self,
        embs_dir,
        captions_dir,
        vidc_dir: str = "dataset/",
        subset="",
        vision_feature_select_strategy="cls",
        max_frames=None,
    ):
        ChaptersFrames.__init__(
            self,
            embs_dir=embs_dir,
            vidc_dir=vidc_dir,
            subset=subset,
            vision_feature_select_strategy=vision_feature_select_strategy,
            max_frames=max_frames,
        )
        ChaptersCaptions.__init__(
            self, captions_dir=captions_dir, vidc_dir=vidc_dir, subset=subset
        )
        ChaptersASR.__init__(self, vidc_dir=vidc_dir, subset=subset)

    def __contains__(self, vid_id):
        return (
            ChaptersFrames.__contains__(self, vid_id)
            and ChaptersASR.__contains__(self, vid_id)
            and ChaptersCaptions.__contains__(self, vid_id)
        )


class PromptFramesCaptionsASR(PromptFrames):
    def __init__(
        self,
        chapters: ChaptersFramesCaptionsASR,
        merging_method="interleave",
    ):
        PromptFrames.__init__(self, chapters)

        assert merging_method in [
            "interleave",
            "frames+interleave_captions+asr",
            "frames+captions+asr",
        ], f"Invalid merging method: {merging_method}"
        if merging_method == "interleave":
            self.merging_method = interleave_frames_asr_captions
        elif merging_method == "frames+interleave_captions+asr":
            self.merging_method = frames_interleave_captions_asr
        elif merging_method == "frames+captions+asr":
            self.merging_method = concatenate_frames_captions_asr

    def get_task_prompt(self):
        # Customize the chapter prompt for caption-based prompts
        return "use the ASR transcript and provided captions and frames to identify distinct chapters based on content shifts.\n"

    def get_transcript(self, vid_id):
        frames_transcript = self.chapters.get_frames_transcript(vid_id)
        captions = self.chapters.get_caption(vid_id)
        asr = self.chapters.get_asr(vid_id)
        return self.merging_method(frames_transcript, asr, captions)

    def get_frames_features(self, vid_id):
        return self.chapters.get_frames_features(vid_id)

    def __contains__(self, vid_id):
        # We already checked that the video exists in the frames dir
        return vid_id in self.chapters.frames_ids


def interleave_frames_asr_captions(
    frames_transcript, asr_input, captions_input, add_data_prefix=True
):
    prefix = "Frame " if add_data_prefix else ""
    frames_data = parse_input(frames_transcript, f"{prefix}")
    # Filter out lines containing "image", it needs to be <|reserved_special_token_0|>
    frames_data = [line for line in frames_data if "image" not in line[1]]

    prefix = "Caption " if add_data_prefix else ""
    captions_data = parse_input(captions_input, f"{prefix}")

    if asr_input is None:
        combined = sorted(captions_data + frames_data, key=lambda x: x[0])
    else:
        prefix = "ASR " if add_data_prefix else ""
        asr_data = parse_input(asr_input, f"{prefix}")
        combined = sorted(asr_data + captions_data + frames_data, key=lambda x: x[0])

    return "\n".join(item[1] for item in combined)


def frames_interleave_captions_asr(
    frames_transcript, asr_input, captions_input, add_data_prefix=True
):
    prefix = "Frame " if add_data_prefix else ""
    frames_data = parse_input(frames_transcript, f"{prefix}")
    # Filter out lines containing "image", it needs to be <|reserved_special_token_0|>
    frames_data = "\n".join([line[1] for line in frames_data if "image" not in line[1]])

    prefix = "Caption " if add_data_prefix else ""
    captions_data = parse_input(captions_input, f"{prefix}")

    if asr_input is None:
        combined = sorted(captions_data, key=lambda x: x[0])
    else:
        prefix = "ASR " if add_data_prefix else ""
        asr_data = parse_input(asr_input, f"{prefix}")
        combined = sorted(asr_data + captions_data, key=lambda x: x[0])

    return frames_data + "\n\n" + "\n".join(item[1] for item in combined)


def concatenate_frames_captions_asr(
    frames_transcript, asr_input, captions_input, add_data_prefix=True
):
    prefix = "Frame " if add_data_prefix else ""
    frames_data = parse_input(frames_transcript, f"{prefix}")
    frames_data = "\n".join([line[1] for line in frames_data if "image" not in line[1]])

    prefix = "Caption " if add_data_prefix else ""
    captions_data = parse_input(captions_input, f"{prefix}")
    captions_data = "\n".join([line[1] for line in captions_data])

    if asr_input is None:
        return frames_data + "\n\n" + captions_data

    prefix = "ASR " if add_data_prefix else ""
    asr_data = parse_input(asr_input, f"{prefix}")
    asr_data = "\n".join([line[1] for line in asr_data])

    return frames_data + "\n\n" + captions_data + "\n\n" + asr_data


if __name__ == "__main__":
    captions = "HwwwH_MiniCPM-V-2/asr_s10k-2_train_preds+no-asr-10s"
    embs_dir = f"dataset/embs/google/siglip-so400m-patch14-384/{captions}/"
    captions_dir = f"dataset/captions/{captions}/"

    chapters = ChaptersFramesCaptionsASR(
        embs_dir=embs_dir,
        captions_dir=captions_dir,
        subset="s1k_train",
        vision_feature_select_strategy="cls",
        max_frames=2,
    )
    vid_id = chapters.sample()

    # merging_method = "interleave"
    # merging_method = "frames+interleave_captions+asr"
    merging_method = "frames+captions+asr"

    prompt = PromptFramesCaptionsASR(chapters=chapters, merging_method=merging_method)
    print(prompt.get_prompt_train(vid_id), end="")
    print(prompt.get_transcript(vid_id))
    print(chapters.get_frames_features(vid_id).shape)
