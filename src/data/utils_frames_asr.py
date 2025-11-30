from src.data.utils_asr import ChaptersASR
from src.data.utils_captions_asr import parse_input
from src.data.utils_frames import ChaptersFrames, PromptFrames
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ChaptersFramesASR(ChaptersFrames, ChaptersASR):
    def __init__(
        self,
        embs_dir,
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
        ChaptersASR.__init__(self, vidc_dir=vidc_dir, subset=subset)

    def __contains__(self, vid_id):
        return ChaptersFrames.__contains__(self, vid_id) and ChaptersASR.__contains__(
            self, vid_id
        )


class PromptFramesASR(PromptFrames):
    def __init__(
        self,
        chapters: ChaptersFramesASR,
        merging_method="interleave",
        add_data_prefix=True,
        add_end=False,
    ):
        PromptFrames.__init__(self, chapters)

        assert merging_method in [
            "interleave",
            "frames+asr",
            "asr+frames",
        ], f"Invalid merging method: {merging_method}"
        if merging_method == "interleave":
            self.merging_method = interleave_frames_and_asr
        elif merging_method == "frames+asr":
            self.merging_method = concatenate_frames_and_asr
        elif merging_method == "asr+frames":
            self.merging_method = concatenate_asr_and_frames

        self.add_data_prefix = add_data_prefix
        self.add_end = add_end

    def get_task_prompt(self):
        # Customize the chapter prompt for caption-based prompts
        return "use the provided frames and ASR transcript to identify distinct chapters based on content shifts.\n"

    def get_transcript(self, vid_id):
        frames = self.chapters.get_frames_transcript(vid_id)
        asr = self.chapters.get_asr(vid_id, add_end=self.add_end)
        return self.merging_method(frames, asr, add_data_prefix=self.add_data_prefix)

    def get_frames_features(self, vid_id):
        return self.chapters.get_frames_features(vid_id)

    def __contains__(self, vid_id):
        # We already checked that the video exists in the frames dir
        return vid_id in self.chapters.frames_ids


def interleave_frames_and_asr(frames_input, asr_input, add_data_prefix=True):
    prefix = "Frame " if add_data_prefix else ""
    frames_data = parse_input(frames_input, f"{prefix}")

    if asr_input is None:
        combined = frames_data
    else:
        prefix = "ASR " if add_data_prefix else ""
        asr_data = parse_input(asr_input, f"{prefix}")
        combined = sorted(asr_data + frames_data, key=lambda x: x[0])

    return "\n".join(item[1] for item in combined)


def concatenate_frames_and_asr(frames_input, asr_input, add_data_prefix=True):
    prefix = "Frame " if add_data_prefix else ""
    frames_data = parse_input(frames_input, f"{prefix}")
    frames_data = "\n".join([line[1] for line in frames_data if "image" not in line[1]])

    if asr_input is None:
        return frames_data
    prefix = "ASR " if add_data_prefix else ""
    asr_data = parse_input(asr_input, f"{prefix}")
    asr_data = "\n".join([line[1] for line in asr_data])

    return frames_data + "\n\n" + asr_data


def concatenate_asr_and_frames(frames_input, asr_input, add_data_prefix=True):
    if asr_input is None:
        asr_data = ""
    else:
        prefix = "ASR " if add_data_prefix else ""
        asr_data = parse_input(asr_input, f"{prefix}")
        asr_data = "\n".join([line[1] for line in asr_data])
        asr_data = asr_data + "\n\n"

    prefix = "Frame " if add_data_prefix else ""
    frames_data = parse_input(frames_input, f"{prefix}")
    frames_data = "\n".join([line[1] for line in frames_data if "image" not in line[1]])

    return asr_data + frames_data


if __name__ == "__main__":
    embs_dir = "dataset/embs/google/siglip-so400m-patch14-384/HwwwH_MiniCPM-V-2/asr_s10k-2_train_preds+no-asr-10s"
    chapters = ChaptersFramesASR(
        embs_dir=embs_dir,
        subset="s1k_train",
        vision_feature_select_strategy="cls",
        max_frames=2,
    )
    vid_id = chapters.sample()

    merging_method = "frames+asr"
    # merging_method = "asr+frames"
    # merging_method = "interleave"
    prompt = PromptFramesASR(chapters=chapters, merging_method=merging_method)
    print(prompt.get_prompt_train(vid_id))
    print(prompt.get_transcript(vid_id))
    print(chapters.get_frames_features(vid_id).shape)
