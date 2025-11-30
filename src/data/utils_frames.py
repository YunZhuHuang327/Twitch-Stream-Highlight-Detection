from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from src.data.chapters import Chapters, sec_to_hms
from src.data.prompt import Prompt
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LRUCache(OrderedDict):
    def __init__(self, maxsize=32):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key):
        if key not in self:
            return None
        self.move_to_end(key)
        return self[key]

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)


class ChaptersFrames(Chapters):
    def __init__(
        self,
        embs_dir,
        vidc_dir: str = "dataset/",
        subset="",
        vision_feature_select_strategy="cls",
        max_frames=None,
    ):
        Chapters.__init__(self, vidc_dir=vidc_dir, subset=subset)

        self.embs_dir = Path(embs_dir)
        assert self.embs_dir.exists(), (
            f"Frames directory does not exist: {self.embs_dir}"
        )
        frames_pths = list(self.embs_dir.glob("*/*.pt"))
        frames_ids = {p.stem for p in frames_pths}
        assert len(frames_ids) > 0, f"No frames found in directory {self.embs_dir}"

        self.frames_ids = frames_ids & set(self.video_ids)
        assert len(self.frames_ids) > 0, f"No frames found in directory {self.embs_dir}"

        assert vision_feature_select_strategy in [
            "no_cls",
            "full",
            "cls",
        ], f"Invalid vision_feature_select_strategy: {vision_feature_select_strategy}"
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.max_frames = max_frames

        self._cached_frames = LRUCache()

    def prepare_emb(self, vid_embs: torch.Tensor):
        if len(vid_embs.shape) == 3:
            if self.vision_feature_select_strategy == "cls":
                vid_embs = vid_embs[:, 0, :]
            elif self.vision_feature_select_strategy == "no_cls":
                vid_embs = vid_embs[:, 1:, :]
            elif self.vision_feature_select_strategy == "full":
                pass
            else:
                raise ValueError(
                    f"Invalid vision_feature_select_strategy: {self.vision_feature_select_strategy}"
                )
        elif len(vid_embs.shape) == 2:
            if self.vision_feature_select_strategy == "cls":
                pass
            elif self.vision_feature_select_strategy == "no_cls":
                raise ValueError(
                    f"Invalid vision_feature_select_strategy: {self.vision_feature_select_strategy}"
                )
            elif self.vision_feature_select_strategy == "full":
                pass
        else:
            raise ValueError(f"Invalid vid_embs shape: {vid_embs.shape}")

        if self.max_frames is not None and vid_embs.shape[0] > self.max_frames:
            special_indices = list(
                np.linspace(0, vid_embs.shape[0] - 1, self.max_frames, dtype=int)
            )
            vid_embs = vid_embs[special_indices]

        return vid_embs.to(dtype=torch.float)

    def get_frames(self, video_id):
        # Try to get from cache first
        cached_frames = self._cached_frames.get(video_id)
        if cached_frames is not None:
            return cached_frames

        # Load from disk if not in cache
        vid_frames = torch.load(
            self.embs_dir / f"{video_id[:2]}" / f"{video_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        # Store in cache
        self._cached_frames.put(video_id, vid_frames)
        return vid_frames

    def get_frames_features(self, vid_id):
        vid_frames = self.get_frames(vid_id)
        return self.prepare_emb(vid_frames["embeddings"])

    def get_n_frames(self, vid_id):
        vid_frames = self.get_frames(vid_id)
        frames = vid_frames["frames"]
        if self.max_frames is None:
            return len(frames)
        return min(len(frames), self.max_frames)

    def get_frames_transcript(self, vid_id):
        vid_frames = self.get_frames(vid_id)
        frames = vid_frames["frames"]
        total_frames = vid_frames["total_frames"]
        total_time = vid_frames["total_time"]

        frame_times = [
            sec_to_hms(round(frame * total_time / total_frames)) for frame in frames
        ]

        num_frames = len(frame_times)
        if self.max_frames is not None and num_frames > self.max_frames:
            special_indices = set(
                np.linspace(0, num_frames - 1, self.max_frames, dtype=int)
            )
        else:
            special_indices = set(range(num_frames))

        frames_transcript = ""
        for i, frame_time in enumerate(frame_times):
            if self.max_frames is None or i in special_indices:
                frames_transcript += f"{frame_time}: <|reserved_special_token_0|>\n"
            else:
                frames_transcript += f"{frame_time}: image\n"

        return frames_transcript

    def __contains__(self, vid_id):
        return vid_id in self.frames_ids


class PromptFrames(Prompt):
    def get_task_prompt(self):
        # Customize the chapter prompt for caption-based prompts
        return "use the provided frames to identify distinct chapters based on content shifts.\n"

    def get_transcript(self, vid_id):
        return self.chapters.get_frames_transcript(vid_id)

    def get_frames_features(self, vid_id):
        return self.chapters.get_frames_features(vid_id)

    def __contains__(self, vid_id):
        # We already checked that the video exists in the frames dir
        return vid_id in self.chapters.frames_ids


if __name__ == "__main__":
    embs_dir = "dataset/embs/google/siglip-so400m-patch14-384_cls/HwwwH_MiniCPM-V-2/asr_s10k-2_train_preds+no-asr-10s"
    chapters = ChaptersFrames(
        embs_dir=embs_dir,
        subset="s1k_train",
        vision_feature_select_strategy="cls",
        max_frames=2,
    )
    vid_id = chapters.sample()

    prompt = PromptFrames(chapters=chapters)
    print(prompt.get_prompt_train(vid_id))
    print(prompt.get_transcript(vid_id))
    print(chapters.get_frames_features(vid_id).shape)
