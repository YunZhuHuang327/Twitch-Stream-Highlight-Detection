from pathlib import Path

from lutils import openf

from src.data.chapters import Chapters, sec_to_hms
from src.data.prompt import Prompt
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ChaptersCaptions(Chapters):
    def __init__(self, captions_dir, vidc_dir: str = "dataset/", subset=""):
        super().__init__(vidc_dir=vidc_dir, subset=subset)

        self.captions_dir = Path(captions_dir)
        assert (
            self.captions_dir.exists()
        ), f"Captions directory does not exist: {self.captions_dir}"
        captions_pths = list(self.captions_dir.glob("*/*.json"))
        captions_ids = {p.stem for p in captions_pths}
        assert (
            len(captions_ids) > 0
        ), f"No captions found in directory {self.captions_dir}"

        self.captions_ids = captions_ids

    def get_caption(self, video_id):
        vid_captions = openf(self.captions_dir / f"{video_id[:2]}" / f"{video_id}.json")
        vid_duration = self.get_duration(video_id)

        vid_captions = self.prepare_captions(vid_captions, vid_duration)
        return vid_captions

    @staticmethod
    def prepare_captions(vid_captions, vid_duration):
        caption_clean = ""
        vid_timestamps = sorted(vid_captions, key=lambda x: int(x.split("/")[0]))
        for frame_pct in vid_timestamps:
            frame_idx, n_frames = frame_pct.split("/")
            frame_time = vid_duration * int(frame_idx) / int(n_frames)
            frame_time = sec_to_hms(frame_time)
            caption = vid_captions[frame_pct]
            caption_clean += f"{frame_time}: {caption}\n"

        caption_clean = caption_clean.strip()
        return caption_clean

    def __contains__(self, vid_id):
        return vid_id in self.captions_ids


class PromptCaptions(Prompt):
    def get_task_prompt(self):
        # Customize the chapter prompt for caption-based prompts
        return "use the provided captions to identify distinct chapters based on content shifts.\n"

    def get_transcript(self, vid_id):
        return self.chapters.get_caption(vid_id)

    def __contains__(self, vid_id):
        # We already checked that the video exists in the captions dir
        return vid_id in self.chapters.captions_ids


if __name__ == "__main__":
    captions_dir = Path("dataset/captions/openbmb_MiniCPM-V-2_6/10s")
    chapters = ChaptersCaptions(captions_dir=captions_dir, subset="s1k_train")
    vid_id = chapters.sample()
    # print(chapters.get_caption(vid_id))

    prompt = PromptCaptions(chapters=chapters)
    print(prompt.get_prompt_train(vid_id))
