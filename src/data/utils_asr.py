from lutils import openf, writef

from src.data.chapters import Chapters, sec_to_hms
from src.data.prompt import Prompt
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ChaptersASR(Chapters):
    def __init__(self, vidc_dir: str = "dataset/", subset=""):
        super().__init__(vidc_dir=vidc_dir, subset=subset)

        self._asrs = None

    @property
    def asrs(self):
        if self._asrs is None:
            self.load_asr_data()
        return self._asrs

    def load_asr_data(self):
        if self._asrs is not None:
            return

        if self.subset:
            asr_pth = self.vidc_dir / f"docs/subset_data/asrs/asrs_{self.subset}.json"
            if asr_pth.exists():
                self._asrs = openf(asr_pth)
            else:
                log.info(f"ASR data not found for subset {self.subset}.")
                asr_val_pth = self.vidc_dir / "docs/subset_data/asrs/asrs_val.json"
                asr_train_pth = self.vidc_dir / "docs/subset_data/asrs/asrs_train.json"
                if "val" in self.subset and asr_val_pth.exists():
                    log.info("Loading from ASR validation file.")
                    asrs = openf(asr_val_pth)
                elif "train" in self.subset and asr_train_pth.exists():
                    log.info("Loading from ASR training file.")
                    asrs = openf(asr_train_pth)
                else:
                    log.info("Loading from ASR file.")
                    asrs = openf(self.vidc_dir / "docs/asrs.json")
                video_ids = set(self.video_ids) & set(asrs.keys())
                self._asrs = {vid_id: asrs[vid_id] for vid_id in video_ids}
                asr_pth.parent.mkdir(exist_ok=True)
                writef(asr_pth, self._asrs)
        else:
            self._asrs = openf(self.vidc_dir / "docs/asrs.json")

    def get_asr(self, video_id, add_end=False):
        if video_id not in self.asrs:
            return None

        asr = self.asrs[video_id]
        asr_clean = []
        for t, s, e in zip(asr["text"], asr["start"], asr["end"]):
            t = t.strip()
            s = sec_to_hms(s)
            e = sec_to_hms(e)
            if add_end:
                asr_clean.append(f"{s} - {e}: {t}")
            else:
                asr_clean.append(f"{s}: {t}")

        return "\n".join(asr_clean) + "\n"

    def __contains__(self, vid_id):
        return vid_id in self.asrs


class PromptASR(Prompt):
    def __init__(self, chapters: ChaptersASR, add_end=False):
        super().__init__(chapters=chapters)
        self.add_end = add_end

    def get_task_prompt(self):
        return "segment the text into distinct chapters based on thematic shifts or changes in topics.\n"

    def get_transcript(self, vid_id):
        vid_asr = self.chapters.get_asr(vid_id, add_end=self.add_end)
        assert vid_asr is not None, f"ASR not found for video ID: {vid_id}"
        return vid_asr

    def __contains__(self, vid_id):
        return vid_id in self.chapters


if __name__ == "__main__":
    chapters = ChaptersASR(subset="s10k_train")
    vid_id = chapters.sample()

    prompt = PromptASR(chapters=chapters)
    print(prompt.get_prompt_train(vid_id))
    print(prompt.get_transcript(vid_id))
    print(prompt.get_output(vid_id))
