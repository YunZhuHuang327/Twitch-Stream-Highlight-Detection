from pathlib import Path

from lutils import writef
from tqdm import tqdm

from src.test.utils_chapters import extract_chapters, filter_chapters
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def get_chapters(
    inference,
    prompt,
    max_new_tokens,
    do_sample=False,
    vid_duration=None,
    use_cache=True,
    vid_id="",
):
    output_text = inference(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        add_special_tokens=True,
        do_sample=do_sample,
        use_cache=use_cache,
    )

    if isinstance(output_text, int):
        # the input is too long, return the length of the input
        return output_text, None

    chapters = extract_chapters(output_text)
    chapters = filter_chapters(chapters, vid_duration=vid_duration)

    if not chapters and not do_sample:
        log.info(f"No chapters found for {vid_id}, trying again with sampling")
        return get_chapters(
            inference,
            prompt,
            max_new_tokens,
            do_sample=True,
            vid_duration=vid_duration,
        )

    return output_text, chapters


class VidChaptersTester:
    def __init__(self, save_dir: str, do_sample=False, **kwargs):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.do_sample = do_sample

    def __call__(
        self,
        inference,
        test_dataloader,
        max_new_tokens=1024,
    ):
        pbar = tqdm(
            total=len(test_dataloader),
            desc="Evaluating chapters",
        )

        for batch in test_dataloader:
            vid_id = batch["vid_id"][0]
            prompt = batch["prompt"][0]
            transcript = batch["transcript"][0]
            vid_duration = batch["vid_duration"][0]
            prompt += transcript

            chapters_pth = self.save_dir / f"{vid_id[:2]}" / f"{vid_id}.json"
            chapters_pth.parent.mkdir(exist_ok=True)

            if chapters_pth.exists():
                pbar.update(1)
                continue

            pbar.set_description(f"vid_id: {vid_id}")

            output_text, chapters = get_chapters(
                inference,
                prompt,
                max_new_tokens,
                do_sample=self.do_sample,
                vid_duration=vid_duration,
                vid_id=vid_id,
            )

            if chapters is None:
                log.info(f"Input too long for {vid_id}, {output_text} tokens")
                error_pth = chapters_pth.with_suffix(".txt")
                writef(error_pth, [output_text])
                pbar.update(1)
                continue

            if chapters:
                vid_data = {
                    "chapters": chapters,
                    "output": output_text,
                }
                writef(chapters_pth, vid_data)

            pbar.update(1)

        pbar.close()
