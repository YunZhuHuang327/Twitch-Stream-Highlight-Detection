import re
import shutil
from pathlib import Path

from lutils import writef
from tqdm import tqdm

from src.data.chapters import hms_to_sec, sec_to_hms
from src.test.vidchapters import get_chapters as get_window_chapters
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def get_window(
    prompt: str,
    transcript: str,
    tokenizer,
    start_time: float = 0,
    window_token_size: int = 35_000,
):
    """
    Get a window of the transcript starting from start_time.

    Args:
        transcript: The full transcript text
        start_time: Start time in seconds to begin the window

    Returns:
        Tuple of (windowed_transcript, duration_str, reached_end)
        Returns (None, None, False) if no valid window can be created
    """
    pattern = r"(\d{2}:[0-5]\d:[0-5]\d)\b"

    # check transcript size, if it fits, return the whole transcript
    if len(tokenizer.encode(transcript)) <= window_token_size:
        return prompt, transcript, True

    window_lines = []
    n_tokens = 0
    last_timestamp = None
    reached_end = False
    start_time = hms_to_sec(start_time)

    lines = [line.strip() for line in transcript.split("\n") if line.strip()]
    for i, line in enumerate(lines):
        # Check if we've reached the last line
        if i == len(lines) - 1:
            reached_end = True

        match = re.search(pattern, line)
        if not match:
            # Skip lines that don't have a timestamp
            continue

        line_timestamp = hms_to_sec(match.group(1))

        # Skip lines before start_time
        if line_timestamp < start_time:
            continue

        # Count tokens in this line
        n_tokens += len(tokenizer.encode(line))
        if n_tokens > window_token_size:
            break

        shifted_timestamp = sec_to_hms(line_timestamp - start_time)
        line = re.sub(pattern, shifted_timestamp, line)
        window_lines.append(line)
        last_timestamp = line_timestamp

    # If we couldn't create a valid window
    if not window_lines or last_timestamp is None:
        return None, None, reached_end

    windowed_transcript = "\n".join(window_lines)
    duration = sec_to_hms(last_timestamp - start_time)
    # Change the duration of the video in the prompt
    prompt = re.sub(pattern, duration, prompt)

    return prompt, windowed_transcript, reached_end


def get_chapters(
    inference,
    prompt,
    transcript,
    max_new_tokens,
    do_sample=False,
    vid_duration=None,
    window_token_size=35_000,
    first_window_only=False,
):
    all_chapters = {}
    all_output_texts = []
    start_time = 0
    n_allowed_tries = 1 if first_window_only else 1_000_000 // window_token_size

    for _ in range(n_allowed_tries):
        # Get transcript window starting from start_time
        w_prompt, w_transcript, reached_end = get_window(
            prompt=prompt,
            transcript=transcript,
            tokenizer=inference.tokenizer,
            start_time=start_time,
            window_token_size=window_token_size,
        )

        if not w_transcript:
            break

        # Get chapters for this window
        output_text, chapters = get_window_chapters(
            inference=inference,
            prompt=w_prompt + w_transcript,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            vid_duration=vid_duration,
        )
        if chapters and start_time:
            chapters = {
                sec_to_hms(start_time + hms_to_sec(k)): v for k, v in chapters.items()
            }

        # If we got back a number instead of text, the input was too long
        if isinstance(output_text, int):
            break

        if chapters:
            # Only update keys that don't exist in all_chapters or have different values (to avoid overwriting with "Introduction" chapters)
            for k, v in chapters.items():
                if k not in all_chapters:
                    all_chapters[k] = v
            all_output_texts.append(output_text)

        # Stop if we've processed the whole video
        if reached_end:
            break

        # Move start_time to the last timestamp we processed
        timestamps = [hms_to_sec(k) for k in chapters]
        start_time = max(timestamps) if timestamps else 0

    return all_output_texts, all_chapters if all_chapters else None


class VidChaptersTesterWindow:
    def __init__(
        self, save_dir: str, window_token_size=35_000, first_window_only=False, **kwargs
    ):
        if "window" not in save_dir:
            save_dir = f"{save_dir}_window{window_token_size}"
        if first_window_only:
            save_dir = save_dir.replace("window", "first-window")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.window_token_size = window_token_size
        self.first_window_only = first_window_only

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

            chapters_pth = self.save_dir / f"{vid_id[:2]}" / f"{vid_id}.json"
            chapters_pth.parent.mkdir(exist_ok=True)

            if chapters_pth.exists():
                pbar.update(1)
                continue

            output_text, chapters = get_chapters(
                inference,
                prompt,
                transcript,
                max_new_tokens,
                do_sample=False,
                vid_duration=vid_duration,
                window_token_size=self.window_token_size,
                first_window_only=self.first_window_only,
            )

            if chapters:
                vid_data = {
                    "chapters": chapters,
                    "output": output_text[0] if len(output_text) == 1 else output_text,
                }
                writef(chapters_pth, vid_data)

            pbar.update(1)

        pbar.close()
