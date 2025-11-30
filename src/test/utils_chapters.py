import re


def extract_chapters(output: str | list[str]):
    """
    Extract chapters from the given output string or list of strings.

    Args:
        output (str | list[str]): The input text containing chapter information.
        vid_duration (str | None): The video duration in hh:mm:ss format. Default is None.

    Returns:
        dict: A dictionary of extracted chapters with timestamps as keys and titles as values.
    """

    # Only capture the first timestamp (hh:mm:ss) and ignore the second.
    pattern = r"(\d{2}:[0-5]\d:[0-5]\d)\b"
    chapters = {}

    if isinstance(output, str):
        output = output.split("\n")

    for line in output:
        if len(line) == 0:
            continue

        match = re.search(pattern, line)
        if match:
            time = match.group(1)
            # Strip any additional timestamp or text following it
            title = re.sub(pattern, "", line).strip()
            title = title.lstrip(" -:")  # Remove leading dash, colon, or space
            title = title.strip()
            if len(title) > 0:
                chapters[time] = title

    return chapters


def filter_chapters(chapters: dict, vid_duration: str | None = None):
    if vid_duration:
        filter_chapters = {}
        for k, v in sorted(chapters.items()):
            if k > vid_duration:
                break
            filter_chapters[k] = v
        chapters = filter_chapters

    # Check if chapters are in ordered by time
    times = list(chapters.keys())
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            return {}

    # remove empty chapters
    chapters = {k: v for k, v in chapters.items() if len(v) > 0}

    # if only one chapter at 00:00:00, return empty dict
    if len(chapters) == 1 and list(chapters.keys())[0] == "00:00:00":
        return {}

    return chapters


if __name__ == "__main__":
    # Example usage
    text = """
    00:00:00 Introduction - good
    00:05:30 - 00:05:33: Second Chapter
    00:05:33: Another Chapter
    00:90:00 - Wrong time
    00:42:00 - After video duration
    00:39:00 - What is this?
    01:04:00 - Outside of video duration
    """
    chapters = extract_chapters(text)
    chapters = filter_chapters(chapters, vid_duration="00:40:00")
    for time, title in chapters.items():
        print(f"Time: {time}, Title: {title}")
