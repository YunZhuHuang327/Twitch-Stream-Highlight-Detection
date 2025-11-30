import random
from pathlib import Path

from lutils import openf, writef


class Chapters:
    def __init__(self, vidc_dir: str = "dataset/", subset="", videos_dir="videos"):
        self.vidc_dir = Path(vidc_dir)
        assert self.vidc_dir.exists(), f"Directory {vidc_dir} does not exist."
        self.subset = subset

        self.data = self.load_subset_data(subset=subset)
        self.video_ids = list(self.data.keys())
        assert len(self.video_ids) == len(self.data), (
            f"len(data)= {len(self.data)} != len(ids)= {len(self.video_ids)}."
        )

        self.videos_dir = videos_dir

    def get_subset_ids(self, subset: str):
        return openf(self.vidc_dir / f"docs/subset_data/{subset}.json")

    def load_subset_data(self, subset=""):
        if subset == "":
            data_path = self.vidc_dir / "docs/chapters.json"
            assert data_path.exists(), f"Data file {data_path} does not exist."
            data = openf(data_path)
            return data

        data_path = self.vidc_dir / f"docs/subset_data/chapters/chapters_{subset}.json"
        if not data_path.exists():
            video_ids = openf(self.vidc_dir / f"docs/subset_data/{subset}.json")
            data = openf(self.vidc_dir / "docs/chapters.json")
            data = {video_id: data[video_id] for video_id in video_ids}
            data_path.parent.mkdir(exist_ok=True)
            writef(data, data_path)
        else:
            data = openf(data_path)
        return data

    def __len__(self):
        return len(self.video_ids)

    def __iter__(self):
        return iter(self.video_ids)

    def __contains__(self, vid_id):
        return vid_id in self.data

    def __getitem__(self, idx):
        if isinstance(idx, int):
            video_info = self.get_video_info(self.video_ids[idx])
            video_info["video_id"] = self.video_ids[idx]
            return video_info
        elif isinstance(idx, str):
            return self.get_video_info(idx)
        else:
            raise ValueError(f"Invalid index type {type(idx)}.")

    def get_video_info(self, video_id):
        assert video_id in self.data, f"Video ID {video_id} not found in data."
        return self.data[video_id]

    def get_chapters(self, video_id, hms=False, segments=False):
        """Retrieve chapters for a specific video ID."""
        video_info = self.get_video_info(video_id)

        vid_chapters = video_info.get("chapters", {})
        chapter_timestamps = {}
        for time, label in vid_chapters.items():
            time = sec_to_hms(time) if hms else hms_to_sec(time)
            chapter_timestamps[time] = label
        if not segments:
            return chapter_timestamps

        # If segments is True, we return the timestamps as segments
        assert not hms, "hms must be False if segments is True."
        timestamps = list(chapter_timestamps.keys())
        start_times = timestamps
        end_times = timestamps[1:] + [self.get_duration(video_id)]
        segmented_chapters = {}

        for start_time, end_time in zip(start_times, end_times):
            segment = (start_time, end_time)
            segmented_chapters[segment] = chapter_timestamps[start_time]

        return segmented_chapters

    def get_labels(self, video_id):
        """Retrieve a list of chapter labels for a specific video ID."""
        chapters = self.get_chapters(video_id)
        return list(chapters.values())

    def get_timestamps(
        self, video_id, zero_handling="default", duration_handling="default"
    ):
        """Retrieve a list of chapter timestamps for a specific video ID."""
        assert zero_handling in [
            "default",
            "add",
            "remove",
        ], f"Invalid zero handling {zero_handling}."

        assert duration_handling in [
            "default",
            "add",
            "remove",
        ], f"Invalid duration handling {duration_handling}."

        chapters = self.get_chapters(video_id)
        timestamps = [int(time) for time in chapters]

        # Handle zero timestamps based on the flag
        if zero_handling == "add":
            timestamps = (
                [0] + timestamps if timestamps and timestamps[0] != 0 else timestamps
            )
        elif zero_handling == "remove":
            timestamps = [time for time in timestamps if time != 0]

        if duration_handling == "add":
            duration = self.get_duration(video_id)
            timestamps = (
                timestamps + [duration] if timestamps[-1] != duration else timestamps
            )
        elif duration_handling == "remove":
            duration = self.get_duration(video_id)
            timestamps = timestamps[:-1] if timestamps[-1] == duration else timestamps

        return timestamps

    def get_n_timestamps(self, video_id, zero_handling="default"):
        """Retrieve the number of chapter timestamps for a specific video ID."""
        return len(self.get_timestamps(video_id, zero_handling=zero_handling))

    def get_n_chapters(self, video_id):
        return len(self.get_gt_segments(video_id))

    def get_n_labels(self, video_id):
        return len(self.get_labels(video_id))

    def get_duration(self, video_id, hms=False):
        """Retrieve the duration of a specific video ID."""
        video_info = self.get_video_info(video_id)
        duration = video_info.get("duration")
        if hms:
            return sec_to_hms(duration)
        return duration

    def get_hms_duration(self, video_id, string=True):
        """Retrieve the duration of a specific video ID in hours, minutes, and seconds."""
        h, m, s = self.get_duration(video_id)
        if string:
            return f"{h:02d}:{m:02d}:{s:02d}"
        else:
            return h, m, s

    def get_title(self, video_id):
        """Retrieve the title of a specific video ID."""
        video_info = self.get_video_info(video_id)
        return video_info.get("title")

    def get_description(self, video_id):
        """Retrieve the description of a specific video ID."""
        video_info = self.get_video_info(video_id)
        return video_info.get("description")

    def get_channel_id(self, video_id):
        """Retrieve the channel ID of a specific video ID."""
        video_info = self.get_video_info(video_id)
        return video_info.get("channel_id")

    def get_view_count(self, video_id):
        """Retrieve the view count of a specific video ID."""
        video_info = self.get_video_info(video_id)
        return video_info.get("view_count")

    def get_video_path(self, video_id):
        """Retrieve the path to the video file for a specific video ID."""
        video_pth = (
            self.vidc_dir / self.videos_dir / f"{video_id[:2]}" / f"{video_id}.mp4"
        )
        assert video_pth.exists(), f"Video file {video_pth} does not exist."
        return str(video_pth)

    def sample(self, n=1):
        """Sample n video IDs."""
        sample = random.sample(self.video_ids, n)

        if n == 1:
            return sample[0]
        else:
            return sample

    def get_gt_segments(self, video_id, zero_handling="add"):
        """Generate ground truth segments based on video ID with options to adjust zero timestamps."""
        timestamps = self.get_timestamps(video_id, zero_handling=zero_handling)
        segments = boundary2seg(
            timestamps, self.get_duration(video_id), zero_handling=zero_handling
        )
        return segments

    def get_segments(self, video_id, zero_handling="add"):
        return self.get_gt_segments(
            video_id,
            zero_handling=zero_handling,
        )

    def get_all_gt_segments(self, zero_handling="add"):
        """Generate ground truth segments for all video IDs."""
        return {
            video_id: self.get_gt_segments(video_id, zero_handling=zero_handling)
            for video_id in self.video_ids
        }

    def get_pred_segments(self, vid_id, vid_preds, zero_handling="add"):
        duration = self.get_duration(vid_id)
        if isinstance(vid_preds, list):
            # vid_preds are the timestamps
            vid_preds = (
                [hms_to_sec(hms) for hms in vid_preds]
                if isinstance(vid_preds[0], str)
                else vid_preds
            )
            return boundary2seg(vid_preds, duration, zero_handling=zero_handling)
        elif isinstance(vid_preds, dict):
            # vid_preds are the chapters with key timestamps
            vid_preds_new = {}
            start_times = list(vid_preds.keys())
            end_times = start_times[1:] + [duration]
            for start_time, end_time in zip(start_times, end_times):
                segment = (hms_to_sec(start_time), hms_to_sec(end_time))
                vid_preds_new[segment] = vid_preds[start_time]
            return vid_preds_new

    def convert_predictions_to_segments(self, preds):
        segments = {}
        for video_id, vid_preds in preds.items():
            segments[video_id] = self.get_pred_segments(video_id, vid_preds)

        return segments

    def get_link(self, video_id):
        return f"https://www.youtube.com/watch?v={video_id}"

    def get_url(self, video_id):
        return f"https://www.youtube.com/watch?v={video_id}"

    @staticmethod
    def sec_to_hms(seconds, string=True, short=False):
        return sec_to_hms(seconds, string=string, short=short)

    @staticmethod
    def hms_to_sec(time_str, enable_single_part=False):
        return hms_to_sec(time_str, enable_single_part=enable_single_part)

    @staticmethod
    def clean_segment(segment, zero_handling="add"):
        return clean_segment(segment, zero_handling=zero_handling)

    @staticmethod
    def clean_timestamps(timestamps, zero_handling="remove"):
        return clean_tiemstamps(timestamps, zero_handling=zero_handling)


def boundary2seg(boundaries, duration, zero_handling="add"):
    if zero_handling == "add" and boundaries[0] != 0:
        boundaries = [0] + boundaries

    gt = []
    for i in range(len(boundaries)):
        if i < len(boundaries) - 1:
            gt.append((float(boundaries[i]), float(boundaries[i + 1])))
        else:
            # Check if the last boundary equals the duration
            if boundaries[i] != duration:
                gt.append((float(boundaries[i]), float(duration)))
    return gt


def sec_to_hms(seconds, string=True, short=False):
    """Convert seconds to hours, minutes, and seconds."""
    if isinstance(seconds, str) and ":" in seconds:
        return sec_to_hms(hms_to_sec(seconds), string=string, short=short)
    if isinstance(seconds, str) and seconds.isdigit() or isinstance(seconds, float):
        seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if string:
        if h == 0 and short:
            return f"{m:02d}:{s:02d}"
        return f"{h:02d}:{m:02d}:{s:02d}"
    return h, m, s


def hms_to_sec(time_str, enable_single_part=False):
    """Convert hours, minutes, and seconds to total seconds."""
    if isinstance(time_str, (int, float)):
        return time_str
    if isinstance(time_str, str) and time_str.isdigit():
        return int(time_str)

    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        seconds = float(seconds) if "." in seconds else int(seconds)
        minutes = int(minutes)
        if minutes >= 60 or seconds >= 60:
            return False
        total_seconds = int(hours) * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = parts
        seconds = float(seconds) if "." in seconds else int(seconds)
        minutes = int(minutes)
        if seconds >= 60:
            return False
        total_seconds = int(minutes) * 60 + seconds
    elif len(parts) == 1 and enable_single_part:
        seconds = float(parts[0]) if "." in parts[0] else int(parts[0])
        total_seconds = seconds
    else:
        raise ValueError("Invalid time format")
    return total_seconds


def clean_segment(segment, zero_handling="add"):
    if zero_handling == "add" and segment[0][0] != 0.0:
        segment.insert(0, [0.0, segment[0][0]])
    elif zero_handling == "remove" and segment[0][0] == 0.0:
        segment.pop(0)
    return segment


def clean_tiemstamps(timestamps, zero_handling="remove"):
    if zero_handling == "remove":
        return [time for time in timestamps if time != 0]
    elif zero_handling == "add":
        return [0] + timestamps if timestamps[0] != 0 else timestamps
    else:
        return timestamps
