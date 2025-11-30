from pathlib import Path
from typing import Any

from lutils import openf

from src.data.chapters import hms_to_sec


class CaptionSelection:
    def __init__(
        self,
        sampling_methods: list | tuple = (
            "100f",
            "10s",
            "60s",
            "shot-detection",
            "asr-preds",
        ),
        vidc_dir: Path = Path("dataset/"),
        base_dir: Path = Path("."),
        subset_train: str = "s10k-2_train",
        data_flags: str = "default",
        prompt: str = "asr",
        model: str = "Meta-Llama-3.1-8B-Instruct",
    ):
        assert isinstance(sampling_methods, (list, tuple))
        self.sampling_methods = sampling_methods
        self.shot_detection = ShotDetection(vidc_dir)
        self.asr_preds = ASRPreds(
            subset_train, base_dir, prompt=prompt, data_flags=data_flags, model=model
        )

    def _parse_method(self, method: str):
        """Parse sampling method string to determine type and value."""
        if method.endswith("f"):
            return "frames", int(method[:-1])
        elif method.endswith("s") and not method.startswith("asr"):
            return "seconds", float(method[:-1])
        return method, None

    def _merge_close_timestamps(
        self, timestamps: list, min_distance: float = 1.0
    ) -> list:
        """Merge timestamps that are closer than min_distance by taking their mean.
        Only merges consecutive timestamps one at a time."""
        if not timestamps:
            return timestamps

        timestamps = sorted(timestamps)
        result = [timestamps[0]]
        i = 1

        while i < len(timestamps):
            curr = timestamps[i]
            if curr - result[-1] < min_distance:
                # Merge current timestamp with the previous one
                result[-1] = (result[-1] + curr) / 2
            else:
                result.append(curr)
            i += 1

        return result

    def __call__(self, vid_id: str, duration: float) -> list:
        """Get timestamps for a video using all specified sampling methods."""
        all_timestamps = []

        for method in self.sampling_methods:
            method_type, value = self._parse_method(method)

            if method_type == "frames":
                timestamps = get_n_timestamps(duration, value)
            elif method_type == "seconds":
                timestamps = get_interval_timestamps(duration, value)
            elif method_type == "shot-midpoint":
                if self.shot_detection and vid_id in self.shot_detection:
                    timestamps = self.shot_detection.get_shots_midpoints(vid_id)
                else:
                    print(f"No shot detection for {vid_id}")
                    continue
            elif method_type == "shot-boundary":
                if self.shot_detection and vid_id in self.shot_detection:
                    timestamps = self.shot_detection.get_shots_boundaries(vid_id)
                else:
                    print(f"No shot detection for {vid_id}")
                    continue
            elif method_type == "asr-preds":
                if self.asr_preds and vid_id in self.asr_preds:
                    timestamps = self.asr_preds.get_timestamps(vid_id)
                else:
                    continue
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            all_timestamps.extend(timestamps)

        all_timestamps = sorted(set(all_timestamps))
        all_timestamps = [t for t in all_timestamps if t < duration]
        # Merge timestamps that are too close together
        merged_timestamps = self._merge_close_timestamps(all_timestamps)
        return merged_timestamps


def get_n_timestamps(duration: float, num_frames: int) -> list:
    """Returns a list of equally spaced timestamps for a video.

    Args:
        duration (float): Duration of the video in seconds
        num_frames (int): Number of frames to extract

    Returns:
        list: List of timestamps in seconds
    """
    if num_frames <= 0:
        return []

    if num_frames == 1:
        return [duration / 2]

    interval = duration / (num_frames + 1)
    timestamps = [i * interval for i in range(1, num_frames + 1)]

    return timestamps


def get_interval_timestamps(duration: float, interval_sec: float) -> list:
    """Returns a list of timestamps with equal intervals, centered within the video duration.

    Args:
        duration (float): Duration of the video in seconds
        interval_sec (float): Interval between timestamps in seconds

    Returns:
        list: List of timestamps in seconds
    """
    # Calculate number of intervals that fit within duration
    num_intervals = int(duration / interval_sec)

    if num_intervals <= 0:
        return []

    # Calculate total span of timestamps
    total_span = num_intervals * interval_sec

    # Calculate offset to center the timestamps
    start_offset = (duration - total_span) / 2

    # Generate timestamps
    timestamps = [start_offset + (i * interval_sec) for i in range(num_intervals + 1)]

    return timestamps


def select_furthest_timestamps(timestamp_list, target_count=100):
    # Step 1: Sort the list of timestamps
    timestamp_list.sort()

    # Step 2: Iteratively reduce timestamps until we have target_count left
    while len(timestamp_list) > target_count + 1:
        # Calculate distances between consecutive timestamps
        distances = [
            (timestamp_list[i + 1] - timestamp_list[i], i)
            for i in range(len(timestamp_list) - 1)
        ]

        # Find the index of the minimum distance
        min_distance_idx = min(distances, key=lambda x: x[0])[1]

        # Ensure min_distance_idx + 1 is within bounds
        if (
            min_distance_idx + 1 < len(distances)
            and distances[min_distance_idx][0] <= distances[min_distance_idx + 1][0]
        ):
            # Remove the frame at min_distance_idx + 1
            timestamp_list.pop(min_distance_idx + 1)
        else:
            # Remove the frame at min_distance_idx
            timestamp_list.pop(min_distance_idx)

    return timestamp_list


class ShotDetection:
    def __init__(self, vidc_dir: Path):
        shots_dir = vidc_dir / "shot_detection"
        self.shots_dir = shots_dir
        assert self.shots_dir.exists(), f"{self.shots_dir} does not exist"

        shot_pths = list(self.shots_dir.glob("**/*.json"))
        self.shot_pths = {pth.stem: pth for pth in shot_pths}
        self.shot_ids = list(self.shot_pths.keys())

    def __contains__(self, vid_id):
        return vid_id in self.shot_ids

    def get_shots(self, vid_id, key="seconds"):
        shots_file = self.shot_pths[vid_id]
        shots = openf(shots_file)
        return shots[key]

    def filter_shots(self, shots, max_num_shots=100):
        return select_furthest_timestamps(shots, max_num_shots)

    def get_shots_boundaries(self, vid_id, key="seconds", max_num_shots=100):
        shots = self.get_shots(vid_id, key)
        shots = [1] + shots
        shots = self.filter_shots(shots, max_num_shots)
        return shots

    def get_shots_midpoints(self, vid_id, key="seconds", max_num_shots=100):
        shots = self.get_shots(vid_id, key)
        shots = [1] + shots
        shots = self.filter_shots(shots, max_num_shots)
        return [(s1 + s2) / 2 for s1, s2 in zip(shots[:-1], shots[1:])]


class ASRPreds:
    def __init__(
        self,
        subset_train: str,
        base_dir: Path,
        model: str = "Meta-Llama-3.1-8B-Instruct",
        prompt: str = "asr",
        data_flags: str = "default",
    ):
        preds_dir = (
            base_dir
            / f"outputs/chapterize/{model}/{prompt}/{data_flags}/{subset_train}/default/test/"
        )
        self.subset_train = subset_train

        assert preds_dir.exists(), f"{preds_dir} does not exist"
        self.preds_dir = preds_dir

        preds_pths = list(preds_dir.glob("*/*.json"))
        self.pred2pth = {pth.stem: pth for pth in preds_pths}
        self.preds_ids = list(self.pred2pth.keys())

    def __contains__(self, vid_id):
        return vid_id in self.preds_ids

    def get_timestamps(self, vid_id):
        vid_pth = self.pred2pth[vid_id]
        if not vid_pth.exists():
            return []
        vid_captions = openf(vid_pth)
        if "chapters" not in vid_captions:
            return []
        timestamps = list(vid_captions["chapters"])
        timestamps = [hms_to_sec(t) for t in timestamps]
        # Many videos start with a black frame at t=0, so we replace the first timestamp
        # with t=1 to avoid capturing black frames
        timestamps = [1] + timestamps[1:] if timestamps[0] == 0 else [1] + timestamps
        return timestamps

    def __call__(self, vid_id: str) -> list:
        if vid_id not in self:
            return None
        return self.get_timestamps(vid_id)
