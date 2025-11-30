import shutil
from collections import Counter

import cv2
import numpy as np
from decord import VideoReader, cpu  # pip install decord
from lutils import openf, writef
from PIL import Image
from tqdm import tqdm


def sample_frames(vlen, n_frames=15):
    acc_samples = min(vlen, n_frames)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


def extract_n_video_frames(
    video_path, num_frames=100, return_frame_idxs=False, return_pil=True
):
    """Extracts a specified number of frames from a video and converts them to PIL Images.

    Args:
        video_path (str): Path to the input video file.
        num_frames (int): Number of frames to extract.

    Returns:
        list: List of PIL Image objects.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file: " + video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return [], []

    frames_idxs = sample_frames(total_frames, n_frames=num_frames)

    frames = []
    frame_idxs = []
    for frame_number in frames_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            break

        if return_pil:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_image)
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_idxs.append(f"{frame_number}/{total_frames}")

    cap.release()
    if return_frame_idxs:
        return frames, frame_idxs
    else:
        return frames


def extract_video_frames(
    video_path, interval_sec=1, return_frame_idxs=False, return_pil=True
):
    """Extracts frames from a video at specified intervals and converts them to PIL Images.

    Args:
        video_path (str): Path to the input video file.
        interval_sec (int): Interval in seconds between frames to extract.
        return_frame_idxs (bool): Whether to return the frame indices.
        return_pil (bool): Whether to return frames as PIL Images.

    Returns:
        list: List of PIL Image objects or numpy arrays.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file: " + video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return [], []

    interval_frames = int(fps * interval_sec)

    frames = []
    frame_idxs = []
    for frame_number in range(0, total_frames, interval_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            break

        if return_pil:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_image)
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_idxs.append(f"{frame_number}/{total_frames}")

    cap.release()
    if return_frame_idxs:
        return frames, frame_idxs
    else:
        return frames


def encode_video(video_path, max_num_frames=64, return_pil=True):
    def uniform_sample(lst, n):
        gap = len(lst) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [lst[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = list(range(0, len(vr), sample_fps))
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    if return_pil:
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    else:
        frames = [v.astype("uint8") for v in frames]
    return frames


def extract_frames_at_timestamps(
    video_path, timestamps, return_frame_idxs=False, return_pil=True
):
    """Extracts frames from a video at specified timestamps and converts them to PIL Images.

    Args:
        video_path (str): Path to the input video file.
        timestamps (list): List of timestamps (in seconds) to extract frames from.
        return_frame_idxs (bool): Whether to return the frame indices.
        return_pil (bool): Whether to return frames as PIL Images.

    Returns:
        list: List of PIL Image objects or numpy arrays.
        list: List of frame indices (optional).
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file: " + video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return [], []

    target_frames = {int(timestamp * fps) for timestamp in timestamps}
    target_frames = list({f for f in target_frames if f < total_frames})
    target_frames.sort()

    frames = []
    frame_idxs = []
    for frame_idx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        if return_pil:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_image)
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_idxs.append(f"{frame_idx}/{total_frames}")

    cap.release()
    if return_frame_idxs:
        return frames, frame_idxs
    else:
        return frames


def merge_captions(dir_src, dir_dst):
    # Merge captions from src to dst
    assert dir_dst.exists()
    assert dir_src.exists()

    vid_pths_j = list(dir_src.glob("*/*.json"))

    vid_ids_j = [pth.stem for pth in vid_pths_j]

    total = len(vid_ids_j)
    types = []

    for vid_id in tqdm(vid_ids_j):
        vid_pth_a = dir_dst / vid_id[:2] / f"{vid_id}.json"
        vid_pth_j = dir_src / vid_id[:2] / f"{vid_id}.json"

        if not vid_pth_a.exists():
            # copy
            vid_pth_a.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(vid_pth_j, vid_pth_a)
            vid_pth_j.unlink()
            types.append("copy")

        else:
            cap_a = openf(vid_pth_a)
            cap_j = openf(vid_pth_j)

            if cap_j == cap_a:
                vid_pth_j.unlink()
                types.append("same")
            else:
                cap_aj = {**cap_a, **cap_j}.copy()
                writef(vid_pth_a, cap_aj)
                vid_pth_j.unlink()
                types.append("merge")

    if total > 0:
        print(f"Types: {Counter(types)}")
