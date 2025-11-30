from pathlib import Path

from lutils import openf, writef
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.chapters import Chapters
from tools.captions.caption_selection import CaptionSelection
from tools.captions.minicpm import MiniCPM
from tools.captions.utils import extract_frames_at_timestamps


# class dataset to extract frames from videos
class VideoFrames(Dataset):
    def __init__(
        self,
        vidc_dir,
        output_dir,
        subset="",
        sampling_methods=("100f", "10s", "60s", "shot-detection", "asr-preds"),
        num_shards=1,
        shard_id=0,
    ):
        self.chp = Chapters(vidc_dir=vidc_dir, subset=subset)
        self.output_dir = output_dir

        self.cs = CaptionSelection(sampling_methods)

        vid_ids = set(self.chp)
        # output_paths = list(output_dir.glob("*/*.json"))
        # output_ids = {p.stem for p in output_paths}
        # self.vid_ids = list(vid_ids - output_ids)
        self.vid_ids = list(vid_ids)
        self.vid_ids.sort()
        total_vids = len(self.vid_ids)

        shard_size = total_vids // num_shards
        start_idx = shard_id * shard_size
        end_idx = start_idx + shard_size if shard_id < num_shards - 1 else total_vids
        self.vid_ids = self.vid_ids[start_idx:end_idx]

    def extract_frames(self, vid_id):
        try:
            vid_duration = self.chp.get_duration(vid_id)
            video_path = self.chp.get_video_path(vid_id)
            timestamps = self.cs(vid_id, vid_duration)

            output_path = self.output_dir / f"{vid_id[:2]}/{vid_id}.json"
            if output_path.exists():
                captions = openf(output_path)
                n_frames = int(list(captions.keys())[0].split("/")[1])
                done_timestamps = [
                    int(ts.split("/")[0]) * vid_duration / n_frames for ts in captions
                ]
            else:
                done_timestamps = []

            # Filter out timestamps that are too close to already processed ones
            filtered_timestamps = []
            for ts in timestamps:
                # Check if any done timestamp is within 1 second
                if not any(abs(ts - done_ts) < 1.0 for done_ts in done_timestamps):
                    filtered_timestamps.append(ts)
            timestamps = filtered_timestamps

            if len(timestamps) == 0:
                print(f"No new frames to extract for {vid_id}")
                return [], []

            frames, frame_idxs = extract_frames_at_timestamps(
                video_path, timestamps, return_frame_idxs=True, return_pil=False
            )

            return frames, frame_idxs
        except Exception as e:
            print(f"Error extracting frames for {vid_id}: {e}")
            return [], []

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        frames, frame_idxs = self.extract_frames(vid_id)

        output_path = self.output_dir / f"{vid_id[:2]}/{vid_id}.json"
        output_path.parent.mkdir(exist_ok=True)

        return frames, frame_idxs, str(output_path)


def collate_fn(batch):
    # Unpack the batch (since batch_size=1, batch will be a list of a single item)
    frames, frame_idxs, output_path = batch[0]
    return frames, frame_idxs, output_path


def main(
    vidc_dir,
    model_name,
    output_dir,
    subset="short100_val",
    sampling_methods=("10s", "60s", "shot-detection", "asr-preds"),
    num_shards=1,
    shard_id=0,
):
    model = MiniCPM(model_name=model_name)

    data = VideoFrames(
        vidc_dir=vidc_dir,
        output_dir=output_dir,
        subset=subset,
        sampling_methods=sampling_methods,
        num_shards=num_shards,
        shard_id=shard_id,
    )
    data_loader = DataLoader(data, batch_size=1, num_workers=4, collate_fn=collate_fn)
    pbar = tqdm(
        total=len(data_loader),
        desc="Extracting frames",
    )

    for frames, frame_idxs, output_path in data_loader:
        output_path = Path(output_path)
        vid_id = output_path.stem
        pbar.set_description(f"vid_id: {vid_id}")

        data = openf(output_path) if output_path.exists() else {}
        for frame, frame_idx in zip(frames, frame_idxs):
            frame_caption = model.chat(Image.fromarray(frame), "What is in the image?")
            data[frame_idx] = frame_caption

        if len(data) == 0:
            print(f"No frames extracted for {vid_id}")
            continue

        writef(output_path, data)
        pbar.update(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-n",
        "--num_shards",
        default=1,
        type=int,
        help="Number of shards to subset the data into.",
    )
    parser.add_argument(
        "-i",
        "--shard_id",
        default=0,
        type=int,
        help="Shard id to process.",
    )
    parser.add_argument("--vidc_dir", default="dataset/", type=Path)
    parser.add_argument(
        "--output_dir",
        default="captions",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        default="HwwwH/MiniCPM-V-2",
        type=str,
    )
    parser.add_argument(
        "--subset",
        default="s100_val",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--sampling_methods",
        default="100f,10s,60s,shot-detection,asr-preds",
        help="Frame extraction parameter, e.g., '10s' for 10 second interval.",
    )
    args = parser.parse_args()

    output_dir = (
        args.vidc_dir / args.output_dir / args.model_name.replace("/", "_") / "all"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    args.sampling_methods = args.sampling_methods.split(",")
    args.sampling_methods = [m.strip() for m in args.sampling_methods]
    print(args.sampling_methods)

    main(
        vidc_dir=args.vidc_dir,
        model_name=args.model_name,
        output_dir=output_dir,
        subset=args.subset,
        sampling_methods=args.sampling_methods,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
