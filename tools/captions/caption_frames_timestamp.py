from pathlib import Path

from lutils import openf, writef
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tools.captions.minicpm_batch import MiniCPM
from tools.captions.utils import extract_frames_at_timestamps, merge_captions


# class dataset to extract frames from videos
class VideoFrames(Dataset):
    def __init__(
        self,
        vidc_dir: Path,
        output_dir: Path,
        subset: str,
        shard_id: int,
    ):
        self.vidc_dir = Path(vidc_dir).resolve()
        self.output_dir = Path(output_dir).resolve()

        self.vid2timestamps = openf(
            self.vidc_dir / f"captions/missing_timestamps/{subset}_{shard_id}.json"
        )
        vid_ids = set(self.vid2timestamps.keys())
        output_paths = list(self.output_dir.glob("*/*.json"))
        output_ids = {p.stem for p in output_paths}
        self.vid_ids = list(vid_ids - output_ids)
        self.vid_ids.sort()

    def extract_frames(self, vid_id):
        try:
            video_path = self.vidc_dir / "videos" / f"{vid_id[:2]}" / f"{vid_id}.mp4"
            timestamps = self.vid2timestamps[vid_id]

            frames, frame_idxs = extract_frames_at_timestamps(
                str(video_path), timestamps, return_frame_idxs=True, return_pil=False
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
    shard_id=0,
):
    model = MiniCPM(model_name=model_name)

    data = VideoFrames(
        vidc_dir=vidc_dir,
        output_dir=output_dir,
        subset=subset,
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

        if output_path.exists():
            print(f"Skipping {vid_id} because it already exists")
            continue

        frames = [Image.fromarray(frame) for frame in frames]
        captions = model.caption_frames(frames)
        assert len(captions) == len(frames), (
            "Number of captions and frames do not match"
        )
        data = dict(zip(frame_idxs, captions))

        if len(data) == 0:
            continue

        writef(output_path, data)
        pbar.update(1)

    merge_captions(
        dir_src=output_dir,
        dir_dst=output_dir.parent / "all",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "shard_id",
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
        choices=[
            "openbmb/MiniCPM-V-2_6",
            "openbmb/MiniCPM-Llama3-V-2_5",
            "HwwwH/MiniCPM-V-2",
        ],
    )
    parser.add_argument(
        "--subset",
        default="s100_val",
        type=str,
    )
    args = parser.parse_args()

    output_dir = (
        args.vidc_dir
        / args.output_dir
        / args.model_name.replace("/", "_")
        / "all-missing"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    main(
        vidc_dir=args.vidc_dir,
        model_name=args.model_name,
        output_dir=output_dir,
        subset=args.subset,
        shard_id=args.shard_id,
    )
