from pathlib import Path
from typing import List

import cv2
import torch
from lutils import openf
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoEmbExtractor:
    def __init__(
        self, model_name: str = "google/siglip-so400m-patch14-384", output_dir: str = ""
    ):
        self.model = AutoModel.from_pretrained(
            model_name,  # attn_implementation="flash_attention_2",
        )
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.output_dir = output_dir

    def extract_embeds(self, video_path: str, frames_to_extract: List[int]) -> None:
        """
        Extract specific frames from a video and save them as images.

        Args:
            video_path: Path to the input video file
            frames_to_extract: List of frame numbers to extract (0-based index)
            output_dir: Directory where extracted frames will be saved
        """
        video_id = Path(video_path).stem
        output_path = self.output_dir / video_id[:2] / f"{video_id}.pt"
        if output_path.exists():
            return

        # Open the video file only if needed
        video = cv2.VideoCapture(video_path)

        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # video duration in seconds
        duration = total_frames / video.get(cv2.CAP_PROP_FPS)

        # Validate frame numbers
        valid_frames = [f for f in frames_to_extract if 0 <= f < total_frames]
        valid_frames.sort()
        if len(valid_frames) != len(frames_to_extract):
            print(
                f"Warning: Some requested frames are out of range (0-{total_frames - 1})"
            )

        max_frame = max(frames_to_extract)

        images = []
        for current_frame in range(max_frame + 1):
            ret, frame = video.read()
            if not ret:
                break

            if current_frame in frames_to_extract:
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        video.release()

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        output_path = self.output_dir / video_id[:2] / f"{video_id}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": image_features.cpu(),
                "frames": frames_to_extract,
                "total_frames": total_frames,
                "total_time": duration,
            },
            output_path,
        )


def main(
    vidc_dir: Path,
    captioner,
    frames,
    model_name: str = "google/siglip-so400m-patch14-384",
    subset: str = "s100_val",
    num_shards: int = 1,
    shard_id: int = 0,
) -> None:
    videos = openf(args.vidc_dir / f"docs/subset_data/{subset}.json")
    vid_extractor = VideoEmbExtractor(
        output_dir=vidc_dir / "embs" / f"{model_name}_cls" / captioner / frames,
        model_name=model_name,
    )

    videos.sort()
    videos = videos[shard_id::num_shards]

    for vid_id in tqdm(videos):
        vid_captions_path = (
            vidc_dir / "captions" / captioner / frames / f"{vid_id[:2]}/{vid_id}.json"
        )
        if not vid_captions_path.exists():
            continue
        vid_captions = openf(vid_captions_path)
        vid_frames = [int(frame.split("/")[0]) for frame in list(vid_captions.keys())]
        vid_path = vidc_dir / f"videos/{vid_id[:2]}/{vid_id}.mp4"
        if not vid_path.exists():
            continue
        vid_extractor.extract_embeds(vid_path, vid_frames)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vidc_dir", type=Path, default="dataset/")
    parser.add_argument("-s", "--subset", default="s100_val")
    parser.add_argument("--captioner", default="HwwwH_MiniCPM-V-2")
    parser.add_argument("--frames", default="asr_s10k-2_train_preds+no-asr-10s")
    parser.add_argument("--model_name", default="google/siglip-so400m-patch14-384")
    parser.add_argument("-n", "--num_shards", default=1, type=int)
    parser.add_argument("-i", "--shard_id", default=0, type=int)

    args = parser.parse_args()

    main(
        vidc_dir=args.vidc_dir,
        captioner=args.captioner,
        frames=args.frames,
        model_name=args.model_name,
        subset=args.subset,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
