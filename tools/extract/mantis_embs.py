from pathlib import Path
from typing import List, Union

import torch
from mantis.models.mllava import LlavaForConditionalGeneration, MLlavaProcessor
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageProcessor:
    def __init__(
        self,
        vision_feature_select_strategy: str = "full",
        vision_feature_layer: int = -2,
        attn_implementation=None,  # or "flash_attention_2"
    ):
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        self.processor = MLlavaProcessor.from_pretrained(
            "TIGER-Lab/Mantis-8B-siglip-llama3"
        )
        self.image_processor = self.processor.image_processor

        self.model = LlavaForConditionalGeneration.from_pretrained(
            "TIGER-Lab/Mantis-8B-siglip-llama3",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )

    @torch.no_grad()
    def __call__(self, images: List[Union[str, Image.Image]]):
        if isinstance(images[0], str):
            images = [Image.open(image) for image in images]

        pixel_values = self.image_processor(images=images, return_tensors="pt")[
            "pixel_values"
        ]
        pixel_values = pixel_values.to(device)

        if isinstance(pixel_values, list):
            pixel_values = torch.cat([x for x in pixel_values if x is not None], dim=0)

        if pixel_values.dtype != self.model.vision_tower.dtype:
            pixel_values = pixel_values.type(self.model.vision_tower.dtype)

        pixel_batch_size = 8  # to avoid OOM
        all_image_features = []
        for i in range(0, pixel_values.shape[0], pixel_batch_size):
            batch_pixel_values = pixel_values[i : i + pixel_batch_size]
            batch_image_outputs = self.model.vision_tower(
                batch_pixel_values, output_hidden_states=True
            )
            batch_selected_image_features = batch_image_outputs.hidden_states[
                self.vision_feature_layer
            ]
            if self.vision_feature_select_strategy == "no_cls":
                batch_selected_image_features = batch_selected_image_features[:, 1:]
            elif self.vision_feature_select_strategy == "full":
                batch_selected_image_features = batch_selected_image_features
            elif self.vision_feature_select_strategy == "cls":
                batch_selected_image_features = batch_selected_image_features[:, 0]
            else:
                raise ValueError(
                    f"Unexpected select feature strategy: {self.vision_feature_select_strategy}"
                )

            all_image_features.append(batch_selected_image_features)

        image_features = torch.cat(all_image_features, dim=0)
        return image_features


class VideoEmbExtractor:
    def __init__(self, output_dir: str = ""):
        self.image_processor = ImageProcessor()
        self.output_dir = output_dir

    def extract_embeds(self, video_path: str, frames_to_extract: List[int]) -> None:
        """
        Extract specific frames from a video and save them as embeddings.

        Args:
            video_path: Path to the input video file
            frames_to_extract: List of frame numbers to extract (0-based index)
        """
        import cv2

        if not Path(video_path).exists():
            print(f"Video file does not exist: {video_path}")
            return

        video_id = Path(video_path).stem
        output_path = Path(self.output_dir) / video_id[:2] / f"{video_id}.pt"
        if output_path.exists():
            return

        # Open the video file
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

        image_features = self.image_processor(images)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": image_features.cpu().to(torch.float),
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
    subset: str = "s100_val",
    num_shards: int = 1,
    shard_id: int = 0,
) -> None:
    from lutils import openf

    videos = openf(vidc_dir / f"docs/subset_data/{subset}.json")
    vid_extractor = VideoEmbExtractor(
        output_dir=vidc_dir
        / "embs"
        / "TIGER-Lab/Mantis-8B-siglip-llama3"
        / captioner
        / frames
    )

    videos.sort()
    videos = videos[shard_id::num_shards]

    for vid_id in tqdm(videos):
        vid_captions = openf(
            vidc_dir / "captions" / captioner / frames / f"{vid_id[:2]}/{vid_id}.json"
        )
        vid_frames = [int(frame.split("/")[0]) for frame in list(vid_captions.keys())]
        vid_path = vidc_dir / f"videos/{vid_id[:2]}/{vid_id}.mp4"
        vid_extractor.extract_embeds(vid_path, vid_frames)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vidc_dir", type=Path, default="dataset/")
    parser.add_argument("-s", "--subset", default="s100_val")
    parser.add_argument("--captioner", default="HwwwH_MiniCPM-V-2")
    parser.add_argument("--frames", default="asr_s10k-2_train_preds+no-asr-10s")
    parser.add_argument("-n", "--num_shards", default=1, type=int)
    parser.add_argument("-i", "--shard_id", default=0, type=int)

    args = parser.parse_args()

    main(
        vidc_dir=args.vidc_dir,
        captioner=args.captioner,
        frames=args.frames,
        subset=args.subset,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
