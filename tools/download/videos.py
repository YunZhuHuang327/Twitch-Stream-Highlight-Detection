import json
from pathlib import Path

import yt_dlp
from tqdm import tqdm


def download_videos(json_file_path, output_dir="dataset/videos"):
    """
    Download videos from a JSON file containing video IDs.

    Args:
        json_file_path: Path to the JSON file containing video IDs
        output_dir: Directory to save downloaded videos
    """
    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)

    # Load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Ensure the data is a list of video IDs
    if isinstance(data, dict) and "video_ids" in data:
        video_ids = data["video_ids"]
    elif isinstance(data, list):
        video_ids = data
    else:
        raise ValueError(
            "JSON file must contain either a list of video IDs or a dict with 'video_ids' key"
        )

    print(f"Downloading {len(video_ids)} videos to {output_dir}")

    # Download each video
    tqdm_video_ids = tqdm(video_ids)
    for video_id in tqdm_video_ids:
        tqdm_video_ids.set_postfix(video_id=video_id)

        # Create subdirectory based on first two characters of video ID
        subdir = output_dir / video_id[:2]
        subdir.mkdir(parents=True, exist_ok=True)

        output_path = subdir / f"{video_id}.mp4"

        # Skip if file already exists
        if output_path.exists():
            continue

        # Configure yt-dlp options
        ydl_opts = {
            "format": "best",
            "outtmpl": str(output_path),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "no_progress": True,
        }

        # Download the video
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if not output_path.exists():
                print(f"Failed to download {video_id}: file not found after download")
        except Exception as e:
            print(f"Error downloading {video_id}: {str(e)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download videos from a JSON file of video IDs"
    )
    parser.add_argument(
        "json_file", type=Path, help="Path to JSON file containing video IDs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="dataset/videos",
        help="Directory to save downloaded videos (default: dataset/videos)",
    )

    args = parser.parse_args()

    # Verify the JSON file exists
    if not args.json_file.exists():
        print(f"Error: JSON file {args.json_file} not found")
        return

    # Verify the output directory exists
    if not args.output_dir.exists():
        print(
            f"Error: Output directory {args.output_dir} does not exist, creating it now..."
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)

    download_videos(args.json_file, args.output_dir)


if __name__ == "__main__":
    main()
