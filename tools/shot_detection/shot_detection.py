from pathlib import Path

from lutils import openf, writef
from scenedetect import ContentDetector, SceneManager, StatsManager, open_video
from tqdm import tqdm


def process_video(vid_path, out_path):
    try:
        video = open_video(str(vid_path))

        scene_manager = SceneManager(stats_manager=StatsManager())
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(video=video)
        scenes = scene_manager.get_scene_list()

        vid_scene = {}
        if scenes:
            vid_scene["fps"] = scenes[0][1].get_framerate()
            vid_scene["frames"] = [scene[1].get_frames() for scene in scenes]
            vid_scene["seconds"] = [scene[1].get_seconds() for scene in scenes]
        else:
            vid_scene["fps"] = video.frame_rate
            vid_scene["frames"] = []
            vid_scene["seconds"] = []

        json_pth = out_path.with_suffix(".json")
        writef(vid_scene, json_pth)

        csv_pth = out_path.with_suffix(".csv")
        scene_manager.stats_manager.save_to_csv(csv_file=str(csv_pth))

    except Exception as e:
        print(f"Failed to process video {vid_path}: {e}")


def get_scene_boundaries(vid_path, base_out_dir=Path("dataset/shot_detection")):
    id = Path(vid_path).stem
    out_dir = base_out_dir / f"{id[:2]}"
    out_dir.mkdir(exist_ok=True)
    out_pth = out_dir / f"{id}.pkl"

    if not Path(vid_path).exists():
        process_video(vid_path, out_pth)

    return openf(out_pth)


def get_boundaries(vid_path, base_out_dir=Path("dataset/shot_detection")):
    boundaries = get_scene_boundaries(vid_path, base_out_dir)
    boundaries = [
        round(tuple_boundary[0].get_seconds()) for tuple_boundary in boundaries
    ]
    return boundaries


def main(
    num_shards=1,
    shard_id=0,
    vidc_dir=Path("dataset/"),
    subset="test",
):
    todo_ids = set(openf(vidc_dir / "docs" / "subset_data" / f"{subset}.json"))

    base_out_dir = vidc_dir / "shot_detection"
    base_out_dir.mkdir(exist_ok=True)

    done_pths = set(base_out_dir.glob("*/*.csv"))
    done_ids = {pth.stem for pth in done_pths}
    ids = list(todo_ids - done_ids)
    ids.sort()

    ids = [ids[i] for i in range(len(ids)) if i % num_shards == shard_id]

    for vid_id in tqdm(ids):
        vid_pth = vidc_dir / "videos" / f"{vid_id[:2]}/{vid_id}.mp4"
        if not vid_pth.exists():
            continue

        out_dir = base_out_dir / f"{vid_id[:2]}"
        out_dir.mkdir(exist_ok=True)
        out_pth = out_dir / f"{vid_id}.csv"
        if out_pth.exists():
            continue

        process_video(vid_pth, out_pth)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("num_shards", type=int)
    parser.add_argument("shard_id", type=int)
    parser.add_argument("--vidc_dir", type=Path, default="dataset/")
    parser.add_argument("--subset", type=str, default="test")
    args = parser.parse_args()

    subsets = args.subset.split(",") if "," in args.subset else [args.subset]
    for subset in subsets:
        main(args.num_shards, args.shard_id, args.vidc_dir, subset)
