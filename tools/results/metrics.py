from pathlib import Path

from lutils import openf

from src.data.chapters import Chapters, hms_to_sec
from tools.results.metrics_caption import (
    CaptionSegmentEvaluator,
    get_n_captions,
    get_n_captions_min,
)
from tools.results.metrics_overlap import OverlapMetrics
from tools.results.metrics_soda import SODA
from tools.results.utils import tokens_per_min


class MetricsExperiment:
    def __init__(self, chp: Chapters, exp_dir: Path, strict=True, verbose=True):
        self.chp = chp
        self.exp_dir = Path(exp_dir)

        predictions_pths = list(self.exp_dir.glob("*/*.json"))
        self.vid2preds = {pth.stem: openf(pth)["chapters"] for pth in predictions_pths}
        self.strict = strict
        self.verbose = verbose

    def __contains__(self, vid_id: str) -> bool:
        return vid_id in self.vid2preds

    def compute_metrics(self, subset: str):
        pass

    def get_subset_ids(self, subset: str):
        subset_ids = self.chp.get_subset_ids(subset)
        missing_videos = [
            vid_id for vid_id in subset_ids if vid_id not in self.vid2preds
        ]
        if not missing_videos:
            return subset_ids

        if self.verbose:
            print(
                f"Missing predictions for {len(missing_videos)} videos ({len(missing_videos) * 100 / len(subset_ids):.2f}%)"
            )
        if len(missing_videos) <= 10 and self.verbose:
            print(f"Missing videos: {missing_videos}")

        config_pth = self.exp_dir / ".." / ".hydra_train/overrides.yaml"
        if config_pth.exists():
            exp_flags = openf(config_pth)
            if not any(flag.startswith("subset_test=") for flag in exp_flags):
                exp_flags.append(f"subset_test={subset}")
            exp_flags = " ".join(exp_flags)
            print(f"run with: python test.py {exp_flags}")

        if self.strict:
            return None
        else:
            return [vid_id for vid_id in subset_ids if vid_id in self.vid2preds]

    def get_subset_preds(self, subset: str, hms=False, segments=False):
        subset_ids = self.get_subset_ids(subset)
        if subset_ids is None:
            return None
        vid2preds = {}
        for vid_id in subset_ids:
            vid2preds[vid_id] = self.get_vid_preds(vid_id, hms, segments)
        return vid2preds

    def get_vid_preds(self, vid_id: str, hms=False, segments=False):
        vid_preds = self.vid2preds[vid_id]
        if not hms:
            vid_preds = {hms_to_sec(hms): label for hms, label in vid_preds.items()}
        if segments:
            assert not hms, "With segments, cannot have hms format"
            vid_preds = self.chp.get_pred_segments(vid_id, vid_preds)
        return vid_preds

    def get_subset_refs(self, subset: str, hms=False, segments=False):
        subset_ids = self.chp.get_subset_ids(subset)

        vid2refs = {}
        for vid_id in subset_ids:
            vid2refs[vid_id] = self.chp.get_chapters(vid_id, hms=hms, segments=segments)
        return vid2refs

    def compute_metrics_overlap(self, subset: str, metrics=("F1", "Avg. TIoU")):
        vid2pred_segments = self.get_subset_preds(subset=subset, segments=True)
        if vid2pred_segments is None:
            return None
        vid2refs_segments = self.get_subset_refs(subset=subset, segments=True)
        overlap_metrics = OverlapMetrics(
            vid2pred_segments, vid2refs_segments, verbose=self.verbose
        )
        results = overlap_metrics.get_metrics(metrics)
        return results

    def compute_metrics_soda(self, subset: str, metrics=("F1")):
        vid2pred_segments = self.get_subset_preds(subset=subset, segments=True)
        if vid2pred_segments is None:
            return None
        vid2refs_segments = self.get_subset_refs(subset=subset, segments=True)
        soda = SODA(vid2pred_segments, vid2refs_segments)
        results = soda.get_metrics(metrics)
        return results

    def compute_metrics_captions(
        self, subset: str, metrics=("CIDEr",), tious=(0.3, 0.5, 0.7, 0.9)
    ):
        vid2preds = self.get_subset_preds(subset=subset, segments=True)
        if vid2preds is None:
            return None
        vid2refs = self.get_subset_refs(subset=subset, segments=True)
        caption_metrics = CaptionSegmentEvaluator(
            vid2refs=vid2refs, vid2preds=vid2preds, verbose=self.verbose
        )
        results = caption_metrics.get_metrics(scorers=metrics, tious=tious)
        return results

    def compute_avg_num_chapters(self, subset: str):
        subset_ids = self.get_subset_ids(subset)
        if subset_ids is None:
            return None

        vid_segments = []
        for vid_id in subset_ids:
            vid_preds = self.vid2preds[vid_id]
            vid_timestamps = [hms_to_sec(hms) for hms in vid_preds]
            vid_segments.extend(self.chp.get_pred_segments(vid_id, vid_timestamps))
        num_chapters = [len(self.vid2preds[vid_id]) for vid_id in subset_ids]
        return sum(num_chapters) / len(num_chapters)

    def get_n_captions(self, subset: str):
        subset_ids = self.get_subset_ids(subset)
        if subset_ids is None:
            return None
        return get_n_captions(self.exp_dir, subset_ids)

    def get_n_captions_min(self, subset: str):
        subset_ids = self.get_subset_ids(subset)
        if subset_ids is None:
            return None
        return get_n_captions_min(self.exp_dir, subset_ids, self.chp)

    def get_tokens_per_min(self, subset: str):
        return tokens_per_min(self.exp_dir, subset)
