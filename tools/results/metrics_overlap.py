import numpy as np
import torch
from torchmetrics import Metric


class OverlapMetrics:
    def __init__(
        self,
        vid2preds: dict[
            str, dict[tuple[float, float], str] | list[tuple[float, float]]
        ],
        vid2refs: dict[str, dict[tuple[float, float], str] | list[tuple[float, float]]],
        verbose=False,
    ):
        """
        Args:
            vid2preds: Dictionary mapping video IDs to their predicted chapter segments.
                      Each value can be either:
                      - A list of (start_time, end_time) tuples representing chapter boundaries
                      - A dict mapping (start_time, end_time) tuple to chapter label
            vid2refs: Dictionary mapping video IDs to their ground truth chapter segments.
                     Same format as vid2preds.
        """
        pred_keys = set(vid2preds.keys())
        ref_keys = set(vid2refs.keys())
        missing_in_preds = ref_keys - pred_keys
        missing_in_refs = pred_keys - ref_keys

        # Check that all videos in predictions have corresponding references
        assert missing_in_refs == set(), (
            f"Videos missing in references: {missing_in_refs}"
        )
        if missing_in_preds and verbose:
            print(f"Videos missing in predictions: {missing_in_preds}")

        for vid_id, vid_preds in vid2preds.items():
            # Convert dict values to list of tuples if needed
            if isinstance(vid_preds, dict):
                vid2preds[vid_id] = list(vid_preds.keys())

            # Add segment from 0 to first pred if not already there
            if vid2preds[vid_id][0][0] != 0:
                vid2preds[vid_id] = [(0, vid2preds[vid_id][0][0])] + vid2preds[vid_id]

        for vid_id, vid_refs in vid2refs.items():
            if isinstance(vid_refs, dict):
                vid2refs[vid_id] = list(vid_refs.keys())

            # Add segment from 0 to first ref if not already there
            if vid2refs[vid_id][0][0] != 0:
                vid2refs[vid_id] = [(0, vid2refs[vid_id][0][0])] + vid2refs[vid_id]

        self.vid2preds = vid2preds
        self.vid2refs = vid2refs

        self.vid_ids = list(pred_keys)

    def get_metrics(self, metrics: list[str]):
        results = {}
        if "MAE" in metrics:
            results["MAE"] = round(self.evaluate_chapter_count()[0], 2)
        if "P" in metrics or "R" in metrics or "F1" in metrics:
            avg_overlap_results = self.compute_avg_thresholds()
            if "P" in metrics:
                results["P"] = round(avg_overlap_results["precision"], 2)
            if "R" in metrics:
                results["R"] = round(avg_overlap_results["recall"], 2)
            if "F1" in metrics:
                results["F1"] = round(avg_overlap_results["f1"], 2)
        if "Avg. TIoU" in metrics:
            results["Avg. TIoU"] = round(self.compute_avg_optimal(), 2)

        return results

    def compute_avg_optimal(self):
        vids_avg = []
        for vid_id in self.vid_ids:
            vid_refs = self.vid2refs[vid_id]
            vid_preds = self.vid2preds[vid_id]
            vids_avg.append(compute_vid_avg_optimal(vid_refs, vid_preds))

        return np.mean(vids_avg) * 100

    def compute_tiou_overlap(self, tiou_thr):
        prf_metric = PRFMetric()
        for vid_id in self.vid_ids:
            vid_refs = self.vid2refs[vid_id]
            vid_preds = self.vid2preds[vid_id]
            vid_p, vid_r = evaluate_vid_overlap(vid_refs, vid_preds, tiou_thr)
            prf_metric.update(vid_p, vid_r)
        return prf_metric.compute()

    def compute_avg_thresholds(self, tious_thr=None):
        if tious_thr is None:
            step = 0.05
            tious_thr = np.arange(0.5, 0.95 + step, step)
        thr2prf = {t: PRFMetric() for t in tious_thr}
        for vid_id in self.vid_ids:
            vid_refs = self.vid2refs[vid_id]
            vid_preds = self.vid2preds[vid_id]
            vid_overlaps = get_vid_overlaps(vid_refs, vid_preds)

            for t in tious_thr:
                vid_overlaps_t = vid_overlap_threshold_assignment(vid_overlaps, t)
                ref_set_covered = {ref_i for ref_i, _ in vid_overlaps_t}
                pred_set_covered = {pred_j for _, pred_j in vid_overlaps_t}
                vid_p = float(len(pred_set_covered)) / max(len(vid_preds), 1)
                vid_r = float(len(ref_set_covered)) / len(vid_refs)

                thr2prf[t].update(vid_p, vid_r)

        return {
            metric: np.mean([thr2prf[t].compute()[metric] for t in tious_thr])
            for metric in ["precision", "recall", "f1"]
        }

    def evaluate_chapter_count(self):
        return evaluate_chapter_count(self.vid2refs, self.vid2preds)


def compute_vid_avg_optimal(vid_refs, vid_preds):
    vid_overlaps = get_vid_overlaps(vid_refs, vid_preds)
    vid_overlaps = vid_overlap_optimal_assignment(vid_overlaps)
    return np.mean(list(vid_overlaps.values()))


def iou(interval_1, interval_2):
    """
    interval: list (2 float elements)
    """
    eps = 1e-8  # to avoid zero division
    (s_1, e_1) = interval_1
    (s_2, e_2) = interval_2

    intersection = max(0.0, min(e_1, e_2) - max(s_1, s_2))
    union = min(max(e_1, e_2) - min(s_1, s_2), e_1 - s_1 + e_2 - s_2)
    iou = intersection / (union + eps)
    return iou


def get_vid_overlaps(refs, preds):
    """
    refs are the video ground truths
    preds are the video predictions
    """
    vid_overlaps = {}
    for ref_i, ref in enumerate(refs):
        for pred_j, pred in enumerate(preds):
            iou_ij = iou(ref, pred)
            if iou_ij > 0:
                vid_overlaps[(ref_i, pred_j)] = iou_ij

    return vid_overlaps


def evaluate_vid_overlap(refs, preds, tiou):
    vid_overlaps = get_vid_overlaps(refs, preds)
    vid_overlaps = vid_overlap_threshold_assignment(vid_overlaps, tiou)
    ref_set_covered = {ref_i for ref_i, _ in vid_overlaps}
    pred_set_covered = {pred_j for _, pred_j in vid_overlaps}
    precision = float(len(pred_set_covered)) / max(len(preds), 1)
    recall = float(len(ref_set_covered)) / len(refs)

    return precision, recall


def vid_overlap_optimal_assignment(vid_overlaps):
    """
    vid_overlaps is a dictionary of video overlaps
    """

    # Initialize sets to keep track of covered references and predictions
    ref_set_covered = set()
    pred_set_covered = set()

    # Sort the IoUs in descending order based on their values
    sorted_vid_overlaps = sorted(vid_overlaps.items(), key=lambda x: x[1], reverse=True)

    connections = {}
    # Iterate through the sorted IoUs
    for (ref_i, pred_j), tiou_ij in sorted_vid_overlaps:
        # Check if the reference or prediction is already covered
        if ref_i not in ref_set_covered and pred_j not in pred_set_covered:
            # If not covered, mark them as covered
            ref_set_covered.add(ref_i)
            pred_set_covered.add(pred_j)
            # Output the chosen pair and its IoU

            # Calculate midpoints of predictions and references
            connections[(ref_i, pred_j)] = tiou_ij

    return connections


def vid_overlap_threshold_assignment(vid_overlaps, threshold):
    connections = {
        k: overlap for k, overlap in vid_overlaps.items() if overlap >= threshold
    }
    return connections


def evaluate_chapter_count(references, predictions):
    """
    Calculate the absolute error for each video and the mean absolute error across all videos.
    Args:
        references (dict): Ground truth segments for each video ID.
        predictions (dict): Predicted segments for each video ID.

    Returns:
        dict: Contains absolute errors per video and mean absolute error for all videos.
    """
    errors = []
    for vid_id in references:
        # if vid_id not in predictions:  # missing video prediction
        #     continue
        assert vid_id in predictions, f"Missing video {vid_id} in predictions"
        refs = references[vid_id]
        preds = predictions[vid_id]
        errors.append(len(refs) - len(preds))

    assert len(errors) > 0, "No videos to evaluate"

    errors = np.array(errors)
    errors_p = errors[errors > 0]
    errors_n = errors[errors < 0]
    errors_e = errors[errors == 0]
    # Under chaptering, Equal, Over chaptering
    ueo = [
        round(len(errors_p) * 100 / len(errors), 2),
        round(len(errors_e) * 100 / len(errors), 2),
        round(len(errors_n) * 100 / len(errors), 2),
    ]

    mae = float(sum(abs(errors)) / len(errors))
    return mae, ueo


class PRFMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("t_precision", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("t_recall", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("t_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, vid_p, vid_r) -> None:
        self.t_precision += vid_p
        self.t_recall += vid_r
        self.t_f1 += 2 * (vid_p * vid_r) / (vid_p + vid_r) if vid_p + vid_r else 0.0
        self.n += 1

    def compute(self):
        avg_p = self.t_precision * 100 / self.n
        avg_r = self.t_recall * 100 / self.n
        avg_f1 = self.t_f1 * 100 / self.n
        return {"precision": avg_p, "recall": avg_r, "f1": avg_f1}
