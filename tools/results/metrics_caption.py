# --------------------------------------------------------
# evaluation scripts for dense video captioning, support python 3
# Modified from https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c
# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# --------------------------------------------------------

import random
import string
from pathlib import Path

import numpy as np
from lutils import openf
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from tools.results.metrics_overlap import iou
from tools.results.utils import format_number_with_z, load_config

tokenizer = PTBTokenizer()


def random_string(string_length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(string_length))


def remove_nonascii(text):
    return "".join([i if ord(i) < 128 else " " for i in text])


def preprocess_vid2chapters(vid2chapters):
    """
    Preprocess the chapters to fit the input of pycocoevalcap scorers.


    """
    vid2chapters_segments = {
        vid_id: list(chapters.keys()) for vid_id, chapters in vid2chapters.items()
    }
    vid2chapters_labels = {
        vid_id: [{"caption": remove_nonascii(label)} for label in chapters.values()]
        for vid_id, chapters in vid2chapters.items()
    }

    vid2chapters_tokenized = tokenizer.tokenize(vid2chapters_labels)
    vid2chapters_clean = {}
    for vid_id, segments in vid2chapters_segments.items():
        labels = vid2chapters_tokenized[vid_id]
        if "" in labels:
            # Skip videos with empty captions (different language that tokenizer doesn't understand)
            continue
        assert len(segments) == len(labels), (
            f"len(segments) != len(labels) for vid_id {vid_id}"
        )
        vid2chapters_clean[vid_id] = dict(zip(segments, labels))
    return vid2chapters_clean


class CaptionSegmentEvaluator:
    def __init__(
        self,
        vid2refs,
        vid2preds,
        verbose=False,
    ):
        """
        Initializes the VideoCaptionEvaluator class.

        Parameters:
        - vid2refs: Dictionary containing reference ground truth captions with dicts of (start, end): caption.
        - vid2preds: Dictionary containing predicted captions with dicts of (start, end): caption.

        Both vid2refs and vid2preds are preprocessed to standardize the chapter structure.
        """
        vid2preds = preprocess_vid2chapters(vid2preds)
        vid2refs = preprocess_vid2chapters(vid2refs)

        # Check for missing videos in predictions and remove them from references
        missing_vids = set(vid2refs.keys()) - set(vid2preds.keys())
        if len(missing_vids) > 0:
            if verbose:
                print(
                    f"Missing predictions for {len(missing_vids)} videos "
                    f"({len(missing_vids) / len(vid2refs):.2%})."
                )
            for vid in missing_vids:
                del vid2refs[vid]

        self.vid2refs = vid2refs
        self.vid2preds = vid2preds

        self.scorers = {
            "METEOR": Meteor(),
            "CIDEr": Cider(),
            "Rouge-L": Rouge(),
            "Bleu": Bleu(4),
        }

    def get_metrics(self, tious=(0.3, 0.5, 0.7, 0.9), scorers=("CIDEr",)):
        """
        Evaluates predictions against ground truths for given tIoU thresholds and metrics.

        Parameters:
        - tious: Tuple of temporal Intersection over Union (tIoU) thresholds.
        - scorers: Tuple of scoring metrics to use, default is CIDEr.

        Returns:
        - Dictionary of scores for each metric and tIoU threshold.
        """
        if isinstance(scorers, str):
            scorers = (scorers,)

        scores = {}
        for tiou in tious:
            scores_tiou = self.evaluate_tiou(tiou, scorers)
            for metric, score in scores_tiou.items():
                if metric not in scores:
                    scores[metric] = {}
                scores[metric][tiou] = score * 100

        # average across tious
        for metric, scores_tiou in scores.items():
            scores[metric] = round(np.mean(list(scores_tiou.values())), 2)
        return scores

    def get_metrics_vid(self, vid_id, tious=(0.3, 0.5, 0.7, 0.9), scorers=("CIDEr",)):
        scores = {}
        for tiou in tious:
            scores_tiou = self.evaluate_tiou_vid(vid_id, tiou, scorers)
            for metric, score in scores_tiou.items():
                if metric not in scores:
                    scores[metric] = {}
                scores[metric][tiou] = score * 100
        return scores

    def evaluate_tiou_vid(self, vid_id, tiou, scorers=("CIDEr",)):
        for scorer in scorers:
            assert scorer in self.scorers, (
                f"Scorer {scorer} is not available. Available scorers: {list(self.scorers.keys())}"
            )

        refs_iou, preds_iou = build_iou_matches_vid(
            self.vid2refs[vid_id], self.vid2preds[vid_id], tiou
        )

        output = {}
        for method in scorers:
            scorer = self.scorers[method]
            score, _ = scorer.compute_score(refs_iou, preds_iou)
            output[method] = score
        return output

    def evaluate_tiou(self, tiou, scorers=("CIDEr",)):
        """
        Evaluate video captions based on a single tIoU threshold.

        Parameters:
        - tiou: Temporal Intersection over Union threshold for matching predictions to ground truths.
        - scorers: Tuple of scoring metrics to use for this evaluation.

        Returns:
        - Dictionary with scores for each metric.
        """

        # Verify that all requested scorers are available
        for scorer in scorers:
            assert scorer in self.scorers, (
                f"Scorer {scorer} is not available. Available scorers: {list(self.scorers.keys())}"
            )

        # Create dictionaries of matched references and predictions based on tIoU
        vid2refs_iou, vid2preds_iou = build_iou_matches(
            self.vid2refs, self.vid2preds, tiou
        )

        video_ids = list(vid2refs_iou.keys())

        # Store scores for each scorer
        output = {}
        for method in scorers:
            scorer = self.scorers[method]

            # For each video, take all the valid pairs (based from tIoU) and compute the score
            all_scores = {}

            for vid_id in video_ids:
                assert vid_id in vid2refs_iou and vid_id in vid2preds_iou

                score, _ = scorer.compute_score(
                    vid2refs_iou[vid_id], vid2preds_iou[vid_id]
                )
                all_scores[vid_id] = score

                # Compute the mean score across all videos
                if method == "Bleu":
                    scores = np.mean(list(all_scores.values()), axis=0)
                    for m in range(scorer._n):
                        output[f"{method}_{m + 1}"] = scores[m]
                else:
                    output[method] = np.mean(list(all_scores.values()))
        return output


def build_iou_matches_vid(vid_refs, vid_preds, tiou=0.5):
    """
    Computes IoU matches for a single video's references and predictions.

    Parameters:
    - vid_refs: Dictionary of reference ground truth captions with timestamps for one video.
    - vid_preds: Dictionary of predicted captions with timestamps for one video.
    - tiou: Temporal Intersection over Union threshold.

    Returns:
    - Two dictionaries containing matched references and predictions based on tIoU.
    """
    refs_iou = {}
    preds_iou = {}

    for vid_pred_segment, vid_pred_label in vid_preds.items():
        match_found = False
        for vid_ref_segment, vid_ref_label in vid_refs.items():
            # Check if the predicted segment matches the ground truth segment
            if iou(vid_pred_segment, vid_ref_segment) >= tiou:
                refs_iou[(vid_pred_segment, vid_ref_segment)] = [vid_ref_label]
                preds_iou[(vid_pred_segment, vid_ref_segment)] = [vid_pred_label]
                match_found = True

        if not match_found:
            # No match found, assign random_string in refs_iou
            refs_iou[(vid_pred_segment)] = [random_string(10)]
            preds_iou[(vid_pred_segment)] = [vid_pred_label]

    return refs_iou, preds_iou


def build_iou_matches(vid2refs, vid2preds, tiou=0.5):
    """
    Matches ground truth captions to predicted captions based on tIoU threshold.

    Parameters:
    - vid2refs: Dictionary of reference ground truth captions with timestamps.
    - vid2preds: Dictionary of predicted captions with timestamps.
    - tiou: Temporal Intersection over Union threshold.

    Returns:
    - Two dictionaries containing matched references and predictions based on tIoU.
    """
    vid2refs_iou = {}
    vid2preds_iou = {}

    for vid_id, vid_refs in vid2refs.items():
        if vid_id not in vid2preds:
            continue

        vid_preds = vid2preds[vid_id]
        vid2refs_iou[vid_id], vid2preds_iou[vid_id] = build_iou_matches_vid(
            vid_refs, vid_preds, tiou
        )

    return vid2refs_iou, vid2preds_iou


def get_n_captions(exp_dir, subset_ids):
    cfg = load_config(exp_dir)
    if "captions_dir" not in cfg["data"]["prompter"]["chapters"]:
        return 0.0
    captions_dir = Path("../../") / cfg["data"]["prompter"]["chapters"]["captions_dir"]
    assert captions_dir.exists(), f"Captions directory {captions_dir} does not exist"

    n_captions = []
    for vid_id in subset_ids:
        caption_pth = captions_dir / f"{vid_id[:2]}" / f"{vid_id}.json"
        n_captions.append(len(openf(caption_pth)))
    return np.mean(n_captions)


def get_n_captions_min(exp_dir, subset_ids, chp):
    cfg = load_config(exp_dir)
    if "captions_dir" not in cfg["data"]["prompter"]["chapters"]:
        return 0.0
    captions_dir = Path("../../") / cfg["data"]["prompter"]["chapters"]["captions_dir"]
    captions_dir.exists()

    n_captions = []
    for vid_id in subset_ids:
        caption_pth = captions_dir / f"{vid_id[:2]}" / f"{vid_id}.json"
        vid_n_captions = len(openf(caption_pth))
        vid_duration = chp.get_duration(vid_id)
        n_captions.append(vid_n_captions / vid_duration * 60)

    n_captions = np.mean(n_captions)
    n_captions = format_number_with_z(n_captions)
    return n_captions


def test_build_iou_matches():
    vid2refs = {
        "vid1": {
            (0, 10): "This is a ground truth sentence for segment (0, 10).",
            (15, 25): "Another ground truth sentence for segment (15, 25).",
        }
    }
    vid2preds = {
        "vid1": {
            (0, 5): "Predicted sentence for segment (0, 5).",
            (5, 12): "Predicted sentence for segment (5, 12).",
            (12, 25): "Predicted sentence for segment (12, 25).",
        }
    }

    vid2refs_iou, vid2preds_iou = build_iou_matches(vid2refs, vid2preds, tiou=0.48)

    for vid_refs in vid2refs_iou["vid1"]:
        print(vid_refs)
        print("\t", vid2refs_iou["vid1"][vid_refs])
        print("\t", vid2preds_iou["vid1"][vid_refs])
        print("\t", vid2preds_iou["vid1"][vid_refs])


def test_compute_metrics_captions():
    import cProfile
    import pstats

    from src.data.chapters import Chapters
    from tools.results.metrics import MetricsExperiment

    profiler = cProfile.Profile()
    profiler.enable()

    subset = "s100_val"
    chp = Chapters(vidc_dir="dataset/", subset=subset)
    exp_dir = "outputs/chapterize/Meta-Llama-3.1-8B-Instruct/asr/titleFalse_summaryFalse/s1k_train/test"

    metrics = MetricsExperiment(chp, exp_dir=exp_dir)
    results_captions = metrics.compute_metrics_captions(
        subset=subset, metrics=("CIDEr", "METEOR")
    )
    print(results_captions)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats()


if __name__ == "__main__":
    # test_compute_metrics_captions()
    test_build_iou_matches()
