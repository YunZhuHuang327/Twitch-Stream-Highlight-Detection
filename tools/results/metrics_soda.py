# Code adapted from https://github.com/antoyang/VidChapters/blob/main/dvc_eval/SODA/soda.py
# --------------------------------------------------------
# VidChapters-7M: Video Chapters at Scale
# Copyright (c) 2017 Antoine Yang
# Licensed under The MIT License [see LICENSE for details]
# Written by Antoine Yang
# --------------------------------------------------------

import numpy as np
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from tqdm import tqdm

from tools.results.metrics_caption import preprocess_vid2chapters
from tools.results.metrics_overlap import iou


class SODA:
    def __init__(
        self,
        vid2preds,
        vid2refs,
        soda_type="iou_scorer",
        scorer="Meteor",
        tious=None,
        verbose=False,
    ):
        """
        Args:
            vid2preds: Dictionary with key the video id and value a chapter dictionary with key a segment tuple and value the sentence of the segment
            vid2refs: Dictionary with key the video id and value a chapter dictionary with key a segment tuple and value the sentence of the segment
        """
        vid2preds = preprocess_vid2chapters(vid2preds)
        vid2refs = preprocess_vid2chapters(vid2refs)
        vid2preds = {
            vid_id: {
                "timestamps": list(vid_preds.keys()),
                "sentences": [[s] for s in vid_preds.values()],
            }
            for vid_id, vid_preds in vid2preds.items()
        }
        vid2refs = {
            vid_id: {
                "timestamps": list(vid_refs.keys()),
                "sentences": [[s] for s in vid_refs.values()],
            }
            for vid_id, vid_refs in vid2refs.items()
        }

        # Check for missing videos in predictions and remove them from references
        missing_vids = set(vid2refs.keys()) - set(vid2preds.keys())
        if len(missing_vids) > 0:
            if verbose:
                print(
                    f"Number of videos missing from predictions: {len(missing_vids)} "
                    f"({len(missing_vids) / len(vid2refs):.2%})."
                )
            for vid in missing_vids:
                del vid2refs[vid]

        self.vid2preds = vid2preds
        self.vid2refs = vid2refs
        self.gt_vids = list(vid2refs.keys())
        self.soda_type = soda_type
        self.tious = [0.0] if tious is None else tious
        if scorer not in ["Meteor", "Cider"]:
            raise ValueError(f"scorer must be either 'Meteor' or 'Cider', got {scorer}")
        self.scorer = Meteor() if scorer == "Meteor" else Cider()
        self.scorer_name = scorer
        self.verbose = verbose

        if (
            soda_type == "multi_threshold"
        ):  # averaging F-measure scores with IoU threshold = 0.9, 0.7, 0.5, 0.3
            self.soda_func = self.soda_a
        elif soda_type == "zero_threshold":
            self.soda_func = self.soda_b  # F-measure, where IoU threshold is set to 0
        elif soda_type == "iou_scorer":
            self.soda_func = self.soda_c  # F-measure, utilizing the IoU x METEOR score
        elif soda_type == "iou_only":
            self.soda_func = self.soda_d  # F-measure of IoU score
            self.scorer = None
        else:
            raise NotImplementedError(f"SODA type {soda_type} not implemented")

    @staticmethod
    def calc_iou_matrix(pred_segments, ref_segments):
        return np.array(
            [[iou(pred, ct) for pred in pred_segments] for ct in ref_segments]
        )

    def calc_score_matrix(self, pred_sentences, ref_sentences):
        # Reformat to fit the input of pycocoevalcap scorers.
        res = dict(enumerate(pred_sentences))
        gts = [
            {index: g for index in range(len(pred_sentences))} for g in ref_sentences
        ]
        return np.array([self.scorer.compute_score(res, gt)[1] for gt in gts])

    def get_metrics(
        self,
        metrics=("F1"),
    ):
        if self.verbose:
            print(f"\n| Running SODA {self.soda_type}.")
        tious = self.tious
        p_best = [[] for _ in range(len(tious))]
        r_best = [[] for _ in range(len(tious))]
        f_best = [[] for _ in range(len(tious))]
        n_pred = []
        for vid in tqdm(self.gt_vids, disable=not self.verbose):
            _p = [[] for _ in range(len(tious))]
            _r = [[] for _ in range(len(tious))]
            _f = [[] for _ in range(len(tious))]
            pred = self.vid2preds[vid]
            n_pred.append(len(pred["sentences"]))
            # empty pred
            if not pred["sentences"]:
                for i, _ in enumerate(tious):
                    p_best[i].append(0)
                    r_best[i].append(0)
                    f_best[i].append(0)
                continue

            vid_refs = self.vid2refs[vid]
            # create matrix
            _iou = self.calc_iou_matrix(pred["timestamps"], vid_refs["timestamps"])
            scores = self.calc_score_matrix(pred["sentences"], vid_refs["sentences"])
            for i, tiou in enumerate(tious):
                iou = np.copy(_iou)
                iou[iou < tiou] = 0.0
                try:
                    max_score, _ = self.soda_func(iou, scores)
                except:  # RecursionError  # noqa: E722
                    max_score, _ = 0.0, None
                (n_g, n_p) = iou.shape
                p = max_score / n_p
                r = max_score / n_g
                _p[i].append(p)
                _r[i].append(r)
                _f[i].append(2 * p * r / (p + r) if p + r > 0 else 0)
            best_idx = np.argmax(_f, axis=1)
            for i, _ in enumerate(tious):
                p_best[i].append(_p[i][best_idx[i]])
                r_best[i].append(_r[i][best_idx[i]])
                f_best[i].append(_f[i][best_idx[i]])
        precision = np.mean(p_best, axis=1)
        recall = np.mean(r_best, axis=1)
        f1 = np.mean(f_best, axis=1)
        # average scores across all the tIoUs
        if self.verbose:
            print(f"avg. outputs: {np.mean(n_pred)}")
            for i, tiou in enumerate(tious):
                partial_result = {self.scorer_name: [precision[i], recall[i], f1[i]]}
                print_score(partial_result, description=f"tIoU: {tiou}")

        results = {}
        if "P" in metrics:
            results["P"] = round(np.mean(precision) * 100, 2)
        if "R" in metrics:
            results["R"] = round(np.mean(recall) * 100, 2)
        if "F1" in metrics:
            results["F1"] = round(np.mean(f1) * 100, 2)
        return results

    def soda_a(self, iou, scores):
        _, pairs = self.chased_dp_assignment(iou)
        r, c = (*zip(*pairs),)
        max_score = np.sum(scores[r, c])
        return max_score, pairs

    def soda_b(self, iou, scores):
        # same as soda_a
        _, pairs = self.chased_dp_assignment(iou)
        r, c = (*zip(*pairs),)
        max_score = np.sum(scores[r, c])
        return max_score, pairs

    def soda_c(self, iou, scores):
        max_score, pairs = self.chased_dp_assignment(iou * scores)
        return max_score, pairs

    def soda_d(self, iou, scores):
        max_score, pairs = self.chased_dp_assignment(iou)
        return max_score, pairs

    def chased_dp_assignment(self, scores):
        """
        Run dp matching
        Recurrence:
            dp[i,j] =
                max(dp[i-1,j], dp[i-1,j-1] + scores[i,j], dp[i,j-1])
        """
        M, N = scores.shape
        dp = -np.ones((M, N))
        path = np.zeros((M, N))

        def transition(i, j):
            if dp[i, j] >= 0:
                return dp[i, j]
            elif i == 0 and j == 0:
                state = [-1, -1, scores[i, j]]
            elif i == 0:
                state = [-1, transition(i, j - 1), scores[i, j]]
            elif j == 0:
                state = [transition(i - 1, j), -1, scores[i, j]]
            else:
                state = [
                    transition(i - 1, j),
                    transition(i, j - 1),
                    transition(i - 1, j - 1) + scores[i, j],
                ]
            dp[i, j] = np.max(state)
            path[i, j] = np.argmax(state)
            return dp[i, j]

        def get_pairs(i, j):
            p = np.where(path[i][: j + 1] == 2)[0]
            if i != 0 and len(p) == 0:
                return get_pairs(i - 1, j)
            elif i == 0 or p[-1] == 0:
                return [(i, p[-1])]
            else:
                return get_pairs(i - 1, p[-1] - 1) + [(i, p[-1])]

        N, M = scores.shape
        max_score = transition(N - 1, M - 1)
        pairs = get_pairs(N - 1, M - 1)
        return max_score, pairs


def print_score(result, description="SODA result"):
    prf = ["precision", "recall", "f1_score"]
    print("-" * 80)
    print(description)
    print("-" * 80)
    for scorer_name, score in result.items():
        print(f"| scorer: {scorer_name}")
        for k, v in zip(prf, score):
            print(f"\t{k}: {v * 100:2.2f}%")
