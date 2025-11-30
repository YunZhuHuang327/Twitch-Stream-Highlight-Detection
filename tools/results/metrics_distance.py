# Code based on antoyang/VidChapters

import numpy as np


def add_zero(timestamps: list):
    return [0] + timestamps if timestamps[0] != 0 else timestamps


def remove_zero(timestamps: list):
    return [time for time in timestamps if time != 0]


def clean_tiemstamps(timestamps, zero_handling="remove"):
    if zero_handling == "remove":
        return remove_zero(timestamps)
    elif zero_handling == "add":
        return add_zero(timestamps)
    else:
        return timestamps


def clean_segment(segment, zero_handling="add"):
    if zero_handling == "add" and segment[0][0] != 0.0:
        segment.insert(0, [0.0, segment[0][0]])
    elif zero_handling == "remove" and segment[0][0] == 0.0:
        segment.pop(0)
    return segment


def iou(interval_1, interval_2):
    start_i, end_i = interval_1[0], interval_1[1]
    start, end = interval_2[0], interval_2[1]
    intersection = max(0, min(end, end_i) - max(start, start_i))
    union = min(max(end, end_i) - min(start, start_i), end - start + end_i - start_i)
    iou = float(intersection) / (union + 1e-8)
    return iou


def evaluate_vid_overlap(refs, preds, tiou, zero_handling="add"):
    """
    refs are the video ground truths
    preds are the video predictions
    """
    refs = clean_segment(refs, zero_handling)
    preds = clean_segment(preds, zero_handling)

    ref_set_covered = set()
    pred_set_covered = set()
    for pred_i, pred in enumerate(preds):
        for ref_i, ref in enumerate(refs):
            if iou(pred, ref) > tiou:
                ref_set_covered.add(ref_i)
                pred_set_covered.add(pred_i)

    precision = float(len(pred_set_covered)) / max(len(preds), 1)
    recall = float(len(ref_set_covered)) / len(refs)

    return precision, recall


def evaluate_overlap(ground_truths, predictions, tiou):
    # Recall is the percentage of ground truth that is covered by the predictions
    # Precision is the percentage of predictions that are valid
    recall = []
    precision = []
    for vid_id in ground_truths:
        if vid_id not in predictions:  # missing video
            continue
        refs = ground_truths[vid_id]
        preds = predictions[vid_id]
        vid_precision, vid_recall = evaluate_vid_overlap(refs, preds, tiou)
        precision.append(vid_precision)
        recall.append(vid_recall)

    precision = round(sum(precision) * 100 / len(precision), 2)
    recall = round(sum(recall) * 100 / len(recall), 2)
    return precision, recall


def evaluate_overlaps(ground_truths, predictions, tious):
    results = {}
    for tiou in tious:
        precision, recall = evaluate_overlap(ground_truths, predictions, tiou)
        results[f"Precision@{tiou}"] = precision
        results[f"Recall@{tiou}"] = recall
        results[f"F1@{tiou}"] = (
            2 * recall * precision / (recall + precision) if recall + precision else 0.0
        )

    results["Precision"] = np.array(
        [results[f"Precision@{tiou}"] for tiou in tious[:4]]
    ).mean()
    results["Recall"] = np.array(
        [results[f"Recall@{tiou}"] for tiou in tious[:4]]
    ).mean()
    results["F1"] = np.array([results[f"F1@{tiou}"] for tiou in tious[:4]]).mean()

    return results


def average_overlaps(ground_truths, predictions, zero_handling="add"):
    overlaps = []
    for vid_id in ground_truths:
        if vid_id not in predictions:  # missing video
            continue
        refs = ground_truths[vid_id]
        preds = predictions[vid_id]

        refs = clean_segment(refs, zero_handling)
        preds = clean_segment(preds, zero_handling)

        if len(refs) != len(preds):
            continue

        vid_tiou = 0
        for ref, pred in zip(refs, preds):
            vid_tiou += iou(pred, ref) * 100

        overlaps.append(vid_tiou / len(refs))
    return overlaps


def mean_over_frames(ground_truths, predictions, zero_handling="add"):
    overlaps = []
    for vid_id in ground_truths:
        if vid_id not in predictions:  # missing video
            continue
        refs = ground_truths[vid_id]
        preds = predictions[vid_id]

        refs = clean_segment(refs, zero_handling)
        preds = clean_segment(preds, zero_handling)

        if len(refs) != len(preds):
            continue

        vid_tiou = 0
        for ref, pred in zip(refs, preds):
            start_r, end_r = ref
            start_p, end_p = pred

            start = max(start_r, start_p)
            end = min(end_r, end_p)

            if start < end:
                vid_tiou += end - start
        vid_duration = refs[-1][1]
        overlaps.append(vid_tiou * 100 / vid_duration)
    return overlaps


def evaluate_vid_distance(refs, preds, dist, zero_handling="remove"):
    # If we have start and end time of each segment, we take only the first one
    refs = [it[0] if isinstance(it, list) and len(it) > 0 else it for it in refs]
    preds = [it[0] if isinstance(it, list) and len(it) > 0 else it for it in preds]

    refs = clean_tiemstamps(refs, zero_handling=zero_handling)
    preds = clean_tiemstamps(preds, zero_handling=zero_handling)

    ref_set_covered = set()
    pred_set_covered = set()
    for pred_i, pred in enumerate(preds):
        for ref_i, ref in enumerate(refs):
            if abs(pred - ref) < dist:
                ref_set_covered.add(ref_i)
                pred_set_covered.add(pred_i)

    precision = float(len(pred_set_covered)) / max(len(preds), 1)
    recall = float(len(ref_set_covered)) / len(refs)

    return precision, recall


def average_distances(ground_truths, predictions, zero_handling="remove"):
    id2distance = {}
    for vid_id in ground_truths:
        if vid_id not in predictions:  # missing video
            continue

        refs = ground_truths[vid_id]
        preds = predictions[vid_id]

        # If we have start and end time of each segment, we take only the first one
        refs = [it[0] if isinstance(it, list) and len(it) > 0 else it for it in refs]
        preds = [it[0] if isinstance(it, list) and len(it) > 0 else it for it in preds]

        refs = clean_tiemstamps(refs, zero_handling=zero_handling)
        preds = clean_tiemstamps(preds, zero_handling=zero_handling)

        if len(refs) != len(preds):
            continue

        refs = clean_tiemstamps(refs, zero_handling=zero_handling)
        preds = clean_tiemstamps(preds, zero_handling=zero_handling)

        vid_dist = 0
        for ref, pred in zip(refs, preds):
            vid_dist += abs(ref - pred)

        id2distance[vid_id] = vid_dist / len(refs)

    return id2distance


def evaluate_distance(ground_truths, predictions, dist, zero_handling="remove"):
    # Recall is the percentage of ground truth that is covered by the predictions
    # Precision is the percentage of predictions that are valid
    recall = []
    precision = []
    for vid_id in ground_truths:
        if vid_id not in predictions:  # missing video
            continue

        refs = ground_truths[vid_id]
        preds = predictions[vid_id]

        vid_precision, vid_recall = evaluate_vid_distance(
            refs, preds, dist, zero_handling=zero_handling
        )
        precision.append(vid_precision)
        recall.append(vid_recall)

    precision = round(sum(precision) * 100 / len(precision), 2)
    recall = round(sum(recall) * 100 / len(recall), 2)
    return precision, recall


def evaluate_distances(ground_truths, predictions, distances):
    results = {}
    for dist in distances:
        precision, recall = evaluate_distance(ground_truths, predictions, dist)
        results[f"Precision@{dist}s"] = precision
        results[f"Recall@{dist}s"] = recall
        results[f"F1@{dist}s"] = (
            2 * recall * precision / (recall + precision) if recall + precision else 0.0
        )

    results["Precision"] = np.array(
        [results[f"Precision@{dist}s"] for dist in distances]
    ).mean()
    results["Recall"] = np.array(
        [results[f"Recall@{dist}s"] for dist in distances]
    ).mean()
    results["F1"] = np.array([results[f"F1@{dist}s"] for dist in distances]).mean()
    return results


def evaluate_vid_chapter_count(refs, preds, zero_handling="add"):
    """
    refs are the video ground truths
    preds are the video predictions
    """
    refs = clean_segment(refs, zero_handling)
    preds = clean_segment(preds, zero_handling)

    return len(refs) - len(preds)


def evaluate_chapter_count(ground_truths, predictions):
    """
    Calculate the absolute error for each video and the mean absolute error across all videos.
    Args:
        ground_truths (dict): Ground truth segments for each video ID.
        predictions (dict): Predicted segments for each video ID.

    Returns:
        dict: Contains absolute errors per video and mean absolute error for all videos.
    """
    errors = []
    for vid_id in ground_truths:
        if vid_id not in predictions:  # missing video
            continue
        refs = ground_truths[vid_id]
        preds = predictions[vid_id]
        errors.append(evaluate_vid_chapter_count(refs, preds))
    if not errors:
        return 0, [0, 0, 0]

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

    mae = sum(abs(errors)) / len(errors)
    return mae, ueo


def temporal_eval(
    ground_truths,
    predictions,
    tious=None,
    distances=(1, 3, 5, 10, 30, 60),
):
    """
    predictions is a dict with key vid_id and value the proposals
    temporal IoU overlap (overlap) and Distance-based (distance).
    """
    if len(tious) == 0:
        raise IOError("Please input a valid tIoU.")

    scores = {}
    score_navs = evaluate_distances(ground_truths, predictions, distances)
    score_dets = evaluate_overlaps(ground_truths, predictions, tious)
    scores.update(score_navs)
    scores.update(score_dets)
    return scores


class TemporalEval:
    def __init__(self, chp, split="test") -> None:
        """
        chp is Chapters class
        """
        self.chp = chp
        if split == "test":
            self.gts = {id: chp.get_gt_segments(id) for id in chp.test_ids}
        elif split == "val":
            self.gts = {id: chp.get_gt_segments(id) for id in chp.val_ids}
        elif split == "train":
            self.gts = {id: chp.get_gt_segments(id) for id in chp.train_ids}
        else:
            self.gts = {id: chp.get_gt_segments(id) for id in chp.ids}

    def evaluate_distances(self, predictions, distances=(1, 3, 5, 10, 30, 60)):
        return evaluate_distances(self.gts, predictions, distances)

    def average_distances(self, predictions):
        return average_distances(self.gts, predictions)

    def evaluate_overlaps(self, predictions, tious=(0.3, 0.5, 0.7, 0.9)):
        preds_segments = {}
        for vid_id, vid_preds in predictions.items():
            assert not isinstance(vid_preds[0], list), "vid_preds[0] is a list"
            preds_segments[vid_id] = self.chp.get_pred_segments(vid_id, vid_preds)
        return evaluate_overlaps(self.gts, preds_segments, tious)

    def average_overlaps(self, predictions):
        preds_segments = {}
        for vid_id, vid_preds in predictions.items():
            assert not isinstance(vid_preds[0], list), "vid_preds[0] is a list"
            preds_segments[vid_id] = self.chp.get_pred_segments(vid_id, vid_preds)
        return average_overlaps(self.gts, preds_segments)

    def evaluate_chapter_count(self, predictions):
        preds_segments = {}
        for vid_id, vid_preds in predictions.items():
            assert not isinstance(vid_preds[0], list), "vid_preds[0] is a list"
            preds_segments[vid_id] = self.chp.get_pred_segments(vid_id, vid_preds)

        return evaluate_chapter_count(self.gts, preds_segments)

    def print_distances(self, predictions, distances=(1, 3, 5, 10, 30, 60)):
        print("Distance:")
        id2distances = self.average_distances(predictions)
        avg_distances = list(id2distances.values())
        print(
            f"    Average Distances: {sum(avg_distances)/len(avg_distances):.1f}s ({len(avg_distances) / len(predictions) * 100:.1f}% videos)"
        )
        # do the same but in terms of percentage of the video
        avg_distances = [
            dist * 100 / self.chp.get_duration(vid_id)
            for vid_id, dist in id2distances.items()
        ]
        print(
            f"    Average Distances: {sum(avg_distances)/len(avg_distances):.1f}% ({len(avg_distances) / len(predictions) * 100:.1f}% videos)"
        )
        score_navs = self.evaluate_distances(predictions, distances=distances)
        for k in [
            "Recall@3s",
            "Recall@5s",
            "Precision@3s",
            "Precision@5s",
        ]:
            print(f"    {k}: {score_navs[k]:.0f}")
        for k in [
            "F1@3s",
            "F1@5s",
        ]:
            print(f"    {k}: {score_navs[k]:.1f}")

    def print_overlaps(self, predictions, tious=(0.3, 0.5, 0.7, 0.9)):
        preds_segments = {}
        for vid_id, vid_preds in predictions.items():
            assert not isinstance(vid_preds[0], list), "vid_preds[0] is a list"
            preds_segments[vid_id] = self.chp.get_pred_segments(vid_id, vid_preds)

        print("Overlap:")
        avg_tiou = average_overlaps(self.gts, preds_segments)
        print(
            f"    Average Overlap: {sum(avg_tiou)/len(avg_tiou):.1f}% ({len(avg_tiou) / len(predictions) * 100:.1f}% videos)"
        )
        mof = mean_over_frames(self.gts, preds_segments)
        print(
            f"    MoF: {sum(mof)/len(mof):.1f}% ({len(mof) / len(predictions) * 100:.1f}% videos)"
        )
        score_dets = evaluate_overlaps(self.gts, preds_segments, tious)
        for k in [
            "Recall@0.7",
            "Recall@0.5",
            "Precision@0.7",
            "Precision@0.5",
        ]:
            print(f"    {k}: {score_dets[k]:.0f}")
        for k in [
            "F1@0.7",
            "F1@0.5",
        ]:
            print(f"    {k}: {score_dets[k]:.1f}")

    def print_chapter_count(self, predictions):
        """
        Prints the chapter count accuracy metrics for predictions compared to ground truths.
        Args:
            predictions (dict): Predicted segments for each video ID.
        """
        mae, ueo = self.evaluate_chapter_count(predictions)
        print("Chapter Count Accuracy:")
        print(f"    Mean Absolute Error: {mae:.2f}")
        print(f"    Under: {ueo[0]:.2f}%")
        print(f"    Equal: {ueo[1]:.2f}%")
        print(f"    Over: {ueo[2]:.2f}%")

    def print_results(self, predictions, distances=(3, 5), tious=(0.5, 0.7)):
        self.print_chapter_count(predictions)
        self.print_distances(predictions, distances=distances)
        self.print_overlaps(predictions, tious=tious)
        # self.print_similarities(predictions)
