"""
Evaluate extracted highlights against ground truth labels
Calculate precision, recall, F1-score, and temporal IoU
"""
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def parse_timecode(timecode: str, fps: float = 30.0) -> float:
    """
    Parse timecode to seconds

    Supports formats:
    - HH:MM:SS:FF (with frame number)
    - HH:MM:SS
    """
    timecode = timecode.strip()

    # Try HH:MM:SS:FF format first
    if timecode.count(':') == 3:
        h, m, s, f = timecode.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s) + int(f) / fps

    # Try HH:MM:SS format
    elif timecode.count(':') == 2:
        h, m, s = timecode.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    # Try MM:SS format
    elif timecode.count(':') == 1:
        m, s = timecode.split(':')
        return int(m) * 60 + float(s)

    else:
        return float(timecode)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_ground_truth(csv_path: str, fps: float = 30.0) -> List[Dict]:
    """
    Load ground truth highlights from CSV

    Returns:
        List of segments with start_time, end_time in seconds
    """
    segments = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_time = parse_timecode(row['start_time'], fps)
            end_time = parse_timecode(row['end_time'], fps)

            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'start_timestamp': format_timestamp(start_time),
                'end_timestamp': format_timestamp(end_time),
                'duration': end_time - start_time
            })

    return segments


def load_predicted_highlights(json_path: str) -> List[Dict]:
    """
    Load predicted highlights from JSON

    Returns:
        List of segments with start_time, end_time in seconds
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure we have start_time and end_time in seconds
    for segment in data:
        if 'start_time' not in segment:
            segment['start_time'] = parse_timecode(segment['start_timestamp'])
        if 'end_time' not in segment:
            segment['end_time'] = parse_timecode(segment['end_timestamp'])

    return data


def calculate_iou(seg1: Dict, seg2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two segments

    Args:
        seg1, seg2: Dicts with 'start_time' and 'end_time' in seconds

    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection
    intersection_start = max(seg1['start_time'], seg2['start_time'])
    intersection_end = min(seg1['end_time'], seg2['end_time'])
    intersection = max(0, intersection_end - intersection_start)

    # Calculate union
    union_start = min(seg1['start_time'], seg2['start_time'])
    union_end = max(seg1['end_time'], seg2['end_time'])
    union = union_end - union_start

    if union == 0:
        return 0.0

    return intersection / union


def calculate_overlap(seg1: Dict, seg2: Dict) -> float:
    """
    Calculate overlap (intersection) between two segments

    Returns:
        Overlap duration in seconds
    """
    intersection_start = max(seg1['start_time'], seg2['start_time'])
    intersection_end = min(seg1['end_time'], seg2['end_time'])
    return max(0, intersection_end - intersection_start)


def evaluate_highlights(
    ground_truth: List[Dict],
    predicted: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Evaluate predicted highlights against ground truth

    Args:
        ground_truth: List of GT segments
        predicted: List of predicted segments
        iou_threshold: Minimum IoU to consider a match

    Returns:
        Dict with evaluation metrics
    """
    n_gt = len(ground_truth)
    n_pred = len(predicted)

    # Track which GT segments were matched
    gt_matched = [False] * n_gt
    pred_matched = [False] * n_pred

    # Calculate IoU matrix
    iou_matrix = []
    for i, gt_seg in enumerate(ground_truth):
        row = []
        for j, pred_seg in enumerate(predicted):
            iou = calculate_iou(gt_seg, pred_seg)
            row.append(iou)
        iou_matrix.append(row)

    # Match predictions to GT (greedy matching by highest IoU)
    matches = []
    while True:
        best_iou = 0
        best_match = None

        for i in range(n_gt):
            if gt_matched[i]:
                continue
            for j in range(n_pred):
                if pred_matched[j]:
                    continue
                if iou_matrix[i][j] > best_iou and iou_matrix[i][j] >= iou_threshold:
                    best_iou = iou_matrix[i][j]
                    best_match = (i, j)

        if best_match is None:
            break

        i, j = best_match
        gt_matched[i] = True
        pred_matched[j] = True
        matches.append({
            'gt_idx': i,
            'pred_idx': j,
            'iou': iou_matrix[i][j],
            'gt_segment': ground_truth[i],
            'pred_segment': predicted[j]
        })

    # Calculate metrics
    true_positives = len(matches)
    false_positives = n_pred - true_positives
    false_negatives = n_gt - true_positives

    precision = true_positives / n_pred if n_pred > 0 else 0
    recall = true_positives / n_gt if n_gt > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate average IoU for matched segments
    avg_iou = sum(m['iou'] for m in matches) / len(matches) if matches else 0

    # Calculate coverage (how much of GT time is covered by predictions)
    total_gt_time = sum(seg['duration'] for seg in ground_truth)
    covered_gt_time = 0
    for gt_seg in ground_truth:
        max_overlap = 0
        for pred_seg in predicted:
            overlap = calculate_overlap(gt_seg, pred_seg)
            max_overlap = max(max_overlap, overlap)
        covered_gt_time += max_overlap

    coverage = covered_gt_time / total_gt_time if total_gt_time > 0 else 0

    # Calculate temporal precision (how much of predicted time overlaps with GT)
    total_pred_time = sum(seg['duration'] for seg in predicted)
    overlapping_pred_time = 0
    for pred_seg in predicted:
        max_overlap = 0
        for gt_seg in ground_truth:
            overlap = calculate_overlap(pred_seg, gt_seg)
            max_overlap = max(max_overlap, overlap)
        overlapping_pred_time += max_overlap

    temporal_precision = overlapping_pred_time / total_pred_time if total_pred_time > 0 else 0

    return {
        'n_ground_truth': n_gt,
        'n_predicted': n_pred,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_iou': avg_iou,
        'coverage': coverage,
        'temporal_precision': temporal_precision,
        'matches': matches,
        'unmatched_gt': [ground_truth[i] for i in range(n_gt) if not gt_matched[i]],
        'unmatched_pred': [predicted[i] for i in range(n_pred) if not pred_matched[i]],
        'total_gt_time': total_gt_time,
        'total_pred_time': total_pred_time,
        'covered_gt_time': covered_gt_time,
        'overlapping_pred_time': overlapping_pred_time
    }


def save_evaluation_report(
    results: Dict,
    output_path: Path,
    iou_threshold: float
):
    """Save detailed evaluation report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("HIGHLIGHT EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"IoU Threshold: {iou_threshold}\n\n")

        f.write("Dataset Statistics:\n")
        f.write(f"  Ground Truth Segments: {results['n_ground_truth']}\n")
        f.write(f"  Predicted Segments: {results['n_predicted']}\n")
        f.write(f"  Ground Truth Total Time: {format_timestamp(results['total_gt_time'])}\n")
        f.write(f"  Predicted Total Time: {format_timestamp(results['total_pred_time'])}\n\n")

        f.write("Segment-Level Metrics:\n")
        f.write(f"  True Positives: {results['true_positives']}\n")
        f.write(f"  False Positives: {results['false_positives']}\n")
        f.write(f"  False Negatives: {results['false_negatives']}\n")
        f.write(f"  Precision: {results['precision']:.3f} ({results['precision']*100:.1f}%)\n")
        f.write(f"  Recall: {results['recall']:.3f} ({results['recall']*100:.1f}%)\n")
        f.write(f"  F1-Score: {results['f1_score']:.3f} ({results['f1_score']*100:.1f}%)\n")
        f.write(f"  Average IoU: {results['avg_iou']:.3f}\n\n")

        f.write("Temporal Metrics:\n")
        f.write(f"  Coverage (GT time covered): {results['coverage']:.3f} ({results['coverage']*100:.1f}%)\n")
        f.write(f"  Temporal Precision: {results['temporal_precision']:.3f} ({results['temporal_precision']*100:.1f}%)\n")
        f.write(f"  Covered GT Time: {format_timestamp(results['covered_gt_time'])}\n")
        f.write(f"  Overlapping Pred Time: {format_timestamp(results['overlapping_pred_time'])}\n\n")

        # Matched segments
        f.write("="*70 + "\n")
        f.write(f"MATCHED SEGMENTS ({len(results['matches'])})\n")
        f.write("="*70 + "\n\n")

        for i, match in enumerate(results['matches'], 1):
            gt = match['gt_segment']
            pred = match['pred_segment']
            f.write(f"{i}. IoU: {match['iou']:.3f}\n")
            f.write(f"   GT:   [{gt['start_timestamp']} - {gt['end_timestamp']}] "
                   f"Duration: {format_timestamp(gt['duration'])}\n")
            f.write(f"   Pred: [{pred['start_timestamp']} - {pred['end_timestamp']}] "
                   f"Duration: {format_timestamp(pred['duration'])}\n")
            if 'max_score' in pred:
                f.write(f"   Score: {pred.get('max_score', 'N/A')}/10\n")
            f.write("\n")

        # Unmatched GT (missed highlights)
        f.write("="*70 + "\n")
        f.write(f"MISSED HIGHLIGHTS (False Negatives: {len(results['unmatched_gt'])})\n")
        f.write("="*70 + "\n\n")

        if results['unmatched_gt']:
            for i, seg in enumerate(results['unmatched_gt'], 1):
                f.write(f"{i}. [{seg['start_timestamp']} - {seg['end_timestamp']}] "
                       f"Duration: {format_timestamp(seg['duration'])}\n")
        else:
            f.write("None - All ground truth segments were detected!\n")
        f.write("\n")

        # Unmatched predictions (false alarms)
        f.write("="*70 + "\n")
        f.write(f"FALSE ALARMS (False Positives: {len(results['unmatched_pred'])})\n")
        f.write("="*70 + "\n\n")

        if results['unmatched_pred']:
            for i, seg in enumerate(results['unmatched_pred'], 1):
                f.write(f"{i}. [{seg['start_timestamp']} - {seg['end_timestamp']}] "
                       f"Duration: {format_timestamp(seg['duration'])}\n")
                if 'max_score' in seg:
                    f.write(f"   Score: {seg.get('max_score', 'N/A')}/10\n")
                if 'reasoning' in seg:
                    f.write(f"   Reasoning: {seg.get('reasoning', '')}\n")
                f.write("\n")
        else:
            f.write("None - All predictions matched ground truth!\n")
        f.write("\n")

    print(f"[OK] Saved evaluation report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate extracted highlights against ground truth'
    )
    parser.add_argument('--ground-truth', required=True,
                       help='Path to ground truth CSV file (label.csv)')
    parser.add_argument('--predicted', required=True,
                       help='Path to predicted highlights JSON')
    parser.add_argument('--output', required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Frame rate for timecode parsing (default: 30)')

    args = parser.parse_args()

    print("="*70)
    print("HIGHLIGHT EVALUATION")
    print("="*70)

    # Load data
    print(f"\n[1/3] Loading data...")
    ground_truth = load_ground_truth(args.ground_truth, args.fps)
    print(f"  Loaded {len(ground_truth)} ground truth segments")
    print(f"  Total GT time: {format_timestamp(sum(s['duration'] for s in ground_truth))}")

    predicted = load_predicted_highlights(args.predicted)
    print(f"  Loaded {len(predicted)} predicted segments")
    print(f"  Total predicted time: {format_timestamp(sum(s['duration'] for s in predicted))}")

    # Evaluate
    print(f"\n[2/3] Evaluating (IoU threshold: {args.iou_threshold})...")
    results = evaluate_highlights(ground_truth, predicted, args.iou_threshold)

    # Print summary
    print(f"\n  Results:")
    print(f"    Precision: {results['precision']:.3f} ({results['precision']*100:.1f}%)")
    print(f"    Recall:    {results['recall']:.3f} ({results['recall']*100:.1f}%)")
    print(f"    F1-Score:  {results['f1_score']:.3f} ({results['f1_score']*100:.1f}%)")
    print(f"    Avg IoU:   {results['avg_iou']:.3f}")
    print(f"    Coverage:  {results['coverage']:.3f} ({results['coverage']*100:.1f}%)")

    # Save results
    print(f"\n[3/3] Saving results...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed report
    report_path = output_dir / "evaluation_report.txt"
    save_evaluation_report(results, report_path, args.iou_threshold)

    # Save JSON results
    json_path = output_dir / "evaluation_metrics.json"

    # Prepare JSON-serializable results
    json_results = {
        'iou_threshold': args.iou_threshold,
        'n_ground_truth': results['n_ground_truth'],
        'n_predicted': results['n_predicted'],
        'true_positives': results['true_positives'],
        'false_positives': results['false_positives'],
        'false_negatives': results['false_negatives'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'avg_iou': results['avg_iou'],
        'coverage': results['coverage'],
        'temporal_precision': results['temporal_precision'],
        'total_gt_time': results['total_gt_time'],
        'total_pred_time': results['total_pred_time']
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    print(f"[OK] Saved metrics JSON: {json_path}")

    # Final summary
    print(f"\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Precision: {results['precision']*100:.1f}%")
    print(f"Recall:    {results['recall']*100:.1f}%")
    print(f"F1-Score:  {results['f1_score']*100:.1f}%")
    print(f"\nDetailed report: {report_path}")
    print("="*70)


if __name__ == "__main__":
    main()
