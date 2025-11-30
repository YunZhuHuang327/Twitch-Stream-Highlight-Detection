"""
Evaluation metrics for Highlight Detection

Metrics:
- Precision, Recall, F1
- Mean Average Precision (mAP) at different IoU thresholds
- Temporal localization accuracy
"""

from typing import List, Dict, Tuple
import numpy as np


def calculate_temporal_iou(pred_segment: Tuple[float, float], gt_segment: Tuple[float, float]) -> float:
    """
    Calculate Intersection over Union for time segments
    
    Args:
        pred_segment: (start, end) predicted segment
        gt_segment: (start, end) ground truth segment
    
    Returns:
        IoU value (0-1)
    """
    pred_start, pred_end = pred_segment
    gt_start, gt_end = gt_segment
    
    # Calculate intersection
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)
    
    # Calculate union
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start
    
    return intersection / union if union > 0 else 0


def match_predictions_to_ground_truth(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[List, List, List]:
    """
    Match predictions to ground truth highlights using IoU
    
    Args:
        predictions: List of predicted highlights
        ground_truth: List of ground truth highlights
        iou_threshold: IoU threshold for matching
    
    Returns:
        (true_positives, false_positives, false_negatives)
    """
    true_positives = []
    false_positives = []
    false_negatives = list(ground_truth)
    
    # 按置信度排序（如果有的话）
    sorted_preds = sorted(
        predictions,
        key=lambda x: x.get('confidence', 1.0),
        reverse=True
    )
    
    matched_gt = set()
    
    for pred in sorted_preds:
        pred_seg = (pred['start_time'], pred['end_time'])
        
        # 找到 IoU 最高的 ground truth
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            
            gt_seg = (gt['start_time'], gt['end_time'])
            iou = calculate_temporal_iou(pred_seg, gt_seg)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        # 判断是否匹配
        if best_iou >= iou_threshold:
            true_positives.append({
                'prediction': pred,
                'ground_truth': ground_truth[best_gt_idx],
                'iou': best_iou
            })
            matched_gt.add(best_gt_idx)
            false_negatives.remove(ground_truth[best_gt_idx])
        else:
            false_positives.append(pred)
    
    return true_positives, false_positives, false_negatives


def calculate_precision_recall_f1(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1 Score
    
    Args:
        predictions: List of predicted highlights
        ground_truth: List of ground truth highlights
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with precision, recall, f1, and counts
    """
    tp, fp, fn = match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold)
    
    num_tp = len(tp)
    num_fp = len(fp)
    num_fn = len(fn)
    
    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': num_tp,
        'false_positives': num_fp,
        'false_negatives': num_fn,
        'iou_threshold': iou_threshold
    }


def calculate_mean_average_precision(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_thresholds: List[float] = [0.3, 0.5, 0.7, 0.9]
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) at different IoU thresholds
    
    Args:
        predictions: List of predicted highlights
        ground_truth: List of ground truth highlights
        iou_thresholds: List of IoU thresholds
    
    Returns:
        Dictionary with mAP and AP at each threshold
    """
    ap_scores = []
    results = {}
    
    for iou_thresh in iou_thresholds:
        # 计算该阈值下的 precision-recall
        tp, fp, fn = match_predictions_to_ground_truth(predictions, ground_truth, iou_thresh)
        
        num_gt = len(ground_truth)
        num_pred = len(predictions)
        
        if num_pred == 0:
            ap = 0.0
        elif num_gt == 0:
            ap = 0.0
        else:
            # 按置信度排序计算 precision at each recall level
            sorted_preds = sorted(
                predictions,
                key=lambda x: x.get('confidence', 1.0),
                reverse=True
            )
            
            precisions = []
            recalls = []
            tp_count = 0
            
            matched_gt = set()
            
            for i, pred in enumerate(sorted_preds):
                pred_seg = (pred['start_time'], pred['end_time'])
                
                # 检查是否匹配
                matched = False
                for j, gt in enumerate(ground_truth):
                    if j in matched_gt:
                        continue
                    
                    gt_seg = (gt['start_time'], gt['end_time'])
                    iou = calculate_temporal_iou(pred_seg, gt_seg)
                    
                    if iou >= iou_thresh:
                        matched = True
                        matched_gt.add(j)
                        tp_count += 1
                        break
                
                precision = tp_count / (i + 1)
                recall = tp_count / num_gt
                
                precisions.append(precision)
                recalls.append(recall)
            
            # 计算 AP (area under precision-recall curve)
            ap = np.trapz(precisions, recalls) if len(recalls) > 1 else precisions[0] if precisions else 0
        
        ap_scores.append(ap)
        results[f'AP@{iou_thresh}'] = ap
    
    results['mAP'] = np.mean(ap_scores)
    
    return results


def calculate_all_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_thresholds: List[float] = [0.3, 0.5, 0.7]
) -> Dict:
    """
    Calculate all evaluation metrics
    
    Args:
        predictions: List of predicted highlights
        ground_truth: List of ground truth highlights
        iou_thresholds: List of IoU thresholds
    
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # Precision, Recall, F1 at each threshold
    for iou_thresh in iou_thresholds:
        metrics = calculate_precision_recall_f1(predictions, ground_truth, iou_thresh)
        results[f'@{iou_thresh}'] = metrics
    
    # mAP
    map_results = calculate_mean_average_precision(predictions, ground_truth, iou_thresholds)
    results['mAP'] = map_results
    
    # 统计信息
    results['stats'] = {
        'num_predictions': len(predictions),
        'num_ground_truth': len(ground_truth),
        'avg_pred_duration': np.mean([p['end_time'] - p['start_time'] for p in predictions]) if predictions else 0,
        'avg_gt_duration': np.mean([gt['end_time'] - gt['start_time'] for gt in ground_truth]) if ground_truth else 0
    }
    
    return results


def print_evaluation_results(results: Dict):
    """Print evaluation results in a nice format"""
    print("\n" + "="*60)
    print("📊 Highlight Detection Evaluation Results")
    print("="*60)
    
    # 统计信息
    stats = results['stats']
    print(f"\n📈 Statistics:")
    print(f"  Predictions: {stats['num_predictions']}")
    print(f"  Ground Truth: {stats['num_ground_truth']}")
    print(f"  Avg Predicted Duration: {stats['avg_pred_duration']:.1f}s")
    print(f"  Avg GT Duration: {stats['avg_gt_duration']:.1f}s")
    
    # Precision/Recall/F1 at different thresholds
    print(f"\n🎯 Precision, Recall, F1:")
    for key in sorted(results.keys()):
        if key.startswith('@'):
            metrics = results[key]
            print(f"\n  IoU {key}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall:    {metrics['recall']:.3f}")
            print(f"    F1 Score:  {metrics['f1']:.3f}")
            print(f"    TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    
    # mAP
    map_results = results['mAP']
    print(f"\n⭐ Mean Average Precision:")
    print(f"  mAP: {map_results['mAP']:.3f}")
    for key, value in sorted(map_results.items()):
        if key.startswith('AP@'):
            print(f"  {key}: {value:.3f}")
    
    print("\n" + "="*60)


def evaluate_video(
    predictions_file: str,
    ground_truth_file: str,
    iou_thresholds: List[float] = [0.3, 0.5, 0.7]
):
    """
    Evaluate predictions for a single video
    
    Args:
        predictions_file: Path to predictions JSON
        ground_truth_file: Path to ground truth JSON
        iou_thresholds: IoU thresholds for evaluation
    """
    import json
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    results = calculate_all_metrics(predictions, ground_truth, iou_thresholds)
    print_evaluation_results(results)
    
    return results


# 示例使用
if __name__ == "__main__":
    # 示例数据
    predictions = [
        {'start_time': 930, 'end_time': 1125, 'type': 'exciting_moment', 'confidence': 0.9},
        {'start_time': 4980, 'end_time': 5190, 'type': 'funny_moment', 'confidence': 0.85},
    ]
    
    ground_truth = [
        {'start_time': 920, 'end_time': 1130, 'type': 'exciting_moment'},
        {'start_time': 5000, 'end_time': 5200, 'type': 'funny_moment'},
        {'start_time': 7200, 'end_time': 7350, 'type': 'skill_showcase'},
    ]
    
    results = calculate_all_metrics(predictions, ground_truth)
    print_evaluation_results(results)
