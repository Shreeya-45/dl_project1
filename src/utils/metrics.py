import time
import torch
import numpy as np
import cv2

def compute_confusion_matrix(pred, target, num_classes=3):
    """
    Computes confusion matrix between pred and target numpy arrays.
    """
    mask = (target >= 0) & (target < num_classes)
    hist = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return hist


def calculate_segmentation_metrics(hist):
    """
    Calculates mIoU, Dice Score (F1), Precision, Recall from confusion matrix.
    hist: (C, C) confusion matrix where hist[i, j] is count of class i predicted as class j.
    """
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    
    # Intersection over Union (IoU) per class
    iou_denom = tp + fp + fn
    iou = np.divide(tp, iou_denom, out=np.zeros_like(tp, dtype=float), where=iou_denom > 0)
    miou = np.nanmean(iou)

    # Dice score / F1-score per class
    dice_denom = 2 * tp + fp + fn
    dice = np.divide(2 * tp, dice_denom, out=np.zeros_like(tp, dtype=float), where=dice_denom > 0)
    mean_dice = np.nanmean(dice)

    # Precision & Recall
    precision_denom = tp + fp
    precision = np.divide(tp, precision_denom, out=np.zeros_like(tp, dtype=float), where=precision_denom > 0)
    
    recall_denom = tp + fn
    recall = np.divide(tp, recall_denom, out=np.zeros_like(tp, dtype=float), where=recall_denom > 0)

    # Specific metrics
    flooded_road_iou = iou[2] if len(iou) > 2 else 0.0  # Class 2: Water/Flood
    road_iou = iou[1] if len(iou) > 1 else 0.0          # Class 1: Road

    return {
        'mIoU': float(miou),
        'mean_dice': float(mean_dice),
        'class_iou': [float(x) for x in iou],
        'class_dice': [float(x) for x in dice],
        'class_precision': [float(x) for x in precision],
        'class_recall': [float(x) for x in recall],
        'road_iou': float(road_iou),
        'flooded_road_iou': float(flooded_road_iou)
    }


def calculate_boundary_f1(pred_mask, gt_mask, tolerance=2):
    """
    Calculates Boundary F1-score for mask contour precision.
    """
    # Extract boundaries
    pred_edges = cv2.Canny((pred_mask > 0).astype(np.uint8) * 255, 100, 200) > 0
    gt_edges = cv2.Canny((gt_mask > 0).astype(np.uint8) * 255, 100, 200) > 0

    if not gt_edges.any() and not pred_edges.any():
        return 1.0
    if not gt_edges.any() or not pred_edges.any():
        return 0.0

    # Distance transform for boundary matching
    gt_dist = cv2.distanceTransform((~gt_edges).astype(np.uint8), cv2.DIST_L2, 5)
    pred_dist = cv2.distanceTransform((~pred_edges).astype(np.uint8), cv2.DIST_L2, 5)

    precision = np.mean(gt_dist[pred_edges] <= tolerance) if pred_edges.any() else 0.0
    recall = np.mean(pred_dist[gt_edges] <= tolerance) if gt_edges.any() else 0.0

    if precision + recall == 0:
        return 0.0
    boundary_f1 = (2 * precision * recall) / (precision + recall)
    return float(boundary_f1)


def measure_model_performance(model, input_size=(1, 3, 256, 256), num_warmup=10, num_runs=50):
    """
    Measures Model Parameters, Memory usage, Inference Latency (ms) and FPS.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dummy_input = torch.randn(*input_size).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6
    model_size_mb = (num_params * 4) / (1024 * 1024)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Timing latency
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    total_time = time.time() - start_time

    avg_latency_ms = (total_time / num_runs) * 1000.0
    fps = num_runs / total_time

    return {
        'num_params_m': round(num_params_m, 3),
        'model_size_mb': round(model_size_mb, 2),
        'latency_ms': round(avg_latency_ms, 2),
        'fps': round(fps, 1),
        'device': str(device)
    }
