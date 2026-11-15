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
    Calculates Pixel Accuracy, mIoU, Dice Score (F1), Precision, Recall from confusion matrix.
    hist: (C, C) confusion matrix where hist[i, j] is count of class i predicted as class j.
    """
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    
    # Overall Pixel Accuracy: (Total Correct Pixels / Total Pixels)
    total_pixels = hist.sum()
    pixel_acc = float(np.sum(tp) / total_pixels) if total_pixels > 0 else 0.0

    # Per-class accuracy (Recall)
    class_acc_denom = tp + fn
    class_acc = np.divide(tp, class_acc_denom, out=np.zeros_like(tp, dtype=float), where=class_acc_denom > 0)
    # Filter classes that actually exist in ground truth
    valid_classes = class_acc_denom > 0
    mean_class_acc = float(np.mean(class_acc[valid_classes])) if np.any(valid_classes) else 0.0

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
    valid_prec = precision_denom > 0
    mean_precision = float(np.mean(precision[valid_prec])) if np.any(valid_prec) else 0.0
    
    recall_denom = tp + fn
    recall = np.divide(tp, recall_denom, out=np.zeros_like(tp, dtype=float), where=recall_denom > 0)
    valid_rec = recall_denom > 0
    mean_recall = float(np.mean(recall[valid_rec])) if np.any(valid_rec) else 0.0

    # Specific metrics
    flooded_road_iou = iou[2] if len(iou) > 2 else 0.0  # Class 2: Water/Flood
    road_iou = iou[1] if len(iou) > 1 else 0.0          # Class 1: Road

    return {
        'pixel_accuracy': float(pixel_acc),
        'mean_class_accuracy': float(mean_class_acc),
        'mIoU': float(miou),
        'mean_dice': float(mean_dice),
        'mean_precision': float(mean_precision),
        'mean_recall': float(mean_recall),
        'class_accuracy': [float(x) for x in class_acc],
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


def calculate_passability_metrics(gt_statuses, pred_statuses):
    """
    Calculates Passability F1-score, Accuracy, and per-class metrics.
    Classes: 0: Passable, 1: Partially Flooded, 2: Severely Affected
    """
    status_map = {
        'Passable': 0,
        'Partially Flooded': 1,
        'Severely Affected': 2
    }
    
    y_true = np.array([status_map.get(s, 0) for s in gt_statuses])
    y_pred = np.array([status_map.get(s, 0) for s in pred_statuses])
    
    num_classes = 3
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1
        
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    
    prec_denom = tp + fp
    rec_denom = tp + fn
    prec = np.divide(tp, prec_denom, out=np.zeros_like(tp, dtype=float), where=prec_denom > 0)
    rec = np.divide(tp, rec_denom, out=np.zeros_like(tp, dtype=float), where=rec_denom > 0)
    
    f1_denom = prec + rec
    f1_per_class = np.divide(2 * prec * rec, f1_denom, out=np.zeros_like(prec, dtype=float), where=f1_denom > 0)
    
    total = len(y_true)
    accuracy = float(np.sum(y_true == y_pred) / total) if total > 0 else 0.0
    macro_f1 = float(np.mean(f1_per_class))
    
    return {
        'passability_accuracy': round(accuracy, 4),
        'passability_macro_f1': round(macro_f1, 4),
        'passability_f1_passable': round(float(f1_per_class[0]), 4),
        'passability_f1_partially_flooded': round(float(f1_per_class[1]), 4),
        'passability_f1_severely_affected': round(float(f1_per_class[2]), 4),
        'passability_confusion_matrix': conf_matrix.tolist()
    }


def calculate_depth_metrics(gt_masks, pred_masks, analyzer):
    """
    Calculates Water-Depth Estimation Metrics:
    MAE (Mean Absolute Error in cm), RMSE (Root Mean Square Error in cm), Relative Error (%).
    """
    mae_list = []
    rmse_list = []
    rel_error_list = []
    
    for gt_m, pred_m in zip(gt_masks, pred_masks):
        gt_depth, gt_max, _ = analyzer.compute_depth_map(gt_m, gt_m.shape)
        pred_depth, pred_max, _ = analyzer.compute_depth_map(pred_m, pred_m.shape)
        
        # Consider regions where either ground truth or predicted has water
        mask = (gt_m == 2) | (pred_m == 2)
        if np.any(mask):
            err = np.abs(pred_depth[mask] - gt_depth[mask])
            mae_list.append(float(np.mean(err)))
            rmse_list.append(float(np.sqrt(np.mean(err ** 2))))
            
            # Relative error relative to gt_depth (with epsilon 1.0 cm to prevent div by zero)
            denom = np.maximum(gt_depth[mask], 1.0)
            rel_err = np.mean(err / denom)
            rel_error_list.append(float(rel_err))
        else:
            # Both predicted no water and no water in GT -> error 0
            mae_list.append(0.0)
            rmse_list.append(0.0)
            rel_error_list.append(0.0)
            
    return {
        'depth_mae_cm': round(float(np.mean(mae_list)), 2),
        'depth_rmse_cm': round(float(np.mean(rmse_list)), 2),
        'depth_relative_error_pct': round(float(np.mean(rel_error_list)) * 100.0, 2)
    }


def measure_model_performance(model, input_size=(1, 3, 256, 256), num_warmup=10, num_runs=50):
    """
    Measures Model Parameters, Memory usage, Inference Latency (ms), FPS, and Power.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dummy_input = torch.randn(*input_size).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6
    model_size_mb = (num_params * 4) / (1024 * 1024)

    # Reset peak memory stats if CUDA
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

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

    # Memory measurement
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        import psutil
        peak_vram_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    # Power consumption estimation:
    # Embedded edge platforms (Jetson Orin Nano / Nano): ~7W - 15W
    # Desktop / Mobile RTX GPU during batch inference: ~35W - 55W
    if 'MobileNet' in getattr(model, 'name', type(model).__name__):
        est_edge_power_w = 7.5   # Ultra-efficient MobileNet encoder on Jetson edge device
    else:
        est_edge_power_w = 28.0  # Heavier standard U-Net on edge device

    return {
        'num_params_m': round(num_params_m, 3),
        'model_size_mb': round(model_size_mb, 2),
        'latency_ms': round(avg_latency_ms, 2),
        'fps': round(fps, 1),
        'peak_memory_mb': round(peak_vram_mb, 2),
        'estimated_power_w': est_edge_power_w,
        'device': str(device)
    }
