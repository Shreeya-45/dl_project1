import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from models import UNetBaseline, MobileNetUNet
from utils import (
    FloodRoadDataset, split_dataset, load_cvfd_dataset, compute_confusion_matrix,
    calculate_segmentation_metrics, calculate_boundary_f1, measure_model_performance,
    calculate_passability_metrics, calculate_depth_metrics, PassabilityAnalyzer
)

def evaluate_on_test_set(model, test_loader, device, num_classes=3):
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=int)
    boundary_f1s = []
    
    analyzer = PassabilityAnalyzer()
    gt_statuses = []
    pred_statuses = []
    all_gt_masks = []
    all_pred_masks = []

    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs_dev = imgs.to(device)
            logits = model(imgs_dev)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = masks.numpy()

            for p, t in zip(preds, targets):
                hist += compute_confusion_matrix(p, t, num_classes=num_classes)
                bf1 = calculate_boundary_f1(p, t)
                boundary_f1s.append(bf1)
                
                # Analyze passability status for ground truth vs prediction
                gt_pass = analyzer.analyze(t)
                pred_pass = analyzer.analyze(p)
                gt_statuses.append(gt_pass['status'])
                pred_statuses.append(pred_pass['status'])
                
                all_gt_masks.append(t)
                all_pred_masks.append(p)

    # 1. Semantic Segmentation Metrics
    metrics = calculate_segmentation_metrics(hist)
    metrics['boundary_f1'] = float(np.mean(boundary_f1s))

    # 2. Road Passability Classification Metrics
    pass_metrics = calculate_passability_metrics(gt_statuses, pred_statuses)
    metrics.update(pass_metrics)

    # 3. Water-Depth Estimation Metrics (MAE, RMSE, Relative Error)
    depth_metrics = calculate_depth_metrics(all_gt_masks, all_pred_masks, analyzer)
    metrics.update(depth_metrics)

    # 4. Hardware & Efficiency Benchmarks
    perf = measure_model_performance(model, input_size=(1, 3, 256, 256))
    metrics.update(perf)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate & Compare Baseline U-Net vs Proposed MobileNet-UNet")
    parser.add_argument('--cvfd-dir', type=str, default=r'C:\Users\HP\Downloads\archive (11)\cvfd', help='Path to CVFD dataset root')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Path to raw data directory (fallback)')
    parser.add_argument('--weights-dir', type=str, default='weights', help='Path to weights directory')
    parser.add_argument('--out-dir', type=str, default='outputs', help='Path to output report directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating models on device: {device}")

    # Load Test Split from CVFD if available
    if args.cvfd_dir and os.path.exists(args.cvfd_dir):
        print(f"Loading CVFD Test Split from: {args.cvfd_dir}")
        _, _, (test_imgs, test_masks) = load_cvfd_dataset(args.cvfd_dir)
    else:
        img_dir = os.path.join(args.data_dir, "images")
        mask_dir = os.path.join(args.data_dir, "masks")
        _, _, (test_imgs, test_masks) = split_dataset(img_dir, mask_dir)

    print(f"Test Set Size: {len(test_imgs)} samples")

    test_ds = FloodRoadDataset(test_imgs, test_masks, img_size=(256, 256), augment=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    results = {}

    # 1. Baseline U-Net Model
    print("\n--- Evaluating Baseline U-Net Model ---")
    unet_model = UNetBaseline(in_channels=3, num_classes=3).to(device)
    unet_ckpt = os.path.join(args.weights_dir, "best_unet.pth")
    if os.path.exists(unet_ckpt):
        checkpoint = torch.load(unet_ckpt, map_location=device)
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded U-Net trained weights.")
    else:
        print("Using initialized/benchmark U-Net model.")

    unet_res = evaluate_on_test_set(unet_model, test_loader, device)
    results['U-Net Baseline'] = unet_res

    # 2. Proposed MobileNet-UNet Lite Model
    print("\n--- Evaluating Proposed MobileNet-UNet Lite Model ---")
    mobilenet_model = MobileNetUNet(in_channels=3, num_classes=3).to(device)
    mobilenet_ckpt = os.path.join(args.weights_dir, "best_mobilenet_unet.pth")
    if os.path.exists(mobilenet_ckpt):
        checkpoint = torch.load(mobilenet_ckpt, map_location=device)
        mobilenet_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded MobileNet-UNet trained weights.")
    else:
        print("Using initialized MobileNet-UNet weights.")

    mobilenet_res = evaluate_on_test_set(mobilenet_model, test_loader, device)
    results['MobileNet-UNet (Proposed)'] = mobilenet_res

    # Print Comparative Metrics Table
    print("\n" + "="*85)
    print(f"{'CAPSTONE EXPERIMENTAL & COMPARATIVE EVALUATION SUMMARY':^85}")
    print("="*85)
    header = f"{'Metric Category / Specific Metric':<35} | {'U-Net Baseline':<20} | {'MobileNet-UNet (Proposed)':<25}"
    print(header)
    print("-" * len(header))

    metrics_groups = [
        ("=== Segmentation Metrics ===", None, None),
        ("Pixel Accuracy", "pixel_accuracy", "{:.2%}"),
        ("Mean IoU (mIoU)", "mIoU", "{:.4f}"),
        ("Mean Dice Score (F1)", "mean_dice", "{:.4f}"),
        ("Flooded-Road IoU", "flooded_road_iou", "{:.4f}"),
        ("Road IoU", "road_iou", "{:.4f}"),
        ("Mean Precision", "mean_precision", "{:.4f}"),
        ("Mean Recall", "mean_recall", "{:.4f}"),
        ("Boundary Contour F1", "boundary_f1", "{:.4f}"),
        
        ("=== Passability Classification ===", None, None),
        ("Passability Macro F1", "passability_macro_f1", "{:.4f}"),
        ("Passability Accuracy", "passability_accuracy", "{:.2%}"),
        ("Passable F1", "passability_f1_passable", "{:.4f}"),
        ("Partially Flooded F1", "passability_f1_partially_flooded", "{:.4f}"),
        ("Severely Affected F1", "passability_f1_severely_affected", "{:.4f}"),

        ("=== Depth Estimation Metrics ===", None, None),
        ("Depth MAE (cm)", "depth_mae_cm", "{:.2f} cm"),
        ("Depth RMSE (cm)", "depth_rmse_cm", "{:.2f} cm"),
        ("Relative Depth Error (%)", "depth_relative_error_pct", "{:.2f}%"),

        ("=== Edge Deployment & Efficiency ===", None, None),
        ("Parameters (M)", "num_params_m", "{:.3f} M"),
        ("Model Size (MB)", "model_size_mb", "{:.2f} MB"),
        ("Inference Latency (ms)", "latency_ms", "{:.2f} ms"),
        ("Throughput (FPS)", "fps", "{:.1f} FPS"),
        ("Peak Memory (VRAM MB)", "peak_memory_mb", "{:.2f} MB"),
        ("Estimated Power (Watts)", "estimated_power_w", "{:.1f} W")
    ]

    for label, key, fmt in metrics_groups:
        if key is None:
            print(f"\n{label}")
            continue
        u_val = fmt.format(unet_res[key])
        m_val = fmt.format(mobilenet_res[key])
        print(f"{label:<35} | {u_val:<20} | {m_val:<25}")

    print("="*85)

    # Save summary report JSON
    report_path = os.path.join(args.out_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete comparative report to '{report_path}'!")

if __name__ == "__main__":
    main()
