import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from models import UNetBaseline, MobileNetUNet
from utils import (
    FloodRoadDataset, split_dataset, load_cvfd_dataset, compute_confusion_matrix,
    calculate_segmentation_metrics, calculate_boundary_f1, measure_model_performance
)

def evaluate_on_test_set(model, test_loader, device, num_classes=3):
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=int)
    boundary_f1s = []

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

    metrics = calculate_segmentation_metrics(hist)
    metrics['boundary_f1'] = float(np.mean(boundary_f1s))
    
    # Efficiency benchmarks
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
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

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
        print("Using initialized U-Net weights for evaluation benchmark.")

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
        print("Using initialized MobileNet-UNet weights for evaluation benchmark.")

    mobilenet_res = evaluate_on_test_set(mobilenet_model, test_loader, device)
    results['MobileNet-UNet (Proposed)'] = mobilenet_res

    # Print Comparative Metrics Table
    print("\n" + "="*80)
    print(f"{'CAPSTONE EXPERIMENTAL & COMPARATIVE EVALUATION SUMMARY':^80}")
    print("="*80)
    header = f"{'Metric':<25} | {'U-Net Baseline':<22} | {'MobileNet-UNet (Proposed)':<25}"
    print(header)
    print("-" * len(header))

    metrics_to_show = [
        ('Mean IoU (mIoU)', 'mIoU', '{:.4f}'),
        ('Mean Dice Score (F1)', 'mean_dice', '{:.4f}'),
        ('Road IoU', 'road_iou', '{:.4f}'),
        ('Flooded-Road IoU', 'flooded_road_iou', '{:.4f}'),
        ('Boundary Quality F1', 'boundary_f1', '{:.4f}'),
        ('Parameters (M)', 'num_params_m', '{:.3f}M'),
        ('Model Size (MB)', 'model_size_mb', '{:.2f} MB'),
        ('Inference Latency', 'latency_ms', '{:.2f} ms'),
        ('FPS (Frames / Sec)', 'fps', '{:.1f} FPS')
    ]

    for label, key, fmt in metrics_to_show:
        u_val = fmt.format(unet_res[key])
        m_val = fmt.format(mobilenet_res[key])
        print(f"{label:<25} | {u_val:<22} | {m_val:<25}")

    print("="*80)

    # Save summary report JSON
    report_path = os.path.join(args.out_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete comparative report to '{report_path}'!")

if __name__ == "__main__":
    main()
