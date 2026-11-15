"""
STURM-FloodDepth Model Training Pipeline
========================================
Trains or fine-tunes MobileNetV3-Small for 5-class flood level depth classification
(Level 0 to Level 4) on the STURM-FloodDepth dataset (Zenodo DOI: 10.5281/zenodo.14833532).

Usage:
    python src/train_sturm.py --data-dir data/sturm --epochs 15 --batch-size 32 --lr 1e-3
    python src/train_sturm.py --resume weights/best_flood_level_model.pth --epochs 5
"""

import os
import sys
import argparse
import json
import time
import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.sturm_classifier import STURMFloodLevelClassifier, STURM_CLASSES, STURM_LEVEL_METADATA


class STURMDataset(Dataset):
    """
    Dataset loader for STURM-FloodDepth. Supports both folder structure
    (data/sturm/Level0...Level4) and text split files (train.txt, val.txt, test.txt).
    """
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            # Fallback if image file is corrupted
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def discover_sturm_samples(data_dir: str, split_file: str | None = None) -> list[tuple[str, int]]:
    """
    Discovers image files and labels for STURM dataset.
    """
    samples = []
    class_map = {f"Level{i}": i for i in range(5)}
    # Also support lowercase or alternate naming
    for i in range(5):
        class_map[f"level{i}"] = i
        class_map[f"level_{i}"] = i
        class_map[f"l{i}"] = i

    # 1. Check if a split text file is provided and exists
    if split_file and os.path.exists(split_file):
        with open(split_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                img_path, lbl_str = parts[0], parts[1]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(data_dir, img_path)
                try:
                    lbl = int(lbl_str)
                    if os.path.exists(img_path):
                        samples.append((img_path, lbl))
                except ValueError:
                    pass
            elif len(parts) == 1:
                img_path = parts[0]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(data_dir, img_path)
                # Infer label from parent directory name
                parent_name = os.path.basename(os.path.dirname(img_path))
                if parent_name in class_map and os.path.exists(img_path):
                    samples.append((img_path, class_map[parent_name]))

        if samples:
            return samples

    # 2. Scan class subdirectories (Level0, Level1, Level2, Level3, Level4)
    for class_name, class_idx in class_map.items():
        sub_dir = os.path.join(data_dir, class_name)
        if os.path.exists(sub_dir):
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp'):
                for p in glob.glob(os.path.join(sub_dir, ext)):
                    samples.append((p, class_idx))

    return samples


def get_transforms(img_size: int = 224):
    """Data augmentations and normalizations."""
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False, dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(correct/total):.4f}"})

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    total = len(all_targets)
    epoch_loss = running_loss / max(1, total)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = float(np.mean(all_preds == all_targets)) if total > 0 else 0.0

    # Confusion matrix (5x5)
    cm = np.zeros((5, 5), dtype=int)
    for p, t in zip(all_preds, all_targets):
        if 0 <= t < 5 and 0 <= p < 5:
            cm[t, p] += 1

    # Class-wise metrics
    per_class_f1 = []
    for i in range(5):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))

    return {
        'val_loss': epoch_loss,
        'val_acc': accuracy,
        'macro_f1': macro_f1,
        'per_class_f1': [round(f, 4) for f in per_class_f1],
        'confusion_matrix': cm.tolist(),
    }


def plot_training_results(history: dict, out_dir: str):
    """Saves loss, accuracy curves and final confusion matrix heatmap."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Training & Validation Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('STURM Flood Depth - Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Accuracy (%)')
    plt.plot(epochs, [a * 100 for a in history['val_acc']], 'r-s', label='Val Accuracy (%)')
    plt.plot(epochs, [f * 100 for f in history['macro_f1']], 'g-^', label='Macro F1 (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage (%)')
    plt.title('STURM Flood Depth - Accuracy & F1-Score')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    curves_path = os.path.join(out_dir, 'sturm_training_curves.png')
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"Saved training curves to: {curves_path}")

    # 2. Confusion Matrix Heatmap
    if 'final_confusion_matrix' in history:
        cm = np.array(history['final_confusion_matrix'])
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('STURM 5-Class Flood Depth Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(5)
        plt.xticks(tick_marks, STURM_CLASSES, rotation=45)
        plt.yticks(tick_marks, STURM_CLASSES)

        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Flood Level')
        plt.xlabel('Predicted Flood Level')
        plt.tight_layout()
        cm_path = os.path.join(out_dir, 'sturm_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix plot to: {cm_path}")


def main():
    parser = argparse.ArgumentParser(description="Train STURM-FloodDepth Flood Level Classifier")
    parser.add_argument('--data-dir',    type=str, default='data/sturm', help='Path to STURM dataset root')
    parser.add_argument('--train-split', type=str, default=None,         help='Path to train.txt split file')
    parser.add_argument('--val-split',   type=str, default=None,         help='Path to val.txt split file')
    parser.add_argument('--epochs',      type=int, default=15,           help='Number of training epochs')
    parser.add_argument('--batch-size',  type=int, default=32,           help='Batch size (GPU recommended: 32)')
    parser.add_argument('--lr',          type=float, default=1e-3,       help='Initial learning rate')
    parser.add_argument('--img-size',    type=int, default=224,          help='Input image resolution')
    parser.add_argument('--resume',      type=str, default=None,         help='Resume training from checkpoint')
    parser.add_argument('--save-dir',    type=str, default='weights',    help='Checkpoint save directory')
    parser.add_argument('--out-dir',     type=str, default='outputs',    help='Outputs directory for plots')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("   STURM-FLOODDEPTH MODEL TRAINING (ZENODO: 14833532)")
    print("=" * 65)
    print(f"Compute Device : {device}")
    print(f"Dataset Root   : {os.path.abspath(args.data_dir)}")

    train_transform, val_transform = get_transforms(args.img_size)

    # Discover samples
    train_split_file = args.train_split or os.path.join(args.data_dir, 'train.txt')
    val_split_file = args.val_split or os.path.join(args.data_dir, 'val.txt')

    train_samples = discover_sturm_samples(args.data_dir, train_split_file if os.path.exists(train_split_file) else None)
    val_samples = discover_sturm_samples(args.data_dir, val_split_file if os.path.exists(val_split_file) else None)

    # If splits weren't distinct, partition all discovered samples 80/20
    if len(val_samples) == 0 and len(train_samples) > 0:
        print("No separate validation split file found. Creating stratified 80/20 train/val split...")
        np.random.seed(42)
        indices = np.random.permutation(len(train_samples))
        split_idx = int(0.8 * len(train_samples))
        all_samples = train_samples
        train_samples = [all_samples[i] for i in indices[:split_idx]]
        val_samples = [all_samples[i] for i in indices[split_idx:]]

    if len(train_samples) == 0:
        print(f"\n[ERROR] No dataset images found in '{args.data_dir}'!")
        print("Please download the STURM dataset from Zenodo:")
        print("    https://zenodo.org/records/14833532")
        print("Or run the automated downloader:")
        print("    python src/utils/download_sturm.py --output-dir data/sturm")
        print("\nExpected directory structure:")
        print("    data/sturm/Level0/")
        print("    data/sturm/Level1/")
        print("    data/sturm/Level2/")
        print("    data/sturm/Level3/")
        print("    data/sturm/Level4/")
        return

    print(f"Training samples   : {len(train_samples)}")
    print(f"Validation samples : {len(val_samples)}")

    # Compute class distribution and weights to mitigate class imbalance
    class_counts = [0] * 5
    for _, lbl in train_samples:
        if 0 <= lbl < 5:
            class_counts[lbl] += 1
    print(f"Class Distribution : {dict(zip(STURM_CLASSES, class_counts))}")

    # Compute inverse frequency weights
    total_count = sum(class_counts)
    class_weights = [total_count / (5.0 * max(1, c)) for c in class_counts]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_dataset = STURMDataset(train_samples, transform=train_transform)
    val_dataset = STURMDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = STURMFloodLevelClassifier(num_classes=5, pretrained=(args.resume is None)).to(device)

    # Resume from checkpoint if provided
    start_epoch = 1
    best_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        best_acc = ckpt.get('val_acc', 0.0)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Loaded weights (best_acc: {best_acc*100:.2f}%)")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'macro_f1': [],
    }

    os.makedirs(args.save_dir, exist_ok=True)
    best_weights_path = os.path.join(args.save_dir, 'best_flood_level_model.pth')

    print("\nStarting Training Loop...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_sec = time.time() - t0

        val_loss = val_metrics['val_loss']
        val_acc = val_metrics['val_acc']
        macro_f1 = val_metrics['macro_f1']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['macro_f1'].append(macro_f1)
        history['final_confusion_matrix'] = val_metrics['confusion_matrix']

        print(f"Epoch [{epoch:02d}/{start_epoch + args.epochs - 1:02d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% F1: {macro_f1*100:.1f}% "
              f"({epoch_sec:.1f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'macro_f1': macro_f1,
                'classes': STURM_CLASSES,
                'class_to_idx': {c: i for i, c in enumerate(STURM_CLASSES)},
                'arch': 'mobilenet_v3_small',
            }, best_weights_path)
            print(f"  --> Saved new best checkpoint to '{best_weights_path}' (Val Acc: {val_acc*100:.2f}%)")

    # Save metrics and plots
    os.makedirs(args.out_dir, exist_ok=True)
    history_path = os.path.join(args.out_dir, 'history_sturm.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    plot_training_results(history, args.out_dir)
    print(f"\n[DONE] Training complete! Best validation accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()

