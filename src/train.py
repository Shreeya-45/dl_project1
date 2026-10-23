import os
import argparse
import json
import time
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from models import UNetBaseline, MobileNetUNet, CombinedCrossEntropyDiceLoss
from utils import FloodRoadDataset, split_dataset, load_cvfd_dataset, compute_confusion_matrix, calculate_segmentation_metrics

def train_epoch(model, dataloader, criterion, optimizer, device, epoch=1, total_epochs=10, scaler=None):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Train", leave=False, dynamic_ncols=True)
    use_amp = (scaler is not None) and (device.type == 'cuda')

    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss, ce, dice = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss, ce, dice = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_ce += ce.item() * imgs.size(0)
        running_dice += dice.item() * imgs.size(0)

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{dice.item():.4f}"})

    total_samples = len(dataloader.dataset)
    return running_loss / total_samples, running_ce / total_samples, running_dice / total_samples


def evaluate(model, dataloader, criterion, device, num_classes=3):
    model.eval()
    running_loss = 0.0
    hist = np.zeros((num_classes, num_classes), dtype=int)

    pbar = tqdm(dataloader, desc="Validation", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for imgs, masks, _ in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss, _, _ = criterion(logits, masks)
            running_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()

            for p, t in zip(preds, targets):
                hist += compute_confusion_matrix(p, t, num_classes=num_classes)

    total_samples = len(dataloader.dataset)
    val_loss = running_loss / total_samples
    metrics = calculate_segmentation_metrics(hist)
    metrics['val_loss'] = val_loss
    return metrics



def main():
    parser = argparse.ArgumentParser(description="Train Flood & Road Segmentation Models")
    parser.add_argument('--model', type=str, default='mobilenet_unet', choices=['unet', 'mobilenet_unet'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (GPU recommended: 16)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--img-size', type=int, default=256, help='Input image height/width')
    parser.add_argument('--cvfd-dir', type=str, default=r'C:\Users\HP\Downloads\archive (11)\cvfd', help='Path to CVFD dataset root')
    parser.add_argument('--include-non-flooded', action='store_true', help='Include non-flooded images as negative samples (empty masks)')
    parser.add_argument('--no-amp', action='store_true', help='Disable Automatic Mixed Precision (AMP)')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Path to raw synthetic data directory (fallback)')
    parser.add_argument('--limit-train', type=int, default=None, help='Limit number of training samples for quick runs')
    parser.add_argument('--limit-val', type=int, default=None, help='Limit number of validation samples')
    parser.add_argument('--save-dir', type=str, default='weights', help='Path to save checkpoints')
    parser.add_argument('--out-dir', type=str, default='outputs', help='Path to save metrics/plots')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load CVFD Dataset if available, else fallback to standard split_dataset
    if args.cvfd_dir and os.path.exists(args.cvfd_dir):
        print(f"Loading Close-View Flood Dataset (CVFD) from: {args.cvfd_dir}")
        (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks) = load_cvfd_dataset(
            args.cvfd_dir, include_non_flooded=args.include_non_flooded
        )
    else:
        img_dir = os.path.join(args.data_dir, "images")
        mask_dir = os.path.join(args.data_dir, "masks")
        (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks) = split_dataset(
            img_dir, mask_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
        )

    # Optional sample limiting
    if args.limit_train and args.limit_train < len(train_imgs):
        train_imgs, train_masks = train_imgs[:args.limit_train], train_masks[:args.limit_train]
    if args.limit_val and args.limit_val < len(val_imgs):
        val_imgs, val_masks = val_imgs[:args.limit_val], val_masks[:args.limit_val]

    print(f"Dataset Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    train_ds = FloodRoadDataset(train_imgs, train_masks, img_size=(args.img_size, args.img_size), augment=True)
    val_ds = FloodRoadDataset(val_imgs, val_masks, img_size=(args.img_size, args.img_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    # Initialize Model
    if args.model == 'unet':
        model = UNetBaseline(in_channels=3, num_classes=3).to(device)
    else:
        model = MobileNetUNet(in_channels=3, num_classes=3).to(device)

    print(f"Model Initialized: {model.name}")

    criterion = CombinedCrossEntropyDiceLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (not args.no_amp) and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Automatic Mixed Precision (AMP) enabled for accelerated training.")

    best_miou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_miou': [], 'val_dice': []}

    print("Starting Model Training...")
    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        t_loss, t_ce, t_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch, total_epochs=args.epochs, scaler=scaler
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        v_loss = val_metrics['val_loss']
        v_miou = val_metrics['mIoU']
        v_dice = val_metrics['mean_dice']

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_miou'].append(v_miou)
        history['val_dice'].append(v_dice)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
              f"Val mIoU: {v_miou:.4f} | Val Dice: {v_dice:.4f}", flush=True)

        # Save Best Checkpoint
        if v_miou > best_miou:
            best_miou = v_miou
            save_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_miou': best_miou,
                'model_name': model.name
            }, save_path)
            print(f" --> Saved best checkpoint to {save_path} (mIoU: {best_miou:.4f})", flush=True)


    total_train_time = time.time() - start_train_time
    print(f"\nTraining Complete in {total_train_time / 60:.2f} mins! Best Val mIoU: {best_miou:.4f}")

    # Plot & Save Training History
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), history['train_loss'], label='Train Loss', color='blue')
    plt.plot(range(1, args.epochs + 1), history['val_loss'], label='Val Loss', color='orange')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), history['val_miou'], label='Val mIoU', color='green')
    plt.plot(range(1, args.epochs + 1), history['val_dice'], label='Val Dice', color='purple')
    plt.title('Validation Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"training_history_{args.model}.png"))
    plt.close()

    # Save log JSON
    with open(os.path.join(args.out_dir, f"history_{args.model}.json"), 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
