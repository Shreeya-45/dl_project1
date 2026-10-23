import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

class FloodRoadDataset(Dataset):
    """
    PyTorch Dataset for Flood & Road Segmentation with Harmonized Labels.
    Supports Image and Mask loading with augmentations.
    """
    def __init__(self, image_paths, mask_paths, transform=None, img_size=(256, 256), augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx] if self.mask_paths is not None else None

        # Load RGB Image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.img_size, Image.BILINEAR)
        img_arr = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

        # Load Ground Truth Mask (0: Background, 1: Road, 2: Water)
        if mask_path is None or mask_path == "":
            # Non-flooded sample: background mask (all 0s)
            mask_arr = np.zeros(self.img_size, dtype=np.int64)
        else:
            mask = Image.open(mask_path)
            mask = mask.resize(self.img_size, Image.NEAREST)
            mask_arr = np.array(mask, dtype=np.int64)
            
            # Handle CVFD RGB masks or single-channel masks
            if mask_arr.ndim == 3:
                # In CVFD, water is highlighted in Blue (channel 2 > 128) or Red (channel 0 > 128)
                water_mask = (mask_arr[:, :, 2] > 128) | (mask_arr[:, :, 0] > 128)
                label_mask = np.zeros(mask_arr.shape[:2], dtype=np.int64)
                label_mask[water_mask] = 2  # Class 2: Water / Floodwater
                mask_arr = label_mask
            else:
                mask_arr = np.clip(mask_arr, 0, 2)

        # Apply simple spatial augmentations (Random Horizontal Flip) if enabled
        if self.augment and np.random.rand() > 0.5:
            img_arr = np.fliplr(img_arr).copy()
            mask_arr = np.fliplr(mask_arr).copy()

        # Convert to PyTorch tensors
        # Image tensor shape: (C, H, W)
        img_tensor = torch.tensor(img_arr).permute(2, 0, 1).float()
        # Mask tensor shape: (H, W)
        mask_tensor = torch.tensor(mask_arr).long()

        return img_tensor, mask_tensor, img_path


def load_cvfd_dataset(base_dir=r"C:\Users\HP\Downloads\archive (11)\cvfd", include_non_flooded=False):
    """
    Loads matched image and mask paths from the Close-View-Flood-Dataset (CVFD).
    If include_non_flooded=True, also loads non-flooded images with zero/background masks.
    Returns (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks)
    """
    splits = {}
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(base_dir, 'images', split, 'flooded')
        mask_dir = os.path.join(base_dir, 'masks', 'flood', split, 'flooded')

        img_paths = sorted(glob.glob(os.path.join(img_dir, '*.*')))
        matched_imgs = []
        matched_masks = []

        for img_p in img_paths:
            base_name = os.path.splitext(os.path.basename(img_p))[0]
            mask_p = os.path.join(mask_dir, f"{base_name}.png")
            if os.path.exists(mask_p):
                matched_imgs.append(img_p)
                matched_masks.append(mask_p)

        if include_non_flooded:
            non_flooded_dir = os.path.join(base_dir, 'images', split, 'non-flooded')
            non_flooded_imgs = sorted(glob.glob(os.path.join(non_flooded_dir, '*.*')))
            for nf_img in non_flooded_imgs:
                matched_imgs.append(nf_img)
                matched_masks.append(None)

        splits[split] = (matched_imgs, matched_masks)

    return splits['train'], splits['val'], splits['test']


def split_dataset(image_dir, mask_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits dataset into 70% Train, 15% Validation, and 15% Testing.
    Returns (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks)
    """
    np.random.seed(seed)
    
    img_extensions = ['*.jpg', '*.png', '*.jpeg']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    img_paths = sorted(img_paths)
    mask_paths = []
    
    valid_imgs = []
    for img_p in img_paths:
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        # Look for corresponding mask file (.png or .jpg)
        mask_p = os.path.join(mask_dir, f"{base_name}.png")
        if not os.path.exists(mask_p):
            mask_p = os.path.join(mask_dir, f"{base_name}.jpg")
        
        if os.path.exists(mask_p):
            valid_imgs.append(img_p)
            mask_paths.append(mask_p)

    total = len(valid_imgs)
    indices = np.arange(total)
    np.random.shuffle(indices)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_imgs = [valid_imgs[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]

    val_imgs = [valid_imgs[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    test_imgs = [valid_imgs[i] for i in test_idx]
    test_masks = [mask_paths[i] for i in test_idx]

    return (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks)

