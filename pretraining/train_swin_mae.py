"""
Training script for Swin3D MAE on vesuvius ink detection data
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import random
from warmup_scheduler import GradualWarmupScheduler

# Add parent directory to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from pretraining.swin_mae import VideoMAE3D


class CFG:
    # Paths
    segment_path = './pretraining_scrolls/'

    # Data
    in_chans = 35
    valid_chans = 30  # Number of frames to use
    size = 64

    # Training
    train_batch_size = 8
    valid_batch_size = 4
    lr = 1e-4
    epochs = 100
    warmup_epochs = 10

    # Model
    mask_ratio = 0.75

    # Optimizer
    weight_decay = 0.05
    max_grad_norm = 1.0
    min_lr = 1e-6

    # System
    num_workers = 8
    seed = 42

    # Augmentation
    train_aug_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.15, p=0.75),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        ToTensorV2(transpose_mask=True),
    ]


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(split, cfg):
    """Get augmentation transforms"""
    if split == 'train':
        return A.Compose(cfg.train_aug_list)
    return A.Compose(cfg.valid_aug_list)


class TileDataset(Dataset):
    """Dataset for loading 3D tiles"""

    def __init__(self, base_path, splits=["train"], transform=None, cfg=None):
        self.cfg = cfg or CFG()
        self.tile_paths = []

        for split in splits:
            split_path = os.path.join(base_path, "64_tiles_s4", split)
            if os.path.exists(split_path):
                self.tile_paths += [
                    os.path.join(split_path, f)
                    for f in os.listdir(split_path) if f.endswith(".npy")
                ]

        self.transform = transform
        self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

        print(f"Loaded {len(self.tile_paths)} tiles from {splits}")

    def __len__(self):
        return len(self.tile_paths)

    def random_channel_crop(self, image):
        """Randomly crop consecutive channels (temporal dimension)"""
        cropping_num = self.cfg.valid_chans
        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)
        return image[..., crop_indices]

    def __getitem__(self, idx):
        # Load tile: (H, W, C) where C is depth/temporal
        image = np.load(self.tile_paths[idx])  # (64, 64, 35)

        # Random channel crop
        image = self.random_channel_crop(image)  # (64, 64, 30)

        # Apply spatial augmentation
        if self.transform:
            data = self.transform(image=image)
            image = data['image']  # (30, 64, 64)

        # Reshape to video format: (C, T, H, W) -> (C, D, H, W)
        # We have (T, H, W), need to add channel dimension
        image = image.unsqueeze(0)  # (1, 30, 64, 64)

        # Normalize each frame
        image = image.float()
        image = (image - 0.5) / 0.5

        return image  # (1, 30, 64, 64)


def train_epoch(model, train_loader, optimizer, device, epoch, cfg):
    """Train for one epoch"""
    model.train()
    train_losses = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.epochs} [Train]')
    for batch in pbar:
        batch = batch.to(device)  # (B, 1, 30, 64, 64)

        # Forward pass
        loss, pred, mask = model(batch, mask_ratio=cfg.mask_ratio)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        train_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return np.mean(train_losses)


@torch.no_grad()
def validate(model, val_loader, device, epoch, cfg):
    """Validate the model"""
    model.eval()
    val_losses = []

    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{cfg.epochs} [Val]')
    for batch in pbar:
        batch = batch.to(device)

        loss, pred, mask = model(batch, mask_ratio=cfg.mask_ratio)
        val_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return np.mean(val_losses)


def main():
    # Set seed
    set_seed(CFG.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    full_dataset = TileDataset(
        CFG.segment_path,
        splits=["train", "valid"],
        transform=get_transforms('train', CFG),
        cfg=CFG
    )

    # Split train/val
    val_size = max(int(0.05 * len(full_dataset)), 1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    # Input: (B, C, D, H, W) = (B, 1, 30, 64, 64)
    model = VideoMAE3D(
        img_size=(30, 64, 64),  # (D, H, W)
        patch_size=(2, 4, 4),   # (D, H, W)
        in_channels=1,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        decoder_embed_dim=384,
        decoder_depths=(2, 2),
        decoder_num_heads=(12, 12),
        window_size=(8, 4, 4),  # Adjusted for input size
        mask_ratio=CFG.mask_ratio,
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
        betas=(0.9, 0.95)
    )

    # Scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CFG.epochs,
        eta_min=CFG.min_lr
    )

    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=CFG.warmup_epochs,
        after_scheduler=cosine_scheduler
    )

    # Training loop
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(CFG.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, CFG)

        # Validate
        val_loss = validate(model, val_loader, device, epoch, CFG)

        # Step scheduler
        scheduler.step()

        # Log
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{CFG.epochs} - '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'LR: {lr:.6f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'checkpoints/swin_mae_best.pth')
            print(f'✓ Saved best model (val_loss: {val_loss:.4f})')

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'checkpoints/swin_mae_epoch_{epoch+1}.pth')
            print(f'✓ Saved checkpoint at epoch {epoch+1}')

    print("\n=== Training completed! ===")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
