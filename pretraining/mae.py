import os
import sys
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins.io import TorchCheckpointIO
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from warmup_scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt
from transformers import VideoMAEConfig, VideoMAEForPreTraining

# Increase PIL image size limit
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000


class CustomCheckpointIO(TorchCheckpointIO):
    """Custom checkpoint I/O that uses weights_only=False for PyTorch 2.6+ compatibility"""

    def load_checkpoint(self, path, map_location=None):
        """Load checkpoint with weights_only=False since we trust our own checkpoints"""
        return torch.load(path, map_location=map_location, weights_only=False)

class CFG:
    """Centralized configuration"""
    # Paths
    current_dir = '../'
    segment_path = './pretraining_scrolls/'

    # Model architecture
    start_idx = 22
    in_chans = 18
    valid_chans = 16
    size = 224
    tile_size = 224

    # Training hyperparameters
    train_batch_size = 128  # Reduced for video processing
    valid_batch_size = 128
    lr = 1e-4
    epochs = 200
    warmup_epochs = 5

    # Checkpoint resumption
    resume_from_checkpoint = None#'checkpoints/videomae_epoch=008_val_loss=0.5532.ckpt'  # Set to checkpoint path to resume training (e.g., 'checkpoints/videomae_epoch=008_val_loss=0.5532.ckpt')

    # Masking strategy
    mask_ratio = 0.85 # VideoMAE typically uses high mask ratios (0.75-0.9)

    # VideoMAE Architecture
    image_size = size
    patch_size = 16  # Spatial patch size
    num_channels = 1  # VideoMAE expects 3 channels (RGB-like)
    input_channels = 1  # Our actual input channels (grayscale)
    hidden_size = 768  # Smaller for efficiency
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = 1536
    decoder_num_hidden_layers = 4
    decoder_hidden_size = 512
    decoder_num_attention_heads = 8
    decoder_intermediate_size = 768

    # Video dimensions
    input_frames = 16
    tubelet_size = 2  # Temporal patch size

    # Optimizer
    weight_decay = 0.05
    max_grad_norm = 1.0

    # Data
    val_split = 0.005
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
    """Dataset for loading tile data with channel augmentation"""
    
    def __init__(self, base_path, splits=["train"], transform=None, cfg=None):
        self.cfg = cfg or CFG()
        self.tile_paths = []
        
        for split in splits:
            split_path = Path(base_path) / "224_tiles" / split
            if split_path.exists():
                self.tile_paths.extend([
                    str(p) for p in split_path.glob("*.npy")
                ])
        
        self.transform = transform
        self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        
        print(f"Loaded {len(self.tile_paths)} tiles from {splits}")
    
    def __len__(self):
        return len(self.tile_paths)
    
    def random_channel_crop(self, image):
        """Randomly crop consecutive channels"""
        cropping_num = self.cfg.valid_chans
        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)
        return image[..., crop_indices]
    
    def __getitem__(self, idx):
        # Load and augment channels
        try:
        
            image = np.load(self.tile_paths[idx])  # (H, W, C)
        
            image = self.random_channel_crop(image)
            
            # Apply spatial augmentation
            if self.transform:
                data = self.transform(image=image)
                image = data['image'].unsqueeze(0)  # (1, C, H, W)
            
            # Rearrange to video format: (T, C, H, W)
            image = image.permute(1, 0, 2, 3)  # (C, 1, H, W) -> (T, C, H, W)
            
            # Apply normalization per frame
            image = torch.stack([self.video_transform(f) for f in image])
        except Exception as e:
            print()
            # print(f"Error loading {self.tile_paths[idx]}: {e}")
            # # Return a zero tensor on error
            # image = torch.zeros((self.cfg.valid_chans, 1, self.cfg.size, self.cfg.size), dtype=torch.float32)
        return image


class VideoMAEModel(pl.LightningModule):
    """
    Video Masked Autoencoder using VideoMAE architecture from Hugging Face.
    VideoMAE processes videos with spatiotemporal tube masking.
    """
    
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or CFG()
        self.save_hyperparameters()
        
        # Configure VideoMAE
        videomae_config = VideoMAEConfig(
            image_size=self.cfg.image_size,
            patch_size=self.cfg.patch_size,
            num_channels=self.cfg.num_channels,
            num_frames=self.cfg.input_frames,
            tubelet_size=self.cfg.tubelet_size,
            hidden_size=self.cfg.hidden_size,
            num_hidden_layers=self.cfg.num_hidden_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            intermediate_size=self.cfg.intermediate_size,
            decoder_num_hidden_layers=self.cfg.decoder_num_hidden_layers,
            decoder_hidden_size=self.cfg.decoder_hidden_size,
            decoder_num_attention_heads=self.cfg.decoder_num_attention_heads,
            decoder_intermediate_size=self.cfg.decoder_intermediate_size,
            norm_pix_loss=True,
            mask_ratio=self.cfg.mask_ratio,
        )
        
        # Initialize VideoMAE model
        self.videomae = VideoMAEForPreTraining(videomae_config)
        
        
        # # Calculate patch information
        self.num_patches_per_frame = (self.cfg.image_size // self.cfg.patch_size) ** 2
        self.num_frames_patches = self.cfg.input_frames // self.cfg.tubelet_size
        self.total_patches = self.num_patches_per_frame * self.num_frames_patches
        
        print(f"VideoMAE initialized:")
        print(f"  - Input frames: {self.cfg.input_frames}")
        print(f"  - Tubelet size: {self.cfg.tubelet_size}")
        print(f"  - Patch size: {self.cfg.patch_size}")
        print(f"  - Patches per frame: {self.num_patches_per_frame}")
        print(f"  - Frame patches: {self.num_frames_patches}")
        print(f"  - Total patches: {self.total_patches}")
        print(f"  - Mask ratio: {self.cfg.mask_ratio}")
    
    def _unpatchify(self, patches, num_frames, height, width, patch_size, tubelet_size, channels):
        """
        patches: (B, N, P) = (B, T'*H'*W', C * Ps * Ps * Ts)
        returns: (B, C, T, H, W)
        """
        B, N, P = patches.shape
        H_p = height // patch_size     # number of patches per frame (height)
        W_p = width // patch_size      # number of patches per frame (width)
        T_p = num_frames // tubelet_size

        assert N == H_p * W_p * T_p, f"Patch numbers mismatch: N={N}, expected={H_p*W_p*T_p}"

        # reshape into (B, T_p, H_p, W_p, tubelet, patch, patch, C)
        patches = patches.reshape(
            B,
            T_p, H_p, W_p,
            channels, tubelet_size,
            patch_size, patch_size
        )

        # move channels outward
        patches = patches.permute(0, 4, 1, 5, 2, 3, 6, 7)
        # now: (B, C, T_p, tubelet_size, H_p, W_p, Ps, Ps)

        # merge tubelet dimension
        patches = patches.reshape(
            B, channels, num_frames, H_p, W_p, patch_size, patch_size
        )

        # merge patch grid → full image
        video = patches.permute(0, 1, 2, 3, 5, 4, 6).reshape(
            B, channels, num_frames, height, width
        )

        return video

    def _make_bool_mask(self, batch_size, device):
        """
        Create a boolean mask tensor of shape (batch_size, seq_length).
        Each row will have the same number of masked positions (True).
        """
        seq_length = self.total_patches  # (num_frames // tubelet_size) * patches_per_frame
        num_mask = int(round(self.cfg.mask_ratio * seq_length))
        if num_mask <= 0:
            # no masking requested
            return torch.zeros((batch_size, seq_length), dtype=torch.bool, device=device)

        # Start with zeros and set num_mask positions to True per row.
        # To ensure the same number per example, we'll sample indices once and use the same pattern
        # for every sample in the batch (common practice for performance / simplicity).
        perm = torch.randperm(seq_length, device=device)
        mask_idx = perm[:num_mask]                # indices to mask
        bool_mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=device)
        bool_mask[:, mask_idx] = True
        return bool_mask

    # def forward(self, video):
    #     """
    #     video: (B, T, C, H, W) where C might be 1 (grayscale)
    #     Returns: loss, reconstructed (B, T, C, H, W), mask (bool tensor)
    #     """
    #     # Basic asserts to help debugging
    #     assert video.dim() == 5, f"Expected video to be (B, T, C, H, W), got {video.shape}"

    #     batch_size, T, C, H, W = video.shape
    #     device = video.device

    #     # If single channel, expand to 3 channels by repeating (VideoMAE expects RGB-like).
    #     if C == 1 and self.videomae.config.num_channels == 3:
    #         # video: (B, T, 1, H, W) -> (B, T, 3, H, W)
    #         video = video.repeat(1, 1, 3, 1, 1)

    #     # Make sure pixel_values are floats in the expected range / dtype (model may expect float32)
    #     video = video.to(dtype=torch.float32)

    #     # Build boolean mask expected by VideoMAE: shape (batch_size, seq_length)
    #     bool_masked_pos = self._make_bool_mask(batch_size, device)

    #     # Call the model. The HF VideoMAEForPreTraining expects pixel_values in (B, T, C, H, W)
    #     outputs = self.videomae(
    #         pixel_values=video,
    #         bool_masked_pos=bool_masked_pos,
    #     )

    #     # outputs.loss is the reconstruction loss
    #     loss = outputs.loss

    #     # recreate reconstructed video from logits using model helper if available
    #     # outputs.logits shape: (B, seq_length, tubelet_size * patch_size^2 * C)
    #     # HF model might provide .unpatchify on the module or within outputs; use model's unpatchify
    #     logits = outputs.logits  # (B, N_visible, P)
    #     mask = bool_masked_pos   # (B, N_total)

    #     B, N_total = mask.shape
    #     _, N_visible, P = logits.shape

    #     # Create full sequence filled with zeros
    #     full_seq = torch.zeros(B, N_total, P, device=logits.device, dtype=logits.dtype)

    #     # Do NOT assign anything — all patches are zero

    #     # Compute channels from patch embedding
    #     patch_vec_len = P
    #     patch_size = self.cfg.patch_size
    #     tubelet_size = self.cfg.tubelet_size
    #     channels = patch_vec_len // (patch_size * patch_size * tubelet_size)

    #     # Unpatchify
    #     reconstructed = self.unpatchify(
    #         full_seq,
    #         num_frames=self.cfg.input_frames,
    #         height=self.cfg.image_size,
    #         width=self.cfg.image_size,
    #         patch_size=self.cfg.patch_size,
    #         tubelet_size=self.cfg.tubelet_size,
    #         channels=channels,
    #     )

    #     # Convert to (B, T, C, H, W)
    #     if reconstructed.shape[1] == 3:
    #         reconstructed = reconstructed.mean(dim=1, keepdim=True)
    #     reconstructed = reconstructed.permute(0, 2, 1, 3, 4)


    #     return loss, reconstructed, bool_masked_pos
    def forward(self, video):
        """
        video: (B, T, C, H, W) where C might be 1 (grayscale)
        Returns: loss, reconstructed (B, T, C, H, W), mask (bool tensor)
        """
        assert video.dim() == 5, f"Expected video to be (B, T, C, H, W), got {video.shape}"

        batch_size, T, C, H, W = video.shape
        device = video.device

        # Store original for denormalization reference
        original_video = video.clone()

        # If single channel, expand to 3 channels
        if C == 1 and self.videomae.config.num_channels == 3:
            video = video.repeat(1, 1, 3, 1, 1)
            original_video = original_video.repeat(1, 1, 3, 1, 1)

        video = video.to(dtype=torch.float32)

        # Build boolean mask: True = masked, False = visible
        bool_masked_pos = self._make_bool_mask(batch_size, device)

        # Forward pass
        outputs = self.videomae(
            pixel_values=video,
            bool_masked_pos=bool_masked_pos,
        )

        loss = outputs.loss
        logits = outputs.logits  # (B, num_masked_patches, patch_dim)
        mask = bool_masked_pos   # (B, N_total), True = masked

        B, N_total = mask.shape
        _, N_masked, P = logits.shape

        # Compute patch parameters
        patch_size = self.cfg.patch_size
        tubelet_size = self.cfg.tubelet_size
        channels = P // (patch_size * patch_size * tubelet_size)

        # === Denormalization ===
        # When norm_pix_loss=True, the model predicts normalized patches.
        # We need to denormalize using the original video's patch statistics.
        
        # Patchify original video to get mean/var per patch
        original_patches = self._patchify(original_video)  # (B, N_total, P)
        
        # Compute mean and var for each patch
        patch_mean = original_patches.mean(dim=-1, keepdim=True)  # (B, N_total, 1)
        patch_var = original_patches.var(dim=-1, keepdim=True)    # (B, N_total, 1)
        
        # Get mean/var only for masked positions
        masked_mean = patch_mean[mask].reshape(B, N_masked, 1)  # (B, N_masked, 1)
        masked_var = patch_var[mask].reshape(B, N_masked, 1)    # (B, N_masked, 1)
        
        # Denormalize predictions: pred * sqrt(var + eps) + mean
        denorm_logits = logits * (masked_var + 1e-6).sqrt() + masked_mean

        # === Build full reconstruction ===
        # Start with original patches for visible positions
        full_seq = original_patches.clone()  # (B, N_total, P)
        
        # Replace masked positions with denormalized predictions
        full_seq[mask] = denorm_logits.reshape(-1, P)

        # Unpatchify
        reconstructed = self._unpatchify(
            full_seq,
            num_frames=self.cfg.input_frames,
            height=self.cfg.image_size,
            width=self.cfg.image_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            channels=channels,
        )

        # Convert back to original format
        if reconstructed.shape[1] == 3 and C == 1:
            reconstructed = reconstructed.mean(dim=1, keepdim=True)
        reconstructed = reconstructed.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)

        return loss, reconstructed, bool_masked_pos


    def _patchify(self, video):
        """
        Convert video to patches.
        video: (B, T, C, H, W)
        returns: (B, N, P) where N = num_patches, P = patch_dim
        """
        B, T, C, H, W = video.shape
        patch_size = self.cfg.patch_size
        tubelet_size = self.cfg.tubelet_size
        
        # Reshape to (B, C, T, H, W)
        video = video.permute(0, 2, 1, 3, 4)
        
        T_p = T // tubelet_size
        H_p = H // patch_size
        W_p = W // patch_size
        
        # Reshape: (B, C, T_p, tubelet, H_p, patch, W_p, patch)
        video = video.reshape(B, C, T_p, tubelet_size, H_p, patch_size, W_p, patch_size)
        
        # Permute to: (B, T_p, H_p, W_p, C, tubelet, patch, patch)
        video = video.permute(0, 2, 4, 6, 1, 3, 5, 7)
        
        # Flatten to patches: (B, N, P)
        patches = video.reshape(B, T_p * H_p * W_p, -1)
        
        return patches

    
    def training_step(self, batch, batch_idx):
        video = batch  # (B, T, C, H, W)
        loss, _, _ = self(video)
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        video = batch
        loss, recon, mask = self(video)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        # Store first batch for visualization
        if batch_idx == 0:
            self.val_batch_viz = (video, recon, mask)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Visualize reconstruction after validation"""
        if self.global_rank == 0 and hasattr(self, 'val_batch_viz'):
            video, recon, mask = self.val_batch_viz
            self._save_reconstruction(video, recon, mask)
        
        # Log metrics to CSV
        if self.global_rank == 0:
            self._log_metrics_to_csv()
    
    def _save_reconstruction(self, original, reconstructed, mask, num_frames=16):
        """Save reconstruction visualization"""
        orig = original[0].cpu().numpy()  # (T, C, H, W)
        recon = reconstructed[0].cpu().detach().numpy()
        
        if orig.shape[1] == 1:
            orig = orig.squeeze(1)
            recon = recon.squeeze(1)
        
        num_frames = min(num_frames, orig.shape[0])
        
        fig, axes = plt.subplots(2, num_frames, figsize=(3 * num_frames, 6))
        
        for i in range(num_frames):
            axes[0, i].imshow(orig[i], cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f"Original {i}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(recon[i], cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title(f"Reconstructed {i}")
            axes[1, i].axis('off')
        
        save_dir = Path('./results')
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f'videomae_reconstruction_epoch_{self.current_epoch}.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved reconstruction to {save_path}")
    
    def _log_metrics_to_csv(self):
        """Log training metrics to CSV file"""
        import csv
        
        val_loss = self.trainer.callback_metrics.get('val_loss', None)
        train_loss = self.trainer.callback_metrics.get('train_loss', None)
        
        log_path = "training_history_videomae.csv"
        file_exists = os.path.isfile(log_path)
        
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
            
            train_loss_val = train_loss.item() if train_loss is not None else 'N/A'
            val_loss_val = val_loss.item() if val_loss is not None else 'N/A'
            writer.writerow([self.current_epoch, train_loss_val, val_loss_val])
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # AdamW optimizer (as used in VideoMAE paper)
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Cosine annealing with warmup
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.epochs,
            eta_min=1e-6
        )
        
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=self.cfg.warmup_epochs,
            after_scheduler=cosine_scheduler
        )
        
        return [optimizer], [scheduler]

import json
from dataclasses import asdict



def get_serializable_config(cfg_class):
    """Extract only JSON-serializable attributes from CFG class"""
    config = {}
    for key, value in vars(cfg_class).items():
        if key.startswith('_'):
            continue
        # Only include serializable types
        if isinstance(value, (int, float, str, bool, type(None))):
            config[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in value):
            config[key] = value
    return config


def save_config(cfg_class, path):
    """Save config to JSON file"""
    config = get_serializable_config(cfg_class)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {path}")


def load_config_into_class(path, cfg_class):
    """Load config from JSON and update CFG class"""
    with open(path, 'r') as f:
        config = json.load(f)
    for key, value in config.items():
        setattr(cfg_class, key, value)
    return cfg_class

def main():
    """Main training function"""
    # Set seed for reproducibility

    # In main(), after creating model:
    # set_seed(CFG.seed)

    # Save config alongside checkpoints
    set_seed(CFG.seed)

    # Create datasets
    full_dataset = TileDataset(
        CFG.segment_path,
        splits=["train", "valid"],
        transform=get_transforms('train', CFG),
        cfg=CFG
    )
    
    # Split into train/val
    val_size = int(CFG.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="videomae_{epoch:03d}_{val_loss:.4f}",
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        save_last=True,
        every_n_epochs=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Initialize model and trainer
    model = VideoMAEModel(CFG)
    
    os.makedirs('checkpoints', exist_ok=True)
    save_config(CFG, 'checkpoints/config.json')

    # Create custom checkpoint I/O plugin for PyTorch 2.6+ compatibility
    checkpoint_plugin = CustomCheckpointIO()

    # Create trainer with custom checkpoint I/O
    trainer = pl.Trainer(
        max_epochs=CFG.epochs,
        accelerator="auto",
        devices=-1,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        log_every_n_steps=20,
        val_check_interval=1.0,
        gradient_clip_val=CFG.max_grad_norm,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_callback, lr_monitor],
        precision="16",  # Mixed precision training
        deterministic=True,
        plugins=[checkpoint_plugin]
    )

    # Start training - optionally resume from checkpoint
    if CFG.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {CFG.resume_from_checkpoint}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=CFG.resume_from_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.fit(model, train_loader, val_loader)

    print("Training completed!")


if __name__ == "__main__":
    main()