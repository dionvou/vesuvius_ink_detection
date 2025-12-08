import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import random
import cv2
import PIL
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torchvision.models.video import swin_transformer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from warmup_scheduler import GradualWarmupScheduler

PIL.Image.MAX_IMAGE_PIXELS = 933120000

# Add parent directory to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


class CFG:
    # ============== paths =============
    current_dir = '../'
    segment_path = './pretraining_scrolls/'

    # ============== data config =============
    start_idx = 22
    in_chans = 18
    valid_chans = 16  # channels used after augmentation

    size = 224
    tile_size = 224
    stride = tile_size // 1

    # ============== training config =============
    train_batch_size = 32  # Reduced for 224x224
    valid_batch_size = 32
    lr = 1e-4
    scheduler = 'cosine'
    epochs = 200
    warmup_epochs = 5

    # ============== model config =============
    embed_dim = 192  # Swin3D Tiny base dimension (NOT 768!)
    decoder_dim = 512
    decoder_layers = 4
    mask_ratio = 0.75

    # Patch configuration for 224x224 input
    patch_size = 16  # spatial patch size (224/16 = 14 patches per side)
    tubelet_size = 2  # temporal patch size

    # ============== validation =============
    segments = ['20240304141531']
    valid_id = '20240304141531'

    # ============== optimization =============
    min_lr = 1e-7
    weight_decay = 0.05
    max_grad_norm = 1.0
    num_workers = 16
    seed = 0

    # ============== augmentation =============
    train_aug_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.15, p=0.75),
        A.RandomBrightnessContrast(p=0.3),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        ToTensorV2(transpose_mask=True),
    ]


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug


class TileDataset(Dataset):
    def __init__(self, base_path, splits=["train"], transform=None, size=224):
        self.tile_paths = []
        self.size = size
        for split in splits:
            split_path = os.path.join(base_path, "64_tiles", split)
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

    def fourth_augment(self, image):
        """Randomly crop valid_chans from in_chans channels"""
        cropping_num = CFG.valid_chans
        start_idx = random.randint(0, CFG.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)
        return image[..., crop_indices]

    def __getitem__(self, idx):
        image = np.load(self.tile_paths[idx])  # H x W x C (64x64xC)

        # Resize from 64x64 to 224x224
        # Resize each channel separately
        h, w, c = image.shape
        if h != self.size or w != self.size:
            resized_channels = []
            for i in range(c):
                channel = cv2.resize(image[:, :, i], (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                resized_channels.append(channel)
            image = np.stack(resized_channels, axis=-1)

        image = self.fourth_augment(image)

        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)  # C x 1 x H x W

        # Rearrange to T x C x H x W format
        image = image.permute(1, 0, 2, 3)  # 1 x C x H x W -> C x 1 x H x W
        image = torch.stack([self.video_transform(f) for f in image])  # T x C x H x W
        return image


def visualize_reconstruction(original, reconstructed, sample_idx=0, num_frames=16, save_path='./results/', epoch=0):
    """Visualize original and reconstructed video frames side by side"""
    orig = original[sample_idx]     # (T, C, H, W)
    recon = reconstructed[sample_idx]  # (T, C, H, W)

    # If grayscale, squeeze channel dim
    if orig.shape[1] == 1:
        orig = orig.squeeze(1)
        recon = recon.squeeze(1)

    # Clamp and convert to numpy
    orig = orig.cpu().numpy()
    recon = recon.cpu().detach().numpy()

    # Only show first 16 frames
    num_frames = min(num_frames, orig.shape[0])

    fig, axes = plt.subplots(2, num_frames, figsize=(3 * num_frames, 6))

    for i in range(num_frames):
        # Original frame
        ax = axes[0, i]
        ax.imshow(orig[i], cmap='gray')
        ax.set_title(f"Original {i}")
        ax.axis('off')

        # Reconstructed frame
        ax = axes[1, i]
        ax.imshow(recon[i], cmap='gray')
        ax.set_title(f"Reconstructed {i}")
        ax.axis('off')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/reconstruction_224_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


class SwinMAEPretrain(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.save_hyperparameters()

        if cfg is None:
            cfg = CFG()
        self.cfg = cfg

        # Load Swin3D Tiny backbone
        print("Initializing Swin3D Tiny backbone...")
        backbone = swin_transformer.SwinTransformer3d(
            patch_size=[2, 4, 4],      # temporal=2, spatial=4x4 patches
            embed_dim=96,              # Tiny config starts at 96
            depths=[2, 2, 6, 2],       # Tiny config
            num_heads=[3, 6, 12, 24],  # heads per stage
            window_size=[8, 7, 7],     # attention window
            stochastic_depth_prob=0.1,
        )

        # Adapt first conv layer from RGB (3 channels) to grayscale (1 channel)
        old_conv = backbone.patch_embed.proj  # Conv3d(3, 96, ...)
        weight = old_conv.weight  # [96, 3, 2, 4, 4]
        bias = old_conv.bias      # [96]

        # Average across RGB → 1 channel
        new_weight = weight.sum(dim=1, keepdim=True)  # [96, 1, 2, 4, 4]

        # Replace conv
        backbone.patch_embed.proj = nn.Conv3d(
            in_channels=1,
            out_channels=96,  # Keep same output channels!
            kernel_size=(2, 4, 4),
            stride=(2, 4, 4),
            bias=True
        )

        # Load adapted weights
        backbone.patch_embed.proj.weight = nn.Parameter(new_weight)
        backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

        # Remove classification head (last 2 layers)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        # Calculate patch dimensions
        self.input_T = cfg.valid_chans  # 16 frames
        self.input_H = cfg.size  # 224
        self.input_W = cfg.size  # 224
        self.tubelet_size = cfg.tubelet_size  # 2
        self.patch_size = cfg.patch_size  # 16
        self.mask_ratio = cfg.mask_ratio  # 0.75

        # Total number of patches after patchification
        # T: 16 frames / 2 tubelet = 8 temporal patches
        # H: 224 / 16 = 14 spatial patches
        # W: 224 / 16 = 14 spatial patches
        # Total: 8 * 14 * 14 = 1568 patches
        self.N = (self.input_T // self.tubelet_size) * \
                 (self.input_H // self.patch_size) * \
                 (self.input_W // self.patch_size)
        print(f"Total patches: {self.N}")

        # Number of patches after masking (keep 25%)
        self.unmasked_patches = int((1 - self.mask_ratio) * self.input_T / self.tubelet_size) * \
                                (self.input_H // self.patch_size) * (self.input_W // self.patch_size)
        print(f"Unmasked patches: {self.unmasked_patches}/{self.N} ({self.unmasked_patches/self.N:.2%})")

        # Get actual encoder output dimension from Swin3D
        # Swin3D Tiny: 96 -> 192 -> 384 -> 768
        self.encoder_out_dim = 768

        # Decoder components
        self.decoder_embed = nn.Linear(self.encoder_out_dim, cfg.decoder_dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.N, cfg.decoder_dim) * 0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.decoder_dim,
            nhead=8,
            dim_feedforward=cfg.decoder_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=cfg.decoder_layers)

        # Predict flattened patch
        patch_dim = self.patch_size ** 2 * self.tubelet_size  # 16*16*2 = 512
        self.decoder_pred = nn.Linear(cfg.decoder_dim, patch_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Loss
        self.criterion = nn.MSELoss()


    def patchify(self, x):
        """
        Convert video to patches
        x: (B, T, C, H, W)
        Returns: (B, N, patch_dim) where patch_dim = C * tubelet * ps^2
        """
        B, T, C, H, W = x.shape
        tubelet = self.tubelet_size   # 2
        ps = self.patch_size          # 16

        # (B, T, C, H, W) → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Unfold to create patches
        x = x.unfold(2, tubelet, tubelet) \
            .unfold(3, ps, ps) \
            .unfold(4, ps, ps)
        # shape: (B, C, T_patches, H_patches, W_patches, tubelet, ps, ps)

        T_patches = x.size(2)
        H_patches = x.size(3)
        W_patches = x.size(4)

        # Reorder: (B, T_p, H_p, W_p, tubelet, ps, ps, C)
        x = x.permute(0, 2, 3, 4, 5, 6, 7, 1)

        # Flatten each patch: (B, N, C * tubelet * ps * ps)
        x = x.reshape(B, T_patches * H_patches * W_patches,
                      C * tubelet * ps * ps)
        return x


    def unpatchify(self, x, patch_shape):
        """
        Convert patches back to video
        x: (B, N, D)
        patch_shape: (pt, ph, pw) - number of patches in each dimension
        Returns: (B, C, T, H, W)
        """
        B, N, D = x.shape
        pt, ph, pw = patch_shape
        ps = self.patch_size
        tubelet = self.tubelet_size
        C = 1

        assert ph * pw * pt == N, f"Patch count mismatch: {ph}*{pw}*{pt}={ph*pw*pt} != {N}"

        x = x.view(B, pt, ph, pw, C, tubelet, ps, ps)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, pt * tubelet, ph * ps, pw * ps)
        return x


    def random_masking(self, x, mask_ratio=0.75):
        """
        Random masking with restore indices
        x: (B, N, D)
        Returns:
            x_masked: (B, n_keep, D)
            ids_keep: (B, n_keep)
            ids_masked: (B, n_mask)
            ids_restore: (B, N)
        """
        B, N, D = x.shape
        n_keep = self.unmasked_patches

        ids_keep = []
        ids_masked = []
        ids_restore = []

        for b in range(B):
            # Random permutation
            perm = torch.randperm(N, device=x.device)
            keep = perm[:n_keep]
            masked = perm[n_keep:]

            ids_keep.append(keep)
            ids_masked.append(masked)

            # Build restore index (inverse permutation)
            ids_restore_b = torch.empty_like(perm)
            ids_restore_b[perm] = torch.arange(N, device=x.device)
            ids_restore.append(ids_restore_b)

        ids_keep = torch.stack(ids_keep, dim=0)
        ids_masked = torch.stack(ids_masked, dim=0)
        ids_restore = torch.stack(ids_restore, dim=0)

        # Gather kept patches
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_masked, ids_keep, ids_masked, ids_restore


    def forward(self, x):
        B, T, C, H, W = x.shape

        # 1. Patchify input video
        x_patched = self.patchify(x)  # (B, N, patch_dim)
        N = x_patched.shape[1]

        # 2. Random masking
        x_masked, ids_keep, ids_masked, ids_restore = self.random_masking(x_patched, self.mask_ratio)

        ids_keep = ids_keep.long()
        ids_masked = ids_masked.long()
        ids_restore = ids_restore.long()

        # 3. Unpatchify visible patches for encoder input
        # For 224x224, 16 frames, mask_ratio=0.75:
        # pt = 8 * 0.25 = 2 temporal patches
        # ph = pw = sqrt(392 / 2) = 14 spatial patches per side
        pt = int((self.input_T // self.tubelet_size) * (1 - self.mask_ratio))
        ph = pw = int((self.unmasked_patches // pt) ** 0.5)
        assert ph * pw * pt == self.unmasked_patches, f"Patch grid mismatch: {ph}*{pw}*{pt} != {self.unmasked_patches}"

        x_masked_video = self.unpatchify(x_masked, (pt, ph, pw))  # (B, C, T_mask, H_mask, W_mask)

        # 4. Encoder forward on masked video
        encoder_out = self.encoder(x_masked_video)  # (B, encoder_out_dim, T', H', W')

        # Debug: print encoder output shape on first forward pass
        if not hasattr(self, '_printed_shapes'):
            print(f"Input to encoder: {x_masked_video.shape}")
            print(f"Encoder output shape: {encoder_out.shape}")
            print(f"Expected unmasked_patches: {self.unmasked_patches}")
            self._printed_shapes = True

        # 5. Handle encoder output dimensions
        if encoder_out.dim() == 5:
            # If it's (B, T, H, W, C) - need to permute to (B, C, T, H, W) first
            if encoder_out.shape[-1] == 768:  # Last dim is channel
                encoder_out = encoder_out.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)

            # Flatten spatial-temporal dimensions: (B, C, T*H*W)
            B_enc, C_enc = encoder_out.shape[:2]
            encoder_out = encoder_out.view(B_enc, C_enc, -1)  # (B, 768, T*H*W)
            encoder_out = encoder_out.transpose(1, 2)  # (B, T*H*W, 768)

            if not hasattr(self, '_printed_shapes'):
                print(f"After flatten: {encoder_out.shape}")

        # Adapt sequence length to match unmasked_patches using interpolation
        if encoder_out.shape[1] != self.unmasked_patches:
            # Interpolate along sequence dimension
            encoder_out = encoder_out.transpose(1, 2)  # (B, 768, seq_len)
            encoder_out = F.adaptive_avg_pool1d(encoder_out, self.unmasked_patches)
            encoder_out = encoder_out.transpose(1, 2)  # (B, unmasked_patches, 768)

            if not hasattr(self, '_printed_shapes2'):
                print(f"After interpolation: {encoder_out.shape}")
                self._printed_shapes2 = True

        # 6. Project encoder features to decoder dimension
        x_vis = self.decoder_embed(encoder_out)  # (B, n_visible, decoder_dim)

        # 7. Prepare mask tokens
        mask_tokens = self.mask_token.expand(B, ids_masked.shape[1], -1)

        # 8. Concatenate visible and mask tokens, then restore order
        x_ = torch.cat([x_vis, mask_tokens], dim=1)  # (B, N, decoder_dim)
        x_dec = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))

        # 9. Add positional embedding
        x_dec = x_dec + self.decoder_pos_embed

        # 10. Decode
        x_dec = self.decoder_transformer(x_dec)
        pred = self.decoder_pred(x_dec)  # (B, N, patch_dim)

        # 11. Unpatchify prediction to video
        pt = self.input_T // self.tubelet_size
        ph = pw = int((self.N // pt) ** 0.5)

        recon = self.unpatchify(pred, (pt, ph, pw))  # (B, C, T, H, W)
        recon = recon.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        return recon, x_masked_video, ids_masked, pred, x_patched


    def training_step(self, batch, batch_idx):
        x = batch  # (B, T, C, H, W)
        B = x.shape[0]

        recon, x_masked, ids_masked, pred, target = self(x)

        # Compute loss only on masked patches
        B, N, D = pred.shape
        ids_masked_exp = ids_masked.unsqueeze(-1).expand(-1, -1, D)

        pred_masked = torch.gather(pred, 1, ids_masked_exp)
        target_masked = torch.gather(target, 1, ids_masked_exp)

        loss = self.criterion(pred_masked, target_masked)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x = batch
        recon, _, ids_masked, _, _ = self(x)
        loss = self.criterion(recon, x)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        # Save first batch for visualization
        if batch_idx == 0:
            self.val_batch_for_viz = (x, recon)
        return loss


    def on_validation_epoch_end(self):
        if self.global_rank == 0 and hasattr(self, 'val_batch_for_viz'):
            x, recon = self.val_batch_for_viz
            visualize_reconstruction(x, recon, sample_idx=0, num_frames=16,
                                     save_path='./results/', epoch=self.current_epoch)


    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

        # Cosine LR with warmup
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.epochs,
            eta_min=self.cfg.min_lr
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=self.cfg.warmup_epochs,
            after_scheduler=cosine_scheduler
        )

        return [optimizer], [scheduler]


def main():
    # Set random seeds
    pl.seed_everything(CFG.seed)

    # Create datasets
    print("Loading datasets...")
    full_train_dataset = TileDataset(
        CFG.segment_path,
        splits=["train", "valid"],
        transform=get_transforms('train', CFG),
        size=CFG.size
    )

    # Split into train/val
    val_size = int(0.001 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True
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
    print("Initializing model...")
    model = SwinMAEPretrain(cfg=CFG)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints_swin_224",
        filename="swin_mae_224_{epoch:03d}",
        save_top_k=-1,
        every_n_epochs=1
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=CFG.epochs,
        accelerator="auto",
        devices=-1,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        gradient_clip_val=CFG.max_grad_norm,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_callback],
        precision="16-mixed"  # Use mixed precision for faster training
    )

    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training complete!")


if __name__ == "__main__":
    main()
