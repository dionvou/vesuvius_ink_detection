import os
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoImageProcessor
import numpy as np
import torch
import torchvision.transforms as T
import PIL
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import random
import threading
import glob

import numpy as np
import wandb
from torch.utils.data import DataLoader
import os
import random
import cv2
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler


PIL.Image.MAX_IMAGE_PIXELS = 933120000
import glob
import os
import cv2
from PIL import Image
import tifffile as tiff
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import wandb

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
import sys
sys.path.append(parent_dir)

        
class CFG:
    # ============== comp exp name =============
    current_dir = '../'
    segment_path = './pretraining_scrolls/'
    
    start_idx = 15
    in_chans = 30
    valid_chans = 24 # chans used 
    
    size = 64
    tile_size = 64
    stride = tile_size // 1
    
    train_batch_size =  256
    valid_batch_size = 20
    lr = 1e-4
    
    # ============== model cfg =============
    scheduler = 'cosine'
    epochs = 16
    warmup_factor = 10
    
    # Change the size of fragments
    frags_ratio1 = ['frag','re']
    frags_ratio2 = ['202','s4','left']
    ratio1 = 2
    ratio2 = 2
    
    # ============== valid =============
    segments = ['20240304141531'] 
    valid_id = '20240304141531'
    
    # ============== fixed =============
    min_lr = 1e-7
    weight_decay = 1e-6
    max_grad_norm = 100
    num_workers = 16
    seed = 0
        
    # ============== augmentation =============
    train_aug_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,p=0.75),
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
    def __init__(self, base_path, splits=["train"], transform=None):
        self.tile_paths = []
        for split in splits:
            split_path = os.path.join(base_path, "64_tiles", split)
            self.tile_paths += [
                os.path.join(split_path, f) 
                for f in os.listdir(split_path) if f.endswith(".npy")
            ]

        self.transform = transform
        self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32), 
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.tile_paths)
    
    def fourth_augment(self, image):
        cropping_num = CFG.valid_chans
        start_idx = random.randint(0, CFG.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)
        return image[..., crop_indices]

    def __getitem__(self, idx):
        image = np.load(self.tile_paths[idx])
        image = self.fourth_augment(image)          
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)
        image = image.permute(1,0,2,3)
        image = torch.stack([self.video_transform(f) for f in image])
        return image
    
from torch.utils.data import random_split, DataLoader
   
full_train_dataset = TileDataset(CFG.segment_path, splits=["train","valid"], transform=get_transforms('train', CFG))

val_size = int(0.001 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=8)
val_loader   = DataLoader(val_dataset, batch_size=CFG.valid_batch_size, shuffle=True, num_workers=8)

print(f"Train loader length: {len(train_loader)}")

from transformers import TimesformerModel, TimesformerConfig
import matplotlib.pyplot as plt
import numpy as np
import math

def visualize_latent_predictions(context_feats, target_feats, pred_feats, sample_idx=0, save_path='./results/jepa_latents.png', epoch=0):
    """
    Visualize latent space predictions (PCA or t-SNE projection).
    
    Args:
        context_feats: (B, n_context, D) context patch features
        target_feats: (B, n_target, D) target patch features  
        pred_feats: (B, n_target, D) predicted target features
        sample_idx: which sample to visualize
    """
    # Take one sample
    ctx = context_feats[sample_idx].detach().cpu().numpy()
    tgt = target_feats[sample_idx].detach().cpu().numpy()
    pred = pred_feats[sample_idx].detach().cpu().numpy()
    
    # Simple PCA projection to 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    all_feats = np.concatenate([ctx, tgt, pred], axis=0)
    proj = pca.fit_transform(all_feats)
    
    n_ctx = ctx.shape[0]
    n_tgt = tgt.shape[0]
    
    ctx_proj = proj[:n_ctx]
    tgt_proj = proj[n_ctx:n_ctx+n_tgt]
    pred_proj = proj[n_ctx+n_tgt:]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ctx_proj[:, 0], ctx_proj[:, 1], c='blue', label='Context', alpha=0.6, s=50)
    ax.scatter(tgt_proj[:, 0], tgt_proj[:, 1], c='green', label='Target (GT)', alpha=0.6, s=50)
    ax.scatter(pred_proj[:, 0], pred_proj[:, 1], c='red', label='Predicted', alpha=0.6, s=50, marker='x')
    
    ax.legend()
    ax.set_title(f'JEPA Latent Space - Epoch {epoch}')
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else './results/', exist_ok=True)
    plt.savefig(f'./results/jepa_latents_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


class JEPAPretrain(pl.LightningModule):
    """
    JEPA (Joint-Embedding Predictive Architecture) for video.
    
    Key differences from MAE:
    1. No pixel reconstruction - operates in latent space
    2. Context encoder processes visible patches
    3. Target encoder (EMA) processes target patches  
    4. Predictor predicts target representations from context
    """
    def __init__(self, lr=1e-4, mask_ratio=0.75, embed_dim=768, predictor_dim=768, predictor_layers=4, ema_tau=0.996):
        super().__init__()
        self.save_hyperparameters()
        self.print_shape = False
        
        # Context encoder (trainable)
        config = TimesformerConfig(
            num_frames=24,
            image_size=64,
            patch_size=8,
            num_channels=1,
            attention_type="divided_space_time",
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=768,
            num_hidden_layers=8
        )
        self.context_encoder = TimesformerModel(config)
        
        # Target encoder (EMA - exponential moving average, not trained directly)
        self.target_encoder = TimesformerModel(config)
        
        # Initialize target encoder with same weights as context encoder
        for param_context, param_target in zip(self.context_encoder.parameters(), 
                                                 self.target_encoder.parameters()):
            param_target.data.copy_(param_context.data)
            param_target.requires_grad = False  # Don't train target encoder directly
        
        self.patch_size = config.patch_size
        self.tubelet_size = 1
        self.input_T = config.num_frames
        self.input_H = config.image_size
        self.input_W = config.image_size
        self.mask_ratio = self.hparams.mask_ratio
        self.ema_tau = ema_tau
        
        self.N = self.input_T * self.input_H * self.input_W // (self.patch_size**2 * self.tubelet_size)
        print(f"Total patches: {self.N}")

        target = (1 - self.mask_ratio) * self.N
        max_y = math.floor(math.sqrt(target / self.input_T))
        candidates = [y*y * self.input_T for y in range(max_y, 0, -1)]
        closest = min(candidates, key=lambda x: abs(x - target))
        
        self.unmasked_patches = closest
        print(f"Context patches: {self.unmasked_patches}/{self.N} : {self.unmasked_patches/self.N:.2f}")

        # Predictor: predicts target representations from context
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, predictor_dim),
            nn.GELU(),
            *[nn.TransformerEncoderLayer(d_model=predictor_dim, nhead=8, 
                                         dim_feedforward=predictor_dim, batch_first=True)
              for _ in range(predictor_layers)],
            nn.Linear(predictor_dim, embed_dim)
        )
        
        # Learnable positional embeddings for predictor
        self.predictor_pos_embed = nn.Parameter(torch.randn(1, self.N, predictor_dim))
        
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder"""
        for param_context, param_target in zip(self.context_encoder.parameters(),
                                                 self.target_encoder.parameters()):
            param_target.data.mul_(self.ema_tau).add_(param_context.data, alpha=1 - self.ema_tau)
    
    def random_masking(self, x, mask_ratio=0.75):
        """
        Random masking for JEPA.
        Returns indices for context and target patches.
        """
        B, N, D = x.shape
        n_keep = self.unmasked_patches

        ids_keep = []
        ids_masked = []
        ids_restore = []

        for b in range(B):
            perm = torch.randperm(N, device=x.device)
            keep = perm[:n_keep]
            masked = perm[n_keep:]

            ids_keep.append(keep)
            ids_masked.append(masked)

            ids_restore_b = torch.empty_like(perm)
            ids_restore_b[perm] = torch.arange(N, device=x.device)
            ids_restore.append(ids_restore_b)

        ids_keep = torch.stack(ids_keep, dim=0)
        ids_masked = torch.stack(ids_masked, dim=0)
        ids_restore = torch.stack(ids_restore, dim=0)

        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_masked, ids_keep, ids_masked, ids_restore

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # 1. Patchify input
        x_patched = self.patchify(x)  # (B, N, D)
        if self.print_shape:
            print('x_patched:', x_patched.shape)
        
        N = x_patched.shape[1]
        
        # 2. Create context and target masks
        x_context, ids_context, ids_target, ids_restore = self.random_masking(x_patched, self.mask_ratio)
        
        # 3. Prepare context video (visible patches only)
        pt = T // self.tubelet_size
        ph = pw = int((self.unmasked_patches // pt) ** 0.5)
        assert ph * pw * pt == self.unmasked_patches, "Patch grid mismatch"
        
        x_context_video = self.unpatchify(x_context, (pt, ph, pw))  # (B, C, T, H_ctx, W_ctx)
        x_context_video = x_context_video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        
        # 4. Encode context patches (trainable)
        context_outputs = self.context_encoder(x_context_video, output_hidden_states=True)
        context_tokens = context_outputs.last_hidden_state[:, 1:, :]  # Skip CLS token
        
        # 5. Encode target patches with EMA encoder (for computing loss only)
        with torch.no_grad():
            full_video = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
            target_outputs = self.target_encoder(x, output_hidden_states=True)
            target_tokens = target_outputs.last_hidden_state[:, 1:, :]  # (B, N, D)
            
            # Get only target patches
            ids_target_exp = ids_target.unsqueeze(-1).expand(-1, -1, target_tokens.shape[-1])
            target_feats = torch.gather(target_tokens, 1, ids_target_exp)  # (B, n_target, D)
        
        # 6. Predict target representations from context
        # Create full sequence with zero placeholders for targets
        full_seq = torch.zeros(B, N, context_tokens.shape[-1], device=x.device)
        
        # Fill in context features
        ids_context_exp = ids_context.unsqueeze(-1).expand(-1, -1, context_tokens.shape[-1])
        full_seq.scatter_(1, ids_context_exp, context_tokens)
        
        # Add positional embeddings and predict
        pred_input = full_seq[:, :, :self.hparams.predictor_dim] if full_seq.shape[-1] > self.hparams.predictor_dim else full_seq
        pred_feats = self.predictor(pred_input)  # (B, N, embed_dim)
        
        # Extract predicted target features
        ids_target_exp = ids_target.unsqueeze(-1).expand(-1, -1, pred_feats.shape[-1])
        pred_target_feats = torch.gather(pred_feats, 1, ids_target_exp)  # (B, n_target, D)
        
        return pred_target_feats, target_feats, context_tokens, ids_target

    def patchify(self, x):
        """
        x: (B, T, C, H, W)
        Returns: patches (B, N, patch_dim)
        """
        B, T, C, H, W = x.shape
        tubelet = self.tubelet_size
        ps = self.patch_size

        x = x.permute(0, 2, 1, 3, 4)
        
        x = x.unfold(2, tubelet, tubelet) \
            .unfold(3, ps, ps) \
            .unfold(4, ps, ps)

        T_patches = x.size(2)
        H_patches = x.size(3)
        W_patches = x.size(4)

        x = x.permute(0, 2, 3, 4, 5, 6, 7, 1)
        x = x.reshape(B, T_patches * H_patches * W_patches,
                    C * tubelet * ps * ps)
        return x

    def unpatchify(self, x, patch_shape):
        B, N, D = x.shape
        pt, ph, pw = patch_shape
        ps = self.patch_size
        tubelet = self.tubelet_size
        C = 1
        assert ph * pw * pt == N, "Patch count mismatch"
        x = x.view(B, pt, ph, pw, C, tubelet, ps, ps)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, pt * tubelet, ph * ps, pw * ps)
        return x

    def training_step(self, batch, batch_idx):
        x = batch
        
        pred_target_feats, target_feats, context_feats, ids_target = self(x)
        
        # JEPA uses cosine similarity loss in latent space
        # Normalize features
        pred_norm = F.normalize(pred_target_feats, dim=-1)
        target_norm = F.normalize(target_feats, dim=-1)
        
        # Cosine similarity loss (maximize similarity)
        loss = -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))
        
        # Alternative: MSE loss in latent space
        # loss = F.mse_loss(pred_target_feats, target_feats)
        
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, sync_dist=True)
        
        # Update target encoder with EMA
        self.update_target_encoder()
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        pred_target_feats, target_feats, context_feats, ids_target = self(x)
        
        # Same loss as training
        pred_norm = F.normalize(pred_target_feats, dim=-1)
        target_norm = F.normalize(target_feats, dim=-1)
        loss = -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        if batch_idx == 0:
            self.val_batch_for_viz = (context_feats, target_feats, pred_target_feats)
        
        return loss

    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            # Log metrics to CSV
            val_loss = self.trainer.callback_metrics.get('val_loss', None)
            train_loss = self.trainer.callback_metrics.get('train_loss', None)
            
            import csv
            log_path = "training_history_jepa.csv"
            file_exists = os.path.isfile(log_path)
            
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['epoch', 'train_loss', 'val_loss'])
                
                train_loss_val = train_loss.item() if train_loss is not None else 'N/A'
                val_loss_val = val_loss.item() if val_loss is not None else 'N/A'
                writer.writerow([self.current_epoch, train_loss_val, val_loss_val])
            
            # Visualize latent predictions
            if hasattr(self, 'val_batch_for_viz'):
                ctx, tgt, pred = self.val_batch_for_viz
                visualize_latent_predictions(ctx, tgt, pred, sample_idx=0, epoch=self.current_epoch)

    def configure_optimizers(self):
        # Only optimize context encoder + predictor (NOT target encoder)
        params = list(self.context_encoder.parameters()) + list(self.predictor.parameters())
        
        optimizer = AdamW(
            params,
            lr=1.5e-4,
            betas=(0.9, 0.95),
            weight_decay=0.05,
        )

        total_epochs = 100
        warmup_epochs = 10
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6
        )
        
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=warmup_epochs,
            after_scheduler=cosine_scheduler
        )

        return [optimizer], [scheduler]

if __name__ == "__main__":
    # Training setup
    torch.set_float32_matmul_precision('medium')

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_jepa",
        filename="64_jepa_{epoch}",
        save_top_k=-1,
        every_n_epochs=1
    )

    model = JEPAPretrain(ema_tau=0.996)

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=-1,
        precision="16",
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_callback],
    )

    trainer.validate(model, val_loader, verbose=True)
    trainer.fit(model, train_loader, val_loader)