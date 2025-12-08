# import os
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist

# from transformers import AutoImageProcessor
# import numpy as np
# import torch
# import torchvision.transforms as T
# import PIL
# import torch.nn as nn

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger

# import random
# import threading
# import glob

# import numpy as np
# import wandb
# from torch.utils.data import DataLoader
# import os
# import random
# import cv2
# import numpy as np
# from tqdm.auto import tqdm
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# import segmentation_models_pytorch as smp
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from torch.utils.data import DataLoader, Dataset
# import torch.nn as nn
# import torch
# from warmup_scheduler import GradualWarmupScheduler


# PIL.Image.MAX_IMAGE_PIXELS = 933120000
# import glob
# import os
# import cv2
# from PIL import Image
# import tifffile as tiff
# import numpy as np
# from scipy import ndimage
# from tqdm import tqdm
# import random
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import torch
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import _LRScheduler
# from torch.optim.lr_scheduler import LinearLR
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.optim import AdamW
# import wandb

# current_dir = os.path.dirname(__file__)
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# import sys
# sys.path.append(parent_dir)

        
# class CFG:
#     # ============== comp exp name =============
#     current_dir = '../'
#     segment_path = './pretraining_scrolls/'
    
#     start_idx = 15
#     in_chans = 30
#     valid_chans = 24 # chans used 
    
#     size = 64
#     tile_size = 64
#     stride = tile_size // 1
    
#     train_batch_size =  256
#     valid_batch_size = 20
#     lr = 1e-4
    
#     # ============== model cfg =============
#     scheduler = 'cosine'
#     epochs = 16
#     warmup_factor = 10
    
#     # Change the size of fragments
#     frags_ratio1 = ['frag','re']
#     frags_ratio2 = ['202','s4','left']
#     ratio1 = 2
#     ratio2 = 2
    
#     # ============== valid =============
#     segments = ['20240304141531'] 
#     valid_id = '20240304141531'
    
#     # ============== fixed =============
#     min_lr = 1e-7
#     weight_decay = 1e-6
#     max_grad_norm = 100
#     num_workers = 16
#     seed = 0
        
#     # ============== augmentation =============
#     train_aug_list = [
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,p=0.75),
#         ToTensorV2(transpose_mask=True),
#     ]

#     valid_aug_list = [
#         ToTensorV2(transpose_mask=True),  
#     ]
    
# def get_transforms(data, cfg):
#     if data == 'train':
#         aug = A.Compose(cfg.train_aug_list)
#     elif data == 'valid':
#         aug = A.Compose(cfg.valid_aug_list)
#     return aug   

# class TileDataset(Dataset):
#     def __init__(self, base_path, splits=["train"], transform=None):
#         self.tile_paths = []
#         for split in splits:
#             split_path = os.path.join(base_path, "64_tiles", split)
#             self.tile_paths += [
#                 os.path.join(split_path, f) 
#                 for f in os.listdir(split_path) if f.endswith(".npy")
#             ]

#         self.transform = transform
#         self.video_transform = T.Compose([
#             T.ConvertImageDtype(torch.float32), 
#             T.Normalize(mean=[0.5], std=[0.5])
#         ])

#     def __len__(self):
#         return len(self.tile_paths)
    
#     def fourth_augment(self, image):
#         cropping_num = CFG.valid_chans
#         start_idx = random.randint(0, CFG.in_chans - cropping_num)
#         crop_indices = np.arange(start_idx, start_idx + cropping_num)
#         return image[..., crop_indices]

#     def __getitem__(self, idx):
#         image = np.load(self.tile_paths[idx])
#         image = self.fourth_augment(image)          
#         if self.transform:
#             data = self.transform(image=image)
#             image = data['image'].unsqueeze(0)
#         image = image.permute(1,0,2,3)
#         image = torch.stack([self.video_transform(f) for f in image])
#         return image
    
# from torch.utils.data import random_split, DataLoader
   
# full_train_dataset = TileDataset(CFG.segment_path, splits=["train","valid"], transform=get_transforms('train', CFG))

# val_size = int(0.001 * len(full_train_dataset))
# train_size = len(full_train_dataset) - val_size

# train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=8)
# val_loader   = DataLoader(val_dataset, batch_size=CFG.valid_batch_size, shuffle=True, num_workers=8)

# print(f"Train loader length: {len(train_loader)}")

# from transformers import TimesformerModel, TimesformerConfig
# import matplotlib.pyplot as plt
# import numpy as np
# import math

# def visualize_latent_predictions(context_feats, target_feats, pred_feats, sample_idx=0, save_path='./results/jepa_latents.png', epoch=0):
#     """
#     Visualize latent space predictions (PCA or t-SNE projection).
    
#     Args:
#         context_feats: (B, n_context, D) context patch features
#         target_feats: (B, n_target, D) target patch features  
#         pred_feats: (B, n_target, D) predicted target features
#         sample_idx: which sample to visualize
#     """
#     # Take one sample
#     ctx = context_feats[sample_idx].detach().cpu().numpy()
#     tgt = target_feats[sample_idx].detach().cpu().numpy()
#     pred = pred_feats[sample_idx].detach().cpu().numpy()
    
#     # Simple PCA projection to 2D
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
    
#     all_feats = np.concatenate([ctx, tgt, pred], axis=0)
#     proj = pca.fit_transform(all_feats)
    
#     n_ctx = ctx.shape[0]
#     n_tgt = tgt.shape[0]
    
#     ctx_proj = proj[:n_ctx]
#     tgt_proj = proj[n_ctx:n_ctx+n_tgt]
#     pred_proj = proj[n_ctx+n_tgt:]
    
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.scatter(ctx_proj[:, 0], ctx_proj[:, 1], c='blue', label='Context', alpha=0.6, s=50)
#     ax.scatter(tgt_proj[:, 0], tgt_proj[:, 1], c='green', label='Target (GT)', alpha=0.6, s=50)
#     ax.scatter(pred_proj[:, 0], pred_proj[:, 1], c='red', label='Predicted', alpha=0.6, s=50, marker='x')
    
#     ax.legend()
#     ax.set_title(f'JEPA Latent Space - Epoch {epoch}')
    
#     os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else './results/', exist_ok=True)
#     plt.savefig(f'./results/jepa_latents_{epoch}.png', dpi=300, bbox_inches='tight')
#     plt.close(fig)


# class JEPAPretrain(pl.LightningModule):
#     """
#     JEPA (Joint-Embedding Predictive Architecture) for video.
    
#     Key differences from MAE:
#     1. No pixel reconstruction - operates in latent space
#     2. Context encoder processes visible patches
#     3. Target encoder (EMA) processes target patches  
#     4. Predictor predicts target representations from context
#     """
#     def __init__(self, lr=1e-4, mask_ratio=0.75, embed_dim=768, predictor_dim=768, predictor_layers=4, ema_tau=0.996):
#         super().__init__()
#         self.save_hyperparameters()
#         self.print_shape = False
        
#         # Context encoder (trainable)
#         config = TimesformerConfig(
#             num_frames=24,
#             image_size=64,
#             patch_size=8,
#             num_channels=1,
#             attention_type="divided_space_time",
#             hidden_size=768,
#             num_attention_heads=12,
#             intermediate_size=768,
#             num_hidden_layers=8
#         )
#         self.context_encoder = TimesformerModel(config)
        
#         # Target encoder (EMA - exponential moving average, not trained directly)
#         self.target_encoder = TimesformerModel(config)
        
#         # Initialize target encoder with same weights as context encoder
#         for param_context, param_target in zip(self.context_encoder.parameters(), 
#                                                  self.target_encoder.parameters()):
#             param_target.data.copy_(param_context.data)
#             param_target.requires_grad = False  # Don't train target encoder directly
        
#         self.patch_size = config.patch_size
#         self.tubelet_size = 1
#         self.input_T = config.num_frames
#         self.input_H = config.image_size
#         self.input_W = config.image_size
#         self.mask_ratio = self.hparams.mask_ratio
#         self.ema_tau = ema_tau
        
#         self.N = self.input_T * self.input_H * self.input_W // (self.patch_size**2 * self.tubelet_size)
#         print(f"Total patches: {self.N}")

#         target = (1 - self.mask_ratio) * self.N
#         max_y = math.floor(math.sqrt(target / self.input_T))
#         candidates = [y*y * self.input_T for y in range(max_y, 0, -1)]
#         closest = min(candidates, key=lambda x: abs(x - target))
        
#         self.unmasked_patches = closest
#         print(f"Context patches: {self.unmasked_patches}/{self.N} : {self.unmasked_patches/self.N:.2f}")

#         # Predictor: predicts target representations from context
#         self.predictor = nn.Sequential(
#             nn.Linear(embed_dim, predictor_dim),
#             nn.GELU(),
#             *[nn.TransformerEncoderLayer(d_model=predictor_dim, nhead=8, 
#                                          dim_feedforward=predictor_dim, batch_first=True)
#               for _ in range(predictor_layers)],
#             nn.Linear(predictor_dim, embed_dim)
#         )
        
#         # Learnable positional embeddings for predictor
#         self.predictor_pos_embed = nn.Parameter(torch.randn(1, self.N, predictor_dim))
        
#     @torch.no_grad()
#     def update_target_encoder(self):
#         """EMA update of target encoder"""
#         for param_context, param_target in zip(self.context_encoder.parameters(),
#                                                  self.target_encoder.parameters()):
#             param_target.data.mul_(self.ema_tau).add_(param_context.data, alpha=1 - self.ema_tau)
    
#     def random_masking(self, x, mask_ratio=0.75):
#         """
#         Random masking for JEPA.
#         Returns indices for context and target patches.
#         """
#         B, N, D = x.shape
#         n_keep = self.unmasked_patches

#         ids_keep = []
#         ids_masked = []
#         ids_restore = []

#         for b in range(B):
#             perm = torch.randperm(N, device=x.device)
#             keep = perm[:n_keep]
#             masked = perm[n_keep:]

#             ids_keep.append(keep)
#             ids_masked.append(masked)

#             ids_restore_b = torch.empty_like(perm)
#             ids_restore_b[perm] = torch.arange(N, device=x.device)
#             ids_restore.append(ids_restore_b)

#         ids_keep = torch.stack(ids_keep, dim=0)
#         ids_masked = torch.stack(ids_masked, dim=0)
#         ids_restore = torch.stack(ids_restore, dim=0)

#         x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

#         return x_masked, ids_keep, ids_masked, ids_restore

#     def forward(self, x):
#         B, T, C, H, W = x.shape
        
#         # 1. Patchify input
#         x_patched = self.patchify(x)  # (B, N, D)
#         if self.print_shape:
#             print('x_patched:', x_patched.shape)
        
#         N = x_patched.shape[1]
        
#         # 2. Create context and target masks
#         x_context, ids_context, ids_target, ids_restore = self.random_masking(x_patched, self.mask_ratio)
        
#         # 3. Prepare context video (visible patches only)
#         pt = T // self.tubelet_size
#         ph = pw = int((self.unmasked_patches // pt) ** 0.5)
#         assert ph * pw * pt == self.unmasked_patches, "Patch grid mismatch"
        
#         x_context_video = self.unpatchify(x_context, (pt, ph, pw))  # (B, C, T, H_ctx, W_ctx)
#         x_context_video = x_context_video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        
#         # 4. Encode context patches (trainable)
#         context_outputs = self.context_encoder(x_context_video, output_hidden_states=True)
#         context_tokens = context_outputs.last_hidden_state[:, 1:, :]  # Skip CLS token
        
#         # 5. Encode target patches with EMA encoder (for computing loss only)
#         with torch.no_grad():
#             full_video = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
#             target_outputs = self.target_encoder(x, output_hidden_states=True)
#             target_tokens = target_outputs.last_hidden_state[:, 1:, :]  # (B, N, D)
            
#             # Get only target patches
#             ids_target_exp = ids_target.unsqueeze(-1).expand(-1, -1, target_tokens.shape[-1])
#             target_feats = torch.gather(target_tokens, 1, ids_target_exp)  # (B, n_target, D)
        
#         # 6. Predict target representations from context
#         # Create full sequence with zero placeholders for targets
#         full_seq = torch.zeros(B, N, context_tokens.shape[-1], device=x.device)
        
#         # Fill in context features
#         ids_context_exp = ids_context.unsqueeze(-1).expand(-1, -1, context_tokens.shape[-1])
#         full_seq.scatter_(1, ids_context_exp, context_tokens)
        
#         # Add positional embeddings and predict
#         pred_input = full_seq[:, :, :self.hparams.predictor_dim] if full_seq.shape[-1] > self.hparams.predictor_dim else full_seq
#         pred_feats = self.predictor(pred_input)  # (B, N, embed_dim)
        
#         # Extract predicted target features
#         ids_target_exp = ids_target.unsqueeze(-1).expand(-1, -1, pred_feats.shape[-1])
#         pred_target_feats = torch.gather(pred_feats, 1, ids_target_exp)  # (B, n_target, D)
        
#         return pred_target_feats, target_feats, context_tokens, ids_target

#     def patchify(self, x):
#         """
#         x: (B, T, C, H, W)
#         Returns: patches (B, N, patch_dim)
#         """
#         B, T, C, H, W = x.shape
#         tubelet = self.tubelet_size
#         ps = self.patch_size

#         x = x.permute(0, 2, 1, 3, 4)
        
#         x = x.unfold(2, tubelet, tubelet) \
#             .unfold(3, ps, ps) \
#             .unfold(4, ps, ps)

#         T_patches = x.size(2)
#         H_patches = x.size(3)
#         W_patches = x.size(4)

#         x = x.permute(0, 2, 3, 4, 5, 6, 7, 1)
#         x = x.reshape(B, T_patches * H_patches * W_patches,
#                     C * tubelet * ps * ps)
#         return x

#     def unpatchify(self, x, patch_shape):
#         B, N, D = x.shape
#         pt, ph, pw = patch_shape
#         ps = self.patch_size
#         tubelet = self.tubelet_size
#         C = 1
#         assert ph * pw * pt == N, "Patch count mismatch"
#         x = x.view(B, pt, ph, pw, C, tubelet, ps, ps)
#         x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
#         x = x.view(B, C, pt * tubelet, ph * ps, pw * ps)
#         return x

#     def training_step(self, batch, batch_idx):
#         x = batch
        
#         pred_target_feats, target_feats, context_feats, ids_target = self(x)
        
#         # JEPA uses cosine similarity loss in latent space
#         # Normalize features
#         pred_norm = F.normalize(pred_target_feats, dim=-1)
#         target_norm = F.normalize(target_feats, dim=-1)
        
#         # Cosine similarity loss (maximize similarity)
#         loss = -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))
        
#         # Alternative: MSE loss in latent space
#         # loss = F.mse_loss(pred_target_feats, target_feats)
        
#         self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
#         self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True, sync_dist=True)
        
#         # Update target encoder with EMA
#         self.update_target_encoder()
        
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x = batch
#         pred_target_feats, target_feats, context_feats, ids_target = self(x)
        
#         # Same loss as training
#         pred_norm = F.normalize(pred_target_feats, dim=-1)
#         target_norm = F.normalize(target_feats, dim=-1)
#         loss = -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))
        
#         self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
#         if batch_idx == 0:
#             self.val_batch_for_viz = (context_feats, target_feats, pred_target_feats)
        
#         return loss

#     def on_validation_epoch_end(self):
#         if self.global_rank == 0:
#             # Log metrics to CSV
#             val_loss = self.trainer.callback_metrics.get('val_loss', None)
#             train_loss = self.trainer.callback_metrics.get('train_loss', None)
            
#             import csv
#             log_path = "training_history_jepa.csv"
#             file_exists = os.path.isfile(log_path)
            
#             with open(log_path, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 if not file_exists:
#                     writer.writerow(['epoch', 'train_loss', 'val_loss'])
                
#                 train_loss_val = train_loss.item() if train_loss is not None else 'N/A'
#                 val_loss_val = val_loss.item() if val_loss is not None else 'N/A'
#                 writer.writerow([self.current_epoch, train_loss_val, val_loss_val])
            
#             # Visualize latent predictions
#             if hasattr(self, 'val_batch_for_viz'):
#                 ctx, tgt, pred = self.val_batch_for_viz
#                 visualize_latent_predictions(ctx, tgt, pred, sample_idx=0, epoch=self.current_epoch)

#     def configure_optimizers(self):
#         # Only optimize context encoder + predictor (NOT target encoder)
#         params = list(self.context_encoder.parameters()) + list(self.predictor.parameters())
        
#         optimizer = AdamW(
#             params,
#             lr=1.5e-4,
#             betas=(0.9, 0.95),
#             weight_decay=0.05,
#         )

#         total_epochs = 100
#         warmup_epochs = 10
        
#         cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, 
#             T_max=total_epochs - warmup_epochs,
#             eta_min=1e-6
#         )
        
#         scheduler = GradualWarmupScheduler(
#             optimizer,
#             multiplier=1,
#             total_epoch=warmup_epochs,
#             after_scheduler=cosine_scheduler
#         )

#         return [optimizer], [scheduler]

# if __name__ == "__main__":
#     # Training setup
#     torch.set_float32_matmul_precision('medium')

#     checkpoint_callback = ModelCheckpoint(
#         dirpath="checkpoints_jepa",
#         filename="64_jepa_{epoch}",
#         save_top_k=-1,
#         every_n_epochs=1
#     )

#     model = JEPAPretrain(ema_tau=0.996)

#     trainer = pl.Trainer(
#         max_epochs=100,
#         accelerator="auto",
#         devices=-1,
#         precision="16",
#         log_every_n_steps=20,
#         check_val_every_n_epoch=1,
#         gradient_clip_val=1.0,
#         gradient_clip_algorithm="norm",
#         callbacks=[checkpoint_callback],
#     )

#     trainer.validate(model, val_loader, verbose=True)
#     trainer.fit(model, train_loader, val_loader)
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
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from warmup_scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt
from PIL import Image

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = 933120000


class CFG:
    """Centralized configuration for V-JEPA"""
    # Paths
    current_dir = '../'
    segment_path = './pretraining_scrolls/'
    
    # Model architecture
    start_idx = 22
    in_chans = 18
    valid_chans = 16
    size = 64
    tile_size = 64
    
    # Training hyperparameters
    train_batch_size = 128  # V-JEPA can use larger batches
    valid_batch_size = 128
    lr = 1.5e-4
    epochs = 200
    warmup_epochs = 40  # V-JEPA uses longer warmup
    
    # V-JEPA specific
    pred_depth = 12  # Predictor depth
    pred_embed_dim = 384  # Predictor embedding dimension
    
    # Encoder Architecture (Vision Transformer)
    image_size = size
    patch_size = 8
    num_channels = 1
    embed_dim = 384  # Encoder embedding dimension
    depth = 12  # Encoder depth
    num_heads = 6
    mlp_ratio = 4.0
    
    # Video dimensions
    input_frames = 16
    tubelet_size = 2
    
    # Target Encoder (EMA)
    momentum_teacher = 0.996  # EMA momentum for target encoder
    
    # Masking strategy (spatiotemporal blocks)
    num_mask_patches = 4  # Number of mask regions
    mask_aspect_ratio = (0.75, 1.5)  # Aspect ratio range for masks
    mask_scale = (0.15, 0.2)  # Scale range for each mask
    
    # Optimizer
    weight_decay = 0.05
    max_grad_norm = 1.0
    
    # Data
    val_split = 0.0005
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
            split_path = Path(base_path) / "64_tiles" / split
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
            print(f"Error loading {self.tile_paths[idx]}: {e}")
            image = torch.zeros((self.cfg.valid_chans, 1, self.cfg.size, self.cfg.size), 
                              dtype=torch.float32)
        
        return image


class PatchEmbed(nn.Module):
    """Video to Patch Embedding with spatiotemporal tubes"""
    
    def __init__(self, img_size=64, patch_size=8, tubelet_size=2, 
                 in_chans=1, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        
        # 3D convolution for spatiotemporal patch embedding
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Rearrange to (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        
        # Apply 3D convolution
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        
        # Flatten spatial and temporal dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        return x


class Attention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP with GELU activation"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer Encoder for V-JEPA"""
    
    def __init__(self, img_size=64, patch_size=8, tubelet_size=2, in_chans=1,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, tubelet_size=tubelet_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        
        num_frames_patches = 16 // tubelet_size  # Assuming 16 input frames
        num_patches = self.patch_embed.num_patches_per_frame * num_frames_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask=None):
        # x: (B, T, C, H, W)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply mask if provided (mask out certain patches)
        if mask is not None:
            # mask: (B, N) boolean, True means keep, False means mask
            # We keep cls token always (index 0)
            mask_with_cls = torch.cat([torch.ones(B, 1, device=mask.device, dtype=torch.bool), mask], dim=1)
            x = x[mask_with_cls].reshape(B, -1, x.shape[-1])
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x


class Predictor(nn.Module):
    """Predictor network for V-JEPA"""
    
    def __init__(self, embed_dim=384, pred_embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.):
        super().__init__()
        
        # Project encoder dimension to predictor dimension if different
        if embed_dim != pred_embed_dim:
            self.proj = nn.Linear(embed_dim, pred_embed_dim)
        else:
            self.proj = nn.Identity()
        
        # Mask token for masked regions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_embed_dim))
        
        # Predictor blocks
        self.blocks = nn.ModuleList([
            Block(pred_embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(pred_embed_dim)
        
        # Project back to encoder dimension
        self.pred_proj = nn.Linear(pred_embed_dim, embed_dim)
        
        # Initialize
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask_indices, total_patches):
        """
        x: (B, N_visible, embed_dim) - context (visible) patches
        mask_indices: (B, N_masked) - indices of masked patches
        total_patches: total number of patches (including cls token)
        """
        B, N_visible, _ = x.shape
        
        # Project to predictor dimension
        x = self.proj(x)
        
        # Create full sequence with mask tokens
        N_masked = mask_indices.shape[1]
        mask_tokens = self.mask_token.expand(B, N_masked, -1)
        
        # Combine context and mask tokens
        # We need to place them in correct positions
        full_seq = torch.zeros(B, total_patches, x.shape[-1], device=x.device, dtype=x.dtype)
        
        # This is simplified - in practice, need proper position handling
        full_seq[:, :N_visible] = x
        full_seq[:, N_visible:N_visible + N_masked] = mask_tokens
        
        # Apply predictor blocks
        for block in self.blocks:
            full_seq = block(full_seq)
        
        full_seq = self.norm(full_seq)
        
        # Extract predictions for masked positions
        predictions = full_seq[:, N_visible:N_visible + N_masked]
        
        # Project back to encoder dimension
        predictions = self.pred_proj(predictions)
        
        return predictions


class VJEPAModel(pl.LightningModule):
    """
    V-JEPA: Video Joint-Embedding Predictive Architecture
    
    Predicts representations of masked spatiotemporal regions from context regions.
    Uses a target encoder (EMA of context encoder) to compute target representations.
    """
    
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or CFG()
        self.save_hyperparameters()
        
        # Context encoder (trainable)
        self.context_encoder = VisionTransformerEncoder(
            img_size=self.cfg.image_size,
            patch_size=self.cfg.patch_size,
            tubelet_size=self.cfg.tubelet_size,
            in_chans=self.cfg.num_channels,
            embed_dim=self.cfg.embed_dim,
            depth=self.cfg.depth,
            num_heads=self.cfg.num_heads,
            mlp_ratio=self.cfg.mlp_ratio
        )
        
        # Target encoder (EMA of context encoder, not directly trained)
        self.target_encoder = VisionTransformerEncoder(
            img_size=self.cfg.image_size,
            patch_size=self.cfg.patch_size,
            tubelet_size=self.cfg.tubelet_size,
            in_chans=self.cfg.num_channels,
            embed_dim=self.cfg.embed_dim,
            depth=self.cfg.depth,
            num_heads=self.cfg.num_heads,
            mlp_ratio=self.cfg.mlp_ratio
        )
        
        # Initialize target encoder with context encoder weights
        for param_q, param_k in zip(self.context_encoder.parameters(), 
                                    self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Target encoder is not trained via backprop
        
        # Predictor
        self.predictor = Predictor(
            embed_dim=self.cfg.embed_dim,
            pred_embed_dim=self.cfg.pred_embed_dim,
            depth=self.cfg.pred_depth,
            num_heads=self.cfg.num_heads,
            mlp_ratio=self.cfg.mlp_ratio
        )
        
        # Calculate patch information
        self.num_patches_per_frame = (self.cfg.image_size // self.cfg.patch_size) ** 2
        self.num_frames_patches = self.cfg.input_frames // self.cfg.tubelet_size
        self.total_patches = self.num_patches_per_frame * self.num_frames_patches
        
        print(f"V-JEPA initialized:")
        print(f"  - Input frames: {self.cfg.input_frames}")
        print(f"  - Tubelet size: {self.cfg.tubelet_size}")
        print(f"  - Patch size: {self.cfg.patch_size}")
        print(f"  - Patches per frame: {self.num_patches_per_frame}")
        print(f"  - Frame patches: {self.num_frames_patches}")
        print(f"  - Total patches: {self.total_patches}")
        print(f"  - Encoder embed dim: {self.cfg.embed_dim}")
        print(f"  - Predictor embed dim: {self.cfg.pred_embed_dim}")
    
    @torch.no_grad()
    def _update_target_encoder(self):
        """Update target encoder using EMA"""
        for param_q, param_k in zip(self.context_encoder.parameters(), 
                                    self.target_encoder.parameters()):
            param_k.data = param_k.data * self.cfg.momentum_teacher + \
                          param_q.data * (1. - self.cfg.momentum_teacher)
    
    def _sample_block_mask(self, batch_size, device):
        """
        Sample block-based mask for V-JEPA.
        Returns context_mask (visible) and target_mask (to predict).
        
        context_mask: (B, N) boolean, True = visible to context encoder
        target_mask: (B, N) boolean, True = predict these positions
        """
        H_patches = W_patches = self.cfg.image_size // self.cfg.patch_size
        T_patches = self.cfg.input_frames // self.cfg.tubelet_size
        N = self.total_patches
        
        context_mask = torch.ones(batch_size, N, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            # Sample multiple mask blocks
            for _ in range(self.cfg.num_mask_patches):
                # Sample aspect ratio and scale
                aspect_ratio = np.random.uniform(*self.cfg.mask_aspect_ratio)
                scale = np.random.uniform(*self.cfg.mask_scale)
                
                # Calculate block size
                block_area = scale * N
                block_h = int(np.sqrt(block_area / aspect_ratio))
                block_w = int(aspect_ratio * block_h)
                block_t = max(1, T_patches // 4)  # Temporal extent
                
                # Ensure block fits
                block_h = min(block_h, H_patches)
                block_w = min(block_w, W_patches)
                block_t = min(block_t, T_patches)
                
                # Sample starting position
                start_h = np.random.randint(0, H_patches - block_h + 1) if block_h < H_patches else 0
                start_w = np.random.randint(0, W_patches - block_w + 1) if block_w < W_patches else 0
                start_t = np.random.randint(0, T_patches - block_t + 1) if block_t < T_patches else 0
                
                # Create mask for this block
                for t in range(start_t, start_t + block_t):
                    for h in range(start_h, start_h + block_h):
                        for w in range(start_w, start_w + block_w):
                            idx = t * (H_patches * W_patches) + h * W_patches + w
                            if idx < N:
                                context_mask[b, idx] = False
                                target_mask[b, idx] = True
        
        return context_mask, target_mask
    
    def forward(self, video):
        """
        video: (B, T, C, H, W)
        Returns: loss
        """
        B = video.shape[0]
        device = video.device
        
        # Sample block mask
        context_mask, target_mask = self._sample_block_mask(B, device)
        
        # Encode context (visible patches)
        context_tokens = self.context_encoder(video, mask=context_mask)
        # context_tokens: (B, N_visible + 1, embed_dim) [+1 for cls token]
        
        # Get target representations (no gradients)
        with torch.no_grad():
            target_tokens = self.target_encoder(video, mask=None)
            # Extract target tokens for masked positions only
            # target_tokens: (B, N_total + 1, embed_dim)
            # We want target_tokens at masked positions (skip cls token)
            target_tokens = target_tokens[:, 1:, :]  # Remove cls token
            masked_targets = target_tokens[target_mask]  # (total_masked, embed_dim)
            masked_targets = masked_targets.reshape(B, -1, self.cfg.embed_dim)
        
        # Get mask indices
        mask_indices = target_mask.nonzero(as_tuple=False)[:, 1].reshape(B, -1)
        
        # Predict masked tokens
        predictions = self.predictor(
            context_tokens, 
            mask_indices, 
            self.total_patches + 1  # +1 for cls token
        )
        # predictions: (B, N_masked, embed_dim)
        
        # Compute loss (L2 loss in representation space)
        loss = F.mse_loss(predictions, masked_targets)
        
        return loss, predictions, masked_targets, context_mask, target_mask
    
    def training_step(self, batch, batch_idx):
        video = batch
        loss, _, _, _, _ = self(video)
        
        # Update target encoder
        self._update_target_encoder()
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True)
        self.log("momentum", self.cfg.momentum_teacher, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        video = batch
        loss, preds, targets, context_mask, target_mask = self(video)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        # Compute prediction quality metrics
        with torch.no_grad():
            # Cosine similarity
            preds_norm = F.normalize(preds, dim=-1)
            targets_norm = F.normalize(targets, dim=-1)
            cos_sim = (preds_norm * targets_norm).sum(dim=-1).mean()
            self.log('val_cos_sim', cos_sim, prog_bar=True, sync_dist=True)
        
        # Store first batch for visualization
        if batch_idx == 0:
            self.val_batch_viz = (video, context_mask, target_mask)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Visualize masking strategy after validation"""
        if self.global_rank == 0 and hasattr(self, 'val_batch_viz'):
            video, context_mask, target_mask = self.val_batch_viz
            self._save_mask_visualization(video, context_mask, target_mask)
        
        # Log metrics to CSV
        if self.global_rank == 0:
            self._log_metrics_to_csv()
    
    def _save_mask_visualization(self, video, context_mask, target_mask, num_frames=16):
        """Visualize the masking strategy"""
        # Take first sample
        vid = video[0].cpu().numpy()  # (T, C, H, W)
        ctx_mask = context_mask[0].cpu().numpy()  # (N,)
        tgt_mask = target_mask[0].cpu().numpy()  # (N,)
        
        if vid.shape[1] == 1:
            vid = vid.squeeze(1)  # (T, H, W)
        
        H_patches = W_patches = self.cfg.image_size // self.cfg.patch_size
        T_patches = self.cfg.input_frames // self.cfg.tubelet_size
        
        num_frames = min(num_frames, vid.shape[0])
        
        fig, axes = plt.subplots(3, num_frames, figsize=(3 * num_frames, 9))
        
        for i in range(num_frames):
            # Original frame
            axes[0, i].imshow(vid[i], cmap='gray', vmin=-1, vmax=1)
            axes[0, i].set_title(f"Original {i}")
            axes[0, i].axis('off')
            
            # Context mask overlay
            frame_with_context = vid[i].copy()
            t_idx = i // self.cfg.tubelet_size
            if t_idx < T_patches:
                for h in range(H_patches):
                    for w in range(W_patches):
                        patch_idx = t_idx * (H_patches * W_patches) + h * W_patches + w
                        if patch_idx < len(ctx_mask) and not ctx_mask[patch_idx]:
                            # Mask this patch (make it darker)
                            h_start = h * self.cfg.patch_size
                            w_start = w * self.cfg.patch_size
                            frame_with_context[h_start:h_start+self.cfg.patch_size,
                                             w_start:w_start+self.cfg.patch_size] *= 0.3
            
            axes[1, i].imshow(frame_with_context, cmap='gray', vmin=-1, vmax=1)
            axes[1, i].set_title(f"Context {i}")
            axes[1, i].axis('off')
            
            # Target mask overlay
            frame_with_target = vid[i].copy() * 0.3
            if t_idx < T_patches:
                for h in range(H_patches):
                    for w in range(W_patches):
                        patch_idx = t_idx * (H_patches * W_patches) + h * W_patches + w
                        if patch_idx < len(tgt_mask) and tgt_mask[patch_idx]:
                            # Highlight this patch
                            h_start = h * self.cfg.patch_size
                            w_start = w * self.cfg.patch_size
                            frame_with_target[h_start:h_start+self.cfg.patch_size,
                                            w_start:w_start+self.cfg.patch_size] = vid[i][h_start:h_start+self.cfg.patch_size,
                                                                                           w_start:w_start+self.cfg.patch_size]
            
            axes[2, i].imshow(frame_with_target, cmap='gray', vmin=-1, vmax=1)
            axes[2, i].set_title(f"Target {i}")
            axes[2, i].axis('off')
        
        save_dir = Path('./results')
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f'vjepa_masking_epoch_{self.current_epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved mask visualization to {save_path}")
    
    def _log_metrics_to_csv(self):
        """Log training metrics to CSV file"""
        import csv
        
        val_loss = self.trainer.callback_metrics.get('val_loss', None)
        train_loss = self.trainer.callback_metrics.get('train_loss', None)
        cos_sim = self.trainer.callback_metrics.get('val_cos_sim', None)
        
        log_path = "training_history_vjepa.csv"
        file_exists = os.path.isfile(log_path)
        
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_cos_sim'])
            
            train_loss_val = train_loss.item() if train_loss is not None else 'N/A'
            val_loss_val = val_loss.item() if val_loss is not None else 'N/A'
            cos_sim_val = cos_sim.item() if cos_sim is not None else 'N/A'
            
            writer.writerow([self.current_epoch, train_loss_val, val_loss_val, cos_sim_val])
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # AdamW optimizer
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
        dirpath="checkpoints_vjepa",
        filename="vjepa_{epoch:03d}_{val_loss:.4f}",
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        save_last=True,
        every_n_epochs=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Initialize model and trainer
    model = VJEPAModel(CFG)
    
    os.makedirs('checkpoints_vjepa', exist_ok=True)
    save_config(CFG, 'checkpoints_vjepa/config.json')
    
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
        deterministic=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")


if __name__ == "__main__":
    main()