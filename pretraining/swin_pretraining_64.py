import os
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoImageProcessor#, TimesformerModel
import numpy as np
import torch
import torchvision.transforms as T
import PIL
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# from timesformer_pytorch import TimeSformer

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
    
    start_idx = 22
    in_chans = 18
    valid_chans = 16 # chans used 
    
    size = 224
    tile_size = 224
    stride = tile_size // 1
    
    train_batch_size = 20
    valid_batch_size = 5
    lr = 1e-4
    # num_workers = 16
    
    # ============== model cfg =============
    scheduler = 'cosine'
    epochs = 16
    warmup_factor = 10
    
    # Change the size of fragments
    frags_ratio1 = ['frag','re']
    frags_ratio2 = ['202','s4','left']
    ratio1 = 2
    ratio2 = 1
    
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
        # A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
        #                 mask_fill_value=0, p=0.5),

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

# class TileDataset(Dataset):
#     def __init__(self, base_path, split="train", transform=None):
#         self.tile_paths = [
#             os.path.join(base_path, "tiles", split, f) 
#             for f in os.listdir(os.path.join(base_path, "tiles", split)) 
#             if f.endswith(".npy")
#         ]
#         self.transform = transform
#         self.video_transform = T.Compose([
#             T.ConvertImageDtype(torch.float32), 
#             T.Normalize(mean=[0.5], std=[0.5])
#             ])

#     def __len__(self):
#         return len(self.tile_paths)
    
#     def fourth_augment(self, image):
#         """
#         Randomly crop 8 channels from 12 and return them.
#         Input: image (H, W, 12)
#         Output: image (H, W, 8)
#         """

#         cropping_num = CFG.valid_chans  # Number of channels to crop

#         # Random start index
#         start_idx = random.randint(0, CFG.in_chans - cropping_num)

#         # Select consecutive 8 channels
#         crop_indices = np.arange(start_idx, start_idx + cropping_num)

#         # Slice directly
#         image_out = image[..., crop_indices]

#         return image_out

#     def __getitem__(self, idx):
#             image = np.load(self.tile_paths[idx])  # H x W x C
#             image = self.fourth_augment(image)          
#             if self.transform:
#                 data = self.transform(image=image)
#                 image = data['image'].unsqueeze(0)
                
#             image = image.permute(1,0,2,3)
#             image = torch.stack([self.video_transform(f) for f in image])

#             return image
class TileDataset(Dataset):
    def __init__(self, base_path, splits=["train"], transform=None):
        self.tile_paths = []
        for split in splits:
            split_path = os.path.join(base_path, "224_tiles", split)
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
        cropping_num = CFG.valid_chans  # Number of channels to crop
        start_idx = random.randint(0, CFG.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)
        return image[..., crop_indices]

    def __getitem__(self, idx):
        image = np.load(self.tile_paths[idx])  # H x W x C
        image = self.fourth_augment(image)          
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)
        image = image.permute(1,0,2,3)
        image = torch.stack([self.video_transform(f) for f in image])
        return image
    
from torch.utils.data import random_split, DataLoader
   
full_train_dataset = TileDataset(CFG.segment_path, splits=["train","valid"], transform=get_transforms('train', CFG))

# Suppose you want 5% for validation
val_size = int(0.001 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=16)
val_loader   = DataLoader(val_dataset, batch_size=CFG.valid_batch_size, shuffle=True, num_workers=16)
    
# train_dataset = TileDataset(
#     CFG.segment_path, 
#     splits=["train", "valid"],  # both folders now included
#     transform=get_transforms(data='train', cfg=CFG)
# )

# # train_dataset = TileDataset(CFG.segment_path, split="train", transform=get_transforms(data='train',cfg=CFG))
# # valid_dataset = TileDataset(CFG.segment_path, split="valid",transform=get_transforms(data='valid',cfg=CFG))

# train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=CFG.num_workers)
# valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=True, num_workers=CFG.num_workers)


print(f"Train loader length: {len(train_loader)}")
from torch.optim import AdamW


from transformers import SegformerForSemanticSegmentation
from torchvision.models.video import swin_transformer
import albumentations as A
from transformers import AutoImageProcessor, TimesformerModel
from transformers import TimesformerModel, TimesformerConfig
import matplotlib.pyplot as plt
import numpy as np
    
def visualize_reconstruction(original, reconstructed, sample_idx=0, num_frames=16,save_path='./results/reconstruction.png',epoch=0):
    """
    Visualize original and reconstructed video frames side by side.

    Args:
        original: tensor (B, T, C, H, W) original video batch
        reconstructed: tensor (B, T, C, H, W) reconstructed video batch
        sample_idx: int, index in batch to visualize
        num_frames: int, number of frames to display
    """
    # orig = denormalize(orig)
    # recon = denormalize(recon)
    orig = original[sample_idx]     # (T, C, H, W)
    recon = reconstructed[sample_idx]  # (T, C, H, W)

    # If grayscale, squeeze channel dim
    if orig.shape[1] == 1:
        orig = orig.squeeze(1)
        recon = recon.squeeze(1)

    # Clamp and convert to numpy
    orig = orig.cpu().numpy()
    recon = recon.cpu().detach().numpy()

    fig, axes = plt.subplots(2, num_frames, figsize=(3 * num_frames, 6))

    for i in range(num_frames):
        # Original frame
        ax = axes[0, i]
        ax.imshow(orig[i], cmap='gray')
        ax.set_title(f"Original Frame {i}")
        ax.axis('off')

        # Reconstructed frame
        ax = axes[1, i]
        ax.imshow(recon[i], cmap='gray')
        ax.set_title(f"Reconstructed Frame {i}")
        ax.axis('off')

    if save_path is not None:
        os.makedirs(os.path.dirname(f'./results/reconstruction_64_tiny_16_fs_{epoch}.png'), exist_ok=True)
        plt.savefig(f'./results/reconstruction_64_tiny_16_fs_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
 
mse_loss = nn.MSELoss()
   
def mae_pretrain_loss(pred, target):
    return mse_loss(pred, target)



class MAEPretrain(pl.LightningModule):
    def __init__(self, lr=1e-4, mask_ratio=0.75, embed_dim=768, decoder_dim=512, decoder_layers=4):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained Swin3D backbone
        backbone = swin_transformer.swin3d_b(weights='KINETICS400_IMAGENET22K_V1')
        # backbone = swin_transformer.SwinTransformer3d(
        #     patch_size=[2, 4, 4],      # temporal=2, spatial=4x4 patches
        #     embed_dim=192,              # base dimension
        #     depths=[2, 2, 12],       # Tiny config
        #     num_heads=[3, 6, 12],  # heads per stage
        #     window_size=[8, 7, 7],     # attention window
        #     stochastic_depth_prob=0.1, # DropPath
        # )


        # Get old weights
        old_conv = backbone.patch_embed.proj  # Conv3d(3, 128, ...)
        weight = old_conv.weight  # [128, 3, 2, 4, 4]
        bias = old_conv.bias      # [128]

        # Adapt weights: average across RGB → 1 channel
        new_weight = weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]

        # Replace conv with new one (keep out_channels=128!)
        backbone.patch_embed.proj = nn.Conv3d(
            in_channels=1,
            out_channels=128,
            kernel_size=(2, 4, 4),
            stride=(2, 4, 4),
            bias=True
        )

        # Load adapted weights
        backbone.patch_embed.proj.weight = nn.Parameter(new_weight)
        backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())  # shape [128]
        self.encoder = nn.Sequential(*list(backbone.children())[:-2]) 
        
        self.patch_size = 32
        self.input_T = 16
        self.input_H = 224
        self.input_W = 224
        self.tubelet_size = 2
        self.mask_ratio = self.hparams.mask_ratio

        # self.criterion = lambda pred, target: 1 * mse_loss(pred, target)  
        self.criterion = mae_pretrain_loss
  
        
        # self.N = self.input_T * self.input_H * self.input_W // (self.patch_size**2*2)
        self.N = (self.input_T // self.tubelet_size) * \
         (self.input_H // self.patch_size) * \
         (self.input_W // self.patch_size)
        print(f"Total patches: {self.N}")

        self.unmasked_patches =  int((1- self.mask_ratio)* self.input_T/self.tubelet_size)* self.input_H//self.patch_size* self.input_W//self.patch_size
        print(f"Actual patches used: {self.unmasked_patches}/{self.N} : {self.unmasked_patches/self.N:.2f}")

        # Transformer decoder components
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.N, decoder_dim))
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=8, dim_feedforward=decoder_dim, batch_first=True)
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_pred = nn.Sequential(
            nn.Linear(decoder_dim, self.patch_size**2*self.tubelet_size),
            # nn.Tanh()
        )

        # Mask token for masked patches in decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
    def patchify(self, x):
        """
        x: (B, T, C, H, W)
        Returns:
            patches: (B, N, patch_dim)
            where N = (T//tubelet_size) * (H//patch_size) * (W//patch_size)
                    patch_dim = C * tubelet_size * patch_size * patch_size
        """
        B, T, C, H, W = x.shape
        tubelet = self.tubelet_size   # e.g., 2
        ps = self.patch_size          # e.g., 32

        # (B, T, C, H, W) → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Unfold temporal and spatial dims
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
        # print(x.shape)
        return x

    
    def unpatchify(self, x, patch_shape):
        # x: (B, N, D)
        B, N, D = x.shape
        pt, ph, pw = patch_shape
        ps = self.patch_size
        tubelet = self.tubelet_size
        C = 1
        assert ph * pw * pt == N, "Patch count mismatch"
        x = x.view(B, pt, ph, pw, C, tubelet, ps, ps)                  # (..., C, tubelet, ps, ps)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()             # (B, C, pt, tubelet, ph, ps, pw, ps)
        x = x.view(B, C, pt * tubelet, ph * ps, pw * ps)               # (B, C, T, H, W)
        return x
    
    def frame_masking(self, x, mask_ratio=0.85):
        """
        Frame-level masking applied to patchified input.
        x: (B, N, D) where 
        N = (T//tubelet) * (H//ps) * (W//ps)
        Returns:
            x_masked:   (B, n_keep*H_p*W_p, D)
            ids_keep:   (B, n_keep*H_p*W_p)
            ids_masked: (B, n_mask*H_p*W_p)
            ids_restore:(B, N)  - to restore original order
        """
        B, N, D = x.shape
        T_groups = self.input_T // self.tubelet_size   # e.g., 4
        H_groups = self.input_H // self.patch_size     # e.g., 7
        W_groups = self.input_W // self.patch_size     # e.g., 7
        # print(T_groups,H_groups,W_groups)
        # print(N)
        patches_per_frame = H_groups * W_groups
        assert N == T_groups * patches_per_frame, "Patch count mismatch"

        n_keep_frames = int((1 - mask_ratio) * T_groups)
        n_keep = n_keep_frames * patches_per_frame

        ids_keep, ids_masked, ids_restore = [], [], []

        for b in range(B):
            # permute frames
            perm_frames = torch.randperm(T_groups, device=x.device)
            keep_frames = perm_frames[:n_keep_frames]
            mask_frames = perm_frames[n_keep_frames:]

            # expand to patch indices
            keep_idx = (keep_frames[:, None] * patches_per_frame +
                        torch.arange(patches_per_frame, device=x.device)[None, :])
            mask_idx = (mask_frames[:, None] * patches_per_frame +
                        torch.arange(patches_per_frame, device=x.device)[None, :])

            keep_idx = keep_idx.flatten()
            mask_idx = mask_idx.flatten()

            # build restore index
            perm = torch.cat([keep_idx, mask_idx], dim=0)
            ids_restore_b = torch.empty_like(perm)
            ids_restore_b[perm] = torch.arange(N, device=x.device)

            ids_keep.append(keep_idx)
            ids_masked.append(mask_idx)
            ids_restore.append(ids_restore_b)

        ids_keep = torch.stack(ids_keep, dim=0)      # (B, n_keep)
        ids_masked = torch.stack(ids_masked, dim=0)  # (B, n_mask)
        ids_restore = torch.stack(ids_restore, dim=0) # (B, N)

        # gather visible patches
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_masked, ids_keep, ids_masked, ids_restore

    def tube_masking(self, x, mask_ratio=0.75):
        """
        Tube-level masking applied to patchified input.
        x: (B, N, D) where 
            N = (T//tubelet) * (H//ps) * (W//ps)
        Returns:
            x_masked:   (B, n_keep*D)
            ids_keep:   (B, n_keep)
            ids_masked: (B, n_mask)
            ids_restore:(B, N)  - to restore original order
        """
        B, N, D = x.shape
        T_groups = self.input_T // self.tubelet_size   # temporal groups
        H_groups = self.input_H // self.patch_size     # spatial groups (height)
        W_groups = self.input_W // self.patch_size     # spatial groups (width)

        num_tubes = H_groups * W_groups  # number of spatial tube locations
        tube_len = T_groups              # length of each tube (temporal depth)

        # # each "tube" corresponds to all temporal positions at one (h, w)
        assert N == num_tubes * tube_len, "Patch count mismatch"

        n_keep_tubes = self.N - self.unmasked_patches

        ids_keep, ids_masked, ids_restore = [], [], []

        for b in range(B):
            # permute tube positions
            perm_tubes = torch.randperm(num_tubes, device=x.device)
            keep_tubes = perm_tubes[:n_keep_tubes]
            mask_tubes = perm_tubes[n_keep_tubes:]

            # expand tube indices across time
            keep_idx = (keep_tubes[:, None] * tube_len +
                        torch.arange(tube_len, device=x.device)[None, :])
            mask_idx = (mask_tubes[:, None] * tube_len +
                        torch.arange(tube_len, device=x.device)[None, :])

            keep_idx = keep_idx.flatten()
            mask_idx = mask_idx.flatten()

            # build restore index
            perm = torch.cat([keep_idx, mask_idx], dim=0)
            ids_restore_b = torch.empty_like(perm)
            ids_restore_b[perm] = torch.arange(N, device=x.device)

            ids_keep.append(keep_idx)
            ids_masked.append(mask_idx)
            ids_restore.append(ids_restore_b)

        ids_keep = torch.stack(ids_keep, dim=0)      # (B, n_keep)
        ids_masked = torch.stack(ids_masked, dim=0)  # (B, n_mask)
        ids_restore = torch.stack(ids_restore, dim=0) # (B, N)

        # gather visible patches
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_masked, ids_keep, ids_masked, ids_restore


    def random_masking(self, x, mask_ratio=0.75):
        """
        MAE-style random masking with restore indices.
        x: (B, N, D)
        Returns:
            x_masked: (B, n_keep, D)    - visible patches
            ids_keep: (B, n_keep)       - indices of kept patches
            ids_masked: (B, n_mask)     - indices of masked patches
            ids_restore: (B, N)         - to restore original order
        """
        B, N, D = x.shape
        n_keep = self.unmasked_patches

        ids_keep = []
        ids_masked = []
        ids_restore = []

        for b in range(B):
            # 1. Random permutation of all patches
            perm = torch.randperm(N, device=x.device)
            keep = perm[:n_keep]
            masked = perm[n_keep:]

            # 2. Save indices
            ids_keep.append(keep)
            ids_masked.append(masked)

            # 3. Build restore index (inverse of permutation)
            ids_restore_b = torch.empty_like(perm)
            ids_restore_b[perm] = torch.arange(N, device=x.device)
            ids_restore.append(ids_restore_b)

        ids_keep = torch.stack(ids_keep, dim=0)      # (B, n_keep)
        ids_masked = torch.stack(ids_masked, dim=0)  # (B, n_mask)
        ids_restore = torch.stack(ids_restore, dim=0) # (B, N)

        # 4. Gather kept tokens
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_masked, ids_keep, ids_masked, ids_restore
    

    def forward(self, x):
        B, T, C, H, W = x.shape # (B, T, C, H, W)

        # print(x.shape)
        # 1. Patchify input video
        x_patched = self.patchify(x)  # (B, N, D)
        # print('x_patched',x_patched.shape) # (B,N,T,D)
        
        N = x_patched.shape[1]

        # 2. Mask patches
        x_masked, ids_keep, ids_masked, ids_restore = self.random_masking(x_patched, self.mask_ratio)
        
        # print('x_masked',x_masked.shape) # (B,N,T,D)
        ids_keep = ids_keep.long() # Ids of unmasked
        ids_masked = ids_masked.long()
        ids_restore = ids_restore.long()

        # Calculate masked patch indices
        all_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)  # (B, N)
        mask = torch.ones_like(all_ids, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)
        
        pt = int((self.input_T // self.tubelet_size)*(1-self.mask_ratio))
        ph = pw = int((self.unmasked_patches // pt) ** 0.5)
        assert ph * pw * pt == self.unmasked_patches, "Patch grid mismatch"

        # We know the tube shape is (pt, square_size, square_size)
        x_masked_video = self.unpatchify(x_masked, (pt, ph, pw))  # (B, C, T, H_mask, W_mask)
        # print('x_masked_unpatchify',x_masked_video.shape)
        
        # 4. Encoder forward on masked video        
        # x_masked_video = x_masked_video.permute(0,2,1,3,4) # (B,T,C,H,W)
        outputs = self.encoder(x_masked_video)
        # print('outputs',outputs.shape)
        # Group first
        tokens = outputs#.view(B, pt, ph, pw, self.hparams.embed_dim)  # (B, 16, 8, 8, D)
        # tokens = tokens.view(B, ph, pw, T, self.hparams.embed_dim)  # (B, 16, 8, 8, D)
        # tokens = tokens.permute(0, 3, 1, 2, 4).contiguous()         # (B, ph, pw, T, D)  <-- matches your compact cube order
        # print(tokens.shape)

        # 5. Embed encoder features to decoder_dim
        x_vis = self.decoder_embed(tokens)  # (B, n_visible, decoder_dim)
        
        x_vis = x_vis.view(B,-1,self.hparams.decoder_dim)
        # print(x_vis.shape)

        # 6. Prepare mask tokens for masked patches
        # print(ids_masked.shape[1])

        mask_tokens = self.mask_token.expand(B, ids_masked.shape[1], -1)  # (B, n_masked, decoder_dim)
        # x_ = torch.cat([x_vis, mask_tokens], dim=1)   # (B, N, D) but shuffled
        # x_dec = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_vis.shape[2]))

        # print(mask_tokens.shape)
        # 7. Create full sequence tensor for decoder input
        # Restore to original order
        x_ = torch.cat([x_vis, mask_tokens], dim=1)  # (B, n_keep + n_masked, D)
        # print(x.shape)
        x_dec = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))

        # 8. Add positional embedding
        x_dec = x_dec + self.decoder_pos_embed
    
        # 9. Decode full sequence
        x_dec = self.decoder_transformer(x_dec)
        # print('x_dec',x_dec.shape)
        pred = self.decoder_pred(x_dec)  # (B, N, patch_dim)
        
        # 3. Unpatchify visible patches to video for encoder
        pt = self.input_T // self.tubelet_size
        ph = pw = int((self.N // pt) ** 0.5)
        # print(pt,ph,pw)
        # 
        recon = self.unpatchify(pred, (pt, ph, pw))  # (B, C, T, H, W)
        recon = recon.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        # print(recon.shape)
        return recon, x_masked_video, mask, ids_masked, pred, x_patched



    def training_step(self, batch, batch_idx):
        x = batch  # (B, 1, T, H, W)
        B = x.shape[0]

        recon, x_masked, mask, ids_masked, pred, target = self(x)

        # Gather masked predictions & targets per device
        B, N, D = pred.shape
        device = pred.device

        # ids_masked: (B, n_mask)
        ids_masked_exp = ids_masked.unsqueeze(-1).expand(-1, -1, D)  # (B, n_mask, D)

        pred_masked   = torch.gather(pred,   1, ids_masked_exp)  # (B, n_mask, D)
        target_masked = torch.gather(target, 1, ids_masked_exp)  # (B, n_mask, D)

        # Compute loss only on masked patches
        loss = self.criterion(pred_masked, target_masked)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)  # sync across devices

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        recon, _, mask, _, _, _ = self(x)
        loss = self.criterion(recon, x)
        self.log('val_loss', loss, prog_bar=True)
        # Save first batch to visualize later
        if batch_idx == 0:
            self.val_batch_for_viz = (x, mask, recon)
        return loss

    def on_validation_epoch_end(self):
        if self.global_rank == 0 and hasattr(self, 'val_batch_for_viz'):
            x, mask, recon = self.val_batch_for_viz
            visualize_reconstruction(x, recon, sample_idx=0, num_frames=16,epoch=self.current_epoch)

    def configure_optimizers(self):
        # optimizer = AdamW([
        #     {'params': self.encoder.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
        #     {'params': list(self.decoder_embed.parameters()) +
        #             list(self.decoder_transformer.parameters()) +
        #             list(self.decoder_pred.parameters()), 
        #     'lr': 1e-4, 'weight_decay': 1e-3}
        # ])
        optimizer = AdamW(
            self.parameters(),  # encoder + decoder
            lr=1e-4,          # scaled for batch 800
            weight_decay=0.05
        )

        # Cosine LR with warmup
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=5, after_scheduler=cosine_scheduler
        )

        return [optimizer], [scheduler]

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Save a checkpoint at the end of every epoch
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",             # folder to save
    filename="224_tiny_16_{epoch}",           # filename pattern
    save_top_k=-1,                     # save all checkpoints
    every_n_epochs=1                   # save every epoch
)
if __name__ == '__main__':
    model = MAEPretrain()
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=-1,                     # use all available GPUs
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,          # clip gradients to stabilize training
        gradient_clip_algorithm="norm", # clip by norm (recommended for Transformers)
        callbacks=[checkpoint_callback],
        strategy='ddp',
    )# trainer.validate(model, valid_loader)

    trainer.fit(model, train_loader, val_loader)