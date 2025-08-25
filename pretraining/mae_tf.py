import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoImageProcessor#, TimesformerModel
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
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
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = 933120000

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
import sys
sys.path.append(parent_dir)

import utils
import models.swin as swin
import models.timesformer_hug as timesformer_hug

class TimesformerDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.xyxys=xyxys
        self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32), 
            T.Normalize(mean=[0.5], std=[0.25])
            ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.xyxys is not None: #VALID
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            
            image = image.permute(1,0,2,3)
            image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            return image, label
        else:
            image = self.images[idx]
            label = self.labels[idx]
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
                
            image = image.permute(1,0,2,3)
            image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            return image, label
        
        
class CFG:
    # ============== comp exp name =============
    current_dir = './'
    segment_path = '../train_scrolls/'
    
    start_idx = 24
    in_chans = 16
    
    size = 128
    tile_size = 128
    stride = tile_size // 4
    
    train_batch_size =  10 # 32
    valid_batch_size = 10
    lr = 1e-4
    num_workers = 8
    # ============== model cfg =============
    scheduler = 'cosine'#, 'linear'
    epochs = 16
    warmup_factor = 10
    
    # Change the size of fragments
    frags_ratio1 = ['frag','re']
    frags_ratio2 = ['202','s4','left']
    ratio1 = 2
    ratio2 = 1
    
    # ============== fold =============
    segments = ['remaining5','rect5'] 
    valid_id = 'rect5'
    # ============== fixed =============
    min_lr = 1e-7
    weight_decay = 1e-6
    max_grad_norm = 100
    num_workers = 8
    seed = 0
    
    # ============== comp exp name =============
    comp_name = 'vesuvius'
    exp_name = 'pretraining_all'

    outputs_path = f'../outputs/{comp_name}/{exp_name}/'
    model_dir = outputs_path + \
        f'{comp_name}-models/'
        
    # ============== augmentation =============
    train_aug_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0,p=0.75),
        # A.OneOf([
        #         A.GaussNoise(var_limit=[10, 50]),
        #         A.GaussianBlur(),
        #         A.MotionBlur(),
        #         ], p=0.4),
        # A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
        #                 mask_fill_value=0, p=0.5),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ToTensorV2(transpose_mask=True),  
    ]
    
def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug   

# End any existing run (if still active)
if wandb.run is not None:
    wandb.finish()
        
utils.cfg_init(CFG)
torch.set_float32_matmul_precision('medium')

fragment_id = CFG.valid_id
run_slug=f'SWIN_{CFG.segments}_valid={CFG.valid_id}_size={CFG.size}_lr={CFG.lr}_in_chans={CFG.in_chans}'
valid_mask_gt = cv2.imread(f"{CFG.segment_path}{fragment_id}/{fragment_id}_inklabels.png", 0)

if any(sub in fragment_id for sub in CFG.frags_ratio1):
    scale = 1 / CFG.ratio1
    new_w = int(valid_mask_gt.shape[1] * scale)
    new_h = int(valid_mask_gt.shape[0] * scale)
    valid_mask_gt = cv2.resize(valid_mask_gt, (new_w, new_h), interpolation=cv2.INTER_AREA)

elif any(sub in fragment_id for sub in CFG.frags_ratio2):
    scale = 1 / CFG.ratio2
    new_w = int(valid_mask_gt.shape[1] * scale)
    new_h = int(valid_mask_gt.shape[0] * scale)
    valid_mask_gt = cv2.resize(valid_mask_gt, (new_w, new_h), interpolation=cv2.INTER_AREA)
pred_shape=valid_mask_gt.shape

train_images, train_masks, valid_images, valid_masks, valid_xyxys = utils.get_train_valid_dataset(CFG)

print('train_images',train_images[0].shape)
print("Length of train images:", len(train_images))

valid_xyxys = np.stack(valid_xyxys)
train_dataset = TimesformerDataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='valid', cfg=CFG))
valid_dataset = TimesformerDataset(
    valid_images, CFG, xyxys=valid_xyxys, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

train_loader = DataLoader(train_dataset,
                            batch_size=CFG.train_batch_size,
                            shuffle=True,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                            )
valid_loader = DataLoader(valid_dataset,
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

print(f"Train loader length: {len(train_loader)}")
print(f"Valid loader length: {len(valid_loader)}")

from torch.optim import AdamW

import utils

from transformers import SegformerForSemanticSegmentation
from torchvision.models.video import swin_transformer
import albumentations as A
from transformers import AutoImageProcessor, TimesformerModel
from transformers import TimesformerModel, TimesformerConfig
import matplotlib.pyplot as plt
import numpy as np

def denormalize(t):
    return t * 0.25 + 0.5  # undo mean/std

   
    
def visualize_reconstruction(original, reconstructed, sample_idx=0, num_frames=4,save_path=None):
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# class MAEPretrainSwin(pl.LightningModule):
#     def __init__(self, lr=1e-3, mask_ratio=0.75, embed_dim=768, decoder_dim=768, decoder_layers=4):
#         super().__init__()
#         self.save_hyperparameters()
        
#         config = TimesformerConfig(
#             num_frames=16,
#             image_size=128,
#             patch_size=8,
#             num_channels=1,
#             attention_type="divided_space_time"
#         )
        
#         self.encoder = TimesformerModel(config)
        
#         self.patch_size = config.patch_size
#         self.input_T = config.num_frames
#         self.input_H = config.image_size
#         self.input_W = config.image_size
#         self.mask_ratio = self.hparams.mask_ratio
        
#         self.criterion = nn.MSELoss()
#         self.train_loss_history = []
#         self.val_loss_history = []
#         self.lr_history = []
#         N = self.input_T * self.input_H * self.input_W / (self.patch_size**3)
#         print(f"Actual mask ratio: {int(N * (1 - self.mask_ratio))}/{N}:{int(N * (self.mask_ratio))/N}")
#         self.unmasked_patches = int(N * (1 - self.mask_ratio))

#         # Transformer decoder components
#         self.merge_layer = nn.Linear(self.input_T, 2) 
#         self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
#         self.decoder_pos_embed = nn.Parameter(torch.randn(1, 1000, decoder_dim))  # Assume max 1000 patches
#         decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=8, dim_feedforward=2048)
#         self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
#         self.decoder_pred = nn.Linear(decoder_dim, self.patch_size**3)  # Output prediction per patch

#         # Mask token for masked patches in decoder
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
#         nn.init.normal_(self.mask_token, std=0.02)

#     def patchify(self, x):
#         x = x.permute(0,2,1,3,4)  # (B,C,T,H,W)
#         B, C, T, H, W = x.shape
#         assert T % self.patch_size == 0 and H % self.patch_size == 0 and W % self.patch_size == 0
#         x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
#         x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size, self.patch_size)
#         x = x.permute(0, 2, 1, 3, 4, 5)  # (B, N, C, ps, ps, ps)
#         # print(x.shape)
#         x = x.reshape(B, -1, C * self.patch_size**3)
#         # print(x.shape)
#         return x

#     def unpatchify(self, x, patch_shape):
#         B, N, D = x.shape
#         pt, ph, pw = patch_shape
#         ps = self.patch_size
#         C = D // (ps**3)
#         # print(N,patch_shape)
#         assert pt * ph * pw == N
#         x = x.view(B, pt, ph, pw, C, ps, ps, ps)
#         x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
#         x = x.contiguous().view(B, C, pt * ps, ph * ps, pw * ps)
#         return x

#     def random_masking(self, x, mask_ratio=0.75):
#         """
#         Randomly masks patches in input x.
#         Args:
#             x: input tensor of shape (B, N, D) -- batch, patches, embedding_dim
#             mask_ratio: fraction of patches to mask (e.g. 0.75 means 75% masked)
#         Returns:
#             x_masked: visible patches tensor (B, n_keep, D)
#             ids_keep: indices of visible patches (B, n_keep)
#             ids_masked: indices of masked patches (B, n_mask)
#         """
#         B, N, D = x.shape
#         n_keep = int(N * (1 - mask_ratio))

#         ids_keep = []
#         ids_masked = []

#         for b in range(B):
#             perm = torch.randperm(N, device=x.device)
#             keep = perm[:n_keep]
#             masked = perm[n_keep:]

#             ids_keep.append(keep)
#             ids_masked.append(masked)

#         ids_keep = torch.stack(ids_keep, dim=0)    # (B, n_keep)
#         ids_masked = torch.stack(ids_masked, dim=0)  # (B, n_mask)

#         x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

#         return x_masked, ids_keep, ids_masked

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         # print('x.shape',x.shape)

#         # 1. Patchify input video
#         x_patched = self.patchify(x)  # (B, N, D)
#         N = x_patched.shape[1]
#         # ('x_patched',x_patched.shape)

#         # 2. Mask patches, keep a tube of patches visible to encoder
#         # x_masked, ids_keep, _ = self.tube_masking(x_patched, square_size=model.hparams.mask_ratio)  # (B, n_visible, D)
#         x_masked, ids_keep, ids_masked = self.random_masking(x_patched)
#         ids_keep = ids_keep.long() # Ids of unmasked
#         ids_masked = ids_masked.long()
#         # print(ids_keep[0])
#         # print(ids_masked.shape)
#         # print('x_masked',x_masked.shape)

#         # # Calculate masked patch indices
#         all_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)  # (B, N)
#         mask = torch.ones_like(all_ids, dtype=torch.bool)
#         mask.scatter_(1, ids_keep, False)
#         # ids_masked = all_ids[mask].view(B, -1)
#         # ids_masked_sorted = torch.sort(idis_masked, dim=1)[0]
#         # print(torch.equal(ids_masked_sorted, ids_masked))
#         # # print(idis_masked)
#         # print(torch.equal(ids_masked, idis_masked))

#         # 3. Unpatchify visible patches to video for encoder
#         pt= T // self.patch_size#, H // self.patch_size, W // self.patch_size
#         ph = int((self.unmasked_patches/pt)**(1/2))
#         pw = int((self.unmasked_patches/pt)**(1/2))
#         # print(pt, ph, pw)
#         # We know the tube shape is (pt, square_size, square_size)
#         x_masked_video = self.unpatchify(x_masked, (pt, ph, pw))  # (B, C, T, H_mask, W_mask)
#         # print('x_masked_unpatchify',x_masked_video.shape)
#         # x_masked_video = x_masked_video.permute(0,2,1,3,4)  # (B,C,T,H,W)
        
        
#         # # 4. Encoder forward on masked video        
#         x_masked_video = x_masked_video.permute(0,2,1,3,4)
#         # print(x_masked_video.shape)
#         outputs = self.encoder(x_masked_video, output_hidden_states=True)
#         # print(outputs.last_hidden_state.shape)
#         tokens = outputs.last_hidden_state[:,1:,:]  # tuple of all hidden layers
        
#         # Group first
#         tokens = tokens.view(B,T, ph, pw,self.hparams.embed_dim)  # (B, 2, 8, 8, D)
#         # print(tokens.shape)
#         # Merge with learnable weights along time 
#         # print(merge_layer.shape)
#         tokens = self.merge_layer(tokens.permute(0,4,2,3,1)).permute(0,4,2,3,1)  # (B, 768, 8, 8, D)
#         # print(tokens.shape)

#         # 5. Embed encoder features to decoder_dim
#         x_vis = self.decoder_embed(tokens)  # (B, n_visible, decoder_dim)
#         x_vis = x_vis.view(B,-1,self.hparams.embed_dim)
#         # print(x_vis.shape)

#         # 6. Prepare mask tokens for masked patches
#         mask_tokens = self.mask_token.expand(B, ids_masked.shape[1], -1)  # (B, n_masked, decoder_dim)

#         # 7. Create full sequence tensor for decoder input
#         x_dec = torch.zeros(B, N, x_vis.shape[2], device=x.device, dtype=x_vis.dtype)
#         # print(x_dec.shape)
#         # print(ids_keep.unsqueeze(-1).expand(-1, -1, x_vis.shape[2]).shape)
#         x_dec.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, x_vis.shape[2]), x_vis)
#         x_dec.scatter_(1, ids_masked.unsqueeze(-1).expand(-1, -1, x_vis.shape[2]), mask_tokens)

#         # 8. Add positional embedding
#         pos_embed = self.decoder_pos_embed[:, :N, :]
#         x_dec = x_dec + pos_embed

#         # 9. Decode full sequence
#         x_dec = self.decoder_transformer(x_dec)
#         # print('x_dec',x_dec.shape)
#         pred = self.decoder_pred(x_dec)  # (B, N, patch_dim)
#         # print('pred',pred.shape)
#         # print('x_patched',x_patched.shape)
#         # pred.reshape(B, C, T, H, W)
#         # print(pred.shape)
#         # 10. Compute reconstruction loss only on masked patches (in training_step)
#         # 11. Unpatchify pred to video
#         pt, ph, pw = T // self.patch_size, H // self.patch_size, W // self.patch_size
#         recon = self.unpatchify(pred, (pt, ph, pw))  # (B, C, T, H, W)
#         recon = recon.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
#         # print('recon',recon.shape)
#         # recon =1


#         return recon, x_masked_video, mask, ids_masked, pred, x_patched

#     def training_step(self, batch, batch_idx):
#         x, y = batch  # (B, 1, T, H, W)
#         B = x.shape[0]
#         # print('x',x.shape)

#         recon, x_masked, mask, ids_masked, pred, target = self(x)

#         # Compute loss only on masked patches
#         mask = mask.bool()

#         # Flatten batch and positions so we can index directly
#         pred_masked = pred[mask].view(B, -1, pred.size(-1))
#         target_masked = target[mask].view(B, -1, target.size(-1))


#         # Compute loss only on masked patches
#         loss = self.criterion(pred_masked, target_masked)
#         self.train_loss_history.append(loss.item())  # ✅ Track loss here
#         self.log("train_loss", loss, prog_bar=True, logger=True)

#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         recon,_ ,  ids,_, _, _ = self(x)
#         loss = self.criterion(recon, x)
#         self.log('val_loss', loss, prog_bar=True)
#         self.val_loss_history.append(loss.item())
#         # Save first batch to visualize later
#         if batch_idx == 0:
#             # print('YYYYYYYYYYYYYYYY')
#             self.val_batch_for_viz = (x,ids, recon)
            
#         return loss
#     def on_train_epoch_end(self):
#         # Save LR for plotting later
#         opt = self.optimizers()
#         lr = opt.param_groups[0]['lr']
#         self.lr_history.append(lr)
#         self.log("lr", lr, prog_bar=True, logger=True)
        
#     def on_validation_epoch_end(self):
#         # Visualize every 10 epochs
#         # if (self.current_epoch + 1) % 10 != 0:
#             if hasattr(self, 'val_batch_for_viz'):
#                 x,ids, recon = self.val_batch_for_viz
#                 visualize_reconstruction(x, recon,
#                          sample_idx=0, num_frames=2, 
#                          save_path="results/reconstruction_.png")


#     def configure_optimizers(self):
#         optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-6)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
#         # warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
#         return [optimizer], [scheduler]
    
#     def on_fit_end(self):
#         os.makedirs("results", exist_ok=True)

#         # Plot Learning Curves
#         plt.figure(figsize=(10,4))
#         plt.plot(self.train_loss_history, label="Train Loss")
#         plt.plot(self.val_loss_history, label="Val Loss")
#         plt.xlabel("Steps")
#         plt.ylabel("Loss")
#         plt.title("Learning Curves")
#         plt.legend()
#         plt.grid(True)
#         plt.savefig("results/learning_curves.png", dpi=300, bbox_inches='tight')
#         plt.close()

#         # Plot LR Schedule
#         plt.figure(figsize=(6,4))
#         plt.plot(self.lr_history, label="Learning Rate")
#         plt.xlabel("Epoch")
#         plt.ylabel("LR")
#         plt.title("LR per Epoch")
#         plt.grid(True)
#         plt.legend()
#         plt.savefig("results/lr_schedule.png", dpi=300, bbox_inches='tight')
#         plt.close()


class MAEPretrainSwin(pl.LightningModule):
    def __init__(self, lr=1e-5, mask_ratio=0.75, embed_dim=768, decoder_dim=768, decoder_layers=4):
        super().__init__()
        self.save_hyperparameters()
        
        config = TimesformerConfig(
            num_frames=16,
            image_size=128,
            patch_size=8,
            num_channels=1,
            attention_type="divided_space_time",
        )
        self.encoder = TimesformerModel(config)
        
        self.patch_size = config.patch_size
        self.input_T = config.num_frames
        self.input_H = config.image_size
        self.input_W = config.image_size
        self.mask_ratio = self.hparams.mask_ratio
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        # def criterion(pred, target, alpha=0.5):
        #     return alpha * mse_loss(pred, target) + (1 - alpha) * l1_loss(pred, target)
        self.criterion = lambda pred, target: 1 * mse_loss(pred, target) + 0 * l1_loss(pred, target)
        
        # self.criterion = nn.MSELoss()
        self.train_loss_history = []
        
        self.N = self.input_T * self.input_H * self.input_W // (self.patch_size**2)
        # self.unmasked_patches = int(self.N * (1 - mask_ratio))
        # Step 1: keep count divisible by pt
        self.unmasked_patches = int(self.N * (1 - mask_ratio)) // (self.input_T **2) * (self.input_T **2)
        # Step 2: patches per frame, force perfect square
        # patches_per_frame = int((n_keep // pt) ** 0.5) ** 2
        # Step 3: adjust n_keep to match square
        # n_keep = patches_per_frame * pt
        # self.unmasked_patches = n_keep
        print(f"Actual patches used: {self.unmasked_patches}/{self.N} : {int(self.N * (self.mask_ratio))/self.N}")

        # Transformer decoder components
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.N, decoder_dim)) 
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=8, dim_feedforward=decoder_dim)
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        # self.decoder_pred = nn.Linear(decoder_dim, self.patch_size**2)  # Output prediction per patch
        self.decoder_pred = nn.Sequential(
            nn.Linear(decoder_dim, self.patch_size**2),
            # nn.Tanh()  # ensures outputs in [-1, 1]
        )


        # Mask token for masked patches in decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # print('x.shape',x.shape)
        # Only unfold spatial dimensions
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = x.unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
        # x: (B, C, T, H_patches, W_patches, patch_size, patch_size)
        
        H_patches = x.size(3)
        W_patches = x.size(4)
        
        # Rearrange to (B, N, C, patch_size, patch_size)
        x = x.contiguous().view(B, C, T, H_patches * W_patches, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 2, 1, 4, 5)  # (B, N, T, C, patch_size, patch_size)
        # Flatten patch: (C * patch_size * patch_size) and keep T
        x = x.reshape(B, H_patches * W_patches, T, C * self.patch_size**2)
        x = x.reshape(B, H_patches * W_patches*T, C * self.patch_size**2)

        # print('x.shape',x.shape)
        return x  # shape: (B, N, patch_dim)

    def unpatchify(self, x, patch_shape):
        """
        x: (B, N, D) from masked patches
        patch_shape: (pt, ph, pw) where
            pt = number of frames (T),
            ph = patches along height,
            pw = patches along width
        """
        B, N, D = x.shape
        pt, ph, pw = patch_shape
        ps = self.patch_size
        C = D // (ps**2)  # only spatial patches

        # assert pt == T, "Temporal dim mismatch"
        assert ph * pw * pt == N, "Spatial patch count mismatch"

        # Reshape back to grid of patches
        x = x.view(B, ph, pw, pt, C, ps, ps)  # (B, ph, pw, T, C, ps, ps)
        x = x.permute(0, 4, 3, 1, 5, 2, 6)  # (B, C, T, ph, ps, pw, ps)
        x = x.contiguous().view(B, C, pt, ph * ps, pw * ps)

        return x


    def random_masking(self, x, mask_ratio=0.75):
        """
        Randomly mask spatial patches across all frames.
        x: (B, N, D)
        Returns:
            x_masked: (B, n_keep,D)
            ids_keep: (B, n_keep)
            ids_masked: (B, n_mask)
        """
        B, N, D = x.shape
        n_keep = self.unmasked_patches
        # print('Keep: ',n_keep, 'out of ',N)
    

        ids_keep = []
        ids_masked = []

        for b in range(B):
            perm = torch.randperm(N, device=x.device)
            keep = perm[:n_keep]
            masked = perm[n_keep:]

            ids_keep.append(keep)
            ids_masked.append(masked)

        ids_keep = torch.stack(ids_keep, dim=0)      # (B, n_keep)
        ids_masked = torch.stack(ids_masked, dim=0)  # (B, n_mask)
        # Gather patches along the N dimension, D intact
        x_masked = torch.gather(
            x, 1, ids_keep.unsqueeze(-1).expand(-1,-1,D)
        )
    

        return x_masked, ids_keep, ids_masked




    def forward(self, x):
        B, T, C, H, W = x.shape 
        # print('x.shape',x.shape)
        # print(x.max())
        # print(x.min())
        # 1. Patchify input video
        x_patched = self.patchify(x)  # (B, N, D)
        # print('x_patched',x_patched.shape) # (B,N,T,D)
        
        N = x_patched.shape[1]

        # 2. Mask patches
        x_masked, ids_keep, ids_masked = self.random_masking(x_patched,self.mask_ratio)
        ids_keep = ids_keep.long() # Ids of unmasked
        ids_masked = ids_masked.long()
        # print(ids_masked.shape)
        # print('x_masked',x_masked.shape)

        # # Calculate masked patch indices
        all_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)  # (B, N)
        mask = torch.ones_like(all_ids, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)
        # ids_masked = all_ids[mask].view(B, -1)
        # ids_masked_sorted = torch.sort(idis_masked, dim=1)[0]
        # print(torch.equal(ids_masked_sorted, ids_masked))
        # # print(idis_masked)
        # print(torch.equal(ids_masked, idis_masked))
        pt = T  
        # patches_per_frame = self.unmasked_patches // T  
        # print(patches_per_frame)
        # ph = int(patches_per_frame // (2)  )         # floor
        # pw = math.ceil(patches_per_frame / (2))      # ceil
        # 3. Unpatchify visible patches to video for encoder
        # pt, ph , pw = T, self.unmasked_patches/(2*T*16), self.unmasked_patches/(2*T*16)
        ph = math.floor((self.unmasked_patches/pt)**(0.5))
        pw= math.floor((self.unmasked_patches/pt)**(0.5))
        # print(pt, ph, pw)
        # print(self.unmasked_patches)
        # We know the tube shape is (pt, square_size, square_size)
        x_masked_video = self.unpatchify(x_masked, (pt, ph, pw))  # (B, C, T, H_mask, W_mask)
        # print('x_masked_unpatchify',x_masked_video.shape)
        # x_masked_video = x_masked_video.permute(0,2,1,3,4)  # (B,C,T,H,W)
        
        
        # # 4. Encoder forward on masked video        
        x_masked_video = x_masked_video.permute(0,2,1,3,4)
        # print(x_masked_video.shape)
        outputs = self.encoder(x_masked_video, output_hidden_states=True)
        # print(outputs.last_hidden_state.shape)
        tokens = outputs.last_hidden_state[:,1:,:]  # tuple of all hidden layers
        # print(tokens.shape)
        # Group first
        tokens = tokens.view(B, T, ph, pw,self.hparams.embed_dim)  # (B, 2, 8, 8, D)
        # print(tokens.shape)
        # Merge with learnable weights along time 
        # print(merge_layer.shape)
        # tokens = self.merge_layer(tokens.permute(0,4,2,3,1)).permute(0,4,2,3,1)  # (B, 768, 8, 8, D)
        # print(tokens.shape)

        # 5. Embed encoder features to decoder_dim
        x_vis = self.decoder_embed(tokens)  # (B, n_visible, decoder_dim)
        x_vis = x_vis.view(B,-1,self.hparams.embed_dim)
        # print(x_vis.shape)

        # 6. Prepare mask tokens for masked patches
        mask_tokens = self.mask_token.expand(B, ids_masked.shape[1], -1)  # (B, n_masked, decoder_dim)

        # 7. Create full sequence tensor for decoder input
        x_dec = torch.zeros(B, N, x_vis.shape[2], device=x.device, dtype=x_vis.dtype)
        # print(x_dec.shape)
        # print(ids_keep.unsqueeze(-1).expand(-1, -1, x_vis.shape[2]).shape)
        x_dec.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, x_vis.shape[2]), x_vis)
        x_dec.scatter_(1, ids_masked.unsqueeze(-1).expand(-1, -1, x_vis.shape[2]), mask_tokens)

        # 8. Add positional embedding
        pos_embed = self.decoder_pos_embed[:, :N, :]
        x_dec = x_dec + pos_embed

        # 9. Decode full sequence
        x_dec = self.decoder_transformer(x_dec)
        # print('x_dec',x_dec.shape)
        pred = self.decoder_pred(x_dec)  # (B, N, patch_dim)
        # print('pred',pred.shape)
        # print('x_patched',x_patched.shape)
        # pred.reshape(B, C, T, H, W)
        # print(pred.shape)
        # 10. Compute reconstruction loss only on masked patches (in training_step)
        # 11. Unpatchify pred to video
        # pt, ph, pw = T // self.patch_size, H // self.patch_size, W // self.patch_size
        # pt = T  
        # 3. Unpatchify visible patches to video for encoder
        ph = math.floor((self.N/pt)**(0.5))
        pw= math.floor((self.N/pt)**(0.5))
        recon = self.unpatchify(pred, (pt, ph, pw))  # (B, C, T, H, W)
        recon = recon.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        # print('recon',recon.shape)

        return recon, x_masked_video, mask, ids_masked, pred, x_patched

    def training_step(self, batch, batch_idx):
        x, y = batch  # (B, 1, T, H, W)
        B = x.shape[0]

        recon, x_masked, mask, ids_masked, pred, target = self(x)

        # Compute loss only on masked patches
        mask = mask.bool()
        # print(pred.shape)
        # print(target.shape)
        # print(mask.shape)
        # Flatten batch and positions so we can index directly
        pred_masked   = pred[mask]    # (num_masked, embed_dim)
        target_masked = target[mask]  # (num_masked, embed_dim)
        # print('pred_masked min',pred_masked.min())
        # print('pred_masked max',pred_masked.max())
        # print('target_masked min',target_masked.min())
        # print('target_masked max',target_masked.max())
        
        # pred_masked = pred[mask].view(B, -1, D)
        # target_masked = target[mask].view(B, -1, D)
        
        # print(pred_masked.shape)
        # print(target_masked.shape)
        # pred_masked = pred[mask].view(B, -1, pred.size(-1))
        # target_masked = target[mask].view(B, -1, target.size(-1))


        # Compute loss only on masked patches
        loss = self.criterion(pred_masked, target_masked)
        self.train_loss_history.append(loss.item())  # ✅ Track loss here
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        recon, mask, _, _, _, _ = self(x)
        loss = self.criterion(recon, x)
        self.log('val_loss', loss, prog_bar=True)
        # Save first batch to visualize later
        if batch_idx == 0:
            self.val_batch_for_viz = (x, mask, recon)
        return loss

    def on_validation_epoch_end(self):
        # Visualize every 10 epochs
        if (self.current_epoch + 1) % 2 == 0:
            if hasattr(self, 'val_batch_for_viz'):
                x,mask, recon = self.val_batch_for_viz
                # visualize_reconstruction_with_mask(x, recon, mask, sample_idx=0, num_frames=4)
                visualize_reconstruction(x, recon, sample_idx=0, num_frames=4,save_path="results/reconstruction_.png")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)#, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return [optimizer], [scheduler]


model = MAEPretrainSwin()
trainer = pl.Trainer(
    max_epochs=200,
    accelerator='auto',
    log_every_n_steps=20,
    check_val_every_n_epoch=20,
)
trainer.fit(model, train_loader,train_loader)