import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


from transformers import AutoImageProcessor

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import random

import numpy as np
import pandas as pd
import scipy as sp
import cv2

import wandb

from contextlib import contextmanager

from tqdm.auto import tqdm

import argparse

import datetime
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from warmup_scheduler import GradualWarmupScheduler
from scipy import ndimage
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000


import sys
sys.path.append('/home/ubuntu/TimeSformer')  # Path to TimeSformer repository
from timesformer.models.vit import TimeSformer
import vesuvius_ink_detection.utils as utils

class CFG:
    current_dir = './'
    segment_path = './train_scrolls/'
    
    start_idx = 24
    in_chans = 16
    
    size = 224
    tile_size = 224
    stride = tile_size //2
    
    train_batch_size =  10 # 32
    valid_batch_size = 25
    
    lr = 1e-4
    num_workers = 8
    # ============== fold =============
    valid_id = 'None'
    # # ============== comp exp name =============
    comp_name = 'vesuvius'
    exp_name = 'pretraining_all'

    seed = 130697

    outputs_path = f'./outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'
    
    
    outputs_path = f'./outputs/{comp_name}/{exp_name}/'
    model_dir = outputs_path + \
        f'{comp_name}-models/'
        
    # ============== augmentation =============
    train_aug_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.15,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
                        mask_fill_value=0, p=0.5),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),

        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),

        ToTensorV2(transpose_mask=True),
        
    ]
    pil_transform = T.Compose([
        T.ToPILImage(),                    # convert (C, H, W) to PIL
        T.Grayscale(num_output_channels=3),  # convert to 3 channels
    ])

    rotate = A.Compose([A.Rotate(5,p=1)])
    
def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
    
def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)
        
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)

    if mode == 'train':
        make_dirs(cfg)
        
cfg_init(CFG)

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

class CustomDataset(Dataset):
    def __init__(self, images ,cfg,xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        
        self.transform = transform
        self.xyxys=xyxys
        self.rotate=CFG.rotate
        
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600", use_fast=True)
        self.pil_transform  = CFG.pil_transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.xyxys is not None: # VALID
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            
            image = image.permute(1,0,2,3)
            frames = [self.pil_transform(frame.squeeze(0)) for frame in image] 

            encoding = self.processor(
                [frame for frame in frames],   # list of PIL
                return_tensors='pt'
                )
            pixel_values = encoding["pixel_values"].squeeze(0)
            return pixel_values, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            image = image.permute(1,0,2,3)
            frames = [self.pil_transform(frame.squeeze(0)) for frame in image] 
            
            encoding = self.processor(
                [frame for frame in frames],   # list of PIL
                return_tensors='pt'
                )
            pixel_values = encoding["pixel_values"].squeeze(0)
            
            return pixel_values, label

class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch_size, patch_size),
                              stride=(1, patch_size, patch_size))
        self.num_patches = (img_size // patch_size) ** 2 * num_frames

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.proj(x)  # -> [B, D, T, H', W']
        B, D, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, D)  # [B, N, D]
        return x
    
class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=768, decoder_dim=512, num_patches=196*8, num_layers=4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, decoder_dim))

        self.decoder_blocks = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=8) for _ in range(num_layers)]
        )
        self.decoder_pred = nn.Linear(decoder_dim, embed_dim)

    def forward(self, x_encoded, ids_restore):
        B, N_vis, D = x_encoded.shape
        N_full = ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(B, N_full - N_vis, -1)

        x_ = torch.cat([x_encoded, mask_tokens], dim=1)  # [B, N_full, D]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x_ = x_ + self.pos_embed

        x_ = self.decoder_blocks(x_)
        return self.decoder_pred(x_)

class MAETimeSformerEncoder(TimeSformer):
    def __init__(self, *args, mask_ratio=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio

    def random_masking(self, x):
        """
        x: (B, N, D)
        Returns:
            x_masked: (B, N_visible, D)
            mask: (B, N), 0 is keep, 1 is mask
            ids_restore: (B, N)
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Create mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):
        # Step 1: Patch embed
        x = self.model.patch_embed(x)  # (B, C, T, H', W') → (B, N_patches, D)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # → (B, N, D)

        # Step 2: Positional encoding (skip cls token)
        x = x + self.pos_embed[:, 1:, :]  # (B, N, D)

        # Step 3: MAE masking
        x_visible, mask, ids_restore = self.random_masking(x)

        # Step 4: Dropout and Transformer blocks
        x_visible = self.pos_drop(x_visible)

        for blk in self.blocks:
            x_visible = blk(x_visible)

        x_visible = self.norm(x_visible)

        return x_visible, mask, ids_restore
    

class MAEPLModel(pl.LightningModule):
    def __init__(self, size=CFG.size, enc='', total_steps=780):
        super(MAEPLModel, self).__init__()

        self.save_hyperparameters()
        self.loss_func = utils.mae_loss
        
        self.backbone = MAETimeSformerEncoder(img_size=448, num_classes=0, num_frames=16, attention_type='divided_space_time', in_chans=3)
        checkpoint = torch.load('checkpoints/TimeSformer_divST_16x16_448_K600.pyth')
        state_dict = checkpoint['model_state']
        
        state_dict.pop('model.head.weight', None)
        state_dict.pop('model.head.bias', None)

        self.backbone.load_state_dict(state_dict, strict=True)
        
        self.decoder = MAEDecoder()

    def forward(self, x):
        x = self.backbone(torch.permute(x, (0, 2, 1,3,4)))
        print(x.shape)
        # x=x.view(-1,1,4,4)
        return x
    
    def random_masking(x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # binary mask: 0 for keep, 1 for remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=CFG.lr)
        # scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4,pct_start=0.15, steps_per_epoch=self.hparams.total_steps, epochs=50,final_div_factor=1e2)
        scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4,pct_start=0.15, steps_per_epoch=self.hparams.total_steps, epochs=20,final_div_factor=1e2)

        return [optimizer],[scheduler]



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 50, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)



torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')


train_images, train_masks, valid_images, valid_masks, valid_xyxys = utils.get_train_valid_dataset(CFG)
print(f"Train images shape: {len(train_images)}")
train_dataset = CustomDataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))

train_loader = DataLoader(train_dataset,
                                batch_size=CFG.train_batch_size,
                                shuffle=True,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                )


model = MAEPLModel(enc='timesformer', size=CFG.size, total_steps=len(train_loader))


wandb_logger = WandbLogger(project="vesivus",name="MAE")
wandb_logger.watch(model, log="all", log_freq=100)

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu",
    devices=-1,
    check_val_every_n_epoch=2,
    default_root_dir="./models",
    accumulate_grad_batches=1,
    logger=wandb_logger,
    precision='16-mixed',
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    strategy='ddp_find_unused_parameters_true',
)
trainer.fit(model=model, train_dataloaders=train_loader)

wandb.finish()

# fragment_id = CFG.valid_id

# # valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_mask.png", 0)
# # valid_mask_gt=cv2.resize(valid_mask_gt,(valid_mask_gt.shape[1]//2,valid_mask_gt.shape[0]//2),cv2.INTER_AREA)
# # pred_shape=valid_mask_gt.shape
# # pred_shape = tuple(s // 2 for s in valid_mask_gt.shape)

# torch.set_float32_matmul_precision('medium')

# # fragments=['20231210121321']
# # fragments=['vals4']

# enc_i,enc,fold=0,'i3d',0
# for fid in fragments:
#     CFG.valid_id=fid
#     fragment_id = CFG.valid_id
#     run_slug=f'{CFG.valid_id}_{CFG.size}x{CFG.size}_lr={CFG.lr}_in_chans={CFG.in_chans}'


#     valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_mask.png", 0)

#     # pred_shape=valid_mask_gt.shape
#     pred_shape = tuple(s //2 for s in valid_mask_gt.shape)

#     train_images, train_masks, valid_images, valid_masks, valid_xyxys = utils.get_train_valid_dataset()
#     print(len(train_images))
    
#     valid_xyxys = np.stack(valid_xyxys)
#     train_dataset = CustomDataset(
#         train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
    
#     valid_dataset = CustomDataset(
#         valid_images, CFG,xyxys=valid_xyxys, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))
    
#     train_loader = DataLoader(train_dataset,
#                                 batch_size=CFG.train_batch_size,
#                                 shuffle=True,
#                                 num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
#                                 )
#     valid_loader = DataLoader(valid_dataset,
#                                 batch_size=CFG.valid_batch_size,
#                                 shuffle=False,
#                                 num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    
#     print('Trainloader lenth: ',len(train_loader))

    # wandb_logger = WandbLogger(project="vesivus",name=run_slug+f'{enc}_finetune')
#     norm=fold==1
#     model = RegressionPLModel(enc='timesformer',pred_shape=pred_shape,size=CFG.size,total_steps=len(train_loader))
#     print('FOLD : ',fold)
#     wandb_logger.watch(model, log="all", log_freq=100)
#     multiplicative = lambda epoch: 0.9
    


#     trainer = pl.Trainer(
#         max_epochs=30,
#         accelerator="gpu",
#         devices=-1,
#         check_val_every_n_epoch=2,
#         logger=wandb_logger,
#         default_root_dir="./models",
#         accumulate_grad_batches=1,
#         precision='16-mixed',
#         gradient_clip_val=1.0,
#         gradient_clip_algorithm="norm",
#         strategy='ddp_find_unused_parameters_true',
#         callbacks=[ModelCheckpoint(filename=f'f15_div2_l8-38_{fid}_{fold}_fr_{enc}'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),
#                    ],
#     )
    
#     trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

