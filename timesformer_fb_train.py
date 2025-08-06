import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoImageProcessor
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
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
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = 933120000

import utils
import models.timesformer_fb as timesformer_fb

class CFG:
    # ============== comp exp name =============
    current_dir = './'
    segment_path = './train_scrolls/'
    
    start_idx = 14
    in_chans = 32
    
    size = 224
    tile_size = 224
    stride = tile_size // 8 
    
    train_batch_size =  8 # 32
    valid_batch_size = 8
    
    # Size of fragments
    frags_ratio1 = ['frag','202','left','s4']
    frags_ratio2 = ['nothing']
    ratio1 = 2
    ratio2 = 1
    
    # ============== fold =============
    segments = ['frag1','20231210132040'] 
    valid_id = '20231210132040'
    
    # ============== model cfg =============

    num_workers = 8
    scheduler = 'linear' # 'cosine', 'linear'
    epochs = 30
    warmup_factor = 10
    lr = 5e-5
    # ============== fixed =============
    min_lr = 1e-7
    weight_decay = 1e-6
    max_grad_norm = 100
    num_workers = 8
    seed = 0
    
    # ============== comp exp name =============
    comp_name = 'vesuvius'
    exp_name = 'pretraining_all'

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
        # A.Normalize(
        #     mean= [0] * in_chans,
        #     std= [1] * in_chans
        # ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        # A.Normalize(
        #     mean= [0] * in_chans,
        #     std= [1] * in_chans
        # ),
        ToTensorV2(transpose_mask=True),  
    ]
    
def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug   
        
utils.cfg_init(CFG)
torch.set_float32_matmul_precision('medium')


fragment_id = CFG.valid_id
run_slug=f'TF_{CFG.segments}_valid={CFG.valid_id}_size={CFG.size}_lr={CFG.lr}_in_chans={CFG.in_chans}'
valid_mask_gt = cv2.imread(f"{CFG.segment_path}{fragment_id}/{fragment_id}_inklabels.png", 0)

# pred_shape=valid_mask_gt.shape
# if (any(sub in fragment_id for sub in ["frag", "rect", "vals", "remaining"]) or fragment_id in ["20231210132040"]):
#     # For fragments with 2x downsampled masks
#     pred_shape = tuple(s // 2 for s in valid_mask_gt.shape)

pred_shape=valid_mask_gt.shape
if (any(sub in fragment_id for sub in CFG.frags_ratio1)):
    pred_shape = tuple(s // CFG.ratio1 for s in valid_mask_gt.shape)
elif (any(sub in fragment_id for sub in CFG.frags_ratio2)):
    pred_shape = tuple(s // CFG.ratio2 for s in valid_mask_gt.shape)
else:
    pass

# pred_shape=valid_mask_gt.shape
# if (any(sub in fragment_id for sub in ["frag", "rect", "vals", "remaining"]) or fragment_id in ["20231210132040"]):
#     pred_shape = tuple(s // 2.5 for s in valid_mask_gt.shape)
#     # pred_shape = tuple(int(s / 1.8)  for s in valid_mask_gt.shape)

# elif any(sub in fragment_id for sub in ["PHerc"]):
#     pred_shape = tuple(s // 3.5 for s in valid_mask_gt.shape)
#     # pred_shape = tuple(int(s / 3.5)  for s in valid_mask_gt.shape)
train_images, train_masks, valid_images, valid_masks, valid_xyxys = utils.get_train_valid_dataset(CFG)
print('train_images',train_images[0].shape)
print("Length of train images:", len(train_images))

valid_xyxys = np.stack(valid_xyxys)
train_dataset = timesformer_fb.TimesformerDataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
valid_dataset = timesformer_fb.TimesformerDataset(
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

wandb_logger = WandbLogger(project="vesivus",name=run_slug)  

model = timesformer_fb.TimesfomerModel(pred_shape=pred_shape, size=CFG.size, lr=CFG.lr, scheduler=CFG.scheduler, wandb_logger=wandb_logger)
wandb_logger.watch(model, log="all", log_freq=100)
# model = timesformer_fb.load_weights(model,"outputs/vesuvius/pretraining_all/vesuvius-models/TF_['rect55', 'remaining5']_valid=rect55_size=224_lr=0.0001_in_chans=16_epoch=15.ckpt")
trainer = pl.Trainer(
    max_epochs=CFG.epochs,
    accelerator="gpu",
    check_val_every_n_epoch=2,
    devices=-1,
    logger=wandb_logger,
    default_root_dir="./modelss",
    accumulate_grad_batches=1,
    precision='16-mixed',
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    strategy='ddp',
    callbacks=[ModelCheckpoint(filename=f'{run_slug}_'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),
    ]

)
# trainer.validate(model=model, dataloaders=valid_loader, verbose=True)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
wandb.finish()