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

PIL.Image.MAX_IMAGE_PIXELS = 9331200000

import utils
import models.timesformer_hug as timesformer_hug


class CFG:
    # ============== comp exp name =============
    current_dir = './'
    segment_path = './train_scrolls/'
    
    start_idx = 20
    in_chans = 16
    
    size = 224
    tile_size = 224
    stride = tile_size // 8
    
    train_batch_size =  12# 32
    valid_batch_size = 25
    check_val = 4
    lr = 2e-5
    
    # Change the size of fragments
    
    frags_ratio1 = ['frag','re']
    frags_ratio2 = ['202','s4','left']
    ratio1 = 2
    ratio2 = 2
    
    # ============== fold =============
    segments = ['frag5','frag1','20231210132040'] 
    valid_id = '20231210132040'
    # segments = ['rect1','remaining1'] 
    # valid_id = 'rect1'#20231210132040'20231215151901
    
    num_workers = 8
    # ============== model cfg =============
    scheduler = 'cosine' # 'cosine', 'linear'
    epochs = 50
    warmup_factor = 10
    min_lr = 1e-7
    weight_decay = 1e-6
    max_grad_norm = 100
    num_workers = 16
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
        
utils.cfg_init(CFG)
torch.set_float32_matmul_precision('medium')

fragment_id = CFG.valid_id
run_slug=f'TF_{CFG.segments}_valid={CFG.valid_id}_size={CFG.size}_lr={CFG.lr}_in_chans={CFG.in_chans}'
# valid_mask_gt = cv2.imread(f"{CFG.segment_path}{fragment_id}/{fragment_id}_inklabels.png", 0)
path = f"{CFG.segment_path}{fragment_id}/{fragment_id}_inklabels.png"
valid_mask_gt = Image.open(path).convert("L")
valid_mask_gt = np.array(valid_mask_gt)

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
train_dataset = timesformer_hug.TimesformerDataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
valid_dataset = timesformer_hug.TimesformerDataset(
    valid_images, CFG, xyxys=valid_xyxys, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

train_loader = DataLoader(train_dataset,
                            batch_size=CFG.train_batch_size,
                            shuffle=True,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True,persistent_workers=True,prefetch_factor=2
                            )
valid_loader = DataLoader(valid_dataset,
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True,persistent_workers=True,prefetch_factor=2)

print(f"Train loader length: {len(train_loader)}")
print(f"Valid loader length: {len(valid_loader)}")

wandb_logger = WandbLogger(project="vesivus",name=run_slug)  

model = timesformer_hug.TimesfomerModel(pred_shape=pred_shape, size=CFG.size, lr=CFG.lr, scheduler=CFG.scheduler, wandb_logger=wandb_logger)
wandb_logger.watch(model, log="all", log_freq=100)

model = timesformer_hug.load_weights(model,"outputs/vesuvius/pretraining_all/vesuvius-models/TF_['frag5', 'frag1', '20231215151901']_valid=20231215151901_size=224_lr=2e-05_in_chans=16_epoch=7.ckpt")
trainer = pl.Trainer(
    max_epochs=CFG.epochs,
    accelerator="gpu",
    check_val_every_n_epoch=CFG.check_val,
    devices=-1,
    logger=wandb_logger,
    default_root_dir="./modelss",
    accumulate_grad_batches=1,
    precision='16-mixed',
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    strategy='ddp_find_unused_parameters_true',
    callbacks=[ModelCheckpoint(filename=f'{run_slug}_'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),
    ]

)
trainer.validate(model=model, dataloaders=valid_loader, verbose=True)
# trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
wandb.finish()