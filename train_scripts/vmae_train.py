import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import cv2
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from warmup_scheduler import GradualWarmupScheduler
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import utils
from models import vmae
# import models.vmae as vmae

class CFG:
    # ============== comp exp name =============
    current_dir = './'
    segment_path = './train_scrolls/'
    
    start_idx = 28
    in_chans = 8
    valid_chans = 8
    
    size = 64
    tile_size = 128
    stride = tile_size // 8

    train_batch_size = 60
    valid_batch_size = 512
    lr = 5e-5
    # ============== model cfg =============
    scheduler = 'cosine'
    epochs = 4
    
    # Change the size of fragments2
    frags_ratio1 = ['Frag','re']
    frags_ratio2 = ['s4','202','left']
    ratio1 = 2
    ratio2 = 2
    
    # ============== fold =============
    segments = ['20240304144031','Frag5']#['Frag5','20231210132040']#,'frag4','frag3','frag2','frag1']
    valid_id = '20240304144031'#20231210132040'20231215151901
    norm = True
    aug = None
    # ============== fixed =============
    min_lr = 1e-7
    weight_decay = 1e-6
    max_grad_norm = 1
    num_workers = 8
    warmup_factor = 10

    seed = 0
    
    # ============== comp exp name =============
    comp_name = 'vesuvius'
    exp_name = 'videomae'

    outputs_path = f'./outputs/{exp_name}/'
    model_dir = outputs_path
        
    # ============== augmentation =============
    train_aug_list = [
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.75),
        # A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.,p=0.75),
        # # A.OneOf([
        # #         A.GaussNoise(var_limit=[10, 50]),
        # #         # A.GaussianBlur(),
        # #         # A.MotionBlur(),
        # #         ], p=0.4),
        # A.CoarseDropout(max_holes=5, max_width=int(size * 0.1), max_height=int(size * 0.2), 
        #                 mask_fill_value=0, p=0.5),
        # ToTensorV2(transpose_mask=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        ),

        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.0,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7
        ),

        A.GaussNoise(var_limit=(3, 10), p=0.3),   # VERY mild noise only
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

# End any existing run (if still active)
if wandb.run is not None:
    wandb.finish()
  
utils.cfg_init(CFG)
torch.set_float32_matmul_precision('medium')

fragment_id = CFG.valid_id
run_slug=f'_VIDEOMAE_{CFG.segments}_valid={CFG.valid_id}_size={CFG.size}_lr={CFG.lr}_in_chans={CFG.valid_chans},norm={CFG.norm},fourth={CFG.aug}'

valid_mask_gt = cv2.imread(f"{CFG.segment_path}{fragment_id}/layers/32.tif", 0)
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
train_dataset = utils.VideoDataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG),norm=True, aug=None,scale_factor=16)
valid_dataset = utils.VideoDataset(
    valid_images, CFG, xyxys=valid_xyxys, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG),norm=True,scale_factor=16)

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

wandb_logger = WandbLogger(project="vesuvius_ink_detection", name=run_slug)  
model = vmae.VideoMaeModel(pred_shape=pred_shape, size=CFG.size, lr=CFG.lr, scheduler=CFG.scheduler, wandb_logger=wandb_logger,freeze=False)
wandb_logger.watch(model, log="all", log_freq=50)

# model = swin.load_weights(model,"outputs/vesuvius/pretraining_all/vesuvius-models/1_SWIN_['frag5', '20231215151901']_valid=20231215151901_size=224_lr=2e-05_in_chans=8,norm=False,fourth=shift_epoch=7.ckpt")
trainer = pl.Trainer(
    max_epochs=40,
    accelerator="gpu",
    check_val_every_n_epoch=4,
    devices=-1,
    logger=wandb_logger,
    default_root_dir="./modelss",
    accumulate_grad_batches=1,
    precision='16',
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    strategy="ddp_find_unused_parameters_false",

    # callbacks=[ModelCheckpoint(filename=f'{run_slug}_'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),
    # ]

)
# trainer.validate(model=model, dataloaders=valid_loader, verbose=True)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
wandb.finish()