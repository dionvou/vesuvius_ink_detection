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

import utils
import models.swin as swin

class CFG:
    # ============== comp exp name =============
    current_dir = './'
    segment_path = './train_scrolls/'
    
    start_idx = 22
    in_chans = 16
    
    size = 224
    tile_size = 224
    stride = tile_size // 8
    
    train_batch_size =  20 # 32
    valid_batch_size = 20
    lr = 8e-5
    num_workers = 8
    # ============== model cfg =============
    scheduler = 'onecycle'#'cosine', 'linear'
    epochs = 50
    warmup_factor = 10
    
    # Change the size of fragments
    frags_ratio1 = ['frag','re']
    frags_ratio2 = ['s4','20231210132040']
    ratio1 = 2
    ratio2 = 1
    
    # ============== fold =============
    segments = ['frag5','frag1','20231210132040'] 
    valid_id = '20231210132040'#20231215151901'
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
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
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
for inchans in [16]:
    for lr in [2e-5]:
        for freeze in [False]:
            for frags in [['s4','Frag5','20231210132040'],['s4','20231210132040'],['Frag5','20231215151901'],['Frag1','20231215151901']]:
                CFG.segments = frags
                CFG.lr = lr
                CFG.in_chans = inchans
                if inchans ==20:
                    CFG.train_batch_size=7
                    CFG.valid_batch_size=7
                    CFG.start_idx=20
                elif inchans==16:
                    CFG.train_batch_size=12
                    CFG.valid_batch_size=12
                    CFG.start_idx=23
               

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
                train_dataset = swin.TimesformerDataset(
                    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
                valid_dataset = swin.TimesformerDataset(
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

                wandb_logger = WandbLogger(project="vesivus", name=run_slug)  
                wandb.finish()
                model = swin.SwinModel(pred_shape=pred_shape, size=CFG.size, lr=CFG.lr, scheduler=CFG.scheduler, wandb_logger=wandb_logger,freeze=False)
                wandb_logger.watch(model, log="all", log_freq=50)

                # model = swin.load_weights(model,"outputs/vesuvius/pretraining_all/vesuvius-models/SWIN_['frag5', '20231210132040']_valid=20231210132040_size=224_lr=2e-05_in_chans=16_epoch=5.ckpt")
                trainer = pl.Trainer(
                    max_epochs=CFG.epochs,
                    accelerator="gpu",
                    check_val_every_n_epoch=4,
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