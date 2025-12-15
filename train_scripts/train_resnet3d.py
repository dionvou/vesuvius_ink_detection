import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import random
from transformers import SegformerForSemanticSegmentation
import yaml

import numpy as np

import wandb

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import pandas as pd
import os
import random
from contextlib import contextmanager
import cv2

import scipy as sp
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW

import datetime
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from models.i3dallnl import InceptionI3d
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
from scipy import ndimage
from models.resnetall import generate_model
import PIL.Image
from PIL import Image, ImageOps
from skimage.transform import resize
import tifffile as tiff
from models import unetr
import numpy as np
from scipy.ndimage import zoom

from pytorch_lightning.utilities import rank_zero_only
import csv
import os
from datetime import datetime
import utils

print = rank_zero_only(print)

PIL.Image.MAX_IMAGE_PIXELS = 933120000

class CFG:
    # ============== comp exp name =============
    current_dir = './'
    segment_path = './train_scrolls/'
    
    start_idx = 15
    in_chans = 30
    valid_chans = 26
    
    size = 256
    tile_size = 256
    stride = tile_size // 8

    train_batch_size = 30
    valid_batch_size = 50   
    lr = 2e-5
    # ============== model cfg =============
    scheduler = 'cosine'
    epochs = 4
    
    # Change the size of fragments2
    frags_ratio1 = ['Frag','re']
    frags_ratio2 = ['s4','202','flat']
    ratio1 = 2
    ratio2 = 2
    
    # ============== fold =============
    segments = ['20231210132040','Frag5','Frag1']#['Frag5','20231210132040']#,'frag4','frag3','frag2','frag1']
    valid_id = '20231210132040'#20231210132040'20231215151901
    norm = False
    aug = None
    # ============== fixed =============
    min_lr = 1e-7
    weight_decay = 1e-5
    max_grad_norm = 1
    num_workers = 8
    warmup_factor = 10

    seed = 0
    
    # ============== comp exp name =============
    comp_name = 'vesuvius'
    exp_name = 'resnet3d101'

    outputs_path = f'./outputs/{exp_name}/'
    model_dir = outputs_path
        
    train_aug_list = [
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.15,p=0.75),
        # A.RandomBrightnessContrast(p=0.75, brightness_limit=(-0.2, 0.4), contrast_limit=(-0.2, 0.2)),
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
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    

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


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')

class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask

class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=CFG.size,enc='',with_norm=False,total_steps=780,a=0.7,b=0.3, smooth = 0.25, dropout=None, max=True):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')#,smooth = self.hparams.smooth)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=self.hparams.smooth)
        self.loss_func= lambda x,y: self.hparams.a * self.loss_func1(x,y) + self.hparams.b*self.loss_func2(x,y)
        self.mask_gt = np.zeros(self.hparams.pred_shape)
        
        self.backbone = generate_model(model_depth=101, n_input_channels=1,forward_features=True,n_classes=1039)
        state_dict=torch.load('./checkpoints/r3d101_KM_200ep.pth')["state_dict"]
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.backbone.load_state_dict(state_dict,strict=False)
        # self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=True)
        # self.backbone.load_state_dict(torch.load('./pretraining_i3d_epoch=3.pt'),strict=False)
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)
        
        # # Example: freeze conv1 and layer1
        # for name, param in self.backbone.named_parameters():
        #     if name.startswith(("conv1", "layer1", "layer2",'layer3')):
        #         param.requires_grad = False
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # # Segformer expects 2D input with shape (B, C, H, W)
        # self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
        #     "nvidia/mit-b0",
        #     num_labels=1,
        #     ignore_mismatched_sizes=True,
        #     num_channels=3
        # )        
        # for param in self.encoder_2d.parameters():
        #     param.requires_grad = False

        # init_weights(self.decoder)

        self.normalization=nn.BatchNorm3d(num_features=1) 
    
    # BACKBONE FORWARD
    def forward(self, x):
        x = x.permute(0,2,1,3,4)  # (B, C, D, H, W) -> (B, D, C, H, W)
        if self.hparams.with_norm == True:
            x=self.normalization(x)
        feat_maps = self.backbone(x)
        if self.hparams.max==True:
            feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        else:
            feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]

        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        
        if torch.isnan(loss1):
            print("Loss nan encountered")
        # Log the loss
        self.log("train/total_loss", loss1.item(), on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"],on_step=True, prog_bar=True,sync_dist=True)

        return {"loss": loss1}
    
    def validation_step(self, batch, batch_idx):
        x, y, xyxys = batch
        batch_size = x.size(0)
        outputs = self(x)
        
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        y_cpu = y.to('cpu')  # Move ground truth to CPU
        
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            # Prediction
            pred_patch = F.interpolate(
                y_preds[i].unsqueeze(0).float(),
                size=(self.hparams.size, self.hparams.size),
                mode='bilinear'
            ).squeeze(0).squeeze(0).numpy()
            self.mask_pred[y1:y2, x1:x2] += pred_patch
            
            # Ground truth
            gt_patch = F.interpolate(
                y_cpu[i].unsqueeze(0).float(),
                size=(self.hparams.size, self.hparams.size),
                mode='bilinear'
            ).squeeze(0).squeeze(0).numpy()
            self.mask_gt[y1:y2, x1:x2] = gt_patch  # Use = not += for ground truth
            
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)        
        return {"loss": loss1}

    def on_validation_epoch_end(self):
        # Average overlapping predictions
        self.mask_pred = np.divide(
            self.mask_pred, 
            self.mask_count, 
            out=np.zeros_like(self.mask_pred), 
            where=self.mask_count != 0
        )
        
        # Get ground truth mask
        pred_binary = (self.mask_pred > 0.5).astype(np.float32)
        
        # Calculate metrics
        if hasattr(self, 'mask_gt'):
            gt_binary = self.mask_gt.astype(np.float32)
            
            # Intersection and Union for IoU
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            iou = intersection / (union + 1e-8)
            
            # Dice coefficient
            dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
            
            # Pixel accuracy
            correct = (pred_binary == gt_binary).sum()
            total = gt_binary.size
            pixel_acc = correct / total
            
            # Precision and Recall
            tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
            fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
            fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Log metrics
            self.log("val/iou", iou, prog_bar=True, sync_dist=True)
            self.log("val/dice", dice, prog_bar=True, sync_dist=True)
            self.log("val/pixel_acc", pixel_acc, prog_bar=True, sync_dist=True)
            self.log("val/precision", precision, sync_dist=True)
            self.log("val/recall", recall, sync_dist=True)
            self.log("val/f1", f1, sync_dist=True)
            
            # Print only on master process (rank 0)
            if self.trainer.is_global_zero:
                print(f"\n=== Validation Metrics ===")
                print(f"IoU: {iou:.4f}")
                print(f"Dice: {dice:.4f}")
                print(f"Pixel Accuracy: {pixel_acc:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
                
        # === Save metrics locally per experiment ===
        if self.trainer.is_global_zero:
            exp_dir = os.path.join(self.trainer.default_root_dir, "metrics")
            os.makedirs(exp_dir, exist_ok=True)

            csv_path = os.path.join(exp_dir, "metrics.csv")

            # Prepare header if file does not exist
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                
                # Write CSV header once
                if not file_exists:
                    writer.writerow([
                        "timestamp",
                        "epoch",
                        "iou",
                        "dice",
                        "pixel_acc",
                        "precision",
                        "recall",
                        "f1"
                    ])

                # Append this epoch’s metrics
                writer.writerow([
                    datetime.now().isoformat(),
                    self.current_epoch,
                    float(iou),
                    float(dice),
                    float(pixel_acc),
                    float(precision),
                    float(recall),
                    float(f1)
                ])
        
        # Log image only on master process
        if self.trainer.is_global_zero:
            wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred, 0, 1)], caption=["probs"])

        # Reset masks
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        if hasattr(self, 'mask_gt'):
            self.mask_gt = np.zeros(self.hparams.pred_shape)
                
    def configure_optimizers(self):
        based_lr = CFG.lr
        param_groups = [
            {'params': self.backbone.parameters(), 'lr': based_lr, 'weight_decay': 5e-3},
            {'params': self.decoder.parameters(), 'lr': based_lr, 'weight_decay': 5e-3},
        ]
           
            
        optimizer = AdamW(param_groups)
        # # Scheduler for OneCycleLR
        # steps_per_epoch = 143 * 4  # adjust as per your dataloader
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=[group['lr'] for group in param_groups],  # pass per-group max_lr
        #     pct_start=0.35,
        #     steps_per_epoch=143,
        #     epochs=25,
        #     final_div_factor=1e2
        # )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[group['lr'] for group in param_groups],
            steps_per_epoch=143,
            epochs=self.trainer.max_epochs,
            pct_start=0.1
            
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def predict_step(self, batch, batch_idx):
        """Add this to your RegressionPLModel class"""
        x, y, xyxys = batch
        
        outputs = self(x)
        y_preds = torch.sigmoid(outputs)
        
        return {
            'preds': y_preds.cpu(),
            'xyxys': xyxys
        }
    

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

torch.cuda.empty_cache()

torch.set_float32_matmul_precision('medium')

         
for f in [['20231210132040','Frag5','Frag1']]:  #'s4','omega',
    smooth = 0.25
    
    for norm in [True,False]:
        # norm = False
        # CFG.valid_id = f[0]
        CFG.frags = f
        max = True
        a=0.6

        enc='resnet101'
        fragment_id = CFG.valid_id
        valid_mask_gt = cv2.imread(f"train_scrolls/{fragment_id}/layers/32.tif", 0)

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
        print(len(train_images))

        valid_xyxys = np.stack(valid_xyxys)

        train_dataset = utils.VideoDataset(
            train_images, CFG, labels=train_masks, transform=utils.get_transforms(data='train', cfg=CFG), scale_factor=4,norm=CFG.norm)
        
        valid_dataset = utils.VideoDataset(
            valid_images, CFG,xyxys=valid_xyxys, labels=valid_masks, transform=utils.get_transforms(data='valid', cfg=CFG), scale_factor=4,norm=CFG.norm)
        
        train_loader = DataLoader(train_dataset,
                                    batch_size=CFG.train_batch_size,
                                    shuffle=True,
                                    num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                    )
        valid_loader = DataLoader(valid_dataset,
                                    batch_size=CFG.valid_batch_size,
                                    shuffle=False,
                                    num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
        
        print('Trainloader lenth: ',len(train_loader))

        run_slug=f'RESNET_{CFG.frags}_valid={CFG.valid_id}_size={CFG.size}_lr={CFG.lr}_in_chans={CFG.in_chans}_norm={norm}_a={a}_max={max}_smooth={smooth}'

        wandb_logger = WandbLogger(project="vesuvius_ink_detection",name=run_slug)
        model=RegressionPLModel(enc='resnet101',pred_shape=pred_shape,size=CFG.size,total_steps=len(train_loader), with_norm=norm, a = a,b = 1-a,max= max, smooth=smooth)
        
        # # DION
        # checkpoint = torch.load("checkpoints/RESNET_['20231210132040', 'Frag5']_valid=flatboi3_size=256_lr=5e-05_in_chans=30_norm=True_a=0.7_max=True_smooth=0.25_epoch=epoch=7.ckpt", map_location="cpu", weights_only=False)
        # model.load_state_dict(checkpoint["state_dict"], strict=True)

        wandb_logger.watch(model, log="all", log_freq=50)
        trainer = pl.Trainer(
        max_epochs=40,
        accelerator="gpu",
        devices=-1,
        check_val_every_n_epoch=4,
        logger=wandb_logger,
        default_root_dir="./models",
        accumulate_grad_batches=1,
        precision='16',
        gradient_clip_val=5.0,
        gradient_clip_algorithm="norm",
        strategy='ddp_find_unused_parameters_true',
        callbacks=[ModelCheckpoint(filename=run_slug+f'_epoch='+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),
                    ],
        )
        
        # trainer.validate(model=model, dataloaders=valid_loader, verbose=True)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        wandb.finish()


