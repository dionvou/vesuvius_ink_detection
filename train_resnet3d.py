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
import pandas as pd

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
from PIL import Image
from skimage.transform import resize
import tifffile as tiff

PIL.Image.MAX_IMAGE_PIXELS = 933120000
class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './'
    comp_folder_name = './'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'./'
    
    exp_name = 'pretraining_all'
    
    # ============== model cfg =============
    frags = ['20231210132040','frag1','frag5']#,'20231210132040']#'frag1',
    valid_id = '20231210132040'#'20240304141530'#'20231210132040'
    backbone='resnet3d'
    # ============== training cfg =============
    size = 256
    tile_size = 256
    stride = tile_size // 8

    train_batch_size =  12
    valid_batch_size = 25


    scheduler = 'GradualWarmupSchedulerV2'
    
    start_idx = 15
    in_chans = 30
    
    epochs = 21 # 30
    lr = 2e-5
    # ============== fold =============
    

    # ============== fixed =============
    pretrained = True

    num_workers = 8

    seed = 0
    
    frags_ = []

    outputs_path = f'./outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

       # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.6),

        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.1,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=4, max_width=int(size * 0.2), max_height=int(size * 0.2), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * 24,
            std= [1] * 24
        ),
        ToTensorV2(transpose_mask=True),
    ]


    valid_aug_list = [
        A.Resize(size, size),
        # A.ToFloat(max_value=255.0),
        A.Normalize(
            mean= [0] * 24,
            std= [1] * 24
        ),
        ToTensorV2(transpose_mask=True),
    ]

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image_mask(fragment_id, start_idx, in_chans):

    images = []
    start_idx = int(start_idx)
    end_idx = start_idx + in_chans
    idxs = range(start_idx, end_idx)

    for i in idxs:
    
        tif_path = f"train_scrolls/{fragment_id}/layers/{i:02}.tif"
        # jpg_path = f"train_scrolls/{fragment_id}/layers/{i:02}.jpg"

        if os.path.exists(tif_path):
            # image = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
            # image = image.astype(np.float32) / 65535.0  # Normalize to [0, 1] range

            image = cv2.imread(f"train_scrolls/{fragment_id}/layers/{i:02}.tif", 0)
            # print(image.dtype)  # Should say uint16 if it's 16-bit
            # print(image.max())  # Should go up to ~65535
        else:
        # elif  os.path.exists(jpg_path):
        #     image = cv2.imread(f"train_scrolls/{fragment_id}/layers/{i:02}.jpg", 0)
        # else:
            image = cv2.imread(f"train_scrolls/{fragment_id}/layers/{i:02}.png", 0)
            
        # pad0 = (256 - image.shape[0] % 256)
        # pad1 = (256 - image.shape[1] % 256)
        # image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        # if (any(sub in fragment_id for sub in  ["frag", "rect", "remaining","202","left"])):
        #     image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        # else:
        #     image = cv2.resize(image, (image.shape[1]//1,image.shape[0]//1), interpolation = cv2.INTER_AREA)
            # Resize safely if required
        if any(sub in fragment_id for sub in ["frag",'re']):
            new_h, new_w = int(image.shape[0] / 2), int(image.shape[1] / 2)
        elif any(sub in fragment_id for sub in ["202"]):
            new_h, new_w = int(image.shape[0] / 2), int(image.shape[1] / 2)

            
            
        # # HERE RESIZE
        # new_h, new_w = image.shape[0] //2, image.shape[1] //2  # height, width order corrected
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        image_shape = (image.shape[1], image.shape[0])
        # print(image_shape)
        
        # pad0 = (CFG.size - image.shape[0] % CFG.size)
        # pad1 = (CFG.size - image.shape[1] % CFG.size) % CFG.size

        # image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        # image = image.astype(np.float32) / 255  # Normalize 
        image=np.clip(image,20,200)
        images.append(image)
        
    images = np.stack(images, axis=2)
    print(images.shape)
    
    
    # if fragment_id == 'left':
    # images=images[:,:,::-1]

    mask = cv2.imread( f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)
    
    fragment_mask=cv2.imread(f"train_scrolls/{fragment_id}/{fragment_id}_mask.png", 0)
    print("mask_2",fragment_mask.shape)
    # fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    # mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)


    # if fragment_id =='s4':
    #     mask = cv2.resize(mask, (images.shape[1], images.shape[0]), interpolation=cv2.INTER_AREA)
    # else:
    fragment_mask = cv2.resize(fragment_mask, image_shape, interpolation = cv2.INTER_AREA)
    # mask = cv2.resize(mask , (mask.shape[1]//2,mask.shape[0]//2), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask , image_shape, interpolation=cv2.INTER_AREA)
        # mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    mask = mask.astype('float32')
    mask/=255 # Binarize
    
    # fragment_mask = cv2.resize(fragment_mask , image_shape, interpolation=cv2.INTER_AREA)
    # fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    print("mask",mask.shape)
    assert images.shape[0]==mask.shape[0]
    return images, mask,fragment_mask

def get_train_valid_dataset():
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []
        
    for fragment_id in CFG.frags:
        CFG.frags_.append(fragment_id)    
        print('reading ',fragment_id)
        image, mask,fragment_mask = read_image_mask(fragment_id,CFG.start_idx,CFG.in_chans)

        stride= CFG.stride
        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1,stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, stride))
        windows_dict={}
        for a in y1_list:
            for b in x1_list:
                for yi in range(0,CFG.tile_size,CFG.size):
                    for xi in range(0,CFG.tile_size,CFG.size):
                        y1=a+yi
                        x1=b+xi
                        y2=y1+CFG.size
                        x2=x1+CFG.size
                        if fragment_id!=CFG.valid_id:
                            if (y1,y2,x1,x2) not in windows_dict:
                                if not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size]<0.01):
                                    if not np.any(fragment_mask[a:a+ CFG.tile_size, b:b + CFG.tile_size]==0):
                                        train_images.append(image[y1:y2, x1:x2])
                                        
                                        train_masks.append(mask[y1:y2, x1:x2, None])
                                        assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                                        windows_dict[(y1,y2,x1,x2)]='1'
                        if fragment_id==CFG.valid_id:
                            if (y1,y2,x1,x2) not in windows_dict:
                                if not np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size]==0):
                                        valid_images.append(image[y1:y2, x1:x2])
                                        valid_masks.append(mask[y1:y2, x1:x2, None])

                                        valid_xyxys.append([x1, y1, x2, y2])
                                        assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                                        windows_dict[(y1,y2,x1,x2)]='1'


    return train_images, train_masks, valid_images, valid_masks, valid_xyxys

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

class CustomDataset(Dataset):
    def __init__(self, images ,cfg,xyxys=None, labels=None, transform=None, reference_image=None, do_hist_match=False):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        
        self.transform = transform
        self.xyxys=xyxys
        self.rotate=CFG.rotate
        
    def __len__(self):
        return len(self.images)
    
    # def fourth_augment(self,image):
    #     image_tmp = np.zeros_like(image)
    #     cropping_num = random.randint(24, 30)

    #     start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
    #     crop_indices = np.arange(start_idx, start_idx + cropping_num)

    #     start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

    #     tmp = np.arange(start_paste_idx, cropping_num)
    #     np.random.shuffle(tmp)

    #     cutout_idx = random.randint(0, 2)
    #     temporal_random_cutout_idx = tmp[:cutout_idx]

    #     image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

    #     if random.random() > 0.4:
    #         image_tmp[..., temporal_random_cutout_idx] = 0
    #     image = image_tmp
    #     return image
    def fourth_augment(self, image):
        """
        Custom channel augmentation that returns exactly 24 channels.
        """
        # always select 24 channels
        cropping_num = 24

        # pick crop indices
        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        # pick where to paste them
        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        # container
        image_tmp = np.zeros_like(image)

        # paste cropped channels
        image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

        # optional random cutout
        cutout_idx = random.randint(0, 2)
        # temporal_random_cutout_idx = np.arange(start_paste_idx, start_paste_idx + cutout_idx)
        # if random.random() > 0.4:
        #     image_tmp[..., temporal_random_cutout_idx] = 0

        # finally, keep only 24 channels
        image = image_tmp[..., start_paste_idx:start_paste_idx + cropping_num]

        return image


    
    def z_circular_shift_np(self, volume, max_shift=8, prob=0.5, cutout_size=1, cutout_prob=0.2):
        """
        Circularly shift slices along Z-axis (last dim) by a random integer in [-max_shift, max_shift].
        Then randomly cut out (zero out) a contiguous block of slices along Z-axis.
        
        Args:
            volume: np.ndarray shape (H, W, D) or (C, H, W, D)
            max_shift: max absolute shift (int). shift=0 means no-op.
            prob: probability to apply shift
            cutout_size: number of consecutive slices to cut out
            cutout_prob: probability to apply cutout
        
        Returns:
            volume after augmentation
        """

        if (random.random() > prob) or (max_shift == 0):
            shifted_volume = volume
        else:
            D = volume.shape[-1]
            shift = random.randint(-max_shift, max_shift)
            if shift == 0:
                shifted_volume = volume
            else:
                shifted_volume = np.roll(volume, shift=shift, axis=-1)

        # Apply cutout with given probability
        if (random.random() < cutout_prob) and (cutout_size > 0):
            D = shifted_volume.shape[-1]
            # Ensure cutout size is not larger than volume depth
            cutout_size_clamped = min(cutout_size, D)
            start_idx = random.randint(0, D - cutout_size_clamped)
            # Zero out the block along the last axis
            if shifted_volume.ndim == 3:
                # shape: (H, W, D)
                shifted_volume[:, :, start_idx:start_idx + cutout_size_clamped] = 0
            elif shifted_volume.ndim == 4:
                # shape: (C, H, W, D)
                shifted_volume[:, :, :, start_idx:start_idx + cutout_size_clamped] = 0
            else:
                raise ValueError("Unsupported volume shape for cutout")

        return shifted_volume

    def __getitem__(self, idx):
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            image=self.fourth_augment(image)
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//4,self.cfg.size//4)).squeeze(0)
            return image, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]
            
            image=self.fourth_augment(image)
            # image = self.shuffle_d_axis(image)
            # image = self.z_circular_shift_np(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//4,self.cfg.size//4)).squeeze(0)
            return image, label

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
    def __init__(self,pred_shape,size=CFG.size,enc='',with_norm=False,total_steps=780):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.05)
        self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+ 0.5*self.loss_func2(x,y)
        
        self.backbone = generate_model(model_depth=101, n_input_channels=1,forward_features=True,n_classes=1039)
        state_dict=torch.load('./checkpoints/r3d101_KM_200ep.pth')["state_dict"]
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.backbone.load_state_dict(state_dict,strict=False)
        # self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=True)
        # # self.backbone.load_state_dict(torch.load('./pretraining_i3d_epoch=3.pt'),strict=False)
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)
            
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # # Segformer expects 2D input with shape (B, C, H, W)
        # self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
        #     "nvidia/mit-b0",
        #     num_labels=1,
        #     ignore_mismatched_sizes=True,
        #     num_channels=3
        # )        

        # init_weights(self.decoder)
        self.dropout = nn.Dropout(0.2)
        # self.normalization = nn.batchnono
        # self.normalization = nn.InstanceNorm3d(1)
        # self.normalization = nn.BatchNorm3d(1)

    
    # BACKBONE FORWARD
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        # if self.hparams.with_norm:
        #   x=self.normalization(x)
        feat_maps = self.backbone(x) 
    
        feat_maps_pooled = [torch.max(f, dim=2)[0]  for f in feat_maps]
        feat_maps_pooled = [self.dropout(f).contiguous() for f in feat_maps_pooled]

        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask
    
    #  # BACKBONE FORWARD
    # def forward(self, x):
    #     if x.ndim==4:
    #         x=x[:,None]
    #     if self.hparams.with_norm:
    #         x=self.normalization(x)
    #     feat_maps = self.backbone(x)        
    #     feat_maps_pooled = [torch.max(f, dim=2)[0].contiguous() for f in feat_maps]
    #     # pred_mask = self.decoder(feat_maps_pooled)
    #     pred_mask = self.decoder(feat_maps_pooled).contiguous()

    #     x = self.encoder_2d(pred_mask)
    #     return x.logits
    
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
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        # print(outputs.shape)
        # print(y.shape)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            # pred_patch = F.interpolate(y_preds[i].unsqueeze(0).float(),size=(self.hparams.size, self.hparams.size),mode='bilinear').squeeze(0).squeeze(0).numpy()
            # self.mask_pred[y1:y2, x1:x2] += pred_patch
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)        
        return {"loss": loss1}

    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
       
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
    
    # def configure_optimizers(self):

    #     optimizer = AdamW(self.parameters(), lr=CFG.lr)
    #     scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4,pct_start=0.15, steps_per_epoch=self.hparams.total_steps, epochs=50,final_div_factor=1e2)
    #     # scheduler = get_scheduler(CFG, optimizer)
    #     return [optimizer],[scheduler]
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=CFG.lr)
        scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4,pct_start=0.15, steps_per_epoch=143*4, epochs=25,final_div_factor=1e2)
        # scheduler = get_scheduler(CFG, optimizer)
        return [optimizer]#,[scheduler]
    # def configure_optimizers(self):
       
    #     param_groups = [
    #         {'params': self.backbone.parameters(), 'lr': 2e-5, 'weight_decay': 1e-2},
    #         {'params': self.decoder.parameters(), 'lr': 1e-4, 'weight_decay': 1e-2},
    #     ]
        
    #         # Conditionally add encoder_2d parameters if it exists
    #     if hasattr(self, 'encoder_2d') and self.encoder_2d is not None:
    #         param_groups.append({
    #             'params': self.encoder_2d.parameters(),
    #             'lr': 1e-5,  # smaller LR for pretrained encoder
    #             'weight_decay': 0
    #         })
            
            
    #     optimizer = AdamW(param_groups)
    #     # # Scheduler for OneCycleLR
    #     # steps_per_epoch = 143 * 4  # adjust as per your dataloader
    #     # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     #     optimizer,
    #     #     max_lr=[group['lr'] for group in param_groups],  # pass per-group max_lr
    #     #     pct_start=0.15,
    #     #     steps_per_epoch=steps_per_epoch,
    #     #     epochs=25,
    #     #     final_div_factor=1e2
    #     # )

    #     return [optimizer]#, [{'scheduler': scheduler, 'interval': 'step'}]

    

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

torch.cuda.empty_cache()
print(CFG.valid_id)
fragment_id = CFG.valid_id

valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_mask.png", 0)
# valid_mask_gt = cv2.resize(valid_mask_gt, (valid_mask_gt.shape[1]//2, valid_mask_gt.shape[0]//2), interpolation=cv2.INTER_AREA)
# pad0 = (CFG.size - valid_mask_gt.shape[0] % CFG.size)
# pad1 = (CFG.size - valid_mask_gt.shape[1] % CFG.size)
# valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
valid_mask_gt = cv2.resize(
    valid_mask_gt,
    (int(valid_mask_gt.shape[1] / 2), int(valid_mask_gt.shape[0] / 2)),
    interpolation=cv2.INTER_AREA
)
pred_shape=valid_mask_gt.shape
torch.set_float32_matmul_precision('medium')

fragments=[CFG.valid_id]
print('Fragments: ', fragments) 

for lr in [['20231210132040','frag1']]:

        # CFG.start_idx=lr
        # CFG.frags = lr  

        enc_i,enc,fold=0,'i3d',0
        for fid in fragments:
            CFG.valid_id=fid
            fragment_id = CFG.valid_id

            pred_shape=valid_mask_gt.shape

            train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()
            print(len(train_images))

            valid_xyxys = np.stack(valid_xyxys)
            
            train_dataset = CustomDataset(
                train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG)
            )
            
            valid_dataset = CustomDataset(
                valid_images, CFG,xyxys=valid_xyxys, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))
            
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


            run_slug=f'RESNET_{CFG.frags}_valid={CFG.valid_id}_size={CFG.size}_lr={CFG.lr}_in_chans={CFG.in_chans}'

            wandb_logger = WandbLogger(project="vesivus",name=run_slug+f'{enc}')
            norm=fold==1
            model=RegressionPLModel(enc='resnet101',pred_shape=pred_shape,size=CFG.size,total_steps=len(train_loader))
            
            # # # # DION
            # checkpoint = torch.load("outputs/vesuvius/pretraining_all/vesuvius-models/f15_div2_l15-35_20231210132040_0_fr_i3depoch=11.ckpt", map_location="cpu", weights_only=False)
            # model.load_state_dict(checkpoint["state_dict"], strict=True)

            # print('FOLD : ',fold)
            wandb_logger.watch(model, log="all", log_freq=50)
            # multiplicative = lambda epoch: 0.9
            trainer = pl.Trainer(
            max_epochs=50,
            accelerator="gpu",
            devices=-1,
            check_val_every_n_epoch=4,
            logger=wandb_logger,
            default_root_dir="./models",
            accumulate_grad_batches=1,
            precision='16-mixed',
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            strategy='ddp_find_unused_parameters_true',
            callbacks=[ModelCheckpoint(filename=f'f15_div2_l15-35_{fid}_{fold}_fr_{enc}'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),
                        ],
            )
            # trainer.validate(model=model, dataloaders=valid_loader, verbose=True)
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

            wandb.finish()