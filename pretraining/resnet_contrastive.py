import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

import numpy as np
import pandas as pd

from pytorch_lightning.loggers import WandbLogger

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
import pickle
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
import sys
import pandas as pd
import gc
import sys
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import cv2

import scipy as sp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from functools import partial

import importlib

from torch.optim import Adam, SGD, AdamW
from ema_pytorch import EMA

import datetime
import math
import time
import numpy as np
import torch

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import segmentation_models_pytorch as smp


# Log in to your W&B account
import wandb
wandb.login()

class CFG:
    # ============== comp exp name =============
    current_dir = './'
    segment_path = './train_scrolls/'
    
    start_idx = 24
    in_chans = 16
    
    size = 224
    tile_size = 224
    stride = tile_size // 8 
    
    train_batch_size =  10 # 32
    valid_batch_size = 4
    
    lr = 1e-4
    num_workers = 8
    # ============== model cfg =============
    scheduler = 'linear' # 'cosine', 'linear'
    epochs = 30
    warmup_factor = 10
    
    # ============== fold =============
    segments = ['vals42','20231210132040'] 
    valid_id = 'vals42'
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
        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(rotate_limit=45,shift_limit=0.15,scale_limit=0.15,p=0.5),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.GridDistortion(num_steps=2, distort_limit=0.3, p=0.4),
        A.CoarseDropout(max_holes=4, max_width=int(size * 0.06), max_height=int(size * 0.06), 
                        mask_fill_value=0, p=0.5),
        A.Cutout(max_h_size=int(size * 0.15),
                 max_w_size=int(size * 0.15), num_holes=1, p=0.5),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
        
    ]
    global_aug_list=[
        A.Resize(size, size),
        A.ShiftScaleRotate(rotate_limit=0,shift_limit=0.25,scale_limit=0.25,p=0.75),
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
    
def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)
        
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    if mode == 'train':
        make_dirs(cfg)
        
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

