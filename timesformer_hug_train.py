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


# import os
# import cv2
# import numpy as np
# import tifffile as tiff
# from tqdm import tqdm

# import os
# import cv2
# import numpy as np
# import tifffile as tiff
# from tqdm import tqdm
# from PIL import Image

# def split_left_mid_right_and_save(fragment_id, CFG):
#     start_idx = CFG.start_idx
#     end_idx = start_idx + CFG.in_chans
#     idxs = range(start_idx, end_idx)

#     # === Split each TIFF layer ===
#     for i in tqdm(idxs, desc=f"Splitting layers of {fragment_id}"):
#         tif_path = os.path.join(CFG.segment_path, fragment_id, "layers", f"{i:02}.tif")
#         if not os.path.exists(tif_path):
#             print(f"Missing: {tif_path}")
#             continue

#         image = tiff.imread(tif_path)
#         h, w = image.shape
#         w1, w2 = w // 3, 2 * w // 3

#         left = image[:, :w1]
#         mid = image[:, w1:w2]
#         right = image[:, w2:]

#         for part, img in zip(["left"], [left]):
#             out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), fragment_id, "layers")
#             os.makedirs(out_dir, exist_ok=True)
#             cv2.imwrite(os.path.join(out_dir, f"{i:02}.png"), img)

#     # === Split inklabels mask ===
#     mask_path = os.path.join(CFG.segment_path, fragment_id, f"{fragment_id}_inklabels.png")
#     if os.path.exists(mask_path):
#         mask = np.array(Image.open(mask_path).convert("L"))
#         h, w = mask.shape
#         w1, w2 = w // 3, 2 * w // 3
#         mask_parts = [mask[:, :w1], mask[:, w1:w2], mask[:, w2:]]

#         for part, m in zip(["left"], mask_parts):
#             out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), fragment_id)
#             os.makedirs(out_dir, exist_ok=True)
#             cv2.imwrite(os.path.join(out_dir, f"{fragment_id}_inklabels.png"), m)
#     else:
#         print(f"⚠️ Inklabels mask not found for {fragment_id}")

#     # === Split fragment mask ===
#     fragmask_path = os.path.join(CFG.segment_path, fragment_id, f"{fragment_id}_mask.png")
#     if os.path.exists(fragmask_path):
#         frag_mask = np.array(Image.open(fragmask_path).convert("L"))
#         h, w = frag_mask.shape
#         w1, w2 = w // 3, 2 * w // 3
#         frag_parts = [frag_mask[:, :w1], frag_mask[:, w1:w2], frag_mask[:, w2:]]

#         for part, fm in zip(["left"], frag_parts):
#             out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), fragment_id)
#             os.makedirs(out_dir, exist_ok=True)
#             cv2.imwrite(os.path.join(out_dir, f"{fragment_id}_mask.png"), fm)
#     else:
#         print(f"⚠️ Fragment mask not found for {fragment_id}")
        

# import os
# import cv2
# import numpy as np
# import tifffile as tiff
# from tqdm import tqdm
# from PIL import Image

# def split_left_mid_right_and_save(fragment_id, CFG):
#     start_idx = CFG.start_idx
#     end_idx = start_idx + CFG.in_chans
#     idxs = range(start_idx, end_idx)

#     # === Split each TIFF layer ===
#     for i in tqdm(idxs, desc=f"Splitting layers of {fragment_id}"):
#         tif_path = os.path.join(CFG.segment_path, fragment_id, "layers", f"{i:02}.tif")
#         if not os.path.exists(tif_path):
#             print(f"Missing: {tif_path}")
#             continue

#         image = tiff.imread(tif_path)  # Preserves uint16
#         h, w = image.shape
#         w1, w2 = w // 3, 2 * w // 3

#         left = image[:, :w1]
#         mid = image[:, w1:w2]
#         right = image[:, w2:]

#         for part, img in zip(["left", "mid", "right"], [left, mid, right]):
#             out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), "layers")
#             os.makedirs(out_dir, exist_ok=True)
#             out_path = os.path.join(out_dir, f"{i:02}.tif")
#             tiff.imwrite(out_path, img)  # Save as TIFF to preserve uint16

#     # === Split inklabels mask ===
#     mask_path = os.path.join(CFG.segment_path, fragment_id, f"{fragment_id}_inklabels.png")
#     if os.path.exists(mask_path):
#         mask = np.array(Image.open(mask_path).convert("L"))
#         h, w = mask.shape
#         w1, w2 = w // 3, 2 * w // 3
#         mask_parts = [mask[:, :w1], mask[:, w1:w2], mask[:, w2:]]

#         for part, m in zip(["left", "mid", "right"], mask_parts):
#             out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), fragment_id)
#             os.makedirs(out_dir, exist_ok=True)
#             tiff.imwrite(os.path.join(out_dir, f"{fragment_id}_inklabels.tif"), m.astype(np.uint8))  # PNG is uint8 anyway
#     else:
#         print(f"⚠️ Inklabels mask not found for {fragment_id}")

#     # === Split fragment mask ===
#     fragmask_path = os.path.join(CFG.segment_path, fragment_id, f"{fragment_id}_mask.png")
#     if os.path.exists(fragmask_path):
#         frag_mask = np.array(Image.open(fragmask_path).convert("L"))
#         h, w = frag_mask.shape
#         w1, w2 = w // 3, 2 * w // 3
#         frag_parts = [frag_mask[:, :w1], frag_mask[:, w1:w2], frag_mask[:, w2:]]

#         for part, fm in zip(["left", "mid", "right"], frag_parts):
#             out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), fragment_id)
#             os.makedirs(out_dir, exist_ok=True)
#             tiff.imwrite(os.path.join(out_dir, f"{fragment_id}_mask.tif"), fm.astype(np.uint8))
#     else:
#         print(f"⚠️ Fragment mask not found for {fragment_id}")
#         # === Split inklabels mask ===

        
# import gc
# gc.collect()     
# class CFG:
#     segment_path = "train_scrolls/"
#     output_left_path = "train_scrolls/left"
#     output_mid_path = "train_scrolls/mid"
#     output_right_path = "train_scrolls/right"
    
#     start_idx = 14
#     in_chans = 47
    
#     size = 224
#     tile_size = 224
#     stride = tile_size // 8 
    
#     train_batch_size =  15 # 32
#     valid_batch_size = 15
    
#     lr = 1e-4
#     num_workers = 8
#     # ============== model cfg =============
#     scheduler = 'linear' # 'cosine', 'linear'
#     epochs = 30
#     warmup_factor = 10
#     lr = 1e-4
    
    
# cfg = CFG()
# torch.cuda.empty_cache()

# split_left_mid_right_and_save("bigone",cfg)


class CFG:
    # ============== comp exp name =============
    current_dir = './'
    segment_path = './train_scrolls/'
    
    start_idx = 30
    in_chans = 16
    
    size = 224
    tile_size = 224
    stride = tile_size // 8 
    
    train_batch_size =  15 # 32
    valid_batch_size = 15
    check_val = 4
    lr = 1e-4
    
    # Size of fragments
    frags_ratio1 = ['frag','202','left']
    frags_ratio2 = ['nothing']
    ratio1 = 2
    ratio2 = 1
    
    # ============== fold =============
    segments = ['frag5','20231210132040'] 
    valid_id = '20231210132040'
    
    
    num_workers = 8
    # ============== model cfg =============
    scheduler = 'linear' # 'cosine', 'linear'
    epochs = 30
    warmup_factor = 10
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
        ToTensorV2(transpose_mask=True),
    ]
    # train_aug_list = [
    #     # A.RandomResizedCrop(
    #     #     size, size, scale=(0.7, 1.0)),
    #     A.Resize(size, size),
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.RandomRotate90(p=0.6),

    #     A.RandomBrightnessContrast(p=0.75),
    #     A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.1,p=0.75),
    #     A.OneOf([
    #             A.GaussNoise(var_limit=[10, 50]),
    #             A.GaussianBlur(),
    #             A.MotionBlur(),
    #             ], p=0.4),
    #     # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    #     A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
    #                     mask_fill_value=0, p=0.5),
    #     # A.Cutout(max_h_size=int(size * 0.6),
    #     #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
    #     A.Normalize(
    #         mean= [0] * in_chans,
    #         std= [1] * in_chans
    #     ),
    #     ToTensorV2(transpose_mask=True),
    # ]

    valid_aug_list = [
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
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

pred_shape=valid_mask_gt.shape
if (any(sub in fragment_id for sub in CFG.frags_ratio1)):
    pred_shape = tuple(s // CFG.ratio1 for s in valid_mask_gt.shape)
elif (any(sub in fragment_id for sub in CFG.frags_ratio2)):
    pred_shape = tuple(s // CFG.ratio2 for s in valid_mask_gt.shape)
else:
    pass

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
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                            )
valid_loader = DataLoader(valid_dataset,
                            batch_size=CFG.valid_batch_size,
                            shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

print(f"Train loader length: {len(train_loader)}")
print(f"Valid loader length: {len(valid_loader)}")

wandb_logger = WandbLogger(project="vesivus",name=run_slug)  

model = timesformer_hug.TimesfomerModel(pred_shape=pred_shape, size=CFG.size, lr=CFG.lr, scheduler=CFG.scheduler, wandb_logger=wandb_logger)
wandb_logger.watch(model, log="all", log_freq=100)

model = timesformer_hug.load_weights(model,"outputs/vesuvius/pretraining_all/vesuvius-models/TF_['frag5', '20231210132040']_valid=20231210132040_size=224_lr=0.0001_in_chans=16_epoch=27.ckpt")
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
    strategy='ddp',
    callbacks=[ModelCheckpoint(filename=f'{run_slug}_'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),
    ]

)
trainer.validate(model=model, dataloaders=valid_loader, verbose=True)
# trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
wandb.finish()