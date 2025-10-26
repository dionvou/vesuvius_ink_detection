import glob
import os
import cv2
from PIL import Image, ImageOps
import tifffile as tiff
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import wandb

def read_image_mask(fragment_id, CFG=None):
    """ 
    Reads a fragment image and its corresponding masks.    
    """
    images = []
    start_idx = CFG.start_idx 
    end_idx = start_idx + CFG.in_chans
    
    idxs = range(start_idx, end_idx)
    image_shape = 0
    
    if fragment_id==CFG.valid_id:
        print('valid')
        start_idx = int(start_idx+(CFG.in_chans-CFG.valid_chans)//2)
        end_idx = start_idx + CFG.valid_chans
        idxs = range(start_idx, end_idx)
        print(start_idx)
        
    
    try:

        for i in tqdm(idxs):
            tif_path = os.path.join(CFG.segment_path, fragment_id, "layers", f"{i:02}.tif")
            jpg_path = os.path.join(CFG.segment_path, fragment_id, "layers", f"{i:02}.jpg")
            png_path = os.path.join(CFG.segment_path, fragment_id, "layers", f"{i:02}.png") 
            
            if os.path.exists(tif_path):
                image = cv2.imread(tif_path, 0)
            elif os.path.exists(jpg_path):
                image = cv2.imread(jpg_path, 0)
            else:
                image = cv2.imread(png_path, 0)
            
            # Resize the image to match the expected size
            if (any(sub in fragment_id for sub in CFG.frags_ratio1)):
                scale = 1 / CFG.ratio1
                new_w = int(image.shape[1] * scale) 
                new_h = int(image.shape[0] * scale) 
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            elif (any(sub in fragment_id for sub in CFG.frags_ratio2)):
                scale = 1 / CFG.ratio2
                new_w = int(image.shape[1] * scale)
                new_h = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                scale = 1 / 1
                new_w = int(image.shape[1] * scale)
                new_h = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            image_shape = (image.shape[1], image.shape[0])
                
            pad0 = (CFG.size - image.shape[0] % CFG.size) % CFG.size
            pad1 = (CFG.size - image.shape[1] % CFG.size) % CFG.size
            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
           
            image=np.clip(image,0,200)
            images.append(image)

        images = np.stack(images, axis=2)
        print(f" Shape of {fragment_id} segment: {images.shape}")
        # if fragment_id == '20231024093300':
        #     images=images[:,:,::-1]
        
        mask = np.zeros(images.shape[:2], dtype=np.uint8)  # shape = (H, W)
        fragment_mask = np.zeros(images.shape[:2], dtype=np.uint8)  # shape = (H, W)
        # # READ INK LABELS
        inklabel_files = glob.glob(f"{CFG.segment_path}/{fragment_id}/*inklabels.*")
        
        if len(inklabel_files) > 0:
            mask = cv2.imread(inklabel_files[0], 0)
        else:
            print(f"Creating empty mask for {fragment_id}")
            mask = np.zeros(images[0].shape)
            
        mask =  cv2.resize(mask , image_shape, interpolation=cv2.INTER_AREA)
    
        path = f"{CFG.segment_path}{fragment_id}/{fragment_id}_mask.png"
        fragment_mask = cv2.imread(path,0)
            

        fragment_mask =  cv2.resize(fragment_mask , image_shape, interpolation=cv2.INTER_AREA)
        pad0 = (CFG.size - fragment_mask.shape[0] % CFG.size) % CFG.size
        pad1 = (CFG.size - fragment_mask.shape[1] % CFG.size) % CFG.size
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

        mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
        
        
        mask = mask.astype('float32')
        mask/=255
        fragment_mask = fragment_mask.astype('float32')/255
        
            
    except:
            print(fragment_id,"no used")
    return images, mask, fragment_mask

def get_train_valid_dataset(CFG=None):
    train_images = []
    train_masks = []
    
    valid_images = []
    valid_masks = []
    valid_xyxys = []
    
    segments = CFG.segments  # List of segment IDs (subdirectory names or file prefixes)
    path = CFG.segment_path  # Path to the directory containing all segments
    
    for fragment_id in segments:
        fragment_path = os.path.join(path, fragment_id)

        if not os.path.isdir(fragment_path):
            continue  # Skip

        print('reading', fragment_id)

        image, mask, fragment_mask = read_image_mask(fragment_id, CFG=CFG)

        x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, CFG.stride))
        y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, CFG.stride))
        windows_dict = {}

        for a in y1_list:
            for b in x1_list:
                if np.mean(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size]) >= 0.3:
                    
                    # if fragment_id =='s4':
                    #     # if not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size]<0.95):
                    #         for yi in range(0, CFG.tile_size, CFG.size):
                    #             for xi in range(0, CFG.tile_size, CFG.size):
                    #                 y1 = a + yi
                    #                 x1 = b + xi
                    #                 y2 = y1 + CFG.size
                    #                 x2 = x1 + CFG.size
                    #                 tile_mask = mask[y1:y2, x1:x2, None].copy()  # copy the patch
                    #                 all_gray = np.all((tile_mask < 0.95))
                    #                 if not all_gray:
                    #                     # Set all pixels where mask==0 to IGNORE INDEX, keep 1s as is
                    #                     tile_mask[(tile_mask <0.9) & (tile_mask>0.1)] = 127 # mask
                    #                     train_images.append(image[y1:y2, x1:x2])
                    #                     train_masks.append(tile_mask)
                    #                     windows_dict[(y1, y2, x1, x2)] = '1'
                    #                     assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                    # if fragment_id == 's4121124252':
                    #     print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
                    #     for yi in range(0, CFG.tile_size, CFG.size):
                    #         for xi in range(0, CFG.tile_size, CFG.size):
                    #             y1 = a + yi
                    #             x1 = b + xi
                    #             y2 = y1 + CFG.size
                    #             x2 = x1 + CFG.size

                    #             tile_mask = mask[y1:y2, x1:x2, None].copy()  # copy the patch

                    #             # ✅ Check if tile is all 1s or all 0s
                    #             all_ones = np.all(tile_mask == 1)
                    #             all_zeros = np.all(tile_mask == 0)

                    #             if all_ones or all_zeros:

                    #                 train_images.append(image[y1:y2, x1:x2])
                    #                 train_masks.append(tile_mask)
                    #                 windows_dict[(y1, y2, x1, x2)] = '1'

                    #                 assert image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans)
                                    
                    if fragment_id == CFG.valid_id or not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size] < 0.05):
                        for yi in range(0, CFG.tile_size, CFG.size):
                            for xi in range(0, CFG.tile_size, CFG.size):
                                y1 = a + yi
                                x1 = b + xi
                                y2 = y1 + CFG.size
                                x2 = x1 + CFG.size
                                if fragment_id != CFG.valid_id :
                                    if (y1, y2, x1, x2) not in windows_dict:
                                        train_images.append(image[y1:y2, x1:x2])
                                        train_masks.append(mask[y1:y2, x1:x2, None])
                                        assert image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans)
                                        windows_dict[(y1, y2, x1, x2)] = '1'
                                else:
                                    if (y1, y2, x1, x2) not in windows_dict:
                                        valid_images.append(image[y1:y2, x1:x2])
                                        valid_masks.append(mask[y1:y2, x1:x2, None])
                                        valid_xyxys.append([x1, y1, x2, y2])
                                        assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.valid_chans)
                                        windows_dict[(y1, y2, x1, x2)] = '1'

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

def mae_loss(pred, target, mask):
    """
    pred: (B, C, H, W) — model's reconstructed output
    target: (B, C, H, W) — ground truth input image
    mask: (B, 1, H, W) — binary mask, 1 where masked (loss applied), 0 where visible
    """
    loss = (pred - target) ** 2
    loss = loss * mask  # apply mask to compute loss only on masked areas
    return loss.sum() / mask.sum().clamp(min=1.0)  # prevent divide-by-zero
    
    
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
    for dir in [cfg.model_dir]:
        os.makedirs(dir, exist_ok=True)
        
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    if mode == 'train':
        make_dirs(cfg)