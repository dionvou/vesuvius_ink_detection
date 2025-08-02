import os
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm

import os
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from PIL import Image

import torch

def split_left_mid_right_and_save(fragment_id, CFG):
    start_idx = CFG.start_idx
    end_idx = start_idx + CFG.in_chans
    idxs = range(start_idx, end_idx)

    # === Split each TIFF layer ===
    for i in tqdm(idxs, desc=f"Splitting layers of {fragment_id}"):
        tif_path = os.path.join(CFG.segment_path, fragment_id, "layers", f"{i:02}.tif")
        if not os.path.exists(tif_path):
            print(f"Missing: {tif_path}")
            continue

        image = tiff.imread(tif_path)
        h, w = image.shape
        w1, w2 = w // 3, 2 * w // 3

        left = image[:, :w1]
        mid = image[:, w1:w2]
        right = image[:, w2:]

        for part, img in zip(["left"], [left]):
            out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), fragment_id, "layers")
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{i:02}.png"), img)

    # === Split inklabels mask ===
    mask_path = os.path.join(CFG.segment_path, fragment_id, f"{fragment_id}_inklabels.png")
    if os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path).convert("L"))
        h, w = mask.shape
        w1, w2 = w // 3, 2 * w // 3
        mask_parts = [mask[:, :w1], mask[:, w1:w2], mask[:, w2:]]

        for part, m in zip(["left"], mask_parts):
            out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), fragment_id)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{fragment_id}_inklabels.png"), m)
    else:
        print(f"⚠️ Inklabels mask not found for {fragment_id}")

    # === Split fragment mask ===
    fragmask_path = os.path.join(CFG.segment_path, fragment_id, f"{fragment_id}_mask.png")
    if os.path.exists(fragmask_path):
        frag_mask = np.array(Image.open(fragmask_path).convert("L"))
        h, w = frag_mask.shape
        w1, w2 = w // 3, 2 * w // 3
        frag_parts = [frag_mask[:, :w1], frag_mask[:, w1:w2], frag_mask[:, w2:]]

        for part, fm in zip(["left"], frag_parts):
            out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), fragment_id)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{fragment_id}_mask.png"), fm)
    else:
        print(f"⚠️ Fragment mask not found for {fragment_id}")
        

import os
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from PIL import Image

def split_left_mid_right_and_save(fragment_id, CFG):
    start_idx = CFG.start_idx
    end_idx = start_idx + CFG.in_chans
    idxs = range(start_idx, end_idx)

    # === Split each TIFF layer ===
    for i in tqdm(idxs, desc=f"Splitting layers of {fragment_id}"):
        tif_path = os.path.join(CFG.segment_path, fragment_id, "layers", f"{i:02}.tif")
        if not os.path.exists(tif_path):
            print(f"Missing: {tif_path}")
            continue

        image = tiff.imread(tif_path)  # Preserves uint16
        h, w = image.shape
        w1, w2 = w // 3, 2 * w // 3

        left = image[:, :w1]
        mid = image[:, w1:w2]
        right = image[:, w2:]

        for part, img in zip(["left", "mid", "right"], [left, mid, right]):
            out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), "layers")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{i:02}.tif")
            tiff.imwrite(out_path, img)  # Save as TIFF to preserve uint16

    # === Split inklabels mask ===
    mask_path = os.path.join(CFG.segment_path, fragment_id, f"{fragment_id}_inklabels.png")
    if os.path.exists(mask_path):
        mask = np.array(Image.open(mask_path).convert("L"))
        h, w = mask.shape
        w1, w2 = w // 3, 2 * w // 3
        mask_parts = [mask[:, :w1], mask[:, w1:w2], mask[:, w2:]]

        for part, m in zip(["left", "mid", "right"], mask_parts):
            out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), fragment_id)
            os.makedirs(out_dir, exist_ok=True)
            tiff.imwrite(os.path.join(out_dir, f"{fragment_id}_inklabels.tif"), m.astype(np.uint8))  # PNG is uint8 anyway
    else:
        print(f"⚠️ Inklabels mask not found for {fragment_id}")

    # === Split fragment mask ===
    fragmask_path = os.path.join(CFG.segment_path, fragment_id, f"{fragment_id}_mask.png")
    if os.path.exists(fragmask_path):
        frag_mask = np.array(Image.open(fragmask_path).convert("L"))
        h, w = frag_mask.shape
        w1, w2 = w // 3, 2 * w // 3
        frag_parts = [frag_mask[:, :w1], frag_mask[:, w1:w2], frag_mask[:, w2:]]

        for part, fm in zip(["left", "mid", "right"], frag_parts):
            out_dir = os.path.join(getattr(CFG, f"output_{part}_path"), fragment_id)
            os.makedirs(out_dir, exist_ok=True)
            tiff.imwrite(os.path.join(out_dir, f"{fragment_id}_mask.tif"), fm.astype(np.uint8))
    else:
        print(f"⚠️ Fragment mask not found for {fragment_id}")
        # === Split inklabels mask ===

        
import gc
gc.collect()     
class CFG:
    segment_path = "train_scrolls/"
    output_left_path = "train_scrolls/left"
    output_mid_path = "train_scrolls/mid"
    output_right_path = "train_scrolls/right"
    
    start_idx = 14
    in_chans = 47
    
    size = 224
    tile_size = 224
    stride = tile_size // 8 
    
    train_batch_size =  15 # 32
    valid_batch_size = 15
    
    lr = 1e-4
    num_workers = 8
    # ============== model cfg =============
    scheduler = 'linear' # 'cosine', 'linear'
    epochs = 30
    warmup_factor = 10
    lr = 1e-4
    
    
cfg = CFG()

split_left_mid_right_and_save("bigone",cfg)