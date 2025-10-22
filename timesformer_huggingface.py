import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoImageProcessor, TimesformerModel
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
class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'
    comp_dir_path = './'
    comp_folder_name = './'
    comp_dataset_path = f'./'
    exp_name = 'pretraining_all'
    # ============== model cfg =============
    in_chans = 16
    # ============== training cfg =============
    size = 224
    tile_size = 224
    stride = tile_size // 8
    train_batch_size = 7
    valid_batch_size = train_batch_size

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 30 # 30
    warmup_factor = 10
    lr = 1e-4
    # ============== fold =============
    valid_id = None
    # ============== fixed =============

    min_lr = 1e-7
    weight_decay = 1e-6
    max_grad_norm = 100

    num_workers = 4

    seed = 0
    frags = []

    outputs_path = f'./outputs/{comp_name}/{exp_name}/'
    model_dir = outputs_path + \
        f'{comp_name}-models/'
        
    # ============== augmentation =============
    train_aug_list = [
        # A.Resize(size, size),
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
        # A.Normalize(mean=[0.449]* in_chans, 
        #             std=[0.226]*in_chans), 

        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        # A.Resize(size, size),
        # A.Normalize(
        #     mean= [0] * in_chans,
        #     std= [1] * in_chans
        # ),
        # A.Normalize(mean=[0.449]* in_chans, 
        #             std=[0.226]*in_chans), 
        ToTensorV2(transpose_mask=True),
        
    ]
    pil_transform = T.Compose([
        T.ToPILImage(),                    # convert (C, H, W) to PIL
        T.Grayscale(num_output_channels=3),  # convert to 3 channels
    ])

    rotate = A.Compose([A.Rotate(5,p=1)])
    
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
        
cfg_init(CFG)


def read_image_mask(fragment_id,start_idx=22,end_idx=22+CFG.in_chans, CFG=CFG): #17,43
    fragment_id_ = fragment_id.split("_")[0]
    images = []
    idxs = range(start_idx, end_idx)
    # if fragment_id in ['vas42']:
    #     idxs = range(start_idx, end_idx-1,2)
    # if fragment_id in ['vals3','s30']:
    #     CFG.tile_size = 112
    #     CFG.size = 112
    # else:
    #     CFG.tile_size = 224
    #     CFG.size = 224

    print(list(idxs))

    for i in idxs:
        if os.path.exists(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif"):
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif", 0)
        else:
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.jpg", 0)
        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0) 
        
        if "frag" in fragment_id or fragment_id in ["20231210132040", "vals4","rect11","remaining1"]:
            image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
       
        image=np.clip(image,0,200)
        images.append(image)
        
    images = np.stack(images, axis=2)

    
    inklabel_files = glob.glob(f"train_scrolls/{fragment_id}/*inklabels.*")
    if len(inklabel_files) > 0:
        mask = cv2.imread(inklabel_files[0], 0)
    else:
        print(f"Creating empty mask for {fragment_id}")
        mask = np.zeros(images[0].shape)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    fragment_mask=cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_mask.png", 0)
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    mask = mask.astype('float32')
    mask/=255
    
    if "frag" in fragment_id or fragment_id in ["20231210132040", "vals4","rect11","remaining1"]:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//2,fragment_mask.shape[0]//2), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask , (mask.shape[1]//2,mask.shape[0]//2), interpolation = cv2.INTER_AREA)
    
    return images, mask, fragment_mask

def worker_function(fragment_id, CFG):
    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []

    if not os.path.exists(f"train_scrolls/{fragment_id}"):
        fragment_id = fragment_id + "_superseded"
    print('reading ',fragment_id)
    try:
        image, mask, fragment_mask = read_image_mask(fragment_id, CFG=CFG)
    except:
        print("aborted reading fragment", fragment_id)
        return None
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    windows_dict={}
    for a in y1_list:
        for b in x1_list:

            if (not np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size]==0)):# and (fragment_id!='20231210132040'):
                if (fragment_id==CFG.valid_id) or (not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size]<0.05)):
                    for yi in range(0,CFG.tile_size,CFG.size):
                        for xi in range(0,CFG.tile_size,CFG.size):
                            y1=a+yi
                            x1=b+xi
                            y2=y1+CFG.size
                            x2=x1+CFG.size
                            
                            if fragment_id!=CFG.valid_id:
                                train_images.append(image[y1:y2, x1:x2])
                                train_masks.append(mask[y1:y2, x1:x2, None])
                                assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                            if fragment_id==CFG.valid_id:
                                if (y1,y2,x1,x2) not in windows_dict:
                                    valid_images.append(image[y1:y2, x1:x2])
                                    valid_masks.append(mask[y1:y2, x1:x2, None])
                                    valid_xyxys.append([x1, y1, x2, y2])
                                    assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                                    windows_dict[(y1,y2,x1,x2)]='1'

    print("finished reading fragment", fragment_id)

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def get_train_valid_dataset(fragment_ids=['20231210132040','frag5','frag1']):
    threads = []
    results = [None] * len(fragment_ids)
    

    # Function to run in each thread
    def thread_target(idx, fragment_id):
        results[idx] = worker_function(fragment_id, CFG)

    # Create and start threads
    for idx, fragment_id in enumerate(fragment_ids):
        CFG.frags.append(fragment_id)
        thread = threading.Thread(target=thread_target, args=(idx, fragment_id))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []
    print("Aggregating results")
    for r in results:
        if r is None:
            continue
        train_images += r[0]
        train_masks += r[1]
        valid_images += r[2]
        valid_masks += r[3]
        valid_xyxys += r[4]

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

class CustomDataset(Dataset):
    def __init__(self, images ,cfg,xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        
        self.transform = transform
        self.xyxys=xyxys
        self.rotate=CFG.rotate
        
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600", use_fast=True)
        # self.processor.do_normalize = False
        self.pil_transform  = CFG.pil_transform
    def __len__(self):
        return len(self.images)
    def fourth_augment(self,image):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(14, 16)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        # if random.random() > 0.4:
        #     image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image

    def __getitem__(self, idx):
        if self.xyxys is not None: #VALID
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            
            image = image.permute(1,0,2,3)
            frames = [self.pil_transform(frame.squeeze(0)) for frame in image] 

            # now run the Timesformer processor ONCE:
            encoding = self.processor(
                [frame for frame in frames],   # list of PIL
                return_tensors='pt'
                )
            # encoding["pixel_values"] is (1, T, C, H, W)
            pixel_values = encoding["pixel_values"].squeeze(0)
            return pixel_values, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]
            
            # #3d rotate
            # image=image.transpose(2,1,0)#(c,w,h)
            # image=self.rotate(image=image)['image']
            # image=image.transpose(0,2,1)#(c,h,w)
            # image=self.rotate(image=image)['image']
            # image=image.transpose(0,2,1)#(c,w,h)
            # image=image.transpose(2,1,0)#(h,w,c)

            image=self.fourth_augment(image)
            # print(image.shape)
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            image = image.permute(1,0,2,3)
            frames = [self.pil_transform(frame.squeeze(0)) for frame in image] 

            # now run the Timesformer processor ONCE:
            encoding = self.processor(
                [frame for frame in frames],   # list of PIL
                return_tensors='pt'
                )
            pixel_values = encoding["pixel_values"].squeeze(0)
            
            return pixel_values, label

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_layers=1, num_heads=8, ff_dim=2048):
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.decoder(x)
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)  # even indices
#         pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
#         pe = pe.unsqueeze(0)  # (1, max_len, d_model)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape (B, T, C)
#         Returns:
#             x with positional encoding added
#         """
#         return x + self.pe[:, :x.size(1)]

# class CLSBasedDecoder(nn.Module):
#     def __init__(self, cls_dim=768, embed_dim=768, spatial_size=14, num_layers=3, num_heads=8):
#         """
#         spatial_size: H=W of output grid tokens before upsampling
#         """
#         super().__init__()
#         self.spatial_size = spatial_size
#         self.num_tokens = spatial_size * spatial_size

#         # Project CLS to memory for decoder
#         self.cls_proj = nn.Linear(cls_dim, embed_dim)

#         # Learnable queries initialized as parameters for spatial tokens
#         self.query_tokens = nn.Parameter(torch.randn(self.num_tokens, embed_dim))

#         # Transformer decoder layers
#         decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

#         # Positional encoding for queries
#         self.pos_enc = PositionalEncoding(embed_dim)

#         # Upsample head: ConvTranspose2d to upsample spatial grid to segmentation map
#         self.upsample = nn.Sequential(
#             nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(embed_dim // 4, 1, kernel_size=1)  # output 1 channel segmentation map
#         )

#     def forward(self, cls_token):
#         """
#         cls_token: (B, cls_dim)
#         """
#         B = cls_token.size(0)

#         # Project CLS token to decoder memory (B, 1, embed_dim)
#         memory = self.cls_proj(cls_token).unsqueeze(1)  # (B, 1, embed_dim)

#         # Expand query tokens to batch size (B, num_tokens, embed_dim)
#         queries = self.query_tokens.unsqueeze(0).repeat(B, 1, 1)

#         # Add positional encoding to queries
#         queries = self.pos_enc(queries)

#         # Decode: queries attend to CLS memory
#         decoded = self.decoder(tgt=queries, memory=memory)  # (B, num_tokens, embed_dim)

#         # Reshape decoded tokens to spatial grid
#         H = W = self.spatial_size
#         decoded = decoded.permute(0, 2, 1).contiguous().view(B, -1, H, W)  # (B, embed_dim, H, W)

#         # Upsample to full size segmentation map
#         seg_map = self.upsample(decoded)  # (B, 1, H_out, W_out)

#         return seg_map


class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=CFG.size,with_norm=False, lr=CFG.lr):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.15)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y: 0.6* self.loss_func1(x,y)+0.4*self.loss_func2(x,y)
    
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-hr-finetuned-k600")
        # self.decoder_for_cls = CLSBasedDecoder(cls_dim=768, embed_dim=768, spatial_size=14, num_layers=3, num_heads=8)
        # self.decoder = TransformerDecoder(embed_dim=768, num_layers=1, num_heads=8)

        # self.upsample_head = nn.Sequential(
        #     nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 1, kernel_size=1)  # Binary mask, use more channels for multi-class
        # )


        self.classifier = nn.Sequential(
            nn.Linear(768, (CFG.size//16)**2),  
        )
        
    def forward(self, x):

        outputs = self.backbone(x, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state  # tuple of all hidden layers
        cls = last_hidden_state[:,0,:]
        preds = self.classifier(cls)
        preds = preds.view(-1,1,CFG.tile_size//16,CFG.tile_size//16)
        return preds
    # def forward(self, x):
    #     outputs = self.backbone(x, output_hidden_states=True)
    #     last_hidden_state = outputs.last_hidden_state
    #     cls_token = last_hidden_state[:, 0, :]  # (B, 768)

    #     preds = self.decoder_for_cls(cls_token)  # spatial segmentation map
    #     return preds

    # def forward(self, x):
    #     """
    #     x: (B, C, T, H, W)
    #     """

    #     B, C, T, H, W = x.shape

    #     # Pass through TimeSformer
    #     outputs = self.backbone(x, output_hidden_states=True)
    #     tokens = outputs.last_hidden_state  # (B, N+1, 768)

    #     # Remove CLS token if present
    #     tokens = tokens[:, 1:, :]  # (B, N, 768)
    #     # tokens= tokens.permute(0, 2, 1).contiguous()  # (B, 768, N)
    #     # Pass through transformer decoder
    #     decoded = self.decoder(tokens)  # (B, N, 768)

    #     # Reshape tokens to (B, 768, H_patch, W_patch)
    #     N = decoded.size(1)
    #     H_patch = W_patch = int(N ** 0.5)
    #     decoded = decoded.permute(0, 2, 1).contiguous().view(B, 768, H_patch, W_patch)

    #     # Upsample to 2D map
    #     seg_map = self.upsample_head(decoded)  # (B, 1, H_out, W_out)
    #     # print("seg_map",seg_map.shape)
    #     return seg_map


    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        # loss1 = self.masked_loss(outputs, y)  # <--- USE MASKED LOSS
        loss = self.loss_func(outputs, y)
        if torch.isnan(loss):
            print("Loss nan encountered")
        self.log("train/total_loss", loss.item(),on_step=True, on_epoch=True, prog_bar=True)
        
            # Log learning rate (get it from optimizer)
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        # print(outputs.shape)
        # print(y.shape)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=16,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    # def configure_optimizers(self):

    #     optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
    
    #     scheduler = get_scheduler(CFG, optimizer)
    #     return [optimizer],[scheduler]
    def configure_optimizers(self):
        # Separate the head parameters
        head_params = list(self.classifier.parameters())
        # head_params = list(self.upsample_head.parameters()) + list(self.decoder.parameters())
        other_params = [p for n, p in self.backbone.named_parameters() if "classifier" not in n]
        # other_params = [p for n, p in self.backbone.named_parameters() if "upsample_head" or "decoder" or "classifier" not in n]

        # Define parameter groups
        param_groups = [
            {'params': other_params, 'lr': self.hparams.lr},
            {'params': head_params, 'lr': self.hparams.lr },  # 10x LR for the head
        ]

        optimizer = AdamW(param_groups)
        

        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer], [scheduler]

        
    def on_validation_epoch_end(self):
        mask_pred_tensor = torch.tensor(self.mask_pred, dtype=torch.float32, device=self.device)
        mask_count_tensor = torch.tensor(self.mask_count, dtype=torch.float32, device=self.device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(mask_pred_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mask_count_tensor, op=dist.ReduceOp.SUM)

        if self.trainer.is_global_zero:
            mask_pred_np = mask_pred_tensor.cpu().numpy()
            mask_count_np = mask_count_tensor.cpu().numpy()
            final_mask = np.divide(
                mask_pred_np,
                mask_count_np,
                out=np.zeros_like(mask_pred_np),
                where=mask_count_np != 0
            )
            wandb_logger.log_image(key="masks", images=[np.clip(final_mask, 0, 1)], caption=["probs"])

        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)





class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 20, eta_min=1e-6)
    # scheduler_linear = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1.0,   # start at the current learning rate
    #     end_factor=0.05,    # end at 1% of the current learning rate
    #     total_iters=30     # over 20 epochs or iterations
    # )
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)
    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

torch.set_float32_matmul_precision('medium')

fragments=['20231210132040'] # valid fragment
for fid in fragments:
    CFG.valid_id=fid
    fragment_id = CFG.valid_id
    run_slug=f'{CFG.valid_id}_{CFG.size}x{CFG.size}_lr={CFG.lr}_in_chans={CFG.in_chans}'

    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)

    # pred_shape=valid_mask_gt.shape
    pred_shape = tuple(s // 2 for s in valid_mask_gt.shape)
    
    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()
    print('train_images',train_images[0].shape)
    
    valid_xyxys = np.stack(valid_xyxys)
    train_dataset = CustomDataset(
        train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
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
    
    print('train',len(train_loader))
    print('valid',len(valid_loader))
    model=RegressionPLModel(pred_shape=pred_shape,size=CFG.size)


    wandb_logger = WandbLogger(project="vesivus",name=run_slug+'_'+str(pred_shape[0])+'_timesformer')    
    # checkpoint = torch.load('outputs/vesuvius/pretraining_all/vesuvius-models/timesformer_wild16_vals42_frepoch=13.ckpt', map_location="cpu", weights_only=False)
    # model.load_state_dict(checkpoint["state_dict"], strict=True)
    wandb_logger.watch(model, log="all", log_freq=100)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        check_val_every_n_epoch=4,
        devices=-1,
        logger=wandb_logger,
        default_root_dir="./models",
        accumulate_grad_batches=2,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='ddp',
        # callbacks=[ModelCheckpoint(filename=f'timesformer_wild16_{fid}_fr'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),
        # ]

    )
    # from pytorch_lightning.tuner import Tuner

    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    # trainer.validate(model, valid_loader)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    # Plot
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig("lr_finder_plot.png")  # Save the plot as a PNG file
    # fig.show()  # Then show it if you want
    # model.hparams.lr = lr_finder.suggestion()
    wandb.finish()