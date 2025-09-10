import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor
import torchvision.transforms as T
import numpy as np
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import segmentation_models_pytorch as smp

from torch.optim import AdamW
from warmup_scheduler import GradualWarmupScheduler

import utils

from torchvision.models.video import swin_transformer
import albumentations as A


class TimesformerDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, transform=None, norm=False, aug=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.rotate = A.Compose([A.Rotate(8,p=1)])
        self.xyxys=xyxys
        self.aug = aug
        self.scale_factor = 16
        
        self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32),  # scales to [0.0, 1.0]
        ])
        
        # Conditionally add Normalize transformation
        if norm:
            self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
                
    def __len__(self):
        return len(self.images)
    
    # def fourth_augment(self, image, in_chans=15):
    #     """
    #     Temporal chunk copy-paste augmentation.
    #     Keeps all other frames intact (no zeroing out).
        
    #     image: np.ndarray of shape (H, W, D)
    #     in_chans: total number of frames in depth axis
    #     """
    #     # Start from original instead of zeros
    #     image_tmp = image.copy()

    #     # Pick how many frames to copy
    #     cropping_num = random.randint(in_chans - 6, in_chans)

    #     # Select crop start index
    #     start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
    #     crop_indices = np.arange(start_idx, start_idx + cropping_num)

    #     # Select paste start index
    #     start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

    #     # Paste cropped frames over the target location
    #     image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

    #     # (Optional) temporal cutout of a few random frames
    #     cutout_idx = random.randint(0, 2)
    #     if cutout_idx > 0:
    #         cutout_positions = random.sample(range(start_paste_idx, start_paste_idx + cropping_num), cutout_idx)
    #         for pos in cutout_positions:
    #             image_tmp[..., pos] = 0

    #     return image_tmp
    def fourth_augment(self, image):
        """
        Custom channel augmentation that returns exactly 24 channels.
        """
        # always select 8
        cropping_num =  12

        # pick crop indices
        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        # pick where to paste them
        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        # container
        image_tmp = np.zeros_like(image)

        # paste cropped channels
        image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

        # # optional random cutout
        # cutout_idx = random.randint(0, 2)
        # temporal_random_cutout_idx = np.arange(start_paste_idx, start_paste_idx + cutout_idx)
        # if random.random() > 0.4:
        #     image_tmp[..., temporal_random_cutout_idx] = 0

        # finally, keep only 24 channels
        image = image_tmp[..., start_paste_idx:start_paste_idx + cropping_num]

        return image
    
    def z_circular_shift_np(self, volume, max_shift=4, prob=0.5, cutout_size=2, cutout_prob=0.1):
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


    
    def shuffle_d_axis(self,image):
        # image shape: (H, W, D)
        d = image.shape[2]
        shuffled_indices = np.arange(d)
        np.random.shuffle(shuffled_indices)

        # Reorder along D axis
        image_shuffled = image[:, :, shuffled_indices]
        return image_shuffled

    def __getitem__(self, idx):
        if self.xyxys is not None: #VALID
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            # image=self.fourth_augment(image)#, self.cfg.in_chans)
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//self.scale_factor,self.cfg.size//self.scale_factor)).squeeze(0)
                
            image = image.permute(1,0,2,3)
            image = image.repeat(1, 3, 1, 1)
            image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            return image, label,xy
            
        else:
            image = self.images[idx]
            label = self.labels[idx]

            if self.aug == 'shift':
                image = self.z_circular_shift_np(image)
            elif self.aug == 'fourth':
                image=self.fourth_augment(image)#, self.cfg.in_chans)
            elif self.aug == 'shuffle':
                image = self.shuffle_d_axis(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//self.scale_factor,self.cfg.size//self.scale_factor)).squeeze(0)

            image = image.permute(1,0,2,3)
            image = image.repeat(1, 3, 1, 1)
            image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            return image, label
        
# class SwinModel(pl.LightningModule):
    # def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
    #     super(SwinModel, self).__init__()

    #     self.save_hyperparameters()
    #     self.mask_pred = np.zeros(self.hparams.pred_shape)
    #     self.mask_count = np.zeros(self.hparams.pred_shape)
    #     self.IGNORE_INDEX = 127

    #     self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.25,ignore_index=self.IGNORE_INDEX)
    #     self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

    #     self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
    #     self.backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    #     self.scale_factor = 8
    #     self.classifier = nn.Linear(768,4)
        
    
    # def forward(self, x):

    #     # x = x.permute(0,2,1,3,4)
    #     output = self.backbone(x)  # runs backbone, sets self.feature
    #     output = output.last_hidden_state  # (B, T', H', W', C)
    #     output = output.view(output.size(0),8, 196, -1)  # (B, T', C, H', W')
    #     output = output.mean(dim=1)  # temporal pooling (B, C, H', W')
    #     output = self.classifier(output)  # (B, 16, H', W')
    #     output = output.permute(0,2,1).view(output.size(0),4,14,14)  # (B, 4, 4, H', W')
    #     output = F.pixel_shuffle(output, upscale_factor=2)  # (B, 1, H, W)
        
#     #     return output

# # GOOOOOOOOOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
# class Patch3DTransformerSegmentation(nn.Module):
#     def __init__(self, num_classes=1, embed_dim=768, num_heads=4, depth=2, patch_output=4):
#         super().__init__()
#         self.patch_output = patch_output
#         self.num_classes = num_classes

#         # Swin3D backbone
#         backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1') #KINETICS400_IMAGENET22K_V1
#         self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # remove global pool + head

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim,
#             dropout=0.2,
#             activation="gelu",
#             batch_first=True  # (B, N, C)
#         )
#         self.embed_dim = embed_dim
#         self.pos_embedding = None  # will be initialized dynamically after seeing Hf, Wf
#         self.pos_embedding_x = None
#         self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

#         # Classifier: per patch -> small 3D patch
#         # self.classifier = nn.Linear(embed_dim, patch_output ** 2)
#         # self.proj = nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1)  # learns spatial filters
#         self.head = nn.Sequential(
#             nn.Conv2d(embed_dim,self.patch_output**2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.patch_output**2,self.patch_output**2, kernel_size=1),
#             # nn.ConvTranspose2d(256, num_classes, kernel_size=4, stride=4)
#             # nn.ReLU(),
#             # nn.Conv2d(self.patch_output**2,self.patch_output**2, kernel_size=1) 
#         )
#         # self.conv = nn.Conv2d(48, 1, kernel_size=1)


#     def forward(self, x):
#         B, C, T, H, W = x.shape
#         feats = self.backbone(x)  # (B, T', H', W',embed_dim)

#         # Adaptive temporal pooling
#         feats = feats.max(dim=1)[0] # (B, 1, Hf, Wf, embed_dim,)
        
#         B, Hf, Wf, D = feats.shape

#         # # Move embed_dim to dim 1 and flatten patches
#         patch_tokens = feats.permute(0, 3, 1, 2).contiguous()  # (B, 768, 8, 7, 7)
#         patch_tokens = patch_tokens.view(B, Hf*Wf, -1)        # (B, 768, 8*7*7=392)
#         # patch_tokens = patch_tokens.permute(1, 0, 2)               # (392, B, 768)
#         # if self.pos_embedding is None or self.pos_embedding.shape[1] != Hf*Wf:
#         #     self.pos_embedding = nn.Parameter(torch.zeros(1, Hf*Wf, self.embed_dim, device=x.device))
#         #     nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            
#         # Add positional embeddings
#         # patch_tokens = patch_tokens #+ self.pos_embedding  # (B, N, C)
#         # patch_tokens = self.input_proj(patch_tokens)  # (B, N, 16)
#         # Transformer
#         transformed_tokens = self.decoder(patch_tokens) 
#         transformed_tokens = transformed_tokens.permute(0, 2, 1).view(B, self.embed_dim, Hf, Wf)  # (B, C, Hf, Wf)

#         # if self.pos_embedding_x is None:
#         #     self.pos_embedding_x = nn.Parameter(torch.zeros(1,  self.embed_dim,  Hf, Wf, device=x.device))
#         #     nn.init.trunc_normal_(self.pos_embedding_x, std=0.02)
            
#         # transformed_tokens = transformed_tokens + self.pos_embedding_x  # (B, C, Hf, Wf)
#         # print(transformed_tsokens.shape)
#         # transformed_tokens = transformed_tokens.permute(0, 2, 3, 1).contiguous().view(B, Hf*Wf, -1)  # (B, N, C)
#         # logits = self.classifier(transformed_tokens)
#         # logits = logits.permute(0, 2, 1).view(B, self.patch_output**2, Hf, Wf)  # (B, patch_output^2, Hf, Wf)
        
#         logits = self.head(transformed_tokens)   # (B, num_classes, Hf, Wf)
#         logits = F.pixel_shuffle(logits, upscale_factor=4) 
#         return logits
    
# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.15,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
#         self.backbone = Patch3DTransformerSegmentation(num_classes=1, patch_output=4)
#         self.scale_factor = 8
    
#     def forward(self, x):

#         x = x.permute(0,2,1,3,4)
#         output = self.backbone(x)  # runs backbone, sets self.feature
#         return output



# GOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD  
class SwinModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
        super(SwinModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.IGNORE_INDEX = 127

        self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

        self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

        self.backbone = swin_transformer.swin3d_t(weights="KINETICS400_V1") #KINETICS400_IMAGENET22K_V1
                
        # Remove global pooling + classification head
        embed_dim = self.backbone.head.in_features 

        # self.backbone.avgpool = nn.Identity()
        self.backbone.head = nn.Identity()

        self.classifier = nn.Sequential(
                nn.Linear(embed_dim,(self.hparams.size//16)**2),
        )
    
    def forward(self, x):

        x = x.permute(0,2,1,3,4)
        preds = self.backbone(x)  # runs backbone, sets self.feature
        preds = self.classifier(preds)
        preds = preds.view(-1,1,self.hparams.size//16,self.hparams.size//16)
        return preds

# ## GOOOOOOOOOOOOOOOOOODDDDDDDDDDDDD
# class SwinModel(pl.LightningModule):

#     def __init__(self, pred_shape, size, lr, scheduler=None,wandb_logger=None,freeze=False):
#         super(SwinModel, self).__init__()
#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.25,ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
#         # model = swin_transformer.swin3d_t(weights="KINETICS400_V1")#KINETICS400_IMAGENET22K_V1 
        
        
#         # self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
#         # self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

#         self.backbone = swin_transformer.swin3d_t(weights="KINETICS400_V1") #KINETICS400_IMAGENET22K_V1 
#         self.backbone.head = nn.Identity()
#         # self.decoder = Decoder2D(in_channels=768, num_classes=1)
#         # self.decoder = Decoder2D(in_channels=512, num_classes=1)
#         # self.classifier = nn.Sequential(
#         #     nn.Linear(384, 1),  
#         # )
#         self.classifier = nn.Sequential(
#             nn.Linear(384, 1),  
#         )

#         self.features = None
#         # self.hook_handle = self.backbone.norm.register_forward_hook(self._hook_fn)
#         # self.hook_handle = self.backbone.features[-1].register_forward_hook(self._hook_fn)
#         self.hook_handle = self.backbone.features[4].register_forward_hook(self._hook_fn)
#         # self.hook_handle = self.backbone.norm.register_forward_hook(self._hook_fn)

#     def _hook_fn(self, module, input, output):
#         self.features = output

#     def forward(self, x):
#         x = x.permute(0,2,1,3,4)
#         _ = self.backbone(x)  # runs backbone, sets self.features
#         feat = self.features  # (B, T_patch, H_patch, W_patch, C)
#         feat = feat.permute(0, 4, 1, 2, 3)  # (B, C, T_patch, H_patch, W_patch)
#         feat_2d = feat.max(dim=2)[0]  # average temporal patches: (B, C, H_patch, W_patch)
#         # print(feat_2d.shape)
#         seg_logits  = self.classifier(feat_2d.permute(0, 2, 3, 1)).view(-1,1,self.hparams.size//16,self.hparams.size//16)
#         # print(seg_logits.shape)
#         # seg_logits = self.decoder(feat_2d)  # (B, num_classes, 224, 224)
#         return seg_logits

   
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        # print(outputs.shape)
        # print(y.shape)
        loss = self.loss_func(outputs, y)
        if torch.isnan(loss):
            print("Loss nan encountered")
        self.log("train/total_loss", loss.item(),on_step=True, on_epoch=True, prog_bar=True)
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
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
    
    def configure_optimizers(self):
        weight_decay = 0.001
        base_lr = self.hparams.lr

        # 1️⃣ Backbone parameters in their own group
        backbone_params = list(self.parameters())


        # 4️⃣ Define param groups
        param_groups = [
            {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
        ]

        # 5 4Optimizer
        optimizer = AdamW(param_groups)

        # 6 Scheduler
        scheduler = get_scheduler(optimizer, scheduler="cosine", epochs=15, steps_per_epoch=29)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    # def configure_optimizers(self):
    #     weight_decay = 0.001
    #     base_lr = self.hparams.lr

    #     # 1️⃣ Backbone parameters in their own group
    #     backbone_params = list(self.backbone.backbone.parameters())

    #     # 2️⃣ Everything else (decoder, head, etc.)
    #     other_params = [
    #         p for n, p in self.backbone.named_parameters()
    #         if p.requires_grad and not any(k in n for k in ["backbone"])
    #     ]

    #     # 3️⃣ Sanity checks
    #     assert len(other_params) > 0, "No parameters found for 'other_params' group"
    #     assert len(backbone_params) > 0, "No parameters found for 'backbone_params' group"
        
    #     # total_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
    #     # backbone_count = sum(p.numel() for p in backbone_params)
    #     # other_count = sum(p.numel() for p in other_params)

    #     # assert backbone_count + other_count == total_params, \
    #     #     f"Mismatch: total={total_params}, backbone={backbone_count}, other={other_count}"

    #     # 4️⃣ Define param groups
    #     param_groups = [
    #         {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
    #         {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
    #     ]

    #     # 5 4Optimizer
    #     optimizer = AdamW(param_groups)

    #     # 6 Scheduler
    #     scheduler = get_scheduler(optimizer, scheduler="cosine", epochs=15, steps_per_epoch=50)

    #     return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


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

            self.hparams.wandb_logger.log_image(key="masks", images=[np.clip(final_mask, 0, 1)], caption=["probs"])

        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        
        
def load_weights(model, ckpt_path, strict=True, map_location='cpu'):
    """
    Loads weights from a checkpoint into the model.
    
    Args:
        model: An instance of TimesfomerModel.
        ckpt_path: Path to the .ckpt file saved by PyTorch Lightning.
        strict: Whether to strictly enforce that the keys in state_dict match.
        map_location: Where to load the checkpoint (e.g., 'cpu', 'cuda').

    Returns:
        model: The model with loaded weights.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    
    # For Lightning checkpoints, weights are under 'state_dict'
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Strip 'model.' prefix if saved with Lightning
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[new_key] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
    
    print("Loaded checkpoint from:", ckpt_path)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    return model
        
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

def get_scheduler(optimizer, scheduler=None, epochs=15, steps_per_epoch=10):
    if scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group['lr'] for group in optimizer.param_groups],
            total_steps=epochs * 50,
            pct_start=0.1,       # 10% warmup
            anneal_strategy='cos',  # Cosine decay after warmup
            cycle_momentum=False  # Turn off momentum scheduling (common for AdamW)
        )
    elif scheduler == 'cosine':
        scheduler_after = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=1e-6)
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=1.0, total_epoch=4, after_scheduler=scheduler_after)
    elif scheduler == 'linear':
        scheduler_after = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.05, total_iters=epochs)
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=5.0, total_epoch=4, after_scheduler=scheduler_after)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)