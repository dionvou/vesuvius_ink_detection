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
        self.scale_factor = 4
        
        self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32),  # scales to [0.0, 1.0]
        ])
        
        # Conditionally add Normalize transformation
        if norm:
            self.video_transform = T.Compose([
            T.ConvertImageDtype(torch.float32), 
            T.Normalize(mean=[0.5], std=[0.5])
            ])
                
    def __len__(self):
        return len(self.images)
    
    # def fourth_augment(self, image):
    #     """
    #     Custom channel augmentation that returns exactly 24 channels.
    #     """
    #     # always select 8
    #     cropping_num =  self.cfg.valid_chans 

    #     # pick crop indices
    #     start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
    #     crop_indices = np.arange(start_idx, start_idx + cropping_num)

    #     # pick where to paste them
    #     start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

    #     # container
    #     image_tmp = np.zeros_like(image)

    #     # paste cropped channels
    #     image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

    #     # # optional random cutout
    #     # cutout_idx = random.randint(0, 1)
    #     # temporal_random_cutout_idx = np.arange(start_paste_idx, start_paste_idx + cutout_idx)
    #     # if random.random() > 0.4:
    #     #     image_tmp[..., temporal_random_cutout_idx] = 0

    #     # finally, keep only 24 channels
    #     image = image_tmp[..., start_paste_idx:start_paste_idx + cropping_num]

    #     return image
    def fourth_augment(self, image):
        """
        Custom channel augmentation that returns exactly 24 channels.
        """
        # always select 24 channels
        remove_n = 0
        cropping_num = self.cfg.valid_chans  + remove_n

        # pick crop indices
        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        # pick where to paste them
        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        
        # container
        image_tmp = np.zeros_like(image)
        image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]
        
        # --- remove 2 random channels (instead of blacking out) ---
        all_indices = np.arange(start_paste_idx, start_paste_idx + cropping_num)
        remove_idx = np.random.choice(all_indices, remove_n, replace=False)
        # keep all except removed ones
        keep_idx = np.array([i for i in all_indices if i not in remove_idx])
        image_tmp = image_tmp[..., keep_idx]

        return image_tmp
    
    def shuffle(self, image):
        """
        Channel shuffle augmentation that returns exactly 24 channels.
        """
        # Shuffle channels randomly
        shuffled_indices = np.random.permutation(self.cfg.valid_chans)
        image_shuffled = image[..., shuffled_indices]

        return image_shuffled


    def __getitem__(self, idx):
        if self.xyxys is not None: #VALID
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//self.scale_factor,self.cfg.size//self.scale_factor)).squeeze(0)
                
            image = image.permute(1,0,2,3)
            image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            return image, label,xy
            
        else:
            image = self.images[idx]
            label = self.labels[idx]

            if self.aug == 'fourth':
                image=self.fourth_augment(image)
                # image = self.shuffle(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//self.scale_factor,self.cfg.size//self.scale_factor)).squeeze(0)

            image = image.permute(1,0,2,3)
            image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            return image, label

# # HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH  GOOOOOOOOOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
# class Patch3DTransformerSegmentation(nn.Module):
#     def __init__(self, num_classes=1, embed_dim=768, num_heads=8, depth=2, patch_output=4):
#         super().__init__()
#         self.patch_output = patch_output
#         self.num_classes = num_classes

#         backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # Modify first conv layer to accept 1 channel instead of 3
#         old_proj = backbone.patch_embed.proj
#         new_proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_proj.out_channels,
#             kernel_size=old_proj.kernel_size,
#             stride=old_proj.stride,
#             padding=old_proj.padding,
#             bias=old_proj.bias is not None
#         )
#         # Initialize weights by summing across RGB channels
#         with torch.no_grad():
#             # old_proj.weight shape: [out_channels, 3, kT, kH, kW]
#             summed = old_proj.weight.sum(dim=1, keepdim=True)   # -> [out_channels, 1, kT, kH, kW]
#             new_proj.weight.copy_(summed)

#             # If bias exists, copy it too
#             if old_proj.bias is not None:
#                 new_proj.bias.copy_(old_proj.bias)

#         # Replace the old conv with the new one
#         backbone.patch_embed.proj = new_proj
        
#         backbone = nn.Sequential(*list(backbone.children())[:-2]) 
        
#         ckpt_path = "pretraining/checkpoints/tiny_epoch=72.ckpt"
#         ckpt = torch.load(ckpt_path, map_location='cpu',weights_only=False)  # CPU first, move to GPU later if needed

#         if 'state_dict' in ckpt:
#             state_dict = ckpt['state_dict']
#         else:
#             state_dict = ckpt
#         # Filter only encoder weights (keys start with 'encoder.')
#         encoder_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('encoder.'):
#                 # remove 'encoder.' prefix to match backbone keys
#                 encoder_state_dict[k.replace('encoder.', '')] = v

#         # missing, unexpected =  backbone.load_state_dict(encoder_state_dict, strict=False)
#         # print("Missing keys:", missing)
#         # print("Unexpected keys:", unexpected)
#         self.backbone = backbone
#         # # Freeze the backbone
#         # for param in self.backbone.parameters():
#         #     param.requires_grad = False
        

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim,
#             dropout=0.2,
#             activation="gelu",
#             batch_first=True 
#         )
#         self.embed_dim = embed_dim

#         self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim, (patch_output ** 2))
#         )

#     def forward(self, x):
#         B, C, T, H, W = x.shape
#         # print(x.shape)
#         feats = self.backbone(x)  # (B, T', H', W',embed_dim)

#         # Temporal pooling
#         feats = feats.max(dim=1)[0]# (B, 1, Hf, Wf, embed_dim,)
        
#         B, Hf, Wf, D = feats.shape

#         # Move embed_dim to dim 1 and flatten patches
#         patch_tokens = feats.permute(0, 3, 1, 2).contiguous()  # (B, 768, 8, 7, 7)
#         patch_tokens = patch_tokens.view(B, Hf*Wf, -1)        # (B, 768, 8*7*7=392)
       
#         # print(patch_tokens.shape)
#         transformed_tokens = self.decoder(patch_tokens) 
#         logits = self.classifier(transformed_tokens)
#         logits = logits.permute(0, 2, 1).view(B, self.patch_output**2, Hf, Wf)  # (B, patch_output^2, Hf, Wf)
        
#         # logits = self.head(transformed_tokens)   # (B, num_classes, Hf, Wf)
#         logits = F.pixel_shuffle(logits, upscale_factor=self.patch_output) 
#         return logits
    
# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.15,ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.6 * self.loss_func1(x,y) +0.4*self.loss_func2(x,y)
#         self.backbone = Patch3DTransformerSegmentation(num_classes=1, patch_output=4)
    
#     def forward(self, x):

#         x = x.permute(0,2,1,3,4)
#         output = self.backbone(x)  # runs backbone, sets self.feature
#         return output

# # class Swin3DEncoder(nn.Module):
# #     def __init__(self,pretrained_ckpt='pretraining/checkpoints/64_tiny_16_epoch=16.ckpt'):
# #         super().__init__()
# #         # backbone = swin_transformer.SwinTransformer3d(
# #         #     patch_size=[2, 4, 4],
# #         #     embed_dim=96,
# #         #     depths=[2, 2, 12, 2],
# #         #     num_heads=[3, 6, 12, 24],
# #         #     window_size=[8, 7, 7],
# #         #     stochastic_depth_prob=0.1,
# #         # )
# #         # backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')
# #         #         # Get old weights
# #         # old_conv = backbone.patch_embed.proj  # Conv3d(3, 128, ...)
# #         # weight = old_conv.weight  # [128, 3, 2, 4, 4]
# #         # bias = old_conv.bias      # [128]

# #         # # Adapt weights: average across RGB → 1 channel
# #         # new_weight = weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]

# #         # # Replace conv with new one (keep out_channels=128!)
# #         # backbone.patch_embed.proj = nn.Conv3d(
# #         #     in_channels=1,
# #         #     out_channels=128,
# #         #     kernel_size=(2, 4, 4),
# #         #     stride=(2, 4, 4),
# #         #     bias=True
# #         # )

# #         # # # Load adapted weights
#         # backbone.patch_embed.proj.weight = nn.Parameter(new_weight)
#         # backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())  # shape [128]
#         # self.backbone = backbone
#         self.backbone = swin_transformer.swin3d_b(weights='KINETICS400_IMAGENET22K_V1')

#         # --- patch_embed adaptation for 1 channel ---
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

#         # remove classifier head, eep norm
#         self.backbone.head = nn.Identity()

#         # if pretrained_ckpt:
#         #     ckpt = torch.load(pretrained_ckpt, map_location='cpu',weights_only=False)
#         #     ckpt_state_dict = ckpt['state_dict']  # your loaded checkpoint
#         #     model_state_dict = self.backbone.state_dict()

#         #     new_state_dict = {}
#         #     used_ckpt_keys = set()  # keep track of keys already used


#         #     for k_model in model_state_dict.keys():
#         #         # Skip decoder stuff
#         #         if any(x in k_model for x in ['decoder_pos_embed', 'mask_token', 'decoder']):
#         #             continue
#         #         print(k_model)
#         #         # Find corresponding key in checkpoint
#         #         if k_model in ckpt_state_dict:
#         #             new_state_dict[k_model] = ckpt_state_dict[k_model]
#         #         else:
#         #             # Fallback: find first unused key with matching shape
#         #             for k_ckpt, v_ckpt in ckpt_state_dict.items():
#         #                 if k_ckpt in used_ckpt_keys:
#         #                     continue  # skip already used keys
#         #                 if v_ckpt.shape == model_state_dict[k_model].shape:
#         #                     new_state_dict[k_model] = v_ckpt
#         #                     used_ckpt_keys.add(k_ckpt)
#         #                     print(f"Fallback match: {k_model} <- {k_ckpt}")
#         #                     break
                            
#         #     # Load into model
#         #     msg = self.backbone.load_state_dict(new_state_dict, strict=False)
#         #     print("Loaded:", msg)
        
        

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)   # (B, T/2, H/4, W/4, 96)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):  # stage output
#                 skips.append(x)

#         return skips  # [s1, s2, s3, s4]





class Swin3DEncoder(nn.Module):
    def __init__(self, pretrained_ckpt='pretraining/checkpoints/64_tiny_16_epoch=17.ckpt', in_chans=1):
        super().__init__()
        # Load Swin3D
        self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

        # --- patch_embed adaptation for 1 channel ---
        old_conv = self.backbone.patch_embed.proj
        weight = old_conv.weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]
        bias = old_conv.bias

        self.backbone.patch_embed.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            bias=True
        )
        self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
        self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

        # remove classifier head, keep norm
        self.backbone.head = nn.Identity()

        # pretrained_ckpt = 'pretraining/checkpoints/tiny_epoch=71.ckpt'
        # # optionally load ckpt
        # if pretrained_ckpt:
        #     ckpt = torch.load(pretrained_ckpt, map_location='cpu',weights_only=False)
        #     ckpt_state_dict = ckpt['state_dict']  # your loaded checkpoint
        #     model_state_dict = self.backbone.state_dict()

        #     new_state_dict = {}
        #     used_ckpt_keys = set()  # keep track of keys already used


        #     for k_model in model_state_dict.keys():
        #         # Skip decoder stuff
        #         if any(x in k_model for x in ['decoder_pos_embed', 'mask_token', 'decoder']):
        #             continue
        #         print(k_model)
        #         # Find corresponding key in checkpoint
        #         if k_model in ckpt_state_dict:
        #             new_state_dict[k_model] = ckpt_state_dict[k_model]
        #         else:
        #             # Fallback: find first unused key with matching shape
        #             for k_ckpt, v_ckpt in ckpt_state_dict.items():
        #                 if k_ckpt in used_ckpt_keys:
        #                     continue  # skip already used keys
        #                 if v_ckpt.shape == model_state_dict[k_model].shape:
        #                     new_state_dict[k_model] = v_ckpt
        #                     used_ckpt_keys.add(k_ckpt)
        #                     print(f"Fallback match: {k_model} <- {k_ckpt}")
        #                     break
                            
        #     # Load into model
        #     msg = self.backbone.load_state_dict(new_state_dict, strict=False)
        #     print("Loaded:", msg)

    def forward(self, x):
        x = self.backbone.patch_embed(x)   # (B, T/2, H/4, W/4, 96)
        x = self.backbone.pos_drop(x)

        skips = []
        # backbone.features = [stage0, PatchMerging, stage1, PatchMerging, stage2, PatchMerging, stage3]
        for i, block in enumerate(self.backbone.features):
            x = block(x)
            if isinstance(block, nn.Sequential):  # stage output
                skips.append(x)

        return skips  # [s1, s2, s3, s4]

class UpBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        # squeeze away the temporal dim if present
        if x.dim() == 5:  # (B, C, 1, H, W)
            x = x.squeeze(2)
        if skip.dim() == 5:
            skip = skip.squeeze(2)

        x = self.up(x)

        # pad if needed
        if x.shape[2:] != skip.shape[2:]:
            diff = [s - d for s, d in zip(skip.shape[2:], x.shape[2:])]
            x = nn.functional.pad(x, [0, diff[1], 0, diff[0]])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    
class TransformerHead2D(nn.Module):
    def __init__(self, in_dim, num_heads=8, mlp_ratio=4.0, out_dim=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim)
        )
        self.final = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_seq = x.flatten(2).permute(2, 0, 1)       # (H*W, B, C)
        x_attn = self.attn(self.norm1(x_seq),
                           self.norm1(x_seq),
                           self.norm1(x_seq))[0]
        x_seq = x_seq + x_attn
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        x_seq = self.final(x_seq)                   # (H*W, B, out_dim)
        x = x_seq.permute(1, 2, 0).view(B, -1, H, W)
        return x
    
class SwinModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
        super(SwinModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.IGNORE_INDEX = 127

        self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.15,ignore_index=self.IGNORE_INDEX)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

        self.loss_func= lambda x,y: 0.7 * self.loss_func1(x,y) +0.3*self.loss_func2(x,y)
        self.backbone = Swin3DEncoder()
        self.normalization=nn.BatchNorm3d(num_features=1) 

        # Channels per stage (tiny config)
        self.enc_channels = [96, 192, 384, 768]

        # Bottleneck conv to reshape final feature
        self.bottleneck = nn.Conv3d(self.enc_channels[-1], self.enc_channels[-1], kernel_size=1)

        # Decoder (reverse order)
        self.up3 = UpBlock2D(self.enc_channels[-1], self.enc_channels[-2])
        self.up2 = UpBlock2D(self.enc_channels[-2], self.enc_channels[-3])
        self.up1 = UpBlock2D(self.enc_channels[-3], self.enc_channels[-4])

        # Final segmentation head
        self.head = TransformerHead2D(in_dim=96, num_heads=8, mlp_ratio=4.0, out_dim=1)

    def forward(self, x):
        x= x.permute(0,2,1,3,4)
        x=self.normalization(x)
        skips = self.backbone(x) 
        s1, s2, s3, s4 = skips

        # Swin returns (B, T, H, W, C) → convert to (B, C, T, H, W)
        def permute_fmap(fmap):
            return fmap.permute(0, 4, 1, 2, 3).contiguous()

        feat_maps = map(permute_fmap, (s1, s2, s3, s4))

        s1, s2, s3, s4 = [torch.max(f, dim=2, keepdim=True)[0] for f in feat_maps]

        x = self.up3(s4, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        out = self.head(x)

        return out

# # ---------------- Full UNet ---------------- #
# class SwinUNet3D(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         self.encoder = Swin3DEncoder()

#         # Channels per stage (tiny config)
#         self.enc_channels = [96, 192, 384, 768]

#         # Bottleneck conv to reshape final feature
#         self.bottleneck = nn.Conv3d(self.enc_channels[-1], 768, kernel_size=1)

#         # Decoder (reverse order)
#         self.up3 = UpBlock3D(768, 384)
#         self.up2 = UpBlock3D(384, 192)
#         self.up1 = UpBlock3D(192, 96)

#         # Final segmentation head
#         self.head = nn.Conv3d(96, num_classes, kernel_size=1)

#     def forward(self, x):
#         skips = self.encoder(x)   # list of [stage1, stage2, stage3, stage4]
#         s1, s2, s3, s4 = skips

#         # Swin returns (B, T, H, W, C) → convert to (B, C, T, H, W)
#         def permute_fmap(fmap):
#             return fmap.permute(0, 4, 1, 2, 3).contiguous()

#         s1, s2, s3, s4 = map(permute_fmap, (s1, s2, s3, s4))

#         x = self.bottleneck(s4)
#         print(x.shape)

#         x = self.up3(x, s3)
#         print(x.shape)
#         x = self.up2(x, s2)
#         print(x.shape)
#         x = self.up1(x, s1)
#         print(x.shape)
#         out = self.head(x)
#         return out


# GOOOOOOOOOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
# class Patch3DTransformerSegmentation(nn.Module):
#     def __init__(self, num_classes=1, embed_dim=1024, num_heads=4, depth=2, patch_output=4):
#         super().__init__()
#         self.patch_output = patch_output
#         self.num_classes = num_classes
        
#         # backbone = swin_transformer.swin3d_b(weights=None)
#         backbone = swin_transformer.swin3d_b(weights=None)


#         # Modify first conv layer to accept 1 channel instead of 3
#         old_proj = backbone.patch_embed.proj
#         backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_proj.out_channels,
#             kernel_size=old_proj.kernel_size,
#             stride=old_proj.stride,
#             padding=old_proj.padding,
#             bias=old_proj.bias is not None
#         )
        
#         backbone = nn.Sequential(*list(backbone.children())[:-3]) 

#         # Load checkpoint
#         ckpt_path = "pretraining/checkpoints/epochepoch=20.ckpt"
#         ckpt = torch.load(ckpt_path, map_location='cpu',weights_only=False)  # CPU first, move to GPU later if needed

#         # Check if it's a Lightning checkpoint
#         if 'state_dict' in ckpt:
#             state_dict = ckpt['state_dict']
#         else:
#             state_dict = ckpt
#                 # # Filter only encoder weights (keys start with 'encoder.')
#         encoder_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('encoder.'):
#                 # remove 'encoder.' prefix to match backbone keys
#                 encoder_state_dict[k.replace('encoder.', '')] = v
#         # new_state_dict = {}
#         # # for k, v in encoder_state_dict.items():
#         # #     new_key = k.replace("blocks", "features").replace("patch_embed", "0")  # example
#         # #     new_state_dict[new_key] = v
#         missing, unexpected =  backbone.load_state_dict(encoder_state_dict, strict=False)
#         print("Missing keys:", missing)
#         print("Unexpected keys:", unexpected)
#         self.backbone = backbone
#         # Freeze the backbone
#         # for param in self.backbone.parameters():
#         #     param.requires_grad = False
        

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim,
#             dropout=0.1,
#             activation="gelu",
#             batch_first=True  # (B, N, C)
#         )
#         self.embed_dim = embed_dim
#         self.pos_embedding = None  # will be initialized dynamically after seeing Hf, Wf
#         self.pos_embedding_x = None
#         self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

#         # Classifier: per patch -> small 3D patch
#         self.classifier = nn.Linear(embed_dim, patch_output ** 2)
#         # self.proj = nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1)  # learns spatial filters
#         self.head = nn.Sequential(
#             nn.Conv2d(embed_dim,self.patch_output**2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.patch_output**2,self.patch_output**2, kernel_size=1),
#         )


#     def forward(self, x):
#         B, C, T, H, W = x.shape
#         feats = self.backbone(x)  # (B, T', H', W',embed_dim)

#         # Adaptive temporal pooling
#         feats = feats.mean(dim=1) # (B, 1, Hf, Wf, embed_dim,)
        
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
#         transformed_tokens = transformed_tokens.permute(0, 2, 3, 1).contiguous().view(B, Hf*Wf, -1)  # (B, N, C)
#         logits = self.classifier(transformed_tokens)
#         # logits = logits.permute(0, 2, 1).view(B, self.patch_output**2, Hf, Wf)  # (B, patch_output^2, Hf, Wf)
        
#         # # logits = self.head(transformed_tokens)   # (B, num_classes, Hf, Wf)
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
#         output = self.backbone(x) 
#         return output


# # GOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD  
# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

#         backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')


#         # Modify first conv layer to accept 1 channel instead of 3
#         old_proj = backbone.patch_embed.proj
#         new_proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_proj.out_channels,
#             kernel_size=old_proj.kernel_size,
#             stride=old_proj.stride,
#             padding=old_proj.padding,
#             bias=old_proj.bias is not None
#         )
#         # Initialize weights by summing across RGB channels
#         with torch.no_grad():
#             # old_proj.weight shape: [out_channels, 3, kT, kH, kW]
#             summed = old_proj.weight.sum(dim=1, keepdim=True)  # -> [out_channels, 1, kT, kH, kW]
#             new_proj.weight.copy_(summed)

#             # If bias exists, copy it too
#             if old_proj.bias is not None:
#                 new_proj.bias.copy_(old_proj.bias)
#             # embed_dim = out.shape[1]  # channel dimension


#         # Replace the old conv with the new one
#         backbone.patch_embed.proj = new_proj
#         embed_dim = 768
#         self.backbone = backbone
#         self.backbone.head = nn.Identity()


#         self.classifier = nn.Sequential(
#                 nn.Linear(embed_dim,(self.hparams.size//16)**2),
#         )
    
#     def forward(self, x):

#         x = x.permute(0,2,1,3,4)
#         preds = self.backbone(x)  # runs backbone, sets self.feature
#         preds = self.classifier(preds)
#         preds = preds.view(-1,1,self.hparams.size//16,self.hparams.size//16)
#         return preds


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
#         backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # Modify first conv layer to accept 1 channel instead of 3
#         old_proj = backbone.patch_embed.proj
#         new_proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_proj.out_channels,
#             kernel_size=old_proj.kernel_size,
#             stride=old_proj.stride,
#             padding=old_proj.padding,
#             bias=old_proj.bias is not None
#         )
#         # Initialize weights by summing across RGB channels
#         with torch.no_grad():
#             # old_proj.weight shape: [out_channels, 3, kT, kH, kW]
#             summed = old_proj.weight.sum(dim=1, keepdim=True)   # -> [out_channels, 1, kT, kH, kW]
#             new_proj.weight.copy_(summed)

#             # If bias exists, copy it too
#             if old_proj.bias is not None:
#                 new_proj.bias.copy_(old_proj.bias)

#         # Replace the old conv with the new one
#         backbone.patch_embed.proj = new_proj
#         self.backbone = backbone
        
#         # backbone = nn.Sequential(*list(backbone.children())[:-2]) 

#         # self.backbone = swin_transformer.swin3d_t(weights="KINETICS400_V1") #KINETICS400_IMAGENET22K_V1 
#         self.backbone.head = nn.Identity()
#         # self.decoder = Decoder2D(in_channels=768, num_classes=1)
#         # self.decoder = Decoder2D(in_channels=512, num_classes=1)
#         # self.classifier = nn.Sequential(
#         #     nn.Linear(384, 1),  
#         # )
#         self.classifier = nn.Sequential(
#             nn.Linear(768, 1),  
#         )

#         self.features = None
#         # self.hook_handle = self.backbone.norm.register_forward_hook(self._hook_fn)
#         # self.hook_handle = self.backbone.features[-1].register_forward_hook(self._hook_fn)
#         self.hook_handle = self.backbone.norm.register_forward_hook(self._hook_fn)
#         # self.post_norm = nn.LayerNorm(normalized_shape=(768,), eps=1e-05, elementwise_affine=True)
#         self.pool = nn.AdaptiveAvgPool3d((1, None, None))  # collapse T → 1


#         # self.hook_handle = self.backbone.norm.register_forward_hook(self._hook_fn)

#     def _hook_fn(self, module, input, output):
#         self.features = output

#     def forward(self, x):
#         x = x.permute(0,2,1,3,4)
#         _ = self.backbone(x)  # runs backbone, sets self.features
#         feat = self.features  # (B, T_patch, H_patch, W_patch, C)
#         # feat = self.post_norm(feat) 
#         feat = feat.permute(0, 4, 1, 2, 3)  # (B, C, T_patch, H_patch, W_patch)
#         # feat_2d = self.pool(feat).squeeze()
#         feat_2d = feat.max(dim=2)[0]  # average temporal patches: (B, C, H_patch, W_patch)
#         # print(feat_2d.shape)
#         seg_logits  = self.classifier(feat_2d.permute(0, 2, 3, 1)).view(-1,1,self.hparams.size//32,self.hparams.size//32)
#         # print(seg_logits.shape)
#         # seg_logits = self.decoder(feat_2d)  # (B, num_classes, 224, 224)
#         return seg_logits
# # GOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD  
# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

#         backbone = swin_transformer.swin3d_t(weights=None)

#         embed_dim = backbone.norm.normalized_shape[0]  # usually 1024 for swin3d_b

#         # Modify first conv layer to accept 1 channel instead of 3
#         old_proj = backbone.patch_embed.proj
#         backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=1,
#             out_channels=old_proj.out_channels,
#             kernel_size=old_proj.kernel_size,
#             stride=old_proj.stride,
#             padding=old_proj.padding,
#             bias=old_proj.bias is not None
#         )
        
#         # backbone = nn.Sequential(*list(backbone.children())[:-2]) 
        

#         # # Load checkpoint
#         # ckpt_path = "pretraining/checkpoints/epochepoch=93.ckpt"
#         # ckpt = torch.load(ckpt_path, map_location='cpu',weights_only=False)  # CPU first, move to GPU later if needed

#         # # Check if it's a Lightning checkpoint
#         # if 'state_dict' in ckpt:
#         #     state_dict = ckpt['state_dict']
#         # else:
#         #     state_dict = ckpt
#         #         # # Filter only encoder weights (keys start with 'encoder.')
#         # encoder_state_dict = {}
#         # for k, v in state_dict.items():
#         #     if k.startswith('encoder.'):
#         #         # remove 'encoder.' prefix to match backbone keys
#         #         encoder_state_dict[k.replace('encoder.', '')] = v
#         # new_state_dict = {}
#         # # for k, v in encoder_state_dict.items():
#         # #     new_key = k.replace("blocks", "features").replace("patch_embed", "0")  # example
#         # #     new_state_dict[new_key] = v
#         # missing, unexpected =  backbone.load_state_dict(encoder_state_dict, strict=False)
#         # print("Missing keys:", missing)
#         # print("Unexpected keys:", unexpected)
#         backbone.head = nn.Identity()
#         self.backbone = backbone
#         self.adaptivepool = nn.AdaptiveAvgPool3d(output_size=1)

#         self.classifier = nn.Sequential(
#                 nn.Linear(embed_dim,(self.hparams.size//16)**2),
#         )
    
#     def forward(self, x):

#         x = x.permute(0,2,1,3,4)
#         preds = self.backbone(x)  # runs backbone, sets self.feature
#         preds = self.classifier(preds)
#         preds = preds.view(-1,1,self.hparams.size//16,self.hparams.size//16)
#         return preds
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# class Swin3DEncoder(nn.Module):
#     def __init__(self, pretrained_ckpt='pretraining/checkpoints/64_tiny_16_epoch=15.ckpt', in_chans=1):
#         super().__init__()
#         # Load Swin3D
#         self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # --- patch_embed adaptation for 1 channel ---
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=in_chans,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

#         # remove classifier head, keep norm
#         self.backbone.head = nn.Identity()
        
#         # for param in self.backbone.parameters():
#         #     param.requires_grad = False

#         # optionally load ckpt
#         if pretrained_ckpt:
#             ckpt = torch.load(pretrained_ckpt, map_location='cpu',weights_only=False)
#             ckpt_state_dict = ckpt['state_dict']  # your loaded checkpoint
#             model_state_dict = self.backbone.state_dict()

#             new_state_dict = {}
#             used_ckpt_keys = set()  # keep track of keys already used


#             for k_model in model_state_dict.keys():
#                 # Skip decoder stuff
#                 if any(x in k_model for x in ['decoder_pos_embed', 'mask_token', 'decoder']):
#                     continue
#                 print(k_model)
#                 # Find corresponding key in checkpoint
#                 if k_model in ckpt_state_dict:
#                     new_state_dict[k_model] = ckpt_state_dict[k_model]
#                 else:
#                     # Fallback: find first unused key with matching shape
#                     for k_ckpt, v_ckpt in ckpt_state_dict.items():
#                         if k_ckpt in used_ckpt_keys:
#                             continue  # skip already used keys
#                         if v_ckpt.shape == model_state_dict[k_model].shape:
#                             new_state_dict[k_model] = v_ckpt
#                             used_ckpt_keys.add(k_ckpt)
#                             print(f"Fallback match: {k_model} <- {k_ckpt}")
#                             break
                            
#             # Load into model
#             msg = self.backbone.load_state_dict(new_state_dict, strict=False)
#             print("Loaded:", msg)

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)   # (B, T/2, H/4, W/4, 96)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         # backbone.features = [stage0, PatchMerging, stage1, PatchMerging, stage2, PatchMerging, stage3]
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):  # stage output
#                 skips.append(x)

#         return skips  # [s1, s2, s3, s4]

# # patch
# class Swin3DEncoder(nn.Module):
#     def __init__(self, pretrained_ckpt='pretraining/checkpoints/tiny_epoch=70.ckpt', in_chans=1):
#         super().__init__()
#         # Load Swin3D
#         self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

#         # --- patch_embed adaptation for 1 channel ---
#         old_conv = self.backbone.patch_embed.proj
#         weight = old_conv.weight.sum(dim=1, keepdim=True)  # [128, 1, 2, 4, 4]
#         bias = old_conv.bias

#         self.backbone.patch_embed.proj = nn.Conv3d(
#             in_channels=in_chans,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             bias=True
#         )
#         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
#         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())

#         # remove classifier head, keep norm
#         self.backbone.head = nn.Identity()

#         # # optionally load ckpt
#         # if pretrained_ckpt:
#         #     ckpt = torch.load(pretrained_ckpt, map_location='cpu',weights_only=False)
#         #     ckpt_state_dict = ckpt['state_dict']  # your loaded checkpoint
#         #     model_state_dict = self.backbone.state_dict()

#         #     new_state_dict = {}
#         #     used_ckpt_keys = set()  # keep track of keys already used


#         #     for k_model in model_state_dict.keys():
#         #         # Skip decoder stuff
#         #         if any(x in k_model for x in ['decoder_pos_embed', 'mask_token', 'decoder']):
#         #             continue
#         #         print(k_model)
#         #         # Find corresponding key in checkpoint
#         #         if k_model in ckpt_state_dict:
#         #             new_state_dict[k_model] = ckpt_state_dict[k_model]
#         #         else:
#         #             # Fallback: find first unused key with matching shape
#         #             for k_ckpt, v_ckpt in ckpt_state_dict.items():
#         #                 if k_ckpt in used_ckpt_keys:
#         #                     continue  # skip already used keys
#         #                 if v_ckpt.shape == model_state_dict[k_model].shape:
#         #                     new_state_dict[k_model] = v_ckpt
#         #                     used_ckpt_keys.add(k_ckpt)
#         #                     print(f"Fallback match: {k_model} <- {k_ckpt}")
#         #                     break
                            
#         #     # Load into model
#         #     msg = self.backbone.load_state_dict(new_state_dict, strict=False)
#         #     print("Loaded:", msg)

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)   # (B, T/2, H/4, W/4, 96)
#         x = self.backbone.pos_drop(x)

#         skips = []
#         # backbone.features = [stage0, PatchMerging, stage1, PatchMerging, stage2, PatchMerging, stage3]
#         for i, block in enumerate(self.backbone.features):
#             x = block(x)
#             if isinstance(block, nn.Sequential):  # stage output
#                 skips.append(x)

#         return skips
    
          
# from einops import rearrange
# # ---------------- 2D PatchExpansion ---------------- #
# class PatchExpansion2D(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim_in = dim
#         self.expand = nn.Linear(dim, dim * 4, bias=False)  # expand for 2x2
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         # x: (B, H, W, C)
#         B, H, W, C = x.shape
#         x = self.expand(x)  # (B, H, W, 4*C)
#         x = x.view(B, H, W, 2, 2, C)  # split channels into 2×2 upsampling
#         x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H, 2, W, 2, C)
#         x = x.view(B, H * 2, W * 2, C)  # (B, H*2, W*2, C)
#         x = self.norm(x)
#         return x


# # ---------------- PatchExpansion2D Decoder Block ---------------- #
# class UpBlockPatch2D(nn.Module):
#     def __init__(self, in_ch, out_ch, skip_ch):
#         super().__init__()
#         self.dim_in = in_ch
#         self.dim_out = out_ch
#         self.expand = PatchExpansion2D(in_ch)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x, skip):
#         # x: (B, C, H, W)
#         B, C, H, W = x.shape

#         # (B, H, W, C) → expand → (B, H*2, W*2, C)
#         x_reshaped = x.permute(0, 2, 3, 1).contiguous()
#         x_ups = self.expand(x_reshaped)
#         # back to (B, C, H, W)
#         x_ups = x_ups.permute(0, 3, 1, 2).contiguous()

#         # match skip size if different
#         if x_ups.shape[2:] != skip.shape[2:]:
#             diff = [s - d for s, d in zip(skip.shape[2:], x_ups.shape[2:])]
#             x_ups = F.pad(x_ups, [0, diff[1], 0, diff[0]])

#         # concatenate skip connection
#         x_cat = torch.cat([x_ups, skip], dim=1)
#         return self.conv(x_cat)

    
# class SwinModel(pl.LightningModule):
#     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
#         super(SwinModel, self).__init__()

#         self.save_hyperparameters()
#         self.mask_pred = np.zeros(self.hparams.pred_shape)
#         self.mask_count = np.zeros(self.hparams.pred_shape)
#         self.IGNORE_INDEX = 127

#         self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.15,ignore_index=self.IGNORE_INDEX)
#         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

#         self.loss_func= lambda x,y: 0.6 * self.loss_func1(x,y) +0.4*self.loss_func2(x,y)

#         self.backbone = Swin3DEncoder()
#         self.enc_channels = [96, 192, 384, 768]
#         self.pool = nn.AdaptiveAvgPool3d((1, None, None))

#         # Decoder (2D)
#         self.up3 = UpBlockPatch2D(self.enc_channels[-1],self.enc_channels[-2], self.enc_channels[-2])
#         self.up2 = UpBlockPatch2D(self.enc_channels[-2],self.enc_channels[-3], self.enc_channels[-3])
#         self.up1 = UpBlockPatch2D(self.enc_channels[-3],self.enc_channels[-4], self.enc_channels[-4])

#         # Final head
#         self.head = nn.Conv2d(self.enc_channels[0], 1, kernel_size=1)

#     def forward(self, x):
#         # get swin feature maps [s1, s2, s3, s4]
#         x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) → (B, T, C, H, W)
#         feat_maps = self.backbone(x)
#         feat_maps = [f.permute(0, 4, 1, 2, 3).contiguous() for f in feat_maps]  # → (B, C, T, H, W)

        
#         # temporal aggregation
#         # s1, s2, s3, s4 = [torch.max(f, dim=2)[0] for f in feat_maps]  # → (B, C, H, W)
#         s1, s2, s3, s4 = [self.pool(f).squeeze() for f in feat_maps]
#         # print(s1.shape,s2.shape,s3.shape,s4.shape)


#         x = self.up3(s4, s3)
#         x = self.up2(x, s2)
#         x = self.up1(x, s1)

#         out = self.head(x)  # (B, 1, H, W)
#         return out
    






# # import torch
# # import torch.nn as nn
# # import pytorch_lightning as pl
# # import segmentation_models_pytorch as smp

# # # Assume swin_transformer is imported already

# # # ---------------- Transformer Head ---------------- #
# # class TransformerHead2D(nn.Module):
# #     def __init__(self, in_dim, num_heads=8, mlp_ratio=4.0, out_dim=1):
# #         super().__init__()
# #         self.norm1 = nn.LayerNorm(in_dim)
# #         self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)
# #         self.norm2 = nn.LayerNorm(in_dim)
# #         self.mlp = nn.Sequential(
# #             nn.Linear(in_dim, int(in_dim * mlp_ratio)),
# #             nn.GELU(),
# #             nn.Linear(int(in_dim * mlp_ratio), in_dim)
# #         )
# #         self.final = nn.Linear(in_dim, out_dim)

# #     def forward(self, x):
# #         # x: (B, C, H, W)
# #         B, C, H, W = x.shape
# #         x_seq = x.flatten(2).permute(2, 0, 1)       # (H*W, B, C)
# #         x_attn = self.attn(self.norm1(x_seq),
# #                            self.norm1(x_seq),
# #                            self.norm1(x_seq))[0]
# #         x_seq = x_seq + x_attn
# #         x_seq = x_seq + self.mlp(self.norm2(x_seq))
# #         x_seq = self.final(x_seq)                   # (H*W, B, out_dim)
# #         x = x_seq.permute(1, 2, 0).view(B, -1, H, W)
# #         return x

# # # ---------------- Bottleneck ---------------- #
# # class BottleneckSwinBlock2D(nn.Module):
# #     def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
# #         super().__init__()
# #         self.norm1 = nn.LayerNorm(dim)
# #         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
# #         self.norm2 = nn.LayerNorm(dim)
# #         self.mlp = nn.Sequential(
# #             nn.Linear(dim, int(dim * mlp_ratio)),
# #             nn.GELU(),
# #             nn.Linear(int(dim * mlp_ratio), dim)
# #         )

# #     def forward(self, x):
# #         B, C, H, W = x.shape
# #         x_seq = x.flatten(2).permute(2, 0, 1)
# #         x_attn = self.attn(self.norm1(x_seq),
# #                            self.norm1(x_seq),
# #                            self.norm1(x_seq))[0]
# #         x_seq = x_seq + x_attn
# #         x_seq = x_seq + self.mlp(self.norm2(x_seq))
# #         x = x_seq.permute(1, 2, 0).view(B, C, H, W)
# #         return x

# # # ---------------- Swin3D Encoder ---------------- #
# # class Swin3DEncoder(nn.Module):
# #     def __init__(self, in_chans=1):
# #         super().__init__()
# #         self.backbone = swin_transformer.swin3d_t(weights='KINETICS400_V1')

# #         old_conv = self.backbone.patch_embed.proj
# #         weight = old_conv.weight.sum(dim=1, keepdim=True)
# #         bias = old_conv.bias

# #         self.backbone.patch_embed.proj = nn.Conv3d(
# #             in_channels=in_chans,
# #             out_channels=old_conv.out_channels,
# #             kernel_size=old_conv.kernel_size,
# #             stride=old_conv.stride,
# #             bias=True
# #         )
# #         self.backbone.patch_embed.proj.weight = nn.Parameter(weight)
# #         self.backbone.patch_embed.proj.bias = nn.Parameter(bias.clone())
# #         self.backbone.head = nn.Identity()

# #     def forward(self, x):
# #         x = self.backbone.patch_embed(x)
# #         x = self.backbone.pos_drop(x)
# #         skips = []
# #         for block in self.backbone.features:
# #             x = block(x)
# #             if isinstance(block, nn.Sequential):
# #                 skips.append(x)
# #         return skips

# # # ---------------- Patch Expansion ---------------- #
# # class PatchExpansion2D(nn.Module):
# #     def __init__(self, dim):
# #         super().__init__()
# #         self.norm = nn.LayerNorm(dim)

# #     def forward(self, x):
# #         # x: (B, H, W, C)
# #         B, H, W, C = x.shape
# #         # 2x spatial upsample by reshaping
# #         x = x.view(B, H, W, 2, 2, C)
# #         x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
# #         x = x.view(B, H*2, W*2, C)
# #         x = self.norm(x)
# #         return x

# # class UpBlockPatch2D(nn.Module):
# #     def __init__(self, in_ch, out_ch, skip_ch=None):
# #         super().__init__()
# #         self.proj = nn.Linear(in_ch, out_ch)  # project channels

# #     def forward(self, x, skip=None):
# #         # x: (B, C, H, W)
# #         B, C, H, W = x.shape
# #         # Simple 2x upsample spatially
# #         x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
# #         # Project channels
# #         # Need to reshape to (B, H*W, C) for linear
# #         x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
# #         x = self.proj(x)                        # (B, H, W, out_ch)
# #         x = x.permute(0, 3, 1, 2).contiguous()  # (B, out_ch, H, W)
# #         return x


# # # ---------------- SwinModel ---------------- #
# # class SwinModel(pl.LightningModule):
# #     def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
# #         super(SwinModel, self).__init__()

# #         self.save_hyperparameters()
# #         self.mask_pred = np.zeros(self.hparams.pred_shape)
# #         self.mask_count = np.zeros(self.hparams.pred_shape)
# #         self.IGNORE_INDEX = 127

# #         self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.15,ignore_index=self.IGNORE_INDEX)
# #         self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)
# #         self.loss_func = lambda x, y: 0.6 * self.loss_func1(x, y) + 0.4 * self.loss_func2(x, y)

# #         # Encoder
# #         self.backbone = Swin3DEncoder()
# #         self.enc_channels = [96, 192, 384, 768]

# #         # Bottleneck
# #         self.bottleneck = BottleneckSwinBlock2D(dim=768, num_heads=8)

# #         # Decoder
# #         self.up3 = UpBlockPatch2D(in_ch=768, out_ch=384)
# #         self.up2 = UpBlockPatch2D(in_ch=384, out_ch=192)
# #         self.up1 = UpBlockPatch2D(in_ch=192, out_ch=96)

# #         # Head
# #         self.head = TransformerHead2D(in_dim=96, num_heads=8, mlp_ratio=4.0, out_dim=1)

# #     def forward(self, x):
# #         # x: (B, T, C, H, W) → (B, C, T, H, W)
# #         x = x.permute(0, 2, 1, 3, 4)
# #         feat_maps = self.backbone(x)
# #         feat_maps = [f.permute(0, 4, 1, 2, 3).contiguous() for f in feat_maps]

# #         # Temporal aggregation
# #         s1, s2, s3, s4 = [torch.max(f, dim=2)[0] for f in feat_maps]

# #         # Bottleneck
# #         x = self.bottleneck(s4)

# #         # Decoder
# #         x = self.up3(x)
# #         x = self.up2(x)
# #         x = self.up1(x)

# #         # Head
# #         out = self.head(x)
#         return out

    
   
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
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

        
    # def configure_optimizers(self):
    #     weight_decay = 0.00001
    #     base_lr = self.hparams.lr

    #     # 1️⃣ Backbone parameters in their own group
    #     backbone_params = list(self.backbone.backbone.parameters())

    #     # 2️⃣ Everything else (decoder, head, etc.)
    #     other_params = [
    #         p for n, p in self.backbone.named_parameters()
    #         if p.requires_grad and not any(k in n for k in ["backbone"])
    #     ]
# d
    #     # 3️⃣ Sanity checks
    #     assert len(other_params) > 0, "No parameters found for 'other_params' group"
    #     assert len(backbone_params) > 0, "No parameters found for 'backbone_params' group"
        
    #     # 4️⃣ Define param groups
    #     param_groups = [
    #         {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
    #         {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
    #     ]

    #     # 5 4Optimizer
    #     optimizer = AdamW(param_groups)

    #     # 6 Scheduler
    #     scheduler = get_scheduler(optimizer, scheduler="linear", epochs=25, steps_per_epoch=50)

    #     return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    def configure_optimizers(self):
        weight_decay = 0.0001
        base_lr = self.hparams.lr

        # 1️⃣ Backbone parameters in their own group
        backbone_params = list(self.parameters())

        # # 2️⃣ Everything else (decoder, head, etc.)
        # other_params = [
        #     p for n, p in self.backbone.named_parameters()
        #     if p.requires_grad and not any(k in n for k in ["backbone"])
        # ]

        # # 3️⃣ Sanity checks
        # assert len(other_params) > 0, "No parameters found for 'other_params' group"
        # assert len(backbone_params) > 0, "No parameters found for 'backbone_params' group"
        
        # # 4️⃣ Define param groups
        # param_groups = [
        #     {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
        #     {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
        # ]

        # 5 4Optimizer
        optimizer = AdamW(backbone_params, lr=base_lr, weight_decay=weight_decay)

        # 6 Scheduler

        return [optimizer]



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