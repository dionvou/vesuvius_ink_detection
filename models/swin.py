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

from transformers import SegformerForSemanticSegmentation
from torchvision.models.video import swin_transformer
import albumentations as A

# # Convert to PIL and then to 3 channels
pil_transform = T.Compose([
    T.ToPILImage(),                    # convert (C, H, W) to PIL
    T.Grayscale(num_output_channels=3),  # convert to 3 channels
])
class TimesformerDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.rotate = A.Compose([A.Rotate(8,p=1)])
        self.xyxys=xyxys
        
        
        # self.video_transform = T.Compose([
        #     T.ConvertImageDtype(torch.float32),  # scales to [0.0, 1.0]
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600", use_fast=True)
        # print("Using Timesformer processor with image size:", self.processor)
        self.pil_transform  = pil_transform
        # print("Model preprocess: ",swin_transformer.Swin3D_S_Weights.KINETICS400_V1.transforms)
        # self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600", use_fast=True)

        if isinstance(self.processor.size, dict):
            proc_size = self.processor.size.get("shortest_edge")
        else:
            proc_size = self.processor.size

        if self.cfg.size != proc_size:
            print(f"Processor size: {self.processor.size}")
            print(f"Config size: {self.cfg.size}")
            print('Size does not equal processor size')
            self.processor.do_resize = False
            self.processor.do_center_crop = False
        self.processor.do_normalize = False
        # self.processor.do_resize = False
        # self.processor.do_center_crop = False
        # self.processor.do_rescale = False
        # self.channel_tranform  = T.Compose([
        #     T.Grayscale(num_output_channels=3),  # convert to 3 channels
        # ])
                
    def __len__(self):
        return len(self.images)
    
    def fourth_augment(self,image, in_chans=16):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(in_chans-5, in_chans)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image
    
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
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            image = image.permute(1,0,2,3)
            frames = [self.pil_transform(frame.squeeze(0)) for frame in image] 

            encoding = self.processor(
                [frame for frame in frames],   # list of PIL
                return_tensors='pt'
                )
            pixel_values = encoding["pixel_values"].squeeze(0)
            return pixel_values, label,xy
            # image = image.permute(1,0,2,3)
            # image = image.repeat(1, 3, 1, 1)
            # image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            # return image, label,xy
            
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

            image=self.fourth_augment(image, self.cfg.in_chans)
            # image = self.shuffle_d_axis(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            image = image.permute(1,0,2,3)
            frames = [self.pil_transform(frame.squeeze(0)) for frame in image] 

            encoding = self.processor(
                [frame for frame in frames],   # list of PIL
                return_tensors='pt'
                )
            pixel_values = encoding["pixel_values"].squeeze(0)
            
            return pixel_values, label
            # print(image.shape)
            # image = image.permute(1,0,2,3)
            # image = image.repeat(1, 3, 1, 1)
            # image = torch.stack([self.video_transform(f) for f in image]) # list of frames
            # return image, label
        
class SwinModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False):
        super(SwinModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.IGNORE_INDEX = 127

        self.loss_func1 = smp.losses.DiceLoss(mode='binary', ignore_index=self.IGNORE_INDEX)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25,ignore_index=self.IGNORE_INDEX)

        self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

        self.backbone = swin_transformer.swin3d_b(weights="KINETICS400_IMAGENET22K_V1") #KINETICS400_IMAGENET22K_V1
        # if freeze:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False

        embed_dim = self.backbone.head.in_features  # works before replacing with Identity()
        self.backbone.head = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, (self.hparams.size // 16) ** 2)
        )
        self.dropout = nn.Dropout(p=0.2)
        # self.classifier = nn.Sequential(
        #     nn.Linear(384, 1),  
        # )
    

        # self.features = None
        # self.hook_handle = self.backbone.features[-1].register_forward_hook(self._hook_fn)
        # self.hook_handle = self.backbone.features[4].register_forward_hook(self._hook_fn)
        # self.hook_handle = self.backbone.norm.register_forward_hook(self._hook_fn)
        # 
    # def _hook_fn(self, module, input, output):
    #     self.features = output
        

    # def forward(self, x):
    #     x = x.permute(0,2,1,3,4)
    #     _ = self.backbone(x)  # runs backbone, sets self.features
    #     feat = self.features  # (B, T_patch, H_patch, W_patch, C)
    #     feat_2d = feat.max(dim=1)[0]  # average temporal patches: (B, C, H_patch, W_patch)
    #     seg_logits = self.classifier(feat_2d)  # (B, num_classes, 14, 14)
        # return seg_logits.permute(0,3,1,2)

    def forward(self, x):

        x = x.permute(0,2,1,3,4)
        output = self.backbone(x)  # runs backbone, sets self.features
        # output = self.dropout(output)
        preds = self.classifier(output)
        preds = preds.view(-1,1,self.hparams.size//16,self.hparams.size//16)
        return preds
   
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
        weight_decay = 0.0  # or use self.hparams.weight_decay

        # Separate the classifier (head) and backbone (other) parameters
        head_params = list(self.classifier.parameters())
        other_params = [p for n, p in self.backbone.named_parameters() if "classifier" not in n]

        # Define param groups with different learning rates
        base_lr = self.hparams.lr

        param_groups = [
            {
                'params': other_params,
                'lr': base_lr,
                'weight_decay': weight_decay
            },
            {
                'params': head_params,
                'lr': base_lr ,  # 100x multiplier for the head
                'weight_decay': weight_decay
            },
        ]

        # Optimizer
        optimizer = AdamW(param_groups)
        scheduler= get_scheduler(optimizer,scheduler='cosine')
        # # Scheduler with per-group max_lr
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=[base_lr, base_lr * 10],  # 👈 match param group order!
        #     total_steps=64*25,
        #     pct_start=0.1,
        #     anneal_strategy='cos',
        #     final_div_factor=1e2,
        #     cycle_momentum=False,
        # )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # def configure_optimizers(self):
    #     weight_decay = 0.01#self.hparams.weight_decay if hasattr(self.hparams, 'weight_decay') else 0.01

    #     # Separate the head parameters
    #     head_params = list(self.classifier.parameters())
    #     other_params = [p for n, p in self.backbone.named_parameters() if "classifier" not in n]

    #     param_groups = [
    #         {
    #             'params': other_params,
    #             'lr': self.hparams.lr,
    #             'weight_decay': weight_decay
    #         },
    #         {
    #             'params': head_params,
    #             'lr': self.hparams.lr * 200,
    #             'weight_decay': weight_decay
    #         },
    #     ]

    #     optimizer = AdamW(param_groups)
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer,
    #         max_lr=[group['lr'] for group in optimizer.param_groups],
    #         total_steps=780,
    #         pct_start=0.1,       # 10% warmup
    #         anneal_strategy='cos',  # Cosine decay after warmup
    #         cycle_momentum=False  # Turn off momentum scheduling (common for AdamW)
    #     )
    #     # scheduler = get_scheduler(optimizer, scheduler=self.hparams.scheduler)
    #     return [optimizer], [scheduler]
        
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

def get_scheduler(optimizer, scheduler=None, epochs=40, steps_per_epoch=10):
#     if scheduler == 'cosine':
#         scheduler_after = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, 40, eta_min=1e-6)
#     else:
#         scheduler_after = torch.optim.lr_scheduler.LinearLR(
#             optimizer,
#             start_factor=1.0,   # start at the current learning rate
#             end_factor=0.05,    # end at 1% of the current learning rate
#             total_iters = epochs
#         )
#     scheduler = GradualWarmupSchedulerV2(
#         optimizer, multiplier=1.0, total_epoch=4, after_scheduler=scheduler_after)
#     return scheduler
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
    else:
        scheduler_after = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.05, total_iters=epochs)
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=1.0, total_epoch=4, after_scheduler=scheduler_after)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)