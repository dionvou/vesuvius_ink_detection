import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, TimesformerModel
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
import sys
sys.path.append('/home/ubuntu/TimeSformer')  # adjust to your path
from timesformer.models.vit import TimeSformer 


# Convert to PIL and then to 3 channels
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
        self.xyxys=xyxys
        
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k600", use_fast=True)
        self.pil_transform  = pil_transform
        
    def __len__(self):
        return len(self.images)
    
    def fourth_augment(self,image, in_chans=32):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(in_chans-10, in_chans)

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

            image=self.fourth_augment(image, self.cfg.in_chans)
            
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

class TimesfomerModel(pl.LightningModule):
    def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None):
        super(TimesfomerModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary',smooth=0.25)
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y: 0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
    
        self.backbone=TimeSformer(img_size=224, num_classes=0, num_frames=32,attention_type='divided_space_time', in_chans=3)
        checkpoint = torch.load('checkpoints/TimeSformer_divST_32x32_224_HowTo100M.pyth')
        state_dict = checkpoint['model_state']
        state_dict.pop('model.head.weight', None)
        state_dict.pop('model.head.bias', None)

        try:
            self.backbone.load_state_dict(state_dict, strict=True)
            print("Backbone weights loaded successfully.")
        except RuntimeError as e:
            print(f"Failed to load backbone weights: {e}")
        
        self.classifier = nn.Sequential(
            nn.Linear(768, (self.hparams.size // 16)**2)
        )
        
    def forward(self, x):

        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
        features = self.backbone(x)  # (B, 768)
        preds = self.classifier(features)  # (B, H'*W')
        preds = preds.reshape(-1,1,self.hparams.size//16,self.hparams.size//16)
        return preds
   
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        outputs = outputs
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
        outputs = outputs
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
        # Separate the head parameters
        head_params = list(self.classifier.parameters())
        other_params = [p for n, p in self.backbone.named_parameters() if "classifier" not in n]

        param_groups = [
            {'params': other_params, 'lr': self.hparams.lr},
            {'params': head_params, 'lr': self.hparams.lr*10 },  # 10x LR for the head
        ]

        optimizer = AdamW(param_groups)
    
        scheduler = get_scheduler(optimizer,scheduler=self.hparams.scheduler)
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

def get_scheduler(optimizer, scheduler=None, epochs=30):
    if scheduler == 'cosine':
        scheduler_after = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 20, eta_min=1e-6)
    else:
        scheduler_after = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,   # start at the current learning rate
            end_factor=0.05,    # end at 1% of the current learning rate
            total_iters = epochs
        )
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_after)
    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)