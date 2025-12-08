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
from transformers import VideoMAEConfig, VideoMAEForPreTraining
from transformers import VideoMAEModel


class VideoMaeModelV2(pl.LightningModule):
    """
    Enhanced VideoMAE model with comprehensive validation metrics.
    """

    def __init__(self, pred_shape, size, lr, scheduler=None, wandb_logger=None, freeze=False, valid_mask_gt=None):
        super(VideoMaeModelV2, self).__init__()

        self.save_hyperparameters(ignore=['wandb_logger', 'valid_mask_gt'])
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.valid_mask_gt = valid_mask_gt  # Store ground truth for metrics
        self.IGNORE_INDEX = 127

        # Loss functions
        self.loss_func1 = smp.losses.DiceLoss(mode='binary', ignore_index=self.IGNORE_INDEX)
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25, ignore_index=self.IGNORE_INDEX)
        self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)

        # VideoMAE configuration
        videomae_config = VideoMAEConfig(
            image_size=64,
            patch_size=8,
            num_channels=1,
            num_frames=24,
            tubelet_size=2,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=1536,

            decoder_num_hidden_layers=4,
            decoder_hidden_size=512,
            decoder_num_attention_heads=8,
            decoder_intermediate_size=768,

            norm_pix_loss=True,
        )

        self.encoder = VideoMAEModel(videomae_config)

        # Load pretrained weights if available
        try:
            ckpt = torch.load(
                "checkpoints/videomae_epoch=063_val_loss=0.3684.ckpt",
                map_location="cpu",
                weights_only=False
            )
            state_dict = ckpt["state_dict"]

            # Extract only the encoder weights from the Lightning checkpoint
            encoder_state = {}
            for k, v in state_dict.items():
                if k.startswith("videomae.videomae"):
                    new_k = k.replace("videomae.videomae.", "")
                    encoder_state[new_k] = v

            # Load into your fresh VideoMAEModel
            self.encoder.load_state_dict(encoder_state, strict=False)
            print("✅ VideoMAE checkpoint loaded successfully!")

        except Exception as e:
            print("⚠️ No pretrained checkpoint found, training from scratch")
            print(f"   Error: {e}")

        # Remove classifier head, keep norm
        self.encoder.head = nn.Identity()

        embed_dim = 768
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, (self.hparams.size // 8) ** 2),
        )

    def forward(self, x):
        features = self.encoder(x)  # BaseModelOutput
        tokens = features.last_hidden_state  # (B, N, D)

        # Mean pooling over patches
        B, N, D = tokens.shape
        x_flat = tokens.mean(dim=1)  # (B, D)

        # Pass to classifier
        out = self.classifier(x_flat)
        out = out.view(-1, 1, self.hparams.size // 8, self.hparams.size // 8)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)

        # Compute individual losses
        dice_loss = self.loss_func1(outputs, y)
        bce_loss = self.loss_func2(outputs, y)
        total_loss = 0.5 * dice_loss + 0.5 * bce_loss

        if torch.isnan(total_loss):
            print("⚠️ Loss nan encountered")

        # Log metrics
        self.log("train/total_loss", total_loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/dice_loss", dice_loss.item(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/bce_loss", bce_loss.item(), on_step=True, on_epoch=True, sync_dist=True)

        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        x, y, xyxys = batch

        outputs = self(x)

        # Compute individual losses
        dice_loss = self.loss_func1(outputs, y)
        bce_loss = self.loss_func2(outputs, y)
        total_loss = 0.5 * dice_loss + 0.5 * bce_loss

        y_preds = torch.sigmoid(outputs).to('cpu')

        # Accumulate predictions
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            upsampled = F.interpolate(
                y_preds[i].unsqueeze(0).float(),
                scale_factor=8,
                mode='bilinear'
            ).squeeze(0).squeeze(0).numpy()

            self.mask_pred[y1:y2, x1:x2] += upsampled
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        # Log losses
        self.log("val/total_loss", total_loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/dice_loss", dice_loss.item(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/bce_loss", bce_loss.item(), on_step=True, on_epoch=True, sync_dist=True)

        return {"loss": total_loss}

    def on_validation_epoch_end(self):
        """Compute comprehensive metrics at end of validation epoch"""
        mask_pred_tensor = torch.tensor(self.mask_pred, dtype=torch.float32, device=self.device)
        mask_count_tensor = torch.tensor(self.mask_count, dtype=torch.float32, device=self.device)

        # Synchronize across GPUs if using DDP
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(mask_pred_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mask_count_tensor, op=dist.ReduceOp.SUM)

        if self.trainer.is_global_zero:
            mask_pred_np = mask_pred_tensor.cpu().numpy()
            mask_count_np = mask_count_tensor.cpu().numpy()

            # Average predictions
            final_mask = np.divide(
                mask_pred_np,
                mask_count_np,
                out=np.zeros_like(mask_pred_np),
                where=mask_count_np != 0
            )

            # Clip to [0, 1]
            final_mask = np.clip(final_mask, 0, 1)

            # Log image to wandb
            if hasattr(self.hparams, 'wandb_logger') and self.hparams.wandb_logger is not None:
                self.hparams.wandb_logger.log_image(
                    key="masks",
                    images=[final_mask],
                    caption=["predictions"]
                )

            # Compute metrics if ground truth is available
            if self.valid_mask_gt is not None:
                metrics = self.compute_validation_metrics(final_mask, self.valid_mask_gt)

                # Log all metrics
                for metric_name, metric_value in metrics.items():
                    self.log(f"val/{metric_name}", metric_value, prog_bar=True, sync_dist=False)

                # Print summary
                print(f"\n{'='*60}")
                print(f"📊 Validation Metrics (Epoch {self.current_epoch})")
                print(f"{'='*60}")
                print(f"  Dice Score:      {metrics['dice_score']:.4f} (threshold={metrics['best_threshold']:.2f})")
                print(f"  IoU:             {metrics['iou']:.4f}")
                print(f"  F1 Score:        {metrics['f1']:.4f}")
                print(f"  Precision:       {metrics['precision']:.4f}")
                print(f"  Recall:          {metrics['recall']:.4f}")
                print(f"  F-beta (0.5):    {metrics['fbeta_0.5']:.4f}")
                print(f"  F-beta (2.0):    {metrics['fbeta_2.0']:.4f}")
                print(f"{'='*60}\n")

        # Reset accumulators
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

    def compute_validation_metrics(self, pred_mask, gt_mask):
        """
        Compute comprehensive segmentation metrics.

        Args:
            pred_mask: Predicted probability mask (H, W) in range [0, 1]
            gt_mask: Ground truth binary mask (H, W) in range [0, 255]

        Returns:
            Dictionary of metrics
        """
        # Normalize ground truth to binary
        gt_binary = (gt_mask > 127).astype(np.float32)

        # Test multiple thresholds to find the best
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_dice = 0
        best_metrics = {}

        for threshold in thresholds:
            pred_binary = (pred_mask >= threshold).astype(np.float32)

            # Compute confusion matrix elements
            tp = np.sum((pred_binary == 1) & (gt_binary == 1))
            fp = np.sum((pred_binary == 1) & (gt_binary == 0))
            fn = np.sum((pred_binary == 0) & (gt_binary == 1))
            tn = np.sum((pred_binary == 0) & (gt_binary == 0))

            # Compute metrics
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            # F-beta scores
            fbeta_0_5 = (1 + 0.5**2) * (precision * recall) / (0.5**2 * precision + recall + 1e-8)
            fbeta_2_0 = (1 + 2.0**2) * (precision * recall) / (2.0**2 * precision + recall + 1e-8)

            # Keep best metrics
            if dice > best_dice:
                best_dice = dice
                best_metrics = {
                    'dice_score': dice,
                    'iou': iou,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'fbeta_0.5': fbeta_0_5,
                    'fbeta_2.0': fbeta_2_0,
                    'best_threshold': threshold,
                    'tp': int(tp),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tn': int(tn),
                }

        return best_metrics

    def configure_optimizers(self):
        weight_decay = 0.
        base_lr = self.hparams.lr

        backbone_params = list(self.parameters())
        optimizer = AdamW(backbone_params, lr=base_lr, weight_decay=weight_decay)

        return [optimizer]


def load_weights(model, ckpt_path, strict=True, map_location='cpu'):
    """
    Loads weights from a checkpoint into the model.

    Args:
        model: An instance of VideoMaeModelV2.
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
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]


def get_scheduler(optimizer, scheduler=None, epochs=15, steps_per_epoch=10):
    if scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group['lr'] for group in optimizer.param_groups],
            total_steps=epochs * 50,
            pct_start=0.1,  # 10% warmup
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
