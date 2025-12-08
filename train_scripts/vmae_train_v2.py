import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import cv2
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image
import pandas as pd
from datetime import datetime
from pathlib import Path

PIL.Image.MAX_IMAGE_PIXELS = 933120000

import utils
from models import vmae_v2 as vmae

class CFG:
    # ============== comp exp name =============
    current_dir = './'
    segment_path = './train_scrolls/'

    start_idx = 15
    in_chans = 30
    valid_chans = 24

    size = 64
    tile_size = 256
    stride = tile_size // 16

    train_batch_size = 50
    valid_batch_size = 100
    lr = 5e-5
    # ============== model cfg =============
    scheduler = 'cosine'
    epochs = 40

    # Change the size of fragments2
    frags_ratio1 = ['frag','re']
    frags_ratio2 = ['s4','202','left']
    ratio1 = 2
    ratio2 = 1

    # ============== fold =============
    segments = ['remaining5','rect5']#,'frag4','frag3','frag2','frag1']
    valid_id = 'rect5'#20231210132040'20231215151901
    norm = False
    aug = None
    # ============== fixed =============
    min_lr = 1e-7
    weight_decay = 1e-6
    max_grad_norm = 1
    num_workers = 8
    warmup_factor = 10

    seed = 0

    # ============== comp exp name =============
    comp_name = 'vesuvius'
    exp_name = 'videomae_v2'

    outputs_path = f'./outputs/{exp_name}/'
    model_dir = outputs_path

    # ============== CSV logging =============
    log_metrics = True  # Enable CSV logging
    csv_log_path = f'./outputs/{exp_name}/training_metrics.csv'

    # ============== augmentation =============
    train_aug_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.1,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.CoarseDropout(max_holes=5, max_width=int(size * 0.1), max_height=int(size * 0.2),
                        mask_fill_value=0, p=0.5),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        ToTensorV2(transpose_mask=True),
    ]


class MetricsCallback(pl.Callback):
    """Callback to log metrics to CSV file"""

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []

        # Initialize CSV with headers if it doesn't exist
        if not self.csv_path.exists():
            df = pd.DataFrame(columns=[
                'timestamp', 'epoch', 'step',
                'train_loss', 'train_dice_loss', 'train_bce_loss', 'train_lr',
                'val_loss', 'val_dice_loss', 'val_bce_loss',
                'val_dice_score', 'val_iou', 'val_precision', 'val_recall', 'val_f1',
                'val_best_threshold', 'val_fbeta_0.5', 'val_fbeta_2.0'
            ])
            df.to_csv(self.csv_path, index=False)

    def on_train_epoch_end(self, trainer, pl_module):
        """Save metrics at end of training epoch"""
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Prepare metrics dict
        metrics_dict = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': trainer.global_step,
            'train_loss': self._get_metric(metrics, 'train/total_loss_epoch'),
            'train_dice_loss': self._get_metric(metrics, 'train/dice_loss_epoch'),
            'train_bce_loss': self._get_metric(metrics, 'train/bce_loss_epoch'),
            'train_lr': self._get_metric(metrics, 'train/lr'),
        }

        self.metrics_history.append(metrics_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Save metrics at end of validation epoch"""
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics

        if len(self.metrics_history) > 0:
            # Update the last metrics dict with validation metrics
            self.metrics_history[-1].update({
                'val_loss': self._get_metric(metrics, 'val/total_loss'),
                'val_dice_loss': self._get_metric(metrics, 'val/dice_loss'),
                'val_bce_loss': self._get_metric(metrics, 'val/bce_loss'),
                'val_dice_score': self._get_metric(metrics, 'val/dice_score'),
                'val_iou': self._get_metric(metrics, 'val/iou'),
                'val_precision': self._get_metric(metrics, 'val/precision'),
                'val_recall': self._get_metric(metrics, 'val/recall'),
                'val_f1': self._get_metric(metrics, 'val/f1'),
                'val_best_threshold': self._get_metric(metrics, 'val/best_threshold'),
                'val_fbeta_0.5': self._get_metric(metrics, 'val/fbeta_0.5'),
                'val_fbeta_2.0': self._get_metric(metrics, 'val/fbeta_2.0'),
            })

            # Append to CSV
            df = pd.DataFrame([self.metrics_history[-1]])
            df.to_csv(self.csv_path, mode='a', header=False, index=False)

            print(f"\n✅ Metrics saved to {self.csv_path}")

    def _get_metric(self, metrics, key):
        """Safely get metric value"""
        if key in metrics:
            val = metrics[key]
            if torch.is_tensor(val):
                return val.item()
            return val
        return None




def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug


if __name__ == '__main__':
    # End any existing run (if still active)
    if wandb.run is not None:
        wandb.finish()

    utils.cfg_init(CFG)
    torch.set_float32_matmul_precision('medium')

    fragment_id = CFG.valid_id
    run_slug = f'VIDEOMAE_{CFG.segments}_valid={CFG.valid_id}_size={CFG.size}_lr={CFG.lr}_in_chans={CFG.valid_chans}_norm={CFG.norm}_fourth={CFG.aug}'

    # Read mask and resize to match the output resolution
    valid_mask_gt = cv2.imread(f"{CFG.segment_path}{fragment_id}/{fragment_id}_inklabels.png", 0)
    if any(sub in fragment_id for sub in CFG.frags_ratio1):
        scale = 1 / CFG.ratio1
        new_w = int(valid_mask_gt.shape[1] * scale)
        new_h = int(valid_mask_gt.shape[0] * scale)
        valid_mask_gt = cv2.resize(valid_mask_gt, (new_w, new_h), interpolation=cv2.INTER_AREA)

    elif any(sub in fragment_id for sub in CFG.frags_ratio2):
        scale = 1 / CFG.ratio2
        new_w = int(valid_mask_gt.shape[1] * scale)
        new_h = int(valid_mask_gt.shape[0] * scale)
        valid_mask_gt = cv2.resize(valid_mask_gt, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad0 = (CFG.size - valid_mask_gt.shape[0] % CFG.size) % CFG.size
    pad1 = (CFG.size - valid_mask_gt.shape[1] % CFG.size) % CFG.size
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    pred_shape = valid_mask_gt.shape

    train_images, train_masks, valid_images, valid_masks, valid_xyxys = utils.get_train_valid_dataset(CFG)

    print('train_images', train_images[0].shape)
    print("Length of train images:", len(train_images))

    valid_xyxys = np.stack(valid_xyxys)
    train_dataset = utils.VideoDataset(
        train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG),
        norm=True, aug='fourth', scale_factor=8)
    valid_dataset = utils.VideoDataset(
        valid_images, CFG, xyxys=valid_xyxys, labels=valid_masks,
        transform=get_transforms(data='valid', cfg=CFG), norm=True)

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

    wandb_logger = WandbLogger(project="vesuvius_ink_detection", name=run_slug)
    model = vmae.VideoMaeModelV2(
        pred_shape=pred_shape,
        size=CFG.size,
        lr=CFG.lr,
        scheduler=CFG.scheduler,
        wandb_logger=wandb_logger,
        freeze=False,
        valid_mask_gt=valid_mask_gt  # Pass ground truth for metrics computation
    )
    wandb_logger.watch(model, log="all", log_freq=50)

    # Setup callbacks
    callbacks = []

    # Metrics logging callback
    if CFG.log_metrics:
        metrics_callback = MetricsCallback(csv_path=CFG.csv_log_path)
        callbacks.append(metrics_callback)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filename=f'{run_slug}_' + 'epoch={epoch:02d}_val_loss={val/total_loss:.4f}_dice={val/dice_score:.4f}',
        dirpath=CFG.model_dir,
        monitor='val/dice_score',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=CFG.epochs,
        accelerator="gpu",
        check_val_every_n_epoch=1,  # Validate every epoch for better tracking
        devices=-1,
        logger=wandb_logger,
        default_root_dir="./models",
        accumulate_grad_batches=1,
        precision='16',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='ddp',
        callbacks=callbacks,
    )

    print(f"\n{'='*60}")
    print(f"🚀 Starting training: {run_slug}")
    print(f"📊 Metrics will be saved to: {CFG.csv_log_path}")
    print(f"💾 Checkpoints will be saved to: {CFG.model_dir}")
    print(f"{'='*60}\n")

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"📊 Final metrics saved to: {CFG.csv_log_path}")
    print(f"{'='*60}\n")

    wandb.finish()
