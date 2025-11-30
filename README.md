# Vesuvius Ink Detection

Deep learning models for detecting ink on ancient Vesuvius scroll fragments using multi-channel volumetric CT scan data.

## Overview

This project implements state-of-the-art computer vision models to identify ink writing on 2,000-year-old papyrus scrolls from Herculaneum (destroyed by Mount Vesuvius in 79 AD). The models process 3D volumetric data from CT scans, treating stacked depth layers as temporal sequences to perform semantic segmentation of ink locations.

### Key Features

- **Multiple Model Architectures**: SWIN Transformer, VideoMAE, TimeSformer, 3D ResNet, I3D, UNETR
- **Self-Supervised Pretraining**: MAE, VideoMAE, JEPA for learning from unlabeled scroll data
- **Multi-Fragment Training**: Combines data from multiple scroll fragments for robust learning
- **Advanced Augmentation**: Spatial, intensity, and channel-level augmentations
- **Experiment Tracking**: Full integration with Weights & Biases

## Project Structure

```
vesuvius_ink_detection/
├── models/                      # Model architectures
│   ├── swin.py                 # SWIN Transformer (primary model)
│   ├── vmae.py                 # VideoMAE
│   ├── timesformer_hug.py      # TimeSformer (HuggingFace)
│   ├── resnetall.py            # 3D ResNet variants
│   ├── i3dallnl.py             # I3D with non-local blocks
│   └── unetr.py                # UNETR segmentation
│
├── pretraining/                 # Self-supervised pretraining
│   ├── mae.py                  # VideoMAE pretraining
│   ├── jepa.py                 # Joint Embedding Predictive Architecture
│   ├── mae_swin.py             # MAE for SWIN
│   ├── prepare_data.py         # Tile extraction for pretraining
│   └── training_history_*.csv  # Training logs
│
├── train_scripts/               # Training utilities
│   ├── vmae_train.py           # VideoMAE training wrapper
│   └── utils.py                # Helper functions
│
├── Training Scripts (root):
│   ├── swin_train.py           # SWIN Transformer training
│   ├── timesformer_hug_train.py # TimeSformer training
│   ├── train_resnet3d.py       # 3D ResNet training
│   ├── z_cv.py                 # Cross-validation experiments
│   └── utils.py                # Shared utilities
│
├── train_scrolls/               # Training data (per-fragment)
│   ├── frag5/                  # Fragment 5 (primary)
│   │   ├── layers/             # CT scan layers (22.tif, 23.tif, ...)
│   │   ├── frag5_inklabels.png # Ground truth annotations
│   │   └── frag5_mask.png      # Fragment boundary mask
│   └── [other fragments]
│
├── checkpoints/                 # Saved model weights
├── outputs/                     # Predictions and results
├── notebooks/                   # Exploratory analysis
└── pseudo/                      # Pseudo-labeling code
```

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- PyTorch with CUDA support

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd vesuvius_ink_detection

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch, torchvision, torchaudio
pytorch-lightning
wandb
opencv-python
numpy
Pillow
segmentation-models-pytorch
albumentations
warmup-scheduler
scikit-image
einops
transformers
pandas
```

## Data Organization

Each fragment directory follows this structure:

```
fragment_id/
├── layers/                      # Volumetric CT scan data
│   ├── 22.tif                  # Individual depth layers
│   ├── 23.tif
│   └── ... (16-54 layers)
├── {fragment_id}_inklabels.png # Ground truth ink labels (binary mask)
└── {fragment_id}_mask.png      # Fragment boundary mask
```

### Available Fragments

- `frag5` - Primary training fragment
- `frag1`, `frag2`, `frag6` - Additional fragments
- `s4`, `rect5`, `remaining5` - Segment variants
- `20240304141530`, `20231215151901` - Timestamped fragments

## Quick Start

### Training SWIN Transformer (Recommended)

```bash
python swin_train.py
```

Configuration is controlled via the `CFG` class at the top of the file:

```python
class CFG:
    # Data settings
    segments = ['frag5', 's4']      # Training fragments
    valid_id = 'frag5'               # Validation fragment

    # Input configuration
    start_idx = 22                   # First layer index
    in_chans = 18                    # Number of input channels
    valid_chans = 16                 # Channels for validation
    size = 224                       # Spatial resolution

    # Training hyperparameters
    train_batch_size = 5
    valid_batch_size = 10
    lr = 5e-5
    epochs = 4
    scheduler = 'cosine'
```

### Other Model Training

```bash
# TimeSformer
python timesformer_hug_train.py

# 3D ResNet
python train_resnet3d.py

# Cross-validation experiments
python z_cv.py
```

## Model Architectures

### 1. SWIN Transformer ([models/swin.py](models/swin.py))

**Primary model** - Shifted Window Vision Transformer adapted for volumetric data.

- **Input**: 224×224 spatial, 16-24 depth channels
- **Output**: Binary segmentation mask (ink vs. no-ink)
- **Features**:
  - Hierarchical shifted window attention
  - Variable input channels (8-54)
  - Combined loss: DiceLoss + SoftBCEWithLogitsLoss
- **Training**: [swin_train.py](swin_train.py)

### 2. VideoMAE ([models/vmae.py](models/vmae.py))

Video Masked Autoencoder for self-supervised pretraining.

- **Input**: 64×64 or 224×224, 16-24 frames
- **Pretraining**: 75-90% mask ratio, pixel reconstruction
- **Fine-tuning**: Linear classifier head
- **Training**: [pretraining/mae.py](pretraining/mae.py)

### 3. TimeSformer ([models/timesformer_hug.py](models/timesformer_hug.py))

Transformer designed for video/temporal understanding.

- **Variants**: HuggingFace and Facebook implementations
- **Features**: Divided space-time attention
- **Training**: [timesformer_hug_train.py](timesformer_hug_train.py)

### 4. 3D ResNet ([models/resnetall.py](models/resnetall.py))

ResNet extended to 3D convolutions.

- **Depths**: 10, 18, 34, 50, 101, 152, 200 layers
- **Pretrained**: r3d101_KM_200ep.pth (Kinetics-400)
- **Training**: [train_resnet3d.py](train_resnet3d.py)

### 5. I3D with Non-local Blocks ([models/i3dallnl.py](models/i3dallnl.py))

Inception 3D with non-local attention operations.

### 6. UNETR ([models/unetr.py](models/unetr.py))

Transformer-based U-Net for hierarchical segmentation.

## Self-Supervised Pretraining

Pretraining on unlabeled scroll data improves downstream performance.

### Masked Autoencoder (MAE)

```bash
cd pretraining
python mae.py
```

- **Method**: Mask 75-90% of patches, reconstruct pixel values
- **Configuration**: 16-channel input, 16-24 frames
- **Loss**: L1 norm pixel prediction
- **Checkpoints**: `videomae_epoch=063_val_loss=0.3684.ckpt`

### Joint Embedding Predictive Architecture (JEPA)

```bash
cd pretraining
python jepa.py
```

- **Method**: Predict representations of masked regions (no pixel targets)
- **Advantage**: Learns higher-level semantic features
- **Configuration**: 64×64 input, 16 frames

### Data Preparation for Pretraining

```bash
cd pretraining
python prepare_data.py
```

Extracts 64×64 tiles with 50% stride from full scroll volumes.

## Training Infrastructure

### PyTorch Lightning Framework

All training scripts use PyTorch Lightning with:

- **Distributed Training**: DDP (multi-GPU)
- **Mixed Precision**: FP16 for memory efficiency
- **Gradient Clipping**: Max norm 1.0
- **Learning Rate Scheduling**: Cosine with warmup
- **Checkpointing**: Automatic saves with encoded metadata

### Weights & Biases Integration

Experiment tracking and hyperparameter logging:

```python
wandb.init(project='vesuvius', name='experiment_name')
```

View runs at: wandb.ai (requires login)

### Data Augmentation

Albumentations pipeline:

```python
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.5)
A.RandomBrightnessContrast(p=0.75)
A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.15, scale_limit=0.1, p=0.75)
A.OneOf([A.GaussNoise(), A.GaussianBlur(), A.MotionBlur()], p=0.4)
A.CoarseDropout(max_holes=5, max_width=int(size*0.1), max_height=int(size*0.2))
```

**Channel-level augmentation** ("fourth"): Random channel shuffling/removal for robustness.

## Checkpoint Naming Convention

Checkpoints encode complete hyperparameter information:

```
{MODEL}_{FRAGMENTS}_valid={VALID_ID}_size={SIZE}_lr={LR}_in_chans={CHANS}_norm={NORM}_fourth={AUG}_epoch={EPOCH}.ckpt
```

**Example**:
```
SWIN_['frag5','s4']_valid=frag5_size=224_lr=2e-05_in_chans=16_norm=True_epoch=7.ckpt
```

This enables:
- Easy checkpoint identification
- Reproducible experiment tracking
- Automated checkpoint selection

## Cross-Validation

Run k-fold cross-validation across fragments:

```bash
python z_cv.py
```

Features:
- Multiple model backbones (ResNet3D, I3D, UNETR)
- Fragment-level train/validation splits
- Comprehensive evaluation metrics

## Pseudo-Labeling

Generate soft pseudo-labels for semi-supervised learning:

```bash
python pseudo.py
```

Workflow:
1. Train model on labeled fragments
2. Generate predictions on unlabeled fragments
3. Use high-confidence predictions as pseudo-labels
4. Retrain with combined labeled + pseudo-labeled data

## Advanced Configuration

### Multi-Fragment Training

Train on multiple fragments simultaneously:

```python
segments = ['frag5', 's4', 'rect5']  # Training fragments
valid_id = 'frag5'                    # Hold out for validation
```

### Variable Input Channels

Experiment with different depth ranges:

```python
start_idx = 22      # First layer to use
in_chans = 18       # Total channels (22-39 inclusive)
valid_chans = 16    # Subset for validation (center crop)
```

### Resolution Scaling

Different fragments may require different scaling:

```python
frags_ratio1 = ['frag', 're']  # Scale by ratio1
frags_ratio2 = ['s4', '202']   # Scale by ratio2
ratio1 = 2  # Divide by 2
ratio2 = 1  # No scaling
```

## Output Structure

### Model Checkpoints

```
outputs/vesuvius/pretraining_all/vesuvius-models/
└── SWIN_['frag5','s4']_valid=frag5_size=224_lr=2e-05_in_chans=16_norm=True_epoch=7.ckpt
```

### Predictions

```
outputs/vesuvius/pretraining_all/
└── predictions/
    └── [fragment_id]_prediction.png
```

## Utilities

### Core Functions ([utils.py](utils.py))

```python
# Load volumetric data with mask
read_image_mask(fragment_id, s=22, e=38, rotate=0)

# Split data by fragment
get_train_valid_dataset(segments, valid_id)

# Initialize configuration
cfg_init(CFG, mode='train')
```

### Training Utilities ([train_scripts/utils.py](train_scripts/utils.py))

Helper functions for:
- Custom data loading
- Metric computation
- Checkpoint management

## Common Issues and Solutions

### CUDA Out of Memory

```python
# Reduce batch size
train_batch_size = 3  # Instead of 5

# Reduce spatial resolution
size = 64  # Instead of 224

# Enable gradient checkpointing (in model code)
```

Or set environment variable:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Large Image Files

Increase PIL limit (already done in training scripts):

```python
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
```

### Experiment Tracking

Login to Weights & Biases:

```bash
wandb login
```

Or disable:

```python
wandb.init(mode='disabled')
```

## Best Practices

### Model Selection

1. **SWIN Transformer**: Best overall performance, recommended starting point
2. **VideoMAE**: Good for pretraining, requires fine-tuning
3. **3D ResNet**: Lightweight alternative, faster training
4. **TimeSformer**: Good temporal modeling, higher memory usage

### Training Strategy

1. **Pretrain** on unlabeled data (MAE/JEPA)
2. **Fine-tune** on labeled fragments
3. **Multi-fragment training** for robustness
4. **Cross-validation** for model selection
5. **Ensemble** multiple checkpoints for final predictions

### Hyperparameter Tuning

Recommended ranges:
- **Learning rate**: 1e-5 to 5e-5
- **Batch size**: 3-12 (depends on GPU memory)
- **Input channels**: 16-24 (sweet spot for most fragments)
- **Spatial size**: 224 (SWIN), 64 (VideoMAE pretraining)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{vesuvius-ink-detection,
  title={Vesuvius Ink Detection},
  author={[Author Names]},
  year={2024},
  howpublished={\url{https://github.com/[repository]}},
}
```

## License

[Specify license]

## Acknowledgments

- Vesuvius Challenge organizers
- PyTorch Lightning team
- HuggingFace Transformers library
- Albumentations library

## Contact

For questions or issues, please open a GitHub issue or contact [contact information].

---

**Project Status**: Active Development

**Last Updated**: 2024

**Contributors**: [List contributors]
