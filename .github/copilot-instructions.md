# Copilot / AI Agent Instructions for vesuvius_ink_detection

Purpose: give an AI coding agent the minimal, high-value context to be productive in this repo.

- Quick summary: This repo implements multiple deep-learning models for ink-detection on Vesuvius scroll fragments. Main training entry points live at the repository root and under `pretraining/`. Trained weights are stored in `checkpoints/`. Datasets and per-fragment splits live in `train_scrolls/` and outputs/predictions land under `outputs/vesuvius/`.

- Key directories & files (read first):
  - `swin_train.py`, `timesformer_hug_train.py`, `timesformer_fb_train.py`, `train_resnet3d.py` — primary training scripts (different architectures).
  - `models/` — model definitions. Example: `models/swin.py` contains the SWIN model implementation used by `swin_train.py`.
  - `pretraining/` — pretraining and self-supervised experiments (MAE / JEPA, helpers and separate checkpoints).
  - `checkpoints/` — saved checkpoints. Filenames encode model, dataset fragment, epoch and hyperparams (useful for inference/resume logic).
  - `train_scrolls/` — data shards / fragment folders (e.g., `frag5`) used during training.
  - `outputs/vesuvius/` — prediction outputs, submission artifacts and evaluation outputs.
  - `pseudo/`, `pseudo_*/` — pseudo-labeling related code and outputs.
  - `notebooks/` — exploratory analysis (clustering/distribution/helper notebooks).

- High-level architecture / data flow (how code typically moves data):
  1. Data is prepared/organized under `train_scrolls/<fragment>/` and referenced by training scripts.
  2. A training script (e.g., `python swin_train.py`) loads a model from `models/`, constructs dataloaders referencing `train_scrolls/`, and logs metrics to `wandb` (wandb usage is present in runs/links).
  3. Checkpoints are written to `checkpoints/` and evaluation/prediction artifacts are written under `outputs/vesuvius/`.
  4. Downstream scripts/notebooks load checkpoints from `checkpoints/` for inference or for pseudo-label generation in `pseudo/`.

- Practical developer workflows (concrete commands/examples):
  - Run SWIN training: `python swin_train.py` (common args are visible inline in the script; edit `cfg`/hyperparams there).
  - Run TimeSformer (HuggingFace variant): `python timesformer_hug_train.py` or `python timesformer_huggingface.py`.
  - Run ResNet3D training: `python train_resnet3d.py`.
  - Pretraining tasks: see `pretraining/download.sh` and `pretraining/*_pretraining_64.py` for MAE/Jepa flows.
  - If you hit CUDA fragmentation/OOMs (common when resuming large runs): consider setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment (observed in runtime logs) or lowering batch size/sequence length.

- Project conventions and patterns an agent should follow:
  - Script-first: CLI args are minimal — most experiments are configured by editing a `cfg` object or top-of-file variables in the training scripts. Look at the top of `swin_train.py` to find experiment defaults.
  - Checkpoint naming encodes metadata: filenames include fragment ids, timestamp, `in_chans`, `size`, `lr`, `epoch` and valid/test markers — use these names when searching for a matching checkpoint.
  - Multiple parallel experiments: Many directories contain per-run timestamps (see `train_scrolls/` and `wandb/` run directories); be careful when auto-detecting "latest" checkpoint.
  - Models live under `models/` and typically expose a constructor used by the training scripts. When modifying model signatures, update corresponding training scripts that import them.
  - WandB is used for experiment tracking — the code expects `wandb` to be available; check `swin_train.py` or other trainers for initialization details.

- Files to inspect when making changes:
  - `swin_train.py` — example of how the SWIN model is instantiated and trained.
  - `models/swin.py` — model details (backbone and any project-specific modifications).
  - `utils.py` — shared helpers used across training scripts.
  - `pseudo.py` and `pseudolabeling.ipynb` — examples of how pseudo-labels are generated and applied.

- Integration & external dependencies:
  - `wandb` for logging/experiment tracking — look for `wandb.init()` in training scripts.
  - PyTorch for model/training code (GPU usage expected).
  - Pretrained weights are present in `checkpoints/` and are loaded by name — maintain consistent checkpoint path resolution.

- Safe edits & common pitfalls for an AI agent:
  - When changing model I/O (number of input channels or expected size), update all training scripts that instantiate the model and any checkpoint-loading logic.
  - Avoid changing checkpoint filenames or formats without updating code that parses them (scripts often rely on encoded hyperparams in filenames).
  - Keep changes minimal and local: prefer small focused edits (e.g., add an argument to a single train script) rather than sweeping repo-wide changes.

- Example small tasks an agent can safely do:
  - Add a new CLI argument to `swin_train.py` to control `in_chans` and wire it into model construction.
  - Add a small helper in `utils.py` to standardize checkpoint path resolution and use it in a single train script.
  - Add a notebook under `notebooks/` demonstrating how to load a checkpoint and run inference over `train_scrolls/frag5/`.

- What to ask a human before larger changes:
  - Which fragment(s) should be considered canonical for experiments (the repo historically used `frag5`).
  - Preferred conventions for adding new checkpoints (naming pattern) to keep compatibility with inference tools.
  - Whether `wandb` logging should be mandatory or optional for scripted runs.

If anything here is unclear or you'd like the agent to include more detail about specific scripts (for example, explicit arg/flag names from `swin_train.py` or `timesformer_hug_train.py`), tell me which file to inspect and I'll incorporate exact examples.
