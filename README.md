# Skin lesion ViT training (PAD-UFES + HAM10000)

This repository organizes PAD-UFES and HAM10000 images into shared class folders, optionally **balances** class counts on disk with the [Augmentor](https://augmentor.readthedocs.io/) library, and trains a **ViT-Base-Patch32-384** classifier (`google/vit-base-patch32-384`).

## Layout

- **`data/pad/by_class/<class>/`** — PAD-UFES images (canonical names: `akiec`, `bcc`, `mel`, `nv`, `scc`, `sek`).
- **`data/ham/by_class/<class>/`** — HAM10000 images (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
- **`data/balanced/<mode>/by_class/<class>/`** — optional **balanced** copies + Augmentor-generated images (see below).

Modes for training and balancing:

| Mode        | Classes used |
|------------|----------------|
| `pad`      | PAD classes only |
| `ham`      | HAM classes only |
| `combined` | Union of both label sets; per-class images can come from PAD, HAM, or both |

## Prerequisites

- Python 3.10+ recommended.
- PyTorch + torchvision (install for your CUDA/CPU setup).
- Other Python deps:

```bash
pip install -r requirements-training.txt
```

`requirements-training.txt` includes **Augmentor** for the balancing script and **transformers**, **scikit-learn**, etc. for training.

## 1. Organize raw images into `by_class`

From the repo root:

```bash
python organize_datasets_by_class.py
```

This reads `data/pad/metadata.csv` and `data/ham/HAM10000_metadata.csv` and fills `data/*/by_class/`. It also writes `data/class_distribution.json`.

## 2. Balance classes with Augmentor (recommended before training)

Training **defaults** to reading **`data/balanced/<mode>/by_class/`**. Generate that layout with:

```bash
python scripts/balance_classes_augmentor.py --mode pad --overwrite
python scripts/balance_classes_augmentor.py --mode ham --overwrite
python scripts/balance_classes_augmentor.py --mode combined --overwrite
```

### What the script does

1. **Copies** originals from the raw `by_class` trees into `data/balanced/<mode>/by_class/<class>/` (originals are not modified).
2. For **combined**, filenames are prefixed with `pad_` or `ham_` when needed to avoid collisions.
3. Computes a **target count per class** (see `--target`).
4. For each class with fewer images than the target, runs an **Augmentor** pipeline (rotation, flips, brightness/contrast jitter) and **`sample()`**s the shortfall. Augmented files are moved out of Augmentor’s `output/` subfolder into the class folder so training can glob a single directory per class.

### CLI options (`balance_classes_augmentor.py`)

| Option | Description |
|--------|-------------|
| `--mode` | Required. `pad`, `ham`, or `combined`. |
| `--overwrite` | Delete `data/balanced/<mode>/` (or `--output`) before writing. |
| `--target` | `max` (default): match the largest class count among classes that have at least one image. `mean`: round mean of those counts. Or a **positive integer** for a fixed per-class target. |
| `--output` | Optional custom output root (default: `data/balanced/<mode>`). |
| `--seed` | RNG seed (default `42`) for reproducibility of Augmentor sampling. |

Classes with **zero** seed images are skipped for augmentation (nothing to augment).

### When to re-run

Re-run with `--overwrite` after adding or removing raw images, or when you want a different `--target`.

## 3. Train ViT (`train_vit_skin.py`)

```bash
python scripts/train_vit_skin.py --mode combined --epochs 25 --batch_size 16
```

### Data layout: `balanced` vs `raw`

| `--data_layout` | Source paths | When to use |
|-----------------|--------------|-------------|
| **`balanced`** (default) | `data/balanced/<mode>/by_class/` | After running `balance_classes_augmentor.py` for that `mode`. |
| `raw` | `data/pad/by_class` and/or `data/ham/by_class` | Skip disk balancing; use loss reweighting or sampling instead. |

If `balanced` is selected but `data/balanced/<mode>/by_class/` is missing, training exits with a short message pointing to the balancing script.

### Imbalance handling during training: `balance_strategy`

Separate from **disk** balancing, the trainer can reweight or resample **within the training split**:

| `--balance_strategy` | Behavior |
|----------------------|----------|
| **`none`** (default) | Standard cross-entropy. Intended for **class-balanced** folders (`--data_layout balanced`). |
| `loss_weights` | Inverse-frequency **class weights** in `CrossEntropyLoss`. Typical for **`--data_layout raw`** when classes are skewed. |
| `weighted_sampler` | `WeightedRandomSampler` on the training set. Often pair with `--no_class_weights_with_sampler` to avoid double compensation. |

**Recommended combinations**

- **Default pipeline:** build balanced data, then train with `--data_layout balanced` and `--balance_strategy none`.
- **Raw unbalanced folders:** `--data_layout raw --balance_strategy loss_weights` (and omit or ignore balanced data).

If you use `balanced` data **and** `loss_weights`, the script prints a note that loss weighting is usually optional.

### Other useful flags

- `--amp` — CUDA mixed precision.
- `--image_size` — default `384` (matches the ViT checkpoint).
- `--epochs`, `--lr`, `--batch_size`, `--num_workers`, etc.

Artifacts go to **`runs/vit_<mode>/`** (`best_model.pt`, `run_config.json`).

## File reference

| Path | Role |
|------|------|
| `organize_datasets_by_class.py` | Build `data/*/by_class` from CSV metadata. |
| `scripts/balance_classes_augmentor.py` | Augmentor-based class balancing into `data/balanced/`. |
| `scripts/train_vit_skin.py` | ViT training. |
| `requirements-training.txt` | Pip requirements for balancing + training. |
