#!/usr/bin/env python3
"""
Train ViT-Base-Patch32-384 (ImageNet-21k / ImageNet-1k weights) on skin lesion data.

Preprocessing (training pipeline):
  1) Resize to 384x384 RGB (matches vit-base-patch32-384; override with --image_size)
  2) Scale to [0, 1]
  3) Per-channel percentile contrast stretch (defaults: 2nd–98th percentile)
  4) ImageNet normalization (mean/std) for transfer learning

Split: stratified 80% train / 20% validation. Train split uses augmentation only.

Data layout (see --data_layout):
  * balanced (default): reads data/balanced/<mode>/by_class/ produced by
    scripts/balance_classes_augmentor.py (Augmentor-based oversampling).
  * raw: reads original data/pad/by_class and data/ham/by_class (unbalanced counts).

Class imbalance during training (see --balance_strategy; use with --data_layout raw or if
you still want reweighting on top of balanced files):
  * none (default with balanced data): plain CrossEntropyLoss.
  * loss_weights: inverse-frequency weights in CrossEntropyLoss (typical for raw data).
  * weighted_sampler: WeightedRandomSampler; use with --no_class_weights_with_sampler to
    avoid double-compensation, or experiment carefully.

Runs three modes (same architecture / hyperparameters):
  --mode combined | pad | ham

Example:
  python scripts/balance_classes_augmentor.py --mode combined --overwrite
  python scripts/train_vit_skin.py --mode combined --epochs 30 --batch_size 16
  python scripts/train_vit_skin.py --mode pad --data_layout raw --balance_strategy loss_weights
  python scripts/train_vit_skin.py --mode ham
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm
from transformers import ViTModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Paths & label spaces
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
PAD_BY_CLASS = DATA / "pad" / "by_class"
HAM_BY_CLASS = DATA / "ham" / "by_class"
BALANCED_ROOT = DATA / "balanced"
RUNS_ROOT = REPO_ROOT / "runs"

# PAD / HAM canonical folder names (must match on-disk by_class subdirs)
PAD_CLASSES = ["akiec", "bcc", "mel", "nv", "scc", "sek"]
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
COMBINED_CLASSES = sorted(set(PAD_CLASSES) | set(HAM_CLASSES))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PRETRAINED_ID = "google/vit-base-patch32-384"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def contrast_stretch_tensor(
    x: torch.Tensor, p_low: float = 2.0, p_high: float = 98.0
) -> torch.Tensor:
    """x: C x H x W float in [0, 1]. Per-channel percentile stretch, clamp to [0, 1]."""
    out = x.clone()
    for c in range(out.shape[0]):
        flat = out[c].flatten()
        if flat.numel() == 0:
            continue
        lo = torch.quantile(flat, p_low / 100.0)
        hi = torch.quantile(flat, p_high / 100.0)
        denom = hi - lo
        if denom < 1e-6:
            continue
        out[c] = torch.clamp((out[c] - lo) / denom, 0.0, 1.0)
    return out


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def list_images_in_dir(d: Path) -> list[Path]:
    """Glob common image extensions in ``d`` (non-recursive). Dedupe by resolved path for case-insensitive FS."""
    if not d.is_dir():
        return []
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        for p in d.glob(pat):
            if not p.is_file():
                continue
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return sorted(out, key=lambda x: x.name.lower())


def gather_paths_and_labels(
    mode: str, data_layout: str
) -> tuple[list[Path], list[int], list[str]]:
    """Returns paths, integer labels, and ordered class names for this mode."""
    if mode == "pad":
        class_names = PAD_CLASSES
    elif mode == "ham":
        class_names = HAM_CLASSES
    elif mode == "combined":
        class_names = COMBINED_CLASSES
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    name_to_idx = {n: i for i, n in enumerate(class_names)}
    paths: list[Path] = []
    labels: list[int] = []

    if data_layout == "balanced":
        balanced_by_class = BALANCED_ROOT / mode / "by_class"
        if not balanced_by_class.is_dir():
            raise RuntimeError(
                f"Balanced data not found at {balanced_by_class}. "
                f"Run: python scripts/balance_classes_augmentor.py --mode {mode} --overwrite\n"
                "Or train on raw folders with: --data_layout raw"
            )
        for cls in class_names:
            d = balanced_by_class / cls
            if not d.is_dir():
                continue
            idx = name_to_idx[cls]
            for p in list_images_in_dir(d):
                paths.append(p)
                labels.append(idx)
    elif data_layout == "raw":
        if mode == "combined":
            for cls in class_names:
                idx = name_to_idx[cls]
                for root in (PAD_BY_CLASS, HAM_BY_CLASS):
                    d = root / cls
                    if not d.is_dir():
                        continue
                    for p in list_images_in_dir(d):
                        paths.append(p)
                        labels.append(idx)
        else:
            root = PAD_BY_CLASS if mode == "pad" else HAM_BY_CLASS
            for cls in class_names:
                d = root / cls
                if not d.is_dir():
                    continue
                idx = name_to_idx[cls]
                for p in list_images_in_dir(d):
                    paths.append(p)
                    labels.append(idx)
    else:
        raise ValueError(f"Unknown data_layout {data_layout!r}")

    if len(paths) == 0:
        hint = (
            BALANCED_ROOT / mode / "by_class"
            if data_layout == "balanced"
            else f"{PAD_BY_CLASS} / {HAM_BY_CLASS}"
        )
        raise RuntimeError(
            f"No images found for mode={mode!r} data_layout={data_layout!r}. Expected under {hint}."
        )
    return paths, labels, class_names


class SkinImageDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        augment: bool,
        size: int = 384,
        stretch_p_low: float = 2.0,
        stretch_p_high: float = 98.0,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.augment = augment
        self.size = size
        self.stretch_p_low = stretch_p_low
        self.stretch_p_high = stretch_p_high

    def __len__(self) -> int:
        return len(self.paths)

    def _geom_augment(self, img: Image.Image) -> Image.Image:
        if not self.augment:
            return img
        if random.random() < 0.5:
            img = TF.hflip(img)
        if random.random() < 0.5:
            img = TF.vflip(img)
        angle = random.uniform(-25.0, 25.0)
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        # Mild photometric jitter after resize in tensor space is applied below
        return img

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path = self.paths[i]
        y = self.labels[i]
        img = load_rgb(path)
        img = self._geom_augment(img)
        t = TF.to_tensor(TF.resize(img, [self.size, self.size]))
        t = contrast_stretch_tensor(t, self.stretch_p_low, self.stretch_p_high)
        if self.augment:
            t = TF.adjust_brightness(t, random.uniform(0.85, 1.15))
            t = TF.adjust_contrast(t, random.uniform(0.85, 1.15))
            t = TF.adjust_saturation(t, random.uniform(0.85, 1.15))
            t = torch.clamp(t, 0.0, 1.0)
        t = imagenet_normalize(t)
        return t, y


class SkinViT(nn.Module):
    """ViT backbone + 2x256-dim FCN head with dropout (categorical logits)."""

    def __init__(
        self,
        num_classes: int,
        pretrained_id: str = PRETRAINED_ID,
        dropout: float = 0.5,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = ViTModel.from_pretrained(pretrained_id)
        dim = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cfg = self.backbone.config
        exp = cfg.image_size
        h, w = pixel_values.shape[-2], pixel_values.shape[-1]
        need_interp = h != exp or w != exp
        out = self.backbone(
            pixel_values=pixel_values,
            interpolate_pos_encoding=need_interp,
        )
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    n = len(labels)
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        if cnt > 0:
            weights[c] = n / (num_classes * float(cnt))
        else:
            weights[c] = 0.0
    # Normalize so mean weight over present classes is ~1
    present = weights > 0
    if present.any():
        weights[present] = weights[present] / weights[present].mean()
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, labels, class_names = gather_paths_and_labels(
        args.mode, args.data_layout
    )
    num_classes = len(class_names)

    idx_train, idx_val = train_test_split(
        range(len(paths)),
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )
    paths_tr = [paths[i] for i in idx_train]
    labels_tr = [labels[i] for i in idx_train]
    paths_va = [paths[i] for i in idx_val]
    labels_va = [labels[i] for i in idx_val]

    train_ds = SkinImageDataset(paths_tr, labels_tr, augment=True, size=args.image_size)
    val_ds = SkinImageDataset(paths_va, labels_va, augment=False, size=args.image_size)

    sampler = None
    if args.balance_strategy == "weighted_sampler":
        counts = Counter(labels_tr)
        sample_w = [1.0 / counts[y] for y in labels_tr]
        sampler = WeightedRandomSampler(
            torch.tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SkinViT(
        num_classes=num_classes,
        pretrained_id=args.pretrained_id,
        dropout=args.dropout,
        hidden=args.head_hidden,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    class_weights = None
    if args.balance_strategy == "loss_weights" or (
        args.balance_strategy == "weighted_sampler" and not args.no_class_weights_with_sampler
    ):
        class_weights = compute_class_weights(labels_tr, num_classes).to(device)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    out_dir = RUNS_ROOT / f"vit_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "mode": args.mode,
        "classes": class_names,
        "num_train": len(paths_tr),
        "num_val": len(paths_va),
        "train_class_counts": dict(Counter(labels_tr)),
        "data_layout": args.data_layout,
        "balance_strategy": args.balance_strategy,
        "pretrained_id": args.pretrained_id,
        "image_size": args.image_size,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}"
        )
        if va_acc > best_val:
            best_val = va_acc
            ckpt = {
                "model_state": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "val_acc": va_acc,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "best_model.pt")
    print(f"Done. Best val acc={best_val:.4f}. Saved to {out_dir / 'best_model.pt'}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["combined", "pad", "ham"],
        required=True,
        help="Dataset: combined (union of classes), pad only, or ham only.",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout after each FCN block")
    p.add_argument("--head_hidden", type=int, default=256, help="FCN width (both layers)")
    p.add_argument(
        "--image_size",
        type=int,
        default=384,
        help="Resize to H=W (384 matches google/vit-base-patch32-384; other sizes use pos-interpolation)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pretrained_id", type=str, default=PRETRAINED_ID)
    p.add_argument(
        "--data_layout",
        choices=["balanced", "raw"],
        default="balanced",
        help="balanced: data/balanced/<mode>/by_class (run balance_classes_augmentor.py first). "
        "raw: original PAD/HAM by_class folders.",
    )
    p.add_argument(
        "--balance_strategy",
        choices=["loss_weights", "weighted_sampler", "none"],
        default="none",
        help="Training-time imbalance handling. Default none suits class-balanced folders; "
        "use loss_weights with --data_layout raw for skewed originals.",
    )
    p.add_argument(
        "--no_class_weights_with_sampler",
        action="store_true",
        help="If set with weighted_sampler, do not also use CE class weights (recommended).",
    )
    p.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.balance_strategy == "weighted_sampler" and not args.no_class_weights_with_sampler:
        print(
            "Note: using both weighted_sampler and class weights can over-compensate. "
            "Consider --no_class_weights_with_sampler."
        )
    if (
        args.data_layout == "balanced"
        and args.balance_strategy == "loss_weights"
    ):
        print(
            "Note: data_layout=balanced already evens class counts on disk; "
            "loss_weights is optional and may over-emphasize tiny residual skew."
        )
    run_training(args)


if __name__ == "__main__":
    main()
