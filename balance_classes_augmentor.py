#!/usr/bin/env python3
"""
Balance class folders by copying originals into data/balanced/<mode>/by_class/<class>/
and generating extra images with Augmentor until each class reaches a target count.

Example:
  python scripts/balance_classes_augmentor.py --mode ham --overwrite
  python scripts/balance_classes_augmentor.py --mode combined --target max
  python scripts/balance_classes_augmentor.py --mode pad --target 500
"""
from __future__ import annotations

import argparse
import random
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
PAD_BY_CLASS = DATA / "pad" / "by_class"
HAM_BY_CLASS = DATA / "ham" / "by_class"

PAD_CLASSES = ["akiec", "bcc", "mel", "nv", "scc", "sek"]
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
COMBINED_CLASSES = sorted(set(PAD_CLASSES) | set(HAM_CLASSES))

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def list_images(d: Path) -> list[Path]:
    """List image files in ``d`` (non-recursive). Dedupe by resolved path (Windows is case-insensitive)."""
    if not d.is_dir():
        return []
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in IMAGE_GLOBS:
        for p in d.glob(pat):
            if not p.is_file():
                continue
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return sorted(out, key=lambda x: x.name.lower())


def class_names_for_mode(mode: str) -> list[str]:
    if mode == "pad":
        return PAD_CLASSES
    if mode == "ham":
        return HAM_CLASSES
    if mode == "combined":
        return COMBINED_CLASSES
    raise ValueError(f"Unknown mode {mode!r}")


def collect_raw_sources(
    mode: str,
) -> dict[str, list[Path]] | dict[str, list[tuple[Path, str]]]:
    """Per class: list of paths (pad/ham) or list of (path, 'pad'|'ham') for combined."""
    class_names = class_names_for_mode(mode)

    if mode == "combined":
        by_class: dict[str, list[tuple[Path, str]]] = {c: [] for c in class_names}
        for cls in class_names:
            for p in list_images(PAD_BY_CLASS / cls):
                by_class[cls].append((p, "pad"))
            for p in list_images(HAM_BY_CLASS / cls):
                by_class[cls].append((p, "ham"))
        return by_class

    root = PAD_BY_CLASS if mode == "pad" else HAM_BY_CLASS
    by_path: dict[str, list[Path]] = {c: [] for c in class_names}
    for cls in class_names:
        by_path[cls].extend(list_images(root / cls))
    return by_path


def parse_target(s: str, counts: dict[str, int]) -> int:
    key = s.strip().lower()
    nonzero = [c for c in counts.values() if c > 0]
    if not nonzero:
        return 0
    if key == "max":
        return max(nonzero)
    if key == "mean":
        return max(1, int(round(statistics.mean(nonzero))))
    try:
        n = int(s, 10)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"target must be 'max', 'mean', or a positive integer, got {s!r}"
        ) from e
    if n < 1:
        raise argparse.ArgumentTypeError("integer target must be >= 1")
    return n


def build_augmentor_pipeline(class_dir: Path):
    import Augmentor

    p = Augmentor.Pipeline(str(class_dir))
    p.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=20)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.4)
    p.random_brightness(probability=0.5, min_factor=0.85, max_factor=1.15)
    p.random_contrast(probability=0.5, min_factor=0.85, max_factor=1.15)
    return p


def move_output_into_class_dir(class_dir: Path) -> int:
    """Move Augmentor's class_dir/output/* into class_dir; remove output. Returns files moved."""
    out_sub = class_dir / "output"
    if not out_sub.is_dir():
        return 0
    n = 0
    for f in sorted(out_sub.iterdir()):
        if not f.is_file():
            continue
        dest = class_dir / f.name
        if dest.exists():
            stem, suf = f.stem, f.suffix
            k = 1
            while dest.exists():
                dest = class_dir / f"{stem}_{k}{suf}"
                k += 1
        shutil.move(str(f), dest)
        n += 1
    try:
        out_sub.rmdir()
    except OSError:
        pass
    return n


def copy_sources_to_balanced(
    mode: str,
    dest_by_class: Path,
    sources: dict[str, list[Path]] | dict[str, list[tuple[Path, str]]],
) -> None:
    dest_by_class.mkdir(parents=True, exist_ok=True)
    for cls, items in sources.items():
        d = dest_by_class / cls
        d.mkdir(parents=True, exist_ok=True)
        if mode == "combined":
            for src, tag in items:  # type: ignore[misc]
                dest_name = f"{tag}_{src.name}"
                dest = d / dest_name
                if dest.exists():
                    stem, suf = Path(dest_name).stem, Path(dest_name).suffix
                    k = 1
                    while dest.exists():
                        dest = d / f"{stem}_{k}{suf}"
                        k += 1
                shutil.copy2(src, dest)
        else:
            for src in items:  # type: ignore[assignment]
                dest = d / src.name
                if dest.exists():
                    stem, suf = src.stem, src.suffix
                    k = 1
                    while dest.exists():
                        dest = d / f"{stem}_{k}{suf}"
                        k += 1
                shutil.copy2(src, dest)


def run_balance(args: argparse.Namespace) -> None:
    try:
        import Augmentor  # noqa: F401
    except ImportError:
        print("Install Augmentor: pip install Augmentor", file=sys.stderr)
        raise SystemExit(1) from None

    random.seed(args.seed)
    np.random.seed(args.seed)

    sources = collect_raw_sources(args.mode)
    counts = {c: len(sources[c]) for c in sources}
    for cls, n in sorted(counts.items()):
        print(f"  raw {cls}: {n} images")

    target = parse_target(args.target, counts)
    if target <= 0:
        print("No images found; nothing to balance.", file=sys.stderr)
        raise SystemExit(1)

    out_root = Path(args.output).resolve() if args.output else (DATA / "balanced" / args.mode)
    by_class_dest = out_root / "by_class"

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    elif by_class_dest.exists() and any(by_class_dest.iterdir()) and not args.overwrite:
        print(
            f"Output already exists: {out_root}\n"
            "Use --overwrite to replace it.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    copy_sources_to_balanced(args.mode, by_class_dest, sources)

    summary: dict[str, dict[str, int]] = {}
    for cls in class_names_for_mode(args.mode):
        class_dir = by_class_dest / cls
        if not class_dir.is_dir():
            summary[cls] = {"before": 0, "added": 0, "after": 0}
            continue
        n = len(list_images(class_dir))
        need = max(0, target - n)
        added = 0
        if need > 0:
            if n == 0:
                print(f"  skip augment {cls}: no seed images", file=sys.stderr)
            else:
                p = build_augmentor_pipeline(class_dir)
                p.sample(need)
                added = move_output_into_class_dir(class_dir)
                if added < need:
                    print(
                        f"  warning {cls}: requested {need} augmentations, moved {added} files",
                        file=sys.stderr,
                    )
        n_after = len(list_images(class_dir))
        summary[cls] = {"before": n, "added": added, "after": n_after}

    print(f"\nBalanced dataset written to: {out_root}")
    print(f"Target count per class (where seeds exist): {target}")
    for cls, s in sorted(summary.items()):
        print(f"  {cls}: {s['before']} + {s['added']} -> {s['after']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["combined", "pad", "ham"], required=True)
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output root (default: data/balanced/<mode>)",
    )
    p.add_argument(
        "--target",
        type=str,
        default="max",
        help="Per-class target count: max (largest class), mean (round mean of non-empty), or a positive integer",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output root for this run before writing",
    )
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_balance(args)


if __name__ == "__main__":
    main()
