"""
Organize PAD-UFES and HAM10000 images into by_class/<label>/ from metadata.
Run from repo root: python organize_datasets_by_class.py
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PAD_ROOT = DATA / "pad"
HAM_ROOT = DATA / "ham"

PAD_META = PAD_ROOT / "metadata.csv"
HAM_META = HAM_ROOT / "HAM10000_metadata.csv"

# CSV uses ACK/NEV; on-disk class dirs use HAM-aligned names akiec/nv.
PAD_RAW_TO_CANONICAL = {
    "ack": "akiec",
    "bcc": "bcc",
    "mel": "mel",
    "nev": "nv",
    "scc": "scc",
    "sek": "sek",
}
PAD_EXPECTED_RAW = frozenset(PAD_RAW_TO_CANONICAL.keys())
PAD_EXPECTED_CANONICAL = frozenset(PAD_RAW_TO_CANONICAL.values())

HAM_EXPECTED = frozenset({"bkl", "df", "vasc", "bcc", "akiec", "mel", "nv"})
STATS_OUT = DATA / "class_distribution.json"


def _folder_image_counts(dest_root: Path, pattern: str) -> dict[str, int]:
    if not dest_root.is_dir():
        return {}
    return {
        d.name: sum(1 for _ in d.glob(pattern))
        for d in sorted(dest_root.iterdir(), key=lambda p: p.name)
        if d.is_dir()
    }


def index_by_basename(root: Path, suffix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in root.rglob(f"*{suffix}"):
        if not p.is_file():
            continue
        key = p.name
        if key in out and out[key] != p:
            raise RuntimeError(f"Duplicate basename: {key}\n  {out[key]}\n  {p}")
        out[key] = p
    return out


def organize_pad() -> dict:
    by_name = index_by_basename(PAD_ROOT, ".png")
    dest_root = PAD_ROOT / "by_class"
    meta_counts: Counter[str] = Counter()
    moved_counts: Counter[str] = Counter()
    missing: list[str] = []

    with PAD_META.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row["diagnostic"].strip().lower()
            if raw not in PAD_EXPECTED_RAW:
                missing.append(f"PAD unknown diagnostic {row['diagnostic']!r} img {row['img_id']}")
                continue
            cls = PAD_RAW_TO_CANONICAL[raw]
            meta_counts[cls] += 1
            img_id = row["img_id"].strip()
            src = by_name.get(img_id)
            if src is None:
                missing.append(f"PAD file not found: {img_id}")
                continue
            dest_dir = dest_root / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / img_id
            if dest.resolve() == src.resolve():
                continue
            if dest.exists():
                missing.append(f"PAD dest already exists, skip: {dest}")
                continue
            shutil.move(str(src), str(dest))
            moved_counts[cls] += 1

    if sum(moved_counts.values()) == 0:
        moved_counts = Counter(_folder_image_counts(dest_root, "*.png"))

    return {
        "metadata_row_counts": dict(sorted(meta_counts.items())),
        "files_moved_by_class": dict(sorted(moved_counts.items())),
        "expected_classes": sorted(PAD_EXPECTED_CANONICAL),
        "issues": missing,
    }


def organize_ham() -> dict:
    by_name = index_by_basename(HAM_ROOT, ".jpg")
    dest_root = HAM_ROOT / "by_class"
    meta_counts: Counter[str] = Counter()
    moved_counts: Counter[str] = Counter()
    missing: list[str] = []

    with HAM_META.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dx = row["dx"].strip().lower()
            meta_counts[dx] += 1
            if dx not in HAM_EXPECTED:
                missing.append(f"HAM unknown dx {row['dx']!r} image {row['image_id']}")
                continue
            fname = f"{row['image_id'].strip()}.jpg"
            src = by_name.get(fname)
            if src is None:
                missing.append(f"HAM file not found: {fname}")
                continue
            dest_dir = dest_root / dx
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / fname
            if dest.resolve() == src.resolve():
                continue
            if dest.exists():
                missing.append(f"HAM dest already exists, skip: {dest}")
                continue
            shutil.move(str(src), str(dest))
            moved_counts[dx] += 1

    if sum(moved_counts.values()) == 0:
        moved_counts = Counter(_folder_image_counts(dest_root, "*.jpg"))

    return {
        "metadata_row_counts": dict(sorted(meta_counts.items())),
        "files_moved_by_class": dict(sorted(moved_counts.items())),
        "expected_classes": sorted(HAM_EXPECTED),
        "issues": missing,
    }


def main() -> None:
    pad_stats = organize_pad()
    ham_stats = organize_ham()
    report = {
        "pad_ufes": pad_stats,
        "ham10000": ham_stats,
    }
    STATS_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
