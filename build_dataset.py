#!/usr/bin/env python3
"""
Build a hierarchical image dataset for flat vs hierarchy-aware classification.

Output schema (each sample):
  - image: relative path under project root (forward slashes)
  - label: leaf class name (string)
  - path: list[str], shallow hierarchy (length 3 for this builder)

Data source: CIFAR-100 via torchvision. Provides 100 fine-grained labels under
20 coarse superclasses. We add a top "domain" bucket so each path has 3 levels.

Modes:
  Default (full CIFAR-100 with official test split):
    python build_dataset.py

  Small subset (original behaviour):
    python build_dataset.py --num-coarse 15 --per-class 48
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR100

COARSE_IDX_TO_DOMAIN = [
    "animal",   # 0  aquatic mammals
    "animal",   # 1  fish
    "plant",    # 2  flowers
    "object",   # 3  food containers
    "plant",    # 4  fruit and vegetables
    "object",   # 5  household electrical devices
    "object",   # 6  household furniture
    "animal",   # 7  insects
    "animal",   # 8  large carnivores
    "scene",    # 9  large man-made outdoor things
    "scene",    # 10 large natural outdoor scenes
    "animal",   # 11 large omnivores and herbivores
    "animal",   # 12 medium-sized mammals
    "animal",   # 13 non-insect invertebrates
    "person",   # 14 people
    "animal",   # 15 reptiles
    "animal",   # 16 small mammals
    "plant",    # 17 trees
    "vehicle",  # 18 vehicles 1
    "vehicle",  # 19 vehicles 2
]


def _slug(s: str) -> str:
    return s.replace(" ", "_").replace("-", "_").lower()


def _coarse_targets(ds: CIFAR100) -> list[int]:
    """Per-image coarse label (0..19), compatible across torchvision versions."""
    if hasattr(ds, "coarse_targets"):
        return list(ds.coarse_targets)
    split_file = "train" if ds.train else "test"
    base = Path(ds.root) / getattr(ds, "base_folder", "cifar-100-python")
    with open(base / split_file, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    return list(entry["coarse_labels"])


def _coarse_class_names(ds: CIFAR100) -> list[str]:
    if hasattr(ds, "coarse_classes"):
        return list(ds.coarse_classes)
    base = Path(ds.root) / getattr(ds, "base_folder", "cifar-100-python")
    with open(base / "meta", "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    return list(meta["coarse_label_names"])


def build_records_from_split(
    ds: CIFAR100,
    *,
    project_root: Path,
    images_dir: Path,
    num_coarse_groups: int,
    samples_per_leaf: int,
    seed: int,
    filename_offset: int = 1,
) -> tuple[list[dict], int]:
    """
    Build JSON records from a CIFAR-100 split.

    Args:
        samples_per_leaf: images per leaf class; 0 means use all available.
        filename_offset: starting image counter for filenames.

    Returns:
        (records, next_filename_offset)
    """
    fine_names = [_slug(x) for x in ds.classes]
    coarse_names = [_slug(x) for x in _coarse_class_names(ds)]
    coarse_per_sample = _coarse_targets(ds)

    fine_to_coarse: dict[int, int] = {}
    for idx, y in enumerate(ds.targets):
        c = coarse_per_sample[idx]
        if y in fine_to_coarse and fine_to_coarse[y] != c:
            raise RuntimeError("Inconsistent fine->coarse mapping in CIFAR-100.")
        fine_to_coarse[y] = c

    selected_coarse = set(range(num_coarse_groups))
    allowed_fine = {f for f, c in fine_to_coarse.items() if c in selected_coarse}

    by_fine: dict[int, list[int]] = {i: [] for i in allowed_fine}
    for idx, y in enumerate(ds.targets):
        if y in allowed_fine:
            by_fine[y].append(idx)

    rng = random.Random(seed)
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    counter = filename_offset

    for fine_idx in sorted(allowed_fine):
        pool = by_fine[fine_idx]
        if samples_per_leaf <= 0:
            chosen = pool
        else:
            if len(pool) < samples_per_leaf:
                raise RuntimeError(
                    f"Not enough images for class {fine_idx}: "
                    f"need {samples_per_leaf}, have {len(pool)}"
                )
            chosen = rng.sample(pool, samples_per_leaf)

        for src_i in chosen:
            image_name = f"{counter:06d}.jpg"
            rel_path = f"data/images/{image_name}"
            out_path = project_root / rel_path.replace("/", os.sep)
            Image.fromarray(ds.data[src_i]).save(out_path, format="JPEG", quality=92)

            y = ds.targets[src_i]
            coarse_i = coarse_per_sample[src_i]
            records.append({
                "image": rel_path.replace("\\", "/"),
                "label": fine_names[y],
                "path": [COARSE_IDX_TO_DOMAIN[coarse_i], coarse_names[coarse_i], fine_names[y]],
            })
            counter += 1

    return records, counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hierarchical dataset JSON + image folder.")
    parser.add_argument(
        "--root", type=str, default="data",
        help="Output directory for JSON files and images/ (relative to cwd).",
    )
    parser.add_argument(
        "--num-coarse", type=int, default=20,
        help="Number of CIFAR-100 coarse superclasses to include (max 20, 5 fine each).",
    )
    parser.add_argument(
        "--per-class", type=int, default=0,
        help="Samples per leaf class from each split; 0 = use all available.",
    )
    parser.add_argument(
        "--use-test-split", action="store_true",
        help="Use official CIFAR-100 test split instead of random 80/20.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test fraction when not using --use-test-split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    data_dir = (project_root / args.root).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"

    cifar_root = project_root / ".cifar_cache"

    ds_train = CIFAR100(root=str(cifar_root), train=True, download=True, transform=None)
    train_records, next_offset = build_records_from_split(
        ds_train,
        project_root=project_root,
        images_dir=images_dir,
        num_coarse_groups=args.num_coarse,
        samples_per_leaf=args.per_class,
        seed=args.seed,
        filename_offset=1,
    )

    if args.use_test_split:
        ds_test = CIFAR100(root=str(cifar_root), train=False, download=True, transform=None)
        test_records, _ = build_records_from_split(
            ds_test,
            project_root=project_root,
            images_dir=images_dir,
            num_coarse_groups=args.num_coarse,
            samples_per_leaf=args.per_class,
            seed=args.seed,
            filename_offset=next_offset,
        )
        all_records = train_records + test_records
        train_r, test_r = train_records, test_records
    else:
        all_records = train_records
        labels = [r["label"] for r in all_records]
        train_r, test_r = train_test_split(
            all_records,
            test_size=args.test_size,
            random_state=args.seed,
            shuffle=True,
            stratify=labels,
        )

    with open(data_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    with open(data_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_r, f, indent=2, ensure_ascii=False)
    with open(data_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_r, f, indent=2, ensure_ascii=False)

    n_classes = len({r["label"] for r in all_records})
    n_parents = len({r["path"][-2] for r in all_records})
    print("Done.")
    print(f"  Total samples: {len(all_records)} (train={len(train_r)}, test={len(test_r)})")
    print(f"  Leaf classes:  {n_classes}")
    print(f"  Parent groups: {n_parents}")
    print(f"  Domains:       {sorted({r['path'][0] for r in all_records})}")
    if all_records:
        print("  Example entry:")
        print(json.dumps(all_records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
