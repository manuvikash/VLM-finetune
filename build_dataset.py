#!/usr/bin/env python3
"""
Build a small hierarchical image dataset for flat vs hierarchy-aware classification.

Output schema (each sample):
  - image: relative path under project root (forward slashes)
  - label: leaf class name (string)
  - path: list[str], shallow hierarchy (length 3 for this builder)

Data source
-----------
We do NOT ship a full OVEN pipeline here: OVEN (Open-domain Visual Entity Recognition)
ties images to Wikipedia/Wikidata at massive scale; usable subsets usually need the
Hugging Face `datasets` stack, gated access, and heavy image or metadata downloads.
For an MVP you can swap the loader below for OVEN later while keeping the same JSON.

Fallback used here: CIFAR-100 via torchvision (one `pip install` stack you likely
already have). It provides 100 fine-grained labels under 20 coarse superclasses.
We pick a subset of superclasses (default: 15 → 75 leaf classes), sample a fixed
number of train images per leaf (default: 48), save PNGs under data/images/, and build:
  data/dataset.json, data/train.json, data/test.json

Usage:
  python build_dataset.py
  python build_dataset.py --root data --seed 42
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

# ---------------------------------------------------------------------------
# Hierarchy: 3-level paths without a full knowledge graph
# ---------------------------------------------------------------------------
# CIFAR-100 gives coarse + fine labels. We add a top "domain" bucket so each path
# has three meaningful levels, similar in spirit to ["animal", "mammal", "dog"].
# Domains are coarse groupings over the 20 CIFAR coarse superclasses (fixed order
# from the official CIFAR-100 metadata bundled with torchvision).
COARSE_IDX_TO_DOMAIN = [
    "animal",  # aquatic mammals
    "animal",  # fish
    "plant",  # flowers
    "object",  # food containers
    "plant",  # fruit and vegetables
    "object",  # household electrical devices
    "object",  # household furniture
    "animal",  # insects
    "animal",  # large carnivores
    "scene",  # large man-made outdoor things
    "scene",  # large natural outdoor scenes
    "animal",  # large omnivores and herbivores
    "animal",  # medium-sized mammals
    "animal",  # non-insect invertebrates
    "person",  # people
    "animal",  # reptiles
    "animal",  # small mammals
    "plant",  # trees
    "vehicle",  # vehicles 1
    "vehicle",  # vehicles 2
]


def _slug(s: str) -> str:
    return s.replace(" ", "_").replace("-", "_").lower()


def _coarse_targets(ds: CIFAR100) -> list[int]:
    """Per-image coarse label (0..19), compatible across torchvision versions."""
    if hasattr(ds, "coarse_targets"):
        return list(ds.coarse_targets)
    base = Path(ds.root) / getattr(ds, "base_folder", "cifar-100-python")
    with open(base / "train", "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    return list(entry["coarse_labels"])


def _coarse_class_names(ds: CIFAR100) -> list[str]:
    if hasattr(ds, "coarse_classes"):
        return list(ds.coarse_classes)
    base = Path(ds.root) / getattr(ds, "base_folder", "cifar-100-python")
    with open(base / "meta", "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    return list(meta["coarse_label_names"])


def build_records(
    *,
    project_root: Path,
    data_dir: Path,
    num_coarse_groups: int,
    samples_per_leaf: int,
    seed: int,
) -> list[dict]:
    """Load CIFAR-100 train split, subsample, save images, return JSON-ready rows."""
    # Download / load training images (50k); we only export a small subset to disk.
    cifar_root = project_root / ".cifar_cache"
    ds = CIFAR100(root=str(cifar_root), train=True, download=True, transform=None)

    fine_names = [_slug(x) for x in ds.classes]
    coarse_names = [_slug(x) for x in _coarse_class_names(ds)]
    coarse_per_sample = _coarse_targets(ds)

    # Map each fine label id (0..99) to its coarse superclass id (0..19).
    # CIFAR-100 fine indices are not contiguous blocks by coarse; derive from data.
    fine_to_coarse: dict[int, int] = {}
    for idx, y in enumerate(ds.targets):
        c = coarse_per_sample[idx]
        if y in fine_to_coarse and fine_to_coarse[y] != c:
            raise RuntimeError("Inconsistent fine→coarse mapping in CIFAR-100 train split.")
        fine_to_coarse[y] = c

    # Pick the first N coarse superclasses (deterministic); each has exactly 5 fine labels.
    selected_coarse = set(range(num_coarse_groups))
    allowed_fine = {f for f, c in fine_to_coarse.items() if c in selected_coarse}

    # Index training images by fine label for balanced sampling.
    by_fine: dict[int, list[int]] = {i: [] for i in allowed_fine}
    for idx, y in enumerate(ds.targets):
        if y in allowed_fine:
            by_fine[y].append(idx)

    rng = random.Random(seed)
    images_out = data_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    out_counter = 1  # 0001.jpg style filenames

    for fine_idx in sorted(allowed_fine):
        pool = by_fine[fine_idx]
        if len(pool) < samples_per_leaf:
            raise RuntimeError(
                f"Not enough images for fine class {fine_idx}: need {samples_per_leaf}, have {len(pool)}"
            )
        chosen = rng.sample(pool, samples_per_leaf)

        for src_i in chosen:
            img_array = ds.data[src_i]
            image_name = f"{out_counter:04d}.jpg"
            rel_path = f"data/images/{image_name}"
            out_path = project_root / rel_path.replace("/", os.sep)
            Image.fromarray(img_array).save(out_path, format="JPEG", quality=92)

            y = ds.targets[src_i]
            coarse_i = coarse_per_sample[src_i]
            domain = COARSE_IDX_TO_DOMAIN[coarse_i]
            leaf = fine_names[y]
            hierarchy_path = [
                domain,
                coarse_names[coarse_i],
                leaf,
            ]

            records.append(
                {
                    "image": rel_path.replace("\\", "/"),
                    "label": leaf,
                    "path": hierarchy_path,
                }
            )
            out_counter += 1

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hierarchical dataset JSON + image folder.")
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="Output directory for dataset.json, train.json, test.json and images/ (relative to cwd).",
    )
    parser.add_argument(
        "--num-coarse",
        type=int,
        default=15,
        help="Number of CIFAR-100 coarse superclasses to include (5 fine classes each).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=48,
        help="Samples per leaf class (total = num_coarse * 5 * per_class).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and train/test split.")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    data_dir = (project_root / args.root).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect + materialize images + full list
    records = build_records(
        project_root=project_root,
        data_dir=data_dir,
        num_coarse_groups=args.num_coarse,
        samples_per_leaf=args.per_class,
        seed=args.seed,
    )

    # 2) Write dataset.json (full manifest)
    dataset_path = data_dir / "dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # 3) Stratified split: 80% train / 20% test (same seed for reproducibility)
    labels = [r["label"] for r in records]
    train_r, test_r = train_test_split(
        records,
        test_size=0.2,
        random_state=args.seed,
        shuffle=True,
        stratify=labels,
    )
    with open(data_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_r, f, indent=2, ensure_ascii=False)
    with open(data_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_r, f, indent=2, ensure_ascii=False)

    # 4) Short summary for the console
    n_classes = len({r["label"] for r in records})
    print("Done.")
    print(f"  Total samples: {len(records)}")
    print(f"  Leaf classes: {n_classes}")
    print(f"  Example leaf labels: {sorted({r['label'] for r in records})[:8]} ...")
    print("  Example entry:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
