#!/usr/bin/env python3
"""
JSON image manifest loader for hierarchical CLIP fine-tuning.

Expected row shape (see PRD): image, label (leaf string), path: [..., parent, leaf].
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_manifest(path: Path | str) -> list[dict]:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_leaf_mapping(samples: list[dict]) -> dict[str, int]:
    """Stable leaf label -> index (uses field 'label')."""
    labels = sorted({s["label"] for s in samples})
    return {lab: i for i, lab in enumerate(labels)}


def extract_parent(path: list[str]) -> str:
    if len(path) < 2:
        raise ValueError(f"path must have at least 2 segments, got {path!r}")
    return path[-2]


def build_parent_mapping(samples: list[dict]) -> dict[str, int]:
    """Stable parent label -> index (path[-2] per sample)."""
    parents = sorted({extract_parent(s["path"]) for s in samples})
    return {p: i for i, p in enumerate(parents)}


def leaf_to_parent_maps(samples: list[dict]) -> dict[str, str]:
    """Map leaf name -> canonical parent string (must be consistent for one leaf)."""
    by_leaf: dict[str, set[str]] = {}
    for s in samples:
        leaf = s["label"]
        par = extract_parent(s["path"])
        by_leaf.setdefault(leaf, set()).add(par)

    out: dict[str, str] = {}
    inconsistent: list[tuple[str, set[str]]] = []
    for leaf, ps in by_leaf.items():
        if len(ps) != 1:
            inconsistent.append((leaf, ps))
            continue
        out[leaf] = next(iter(ps))
    if inconsistent:
        detail = "; ".join(f"{l}: {sorted(s)}" for l, s in inconsistent[:10])
        raise ValueError(
            f"Inconsistent parent for same leaf ({len(inconsistent)} cases). Examples: {detail}"
        )
    return out


def stratified_train_val_split(
    train_entries: list[dict],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Hold out stratified fraction of train by leaf label."""
    if val_fraction <= 0:
        return train_entries, []
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be in (0, 1) or 0.")

    labels = [s["label"] for s in train_entries]
    try:
        return train_test_split(
            train_entries,
            test_size=val_fraction,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        return train_test_split(
            train_entries,
            test_size=val_fraction,
            random_state=seed,
        )


def make_noisy_parents(
    entries: list[dict],
    parent2id: dict[str, int],
    *,
    fraction: float = 0.2,
    seed: int = 42,
) -> list[int]:
    """
    Per-sample corrupted parent IDs for hierarchy loss (training only).

    Each entry: with probability `fraction`, replace true parent id with uniform
    random *different* parent id. Otherwise keep true parent.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    rng = random.Random(seed)
    parent_vals = sorted(parent2id.values())
    if fraction > 0 and len(parent_vals) < 2:
        raise ValueError("Need at least 2 distinct parents to apply taxonomy noise.")

    out: list[int] = []
    for s in entries:
        true_par = extract_parent(s["path"])
        tid = parent2id[true_par]
        if fraction <= 0 or rng.random() >= fraction:
            out.append(tid)
        else:
            others = [i for i in parent_vals if i != tid]
            out.append(rng.choice(others))
    return out


class HierarchyJsonDataset(Dataset):
    """
    Loads images from manifest.

    Returns (preprocessed_tensor, leaf_id, parent_target_id).

    When `training_parent_targets` is set (same length as entries), batch parent loss
    uses those IDs (noisy taxonomy). Otherwise uses true parent from path.
    """

    def __init__(
        self,
        entries: list[dict],
        leaf2id: dict[str, int],
        parent2id: dict[str, int],
        preprocess: Callable,
        root: Path,
        training_parent_targets: list[int] | None = None,
    ) -> None:
        self.entries = entries
        self.leaf2id = leaf2id
        self.parent2id = parent2id
        self.preprocess = preprocess
        self.root = root.resolve()

        if training_parent_targets is not None and len(training_parent_targets) != len(entries):
            raise ValueError(
                "training_parent_targets length must match entries when provided."
            )
        self.training_parent_targets = training_parent_targets

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        item = self.entries[idx]
        rel = Path(item["image"])
        path = self.root / rel if not rel.is_absolute() else rel
        leaf = self.leaf2id[item["label"]]

        img = Image.open(path).convert("RGB")
        x = self.preprocess(img)

        if self.training_parent_targets is not None:
            par = self.training_parent_targets[idx]
        else:
            par = self.parent2id[extract_parent(item["path"])]
        return x, leaf, par


def true_parent_ids_for_entries(entries: list[dict], parent2id: dict[str, int]) -> list[int]:
    """Ground-truth parent id per sample (for val metrics aligned with taxonomy)."""
    return [parent2id[extract_parent(s["path"])] for s in entries]


def append_results_json(path: Path | str, row: dict[str, Any]) -> None:
    """Append one JSON-serializable object to a list stored at path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    rows.append(row)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

