#!/usr/bin/env python3
"""
Cached CLIP-feature experiments for hierarchy-noise sweeps.

The full fine-tuning loop in train.py remains available, but a complete
multi-seed degradation study is much faster if the ViT-B/32 image embeddings are
extracted once and only the classification heads are trained repeatedly. This
script keeps the same leaf + parent objective and metrics as train.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Literal

import numpy as np
import open_clip
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    HierarchyJsonDataset,
    append_results_json,
    build_leaf_mapping,
    build_parent_mapping,
    build_parent_to_domain,
    leaf_to_parent_maps,
    load_manifest,
    make_noisy_parents,
    true_parent_ids_for_entries,
)
from metrics import compute_all_metrics, per_class_accuracy
from model import build_zeroshot_text_features

NOISE_TYPES = ["uniform", "in_domain", "cyclic"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LinearProbe(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_leaf: int,
        num_parent: int | None,
        *,
        adapter_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )
        self.leaf_head = nn.Linear(feat_dim, num_leaf)
        self.parent_head = nn.Linear(feat_dim, num_parent) if num_parent is not None else None
        nn.init.normal_(self.leaf_head.weight, std=0.02)
        nn.init.zeros_(self.leaf_head.bias)
        if self.parent_head is not None:
            nn.init.normal_(self.parent_head.weight, std=0.02)
            nn.init.zeros_(self.parent_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = x + self.adapter(x)
        return self.leaf_head(z), self.parent_head(z) if self.parent_head is not None else None


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def leaf_parent_ids_from_maps(id_to_leaf: list[str], leaf_to_par: dict[str, str], parent2id: dict[str, int]) -> torch.Tensor:
    return torch.tensor([parent2id[leaf_to_par[name]] for name in id_to_leaf], dtype=torch.long)


def aggregate_leaf_logits_to_parent(
    leaf_logits: torch.Tensor,
    leaf_parent_ids: torch.Tensor,
    num_parent: int,
) -> torch.Tensor:
    """Differentiable parent logits from leaf logits via parent-wise logsumexp."""
    cols = []
    for pid in range(num_parent):
        mask = leaf_parent_ids == pid
        cols.append(torch.logsumexp(leaf_logits[:, mask], dim=1))
    return torch.stack(cols, dim=1)


@torch.no_grad()
def extract_features(
    entries: list[dict],
    *,
    root: Path,
    leaf2id: dict[str, int],
    parent2id: dict[str, int],
    preprocess,
    clip_model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ds = HierarchyJsonDataset(entries, leaf2id, parent2id, preprocess, root)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feats, leaf_targets, parent_targets = [], [], []
    for x, y_leaf, y_parent in dl:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            z = clip_model.encode_image(x).float()
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        feats.append(z.cpu())
        leaf_targets.append(y_leaf.cpu())
        parent_targets.append(y_parent.cpu())
    return torch.cat(feats), torch.cat(leaf_targets), torch.cat(parent_targets)


def load_or_build_feature_cache(args: argparse.Namespace, device: torch.device):
    root = args.data_root.resolve()
    train_entries = load_manifest(args.train_json)
    test_entries = load_manifest(args.test_json)
    leaf2id = build_leaf_mapping(train_entries)
    parent2id = build_parent_mapping(train_entries)
    id_to_leaf = [name for name, _ in sorted(leaf2id.items(), key=lambda kv: kv[1])]

    args.features_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.features_dir / "clip_vit_b32_features.pt"

    if cache_path.exists() and not args.rebuild_features:
        cache = _torch_load(cache_path)
        if (
            cache.get("num_train") == len(train_entries)
            and cache.get("num_test") == len(test_entries)
            and cache.get("leaf2id") == leaf2id
            and cache.get("parent2id") == parent2id
        ):
            print(f"Loaded feature cache: {cache_path}")
            return cache, train_entries, test_entries, leaf2id, parent2id, id_to_leaf

    print("Extracting CLIP ViT-B/32 features...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device,
    )
    clip_model.eval()

    train_x, train_y, train_p = extract_features(
        train_entries,
        root=root,
        leaf2id=leaf2id,
        parent2id=parent2id,
        preprocess=preprocess,
        clip_model=clip_model,
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )
    test_x, test_y, test_p = extract_features(
        test_entries,
        root=root,
        leaf2id=leaf2id,
        parent2id=parent2id,
        preprocess=preprocess,
        clip_model=clip_model,
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )

    cache = {
        "model": "open_clip ViT-B-32 openai",
        "feature_normalized": True,
        "num_train": len(train_entries),
        "num_test": len(test_entries),
        "leaf2id": leaf2id,
        "parent2id": parent2id,
        "train_features": train_x,
        "train_leaf_targets": train_y,
        "train_parent_targets": train_p,
        "test_features": test_x,
        "test_leaf_targets": test_y,
        "test_parent_targets": test_p,
    }
    torch.save(cache, cache_path)
    print(f"Saved feature cache: {cache_path}")
    return cache, train_entries, test_entries, leaf2id, parent2id, id_to_leaf


@torch.no_grad()
def evaluate_probe(
    probe: LinearProbe,
    x: torch.Tensor,
    y_leaf: torch.Tensor,
    y_parent: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    probe.eval()
    leaf_logits, parent_logits = [], []
    for start in range(0, x.size(0), batch_size):
        xb = x[start:start + batch_size].to(device)
        ll, pp = probe(xb)
        leaf_logits.append(ll.float().cpu())
        if pp is not None:
            parent_logits.append(pp.float().cpu())
    return torch.cat(leaf_logits), torch.cat(parent_logits) if parent_logits else None


def train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    parent_targets: torch.Tensor,
    *,
    num_leaf: int,
    num_parent: int,
    mode: Literal["flat", "hierarchy"],
    lambda_parent: float,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    adapter_dim: int,
    dropout: float,
    leaf_parent_ids: torch.Tensor,
    consistency_weight: float,
) -> LinearProbe:
    set_seed(seed)
    probe = LinearProbe(
        train_x.size(1),
        num_leaf,
        num_parent if mode == "hierarchy" else None,
        adapter_dim=adapter_dim,
        dropout=dropout,
    ).to(device)
    opt = AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    leaf_parent_ids = leaf_parent_ids.to(device)
    ds = TensorDataset(train_x, train_y, parent_targets)
    generator = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)

    for _ in range(epochs):
        probe.train()
        for xb, yb, pb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ll, pp = probe(xb)
            loss = criterion(ll, yb)
            if mode == "hierarchy" and pp is not None:
                direct_parent_loss = criterion(pp, pb)
                leaf_parent_logits = aggregate_leaf_logits_to_parent(
                    ll,
                    leaf_parent_ids,
                    num_parent,
                )
                consistency_loss = criterion(leaf_parent_logits, pb)
                parent_loss = (
                    (1.0 - consistency_weight) * direct_parent_loss
                    + consistency_weight * consistency_loss
                )
                loss = loss + lambda_parent * parent_loss
            loss.backward()
            opt.step()
    return probe


def run_zeroshot(
    *,
    cache: dict,
    id_to_leaf: list[str],
    leaf_to_par: dict[str, str],
    parent_to_domain: dict[str, str],
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device,
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        text_feats = build_zeroshot_text_features(clip_model, tokenizer, id_to_leaf, device)
    scale = float(clip_model.logit_scale.exp().detach().cpu()) if hasattr(clip_model, "logit_scale") else 100.0
    logits = scale * cache["test_features"] @ text_feats.cpu().T
    metrics = compute_all_metrics(
        logits,
        cache["test_leaf_targets"],
        id_to_leaf,
        leaf_to_par,
        parent_to_domain,
    )
    return {
        "run_name": "clip_zeroshot",
        "run_id": "feature_clip_zeroshot",
        "model": "CLIP ViT-B/32",
        "pipeline": "zero-shot",
        "training": "zero-shot",
        "taxonomy": "N/A",
        "seed": args.seeds[0],
        "noise_fraction": 0.0,
        "noise_type": "none",
        "lambda_parent": 0.0,
        "epochs": 0,
        "final_accuracy": metrics["leaf_acc"],
        "final_parent_accuracy": metrics["parent_acc"],
        "top5_accuracy": metrics["top5_acc"],
        "hier_dist": metrics["hier_dist"],
        "severity_weighted_acc": metrics["severity_weighted_acc"],
        "ece": metrics["ece"],
    }


def result_row(
    *,
    run_name: str,
    run_id: str,
    mode: str,
    taxonomy: str,
    noise_fraction: float,
    noise_type: str,
    lambda_parent: float,
    seed: int,
    epochs: int,
    metrics: dict[str, float],
) -> dict:
    return {
        "run_name": run_name,
        "run_id": run_id,
        "model": "CLIP ViT-B/32",
        "pipeline": "cached-feature linear probe",
        "training": "fine-tuned" if mode == "flat" else "hierarchy",
        "taxonomy": taxonomy,
        "seed": seed,
        "noise_fraction": noise_fraction,
        "noise_type": noise_type,
        "lambda_parent": lambda_parent if mode == "hierarchy" else 0.0,
        "epochs": epochs,
        "final_accuracy": metrics["leaf_acc"],
        "final_parent_accuracy": metrics["parent_acc"],
        "top5_accuracy": metrics["top5_acc"],
        "hier_dist": metrics["hier_dist"],
        "severity_weighted_acc": metrics["severity_weighted_acc"],
        "ece": metrics["ece"],
    }


def append_per_parent_rows(
    path: Path,
    *,
    run_id: str,
    run_name: str,
    seed: int,
    noise_type: str,
    noise_fraction: float,
    leaf_logits: torch.Tensor,
    parent_logits: torch.Tensor | None,
    test_leaf_targets: torch.Tensor,
    test_parent_targets: torch.Tensor,
    id_to_parent: list[str],
    leaf_parent_ids: torch.Tensor,
) -> None:
    """Append per-true-parent leaf and parent accuracy rows for heatmaps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    leaf_preds = leaf_logits.argmax(dim=1)
    if parent_logits is not None:
        parent_preds = parent_logits.argmax(dim=1)
    else:
        parent_preds = leaf_parent_ids[leaf_preds]

    fields = [
        "run_id",
        "run_name",
        "seed",
        "noise_type",
        "noise_fraction",
        "parent",
        "parent_id",
        "n",
        "leaf_accuracy",
        "parent_accuracy",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for parent_id, parent in enumerate(id_to_parent):
            mask = test_parent_targets == parent_id
            n = int(mask.sum().item())
            if n == 0:
                continue
            leaf_acc = 100.0 * (leaf_preds[mask] == test_leaf_targets[mask]).float().mean().item()
            parent_acc = 100.0 * (parent_preds[mask] == test_parent_targets[mask]).float().mean().item()
            writer.writerow({
                "run_id": run_id,
                "run_name": run_name,
                "seed": seed,
                "noise_type": noise_type,
                "noise_fraction": noise_fraction,
                "parent": parent,
                "parent_id": parent_id,
                "n": n,
                "leaf_accuracy": f"{leaf_acc:.6f}",
                "parent_accuracy": f"{parent_acc:.6f}",
            })


def main() -> None:
    p = argparse.ArgumentParser(description="Run cached CLIP-feature hierarchy noise sweeps.")
    p.add_argument("--data-root", type=Path, default=Path("."))
    p.add_argument("--train-json", type=Path, default=Path("data/train.json"))
    p.add_argument("--test-json", type=Path, default=Path("data/test.json"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--features-dir", type=Path, default=Path("runs/features"))
    p.add_argument("--rebuild-features", action="store_true")
    p.add_argument("--feature-batch-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--adapter-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lambda-parent", type=float, default=0.4)
    p.add_argument("--consistency-weight", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument(
        "--noise-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    )
    p.add_argument("--noise-types", nargs="+", default=NOISE_TYPES)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache, train_entries, _test_entries, leaf2id, parent2id, id_to_leaf = load_or_build_feature_cache(args, device)
    leaf_to_par = leaf_to_parent_maps(train_entries)
    parent_to_domain = build_parent_to_domain(train_entries)
    leaf_parent_ids = leaf_parent_ids_from_maps(id_to_leaf, leaf_to_par, parent2id)
    id_to_parent = [name for name, _ in sorted(parent2id.items(), key=lambda kv: kv[1])]

    results_path = args.runs_dir / "results.json"
    per_parent_path = args.runs_dir / "per_parent_metrics.csv"
    t0 = time.perf_counter()

    zs = run_zeroshot(
        cache=cache,
        id_to_leaf=id_to_leaf,
        leaf_to_par=leaf_to_par,
        parent_to_domain=parent_to_domain,
        device=device,
        args=args,
    )
    append_results_json(results_path, zs)
    print(f"zero-shot: leaf={zs['final_accuracy']:.2f} parent={zs['final_parent_accuracy']:.2f}")

    train_x = cache["train_features"]
    train_y = cache["train_leaf_targets"]
    true_train_p = cache["train_parent_targets"]
    test_x = cache["test_features"]
    test_y = cache["test_leaf_targets"]
    test_p = cache["test_parent_targets"]

    for seed in args.seeds:
        for mode, taxonomy, nf, noise_type in [
            ("flat", "n/a", 0.0, "none"),
            ("hierarchy", "clean", 0.0, "none"),
        ]:
            probe = train_probe(
                train_x,
                train_y,
                true_train_p,
                num_leaf=len(leaf2id),
                num_parent=len(parent2id),
                mode=mode,  # type: ignore[arg-type]
                lambda_parent=args.lambda_parent,
                seed=seed,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                adapter_dim=args.adapter_dim,
                dropout=args.dropout,
                leaf_parent_ids=leaf_parent_ids,
                consistency_weight=args.consistency_weight,
            )
            ll, pp = evaluate_probe(probe, test_x, test_y, test_p, device=device, batch_size=args.batch_size)
            metrics = compute_all_metrics(
                ll,
                test_y,
                id_to_leaf,
                leaf_to_par,
                parent_to_domain,
                all_parent_logits=pp,
                all_parent_targets=test_p,
            )
            rn = "clip_flat" if mode == "flat" else "clip_hier_clean"
            row = result_row(
                run_name=rn,
                run_id=f"feature_{rn}_s{seed}",
                mode=mode,
                taxonomy=taxonomy,
                noise_fraction=nf,
                noise_type=noise_type,
                lambda_parent=args.lambda_parent,
                seed=seed,
                epochs=args.epochs,
                metrics=metrics,
            )
            append_results_json(results_path, row)
            append_per_parent_rows(
                per_parent_path,
                run_id=row["run_id"],
                run_name=row["run_name"],
                seed=seed,
                noise_type=noise_type,
                noise_fraction=nf,
                leaf_logits=ll,
                parent_logits=pp,
                test_leaf_targets=test_y,
                test_parent_targets=test_p,
                id_to_parent=id_to_parent,
                leaf_parent_ids=leaf_parent_ids,
            )
            print(f"{row['run_id']}: leaf={row['final_accuracy']:.2f} parent={row['final_parent_accuracy']:.2f}")

        for noise_type in args.noise_types:
            for nf in [x for x in args.noise_fractions if x > 0]:
                noisy_targets = make_noisy_parents(
                    train_entries,
                    parent2id,
                    fraction=nf,
                    seed=seed,
                    noise_type=noise_type,
                    parent_to_domain=parent_to_domain,
                )
                train_p = torch.tensor(noisy_targets, dtype=torch.long)
                probe = train_probe(
                    train_x,
                    train_y,
                    train_p,
                    num_leaf=len(leaf2id),
                    num_parent=len(parent2id),
                    mode="hierarchy",
                    lambda_parent=args.lambda_parent,
                    seed=seed,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    adapter_dim=args.adapter_dim,
                    dropout=args.dropout,
                    leaf_parent_ids=leaf_parent_ids,
                    consistency_weight=args.consistency_weight,
                )
                ll, pp = evaluate_probe(probe, test_x, test_y, test_p, device=device, batch_size=args.batch_size)
                metrics = compute_all_metrics(
                    ll,
                    test_y,
                    id_to_leaf,
                    leaf_to_par,
                    parent_to_domain,
                    all_parent_logits=pp,
                    all_parent_targets=test_p,
                )
                row = result_row(
                    run_name="clip_hier_noisy",
                    run_id=f"feature_clip_hier_{noise_type}_nf{nf}_s{seed}",
                    mode="hierarchy",
                    taxonomy="noisy",
                    noise_fraction=nf,
                    noise_type=noise_type,
                    lambda_parent=args.lambda_parent,
                    seed=seed,
                    epochs=args.epochs,
                    metrics=metrics,
                )
                append_results_json(results_path, row)
                append_per_parent_rows(
                    per_parent_path,
                    run_id=row["run_id"],
                    run_name=row["run_name"],
                    seed=seed,
                    noise_type=noise_type,
                    noise_fraction=nf,
                    leaf_logits=ll,
                    parent_logits=pp,
                    test_leaf_targets=test_y,
                    test_parent_targets=test_p,
                    id_to_parent=id_to_parent,
                    leaf_parent_ids=leaf_parent_ids,
                )
                print(
                    f"{row['run_id']}: leaf={row['final_accuracy']:.2f} "
                    f"parent={row['final_parent_accuracy']:.2f} hier_dist={row['hier_dist']:.3f}"
                )

    pc_dir = args.runs_dir / "feature_probe_reference"
    pc_dir.mkdir(parents=True, exist_ok=True)
    with open(pc_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": {
                    "train": len(train_entries),
                    "test": int(test_y.numel()),
                    "leaf_classes": len(leaf2id),
                    "parent_classes": len(parent2id),
                },
                "epochs": args.epochs,
                "lambda_parent": args.lambda_parent,
                "adapter_dim": args.adapter_dim,
                "dropout": args.dropout,
                "consistency_weight": args.consistency_weight,
                "seeds": args.seeds,
                "noise_fractions": args.noise_fractions,
                "noise_types": args.noise_types,
                "elapsed_sec": time.perf_counter() - t0,
            },
            f,
            indent=2,
        )

    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min. Results: {results_path}")


if __name__ == "__main__":
    main()
