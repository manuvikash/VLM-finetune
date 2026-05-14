#!/usr/bin/env python3
"""
CLIP fine-tuning (flat vs hierarchy-aware) with W&B logging and checkpointing.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import (
    HierarchyJsonDataset,
    append_results_json,
    build_leaf_mapping,
    build_parent_mapping,
    build_parent_to_domain,
    leaf_to_parent_maps,
    load_manifest,
    make_noisy_parents,
    stratified_train_val_split,
)
from metrics import compute_all_metrics, per_class_accuracy
from model import CLIPHierClassifier, create_clip_preprocesses


WANDB_PROJECT = "vlm-hierarchy-noise"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def wandb_run_name(mode: Literal["flat", "hierarchy"], taxonomy: str) -> str:
    if mode == "flat":
        return "clip_flat"
    return "clip_hier_clean" if taxonomy == "clean" else "clip_hier_noisy"


def build_optimizer(
    model: CLIPHierClassifier,
    *,
    head_lr: float,
    backbone_lr: float,
    backbone_trainable: bool,
) -> AdamW:
    head_params = list(model.leaf_head.parameters())
    if model.parent_head is not None:
        head_params += list(model.parent_head.parameters())
    if not backbone_trainable:
        return AdamW(head_params, lr=head_lr)
    backbone = list(model.visual.parameters())
    return AdamW(
        [
            {"params": backbone, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    )


@torch.no_grad()
def evaluate_loader(
    model: CLIPHierClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    mode: Literal["flat", "hierarchy"],
    criterion: nn.CrossEntropyLoss,
    lambda_parent: float,
    use_amp_cuda: bool,
    id_to_leaf: list[str],
    leaf_to_par: dict[str, str],
) -> dict[str, float]:
    """Validation / test metrics: loss components, leaf + parent accuracy."""
    model.eval()
    total = 0
    leaf_correct = 0
    parent_correct = 0
    sum_leaf = 0.0
    sum_total = 0.0
    use_parent_head = model.parent_head is not None

    for x, y_leaf, y_parent in loader:
        x = x.to(device, non_blocking=True)
        y_leaf = y_leaf.to(device, non_blocking=True)
        y_parent = y_parent.to(device, non_blocking=True)
        bs = x.size(0)

        with torch.amp.autocast("cuda", enabled=use_amp_cuda):
            logits_leaf, logits_parent = model(x)
            leaf_loss = criterion(logits_leaf, y_leaf)

            if mode == "hierarchy" and logits_parent is not None:
                par_loss = criterion(logits_parent, y_parent)
                loss_batch = leaf_loss + lambda_parent * par_loss
            else:
                loss_batch = leaf_loss

        sum_leaf += leaf_loss.detach().cpu().item() * bs
        sum_total += loss_batch.detach().cpu().item() * bs

        preds = logits_leaf.argmax(dim=1)
        leaf_correct += (preds == y_leaf).sum().item()

        if use_parent_head and logits_parent is not None:
            pp = logits_parent.argmax(dim=1)
            parent_correct += (pp == y_parent).sum().item()
        else:
            for pidx, yl in zip(preds.tolist(), y_leaf.tolist()):
                pred_leaf_name = id_to_leaf[int(pidx)]
                gt_parent = leaf_to_par[id_to_leaf[int(yl)]]
                pred_parent = leaf_to_par[pred_leaf_name]
                parent_correct += int(pred_parent == gt_parent)

        total += bs

    n = max(total, 1)
    return {
        "leaf_loss_mean": sum_leaf / n,
        "total_loss_mean": sum_total / n,
        "leaf_acc": 100.0 * leaf_correct / n,
        "parent_acc": 100.0 * parent_correct / n,
    }


@torch.no_grad()
def collect_predictions(
    model: CLIPHierClassifier,
    loader: DataLoader,
    device: torch.device,
    use_amp_cuda: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run model on loader, return (leaf_logits, leaf_targets, parent_logits|None, parent_targets)."""
    model.eval()
    ll_buf: list[torch.Tensor] = []
    lt_buf: list[torch.Tensor] = []
    pl_buf: list[torch.Tensor] = []
    pt_buf: list[torch.Tensor] = []

    for x, y_leaf, y_parent in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp_cuda):
            logits_leaf, logits_parent = model(x)
        ll_buf.append(logits_leaf.float().cpu())
        lt_buf.append(y_leaf)
        if logits_parent is not None:
            pl_buf.append(logits_parent.float().cpu())
        pt_buf.append(y_parent)

    return (
        torch.cat(ll_buf),
        torch.cat(lt_buf),
        torch.cat(pl_buf) if pl_buf else None,
        torch.cat(pt_buf),
    )


def train_one_epoch(
    model: CLIPHierClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    mode: Literal["flat", "hierarchy"],
    optimizer: AdamW,
    criterion: nn.CrossEntropyLoss,
    lambda_parent: float,
    scaler: torch.amp.GradScaler | None,
    use_amp_cuda: bool,
    epoch_idx: int,
) -> dict[str, float]:
    _ = epoch_idx
    model.train()
    total = 0
    leaf_correct = 0
    parent_correct = 0
    sum_leaf = 0.0
    sum_par = 0.0
    sum_total = 0.0

    for x, y_leaf, y_parent in loader:
        x = x.to(device, non_blocking=True)
        y_leaf = y_leaf.to(device, non_blocking=True)
        y_parent = y_parent.to(device, non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad(set_to_none=True)
        if use_amp_cuda and scaler is not None:
            with torch.amp.autocast("cuda", enabled=True):
                logits_leaf, logits_parent = model(x)
                leaf_loss = criterion(logits_leaf, y_leaf)

                if mode == "hierarchy" and logits_parent is not None:
                    parent_loss = criterion(logits_parent, y_parent)
                    loss = leaf_loss + lambda_parent * parent_loss
                else:
                    parent_loss = torch.zeros((), device=device)
                    loss = leaf_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_leaf, logits_parent = model(x)
            leaf_loss = criterion(logits_leaf, y_leaf)
            if mode == "hierarchy" and logits_parent is not None:
                parent_loss = criterion(logits_parent, y_parent)
                loss = leaf_loss + lambda_parent * parent_loss
            else:
                parent_loss = torch.zeros((), device=device)
                loss = leaf_loss
            loss.backward()
            optimizer.step()

        preds = logits_leaf.argmax(dim=1)
        leaf_correct += (preds == y_leaf).sum().item()

        sum_leaf += leaf_loss.detach().cpu().item() * bs

        if mode == "hierarchy" and logits_parent is not None:
            sum_par += parent_loss.detach().cpu().item() * bs
            pp = logits_parent.argmax(dim=1)
            parent_correct += (pp == y_parent).sum().item()

        sum_total += loss.detach().cpu().item() * bs
        total += bs

    n = max(total, 1)
    out: dict[str, float] = {
        "train_leaf_loss": sum_leaf / n,
        "train_total_loss": sum_total / n,
        "train_leaf_acc": 100.0 * leaf_correct / n,
    }
    if mode == "hierarchy":
        out["train_parent_loss"] = sum_par / n
        out["train_parent_acc"] = 100.0 * parent_correct / n
    else:
        out["train_parent_loss"] = 0.0
        out["train_parent_acc"] = 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP ViT-B/32 on hierarchical JSON data.")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--train-json", type=Path, default=Path("data/train.json"))
    parser.add_argument("--test-json", type=Path, default=Path("data/test.json"))
    parser.add_argument("--mode", choices=["flat", "hierarchy"], required=True)
    parser.add_argument(
        "--taxonomy",
        choices=["clean", "noisy"],
        default="clean",
        help="Only applies to hierarchy mode (training parent targets). Val/test taxonomy stays clean.",
    )
    parser.add_argument("--lambda-parent", type=float, default=0.4)
    parser.add_argument("--noise-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases.",
    )
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    set_seed(args.seed)

    taxonomy_for_log = "n/a" if args.mode == "flat" else args.taxonomy
    root = args.data_root.resolve()
    full_train = load_manifest(args.train_json)
    test_entries = load_manifest(args.test_json)

    leaf2id = build_leaf_mapping(full_train)
    parent2id = build_parent_mapping(full_train)
    id_to_leaf = [name for name, _ in sorted(leaf2id.items(), key=lambda kv: kv[1])]
    leaf_to_par_map = leaf_to_parent_maps(full_train)
    parent_to_dom = build_parent_to_domain(full_train)

    train_labels = set(leaf2id)
    for s in test_entries:
        if s["label"] not in train_labels:
            raise ValueError(f"Test label {s['label']!r} missing from train label set.")

    train_fit, val_entries = stratified_train_val_split(
        full_train, val_fraction=args.val_fraction, seed=args.seed
    )

    noisy_parent_targets: list[int] | None = None
    if args.mode == "hierarchy" and args.taxonomy == "noisy":
        noisy_parent_targets = make_noisy_parents(
            train_fit,
            parent2id,
            fraction=args.noise_fraction,
            seed=args.seed,
        )

    train_pp, eval_pp = create_clip_preprocesses()

    train_ds = HierarchyJsonDataset(
        train_fit,
        leaf2id,
        parent2id,
        train_pp,
        root,
        training_parent_targets=noisy_parent_targets,
    )
    val_ds = HierarchyJsonDataset(val_entries, leaf2id, parent2id, eval_pp, root)
    test_ds = HierarchyJsonDataset(test_entries, leaf2id, parent2id, eval_pp, root)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp_cuda else None

    model = CLIPHierClassifier(
        num_leaf=len(leaf2id),
        mode=args.mode,
        num_parent=len(parent2id),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    head_epochs = min(max(args.head_epochs, 0), args.epochs)

    model.freeze_backbone(head_epochs > 0)
    optimizer = build_optimizer(
        model,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        backbone_trainable=head_epochs <= 0,
    )

    run_name = wandb_run_name(args.mode, args.taxonomy)

    if args.mode == "flat":
        run_id = f"clip_flat_s{args.seed}"
    elif args.taxonomy == "clean":
        run_id = f"clip_hier_clean_lam{args.lambda_parent}_s{args.seed}"
    else:
        run_id = f"clip_hier_noisy_nf{args.noise_fraction}_lam{args.lambda_parent}_s{args.seed}"

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"{run_name}_s{args.seed}",
            config={
                "mode": args.mode,
                "taxonomy_type": taxonomy_for_log,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "head_epochs": args.head_epochs,
                "head_lr": args.head_lr,
                "backbone_lr": args.backbone_lr,
                "lambda_parent": args.lambda_parent if args.mode == "hierarchy" else 0.0,
                "noise_fraction": args.noise_fraction,
                "val_fraction": args.val_fraction,
                "seed": args.seed,
                "device": str(device),
            },
        )

    ckpt_dir = (args.runs_dir / run_id).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    best_val_total = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        if epoch == head_epochs + 1 and head_epochs > 0:
            model.freeze_backbone(False)
            optimizer = build_optimizer(
                model,
                head_lr=args.head_lr,
                backbone_lr=args.backbone_lr,
                backbone_trainable=True,
            )
            print(f"--- Epoch {epoch}: backbone unfrozen; optimizer rebuilt ---")

        t0 = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            device,
            mode=args.mode,
            optimizer=optimizer,
            criterion=criterion,
            lambda_parent=args.lambda_parent,
            scaler=scaler,
            use_amp_cuda=use_amp_cuda,
            epoch_idx=epoch,
        )

        val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            mode=args.mode,
            criterion=criterion,
            lambda_parent=args.lambda_parent,
            use_amp_cuda=use_amp_cuda,
            id_to_leaf=id_to_leaf,
            leaf_to_par=leaf_to_par_map,
        )

        msg = (
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['train_total_loss']:.4f} "
            f"val_loss={val_metrics['total_loss_mean']:.4f} "
            f"acc={val_metrics['leaf_acc']:.2f} "
            f"parent_acc={val_metrics['parent_acc']:.2f} "
            f"lr_heads={optimizer.param_groups[-1]['lr']:.2e}"
        )
        if len(optimizer.param_groups) > 1:
            msg += f" lr_backbone={optimizer.param_groups[0]['lr']:.2e}"
        msg += f" | {time.perf_counter() - t0:.1f}s"
        print(msg)

        if val_metrics["total_loss_mean"] < best_val_total:
            best_val_total = val_metrics["total_loss_mean"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ckpt_payload = {
                "model_state": best_state,
                "leaf2id": leaf2id,
                "parent2id": parent2id,
                "mode": args.mode,
                "epoch": epoch,
                "val_total_loss": best_val_total,
            }
            torch.save(ckpt_payload, best_path)

        if wandb_run is not None:
            log_payload: dict[str, float | str | int] = {
                "epoch": epoch,
                "train_loss": train_metrics["train_total_loss"],
                "val_loss": val_metrics["total_loss_mean"],
                "accuracy": val_metrics["leaf_acc"],
                "parent_accuracy": val_metrics["parent_acc"],
                "learning_rate": float(optimizer.param_groups[-1]["lr"]),
                "taxonomy_type": taxonomy_for_log,
                "mode": str(args.mode),
            }
            if args.mode == "hierarchy":
                log_payload["leaf_loss"] = train_metrics["train_leaf_loss"]
                log_payload["parent_loss"] = train_metrics["train_parent_loss"]

            wandb_run.log(log_payload)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # --- Detailed test evaluation ---
    leaf_logits, leaf_targets, parent_logits, parent_targets = collect_predictions(
        model, test_loader, device, use_amp_cuda
    )
    test_m = compute_all_metrics(
        leaf_logits,
        leaf_targets,
        id_to_leaf,
        leaf_to_par_map,
        parent_to_dom,
        all_parent_logits=parent_logits,
        all_parent_targets=parent_targets,
    )

    print()
    print(
        f"Test leaf_acc={test_m['leaf_acc']:.2f}% top5={test_m['top5_acc']:.2f}% "
        f"parent_acc={test_m['parent_acc']:.2f}% hier_dist={test_m['hier_dist']:.3f} "
        f"sev_w_acc={test_m['severity_weighted_acc']:.2f}% ece={test_m['ece']:.4f}"
    )

    pc = per_class_accuracy(
        leaf_logits.argmax(dim=1).tolist(),
        leaf_targets.tolist(),
        id_to_leaf,
        leaf_to_par_map,
    )
    with open(ckpt_dir / "per_class.json", "w", encoding="utf-8") as f:
        json.dump(pc, f, indent=2)

    if wandb_run is not None:
        import wandb

        for k, v in test_m.items():
            wandb_run.summary[f"final_{k}"] = v
        artifact = wandb.Artifact(run_id + "_checkpoint", type="model")
        if best_path.exists():
            artifact.add_file(str(best_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    append_results_json(
        args.runs_dir / "results.json",
        {
            "run_name": run_name,
            "run_id": run_id,
            "model": "CLIP",
            "training": "fine-tuned" if args.mode == "flat" else "hierarchy",
            "taxonomy": taxonomy_for_log,
            "seed": args.seed,
            "noise_fraction": (
                args.noise_fraction
                if args.mode == "hierarchy" and args.taxonomy == "noisy"
                else 0.0
            ),
            "lambda_parent": args.lambda_parent if args.mode == "hierarchy" else 0.0,
            "epochs": args.epochs,
            "final_accuracy": test_m["leaf_acc"],
            "final_parent_accuracy": test_m["parent_acc"],
            "top5_accuracy": test_m["top5_acc"],
            "hier_dist": test_m["hier_dist"],
            "severity_weighted_acc": test_m["severity_weighted_acc"],
            "ece": test_m["ece"],
            "checkpoint": str(best_path) if best_path.exists() else None,
        },
    )


if __name__ == "__main__":
    main()
