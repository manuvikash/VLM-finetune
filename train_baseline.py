#!/usr/bin/env python3
"""
Baseline flat image classifier: ResNet18 (ImageNet pretrained) + cross-entropy.

Loads train.json / test.json with fields "image", "label", "path" (path ignored).

Training uses train-time augmentation, then optional two-phase fine-tuning: frozen backbone + classifier head, then full model with a lower backbone learning rate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_label_mapping(samples: list[dict]) -> dict[str, int]:
    """Stable string label -> class index (0 .. num_classes-1)."""
    labels = sorted({s["label"] for s in samples})
    return {lab: i for i, lab in enumerate(labels)}


class JsonImageDataset(Dataset):
    def __init__(
        self,
        entries: list[dict],
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
        root: Path,
    ) -> None:
        self.entries = entries
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.root = root

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.entries[idx]
        rel = Path(item["image"])
        path = self.root / rel if not rel.is_absolute() else rel
        label = self.class_to_idx[item["label"]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


@torch.no_grad()
def mean_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    running = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / max(n, 1)


def imagenet_normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(
                224,
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
            transforms.ToTensor(),
            imagenet_normalize(),
        ]
    )


def eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            imagenet_normalize(),
        ]
    )


def set_backbone_requires_grad(model: nn.Module, requires: bool) -> None:
    for name, p in model.named_parameters():
        if name.startswith("fc."):
            continue
        p.requires_grad = requires


def build_optimizer(
    model: nn.Module,
    *,
    head_only: bool,
    head_lr: float,
    lr: float,
    backbone_lr: float,
) -> torch.optim.AdamW:
    if head_only:
        return torch.optim.AdamW(
            (p for p in model.fc.parameters() if p.requires_grad),
            lr=head_lr,
        )
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if n.startswith("fc.") and p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": backbone_lr},
            {"params": head, "lr": lr},
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 baseline on JSON image lists.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Project root for resolving relative image paths in JSON (default: current directory).",
    )
    parser.add_argument("--train-json", type=Path, default=Path("data/train.json"))
    parser.add_argument("--test-json", type=Path, default=Path("data/test.json"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the classifier head in the full-model phase.",
    )
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=3,
        help="First N epochs: train classifier head only (backbone frozen). 0 = skip this phase.",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
        help="Learning rate while the backbone is frozen.",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="Learning rate for backbone weights in the full-model phase.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows).")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of train.json held out for validation loss and early stopping (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for train/val split.")
    parser.add_argument(
        "--early-stop-tolerance",
        type=int,
        default=2,
        help="Stop after this many consecutive epochs without a lower val loss (0 = log val only, no stop).",
    )
    args = parser.parse_args()

    root = args.data_root.resolve()
    train_entries = load_manifest(args.train_json)
    test_entries = load_manifest(args.test_json)

    class_to_idx = build_label_mapping(train_entries)
    num_classes = len(class_to_idx)

    # Fail fast if test contains unknown labels
    train_labels = set(class_to_idx)
    for s in test_entries:
        if s["label"] not in train_labels:
            raise ValueError(f"Test label {s['label']!r} not in training label set.")

    val_entries: list[dict] = []
    if args.val_fraction > 0:
        if not (0 < args.val_fraction < 1):
            raise ValueError("--val-fraction must be in (0, 1) or 0 to disable validation.")
        labels = [s["label"] for s in train_entries]
        try:
            train_entries, val_entries = train_test_split(
                train_entries,
                test_size=args.val_fraction,
                random_state=args.seed,
                stratify=labels,
            )
        except ValueError:
            train_entries, val_entries = train_test_split(
                train_entries,
                test_size=args.val_fraction,
                random_state=args.seed,
            )

    train_tfm = train_transforms()
    eval_tfm = eval_transforms()

    train_ds = JsonImageDataset(train_entries, class_to_idx, train_tfm, root)
    val_ds = (
        JsonImageDataset(val_entries, class_to_idx, eval_tfm, root) if val_entries else None
    )
    test_ds = JsonImageDataset(test_entries, class_to_idx, eval_tfm, root)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_ds is not None
        else None
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    head_phase_epochs = min(max(args.head_epochs, 0), args.epochs)
    if head_phase_epochs > 0:
        set_backbone_requires_grad(model, False)
        for p in model.fc.parameters():
            p.requires_grad = True
        optimizer = build_optimizer(
            model,
            head_only=True,
            head_lr=args.head_lr,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
        )
    else:
        for p in model.parameters():
            p.requires_grad = True
        optimizer = build_optimizer(
            model,
            head_only=False,
            head_lr=args.head_lr,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
        )

    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    val_msg = f" | Val samples: {len(val_ds)}" if val_ds is not None else ""
    print(f"Train samples: {len(train_ds)}{val_msg} | Test samples: {len(test_ds)}")
    if head_phase_epochs > 0:
        print(
            f"Fine-tune: {head_phase_epochs} epoch(s) head-only (lr={args.head_lr}), "
            f"then full model (head lr={args.lr}, backbone lr={args.backbone_lr})."
        )
    print()

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stopped_early = False
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == head_phase_epochs + 1 and head_phase_epochs > 0:
            set_backbone_requires_grad(model, True)
            optimizer = build_optimizer(
                model,
                head_only=False,
                head_lr=args.head_lr,
                lr=args.lr,
                backbone_lr=args.backbone_lr,
            )
            print(f"--- Epoch {epoch}: unfrozen backbone; optimizer reinitialized ---")

        model.train()
        running = 0.0
        n = 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        avg_loss = running / max(n, 1)
        val_loss: float | None = None
        if val_loader is not None:
            val_loss = mean_loss(model, val_loader, device, criterion)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        line = (
            f"Epoch {epoch}/{args.epochs} | train loss: {avg_loss:.4f}"
            + (f" | val loss: {val_loss:.4f}" if val_loss is not None else "")
            + f" | {time.perf_counter() - t0:.1f}s"
        )
        print(line)

        if (
            val_loader is not None
            and args.early_stop_tolerance > 0
            and epochs_no_improve >= args.early_stop_tolerance
        ):
            if best_state is not None:
                model.load_state_dict(best_state)
            print(
                f"Early stop: no val improvement for {args.early_stop_tolerance} consecutive epoch(s) "
                f"(best val loss {best_val:.4f}; restored best checkpoint)."
            )
            stopped_early = True
            break

    if stopped_early:
        print(f"Stopped at epoch {epoch}/{args.epochs}.")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / max(total, 1)
    print()
    print(f"Test top-1 accuracy: {acc:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
