#!/usr/bin/env python3
"""
Baseline flat image classifier: ResNet18 (ImageNet pretrained) + cross-entropy.

Loads train.json / test.json with fields "image", "label", "path" (path ignored).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
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
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows).")
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

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = JsonImageDataset(train_entries, class_to_idx, tfm, root)
    test_ds = JsonImageDataset(test_entries, class_to_idx, tfm, root)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")
    print()

    for epoch in range(1, args.epochs + 1):
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
        print(
            f"Epoch {epoch}/{args.epochs} | train loss: {avg_loss:.4f} | "
            f"{time.perf_counter() - t0:.1f}s"
        )

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
