#!/usr/bin/env python3
"""
CLIP zero-shot classification baseline: cosine similarity of image/text embeddings.

Prompt template: "a photo of a {leaf}" with underscores replaced by spaces.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import open_clip
import torch

from dataset import append_results_json, build_leaf_mapping, leaf_to_parent_maps, load_manifest
from model import build_zeroshot_text_features


class _ZsDataset(torch.utils.data.Dataset):
    """Returns (tensor, leaf_index)."""

    def __init__(
        self,
        root: Path,
        entries: list[dict],
        preprocess,
        leaf2id: dict[str, int],
    ):
        self.root = root.resolve()
        self.entries = entries
        self.preprocess = preprocess
        self.leaf2id = leaf2id

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        from PIL import Image

        item = self.entries[idx]
        rel = Path(item["image"])
        path = self.root / rel if not rel.is_absolute() else rel
        leaf_idx = self.leaf2id[item["label"]]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), leaf_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP ViT-B/32 zero-shot on JSON manifests.")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--train-json", type=Path, default=Path("data/train.json"))
    parser.add_argument("--test-json", type=Path, default=Path("data/test.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases (useful for offline smoke runs).",
    )
    parser.add_argument("--wandb-project", type=str, default="vlm-hierarchy-noise")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    root = args.data_root.resolve()
    train_entries = load_manifest(args.train_json)
    test_entries = load_manifest(args.test_json)

    leaf2id = build_leaf_mapping(train_entries)
    leaf_names = [name for name, _ in sorted(leaf2id.items(), key=lambda kv: kv[1])]

    leaf_to_par = leaf_to_parent_maps(train_entries)

    train_labels = set(leaf2id)
    for s in test_entries:
        if s["label"] not in train_labels:
            raise ValueError(f"Test label {s['label']!r} missing from train label set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="openai",
        device=device,
    )
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    text_feats = build_zeroshot_text_features(
        clip_model, tokenizer, leaf_names, device=device
    )

    run_cfg = {"mode": "zero_shot", "train_json": str(args.train_json)}
    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name="clip_zeroshot", config=run_cfg)
        wandb_run.config.setdefault("taxonomy_type", "n/a")
        wandb_run.config["training"] = "zero-shot"

    loader = torch.utils.data.DataLoader(
        _ZsDataset(root, test_entries, preprocess, leaf2id),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    correct_leaf = 0
    correct_parent = 0
    total = 0

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for imgs, ys in loader:
            imgs = imgs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            feats = clip_model.encode_image(imgs).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = feats @ text_feats.T
            pred = logits.argmax(dim=1)

            correct_leaf += (pred == ys).sum().item()
            for p, yt in zip(pred.tolist(), ys.tolist()):
                true_leaf = leaf_names[yt]
                pred_leaf = leaf_names[int(p)]
                true_par = leaf_to_par[true_leaf]
                pred_par = leaf_to_par[pred_leaf]
                correct_parent += int(pred_par == true_par)
            total += ys.numel()

    acc_leaf = 100.0 * correct_leaf / max(total, 1)
    acc_par = 100.0 * correct_parent / max(total, 1)
    print(f"Zero-shot top-1 (leaf): {acc_leaf:.2f}% ({correct_leaf}/{total})")
    print(f"Parent accuracy (derived): {acc_par:.2f}% ({correct_parent}/{total})")

    if wandb_run is not None:
        wandb_run.summary["final_accuracy"] = acc_leaf
        wandb_run.summary["final_parent_accuracy"] = acc_par
        wandb_run.finish()

    append_results_json(
        args.runs_dir / "results.json",
        {
            "run_name": "clip_zeroshot",
            "model": "CLIP",
            "training": "zero-shot",
            "taxonomy": "N/A",
            "final_accuracy": acc_leaf,
            "final_parent_accuracy": acc_par,
        },
    )


if __name__ == "__main__":
    main()
