#!/usr/bin/env python3
"""
CLIP zero-shot classification baseline: cosine similarity of image/text embeddings.

Prompt template: "a photo of a {leaf}" with underscores replaced by spaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import open_clip
import torch

from dataset import (
    append_results_json,
    build_leaf_mapping,
    build_parent_to_domain,
    leaf_to_parent_maps,
    load_manifest,
)
from metrics import compute_all_metrics, per_class_accuracy
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
        "--seed",
        type=int,
        default=42,
        help="Recorded in results for consistency (evaluation is deterministic).",
    )
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
    parent_to_dom = build_parent_to_domain(train_entries)

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

    run_cfg = {"mode": "zero_shot", "train_json": str(args.train_json), "seed": args.seed}
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

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for imgs, ys in loader:
            imgs = imgs.to(device, non_blocking=True)
            feats = clip_model.encode_image(imgs).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = feats @ text_feats.T
            all_logits.append(logits.cpu())
            all_targets.append(ys)

    cat_logits = torch.cat(all_logits)
    cat_targets = torch.cat(all_targets)

    test_m = compute_all_metrics(
        cat_logits,
        cat_targets,
        leaf_names,
        leaf_to_par,
        parent_to_dom,
    )

    print(
        f"Zero-shot  leaf_acc={test_m['leaf_acc']:.2f}% top5={test_m['top5_acc']:.2f}% "
        f"parent_acc={test_m['parent_acc']:.2f}% hier_dist={test_m['hier_dist']:.3f} "
        f"sev_w_acc={test_m['severity_weighted_acc']:.2f}% ece={test_m['ece']:.4f}"
    )

    pc = per_class_accuracy(
        cat_logits.argmax(dim=1).tolist(),
        cat_targets.tolist(),
        leaf_names,
        leaf_to_par,
    )
    pc_dir = Path(args.runs_dir) / "clip_zeroshot"
    pc_dir.mkdir(parents=True, exist_ok=True)
    with open(pc_dir / "per_class.json", "w", encoding="utf-8") as f:
        json.dump(pc, f, indent=2)

    if wandb_run is not None:
        for k, v in test_m.items():
            wandb_run.summary[f"final_{k}"] = v
        wandb_run.finish()

    append_results_json(
        args.runs_dir / "results.json",
        {
            "run_name": "clip_zeroshot",
            "run_id": "clip_zeroshot",
            "model": "CLIP",
            "training": "zero-shot",
            "taxonomy": "N/A",
            "seed": args.seed,
            "noise_fraction": 0.0,
            "lambda_parent": 0.0,
            "epochs": 0,
            "final_accuracy": test_m["leaf_acc"],
            "final_parent_accuracy": test_m["parent_acc"],
            "top5_accuracy": test_m["top5_acc"],
            "hier_dist": test_m["hier_dist"],
            "severity_weighted_acc": test_m["severity_weighted_acc"],
            "ece": test_m["ece"],
        },
    )


if __name__ == "__main__":
    main()
