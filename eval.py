#!/usr/bin/env python3
"""
Aggregate experiment rows from runs/results.json (PRD comparison table).

Optional: load a checkpoint and print test metrics (--checkpoint).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import HierarchyJsonDataset, leaf_to_parent_maps, load_manifest
from model import CLIPHierClassifier, create_clip_preprocesses


# (run_key in JSON, Model, Training column, Taxonomy column)
PRD_ROWS = [
    ("clip_zeroshot", "CLIP", "zero-shot", "N/A"),
    ("clip_flat", "CLIP", "fine-tuned", "n/a"),
    ("clip_hier_clean", "CLIP", "hierarchy", "clean"),
    ("clip_hier_noisy", "CLIP", "hierarchy", "noisy"),
]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rows_by_latest_run(entries: list[dict]) -> dict[str, dict]:
    """Last occurrence wins per run_name."""
    out: dict[str, dict] = {}
    for r in entries:
        rn = r.get("run_name")
        if rn:
            out[str(rn)] = r
    return out


def fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def print_table(by_run: dict[str, dict]) -> None:
    print("| Model | Training | Taxonomy | Accuracy | Parent Acc |")
    print("| ----- | ---------- | -------- | -------- | ----------- |")
    for run_key, model, train_label, tax_label in PRD_ROWS:
        r = by_run.get(run_key)
        acc_s = "---"
        par_s = "---"
        if r is not None:
            fa = r.get("final_accuracy")
            fp = r.get("final_parent_accuracy")
            if isinstance(fa, (int, float)):
                acc_s = fmt_pct(float(fa))
            if isinstance(fp, (int, float)):
                par_s = fmt_pct(float(fp))
        print(f"| {model} | {train_label} | {tax_label} | {acc_s} | {par_s} |")


def _torch_load_ckpt(path: Path, *, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@torch.no_grad()
def eval_checkpoint(path: Path, ns: argparse.Namespace) -> None:
    root = ns.data_root.resolve()
    ckpt = _torch_load_ckpt(path, map_location="cpu")
    leaf2id = ckpt["leaf2id"]
    mode = ckpt["mode"]

    _, eval_pp = create_clip_preprocesses()
    test_entries = load_manifest(ns.test_json)
    train_entries = load_manifest(ns.train_json)
    parent2id = ckpt["parent2id"]
    leaf_to_par_map = leaf_to_parent_maps(train_entries)

    ds = HierarchyJsonDataset(
        test_entries,
        leaf2id,
        parent2id,
        eval_pp,
        root,
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=ns.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPHierClassifier(num_leaf=len(leaf2id), mode=mode, num_parent=len(parent2id))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    id_to_leaf = [name for name, _ in sorted(leaf2id.items(), key=lambda kv: kv[1])]
    use_amp_cuda = device.type == "cuda"

    total = leaf_ok = parent_ok = 0

    for x, y_leaf, y_parent in dl:
        x = x.to(device, non_blocking=True)
        y_leaf = y_leaf.to(device, non_blocking=True)
        y_parent = y_parent.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp_cuda):
            ll, lp = model(x)

        preds = ll.argmax(dim=1)
        leaf_ok += (preds == y_leaf).sum().item()

        if lp is not None:
            pp = lp.argmax(dim=1)
            parent_ok += (pp == y_parent).sum().item()
        else:
            for px, ly in zip(preds.tolist(), y_leaf.tolist()):
                pn = id_to_leaf[int(px)]
                gt_p = leaf_to_par_map[id_to_leaf[int(ly)]]
                parent_ok += int(leaf_to_par_map[pn] == gt_p)

        total += y_leaf.numel()

    n = max(total, 1)
    print(
        f"Checkpoint {path.name}: leaf_acc={100.0 * leaf_ok / n:.2f}% "
        f"parent_acc={100.0 * parent_ok / n:.2f}% (mode={mode})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Print PRD comparison table from runs/results.json.")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--results-json", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=Path("."))
    p.add_argument("--train-json", type=Path, default=Path("data/train.json"))
    p.add_argument("--test-json", type=Path, default=Path("data/test.json"))
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    if args.checkpoint is not None:
        eval_checkpoint(args.checkpoint.resolve(), args)
        print()

    if args.results_json is None:
        res_path = (args.runs_dir / "results.json").resolve()
    elif args.results_json.is_absolute():
        res_path = args.results_json.resolve()
    else:
        cand = args.runs_dir / args.results_json.name
        res_path = cand.resolve() if cand.exists() else args.results_json.resolve()

    rows = load_rows(res_path)
    if not rows:
        print(f"No results at {res_path}.")
        return

    by_run = rows_by_latest_run(rows)
    print(f"Loaded {len(rows)} row(s) from {res_path}.\n")
    print_table(by_run)


if __name__ == "__main__":
    main()
