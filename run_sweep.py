#!/usr/bin/env python3
"""
Sweep runner for paper experiments.

Orchestrates multi-seed baselines, noise fraction sweeps, and lambda sweeps.

Usage:
  python run_sweep.py baselines              # 4 conditions x 5 seeds
  python run_sweep.py noise                  # 8 noise fracs x 5 seeds
  python run_sweep.py lambda                 # 6 lambdas x 3 noise levels x 3 seeds
  python run_sweep.py all                    # everything
  python run_sweep.py baselines --dry-run    # preview commands
  python run_sweep.py noise --resume         # skip completed runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

SEEDS = [42, 123, 456, 789, 1337]
NOISE_FRACTIONS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
LAMBDA_VALUES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
LAMBDA_NOISE_LEVELS = [0.0, 0.2, 0.5]


def build_configs(sweep: str, seeds: list[int] | None = None) -> list[dict]:
    if seeds is None:
        seeds = SEEDS
    configs: list[dict] = []

    if sweep in ("baselines", "all"):
        configs.append({"type": "zeroshot", "seed": seeds[0]})
        for seed in seeds:
            configs.append({
                "type": "train", "mode": "flat", "taxonomy": "clean",
                "noise_fraction": 0.0, "lambda_parent": 0.4, "seed": seed,
            })
            configs.append({
                "type": "train", "mode": "hierarchy", "taxonomy": "clean",
                "noise_fraction": 0.0, "lambda_parent": 0.4, "seed": seed,
            })
            configs.append({
                "type": "train", "mode": "hierarchy", "taxonomy": "noisy",
                "noise_fraction": 0.2, "lambda_parent": 0.4, "seed": seed,
            })

    if sweep in ("noise", "all"):
        for nf, seed in product(NOISE_FRACTIONS, seeds):
            tax = "clean" if nf == 0.0 else "noisy"
            configs.append({
                "type": "train", "mode": "hierarchy", "taxonomy": tax,
                "noise_fraction": nf, "lambda_parent": 0.4, "seed": seed,
            })

    if sweep in ("lambda", "all"):
        for lam, nf, seed in product(LAMBDA_VALUES, LAMBDA_NOISE_LEVELS, seeds[:3]):
            tax = "clean" if nf == 0.0 else "noisy"
            configs.append({
                "type": "train", "mode": "hierarchy", "taxonomy": tax,
                "noise_fraction": nf, "lambda_parent": lam, "seed": seed,
            })

    seen: set[str] = set()
    unique: list[dict] = []
    for c in configs:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def config_to_cmd(cfg: dict, common: dict) -> list[str]:
    if cfg["type"] == "zeroshot":
        return [
            sys.executable, "zeroshot.py",
            "--seed", str(cfg["seed"]),
            "--no-wandb",
            "--runs-dir", common["runs_dir"],
        ]
    return [
        sys.executable, "train.py",
        "--mode", cfg["mode"],
        "--taxonomy", cfg["taxonomy"],
        "--noise-fraction", str(cfg["noise_fraction"]),
        "--lambda-parent", str(cfg["lambda_parent"]),
        "--seed", str(cfg["seed"]),
        "--epochs", str(common["epochs"]),
        "--batch-size", str(common["batch_size"]),
        "--no-wandb",
        "--runs-dir", common["runs_dir"],
    ]


def _run_key(cfg: dict) -> tuple:
    if cfg["type"] == "zeroshot":
        return ("zeroshot", cfg["seed"])
    return (
        cfg["mode"], cfg["taxonomy"],
        cfg["noise_fraction"], cfg["lambda_parent"], cfg["seed"],
    )


def load_existing_keys(results_path: Path) -> set[tuple]:
    if not results_path.exists():
        return set()
    with open(results_path, encoding="utf-8") as f:
        rows = json.load(f)
    keys: set[tuple] = set()
    for r in rows:
        if r.get("training") == "zero-shot":
            keys.add(("zeroshot", r.get("seed", 42)))
        else:
            mode = "flat" if r.get("training") == "fine-tuned" else "hierarchy"
            tax = r.get("taxonomy", "clean")
            keys.add((
                mode, tax,
                r.get("noise_fraction", 0.0),
                r.get("lambda_parent", 0.4),
                r.get("seed", 42),
            ))
    return keys


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run experiment sweeps for paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("sweep", choices=["baselines", "noise", "lambda", "all"])
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    p.add_argument("--resume", action="store_true", help="Skip configs already in results.json.")
    args = p.parse_args()

    configs = build_configs(args.sweep, seeds=args.seeds)
    results_path = Path(args.runs_dir) / "results.json"
    common = {"runs_dir": args.runs_dir, "epochs": args.epochs, "batch_size": args.batch_size}

    skipped = 0
    if args.resume:
        existing = load_existing_keys(results_path)
        filtered = []
        for c in configs:
            if _run_key(c) in existing:
                skipped += 1
            else:
                filtered.append(c)
        configs = filtered

    total = len(configs)
    print(f"Sweep: {args.sweep} | {total} runs" + (f" ({skipped} skipped)" if skipped else ""))
    print()

    failed = 0
    for i, cfg in enumerate(configs, 1):
        cmd = config_to_cmd(cfg, common)
        if cfg["type"] == "zeroshot":
            desc = f"zeroshot s={cfg['seed']}"
        else:
            desc = (
                f"{cfg['mode']} {cfg['taxonomy']} "
                f"nf={cfg['noise_fraction']} lam={cfg['lambda_parent']} s={cfg['seed']}"
            )

        if args.dry_run:
            print(f"  [{i}/{total}] DRY: {' '.join(cmd)}")
            continue

        print(f"[{i}/{total}] {desc} ...", end=" ", flush=True)
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"FAILED ({elapsed:.0f}s)")
            err_lines = (result.stderr or result.stdout or "").strip().split("\n")
            for line in err_lines[-3:]:
                print(f"    {line}")
            failed += 1
        else:
            last_line = (result.stdout or "").strip().split("\n")[-1]
            print(f"OK ({elapsed:.0f}s) {last_line}")

    print(f"\nDone. {total - failed}/{total} succeeded.")


if __name__ == "__main__":
    main()
