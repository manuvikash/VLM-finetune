#!/usr/bin/env python3
"""
Aggregate multi-seed experiment results into tables and CSVs for paper figures.

Usage:
  python analysis.py                              # print all tables
  python analysis.py --export-csv results/        # export to results/ directory
  python analysis.py --plot --plot-dir figures/   # generate matplotlib figures
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRIC_KEYS = [
    ("final_accuracy", "Leaf Acc (%)"),
    ("top5_accuracy", "Top-5 (%)"),
    ("final_parent_accuracy", "Parent Acc (%)"),
    ("hier_dist", "Hier Dist"),
    ("severity_weighted_acc", "Sev-W Acc (%)"),
    ("ece", "ECE"),
]


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _agg(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def _fmt(stats: dict[str, float], key: str) -> str:
    if stats["n"] == 0:
        return "---"
    m, s, n = stats["mean"], stats["std"], stats["n"]
    is_small = key in ("ece", "hier_dist")
    if n == 1:
        return f"{m:.4f}" if is_small else f"{m:.2f}"
    return f"{m:.4f} +/- {s:.4f}" if is_small else f"{m:.2f} +/- {s:.2f}"


# ---------- Table 1: Main baselines ----------

BASELINE_CONDITIONS = [
    ("clip_zeroshot", "Zero-shot"),
    ("clip_flat", "Flat fine-tune"),
    ("clip_hier_clean", "Hierarchy (clean)"),
    ("clip_hier_noisy", "Hierarchy (noisy 20%)"),
]


def print_baselines_table(rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rn = r.get("run_name")
        if rn:
            groups[rn].append(r)

    print("=" * 100)
    print("TABLE 1: Main Results (mean +/- std across seeds)")
    print("=" * 100)
    header = f"{'Condition':<25}" + "".join(f"{h:>15}" for _, h in METRIC_KEYS)
    print(header)
    print("-" * len(header))

    for run_name, label in BASELINE_CONDITIONS:
        rs = groups.get(run_name, [])
        if run_name == "clip_hier_noisy":
            rs = [
                r for r in rs
                if abs(r.get("noise_fraction", 0.2) - 0.2) < 0.01
                and r.get("noise_type", "uniform") == "uniform"
            ]
        cols = []
        for key, _ in METRIC_KEYS:
            vals = [r[key] for r in rs if key in r and isinstance(r[key], (int, float))]
            cols.append(_fmt(_agg(vals), key))
        print(f"{label:<25}" + "".join(f"{c:>15}" for c in cols))
    print()


# ---------- Table 2: Noise sweep ----------

def print_noise_sweep_table(rows: list[dict]) -> None:
    hier_rows = [r for r in rows if r.get("run_name", "").startswith("clip_hier")]
    if not hier_rows:
        return

    by_nf: dict[float, list[dict]] = defaultdict(list)
    for r in hier_rows:
        nf = r.get("noise_fraction", 0.0)
        by_nf[nf].append(r)

    if len(by_nf) < 2:
        return

    print("=" * 100)
    print("TABLE 2: Noise Fraction Sweep (all recorded hierarchy rows)")
    print("=" * 100)

    header = f"{'Noise Frac':<12}" + "".join(f"{h:>15}" for _, h in METRIC_KEYS) + f"{'N':>5}"
    print(header)
    print("-" * len(header))

    for nf in sorted(by_nf):
        rs = by_nf[nf]
        cols = []
        for key, _ in METRIC_KEYS:
            vals = [r[key] for r in rs if key in r and isinstance(r[key], (int, float))]
            cols.append(_fmt(_agg(vals), key))
        print(f"{nf:<12.2f}" + "".join(f"{c:>15}" for c in cols) + f"{len(rs):>5}")
    print()


# ---------- Table 2b: Noise type sweep ----------

def print_noise_type_sweep_table(rows: list[dict]) -> None:
    noisy_rows = [
        r for r in rows
        if r.get("run_name") == "clip_hier_noisy"
        and "noise_type" in r
    ]
    clean_rows = [
        r for r in rows
        if r.get("run_name") == "clip_hier_clean"
    ]
    if not noisy_rows:
        return

    by_key: dict[tuple[str, float], list[dict]] = defaultdict(list)
    noise_types = sorted({str(r.get("noise_type", "uniform")) for r in noisy_rows})
    for nt in noise_types:
        by_key[(nt, 0.0)].extend(clean_rows)
    for r in noisy_rows:
        by_key[(str(r.get("noise_type", "uniform")), r.get("noise_fraction", 0.0))].append(r)

    print("=" * 100)
    print("TABLE 2B: Noise Type Degradation (Leaf Accuracy)")
    print("=" * 100)

    nfs = sorted({nf for _, nf in by_key})
    header = f"{'Noise Type':<14}" + "".join(f"{'nf=' + f'{nf:.2f}':>15}" for nf in nfs)
    print(header)
    print("-" * len(header))
    for nt in noise_types:
        cols = []
        for nf in nfs:
            vals = [
                r["final_accuracy"] for r in by_key.get((nt, nf), [])
                if "final_accuracy" in r
            ]
            cols.append(_fmt(_agg(vals), "final_accuracy"))
        print(f"{nt:<14}" + "".join(f"{c:>15}" for c in cols))
    print()


# ---------- Table 3: Lambda sweep ----------

def print_lambda_sweep_table(rows: list[dict]) -> None:
    hier_rows = [r for r in rows if r.get("run_name", "").startswith("clip_hier")]
    if not hier_rows:
        return

    by_lam_nf: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for r in hier_rows:
        lam = r.get("lambda_parent", 0.4)
        nf = r.get("noise_fraction", 0.0)
        by_lam_nf[(lam, nf)].append(r)

    lambdas = sorted({k[0] for k in by_lam_nf})
    nfs = sorted({k[1] for k in by_lam_nf})
    if len(lambdas) < 2:
        return

    print("=" * 100)
    print("TABLE 3: Lambda x Noise Fraction (Leaf Accuracy)")
    print("=" * 100)

    header = f"{'Lambda':<10}" + "".join(f"{'nf=' + f'{nf:.2f}':>15}" for nf in nfs)
    print(header)
    print("-" * len(header))

    for lam in lambdas:
        cols = []
        for nf in nfs:
            rs = by_lam_nf.get((lam, nf), [])
            vals = [r["final_accuracy"] for r in rs if "final_accuracy" in r]
            cols.append(_fmt(_agg(vals), "final_accuracy"))
        print(f"{lam:<10.2f}" + "".join(f"{c:>15}" for c in cols))
    print()


# ---------- CSV export ----------

def export_csvs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_keys = [
        "run_name", "run_id", "pipeline", "training", "taxonomy",
        "seed", "noise_type", "noise_fraction", "lambda_parent",
    ] + [k for k, _ in METRIC_KEYS]

    with open(out_dir / "all_results.csv", "w", encoding="utf-8") as f:
        f.write(",".join(all_keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in all_keys) + "\n")

    hier_rows = [
        r for r in rows
        if r.get("run_name", "").startswith("clip_hier")
    ]
    by_nf: dict[float, list[dict]] = defaultdict(list)
    for r in hier_rows:
        by_nf[r.get("noise_fraction", 0.0)].append(r)

    if by_nf:
        mk = [k for k, _ in METRIC_KEYS]
        with open(out_dir / "noise_sweep.csv", "w", encoding="utf-8") as f:
            header = ["noise_fraction"] + [f"{k}_mean" for k in mk] + [f"{k}_std" for k in mk]
            f.write(",".join(header) + "\n")
            for nf in sorted(by_nf):
                rs = by_nf[nf]
                means, stds = [], []
                for k in mk:
                    vals = [r[k] for r in rs if k in r and isinstance(r[k], (int, float))]
                    a = _agg(vals)
                    means.append(f"{a['mean']:.6f}")
                    stds.append(f"{a['std']:.6f}")
                f.write(",".join([f"{nf:.4f}"] + means + stds) + "\n")

    noisy_rows = [
        r for r in rows
        if r.get("run_name") == "clip_hier_noisy" and "noise_type" in r
    ]
    clean_rows = [
        r for r in rows
        if r.get("run_name") == "clip_hier_clean"
    ]
    if noisy_rows:
        by_type_nf: dict[tuple[str, float], list[dict]] = defaultdict(list)
        noise_types = sorted({str(r.get("noise_type", "uniform")) for r in noisy_rows})
        for nt in noise_types:
            by_type_nf[(nt, 0.0)].extend(clean_rows)
        for r in noisy_rows:
            by_type_nf[(str(r.get("noise_type", "uniform")), r.get("noise_fraction", 0.0))].append(r)

        mk = [k for k, _ in METRIC_KEYS]
        with open(out_dir / "noise_type_sweep.csv", "w", encoding="utf-8") as f:
            header = ["noise_type", "noise_fraction", "n"]
            header += [f"{k}_mean" for k in mk] + [f"{k}_std" for k in mk]
            f.write(",".join(header) + "\n")
            for nt, nf in sorted(by_type_nf):
                rs = by_type_nf[(nt, nf)]
                means, stds = [], []
                for k in mk:
                    vals = [r[k] for r in rs if k in r and isinstance(r[k], (int, float))]
                    a = _agg(vals)
                    means.append(f"{a['mean']:.6f}")
                    stds.append(f"{a['std']:.6f}")
                f.write(",".join([nt, f"{nf:.4f}", str(len(rs))] + means + stds) + "\n")

    print(f"Exported CSVs to {out_dir}/")


# ---------- Matplotlib plots ----------

def generate_plots(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Noise degradation curve
    hier_rows = [
        r for r in rows
        if r.get("run_name", "").startswith("clip_hier")
    ]
    by_nf: dict[float, list[float]] = defaultdict(list)
    by_nf_par: dict[float, list[float]] = defaultdict(list)
    for r in hier_rows:
        nf = r.get("noise_fraction", 0.0)
        if "final_accuracy" in r:
            by_nf[nf].append(r["final_accuracy"])
        if "final_parent_accuracy" in r:
            by_nf_par[nf].append(r["final_parent_accuracy"])

    if len(by_nf) >= 2:
        nfs = sorted(by_nf)
        leaf_m = [np.mean(by_nf[n]) for n in nfs]
        leaf_s = [np.std(by_nf[n], ddof=1) if len(by_nf[n]) > 1 else 0 for n in nfs]
        par_nfs = [n for n in nfs if n in by_nf_par]
        par_m = [np.mean(by_nf_par[n]) for n in par_nfs]
        par_s = [np.std(by_nf_par[n], ddof=1) if len(by_nf_par[n]) > 1 else 0 for n in par_nfs]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(nfs, leaf_m, yerr=leaf_s, marker="o", capsize=4, label="Leaf Accuracy")
        if par_m:
            ax.errorbar(par_nfs, par_m, yerr=par_s, marker="s", capsize=4, label="Parent Accuracy")
        ax.set_xlabel("Taxonomy Noise Fraction")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Hierarchy-Aware Training: Noise Robustness")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "noise_degradation.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {out_dir / 'noise_degradation.png'}")

    # Plot 1b: Noise degradation by corruption type
    noisy_rows = [
        r for r in rows
        if r.get("run_name") == "clip_hier_noisy" and "noise_type" in r
    ]
    clean_rows = [
        r for r in rows
        if r.get("run_name") == "clip_hier_clean"
    ]
    if noisy_rows:
        noise_types = sorted({str(r.get("noise_type", "uniform")) for r in noisy_rows})
        fig, ax = plt.subplots(figsize=(8, 5))
        for nt in noise_types:
            by_nf: dict[float, list[float]] = defaultdict(list)
            by_nf[0.0].extend([r["final_accuracy"] for r in clean_rows if "final_accuracy" in r])
            for r in noisy_rows:
                if r.get("noise_type") == nt and "final_accuracy" in r:
                    by_nf[r.get("noise_fraction", 0.0)].append(r["final_accuracy"])
            nfs = sorted(n for n, vals in by_nf.items() if vals)
            means = [np.mean(by_nf[n]) for n in nfs]
            stds = [np.std(by_nf[n], ddof=1) if len(by_nf[n]) > 1 else 0 for n in nfs]
            ax.errorbar(nfs, means, yerr=stds, marker="o", capsize=4, label=nt)
        ax.set_xlabel("Taxonomy Noise Fraction")
        ax.set_ylabel("Leaf Accuracy (%)")
        ax.set_title("Degradation by Taxonomy Noise Type")
        ax.legend(title="Noise type")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "noise_degradation_by_type.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {out_dir / 'noise_degradation_by_type.png'}")

        # Plot 1c: Calibration degradation by corruption type
        fig, ax = plt.subplots(figsize=(8, 5))
        for nt in noise_types:
            by_nf: dict[float, list[float]] = defaultdict(list)
            by_nf[0.0].extend([r["ece"] for r in clean_rows if "ece" in r])
            for r in noisy_rows:
                if r.get("noise_type") == nt and "ece" in r:
                    by_nf[r.get("noise_fraction", 0.0)].append(r["ece"])
            nfs = sorted(n for n, vals in by_nf.items() if vals)
            means = [np.mean(by_nf[n]) for n in nfs]
            stds = [np.std(by_nf[n], ddof=1) if len(by_nf[n]) > 1 else 0 for n in nfs]
            ax.errorbar(nfs, means, yerr=stds, marker="o", capsize=4, label=nt)
        ax.set_xlabel("Taxonomy Noise Fraction")
        ax.set_ylabel("Expected Calibration Error")
        ax.set_title("Calibration Degradation by Noise Type")
        ax.legend(title="Noise type")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "ece_by_noise_type.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {out_dir / 'ece_by_noise_type.png'}")

        # Plot 1d: Hierarchical distance by corruption type
        fig, ax = plt.subplots(figsize=(8, 5))
        for nt in noise_types:
            by_nf: dict[float, list[float]] = defaultdict(list)
            by_nf[0.0].extend([r["hier_dist"] for r in clean_rows if "hier_dist" in r])
            for r in noisy_rows:
                if r.get("noise_type") == nt and "hier_dist" in r:
                    by_nf[r.get("noise_fraction", 0.0)].append(r["hier_dist"])
            nfs = sorted(n for n, vals in by_nf.items() if vals)
            means = [np.mean(by_nf[n]) for n in nfs]
            stds = [np.std(by_nf[n], ddof=1) if len(by_nf[n]) > 1 else 0 for n in nfs]
            ax.errorbar(nfs, means, yerr=stds, marker="o", capsize=4, label=nt)
        ax.set_xlabel("Taxonomy Noise Fraction")
        ax.set_ylabel("Mean Hierarchical Distance")
        ax.set_title("Semantic Error Severity by Noise Type")
        ax.legend(title="Noise type")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "hier_distance_by_noise_type.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {out_dir / 'hier_distance_by_noise_type.png'}")

        # Plot 1e: Accuracy drop relative to clean hierarchy baseline
        clean_vals = [r["final_accuracy"] for r in clean_rows if "final_accuracy" in r]
        clean_mean = float(np.mean(clean_vals)) if clean_vals else float("nan")
        fig, ax = plt.subplots(figsize=(8, 5))
        for nt in noise_types:
            by_nf: dict[float, list[float]] = defaultdict(list)
            by_nf[0.0].append(clean_mean)
            for r in noisy_rows:
                if r.get("noise_type") == nt and "final_accuracy" in r:
                    by_nf[r.get("noise_fraction", 0.0)].append(r["final_accuracy"])
            nfs = sorted(n for n, vals in by_nf.items() if vals)
            means = [np.mean(by_nf[n]) - clean_mean for n in nfs]
            stds = [np.std(by_nf[n], ddof=1) if len(by_nf[n]) > 1 else 0 for n in nfs]
            ax.errorbar(nfs, means, yerr=stds, marker="o", capsize=4, label=nt)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_xlabel("Taxonomy Noise Fraction")
        ax.set_ylabel("Leaf Accuracy Change vs. Clean (pp)")
        ax.set_title("Accuracy Degradation Relative to Clean Hierarchy")
        ax.legend(title="Noise type")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_delta_by_noise_type.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {out_dir / 'accuracy_delta_by_noise_type.png'}")

        # Plot 1f: Leaf accuracy vs. parent accuracy
        fig, ax = plt.subplots(figsize=(7, 5))
        markers = {"uniform": "o", "in_domain": "s", "cyclic": "^"}
        for nt in noise_types:
            xs = [
                r["final_parent_accuracy"] for r in noisy_rows
                if r.get("noise_type") == nt and "final_parent_accuracy" in r
            ]
            ys = [
                r["final_accuracy"] for r in noisy_rows
                if r.get("noise_type") == nt and "final_accuracy" in r
            ]
            ax.scatter(xs, ys, alpha=0.75, s=55, marker=markers.get(nt, "o"), label=nt)
        clean_parent = [r["final_parent_accuracy"] for r in clean_rows if "final_parent_accuracy" in r]
        clean_leaf = [r["final_accuracy"] for r in clean_rows if "final_accuracy" in r]
        if clean_parent and clean_leaf:
            ax.scatter(
                [np.mean(clean_parent)],
                [np.mean(clean_leaf)],
                color="black",
                marker="*",
                s=160,
                label="clean mean",
                zorder=5,
            )
        ax.set_xlabel("Parent Accuracy (%)")
        ax.set_ylabel("Leaf Accuracy (%)")
        ax.set_title("Leaf Accuracy vs. Parent Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "leaf_vs_parent_accuracy.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {out_dir / 'leaf_vs_parent_accuracy.png'}")

    # Plot 2: Main comparison bar chart
    conds = [
        ("clip_zeroshot", "Zero-shot"),
        ("clip_flat", "Flat"),
        ("clip_hier_clean", "Hier (clean)"),
        ("clip_hier_noisy", "Hier (noisy)"),
    ]
    grp: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        rn = r.get("run_name")
        if rn and "final_accuracy" in r:
            if rn == "clip_hier_noisy":
                if (abs(r.get("noise_fraction", 0.2) - 0.2) < 0.01
                        and r.get("noise_type", "uniform") == "uniform"):
                    grp[rn].append(r["final_accuracy"])
            else:
                grp[rn].append(r["final_accuracy"])

    if grp:
        labels, means, stds = [], [], []
        colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
        for rn, display in conds:
            if rn in grp:
                labels.append(display)
                means.append(np.mean(grp[rn]))
                stds.append(np.std(grp[rn], ddof=1) if len(grp[rn]) > 1 else 0)

        if labels:
            fig, ax = plt.subplots(figsize=(7, 5))
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=5, color=colors[: len(labels)])
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Leaf Accuracy (%)")
            ax.set_title("Model Comparison")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "comparison_bar.png", dpi=150)
            plt.close(fig)
            print(f"  Saved {out_dir / 'comparison_bar.png'}")


def generate_parent_heatmap(per_parent_csv: Path, out_dir: Path, *, noise_type: str = "cyclic") -> None:
    if not per_parent_csv.exists():
        print(f"No per-parent CSV at {per_parent_csv}; skipping heatmap.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping heatmap.")
        return

    rows: list[dict] = []
    with open(per_parent_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    clean = [r for r in rows if r.get("run_name") == "clip_hier_clean"]
    noisy = [
        r for r in rows
        if r.get("run_name") == "clip_hier_noisy" and r.get("noise_type") == noise_type
    ]
    if not noisy:
        print(f"No per-parent rows for noise_type={noise_type}; skipping heatmap.")
        return

    by_parent_nf: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in clean:
        by_parent_nf[(r["parent"], 0.0)].append(float(r["leaf_accuracy"]))
    for r in noisy:
        by_parent_nf[(r["parent"], float(r["noise_fraction"]))].append(float(r["leaf_accuracy"]))

    parents = sorted({p for p, _ in by_parent_nf})
    nfs = sorted({nf for _, nf in by_parent_nf})
    matrix = np.full((len(parents), len(nfs)), np.nan)
    for i, parent in enumerate(parents):
        for j, nf in enumerate(nfs):
            vals = by_parent_nf.get((parent, nf), [])
            if vals:
                matrix[i, j] = np.mean(vals)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=25, vmax=90)
    ax.set_xticks(np.arange(len(nfs)))
    ax.set_xticklabels([f"{nf:.2f}" for nf in nfs], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(parents)))
    ax.set_yticklabels([p.replace("_", " ") for p in parents], fontsize=8)
    ax.set_xlabel("Taxonomy Noise Fraction")
    ax.set_title(f"Per-Parent Leaf Accuracy Heatmap ({noise_type} noise)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Leaf Accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_dir / f"per_parent_heatmap_{noise_type}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / f'per_parent_heatmap_{noise_type}.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze experiment results for paper figures.")
    p.add_argument("--results-json", type=Path, default=Path("runs/results.json"))
    p.add_argument(
        "--export-csv", type=Path, default=None, metavar="DIR",
        help="Export aggregated CSVs to directory.",
    )
    p.add_argument("--plot", action="store_true", help="Generate matplotlib figures.")
    p.add_argument("--plot-dir", type=Path, default=Path("figures"))
    p.add_argument("--per-parent-csv", type=Path, default=Path("runs/per_parent_metrics.csv"))
    args = p.parse_args()

    rows = load_results(args.results_json)
    print(f"Loaded {len(rows)} results from {args.results_json}\n")

    print_baselines_table(rows)
    print_noise_sweep_table(rows)
    print_noise_type_sweep_table(rows)
    print_lambda_sweep_table(rows)

    if args.export_csv is not None:
        export_csvs(rows, args.export_csv)

    if args.plot:
        generate_plots(rows, args.plot_dir)
        generate_parent_heatmap(args.per_parent_csv, args.plot_dir, noise_type="cyclic")


if __name__ == "__main__":
    main()
