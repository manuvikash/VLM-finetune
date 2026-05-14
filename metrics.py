#!/usr/bin/env python3
"""Extended evaluation metrics for hierarchy-aware CLIP classification."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Percentage of samples where the true label is in the top-k predictions."""
    if logits.size(0) == 0:
        return 0.0
    k = min(k, logits.size(1))
    _, topk = logits.topk(k, dim=1)
    return 100.0 * topk.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item()


def hierarchical_distance(
    pred_leaves: list[int],
    true_leaves: list[int],
    id_to_leaf: list[str],
    leaf_to_parent: dict[str, str],
    parent_to_domain: dict[str, str],
) -> float:
    """
    Mean tree distance between predicted and true leaves.

    3-level tree (root -> domain -> parent -> leaf):
      same leaf = 0, same parent = 2, same domain = 4, different domain = 6.
    """
    if not pred_leaves:
        return 0.0
    total = 0.0
    for p, t in zip(pred_leaves, true_leaves):
        if p == t:
            continue
        pp = leaf_to_parent[id_to_leaf[p]]
        tp = leaf_to_parent[id_to_leaf[t]]
        if pp == tp:
            total += 2
        elif parent_to_domain.get(pp) == parent_to_domain.get(tp):
            total += 4
        else:
            total += 6
    return total / len(pred_leaves)


def severity_weighted_accuracy(
    pred_leaves: list[int],
    true_leaves: list[int],
    id_to_leaf: list[str],
    leaf_to_parent: dict[str, str],
) -> float:
    """1.0 correct, 0.5 same-parent wrong leaf, 0.0 otherwise. Returns percentage."""
    if not pred_leaves:
        return 0.0
    score = 0.0
    for p, t in zip(pred_leaves, true_leaves):
        if p == t:
            score += 1.0
        elif leaf_to_parent[id_to_leaf[p]] == leaf_to_parent[id_to_leaf[t]]:
            score += 0.5
    return 100.0 * score / len(pred_leaves)


def per_class_accuracy(
    pred_leaves: list[int],
    true_leaves: list[int],
    id_to_leaf: list[str],
    leaf_to_parent: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Per-leaf and per-parent accuracy breakdown."""
    leaf_c: dict[str, int] = defaultdict(int)
    leaf_t: dict[str, int] = defaultdict(int)
    par_c: dict[str, int] = defaultdict(int)
    par_t: dict[str, int] = defaultdict(int)

    for p, t in zip(pred_leaves, true_leaves):
        tn = id_to_leaf[t]
        tp = leaf_to_parent[tn]
        leaf_t[tn] += 1
        par_t[tp] += 1
        if p == t:
            leaf_c[tn] += 1
        if leaf_to_parent[id_to_leaf[p]] == tp:
            par_c[tp] += 1

    return {
        "leaf_per_class": {
            n: 100.0 * leaf_c.get(n, 0) / leaf_t[n] for n in sorted(leaf_t)
        },
        "parent_per_class": {
            n: 100.0 * par_c.get(n, 0) / par_t[n] for n in sorted(par_t)
        },
    }


def expected_calibration_error(
    logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> float:
    """ECE with equal-width confidence bins."""
    if logits.size(0) == 0:
        return 0.0
    probs = torch.softmax(logits.float(), dim=1)
    confs, preds = probs.max(dim=1)
    accs = preds.eq(targets).float()
    c = confs.cpu().numpy()
    a = accs.cpu().numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (c > lo) & (c <= hi)
        if not mask.any():
            continue
        ece += mask.sum() * abs(a[mask].mean() - c[mask].mean())
    return float(ece / len(c))


def compute_all_metrics(
    all_logits: torch.Tensor,
    all_targets: torch.Tensor,
    id_to_leaf: list[str],
    leaf_to_parent: dict[str, str],
    parent_to_domain: dict[str, str],
    *,
    all_parent_logits: torch.Tensor | None = None,
    all_parent_targets: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Full metric suite from accumulated predictions.

    Returns flat dict of scalar metrics (no per-class, to keep results.json compact).
    Call per_class_accuracy() separately for detailed breakdowns.
    """
    preds = all_logits.argmax(dim=1).tolist()
    trues = all_targets.tolist()
    n = max(len(preds), 1)

    leaf_acc = 100.0 * sum(p == t for p, t in zip(preds, trues)) / n
    top5 = top_k_accuracy(all_logits, all_targets, k=5)

    if all_parent_logits is not None and all_parent_targets is not None:
        pp = all_parent_logits.argmax(dim=1)
        parent_acc = 100.0 * (pp == all_parent_targets).float().mean().item()
    else:
        ok = sum(
            int(leaf_to_parent[id_to_leaf[p]] == leaf_to_parent[id_to_leaf[t]])
            for p, t in zip(preds, trues)
        )
        parent_acc = 100.0 * ok / n

    return {
        "leaf_acc": leaf_acc,
        "top5_acc": top5,
        "parent_acc": parent_acc,
        "hier_dist": hierarchical_distance(
            preds, trues, id_to_leaf, leaf_to_parent, parent_to_domain
        ),
        "severity_weighted_acc": severity_weighted_accuracy(
            preds, trues, id_to_leaf, leaf_to_parent
        ),
        "ece": expected_calibration_error(all_logits, all_targets),
    }
