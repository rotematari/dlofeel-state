#!/usr/bin/env python3
"""Summarize the K-resample ablation: per-K best metrics -> CSV + plot.

Reads each K's ``best_metrics.json`` written by ``train.py`` and produces a
single two-panel figure (L2-vs-K, Chamfer-vs-K) plus a CSV table.

Caveat on the L2 column: absolute L2 is mildly confounded across K because the
ground-truth marker count changes with K (more output markers -> a different
per-marker distance scale). Chamfer is the more K-robust comparison since it is
a set-to-set distance; report both, but lean on Chamfer when ranking K.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed for a saved-figure script

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

DEFAULT_KS: list[int] = [8, 16, 32, 64, 128]


def read_k_metrics(results_root: Path, ks: list[int]) -> list[dict]:
    """Read per-K best metrics from the sweep output tree.

    For each K the metrics live at::

        results_root / f"k{K}" / "checkpoints" / f"k{K}" / "best_metrics.json"

    The doubled ``k{K}`` nesting is intentional: the sweep passes both
    ``--out_dir results/k_resample_ablation/k{K}`` and ``--exp_name k{K}``, and
    ``train.py`` writes under ``<out_dir>/checkpoints/<exp_name>/``.

    Missing K's are skipped with a warning rather than raising, so a partial
    sweep still summarizes.

    Args:
        results_root: Root directory holding the per-K ``k{K}`` subdirectories.
        ks: K values to look for.

    Returns:
        One row dict per K that was found, each with keys ``k``,
        ``val_l2_mean``, ``val_l2_std``, ``val_chamfer_mean``,
        ``val_chamfer_std``.
    """
    rows: list[dict] = []
    for k in ks:
        metrics_path = results_root / f"k{k}" / "checkpoints" / f"k{k}" / "best_metrics.json"
        if not metrics_path.is_file():
            print(f"WARNING: missing metrics for K={k} at {metrics_path}; skipping.")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append(
            {
                "k": k,
                "val_l2_mean": float(metrics["val_l2_mean"]),
                "val_l2_std": float(metrics["val_l2_std"]),
                "val_chamfer_mean": float(metrics["val_chamfer_mean"]),
                "val_chamfer_std": float(metrics["val_chamfer_std"]),
            }
        )
    return rows


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    """Write the per-K rows to CSV, sorted by K ascending.

    Args:
        rows: Row dicts as returned by :func:`read_k_metrics`.
        out_path: Destination CSV path.
    """
    fieldnames = ["k", "val_l2_mean", "val_l2_std", "val_chamfer_mean", "val_chamfer_std"]
    sorted_rows = sorted(rows, key=lambda r: r["k"])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)


def plot_summary(rows: list[dict], out_path: Path) -> None:
    """Plot L2-vs-K and Chamfer-vs-K side by side with +/- std bands.

    The x-axis is log2-scaled (K doubles each step) with the actual K values as
    ticks.

    Args:
        rows: Row dicts as returned by :func:`read_k_metrics`.
        out_path: Destination image path.
    """
    sorted_rows = sorted(rows, key=lambda r: r["k"])
    ks = [r["k"] for r in sorted_rows]

    fig, (ax_l2, ax_ch) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, mean_key, std_key, ylabel, title in (
        (ax_l2, "val_l2_mean", "val_l2_std", "Validation L2", "L2 vs K (confounded; see Chamfer)"),
        (ax_ch, "val_chamfer_mean", "val_chamfer_std", "Validation Chamfer", "Chamfer vs K"),
    ):
        means = [r[mean_key] for r in sorted_rows]
        stds = [r[std_key] for r in sorted_rows]
        low = [m - s for m, s in zip(means, stds)]
        high = [m + s for m, s in zip(means, stds)]

        ax.plot(ks, means, marker="o", linewidth=2, label="Best mean")
        ax.fill_between(ks, low, high, alpha=0.2, label="+/- std")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("K (resample points = control points)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the K-resample ablation into a CSV table and an L2/Chamfer-vs-K plot."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results/k_resample_ablation",
        help="Root directory holding the per-K k{K} subdirectories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for the CSV + PNG (default: same as --results_root).",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_k_metrics(results_root, DEFAULT_KS)
    if not rows:
        raise SystemExit(f"No best_metrics.json found under {results_root} for K in {DEFAULT_KS}.")

    csv_path = out_dir / "k_resample_summary.csv"
    png_path = out_dir / "k_resample_summary.png"
    write_summary_csv(rows, csv_path)
    plot_summary(rows, png_path)

    print(f"Read {len(rows)} of {len(DEFAULT_KS)} K runs.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
