#!/usr/bin/env python3
"""Plot session-scaling summary metrics from CSV.

Generates two figures:
1) L2 plot (final and best mean with +/- std bands)
2) Chamfer plot (final and best mean with +/- std bands)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def _read_summary_csv(csv_path: Path) -> dict[str, list[float]]:
    columns = {
        "n_sessions": [],
        "final_l2_mean_of_means": [],
        "final_l2_std_of_means": [],
        "best_l2_mean_of_means": [],
        "best_l2_std_of_means": [],
        "final_chamfer_mean_of_means": [],
        "final_chamfer_std_of_means": [],
        "best_chamfer_mean_of_means": [],
        "best_chamfer_std_of_means": [],
    }

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_sessions = int(row["n_sessions"])
            values = {
                "final_l2_mean_of_means": float(row["final_l2_mean_of_means"]),
                "final_l2_std_of_means": float(row["final_l2_std_of_means"]),
                "best_l2_mean_of_means": float(row["best_l2_mean_of_means"]),
                "best_l2_std_of_means": float(row["best_l2_std_of_means"]),
                "final_chamfer_mean_of_means": float(row["final_chamfer_mean_of_means"]),
                "final_chamfer_std_of_means": float(row["final_chamfer_std_of_means"]),
                "best_chamfer_mean_of_means": float(row["best_chamfer_mean_of_means"]),
                "best_chamfer_std_of_means": float(row["best_chamfer_std_of_means"]),
            }

            if any(math.isnan(v) for v in values.values()):
                continue

            columns["n_sessions"].append(n_sessions)
            for k, v in values.items():
                columns[k].append(v)

    if not columns["n_sessions"]:
        raise ValueError(f"No valid numeric rows found in {csv_path}")
    return columns


def _plot_metric(
    x: list[int],
    final_mean: list[float],
    final_std: list[float],
    best_mean: list[float],
    best_std: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # final_low = [m - s for m, s in zip(final_mean, final_std)]
    # final_high = [m + s for m, s in zip(final_mean, final_std)]
    best_low = [m - s for m, s in zip(best_mean, best_std)]
    best_high = [m + s for m, s in zip(best_mean, best_std)]

    # ax.plot(x, final_mean, marker="o", linewidth=2, label="Final mean")
    # ax.fill_between(x, final_low, final_high, alpha=0.2, label="Final +/- std")

    ax.plot(x, best_mean, marker="s", linewidth=2, label="Best mean")
    ax.fill_between(x, best_low, best_high, alpha=0.2, label="Best +/- std")

    ax.set_xlabel("Number of training sessions (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    default_summary = (
        "experiments/fig4_data_efficiency/results/"
        "session_scaling/20260225_171034/sparsh_vit/"
        "session_scaling_summary_full_full_20260225_171037.csv"
    )
    parser = argparse.ArgumentParser(
        description="Generate L2 and Chamfer plots from session-scaling summary CSV."
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=default_summary,
        help="Path to session_scaling_summary_*.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same folder as summary CSV).",
    )
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else summary_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _read_summary_csv(summary_csv)
    x = data["n_sessions"]

    l2_plot = out_dir / "session_scaling_l2.png"
    chamfer_plot = out_dir / "session_scaling_chamfer.png"

    _plot_metric(
        x=x,
        final_mean=data["final_l2_mean_of_means"],
        final_std=data["final_l2_std_of_means"],
        best_mean=data["best_l2_mean_of_means"],
        best_std=data["best_l2_std_of_means"],
        ylabel="L2 distance",
        title="Session Scaling - L2 vs Number of Sessions",
        out_path=l2_plot,
    )

    _plot_metric(
        x=x,
        final_mean=data["final_chamfer_mean_of_means"],
        final_std=data["final_chamfer_std_of_means"],
        best_mean=data["best_chamfer_mean_of_means"],
        best_std=data["best_chamfer_std_of_means"],
        ylabel="Chamfer distance",
        title="Session Scaling - Chamfer vs Number of Sessions",
        out_path=chamfer_plot,
    )

    print(f"Saved L2 plot: {l2_plot}")
    print(f"Saved Chamfer plot: {chamfer_plot}")


if __name__ == "__main__":
    main()
