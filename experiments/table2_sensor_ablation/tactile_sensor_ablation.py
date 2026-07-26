#!/usr/bin/env python3
"""
Tactile Sensor Ablation (Table IV — tab:tactile_ablation).

Run Sparsh ViT on SWR-A for every possible sensor subset (16 configurations).
Sensors: fu=0 (fixed up, serial 22), fd=1 (fixed down, 18), ru=2 (robot up, 23), rd=3 (robot down, 24).
Excluded sensors are zeroed out in the input tensor so architecture stays identical.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add src so that "from dlofeel_state.engine.train import train" works (run from repo root or src)
_src_dir = Path(__file__).resolve().parents[2] / "src"  # .../src
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

def _maybe_set_cublas_workspace_from_argv() -> None:
    """Pre-configure CuBLAS reproducibility env when deterministic mode is requested."""
    if "--deterministic" in sys.argv and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        print("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA operations.")


_maybe_set_cublas_workspace_from_argv()

from dlofeel_state.engine.train import train as train_main  # noqa: E402

DEFAULT_RUN_CONFIG = "src/dlofeel_state/configs/default.yaml"
DEFAULT_NORM_STATS = "src/dlofeel_state/configs/norm_stats_k16.yaml"


def _read_back_metrics(ckpt_dir: Path, row_index: int, k: int, subset_label: str) -> dict:
    """Read a trained subset's table row back from its checkpoint dir.

    Prefers the T0-A ``run_metrics.json`` (``{"best": {aggregate, per_session}}``,
    written for runs trained post-T0-A) so the row carries the per-session detail;
    falls back to the legacy 4-key ``best_metrics.json`` when ``run_metrics.json``
    is absent (older runs). A NaN row is returned when neither file exists.

    Args:
        ckpt_dir: the run's checkpoint dir (``out_dir/checkpoints/<exp_name>``).
        row_index: Table IV row index (carried through to the result).
        k: number of active sensors for this subset.
        subset_label: human-readable sensor subset label.

    Returns:
        A row dict with ``row,k,subset,val_l2_mean/std,val_chamfer_mean/std`` and,
        when sourced from ``run_metrics.json``, also ``aggregate`` + ``per_session``.
    """
    base = {"row": row_index, "k": k, "subset": subset_label}

    run_metrics_path = ckpt_dir / "run_metrics.json"
    if run_metrics_path.exists():
        with open(run_metrics_path, "r") as f:
            best = json.load(f)["best"]
        agg = best["aggregate"]
        # T0-A aggregate uses bare l2_mean/chamfer_mean; the table column is val_*.
        return {
            **base,
            "val_l2_mean": agg["l2_mean"],
            "val_l2_std": agg["l2_std"],
            "val_chamfer_mean": agg["chamfer_mean"],
            "val_chamfer_std": agg["chamfer_std"],
            "aggregate": agg,
            "per_session": best.get("per_session", {}),
        }

    best_metrics_path = ckpt_dir / "best_metrics.json"
    if best_metrics_path.exists():
        with open(best_metrics_path, "r") as f:
            m = json.load(f)
        return {
            **base,
            "val_l2_mean": m["val_l2_mean"],
            "val_l2_std": m["val_l2_std"],
            "val_chamfer_mean": m["val_chamfer_mean"],
            "val_chamfer_std": m["val_chamfer_std"],
        }

    return {
        **base,
        "val_l2_mean": float("nan"),
        "val_l2_std": float("nan"),
        "val_chamfer_mean": float("nan"),
        "val_chamfer_std": float("nan"),
    }

# Table IV rows: (k, subset_label, sensor_names_tuple, only_pose, active_sensors_list)
# active_sensors_list is None for "all four" (k=4), and we use only_pose=True for k=0.
TABLE_IV_SPEC = [
    (0, "— (Pose-Only)", (), True, None),
    (1, "fu", ("fu",), False, [0]),
    (1, "fd", ("fd",), False, [1]),
    (1, "ru", ("ru",), False, [2]),
    (1, "rd", ("rd",), False, [3]),
    (2, "fu, fd", ("fu", "fd"), False, [0, 1]),
    (2, "fu, ru", ("fu", "ru"), False, [0, 2]),
    (2, "fu, rd", ("fu", "rd"), False, [0, 3]),
    (2, "fd, ru", ("fd", "ru"), False, [1, 2]),
    (2, "fd, rd", ("fd", "rd"), False, [1, 3]),
    (2, "ru, rd", ("ru", "rd"), False, [2, 3]),
    (3, "fu, fd, ru", ("fu", "fd", "ru"), False, [0, 1, 2]),
    (3, "fu, fd, rd", ("fu", "fd", "rd"), False, [0, 1, 3]),
    (3, "fu, ru, rd", ("fu", "ru", "rd"), False, [0, 2, 3]),
    (3, "fd, ru, rd", ("fd", "ru", "rd"), False, [1, 2, 3]),
    (4, "fu, fd, ru, rd", ("fu", "fd", "ru", "rd"), False, None),  # all four
]


def run_single(
    row_index: int,
    k: int,
    subset_label: str,
    only_pose: bool,
    active_sensors: list[int] | None,
    args: argparse.Namespace,
) -> dict:
    """Run one ablation configuration and return metrics."""
    exp_name = f"tactile_ablation_3_{row_index}_{subset_label.replace(', ', '_').replace('— (Pose-Only)', 'pose')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints" / exp_name

    def _load_saved_best_or_nan() -> dict:
        return _read_back_metrics(ckpt_dir, row_index, k, subset_label)

    try:
        train_main(
            train_data_root=args.train_data_root,
            val_data_root=args.val_data_root,
            blank_ref_root=args.blank_ref_root,
            config_path=args.config_path,
            norm_stats_path=args.norm_stats_path,
            num_resample_points=args.num_resample_points,
            n_control_points=args.n_control_points,
            normalize=args.normalize,
            normalize_inputs=args.normalize_inputs,
            with_att=args.with_att,
            use_diff_images=args.use_diff_images,
            use_blank_ref=args.use_blank_ref,
            only_pose=only_pose,
            single_encoder=args.single_encoder,
            out_dir=args.out_dir,
            img_size_H=args.img_size_H,
            img_size_W=args.img_size_W,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            num_workers=args.num_workers,
            seed=args.seed,
            use_wandb=False,
            max_batches_per_epoch=args.max_batches_per_epoch,
            max_eval_batches=args.max_eval_batches,
            active_sensors=active_sensors,
            exp_name_override=exp_name,
            deterministic=args.deterministic,
            deterministic_warn_only=args.deterministic_warn_only,
            image_backbone_type=args.image_backbone_type,
        )
    except Exception as e:
        # Keep the ablation sweep running even if one row fails (e.g., NaN loss).
        print(f"[WARN] Run failed for row {row_index} ({subset_label}): {e}")
        saved = _load_saved_best_or_nan()
        if saved["val_l2_mean"] == saved["val_l2_mean"]:
            print(f"[INFO] Using last saved best metrics for row {row_index} ({subset_label}).")
        return saved

    return _load_saved_best_or_nan()


def print_table(results: list[dict]) -> None:
    """Print Table IV with best per k-group bolded."""
    # Find best (min val_l2_mean) per k
    by_k: dict[int, list[dict]] = {}
    for r in results:
        by_k.setdefault(r["k"], []).append(r)
    best_l2_per_k = {k: min(rows, key=lambda x: x["val_l2_mean"]) for k, rows in by_k.items()}

    print("\n" + "=" * 80)
    print("Table IV — Tactile sensor ablation (d_eval ± std, Chamfer)")
    print("=" * 80)
    print(f"{'#k':<4} {'Sensor Subset':<25} {'Status':<8} {'d_eval (L2) ± std':<22} {'Chamfer':<12}")
    print("-" * 80)
    for r in results:
        best = best_l2_per_k.get(r["k"]) is r
        l2_str = f"{r['val_l2_mean']:.4f} ± {r['val_l2_std']:.4f}"
        chamfer_str = f"{r['val_chamfer_mean']:.6f}"
        if best:
            l2_str = f"**{l2_str}**"
            chamfer_str = f"**{chamfer_str}**"
        status = "✓" if r["val_l2_mean"] == r["val_l2_mean"] else "—"
        print(f"{r['k']:<4} {r['subset']:<25} {status:<8} {l2_str:<22} {chamfer_str:<12}")
    print("=" * 80)
    print("(Best result per k-group in bold.)\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Tactile sensor ablation — Table IV (16 runs on SWR-A).")
    def add_bool_flag(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
        parser.add_argument(f"--{name}", dest=name, action="store_true", help=help_text)
        parser.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable {help_text.lower()}")
        parser.set_defaults(**{name: None})

    p.add_argument("--train_data_root", type=str,
                   default=None,
                   help="SWR-A train data (directory or packed H5).")
    p.add_argument("--val_data_root", type=str,
                   default=None,
                   help="SWR-A val/test data.")
    p.add_argument("--blank_ref_root", type=str,
                   default=None,
                   help="Blank reference images for DIGIT.")
    p.add_argument("--config_path", type=str,
                   default=DEFAULT_RUN_CONFIG,
                   help="Training config YAML.")
    p.add_argument("--norm_stats_path", type=str,
                   default=DEFAULT_NORM_STATS,
                   help="Normalization stats YAML.")
    p.add_argument("--out_dir", type=str, default="results/tactile_ablation",
                  help="Output directory; best_metrics.json per run in out_dir/checkpoints/<exp_name>/.")
    p.add_argument("--num_resample_points", type=int, default=16)
    p.add_argument("--n_control_points", type=int, default=16,
                   help="B-spline control-point count; must match the k16 base "
                        "(train.main reads this independently of num_resample_points).")
    add_bool_flag(p, "normalize", "Normalize targets.")
    add_bool_flag(p, "normalize_inputs", "Normalize model inputs.")
    add_bool_flag(p, "with_att", "Use attention mechanism.")
    add_bool_flag(p, "use_diff_images", "Use difference images as input.")
    add_bool_flag(p, "use_blank_ref", "Use blank reference image for diff images.")
    add_bool_flag(p, "single_encoder", "Use a single encoder for all digit images.")
    p.add_argument("--image_backbone_type", type=str, default=None,
                   choices=["sparsh_vit", "resnet18"],
                   help="Image backbone type override (default: use config).")
    p.add_argument("--img_size_H", type=int, default=None)
    p.add_argument("--img_size_W", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_batches_per_epoch", "--max_batches", dest="max_batches_per_epoch", type=int, default=0)
    p.add_argument("--max_eval_batches", type=int, default=0)
    add_bool_flag(p, "deterministic", "Enable deterministic PyTorch behavior for reproducibility.")
    add_bool_flag(p, "deterministic_warn_only", "Warn (instead of error) on non-deterministic ops.")
    p.add_argument("--skip_rows", type=str, default="",
                   help="Comma-separated row indices (0..15) to skip, e.g. '0,15' to reuse Exp 1.1 and 1.3.")
    p.add_argument("--dry_run", action="store_true", help="Only print planned runs, do not train.")
    args = p.parse_args()

    skip_set = set()
    if args.skip_rows:
        skip_set = {int(x.strip()) for x in args.skip_rows.split(",") if x.strip()}

    if args.dry_run:
        print("Planned runs (Table IV):")
        for i, (k, subset_label, _names, only_pose, active_sensors) in enumerate(TABLE_IV_SPEC):
            skip = " [SKIP]" if i in skip_set else ""
            print(f"  {i}: k={k} {subset_label!r}  only_pose={only_pose}  active_sensors={active_sensors}{skip}")
        return

    results = []
    for i, (k, subset_label, _names, only_pose, active_sensors) in enumerate(TABLE_IV_SPEC):
        if i in skip_set:
            # Reuse an existing run's metrics (run_metrics.json-first, 4-key fallback).
            exp_name = f"tactile_ablation_3_{i}_{subset_label.replace(', ', '_').replace('— (Pose-Only)', 'pose')}"
            ckpt_dir = Path(args.out_dir) / "checkpoints" / exp_name
            results.append(_read_back_metrics(ckpt_dir, i, k, subset_label))
            continue
        print(f"\n--- Run {i+1}/16: k={k} {subset_label} ---")
        r = run_single(i, k, subset_label, only_pose, active_sensors, args)
        results.append(r)

    results.sort(key=lambda x: (x["k"], x["row"]))
    print_table(results)

    # Save results for later analysis
    out_path = Path(args.out_dir) / "table_iv_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
