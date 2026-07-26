#!/usr/bin/env python3
"""
Drop-augmented training entry point for the C1 ``DLOStateModel``.

Trains the standard C1 model from scratch with sensor-drop augmentation
(``train.main(..., drop_augment=True)``): each train sample is expanded into 15
variants (real + all 1/2/3-sensor drops). Validation / ``best.pt`` track the
CLEAN test set (k=0). Hyperparameters mirror the clean C1 best run, so drop
augmentation is the single varying factor.

The resulting checkpoint loads directly into ``eval_tactile_drop.py`` for the
15-mask degradation comparison vs the clean baseline.

Usage:
    .venv/bin/python experiments/table2_sensor_ablation/train_tactile_drop.py \
        --train_data_root src/datasets/c1/ds/train \
        --val_data_root src/datasets/c1/ds/test \
        --epochs 10 --batch_size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parents[2] / "src"  # .../src
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from dlofeel_state.engine.train import train as train_main  # noqa: E402

DEFAULT_CONFIG = str(_src_dir / "dlofeel_state" / "configs" / "default.yaml")
DEFAULT_NORM_STATS = str(_src_dir / "dlofeel_state" / "configs" / "norm_stats_k16.yaml")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Drop-augmented C1 training: DLOStateModel with the 15-mask sensor-drop augmentation.",
    )
    p.add_argument("--train_data_root", type=str, default="src/datasets/c1/ds/train")
    p.add_argument("--val_data_root", type=str, default="src/datasets/c1/ds/test")
    p.add_argument("--blank_ref_root", type=str,
                   default="src/datasets/ds/digit_blank_closed_refs")
    p.add_argument("--config_path", type=str, default=DEFAULT_CONFIG)
    p.add_argument("--norm_stats_path", type=str, default=DEFAULT_NORM_STATS)
    p.add_argument("--out_dir", type=str, default="results/tactile_drop_train")

    p.add_argument("--num_resample_points", type=int, default=16)
    p.add_argument("--n_control_points", type=int, default=16,
                   help="B-spline control-point count; must match the k16 base "
                        "(train.main reads this independently of num_resample_points).")
    p.add_argument("--img_size_H", type=int, default=320)
    p.add_argument("--img_size_W", type=int, default=240)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=7.147008552233424e-5)
    p.add_argument("--weight_decay", type=float, default=1.0701368128884272e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_batches_per_epoch", type=int, default=0)
    p.add_argument("--max_eval_batches", type=int, default=0)

    p.add_argument("--wandb_project", type=str, default="dlo-state-estimation")
    p.add_argument("--wandb_entity", type=str, default="dlo")
    p.add_argument("--use_wandb", action="store_true")

    p.add_argument("--exp_name", type=str, default=None,
                   help="Override experiment name. Default: 'drop_aug'.")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--deterministic_warn_only", action="store_true")

    args = p.parse_args()

    exp_name = args.exp_name or "drop_aug"

    print("=" * 60)
    print("Drop-Augmented C1 Training")
    print("  Model:     DLOStateModel (Sparsh ViT)")
    print("  Augment:   15-mask sensor-drop enumeration (train only)")
    print(f"  Exp:       {exp_name}")
    print("=" * 60)

    train_main(
        train_data_root=args.train_data_root,
        val_data_root=args.val_data_root,
        blank_ref_root=args.blank_ref_root,
        config_path=args.config_path,
        norm_stats_path=args.norm_stats_path,
        num_resample_points=args.num_resample_points,
        n_control_points=args.n_control_points,
        normalize=True,
        normalize_inputs=True,
        with_att=True,
        use_diff_images=True,
        use_blank_ref=True,
        only_pose=False,
        single_encoder=True,
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
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        max_batches_per_epoch=args.max_batches_per_epoch,
        max_eval_batches=args.max_eval_batches,
        exp_name_override=exp_name,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
        drop_augment=True,
    )


if __name__ == "__main__":
    main()
