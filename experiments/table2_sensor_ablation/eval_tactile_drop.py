#!/usr/bin/env python3
"""
Tactile Drop Ablation (missing-sensor robustness, test-only).

Load ONE pretrained C1 ``DLOStateModel`` checkpoint and measure graceful
degradation when 1-3 of the 4 DIGIT tactile sensors are missing at test time.
A "drop" zeros the dropped sensor's already-diffed ``digit_images`` slice
(sensor axis = dim 1) before the encoder, so the architecture is untouched.

Every test sample is evaluated under each of 15 masks (full + all 1/2/3-sensor
drops; the all-4 drop is excluded), aggregated per mask, and reported as L2
(``d_eval``) and Chamfer mean ± std (per-sample aggregation, denormalized
metric units). No training, no optimizer, no augmentation. The pose-only floor
is owned by ``model_comparison`` (different checkpoint) and is NOT computed here.

The mask layer is pure and dependency-free so ``--help`` and the unit probes
never import the dataset, the model, GPU code, or Sparsh. ``DROPPED_SENSOR_NAMES``
and ``enumerate_drop_masks`` live in ``drop_masks.py`` (shared with the train-time
wrapper) and are re-exported here; ``apply_drop`` (batched) is defined below.
``DLOStateModel``, ``DLOH5SampleDataset`` and the loss fns are lazy-imported
inside the eval core.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add src so that "from dlofeel_state..." resolves when run from repo root or src.
_src_dir = Path(__file__).resolve().parents[2] / "src"  # .../src
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Mask layer moved to the package (single source of truth, shared with the
# train-time wrapper). Re-exported here so existing probes that do
# `from ...eval_tactile_drop import enumerate_drop_masks` keep resolving.
from dlofeel_state.data.drop_masks import (  # noqa: E402
    DROPPED_SENSOR_NAMES,
    enumerate_drop_masks,
)

DEFAULT_CONFIG = "src/dlofeel_state/configs/default.yaml"


# ----------------------------- Pure mask layer -----------------------------
# ``DROPPED_SENSOR_NAMES`` and ``enumerate_drop_masks`` now live in drop_masks.py
# (imported above). ``apply_drop`` (batched) stays here.


def apply_drop(imgs: torch.Tensor, dropped_sensors: list[int], n_digits: int = 4) -> torch.Tensor:
    """Zero the dropped sensors' slices on a clone of ``imgs`` (input unmutated).

    Args:
        imgs: batched tactile tensor ``[B, n_digits, C, H, W]`` (sensor axis = dim 1).
        dropped_sensors: sensor indices in ``range(n_digits)`` to zero out.
        n_digits: number of tactile sensors (default 4); kept for signature
            parity with ``train.apply_active_sensors_mask``.

    Returns:
        A clone with rows ``i in dropped_sensors`` set to 0; other rows untouched.
    """
    # Inline clone-and-zero (mirrors train.apply_active_sensors_mask) so this
    # probe stays dependency-free — importing dlofeel_state.engine.train pulls heavy deps.
    out = imgs.clone()
    for i in dropped_sensors:
        out[:, i] = 0
    return out


# ----------------------------- Eval core -----------------------------


def _load_config(config_path: str) -> dict:
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_drops(
    data_root: str,
    config: dict,
    ckpt_path: str,
    out_dir: str = "results/tactile_drop_ablation",
    max_batches: int = 0,
) -> list[dict]:
    """Evaluate one checkpoint under all 15 drop masks and write the table.

    Mirrors ``evaluate.py``'s dataset/loader/model-load block: the model and
    checkpoint are loaded EXACTLY ONCE, then reused across all 15 masks (outer
    loop = masks, inner loop = DataLoader). Per-sample L2 and Chamfer are
    accumulated across batches and reduced to mean/std over the full per-sample
    vector (NOT mean-of-batch-means).

    Args:
        data_root: directory of ``sample*.h5`` test files (or packed H5).
        config: run config (from the ckpt-dir ``config.yaml`` or the default).
        ckpt_path: path to the ``best.pt`` checkpoint.
        out_dir: results root; writes ``<out_dir>/<ckpt-parent-name>/drop_table.json``.
        max_batches: if > 0, cap batches per mask (smoke runs); 0 = all.

    Returns:
        The 15 row dicts, each augmented with
        ``val_l2_mean/std`` and ``val_chamfer_mean/std``.
    """
    # Lazy imports keep --help and the unit probes free of torch-data/GPU/Sparsh.
    from torch.utils.data import DataLoader

    from dlofeel_state.data.dataset import DLOH5SampleDataset
    from dlofeel_state.engine.metrics import reduce_metrics
    from dlofeel_state.losses.chamfer import (
        chamfer_distance_per_sample,
        mean_point_distance_per_sample,
    )
    from dlofeel_state.models.dlo_state_model import DLOStateModel

    torch.manual_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = Path(ckpt_path).parent.name
    out_path = Path(out_dir) / exp_name
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_path}")
    print(f"Using device: {device}")

    is_packed = "packed" in data_root

    print("Loading dataset...")
    val_ds = DLOH5SampleDataset(
        root_dir=data_root,
        norm_stats_path=config["norm_stats_path"],
        normalize_inputs=config.get("normalize_inputs", True),
        normalize=config["normalize"],
        blank_ref_root=config["blank_ref_root"],
        use_diff_images=config["use_diff_images"],
        use_blank_ref=config["use_blank_ref"],
        num_resample_points=config["num_output_pts"],
        img_size_H=config["img_size_H"],
        img_size_W=config["img_size_W"],
        transform=None,
        is_packed=is_packed,
        past_window_steps=config.get("past_window_steps", 1),
        max_time_delta=config.get("max_time_delta", 0.13),
        device=device,
    )
    loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 16),
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=1,
    )
    print(f"loaded {len(val_ds)} test samples.")

    # Load model + checkpoint EXACTLY ONCE; reused across all 15 masks.
    model = DLOStateModel(config=config).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    normalize = config["normalize"]
    masks = enumerate_drop_masks()
    results: list[dict] = []

    with torch.no_grad():
        for mask in masks:
            dropped = mask["dropped_sensors"]
            l2_chunks: list[torch.Tensor] = []
            chamfer_chunks: list[torch.Tensor] = []
            pred_chunks: list[torch.Tensor] = []
            tgt_chunks: list[torch.Tensor] = []
            paths: list[str] = []

            for b_idx, batch in enumerate(loader):
                if max_batches and b_idx >= max_batches:
                    break
                poses = batch["ee_pose"].to(device)  # [B, 7]
                tgt = batch["marker_positions"].to(device)  # [B, n_markers, 3]
                imgs = batch["digit_images"].to(device)  # [B, 4, 6, H, W]

                imgs = apply_drop(imgs, dropped)
                pred = model(poses, imgs)

                if normalize:
                    pred_pts = val_ds.denormalize_sample(pred.cpu(), val_ds.marker_mean, val_ds.marker_std)
                    tgt_pts = val_ds.denormalize_sample(tgt.cpu(), val_ds.marker_mean, val_ds.marker_std)
                else:
                    pred_pts = pred.cpu()
                    tgt_pts = tgt.cpu()

                l2_chunks.append(mean_point_distance_per_sample(pred_pts, tgt_pts))
                chamfer_chunks.append(chamfer_distance_per_sample(pred_pts, tgt_pts, bidirectional=True))
                pred_chunks.append(pred_pts)
                tgt_chunks.append(tgt_pts)
                paths.extend(list(batch["path"]))

            l2 = torch.cat(l2_chunks)
            chamfer = torch.cat(chamfer_chunks)
            # Per-session detailed payload (RA-L deliverable). reduce_metrics
            # recomputes l2/chamfer internally on the same denorm tensors, so the
            # superset's aggregate matches val_* up to ddof (reducer uses pop-std).
            detailed = reduce_metrics(torch.cat(pred_chunks), torch.cat(tgt_chunks), paths)
            row = {
                **mask,
                "val_l2_mean": float(l2.mean().item()),
                "val_l2_std": float(l2.std().item()),
                "val_chamfer_mean": float(chamfer.mean().item()),
                "val_chamfer_std": float(chamfer.std().item()),
                "aggregate": detailed["aggregate"],
                "per_session": detailed["per_session"],
                "n_samples": detailed["n_samples"],
                "n_sessions": detailed["n_sessions"],
            }
            results.append(row)
            print(
                f"k={row['k']} [{row['drop_label']}]: "
                f"L2 {row['val_l2_mean']:.4f} ± {row['val_l2_std']:.4f}  "
                f"Chamfer {row['val_chamfer_mean']:.6f}"
            )

    print_table(results)

    json_path = out_path / "drop_table.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    return results


# ----------------------------- Output -----------------------------


def print_table(results: list[dict]) -> None:
    """Print the 15-row drop table (tactile_sensor_ablation style, best per k bold)."""
    by_k: dict[int, list[dict]] = {}
    for r in results:
        by_k.setdefault(r["k"], []).append(r)
    best_l2_per_k = {k: min(rows, key=lambda x: x["val_l2_mean"]) for k, rows in by_k.items()}

    print("\n" + "=" * 80)
    print("Tactile drop ablation — missing-sensor robustness (d_eval ± std, Chamfer)")
    print("=" * 80)
    print(f"{'k':<4} {'Dropped':<25} {'d_eval (L2) ± std':<22} {'Chamfer':<12}")
    print("-" * 80)
    for r in results:
        best = best_l2_per_k.get(r["k"]) is r
        l2_str = f"{r['val_l2_mean']:.4f} ± {r['val_l2_std']:.4f}"
        chamfer_str = f"{r['val_chamfer_mean']:.6f}"
        if best:
            l2_str = f"**{l2_str}**"
            chamfer_str = f"**{chamfer_str}**"
        print(f"{r['k']:<4} {r['drop_label']:<25} {l2_str:<22} {chamfer_str:<12}")
    print("=" * 80)
    print("(Best result per k-group in bold.)")
    print("(Pose-only floor not shown here — owned by the model_comparison experiment.)\n")


# ----------------------------- Self-test -----------------------------


def _selftest() -> None:
    """Probe the pure mask layer + the reduce_metrics integration; exit 0.

    Both checks run on synthetic CPU tensors with no data, GPU, model, or
    Sparsh: (1) the ``apply_drop`` / ``enumerate_drop_masks`` mask layer, and
    (2) the per-mask ``reduce_metrics`` wiring on a ``[N, K, 3]`` pred/tgt batch
    with two sessions. Raises ``AssertionError`` on failure.
    """
    from dlofeel_state.engine.metrics import reduce_metrics

    # 1. Mask layer: enumerate_drop_masks yields the 15 non-empty drops; apply_drop
    #    zeros exactly the dropped sensor rows on a clone (input unmutated).
    masks = enumerate_drop_masks()
    assert len(masks) == 15, f"expected 15 masks, got {len(masks)}"
    assert masks[0]["dropped_sensors"] == [], "first mask must be the full (no-drop) config"
    assert {m["k"] for m in masks} == {0, 1, 2, 3}, "masks must cover k in 0..3"

    imgs = torch.ones(2, 4, 6, 8, 8)
    dropped = [1, 3]
    out = apply_drop(imgs, dropped)
    assert out is not imgs and torch.equal(imgs, torch.ones_like(imgs)), "apply_drop mutated input"
    for i in range(4):
        if i in dropped:
            assert torch.count_nonzero(out[:, i]) == 0, f"sensor {i} not zeroed"
        else:
            assert torch.all(out[:, i] == 1.0), f"sensor {i} wrongly altered"

    # 2. reduce_metrics integration: 3 samples in session "s0", 1 in "s1".
    #    A constant per-session offset makes the per-session means hand-checkable.
    n, k = 4, 5
    tgt = torch.zeros(n, k, 3)
    pred = torch.zeros(n, k, 3)
    pred[:3, :, 0] = 1.0  # s0 samples: per-sample L2 == 1.0
    pred[3, :, 0] = 4.0   # s1 sample:  per-sample L2 == 4.0
    paths = [
        "/data/s0/sample0.h5",
        "/data/s0/sample1.h5",
        "/data/s0/sample2.h5",
        "/data/s1/sample0.h5",
    ]
    detailed = reduce_metrics(pred, tgt, paths)
    assert set(detailed["aggregate"]) >= {"l2_mean", "l2_std", "chamfer_mean", "chamfer_std"}, (
        f"aggregate missing keys: {detailed['aggregate'].keys()}"
    )
    assert detailed["n_samples"] == 4 and detailed["n_sessions"] == 2, "n_* wrong"
    assert set(detailed["per_session"]) == {"s0", "s1"}, "session grouping wrong"
    assert detailed["per_session"]["s0"]["n"] == 3 and detailed["per_session"]["s1"]["n"] == 1, (
        "per-session counts wrong"
    )
    assert abs(detailed["per_session"]["s0"]["l2_mean"] - 1.0) < 1e-5, "s0 L2 mean off"
    assert abs(detailed["per_session"]["s1"]["l2_mean"] - 4.0) < 1e-5, "s1 L2 mean off"
    # Global L2 mean = (3*1 + 1*4) / 4 = 1.75.
    assert abs(detailed["aggregate"]["l2_mean"] - 1.75) < 1e-5, "aggregate L2 mean off"

    print(
        "selftest OK: 15 drop masks; apply_drop zeros dropped sensors (input "
        "unmutated); reduce_metrics groups 2 sessions, per-session L2 s0=1.0 "
        "s1=4.0, aggregate=1.75."
    )


# ----------------------------- CLI -----------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tactile drop ablation — missing-sensor robustness (test-only, 15 masks)."
    )
    p.add_argument("--data_root", type=str, default="src/datasets/c1/ds/test",
                   help="Directory containing sample*.h5 (or packed H5).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to the C1 model checkpoint best.pt (omit only with --selftest).")
    p.add_argument("--out_dir", type=str, default="results/tactile_drop_ablation",
                   help="Output root; drop_table.json written under out_dir/<ckpt-parent-name>/.")
    p.add_argument("--blank_ref_root", type=str,
                   default="src/datasets/ds/digit_blank_closed_refs",
                   help="Path to digit blank reference images directory.")
    p.add_argument("--max_batches", type=int, default=0,
                   help="Cap batches per mask for smoke runs (0 = all).")
    p.add_argument("--selftest", action="store_true",
                   help="Run the synthetic mask-layer + reduce_metrics probes and exit 0.")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.ckpt is None:
        p.error("--ckpt is required unless --selftest is given")

    ckpt_dir = Path(args.ckpt).parent
    config_path = ckpt_dir / "config.yaml"
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        config = _load_config(str(config_path))
    else:
        print(f"Config not found at {config_path}, using default config")
        config = _load_config(DEFAULT_CONFIG)

    config["data_root"] = args.data_root
    config["blank_ref_root"] = args.blank_ref_root

    print(f"Evaluating experiment: {ckpt_dir.name}")
    evaluate_drops(
        data_root=args.data_root,
        config=config,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
