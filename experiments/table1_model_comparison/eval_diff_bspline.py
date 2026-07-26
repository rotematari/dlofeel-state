#!/usr/bin/env python3
"""
Differentiable B-spline output-head ablation (test-only).

Quantify what the B-spline output head buys. Two C1 checkpoints are trained
identically except ``model_params.use_spline`` (true=spline_on, false=spline_off,
see ``run_diff_bspline.sh``); this script evaluates ONE checkpoint per call and
``--combine`` prints the 2-arm comparison. Six metrics per arm:

  - L2 (``d_eval``) and Chamfer  — denormalized (physical, m), per-sample mean±std.
  - dMAC and CTV                 — denormalized curvature shape metrics
                                   (1/len), per-sample mean±std.
  - kb_loss and length_loss      — normalized-space WeightedLoss components
                                   (raw MSE), sample-weighted scalar each.

The normalized / denormalized split mirrors training: ``train.py`` computes the
WeightedLoss on the NORMALIZED model output, and only denormalizes for the val
L2/Chamfer metrics. So kb_loss/length_loss are reported in normalized MSE units
and L2/Chamfer/dMAC/CTV in denormalized metric units (each compared within
itself across arms; units are labeled in the JSON).

The pure smoothness layer (``mean_absolute_curvature``, ``curvature_total_variation``,
``smoothness_metrics``) depends only on torch + ``loss/der.py`` (which imports
only torch), so ``--help`` and ``--selftest`` never touch the dataset, the model,
GPU code, or Sparsh. ``DLOH5SampleDataset``, the per-sample metric fns,
``DLOStateModel``, ``WeightedLoss`` and ``yaml`` are lazy-imported inside the eval
core.
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

# der.py imports only torch / torch.nn.functional, so this top-level import keeps
# --help and --selftest free of dataset/model/GPU/Sparsh deps.
from dlofeel_state.losses.der import (  # noqa: E402  (after sys.path insert)
    compute_curvature_binormal,
    compute_edge_lengths,
    compute_edges,
)

DEFAULT_CONFIG = "src/dlofeel_state/configs/default.yaml"


# ----------------------------- Pure smoothness layer -----------------------------
# These three symbols are importable with only torch + der.py (no torch-data/GPU/Sparsh).


def mean_absolute_curvature(points: torch.Tensor) -> torch.Tensor:
    """Mean discrete curvature-binormal magnitude over all rows.

    Args:
        points: ordered markers ``[B, N, 3]`` (N=32 → kb is ``[B, 31, 3]``).

    Returns:
        ``[B]`` mean over ALL N-1 kb rows, including the structurally-zero
        ``kb[:, 0]`` row (consistent with how ``WeightedLoss`` averages kb).
    """
    edges = compute_edges(points)
    lengths = compute_edge_lengths(edges)
    kb = compute_curvature_binormal(edges, lengths)  # [B, N-1, 3], kb[:, 0] == 0
    return kb.norm(dim=-1).mean(dim=-1)


def curvature_total_variation(points: torch.Tensor) -> torch.Tensor:
    """Mean L2 magnitude of consecutive kb differences (curvature roughness).

    Args:
        points: ordered markers ``[B, N, 3]``.

    Returns:
        ``[B]`` mean over the N-2 consecutive kb differences.
    """
    edges = compute_edges(points)
    lengths = compute_edge_lengths(edges)
    kb = compute_curvature_binormal(edges, lengths)  # [B, N-1, 3]
    return (kb[:, 1:] - kb[:, :-1]).norm(dim=-1).mean(dim=-1)


def smoothness_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample smoothness metrics for a predicted vs ground-truth shape batch.

    Args:
        pred: predicted markers ``[B, N, 3]``.
        gt: ground-truth markers ``[B, N, 3]``.

    Returns:
        Dict of per-sample ``[B]`` tensors:
        ``dmac`` = MAC(pred) - MAC(gt); ``ctv`` = CTV(pred); plus ``mac_pred``,
        ``mac_gt`` for inspection.
    """
    mac_pred = mean_absolute_curvature(pred)
    mac_gt = mean_absolute_curvature(gt)
    return {
        "dmac": mac_pred - mac_gt,
        "ctv": curvature_total_variation(pred),
        "mac_pred": mac_pred,
        "mac_gt": mac_gt,
    }


# ----------------------------- Eval core -----------------------------


def _load_config(config_path: str) -> dict:
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_bspline_arm(
    data_root: str,
    config: dict,
    ckpt_path: str,
    out_dir: str = "results/diff_bspline_ablation",
    max_batches: int = 0,
    arm_name: str | None = None,
) -> dict:
    """Evaluate one arm's checkpoint and write ``<out_dir>/<arm>.json``.

    Mirrors the drop harness: dataset/loader/model loaded EXACTLY ONCE, then a
    single pass over the test loader. Per batch the loss-component metrics
    (kb_loss, length_loss) are computed in NORMALIZED space on the raw model
    output, and the geometric metrics (L2, Chamfer, dMAC, CTV) in DENORMALIZED
    space — matching how ``train.py`` splits its training loss vs val metrics.

    Args:
        data_root: directory of ``sample*.h5`` test files (or packed H5).
        config: run config (from the per-arm ckpt-dir ``config.yaml`` or default);
            recovers this arm's ``use_spline`` so the rebuilt model matches the ckpt.
        ckpt_path: path to the arm's ``best.pt``; the arm label defaults to
            ``Path(ckpt).parent.name``.
        out_dir: results root; writes ``<out_dir>/<arm>.json``.
        max_batches: if > 0, cap batches (smoke runs); 0 = all.
        arm_name: explicit arm label overriding ``Path(ckpt).parent.name``; sets
            both the JSON basename and the ``"arm"`` field. Used when the spline-on
            reference is the suite's k16 base checkpoint, whose parent dir is not
            ``spline_on`` but whose ``--combine`` input must still be ``spline_on.json``.

    Returns:
        The arm's metric dict (also written to JSON).
    """
    # Lazy imports keep --help and --selftest free of torch-data/GPU/Sparsh.
    from torch.utils.data import DataLoader

    from dlofeel_state.data.dataset import DLOH5SampleDataset
    from dlofeel_state.engine.metrics import reduce_metrics
    from dlofeel_state.losses.weighted import WeightedLoss
    from dlofeel_state.losses.chamfer import (
        chamfer_distance_per_sample,
        mean_point_distance_per_sample,
    )
    from dlofeel_state.models.dlo_state_model import DLOStateModel

    torch.manual_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arm = arm_name if arm_name is not None else Path(ckpt_path).parent.name
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_path}")
    print(f"arm: {arm}")
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

    # Load model + checkpoint EXACTLY ONCE. The per-arm config.yaml carries this
    # arm's use_spline, so the rebuilt head matches the checkpoint's state_dict.
    model = DLOStateModel(config=config).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # Same weights as training, so the normalized-space components match what was
    # optimized (loss_fc in train.py:678-681).
    wloss = WeightedLoss(
        w_nodes=config["w_nodes"],
        w_edges=config["w_edges"],
        w_lengths=config["w_lengths"],
        w_kb=config["w_kb"],
    )

    normalize = config["normalize"]

    l2_chunks: list[torch.Tensor] = []
    chamfer_chunks: list[torch.Tensor] = []
    dmac_chunks: list[torch.Tensor] = []
    ctv_chunks: list[torch.Tensor] = []
    pred_chunks: list[torch.Tensor] = []
    tgt_chunks: list[torch.Tensor] = []
    paths: list[str] = []
    # Sample-weighted accumulators for the normalized-space loss components.
    kb_loss_sum = 0.0
    length_loss_sum = 0.0
    n_samples = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches and b_idx >= max_batches:
                break
            poses = batch["ee_pose"].to(device)  # [B, 7]
            tgt = batch["marker_positions"].to(device)  # [B, n_markers, 3]
            imgs = batch["digit_images"].to(device)  # [B, 4, 6, H, W]

            pred = model(poses, imgs)
            bsz = pred.shape[0]

            # (a) NORMALIZED space: WeightedLoss components on the RAW model
            # output + batch target (no denorm), matching train.py:732's loss_fc.
            _, comps = wloss(pred, tgt, return_components=True)
            kb_loss_sum += float(comps["kb_mse"].item()) * bsz
            length_loss_sum += float(comps["lengths_mse"].item()) * bsz
            n_samples += bsz

            # (b) DENORMALIZED space: geometric + curvature-shape metrics.
            if normalize:
                pred_pts = val_ds.denormalize_sample(pred.cpu(), val_ds.marker_mean, val_ds.marker_std)
                tgt_pts = val_ds.denormalize_sample(tgt.cpu(), val_ds.marker_mean, val_ds.marker_std)
            else:
                pred_pts = pred.cpu()
                tgt_pts = tgt.cpu()

            l2_chunks.append(mean_point_distance_per_sample(pred_pts, tgt_pts))
            chamfer_chunks.append(chamfer_distance_per_sample(pred_pts, tgt_pts, bidirectional=True))
            smooth = smoothness_metrics(pred_pts, tgt_pts)
            dmac_chunks.append(smooth["dmac"])
            ctv_chunks.append(smooth["ctv"])
            pred_chunks.append(pred_pts)
            tgt_chunks.append(tgt_pts)
            paths.extend(list(batch["path"]))

    l2 = torch.cat(l2_chunks)
    chamfer = torch.cat(chamfer_chunks)
    dmac = torch.cat(dmac_chunks)
    ctv = torch.cat(ctv_chunks)

    # Per-session detailed payload (RA-L deliverable). dMAC/CTV are already
    # per-sample, so they flow through reduce_metrics' extra= channel and gain
    # <name>_mean/_std in both aggregate and every per_session group.
    detailed = reduce_metrics(
        torch.cat(pred_chunks),
        torch.cat(tgt_chunks),
        paths,
        extra={"dmac": dmac.cpu().numpy(), "ctv": ctv.cpu().numpy()},
    )

    result = {
        "arm": arm,
        "l2_mean": float(l2.mean().item()),
        "l2_std": float(l2.std().item()),
        "chamfer_mean": float(chamfer.mean().item()),
        "chamfer_std": float(chamfer.std().item()),
        "kb_loss": kb_loss_sum / n_samples,
        "length_loss": length_loss_sum / n_samples,
        "dmac_mean": float(dmac.mean().item()),
        "dmac_std": float(dmac.std().item()),
        "ctv_mean": float(ctv.mean().item()),
        "ctv_std": float(ctv.std().item()),
        "units": {
            "l2": "m (denorm)",
            "chamfer": "m (denorm)",
            "dmac": "1/len (denorm)",
            "ctv": "1/len (denorm)",
            "kb_loss": "normalized MSE",
            "length_loss": "normalized MSE",
        },
        "aggregate": detailed["aggregate"],
        "per_session": detailed["per_session"],
        "n_samples": detailed["n_samples"],
        "n_sessions": detailed["n_sessions"],
    }

    json_path = out_path / f"{arm}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[{arm}] L2 {result['l2_mean']:.4f} ± {result['l2_std']:.4f}  "
        f"Chamfer {result['chamfer_mean']:.6f}  "
        f"kb_loss {result['kb_loss']:.6e}  length_loss {result['length_loss']:.6e}  "
        f"dMAC {result['dmac_mean']:.4f} ± {result['dmac_std']:.4f}  "
        f"CTV {result['ctv_mean']:.4f} ± {result['ctv_std']:.4f}"
    )
    print(f"Results saved to {json_path}")

    return result


# ----------------------------- Combiner -----------------------------


def print_comparison(on_path: str, off_path: str) -> None:
    """Print the 2-row spline_on / spline_off comparison over the six metrics.

    Args:
        on_path: path to the spline_on arm JSON.
        off_path: path to the spline_off arm JSON.
    """
    with open(on_path, "r") as f:
        on = json.load(f)
    with open(off_path, "r") as f:
        off = json.load(f)

    print("\n" + "=" * 92)
    print("Diff B-spline ablation — output-head comparison")
    print("(L2/Chamfer/dMAC/CTV denormalized; kb_loss/length_loss normalized MSE)")
    print("=" * 92)
    header = (
        f"{'arm':<12} {'L2 (m)':<12} {'Chamfer (m)':<14} "
        f"{'kb_loss':<14} {'length_loss':<14} {'dMAC':<12} {'CTV':<12}"
    )
    print(header)
    print("-" * 92)
    for row in (on, off):
        print(
            f"{row['arm']:<12} "
            f"{row['l2_mean']:<12.4f} "
            f"{row['chamfer_mean']:<14.6f} "
            f"{row['kb_loss']:<14.6e} "
            f"{row['length_loss']:<14.6e} "
            f"{row['dmac_mean']:<12.4f} "
            f"{row['ctv_mean']:<12.4f}"
        )
    print("=" * 92 + "\n")


# ----------------------------- Self-test -----------------------------


def _selftest() -> None:
    """Probe the pure smoothness layer + the WeightedLoss component API; exit 0.

    Collinear points → MAC≈0, CTV≈0; a zig-zag → MAC>0, CTV>0. Also exercises
    ``WeightedLoss(...).forward(..., return_components=True)`` → (scalar, 4-key
    finite dict). Raises ``AssertionError`` on failure.
    """
    b, n = 4, 32

    # Collinear line along x → no curvature anywhere.
    t = torch.linspace(0.0, 1.0, n)
    line = torch.zeros(b, n, 3)
    line[:, :, 0] = t
    mac_line = mean_absolute_curvature(line)
    ctv_line = curvature_total_variation(line)
    assert torch.all(mac_line < 1e-5), f"collinear MAC not ~0: {mac_line}"
    assert torch.all(ctv_line < 1e-5), f"collinear CTV not ~0: {ctv_line}"

    # Zig-zag in the x-y plane → nonzero, finite curvature + roughness.
    zig = torch.zeros(b, n, 3)
    zig[:, :, 0] = t
    zig[:, :, 1] = torch.tensor([0.0, 0.1] * (n // 2))
    mac_zig = mean_absolute_curvature(zig)
    ctv_zig = curvature_total_variation(zig)
    assert torch.all(mac_zig > 0) and torch.all(torch.isfinite(mac_zig)), f"zigzag MAC bad: {mac_zig}"
    assert torch.all(ctv_zig > 0) and torch.all(torch.isfinite(ctv_zig)), f"zigzag CTV bad: {ctv_zig}"

    # WeightedLoss component-API probe.
    from dlofeel_state.losses.weighted import WeightedLoss

    wloss = WeightedLoss(w_nodes=1, w_edges=0.1, w_lengths=0.5, w_kb=0.01)
    total, comps = wloss(torch.randn(4, 32, 3), torch.randn(4, 32, 3), return_components=True)
    assert total.dim() == 0 and torch.isfinite(total), f"total not finite scalar: {total}"
    expected_keys = {"nodes_mse", "edges_mse", "lengths_mse", "kb_mse"}
    assert set(comps.keys()) == expected_keys, f"unexpected component keys: {comps.keys()}"
    assert all(torch.isfinite(v) for v in comps.values()), f"non-finite component: {comps}"

    # reduce_metrics integration: the dMAC/CTV per-sample arrays flow through the
    # extra= channel into aggregate AND every per_session group, hand-checkable.
    from dlofeel_state.engine.metrics import reduce_metrics

    n, npts = 4, 5
    tgt = torch.zeros(n, npts, 3)
    pred = torch.zeros(n, npts, 3)
    pred[:3, :, 0] = 1.0  # s0: per-sample L2 == 1.0
    pred[3, :, 0] = 4.0   # s1: per-sample L2 == 4.0
    paths = [
        "/data/s0/sample0.h5",
        "/data/s0/sample1.h5",
        "/data/s0/sample2.h5",
        "/data/s1/sample0.h5",
    ]
    dmac_arr = torch.tensor([2.0, 2.0, 2.0, 6.0])  # s0 mean 2.0, s1 mean 6.0, global 3.0
    ctv_arr = torch.tensor([0.5, 0.5, 0.5, 0.5])
    detailed = reduce_metrics(
        pred, tgt, paths, extra={"dmac": dmac_arr.numpy(), "ctv": ctv_arr.numpy()}
    )
    assert "dmac_mean" in detailed["aggregate"] and "ctv_mean" in detailed["aggregate"], (
        "extra metrics missing from aggregate"
    )
    assert abs(detailed["aggregate"]["dmac_mean"] - 3.0) < 1e-5, "aggregate dmac_mean off"
    assert set(detailed["per_session"]) == {"s0", "s1"}, "session grouping wrong"
    for sid in ("s0", "s1"):
        assert "dmac_mean" in detailed["per_session"][sid], f"{sid} missing per-session dmac_mean"
    assert abs(detailed["per_session"]["s0"]["dmac_mean"] - 2.0) < 1e-5, "s0 dmac_mean off"
    assert abs(detailed["per_session"]["s1"]["dmac_mean"] - 6.0) < 1e-5, "s1 dmac_mean off"
    assert abs(detailed["per_session"]["s0"]["l2_mean"] - 1.0) < 1e-5, "s0 L2 mean off"

    print(
        "selftest OK: collinear MAC/CTV ~0; zig-zag MAC/CTV > 0; WeightedLoss "
        "4-key dict finite; reduce_metrics extra dmac flows to aggregate (3.0) "
        "and per_session (s0=2.0 s1=6.0)."
    )


# ----------------------------- CLI -----------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Diff B-spline output-head ablation (test-only; per-arm eval + --combine table)."
    )
    p.add_argument("--data_root", type=str, default="src/datasets/c1/ds/test",
                   help="Directory containing sample*.h5 (or packed H5).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to the arm's checkpoint best.pt (omit when using --combine/--selftest).")
    p.add_argument("--out_dir", type=str, default="results/diff_bspline_ablation",
                   help="Output root; <arm>.json written under out_dir/.")
    p.add_argument("--arm_name", type=str, default=None,
                   help="Override the arm label (JSON basename + 'arm' field). "
                        "Default: the checkpoint's parent dir name. Use 'spline_on' "
                        "when the spline-on reference is the suite's k16 base ckpt.")
    p.add_argument("--blank_ref_root", type=str,
                   default="src/datasets/ds/digit_blank_closed_refs",
                   help="Path to digit blank reference images directory.")
    p.add_argument("--max_batches", type=int, default=0,
                   help="Cap batches for smoke runs (0 = all).")
    p.add_argument("--combine", type=str, nargs=2, default=None,
                   metavar=("ON_JSON", "OFF_JSON"),
                   help="Skip eval; print the 2-row spline_on/spline_off comparison from two JSONs.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the synthetic smoothness + return_components probes and exit 0.")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.combine is not None:
        print_comparison(args.combine[0], args.combine[1])
        return

    if args.ckpt is None:
        p.error("one of --ckpt, --combine, or --selftest is required")

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

    print(f"Evaluating arm: {args.arm_name or ckpt_dir.name}")
    evaluate_bspline_arm(
        data_root=args.data_root,
        config=config,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        max_batches=args.max_batches,
        arm_name=args.arm_name,
    )


if __name__ == "__main__":
    main()
