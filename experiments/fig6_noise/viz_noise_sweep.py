#!/usr/bin/env python3
"""Visualize the tactile-noise blend: one image per noise level, clean -> pure white noise.

Pulls ONE real test sample (the normalized, diffed ``digit_images`` that actually
enter the model) and renders it under the SAME per-channel-matched blend the noise
ablation uses (``apply_noise_blend`` from ``eval_tactile_noise.py``), at
``--num_levels`` alpha values evenly spaced from 0 (clean) to 1 (pure white noise).
Writes ONE PNG per level so you can flip through clean -> white noise.

No checkpoint / model is needed: this visualizes the INPUT, not the model output.
It reuses ``apply_noise_blend`` / ``compute_channel_stats`` / ``effective_snr`` from
the eval so the rendered corruption matches the eval exactly (same per-channel
``(mu_c, sigma_c)``, same blend). Each per-level PNG shows the chosen sensors'
*current-diff* RGB (channels 0:3 of the 6) on a fixed display scale so the
progression reads honestly.

``--selftest`` renders a short sweep on a SYNTHETIC sample (no data/GPU/Sparsh) and
checks the PNGs are written, for an offline check while FAST_DATA is unmounted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add src (installed package) and this script's own dir (sibling eval module) to
# path so imports resolve when run from repo root or src.
_src_dir = Path(__file__).resolve().parents[2] / "src"  # .../src
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from eval_tactile_noise import (
    apply_noise_blend,
    compute_channel_stats,
    effective_snr,
    _load_config,
)

DEFAULT_CONFIG = "src/dlofeel_state/configs/default.yaml"
SENSOR_NAMES = ("fu", "fd", "ru", "rd")


# ----------------------------- Rendering -----------------------------


def _to_display_rgb(chw3: torch.Tensor, vmin: float, vmax: float) -> np.ndarray:
    """``[3,H,W]`` tensor -> ``[H,W,3]`` float in [0,1] via a fixed (vmin,vmax) clip."""
    img = chw3.detach().cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))  # H,W,3
    img = (img - vmin) / (vmax - vmin + 1e-8)
    return np.clip(img, 0.0, 1.0)


def render_level(
    blended: torch.Tensor,
    alpha: float,
    sensors: list[int],
    vmin: float,
    vmax: float,
    out_path: Path,
    level_idx: int,
    num_levels: int,
) -> None:
    """Save ONE PNG of `sensors`' current-diff RGB from a blended ``[1,4,6,H,W]`` tensor."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr, snr_db = effective_snr(alpha)
    n = len(sensors)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, sensors):
        rgb = _to_display_rgb(blended[0, s, 0:3], vmin, vmax)  # channels 0:3 = current-diff RGB
        ax.imshow(rgb)
        ax.set_title(SENSOR_NAMES[s], fontsize=9)
        ax.axis("off")
    snr_txt = "inf" if not np.isfinite(snr) else f"{snr:.2f}"
    db_txt = (("inf" if snr_db > 0 else "-inf") + " dB") if not np.isfinite(snr_db) else f"{snr_db:.1f} dB"
    tag = "clean" if alpha == 0 else ("pure white noise" if alpha == 1 else "blend")
    fig.suptitle(
        f"level {level_idx + 1}/{num_levels}   alpha={alpha:.3f}   SNR={snr_txt} ({db_txt})   [{tag}]",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------- Sweep core -----------------------------


def sweep_sample(
    imgs: torch.Tensor,
    mean_c: torch.Tensor,
    std_c: torch.Tensor,
    sensors: list[int],
    num_levels: int,
    out_dir: Path,
    seed: int,
) -> list[Path]:
    """Render `num_levels` PNGs (alpha 0->1) for one sample ``imgs`` ``[1,4,6,H,W]``.

    The RNG is re-seeded to `seed` before EACH level so the underlying white-noise
    pattern is identical across levels (only alpha grows) - same aligned-noise design
    as the eval, which makes the clean->noise progression read cleanly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fixed display range from the clean current-diff over the chosen sensors (1/99 pct,
    # robust to outliers). White noise at alpha=1 has std ~ sigma_c ~ signal std, so it
    # fills roughly the same range -> the progression is comparable across panels.
    clean = imgs[0, sensors, 0:3].detach().cpu().numpy()
    vmin = float(np.percentile(clean, 1))
    vmax = float(np.percentile(clean, 99))

    gen = torch.Generator(device=imgs.device)
    alphas = np.linspace(0.0, 1.0, num_levels)
    paths: list[Path] = []
    for i, a in enumerate(alphas):
        alpha = float(a)
        gen.manual_seed(seed)
        blended = apply_noise_blend(imgs, alpha, mean_c, std_c, generator=gen)  # [1,4,6,H,W]
        fp = out_dir / f"level_{i:02d}_alpha_{alpha:.3f}.png"
        render_level(blended, alpha, sensors, vmin, vmax, fp, i, num_levels)
        snr, snr_db = effective_snr(alpha)
        db_txt = ("inf" if snr_db > 0 else "-inf") if not np.isfinite(snr_db) else f"{snr_db:6.2f}"
        print(
            f"  level {i + 1:>2}/{num_levels}  alpha={alpha:.3f}  SNR_dB={db_txt}  "
            f"min/mean/max={blended.min():.3f}/{blended.mean():.3f}/{blended.max():.3f}  -> {fp.name}"
        )
        paths.append(fp)
    return paths


# ----------------------------- Selftest (synthetic; no data/GPU/Sparsh) -----------------------------


def _selftest() -> None:
    """Render a short synthetic sweep to a temp dir and assert the PNGs are written."""
    import tempfile

    torch.manual_seed(0)
    H = W = 24
    # Synthetic "sample": a smooth per-channel gradient so the clean image is structured.
    base = torch.linspace(-1.0, 1.0, H).view(1, 1, 1, H, 1).expand(1, 4, 6, H, W).clone()
    base = base + 0.1 * torch.randn(1, 4, 6, H, W)
    mean_c = torch.zeros(6)
    std_c = torch.ones(6)
    num_levels = 4
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        paths = sweep_sample(base, mean_c, std_c, sensors=[0, 1, 2, 3], num_levels=num_levels, out_dir=out, seed=42)
        assert len(paths) == num_levels, f"expected {num_levels} PNGs, got {len(paths)}"
        for p in paths:
            assert p.exists() and p.stat().st_size > 0, f"missing/empty PNG: {p}"
        # alpha=0 is clean passthrough; alpha=1 is pure white noise (signal gone).
        g = torch.Generator().manual_seed(42)
        clean = apply_noise_blend(base, 0.0, mean_c, std_c, generator=g)
        assert torch.equal(clean, base), "alpha=0 must be an exact clean passthrough"
        g.manual_seed(42)
        pure = apply_noise_blend(base, 1.0, mean_c, std_c, generator=g)
        assert not torch.equal(pure, base), "alpha=1 must replace the signal with white noise"
    print(f"selftest OK: rendered {num_levels} synthetic level PNGs; alpha=0 clean, alpha=1 pure noise.")


# ----------------------------- CLI -----------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualize the tactile-noise blend: one PNG per noise level (clean -> pure white noise)."
    )
    p.add_argument("--data_root", type=str, default="src/datasets/c1/ds/test",
                   help="Directory of sample*.h5 test files (or packed H5).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Optional: load this ckpt's sibling config.yaml for dataset params (no model is built).")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                   help="Fallback config when --ckpt is not given.")
    p.add_argument("--blank_ref_root", type=str, default="src/datasets/ds/digit_blank_closed_refs",
                   help="Digit blank-reference dir (local, mount-independent).")
    p.add_argument("--out_dir", type=str, default="results/tactile_noise_ablation/noise_sweep_viz",
                   help="Where the per-level PNGs are written.")
    p.add_argument("--num_levels", type=int, default=10,
                   help="Number of images / noise levels, alpha evenly spaced 0->1 inclusive (default 10).")
    p.add_argument("--sample_idx", type=int, default=0, help="Which test sample to visualize.")
    p.add_argument("--sensor", type=int, default=-1,
                   help="-1 = all 4 sensors per image; else a single sensor 0..3 (fu/fd/ru/rd).")
    p.add_argument("--stats_batches", type=int, default=4,
                   help="Batches for the per-channel (mu_c, sigma_c) pass (0 = full test set).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed; re-applied at each level (aligned noise).")
    p.add_argument("--selftest", action="store_true",
                   help="Render a synthetic sweep (no data/GPU/Sparsh) and exit 0.")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return

    # Lazy heavy imports (kept out of --help / --selftest).
    from torch.utils.data import DataLoader

    from dlofeel_state.data.dataset import DLOH5SampleDataset

    if args.ckpt:
        cfg_path = Path(args.ckpt).parent / "config.yaml"
        config = _load_config(str(cfg_path)) if cfg_path.exists() else _load_config(args.config)
    else:
        config = _load_config(args.config)
    config["blank_ref_root"] = args.blank_ref_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_packed = "packed" in args.data_root

    print(f"Loading dataset from {args.data_root} ...")
    ds = DLOH5SampleDataset(
        root_dir=args.data_root,
        norm_stats_path=config["norm_stats_path"],
        normalize_inputs=config.get("normalize_inputs", True),
        normalize=config["normalize"],
        blank_ref_root=args.blank_ref_root,
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
        ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 16),
        drop_last=False,
    )
    print(f"{len(ds)} test samples. Computing per-channel (mu_c, sigma_c) ...")
    mean_c, std_c = compute_channel_stats(loader, device, max_batches=args.stats_batches)
    print(f"  mu_c={[round(float(m), 4) for m in mean_c]}")
    print(f"  sigma_c={[round(float(s), 4) for s in std_c]}")

    if not (0 <= args.sample_idx < len(ds)):
        p.error(f"--sample_idx {args.sample_idx} out of range [0, {len(ds)})")
    sample = ds[args.sample_idx]
    imgs = sample["digit_images"].unsqueeze(0).to(device)  # [1, 4, 6, H, W]

    sensors = list(range(4)) if args.sensor < 0 else [args.sensor]
    out_dir = Path(args.out_dir)
    print(
        f"sample_idx={args.sample_idx}  sensors={[SENSOR_NAMES[s] for s in sensors]}  "
        f"levels={args.num_levels}  out={out_dir}"
    )
    paths = sweep_sample(imgs, mean_c, std_c, sensors, args.num_levels, out_dir, args.seed)
    print(f"Saved {len(paths)} images (level_00 clean -> level_{len(paths) - 1:02d} pure white noise) to {out_dir}")


if __name__ == "__main__":
    main()
