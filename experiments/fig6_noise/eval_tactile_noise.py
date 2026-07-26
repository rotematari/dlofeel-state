#!/usr/bin/env python3
"""
Tactile Noise Ablation (clean<->white-noise blend robustness, test-only).

Load ONE pretrained C1 ``DLOStateModel`` checkpoint and measure how the
marker-prediction error grows as the (already-diffed, normalized)
``digit_images`` are progressively replaced by per-channel-matched white noise
just before the encoder. The architecture is untouched.

Every test sample is evaluated under each of 11 levels
``p in {0, 10, ..., 100}`` (percent), mapped to a blend fraction
``alpha = p / 100``. The corruption is a convex blend

    out = (1 - alpha) * signal + alpha * eps

where ``eps[..., c, ...] ~ N(mu_c, sigma_c^2)`` is i.i.d. white noise whose
per-channel moments ``(mu_c, sigma_c)`` match the empirical mean/std of the
real diffed-normalized ``digit_images`` over the test set (6 channels = dim 2 of
``[B, 4, 6, H, W]``, pooled over sensor + spatial). alpha=0 is an exact clean
passthrough; alpha=1 replaces the signal entirely with signal-matched white
noise (a true "no-tactile-information" floor, NOT the additive p=100 of the old
design, NOT the pose-only floor owned by ``model_comparison``).

Because ``eps`` is drawn at the empirical signal std ``sigma_c``, the designed
per-channel signal/noise power ratio is channel-independent:
``snr = ((1 - alpha) / alpha)^2`` (``snr_db = 20*log10((1 - alpha)/alpha)``),
reported per level. Results are aggregated per level and reported as L2
(``d_eval``) and Chamfer mean +/- std (per-sample aggregation, denormalized
metric units). No training, no optimizer, no augmentation. The pose-only floor
is owned by ``model_comparison`` (different checkpoint) and is NOT computed here.

The pure layer (``noise_levels``, ``apply_noise_blend``, ``effective_snr``) is
torch-only and dependency-free so ``--help`` and ``--selftest`` never import the
dataset, the model, GPU code, Sparsh, yaml, or matplotlib. ``compute_channel_stats``
needs the loader and so lives in the eval section (like the old stats pass);
``DLOStateModel``, ``DLOH5SampleDataset``, the loss fns, ``yaml`` and
``matplotlib`` are lazy-imported inside the eval / output functions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

# Add src so that "from dlofeel_state..." resolves when run from repo root or src.
_src_dir = Path(__file__).resolve().parents[2] / "src"  # .../src
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

DEFAULT_CONFIG = "src/dlofeel_state/configs/default.yaml"


# ----------------------------- Pure noise layer -----------------------------
# These symbols are the ONLY things importable without torch-data/GPU/Sparsh.


def noise_levels() -> list[int]:
    """Enumerate the 11 blend levels (percent, ``alpha = p/100``): 0, 10, ..., 100."""
    return list(range(0, 101, 10))


def apply_noise_blend(
    imgs: torch.Tensor,
    alpha: float,
    mean_c: torch.Tensor,
    std_c: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Convex-blend a copy of ``imgs`` toward per-channel-matched white noise.

    Computes ``(1 - alpha) * imgs + alpha * eps`` where
    ``eps[:, :, c] ~ N(mean_c[c], std_c[c]^2)`` i.i.d. per element. The
    per-channel moments broadcast from ``[6]`` to ``[1, 1, 6, 1, 1]`` over dim 2
    (the channel axis of ``[B, n_digits, C=6, H, W]``).

    Args:
        imgs: batched tactile tensor ``[B, n_digits, C, H, W]`` in normalized,
            diffed ``digit_images`` units.
        alpha: blend fraction in ``[0, 1]``. ``alpha == 0`` is an exact clean
            passthrough (RNG never touched); ``alpha == 1`` returns ``eps``
            exactly (signal fully replaced).
        mean_c: per-channel mean ``[C]`` (``C == imgs.shape[2]``).
        std_c: per-channel std ``[C]``.
        generator: optional RNG for reproducibility; must live on ``imgs.device``.

    Returns:
        A new tensor (input never mutated) of the same shape, dtype, and device:
        a clone when ``alpha == 0``, ``eps`` when ``alpha == 1``, else the blend.
    """
    assert imgs.shape[2] == mean_c.numel(), (
        f"channel mismatch: imgs C={imgs.shape[2]} vs mean_c {mean_c.numel()}"
    )
    if alpha == 0:
        # Clone BEFORE any torch.randn so the RNG stream is untouched at alpha=0.
        return imgs.clone()

    mean_c_b = mean_c.to(dtype=imgs.dtype, device=imgs.device).reshape(1, 1, -1, 1, 1)
    std_c_b = std_c.to(dtype=imgs.dtype, device=imgs.device).reshape(1, 1, -1, 1, 1)
    std_normal = torch.randn(
        imgs.shape, generator=generator, dtype=imgs.dtype, device=imgs.device
    )
    eps = mean_c_b + std_normal * std_c_b
    if alpha == 1:
        return eps
    return (1 - alpha) * imgs + alpha * eps


def effective_snr(alpha: float) -> tuple[float, float]:
    """Designed signal/noise power ratio of the blend at fraction ``alpha``.

    By construction ``eps`` is drawn at the empirical per-channel signal std, so
    the per-channel ``Var(signal)/Var(noise)`` ratio is channel-independent and a
    pure function of ``alpha``: ``snr = ((1 - alpha) / alpha)^2``.

    Args:
        alpha: blend fraction in ``[0, 1]``.

    Returns:
        ``(snr, snr_db)``. ``alpha == 0`` -> ``(inf, inf)`` (no noise);
        ``alpha == 1`` -> ``(0.0, -inf)`` (no signal).
    """
    if alpha == 0:
        return math.inf, math.inf
    if alpha == 1:
        return 0.0, -math.inf
    ratio = (1 - alpha) / alpha
    return ratio * ratio, 20.0 * math.log10(ratio)


# ----------------------------- Eval core -----------------------------


def _load_config(config_path: str) -> dict:
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_channel_stats(
    loader, device: torch.device, max_batches: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std of ``digit_images`` over the loader (no model forward).

    One float64 streaming pass that reduces over ``dim=(0, 1, 3, 4)`` of each
    ``[B, n_digits, C, H, W]`` batch, pooling over batch, sensor, and spatial
    axes so the 6 channels (dim 2) each get one ``(mean, std)`` pair. These are
    the empirical moments of the *already-normalized, already-diffed*
    ``digit_images`` (a DIFFERENT quantity from
    ``norm_stats_k16.yaml:digit_image_stats``, the input-normalization mean/std), and
    they set both the ``eps`` distribution and the SNR scale of the sweep.

    Args:
        loader: the test DataLoader (same one used for the sweep).
        device: device to accumulate on (matches the eval device).
        max_batches: if > 0, cap batches for cheaper estimates; 0 = all.

    Returns:
        ``(mean_c, std_c)``, each a float32 ``[C]`` tensor on ``device``.
    """
    count = 0
    sum_c: torch.Tensor | None = None
    sumsq_c: torch.Tensor | None = None

    for b_idx, batch in enumerate(loader):
        if max_batches and b_idx >= max_batches:
            break
        imgs = batch["digit_images"].to(device).double()  # [B, n_digits, C, H, W]
        b, n_digits, _, h, w = imgs.shape
        batch_sum = imgs.sum(dim=(0, 1, 3, 4))
        batch_sumsq = (imgs * imgs).sum(dim=(0, 1, 3, 4))
        if sum_c is None:
            sum_c = batch_sum
            sumsq_c = batch_sumsq
        else:
            sum_c += batch_sum
            sumsq_c += batch_sumsq
        count += b * n_digits * h * w

    assert sum_c is not None and sumsq_c is not None and count > 0, "empty loader"
    mean_c = sum_c / count
    var_c = (sumsq_c / count) - (mean_c * mean_c)
    std_c = var_c.clamp_min(0).sqrt()
    return mean_c.float().to(device), std_c.float().to(device)


def evaluate_noise(
    data_root: str,
    config: dict,
    ckpt_path: str,
    out_dir: str = "results/tactile_noise_ablation",
    max_batches: int = 0,
    seed: int = 42,
    stats_batches: int = 0,
) -> list[dict]:
    """Evaluate one checkpoint under all 11 blend levels and write the table.

    Mirrors ``evaluate.py``'s dataset/loader/model-load block: the model and
    checkpoint are loaded EXACTLY ONCE, then reused across all 11 levels (outer
    loop = levels, inner loop = DataLoader). The per-channel ``(mean_c, std_c)``
    are computed once up front from the loader. Per-sample L2 and Chamfer are
    accumulated across batches and reduced to mean/std over the full per-sample
    vector (NOT mean-of-batch-means).

    Args:
        data_root: directory of ``sample*.h5`` test files (or packed H5).
        config: run config (from the ckpt-dir ``config.yaml`` or the default).
        ckpt_path: path to the ``best.pt`` checkpoint.
        out_dir: results root; writes ``<out_dir>/<ckpt-parent-name>/noise_table.json``.
        max_batches: if > 0, cap batches per level (smoke runs); 0 = all.
        seed: RNG seed; the generator is re-seeded to this at the start of every
            level so levels differ only by alpha (the same ``std_normal`` underlies
            every level's ``eps``).
        stats_batches: if > 0, cap batches for the per-channel-stats pass; 0 = all.

    Returns:
        The 11 row dicts, each with ``p``, ``alpha``, ``snr``, ``snr_db``,
        ``val_l2_mean/std`` and ``val_chamfer_mean/std``. Non-finite ``snr``/
        ``snr_db`` are serialized to JSON as ``null``.
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

    mean_c, std_c = compute_channel_stats(loader, device, max_batches=stats_batches)
    print(f"per-channel mean_c (6): {mean_c.tolist()}")
    print(f"per-channel std_c  (6): {std_c.tolist()}")

    # Load model + checkpoint EXACTLY ONCE; reused across all 11 levels.
    model = DLOStateModel(config=config).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    normalize = config["normalize"]
    gen = torch.Generator(device=device)
    results: list[dict] = []

    with torch.no_grad():
        for p in noise_levels():
            alpha = p / 100.0
            snr, snr_db = effective_snr(alpha)
            # Re-seed per level so levels differ only by alpha (aligned, reproducible).
            gen.manual_seed(seed)
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

                imgs = apply_noise_blend(imgs, alpha, mean_c, std_c, generator=gen)
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
            # Per-session detailed payload (RA-L deliverable), same denorm tensors.
            detailed = reduce_metrics(torch.cat(pred_chunks), torch.cat(tgt_chunks), paths)
            row = {
                "p": p,
                "alpha": alpha,
                "snr": snr,
                "snr_db": snr_db,
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
            snr_db_str = f"{snr_db:.2f}" if math.isfinite(snr_db) else ("∞" if snr_db > 0 else "-∞")
            print(
                f"alpha={alpha:.2f} (p={p:>3}%)  SNR={snr_db_str} dB: "
                f"L2 {row['val_l2_mean']:.4f} ± {row['val_l2_std']:.4f}  "
                f"Chamfer {row['val_chamfer_mean']:.6f}"
            )

    print_table(results)

    json_path = out_path / "noise_table.json"
    with open(json_path, "w") as f:
        json.dump([_sanitize_row(r) for r in results], f, indent=2)
    print(f"Results saved to {json_path}")

    return results


def _sanitize_row(row: dict) -> dict:
    """Return a JSON-safe copy: non-finite ``snr``/``snr_db`` become ``None`` (null)."""
    out = dict(row)
    for k in ("snr", "snr_db"):
        v = out.get(k)
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
    return out


# ----------------------------- Output -----------------------------


def _snr_db_cell(snr_db: float) -> str:
    """Format an SNR(dB) value for the table ('∞'/'-∞' for non-finite)."""
    if math.isfinite(snr_db):
        return f"{snr_db:.2f}"
    return "∞" if snr_db > 0 else "-∞"


def print_table(results: list[dict]) -> None:
    """Print the 11-row blend table (alpha, SNR(dB), d_eval ± std, Chamfer; best L2 bold)."""
    best_l2 = min(results, key=lambda x: x["val_l2_mean"]) if results else None

    print("\n" + "=" * 80)
    print("Tactile noise ablation — clean↔white-noise blend (d_eval ± std, Chamfer)")
    print("=" * 80)
    print(f"{'α':<6} {'SNR(dB)':<10} {'d_eval (L2) ± std':<22} {'Chamfer':<12}")
    print("-" * 80)
    for r in results:
        best = r is best_l2
        l2_str = f"{r['val_l2_mean']:.4f} ± {r['val_l2_std']:.4f}"
        chamfer_str = f"{r['val_chamfer_mean']:.6f}"
        if best:
            l2_str = f"**{l2_str}**"
            chamfer_str = f"**{chamfer_str}**"
        print(f"{r['alpha']:<6.2f} {_snr_db_cell(r['snr_db']):<10} {l2_str:<22} {chamfer_str:<12}")
    print("=" * 80)
    print("(Best L2 row in bold.)")
    print("(Pose-only floor not shown here — owned by the model_comparison experiment.)\n")


def _save_curve(results: list[dict], out_path: str) -> None:
    """Plot L2 and Chamfer vs blend fraction alpha and write the PNG (lazy matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = [r["alpha"] for r in results]
    l2_means = [r["val_l2_mean"] for r in results]
    l2_stds = [r["val_l2_std"] for r in results]
    chamfer_means = [r["val_chamfer_mean"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color_l2 = "tab:blue"
    ax1.set_xlabel("noise fraction α")
    ax1.set_ylabel("d_eval (L2)", color=color_l2)
    ax1.errorbar(alphas, l2_means, yerr=l2_stds, color=color_l2, marker="o", capsize=3, label="L2")
    ax1.tick_params(axis="y", labelcolor=color_l2)

    ax2 = ax1.twinx()
    color_ch = "tab:red"
    ax2.set_ylabel("Chamfer", color=color_ch)
    ax2.plot(alphas, chamfer_means, color=color_ch, marker="s", linestyle="--", label="Chamfer")
    ax2.tick_params(axis="y", labelcolor=color_ch)

    fig.suptitle("Tactile noise ablation — error vs clean↔white-noise blend (α)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Curve saved to {out_path}")


# ----------------------------- Self-test -----------------------------


def _selftest() -> None:
    """Probe the pure blend layer, the stats pass, and ``effective_snr``; exit 0.

    All assertions run on synthetic CPU tensors with a fixed seed; no data, GPU,
    Sparsh, yaml, or matplotlib. Raises ``AssertionError`` on failure.
    """
    shape = (2, 4, 6, 32, 32)
    mean_c = torch.zeros(6)
    std_c = torch.ones(6)

    # 1. alpha=0: exact passthrough, new tensor, RNG untouched.
    x = torch.randn(shape)
    gen = torch.Generator()
    gen.manual_seed(123)
    state_before = gen.get_state()
    out0 = apply_noise_blend(x, 0.0, mean_c, std_c, generator=gen)
    assert torch.equal(out0, x), "alpha=0 not an exact passthrough"
    assert out0 is not x, "alpha=0 must return a new tensor"
    assert torch.equal(gen.get_state(), state_before), "alpha=0 touched the RNG"

    # 2. alpha=1: pure noise, signal-independent, per-channel moments matched.
    signal = torch.full(shape, 7.0)
    gen.manual_seed(7)
    out1 = apply_noise_blend(signal, 1.0, mean_c, std_c, generator=gen)
    assert not torch.equal(out1, signal), "alpha=1 must drop the signal"
    big = torch.zeros(8, 4, 6, 128, 128)
    gen.manual_seed(11)
    out_big = apply_noise_blend(big, 1.0, mean_c, std_c, generator=gen)
    emp_mean = out_big.mean(dim=(0, 1, 3, 4))
    emp_std = out_big.std(dim=(0, 1, 3, 4))
    assert torch.allclose(emp_mean, mean_c, atol=0.05), f"alpha=1 mean off: {emp_mean}"
    assert torch.allclose(emp_std, std_c, rtol=0.05), f"alpha=1 std off: {emp_std}"

    # 3. 0<alpha<1 exact convexity: reconstruct eps from an identically-seeded RNG.
    alpha = 0.3
    sig = torch.randn(shape)
    gen.manual_seed(99)
    out_blend = apply_noise_blend(sig, alpha, mean_c, std_c, generator=gen)
    gen2 = torch.Generator()
    gen2.manual_seed(99)
    eps = mean_c.reshape(1, 1, -1, 1, 1) + torch.randn(
        shape, generator=gen2
    ) * std_c.reshape(1, 1, -1, 1, 1)
    expected = (1 - alpha) * sig + alpha * eps
    assert torch.allclose(out_blend, expected, atol=1e-6), "0<alpha<1 not exact convex blend"

    # 4. per-channel broadcast hits dim 2: distinct per-channel mean recovered.
    mean_d = torch.arange(6).float() * 10.0
    std_d = torch.arange(1, 7).float() * 1.0
    gen.manual_seed(5)
    out_pc = apply_noise_blend(torch.zeros(8, 4, 6, 128, 128), 1.0, mean_d, std_d, generator=gen)
    emp_mean_pc = out_pc.mean(dim=(0, 1, 3, 4))
    assert torch.allclose(emp_mean_pc, mean_d, atol=0.2), f"per-channel mean broadcast off: {emp_mean_pc}"

    # 5. determinism: same seed -> equal; different seed -> not equal (0<alpha<=1).
    gen.manual_seed(1)
    a = apply_noise_blend(sig, 0.5, mean_c, std_c, generator=gen)
    gen.manual_seed(1)
    b = apply_noise_blend(sig, 0.5, mean_c, std_c, generator=gen)
    assert torch.equal(a, b), "same seed must reproduce"
    gen.manual_seed(2)
    c = apply_noise_blend(sig, 0.5, mean_c, std_c, generator=gen)
    assert not torch.equal(a, c), "different seed must differ"

    # 6. compute_channel_stats: pooled-over-sensors per-channel moments.
    # Two sensors carry different constants in channel 0; stats must pool them.
    known = torch.zeros(3, 4, 6, 5, 5)
    known[:, 0, 0] = 2.0
    known[:, 1, 0] = 4.0  # sensor 1, channel 0 = 4 (pooled mean over sensors -> mix)
    known[:, 2, 0] = 2.0
    known[:, 3, 0] = 4.0
    loader = [{"digit_images": known}]
    m, s = compute_channel_stats(loader, torch.device("cpu"))
    assert m.shape == (6,) and s.shape == (6,), "stats must be [6]"
    # Channel 0 pools {2,4,2,4} equally over the 4 sensors -> mean 3.0.
    assert abs(float(m[0]) - 3.0) < 1e-5, f"ch0 pooled mean wrong: {m[0]}"
    assert abs(float(s[0]) - 1.0) < 1e-5, f"ch0 pooled std wrong: {s[0]}"
    assert abs(float(m[1])) < 1e-6 and abs(float(s[1])) < 1e-6, "ch1 should be all-zero"

    # 7. effective_snr.
    assert effective_snr(0.0) == (math.inf, math.inf), "snr(0) should be inf"
    assert effective_snr(1.0) == (0.0, -math.inf), "snr(1) should be 0 / -inf"
    snr_half, db_half = effective_snr(0.5)
    assert abs(snr_half - 1.0) < 1e-9 and abs(db_half) < 1e-9, "snr(0.5) should be 1.0 / 0 dB"
    _, db_01 = effective_snr(0.1)
    assert abs(db_01 - 19.0849) < 1e-3, f"snr_db(0.1) off: {db_01}"
    prev = math.inf
    for a_val in (0.1, 0.3, 0.5, 0.7, 0.9):
        snr_a, _ = effective_snr(a_val)
        assert snr_a < prev, "snr must be monotone-decreasing in alpha"
        prev = snr_a

    # 8. reduce_metrics integration: 2 sessions, hand-checkable per-session L2.
    from dlofeel_state.engine.metrics import reduce_metrics

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
    assert abs(detailed["aggregate"]["l2_mean"] - 1.75) < 1e-5, "aggregate L2 mean off"

    print(
        "selftest OK: alpha=0 passthrough (RNG untouched); alpha=1 per-channel "
        "white noise; exact convex blend; broadcast hits dim 2; deterministic; "
        "compute_channel_stats pools sensors; effective_snr(0.5)=1.0/0dB; "
        "reduce_metrics groups 2 sessions (s0=1.0 s1=4.0, aggregate=1.75)."
    )


# ----------------------------- CLI -----------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tactile noise ablation — clean↔white-noise blend robustness (test-only, 11 levels)."
    )
    p.add_argument("--data_root", type=str, default="src/datasets/c1/ds/test",
                   help="Directory containing sample*.h5 (or packed H5).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to the C1 model checkpoint best.pt (omit only with --selftest).")
    p.add_argument("--out_dir", type=str, default="results/tactile_noise_ablation",
                   help="Output root; noise_table.json written under out_dir/<ckpt-parent-name>/.")
    p.add_argument("--blank_ref_root", type=str,
                   default="src/datasets/ds/digit_blank_closed_refs",
                   help="Path to digit blank reference images directory.")
    p.add_argument("--max_batches", type=int, default=0,
                   help="Cap batches per level for smoke runs (0 = all).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed; re-applied at the start of each level.")
    p.add_argument("--stats_batches", type=int, default=0,
                   help="Cap batches for the per-channel-stats pass (0 = all).")
    p.add_argument("--selftest", action="store_true",
                   help="Run the synthetic blend / stats / SNR probes and exit 0.")
    p.add_argument("--plot", dest="plot", action="store_true", default=True,
                   help="Write noise_curve.png after the sweep (default on).")
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   help="Skip the noise_curve.png plot.")
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
    results = evaluate_noise(
        data_root=args.data_root,
        config=config,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        max_batches=args.max_batches,
        seed=args.seed,
        stats_batches=args.stats_batches,
    )

    if args.plot:
        exp_name = ckpt_dir.name
        curve_path = Path(args.out_dir) / exp_name / "noise_curve.png"
        _save_curve(results, str(curve_path))


if __name__ == "__main__":
    main()
