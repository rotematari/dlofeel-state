#!/usr/bin/env python3
"""
Per-K marker-normalization-stats generator for the K-resample ablation.

Each K-arm of the sweep resamples the GT markers to ``K`` points, so its
``per_marker_stats`` must be ``[K, 3]`` rather than the base file's array
(normalization broadcasts ``[K,3] / [K,3]``; a mismatched row count crashes).
This script builds a dataset with
``num_resample_points=K``, computes the marker mean/std over the resampled GT via
``compute_all_stats(..., num_out_samples=K)``, and writes a per-K stats YAML whose
schema matches the base file. The ``ee_position_stats`` and ``digit_image_stats``
blocks are copied verbatim from the base (they are K-independent).

K == 32 is the published-C1 anchor: its ``per_marker_stats`` is copied verbatim from
the base file (no data pass) for an exact reproduction of the C1 normalization. That
one path is the only use of the base's ``per_marker_stats``, so it requires a 32-row
base and rejects anything else rather than writing a wrong-shaped file.

``--recompute_all`` is the cross-cable mode: it recomputes ALL THREE blocks (digit +
ee + per-marker) from ``--train_root`` itself, copying nothing from the base. It is for
cables whose EE-pose / DIGIT-image distributions differ from c1 (c2, c3) and for the
c1+c2+c3 union, where the base-copy default path would write wrong ee/digit blocks.

The output (``configs/norm_stats_k{K}.yaml``) is git-tracked and generate-once: the
script skips regeneration when the file already exists and validates that its
``per_marker_stats.mean`` has K rows (``--force`` overrides).

``--selftest`` / ``--help`` exercise the pure YAML-assembly + K=32 anchor-copy +
full-stats helpers on synthetic data and never touch GPU, Sparsh, the dataset, or any
data on disk.

Usage (from repo root):
    uv run python experiments/k_resample/gen_k_norm_stats.py \
      --K 8 \
      --train_root src/datasets/c1/ds/train \
      --base_stats src/dlofeel_state/configs/norm_stats_k16.yaml \
      --out src/dlofeel_state/configs/norm_stats_k8.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Add src so that "from dlofeel_state..." resolves when run from repo root or src.
_src_dir = Path(__file__).resolve().parents[2] / "src"  # .../src
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

DEFAULT_TRAIN_ROOT = "src/datasets/c1/ds/train"
# The shipped stats file. Only its K-independent ``digit_image_stats`` /
# ``ee_position_stats`` blocks are read on the default path, and those are identical
# across every per-K file, so the K=16 release file is a valid base for any K != 32.
# K == 32 additionally copies ``per_marker_stats`` and therefore demands a 32-row base
# (see ``_assemble_k32_anchor``), which this file is not.
DEFAULT_BASE_STATS = "src/dlofeel_state/configs/norm_stats_k16.yaml"
DEFAULT_BLANK_REF_ROOT = "src/datasets/ds/digit_blank_closed_refs"


# ----------------------------- Pure helpers (no data/GPU/Sparsh) -----------------------------


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _assemble_stats_yaml(
    base_stats: dict[str, Any],
    marker_mean: list[list[float]],
    marker_std: list[list[float]],
) -> dict[str, Any]:
    """Build the per-K stats dict.

    ``ee_position_stats`` and ``digit_image_stats`` are copied verbatim from the
    base (K-independent); ``per_marker_stats`` is replaced by the supplied ``[K,3]``
    mean/std. Schema matches the base stats YAML.

    Args:
        base_stats: Parsed base stats-YAML dict.
        marker_mean: ``[K, 3]`` per-marker mean (e.g. ``stats["marker_mean"]``).
        marker_std: ``[K, 3]`` per-marker std (e.g. ``stats["marker_std"]``).

    Returns:
        A stats dict with keys ``digit_image_stats``, ``ee_position_stats``,
        ``per_marker_stats`` (each ``mean``/``std``).
    """
    return {
        "digit_image_stats": base_stats["digit_image_stats"],
        "ee_position_stats": base_stats["ee_position_stats"],
        "per_marker_stats": {"mean": marker_mean, "std": marker_std},
    }


def _assemble_k32_anchor(base_stats: dict[str, Any]) -> dict[str, Any]:
    """K==32 anchor: per-K stats == base verbatim (per-marker, ee, digit all copied).

    This is the only path that reads the base's ``per_marker_stats``, so the base must
    itself be the 32-marker file. A shorter base (e.g. the shipped ``norm_stats_k16``)
    would otherwise be copied through silently and yield a K=32 stats file holding the
    wrong number of rows.
    """
    rows = _row_count(base_stats)
    if rows != 32:
        raise SystemExit(
            f"--base_stats has per_marker_stats with {rows} rows, but K=32 copies that "
            f"block verbatim and needs 32. The released repository ships only "
            f"{DEFAULT_BASE_STATS}; pass the 32-marker stats file explicitly, or "
            f"regenerate the K=32 arm from data with --recompute_all."
        )
    return _assemble_stats_yaml(
        base_stats,
        base_stats["per_marker_stats"]["mean"],
        base_stats["per_marker_stats"]["std"],
    )


def _assemble_full_stats_yaml(
    marker_mean: list[list[float]],
    marker_std: list[list[float]],
    ee_mean: list[float],
    ee_std: list[float],
    digit_mean: list[float],
    digit_std: list[float],
) -> dict[str, Any]:
    """Build the full 3-block stats dict from a fresh ``compute_all_stats`` pass.

    Unlike :func:`_assemble_stats_yaml`, NOTHING is copied from a base file: all
    three blocks (digit, ee, per-marker) come from the target cable's own train
    split. Used by ``--recompute_all`` for cables whose EE/digit distributions
    differ from c1 (the base-copy path would be wrong for them).

    Schema matches the shipped stats YAML exactly: ``digit_image_stats``,
    ``ee_position_stats``, ``per_marker_stats``, each with ``mean``/``std``.

    Args:
        marker_mean: ``[K, 3]`` per-marker mean (``stats["marker_mean"]``).
        marker_std: ``[K, 3]`` per-marker std (``stats["marker_std"]``).
        ee_mean: ``[7]`` EE-pose mean (3 pos + 4 ori).
        ee_std: ``[7]`` EE-pose std.
        digit_mean: ``[3]`` per-channel DIGIT image mean (R, G, B).
        digit_std: ``[3]`` per-channel DIGIT image std.

    Returns:
        A stats dict with keys ``digit_image_stats``, ``ee_position_stats``,
        ``per_marker_stats``.
    """
    import math

    assert len(digit_mean) == 3 and len(digit_std) == 3, (
        f"digit stats must be [3], got mean={len(digit_mean)}, std={len(digit_std)}"
    )
    assert len(ee_mean) == 7 and len(ee_std) == 7, (
        f"ee stats must be [7], got mean={len(ee_mean)}, std={len(ee_std)}"
    )
    assert len(marker_mean) == len(marker_std), "marker mean/std row count mismatch"
    assert all(len(row) == 3 for row in marker_mean) and all(len(row) == 3 for row in marker_std), (
        "per_marker stats rows must be length 3 ([K,3])"
    )
    assert all(math.isfinite(v) for v in digit_mean) and all(math.isfinite(v) for v in digit_std), (
        "non-finite digit stats"
    )
    assert all(math.isfinite(v) for v in ee_mean) and all(math.isfinite(v) for v in ee_std), (
        "non-finite ee stats"
    )
    assert all(math.isfinite(v) for row in marker_mean for v in row), "non-finite per_marker mean"
    assert all(math.isfinite(v) for row in marker_std for v in row), "non-finite per_marker std"

    return {
        "digit_image_stats": {"mean": digit_mean, "std": digit_std},
        "ee_position_stats": {"mean": ee_mean, "std": ee_std},
        "per_marker_stats": {"mean": marker_mean, "std": marker_std},
    }


def _write_yaml(out_path: str | Path, stats: dict[str, Any]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=None, sort_keys=False)


def _row_count(stats: dict[str, Any]) -> int:
    return len(stats["per_marker_stats"]["mean"])


# ----------------------------- Generation core (lazy data/GPU/Sparsh imports) -----------------------------


def generate_k_stats(
    K: int,
    train_root: str | Path,
    base_stats_path: str | Path,
    out_path: str | Path,
    blank_ref_root: str | Path = DEFAULT_BLANK_REF_ROOT,
    num_workers: int = 16,
    max_samples: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Generate (or load from cache) the per-K marker-normalization stats YAML.

    Idempotent: if ``out_path`` exists, is not ``force``-overridden, and its
    ``per_marker_stats`` has exactly K rows, the cached file is loaded and returned
    without a data pass. A row-count mismatch triggers regeneration.

    K == 32 copies the base ``per_marker_stats`` verbatim (no data pass). For other K,
    a ``DLOH5SampleDataset(num_resample_points=K, normalize=False, normalize_inputs=False)``
    is built over ``train_root`` and ``compute_all_stats(..., num_out_samples=K,
    with_images=False)`` accumulates the marker mean/std over the resampled GT.

    Args:
        K: Number of resampled output markers (per-marker stats become ``[K, 3]``).
        train_root: Train split root (``sample*.h5`` dir or packed ``.h5``).
        base_stats_path: Base stats YAML (source of ee/digit blocks, K=32 anchor).
        out_path: Destination ``norm_stats_k{K}.yaml``.
        blank_ref_root: Digit blank-reference dir (dataset construction; unused for stats).
        num_workers: DataLoader workers for the stats pass.
        max_samples: Cap the stats pass to the first N dataset samples (0 = full set).
        force: Recompute even when a valid cached file exists.

    Returns:
        The stats dict that was written (or loaded from a valid cache).
    """
    out_path = Path(out_path)
    base_stats = _load_yaml(base_stats_path)

    # Idempotent cache hit: valid existing file with the right row count.
    if out_path.exists() and not force:
        cached = _load_yaml(out_path)
        if _row_count(cached) == K:
            print(f"[gen_k_norm_stats] cache hit: {out_path} (K={K}, {K} rows) — skipping.")
            return cached
        print(
            f"[gen_k_norm_stats] {out_path} exists but has {_row_count(cached)} rows "
            f"(expected {K}) — regenerating."
        )

    # K==32 anchor: copy the base per_marker_stats verbatim, no data pass.
    if K == 32:
        stats_yaml = _assemble_k32_anchor(base_stats)
        _write_yaml(out_path, stats_yaml)
        print(f"[gen_k_norm_stats] K=32 anchor: copied base per_marker_stats verbatim → {out_path}")
        return stats_yaml

    # Data pass for K != 32. Lazy imports keep --selftest / --help free of GPU/Sparsh/data.
    from dlofeel_state.data.dataset import DLOH5SampleDataset
    from dlofeel_state.data.transforms import compute_all_stats

    # Mirror train.py:567-570 packed inference.
    is_packed = "packed" in str(train_root)

    dataset = DLOH5SampleDataset(
        root_dir=train_root,
        norm_stats_path=base_stats_path,  # loaded at construction but never applied (normalize=False)
        blank_ref_root=blank_ref_root,
        is_packed=is_packed,
        normalize=False,
        normalize_inputs=False,
        num_resample_points=K,
    )

    # --max_samples cap: truncate the dataset's OWN flat index in place rather than
    # wrapping in a Subset. compute_all_stats toggles dataset.normalize / .normalize_inputs
    # to force-disable normalization during the pass; a Subset wrapper has no such
    # attributes, and setting them on the wrapper would leave the underlying real
    # dataset still normalizing -> silently contaminated [K,3] stats. self.length drives
    # __len__ in BOTH modes (line 196 packed via len(f['paths']); line 206 non-packed via
    # len(self.files)), and __getitem__ indexes from the front in both, so a prefix cap is
    # exact. Non-packed also slices self.files to preserve the (files, length) invariant.
    if max_samples and max_samples > 0 and max_samples < len(dataset):
        if not is_packed:
            dataset.files = dataset.files[:max_samples]
        dataset.length = max_samples
        print(f"[gen_k_norm_stats] capping stats pass to first {max_samples} samples.")

    stats = compute_all_stats(
        dataset,
        num_workers=num_workers,
        num_out_samples=K,
        with_images=False,
    )

    marker_mean = stats["marker_mean"]
    marker_std = stats["marker_std"]
    assert len(marker_mean) == K, f"compute_all_stats returned {len(marker_mean)} marker rows, expected K={K}"

    stats_yaml = _assemble_stats_yaml(base_stats, marker_mean, marker_std)
    _write_yaml(out_path, stats_yaml)
    print(f"[gen_k_norm_stats] wrote {out_path} (per_marker_stats: [{K}, 3]).")
    return stats_yaml


def generate_full_stats(
    K: int,
    train_root: str | Path,
    out_path: str | Path,
    blank_ref_root: str | Path = DEFAULT_BLANK_REF_ROOT,
    num_workers: int = 16,
    max_samples: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Recompute ALL THREE stats blocks (digit + ee + per-marker) from one cable's train split.

    The default ``generate_k_stats`` path recomputes only ``per_marker_stats`` and
    copies ee/digit verbatim from the base c1 stats file. That is wrong for
    a different cable (c2/c3) or the c1+c2+c3 union, whose EE-pose and DIGIT-image
    distributions differ. This path recomputes every block from ``train_root`` itself.

    Mirrors the canonical full-stats pass (``dataset_seq.py:499-513``):
    ``use_blank_ref=False``, ``use_diff_images=False`` (so the DIGIT tensor is
    ``[4,3,H,W]``, matching ``compute_all_stats``'s ``[:,:,:3]`` slice),
    ``normalize=False``, ``normalize_inputs=False``, then
    ``compute_all_stats(..., num_out_samples=K, with_images=True)`` — ``with_images``
    MUST be True or ``compute_all_stats`` returns ``digit_*=None`` (``utils.py:271-272``).

    Idempotent: a valid cached ``out_path`` with K per-marker rows is loaded without a
    data pass. There is NO K=32 anchor here (that anchor is c1-specific).

    Args:
        K: Number of resampled output markers (``per_marker_stats`` -> ``[K, 3]``).
        train_root: This cable's (or the union's) train split root.
        out_path: Destination ``norm_stats_k{K}_{cable}.yaml``.
        blank_ref_root: Digit blank-reference dir (dataset construction; unused for stats).
        num_workers: DataLoader workers for the stats pass.
        max_samples: Cap the stats pass to the first N samples (0 = full set).
        force: Recompute even when a valid cached file exists.

    Returns:
        The stats dict that was written (or loaded from a valid cache).
    """
    out_path = Path(out_path)

    # Idempotent cache hit: valid existing file with the right per-marker row count.
    if out_path.exists() and not force:
        cached = _load_yaml(out_path)
        if _row_count(cached) == K:
            print(f"[gen_k_norm_stats] cache hit: {out_path} (K={K}, {K} rows) — skipping.")
            return cached
        print(
            f"[gen_k_norm_stats] {out_path} exists but has {_row_count(cached)} rows "
            f"(expected {K}) — regenerating."
        )

    # Lazy imports keep --selftest / --help free of GPU/Sparsh/data.
    from dlofeel_state.data.dataset import DLOH5SampleDataset
    from dlofeel_state.data.transforms import compute_all_stats

    is_packed = "packed" in str(train_root)

    # Mirror the canonical full-stats pass (dataset_seq.py:499-513). norm_stats_path is
    # left empty: with normalize=False / normalize_inputs=False nothing is applied, and
    # there is no base file to copy from in this recompute-all path.
    dataset = DLOH5SampleDataset(
        root_dir=train_root,
        # A readable 3-block YAML is required: __init__ opens norm_stats_path
        # unconditionally (dataset_seq.py:211->214), even though the values are
        # never applied here (normalize=False/normalize_inputs=False). Same trick
        # as generate_k_stats. Empty "" would crash with FileNotFoundError.
        norm_stats_path=DEFAULT_BASE_STATS,
        blank_ref_root=blank_ref_root,
        is_packed=is_packed,
        use_blank_ref=False,
        use_diff_images=False,
        normalize=False,
        normalize_inputs=False,
        num_resample_points=K,
    )

    # --max_samples cap: truncate the dataset's OWN flat index in place rather than
    # wrapping in a Subset/ConcatDataset. compute_all_stats toggles
    # dataset.normalize/.normalize_inputs to force-disable normalization; a wrapper has
    # no such attributes, so setting them on the wrapper leaves the underlying dataset
    # still normalizing -> silently contaminated stats. self.length drives __len__ in
    # both modes and __getitem__ indexes from the front, so a prefix cap is exact.
    if max_samples and max_samples > 0 and max_samples < len(dataset):
        if not is_packed:
            dataset.files = dataset.files[:max_samples]
        dataset.length = max_samples
        print(f"[gen_k_norm_stats] capping stats pass to first {max_samples} samples.")

    stats = compute_all_stats(
        dataset,
        num_workers=num_workers,
        num_out_samples=K,
        with_images=True,
    )

    assert len(stats["marker_mean"]) == K, (
        f"compute_all_stats returned {len(stats['marker_mean'])} marker rows, expected K={K}"
    )

    stats_yaml = _assemble_full_stats_yaml(
        marker_mean=stats["marker_mean"],
        marker_std=stats["marker_std"],
        ee_mean=stats["ee_mean"],
        ee_std=stats["ee_std"],
        digit_mean=stats["digit_mean"],
        digit_std=stats["digit_std"],
    )
    _write_yaml(out_path, stats_yaml)
    print(f"[gen_k_norm_stats] wrote {out_path} (full recompute: digit[3] ee[7] per_marker[{K},3]).")
    return stats_yaml


# ----------------------------- Selftest (synthetic; no data/GPU/Sparsh) -----------------------------


def _selftest() -> None:
    """Probe the pure YAML-assembly + K=32 anchor helpers on synthetic data; exit 0.

    Asserts: per-K assembly yields ``[K,3]`` finite per-marker mean/std with ee/digit
    blocks copied verbatim; the K=32 anchor returns the base ``per_marker_stats``
    unchanged. Never constructs a dataset or touches GPU/Sparsh/data.
    """
    import math

    # Synthetic base in the shipped stats schema, 32 markers (what the K=32 anchor needs).
    base_digit = {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}
    base_ee = {"mean": [0.0] * 7, "std": [1.0] * 7}
    base_marker_mean = [[float(i), float(i) + 0.1, float(i) + 0.2] for i in range(32)]
    base_marker_std = [[1.0, 1.0, 1.0] for _ in range(32)]
    base_stats: dict[str, Any] = {
        "digit_image_stats": base_digit,
        "ee_position_stats": base_ee,
        "per_marker_stats": {"mean": base_marker_mean, "std": base_marker_std},
    }

    # Per-K assembly (synthetic compute_all_stats-shaped return for K=8).
    K = 8
    fake_mean = [[0.5 * j, 0.5 * j + 1.0, 0.5 * j + 2.0] for j in range(K)]
    fake_std = [[2.0, 2.0, 2.0] for _ in range(K)]
    out = _assemble_stats_yaml(base_stats, fake_mean, fake_std)

    assert set(out.keys()) == {"digit_image_stats", "ee_position_stats", "per_marker_stats"}, (
        f"unexpected top-level keys: {out.keys()}"
    )
    pm_mean = out["per_marker_stats"]["mean"]
    pm_std = out["per_marker_stats"]["std"]
    assert len(pm_mean) == K and len(pm_std) == K, f"per_marker rows != K={K}"
    assert all(len(row) == 3 for row in pm_mean), "per_marker_stats.mean rows not length 3"
    assert all(len(row) == 3 for row in pm_std), "per_marker_stats.std rows not length 3"
    assert all(math.isfinite(v) for row in pm_mean for v in row), "non-finite per_marker mean"
    assert all(math.isfinite(v) for row in pm_std for v in row), "non-finite per_marker std"
    # ee / digit copied verbatim.
    assert out["digit_image_stats"] is base_stats["digit_image_stats"], "digit_image_stats not verbatim"
    assert out["ee_position_stats"] is base_stats["ee_position_stats"], "ee_position_stats not verbatim"

    # K=32 anchor: per_marker_stats unchanged from base.
    anchor = _assemble_k32_anchor(base_stats)
    assert anchor["per_marker_stats"]["mean"] == base_marker_mean, "K=32 anchor mean drifted from base"
    assert anchor["per_marker_stats"]["std"] == base_marker_std, "K=32 anchor std drifted from base"
    assert anchor["ee_position_stats"] is base_stats["ee_position_stats"], "K=32 ee not verbatim"
    assert anchor["digit_image_stats"] is base_stats["digit_image_stats"], "K=32 digit not verbatim"

    # _row_count agrees with K.
    assert _row_count(out) == K and _row_count(anchor) == 32, "row-count helper mismatch"

    # K=32 anchor rejects a short base (the shipped K=16 file) instead of copying it through.
    short_base = dict(base_stats, per_marker_stats={"mean": fake_mean, "std": fake_std})
    try:
        _assemble_k32_anchor(short_base)
    except SystemExit:
        pass
    else:
        raise AssertionError("K=32 anchor accepted a base whose per_marker_stats is not [32,3]")

    # --recompute_all assembly: full 3-block dict from synthetic compute_all_stats output.
    full_digit_mean = [0.07, 0.07, 0.07]
    full_digit_std = [0.16, 0.17, 0.17]
    full_ee_mean = [float(i) * 0.1 for i in range(7)]
    full_ee_std = [1.0 + 0.01 * i for i in range(7)]
    full_marker_mean = [[0.3 * j, 0.3 * j + 0.1, 0.3 * j + 0.2] for j in range(K)]
    full_marker_std = [[1.5, 1.5, 1.5] for _ in range(K)]
    full = _assemble_full_stats_yaml(
        marker_mean=full_marker_mean,
        marker_std=full_marker_std,
        ee_mean=full_ee_mean,
        ee_std=full_ee_std,
        digit_mean=full_digit_mean,
        digit_std=full_digit_std,
    )
    assert set(full.keys()) == {"digit_image_stats", "ee_position_stats", "per_marker_stats"}, (
        f"full-stats unexpected top-level keys: {full.keys()}"
    )
    assert len(full["digit_image_stats"]["mean"]) == 3 and len(full["digit_image_stats"]["std"]) == 3, (
        "full digit_image_stats not [3]"
    )
    assert len(full["ee_position_stats"]["mean"]) == 7 and len(full["ee_position_stats"]["std"]) == 7, (
        "full ee_position_stats not [7]"
    )
    assert len(full["per_marker_stats"]["mean"]) == K and len(full["per_marker_stats"]["std"]) == K, (
        "full per_marker_stats not [K,...]"
    )
    assert all(len(row) == 3 for row in full["per_marker_stats"]["mean"]), "full per_marker rows not [.,3]"
    # Block-level schema identity against the real on-disk stats file (mean/std keys per block).
    real_stats_path = _src_dir.parent / DEFAULT_BASE_STATS
    if real_stats_path.exists():
        real = _load_yaml(real_stats_path)
        assert set(full.keys()) == set(real.keys()), (
            f"full-stats top-level keys {set(full.keys())} != {real_stats_path.name} {set(real.keys())}"
        )
        for block in full:
            assert set(full[block].keys()) == set(real[block].keys()), (
                f"block '{block}' keys {set(full[block].keys())} != {real_stats_path.name} {set(real[block].keys())}"
            )
        # Shapes match the published c1 base (digit[3], ee[7]) — K-row count is per-arm.
        assert len(real["digit_image_stats"]["mean"]) == len(full["digit_image_stats"]["mean"]), (
            "digit block width drifted from the shipped stats file"
        )
        assert len(real["ee_position_stats"]["mean"]) == len(full["ee_position_stats"]["mean"]), (
            "ee block width drifted from the shipped stats file"
        )

    # Bad shapes must be rejected by the assertions in _assemble_full_stats_yaml.
    for bad_kwargs in (
        {"digit_mean": [0.0, 0.0]},  # digit not [3]
        {"ee_mean": [0.0] * 6},  # ee not [7]
    ):
        kwargs: dict[str, Any] = {
            "marker_mean": full_marker_mean,
            "marker_std": full_marker_std,
            "ee_mean": full_ee_mean,
            "ee_std": full_ee_std,
            "digit_mean": full_digit_mean,
            "digit_std": full_digit_std,
        }
        kwargs.update(bad_kwargs)
        try:
            _assemble_full_stats_yaml(**kwargs)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"_assemble_full_stats_yaml accepted bad shapes: {bad_kwargs}")

    print(
        "selftest OK: per-K assembly [K,3] finite, ee/digit verbatim; K=32 anchor unchanged; "
        "full-stats 3-block digit[3]/ee[7]/per_marker[K,3] schema-key-identical to the shipped "
        "stats file; K=32 anchor rejects a non-32-row base."
    )


# ----------------------------- CLI -----------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-K marker-normalization-stats generator for the K-resample ablation."
    )
    p.add_argument("--K", type=int, default=None, help="Number of resampled output markers (per_marker_stats -> [K,3]).")
    p.add_argument("--train_root", type=str, default=DEFAULT_TRAIN_ROOT,
                   help="Train split root (sample*.h5 dir or packed .h5).")
    p.add_argument("--base_stats", type=str, default=DEFAULT_BASE_STATS,
                   help="Base stats YAML (ee/digit source; K=32 also copies its per-marker "
                        "block and so needs a 32-row file). Default: the shipped K=16 file.")
    p.add_argument("--out", type=str, default=None,
                   help="Output norm_stats_k{K}.yaml path (default: configs/norm_stats_k{K}.yaml).")
    p.add_argument("--blank_ref_root", type=str, default=DEFAULT_BLANK_REF_ROOT,
                   help="Digit blank-reference dir (dataset construction; unused for stats).")
    p.add_argument("--num_workers", type=int, default=16, help="DataLoader workers for the stats pass.")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Cap the stats pass to the first N samples (0 = full train set).")
    p.add_argument("--force", action="store_true", help="Recompute even if a valid cached file exists.")
    p.add_argument("--recompute_all", action="store_true",
                   help="Recompute ALL 3 blocks (digit+ee+marker) from --train_root itself "
                        "(no base-copy; for c2/c3/union whose ee/digit differ from c1).")
    p.add_argument("--selftest", action="store_true",
                   help="Run the synthetic assembly + K=32 anchor probes and exit 0 (no GPU/Sparsh/data).")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.K is None:
        p.error("--K is required (omit only for --selftest).")

    out_path = args.out if args.out is not None else f"src/dlofeel_state/configs/norm_stats_k{args.K}.yaml"

    if args.recompute_all:
        generate_full_stats(
            K=args.K,
            train_root=args.train_root,
            out_path=out_path,
            blank_ref_root=args.blank_ref_root,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            force=args.force,
        )
        return

    generate_k_stats(
        K=args.K,
        train_root=args.train_root,
        base_stats_path=args.base_stats,
        out_path=out_path,
        blank_ref_root=args.blank_ref_root,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        force=args.force,
    )


if __name__ == "__main__":
    main()
