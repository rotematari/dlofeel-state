#!/bin/bash
# run_k_resample.sh
# B-spline output-resolution sweep over the C1 best run.
#
# Trains the C1 model 5x, once per K in {8, 16, 32, 64, 128}, tying the number
# of B-spline control points to the number of resampled output markers
# (--num_resample_points K --n_control_points K). use_spline stays true via the
# config, so the head emits K control points which DifferentiableBSpline
# resamples to K markers.
#
# Usage:
#   bash experiments/k_resample/run_k_resample.sh
#   (invocation cwd does not matter; the script cd's to the repo root)
#
# After the sweep, summarize with:
#   uv run python \
#     experiments/k_resample/plot_k_resample.py
#
# Note: Raises ulimit -n to avoid "too many open files" (Errno 24) during
# long multi-run sweeps.

set -euo pipefail

# Avoid OSError: [Errno 24] Too many open files during long multi-run sweeps.
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

cd "$REPO_ROOT"

# Single source of truth for the blank-reference dir: used both by the per-K
# stats generator (its DLOH5SampleDataset construction opens these PNGs in
# __init__) and by train.py below, so the two can never diverge on the path.
# The local src/datasets/ds copy is mount-independent (not on the FAST_DATA SSD).
BLANK_REF="src/datasets/ds/digit_blank_closed_refs"

# C1 best-run hyperparameters (shared across every K). The K-dependent flags
# (--num_resample_points, --n_control_points, --norm_stats_path, --exp_name,
# --out_dir) are appended per-iteration in the loop below. --norm_stats_path is
# NOT here: each K-arm needs its own [K,3] per_marker_stats (the base
# norm_stats.yaml is [32,3] and only valid for K=32).
COMMON_ARGS=(
    --config_path        src/dlofeel_state/configs/default.yaml
    --train_data_root    src/datasets/c1/ds/train
    --val_data_root      src/datasets/c1/ds/test
    --blank_ref_root     "$BLANK_REF"
    --image_backbone_type sparsh_vit
    --with_att
    --normalize
    --normalize_inputs
    --use_diff_images
    --use_blank_ref
    --single_encoder
    --batch_size         32
    --lr                 7.147008552233424e-5
    --weight_decay       0.00010701368128884272
    --dropout            0.1
    --epochs             10
    --seed               42
    --num_workers        8
    --img_size_H         320
    --img_size_W         240
    --no-use_wandb
)

for K in 8 16 32 64 128; do
    echo ""
    echo "=============================================="
    echo " K-resample sweep: K=${K}"
    echo " control points -> output markers: ${K} -> ${K}"
    echo " Output: results/k_resample_ablation/k${K}"
    echo "=============================================="

    # Per-K normalization stats: [K,3] per_marker_stats from this K's resampled GT
    # (K=32 copies the base anchor verbatim). Idempotent — skips if the file already
    # exists with K rows. Committed to configs/ so reruns reuse it.
    K_STATS="src/dlofeel_state/configs/norm_stats_k${K}.yaml"
    uv run python \
        experiments/k_resample/gen_k_norm_stats.py \
        --K "$K" \
        --train_root src/datasets/c1/ds/train \
        --base_stats src/dlofeel_state/configs/norm_stats.yaml \
        --blank_ref_root "$BLANK_REF" \
        --out "$K_STATS"

    uv run dlofeel-train "${COMMON_ARGS[@]}" \
        --norm_stats_path "$K_STATS" \
        --num_resample_points "$K" \
        --n_control_points "$K" \
        --exp_name "k${K}" \
        --out_dir "results/k_resample_ablation/k${K}"
done

echo ""
echo "Done. All results saved under results/k_resample_ablation/"
echo "Summarize with:"
echo "  uv run python \\"
echo "    experiments/k_resample/plot_k_resample.py"
