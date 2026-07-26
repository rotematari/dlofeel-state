#!/bin/bash
# run_diff_bspline.sh
# Differentiable B-spline output-head ablation over the C1 best run.
#
# Trains the single spline-OFF arm at K=16 (--no-use_spline — the head emits
# num_output_pts markers directly). The spline-ON reference is the k16 base run
# itself (use_spline true, inherited from default.yaml), so no spline_on arm is
# retrained here. K=16 is fixed via --num_resample_points 16 with the matched
# --n_control_points 16, so both this arm and the k16 base emit [B,16,3].
#
# Usage:
#   bash experiments/table1_model_comparison/run_diff_bspline.sh
#   (invocation cwd does not matter; the script cd's to the repo root)
#
# Note: Raises ulimit -n to avoid "too many open files" (Errno 24) during the runs.

set -euo pipefail

# Avoid OSError: [Errno 24] Too many open files during the training runs.
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

cd "$REPO_ROOT"

# C1 best-run hyperparameters for the spline-off arm. The arm-specific flags
# (--no-use_spline, --exp_name, --out_dir) are appended in the invocation below.
# --num_resample_points 16 with --n_control_points 16 fixes K=16 (and resamples
# the GT to 16), matching the k16 base (spline-on reference), so both emit [B,16,3].
COMMON_ARGS=(
    --config_path        src/dlofeel_state/configs/default.yaml
    --norm_stats_path    src/dlofeel_state/configs/norm_stats_k16.yaml
    --train_data_root    src/datasets/c1/ds/train
    --val_data_root      src/datasets/c1/ds/test
    --blank_ref_root     src/datasets/ds/digit_blank_closed_refs
    --image_backbone_type sparsh_vit
    --with_att
    --normalize
    --normalize_inputs
    --use_diff_images
    --use_blank_ref
    --single_encoder
    --num_resample_points 16
    --n_control_points   16
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

echo ""
echo "=============================================="
echo " Diff B-spline ablation: arm B (spline_off)"
echo " --no-use_spline (head emits 16 markers directly)"
echo " Output: results/diff_bspline_ablation/spline_off"
echo "=============================================="
uv run dlofeel-train "${COMMON_ARGS[@]}" \
    --no-use_spline \
    --exp_name spline_off \
    --out_dir results/diff_bspline_ablation/spline_off

echo ""
echo "Done. spline_off arm trained under results/diff_bspline_ablation/spline_off"
echo "The spline_on reference is the k16 base run; evaluate it with --arm_name spline_on,"
echo "then evaluate this arm and combine into the 2-row table:"
echo "  uv run python \\"
echo "    experiments/table1_model_comparison/eval_diff_bspline.py \\"
echo "    --ckpt <k16_base>/best.pt --arm_name spline_on"
echo "  uv run python \\"
echo "    experiments/table1_model_comparison/eval_diff_bspline.py \\"
echo "    --ckpt results/diff_bspline_ablation/spline_off/checkpoints/spline_off/best.pt"
echo "  uv run python \\"
echo "    experiments/table1_model_comparison/eval_diff_bspline.py \\"
echo "    --combine results/diff_bspline_ablation/spline_on.json \\"
echo "             results/diff_bspline_ablation/spline_off.json"
