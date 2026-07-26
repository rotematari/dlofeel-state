#!/bin/bash
# run_session_scaling.sh
# Runs the session-scaling experiment three times:
#   1. Sparsh backbone (tactile images + pose)
#   2. ResNet18 backbone (tactile images + pose)
#   3. Pose-only ablation
#
# Usage:
#   bash experiments/fig4_data_efficiency/run_session_scaling.sh
#   (run from the repo root)
#
# Note: Raises ulimit -n to avoid "too many open files" (Errno 24) during
# long runs with many trials.

set -euo pipefail

# Avoid OSError: [Errno 24] Too many open files during long multi-trial runs
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/fig4_data_efficiency/train_session_scaling.py"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python executable not found at $PYTHON"
    echo "Activate/create the project venv at ${REPO_ROOT}/.venv first."
    exit 1
fi

# Default config from selected good run.
# Override by passing a config path as first arg:
#   bash .../run_session_scaling.sh /path/to/config.yaml
BEST_CONFIG="${1:-results/good_ones/img/25_02_11_13_np32_sinencTrue_attTrue_normTrue_norminpTrue_poseFalse_diffTrue_blankrefTrue_img320x240_bs32_lr7.147008552233424e-05_wd0.00010701368128884272_dp0.1_ep10/config.yaml}"

COMMON_ARGS=(
    --train_data_root  "src/datasets/c1/ds/train"
    --val_data_root    "src/datasets/c1/ds/test"
    --blank_ref_root   "src/datasets/ds/digit_blank_closed_refs"
    --config_path      "$BEST_CONFIG"
    --norm_stats_path  "src/dlofeel_state/configs/norm_stats_k16.yaml"
    --k_repeats        3
    --epochs_per_trial 10
    --with_att
    --normalize_inputs
    --normalize
    --use_diff_images
    --use_blank_ref
    --single_encoder
    # K must match the per_marker_stats row count of --norm_stats_path: the
    # dataset normalizes the [K,3] resampled GT against it elementwise, so a
    # mismatch is a broadcast error. The stats file this release ships is K=16.
    --num_resample_points 16
    --batch_size       32
    --lr               7.147008552233424e-5
    --dropout          0.1
    --weight_decay     0.00010701368128884272
    --num_workers      16
    --img_size_H       320
    --img_size_W       240
    --seed             42
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUT="results/session_scaling/${TIMESTAMP}"

echo "=============================================="
echo " Run 1: Sparsh backbone (images + pose)"
echo " Output: ${BASE_OUT}/sparsh_vit"
echo "=============================================="
"$PYTHON" "$SCRIPT" "${COMMON_ARGS[@]}" --image_backbone_type sparsh_vit \
    --out_dir "${BASE_OUT}/sparsh_vit"

echo ""
echo "=============================================="
echo " Run 2: ResNet18 backbone (images + pose)"
echo " Output: ${BASE_OUT}/resnet18"
echo "=============================================="
"$PYTHON" "$SCRIPT" "${COMMON_ARGS[@]}" --image_backbone_type resnet18 \
    --out_dir "${BASE_OUT}/resnet18"

echo ""
echo "=============================================="
echo " Run 3: Pose-only ablation"
echo " Output: ${BASE_OUT}/pose_only"
echo "=============================================="
"$PYTHON" "$SCRIPT" "${COMMON_ARGS[@]}" --only_pose \
    --out_dir "${BASE_OUT}/pose_only"

echo ""
echo "Done. All results saved under ${BASE_OUT}/"
