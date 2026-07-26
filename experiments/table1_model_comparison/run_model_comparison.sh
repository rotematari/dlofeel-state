#!/bin/bash
# run_model_comparison.sh
# Compare three model variants:
#   1) Sparsh ViT (images + pose)
#   2) ResNet18   (images + pose)
#   3) Pose-only  (no images)
#
# Usage:
#   bash experiments/table1_model_comparison/run_model_comparison.sh
#   bash experiments/table1_model_comparison/run_model_comparison.sh --fast_test
#   bash experiments/table1_model_comparison/run_model_comparison.sh --fast_test --dry_run

set -euo pipefail

FAST_TEST=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast_test)
            FAST_TEST=1
            shift
            ;;
        --dry_run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--fast_test] [--dry_run]"
            exit 1
            ;;
    esac
done

# Avoid OSError: [Errno 24] Too many open files in long runs
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
# Launch via the installed `dlofeel-train` console script (draccus entrypoint).
# Array form so run_cmd's "$@" expansion keeps each token separate.
TRAIN_CMD=(uv run dlofeel-train)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="results/model_comparison/${TIMESTAMP}"

COMMON_ARGS=(
    --train_data_root "src/datasets/c1/ds/train"
    --val_data_root "src/datasets/c1/ds/test"
    --blank_ref_root "src/datasets/ds/digit_blank_closed_refs"
    --config_path "src/dlofeel_state/configs/default.yaml"
    --norm_stats_path "src/dlofeel_state/configs/norm_stats_k16.yaml"
    --num_resample_points 16
    --n_control_points 16
    --with_att
    --normalize_inputs
    --normalize
    --use_diff_images
    --use_blank_ref
    --single_encoder
    --img_size_H 320
    --img_size_W 240
    --batch_size 32
    --lr 7.147008552233424e-05
    --weight_decay 0.00010701368128884272
    --dropout 0.1
    --num_workers 16
    --seed 42
)

if [[ "$FAST_TEST" -eq 1 ]]; then
    EPOCHS=1
    EXTRA_TEST_ARGS=(--max_batches 10 --max_eval_batches 10)
    echo "Running in FAST TEST mode."
else
    EPOCHS=10
    EXTRA_TEST_ARGS=()
fi

run_cmd() {
    local title="$1"
    shift
    echo ""
    echo "=============================================="
    echo " ${title}"
    echo "=============================================="
    echo "$*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

run_cmd "Run 1: Sparsh ViT (images + pose) -> ${BASE_OUT}/sparsh_vit" \
    "${TRAIN_CMD[@]}" \
    "${COMMON_ARGS[@]}" \
    --epochs "$EPOCHS" \
    "${EXTRA_TEST_ARGS[@]}" \
    --out_dir "${BASE_OUT}/sparsh_vit" \
    --image_backbone_type sparsh_vit

run_cmd "Run 2: ResNet18 (images + pose) -> ${BASE_OUT}/resnet18" \
    "${TRAIN_CMD[@]}" \
    "${COMMON_ARGS[@]}" \
    --epochs "$EPOCHS" \
    "${EXTRA_TEST_ARGS[@]}" \
    --out_dir "${BASE_OUT}/resnet18" \
    --image_backbone_type resnet18

run_cmd "Run 3: Pose-only -> ${BASE_OUT}/pose" \
    "${TRAIN_CMD[@]}" \
    "${COMMON_ARGS[@]}" \
    --epochs "$EPOCHS" \
    "${EXTRA_TEST_ARGS[@]}" \
    --out_dir "${BASE_OUT}/pose" \
    --only_pose

echo ""
echo "Done. Results under ${BASE_OUT}/"
