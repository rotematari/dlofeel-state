#!/bin/bash
# run_c3_xblock.sh - the C3 cross-cable runs that were missed in the X-block.
#
# Runs the three C3 transfer experiments at the k16 base, all into
# results/ral_bigrun/ (the c2 + combined runs already completed):
#   X3  c3_scratch     : train from scratch on C3, C3 norm-stats
#   X4  c3_ft_c1norm   : fine-tune the C1 k16 model on C3, C1 norm-stats
#   X4  c3_ft_c3norm   : fine-tune the C1 k16 model on C3, C3 norm-stats
# (X-4 is the normalization ablation: c1-norm vs cable-norm.)
#
# Usage (cwd does not matter; the script cd's to the repo root):
#   bash experiments/fig5_transfer/run_c3_xblock.sh [--dry-run]
#
# Prereqs (all already present): the k16 best.pt, norm_stats_k16{,_c3}.yaml,
# c3/ds/{train,test}, the Sparsh backbone, a GPU. Each run writes the detailed
# run_metrics.json (per-epoch x per-session) under its out_dir/checkpoints/<exp>/.
# Continue-on-error: every run is attempted; a PASS/FAIL summary prints at the end.

set -uo pipefail   # NO -e: failures are handled per-step so the rest still run.

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$REPO_ROOT"
export PYTHONPATH="src:${PYTHONPATH:-}"
ulimit -n 65536 2>/dev/null || true

K16=results/ral_bigrun/k_resample_ablation/k16/best_runs/k16/best.pt
R=results/ral_bigrun
CFG=src/dlofeel_state/configs

# k16 base train args (use_spline=true inherited from default.yaml; WITH L_weighted).
COMMON=(
    --train_config_path  src/dlofeel_state/configs/default.yaml
    --train_data_root    src/datasets/c3/ds/train
    --val_data_root      src/datasets/c3/ds/test
    --blank_ref_root     src/datasets/ds/digit_blank_closed_refs
    --image_backbone_type sparsh_vit
    --with_att --normalize --normalize_inputs --use_diff_images --use_blank_ref --single_encoder
    --num_resample_points 16 --n_control_points 16
    --batch_size 32 --lr 7.147008552233424e-5 --weight_decay 0.00010701368128884272
    --dropout 0.1 --epochs 10 --seed 42 --num_workers 8 --img_size_H 320 --img_size_W 240
    --no-use_wandb
)

# --- prereq guards (skip in dry-run) ---
if [[ "$DRY_RUN" -eq 0 ]]; then
    miss=0
    [[ -f "$K16" ]] || { echo "ERROR: k16 ckpt not found: $K16" >&2; miss=1; }
    [[ -f "$CFG/norm_stats_k16.yaml" ]]    || { echo "ERROR: missing $CFG/norm_stats_k16.yaml" >&2; miss=1; }
    [[ -f "$CFG/norm_stats_k16_c3.yaml" ]] || { echo "ERROR: missing $CFG/norm_stats_k16_c3.yaml (run gen_cross_cable_stats.sh)" >&2; miss=1; }
    [[ -e src/datasets/c3/ds/train ]]      || { echo "ERROR: c3 train data not found (FAST_DATA mounted?)" >&2; miss=1; }
    [[ "$miss" -eq 0 ]] || exit 2
fi

declare -a NAMES=() STATUS=()
run_step() {
    local name="$1"; shift
    echo ""
    echo "------------------------------------------------------------------"
    echo "  RUN : ${name}"
    echo "  CMD : $*"
    echo "------------------------------------------------------------------"
    NAMES+=("$name")
    if [[ "$DRY_RUN" -eq 1 ]]; then STATUS+=("(dry-run)"); return 0; fi
    if "$@"; then STATUS+=("PASS"); else local rc=$?; STATUS+=("FAIL(rc=${rc})"); echo "[WARN] '${name}' failed (rc=${rc}); continuing."; fi
}

# X-3  C3 from-scratch (C3 norm-stats)
run_step "X3 c3_scratch" \
    uv run dlofeel-train "${COMMON[@]}" \
        --norm_stats_path "$CFG/norm_stats_k16_c3.yaml" \
        --exp_name c3_scratch --out_dir "$R/X3_scratch_c3"

# X-4  C3 fine-tune, C1 norm-stats (matches the loaded weights' normalization)
run_step "X4 c3_ft_c1norm" \
    uv run dlofeel-train "${COMMON[@]}" \
        --init_from "$K16" \
        --norm_stats_path "$CFG/norm_stats_k16.yaml" \
        --exp_name c3_ft_c1norm --out_dir "$R/X4_ft_c3_c1norm"

# X-4  C3 fine-tune, C3 norm-stats (adapt fully to the cable)
run_step "X4 c3_ft_c3norm" \
    uv run dlofeel-train "${COMMON[@]}" \
        --init_from "$K16" \
        --norm_stats_path "$CFG/norm_stats_k16_c3.yaml" \
        --exp_name c3_ft_c3norm --out_dir "$R/X4_ft_c3_c3norm"

echo ""
echo "=================================================================="
echo " C3 X-BLOCK SUMMARY"
echo "=================================================================="
for i in "${!NAMES[@]}"; do printf "  %-20s %s\n" "${NAMES[$i]}" "${STATUS[$i]}"; done
echo " Results under $R/{X3_scratch_c3,X4_ft_c3_c1norm,X4_ft_c3_c3norm}/"
echo "=================================================================="
