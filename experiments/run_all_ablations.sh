#!/bin/bash
# run_all_ablations.sh - run the full C1 ablation suite in one go.
#
# Runs the four planned ablations in sequence (quick test-only evals first, then the
# multi-hour training sweeps):
#   1. tactile_drop   (eval-only) - missing-sensor robustness, 15 masks
#   2. tactile_noise  (eval-only) - additive-Gaussian robustness, 11 levels
#   3. K_resample     (5 trainings + summary plot) - B-spline output resolution
#   4. Diff_B_spline  (2 trainings + per-arm eval + 2-row table) - spline-head ablation
#
# Usage:
#   bash experiments/run_all_ablations.sh <c1_best_ckpt> [--dry-run]
#
#   <c1_best_ckpt>  Path to the trained C1 best.pt. REQUIRED: the test-only drop/noise
#                   evals load it. K_resample and Diff_B_spline train their own models
#                   and ignore it.
#   --dry-run       Print the plan (every command) without executing anything.
#
# Data defaults to src/datasets/c1/ds/test and blank-ref to
# src/datasets/ds/digit_blank_closed_refs via each harness's own defaults.
# Prerequisites for a real run: the gitignored c1/ds data, the
# Sparsh src/sparsh/checkpoints/dino_vitsmall.ckpt, and a GPU.
#
# Continue-on-error: every step runs even if an earlier one fails; a PASS/FAIL summary
# prints at the end and the script exits non-zero if any step failed. All output is teed
# to results/ablation_suite_<timestamp>/run.log.

set -uo pipefail   # deliberately NO -e: failures are handled per-step so the suite continues.

# --- args ---
DRY_RUN=0
CKPT=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*) echo "Unknown option: $arg" >&2; exit 2 ;;
        *)  CKPT="$arg" ;;
    esac
done

# --- repo root + env ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$REPO_ROOT"
export PYTHONPATH="src:${PYTHONPATH:-}"
ulimit -n 65536 2>/dev/null || true

if [[ -z "$CKPT" ]]; then
    echo "ERROR: missing <c1_best_ckpt> (the test-only drop/noise evals need it)." >&2
    echo "Usage: bash $0 <c1_best_ckpt> [--dry-run]" >&2
    exit 2
fi
if [[ "$DRY_RUN" -eq 0 && ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 2
fi

# --- logging ---
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/ablation_suite_${TS}"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/run.log"
# tee all stdout+stderr to the log for the rest of the script.
exec > >(tee -a "$LOG") 2>&1

echo "=================================================================="
echo " C1 ABLATION SUITE"
echo " repo:  $REPO_ROOT"
echo " ckpt:  $CKPT"
echo " log:   $LOG"
echo " mode:  $([[ "$DRY_RUN" -eq 1 ]] && echo 'DRY RUN (no execution)' || echo 'EXECUTE')"
echo "=================================================================="

# --- step runner (continue-on-error) ---
declare -a STEP_NAMES=()
declare -a STEP_STATUS=()
run_step() {
    local name="$1"; shift
    echo ""
    echo "------------------------------------------------------------------"
    echo "  STEP: ${name}"
    echo "  CMD : $*"
    echo "------------------------------------------------------------------"
    STEP_NAMES+=("$name")
    if [[ "$DRY_RUN" -eq 1 ]]; then
        STEP_STATUS+=("(dry-run)")
        return 0
    fi
    if "$@"; then
        STEP_STATUS+=("PASS")
    else
        local rc=$?
        STEP_STATUS+=("FAIL(rc=${rc})")
        echo "[WARN] step '${name}' failed (rc=${rc}); continuing with the rest."
    fi
}

DBS="results/diff_bspline_ablation"

# 1-2. test-only evals (fast; need the C1 checkpoint)
run_step "tactile_drop (eval)" \
    uv run python experiments/table2_sensor_ablation/eval_tactile_drop.py --ckpt "$CKPT"

run_step "tactile_noise (eval)" \
    uv run python experiments/fig6_noise/eval_tactile_noise.py --ckpt "$CKPT"

# 3. K_resample (trains 5 models, then summarizes)
run_step "K_resample (train sweep)" \
    bash experiments/k_resample/run_k_resample.sh
run_step "K_resample (summarize)" \
    uv run python experiments/k_resample/plot_k_resample.py

# 4. Diff_B_spline (trains the spline_off arm, evals both, prints the 2-row table).
# The spline_on reference is the suite's k16 base $CKPT, not a retrained arm, so its
# eval points at $CKPT with --arm_name spline_on to land at ${DBS}/spline_on.json.
run_step "Diff_B_spline (train spline_off arm)" \
    bash experiments/table1_model_comparison/run_diff_bspline.sh
run_step "Diff_B_spline (eval spline_on)" \
    uv run python experiments/table1_model_comparison/eval_diff_bspline.py \
        --ckpt "$CKPT" --arm_name spline_on --out_dir "$DBS"
run_step "Diff_B_spline (eval spline_off)" \
    uv run python experiments/table1_model_comparison/eval_diff_bspline.py \
        --ckpt "${DBS}/spline_off/checkpoints/spline_off/best.pt"
run_step "Diff_B_spline (combine table)" \
    uv run python experiments/table1_model_comparison/eval_diff_bspline.py \
        --combine "${DBS}/spline_on.json" "${DBS}/spline_off.json"

# --- summary ---
echo ""
echo "=================================================================="
echo " ABLATION SUITE SUMMARY"
echo "=================================================================="
ANY_FAIL=0
for i in "${!STEP_NAMES[@]}"; do
    printf "  %-34s %s\n" "${STEP_NAMES[$i]}" "${STEP_STATUS[$i]}"
    [[ "${STEP_STATUS[$i]}" == FAIL* ]] && ANY_FAIL=1
done
echo "------------------------------------------------------------------"
echo " Results under results/{tactile_drop,tactile_noise,k_resample,diff_bspline}_ablation/"
echo " Log: $LOG"
echo "=================================================================="

exit "$ANY_FAIL"
