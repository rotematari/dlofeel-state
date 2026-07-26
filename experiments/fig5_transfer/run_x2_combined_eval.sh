#!/bin/bash
# run_x2_combined_eval.sh - deeper eval of the X-2 combined (C1+C2+C3) model.
#
# Evaluates the ONE combined model on ALL THREE cable test sets (c1, c2, c3) - not
# just c1 - and prints the PER-SESSION results for each cable (plus the aggregate).
# evaluate.py reads the model's norm-stats (norm_stats_k16_combined.yaml) from the
# checkpoint config, so all three evals use the union normalization the model trained
# with; metrics are denormalized millimetres. Each eval writes its detailed
# run_metrics.json (aggregate + per_session) under results/ral_bigrun/X2_combined_eval_<cable>/.
#
# NOTE (c3 leak): session 05_04_17_08_c3_r0_f0 is in c3 TEST and was in the union
# TRAIN when X2_combined was trained -> its c3 per-session row is optimistic. Discount
# that one session, or re-train X-2 over the de-leaked union for a fully clean number.
#
# Usage (cwd does not matter; the script cd's to the repo root):
#   bash experiments/fig5_transfer/run_x2_combined_eval.sh [--dry-run]

set -uo pipefail   # NO -e: each eval is attempted; PASS/FAIL summary at the end.

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

R=results/ral_bigrun
X2_CKPT="$R/X2_combined/checkpoints/combined/best.pt"
BLANK=src/datasets/ds/digit_blank_closed_refs

if [[ "$DRY_RUN" -eq 0 ]]; then
    [[ -f "$X2_CKPT" ]] || { echo "ERROR: X2 combined ckpt not found: $X2_CKPT" >&2; exit 2; }
    for c in c1 c2 c3; do
        [[ -e "src/datasets/$c/ds/test" ]] || { echo "ERROR: $c test not found (FAST_DATA mounted?)" >&2; exit 2; }
    done
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

# Eval the combined model on each cable's test set.
for c in c1 c2 c3; do
    mkdir -p "$R/X2_combined_eval_$c"
    run_step "combined -> $c test" \
        uv run dlofeel-eval \
            --ckpt "$X2_CKPT" \
            --data_root "src/datasets/$c/ds/test" \
            --blank_ref_root "$BLANK" \
            --out_dir "$R/X2_combined_eval_$c"
done

# --- per-session summary (reads each run_metrics.json; metrics -> mm) ---
if [[ "$DRY_RUN" -eq 0 ]]; then
    echo ""
    echo "=================================================================="
    echo " X-2 COMBINED MODEL - PER-SESSION RESULTS (denormalized mm)"
    echo "=================================================================="
    uv run python - <<'PY'
import json
from pathlib import Path
R = Path("results/ral_bigrun")
for c in ("c1", "c2", "c3"):
    # evaluate.py nests under exp_name (the ckpt-parent, "combined"), so rglob for it.
    cand = sorted((R / f"X2_combined_eval_{c}").rglob("run_metrics.json"))
    if not cand:
        print(f"\n[{c.upper()}] no run_metrics.json (eval failed?)")
        continue
    d = json.load(open(cand[0]))
    agg = d.get("aggregate") or d.get("best", {}).get("aggregate") or {}
    ps  = d.get("per_session") or d.get("best", {}).get("per_session") or {}
    rmse = agg.get("l2_mean", float("nan")) * 1000
    cham = agg.get("chamfer_mean", float("nan")) * 1000
    print(f"\n================  combined model on {c.upper()} test  "
          f"(n_sessions={len(ps)}, n_samples={d.get('n_samples','?')})  ================")
    print(f"  AGGREGATE      RMSE {rmse:7.2f} mm     Chamfer {cham:7.2f} mm")
    print(f"  {'session':<36} {'RMSE_mm':>9} {'Chamfer_mm':>11} {'n':>7}")
    for sid in sorted(ps):
        s = ps[sid]
        flag = "  <- LEAK (in union train)" if sid == "05_04_17_08_c3_r0_f0" else ""
        print(f"  {sid:<36} {s['l2_mean']*1000:9.2f} {s['chamfer_mean']*1000:11.2f} {s['n']:>7}{flag}")
PY
fi

echo ""
echo "=================================================================="
echo " X-2 DEEPER-EVAL SUMMARY"
echo "=================================================================="
for i in "${!NAMES[@]}"; do printf "  %-22s %s\n" "${NAMES[$i]}" "${STATUS[$i]}"; done
echo " JSONs: $R/X2_combined_eval_{c1,c2,c3}/run_metrics.json"
echo "=================================================================="
