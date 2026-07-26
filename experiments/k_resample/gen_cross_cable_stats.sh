#!/bin/bash
# gen_cross_cable_stats.sh
# Cross-cable (k16) normalization-stats driver for the X-block.
#
# Produces, in order:
#   1. norm_stats_k16_c2.yaml   — full 3-block stats recomputed from c2's own train split
#   2. norm_stats_k16_c3.yaml   — same, from c3
#   3. src/datasets/combined/ds/train — c1+c2+c3 symlink-union dir (gitignored)
#   4. norm_stats_k16_combined.yaml — full 3-block stats over the union dir
#
# Order is load-bearing: the union stats run over a single real DLOH5SampleDataset
# (compute_all_stats mutates dataset.normalize, so no ConcatDataset/Subset), so the
# union symlink dir MUST exist before step 4.
#
# c1 is NOT regenerated here — its norm_stats_k16.yaml is the published k16 model's
# training norm and stays as-is.
#
# Idempotent / re-runnable: each stats file skips if already present with 16 rows
# (--force to override); the union links skip if already correct.
#
# Requires FAST_DATA mounted (c2/c3 symlinks resolvable). Offline verification of the
# tooling is via the per-script --selftest (no data needed).
#
# Usage:
#   bash experiments/k_resample/gen_cross_cable_stats.sh
#   (invocation cwd does not matter; the script cd's to the repo root)

set -euo pipefail

# Avoid OSError: [Errno 24] Too many open files during the with_images stats pass.
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

cd "$REPO_ROOT"

GEN="experiments/k_resample/gen_k_norm_stats.py"
UNION="experiments/fig5_transfer/build_combined_union.py"
# Mount-independent local blank-ref copy (cable-independent; shared by every pass).
BLANK_REF="src/datasets/ds/digit_blank_closed_refs"
K=16

echo "=============================================="
echo " [1/4] c2 full norm-stats (k${K})"
echo "=============================================="
uv run python "$GEN" \
    --K "$K" \
    --recompute_all \
    --train_root src/datasets/c2/ds/train \
    --blank_ref_root "$BLANK_REF" \
    --out src/dlofeel_state/configs/norm_stats_k16_c2.yaml

echo "=============================================="
echo " [2/4] c3 full norm-stats (k${K})"
echo "=============================================="
uv run python "$GEN" \
    --K "$K" \
    --recompute_all \
    --train_root src/datasets/c3/ds/train \
    --blank_ref_root "$BLANK_REF" \
    --out src/dlofeel_state/configs/norm_stats_k16_c3.yaml

echo "=============================================="
echo " [3/4] build c1+c2+c3 symlink-union train dir"
echo "=============================================="
uv run python "$UNION"

echo "=============================================="
echo " [4/4] union (c1+c2+c3) full norm-stats (k${K})"
echo "=============================================="
uv run python "$GEN" \
    --K "$K" \
    --recompute_all \
    --train_root src/datasets/combined/ds/train \
    --blank_ref_root "$BLANK_REF" \
    --out src/dlofeel_state/configs/norm_stats_k16_combined.yaml

echo ""
echo "Done. Wrote norm_stats_k16_{c2,c3,combined}.yaml under src/dlofeel_state/configs/"
echo "Union train dir: src/datasets/combined/ds/train"
