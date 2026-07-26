# K-Resample Ablation (B-spline output resolution)

> **Status: PLANNED** (spec only, not yet run).
> Cable: **C1** (1.5 mm braided steel), data staged at `src/datasets/c1/ds/` (see [`docs/DATA.md`](../../docs/DATA.md)). Main model: `DLOStateModel` (Sparsh ViT).
> Parent: [`../`](../). Siblings: `fig4_data_efficiency/`, `table1_model_comparison/`, `table2_sensor_ablation/`.

## Question

At how many output points K does the DLO need to be represented? Sweep the B-spline resampling resolution and find where accuracy saturates or degrades.

## Hypothesis

Too few points under-represent curvature (high error); beyond some K the curve is over-sampled and error plateaus. Expect a knee near the default K=32.

## Design

Five trainings on C1, sweeping **K ∈ {8, 16, 32, 64, 128}**.

- K is `num_output_pts`, set by the `--num_resample_points` CLI argument. It controls both the number of predicted markers and the number of ground-truth markers the dataset resamples to, so L2/Chamfer stay well-defined per K.
- With `use_spline=true` (kept on for the whole sweep), the regression head emits `n_control_points` control points and the B-spline resamples to K output markers. **`n_control_points` is tied to K** (set equal each run), so the sweep varies the spline's control-point resolution and the output count together.
- Everything else inherits the C1 best-run config: Sparsh ViT, `with_att`, diff + blank ref, lr ~7.15e-5, weight_decay ~1.07e-4, batch 32, dropout 0.1, 10 epochs, seed 42.
- Metric: L2 (`d_eval`) and Chamfer mean ± std on the full C1 test set, per K.

Caveat: absolute L2 is mildly confounded across K because the GT marker count changes with K. Chamfer is the more K-robust comparison; report both.

## Implementation plan

Reuse `train.py` in a five-value loop (a thin `run_k_resample.sh`, same shape as [`../fig4_data_efficiency/run_session_scaling.sh`](../fig4_data_efficiency/run_session_scaling.sh)), passing `--num_resample_points <K>` and `--n_control_points <K>` (tied) with a distinct `--out_dir` per K. `n_control_points` is config-only today (no CLI); expose a `--n_control_points` flag mirroring `--num_resample_points`, or write a per-K config. No model or dataset change otherwise.

## How to run (intended, from the repo root with uv)

```bash
# from the repo root
for K in 8 16 32 64 128; do
  uv run dlofeel-train \
    --train_data_root src/datasets/c1/ds/train \
    --val_data_root   src/datasets/c1/ds/test \
    --blank_ref_root  src/datasets/ds/digit_blank_closed_refs \
    --train_config_path src/dlofeel_state/configs/default.yaml \
    --num_resample_points "$K" --n_control_points "$K" --with_att --normalize --normalize_inputs \
    --use_diff_images --use_blank_ref --single_encoder \
    --image_backbone_type sparsh_vit --epochs 10 --seed 42 \
    --out_dir "results/k_resample_ablation/k${K}"
done
```

## Outputs

`results/k_resample_ablation/k{8,16,32,64,128}/` with per-K `best_metrics.json`, plus a summary table and an L2/Chamfer-vs-K plot.

## Open questions

- [x] With `use_spline=true`, sweep the output K only, or also sweep `n_control_points` (currently also 32)?
- [x] `n_control_points` = K, tied to the output count (decided; spline stays on).