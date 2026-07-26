# Differentiable B-Spline Ablation

> **Status: PLANNED** (spec only, not yet run).
> Cable: **C1** (1.5 mm braided steel). Main model: `DLOStateModel` (Sparsh ViT).
> Parent: `[../](../)`. Implemented siblings: `session_scaling/`, `model_comparison/`, `tactile_sensor_ablation/`.

## Question

Does the differentiable B-spline output head improve DLO shape estimation, or does a model that regresses the marker positions directly do just as well?

## Hypothesis

The B-spline head acts as a smoothness prior on the predicted curve. Removing it should keep mean point error close but increase local jitter and the bending/length error terms (`w_edges`, `w_kb`, `w_lengths`). If the gap is negligible, the spline head is unnecessary complexity.

## Design

Two training runs on C1, identical except for one flag:


| Run                | `model_params.use_spline` | Output head                                                                                                                          |
| ------------------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Baseline (default) | `true`                    | regress `n_control_points` (=32) control points, then `DifferentiableBSpline` (degree 3) resamples to `num_output_pts` (=32) markers |
| Ablation           | `false`                   | regress `num_output_pts` (=32) markers directly, no spline layer                                                                     |


- Everything else inherits the C1 best-run config: Sparsh ViT, `with_att`, diff images + blank ref, `num_output_pts=32`, lr ~7.15e-5, weight_decay ~1.07e-4, batch 32, dropout 0.1, 10 epochs, seed 42.
- Metrics: L2 (`d_eval`) mean ± std and Chamfer mean ± std on the full C1 test set.
- **Report `kb_loss` and `length_loss` per arm.** These are the bending (curvature-binormal MSE) and edge-length MSE terms inside `WeightedLoss`, the two terms the spline prior most directly shapes. They currently live only inside the combined scalar (see Implementation plan).

## Smoothness metric

`kb_loss` and `length_loss` are *error* terms (pred-vs-GT MSE), so they measure fidelity to the true curve, not absolute smoothness: a model can post a low `kb_loss` while still looking jittery if the jitter averages out against the target. So the ablation also needs an absolute smoothness measure, and the honest one must not punish a model for curvature that is genuinely in the cable.

Proposed (reuses `compute_curvature_binormal` from `loss/der.py`, no new geometry):

- **Excess Mean Absolute Curvature (dMAC)** (primary). `MAC(P) = mean_i ||kb_i(P)||` over interior nodes, where `kb` is the discrete curvature binormal. Report `dMAC = MAC(pred) - MAC(gt)`: near 0 means the prediction is as smooth as the true cable, positive means spurious over-bending. `MAC(pred)` alone would penalize a legitimately curved cable, hence the excess-vs-GT framing.
- **Curvature Total Variation (CTV)** (secondary, GT-independent). `CTV(P) = mean_i ||kb_{i+1}(P) - kb_i(P)||`: pure high-frequency wiggle along the curve, catching oscillation that dMAC can miss when over- and under-bending cancel.

Both are computed at the fixed K=32 used by both arms, so they compare directly spline-on vs spline-off. Expectation: removing the spline raises dMAC and CTV even where mean L2 barely moves, which is exactly the quantity the smoothness prior buys.

## Implementation plan

`model_params.use_spline` already gates the entire B-spline path in `[models/sparsh_enc.py](../../models/sparsh_enc.py)`: the `DifferentiableBSpline` layer (defined ~L469, applied ~L716) is only built and used when `use_spline` is true. **No new model class is needed**, just the flag.

The flag is currently config-only (not a `train.py` CLI argument). Decided: expose a `--use_spline / --no-use_spline` override in `train.py` (same pattern as the other config-overriding flags); a two-line `run_diff_bspline.sh` runs both arms.

Reporting `kb_loss` / `length_loss` needs a small change: `WeightedLoss` returns only the combined scalar today (`loss/loss_with_bending_and_length.py:52`). Add a components return (`nodes_mse, edges_mse, lengths_mse, kb_mse`) and log `kb_mse` and `lengths_mse` on the test set per arm. The smoothness metric reuses `compute_curvature_binormal` from `loss/der.py` (the same `kb` used by `kb_mse`), so no new geometry code is needed.

## How to run (intended, from the repo root with uv)

```bash
# from the repo root
# Arm A: spline ON (baseline):
uv run dlofeel-train \
  --train_data_root src/datasets/c1/ds/train \
  --val_data_root   src/datasets/c1/ds/test \
  --blank_ref_root  src/datasets/c1/ds/digit_blank_closed_refs \
  --train_config_path src/dlofeel_state/configs/default.yaml \
  --num_resample_points 32 --with_att --normalize --normalize_inputs \
  --use_diff_images --use_blank_ref --single_encoder \
  --image_backbone_type sparsh_vit --epochs 10 --seed 42 \
  --out_dir results/diff_bspline_ablation/spline_on

# Arm B: spline OFF (ablation): same command, plus --no-use_spline once exposed,
#   --out_dir results/diff_bspline_ablation/spline_off
```

## Outputs

`results/diff_bspline_ablation/{spline_on,spline_off}/` with per-arm `best_metrics.json` (val L2/Chamfer mean ± std, plus `kb_loss`, `length_loss`, dMAC, CTV), and a two-row table contrasting the arms across all six numbers.

## Open questions

- [x] Expose `use_spline` as a CLI flag, or keep two config files?
- [x] Report `kb_loss` and `length_loss` per arm (decided).
- [x] Smoothness metric = dMAC (excess curvature) + CTV, both reported (decided).