# Tactile Drop Ablation (missing-sensor robustness, test-time)

> **Status: PLANNED** (spec only, not yet run).
> Cable: **C1** (1.5 mm braided steel). Model: the **pretrained C1 `DLOStateModel`** (Sparsh ViT), evaluated as-is.
> Parent: [`../`](../). Implemented siblings: `session_scaling/`, `model_comparison/`, `tactile_sensor_ablation/`.

## Question

How gracefully does the pretrained C1 model degrade when 1 to 3 of the 4 DIGIT tactile images are missing at test time? No retraining: this measures the robustness the model already has.

## Hypothesis

Accuracy drops with each missing sensor but stays well above the pose-only floor: 1-drop close to full, 3-drop approaching pose-only.

## Design

- **Test-only on the pretrained model.** Load the C1 best checkpoint and evaluate it under each drop mask. No training, no augmentation.
- **Drop = zeroed diff image.** A dropped sensor's input is its diff image set to all zeros (after the diff). Same mechanism as `tactile_sensor_ablation`.
- **Masks.** 1, 2, or 3 of 4 sensors dropped: C(4,1)+C(4,2)+C(4,3) = 14 masks, plus the full 0-drop case = **15 cases**. The 4-drop case is excluded (it is pose-only).
- **Report.** Evaluate every test sample under all 15 masks, aggregate per mask: a 15-row degradation table (L2 `d_eval` and Chamfer mean ± std). The pose-only reference comes from the separately-trained pose-only model (see `model_comparison`), not from 4-drop of this checkpoint.

**Relationship to `tactile_sensor_ablation`:** same zeroing mechanism, different question. `tactile_sensor_ablation` *retrains* a clean model per fixed sensor subset (16 subsets) to measure each sensor's *contribution*. This evaluates the *single pretrained model* under missing-sensor masks to measure its out-of-the-box *graceful degradation*.

## Implementation plan

- Eval-only harness reusing `evaluate.py` and `DLOStateModel` (load checkpoint, no optimizer).
- Apply each mask by zeroing the selected sensors' diff channels in the `digit_images` tensor, after the diff/blank-ref step, before the encoder.
- Loop the 15 masks, emit a per-mask metrics table.

## How to run (intended, from the repo root with uv)

```bash
# from the repo root  (harness: experiments/table2_sensor_ablation/eval_tactile_drop.py)
uv run python experiments/table2_sensor_ablation/eval_tactile_drop.py \
  --data_root      src/datasets/c1/ds/test \
  --blank_ref_root src/datasets/ds/digit_blank_closed_refs \
  --ckpt           <path-to-C1-best-checkpoint>/best.pt \
  --out_dir        results/tactile_drop_ablation
# the C1 best checkpoint is provided by Rotem (not git-tracked).
```

## Outputs

`results/tactile_drop_ablation/` with a 15-row per-mask table (L2/Chamfer mean ± std) and the pose-only reference row.

## Future investigation

Also *train* with drop augmentation and check whether it beats the pretrained model's degradation curve. Two ways to inject drops at train time:

- **Stochastic (recommended):** per sample, with probability `p_drop` draw d ~ Uniform{1,2,3} and zero a random d-subset, else keep full. About 1x compute, adds regularization noise, and `p_drop` protects the clean case.
- **Full enumeration:** expand every sample into all 15 variants. Exhaustive but about 15x compute, and a 14:1 degraded:clean ratio risks lowering clean-case accuracy.

Cheapest comparison: train both for a few epochs on C1, compare clean-case L2 and mean-over-drops L2 against the pretrained baseline curve above.

## Open questions

- [x] Drop = zero the diff image after the diff (decided).
- [x] Test-only on the pretrained model; 15 masks, 4-drop excluded (decided).
