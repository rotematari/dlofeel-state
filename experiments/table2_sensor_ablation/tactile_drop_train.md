# Tactile Drop-Augmented Training

> **Status: PLANNED** (spec only, not yet run).
> Cable: **C1** (1.5 mm braided steel). Model: `DLOStateModel` (Sparsh ViT), trained from scratch.
> Parent: [`../`](../). Scored by the sibling [`eval_tactile_drop.py`](eval_tactile_drop.py); baseline is the clean model measured by [`tactile_drop_ablation.md`](tactile_drop_ablation.md).

## Question

Does training the C1 model with sensor-drop augmentation make it robust to missing sensors, i.e. flatten the k=1/2/3 degradation curve relative to the clean-trained model, without sacrificing the k=0 (all-sensors) accuracy?

## Hypothesis

Drop augmentation substantially flattens the degradation: a drop-trained model loses much less per missing sensor than the clean model. The open risk is **k=0**: with a 14:1 degraded-to-clean training ratio, the all-sensors accuracy may slip below the clean model's. The experiment measures exactly this trade-off.

## Baseline to beat (clean-trained C1 model, from the test-only eval)

| k dropped | clean-trained L2 (m) |
|---|---|
| 0 (none) | 0.0326 |
| 1 (mean) | ~0.045 |
| 2 (mean) | ~0.058 |
| 3 (mean) | ~0.079 |

Success = the drop-trained curve sits well below this at k>=1 while k=0 stays close to 0.0326.

## Design

- **Single varying factor.** Everything (config, lr, weight_decay, batch_size, dropout, seed, backbone, `num_resample_points=32`, **epochs=10**) is identical to the clean C1 best run. The *only* difference is the drop augmentation on the training inputs. This keeps the clean-vs-drop comparison controlled.
- **Augmentation = full enumeration.** Each C1 training sample is expanded into **15 variants**: the real 0-drop sample plus all **14 drop masks** (4 one-drop, 6 two-drop, 4 three-drop; 4-drop excluded). The training set grows ~15x; epochs stay 10 (so ~15x the clean compute, accepted). The mask set is the same `enumerate_drop_masks()` (15 entries incl. k=0) used by the eval harness.
- **Drop = zeroed diff image.** A dropped sensor's input is its diff image set to all zeros (`digit_images[:, i] = 0`, post-diff), reusing `apply_drop()` from [`eval_tactile_drop.py`](eval_tactile_drop.py). Identical mechanism to the eval, so train-time and test-time drops match exactly.
- **Validation/checkpointing** uses the clean (un-augmented) C1 test set, so `best.pt` is selected on real all-sensors performance (k=0), the same selection criterion as the clean model.

## Implementation plan

- A **drop-augmenting dataset wrapper** over `DLOH5SampleDataset`: `__len__ = 15 * base_len`; `__getitem__(idx)` maps `base_idx = idx // 15`, `mask = MASKS[idx % 15]`, returns the base sample with `apply_drop` applied. Deterministic enumeration, so every (sample, mask) pair appears exactly once per epoch; the DataLoader shuffles over the 15x index space.
  - Known property (document, do not fix): a shuffled batch can contain several variants of the same base sample (correlated), and 14/15 of training inputs are degraded. This is the clean-case-dilution cost the k=0 metric will expose.
- **Training reuses `train.py`'s machinery** (model build, AdamW + scheduler, the weighted + Chamfer loss, AMP, and the `best.pt`/`config.yaml` checkpoint layout) so the output loads directly in `eval_tactile_drop.py`. Mechanism (a `--drop_augment` hook into `train.py` vs a thin standalone harness wrapping the dataset) is finalized at the `/code-task` stage; prefer maximal reuse of `train.py`.
- Harness: `train_tactile_drop.py` in this folder; CLI mirrors the C1 training flags + the data/blank-ref roots.

## How to run (intended, from the repo root with uv)

```bash
# from the repo root  (harness: train_tactile_drop.py, to be implemented; needs c1/ds data + GPU + the Sparsh backbone)
uv run python experiments/table2_sensor_ablation/train_tactile_drop.py \
  --train_data_root src/datasets/c1/ds/train \
  --val_data_root   src/datasets/c1/ds/test \
  --blank_ref_root  src/datasets/ds/digit_blank_closed_refs \
  --config_path     src/dlofeel_state/configs/default.yaml \
  --num_resample_points 32 --with_att --normalize --normalize_inputs \
  --use_diff_images --use_blank_ref --single_encoder \
  --image_backbone_type sparsh_vit --epochs 10 --seed 42 --no-use_wandb \
  --exp_name drop_aug --out_dir results/tactile_drop_train

# then score it with the existing harness and compare to the clean model's table:
uv run python experiments/table2_sensor_ablation/eval_tactile_drop.py \
  --ckpt results/tactile_drop_train/checkpoints/drop_aug/best.pt
```

## Outputs

A drop-trained checkpoint at `results/tactile_drop_train/checkpoints/drop_aug/{best.pt,config.yaml}`, its `drop_table.json` (15-mask degradation, via the eval harness), and the headline comparison: the drop-trained degradation curve overlaid on the clean-trained one (the table above).

## Relationship to siblings

- `tactile_drop_ablation.md` / `eval_tactile_drop.py`: evaluate the *clean pretrained* model under drops (the baseline this experiment tries to beat). Same drop mechanism, same masks.
- `tactile_sensor_ablation.py`: trains a *separate clean model per fixed sensor subset* (16 models) to measure each sensor's contribution. This experiment trains *one model with random drops* to measure learned robustness. Different question, different mechanism.

## Open questions / future

- [ ] Watch k=0 in the result: if it slips below 0.0326, that quantifies the enumeration clean-case-dilution cost. If it does, the cheaper **stochastic** alternative (one random drop mask per sample per epoch, ~1x compute, tunable clean fraction) is the natural follow-up, addable as a `--drop_mode stochastic` switch. Not built now (keeps to one varying factor).
- [ ] Report the augmentation's compute cost (wall-clock vs the clean 10-epoch run) alongside the accuracy gain.
