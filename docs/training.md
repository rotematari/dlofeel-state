# Training & Evaluation Documentation

Trains DLO marker prediction from DIGIT tactile images + end-effector pose using `DLOH5SampleDataset`, `DLOStateModel`, and a composite physics-informed loss.

---

## 1. Running Training and Evaluation

Installing the package (`pip install -e .`) provides two console scripts:

| Script | Entrypoint | Typed config |
|--------|-----------|--------------|
| `dlofeel-train` | `dlofeel_state.engine.train:main` | `TrainConfig` (`src/dlofeel_state/configs/train.py`) |
| `dlofeel-eval` | `dlofeel_state.engine.evaluate:main` | `EvalConfig` (`src/dlofeel_state/configs/eval.py`) |

### Training

```bash
dlofeel-train \
  --train_data_root <path>/train \
  --val_data_root <path>/test \
  --num_resample_points 16 \
  --batch_size 32 --epochs 10 \
  --lr 7.147e-05 --weight_decay 1.07e-04 --dropout 0.1 \
  --use_wandb false
```

Everything not passed comes from the model YAML (`--train_config_path`, default
`src/dlofeel_state/configs/default.yaml`), whose production values already set
`with_att`, `normalize`, `normalize_inputs`, `use_diff_images`, `use_blank_ref`,
`single_encoder`, and `use_spline`. Passing those flags again is redundant, not required.

To reproduce the released \(K = 16\) architecture, override both point counts and
the matching normalization statistics:

```bash
dlofeel-train \
  --train_data_root <path>/train \
  --val_data_root <path>/test \
  --blank_ref_root <path>/digit_blank_closed_refs \
  --norm_stats_path src/dlofeel_state/configs/norm_stats_k16.yaml \
  --num_resample_points 16 --n_control_points 16 \
  --batch_size 32 --epochs 10 \
  --lr 7.147008552233424e-05 --weight_decay 0.00010701368128884272 --dropout 0.1 \
  --seed 42 --use_wandb false
```

`--num_resample_points` sets `num_output_pts` (the number of predicted markers) and
`--n_control_points` sets the spline's control-point count; they are independent
knobs, so a run that overrides only the first keeps the YAML's 32 control points.

### Evaluation

```bash
dlofeel-eval \
  --ckpt_path <path>/best.pt \
  --data_root <path>/test \
  --blank_ref_root <path>/digit_blank_closed_refs \
  --out_dir results_eval
```

`--model_config_path` defaults to `<ckpt_dir>/config.yaml`, so a checkpoint
evaluates with the architecture it was trained under and no config flag is needed.

### Config Precedence

```
model YAML (--train_config_path) ← CLI flags override ← W&B sweep overrides (when active)
```

Sweep overrides use dot notation for nested keys (e.g., `image_backbone.fusion_method` maps to `config['model_params']['image_backbone']['fusion_method']`).

---

## 2. CLI Arguments

Both scripts are draccus entrypoints: every flag is a typed field on `TrainConfig`
(`src/dlofeel_state/configs/train.py`) or `EvalConfig`
(`src/dlofeel_state/configs/eval.py`), so `dlofeel-train --help` /
`dlofeel-eval --help` is the authoritative list, and a YAML file of field values can
be passed with draccus's own `--config_path` (see `configs/eval/k16.yaml` for a
worked example). The tables below are a reading aid, not the contract.

**Two flag names differ from the original argparse CLI:**

| Original flag | Status now |
|---------------|-----------|
| `--balank_ref_root` | **Gone.** The original CLI had a typo; the flag is now `--blank_ref_root` |
| `--config_path` (model YAML) | **Repurposed.** draccus reserves `--config_path` for its own YAML-override loader. The model-architecture YAML is now `--train_config_path` |

Every optional scalar defaults to `None`, meaning "do not override the YAML". The
old CLI's own defaults (batch size 128, lr 1e-3, ...) are gone: omitting a flag now
yields the YAML value rather than an argparse default. The tri-state booleans accept
`--flag`, `--flag true`, `--flag false`, or omission.

### `dlofeel-train`

| Group | Flag | Default | Notes |
|-------|------|---------|-------|
| **Data** | `--train_data_root` | YAML | Auto-detects packed if the path contains `"packed"` |
| | `--val_data_root` | YAML | |
| | `--blank_ref_root` | YAML | DIGIT blank reference images |
| | `--train_config_path` | `src/dlofeel_state/configs/default.yaml` | Model / training YAML |
| | `--norm_stats_path` | YAML | Normalization statistics YAML |
| **Geometry** | `--num_resample_points` | YAML (`num_output_pts`) | Output marker count \(K\) |
| | `--n_control_points` | YAML | Spline control points (used when `use_spline`) |
| | `--img_size_H` / `--img_size_W` | YAML (320 / 240) | |
| **Model** | `--model_type` | `dlo_state` | `dlo_state` or `direction` (CV ablation) |
| | `--image_backbone_type` | YAML (`sparsh_vit`) | `sparsh_vit` or `resnet18` |
| | `--with_att` | YAML (`true`) | Cross-attention decoder |
| | `--only_pose` | YAML (`false`) | No images (pose-only ablation) |
| | `--single_encoder` | YAML (`true`) | Share the ViT across all 4 DIGIT cameras |
| | `--use_spline` | YAML (`true`) | Fit a B-spline through the predicted nodes |
| | `--init_from` | — | Checkpoint `.pt` for weights-only fine-tune init (`strict=False`) |
| **Inputs** | `--use_diff_images` | YAML (`true`) | Subtract reference images |
| | `--use_blank_ref` | YAML (`true`) | Blank refs (vs. per-session refs) |
| | `--normalize` | YAML (`true`) | Z-score normalise marker targets |
| | `--normalize_inputs` | YAML (`true`) | Z-score normalise images + EE pose |
| | `--active_sensors` | all | Sensor subset, e.g. `0,1,2,3` or `fu,fd,ru,rd` |
| | `--drop_augment` | `false` | Wrap the train set with the 15-mask sensor-drop augmentation |
| **Optimization** | `--epochs` | YAML (20) | |
| | `--batch_size` | YAML (32) | |
| | `--lr` | YAML (7e-05) | |
| | `--weight_decay` | YAML (1e-04) | |
| | `--dropout` | YAML (0.1) | |
| | `--use_weighted_loss` | YAML | `false` zeros `w_edges` / `w_lengths` / `w_kb` |
| | `--num_workers` | YAML (16) | DataLoader workers |
| | `--seed` | YAML (42) | |
| **Determinism** | `--deterministic` | `false` | Enable deterministic PyTorch kernels |
| | `--deterministic_warn_only` | `false` | With `--deterministic`, warn instead of error |
| **Output** | `--out_dir` | `results` | Checkpoints land under `<out_dir>/checkpoints/<exp_name>/` |
| | `--exp_name` | auto (time + params) | Override the experiment name |
| **Logging** | `--use_wandb` | `true` when omitted | Enable W&B logging |
| | `--wandb_project` | `dlo-state-estimation` | |
| | `--wandb_entity` | `dlo` | |
| **Fast tests** | `--max_batches` | 0 | Training batches per epoch (0 = unlimited) |
| | `--max_eval_batches` | 0 | Validation batches per epoch (0 = full eval) |

### `dlofeel-eval`

| Flag | Default | Notes |
|------|---------|-------|
| `--data_root` | **required** | Directory of eval samples (`sample*.h5` / session dirs) |
| `--ckpt_path` | **required** | Checkpoint `.pt`, loaded strict |
| `--blank_ref_root` | **required** | DIGIT blank reference images |
| `--out_dir` | `results_eval` | Results land under `<out_dir>/<ckpt-parent-name>/` |
| `--batch_size` | model config | |
| `--low_rmse_threshold` | 0.055 | Per-sample RMSE (m) below which samples are logged to `low_rmse_samples.json` |
| `--model_config_path` | `<ckpt_dir>/config.yaml` | Model-architecture YAML |
| `--norm_stats_path` | model config | Override the normalization statistics |

---

## 3. Training Pipeline

### 3.1 Initialisation

1. Load the model YAML (`--train_config_path`); override with the CLI fields that were set.
2. Unless `--use_wandb false`: init W&B run; apply sweep overrides (dotted keys → nested config).
3. Set random seed (`setup_reproducibility`, honouring `--deterministic`).
4. Verify no session overlap between train/val splits (`check_no_session_overlap`).
5. Create experiment directory: `{out_dir}/checkpoints/{exp_name}/`.
6. Save merged config as `config.yaml`.
7. Construct `DLOH5SampleDataset` for train and val (auto-detect packed vs. per-file).
8. Create model `DLOStateModel(config)` and log summary via `log_model_info`.

### 3.2 Experiment Naming

```
{DD_MM_HH_MM}_np{K}_sinenc{single}_att{att}_norm{norm}_norminp{norminp}_pose{pose}_diff{diff}_blankref{blank}_img{H}x{W}_bs{B}_lr{lr}_wd{wd}_dp{dp}_ep{ep}
```

Example: `09_02_20_05_np32_sinencTrue_attTrue_normTrue_norminpTrue_poseFalse_diffTrue_blankrefTrue_img320x240_bs32_lr0.001_wd0.0001_dp0.1_ep10`

`--exp_name` replaces the whole string, which is what the ablation drivers under `experiments/` use to get stable directory names.

### 3.3 Optimizer

**AdamW** with optional **differential learning rates** for the image backbone:

- If `unfreeze_last_n_blocks > 0` and backbone params exist:
  - Backbone param group: `lr × backbone_lr_factor`
  - All other params: `lr`
- Otherwise: single param group at `lr`.

| Config Key | Default | Description |
|------------|---------|-------------|
| `lr` | 7.147e-5 | Base learning rate |
| `weight_decay` | 1.07e-4 | AdamW weight decay |
| `model_params.image_backbone.backbone_lr_factor` | 0.5 | LR multiplier for unfrozen backbone |

### 3.4 Learning Rate Scheduler

Optional **OneCycleLR** (cosine annealing with warmup):

| Config Key | Default | Description |
|------------|---------|-------------|
| `use_lr_scheduler` | `true` | Enable scheduler |
| `lr_warmup_pct` | 0.1 | Fraction of training for warmup (10%) |
| `lr_div_factor` | 10.0 | Start LR = `base_lr / div_factor` |
| `lr_final_div_factor` | 100.0 | End LR = `base_lr / final_div_factor` |

LR profile: ramp from `lr/10` to `lr` over 10% of steps, then cosine decay to `lr/100`.

When differential LR is active, `max_lr` is set per param group: `[lr × backbone_lr_factor, lr]`.

### 3.5 Mixed-Precision Training

Automatic mixed precision (AMP) is enabled when running on CUDA:
- `torch.autocast(device_type="cuda")` wraps the forward + loss computation.
- `torch.GradScaler` handles gradient scaling for numerical stability.

### 3.6 Gradient Clipping

| Config Key | Default | Description |
|------------|---------|-------------|
| `max_grad_norm` | 0.5 | Max gradient norm (0 = disabled) |

When enabled, `clip_grad_norm_` is called after `scaler.unscale_()` and before `scaler.step()`.

### 3.7 NaN Detection

If `loss` becomes NaN or Inf during training:
1. Log `nan_detected`, `nan_epoch`, `nan_batch` to W&B.
2. Set run summary `termination_reason = "nan_loss"`.
3. Raise `ValueError` to terminate the run.

### 3.8 Training Loop

```
for epoch in 1..epochs:
    model.train()
    for batch in train_loader:
        ① Forward: pred = model(poses, imgs)
        ② Loss:   L = chamfer_weight * chamfer(pred, tgt) + mse_weight * WeightedLoss(pred, tgt)
        ③ NaN check
        ④ Backward: scaler.scale(L).backward()
        ⑤ Gradient clipping (if enabled)
        ⑥ Optimizer step + scaler update
        ⑦ Scheduler step (if enabled)

    model.eval()
    for batch in val_loader:
        ① Forward: pred = model(poses, imgs)
        ② Denormalise pred and tgt (if normalisation was used)
        ③ Compute mean_point_distance_per_sample (in physical units)
        ④ Accumulate per-session RMSE and Chamfer

    val_loss = mean(val_running_list)
    if val_loss < best_val → save checkpoint + update leaderboard
```

### 3.9 DataLoader Configuration

| Parameter | Train | Val |
|-----------|-------|-----|
| `batch_size` | 32 (YAML) | 32 |
| `shuffle` | `True` | `False` |
| `drop_last` | `True` | `True` |
| `num_workers` | 16 | 16 |
| `persistent_workers` | `True` | `True` |
| `pin_memory` | `True` | `True` |
| `prefetch_factor` | 1 | 1 |

---

## 4. Validation

### Metrics

All validation metrics are computed **after denormalisation** using the training-set statistics, reporting errors in physical units (metres).

| Metric | Function | W&B Key |
|--------|----------|---------|
| Val loss (model selection) | `mean_point_distance_per_sample` → mean over batches | `val/loss` |
| Val loss std | `np.std(val_running_list)` | `val/loss_std` |
| Per-session RMSE | Per-sample mean L2 → grouped by session path | `val/session/{name}/rmse` |
| Per-session Chamfer | Inline symmetric Chamfer (via `torch.cdist`) → grouped | `val/session/{name}/chamfer` |

### Best-Model Selection

The checkpoint with the lowest `val/loss` across all epochs is saved as `best.pt`. The current-best is also registered with the `BestRunsManager` leaderboard.

---

## 5. Outputs & Artifacts

### Experiment Directory

```
results/
├── checkpoints/
│   └── {exp_name}/
│       ├── config.yaml          # Merged config (YAML + CLI + sweep overrides)
│       ├── best.pt              # Best checkpoint (by val loss)
│       ├── last_weights.pt      # Final epoch state_dict (weights only)
│       └── training_log.txt     # Model summary + per-epoch train/val losses
└── best_runs/
    ├── leaderboard.json         # Top-5 experiments sorted by val_loss
    └── {exp_name}/              # Copied checkpoint dirs for top-5 runs
        ├── config.yaml
        ├── best.pt
        └── training_log.txt
```

### Checkpoint Format (`best.pt`)

```python
{
    "epoch": int,
    "state_dict": OrderedDict,        # model.state_dict()
    "optimizer": dict,                 # optimizer.state_dict()
    "val_loss": float,                 # best validation loss
    "config": {
        "img_size_H": int,
        "img_size_W": int,
        "n_digits": int,
        "n_markers": int,
        "with_att": bool,
    }
}
```

### BestRunsManager

Maintains a **top-5 leaderboard** (`leaderboard.json`) sorted ascending by `val_loss`. Each experiment occupies at most one slot (its personal-best epoch). When a new run beats the worst entry, the worst entry's directory is deleted and replaced.

**Leaderboard entry format:**
```json
{
    "exp_name": "09_02_20_05_...",
    "val_loss": 0.03181,
    "epoch": 7,
    "timestamp": "2026-02-09T21:14:43"
}
```

### Training Log Format

Each `training_log.txt` begins with a model summary block (parameter counts, inference time, session counts), followed by per-epoch lines:

```
epoch 001  train 0.6182  val 0.0428 ± 0.0177
epoch 002  train 0.2534  val 0.0368 ± 0.0189
...
```

### W&B Logging Schema

| Key | When | Description |
|-----|------|-------------|
| `train/loss_step` | Each step | Last batch loss |
| `train/lr` | Each step | Current learning rate |
| `val/loss` | Each epoch | Mean per-point L2 distance (metres) |
| `val/loss_std` | Each epoch | Std of per-batch val losses |
| `val/session/{name}/rmse` | Each epoch | Per-session mean L2 |
| `val/session/{name}/chamfer` | Each epoch | Per-session Chamfer distance |
| `model/*` | Once (init) | Parameter counts, size, inference time |
| `dataset/*` | Once (init) | Train/val session counts |
| `best_val` | Summary | Best validation loss across all epochs |

### Visualisation Viewers

Two optional [viser](https://viser.studio) debug viewers ship under
`src/dlofeel_state/viz/`, each run directly as a script:

| Module | Shows |
|---|---|
| `viz/dataset.py` | A dataset split: ground-truth vertex chain, EE pose, DIGIT frames |
| `viz/predictions.py` | A `preds_gts.h5` written by `dlofeel-eval`: prediction against ground truth, per-frame loss, DIGIT frames |

Both hard-code their input paths at the top of the file; edit those before running. They
are inspection tools only — no metric, checkpoint, or published number depends on them.

> **Colour channels.** `viz/predictions.py` decodes its DIGIT thumbnails through
> `_decode_image_1d` from `data/dataset.py`, which converts `bgr8` payloads to **RGB**.
> The corresponding viewer in the original research tree imported a different decoder,
> one that returned `bgr8` payloads unchanged, so its tactile thumbnails show red and
> blue swapped relative to this one. The difference is confined to the debug viewer:
> the dataset feeding the model has always applied the RGB conversion, so no metric,
> checkpoint, or published number is affected. Expect the channel order here to differ
> from the tactile images printed in the paper's figures.

---

## 6. Hyperparameter Sweeps

The production hyperparameters in `default.yaml` came from two W&B sweeps. The sweep
YAMLs themselves are not part of this release (they encode a private W&B
entity/project and the original CLI's flag names); the search spaces are recorded
here so the reported numbers can be situated, and are straightforward to re-create
against `dlofeel-train`.

### Exploration Sweep

- **Method:** Bayesian optimisation
- **Metric:** `val/loss` (minimise)
- **Early termination:** Hyperband (`min_iter=3`, `η=2`)
- **Fixed flags:** `--with_att --use_diff_images --use_blank_ref --single_encoder --normalize --normalize_inputs --num_resample_points=32 --epochs=10 --batch_size=32`

**Search space:**

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `lr` | log-uniform | [1e-5, 1e-3] |
| `dropout` | categorical | {0.1, 0.2, 0.3} |
| `weight_decay` | log-uniform | [1e-5, 1e-3] |
| `max_grad_norm` | categorical | {0.5, 1.0, 2.0} |
| `chamfer_weight` | categorical | {0.0, 0.1, 0.5, 1.0} |
| `mse_weight` | categorical | {0.5, 1.0} |
| `w_edges` | categorical | {0.0, 0.05, 0.1, 0.5} |
| `w_lengths` | categorical | {0.0, 0.05, 0.1, 0.5} |
| `w_kb` | categorical | {0.0, 0.01, 0.02, 0.1} |
| `image_backbone.fusion_layers` | categorical | 3 configs |
| `image_backbone.unfreeze_last_n_blocks` | categorical | {2, 3, 4} |
| `image_backbone.backbone_lr_factor` | categorical | {0.05, 0.1, 0.2} |
| `lr_warmup_pct` | categorical | {0.05, 0.1, 0.15} |

### Refined Sweep

Narrowed ranges based on findings from the exploration sweep:

| Parameter | Change from the exploration sweep | Rationale (from sweep analysis) |
|-----------|--------------------------|--------------------------------|
| `lr` | [1e-5, 1e-4] (narrowed) | Higher LR was worse (green correlation) |
| `backbone_lr_factor` | [0.5, 1.0] (shifted up) | Higher was better (red correlation) |
| `chamfer_weight` | [0.2, 1.0] (removed 0) | Non-zero was better (red correlation) |
| `w_kb` | [0.001, 0.05] (lowered) | Higher was worse (green correlation) |
| `dropout` | [0.1, 0.3] log-uniform | Lower was better |
| `fusion_layers` | Added `[12]` only and `[3,6,9,12]` DPT-style | Testing more layer configs |

### Running a Sweep

Write a sweep YAML whose `command` invokes `dlofeel-train` with your own
`--wandb_entity` / `--wandb_project`, then:

```bash
# Create sweep
wandb sweep <your-sweep>.yaml

# Start agent (uses the returned sweep ID)
wandb agent <entity>/<project>/<sweep_id>
```

---

## 7. Session-Scaling Experiment

**Script:** `experiments/fig4_data_efficiency/train_session_scaling.py`
(driver: `experiments/fig4_data_efficiency/run_session_scaling.sh`)

Measures how model performance scales with the number of training sessions. This
script keeps its own argparse CLI rather than going through `dlofeel-train`, because
it drives many training runs per invocation.

**Protocol:**

1. For each session count \(n \in \{1, 2, \ldots, N_\text{sessions}\}\):
   - Draw \(k\) random subsets of \(n\) sessions.
   - For each subset: train a fresh model for a fixed number of epochs.
   - Evaluate L2 + Chamfer loss on the **full** test set.
2. Save per-trial and aggregated (mean ± std) results to JSONL + CSV, incrementally, with `--resume` for crash recovery.

This requires **non-packed** datasets (per-file HDF5) since it needs to select subsets of sessions by filtering dataset indices.

### 7.1 Direction-Input Experiment

**How to run:** `dlofeel-train --model_type direction --normalize_inputs false ...`

This experiment swaps the learned tactile backbone for `DLODirectionModel`, which:

- extracts one 2D direction vector per DIGIT sensor using a classical CV line-fitting pipeline,
- projects those directions into latent tokens,
- and reuses the same pose encoder + cross-attention decoder structure as the main model.

Key experiment settings in the saved run:

- `with_att=true`
- `use_diff_images=true`
- `use_blank_ref=true`
- `normalize=true`
- `normalize_inputs=false`
- `batch_size=32`
- `epochs=10`

Saved best result:

- `val_l2_mean=0.05063406723587171`
- `val_chamfer_mean=0.04070513910987425`

The model is described in [model.md](model.md) §10.2.

---

## 8. Model Summary Logging

`log_model_info()` is called once at training start and reports:

- **Parameter counts:** Total, trainable, frozen (with percentages).
- **Model size:** In MB (assuming float32).
- **Inference time:** Mean wall-clock over 10 forward passes of one full batch (after 3
  warmup runs), with CUDA sync. The inputs are random dummy tensors, so the figure covers
  the forward pass alone, excluding data loading and preprocessing.
- **Per-component breakdown:** Parameter counts for each top-level submodule.
- **Dataset sessions:** Train and val session counts.

Output is written to the console, `training_log.txt`, and W&B (under `model/*` keys).

**Example output (best run):**

```
Total parameters:       24,796,167
Trainable parameters:    6,377,991 (25.7%)
Frozen parameters:      18,418,176 (74.3%)
Model size:                  94.59 MB
Mean inference time:         60.52 ms (batch_size=32)

Per-component breakdown:
  img_enc                   22,018,692 params (3,600,516 trainable)
  pose_enc                   2,633,728 params (2,633,728 trainable)
  cross_attn                    66,048 params (66,048 trainable)
  simple_decoder                10,627 params (10,627 trainable)
```

> **On the 60.52 ms.** That is one run's own `log_model_info()` line, not the paper's
> reported latency. It is a per-batch (`batch_size=32`) forward-pass timing produced by
> whatever GPU that particular training job ran on, and it moves with the hardware, the
> torch build, and the arm's parameter count — the released K = 16 checkpoint's own
> `training_log.txt`, for instance, records a different value again. The citable number
> is the **61.5 ms** in the [README](../README.md), measured on an RTX 5090; quote that
> one. Read the block above as an illustration of the log format.

---

## 9. Config Reference (Production Defaults)

Complete training-relevant config from `src/dlofeel_state/configs/default.yaml`:

```yaml
# Training hyperparameters
batch_size: 32
epochs: 20
lr: 7.e-05
weight_decay: 1e-04
dropout: 0.1
seed: 42
num_workers: 16
deterministic: true

# LR Scheduler
use_lr_scheduler: true
lr_warmup_pct: 0.1
lr_div_factor: 10.0
lr_final_div_factor: 100.0

# Gradient clipping
max_grad_norm: 0.5

# Loss weights
chamfer_weight: 0.25
mse_weight: 1.0
w_nodes: 1.0
w_edges: 0.1
w_lengths: 0.5
w_kb: 0.01

# Data flags
with_att: true
only_pose: false
use_diff_images: true
use_blank_ref: true
normalize: true
normalize_inputs: true
num_output_pts: 32
n_control_points: 32
past_window_steps: 1
max_time_delta: 0.13
n_markers: 14

# Image dimensions
img_size_H: 320
img_size_W: 240

# Normalization statistics (K = 16, matching the released checkpoint)
norm_stats_path: src/dlofeel_state/configs/norm_stats_k16.yaml
```

> The YAML ships `num_output_pts: 32` (the original packing default) while the
> released checkpoint and the parity fixture are \(K = 16\). Reproducing that
> checkpoint means passing `--num_resample_points 16 --n_control_points 16`; the
> shipped `norm_stats_k16.yaml` already matches \(K = 16\).

---

## 10. File Summary

```
src/dlofeel_state/
├── engine/
│   ├── train.py                 # Training loop, BestRunsManager, log_model_info, main()
│   ├── evaluate.py              # Evaluation loop + main()
│   └── metrics.py               # Per-session metric reduction
└── configs/
    ├── train.py                 # TrainConfig (the dlofeel-train CLI surface)
    ├── eval.py                  # EvalConfig  (the dlofeel-eval CLI surface)
    ├── default.yaml             # Model + training config
    └── norm_stats_k16.yaml      # Normalization statistics (K = 16)

configs/eval/k16.yaml            # Example draccus --config_path override for eval

experiments/                     # Paper-figure ablation drivers
├── fig4_data_efficiency/        # Session-scaling (§7)
├── fig5_transfer/               # Cross-cable transfer
├── fig6_noise/                  # Tactile-noise robustness
├── k_resample/                  # Marker-count (K) ablation + norm-stat generation
├── table1_model_comparison/     # Backbone / pose-only / spline comparisons
└── table2_sensor_ablation/      # Sensor-drop ablations

results/                         # Created by a run (not in the repo)
├── checkpoints/{exp_name}/      # Per-experiment artifacts
│   ├── config.yaml
│   ├── best.pt
│   ├── last_weights.pt
│   └── training_log.txt
└── best_runs/
    ├── leaderboard.json         # Top-5 global leaderboard
    └── {exp_name}/              # Copied best-run checkpoints
```

### Tests

```bash
# Fast: one real training step + one eval run through the console-script entrypoints
.venv/bin/python -m pytest tests/test_train_smoke.py tests/test_eval_cli.py -v

# Checkpoint parity (needs the checkpoint; skips without it)
export DLOFEEL_PARITY_CKPT=<path>/best.pt
.venv/bin/python -m pytest tests/test_parity.py -v
```
