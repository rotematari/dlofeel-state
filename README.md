# DLOFeel-State

Tactile-primary global shape estimation of deformable linear objects. DLOFeel-State
maps four DIGIT tactile images plus a 7-D end-effector pose to the full 3-D vertex
chain of a cable held at both ends, with **no visual input of the object**: the global
shape is inferred from contact at the grippers and proprioception alone. A Sparsh
vision-transformer tactile backbone feeds an attention decoder and a differentiable
B-spline head, trained under a physics-informed discrete-elastic-rod loss.

Acronyms used below: DLO (deformable linear object), DIGIT (vision-based tactile
sensor), EE (end effector), ViT (vision transformer), RMSE (root-mean-square error),
DER (discrete elastic rods), K (number of predicted shape vertices).

## Result

Released K = 16 model, evaluated on held-out test sessions of the primary cable
(braided steel wire, 1.5 mm diameter, 550 mm length):

| Metric | Value |
|---|---|
| Per-vertex RMSE | **30.52 ± 15.22 mm** |
| Chamfer distance | **27.29 ± 12.30 mm** |
| Inference | 61.5 ms (~16 Hz), RTX 5090 |
| Parameters | 5.33 M trainable / 23.75 M total |

Parameter counts are for the released K = 16 checkpoint: 23,747,607 total tensor
elements, of which 5,327,367 carry gradients under the default freezing strategy
(the Sparsh backbone is frozen except its last 2 blocks). The count is K-dependent,
so a model trained at a different K will differ.

The 61.5 ms is the latency reported in the paper, measured on an RTX 5090; the paper
does not state the batch size behind it, so it is quoted here exactly as published.
The released K = 16 run logs its own reproduced measurement: **64.20 ms at
`batch_size=32`**, from the `log_model_info()` summary at the head of its
`training_log.txt` (mean wall clock over 10 CUDA-synced forward passes after 3
warmups, forward pass only, excluding data loading). Treat 61.5 ms as the citable
figure and 64.20 ms as what this checkpoint's own training run recorded.

The test split is whole held-out sessions, not a random frame split, so these numbers
are on unseen cable configurations and trajectories.

## Method

- **Tactile backbone.** Sparsh ViT (`vit_small`) over 6-channel temporal image pairs
  (current + past frame) from each of the four DIGIT sensors, one shared encoder,
  partially unfrozen (last 2 blocks) with multi-scale feature fusion over ViT layers
  9-12 (`learned_sum`).
- **Pose encoder.** An MLP that lifts the 7-D EE pose (position + unit quaternion)
  into one latent token per output query.
- **Decoder.** Cross-attention from the pose-derived queries into the tactile tokens,
  then self-attention across the queries, then a per-query MLP head producing the
  control points of the shape.
- **B-spline head.** A differentiable cubic B-spline layer evaluates K vertices on the
  curve through the predicted control points, acting as a built-in smoothness prior.
- **Loss.** A physics-informed composite over vertex position, unit edge direction
  (tangent), edge length, and curvature binormal, following the DER formulation, plus
  a symmetric Chamfer term for permutation-invariant matching.

Full architecture: [docs/model.md](docs/model.md). Full loss formulation and weights:
[docs/loss.md](docs/loss.md). Data pipeline: [docs/data.md](docs/data.md).

## Installation

Requires Python >= 3.12. The Sparsh tactile encoder is consumed as a git submodule
(it is not redistributed here), so the clone must pull submodules:

```bash
git clone --recurse-submodules <repo-url>
cd dlofeel-state
uv venv && uv pip install -e .      # or: python3.12 -m venv .venv && .venv/bin/pip install -e .
```

Already cloned without `--recurse-submodules`? Run
`git submodule update --init --recursive`.

## Quickstart

Evaluating the released checkpoint on a test split:

```bash
dlofeel-eval \
  --ckpt_path <path>/best.pt \
  --data_root <path>/test \
  --blank_ref_root <path>/digit_blank_closed_refs \
  --out_dir results_eval
```

`--model_config_path` defaults to `<ckpt_dir>/config.yaml`, so a checkpoint evaluates
under the architecture it was trained with and no config flag is needed.

To get the dataset and the trained weights, and to learn where to stage them so the
experiment drivers find them, see [docs/DATA.md](docs/DATA.md). Training commands and
the full flag reference for `dlofeel-train` / `dlofeel-eval` are in
[docs/training.md](docs/training.md).

## Repository layout

```
src/dlofeel_state/
├── models/          # DLOStateModel, Sparsh/ResNet backbones, pose encoder, B-spline
├── losses/          # Chamfer, weighted physics loss, DER edges/curvature, bending
├── data/            # HDF5 dataset, spline resampling, transforms, sensor-drop masks
├── engine/          # train.py, evaluate.py, metrics.py (the console-script entrypoints)
├── configs/         # TrainConfig/EvalConfig CLI surfaces, default.yaml, norm stats
└── viz/             # Dataset and prediction visualisation
configs/             # Example draccus YAML overrides for the console scripts
experiments/         # Per-artifact ablation drivers, one directory per paper table/figure
scripts/             # Data download, bag processing, dataset packing
docs/                # Data, model, loss, training, acquisition, third-party docs
tests/               # Parity gate, smoke tests, doc-hygiene invariants
third_party/sparsh   # Sparsh submodule (not redistributed; CC-BY-NC 4.0)
```

## Citation

If you use this code, please cite the paper:

> Physics-Informed Tactile-based Shape Estimation of Deformable Linear Objects.

The paper is under review; the full reference (venue, DOI) will be added here on
acceptance. Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

## License

This repository's own source code is MIT-licensed; see [LICENSE](LICENSE).

The Sparsh tactile encoder is a separate work by Meta AI, licensed **CC-BY-NC 4.0**
and **not redistributed here**: it is pulled as a pinned git submodule at
`third_party/sparsh`. Because Sparsh and its pretrained weights are NonCommercial,
**any commercial use of this project is gated by Sparsh's terms**. The MIT grant
covers only this repository's own code and does not relax them. Every third-party
component, its licence, and how it is consumed are recorded in
[NOTICE](NOTICE) and, authoritatively, in [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).

## Acknowledgments

- **Sparsh** (Meta AI) for the pretrained self-supervised tactile encoder that this
  model builds on.
- The **DIGIT** vision-based tactile sensor, which produces the tactile images.
- **Bergou et al., "Discrete elastic rods" (ACM TOG 27(3), 2008)** for the discrete
  elastic-rod formulation underlying the physics-informed loss.
