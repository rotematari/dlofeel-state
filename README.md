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

## Reproducing the paper

Each directory holds the driver scripts for one paper artifact.

| Paper artifact | Directory | Arms with a shipped driver |
|---|---|---|
| Table I: model comparison | `experiments/table1_model_comparison/` | Sparsh ViT, ResNet-18, pose-only (`run_model_comparison.sh`); w/o B-spline (`run_diff_bspline.sh`) |
| Table II: sensor removal + runtime sensor faults | `experiments/table2_sensor_ablation/` | all |
| Fig. 4: data efficiency vs. number of samples | `experiments/fig4_data_efficiency/` | all |
| Fig. 5: transfer to unseen DLOs | `experiments/fig5_transfer/` | `c3` (2 mm steel) scratch + fine-tune (`run_c3_xblock.sh`); combined-model eval (`run_x2_combined_eval.sh`) |
| Fig. 6: robustness to tactile-image noise | `experiments/fig6_noise/` | all |
| Choice of K (vertex resampling) | `experiments/k_resample/` | all |

Two Table-I arms are **model-supported but ship no driver**: the pose+direction variant
(`--model_type direction`) and the w/o-physics-loss variant (`--use_weighted_loss`) are
implemented in the package and reachable from the CLI, but no shell script in
`experiments/` launches them. Likewise, Fig. 5 ships the `c3` transfer runs only; the
`c2` (4.5 mm USB) fine-tuning driver is not included, and `run_x2_combined_eval.sh`
evaluates a combined C1+C2+C3 checkpoint that no shipped script trains.

The drivers `cd` to the repository root and use repo-root-relative dataset paths; the
staging layout they expect is in [docs/DATA.md](docs/DATA.md).

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
experiments/         # Per-artifact ablation drivers (see the table above)
scripts/             # Data download, bag processing, dataset packing
docs/                # Data, model, loss, training, acquisition, third-party docs
tests/               # Parity gate, smoke tests, doc-hygiene invariants
third_party/sparsh   # Sparsh submodule (not redistributed; CC-BY-NC 4.0)
```

## Reproducibility

The release ships a parity gate, not just tests. `tests/test_parity.py` loads a
released checkpoint (path taken from the `DLOFEEL_PARITY_CKPT` environment variable),
runs the packaged evaluation path over the 8 real frames committed at
`tests/fixtures/parity/samples/`, and requires the predictions and the loaded ground
truth to match `tests/fixtures/parity/golden_preds.h5` within `atol=1e-5`. That golden
snapshot was generated by the original research codebase before any code was ported,
so the target was fixed in advance: the packaged inference path is verified to
reproduce the numbers behind the paper, not merely to run.

```bash
pytest                                    # full suite
export DLOFEEL_PARITY_CKPT=<path>/best.pt # then re-run to include the parity gate
pytest tests/test_parity.py -v
```

The suite needs no data download: the committed 8-frame fixture is the only data in
git. The checkpoint-dependent gates (`test_parity.py`, `test_model_loads_ckpt.py`,
`test_eval_cli.py`) skip cleanly when `DLOFEEL_PARITY_CKPT` is unset, since the
137 MB checkpoint is not committed.

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
