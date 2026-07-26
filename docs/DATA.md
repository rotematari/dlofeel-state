# Obtaining and Staging the Data

How to get the DLOFeel-State dataset and trained weights, and where to put them
so the training scripts and the experiment drivers find them.

This document covers **acquisition and staging only**. For what is *inside* the
data (HDF5 schemas, coordinate frames, synchronization, preprocessing,
normalization formulation, loader behaviour) see [`data.md`](data.md). For the
command-line interface that consumes it, see [`training.md`](training.md).

Acronyms used here: DLO (deformable linear object), DIGIT (the vision-based
tactile sensor), EE (end effector), GT (ground truth), HF (Hugging Face Hub),
K (number of resampled shape vertices the model predicts).

---

## 1. What the dataset is

Synchronized tactile plus proprioceptive recordings of a DLO held at both ends
by two grippers, with optical motion-capture ground truth for its 3D shape.

| Component | Details |
|-----------|---------|
| Tactile sensing | 4x DIGIT sensors (serials `D21118`, `D21122`, `D21123`, `D21124`), one per finger, RGB 320 x 240 px (H x W) |
| Grippers | Two Robotiq 2F-85 parallel grippers holding opposite ends of the DLO. G1 is fixed to the workspace; G2 is mounted on a Kinova Gen3 7-DOF arm |
| Proprioception | 7D EE pose (3D position + unit quaternion), plus the fixed-frame pose of G1 |
| Ground truth | OptiTrack (NatNet) motion capture tracking 14 reflective markers affixed along the DLO |
| Capture rate | ~15 Hz (throttled ROS 2 topics; offline approximate-time sync with a 50 ms tolerance) |

Each sample is one synchronized frame: 4 DIGIT images, the 7D EE pose, and the
14 x 3 GT marker positions. The 14 unordered markers are sorted and B-spline
resampled to K evenly spaced vertices at load time (K = 16 for the released
model). Frames missing any stream, or with a marker count other than 14, are
discarded during extraction.

Sessions are recorded as complete episodes; the test split is held-out whole
sessions, not a random frame split, so evaluation is on unseen cable
configurations and trajectories.

---

## 2. Cables and splits

Three DLOs were recorded. `c1` is the primary cable that the released model is
trained and evaluated on; `c2` and `c3` are the transfer-learning cables.

| Cable | Description | Role |
|-------|-------------|------|
| `c1` | Braided steel wire, 1.5 mm diameter, 550 mm length | Primary DLO. Training + evaluation of the released model |
| `c2` | USB cable, 4.5 mm diameter, 550 mm length | Cross-DLO transfer (fine-tuning) |
| `c3` | Braided steel wire, 2.0 mm diameter, 750 mm length | Cross-DLO transfer (fine-tuning) |

### `c1` split sizes

| Split | Sessions | Samples | Percentage |
|-------|----------|---------|------------|
| Train | 36 | 196,968 | 90.1% |
| Test | 4 | 21,563 | 9.9% |
| **Total** | **40** | **218,531** | **100%** |

The full per-session breakdown is in [`data.md`](data.md) section 3; these
totals are the sum of that table.

> **Note on sample counts.** The per-session totals above are derived from the
> dataset itself. An earlier draft of the paper reports slightly different
> totals (196,678 train / 21,853 test); the numbers here are authoritative for
> this release. <!-- NEEDS-CONFIRM: Rotem to reconcile with the paper's Table/text before publishing -->

The two accounts agree on the grand total (218,531) and on the session counts
(36 train, 4 test); they differ only in where the split boundary falls, by
exactly 290 samples.

For `c2` and `c3` the fine-tuning study used roughly 50,000 training and 14,000
test samples per cable. Those figures are as reported in the paper and are not
derived from a per-session table here.

---

## 3. Getting the data and weights

The dataset and the trained checkpoints are distributed via the Hugging Face
Hub. [`scripts/download_data.py`](../scripts/download_data.py) wraps
`huggingface_hub.snapshot_download`:

```bash
# Dataset only
python scripts/download_data.py --what dataset --dest data/

# Trained weights only
python scripts/download_data.py --what weights --dest data/

# Both (the default for --what)
python scripts/download_data.py --what all --dest data/
```

Artifacts land in `<dest>/dataset/` and `<dest>/weights/`. `--dest` defaults to
`<repo_root>/data`, resolved from the git root, so it never depends on your
working directory.

> **The Hub repository IDs are placeholders.** `DATASET_REPO_ID` and
> `WEIGHTS_REPO_ID` in [`scripts/download_data.py`](../scripts/download_data.py)
> name repositories that do not exist yet; they are finalized when the packed
> dataset and checkpoints are uploaded. Until then the script fails with a Hub
> "repo not found" error. The interface (constants, CLI, and the two download
> functions) is stable now so code and docs can wire against it in advance.

Nothing in this repository requires the download. The parity gate and the
dataset smoke test run off the committed 8-frame sample (section 7), and the
checkpoint-dependent tests skip cleanly when no checkpoint is present.

---

## 4. Staging layout

There are two ways to point the code at your data.

### Option A: explicit flags (recommended for one-off runs)

Both console scripts take dataset roots as arguments, so the data can live
anywhere:

```bash
dlofeel-eval \
  --ckpt_path /abs/path/to/best.pt \
  --data_root /abs/path/to/c1/ds/test \
  --blank_ref_root /abs/path/to/digit_blank_closed_refs \
  --out_dir results_eval
```

`dlofeel-train` takes `--train_data_root`, `--val_data_root`, and
`--blank_ref_root`. Nothing in the package hard-codes a dataset location.

### Option B: stage under `src/datasets/` (required by the experiment drivers)

The shell drivers under [`experiments/`](../experiments) reproduce the paper's
tables and figures. Each one `cd`s to the repository root and then uses
repo-root-relative dataset paths, so it expects this layout:

```
src/datasets/
├── c1/ds/train/<session>/sample_*.h5     # primary cable, train split
├── c1/ds/test/<session>/sample_*.h5      # primary cable, test split
├── c2/ds/{train,test}/<session>/...      # transfer cable 2
├── c3/ds/{train,test}/<session>/...      # transfer cable 3
└── ds/digit_blank_closed_refs/           # shared DIGIT blank references
    ├── D21118.png
    ├── D21122.png
    ├── D21123.png
    └── D21124.png
```

Note that the blank references sit at `src/datasets/ds/`, one level above the
per-cable directories, because every cable shares them.

Symlinking is the cheapest way to satisfy this without copying:

```bash
mkdir -p src/datasets
ln -s /abs/path/to/downloaded/c1 src/datasets/c1
```

Repeat for `c2` and `c3` if you are reproducing the transfer experiments, and
symlink the blank-reference directory to `src/datasets/ds`. The cross-cable
normalization pass additionally wants a `c1` + `c2` + `c3` union directory at
`src/datasets/combined/ds/train`; build it with
[`experiments/fig5_transfer/build_combined_union.py`](../experiments/fig5_transfer/build_combined_union.py)
rather than by hand.

`src/datasets/` is git-ignored in full, so staging the data leaves
`git status` clean: neither the symlinks nor the `*.h5` files under them show up
as changes. Nothing in that directory is tracked.

---

## 5. Reference (blank) images

Background subtraction isolates the tactile contact pattern from each DIGIT
sensor's static background. With `use_blank_ref` enabled (the released model's
setting) the background comes from four fixed per-camera PNGs captured once with
the gripper closed and no cable contact, one per sensor serial. Pass the
directory holding them via `--blank_ref_root` on either console script.

Four such PNGs ship in this repository as the format example, at
[`tests/fixtures/parity/blank_refs/`](../tests/fixtures/parity/blank_refs):
`D21118.png`, `D21122.png`, `D21123.png`, `D21124.png`, each RGB 320 x 240
(H x W). Filenames must match the sensor serials.

The alternative, `use_blank_ref=false`, uses per-session references
(`digit_refs/<session>/digit_refs.h5`) instead. See [`data.md`](data.md)
section 5 for both formats.

---

## 6. Normalization statistics

Inputs and targets are z-score normalized with statistics computed over the
training split. The release ships the statistics matching the K = 16 model at
[`src/dlofeel_state/configs/norm_stats_k16.yaml`](../src/dlofeel_state/configs/norm_stats_k16.yaml),
holding three blocks: `digit_image_stats`, `ee_position_stats`, and
`per_marker_stats` (shape `[K, 3]`).

Because `per_marker_stats` is K-dependent, a run at a different K needs its own
statistics file. Generate one with
[`experiments/k_resample/gen_k_norm_stats.py`](../experiments/k_resample/gen_k_norm_stats.py):

```bash
python experiments/k_resample/gen_k_norm_stats.py \
  --K 64 \
  --train_root src/datasets/c1/ds/train \
  --blank_ref_root src/datasets/ds/digit_blank_closed_refs \
  --out src/dlofeel_state/configs/norm_stats_k64.yaml
```

Three things to know about that command:

- `--base_stats` defaults to the shipped `norm_stats_k16.yaml` and normally needs
  no flag. The base file supplies only the K-independent `digit_image_stats` and
  `ee_position_stats` blocks, which are identical in every per-K file; the
  `per_marker_stats` block is measured from your data.
- `--K 32` is the exception. It is the published-`c1` anchor and copies
  `per_marker_stats` straight from the base rather than measuring it, so it needs
  a 32-marker base file, which this release does not ship. Run it with
  `--recompute_all` to measure the block from data instead. Passing a shorter base
  is an error, not a silent copy-through.
- Add `--recompute_all` in any case when generating statistics for `c2`, `c3`, or
  the union, whose DIGIT image and EE pose distributions differ from `c1`. Without
  it those two blocks are copied from the base file rather than measured.

The generator is idempotent: it reuses an existing output file whose
`per_marker_stats` already has K rows unless you pass `--force`. Run
`--selftest` to exercise it with no data present, or `--help` for the full
flag list.

---

## 7. The committed sample

[`tests/fixtures/parity/samples/`](../tests/fixtures/parity/samples) holds 8
real frames from the `c1` test split (session `01_13_14_46_deg115`), about
7 MB total. **These are the only data files committed to git**, and they exist
so the repository is testable out of the box:

- `tests/test_parity.py` reproduces the original implementation's predictions on
  these 8 frames against a committed golden snapshot.
- `tests/test_dataset_smoke.py` constructs the dataset over them and checks the
  output tensor shapes.

The session subdirectory is preserved so the loader's recursive directory walk
discovers the frames exactly as it would in the full dataset. The fixture also
carries its own `config.yaml`, `norm_stats.yaml`, `blank_refs/`, and
`golden_preds.h5`, making it fully self-contained.
[`tests/fixtures/parity/README.md`](../tests/fixtures/parity/README.md) records
the fixture's provenance and the exact command that produced the golden.

The K = 16 checkpoint itself (137 MB) is not committed. Tests that need it read
its path from the `DLOFEEL_PARITY_CKPT` environment variable and skip when it is
unset, which is the expected state in public CI.
