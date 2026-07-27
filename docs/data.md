# DLOFeel-State Data Documentation

## 1. Overview

This dataset supports training multi-modal deep learning models to estimate deformable linear object (DLO) marker positions from tactile images and robot end-effector (EE) pose. Each sample is a synchronized snapshot of:

- **4 DIGIT tactile cameras** (RGB images, 320 x 240 px)
- **7D end-effector pose** (3D position + unit quaternion orientation)
- **Ground-truth 3D marker positions** along the cable (tracked via optical motion capture)

---

## 2. Data Collection Setup

### Hardware

| Component | Details |
|-----------|---------|
| Robot arm | 7-DOF manipulator (Kinova Gen3, joints `joint_1` .. `joint_7`) |
| Gripper | Robotiq 85 (2-finger, reported as `robotiq_85_left_knuckle_joint`) |
| Tactile sensors | 4x GelSight DIGIT (serials: D21118, D21122, D21123, D21124) |
| Motion capture | NatNet (OptiTrack) for marker tracking and fixed-frame / EE pose |
| Cable | Deformable linear object with 14 physical reflective markers |

### ROS 2 Topics Recorded

Each sensor stream is published through a ROS 2 **throttle node** to enforce a consistent output rate. The throttled topics are:

| Topic | Description |
|-------|-------------|
| `/digit/D211XX/image_raw_throttled` | DIGIT tactile images (4 cameras) |
| `/end_effector_pose_throttled` | EE pose from motion capture |
| `/natnet/fixed_ee_pose_throttled` | Fixed calibration frame pose from motion capture |
| `/natnet/unlabeled_marker_data_throttled` | 3D marker positions from motion capture |
| `/joint_states_throttled` | Robot joint states |

Older recording sessions (`25_12`, `28_12`, `29_12`) used a legacy `FullDataPub` node with `/full_data/...` topic prefixes. The extraction script (`scripts/process_bags.py`) auto-detects the topic format.

### Synchronization

Sensor streams are synchronized **offline** during bag extraction using approximate time synchronization (`scripts/process_bags.py`):

1. All messages are buffered and sorted by header timestamp.
2. The topic with the **fewest messages** is selected as the pivot.
3. For each pivot message, the closest message from every other topic is found via binary search (`bisect.bisect_left`).
4. A match is accepted only if the absolute time difference is within a **50 ms tolerance** (`SYNC_TOLERANCE_NS = 50 × 10⁶ ns`).
5. Each message is consumed at most once (greedy, index-tracking matching prevents duplicates).
6. Frames missing any required stream (4 DIGIT images, EE pose, fixed pose, markers) are discarded.

Additional quality filters applied before saving:
- Frames with marker count ≠ 14 are rejected.
- Frames where any pair of consecutive markers is closer than 2 cm (`MIN_MARKER_DISTANCE_M = 0.02`) are rejected (degenerate cable configuration).
- Self-intersection detection (`SELF_INTERSECTION_THRESHOLD_M = 0.015`) is implemented but currently **disabled** in the codebase.

---

## 3. Dataset Statistics

### Split Summary

| Split | Sessions | Samples | Percentage |
|-------|----------|---------|------------|
| Train | 36 | 196,968 | 90.1% |
| Test | 4 | 21,563 | 9.9% |
| **Total** | **40** | **218,531** | **100%** |

These totals are the sum of the per-session tables below, i.e. the raw session
inventory. The split sizes reported for the released results are 196,678 train /
21,853 test: the same 218,531 samples and the same 36/4 session counts, with the
split boundary placing 290 further samples on the test side. See
[`DATA.md`](DATA.md) for the authoritative split figures.

### Train Sessions (36 total)

| Session | Samples | Config |
|---------|---------|--------|
| `25_12_15_08` | 2,450 | Default cable angle |
| `25_12_15_51` | 4,054 | Default cable angle |
| `25_12_16_44` | 5,512 | Default cable angle |
| `25_12_16_57` | 756 | Default cable angle |
| `25_12_17_29` | 3,195 | Default cable angle |
| `25_12_18_04` | 5,960 | Default cable angle |
| `28_12_10_10` | 5,939 | Default cable angle |
| `28_12_10_24` | 5,190 | Default cable angle |
| `28_12_10_57` | 3,561 | Default cable angle |
| `28_12_11_09` | 5,026 | Default cable angle |
| `28_12_11_24` | 5,940 | Default cable angle |
| `28_12_11_45` | 5,897 | Default cable angle |
| `28_12_11_56` | 5,868 | Default cable angle |
| `28_12_12_09` | 5,857 | Default cable angle |
| `28_12_12_39_deg_45_c` | 5,887 | Cable at 45° |
| `29_12_13_35_deg_45_c` | 5,971 | Cable at 45° |
| `29_12_14_05_deg_60_c` | 3,099 | Cable at 60° |
| `29_12_14_37_deg_60_c` | 5,920 | Cable at 60° |
| `29_12_15_07_deg_135_c` | 4,068 | Cable at 135° |
| `29_12_16_17_deg_135_c` | 2,625 | Cable at 135° |
| `01_13_15_14_deg115` | 2,562 | Cable at 115° |
| `01_13_15_21_deg115` | 9,294 | Cable at 115° |
| `01_13_15_33_deg115` | 9,408 | Cable at 115° |
| `01_13_15_44_deg0` | 7,459 | Cable at 0° |
| `01_13_16_17_deg30` | 9,423 | Cable at 30° |
| `01_13_16_29_deg135` | 4,266 | Cable at 135° |
| `01_13_16_41_deg135` | 671 | Cable at 135° |
| `01_20_17_47_deg135_45` | 9,401 | Cable at 135°/45° |
| `01_20_17_57_deg135_45` | 3,919 | Cable at 135°/45° |
| `01_20_18_25_deg135_135` | 9,451 | Cable at 135°/135° |
| `01_20_18_43_deg135_135` | 8,773 | Cable at 135°/135° |
| `01_22_13_58_deg30_135` | 9,529 | Cable at 30°/135° |
| `01_22_14_09_deg30_135` | 9,094 | Cable at 30°/135° |
| `02_05_15_56_deg75_45` | 4,458 | Cable at 75°/45° |
| `02_05_16_03_deg75_45` | 2,017 | Cable at 75°/45° |
| `02_05_16_30_deg45_135` | 4,468 | Cable at 45°/135° |

### Test Sessions (4 total)

| Session | Samples | Config |
|---------|---------|--------|
| `01_13_14_46_deg115` | 6,740 | Cable at 115° |
| `02_05_16_15_deg75_45` | 4,461 | Cable at 75°/45° |
| `02_05_16_39_deg45_135` | 4,481 | Cable at 45°/135° |
| `25_12_16_28` | 5,881 | Default cable angle |

### Session Naming Convention

Session names follow the pattern `MM_DD_HH_MM[_config]`:
- **Date/time**: `MM_DD_HH_MM` (month, day, hour, minute of recording)
- **Config suffix**: Optional cable configuration descriptor
  - `degXX` = cable initial angle (e.g., `deg115` = 115°)
  - `degXX_YY` = two-angle configuration (e.g., `deg75_45`)
  - `_c` suffix = alternative cable setup variant

### Temporal Properties

| Property | Value |
|----------|-------|
| Capture rate | ~15 FPS (66.7 ms inter-frame interval) |
| Timestamp format | int64, nanoseconds (ROS 2 header stamp) |
| Session duration | 45 s – 300 s per session (typical ~200–400 s) |
| Inter-frame delta (mean) | ~66.7 ms |
| Inter-frame delta (std) | < 1 ms |

---

## 4. Data Formats

### 4.1 Per-Sample HDF5 Files (Raw)

Individual `.h5` files, one per timestep. This is the primary archival format, produced directly from ROS bag extraction by `scripts/process_bags.py`.

```
<data_root>/train/<session>/sample_NNNN.h5
<data_root>/test/<session>/sample_NNNN.h5
```

`<data_root>` is wherever you unpacked the dataset; nothing in this repo depends on an absolute location. `scripts/download_data.py --what dataset` fetches into `data/` by default, and the data-prep scripts and configs carry repo-root-relative placeholder roots under `src/datasets/` that you point at your own copy.

**HDF5 Structure:**

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `timestamp` | scalar | int64 | ROS header timestamp (nanoseconds) |
| `ee_position` | (3,) | float64 | End-effector XYZ position (world frame) |
| `ee_orientation` | (4,) | float64 | End-effector quaternion \([q_x, q_y, q_z, q_w]\) (world frame) |
| `fixed_position` | (3,) | float64 | Fixed calibration frame XYZ position (world frame) |
| `fixed_orientation` | (4,) | float64 | Fixed calibration frame quaternion (world frame) |
| `marker_positions` | (14, 3) | float64 | Ground-truth marker XYZ positions (world frame) |
| `joint_names` | (8,) | bytes | Robot joint names (7 arm + 1 gripper) |
| `joint_positions` | (8,) | float64 | Robot joint angles (radians) |
| `joint_velocities` | (8,) | float64 | Robot joint velocities (rad/s) |
| `digit_images/img_0` | (230400,) | uint8 | DIGIT camera 0 raw buffer (D21118) |
| `digit_images/img_1` | (230400,) | uint8 | DIGIT camera 1 raw buffer (D21122) |
| `digit_images/img_2` | (230400,) | uint8 | DIGIT camera 2 raw buffer (D21123) |
| `digit_images/img_3` | (230400,) | uint8 | DIGIT camera 3 raw buffer (D21124) |

**Image buffer attributes** (per `img_k` dataset):

| Attribute | Value | Description |
|-----------|-------|-------------|
| `height` | 320 | Image height in pixels |
| `width` | 240 | Image width in pixels |
| `step` | 720 | Row stride in bytes (= width × 3 for BGR8) |
| `encoding` | `bgr8` | Pixel format (converted to RGB during loading) |

**Image decoding:** Each raw buffer is 320 × 720 = 230,400 bytes. Reshaped to `(320, 240, 3)` in BGR order, then converted to RGB by channel reversal.

**Notes:**
- Poses and markers are stored in the **world (NatNet) frame**. Coordinate transforms to the fixed frame are applied at load time (non-packed) or during packing.
- Marker count is 14 physical markers per sample (before spline resampling).
- Joint positions/velocities are available but not currently used in training.

---

### 4.2 Packed HDF5 Files

A single `.h5` file per split, pre-processed for faster sequential loading. Produced by `scripts/pack_dataset.py`.

```
<packed_root>/pts32/train.h5
<packed_root>/pts32/test.h5
```

**HDF5 Structure:**

| Field | Shape | Type | Compression | Description |
|-------|-------|------|-------------|-------------|
| `timestamp` | \((N,)\) | int64 | — | All timestamps (nanoseconds) |
| `ee_position` | \((N, 3)\) | float32 | — | EE positions (**fixed frame**) |
| `ee_orientation` | \((N, 4)\) | float32 | — | EE orientations (**fixed frame**) |
| `fixed_position` | \((N, 3)\) | float32 | — | Fixed frame positions (world, retained for reference) |
| `fixed_orientation` | \((N, 4)\) | float32 | — | Fixed frame orientations (world, retained for reference) |
| `resampled_marker_positions` | \((N, K, 3)\) | float32 | — | Spline-resampled markers (**fixed frame**, \(K=32\) default) |
| `digit_images` | \((N, 4, 3, H, W)\) | uint8 | LZ4 (fallback: gzip) | Pre-decoded RGB images in CHW layout. Chunked \((1, 4, 3, H, W)\). |
| `paths` | \((N,)\) | variable-length str | — | Original per-sample file paths |
| `session_ids` | \((N,)\) | int32 | — | Session index for temporal pairing validation |

**Pre-processing applied during packing:**

1. Coordinate transform of EE pose and markers from world frame to fixed frame.
2. BGR → RGB conversion and HWC → CHW transposition.
3. Spline resampling from 14 raw markers to \(K\) evenly-spaced points (default \(K = 32\), cubic B-spline \(k = 3\)).
4. LZ4 compression on images (via `hdf5plugin`), with automatic gzip fallback.
5. Normalization statistics computed and saved to a companion YAML file.

**Key difference from raw format:** Packed files store data **already in the fixed coordinate frame** with markers **already resampled**, so the dataset loader (`src/dlofeel_state/data/dataset.py`) skips both the coordinate transform and spline resampling steps when `is_packed=True`.

**Companion config YAML** (saved alongside the `.h5` file):

```yaml
version: 2
created: <ISO timestamp>
source: <path>
packing_params:
  img_size_H: 320
  img_size_W: 240
  num_resample_points: 32
  spline_degree: 3
  pre_processed: true
timestep_stats: { min_sec, max_sec, mean_sec, std_sec, median_sec }
sample_stats: { total_samples }
norm_stats:
  digit_image_stats: { mean: [R, G, B], std: [R, G, B] }
  ee_position_stats: { mean: [7], std: [7] }
  per_marker_stats:  { mean: [[K, 3]], std: [[K, 3]] }
sessions:
  names: [<list>]
  id_map: { <name>: <int_id> }
  stats: { <per-session stats> }
```

---

## 5. Reference Images

Background subtraction isolates tactile contact patterns from the static DIGIT sensor background. Two reference sources are supported.

### Blank References (Fixed, Camera-Specific)

Captured once with the gripper closed and no cable contact. Used when `use_blank_ref=True`.

```
<data_root>/digit_blank_closed_refs/
├── D21118.png   # DIGIT camera 0 (RGB, 320×240)
├── D21122.png   # DIGIT camera 1
├── D21123.png   # DIGIT camera 2
└── D21124.png   # DIGIT camera 3
```

The four blank references for the parity fixture are committed at `tests/fixtures/parity/blank_refs/`; pass the directory to `--blank_ref_root` on either console script.

### Per-Session References

Captured at the start of each recording session (first frame of each DIGIT stream). Used when `use_blank_ref=False` for more accurate per-session background.

```
<data_root>/digit_refs/<session>/digit_refs.h5
```

**Structure per file:**

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `digit_refs/digit_1` | (230400,) | uint8 | Camera 0 reference (raw BGR8 buffer) |
| `digit_refs/digit_2` | (230400,) | uint8 | Camera 1 reference |
| `digit_refs/digit_3` | (230400,) | uint8 | Camera 2 reference |
| `digit_refs/digit_4` | (230400,) | uint8 | Camera 3 reference |

Each dataset carries the same `height`, `width`, `step`, `encoding` attributes as the sample images.

---

## 6. Coordinate Frames and Transforms

### Frames

| Frame | Description |
|-------|-------------|
| **World (NatNet)** | OptiTrack motion-capture global frame. All raw data is recorded here. |
| **Fixed** | A calibration rigid body tracked by OptiTrack, attached near the robot base. Serves as the canonical reference frame for training. |
| **End-Effector** | Robot tool frame at the gripper. |

### Mathematical Formulation

Let the world-frame quantities be:

- \(\mathbf{p}^\text{ee}, \mathbf{q}^\text{ee}\): EE position and orientation (quaternion xyzw)
- \(\mathbf{p}^\text{fx}, \mathbf{q}^\text{fx}\): Fixed-frame position and orientation
- \(\mathbf{M} \in \mathbb{R}^{N \times 3}\): Marker positions

Denote the rotation matrices obtained via `scipy.spatial.transform.Rotation.from_quat`:

\[
\mathbf{R}_\text{ee} = \text{Rot}(\mathbf{q}^\text{ee}), \quad \mathbf{R}_\text{fx} = \text{Rot}(\mathbf{q}^\text{fx})
\]

**EE pose in fixed frame** (`get_ee_relative_to_fixed`):

\[
\mathbf{t}^\text{fx \to ee} = \mathbf{R}_\text{fx}^\top \left( \mathbf{p}^\text{ee} - \mathbf{p}^\text{fx} \right)
\]

\[
\mathbf{R}^\text{fx \to ee} = \mathbf{R}_\text{fx}^\top \, \mathbf{R}_\text{ee}
\]

The relative quaternion is recovered as \(\mathbf{q}^\text{rel} = \text{Rot}^{-1}(\mathbf{R}^\text{fx \to ee})\). The final 7D EE pose vector is \([\mathbf{t}^\text{fx \to ee};\; \mathbf{q}^\text{rel}]\).

**Markers in fixed frame** (`get_markers_relative_to_fixed`):

\[
\mathbf{M}^\text{fx} = \left( \mathbf{R}_\text{fx}^\top \left( \mathbf{M} - \mathbf{1} \, \mathbf{p}^{\text{fx}\top} \right)^\top \right)^\top
\]

equivalently, for each marker \(\mathbf{m}_i\):

\[
\mathbf{m}_i^\text{fx} = \mathbf{R}_\text{fx}^\top \left( \mathbf{m}_i - \mathbf{p}^\text{fx} \right)
\]

### Typical Value Ranges (Fixed Frame, Meters)

| Axis | Marker Range | EE Position Range |
|------|-------------|-------------------|
| X | [−0.03, 0.27] | [−0.10, 0.10] |
| Y | [−0.23, −0.01] | [−0.10, 0.10] |
| Z | [0.10, 0.34] | [0.47, 0.62] |

### Pose Format

End-effector pose is a 7D vector: `[x, y, z, qx, qy, qz, qw]` — position concatenated with quaternion in the **xyzw convention** (as used by `scipy.spatial.transform.Rotation`).

---

## 7. Preprocessing Pipeline

```
ROS 2 Bags (.mcap)
    │  scripts/process_bags.py
    │  - Deserialize all topics (auto-detect topic naming convention)
    │  - Approximate sync: pivot topic + binary search (50 ms tolerance)
    │  - Validate: marker count == 14, min inter-marker distance >= 2 cm
    │  - Save per-sample HDF5 files
    ▼
Per-Sample HDF5  (218,531 files, ~200 GB)
    │  scripts/pack_dataset.py
    │  - Transform to fixed frame (EE pose + markers)
    │  - Spline resample: 14 markers → K points (default K=32, cubic B-spline k=3)
    │  - Decode & transpose images (BGR→RGB, HWC→CHW)
    │  - LZ4 compression
    │  - Compute normalization statistics
    │  - Save companion YAML config
    ▼
Packed HDF5  (1 file per split, ~40 GB each)
```

> **Legacy path:** The earliest sessions were originally extracted by a reader that relied on pre-synchronized header timestamps from the `FullDataPub` node rather than approximate matching. That reader is not part of this repository; `scripts/process_bags.py` handles both topic-naming conventions and is the only supported extractor.

### Spline Resampling

The 14 raw physical markers are **unordered** (OptiTrack returns an unlabeled point cloud). The resampling pipeline converts them to a consistent, ordered representation:

1. **Sort** markers into a plausible wire path using a momentum-based greedy algorithm (`sort_points_from_origin`):
   - Start point: closest to the coordinate origin.
   - Subsequent points: selected by minimizing a cost combining proximity and directional momentum (weight \(\alpha = 0.5\) by default for CPU, \(0.2\) for GPU variant).

2. **Fit** a B-spline through the sorted points using `scipy.interpolate.splprep`:
   - Degree \(k\): **3** during packing (`scripts/pack_dataset.py`, `--spline_degree`); **5** when resampling at runtime for non-packed data (`src/dlofeel_state/data/dataset.py`).
   - Smoothing: `smoothing=0.1` during packing; `smoothing=0` at runtime.

3. **Resample** to \(K\) evenly-spaced points along the spline arc using `scipy.interpolate.splev` with uniformly spaced parameter values \(u \in [0, 1]\).

> **Note on spline degree discrepancy:** The packing script uses cubic splines (\(k=3\)) with light smoothing, while the runtime resampling path (for non-packed data) uses quintic splines (\(k=5\)) with zero smoothing. Both produce \(K\) output points but may yield subtly different geometries for the same input markers. The checkpoint-parity gate (`tests/test_parity.py`) exercises the runtime path, so the \(k=5\) branch is the one pinned by tests.

---

## 8. Temporal Pairing

`DLOH5SampleDataset` (`src/dlofeel_state/data/dataset.py`) pairs each frame with a recent past frame to provide short-term motion cues. Both frames' images are concatenated along the channel dimension:

\[
\mathbf{I}^\text{pair} = [\mathbf{I}^\text{curr};\; \mathbf{I}^\text{past}] \in \mathbb{R}^{4 \times 6 \times H \times W}
\]

where channels 0–2 are the current frame (RGB) and channels 3–5 are the past frame (RGB), per DIGIT sensor.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `past_window_steps` | **5** | Number of index steps to look back (~333 ms at 15 FPS) |
| `max_time_delta` | 0.1 s | Maximum allowed time gap between paired frames |

**Pairing validation (all conditions must hold):**

1. Past index \(\geq 0\) (within dataset bounds).
2. Past frame is from the **same session** (identical parent directory name).
3. Absolute time delta \(\leq\) `max_time_delta`: \(\lvert t_\text{curr} - t_\text{past} \rvert / 10^9 \leq 0.1\) s.

If any condition fails, the current frame is duplicated as its own pair (zero-velocity fallback).

---

## 9. Normalization

### Statistics Source

Normalization statistics are computed over the full training set (in the fixed frame) and stored in `src/dlofeel_state/configs/norm_stats_k16.yaml`, the stats file shipped with this repo (\(K = 16\) resampled markers, the configuration the released checkpoint was trained at). `scripts/pack_dataset.py` also embeds equivalent statistics in the companion YAML alongside each packed `.h5` file, at whatever \(K\) that packing run used.

Statistics are consumed via the dataset's `norm_stats_path` argument, so a run at a different \(K\) needs a matching stats file: regenerate one with `experiments/k_resample/gen_k_norm_stats.py`.

### Formulation

For a feature vector \(\mathbf{x}\), z-score normalization is:

\[
\hat{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}
\]

and the inverse:

\[
\mathbf{x} = \hat{\mathbf{x}} \cdot \boldsymbol{\sigma} + \boldsymbol{\mu}
\]

Statistics are computed via a single-pass Welford-style accumulation: \(\boldsymbol{\mu} = \bar{\mathbf{x}}\), \(\boldsymbol{\sigma} = \sqrt{\max(\overline{\mathbf{x}^2} - \bar{\mathbf{x}}^2,\; 0)}\).

### DIGIT Image Statistics (per-channel, after \(\div 255\) normalization to \([0, 1]\))

| Channel | Mean | Std |
|---------|------|-----|
| R | 0.0681 | 0.1586 |
| G | 0.0717 | 0.1669 |
| B | 0.0690 | 0.1691 |

> The low means reflect that DIGIT images are mostly dark except where tactile contact occurs.

### End-Effector Pose Statistics (7D, fixed frame)

| Component | Mean | Std |
|-----------|------|-----|
| \(x\) | 0.0052 | 0.0954 |
| \(y\) | 0.0224 | 0.0939 |
| \(z\) | 0.5693 | 0.0486 |
| \(q_x\) | 0.3841 | 0.5447 |
| \(q_y\) | 0.4765 | 0.5309 |
| \(q_z\) | −0.0093 | 0.1526 |
| \(q_w\) | 0.0149 | 0.1521 |

### Marker Position Statistics (16 points × 3, fixed frame)

Per-point mean and std are stored as \((K, 3)\) arrays; the shipped `norm_stats_k16.yaml` carries \(K = 16\). Across those 16 points the per-point means span:

| Axis | Range of per-point means | Shape of the profile |
|------|--------------------------|----------------------|
| X | [−0.007, 0.028] | Near zero, drifting positive toward the far end |
| Y | [−0.111, 0.003] | ~0 at both ends, most negative near the midpoint (the cable sags out of plane) |
| Z | [0.128, 0.338] | Monotonic from the gripped end to the free end |

Per-point standard deviations run 0.013–0.117 (X), 0.002–0.117 (Y), and 0.005–0.055 (Z): the endpoints are far more tightly constrained than the middle of the cable.

### Normalization Order in the Dataset

When both `normalize_inputs=True` and `use_diff_images=True`, the pipeline applies z-score normalization **before** reference subtraction:

```
1. imgs_curr ← normalize(imgs_curr, μ_digit, σ_digit)
2. imgs_prev ← normalize(imgs_prev, μ_digit, σ_digit)
3. refs      ← normalize(refs,      μ_digit, σ_digit)
4. imgs_curr ← imgs_curr − refs
5. imgs_prev ← imgs_prev − refs
6. imgs_combined ← cat(imgs_curr, imgs_prev, dim=1)  →  [4, 6, H, W]
```

This ordering means the subtracted result is no longer zero-mean unit-variance; the difference image has its own implicit distribution.

---

## 10. Data Loading

### Dataset Classes

| Class | File | Role |
|-------|------|------|
| `DLOH5SampleDataset` | `src/dlofeel_state/data/dataset.py` | The loader. Temporal (current + past frame); used by both training and evaluation |
| `DropAugmentDataset` | `src/dlofeel_state/data/augment.py` | Wrapper that expands a base dataset by the 15 sensor-drop masks (train-time augmentation, opt-in via `--drop_augment`) |

> The 15 masks (all-present plus the 14 non-empty drop subsets of 4 sensors) are defined once in `src/dlofeel_state/data/drop_masks.py`, shared by the train-time wrapper and the test-time drop ablation so they cannot diverge.

### Constructor Parameters (`DLOH5SampleDataset`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_dir` | `str \| Path` | — | Path to split directory or packed `.h5` file |
| `is_packed` | `bool` | `True` | Whether `root_dir` points to a packed HDF5 file |
| `blank_ref_root` | `str \| Path` | `""` | Path to blank reference images directory |
| `img_size_H` | `int` | 320 | Target image height (resize applied if mismatch) |
| `img_size_W` | `int` | 240 | Target image width |
| `norm_stats_path` | `str \| Path` | `""` | Path to the normalization-stats YAML (e.g. `src/dlofeel_state/configs/norm_stats_k16.yaml`) |
| `normalize` | `bool` | `False` | Normalize marker positions (output targets) |
| `normalize_inputs` | `bool` | `False` | Normalize images and EE pose (inputs) |
| `use_diff_images` | `bool` | `False` | Subtract reference images from tactile inputs |
| `use_blank_ref` | `bool` | `False` | Use blank refs (`True`) or per-session refs (`False`) |
| `transform` | `torchvision.transforms.Compose \| None` | `None` | Optional extra image transform |
| `num_resample_points` | `int` | 0 | Spline resampling count (`0` = keep the 14 raw markers; any other value resamples non-packed data) |
| `past_window_steps` | `int` | **5** | Temporal pairing look-back (index steps) |
| `max_time_delta` | `float` | 0.1 | Max time gap for valid pair (seconds) |
| `device` | `torch.device \| str \| None` | `None` | Optional device placement (only used outside DataLoader workers) |

### Output Dictionary (per sample)

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `digit_images` | `[4, 6, H, W]` | float32 | Temporal pair: channels 0–2 = current RGB, 3–5 = past RGB |
| `ee_pose` | `[7]` | float32 | EE pose in fixed frame `[x, y, z, qx, qy, qz, qw]` |
| `marker_positions` | `[K, 3]` | float32 | Marker positions (\(K\) resampled, or the 14 raw markers when `num_resample_points=0`) |
| `path` | `str` | — | Source file path (for debugging / session identification) |

### Data Loading Pipeline (`DLOH5SampleDataset.__getitem__`)

```
① Fetch current sample (idx) from HDF5
② Fetch past sample (idx − past_window_steps) from HDF5
③ Validate temporal pair (same session, time delta ≤ threshold)
   └─ If invalid → duplicate current frame as past
④ Convert images to float32 tensors [0, 1]
⑤ Resize images to (img_size_H, img_size_W) if needed
⑥ [if normalize_inputs] Z-score normalize images + EE pose
⑦ [if use_diff_images] Subtract reference images
⑧ Concatenate temporal pair → [4, 6, H, W]
⑨ [if not is_packed] Transform EE pose + markers to fixed frame
⑩ [if not is_packed and num_resample_points != current count] Spline resample markers
⑪ Compose 7D EE pose vector: [position; quaternion]
⑫ [if normalize] Z-score normalize marker positions
⑬ Return dict
```

### Data Loading Performance

| Format | Access Pattern | Approx. Throughput | Best For |
|--------|---------------|-------------------|----------|
| Per-Sample HDF5 | Random file I/O | ~5–10 ms/sample | Debugging, small tests |
| Packed HDF5 | Sequential HDF5 read | ~2–5 ms/sample | Development, training |

> **Multi-worker note:** Packed HDF5 files are opened lazily per DataLoader worker to avoid file-handle sharing across processes. Worker-specific handles are cached via `_get_archive()`.

---

## 11. Data Quality Notes

- **Synchronization:** Hardware-synced via shared ROS 2 header timestamps; only complete frames (all required streams present) are saved.
- **Frame rate:** Consistent ~15 FPS with <1 ms jitter across all sessions.
- **Marker count:** Physical marker count is fixed at 14 per frame. Occasional marker occlusion or misdetection may affect some frames (filtered at extraction time).
- **Cable configurations:** Dataset covers multiple cable mounting angles (0°, 30°, 45°, 60°, 75°, 115°, 135° and combinations), providing geometric diversity.
- **Train/test split:** Test sessions are **held-out complete sessions** (not random frame splits), ensuring evaluation on unseen cable configurations and manipulation trajectories.

---

## 12. File Structure

### Dataset Layout (on disk, outside the repo)

```
<data_root>/
├── train/                                 # Training split
│   └── <session>/                         # Per-session directories
│       ├── sample_0.h5                    # Individual samples
│       ├── sample_1.h5
│       └── ...
├── test/                                  # Test split
│   └── <session>/
├── digit_refs/                            # Per-session reference images
│   └── <session>/digit_refs.h5
└── digit_blank_closed_refs/               # Blank reference images
    ├── D21118.png
    ├── D21122.png
    ├── D21123.png
    └── D21124.png

<packed_root>/                             # Optional packed HDF5 files
└── pts32/
    ├── train.h5
    ├── train.yaml                         # Packing config + norm stats
    ├── test.h5
    └── test.yaml
```

Point the console scripts at your own copy via `--train_data_root` / `--val_data_root` / `--data_root` / `--blank_ref_root`; nothing in the repo hard-codes a dataset location.

### Repo Modules That Touch Data

```
src/dlofeel_state/
├── data/
│   ├── dataset.py                         # DLOH5SampleDataset (temporal pair) — the loader
│   ├── augment.py                         # DropAugmentDataset (15 sensor-drop masks)
│   ├── drop_masks.py                      # Shared drop-mask definitions
│   ├── spline_utils.py                    # Point sorting + B-spline fitting / resampling
│   └── transforms.py                      # Coordinate transforms, split checks, stats computation
└── configs/
    ├── default.yaml                       # Model + training config
    └── norm_stats_k16.yaml                # Training-set normalization statistics (K = 16)

scripts/
├── process_bags.py                        # ROS 2 bag extraction -> per-sample HDF5
├── pack_dataset.py                        # Per-sample HDF5 -> packed HDF5 + norm stats
└── download_data.py                       # Fetch dataset / weights from the Hugging Face Hub

tests/fixtures/parity/                     # Committed 8-frame fixture (samples + blank refs + stats)
```
