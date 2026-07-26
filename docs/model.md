# Model Architecture Documentation

This document describes the neural network architectures in DLOFeel-State, centred on the primary model `DLOStateModel` (`src/dlofeel_state/models/dlo_state_model.py`) and its supporting modules.

---

## 1. High-Level Overview

`DLOStateModel` is a multi-modal encoder–decoder that predicts 3D marker positions along a deformable linear object (cable) from DIGIT tactile images and the robot end-effector (EE) pose.

```
DIGIT images [B, 4, 6, H, W]  ──►  Image Backbone  ──►  [B, N_tokens, D]
                                                                │
                                                        + Digit Embedding
                                                                │
                                                         Cross-Attention
                                                     (Q=markers, K/V=images)
                                                                │
EE pose [B, 7]  ──►  Pose Backbone  ──►  [B, N_out·D]         │
                            │                                   │
                     reshape + pos enc  ──►  Marker Queries ────┘
                                                    │
                                              Self-Attention
                                                    │
                                              MLP  ──►  [B, N_out, 3]
                                                    │
                                         (optional) DifferentiableBSpline
                                                    │
                                              [B, K, 3]  marker positions
```

### Two Decoder Modes

| Mode | Flag | Description |
|------|------|-------------|
| **Attention** | `with_att=True` | Cross-attention (images → markers) + self-attention, per-marker 3D prediction |
| **MLP** | `with_att=False` | Concatenate flattened image features + pose, single MLP to all points |

---

## 2. DLOStateModel

**File:** `src/dlofeel_state/models/dlo_state_model.py`

The top-level module that wires encoders, fusion, and decoder. Its collaborators live in sibling modules: `models/backbones/sparsh_backbone.py` (`SparshImgBackbone`), `models/backbones/resnet_backbone.py` (`ImgBackbone`), `models/backbones/pose_encoder.py` (`PoseBackbone`), and `models/bspline.py` (`DifferentiableBSpline`).

### Config Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_output_pts` | int | **required** | Number of output marker points \(K\) |
| `model_params.latent_dim` | int | 512 | Feature dimension \(D\) throughout the model |
| `model_params.use_spline` | bool | False | Predict control points → sample via B-spline |
| `n_control_points` | int | \(K/2\) | Number of B-spline control points |
| `spline_degree` | int | 3 | B-spline degree |
| `with_att` | bool | **required** | Use cross-attention decoder |
| `only_pose` | bool | **required** | Ablation: no tactile input |
| `dropout` | float | **required** | Dropout rate for decoder MLPs |
| `model_params.image_backbone.type` | str | `"resnet18"` | `"sparsh_vit"` or `"resnet18"` |
| `model_params.use_sinusoidal_pos_encoding` | bool | True | Fixed sinusoidal vs. learned positional encoding |

> **Production defaults** (from `src/dlofeel_state/configs/default.yaml`): `latent_dim=128`, `use_spline=True`, `n_control_points=32`, `num_output_pts=32`, `with_att=True`, `only_pose=False`, `dropout=0.1`, `image_backbone.type="sparsh_vit"`.

### Forward Pass

```python
def forward(self, pos: Tensor, imgs: Tensor | None = None) -> Tensor:
    # pos:  [B, 7]            (x, y, z, qx, qy, qz, qw)
    # imgs: [B, 4, 6, H, W]  (4 DIGIT cameras, 6-ch temporal pair)
    # returns: [B, K, 3]      marker positions
```

**Step-by-step (attention mode, with images):**

1. **Pose encoding** — `PoseBackbone(pos)` → `[B, D · N_out]`
2. **Image encoding** — `SparshImgBackbone(imgs)` → `[B, N_digits, N_patches, D]` → flatten to `[B, N_tokens, D]`
3. **Digit embeddings** — add learned per-camera embedding `[1, 4, 1, D]` to image features before flattening
4. **Marker queries** — reshape pose embed to `[B, N_out, D]`, add positional encoding
5. **Cross-attention** — `MultiheadAttention(Q=markers, K=V=images)` + residual + LayerNorm
6. **Self-attention** — `MultiheadAttention(Q=K=V=markers)` + residual + LayerNorm
7. **Decoder MLP** — per-token `Linear → LN → ReLU → Dropout → Linear → LN → ReLU → Dropout → Linear(3)` → `[B, N_out, 3]`
8. **Spline** (optional) — `DifferentiableBSpline(control_points)` → `[B, K, 3]`

---

## 3. Image Backbones

### 3.1 SparshImgBackbone (Sparsh ViT)

**File:** `src/dlofeel_state/models/backbones/sparsh_backbone.py`

Pretrained DINO-style Vision Transformer for tactile images, with support for partial fine-tuning and multi-scale feature fusion. The Sparsh ViT itself is not vendored: it is reached through the single import choke point `models/backbones/_sparsh_import.py`, which resolves the `third_party/sparsh` submodule (see [THIRD_PARTY.md](THIRD_PARTY.md) for the licensing terms and the pinned commit).

**Input:** `[B, 4, 6, H, W]` — 4 DIGIT cameras, 6-channel temporal pair (current + past frame concatenated on channel dim).

**Output:** `[B, N_digits · N_patches, D]` (single encoder) or `[B, N_digits, N_patches, D]` (ensemble)

where \(N_\text{patches} = \lfloor H/16 \rfloor \cdot \lfloor W/16 \rfloor\) (patch size 16). For the default 320 × 240 input: \(N_\text{patches} = 20 \times 15 = 300\).

#### Config Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `image_backbone.model_type` | str | `"vit_small"` | `"vit_tiny"` (D=192), `"vit_small"` (D=384), or `"vit_base"` (D=768) |
| `image_backbone.single_encoder` | bool | True | Share one ViT across all 4 cameras |
| `image_backbone.pretrained_weights_path` | str | — | Path to DINO checkpoint |
| `image_backbone.pos_embed_fn` | str | `"auto"` | `"auto"`, `"learned"`, or `"sinusoidal"` |
| `image_backbone.unfreeze_last_n_blocks` | int | 0 | ViT blocks to unfreeze from the end (0 = fully frozen) |
| `image_backbone.backbone_lr_factor` | float | 0.05 | LR multiplier for unfrozen backbone params (differential LR) |
| `image_backbone.multi_scale_fusion` | bool | False | Enable multi-scale feature fusion |
| `image_backbone.fusion_layers` | list[int] | `[6, 12]` | 1-indexed ViT layers to extract features from |
| `image_backbone.fusion_method` | str | `"learned_sum"` | `"learned_sum"` (attentive probing) or `"concat"` (DINOv2-style) |

> **Production defaults** (from `src/dlofeel_state/configs/default.yaml`): `model_type="vit_small"`, `single_encoder=True`, `pos_embed_fn="learned"`, `unfreeze_last_n_blocks=2`, `backbone_lr_factor=0.5`, `multi_scale_fusion=True`, `fusion_layers=[9, 10, 11, 12]`, `fusion_method="learned_sum"`.

#### Freezing Strategy (`_apply_freezing_strategy`)

1. **All parameters frozen** by default.
2. If `unfreeze_last_n_blocks > 0`:
   - Last \(N\) transformer blocks are unfrozen.
   - Final `LayerNorm` (`backbone.norm`) is always unfrozen.
   - These parameters receive `lr × backbone_lr_factor` via differential LR in the optimizer.

#### Multi-Scale Fusion

When `multi_scale_fusion=True`, intermediate features are extracted from specified ViT layers via `backbone.get_intermediate_layers()`. Two fusion strategies are supported:

| Method | Mechanism | Trainable Parameters |
|--------|-----------|---------------------|
| `learned_sum` | Softmax-weighted sum of layer outputs: \(\mathbf{f} = \sum_l \text{softmax}(\mathbf{w})_l \cdot \mathbf{f}_l\) | 1 scalar weight per layer |
| `concat` | Concatenate layer outputs on feature dim → linear projection: \(\mathbf{f} = \mathbf{W}_\text{proj} [\mathbf{f}_1 \| \cdots \| \mathbf{f}_L] + \mathbf{b}\) | `Linear(D × L, D)` |

The `get_layer_weights()` method returns the current softmax-normalised layer weights (for `learned_sum` only).

**Layer index convention:** Config uses **1-indexed** values (e.g., `[9, 10, 11, 12]`), converted to 0-indexed internally. Layer 12 (the final block) is always included; if `fusion_layers=[12]`, fusion is disabled and standard encoding is used.

#### Weight Loading (`_load_weights`)

1. Load checkpoint and extract keys with prefix `student_encoder.backbone.`.
2. Infer positional embedding type from checkpoint keys (`"sinusoidal"` if `pos_embed.frequency_bands` exists, else `"learned"`).
3. Validate compatibility with the model's `pos_embed_fn` setting. Raise error on mismatch (unless `pos_embed_fn="auto"`).
4. Load state dict with `strict=False`. Warn if `patch_embed.proj.weight` is missing (6-channel projection initialised randomly).

---

### 3.2 ImgBackbone (ResNet18)

**File:** `src/dlofeel_state/models/backbones/resnet_backbone.py`

Trainable ResNet18 truncated at `layer2`. Used as a lighter alternative to Sparsh ViT (select it with `--image_backbone_type resnet18`).

**Input:** `[B, 4, 6, H, W]`

**Output:** `[B, N_digits, N_tokens, D]` where \(N_\text{tokens} = (H/8) \times (W/8)\)

| Detail | Value |
|--------|-------|
| Feature channels (before projection) | 128 |
| Spatial downsampling | ×8 (conv1 + maxpool + layer1 + layer2) |
| First conv modified to | 6 input channels (temporal pair) |
| Projection | `Linear(128, latent_dim)` |

Supports both single-encoder (shared across cameras) and ensemble (per-digit) modes.

---

## 4. PoseBackbone

**File:** `src/dlofeel_state/models/backbones/pose_encoder.py`

A 4-layer MLP that encodes the 7D end-effector pose.

```
Linear(7, hidden_dim) → LayerNorm → ReLU → Dropout
Linear(hidden_dim, hidden_dim) → LayerNorm → ReLU → Dropout
Linear(hidden_dim, hidden_dim) → LayerNorm → ReLU → Dropout
Linear(hidden_dim, D · N_out)
```

**Input:** `[B, 7]` — `[x, y, z, qx, qy, qz, qw]`

**Output:** `[B, D · N_out]` — reshaped to `[B, N_out, D]` in the attention decoder

| Config Key | Type | Default | Description |
|------------|------|---------|-------------|
| `model_params.pose_backbone.hidden_dim` | int | 256 | MLP hidden layer width |
| `dropout` | float | — | Dropout rate |

> **Production default:** `hidden_dim=512`.

The output dimension depends on whether splines are used:
- With spline: `D × n_control_points`
- Without spline: `D × num_output_pts`

---

## 5. Attention Decoder (`with_att=True`)

The attention decoder treats each output point as a learnable query that attends to image features.

### 5.1 With Images (`only_pose=False`)

```
Pose embed [B, N_out·D]  ──reshape──►  Marker queries [B, N_out, D]
                                              │
                                     + Positional encoding
                                              │
                                       Cross-Attention
                                 (Q=markers, K/V=image tokens)
                                              │
                                     + Residual + LayerNorm
                                              │
                                        Self-Attention
                                  (Q=K=V=marker features)
                                              │
                                     + Residual + LayerNorm
                                              │
                                    Decoder MLP per token → [B, N_out, 3]
```

**Components:**

| Component | Details |
|-----------|---------|
| `cross_attn` | `nn.MultiheadAttention(D, num_heads, dropout, batch_first=True)` |
| `cross_attn_norm` | `LayerNorm(D)` |
| `self_attn` | `nn.MultiheadAttention(D, num_heads, dropout, batch_first=True)` |
| `self_attn_norm` | `LayerNorm(D)` |
| `digit_embedding` | `Parameter([1, 4, 1, D])` — per-camera learned embedding |
| `marker_pos_encoding` | Sinusoidal (buffer) or learned (parameter), shape `[1, N_out, D]` |

**Number of attention heads:**

\[
n_\text{heads} = \max\!\left(8,\; \left\lfloor D / 32 \right\rfloor\right)
\]

This keeps the per-head dimension at 32. Examples: \(D=384 \Rightarrow 12\) heads; \(D=512 \Rightarrow 16\); \(D=128 \Rightarrow 8\).

### 5.2 Pose-Only (`only_pose=True`)

Replaces cross-attention with a 2-layer `TransformerEncoder` (self-attention only, `nhead=8`, `dim_feedforward=4D`):

```
Pose embed ──reshape──► [B, N_out, D] + Pos encoding ──► TransformerEncoder(2 layers) ──► MLP ──► [B, N_out, 3]
```

### 5.3 Positional Encoding

| Config Value | Type | Description |
|--------------|------|-------------|
| `use_sinusoidal_pos_encoding=True` | Buffer (non-trainable) | Standard transformer sinusoidal encoding (see §8) |
| `use_sinusoidal_pos_encoding=False` | `nn.Parameter` | Learned embedding, initialised from \(\mathcal{N}(0,\, 0.02)\) |

---

## 6. MLP Decoder (`with_att=False`)

A 3-layer MLP that maps concatenated features to all marker positions jointly.

```
[img_feat.flatten(), pose_embed]  ──►  Linear(input, input//2) → LN → ReLU → Dropout
                                        Linear(input//2, input//4) → LN → ReLU → Dropout
                                        Linear(input//4, 3 · N_out)
                                        ──reshape──►  [B, N_out, 3]
```

**Input dimension:**
- With images: \(N_\text{patches} \times D_\text{embed} + D_\text{latent}\) (e.g., \(300 \times 384 + 128 = 115\,328\))
- Pose only: \(D_\text{latent}\)

---

## 7. DifferentiableBSpline

**File:** `src/dlofeel_state/models/bspline.py`

Optional post-processing layer that evaluates a smooth B-spline curve through predicted control points.

```python
DifferentiableBSpline(n_control_points, n_output_points, degree=3)
```

**Input:** `[B, n_control_points, 3]` — predicted control-point positions

**Output:** `[B, n_output_points, 3]` — uniformly sampled points on the B-spline curve

| Detail | Value |
|--------|-------|
| Parameter domain | \(t \in [-1, 1]\) (uniform linspace) |
| Backend | `torchcurves.functional.bspline_curves` |
| AMP safety | Forces float32 to avoid `scatter_add_` dtype mismatch |
| Knots | Uniform by default (optional custom knots) |

**Mathematical formulation.** Given \(n\) control points \(\{\mathbf{c}_j\}_{j=0}^{n-1}\) and degree \(k\), the B-spline curve is:

\[
\mathbf{C}(t) = \sum_{j=0}^{n-1} N_{j,k}(t)\, \mathbf{c}_j
\]

where \(N_{j,k}(t)\) are the B-spline basis functions defined recursively over a uniform knot vector. The output samples \(\mathbf{C}(t_i)\) at \(n_\text{output}\) uniformly-spaced values \(t_i \in [-1, 1]\).

When `use_spline=True`, the network predicts `n_control_points` control points, then the spline layer evaluates `n_output_points` on the resulting curve. In the default configuration, `n_control_points = num_output_pts = 32`, so the spline acts purely as a smoothness prior.

---

## 8. Sinusoidal Positional Encoding

**Function:** `get_sinusoidal_pos_encoding(n_positions, d_model) → [1, n_positions, d_model]`

Standard transformer-style fixed positional encoding:

\[
\text{PE}(p, 2i) = \sin\!\left(\frac{p}{10000^{2i / d}}\right), \qquad
\text{PE}(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i / d}}\right)
\]

where \(p \in \{0, \ldots, n_\text{positions}-1\}\) and \(i \in \{0, \ldots, d/2 - 1\}\).

Registered as a non-trainable buffer in `DLOStateModel`. Added to marker queries to encode point ordering.

---

## 9. Sparsh ViT Internals

**File:** `third_party/sparsh/tactile_ssl/model/vision_transformer.py` (submodule; not redistributed here)

The `VisionTransformer` class used by `SparshImgBackbone` follows the DINOv2 architecture:

| Parameter | ViT-Tiny | ViT-Small | ViT-Base |
|-----------|----------|-----------|----------|
| `embed_dim` | 192 | 384 | 768 |
| `depth` (blocks) | 12 | 12 | 12 |
| `num_heads` | 3 | 6 | 12 |
| `mlp_ratio` | 4.0 | 4.0 | 4.0 |
| `patch_size` | 16 | 16 | 16 |

**Patch embedding:** `PatchEmbed` — `nn.Conv2d(in_chans, embed_dim, kernel_size=16, stride=16)` producing \(N_\text{patches}\) tokens. For 6-channel input (temporal pair), `in_chans=6`.

**Token sequence:** `[CLS] + [REG_1, ..., REG_R] + [PATCH_1, ..., PATCH_N]` where \(R\) is the number of register tokens (0 by default in our checkpoints).

**Block architecture:** Each of the 12 blocks contains:
1. `LayerNorm` → Multi-Head Self-Attention → Residual
2. `LayerNorm` → MLP (`Linear(D, 4D)` → GELU → `Linear(4D, D)`) → Residual
3. Optional `LayerScale` (learned per-element gating, init = 1.0)
4. Stochastic depth (drop-path)

**`get_intermediate_layers(x, n)`** — extracts features from specified block indices. When `n` is a list of 0-indexed block indices, returns a tuple of `[B, N_patches, D]` tensors (CLS and register tokens stripped, optionally normalised). This is the entry point for multi-scale fusion.

**Positional embedding types:**
- `"learned"`: `nn.Parameter([1, 1 + N_patches, D])` — trained jointly with the model.
- `"sinusoidal"`: Fixed embedding using `SinusoidalPositionalEmbedding` (frequency-band based).

Both support interpolation to different image sizes via bicubic interpolation of the grid.

---

## 10. Model Variants

Two variants ship alongside the primary model. The original codebase also carried
sequence-input and dedicated-spline variants (`DLOStateModelV2`,
`DLOStateModelWithSpline`); neither had a training entrypoint or a reported result,
so neither was carried into this release.

### 10.1 `models/basic.py` — `DLOStateModel` (Legacy)

Earlier, simpler architecture used before the Sparsh ViT integration.

| Feature | `models/basic.py` | `models/dlo_state_model.py` |
|---------|-----------|----------------|
| Image backbone | ResNet18 only | Sparsh ViT or ResNet18 |
| Input images | `[B, 4, 3, H, W]` (single frame) | `[B, 4, 6, H, W]` (temporal pair) |
| Pose output | `[B, D]` (broadcast to markers) | `[B, D · N_out]` (per-marker) |
| Cross-attention | `nn.TransformerDecoder` (2 layers) | `nn.MultiheadAttention` (1 layer) |
| Digit embeddings | No | Yes |
| Positional encoding | Learned only | Sinusoidal or learned |
| Multi-scale fusion | No | Yes |
| Spline support | No | Yes |

### 10.2 `models/direction_model.py` — `DLODirectionModel`

Direction-input ablation that replaces the learned tactile image backbone with a classical CV pipeline. Select it with `dlofeel-train --model_type direction`.

**High-level change:**
- Input tactile images are converted into one direction token per DIGIT sensor.
- The direction is extracted with `get_line_params_from_diff()` in `src/dlofeel_state/models/direction_lines.py`.
- Only the fitted direction `[vx, vy]` is kept; the line anchor `[x0, y0]` is discarded.
- A small MLP projects each 2D direction into latent space before the same cross-attention decoder is applied.

| Feature | `DLODirectionModel` | `DLOStateModel` |
|---------|---------------------|-----------------|
| Tactile encoder | Classical CV line fit + MLP projection | Learned image backbone (`sparsh_vit` / `resnet18`) |
| Tactile tokens per sensor | 1 | Many spatial tokens / patches |
| Differentiable tactile path | Only projection MLP | Full tactile encoder + decoder |
| Recommended input handling | `normalize_inputs=False` | `normalize_inputs=True` in main runs |
| Decoder | Same cross-attention / self-attention structure | Same |

**Use case:**
- Tests whether coarse tactile orientation cues are sufficient for state estimation.
- Provides a compact, interpretable baseline that isolates the value of the learned image backbone.

See [training.md](training.md) §7.1 for the run configuration and the reported result.

---

## 11. Discrete Elastic Rod Physics

**Files:** `src/dlofeel_state/losses/der.py`, `src/dlofeel_state/losses/bending.py`

The Discrete Elastic Rod (DER) geometry that the physics-informed loss is built on. These are first-party reimplementations of the published Bergou et al. (2008) formulation, not a dependency on any rod simulator; [THIRD_PARTY.md](THIRD_PARTY.md) records the provenance and the prior-art credit. [loss.md](loss.md) §3–§4 documents the shipped functions and their tensor shapes; this section states the underlying continuum-to-discrete formulation they encode.

### Core Mathematical Framework

The DER formulation from Bergou et al. (2008, 2010):

**Curvature binormal** (discrete curvature at vertex \(i\)):

\[
(\kappa \mathbf{b})_i = \frac{2\, \mathbf{e}_{i-1} \times \mathbf{e}_i}{\|\mathbf{e}_{i-1}\| \|\mathbf{e}_i\| + \mathbf{e}_{i-1} \cdot \mathbf{e}_i}
\]

where \(\mathbf{e}_i = \mathbf{x}_{i+1} - \mathbf{x}_i\) is the edge vector.

**Material curvature:**

\[
\boldsymbol{\omega}_i = \begin{bmatrix} (\kappa \mathbf{b})_i \cdot \mathbf{m}_2^i \\ -(\kappa \mathbf{b})_i \cdot \mathbf{m}_1^i \end{bmatrix}
\]

where \(\mathbf{m}_1^i, \mathbf{m}_2^i\) are the material frame directors at vertex \(i\).

**Elastic energy:**

\[
E = \underbrace{\frac{1}{2} \sum_i \frac{k_b \|\boldsymbol{\omega}_i - \bar{\boldsymbol{\omega}}_i\|^2}{\bar{l}_i}}_{\text{bending}} + \underbrace{\frac{1}{2} \sum_i \frac{k_t (\theta_{i+1} - \theta_i)^2}{\bar{l}_i}}_{\text{twisting}}
\]

where \(k_b, k_t\) are bending and twist stiffnesses, \(\bar{l}_i\) are Voronoi region lengths, and \(\theta_i\) are twist angles.

### What Is Implemented Here

| Quantity | Where | Status |
|----------|-------|--------|
| Edges, edge lengths, curvature binormal \(\kappa\mathbf{b}\) | `losses/der.py` | Used by the training loss (`WeightedLoss`) |
| Bishop frame (quaternion parallel transport), material frame \(\mathbf{m}_1, \mathbf{m}_2\) | `losses/der.py` | Implemented; reachable via `pipeline_points_to_material_frame` |
| Material curvature \(\boldsymbol{\omega}\), bending energy \(E_\text{bend}\) | `losses/bending.py` | Implemented as a **reference only**; nothing in this release calls it ([loss.md](loss.md) §4) |
| Twist angle \(\theta\) and the twisting energy term | — | **Not implemented.** \(\theta = 0\) throughout (twist-free rod); no nonlinear twist solve |
| Rod simulation (forward dynamics, PBD constraints, gravity) | — | **Not part of this repository** |

The twisting term above is stated for completeness: because \(\theta = 0\), only the bending term is ever evaluated.

---

## 12. Complete Config Reference

All model-related config keys with their **production defaults** (from `src/dlofeel_state/configs/default.yaml`):

```yaml
num_output_pts: 32
n_control_points: 32
spline_degree: 3

with_att: true
only_pose: false
dropout: 0.1

model_params:
  latent_dim: 128
  use_spline: true
  use_sinusoidal_pos_encoding: true

  pose_backbone:
    hidden_dim: 512

  image_backbone:
    type: "sparsh_vit"
    model_type: "vit_small"
    single_encoder: true
    pretrained_weights_path: "third_party/sparsh/checkpoints/dino_vitsmall.ckpt"
    pos_embed_fn: "learned"

    # Partial fine-tuning
    unfreeze_last_n_blocks: 2
    backbone_lr_factor: 0.5

    # Multi-scale fusion
    multi_scale_fusion: true
    fusion_layers: [9, 10, 11, 12]
    fusion_method: "learned_sum"
    upto_layer: 2

img_size_H: 320
img_size_W: 240
```

The DINO checkpoint at `pretrained_weights_path` is fetched separately (`scripts/download_data.py --what weights`) and is only needed to train from scratch: evaluating a released checkpoint loads full weights and ignores it.

`num_output_pts` / `n_control_points` set \(K\) for the architecture. The released checkpoint and the committed parity fixture use \(K = 16\) (`--num_resample_points 16 --n_control_points 16`), which also requires the matching `norm_stats_k16.yaml`; the YAML default of 32 is the original packing default.

### Derived Dimensions (for default config)

| Quantity | Formula | Value |
|----------|---------|-------|
| ViT embed dim (\(D_\text{embed}\)) | ViT-Small | 384 |
| Latent dim (\(D\)) | `latent_dim` | 128 |
| \(N_\text{patches}\) | \(\lfloor 320/16 \rfloor \times \lfloor 240/16 \rfloor\) | 300 |
| \(N_\text{tokens}\) (single encoder) | \(4 \times 300\) | 1200 |
| Attention heads | \(\max(8, 128/32)\) | 8 |
| Pose output dim | \(128 \times 32\) | 4096 |
| Decoder MLP: input → hidden1 → hidden2 → out | \(D \to D/2 \to D/4 \to 3\) | 128 → 64 → 32 → 3 |

---

## 13. Tests

After modifying the model, run the model-facing tests:

```bash
.venv/bin/python -m pytest tests/test_model_loads_ckpt.py tests/test_sparsh_import.py -v
```

| Test | What it pins |
|------|--------------|
| `tests/test_sparsh_import.py` | The Sparsh submodule choke point: ViT builders import and construct, and the vendored positional-embedding patch is applied |
| `tests/test_model_loads_ckpt.py` | The released checkpoint's `state_dict` loads into a freshly constructed `DLOStateModel` with `strict=True` — every parameter name and shape matches |
| `tests/test_parity.py` | End-to-end: predictions on the frozen 8-frame fixture match the original implementation's golden snapshot within tolerance |

The last two need the checkpoint, supplied through the `DLOFEEL_PARITY_CKPT`
environment variable, and skip without it.

---

## 14. File Summary

```
src/dlofeel_state/models/
├── dlo_state_model.py         # Primary model: DLOStateModel + get_sinusoidal_pos_encoding
├── bspline.py                 # DifferentiableBSpline
├── basic.py                   # Legacy model (ResNet18, single-frame, no fusion)
├── direction_model.py         # DLODirectionModel (CV direction ablation)
├── direction_lines.py         # Classical line fit: get_line_params_from_diff
└── backbones/
    ├── sparsh_backbone.py     # SparshImgBackbone (Sparsh ViT + multi-scale fusion)
    ├── resnet_backbone.py     # ImgBackbone (ResNet18)
    ├── pose_encoder.py        # PoseBackbone (4-layer MLP)
    └── _sparsh_import.py      # Single import choke point for the Sparsh submodule

third_party/sparsh/            # Git submodule, CC-BY-NC 4.0, NOT redistributed here
└── tactile_ssl/model/
    ├── vision_transformer.py  # VisionTransformer (ViT-Tiny/Small/Base)
    └── layers/                # Attention, MLP, PatchEmbed building blocks

src/dlofeel_state/losses/      # DER geometry + physics-informed loss (see docs/loss.md)
```
