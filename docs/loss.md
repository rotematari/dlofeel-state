# Loss Functions Documentation

This document describes every loss function and validation metric used in DLOFeel-State, how they are composed during training, and the config knobs that control them.

---

## 1. Composite Training Loss

During training (`src/dlofeel_state/engine/train.py`), the total loss is a weighted sum of two top-level terms:

\[
\mathcal{L}_\text{total} = w_\text{chamfer} \cdot \mathcal{L}_\text{Chamfer}(\hat{\mathbf{P}}, \mathbf{P}) + w_\text{mse} \cdot \mathcal{L}_\text{Weighted}(\hat{\mathbf{P}}, \mathbf{P})
\]

where \(\hat{\mathbf{P}} \in \mathbb{R}^{B \times K \times 3}\) are predicted marker positions and \(\mathbf{P}\) are ground-truth targets.

| Config Key | Default (YAML) | Description |
|------------|----------------|-------------|
| `chamfer_weight` | **0.25** | Scalar weight on Chamfer distance |
| `mse_weight` | **1.0** | Scalar weight on `WeightedLoss` |

With the production defaults, both terms are active. The Chamfer term provides permutation-invariant matching as a regulariser, while the physics-informed `WeightedLoss` is the dominant training signal.

`dlofeel-train --use_weighted_loss false` runs the physics-free ablation: it zeros `w_edges`, `w_lengths`, and `w_kb`, leaving `WeightedLoss` as pure node-position MSE. `chamfer_weight`, `mse_weight`, and `w_nodes` are untouched, so the composite structure is unchanged.

---

## 2. Individual Loss Functions

### 2.1 Mean Per-Point Distance (`src/dlofeel_state/losses/chamfer.py`)

**Function:** `mean_point_distance_per_sample(pred, target)`

Computes the mean L2 (Euclidean) distance between corresponding predicted and ground-truth points for each sample in the batch. Used as the **primary validation metric** and for best-model selection.

**Mathematical formulation.** For a single sample with \(K\) points:

\[
d(\hat{\mathbf{P}}, \mathbf{P}) = \frac{1}{K} \sum_{i=1}^{K} \|\hat{\mathbf{p}}_i - \mathbf{p}_i\|_2
\]

| Argument | Shape | Description |
|----------|-------|-------------|
| `pred` | `[B, K, 3]` | Predicted 3D points |
| `target` | `[B, K, 3]` | Ground-truth 3D points |
| **Returns** | `[B]` | Mean L2 distance per sample |

**Implementation:**

```python
l2_per_point = torch.norm(pred - target, dim=-1)  # [B, K]
return l2_per_point.mean(dim=-1)                    # [B]
```

**Where used:** Validation loop — reported as `val/loss` in W&B and used for best-model checkpoint selection.

---

### 2.2 Chamfer Distance (`src/dlofeel_state/losses/chamfer.py`)

**Function:** `chamfer_loss(pred, target, bidirectional=True)`

Symmetric nearest-neighbour distance between two point sets. **Permutation-invariant** — does not assume a fixed point ordering.

**Mathematical formulation.**

\[
d_\text{fwd} = \frac{1}{N} \sum_{i=1}^{N} \min_{j} \|\hat{\mathbf{p}}_i - \mathbf{p}_j\|_2, \qquad
d_\text{bwd} = \frac{1}{M} \sum_{j=1}^{M} \min_{i} \|\mathbf{p}_j - \hat{\mathbf{p}}_i\|_2
\]

\[
\mathcal{L}_\text{Chamfer} = \frac{1}{B} \sum_{b=1}^{B} \frac{d_\text{fwd}^{(b)} + d_\text{bwd}^{(b)}}{2}
\]

| Argument | Shape | Description |
|----------|-------|-------------|
| `pred` | `[B, N, 3]` | Predicted point cloud |
| `target` | `[B, M, 3]` | Ground-truth point cloud (typically \(N = M\)) |
| `bidirectional` | `bool` | If `True`, average both directions |
| **Returns** | scalar | Mean Chamfer distance across batch |

**Implementation detail:** Uses `torch.cdist(pred, target, p=2)` for efficient pairwise distance computation.

**Per-sample variant:** `chamfer_distance_per_sample()` returns shape `[B]` instead of a scalar.

**Where used:**
- Training loss (weighted by `chamfer_weight`)
- Validation per-session metrics (`val/session/<name>/chamfer`), computed inline in `src/dlofeel_state/engine/train.py`

---

### 2.3 WeightedLoss — Physics-Informed Loss (`src/dlofeel_state/losses/weighted.py`)

**Class:** `WeightedLoss(w_nodes, w_edges, w_lengths, w_kb, eps=1e-8)`

A composite MSE loss that penalises errors in node positions **and** geometric/physical properties derived from the Discrete Elastic Rods (DER) framework:

\[
\mathcal{L}_\text{Weighted} = w_n \cdot \text{MSE}(\hat{\mathbf{P}}, \mathbf{P}) + w_e \cdot \text{MSE}(\hat{\mathbf{d}}, \mathbf{d}) + w_l \cdot \text{MSE}(\hat{\ell}, \ell) + w_\kappa \cdot \text{MSE}(\widehat{\kappa\mathbf{b}}, \kappa\mathbf{b})
\]

where the derived quantities are computed identically for predicted and target point sets:

| Term | Symbol | Derived From | What It Enforces |
|------|--------|-------------|-----------------|
| **Nodes MSE** | \(\mathbf{P}\) | Raw point positions `[B, K, 3]` | Positional accuracy |
| **Edges MSE** | \(\mathbf{d}\) | Unit edge directions `[B, K-1, 3]` | Local tangent orientation |
| **Lengths MSE** | \(\ell\) | Edge lengths `[B, K-1]` | Length preservation between adjacent points |
| **KB MSE** | \(\kappa\mathbf{b}\) | Curvature binormal `[B, K-1, 3]` | Physically plausible curvature |

#### Config Weights

| Config Key | Code Default | YAML Default | Description |
|------------|-------------|--------------|-------------|
| `w_nodes` | 1.0 | **1.0** | Weight on node-position MSE |
| `w_edges` | 1.0 | **0.1** | Weight on edge-direction MSE |
| `w_lengths` | 1.0 | **0.5** | Weight on edge-length MSE |
| `w_kb` | 0.01 | **0.01** | Weight on curvature-binormal MSE |

> All four sub-terms are **active** in the production configuration.

#### Internal Pipeline

`WeightedLoss.forward(preds, tgts)` performs these steps:

1. **Edges** — `compute_edges(P)`: finite differences between consecutive points → `[B, K-1, 3]`
2. **Edge lengths** — `compute_edge_lengths(e)`: L2 norm of each edge → `[B, K-1]`
3. **Unit directions** — edges divided by their lengths (clamped to `eps` for numerical safety)
4. **Curvature binormal** — `compute_curvature_binormal(e, ℓ)`: discrete bending measure → `[B, K-1, 3]`
5. Four `nn.MSELoss()` terms are computed and returned as a weighted sum

#### Alternative: Gaussian-Weighted Node Distance

The class also contains an unused method `normal_weight_distance()` that applies a Gaussian weighting profile over point indices (\(\mu=0, \sigma=0.5\) over \([-1, 1]\)), emphasising cable midpoints over endpoints. This is **not** active in the current configuration (the standard `MSELoss` is used for the nodes term).

---

## 3. DER Geometric Primitives (`src/dlofeel_state/losses/der.py`)

The helper functions that `WeightedLoss` relies on. All operate in batch and are fully differentiable.

### 3.1 Compute Edges

\[
\mathbf{e}_i = \mathbf{p}_{i+1} - \mathbf{p}_i, \quad i = 0, \ldots, K-2
\]

| Input | `[B, K, 3]` — ordered 3D vertices |
|-------|-----------------------------------|
| Output | `[B, K-1, 3]` — edge vectors |

### 3.2 Compute Edge Lengths

\[
\ell_i = \|\mathbf{e}_i\|_2
\]

| Input | `[B, K-1, 3]` — edge vectors |
|-------|------------------------------|
| Output | `[B, K-1]` — scalar lengths |

### 3.3 Compute Curvature Binormal

Discrete curvature at interior vertices, from the DER formulation (Bergou et al., 2008, Eq. 1):

\[
(\kappa\mathbf{b})_i = \frac{2\, \mathbf{e}_{i-1} \times \mathbf{e}_i}{\ell_{i-1}\, \ell_i + \mathbf{e}_{i-1} \cdot \mathbf{e}_i}, \qquad i = 1, \ldots, K-2
\]

\((\kappa\mathbf{b})_0 = \mathbf{0}\) (no predecessor edge). The result is clamped to \([-20, 20]\) component-wise for numerical stability.

| Input | `edges [B, K-1, 3]`, `lengths [B, K-1]` |
|-------|------------------------------------------|
| Output | `[B, K-1, 3]` — curvature binormal vectors |

### 3.4 Full DER Pipeline (`pipeline_points_to_material_frame`)

Complete pipeline from ordered 3D points to the DER material frame. Not used directly in the training loss, but available for bending-energy computation and analysis:

```
points → edges → edge_lengths → curvature_binormal (κb)
                                        ↓
                   bishop_u0 → bishop_frame (u, v)
                                        ↓
                             twist_angle (θ = 0)
                                        ↓
                           material_frame (m₁, m₂)
```

**Returns:** `(m1, m2, theta, kb, b_u, b_v)` — all `[B, K-1, 3]` except `theta [B, K-1]`.

**Bishop frame computation.** The twist-minimised reference frame is propagated via quaternion-based parallel transport along the rod:

1. Initialise \(\mathbf{u}_0\) perpendicular to the first edge \(\mathbf{e}_0\) via double cross product.
2. For each subsequent edge, rotate \(\mathbf{u}\) using a quaternion derived from the curvature binormal:
   - Rotation angle from \(\|\kappa\mathbf{b}\|^2\): \(\sin\phi, \cos\phi = f(\|\kappa\mathbf{b}\|^2)\)
   - Quaternion: \(\mathbf{q} = [\cos\phi,\; \sin\phi \cdot \widehat{\kappa\mathbf{b}}]\)
3. If rotation is negligible (\(1 - \cos\phi < 10^{-6}\)), reuse the previous frame.

**Material frame.** Obtained by rotating the Bishop frame by twist angle \(\theta\):

\[
\mathbf{m}_1 = \cos\theta\, \mathbf{u} + \sin\theta\, \mathbf{v}, \qquad
\mathbf{m}_2 = -\sin\theta\, \mathbf{u} + \cos\theta\, \mathbf{v}
\]

> The current implementation sets \(\theta = 0\) (twist-free rod): `optimize_twist_angle()` returns zeros rather than solving for twist. A nonlinear twist solve is deliberately out of scope for this release (it would add a heavyweight optimizer dependency for a term the loss never uses); see [THIRD_PARTY.md](THIRD_PARTY.md).

---

## 4. Bending Energy (`src/dlofeel_state/losses/bending.py`)

Full bending-energy computation based on the Discrete Elastic Rods model. **Not currently used in the default training loss**, but available as a standalone module for analysis and potential future regularisation.

### `compute_bending_energy(points_3d, m1, m2, kb, edge_lengths, ...)`

\[
E_\text{bend} = \frac{1}{2} \sum_{i=1}^{K-2} \frac{k_b^{(i-1)} \|\boldsymbol{\omega}_i - \bar{\boldsymbol{\omega}}_i^{\text{prev}}\|^2 + k_b^{(i)} \|\boldsymbol{\omega}_i - \bar{\boldsymbol{\omega}}_i^{\text{next}}\|^2}{\bar{l}_i}
\]

where:
- \(\boldsymbol{\omega}_i = [\kappa\mathbf{b}_i \cdot \mathbf{m}_2^i,\; -\kappa\mathbf{b}_i \cdot \mathbf{m}_1^i]^\top\) is the material curvature (2D)
- \(\bar{\boldsymbol{\omega}}_i\) is the rest-state curvature (default: \(\mathbf{0}\) for a straight rod)
- \(\bar{l}_i = \ell_{i-1} + \ell_i\) is the Voronoi region length at vertex \(i\)
- \(k_b\) is the bending stiffness (default: \(5 \times 10^{-5}\))

| Argument | Shape | Description |
|----------|-------|-------------|
| `points_3d` | `[B, K, 3]` | Current vertex positions |
| `m1, m2` | `[B, K-1, 3]` | Material frame vectors |
| `kb` | `[B, K-1, 3]` | Curvature binormal |
| `edge_lengths` | `[B, K-1]` | Length of each edge |
| `rest_curvature_prev/next` | `[B, K-1, 2]` | Undeformed curvature (default: zero) |
| `bend_stiffness` | scalar or `[B, K-1]` | Material stiffness |
| **Returns** | `[B]` | Total bending energy per sample |

### Helper: `compute_voronoi_lengths(edge_lengths)`

\[
\bar{l}_0 = 0, \qquad \bar{l}_i = \ell_{i-1} + \ell_i \quad (i = 1, \ldots, K-2)
\]

| Input | `[B, K-1]` — edge lengths |
|-------|---------------------------|
| Output | `[B, K-1]` — Voronoi region lengths |

### Helper: `project_curvature_to_material_frame(kb, m1, m2)`

Projects the 3D curvature binormal onto the 2D material frame (DER Eq. 2):

\[
\omega_1 = \kappa\mathbf{b} \cdot \mathbf{m}_2, \qquad \omega_2 = -\kappa\mathbf{b} \cdot \mathbf{m}_1
\]

| Input | `kb, m1, m2` — all `[B, K-1, 3]` |
|-------|-----------------------------------|
| Output | `[B, K-1, 2]` — material curvature \((\omega_1, \omega_2)\) |

### Convenience: `compute_bending_energy_from_points(points_3d, rest_points, bend_stiffness)`

End-to-end pipeline: raw 3D points → full DER pipeline → bending energy + details dict containing intermediate quantities (material frame, curvature, edge lengths, twist angles).

---

## 5. Validation Metrics

All validation metrics are computed **after denormalisation** to report errors in physical units (metres).

| Metric | Function / Computation | Logged As |
|--------|----------------------|-----------|
| Val loss (model selection) | `mean_point_distance_per_sample()` → mean over batches | `val/loss` |
| Val loss std | Standard deviation of per-batch mean distances | `val/loss_std` |
| Per-session RMSE | `mean_point_distance_per_sample()` per session | `val/session/<name>/rmse` |
| Per-session Chamfer | Inline symmetric Chamfer via `torch.cdist` | `val/session/<name>/chamfer` |
| Per-sample distances | `mean_point_distance_per_sample()` → `.detach().cpu().numpy()` | Aggregated into session stats |

### Validation Loop Detail

1. For each batch, model predicts `pred [B, K, 3]` and both `pred` and `tgt` are **denormalised** using the training-set statistics.
2. `mean_point_distance_per_sample(pred_unnorm, tgt_unnorm)` → `[B]` appended to `val_running_list`.
3. Per-sample session metrics are accumulated using the `path` field to identify session membership.
4. After all batches: `val_loss = np.mean(val_running_list)`, `val_loss_std = np.std(val_running_list)`.
5. Per-session RMSE and Chamfer are aggregated and logged.

---

## 6. Loss Function Dependency Graph

All paths below are relative to `src/dlofeel_state/`.

```
engine/train.py
  │
  ├── chamfer_loss()               ← losses/chamfer.py
  │     └── torch.cdist()
  │
  └── WeightedLoss()               ← losses/weighted.py
        ├── compute_edges()        ← losses/der.py
        ├── compute_edge_lengths() ← losses/der.py
        └── compute_curvature_binormal() ← losses/der.py

Validation:
  └── mean_point_distance_per_sample() ← losses/chamfer.py
        └── torch.norm()

Standalone (not in training loop):
  └── compute_bending_energy_from_points() ← losses/bending.py
        └── pipeline_points_to_material_frame() ← losses/der.py
              ├── compute_bishop_u0()
              ├── compute_bishop_frame()
              ├── optimize_twist_angle()  (returns θ=0)
              └── compute_material_frame()
```

---

## 7. Quick Reference — Config (`src/dlofeel_state/configs/default.yaml`)

```yaml
# Top-level loss composition
chamfer_weight: 0.25   # Weight on chamfer_loss in training
mse_weight: 1.0        # Weight on WeightedLoss in training

# WeightedLoss sub-weights
w_nodes: 1.0           # Node position MSE
w_edges: 0.1           # Edge direction MSE
w_lengths: 0.5         # Edge length MSE
w_kb: 0.01             # Curvature binormal MSE
```

### Effective Training Objective (Expanded)

Substituting the production defaults:

\[
\mathcal{L} = 0.25 \cdot \mathcal{L}_\text{Chamfer} + 1.0 \cdot \left(1.0 \cdot \text{MSE}_\text{nodes} + 0.1 \cdot \text{MSE}_\text{dirs} + 0.5 \cdot \text{MSE}_\text{lengths} + 0.01 \cdot \text{MSE}_\text{kb}\right)
\]

---

## 8. File Summary

```
src/dlofeel_state/losses/
├── chamfer.py     # mean_point_distance_per_sample, chamfer_loss,
│                  #   chamfer_distance_per_sample
├── weighted.py    # WeightedLoss (composite physics-informed loss) + compute_weighted_loss
├── der.py         # DER primitives: edges, lengths, curvature binormal,
│                  #   Bishop frame, material frame, full pipeline
└── bending.py     # Bending energy computation (standalone, not in training)

src/dlofeel_state/engine/
├── train.py       # Composes the training loss; owns the validation loop
└── metrics.py     # Per-session metric reduction (extract_session, reduce_metrics)
```

### Numeric Parity

`tests/test_losses_parity.py` pins every function above against the original
implementation on fixed inputs, so a refactor that changes a number fails loudly:

```bash
.venv/bin/python -m pytest tests/test_losses_parity.py -v
```
