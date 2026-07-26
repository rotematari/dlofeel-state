# Tactile Noise Ablation (test-time corruption robustness)

> **Status: IMPLEMENTED** (harness ready; full 11-level run is Rotem-side, needs FAST_DATA + a C1 `best.pt`).
> Cable: **C1** (1.5 mm braided steel). Model: the **pretrained C1 `DLOStateModel`** (Sparsh ViT), evaluated as-is.
> Parent: [`../`](../). Implemented siblings: `session_scaling/`, `model_comparison/`, `tactile_sensor_ablation/`, `Diff_B_spline_ablation/`.

## Question

How robust is the pretrained C1 model to corruption of the tactile images, and how fast does accuracy fall off as the tactile signal is replaced by noise? No retraining.

## Hypothesis

Error rises monotonically as the signal is replaced by noise. The interesting quantities are the slope and where the tactile model crosses the pose-only baseline, i.e. where tactile input stops helping.

## Design

- **Test-only on the pretrained model.** Load the C1 best checkpoint and evaluate it with the tactile input corrupted at each level. No training.
- **Noise model: clean↔white-noise convex blend.** For level `p`, the tactile diff image `x` (model input, post-diff and normalized) is corrupted as

  ```
  out = (1 - alpha) * x + alpha * eps,   alpha = p / 100
  ```

  where `eps` is **per-channel-matched white noise**: `eps[:, :, c] ~ N(mu_c, sigma_c^2)` i.i.d. per element, drawn over the **6 channels** (dim 2 of `[B, 4, 6, H, W]` = 3 current-diff RGB + 3 previous-diff RGB). The per-channel moments `(mu_c, sigma_c)` are the empirical mean/std of the real diffed-normalized `digit_images` over the test set, pooled over batch, sensor (4), and spatial axes (one pair per channel; NOT per-(sensor,channel)=24, NOT a global scalar).
- **alpha endpoints.** `alpha = 0` is an **exact clean passthrough** (RNG untouched). `alpha = 1` replaces the signal entirely: `out = eps`, signal-matched white noise — a true "no-tactile-information" regime.
- **SNR axis.** Because `eps` is drawn at the empirical signal std `sigma_c`, the designed per-channel signal/noise power ratio is channel-independent and a pure function of `alpha`:

  ```
  snr = ((1 - alpha) / alpha)^2,   snr_db = 20 * log10((1 - alpha) / alpha)
  ```

  `alpha = 0` → `snr = inf` (clean); `alpha = 0.5` → `snr = 1` (0 dB); `alpha = 1` → `snr = 0` (`snr_db = -inf`). The per-channel `sigma_c` cancels, so SNR is reported per level from `alpha` alone.
- **11 levels:** `p` in 0%, 10%, ..., 100% (`alpha` 0.0 ... 1.0); `p = 0` is the clean reference.
- **Report.** L2 (`d_eval`) and Chamfer mean ± std vs `alpha`. The pose-only reference comes from the separately-trained pose-only model (see `model_comparison`).

### What the endpoints are NOT

- **`alpha = 1` is NOT the old additive `p = 100`.** The previous design added `N(0, sigma^2)` on top of the signal (`out = x + noise`, SNR ≈ 1 at the right end). The blend's `alpha = 1` is `out = eps` — the signal is *gone*, not merely buried under equal-power noise. The two right-hand floors are different regimes and must not be compared directly.
- **`alpha = 1` is NOT the pose-only floor.** Pose-only (the model with no tactile branch) is a *different checkpoint* owned by `model_comparison`. `alpha = 1` here still feeds the tactile branch — it just feeds it signal-matched white noise carrying no marker information. The two can differ (the encoder's response to white noise is not the same as ablating the branch).

## Implementation

`eval_tactile_noise.py` (single harness; pure layer + lazy eval/output):

- **Pure layer** (torch-only, no data/GPU/Sparsh/yaml/matplotlib — keeps `--help`/`--selftest` light):
  - `noise_levels() -> [0, 10, ..., 100]`.
  - `apply_noise_blend(imgs, alpha, mean_c, std_c, generator=None)` — the convex blend; `alpha = 0` clones before any `torch.randn` (RNG untouched), `alpha = 1` returns `eps` exactly, broadcasts `mean_c/std_c` `[6]` → `[1, 1, 6, 1, 1]` over dim 2, never mutates the input.
  - `effective_snr(alpha) -> (snr, snr_db)` — the closed form above.
- **Eval section** (`compute_channel_stats` needs the loader, so it lives here like the old stats pass):
  - `compute_channel_stats(loader, device, max_batches=0) -> (mean_c, std_c)` — one float64 streaming pass reducing over `dim=(0, 1, 3, 4)` of `[B, 4, 6, H, W]`; each output is a float32 `[6]` tensor. Distinct from `norm_stats_k16.yaml:digit_image_stats` (the input-normalization mean/std); this measures the *already-normalized, already-diffed* images.
  - `evaluate_noise(...)` — loads model + checkpoint once, computes `(mean_c, sigma_c)` once, sweeps the 11 levels (re-seeding the generator per level so levels differ only by `alpha`), aggregates per-sample L2 + Chamfer, prints the table and writes the JSON.
- **CLI:** `--ckpt` (omit only with `--selftest`), `--data_root`, `--blank_ref_root`, `--out_dir`, `--max_batches`, `--stats_batches` (caps the per-channel-stats pass), `--seed`, `--plot/--no-plot`, `--selftest`.
- **`--selftest`** runs 7 offline assertions on synthetic CPU tensors (alpha=0 passthrough + RNG untouched; alpha=1 signal-independent per-channel moments; exact convex blend at 0<alpha<1; per-channel broadcast onto dim 2; determinism; `compute_channel_stats` pooling over sensors; `effective_snr` values) and exits 0. No data/GPU needed.

## How to run (from the repo root with uv)

```bash
# offline self-test (no data/GPU/ckpt)
uv run python experiments/fig6_noise/eval_tactile_noise.py --selftest

# full 11-level sweep (needs FAST_DATA + a C1 best.pt)
uv run python experiments/fig6_noise/eval_tactile_noise.py \
  --data_root      src/datasets/c1/ds/test \
  --blank_ref_root src/datasets/ds/digit_blank_closed_refs \
  --ckpt           <path-to-C1-best-checkpoint>/best.pt \
  --out_dir        results/tactile_noise_ablation
# the C1 best checkpoint is provided by Rotem (not git-tracked).
```

The sweep loads the checkpoint once and evaluates all 11 levels internally (no per-level shell loop).

## Outputs

`results/tactile_noise_ablation/<ckpt-parent-name>/`:

- `noise_table.json` — 11 rows, each carrying `p`, `alpha`, `snr`, `snr_db`, `val_l2_mean/std`, `val_chamfer_mean/std`. Non-finite `snr`/`snr_db` (the `alpha = 0` and `alpha = 1` endpoints) are serialized as JSON `null`.
- `noise_curve.png` — L2 + Chamfer vs `alpha` ∈ [0, 1] (x-axis = "noise fraction α"), dual-axis.
- The printed table (alpha, SNR(dB), d_eval ± std, Chamfer; best L2 bold; "∞"/"-∞" for the non-finite SNR endpoints).

## Future investigation

Also *train* with noise (noise augmentation, or a fresh model trained per level) and check whether it flattens the degradation curve relative to the pretrained model evaluated above.

## Open questions

- [x] Noise = clean↔white-noise convex blend `(1-alpha)*x + alpha*eps` on the normalized diff image (decided 2026-06-24, supersedes additive Gaussian).
- [x] `eps` = per-channel-matched white noise over the 6 channels, moments pooled over sensor + spatial (decided).
- [x] `alpha = 1` endpoint = signal-matched white noise (`out = eps`), distinct from the old additive `p = 100` and from the pose-only floor (decided).
- [x] SNR reported per level via the closed form `((1-alpha)/alpha)^2` (decided).
- [x] Test-only on the pretrained model (decided).
- [ ] Where does the L2 curve cross the pose-only baseline (`model_comparison`)? — needs the full run.
