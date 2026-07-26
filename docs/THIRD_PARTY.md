# Third-Party Provenance

This project's own code is MIT-licensed. It relies on the third-party components
below. This file is the authoritative provenance record; NOTICE is the short
human-facing summary.

## Provenance table

| Component | Source | Version / pin | License | How consumed | Attribution required | Redistributed here? |
|---|---|---|---|---|---|---|
| Sparsh (tactile SSL encoder) | github.com/facebookresearch/sparsh | submodule pinned to `fee6a05a97330eca014b59c71ac41596fe3974a7` (branch `main`) | CC-BY-NC 4.0 (repo + weights); some source files carry per-file Apache-2.0 Meta headers | Git submodule (user clones separately); provides `tactile_ssl` ViT encoder + pretrained weights | Yes — attribute Meta / Sparsh; preserve CC-BY-NC 4.0; NonCommercial gate applies to Sparsh code + weights | No, except ~50 Apache-2.0 lines of `interpolate_pos_encoding` in `_sparsh_import.py` (see below) |
| Discrete Elastic Rods (DER) formulation | Bergou et al., "Discrete elastic rods," ACM TOG 27(3), 2008, doi:10.1145/1360612.1360662 | Formulation (paper) | N/A (published mathematical formulation; not code) | Independent reimplementation in `src/dlofeel_state/losses/der.py` and `bending.py` | Yes — cite Bergou et al. 2008 | Reimplemented (first-party), not copied |
| DEFORM (reference implementation) | github.com/roahmlab/DEFORM (Chen et al., CoRL 2024, arXiv:2406.05931) | main (clone date 2024-11; DEFORM_func.py referenced) | NONE — no LICENSE file (all rights reserved) | NOT a dependency and NOT copied; guided the structure of the first-party DER reimplementation; DEFORM's Theseus twist solver is intentionally omitted, and no `theseus` dependency exists here (see below) | Courtesy credit as prior art | No |
| torchcurves | github.com/alexshtf/torchcurves (PyPI: torchcurves) | `torchcurves>=0.2.1` | Apache-2.0 | PyPI dependency (runtime) | No obligation beyond dependency declaration (Apache-2.0, not redistributed) | No |

## Notes and residual risk

### Sparsh (CC-BY-NC 4.0)
- The root `LICENSE.md` of Sparsh is Creative Commons Attribution-NonCommercial
  4.0 International. The pretrained weights inherit these NonCommercial terms.
- Some individual Sparsh `.py` files (e.g. `tactile_ssl/model/vision_transformer.py`,
  adapted from DINO/DINOv2/timm) carry a per-file
  `# Copyright (c) Meta Platforms, Inc. and affiliates. ... Apache License, Version 2.0`
  header. The repository as a whole is still governed by the root CC-BY-NC 4.0
  license. Because Sparsh is a submodule, essentially none of it is redistributed
  here and consumers obtain it directly from Meta; the one exception is the
  Apache-2.0 excerpt recorded in the next bullet.
- **The one reproduced Sparsh excerpt (Apache-2.0).**
  `src/dlofeel_state/models/backbones/_sparsh_import.py` reproduces roughly 50
  lines derived from the `interpolate_pos_encoding` method of Sparsh's
  `tactile_ssl/model/vision_transformer.py`, as
  `_vendored_interpolate_pos_encoding`. The verbatim per-file header of that
  upstream file is:

  ```
  # Copyright (c) Meta Platforms, Inc. and affiliates.
  #
  # This source code is licensed under the Apache License, Version 2.0
  # found in the LICENSE file in the root directory of this source tree.
  ```

  So the excerpt is governed by **Apache-2.0**, not by the repository-level
  CC-BY-NC 4.0, and is reproduced here under Apache-2.0 with attribution to Meta
  Platforms, Inc. Caveat worth recording: upstream's header points at "the
  LICENSE file in the root directory", and upstream's root licence file is
  `LICENSE.md` = CC-BY-NC 4.0, so the pointer is internally inconsistent; the
  per-file grant is the more specific of the two and names Apache-2.0
  explicitly. The lines are reproduced rather than merely called because the
  released checkpoint was trained against a locally modified copy of the method
  (original-grid `t1/w1/h1` reshape instead of the pinned submodule's
  target-grid reshape); monkeypatching the vendored version back over the
  submodule is what preserves numerical parity with the released weights, and
  the parity gate fails loudly if it is dropped.
- **Commit pin rationale.** Upstream Sparsh has only three commits and only a
  `main` branch (no tags, no releases): `e1a97194` (2024-11-01, "Public release
  commit"), `04b98685` (2025-02-20, "updates demo forcefield"), and
  `fee6a05a` (2025-02-27, "Update README.md", HEAD). The pin is HEAD
  (`fee6a05a97330eca014b59c71ac41596fe3974a7`) because the vendored copy in the
  frozen original matches HEAD's file layout: it contains the renamed
  `downstream_task/forcefield_sl.py` (introduced between the release commit and
  HEAD), not the release commit's `forcefield.py`. Between `04b98685` and HEAD
  only `README.md` changed, so HEAD is code-identical to `04b98685` for all of
  `tactile_ssl`. The model file the project imports,
  `tactile_ssl/model/vision_transformer.py`, has exactly one version across the
  entire upstream history, so pinning HEAD reproduces the exact ViT base the
  project vendored.
- **Local modifications carried on top of the pin.** The vendored `src/sparsh/`
  diverges from upstream HEAD in 20 files: 15 are mechanical vendoring import
  rewrites (`tactile_ssl.` -> `sparsh.tactile_ssl.`), and 5 carry real edits.
  Four of the five are behaviorally inert (making xFormers an optional import in
  `algorithm/dinov2.py`, commenting out an xFormers-missing warning in
  `model/layers/attention.py`, an import reorder in `downstream_task/force_sl.py`,
  a blank line in `downstream_task/pose_sl.py`). The one functional change is in
  `model/vision_transformer.py`: `interpolate_pos_encoding` recomputes the
  target grid as `t1/w1/h1` from `img_size`/`patch_size` instead of reusing
  `t0/w0/h0`. This positional-embedding change is re-applied on top of the
  submodule by the monkeypatch in
  `src/dlofeel_state/models/backbones/_sparsh_import.py` (see the Apache-2.0
  excerpt bullet above); the checkpoint-parity gate fails loudly if it is
  dropped.

### DER / DEFORM (IP gray area to flag)
- `src/dlofeel_state/losses/der.py` and `bending.py` are first-party code
  with no license header. They implement the published Bergou et al. 2008 DER
  formulation. Their structure clearly parallels roahmlab/DEFORM's
  implementation (same discrete curvature-binormal "eq 1" construction, the same
  quaternion-based Bishop-frame parallel-transport `torch.where` loop, the same
  sin/cos extraction), so they are best described as adapted-from rather than
  clean-room, even though the code is substantially rewritten (snake_case free
  functions, an explicit init direction, curvature clamping, and a `theta = 0`
  shortcut in place of DEFORM's Theseus optimizer, with no `theseus` dependency).
- **Precisely what replaces DEFORM's twist solver, and what actually runs.**
  `losses/der.py::optimize_twist_angle` is a self-contained entry point that
  performs no solve: it returns `torch.zeros(batch, n_edge)`, i.e. a twist-free
  rod (`der.py:186-214`). Its only call site is
  `pipeline_points_to_material_frame` (`der.py:41`), which is itself reached only
  from the optional bending-energy path `losses/bending.py`
  (`bending.py:194,197,203`) plus each file's own `__main__` demo block
  (`der.py:291`, `bending.py:237`). Nothing in the shipped training or evaluation
  path calls `losses/bending.py` at all. The released training objective,
  `losses/weighted.py::WeightedLoss`, imports only `compute_edges`,
  `compute_edge_lengths`, and `compute_curvature_binormal` from `losses/der.py`
  (`weighted.py:4`), so neither the twist entry point nor the Bishop/material
  frame participates in it. DEFORM's Theseus solver is therefore not merely
  unreproduced, it is unreachable from anything this release runs.
- roahmlab/DEFORM ships **no license** (all rights reserved). Copyright protects
  the specific expression, not the underlying published DER algorithm, and the
  code here is a rewrite rather than a copy, so the MIT release is defensible.
  Because DEFORM is unlicensed, turning it into a dependency would NOT be safer
  than reimplementing; the independent reimplementation is the correct path.
  Recommended belt-and-suspenders: keep the Bergou-et-al. citation + DEFORM
  courtesy credit prominent, and optionally have the author confirm no lines were
  pasted from DEFORM (a clean-room diff pass) before public release.

### torchcurves
- Apache-2.0, ordinary PyPI dependency, not redistributed. No action beyond the
  dependency declaration in `pyproject.toml`.
