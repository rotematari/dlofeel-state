# Third-Party Provenance

This project's own code is MIT-licensed. It relies on the third-party components
below. This file is the authoritative provenance record; NOTICE is the short
human-facing summary.

## Provenance table

| Component | Source | Version / pin | License | How consumed | Attribution required | Redistributed here? |
|---|---|---|---|---|---|---|
| Sparsh (tactile SSL encoder) | github.com/facebookresearch/sparsh | submodule pinned to `fee6a05a97330eca014b59c71ac41596fe3974a7` (branch `main`) | CC-BY-NC 4.0 (repo + weights); some source files carry per-file Apache-2.0 Meta headers | Git submodule (user clones separately); provides `tactile_ssl` ViT encoder + pretrained weights | Yes — attribute Meta / Sparsh; preserve CC-BY-NC 4.0; NonCommercial gate applies to Sparsh code + weights | No |
| Discrete Elastic Rods (DER) formulation | Bergou et al., "Discrete elastic rods," ACM TOG 27(3), 2008, doi:10.1145/1360612.1360662 | Formulation (paper) | N/A (published mathematical formulation; not code) | Independent reimplementation in `src/dlofeel_state/losses/der.py` and `bending.py` | Yes — cite Bergou et al. 2008 | Reimplemented (first-party), not copied |
| DEFORM (reference implementation) | github.com/roahmlab/DEFORM (Chen et al., CoRL 2024, arXiv:2406.05931) | main (clone date 2024-11; DEFORM_func.py referenced) | NONE — no LICENSE file (all rights reserved) | NOT a dependency and NOT copied; guided the structure of the first-party DER reimplementation; DEFORM's Theseus twist optimizer is intentionally omitted | Courtesy credit as prior art | No |
| torchcurves | github.com/alexshtf/torchcurves (PyPI: torchcurves) | `torchcurves>=0.2.1` | Apache-2.0 | PyPI dependency (runtime) | No obligation beyond dependency declaration (Apache-2.0, not redistributed) | No |

## Notes and residual risk

### Sparsh (CC-BY-NC 4.0)
- The root `LICENSE.md` of Sparsh is Creative Commons Attribution-NonCommercial
  4.0 International. The pretrained weights inherit these NonCommercial terms.
- Some individual Sparsh `.py` files (e.g. `tactile_ssl/model/vision_transformer.py`,
  adapted from DINO/DINOv2/timm) carry a per-file
  `# Copyright (c) Meta Platforms, Inc. and affiliates. ... Apache License, Version 2.0`
  header. The repository as a whole is still governed by the root CC-BY-NC 4.0
  license. Because Sparsh is a submodule (not redistributed here), neither the
  per-file Apache headers nor the root CC-BY-NC 4.0 impose redistribution
  obligations on this repository; consumers obtain Sparsh directly from Meta.
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
  `t0/w0/h0`. A later task must re-apply this positional-embedding change on top
  of the submodule (patch or thin wrapper); the checkpoint-parity gate will fail
  loudly if it is missed.

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
