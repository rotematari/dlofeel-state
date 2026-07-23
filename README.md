# DLOFeel-State

Tactile + proprioceptive global shape estimation of deformable linear objects
(DLOFeel-State).

This repository is a curated, public re-implementation of a research codebase
for estimating the full 3D shape of a deformable linear object (DLO, e.g. a rope
or cable) from tactile and proprioceptive sensing. It is packaged as a clean
`src`-layout Python package (`dlofeel_state`).

> Status: scaffold. The package is installable and importable; ported logic is
> being added task by task.

## Installation

Requires Python >= 3.12.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Licensing

This repository's own source code is MIT-licensed (see `LICENSE`). It depends on
and references third-party components under their own terms; see `NOTICE` for the
short summary and `docs/THIRD_PARTY.md` for the authoritative provenance record.
Notably, the Sparsh tactile encoder is consumed as a git submodule under
CC-BY-NC 4.0 and is not redistributed here.
