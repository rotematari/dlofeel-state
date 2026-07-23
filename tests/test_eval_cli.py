"""CLI PARITY GATE: driving the ported eval through the NEW draccus entrypoint
(``dlofeel_state.engine.evaluate.main`` + ``EvalConfig``) must reproduce the same
golden predictions as the pure ``evaluate()`` parity gate. This proves the typed
config / CLI wiring reproduces the eval, not merely that the CLI runs.

Set ``DLOFEEL_PARITY_CKPT`` to the original k16 ``best.pt`` before invoking pytest;
without it the test skips (the fixture is committed but the 137 MB checkpoint is not).
"""

import os

import h5py
import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "parity")
CKPT = os.environ.get("DLOFEEL_PARITY_CKPT")


@pytest.mark.skipif(
    not CKPT,
    reason="set DLOFEEL_PARITY_CKPT to the original k16 best.pt to run the CLI parity gate",
)
def test_eval_cli_matches_golden(tmp_path):
    # Pin full FP32 before constructing anything torch-backed, exactly as
    # tests/test_parity.py does: torch 2.13's default cuDNN TF32 perturbs the
    # Sparsh ViT patch-embed conv by ~4e-5, so disabling it is what makes the
    # 1e-5 gate an apples-to-apples comparison across toolchain drift, NOT a
    # relaxation of the tolerance.
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from dlofeel_state.configs.eval import EvalConfig
    from dlofeel_state.engine.evaluate import main

    # Build the typed config the way a user would through the YAML override: point
    # every path-valued field at the self-contained fixture. main() auto-loads the
    # model config from config_path, repoints data_root/blank_ref_root/norm_stats,
    # blanks the Sparsh pretrained path, and calls the parity-locked evaluate().
    cfg = EvalConfig(
        data_root=os.path.join(FIX, "samples"),
        ckpt_path=CKPT,
        blank_ref_root=os.path.join(FIX, "blank_refs"),
        out_dir=str(tmp_path),
        batch_size=8,
        model_config_path=os.path.join(FIX, "config.yaml"),
        norm_stats_path=os.path.join(FIX, "norm_stats.yaml"),
    )

    # draccus.wrap() passes an already-constructed config instance through untouched
    # (type(args[0]) is EvalConfig), so this drives the real entrypoint in-process
    # with no CLI parsing.
    main(cfg)

    # main() -> evaluate() writes <out_dir>/<ckpt-parent-name>/preds_gts.h5.
    produced = sorted(tmp_path.glob("**/preds_gts.h5"))
    assert produced, f"main() wrote no preds_gts.h5 under {tmp_path}"
    new_h5 = produced[0]

    golden_path = os.path.join(FIX, "golden_preds.h5")
    with h5py.File(golden_path) as g, h5py.File(new_h5) as n:
        golden_preds = g["preds"][:]
        new_preds = n["preds"][:]

    assert golden_preds.shape == new_preds.shape, (
        f"preds shape mismatch: golden {golden_preds.shape} vs new {new_preds.shape}"
    )
    max_abs = float(np.abs(golden_preds - new_preds).max())
    assert np.allclose(golden_preds, new_preds, atol=1e-5), (
        f"CLI preds diverge from golden: max abs diff {max_abs:.2e} (atol=1e-5)"
    )
