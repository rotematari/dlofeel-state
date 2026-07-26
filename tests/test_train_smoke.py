"""SMOKE TEST: one real training step on the parity fixture, driven through the
NEW draccus entrypoint (``dlofeel_state.engine.train.main`` + ``TrainConfig``).

This checks that the TRAINING LOOP runs end to end -- dataset -> model -> loss ->
backward -> optimizer -> checkpoint -- NOT result reproduction. The ViT backbone
is random-initialized (``pretrained_weights_path=""``), so no Sparsh DINO file is
needed; the assertions are only that training finishes, the recorded loss is
finite, and a checkpoint is written.

Self-contained: uses the committed parity fixture (samples / blank_refs /
norm_stats), so it needs no ``DLOFEEL_PARITY_CKPT`` and no network. Runs on CPU or
GPU (device is auto-selected inside the training loop).
"""

import json
import math
import shutil
from pathlib import Path

import yaml

FIX = Path(__file__).parent / "fixtures" / "parity"


def test_train_one_step_writes_finite_checkpoint(tmp_path):
    import torch

    # Model config: start from the committed k16 fixture config (its
    # num_output_pts / norm_stats match the fixture data), then blank the Sparsh
    # DINO pretrained path so the ViT is random-initialized -- no external
    # checkpoint file required. The smoke checks the loop, not parity.
    with open(FIX / "config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    model_cfg["model_params"]["image_backbone"]["pretrained_weights_path"] = ""
    model_config_path = tmp_path / "model_config.yaml"
    with open(model_config_path, "w") as f:
        yaml.dump(model_cfg, f, sort_keys=False)

    # check_no_session_overlap() (a verbatim part of the ported training loop)
    # raises if train and val share a session (identified by parent-dir name). The
    # fixture holds a single session, so give val its own renamed copy of that
    # session; train points at the fixture path directly. Both wrap the same real
    # 8-sample data.
    val_root = tmp_path / "val"
    shutil.copytree(FIX / "samples" / "01_13_14_46_deg115", val_root / "val_session")

    out_dir = tmp_path / "out"

    from dlofeel_state.configs.train import TrainConfig
    from dlofeel_state.engine.train import main

    cfg = TrainConfig(
        train_data_root=str(FIX / "samples"),
        val_data_root=str(val_root),
        blank_ref_root=str(FIX / "blank_refs"),
        norm_stats_path=str(FIX / "norm_stats.yaml"),
        train_config_path=str(model_config_path),
        out_dir=str(out_dir),
        epochs=1,
        batch_size=2,
        num_workers=2,        # >=1: the ported loader hardcodes persistent_workers=True
        max_batches=1,        # ONE training step
        max_eval_batches=1,   # ONE validation batch
        use_wandb=False,
        seed=42,
    )

    # draccus.wrap() passes an already-constructed config through untouched
    # (type(args[0]) is TrainConfig), so this drives the real console entrypoint
    # in-process with no CLI parsing.
    main(cfg)

    # A checkpoint must land at <out_dir>/checkpoints/<exp_name>/best.pt
    # (best_val starts at inf, so epoch 1 always saves a best checkpoint).
    ckpts = sorted(out_dir.glob("checkpoints/*/best.pt"))
    assert ckpts, f"no best.pt checkpoint written under {out_dir}"
    ckpt_path = ckpts[0]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt.get("state_dict"), "checkpoint has no non-empty state_dict"

    # The recorded validation loss must be finite (not NaN/inf). A NaN *train*
    # loss would already have raised inside the loop; this pins the val metric.
    val_loss = float(ckpt["val_loss"])
    assert math.isfinite(val_loss), f"checkpoint val_loss is not finite: {val_loss}"

    # best_metrics.json is written alongside best.pt; its aggregate must be finite too.
    best_metrics = json.loads((ckpt_path.parent / "best_metrics.json").read_text())
    assert math.isfinite(best_metrics["val_l2_mean"]), (
        f"best_metrics val_l2_mean is not finite: {best_metrics['val_l2_mean']}"
    )
