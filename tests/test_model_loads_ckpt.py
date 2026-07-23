"""Strict ``state_dict`` load test for the ported ``DLOStateModel`` (Task 8).

The teeth of the model port: the k16 checkpoint's ``state_dict`` must load into a
freshly constructed ``DLOStateModel`` under ``strict=True`` with zero missing and
zero unexpected keys. Any drift in a class name, an ``nn.Module`` attribute name,
or the module hierarchy breaks this immediately.

Run with the original k16 ``best.pt`` exported as ``DLOFEEL_PARITY_CKPT``::

    DLOFEEL_PARITY_CKPT=.../k16/best.pt pytest tests/test_model_loads_ckpt.py

Without the env var the strict-load test skips (the 137 MB checkpoint is not
committed); the baseline import-smoke always runs.
"""

import os

import pytest
import torch
import yaml

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "parity")
CKPT = os.environ.get("DLOFEEL_PARITY_CKPT")


@pytest.mark.skipif(
    not CKPT,
    reason="set DLOFEEL_PARITY_CKPT to the original k16 best.pt to run the strict-load gate",
)
def test_model_loads_k16_state_dict():
    from dlofeel_state.models.dlo_state_model import DLOStateModel

    with open(os.path.join(FIX, "config.yaml")) as f:
        config = yaml.safe_load(f)

    # Construction must not depend on the Sparsh DINO pretrained file: the config
    # pins pos_embed_fn=learned, and best.pt supplies every ViT weight, so an empty
    # path skips the pretrained load while keeping the architecture identical.
    config["model_params"]["image_backbone"]["pretrained_weights_path"] = ""

    model = DLOStateModel(config)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ckpt["state_dict"], strict=True)

    assert result.missing_keys == [], f"missing keys: {result.missing_keys}"
    assert result.unexpected_keys == [], f"unexpected keys: {result.unexpected_keys}"


def test_basic_baseline_imports_and_constructs():
    # Lightweight coverage for the ResNet-18 baseline (basic.py): its own
    # ImgBackbone/PoseBackbone/DLOStateModel namespace imports and constructs.
    from dlofeel_state.models.basic import (
        DLOStateModel as BasicDLOStateModel,
        ImgBackbone,
        PoseBackbone,
    )

    config = {
        "num_output_pts": 16,
        "img_size": 240,
        "with_att": True,
        "only_pose": False,
        "dropout": 0.0,
        "model_params": {
            "latent_dim": 128,
            "pose_backbone": {"hidden_dim": 256},
            "image_backbone": {"single_encoder": True},
        },
    }
    model = BasicDLOStateModel(config)
    assert isinstance(model, BasicDLOStateModel)
    assert isinstance(model.img_enc, ImgBackbone)
    assert isinstance(model.pose_enc, PoseBackbone)
