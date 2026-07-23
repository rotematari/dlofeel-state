"""Smoke test for the ported sequential DLO dataset.

Constructs ``DLOH5SampleDataset`` on the committed parity fixture (the same
8-frame c1-test snapshot the parity gate uses) with the k16 config's dataset
knobs, and asserts it loads and yields one sample with the expected tensor
shapes. This is a load/shape smoke test, not a numeric-parity gate
(``test_parity.py`` owns parity).

The constructor call mirrors how the original ``simplepredict.evaluate`` builds
its eval dataset (evaluate.py ~L107-122), with the two path-valued knobs
repointed at the self-contained fixture copies.
"""

import os

import torch
import yaml

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "parity")


def _make_dataset():
    from dlofeel_state.data.dataset import DLOH5SampleDataset

    with open(os.path.join(FIX, "config.yaml")) as f:
        config = yaml.safe_load(f)

    return DLOH5SampleDataset(
        root_dir=os.path.join(FIX, "samples"),
        norm_stats_path=os.path.join(FIX, "norm_stats.yaml"),
        normalize_inputs=config.get("normalize_inputs", True),
        normalize=config["normalize"],
        blank_ref_root=os.path.join(FIX, "blank_refs"),
        use_diff_images=config["use_diff_images"],
        use_blank_ref=config["use_blank_ref"],
        num_resample_points=config["num_output_pts"],
        img_size_H=config["img_size_H"],
        img_size_W=config["img_size_W"],
        transform=None,
        is_packed=False,
        past_window_steps=config.get("past_window_steps", 1),
        max_time_delta=config.get("max_time_delta", 0.13),
    )


def test_dataset_loads_fixture():
    ds = _make_dataset()
    # The fixture is 8 frozen frames under one session dir.
    assert len(ds) == 8


def test_sample_has_expected_shapes():
    ds = _make_dataset()
    sample = ds[0]

    imgs = sample["digit_images"]
    assert isinstance(imgs, torch.Tensor)
    # 4 DIGIT sensors, current+past stacked on the channel dim (3+3), 320x240.
    assert imgs.shape == (4, 6, 320, 240)
    assert imgs.dtype == torch.float32

    ee = sample["ee_pose"]
    assert isinstance(ee, torch.Tensor)
    # 3 position + 4 quaternion, EE expressed relative to the fixed frame.
    assert ee.shape == (7,)
    assert ee.dtype == torch.float32

    markers = sample["marker_positions"]
    assert isinstance(markers, torch.Tensor)
    # Raw 14 markers resampled to K=16 via the spline (num_output_pts=16).
    assert markers.shape == (16, 3)
    assert markers.dtype == torch.float32
