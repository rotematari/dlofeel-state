"""Library-port smoke tests for the training engine's optional features.

Covers the two lazy forward-deps the ported ``train.py`` resolves at runtime:
the direction-baseline model (``dlofeel_state.models.direction_model``) and its
CV core (``direction_lines``), plus the drop-augmentation dataset
(``dlofeel_state.data.augment`` / ``drop_masks``). These are pure-library ports;
no checkpoint, dataset, or GPU is required.
"""

import importlib

import numpy as np
import torch
from torch.utils.data import Dataset

# --- Port 1: direction baseline model + CV core -----------------------------

def test_direction_model_constructs_and_runs():
    from dlofeel_state.models.direction_model import DLODirectionModel

    # Minimal config exercising the required __init__ kwargs (num_output_pts,
    # dropout, model_params.{latent_dim,pose_backbone.hidden_dim}).
    config = {
        "num_output_pts": 8,
        "n_control_points": 8,
        "spline_degree": 3,
        "dropout": 0.1,
        "n_digits": 4,
        "model_params": {
            "latent_dim": 32,
            "use_spline": False,
            "use_sinusoidal_pos_encoding": True,
            "pose_backbone": {"hidden_dim": 16},
        },
    }
    model = DLODirectionModel(config)

    pos = torch.randn(2, 7)
    imgs = torch.randn(2, 4, 6, 40, 30)  # un-normalized diff pair
    out = model(pos, imgs)
    assert out.shape == (2, 8, 3), out.shape

    # imgs=None fallback path (zeros direction tokens).
    out_no_img = model(pos, None)
    assert out_no_img.shape == (2, 8, 3), out_no_img.shape


def test_get_line_params_from_diff_callable():
    from dlofeel_state.models.direction_lines import get_line_params_from_diff

    # A diagonal streak of "active" (near-1) pixels on an (H, W, 3) float diff:
    # the CV pipeline should return a 4-vector [vx, vy, x0, y0].
    diff = np.zeros((40, 30, 3), dtype=np.float32)
    for i in range(6, 34):
        diff[i, min(i - 6 + 3, 29), :] = 1.0
    params = get_line_params_from_diff(diff)
    assert isinstance(params, np.ndarray)
    assert params.shape == (4,), params.shape

    # Empty diff -> no line found -> zeros (fit_line_l1 fallback).
    zeros = get_line_params_from_diff(np.zeros((40, 30, 3), dtype=np.float32))
    assert zeros.shape == (4,)
    assert np.allclose(zeros, 0.0)


# --- Port 2: drop-augmentation dataset --------------------------------------

def test_enumerate_drop_masks_full_set():
    from dlofeel_state.data.drop_masks import enumerate_drop_masks

    masks = enumerate_drop_masks()
    # 1 (k=0) + 4 (k=1) + 6 (k=2) + 4 (k=3) = 15; no all-4 drop.
    assert len(masks) == 15, len(masks)
    ks = [m["k"] for m in masks]
    assert ks.count(0) == 1 and ks.count(1) == 4 and ks.count(2) == 6 and ks.count(3) == 4
    assert masks[0]["dropped_sensors"] == [] and masks[0]["drop_label"] == "none dropped"
    dropped_sets = {tuple(m["dropped_sensors"]) for m in masks}
    assert (0, 1, 2, 3) not in dropped_sets  # never drop all four


def test_drop_augment_dataset_expands_and_zeros():
    from dlofeel_state.data.augment import DropAugmentDataset
    from dlofeel_state.data.drop_masks import enumerate_drop_masks

    class TinyDataset(Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int) -> dict:
            return {
                "digit_images": torch.ones(4, 6, 8, 8),
                "ee_pose": torch.zeros(7),
                "marker_positions": torch.zeros(14, 3),
                "path": "p",
            }

    base = TinyDataset()
    wrapped = DropAugmentDataset(base)
    assert len(wrapped) == 15 * 2

    masks = enumerate_drop_masks()
    mask_idx = next(j for j, m in enumerate(masks) if m["dropped_sensors"] == [1, 3])
    out = wrapped[0 * 15 + mask_idx]
    assert out["digit_images"][1].sum() == 0
    assert out["digit_images"][3].sum() == 0
    assert out["digit_images"][0].sum() == 6 * 8 * 8
    assert out["digit_images"][2].sum() == 6 * 8 * 8
    # Base container untouched by the per-sample clone.
    assert bool((base[0]["digit_images"] == 1).all())


# --- train.py lazy branches resolve -----------------------------------------

def test_train_lazy_imports_resolve():
    # The two modules train.py imports lazily inside its branches must import.
    assert importlib.import_module("dlofeel_state.data.augment") is not None
    assert importlib.import_module("dlofeel_state.models.direction_model") is not None
    # And the exact symbols those branches pull.
    from dlofeel_state.data.augment import DropAugmentDataset
    from dlofeel_state.models.direction_model import DLODirectionModel
    assert DropAugmentDataset is not None and DLODirectionModel is not None
