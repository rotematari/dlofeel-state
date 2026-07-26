"""Train-time sensor-drop augmentation wrapper for the C1 ``DLOStateModel``.

Wraps a base ``Dataset`` whose ``__getitem__`` returns a per-sample
``digit_images`` tensor ``[4, 6, H, W]`` (dim 0 = sensor axis) and expands it by
the 15 drop masks from ``drop_masks.enumerate_drop_masks`` (real + all
1/2/3-sensor drops). Each base sample yields 15 variants; dropped sensors are
zeroed on a per-sample clone so the base container is never mutated.

This is the train-time mirror of the eval's batched ``apply_drop`` (which zeros
dim 1 of a ``[B, 4, 6, H, W]`` batch); the per-sample axis differs, so the two
must not share a function.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from dlofeel_state.data.drop_masks import enumerate_drop_masks


class DropAugmentDataset(Dataset):
    """Expand a base dataset by the 15 sensor-drop masks (real + 14 drops).

    Args:
        base: dataset whose ``__getitem__`` returns a dict with a per-sample
            ``digit_images`` tensor of shape ``[4, 6, H, W]`` (dim 0 = sensor).
        masks: drop-mask rows (as from ``enumerate_drop_masks``); defaults to the
            full 15-mask enumeration (includes the k=0 "none dropped" identity).
    """

    def __init__(self, base: Dataset, masks: list[dict] | None = None) -> None:
        self.base = base
        self.masks = masks if masks is not None else enumerate_drop_masks()

    def __len__(self) -> int:
        return len(self.masks) * len(self.base)

    def __getitem__(self, idx: int) -> dict:
        n = len(self.masks)
        # Fresh dict so the per-sample zeroing never mutates the base container.
        sample = dict(self.base[idx // n])
        dropped = self.masks[idx % n]["dropped_sensors"]
        if dropped:
            imgs = sample["digit_images"].clone()  # [4, 6, H, W]; dim 0 = sensor
            for i in dropped:
                imgs[i] = 0
            sample["digit_images"] = imgs
        return sample


def _selftest() -> None:
    """Verify expansion length, per-sample zeroing, key pass-through, no mutation."""

    class StubDataset(Dataset):
        def __len__(self) -> int:
            return 3

        def __getitem__(self, idx: int) -> dict:
            return {
                "digit_images": torch.ones(4, 6, 8, 8),
                "ee_pose": torch.zeros(7),
                "marker_positions": torch.zeros(14, 3),
                "path": "x",
            }

    stub = StubDataset()
    wrapped = DropAugmentDataset(stub)
    assert len(wrapped) == 45, len(wrapped)

    masks = enumerate_drop_masks()
    mask_idx = next(j for j, m in enumerate(masks) if m["dropped_sensors"] == [1, 3])
    idx = 0 * 15 + mask_idx
    out = wrapped[idx]

    assert out["digit_images"][1].sum() == 0
    assert out["digit_images"][3].sum() == 0
    assert out["digit_images"][0].sum() == 6 * 8 * 8
    assert out["digit_images"][2].sum() == 6 * 8 * 8
    assert "ee_pose" in out and "marker_positions" in out and "path" in out

    # Base container must be untouched by the per-sample clone.
    assert bool((stub[0]["digit_images"] == 1).all())

    assert wrapped.masks == enumerate_drop_masks()

    print("drop_augment selftest OK")


if __name__ == "__main__":
    _selftest()
