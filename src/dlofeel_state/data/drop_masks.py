"""Pure drop-mask layer shared by the tactile-drop eval and the train-time wrapper.

Single source of truth for the 15 sensor-drop masks so the test-time degradation
masks (``eval_tactile_drop.py``) and the train-time augmentation masks
(``drop_augment.py``) can never diverge. Torch-free and side-effect-free so it
imports without pulling the dataset, the model, GPU code, or Sparsh.
"""

from __future__ import annotations

import itertools

# fu=0 (fixed up), fd=1 (fixed down), ru=2 (robot up), rd=3 (robot down).
DROPPED_SENSOR_NAMES = ("fu", "fd", "ru", "rd")


def enumerate_drop_masks() -> list[dict]:
    """Enumerate the 15 drop masks (full + all 1/2/3-sensor drops; no all-4).

    Returns:
        List of 15 rows ordered by k (0,1,2,3): 1 with k=0, 4 with k=1,
        6 with k=2, 4 with k=3. Each row is
        ``{"k": int, "dropped_sensors": list[int], "drop_label": str}`` where
        ``drop_label`` names the DROPPED sensors (k=0 → "none dropped").
    """
    masks: list[dict] = []
    for k in (0, 1, 2, 3):
        for combo in itertools.combinations(range(4), k):
            dropped = list(combo)
            if k == 0:
                label = "none dropped"
            else:
                label = ", ".join(DROPPED_SENSOR_NAMES[i] for i in dropped)
            masks.append({"k": k, "dropped_sensors": dropped, "drop_label": label})
    return masks
