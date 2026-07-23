"""Shared, denormalized evaluation metric reducer for DLO state prediction.

Single source of truth for turning per-sample predictions into the aggregate +
per-session metric schema consumed by ``train.py`` (per-epoch learning curve),
``evaluate.py`` (single zero-shot pass), and the T0-D eval harnesses.

All inputs are ALREADY-DENORMALIZED point sets in millimetres ``[N, K, 3]``.
Standard deviations are population (ddof=0). Outputs are native Python
``float``/``int`` so the schema is JSON-serializable as-is.
"""

from pathlib import Path

import numpy as np
import torch

from dlofeel_state.losses.chamfer import (
    chamfer_distance_per_sample,
    mean_point_distance_per_sample,
)


def extract_session(path: str) -> str:
    """Extract the session id from a sample path (its parent directory name)."""
    return Path(path).parent.name


def _to_tensor(pts: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Coerce point input to a float32 CPU tensor of shape ``[N, K, 3]``."""
    if isinstance(pts, torch.Tensor):
        return pts.detach().to(dtype=torch.float32, device="cpu")
    return torch.as_tensor(np.asarray(pts), dtype=torch.float32)


def reduce_metrics(
    pred_pts: np.ndarray | torch.Tensor,
    tgt_pts: np.ndarray | torch.Tensor,
    paths: list[str],
    extra: dict[str, np.ndarray | list[float]] | None = None,
) -> dict:
    """Reduce per-sample denormalized predictions to aggregate + per-session metrics.

    Args:
        pred_pts: Predicted points, already denormalized (mm). Shape ``[N, K, 3]``.
        tgt_pts: Ground-truth points, already denormalized (mm). Shape ``[N, K, 3]``.
        paths: Length-``N`` list of sample paths; session id = parent dir name.
        extra: Optional ``{metric_name: array-of-length-N}`` of pre-computed
            per-sample scalars to aggregate alongside l2/chamfer (e.g. a
            sim-to-real gap). Each value must have length ``N`` (else ValueError).
            Aggregated globally and grouped per session as ``<name>_mean/_std``.

    Returns:
        Dict with the schema::

            {
              "aggregate": {l2_mean, l2_std, chamfer_mean, chamfer_std, <extra>_mean/_std},
              "per_session": {sid: {l2_mean, l2_std, chamfer_mean, chamfer_std, n, <extra>...}},
              "n_samples": int,
              "n_sessions": int,
            }

        Population std (ddof=0). ``N == 0`` yields nan aggregate values and an
        empty ``per_session`` (no crash). All numbers are native float/int.

    Raises:
        ValueError: If any ``extra`` array does not have length ``N``.
    """
    pred = _to_tensor(pred_pts)
    tgt = _to_tensor(tgt_pts)
    n = int(pred.shape[0])

    extra = extra or {}
    extra_np: dict[str, np.ndarray] = {}
    for name, values in extra.items():
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(
                f"extra['{name}'] has length {arr.shape[0]}, expected N={n}."
            )
        extra_np[name] = arr

    if n == 0:
        aggregate: dict[str, float] = {
            "l2_mean": float("nan"),
            "l2_std": float("nan"),
            "chamfer_mean": float("nan"),
            "chamfer_std": float("nan"),
        }
        for name in extra_np:
            aggregate[f"{name}_mean"] = float("nan")
            aggregate[f"{name}_std"] = float("nan")
        return {
            "aggregate": aggregate,
            "per_session": {},
            "n_samples": 0,
            "n_sessions": 0,
        }

    l2 = mean_point_distance_per_sample(pred, tgt).cpu().numpy().astype(np.float64)  # [N]
    chamfer = chamfer_distance_per_sample(pred, tgt).cpu().numpy().astype(np.float64)  # [N]

    sessions = [extract_session(p) for p in paths]

    def _stats(idx: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {
            "l2_mean": float(np.mean(l2[idx])),
            "l2_std": float(np.std(l2[idx])),
            "chamfer_mean": float(np.mean(chamfer[idx])),
            "chamfer_std": float(np.std(chamfer[idx])),
        }
        for name, arr in extra_np.items():
            out[f"{name}_mean"] = float(np.mean(arr[idx]))
            out[f"{name}_std"] = float(np.std(arr[idx]))
        return out

    all_idx = np.arange(n)
    aggregate = _stats(all_idx)

    per_session: dict[str, dict[str, float | int]] = {}
    for sid in sorted(set(sessions)):
        idx = np.array([i for i, s in enumerate(sessions) if s == sid], dtype=np.int64)
        stats: dict[str, float | int] = dict(_stats(idx))
        stats["n"] = int(idx.shape[0])
        per_session[sid] = stats

    return {
        "aggregate": aggregate,
        "per_session": per_session,
        "n_samples": n,
        "n_sessions": len(per_session),
    }
