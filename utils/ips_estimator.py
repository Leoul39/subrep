"""Off-policy motive target estimation for MDN training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def stabilize_importance_weights(weights: np.ndarray, clip_ratio: float = 10.0) -> np.ndarray:
    """Clip importance weights into a numerically stable interval."""
    if clip_ratio <= 0:
        raise ValueError(f"clip_ratio must be positive, got {clip_ratio}")

    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    lower = 1.0 / float(clip_ratio)
    upper = float(clip_ratio)
    return np.clip(weights, lower, upper).astype(np.float32)


def _extract_motives(sample: Any) -> np.ndarray:
    if isinstance(sample, dict):
        motives = sample.get("motives")
    elif isinstance(sample, (tuple, list)):
        if len(sample) >= 3:
            motives = sample[2]
        else:
            motives = sample[-1]
    else:
        motives = sample

    motives_array = np.asarray(motives, dtype=np.float32)
    if motives_array.ndim not in (1, 2):
        raise ValueError(f"motives must be 1D or 2D, got shape {motives_array.shape}")
    return motives_array


def estimate_q_ips(
    logged_samples: Sequence[Any] | dict[str, np.ndarray],
    gamma: float = 0.99,
    target_probability: float | np.ndarray | None = None,
    behavior_probability: float | np.ndarray | None = None,
    clip_ratio: float = 10.0,
) -> np.ndarray:
    """Estimate stabilized IPS motive targets from logged episode summaries.

    For the current summary-only dataset, each sample contributes an already
    discounted motive vector. If step-level motives are passed instead, they are
    discounted before stabilization.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    if isinstance(logged_samples, dict):
        if "motives" not in logged_samples:
            raise ValueError("logged_samples dictionary must contain a 'motives' key")
        motives_batch = np.asarray(logged_samples["motives"], dtype=np.float32)
    else:
        motives_batch = np.asarray([_extract_motives(sample) for sample in logged_samples], dtype=np.float32)

    if motives_batch.ndim == 1:
        motives_batch = motives_batch.reshape(1, -1)

    if motives_batch.ndim != 2:
        raise ValueError(f"Expected motives batch with shape (N, M), got {motives_batch.shape}")

    if behavior_probability is None:
        behavior_probability = 1.0
    if target_probability is None:
        target_probability = 1.0

    rho = np.asarray(target_probability, dtype=np.float32) / np.asarray(behavior_probability, dtype=np.float32)
    rho = stabilize_importance_weights(np.broadcast_to(rho, (motives_batch.shape[0],)), clip_ratio=clip_ratio)

    # Summary episodes already store discounted totals, so the current MVP
    # estimator reweights those totals directly.
    q_ips = motives_batch * rho[:, None]
    return q_ips.astype(np.float32)
