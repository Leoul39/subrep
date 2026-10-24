"""
Confidence-bounded CDS/PDS admission over paired rollouts.

Over the full simplex the admission score is

    inf_w E[delta_r + w . delta_n] = min_j E[delta_r + delta_n_j]

The gate puts a one-sided lower confidence bound on each of the M
expectations E[delta_r + delta_n_j] (Student t, with the error rate split
evenly across the M coordinates so the bounds hold simultaneously) and admits
on the smallest of them:

    CDS:  lower_bound >= 0
    PDS:  lower_bound >= -epsilon

Each bound is at most its sample mean, so anything admitted here also passes
the point-estimate gates that the skill library re-verifies.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np


def student_t_quantile(p: float, df: int) -> float:
    """Quantile of Student's t distribution.

    Uses the Cornish-Fisher expansion around the normal quantile (Abramowitz
    and Stegun 26.7.5); the absolute error is below 1e-3 for ``df >= 5``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if df < 1:
        raise ValueError(f"df must be at least 1, got {df}")
    z = NormalDist().inv_cdf(p)
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


@dataclass(frozen=True)
class GateStatistics:
    """Sample summary of paired deltas and the bound the gate decides on."""

    n: int
    confidence: float
    mean_delta_r: float
    mean_delta_n: tuple[float, ...]
    worst_mean: float
    lower_bound: float
    worst_coordinate: int

    def admitting_gate(self, epsilon: float) -> str | None:
        """'CDS', 'PDS' or None, deciding on the lower bound."""
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        if self.lower_bound >= 0.0:
            return "CDS"
        if self.lower_bound >= -epsilon:
            return "PDS"
        return None


def gate_statistics(
    delta_r: Sequence[float],
    delta_n: Sequence[Sequence[float]],
    confidence: float = 0.95,
) -> GateStatistics:
    """Summarise ``n`` paired deltas into the confidence-bounded admission score.

    Args:
        delta_r: Payoff deltas, one per pair.
        delta_n: Motive deltas, one row of length M per pair.
        confidence: Simultaneous coverage of the M one-sided bounds.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    dr = np.asarray(delta_r, dtype=np.float64).reshape(-1)
    dn = np.asarray(delta_n, dtype=np.float64)
    if dn.ndim != 2 or dn.shape[0] != dr.shape[0]:
        raise ValueError(f"delta_n must have one row per delta_r, got shapes {dr.shape} and {dn.shape}")
    n, m = dn.shape
    if n < 2:
        raise ValueError(f"at least 2 pairs are needed for a confidence bound, got {n}")
    if m < 2:
        raise ValueError(f"delta_n must describe at least 2 objectives, got {m}")
    if not (np.all(np.isfinite(dr)) and np.all(np.isfinite(dn))):
        raise ValueError("deltas must be finite")

    scores = dr[:, None] + dn
    means = scores.mean(axis=0)
    standard_errors = scores.std(axis=0, ddof=1) / np.sqrt(n)
    t = student_t_quantile(1.0 - (1.0 - confidence) / m, n - 1)
    bounds = means - t * standard_errors
    worst = int(np.argmin(bounds))
    return GateStatistics(
        n=int(n),
        confidence=float(confidence),
        mean_delta_r=float(dr.mean()),
        mean_delta_n=tuple(float(v) for v in dn.mean(axis=0)),
        worst_mean=float(means.min()),
        lower_bound=float(bounds[worst]),
        worst_coordinate=worst,
    )
