"""Tests for the confidence-bounded CDS/PDS gate."""

from __future__ import annotations

import numpy as np
import pytest

from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from certification.statistical_gate import gate_statistics, student_t_quantile


@pytest.mark.parametrize("p, df, expected", [
    (0.95, 5, 2.015),
    (0.975, 10, 2.228),
    (0.975, 29, 2.045),
    (0.995, 20, 2.845),
    (0.99, 60, 2.390),
])
def test_t_quantile_matches_tables(p, df, expected):
    assert student_t_quantile(p, df) == pytest.approx(expected, abs=2e-3)


def test_t_quantile_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        student_t_quantile(1.0, 5)
    with pytest.raises(ValueError):
        student_t_quantile(0.9, 0)


def test_identical_pairs_give_a_bound_equal_to_the_mean():
    stats = gate_statistics([0.0] * 5, [[0.0, 0.03, 0.0]] * 5)
    assert stats.lower_bound == 0.0 == stats.worst_mean
    assert stats.admitting_gate(0.1) == "CDS"


def test_the_bound_is_below_the_worst_mean_and_names_its_coordinate():
    rng = np.random.default_rng(0)
    dr = rng.normal(0.0, 0.1, 30)
    dn = rng.normal([0.2, -0.05, 0.1], 0.05, (30, 3))
    stats = gate_statistics(dr, dn)
    assert stats.lower_bound < stats.worst_mean
    assert stats.worst_coordinate == 1
    assert stats.n == 30 and len(stats.mean_delta_n) == 3


def test_gate_thresholds():
    stats = gate_statistics([0.0, 0.0], [[-0.05, 0.1], [-0.05, 0.1]])
    assert stats.admitting_gate(0.1) == "PDS"
    assert stats.admitting_gate(0.01) is None
    with pytest.raises(ValueError):
        stats.admitting_gate(-0.1)


def test_more_objectives_widen_the_bound():
    rng = np.random.default_rng(1)
    dr = np.zeros(30)
    noise = rng.normal(0.0, 0.1, (30, 1))
    two = gate_statistics(dr, np.hstack([noise, np.ones((30, 1))]))
    six = gate_statistics(dr, np.hstack([noise, np.ones((30, 5))]))
    assert six.lower_bound < two.lower_bound


def test_admission_implies_the_point_estimate_gates_pass():
    rng = np.random.default_rng(2)
    for _ in range(200):
        dr = rng.normal(0.05, 0.1, 30)
        dn = rng.normal(rng.normal(0.0, 0.05, 6), 0.05, (30, 6))
        stats = gate_statistics(dr, dn)
        gate = stats.admitting_gate(0.1)
        if gate == "CDS":
            assert CDSGate().admit(stats.mean_delta_r, np.array(stats.mean_delta_n))
        elif gate == "PDS":
            assert PDSGate(epsilon=0.1).admit(stats.mean_delta_r, np.array(stats.mean_delta_n))


def test_simultaneous_coverage_is_at_least_the_confidence_level():
    rng = np.random.default_rng(3)
    true_means = np.array([0.1, -0.02, 0.05, 0.0, 0.03, -0.01])
    covered = 0
    trials = 2000
    for _ in range(trials):
        dn = rng.normal(true_means, 0.08, (30, 6))
        if gate_statistics(np.zeros(30), dn).lower_bound <= true_means.min():
            covered += 1
    assert covered / trials >= 0.95


@pytest.mark.parametrize("delta_r, delta_n, message", [
    ([0.0], [[0.0, 0.0]], "at least 2 pairs"),
    ([0.0, 0.0], [[0.0], [0.0]], "at least 2 objectives"),
    ([0.0, 0.0], [[0.0, 0.0]], "one row per delta_r"),
    ([0.0, np.nan], [[0.0, 0.0], [0.0, 0.0]], "finite"),
])
def test_invalid_inputs_are_rejected(delta_r, delta_n, message):
    with pytest.raises(ValueError, match=message):
        gate_statistics(delta_r, delta_n)


def test_confidence_must_be_a_probability():
    with pytest.raises(ValueError, match="confidence"):
        gate_statistics([0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]], confidence=1.0)
