"""Tests for the village weight sources: the scenario schedule and the trained MDN."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("hyperon")

from generator.mdn import MotiveDecompositionNetwork
from library.skill_library import SkillLibrary
from schemas.minecraft_objective_schema import VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA
from village_sim.contexts import PARTITIONS, sample_partition_state
from village_sim.motives import MOTIVE_NAMES, PSI_NAMES
from village_sim.night import NightRunner, add_certified
from village_sim.state import dusk_state
from village_sim.weights import (
    CONCENTRATION,
    MOTIVE_SHIFT,
    SCENARIO_WEIGHTS,
    MDNWeights,
    ScenarioWeights,
    evaluate_village_mdn,
    load_weight_model,
    sample_training_contexts,
    scenario_region,
    scenario_weights,
    train_village_mdn,
)


@pytest.fixture(scope="module")
def mdn():
    weights, _ = train_village_mdn(sample_training_contexts(60, seed=1), epochs=600, seed=0)
    return weights


def test_scenario_regions_are_feasible_and_contain_the_schedule():
    import random

    for partition in PARTITIONS:
        state = sample_partition_state(partition, random.Random(0))
        s = scenario_region(state)
        assert np.all(s <= 1.0) and s.sum() >= 1.0
        assert np.allclose(s, np.asarray(SCENARIO_WEIGHTS[partition]) + MOTIVE_SHIFT)
        assert np.all(scenario_weights(state) <= s)


def test_scenario_source_certifies_over_the_full_simplex():
    view = ScenarioWeights()(dusk_state())
    assert view.source == "scenario" and view.support_values is None and view.alpha is None
    assert np.allclose(view.weights, SCENARIO_WEIGHTS["dusk"])


def test_the_mdn_reproduces_the_schedule_on_held_out_contexts(mdn):
    metrics = evaluate_village_mdn(mdn, sample_training_contexts(40, seed=99))
    assert metrics["feasible_region_rate"] == 1.0
    assert metrics["mean_weight_error"] < 0.05
    assert metrics["max_support_error"] < 0.2


def test_the_mdn_gives_dirichlet_weights_and_a_region(mdn):
    view = mdn(dusk_state())
    assert view.source == "mdn"
    assert view.weights.sum() == pytest.approx(1.0)
    assert np.all(view.alpha > 0.0)
    assert view.alpha.sum() == pytest.approx(CONCENTRATION, rel=0.3)
    assert view.context.shape == (len(PSI_NAMES),)
    assert np.all(view.support_values >= view.weights - 1e-6)


def test_checkpoints_round_trip_and_refuse_other_schemas(mdn, tmp_path):
    path = mdn.save(tmp_path / "village_mdn.pth")
    again = MDNWeights.load(path)
    state = dusk_state()
    assert np.allclose(again(state).alpha, mdn(state).alpha)

    checkpoint = torch.load(path, weights_only=False)
    checkpoint["psi_names"] = checkpoint["psi_names"][:-1]
    torch.save(checkpoint, tmp_path / "other.pth")
    with pytest.raises(ValueError, match="different context"):
        MDNWeights.load(tmp_path / "other.pth")


def test_an_mdn_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="village needs"):
        MDNWeights(MotiveDecompositionNetwork(input_dim=8, num_objectives=2))


def test_an_unusable_checkpoint_falls_back_to_the_schedule(tmp_path):
    model, reason = load_weight_model(tmp_path / "missing.pth")
    assert isinstance(model, ScenarioWeights) and "unusable" in reason
    model, reason = load_weight_model(None)
    assert isinstance(model, ScenarioWeights) and reason


def test_an_mdn_night_certifies_over_the_contextual_region(mdn):
    from demo.run_village_night import SCENARIOS

    night = NightRunner(weight_model=mdn, model_seeds=6).run(seed=3, start=SCENARIOS["raid_at_dusk"]())
    assert night.certificates
    for decision in night.decisions:
        assert decision.weight_source == "mdn"
        s = np.asarray(decision.support_values)
        assert np.all(s <= 1.0) and s.sum() >= 1.0 - 1e-9
        for assessment in decision.assessments:
            if assessment.gate is not None:
                assert assessment.cvar is not None
    library = SkillLibrary(schema=VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA)
    for cert in night.certificates:
        assert cert.weight_region_type == "MDN_WX"
        assert len(cert.mdn_alpha) == len(MOTIVE_NAMES)
        assert add_certified(library, cert, policy=lambda state: None)
