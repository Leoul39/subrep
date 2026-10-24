"""Tests for running a village night as a chain of certified decisions."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("hyperon")

from village_sim.contexts import partition_of
from village_sim.night import SCENARIO_WEIGHTS, NightRunner, scenario_weights
from village_sim.options import PROTECTED_WAIT
from village_sim.state import TASK_OPEN, dusk_state


@pytest.fixture(scope="module")
def runner():
    return NightRunner(model_seeds=6)


@pytest.fixture(scope="module")
def raid_night(runner):
    start = dusk_state()
    start.raid_active, start.raid_waves_left, start.raiders = True, 3, 3
    return runner.run(seed=4, start=start)


def test_scenario_weights_are_simplex_points_for_every_partition():
    for weights in SCENARIO_WEIGHTS.values():
        assert len(weights) == 6
        assert sum(weights) == pytest.approx(1.0) and min(weights) >= 0.0
    assert np.allclose(scenario_weights(dusk_state()), SCENARIO_WEIGHTS["dusk"])


def test_a_night_ends_and_records_its_outcome(raid_night):
    assert raid_night.task_status != TASK_OPEN
    ticks = [d.world_tick for d in raid_night.decisions]
    assert ticks == sorted(ticks) and len(set(ticks)) == len(ticks)
    record = json.loads(json.dumps(raid_night.to_dict()))
    assert record["task_status"] == raid_night.task_status
    assert len(record["decisions"]) == len(raid_night.decisions)


def test_nights_are_reproducible(runner):
    first = runner.run(seed=2)
    again = runner.run(seed=2)
    assert [(d.world_tick, d.chosen, d.steps) for d in first.decisions] == [
        (d.world_tick, d.chosen, d.steps) for d in again.decisions
    ]
    assert first.final_state == again.final_state


def test_every_choice_is_admitted_and_best_or_a_protected_wait(raid_night):
    for decision in raid_night.decisions:
        admitted = {a.option_id: a for a in decision.assessments if a.gate is not None}
        if decision.chosen == PROTECTED_WAIT.option_id:
            assert not admitted or max(a.score for a in admitted.values()) <= 0.0
            continue
        assert decision.chosen in admitted
        assert decision.score == pytest.approx(max(a.score for a in admitted.values()))


def test_weights_follow_the_situation_at_each_decision(raid_night):
    for decision in raid_night.decisions:
        assert decision.weights == SCENARIO_WEIGHTS[decision.partition]


def test_every_admission_has_a_consistent_certificate(raid_night):
    certificates = {c.skill_id: c for c in raid_night.certificates}
    assert len(certificates) == len(raid_night.certificates)
    for decision in raid_night.decisions:
        label = f"night-4-tick-{decision.world_tick}"
        for assessment in decision.assessments:
            skill_id = f"{assessment.option_id}@{label}"
            if assessment.gate is None:
                assert skill_id not in certificates
                continue
            cert = certificates[skill_id]
            assert cert.context_id == label and cert.gate_type == assessment.gate
            assert cert.lower_bound == pytest.approx(assessment.lower_bound)
            assert cert.lower_bound >= -cert.epsilon


def test_without_options_the_agent_waits_until_the_contract_fails():
    night = NightRunner(options=(), model_seeds=2).run(seed=0)
    assert night.task_status == "failed"
    assert {d.chosen for d in night.decisions} == {PROTECTED_WAIT.option_id}
    assert night.final_state.world_tick >= 23000


def test_ungated_selection_picks_the_best_score_whether_certified_or_not(runner):
    ungated = NightRunner(model_seeds=6, gated=False)
    night = ungated.run(seed=1)
    for decision in night.decisions:
        if decision.chosen == PROTECTED_WAIT.option_id:
            continue
        assert decision.score == pytest.approx(max(a.score for a in decision.assessments))


def test_decisions_are_made_in_the_partition_of_their_state(raid_night):
    assert raid_night.decisions[0].partition == partition_of(
        dusk_state().__class__(**{**dusk_state().__dict__, "raid_active": True, "raid_waves_left": 3, "raiders": 3})
    )


def test_too_few_model_seeds_are_rejected():
    with pytest.raises(ValueError, match="model_seeds"):
        NightRunner(model_seeds=1)
