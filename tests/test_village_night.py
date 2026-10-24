"""Tests for running a village night as a chain of certified decisions."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("hyperon")

from village_sim.contexts import partition_of
from village_sim.night import SCENARIO_WEIGHTS, NightRunner, scenario_weights
from village_sim.options import PROTECTED_WAIT
from village_sim.mission import VillageMission
from village_sim.state import dusk_state


@pytest.fixture(scope="module")
def runner():
    return NightRunner(model_seeds=6)


@pytest.fixture(scope="module")
def raid_night(runner):
    from demo.run_village_night import SCENARIOS

    return runner.run(seed=4, start=SCENARIOS["raid_at_dusk"]())


def test_scenario_weights_are_simplex_points_for_every_partition():
    for weights in SCENARIO_WEIGHTS.values():
        assert len(weights) == 6
        assert sum(weights) == pytest.approx(1.0) and min(weights) >= 0.0
    assert np.allclose(scenario_weights(dusk_state()), SCENARIO_WEIGHTS["dusk"])


def test_a_night_ends_and_records_its_outcome(raid_night):
    assert raid_night.final_state.night_over(VillageMission())
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
    from demo.run_village_night import SCENARIOS

    assert raid_night.decisions[0].partition == partition_of(SCENARIOS["raid_at_dusk"]()) == "raid"


def test_too_few_model_seeds_are_rejected():
    with pytest.raises(ValueError, match="model_seeds"):
        NightRunner(model_seeds=1)


def test_runtime_certificates_can_be_recomputed_from_the_decision_evidence(raid_night):
    from collections import defaultdict

    from certification.statistical_gate import gate_statistics

    pairs = defaultdict(list)
    for record in raid_night.evidence:
        pairs[record["skill_id"]].append(record)
    assert raid_night.certificates
    for cert in raid_night.certificates:
        records = pairs[cert.skill_id]
        stats = gate_statistics(
            [r["delta"]["payoff"] for r in records],
            [r["delta"]["motives"] for r in records],
            cert.confidence_level,
        )
        assert stats.n == cert.n_rollouts
        assert stats.lower_bound == pytest.approx(cert.lower_bound)
        assert np.allclose(stats.mean_delta_n, cert.delta_n)


def test_every_night_runs_to_sunrise_unless_the_player_dies(raid_night):
    final = raid_night.final_state
    assert final.world_tick >= 23000 or final.player_health <= 0.0 or final.task_status == "failed"


def test_the_night_report_compares_policies_at_sunrise(tmp_path):
    from demo.run_village_night import POLICIES, SCENARIOS, run_comparison

    out = run_comparison(nights=1, report_dir=tmp_path, model_seeds=3, verbose=False)
    report = out["report"]
    assert set(report["scenarios"]) == set(SCENARIOS)
    for scenario in SCENARIOS:
        assert set(report["scenarios"][scenario]) == set(POLICIES)
        assert report["scenarios"][scenario]["wait"]["delivered_rate"] == 0.0
        for name in ("night_{}.json", "night_{}_certificates.metta", "night_{}_evidence.jsonl"):
            assert (tmp_path / name.format(scenario)).stat().st_size > 0
    assert "First SubRep night" in out["report_md"].read_text(encoding="utf-8")
