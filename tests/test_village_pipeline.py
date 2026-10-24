"""Tests for the village certification pipeline and its artifacts."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict

import numpy as np
import pytest

pytest.importorskip("hyperon")

from certification.metta_storage import CertificateStore
from certification.statistical_gate import gate_statistics
from demo.run_village_pipeline import CONFIDENCE, MIN_PAIRS, PDS_EPSILON, run_pipeline
from library.skill_library import SkillLibrary
from schemas.minecraft_objective_schema import VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA
from village_sim.contexts import PARTITIONS, option_contexts, partition_of, sample_partition_state
from village_sim.mission import VillageMission
from village_sim.motives import MOTIVE_NAMES
from village_sim.options import CANDIDATE_OPTIONS, OPTIONS, SAFE_SPRINT_PATH_FOLLOW, TORCH_CORRIDOR
from village_sim.rollouts import return_scale


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("village")
    return run_pipeline(root / "data", root / "report", verbose=False)


def _store(run) -> CertificateStore:
    store = CertificateStore()
    store.load_from_file(run["cert_file"])
    return store


def _evidence(run) -> dict[str, list[dict]]:
    by_skill: dict[str, list[dict]] = defaultdict(list)
    for line in run["evidence_file"].read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        by_skill[record["skill_id"]].append(record)
    return by_skill


# ── Contexts ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("partition", PARTITIONS)
def test_sampled_states_fall_in_their_partition(partition):
    for seed in range(30):
        assert partition_of(sample_partition_state(partition, random.Random(seed))) == partition


def test_unknown_partition_is_rejected():
    with pytest.raises(ValueError, match="partition"):
        sample_partition_state("noon", random.Random(0))


def test_option_contexts_are_deterministic_and_startable():
    first = option_contexts(SAFE_SPRINT_PATH_FOLLOW, "dusk", 5, seed=7)
    again = option_contexts(SAFE_SPRINT_PATH_FOLLOW, "dusk", 5, seed=7)
    assert first == again and len(first) == 5
    assert all(SAFE_SPRINT_PATH_FOLLOW.can_start(s, VillageMission()) for s in first)
    assert option_contexts(TORCH_CORRIDOR, "dusk", 5, seed=8) != first


def test_options_that_never_start_in_a_partition_get_no_contexts():
    assert option_contexts(OPTIONS["archer_kite_v1"], "dusk", 5, seed=1) == []


# ── Artifacts ───────────────────────────────────────────────────────────────

def test_every_artifact_is_written(run):
    for key in ("cert_file", "library_file", "evidence_file", "manifest_file", "report_json", "report_md"):
        assert run[key].exists() and run[key].stat().st_size > 0, key


def test_some_options_are_admitted_and_some_rejected(run):
    admitted = [r for r in run["records"] if r["admitted"]]
    assert 0 < len(admitted) < len(run["records"])
    assert {r["gate_type"] for r in admitted} == {"CDS", "PDS"}


def test_skill_ids_name_an_option_and_a_partition(run):
    for record in run["records"]:
        option_id, partition = record["skill_id"].split("@")
        assert option_id in {o.option_id for o in CANDIDATE_OPTIONS}
        assert partition in PARTITIONS


def test_store_library_and_records_agree(run):
    store = _store(run)
    library = SkillLibrary(cert_store=store, schema=VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA)
    library.load(str(run["library_file"]))
    admitted = {r["skill_id"] for r in run["records"] if r["admitted"]}
    assert set(store.get_all_skill_ids()) == admitted
    assert set(library.get_all_skill_ids()) == admitted


def test_certificates_carry_identity_and_evidence(run):
    for cert in _store(run).load_all():
        assert cert.weight_region_type == "FULL_SIMPLEX"
        assert len(cert.delta_n) == 6 and list(cert.motive_names) == MOTIVE_NAMES
        assert cert.domain_id == "village_synthetic" and cert.motive_schema_version == "2.0"
        assert cert.baseline_id == "protected_wait_v1"
        assert cert.context_id == cert.skill_id.split("@")[1]
        assert cert.n_rollouts >= MIN_PAIRS
        assert cert.confidence_level == CONFIDENCE
        assert cert.epsilon == (PDS_EPSILON if cert.gate_type == "PDS" else 0.0)
        assert cert.lower_bound >= -cert.epsilon
        assert cert.evidence_ref.endswith(run["run_id"])


def test_every_certificate_can_be_recomputed_from_the_evidence(run):
    evidence = _evidence(run)
    for cert in _store(run).load_all():
        pairs = evidence[cert.skill_id]
        stats = gate_statistics(
            [p["delta"]["payoff"] for p in pairs],
            [p["delta"]["motives"] for p in pairs],
            cert.confidence_level,
        )
        assert stats.n == cert.n_rollouts
        assert stats.mean_delta_r == pytest.approx(cert.delta_r)
        assert np.allclose(stats.mean_delta_n, cert.delta_n)
        assert stats.lower_bound == pytest.approx(cert.lower_bound)


def test_rejected_options_keep_their_evidence_but_not_a_certificate(run):
    evidence = _evidence(run)
    store = _store(run)
    rejected = [r for r in run["records"] if not r["admitted"]]
    assert rejected
    for record in rejected:
        assert len(evidence[record["skill_id"]]) == record["pairs"]
        assert not store.contains(record["skill_id"])
        assert record["rejection_category"] == "GATE_FAILED"
        assert record["lower_bound"] < -PDS_EPSILON


def test_evidence_pairs_run_until_sunrise_with_a_matching_scale(run):
    mission = VillageMission()
    for pairs in _evidence(run).values():
        for pair in pairs:
            assert pair["baseline_id"] == "protected_wait_v1"
            assert 1 <= pair["horizon"] <= mission.night_steps
            assert pair["return_scale"] == pytest.approx(return_scale(pair["gamma"], pair["horizon"]))
            assert pair["candidate"]["option_steps"] <= OPTIONS[pair["option_id"]].max_steps
            assert pair["candidate"]["env_steps"] <= pair["horizon"]


def test_manifest_records_config_counts_and_checksums(run):
    manifest = json.loads(run["manifest_file"].read_text(encoding="utf-8"))
    assert manifest["config"]["pds_epsilon"] == PDS_EPSILON
    assert manifest["config"]["confidence"] == CONFIDENCE
    assert manifest["motive_names"] == MOTIVE_NAMES
    assert manifest["counts"]["pairs"] == sum(r["pairs"] for r in run["records"])
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((run["cert_file"].parent / name).read_bytes()).hexdigest() == digest


def test_report_matches_the_records(run):
    report = json.loads(run["report_json"].read_text(encoding="utf-8"))
    assert report["admitted"] == sum(r["admitted"] for r in run["records"])
    assert report["total_attempted"] == len(run["records"])


def test_a_second_run_reproduces_the_evidence_and_certificates(run, tmp_path):
    again = run_pipeline(tmp_path / "data", tmp_path / "report", verbose=False)
    assert again["run_id"] == run["run_id"]
    assert again["evidence_file"].read_bytes() == run["evidence_file"].read_bytes()
    first = {c.skill_id: c.to_dict() | {"timestamp": None} for c in _store(run).load_all()}
    second = {c.skill_id: c.to_dict() | {"timestamp": None} for c in _store(again).load_all()}
    assert first == second


def test_too_few_contexts_is_insufficient_evidence(tmp_path):
    small = run_pipeline(
        tmp_path / "data", tmp_path / "report",
        contexts_per_partition=2, seeds_per_context=3,
        partitions=("dusk",), options=(TORCH_CORRIDOR,), verbose=False,
    )
    (record,) = small["records"]
    assert not record["admitted"]
    assert record["rejection_category"] == "INSUFFICIENT_EVIDENCE"
    assert _store(small).count() == 0
