"""
tests/test_phase4_village_synthetic.py
Phase 4 exit-gate tests for the Synthetic Six-Objective Village Pipeline.

These tests verify that:
  - The pipeline produces all required artifacts
  - At least one option was admitted and at least one rejected
  - All admitted certificates carry correct 6D schema identity
  - No rejected option entered the library
  - Artifacts round-trip correctly from disk
  - SkillGenerator and MDN forward-pass shapes are correct for 6D (torch-gated)
  - Paired rollouts are reproducible (same seed -> same delta)

Run with the project venv (hyperon + torch required for all tests):
    .venv\Scripts\python.exe -m pytest tests/test_phase4_village_synthetic.py -v

Run without torch (skips generator/MDN smoke tests):
    python -m pytest tests/test_phase4_village_synthetic.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Expected values
EXPECTED_DOMAIN        = "village_synthetic"
EXPECTED_SCHEMA_VER    = "1.0"
EXPECTED_MOTIVE_NAMES  = [
    "Safety", "Reputation", "DeadlineSlack",
    "InventoryValue", "Sustainability", "Infrastructure",
]
EXPECTED_MOTIVE_DIM    = 6

# Known rejected skills (not admitted)
KNOWN_REJECTED = {
    "infrastructure_first_v1",
    "wasteful_fuel_v1",
    "aggressive_burn_v1",
}


@pytest.fixture(scope="session")
def village_synthetic_artifacts(tmp_path_factory):
    """
    Session-scoped fixture that runs the synthetic pipeline in a temporary directory.

    This makes tests hermetic: they work from a clean clone without manual setup.
    """
    from demo.run_village_synthetic_pipeline import run_pipeline

    tmp_dir = tmp_path_factory.mktemp("village_synthetic")
    data_dir = tmp_dir / "data"

    return run_pipeline(
        data_dir=data_dir,
        cert_file=data_dir / "certificates.metta",
        library_file=data_dir / "library.json",
        evidence_file=data_dir / "paired_evidence.jsonl",
        report_json=tmp_dir / "admission_report.json",
        report_md=tmp_dir / "admission_report.md",
    )


def _load_cert_store(cert_file: Path):
    """Load the MeTTa certificate store from the generated file."""
    from certification.metta_storage import CertificateStore
    store = CertificateStore()
    store.load_from_file(str(cert_file))
    return store


def _load_library_json(library_file: Path) -> dict:
    """Load the raw library JSON."""
    return json.loads(library_file.read_text(encoding="utf-8"))


# ── Artifact existence tests ───────────────────────────────────────────────────

class TestArtifactsExist:
    """Verify the pipeline produced all required output files."""

    def test_cert_file_exists(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        assert cert_file.exists(), f"{cert_file} missing"

    def test_cert_file_not_empty(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        assert cert_file.exists()
        assert cert_file.stat().st_size > 0, "certificates.metta is empty"

    def test_library_file_exists(self, village_synthetic_artifacts):
        library_file = Path(village_synthetic_artifacts["library_file"])
        assert library_file.exists(), f"{library_file} missing"

    def test_report_json_exists(self, village_synthetic_artifacts):
        report_json = Path(village_synthetic_artifacts["report_json"])
        assert report_json.exists(), f"{report_json} missing"

    def test_report_md_exists(self, village_synthetic_artifacts):
        report_md = Path(village_synthetic_artifacts["report_md"])
        assert report_md.exists(), f"{report_md} missing"

    def test_report_json_is_valid(self, village_synthetic_artifacts):
        report_json = Path(village_synthetic_artifacts["report_json"])
        data = json.loads(report_json.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "admitted" in data
        assert "rejected" in data
        assert "total_attempted" in data


# ── Admission outcome tests ────────────────────────────────────────────────────

class TestAdmissionOutcomes:
    """Verify the pipeline admitted and rejected the correct number of options."""

    def test_at_least_one_admitted(self, village_synthetic_artifacts):
        assert village_synthetic_artifacts["admitted"] >= 1, (
            f"Expected at least 1 admission, got {village_synthetic_artifacts['admitted']}"
        )

    def test_at_least_one_rejected(self, village_synthetic_artifacts):
        assert village_synthetic_artifacts["rejected"] >= 1, (
            f"Expected at least 1 rejection, got {village_synthetic_artifacts['rejected']}"
        )

    def test_admitted_plus_rejected_equals_total(self, village_synthetic_artifacts):
        assert (
            village_synthetic_artifacts["admitted"] + village_synthetic_artifacts["rejected"]
            == village_synthetic_artifacts["total_attempted"]
        )

    def test_library_size_matches_admitted_count(self, village_synthetic_artifacts):
        library_file = Path(village_synthetic_artifacts["library_file"])
        lib = _load_library_json(library_file)
        lib_skills = lib.get("skills", lib.get("entries", []))
        assert len(lib_skills) == village_synthetic_artifacts["admitted"], (
            f"Library has {len(lib_skills)} entries but report says {village_synthetic_artifacts['admitted']} admitted"
        )

    def test_cert_store_count_matches_admitted_count(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        assert store.count() == village_synthetic_artifacts["admitted"], (
            f"Cert store has {store.count()} certs but report says {village_synthetic_artifacts['admitted']} admitted"
        )


# ── Certificate content tests ──────────────────────────────────────────────────

class TestCertificateContent:
    """Verify every admitted certificate carries correct 6D schema identity."""

    def test_all_certs_use_full_simplex(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        for cert in store.load_all():
            assert cert.weight_region_type == "FULL_SIMPLEX", (
                f"cert '{cert.skill_id}' has weight_region_type={cert.weight_region_type!r}, "
                "expected 'FULL_SIMPLEX'"
            )

    def test_no_cert_uses_mdn_wx(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        for cert in store.load_all():
            assert cert.weight_region_type != "MDN_WX", (
                f"cert '{cert.skill_id}' incorrectly uses MDN_WX in a FULL_SIMPLEX pipeline"
            )

    def test_delta_n_is_six_dimensional(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        for cert in store.load_all():
            assert len(cert.delta_n) == EXPECTED_MOTIVE_DIM, (
                f"cert '{cert.skill_id}' has delta_n length={len(cert.delta_n)}, "
                f"expected {EXPECTED_MOTIVE_DIM}"
            )

    def test_cert_has_domain_id(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        for cert in store.load_all():
            assert cert.domain_id == EXPECTED_DOMAIN, (
                f"cert '{cert.skill_id}' domain_id={cert.domain_id!r}, "
                f"expected {EXPECTED_DOMAIN!r}"
            )

    def test_cert_has_motive_schema_version(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        for cert in store.load_all():
            assert cert.motive_schema_version == EXPECTED_SCHEMA_VER, (
                f"cert '{cert.skill_id}' motive_schema_version={cert.motive_schema_version!r}, "
                f"expected {EXPECTED_SCHEMA_VER!r}"
            )

    def test_cert_has_correct_motive_names(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        for cert in store.load_all():
            assert cert.motive_names is not None, (
                f"cert '{cert.skill_id}' has motive_names=None"
            )
            assert list(cert.motive_names) == EXPECTED_MOTIVE_NAMES, (
                f"cert '{cert.skill_id}' motive_names={list(cert.motive_names)}, "
                f"expected {EXPECTED_MOTIVE_NAMES}"
            )

    def test_cds_cert_margin_is_non_negative(self, village_synthetic_artifacts):
        """CDS admission margin = delta_r + min(delta_n) must be >= 0."""
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        for cert in store.load_all():
            if cert.gate_type == "CDS":
                margin = float(cert.delta_r) + float(np.min(cert.delta_n))
                assert margin >= 0.0, (
                    f"CDS cert '{cert.skill_id}' has negative margin {margin:.6f}"
                )

    def test_all_certs_have_environment_field(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        for cert in store.load_all():
            assert cert.environment is not None and cert.environment != "", (
                f"cert '{cert.skill_id}' is missing environment field"
            )

    def test_all_certs_have_baseline_id(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        for cert in store.load_all():
            assert cert.baseline_id == "idle_policy_v1", (
                f"cert '{cert.skill_id}' baseline_id={cert.baseline_id!r}"
            )


# ── Round-trip / reload tests ──────────────────────────────────────────────────

class TestArtifactRoundTrip:
    """Verify saved artifacts reload correctly from disk."""

    def test_cert_file_round_trips_count(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store1 = _load_cert_store(cert_file)
        from certification.metta_storage import CertificateStore
        store2 = CertificateStore()
        store2.load_from_file(str(cert_file))
        assert store2.count() == store1.count(), (
            f"Reload count mismatch: {store2.count()} != {store1.count()}"
        )

    def test_cert_file_round_trips_skill_ids(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store1 = _load_cert_store(cert_file)
        from certification.metta_storage import CertificateStore
        store2 = CertificateStore()
        store2.load_from_file(str(cert_file))
        ids1 = {c.skill_id for c in store1.load_all()}
        ids2 = {c.skill_id for c in store2.load_all()}
        assert ids1 == ids2, f"Skill IDs differ on reload: {ids1} != {ids2}"

    def test_cert_file_round_trips_delta_n(self, village_synthetic_artifacts):
        """delta_n values survive MeTTa serialise → deserialise within float32 precision."""
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store1 = _load_cert_store(cert_file)
        from certification.metta_storage import CertificateStore
        store2 = CertificateStore()
        store2.load_from_file(str(cert_file))
        for c1 in store1.load_all():
            c2 = store2.get_certificate(c1.skill_id)
            assert c2 is not None
            np.testing.assert_allclose(
                np.array(c1.delta_n), np.array(c2.delta_n), rtol=1e-5,
                err_msg=f"delta_n mismatch for '{c1.skill_id}' after reload"
            )

    def test_library_file_round_trips_skill_ids(self, village_synthetic_artifacts):
        library_file = Path(village_synthetic_artifacts["library_file"])
        lib = _load_library_json(library_file)
        lib_skills = lib.get("skills", lib.get("entries", []))
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        cert_ids = {c.skill_id for c in store.load_all()}
        lib_ids  = {s["skill_id"] if isinstance(s, dict) else s for s in lib_skills}
        assert lib_ids == cert_ids, (
            f"Library skill IDs {lib_ids} don't match cert store IDs {cert_ids}"
        )


# ── Rejected skills safety test ────────────────────────────────────────────────

class TestRejectedSkillsSafety:
    """Verify no rejected option entered the library."""

    def test_rejected_skills_not_in_cert_store(self, village_synthetic_artifacts):
        cert_file = Path(village_synthetic_artifacts["cert_file"])
        store = _load_cert_store(cert_file)
        cert_ids = {c.skill_id for c in store.load_all()}
        overlap = KNOWN_REJECTED & cert_ids
        assert not overlap, f"Rejected skills found in cert store: {overlap}"

    def test_rejected_skills_not_in_library(self, village_synthetic_artifacts):
        library_file = Path(village_synthetic_artifacts["library_file"])
        lib = _load_library_json(library_file)
        lib_skills = lib.get("skills", lib.get("entries", []))
        lib_ids = {s["skill_id"] if isinstance(s, dict) else s for s in lib_skills}
        overlap = KNOWN_REJECTED & lib_ids
        assert not overlap, f"Rejected skills found in library: {overlap}"


# ── Reproducibility test ──────────────────────────────────────────────────────

class TestPairedRolloutReproducibility:
    """Same seed must produce identical delta values on re-run."""

    def test_same_seed_produces_same_delta_r(self):
        from village_sim.baseline import idle_policy, run_policy
        from demo.run_village_synthetic_pipeline import (
            _trade_focus_policy, GAMMA, MAX_STEPS,
        )
        seed = 42
        r_base_1, n_base_1 = run_policy(idle_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        r_cand_1, n_cand_1 = run_policy(_trade_focus_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        r_base_2, n_base_2 = run_policy(idle_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        r_cand_2, n_cand_2 = run_policy(_trade_focus_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        assert r_base_1 == r_base_2, "Baseline payoff not reproducible with same seed"
        assert r_cand_1 == r_cand_2, "Candidate payoff not reproducible with same seed"

    def test_same_seed_produces_same_delta_n(self):
        from village_sim.baseline import idle_policy, run_policy
        from demo.run_village_synthetic_pipeline import (
            _trade_focus_policy, GAMMA, MAX_STEPS,
        )
        seed = 123
        _, n_base_1 = run_policy(idle_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        _, n_cand_1 = run_policy(_trade_focus_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        _, n_base_2 = run_policy(idle_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        _, n_cand_2 = run_policy(_trade_focus_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        np.testing.assert_array_equal(n_base_1, n_base_2,
                                      err_msg="Baseline motives not reproducible")
        np.testing.assert_array_equal(n_cand_1, n_cand_2,
                                      err_msg="Candidate motives not reproducible")

    def test_different_seeds_produce_different_states(self):
        """Seeds must actually change the environment state (not degenerate)."""
        from village_sim.baseline import idle_policy, run_policy
        from demo.run_village_synthetic_pipeline import GAMMA, MAX_STEPS
        _, n1 = run_policy(idle_policy, seed=42,  max_steps=MAX_STEPS, gamma=GAMMA)
        _, n2 = run_policy(idle_policy, seed=999, max_steps=MAX_STEPS, gamma=GAMMA)
        assert not np.array_equal(n1, n2), (
            "Seeds 42 and 999 produced identical motive vectors — RNG seeding is broken"
        )


# ── SkillGenerator and MDN 6D smoke tests (torch-gated) ──────────────────────

class TestGeneratorMDNSixDimensionalSmoke:
    """
    Verify that SkillGenerator and MotiveDecompositionNetwork accept 6D inputs.

    These are pure shape/forward-pass tests — no training, no quality claims.
    Skipped automatically when torch is not installed.
    """

    def test_skill_generator_6d_output_shapes(self):
        torch = pytest.importorskip("torch")
        from generator.skill_generator import SkillGenerator

        model = SkillGenerator(input_dim=6, hidden_dim=32, motive_dim=6)
        model.eval()

        obs = torch.randn(6)
        with torch.no_grad():
            payoff, motives = model(obs)
        assert payoff.shape  == (1,), f"Expected payoff shape (1,), got {payoff.shape}"
        assert motives.shape == (6,), f"Expected motives shape (6,), got {motives.shape}"

    def test_skill_generator_6d_batch_output_shapes(self):
        torch = pytest.importorskip("torch")
        from generator.skill_generator import SkillGenerator

        model = SkillGenerator(input_dim=6, hidden_dim=32, motive_dim=6)
        model.eval()

        batch = torch.randn(4, 6)
        with torch.no_grad():
            payoff, motives = model(batch)
        assert payoff.shape  == (4, 1), f"Expected (4, 1), got {payoff.shape}"
        assert motives.shape == (4, 6), f"Expected (4, 6), got {motives.shape}"

    def test_mdn_6d_forward_inference_shapes(self):
        torch = pytest.importorskip("torch")
        from generator.mdn import MotiveDecompositionNetwork

        mdn = MotiveDecompositionNetwork(
            input_dim=6,
            num_objectives=6,
            hidden_dim=32,
            num_skills=10,
        )
        mdn.eval()

        obs = torch.randn(6)
        with torch.no_grad():
            alpha, support = mdn.forward_inference(obs)
        assert alpha.shape   == (6,), f"Expected alpha shape (6,), got {alpha.shape}"
        assert support.shape == (6,), f"Expected support shape (6,), got {support.shape}"

    def test_mdn_6d_forward_auxiliary_shapes(self):
        torch = pytest.importorskip("torch")
        from generator.mdn import MotiveDecompositionNetwork

        mdn = MotiveDecompositionNetwork(
            input_dim=6,
            num_objectives=6,
            hidden_dim=32,
            num_skills=10,
        )
        mdn.eval()

        obs      = torch.randn(6)
        skill_id = torch.tensor(0, dtype=torch.long)
        with torch.no_grad():
            gate_logit, q_hat = mdn.forward_auxiliary(obs, skill_id)
        assert gate_logit.shape == (), f"Expected scalar gate_logit, got {gate_logit.shape}"
        assert q_hat.shape      == (6,), f"Expected q_hat shape (6,), got {q_hat.shape}"

    def test_mdn_6d_alpha_is_positive(self):
        """Softplus output for alpha must be strictly positive (valid Dirichlet params)."""
        torch = pytest.importorskip("torch")
        from generator.mdn import MotiveDecompositionNetwork

        mdn = MotiveDecompositionNetwork(input_dim=6, num_objectives=6, num_skills=10)
        mdn.eval()
        obs = torch.randn(6)
        with torch.no_grad():
            alpha, _ = mdn.forward_inference(obs)
        assert torch.all(alpha > 0).item(), (
            f"All alpha values must be positive, got {alpha.tolist()}"
        )