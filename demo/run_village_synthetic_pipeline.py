"""
demo/run_village_synthetic_pipeline.py
Phase 4 — Synthetic Six-Objective Village SubRep Pipeline.

Runs paired baseline/candidate rollouts through the complete SubRep flow:

    same-seed paired rollouts → delta_r / delta_n[6]
    → CDS/PDS gate → Certificate (6D, FULL_SIMPLEX, Minecraft domain)
    → MeTTa cert store → SkillLibrary
    → AdmissionReport (JSON + Markdown)
    → reload-verification (artifacts round-trip)

This is a pure-Python, torch-free synthetic validation of the 6D SubRep
pipeline before any real Minecraft server is involved (Phase 5+).

Exit gates (asserted at end of run):
    - At least one option admitted to the library
    - At least one option rejected
    - cert_store.count() == library.count() after every admission
    - Reloaded cert file contains the same number of certificates

Usage:
    python -m demo.run_village_synthetic_pipeline
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from schemas.minecraft_objective_schema import VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA

from village_sim.baseline import idle_policy, run_policy
from village_sim.motives import MOTIVE_NAMES
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from certification.certificate_schema import Certificate
from certification.metta_storage import CertificateStore
from library.skill_library import SkillLibrary
from utils.admission_report import AdmissionReport


# ── Configuration ──────────────────────────────────────────────────────────────
# Motive order (canonical, fixed for Minecraft domain):
#   [Safety, Reputation, DeadlineSlack, InventoryValue, Sustainability, Infrastructure]

GAMMA         = 0.99
MAX_STEPS     = 100
# Three seeds per option — multi-seed evidence mirrors Phase 8 statistical pattern
SEEDS: list[int] = [42, 123, 7]
# PDS epsilon calibrated for discounted motive return scale (~63 per coordinate at gamma=0.99).
# Predeclared before final outcomes. Chosen so one deliberate trade-off candidate passes PDS only.
PDS_EPSILON   = 35.0
VERSION       = "0.1.0"
BASELINE_ID   = "idle_policy_v1"
# Synthetic domain identity — distinct from real Minecraft domain to prevent
# runtime mixing. See Phase 4 review finding #2.
DOMAIN        = "village_synthetic"
SCHEMA_VER    = "1.0"
ENVIRONMENT   = "village_sim_v1"

# Default artifact paths (can be overridden via run_pipeline() arguments)
DEFAULT_DATA_DIR      = Path("data/village_synthetic/v1")
DEFAULT_CERT_FILE     = DEFAULT_DATA_DIR / "certificates.metta"
DEFAULT_LIBRARY_FILE  = DEFAULT_DATA_DIR / "library.json"
DEFAULT_EVIDENCE_FILE = DEFAULT_DATA_DIR / "paired_evidence.jsonl"
DEFAULT_REPORT_JSON   = Path("demo/artifacts/village_synthetic/v1/admission_report.json")
DEFAULT_REPORT_MD     = Path("demo/artifacts/village_synthetic/v1/admission_report.md")
# Fixed timestamp for deterministic artifact generation (Phase 4 review finding #10).
FIXED_TIMESTAMP = "2025-09-12T00:00:00+00:00"
# ──────────────────────────────────────────────────────────────────────────────


# ── Policy definitions ──────────────────────────────────────────────────────────
# Each policy takes a VillageState snapshot and returns an action string.
# Action set: "idle", "trade", "torch_corridor", "iron_golem_spawn",
#             "reputation_first", "discount_chain", "archer_kite"

# Register policies for evidence persistence
_POLICY_REGISTRY: dict[str, object] = {}


def _torch_corridor_policy(state) -> str:
    """Improve infrastructure via torches up to 90%, then trade to complete the task."""
    if state.fuel >= 1 and state.infrastructure_pct < 0.9:
        return "torch_corridor"
    return "trade"


def _infrastructure_first_policy(state) -> str:
    """Prioritise iron-golem spawning for strong defence, then torch, then trade."""
    if state.fuel >= 3 and state.infrastructure_pct < 0.8:
        return "iron_golem_spawn"
    if state.fuel >= 1 and state.infrastructure_pct < 0.9:
        return "torch_corridor"
    return "trade"


def _trade_focus_policy(state) -> str:
    """Always trade — completes the delivery contract in the fewest steps."""
    return "trade"


def _reputation_builder_policy(state) -> str:
    """Build reputation and unlock discounts before trading heavily."""
    if state.reputation < 0.8:
        return "reputation_first"
    if state.emerald_price > 2.0:
        return "discount_chain"
    return "trade"


def _wasteful_fuel_policy(state) -> str:
    """Burn all fuel on iron-golem spawns then idle — never trades.

    Engineered to fail certification: Sustainability collapses (fuel exhausted)
    while the delivery task never completes (no trading), so delta_r is near 0
    and delta_n[Sustainability] << 0.  Both CDS and PDS gates will reject this.
    """
    if state.fuel >= 3:
        return "iron_golem_spawn"
    return "idle"  # fuel exhausted — no trading means task always fails


def _aggressive_burn_policy(state) -> str:
    """Burn all fuel on golems then trade — high Infrastructure, collapsed Sustainability.

    Completes the task (so delta_r > 0) but Sustainability suffers so severely
    that min(delta_n) << 0 and delta_r cannot compensate within CDS or PDS-0.3.
    This is a real-world anti-pattern: over-investing in defence infrastructure
    at the expense of sustainable resource usage.
    """
    if state.fuel >= 3:
        return "iron_golem_spawn"
    return "trade"  # exhausted fuel — finish task but Sustainability stays low


def _moderate_tradeoff_policy(state) -> str:
    """Moderate trade-off candidate: torch corridor for defence, then trade.

    Designed to fail CDS (Sustainability cost) but pass PDS with epsilon=35.0.
    Torches improve Safety and Infrastructure at fuel cost.
    """
    if state.fuel >= 1 and state.infrastructure_pct < 0.7:
        return "torch_corridor"
    return "trade"


# Ordered list of (option_id, policy_fn) pairs evaluated by the pipeline.
CANDIDATES: list[tuple[str, object]] = [
    ("torch_corridor_v1",       _torch_corridor_policy),
    ("infrastructure_first_v1", _infrastructure_first_policy),
    ("trade_focus_v1",          _trade_focus_policy),
    ("reputation_builder_v1",   _reputation_builder_policy),
    ("moderate_tradeoff_v1",    _moderate_tradeoff_policy),  # PDS-only admission target
    ("wasteful_fuel_v1",        _wasteful_fuel_policy),
    ("aggressive_burn_v1",      _aggressive_burn_policy),
]

# Register all policies for evidence persistence
for option_id, policy_fn in CANDIDATES:
    _POLICY_REGISTRY[option_id] = policy_fn


# ── Rollout helpers ────────────────────────────────────────────────────────────

def _run_paired(candidate_fn, seed: int) -> tuple[float, np.ndarray, float, np.ndarray, float, np.ndarray]:
    """Run baseline then candidate from the SAME seed.

    Identical reset seed guarantees both policies start from the same world
    state, mirroring the real Minecraft snapshot-restore paired-rollout
    pattern defined in Phase 8.

    Returns:
        delta_r: Scalar payoff improvement over baseline.
        delta_n: 6D motive improvement vector over baseline.
        r_base: Baseline payoff.
        n_base: Baseline motives.
        r_cand: Candidate payoff.
        n_cand: Candidate motives.
    """
    r_base, n_base = run_policy(idle_policy,  seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
    r_cand, n_cand = run_policy(candidate_fn, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
    delta_r = float(r_cand) - float(r_base)
    delta_n = (np.asarray(n_cand, dtype=np.float32)
               - np.asarray(n_base, dtype=np.float32))
    return delta_r, delta_n, r_base, n_base, r_cand, n_cand


def _aggregate_seeds(
    candidate_fn,
    seeds: list[int],
) -> tuple[float, np.ndarray, list[tuple[float, np.ndarray, float, np.ndarray, float, np.ndarray]]]:
    """Aggregate paired rollouts over multiple seeds.

    Returns:
        mean_delta_r: Mean payoff improvement across seeds.
        mean_delta_n: Mean 6D motive improvement across seeds.
        pairs:        Raw (delta_r, delta_n, r_base, n_base, r_cand, n_cand) per seed for audit logging.
    """
    pairs: list[tuple[float, np.ndarray, float, np.ndarray, float, np.ndarray]] = [
        _run_paired(candidate_fn, seed) for seed in seeds
    ]
    mean_delta_r = float(np.mean([p[0] for p in pairs]))
    mean_delta_n = np.mean([p[1] for p in pairs], axis=0).astype(np.float32)
    return mean_delta_r, mean_delta_n, pairs


# ── Certificate builder ────────────────────────────────────────────────────────

def _make_certificate(
    option_id: str,
    delta_r: float,
    delta_n: np.ndarray,
    margin: float,
    gate_type: str,
    epsilon: float,
    seeds: list[int],
) -> Certificate:
    """Build a fully-validated 6D Certificate with synthetic domain identity.

    The domain_id / motive_schema_version / motive_names triple is required
    by the Phase 3 schema contract so the SkillLibrary can enforce cross-domain
    runtime rejection and legacy quarantine rules.
    """
    return Certificate(
        skill_id=option_id,
        gate_type=gate_type,
        delta_r=float(delta_r),
        delta_n=tuple(float(v) for v in delta_n),
        admission_margin=float(margin),
        epsilon=float(epsilon),
        timestamp=FIXED_TIMESTAMP,
        seed=seeds[0],  # representative seed; full list in evidence artifact
        gamma=GAMMA,
        baseline_id=BASELINE_ID,
        environment=ENVIRONMENT,
        episode_length=MAX_STEPS,
        version=VERSION,
        weight_region_type="FULL_SIMPLEX",
        # Synthetic domain identity — marks this as a 6D synthetic artifact
        domain_id=DOMAIN,
        motive_schema_version=SCHEMA_VER,
        motive_names=tuple(MOTIVE_NAMES),
        # MDN audit fields not used in synthetic Phase 4
        certification_context=None,
        mdn_alpha=None,
        wx_support_directions=None,
        wx_support_values=None,
    )


def _persist_paired_evidence(
    option_id: str,
    pairs: list[tuple[float, np.ndarray, float, np.ndarray, float, np.ndarray]],
    seeds: list[int],
    baseline_id: str,
    candidate_id: str,
    run_id: str,
    run_timestamp: str,
    evidence_file: Path,
) -> None:
    """Persist raw paired rollout evidence to JSONL for auditability.

    Each line contains one seed's complete baseline/candidate pair outcome,
    tagged with the run_id and run_timestamp for multi-run traceability.
    """
    evidence_file.parent.mkdir(parents=True, exist_ok=True)
    with evidence_file.open("a", encoding="utf-8") as f:
        for idx, (delta_r, delta_n, r_base, n_base, r_cand, n_cand) in enumerate(pairs):
            seed = seeds[idx]
            record = {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "option_id": option_id,
                "baseline_id": baseline_id,
                "seed": seed,
                "gamma": GAMMA,
                "max_steps": MAX_STEPS,
                "baseline": {
                    "payoff": float(r_base),
                    "motives": [float(v) for v in n_base],
                },
                "candidate": {
                    "payoff": float(r_cand),
                    "motives": [float(v) for v in n_cand],
                },
                "delta": {
                    "payoff": float(delta_r),
                    "motives": [float(v) for v in delta_n],
                },
            }
            f.write(json.dumps(record) + "\n")


# Policy function registry for evidence persistence
_POLICY_REGISTRY: dict[str, object] = {}

def _get_policy_fn(option_id: str):
    """Retrieve policy function by option_id."""
    return _POLICY_REGISTRY.get(option_id)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    data_dir: Path | None = None,
    cert_file: Path | None = None,
    library_file: Path | None = None,
    evidence_file: Path | None = None,
    report_json: Path | None = None,
    report_md: Path | None = None,
) -> dict:
    """Run the Phase 4 synthetic six-objective village pipeline.

    Args:
        data_dir: Base directory for data artifacts (default: data/village_synthetic/v1)
        cert_file: Certificate output path
        library_file: SkillLibrary output path
        evidence_file: Paired evidence JSONL output path
        report_json: Admission report JSON output path
        report_md: Admission report Markdown output path

    Returns a dict of summary statistics suitable for use in exit-gate tests.
    """
    # Resolve paths (use defaults if not provided)
    data_dir = data_dir or DEFAULT_DATA_DIR
    cert_file = cert_file or data_dir / "certificates.metta"
    library_file = library_file or data_dir / "library.json"
    evidence_file = evidence_file or data_dir / "paired_evidence.jsonl"
    report_json = report_json or DEFAULT_REPORT_JSON
    report_md = report_md or DEFAULT_REPORT_MD

    # Generate unique run ID for this pipeline execution (for evidence traceability)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_timestamp = datetime.now(timezone.utc).isoformat()

    print("=" * 68)
    print("  Phase 4 -- Synthetic Six-Objective Village SubRep Pipeline")
    print("=" * 68)
    print(f"  Domain   : {DOMAIN}")
    print(f"  Schema   : v{SCHEMA_VER}")
    print(f"  Motives  : {MOTIVE_NAMES}")
    print(f"  Seeds    : {SEEDS}  ({len(SEEDS)} per option)")
    print(f"  Gamma    : {GAMMA}  |  Max steps : {MAX_STEPS}  |  PDS e : {PDS_EPSILON}")
    print(f"  Artifacts: {data_dir}")
    print(f"  Run ID   : {run_id}")
    print()

    # ── 1. Setup: gates and stores ─────────────────────────────────────────────
    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=PDS_EPSILON)

    data_dir.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    # Fresh stores — namespaced to village_synthetic/v1, separate from LunarLander data/
    cert_store = CertificateStore()
    library    = SkillLibrary(
        cert_store=cert_store,
        save_path=str(library_file),
        schema=VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA,
    )
    report     = AdmissionReport()

    # Clear any previous evidence file (we append with run_id, but fresh is cleaner for demos)
    if evidence_file.exists():
        evidence_file.unlink()

    # ── 2. Candidate evaluation loop ───────────────────────────────────────────
    admitted  = 0
    rejected  = 0
    cds_count = 0
    pds_count = 0
    episode_records: list[dict] = []

    print(
        f"{'Option':<28}  {'dR':>8}  {'min(dN)':>9}  "
        f"{'CDS':>4}  {'PDS':>4}  {'Result':<16}  {'Lib':>4}"
    )
    print("-" * 82)

    for option_id, policy_fn in CANDIDATES:
        # Multi-seed paired rollouts
        mean_dr, mean_dn, pairs = _aggregate_seeds(policy_fn, SEEDS)

        admitted_cds  = cds_gate.admit(mean_dr, mean_dn)
        admitted_pds  = pds_gate.admit(mean_dr, mean_dn)
        admitted_flag = admitted_cds or admitted_pds
        failure_reason: str | None = None

        if admitted_flag:
            gate_type = "CDS" if admitted_cds else "PDS"
            epsilon   = 0.0 if admitted_cds else PDS_EPSILON
            margin    = (cds_gate.get_admission_margin(mean_dr, mean_dn)
                         if admitted_cds
                         else pds_gate.get_admission_margin(mean_dr, mean_dn))

            cert = _make_certificate(option_id, mean_dr, mean_dn, margin, gate_type, epsilon, SEEDS)

            store_added = cert_store.add(cert)
            lib_added   = False

            if store_added:
                lib_added = library.add_skill(option_id, cert, policy_fn)
                if lib_added:
                    cds_count += int(admitted_cds)
                    pds_count += int(not admitted_cds)
                    admitted  += 1
                    result_str = f"ADMITTED [{gate_type}]"
                    # Persist raw paired evidence for admitted candidates with run ID
                    _persist_paired_evidence(option_id, pairs, SEEDS, BASELINE_ID, option_id, run_id, run_timestamp, evidence_file)
                else:
                    # Library re-verification failed — roll back cert_store to stay in sync
                    cert_store.remove_skill(option_id)
                    failure_reason = "library.add_skill() re-verification failed"
                    admitted_flag  = False
                    rejected      += 1
                    result_str     = "REJECTED"
            else:
                failure_reason = "duplicate option_id in cert_store"
                admitted_flag  = False
                rejected      += 1
                result_str     = "REJECTED"
        else:
            gate_type = None
            epsilon   = 0.0
            margin    = cds_gate.get_admission_margin(mean_dr, mean_dn)
            worst     = float(mean_dr) + float(np.min(mean_dn))
            failure_reason = (
                f"dr + min(dn) = {worst:.4f} < -e={PDS_EPSILON} (fails PDS)"
            )
            rejected  += 1
            result_str = "REJECTED"

        # Stronger invariant: cert_store and library must have identical skill IDs
        # and certificate content (not just count)
        cert_ids = set(cert_store.get_all_skill_ids())
        lib_ids  = set(library.get_all_skill_ids())
        assert cert_ids == lib_ids, (
            f"SYNC ERROR after '{option_id}': "
            f"cert_store IDs={cert_ids} != library IDs={lib_ids}"
        )
        # Also verify certificate identity for each skill
        for sid in cert_ids:
            c1 = cert_store.get_certificate(sid)
            c2 = library.get_certificate(sid)
            assert c1 is not None and c2 is not None
            assert c1.skill_id == c2.skill_id
            assert c1.delta_n == c2.delta_n
            assert c1.delta_r == c2.delta_r
            assert c1.gate_type == c2.gate_type

        print(
            f"{option_id:<28}  {mean_dr:>8.4f}  {float(np.min(mean_dn)):>9.4f}  "
            f"{'Y' if admitted_cds else 'N':>4}  {'Y' if admitted_pds else 'N':>4}  "
            f"{result_str:<18}  {library.count():>4}"
        )

        record: dict = {
            "skill_id":         option_id,
            "candidate_policy": option_id,
            "admitted":         admitted_flag,
            "gate_type":        gate_type if admitted_flag else None,
            "delta_r":          float(mean_dr),
            "delta_n":          tuple(float(v) for v in mean_dn),
            "margin":           float(margin),
            "epsilon":          float(epsilon),
            "failure_reason":   failure_reason,
            "seeds":            list(SEEDS),
            "n_pairs":          len(pairs),
        }
        episode_records.append(record)
        report.add_from_dict(record)

    # ── 3. Persist artifacts ───────────────────────────────────────────────────
    print()
    cert_store.save_to_file(str(cert_file))
    library.save(str(library_file))
    print(f"[Save] certificates -> {cert_file}")
    print(f"[Save] library      -> {library_file}")

    report.save_json(str(report_json))
    report.save_markdown(str(report_md))
    print(f"[Save] report JSON  -> {report_json}")
    print(f"[Save] report MD    -> {report_md}")
    if evidence_file.exists():
        print(f"[Save] paired evidence -> {evidence_file}")

    # ── 4. Reload verification — artifacts must round-trip ─────────────────────
    print("\n[Verify] Reloading artifacts from disk...")

    reload_store = CertificateStore()
    reload_store.load_from_file(str(cert_file))
    assert reload_store.count() == cert_store.count(), (
        f"Cert reload count mismatch: "
        f"got {reload_store.count()}, expected {cert_store.count()}"
    )

    # Verify exact certificate identity on reload
    for cert in cert_store.load_all():
        reloaded = reload_store.get_certificate(cert.skill_id)
        assert reloaded is not None, f"Missing certificate on reload: {cert.skill_id}"
        assert reloaded.delta_n == cert.delta_n
        assert reloaded.delta_r == cert.delta_r
        assert reloaded.gate_type == cert.gate_type
        assert reloaded.domain_id == cert.domain_id
        assert reloaded.motive_schema_version == cert.motive_schema_version
        assert reloaded.motive_names == cert.motive_names

    # Actual SkillLibrary round-trip with schema verification
    reload_library = SkillLibrary(
        cert_store=reload_store,
        save_path=str(library_file),
        schema=VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA,
    )
    reload_library.load(str(library_file))
    assert reload_library.count() == library.count(), (
        f"Library reload count mismatch: "
        f"got {reload_library.count()}, expected {library.count()}"
    )
    # Verify schema is persisted and matches
    assert reload_library.schema == VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA, (
        f"Library schema mismatch on reload: "
        f"got {reload_library.schema}, expected {VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA}"
    )
    # Verify skill IDs and certificates match
    assert set(reload_library.get_all_skill_ids()) == set(library.get_all_skill_ids())
    for sid in library.get_all_skill_ids():
        orig_cert = library.get_certificate(sid)
        reloaded_cert = reload_library.get_certificate(sid)
        assert orig_cert.delta_n == reloaded_cert.delta_n
        assert orig_cert.delta_r == reloaded_cert.delta_r
        assert orig_cert.gate_type == reloaded_cert.gate_type

    report_raw = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_raw["admitted"] == admitted, (
        f"Report admitted mismatch: got {report_raw['admitted']}, expected {admitted}"
    )

    print(f"[Verify] {reload_store.count()} certificates reloaded OK")
    print(f"[Verify] {reload_library.count()} library entries reloaded OK (schema-bound)")
    print("[Verify] Admission report JSON valid OK")

    # ── 5. Exit-gate assertions ────────────────────────────────────────────────
    assert admitted >= 1, (
        "EXIT GATE FAILED: at least one option must be admitted to the library"
    )
    assert rejected >= 1, (
        "EXIT GATE FAILED: at least one option must be rejected"
    )
    # Require at least one PDS-only admission (Phase 4 review finding #6)
    assert pds_count >= 1, (
        "EXIT GATE FAILED: at least one PDS-only admission required"
    )

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    total = admitted + rejected
    print("\n" + "=" * 68)
    print("  Pipeline Summary")
    print("=" * 68)
    print(f"  Options evaluated : {total}")
    print(f"  Admitted (CDS)    : {cds_count}")
    print(f"  Admitted (PDS)    : {pds_count}")
    print(f"  Rejected          : {rejected}")
    print(f"  Library size      : {library.count()}")
    print(f"  Weight region     : FULL_SIMPLEX (6-D)")
    print(f"  Domain            : {DOMAIN} v{SCHEMA_VER}")
    print(f"  Safety guarantee  : Zero rejected options entered the library")
    print("=" * 68 + "\n")

    return {
        "admitted":         admitted,
        "rejected":         rejected,
        "total_attempted":  admitted + rejected,
        "cds_count":        cds_count,
        "pds_count":        pds_count,
        "library_size":     library.count(),
        "cert_file":        str(cert_file),
        "library_file":     str(library_file),
        "evidence_file":    str(evidence_file),
        "report_json":      str(report_json),
        "report_md":        str(report_md),
        "episode_records":  episode_records,
        "domain":           DOMAIN,
        "schema_version":   SCHEMA_VER,
        "motive_names":     list(MOTIVE_NAMES),
        "run_id":           run_id,
        "run_timestamp":    run_timestamp,
    }


if __name__ == "__main__":
    run_pipeline()
