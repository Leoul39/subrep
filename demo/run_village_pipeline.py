"""
Certify the village options and build the six-objective skill library.

For every certification partition and every option that can start there:

    sampled contexts from the option's initiation set
        -> paired option/baseline rollouts over the evaluation horizon
        -> confidence-bounded CDS / PDS gate (FULL_SIMPLEX, six objectives)
        -> certificate with its evidence -> MeTTa store -> skill library
        -> admission audit report

Every pair, admitted or rejected, is written to the evidence file, and each
certificate can be recomputed from it. The gate thresholds and sample sizes
below are fixed before any rollout runs.

Usage:
    python -m demo.run_village_pipeline
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from certification.certificate_schema import Certificate
from certification.metta_storage import CertificateStore
from certification.statistical_gate import GateStatistics, gate_statistics
from library.skill_library import SkillLibrary
from schemas.minecraft_objective_schema import VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA
from utils.admission_report import AdmissionReport, GateEvaluation, RejectionCategory
from village_sim.contexts import PARTITIONS, option_contexts
from village_sim.env import VillageEnv
from village_sim.motives import MOTIVE_NAMES
from village_sim.options import CANDIDATE_OPTIONS, PROTECTED_WAIT, VillageOption
from village_sim.rollouts import EVALUATION_HORIZON, PairedOutcome, paired_rollout, return_scale

GAMMA = 0.99
PDS_EPSILON = 0.10
CONFIDENCE = 0.95
CONTEXTS_PER_PARTITION = 5
SEEDS_PER_CONTEXT = 6
MIN_PAIRS = 30
CONTEXT_SEED = 2026
VERSION = "1.0"

DEFAULT_OUTPUT_DIR = Path("data/village")
DEFAULT_REPORT_DIR = Path("demo/artifacts/village")
EVIDENCE_FILE = "paired_evidence.jsonl"


def _config(contexts_per_partition: int, seeds_per_context: int, partitions, options) -> dict:
    return {
        "gamma": GAMMA,
        "horizon": EVALUATION_HORIZON,
        "pds_epsilon": PDS_EPSILON,
        "confidence": CONFIDENCE,
        "contexts_per_partition": contexts_per_partition,
        "seeds_per_context": seeds_per_context,
        "min_pairs": MIN_PAIRS,
        "context_seed": CONTEXT_SEED,
        "partitions": list(partitions),
        "options": [option.option_id for option in options],
        "baseline_id": PROTECTED_WAIT.option_id,
        "version": VERSION,
    }


def _gate_evaluations(stats: GateStatistics) -> tuple[GateEvaluation, ...]:
    level = f"lcb{round(stats.confidence * 100)}"
    lhs = stats.lower_bound
    return (
        GateEvaluation("CDS", lhs, 0.0, lhs >= 0.0, f"{level}(delta_r + min(delta_n)) >= 0"),
        GateEvaluation("PDS", lhs, -PDS_EPSILON, lhs >= -PDS_EPSILON,
                       f"{level}(delta_r + min(delta_n)) >= -epsilon"),
    )


def _certificate(
    skill_id: str,
    partition: str,
    gate: str,
    stats: GateStatistics,
    seeds: tuple[int, ...],
    env: VillageEnv,
    created: str,
    run_id: str,
) -> Certificate:
    epsilon = PDS_EPSILON if gate == "PDS" else 0.0
    schema = VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA
    return Certificate(
        skill_id=skill_id,
        gate_type=gate,
        delta_r=stats.mean_delta_r,
        delta_n=stats.mean_delta_n,
        admission_margin=stats.lower_bound + epsilon,
        epsilon=epsilon,
        timestamp=created,
        seed=seeds[0],
        gamma=GAMMA,
        baseline_id=PROTECTED_WAIT.option_id,
        environment=env.metadata["environment_id"],
        episode_length=EVALUATION_HORIZON,
        version=VERSION,
        weight_region_type="FULL_SIMPLEX",
        domain_id=schema.domain_id,
        motive_schema_version=schema.motive_schema_version,
        motive_names=schema.motive_names,
        context_id=partition,
        n_rollouts=stats.n,
        rollout_seeds=seeds,
        confidence_level=stats.confidence,
        lower_bound=stats.lower_bound,
        return_scale=return_scale(GAMMA, EVALUATION_HORIZON),
        evidence_ref=f"{EVIDENCE_FILE}#run={run_id}",
    )


def _evidence_line(run_id: str, skill_id: str, partition: str, context_index: int, outcome: PairedOutcome) -> str:
    record = {"run_id": run_id, "skill_id": skill_id, "partition": partition, "context_index": context_index}
    record.update(outcome.to_dict())
    return json.dumps(record, sort_keys=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pipeline(
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    *,
    contexts_per_partition: int = CONTEXTS_PER_PARTITION,
    seeds_per_context: int = SEEDS_PER_CONTEXT,
    partitions: tuple[str, ...] = PARTITIONS,
    options: tuple[VillageOption, ...] = CANDIDATE_OPTIONS,
    verbose: bool = True,
) -> dict:
    """Certify ``options`` in ``partitions`` and write the village artifacts."""
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    report_dir = Path(report_dir or DEFAULT_REPORT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    cert_file = output_dir / "certificates.metta"
    library_file = output_dir / "library.json"
    evidence_file = output_dir / EVIDENCE_FILE
    manifest_file = output_dir / "manifest.json"
    report_json = report_dir / "admission_report.json"
    report_md = report_dir / "admission_report.md"

    config = _config(contexts_per_partition, seeds_per_context, partitions, options)
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seeds = tuple(range(seeds_per_context))

    env = VillageEnv()
    store = CertificateStore()
    library = SkillLibrary(cert_store=store, save_path=str(library_file), schema=VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA)
    report = AdmissionReport()
    evidence: list[str] = []
    records: list[dict] = []

    if verbose:
        print(f"Village certification  run={run_id}  horizon={EVALUATION_HORIZON}  "
              f"epsilon={PDS_EPSILON}  confidence={CONFIDENCE}")
        print(f"{'skill':40s} {'pairs':>5s} {'mean dr':>8s} {'worst':>8s} {'lcb':>8s}  result")

    for partition in partitions:
        for option in options:
            contexts = option_contexts(option, partition, contexts_per_partition, CONTEXT_SEED, env.mission, env.params)
            if not contexts:
                continue
            skill_id = f"{option.option_id}@{partition}"
            pairs: list[PairedOutcome] = []
            for index, context in enumerate(contexts):
                for seed in seeds:
                    outcome = paired_rollout(env, option, context, seed, GAMMA, horizon=EVALUATION_HORIZON)
                    pairs.append(outcome)
                    evidence.append(_evidence_line(run_id, skill_id, partition, index, outcome))

            stats = gate_statistics([p.delta_r for p in pairs], [p.delta_n for p in pairs], CONFIDENCE)
            enough = len(contexts) >= contexts_per_partition and len(pairs) >= MIN_PAIRS
            gate = stats.admitting_gate(PDS_EPSILON) if enough else None
            failure_reason = None
            category = None
            if not enough:
                category = RejectionCategory.INSUFFICIENT_EVIDENCE
                failure_reason = (
                    f"{len(pairs)} pairs from {len(contexts)} contexts; "
                    f"need {MIN_PAIRS} pairs from {contexts_per_partition}"
                )
            elif gate is None:
                category = RejectionCategory.GATE_FAILED
                failure_reason = (
                    f"lower bound of delta_r + min(delta_n) = {stats.lower_bound:.4f} "
                    f"< -epsilon = {-PDS_EPSILON}"
                )
            else:
                cert = _certificate(skill_id, partition, gate, stats, seeds, env, created, run_id)
                if not store.add(cert):
                    gate, category, failure_reason = None, RejectionCategory.DUPLICATE_SKILL_ID, "duplicate skill_id"
                elif not library.add_skill(skill_id, cert, option.new_run):
                    store.remove_skill(skill_id)
                    gate, category = None, RejectionCategory.LIBRARY_REVERIFICATION_FAILED
                    failure_reason = "library re-verification rejected the certificate"

            record = {
                "skill_id": skill_id,
                "candidate_policy": option.option_id,
                "admitted": gate is not None,
                "gate_type": gate,
                "delta_r": stats.mean_delta_r,
                "delta_n": stats.mean_delta_n,
                "margin": stats.lower_bound + (PDS_EPSILON if gate == "PDS" else 0.0),
                "epsilon": PDS_EPSILON if gate == "PDS" else 0.0,
                "failure_reason": failure_reason,
                "weight_region_type": "FULL_SIMPLEX",
                "gate_evaluations": _gate_evaluations(stats),
                "rejection_category": category,
                "baseline_id": PROTECTED_WAIT.option_id,
                "environment": env.metadata["environment_id"],
                "seed": seeds[0],
                "episode_length": EVALUATION_HORIZON,
            }
            report.add_from_dict(record)
            records.append({**record, "partition": partition, "pairs": len(pairs),
                            "lower_bound": stats.lower_bound, "worst_motive": MOTIVE_NAMES[stats.worst_coordinate]})
            if verbose:
                print(f"{skill_id:40s} {len(pairs):5d} {stats.mean_delta_r:8.3f} {stats.worst_mean:8.3f} "
                      f"{stats.lower_bound:8.3f}  {gate or 'REJECTED'}")

            ids_store, ids_library = set(store.get_all_skill_ids()), set(library.get_all_skill_ids())
            assert ids_store == ids_library, f"store {ids_store} and library {ids_library} diverged"

    store.save_to_file(str(cert_file))
    library.save(str(library_file))
    evidence_file.write_text("\n".join(evidence) + "\n", encoding="utf-8")
    report.save_json(str(report_json))
    report.save_markdown(str(report_md))

    reloaded = CertificateStore()
    reloaded.load_from_file(str(cert_file))
    for cert in store.load_all():
        assert reloaded.get_certificate(cert.skill_id) == cert, f"{cert.skill_id} changed on reload"
    reloaded_library = SkillLibrary(cert_store=reloaded, save_path=str(library_file),
                                    schema=VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA)
    reloaded_library.load(str(library_file))
    assert set(reloaded_library.get_all_skill_ids()) == set(library.get_all_skill_ids())

    admitted = [r for r in records if r["admitted"]]
    manifest = {
        "artifact_id": f"village-library-{run_id}",
        "run_id": run_id,
        "created": created,
        "producer": "demo.run_village_pipeline",
        "producer_version": VERSION,
        "environment": env.metadata,
        "mission": env.mission.mission_id,
        "domain_id": VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA.domain_id,
        "motive_schema_version": VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA.motive_schema_version,
        "motive_names": list(MOTIVE_NAMES),
        "config": config,
        "counts": {
            "evaluated": len(records),
            "admitted": len(admitted),
            "cds": sum(r["gate_type"] == "CDS" for r in admitted),
            "pds": sum(r["gate_type"] == "PDS" for r in admitted),
            "rejected": len(records) - len(admitted),
            "pairs": len(evidence),
        },
        "files": {path.name: _sha256(path) for path in (cert_file, library_file, evidence_file)},
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if verbose:
        counts = manifest["counts"]
        print(f"\nadmitted {counts['admitted']} (CDS {counts['cds']}, PDS {counts['pds']}), "
              f"rejected {counts['rejected']}, pairs {counts['pairs']}")
        print(f"artifacts in {output_dir} and {report_dir}")

    return {
        "run_id": run_id,
        "records": records,
        "manifest": manifest,
        "cert_file": cert_file,
        "library_file": library_file,
        "evidence_file": evidence_file,
        "manifest_file": manifest_file,
        "report_json": report_json,
        "report_md": report_md,
    }


if __name__ == "__main__":
    run_pipeline()
