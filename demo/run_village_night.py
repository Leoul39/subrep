"""
Run village nights with SubRep option selection and compare it with two references.

Scenarios:
    calm_dusk     the reference start: dusk at the trading hall
    raid_at_dusk  the same start with a raid already under way

Policies:
    subrep       certified selection with the scenario weight schedule
    subrep_mdn   certified selection with MDN weights over the MDN's weight
                 region (falls back to the schedule, and says so, when the
                 checkpoint is missing or incompatible)
    ungated      the schedule's scores with certification ignored
    wait         the protected-wait baseline only

Every night runs to sunrise, so outcomes are compared at the same moment. For
each scenario and policy the report gives the delivery rate, casualties, raid
outcomes, reputation and mean final motives. For the first SubRep night of each
scenario it also writes the decision timeline, the runtime certificates and the
paired rollouts behind them.

Usage:
    python -m demo.run_village_night [--nights N] [--model-seeds K] [--mdn-checkpoint PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from certification.metta_storage import CertificateStore
from village_sim.motives import MOTIVE_NAMES, phi
from village_sim.night import NightResult, NightRunner
from village_sim.options import PROTECTED_WAIT
from village_sim.state import TASK_DELIVERED, VillageState, dusk_state
from village_sim.weights import load_weight_model

DEFAULT_REPORT_DIR = Path("demo/artifacts/village")
DEFAULT_MDN_CHECKPOINT = Path("models/village_mdn.pth")


def _raid_at_dusk() -> VillageState:
    state = dusk_state()
    state.raid_active = True
    state.raid_waves_left = 5
    state.raiders = 5
    return state


SCENARIOS = {"calm_dusk": dusk_state, "raid_at_dusk": _raid_at_dusk}
POLICIES = ("subrep", "subrep_mdn", "ungated", "wait")


def _summarise(results: list[NightResult]) -> dict:
    finals = np.array([phi(r.final_state) for r in results])
    chosen = Counter(d.chosen for r in results for d in r.decisions if d.chosen != PROTECTED_WAIT.option_id)
    return {
        "nights": len(results),
        "delivered_rate": float(np.mean([r.task_status == TASK_DELIVERED for r in results])),
        "mean_casualties": float(np.mean([r.casualties for r in results])),
        "raid_outcomes": dict(sorted(Counter(r.final_state.raid_outcome for r in results).items())),
        "mean_final_reputation": float(np.mean([r.final_state.reputation for r in results])),
        "mean_final_phi": dict(zip(MOTIVE_NAMES, (float(v) for v in finals.mean(axis=0)))),
        "options_run": dict(sorted(chosen.items())),
    }


def _timeline_markdown(night: NightResult) -> list[str]:
    lines = [
        "| Tick | Situation | Admitted | Chosen | Score | Steps | Events |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in night.decisions:
        if d.chosen == PROTECTED_WAIT.option_id and not d.assessments and not d.events:
            continue
        admitted = ", ".join(a.option_id.removesuffix("_v1") for a in d.assessments if a.gate) or "none"
        score = "" if d.score is None else f"{d.score:.3f}"
        lines.append(f"| {d.world_tick} | {d.partition} | {admitted} | {d.chosen.removesuffix('_v1')} "
                     f"| {score} | {d.steps} | {', '.join(d.events)} |")
    return lines


def _write_night_artifacts(night: NightResult, report_dir: Path, scenario: str) -> None:
    (report_dir / f"night_{scenario}.json").write_text(json.dumps(night.to_dict(), indent=2) + "\n", encoding="utf-8")
    store = CertificateStore()
    for cert in night.certificates:
        store.add(cert)
    store.save_to_file(report_dir / f"night_{scenario}_certificates.metta")
    lines = (json.dumps(record, sort_keys=True) for record in night.evidence)
    (report_dir / f"night_{scenario}_evidence.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(
    nights: int = 10,
    report_dir: Path | None = None,
    *,
    model_seeds: int = 30,
    mdn_checkpoint: Path | None = DEFAULT_MDN_CHECKPOINT,
    verbose: bool = True,
) -> dict:
    """Run every scenario under every policy and write the night report."""
    report_dir = Path(report_dir or DEFAULT_REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    mdn_weights, mdn_fallback = load_weight_model(mdn_checkpoint)
    runners = {
        "subrep": dict(gated=True),
        "subrep_mdn": dict(gated=True, weight_model=mdn_weights),
        "ungated": dict(gated=False),
        "wait": dict(options=()),
    }
    report: dict = {
        "nights_per_cell": nights,
        "model_seeds": model_seeds,
        "mdn_checkpoint": None if mdn_checkpoint is None else str(mdn_checkpoint),
        "mdn_fallback": mdn_fallback,
        "scenarios": {},
    }
    markdown = ["# Village night report", "",
                f"{nights} nights per scenario and policy; {model_seeds} paired rollouts per option "
                "at each decision; every night runs to sunrise.", ""]
    if mdn_fallback:
        markdown += [f"**subrep_mdn fell back to the scenario schedule:** {mdn_fallback}", ""]

    for scenario, make_start in SCENARIOS.items():
        markdown += [f"## Scenario: {scenario}", "",
                     "| Policy | Delivered | Casualties | Raid outcomes | Reputation | "
                     + " | ".join(MOTIVE_NAMES) + " |",
                     "|---|---|---|---|---|" + "---|" * len(MOTIVE_NAMES)]
        report["scenarios"][scenario] = {}
        first_subrep: NightResult | None = None
        for policy in POLICIES:
            runner = NightRunner(model_seeds=model_seeds, **runners[policy])
            results = [runner.run(seed, start=make_start()) for seed in range(nights)]
            summary = _summarise(results)
            report["scenarios"][scenario][policy] = summary
            if policy == "subrep":
                first_subrep = results[0]
            raids = ", ".join(f"{k} {v}" for k, v in summary["raid_outcomes"].items())
            motives = " | ".join(f"{v:.3f}" for v in summary["mean_final_phi"].values())
            markdown.append(f"| {policy} | {summary['delivered_rate']:.0%} | {summary['mean_casualties']:.2f} "
                            f"| {raids} | {summary['mean_final_reputation']:.2f} | {motives} |")
            if verbose:
                print(f"{scenario:13s} {policy:8s} delivered {summary['delivered_rate']:.0%}  "
                      f"casualties {summary['mean_casualties']:.2f}  raids {summary['raid_outcomes']}  "
                      f"options {summary['options_run']}")

        _write_night_artifacts(first_subrep, report_dir, scenario)
        markdown += ["", f"### First SubRep night: {first_subrep.task_status}, "
                     f"{first_subrep.casualties} casualties, raid {first_subrep.final_state.raid_outcome}", ""]
        markdown += _timeline_markdown(first_subrep) + [""]

    report_json = report_dir / "night_report.json"
    report_md = report_dir / "night_report.md"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if verbose:
        print(f"report: {report_md}")
    return {"report": report, "report_json": report_json, "report_md": report_md, "report_dir": report_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nights", type=int, default=10)
    parser.add_argument("--model-seeds", type=int, default=30)
    parser.add_argument("--mdn-checkpoint", type=Path, default=DEFAULT_MDN_CHECKPOINT)
    args = parser.parse_args()
    run_comparison(nights=args.nights, model_seeds=args.model_seeds, mdn_checkpoint=args.mdn_checkpoint)
