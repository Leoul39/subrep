"""
Read the village demo artifacts and shape them for presentation.

Everything here is read-only: the certification pipeline, the night report and
the MDN training script write the artifacts; these helpers only load and
reshape them for the Streamlit page.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from village_sim.motives import MOTIVE_NAMES

DEFAULT_DATA_DIR = Path("data/village")
DEFAULT_REPORT_DIR = Path("demo/artifacts/village")
DEFAULT_MDN_MANIFEST = Path("models/village_mdn.json")


def _read_json(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_village_artifacts(
    data_dir: Path = DEFAULT_DATA_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    mdn_manifest: Path = DEFAULT_MDN_MANIFEST,
) -> dict:
    """Load whichever village artifacts exist; missing ones are None."""
    data_dir, report_dir = Path(data_dir), Path(report_dir)
    nights = {
        path.stem.removeprefix("night_"): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(report_dir.glob("night_*.json"))
        if path.stem != "night_report"
    }
    return {
        "manifest": _read_json(data_dir / "manifest.json"),
        "admission_report": _read_json(report_dir / "admission_report.json"),
        "night_report": _read_json(report_dir / "night_report.json"),
        "nights": nights,
        "mdn": _read_json(Path(mdn_manifest)),
    }


def certification_matrix(admission_report: dict) -> list[dict]:
    """One row per option, one cell per partition: the gate or 'rejected', with the lower bound."""
    rows: dict[str, dict] = {}
    for entry in admission_report.get("audit_entries", []):
        option_id, partition = entry["skill_id"].split("@", 1)
        evaluations = entry.get("gate_evaluations") or []
        bound = evaluations[0]["lhs"] if evaluations else None
        result = entry["gate_type"] if entry["admitted"] else "rejected"
        rows.setdefault(option_id, {"option": option_id})[partition] = (
            result if bound is None else f"{result} ({bound:+.3f})"
        )
    return list(rows.values())


def policy_comparison(night_report: dict) -> list[dict]:
    """One row per scenario and policy with outcomes at sunrise."""
    rows = []
    for scenario, policies in night_report.get("scenarios", {}).items():
        for policy, summary in policies.items():
            rows.append({
                "scenario": scenario,
                "policy": policy,
                "delivered": summary["delivered_rate"],
                "casualties": summary["mean_casualties"],
                "raids": ", ".join(f"{k} {v}" for k, v in summary["raid_outcomes"].items()),
                "reputation": summary["mean_final_reputation"],
                **summary["mean_final_phi"],
            })
    return rows


def night_timeline(night: dict) -> list[dict]:
    """The decisions that did something: options run, or waits during which events happened."""
    rows = []
    for decision in night["decisions"]:
        if decision["chosen"] == "protected_wait_v1" and not decision["assessments"] and not decision["events"]:
            continue
        admitted = [a["option_id"].removesuffix("_v1") for a in decision["assessments"] if a["gate"]]
        rows.append({
            "tick": decision["world_tick"],
            "situation": decision["partition"],
            "weights from": decision.get("weight_source", "scenario"),
            "admitted": ", ".join(admitted) or "none",
            "chosen": decision["chosen"].removesuffix("_v1"),
            "score": decision["score"],
            "steps": decision["steps"],
            "events": ", ".join(decision["events"]),
        })
    return rows


def motive_trajectory(night: dict) -> list[dict]:
    """The six motives after every decision, keyed by world tick."""
    return [
        {"tick": decision["world_tick"], **dict(zip(MOTIVE_NAMES, decision["phi_after"]))}
        for decision in night["decisions"]
    ]
