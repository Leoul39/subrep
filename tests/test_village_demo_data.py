"""Tests for the helpers behind the village Streamlit page."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

pytest.importorskip("hyperon")

from demo.run_village_night import run_comparison
from demo.run_village_pipeline import run_pipeline
from utils.village_demo_data import (
    certification_matrix,
    load_village_artifacts,
    motive_trajectory,
    night_timeline,
    policy_comparison,
)
from village_sim.motives import MOTIVE_NAMES
from village_sim.options import TORCH_CORRIDOR, TRADE_MENU_HOTKEYS


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("village_demo")
    run_pipeline(
        root / "data", root / "report",
        partitions=("dusk", "night_calm"), options=(TORCH_CORRIDOR, TRADE_MENU_HOTKEYS), verbose=False,
    )
    run_comparison(nights=1, report_dir=root / "report", model_seeds=3, mdn_checkpoint=None, verbose=False)
    return load_village_artifacts(root / "data", root / "report", root / "missing_mdn.json")


def test_every_artifact_is_loaded_and_missing_ones_are_none(artifacts):
    assert artifacts["manifest"]["counts"]["evaluated"] == 4
    assert artifacts["admission_report"]["total_attempted"] == 4
    assert set(artifacts["nights"]) == {"calm_dusk", "raid_at_dusk"}
    assert artifacts["mdn"] is None


def test_the_certification_matrix_has_a_cell_per_option_and_situation(artifacts):
    rows = {row["option"]: row for row in certification_matrix(artifacts["admission_report"])}
    assert set(rows) == {"torch_corridor_v1", "trade_menu_hotkeys_v1"}
    for row in rows.values():
        for partition in ("dusk", "night_calm"):
            assert row[partition].split(" ")[0] in {"CDS", "PDS", "rejected"}


def test_policy_comparison_covers_every_scenario_and_policy(artifacts):
    rows = policy_comparison(artifacts["night_report"])
    assert len(rows) == 2 * 4
    assert all(set(MOTIVE_NAMES) <= set(row) for row in rows)


def test_timeline_and_trajectory_follow_the_night(artifacts):
    night = artifacts["nights"]["raid_at_dusk"]
    timeline = night_timeline(night)
    assert timeline and all("chosen" in row for row in timeline)
    trajectory = motive_trajectory(night)
    assert len(trajectory) == len(night["decisions"])
    assert [row["tick"] for row in trajectory] == sorted(row["tick"] for row in trajectory)


def test_the_streamlit_page_compiles():
    py_compile.compile(str(Path("demo/streamlit_village_demo.py")), doraise=True)
