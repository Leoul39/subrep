"""Streamlit page for the Minecraft Village Defense & Trade demo.

Run from the repository root after producing the artifacts:

    python -m demo.train_village_mdn
    python -m demo.run_village_pipeline
    python -m demo.run_village_night
    streamlit run demo/streamlit_village_demo.py

The page only reads saved artifacts; it never certifies or runs a night itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.village_demo_data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_MDN_MANIFEST,
    DEFAULT_REPORT_DIR,
    certification_matrix,
    load_village_artifacts,
    motive_trajectory,
    night_timeline,
    policy_comparison,
)
from village_sim.motives import MOTIVE_NAMES  # noqa: E402
from village_sim.options import CANDIDATE_OPTIONS  # noqa: E402

COMMANDS = (
    "python -m demo.train_village_mdn\n"
    "python -m demo.run_village_pipeline\n"
    "python -m demo.run_village_night"
)


def main() -> None:
    st.set_page_config(page_title="SubRep Minecraft Demo", page_icon="SR", layout="wide")
    with st.sidebar:
        st.title("SubRep · Minecraft")
        st.write("**Artifacts**")
        st.code("\n".join(str(p) for p in (DEFAULT_DATA_DIR, DEFAULT_REPORT_DIR, DEFAULT_MDN_MANIFEST)))
        st.write("**Regenerate**")
        st.code(COMMANDS, language="bash")
        if st.button("Reload artifacts"):
            st.cache_data.clear()
            st.rerun()

    artifacts = _artifacts()
    st.title("Village Defense & Trade")
    st.write(
        "Deliver 24 emeralds' worth of enchanted books before sunrise while keeping villagers alive, "
        "reputation up and resources in hand. At every decision SubRep tests each option against a "
        "protected wait from the current state, certifies it with the CDS / PDS gates, and runs the "
        "best certified option for the current motive weights."
    )
    _scenario()
    _certification(artifacts)
    _nights(artifacts)
    _mdn(artifacts)
    _provenance(artifacts)


@st.cache_data(show_spinner=False)
def _artifacts() -> dict:
    return load_village_artifacts()


def _scenario() -> None:
    st.header("1. Motives and options")
    left, right = st.columns(2)
    left.write("**Six motives** (each in [0, 1], larger is better)")
    left.write(", ".join(MOTIVE_NAMES))
    right.write("**Options**")
    right.write(", ".join(f"{o.option_id.removesuffix('_v1')} ({o.group})" for o in CANDIDATE_OPTIONS))


def _certification(artifacts: dict) -> None:
    st.header("2. Certification by situation")
    report = artifacts["admission_report"]
    if report is None:
        st.info("No certification report yet. Run `python -m demo.run_village_pipeline`.")
        return
    a, b, c = st.columns(3)
    a.metric("Admitted (CDS)", report["cds_pass_count"])
    b.metric("Admitted (PDS)", report["pds_pass_count"])
    c.metric("Rejected", report["rejected"])
    st.caption("Each cell: the admitting gate, or 'rejected', with the 95% lower bound of "
               "delta_r + min(delta_n) over 30 paired rollouts.")
    st.dataframe(certification_matrix(report), hide_index=True, width="stretch")


def _nights(artifacts: dict) -> None:
    st.header("3. Nights, judged at sunrise")
    night_report = artifacts["night_report"]
    if night_report is None:
        st.info("No night report yet. Run `python -m demo.run_village_night`.")
        return
    if night_report.get("mdn_fallback"):
        st.warning(f"subrep_mdn used the scenario schedule: {night_report['mdn_fallback']}")
    st.dataframe(policy_comparison(night_report), hide_index=True, width="stretch")

    if artifacts["nights"]:
        st.subheader("One SubRep night, decision by decision")
        scenario = st.selectbox("Scenario", sorted(artifacts["nights"]))
        night = artifacts["nights"][scenario]
        st.write(f"Outcome: **{night['task_status']}**, {night['casualties']} casualties, "
                 f"{len(night['certificate_ids'])} runtime certificates.")
        st.dataframe(night_timeline(night), hide_index=True, width="stretch")
        st.line_chart(motive_trajectory(night), x="tick", y=list(MOTIVE_NAMES))


def _mdn(artifacts: dict) -> None:
    st.header("4. Weight model")
    mdn = artifacts["mdn"]
    if mdn is None:
        st.info("No trained MDN. Run `python -m demo.train_village_mdn`; nights then use the scenario schedule.")
        return
    held_out = mdn["held_out"]
    a, b, c = st.columns(3)
    a.metric("Mean weight error (held-out)", f"{held_out['mean_weight_error']:.3f}")
    b.metric("Max weight error (held-out)", f"{held_out['max_weight_error']:.3f}")
    c.metric("Feasible regions", f"{held_out['feasible_region_rate']:.0%}")
    st.write("Thresholds " + ("passed" if mdn["passed"] else "**not passed**") + f": {mdn['thresholds']}")


def _provenance(artifacts: dict) -> None:
    st.header("5. Reproducible artifacts")
    manifest = artifacts["manifest"]
    if manifest is None:
        st.info("No certification manifest yet.")
        return
    st.write(f"Run `{manifest['run_id']}` · created {manifest['created']} · "
             f"{manifest['counts']['pairs']} paired rollouts")
    st.json({"config": manifest["config"], "files": manifest["files"]}, expanded=False)


if __name__ == "__main__":
    main()
