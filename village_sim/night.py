"""
One village night as a chain of certified decisions.

At each decision point the agent observes the world state x and:

1. takes the motive weights for x from a weight source: the scenario schedule
   (certified over the full simplex), or a trained MDN that also gives a
   region of plausible weights W_x to certify over;
2. evaluates every option that can start at x with paired rollouts in a
   separate simulator started from a copy of x, so the world itself is never
   used for testing;
3. admits options with the confidence-bounded CDS / PDS gate and places their
   certificates in a skill library that holds only the options admitted at x;
4. selects the admitted option with the highest delta_r + w . delta_n, or
   waits in protection for one step when none beats the baseline (with
   ``gated=False`` every assessed option competes, certified or not, which
   shows what the gates prevent);
5. runs the chosen option in the world until it terminates.

Options are judged over the rest of the night (option, then the baseline until
sunrise). The night always runs to sunrise, after delivery too, unless the
player dies or the contract fails first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable

import numpy as np

from certification.certificate_schema import Certificate
from certification.cvar_test import CVaRGate
from certification.statistical_gate import gate_statistics
from env.skill_executor import SkillExecutor
from library.skill_library import SkillLibrary
from schemas.minecraft_objective_schema import VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_selection import select_best_candidate
from utils.support_geometry import make_basis_query_directions
from village_sim.contexts import partition_of
from village_sim.env import VillageEnv
from village_sim.motives import phi
from village_sim.options import CANDIDATE_OPTIONS, PROTECTED_WAIT, VillageOption
from village_sim.rollouts import evaluate_option, return_scale, steps_to_sunrise
from village_sim.state import VillageState
from village_sim.weights import ScenarioWeights, WeightView

WeightModel = Callable[[VillageState], WeightView]
CVAR_LEVEL = 0.1


@dataclass(frozen=True)
class Assessment:
    """How one option fared at a decision point."""

    option_id: str
    gate: str | None
    mean_delta_r: float
    mean_delta_n: tuple[float, ...]
    lower_bound: float
    mean_option_steps: float
    score: float
    cvar: float | None = None


@dataclass(frozen=True)
class Decision:
    """One decision and what happened when its option ran in the world."""

    index: int
    world_tick: int
    partition: str
    weight_source: str
    weights: tuple[float, ...]
    support_values: tuple[float, ...] | None
    assessments: tuple[Assessment, ...]
    chosen: str
    score: float | None
    steps: int
    payoff: float
    events: tuple[str, ...]
    phi_before: tuple[float, ...]
    phi_after: tuple[float, ...]


@dataclass
class NightResult:
    """The full trace of one night."""

    seed: int
    decisions: list[Decision]
    final_state: VillageState
    certificates: list[Certificate] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)

    @property
    def task_status(self) -> str:
        return self.final_state.task_status

    @property
    def casualties(self) -> int:
        return len(self.final_state.villager_health) - self.final_state.villagers_alive

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "task_status": self.task_status,
            "casualties": self.casualties,
            "final_world_tick": self.final_state.world_tick,
            "final_reputation": self.final_state.reputation,
            "final_phi": [float(v) for v in phi(self.final_state)],
            "decisions": [asdict(d) for d in self.decisions],
            "certificate_ids": [c.skill_id for c in self.certificates],
            "evidence_pairs": len(self.evidence),
        }


def add_certified(library: SkillLibrary, cert: Certificate, policy: Callable) -> bool:
    """Add ``cert`` to ``library`` with the weight region it was certified over."""
    return library.add_skill(
        cert.skill_id,
        cert,
        policy,
        weight_region_type=cert.weight_region_type,
        certification_context=cert.certification_context,
        mdn_alpha=cert.mdn_alpha,
        wx_support_directions=cert.wx_support_directions,
        wx_support_values=cert.wx_support_values,
    )


def _best_assessed(assessments: list[Assessment]) -> tuple[str | None, float | None]:
    """Highest-scoring option among all assessed ones, certified or not."""
    if not assessments:
        return None, None
    best = sorted(assessments, key=lambda a: (-a.score, a.option_id))[0]
    return best.option_id, best.score


class _EventLog:
    """Pass-through env wrapper that keeps the events and payoffs of each step."""

    def __init__(self, env: VillageEnv) -> None:
        self.env = env
        self.events: list[str] = []
        self.payoff = 0.0

    @property
    def metadata(self) -> dict:
        return self.env.metadata

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        out = self.env.step(action)
        self.events.extend(out[4].get("events", ()))
        self.payoff += float(out[4]["task_payoff"])
        return out

    def close(self) -> None:
        self.env.close()


class NightRunner:
    """Runs village nights with gated, weight-driven option selection."""

    def __init__(
        self,
        weight_model: WeightModel | None = None,
        options: Iterable[VillageOption] = CANDIDATE_OPTIONS,
        model_seeds: int = 30,
        gamma: float = 0.99,
        epsilon: float = 0.10,
        confidence: float = 0.95,
        max_decisions: int = 500,
        gated: bool = True,
        wait_steps: int = 5,
    ) -> None:
        if model_seeds < 2:
            raise ValueError("model_seeds must be at least 2 for a confidence bound")
        self.weight_model = weight_model or ScenarioWeights()
        self.options = tuple(options)
        self.model_seeds = tuple(range(model_seeds))
        self.gamma = gamma
        self.epsilon = epsilon
        self.confidence = confidence
        self.max_decisions = max_decisions
        self.gated = gated
        self.wait_steps = wait_steps
        self.model = VillageEnv()

    def assess(
        self,
        state: VillageState,
        view: WeightView,
        label: str,
        created: str,
        cvar_seed: int = 0,
    ) -> tuple[list[Assessment], list[Certificate], list[dict]]:
        """Gate every startable option at ``state``; certify the admitted ones.

        Options are certified over ``view.support_values`` when the weight source
        gives a region, otherwise over the full simplex. With Dirichlet
        parameters available, each admitted option also gets a seeded CVaR of its
        score under w ~ Dirichlet(alpha) for the audit.

        Returns the assessments, the certificates of the admitted options, and
        every paired rollout behind them, tagged with the certificate skill id.
        """
        schema = VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA
        horizon = steps_to_sunrise(state, self.model.mission)
        region = view.support_values
        basis = make_basis_query_directions(len(view.weights))
        assessments: list[Assessment] = []
        certificates: list[Certificate] = []
        evidence: list[dict] = []
        for option in self.options:
            if not option.can_start(state, self.model.mission):
                continue
            skill_id = f"{option.option_id}@{label}"
            pairs = evaluate_option(self.model, option, state, self.model_seeds, self.gamma)
            evidence.extend({"skill_id": skill_id, "context_id": label, **pair.to_dict()} for pair in pairs)
            stats = gate_statistics([p.delta_r for p in pairs], [p.delta_n for p in pairs], self.confidence)
            gate = stats.admitting_gate(self.epsilon, region)
            bound = stats.lower_bound_over(region)
            score = float(stats.mean_delta_r + np.dot(view.weights, stats.mean_delta_n))
            cvar = None
            if gate is not None:
                epsilon = self.epsilon if gate == "PDS" else 0.0
                region_fields = {}
                if region is not None:
                    region_fields = {
                        "weight_region_type": "MDN_WX",
                        "certification_context": tuple(float(v) for v in view.context),
                        "mdn_alpha": tuple(float(v) for v in view.alpha),
                        "wx_support_directions": tuple(tuple(float(v) for v in row) for row in basis),
                        "wx_support_values": tuple(float(v) for v in region),
                    }
                if view.alpha is not None:
                    cvar = CVaRGate(confidence=CVAR_LEVEL, seed=cvar_seed).get_cvar(
                        stats.mean_delta_r, np.asarray(stats.mean_delta_n), view.alpha
                    )
                certificates.append(Certificate(
                    skill_id=skill_id,
                    gate_type=gate,
                    delta_r=stats.mean_delta_r,
                    delta_n=stats.mean_delta_n,
                    admission_margin=bound + epsilon,
                    epsilon=epsilon,
                    timestamp=created,
                    seed=self.model_seeds[0],
                    gamma=self.gamma,
                    baseline_id=PROTECTED_WAIT.option_id,
                    environment=self.model.metadata["environment_id"],
                    episode_length=horizon,
                    version="1.0",
                    domain_id=schema.domain_id,
                    motive_schema_version=schema.motive_schema_version,
                    motive_names=schema.motive_names,
                    context_id=label,
                    n_rollouts=stats.n,
                    rollout_seeds=self.model_seeds,
                    confidence_level=stats.confidence,
                    lower_bound=bound,
                    return_scale=return_scale(self.gamma, horizon),
                    evidence_ref=f"decision-evidence#{label}",
                    **region_fields,
                ))
            assessments.append(Assessment(
                option_id=option.option_id,
                gate=gate,
                mean_delta_r=stats.mean_delta_r,
                mean_delta_n=stats.mean_delta_n,
                lower_bound=bound,
                mean_option_steps=float(np.mean([p.candidate.option_steps for p in pairs])),
                score=score,
                cvar=cvar,
            ))
        return assessments, certificates, evidence

    def select(self, certificates: list[Certificate], view: WeightView) -> tuple[str | None, float | None]:
        """The admitted option with the highest score, through the skill library."""
        lookup = {option.option_id: option for option in self.options}
        library = SkillLibrary(schema=VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA)
        for cert in certificates:
            if not add_certified(library, cert, lookup[cert.skill_id.split("@")[0]].new_run):
                raise RuntimeError(f"the skill library refused certificate {cert.skill_id!r} that the gate admitted")
        region = view.support_values
        admissible = library.query_admissible(
            view.weights,
            support_directions=None if region is None else make_basis_query_directions(len(view.weights)),
            support_values=region,
            schema=VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA,
        )
        if not admissible:
            return None, None
        records = [
            CandidateSkillRecord(
                skill_id=entry.certificate.skill_id,
                delta_r=entry.certificate.delta_r,
                delta_n=entry.certificate.delta_n,
                is_certified=True,
                gate_type=entry.certificate.gate_type,
            )
            for entry in admissible
        ]
        skill_id, score = select_best_candidate(records, view.weights)
        return skill_id.split("@")[0], score

    def run(self, seed: int, start: VillageState | None = None) -> NightResult:
        """Run one night in a world seeded with ``seed``."""
        world = _EventLog(VillageEnv())
        state, _ = world.reset(seed=seed, options={"state": start} if start is not None else None)
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        decisions: list[Decision] = []
        audit: list[Certificate] = []
        evidence: list[dict] = []
        lookup = {option.option_id: option for option in self.options}

        mission = self.model.mission
        for index in range(self.max_decisions):
            if state.night_over(mission):
                break
            label = f"night-{seed}-tick-{state.world_tick}"
            view = self.weight_model(state)
            assessments, certificates, pairs = self.assess(state, view, label, created, cvar_seed=index)
            audit.extend(certificates)
            evidence.extend(pairs)
            if self.gated:
                chosen_id, score = self.select(certificates, view)
            else:
                chosen_id, score = _best_assessed(assessments)
            situation = partition_of(state)
            if chosen_id is None or score <= 0.0:
                chosen, chosen_id = PROTECTED_WAIT, PROTECTED_WAIT.option_id
            else:
                chosen = lookup[chosen_id]
            run = chosen.new_run(state, mission)
            if chosen is PROTECTED_WAIT:
                max_steps = self.wait_steps

                def run_ends(obs, info, situation=situation):
                    return partition_of(obs) != situation
            else:
                max_steps = chosen.max_steps

                def run_ends(obs, info, run=run):
                    return run.done(obs)

            world.events, world.payoff = [], 0.0
            executor = SkillExecutor(
                env=world,
                policy_fn=run.act,
                gamma=self.gamma,
                max_steps=max_steps,
                stop_fn=run_ends,
                strict=True,
                verbose=False,
            )
            before = state
            result = executor.run_episode(initial_obs=state)
            state = world.env.state.copy()
            decisions.append(Decision(
                index=index,
                world_tick=before.world_tick,
                partition=situation,
                weight_source=view.source,
                weights=tuple(float(w) for w in view.weights),
                support_values=None if view.support_values is None else tuple(float(v) for v in view.support_values),
                assessments=tuple(assessments),
                chosen=chosen_id,
                score=score,
                steps=result.steps,
                payoff=world.payoff,
                events=tuple(world.events),
                phi_before=tuple(float(v) for v in phi(before)),
                phi_after=tuple(float(v) for v in phi(state)),
            ))

        return NightResult(seed=seed, decisions=decisions, final_state=state, certificates=audit, evidence=evidence)
