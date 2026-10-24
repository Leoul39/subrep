"""
One village night as a chain of certified decisions.

At each decision point the agent observes the world state x and:

1. takes the motive weights for x's situation from a weight schedule;
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
from certification.metta_storage import CertificateStore
from certification.statistical_gate import gate_statistics
from env.skill_executor import SkillExecutor
from library.skill_library import SkillLibrary
from schemas.minecraft_objective_schema import VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_selection import select_best_candidate
from village_sim.contexts import partition_of
from village_sim.env import VillageEnv
from village_sim.motives import phi
from village_sim.options import CANDIDATE_OPTIONS, PROTECTED_WAIT, VillageOption
from village_sim.rollouts import evaluate_option, return_scale, steps_to_sunrise
from village_sim.state import VillageState

# The reference scenario's weight schedule: moderate safety priority at dusk,
# a bounded shift toward safety while threats are present, and a shift toward
# trade once the village is calm again.
SCENARIO_WEIGHTS = {
    "dusk": (0.35, 0.15, 0.20, 0.20, 0.05, 0.05),
    "night_attack": (0.38, 0.17, 0.18, 0.17, 0.05, 0.05),
    "raid": (0.38, 0.17, 0.18, 0.17, 0.05, 0.05),
    "night_calm": (0.25, 0.25, 0.20, 0.20, 0.05, 0.05),
}

WeightFn = Callable[[VillageState], np.ndarray]


def scenario_weights(state: VillageState) -> np.ndarray:
    """Weights of the reference scenario for the situation ``state`` is in."""
    return np.asarray(SCENARIO_WEIGHTS[partition_of(state)], dtype=np.float64)


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


@dataclass(frozen=True)
class Decision:
    """One decision and what happened when its option ran in the world."""

    index: int
    world_tick: int
    partition: str
    weights: tuple[float, ...]
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
        weight_fn: WeightFn = scenario_weights,
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
        self.weight_fn = weight_fn
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
        weights: np.ndarray,
        label: str,
        created: str,
    ) -> tuple[list[Assessment], list[Certificate], list[dict]]:
        """Gate every startable option at ``state``; certify the admitted ones.

        Returns the assessments, the certificates of the admitted options, and
        every paired rollout behind them, tagged with the certificate skill id.
        """
        schema = VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA
        horizon = steps_to_sunrise(state, self.model.mission)
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
            gate = stats.admitting_gate(self.epsilon)
            score = float(stats.mean_delta_r + np.dot(weights, stats.mean_delta_n))
            if gate is not None:
                epsilon = self.epsilon if gate == "PDS" else 0.0
                certificates.append(Certificate(
                    skill_id=skill_id,
                    gate_type=gate,
                    delta_r=stats.mean_delta_r,
                    delta_n=stats.mean_delta_n,
                    admission_margin=stats.lower_bound + epsilon,
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
                    lower_bound=stats.lower_bound,
                    return_scale=return_scale(self.gamma, horizon),
                    evidence_ref=f"decision-evidence#{label}",
                ))
            assessments.append(Assessment(
                option_id=option.option_id,
                gate=gate,
                mean_delta_r=stats.mean_delta_r,
                mean_delta_n=stats.mean_delta_n,
                lower_bound=stats.lower_bound,
                mean_option_steps=float(np.mean([p.candidate.option_steps for p in pairs])),
                score=score,
            ))
        return assessments, certificates, evidence

    def select(self, certificates: list[Certificate], weights: np.ndarray) -> tuple[str | None, float | None]:
        """The admitted option with the highest score, through the skill library."""
        lookup = {option.option_id: option for option in self.options}
        library = SkillLibrary(schema=VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA)
        for cert in certificates:
            library.add_skill(cert.skill_id, cert, policy=lookup[cert.skill_id.split("@")[0]].new_run)
        admissible = library.query_admissible(weights, schema=VILLAGE_SYNTHETIC_OBJECTIVE_SCHEMA)
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
        skill_id, score = select_best_candidate(records, weights)
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
            weights = self.weight_fn(state)
            assessments, certificates, pairs = self.assess(state, weights, label, created)
            audit.extend(certificates)
            evidence.extend(pairs)
            if self.gated:
                chosen_id, score = self.select(certificates, weights)
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
                weights=tuple(float(w) for w in weights),
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
