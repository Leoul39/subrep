"""
Paired option rollouts for the village night.

Candidate and baseline are compared over the same horizon from the same
restored state and seed: the rest of the night, H steps until sunrise. The
candidate runs its option until the option terminates (tau steps) and then
follows the baseline until sunrise; the baseline runs for all H steps. This is
a rollout of the option against the baseline policy, so choosing the best
option this way does at least as well as the baseline. Both sides go through
the shared ``SkillExecutor``. A rollout that ends early (the player dies or the
contract fails) keeps its final state for the remaining steps, so both returns
always cover H steps:

    r_hat = sum_{t<H} gamma^t r_t
    n_hat = return_scale * sum_{t<H} gamma^t phi(x_{t+1})

``return_scale = 1 / sum_{t<H} gamma^t`` turns the motive return into a
discounted average in [0, 1] per coordinate, so motive deltas from different
start times and the PDS budget share one declared scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from env.skill_executor import SkillExecutor
from village_sim.env import VillageEnv
from village_sim.mission import VillageMission
from village_sim.options import PROTECTED_WAIT, VillageOption
from village_sim.state import VillageState


def steps_to_sunrise(state: VillageState, mission: VillageMission) -> int:
    """Evaluation horizon from ``state``: whole steps left in the night, at least one."""
    ticks_left = mission.sunrise_tick - state.world_tick
    return max(1, -(-ticks_left // mission.ticks_per_step))


def return_scale(gamma: float, horizon: int) -> float:
    """Factor that turns an H-step discounted motive sum into a discounted average."""
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    return 1.0 / sum(gamma ** t for t in range(horizon))


@dataclass(frozen=True)
class Rollout:
    """One H-step rollout: an option followed by the baseline."""

    payoff: float
    motives: tuple[float, ...]
    option_steps: int
    env_steps: int
    stop_reason: str
    task_status: str


@dataclass(frozen=True)
class PairedOutcome:
    """One candidate/baseline pair from the same state and seed."""

    option_id: str
    baseline_id: str
    seed: int
    horizon: int
    gamma: float
    candidate: Rollout
    baseline: Rollout

    @property
    def delta_r(self) -> float:
        return self.candidate.payoff - self.baseline.payoff

    @property
    def delta_n(self) -> np.ndarray:
        return np.asarray(self.candidate.motives) - np.asarray(self.baseline.motives)

    def to_dict(self) -> dict:
        def side(rollout: Rollout) -> dict:
            return {
                "payoff": rollout.payoff,
                "motives": list(rollout.motives),
                "option_steps": rollout.option_steps,
                "env_steps": rollout.env_steps,
                "stop_reason": rollout.stop_reason,
                "task_status": rollout.task_status,
            }

        return {
            "option_id": self.option_id,
            "baseline_id": self.baseline_id,
            "seed": self.seed,
            "horizon": self.horizon,
            "gamma": self.gamma,
            "return_scale": return_scale(self.gamma, self.horizon),
            "candidate": side(self.candidate),
            "baseline": side(self.baseline),
            "delta": {"payoff": self.delta_r, "motives": [float(v) for v in self.delta_n]},
        }


class _OptionThenBaseline:
    """Policy that follows an option until it terminates, then the baseline.

    The option's termination is checked on the state each step leads to, so
    the switch happens on the step after the option's last action.
    """

    def __init__(
        self,
        option: VillageOption,
        start: VillageState,
        env: VillageEnv,
        baseline: VillageOption,
    ) -> None:
        self.option = option
        self.run = option.new_run(start, env.mission)
        self.follow = baseline.new_run(start, env.mission)
        self.steps = 0
        self.option_steps: int | None = None
        self.stop_reason = "horizon"

    def _finish(self, state: VillageState) -> None:
        if self.steps >= self.option.max_steps:
            self.option_steps, self.stop_reason = self.steps, "step_limit"
        elif self.run.done(state):
            self.option_steps, self.stop_reason = self.steps, "done"

    def __call__(self, state: VillageState) -> str:
        if self.option_steps is None and self.steps > 0:
            self._finish(state)
        self.steps += 1
        return self.run.act(state) if self.option_steps is None else self.follow.act(state)


def run_option(
    env: VillageEnv,
    option: VillageOption,
    start: VillageState,
    seed: int,
    gamma: float,
    horizon: int | None = None,
    baseline: VillageOption = PROTECTED_WAIT,
) -> Rollout:
    """Run ``option`` from ``start`` with ``seed``, then ``baseline`` until
    ``horizon`` steps have passed (by default, until sunrise).

    The whole horizon is one executor pass, so candidate and baseline returns
    are summed the same way and identical motive sequences give identical sums.
    """
    if horizon is None:
        horizon = steps_to_sunrise(start, env.mission)
    policy = _OptionThenBaseline(option, start, env, baseline)
    executor = SkillExecutor(
        env=env,
        policy_fn=policy,
        gamma=gamma,
        max_steps=horizon,
        strict=True,
        verbose=False,
    )
    result = executor.run_episode(seed=seed, options={"state": start})
    if policy.option_steps is None:
        if result.terminated:
            policy.option_steps, policy.stop_reason = result.steps, "env_terminated"
        else:
            policy._finish(env.state)
            if policy.option_steps is None:
                policy.option_steps = result.steps

    motives = np.asarray(result.motive_deltas, dtype=np.float64)
    last = np.asarray(executor.last_run_info["final_reward"], dtype=np.float64)
    for t in range(result.steps, horizon):
        motives = motives + gamma ** t * last

    return Rollout(
        payoff=float(result.total_payoff),
        motives=tuple(float(v) for v in motives * return_scale(gamma, horizon)),
        option_steps=policy.option_steps,
        env_steps=result.steps,
        stop_reason=policy.stop_reason,
        task_status=env.state.task_status,
    )


def paired_rollout(
    env: VillageEnv,
    option: VillageOption,
    start: VillageState,
    seed: int,
    gamma: float,
    baseline: VillageOption = PROTECTED_WAIT,
    horizon: int | None = None,
) -> PairedOutcome:
    """Candidate and baseline from the same state and seed over the same horizon."""
    if horizon is None:
        horizon = steps_to_sunrise(start, env.mission)
    return PairedOutcome(
        option_id=option.option_id,
        baseline_id=baseline.option_id,
        seed=int(seed),
        horizon=horizon,
        gamma=float(gamma),
        candidate=run_option(env, option, start, seed, gamma, horizon, baseline),
        baseline=run_option(env, baseline, start, seed, gamma, horizon, baseline),
    )


def evaluate_option(
    env: VillageEnv,
    option: VillageOption,
    start: VillageState,
    seeds: Iterable[int],
    gamma: float,
    baseline: VillageOption = PROTECTED_WAIT,
    horizon: int | None = None,
) -> list[PairedOutcome]:
    """Paired rollouts of ``option`` from ``start`` for every seed."""
    return [paired_rollout(env, option, start, seed, gamma, baseline, horizon) for seed in seeds]
