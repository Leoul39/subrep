"""
Where the motive weights come from at a decision point.

``ScenarioWeights`` uses the reference scenario's weight schedule, a stand-in
for MetaMo, and certifies over the full simplex.

``MDNWeights`` uses a Motive Decomposition Network trained to reproduce that
schedule from the model context psi(x). It returns Dirichlet parameters alpha
(the weights are their mean) and support values s that define the region of
plausible weights W_x = {w in simplex : w_i <= s_i}; options are then certified
over W_x instead of the whole simplex. The training targets are:

    alpha*(x) = CONCENTRATION * w_bar(x)       spread of about 0.05 per weight
    s*(x)     = min(1, w_bar(x) + MOTIVE_SHIFT)  each weight may rise by 0.05

where w_bar(x) is the schedule for x's situation and 0.05 is the scenario's
bound on how far a weight moves between decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from generator.mdn import MotiveDecompositionNetwork
from village_sim.contexts import PARTITIONS, partition_of, sample_partition_state
from village_sim.mission import VillageMission, VillageModelParams
from village_sim.motives import MOTIVE_NAMES, PSI_NAMES, psi
from village_sim.options import CANDIDATE_OPTIONS
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
MOTIVE_SHIFT = 0.05
CONCENTRATION = 50.0
CHECKPOINT_VERSION = "1.0"


def scenario_weights(state: VillageState) -> np.ndarray:
    """Schedule weights for the situation ``state`` is in."""
    return np.asarray(SCENARIO_WEIGHTS[partition_of(state)], dtype=np.float64)


def scenario_region(state: VillageState) -> np.ndarray:
    """Support values of the plausible weights around the schedule."""
    return np.minimum(1.0, scenario_weights(state) + MOTIVE_SHIFT)


@dataclass(frozen=True)
class WeightView:
    """Weights for one decision, and the region they are certified over."""

    source: str
    weights: np.ndarray
    support_values: np.ndarray | None = None
    alpha: np.ndarray | None = None
    context: np.ndarray | None = None


class ScenarioWeights:
    """The schedule's weights, certified over the full simplex."""

    source = "scenario"

    def __call__(self, state: VillageState) -> WeightView:
        return WeightView(source=self.source, weights=scenario_weights(state))


class MDNWeights:
    """Weights and a contextual region W_x from a trained village MDN."""

    source = "mdn"

    def __init__(self, model: MotiveDecompositionNetwork, mission: VillageMission | None = None) -> None:
        if model.input_dim != len(PSI_NAMES) or model.num_objectives != len(MOTIVE_NAMES):
            raise ValueError(
                f"MDN has input_dim={model.input_dim}, num_objectives={model.num_objectives}; "
                f"the village needs {len(PSI_NAMES)} and {len(MOTIVE_NAMES)}"
            )
        self.model = model.eval()
        self.mission = mission or VillageMission()

    def __call__(self, state: VillageState) -> WeightView:
        context = psi(state, self.mission)
        with torch.no_grad():
            alpha, support = self.model.forward_inference(torch.tensor(context, dtype=torch.float32))
        alpha_np = alpha.numpy().astype(np.float64)
        return WeightView(
            source=self.source,
            weights=alpha_np / alpha_np.sum(),
            support_values=support.numpy().astype(np.float64),
            alpha=alpha_np,
            context=context.astype(np.float64),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "checkpoint_version": CHECKPOINT_VERSION,
            "psi_names": list(PSI_NAMES),
            "motive_names": list(MOTIVE_NAMES),
            "hidden_dim": self.model.hidden_dim,
            "num_skills": self.model.num_skills,
            "concentration": CONCENTRATION,
            "motive_shift": MOTIVE_SHIFT,
            "state_dict": self.model.state_dict(),
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MDNWeights":
        """Load a checkpoint, refusing one built for another context or motive schema."""
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported village MDN checkpoint version {checkpoint.get('checkpoint_version')!r}")
        if checkpoint.get("psi_names") != list(PSI_NAMES) or checkpoint.get("motive_names") != list(MOTIVE_NAMES):
            raise ValueError("village MDN checkpoint was trained for a different context or motive schema")
        model = MotiveDecompositionNetwork(
            input_dim=len(PSI_NAMES),
            num_objectives=len(MOTIVE_NAMES),
            hidden_dim=checkpoint["hidden_dim"],
            num_skills=checkpoint["num_skills"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model)


def sample_training_contexts(
    count_per_partition: int,
    seed: int,
    mission: VillageMission | None = None,
    params: VillageModelParams | None = None,
) -> list[VillageState]:
    """Contexts from every partition, reproducible from ``seed``."""
    import random

    rng = random.Random(seed)
    return [
        sample_partition_state(partition, rng, mission, params)
        for partition in PARTITIONS
        for _ in range(count_per_partition)
    ]


def _targets(states: Sequence[VillageState], mission: VillageMission) -> tuple[torch.Tensor, ...]:
    contexts = torch.tensor(np.stack([psi(s, mission) for s in states]), dtype=torch.float32)
    alpha = torch.tensor(np.stack([CONCENTRATION * scenario_weights(s) for s in states]), dtype=torch.float32)
    support = torch.tensor(np.stack([scenario_region(s) for s in states]), dtype=torch.float32)
    return contexts, alpha, support


def train_village_mdn(
    states: Sequence[VillageState],
    *,
    epochs: int = 2000,
    learning_rate: float = 3e-3,
    hidden_dim: int = 64,
    seed: int = 0,
    mission: VillageMission | None = None,
) -> tuple[MDNWeights, dict[str, float]]:
    """Fit the MDN's weight and support heads to the schedule targets."""
    mission = mission or VillageMission()
    torch.manual_seed(seed)
    model = MotiveDecompositionNetwork(
        input_dim=len(PSI_NAMES),
        num_objectives=len(MOTIVE_NAMES),
        hidden_dim=hidden_dim,
        num_skills=len(CANDIDATE_OPTIONS),
    )
    contexts, alpha_target, support_target = _targets(states, mission)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimiser.zero_grad()
        alpha, support = model.forward_inference(contexts)
        alpha_loss = torch.mean((torch.log(alpha) - torch.log(alpha_target)) ** 2)
        support_loss = torch.mean((support - support_target) ** 2)
        (alpha_loss + support_loss).backward()
        optimiser.step()
    return MDNWeights(model, mission), {
        "alpha_log_mse": float(alpha_loss.item()),
        "support_mse": float(support_loss.item()),
    }


def evaluate_village_mdn(weights: MDNWeights, states: Iterable[VillageState]) -> dict[str, float]:
    """Held-out agreement with the schedule and feasibility of every region."""
    weight_errors, support_errors, feasible = [], [], []
    for state in states:
        view = weights(state)
        weight_errors.append(np.abs(view.weights - scenario_weights(state)).max())
        support_errors.append(np.abs(view.support_values - scenario_region(state)).max())
        s = view.support_values
        feasible.append(bool(np.all(s >= 0.0) and np.all(s <= 1.0) and s.sum() >= 1.0 - 1e-9))
    return {
        "contexts": float(len(weight_errors)),
        "max_weight_error": float(np.max(weight_errors)),
        "mean_weight_error": float(np.mean(weight_errors)),
        "max_support_error": float(np.max(support_errors)),
        "feasible_region_rate": float(np.mean(feasible)),
    }


def load_weight_model(path: str | Path | None) -> tuple[ScenarioWeights | MDNWeights, str | None]:
    """The MDN at ``path``, or the scenario schedule when it cannot be used.

    Returns the weight model and, when falling back, the reason. A missing or
    incompatible checkpoint never stops a night; it only disables the MDN mode.
    """
    if path is None:
        return ScenarioWeights(), "no MDN checkpoint configured"
    try:
        return MDNWeights.load(path), None
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return ScenarioWeights(), f"MDN checkpoint unusable: {exc}"
