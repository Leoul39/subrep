"""
Certification partitions and context sampling for the village night.

Certificates are issued per option and partition. A partition is a function of
the observable state, so the runtime can tell which certificates apply:

    raid          a raid is under way
    dusk          before hostile mobs start spawning
    night_attack  night, hostile mobs inside the village
    night_calm    night, no hostile mobs

Contexts for an option are drawn from the partition and kept only when the
option can start there, so evidence always comes from the option's own
initiation set.
"""

from __future__ import annotations

import random

from village_sim.mission import VillageMission, VillageModelParams
from village_sim.options import VillageOption
from village_sim.state import VillageState, sample_start_state

PARTITIONS = ("dusk", "night_calm", "night_attack", "raid")


def partition_of(
    state: VillageState,
    params: VillageModelParams | None = None,
) -> str:
    """The certification partition a state belongs to."""
    params = params or VillageModelParams()
    if state.raid_active:
        return "raid"
    if state.world_tick < params.hostile_spawn_start_tick:
        return "dusk"
    return "night_attack" if state.hostiles > 0 else "night_calm"


def sample_partition_state(
    partition: str,
    rng: random.Random,
    mission: VillageMission | None = None,
    params: VillageModelParams | None = None,
) -> VillageState:
    """Draw a start state inside ``partition``."""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    mission = mission or VillageMission()
    params = params or VillageModelParams()
    state = sample_start_state(rng, mission)
    step = mission.ticks_per_step
    night_start = params.hostile_spawn_start_tick
    latest = mission.sunrise_tick - 20 * step

    if partition == "dusk":
        state.world_tick = mission.dusk_tick + step * rng.randrange((night_start - mission.dusk_tick) // step)
        state.hostiles = 0
    else:
        state.world_tick = night_start + step * rng.randrange((latest - night_start) // step)
        state.hostiles = 0
        if partition == "night_attack":
            state.hostiles = rng.randint(1, 5)
        elif partition == "raid":
            state.hostiles = rng.randint(0, 2)
            state.raid_active = True
            state.raid_waves_left = rng.randint(1, params.raid_waves)
            state.raiders = rng.randint(1, params.raiders_first_wave + params.raid_waves - 1)
    if rng.random() < 0.25:
        state.books = mission.books_required
    return state


def option_contexts(
    option: VillageOption,
    partition: str,
    count: int,
    seed: int,
    mission: VillageMission | None = None,
    params: VillageModelParams | None = None,
    max_draws: int = 200,
) -> list[VillageState]:
    """Up to ``count`` start states in ``partition`` where ``option`` can start.

    Deterministic for a given (seed, partition, option). Returns fewer than
    ``count`` states only when the initiation set is rarely or never met.
    """
    mission = mission or VillageMission()
    rng = random.Random(f"{seed}:{partition}:{option.option_id}")
    contexts: list[VillageState] = []
    for _ in range(max_draws):
        if len(contexts) == count:
            break
        state = sample_partition_state(partition, rng, mission, params)
        if option.can_start(state, mission):
            contexts.append(state)
    return contexts
