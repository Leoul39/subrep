"""
Motive features phi(x) and model context psi(x) for the village night.

phi(x) has the six canonical coordinates, each bounded in [0, 1] with larger
meaning better. psi(x) is the numeric context the MDN reads: phi(x), time of
night, threat indicators, inventory, recent outcomes and the agent's location.
"""

from __future__ import annotations

import numpy as np

from village_sim.mission import LOCATIONS, VillageMission, VillageModelParams
from village_sim.state import TASK_DELIVERED, VillageState

MOTIVE_NAMES = ["Safety", "Reputation", "DeadlineSlack",
                "InventoryValue", "Sustainability", "Infrastructure"]

PSI_NAMES = (
    *(f"phi_{name}" for name in MOTIVE_NAMES),
    "night_elapsed",
    "hostiles", "raiders", "raid_active", "raid_waves_left",
    "emeralds", "books", "torches", "charcoal", "coal", "iron_blocks", "food_items", "player_food",
    "casualty_fraction", "raid_won", "raid_lost", "contract_valid",
    *(f"at_{location}" for location in LOCATIONS),
)


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def shelter(state: VillageState, params: VillageModelParams) -> float:
    """One minus the expected net hostile inflow per night step, as a share of
    the inflow into an unlit, unwalled village without golems."""
    inflow = params.hostile_spawn_rate * (1.0 - state.lit_fraction) * (1.0 - state.wall_fraction)
    net = max(0.0, inflow - state.golems * params.golem_kill_chance)
    return _clip01(1.0 - net / params.hostile_spawn_rate)


def safety(state: VillageState, mission: VillageMission, params: VillageModelParams) -> float:
    """Villager health and casualties, player health, armor, hostile proximity,
    raid intensity and threat exposure, equally weighted."""
    villagers = sum(state.villager_health) / (20.0 * mission.protected_villagers)
    player = state.player_health / 20.0
    armor = state.armor / 20.0
    hostiles = 1.0 - min(1.0, state.hostiles / mission.threat_cap)
    raid_pressure = state.raiders + params.raiders_first_wave * state.raid_waves_left if state.raid_active else 0
    raid = 1.0 - min(1.0, raid_pressure / mission.threat_cap)
    terms = (villagers, player, armor, hostiles, raid, shelter(state, params))
    return _clip01(sum(terms) / len(terms))


def reputation(state: VillageState) -> float:
    """Standing with the village; an invalidated contract has none."""
    return _clip01(state.reputation) if state.contract_valid else 0.0


def work_steps_left(state: VillageState, mission: VillageMission, params: VillageModelParams) -> float:
    """Steps still needed to buy the missing books, walk the route and deliver."""
    if state.task_status == TASK_DELIVERED:
        return 0.0
    steps = 0.0
    to_buy = state.books_still_needed(mission)
    if to_buy > 0:
        if state.location != "hall":
            steps += 1.0
        trade_steps = params.fast_trade_steps if state.fast_trading else params.trade_steps
        steps += max(0, to_buy * trade_steps - state.trade_progress)
    if state.location != "settlement":
        progress = state.route_progress if state.location == "route" else 0.0
        steps += mission.route_steps * (1.0 - progress)
    return steps + 1.0


def deadline_slack(state: VillageState, mission: VillageMission, params: VillageModelParams) -> float:
    """Night time left after the remaining work, as a share of the whole night."""
    ticks_left = max(0, mission.sunrise_tick - state.world_tick)
    work_ticks = work_steps_left(state, mission, params) * mission.ticks_per_step
    return _clip01((ticks_left - work_ticks) / (mission.sunrise_tick - mission.dusk_tick))


def inventory_value(state: VillageState, mission: VillageMission) -> float:
    """Contract value delivered, carried, or affordable at the current price."""
    affordable = state.emeralds // max(1, state.book_price) if state.can_trade(mission) else 0
    books = state.books_delivered + state.books + affordable
    return _clip01(books * mission.book_value / mission.contract_value)


def sustainability(state: VillageState, mission: VillageMission) -> float:
    """Depleting coal, iron and food reserves; renewable charcoal costs nothing."""
    coal = min(1.0, state.coal / mission.coal_capacity)
    iron = min(1.0, state.iron_blocks / mission.iron_block_capacity)
    food_capacity = 20.0 + mission.food_item_capacity * mission.food_item_value
    food = min(1.0, (state.player_food + state.food_items * mission.food_item_value) / food_capacity)
    return _clip01((coal + iron + food) / 3.0)


def infrastructure(state: VillageState, mission: VillageMission) -> float:
    """Lighting, walls and golems, halved while trading access is blocked."""
    golems = min(1.0, state.golems / mission.golem_target)
    defence = (state.lit_fraction + state.wall_fraction + golems) / 3.0
    return _clip01(defence if state.trading_access else 0.5 * defence)


def phi(
    state: VillageState,
    mission: VillageMission | None = None,
    params: VillageModelParams | None = None,
) -> np.ndarray:
    """The six motive features in canonical order."""
    mission = mission or VillageMission()
    params = params or VillageModelParams()
    return np.array([
        safety(state, mission, params),
        reputation(state),
        deadline_slack(state, mission, params),
        inventory_value(state, mission),
        sustainability(state, mission),
        infrastructure(state, mission),
    ], dtype=np.float32)


def psi(
    state: VillageState,
    mission: VillageMission | None = None,
    params: VillageModelParams | None = None,
) -> np.ndarray:
    """Numeric context for the MDN, ordered as ``PSI_NAMES``."""
    mission = mission or VillageMission()
    night = mission.sunrise_tick - mission.dusk_tick
    casualties = 1.0 - state.villagers_alive / mission.protected_villagers
    values = [
        *phi(state, mission, params),
        _clip01((state.world_tick - mission.dusk_tick) / night),
        min(1.0, state.hostiles / mission.threat_cap),
        min(1.0, state.raiders / mission.threat_cap),
        float(state.raid_active),
        state.raid_waves_left / max(1, (params or VillageModelParams()).raid_waves),
        min(1.0, state.emeralds / 64.0),
        min(1.0, state.books / mission.books_required),
        min(1.0, state.torches / 64.0),
        min(1.0, state.charcoal / mission.coal_capacity),
        min(1.0, state.coal / mission.coal_capacity),
        min(1.0, state.iron_blocks / mission.iron_block_capacity),
        min(1.0, state.food_items / mission.food_item_capacity),
        state.player_food / 20.0,
        casualties,
        float(state.raid_outcome == "won"),
        float(state.raid_outcome == "lost"),
        float(state.contract_valid),
        *(float(state.location == location) for location in LOCATIONS),
    ]
    return np.asarray(values, dtype=np.float32)
