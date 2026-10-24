"""
Village Defense & Trade mission and model parameters.

``VillageMission`` freezes the scenario: the night window, the delivery
contract, the protected villagers and the capacities used to normalise motive
features. ``VillageModelParams`` holds the simulator's dynamics constants
(spawn rates, damage, action durations). They are kept apart because the
mission is shared with the real-server adapter, while the parameters describe
only the simulator and are meant to be fitted against real-server rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

LOCATIONS = ("hall", "perimeter", "route", "settlement")


@dataclass(frozen=True)
class VillageMission:
    """Fixed scenario specification for one Village Defense & Trade night."""

    mission_id: str = "village_defense_trade_v1"
    dusk_tick: int = 12000
    sunrise_tick: int = 23000
    ticks_per_step: int = 100
    contract_value: float = 24.0
    book_value: float = 8.0
    protected_villagers: int = 4
    librarians: int = 2
    base_book_price: int = 7
    route_steps: int = 6
    sprint_route_steps: int = 4
    coal_capacity: int = 8
    iron_block_capacity: int = 8
    food_item_value: float = 6.0
    food_item_capacity: int = 4
    golem_target: int = 2
    threat_cap: float = 10.0

    @property
    def night_steps(self) -> int:
        return (self.sunrise_tick - self.dusk_tick) // self.ticks_per_step

    @property
    def books_required(self) -> int:
        return int(-(-self.contract_value // self.book_value))


@dataclass(frozen=True)
class VillageModelParams:
    """Simulator dynamics constants, to be calibrated against real rollouts."""

    hostile_spawn_start_tick: int = 13000
    hostile_spawn_rate: float = 1.2
    hostile_villager_hit_chance: float = 0.3
    hostile_player_hit_chance: float = 0.2
    hostile_damage: float = 3.0
    golem_kill_chance: float = 0.8
    patrol_encounter_chance: float = 0.04
    raid_waves: int = 3
    raiders_first_wave: int = 3
    raider_hit_chance: float = 0.4
    raider_damage: float = 4.0
    archer_kills_per_step: int = 2
    archer_bad_outcome_chance: float = 0.05
    archer_bad_outcome_damage: float = 4.0
    torch_light_gain: float = 0.06
    gate_wall_gain: float = 0.1
    golem_build_steps: int = 2
    cure_steps: int = 4
    cure_price_discount: int = 2
    cure_reputation_gain: float = 0.15
    cheap_trade_reputation_gain: float = 0.1
    trade_reputation_gain: float = 0.01
    hero_price_factor: float = 0.7
    hero_reputation_gain: float = 0.1
    reputation_price_slope: float = 3.0
    trade_steps: int = 2
    fast_trade_steps: int = 1
    hunger_per_step: float = 0.25
    sprint_hunger_per_step: float = 1.0
    starvation_damage: float = 1.0
    regen_food_threshold: float = 18.0
    regen_per_step: float = 1.0
