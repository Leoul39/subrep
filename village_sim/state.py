"""
Symbolic state of the Village Defense & Trade night.

``VillageState`` is the contract between the simulator and the real-server
adapter: every field is something a Mineflayer bot can observe (world time,
health and hunger, villagers, hostile mobs, raid status, inventory, trade
prices, position), so the same motive features, options and certification
code run on both.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from village_sim.mission import LOCATIONS, VillageMission

TASK_OPEN = "open"
TASK_DELIVERED = "delivered"
TASK_FAILED = "failed"
RAID_NONE = "none"
RAID_WON = "won"
RAID_LOST = "lost"


@dataclass
class VillageState:
    """One observation of the village night."""

    world_tick: int = 12000
    location: str = "hall"
    route_progress: float = 0.0

    player_health: float = 20.0
    player_food: float = 14.0
    armor: float = 4.0

    villager_health: tuple[float, ...] = (20.0, 20.0, 20.0, 20.0)
    hostiles: int = 0
    raid_active: bool = False
    raid_waves_left: int = 0
    raiders: int = 0
    raid_outcome: str = RAID_NONE
    bad_omen: bool = False
    hero_of_the_village: bool = False

    golems: int = 0
    golem_progress: int = 0
    lit_fraction: float = 0.3
    wall_fraction: float = 0.6
    trading_access: bool = True

    emeralds: int = 20
    books: int = 0
    books_delivered: int = 0
    book_price: int = 7
    reputation: float = 0.5
    cured_villagers: int = 0
    zombie_villagers: int = 1
    cure_kits: int = 1
    cure_progress: int = 0
    trade_progress: int = 0
    fast_trading: bool = False

    torches: int = 4
    charcoal: int = 2
    coal: int = 4
    iron_blocks: int = 4
    pumpkins: int = 1
    food_items: int = 3

    contract_valid: bool = True
    task_status: str = TASK_OPEN
    facts: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.location not in LOCATIONS:
            raise ValueError(f"location must be one of {LOCATIONS}, got {self.location!r}")
        self.villager_health = tuple(float(h) for h in self.villager_health)
        self.facts = frozenset(self.facts)

    def copy(self) -> "VillageState":
        return copy.deepcopy(self)

    @property
    def villagers_alive(self) -> int:
        return sum(1 for h in self.villager_health if h > 0.0)

    @property
    def threats(self) -> int:
        return self.hostiles + self.raiders

    def night_over(self, mission: VillageMission) -> bool:
        """The sun is up, the player is dead, or the contract has failed."""
        return (
            self.world_tick >= mission.sunrise_tick
            or self.player_health <= 0.0
            or self.task_status == TASK_FAILED
        )

    def can_trade(self, mission: VillageMission) -> bool:
        """A librarian is alive and reachable; the first villagers are the librarians."""
        librarians = self.villager_health[: mission.librarians]
        return self.trading_access and any(h > 0.0 for h in librarians)

    def contract_value_held(self, mission: VillageMission) -> float:
        """Contract value already delivered or carried as books."""
        return (self.books_delivered + self.books) * mission.book_value

    def books_still_needed(self, mission: VillageMission) -> int:
        return max(0, mission.books_required - self.books_delivered - self.books)


def dusk_state(mission: VillageMission | None = None) -> VillageState:
    """The scenario's start: dusk at the trading hall with the paper's resources."""
    mission = mission or VillageMission()
    return VillageState(
        world_tick=mission.dusk_tick,
        villager_health=(20.0,) * mission.protected_villagers,
        book_price=mission.base_book_price,
    )


def sample_start_state(rng: random.Random, mission: VillageMission | None = None) -> VillageState:
    """Draw a start context around the paper's dusk scenario.

    Varies the quantities the scenario leaves open (resources, prices,
    reputation, defences, early threats) so certificates cover more than a
    single start state. The same ``rng`` seed always gives the same state.
    """
    mission = mission or VillageMission()
    state = dusk_state(mission)
    state.world_tick = mission.dusk_tick + 100 * rng.randint(0, 10)
    state.player_food = float(rng.randint(10, 18))
    state.armor = float(rng.choice((2, 4, 6)))
    state.villager_health = tuple(float(rng.randint(14, 20)) for _ in range(mission.protected_villagers))
    state.hostiles = rng.randint(0, 2) if state.world_tick >= 13000 else 0
    state.lit_fraction = round(rng.uniform(0.2, 0.5), 2)
    state.wall_fraction = round(rng.uniform(0.5, 0.7), 2)
    state.emeralds = rng.randint(18, 24)
    state.reputation = round(rng.uniform(0.4, 0.6), 2)
    state.torches = rng.randint(0, 6)
    state.charcoal = rng.randint(0, 4)
    state.coal = rng.randint(2, mission.coal_capacity)
    state.iron_blocks = rng.choice((0, 4, 8))
    state.food_items = rng.randint(1, mission.food_item_capacity)
    state.cure_kits = rng.choice((0, 1))
    return state
