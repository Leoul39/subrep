"""
Temporally extended options for the village night.

Each option has an initiation condition (``can_start``), a closed-loop
behaviour over primitive actions (``OptionRun.act``), a termination condition
(``OptionRun.done``) and a step limit, so its duration is random and depends
on the state it starts from. ``PROTECTED_WAIT`` is the conservative baseline
every candidate is compared against: it steps back into the hall from the
perimeter, eats when hungry and otherwise waits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from village_sim.mission import VillageMission
from village_sim.state import TASK_DELIVERED, VillageState

IN_VILLAGE = ("hall", "perimeter")


class OptionRun:
    """One execution of an option; holds the bookkeeping its termination needs."""

    def __init__(self, start: VillageState, mission: VillageMission) -> None:
        self.start = start.copy()
        self.mission = mission

    def act(self, state: VillageState) -> str:
        raise NotImplementedError

    def done(self, state: VillageState) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class VillageOption:
    """A scripted option: initiation set, behaviour, termination and step limit."""

    option_id: str
    version: str
    group: str
    max_steps: int
    can_start: Callable[[VillageState, VillageMission], bool]
    run_type: type

    def new_run(self, start: VillageState, mission: VillageMission) -> OptionRun:
        if not self.can_start(start, mission):
            raise ValueError(f"{self.option_id} cannot start from this state")
        return self.run_type(start, mission)


def _buying_done(state: VillageState, mission: VillageMission) -> bool:
    return (
        state.books_still_needed(mission) == 0
        or state.emeralds < state.book_price
        or not state.can_trade(mission)
    )


class _ProtectedWait(OptionRun):
    def act(self, state):
        if state.location == "perimeter":
            return "move_hall"
        if state.player_food <= 6.0 and state.food_items > 0:
            return "eat"
        return "wait"

    def done(self, state):
        return False


class _TorchCorridor(OptionRun):
    TORCHES = 4

    def __init__(self, start, mission):
        super().__init__(start, mission)
        self.placed = 0

    def act(self, state):
        if state.location != "perimeter":
            return "move_perimeter"
        if state.torches == 0:
            return "craft_torches"
        self.placed += 1
        return "place_torch"

    def done(self, state):
        out_of_light = state.torches == 0 and state.charcoal + state.coal == 0
        return self.placed >= self.TORCHES or state.lit_fraction >= 1.0 or out_of_light


class _IronGolemSpawn(OptionRun):
    def act(self, state):
        return "build_golem"

    def done(self, state):
        return state.golems > self.start.golems


class _SwingGateBarricade(OptionRun):
    GATES = 2

    def __init__(self, start, mission):
        super().__init__(start, mission)
        self.built = 0

    def act(self, state):
        if state.location != "perimeter":
            return "move_perimeter"
        self.built += 1
        return "build_gate"

    def done(self, state):
        return self.built >= self.GATES or state.wall_fraction >= 1.0


class _DiscountChain(OptionRun):
    def act(self, state):
        if state.location != "hall":
            return "move_hall"
        if state.cured_villagers == self.start.cured_villagers:
            return "cure_villager"
        return "buy_book"

    def done(self, state):
        if state.cured_villagers > self.start.cured_villagers:
            return _buying_done(state, self.mission)
        return state.zombie_villagers == 0 or state.cure_kits == 0


class _ReputationFirst(OptionRun):
    CHEAP_TRADES = 2

    def __init__(self, start, mission):
        super().__init__(start, mission)
        self.cheap_trades = 0

    def act(self, state):
        if state.location != "hall":
            return "move_hall"
        if self.cheap_trades < self.CHEAP_TRADES and state.emeralds >= 1:
            self.cheap_trades += 1
            return "cheap_trade"
        return "buy_book"

    def done(self, state):
        if not state.can_trade(self.mission):
            return True
        warmed_up = self.cheap_trades >= self.CHEAP_TRADES or state.emeralds < 1
        return warmed_up and _buying_done(state, self.mission)


class _BuyBooks(OptionRun):
    def act(self, state):
        if state.location != "hall":
            return "move_hall"
        return "buy_book"

    def done(self, state):
        return _buying_done(state, self.mission)


class _ArcherKite(OptionRun):
    def act(self, state):
        return "shoot"

    def done(self, state):
        return state.threats == 0


class _TradeMenuHotkeys(OptionRun):
    def act(self, state):
        return "set_trade_hotkeys"

    def done(self, state):
        return state.fast_trading


class _SafeSprintPathFollow(OptionRun):
    def act(self, state):
        if state.location == "settlement":
            return "deliver"
        if state.player_food <= 6.0 and state.food_items > 0:
            return "eat"
        return "sprint_route" if state.player_food > 6.0 else "walk_route"

    def done(self, state):
        return state.task_status == TASK_DELIVERED or (state.location == "settlement" and state.books == 0)


def _carries_contract(state: VillageState, mission: VillageMission) -> bool:
    return state.books_still_needed(mission) == 0 and state.books > 0


PROTECTED_WAIT = VillageOption(
    "protected_wait_v1", "1.0", "baseline", 200,
    lambda s, m: True, _ProtectedWait,
)
TORCH_CORRIDOR = VillageOption(
    "torch_corridor_v1", "1.0", "defend", 8,
    lambda s, m: s.location in IN_VILLAGE and s.lit_fraction < 1.0 and s.torches + s.charcoal + s.coal > 0,
    _TorchCorridor,
)
IRON_GOLEM_SPAWN = VillageOption(
    "iron_golem_spawn_v1", "1.0", "defend", 3,
    lambda s, m: s.location in IN_VILLAGE and s.iron_blocks >= 4 and s.pumpkins > 0,
    _IronGolemSpawn,
)
SWING_GATE_BARRICADE = VillageOption(
    "swing_gate_barricade_v1", "1.0", "defend", 4,
    lambda s, m: s.location in IN_VILLAGE and s.wall_fraction < 1.0,
    _SwingGateBarricade,
)
DISCOUNT_CHAIN = VillageOption(
    "discount_chain_v1", "1.0", "trade", 12,
    lambda s, m: (
        s.location in IN_VILLAGE and s.zombie_villagers > 0 and s.cure_kits > 0 and s.can_trade(m)
    ),
    _DiscountChain,
)
REPUTATION_FIRST = VillageOption(
    "reputation_first_v1", "1.0", "trade", 10,
    lambda s, m: s.location in IN_VILLAGE and s.can_trade(m) and s.emeralds >= 1,
    _ReputationFirst,
)
BUY_BOOKS = VillageOption(
    "buy_books_v1", "1.0", "trade", 8,
    lambda s, m: (
        s.location in IN_VILLAGE and s.can_trade(m)
        and s.emeralds >= s.book_price and s.books_still_needed(m) > 0
    ),
    _BuyBooks,
)
ARCHER_KITE = VillageOption(
    "archer_kite_v1", "1.0", "threat", 6,
    lambda s, m: s.location in IN_VILLAGE and s.threats > 0,
    _ArcherKite,
)
TRADE_MENU_HOTKEYS = VillageOption(
    "trade_menu_hotkeys_v1", "1.0", "threat", 1,
    lambda s, m: not s.fast_trading and s.books_still_needed(m) > 0,
    _TradeMenuHotkeys,
)
SAFE_SPRINT_PATH_FOLLOW = VillageOption(
    "safe_sprint_path_follow_v1", "1.0", "logistics", 10,
    lambda s, m: s.location in ("hall", "route", "settlement") and _carries_contract(s, m),
    _SafeSprintPathFollow,
)

CANDIDATE_OPTIONS = (
    TORCH_CORRIDOR,
    IRON_GOLEM_SPAWN,
    SWING_GATE_BARRICADE,
    DISCOUNT_CHAIN,
    REPUTATION_FIRST,
    BUY_BOOKS,
    ARCHER_KITE,
    TRADE_MENU_HOTKEYS,
    SAFE_SPRINT_PATH_FOLLOW,
)
OPTIONS = {option.option_id: option for option in (PROTECTED_WAIT, *CANDIDATE_OPTIONS)}
