"""
Village Defense & Trade simulator.

A fast, seeded model of the Minecraft village night that conforms to the
SubRepBaseEnv contract:

    reset(seed=None, options=None) -> (VillageState, info)
    step(action) -> (VillageState, phi(x_t), terminated, truncated, info)

One step is ``mission.ticks_per_step`` game ticks. Actions are the primitive
behaviours a Mineflayer bot can perform (move, craft, place, trade, shoot,
eat). The motive vector returned by ``step`` is phi of the state the action
leads to, so an option's own result is part of its motive return.
``info["task_payoff"]`` is the contract payoff: +1 on the step the books are
delivered, -1 on the step the contract fails (the player dies, a lost raid
invalidates it, or the sun rises first), 0 otherwise. The night runs to
sunrise even after delivery, so casualties and reputation are always judged
at the same moment; it ends early only if the player dies or the contract
fails.

``reset(options={"state": s})`` restores a saved ``VillageState`` so paired
candidate and baseline rollouts can start from the same state with the same
seed. Unknown actions and unmet preconditions raise ``ValueError`` before any
state changes.
"""

from __future__ import annotations

import math
import random

from village_sim.mission import VillageMission, VillageModelParams
from village_sim.motives import MOTIVE_NAMES, phi
from village_sim.state import (
    RAID_LOST,
    RAID_NONE,
    RAID_WON,
    TASK_DELIVERED,
    TASK_FAILED,
    TASK_OPEN,
    VillageState,
    dusk_state,
)

ACTIONS = (
    "wait",
    "move_hall",
    "move_perimeter",
    "walk_route",
    "sprint_route",
    "deliver",
    "craft_torches",
    "place_torch",
    "build_gate",
    "build_golem",
    "buy_book",
    "cheap_trade",
    "cure_villager",
    "set_trade_hotkeys",
    "shoot",
    "eat",
)

TORCHES_PER_FUEL = 4
IRON_BLOCKS_PER_GOLEM = 4


class VillageEnv:
    """Seeded simulator of one Village Defense & Trade night."""

    def __init__(
        self,
        seed: int | None = None,
        mission: VillageMission | None = None,
        params: VillageModelParams | None = None,
    ) -> None:
        self.mission = mission or VillageMission()
        self.params = params or VillageModelParams()
        self.rng = random.Random(seed)
        self.state: VillageState | None = None

    @property
    def metadata(self) -> dict:
        return {
            "environment_id": "village_sim_v2",
            "mission_id": self.mission.mission_id,
            "motive_names": list(MOTIVE_NAMES),
            "motive_schema_version": "2.0",
            "payoff_schema_version": "2.0",
            "observation_schema_version": "2.0",
            "action_schema_version": "2.0",
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[VillageState, dict]:
        if seed is not None:
            self.rng.seed(seed)
        start = (options or {}).get("state")
        if start is None:
            self.state = dusk_state(self.mission)
        elif isinstance(start, VillageState):
            self.state = start.copy()
        else:
            raise ValueError(f"options['state'] must be a VillageState, got {type(start).__name__}")
        return self.state.copy(), {}

    def step(self, action: str):
        if self.state is None:
            raise ValueError("reset() must be called before step()")
        self._validate_action(action)
        s = self.state
        events: list[str] = []

        self._apply_action(action, events)
        payoff = self._settle_delivery(events)
        self._advance_world(action, events)
        payoff += self._settle_failure(events)

        motives = phi(s, self.mission, self.params)
        info = {
            "task_payoff": float(payoff),
            "task_reward": float(payoff),
            "raid_active": s.raid_active,
            "events": events,
        }
        return s.copy(), motives, s.night_over(self.mission), False, info

    def close(self) -> None:
        pass

    def book_price(self, state: VillageState) -> int:
        """Librarian price for one enchanted book under the state's discounts."""
        p = self.params
        price = self.mission.base_book_price
        if state.hero_of_the_village:
            price *= p.hero_price_factor
        price -= p.cure_price_discount * state.cured_villagers
        price -= p.reputation_price_slope * (state.reputation - 0.5)
        return max(1, int(round(price)))

    # ── Action validation and effects ─────────────────────────────────────────

    def _validate_action(self, action: str) -> None:
        if action not in ACTIONS:
            raise ValueError(f"Unknown village action: {action!r}")
        s, m = self.state, self.mission
        if s.night_over(m):
            raise ValueError("The night is over; reset() before stepping again")
        in_village = s.location in ("hall", "perimeter")
        requirements = {
            "move_hall": (s.location == "perimeter", "move_hall needs the player at the perimeter"),
            "move_perimeter": (s.location == "hall", "move_perimeter needs the player at the hall"),
            "walk_route": (s.location in ("hall", "route"), "the route starts at the hall"),
            "sprint_route": (
                s.location in ("hall", "route") and s.player_food > 6.0,
                "sprinting needs the player at the hall or on the route with food above 6",
            ),
            "deliver": (s.location == "settlement" and s.books > 0, "deliver needs books at the settlement"),
            "craft_torches": (s.charcoal + s.coal > 0, "craft_torches needs charcoal or coal"),
            "place_torch": (
                s.location == "perimeter" and s.torches > 0,
                "place_torch needs a torch at the perimeter",
            ),
            "build_gate": (
                s.location == "perimeter" and s.wall_fraction < 1.0,
                "build_gate needs an open gap at the perimeter",
            ),
            "build_golem": (
                in_village and s.iron_blocks >= IRON_BLOCKS_PER_GOLEM and s.pumpkins > 0,
                "build_golem needs 4 iron blocks and a pumpkin inside the village",
            ),
            "buy_book": (
                s.location == "hall" and s.can_trade(m) and s.emeralds >= s.book_price,
                "buy_book needs a reachable librarian and enough emeralds at the hall",
            ),
            "cheap_trade": (
                s.location == "hall" and s.can_trade(m) and s.emeralds >= 1,
                "cheap_trade needs a reachable librarian and an emerald at the hall",
            ),
            "cure_villager": (
                s.location == "hall" and s.zombie_villagers > 0 and s.cure_kits > 0,
                "cure_villager needs a zombie villager and a cure kit at the hall",
            ),
            "set_trade_hotkeys": (not s.fast_trading, "trade hotkeys are already set"),
            "shoot": (in_village and s.threats > 0, "shoot needs a hostile or raider in the village"),
            "eat": (s.food_items > 0 and s.player_food < 20.0, "eat needs food and an unfilled hunger bar"),
        }
        ok, reason = requirements.get(action, (True, ""))
        if not ok:
            raise ValueError(f"Cannot {action}: {reason}")

    def _apply_action(self, action: str, events: list[str]) -> None:
        s, m, p = self.state, self.mission, self.params
        if action == "move_hall":
            s.location = "hall"
        elif action == "move_perimeter":
            s.location = "perimeter"
        elif action in ("walk_route", "sprint_route"):
            steps = m.sprint_route_steps if action == "sprint_route" else m.route_steps
            progress = s.route_progress if s.location == "route" else 0.0
            s.route_progress = min(1.0, progress + 1.0 / steps)
            s.location = "settlement" if s.route_progress >= 1.0 else "route"
        elif action == "deliver":
            s.books_delivered += s.books
            s.books = 0
            events.append("books_delivered")
        elif action == "craft_torches":
            if s.charcoal > 0:
                s.charcoal -= 1
            else:
                s.coal -= 1
            s.torches += TORCHES_PER_FUEL
        elif action == "place_torch":
            s.torches -= 1
            s.lit_fraction = min(1.0, s.lit_fraction + p.torch_light_gain)
        elif action == "build_gate":
            s.wall_fraction = min(1.0, s.wall_fraction + p.gate_wall_gain)
        elif action == "build_golem":
            s.golem_progress += 1
            if s.golem_progress >= p.golem_build_steps:
                s.iron_blocks -= IRON_BLOCKS_PER_GOLEM
                s.pumpkins -= 1
                s.golems += 1
                s.golem_progress = 0
                events.append("golem_built")
        elif action == "buy_book":
            s.trade_progress += 1
            needed = p.fast_trade_steps if s.fast_trading else p.trade_steps
            if s.trade_progress >= needed:
                s.emeralds -= s.book_price
                s.books += 1
                s.trade_progress = 0
                self._change_reputation(p.trade_reputation_gain)
                events.append("book_bought")
        elif action == "cheap_trade":
            s.emeralds -= 1
            self._change_reputation(p.cheap_trade_reputation_gain)
        elif action == "cure_villager":
            s.cure_progress += 1
            if s.cure_progress >= p.cure_steps:
                s.zombie_villagers -= 1
                s.cure_kits -= 1
                s.cured_villagers += 1
                s.cure_progress = 0
                self._change_reputation(p.cure_reputation_gain)
                events.append("villager_cured")
        elif action == "set_trade_hotkeys":
            s.fast_trading = True
        elif action == "shoot":
            if self.rng.random() < p.archer_bad_outcome_chance:
                self._hurt_player(p.archer_bad_outcome_damage)
                events.append("archer_hit")
            else:
                self._remove_threats(p.archer_kills_per_step)
        elif action == "eat":
            s.food_items -= 1
            s.player_food = min(20.0, s.player_food + m.food_item_value)
        if action != "buy_book":
            s.trade_progress = 0
        if action != "cure_villager":
            s.cure_progress = 0
        if action != "build_golem":
            s.golem_progress = 0

    # ── World dynamics ────────────────────────────────────────────────────────

    def _advance_world(self, action: str, events: list[str]) -> None:
        s, m, p = self.state, self.mission, self.params
        s.world_tick += m.ticks_per_step
        night = p.hostile_spawn_start_tick <= s.world_tick < m.sunrise_tick

        if night:
            rate = p.hostile_spawn_rate * (1.0 - s.lit_fraction) * (1.0 - s.wall_fraction)
            s.hostiles += self._poisson(rate)

        for _ in range(s.golems):
            if s.threats > 0 and self.rng.random() < p.golem_kill_chance:
                self._remove_threats(1)

        for _ in range(s.hostiles):
            if not p.villagers_shelter_from_hostiles and self.rng.random() < p.hostile_villager_hit_chance:
                self._hurt_random_villager(p.hostile_damage, events)
            if s.location == "perimeter" and self.rng.random() < p.hostile_player_hit_chance:
                self._hurt_player(p.hostile_damage)
        if night and s.location == "route" and self.rng.random() < p.hostile_player_hit_chance:
            self._hurt_player(p.hostile_damage)

        self._advance_raid(night, events)

        hunger = p.sprint_hunger_per_step if action == "sprint_route" else p.hunger_per_step
        s.player_food = max(0.0, s.player_food - hunger)
        if s.player_food <= 0.0:
            self._hurt_player(p.starvation_damage, armor=False)
        elif s.player_food >= p.regen_food_threshold and s.player_health > 0.0:
            s.player_health = min(20.0, s.player_health + p.regen_per_step)

        if s.world_tick >= m.sunrise_tick:
            s.hostiles = 0

    def _advance_raid(self, night: bool, events: list[str]) -> None:
        s, p = self.state, self.params
        if (
            night
            and not s.bad_omen
            and not s.raid_active
            and s.raid_outcome == RAID_NONE
            and s.location in ("perimeter", "route")
            and self.rng.random() < p.patrol_encounter_chance
        ):
            s.bad_omen = True
            events.append("bad_omen")
        if s.bad_omen and not s.raid_active and s.location in ("hall", "perimeter"):
            s.bad_omen = False
            s.raid_active = True
            s.raid_waves_left = p.raid_waves
            s.raiders = p.raiders_in_wave(p.raid_waves)
            events.append("raid_started")
        if not s.raid_active:
            return

        for _ in range(s.raiders):
            if self.rng.random() < p.raider_hit_chance * (1.0 - s.wall_fraction):
                self._hurt_random_villager(p.raider_damage, events)

        if s.villagers_alive == 0:
            s.raid_active = False
            s.raiders = 0
            s.raid_waves_left = 0
            s.raid_outcome = RAID_LOST
            if s.task_status == TASK_OPEN:
                s.contract_valid = False
            events.append("raid_lost")
        elif s.raiders == 0:
            s.raid_waves_left -= 1
            if s.raid_waves_left > 0:
                s.raiders = p.raiders_in_wave(s.raid_waves_left)
                events.append("raid_wave")
            else:
                s.raid_active = False
                s.raid_outcome = RAID_WON
                s.hero_of_the_village = True
                self._change_reputation(p.hero_reputation_gain)
                events.append("raid_won")

    # ── Task outcome ─────────────────────────────────────────────────────────

    def _settle_delivery(self, events: list[str]) -> float:
        s, m = self.state, self.mission
        if s.task_status != TASK_OPEN:
            return 0.0
        delivered = s.books_delivered * m.book_value >= m.contract_value
        if delivered and s.contract_valid and s.world_tick < m.sunrise_tick:
            s.task_status = TASK_DELIVERED
            events.append("contract_completed")
            return 1.0
        return 0.0

    def _settle_failure(self, events: list[str]) -> float:
        s, m = self.state, self.mission
        if s.task_status != TASK_OPEN:
            return 0.0
        if s.player_health <= 0.0 or not s.contract_valid or s.world_tick >= m.sunrise_tick:
            s.task_status = TASK_FAILED
            events.append("contract_failed")
            return -1.0
        return 0.0

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _change_reputation(self, amount: float) -> None:
        s = self.state
        s.reputation = min(1.0, max(0.0, s.reputation + amount))
        s.book_price = self.book_price(s)

    def _remove_threats(self, count: int) -> None:
        s = self.state
        from_raid = min(count, s.raiders)
        s.raiders -= from_raid
        s.hostiles = max(0, s.hostiles - (count - from_raid))

    def _hurt_player(self, damage: float, armor: bool = True) -> None:
        s = self.state
        factor = 1.0 - s.armor / 25.0 if armor else 1.0
        s.player_health = max(0.0, s.player_health - damage * factor)

    def _hurt_random_villager(self, damage: float, events: list[str]) -> None:
        s = self.state
        alive = [i for i, h in enumerate(s.villager_health) if h > 0.0]
        if not alive:
            return
        i = self.rng.choice(alive)
        health = list(s.villager_health)
        health[i] = max(0.0, health[i] - damage)
        s.villager_health = tuple(health)
        if health[i] == 0.0:
            events.append("villager_died")

    def _poisson(self, rate: float) -> int:
        """Knuth's Poisson sampler; rates here are small."""
        if rate <= 0.0:
            return 0
        limit, k, product = math.exp(-rate), 0, self.rng.random()
        while product > limit:
            k += 1
            product *= self.rng.random()
        return k
