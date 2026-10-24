"""Tests for the Village Defense & Trade simulator, its options and paired rollouts."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from env.base_env import SubRepBaseEnv, validate_env_metadata
from village_sim.env import ACTIONS, VillageEnv
from village_sim.mission import VillageMission, VillageModelParams
from village_sim.motives import MOTIVE_NAMES, PSI_NAMES, phi, psi, shelter
from village_sim.options import (
    ARCHER_KITE,
    CANDIDATE_OPTIONS,
    DISCOUNT_CHAIN,
    IRON_GOLEM_SPAWN,
    OPTIONS,
    PROTECTED_WAIT,
    REPUTATION_FIRST,
    SAFE_SPRINT_PATH_FOLLOW,
    SWING_GATE_BARRICADE,
    TORCH_CORRIDOR,
    TRADE_MENU_HOTKEYS,
)
from village_sim.rollouts import EVALUATION_HORIZON, evaluate_option, paired_rollout, return_scale, run_option
from village_sim.state import (
    RAID_LOST,
    RAID_WON,
    TASK_DELIVERED,
    TASK_FAILED,
    VillageState,
    dusk_state,
    sample_start_state,
)

MISSION = VillageMission()
PARAMS = VillageModelParams()
GAMMA = 0.99
SAFETY, REPUTATION, DEADLINE, INVENTORY, SUSTAINABILITY, INFRASTRUCTURE = range(6)


def _env_at(state: VillageState, seed: int = 0) -> VillageEnv:
    env = VillageEnv()
    env.reset(seed=seed, options={"state": state})
    return env


def _dusk(**changes) -> VillageState:
    state = dusk_state()
    for key, value in changes.items():
        setattr(state, key, value)
    return state


def _night_state(**changes) -> VillageState:
    return _dusk(world_tick=15000, **changes)


def _mean_delta(option, start, seeds=range(20)) -> tuple[float, np.ndarray]:
    outcomes = evaluate_option(VillageEnv(), option, start, seeds, GAMMA)
    return float(np.mean([o.delta_r for o in outcomes])), np.mean([o.delta_n for o in outcomes], axis=0)


# ── Mission and state ───────────────────────────────────────────────────────

def test_mission_night_and_contract_sizes():
    assert MISSION.night_steps == 110
    assert MISSION.books_required == 3


def test_dusk_state_matches_mission():
    state = dusk_state()
    assert state.world_tick == MISSION.dusk_tick
    assert len(state.villager_health) == MISSION.protected_villagers
    assert state.book_price == MISSION.base_book_price
    assert state.location == "hall"


def test_unknown_location_is_rejected():
    with pytest.raises(ValueError, match="location"):
        VillageState(location="nether")


def test_sampled_start_states_are_reproducible_and_varied():
    first = sample_start_state(random.Random(5))
    again = sample_start_state(random.Random(5))
    assert first == again
    states = [sample_start_state(random.Random(seed)) for seed in range(20)]
    assert len({(s.emeralds, s.coal, s.lit_fraction) for s in states}) > 10
    for s in states:
        assert MISSION.dusk_tick <= s.world_tick < MISSION.sunrise_tick
        assert 0.0 <= s.lit_fraction <= 1.0 and 0.0 <= s.wall_fraction <= 1.0


def test_librarians_gate_trading():
    state = dusk_state()
    assert state.can_trade(MISSION)
    state.villager_health = (0.0, 0.0, 20.0, 20.0)
    assert not state.can_trade(MISSION)
    state.villager_health = (0.0, 5.0, 20.0, 20.0)
    state.trading_access = False
    assert not state.can_trade(MISSION)


# ── Motive features and model context ───────────────────────────────────────

def test_motive_names_are_canonical():
    assert MOTIVE_NAMES == [
        "Safety", "Reputation", "DeadlineSlack", "InventoryValue", "Sustainability", "Infrastructure",
    ]


@pytest.mark.parametrize("seed", range(10))
def test_phi_is_bounded_float32_six_vector(seed):
    for state in (dusk_state(), sample_start_state(random.Random(seed))):
        vector = phi(state)
        assert vector.shape == (6,) and vector.dtype == np.float32
        assert np.all(vector >= 0.0) and np.all(vector <= 1.0)


def test_phi_stays_bounded_in_extreme_states():
    state = dusk_state()
    state.world_tick = MISSION.sunrise_tick + 500
    state.villager_health = (0.0,) * 4
    state.player_health = 0.0
    state.hostiles = 500
    state.raid_active, state.raiders, state.raid_waves_left = True, 50, 3
    state.emeralds, state.coal, state.iron_blocks = 10_000, 100, 100
    vector = phi(state)
    assert np.all(np.isfinite(vector)) and np.all(vector >= 0.0) and np.all(vector <= 1.0)


def test_safety_falls_with_casualties_hostiles_and_raids():
    base = phi(dusk_state())[SAFETY]
    assert phi(_night_state(villager_health=(0.0, 20.0, 20.0, 20.0)))[SAFETY] < base
    assert phi(_night_state(hostiles=4))[SAFETY] < base
    assert phi(_night_state(raid_active=True, raiders=3, raid_waves_left=3))[SAFETY] < base


def test_shelter_rises_with_light_walls_and_golems():
    base = shelter(dusk_state(), PARAMS)
    lit = dusk_state()
    lit.lit_fraction = 0.9
    walled = dusk_state()
    walled.wall_fraction = 0.9
    guarded = dusk_state()
    guarded.golems = 1
    assert shelter(lit, PARAMS) > base
    assert shelter(walled, PARAMS) > base
    assert shelter(guarded, PARAMS) == 1.0


def test_reputation_is_zero_once_the_contract_is_invalid():
    state = dusk_state()
    state.contract_valid = False
    assert phi(state)[REPUTATION] == 0.0


def test_deadline_slack_tracks_time_and_remaining_work():
    early, late = dusk_state(), dusk_state()
    late.world_tick += 3000
    assert phi(late)[DEADLINE] < phi(early)[DEADLINE]

    with_books = dusk_state()
    with_books.books = 3
    assert phi(with_books)[DEADLINE] > phi(early)[DEADLINE]

    hotkeys = dusk_state()
    hotkeys.fast_trading = True
    assert phi(hotkeys)[DEADLINE] > phi(early)[DEADLINE]


def test_inventory_value_counts_delivered_carried_and_affordable_books():
    state = dusk_state()
    assert phi(state)[INVENTORY] == pytest.approx(2 * 8 / 24)
    state.book_price = 5
    assert phi(state)[INVENTORY] == pytest.approx(1.0)
    state.villager_health = (0.0, 0.0, 20.0, 20.0)
    assert phi(state)[INVENTORY] == 0.0
    state.books = 1
    assert phi(state)[INVENTORY] == pytest.approx(8 / 24)


def test_sustainability_ignores_renewable_charcoal():
    state = dusk_state()
    charcoal_used = dusk_state()
    charcoal_used.charcoal = 0
    coal_used = dusk_state()
    coal_used.coal -= 2
    assert phi(charcoal_used)[SUSTAINABILITY] == phi(state)[SUSTAINABILITY]
    assert phi(coal_used)[SUSTAINABILITY] < phi(state)[SUSTAINABILITY]


def test_blocked_trading_access_halves_infrastructure():
    open_state, blocked = dusk_state(), dusk_state()
    blocked.trading_access = False
    assert phi(blocked)[INFRASTRUCTURE] == pytest.approx(0.5 * phi(open_state)[INFRASTRUCTURE])


@pytest.mark.parametrize("seed", range(5))
def test_psi_matches_its_names_and_is_bounded(seed):
    vector = psi(sample_start_state(random.Random(seed)))
    assert vector.shape == (len(PSI_NAMES),)
    assert len(set(PSI_NAMES)) == len(PSI_NAMES)
    assert np.all(np.isfinite(vector)) and np.all(vector >= 0.0) and np.all(vector <= 1.0)
    assert np.allclose(vector[:6], phi(sample_start_state(random.Random(seed))))


# ── Environment contract and dynamics ───────────────────────────────────────

def test_env_satisfies_the_shared_contract():
    env = VillageEnv(seed=1)
    assert isinstance(env, SubRepBaseEnv)
    validate_env_metadata(env.metadata)
    assert env.metadata["motive_names"] == MOTIVE_NAMES
    state, info = env.reset()
    assert state == dusk_state() and info == {}
    _, motives, terminated, truncated, info = env.step("wait")
    assert motives.shape == (6,) and not terminated and not truncated
    assert info["task_payoff"] == 0.0


def test_step_returns_phi_of_the_state_the_action_leads_to():
    env = _env_at(dusk_state())
    state, motives, *_ = env.step("move_perimeter")
    assert np.array_equal(motives, phi(state))


def test_reset_restores_a_saved_state_as_a_copy():
    saved = _night_state(hostiles=2, emeralds=11)
    env = VillageEnv()
    state, _ = env.reset(seed=3, options={"state": saved})
    assert state == saved
    state.emeralds = 0
    saved.emeralds = 0
    assert env.state.emeralds == 11


def test_reset_rejects_a_non_state_option():
    with pytest.raises(ValueError, match="VillageState"):
        VillageEnv().reset(options={"state": {"emeralds": 3}})


def test_same_seed_and_actions_give_the_same_trace():
    def trace(seed):
        env = _env_at(_night_state(location="perimeter"), seed)
        return [env.step(action)[0] for action in ["wait", "place_torch", "wait", "move_hall", "wait"]]

    assert trace(4) == trace(4)
    assert any(trace(seed) != trace(4) for seed in range(5, 15))


def test_unknown_action_is_rejected_without_changing_state():
    env = _env_at(dusk_state())
    before = env.state.copy()
    with pytest.raises(ValueError, match="Unknown village action"):
        env.step("fly")
    assert env.state == before


@pytest.mark.parametrize("state_changes, action", [
    ({}, "move_hall"),
    ({"location": "perimeter"}, "move_perimeter"),
    ({}, "place_torch"),
    ({"charcoal": 0, "coal": 0}, "craft_torches"),
    ({"iron_blocks": 0}, "build_golem"),
    ({"emeralds": 3}, "buy_book"),
    ({"location": "perimeter"}, "buy_book"),
    ({"cure_kits": 0}, "cure_villager"),
    ({}, "shoot"),
    ({}, "deliver"),
    ({"player_food": 4.0}, "sprint_route"),
    ({"fast_trading": True}, "set_trade_hotkeys"),
])
def test_unmet_preconditions_are_rejected_without_changing_state(state_changes, action):
    state = dusk_state()
    for key, value in state_changes.items():
        setattr(state, key, value)
    env = _env_at(state)
    before = env.state.copy()
    with pytest.raises(ValueError, match="Cannot"):
        env.step(action)
    assert env.state == before


def test_every_action_has_a_named_precondition_or_none():
    assert len(set(ACTIONS)) == len(ACTIONS) == 16


def test_torches_are_crafted_from_charcoal_first_then_coal():
    env = _env_at(dusk_state())
    env.step("craft_torches")
    assert (env.state.charcoal, env.state.coal, env.state.torches) == (1, 4, 8)
    env.state.charcoal = 0
    env.step("craft_torches")
    assert (env.state.coal, env.state.torches) == (3, 12)


def test_placing_torches_and_gates_raises_defences():
    env = _env_at(_dusk(location="perimeter"))
    env.step("place_torch")
    env.step("build_gate")
    assert env.state.lit_fraction == pytest.approx(0.3 + PARAMS.torch_light_gain)
    assert env.state.wall_fraction == pytest.approx(0.6 + PARAMS.gate_wall_gain)
    assert env.state.torches == 3


def test_golem_takes_build_steps_and_consumes_iron_and_pumpkin():
    env = _env_at(dusk_state())
    env.step("build_golem")
    assert env.state.golems == 0
    env.step("build_golem")
    assert (env.state.golems, env.state.iron_blocks, env.state.pumpkins) == (1, 0, 0)


def test_interrupting_a_multi_step_action_loses_its_progress():
    env = _env_at(dusk_state())
    env.step("build_golem")
    env.step("wait")
    env.step("build_golem")
    assert env.state.golems == 0


def test_buying_a_book_takes_two_steps_or_one_with_hotkeys():
    env = _env_at(dusk_state())
    env.step("buy_book")
    assert env.state.books == 0
    env.step("buy_book")
    assert (env.state.books, env.state.emeralds) == (1, 13)

    fast = dusk_state()
    fast.fast_trading = True
    env = _env_at(fast)
    env.step("buy_book")
    assert env.state.books == 1


def test_cheap_trades_raise_reputation_and_lower_the_price():
    env = _env_at(dusk_state())
    env.step("cheap_trade")
    env.step("cheap_trade")
    assert env.state.reputation == pytest.approx(0.7)
    assert env.state.book_price == 6 and env.state.emeralds == 18


def test_curing_a_zombie_villager_discounts_books():
    env = _env_at(dusk_state())
    for _ in range(PARAMS.cure_steps):
        env.step("cure_villager")
    s = env.state
    assert (s.cured_villagers, s.zombie_villagers, s.cure_kits) == (1, 0, 0)
    assert s.book_price <= MISSION.base_book_price - PARAMS.cure_price_discount


def test_hunger_and_starvation():
    env = _env_at(_dusk(player_food=0.25))
    env.step("wait")
    assert env.state.player_food == 0.0
    assert env.state.player_health == 20.0 - PARAMS.starvation_damage
    env.step("eat")
    assert env.state.player_food == MISSION.food_item_value - PARAMS.hunger_per_step
    assert env.state.player_health == 20.0 - PARAMS.starvation_damage


def test_hostiles_spawn_only_at_night_and_golems_kill_them():
    env = _env_at(dusk_state(), seed=1)
    for _ in range(5):
        env.step("wait")
    assert env.state.hostiles == 0

    dark = _night_state(lit_fraction=0.0, wall_fraction=0.0)
    totals = []
    for seed in range(10):
        env = _env_at(dark, seed)
        for _ in range(10):
            env.step("wait")
        totals.append(env.state.hostiles)
    assert np.mean(totals) > 5

    guarded = _night_state(lit_fraction=0.0, wall_fraction=0.0, golems=2)
    totals = []
    for seed in range(10):
        env = _env_at(guarded, seed)
        for _ in range(10):
            env.step("wait")
        totals.append(env.state.hostiles)
    assert np.mean(totals) < 2


def _raid_state(**changes) -> VillageState:
    raid = {"raid_active": True, "raid_waves_left": PARAMS.raid_waves, "raiders": 0}
    return _night_state(**{**raid, **changes})


def test_clearing_every_raid_wave_wins_hero_of_the_village():
    env = _env_at(_raid_state(), seed=2)
    for _ in range(40):
        if not env.state.raid_active:
            break
        env.step("shoot" if env.state.threats else "wait")
    s = env.state
    assert s.raid_outcome == RAID_WON and s.hero_of_the_village
    assert s.book_price < MISSION.base_book_price


def test_losing_a_raid_invalidates_the_contract():
    env = _env_at(_raid_state(raiders=6, villager_health=(1.0, 1.0, 1.0, 1.0), wall_fraction=0.0), seed=0)
    for _ in range(20):
        _, _, terminated, _, info = env.step("wait")
        if terminated:
            break
    assert env.state.raid_outcome == RAID_LOST
    assert env.state.task_status == TASK_FAILED and info["task_payoff"] == -1.0


def test_delivering_the_contract_pays_once_and_ends_the_night():
    state = _dusk(location="settlement", books=3)
    env = _env_at(state)
    _, _, terminated, _, info = env.step("deliver")
    assert terminated and info["task_payoff"] == 1.0
    assert env.state.task_status == TASK_DELIVERED
    with pytest.raises(ValueError, match="over"):
        env.step("wait")


def test_sunrise_without_delivery_fails_the_contract():
    env = _env_at(_dusk(world_tick=MISSION.sunrise_tick - 100))
    _, _, terminated, _, info = env.step("wait")
    assert terminated and info["task_payoff"] == -1.0
    assert env.state.hostiles == 0


def test_player_death_fails_the_contract():
    env = _env_at(_night_state(location="perimeter", player_health=1.0, hostiles=30, armor=0.0), seed=1)
    _, _, terminated, _, info = env.step("wait")
    assert terminated and info["task_payoff"] == -1.0


# ── Options ─────────────────────────────────────────────────────────────────

def test_option_ids_are_unique_and_fit_the_evaluation_horizon():
    assert len(OPTIONS) == len(CANDIDATE_OPTIONS) + 1
    assert PROTECTED_WAIT.group == "baseline"
    assert all(option.max_steps <= EVALUATION_HORIZON for option in CANDIDATE_OPTIONS)


def test_the_protected_wait_retreats_and_eats_but_never_spends():
    env = _env_at(_dusk(location="perimeter", player_food=5.0))
    rollout = run_option(env, PROTECTED_WAIT, env.state.copy(), seed=0, gamma=GAMMA, horizon=3)
    assert rollout.env_steps == 3
    s = env.state
    assert s.location == "hall" and s.food_items == 2
    assert (s.emeralds, s.coal, s.charcoal, s.torches) == (20, 4, 2, 4)


def test_an_option_cannot_start_outside_its_initiation_set():
    with pytest.raises(ValueError, match="cannot start"):
        ARCHER_KITE.new_run(dusk_state(), MISSION)


@pytest.mark.parametrize("seed", range(40))
def test_every_startable_option_runs_to_completion_from_sampled_states(seed):
    rng = random.Random(seed)
    start = sample_start_state(rng)
    if rng.random() < 0.5:
        start.world_tick = 15000
        start.hostiles = rng.randint(0, 4)
    if rng.random() < 0.3:
        start.raid_active, start.raid_waves_left, start.raiders = True, 3, 3
    if rng.random() < 0.3:
        start.books = 3
    env = VillageEnv()
    for option in CANDIDATE_OPTIONS:
        if option.can_start(start, MISSION):
            rollout = run_option(env, option, start, seed, GAMMA)
            assert 1 <= rollout.option_steps <= option.max_steps
            assert rollout.env_steps <= EVALUATION_HORIZON


def test_torch_corridor_places_four_torches_at_the_perimeter():
    env = VillageEnv()
    run_option(env, TORCH_CORRIDOR, _dusk(torches=0), seed=0, gamma=GAMMA)
    assert env.state.lit_fraction == pytest.approx(0.3 + 4 * PARAMS.torch_light_gain)
    assert env.state.charcoal == 1 and env.state.coal == 4


def test_discount_chain_cures_then_buys_the_books():
    env = VillageEnv()
    rollout = run_option(env, DISCOUNT_CHAIN, dusk_state(), seed=0, gamma=GAMMA)
    assert env.state.cured_villagers == 1
    assert env.state.books == MISSION.books_required
    assert rollout.stop_reason == "done"


def test_safe_sprint_delivers_a_carried_contract():
    env = VillageEnv()
    rollout = run_option(env, SAFE_SPRINT_PATH_FOLLOW, _dusk(books=3), seed=0, gamma=GAMMA)
    assert rollout.task_status == TASK_DELIVERED
    assert rollout.payoff > 0.9


def test_archer_kite_stops_when_the_threats_are_gone():
    rollout = run_option(VillageEnv(), ARCHER_KITE, _night_state(hostiles=3), seed=0, gamma=GAMMA)
    assert rollout.stop_reason == "done"
    assert rollout.option_steps < ARCHER_KITE.max_steps


# ── Paired rollouts ─────────────────────────────────────────────────────────

def test_return_scale_is_the_inverse_discounted_horizon():
    assert return_scale(1.0, 4) == 0.25
    assert return_scale(GAMMA) == pytest.approx(1.0 / sum(GAMMA ** t for t in range(EVALUATION_HORIZON)))
    with pytest.raises(ValueError):
        return_scale(GAMMA, 0)


def test_paired_rollouts_are_reproducible_and_bounded():
    start = _night_state(hostiles=2)
    first = paired_rollout(VillageEnv(), SWING_GATE_BARRICADE, start, seed=9, gamma=GAMMA)
    again = paired_rollout(VillageEnv(), SWING_GATE_BARRICADE, start, seed=9, gamma=GAMMA)
    assert first == again
    for side in (first.candidate, first.baseline):
        assert all(0.0 <= v <= 1.0 for v in side.motives)
    assert first.delta_r == first.candidate.payoff - first.baseline.payoff
    assert np.allclose(first.delta_n, np.subtract(first.candidate.motives, first.baseline.motives))


def test_the_baseline_against_itself_has_zero_deltas():
    outcome = paired_rollout(VillageEnv(), PROTECTED_WAIT, _night_state(hostiles=3), seed=4, gamma=GAMMA)
    assert outcome.delta_r == 0.0
    assert np.all(outcome.delta_n == 0.0)


def test_a_rollout_that_ends_early_keeps_its_final_state_for_the_horizon():
    env = VillageEnv()
    rollout = run_option(env, SAFE_SPRINT_PATH_FOLLOW, _dusk(location="settlement", books=3), 0, GAMMA)
    assert rollout.env_steps == 1
    assert np.allclose(rollout.motives, phi(env.state), atol=1e-6)


def test_the_payoff_is_not_a_function_of_the_motives():
    outcome = paired_rollout(VillageEnv(), SAFE_SPRINT_PATH_FOLLOW, _dusk(books=3), seed=0, gamma=GAMMA)
    assert outcome.delta_r > 0.9
    assert abs(outcome.delta_r - float(np.sum(outcome.delta_n))) > 0.5


def test_paired_outcome_serialises_to_json():
    outcome = paired_rollout(VillageEnv(), TRADE_MENU_HOTKEYS, dusk_state(), seed=0, gamma=GAMMA)
    record = json.loads(json.dumps(outcome.to_dict()))
    assert record["option_id"] == "trade_menu_hotkeys_v1"
    assert record["baseline_id"] == "protected_wait_v1"
    assert len(record["delta"]["motives"]) == 6


# ── Option groups behave as the scenario describes ──────────────────────────

@pytest.mark.parametrize("option", [TORCH_CORRIDOR, IRON_GOLEM_SPAWN, SWING_GATE_BARRICADE])
def test_defensive_options_raise_safety_and_infrastructure_at_dusk(option):
    _, delta_n = _mean_delta(option, dusk_state())
    assert delta_n[SAFETY] > 0.0
    assert delta_n[INFRASTRUCTURE] > 0.0


def test_the_golem_costs_sustainability():
    _, delta_n = _mean_delta(IRON_GOLEM_SPAWN, dusk_state())
    assert delta_n[SUSTAINABILITY] < 0.0


@pytest.mark.parametrize("option", [DISCOUNT_CHAIN, REPUTATION_FIRST])
def test_trade_options_raise_reputation_and_inventory_value(option):
    _, delta_n = _mean_delta(option, dusk_state())
    assert delta_n[REPUTATION] > 0.0
    assert delta_n[INVENTORY] > 0.0


def test_trade_hotkeys_only_add_deadline_slack_and_pass_cds():
    delta_r, delta_n = _mean_delta(TRADE_MENU_HOTKEYS, dusk_state())
    assert delta_n[DEADLINE] > 0.0
    assert np.allclose(np.delete(delta_n, DEADLINE), 0.0, atol=1e-6)
    assert CDSGate().admit(delta_r, delta_n)


def test_archer_kite_raises_safety_under_attack():
    delta_r, delta_n = _mean_delta(ARCHER_KITE, _night_state(hostiles=3))
    assert delta_n[SAFETY] > 0.0
    assert PDSGate(epsilon=0.1).admit(delta_r, delta_n)
