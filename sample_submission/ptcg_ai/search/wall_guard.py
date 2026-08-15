"""Conservative, opt-in guards for "wall" matchups where the opponent's Active Pokemon (or a
whole line of Pokemon, e.g. Crustle's "Mysterious Rock Inn") prevents all damage from our
current attacker(s). Mirrors the calling-convention style of search/attack_plan.py (config
dict with an "enabled" gate, obs/state-driven, defensive against exceptions), but needs none
of attack_plan's search_begin/search_step machinery: everything this module reads (our own
bench, the opponent's Active + bench, energy attached to any of them) is already visible in
obs.current — Pokemon in play are public information in this game (only hand is hidden), so
there is nothing hidden to simulate here.

Damage/immunity itself is never recomputed here: every damage number goes through
board_evaluation.attack_features.resolve_damage (which itself calls damage_prevented), so any
wall ability that resolve_damage already understands generalizes to this module for free (no
card ID for a defending wall is hardcoded anywhere below).

Guard A ("don't stand there swinging for zero" — guard_a_retreat): if attacking with our
current Active is one of the choices on the table this decision but would deal 0 resolved
damage no matter which attack is picked, while a benched Pokemon of ours could deal >0 to the
opponent's Active with the energy it already has attached, and retreating is legal, retreat to
that benched Pokemon instead. Runs as a pre-step (called from ml_policy_agent.py at the same
point as Guard B, before lethal_search/pipeline/the policy model) rather than as a post-check
on whatever the model/pipeline would have chosen: `_try_pipeline` returns its own action before
that baseline is ever computed, so a post-check there would never get a turn — confirmed
empirically (0/100 real games) with an earlier version of this guard before it was moved here.

Guard B ("go around the wall" — guard_b_boss_orders): if nothing we have (our Active, or any
benched Pokemon of ours swapped in) can dent the opponent's Active at all, but something of
ours could dent one of the opponent's *benched* Pokemon, and Boss's Orders (1182) is playable
this turn, play it targeting the highest-prize-value (ties: highest-damage) such Pokemon.

Both guards can only supply the *first* half of what is, on the wire, a two-step sequence:
    RETREAT -> [discard energy for the cost] -> SWITCH/CARD select (pick the bench Pokemon)
    PLAY Boss's Orders -> SWITCH/CARD select (pick the opponent's bench Pokemon)
The engine asks for the destination in a separate select on a *later* call to agent() — this
was confirmed empirically (see scratchpad probe_retreat_boss.py) rather than assumed:
choosing RETREAT in a MAIN select is always followed by a DISCARD_ENERGY/ENERGY select (to pay
the retreat cost) and *then* a SWITCH/CARD select whose options resolve, via
(AreaType.BENCH, playerIndex, index), to the candidate bench Pokemon; playing a PLAY option
that resolves (via the hand) to Boss's Orders is followed directly by a SWITCH/CARD select
built the same way, over the *opponent's* bench. `_pending_target_serial` remembers which
Pokemon (by Pokemon.serial, unique per card for the whole match) a guard intended to land on;
`try_consume_pending_target` fills it in the moment that later select appears.
"""
from __future__ import annotations

from cg.api import AreaType, Observation, OptionType, Pokemon, SelectContext, SelectType, State

from ptcg_ai.board_evaluation import attack_features, energy_requirements
from ptcg_ai.shared import card_cache

DEFAULTS = {"enabled": False}

# Generic gate condition ("is Boss's Orders playable"), not a damage/immunity special-case —
# the damage math below never hardcodes a defending Pokemon's card ID (per the task's
# generalization requirement), only this one offensive card that both guards route through.
BOSS_ORDERS_CARD_ID = 1182

# Remembers which Pokemon (by unique Pokemon.serial) a guard-initiated RETREAT / Boss's Orders
# should land on, across the engine's later SWITCH/CARD select. Reset at match boundaries by
# ml_policy_agent.agent() (obs.select is None) the same way `_selects_seen`/`_match_start_perf`
# are — this module is otherwise stateless.
_pending_target_serial: int | None = None

# Lightweight firing-rate counters, mirroring search/attack_plan.py's `_stats`/`get_stats`
# pattern. Purely observational (no effect on decisions); used to measure how often each guard's
# trigger condition is actually matched in practice, separate from win/loss outcomes.
_ZERO_STATS = {
    "guard_a_checks": 0, "guard_a_fired": 0,
    "guard_b_checks": 0, "guard_b_fired": 0,
    "pending_checks": 0, "pending_consumed": 0, "pending_dropped": 0,
}
_stats = dict(_ZERO_STATS)


def reset_stats() -> None:
    _stats.update(_ZERO_STATS)


def get_stats() -> dict:
    return dict(_stats)


def reset_pending_target() -> None:
    global _pending_target_serial
    _pending_target_serial = None


def _prize_value(card_id: int) -> int:
    """CardData.megaEx -> 3 prizes, CardData.ex -> 2, else -> 1 (standard Knocked Out payout)."""
    card = card_cache.get_card(card_id)
    if getattr(card, "megaEx", False):
        return 3
    if getattr(card, "ex", False):
        return 2
    return 1


def _side(active: Pokemon | None, bench: list) -> list:
    return [active, *[p for p in bench if p is not None]]


def _side_with_first(active: Pokemon | None, bench: list, first_mon: Pokemon) -> list:
    """attacker_side_pokemon with `first_mon` in the "battle spot" slot (index 0) — required
    because some board-dependent variable-damage attacks (attack_features._board_dependent_
    pattern's "active_both" kind) read index 0 as "the attacker's Active", which is exactly
    what `first_mon` would become if we actually retreated to it."""
    roster = [p for p in (active, *bench) if p is not None and p is not first_mon]
    return [first_mon, *roster]


def _our_context(state: State, your_index: int):
    me = state.players[your_index]
    active = me.active[0] if me.active else None
    bench = list(me.bench or [])
    return me, active, bench


def _opponent_context(state: State, your_index: int):
    opp = state.players[1 - your_index]
    opp_active = opp.active[0] if opp.active else None
    opp_bench = list(opp.bench or [])
    return opp_active, opp_bench, _side(opp_active, opp_bench)


def _active_attack_damage(select_options, attacker, attacker_side, defender, defender_card,
                           defender_side, defender_is_benched, defender_active) -> int:
    """Max resolved damage across `attacker`'s currently *legal* ATTACK options (as offered by
    the real select — this already accounts for energy, status conditions, etc.) vs `defender`.
    """
    best = 0
    for option in select_options:
        if option.type != OptionType.ATTACK or option.attackId is None:
            continue
        attack = card_cache.get_attack(option.attackId)
        damage = attack_features.resolve_damage(
            attack, attacker, defender_card.weakness, defender_card.resistance,
            defender=defender, defender_side_pokemon=defender_side,
            defender_is_benched=defender_is_benched,
            damage_is_effect=attack_features.damage_is_effect_based(attack),
            attacker_side_pokemon=attacker_side, defender_active_pokemon=defender_active,
        )
        best = max(best, damage)
    return best


def _hypothetical_damage(mon: Pokemon, attacker_side, defender, defender_card, defender_side,
                          defender_is_benched, defender_active) -> int:
    """Max resolved damage `mon` could deal *if* it were our Active, using only attacks its
    currently-attached energy affords (same affordability check as
    rule_based/main_turn_parts/priorities/retreat.py's retreat-cost check, applied to attack
    cost instead). Used for benched Pokemon, which cannot attack this turn but could next turn
    if we retreated to them."""
    best = 0
    card = card_cache.get_card(mon.id)
    for attack_id in card.attacks:
        attack = card_cache.get_attack(attack_id)
        if not energy_requirements.is_energy_sufficient(attack, mon.energies or []):
            continue
        damage = attack_features.resolve_damage(
            attack, mon, defender_card.weakness, defender_card.resistance,
            defender=defender, defender_side_pokemon=defender_side,
            defender_is_benched=defender_is_benched,
            damage_is_effect=attack_features.damage_is_effect_based(attack),
            attacker_side_pokemon=attacker_side, defender_active_pokemon=defender_active,
        )
        best = max(best, damage)
    return best


def guard_a_retreat(obs: Observation, config: dict | None) -> list[int] | None:
    """If attacking with our current Active is on the table this decision (the MAIN select
    offers at least one ATTACK option) but would deal 0 resolved damage no matter which such
    option is picked, while some benched Pokemon of ours could hit the opponent's Active for >0
    with its current energy, and retreating is legal (State.retreated is False and the retreat
    cost is covered), return the RETREAT option's index instead (and remember which bench
    Pokemon to land on for `try_consume_pending_target`). Returns None if the guard doesn't
    apply.

    Deliberately does *not* take a `chosen_action` / baseline decision as input (an earlier
    version did, mirroring attack_plan.py's post-baseline-check style): `_try_pipeline` in
    ml_policy_agent.py returns its own action *before* the model/baseline_action code path ever
    runs, so a post-baseline check here would never fire whenever pipeline succeeds — confirmed
    empirically (0/100 real games) before this was reworked into a pre-step, called at the same
    point as guard_b_boss_orders, that only needs the select actually on the table right now."""
    cfg = {**DEFAULTS, **(config or {})}
    if not cfg.get("enabled", False):
        return None
    global _pending_target_serial
    if _pending_target_serial is not None:
        return None  # a previous guard action hasn't been resolved yet

    state = obs.current
    select = obs.select
    if state is None or select is None or select.type != SelectType.MAIN:
        return None
    if not any(option.type == OptionType.ATTACK for option in select.option):
        return None  # attacking isn't actually one of the choices on the table
    _stats["guard_a_checks"] += 1

    your_index = state.yourIndex
    _me, active, bench = _our_context(state, your_index)
    if active is None:
        return None
    opp_active, _opp_bench, opp_side = _opponent_context(state, your_index)
    if opp_active is None:
        return None
    opp_card = card_cache.get_card(opp_active.id)

    own_side = _side(active, bench)
    active_now_best = _active_attack_damage(
        select.option, active, own_side, opp_active, opp_card, opp_side, False, opp_active,
    )
    if active_now_best > 0:
        return None  # not actually swinging for zero

    if state.retreated:
        return None
    active_card = card_cache.get_card(active.id)
    if not energy_requirements.can_afford_retreat(active_card.retreatCost, active.energies or []):
        return None
    retreat_index = next(
        (i for i, option in enumerate(select.option) if option.type == OptionType.RETREAT), None
    )
    if retreat_index is None:
        return None  # the engine never offered a retreat at this decision point

    best_mon, best_damage = None, 0
    for mon in bench:
        if mon is None:
            continue
        side = _side_with_first(active, bench, mon)
        damage = _hypothetical_damage(mon, side, opp_active, opp_card, opp_side, False, opp_active)
        if damage > best_damage:
            best_mon, best_damage = mon, damage
    if best_mon is None:
        return None

    _pending_target_serial = best_mon.serial
    _stats["guard_a_fired"] += 1
    return [retreat_index]


def guard_b_boss_orders(obs: Observation, config: dict | None) -> list[int] | None:
    """If neither our Active nor any benched Pokemon of ours (swapped in) can dent the
    opponent's Active, but something of ours could dent one of the opponent's benched Pokemon,
    and Boss's Orders is playable this turn, return the index of the PLAY option that plays it
    (and remember the best target for `try_consume_pending_target`). Returns None if the guard
    doesn't apply."""
    cfg = {**DEFAULTS, **(config or {})}
    if not cfg.get("enabled", False):
        return None
    global _pending_target_serial
    if _pending_target_serial is not None:
        return None

    state = obs.current
    select = obs.select
    if state is None or select is None or select.type != SelectType.MAIN:
        return None
    if state.supporterPlayed:
        return None

    your_index = state.yourIndex
    me, active, bench = _our_context(state, your_index)
    if active is None:
        return None
    opp_active, opp_bench, opp_side = _opponent_context(state, your_index)
    if opp_active is None or not opp_bench:
        return None

    hand = me.hand or []
    boss_index = None
    for i, option in enumerate(select.option):
        if option.type != OptionType.PLAY or option.index is None:
            continue
        if not (0 <= option.index < len(hand)):
            continue
        if hand[option.index].id == BOSS_ORDERS_CARD_ID:
            boss_index = i
            break
    if boss_index is None:
        return None
    _stats["guard_b_checks"] += 1  # Boss's Orders is playable this turn

    own_side = _side(active, bench)
    opp_active_card = card_cache.get_card(opp_active.id)
    active_now_best = _active_attack_damage(
        select.option, active, own_side, opp_active, opp_active_card, opp_side, False, opp_active,
    )
    bench_vs_active_best = 0
    for mon in bench:
        if mon is None:
            continue
        side = _side_with_first(active, bench, mon)
        bench_vs_active_best = max(bench_vs_active_best, _hypothetical_damage(
            mon, side, opp_active, opp_active_card, opp_side, False, opp_active,
        ))
    if max(active_now_best, bench_vs_active_best) > 0:
        return None  # the opponent's Active isn't actually walling all of our attackers

    active_attack_ids = [
        option.attackId for option in select.option
        if option.type == OptionType.ATTACK and option.attackId is not None
    ]
    best_target, best_score = None, None
    for mon in opp_bench:
        if mon is None:
            continue
        mon_card = card_cache.get_card(mon.id)
        best_damage = 0
        for attack_id in active_attack_ids:
            attack = card_cache.get_attack(attack_id)
            best_damage = max(best_damage, attack_features.resolve_damage(
                attack, active, mon_card.weakness, mon_card.resistance,
                defender=mon, defender_side_pokemon=opp_side, defender_is_benched=True,
                damage_is_effect=attack_features.damage_is_effect_based(attack),
                attacker_side_pokemon=own_side, defender_active_pokemon=opp_active,
            ))
        for bmon in bench:
            if bmon is None:
                continue
            side = _side_with_first(active, bench, bmon)
            best_damage = max(best_damage, _hypothetical_damage(
                bmon, side, mon, mon_card, opp_side, True, opp_active,
            ))
        if best_damage <= 0:
            continue
        score = (_prize_value(mon.id), best_damage)
        if best_score is None or score > best_score:
            best_target, best_score = mon, score

    if best_target is None:
        return None

    _pending_target_serial = best_target.serial
    _stats["guard_b_fired"] += 1
    return [boss_index]


def try_consume_pending_target(obs: Observation, config: dict | None) -> list[int] | None:
    """Fill in the destination of a guard-initiated RETREAT/Boss's Orders once the engine's
    SWITCH/CARD select for it appears (a later, separate call to agent()). Returns None (and
    leaves the pending target untouched) on any select that isn't it, e.g. the DISCARD_ENERGY
    select for paying the retreat cost."""
    cfg = {**DEFAULTS, **(config or {})}
    global _pending_target_serial
    if _pending_target_serial is None or not cfg.get("enabled", False):
        return None

    select = obs.select
    state = obs.current
    if select is None or state is None:
        return None
    if select.context == SelectContext.MAIN:
        # Back at a MAIN decision without ever matching: the switch/target phase concluded one
        # way or another. Don't let a stale target leak into an unrelated future switch.
        _pending_target_serial = None
        return None
    if select.context != SelectContext.SWITCH or select.type != SelectType.CARD:
        return None
    _stats["pending_checks"] += 1

    for i, option in enumerate(select.option):
        if option.area != AreaType.BENCH or option.playerIndex is None or option.index is None:
            continue
        if not (0 <= option.playerIndex < len(state.players)):
            continue
        bench = state.players[option.playerIndex].bench or []
        if not (0 <= option.index < len(bench)):
            continue
        mon = bench[option.index]
        if mon is not None and mon.serial == _pending_target_serial:
            _pending_target_serial = None
            _stats["pending_consumed"] += 1
            return [i]

    _pending_target_serial = None
    _stats["pending_dropped"] += 1
    return None
