"""Observation encoding and action masking for the Maskable PPO agent.

This module is SHARED between the training environment (tcg_rl/env.py) and the
submission-time inference in main.py. It must depend ONLY on numpy and cg.api so
that it can be bundled into the Kaggle submission without torch/gymnasium.

CRITICAL: The feature layout and constants here define the policy's input. If you
change anything that alters the vector length or the meaning of a dimension, any
previously trained policy.npz becomes invalid and must be retrained.

Perspective convention: features are always encoded from the point of view of the
player who must act now (state.yourIndex). "self" == yourIndex, "opp" == 1-yourIndex.
"""

from __future__ import annotations

import numpy as np

from cg.api import (
    AreaType,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    SelectType,
    all_attack,
    all_card_data,
)

# --------------------------------------------------------------------------- #
# Constants (changing these invalidates trained weights)
# --------------------------------------------------------------------------- #
MAX_OPTIONS = 50          # max real option slots; selections with more are heuristic-handled
STOP_ACTION = MAX_OPTIONS  # last action index = "submit current multi-select"
ACTION_DIM = MAX_OPTIONS + 1

POKE_SLOTS = 6            # 1 active + 5 bench, per player
NUM_CONTEXT = 49         # SelectContext ids 0..48 (unknown -> all zeros)
NUM_SELTYPE = 11         # SelectType ids 0..10
NUM_OPTTYPE = 17         # OptionType ids 0..16
NUM_AREA = 12            # AreaType ids 1..12
NUM_CARDTYPE = 7         # CardType ids 0..6
NUM_ENERGY = 12          # EnergyType ids 0..11

# normalization scales (rough upper bounds)
_TURN_SCALE = 40.0
_HP_SCALE = 340.0
_ENERGY_SCALE = 6.0
_DECK_SCALE = 60.0
_HAND_SCALE = 15.0
_DMG_SCALE = 300.0

PER_POKE = 22            # see _encode_pokemon
PER_OPTION = 44          # see _encode_option

# --------------------------------------------------------------------------- #
# Static card / attack database (loaded once, cached)
# --------------------------------------------------------------------------- #
_CARD_DB: dict[int, object] | None = None
_ATTACK_DB: dict[int, object] | None = None


def _card_db() -> dict[int, object]:
    global _CARD_DB
    if _CARD_DB is None:
        _CARD_DB = {c.cardId: c for c in all_card_data()}
    return _CARD_DB


def _attack_db() -> dict[int, object]:
    global _ATTACK_DB
    if _ATTACK_DB is None:
        _ATTACK_DB = {a.attackId: a for a in all_attack()}
    return _ATTACK_DB


# --------------------------------------------------------------------------- #
# Feature dimension (computed from constants; asserted at import via self-test)
# --------------------------------------------------------------------------- #
_GLOBAL_DIM = 19
_SELECT_DIM = NUM_CONTEXT + NUM_SELTYPE + 5
OBS_DIM = (
    _GLOBAL_DIM
    + 2 * POKE_SLOTS * PER_POKE
    + 10                       # special conditions
    + _SELECT_DIM
    + ACTION_DIM * PER_OPTION
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _onehot(idx, size: int) -> list[float]:
    v = [0.0] * size
    if idx is not None and 0 <= int(idx) < size:
        v[int(idx)] = 1.0
    return v


def _safe(x, default=0.0):
    return default if x is None else x


def _area_list(pstate, area):
    """Return the list of cards/pokemon for a given AreaType in a PlayerState."""
    if pstate is None or area is None:
        return []
    a = int(area)
    if a == int(AreaType.HAND):
        return pstate.hand or []
    if a == int(AreaType.ACTIVE):
        return pstate.active or []
    if a == int(AreaType.BENCH):
        return pstate.bench or []
    if a == int(AreaType.DISCARD):
        return pstate.discard or []
    return []


def _resolve_card_id(state, actor_index: int, opt) -> int:
    """Best-effort resolution of the card id an option refers to (0 = unknown)."""
    try:
        if opt.cardId:
            return int(opt.cardId)
        ot = int(opt.type)
        if ot == int(OptionType.PLAY):
            hand = state.players[actor_index].hand or []
            if opt.index is not None and 0 <= opt.index < len(hand):
                return int(hand[opt.index].id)
            return 0
        # options that reference a card located in an area
        pi = opt.playerIndex if opt.playerIndex is not None else actor_index
        area = opt.area
        idx = opt.index
        if ot in (int(OptionType.ATTACK), int(OptionType.RETREAT)):
            # refers to the active pokemon implicitly
            act = state.players[actor_index].active or []
            if act and act[0] is not None:
                return int(act[0].id)
            return 0
        if area is not None and idx is not None and 0 <= pi < len(state.players):
            lst = _area_list(state.players[pi], area)
            if 0 <= idx < len(lst) and lst[idx] is not None:
                return int(lst[idx].id)
    except Exception:
        return 0
    return 0


# --------------------------------------------------------------------------- #
# Pokemon encoding
# --------------------------------------------------------------------------- #
def _encode_pokemon(poke) -> list[float]:
    if poke is None:
        return [0.0] * PER_POKE
    card = _card_db().get(int(poke.id))
    max_hp = poke.maxHp if poke.maxHp else (poke.hp if poke.hp else 1)
    hp_ratio = (poke.hp / max_hp) if max_hp else 0.0
    feats = [
        1.0,
        float(np.clip(hp_ratio, 0.0, 1.0)),
        float(np.clip(max_hp / _HP_SCALE, 0.0, 1.5)),
        float(np.clip(len(poke.energies or []) / _ENERGY_SCALE, 0.0, 1.5)),
        float(np.clip(len(poke.tools or []) / 2.0, 0.0, 1.0)),
        1.0 if poke.appearThisTurn else 0.0,
    ]
    if card is not None:
        feats += [
            1.0 if card.basic else 0.0,
            1.0 if card.stage1 else 0.0,
            1.0 if card.stage2 else 0.0,
            1.0 if card.ex else 0.0,
        ]
        feats += _onehot(int(card.energyType) if card.energyType is not None else None, NUM_ENERGY)
    else:
        feats += [0.0, 0.0, 0.0, 0.0] + [0.0] * NUM_ENERGY
    return feats


# --------------------------------------------------------------------------- #
# Option encoding
# --------------------------------------------------------------------------- #
def _encode_option(state, actor_index: int, opt, in_buffer: bool, valid: bool,
                   is_stop: bool) -> list[float]:
    if is_stop:
        feats = [0.0] * NUM_OPTTYPE
        feats += [1.0, 1.0 if in_buffer else 0.0, 1.0 if valid else 0.0]
        feats += [0.0, 0.0]                       # targets self/opp
        feats += [0.0] * NUM_AREA
        feats += [0.0] * NUM_CARDTYPE
        feats += [0.0, 0.0, 0.0]                  # is_ex, dmg, cost
        return feats
    if opt is None:                                # padding slot
        return [0.0] * PER_OPTION

    ot = int(opt.type)
    feats = _onehot(ot, NUM_OPTTYPE)
    feats += [0.0, 1.0 if in_buffer else 0.0, 1.0 if valid else 0.0]  # is_stop, in_buffer, valid

    pi = opt.playerIndex if opt.playerIndex is not None else actor_index
    feats += [1.0 if pi == actor_index else 0.0, 1.0 if pi != actor_index else 0.0]

    area = opt.area if opt.area is not None else opt.inPlayArea
    feats += _onehot((int(area) - 1) if area is not None else None, NUM_AREA)

    cid = _resolve_card_id(state, actor_index, opt)
    card = _card_db().get(cid)
    feats += _onehot(int(card.cardType) if card is not None else None, NUM_CARDTYPE)
    feats += [1.0 if (card is not None and card.ex) else 0.0]

    dmg = 0.0
    cost = 0.0
    if ot == int(OptionType.ATTACK) and opt.attackId is not None:
        atk = _attack_db().get(int(opt.attackId))
        if atk is not None:
            dmg = np.clip(_safe(atk.damage) / _DMG_SCALE, 0.0, 1.5)
            cost = np.clip(len(atk.energies or []) / _ENERGY_SCALE, 0.0, 1.0)
    feats += [float(dmg), float(cost)]
    return feats


# --------------------------------------------------------------------------- #
# Action mask
# --------------------------------------------------------------------------- #
def action_mask(obs: Observation, buffer: list[int]) -> np.ndarray:
    """Boolean mask of length ACTION_DIM. True = legal action right now."""
    mask = np.zeros(ACTION_DIM, dtype=bool)
    sel = obs.select
    if sel is None:
        return mask
    n = min(len(sel.option), MAX_OPTIONS)
    bset = set(buffer)
    # may still pick options only while below maxCount
    if len(buffer) < sel.maxCount:
        for i in range(n):
            if i not in bset:
                mask[i] = True
    # STOP is legal once we have selected at least minCount
    if len(buffer) >= sel.minCount:
        mask[STOP_ACTION] = True
    # if nothing is selectable (e.g., minCount==0 and no options), allow STOP
    if not mask.any():
        mask[STOP_ACTION] = True
    return mask


# --------------------------------------------------------------------------- #
# Full observation encoding
# --------------------------------------------------------------------------- #
def encode_obs(obs: Observation, buffer: list[int]) -> np.ndarray:
    """Encode an Observation (+ current multi-select buffer) into a fixed vector."""
    feats: list[float] = []
    state = obs.current
    sel = obs.select

    if state is None:
        return np.zeros(OBS_DIM, dtype=np.float32)

    me = int(state.yourIndex)
    opp = 1 - me
    pme = state.players[me]
    popp = state.players[opp]

    # ---- global block (19) ----
    feats += [
        float(np.clip(_safe(state.turn) / _TURN_SCALE, 0.0, 1.5)),
        float(np.clip(_safe(state.turnActionCount) / 15.0, 0.0, 1.5)),
        1.0 if state.supporterPlayed else 0.0,
        1.0 if state.stadiumPlayed else 0.0,
        1.0 if state.energyAttached else 0.0,
        1.0 if state.retreated else 0.0,
        float(np.clip(len(pme.prize) / 6.0, 0.0, 1.0)),
        float(np.clip(len(popp.prize) / 6.0, 0.0, 1.0)),
        float(np.clip(_safe(pme.deckCount) / _DECK_SCALE, 0.0, 1.0)),
        float(np.clip(_safe(popp.deckCount) / _DECK_SCALE, 0.0, 1.0)),
        float(np.clip(_safe(pme.handCount) / _HAND_SCALE, 0.0, 1.0)),
        float(np.clip(_safe(popp.handCount) / _HAND_SCALE, 0.0, 1.0)),
        float(np.clip(len(pme.bench) / 5.0, 0.0, 1.0)),
        float(np.clip(len(popp.bench) / 5.0, 0.0, 1.0)),
        float(np.clip(len(pme.discard) / _DECK_SCALE, 0.0, 1.0)),
        float(np.clip(len(popp.discard) / _DECK_SCALE, 0.0, 1.0)),
        1.0 if state.stadium else 0.0,
        1.0 if (pme.active and pme.active[0] is not None) else 0.0,
        1.0 if (popp.active and popp.active[0] is not None) else 0.0,
    ]

    # ---- self pokemon (6 slots) ----
    def pokemons(pstate):
        active = (pstate.active[0] if pstate.active else None)
        bench = list(pstate.bench or [])
        slots = [active] + bench
        slots = slots[:POKE_SLOTS]
        while len(slots) < POKE_SLOTS:
            slots.append(None)
        return slots

    for poke in pokemons(pme):
        feats += _encode_pokemon(poke)
    for poke in pokemons(popp):
        feats += _encode_pokemon(poke)

    # ---- special conditions (10) ----
    feats += [
        1.0 if pme.poisoned else 0.0,
        1.0 if pme.burned else 0.0,
        1.0 if pme.asleep else 0.0,
        1.0 if pme.paralyzed else 0.0,
        1.0 if pme.confused else 0.0,
        1.0 if popp.poisoned else 0.0,
        1.0 if popp.burned else 0.0,
        1.0 if popp.asleep else 0.0,
        1.0 if popp.paralyzed else 0.0,
        1.0 if popp.confused else 0.0,
    ]

    # ---- select block ----
    if sel is not None:
        feats += _onehot(int(sel.context), NUM_CONTEXT)
        feats += _onehot(int(sel.type), NUM_SELTYPE)
        feats += [
            float(np.clip(sel.minCount / 4.0, 0.0, 1.0)),
            float(np.clip(sel.maxCount / 4.0, 0.0, 1.0)),
            float(np.clip(_safe(sel.remainEnergyCost) / _ENERGY_SCALE, 0.0, 1.0)),
            float(np.clip(_safe(sel.remainDamageCounter) / 20.0, 0.0, 1.0)),
            float(np.clip(len(buffer) / 4.0, 0.0, 1.0)),
        ]
    else:
        feats += [0.0] * _SELECT_DIM

    # ---- options block (MAX_OPTIONS + STOP) ----
    bset = set(buffer)
    mask = action_mask(obs, buffer) if sel is not None else np.zeros(ACTION_DIM, dtype=bool)
    n = min(len(sel.option), MAX_OPTIONS) if sel is not None else 0
    for i in range(MAX_OPTIONS):
        if i < n:
            opt = sel.option[i]
            feats += _encode_option(state, me, opt, in_buffer=(i in bset),
                                    valid=bool(mask[i]), is_stop=False)
        else:
            feats += _encode_option(state, me, None, in_buffer=False, valid=False, is_stop=False)
    # STOP slot
    feats += _encode_option(state, me, None, in_buffer=False,
                            valid=bool(mask[STOP_ACTION]) if sel is not None else False,
                            is_stop=True)

    arr = np.asarray(feats, dtype=np.float32)
    if arr.shape[0] != OBS_DIM:  # pragma: no cover - guards against drift
        raise ValueError(f"feature length {arr.shape[0]} != OBS_DIM {OBS_DIM}")
    return arr


def _self_test() -> None:
    """Lightweight import-time sanity check on dimensions."""
    assert PER_POKE == 6 + 4 + NUM_ENERGY, PER_POKE
    assert PER_OPTION == NUM_OPTTYPE + 3 + 2 + NUM_AREA + NUM_CARDTYPE + 3, PER_OPTION


_self_test()
