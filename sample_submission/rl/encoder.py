"""
Encode Observation and Option into fixed-size feature vectors for the RL model.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cg.api import Observation, Pokemon, OptionType, CardType, all_card_data

# Lazy card data cache
_CARD_TYPE: dict[int, CardType] = {}
_CARD_HP: dict[int, int] = {}
_CARD_STAGE: dict[int, int] = {}  # 0=basic, 1=stage1, 2=stage2
_CARD_ATTACK_DAMAGE: dict[int, float] = {}  # max attack damage

def _ensure_card_cache():
    if _CARD_TYPE:
        return
    from cg.api import all_card_data, all_attack
    attacks = {a.attackId: a for a in all_attack()}
    for cd in all_card_data():
        _CARD_TYPE[cd.cardId] = cd.cardType
        _CARD_HP[cd.cardId] = cd.hp
        stage = 0
        if cd.stage1:
            stage = 1
        elif cd.stage2:
            stage = 2
        _CARD_STAGE[cd.cardId] = stage
        max_dmg = 0
        for aid in cd.attacks:
            if aid in attacks:
                max_dmg = max(max_dmg, attacks[aid].damage)
        _CARD_ATTACK_DAMAGE[cd.cardId] = max_dmg / 300.0  # normalize


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
POKEMON_FEAT = 6   # hp_ratio, energy_count, is_basic, is_stage1, is_stage2, is_active_mon
BENCH_MAX = 5
STATE_DIM = (
    5               # global flags: turn, yourIndex, supporterPlayed, energyAttached, retreated
    + POKEMON_FEAT  # my active
    + BENCH_MAX * POKEMON_FEAT  # my bench
    + 3             # my hand_count, deck_count, prize_count
    + POKEMON_FEAT  # opp active
    + BENCH_MAX * POKEMON_FEAT  # opp bench
    + 3             # opp hand_count, deck_count, prize_count
)  # = 5 + 6 + 30 + 3 + 6 + 30 + 3 = 83

OPTION_TYPE_COUNT = 17  # OptionType max value + 1 (0..16)
OPTION_DIM = OPTION_TYPE_COUNT + 5  # type one-hot + (card_type_poke, card_type_item, card_type_supporter, attack_dmg, is_end)


def _encode_pokemon(poke: Pokemon | None, max_hp_hint: int = 300) -> list[float]:
    if poke is None:
        return [0.0] * POKEMON_FEAT
    hp_ratio = poke.hp / max(poke.maxHp, 1)
    energy_count = min(len(poke.energies), 6) / 6.0
    _ensure_card_cache()
    stage = _CARD_STAGE.get(poke.id, 0)
    return [
        hp_ratio,
        energy_count,
        float(stage == 0),
        float(stage == 1),
        float(stage == 2),
        _CARD_ATTACK_DAMAGE.get(poke.id, 0.0),
    ]


def encode_state(obs: Observation) -> list[float]:
    """Encode the current game state into a flat float list of length STATE_DIM."""
    _ensure_card_cache()
    state = obs.current
    if state is None:
        return [0.0] * STATE_DIM

    me = state.yourIndex
    opp = 1 - me
    mp = state.players[me]
    op = state.players[opp]

    feats: list[float] = []

    # Global
    feats.append(min(state.turn, 50) / 50.0)
    feats.append(float(me))
    feats.append(float(state.supporterPlayed))
    feats.append(float(state.energyAttached))
    feats.append(float(state.retreated))

    # My active
    my_active = mp.active[0] if mp.active else None
    feats.extend(_encode_pokemon(my_active))

    # My bench (pad to BENCH_MAX)
    for i in range(BENCH_MAX):
        p = mp.bench[i] if i < len(mp.bench) else None
        feats.extend(_encode_pokemon(p))

    # My counts
    feats.append(min(mp.handCount, 10) / 10.0)
    feats.append(mp.deckCount / 60.0)
    feats.append(len(mp.prize) / 6.0)

    # Opp active
    opp_active = op.active[0] if op.active else None
    feats.extend(_encode_pokemon(opp_active))

    # Opp bench
    for i in range(BENCH_MAX):
        p = op.bench[i] if i < len(op.bench) else None
        feats.extend(_encode_pokemon(p))

    # Opp counts
    feats.append(min(op.handCount, 10) / 10.0)
    feats.append(op.deckCount / 60.0)
    feats.append(len(op.prize) / 6.0)

    assert len(feats) == STATE_DIM, f"STATE_DIM mismatch: {len(feats)} vs {STATE_DIM}"
    return feats


def encode_option(opt, obs: Observation) -> list[float]:
    """Encode a single Option into a flat float list of length OPTION_DIM."""
    _ensure_card_cache()
    # type one-hot
    type_vec = [0.0] * OPTION_TYPE_COUNT
    t = int(opt.type)
    if 0 <= t < OPTION_TYPE_COUNT:
        type_vec[t] = 1.0

    # extra features
    is_poke = 0.0
    is_item = 0.0
    is_supporter = 0.0
    atk_dmg = 0.0
    is_end = 1.0 if opt.type == OptionType.END else 0.0

    # Try to get card info
    card_id = opt.cardId
    if card_id is None and opt.type == OptionType.PLAY:
        # look up from hand
        state = obs.current
        if state is not None:
            me = state.yourIndex
            hand = state.players[me].hand
            if hand is not None and opt.index is not None and opt.index < len(hand):
                card_id = hand[opt.index].id

    if card_id is not None:
        ct = _CARD_TYPE.get(card_id)
        if ct == CardType.POKEMON:
            is_poke = 1.0
        elif ct == CardType.ITEM:
            is_item = 1.0
        elif ct == CardType.SUPPORTER:
            is_supporter = 1.0
        atk_dmg = _CARD_ATTACK_DAMAGE.get(card_id, 0.0)

    if opt.type == OptionType.ATTACK and opt.attackId is not None:
        from cg.api import all_attack
        # Expensive but only called during encoding
        pass  # attack damage is approximated via card features above

    return type_vec + [is_poke, is_item, is_supporter, atk_dmg, is_end]
