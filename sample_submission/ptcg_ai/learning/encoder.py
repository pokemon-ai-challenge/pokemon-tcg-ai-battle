"""状態エンコーダ（バリューネットワーク Step1）.

Observation(現在盤面)を固定長・決定的な特徴ベクトルへ変換する。全 ML パイプラインの
共通土台であり、学習(リプレイ JSON)とランタイム(obs_dict)を同一コードパスで扱う。

設計書: ``sample_submission/docs/plans/ml-value-network/design.md`` の §3。

重要な制約:
- **公開情報のみ**を使う。自分の手札の中身は集約特徴として使ってよいが、相手は handCount のみ。
  相手の非公開情報(手札の中身・山札の並び・伏せカードの中身)は一切使わない。
- **``obs.logs`` / ``obs.select`` に依存しない**。学習データでは logs が削除されているため、
  logs が空/欠損でも同一の出力になること。current 盤面のみからエンコードする。
- **固定長・決定的**。同じ入力には常に同じ長さ・同じ値を返す。ベンチは固定スロット数で
  ゼロ埋めし、active が None(伏せ)でも壊れない。
- **視点正規化**。``State.yourIndex`` を基準に自分/相手へ並べ替える(§3.2)。

学習側から使う場合は :func:`encode_obs_dict` を呼ぶ(dict → Observation → ベクトル)。
実行時(agent)は既に Observation を持っているので :func:`encode_state` を直接呼べる。
"""

from __future__ import annotations

from cg.api import (
    AreaType,
    CardType,
    Observation,
    Option,
    OptionType,
    Pokemon,
    SelectData,
    SelectType,
    State,
    to_observation_class,
)

from ptcg_ai.board_evaluation import attack_features, board_features, energy_requirements
from ptcg_ai.shared import card_cache

# ベンチ上限。cg/api.py の PlayerState.benchMax は実データで 5。固定スロット数として扱い、
# 実際の benchMax がこれと異なっても先頭 BENCH_SLOTS 枠までを見る(超過分は無視)。
BENCH_SLOTS = 5
# active(1) + ベンチ固定スロット。
POKEMON_SLOTS = 1 + BENCH_SLOTS

# エネルギー不足数のクリップ上限(技が撃てない/攻撃を持たない場合の既定値)。
_MAX_SHORTFALL = 10.0

# ポケモン1体あたりの特徴名(present は存在フラグ)。
_POKEMON_FEATURE_NAMES = [
    "present",
    "hp_ratio",
    "remaining_hp",
    "damage_counters",
    "energy_count",
    "best_attack_damage",
    "can_ko_defender",
    "min_energy_shortfall",
    "has_ready_attack",
    "attacker_score",
    "is_likely_ko_next_turn",
]

# ゼロ埋め用(存在しないスロット)。
_ZERO_POKEMON = [0.0] * len(_POKEMON_FEATURE_NAMES)


def _build_feature_names() -> list[str]:
    names: list[str] = []

    # --- ポケモン毎(自分 → 相手、それぞれ active + ベンチ固定スロット) ---
    for side in ("self", "opp"):
        for slot in ["active"] + [f"bench{i}" for i in range(BENCH_SLOTS)]:
            for feat in _POKEMON_FEATURE_NAMES:
                names.append(f"{side}_{slot}_{feat}")

    # --- カウント系: 自分(内訳あり) ---
    names += [
        "self_hand_count",
        "self_hand_pokemon",
        "self_hand_trainer",
        "self_hand_energy",
        "self_deck_count",
        "self_prize_remaining",
        "self_discard_count",
        "self_bench_count",
    ]
    # --- カウント系: 相手(枚数のみ) ---
    names += [
        "opp_hand_count",
        "opp_deck_count",
        "opp_prize_remaining",
        "opp_discard_count",
        "opp_bench_count",
    ]

    # --- 特殊状態(バトル場、公開情報) ---
    for side in ("self", "opp"):
        for cond in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
            names.append(f"{side}_{cond}")

    # --- ゲーム進行 ---
    names += [
        "turn",
        "is_first_player",
        "supporter_played",
        "stadium_played",
        "energy_attached",
        "retreated",
        "has_stadium",
    ]

    # --- 集約 ---
    names += [
        "prize_diff",
        "bench_count_diff",
        "self_energy_on_board",
        "opp_energy_on_board",
    ]

    return names


#: 特徴ベクトルの各次元の名前。ベクトルと同じ順序・同じ長さ(``extra_features`` を除く)。
#: 学習側で係数の解釈に使う。
FEATURE_NAMES: list[str] = _build_feature_names()

#: 基本特徴ベクトルの長さ(``extra_features`` を含まない)。
BASE_FEATURE_COUNT: int = len(FEATURE_NAMES)


def _card_or_none(card_id: int):
    """card_cache から CardData を安全に引く(未知IDなら None)。"""
    try:
        return card_cache.get_card(card_id)
    except Exception:  # noqa: BLE001 - 未知IDでも決定的にゼロ埋めへフォールバック
        return None


def _attack_or_none(attack_id: int):
    try:
        return card_cache.get_attack(attack_id)
    except Exception:  # noqa: BLE001
        return None


def _pokemon_features(
    pokemon: Pokemon | None,
    defender: Pokemon | None,
    attacker_hand_size: int,
    state: State,
    owner_index: int,
) -> list[float]:
    """ポケモン1体を特徴ベクトル化する(None/伏せはゼロ埋め)。

    Args:
        pokemon: 対象ポケモン(None/伏せなら全 0)。
        defender: 打点計算の相手(通常は敵バトル場)。None なら弱点/抵抗力なしで素点。
        attacker_hand_size: 可変ダメージ推定用の攻撃側手札枚数(public な handCount)。
        state: 盤面(is_likely_ko_next_turn 用)。
        owner_index: この pokemon の所有プレイヤーの index。
    """
    if pokemon is None:
        return list(_ZERO_POKEMON)

    max_hp = pokemon.maxHp or 0
    hp_ratio = (pokemon.hp / max_hp) if max_hp else 0.0
    remaining_hp = float(pokemon.hp)
    damage_counters = float(max(0, max_hp - pokemon.hp)) / 10.0
    energy_count = float(len(pokemon.energies or []))

    defender_weakness = None
    defender_resistance = None
    if defender is not None:
        def_card = _card_or_none(defender.id)
        if def_card is not None:
            defender_weakness = def_card.weakness
            defender_resistance = def_card.resistance

    best_damage = 0.0
    can_ko = 0.0
    min_shortfall = _MAX_SHORTFALL
    has_ready = 0.0

    card = _card_or_none(pokemon.id)
    if card is not None:
        for attack_id in card.attacks or []:
            attack = _attack_or_none(attack_id)
            if attack is None:
                continue
            shortfall = energy_requirements.energy_shortfall(attack, pokemon.energies or [])
            shortfall_sum = float(sum(shortfall.values()))
            min_shortfall = min(min_shortfall, shortfall_sum)
            if not shortfall:
                has_ready = 1.0
            damage_is_effect = attack_features.damage_is_effect_based(attack)
            damage = attack_features.resolve_damage(
                attack, pokemon, defender_weakness, defender_resistance, attacker_hand_size,
                defender=defender, defender_is_benched=False, damage_is_effect=damage_is_effect,
            )
            best_damage = max(best_damage, float(damage))
            if defender is not None and attack_features.can_ko(
                attack, pokemon, defender, defender_weakness, defender_resistance, attacker_hand_size,
                defender_is_benched=False, damage_is_effect=damage_is_effect,
            ):
                can_ko = 1.0

    min_shortfall = min(min_shortfall, _MAX_SHORTFALL)

    attacker_score = board_features.attacker_score(pokemon)
    likely_ko = 1.0 if board_features.is_likely_ko_next_turn(pokemon, state, owner_index) else 0.0

    return [
        1.0,  # present
        hp_ratio,
        remaining_hp,
        damage_counters,
        energy_count,
        best_damage,
        can_ko,
        min_shortfall,
        has_ready,
        attacker_score,
        likely_ko,
    ]


def _active_pokemon(player) -> Pokemon | None:
    """PlayerState.active(サイズ 0 or 1)から Pokemon か None を取り出す。"""
    active = player.active or []
    if len(active) == 0:
        return None
    return active[0]


def _side_pokemon_block(
    player,
    defender: Pokemon | None,
    attacker_hand_size: int,
    state: State,
    owner_index: int,
) -> list[float]:
    """1プレイヤー分のポケモン特徴(active + ベンチ固定スロット)を連結する。"""
    feats: list[float] = []
    feats += _pokemon_features(_active_pokemon(player), defender, attacker_hand_size, state, owner_index)

    bench = player.bench or []
    for i in range(BENCH_SLOTS):
        pkmn = bench[i] if i < len(bench) else None
        feats += _pokemon_features(pkmn, defender, attacker_hand_size, state, owner_index)
    return feats


def _hand_breakdown(hand) -> tuple[float, float, float]:
    """自分の手札(list[Card])を ポケモン/トレーナーズ/エネルギー の枚数へ集約する。"""
    pokemon = trainer = energy = 0
    for card in hand or []:
        c = _card_or_none(card.id)
        if c is None:
            continue
        ct = c.cardType
        if ct == CardType.POKEMON:
            pokemon += 1
        elif ct in (CardType.ITEM, CardType.TOOL, CardType.SUPPORTER, CardType.STADIUM):
            trainer += 1
        elif ct in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            energy += 1
    return float(pokemon), float(trainer), float(energy)


def _energy_on_board(player) -> float:
    total = 0
    active = _active_pokemon(player)
    if active is not None:
        total += len(active.energies or [])
    for pkmn in player.bench or []:
        total += len(pkmn.energies or [])
    return float(total)


def encode_state_from_state(state: State | None, extra_features: list[float] | None = None) -> list[float]:
    """``encode_state()`` の本体。State を直接受け取る版(Observation を作れない呼び出し元向け)。

    ``state`` が None の場合(初回デッキ選択など)はゼロベクトルを返す(壊れない)。

    Args:
        state: エージェント/リプレイ由来の State(``obs.current`` 相当)。
        extra_features: 将来の hidden_information 由来特徴などの差し込み口。渡された場合は
            基本特徴ベクトルの末尾へそのまま連結する(FEATURE_NAMES には含まれない)。

    Returns:
        list[float]: 長さ ``BASE_FEATURE_COUNT`` (+ len(extra_features)) の決定的ベクトル。
    """
    if state is None:
        base = [0.0] * BASE_FEATURE_COUNT
        if extra_features:
            base += [float(x) for x in extra_features]
        return base

    your_index = state.yourIndex
    opp_index = 1 - your_index
    me = state.players[your_index]
    opp = state.players[opp_index]

    my_active = _active_pokemon(me)
    opp_active = _active_pokemon(opp)

    feats: list[float] = []

    # --- ポケモン毎: 自分(相手のバトル場を defender として打点計算) ---
    feats += _side_pokemon_block(me, opp_active, me.handCount, state, your_index)
    # --- ポケモン毎: 相手(自分のバトル場を defender として打点計算) ---
    feats += _side_pokemon_block(opp, my_active, opp.handCount, state, opp_index)

    # --- カウント系: 自分(内訳あり) ---
    hand_pkmn, hand_trainer, hand_energy = _hand_breakdown(me.hand)
    feats += [
        float(me.handCount),
        hand_pkmn,
        hand_trainer,
        hand_energy,
        float(me.deckCount),
        float(len(me.prize or [])),
        float(len(me.discard or [])),
        float(len(me.bench or [])),
    ]
    # --- カウント系: 相手(枚数のみ) ---
    feats += [
        float(opp.handCount),
        float(opp.deckCount),
        float(len(opp.prize or [])),
        float(len(opp.discard or [])),
        float(len(opp.bench or [])),
    ]

    # --- 特殊状態(バトル場) ---
    feats += [
        1.0 if me.poisoned else 0.0,
        1.0 if me.burned else 0.0,
        1.0 if me.asleep else 0.0,
        1.0 if me.paralyzed else 0.0,
        1.0 if me.confused else 0.0,
    ]
    feats += [
        1.0 if opp.poisoned else 0.0,
        1.0 if opp.burned else 0.0,
        1.0 if opp.asleep else 0.0,
        1.0 if opp.paralyzed else 0.0,
        1.0 if opp.confused else 0.0,
    ]

    # --- ゲーム進行 ---
    is_first = 1.0 if (state.firstPlayer == your_index) else 0.0
    feats += [
        float(state.turn),
        is_first,
        1.0 if state.supporterPlayed else 0.0,
        1.0 if state.stadiumPlayed else 0.0,
        1.0 if state.energyAttached else 0.0,
        1.0 if state.retreated else 0.0,
        1.0 if (state.stadium and len(state.stadium) > 0) else 0.0,
    ]

    # --- 集約 ---
    feats += [
        float(board_features.prize_diff(state, your_index)),
        float(len(me.bench or []) - len(opp.bench or [])),
        _energy_on_board(me),
        _energy_on_board(opp),
    ]

    if extra_features:
        feats += [float(x) for x in extra_features]

    return feats


def encode_state(obs: Observation, extra_features: list[float] | None = None) -> list[float]:
    """Observation(現在盤面)を固定長の特徴ベクトルへ変換する。

    ``obs.logs`` / ``obs.select`` は参照しない(current 盤面のみ)。``obs.current`` が None の
    場合(初回デッキ選択など)はゼロベクトルを返す(壊れない)。実体は
    :func:`encode_state_from_state`(``obs.current`` を直接受け取る版)。

    Args:
        obs: エージェント/リプレイ由来の Observation。
        extra_features: 将来の hidden_information 由来特徴などの差し込み口。渡された場合は
            基本特徴ベクトルの末尾へそのまま連結する(FEATURE_NAMES には含まれない)。

    Returns:
        list[float]: 長さ ``BASE_FEATURE_COUNT`` (+ len(extra_features)) の決定的ベクトル。
    """
    return encode_state_from_state(obs.current, extra_features=extra_features)


def encode_obs_dict(obs_dict: dict, extra_features: list[float] | None = None) -> list[float]:
    """学習側ヘルパー: リプレイ/ランタイムの obs_dict を Observation にしてエンコードする。

    dict → Observation の変換は ``cg.api.to_observation_class`` (agent() が使うのと同一経路)を
    再利用する。これにより学習(JSON dict)と実行時(obs_dict)が同一コードパスになる。

    学習データでは ``logs``(と ``select``)が削除されている想定。``to_observation_class`` が
    使う ``to_dataclass`` は必須フィールドの欠損でエラーになるため、欠損キーをここで既定値
    (logs=[]、select=None、current=None)に補ってから変換する。元の dict は変更しない。
    """
    d = dict(obs_dict)
    d.setdefault("logs", [])
    d.setdefault("select", None)
    d.setdefault("current", None)
    obs = to_observation_class(d)
    return encode_state(obs, extra_features=extra_features)


#
# --- 選択肢エンコーダ(Step2: 模倣ポリシー) --------------------------------------------
#
# SelectData.option の各要素を固定長ベクトルへ変換する。状態エンコーダ(上)とは独立に
# 呼べるが、対象ポケモンの特徴には上の _pokemon_features / attack_features /
# energy_requirements をそのまま再利用し、二重実装しない(design.md の踏襲方針)。
#
# 対象カード/ポケモンの解決(area/index -> 実体)は ptcg_ai/rule_based/card_move/common.py
# の resolve_card_id / resolve_pokemon と同じ規則を使うが、learning 側を rule_based に
# 依存させない(design.md の「rule_based は変更しない・依存を増やさない」方針)ため、
# 解決ロジックをここに複製する。

# Option.type の既知値(cg/api.py 記載時点)。CLAUDE.md が警告する通りコンペ期間中に
# Enum へ要素が追加される可能性があるため、未知値は "other" バケットへ落として
# ベクトル長を固定に保つ。
_KNOWN_OPTION_TYPES: list[OptionType] = [
    OptionType.NUMBER,
    OptionType.YES,
    OptionType.NO,
    OptionType.CARD,
    OptionType.TOOL_CARD,
    OptionType.ENERGY_CARD,
    OptionType.ENERGY,
    OptionType.PLAY,
    OptionType.ATTACH,
    OptionType.EVOLVE,
    OptionType.ABILITY,
    OptionType.DISCARD,
    OptionType.RETREAT,
    OptionType.ATTACK,
    OptionType.END,
    OptionType.SKILL,
    OptionType.SPECIAL_CONDITION,
]

# SelectData.type の既知値。同一 select 内の全選択肢に共通する文脈特徴として使う。
_KNOWN_SELECT_TYPES: list[SelectType] = [
    SelectType.MAIN,
    SelectType.CARD,
    SelectType.ATTACHED_CARD,
    SelectType.CARD_OR_ATTACHED_CARD,
    SelectType.ENERGY,
    SelectType.SKILL,
    SelectType.ATTACK,
    SelectType.EVOLVE,
    SelectType.COUNT,
    SelectType.YES_NO,
    SelectType.SPECIAL_CONDITION,
]

_CARD_TYPE_ORDER: list[CardType] = [
    CardType.POKEMON,
    CardType.ITEM,
    CardType.TOOL,
    CardType.SUPPORTER,
    CardType.STADIUM,
    CardType.BASIC_ENERGY,
    CardType.SPECIAL_ENERGY,
]

# area から PlayerState の属性名への対応(rule_based/card_move/common.py の
# _AREA_TO_PLAYER_ZONE と同じ表)。DECK(非公開)/ PRE_EVOLUTION・PLAYER・ENERGY・TOOL
# (親ポケモン側から辿るべき情報)はここでは解決しない。
_AREA_TO_PLAYER_ZONE: dict[AreaType, str] = {
    AreaType.HAND: "hand",
    AreaType.DISCARD: "discard",
    AreaType.PRIZE: "prize",
    AreaType.ACTIVE: "active",
    AreaType.BENCH: "bench",
}


def _build_option_feature_names() -> list[str]:
    names: list[str] = []
    names += [f"opttype_{t.name.lower()}" for t in _KNOWN_OPTION_TYPES]
    names.append("opttype_other")
    names += [f"seltype_{t.name.lower()}" for t in _KNOWN_SELECT_TYPES]
    names.append("seltype_other")
    names += ["is_own", "number_norm", "count_norm", "option_position_norm", "n_options"]
    names.append("has_target_pokemon")
    names += [f"target_pokemon_{feat}" for feat in _POKEMON_FEATURE_NAMES]
    names.append("has_target_card")
    names += [f"target_card_is_{t.name.lower()}" for t in _CARD_TYPE_ORDER]
    names += [
        "target_card_hp_norm",
        "target_card_is_basic",
        "target_card_is_stage1",
        "target_card_is_stage2",
        "target_card_is_ex",
    ]
    names.append("has_attack")
    names += ["attack_damage_norm", "attack_can_ko", "attack_min_shortfall_norm", "attack_has_ready"]
    return names


#: 選択肢1件あたりの特徴ベクトルの各次元の名前。_option_features() と同じ順序。
OPTION_FEATURE_NAMES: list[str] = _build_option_feature_names()

#: 選択肢1件あたりの特徴ベクトルの長さ。
OPTION_FEATURE_COUNT: int = len(OPTION_FEATURE_NAMES)


def _zone_entries_for_option(area: AreaType, player, state: State) -> list | None:
    if area == AreaType.STADIUM:
        return state.stadium
    attr = _AREA_TO_PLAYER_ZONE.get(area)
    if attr is None:
        return None
    return getattr(player, attr, None)


def _resolve_card_id(option: Option, state: State) -> int | None:
    """Option が指すカード/ポケモンの card_id (CardData.id) を特定する。

    OptionType.SKILL 以外は option.cardId が None のことが多いため、area/index
    (PLAY は area 省略・index は hand 内インデックス)と、どうぐ/エネルギーの場合は
    toolIndex/energyIndex を辿って解決する。解決できない場合は None。
    """
    if option.cardId is not None:
        return option.cardId

    area = option.area
    if area is None and option.type == OptionType.PLAY:
        area = AreaType.HAND

    player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
    if area is None or option.index is None or not (0 <= player_index < len(state.players)):
        return None

    zone = _zone_entries_for_option(area, state.players[player_index], state)
    if zone is None or not (0 <= option.index < len(zone)):
        return None
    target = zone[option.index]
    if target is None:
        return None

    if option.toolIndex is not None:
        tools = getattr(target, "tools", None)
        if tools is None or not (0 <= option.toolIndex < len(tools)):
            return None
        return tools[option.toolIndex].id
    if option.energyIndex is not None:
        energy_cards = getattr(target, "energyCards", None)
        if energy_cards is None or not (0 <= option.energyIndex < len(energy_cards)):
            return None
        return energy_cards[option.energyIndex].id

    return target.id


def _resolve_pokemon(option: Option, state: State) -> Pokemon | None:
    """area/index が ACTIVE/BENCH を指す Option を Pokemon に解決する(SWITCH/DAMAGE 等)。"""
    if option.area not in (AreaType.ACTIVE, AreaType.BENCH) or option.index is None:
        return None
    player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
    if not (0 <= player_index < len(state.players)):
        return None
    player = state.players[player_index]
    zone = player.active if option.area == AreaType.ACTIVE else player.bench
    if not (0 <= option.index < len(zone)):
        return None
    return zone[option.index]


def _resolve_in_play_pokemon(option: Option, state: State) -> Pokemon | None:
    """inPlayArea/inPlayIndex が指す場のポケモンを解決する(ATTACH/EVOLVE の対象側)。

    ATTACH/EVOLVE は「area/index = 手札等にあるカード」「inPlayArea/inPlayIndex =
    それを適用する場のポケモン」という2系統のフィールドを持つ(cg/api.py の OptionType
    コメント参照)。_resolve_pokemon は前者(area が直接 ACTIVE/BENCH の場合)しか
    見ないため、こちらは後者専用。
    """
    area = option.inPlayArea
    index = option.inPlayIndex
    if area not in (AreaType.ACTIVE, AreaType.BENCH) or index is None:
        return None
    player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
    if not (0 <= player_index < len(state.players)):
        return None
    player = state.players[player_index]
    zone = player.active if area == AreaType.ACTIVE else player.bench
    if not (0 <= index < len(zone)):
        return None
    return zone[index]


def _option_features(
    option: Option,
    select: SelectData,
    position: int,
    n_options: int,
    state: State,
) -> list[float]:
    """選択肢1件を固定長ベクトル化する。長さは OPTION_FEATURE_COUNT に一致する。"""
    feats: list[float] = []

    # --- Option.type / SelectData.type の one-hot(未知値は other) ---
    for known in _KNOWN_OPTION_TYPES:
        feats.append(1.0 if option.type == known else 0.0)
    feats.append(1.0 if option.type not in _KNOWN_OPTION_TYPES else 0.0)

    for known in _KNOWN_SELECT_TYPES:
        feats.append(1.0 if select.type == known else 0.0)
    feats.append(1.0 if select.type not in _KNOWN_SELECT_TYPES else 0.0)

    # --- 汎用スカラー ---
    your_index = state.yourIndex
    player_index = option.playerIndex if option.playerIndex is not None else your_index
    if not (0 <= player_index < len(state.players)):
        player_index = your_index
    is_own = 1.0 if player_index == your_index else 0.0

    number_norm = (float(option.number) / 10.0) if option.number is not None else 0.0
    count_norm = (float(option.count) / 10.0) if option.count is not None else 0.0
    position_norm = float(position) / float(max(1, n_options - 1))
    feats += [is_own, number_norm, count_norm, position_norm, float(n_options)]

    # --- 対象ポケモン(area/index、または ATTACH/EVOLVE の inPlayArea/inPlayIndex) ---
    # _pokemon_features をそのまま再利用し、盤面評価ロジックを二重実装しない。
    target_pokemon = _resolve_pokemon(option, state)
    if target_pokemon is None:
        target_pokemon = _resolve_in_play_pokemon(option, state)
    if target_pokemon is not None:
        owner_index = player_index
        defender = _active_pokemon(state.players[1 - owner_index])
        hand_size = state.players[owner_index].handCount
        feats.append(1.0)
        feats += _pokemon_features(target_pokemon, defender, hand_size, state, owner_index)
    else:
        feats.append(0.0)
        feats += list(_ZERO_POKEMON)

    # --- 対象カード(手札・トラッシュ・サイド等。場に出ているポケモンではない) ---
    card_id = _resolve_card_id(option, state)
    card = _card_or_none(card_id) if card_id is not None else None
    if card is not None:
        feats.append(1.0)
        for known in _CARD_TYPE_ORDER:
            feats.append(1.0 if card.cardType == known else 0.0)
        if card.cardType == CardType.POKEMON:
            feats += [
                float(card.hp) / 300.0,
                1.0 if card.basic else 0.0,
                1.0 if card.stage1 else 0.0,
                1.0 if card.stage2 else 0.0,
                1.0 if card.ex else 0.0,
            ]
        else:
            feats += [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        feats.append(0.0)
        feats += [0.0] * len(_CARD_TYPE_ORDER)
        feats += [0.0, 0.0, 0.0, 0.0, 0.0]

    # --- 攻撃(ATTACK/SKILL 選択肢): 自分のバトル場ポケモンでこの技を使った場合の打点 ---
    attack = _attack_or_none(option.attackId) if option.attackId is not None else None
    me_active = _active_pokemon(state.players[your_index])
    opp_active = _active_pokemon(state.players[1 - your_index])
    if attack is not None and me_active is not None:
        defender_weakness = defender_resistance = None
        if opp_active is not None:
            opp_card = _card_or_none(opp_active.id)
            if opp_card is not None:
                defender_weakness = opp_card.weakness
                defender_resistance = opp_card.resistance
        hand_size = state.players[your_index].handCount
        shortfall = energy_requirements.energy_shortfall(attack, me_active.energies or [])
        shortfall_sum = min(float(sum(shortfall.values())), _MAX_SHORTFALL)
        has_ready = 1.0 if not shortfall else 0.0
        damage_is_effect = attack_features.damage_is_effect_based(attack)
        damage = attack_features.resolve_damage(
            attack, me_active, defender_weakness, defender_resistance, hand_size,
            defender=opp_active, defender_is_benched=False, damage_is_effect=damage_is_effect,
        )
        can_ko = 0.0
        if opp_active is not None and attack_features.can_ko(
            attack, me_active, opp_active, defender_weakness, defender_resistance, hand_size,
            defender_is_benched=False, damage_is_effect=damage_is_effect,
        ):
            can_ko = 1.0
        feats += [1.0, float(damage) / 200.0, can_ko, shortfall_sum, has_ready]
    else:
        feats += [0.0, 0.0, 0.0, 0.0, 0.0]

    return feats


def encode_options_from_state(state: State | None, select: SelectData | None) -> list[list[float]]:
    """``encode_options()`` の本体。State/SelectData を直接受け取る版(単体テスト向け)。

    ``state`` か ``select`` が None、または選択肢が0件の場合は空リストを返す(壊れない)。

    Returns:
        list[list[float]]: 選択肢と同じ長さのリスト。各要素は長さ OPTION_FEATURE_COUNT の
        決定的ベクトル(``select.option`` と同じ順序)。
    """
    if state is None or select is None or not select.option:
        return []
    n_options = len(select.option)
    return [
        _option_features(option, select, position, n_options, state)
        for position, option in enumerate(select.option)
    ]


def encode_options(obs: Observation) -> list[list[float]]:
    """選択肢(SelectData.option)ごとの特徴量(Step2: 模倣ポリシー)。

    ``obs.current`` / ``obs.select`` が無い場合(初回デッキ選択など)は空リストを返す。
    実体は :func:`encode_options_from_state`。
    """
    return encode_options_from_state(obs.current, obs.select)


def encode_option_card_ids(state: State | None, select: SelectData | None) -> list[int]:
    """選択肢ごとの対象カード/ポケモンの identity(``CardData.cardId``)を返す。

    ``encode_options()`` の連続値ベクトルとは別枠の、埋め込み(embedding)用の生の整数キー列
    (標準化はしない。埋め込みテーブルへの直接のインデックスとして使う想定)。

    背景: ``_option_features()`` の「対象カード」ブロックはカード種別(ポケモン/アイテム/
    どうぐ/…)までしか区別せず、同じ種別内の個別カード(例:「博士の研究」と「ハイパーボール」、
    「基本超エネルギー」と「基本闘エネルギー」)を識別できない。この識別情報の欠如が
    PLAY/ATTACH 選択肢の精度低下の主因と特定された(step2-algorithm-selection.md §7 参照)。
    本関数はカード名をロジックに直書きせず、``card_id`` を特徴として渡すことで学習側が
    データから個別カードの傾向を獲得できるようにする。

    解決順序は ``_option_features()`` と同じ: まず ``_resolve_card_id``(手札/トラッシュ/
    サイド等の対象カード)を試し、解決できなければ ``_resolve_pokemon`` /
    ``_resolve_in_play_pokemon``(場のポケモン)の ``.id`` を使う。どちらも解決できない
    (END/RETREAT の宣言そのもの等)場合は 0(識別なし)。

    デッキ非依存: ``card_id`` はデッキではなくゲーム全体のカードデータ(``all_card_data()``)
    に基づく識別子であり、特定デッキ用のコードではない。埋め込みテーブルは学習時に
    ``all_card_data()`` から動的にサイズを決める想定(将来別デッキで再学習しても本関数・
    ``policy_model.py`` のコードは変更不要。学習データに出現しないカードは埋め込みが
    ただ未学習のままになるだけで、安全側にフォールバックする)。

    Returns:
        list[int]: ``select.option`` と同じ長さ・順序の card_id 列(未解決は 0)。
        ``state``/``select`` が無い、または選択肢が0件の場合は空リスト。
    """
    if state is None or select is None or not select.option:
        return []

    card_ids: list[int] = []
    for option in select.option:
        card_id = _resolve_card_id(option, state)
        if card_id is None:
            pokemon = _resolve_pokemon(option, state)
            if pokemon is None:
                pokemon = _resolve_in_play_pokemon(option, state)
            card_id = pokemon.id if pokemon is not None else None
        card_ids.append(card_id if card_id is not None else 0)
    return card_ids


# ---------------------------------------------------------------------------
# consequence特徴(Tier3 Stage3c、docs/plans/policy-feature-expansion/
# tier3-consequence-features-design-and-implementation-plan.md)。
#
# **本セクションだけ、上記の encode_* 系と契約が異なる。** encode_state/encode_options/
# encode_option_card_ids はいずれも「current盤面のみからの決定的・純粋な読み取り」だが、
# encode_option_consequence_features はゲームエンジンの仮実行(cg.api.search_step、
# ptcg_ai.board_evaluation.consequence 経由)を要求する。そのため hidden_state_factory
# (相手の非公開情報のスタブ。実行時は build_dummy_search_state 等、学習時も同じ経路を
# 使うこと。§2.3 のtrain/runtime parity要件)と deadline(時間予算)を追加引数に取る。
# 例外は投げず、計算できない選択肢は全特徴0で埋める(fail-soft。§6 Stage3c)。
# ---------------------------------------------------------------------------

#: option_consequence の各フィールド名(OptionConsequence.option_index を除く)。
#: この順序で encode_option_consequence_features のベクトルに並ぶ。
CONSEQUENCE_FEATURE_NAMES: list[str] = [
    "opp_hp_loss",
    "self_hp_gain",
    "opp_energy_removed",
    "opp_special_energy_removed",
    "self_energy_added",
    "cards_drawn",
    "pokemon_evolved",
    "stadium_changed",
    "delta_best_effective_attack_damage",
    "delta_can_ko",
    "delta_attack_ready",
    "delta_energy_shortfall",
]

CONSEQUENCE_FEATURE_COUNT: int = len(CONSEQUENCE_FEATURE_NAMES)

_ZERO_CONSEQUENCE = [0.0] * CONSEQUENCE_FEATURE_COUNT

# consequenceを計算する選択肢の型(Tier3方針書 §6 Stage3b: ATTACH/EVOLVE/ITEM系)。
# ATTACKは対象外(attack_planと機能が重複するため。方針書 §1.2/§6 Stage3a)。
# PLAYはITEM/SUPPORTERの発動を含む(改造ハンマー等はPLAYで表現される)。
_CONSEQUENCE_OPTION_TYPES: frozenset = frozenset({
    OptionType.ATTACH, OptionType.EVOLVE, OptionType.PLAY,
})


def _consequence_to_vector(result) -> list[float]:
    return [
        float(result.opp_hp_loss),
        float(result.self_hp_gain),
        1.0 if result.opp_energy_removed else 0.0,
        1.0 if result.opp_special_energy_removed else 0.0,
        1.0 if result.self_energy_added else 0.0,
        float(result.cards_drawn),
        1.0 if result.pokemon_evolved else 0.0,
        1.0 if result.stadium_changed else 0.0,
        float(result.delta_best_effective_attack_damage),
        1.0 if result.delta_can_ko else 0.0,
        1.0 if result.delta_attack_ready else 0.0,
        float(result.delta_energy_shortfall),
    ]


def encode_option_consequence_features(
    obs: Observation,
    hidden_state_factory,
    deadline: float,
) -> list[list[float]]:
    """選択肢ごとの consequence 特徴(Tier3方針書 §3)を返す。

    ``obs.select.option`` のうち ``_CONSEQUENCE_OPTION_TYPES`` に含まれる型だけを
    ``ptcg_ai.board_evaluation.consequence.option_consequence`` で仮実行して評価する
    (ATTACK・その他の型・解決不能・タイムアウトは全特徴0で埋める)。

    Args:
        obs: 現在の Observation(``obs.current``/``obs.select`` が必要)。
        hidden_state_factory: ``consequence.option_consequence`` に渡す0引数callable
            (相手の非公開情報スタブを返す。学習時・実行時で同じ経路を使うこと)。
        deadline: ``time.perf_counter()`` 基準の締め切り(選択肢1件あたりではなく
            呼び出し全体で共有する想定。呼び出し側が予算を管理する)。

    Returns:
        list[list[float]]: ``select.option`` と同じ長さ・順序。各要素は長さ
        ``CONSEQUENCE_FEATURE_COUNT`` のベクトル。``obs.current``/``obs.select`` が
        無い、または選択肢が0件の場合は空リスト。
    """
    from ptcg_ai.board_evaluation import consequence  # 遅延import(循環回避・軽量化)

    state = obs.current
    select = obs.select
    if state is None or select is None or not select.option:
        return []

    vectors: list[list[float]] = []
    for i, option in enumerate(select.option):
        if option.type not in _CONSEQUENCE_OPTION_TYPES:
            vectors.append(list(_ZERO_CONSEQUENCE))
            continue
        try:
            result = consequence.option_consequence(obs, i, hidden_state_factory, deadline)
        except Exception:
            result = None
        vectors.append(_consequence_to_vector(result) if result is not None else list(_ZERO_CONSEQUENCE))
    return vectors
