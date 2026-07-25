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

from cg.api import CardType, Observation, Pokemon, State, to_observation_class

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
            damage = attack_features.resolve_damage(
                attack, pokemon, defender_weakness, defender_resistance, attacker_hand_size
            )
            best_damage = max(best_damage, float(damage))
            if defender is not None and attack_features.can_ko(
                attack, pokemon, defender, defender_weakness, defender_resistance, attacker_hand_size
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


def encode_options(obs: Observation) -> list[list[float]]:
    """選択肢(SelectData.option)ごとの特徴量(Step2 スコープ)。

    ``ml-agent-plan.md`` の Step2(模倣ポリシー: 各選択肢をスコアリングして選ぶ)で実装する。
    本 Step1(バリューネットワーク)では状態特徴のみを扱うため未実装。インターフェースの
    空きだけをここに確保する。
    """
    raise NotImplementedError(
        "encode_options() は Step2(模倣ポリシー)のスコープ。Step1 では状態特徴のみを扱う。"
    )
