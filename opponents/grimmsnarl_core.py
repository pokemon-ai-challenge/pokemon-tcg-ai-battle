"""マリィのオーロンゲex ルールベース対戦相手(sparring partner)の全ロジック。

出典/位置づけ: `sample_submission/docs/plans/opponent-training/` 配下の2文書。
  - grimmsnarl-rule-opponent-plan.md      … 戦略方針 + Phase 0 実測(確定事項)。
  - grimmsnarl-rule-implementation-design.md … 本モジュールの実装契約(§A〜§G)。
python-engineer は上記契約の粒度でそのまま実装したもの。設計判断は行っていない。

``opponents/dragapult_rule_agent.py``(kiyotah 製, ドラパルトex ルールベース)を雛形に、
骨格(スコアリング方式・カードカウント・出力選択ロジック)を流用しつつ、オーロンゲ固有の
新規ロジック(パンクアップのエネルギー配分・アドレナブレインののせ替え・シャドーバレット
一括プラン)を追加した。対戦相手(opponents/)であり、こちらの提出物(sample_submission/)
やproduction の rule_based とは無関係。

``agent(obs, *, deck, profile)`` が本体。5つの薄いラッパ(grimmsnarl_rule_01.py 〜 _05.py)が
各デッキCSVを読み込んで ``build_profile`` した結果をこの関数に渡す。

生int→enum変換(契約 §G-4 の最重要注意)
----------------------------------------
エンジンは ``select.context``/``select.type``/``option.type`` などを生intで返す
(``cg/utils.py`` の ``to_dataclass`` はスカラー値をそのまま代入するだけで、
IntEnumへの変換は行わない)。dragapult は enum 比較のコードだが、IntEnum は int と
値比較できるため実際には動作する。本モジュールは契約の指示どおり、受け取り直後に
``_normalize_select`` で1箇所だけ明示的に正規化してから以降のロジックに渡す
(将来 Enum に要素が追加された場合の安全側フォールバックも兼ねる)。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from cg.api import (
    AreaType,
    CardType,
    EnergyType,
    LogType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    SelectType,
    all_card_data,
    to_observation_class,
)

# ---------------------------------------------------------------------------
# カード定数(契約 §定数テーブル)
# ---------------------------------------------------------------------------
DARK_ENERGY = 7
IMPIDIMP = 646
MORGREM = 647
GRIMMSNARL_EX = 648
MORPEKO = 649
MUNKIDORI = 112
FROSLASS = 104
SNORUNT = 860
BUDEW = 235
DUNSPARCE = 305
DUDUNSPARCE = 66

RARE_CANDY = 1079
UNFAIR_STAMP = 1080
BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
ENERGY_TRANSFER = 1119
POKEGEAR = 1122
TOOL_SCRAPPER = 1137
ENERGY_RECYCLE = 1139
POKE_PAD = 1152
HERO_MANTLE = 1159
HANDY_CIRC = 1161
BALLOON = 1174
BOSS_ORDERS = 1182
XEROSIC = 1197
ROCKET_LAMBDA = 1219
LILLIE_DETERM = 1227
DAWN_HIKARI = 1231
SPIKEMUTH_GYM = 1259

# 攻撃ID(`all_attack()` 実測。契約冒頭の定数テーブル参照)
SHADOW_BULLET = 937
IMPIDIMP_FILCH = 934
IMPIDIMP_PUNCH = 935
MORGREM_PUNCH = 936
MORPEKO_WHEEL = 938
BUDEW_ITCHY = 323

UNNECESSARY = -10000000

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# ATTACH_TO/TO_HAND(min0のとき)等、min件数を超える分をスコア<0で見送れる契約(§F)。
# dragapult は TO_BENCH/SETUP_BENCH_POKEMON のみ許容していたが、本エージェントは
# パンクアップの全取り抑制(no_draw)とスパイクタウンジムの0枚見送りのために追加する。
_SKIPPABLE_CONTEXTS = frozenset(
    {
        SelectContext.TO_BENCH,
        SelectContext.SETUP_BENCH_POKEMON,
        SelectContext.ATTACH_TO,
        SelectContext.TO_HAND,
    }
)


# ---------------------------------------------------------------------------
# 生int → enum 正規化(契約 §G-4-1)
# ---------------------------------------------------------------------------
def _to_enum(cls, value):
    """生int/IntEnumどちらでも安全にenumへ変換する。未知値(将来のenum追加)はintのまま返す。"""
    if value is None:
        return None
    try:
        return cls(int(value))
    except (ValueError, TypeError):
        return value


def _normalize_select(select) -> None:
    """select.context/type と各option.type/area/inPlayArea を生int→IntEnumへ正規化する。

    受け取り直後にここ1箇所だけで変換する(契約 §G-4-1)。dragapult 由来の enum 比較コードを
    そのまま流用するための必須処理(怠ると新規追加した比較ロジックが将来のEnum拡張に対して
    脆くなる)。
    """
    select.context = _to_enum(SelectContext, select.context)
    select.type = _to_enum(SelectType, select.type)
    for o in select.option:
        o.type = _to_enum(OptionType, o.type)
        o.area = _to_enum(AreaType, o.area)
        o.inPlayArea = _to_enum(AreaType, o.inPlayArea)


# ---------------------------------------------------------------------------
# プラン用データクラス(契約 §B-4)
# ---------------------------------------------------------------------------
@dataclass
class AdrenaMove:
    """アドレナブレイン1回分の移動計画(のせ替え元/個数/のせ替え先)。"""

    source_serial: int
    count: int
    target_serial: int


@dataclass
class Plan:
    """`main_option_proc` が MAIN 冒頭で確定し global 保存するアタックプラン(契約 §B-4)。"""

    prizes: int = 0
    score: float = 0.0
    primary: int | None = None  # targets(0=active,1..=bench)のうちシャドーバレット本体が狙う index
    needs_boss: bool = False
    can_main_attack: bool = False
    own_counters: int = 0
    bench30_target_serial: int | None = None
    adrena_move_specs: list = field(default_factory=list)  # [{"target_serial":.., "count":..}]
    adrena_moves: list[AdrenaMove] = field(default_factory=list)


@dataclass
class _FinishResult:
    prize: int = 0
    killed: set = field(default_factory=set)
    bench30: int | None = None
    moves: list = field(default_factory=list)
    tie: int = 0


# ---------------------------------------------------------------------------
# グローバル状態(dragapult 同様。state.turn==0 で全リセット。契約 §G-1)
# ---------------------------------------------------------------------------
can_switch = False
can_attack = False
can_main_attack = False
bench_attacker = False
use_support = 0

pre_turn_log: list = []
current_turn_log: list = []

prize: list[int] = []
card_counts: defaultdict = defaultdict(int)
serial_set: set[int] = set()

plan = Plan()
_current_adrena_source_serial: int | None = None


# ---------------------------------------------------------------------------
# 免疫テーブル(契約 §G-2: 初期は空/最小)
# ---------------------------------------------------------------------------
def no_damage_dex(card_id: int) -> bool:
    """シャドーバレットが効かない相手。v1は空(replay後に追加)。"""
    return False


def no_damage_counter(pokemon: Pokemon) -> bool:
    """ダメカン設置が効かない相手。v1は空(replay後に追加)。"""
    return False


# ---------------------------------------------------------------------------
# 評価関数(dragapult流用/簡略化。契約 §G-2)
# ---------------------------------------------------------------------------
def prize_count(pokemon: Pokemon, is_attack_damage: bool) -> int:
    data = card_table.get(pokemon.id)
    if data is None:
        return 1
    return 3 if data.megaEx else 2 if data.ex else 1


def pokemon_score(pokemon: Pokemon, is_attack_damage: bool) -> int:
    data = card_table.get(pokemon.id)
    score = prize_count(pokemon, is_attack_damage) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data is not None:
        if data.stage2:
            score += 250
        elif data.stage1:
            score += 130
    if pokemon.id == MUNKIDORI and len(pokemon.energies) >= 1:
        score += 300
    score += pokemon.hp
    return score


def weak_mult(pokemon: Pokemon) -> int:
    """相手が悪弱点なら×2(契約: 弱点×2を lethal に必ず入れる)。"""
    data = card_table.get(pokemon.id)
    if data is not None and data.weakness == EnergyType.DARKNESS:
        return 2
    return 1


def heal_value(pokemon: Pokemon, active: bool) -> int:
    """REMOVE_DAMAGE_COUNTER のfallback用、剥がす元としての優先度(契約 §A)。"""
    dmg = pokemon.maxHp - pokemon.hp
    if dmg <= 0:
        return -1
    if pokemon.id == GRIMMSNARL_EX and active:
        return 50000 + dmg
    if pokemon.id == MUNKIDORI and pokemon.hp <= 30:
        return 30000 + dmg
    return dmg * 10


def attach_score(attach_id: int, pokemon: Pokemon, active: bool, is_pankup: bool) -> int:
    """エネルギー/どうぐの付け先スコア(契約 §C)。悪エネ以外の弾は無い。"""
    e = len(pokemon.energies)
    data = card_table.get(attach_id)
    if data is not None and data.cardType == CardType.TOOL:
        score = 60000
        if active:
            score += 1000
        return score

    if not is_pankup:
        # 手貼り(1/ターン): マシマシラ最優先。
        if pokemon.id == MUNKIDORI:
            return 30000 if e == 0 else -1
        if pokemon.id == GRIMMSNARL_EX:
            if active:
                return 22000 if e < 2 else -1
            return 15000 if e < 2 else -1
        if pokemon.id in (IMPIDIMP, MORGREM):
            return -1
        if pokemon.id == MORPEKO:
            return 8000 + e * 500
        return -1

    # パンクアップ(マシラは ATTACH_FROM の options に来ない契約)。
    if pokemon.id == GRIMMSNARL_EX:
        if active:
            return 40000 - e * 1000 if e < 3 else 100
        return 25000 - e * 800 if e < 2 else 100
    if pokemon.id == MORGREM:
        return 12000 - e * 800
    if pokemon.id == MORPEKO:
        return 6000 + e * 300
    if pokemon.id == IMPIDIMP:
        return 3000
    return 500


# ---------------------------------------------------------------------------
# デッキプロファイル(契約 §E)
# ---------------------------------------------------------------------------
def build_profile(deck: list[int]) -> dict:
    counts: defaultdict = defaultdict(int)
    for cid in deck:
        counts[cid] += 1
    return {
        "has_snow": counts[FROSLASS] + counts[SNORUNT] > 0,
        "has_nokocchi": counts[DUDUNSPARCE] + counts[DUNSPARCE] > 0,
        "has_subomi": counts[BUDEW] > 0,
        "has_morpeko": counts[MORPEKO] > 0,
        "rare_candy": counts[RARE_CANDY],
        "boss": counts[BOSS_ORDERS],
        "xerosic": counts[XEROSIC],
        "lambda": counts[ROCKET_LAMBDA],
        "hikari": counts[DAWN_HIKARI],
        "has_balloon": counts[BALLOON] > 0,
        "has_hero_mantle": counts[HERO_MANTLE] > 0,
        "has_handy_circ": counts[HANDY_CIRC] > 0,
        "dark_energy": counts[DARK_ENERGY],
    }


# ---------------------------------------------------------------------------
# カードカウント(dragapult流用。deckを引数化)
# ---------------------------------------------------------------------------
def add_card_count(card, my_index: int) -> None:
    if card is None:
        return
    if isinstance(card, Pokemon) or card.playerIndex == my_index:
        if card.serial not in serial_set:
            card_counts[card.id] -= 1
            serial_set.add(card.serial)
    if isinstance(card, Pokemon):
        for c in card.energyCards:
            add_card_count(c, my_index)
        for c in card.tools:
            add_card_count(c, my_index)
        for c in card.preEvolution:
            add_card_count(c, my_index)


def set_card_counts(obs: Observation, my_index: int, deck: list[int]) -> None:
    card_counts.clear()
    serial_set.clear()
    for cid in deck:
        card_counts[cid] += 1

    state = obs.current
    my_state = state.players[my_index]
    for card in my_state.hand:
        add_card_count(card, my_index)
    for card in my_state.discard:
        add_card_count(card, my_index)
    for card in my_state.bench:
        add_card_count(card, my_index)
    for card in my_state.active:
        add_card_count(card, my_index)
    for card in state.stadium:
        add_card_count(card, my_index)
    if state.looking is not None:
        for card in state.looking:
            add_card_count(card, my_index)
    add_card_count(obs.select.effect, my_index)


def get_card(obs: Observation, area, index: int, player_index: int):
    """指定ゾーンからCard/Pokemonを取り出す(dragapult流用。areaは正規化済み前提)。"""
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


# ---------------------------------------------------------------------------
# セットアップ補助(契約 §A)
# ---------------------------------------------------------------------------
def _setup_active_score(card, profile: dict, going_second: bool, energy_count: int = 0, hp: int = 0) -> int:
    """SETUP_ACTIVE_POKEMON のスコア。この時点の候補は手札由来の ``Card``(Pokemon dataclass
    ではない)のことが多く energies/hp を持たないため、呼び出し側で安全に算出した
    energy_count/hp を渡す(直接 card.energies/card.hp へアクセスしない)。"""
    if card.id == BUDEW:
        base = 120000 if profile.get("has_subomi") and going_second else 500
    elif card.id == IMPIDIMP:
        base = 100000
    elif card.id == SNORUNT:
        base = 30000
    elif card.id == MORPEKO:
        base = 20000
    elif card.id == MUNKIDORI:
        base = -5000
    elif card.id == DUNSPARCE:
        base = 40000
    else:
        base = 0
    return base + energy_count * 1000 + hp


def _setup_bench_score(card: Pokemon, profile: dict) -> int:
    if card.id == IMPIDIMP:
        return 100000
    if card.id == MUNKIDORI:
        return 55000
    if card.id == SNORUNT:
        return 40000 if profile.get("has_snow") else -1
    if card.id == MORPEKO:
        return 10000
    if card.id == BUDEW:
        return -1
    if card.id == DUNSPARCE:
        return 45000
    return 0


# ---------------------------------------------------------------------------
# アタックプラン(契約 §B)
# ---------------------------------------------------------------------------
def _enumerate_finish(
    targets: list[Pokemon],
    rem: list[int],
    killed: set[int],
    bench30_ok: bool,
    adrena_bins: int,
    adrena_total: int,
) -> _FinishResult:
    """軽量ビンパッキング(契約 §B-2)。primary で倒しきれなかった相手のうち、
    ベンチ30とアドレナブレインの配分で追加KOできる部分集合を全列挙して最良を選ぶ。
    n<=6 なので2^6=64通り、ベンチ30の割当も高々7通り(=候補+Noneを掛けても数百通り)で軽量。
    """
    candidates = [i for i in range(len(targets)) if i not in killed and rem[i] > 0]
    n = len(candidates)
    best = _FinishResult()
    for mask in range(1, 1 << n):
        subset = [candidates[j] for j in range(n) if mask & (1 << j)]
        bench30_choices: list[int | None] = [None]
        if bench30_ok:
            bench30_choices += [
                i
                for i in subset
                if i >= 1 and not (card_table.get(targets[i].id) and card_table[targets[i].id].tera)
            ]
        for b30 in bench30_choices:
            needs: dict[int, int] = {}
            for i in subset:
                left = rem[i] - (30 if i == b30 else 0)
                needs[i] = 0 if left <= 0 else -(-left // 10)  # ceil(left/10)
            total_need = sum(needs.values())
            if total_need > adrena_total:
                continue
            # 容量3のbin(=マシマシラ1体)へ降順グリーディで詰める(first-fit-decreasing)。
            sorted_needs = sorted((v for v in needs.values() if v > 0), reverse=True)
            bins: list[int] = []
            feasible = True
            for v in sorted_needs:
                if v > 3:
                    # 1体で3個超必要な場合は複数binを消費する(需要側は移動時に3ずつに分割)。
                    bins.extend([3] * -(-v // 3))
                    continue
                placed = False
                for idx in range(len(bins)):
                    if bins[idx] + v <= 3:
                        bins[idx] += v
                        placed = True
                        break
                if not placed:
                    bins.append(v)
            if len(bins) > adrena_bins:
                feasible = False
            if not feasible:
                continue

            prize = sum(prize_count(targets[i], False) for i in subset)
            tie = sum(pokemon_score(targets[i], False) for i in subset)
            if (prize, tie) > (best.prize, best.tie):
                moves = []
                for i in subset:
                    v = needs[i]
                    while v > 0:
                        take = min(3, v)
                        moves.append({"target_serial": targets[i].serial, "count": take})
                        v -= take
                best = _FinishResult(prize=prize, killed=set(subset), bench30=b30, moves=moves, tie=tie)
    return best


def _tiebreak(total_prize: int, remain_prize: int, needs_boss: bool, pokemon_tie: int) -> float:
    score = float(pokemon_tie)
    if remain_prize > 0 and total_prize >= remain_prize:
        score += 50000
    elif total_prize >= 2:
        score -= 300
    elif total_prize == 1:
        score -= 100
    if needs_boss:
        score -= 500
    return score


def _assign_adrena_sources(moves: list[dict], own_sources: list[list[int]]) -> list[AdrenaMove]:
    """各moveへ実際の「のせ替え元」ポケモンを割り当てる(heal_value優先度順に消費)。

    1回の発動は必ず単一の元ポケモンから最大3個までしか剥がせない(契約 §G-4-3)ため、
    ここでは既に優先度順に並んだ own_sources を先頭から食いつぶす形で割り当てる。
    """
    remaining = [list(x) for x in own_sources]
    assigned: list[AdrenaMove] = []
    for mv in moves:
        for entry in remaining:
            if entry[1] <= 0:
                continue
            take = min(mv["count"], entry[1], 3)
            if take <= 0:
                continue
            entry[1] -= take
            assigned.append(AdrenaMove(source_serial=entry[0], count=take, target_serial=mv["target_serial"]))
            break
    return assigned


def main_option_proc(obs: Observation, profile: dict) -> None:
    """MAIN冒頭で毎回呼ぶ攻撃/アドレナ/ボス統合プラン(契約 §B)。

    can_switch/can_attack/can_main_attack/bench_attacker/plan(global)を更新する。
    dragapult の `main_option_proc` に相当するが、本デッキ固有のシャドーバレット+
    アドレナブレイン+ボスの指令の一括最適化(§B)をここに実装する。
    """
    global can_switch, can_attack, can_main_attack, bench_attacker, plan

    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    can_switch = False
    can_attack = False
    can_main_attack = False
    ability_ready_serials: set[int] = set()
    for o in select.option:
        if o.type == OptionType.RETREAT:
            can_switch = True
        elif o.type == OptionType.ATTACK:
            can_attack = True
            if o.attackId == SHADOW_BULLET:
                can_main_attack = True
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == MUNKIDORI:
                ability_ready_serials.add(card.serial)

    bench_attacker = any(c.id == GRIMMSNARL_EX and len(c.energies) >= 2 for c in my_state.bench)

    own_field = [p for p in my_state.active if p is not None] + list(my_state.bench)

    active_serial = None
    if my_state.active and my_state.active[0] is not None:
        active_serial = my_state.active[0].serial

    # 契約 §B-1: 悪エネが付いているだけでは不十分。そのMAIN決定の select.option に
    # ABILITY(アドレナブレイン)が実際に列挙されているマシマシラのみ「使用可能」と数える。
    # (今ターン既に発動済みの個体はエンジンがABILITY optionを出さないため自然に除外される)
    masila_usable = [
        p
        for p in own_field
        if p.id == MUNKIDORI and len(p.energies) >= 1 and p.serial in ability_ready_serials
    ]
    adrena_bins = len(masila_usable)

    damaged = [p for p in own_field if p.maxHp > p.hp]
    damaged.sort(key=lambda p: -heal_value(p, p.serial == active_serial))
    own_sources = [[p.serial, (p.maxHp - p.hp) // 10] for p in damaged]
    own_counters = sum(c for _, c in own_sources)
    adrena_total = min(adrena_bins * 3, own_counters)

    op_active = op_state.active[0] if op_state.active and op_state.active[0] is not None else None
    targets: list[Pokemon] = ([op_active] if op_active is not None else []) + list(op_state.bench)

    if not targets:
        plan = Plan(can_main_attack=can_main_attack, own_counters=own_counters)
        return

    boss_available = (
        any(c.id == BOSS_ORDERS for c in (my_state.hand or []))
        and not state.supporterPlayed
        and profile.get("boss", 0) > 0
    )

    remain_prize = len(my_state.prize)

    best: Plan | None = None
    primary_candidates: list[int | None] = [None]
    if can_main_attack:
        primary_candidates.append(0)
        if boss_available:
            primary_candidates.extend(range(1, len(targets)))

    for p in primary_candidates:
        needs_boss = p is not None and p != 0
        rem = [t.hp for t in targets]
        killed: set[int] = set()
        prize_taken = 0
        if p is not None and not no_damage_dex(targets[p].id):
            dmg = int(180 * weak_mult(targets[p]))
            rem[p] -= dmg
            if rem[p] <= 0:
                prize_taken += prize_count(targets[p], True)
                killed.add(p)
        bench30_ok = p is not None
        add = _enumerate_finish(targets, rem, killed, bench30_ok, adrena_bins, adrena_total)
        total_prize = prize_taken + add.prize
        tie = _tiebreak(total_prize, remain_prize, needs_boss, add.tie)
        if best is None or (total_prize, tie) > (best.prizes, best.score):
            best = Plan(
                prizes=total_prize,
                score=tie,
                primary=p,
                needs_boss=needs_boss,
                can_main_attack=can_main_attack,
                own_counters=own_counters,
                bench30_target_serial=(targets[add.bench30].serial if add.bench30 is not None else None),
                adrena_move_specs=add.moves,
            )

    assert best is not None
    best.adrena_moves = _assign_adrena_sources(best.adrena_move_specs, own_sources)
    plan = best


# ---------------------------------------------------------------------------
# メインエージェント
# ---------------------------------------------------------------------------
def agent(obs_in: Observation | dict, *, deck: list[int], profile: dict) -> list[int]:
    """マリィのオーロンゲex ルールベースの本体。

    Args:
        obs_in: エージェント関数に渡される Observation(または dict)。
        deck: 60枚デッキ(カードIDリスト)。ラッパ側 module 定数。
        profile: ``build_profile(deck)`` の結果。ラッパ側 module 定数。

    Returns:
        list[int]: 選択肢のインデックスリスト(初回は60枚デッキそのもの)。
    """
    obs = obs_in if isinstance(obs_in, Observation) else to_observation_class(obs_in)
    if obs.select is None:
        return deck

    _normalize_select(obs.select)  # 生int -> enum 正規化(契約 §G-4-1、1箇所)

    global pre_turn_log, current_turn_log
    global can_switch, can_attack, can_main_attack, bench_attacker
    global use_support, plan, _current_adrena_source_serial

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    if state.turn == 0:
        prize.clear()
        pre_turn_log.clear()
        current_turn_log.clear()
        card_counts.clear()
        serial_set.clear()
        plan = Plan()
        use_support = 0
        can_switch = can_attack = can_main_attack = bench_attacker = False
        _current_adrena_source_serial = None
    else:
        for log in obs.logs:
            current_turn_log.append(log)
            if log.type == LogType.TURN_END:
                pre_turn_log = current_turn_log
                current_turn_log = []

    pre_ko = False
    no_item = False
    for log in pre_turn_log:
        if log.type == LogType.ATTACK:
            if log.attackId == BUDEW_ITCHY:
                no_item = True
        elif log.type == LogType.MOVE_CARD:
            if (
                log.playerIndex == my_index
                and (log.fromArea == AreaType.BENCH or log.fromArea == AreaType.ACTIVE)
                and log.toArea == AreaType.DISCARD
            ):
                pre_ko = True

    if select.deck is not None:
        set_card_counts(obs, my_index, deck)
        for card in select.deck:
            card_counts[card.id] -= 1
        prize.clear()
        for cid in card_counts:
            for _ in range(card_counts[cid]):
                prize.append(cid)

    set_card_counts(obs, my_index, deck)
    for cid in prize:
        card_counts[cid] -= 1
    deck_counts = card_counts

    # --- 場の集計 ---
    field_counts: defaultdict = defaultdict(int)
    hand_counts: defaultdict = defaultdict(int)
    discard_counts: defaultdict = defaultdict(int)

    active_id = 0
    active_pokemon = None
    for card in my_state.active:
        if card is None:
            continue
        active_id = card.id
        active_pokemon = card
        field_counts[card.id] += 1
    for card in my_state.bench:
        field_counts[card.id] += 1

    main_line_count = field_counts[IMPIDIMP] + field_counts[MORGREM] + field_counts[GRIMMSNARL_EX]
    # 攻撃成熟ゲート(契約 §F): 展開途中の即攻撃を防ぐ。
    mature = main_line_count >= 2

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    for card in my_state.discard:
        discard_counts[card.id] += 1

    no_draw = my_state.deckCount <= 8  # 契約 §D-3
    op_has_tool = any(len(p.tools) > 0 for p in ([op_state.active[0]] if op_state.active and op_state.active[0] else []) + list(op_state.bench))
    has_tool_active = any(len(p.tools) > 0 for p in my_state.active if p is not None)

    going_second = state.firstPlayer not in (-1, my_index)

    support_count = 0

    def hand_score(id_: int, ignore_count: bool) -> int:
        score = 0
        if id_ == IMPIDIMP:
            score = 1000 if main_line_count >= 3 else 18000
        elif id_ == MORGREM:
            score = 20000 if field_counts[IMPIDIMP] >= 1 else 3000
        elif id_ == GRIMMSNARL_EX:
            existing = field_counts[GRIMMSNARL_EX]
            if existing >= 2 or (existing >= 1 and len(op_state.prize) <= 2):
                score = UNNECESSARY
            elif field_counts[IMPIDIMP] >= 1 and hand_counts[RARE_CANDY] >= 1 and not no_item:
                score = 45000
            elif field_counts[MORGREM] >= 1:
                score = 40000
            else:
                score = 2000
        elif id_ == MORPEKO:
            score = 12000 if profile.get("has_morpeko") else 100
        elif id_ == FROSLASS:
            score = 45000 if profile.get("has_snow") else UNNECESSARY
        elif id_ == SNORUNT:
            score = 40000 if profile.get("has_snow") and field_counts[FROSLASS] == 0 else 100
        elif id_ == MUNKIDORI:
            score = 50000 if field_counts[MUNKIDORI] < 3 else 500
        elif id_ == BUDEW:
            score = 30000 if profile.get("has_subomi") and field_counts[BUDEW] == 0 else UNNECESSARY
        elif id_ == DUNSPARCE:
            score = 35000 if profile.get("has_nokocchi") else UNNECESSARY
        elif id_ == DUDUNSPARCE:
            score = 15000 if profile.get("has_nokocchi") and field_counts[DUNSPARCE] >= 1 else UNNECESSARY
        elif id_ == RARE_CANDY:
            if field_counts[GRIMMSNARL_EX] >= 2:
                score = UNNECESSARY
            elif field_counts[IMPIDIMP] >= 1 and hand_counts[GRIMMSNARL_EX] >= 1 and not no_item:
                score = 40000
            else:
                score = 50
        elif id_ == UNFAIR_STAMP:
            if pre_ko:
                score = 80000
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
            else:
                score = 15000
        elif id_ == BUDDY_POFFIN:
            count = deck_counts[IMPIDIMP] + deck_counts[SNORUNT] + deck_counts[MORPEKO] + deck_counts[BUDEW]
            if no_draw or count == 0:
                score = UNNECESSARY
            else:
                score = 35000 if count >= 2 else 100
        elif id_ == POKE_PAD:
            need = (deck_counts[MUNKIDORI] > 0 and field_counts[MUNKIDORI] < 3) or (
                deck_counts[SNORUNT] + deck_counts[FROSLASS] > 0
            )
            score = UNNECESSARY if no_draw or not need else 45000
        elif id_ == NIGHT_STRETCHER:
            for cid in discard_counts:
                if discard_counts[cid] >= 1:
                    ct = card_table.get(cid)
                    if ct is not None and ct.cardType in (CardType.POKEMON, CardType.BASIC_ENERGY):
                        score = max(score, hand_score(cid, True) // 2)
            if no_draw:
                score = min(score, 5000)
        elif id_ == ENERGY_TRANSFER:
            score = UNNECESSARY if no_draw else 8000
        elif id_ == ENERGY_RECYCLE:
            score = UNNECESSARY if no_draw else (12000 if discard_counts[DARK_ENERGY] >= 3 else 100)
        elif id_ == POKEGEAR:
            score = UNNECESSARY if no_draw else 20000
        elif id_ == TOOL_SCRAPPER:
            score = 4000 if op_has_tool else 5
        elif id_ == HERO_MANTLE:
            score = 30000 if active_id == GRIMMSNARL_EX and not has_tool_active else 100
        elif id_ == HANDY_CIRC:
            score = 28000 if active_id == GRIMMSNARL_EX and not has_tool_active else 100
        elif id_ == BALLOON:
            score = 20000 if active_id == GRIMMSNARL_EX and not has_tool_active else 100
        elif id_ == BOSS_ORDERS:
            score = 60000 if plan.prizes >= 1 else 0
        elif id_ == LILLIE_DETERM:
            if no_draw:
                score = UNNECESSARY
            elif len(my_state.hand) <= 4 or len(my_state.prize) <= 2:
                score = 45000
        elif id_ == DAWN_HIKARI:
            score = UNNECESSARY if no_draw else (50000 if main_line_count < 3 else 20000)
        elif id_ == ROCKET_LAMBDA:
            score = UNNECESSARY if no_draw else 40000
        elif id_ == XEROSIC:
            score = 25000 if op_state.handCount >= 4 else 5000
        elif id_ == SPIKEMUTH_GYM:
            score = 3000 if stadium_id != SPIKEMUTH_GYM else 10
        elif id_ == DARK_ENERGY:
            if can_main_attack and (plan.prizes >= 1 or bench_attacker):
                score = UNNECESSARY
            else:
                max_score = -10000
                for pokemon in my_state.active:
                    if pokemon is None:
                        continue
                    max_score = max(max_score, attach_score(DARK_ENERGY, pokemon, True, False))
                for pokemon in my_state.bench:
                    max_score = max(max_score, attach_score(DARK_ENERGY, pokemon, False, False))
                score = max_score - 5000

        if not ignore_count and hand_counts[id_] > 0:
            if id_ == IMPIDIMP:
                score -= 100
            else:
                score -= 100000
        return score

    if context == SelectContext.MAIN:
        main_option_proc(obs, profile)

        use_support = 0
        if not state.supporterPlayed:
            support_score = 0
            for o in select.option:
                if o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, o.index, my_index)
                    if card_table.get(card.id) and card_table[card.id].cardType == CardType.SUPPORTER:
                        s = hand_score(card.id, True)
                        if support_score < s:
                            support_score = s
                            use_support = card.id

    hand_scores = []
    for card in my_state.hand:
        s = hand_score(card.id, False)
        hand_scores.append(s)
        hand_counts[card.id] += 1
        if card_table.get(card.id) and card_table[card.id].cardType == CardType.SUPPORTER and card.id != BOSS_ORDERS:
            support_count += 1

    do_switch = (not can_main_attack) and (
        bench_attacker or (active_id in (BUDEW, IMPIDIMP) and state.turn >= 2)
    )
    effect_card_id = 0 if select.effect is None else select.effect.id
    context_card_id = 0 if select.contextCard is None else select.contextCard.id

    take_count = 0
    if context == SelectContext.ATTACH_TO:
        if no_draw:
            need = 0
            if active_pokemon is not None and active_pokemon.id == GRIMMSNARL_EX:
                need += max(0, 2 - len(active_pokemon.energies))
            for c in my_state.bench:
                if c.id in (GRIMMSNARL_EX, MORGREM):
                    need += max(0, 2 - len(c.energies))
            take_count = min(select.maxCount, need)
        else:
            take_count = select.maxCount

    scores = []
    for pos, o in enumerate(select.option):
        score = 0
        if o.type == OptionType.NUMBER:
            if context == SelectContext.REMOVE_DAMAGE_COUNTER_COUNT:
                source_card = select.contextCard
                _current_adrena_source_serial = source_card.serial if source_card is not None else None
                matched = next(
                    (m for m in plan.adrena_moves if m.source_serial == _current_adrena_source_serial), None
                )
                score = 100000 if (matched is not None and o.number == matched.count) else o.number
            else:
                score = o.number
        elif o.type == OptionType.YES:
            score = -1 if context == SelectContext.IS_FIRST else 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = 0
                hp = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                    hp = card.hp
                if context == SelectContext.SETUP_ACTIVE_POKEMON:
                    score = _setup_active_score(card, profile, going_second, energy_count, hp)
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = _setup_bench_score(card, profile)
                elif context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                    if o.playerIndex == my_index:
                        if card.id == GRIMMSNARL_EX:
                            score = 50000 if energy_count >= 2 else 15000
                        elif card.id == MORGREM:
                            score = 20000 if energy_count >= 2 else 5000
                        elif card.id == IMPIDIMP:
                            score = 10000
                        elif card.id == BUDEW:
                            score = -8000
                        elif card.id == MUNKIDORI:
                            score = -10000
                        else:
                            score = 0
                        score += energy_count * 1000 + hp
                    else:
                        if plan.needs_boss and plan.primary == o.index + 1:
                            score = 100000
                elif context == SelectContext.TO_BENCH:
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                elif context == SelectContext.TO_HAND:
                    if effect_card_id == SPIKEMUTH_GYM:
                        score = -1 if no_draw else hand_score(card.id, False)
                    elif effect_card_id == POKE_PAD:
                        if card.id == MUNKIDORI:
                            score = 60000
                        elif card.id in (SNORUNT, MORGREM, IMPIDIMP):
                            score = 40000
                        else:
                            score = 100
                    elif effect_card_id == DAWN_HIKARI:
                        score = hand_score(card.id, False)
                    elif effect_card_id == ROCKET_LAMBDA:
                        if plan.prizes >= 1 and hand_counts[BOSS_ORDERS] == 0 and profile.get("boss", 0) > 0:
                            score = 60000 if card.id == BOSS_ORDERS else -1
                        elif hand_counts[RARE_CANDY] == 0 and field_counts[IMPIDIMP] >= 1:
                            score = 50000 if card.id == RARE_CANDY else -1
                        elif stadium_id != SPIKEMUTH_GYM:
                            score = 40000 if card.id == SPIKEMUTH_GYM else -1
                        else:
                            score = 30000 if card.id == LILLIE_DETERM else -1
                    elif effect_card_id == NIGHT_STRETCHER:
                        ct = card_table.get(card.id)
                        if ct is not None and ct.cardType in (CardType.POKEMON, CardType.BASIC_ENERGY):
                            score = hand_score(card.id, False)
                        else:
                            score = -1
                    else:
                        score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                elif context == SelectContext.DISCARD:
                    hand_counts[card.id] -= 1
                    ct = card_table.get(card.id)
                    if ct is not None and ct.cardType == CardType.SUPPORTER and card.id != BOSS_ORDERS:
                        support_count -= 1
                    score = -hand_score(card.id, False)
                elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    if hp <= 0:
                        score = -1
                    else:
                        matched_target = None
                        if context == SelectContext.DAMAGE_COUNTER:
                            matched_target = next(
                                (m for m in plan.adrena_moves if m.source_serial == _current_adrena_source_serial),
                                None,
                            )
                        if matched_target is not None and card.serial == matched_target.target_serial:
                            score = 100000
                            plan.adrena_moves.remove(matched_target)
                        else:
                            score = 50000 - 10 * hp + pokemon_score(card, False)
                            if hp <= 30:
                                score += 20000
                            elif hp <= 90:
                                score += 8000
                            if no_damage_counter(card):
                                score = -1
                elif context == SelectContext.DAMAGE:
                    if plan.bench30_target_serial is not None and card.serial == plan.bench30_target_serial:
                        score = 100000
                    else:
                        ct = card_table.get(card.id)
                        score = pokemon_score(card, True)
                        if hp <= 30:
                            score += 50000
                        if ct is not None and ct.tera:
                            score = -1
                elif context == SelectContext.REMOVE_DAMAGE_COUNTER:
                    is_plan_source = any(m.source_serial == card.serial for m in plan.adrena_moves)
                    base = heal_value(card, o.area == AreaType.ACTIVE)
                    score = (100000 + base) if is_plan_source else base
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE, True)
                elif context == SelectContext.ATTACH_TO:
                    score = 1 if pos < take_count else -1
                elif context == SelectContext.EVOLVES_FROM:
                    # ふしぎなアメの飛び進化(たね側選択)。実測未確認のため保守的な推測実装
                    # (Rare Candyの実際のselectシーケンスがEVOLVES_FROM/TOを使わない場合でも、
                    # ここは呼ばれず、下の未知context用score=0フォールバックが安全に効く)。
                    existing = field_counts[GRIMMSNARL_EX]
                    if existing >= 2 or (existing >= 1 and len(op_state.prize) <= 2):
                        score = -1
                    elif card.id == IMPIDIMP:
                        score = 75000 + energy_count
                    else:
                        score = 1000 + energy_count
                elif context == SelectContext.EVOLVES_TO:
                    score = 75000 if card.id == GRIMMSNARL_EX else 1000
        elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
            if o.playerIndex != my_index:
                score = 20 if o.area == AreaType.BENCH else 10
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            card_score = hand_scores[o.index]
            cid = card.id
            if cid == IMPIDIMP:
                score = 51000
            elif cid == BUDEW:
                score = 52000 if card_score > 0 else -1
            elif cid == MUNKIDORI:
                score = 53000 if field_counts[MUNKIDORI] < 3 else -1
            elif cid == SNORUNT:
                score = 51000 if profile.get("has_snow") and field_counts[FROSLASS] == 0 else -1
            elif cid == MORPEKO:
                score = 51000 if profile.get("has_morpeko") else -1
            elif cid == RARE_CANDY:
                score = 75000 if card_score > 0 else -1
            elif cid == UNFAIR_STAMP:
                score = 15000 if card_score > 0 else -1
            elif cid == NIGHT_STRETCHER:
                score = 42000 if card_score >= 9000 else -1
            elif cid == BOSS_ORDERS:
                score = 35000 if cid == use_support else -1
            elif cid == LILLIE_DETERM:
                score = 14000 if cid == use_support else -1
            elif cid == XEROSIC:
                score = 33000 if cid == use_support else -1
            elif cid == SPIKEMUTH_GYM:
                score = 20000 if stadium_id != cid else -1
            elif cid in (HERO_MANTLE, HANDY_CIRC, BALLOON):
                score = 30000 if card_score > 0 else -1
            elif cid == TOOL_SCRAPPER:
                score = 20000 if card_score > 0 else -1
            elif no_draw:
                score = -1
            elif cid == BUDDY_POFFIN:
                score = 46000 if card_score > 0 else -1
            elif cid == POKE_PAD:
                score = 45000 if card_score > 0 else -1
            elif cid == DAWN_HIKARI:
                score = 44000 if use_support == cid else -1
            elif cid == ROCKET_LAMBDA:
                score = 43000 if use_support == cid else -1
            elif cid == POKEGEAR:
                score = 22000 if card_score > 0 else -1
            elif cid in (ENERGY_TRANSFER, ENERGY_RECYCLE):
                score = 21000 if card_score > 0 else -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE, False)
        elif o.type == OptionType.EVOLVE:
            pre_evo = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pre_evo.energies)
            if pre_evo.id == IMPIDIMP:
                evo_card = get_card(obs, o.area, o.index, my_index)
                if evo_card is not None and evo_card.id == GRIMMSNARL_EX:
                    existing = field_counts[GRIMMSNARL_EX]
                    if existing >= 2 or (existing >= 1 and len(op_state.prize) <= 2):
                        score = -1
                    else:
                        score += 75000
                else:
                    score += 30000
                    if hand_counts[GRIMMSNARL_EX] >= 1 or hand_counts[RARE_CANDY] >= 1:
                        score += 10000
            elif pre_evo.id == MORGREM:
                existing = field_counts[GRIMMSNARL_EX]
                if existing >= 2 or (existing >= 1 and len(op_state.prize) <= 2):
                    score = -1
                else:
                    score += 70000
            elif pre_evo.id == SNORUNT:
                score += 60000 if profile.get("has_snow") else -1
            else:
                score += 1000
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == MUNKIDORI:
                if plan.adrena_moves:
                    score = 95000
                elif plan.own_counters > 0:
                    score = 40000
                else:
                    score = -1
            else:
                score = 1
        elif o.type == OptionType.RETREAT:
            score = 10000 if do_switch else -1
        elif o.type == OptionType.ATTACK:
            if o.attackId == SHADOW_BULLET:
                if plan.prizes >= 1:
                    score = 90000
                else:
                    score = 1 if mature else -1
            elif o.attackId == IMPIDIMP_FILCH:
                score = 90000 if plan.prizes >= 1 else 3
            elif o.attackId in (IMPIDIMP_PUNCH, MORGREM_PUNCH, MORPEKO_WHEEL):
                score = 90000 if plan.prizes >= 1 else 2
            else:
                score = 90000 if plan.prizes >= 1 else 1
        elif o.type == OptionType.SKILL:
            score = -pos

        scores.append(score)

    output = []
    if len(scores) >= 1:
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            if (
                sorted_scores[i][1] >= 0
                or select.minCount > i
                or context not in _SKIPPABLE_CONTEXTS
            ):
                output.append(sorted_scores[i][0])

    return output
