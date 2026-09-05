"""ルールベース ドラパルトex 対戦相手エージェント(**ダメカン配置プランナ版**・研究用)。

``opponents/dragapult_rule_agent.py`` (kiyotah 版) からの派生。**差分は
``SelectContext.DAMAGE_COUNTER_ANY`` の選択をプランナ
(``ptcg_ai/search/damage_counter_plan.py``)に委譲する点だけ**で、それ以外のスコアリング・
プラン探索・グローバル状態の扱いは kiyotah 版のまま。対照群を凍結したまま実験群を作るため、
import ではなくファイルのコピーとして分離してある(派生元は1バイトも変更しない)。

なぜ委譲するのか: kiyotah 版のダメカン配置は「今すぐKOできる部分集合の列挙 + HP帯の固定
ボーナス」で、複数ターン計画・サイド算術・ボス連動が無い。また本番の探索パイプラインは
``SelectType.MAIN`` かつ ``maxCount == 1`` にしか適用されないため、1個ずつ降ってくる
ダメカン配置(sub-select)には探索が構造的に届かない。明示的なプランナしか手段が無い。

プランナが None を返す/例外を投げる/選んだ対象が選択肢に無い場合は、**kiyotah 版の
元ロジックにそのままフォールバック**する(安全側)。``PLANNER_ENABLED`` を False にすれば
挙動は派生元と完全に一致する(ablation 用)。

出典(派生元): Kaggle ノートブック "A Sample Rule-Based Agent Dragapult ex Deck"
(著者: kiyotah, https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-dragapult-ex-deck)。
対戦相手(sparring partner)として league から使うために取り込んだもので、こちらの提出
(sample_submission/)とは無関係。

原典(Kaggle 提出用 main.py)からの差分は、上記のプランナ委譲に加えて、対戦基盤に載せるための
以下2点:

1. デッキの読み込み元。原典は cwd の ``deck.csv`` を読むが、league は cwd を
   ``sample_submission/`` にそろえるため、代わりにこのファイルと同じディレクトリの
   ``dragapult_ex_deck.csv`` を読む(league では ``--deck-b`` に同じ CSV を渡すこと。
   set_card_counts が my_deck の全構成に依存するため、実際に対戦で使うデッキと
   一致していなければならない)。

2. ``agent()`` の入力。Kaggle 基盤は dict を渡すが、``league/run_match.py`` は
   すでに ``Observation`` に変換したものを渡す。両契約で動くよう、先頭で
   Observation ならそのまま使い、dict なら変換する分岐だけ足した。

上記以外(スコアリング・プラン探索・グローバル状態の扱い)は原典のまま。グローバル状態は
``state.turn == 0`` で毎ゲーム冒頭にリセットされるため、league がワーカープロセスで
このエージェントを複数試合に使い回しても、ゲーム間で状態は持ち越さない。
"""
import os
import sys
from collections import defaultdict

from cg.api import AreaType, CardType, Log, LogType, Observation, SelectContext, OptionType, Card, Pokemon, State, all_card_data, to_observation_class

# --- ここから派生版の追加 (ダメカン配置プランナ) --------------------------------
# ptcg_ai は sample_submission/ 配下にある。cg が import できている時点で通っている
# はずだが、単体でこのモジュールを import されても動くようにパスを補う。
_OPPONENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_OPPONENTS_DIR)
for _candidate in (os.path.join(_REPO_ROOT, "sample_submission"), _REPO_ROOT):
    if _candidate not in sys.path:
        sys.path.append(_candidate)

try:
    from ptcg_ai.search import damage_counter_plan
except Exception:  # noqa: BLE001 - プランナが読めない環境では派生元と同じ挙動に落とす
    damage_counter_plan = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ablation 用スイッチ。False にすると派生元(kiyotah 版)と完全に同じ挙動になる。
PLANNER_ENABLED = _env_flag("DRAGAPULT_PLANNER_ENABLED", True) and damage_counter_plan is not None
# 先読みする攻撃ターン数(現在のターンを1と数える)。
PLANNER_HORIZON = _env_int("DRAGAPULT_PLANNER_HORIZON", 2)
# --- 追加ここまで --------------------------------------------------------------

"""
Dragapult ex Deck
Advanced Level
This deck focuses on setting up multiple knockouts to take at least three Prize cards in a single turn with its Phantom Dive attack.
"""

# Load the deck sitting next to this module (league keeps cwd at sample_submission/,
# so a cwd-relative "deck.csv" would resolve to the wrong deck).
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dragapult_ex_deck.csv")
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId:c for c in all_card}

# Decklist
Dreepy = 119  # ×4
Drakloak = 120  # ×4
Dragapult_ex = 121  # ×3
Fezandipiti_ex = 140  # ×1
Latias_ex = 184  # ×1
Budew = 235  # ×2
Meowth_ex = 1071  # ×1
Rare_Candy = 1079  # ×2
Unfair_Stamp = 1080  # ×1
Buddy_Buddy_Poffin = 1086  # ×4
Night_Stretcher = 1097  # ×2
Crushing_Hammer = 1120  # ×4
Ultra_Ball = 1121  # ×4
Poke_Pad = 1152  # x3
Lucky_Helmet = 1156  # ×1
Boss_Orders = 1182  # ×3
Crispin = 1198  # ×4
Brock_Scouting = 1210  # ×2
Lillie_Determination = 1227  # ×4
Team_Rocket_Watchtower = 1256  # ×2
Basic_Fire_Energy = 2  # ×4
Basic_Psychic_Energy = 5  # ×4

UNNECESSARY = -10000000

class AttackPlan:
    attack: int = 0
    counter: list[int] = []

can_switch = False
can_attack = False
can_main_attack = False
can_energy_attach = False
use_support = 0  # The Supporter card planned for use.
bench_attacker = False  # Whether there is a Benched Pokémon that is ready to attack
pre_turn_log: list[Log] = []
current_turn_log: list[Log] = []

prize: list[int] = []
card_counts: defaultdict[int, int] = defaultdict(int)
serial_set: set[int] = set()
plan_a = AttackPlan()
plan_b = AttackPlan()


def no_damage_dex(id: int) -> bool:
    """Checks if the defending Pokémon possesses innate immunities preventing Dragapult ex from hitting it."""
    # Drednaw, Milotic ex, Sylveon, Crustle
    return id == 158 or id == 207 or id == 330 or id == 345


def no_damage_counter(pokemon: Pokemon) -> bool:
    """Checks if a target prevents placement of Phantom Dive's 6 bench damage counters (via abilities/Energy)."""
    # Poltchageist, Empoleon ex, Skeledirge, Milotic ex, Misty's Magikarp, Antique Cover Fossil
    if pokemon.id == 28 or pokemon.id == 199 or pokemon.id == 203 or pokemon.id == 207 or pokemon.id == 362 or pokemon.id == 1136:
        return True
    for card in pokemon.energyCards:
        # Mist Energy, Rock Fighting Energy
        if card.id == 11 or card.id == 20:
            return True
    return False


def prize_count(pokemon: Pokemon, is_attack_damage: bool) -> int:
    """Calculates how many Prize cards a Pokémon yields upon being Knocked Out, factoring in modifiers."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    if is_attack_damage:
        for card in pokemon.energyCards:
            if card.id == 12:  # Legacy Energy
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and "Lillie" in data.name:  # Lillie’s Pearl
                count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon, is_attack_damage: bool) -> int:
    """Heuristically evaluates the tactical worth of targeting a specific Pokémon on the opponent's field."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon, is_attack_damage) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    id = pokemon.id
    # Noctowl, Fan Rotom, Archaludon ex, Meowth ex
    if id == 173 or id == 174 or id == 190 or id == 1071:
        score -= 200
    if id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


def add_card_count(card: Card | Pokemon | None, my_index: int):
    if card == None:
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

def set_card_counts(obs: Observation, my_index: int):
    card_counts.clear()
    serial_set.clear()
    for id in my_deck:
        card_counts[id] += 1

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
    if state.looking != None:
        for card in state.looking:
            add_card_count(card, my_index)
    add_card_count(obs.select.effect, my_index)


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
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

def main_option_proc(obs: Observation, damage: int):
    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global can_switch
    global can_attack
    global can_main_attack
    global can_energy_attach

    can_switch = False
    can_attack = False
    can_main_attack = False
    can_energy_attach = False
    for o in select.option:
        if o.type == OptionType.RETREAT:
            can_switch = True
        elif o.type == OptionType.ATTACK:
            can_attack = True
            if o.attackId == 154:  # Phantom Dive
                can_main_attack = True

    plan_a.attack = -1
    plan_b.attack = -1
    if not can_main_attack and not (bench_attacker and can_switch):
        return

    cards = [op_state.active[0]]
    for pokemon in op_state.bench:
        cards.append(pokemon)
    counter_indices = []
    ci = []
    ci.append(0)
    remain_damage = 60
    while ci:
        index = ci[-1]
        hp = cards[index].hp
        if remain_damage >= hp:
            counter_indices.append(ci.copy())
            if index < len(cards) - 1:
                remain_damage -= hp
                ci.append(index + 1)
                continue
        if index == len(cards) - 1:
            ci.pop()
            if ci:
                remain_damage += cards[ci[-1]].hp
        if ci:
            ci[-1] += 1
    counter_indices.append([])

    remain_prize = len(my_state.prize)
    plan_score = 0
    for i, pokemon in enumerate(cards):
        base_prize_count = 0
        base_score = pokemon_score(pokemon, True)
        active_damage = 0 if no_damage_dex(pokemon.id) else damage
        if pokemon.hp <= active_damage:
            base_prize_count += prize_count(pokemon, True)
        else:
            base_score *= active_damage / pokemon.hp
        ci = []
        max_score = base_score
        if remain_prize <= base_prize_count:
            max_score = 50000
        else:
            for indices in counter_indices:
                if i in indices:
                    continue
                prize = base_prize_count
                score = base_score
                for index in indices:
                    prize += prize_count(cards[index], False)
                    score += pokemon_score(cards[index], False)
                if remain_prize <= prize:
                    score = 50000
                else:
                    if prize >= 2:
                        if remain_prize <= 4:
                            score -= 1200
                    elif prize == 1:
                        score -= 300
                    else:
                        score += 1200
                if max_score < score:
                    max_score = score
                    ci = indices
        if plan_score < max_score:
            plan_score = max_score
            plan_a.attack = i
            plan_a.counter = ci
        if i == 0:
            plan_b.attack = plan_a.attack
            plan_b.counter = plan_a.counter

def _validate_output(output, select) -> bool:
    """返り値contract(範囲内・個数 min..max・重複なし)を満たすか。

    プランナ由来の手はここを通らないと採用しない。1つでも破ると Kaggle 基盤では
    その時点で反則負けになるため、フォールバックできる形で必ず検査する。
    """
    if not isinstance(output, list):
        return False
    if len(output) < select.minCount or len(output) > select.maxCount:
        return False
    if len(set(output)) != len(output):
        return False
    count = len(select.option)
    return all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < count for i in output)


def _planner_damage_counter_choice(obs: Observation) -> list[int] | None:
    """ファントムダイブのダメカン1個を、ターン跨ぎのDPプランナに決めさせる。

    盤面から純粋なデータ(``TargetState``)を作り、``choose_next_counter`` に「次の1個を
    どこに置くか」だけを聞く。計画はキャッシュせず毎回作り直す(相手の割り込み・エンジンの
    処理順に依存しないため)。

    プランナが答えられない条件(対象なし・自分側に置く効果・複数個まとめ選択など)では
    None を返し、呼び出し側は kiyotah 版の元ロジックにフォールバックする。

    Returns:
        list[int] | None: 採用する選択肢インデックス1個、または None(フォールバック)。
    """
    if not PLANNER_ENABLED or damage_counter_plan is None:
        return None
    select = obs.select
    state = obs.current
    if select is None or state is None:
        return None
    if select.context != SelectContext.DAMAGE_COUNTER_ANY:
        return None
    # ダメカンは1個ずつ降ってくる前提。まとめ選択が来たらプランナの守備範囲外。
    if select.minCount != 1 or select.maxCount != 1:
        return None
    remain = select.remainDamageCounter
    if remain is None or remain < 1:
        return None

    my_index = state.yourIndex
    op_index = 1 - my_index
    my_state = state.players[my_index]
    op_state = state.players[op_index]

    # 選択肢 -> 相手ポケモンの serial。相手以外/ポケモン以外が混ざる効果は対象外。
    option_index_by_serial: dict[int, int] = {}
    for i, o in enumerate(select.option):
        if o.type != OptionType.CARD or o.playerIndex != op_index:
            return None
        card = get_card(obs, o.area, o.index, o.playerIndex)
        if not isinstance(card, Pokemon):
            return None
        if card.hp <= 0:
            continue
        if card.serial not in option_index_by_serial:
            option_index_by_serial[card.serial] = i
    if not option_index_by_serial:
        return None

    board: list[tuple[Pokemon, bool]] = []
    for pokemon in op_state.active:
        if pokemon is not None:
            board.append((pokemon, True))
    for pokemon in op_state.bench:
        if pokemon is not None:
            board.append((pokemon, False))

    # ダメカン配置の時点で、直接攻撃(200)はすでに解決済み(バトルポケモンが hp=0 のまま
    # 場に残っている)。ただし **そのKOのサイドはまだ取られていない**(実測: 配置が全部
    # 終わってから myprize が減る)。したがって残りサイドから「確定しているKO分」を先に
    # 引かないと、プランナのサイド算術が丸ごとずれる。
    pending_prizes = 0
    targets = []
    for pokemon, is_active in board:
        if pokemon.hp <= 0:
            pending_prizes += prize_count(pokemon, is_active)
            continue
        placeable = pokemon.serial in option_index_by_serial
        targets.append(
            damage_counter_plan.TargetState(
                key=pokemon.serial,
                hp10=(pokemon.hp + 9) // 10,  # Pokemon.hp は残りHP(二重に引かない)
                # 直接攻撃(200)はアクティブに、ダメカンはベンチに当たる。レガシーエネルギー等の
                # サイド軽減は「攻撃のダメージ」でのみ効くので is_attack_damage を使い分ける。
                prize=prize_count(pokemon, is_active),
                is_active=is_active,
                counter_protected=(not placeable) or no_damage_counter(pokemon),
                threat=pokemon_score(pokemon, False) / 1000.0,
                direct_immune=no_damage_dex(pokemon.id),
            )
        )
    if not targets:
        return None

    # 自分のサイドの残り枚数 - 確定済みKO分 = これから追加で取らないといけない枚数。
    remaining_prizes = len(my_state.prize) - pending_prizes
    # 手札にボスの指令があれば、次ターン以降にベンチを1体引きずり出せるものとして読む。
    boss_budget = 0
    for card in (my_state.hand or []):
        if card.id == Boss_Orders:
            boss_budget = 1
            break

    config = damage_counter_plan.DamagePlanConfig(horizon=max(1, PLANNER_HORIZON))
    key = damage_counter_plan.choose_next_counter(
        targets,
        int(remain),
        remaining_prizes,
        config=config,
        boss_budget=boss_budget,
    )
    if key is None:
        return None
    index = option_index_by_serial.get(key)
    if index is None:
        return None
    output = [index]
    if not _validate_output(output, select):
        return None
    return output


def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    # league/run_match.py はすでに Observation を渡す。Kaggle 基盤は dict を渡す。両対応。
    obs = obs_dict if isinstance(obs_dict, Observation) else to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return my_deck

    global pre_turn_log
    global current_turn_log

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
            if log.attackId == 323:  # Itchy Pollen
                no_item = True
        elif log.type == LogType.MOVE_CARD:
            if (log.playerIndex == my_index
                and (log.fromArea == AreaType.BENCH or log.fromArea == AreaType.ACTIVE)
                and log.toArea == AreaType.DISCARD):
                pre_ko = True

    if select.deck != None:
        set_card_counts(obs, my_index)
        for card in select.deck:
            card_counts[card.id] -= 1
        prize.clear()
        for id in card_counts:
            for _ in range(card_counts[id]):
                prize.append(id)

    set_card_counts(obs, my_index)
    for id in prize:
        card_counts[id] -= 1
    deck_counts = card_counts

    prize_diff = len(my_state.prize) - len(op_state.prize)

    global bench_attacker

    # Number of cards per card ID on the Bench and in the Active Spot
    field_counts = defaultdict(int)
    # Number of cards per card ID in hand
    hand_counts = defaultdict(int)
    # Number of cards per card ID in discard pile
    discard_counts = defaultdict(int)

    active_id = 0
    bench_attacker = False
    can_evolve_dreepy = False
    evolve_dreepy_count = 0
    can_evolve_drakloak = False
    damage = 200
    for card in my_state.active:
        if card == None:
            continue
        active_id = card.id
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
    for card in my_state.bench:
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
        if card.id == Dragapult_ex and len(card.energies) >= 2:
            bench_attacker = True
    main_pokemon_count = field_counts[Dreepy] + field_counts[Drakloak] + field_counts[Dragapult_ex]
    no_more_dex = (field_counts[Dragapult_ex] * 2 >= len(op_state.prize))

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    support_count = 0

    for card in my_state.discard:
        discard_counts[card.id] += 1

    def attach_score(attach_id: int, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        if card_table[attach_id].cardType == CardType.TOOL:
            # Attach tool
            score = 60000
            if active:
                score += 1000
            return score

        # Attach energy
        if pokemon.id == Budew:
            return -1
        elif pokemon.id == Meowth_ex or pokemon.id == Fezandipiti_ex or pokemon.id == Latias_ex:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                if bench_attacker or field_counts[Budew] >= 1:
                    return 22000
                else:
                    return 18000
            else:
                return -1
        if active and can_main_attack:
            return -1
        score = 20000
        if energy_count >= 2:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                score += 200
            else:
                return -1
        elif energy_count == 1:
            if attach_id == pokemon.energyCards[0].id:
                return -1
            if pokemon.id == Dragapult_ex:
                score += 250
            elif pokemon.id == Dreepy:
                score -= 150
            else:
                score -= 200
            if active:
                score += 200
        else:  # energy_count == 0
            if active:
                if bench_attacker:
                    score += 400
            else:
                if pokemon.id == Dragapult_ex:
                    score += 150
                elif pokemon.id == Dreepy:
                    score += 100
                else:
                    score += 50
                if bench_attacker:
                    score -= 200
        if no_more_dex and (pokemon.id == Dreepy or pokemon.id == Drakloak):
            score -= 500
        return score

    def hand_score(id: int, ignore_count: bool):
        score = 0
        if id == Dreepy:
            if main_pokemon_count >= 3:
                score = 1000
            else:
                score = 18000
        elif id == Drakloak:
            if can_evolve_dreepy:
                score = 20000
            else:
                score = 3000
        elif id == Dragapult_ex:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Rare_Candy] >= 1 and not no_item:
                score = 40000
            elif can_evolve_drakloak:
                if field_counts[id] == 0:
                    score = 30000
                elif field_counts[id] == 1:
                    score = 10000
                else:
                    score = 50
            else:
                if field_counts[id] >= 2:
                    score = 50
                else:
                    score = 2000
        elif id == Fezandipiti_ex:
            if pre_ko:
                score = 50000
            elif prize_diff <= -2:
                score = 5
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
        elif id == Latias_ex:
            if active_id == Fezandipiti_ex or active_id == Meowth_ex or active_id == Dreepy:
                if field_counts[Drakloak] + field_counts[Dragapult_ex] == 0:
                    score = 28000
                else:
                    score = 15000
            else:
                score = 10
        elif id == Budew:
            if field_counts[id] + field_counts[Drakloak] + field_counts[Dragapult_ex] >= 1:
                score = UNNECESSARY
            elif state.turn >= 2:
                score = 30000
        elif id == Meowth_ex:
            if support_count > hand_counts[Boss_Orders] or stadium_id == Team_Rocket_Watchtower:
                score = 5
            elif state.supporterPlayed:
                score = 40
            else:
                score = 35000
        elif id == Rare_Candy:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Dragapult_ex] >= 1:
                score = 40000
        elif id == Unfair_Stamp:
            if pre_ko:
                score = 80000
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
            else:
                score = 80
        elif id == Buddy_Buddy_Poffin:
            count = deck_counts[Dreepy]
            if count == 0:
                score = UNNECESSARY
            else:
                if state.turn <= 2 and field_counts[Budew] == 0 and deck_counts[Budew] >= 1:
                    count += 1
                if count >= 2:
                    score = 35000
        elif id == Night_Stretcher:
            for i in discard_counts:
                if discard_counts[i] >= 1:
                    card_type = card_table[i].cardType
                    if card_type == CardType.POKEMON or card_type == CardType.BASIC_ENERGY:
                        score = max(score, hand_score(i, ignore_count))
        elif id == Crushing_Hammer:
            score = 20
        elif id == Ultra_Ball:
            if main_pokemon_count <= 2 or field_counts[Dreepy] >= 1:
                score = 70
            else:
                score = 5
        elif id == Poke_Pad:
            score = max(hand_score(Dreepy, ignore_count), hand_score(Drakloak, ignore_count))
        elif id == Lucky_Helmet:
            score = 15
        elif id == Boss_Orders:
            if plan_a.attack > 0:
                score = 60000
        elif id == Crispin:
            if not ignore_count or support_count == 0:
                if deck_counts[Basic_Fire_Energy] == 0 or deck_counts[Basic_Psychic_Energy] == 0:
                    score = 10
                if not can_main_attack and not bench_attacker and field_counts[Dragapult_ex] >= 1:
                    score = 55000
                else:
                    score = 25000
        elif id == Brock_Scouting:
            if not ignore_count or support_count == 0:
                if state.turn == 2 and field_counts[Budew] + field_counts[Latias_ex] == 0:
                    score = 50000
                else:
                    score = 30000
        elif id == Lillie_Determination:
            if not ignore_count or support_count == 0:
                score = 45000
        elif id == Team_Rocket_Watchtower:
            if stadium_id != 0 and stadium_id != Team_Rocket_Watchtower:
                score = 4000
        elif id == Basic_Fire_Energy or id == Basic_Psychic_Energy:
            if can_main_attack and (len(op_state.prize) <= 2
                or (bench_attacker and len(op_state.prize) <= 4)):
                score = UNNECESSARY
            else:
                max_score = -10000
                for pokemon in my_state.active:
                    if pokemon == None:
                        continue
                    max_score = max(max_score, attach_score(id, pokemon, True))
                for pokemon in my_state.bench:
                    max_score = max(max_score, attach_score(id, pokemon, False))
                score = max_score - 5000
                if can_main_attack or bench_attacker:
                    score /= 10

        if not ignore_count and hand_counts[id] > 0:
            if id == Drakloak and hand_counts[id] < evolve_dreepy_count:
                score -= 10
            elif id == Dreepy:
                score -= 100
            else:
                score -= 100000
        return score

    global use_support
    if context == SelectContext.MAIN:
        main_option_proc(obs, damage)

        use_support = 0
        if not state.supporterPlayed:
            support_score = 0
            for o in select.option:
                if o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, o.index, state.yourIndex)
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        score = hand_score(card.id, True)
                        if support_score < score:
                            support_score = score
                            use_support = card.id

    hand_scores = []
    negative_hand_count = 0
    for card in my_state.hand:
        score = hand_score(card.id, False)
        hand_scores.append(score)
        if score < 0:
            negative_hand_count += 1
        hand_counts[card.id] += 1
        if card_table[card.id].cardType == CardType.SUPPORTER and card.id != Boss_Orders:
            support_count += 1

    no_draw = (my_state.deckCount <= 8)  # Whether to restrict actions that reduce the deck
    do_switch = (not can_main_attack and (bench_attacker or (active_id != Budew and field_counts[Budew] >= 1 and state.turn >= 2)))
    effect_card_id = 0 if select.effect == None else select.effect.id
    context_card_id = 0 if select.contextCard == None else select.contextCard.id

    scores = []  # Score for each action
    for o in select.option:
        score = 0  # The default and baseline score is 0.
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = -1
            else:
                score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card != None:
                energy_count = 0
                hp = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                    hp = card.hp
                if (context == SelectContext.SWITCH
                    or context == SelectContext.TO_ACTIVE
                    or context == SelectContext.SETUP_ACTIVE_POKEMON):
                    # Selection of the Pokémon to send to the Active Spot
                    if o.playerIndex == my_index:
                        if card.id == Dreepy:
                            score += 10000
                        elif card.id == Drakloak:
                            if energy_count >= 1:
                                score += 20000
                            else:
                                score -= 10000
                        elif card.id == Dragapult_ex:
                            score += 50000
                        elif card.id == Budew:
                            if context != SelectContext.SWITCH:
                                score += 100000
                            elif not bench_attacker:
                                score += 30000
                        elif card.id == Fezandipiti_ex:
                            score -= 1000
                        elif card.id == Meowth_ex:
                            score -= 2000
                    else:
                        if plan_a.attack == o.index + 1:
                            score += 100000
                    score += energy_count * 1000
                    score += hp
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if my_index == state.firstPlayer or card.id != Dreepy:
                        score = -1
                elif context == SelectContext.TO_BENCH or context == SelectContext.TO_HAND:
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                    if effect_card_id == Crispin:
                        # Reverse scoring
                        score = 100000 - hand_score(card.id, True)
                elif context == SelectContext.DISCARD:
                    hand_counts[card.id] -= 1
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        support_count -= 1
                    score = -hand_score(card.id, False)
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE_COUNTER_ANY:
                    if hp > 0:
                        score = 100000 - 10 * hp + pokemon_score(card, False)
                        if context == SelectContext.DAMAGE_COUNTER:
                            if 210 <= hp <= 230:
                                score += 20000 + hp * 20
                                if o.area == AreaType.ACTIVE:
                                    score += 10000
                            elif 40 <= hp <= 90:
                                score += 10000 + hp * 20
                            elif hp <= 30:
                                score += -10000 + hp * 20
                            if card.id == 133 or card.id == 351:
                                score += 30000
                        else:
                            index = o.index + 1
                            if index in plan_b.counter:
                                score += 100000
                            else:
                                remain_damage = select.remainDamageCounter * 10
                                if 210 <= hp <= 200 + remain_damage:
                                    score += 30000
                                elif 20 <= hp <= 60 + remain_damage:
                                    score += 10000
                                elif hp == 10:
                                    score -= 100000
                            if no_damage_counter(card):
                                score = -1
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    if card.id == Dragapult_ex:
                        score += 200
        elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
            # Discarding energy (Retreat or Crushing Hammer)
            if o.playerIndex != state.yourIndex:
                if o.area == AreaType.BENCH:
                    score = 20
                else:
                    score = 10
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if card_table[card.id].cardType == CardType.SPECIAL_ENERGY:
                    score += 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            card_score = hand_scores[o.index]
            if card.id == Dreepy:
                score = 51000
            elif card.id == Fezandipiti_ex:
                if card_score > 0:
                    score = 53000
                else:
                    score = -1
            elif card.id == Latias_ex:
                if active_id != Drakloak and active_id != Dragapult_ex:
                    score = 51000
                else:
                    score = -1
            elif card.id == Budew:
                if field_counts[Budew] == 0 and field_counts[Dragapult_ex] == 0:
                    score = 52000
                else:
                    score = -1
            elif card.id == Meowth_ex:
                if state.supporterPlayed or stadium_id == Team_Rocket_Watchtower:
                    score = -1
                elif support_count == 0:
                    score = 50000
                elif support_count == hand_counts[Boss_Orders] and not plan_a.attack <= 0:
                    score = 50000
                else:
                    score = -1
            elif card.id == Rare_Candy:
                if no_more_dex:
                    score = -1
                else:
                    score = 75000
            elif card.id == Unfair_Stamp:
                score = 15000
            elif card.id == Night_Stretcher:
                if card_score >= 18000:
                    score = 42000
                else:
                    score = -1
            elif card.id == Crushing_Hammer:
                score = 40000
            elif card.id == Boss_Orders:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
            elif card.id == Lillie_Determination:
                if card.id == use_support:
                    score = 14000
                else:
                    score = -1
            elif card.id == Team_Rocket_Watchtower:
                if stadium_id > 0 or state.turn == 1:
                    score = 80000
                else:
                    score = -1
            elif no_draw:
                score = -1
            elif card.id == Buddy_Buddy_Poffin:
                if deck_counts[Dreepy] > 0:
                    score = 46000
                else:
                    score = -1
            elif card.id == Ultra_Ball:
                if negative_hand_count >= 2:
                    score = 44000
                else:
                    score = -1
            elif card.id == Poke_Pad:
                if deck_counts[Dreepy] + deck_counts[Drakloak] > 0:
                    score = 45000
                else:
                    score = -1
            elif card.id == Crispin or card.id == Brock_Scouting:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pokemon.energies)
            if pokemon.id == Dreepy:
                score += 30000
            elif field_counts[Dragapult_ex] >= 2 or (field_counts[Dragapult_ex] == 1 and len(op_state.prize) <= 2):
                score = -1
            else:
                score += 70000
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if no_draw:
                score = -1
            elif card.id == 1267:  # Lumiose City
                score = 1
            else:
                score = 40000
        elif o.type == OptionType.RETREAT:
            if do_switch:
                score = 10000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = o.attackId

        scores.append(score)

    output = []
    if len(scores) >= 1:
        # Select in descending order of score
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            # If the score is negative, do not select it if skipping is possible
            if (sorted_scores[i][1] >= 0
                or select.minCount > i
                or (context != SelectContext.TO_BENCH and context != SelectContext.SETUP_BENCH_POKEMON)):
                output.append(sorted_scores[i][0])

    # --- 派生版の唯一の差分: ダメカン任意配置だけプランナに置き換える ---------------
    # ここまでで kiyotah 版の output は完成している。プランナが妥当な手を返したときだけ
    # 差し替え、それ以外(None/例外/選択肢に無い)は元の output をそのまま使う。
    if context == SelectContext.DAMAGE_COUNTER_ANY:
        try:
            planned = _planner_damage_counter_choice(obs)
        except Exception:  # noqa: BLE001 - プランナの不調で対戦を止めない
            planned = None
        if planned is not None:
            return planned

    return output
