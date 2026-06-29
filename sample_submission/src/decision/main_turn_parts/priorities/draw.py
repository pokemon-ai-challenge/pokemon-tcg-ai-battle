from cg.api import Observation

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS
from src.knowledge.card_cache import load_card_data

# ---------------------------------------------------------------------------
# Skill.text のキーワードでサポーター・グッズの種別を判定する。
# ---------------------------------------------------------------------------

_DRAW_SCORE_BONUS = {
    "emergency": 30,   # 0〜2枚: 緊急 → 最優先でドロー
    "normal":    15,   # 3〜4枚: 普通 → 積極的に使う
    "enough":     0,   # 5〜6枚: 十分 → 使えるなら使う程度
    # 7枚以上はドロー目的では発動しない
}

_SUPPORTER_TYPE_BONUS = {
    "draw":                10,   # 純粋ドロー（シロナ・ホップ等）
    "search":               5,   # サーチ系（ネジキ・ポフィン等）
    "discard_draw":         0,   # 手札捨て/戻しドロー（博士の研究・マリィ等）
    "draw_to_x":            0,   # 手札をX枚まで引く（ポケモン研究者等）
    "discard_draw_penalty": -40, # 先に使えるカードがある場合の減点（discard_draw / draw_to_x 共通）
    "other":                0,
}

# 進化ペアが揃っているときの追加ペナルティ（手札から進化後を失うリスク）
_EVOLUTION_PAIR_DISCARD_PENALTY = -20


def _classify_supporter_text(text: str) -> str:
    """サポーターの効果テキストから種別を返す。

    Returns:
        "discard_draw" : 手札を捨てて/山に戻してドロー（博士の研究・マリィ等）
        "draw_to_x"    : 手札が X 枚になるまでドロー（ポケモン研究者等）
                         → 手札が多いと効果が薄いので discard_draw と同様に扱う
        "draw"         : 純粋なドロー（シロナ・ホップ等）
        "search"       : 山札からサーチ（ネジキ・ポフィン等）
        "other"        : 上記以外（ボスの指令等）
    """
    t = text.lower()
    has_draw    = "draw" in t
    has_discard = "discard" in t or "shuffle" in t
    has_search  = "search" in t or "look at" in t

    # 「手札をX枚になるまで引く」パターン
    has_draw_to_x = (
        ("until you have" in t or "so that you have" in t or "up to" in t)
        and "hand" in t
        and has_draw
    )

    if has_draw and has_discard:
        return "discard_draw"
    if has_draw_to_x:
        return "draw_to_x"
    if has_draw:
        return "draw"
    if has_search:
        return "search"
    return "other"


def _classify_item_text(text: str) -> str:
    """グッズの効果テキストから種別を返す。"""
    t = text.lower()
    has_draw = "draw" in t
    has_draw_to_x = (
        ("until you have" in t or "so that you have" in t or "up to" in t)
        and "hand" in t
        and has_draw
    )
    if "search" in t or "look at" in t:
        return "search"
    if has_draw_to_x:
        return "draw_to_x"
    if has_draw:
        return "draw"
    return "other"


def _get_card_skill_text(option_index: int, obs: Observation) -> str | None:
    """手札の option_index 番目のカードの最初の Skill テキストを返す。"""
    if obs.current is None or obs.select is None:
        return None

    option = obs.select.option[option_index]
    your_index = obs.current.yourIndex
    hand = obs.current.players[your_index].hand
    if hand is None or option.index is None or option.index >= len(hand):
        return None

    hand_card = hand[option.index]
    card_data_by_id = {card.cardId: card for card in load_card_data()}
    card_data = card_data_by_id.get(hand_card.id)
    if card_data is None or not card_data.skills:
        return None

    return card_data.skills[0].text


def _get_supporter_kind(option_index: int, obs: Observation) -> str:
    text = _get_card_skill_text(option_index, obs)
    if text is None:
        return "other"
    return _classify_supporter_text(text)


def _get_item_kind(option_index: int, obs: Observation) -> str:
    text = _get_card_skill_text(option_index, obs)
    if text is None:
        return "other"
    return _classify_item_text(text)


def _hand_bonus(hand_count: int) -> int | None:
    """手札枚数に応じた加点を返す。7枚以上は None でドロー不要を示す。"""
    if hand_count <= 2:
        return _DRAW_SCORE_BONUS["emergency"]
    if hand_count <= 4:
        return _DRAW_SCORE_BONUS["normal"]
    if hand_count <= 6:
        return _DRAW_SCORE_BONUS["enough"]
    return None


def _has_usable_non_draw_cards(obs: Observation, buckets: MainOptionBuckets) -> bool:
    """手札に「先に使うべきカード」があるか。"""
    has_non_draw_items = False
    for idx in buckets.item_play:
        kind = _get_item_kind(idx, obs)
        if kind not in ("draw", "draw_to_x", "search"):
            has_non_draw_items = True
            break

    return bool(
        buckets.pokemon_play
        or buckets.evolve
        or has_non_draw_items
        or buckets.tool_play
        or buckets.stadium_play
        or buckets.ability
        or buckets.attach
    )


def _count_evolution_pairs_on_bench(obs: Observation) -> int:
    """ベンチにいる進化前ポケモンに対して、手札に進化後カードが何枚あるかを返す。

    board.py の同名関数と同じロジックだが、draw.py は board.py に依存しないため
    ここに独立して定義する。
    CardData.evolvesFrom でつながりを確認する。
    """
    if obs.current is None:
        return 0

    card_data_list = load_card_data()
    card_data_by_id = {c.cardId: c for c in card_data_list}

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    hand = player.hand
    if hand is None:
        return 0

    bench_ids = {poke.id for poke in player.bench if poke is not None}
    if not bench_ids:
        return 0

    pair_count = 0
    for hand_card in hand:
        card = card_data_by_id.get(hand_card.id)
        if card is None:
            continue
        if hasattr(card, 'evolvesFrom') and card.evolvesFrom:
            for bench_id in bench_ids:
                bench_card = card_data_by_id.get(bench_id)
                if bench_card is None:
                    continue
                if bench_card.name and card.evolvesFrom.lower() in bench_card.name.lower():
                    pair_count += 1
                    break

    return pair_count


# discard_draw / draw_to_x の両方に減点を適用するカテゴリ集合
_LOSS_ON_HAND_KINDS: frozenset[str] = frozenset({"discard_draw", "draw_to_x"})


def propose_draw_or_search_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """手札状況に応じて、ドロー・サーチ系カードを優先候補として返す。

    優先順位（スコアが高い順）:
        1. 純粋ドロー系サポーター（シロナ・ホップ等）
        2. サーチ系サポーター（ネジキ・ポフィン等）
        3. 手札捨て/戻し系・X枚までドロー系サポーター
           （博士の研究・マリィ・ポケモン研究者等）
           → 先に使えるカードがある場合はスコアを下げる
           → ベンチに進化前・手札に進化後ペアがある場合はさらにスコアを下げる
        4. ドロー・サーチ系グッズ（ボール系・ポケギア等）

    手札 7 枚以上の場合は提案しない。
    """
    if obs.current is None:
        return None

    your_index = obs.current.yourIndex
    hand_count = obs.current.players[your_index].handCount

    hand_bonus = _hand_bonus(hand_count)
    if hand_bonus is None:
        return None

    base = MAIN_ACTION_BASE_WEIGHTS["draw_or_search"]

    KIND_PRIORITY = {"draw": 3, "search": 2, "discard_draw": 1, "draw_to_x": 1, "other": 0}

    best_supporter_index: int | None = None
    best_supporter_kind: str = "other"
    best_priority: int = 0

    for idx in buckets.supporter_play:
        kind = _get_supporter_kind(idx, obs)
        priority = KIND_PRIORITY.get(kind, 0)
        if priority > best_priority:
            best_supporter_index = idx
            best_supporter_kind = kind
            best_priority = priority
            if kind == "draw":
                break

    if best_supporter_index is not None and best_supporter_kind != "other":
        type_bonus = _SUPPORTER_TYPE_BONUS.get(best_supporter_kind, 0)

        if best_supporter_kind in _LOSS_ON_HAND_KINDS:
            # 先に使えるカードがあるとペナルティ
            if _has_usable_non_draw_cards(obs, buckets):
                type_bonus += _SUPPORTER_TYPE_BONUS["discard_draw_penalty"]

            # さらに、ベンチに進化前・手札に進化後のペアがある場合は
            # 手札から進化後カードを失うリスクがあるのでペナルティを追加
            # （例: リオルがベンチにいてルカリオEXが手札にある状態で博士の研究を使うと
            #        ルカリオEXを手札から失う可能性がある）
            evolution_pairs = _count_evolution_pairs_on_bench(obs)
            if evolution_pairs > 0:
                type_bonus += _EVOLUTION_PAIR_DISCARD_PENALTY * min(evolution_pairs, 2)

        score = base + hand_bonus + type_bonus
        return MainActionProposal(
            action=[best_supporter_index],
            score=score,
            label=f"draw_or_search_supporter_{best_supporter_kind}",
        )

    return None