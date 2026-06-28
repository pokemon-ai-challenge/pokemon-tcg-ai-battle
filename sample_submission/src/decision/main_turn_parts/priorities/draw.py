from cg.api import Observation

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS
from src.knowledge.card_cache import load_card_data

# ---------------------------------------------------------------------------
# Skill.text のキーワードでサポーター・グッズの種別を判定する。
# カード名リストは持たず、効果テキストだけで分類するため、
# 未知のカードにも対応できる。
# ---------------------------------------------------------------------------

# 手札枚数の段階ごとの加点。base に足す。
_DRAW_SCORE_BONUS = {
    "emergency": 30,   # 0〜2枚: 緊急 → 最優先でドロー
    "normal":    15,   # 3〜4枚: 普通 → 積極的に使う
    "enough":     0,   # 5〜6枚: 十分 → 使えるなら使う程度
    # 7枚以上はドロー目的では発動しない
}

# サポーターの種別ごとの追加点。draw > search > discard_draw の順で優先される。
_SUPPORTER_TYPE_BONUS = {
    "draw":                10,   # 純粋ドロー（シロナ・ホップ等）
    "search":               5,   # サーチ系（ネジキ・ポフィン等）
    "discard_draw":         0,   # 手札捨て/戻しドロー（博士の研究・マリィ等）
    "discard_draw_penalty": -20, # 先に使えるカードがある場合の減点
    "other":                0,   # ボスの指令等（ここでは提案しない）
}


def _classify_supporter_text(text: str) -> str:
    """サポーターの効果テキストから種別を返す。

    Returns:
        "discard_draw"  : 手札を捨てて/山に戻してドロー（博士の研究・マリィ等）
        "draw"          : 純粋なドロー（シロナ・ホップ等）
        "search"        : 山札からサーチ（ネジキ・ポフィン等）
        "other"         : 上記以外（ボスの指令等）
    """
    t = text.lower()
    has_draw = "draw" in t
    has_discard = "discard" in t or "shuffle" in t
    has_search = "search" in t or "look at" in t

    if has_draw and has_discard:
        return "discard_draw"
    if has_draw:
        return "draw"
    if has_search:
        return "search"
    return "other"


def _classify_item_text(text: str) -> str:
    """グッズの効果テキストから種別を返す。

    Returns:
        "search" : 山札からポケモン等を持ってくる（ボール系等）
        "draw"   : 手札を増やす（ポケギア等）
        "other"  : 上記以外
    """
    t = text.lower()
    if "search" in t or "look at" in t:
        return "search"
    if "draw" in t:
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
    """supporter_play 内の選択肢の種別を返す。

    Returns:
        "discard_draw" | "draw" | "search" | "other"
    """
    text = _get_card_skill_text(option_index, obs)
    if text is None:
        return "other"
    return _classify_supporter_text(text)


def _get_item_kind(option_index: int, obs: Observation) -> str:
    """item_play 内の選択肢の種別を返す。

    Returns:
        "search" | "draw" | "other"
    """
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
    return None  # 7枚以上: ドロー目的では使わない


def _has_usable_non_draw_cards(obs: Observation, buckets: MainOptionBuckets) -> bool:
    """手札に、進化・グッズ・ポケモン展開など「先に使うべきカード」があるか。

    手札を捨てる/戻すドロー系（discard_draw）を使う前に確認し、
    True なら discard_draw のスコアを下げて後回しにする。
    """
    return bool(
        buckets.pokemon_play
        or buckets.evolve
        or buckets.item_play
        or buckets.tool_play
        or buckets.stadium_play
        or buckets.ability
        or buckets.attach
    )


def propose_draw_or_search_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """手札状況に応じて、ドロー・サーチ系カードを優先候補として返す。

    優先順位（スコアが高い順）:
        1. ドロー系サポーター（純粋ドロー: シロナ・ホップ等）
        2. サーチ系サポーター（ネジキ・ポフィン等）
        3. 手札捨て/戻し系サポーター（博士の研究・マリィ等）
           ただし先に使えるカードがある場合はスコアを下げる
        4. ドロー・サーチ系グッズ（ボール系・ポケギア等）

    ボスの指令など非ドロー系サポーターはここでは扱わない（スコア提案しない）。
    手札 7 枚以上の場合はドロー目的では提案しない。
    """
    if obs.current is None:
        return None

    your_index = obs.current.yourIndex
    hand_count = obs.current.players[your_index].handCount

    hand_bonus = _hand_bonus(hand_count)
    if hand_bonus is None:
        return None  # 手札が十分なのでドロー提案をしない

    base = MAIN_ACTION_BASE_WEIGHTS["draw_or_search"]

    # --- サポーターを種別で走査し、最も優先度の高いものを選ぶ ---
    # 優先度: draw > search > discard_draw > other（other は提案しない）
    KIND_PRIORITY = {"draw": 3, "search": 2, "discard_draw": 1, "other": 0}

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
                break  # 最高優先度が見つかれば即確定

    if best_supporter_index is not None and best_supporter_kind != "other":
        type_bonus = _SUPPORTER_TYPE_BONUS.get(best_supporter_kind, 0)

        # discard_draw（手札捨て/戻し系）は先に使えるカードがあればスコアを下げる
        if best_supporter_kind == "discard_draw" and _has_usable_non_draw_cards(obs, buckets):
            type_bonus += _SUPPORTER_TYPE_BONUS["discard_draw_penalty"]

        score = base + hand_bonus + type_bonus
        return MainActionProposal(
            action=[best_supporter_index],
            score=score,
            label=f"draw_or_search_supporter_{best_supporter_kind}",
        )

    # --- グッズからドロー・サーチ系を探す ---
    for idx in buckets.item_play:
        kind = _get_item_kind(idx, obs)
        if kind in ("search", "draw"):
            score = base + hand_bonus - 5  # サポーターより若干低め
            return MainActionProposal(
                action=[idx],
                score=score,
                label=f"draw_or_search_item_{kind}",
            )

    return None