from cg.api import Observation, OptionType, all_card_data, CardData

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS

# カードデータのキャッシュ（毎ターン呼び出しを避ける）
_card_data_cache: dict[int, CardData] | None = None


def _get_card_data_map() -> dict[int, CardData]:
    """cardId → CardData の辞書を返す（初回のみロード）。"""
    global _card_data_cache
    if _card_data_cache is None:
        _card_data_cache = {c.cardId: c for c in all_card_data()}
    return _card_data_cache


# ドロー・サーチ系とみなす英語キーワード（skillテキスト内を検索）
_DRAW_KEYWORDS = (
    "draw",
    "search",
    "look at",
    "put into your hand",
    "add to your hand",
    "into your hand",
)


def _is_draw_ability(card: CardData) -> bool:
    """特性テキストにドロー・サーチ系の記述があるか判定する。"""
    for skill in card.skills:
        text_lower = skill.text.lower()
        if any(kw in text_lower for kw in _DRAW_KEYWORDS):
            return True
    return False


def propose_ability_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """特性を使う候補を返す。ドロー・サーチ系特性は最優先で使う。"""
    if not buckets.ability:
        return None
    if obs.current is None:
        return None

    card_map = _get_card_data_map()
    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]

    def get_card_id_from_option_idx(opt_idx: int) -> int | None:
        """optionインデックス → そのカードID を返す。"""
        if obs.select is None:
            return None
        opt = obs.select.option[opt_idx]
        if opt.type != OptionType.ABILITY:
            return None
        from cg.api import AreaType
        if opt.area == AreaType.ACTIVE:
            poke = player.active[opt.index] if player.active else None
        elif opt.area == AreaType.BENCH:
            poke = player.bench[opt.index] if opt.index < len(player.bench) else None
        else:
            return None
        return poke.id if poke is not None else None

    draw_candidates: list[int] = []
    normal_candidates: list[int] = []

    for opt_idx in buckets.ability:
        card_id = get_card_id_from_option_idx(opt_idx)
        if card_id is not None and card_id in card_map:
            card = card_map[card_id]
            if _is_draw_ability(card):
                draw_candidates.append(opt_idx)
            else:
                normal_candidates.append(opt_idx)
        else:
            normal_candidates.append(opt_idx)

    if draw_candidates:
        score = MAIN_ACTION_BASE_WEIGHTS["draw_or_search"] + 15  # 85
        return MainActionProposal(
            action=[draw_candidates[0]],
            score=score,
            label="ability_draw",
        )

    if normal_candidates:
        return MainActionProposal(
            action=[normal_candidates[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["ability"],  # 58
            label="ability",
        )

    return None