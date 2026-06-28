from cg.api import Attack, CardData, CardType, LogType, Observation, all_attack, all_card_data

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS

# ---------------------------------------------------------------------------
# カードデータ・攻撃データのキャッシュ（呼び出しごとに毎回生成しない）
# ---------------------------------------------------------------------------

_card_cache: dict[int, CardData] | None = None
_attack_cache: dict[int, Attack] | None = None


def _get_card_data() -> dict[int, CardData]:
    global _card_cache
    if _card_cache is None:
        _card_cache = {c.cardId: c for c in all_card_data()}
    return _card_cache


def _get_attack_data() -> dict[int, Attack]:
    global _attack_cache
    if _attack_cache is None:
        _attack_cache = {a.attackId: a for a in all_attack()}
    return _attack_cache

def _has_usable_non_draw_cards(obs: Observation, buckets: MainOptionBuckets) -> bool:
    """手札に「先に使うべきカード」があるか。

    discard_draw / draw_to_x を使う前に確認し、True ならスコアを下げる。
    item_play はドロー系グッズ（ボール・ポケギア等）も含むため、
    手札に「ドロー目的以外のグッズ」があるかをテキストで絞り込む。
    """
    has_non_draw_items = False
    for idx in buckets.item_play:
        kind = _get_item_kind(idx, obs)
        if kind not in ("draw", "draw_to_x", "search"):
            # 効果がドロー・サーチ以外のグッズが1枚でもあれば真
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

def _get_item_kind(option_index: int, obs: Observation) -> str:
    text = _get_card_skill_text(option_index, obs)
    if text is None:
        return "other"
    return _classify_item_text(text)

def _classify_item_text(text: str) -> str:
    """グッズの効果テキストから種別を返す。

    Returns:
        "search"       : 山札からポケモン等を持ってくる（ボール系等）
        "draw"         : 手札を増やす（ポケギア等）
        "draw_to_x"    : 手札がX枚になるまでドロー
        "other"        : 上記以外
    """
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
    card_data = _get_card_data().get(hand_card.id)
    if card_data is None or not card_data.skills:
        return None

    return card_data.skills[0].text


# ---------------------------------------------------------------------------
# 効果テキスト判定ユーティリティ
# ---------------------------------------------------------------------------

def _is_cant_use_next_turn(attack_text: str) -> bool:
    """「次のターン技が使えない」系の効果テキストか判定する。

    例: "can't use" + "next turn" / "during your next turn" 等
    """
    t = attack_text.lower()
    return ("can't use" in t or "cannot use" in t) and "next turn" in t


def _is_retreat_cost_reducer(skill_text: str) -> bool:
    """にげるコストを下げる/0にするどうぐのテキストか判定する。

    例: "retreat cost" + ("less" / "0" / "no" / "free" / "reduced")
    """
    t = skill_text.lower()
    if "retreat cost" not in t:
        return False
    return any(kw in t for kw in ("less", " 0", "no ", "free", "reduc"))


# ---------------------------------------------------------------------------
# 前ターンの攻撃ログから「次ターン技不可」状態かを判定
# ---------------------------------------------------------------------------

def _active_used_cant_use_attack(obs: Observation) -> bool:
    """バトルポケモンが直前ターンに「次のターン技が使えない」技を使ったか。

    obs.logs から自分の ATTACK ログを探し、その技テキストで判定する。
    """
    if obs.current is None:
        return False

    your_index = obs.current.yourIndex
    attack_data = _get_attack_data()

    for log in obs.logs:
        if log.type == LogType.ATTACK and log.playerIndex == your_index:
            if log.attackId is None:
                continue
            attack = attack_data.get(log.attackId)
            if attack and _is_cant_use_next_turn(attack.text):
                return True
    return False


# ---------------------------------------------------------------------------
# ベンチの最大打点計算（エネルギー充足チェック付き）
# ---------------------------------------------------------------------------

def _max_damage_of_pokemon(pokemon_id: int, energies: list) -> int:
    """ポケモンが現在のエネルギーで出せる最大ダメージを返す。

    エネルギーが足りない技はスキップする。
    """
    card_data = _get_card_data()
    attack_data = _get_attack_data()

    card = card_data.get(pokemon_id)
    if card is None:
        return 0

    energy_counts: dict[int, int] = {}
    for e in energies:
        energy_counts[int(e)] = energy_counts.get(int(e), 0) + 1

    max_dmg = 0
    for attack_id in card.attacks:
        attack = attack_data.get(attack_id)
        if attack is None:
            continue
        # エネルギー充足チェック（COLORLESS=0 は何でも代替可）
        required: dict[int, int] = {}
        colorless_needed = 0
        for e in attack.energies:
            if int(e) == 0:  # COLORLESS
                colorless_needed += 1
            else:
                required[int(e)] = required.get(int(e), 0) + 1

        can_use = True
        remaining = dict(energy_counts)
        for e_type, count in required.items():
            have = remaining.get(e_type, 0)
            if have < count:
                can_use = False
                break
            remaining[e_type] = have - count

        if can_use:
            # 残りエネルギーでCOLORLESSを補填
            total_remaining = sum(remaining.values())
            if total_remaining >= colorless_needed:
                max_dmg = max(max_dmg, attack.damage)

    return max_dmg


def _bench_has_higher_damage_pokemon(obs: Observation) -> bool:
    """ベンチに、バトルポケモンより高い打点を出せるポケモンがいるか。"""
    if obs.current is None:
        return False

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]

    if not player.active or player.active[0] is None:
        return False

    active = player.active[0]
    active_dmg = _max_damage_of_pokemon(active.id, active.energies)

    for bench_poke in player.bench:
        bench_dmg = _max_damage_of_pokemon(bench_poke.id, bench_poke.energies)
        if bench_dmg > active_dmg:
            return True
    return False


# ---------------------------------------------------------------------------
# にげるコスト軽減どうぐを手札から探す
# ---------------------------------------------------------------------------

def _find_retreat_tool_index(obs: Observation, buckets: MainOptionBuckets) -> int | None:
    """tool_play の中から「にげるコスト軽減」どうぐの option_index を返す。"""
    if obs.current is None or obs.select is None:
        return None

    your_index = obs.current.yourIndex
    hand = obs.current.players[your_index].hand
    if hand is None:
        return None

    card_data = _get_card_data()

    for idx in buckets.tool_play:
        option = obs.select.option[idx]
        if option.index is None or option.index >= len(hand):
            continue
        hand_card = hand[option.index]
        card = card_data.get(hand_card.id)
        if card is None:
            continue
        for skill in card.skills:
            if _is_retreat_cost_reducer(skill.text):
                return idx
    return None


# ---------------------------------------------------------------------------
# propose 関数
# ---------------------------------------------------------------------------

def propose_pokemon_or_evolve_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """ポケモン展開や進化を候補として返す。

    - 進化を展開より優先する（進化できるなら即進化）
    - ベンチに空きがあればポケモンを展開する
    """
    if obs.current is None:
        return None

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    bench_space = player.benchMax - len(player.bench)

    # 進化を最優先（進化すれば即戦力が上がる）
    if buckets.evolve:
        return MainActionProposal(
            action=[buckets.evolve[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["evolve"] + 20,
            label="evolve",
        )

    if buckets.pokemon_play and bench_space > 0:
        score = MAIN_ACTION_BASE_WEIGHTS["pokemon_play"] + min(bench_space, 2) * 2
        return MainActionProposal(
            action=[buckets.pokemon_play[0]],
            score=score + 20,
            label="pokemon_play",
        )

    return None


def propose_board_item_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """グッズ、スタジアム、どうぐで盤面を整える候補を返す。

    優先順位:
        1. 「次ターン技不可」状態 + ベンチに高打点がいる → にげるコスト軽減どうぐ
        2. スタジアム（場に出ていなければ即置く）
        3. その他グッズ
        4. その他どうぐ
    """
    if obs.current is None:
        return None

    # --- にげるコスト軽減どうぐ（メガブレイブ後の交代準備）---
    if (
        not obs.current.retreated          # まだ逃げていない
        and _active_used_cant_use_attack(obs)
        and _bench_has_higher_damage_pokemon(obs)
    ):
        retreat_tool_idx = _find_retreat_tool_index(obs, buckets)
        if retreat_tool_idx is not None:
            return MainActionProposal(
                action=[retreat_tool_idx],
                score=MAIN_ACTION_BASE_WEIGHTS["tool"] + 30,  # 通常より高めに設定
                label="retreat_tool_for_swap",
            )

    # --- スタジアム（場に出ていなければ置く）---
    if buckets.stadium_play and not obs.current.stadium:
        return MainActionProposal(
            action=[buckets.stadium_play[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["stadium"] + 10,
            label="stadium",
        )
    
        # --- ドロー・サーチ系グッズ ---
    for idx in buckets.item_play:
        # kind を判定する関数は draw.py から移すか共通化する
        kind = _get_item_kind(idx, obs)
        if kind in ("search", "draw", "draw_to_x"):
            score = MAIN_ACTION_BASE_WEIGHTS["board_item"] + 5

            if (
                kind == "draw_to_x"
                and _has_usable_non_draw_cards(obs, buckets)
            ):
                score -= 40

            return MainActionProposal(
                action=[idx],
                score=score,
                label=f"board_item_{kind}",
            )

    # --- その他グッズ ---
    if buckets.item_play:
        return MainActionProposal(
            action=[buckets.item_play[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["board_item"],
            label="board_item",
        )

    # --- その他どうぐ ---
    if buckets.tool_play:
        return MainActionProposal(
            action=[buckets.tool_play[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["tool"],
            label="tool",
        )

    return None