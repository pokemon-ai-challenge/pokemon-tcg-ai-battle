"""クセロシキ等の「手札を N 枚捨てて 3 枚に減らす」discard select 用の keep-set 評価(手書き)。

方針(ユーザー設計): per-card の factor 加点だけでなく **keep-SET(残す3枚の組)** を評価する。
factor: (1)ドロー/サーチ札=手札(=フーディン打点)の建て直し最優先, (2)ノコッチ=deckout連動で
山低い時に加点(ループ維持), (3)ボスの指令=lethal/KOが見える時に加点, (4)回収札, (5)進化パーツ
(ふしぎなアメ+進化先のシナジー), (6)エネ不足なら必要エネ, (7)相手手札多い時は自分のクセロシキ。
組合せ補正: 進化シナジー加点 / 重複サポート減点(次ターン1枚しか使えない)。

すべて現在 State + カードデータからの決定的評価(ML でなく手書き=off-distribution 無し, deckout の
ルール成功と同じ思想)。ID/重みは tunable(このデッキ用の初期値)。
"""
from __future__ import annotations

import itertools

from cg.api import CardType

from ptcg_ai.shared import card_cache

# --- このデッキのカード分類(tunable) ---
NOKOCCHI_ID = 66          # ノココッチ(にげあしドロー=ドロー+ループ)
DUNSPARCE_ID = 65         # ノコッチ(ループの進化前)
BOSS_ID = 1182            # ボスの指令(lethal時に釣り出し)
XEROSIC_ID = 1197         # クセロシキのたくらみ(ハンデス=counter)
RARE_CANDY_ID = 1079      # ふしぎなアメ
ALAKAZAM_ID = 743         # アラカザム(進化先/主砲)
ABRA_ID = 741
KADABRA_ID = 742
DRAW_SEARCH_IDS = {66, 1086, 1152, 1225, 1231}   # ノコッチ特性/ポフィン/ポケパッド/ドローサポ(手札建て直し)
RECOVERY_IDS = {1129, 1097, 1184}                # せいなるはい/夜の鉱山/ラナ
EVO_PIECE_IDS = {1079, 741, 742, 743}            # ふしぎなアメ/Abra/Kadabra/Alakazam


def _card(cid: int):
    try:
        return card_cache.get_card(cid)
    except Exception:  # noqa: BLE001
        return None


def keep_value(card_id: int, state, ctx: dict) -> float:
    """1枚の keep 価値(文脈 ctx: deck_low, energy_short, opp_hand_big, can_ko を含む)。"""
    if card_id is None or card_id <= 0:
        return 0.0
    card = _card(card_id)
    ct = card.cardType if card is not None else None
    v = 0.0
    # (1) ドロー/サーチ=手札(=打点)の建て直し最優先
    if card_id in DRAW_SEARCH_IDS:
        v += 5.0
    # (2) ノコッチ=deckout連動(山低い時ループ維持)
    if card_id in (NOKOCCHI_ID, DUNSPARCE_ID):
        v += 4.0 if ctx.get("deck_low") else 1.5
    # (3) ボスの指令=lethal/KOが見える時
    if card_id == BOSS_ID:
        v += 5.0 if ctx.get("can_ko") else 1.0
    # (4) 回収札
    if card_id in RECOVERY_IDS:
        v += 2.0
    # (5) 進化パーツ(単体, シナジーは set 側で加点)
    if card_id in EVO_PIECE_IDS:
        v += 1.5
    # (6) エネ不足なら必要エネ
    if ct in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
        v += 2.5 if ctx.get("energy_short") else 1.0
    # (7) 相手手札多い時は自分のクセロシキ(ハンデス返し)
    if card_id == XEROSIC_ID:
        v += 2.5 if ctx.get("opp_hand_big") else 0.8
    # たね/進化ポケモンは最低限の展開価値
    if ct == CardType.POKEMON and v == 0.0:
        v += 1.0
    return v


# 同種でも複数持ちが有用なポケモン(ループ起点/たね)。ここ以外のポケモンは重複を減点。
_DUP_OK_POKEMON = {NOKOCCHI_ID, DUNSPARCE_ID, ABRA_ID}


def _set_adjust(keep_ids: list[int], ctx: dict) -> float:
    """keep-SET の組合せ補正(進化シナジー加点 / サポート・ポケモン重複減点)。"""
    adj = 0.0
    ids = list(keep_ids)
    # ふしぎなアメ + 進化先(Alakazam)を両方残す=進化シナジー加点
    if RARE_CANDY_ID in ids and (ALAKAZAM_ID in ids or ctx.get("alakazam_in_play")):
        adj += 3.0
    # Abra を残すなら進化の起点として少し加点
    if ABRA_ID in ids and RARE_CANDY_ID in ids:
        adj += 1.5
    # サポートは1ターン1枚しか使えない → keep 内の2枚目以降のサポート(異種含む)を減点。
    n_support = 0
    # 同種ポケモン(非ループ/非たね起点)の2枚目以降を減点(例: Alakazam 2枚は冗長)。
    poke_count: dict[int, int] = {}
    for cid in ids:
        card = _card(cid)
        if card is None:
            continue
        if card.cardType == CardType.SUPPORTER:
            n_support += 1
        elif card.cardType == CardType.POKEMON and cid not in _DUP_OK_POKEMON:
            poke_count[cid] = poke_count.get(cid, 0) + 1
    if n_support > 1:
        adj -= 2.5 * (n_support - 1)  # 2枚目以降のサポート(異種含む)
    for cid, cnt in poke_count.items():
        if cnt > 1:
            adj -= 1.5 * (cnt - 1)     # 同種ポケモン(非ループ)の2枚目以降
    # 少なくとも1枚のドロー/サーチが keep に含まれる=手札建て直し保証ボーナス
    if any(cid in DRAW_SEARCH_IDS for cid in ids):
        adj += 1.0
    return adj


def best_keep_discard_indices(option_ids: list[int], keep_count: int, state, ctx: dict) -> list[int] | None:
    """options(捨てる候補=手札カードID列)から keep_count 枚を残す最良 keep-set を選び、
    **捨てる option index のリスト**(=keepしない index)を返す。判定不能は None。"""
    n = len(option_ids)
    if keep_count <= 0 or keep_count >= n:
        return None
    best_idx_set = None
    best_score = None
    for keep_idx in itertools.combinations(range(n), keep_count):
        ids = [option_ids[i] for i in keep_idx]
        score = sum(keep_value(c, state, ctx) for c in ids) + _set_adjust(ids, ctx)
        if best_score is None or score > best_score:
            best_score, best_idx_set = score, set(keep_idx)
    if best_idx_set is None:
        return None
    return [i for i in range(n) if i not in best_idx_set]  # 捨てる=keepしない index
