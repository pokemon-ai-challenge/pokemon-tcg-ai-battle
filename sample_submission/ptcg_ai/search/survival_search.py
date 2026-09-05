"""延命(自滅deckout回避)の補助: 選択肢を1手仮実行した後の「自分の山札枚数」を返す。

用途: WS3 延命veto。山札が危険域のとき「引いて山を0にする不要な自滅」を検出し、山を残す手に
差し替えるための最小道具。lethal/ko_search と同じ ``cg.api.search_begin``/``search_step`` を使う。
判定不能/例外は None(安全側=vetoしない)。

注意(設計書 piloting-fixes/design.md WS3): deckout負けは"次の自ターン開始時に引けない"時に起きる
ため自ターン内にクリーンな終端が無い。本モジュールは「この1手の後の自山札枚数」という近似で、
"引いて即0"の明白な自滅だけを捉える(完全な延命最適化ではない)。deck_sustainヒューリスティクスが
中立だった前例[[project_decksustain_scaffold]]の通り効果は限定的見込み。
"""
from __future__ import annotations

from cg import api as cg_api


def deckcount_after(obs, option_index: int, hidden_state_factory) -> int | None:
    """``option_index`` を1手仮実行した後の自分の deckCount を返す。不能/例外は None。"""
    try:
        if obs is None or obs.current is None or obs.select is None:
            return None
        if not (0 <= option_index < len(obs.select.option)):
            return None
        hs = hidden_state_factory()
        if hs is None:
            return None
        me = obs.current.yourIndex
        node = cg_api.search_begin(
            obs, hs["your_deck"], hs["your_prize"], hs["opponent_deck"],
            hs["opponent_prize"], hs["opponent_hand"], hs["opponent_active"],
        )
        try:
            child = cg_api.search_step(node.searchId, [option_index])
            cs = child.observation.current
            dc = None
            if cs is not None and 0 <= me < len(cs.players):
                dc = cs.players[me].deckCount
            try:
                cg_api.search_release(child.searchId)
            except Exception:
                pass
            return dc
        finally:
            try:
                cg_api.search_release(node.searchId)
            except Exception:
                pass
    except Exception:  # noqa: BLE001 - 探索失敗が意思決定を止めてはならない
        return None
