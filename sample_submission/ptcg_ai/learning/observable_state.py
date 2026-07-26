"""``observation.current`` からその手番のプレイヤーに見えている盤面だけを取り出す。

もとは ``kaggle_replays/policy_prior/build_dataset.py`` の学習データ生成専用コードだったが、
``ptcg_ai.learning.policy_features`` の状態特徴（本モジュールの出力そのものを入力とする）と
学習データ生成の両方が同一の実装に乗るよう、提出側（唯一の正）へ移した。
``kaggle_replays/policy_prior/build_dataset.py`` はこの関数を import して使う。

## リーク防止（v2 §7 FR-LEAK-001）

返すのは「その手番のプレイヤーに提示された observation」から見える範囲のみ。相手の
非公開情報（手札の中身・伏せカードの中身）は ``current`` 側で既に null/[null]*6 として
渡ってくるため、ここでは observable_state を素通しするだけで非公開情報は混入しない。
"""

from __future__ import annotations


def observable_state(current: dict, me: int) -> dict:
    """その手番のプレイヤーが見えている盤面だけを取り出す。"""
    players = current.get("players") or [{}, {}]
    own, opp = players[me], players[1 - me]

    def in_play(entity):
        if not isinstance(entity, dict):
            return None
        return {
            "card_id": entity.get("id"),
            "hp": entity.get("hp"),
            "max_hp": entity.get("maxHp"),
            "n_energy": len(entity.get("energies") or []),
            "n_tool": len(entity.get("tools") or []),
            "appear_this_turn": entity.get("appearThisTurn"),
        }

    def side(player: dict) -> dict:
        return {
            "active": in_play((player.get("active") or [None])[0]),
            "bench": [in_play(e) for e in (player.get("bench") or [])],
            "n_prize": len(player.get("prize") or []),
            "n_hand": player.get("handCount"),
            "n_deck": player.get("deckCount"),
            # トラッシュは公開領域。カード種の列として持つ
            "discard": [e.get("id") for e in (player.get("discard") or []) if isinstance(e, dict)],
            "asleep": player.get("asleep"),
            "confused": player.get("confused"),
            "paralyzed": player.get("paralyzed"),
            "poisoned": player.get("poisoned"),
            "burned": player.get("burned"),
        }

    return {
        "turn": current.get("turn"),
        "turn_action_count": current.get("turnActionCount"),
        "first_player": current.get("firstPlayer"),
        "energy_attached": current.get("energyAttached"),
        "retreated": current.get("retreated"),
        "supporter_played": current.get("supporterPlayed"),
        "stadium_played": current.get("stadiumPlayed"),
        "stadium": [e.get("id") for e in (current.get("stadium") or []) if isinstance(e, dict)],
        "own": side(own),
        "opponent": side(opp),
    }
