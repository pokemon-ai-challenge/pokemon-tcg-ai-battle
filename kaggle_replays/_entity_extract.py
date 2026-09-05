"""Phase14: raw entity 情報の抽出(engine 非依存の純ロジック部分を含む)。

state166 は**カード identity を一切持たない**(監査済み)。ここでは probe / collision audit
のために、意思決定時にそのプレイヤーが実際に知り得る情報だけを raw のまま取り出す。

情報リーク禁止(§27): 相手の手札・山札の中身・未来の引き・determinization の真値は取らない。
相手について取るのは公開情報(場・トラッシュ・各種カウント)のみ。
"""
from __future__ import annotations

DISCARD_CAP = 60
HAND_CAP = 20
MAX_BENCH = 5

# summarize() が返す数値サマリ(action delta 用)の並び
SUMMARY_KEYS = (
    "self_active_hp", "self_active_dmg", "self_active_energy", "self_active_tools",
    "self_bench_count", "self_bench_hp_total", "self_bench_energy_total",
    "self_hand_count", "self_deck", "self_discard", "self_prize",
    "opp_active_hp", "opp_active_dmg", "opp_active_energy",
    "opp_bench_count", "opp_bench_hp_total", "opp_deck", "opp_discard", "opp_prize",
    "self_energy_board", "opp_energy_board", "turn",
)


def _pkmn(p):
    """1体分の raw entity。None なら None。"""
    if p is None:
        return None
    return {
        "id": int(getattr(p, "id", 0) or 0),
        "hp": int(getattr(p, "hp", 0) or 0),
        "max_hp": int(getattr(p, "maxHp", 0) or 0),
        "dmg": max(0, int(getattr(p, "maxHp", 0) or 0) - int(getattr(p, "hp", 0) or 0)),
        "new": bool(getattr(p, "appearThisTurn", False)),
        "en": [int(getattr(e, "value", e)) for e in (getattr(p, "energies", None) or [])],
        "tools": [int(getattr(c, "id", 0) or 0) for c in (getattr(p, "tools", None) or [])],
        "pre": [int(getattr(c, "id", 0) or 0) for c in (getattr(p, "preEvolution", None) or [])],
    }


def _ids(cards, cap):
    return [int(getattr(c, "id", 0) or 0) for c in (cards or [])[:cap] if c is not None]


def _side(ps, own: bool):
    act = (ps.active[0] if getattr(ps, "active", None) else None)
    out = {
        "active": _pkmn(act),
        "bench": [_pkmn(b) for b in (getattr(ps, "bench", None) or [])[:MAX_BENCH]],
        "discard": _ids(getattr(ps, "discard", None), DISCARD_CAP),
        "deck_count": int(getattr(ps, "deckCount", 0) or 0),
        "hand_count": int(getattr(ps, "handCount", 0) or 0),
        "prize_count": sum(1 for x in (getattr(ps, "prize", None) or []) if x is not None
                           or True),
        "status": [int(bool(getattr(ps, k, False))) for k in
                   ("poisoned", "burned", "asleep", "paralyzed", "confused")],
    }
    if own:
        # 手札の中身は自分だけが知る(相手は None が返る仕様)
        out["hand"] = _ids(getattr(ps, "hand", None), HAND_CAP)
    return out


def extract(state, me: int) -> dict:
    """意思決定時点の raw entity。相手の手札中身は**含めない**(§27)。"""
    mine, opp = state.players[me], state.players[1 - me]
    return {
        "turn": int(getattr(state, "turn", 0) or 0),
        "me": int(me),
        "self": _side(mine, own=True),
        "opp": _side(opp, own=False),
    }


def summarize(state, me: int) -> list[float]:
    """action delta 用の固定長数値サマリ(SUMMARY_KEYS の順)。"""
    e = extract(state, me)
    s, o = e["self"], e["opp"]
    sa, oa = s["active"], o["active"]

    def bsum(side, key):
        return float(sum((b or {}).get(key, 0) for b in side["bench"]))

    def esum(side):
        n = len((side["active"] or {}).get("en", []))
        return float(n + sum(len((b or {}).get("en", [])) for b in side["bench"]))

    return [
        float((sa or {}).get("hp", 0)), float((sa or {}).get("dmg", 0)),
        float(len((sa or {}).get("en", []))), float(len((sa or {}).get("tools", []))),
        float(len(s["bench"])), bsum(s, "hp"),
        float(sum(len((b or {}).get("en", [])) for b in s["bench"])),
        float(s["hand_count"]), float(s["deck_count"]), float(len(s["discard"])),
        float(s["prize_count"]),
        float((oa or {}).get("hp", 0)), float((oa or {}).get("dmg", 0)),
        float(len((oa or {}).get("en", []))),
        float(len(o["bench"])), bsum(o, "hp"),
        float(o["deck_count"]), float(len(o["discard"])), float(o["prize_count"]),
        esum(s), esum(o), float(e["turn"]),
    ]


# ---------------- probe 用のベクトル化(engine 非依存) ----------------

def card_bag(ids, size: int) -> dict:
    """カード ID -> 個数。identity と multiplicity を保つ(§12)。"""
    bag: dict[int, int] = {}
    for c in ids:
        if 0 < c < size:
            bag[c] = bag.get(c, 0) + 1
    return bag


def board_tokens(entity: dict, side: str, max_slots: int = 1 + MAX_BENCH) -> list[dict]:
    """Board entity を「1体=1token」に並べる(active を先頭、以降 bench 順)。"""
    sd = entity[side]
    toks = []
    seq = [sd["active"]] + list(sd["bench"])
    for slot, p in enumerate(seq[:max_slots]):
        if p is None:
            continue
        toks.append({
            "card_id": p["id"], "slot": slot, "is_active": 1 if slot == 0 else 0,
            "hp_ratio": (p["hp"] / p["max_hp"]) if p["max_hp"] else 0.0,
            "hp": p["hp"] / 340.0, "dmg": p["dmg"] / 340.0,
            "n_energy": len(p["en"]) / 5.0, "n_tools": len(p["tools"]),
            "stage": len(p["pre"]),          # 進化段数(0=たね)
            "new": 1.0 if p["new"] else 0.0,
            "tool_id": p["tools"][0] if p["tools"] else 0,
            "energy_types": p["en"],
        })
    return toks
