"""Phase17 §8-§15: raw entity -> token 列 + relation 行列(engine 非依存の純ロジック)。

**情報リーク禁止(§7)**: 相手手札・山札順・未来・determinization 真値は token 化しない。
相手側は公開情報(場・公開トラッシュ・各種カウント)のみ。自分の山札残り構成は v0 では除外。

token 種別(TYPE_*)と、attention に足す relation bias の型(REL_*)を固定する。
schema は test 結果を見て変更しない(§49)。
"""
from __future__ import annotations

from collections import Counter

VOCAB = 2048
MAX_HAND_TOK = 12
MAX_DISCARD_TOK = 10
MAX_POKEMON = 12          # 自分 active+bench5 / 相手 active+bench5

# token type
TYPE_STATE, TYPE_POKEMON, TYPE_HAND, TYPE_DISCARD, TYPE_STADIUM = 0, 1, 2, 3, 4
N_TYPE = 5

# owner / zone / slot
OWN_NONE, OWN_SELF, OWN_OPP = 0, 1, 2
ZONE_NONE, ZONE_ACTIVE, ZONE_BENCH, ZONE_HAND, ZONE_DISCARD, ZONE_STADIUM = 0, 1, 2, 3, 4, 5
N_OWNER, N_ZONE, N_SLOT = 3, 6, 8

# relation type(小さな離散集合。巨大な edge MLP は使わない §16)
REL_NONE = 0          # 無関係
REL_SELF = 1          # 自分自身
REL_SAME_OWNER = 2    # 同じプレイヤーのもの
REL_OPPOSING = 3      # 相手同士
REL_SAME_ZONE = 4     # 同じゾーン
REL_ACTIVE_BENCH = 5  # 同一プレイヤーの active <-> bench
REL_STATE = 6         # STATE token と何か
REL_EVOLUTION = 7     # 進化ライン(card id が pre_evolution に含まれる)
REL_HAND_BOARD = 8    # 自分の手札 <-> 自分の場(使用可能性)
N_REL = 9

N_NUM = 12            # token ごとの数値属性数


def _pk_num(p, is_active, slot):
    mh = p["max_hp"] or 1
    en = p["en"] or []
    c = Counter(en)
    return [
        1.0 if is_active else 0.0,
        slot / 5.0,
        p["hp"] / mh,
        p["hp"] / 340.0,
        p["dmg"] / 340.0,
        len(en) / 5.0,
        len(set(en)) / 3.0,
        (max(c.values()) / 5.0) if c else 0.0,     # 最大同型エネ数(技コスト充足の目安)
        float(len(p["tools"])),
        len(p["pre"]) / 2.0,                        # 進化段数
        1.0 if p["new"] else 0.0,
        1.0,                                        # present flag
    ]


def _zero_num():
    return [0.0] * N_NUM


def tokenize(entity: dict, max_hand=MAX_HAND_TOK, max_discard=MAX_DISCARD_TOK) -> dict:
    """raw entity -> token dict。

    返り値:
      card_id [T], type [T], owner [T], zone [T], slot [T], num [T, N_NUM],
      pre_ids [T, 2](進化 relation 用), key [T](entity 識別子)
    """
    cid, typ, own, zon, slt, num, pre, key = [], [], [], [], [], [], [], []

    def add(c, t, o, z, s, n, p=(0, 0), k=None):
        cid.append(c if 0 < c < VOCAB else 0)
        typ.append(t); own.append(o); zon.append(z); slt.append(min(s, N_SLOT - 1))
        num.append(n); pre.append([p[0] if len(p) > 0 else 0, p[1] if len(p) > 1 else 0])
        key.append(k)

    # --- STATE / global token(§13) ---
    s_, o_ = entity["self"], entity["opp"]
    g = _zero_num()
    g[0] = entity.get("turn", 0) / 30.0
    g[1] = s_["prize_count"] / 6.0
    g[2] = o_["prize_count"] / 6.0
    g[3] = s_["deck_count"] / 60.0
    g[4] = o_["deck_count"] / 60.0
    g[5] = s_["hand_count"] / 10.0
    g[6] = o_["hand_count"] / 10.0
    g[7] = len(s_["discard"]) / 60.0
    g[8] = len(o_["discard"]) / 60.0
    g[9] = float(sum(s_["status"]))
    g[10] = float(sum(o_["status"]))
    g[11] = 1.0
    add(0, TYPE_STATE, OWN_NONE, ZONE_NONE, 0, g, k="STATE")

    # --- Pokémon token(§8/§9/§10) ---
    for owner, side in ((OWN_SELF, s_), (OWN_OPP, o_)):
        seq = [side["active"]] + list(side["bench"])
        for slot, p in enumerate(seq[:6]):
            if p is None:
                continue
            add(p["id"], TYPE_POKEMON, owner,
                ZONE_ACTIVE if slot == 0 else ZONE_BENCH, slot,
                _pk_num(p, slot == 0, slot),
                p["pre"][:2], k=("pk", owner, slot))

    # --- Hand token(自分のみ・§11 compressed: card_id + count)---
    hc = Counter(s_.get("hand", []))
    for c, n in sorted(hc.items(), key=lambda x: -x[1])[:max_hand]:
        v = _zero_num()
        v[0] = n / 4.0
        v[11] = 1.0
        add(c, TYPE_HAND, OWN_SELF, ZONE_HAND, 0, v, k=("hand", c))

    # --- Discard token(§12 compressed)---
    for owner, side in ((OWN_SELF, s_), (OWN_OPP, o_)):
        dc = Counter(side["discard"])
        for c, n in sorted(dc.items(), key=lambda x: -x[1])[:max_discard]:
            v = _zero_num()
            v[0] = n / 4.0
            v[11] = 1.0
            add(c, TYPE_DISCARD, owner, ZONE_DISCARD, 0, v, k=("disc", owner, c))

    return {"card_id": cid, "type": typ, "owner": own, "zone": zon, "slot": slt,
            "num": num, "pre": pre, "key": key}


def relation_matrix(tok: dict) -> list[list[int]]:
    """token 対 -> relation type(§15/§16)。対称でよいものは対称に置く。"""
    T = len(tok["card_id"])
    typ, own, zon, slt, cid, pre = (tok["type"], tok["owner"], tok["zone"],
                                    tok["slot"], tok["card_id"], tok["pre"])
    R = [[REL_NONE] * T for _ in range(T)]
    for i in range(T):
        for j in range(T):
            if i == j:
                R[i][j] = REL_SELF
                continue
            if typ[i] == TYPE_STATE or typ[j] == TYPE_STATE:
                R[i][j] = REL_STATE
                continue
            # 進化ライン: 一方の card_id が他方の pre-evolution に含まれる
            if (cid[i] and cid[i] in pre[j]) or (cid[j] and cid[j] in pre[i]):
                R[i][j] = REL_EVOLUTION
                continue
            if own[i] == own[j]:
                if (typ[i] == TYPE_HAND and typ[j] == TYPE_POKEMON) or \
                   (typ[j] == TYPE_HAND and typ[i] == TYPE_POKEMON):
                    R[i][j] = REL_HAND_BOARD
                elif typ[i] == TYPE_POKEMON and typ[j] == TYPE_POKEMON and \
                        zon[i] != zon[j]:
                    R[i][j] = REL_ACTIVE_BENCH
                elif zon[i] == zon[j]:
                    R[i][j] = REL_SAME_ZONE
                else:
                    R[i][j] = REL_SAME_OWNER
            else:
                R[i][j] = REL_OPPOSING
    return R


def action_features(cand: dict, entity: dict) -> dict:
    """Action token(§14)。source/target を entity slot へ紐づける。"""
    return {"card_id": cand.get("action_card_id", -1),
            "action_type": int(cand.get("option_type", 0))}
