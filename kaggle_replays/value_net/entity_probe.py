"""Phase14 §10-§18: feature-family oracle probe のモデルと特徴量化。

**Transformer は作らない**(§2)。追加情報は単純な embedding + masked pooling だけで入れる。
狙いは「情報が足りないのか、モデル構造が足りないのか」を分離すること。

base(P0) = state166 + option65 + action card ID(= Q0-expanded と同形)
family を **一度に一つだけ** 足す(§11)。

  P1 hand      : 手札 card ID の bag(multiplicity 保持)
  P2 board     : 自分の active/bench を 1体=1token(card ID + HP/damage/energy/stage/tool)
  P3 opponent  : 相手の公開 board + 公開トラッシュ bag
  P4 deck      : 自分のトラッシュ bag + 「まだ見ていない自デッキ構成」(自デッキは既知)
  P5 history   : 直近 N 手(action type / card ID / sel type)
  P6 delta     : 候補ごとの 1step 後サマリ - 現在サマリ(**候補単位**の特徴)

情報リーク禁止(§27): 相手手札・山札順・未来・determinization 真値は使わない。
P4 の「未見の自デッキ」は自分のデッキリストと公開ゾーンだけから作る(実プレイで計算可能)。
"""
from __future__ import annotations

import torch
import torch.nn as nn

VOCAB = 2048
CARD_D = 16
N_SUMMARY = 22
HIST_N = 8

FAMILIES = ("hand", "board", "opp", "deck", "history", "delta")


# ---------------- 特徴量化(numpy/list ベース、engine 非依存) ----------------

def hand_bag(entity, cap=20):
    return [c for c in entity["self"].get("hand", [])[:cap] if 0 < c < VOCAB]


def board_tokens(entity, side="self", max_slots=6):
    sd = entity[side]
    seq = [sd["active"]] + list(sd["bench"])
    out = []
    for slot, p in enumerate(seq[:max_slots]):
        if p is None:
            continue
        mh = p["max_hp"] or 1
        out.append(([p["id"] if 0 < p["id"] < VOCAB else 0,
                     (p["tools"][0] if p["tools"] and 0 < p["tools"][0] < VOCAB else 0)],
                    [1.0 if slot == 0 else 0.0, slot / 5.0, p["hp"] / mh,
                     p["hp"] / 340.0, p["dmg"] / 340.0, len(p["en"]) / 5.0,
                     len(p["tools"]), len(p["pre"]) / 2.0,
                     1.0 if p["new"] else 0.0,
                     float(len(set(p["en"]))) / 3.0]))
    return out


def discard_bag(entity, side="self", cap=60):
    return [c for c in entity[side]["discard"][:cap] if 0 < c < VOCAB]


def unseen_deck_bag(entity, deck_ids):
    """自デッキ構成 - 既に見えた自分のカード = まだ山札/サイドにある構成(自デッキは既知)。"""
    from collections import Counter
    seen = Counter()
    sd = entity["self"]
    seen.update(sd.get("hand", []))
    seen.update(sd["discard"])
    for p in [sd["active"]] + list(sd["bench"]):
        if p:
            seen[p["id"]] += 1
            seen.update(p["tools"])
            seen.update(p["pre"])
    rest = Counter(deck_ids)
    rest.subtract(seen)
    out = []
    for c, n in rest.items():
        if n > 0 and 0 < c < VOCAB:
            out.extend([c] * min(n, 4))
    return out


def history_tokens(hist, n=HIST_N):
    h = list(hist)[-n:]
    return [([x["card_id"] if 0 < x["card_id"] < VOCAB else 0],
             [x["type"] / 20.0, x["sel_type"] / 12.0, (i + 1) / n])
            for i, x in enumerate(h)]


def delta_vec(before, after):
    if after is None:
        return [0.0] * (N_SUMMARY + 1)
    return [(a - b) / 20.0 for a, b in zip(after, before)] + [1.0]


# ---------------- モデル ----------------

class BagEncoder(nn.Module):
    """card ID の bag -> 固定長。sum と mean を併用して個数と構成の両方を残す。"""

    def __init__(self, out_dim=32):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, CARD_D, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(CARD_D * 2, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, ids):                        # ids [B, L]
        m = (ids > 0).float().unsqueeze(-1)
        e = self.emb(ids) * m
        s = e.sum(1)
        mean = s / m.sum(1).clamp(min=1.0)
        return self.mlp(torch.cat([s / 8.0, mean], dim=-1))


class TokenEncoder(nn.Module):
    """(card ids, numeric) の token 列 -> masked mean(**Attention なし**, §13)。"""

    def __init__(self, n_ids, n_num, out_dim=32):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, CARD_D, padding_idx=0)
        self.n_ids = n_ids
        self.tok = nn.Sequential(nn.Linear(CARD_D * n_ids + n_num, out_dim), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, ids, num, mask):             # ids [B,T,n_ids] num [B,T,n_num]
        e = self.emb(ids).flatten(2)
        h = self.tok(torch.cat([e, num], dim=-1)) * mask.unsqueeze(-1)
        s = h.sum(1)
        mean = s / mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.out(torch.cat([s / 4.0, mean], dim=-1))


class ProbeNet(nn.Module):
    """Q0-expanded と同形の base に、選んだ family だけを足した診断用モデル。"""

    def __init__(self, state_dim, option_dim, families=(), hidden=64):
        super().__init__()
        self.families = tuple(families)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.card_emb = nn.Embedding(VOCAB, CARD_D, padding_idx=0)

        ctx = hidden
        self.enc = nn.ModuleDict()
        if "hand" in self.families:
            self.enc["hand"] = BagEncoder(); ctx += 32
        if "board" in self.families:
            self.enc["board"] = TokenEncoder(2, 10); ctx += 32
        if "opp" in self.families:
            self.enc["opp"] = TokenEncoder(2, 10); ctx += 32
            self.enc["opp_disc"] = BagEncoder(); ctx += 32
        if "deck" in self.families:
            self.enc["disc"] = BagEncoder(); ctx += 32
            self.enc["unseen"] = BagEncoder(); ctx += 32
        if "history" in self.families:
            self.enc["history"] = TokenEncoder(1, 3); ctx += 32

        self._use_delta = ("delta" in self.families) or ("delta_shuf" in self.families)
        act_in = option_dim + CARD_D + (N_SUMMARY + 1 if self._use_delta else 0)
        self.act_mlp = nn.Sequential(nn.Linear(act_in, hidden), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(ctx + hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.q_head = nn.Linear(hidden, 1)

    def forward(self, batch):
        parts = [self.state_mlp(batch["state"])]
        if "hand" in self.families:
            parts.append(self.enc["hand"](batch["hand"]))
        if "board" in self.families:
            parts.append(self.enc["board"](batch["board_ids"], batch["board_num"],
                                           batch["board_mask"]))
        if "opp" in self.families:
            parts.append(self.enc["opp"](batch["opp_ids"], batch["opp_num"],
                                         batch["opp_mask"]))
            parts.append(self.enc["opp_disc"](batch["opp_disc"]))
        if "deck" in self.families:
            parts.append(self.enc["disc"](batch["disc"]))
            parts.append(self.enc["unseen"](batch["unseen"]))
        if "history" in self.families:
            parts.append(self.enc["history"](batch["hist_ids"], batch["hist_num"],
                                             batch["hist_mask"]))
        ctx = torch.cat(parts, dim=-1)

        B, C, _ = batch["opt"].shape
        a = [batch["opt"], self.card_emb(batch["card"])]
        if self._use_delta:
            a.append(batch["delta"])
        act = self.act_mlp(torch.cat(a, dim=-1))
        h = self.fuse(torch.cat([ctx.unsqueeze(1).expand(B, C, -1), act], dim=-1))
        q = self.q_head(h).squeeze(-1)
        return q.masked_fill(~batch["mask"], -1e9)
