"""Phase17 §16-§22: relation-bias Transformer による before/after 共有エンコーダ。

  z(s)  = Encoder(tokens(before))     ← STATE token 出力(§18)
  z(s') = Encoder(tokens(after_a))    ← **同一 weight**(§17)
  Q(s,a)= MLP([z(s), z(s'), z(s')-z(s), z(a)])   (§19)

規模は小さく固定(§20): layers=2, d_model=64, heads=4, ffn=128。

対照:
  after_shuffle : group 内で after を1つずらす(容量完全一致・§23)
  no_relation   : relation bias を無効化(owner/zone/slot embedding は残す・§24)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import entity_tokens as ET

D_MODEL, N_HEAD, N_LAYER, D_FF = 64, 4, 2, 128


class RelAttention(nn.Module):
    """relation bias 付き multi-head attention(§16)。"""

    def __init__(self, d=D_MODEL, h=N_HEAD, use_relation=True):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.use_relation = use_relation
        self.rel_bias = nn.Embedding(ET.N_REL, h) if use_relation else None

    def forward(self, x, mask, rel):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)      # [B,h,T,T]
        if self.use_relation:
            att = att + self.rel_bias(rel).permute(0, 3, 1, 2)     # [B,h,T,T]
        att = att.masked_fill(~mask[:, None, None, :], -1e9)
        att = att.softmax(-1)
        self._last_att = att.detach()
        out = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, use_relation=True, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(D_MODEL)
        self.att = RelAttention(use_relation=use_relation)
        self.n2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.ReLU(),
                                nn.Linear(D_FF, D_MODEL))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, rel):
        x = x + self.drop(self.att(self.n1(x), mask, rel))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class EntityEncoder(nn.Module):
    """before / after で**共有**するエンコーダ(§17)。"""

    def __init__(self, use_relation=True, dropout=0.0):
        super().__init__()
        self.card = nn.Embedding(ET.VOCAB, D_MODEL, padding_idx=0)
        self.typ = nn.Embedding(ET.N_TYPE, D_MODEL)
        self.own = nn.Embedding(ET.N_OWNER, D_MODEL)
        self.zon = nn.Embedding(ET.N_ZONE, D_MODEL)
        self.slt = nn.Embedding(ET.N_SLOT, D_MODEL)
        self.num = nn.Linear(ET.N_NUM, D_MODEL)
        self.blocks = nn.ModuleList([Block(use_relation, dropout) for _ in range(N_LAYER)])
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, t):
        x = (self.card(t["card_id"]) + self.typ(t["type"]) + self.own(t["owner"])
             + self.zon(t["zone"]) + self.slt(t["slot"]) + self.num(t["num"]))
        x = x * t["mask"].unsqueeze(-1)
        for b in self.blocks:
            x = b(x, t["mask"], t["rel"])
        x = self.norm(x)
        return x[:, 0]                     # STATE token(index 0)= z(s)  §18

    def attention_entropy(self):
        ents = []
        for b in self.blocks:
            a = getattr(b.att, "_last_att", None)
            if a is None:
                continue
            p = a.clamp_min(1e-9)
            e = -(p * p.log()).sum(-1)                      # [B,h,T]
            ents.append((e / math.log(max(2, a.shape[-1]))).mean().item())
        return sum(ents) / len(ents) if ents else None


class TransitionQNet(nn.Module):
    """mode: 'current'(T2) / 'before_after'(T3) / 'transition'(T4)。"""

    def __init__(self, option_dim, mode="transition", use_relation=True,
                 dropout=0.0, hidden=64):
        super().__init__()
        self.mode = mode
        self.enc = EntityEncoder(use_relation=use_relation, dropout=dropout)
        self.act = nn.Sequential(nn.Linear(option_dim + D_MODEL, hidden), nn.ReLU())
        self.act_card = nn.Embedding(ET.VOCAB, D_MODEL, padding_idx=0)
        n_state = {"current": 1, "before_after": 2, "transition": 3}[mode]
        self.head = nn.Sequential(
            nn.Linear(D_MODEL * n_state + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, b):
        B, C = b["opt"].shape[0], b["opt"].shape[1]
        zb = self.enc(b["before"])                                  # [B, D]
        parts_state = [zb.unsqueeze(1).expand(B, C, -1)]
        if self.mode != "current":
            flat = {k: v.reshape(B * C, *v.shape[2:]) for k, v in b["after"].items()}
            za = self.enc(flat).view(B, C, -1)                       # [B, C, D]
            parts_state.append(za)
            if self.mode == "transition":
                parts_state.append(za - parts_state[0])              # latent delta §19
        a = self.act(torch.cat([b["opt"], self.act_card(b["card"])], dim=-1))
        h = self.head(torch.cat(parts_state + [a], dim=-1)).squeeze(-1)
        return h.masked_fill(~b["mask"], -1e9)
