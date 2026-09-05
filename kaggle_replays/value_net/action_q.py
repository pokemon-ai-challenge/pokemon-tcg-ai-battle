"""Phase7 Track A: Action-Q モデル(Q0 / Q1)。

**将来の Relational / Candidate Transformer へ置き換えられる interface** を保つため、
巨大 MLP を1本書かず、以下を独立モジュールに分ける(§3):

    StateEncoder        : state166 -> 64
    CardSetEncoder      : {hand, discard, opp_public} -> zone vectors   (DeepSets, 後で Set TF へ)
    ActionEncoder       : option65 + action card id + option type -> action vector
    StateActionFusion   : (state, cards, action) -> 融合ベクトル        (後で Cross Attention へ)
    QHead / PolicyAuxHead

`encode_candidates()` は decision group 内の全候補を **まとめて**返すので、
将来 Candidate Set Transformer を挟むときはこのテンソルに attention を掛けるだけでよい。
"""
from __future__ import annotations

import torch
import torch.nn as nn

PAD = 0
N_CARD = 1270        # card_id を +1 シフトして 0 を padding にする
N_OPTTYPE = 64       # OptionType の余裕を持った上限


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, out), nn.ReLU())
        self.out_dim = out

    def forward(self, x):
        return self.net(x)


class CardSetEncoder(nn.Module):
    """zone ごとの多重集合 -> ベクトル(順序不変。sum と mean を連結)。

    将来 Set / Relational Transformer に差し替える箇所。interface は
    `forward(list_of_zone_id_tensors) -> [B, out_dim]` を維持する。
    """

    def __init__(self, n_zones: int = 3, emb_dim: int = 32, hidden: int = 48,
                 zone_out: int = 64):
        super().__init__()
        self.emb = nn.Embedding(N_CARD, emb_dim, padding_idx=PAD)
        self.zone_emb = nn.Embedding(n_zones, 8)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + 8, hidden), nn.ReLU(), nn.Linear(hidden, zone_out), nn.ReLU())
        self.n_zones = n_zones
        self.out_dim = n_zones * zone_out * 2

    def forward(self, zone_ids: list[torch.Tensor]) -> torch.Tensor:
        outs = []
        for z, ids in enumerate(zone_ids):
            mask = (ids != PAD).float().unsqueeze(-1)
            e = self.emb(ids)
            zz = self.zone_emb(torch.full_like(ids, z))
            h = self.mlp(torch.cat([e, zz], dim=-1)) * mask
            s = h.sum(dim=1)
            outs.append(torch.cat([s, s / mask.sum(dim=1).clamp(min=1.0)], dim=-1))
        return torch.cat(outs, dim=-1)


class ActionEncoder(nn.Module):
    """候補行動 -> ベクトル。option65 + 行動カードID + option type。"""

    def __init__(self, option_dim: int, emb_dim: int = 32, type_dim: int = 16,
                 out: int = 64):
        super().__init__()
        self.card_emb = nn.Embedding(N_CARD, emb_dim, padding_idx=PAD)
        self.type_emb = nn.Embedding(N_OPTTYPE, type_dim)
        self.net = nn.Sequential(
            nn.Linear(option_dim + emb_dim + type_dim, 128), nn.ReLU(),
            nn.Linear(128, out), nn.ReLU())
        self.out_dim = out

    def forward(self, opt_feat, card_id, opt_type):
        e = self.card_emb(card_id)
        t = self.type_emb(opt_type)
        return self.net(torch.cat([opt_feat, e, t], dim=-1))


class StateActionFusion(nn.Module):
    """(state, cards, action) -> 融合ベクトル。将来 Cross Attention へ差し替える箇所。"""

    def __init__(self, in_dim: int, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, out), nn.ReLU())
        self.out_dim = out

    def forward(self, x):
        return self.net(x)


class ActionQNet(nn.Module):
    """Q0(use_cards=False) / Q1(use_cards=True)。

    候補は [B, C, ...] でまとめて渡し、[B, C] の Q を返す(batch 評価前提)。
    """

    def __init__(self, state_dim: int, option_dim: int, use_cards: bool = True,
                 use_action_card: bool = True):
        super().__init__()
        self.use_cards = use_cards
        self.use_action_card = use_action_card
        self.state_enc = StateEncoder(state_dim)
        self.card_enc = CardSetEncoder() if use_cards else None
        self.act_enc = ActionEncoder(option_dim)
        fuse_in = self.state_enc.out_dim + self.act_enc.out_dim
        if use_cards:
            fuse_in += self.card_enc.out_dim
        self.fusion = StateActionFusion(fuse_in)
        self.q_head = nn.Linear(self.fusion.out_dim, 1)
        self.policy_head = nn.Linear(self.fusion.out_dim, 1)

    def encode_candidates(self, state, zones, opt_feat, card_id, opt_type):
        """[B, C, D] の候補表現を返す。Candidate Transformer を挟むならここ。"""
        B, C, _ = opt_feat.shape
        s = self.state_enc(state)                                  # [B, S]
        parts = [s]
        if self.use_cards:
            parts.append(self.card_enc(zones))                     # [B, Z]
        ctx = torch.cat(parts, dim=-1).unsqueeze(1).expand(B, C, -1)
        if not self.use_action_card:
            card_id = torch.zeros_like(card_id)
        a = self.act_enc(opt_feat, card_id, opt_type)               # [B, C, A]
        return self.fusion(torch.cat([ctx, a], dim=-1))             # [B, C, F]

    def forward(self, state, zones, opt_feat, card_id, opt_type, mask=None):
        h = self.encode_candidates(state, zones, opt_feat, card_id, opt_type)
        q = self.q_head(h).squeeze(-1)
        p = self.policy_head(h).squeeze(-1)
        if mask is not None:
            q = q.masked_fill(~mask, -1e9)
            p = p.masked_fill(~mask, -1e9)
        return q, p


class CardActionCrossAttention(nn.Module):
    """候補行動を Query、手札カード token を Key/Value とする Cross Attention。

    単純 pooling では「どの手札カードがこの候補行動と関係するか」が失われる
    (Phase7: 手札 pooling の寄与は +0.008 しか無かった)。ここでは候補ごとに
    **異なる Attention 重み**で手札を参照させる。

    候補間 Self-Attention は **入れない**(Candidate Transformer は次フェーズ)。
    出力は [B, C, D] のままなので、将来その次元に候補間 attention を足せる。
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, action_q: torch.Tensor, card_tok: torch.Tensor,
                card_mask: torch.Tensor) -> torch.Tensor:
        """action_q [B,C,D] / card_tok [B,N,D] / card_mask [B,N] (True=有効)。"""
        # 全て padding の行は attention が NaN になるので、ダミーで1枚だけ有効にする
        empty = ~card_mask.any(dim=1)
        if empty.any():
            card_mask = card_mask.clone()
            card_mask[empty, 0] = True
        out, w = self.attn(action_q, card_tok, card_tok,
                           key_padding_mask=~card_mask, need_weights=True,
                           average_attn_weights=True)
        if empty.any():                      # 空手札の寄与は0にする
            out = out.masked_fill(empty.view(-1, 1, 1), 0.0)
        self.last_weights = w                # [B,C,N] 診断用
        return self.norm(action_q + out)


class HandTokenizer(nn.Module):
    """手札を **1枚1token** にする(順序不変。padding は 0)。"""

    def __init__(self, emb_dim: int = 32, d_model: int = 64):
        super().__init__()
        self.emb = nn.Embedding(N_CARD, emb_dim, padding_idx=PAD)
        self.proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.ReLU())
        self.d_model = d_model

    def forward(self, ids: torch.Tensor):
        return self.proj(self.emb(ids)), (ids != PAD)


class CrossActionQNet(nn.Module):
    """QX-Cross: state + action + action-conditioned hand。

    `capacity_only=True` にすると Attention を使わず、同程度のパラメータを持つ
    MLP で手札 pooled ベクトルを処理する(QC-Capacity: 容量対照)。
    `mean_attention=True` は Attention の代わりに手札 token の単純平均を足す。
    """

    def __init__(self, state_dim: int, option_dim: int, d_model: int = 64,
                 n_heads: int = 4, use_action_card: bool = True,
                 capacity_only: bool = False, mean_attention: bool = False):
        super().__init__()
        self.use_action_card = use_action_card
        self.capacity_only = capacity_only
        self.mean_attention = mean_attention
        self.state_enc = StateEncoder(state_dim)
        self.act_enc = ActionEncoder(option_dim, out=d_model)
        self.hand_tok = HandTokenizer(d_model=d_model)
        if capacity_only:
            # Attention の代わりに同規模の MLP(pooled hand を処理)
            self.cap_mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 2), nn.ReLU(),
                nn.Linear(d_model * 2, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU())
        elif not mean_attention:
            self.cross = CardActionCrossAttention(d_model, n_heads)
        fuse_in = self.state_enc.out_dim + d_model + d_model
        self.fusion = StateActionFusion(fuse_in)
        self.q_head = nn.Linear(self.fusion.out_dim, 1)
        self.policy_head = nn.Linear(self.fusion.out_dim, 1)

    def encode_candidates(self, state, hand_ids, opt_feat, card_id, opt_type):
        B, C, _ = opt_feat.shape
        s = self.state_enc(state).unsqueeze(1).expand(B, C, -1)
        if not self.use_action_card:
            card_id = torch.zeros_like(card_id)
        a = self.act_enc(opt_feat, card_id, opt_type)              # [B,C,D]
        tok, mask = self.hand_tok(hand_ids)                        # [B,N,D]
        m = mask.float().unsqueeze(-1)
        pooled = (tok * m).sum(1) / m.sum(1).clamp(min=1.0)        # [B,D]
        if self.capacity_only:
            h = self.cap_mlp(pooled).unsqueeze(1).expand(B, C, -1)
        elif self.mean_attention:
            h = pooled.unsqueeze(1).expand(B, C, -1)
        else:
            h = self.cross(a, tok, mask)                            # [B,C,D]
        return self.fusion(torch.cat([s, a, h], dim=-1))

    def forward(self, state, hand_ids, opt_feat, card_id, opt_type, mask=None):
        h = self.encode_candidates(state, hand_ids, opt_feat, card_id, opt_type)
        q = self.q_head(h).squeeze(-1)
        p = self.policy_head(h).squeeze(-1)
        if mask is not None:
            q = q.masked_fill(~mask, -1e9)
            p = p.masked_fill(~mask, -1e9)
        return q, p


class CandidateSetQNet(nn.Module):
    """Phase10: 同一 decision group 内の **候補同士**を見て採点するモデル群。

    Q0 の CandidateEncoder(`ActionQNet.encode_candidates`)をそのまま再利用し、
    得られた [B,C,D] に対して mode 別の後処理だけを変える。
    **入力情報は Q0 と完全に同じ**(手札 Cross Attention や state token は足さない)ので、
    差分は「候補を独立採点するか / 候補同士を見るか」だけになる。

      mode="none"     : Q0-expanded 相当(後処理なし)
      mode="capacity" : 候補ごとの FFN のみ(候補間通信なし。容量対照 QC)
      mode="mean"     : 候補 embedding の masked mean を各候補へ broadcast(QG)
      mode="attn"     : 候補間 Self-Attention 1層(QT。**本命**)
      mode="identity" : Attention を self のみに制限(ブロック追加の効果だけを分離)

    positional encoding / CLS / state token は入れない(permutation equivariant を保つ)。
    """

    def __init__(self, state_dim: int, option_dim: int, mode: str = "attn",
                 n_heads: int = 4, use_action_card: bool = True):
        super().__init__()
        self.mode = mode
        self.base = ActionQNet(state_dim, option_dim, use_cards=False,
                               use_action_card=use_action_card)
        d = self.base.fusion.out_dim                      # 128
        self.d = d
        if mode in ("attn", "identity"):
            self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.n1 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d))
            self.n2 = nn.LayerNorm(d)
        elif mode == "capacity":
            # 候補間通信なしで QT と同程度のパラメータにするための候補単位 FFN
            self.cap = nn.Sequential(
                nn.Linear(d, d * 4), nn.ReLU(), nn.Linear(d * 4, d), nn.ReLU())
            self.n2 = nn.LayerNorm(d)
        elif mode == "mean":
            self.mix = nn.Sequential(nn.Linear(d * 2, d), nn.ReLU())
            self.n2 = nn.LayerNorm(d)
        self.q_head = nn.Linear(d, 1)
        self.policy_head = nn.Linear(d, 1)
        self.last_attn = None

    def forward(self, state, zones, opt_feat, card_id, opt_type, mask=None):
        x = self.base.encode_candidates(state, zones, opt_feat, card_id, opt_type)
        B, C, _ = x.shape
        if mask is None:
            mask = torch.ones(B, C, dtype=torch.bool, device=x.device)
        m = mask.unsqueeze(-1).float()
        x = x * m                                          # padding を持ち込まない

        if self.mode in ("attn", "identity"):
            attn_mask = None
            if self.mode == "identity":
                # 自分自身しか見られない = 候補間通信を殺す(ブロック効果のみ分離)
                attn_mask = ~torch.eye(C, dtype=torch.bool, device=x.device)
            # 全 padding の行が無いように保護(NaN 回避)
            km = ~mask
            empty = km.all(dim=1)
            if empty.any():
                km = km.clone()
                km[empty, 0] = False
            if self.mode == "identity":
                # self のみ許可する attn_mask と key_padding_mask を併用すると、
                # padding 候補の行が「唯一許可された self も padding」で全マスクになり
                # NaN が出る。identity は self を見るだけなので padding mask は不要
                # (padding 候補の出力は最後の masked_fill で捨てられる)。
                a, w = self.attn(x, x, x, attn_mask=attn_mask,
                                 need_weights=True, average_attn_weights=True)
            else:
                a, w = self.attn(x, x, x, key_padding_mask=km, attn_mask=attn_mask,
                                 need_weights=True, average_attn_weights=True)
            self.last_attn = w
            x = self.n1(x + a * m)
            x = self.n2(x + self.ff(x) * m)
        elif self.mode == "capacity":
            x = self.n2(x + self.cap(x) * m)
        elif self.mode == "mean":
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
            x = self.n2(self.mix(torch.cat([x, pooled.unsqueeze(1).expand(B, C, -1)],
                                           dim=-1)))
        q = self.q_head(x).squeeze(-1)
        p = self.policy_head(x).squeeze(-1)
        q = q.masked_fill(~mask, -1e9)
        p = p.masked_fill(~mask, -1e9)
        return q, p
