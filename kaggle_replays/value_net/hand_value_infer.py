"""学習した V0p / V1 を cg の State から評価するための推論ラッパ(研究用・production 非接続)。

- 特徴量は production と同じ `encoder.encode_state_from_state`(166次元)を使う。
- カード集合は **観測可能な範囲のみ**: 自分の手札・自分のトラッシュ・相手の公開カード
  (場のポケモン + トラッシュ)。相手手札・山札は絶対に読まない。
- 視点: モデルは「観測者(state.yourIndex)が勝つ確率」を出す。``me`` が異なる場合は 1-p にする
  (この反転を間違えると、探索の葉で相手の勝率を最大化することになる)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SUB = _HERE.parents[1] / "sample_submission"
if str(_SUB) not in sys.path:
    sys.path.insert(0, str(_SUB))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ptcg_ai.learning import encoder  # noqa: E402
from train_hand_value import ValueNet  # noqa: E402

MAX_HAND, MAX_DISC, MAX_OPP = 20, 30, 30


def _ids(cards, cap):
    out = []
    for c in cards or []:
        cid = getattr(c, "id", None)
        if cid is None and isinstance(c, dict):
            cid = c.get("id")
        if cid is not None:
            out.append(int(cid))
        if len(out) >= cap:
            break
    return out


def visible_card_sets(state, me: int) -> tuple[list[int], list[int], list[int]]:
    """(自分手札, 自分トラッシュ, 相手公開) を返す。隠れ情報は含めない。"""
    p = state.players[me]
    opp = state.players[1 - me]
    hand = _ids(getattr(p, "hand", None), MAX_HAND)
    disc = _ids(getattr(p, "discard", None), MAX_DISC)
    vis = _ids(getattr(opp, "discard", None), MAX_OPP)
    for slot in (list(getattr(opp, "active", None) or [])
                 + list(getattr(opp, "bench", None) or [])):
        if slot is not None and len(vis) < MAX_OPP:
            cid = getattr(slot, "id", None)
            if cid is not None:
                vis.append(int(cid))
    return hand, disc, vis


class HandValue:
    def __init__(self, ckpt_path: str | Path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.use_cards = bool(ck["use_cards"])
        self.mean = np.asarray(ck["mean"], dtype=np.float32)
        self.std = np.asarray(ck["std"], dtype=np.float32)
        self.model = ValueNet(len(self.mean), self.use_cards)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    def _pad(self, ids, cap):
        a = np.zeros((1, max(1, min(len(ids), cap))), dtype=np.int64)
        for i, v in enumerate(ids[:cap]):
            a[0, i] = v + 1                     # 0 = padding
        return torch.from_numpy(a)

    @torch.no_grad()
    def predict_from_state(self, state, me: int,
                           override_hand: list[int] | None = None) -> float:
        """``me`` 視点の勝率。``override_hand`` を渡すと手札だけ差し替えて評価する
        (反実仮想テスト用。盤面特徴 166 は元のまま = 手札IDだけの効果を見る)。"""
        if state is None:
            return 0.5
        feats = np.asarray(encoder.encode_state_from_state(state), dtype=np.float32)
        x = torch.from_numpy(((feats - self.mean) / self.std)[None, :])
        kw = {}
        if self.use_cards:
            observer = state.yourIndex
            hand, disc, vis = visible_card_sets(state, observer)
            if override_hand is not None:
                hand = list(override_hand)[:MAX_HAND]
            kw = {"hand": self._pad(hand, MAX_HAND), "disc": self._pad(disc, MAX_DISC),
                  "opp": self._pad(vis, MAX_OPP)}
        p = float(torch.sigmoid(self.model(x, **kw)))
        return p if state.yourIndex == me else 1.0 - p
