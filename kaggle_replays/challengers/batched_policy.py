"""ISMCTS v2.11 — Batched(numpy)PolicyModel scorer(意味不変・高速化のみ)。

PolicyModel.score_options_from_state は option ごとに pure-Python 三重ループ matmul を回す(NN が call の ~96%)。
本モジュールは **同一 weights・同一 encoder・同一 standardization・同一 embedding** を使い、legal options を
[N, 239] 行列にまとめて **numpy の batched matmul** で score を計算する。**行動(argmax)は不変**、計算だけ速い。

依存は numpy(既存)のみ。production の PolicyModel は変更しない(研究 challenger 経路でのみ使用)。
"""
from __future__ import annotations

import numpy as np

from ptcg_ai.learning import encoder
from ptcg_ai.learning.policy_model import PolicyModel, _UNKNOWN_CARD_EMBEDDING_INDEX


class BatchedPolicyModel(PolicyModel):
    """PolicyModel を継承し score_options_from_state のみ numpy batched に置換(数値等価)。"""

    def __init__(self, weights_path=None):
        super().__init__(weights_path)
        self._np_ready = False
        if self.is_ready:
            self._sm = np.asarray(self._state_mean, dtype=np.float64)
            self._ss = np.asarray(self._state_std, dtype=np.float64)
            self._om = np.asarray(self._option_mean, dtype=np.float64)
            self._os = np.asarray(self._option_std, dtype=np.float64)
            # std==0 は 0 除算回避(pure-Python 実装は std==0 のとき 0 を返す)
            self._ss_safe = np.where(self._ss != 0.0, self._ss, 1.0)
            self._os_safe = np.where(self._os != 0.0, self._os, 1.0)
            self._ss_nz = (self._ss != 0.0)
            self._os_nz = (self._os != 0.0)
            self._emb = np.asarray(self._card_embedding_table, dtype=np.float64)   # [card_id_max+1, dim]
            self._Wn = [np.asarray(W, dtype=np.float64) for (W, _b) in self._layers]   # [out, in]
            self._bn = [np.asarray(b, dtype=np.float64) for (_W, b) in self._layers]
            self._n_layers = len(self._layers)
            self._np_ready = True

    def _embed_rows(self, card_ids):
        idx = [c if (0 <= c <= self._card_id_max) else _UNKNOWN_CARD_EMBEDDING_INDEX for c in card_ids]
        return self._emb[idx]                          # [N, dim]

    def score_options_from_state(self, state, select):
        # consequence 特徴を使う重みは非対応(H4/Original は未使用)。未 ready は親のフォールバック。
        if self._consequence_fields:
            return super().score_options_from_state(state, select)
        if not self._np_ready or state is None or select is None or not select.option:
            return []
        # --- feature 構築は encoder(旧実装と完全同一の出力)---
        sf = encoder.encode_state_from_state(state)
        option_rows = encoder.encode_options_from_state(state, select)
        card_ids = encoder.encode_option_card_ids(state, select)
        n = len(select.option)
        if not option_rows or len(option_rows) != n:
            return []
        # --- standardize(pure-Python と同一: std==0 → 0)---
        sf_std = np.where(self._ss_nz, (np.asarray(sf, dtype=np.float64) - self._sm) / self._ss_safe, 0.0)   # [166]
        opt = np.asarray(option_rows, dtype=np.float64)                                                       # [N,65]
        opt_std = np.where(self._os_nz, (opt - self._om) / self._os_safe, 0.0)                                # [N,65]
        emb = self._embed_rows(card_ids)                                                                      # [N,dim]
        X = np.concatenate([np.broadcast_to(sf_std, (n, sf_std.shape[0])), opt_std, emb], axis=1)             # [N,239]
        # --- batched forward(最終層のみ活性化なし=生スコア、pure-Python と同一)---
        h = X
        for li in range(self._n_layers):
            z = h @ self._Wn[li].T + self._bn[li]      # [N, out]
            h = z if li == self._n_layers - 1 else np.maximum(z, 0.0)
        return h[:, 0].tolist()
