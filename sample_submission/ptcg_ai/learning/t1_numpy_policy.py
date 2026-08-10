"""T1(盤面Transformer)のNumPy推論(Kaggle提出用、torch不要)。

``kaggle_replays/rl/token_policy_t1.py`` の ``T1OptionPolicy.forward`` を、学習側の
torch実装は一切変更せずにnumpyで再現する(重みは``export_t1_npz.py``でnpzへ変換した
ものを読む)。提出環境にtorchが無い前提のため、このモジュールはnumpyのみに依存する。

対応する構成(token_policy_t1.py通り):
    board tokens -> BoardTokenEmbedding -> BoardTransformer(nn.TransformerEncoderLayer,
    norm_first=True, 2層, 4 head) -> [CLS] -> legacy_global/option特徴と結合 -> MLP -> score
"""

from __future__ import annotations

import numpy as np


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """torch Linear: y = x @ weight.T + bias(weightは(out,in)形状のまま保存)。"""
    return x @ weight.T + bias


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class T1NumpyPolicy:
    """``T1OptionPolicy``のnumpy版(推論専用、勾配・学習機能は無い)。"""

    def __init__(self, weights: dict, num_heads: int = 4, num_layers: int = 2):
        self.w = weights
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_model = weights["board_embed.norm.weight"].shape[0]
        self.head_dim = self.d_model // num_heads

    def _standardize(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        safe = np.where(std != 0, std, 1.0)
        out = (x - mean) / safe
        return np.where(std != 0, out, 0.0)

    def _self_attention(self, x: np.ndarray, key_padding_mask: np.ndarray, prefix: str) -> np.ndarray:
        """x: (B, N, d_model)。key_padding_mask: (B, N) bool、True=無視するkey。"""
        w = self.w
        in_w = w[f"{prefix}.self_attn.in_proj_weight"]  # (3*d_model, d_model)
        in_b = w[f"{prefix}.self_attn.in_proj_bias"]  # (3*d_model,)
        d = self.d_model
        wq, wk, wv = in_w[:d], in_w[d:2 * d], in_w[2 * d:]
        bq, bk, bv = in_b[:d], in_b[d:2 * d], in_b[2 * d:]

        q = _linear(x, wq, bq)
        k = _linear(x, wk, bk)
        v = _linear(x, wv, bv)

        B, N, _ = x.shape
        h, hd = self.num_heads, self.head_dim
        q = q.reshape(B, N, h, hd).transpose(0, 2, 1, 3)  # (B,h,N,hd)
        k = k.reshape(B, N, h, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, h, hd).transpose(0, 2, 1, 3)

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)  # (B,h,N,N)
        mask = key_padding_mask[:, None, None, :]  # (B,1,1,N) broadcast over heads・queries
        scores = np.where(mask, -np.inf, scores)
        attn = _softmax(scores, axis=-1)
        out = attn @ v  # (B,h,N,hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, d)

        out_w = w[f"{prefix}.self_attn.out_proj.weight"]
        out_b = w[f"{prefix}.self_attn.out_proj.bias"]
        return _linear(out, out_w, out_b)

    def _encoder_layer(self, x: np.ndarray, key_padding_mask: np.ndarray, layer_idx: int) -> np.ndarray:
        """norm_first=Trueのnn.TransformerEncoderLayerと同じ計算順序。"""
        w = self.w
        prefix = f"board_transformer.encoder.layers.{layer_idx}"
        normed1 = _layer_norm(x, w[f"{prefix}.norm1.weight"], w[f"{prefix}.norm1.bias"])
        x = x + self._self_attention(normed1, key_padding_mask, prefix)

        normed2 = _layer_norm(x, w[f"{prefix}.norm2.weight"], w[f"{prefix}.norm2.bias"])
        ff = _linear(normed2, w[f"{prefix}.linear1.weight"], w[f"{prefix}.linear1.bias"])
        ff = _relu(ff)
        ff = _linear(ff, w[f"{prefix}.linear2.weight"], w[f"{prefix}.linear2.bias"])
        x = x + ff
        return x

    def _board_transformer(self, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """tokens: (B,N,d_model)。mask: (B,N) bool、True=実トークン。戻り: (B,d_model)。"""
        B = tokens.shape[0]
        cls = np.broadcast_to(self.w["board_transformer.cls"][0], (B, 1, self.d_model))
        x = np.concatenate([cls, tokens], axis=1)
        cls_mask = np.ones((B, 1), dtype=bool)
        full_mask = np.concatenate([cls_mask, mask], axis=1)
        key_padding_mask = ~full_mask
        for i in range(self.num_layers):
            x = self._encoder_layer(x, key_padding_mask, i)
        return x[:, 0]

    def forward(self, board_numeric, board_card_idx, board_zone_ids, board_owner_ids, board_mask,
               global_features, option_features, option_card_idx, option_mask) -> np.ndarray:
        """全て(B, ...)のnumpy配列。戻り: (B, max_opt)のスコア(paddingは-inf)。"""
        w = self.w
        numeric = self._standardize(board_numeric, w["board_numeric_mean"], w["board_numeric_std"])
        board_idx = board_card_idx

        tokens = (_linear(numeric, w["board_embed.numeric_proj.weight"], w["board_embed.numeric_proj.bias"])
                 + w["board_embed.card_embedding.weight"][board_idx]
                 + w["board_embed.zone_embedding.weight"][board_zone_ids]
                 + w["board_embed.owner_embedding.weight"][board_owner_ids])
        tokens = _layer_norm(tokens, w["board_embed.norm.weight"], w["board_embed.norm.bias"])

        cls = self._board_transformer(tokens, board_mask)  # (B, d_model)

        g = self._standardize(global_features, w["global_mean"], w["global_std"])
        o = self._standardize(option_features, w["option_mean"], w["option_std"])
        oe = w["option_card_embedding.weight"][option_card_idx]

        n_opt = o.shape[1]
        cls_b = np.broadcast_to(cls[:, None, :], (cls.shape[0], n_opt, cls.shape[1]))
        g_b = np.broadcast_to(g[:, None, :], (g.shape[0], n_opt, g.shape[1]))
        x = np.concatenate([cls_b, g_b, o, oe], axis=-1)
        h = _relu(_linear(x, w["fc1.weight"], w["fc1.bias"]))
        score = _linear(h, w["fc2.weight"], w["fc2.bias"])[..., 0]
        return np.where(option_mask, score, -np.inf)


def load_t1_numpy_policy(npz_path: str) -> T1NumpyPolicy:
    data = np.load(npz_path)
    weights = {k.replace("__", "."): v for k, v in data.items()}
    return T1NumpyPolicy(weights)


def build_single_decision_inputs(board_tokens, board_card_ids_raw, option_card_ids_raw,
                                 legacy_option_feats, legacy_global_feat, vocab) -> dict:
    """1決定点(batch=1)ぶんの``T1NumpyPolicy.forward``入力を組み立てる。
    batch=1なのでpaddingは不要(mask常に全Trueで、複数決定点をまとめる
    ``token_batch.build_batch``相当の処理は要らない)。"""
    from ptcg_ai.learning.board_tokens import owner_of_zone

    n_board = len(board_card_ids_raw)
    n_opt = len(option_card_ids_raw)
    zone_ids = list(board_tokens.zone_ids)

    board_numeric = (np.asarray(board_tokens.numeric_features, dtype=np.float32).reshape(1, n_board, 11)
                     if n_board else np.zeros((1, 0, 11), dtype=np.float32))
    board_card_idx = np.array([[vocab.index_of(int(c)) for c in board_card_ids_raw]], dtype=np.int64)
    board_zone_ids = np.array([zone_ids], dtype=np.int64)
    board_owner_ids = np.array([[owner_of_zone(z) for z in zone_ids]], dtype=np.int64)
    board_mask = np.ones((1, n_board), dtype=bool)

    global_features = np.asarray(legacy_global_feat, dtype=np.float32).reshape(1, -1)
    option_features = np.asarray(legacy_option_feats, dtype=np.float32).reshape(1, n_opt, -1)
    option_card_idx = np.array([[vocab.index_of(int(c)) for c in option_card_ids_raw]], dtype=np.int64)
    option_mask = np.ones((1, n_opt), dtype=bool)

    return dict(board_numeric=board_numeric, board_card_idx=board_card_idx,
               board_zone_ids=board_zone_ids, board_owner_ids=board_owner_ids,
               board_mask=board_mask, global_features=global_features,
               option_features=option_features, option_card_idx=option_card_idx,
               option_mask=option_mask)
