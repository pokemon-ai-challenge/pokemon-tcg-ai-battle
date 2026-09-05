"""torch 版 選択肢スコア方策(RL用)。既存 policy_weights.json と数値一致(M0)。

既存 `sample_submission/ptcg_ai/learning/policy_model.py` の pure-Python フォワードパスを
torch で厳密に再現する。これにより:
- **BC初期化** = 既存 `policy_weights_<arch>.json` を `from_json` でロードするだけ。
- **可搬性** = RL 後に `to_json_payload` で同じ schema に書き戻し、eval/提出は pure-Python のまま。

フォワード(policy_model._forward と一致):
  x = [ (state-mean)/std (166) ++ (option-mean)/std (65) ++ card_embedding (8) ]  # 239
  h = ReLU(W0 x + b0)                                                             # hidden
  score = W1 h + b1                                                               # 線形(活性化なし)
方策 π(a|s) = softmax(scores / temperature)。

注: consequence 特徴(meta.consequence_fields 非空)の重みは PoC 非対応(既定 dragapult は空)。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class TorchOptionPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        option_dim: int,
        card_id_max: int,
        embed_dim: int,
        hidden_sizes: list[int],
    ):
        super().__init__()
        self.state_dim = state_dim
        self.option_dim = option_dim
        self.card_id_max = card_id_max
        self.embed_dim = embed_dim

        # 標準化パラメータ(buffer: 学習しない)。
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("option_mean", torch.zeros(option_dim))
        self.register_buffer("option_std", torch.ones(option_dim))
        # card 埋め込み(index 0 = 識別なし/範囲外)。RL では学習可(勾配を流す)。
        self.card_embedding = nn.Embedding(card_id_max + 1, embed_dim)

        in_dim = state_dim + option_dim + embed_dim
        layers: list[nn.Module] = []
        prev = in_dim
        self._linears = nn.ModuleList()
        for h in hidden_sizes:
            self._linears.append(nn.Linear(prev, h))
            prev = h
        self._out = nn.Linear(prev, 1)

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, weights_path: str | Path) -> "TorchOptionPolicy":
        payload = json.loads(Path(weights_path).read_text(encoding="utf-8"))
        std = payload["standardization"]
        ce = payload["card_embedding"]
        layers = payload["layers"]
        state_dim = len(std["state_mean"])
        option_dim = len(std["option_mean"])
        embed_dim = int(ce["dim"])
        card_id_max = int(ce["card_id_max"])
        # 中間層のサイズ(最終層以外)。
        hidden_sizes = [len(layer["bias"]) for layer in layers[:-1]]

        model = cls(state_dim, option_dim, card_id_max, embed_dim, hidden_sizes)
        with torch.no_grad():
            model.state_mean.copy_(torch.tensor([float(v) for v in std["state_mean"]]))
            model.state_std.copy_(torch.tensor([float(v) for v in std["state_std"]]))
            model.option_mean.copy_(torch.tensor([float(v) for v in std["option_mean"]]))
            model.option_std.copy_(torch.tensor([float(v) for v in std["option_std"]]))
            model.card_embedding.weight.copy_(
                torch.tensor([[float(v) for v in row] for row in ce["table"]])
            )
            for i, layer in enumerate(layers[:-1]):
                model._linears[i].weight.copy_(
                    torch.tensor([[float(w) for w in row] for row in layer["weight"]])
                )
                model._linears[i].bias.copy_(torch.tensor([float(v) for v in layer["bias"]]))
            last = layers[-1]
            model._out.weight.copy_(
                torch.tensor([[float(w) for w in row] for row in last["weight"]])
            )
            model._out.bias.copy_(torch.tensor([float(v) for v in last["bias"]]))
        return model

    # ------------------------------------------------------------------
    def _standardize_state(self, state_feat: torch.Tensor) -> torch.Tensor:
        # std==0 の次元は 0 にする(policy_model._forward と同じ)。
        safe = torch.where(self.state_std != 0, self.state_std, torch.ones_like(self.state_std))
        out = (state_feat - self.state_mean) / safe
        return torch.where(self.state_std != 0, out, torch.zeros_like(out))

    def _standardize_option(self, option_feats: torch.Tensor) -> torch.Tensor:
        safe = torch.where(self.option_std != 0, self.option_std, torch.ones_like(self.option_std))
        out = (option_feats - self.option_mean) / safe
        return torch.where(self.option_std != 0, out, torch.zeros_like(out))

    def _clip_card_ids(self, card_ids: torch.Tensor) -> torch.Tensor:
        valid = (card_ids >= 0) & (card_ids <= self.card_id_max)
        return torch.where(valid, card_ids, torch.zeros_like(card_ids))

    def option_scores(
        self,
        state_feat: torch.Tensor,   # [state_dim]
        option_feats: torch.Tensor, # [n_options, option_dim]
        card_ids: torch.Tensor,     # [n_options] (long)
    ) -> torch.Tensor:              # [n_options]
        n = option_feats.shape[0]
        s = self._standardize_state(state_feat).unsqueeze(0).expand(n, -1)  # [n, state_dim]
        o = self._standardize_option(option_feats)                          # [n, option_dim]
        e = self.card_embedding(self._clip_card_ids(card_ids.long()))       # [n, embed_dim]
        x = torch.cat([s, o, e], dim=-1)                                    # [n, in_dim]
        h = x
        for lin in self._linears:
            h = torch.relu(lin(h))
        return self._out(h).squeeze(-1)                                     # [n]

    def option_scores_flat(
        self,
        state_rows: torch.Tensor,   # [N, state_dim] (各行がその選択肢の属する決定点の state)
        option_rows: torch.Tensor,  # [N, option_dim]
        card_ids: torch.Tensor,     # [N] (long)
    ) -> torch.Tensor:              # [N]
        """複数決定点の選択肢を平坦化(1バッチ)してスコア。PPO 更新の効率化用
        (勾配を流すので no_grad にしない)。標準化は broadcast で行単位に適用される。"""
        s = self._standardize_state(state_rows)
        o = self._standardize_option(option_rows)
        e = self.card_embedding(self._clip_card_ids(card_ids.long()))
        x = torch.cat([s, o, e], dim=-1)
        h = x
        for lin in self._linears:
            h = torch.relu(lin(h))
        return self._out(h).squeeze(-1)

    def distribution(self, state_feat, option_feats, card_ids, temperature: float = 1.0):
        scores = self.option_scores(state_feat, option_feats, card_ids)
        return torch.distributions.Categorical(logits=scores / temperature)

    # ------------------------------------------------------------------
    def to_json_payload(self, base_payload: dict) -> dict:
        """RL 後の重みを既存 schema の dict に書き戻す(base_payload の meta 等を継承)。"""
        payload = json.loads(json.dumps(base_payload))  # deep copy
        payload["standardization"] = {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "option_mean": self.option_mean.tolist(),
            "option_std": self.option_std.tolist(),
        }
        payload["card_embedding"]["table"] = self.card_embedding.weight.detach().cpu().tolist()
        new_layers = []
        for lin in self._linears:
            new_layers.append({
                "weight": lin.weight.detach().cpu().tolist(),
                "bias": lin.bias.detach().cpu().tolist(),
            })
        new_layers.append({
            "weight": self._out.weight.detach().cpu().tolist(),
            "bias": self._out.bias.detach().cpu().tolist(),
        })
        payload["layers"] = new_layers
        return payload
