"""T1: 盤面Transformer + 既存方式のOption scoring(torch)。T1.1でCard ID変換方式を統一し、
正規化統計・checkpoint契約を追加した。

構成(ユーザー指示通り、これ以上は追加しない):

    board tokens -> Card/Zone/Owner/Numeric embedding -> 小型Board Transformer -> CLS表現
    CLS表現 ++ legacy_global_features ++ 既存方式のoption特徴 -> MLP -> 各合法手のscore

追加しないもの(T2以降の課題): pointer gather・cross-attention・option self-attention・
CriticとのTransformer共有・PPO更新・NumPy推論・worker組み込み。

## 既存 `TorchOptionPolicy`(`kaggle_replays/rl/torch_policy.py`)との対応

``option_scores()``(109〜123行目)は
``x = cat([standardize(state)を選択肢数ぶんbroadcast, standardize(option), card_embedding(option_cid)])``
を作り、``Linear -> ReLU -> Linear`` でスコアを出す。

T1が変える部分・変えない部分:

| 部分 | 既存 | T1 |
|---|---|---|
| 選択肢の65次元特徴(``option_dim``) | 標準化して結合 | **維持。変更なし** |
| 盤面の表現 | ``state_feat``(166/389次元固定ベクトル)を標準化してbroadcast | **置き換え**: 盤面トークン列 -> Board Transformer -> CLS出力(d_model次元)を使う |
| 盤面のうちglobal特徴 | ``state_feat``の一部としてまとめて標準化 | ``legacy_global_features``を別枠で標準化してbroadcast |
| 最終層 | ``Linear(in_dim, hidden) -> ReLU -> Linear(hidden, 1)`` | **維持。同じ2層構成**(in_dimだけ変わる) |

## T1.1での変更: Card ID変換方式の統一

旧T1は「選択肢の対象card id embeddingは既存(``card_embedding``、raw card idを直接index、
card_id_max+1テーブル)をそのまま流用」としていたが、**T1.1でこれをやめた。**
盤面・選択肢のどちらも ``card_vocab.py`` の固定語彙(PAD=0/UNK=1予約、append-only)で
indexし、raw card idを直接Embedding indexに使わない(``_clip_card_ids``も廃止)。

盤面用(``board_card_embedding``、d_model次元)と選択肢用(``option_card_embedding``、
``option_embed_dim``次元)は**別々のテーブル**(重みは共有しない)だが、**indexの体系は
共通**(どちらも同じ ``CardVocab`` を経由する)。既存の選択肢embedding(教師の重み、
raw card idインデックス・card_id_max+1サイズ)を引き継ぐ場合は
``migrate_option_embedding_from_teacher()`` で「raw id -> 固定vocab index」の対応に
従って行を移植する。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoardTokenEmbedding(nn.Module):
    """盤面トークン1個を d_model 次元へ埋め込む(Numeric + Card + Zone + Owner)。"""

    def __init__(self, board_vocab_size: int, num_zones: int, d_model: int,
                numeric_dim: int = 11, pad_index: int = 0):
        super().__init__()
        self.numeric_proj = nn.Linear(numeric_dim, d_model)
        self.card_embedding = nn.Embedding(board_vocab_size, d_model, padding_idx=pad_index)
        self.zone_embedding = nn.Embedding(num_zones, d_model)
        self.owner_embedding = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, numeric: torch.Tensor, card_idx: torch.Tensor,
               zone_ids: torch.Tensor, owner_ids: torch.Tensor) -> torch.Tensor:
        x = (self.numeric_proj(numeric) + self.card_embedding(card_idx)
             + self.zone_embedding(zone_ids) + self.owner_embedding(owner_ids))
        return self.norm(x)


class BoardTransformer(nn.Module):
    """盤面トークン列 -> 学習可能な[CLS]を先頭に足してself-attention -> [CLS]出力。

    盤面トークンが0件(``mask``が全てFalse)でも、[CLS]は常にmask=Trueで残すため、
    どの決定点でも最低1トークン(CLS自身)は残り、softmaxが全て-infになるケースは起きない。
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout,
            activation="relu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, d_model)。mask: (B, N) bool、True=実トークン。戻り: (B, d_model)。"""
        b = tokens.shape[0]
        cls = self.cls.expand(b, 1, -1)
        x = torch.cat([cls, tokens], dim=1)
        cls_mask = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        key_padding_mask = ~full_mask  # nn.Transformerは「Trueを無視する」規約
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return out[:, 0]  # [CLS]


class T1OptionPolicy(nn.Module):
    def __init__(self, global_dim: int, option_dim: int, board_vocab_size: int, num_zones: int,
                d_model: int = 64, num_heads: int = 4, num_board_layers: int = 2,
                ffn_dim: int = 256, dropout: float = 0.05,
                option_embed_dim: int = 8, hidden: int = 32,
                use_board_card_id: bool = True, value_hidden: int = 32):
        super().__init__()
        self.use_board_card_id = use_board_card_id
        self.board_embed = BoardTokenEmbedding(board_vocab_size, num_zones, d_model)
        self.board_transformer = BoardTransformer(d_model, num_heads, num_board_layers, ffn_dim, dropout)

        # T1.1: 盤面と同じvocab(card_vocab.py)でindexする。raw card idは使わない。
        self.option_card_embedding = nn.Embedding(board_vocab_size, option_embed_dim, padding_idx=0)

        self.register_buffer("global_mean", torch.zeros(global_dim))
        self.register_buffer("global_std", torch.ones(global_dim))
        self.register_buffer("option_mean", torch.zeros(option_dim))
        self.register_buffer("option_std", torch.ones(option_dim))
        # 盤面トークンの数値特徴(11次元)の標準化(T1.1で追加。全スロット共通の統計を使う。
        # 現行モデルはスロットごとに別々の平均/分散を持つため流用できず、新規計算が必要
        # ——理由は正規化統計の実装報告§2参照)。
        self.register_buffer("board_numeric_mean", torch.zeros(11))
        self.register_buffer("board_numeric_std", torch.ones(11))

        in_dim = d_model + global_dim + option_dim + option_embed_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

        # [critic] 状態価値V(s)専用ヘッド(D2.1後PPO pilot、オーロンゲcritic実験)。
        # 入力はcls(盤面)+global特徴のみ——option/actionには一切依存しない
        # (合法手の並び順・実際に選んだ行動がvalueへ漏れないようにするため、
        # policyヘッド(fc1/fc2、option方向にbroadcastして計算)とは完全に別の
        # 小さいMLPにしている)。既存checkpoint(value headを持たない)は
        # load_checkpointがこのパラメータだけ新規初期化する(下記参照)。
        self.value_fc1 = nn.Linear(d_model + global_dim, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    @staticmethod
    def _standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        safe = torch.where(std != 0, std, torch.ones_like(std))
        out = (x - mean) / safe
        return torch.where(std != 0, out, torch.zeros_like(out))

    def _encode_state(self, board_numeric, board_card_idx, board_zone_ids, board_owner_ids, board_mask,
                      global_features) -> tuple[torch.Tensor, torch.Tensor]:
        """盤面CLS表現とglobal特徴(標準化済み)を返す(policy/value共通、option非依存)。"""
        numeric = self._standardize(board_numeric, self.board_numeric_mean, self.board_numeric_std)
        board_idx = board_card_idx if self.use_board_card_id else torch.zeros_like(board_card_idx)
        tokens = self.board_embed(numeric, board_idx, board_zone_ids, board_owner_ids)
        cls = self.board_transformer(tokens, board_mask)  # (B, d_model)
        g = self._standardize(global_features, self.global_mean, self.global_std)  # (B, global_dim)
        return cls, g

    def forward(self, board_numeric, board_card_idx, board_zone_ids, board_owner_ids, board_mask,
               global_features, option_features, option_card_idx, option_mask) -> torch.Tensor:
        """全テンソルはbatch_first。``board_card_idx``/``option_card_idx``はどちらも
        ``card_vocab.CardVocab.index_of()``済みの固定vocab index(raw card idではない)。
        戻り: (B, max_opt) のスコア(paddingは-inf)。(既存T1.1と完全に同じ計算、未変更)"""
        cls, g = self._encode_state(board_numeric, board_card_idx, board_zone_ids, board_owner_ids,
                                    board_mask, global_features)
        o = self._standardize(option_features, self.option_mean, self.option_std)  # (B, max_opt, option_dim)
        oe = self.option_card_embedding(option_card_idx)  # (B, max_opt, option_embed_dim)

        n_opt = o.shape[1]
        cls_b = cls.unsqueeze(1).expand(-1, n_opt, -1)
        g_b = g.unsqueeze(1).expand(-1, n_opt, -1)
        x = torch.cat([cls_b, g_b, o, oe], dim=-1)
        h = F.relu(self.fc1(x))
        score = self.fc2(h).squeeze(-1)  # (B, max_opt)
        return score.masked_fill(~option_mask, float("-inf"))

    def value(self, board_numeric, board_card_idx, board_zone_ids, board_owner_ids, board_mask,
             global_features) -> torch.Tensor:
        """[critic] 状態価値V(s)。optionを一切受け取らない(呼び出しシグネチャの時点で
        行動非依存を強制する)。戻り: (B,)。"""
        cls, g = self._encode_state(board_numeric, board_card_idx, board_zone_ids, board_owner_ids,
                                    board_mask, global_features)
        x = torch.cat([cls, g], dim=-1)
        h = F.relu(self.value_fc1(x))
        return self.value_fc2(h).squeeze(-1)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def migrate_option_embedding_from_teacher(teacher_card_embedding_table: list[list[float]],
                                          vocab) -> torch.Tensor:
    """教師(既存MLP)のraw-id-indexedな ``card_embedding.table``(card_id_max+1行)を、
    ``vocab``(``card_vocab.CardVocab``)のindex体系に並べ替える。

    Returns:
        (vocab.size, embed_dim) のtensor。PAD行(index0)はゼロ。UNK行(index1)は
        教師の index0(``policy_model.py`` の ``_UNKNOWN_CARD_EMBEDDING_INDEX``、
        「識別なし/範囲外」用の予約行)をそのまま引き継ぐ。
    """
    table = torch.tensor(teacher_card_embedding_table, dtype=torch.float32)
    embed_dim = table.shape[1]
    out = torch.zeros(vocab.size, embed_dim)
    out[1] = table[0]  # UNK <- 教師の「識別なし」行
    for cid in vocab.card_ids:
        idx = vocab.index_of(cid)
        if 0 <= cid < table.shape[0]:
            out[idx] = table[cid]
    return out


# ---------------------------------------------------------------------------
# checkpoint契約(T1.1)
# ---------------------------------------------------------------------------
# checkpointは重みだけでなく、それが作られた環境(token schemaのバージョン・語彙の
# バージョン/hash・特徴profile・global feature manifestのhash・正規化統計)を一緒に
# 保存する。ロード時にこれらが**現在の環境と一致するか呼び出し側が確認できるように**
# しておかないと、「vocabularyだけ更新されたのにindexの意味が変わったcheckpointを
# 気づかず使い続ける」といった静かな不整合が起こる。


class CheckpointMismatchError(ValueError):
    pass


NORMALIZATION_MODES = ("teacher_stats", "student_train_stats")


def save_checkpoint(model: T1OptionPolicy, path, model_config: dict, *,
                    token_schema_version: str, vocabulary_version: int, vocabulary_hash: str,
                    feature_profile, global_feature_manifest_hash: str,
                    normalization_mode: str, normalization_stats: dict,
                    card_vocab_size: int) -> None:
    if normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(f"normalization_modeは{NORMALIZATION_MODES}のいずれか: {normalization_mode!r}")
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "token_schema_version": token_schema_version,
        "vocabulary_version": vocabulary_version,
        "vocabulary_hash": vocabulary_hash,
        "feature_profile": feature_profile,
        "global_feature_manifest_hash": global_feature_manifest_hash,
        "normalization_mode": normalization_mode,
        "normalization_stats": normalization_stats,
        "use_board_card_id": model_config.get("use_board_card_id", True),
        "card_vocab_size": card_vocab_size,
    }
    torch.save(payload, path)


# T1.1補修: load_checkpointが検査必須にする項目。呼び出し側はこれら全てに対応する
# 「現在の環境の値」を引数として渡さなければならない(省略できない=デフォルト値を
# 持たせない)。渡し忘れて検査が黙って素通りする、という設計事故を防ぐため。
_REQUIRED_CHECKS = (
    "token_schema_version", "vocabulary_version", "vocabulary_hash",
    "card_vocab_size", "feature_profile", "global_feature_manifest_hash",
)


def load_checkpoint(path, *, token_schema_version: str, vocabulary_version: int,
                    vocabulary_hash: str, card_vocab_size: int, feature_profile,
                    global_feature_manifest_hash: str, allow_unsafe_mismatch: bool = False,
                    map_location=None) -> tuple[T1OptionPolicy, dict]:
    """checkpointを読み込む。

    ``token_schema_version``/``vocabulary_version``/``vocabulary_hash``/
    ``card_vocab_size``/``feature_profile``/``global_feature_manifest_hash`` は
    **すべて必須引数**(既定値を持たない)。呼び出し側は「現在の環境の値」を
    必ず渡す必要があり、渡し忘れによって検査が素通りすることはない。

    一致しない項目が1つでもあれば ``CheckpointMismatchError``。意図的に検査を
    無効化したい場合(旧checkpointの中身をとにかく読みたいデバッグ用途等)は
    ``allow_unsafe_mismatch=True`` を明示する(これも既定は``False``で、
    黙って安全性を回避できないようにしている)。

    ``model_config``・``use_board_card_id``・``normalization_mode``/
    ``normalization_stats`` はcheckpoint自身の内容として存在必須(欠けていれば
    ``KeyError``。「現在の環境の値」と比較する対象が無いフィールドなので、
    一致検査ではなく存在検査のみ行う)。
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)

    for key in ("model_config", "use_board_card_id", "normalization_mode", "normalization_stats"):
        if key not in payload:
            raise CheckpointMismatchError(f"checkpointに必須フィールド{key!r}が無い: {path}")
    if payload["normalization_mode"] not in ("teacher_stats", "student_train_stats"):
        raise CheckpointMismatchError(
            f"normalization_modeが不正: {payload['normalization_mode']!r}")
    if payload["model_config"].get("board_vocab_size") != payload["card_vocab_size"]:
        raise CheckpointMismatchError(
            "model_config.board_vocab_size と card_vocab_size が食い違っている"
            f"({payload['model_config'].get('board_vocab_size')} != {payload['card_vocab_size']})")

    current = {
        "token_schema_version": token_schema_version, "vocabulary_version": vocabulary_version,
        "vocabulary_hash": vocabulary_hash, "card_vocab_size": card_vocab_size,
        "feature_profile": feature_profile, "global_feature_manifest_hash": global_feature_manifest_hash,
    }
    mismatches = [(name, current[name], payload[name]) for name in _REQUIRED_CHECKS
                 if current[name] != payload[name]]
    if mismatches and not allow_unsafe_mismatch:
        detail = "; ".join(f"{n}: checkpoint={a!r} != 現在={e!r}" for n, e, a in mismatches)
        raise CheckpointMismatchError(f"checkpointが現在の環境と一致しない: {detail}")

    model = T1OptionPolicy(**payload["model_config"])
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if unexpected:
        raise CheckpointMismatchError(f"checkpointに未知のパラメータがある: {unexpected}")
    # [critic] value headを持たない既存checkpoint(policy蒸留・critic無しPPO)を読む場合、
    # value_fc1/value_fc2だけが「無い」のは正常(新規初期化されたまま使う)。
    # それ以外のキーが欠けているのは真の不整合なので従来通り拒否する。
    if missing and not all(k.startswith("value_fc") for k in missing):
        raise CheckpointMismatchError(f"checkpointに想定外の欠落パラメータがある: {missing}")
    return model, payload
