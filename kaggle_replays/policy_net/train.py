#!/usr/bin/env python3
"""features.npz から模倣ポリシー(選択肢スコアリング MLP)を学習し、policy_weights.json を書き出す。

流れ:
  1. features.npz を読み(``allow_pickle=True``、option_features/option_card_ids が ragged
     object 配列のため)、train/val/test split に分ける。
  2. 標準化: state_features は train split から mean/std を計算(value_net と同じ std<1e-6 クリップ)。
     option_features は train split の全選択肢を縦に連結した1本の行列から mean/std を計算する。
     card_id(``option_card_ids``)は標準化しない(埋め込みテーブルへの生インデックスとして使う)。
  3. ベースライン(比較用、成果物には含めない): 選択肢特徴のみを入力にした隠れ層なし線形モデル
     (card embedding なし、状態特徴なし。変更なし)。
  4. 本命: state_features ++ option_features(選択肢ごと) ++ card_embedding(card_id) を入力にした
     ``Linear(in, 32) -> ReLU -> Linear(32, 1)`` の小型 MLP(選択肢間で重み共有)。
     card_embedding は ``nn.Embedding(card_id_max + 1, 8)``(2026-07-20 追加。個別カード識別の
     ための学習可能な埋め込み。index 0 = 識別なし/範囲外の予約枠)。埋め込みパラメータも
     本命モデルと同じ optimizer で一緒に学習する。
  5. 損失: リストワイズ softmax 交差エントロピー(同一決定点内の選択肢スコアに softmax、
     選ばれた選択肢への NLL を sample weight で重み付けして意思決定点間で平均)。
     ミニバッチ(決定点数十件程度)単位で合算して逆伝播する。パディングは使わない。
  6. val split の重み付き平均 NLL で早期終了。
  7. test split で Top-1一致率・平均重み付きNLLを算出。
  8. sample_submission/ptcg_ai/learning/policy_weights.json へエクスポート(``card_embedding``
     ブロック込み)。書き出し直後に ``ptcg_ai/learning/policy_model.py`` の ``_forward`` と
     数値的に一致する独立実装(pure Python、本ファイル内で JSON を再読み込みして再計算)で
     照合する(誤差 1e-6 未満を要求)。

キャリブレーションは実装しない(step2-algorithm-selection.md §5: argmax は温度に対して不変なので不要)。

メモリ方針: state/option の標準化済み特徴(state_features_std、option_features_std)は
一度だけ計算して使い回すが、決定点ごとに state++option を連結した入力行列は split 全体を
まとめて事前構築しない(187,690件 x 選択肢平均7.75件 x (166+65)次元 で GB 級になり得るため)。
ミニバッチ/評価バッチの分だけその場で連結する(``build_batch``)。card_id は int なので
標準化前後を問わず全件保持してもメモリ負荷は軽い(削除しない)。

使い方:
  PYTHONIOENCODING=utf-8 python train.py
  PYTHONIOENCODING=utf-8 python train.py --limit 5000 --max-epochs 5  (動作確認用)

skill-concentration実験用フラグ(policymodel-skill-concentration-implementation-plan.md Step2):
  --out-weights PATH  重みJSONの書き出し先(既定: 現行の sample_submission/ptcg_ai/learning/
                       policy_weights.json、= 本番が読む場所)。実験構成は本番重みを上書きしない
                       よう、必ず別パス(例: policy_weights_configA.json)を明示指定すること。

対面クラスタ別ファインチューン用フラグ(requirements-kamitsuorochi-2026-08-12.md §10-1。3フラグとも
既定値では従来と完全に同一の挙動):
  --init-weights PATH  既存の policy_weights JSON を本命モデル(fc1/fc2/embedding)の初期値にする
                       (ウォームスタート)。標準化パラメータは部分集合から再計算せず base の値を
                       そのまま使う。hand_card_vocab/opponent_card_vocab/consequence_fields/
                       use_board_set/ablated_features/hidden_size/card_id_max が base と一致
                       しない場合はエラー停止。
  --lr FLOAT           optimizerの学習率(既定: 現行ハードコード値 1e-3 と同一)。ファインチューン
                       では1/5〜1/10に落として使う。
  --row-mask PATH.npy  bool配列。True行のみ train/val/test 全splitに適用して学習・評価する。
                       長さは features.npz の行数と一致必須。適用後にいずれかのsplitが0件なら
                       エラー停止。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.learning.encoder import (  # noqa: E402
    BASE_FEATURE_COUNT,
    CONSEQUENCE_FEATURE_NAMES,
    FEATURE_NAMES,
    OPTION_FEATURE_COUNT,
    OPTION_FEATURE_NAMES,
)

_DEFAULT_FEATURES = _HERE / "features.npz"
_WEIGHTS_OUT_PATH = _SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "learning" / "policy_weights.json"

_SEED = 42
_HIDDEN_SIZE = 32
_EMBED_DIM = 8  # card_embedding の次元。固定(調整対象外、コーディネーター指定)。
_MAX_EPOCHS = 100
_PATIENCE = 10
_BATCH_SIZE = 64  # ミニバッチ = 意思決定点の件数(選択肢数ではない)
_EVAL_BATCH_SIZE = 1024
_LR = 1e-3
_TRAIN, _VAL, _TEST = 0, 1, 2

_IN_DIM_MAIN = BASE_FEATURE_COUNT + OPTION_FEATURE_COUNT  # embed_dim は PolicyScorer 内部で加算
_IN_DIM_BASELINE = OPTION_FEATURE_COUNT

_N_SELF_CHECK = 50
_SELF_CHECK_TOL = 1e-6


# ---------------------------------------------------------------------------
# モデル: 選択肢1件を独立にスコアリングする(選択肢間で重み共有)。
# forward は shape (n_i, in_dim) と shape (n_i,) の card_id を受けて shape (n_i,) のスコアを返す。
# ---------------------------------------------------------------------------
class PolicyScorer(nn.Module):
    """embedding(card_id) ++ Linear(in+embed, 32) -> ReLU -> Linear(32, 1)。

    本命モデル(状態 + 選択肢特徴 + card embedding)。card_embedding.table[0] は
    「識別なし/範囲外」の予約枠(index 0)で、他の埋め込み行と同様に学習される
    (特別な初期化はしない。デフォルトのランダム初期化のまま)。
    """

    def __init__(self, in_dim: int, card_id_max: int, embed_dim: int = _EMBED_DIM, hidden: int = _HIDDEN_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(card_id_max + 1, embed_dim)
        self.fc1 = nn.Linear(in_dim + embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(
        self, x: torch.Tensor, card_ids: torch.Tensor | None, board_card_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # board_card_ids は T2 対応モデル(PolicyScorerBoardSet)とシグネチャを揃えるためだけの
        # 引数で、ここでは無視する(T1 のこのクラス自体は変更しない。比較実験の対照群)。
        emb = self.embedding(card_ids)
        h = torch.relu(self.fc1(torch.cat([x, emb], dim=1)))
        z = self.fc2(h)
        return z.squeeze(-1)


class PolicyScorerBoardSet(nn.Module):
    """T2(design-transformer-representation-2026-08-08.md §5.4 段階1、Deep Sets)。

    PolicyScorer に、盤面12スロット(``encoder.BOARD_SLOTS``、自分6+相手6)の card_id を
    Deep Sets(mean/max/sum pooling、self-attention 無し)で集約した表現を追加する。
    card_embedding テーブルは選択肢側の card_embedding と共有する(§5.3「同じ card_id は
    同じ位置」の方針。手札/場/トラッシュのカウント特徴で先取りした方針を embedding にも適用)。

    盤面の空きスロット(card_id=0、"識別なし/範囲外"の予約枠)も pooling に含まれる
    (embedding index 0 が「不在」を表す値として学習される想定)。マスクして空きスロットを
    pooling から除外する改良は T3(Set Transformer)以降で検討する。

    **計算コストに関する注記(設計書 §6.3)**: ここでは決定点内の選択肢ごとに board pooling を
    繰り返し計算する(build_batch が盤面 card_id を選択肢行へタイル化するため)。学習
    (PyTorch, CPU, 時間制約が緩い)ではこの非効率を許容するが、本番推論(pure Python)側の
    実装(T2: 本番推論側にSet Encoderを実装、policy_model.py)では「1判断につき1回だけ計算する」
    最適化を必ず行うこと。
    """

    def __init__(self, in_dim: int, card_id_max: int, embed_dim: int = _EMBED_DIM, hidden: int = _HIDDEN_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(card_id_max + 1, embed_dim)
        self.embed_dim = embed_dim
        # in_dim(state++option) + card_embedding(選択肢1枚) + board_pooled(mean+max+sum)
        self.fc1 = nn.Linear(in_dim + embed_dim + embed_dim * 3, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def _board_pooled(self, board_card_ids: torch.Tensor) -> torch.Tensor:
        """``board_card_ids``: (n_i, BOARD_SLOTS) -> (n_i, embed_dim*3)。"""
        board_emb = self.embedding(board_card_ids)  # (n_i, BOARD_SLOTS, embed_dim)
        mean_pool = board_emb.mean(dim=1)
        max_pool = board_emb.max(dim=1).values
        sum_pool = board_emb.sum(dim=1)
        return torch.cat([mean_pool, max_pool, sum_pool], dim=1)

    def forward(self, x: torch.Tensor, card_ids: torch.Tensor | None, board_card_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(card_ids)
        board_pooled = self._board_pooled(board_card_ids)
        h = torch.relu(self.fc1(torch.cat([x, emb, board_pooled], dim=1)))
        z = self.fc2(h)
        return z.squeeze(-1)


class LinearScorer(nn.Module):
    """隠れ層なしの線形モデル。ベースライン(選択肢特徴のみ、状態特徴・card embedding なし)。

    今回の card_embedding 追加による効果を切り分けるため、このモデルは変更しない
    (``card_ids``/``board_card_ids`` 引数は他モデルと呼び出しを揃えるためだけに受け取り、
    無視する)。
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(
        self, x: torch.Tensor, card_ids: torch.Tensor | None = None, board_card_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


# ---------------------------------------------------------------------------
# データ: 決定点1件 = 元の行(row)インデックス1つ。入力行列はバッチ単位でその場で作る。
# ---------------------------------------------------------------------------
class SplitData:
    """1件 = 1意思決定点。``row_indices`` は state_features_std/option_features_std/
    option_card_ids への元インデックス。chosen_index/weight は row_indices と同じ並びで
    既に切り出し済み。
    """

    def __init__(self, row_indices: np.ndarray, chosen_index: np.ndarray, weight: np.ndarray):
        self.row_indices = row_indices
        self.chosen_index = chosen_index
        self.weight = weight
        self.n = len(row_indices)


def build_batch(
    state_std: np.ndarray | None,
    option_std: np.ndarray,
    option_card_ids: np.ndarray | None,
    row_indices: np.ndarray,
    board_card_ids: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[int]]:
    """バッチ分の入力を連結した1本のテンソルにする。``state_std`` が None ならベースライン
    (option のみ)。``option_card_ids`` が None なら card_id テンソルは作らない(ベースライン用)。

    ``board_card_ids``: T2(design-transformer-representation-2026-08-08.md §5.4)。
    ``(n_rows, BOARD_SLOTS)`` の盤面 card_id 配列(1決定点につき1行)。渡された場合、
    ``state_std`` と同じ「1決定点内の全選択肢へタイル化」を行う(盤面は選択肢に依らず
    決定点で共通の情報のため)。None なら board_id テンソルは作らない(T2 非対応呼び出し用)。

    戻り値は (連結テンソル, card_idテンソル(またはNone), board_idテンソル(またはNone),
    各決定点の選択肢数のリスト)。
    """
    rows = []
    card_id_rows = []
    board_id_rows = []
    sizes = []
    for i in row_indices:
        opt = option_std[i]
        n_i = opt.shape[0]
        if state_std is not None:
            state_tiled = np.broadcast_to(state_std[i], (n_i, state_std.shape[1]))
            rows.append(np.concatenate([state_tiled, opt], axis=1))
        else:
            rows.append(opt)
        sizes.append(n_i)
        if option_card_ids is not None:
            card_id_rows.append(option_card_ids[i])
        if board_card_ids is not None:
            board_tiled = np.broadcast_to(board_card_ids[i], (n_i, board_card_ids.shape[1]))
            board_id_rows.append(board_tiled)
    x = torch.from_numpy(np.concatenate(rows, axis=0).astype(np.float32, copy=False))
    card_ids = None
    if option_card_ids is not None:
        card_ids = torch.from_numpy(np.concatenate(card_id_rows, axis=0).astype(np.int64, copy=False))
    board_ids = None
    if board_card_ids is not None:
        board_ids = torch.from_numpy(np.concatenate(board_id_rows, axis=0).astype(np.int64, copy=False))
    return x, card_ids, board_ids, sizes


# ---------------------------------------------------------------------------
# 損失: リストワイズ softmax 交差エントロピー(複数正解対応)。
#
# 2026-08-12(requirements-kamitsuorochi-2026-08-12.md「What to build」3): 教師が集合
# (chosen_mask で True の位置、|set|>=1)を選んだ決定点では、選ばれた集合の log-softmax を
# 平均してから符号を反転する(-w * mean(logp[c] for c in chosen_set))。これは標準的な
# 「複数正解」listwise形式で、|set|==1(単一選択)のときは既存の
# ``-w * logp[chosen_index]`` に厳密に一致する(mean over 1要素はその要素そのもの)。
# 他の定式化(例: 集合全体を1つの複合選択肢とみなす、集合の要素ごとに独立した二値損失にする)
# も考えられるが、後者は「選択肢間で正規化されたランキング」という既存モデルの前提
# (softmax は決定点内で和が1になる)を崩す。前者は「集合」という可変長の入力を扱うための
# 別モデルが要る。mean-log-softmax はモデル・推論経路を変えずに済む点でここでの制約に合う。
# ---------------------------------------------------------------------------
def batch_weighted_nll(
    model: nn.Module,
    state_std: np.ndarray | None,
    option_std: np.ndarray,
    option_card_ids: np.ndarray | None,
    chosen_mask: np.ndarray,
    weight: np.ndarray,
    batch_row_indices: np.ndarray,
    board_card_ids: np.ndarray | None = None,
) -> torch.Tensor:
    """ミニバッチ内の決定点をまとめて1回の forward で処理し、weight 付き平均 NLL を返す。

    各選択肢は Linear 層のみ(バッチ内の他行に依存する演算、例えば BatchNorm は使わない)を
    通るため、決定点をまたいで連結した1本のテンソルを一括 forward しても、決定点ごとに
    独立に forward した場合と数値的に完全に同一(§4.1 の「選択肢ごとに独立に前向き計算」を
    崩さない)。1決定点ずつ Python ループで forward するより大幅に速いため、この形にする。

    ``chosen_mask``: features.npz の ``chosen_mask``(ragged object配列、絶対行インデックスで
    引く。``option_std``/``option_card_ids`` と同じ引き方)。``weight`` はバッチ位置で既に
    切り出し済みの配列(従来どおり)。

    ``board_card_ids``: T2。渡された場合 ``model(x, card_ids, board_ids)`` を呼ぶ
    (PolicyScorerBoardSet 用)。None なら ``model(x, card_ids)``(既存モデル互換)。
    """
    x, card_ids, board_ids, sizes = build_batch(
        state_std, option_std, option_card_ids, batch_row_indices, board_card_ids=board_card_ids,
    )
    scores = model(x, card_ids, board_ids) if board_ids is not None else model(x, card_ids)  # (sum(sizes),)

    total_loss = torch.zeros((), dtype=scores.dtype)
    total_weight = 0.0
    offset = 0
    for k, i in enumerate(batch_row_indices):
        n_i = sizes[k]
        group_scores = scores[offset : offset + n_i]
        offset += n_i
        logp = torch.log_softmax(group_scores, dim=0)
        chosen_pos = torch.from_numpy(np.nonzero(chosen_mask[i])[0].astype(np.int64))
        w = float(weight[k])
        total_loss = total_loss - w * logp[chosen_pos].mean()
        total_weight += w
    return total_loss / max(total_weight, 1e-12)


def evaluate_split(
    model: nn.Module,
    state_std: np.ndarray | None,
    option_std: np.ndarray,
    option_card_ids: np.ndarray | None,
    chosen_mask: np.ndarray,
    split_data: SplitData,
    batch_size: int = _EVAL_BATCH_SIZE,
    board_card_ids: np.ndarray | None = None,
) -> dict:
    """指定 split 全件(バッチ分割して forward)の指標を計算する。

    2026-08-12: Top-1一致率は単一選択(|chosen_set|==1)の行に限って計算する
    (複数選択行を混ぜると「Top-1」の定義が崩れ、既存の生産値0.6623等との比較可能性が
    失われるため、requirements-kamitsuorochi-2026-08-12.md「What to build」3の指示どおり
    分離する)。複数選択行には precision@k(モデルの上位|chosen_set|件と実際に選ばれた
    集合の重なり率。k=|chosen_set|なので precision=recall=F1と一致する)を別に報告する。
    重み付き平均NLLは新しい複数正解損失(batch_weighted_nllと同じ定義)で全行から計算する
    (|chosen_set|==1の行では従来の値と一致するので、全体の値としては引き続き意味を持つ)。

    ``board_card_ids``: T2。渡された場合 ``model(x, card_ids, board_ids)`` を呼ぶ。
    """
    model.eval()
    n = split_data.n
    if n == 0:
        return {
            "top1_accuracy": None, "n_single": 0,
            "multi_select_precision_at_k": None, "n_multi": 0,
            "mean_weighted_nll": None, "n": 0,
        }
    correct_single = 0
    n_single = 0
    precision_sum = 0.0
    n_multi = 0
    weighted_nll_sum = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_pos = np.arange(start, min(start + batch_size, n))
            batch_row_indices = split_data.row_indices[batch_pos]
            x, card_ids, board_ids, sizes = build_batch(
                state_std, option_std, option_card_ids, batch_row_indices, board_card_ids=board_card_ids,
            )
            scores = model(x, card_ids, board_ids) if board_ids is not None else model(x, card_ids)
            offset = 0
            for k, pos in enumerate(batch_pos):
                n_i = sizes[k]
                group_scores = scores[offset : offset + n_i]
                offset += n_i
                row_idx = int(batch_row_indices[k])
                chosen_pos_np = np.nonzero(chosen_mask[row_idx])[0]
                logp = torch.log_softmax(group_scores, dim=0)
                w = float(split_data.weight[pos])
                chosen_pos_t = torch.from_numpy(chosen_pos_np.astype(np.int64))
                row_nll = -float(logp[chosen_pos_t].mean().item())
                weighted_nll_sum += w * row_nll
                weight_sum += w

                if len(chosen_pos_np) == 1:
                    n_single += 1
                    pred = int(torch.argmax(group_scores).item())
                    if pred == int(chosen_pos_np[0]):
                        correct_single += 1
                else:
                    n_multi += 1
                    k_set = len(chosen_pos_np)
                    topk_idx = set(torch.topk(group_scores, k_set).indices.tolist())
                    precision_sum += len(topk_idx & set(chosen_pos_np.tolist())) / k_set
    return {
        "top1_accuracy": correct_single / n_single if n_single else None,
        "n_single": n_single,
        "multi_select_precision_at_k": precision_sum / n_multi if n_multi else None,
        "n_multi": n_multi,
        "mean_weighted_nll": weighted_nll_sum / max(weight_sum, 1e-12),
        "n": n,
    }


def _metrics_json(m: dict) -> dict:
    """evaluate_split() の戻り値を重みJSON/metrics-out用の辞書にする。既存の
    ``top1_accuracy``/``mean_weighted_nll`` キー名は変えない(過去の全計測値との
    比較可能性・既存ツールとの後方互換のため)。新フィールド(n_single/
    multi_select_precision_at_k/n_multi)を追加するだけ。"""
    return {
        "top1_accuracy": m["top1_accuracy"],
        "n_single": m["n_single"],
        "multi_select_precision_at_k": m["multi_select_precision_at_k"],
        "n_multi": m["n_multi"],
        "mean_weighted_nll": m["mean_weighted_nll"],
    }


def _fmt_metrics(m: dict) -> str:
    """evaluate_split() の戻り値を1行のログ文字列にする(top1/multi_precision がNoneの
    場合(そのsplitに単一選択/複数選択の行が無い場合)も安全に表示する)。"""
    top1 = f"{m['top1_accuracy']:.4f}" if m["top1_accuracy"] is not None else "n/a"
    multi = (
        f"{m['multi_select_precision_at_k']:.4f}" if m["multi_select_precision_at_k"] is not None else "n/a"
    )
    nll = f"{m['mean_weighted_nll']:.4f}" if m["mean_weighted_nll"] is not None else "n/a"
    return (
        f"top1={top1}(n_single={m['n_single']})  "
        f"multi_prec@k={multi}(n_multi={m['n_multi']})  nll={nll}"
    )


def train_model(
    model: nn.Module,
    state_std: np.ndarray | None,
    option_std: np.ndarray,
    option_card_ids: np.ndarray | None,
    chosen_mask: np.ndarray,
    train_data: SplitData,
    val_data: SplitData,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    label: str,
    board_card_ids: np.ndarray | None = None,
) -> tuple[nn.Module, list[float]]:
    """``board_card_ids``: T2。渡された場合 PolicyScorerBoardSet 用に各バッチへ伝播する。

    ``chosen_mask``: features.npz の ``chosen_mask``(絶対行インデックスで引く、ragged object
    配列)。batch_weighted_nll/evaluate_split にそのまま渡す。
    """
    torch.manual_seed(_SEED)
    rng = np.random.default_rng(_SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = train_data.n
    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0
    val_loss_history: list[float] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(n)
        epoch_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_pos = perm[start : start + batch_size]
            batch_row_indices = train_data.row_indices[batch_pos]
            batch_weight = train_data.weight[batch_pos]

            optimizer.zero_grad()
            loss = batch_weighted_nll(
                model, state_std, option_std, option_card_ids, chosen_mask, batch_weight, batch_row_indices,
                board_card_ids=board_card_ids,
            )
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.item())
            n_batches += 1

        val_metrics = evaluate_split(
            model, state_std, option_std, option_card_ids, chosen_mask, val_data, board_card_ids=board_card_ids,
        )
        val_loss = val_metrics["mean_weighted_nll"]
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            val_top1 = val_metrics["top1_accuracy"]
            val_top1_str = f"{val_top1:.4f}" if val_top1 is not None else "n/a"
            val_multi_str = (
                f"{val_metrics['multi_select_precision_at_k']:.4f}"
                if val_metrics["multi_select_precision_at_k"] is not None else "n/a"
            )
            print(
                f"    [{label}] epoch {epoch:3d}  train_loss={epoch_loss_sum / max(n_batches, 1):.4f}"
                f"  val_nll={val_loss:.4f} (best={best_val_loss:.4f}, no_improve={epochs_since_improve})"
                f"  val_top1={val_top1_str}(n_single={val_metrics['n_single']})"
                f"  val_multi_prec@k={val_multi_str}(n_multi={val_metrics['n_multi']})"
            )

        if epochs_since_improve >= patience:
            print(f"    [{label}] 早期終了: epoch {epoch}(val_nll が {patience} epoch 改善せず)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, val_loss_history


def extract_layers_json(model: PolicyScorer | PolicyScorerBoardSet | LinearScorer) -> list[dict]:
    """PyTorch モデルから weight[out][in] 形式の layers を作る(value_net/train.py の
    ``extract_layers_json`` と同じパターン)。

    nn.Linear.weight は既に shape (out_features, in_features) なので転置不要。活性化関数
    (ReLU)は層の間に暗黙的に適用される前提(policy_model.py 側の規約)なので、value_net と
    異なりここでは activation フィールドを持たせず、weight/bias のみの構造にする(仕様通り)。
    """
    layers = []
    for lin in (model.fc1, model.fc2):
        W = lin.weight.detach().numpy().astype(np.float64)
        b = lin.bias.detach().numpy().astype(np.float64)
        layers.append({"weight": W.tolist(), "bias": b.tolist()})
    return layers


def load_init_weights_into_model(model: PolicyScorer | PolicyScorerBoardSet, weights_json: dict) -> None:
    """--init-weights: JSON の layers(fc1/fc2)と card_embedding.table をウォームスタートの
    初期値としてモデルにロードする(extract_layers_json の逆変換)。呼び出し前に main() 側で
    hidden_size/use_board_set/card_id_max/consequence_fields 等が base と一致することを検証
    済みである前提(不一致なら形状が合わずここで RuntimeError になるが、事前検証で先に
    分かりやすいエラーメッセージを出す)。
    """
    layers = weights_json["layers"]
    table = weights_json["card_embedding"]["table"]
    with torch.no_grad():
        expected_fc1_shape = tuple(model.fc1.weight.shape)
        got_fc1_shape = (len(layers[0]["weight"]), len(layers[0]["weight"][0]))
        if got_fc1_shape != expected_fc1_shape:
            raise ValueError(
                f"--init-weights: fc1 の重み形状が一致しません(base={got_fc1_shape} "
                f"現在のモデル={expected_fc1_shape})。use_board_set/hidden_size/"
                "consequence_fields の不一致が疑われます。"
            )
        model.fc1.weight.copy_(torch.tensor(layers[0]["weight"], dtype=model.fc1.weight.dtype))
        model.fc1.bias.copy_(torch.tensor(layers[0]["bias"], dtype=model.fc1.bias.dtype))
        model.fc2.weight.copy_(torch.tensor(layers[1]["weight"], dtype=model.fc2.weight.dtype))
        model.fc2.bias.copy_(torch.tensor(layers[1]["bias"], dtype=model.fc2.bias.dtype))
        expected_embed_shape = tuple(model.embedding.weight.shape)
        got_embed_shape = (len(table), len(table[0]))
        if got_embed_shape != expected_embed_shape:
            raise ValueError(
                f"--init-weights: card_embedding.table の形状が一致しません(base={got_embed_shape} "
                f"現在のモデル={expected_embed_shape})。card_id_max の不一致が疑われます。"
            )
        model.embedding.weight.copy_(torch.tensor(table, dtype=model.embedding.weight.dtype))


# ---------------------------------------------------------------------------
# 自己検証: policy_model.py の PolicyModel._forward と同じ計算を pure Python で独立実装し、
# 書き出した JSON を再読み込みして数値一致を確認する(value_net/train.py の
# pure_python_forward と同じ考え方)。policy_model.py 自体は変更禁止・import もしない
# (独立実装での照合が目的)。
# ---------------------------------------------------------------------------
def _board_pooled_pure_python(board_card_ids: list[int], table: list[list[float]], card_id_max: int) -> list[float]:
    """T2: PolicyScorerBoardSet._board_pooled の pure Python 独立実装(mean/max/sum の順に連結)。
    ``PolicyScorer._board_pooled`` の PyTorch 実装と数値的に一致することを自己検証で確認する。
    """
    embed_dim = len(table[0])
    n = len(board_card_ids)
    vecs = []
    for cid in board_card_ids:
        idx = cid if 0 <= cid <= card_id_max else 0
        vecs.append(table[idx])
    mean_pool = [sum(v[d] for v in vecs) / n for d in range(embed_dim)]
    max_pool = [max(v[d] for v in vecs) for d in range(embed_dim)]
    sum_pool = [sum(v[d] for v in vecs) for d in range(embed_dim)]
    return mean_pool + max_pool + sum_pool


def pure_python_forward(
    raw_state: list[float],
    raw_option: list[float],
    card_id: int,
    weights_json: dict,
    board_card_ids: list[int] | None = None,
) -> float:
    """``board_card_ids`` を渡すと T2(PolicyScorerBoardSet)の pure Python 独立実装になる
    (mean/max/sum pooling を state++option++card_embedding の末尾へ連結)。None なら
    T1(PolicyScorer)と同じ従来の計算。
    """
    std = weights_json["standardization"]
    state_mean, state_std = std["state_mean"], std["state_std"]
    option_mean, option_std = std["option_mean"], std["option_std"]

    h = [
        (raw_state[i] - state_mean[i]) / state_std[i] if state_std[i] else 0.0
        for i in range(len(raw_state))
    ]
    h += [
        (raw_option[i] - option_mean[i]) / option_std[i] if option_std[i] else 0.0
        for i in range(len(raw_option))
    ]

    card_embedding = weights_json["card_embedding"]
    card_id_max = card_embedding["card_id_max"]
    table = card_embedding["table"]
    idx = card_id if 0 <= card_id <= card_id_max else 0
    h += table[idx]

    if board_card_ids is not None:
        h += _board_pooled_pure_python(board_card_ids, table, card_id_max)

    layers = weights_json["layers"]
    n_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        W, b = layer["weight"], layer["bias"]
        z = [b[k] + sum(W[k][i] * h[i] for i in range(len(h))) for k in range(len(W))]
        is_last = layer_idx == n_layers - 1
        h = z if is_last else [v if v > 0.0 else 0.0 for v in z]
    return h[0]


# ---------------------------------------------------------------------------
# Phase1(案C, beyond-bc): outcome / advantage-aware weighting。
# base の sample weight(構成C等)に勝敗由来の係数を掛ける。C1=discount/filter は
# ValueModel 不要、C2=advantage は ValueModel の V(s) を baseline にした MC advantage。
# ---------------------------------------------------------------------------
def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    """value_model._sigmoid と同じオーバーフロー耐性 sigmoid の numpy 版。"""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def _value_probs_np(state_raw: np.ndarray, turn: np.ndarray, value_weights_path: str) -> np.ndarray:
    """features.npz の生 state 特徴(標準化前)から ValueModel の較正済み勝率をベクトルで計算。

    value_model.py の _forward + _calibrate を numpy で複製する(1行ずつの pure-Python 版は
    18万行で遅いため)。数値一致は呼び出し側で ValueModel と突き合わせて検証する。
    """
    from ptcg_ai.learning.value_model import _turn_band_of  # noqa: E402

    with open(value_weights_path, encoding="utf-8") as fh:
        vw = json.load(fh)
    mean = np.asarray(vw["standardization"]["mean"], dtype=np.float64)
    std = np.asarray(vw["standardization"]["std"], dtype=np.float64)
    # value_model: (f-mean)/std if std else 0.0(std==0 の特徴は 0 に潰す)。
    safe_std = np.where(std == 0.0, 1.0, std)
    h = np.where(std == 0.0, 0.0, (state_raw.astype(np.float64) - mean) / safe_std)
    for layer in vw["layers"]:
        W = np.asarray(layer["W"], dtype=np.float64)  # [out][in]
        b = np.asarray(layer["b"], dtype=np.float64)
        z = h @ W.T + b
        act = layer["activation"]
        if act == "relu":
            h = np.maximum(z, 0.0)
        elif act == "sigmoid":
            h = _sigmoid_np(z)
        else:
            raise ValueError(f"未知の activation: {act}")
    p_raw = h[:, 0]
    buckets = {
        bk["band"]: float(bk["temperature"])
        for bk in vw.get("meta", {}).get("calibration", {}).get("buckets", [])
    }
    eps = 1e-12
    p = np.clip(p_raw, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    temps = np.ones(len(turn), dtype=np.float64)
    bands = np.array([_turn_band_of(int(t)) for t in turn])
    for band, temp in buckets.items():
        temps[bands == band] = temp
    return _sigmoid_np(logit / temps)


def compute_outcome_factor(args, won: np.ndarray, state_raw: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """outcome-weighting の係数(base weight に掛ける)を全行ぶん返す。won=-1(不明)は 1.0。

    - discount: won=1 -> 1.0 / won=0 -> loss_discount γ / won=-1 -> 1.0
    - filter:   won=1 -> 1.0 / won=0 -> 0.0(= γ=0。後段で train/val の 0 重み行は除外)
    - advantage: A=won-V(s), factor=exp(clip(A/β, -c, c))。won=-1 -> 1.0(A=0 扱い)
    """
    mode = args.outcome_weighting
    factor = np.ones(len(won), dtype=np.float64)
    known = won >= 0
    if mode in ("discount", "filter"):
        gamma = 0.0 if mode == "filter" else float(args.loss_discount)
        factor[known & (won == 0)] = gamma
    elif mode == "advantage":
        v = _value_probs_np(state_raw, turn, args.value_weights)
        # 数値一致検証(ValueModel と突き合わせ、val split から数十件)。
        from ptcg_ai.learning.value_model import ValueModel
        vm = ValueModel(args.value_weights)
        if not vm.is_ready:
            raise ValueError(f"value_weights を読めません: {args.value_weights}")
        rng = np.random.default_rng(_SEED)
        check_idx = rng.choice(len(won), size=min(50, len(won)), replace=False)
        max_err = 0.0
        for i in check_idx:
            got = vm.predict_win_prob_from_features(state_raw[int(i)].astype(np.float64).tolist(), int(turn[int(i)]))
            max_err = max(max_err, abs(got - float(v[int(i)])))
        print(f"  V(s) numpy vs ValueModel 最大誤差={max_err:.3e}(許容 1e-6)")
        if max_err > 1e-6:
            raise ValueError(f"V(s) の numpy 実装が value_model と一致しません(誤差 {max_err:.3e})")
        adv = np.where(known, won.astype(np.float64) - v, 0.0)
        beta = float(args.adv_beta)
        clip = float(args.adv_clip)
        factor = np.exp(np.clip(adv / beta, -clip, clip))
    else:
        raise ValueError(f"未知の outcome_weighting: {mode}")
    return factor


def main() -> None:
    global _SEED
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(_DEFAULT_FEATURES))
    parser.add_argument("--max-epochs", type=int, default=_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="先頭N件のみ使う(動作確認モード)")
    parser.add_argument(
        "--out-weights", default=str(_WEIGHTS_OUT_PATH),
        help="重みJSON書き出し先(既定: 本番パス。実験時は別パスを明示指定すること)",
    )
    parser.add_argument(
        "--ablate-features", default=None,
        help="カンマ区切りの特徴名を学習前に 0 で潰す(次元は変えない)。エンコーダを変更せずに『その特徴が無い版』の対照群を作るために使う。状態側は FEATURE_NAMES の完全一致、選択肢側は OPTION_FEATURE_NAMES の完全一致で探す。接尾辞指定(例 retreat_cost)なら、状態側の全スロット(self_active_retreat_cost 等)と選択肢側(target_pokemon_retreat_cost)をまとめて潰す。"
        "接頭辞指定(例 self_board_card_slot)なら、self_board_card_slot_0〜_55 のような連番スロット群を"
        "一括で潰す(T1の card_id カウント特徴群のような『共通接頭辞+連番』の対照群作成用。"
        "design-transformer-representation-2026-08-08.md §7.1/§7.2)。例: "
        "--ablate-features self_board_card_slot,self_discard_card_slot,opp_board_card_slot,opp_discard_card_slot",
    )
    parser.add_argument(
        "--seed", type=int, default=_SEED,
        help="学習の乱数シード(既定 42)。同一設定で複数回まわして run 間分散を測るために使う。行動種別ごとの Top-1 は run 間で ±5pt 程度振れるため、種別単位の比較を主張するなら反復が要る。",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=_HIDDEN_SIZE,
        help="本命MLPの隠れ層サイズ(既定: 32 = 現行本番と同一。モデル容量ablation用。"
        "model-capacity-ablation-implementation-plan.md。hidden以外は一切変えないこと)。",
    )
    parser.add_argument(
        "--metrics-out", default=None,
        help="offline評価指標(train/val/test NLL・Top-1、param数、gap)のJSON書き出し先"
        "(既定: 書き出さない)。容量ablationの記録用。",
    )
    # --- Phase1(案C, beyond-bc): outcome / advantage-aware weighting ---
    parser.add_argument(
        "--outcome-weighting", choices=["none", "discount", "filter", "advantage"], default="none",
        help="勝敗(features.npz の won)で sample weight を再重み付けする(既定 none = 現行と完全に同一)。"
        "discount=負け試合を γ 倍 / filter=勝ち試合のみ / advantage=exp((won-V(s))/β)。"
        "phase1-outcome-aware-implementation-plan.md。none 以外は features.npz に won が必要"
        "(build_features.py --with-outcome)。",
    )
    parser.add_argument("--loss-discount", type=float, default=0.5, help="discount 用: 負け試合の重み係数 γ∈[0,1]")
    parser.add_argument("--adv-beta", type=float, default=1.0, help="advantage 用: 温度 β")
    parser.add_argument("--adv-clip", type=float, default=3.0, help="advantage 用: A/β のクリップ幅 c")
    parser.add_argument(
        "--value-weights",
        default=str(_SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "learning" / "value_weights.json"),
        help="advantage 用の ValueModel 重み(既定: 本番 value_weights.json、読み取り専用)",
    )
    parser.add_argument(
        "--use-board-set", action="store_true",
        help="T2(design-transformer-representation-2026-08-08.md §5.4段階1、Deep Sets): "
        "本命モデルを PolicyScorer から PolicyScorerBoardSet に切り替える。盤面12スロット"
        "(encoder.BOARD_SLOTS)の card_id を card_embedding + mean/max/sum pooling で集約した"
        "表現を追加入力にする。features.npz に build_features.py が生成した board_card_ids "
        "配列が必要(無ければエラー)。既定は付けない(既存のPolicyScorerのまま、T1のみの構成)。",
    )
    parser.add_argument(
        "--consequence-fields", default=None,
        help="Tier3 Stage3c: features.npz の consequence_features(build_features.py "
        "--with-consequence-features で生成)からこのカンマ区切りの特徴名だけを選び、"
        "option特徴の末尾に連結して学習する(例: 'opp_hp_loss,opp_energy_removed,"
        "opp_special_energy_removed' = Experiment B)。既定は None(consequence特徴を"
        "使わない。既存の挙動と完全に同一)。有効な特徴名は "
        "ptcg_ai.learning.encoder.CONSEQUENCE_FEATURE_NAMES 参照。",
    )
    # --- ファインチューン(warm start)用フラグ。requirements-kamitsuorochi-2026-08-12.md §10-1:
    # クラスタごとにゼロから学習するとデータ不足で初期値の運しか測れないため、共有ベースの
    # 重みを初期値にした低学習率の追加学習にする。3フラグとも既定値では従来と完全に同一。---
    parser.add_argument(
        "--init-weights", default=None,
        help="既存の policy_weights JSON を読み、本命モデル(fc1/fc2/embedding)の重みの初期値に"
        "する(ウォームスタート)。標準化パラメータ(state_mean/state_std/option_mean/"
        "option_std)は部分集合から再計算せず、このJSONの値をそのまま使い、出力JSONにも"
        "そのまま書く(部分集合で再計算すると入力分布がbaseと変わり、warm startした重みが"
        "最初の1ステップで壊れるため)。hand_card_vocab/opponent_card_vocab/"
        "consequence_fields/use_board_set/ablated_features/hidden_size/card_id_max が"
        "base(このJSON)と一致しない場合はエラーで停止する(黙って続行しない)。"
        "既定 None(従来どおりランダム初期化・train splitから標準化を計算)。",
    )
    parser.add_argument(
        "--lr", type=float, default=_LR,
        help=f"optimizerの学習率(既定 {_LR} = 従来ハードコード値と同一)。"
        "ファインチューンでは1/5〜1/10に落として使う想定。",
    )
    parser.add_argument(
        "--row-mask", default=None,
        help="bool配列の.npyファイル。Trueの行だけを学習・評価に使う。train splitだけでなく"
        "train/val/test全splitに適用する(このクラスタでの性能をvalで早期終了判定し、"
        "testで報告するため)。長さはfeatures.npzの全行数と一致していなければエラー。"
        "マスク適用後にtrain/val/testのいずれかが0件になった場合もエラーで停止する。"
        "既定 None(全行を使う。従来と完全に同一)。",
    )
    args = parser.parse_args()
    # train_model / 自己検証サンプル抽出は呼び出し時に _SEED を読むので、
    # ここで書き換えれば全体のシードが揃う(既定値のままなら従来と完全に同一)。
    _SEED = int(args.seed)
    weights_out_path = Path(args.out_weights)
    consequence_fields: list[str] = (
        [s.strip() for s in args.consequence_fields.split(",") if s.strip()]
        if args.consequence_fields else []
    )
    if consequence_fields:
        unknown = [f for f in consequence_fields if f not in CONSEQUENCE_FEATURE_NAMES]
        if unknown:
            raise ValueError(f"未知のconsequence特徴名: {unknown} (有効: {CONSEQUENCE_FEATURE_NAMES})")
        print(f"consequence特徴を使用: {consequence_fields}")

    # --- --init-weights: baseのJSONを先に読んでおく(md5・meta比較・標準化流用・重みロードで使う)。
    init_weights_json: dict | None = None
    init_weights_md5: str | None = None
    if args.init_weights:
        init_weights_path = Path(args.init_weights)
        init_weights_bytes = init_weights_path.read_bytes()
        init_weights_md5 = hashlib.md5(init_weights_bytes).hexdigest()
        init_weights_json = json.loads(init_weights_bytes.decode("utf-8"))
        print(f"--init-weights: {init_weights_path} (md5={init_weights_md5}) をウォームスタートの初期値として使用")

    row_mask: np.ndarray | None = None
    if args.row_mask:
        row_mask_path = Path(args.row_mask)
        row_mask = np.load(row_mask_path)
        if row_mask.dtype != np.bool_:
            row_mask = row_mask.astype(bool)
        print(f"--row-mask: {row_mask_path} (True={int(row_mask.sum())}/{len(row_mask)})")

    features_path = Path(args.features)
    print(f"features.npz を読み込み: {features_path}")
    data = np.load(features_path, allow_pickle=True)
    state_features = data["state_features"].astype(np.float32)
    option_features = data["option_features"]  # object array, 各要素 (n_i, OPTION_FEATURE_COUNT)
    option_card_ids = data["option_card_ids"]  # object array, 各要素 (n_i,) 生の card_id(int)
    card_id_max = int(data["card_id_max"].item())

    if row_mask is not None and len(row_mask) != len(state_features):
        raise ValueError(
            f"--row-mask の長さ({len(row_mask)})が features.npz の行数({len(state_features)})と"
            "一致しません。"
        )

    # T2: 盤面12スロットの card_id(固定長、build_features.py が常に保存する)。
    board_card_ids: np.ndarray | None = None
    if args.use_board_set:
        if "board_card_ids" not in data:
            raise ValueError(
                "--use-board-set が指定されましたが features.npz に board_card_ids がありません。"
                "build_features.py を作り直したものを使ってください(新しい build_features.py は "
                "常に board_card_ids を保存します)。"
            )
        board_card_ids = data["board_card_ids"].astype(np.int64)
        print(f"board_card_ids shape={board_card_ids.shape}(--use-board-set 有効、Deep Sets Encoder使用)")
    chosen_index = data["chosen_index"].astype(np.int64)
    # 2026-08-12: 複数選択対応(requirements-kamitsuorochi-2026-08-12.md「What to build」3)。
    # chosen_mask が無い旧形式の features.npz(他アーキタイプの過去ビルド)は chosen_index から
    # 単一選択のマスクを作って後方互換にする。
    if "chosen_mask" in data:
        chosen_mask = data["chosen_mask"]
    else:
        print(
            "chosen_mask が features.npz にありません(旧形式)。chosen_index から単一選択の"
            "マスクを生成します(全行 |chosen_set|==1 として扱う)。"
        )
        chosen_mask = np.empty(len(chosen_index), dtype=object)
        for i in range(len(chosen_index)):
            m = np.zeros(option_features[i].shape[0], dtype=bool)
            m[chosen_index[i]] = True
            chosen_mask[i] = m
    split = data["split"].astype(np.int64)
    weight = data["weight"].astype(np.float64)
    select_type = data["select_type"].astype(np.int64)
    # C2 第一段階(roadmap-2026-08-05.md): build_features.py --deck-csv で作った語彙。
    # 無ければ空配列(古い features.npz にも "hand_card_vocab" キー自体が無い場合がある)。
    hand_card_vocab: list[int] = (
        [int(v) for v in data["hand_card_vocab"].tolist()] if "hand_card_vocab" in data else []
    )
    if hand_card_vocab:
        print(f"hand_card_vocab: {len(hand_card_vocab)}種 {hand_card_vocab}")
    else:
        print("hand_card_vocab: なし(build_features.py --deck-csv 未指定。手札card_idカウント特徴は全0)")

    # T1残り(design-transformer-representation-2026-08-08.md §9論点11): build_features.py
    # --opponent-vocab-dir で作った語彙。無ければ空配列(古い features.npz には
    # "opponent_card_vocab" キー自体が無い場合がある)。
    opponent_card_vocab: list[int] = (
        [int(v) for v in data["opponent_card_vocab"].tolist()] if "opponent_card_vocab" in data else []
    )
    if opponent_card_vocab:
        print(f"opponent_card_vocab: {len(opponent_card_vocab)}種 {opponent_card_vocab}")
    else:
        print(
            "opponent_card_vocab: なし(build_features.py --opponent-vocab-dir 未指定。"
            "相手card_idカウント特徴は全0)"
        )

    # ablate_st_idx/ablate_op_idx は標準化計算の直後(std<1e-6クリップ後)にも使う
    # (measurement-plan-2026-08-04.md §3.1 対策、下記参照)。if 文の外でも参照できるよう
    # ここで初期化しておく。
    ablate_st_idx: list[int] = []
    ablate_op_idx: list[int] = []
    if args.ablate_features:
        names = [t.strip() for t in args.ablate_features.split(",") if t.strip()]
        st_idx, op_idx = [], []
        for nm in names:
            # 完全一致 / 接尾辞一致(例 retreat_cost -> self_active_retreat_cost) / 接頭辞一致
            # (例 self_board_card_slot -> self_board_card_slot_0.._55、T1のcard_idカウント
            # 特徴群のような「共通接頭辞+連番」を一括で潰す。design書 §7.1/§7.2)。
            hit_s = [
                i for i, f in enumerate(FEATURE_NAMES)
                if f == nm or f.endswith("_" + nm) or f.startswith(nm + "_")
            ]
            hit_o = [
                i for i, f in enumerate(OPTION_FEATURE_NAMES)
                if f == nm or f.endswith("_" + nm) or f.startswith(nm + "_")
            ]
            if not hit_s and not hit_o:
                raise ValueError(f"--ablate-features: 一致する特徴名がありません: {nm}")
            st_idx += hit_s
            op_idx += hit_o
        if st_idx:
            state_features[:, st_idx] = 0.0
        if op_idx:
            # option_features は決定点ごとに行数が違う object 配列なので個別に潰す。
            for i in range(len(option_features)):
                option_features[i][:, op_idx] = 0.0
        print(f"  ablate: 状態側 {len(st_idx)} 次元 / 選択肢側 {len(op_idx)} 次元を 0 で潰した {names}")
        ablate_st_idx, ablate_op_idx = st_idx, op_idx

    won = None
    turn = None
    if args.outcome_weighting != "none":
        if "won" not in data:
            raise ValueError(
                "--outcome-weighting が指定されましたが features.npz に won がありません"
                "(build_features.py を --with-outcome で再実行してください)"
            )
        won = data["won"].astype(np.int64)
        turn = data["turn"].astype(np.int64)

    assert state_features.shape[1] == BASE_FEATURE_COUNT
    assert option_features[0].shape[1] == OPTION_FEATURE_COUNT
    print(f"card_id_max={card_id_max}  embedding table size={card_id_max + 1} x {_EMBED_DIM}")

    effective_option_feature_count = OPTION_FEATURE_COUNT
    if consequence_fields:
        if "consequence_features" not in data:
            raise ValueError(
                "--consequence-fields が指定されましたが features.npz に "
                "consequence_features がありません(build_features.py を "
                "--with-consequence-features で再実行してください)"
            )
        consequence_features_all = data["consequence_features"]
        field_indices = [CONSEQUENCE_FEATURE_NAMES.index(f) for f in consequence_fields]
        # option_features の末尾に選択したconsequence列だけを連結する(この時点で連結
        # しておけば、以降の標準化・self-check・PolicyModel._forward はすべて既存の
        # option_features 用ロジックのまま流用でき、consequence専用の分岐が不要になる)。
        option_features = np.array(
            [
                np.concatenate([option_features[i], consequence_features_all[i][:, field_indices]], axis=1)
                for i in range(len(option_features))
            ],
            dtype=object,
        )
        effective_option_feature_count = OPTION_FEATURE_COUNT + len(consequence_fields)
        print(f"option特徴を{OPTION_FEATURE_COUNT}->{effective_option_feature_count}次元に拡張")

    n_total = len(state_features)
    if args.limit is not None:
        n_total = min(args.limit, n_total)
        state_features = state_features[:n_total]
        option_features = option_features[:n_total]
        option_card_ids = option_card_ids[:n_total]
        chosen_index = chosen_index[:n_total]
        chosen_mask = chosen_mask[:n_total]
        split = split[:n_total]
        weight = weight[:n_total]
        select_type = select_type[:n_total]
        if board_card_ids is not None:
            board_card_ids = board_card_ids[:n_total]
        if won is not None:
            won = won[:n_total]
            turn = turn[:n_total]
        if row_mask is not None:
            row_mask = row_mask[:n_total]
        print(f"動作確認モード: 先頭 {n_total} 件のみ使用")

    # --- Phase1(案C): outcome / advantage 係数を base weight に掛ける(state_features は
    # まだ生=標準化前なので advantage の V(s) 計算に使える。標準化・del より前で行う)。---
    outcome_meta: dict | None = None
    if args.outcome_weighting != "none":
        factor = compute_outcome_factor(args, won, state_features, turn)
        n_known = int((won >= 0).sum())
        weight = weight * factor
        outcome_meta = {
            "mode": args.outcome_weighting,
            "loss_discount": float(args.loss_discount) if args.outcome_weighting == "discount" else None,
            "adv_beta": float(args.adv_beta) if args.outcome_weighting == "advantage" else None,
            "adv_clip": float(args.adv_clip) if args.outcome_weighting == "advantage" else None,
            "value_weights": args.value_weights if args.outcome_weighting == "advantage" else None,
            "n_outcome_known": n_known,
            "n_outcome_unknown": int(len(won) - n_known),
            "n_win": int((won == 1).sum()),
            "n_loss": int((won == 0).sum()),
        }
        print(
            f"outcome-weighting={args.outcome_weighting} 適用: 既知={n_known} "
            f"(勝={outcome_meta['n_win']} 負={outcome_meta['n_loss']}) "
            f"factor範囲=[{factor.min():.3f}, {factor.max():.3f}] 平均={factor.mean():.3f}"
        )

    train_mask = split == _TRAIN
    val_mask = split == _VAL
    test_mask = split == _TEST

    # --row-mask: train/val/test 全splitに適用する(§10-1: このクラスタでの性能をvalで
    # 早期終了判定し、testで報告するため。train splitだけに絞ると val/test にクラスタ外の
    # 決定点が残ってしまい、早期終了・報告の両方がクラスタの性能を測れなくなる)。
    if row_mask is not None:
        n_train_pre, n_val_pre, n_test_pre = int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())
        train_mask = train_mask & row_mask
        val_mask = val_mask & row_mask
        test_mask = test_mask & row_mask
        print(
            f"--row-mask 適用後: train {n_train_pre}->{int(train_mask.sum())}"
            f"  val {n_val_pre}->{int(val_mask.sum())}"
            f"  test {n_test_pre}->{int(test_mask.sum())}"
        )
        if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
            raise ValueError(
                "--row-mask 適用後に train/val/test のいずれかが0件になりました"
                f"(train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())})。"
            )

    n_train, n_val, n_test = int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())
    print(f"train={n_train} val={n_val} test={n_test} (全{n_total}件)")

    if args.init_weights:
        # --init-weights: base の meta と現在の設定が一致することを確認する(黙って続行しない。
        # 「次元は合うが語彙が違う」類の不整合が過去に実際の事故になっているため)。
        base_meta = init_weights_json.get("meta", {})
        base_card_id_max = init_weights_json.get("card_embedding", {}).get("card_id_max")
        base_ablate = base_meta.get("ablated_features")
        base_ablate_spec = base_ablate["spec"] if base_ablate else None
        mismatches = []
        if base_card_id_max != card_id_max:
            mismatches.append(f"card_id_max: base={base_card_id_max} 現在={card_id_max}")
        if base_meta.get("hand_card_vocab") != hand_card_vocab:
            mismatches.append(f"hand_card_vocab: base={base_meta.get('hand_card_vocab')} 現在={hand_card_vocab}")
        if base_meta.get("opponent_card_vocab") != opponent_card_vocab:
            mismatches.append(
                f"opponent_card_vocab: base={base_meta.get('opponent_card_vocab')} 現在={opponent_card_vocab}"
            )
        if base_meta.get("consequence_fields") != consequence_fields:
            mismatches.append(
                f"consequence_fields: base={base_meta.get('consequence_fields')} 現在={consequence_fields}"
            )
        if bool(base_meta.get("use_board_set")) != bool(args.use_board_set):
            mismatches.append(f"use_board_set: base={base_meta.get('use_board_set')} 現在={args.use_board_set}")
        if base_ablate_spec != args.ablate_features:
            mismatches.append(f"ablated_features: base={base_ablate_spec} 現在={args.ablate_features}")
        if base_meta.get("hidden_size") != args.hidden_size:
            mismatches.append(f"hidden_size: base={base_meta.get('hidden_size')} 現在={args.hidden_size}")
        if mismatches:
            raise ValueError(
                "--init-weights: base と現在の設定が一致しません(warm start では標準化・"
                "埋め込み語彙・特徴構成が完全一致していないと不正な学習になるため停止します): "
                + " / ".join(mismatches)
            )

        # 標準化パラメータは部分集合から再計算せず、base の値をそのまま使う(部分集合で
        # 再計算すると入力分布が base と変わり、warm start した重みが最初の1ステップで
        # 壊れるため。requirements-kamitsuorochi-2026-08-12.md §10)。
        std_block = init_weights_json["standardization"]
        state_mean = np.asarray(std_block["state_mean"], dtype=np.float32)
        state_std = np.asarray(std_block["state_std"], dtype=np.float32)
        option_mean = np.asarray(std_block["option_mean"], dtype=np.float32)
        option_std_ = np.asarray(std_block["option_std"], dtype=np.float32)
        if len(state_mean) != BASE_FEATURE_COUNT or len(state_std) != BASE_FEATURE_COUNT:
            raise ValueError(
                f"--init-weights: state標準化の次元({len(state_mean)})が"
                f"BASE_FEATURE_COUNT({BASE_FEATURE_COUNT})と一致しません。"
            )
        if len(option_mean) != effective_option_feature_count or len(option_std_) != effective_option_feature_count:
            raise ValueError(
                f"--init-weights: option標準化の次元({len(option_mean)})が"
                f"現在のoption特徴次元({effective_option_feature_count})と一致しません。"
            )
        print(
            "--init-weights: 標準化パラメータ(state/option の mean/std)は部分集合から"
            f"再計算せず、{args.init_weights} の値をそのまま使用します"
        )
    else:
        # --- 標準化(train split のみから計算) ---
        state_mean = state_features[train_mask].mean(axis=0)
        state_std = state_features[train_mask].std(axis=0)
        state_std = np.where(state_std < 1e-6, 1e-6, state_std).astype(np.float32)
        state_mean = state_mean.astype(np.float32)

        train_options_stacked = np.concatenate(
            [option_features[i] for i in np.where(train_mask)[0]], axis=0
        )
        option_mean = train_options_stacked.mean(axis=0)
        option_std_ = train_options_stacked.std(axis=0)
        option_std_ = np.where(option_std_ < 1e-6, 1e-6, option_std_).astype(np.float32)
        option_mean = option_mean.astype(np.float32)
        print(f"option特徴の標準化元行列 shape={train_options_stacked.shape}")

        # measurement-plan-2026-08-04.md §3.1: --ablate-features で潰した列は学習時ずっと0
        # なので、上の std<1e-6 クリップで std=1e-6 になる(0 にはならない)。
        # policy_model._forward は `(x - mean) / std if std else 0.0` という「std が偽(=0)なら
        # 0扱い」の分岐なので、std=1e-6 のままだと本番推論(および eval_diagnostics.py 等の
        # ミラー戦評価)で実際の非ゼロ値が 1e-6 で割られ、桁違いの異常値が入って対照群の判断が
        # 数値的に壊れる(過去に「実験群 +26.8pt で圧勝」という真逆の結果を出した実例と同じ罠)。
        # ablate 対象と分かっている列は、クリップ後にここで明示的に 0 へ戻す
        # (`if std else 0.0` の分岐を確実に踏ませる)。
        if ablate_st_idx:
            state_std[ablate_st_idx] = 0.0
            state_mean[ablate_st_idx] = 0.0
        if ablate_op_idx:
            option_std_[ablate_op_idx] = 0.0
            option_mean[ablate_op_idx] = 0.0
        if ablate_st_idx or ablate_op_idx:
            print(
                f"  ablate対策: 状態側 {len(ablate_st_idx)} / 選択肢側 {len(ablate_op_idx)} 次元の "
                "std を 0 に修正(std<1e-6クリップの上書き。measurement-plan-2026-08-04.md §3.1)"
            )
        del train_options_stacked

    # --- 自己検証用サンプルを、生特徴を破棄する前に確保しておく(val split からランダム抽出)。---
    val_idx_for_check = np.where(val_mask)[0]
    check_rng = random.Random(_SEED)
    self_check_rows = check_rng.sample(
        list(val_idx_for_check), min(_N_SELF_CHECK, len(val_idx_for_check))
    )
    self_check_samples = []
    for row in self_check_rows:
        n_opts = option_features[row].shape[0]
        opt_idx = int(chosen_index[row]) if 0 <= int(chosen_index[row]) < n_opts else 0
        sample = {
            "row": int(row),
            "opt_idx": opt_idx,
            "raw_state": state_features[row].astype(np.float64).tolist(),
            "raw_option": option_features[row][opt_idx].astype(np.float64).tolist(),
            "card_id": int(option_card_ids[row][opt_idx]),
        }
        if board_card_ids is not None:
            sample["board_card_ids"] = [int(v) for v in board_card_ids[row]]
        self_check_samples.append(sample)

    # 標準化済み特徴(state/option とも float32 のまま、split 全体で使い回す)。
    # ablate対策で std=0 にした列は、素の除算だと 0/0 で NaN になる(inf ではなく NaN、
    # 分子の (x - mean) も ablate 列は常に x=0・mean=0 で 0 になるため)。std=0 の列は
    # 結果を 0 に固定する(policy_model._forward の `if std else 0.0` と同じ挙動)。
    _safe_state_std = np.where(state_std == 0.0, 1.0, state_std)
    state_features_std = np.where(
        state_std == 0.0, 0.0, (state_features - state_mean) / _safe_state_std
    ).astype(np.float32, copy=False)
    del state_features
    _safe_option_std = np.where(option_std_ == 0.0, 1.0, option_std_)
    option_features_std = np.empty(n_total, dtype=object)
    for i in range(n_total):
        option_features_std[i] = np.where(
            option_std_ == 0.0, 0.0, (option_features[i] - option_mean) / _safe_option_std
        ).astype(np.float32, copy=False)
    del option_features
    # option_card_ids は標準化しない(埋め込みの生インデックス)。int64 に揃えるだけ。
    for i in range(n_total):
        option_card_ids[i] = option_card_ids[i].astype(np.int64, copy=False)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    # filter モード(および任意の 0 重み)では train/val から重み 0 の行を除く(全 0 の
    # ミニバッチで total_weight≈0 になり不安定化するのを避ける)。none/discount>0/advantage
    # では 0 重みが生じないため影響しない。test は報告用に全件保持する。
    if args.outcome_weighting != "none":
        n_tr0, n_va0 = len(train_idx), len(val_idx)
        train_idx = train_idx[weight[train_idx] > 0.0]
        val_idx = val_idx[weight[val_idx] > 0.0]
        if len(train_idx) != n_tr0 or len(val_idx) != n_va0:
            print(f"  0重み行を除外: train {n_tr0}->{len(train_idx)}  val {n_va0}->{len(val_idx)}")

    train_main = SplitData(train_idx, chosen_index[train_idx], weight[train_idx])
    val_main = SplitData(val_idx, chosen_index[val_idx], weight[val_idx])
    test_main = SplitData(test_idx, chosen_index[test_idx], weight[test_idx])

    # --- ベースライン: 隠れ層なし線形モデル(選択肢特徴のみ、card embedding なし。変更なし) ---
    # consequence_fields 指定時は option_features_std に既に連結済みなので、ベースラインの
    # 入力次元も合わせて拡張する(そうしないと次元不一致になる)。
    print("\n=== ベースライン: 線形モデル(選択肢特徴のみ、状態特徴・card embeddingなし) ===")
    t0 = time.time()
    baseline_model = LinearScorer(effective_option_feature_count)
    baseline_model, _ = train_model(
        baseline_model, None, option_features_std, None, chosen_mask, train_main, val_main,
        args.max_epochs, args.patience, args.batch_size, args.lr, label="baseline",
    )
    print(f"  学習完了 ({time.time() - t0:.1f}s)")
    baseline_val = evaluate_split(baseline_model, None, option_features_std, None, chosen_mask, val_main)
    baseline_test = evaluate_split(baseline_model, None, option_features_std, None, chosen_mask, test_main)
    print(f"  val : {_fmt_metrics(baseline_val)}")
    print(f"  test: {_fmt_metrics(baseline_test)}")

    # --- 本命: 小型 MLP(state ++ option ++ card_embedding[ ++ T2: board Deep Sets]) ---
    hidden_size = args.hidden_size
    model_label = "PolicyScorerBoardSet(T2 Deep Sets)" if args.use_board_set else "PolicyScorer(T1)"
    print(f"\n=== 本命: MLP({model_label}, embedding(card_id) ++ Linear(in,{hidden_size}) -> ReLU -> Linear({hidden_size},1)) ===")
    t0 = time.time()
    in_dim_main = BASE_FEATURE_COUNT + effective_option_feature_count
    if args.use_board_set:
        main_model = PolicyScorerBoardSet(in_dim_main, card_id_max, _EMBED_DIM, hidden_size)
    else:
        main_model = PolicyScorer(in_dim_main, card_id_max, _EMBED_DIM, hidden_size)
    if args.init_weights:
        load_init_weights_into_model(main_model, init_weights_json)
        print(f"--init-weights: fc1/fc2/embedding を {args.init_weights} からウォームスタートの初期値としてロードしました")
    main_model, val_loss_history = train_model(
        main_model, state_features_std, option_features_std, option_card_ids, chosen_mask, train_main, val_main,
        args.max_epochs, args.patience, args.batch_size, args.lr, label="main",
        board_card_ids=board_card_ids,
    )
    print(f"  学習完了 ({time.time() - t0:.1f}s, {len(val_loss_history)} epochs)")

    # train/val/test の全 split で指標を取る(容量ablation の overfitting 確認用。
    # train split の評価は本ファイルでは従来省いていたが、train-val gap を見るために追加する)。
    main_train = evaluate_split(
        main_model, state_features_std, option_features_std, option_card_ids, chosen_mask, train_main,
        board_card_ids=board_card_ids,
    )
    main_val = evaluate_split(
        main_model, state_features_std, option_features_std, option_card_ids, chosen_mask, val_main,
        board_card_ids=board_card_ids,
    )
    main_test = evaluate_split(
        main_model, state_features_std, option_features_std, option_card_ids, chosen_mask, test_main,
        board_card_ids=board_card_ids,
    )
    print(f"  train: {_fmt_metrics(main_train)}")
    print(f"  val  : {_fmt_metrics(main_val)}")
    print(f"  test : {_fmt_metrics(main_test)}")
    # 2026-08-12: top1_accuracy は単一選択のみの値なので None になりうる(そのsplitに単一選択
    # 行が無い場合)。gap は両方Noneでない場合のみ計算する(既存の docs の値との比較可能性は
    # 単一選択のみの run では従来どおり保たれる)。
    if main_train["top1_accuracy"] is not None and main_val["top1_accuracy"] is not None:
        train_val_top1_gap = main_train["top1_accuracy"] - main_val["top1_accuracy"]
    else:
        train_val_top1_gap = None
    train_val_nll_gap = main_val["mean_weighted_nll"] - main_train["mean_weighted_nll"]
    top1_gap_str = f"{train_val_top1_gap:+.4f}" if train_val_top1_gap is not None else "n/a"
    print(f"  train-val gap: top1={top1_gap_str}  nll={train_val_nll_gap:+.4f}(overfitting指標)")

    # パラメータ数(fc1/fc2/embedding/total)。value/latency の記録用。
    param_counts = {
        "fc1": main_model.fc1.weight.numel() + main_model.fc1.bias.numel(),
        "fc2": main_model.fc2.weight.numel() + main_model.fc2.bias.numel(),
        "embedding": main_model.embedding.weight.numel(),
    }
    param_counts["mlp"] = param_counts["fc1"] + param_counts["fc2"]
    param_counts["total"] = param_counts["mlp"] + param_counts["embedding"]
    print(f"  param_counts: {param_counts}")

    print("\n=== 本命 vs ベースライン(val split) ===")
    print(f"  ベースライン(option only): {_fmt_metrics(baseline_val)}")
    print(f"  本命(state ++ option ++ embed): {_fmt_metrics(main_val)}")

    # --- select_type別(本命モデル, test split) ---
    print("\n=== select_type別 指標(本命モデル, test split) ===")
    st_test = select_type[test_idx]
    for st in sorted(set(st_test.tolist())):
        mask = st_test == st
        pos = np.where(mask)[0]
        sub = SplitData(test_main.row_indices[pos], test_main.chosen_index[pos], test_main.weight[pos])
        m = evaluate_split(
            main_model, state_features_std, option_features_std, option_card_ids, chosen_mask, sub,
            board_card_ids=board_card_ids,
        )
        print(f"  select_type={st:2d}  n={m['n']:6d}  {_fmt_metrics(m)}")

    # --- JSON エクスポート ---
    layers_json = extract_layers_json(main_model)
    embedding_table = main_model.embedding.weight.detach().numpy().astype(np.float64)
    weights_json = {
        "meta": {
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "state_feature_count": int(BASE_FEATURE_COUNT),
            "option_feature_count": int(effective_option_feature_count),
            "consequence_fields": consequence_fields,
            "hand_card_vocab": hand_card_vocab,
            "opponent_card_vocab": opponent_card_vocab,
            "use_board_set": bool(args.use_board_set),
            # --ablate-features で潰した特徴の記録。対照群の重みが本番へ昇格することが
            # あるため(2026-08-11: alakazam の T1 ablate 版を policy_weights.json に採用)、
            # 重みファイル自体に素性を残す。潰した列は state_std=0 になっているので、
            # この記録が無いと「なぜ std が 0 の次元があるのか」を後から追えない。
            "ablated_features": (
                {
                    "spec": args.ablate_features,
                    "state_dims": len(ablate_st_idx),
                    "option_dims": len(ablate_op_idx),
                }
                if args.ablate_features
                else None
            ),
            "hidden_size": int(hidden_size),
            "param_counts": param_counts,
            "outcome_weighting": outcome_meta,
            # 2026-08-12: top1_accuracy は単一選択(|chosen_set|==1)の行のみで計算する
            # (既存の生産値 0.6623 等との比較可能性を保つため)。複数選択行の指標は
            # multi_select_precision_at_k(topk と実際の選択集合の重なり率)を別フィールドに
            # 分けて記録する(要件書「What to build」3: 「blend しない」)。
            "train_metrics": _metrics_json(main_train),
            "val_metrics": _metrics_json(main_val),
            "train_val_gap": {
                "top1_accuracy": train_val_top1_gap,
                "mean_weighted_nll": train_val_nll_gap,
            },
            "test_metrics": _metrics_json(main_test),
            "baseline_option_only_test_metrics": _metrics_json(baseline_test),
        },
        "standardization": {
            "state_mean": state_mean.astype(np.float64).tolist(),
            "state_std": state_std.astype(np.float64).tolist(),
            "option_mean": option_mean.astype(np.float64).tolist(),
            "option_std": option_std_.astype(np.float64).tolist(),
        },
        "card_embedding": {
            "dim": _EMBED_DIM,
            "card_id_max": card_id_max,
            "table": embedding_table.tolist(),
        },
        "layers": layers_json,
    }

    # --init-weights: 既存呼び出し(フラグ未指定)ではキー自体を追加しない(出力JSONの
    # スキーマを変えないため)。指定時のみ、元パスと元JSONのmd5を記録する。
    if args.init_weights:
        weights_json["meta"]["init_weights"] = str(args.init_weights)
        weights_json["meta"]["fine_tuned_from"] = init_weights_md5
    # --row-mask: 同様にフラグ未指定なら追加しない。
    if args.row_mask:
        weights_json["meta"]["row_mask"] = {
            "path": str(args.row_mask),
            "n_true": int(row_mask.sum()),
            "n_total": int(len(row_mask)),
        }

    weights_out_path.parent.mkdir(parents=True, exist_ok=True)
    weights_out_path.write_text(json.dumps(weights_json, ensure_ascii=False), encoding="utf-8")
    print(f"\n重みを書き出しました: {weights_out_path} ({weights_out_path.stat().st_size / 1e3:.1f} KB)")

    # --- 自己検証: 本番推論経路(ptcg_ai.learning.policy_model.PolicyModel._forward)と
    # 数値的に一致する独立実装(pure_python_forward、本ファイル内で JSON を再読み込みして
    # 再計算)で照合する。policy_model.py 自体は import しない(独立実装での照合が目的)。
    #
    # 比較は float64 で揃える: PyTorch の学習・推論は float32 で行っているため、素の
    # float32 モデル出力(expected)と float64 の pure_python_forward(got、JSON は float64
    # で書き出し済み)を直接比べると、239次元の内積の丸め誤差の蓄積だけで 1e-6 前後の差が
    # 出てしまい、アルゴリズム自体の一致を検証できない(全件学習でこの丸め誤差が実際に
    # 1e-6 をわずかに超えて自己検証が誤検出した)。学習済みモデルを .double() で float64 化
    # してから同じ標準化式を float64 で適用して比較することで、丸め誤差ではなく実装の
    # 一致(連結順・標準化・埋め込み参照・層の順序)だけを検証する。
    print("\n=== 自己検証: pure_python_forward(JSON再読込) vs PyTorchモデル出力(val split 50件, float64) ===")
    reloaded = json.loads(weights_out_path.read_text(encoding="utf-8"))

    main_model_fp64 = main_model.double()
    main_model_fp64.eval()
    state_mean64 = state_mean.astype(np.float64)
    state_std64 = state_std.astype(np.float64)
    option_mean64 = option_mean.astype(np.float64)
    option_std64 = option_std_.astype(np.float64)

    # ablate対策で std=0 にした列は、上の学習側と同じく 0/0 の NaN を避け、結果を 0 に固定する
    # (pure_python_forward の `if std64[i] else 0.0` と一致させる)。
    _safe_state_std64 = np.where(state_std64 == 0.0, 1.0, state_std64)
    _safe_option_std64 = np.where(option_std64 == 0.0, 1.0, option_std64)

    max_abs_err = 0.0
    n_checked = 0
    with torch.no_grad():
        for sample in self_check_samples:
            card_id = sample["card_id"]
            raw_state = np.asarray(sample["raw_state"], dtype=np.float64)
            raw_option = np.asarray(sample["raw_option"], dtype=np.float64)
            state_std_vec = np.where(
                state_std64 == 0.0, 0.0, (raw_state - state_mean64) / _safe_state_std64
            )
            option_std_vec = np.where(
                option_std64 == 0.0, 0.0, (raw_option - option_mean64) / _safe_option_std64
            )
            x = torch.from_numpy(np.concatenate([state_std_vec, option_std_vec])).unsqueeze(0)
            card_ids_t = torch.tensor([card_id], dtype=torch.int64)
            sample_board_ids = sample.get("board_card_ids")
            if sample_board_ids is not None:
                board_ids_t = torch.tensor([sample_board_ids], dtype=torch.int64)  # (1, BOARD_SLOTS)
                expected = float(main_model_fp64(x, card_ids_t, board_ids_t).item())
            else:
                expected = float(main_model_fp64(x, card_ids_t).item())

            got = pure_python_forward(
                sample["raw_state"], sample["raw_option"], card_id, reloaded,
                board_card_ids=sample_board_ids,
            )
            err = abs(got - expected)
            max_abs_err = max(max_abs_err, err)
            n_checked += 1

    self_check_pass = max_abs_err <= _SELF_CHECK_TOL
    print(f"  検証件数={n_checked}  最大誤差={max_abs_err:.8e}  許容={_SELF_CHECK_TOL}")
    print(f"  結果: {'PASS' if self_check_pass else 'FAIL'}")
    if not self_check_pass:
        print(
            "  エラー: 自己検証に失敗しました。card_embedding の連結順・標準化の適用順序を"
            "確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 最終サマリ ---
    print("\n" + "=" * 60)
    print("学習パイプライン完了サマリ")
    print("=" * 60)
    print(f"  ベースライン(option only)      test: {_fmt_metrics(baseline_test)}")
    print(f"  本命(state ++ option ++ embed) test: {_fmt_metrics(main_test)}")
    print(f"  自己検証: 最大誤差={max_abs_err:.8e} -> {'PASS' if self_check_pass else 'FAIL'}")
    print(f"  出力: {weights_out_path}")

    # --- offline 指標 JSON(容量ablation の記録用、--metrics-out 指定時のみ) ---
    if args.metrics_out is not None:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_json = {
            "hidden_size": int(hidden_size),
            "features": str(features_path),
            "consequence_fields": consequence_fields,
            "outcome_weighting": outcome_meta,
            "seed": _SEED,
            "param_counts": param_counts,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "epochs_trained": len(val_loss_history),
            "train": _metrics_json(main_train),
            "val": _metrics_json(main_val),
            "test": _metrics_json(main_test),
            "train_val_gap": {"top1_accuracy": train_val_top1_gap, "mean_weighted_nll": train_val_nll_gap},
            "self_check_max_abs_err": max_abs_err,
            "out_weights": str(weights_out_path),
        }
        metrics_path.write_text(json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  offline指標を書き出しました: {metrics_path}")


if __name__ == "__main__":
    main()
