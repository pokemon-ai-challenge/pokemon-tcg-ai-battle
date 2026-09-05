"""模倣ポリシー(選択肢スコアリング)の推論モジュール(Step2)。

``kaggle_replays/policy_net/`` の学習パイプライン(オフライン・このモジュールのスコープ外)が
フーディン使用プレイヤーのリプレイからリストワイズ模倣学習(softmax 交差エントロピー)で
「選択肢スコア関数」を学習し、標準化パラメータ・各層の重みを ``policy_weights.json`` へ
エクスポートする。本モジュールはその重みJSONを読み込み、対戦中は純Python(``math`` のみ、
numpy/torch 非依存)のフォワードパスだけで各選択肢のスコアを計算する。

設計は :mod:`ptcg_ai.learning.value_model`(:class:`ValueModel`)と揃える:
- 重みファイルが存在しなくても例外にせず「未ロード状態」とし、``is_ready`` で判定できる。
  未ロード時は先頭の選択肢(index 0)を返す安全側フォールバック(常に有効な選択肢を返す)。
- 外部ライブラリに依存しない。行列積・活性化は素の Python で書く。

重みJSONのスキーマ(学習パイプラインと共有する契約。このファイル側の都合で変えない):

```
{
  "meta": {
    "state_feature_count": 166, "option_feature_count": 65,
    "test_metrics": {...}, ...
  },
  "standardization": {
    "state_mean": [166], "state_std": [166],
    "option_mean": [65], "option_std": [65]
  },
  "card_embedding": {
    "dim": 8,
    "card_id_max": 1267,
    "table": [[...dim個の float...], ...]   # 長さ card_id_max+1(index 0 = 識別なし/範囲外)
  },
  "layers": [
    {"weight": [[...]], "bias": [...]},   # Linear(166+65+dim, H) -> 後段で ReLU
    {"weight": [[...]], "bias": [...]}    # 最終層 Linear(H, 1)。活性化なし(生スコア)
  ]
}
```

## フォワードパス

各選択肢について、状態特徴(``encoder.encode_state``、この意思決定点で共通)・選択肢特徴
(``encoder.encode_options`` の該当行)をそれぞれ標準化し、さらに対象カード/ポケモンの
identity 埋め込み(``encoder.encode_option_card_ids`` の該当値を ``card_embedding.table``
で引いたベクトル。標準化はしない)を連結する
(``x' = state' ++ option' ++ card_embedding``)。中間層に ReLU、最終層は活性化なしの
スカラースコアを出す。``layers`` の最後の1層だけ ReLU を適用しない(value_model.py と異なり
sigmoid も適用しない。選択肢間の相対比較にしか使わないため、確率化・較正は行わない設計判断。
詳細は ``sample_submission/docs/plans/ml-value-network/step2-algorithm-selection.md`` §5)。
選ぶ選択肢は全選択肢中でスコア最大の index(argmax)。

## card_embedding(個別カード識別、2026-07-20 追加)

``encode_options()`` の連続値特徴はカード種別(ポケモン/アイテム/どうぐ/…)までしか
区別せず、同じ種別内の個別カード(「博士の研究」と「ハイパーボール」等)や
エネルギーの種類を識別できない(選択肢特徴だけでは PLAY/ATTACH の精度が伸び悩んだ、
step2-algorithm-selection.md §7 参照)。この個別カードの識別は、``CardData.cardId`` を
キーにした**学習済み埋め込み**(``card_embedding.table``)で補う。

埋め込みテーブルは**デッキではなくゲーム全体のカードデータに基づく**(index は
``card_id`` そのもの、``card_id_max`` は学習時点の ``all_card_data()`` の最大値)。
これにより、将来別のデッキのリプレイで再学習しても本モジュールのコードは変更不要
(埋め込みテーブルの中身・サイズだけが学習データに応じて変わる)。学習データに出現しない
カードの埋め込みは未学習のまま(通常は初期値のゼロ近傍)になるだけで、安全側に劣化する。

``card_id`` が 0、または ``card_id_max`` の範囲外(将来カードデータが増えた場合など)は
index 0(「識別なし」用の予約枠)にフォールバックする。
"""

from __future__ import annotations

import json
from pathlib import Path

from cg.api import Observation, State
from ptcg_ai.learning import encoder

_DEFAULT_WEIGHTS_FILENAME = "policy_weights.json"

# card_embedding.table の index 0 は「識別なし / 範囲外」用に予約されている。
_UNKNOWN_CARD_EMBEDDING_INDEX = 0


class PolicyModel:
    """学習済みの選択肢スコア関数を読み込み、Observation から選ぶべき選択肢を推論する。"""

    def __init__(self, weights_path: str | Path | None = None, deck_card_ids: list[int] | None = None):
        if weights_path is None:
            weights_path = Path(__file__).parent / _DEFAULT_WEIGHTS_FILENAME
        self._weights_path = Path(weights_path)
        # 入力拡張(extra_features)。meta["extra_features"] が設定された重みのみ有効。
        # 既定(production)重みには無いので None=従来どおり(state 166次元)。
        self._extra_features: str | None = None
        self._deck_card_ids: list[int] | None = list(deck_card_ids) if deck_card_ids else None

        self._state_mean: list[float] | None = None
        self._state_std: list[float] | None = None
        self._option_mean: list[float] | None = None
        self._option_std: list[float] | None = None
        self._card_embedding_table: list[list[float]] | None = None
        self._card_id_max: int = 0
        # 各層は (W[out][in], b[out]) のタプル。最終層以外に ReLU を適用する。
        self._layers: list[tuple[list[list[float]], list[float]]] | None = None
        # Tier3 Stage3c: この重みが学習時に使ったconsequence特徴名のリスト(空なら不使用、
        # 旧重み完全後方互換)。encoder.CONSEQUENCE_FEATURE_NAMES の部分集合・同じ順序。
        self._consequence_fields: list[str] = []

        self._load()

    @property
    def is_ready(self) -> bool:
        """重みJSONの読み込みに成功していれば True。"""
        return self._layers is not None

    def _load(self) -> None:
        if not self._weights_path.exists():
            return  # 未ロード状態のまま(重みは学習パイプライン完了後に配置される)。

        with self._weights_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        std = payload["standardization"]
        self._state_mean = [float(v) for v in std["state_mean"]]
        self._state_std = [float(v) for v in std["state_std"]]
        self._option_mean = [float(v) for v in std["option_mean"]]
        self._option_std = [float(v) for v in std["option_std"]]

        card_embedding = payload["card_embedding"]
        self._card_id_max = int(card_embedding["card_id_max"])
        self._card_embedding_table = [
            [float(v) for v in row] for row in card_embedding["table"]
        ]

        self._consequence_fields = list(payload.get("meta", {}).get("consequence_fields") or [])
        self._extra_features = payload.get("meta", {}).get("extra_features") or None

        self._layers = [
            (
                [[float(w) for w in row] for row in layer["weight"]],
                [float(v) for v in layer["bias"]],
            )
            for layer in payload["layers"]
        ]

    def _compute_extra(self, state) -> list[float] | None:
        """入力拡張(extra_features)を計算。meta フラグが無い重みでは None(従来どおり)。

        parity: 学習側(collect_field)と **同一の compute_extra_features** を同じデッキ・同じ
        predictor で通す。"own_resource" / "opp_belief" / "own_resource+opp_belief" を部分文字列で判定。
        "opp_belief" を含むときだけ相手デッキ予測器(match_context._get_predictor)をロードする。
        """
        ef = self._extra_features
        if not ef:
            return None
        from ptcg_ai.learning.extra_features import compute_extra_features
        predictor = None
        if "opp_belief" in ef:
            try:
                from ptcg_ai.hidden_information import match_context
                predictor = match_context._get_predictor()
            except Exception:  # noqa: BLE001 - 予測器のロード失敗で意思決定を止めない
                predictor = None
        return compute_extra_features(state, ef, self._deck_card_ids, predictor)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------
    def score_options(
        self,
        obs: Observation,
        hidden_state_factory=None,
        deadline: float | None = None,
    ) -> list[float]:
        """``obs.select.option`` の各選択肢のスコア(生の値、確率ではない)を返す。

        未ロード時、``obs.current``/``obs.select`` が無い場合、選択肢が0件の場合は
        空リストを返す(例外は出さない)。

        Args:
            hidden_state_factory / deadline: この重みがconsequence特徴
                (``meta.consequence_fields``、Tier3 Stage3c)を使う場合のみ必要。
                `ptcg_ai.search.attack_plan`/`_try_lethal` 等と同じ0引数callable
                (相手の非公開情報スタブ)と ``time.perf_counter()`` 基準の締め切り。
                consequence特徴を使わない重み(既定)では無視される
                (呼び出し側は渡さなくてもよい。完全後方互換)。
        """
        if not self.is_ready or obs.current is None or obs.select is None or not obs.select.option:
            return []
        state_features = encoder.encode_state_from_state(obs.current, extra_features=self._compute_extra(obs.current))
        option_rows = encoder.encode_options_from_state(obs.current, obs.select)
        card_ids = encoder.encode_option_card_ids(obs.current, obs.select)
        if self._consequence_fields:
            option_rows = self._append_consequence_features(obs, option_rows, hidden_state_factory, deadline)
        return [
            self._forward(state_features, option_row, card_id)
            for option_row, card_id in zip(option_rows, card_ids)
        ]

    def score_options_from_state(self, state: State | None, select) -> list[float]:
        """State/SelectData を直接受け取る版(単体テスト・オフライン評価向け)。

        consequence特徴(Tier3 Stage3c)は非対応(``search_begin`` に必要な
        ``Observation.search_begin_input`` が無いため)。consequence特徴を使う重みを
        テストする場合は :meth:`score_options` に完全な ``Observation`` を渡すこと。
        この重みが ``meta.consequence_fields`` を持つ場合、未対応の呼び出しと分かるよう
        例外にする(値0で静かにフォールバックしてスコアが歪むより安全)。
        """
        if self._consequence_fields:
            raise ValueError(
                "この重みは consequence 特徴(meta.consequence_fields)を使うため、"
                "score_options_from_state ではなく完全な Observation を渡す "
                "score_options を使ってください。"
            )
        if not self.is_ready or state is None or select is None or not select.option:
            return []
        state_features = encoder.encode_state_from_state(state, extra_features=self._compute_extra(state))
        option_rows = encoder.encode_options_from_state(state, select)
        card_ids = encoder.encode_option_card_ids(state, select)
        return [
            self._forward(state_features, option_row, card_id)
            for option_row, card_id in zip(option_rows, card_ids)
        ]

    def select_option(
        self,
        obs: Observation,
        hidden_state_factory=None,
        deadline: float | None = None,
    ) -> int | None:
        """スコア最大の選択肢インデックスを返す。

        選択肢が0件(``obs.select`` が無い等)なら None。未ロード時は index 0
        (常に有効な選択肢)を返す安全側フォールバック。``hidden_state_factory``/
        ``deadline`` は :meth:`score_options` と同じ(consequence特徴を使う重みのみ必要)。
        """
        select = obs.select
        if select is None or not select.option:
            return None
        if not self.is_ready:
            return 0
        scores = self.score_options(obs, hidden_state_factory, deadline)
        return max(range(len(scores)), key=lambda i: scores[i])

    def _append_consequence_features(
        self,
        obs: Observation,
        option_rows: list[list[float]],
        hidden_state_factory,
        deadline: float | None,
    ) -> list[list[float]]:
        """``self._consequence_fields`` の並びで consequence 特徴を計算し、各選択肢の
        option_rows 末尾に連結する(学習時: kaggle_replays/policy_net/train.py が
        option_features に同じ並びで連結してから標準化しているのと同じ順序。§ train.py参照)。

        ``hidden_state_factory`` が無ければ全て0で埋める(fail-soft。呼び出し側が
        consequence特徴対応を怠っても例外にしない)。
        """
        if hidden_state_factory is None:
            zeros = [0.0] * len(self._consequence_fields)
            return [row + zeros for row in option_rows]
        effective_deadline = deadline if deadline is not None else float("inf")
        all_vectors = encoder.encode_option_consequence_features(obs, hidden_state_factory, effective_deadline)
        name_to_idx = {name: i for i, name in enumerate(encoder.CONSEQUENCE_FEATURE_NAMES)}
        selected_idx = [name_to_idx[name] for name in self._consequence_fields]
        return [
            row + [vec[i] for i in selected_idx]
            for row, vec in zip(option_rows, all_vectors)
        ]

    # ------------------------------------------------------------------
    # 内部: フォワードパス
    # ------------------------------------------------------------------
    def _card_embedding(self, card_id: int) -> list[float]:
        """card_id の埋め込みベクトルを引く。範囲外・未識別(0)は予約枠にフォールバック。"""
        table = self._card_embedding_table
        index = card_id if 0 <= card_id <= self._card_id_max else _UNKNOWN_CARD_EMBEDDING_INDEX
        return table[index]

    def _forward(self, state_features: list[float], option_features: list[float], card_id: int) -> float:
        """標準化 → 状態/選択肢特徴+カード埋め込みを連結 → 各層フォワード(最終層は活性化なし)。"""
        state_mean, state_std = self._state_mean, self._state_std
        option_mean, option_std = self._option_mean, self._option_std

        h = [
            (state_features[i] - state_mean[i]) / state_std[i] if state_std[i] else 0.0
            for i in range(len(state_features))
        ]
        h += [
            (option_features[i] - option_mean[i]) / option_std[i] if option_std[i] else 0.0
            for i in range(len(option_features))
        ]
        h += self._card_embedding(card_id)

        n_layers = len(self._layers)
        for layer_idx, (W, b) in enumerate(self._layers):
            z = [b[k] + sum(W[k][i] * h[i] for i in range(len(h))) for k in range(len(W))]
            is_last = layer_idx == n_layers - 1
            h = z if is_last else [v if v > 0.0 else 0.0 for v in z]
        return h[0]
