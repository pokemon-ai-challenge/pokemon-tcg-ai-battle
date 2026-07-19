"""1試合分の非公開情報推定状態（``OwnHiddenState`` / ``OpponentHiddenState`` / ``OpponentKnowledge``）を
まとめて保持する、モジュールレベルシングルトン。

``ptcg_ai.shared.card_cache`` と同じパターン（``global`` 変数 + ``reset_cache()`` 相当の ``reset()``）を
踏襲する。``main.agent(obs_dict)`` は1試合を通じて同じ Python プロセス内で繰り返し呼ばれる
（``tests/local_sim/test_local_game.py`` のループ、Kaggle 実行環境も同様）ため、モジュールレベルの
変数で試合中の状態を持ち越せる（実装プラン Phase 4 §4.1）。

呼び出し方（唯一の入口）: ``rule_based_agent.agent(obs)`` の先頭で ``update(obs)`` を呼ぶだけでよい。
新しい試合の開始検知は ``obs.select is None``（デッキ選択ターン。CLAUDE.md の戻り値ルール表と一致）。

## 例外安全性（重要）

本モジュールは Kaggle 提出コードの一部として動く。推定レイヤー（本モジュール以下）の計算が
何らかの理由で失敗しても、対局そのもの（``router.route()`` 以降の意思決定）を絶対に止めては
いけない。そのため ``update()`` の本体は丸ごと ``try/except`` で包み、失敗時は例外を握りつぶして
「今回の更新をスキップする」だけに留める（既存の状態はそのまま残り、次ターンの ``update()`` で
再度更新を試みる。既存コードの ``HybridDeckPredictor`` 等の「未ロード状態にフォールバックし
例外にしない」という思想を、対局ループ接続レベルでも徹底する）。
"""

from __future__ import annotations

from cg.api import Observation

from ptcg_ai.opponent_modeling.hybrid_predictor import HybridDeckPredictor
from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge

from .opponent_hidden_state import OpponentHiddenState
from .own_hidden_state import OwnHiddenState

_knowledge: OpponentKnowledge | None = None
_own_state: OwnHiddenState | None = None
_opponent_state: OpponentHiddenState | None = None

# HybridDeckPredictor は重みJSONの読み込みを伴うため、card_cache.py と同じ思想で
# プロセス内（試合をまたいでも）使い回す。reset() では破棄しない。
_predictor: HybridDeckPredictor | None = None
_predictor_load_failed = False


def reset() -> None:
    """新しい試合の開始を検知したら呼ぶ。1試合分の状態を全て破棄する。

    テストからも直接呼べるよう公開する（``card_cache.reset_cache()`` と同じパターン）。
    ``_predictor`` はプロセス内で使い回す資産なのでここでは破棄しない。
    """
    global _knowledge, _own_state, _opponent_state
    _knowledge = None
    _own_state = None
    _opponent_state = None


def _load_own_deck_ids() -> list[int]:
    """自分の60枚を ``read_deck_csv()`` から取得する。

    ``rule_based_agent`` への依存は呼び出し時点まで遅延importする。本モジュールは
    ``rule_based_agent.agent()`` から呼ばれる想定であり、モジュール先頭で
    ``from ptcg_ai.rule_based.rule_based_agent import read_deck_csv`` としてしまうと、
    「rule_based_agent → match_context → rule_based_agent」という循環importになり
    （rule_based_agent がまだ ``read_deck_csv`` を定義し終える前に再import される）壊れる。
    関数内 import なら、実際に呼ばれる時点（``agent()`` 本体の実行時）では
    rule_based_agent は既にロード完了しているため問題にならない。
    """
    from ptcg_ai.rule_based.rule_based_agent import read_deck_csv

    return read_deck_csv()


def _get_predictor() -> HybridDeckPredictor | None:
    """``HybridDeckPredictor`` をプロセス内で使い回す。構築自体が失敗しても例外を外に
    出さず、``None``（=予測なし。呼び出し側は空の事後分布にフォールバックする）にする。
    """
    global _predictor, _predictor_load_failed
    if _predictor is not None:
        return _predictor
    if _predictor_load_failed:
        return None
    try:
        _predictor = HybridDeckPredictor()
    except Exception:  # noqa: BLE001 -- 予測器の構築失敗は「予測なし」に落とすだけで致命的にしない
        _predictor_load_failed = True
        return None
    return _predictor


def get_own_state() -> OwnHiddenState:
    """``OwnHiddenState`` を返す。未初期化（``reset()`` 直後・最初の ``update()`` 前）でも
    例外を出さないよう、その場で自分の60枚から作り直したインスタンスを返す。
    """
    global _own_state
    if _own_state is None:
        try:
            _own_state = OwnHiddenState(_load_own_deck_ids())
        except Exception:  # noqa: BLE001 -- deck.csv 読み込み失敗時も空デッキで安全側に倒す
            _own_state = OwnHiddenState([])
    return _own_state


def get_opponent_state() -> OpponentHiddenState:
    """``OpponentHiddenState`` を返す。未初期化でも空（プールJSON未検出時と同じ ``is_ready=False``
    相当）のインスタンスを返す（コンストラクタ自体は代表リストが無くても例外にしない設計）。
    """
    global _opponent_state
    if _opponent_state is None:
        _opponent_state = OpponentHiddenState()
    return _opponent_state


def update(obs: Observation) -> None:
    """毎ターン、``rule_based_agent.agent(obs)`` の先頭で呼ぶ。

    - ``obs.select is None``（デッキ選択ターン）: 新しい試合の開始とみなし ``reset()`` してから、
      自分の60枚を ``OwnHiddenState`` として登録するだけ（``obs.current`` がまだ ``None`` のため
      それ以上の更新はできない）。
    - 通常ターン: ``OpponentKnowledge`` の呼び出し順契約（``update_from_logs`` → ``update_from_state``、
      ``opponent_knowledge.py`` 冒頭docstring）を守って更新し、``HybridDeckPredictor.predict()`` の
      結果で ``OpponentHiddenState`` を更新し、``OwnHiddenState.update()``（+ 山札サーチ検知時は
      ``resolve_deck_search()``）を呼ぶ。

    本体は丸ごと ``try/except`` で包む（モジュールdocstring「例外安全性」参照）。失敗時は今回の
    更新を諦めるだけで、既存の推定状態は変更されない（次ターンで再チャレンジする）。
    """
    global _knowledge, _own_state, _opponent_state
    try:
        if obs.select is None:
            reset()
            _own_state = OwnHiddenState(_load_own_deck_ids())
            return

        state = obs.current
        if state is None:
            return

        if _own_state is None:
            _own_state = OwnHiddenState(_load_own_deck_ids())
        if _knowledge is None:
            _knowledge = OpponentKnowledge(opponent_index=1 - state.yourIndex)
        if _opponent_state is None:
            _opponent_state = OpponentHiddenState()

        # 順序契約: update_from_logs → update_from_state（opponent_knowledge.py 冒頭docstring厳守）。
        _knowledge.update_from_logs(obs.logs)
        _knowledge.update_from_state(state)

        features = _knowledge.get_prediction_features()

        predictor = _get_predictor()
        if predictor is not None and getattr(predictor, "is_ready", False):
            posterior = predictor.predict(features["observed_cards"], state.turn)
        else:
            posterior = {}

        opponent_player = state.players[1 - state.yourIndex]
        _opponent_state.update(posterior, features["observed_card_ids"], opponent_player)

        _own_state.update(state, obs.select)
        if obs.select.deck is not None:
            _own_state.resolve_deck_search(obs.select)
    except Exception:  # noqa: BLE001 -- 推定レイヤーの失敗を意思決定に波及させない（モジュールdocstring参照）
        pass
