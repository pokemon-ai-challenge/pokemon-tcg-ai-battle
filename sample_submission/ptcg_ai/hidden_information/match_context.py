"""1試合分の非公開情報推定状態（``OwnHiddenState`` / ``OpponentHiddenState`` / ``OpponentKnowledge``）を
まとめて保持する、モジュールレベルシングルトン。

``ptcg_ai.shared.card_cache`` と同じパターン（``global`` 変数 + ``reset_cache()`` 相当の ``reset()``）を
踏襲する。``main.agent(obs_dict)`` は1試合を通じて同じ Python プロセス内で繰り返し呼ばれる
（``tests/local_sim/test_local_game.py`` のループ、Kaggle 実行環境も同様）ため、モジュールレベルの
変数で試合中の状態を持ち越せる（実装プラン Phase 4 §4.1）。

呼び出し方（唯一の入口）: ``rule_based_agent.agent(obs)`` の先頭で ``update(obs)`` を呼ぶだけでよい。
新しい試合の開始検知は ``obs.select is None``（デッキ選択ターン。CLAUDE.md の戻り値ルール表と一致）。

## プレイヤー分離（Stage2: match_context のプレイヤー分離）

本番の Kaggle 実行（1プロセス1エージェント）ではプロセス内に常に1人分の視点しか存在しないため、
単一のグローバル状態で十分だった。しかしローカルの自己対戦harness（``league/run_match.play_match``）
は1プロセス内で両陣営の ``agent(obs)`` を交互に呼ぶため、単一のグローバル状態のままだと異なる
config の2エージェントを対戦させたときに互いの推定が混線する。これを避けるため、``_own_states`` /
``_opponent_states`` / ``_knowledge`` は ``player_index``（``state.yourIndex``、0 or 1）をキーにした
dict として保持する。``get_own_state``/``get_opponent_state`` は ``player_index`` を必須引数にする
（暗黙のデフォルト値は「暗黙の単一視点」バグを形を変えて再導入するだけなので避ける）。

``update()`` の ``obs.select is None`` 分岐（デッキ選択ターン）だけは例外: ``obs.current`` がまだ
``None`` のため ``yourIndex`` を取得する手段が無く、プレイヤー別化の対象外とする（実害は無い。
ローカルharnessは ``battle_start(deck0, deck1)`` でデッキを直接渡すためこの経路自体を通らないことを
Stage1で確認済み。本番は1プロセス1エージェントなのでそもそも問題が起きない）。

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

_knowledge: dict[int, OpponentKnowledge] = {}
_own_states: dict[int, OwnHiddenState] = {}
_opponent_states: dict[int, OpponentHiddenState] = {}

# HybridDeckPredictor は重みJSONの読み込みを伴うため、card_cache.py と同じ思想で
# プロセス内（試合をまたいでも）使い回す。reset() では破棄しない。
# プレイヤー間で共有しても安全（読み込んだ重みに対する予測のみで、対局固有の状態を持たない）。
_predictor: HybridDeckPredictor | None = None
_predictor_load_failed = False


def reset() -> None:
    """新しい試合の開始を検知したら呼ぶ。1試合分の状態を全て破棄する。

    テストからも直接呼べるよう公開する（``card_cache.reset_cache()`` と同じパターン）。
    ``player_index`` の別なく両方のエントリをまとめてクリアする（既存の全体リセット動作を維持する。
    Non-goals: 片側の ``player_index`` だけをリセットする用途は無い）。
    ``_predictor`` はプロセス内で使い回す資産なのでここでは破棄しない。
    """
    _knowledge.clear()
    _own_states.clear()
    _opponent_states.clear()


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


def get_own_state(player_index: int) -> OwnHiddenState:
    """``player_index`` の ``OwnHiddenState`` を返す。未初期化（``reset()`` 直後・その
    ``player_index`` について最初の ``update()`` 前）でも例外を出さないよう、その場で
    自分の60枚から作り直したインスタンスを返す（``player_index`` 別にフォールバック）。
    """
    state = _own_states.get(player_index)
    if state is None:
        try:
            state = OwnHiddenState(_load_own_deck_ids())
        except Exception:  # noqa: BLE001 -- deck.csv 読み込み失敗時も空デッキで安全側に倒す
            state = OwnHiddenState([])
        _own_states[player_index] = state
    return state


def get_opponent_state(player_index: int) -> OpponentHiddenState:
    """``player_index`` の ``OpponentHiddenState`` を返す。未初期化でも空（プールJSON未検出時と
    同じ ``is_ready=False`` 相当）のインスタンスを返す（コンストラクタ自体は代表リストが無くても
    例外にしない設計）。
    """
    state = _opponent_states.get(player_index)
    if state is None:
        state = OpponentHiddenState()
        _opponent_states[player_index] = state
    return state


def update(obs: Observation) -> None:
    """毎ターン、``rule_based_agent.agent(obs)`` の先頭で呼ぶ。

    - ``obs.select is None``（デッキ選択ターン）: 新しい試合の開始とみなし ``reset()`` するだけ
      （``obs.current`` がまだ ``None`` のため ``yourIndex`` が取得できず、``player_index`` 別の
      登録ができない。自分の60枚の登録は次の通常ターンの ``update()``、あるいは
      ``get_own_state(player_index)`` の未初期化フォールバックに委ねる。モジュールdocstring
      「プレイヤー分離」参照）。
    - 通常ターン: ``state.yourIndex`` をキーに、``OpponentKnowledge`` の呼び出し順契約
      （``update_from_logs`` → ``update_from_state``、``opponent_knowledge.py`` 冒頭docstring）を
      守って更新し、``HybridDeckPredictor.predict()`` の結果で ``OpponentHiddenState`` を更新し、
      ``OwnHiddenState.update()``（+ 山札サーチ検知時は ``resolve_deck_search()``）を呼ぶ。

    本体は丸ごと ``try/except`` で包む（モジュールdocstring「例外安全性」参照）。失敗時は今回の
    更新を諦めるだけで、既存の推定状態は変更されない（次ターンで再チャレンジする）。
    """
    try:
        if obs.select is None:
            reset()
            return

        state = obs.current
        if state is None:
            return

        me = state.yourIndex

        own_state = _own_states.get(me)
        if own_state is None:
            own_state = OwnHiddenState(_load_own_deck_ids())
            _own_states[me] = own_state

        knowledge = _knowledge.get(me)
        if knowledge is None:
            knowledge = OpponentKnowledge(opponent_index=1 - me)
            _knowledge[me] = knowledge

        opponent_state = _opponent_states.get(me)
        if opponent_state is None:
            opponent_state = OpponentHiddenState()
            _opponent_states[me] = opponent_state

        # 順序契約: update_from_logs → update_from_state（opponent_knowledge.py 冒頭docstring厳守）。
        knowledge.update_from_logs(obs.logs)
        knowledge.update_from_state(state)

        features = knowledge.get_prediction_features()

        predictor = _get_predictor()
        if predictor is not None and getattr(predictor, "is_ready", False):
            posterior = predictor.predict(features["observed_cards"], state.turn)
        else:
            posterior = {}

        opponent_player = state.players[1 - me]
        opponent_state.update(posterior, features["observed_card_ids"], opponent_player)

        own_state.update(state, obs.select)
        if obs.select.deck is not None:
            own_state.resolve_deck_search(obs.select)
    except Exception:  # noqa: BLE001 -- 推定レイヤーの失敗を意思決定に波及させない（モジュールdocstring参照）
        pass
