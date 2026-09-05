"""1試合分の対戦を、Kaggle シミュレーション基盤を介さずローカルの cg エンジンで直接駆動する。

対戦リーグ(オンライン評価)基盤の中核。design 上の位置づけは
``sample_submission/docs/plans/ml-value-network/step2-design.md`` §5.2 /
``sample_submission/docs/plans/individual/shogo/ml-agent-plan.md`` の「足りない部品4:
評価基盤」に対応する。

``cg.game`` の ``battle_start``/``battle_select``/``battle_finish`` は ``cg.sim.Battle`` の
モジュールグローバルなポインタ(``Battle.battle_ptr``)を介した**プロセス内シングルトン**の
ネイティブ対戦状態を操作する。そのため:

- 1プロセスでは常に1試合しか同時に進行できない(``play_match`` は呼び出し完了まで
  他の対戦と状態を共有しない前提で実装している。並列実行する場合はプロセスを分ける)。
- どんな終わり方(正常終了・例外・異常な手数)でも必ず ``battle_finish()`` を呼び、
  ネイティブ側のメモリを解放する(``try/finally``)。

エージェント関数のシグネチャは ``ptcg_ai/core/agent.py`` が各 ``<agent_type>_agent.py`` に
委譲するときと同じ ``agent(obs: Observation) -> list[int]`` を使う(``main.py`` の
``agent(obs_dict: dict)`` は経由しない。dict→Observation変換を毎手 二重に行わないための
意図的な選択)。デッキ選択も同じ関数を ``obs.select is None`` で1回呼ぶだけで得られる
(既存の ``rule_based_agent``/``ml_policy_agent`` はいずれもこの契約を満たす)。
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# battle_review_viewer/live_match.py と同じブートストラップ: cg/ptcg_ai は
# sample_submission/ 配下にあるため、cwd に依らず import できるようパスを通す。
_ROOT_DIR = Path(__file__).resolve().parent.parent
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"
for _candidate in (str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from cg.api import Observation, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

AgentFn = Callable[[Observation], list[int]]

# ローカル評価専用の「自分の60枚」宣言のうち **dummy 経路**(`ml_policy_agent`)だけを
# 無効化する脱出口(`_begin_match_state` の3を参照。match_context 側=estimated 経路の宣言は
# 常に有効で、この環境変数の影響を受けない)。
# 既定は有効(宣言する)。`PTCG_HARNESS_DECK_INJECT=0` にすると宣言を行わず、修正前の
# 壊れたハーネス(deck.csv 以外を持つ側の dummy 探索が黙って死ぬ)を再現できる。
# 修正の効果を before/after で計測するために用意している(worker プロセスにも環境変数として
# 伝播するので `--workers` 並列でも同じアームを再現できる)。通常の計測では触らないこと。
_DECK_INJECT_ENV = "PTCG_HARNESS_DECK_INJECT"


def _deck_injection_enabled() -> bool:
    return os.environ.get(_DECK_INJECT_ENV, "1").strip() not in ("0", "false", "False")


def _begin_match_state(deck0: list[int], deck1: list[int]) -> None:
    """1試合の開始時に、非公開情報推定レイヤーの状態を初期化する(ローカル評価専用)。

    本番(Kaggle)は ``obs.select is None``(デッキ選択ターン)を受けて `match_context.update` が
    自分で reset し、自分のデッキは常に `deck.csv` なので、この関数に相当する処理は要らない。
    ローカルの `battle_start(deck0, deck1)` はその経路を通らないため、以下2点を明示的に行う:

    1. `reset()` — worker プロセスは試合をまたいで再利用されるため、前の試合の
       山札/サイド推定・相手モデルが残留する(残すと次の試合の推定が汚染される)。
    2. `set_own_deck_override()` — 両陣営が1プロセスで動くのにデッキはエンジンへ直接渡されるため、
       エージェントは両方とも `deck.csv` を自分の山札だと誤認する。deck.csv 以外を持つ側は
       `OwnHiddenState` のサイド枚数が実盤面と食い違い `search_begin` が必ず例外になり、
       **その陣営だけ PIMC 探索が黙って無効化**される(実測 begin失敗率≒100%)。
       正しい60枚を宣言して、両陣営を同じ条件で探索させる。
    3. `ml_policy_agent.set_own_deck_override()` — 上記2は `hidden_state_source="estimated"`
       の経路しか救わない。既定は **dummy** 経路(`build_dummy_search_state(obs, _get_deck())`)で、
       `_get_deck()` は CWD の `deck.csv` を返すため、それ以外のデッキを渡した側は
       「観測に見えるカードがデッキに無い」で hidden_state が **必ず None** になる。
       すると `lethal_search`(既定 dummy)と `_model_hidden_state_factory` 経由の `ko_search`
       系ガード(briar_gate / low_deck_draw_brake / boss_lethal_gate / ability_draw_brake /
       terastal_rotation / hammer_veto)が**ローカルでは一度も発火しない**
       (本番は 1プロセス1エージェントで deck.csv = 提出デッキなので起きない、ハーネス側の欠陥)。
       dummy 経路にも同じ60枚を宣言する。

    推定レイヤーの初期化失敗で対戦を止めない(production と同じ例外安全方針)。
    """
    try:
        from ptcg_ai.hidden_information import match_context

        match_context.reset()
        match_context.set_own_deck_override(0, deck0)
        match_context.set_own_deck_override(1, deck1)
    except Exception:  # noqa: BLE001 - 推定レイヤーの都合で対戦を落とさない
        pass

    try:
        from ptcg_ai.ml_policy import ml_policy_agent

        if _deck_injection_enabled():
            ml_policy_agent.set_own_deck_override(0, deck0)
            ml_policy_agent.set_own_deck_override(1, deck1)
        else:
            ml_policy_agent.clear_own_deck_override()
    except Exception:  # noqa: BLE001 - 宣言の失敗で対戦を落とさない(未宣言=従来挙動)
        pass

# 異常終了(エージェントの無限ループ相当・エンジン側の想定外挙動)を検知する安全弁。
# 通常の対戦は数十~数百手で終わる(rule_based vs rule_based の実測で100~200手程度)。
MAX_STEPS = 3000


@dataclass
class MatchResult:
    """1試合の結果。

    ``winner`` は正常終了時のみ 0/1(``cg.api.State.result`` そのまま)。異常終了時は
    ``None`` になり、``error`` に理由が入る(呼び出し側は winner が None の試合を
    勝敗集計から除外し、別途 error 件数として報告すること)。
    """

    winner: int | None
    turns: int | None
    steps: int
    seconds: float
    error: str | None = None


def play_match(
    agent0: AgentFn,
    agent1: AgentFn,
    deck0: list[int],
    deck1: list[int],
    seed: int | None = None,
) -> MatchResult:
    """1試合を最初から最後まで実行し、勝敗を返す。

    Args:
        agent0: player_index=0 側のエージェント関数(``agent(obs) -> list[int]``)。
        agent1: player_index=1 側のエージェント関数。
        deck0: player_index=0 の60枚デッキ(カードIDリスト)。
        deck1: player_index=1 の60枚デッキ。
        seed: 指定すれば ``random.seed(seed)`` を ``battle_start`` 前に呼ぶ(このプロセスの
            Python 側乱数を使うエージェントの再現性のため。ネイティブエンジン内部の
            シャッフルまで制御できるかは未確認)。

    Returns:
        MatchResult: 勝敗・手数・所要秒数。エージェントの例外・不正な選択・
        ``MAX_STEPS`` 超過はクラッシュさせず ``error`` 付きの結果として返す
        (対戦リーグ全体を1試合の異常で止めないため)。
    """
    if seed is not None:
        random.seed(seed)

    _begin_match_state(deck0, deck1)

    agents: dict[int, AgentFn] = {0: agent0, 1: agent1}
    t0 = time.time()

    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        return MatchResult(
            winner=None, turns=None, steps=0, seconds=time.time() - t0,
            error=f"battle_start errorType={start_data.errorType}",
        )

    steps = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current is None:
                return MatchResult(
                    winner=None, turns=None, steps=steps, seconds=time.time() - t0,
                    error="current is None mid-match",
                )
            if obs.current.result != -1:
                return MatchResult(
                    winner=obs.current.result, turns=obs.current.turn,
                    steps=steps, seconds=time.time() - t0,
                )
            if steps >= MAX_STEPS:
                return MatchResult(
                    winner=None, turns=obs.current.turn, steps=steps, seconds=time.time() - t0,
                    error=f"max_steps_exceeded({MAX_STEPS})",
                )

            turn_player = obs.current.yourIndex
            action = agents[turn_player](obs)
            obs_dict = battle_select(action)
            steps += 1
    except Exception as exc:  # noqa: BLE001 - 1試合の異常でリーグ全体を止めない
        return MatchResult(
            winner=None, turns=None, steps=steps, seconds=time.time() - t0,
            error=repr(exc),
        )
    finally:
        battle_finish()
