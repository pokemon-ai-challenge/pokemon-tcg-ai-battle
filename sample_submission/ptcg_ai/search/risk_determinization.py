"""リスク調整Determinization探索（EXP-A44 #P3）。

``docs/plans/search/EXP-A44-risk-adjusted-determinization.md`` §5.1〜§5.9 の実装。
MAIN選択の場面に限り、ルールベース（``rule_based.main_turn_parts.proposals``）の上位候補を
複数の決定化サンプル（相手の隠し情報を1通りに固定した局面）上で評価し、
``risk_aggregation.aggregate()`` で集約したスコアで並べ替える層。``lethal_simple`` と同じ
チーム共通インターフェースに従う。

Entry point (team common interface)::

    def search(state, legal_actions, context) -> list[int] | None

``context`` キー（``action_selection.selector.py`` が組み立てる。§5.1）:

- ``observation`` (必須): エージェントに渡された ``Observation``。
- ``config``: config の ``risk_determinization`` セクション（欠損キーは ``DEFAULTS``）。
- ``hidden_state_factory`` (必須): 引数なしで ``search_begin()`` 用 dict を返す callable。
  呼ぶたびに別サンプル（決定化）。``None`` を返すこともある（その回はスキップ）。
- ``candidate_provider`` (必須): 引数なしで候補手 ``list[tuple[list[int], float]]``
  （action, rule_score）を返す callable。setup-first フィルタ後・ルールスコア降順の前提
  （``selector.py`` 側の責務、§5.5）。
- ``deadline_ms`` (任意): この探索が使ってよい壁時計予算（``config.time_limit_ms`` とのmin）。

戻り値: 採用する option index list、または None（＝呼び出し側は router.route へフォールバック）。
例外は全て握り潰して None を返す（呼び出し側 ``selector.select_action`` の契約と同じ）。

``PTCG_DISABLE_RISK_DET=1`` で config の値に関わらず強制的に無効化できる
（``rule_based/main_turn_parts/proposals.py`` の ``PTCG_SETUP_BEFORE_ATTACK`` と同じ流儀）。

## 「残りゲーム時間」の扱い（実装上の判断・要件書に明記が無い点）

``cg/api.py`` の ``State``/``Observation`` には対局全体の残り壁時計時間を返すフィールドが無い
（コンペ全体の持ち時間はエンジン側で管理されており、エージェント側からは見えない）。
そのため本モジュールは、要件書 §2.1 の「1試合10分」という前提を ``_TOTAL_MATCH_BUDGET_MS`` として
定数化し、``State.turn`` が前回観測より減った（＝新しい試合が始まった）タイミングを検知して
試合開始時刻を記録し、そこからの経過時間を差し引いた近似値を「残りゲーム時間」として使う。
真の残り時間ではなく近似であることを明記する（実装報告にも記載）。
"""

from __future__ import annotations

import os
import time
from typing import Callable

from cg import api as cg_api
from cg.api import Observation, SelectType, State

from ptcg_ai.hidden_information import match_context
from ptcg_ai.learning.value_model import ValueModel
from ptcg_ai.search import risk_aggregation

DEFAULTS: dict = {
    # 要件書の「既定は必ずOFF」を守るため、モジュール既定は False にする(configs/rule_lethal.json
    # のように risk_determinization セクション自体が無い設定では、この既定がそのまま使われる)。
    # configs/rule_risk.json 側で明示的に true にする。
    "enabled": False,
    "module": "risk_determinization",
    "mode": "mean",
    "alpha": 0.3,
    "beta": 1.0,
    "determinizations": 6,
    "top_k": 4,
    "risk_weight": 0.25,
    "min_turn": 3,
    "time_limit_ms": 1200,
    "min_remaining_game_ms": 120000,
    "max_evaluations": 32,
    "rng_seed": None,
}

_ENV_DISABLE = "PTCG_DISABLE_RISK_DET"

# エンジンからは取得できない「残りゲーム時間」の近似に使う総予算(要件書 §2.1「1試合10分」)。
_TOTAL_MATCH_BUDGET_MS = 600_000.0

_match_start_time: float | None = None
_last_seen_turn: int | None = None

_value_model: ValueModel | None = None


def _get_value_model() -> ValueModel:
    global _value_model
    if _value_model is None:
        _value_model = ValueModel()
    return _value_model


# ------------------------------------------------------------------
# 計測(§5.9)

_STATS_ZERO: dict = {
    "invocations": 0,      # search() が呼ばれた回数
    "fired": 0,            # ゲートを通り、評価まで実施した回数
    "overrides": 0,        # ルール第1候補と異なる手を採用した回数
    "truncated": 0,        # 予算超過でdeterminization回数が減った回数
    "evaluations_total": 0,  # 実施した (candidate x determinization) 評価回数の累計
}


def _fresh_stats() -> dict:
    stats = dict(_STATS_ZERO)
    stats["reject_reasons"] = {}
    stats["override_score_deltas"] = []
    stats["elapsed_ms"] = []
    return stats


_stats: dict = _fresh_stats()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def get_stats() -> dict:
    """累計統計を返す（``lethal_simple.get_stats()`` と同じ形。§5.9）。"""
    stats = dict(_stats)
    stats["reject_reasons"] = dict(_stats["reject_reasons"])
    elapsed = _stats["elapsed_ms"]
    stats["p50_ms"] = _percentile(elapsed, 0.50)
    stats["p95_ms"] = _percentile(elapsed, 0.95)
    stats["p99_ms"] = _percentile(elapsed, 0.99)
    deltas = _stats["override_score_deltas"]
    stats["avg_override_score_delta"] = (sum(deltas) / len(deltas)) if deltas else 0.0
    return stats


def reset_stats() -> None:
    global _stats
    _stats = _fresh_stats()


def _reject(reason: str) -> None:
    _stats["reject_reasons"][reason] = _stats["reject_reasons"].get(reason, 0) + 1
    return None


def _remaining_game_ms(state: State) -> float:
    """試合開始からの経過時間を差し引いた、残りゲーム時間の近似値(モジュールdocstring参照)。"""
    global _match_start_time, _last_seen_turn
    now = time.perf_counter()
    turn = state.turn
    if _match_start_time is None or (turn is not None and _last_seen_turn is not None and turn < _last_seen_turn):
        _match_start_time = now
    _last_seen_turn = turn
    elapsed_ms = (now - _match_start_time) * 1000.0
    return _TOTAL_MATCH_BUDGET_MS - elapsed_ms


def _normalize(scores: list[float]) -> list[float]:
    """ルールスコアを [0, 1] へ min-max 正規化する（全て同値なら 0.5 で埋める）。"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo <= 0.0:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _prize_diff(state: State) -> int:
    """``adaptive`` モード用のサイド差（自分の残サイド − 相手の残サイド）。負なら優勢。"""
    me = state.yourIndex
    own = len(state.players[me].prize)
    opp = len(state.players[1 - me].prize)
    return own - opp


def _evaluate_candidate(
    obs: Observation, hidden_state: dict, action: list[int], value_model: ValueModel
) -> float | None:
    """1決定化サンプル上で候補手を1手適用し、結果局面の勝率を返す（例外時は None）。"""
    try:
        root = cg_api.search_begin(
            obs,
            hidden_state["your_deck"],
            hidden_state["your_prize"],
            hidden_state["opponent_deck"],
            hidden_state["opponent_prize"],
            hidden_state["opponent_hand"],
            hidden_state["opponent_active"],
        )
    except Exception:
        return None
    try:
        try:
            child = cg_api.search_step(root.searchId, action)
        except Exception:
            return None
        try:
            return value_model.predict_win_prob_from_state(child.observation.current)
        except Exception:
            return None
        finally:
            try:
                cg_api.search_release(child.searchId)
            except Exception:
                pass
    finally:
        try:
            cg_api.search_release(root.searchId)
        except Exception:
            pass


def search(state: State, legal_actions: list, context: dict) -> list[int] | None:
    """候補手をN決定化上で評価し、リスク集約スコアで並べ替えた最良手を返す（§5.2）。"""
    entry_time = time.perf_counter()
    try:
        return _search_impl(state, legal_actions, context, entry_time)
    except Exception:
        return _reject("exception")
    finally:
        _stats["elapsed_ms"].append((time.perf_counter() - entry_time) * 1000.0)
        try:
            cg_api.search_end()
        except Exception:
            pass


def _search_impl(
    state: State, legal_actions: list, context: dict, entry_time: float
) -> list[int] | None:
    _stats["invocations"] += 1

    if os.environ.get(_ENV_DISABLE, "0") == "1":
        return _reject("env_disabled")

    config = {**DEFAULTS, **(context.get("config") or {})}
    if not config.get("enabled", False):
        return _reject("not_enabled")

    obs: Observation | None = context.get("observation")
    if obs is None or obs.select is None or obs.current is None:
        return _reject("no_observation")
    if state is None:
        state = obs.current
    if obs.select.type != SelectType.MAIN:
        return _reject("not_main")

    turn = state.turn
    if turn is None or turn < int(config["min_turn"]):
        return _reject("turn_too_early")

    candidate_provider = context.get("candidate_provider")
    if candidate_provider is None:
        return _reject("no_candidate_provider")
    try:
        all_candidates = list(candidate_provider() or [])
    except Exception:
        return _reject("candidate_provider_error")
    if len(all_candidates) < 2:
        return _reject("insufficient_candidates")

    try:
        opponent_state = match_context.get_opponent_state()
    except Exception:
        opponent_state = None
    if opponent_state is None or not getattr(opponent_state, "is_ready", False):
        return _reject("opponent_not_ready")

    value_model = _get_value_model()
    if not value_model.is_ready:
        return _reject("value_not_ready")

    if _remaining_game_ms(state) < float(config["min_remaining_game_ms"]):
        return _reject("time_budget")

    hidden_state_factory: Callable[[], dict | None] | None = context.get("hidden_state_factory")
    if hidden_state_factory is None:
        return _reject("no_hidden_state_factory")

    time_limit_ms = float(config["time_limit_ms"])
    deadline_ms = context.get("deadline_ms")
    if deadline_ms is not None:
        time_limit_ms = min(time_limit_ms, float(deadline_ms))
    deadline = entry_time + time_limit_ms / 1000.0

    top_k = max(1, int(config["top_k"]))
    candidates = all_candidates[:top_k]

    determinizations = max(1, int(config["determinizations"]))
    max_evaluations = max(1, int(config["max_evaluations"]))

    values_by_candidate: list[list[float]] = [[] for _ in candidates]
    evaluations_done = 0
    truncated = False

    for _n in range(determinizations):
        if time.perf_counter() > deadline or evaluations_done >= max_evaluations:
            truncated = True
            break
        try:
            hidden_state = hidden_state_factory()
        except Exception:
            hidden_state = None
        if hidden_state is None:
            truncated = True
            break

        stop = False
        for idx, (action, _rule_score) in enumerate(candidates):
            if time.perf_counter() > deadline or evaluations_done >= max_evaluations:
                truncated = True
                stop = True
                break
            value = _evaluate_candidate(obs, hidden_state, action, value_model)
            evaluations_done += 1
            if value is not None:
                values_by_candidate[idx].append(value)
        if stop:
            break

    if truncated:
        _stats["truncated"] += 1
    _stats["evaluations_total"] += evaluations_done

    if all(not values for values in values_by_candidate):
        return _reject("no_samples")

    prize_diff = _prize_diff(state)
    mode = config["mode"]
    beta = float(config["beta"])
    alpha = float(config["alpha"])
    risk_weight = float(config["risk_weight"])

    normalized_rule_scores = _normalize([score for _, score in candidates])

    risk_scores: list[float | None] = []
    best_idx: int | None = None
    best_final_score: float | None = None
    for idx, values in enumerate(values_by_candidate):
        if not values:
            risk_scores.append(None)
            continue
        risk_score = risk_aggregation.aggregate(
            values, mode=mode, beta=beta, alpha=alpha, prize_diff=prize_diff
        )
        risk_scores.append(risk_score)
        final_score = (1.0 - risk_weight) * normalized_rule_scores[idx] + risk_weight * risk_score
        if best_final_score is None or final_score > best_final_score:
            best_final_score = final_score
            best_idx = idx

    if best_idx is None:
        return _reject("no_valid_candidate")

    _stats["fired"] += 1
    if best_idx == 0:
        # ルール第1候補と同じ手を選んだ = 上書きなし。呼び出し側の挙動は素のrouterと同じになるため
        # None を返してよい(§5.2 手順6)。
        return None

    _stats["overrides"] += 1
    if risk_scores[0] is not None and risk_scores[best_idx] is not None:
        _stats["override_score_deltas"].append(risk_scores[best_idx] - risk_scores[0])
    return candidates[best_idx][0]
