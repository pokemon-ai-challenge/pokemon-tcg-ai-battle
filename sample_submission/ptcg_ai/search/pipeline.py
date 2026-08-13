"""意思決定パイプライン統合オーケストレータ(Step2〜5)。

`docs/plans/decision-pipeline/design-and-implementation-plan.md` の設計に対応。
チーム共通の探索インターフェース ``search(state, legal_actions, context) -> list[int] | None``
(``lethal_simple``/``pimc`` と同一シグネチャ)を実装し、`ml_policy_agent` から config-gated で
呼ばれる。既定(config に ``pipeline`` キーが無い本番 config)では一切呼ばれない。

処理の流れ(前段で決まれば後段はスキップ):
  Step2  Policy で ``obs.select.option`` をスコアリング → 上位 k の first-move 候補に絞る。
         top1 が ``top1_shortcut_prob`` 以上に集中していれば即 top1(自明手の高速化)。
  Step3  ``context["hidden_state_factory"]`` を N 回呼んで N 決定化(``pimc`` と同じ流用点)。
  Step4  各(世界, 候補)で ``search_begin`` → 候補 first move を適用 → 自ターンの残りを Policy 貪欲で
         展開 → 相手ターンを ``opponent_depth`` 回 Policy 貪欲(=相手モデル)で展開 →
         ``leaf_evaluator.evaluate(leaf_state, me)`` で末端評価。
  Step5  候補ごとに N 世界平均、最大を採用。差が ``tie_eps`` 以下なら Policy スコア上位を採用。

適用範囲: ``SelectType.MAIN`` かつ ``maxCount == 1`` の意思決定点のみ。それ以外は None を返し、
呼び出し側(`ml_policy_agent`)の既存経路(Policy top1 / 貪欲マルチ選択)にフォールバックする
(Step2 の学習スコープと同じ制約。step2-design.md §2.4)。

``pimc.py`` と同様、``lethal_simple`` は import せず、小さな検証・候補生成ヘルパーは
チーム慣例に従って複製する(search 近傍モジュール間で共有しない)。相手ターンをモデル化する点
(Policy を相手視点で適用)と末端評価を差し替え可能にする点が ``pimc`` との違い。
"""

from __future__ import annotations

import math
import time
from typing import Callable

from cg import api as cg_api
from cg.api import Observation, OptionType, SelectData, SelectType, State

from ptcg_ai.search import leaf_eval as leaf_eval_module

DEFAULTS: dict = {
    "enabled": False,
    "module": "pipeline",
    "top_k": 4,
    "top1_shortcut_prob": 0.9,
    "num_determinizations": 8,
    "opponent_depth": 1,        # 相手ターンを何回展開してから末端評価するか(1〜2)
    "max_rollout_steps": 40,    # 1 決定化・1 候補あたりの search_step 上限(暴走防止)
    "time_limit_ms": 400,       # search() 全体の壁時計予算
    "tie_eps": 0.02,            # 平均スコア差がこれ以下なら Policy 上位で決める
    "leaf_eval": {"kind": "handcrafted"},
    # 模倣が低評価しがちな手(特性使用・エネ付与など)を Policy top-k に関係なく候補へ
    # 含めるための OptionType 名リスト(例: ["ABILITY","ATTACH"])。既定 [] = 従来挙動。
    # デッキ非依存: 「模倣が見落とす強い手ほど探索対象から外れる」構造穴を塞ぐための口。
    "extra_candidate_types": [],
    # extra で追加する候補の上限(policy top-k と合わせた総数はこれで頭打ち)。
    "max_candidates": 8,
}


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    if total <= 0:
        n = len(scores)
        return [1.0 / n] * n
    return [e / total for e in exps]


def _is_my_turn(state: State, me: int) -> bool:
    if state.turn < 1 or state.firstPlayer < 0:
        return False
    starter_turn = state.turn % 2 == 1
    return starter_turn == (state.firstPlayer == me)


def _is_legal_selection(selection, select: SelectData) -> bool:
    if not isinstance(selection, list):
        return False
    if not (select.minCount <= len(selection) <= select.maxCount):
        return False
    if len(selection) != len(set(selection)):
        return False
    return all(
        isinstance(index, int) and 0 <= index < len(select.option)
        for index in selection
    )


def _begin(obs: Observation, hidden_state: dict):
    return cg_api.search_begin(
        obs,
        hidden_state["your_deck"],
        hidden_state["your_prize"],
        hidden_state["opponent_deck"],
        hidden_state["opponent_prize"],
        hidden_state["opponent_hand"],
        hidden_state["opponent_active"],
    )


def _hidden_state_factory(context: dict) -> Callable[[], dict | None] | None:
    factory = context.get("hidden_state_factory")
    if factory is not None:
        return factory
    hidden_state = context.get("hidden_state")
    if hidden_state is not None:
        return lambda: hidden_state
    return None


def _select_candidate_indices(select: SelectData, ranked: list[int], config: dict) -> list[int]:
    """探索する first-move 候補のインデックス集合を返す。

    Policy スコア上位 ``top_k`` に加え、``extra_candidate_types``(OptionType 名)に該当する
    合法手を Policy ランクに関係なく含める(模倣が低評価する特性使用・エネ付与などを探索から
    外さないため)。総数は ``max_candidates`` で頭打ち。``ranked`` はスコア降順の全インデックス。
    """
    k = max(1, int(config.get("top_k", 4)))
    indices = list(ranked[:k])
    extra_types = config.get("extra_candidate_types") or []
    if extra_types:
        wanted = {getattr(OptionType, t, None) for t in extra_types}
        wanted.discard(None)
        for i in ranked:  # スコア順で走査し、該当タイプを追記(重複は除く)
            if select.option[i].type in wanted and i not in indices:
                indices.append(i)
    max_candidates = int(config.get("max_candidates", 8))
    if max_candidates > 0:
        indices = indices[:max_candidates]
    return indices


def _greedy_selection(policy_model, obs: Observation) -> list[int]:
    """Policy スコア最大の合法選択を返す(自ターン継続・相手ターンの両方に使う)。

    ``score_options_from_state`` は consequence 特徴(``meta.consequence_fields``)を持つ重みで
    例外を投げる設計。その場合や未ロード時は先頭の合法選択にフォールバックする(rollout の質は
    落ちるが探索は止めない)。
    """
    select = obs.select
    if select is None or not select.option:
        return []
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))
    try:
        scores = policy_model.score_options_from_state(obs.current, select)
    except Exception:
        scores = []
    if not scores or len(scores) != n:
        return list(range(count))
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return ranked[:count]


def _rollout_and_eval(node, me: int, config: dict, deadline: float, evaluator, policy_model) -> float | None:
    """候補 first move 適用後のノードから Policy 貪欲で展開し、末端評価を返す。

    停止条件:
      - 決着(``state.result != -1``)→ その盤面を評価。
      - 相手ターンを ``opponent_depth`` 回終えて自分の手番に戻った時点で評価。
      - 選択肢が無い / ``max_rollout_steps`` 到達 / 予算切れ → 現盤面を評価。
    """
    opponent_depth = max(1, int(config["opponent_depth"]))
    max_steps = int(config["max_rollout_steps"])
    prev_actor = me
    opp_turns = 0

    for _ in range(max_steps):
        if time.perf_counter() > deadline:
            break
        obs = node.observation
        state = obs.current
        if state is None:
            return None
        if state.result != -1:
            return evaluator.evaluate(state, me)

        actor = state.yourIndex
        if prev_actor == me and actor != me:
            opp_turns += 1
        if actor == me and prev_actor != me and opp_turns >= opponent_depth:
            return evaluator.evaluate(state, me)

        if obs.select is None or not obs.select.option:
            return evaluator.evaluate(state, me)

        selection = _greedy_selection(policy_model, obs)
        if not selection:
            return evaluator.evaluate(state, me)
        try:
            node = cg_api.search_step(node.searchId, selection)
        except ValueError:
            return evaluator.evaluate(state, me)
        prev_actor = actor

    # 予算/step 超過: 到達した最新盤面を評価。
    state = node.observation.current
    return evaluator.evaluate(state, me) if state is not None else None


def _evaluate_candidate(root, candidate: list[int], me: int, config: dict, deadline: float, evaluator, policy_model) -> float | None:
    """1 決定化・1 候補の末端スコア。候補がこの決定化で違法なら None。"""
    try:
        child = cg_api.search_step(root.searchId, candidate)
    except ValueError:
        return None
    try:
        return _rollout_and_eval(child, me, config, deadline, evaluator, policy_model)
    finally:
        try:
            cg_api.search_release(child.searchId)
        except Exception:
            pass


def search(state: State, legal_actions: list, context: dict) -> list[int] | None:
    """パイプライン本体。適用外/失敗/予算切れ前に評価不能なら None(呼び出し側が top1 へ)。"""
    try:
        config = {**DEFAULTS, **(context.get("config") or {})}
        if not config["enabled"]:
            return None

        obs: Observation | None = context.get("observation")
        if obs is None or obs.select is None or obs.current is None:
            return None
        select = obs.select
        # 適用範囲: 単一選択の MAIN 意思決定のみ。
        if select.type != SelectType.MAIN or select.maxCount != 1 or not select.option:
            return None
        if state is None:
            state = obs.current
        me = state.yourIndex
        if state.result != -1 or not _is_my_turn(state, me):
            return None

        policy_model = context.get("policy_model")
        if policy_model is None:
            return None

        start = time.perf_counter()
        deadline = start + config["time_limit_ms"] / 1000.0

        # Step2: Policy スコア → top-k first-move 候補。
        try:
            scores = policy_model.score_options(obs, context.get("model_hidden_state_factory"), deadline)
        except Exception:
            scores = []
        if not scores or len(scores) != len(select.option):
            return None
        probs = _softmax(scores)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # top1 集中なら自明手として即返し(探索スキップ)。
        if probs[ranked[0]] >= config["top1_shortcut_prob"]:
            top1 = [ranked[0]]
            return top1 if _is_legal_selection(top1, select) else None

        candidate_indices = _select_candidate_indices(select, ranked, config)
        # デッキ専用の「戦略候補」注入口(既定 None = 何もしない)。extra_candidate_types
        # (OptionType 全件を無条件に足す)と違い、呼び出し側が1件だけ厳選して追加できる。
        # 例: オーガポンの控えアタッカーへの手貼りのうち最も価値の高い1件だけを、
        # Policy top-k に入っていなくても候補へ加える(Policy top-4 + 戦略候補1件 = 最大5件)。
        extra_fn = context.get("extra_candidate_indices_fn")
        if extra_fn is not None:
            try:
                for i in (extra_fn(obs, select, candidate_indices) or []):
                    if isinstance(i, int) and 0 <= i < len(select.option) and i not in candidate_indices:
                        candidate_indices.append(i)
            except Exception:
                pass
        max_candidates = int(config.get("max_candidates", 8))
        if max_candidates > 0:
            candidate_indices = candidate_indices[:max_candidates]
        candidates = [[i] for i in candidate_indices]

        evaluator = context.get("leaf_evaluator") or leaf_eval_module.build_evaluator(config.get("leaf_eval"))
        factory = _hidden_state_factory(context)
        if factory is None:
            return None

        # Step3+4: N 決定化 × 各候補を先読み評価。
        aggregate: dict[int, list[float]] = {i: [] for i in candidate_indices}
        num_worlds = max(1, int(config["num_determinizations"]))
        try:
            for _ in range(num_worlds):
                if time.perf_counter() > deadline:
                    break
                hidden_state = factory()
                if hidden_state is None:
                    continue
                try:
                    root = _begin(obs, hidden_state)
                except Exception:
                    continue
                try:
                    for idx in candidate_indices:
                        if time.perf_counter() > deadline:
                            break
                        s = _evaluate_candidate(root, [idx], me, config, deadline, evaluator, policy_model)
                        if s is not None:
                            aggregate[idx].append(s)
                finally:
                    try:
                        cg_api.search_release(root.searchId)
                    except Exception:
                        pass
        finally:
            try:
                cg_api.search_end()
            except Exception:
                pass

        scored = {i: (sum(v) / len(v)) for i, v in aggregate.items() if v}
        if not scored:
            return None

        # Step5': デッキ専用のリソース評価を **PIMC スコアを捨てずに** 合成する。
        # 既定では bonus_fn が無いので何も起きない(従来挙動そのまま)。
        # 「にげる」と「現在のポケモンで続行」が同じ scored の上で比較されるため、
        # 二重介入(MAIN を別ロジックで上書きする)にはならない。
        bonus_fn = context.get("candidate_bonus_fn")
        shadow = {}
        if bonus_fn is not None:
            try:
                pimc_only_best = max(scored, key=lambda i: scored[i])
                bonuses = bonus_fn(obs, me, list(scored.keys())) or {}
                # スケール確認用の内訳。retreat 候補と非 retreat 候補を分けて記録する。
                retreat_idx = [i for i in scored if i in bonuses]
                non_retreat = [i for i in scored if i not in bonuses]
                best_non_retreat = max((scored[i] for i in non_retreat), default=None)
                best_retreat = max((scored[i] for i in retreat_idx), default=None)
                # planner_scores = PIMCスコア + bonus。shadow_only の値に関わらず**常に**計算する
                # (「PIMC評価を再実行せず、既に得た scored からOFF/ON両方の判断を導出する」ため。
                # 実際にどちらを最終行動として採用するかは shadow_only で分岐する後段のみで決まる)。
                planner_scores = {i: scored[i] + float(bonuses.get(i, 0.0)) for i in scored}
                planner_decision = max(planner_scores, key=lambda i: planner_scores[i])

                def _identity(opt):
                    return (int(getattr(opt, "type", -1)), getattr(opt, "area", None),
                           getattr(opt, "index", None), getattr(opt, "inPlayArea", None),
                           getattr(opt, "inPlayIndex", None), getattr(opt, "playerIndex", None),
                           getattr(opt, "cardId", None), getattr(opt, "serial", None),
                           getattr(opt, "attackId", None))

                shadow = {
                    "pimc_scores": dict(scored), "bonuses": dict(bonuses),
                    "pimc_only_best": pimc_only_best,
                    "planner_scores": dict(planner_scores),
                    "planner_decision": planner_decision,
                    "best_non_retreat_pimc_score": best_non_retreat,
                    "retreat_pimc_score": best_retreat,
                    "pimc_margin": (None if (best_non_retreat is None or best_retreat is None)
                                    else best_non_retreat - best_retreat),
                    "planner_raw_bonus": (max(bonuses.values()) if bonuses else None),
                    "bonus_applied": not config.get("resource_bonus_shadow_only", False),
                    "candidate_option_types": {i: int(getattr(select.option[i], "type", -1))
                                               for i in scored if i < len(select.option)},
                    "candidate_option_identities": {i: _identity(select.option[i])
                                                    for i in scored if i < len(select.option)},
                    "all_option_types": [int(getattr(o, "type", -1)) for o in select.option],
                    "extra_candidate_types_cfg": config.get("extra_candidate_types"),
                }
                if not config.get("resource_bonus_shadow_only", False):
                    for i, b in bonuses.items():
                        if i in scored:
                            scored[i] += float(b)
                shadow["final_scores"] = dict(scored)
            except Exception:
                shadow = {}

        # Step5: 平均最大を採用。tie_eps 以内は Policy 確率上位で決める。
        best_mean = max(scored.values())
        tie_eps = float(config["tie_eps"])
        contenders = [i for i, m in scored.items() if best_mean - m <= tie_eps]
        best_idx = max(contenders, key=lambda i: probs[i])

        sink = context.get("shadow_sink")
        if sink is not None and shadow:
            try:
                shadow["chosen"] = best_idx
                shadow["selection_flipped"] = (best_idx != shadow.get("pimc_only_best"))
                sink(shadow)
            except Exception:
                pass

        best = [best_idx]
        return best if _is_legal_selection(best, select) else None
    except Exception:
        return None
