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
    # anytime 決定化スケジューリング(dict, 既定 None=OFF)。truthy かつ enabled のときだけ
    # Step3+4 のループを「ラウンド方式」に置き換える(_ANYTIME_DEFAULTS 参照)。キーが無ければ
    # 従来の固定 N 決定化ループを一行も変えずに使う。
    "anytime": None,
}

# anytime ラウンド方式の既定値(config["anytime"] で上書き)。
#
# 従来ループ(world 外側 × 候補 内側)は deadline 打切り時に「先頭候補ほど多くの world で
# 評価される」偏りを持つ(実測でも 8 決定化設定に対し 1.7 本しか完了していない)。ラウンド方式は
#   * 1 ラウンド = 決定化 world 1 本 × **全候補**
#   * 途中で予算切れ/シミュ失敗したラウンドは「不完全」として集計から丸ごと捨てる
#   * ラウンドごとに候補順を回転(rotate_candidates)して残る偏りも消す
# ことで、いつ打ち切られても「全候補が同じ world 数で比較されている」状態を保つ(anytime 性)。
_ANYTIME_DEFAULTS: dict = {
    "enabled": True,
    "max_rounds": 16,                 # ラウンド上限(従来の num_determinizations の代替)
    "min_rounds_for_early_stop": 3,   # これ未満の完全ラウンド数では早期打切りしない
    "early_stop_margin": 0.04,        # 1位-2位の平均差がこれ超なら(1位不変を条件に)打切り
    "rotate_candidates": True,        # ラウンド r では候補順を r だけ回転して順序偏りを除去
}

# anytime パスの計装(モジュールレベルの純カウンタ)。**意思決定には一切使わない**。
# 「1決定あたり何ラウンド完走できたか」「どれだけ捨てているか」を後段のランナー
# (`kaggle_replays/challengers/anytime_h2h.py` 等)が読むための観測点。anytime が
# 無効な経路(従来ループ)では一度も触らないので本番挙動・コストは不変。
#   decisions             : anytime ラウンドを回した意思決定点の数
#   complete_rounds_total : 完全ラウンド(全候補が揃った world)の総数
#   rounds_per_decision   : 決定ごとの完全ラウンド数(上限 _ROUNDS_PER_DECISION_CAP 件で頭打ち)
#   deadline_discard      : 予算切れで捨てた不完全ラウンド数
#   simulation_discard    : 決定化失敗/違法手など予算以外の理由で捨てたラウンド数
#   zero_complete_fallback: 完全ラウンド0本で終わった決定(=呼び出し側が Policy top1 へ落ちる)
#   early_stops           : 早期打切りが発火した決定
ANYTIME_STATS: dict = {
    "decisions": 0,
    "complete_rounds_total": 0,
    "rounds_per_decision": [],
    "deadline_discard": 0,
    "simulation_discard": 0,
    "zero_complete_fallback": 0,
    "early_stops": 0,
}

# rounds_per_decision の保持上限(長時間ランでメモリが際限なく増えないための頭打ち)。
_ROUNDS_PER_DECISION_CAP = 100_000


def reset_anytime_stats() -> None:
    """``ANYTIME_STATS`` を初期状態へ戻す(試合単位の計測はこれを起点にする)。"""
    ANYTIME_STATS["decisions"] = 0
    ANYTIME_STATS["complete_rounds_total"] = 0
    ANYTIME_STATS["rounds_per_decision"] = []
    ANYTIME_STATS["deadline_discard"] = 0
    ANYTIME_STATS["simulation_discard"] = 0
    ANYTIME_STATS["zero_complete_fallback"] = 0
    ANYTIME_STATS["early_stops"] = 0


def _notify_expensive_root(context: dict) -> None:
    """``context["on_expensive_root"]`` があれば1回だけ呼ぶ(無ければ何もしない)。

    呼び出し側(`ml_policy_agent`)の時間予算v2は「**重い探索が実際に走った回数**」で残り時間を
    割る。適用外 select や top1 shortcut で即 return する安い呼び出しまで数えると分母が
    水増しされ1手予算が過小になるため、「決定化評価(Step3)へ本当に入った」この位置から
    コールバックで通知する。コールバックの例外は握り潰す(計測が意思決定を止めてはならない)。
    """
    callback = context.get("on_expensive_root")
    if callback is None:
        return
    try:
        callback()
    except Exception:  # noqa: BLE001 - 計測の失敗が探索を止めてはならない
        pass


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


# dynamic_top_k の既定しきい値(選択肢数 -> 候補数)。`_diag_topk_miss.py`(n=46, 2026-08-06)
# で測った「探索最良手が top-4 外に出る率」の帯別内訳に対応させてある:
#   選択肢 5-7: 12.5% / 8-11: 37.5% / 12+: 50.0%
# 見落としが多い帯だけ候補を広げ、少ない帯は従来どおり 4 のままにする。
_DEFAULT_NOPTIONS_THRESHOLDS: list[dict] = [
    {"max_options": 7, "top_k": 4},
    {"max_options": 11, "top_k": 8},
    {"max_options": None, "top_k": 12},   # max_options=None = 上限なし(最後に置く)
]

# NOTE (2026-08-06 diagnostics):
# Current diagnostics found no useful miss-prediction signal from policy
# confidence/entropy. High-confidence states still had substantial top-4
# miss rates. Do not enable confidence-gated dynamic top-k without new evidence.


def _resolve_top_k(config: dict, probs: list[float] | None, ranked: list[int]) -> int:
    """この局面で使う候補数 k を返す。``1 <= k <= n_options`` を必ず満たす。

    既定(``dynamic_top_k`` キー無し / ``enabled`` が偽)は固定 ``top_k`` = 従来挙動そのまま。

    ``mode``:
      - ``"n_options"``(推奨): **選択肢数**でしきい値表を引く。実測で見落とし率が
        選択肢数に対して単調(5-7:12.5% / 8-11:37.5% / 12+:50.0%)だったため、
        候補を広げるべき局面をこれで特定する。しきい値は config から変更可能。
      - ``"confidence"``(**非推奨**): Policy の確信度で k を変える旧実装。
        実測では確信度に見落とし予測力が無く(>=0.6 でも 26.9% 見落とし)、
        このゲートは「見落としている局面をむしろ狭める」方向に働く。
        履歴保持のため残すが、**新しい根拠なしに有効化しないこと**。

    ``n_options`` は ``len(ranked)`` から取る(呼び出し側が全選択肢の降順ランクを渡す)。
    """
    n_options = len(ranked) if ranked else 0
    base_k = max(1, int(config.get("top_k", 4)))
    if n_options <= 0:
        return base_k                      # 通常起きない。呼び出し側が空集合を扱う。

    cfg = config.get("dynamic_top_k") or {}
    if not cfg.get("enabled", False):
        return max(1, min(base_k, n_options))

    mode = cfg.get("mode", "n_options")
    max_permitted = int(cfg.get("max_top_k", 0) or 0)

    if mode == "n_options":
        thresholds = cfg.get("thresholds") or _DEFAULT_NOPTIONS_THRESHOLDS
        k = base_k
        for row in thresholds:
            limit = row.get("max_options")
            if limit is None or n_options <= int(limit):
                k = int(row.get("top_k", base_k))
                break
        else:
            # どの帯にも当たらない(上限なし行が無い)設定は、最後の行の top_k を使う。
            if thresholds:
                k = int(thresholds[-1].get("top_k", base_k))
    elif mode == "confidence":
        # 非推奨経路(上の NOTE 参照)。互換のため残す。
        if not probs:
            return max(1, min(base_k, n_options))
        min_k = max(1, int(cfg.get("min_k", 3)))
        max_k = max(min_k, int(cfg.get("max_k", 12)))
        hi = float(cfg.get("confident_prob", 0.6))
        lo = float(cfg.get("uncertain_prob", 0.3))
        p = probs[ranked[0]]
        if hi <= lo:
            k = base_k
        elif p >= hi:
            k = min_k
        elif p <= lo:
            k = max_k
        else:
            t = (hi - p) / (hi - lo)
            k = int(round(min_k + t * (max_k - min_k)))
    else:
        k = base_k                          # 未知 mode は安全側(従来固定)へ倒す

    if max_permitted > 0:
        k = min(k, max_permitted)
    return max(1, min(int(k), n_options))


def _select_candidate_indices(select: SelectData, ranked: list[int], config: dict,
                              probs: list[float] | None = None) -> list[int]:
    """探索する first-move 候補のインデックス集合を返す。

    Policy スコア上位 ``k`` 件(``_resolve_top_k``: 既定は固定 ``top_k``、``dynamic_top_k``
    有効時は確信度依存)に加え、``extra_candidate_types``(OptionType 名)に該当する
    合法手を Policy ランクに関係なく含める(模倣が低評価する特性使用・エネ付与などを探索から
    外さないため)。総数は ``max_candidates`` で頭打ち。``ranked`` はスコア降順の全インデックス。
    """
    k = _resolve_top_k(config, probs, ranked)
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


def _round_means(rounds: list[dict[int, float]], candidate_indices: list[int]) -> dict[int, float]:
    """完全ラウンド群の候補別平均。``rounds`` は全候補のスコアが揃ったラウンドのみ。"""
    n = len(rounds)
    if n <= 0:
        return {}
    return {i: sum(r[i] for r in rounds) / n for i in candidate_indices}


def _should_early_stop(n_complete: int, leaders: list[int], means: dict[int, float],
                       min_rounds: int, margin: float) -> bool:
    """anytime の早期打切り判定。

    条件(すべて満たすとき打切り):
      - 完全ラウンド数が ``min_rounds_for_early_stop`` 以上
      - **直近2完全ラウンド**の(その時点までの平均での)1位候補が同一
      - 1位平均 - 2位平均 > ``early_stop_margin``
    """
    if n_complete < max(1, int(min_rounds)):
        return False
    if len(leaders) < 2 or leaders[-1] != leaders[-2]:
        return False
    ordered = sorted(means.values(), reverse=True)
    if len(ordered) < 2:
        return True                     # 候補が1つなら比較相手が無い=これ以上回す意味が無い
    return (ordered[0] - ordered[1]) > float(margin)


def _run_anytime_rounds(obs: Observation, me: int, candidate_indices: list[int], config: dict,
                        anytime_config: dict, deadline: float, evaluator, policy_model,
                        factory: Callable[[], dict | None]) -> dict[int, float]:
    """Step3+4 の anytime ラウンド版。候補別平均(完全ラウンドのみ)を返す。

    1 ラウンド = 決定化 world を 1 本作り、その world で **全候補** を評価する。
      - ラウンド開始前に予算超過していれば開始しない。
      - ラウンド内で予算超過して未評価の候補が残ったら、そのラウンドは不完全として捨て、
        ループを終える(打切りは常に「候補間で公平な集計」を残す)。
      - ``_evaluate_candidate`` が None(この world で違法/シミュ失敗)の場合は、予算超過が
        原因でなければそのラウンドだけ捨てて **次のラウンドへ続行** する。
      - 完全ラウンドが 0 本なら空 dict を返す(呼び出し側は None → Policy top1 へフォールバック)。

    副作用として ``ANYTIME_STATS`` にラウンド統計を積む(カウンタ加算のみ。返り値=意思決定には
    一切影響しない)。
    """
    ac = {**_ANYTIME_DEFAULTS, **(anytime_config or {})}
    n_cand = len(candidate_indices)
    if n_cand <= 0:
        return {}
    max_rounds = max(1, int(ac.get("max_rounds", 16)))
    min_rounds_es = int(ac.get("min_rounds_for_early_stop", 3))
    margin = float(ac.get("early_stop_margin", 0.04))
    rotate = bool(ac.get("rotate_candidates", True))

    ANYTIME_STATS["decisions"] += 1
    complete: list[dict[int, float]] = []
    leaders: list[int] = []
    means: dict[int, float] = {}
    try:
        for r in range(max_rounds):
            if time.perf_counter() > deadline:
                break                   # ラウンド未開始 = 破棄ではない(集計しない)。
            hidden_state = factory()
            if hidden_state is None:
                ANYTIME_STATS["simulation_discard"] += 1
                continue                # world が作れない = 不完全ラウンド。次へ。
            try:
                root = _begin(obs, hidden_state)
            except Exception:
                ANYTIME_STATS["simulation_discard"] += 1
                continue
            order = candidate_indices
            if rotate and n_cand > 1:
                offset = r % n_cand
                order = candidate_indices[offset:] + candidate_indices[:offset]
            round_scores: dict[int, float] = {}
            stop = False                # 予算切れ由来の打切り(ループ自体を終える)
            try:
                for idx in order:
                    if time.perf_counter() > deadline:
                        stop = True
                        break
                    s = _evaluate_candidate(root, [idx], me, config, deadline, evaluator, policy_model)
                    if s is None:
                        # 予算切れならループ終了、そうでなければこのラウンドを捨てて次へ。
                        stop = time.perf_counter() > deadline
                        break
                    round_scores[idx] = s
            finally:
                try:
                    cg_api.search_release(root.searchId)
                except Exception:
                    pass
            if len(round_scores) == n_cand:
                complete.append(round_scores)
                means = _round_means(complete, candidate_indices)
                # 平均1位(同点は candidate_indices の順=Policy 上位優先で決定的)。
                leaders.append(max(candidate_indices, key=lambda i: means[i]))
                if _should_early_stop(len(complete), leaders, means, min_rounds_es, margin):
                    ANYTIME_STATS["early_stops"] += 1
                    break
            else:
                # 不完全ラウンド(捨てる)。予算切れ由来かシミュ失敗由来かを分けて数える。
                if stop:
                    ANYTIME_STATS["deadline_discard"] += 1
                else:
                    ANYTIME_STATS["simulation_discard"] += 1
            if stop:
                break
    finally:
        try:
            cg_api.search_end()
        except Exception:
            pass
        # 計測(意思決定に不影響)。例外で抜けた場合も必ず記録する。
        n_complete = len(complete)
        ANYTIME_STATS["complete_rounds_total"] += n_complete
        if len(ANYTIME_STATS["rounds_per_decision"]) < _ROUNDS_PER_DECISION_CAP:
            ANYTIME_STATS["rounds_per_decision"].append(n_complete)
        if n_complete == 0:
            ANYTIME_STATS["zero_complete_fallback"] += 1
    return means


def _pick_best_index(scored: dict[int, float], probs: list[float], tie_eps: float) -> int:
    """Step5: 平均最大を採用。差が ``tie_eps`` 以内の候補は Policy 確率上位で決める。"""
    best_mean = max(scored.values())
    contenders = [i for i, m in scored.items() if best_mean - m <= tie_eps]
    return max(contenders, key=lambda i: probs[i])


def search(state: State, legal_actions: list, context: dict) -> list[int] | None:
    """パイプライン本体。適用外/失敗/予算切れ前に評価不能なら None(呼び出し側が top1 へ)。

    ``context`` の任意キー ``on_expensive_root``(callable, 既定なし)を渡すと、**決定化評価
    (Step3)へ実際に入るときだけ1回**呼ぶ。呼び出し側の時間予算v2が「重い探索の回数」で
    残り時間を割るための通知点(``_notify_expensive_root``)。渡さなければ従来どおり何もしない。
    """
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

        candidate_indices = _select_candidate_indices(select, ranked, config, probs)
        candidates = [[i] for i in candidate_indices]

        evaluator = context.get("leaf_evaluator") or leaf_eval_module.build_evaluator(config.get("leaf_eval"))
        factory = _hidden_state_factory(context)
        if factory is None:
            return None

        # ここから先が「重い探索」(Step3 決定化評価)。適用外 select / top1 shortcut /
        # factory 無しはすべて上で return 済みなので、この1回だけ呼び出し側へ通知する。
        _notify_expensive_root(context)

        # anytime ラウンド方式(config["anytime"] があり enabled のときだけ)。キーが無い/falsy /
        # enabled=false なら以降の従来ループ(固定 N 決定化)に落ちるので本番挙動は不変。
        anytime_config = config.get("anytime")
        if anytime_config and anytime_config.get("enabled", True):
            scored = _run_anytime_rounds(obs, me, candidate_indices, config, anytime_config,
                                         deadline, evaluator, policy_model, factory)
            if not scored:
                return None
            best = [_pick_best_index(scored, probs, float(config["tie_eps"]))]
            return best if _is_legal_selection(best, select) else None

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

        # Step5: 平均最大を採用。tie_eps 以内は Policy 確率上位で決める。
        best = [_pick_best_index(scored, probs, float(config["tie_eps"]))]
        return best if _is_legal_selection(best, select) else None
    except Exception:
        return None
