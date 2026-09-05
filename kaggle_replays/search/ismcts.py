"""ISMCTS v1 — information-set Monte Carlo tree search(First Major Challenger の中核)。

設計: docs/plans/neural-agent/ismcts-v1-design-and-implementation-plan.md。Current Champion(abl_5_full)の
浅い決定化 PIMC(pipeline)を、**同じ Policy prior / 信念決定化 / handcrafted leaf / lethal を再利用**したまま
proper な information-set tree search へ深化する。production は変更せず read-only 再利用。

正しさの要点:
- **leakage-safe**: determinize は match_context 信念(hidden_state_factory)のみ。ground-truth の相手 hidden state を読まない。
- **Value domain 統一(一箇所)**: leaf p∈[0,1] → v=2p−1∈[-1,1]、terminal win=+1/loss=−1、fixed root-perspective negamax。
- **information-set node**: key=観測可能署名のみ(hidden ground truth を混ぜない)。決定化間で統計共有。
- **watchdog/fallback**: 予算超過・失敗は None を返し、呼び出し側が Champion fallback。silent illegal を出さない。

ローカル専用・git 未追跡。cg は import して呼ぶだけ。
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable

from cg import api as cg_api


# ---------------- Value domain(集約・F3 対象)----------------
def leaf_to_value(p: float) -> float:
    """handcrafted leaf p∈[0,1] → tree value v∈[-1,+1]。"""
    return 2.0 * float(p) - 1.0


def flip(v: float) -> float:
    """perspective 反転(negamax)。"""
    return -v


def terminal_value(result: int, ref_player: int) -> float:
    """cg terminal(result=winner index)→ ref_player 視点 [-1,+1]。draw(不定)は 0。"""
    if result == ref_player:
        return 1.0
    if result == (1 - ref_player):
        return -1.0
    return 0.0


def signed_for(v_ref: float, node_to_move: int, ref_player: int) -> float:
    """root_me(ref_player)視点の v を node の手番視点へ(negamax)。"""
    return v_ref if node_to_move == ref_player else flip(v_ref)


# ---------------- information-set key(観測可能のみ)----------------
def _card_ids(pokemons) -> tuple:
    out = []
    for p in (pokemons or []):
        out.append(getattr(p, "cardId", None) if p is not None else None)
    return tuple(out)


def info_set_key(obs) -> tuple:
    """観測可能署名。hidden ground truth(相手手札の中身等)を含めない。

    含める: turn / select.type / select.context / yourIndex / option 署名(type+cardId)/
            自分の hand cardId 多重集合(自分には可視)/ 両者 active+bench cardId(公開)/ prize・deck 枚数。
    含めない: 相手 hand の内容(hidden)。
    """
    s = obs.current
    sel = obs.select
    me = s.yourIndex
    opts = tuple((int(o.type), getattr(o, "cardId", None)) for o in sel.option) if sel and sel.option else ()
    my = s.players[me]
    opp = s.players[1 - me]
    my_hand = tuple(sorted(c for c in (getattr(h, "cardId", None) for h in (my.hand or [])) if c is not None)) \
        if getattr(my, "hand", None) else ()
    return (
        int(getattr(s, "turn", -1)), int(getattr(sel, "type", -1)), int(getattr(sel, "context", -1)), int(me),
        opts, my_hand,
        _card_ids(getattr(my, "active", None)), _card_ids(getattr(my, "bench", None)),
        _card_ids(getattr(opp, "active", None)), _card_ids(getattr(opp, "bench", None)),
        int(getattr(my, "deckCount", -1)), int(getattr(opp, "deckCount", -1)),
        len(getattr(my, "prize", []) or []), len(getattr(opp, "prize", []) or []),
    )


# ---------------- tree ----------------
@dataclass
class Node:
    to_move: int
    actions: list                       # list[tuple[int,...]] legal selections(single-select: [(i,)])
    P: dict = field(default_factory=dict)   # action -> prior
    N: dict = field(default_factory=dict)   # action -> visit
    W: dict = field(default_factory=dict)   # action -> total value(node.to_move 視点)
    n_total: int = 0

    def unexpanded(self) -> list:
        return [a for a in self.actions if self.N.get(a, 0) == 0]


@dataclass
class SearchStats:
    iterations: int = 0
    nodes: int = 0
    expanded_nodes: int = 0
    max_depth: int = 0
    sum_depth: int = 0
    determinizations: int = 0
    unique_information_sets: int = 0
    elapsed_ms: float = 0.0
    budget_ms: float = 0.0
    timeout: bool = False
    fallback_used: bool = False
    invalid_action: bool = False
    root_policy_top1: tuple | None = None
    selected_action: tuple | None = None
    policy_changed_by_search: bool = False
    root_children: list = field(default_factory=list)   # [(action, visits, Q, prior)] top-8(表示用)
    root_children_full: list = field(default_factory=list)  # 全 root action [(action, visits, Q, prior)](v2.5 蒸留 target 用)
    mapping_errors: int = 0
    trajectory: list = field(default_factory=list)      # v2.12: [(iter, top1, share, gap)] at check points(log_trajectory 時)
    early_stopped: bool = False                         # v2.12: adaptive early-stop で終了したか
    stop_iter: int = 0                                  # v2.12: early-stop した iteration(0=しなかった)
    # v2.13 hard-root extension 監査(記録専用・意思決定 byte 不変)。
    checkpoints: list = field(default_factory=list)     # [(cap_ms, iter, action, share, gap, entropy)] 時間基準 snapshot
    iter_checkpoints: list = field(default_factory=list)  # [(cap_iter, iter, action, share, gap, entropy)] iteration 基準(test 用)
    soft_cap_action: tuple | None = None                # soft cap 到達時の most-visited action(None=soft cap 前に終了=非 hard root)
    is_hard_root: bool = False                          # soft cap を early-stop せず到達した=hard root

    @property
    def mean_depth(self) -> float:
        return self.sum_depth / self.iterations if self.iterations else 0.0


def _legal_actions(sel) -> list:
    """single-select(maxCount==1)の legal action 集合。各 action=(index,)。"""
    return [(i,) for i in range(len(sel.option))]


def _priors(policy_model, obs, sel, actions: list) -> dict:
    """Champion Policy から prior(softmax)。失敗時は一様。mapping は index 一対一。"""
    try:
        scores = policy_model.score_options_from_state(obs.current, sel)
    except Exception:
        scores = None
    n = len(actions)
    if not scores or len(scores) != len(sel.option):
        return {a: 1.0 / n for a in actions}
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps) or 1.0
    return {(i,): exps[i] / z for i in range(len(sel.option))}


def _rollout(node_cg, ref_player: int, evaluator, policy_model, config: dict, deadline: float,
             rollout_policy=None):
    """policy 貪欲 rollout → v_ref∈[-1,1](ref_player 視点)。

    **Champion pipeline._rollout_and_eval と同じ停止条件**(policy greedy + opponent_depth 相手ターンで
    handcrafted leaf 評価)を用いる=同一コスト・同一評価。**terminal は leaf でなく ±1 を優先**(B8)。

    leaf_mode(既定 "rollout"=上記 v1 挙動、byte 単位で不変):
      "node" = **rollout を行わず expand した node を即 handcrafted leaf 評価**(ISMCTS v2.1、rollout の
               逐次 Policy forward を除去して iteration を高速化。差分は leaf 評価 timing のみ)。terminal 優先。

    rollout_cutoff(既定 "full"=v1 挙動、byte 単位で不変。ISMCTS v2.3 semantic truncated rollout):
      "one_handoff"  = **最初の turn handoff(actor 交代)で rollout を打ち切り handcrafted leaf 評価**。
      "two_handoff"  = **2 回目の turn handoff で打ち切り**(=典型的な v1 の 2-handoff horizon と概ね一致)。
      "full"         = 既存 opp_turns 基準の停止(v1 完全同一)。max_handoffs=None ゆえ早期 return は発火しない。
      cutoff は **actor 交代直後の合法な decision 境界**(pending 途中選択なし)で leaf 評価。terminal は cutoff より優先。
    """
    if config.get("leaf_mode", "rollout") == "node":
        s = node_cg.observation.current
        if s is None:
            return 0.0
        if s.result != -1:
            return terminal_value(s.result, ref_player)     # terminal 優先(handcrafted leaf より)
        return leaf_to_value(evaluator.evaluate(s, ref_player))
    from ptcg_ai.search import pipeline as _pl
    opponent_depth = max(1, int(config.get("opponent_depth", 1)))
    max_steps = int(config.get("max_rollout_steps", 40))
    # v2.3 semantic cutoff(既定 full=None → 早期 return 無効 = v1 byte 不変)。
    max_handoffs = {"one_handoff": 1, "two_handoff": 2}.get(config.get("rollout_cutoff", "full"))
    o0 = node_cg.observation
    prev_actor = o0.current.yourIndex if o0.current is not None else ref_player
    opp_turns = 0
    handoffs = 0
    for _ in range(max_steps):
        if time.perf_counter() > deadline:
            break
        o = node_cg.observation
        s = o.current
        if s is None:
            return 0.0
        if s.result != -1:
            return terminal_value(s.result, ref_player)               # terminal 優先(cutoff より前)
        actor = s.yourIndex
        if actor != prev_actor:
            handoffs += 1
            if max_handoffs is not None and handoffs >= max_handoffs:
                # semantic cutoff: actor 交代直後の合法 decision 境界(select 有・pending 途中選択なし)で leaf。
                return leaf_to_value(evaluator.evaluate(s, ref_player))
        if prev_actor == ref_player and actor != ref_player:
            opp_turns += 1
        if actor == ref_player and prev_actor != ref_player and opp_turns >= opponent_depth:
            return leaf_to_value(evaluator.evaluate(s, ref_player))
        if o.select is None or not o.select.option:
            return leaf_to_value(evaluator.evaluate(s, ref_player))
        # v2.4: rollout の action selection のみ student(rollout_policy)を使う。None=teacher=v1 不変。
        selection = _pl._greedy_selection(rollout_policy or policy_model, o)
        if not selection:
            return leaf_to_value(evaluator.evaluate(s, ref_player))
        try:
            node_cg = cg_api.search_step(node_cg.searchId, selection)
        except ValueError:
            return leaf_to_value(evaluator.evaluate(s, ref_player))
        prev_actor = actor
    s = node_cg.observation.current
    return leaf_to_value(evaluator.evaluate(s, ref_player)) if s is not None else 0.0


def _puct(node: Node, c_puct: float) -> tuple:
    """PUCT で action 選択(node.to_move 視点で最大化)。"""
    sqrt_total = math.sqrt(max(1, node.n_total))
    best_a, best_score = None, -1e18
    for a in node.actions:
        n = node.N.get(a, 0)
        q = (node.W.get(a, 0.0) / n) if n > 0 else 0.0
        u = c_puct * node.P.get(a, 0.0) * sqrt_total / (1 + n)
        sc = q + u
        if sc > best_score:
            best_score, best_a = sc, a
    return best_a


def _root_snapshot(root: Node) -> dict:
    """現時点の root 統計 snapshot(visits 基準 most-visited)。v2.13 checkpoint 記録専用。読み取りのみ。

    返り dict: action / visits(top1 訪問数)/ n_total / share / gap / entropy。
    """
    if root.n_total <= 0:
        return {"action": None, "visits": 0, "n_total": 0, "share": 0.0, "gap": 0.0, "entropy": 0.0}
    ordered = sorted(root.actions, key=lambda a: root.N.get(a, 0), reverse=True)
    t1 = ordered[0]; n1 = root.N.get(t1, 0)
    n2 = root.N.get(ordered[1], 0) if len(ordered) > 1 else 0
    nt = root.n_total
    ent = 0.0
    for a in root.actions:
        p = root.N.get(a, 0) / nt
        if p > 0.0:
            ent -= p * math.log(p)
    return {"action": tuple(t1), "visits": n1, "n_total": nt,
            "share": round(n1 / nt, 4), "gap": round((n1 - n2) / nt, 4), "entropy": round(ent, 4)}


def search(obs, config: dict, policy_model, evaluator, determinize: Callable[[], dict | None],
           deadline: float, rng: random.Random, stats: SearchStats | None = None,
           rollout_policy=None) -> list[int] | None:
    """ISMCTS 本体。single-select MAIN 前提(呼び出し側で確認)。返り: 最良 action(list[int])or None。

    - determinize: match_context 信念から search_begin kwargs(dict)を返す factory(leakage-safe)。
    - deadline: time.perf_counter() 基準の締切(watchdog)。
    - rollout_policy: v2.4 の student(rollout の greedy 選択のみ使用)。None=teacher=v1 不変。
    """
    st = stats if stats is not None else SearchStats()
    sel = obs.select
    root_me = obs.current.yourIndex
    root_actions = _legal_actions(sel)
    root_key = info_set_key(obs)
    root_priors = _priors(policy_model, obs, sel, root_actions)
    st.root_policy_top1 = max(root_priors, key=root_priors.get) if root_priors else None
    st.budget_ms = max(0.0, (deadline - time.perf_counter()) * 1000)

    tree: dict[tuple, Node] = {}
    tree[root_key] = Node(to_move=root_me, actions=root_actions, P=root_priors,
                          N={a: 0 for a in root_actions}, W={a: 0.0 for a in root_actions})
    st.nodes = 1
    c_puct = float(config.get("c_puct", 1.4))
    max_iter = int(config.get("iterations", 128))
    max_depth_cap = int(config.get("max_depth", 60))
    t0 = time.perf_counter()

    # 決定化 world プールを事前サンプル(belief 推論を pool_size 回に償却)。cg は search_step で
    # state を消費し巻き戻せないため各 iteration で search_begin が要る。world dict を使い回して
    # belief 推論コストだけ償却する(Champion の num_determinizations と同思想の固定プール)。
    pool_size = max(1, int(config.get("world_pool_size", 8)))
    pool = []
    for _ in range(pool_size):
        try:
            w = determinize()
        except Exception:
            w = None
        if w is not None:
            pool.append(w)
    if not pool:
        st.fallback_used = True
        return None  # 信念が無ければ探索できない(呼び出し側 fallback)
    st.determinizations = len(pool)

    # v2.12 adaptive early-stop / trajectory logging(既定 OFF → v2.11 byte 不変)。
    _es = config.get("early_stop") or {}
    _es_on = bool(_es.get("enabled"))
    _log_traj = bool(config.get("log_trajectory"))
    _es_nmin = int(_es.get("n_min", 48)); _es_every = max(1, int(_es.get("check_every", 8)))
    _es_k = int(_es.get("k_stable", 3)); _es_share = float(_es.get("share", 0.6)); _es_gap = float(_es.get("gap", 0.3))
    _t1_hist: list = []

    # v2.13 hard-root extension 監査(記録専用 → 既定 None なら v2.11/v2.12 byte 不変)。deadline 自体は
    # 呼び出し側(agent)が soft/hard cap を選んで渡す。ここは checkpoint snapshot と hard-root 判定のみ。
    _soft_cap_ms = config.get("soft_cap_ms")
    _cp_ms = sorted({float(x) for x in (config.get("checkpoints_ms") or [])})
    if _soft_cap_ms is not None and float(_soft_cap_ms) not in _cp_ms:
        _cp_ms.append(float(_soft_cap_ms)); _cp_ms.sort()
    _cp_iters = sorted({int(x) for x in (config.get("checkpoint_iters") or [])})   # test 用(iteration 基準)

    it = 0
    while it < max_iter:
        if time.perf_counter() > deadline:
            st.timeout = True
            break
        world = pool[it % len(pool)]
        try:
            node_cg = cg_api.search_begin(
                obs, world["your_deck"], world["your_prize"], world["opponent_deck"],
                world["opponent_prize"], world["opponent_hand"], world["opponent_active"])
        except Exception:
            break
        try:
            path = []          # [(key, action, to_move)]
            depth = 0
            v_ref = 0.0
            while True:
                o = node_cg.observation
                s = o.current
                if s is None:
                    v_ref = 0.0; break
                if s.result != -1:
                    v_ref = terminal_value(s.result, root_me); break
                if o.select is None or not o.select.option or o.select.maxCount != 1:
                    # tree は single-select のみ扱う。multi/none は rollout 相当で leaf 評価。
                    v_ref = _rollout(node_cg, root_me, evaluator, policy_model, config, deadline, rollout_policy); break
                key = info_set_key(o)
                to_move = s.yourIndex
                node = tree.get(key)
                if node is None:
                    acts = _legal_actions(o.select)
                    node = Node(to_move=to_move, actions=acts, P=_priors(policy_model, o, o.select, acts),
                                N={a: 0 for a in acts}, W={a: 0.0 for a in acts})
                    tree[key] = node
                    st.nodes += 1
                    # expand: 未訪問 action を1つ(prior 最大)→ rollout
                    unexp = node.unexpanded()
                    a = max(unexp, key=lambda x: node.P.get(x, 0.0)) if unexp else _puct(node, c_puct)
                    path.append((key, a, to_move)); depth += 1
                    try:
                        node_cg = cg_api.search_step(node_cg.searchId, list(a))
                    except ValueError:
                        st.mapping_errors += 1; v_ref = leaf_to_value(evaluator.evaluate(s, root_me)); break
                    st.expanded_nodes += 1
                    v_ref = _rollout(node_cg, root_me, evaluator, policy_model, config, deadline, rollout_policy)
                    break
                unexp = node.unexpanded()
                if unexp:
                    a = max(unexp, key=lambda x: node.P.get(x, 0.0))
                    path.append((key, a, to_move)); depth += 1
                    try:
                        node_cg = cg_api.search_step(node_cg.searchId, list(a))
                    except ValueError:
                        st.mapping_errors += 1; v_ref = leaf_to_value(evaluator.evaluate(s, root_me)); break
                    st.expanded_nodes += 1
                    v_ref = _rollout(node_cg, root_me, evaluator, policy_model, config, deadline, rollout_policy)
                    break
                # 完全展開 → PUCT 降下
                a = _puct(node, c_puct)
                path.append((key, a, to_move)); depth += 1
                if depth >= max_depth_cap:
                    v_ref = _rollout(node_cg, root_me, evaluator, policy_model, config, deadline, rollout_policy); break
                try:
                    node_cg = cg_api.search_step(node_cg.searchId, list(a))
                except ValueError:
                    st.mapping_errors += 1; v_ref = leaf_to_value(evaluator.evaluate(s, root_me)); break
            # backup(negamax: node.to_move 視点)
            for (key, a, to_move) in path:
                nd = tree[key]
                nd.N[a] = nd.N.get(a, 0) + 1
                nd.W[a] = nd.W.get(a, 0.0) + signed_for(v_ref, to_move, root_me)
                nd.n_total += 1
            st.max_depth = max(st.max_depth, depth)
            st.sum_depth += depth
        finally:
            try:
                cg_api.search_release(node_cg.searchId)
            except Exception:
                pass
        it += 1
        # v2.13: 時間/iteration 基準 checkpoint snapshot(記録専用・byte 不変)。soft cap を early-stop せず
        # 到達した root を hard root と判定。early-stop 前に配置=soft cap 前に収束した root は hard 扱いしない。
        if _cp_ms or _cp_iters:
            if _cp_ms:
                _elapsed_ms = (time.perf_counter() - t0) * 1000.0
                while _cp_ms and _elapsed_ms >= _cp_ms[0]:
                    _cap = _cp_ms.pop(0); _snap = _root_snapshot(tree[root_key])
                    st.checkpoints.append({"cap_ms": _cap, "iter": it, **_snap})
                    if _soft_cap_ms is not None and _cap <= float(_soft_cap_ms) + 1e-9 and st.soft_cap_action is None:
                        st.soft_cap_action = _snap["action"]; st.is_hard_root = True
            while _cp_iters and it >= _cp_iters[0]:
                _ci = _cp_iters.pop(0)
                st.iter_checkpoints.append({"cap_iter": _ci, "iter": it, **_root_snapshot(tree[root_key])})
        # v2.12: check point で root 安定性を評価(log or early-stop)。既定 OFF なら以下は実行されない。
        if (_log_traj or _es_on) and it >= _es_nmin and (it % _es_every == 0):
            _rt = tree[root_key]
            if _rt.n_total > 0:
                _ordered = sorted(_rt.actions, key=lambda a: _rt.N.get(a, 0), reverse=True)
                _t1 = _ordered[0]; _n1 = _rt.N.get(_t1, 0)
                _n2 = _rt.N.get(_ordered[1], 0) if len(_ordered) > 1 else 0
                _share = _n1 / _rt.n_total; _gap = (_n1 - _n2) / _rt.n_total
                if _log_traj:
                    st.trajectory.append((it, tuple(_t1), round(_share, 4), round(_gap, 4)))
                if _es_on:
                    _t1_hist.append(tuple(_t1))
                    _stable = len(_t1_hist) >= _es_k and len(set(_t1_hist[-_es_k:])) == 1
                    if _stable and _share >= _es_share and _gap >= _es_gap:
                        st.early_stopped = True; st.stop_iter = it
                        break
    try:
        cg_api.search_end()
    except Exception:
        pass

    st.iterations = it
    st.elapsed_ms = (time.perf_counter() - t0) * 1000
    st.unique_information_sets = len(tree)

    root = tree[root_key]
    if root.n_total == 0:
        st.fallback_used = True
        return None  # 1 回も回せず(予算/信念)→ 呼び出し側 fallback
    # v2.13: deadline break で loop 内記録を逃した checkpoint のみ final snapshot で flush。**実 elapsed に
    # 到達した cap だけ**記録する(early-stop で早期終了した場合、soft cap 時間には未到達 → hard root にしない)。
    if _cp_ms:
        _fsnap = _root_snapshot(root)
        for _cap in _cp_ms:
            if _cap > st.elapsed_ms:
                continue     # この時間には実際到達していない(収束/早期終了)→ 記録も hard 判定もしない
            st.checkpoints.append({"cap_ms": _cap, "iter": it, **_fsnap})
            if _soft_cap_ms is not None and _cap <= float(_soft_cap_ms) + 1e-9 and st.soft_cap_action is None:
                st.soft_cap_action = _fsnap["action"]; st.is_hard_root = True
    # most-visited(tie は Q)
    best = max(root.actions, key=lambda a: (root.N.get(a, 0), root.W.get(a, 0.0) / max(1, root.N.get(a, 0))))
    _rc_all = [(a, root.N.get(a, 0), (root.W.get(a, 0.0) / root.N[a]) if root.N.get(a, 0) else 0.0, root.P.get(a, 0.0))
               for a in root.actions]
    st.root_children_full = sorted(_rc_all, key=lambda x: x[1], reverse=True)   # 全 action(v2.5 蒸留 target)
    st.root_children = st.root_children_full[:8]
    st.selected_action = best
    st.policy_changed_by_search = (best != st.root_policy_top1)
    return list(best)
