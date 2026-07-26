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
    root_children: list = field(default_factory=list)   # [(action, visits, Q, prior)]
    mapping_errors: int = 0

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


def _rollout(node_cg, ref_player: int, evaluator, policy_model, config: dict, deadline: float):
    """policy 貪欲 rollout → v_ref∈[-1,1](ref_player 視点)。

    **Champion pipeline._rollout_and_eval と同じ停止条件**(policy greedy + opponent_depth 相手ターンで
    handcrafted leaf 評価)を用いる=同一コスト・同一評価。**terminal は leaf でなく ±1 を優先**(B8)。

    leaf_mode(既定 "rollout"=上記 v1 挙動、byte 単位で不変):
      "node" = **rollout を行わず expand した node を即 handcrafted leaf 評価**(ISMCTS v2.1、rollout の
               逐次 Policy forward を除去して iteration を高速化。差分は leaf 評価 timing のみ)。terminal 優先。
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
    o0 = node_cg.observation
    prev_actor = o0.current.yourIndex if o0.current is not None else ref_player
    opp_turns = 0
    for _ in range(max_steps):
        if time.perf_counter() > deadline:
            break
        o = node_cg.observation
        s = o.current
        if s is None:
            return 0.0
        if s.result != -1:
            return terminal_value(s.result, ref_player)               # terminal 優先
        actor = s.yourIndex
        if prev_actor == ref_player and actor != ref_player:
            opp_turns += 1
        if actor == ref_player and prev_actor != ref_player and opp_turns >= opponent_depth:
            return leaf_to_value(evaluator.evaluate(s, ref_player))
        if o.select is None or not o.select.option:
            return leaf_to_value(evaluator.evaluate(s, ref_player))
        selection = _pl._greedy_selection(policy_model, o)
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


def search(obs, config: dict, policy_model, evaluator, determinize: Callable[[], dict | None],
           deadline: float, rng: random.Random, stats: SearchStats | None = None) -> list[int] | None:
    """ISMCTS 本体。single-select MAIN 前提(呼び出し側で確認)。返り: 最良 action(list[int])or None。

    - determinize: match_context 信念から search_begin kwargs(dict)を返す factory(leakage-safe)。
    - deadline: time.perf_counter() 基準の締切(watchdog)。
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
                    v_ref = _rollout(node_cg, root_me, evaluator, policy_model, config, deadline); break
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
                    v_ref = _rollout(node_cg, root_me, evaluator, policy_model, config, deadline)
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
                    v_ref = _rollout(node_cg, root_me, evaluator, policy_model, config, deadline)
                    break
                # 完全展開 → PUCT 降下
                a = _puct(node, c_puct)
                path.append((key, a, to_move)); depth += 1
                if depth >= max_depth_cap:
                    v_ref = _rollout(node_cg, root_me, evaluator, policy_model, config, deadline); break
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
    # most-visited(tie は Q)
    best = max(root.actions, key=lambda a: (root.N.get(a, 0), root.W.get(a, 0.0) / max(1, root.N.get(a, 0))))
    st.root_children = sorted(
        [(a, root.N.get(a, 0), (root.W.get(a, 0.0) / root.N[a]) if root.N.get(a, 0) else 0.0, root.P.get(a, 0.0))
         for a in root.actions], key=lambda x: x[1], reverse=True)[:8]
    st.selected_action = best
    st.policy_changed_by_search = (best != st.root_policy_top1)
    return list(best)
