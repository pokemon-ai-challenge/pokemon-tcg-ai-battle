"""ground-truth oracle(Step 1-2b)。**探索側の実装を一切使わない独立実装**。

方針:

- 隠れた山札の**具体的な並びを全部列挙**し(multiset の相異なる順列は等確率)、
  観測可能部分が同じものをまとめて情報集合を作る。
- そこから 3 値 AND-OR を素朴に回す。マクロも InfoKey も超幾何分布の式も使わない
  (確率は「順列を数える」ことだけで出す)。
- 探索側は逆に「multiset + 超幾何分布」で解くので、導出経路が独立になる。

用語:

- Phase 1 相当: **新情報を公開しない行動だけ**で勝てるか(= 確定手順)
- Phase 2 相当: 公開結果に応じて手を変えてよいが、**全 outcome で勝てるか**
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import permutations
from typing import Hashable

from ptcg_ai.search.lethal.types import Proof

from tests.toy.toy_game import BACKEND, END, ToyState

# 決定木ノード: ("win",) か ("action", 行動, {観測キー: 子ノード})
PolicyNode = tuple


@dataclass(frozen=True)
class OracleResult:
    proof: Proof
    winning_root_actions: frozenset
    p_lower: Fraction
    p_upper: Fraction
    min_depth: int | None = None          # Phase 1 相当の最短 primitive 手数
    policy_tree: PolicyNode | None = None  # 勝てる root action のうち最初のもの


def beliefs(state: ToyState) -> tuple[ToyState, ...]:
    """山札の相異なる順列を全部作る(それぞれ等確率)。"""
    orders = sorted(set(permutations(state.deck)))
    return tuple(replace(state, deck=order) for order in orders)


def _group(children: tuple[ToyState, ...]) -> dict[Hashable, tuple[ToyState, ...]]:
    """観測可能部分が同じ子をまとめる(= 情報集合)。"""
    groups: dict[Hashable, list[ToyState]] = {}
    for child in children:
        groups.setdefault(child.observable(), []).append(child)
    return {key: tuple(value) for key, value in groups.items()}


# ---------------------------------------------------------------- Phase 2 相当


def analyze(state: ToyState, depth_limit: int) -> OracleResult:
    """全 outcome を覆う勝ち筋があるか(Phase 2 相当)を厳密に判定する。"""
    belief = beliefs(state)
    proof, p_lower, p_upper, winners, tree = _analyze(belief, depth_limit)
    return OracleResult(
        proof=proof,
        winning_root_actions=frozenset(winners),
        p_lower=p_lower,
        p_upper=p_upper,
        policy_tree=tree,
    )


def _analyze(belief: tuple[ToyState, ...], depth_left: int):
    representative = belief[0]
    if BACKEND.is_win(representative):
        return Proof.PROVEN_WIN, Fraction(1), Fraction(1), set(), ("win",)
    if BACKEND.is_terminal(representative):
        return Proof.PROVEN_NO_WIN, Fraction(0), Fraction(0), set(), None
    if depth_left <= 0:
        # 深さで打ち切った。勝てないと**証明した訳ではない**。
        return Proof.UNKNOWN, Fraction(0), Fraction(1), set(), None

    if BACKEND.is_opponent_node(representative):
        return _analyze_opponent_node(belief, depth_left)

    winners: set = set()
    best_lower = Fraction(0)
    best_upper = Fraction(0)
    saw_unknown = False
    tree: PolicyNode | None = None

    for action in BACKEND.legal_actions(representative):
        children = tuple(BACKEND.apply(s, action) for s in belief)
        groups = _group(children)
        total = len(belief)
        action_proof = Proof.PROVEN_WIN
        lower = Fraction(0)
        upper = Fraction(0)
        subtree: dict[Hashable, PolicyNode] = {}
        for key, group in groups.items():
            weight = Fraction(len(group), total)
            sub_proof, sub_lower, sub_upper, _sub_winners, sub_tree = _analyze(
                group, depth_left - 1
            )
            lower += weight * sub_lower
            upper += weight * sub_upper
            if sub_proof is Proof.PROVEN_WIN:
                subtree[key] = sub_tree if sub_tree is not None else ("win",)
            elif sub_proof is Proof.PROVEN_NO_WIN:
                action_proof = Proof.PROVEN_NO_WIN
            else:
                if action_proof is Proof.PROVEN_WIN:
                    action_proof = Proof.UNKNOWN

        if action_proof is Proof.PROVEN_WIN:
            winners.add(action)
            if tree is None:
                tree = ("action", action, subtree)
        elif action_proof is Proof.UNKNOWN:
            saw_unknown = True
        if lower > best_lower:
            best_lower = lower
        if upper > best_upper:
            best_upper = upper

    if winners:
        return Proof.PROVEN_WIN, best_lower, best_upper, winners, tree
    if saw_unknown:
        return Proof.UNKNOWN, best_lower, best_upper, set(), None
    return Proof.PROVEN_NO_WIN, best_lower, best_upper, set(), None


def _analyze_opponent_node(belief: tuple[ToyState, ...], depth_left: int):
    """相手選択ノード(B4)。**全ての選択で勝てるときだけ勝ち**。

    確率平均しない。相手が我々に有利な選択をする前提も置かない。
    ``p_lethal`` は最悪ケース(min)で持つ ── Phase 3 でこのノードが MIN 相当に
    なることに対応する。
    """
    representative = belief[0]
    actions = BACKEND.legal_actions(representative)
    if not actions:
        return Proof.UNKNOWN, Fraction(0), Fraction(1), set(), None
    proof = Proof.PROVEN_WIN
    worst_lower = Fraction(1)
    worst_upper = Fraction(1)
    subtree: dict[Hashable, PolicyNode] = {}
    for action in actions:
        children = tuple(BACKEND.apply(s, action) for s in belief)
        groups = _group(children)
        for key, group in groups.items():
            sub_proof, sub_lower, sub_upper, _w, sub_tree = _analyze(group, depth_left - 1)
            worst_lower = min(worst_lower, sub_lower)
            worst_upper = min(worst_upper, sub_upper)
            if sub_proof is Proof.PROVEN_NO_WIN:
                proof = Proof.PROVEN_NO_WIN
            elif sub_proof is Proof.UNKNOWN and proof is Proof.PROVEN_WIN:
                proof = Proof.UNKNOWN
            else:
                subtree[key] = sub_tree if sub_tree is not None else ("win",)
    tree = ("opponent", None, subtree) if proof is Proof.PROVEN_WIN else None
    return proof, worst_lower, worst_upper, set(), tree


# ---------------------------------------------------------------- Phase 1 相当


def analyze_deterministic(state: ToyState, depth_limit: int) -> OracleResult:
    """新情報を公開しない行動だけで勝てるか(Phase 1 相当)。

    公開を伴う行動は展開しない。したがって山札の順序に一切依存しない。
    """
    belief = beliefs(state)
    best: dict = {"depth": None, "roots": set(), "truncated": False}
    for representative in (belief[0],):
        _deterministic_search(representative, depth_limit, [], best)
    if best["depth"] is not None:
        return OracleResult(
            proof=Proof.PROVEN_WIN,
            winning_root_actions=frozenset(best["roots"]),
            p_lower=Fraction(1),
            p_upper=Fraction(1),
            min_depth=best["depth"],
        )
    if best["truncated"]:
        return OracleResult(Proof.UNKNOWN, frozenset(), Fraction(0), Fraction(1))
    return OracleResult(Proof.PROVEN_NO_WIN, frozenset(), Fraction(0), Fraction(0))


def _deterministic_search(state: ToyState, depth_left: int, path: list, best: dict) -> None:
    if BACKEND.is_win(state):
        depth = len(path)
        if best["depth"] is None or depth < best["depth"]:
            best["depth"] = depth
            best["roots"] = {path[0]} if path else set()
        elif depth == best["depth"] and path:
            best["roots"].add(path[0])
        return
    if BACKEND.is_terminal(state):
        return
    if depth_left <= 0:
        best["truncated"] = True
        return
    if BACKEND.is_opponent_node(state):
        # 相手選択は AND。全ての選択から勝ちへ行けるときだけ「勝ち」とする。
        _deterministic_opponent(state, depth_left, path, best)
        return
    for action in BACKEND.legal_actions(state):
        if action == END:
            continue
        if BACKEND.reveals(state, action):
            # 公開を伴う手は Phase 1 では展開しない。
            # ただし展開していない以上「勝ち筋が無い」とは言えないので、
            # 調べ尽くしたことにはしない = truncated を立てる。
            # (Step 1-19 P0: ここを立てていなかったため、この oracle は
            #  本番と同じ不健全さを共有し、cap0197 / cap0199 を検出できなかった)
            best["truncated"] = True
            continue
        path.append(action)
        _deterministic_search(BACKEND.apply(state, action), depth_left - 1, path, best)
        path.pop()


def _deterministic_opponent(state: ToyState, depth_left: int, path: list, best: dict) -> None:
    """AND ノード: 全ての相手選択で勝てる場合だけ、この経路を勝ちとして数える。"""
    depths = []
    for action in BACKEND.legal_actions(state):
        branch = {"depth": None, "roots": set(), "truncated": False}
        _deterministic_search(BACKEND.apply(state, action), depth_left - 1, list(path), branch)
        if branch["truncated"]:
            best["truncated"] = True
        if branch["depth"] is None:
            return  # 1 つでも勝てない選択があれば、この経路は確定勝ちではない
        depths.append(branch["depth"])
    if not depths:
        return
    depth = max(depths)  # 最悪ケースの手数
    if best["depth"] is None or depth < best["depth"]:
        best["depth"] = depth
        best["roots"] = {path[0]} if path else set()
    elif depth == best["depth"] and path:
        best["roots"].add(path[0])


# ------------------------------------------------------------ 補助(検証用)


def chance_coverage(state: ToyState, depth_limit: int) -> dict:
    """各 chance ノードで実際に起こりうる観測キーの集合(coverage の正解)。

    「行動列 -> その直後に起こりうる観測キーの集合」を返す。
    探索側が覆った集合と突き合わせるために使う。
    """
    coverage: dict = {}
    _coverage(beliefs(state), depth_limit, (), coverage)
    return coverage


def _coverage(belief, depth_left, path, coverage) -> None:
    representative = belief[0]
    if BACKEND.is_terminal(representative) or depth_left <= 0:
        return
    for action in BACKEND.legal_actions(representative):
        children = tuple(BACKEND.apply(s, action) for s in belief)
        groups = _group(children)
        if BACKEND.reveals(representative, action):
            coverage[path + (action,)] = set(groups)
        for group in groups.values():
            _coverage(group, depth_left - 1, path + (action,), coverage)
