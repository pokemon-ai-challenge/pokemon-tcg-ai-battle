"""比較対象の探索実装(Step 1-2c)と、意図的に壊した実装。

ここに置いた探索は **toy 用の候補実装**であり、製品の Phase 1/2 ではない
(製品側は Step 1-3 以降)。oracle との比較ハーネスを先に完成させ、
本体が入ったら同じハーネスへ差し替えるための土台である。

oracle との独立性:

| | oracle | ここの候補実装 |
|---|---|---|
| 隠れ情報の表現 | 山札の**具体的な順列を全列挙** | 山札の **multiset** のみ |
| outcome の作り方 | 順列を観測キーでグルーピング | 多変量超幾何分布で厳密列挙 |
| 確率 | 順列を数える | ``Fraction`` の解析式 |

候補実装は ``state.deck`` の順序を読まない。読むのは ``deck_multiset()`` だけで、
outcome を作るときだけ「その outcome を実現する並び」を**自分で構築**する
(製品側で ``Scenario`` を ``outcome`` 層に閉じ込めるのと同じ役割)。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable, Hashable

from ptcg_ai.search.lethal.enumeration import (
    SOURCE_SAMPLED,
    Outcome,
    OutcomeSet,
    certify_for_proof,
    draw_outcomes,
)
from ptcg_ai.search.lethal.types import ChanceClass, Proof, StopReason

from tests.toy.toy_game import BACKEND, DRAW_COUNT, END, ToyState


@dataclass
class SearchResult:
    proof: Proof
    winning_root_actions: frozenset = frozenset()
    policy_tree: tuple | None = None
    min_depth: int | None = None
    stop_reason: StopReason | None = None
    coverage: dict = field(default_factory=dict)


def canonical(state: ToyState) -> ToyState:
    """山札を multiset の正準順にそろえる(= 順序情報を捨てる)。"""
    return replace(state, deck=tuple(sorted(state.deck)))


def realize(state: ToyState, drawn: tuple[int, ...]) -> ToyState:
    """``drawn`` を引く並びを自分で構築する(outcome の構築可能性)。

    信念(multiset)から作るので、実際の隠れた順序は使わない。
    """
    remaining = Counter(state.deck)
    for card_id in drawn:
        if remaining[card_id] <= 0:
            raise ValueError("outcome is not constructible from the belief")
        remaining[card_id] -= 1
    tail = tuple(sorted(remaining.elements()))
    return replace(state, deck=tuple(drawn) + tail)


def _label_to_cards(label) -> tuple[int, ...]:
    cards: list[int] = []
    for card_id, count in label:
        cards.extend([card_id] * count)
    return tuple(cards)


# ------------------------------------------------------------- Phase 1 相当


def phase1(
    state: ToyState,
    depth_limit: int,
    *,
    action_source: Callable[[ToyState], tuple] | None = None,
) -> SearchResult:
    """公開を伴わない行動だけの反復深化。山札の順序を一切見ない。"""
    start = canonical(state)
    actions_of = action_source or (lambda node: BACKEND.legal_actions(node))
    truncated = False
    for limit in range(1, depth_limit + 1):
        roots: set = set()
        truncated = False

        def dfs(node: ToyState, depth_left: int, root):
            nonlocal truncated
            if BACKEND.is_win(node):
                if root is not None:
                    roots.add(root)
                return True
            if BACKEND.is_terminal(node):
                return False
            if depth_left <= 0:
                truncated = True
                return False
            found = False
            for action in actions_of(node):
                if action == END:
                    continue
                if BACKEND.reveals(node, action):
                    # 公開枝は Phase 1 では展開しない。展開していない以上
                    # 「勝ち筋が無い」とは言えないので、調べ尽くしたことにしない。
                    # (Step 1-19 P0: toy oracle も本番と同じ誤りを共有していたため、
                    #  cap0197 / cap0199 の欠陥を検出できなかった)
                    truncated = True
                    continue
                child = BACKEND.apply(node, action)
                if dfs(child, depth_left - 1, root if root is not None else action):
                    found = True
            return found

        if dfs(start, limit, None):
            return SearchResult(Proof.PROVEN_WIN, frozenset(roots), min_depth=limit)
    if truncated:
        return SearchResult(Proof.UNKNOWN, stop_reason=StopReason.DEPTH_LIMIT)
    return SearchResult(Proof.PROVEN_NO_WIN)


# ------------------------------------------------------------- Phase 2 相当


def phase2(
    state: ToyState,
    depth_limit: int,
    *,
    outcome_builder: Callable[[ToyState], OutcomeSet] | None = None,
    honor_certification: bool = True,
    unknown_on_depth_limit: bool = True,
    action_source: Callable[[ToyState], tuple] | None = None,
) -> SearchResult:
    """3値 AND-OR。chance ノードは厳密列挙 + 証明可否の判定を通す。

    引数で挙動を差し替えられるのは、意図的に壊した実装を作るため。
    既定値が「正しい実装」である。
    """
    builder = outcome_builder or _default_outcomes
    actions_of = action_source or (lambda node: BACKEND.legal_actions(node))
    coverage: dict = {}

    def search(node: ToyState, depth_left: int, path: tuple):
        if BACKEND.is_win(node):
            return Proof.PROVEN_WIN, ("win",), None
        if BACKEND.is_terminal(node):
            return Proof.PROVEN_NO_WIN, None, None
        if depth_left <= 0:
            if unknown_on_depth_limit:
                return Proof.UNKNOWN, None, StopReason.DEPTH_LIMIT
            # 壊した実装: 「見つからなかった」を PROVEN_NO_WIN にしてしまう
            return Proof.PROVEN_NO_WIN, None, None

        winners: set = set()
        tree = None
        saw_unknown = False
        stop_reason = None

        for action in actions_of(node):
            if BACKEND.reveals(node, action):
                proof, subtree, reason = _chance_node(
                    node, action, depth_left, path, search, builder,
                    honor_certification, coverage,
                )
            else:
                child = BACKEND.apply(node, action)
                proof, subtree, reason = search(child, depth_left - 1, path + (action,))
                subtree = {child.observable(): subtree} if subtree else None
            if proof is Proof.PROVEN_WIN:
                winners.add(action)
                if tree is None:
                    tree = ("action", action, subtree or {})
            elif proof is Proof.UNKNOWN:
                saw_unknown = True
                stop_reason = stop_reason or reason
        if winners:
            return Proof.PROVEN_WIN, tree, None
        if saw_unknown:
            return Proof.UNKNOWN, None, stop_reason
        return Proof.PROVEN_NO_WIN, None, None

    proof, tree, reason = search(canonical(state), depth_limit, ())
    roots: set = set()
    if proof is Proof.PROVEN_WIN:
        roots = _winning_roots(
            canonical(state), depth_limit, builder, honor_certification,
            unknown_on_depth_limit, actions_of, search,
        )
    return SearchResult(proof, frozenset(roots), tree, stop_reason=reason, coverage=coverage)


def _chance_node(node, action, depth_left, path, search, builder, honor_certification, coverage):
    """新情報境界: outcome を厳密列挙し、全 outcome で勝てるかを見る。"""
    outcome_set = builder(node)
    certification = certify_for_proof(
        outcome_set, chance_class=ChanceClass.CONTROLLED_SUPPLY_ORDER
    )
    if honor_certification and not certification.admissible:
        return Proof.UNKNOWN, None, certification.stop_reason

    subtree: dict[Hashable, tuple] = {}
    observed: set = set()
    proof = Proof.PROVEN_WIN
    reason = None
    for outcome in outcome_set.outcomes:
        drawn = _label_to_cards(outcome.label)
        child = BACKEND.apply(realize(node, drawn), action)
        observed.add(child.observable())
        sub_proof, sub_tree, sub_reason = search(child, depth_left - 1, path + (action,))
        if sub_proof is Proof.PROVEN_WIN:
            subtree[child.observable()] = sub_tree if sub_tree else ("win",)
        elif sub_proof is Proof.PROVEN_NO_WIN:
            proof = Proof.PROVEN_NO_WIN
        else:
            if proof is Proof.PROVEN_WIN:
                proof = Proof.UNKNOWN
                reason = sub_reason
    coverage[path + (action,)] = observed
    return proof, subtree if proof is Proof.PROVEN_WIN else None, reason


def _winning_roots(start, depth_limit, builder, honor_certification,
                   unknown_on_depth_limit, actions_of, search) -> set:
    """勝てる root action を全部集める(tie-break の違いで比較がぶれないように)。"""
    roots: set = set()
    for action in actions_of(start):
        if BACKEND.reveals(start, action):
            proof, _tree, _reason = _chance_node(
                start, action, depth_limit, (), search, builder, honor_certification, {},
            )
        else:
            child = BACKEND.apply(start, action)
            proof, _tree, _reason = search(child, depth_limit - 1, (action,))
        if proof is Proof.PROVEN_WIN:
            roots.add(action)
    return roots


def _default_outcomes(node: ToyState) -> OutcomeSet:
    return draw_outcomes(node.deck_multiset(), DRAW_COUNT)


# ------------------------------------------------- 意図的に壊した実装(mutants)


def phase2_dropping_outcome(state: ToyState, depth_limit: int) -> SearchResult:
    """最も確率の低い outcome を捨てて、質量を正規化してしまう実装。

    証明の穴: 捨てた枝で負けていても ``PROVEN_WIN`` を出す。
    """

    def builder(node: ToyState) -> OutcomeSet:
        full = draw_outcomes(node.deck_multiset(), DRAW_COUNT)
        if len(full.outcomes) <= 1:
            return full
        kept = sorted(full.outcomes, key=lambda o: o.mass)[1:]
        total = sum((o.mass for o in kept), Fraction(0))
        renormalized = tuple(
            Outcome(label=o.label, mass=o.mass / total, constructible=True) for o in kept
        )
        return OutcomeSet(renormalized, full.source, coverage_certified=True)

    return phase2(state, depth_limit, outcome_builder=builder)


def phase2_sampling(state: ToyState, depth_limit: int, *, seed: int = 0) -> SearchResult:
    """1つの outcome だけをサンプリングして確定判定してしまう実装。"""
    import random

    rng = random.Random(seed)

    def builder(node: ToyState) -> OutcomeSet:
        full = draw_outcomes(node.deck_multiset(), DRAW_COUNT)
        picked = rng.choice(full.outcomes)
        return OutcomeSet(
            (Outcome(label=picked.label, mass=Fraction(1)),),
            SOURCE_SAMPLED,
            coverage_certified=False,
        )

    # 証明可否の判定を無視する = 壊れている点
    return phase2(state, depth_limit, outcome_builder=builder, honor_certification=False)


def phase2_no_win_on_depth_limit(state: ToyState, depth_limit: int) -> SearchResult:
    """深さで打ち切ったのに ``PROVEN_NO_WIN`` と言ってしまう実装。"""
    return phase2(state, depth_limit, unknown_on_depth_limit=False)


def phase1_peeking(state: ToyState, depth_limit: int) -> SearchResult:
    """山札の一番上を覗いて手を変える実装(determinization 違反)。"""
    top = state.deck[0] if state.deck else 0
    result = phase1(state, depth_limit)
    if result.proof is Proof.PROVEN_WIN and result.winning_root_actions:
        ordered = sorted(result.winning_root_actions)
        chosen = ordered[top % len(ordered)]
        return SearchResult(
            result.proof, frozenset({chosen}), min_depth=result.min_depth
        )
    return result
