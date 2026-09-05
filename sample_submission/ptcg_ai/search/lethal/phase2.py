"""Phase 2: 全 outcome での確定リーサル(設計 §3.2)。

3 値 AND-OR。決定ノードは OR、chance ノードは AND(全 outcome で勝つこと)。

``PROVEN_WIN`` を出す条件は Step 1-2 で固めた通り、**outcome 集合が証明に使える形**
であること(``enumeration.certify_for_proof``)。以下はすべて ``UNKNOWN`` になる:

- outcome を列挙できない / 列挙が不完全 / 質量が 1 にならない
- outcome を状態として構築できない、または構築結果が意図と違う
- 未対応効果(相手選択・多コイン・サイド取得など)
- B1 ガード違反(デッキ公開後にシャッフルを挟まないドロー)
- 予算切れ・深さ制限・chance 深さ制限・候補集合の不完全さ

``PROVEN_NO_WIN`` は「木を辿り切って、拒否も不完全さも無かった」場合だけ。
"""

from __future__ import annotations

from ptcg_ai.search.lethal.backend import LethalBackend
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.enumeration import certify_for_proof
from ptcg_ai.search.lethal.transposition import Transposition
from ptcg_ai.search.lethal.types import LethalResult, Proof, StopReason


def search(backend: LethalBackend, budget: Budget) -> LethalResult:
    budget.start()
    # 列挙・具体化にも同じ予算を効かせる(Step 1-19 指示 10/11)。
    attach = getattr(backend, "attach_budget", None)
    if attach is not None:
        attach(budget)
    reasons: set[StopReason] = set()
    table = Transposition()
    root = backend.root()
    depth = budget.max_depth
    chance = budget.max_chance_depth

    if backend.is_win(root):
        raise ValueError("root is already a win; the caller must handle this")

    saw_unknown = False
    if not backend.action_set_complete(root):
        saw_unknown = True
        reasons.add(StopReason.INCOMPLETE_ACTION_SET)

    for action in backend.legal_actions(root):
        proof = _action_proof(
            backend, root, action, depth, chance, budget, table, reasons
        )
        if proof is Proof.PROVEN_WIN:
            return LethalResult(
                Proof.PROVEN_WIN,
                first_action=action,
                depth=depth,
                stop_reasons=tuple(sorted(reasons, key=lambda r: r.name)),
                nodes=budget.nodes,
                elapsed_ms=budget.elapsed_ms,
            )
        if proof is Proof.UNKNOWN:
            saw_unknown = True
    proof = Proof.UNKNOWN if saw_unknown else Proof.PROVEN_NO_WIN
    return LethalResult(
        proof,
        stop_reasons=tuple(sorted(reasons, key=lambda r: r.name)),
        nodes=budget.nodes,
        elapsed_ms=budget.elapsed_ms,
    )


def _state_proof(backend, state, depth_left, chance_left, budget, table, reasons) -> Proof:
    if backend.is_win(state):
        return Proof.PROVEN_WIN
    if backend.is_terminal(state):
        return Proof.PROVEN_NO_WIN
    if depth_left <= 0:
        reasons.add(StopReason.DEPTH_LIMIT)
        return Proof.UNKNOWN
    exhausted = budget.check()
    if exhausted is not None:
        reasons.add(exhausted)
        return Proof.UNKNOWN

    key = (backend.observable_key(state), chance_left)
    cached = table.lookup(key, depth_left)
    if cached is not None:
        return cached

    complete = backend.action_set_complete(state)
    actions = backend.legal_actions(state)
    if backend.is_opponent_node(state):
        proof = _and_node(
            backend, state, actions, complete, depth_left, chance_left, budget, table, reasons
        )
    else:
        proof = _or_node(
            backend, state, actions, complete, depth_left, chance_left, budget, table, reasons
        )
    if proof is Proof.PROVEN_WIN:
        table.store_win(key, depth_left)
    elif proof is Proof.PROVEN_NO_WIN:
        table.store_no_win(key, depth_left)
    return proof


def _or_node(backend, state, actions, complete, depth_left, chance_left, budget, table, reasons):
    saw_unknown = not complete
    if not complete:
        reasons.add(StopReason.INCOMPLETE_ACTION_SET)
    for action in actions:
        proof = _action_proof(
            backend, state, action, depth_left, chance_left, budget, table, reasons
        )
        if proof is Proof.PROVEN_WIN:
            return Proof.PROVEN_WIN
        if proof is Proof.UNKNOWN:
            saw_unknown = True
    return Proof.UNKNOWN if saw_unknown else Proof.PROVEN_NO_WIN


def _and_node(backend, state, actions, complete, depth_left, chance_left, budget, table, reasons):
    """相手選択ノード(B4)。**全ての合法な相手選択で勝てるときだけ勝ち**。

    確率平均もしないし、相手が我々に有利な選択をする前提も置かない。
    Phase 3 では同じノードを最悪値(min)で評価することになる(未実装)。
    """
    if not actions:
        reasons.add(StopReason.UNSUPPORTED_EFFECT)
        return Proof.UNKNOWN
    proof = Proof.PROVEN_WIN
    for action in actions:
        sub = _action_proof(
            backend, state, action, depth_left, chance_left, budget, table, reasons
        )
        if sub is Proof.PROVEN_NO_WIN:
            return Proof.PROVEN_NO_WIN
        if sub is Proof.UNKNOWN:
            proof = Proof.UNKNOWN
    if proof is Proof.PROVEN_WIN and not complete:
        reasons.add(StopReason.INCOMPLETE_ACTION_SET)
        return Proof.UNKNOWN
    return proof


def _action_proof(backend, state, action, depth_left, chance_left, budget, table, reasons) -> Proof:
    if budget.spend_node() is not None:
        reasons.add(budget.exhausted_reason)
        return Proof.UNKNOWN

    transition = backend.apply(state, action)
    if transition.stop_reason is not None:
        reasons.add(transition.stop_reason)
        return Proof.UNKNOWN
    if transition.state is None:
        # 状態を構築できないことは「勝てない」の証明にはならない(Step 1-19 指示 4)。
        reasons.add(StopReason.OUTCOMES_NOT_ENUMERABLE)
        return Proof.UNKNOWN

    if not transition.revealed:
        return _state_proof(
            backend, transition.state, depth_left - 1, chance_left, budget, table, reasons
        )

    # ---- chance ノード
    if chance_left <= 0:
        reasons.add(StopReason.CHANCE_DEPTH_LIMIT)
        return Proof.UNKNOWN

    enumeration = backend.enumerate_outcomes(state, action)
    if enumeration.outcome_set is None:
        reasons.add(enumeration.stop_reason or StopReason.OUTCOMES_NOT_ENUMERABLE)
        return Proof.UNKNOWN
    certification = certify_for_proof(
        enumeration.outcome_set, chance_class=enumeration.chance_class
    )
    if not certification.admissible:
        reasons.add(certification.stop_reason or StopReason.OUTCOMES_NOT_ENUMERABLE)
        return Proof.UNKNOWN

    proof = Proof.PROVEN_WIN
    for outcome in enumeration.outcome_set.outcomes:
        if budget.spend_node() is not None:
            reasons.add(budget.exhausted_reason)
            return Proof.UNKNOWN
        realized = backend.apply_outcome(state, action, outcome)
        if realized.stop_reason is not None or realized.state is None:
            # 1つでも構築できない outcome があれば、全 outcome を覆えていない。
            reasons.add(realized.stop_reason or StopReason.OUTCOMES_NOT_ENUMERABLE)
            return Proof.UNKNOWN
        sub = _state_proof(
            backend, realized.state, depth_left - 1, chance_left - 1, budget, table, reasons
        )
        if sub is Proof.PROVEN_NO_WIN:
            return Proof.PROVEN_NO_WIN
        if sub is Proof.UNKNOWN:
            proof = Proof.UNKNOWN
    return proof
