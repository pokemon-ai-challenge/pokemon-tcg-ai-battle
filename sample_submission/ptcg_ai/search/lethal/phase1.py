"""Phase 1: 決定的な確定リーサル(設計 §3.1)。

**新情報を公開しない行動だけ**で反復深化する。ドロー・コイン等が起きた枝は
展開しない(それは Phase 2 の仕事)。

3 値の意味:

- ``PROVEN_WIN``    : 公開を伴わない手順で勝てることを確認した
- ``PROVEN_NO_WIN`` : **展開した木を辿り切って**勝ちが無いことを確認した
- ``UNKNOWN``       : 予算切れ・未対応効果・候補集合が不完全・
  **公開を伴う枝を展開しなかった**など

`PROVEN_NO_WIN` の意味は Step 1-19 で厳格化した。以前は「公開を伴わない範囲では
勝てない」という意味で、公開枝を飛ばしても `PROVEN_NO_WIN` を返していたが、
これは不健全だった(`cap0197` / `cap0199`: 公開の先に検証済みリーサルがあった)。
**展開しなかった枝が 1 つでもあれば `UNKNOWN`** とする。

相手選択ノード(B4。KO 後の ``TO_ACTIVE`` など)の扱い:

- **AND ノード**として、全ての合法な相手選択で勝てるときだけ ``PROVEN_WIN``
- 1 つでも ``PROVEN_NO_WIN`` があればその枝は ``PROVEN_NO_WIN``
- 選択を全部確認できない(候補が不完全 / 公開を伴う)なら ``UNKNOWN``
- **相手が自分に都合よく選ぶ前提は置かない**
"""

from __future__ import annotations

from ptcg_ai.search.lethal.backend import LethalBackend
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.transposition import Transposition
from ptcg_ai.search.lethal.types import LethalResult, Proof, StopReason


def search(backend: LethalBackend, budget: Budget) -> LethalResult:
    """Phase 1 探索を実行する。例外は投げない前提(backend 側で吸収する)。"""
    budget.start()
    # 列挙・具体化にも同じ予算を効かせる(Step 1-19 指示 10/11)。
    attach = getattr(backend, "attach_budget", None)
    if attach is not None:
        attach(budget)
    reasons: set[StopReason] = set()
    root = backend.root()

    for limit in range(1, budget.max_depth + 1):
        table = Transposition()
        first_action = None
        saw_unknown = False
        # 停止理由は**反復ごと**に取る。反復深化なので、浅い反復で付いた
        # `DEPTH_LIMIT` を最終結果へ持ち越すと「調べ尽くしたのに深さ制限が付いている」
        # という矛盾した診断になり、`PROVEN_NO_WIN` の契約(Step 1-20 指示 8)を
        # 検証できなくなる。証明そのものは `saw_unknown` が反復ごとに
        # リセットされるので以前から健全だった。
        pass_reasons: set[StopReason] = set()
        if not backend.action_set_complete(root):
            saw_unknown = True
            pass_reasons.add(StopReason.INCOMPLETE_ACTION_SET)
        for action in backend.legal_actions(root):
            proof = _action_proof(backend, root, action, limit, budget, table, pass_reasons)
            if proof is Proof.PROVEN_WIN:
                first_action = action
                break
            if proof is Proof.UNKNOWN:
                saw_unknown = True
        reasons |= pass_reasons
        if first_action is not None:
            return _result(Proof.PROVEN_WIN, reasons, budget, first_action=first_action,
                           depth=limit)
        if budget.check() is not None:
            reasons.add(budget.exhausted_reason)
            return _result(Proof.UNKNOWN, reasons, budget)
        if not saw_unknown:
            # この深さで木を辿り切り、拒否も不完全さも無かった。
            # 報告する理由も**この反復のもの**だけにする。
            return _result(Proof.PROVEN_NO_WIN, pass_reasons, budget)
    reasons.add(StopReason.DEPTH_LIMIT)
    return _result(Proof.UNKNOWN, reasons, budget)


def _state_proof(backend, state, depth_left, budget, table, reasons) -> Proof:
    if backend.is_win(state):
        return Proof.PROVEN_WIN
    if backend.is_terminal(state):
        return Proof.PROVEN_NO_WIN
    if depth_left <= 0:
        reasons.add(StopReason.DEPTH_LIMIT)
        return Proof.UNKNOWN
    if budget.check() is not None:
        reasons.add(budget.exhausted_reason)
        return Proof.UNKNOWN

    key = backend.observable_key(state)
    cached = table.lookup(key, depth_left)
    if cached is not None:
        return cached

    complete = backend.action_set_complete(state)
    actions = backend.legal_actions(state)
    if backend.is_opponent_node(state):
        proof = _and_node(backend, state, actions, complete, depth_left, budget, table, reasons)
    else:
        proof = _or_node(backend, state, actions, complete, depth_left, budget, table, reasons)
    if proof is Proof.PROVEN_WIN:
        table.store_win(key, depth_left)
    elif proof is Proof.PROVEN_NO_WIN:
        table.store_no_win(key, depth_left)
    return proof


def _or_node(backend, state, actions, complete, depth_left, budget, table, reasons) -> Proof:
    saw_unknown = not complete
    if not complete:
        reasons.add(StopReason.INCOMPLETE_ACTION_SET)
    for action in actions:
        proof = _action_proof(backend, state, action, depth_left, budget, table, reasons)
        if proof is Proof.PROVEN_WIN:
            return Proof.PROVEN_WIN
        if proof is Proof.UNKNOWN:
            saw_unknown = True
    return Proof.UNKNOWN if saw_unknown else Proof.PROVEN_NO_WIN


def _and_node(backend, state, actions, complete, depth_left, budget, table, reasons) -> Proof:
    """相手選択ノード。全ての選択で勝てるときだけ勝ち(B4)。"""
    if not actions:
        reasons.add(StopReason.UNSUPPORTED_EFFECT)
        return Proof.UNKNOWN
    proof = Proof.PROVEN_WIN
    for action in actions:
        sub = _action_proof(
            backend, state, action, depth_left, budget, reasons=reasons, table=table,
            opponent=True,
        )
        if sub is Proof.PROVEN_NO_WIN:
            # 相手が 1 つでも防げるなら、この枝は確定勝ちではない。
            return Proof.PROVEN_NO_WIN
        if sub is Proof.UNKNOWN:
            proof = Proof.UNKNOWN
    if proof is Proof.PROVEN_WIN and not complete:
        # 全ての相手選択を確認できていないので勝ちを主張しない。
        reasons.add(StopReason.INCOMPLETE_ACTION_SET)
        return Proof.UNKNOWN
    return proof


def _action_proof(backend, state, action, depth_left, budget, table, reasons, opponent=False) -> Proof:
    if budget.spend_node() is not None:
        reasons.add(budget.exhausted_reason)
        return Proof.UNKNOWN
    transition = backend.apply(state, action)
    if transition.stop_reason is not None:
        reasons.add(transition.stop_reason)
        return Proof.UNKNOWN
    if transition.state is None:
        # 状態を構築できないことは「勝てない」の証明にはならない。
        # 構築失敗は我々の都合であって、盤面の性質ではない(Step 1-19 指示 4)。
        reasons.add(StopReason.OUTCOMES_NOT_ENUMERABLE)
        return Proof.UNKNOWN
    if transition.revealed:
        # 新情報境界。Phase 1 はこの枝を**展開しない**。
        #
        # 以前はここで自分の手番のとき `PROVEN_NO_WIN` を返し、
        # 「公開を伴わない範囲では勝てない」という内部的な意味で正当化していた。
        # しかしこれは不健全だった(Step 1-19 P0):
        # 展開していない枝に勝ち筋が無いとは言えず、実際 `cap0197` / `cap0199` では
        # 1 手目の公開の先に検証済みの 4 手リーサルが存在した。
        # `phase2_enabled=false` の構成では Phase 1 の `PROVEN_NO_WIN` が最終回答に
        # なるため、この誤りはそのまま外へ出ていた。
        reasons.add(StopReason.OUTCOMES_NOT_ENUMERABLE)
        return Proof.UNKNOWN
    return _state_proof(backend, transition.state, depth_left - 1, budget, table, reasons)


def _result(proof, reasons, budget, *, first_action=None, depth=None) -> LethalResult:
    return LethalResult(
        proof,
        first_action=first_action,
        depth=depth,
        stop_reasons=tuple(sorted(reasons, key=lambda r: r.name)),
        nodes=budget.nodes,
        elapsed_ms=budget.elapsed_ms,
    )
