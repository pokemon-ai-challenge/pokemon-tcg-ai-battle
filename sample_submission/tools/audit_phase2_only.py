"""Step 1-21: Phase 2 だけが確定できた 9 件の全件監査。

指示 1/2/3/4/11/12 をまとめて測る。solver は変更しない。

`replay` は本番と同じ規律で行う:
    root -> Phase 2 の first action だけ実行 -> 新しい状態で**再探索** -> ...
長い手順を保存して盲目的に実行はしない。途中で UNKNOWN / 非合法 /
未対応効果が出たら、その proof は不成立として扱う。
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import OptionType, to_observation_class
from main import read_deck_csv
from ptcg_ai.search.lethal import phase1, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession

CORPUS = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_capability.jsonl"
PHASE2_ONLY = ("cap0008", "cap0056", "cap0057", "cap0058", "cap0136",
               "cap0139", "cap0162", "cap0176", "cap0179")
BUDGET_MS = 500.0


class _Rooted:
    """同じ backend を別の状態から根として使う（再探索用）。"""

    def __init__(self, inner, state):
        self._inner, self._state = inner, state

    def root(self):
        return self._state

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Counting:
    def __init__(self, inner):
        self._inner = inner
        self.outcomes = self.materializations = self.chance_nodes = 0

    def enumerate_outcomes(self, state, action):
        enumeration = self._inner.enumerate_outcomes(state, action)
        self.chance_nodes += 1
        if enumeration.outcome_set is not None:
            self.outcomes += len(enumeration.outcome_set.outcomes)
        return enumeration

    def apply_outcome(self, state, action, outcome):
        self.materializations += 1
        return self._inner.apply_outcome(state, action, outcome)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _budget():
    return Budget(time_limit_ms=BUDGET_MS, max_nodes=20_000, max_depth=8,
                  max_chance_depth=1)


def search_once(row, deck, module):
    observation = to_observation_class(row["obs"])
    hidden = HiddenState.from_stub(row["hidden_stub"])
    started = time.perf_counter()
    with SearchSession(observation, hidden) as session:
        backend = _Counting(CgBackend(
            session, observation, hidden,
            deck_composition=KnownDeckComposition.from_card_ids(deck),
        ))
        if backend.is_win(backend.root()):
            return None
        result = module.search(backend, _budget())
        return {
            "proof": result.proof.name,
            "stop": sorted(s.name for s in result.stop_reasons),
            "nodes": result.nodes,
            "depth": result.depth,
            "ms": (time.perf_counter() - started) * 1000.0,
            "outcomes": backend.outcomes,
            "materializations": backend.materializations,
            "chance_nodes": backend.chance_nodes,
            "root_actions": len(backend.legal_actions(backend.root())),
            "first_action": (list(result.first_action)
                             if result.first_action is not None else None),
        }


def replay_by_research(row, deck, module, limit=60):
    """first action だけ実行 → 再探索、を繰り返して実際に勝ち切れるか確認する。

    ``limit`` は**エンジンの選択回数**であって「手数」ではない。
    1 枚のカードを使うだけでも「カードを選ぶ / 対象を選ぶ / 確認する」で
    複数回の選択になるため、oracle の minimum_depth より遥かに大きくなる。
    実測（Step 1-21）: 上限 12 では cap0056 / cap0057 / cap0058 が
    途中で打ち切られ、誤って「replay 失敗」と判定していた。
    """
    observation = to_observation_class(row["obs"])
    hidden = HiddenState.from_stub(row["hidden_stub"])
    trace = []
    with SearchSession(observation, hidden) as session:
        inner = CgBackend(
            session, observation, hidden,
            deck_composition=KnownDeckComposition.from_card_ids(deck),
        )
        state = inner.root()
        for _ in range(limit):
            if inner.is_win(state):
                return True, trace
            if inner.is_terminal(state):
                return False, trace + ["terminal_without_win"]
            try:
                result = module.search(_Rooted(inner, state), _budget())
            except Exception as error:  # noqa: BLE001
                return False, trace + [f"search_error:{type(error).__name__}"]
            if result.proof.name != "PROVEN_WIN" or result.first_action is None:
                return False, trace + [f"lost_proof:{result.proof.name}"]
            action = tuple(result.first_action)
            legal = [tuple(a) for a in inner.legal_actions(state)]
            if action not in legal:
                return False, trace + ["illegal_action"]
            transition = inner.apply(state, action)
            if transition.stop_reason is not None or transition.state is None:
                return False, trace + [
                    f"apply_failed:{transition.stop_reason}"
                ]
            trace.append(list(action))
            state = transition.state
        return False, trace + ["step_limit"]


def classify(row, p1, p2) -> str:
    oracle = row["oracle"]
    stop1 = set(p1["stop"])
    if p2["chance_nodes"] > 0 and p2["outcomes"] > 1:
        return "TypeC_multi_outcome"
    if p2["chance_nodes"] > 0:
        return "TypeB_random_outcome"
    if oracle.get("has_root_deck_reveal") or oracle.get("has_midturn_deck_reveal"):
        return "TypeD_deck_reveal"
    if "TIME_LIMIT" in stop1:
        return "TypeE_phase1_efficiency"
    if "DEPTH_LIMIT" in stop1:
        return "TypeA_phase1_depth"
    return "TypeF_other"


def audit_nine(rows, deck) -> None:
    print("\n## Phase2-only 9 件の監査（budget 500ms / nodes 20k / depth 8 / chance 1）")
    print(f"  {'id':>9}{'gt':>24}{'md':>4}{'chance':>7}{'B4':>4}{'rev':>5}"
          f"{'outc':>6}{'mat':>5}{'P2 ms':>8}  {'replay':>8}  type")
    types = Counter()
    replay_ok = 0
    for fixture_id in PHASE2_ONLY:
        row = rows[fixture_id]
        oracle = row["oracle"]
        p1 = search_once(row, deck, phase1)
        p2 = search_once(row, deck, phase2)
        ok, trace = replay_by_research(row, deck, phase2)
        replay_ok += ok
        kind = classify(row, p1, p2)
        types[kind] += 1
        reveal = oracle.get("has_root_deck_reveal") or oracle.get("has_midturn_deck_reveal")
        print(f"  {fixture_id:>9}{oracle['ground_truth']:>24}"
              f"{str(oracle['minimum_depth']):>4}"
              f"{'yes' if not oracle['chance_free'] else '-':>7}"
              f"{'yes' if oracle.get('has_opponent_choice') else '-':>4}"
              f"{'yes' if reveal else '-':>5}{p2['outcomes']:>6}"
              f"{p2['materializations']:>5}{p2['ms']:>8.1f}"
              f"  {'OK' if ok else 'FAIL':>8}  {kind}")
        if not ok:
            print(f"            replay trace: {trace}")
    print(f"  replay 成功: {replay_ok}/{len(PHASE2_ONLY)}")
    print("  分類:", dict(types.most_common()))


def audit_phase2_no_win(rows, deck) -> None:
    """指示 4: Phase 2 の `PROVEN_NO_WIN` に勝ち筋が隠れていないか。"""
    print("\n## Phase 2 の PROVEN_NO_WIN 監査")
    suspicious = []
    checked = 0
    for row in rows.values():
        if row["turn"] < 1:
            continue
        record = search_once(row, deck, phase2)
        if record is None or record["proof"] != "PROVEN_NO_WIN":
            continue
        checked += 1
        oracle = row["oracle"]
        problems = []
        if oracle["oracle_win"]:
            problems.append("oracle に勝ち筋がある")
        for name in ("TIME_LIMIT", "DEPTH_LIMIT", "NODE_LIMIT",
                     "OUTCOMES_NOT_ENUMERABLE", "UNSUPPORTED_EFFECT",
                     "INCOMPLETE_ACTION_SET", "CHANCE_DEPTH_LIMIT"):
            if name in record["stop"]:
                problems.append(name)
        if problems:
            suspicious.append((row["id"], problems))
    print(f"  PROVEN_NO_WIN を返した fixture: {checked}")
    print(f"  疑わしいもの: {len(suspicious)} {suspicious[:6]}")


def audit_efficiency(rows, deck) -> None:
    """指示 11/12: Phase 1 の p50 と、cap0008 / cap0139 の効率問題。"""
    print("\n## Phase 1 の探索効率")
    for fixture_id in ("cap0008", "cap0139"):
        row = rows[fixture_id]
        p1 = search_once(row, deck, phase1)
        p2 = search_once(row, deck, phase2)
        print(f"  {fixture_id}: minimum_depth={row['oracle']['minimum_depth']} "
              f"root候補={p1['root_actions']}")
        print(f"    Phase 1: {p1['proof']} nodes={p1['nodes']} {p1['ms']:.1f}ms "
              f"stop={p1['stop']}")
        print(f"    Phase 2: {p2['proof']} nodes={p2['nodes']} {p2['ms']:.1f}ms "
              f"depth={p2['depth']} first={p2['first_action']}")
    print("\n  全 valid での node 数比較（Phase 1 の反復深化コスト）")
    n1, n2, d1 = [], [], []
    for row in rows.values():
        if row["turn"] < 1:
            continue
        a = search_once(row, deck, phase1)
        b = search_once(row, deck, phase2)
        if a is None or b is None:
            continue
        n1.append(a["nodes"])
        n2.append(b["nodes"])
        if a["depth"]:
            d1.append(a["depth"])
    print(f"    Phase 1 nodes 平均={sum(n1) / len(n1):.0f}  "
          f"Phase 2 nodes 平均={sum(n2) / len(n2):.0f}")
    print(f"    Phase 1 が勝ちを見つけた深さ 平均={sum(d1) / max(len(d1), 1):.2f}"
          f"（反復深化なのでこの深さまで毎回やり直している）")


def audit_b4(rows) -> None:
    """指示 13: B4 の positive / negative / unknown を分ける。"""
    print("\n## B4 の分類")
    counts = Counter()
    for row in rows.values():
        if row["turn"] < 1 or not row["oracle"].get("has_opponent_choice"):
            continue
        oracle = row["oracle"]
        if oracle["oracle_win"]:
            counts["B4_positive"] += 1
        elif oracle["truncated"]:
            counts["B4_unknown_truncated"] += 1
        else:
            # オラクルは相手選択を経由する線を探索しないので、
            # ここは「リーサル不在」の根拠にできない。
            counts["B4_opponent_choice_unmodeled"] += 1
    for name, count in counts.most_common():
        print(f"  {name:34} {count}")


def main() -> None:
    rows = {json.loads(line)["id"]: json.loads(line)
            for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()}
    deck = read_deck_csv()
    audit_nine(rows, deck)
    audit_efficiency(rows, deck)
    audit_b4(rows)
    audit_phase2_no_win(rows, deck)


if __name__ == "__main__":
    main()
