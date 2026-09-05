"""ISMCTS v2.2 value-leaf correctness。

唯一差分 = leaf evaluator を handcrafted → 既存 ValueModel に差し替え(leaf_mode=node のまま)。
F8 no-rollout / F9 leaf-point / F10 perspective / F11 domain / F12 terminal override /
F13 hidden leakage を、既存 ptcg_ai.search.leaf_eval.ValueModelEvaluator に偽 ValueModel を
注入して検証する(重み JSON 非依存の決定的テスト)。F0-F7 は共有 core(ismcts.py)ゆえ
test_ismcts_unit / test_ismcts_cg を継承、F13 は encoder が公開情報のみ読む性質を直接示す。
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT / "sample_submission"), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import ismcts  # noqa: E402
from ptcg_ai.search.leaf_eval import ValueModelEvaluator  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402


# ---- 軽量 stub ----
class _S:
    def __init__(self, me=0, result=-1):
        self.yourIndex = me
        self.result = result


class _Node:
    def __init__(self, state):
        self.searchId = 1
        class _O:
            current = state
        self.observation = _O()


class _FakeVM:
    """predict_win_prob_from_state を固定 p で返す偽 ValueModel(呼び出し回数を記録)。"""
    def __init__(self, p):
        self.p = float(p)
        self.calls = 0
        self.seen = []

    def predict_win_prob_from_state(self, state):
        self.calls += 1
        self.seen.append(state)
        return self.p


def _boom_policy():
    class P:
        def score_options_from_state(self, s, sel):
            raise AssertionError("value leaf で policy(rollout)が呼ばれた")
    return P()


def _value_eval(p):
    return ValueModelEvaluator(model=_FakeVM(p))


# ---- F8: value leaf は rollout の policy を呼ばない ----
def test_F8_value_leaf_no_rollout_policy_call():
    fake = _FakeVM(0.75)
    ev = ValueModelEvaluator(model=fake)
    v = ismcts._rollout(_Node(_S(me=0)), 0, ev, _boom_policy(), {"leaf_mode": "node"}, 1e18)
    assert abs(v - ismcts.leaf_to_value(0.75)) < 1e-12   # 2*0.75-1 = 0.5
    assert fake.calls == 1                               # value は 1 回だけ、policy(_boom)は未呼び出し


# ---- F9: value 評価対象は expand した node そのもの(先へ進めない)----
def test_F9_value_leaf_point_is_expanded_node():
    st = _S(me=0)
    fake = _FakeVM(0.6)
    ev = ValueModelEvaluator(model=fake)
    ismcts._rollout(_Node(st), 0, ev, _boom_policy(), {"leaf_mode": "node"}, 1e18)
    assert len(fake.seen) == 1 and fake.seen[0] is st, "評価対象が expand node の state でない/余計に進んだ"


# ---- F10: perspective(state.yourIndex==ref なら p、そうでなければ 1-p)----
def test_F10_value_perspective_alignment():
    # ref_player=0, state.yourIndex=0 -> evaluate=0.8 -> value=2*0.8-1=0.6
    v_same = ismcts._rollout(_Node(_S(me=0)), 0, _value_eval(0.8), _boom_policy(), {"leaf_mode": "node"}, 1e18)
    assert abs(v_same - ismcts.leaf_to_value(0.8)) < 1e-12
    # ref_player=0, state.yourIndex=1(相手手番 leaf)-> evaluate=1-0.8=0.2 -> value=-0.6
    v_flip = ismcts._rollout(_Node(_S(me=1)), 0, _value_eval(0.8), _boom_policy(), {"leaf_mode": "node"}, 1e18)
    assert abs(v_flip - ismcts.leaf_to_value(0.2)) < 1e-12
    assert v_same > 0 > v_flip                          # 符号が root 視点で正しく反転


# ---- F11: value domain(p=1→+1 / p=0.5→0 / p=0→-1)----
def test_F11_value_domain_mapping():
    for p, expected in [(1.0, 1.0), (0.5, 0.0), (0.0, -1.0)]:
        v = ismcts._rollout(_Node(_S(me=0)), 0, _value_eval(p), _boom_policy(), {"leaf_mode": "node"}, 1e18)
        assert abs(v - expected) < 1e-12, (p, v, expected)


# ---- F12: terminal は value を呼ばず official terminal を返す ----
def test_F12_terminal_override_skips_value():
    fake_win = _FakeVM(0.99)                             # value は勝ち寄りに出すが terminal が優先
    v_win = ismcts._rollout(_Node(_S(me=0, result=0)), 0, ValueModelEvaluator(model=fake_win),
                            _boom_policy(), {"leaf_mode": "node"}, 1e18)
    assert v_win == 1.0 and fake_win.calls == 0         # root_me=0 の勝ち = +1、value 未呼び出し
    fake_loss = _FakeVM(0.99)
    v_loss = ismcts._rollout(_Node(_S(me=0, result=1)), 0, ValueModelEvaluator(model=fake_loss),
                             _boom_policy(), {"leaf_mode": "node"}, 1e18)
    assert v_loss == -1.0 and fake_loss.calls == 0      # 相手の勝ち = -1、value 未呼び出し


# ---- F13: hidden leakage(相手 hidden を変えても Value 入力=encoder 出力が不変)----
class _Card:
    def __init__(self, cid):
        self.id = cid


class _Player:
    def __init__(self, hand, hand_count, deck_count, prize, discard):
        self.active = []      # 伏せ扱い(pokemon 特徴は全 0、card_cache 非依存)
        self.bench = []
        self.hand = hand
        self.handCount = hand_count
        self.deckCount = deck_count
        self.prize = prize
        self.discard = discard
        self.poisoned = self.burned = self.asleep = self.paralyzed = self.confused = False


class _FullState:
    def __init__(self, me, opp):
        self.yourIndex = 0
        self.players = [me, opp]     # players[yourIndex]=me / players[1-yourIndex]=opp
        self.turn = 7
        self.firstPlayer = 0
        self.supporterPlayed = False
        self.stadiumPlayed = False
        self.energyAttached = False
        self.retreated = False
        self.stadium = []
        self.result = -1


def test_F13_hidden_leakage_invariant_to_opp_hidden():
    # me は両ケースで同一。opp は「手札の中身」と「サイドの中身」だけ差し替え、枚数は同一。
    me = _Player(hand=[], hand_count=5, deck_count=30, prize=[_Card(1)] * 4, discard=[])
    opp_a = _Player(hand=[_Card(101), _Card(102)], hand_count=2, deck_count=28,
                    prize=[_Card(901)] * 4, discard=[])
    opp_b = _Player(hand=[_Card(201), _Card(202)], hand_count=2, deck_count=28,
                    prize=[_Card(802)] * 4, discard=[])
    va = encoder.encode_state_from_state(_FullState(me, opp_a))
    vb = encoder.encode_state_from_state(_FullState(me, opp_b))
    assert va == vb, "相手の手札/サイドの中身の違いが Value 入力(encoder)に漏れている"
    # 対照: 相手の handCount(公開)を変えたら Value 入力は変わるべき(encoder が枚数は読む証明)。
    opp_c = _Player(hand=[_Card(101)], hand_count=1, deck_count=28, prize=[_Card(901)] * 4, discard=[])
    vc = encoder.encode_state_from_state(_FullState(me, opp_c))
    assert vc != va, "公開の handCount 変化が Value 入力に反映されていない(encoder が壊れている)"


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"  FAIL  {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(_run_all())
