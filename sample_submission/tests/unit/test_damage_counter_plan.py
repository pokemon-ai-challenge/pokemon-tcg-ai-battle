"""ptcg_ai.search.damage_counter_plan の単体テスト。

ファントムダイブのダメカン配置プランナ(サイドを取り切るまでの攻撃回数を最小化するDP)を、
cg エンジンを起動せずに純粋なデータ構造だけで検証する。

sample_submission/ から:
    python -m pytest tests/unit/test_damage_counter_plan.py -q
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from ptcg_ai.search.damage_counter_plan import (  # noqa: E402
    DamagePlanConfig,
    DamageState,
    TargetState,
    choose_next_counter,
    solve_damage_plan,
)

# 実戦相当の既定値(ファントムダイブ: 直接200 + ダメカン6個)。
PHANTOM = DamagePlanConfig()
PHANTOM_H1 = DamagePlanConfig(horizon=1)


# ---------------------------------------------------------------------------
# ビルダ
# ---------------------------------------------------------------------------


def active(key: int, hp10: int, prize: int = 2, *, immune: bool = False) -> TargetState:
    """バトルポケモン(ダメカンは置けない=直接攻撃だけが当たる)。"""
    return TargetState(
        key=key,
        hp10=hp10,
        prize=prize,
        is_active=True,
        counter_protected=False,
        threat=0.0,
        direct_immune=immune,
    )


def bench(
    key: int, hp10: int, prize: int = 1, *, protected: bool = False, threat: float = 0.0
) -> TargetState:
    """ベンチポケモン(ダメカンの配置先)。"""
    return TargetState(
        key=key,
        hp10=hp10,
        prize=prize,
        is_active=False,
        counter_protected=protected,
        threat=threat,
    )


def state(targets, remaining_prizes: int, boss_budget: int = 0) -> DamageState:
    return DamageState(
        targets=tuple(targets), remaining_prizes=remaining_prizes, boss_budget=boss_budget
    )


# ---------------------------------------------------------------------------
# 1. 60/200 の境界
# ---------------------------------------------------------------------------


def test_bench_60hp_is_ko_by_six_counters_but_70hp_is_not():
    """残HP60(6単位)はダメカン6個でKOできる。70は届かない。"""
    # アクティブは 300HP なので直接攻撃(200)では落ちない = ベンチKOだけが勝ち筋。
    plan60 = solve_damage_plan(state([active(1, 30), bench(2, 6)], 1), config=PHANTOM_H1)
    assert plan60 is not None
    assert plan60.turns_to_win == 1
    assert plan60.counter_alloc == ((2, 6),)

    plan70 = solve_damage_plan(state([active(1, 30), bench(2, 7)], 1), config=PHANTOM_H1)
    assert plan70 is not None
    assert plan70.turns_to_win is None  # 1ターンでは取り切れない
    # LB は「ベンチにも直接攻撃を通せる」まで緩和した楽観的下界なので 1 のまま。
    # 実際に何ターン要るかは DP が答える(2ターン: 6個 + 1個)。
    assert plan70.lower_bound == 1
    plan70_h2 = solve_damage_plan(state([active(1, 30), bench(2, 7)], 1), config=PHANTOM)
    assert plan70_h2.turns_to_win == 2


def test_active_200hp_dies_to_direct_but_210hp_needs_one_more_counter():
    """残HP200のアクティブは直接攻撃1回でKO。210は「直接1回 + ダメカン1個」が要る。

    ダメカンはベンチにしか置けないので、210のアクティブは1ターンでは絶対に落ちない
    (ボスの指令で他を引きずり出しても、210がベンチに落ちる代わりに直接200を受けない)。
    残った10は次のターンにダメカン1個で取れる。
    """
    plan200 = solve_damage_plan(state([active(1, 20, prize=1), bench(2, 30)], 1), config=PHANTOM_H1)
    assert plan200 is not None
    assert plan200.turns_to_win == 1
    assert plan200.direct_target == 1

    board210 = [active(1, 21, prize=1), bench(2, 30, prize=1)]
    one_turn = solve_damage_plan(state(board210, 1, boss_budget=1), config=PHANTOM_H1)
    assert one_turn is not None
    assert one_turn.turns_to_win is None  # 直接200では10残る / ダメカンはアクティブに置けない
    assert solve_damage_plan(state(board210, 1), config=PHANTOM).turns_to_win == 2

    # 「直接200を受けて10になった相手」がベンチに居るなら、ダメカン1個で取れる。
    left10 = solve_damage_plan(state([active(1, 30, prize=2), bench(2, 1, prize=1)], 1),
                               config=PHANTOM_H1)
    assert left10 is not None
    assert left10.turns_to_win == 1
    assert left10.counter_alloc == ((2, 1),)  # 残り5個は置き場が無く捨てられる


# ---------------------------------------------------------------------------
# 2. 同時複数KO
# ---------------------------------------------------------------------------


def test_two_bench_targets_are_knocked_out_together():
    """残HP30/30のベンチ2体は6個で同時にKOしてサイド2枚を取る。"""
    plan = solve_damage_plan(
        state([active(1, 30), bench(2, 3), bench(3, 3)], 2), config=PHANTOM_H1
    )
    assert plan is not None
    assert plan.turns_to_win == 1
    assert plan.counter_alloc == ((2, 3), (3, 3))
    assert plan.projected_prizes == (2,)


# ---------------------------------------------------------------------------
# 3. 仕込みの優先(中核価値)
# ---------------------------------------------------------------------------


def test_prefers_setup_for_two_prizes_next_turn_over_one_prize_now():
    """「今1枚取れるが次ターン0枚」より「今0枚でも次ターン2枚」を選ぶ。

    盤面: 残りサイド2枚。ベンチAは60HP(サイド1)で今すぐ落とせる。ベンチBは120HP
    (ex=サイド2)で、6個では落ちないが2ターンに分ければ落ちる。
    A を取ると残り1枚だが、次ターンは B(残60)を落とし切れず勝てない。
    B に6個積むと次ターンの6個で B が落ちて 2枚 = 勝ち。
    """
    targets = [active(1, 30, prize=2), bench(2, 6, prize=1), bench(3, 12, prize=2)]

    assert choose_next_counter(targets, 6, 2, config=PHANTOM) == 3

    # horizon=1(先読み無し)なら目先の1枚を取る = 先読みが効いていることの対照。
    assert choose_next_counter(targets, 6, 2, config=PHANTOM_H1) == 2

    # solve_damage_plan は「これから殴るターン」を解くので、上と同じ判断を見るには
    # 直接攻撃が効かないアクティブ(クラブネイル等)にしてダメカンだけの比較にする。
    # (アクティブが直接200で落ちる盤面なら、A を取ってから殴っても2ターンで勝てるため
    #  仕込みと同点になる = プランナはその場合ちゃんと早くサイドを取る方を選ぶ。)
    plan_targets = [
        active(1, 30, prize=2, immune=True),
        bench(2, 6, prize=1),
        bench(3, 12, prize=2),
    ]
    plan = solve_damage_plan(state(plan_targets, 2), config=PHANTOM)
    assert plan.turns_to_win == 2
    assert plan.counter_alloc == ((3, 6),)
    assert plan.projected_prizes == (0, 2)


# ---------------------------------------------------------------------------
# 4. LB の健全性(プロパティテスト)
# ---------------------------------------------------------------------------

# 総当たり参照実装用の小さな設定(状態空間を小さく保つ)。
SMALL = DamagePlanConfig(horizon=3, direct_damage10=4, counters=3, max_states=2_000_000)


def _ref_min_turns(targets, remaining_prizes: int, config: DamagePlanConfig, depth: int):
    """参照実装: 同じモデル上での最短攻撃ターン数を総当たりで求める(深さ制限つき)。

    プランナと同じ仮定に従う:
      * 直接攻撃はアクティブのみ(アクティブ不在なら楽観的に自分で選べる)。
      * ダメカンはアクティブ以外・ダメカン不可でない対象にのみ、合計 counters 個まで。
      * KO で対象は消え、remaining_prizes がその prize 分減る。
    ダメカンは「オーバーキルしない・余らせてもよい」形で全配分を列挙する(プランナが
    取り得る配分の上位集合。ダメージは多いほど得なので最短ターン数は一致する)。
    """
    memo: dict[tuple, int | None] = {}

    def norm(ts, remaining):
        return (tuple(sorted(ts)), remaining)

    def rec(ts, remaining, budget):
        if remaining <= 0:
            return 0
        if budget <= 0 or not ts:
            return None
        key = (norm(ts, remaining), budget)
        if key in memo:
            return memo[key]
        best = None
        actives = [t for t in ts if t[3]]
        direct_options = [t[0] for t in actives] if actives else [t[0] for t in ts]
        for dkey in direct_options:
            after_direct = []
            rem = remaining
            for t in ts:
                k, hp, pz, _is_act, prot, imm = t
                is_act = 1 if k == dkey else 0
                if k == dkey and not imm:
                    hp -= config.direct_damage10
                    if hp <= 0:
                        rem -= pz
                        continue
                after_direct.append((k, hp, pz, is_act, prot, imm))
            if rem <= 0:
                best = 1 if best is None else min(best, 1)
                continue
            slots = [t for t in after_direct if not t[3] and not t[4]]
            ranges = [range(0, min(t[1], config.counters) + 1) for t in slots]
            for combo in itertools.product(*ranges) if ranges else [()]:
                if sum(combo) > config.counters:
                    continue
                nxt = []
                rem2 = rem
                amount = {slots[i][0]: combo[i] for i in range(len(slots))}
                for t in after_direct:
                    k, hp, pz, is_act, prot, imm = t
                    hp -= amount.get(k, 0)
                    if hp <= 0:
                        rem2 -= pz
                        continue
                    nxt.append((k, hp, pz, is_act, prot, imm))
                sub = rec(tuple(nxt), rem2, budget - 1)
                if sub is None:
                    continue
                total = 1 + sub
                if best is None or total < best:
                    best = total
        memo[key] = best
        return best

    ts = tuple(
        (t.key, t.hp10, t.prize, 1 if t.is_active else 0, 1 if t.counter_protected else 0,
         1 if t.direct_immune else 0)
        for t in targets
        if t.hp10 > 0
    )
    return rec(ts, remaining_prizes, depth)


def _random_board(rng: random.Random):
    n = rng.randint(1, 3)
    targets = []
    for i in range(n):
        targets.append(
            TargetState(
                key=i + 1,
                hp10=rng.randint(1, 6),
                prize=rng.randint(0, 2),
                is_active=(i == 0 and rng.random() < 0.8),
                counter_protected=rng.random() < 0.2,
                threat=0.0,
                direct_immune=rng.random() < 0.15,
            )
        )
    remaining = rng.randint(1, 3)
    return targets, remaining


def test_lower_bound_never_exceeds_true_minimum_turns():
    """LB(s) は総当たりで得た真の最短ターン数を超えない(下界として健全)。"""
    rng = random.Random(20260814)
    checked = 0
    for _ in range(100):
        targets, remaining = _random_board(rng)
        plan = solve_damage_plan(state(targets, remaining), config=SMALL)
        assert plan is not None
        truth = _ref_min_turns(targets, remaining, SMALL, depth=SMALL.horizon)
        if truth is None:
            continue  # horizon 内では取り切れない盤面(LB との比較対象が無い)
        checked += 1
        assert plan.lower_bound <= truth, (targets, remaining, plan, truth)
        # horizon 内で解けるなら、DP の最短ターン数は総当たりと一致する。
        assert plan.turns_to_win == truth, (targets, remaining, plan, truth)
    assert checked >= 20  # 比較できた盤面が十分あること


# ---------------------------------------------------------------------------
# 5. 配置不可の対象
# ---------------------------------------------------------------------------


def test_counter_protected_target_is_never_allocated():
    """ダメカンを置けない対象(特性/ミストエネルギー等)には配分しない。"""
    targets = [active(1, 30), bench(2, 3, protected=True), bench(3, 6)]
    plan = solve_damage_plan(state(targets, 2), config=PHANTOM)
    assert plan is not None
    assert all(key != 2 for key, _count in plan.counter_alloc)
    for _ in range(10):
        assert choose_next_counter(targets, 6, 2, config=PHANTOM) != 2

    # 置ける対象が1体も無ければ None(呼び出し側は従来ロジックへフォールバック)。
    assert choose_next_counter([active(1, 30), bench(2, 3, protected=True)], 6, 2,
                               config=PHANTOM) is None


# ---------------------------------------------------------------------------
# 6. 決定性
# ---------------------------------------------------------------------------


def test_deterministic_for_identical_input():
    """同一入力を100回与えても同じ答えを返す。"""
    targets = [
        active(1, 26, prize=2),
        bench(2, 6, prize=1, threat=1.5),
        bench(3, 6, prize=1, threat=0.5),
        bench(4, 13, prize=2),
        bench(5, 4, prize=1),
    ]
    first_key = choose_next_counter(targets, 6, 3, config=PHANTOM)
    first_plan = solve_damage_plan(state(targets, 3), config=PHANTOM)
    for _ in range(100):
        assert choose_next_counter(targets, 6, 3, config=PHANTOM) == first_key
        assert solve_damage_plan(state(targets, 3), config=PHANTOM) == first_plan


# ---------------------------------------------------------------------------
# 7. 打ち切り安全
# ---------------------------------------------------------------------------


def test_truncation_still_returns_a_valid_choice():
    """max_states を極端に絞っても例外を投げず、妥当な手を返す。"""
    tight = DamagePlanConfig(max_states=100)
    targets = [
        active(1, 30, prize=2),
        bench(2, 9, prize=2),
        bench(3, 11, prize=2),
        bench(4, 7, prize=1),
        bench(5, 6, prize=1),
        bench(6, 12, prize=2),
    ]
    key = choose_next_counter(targets, 6, 4, config=tight)
    assert key in {2, 3, 4, 5, 6}
    plan = solve_damage_plan(state(targets, 4), config=tight)
    assert plan is not None
    assert plan.lower_bound >= 1
    assert sum(count for _key, count in plan.counter_alloc) <= tight.counters


# ---------------------------------------------------------------------------
# 8. オーバーキル回避
# ---------------------------------------------------------------------------


def test_does_not_waste_counters_on_a_nearly_dead_target():
    """残10のベンチを取ってもサイドが足りない状況では、ダメカンを集中して勝ちを取る。"""
    # 残りサイド2枚。A(残10/サイド1)を取っても勝てない。B(残60/サイド2)に6個集中すれば勝ち。
    targets = [active(1, 30, prize=2), bench(2, 1, prize=1), bench(3, 6, prize=2)]
    assert choose_next_counter(targets, 6, 2, config=PHANTOM) == 3

    plan = solve_damage_plan(state(targets, 2), config=PHANTOM)
    assert plan is not None
    assert plan.turns_to_win == 1
    assert plan.counter_alloc == ((3, 6),)


def test_allocation_never_exceeds_target_hp():
    """どの対象にも残HPを超える個数は置かない(オーバーキルはモデル上あり得ない)。"""
    rng = random.Random(7)
    for _ in range(50):
        targets, remaining = _random_board(rng)
        plan = solve_damage_plan(state(targets, remaining), config=PHANTOM)
        if plan is None:
            continue
        hp_by_key = {t.key: t.hp10 for t in targets}
        for key, count in plan.counter_alloc:
            assert count <= hp_by_key[key], (targets, plan)
        assert sum(count for _k, count in plan.counter_alloc) <= PHANTOM.counters
