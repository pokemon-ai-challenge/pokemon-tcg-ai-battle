#!/usr/bin/env python3
"""og_r13(閉形式KOファストパス ``use_closed_form_ko``)の再現ゲート。

相手方策を回さない**主判定**。実ラダーの実局面をそのまま観測として食わせ、og_r12 と og_r13 で
挙動がどう変わるかを固定する(`_snapshot_gate_r8/r10/r11/r12.py` と同じ流儀・同じユーティリティ
再利用。あちらは書き換えずに別ファイル)。

**背景**: `ability_draw_brake`(r11 Fix-I)と `low_deck_draw_brake`(r9 Fix-D)は、この
スナップショットでは正しく発火するのに **実対戦120試合で発火0回** だった。KO可否を
エンジン探索(`ko_search.can_ko_this_turn` / `ability_draw_eval.already_ko_without_more_energy`)
で判定しており、みどりのまいを持つオーガポンが複数体並ぶ分岐の大きい盤面では 300ms 予算内に
完走せず、すべて「判定不能」→安全側で介入見送りになっていたため(=ガードが実質的に死んでいた)。

r13 は判定を **閉形式ファストパス → 従来の探索 → それも判定不能なら不介入** の3段にする。
閉形式(`ptcg_ai/search/closed_form_ko.py`)はこのデッキの唯一のアタッカーである
オーガポン みどりのめん ex(96)のワザ「まんようしぐれ」(attackId 120)
= 30 + 30 ×(両バトルポケモンのエネ数)を既知パターンとして持つ。

固定する項目:

  K1 同値性 (a): 93517227 row122 / row146。og_r12 と og_r13 が**同じ結論**を出すこと。
      ただし og_r13 は閉形式で解決し(``closed_form_ko_true`` が立つ)、
      エンジン探索(`ko_search` / `ability_draw_eval`)を**一度も呼ばない**こと。

  K2 打点の閉形式が実ログと一致: 上記局面で `estimate_current_damage` が
      実リプレイの実ダメージ値(row126=300 / row148=180)と一致すること。

  K3 対照(不介入の維持): 93517227 row11(山43=閾値超)/ row158(ATTACK選択肢なし=絶対例外)
      で og_r13 も不介入。row158 は閉形式が「KOできる」と言っても採用されないこと
      (カブルモ506「スノットアップ」でワザが封じられている実例)。

  K4 parity: og_r12(``use_closed_form_ko`` 無し)では閉形式が**一度も呼ばれない**。

使い方:
    python kaggle_replays/_snapshot_gate_r13.py
    python kaggle_replays/_snapshot_gate_r13.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# 局面ローダ・表示ヘルパは r11 のゲートをそのまま再利用する(重複実装を作らない)。
import _snapshot_gate_r11 as g11  # noqa: E402

from cg.api import OptionType  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.search import ability_draw_eval, closed_form_ko, ko_search  # noqa: E402

R12 = g11.load_agent_config("abl_5_full_og_r12")
R13 = g11.load_agent_config("abl_5_full_og_r13")

_OGERPON = 96


class _SearchSpy:
    """エンジン探索が呼ばれたかを数えるためのモンキーパッチ(呼ばれたら実物に委譲)。"""

    def __init__(self) -> None:
        self.can_ko_calls = 0
        self.already_ko_calls = 0
        self._orig_can_ko = ko_search.can_ko_this_turn
        self._orig_already = ability_draw_eval.already_ko_without_more_energy

    def __enter__(self):
        def _can_ko(*a, **k):
            self.can_ko_calls += 1
            return self._orig_can_ko(*a, **k)

        def _already(*a, **k):
            self.already_ko_calls += 1
            return self._orig_already(*a, **k)

        ko_search.can_ko_this_turn = _can_ko
        ability_draw_eval.already_ko_without_more_energy = _already
        return self

    def __exit__(self, *exc):
        ko_search.can_ko_this_turn = self._orig_can_ko
        ability_draw_eval.already_ko_without_more_energy = self._orig_already
        return False

    @property
    def total(self) -> int:
        return self.can_ko_calls + self.already_ko_calls


# --------------------------------------------------------------------------
# K1 / K2: 発火局面で「同じ結論」を「探索なし」で出す
# --------------------------------------------------------------------------

# (episode, row, 実リプレイのログに残っている実ダメージ値, 説明)
# 実ダメージは `_probe_closed_form_ko.py` が同エピソードの ATTACK ログから読んだ値。
# row122/row146 は ABILITY を選んだ decision で、その2行後の ATTACK 決定(row126/row148)の
# 実ログが同じ盤面のエネ数から出た打点になっている。
K1_CASES = [
    (93517227, 122, 300, "T9 山5: アクティブ8エネ+相手1エネ → 30+30×9=300 で相手HP70をKO"),
    (93517227, 146, 180, "T11 山5: アクティブ4エネ+相手1エネ → 30+30×5=180 で相手HP100をKO"),
]


def run_k1(gate: g11.Gate) -> None:
    print("\n" + "=" * 100)
    print("K1/K2 閉形式ファストパス: og_r12 と同じ結論を、エンジン探索を呼ばずに出す")
    for episode_id, row, expected_damage, note in K1_CASES:
        obs, _deck, answer = g11.get_case(episode_id, row)
        state = obs.current
        me = state.yourIndex
        print(f"\n  {episode_id} row{row} (T{state.turn}): {note}")

        # K2: 打点の閉形式が実ログ値と一致するか
        damage = closed_form_ko.estimate_current_damage(state, me)
        gate.check(f"K2-a 閉形式の打点が実ログと一致 ({episode_id} row{row})",
                   damage == expected_damage, f"閉形式={damage} 実ログ={expected_damage}")

        with _SearchSpy() as spy12:
            mpa.reset_guard_stats()
            out12 = mpa._try_ability_draw_brake(obs, list(answer), config=R12)
        with _SearchSpy() as spy13:
            mpa.reset_guard_stats()
            out13 = mpa._try_ability_draw_brake(obs, list(answer), config=R13)
            stats13 = mpa.get_guard_stats()
        print(f"    og_r12 -> {g11.describe_action(obs, out12)}  (探索呼び出し {spy12.total}回)")
        print(f"    og_r13 -> {g11.describe_action(obs, out13)}  (探索呼び出し {spy13.total}回)")

        gate.check(f"K1-a og_r12 が発火する(前提) ({episode_id} row{row})",
                   out12 is not None, g11.describe_action(obs, out12))
        gate.check(f"K1-b og_r13 が og_r12 と同じ結論 ({episode_id} row{row})",
                   out13 == out12, f"r12={out12} r13={out13}")
        gate.check(f"K1-c og_r13 はエンジン探索を一度も呼ばない ({episode_id} row{row})",
                   spy13.total == 0, f"can_ko={spy13.can_ko_calls} already_ko={spy13.already_ko_calls}")
        gate.check(f"K1-d 閉形式で解決したことがカウンタに出る ({episode_id} row{row})",
                   stats13.get("closed_form_ko_true", 0) == 1
                   and stats13.get("ability_draw_brake_closed_form_ko", 0) == 1,
                   str({k: v for k, v in stats13.items() if k.startswith("closed_form")}))
        gate.check(f"K1-e 誤爆番人(攻撃不能なのに発火)が立っていない ({episode_id} row{row})",
                   stats13.get("ability_draw_brake_misfire_no_attack", 0) == 0,
                   str(stats13.get("ability_draw_brake_misfire_no_attack")))


# --------------------------------------------------------------------------
# K3: 対照(不介入の維持)
# --------------------------------------------------------------------------

K3_CASES = [
    (93517227, 11, "T1 山43: 閾値超なので閉形式に到達する前に不介入"),
    (93517227, 158, "T13 山0: ATTACK選択肢が無い(スノットアップでワザ封じ)=絶対例外"),
]


def run_k3(gate: g11.Gate) -> None:
    print("\n" + "=" * 100)
    print("K3 control: og_r13 でも不介入のままであること")
    for episode_id, row, note in K3_CASES:
        obs, _deck, answer = g11.get_case(episode_id, row)
        state = obs.current
        me = state.yourIndex
        has_attack = any(o.type == OptionType.ATTACK for o in obs.select.option)
        print(f"\n  {episode_id} row{row} (T{state.turn}): {note}")
        print(f"    山{state.players[me].deckCount} ATTACK選択肢={has_attack}")

        mpa.reset_guard_stats()
        out13 = mpa._try_ability_draw_brake(obs, list(answer), config=R13)
        chain12 = mpa._apply_action_vetoes(obs, list(answer), config=R12)
        chain13 = mpa._apply_action_vetoes(obs, list(answer), config=R13)
        gate.check(f"K3-a og_r13 は不介入 ({episode_id} row{row})",
                   out13 is None, g11.describe_action(obs, out13))
        gate.check(f"K3-b veto連鎖でも og_r12 と同じ結論 ({episode_id} row{row})",
                   chain13 == chain12, f"r12={chain12} r13={chain13}")

    # row158 は「閉形式は KO できると言うが、エンジンが ATTACK 選択肢を出していない」実例。
    obs, _deck, _answer = g11.get_case(93517227, 158)
    state = obs.current
    me = state.yourIndex
    gate.check("K3-c 前提: row158 は閉形式単体では『KOできる』と出る(静的HP/エネだけでは見えない)",
               closed_form_ko.can_ko_now(state, me) is True,
               str(closed_form_ko.can_ko_now(state, me)))
    gate.check("K3-d その True がガードに採用されていない(ATTACK選択肢が無いので)",
               not any(o.type == OptionType.ATTACK for o in obs.select.option))


# --------------------------------------------------------------------------
# K4: parity(og_r12 では閉形式が一度も呼ばれない)
# --------------------------------------------------------------------------

_K4_CASES = [(93517227, 122), (93517227, 146), (93517227, 11), (93517227, 158)]


def run_k4(gate: g11.Gate) -> None:
    print("\n" + "=" * 100)
    print("K4 parity: og_r12(use_closed_form_ko 無し)では閉形式が一度も呼ばれない")
    calls = {"n": 0}
    orig_now = closed_form_ko.can_ko_now
    orig_imp = closed_form_ko.is_ko_impossible_this_turn

    def _spy_now(*a, **k):
        calls["n"] += 1
        return orig_now(*a, **k)

    def _spy_imp(*a, **k):
        calls["n"] += 1
        return orig_imp(*a, **k)

    closed_form_ko.can_ko_now = _spy_now
    closed_form_ko.is_ko_impossible_this_turn = _spy_imp
    try:
        for episode_id, row in _K4_CASES:
            obs, _deck, answer = g11.get_case(episode_id, row)
            if not answer:
                continue
            mpa._try_ability_draw_brake(obs, list(answer), config=R12)
            mpa._try_low_deck_draw_brake(obs, list(answer), config=R12)
    finally:
        closed_form_ko.can_ko_now = orig_now
        closed_form_ko.is_ko_impossible_this_turn = orig_imp
    gate.check("K4-a og_r12 では closed_form_ko が一度も呼ばれない", calls["n"] == 0,
               f"呼び出し {calls['n']}回")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="og_r13 の実局面スナップショットゲート")
    ap.add_argument("--json", default=None, help="判定結果のJSON出力先")
    args = ap.parse_args()

    print(f"configs: og_r12={g11._CONFIG_DIR / 'abl_5_full_og_r12.json'}")
    print(f"         og_r13={g11._CONFIG_DIR / 'abl_5_full_og_r13.json'}")
    problems = closed_form_ko.verify_known_attacks()
    gate = g11.Gate()
    gate.check("K0 既知パターンの定数がカードデータと一致", not problems, "; ".join(problems))
    mpa.reset_guard_stats()
    run_k1(gate)
    run_k3(gate)
    run_k4(gate)
    failed = [r for r in gate.results if not r[1]]
    print("\n" + "=" * 100)
    print(f"SNAPSHOT GATE r13: {len(gate.results) - len(failed)}/{len(gate.results)} PASS")
    for name, _ok, detail in failed:
        print(f"  FAIL: {name}" + (f"  — {detail}" if detail else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"results": [{"name": n, "pass": ok, "detail": d} for n, ok, d in gate.results]},
            ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
