#!/usr/bin/env python3
"""og_r11(Fix-I ``ability_draw_brake``)の再現ゲート: リプレイの実局面を固定入力にした
PASS/FAIL 判定。

相手方策を回さない**主判定**。実ラダーの実局面をそのまま観測として食わせ、og_r10 と og_r11 で
挙動がどう変わるかを固定する。対戦を回さないので分散が無く、リグレッションの検出に使える
(`_snapshot_gate_r8.py`/`_snapshot_gate_r10.py` と同じ流儀・同じユーティリティ再利用。
あちらは書き換えずに別ファイル)。

**背景**: 実ラダー32敗中3敗(9.4%)が自分の山札0による敗北。真犯人はリーリエ/クラウン
(型は既に `low_deck_draw_brake` で対処済み)ではなく、オーガポン みどりのめん ex(96)の
特性「みどりのまい」(自分自身にエネ装着+1ドロー、各個体1回/ターン)。episode 93517227
(壁無し・サイド2-6でリードし毎ターン1サイド獲得中)は T9〜T11 でこの特性を連打し、
山札 5→2→0 で敗北した。

対象(すべて `kaggle_replays/replays/episode-*-replay.json`。デッキは `steps[1]` の60枚回答=
その試合で実際に使ったデッキを使う。リポジトリの deck.csv は別デッキに変わっているため):

  I1 fires (a) 93517227 row122 (T9)
      アクティブのオーガポン(8エネ)で相手アクティブ(HP70)へ攻撃すれば
      まんようしぐれ=30+30×(8+1)=300ダメージで確実にKOできる状態なのに、
      ATTACKではなく別個体(ベンチ)のみどりのまいを選んだ局面。「足す必要が無い」=(a)。
      og_r11 が別の手に差し替えるか。

  I1 fires (a) 93517227 row146 (T11)
      アクティブのオーガポン(4エネ)で相手アクティブ(HP100)へ攻撃すれば
      30+30×(4+1)=180ダメージで足りる状態なのに、同じくベンチのみどりのまいを選んだ局面。

  I2 control (deck too high) 93517227 row11 (T1、山札43)
      閾値(既定8)を大きく上回る序盤にみどりのまいを使った局面。og_r11 も不介入(この局面は
      ATTACK選択肢も無いため、山札閾値と絶対例外の両方で二重にブロックされる)。

  I3 control (attack unavailable, 絶対例外) 93517227 row158 (T13、山札0)
      山札が既に0でも、選択肢に ATTACK が無い(=攻撃コスト未充足)局面ではvetoしない。
      デッキ切れ寸前でもエネ加速(攻撃可能化)を止めてはいけない、という絶対例外の実例。

  I4 parity  上記すべての局面で og_r10(新キー無し)が従来挙動のままであること。

使い方:
    python kaggle_replays/_snapshot_gate_r11.py
    python kaggle_replays/_snapshot_gate_r11.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# 読み出し/表示ユーティリティは既存診断ハーネスをそのまま使う(sys.path と cwd の設定も
# 向こうで完結している)。重複実装を作らない。
sys.path.insert(0, str(_HERE))
import _probe_missed_lethal as pm  # noqa: E402

from cg.api import OptionType, to_observation_class  # noqa: E402
from ptcg_ai.learning import encoder as _enc  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402

_CONFIG_DIR = _ROOT / "sample_submission" / "configs"


def load_agent_config(name: str) -> dict[str, Any]:
    return json.loads((_CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8"))


R10 = load_agent_config("abl_5_full_og_r10")
R11 = load_agent_config("abl_5_full_og_r11")

_OGERPON = 96


# --------------------------------------------------------------------------
# 局面の取り出し(_snapshot_gate_r8.py / _snapshot_gate_r10.py と同じ流儀)
# --------------------------------------------------------------------------

_replay_cache: dict[int, tuple[dict[int, dict], list[int]]] = {}


def episode_rows(episode_id: int) -> tuple[dict[int, dict], list[int]]:
    cached = _replay_cache.get(episode_id)
    if cached is None:
        replay = pm.load_replay(episode_id)
        own = pm.own_index_of(replay)
        deck = pm.own_deck_of(replay, own)
        rows = {d["row"]: d for d in pm.own_decisions(replay, own)}
        cached = (rows, deck)
        _replay_cache[episode_id] = cached
    return cached


def use_deck(deck: list[int]) -> None:
    """ml_policy_agent が読むデッキを、その試合で実際に使ったデッキに差し替える。"""
    mpa._deck_cache = list(deck)
    mpa._model = None
    mpa._model_cache_by_weights_path.clear()


def get_case(episode_id: int, row: int):
    rows, deck = episode_rows(episode_id)
    dec = rows[row]
    obs = to_observation_class(dec["obs_dict"])
    use_deck(deck)
    return obs, deck, list(dec["answer"] or [])


def _option_card_ids(obs) -> list[int | None]:
    return [_enc._resolve_card_id(o, obs.current) for o in obs.select.option]


def describe_action(obs, action: list[int] | None) -> str:
    if action is None:
        return "None"
    ids = _option_card_ids(obs)
    parts = []
    for i in action:
        opt_type = pm.enum_name(OptionType, obs.select.option[i].type)
        card = pm.card_name(ids[i]) if ids[i] is not None else "-"
        parts.append(f"{i}:{opt_type}/{card}")
    return "[" + ", ".join(parts) + "]"


class Gate:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
        return bool(ok)

    def summary(self) -> int:
        failed = [r for r in self.results if not r[1]]
        print("\n" + "=" * 100)
        print(f"SNAPSHOT GATE r11: {len(self.results) - len(failed)}/{len(self.results)} PASS")
        for name, _ok, detail in failed:
            print(f"  FAIL: {name}" + (f"  — {detail}" if detail else ""))
        return 1 if failed else 0


# --------------------------------------------------------------------------
# I1: 特性ドロー(みどりのまい型)の垂れ流し抑制(Fix-I)
# --------------------------------------------------------------------------

I1_FIRE_CASES = [
    (93517227, 122, "T9 山5: アクティブ8エネで相手アクティブHP70へ攻撃すれば確実にKO(足す必要無し)"),
    (93517227, 146, "T11 山5: アクティブ4エネで相手アクティブHP100へ攻撃すれば足りる(足す必要無し)"),
]


def run_i1(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("I1 ability_draw_brake: みどりのまい連打を、既にKO可能な局面でvetoするか")
    for episode_id, row, note in I1_FIRE_CASES:
        obs, _deck, answer = get_case(episode_id, row)
        ids = _option_card_ids(obs)
        state = obs.current
        me = state.yourIndex
        print(f"\n  {episode_id} row{row} (T{state.turn}): {note}")
        print(f"    山{state.players[me].deckCount} 実戦={describe_action(obs, answer)}")
        gate.check(f"I1-0 実戦の選択がみどりのまい(96)である ({episode_id} row{row})",
                   len(answer) == 1 and ids[answer[0]] == _OGERPON, f"cardId={ids[answer[0]]}")

        r10_out = mpa._try_ability_draw_brake(obs, list(answer), config=R10)
        r11_out = mpa._try_ability_draw_brake(obs, list(answer), config=R11)
        chain_r10 = mpa._apply_action_vetoes(obs, list(answer), config=R10)
        chain_r11 = mpa._apply_action_vetoes(obs, list(answer), config=R11)
        print(f"    og_r10 -> {describe_action(obs, r10_out)}  / 連鎖 {describe_action(obs, chain_r10)}")
        print(f"    og_r11 -> {describe_action(obs, r11_out)}  / 連鎖 {describe_action(obs, chain_r11)}")

        gate.check(f"I1-a og_r10 は不介入(新キー無し) ({episode_id} row{row})",
                   r10_out is None, describe_action(obs, r10_out))
        replaced = (r11_out is not None and r11_out != answer
                    and ids[r11_out[0]] != _OGERPON)
        gate.check(f"I1-b og_r11 がみどりのまいを別の手へ差し替え ({episode_id} row{row})",
                   bool(replaced), describe_action(obs, r11_out))
        gate.check(f"I1-c veto連鎖でも同じ結論 ({episode_id} row{row})",
                   chain_r11 == (r11_out or answer), describe_action(obs, chain_r11))
        stats = mpa.get_guard_stats()
        gate.check(f"I1-d 誤爆番人(攻撃不能なのに発火)が立っていない ({episode_id} row{row})",
                   stats.get("ability_draw_brake_misfire_no_attack", 0) == 0,
                   str(stats.get("ability_draw_brake_misfire_no_attack")))


# --------------------------------------------------------------------------
# I2: 対照(山札閾値を大きく上回る序盤)
# --------------------------------------------------------------------------

I2_CASE = (93517227, 11, "T1 山43: 閾値を大きく上回る序盤のみどりのまい(ATTACK選択肢も無い)")


def run_i2(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("I2 control: 山札が十分残っている序盤には介入しない")
    episode_id, row, note = I2_CASE
    obs, _deck, answer = get_case(episode_id, row)
    ids = _option_card_ids(obs)
    state = obs.current
    me = state.yourIndex
    print(f"\n  {episode_id} row{row} (T{state.turn}): {note}")
    print(f"    山{state.players[me].deckCount} 実戦={describe_action(obs, answer)}")
    gate.check("I2-0 実戦の選択がみどりのまい(96)である",
               len(answer) == 1 and ids[answer[0]] == _OGERPON, f"cardId={ids[answer[0]]}")

    r10_out = mpa._try_ability_draw_brake(obs, list(answer), config=R10)
    r11_out = mpa._try_ability_draw_brake(obs, list(answer), config=R11)
    print(f"    og_r10 -> {describe_action(obs, r10_out)}")
    print(f"    og_r11 -> {describe_action(obs, r11_out)}")
    gate.check("I2-a og_r10 は不介入", r10_out is None, describe_action(obs, r10_out))
    gate.check("I2-b og_r11 も不介入(山札が閾値超)", r11_out is None, describe_action(obs, r11_out))


# --------------------------------------------------------------------------
# I3: 絶対例外(攻撃コスト未充足なら山札0でもvetoしない)
# --------------------------------------------------------------------------

I3_CASE = (93517227, 158, "T13 山0: ATTACK選択肢が無い(攻撃コスト未充足)のでvetoしない絶対例外")


def run_i3(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("I3 control: 攻撃不能局面は山札0でも絶対にvetoしない")
    episode_id, row, note = I3_CASE
    obs, _deck, answer = get_case(episode_id, row)
    ids = _option_card_ids(obs)
    state = obs.current
    me = state.yourIndex
    has_attack = any(o.type == OptionType.ATTACK for o in obs.select.option)
    print(f"\n  {episode_id} row{row} (T{state.turn}): {note}")
    print(f"    山{state.players[me].deckCount} ATTACK選択肢={has_attack} "
          f"実戦={describe_action(obs, answer)}")
    gate.check("I3-0 実戦の選択がみどりのまい(96)である",
               len(answer) == 1 and ids[answer[0]] == _OGERPON, f"cardId={ids[answer[0]]}")
    gate.check("I3-0' 前提: ATTACK選択肢が無い", not has_attack, str(has_attack))

    mpa.reset_guard_stats()
    r11_out = mpa._try_ability_draw_brake(obs, list(answer), config=R11)
    # `_apply_action_vetoes` の全チェーンには r10 で既に有効な `energy_to_active_first`
    # (ATTACK選択肢が無い decision でベンチ→アクティブへ振り替える、r10 Fix-G)も含まれるため、
    # 「連鎖の結果が実戦の生答えと一致する」ではなく「r10単独の結論(chain_r10)と一致する」
    # ことをチェックする(=Fix-I自体は何も変えていない、という切り分け)。
    chain_r10 = mpa._apply_action_vetoes(obs, list(answer), config=R10)
    chain_r11 = mpa._apply_action_vetoes(obs, list(answer), config=R11)
    print(f"    og_r10 連鎖 -> {describe_action(obs, chain_r10)}"
          f"(energy_to_active_first がATTACK不能でベンチ→アクティブへ振替済み)")
    print(f"    og_r11 -> {describe_action(obs, r11_out)}  / 連鎖 {describe_action(obs, chain_r11)}")
    gate.check("I3-a og_r11 は不介入(絶対例外)", r11_out is None, describe_action(obs, r11_out))
    gate.check("I3-b veto連鎖でも r10単独の結論からFix-Iは何も変えない",
               chain_r11 == chain_r10, f"chain_r10={chain_r10} chain_r11={chain_r11}")
    stats = mpa.get_guard_stats()
    gate.check("I3-c 発火カウンタが一切動いていない",
               stats.get("ability_draw_brake_fired", 0) == 0
               and stats.get("ability_draw_brake_misfire_no_attack", 0) == 0,
               str({k: v for k, v in stats.items() if k.startswith("ability_draw_brake")}))


# --------------------------------------------------------------------------
# I4: og_r10(新キー無し)の従来挙動が保たれていること
# --------------------------------------------------------------------------

_I4_CASES = [(93517227, 122), (93517227, 146), (93517227, 11), (93517227, 158)]


def run_i4(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("I4 parity: og_r10(新キー無し)は全局面で ability_draw_brake が一切発火しない")
    # 注意: `_apply_action_vetoes` のフルチェーンは r10 で既に有効な他ガード
    # (row158 は `energy_to_active_first`、ATTACK不能でベンチ→アクティブへ振替)を含むため
    # `chain == answer` にはならない局面がある。ここで固定したいのは
    # 「Fix-I(`_try_ability_draw_brake` 単体)が r10 config では絶対に None を返す」ことだけ
    # (それ以外のチェーンの結論は r8/r10 のスナップショットゲート側の責務)。
    ok = True
    detail: list[str] = []
    for episode_id, row in _I4_CASES:
        obs, _deck, answer = get_case(episode_id, row)
        if not answer:
            continue
        out = mpa._try_ability_draw_brake(obs, list(answer), config=R10)
        if out is not None:
            ok = False
            detail.append(f"{episode_id} row{row} ability_draw_brake -> {out}")
    gate.check("I4-a og_r10 では ability_draw_brake が一切発火しない", ok, "; ".join(detail))


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="og_r11 の実局面スナップショットゲート")
    ap.add_argument("--json", default=None, help="判定結果のJSON出力先")
    args = ap.parse_args()

    print(f"configs: og_r10={_CONFIG_DIR / 'abl_5_full_og_r10.json'}")
    print(f"         og_r11={_CONFIG_DIR / 'abl_5_full_og_r11.json'}")
    gate = Gate()
    mpa.reset_guard_stats()
    run_i1(gate)
    run_i2(gate)
    run_i3(gate)
    run_i4(gate)
    print("\nguard stats(このゲート全体の発火数):",
          json.dumps({k: v for k, v in mpa.get_guard_stats().items() if v}, ensure_ascii=False))
    code = gate.summary()
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"results": [{"name": n, "pass": ok, "detail": d} for n, ok, d in gate.results],
             "guard_stats": mpa.get_guard_stats()},
            ensure_ascii=False, indent=1), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
