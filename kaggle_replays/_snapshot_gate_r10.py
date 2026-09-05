#!/usr/bin/env python3
"""og_r10(Fix-F/G/H)の再現ゲート: リプレイの実局面を固定入力にした PASS/FAIL 判定。

相手方策を回さない**主判定**。実ラダーの実局面をそのまま観測として食わせ、og_r9 と og_r10 で
挙動がどう変わるかを固定する。対戦を回さないので分散が無く、リグレッションの検出に使える
(`_snapshot_gate_r8.py` と同じ流儀・同じユーティリティ再利用。あちらは書き換えずに別ファイル)。

**なぜスナップショットが主判定なのか**: 今回の3修正はいずれも実ラダー31敗中1〜3件の頻度で、
勝率A/B(ローカルのノイズ床 ±7pt / n=420)では原理的に判定できない。よって判定は
「実局面で正しい手を選ぶか」+「発火率・誤爆率メトリクス(`_metrics_gate.py`)」で行う。

対象(すべて `kaggle_replays/replays/episode-*-replay.json`。デッキは `steps[1]` の60枚回答=
その試合で実際に使ったデッキを使う。リポジトリの deck.csv は別デッキに変わっているため):

  H1 tool   93408551 row129 (T10)
      場がジャミングタワー(1246、どうぐの効果をすべて無効化)なのに ヒーローマント(1159、
      最大HP+100)をHP10のオーガポンに装着した局面。og_r10 が別の手に差し替えるか。
      同ターンの row130(装着後、選択肢にどうぐが無い)は**誤発火の対照**。

  H2 energy 93408551 rows84/86/88 (T6)
      アクティブのオーガポンが草2(まんようしぐれ=草3が必要=ATTACK選択肢が出ていない)なのに
      みどりのまいを3回ともベンチ個体に使った局面。og_r10 がアクティブの特性に振り替えるか。
      rows125/127(ATTACK選択肢がある=既に攻撃可能)は**誤発火の対照**。

  H3 search 93473767 row37 (T3)
      むしとりセットの候補(state.looking)にオーガポンex(96)が出ていたのに基本草エネ2枚を
      選んだ局面。og_r10 がポケモンを1枚拾うか。93408551 row131(候補が全部エネ=拾える
      ポケモンが無い)は**誤発火の対照**。

  H4 parity  上記すべての局面で og_r9(新キー無し)が従来挙動のままであること。

使い方:
    python kaggle_replays/_snapshot_gate_r10.py
    python kaggle_replays/_snapshot_gate_r10.py --json out.json
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

from cg.api import AreaType, OptionType, SelectType, to_observation_class  # noqa: E402
from ptcg_ai.learning import encoder as _enc  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402

_CONFIG_DIR = _ROOT / "sample_submission" / "configs"


def load_agent_config(name: str) -> dict[str, Any]:
    return json.loads((_CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8"))


R9 = load_agent_config("abl_5_full_og_r9")
R10 = load_agent_config("abl_5_full_og_r10")

_HERO_CAPE = 1159
_JAMMING_TOWER = 1246
_OGERPON = 96


# --------------------------------------------------------------------------
# 局面の取り出し(_snapshot_gate_r8.py と同じ流儀)
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


def describe_action(obs, action: list[int] | None) -> str:
    if action is None:
        return "None"
    state = obs.current
    me = state.yourIndex
    parts = []
    for i in action:
        option = obs.select.option[i]
        opt_type = pm.enum_name(OptionType, option.type)
        if getattr(option, "area", None) == AreaType.LOOKING:
            card = pm.card_name(mpa._looking_card_id(option, state))
        else:
            card = pm.option_card_name(option, state, me)
        target = ""
        if getattr(option, "inPlayArea", None) is not None:
            target = f"->{pm.enum_name(AreaType, option.inPlayArea)}{option.inPlayIndex}"
        elif option.type == OptionType.ABILITY and option.area is not None:
            target = f"@{pm.enum_name(AreaType, option.area)}{option.index}"
        parts.append(f"{i}:{opt_type}/{card}{target}")
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
        print(f"SNAPSHOT GATE r10: {len(self.results) - len(failed)}/{len(self.results)} PASS")
        for name, _ok, detail in failed:
            print(f"  FAIL: {name}" + (f"  — {detail}" if detail else ""))
        return 1 if failed else 0


def _option_card_ids(obs) -> list[int | None]:
    return [_enc._resolve_card_id(o, obs.current) for o in obs.select.option]


# --------------------------------------------------------------------------
# H1: どうぐ無効スタジアム下のどうぐ装着(Fix-F)
# --------------------------------------------------------------------------

def run_h1(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("H1 tool_stadium_guard: 93408551 row129 (T10) — ジャミングタワー下のヒーローマント")
    obs, _deck, answer = get_case(93408551, 129)
    ids = _option_card_ids(obs)
    stadium = mpa._current_stadium_card_id(obs.current)
    print(f"    stadium={stadium}({pm.card_name(stadium)}) 実戦={describe_action(obs, answer)}")

    gate.check("H1-0 前提: 場がジャミングタワー(1246)である", stadium == _JAMMING_TOWER, f"{stadium}")
    gate.check("H1-0' 前提: 実戦の選択がヒーローマント(1159)の装着である",
               len(answer) == 1 and ids[answer[0]] == _HERO_CAPE, f"cardId={ids[answer[0]]}")

    r9_out = mpa._try_tool_stadium_guard(obs, list(answer), config=R9)
    r10_out = mpa._try_tool_stadium_guard(obs, list(answer), config=R10)
    chain_r9 = mpa._apply_action_vetoes(obs, list(answer), config=R9)
    chain_r10 = mpa._apply_action_vetoes(obs, list(answer), config=R10)
    print(f"    og_r9  -> {describe_action(obs, r9_out)}  / veto連鎖 {describe_action(obs, chain_r9)}")
    print(f"    og_r10 -> {describe_action(obs, r10_out)} / veto連鎖 {describe_action(obs, chain_r10)}")

    gate.check("H1-a og_r9 は不介入(修正前の再現)", r9_out is None, describe_action(obs, r9_out))
    swapped = r10_out is not None and r10_out != answer and not mpa._is_pokemon_tool(ids[r10_out[0]])
    gate.check("H1-b og_r10 がどうぐ装着を非どうぐの手へ差し替え", bool(swapped),
               describe_action(obs, r10_out))
    gate.check("H1-c veto連鎖でも同じ結論", chain_r10 == (r10_out or answer),
               describe_action(obs, chain_r10))

    # 誤発火の対照: 同ターン row130(既に装着済み・選択肢にどうぐが無い)。
    obs2, _d2, answer2 = get_case(93408551, 130)
    out2 = mpa._try_tool_stadium_guard(obs2, list(answer2), config=R10)
    print(f"\n    row130 (誤発火の対照: 選択肢にどうぐが無い): 実戦={describe_action(obs2, answer2)}"
          f" -> {describe_action(obs2, out2)}")
    gate.check("H1-d どうぐを選んでいない decision には介入しない", out2 is None,
               describe_action(obs2, out2))


# --------------------------------------------------------------------------
# H2: 攻撃不能時のアクティブ優先エネ付け(Fix-G)
# --------------------------------------------------------------------------

_H2_ROWS = [84, 86, 88]
_H2_CONTROL_ROWS = [125, 127]


def run_h2(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("H2 energy_to_active_first: 93408551 T6 rows84/86/88 — ベンチに使ったエネをアクティブへ")
    ok_all = True
    detail: list[str] = []
    for row in _H2_ROWS:
        obs, _deck, answer = get_case(93408551, row)
        has_attack = any(o.type == OptionType.ATTACK for o in obs.select.option)
        r9_out = mpa._try_energy_to_active_first(obs, list(answer), config=R9)
        r10_out = mpa._try_energy_to_active_first(obs, list(answer), config=R10)
        chain = mpa._apply_action_vetoes(obs, list(answer), config=R10)
        print(f"\n    row{row}: ATTACK選択肢={has_attack} 実戦={describe_action(obs, answer)}")
        print(f"      og_r9 -> {describe_action(obs, r9_out)} / "
              f"og_r10 -> {describe_action(obs, r10_out)} / 連鎖 {describe_action(obs, chain)}")
        if r9_out is not None:
            ok_all = False
            detail.append(f"row{row}: og_r9 が介入 {r9_out}")
        target_active = (
            r10_out is not None
            and mpa._energy_attach_target_area(
                obs.select.option[r10_out[0]], obs.current, {_OGERPON}) == AreaType.ACTIVE
        )
        if not target_active:
            ok_all = False
            detail.append(f"row{row}: og_r10 がアクティブへ振り替えていない {r10_out}")
        if chain != (r10_out or answer):
            ok_all = False
            detail.append(f"row{row}: 連鎖の結論が違う {chain}")
    gate.check("H2-a og_r9 は3行とも不介入 / og_r10 は3行ともアクティブへ振替(連鎖も一致)",
               ok_all, "; ".join(detail))

    ctrl_ok = True
    ctrl_detail: list[str] = []
    for row in _H2_CONTROL_ROWS:
        obs, _deck, answer = get_case(93408551, row)
        has_attack = any(o.type == OptionType.ATTACK for o in obs.select.option)
        out = mpa._try_energy_to_active_first(obs, list(answer), config=R10)
        print(f"\n    row{row} (誤発火の対照: ATTACK選択肢={has_attack}): "
              f"実戦={describe_action(obs, answer)} -> {describe_action(obs, out)}")
        if not has_attack or out is not None:
            ctrl_ok = False
            ctrl_detail.append(f"row{row}: has_attack={has_attack} out={out}")
    gate.check("H2-b 既に攻撃可能(ATTACK選択肢あり)な decision には介入しない",
               ctrl_ok, "; ".join(ctrl_detail))


# --------------------------------------------------------------------------
# H3: サーチでポケモン優先(Fix-H)
# --------------------------------------------------------------------------

def run_h3(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("H3 search_pick_pokemon_first: 93473767 row37 (T3) — むしとりセットでオーガポンexを拾う")
    obs, _deck, answer = get_case(93473767, 37)
    state = obs.current
    looking = [c.id if c is not None else None for c in (state.looking or [])]
    option_ids = [mpa._looking_card_id(o, state) for o in obs.select.option]
    print(f"    state.looking={looking}")
    print(f"    option の実体={option_ids}  (option.cardId は "
          f"{[o.cardId for o in obs.select.option]})")
    print(f"    実戦={describe_action(obs, answer)}")

    # 観測可能性の結論をゲート化する: option.cardId は None だが looking から実体が引ける。
    gate.check("H3-0 option.cardId は None(=option 単体では候補が見えない)",
               all(o.cardId is None for o in obs.select.option),
               f"{[o.cardId for o in obs.select.option]}")
    gate.check("H3-0' state.looking から候補の実体が解決できる(F6が正しい)",
               all(cid is not None for cid in option_ids) and _OGERPON in option_ids,
               f"{option_ids}")

    r9_out = mpa._try_search_pick_pokemon_first(obs, list(answer), config=R9)
    r10_out = mpa._try_search_pick_pokemon_first(obs, list(answer), config=R10)
    chain = mpa._apply_action_vetoes(obs, list(answer), config=R10)
    print(f"    og_r9 -> {describe_action(obs, r9_out)} / og_r10 -> {describe_action(obs, r10_out)}"
          f" / 連鎖 {describe_action(obs, chain)}")
    gate.check("H3-a og_r9 は不介入(修正前の再現)", r9_out is None, describe_action(obs, r9_out))
    picked = r10_out is not None and any(option_ids[i] == _OGERPON for i in r10_out)
    same_len = r10_out is not None and len(r10_out) == len(answer)
    gate.check("H3-b og_r10 がオーガポンex(96)を拾う(枚数は変えない)", bool(picked and same_len),
               describe_action(obs, r10_out))
    gate.check("H3-c veto連鎖でも同じ結論", chain == (r10_out or answer), describe_action(obs, chain))

    # 誤発火の対照: 候補が全部エネルギー(拾えるポケモンが無い)。
    obs2, _d2, answer2 = get_case(93408551, 131)
    ids2 = [mpa._looking_card_id(o, obs2.current) for o in obs2.select.option]
    out2 = mpa._try_search_pick_pokemon_first(obs2, list(answer2), config=R10)
    print(f"\n    93408551 row131 (誤発火の対照: 候補の実体={ids2}): "
          f"実戦={describe_action(obs2, answer2)} -> {describe_action(obs2, out2)}")
    gate.check("H3-d 候補にたねポケモンが無ければ介入しない", out2 is None, describe_action(obs2, out2))


# --------------------------------------------------------------------------
# H4: og_r9(新キー無し)の従来挙動が保たれていること
# --------------------------------------------------------------------------

_H4_CASES = [
    (93408551, 129), (93408551, 130), (93408551, 84), (93408551, 86), (93408551, 88),
    (93408551, 125), (93408551, 127), (93408551, 131), (93473767, 37),
]


def run_h4(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("H4 parity: og_r9(新キー無し)は全局面で従来挙動")
    ok = True
    detail: list[str] = []
    for episode_id, row in _H4_CASES:
        obs, _deck, answer = get_case(episode_id, row)
        if not answer:
            continue
        for name, fn in (("tool_stadium_guard", mpa._try_tool_stadium_guard),
                         ("energy_to_active_first", mpa._try_energy_to_active_first),
                         ("search_pick_pokemon_first", mpa._try_search_pick_pokemon_first)):
            out = fn(obs, list(answer), config=R9)
            if out is not None:
                ok = False
                detail.append(f"{episode_id} row{row} {name} -> {out}")
    gate.check("H4-a og_r9 では r10 の3ガードが一切発火しない", ok, "; ".join(detail))


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="og_r10 の実局面スナップショットゲート")
    ap.add_argument("--json", default=None, help="判定結果のJSON出力先")
    args = ap.parse_args()

    print(f"configs: og_r9={_CONFIG_DIR / 'abl_5_full_og_r9.json'}")
    print(f"         og_r10={_CONFIG_DIR / 'abl_5_full_og_r10.json'}")
    gate = Gate()
    mpa.reset_guard_stats()
    run_h1(gate)
    run_h2(gate)
    run_h3(gate)
    run_h4(gate)
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
