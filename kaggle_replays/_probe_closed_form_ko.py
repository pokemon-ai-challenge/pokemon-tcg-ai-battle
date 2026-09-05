#!/usr/bin/env python3
"""閉形式打点(まんようしぐれ)の**実リプレイ突合**プローブ。

`ptcg_ai/search/closed_form_ko.py` が返す推定打点を、実ラダーのリプレイに残っている
**実ダメージ値**(`LogType.ATTACK{attackId}` の直後の `LogType.HP_CHANGE{value<0}`)と
1件ずつ突き合わせる。式が崩れる条件(弱点・軽減・効果)を実測で特定するのが目的。

対象のワザ: オーガポン みどりのめん ex(cardId 96)の「まんようしぐれ」(attackId 120)
    30 + 30 × (自分のバトルポケモンのエネ数 + 相手のバトルポケモンのエネ数)

使い方::

    python kaggle_replays/_probe_closed_form_ko.py --limit 40
    python kaggle_replays/_probe_closed_form_ko.py --episodes 93517227 93510303 --verbose
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
import _probe_missed_lethal as pm  # noqa: E402

from cg.api import LogType, OptionType, to_observation_class  # noqa: E402
from ptcg_ai.search import closed_form_ko as cfk  # noqa: E402

_MYRIAD_LEAF_SHOWER = 120


def _attack_damage_after(
    steps: list, own: int, start_row: int, min_row: int,
) -> tuple[int | None, int | None]:
    """``start_row`` の回答(=ATTACK)の後、最初に現れる自分の attackId=120 の実ダメージ。

    Returns:
        ``(damage, row)``。``damage`` は相手側ポケモンへの HP_CHANGE の合計の絶対値。
        **軽減/無効化で 0 になった場合も 0 として返す**(``value == 0`` の HP_CHANGE が
        実際に記録される。実例: Crustle(345)の「Mysterious Rock Inn」= ex のワザのダメージを
        完全に無効化する特性。episode 93542157 row102 で実測)。見つからなければ ``(None, None)``。

    ``min_row``: ここより前の行の ATTACK ログは消費済みとして無視する。ログは後続の
    INACTIVE 行にもそのまま再掲されるため、カーソルを進めないと同じ攻撃を二重計上する。
    """
    for k in range(max(start_row, min_row), min(start_row + 10, len(steps))):
        cell = steps[k][own]
        if cell.get("status") != "ACTIVE":
            continue
        logs = (cell.get("observation") or {}).get("logs") or []
        for i, log in enumerate(logs):
            if int(log.get("type", -1)) != int(LogType.ATTACK):
                continue
            if int(log.get("playerIndex", -1)) != own:
                continue
            if int(log.get("attackId", -1)) != _MYRIAD_LEAF_SHOWER:
                continue
            total = 0
            for nxt in logs[i + 1:]:
                ntype = int(nxt.get("type", -1))
                if ntype in (int(LogType.ATTACK), int(LogType.TURN_END), int(LogType.TURN_START)):
                    break
                if ntype != int(LogType.HP_CHANGE):
                    continue
                if int(nxt.get("playerIndex", -1)) != 1 - own:
                    continue
                if nxt.get("putDamageCounter"):
                    continue  # ダメカン配置は「ワザのダメージ」ではない
                value = int(nxt.get("value", 0) or 0)
                if value < 0:
                    total += -value
            return total, k
    return None, None


def collect(episode_id: int) -> list[dict]:
    """1リプレイから「まんようしぐれを撃った decision」と実ダメージの組を取り出す。"""
    replay = pm.load_replay(episode_id)
    own = pm.own_index_of(replay)
    steps = replay["steps"]
    out: list[dict] = []
    cursor = 0
    for dec in pm.own_decisions(replay, own):
        row, answer = dec["row"], dec["answer"] or []
        if len(answer) != 1:
            continue
        obs = to_observation_class(dec["obs_dict"])
        if obs.select is None or obs.current is None:
            continue
        idx = answer[0]
        if not (0 <= idx < len(obs.select.option)):
            continue
        option = obs.select.option[idx]
        if option.type != OptionType.ATTACK:
            continue
        if int(getattr(option, "attackId", -1) or -1) != _MYRIAD_LEAF_SHOWER:
            continue
        actual, log_row = _attack_damage_after(steps, own, row + 1, cursor)
        if actual is None:
            continue
        cursor = log_row + 1
        state = obs.current
        me = state.yourIndex
        est = cfk.estimate_attack_damage(state, me, _MYRIAD_LEAF_SHOWER)
        my_e = cfk._active_energy_count(state, me)
        op_e = cfk._active_energy_count(state, 1 - me)
        opp = cfk._active_pokemon(state, 1 - me)
        opp_card = cfk._card(None if opp is None else opp.id)
        my_type = cfk._my_attack_energy_type(state, me)
        weak = (opp_card is not None and opp_card.weakness is not None
                and my_type is not None and int(opp_card.weakness) == my_type)
        # 弱点(×2)を適用した「上界」。閉形式は True 判定でこれを使わない(=過小評価で安全)。
        est_weak = None if est is None else (est * 2 if weak else est)
        risk = cfk._damage_modifier_risk(state, me)
        out.append({
            "episode": episode_id, "row": row, "log_row": log_row,
            "turn": int(state.turn), "my_energy": my_e, "opp_energy": op_e,
            "opp_id": None if opp is None else int(opp.id),
            "opp_name": None if opp_card is None else opp_card.name,
            "opp_hp": None if opp is None else int(opp.hp),
            "opp_max_hp": None if opp is None else int(opp.maxHp),
            "weakness": bool(weak),
            "modifier_risk_card_id": risk,
            "estimate": est, "estimate_with_weakness": est_weak, "actual": actual,
            "match": est is not None and int(est) == int(actual),
            "match_with_weakness": est_weak is not None and int(est_weak) == int(actual),
            # 閉形式が「黙る」(判定不能を返す)局面かどうか。式が崩れる行はここが True で
            # あるべき = 実戦では絶対に誤った True/False を返さない、という主張の検証。
            "closed_form_silent": risk is not None,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="まんようしぐれ閉形式打点の実リプレイ突合")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=40, help="走査するリプレイ数の上限")
    ap.add_argument("--min-episode", type=int, default=93000000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.episodes:
        ids = list(args.episodes)
    else:
        ids = sorted(
            int(p.name.split("-")[1]) for p in (_HERE / "replays").glob("episode-*-replay.json")
        )
        ids = [e for e in ids if e >= args.min_episode][::-1][: args.limit * 4]

    rows: list[dict] = []
    scanned = 0
    for episode_id in ids:
        try:
            found = collect(episode_id)
        except Exception:  # noqa: BLE001 - 自チームが出ていない/デッキ回答が無いリプレイは飛ばす
            continue
        scanned += 1
        rows.extend(found)
        if len(rows) >= args.limit:
            break

    print(f"scanned replays: {scanned} / rows: {len(rows)}")
    print(f"{'episode':>10} {'row':>5} {'T':>3} {'myE':>4} {'opE':>4} {'oppHP':>6} "
          f"{'est':>5} {'est*W':>6} {'actual':>7} {'weak':>5} {'silent':>7}  verdict")
    ok = weak_ok = silent = unexplained = 0
    for r in rows:
        if r["match"]:
            verdict, ok = "OK", ok + 1
        elif r["match_with_weakness"]:
            verdict, weak_ok = "OK(weak x2)", weak_ok + 1
        elif r["closed_form_silent"]:
            verdict, silent = "SILENT(判定不能)", silent + 1
        else:
            verdict, unexplained = "UNEXPLAINED", unexplained + 1
        if r["closed_form_silent"] and verdict.startswith("OK"):
            silent += 1
        print(f"{r['episode']:>10} {r['row']:>5} {r['turn']:>3} {r['my_energy']:>4} "
              f"{r['opp_energy']:>4} {str(r['opp_hp']):>6} {str(r['estimate']):>5} "
              f"{str(r['estimate_with_weakness']):>6} {r['actual']:>7} "
              f"{str(r['weakness']):>5} {str(r['closed_form_silent']):>7}  {verdict}")
    print(f"\n素の式が一致: {ok}/{len(rows)}")
    print(f"弱点×2 で一致: {weak_ok}  (合計 {ok + weak_ok}/{len(rows)})")
    print(f"閉形式が判定不能を返す局面(=誤答しない): {silent}")
    print(f"未説明(要調査): {unexplained}")
    if args.json:
        # `_probe_missed_lethal` が import 時に cwd を sample_submission へ移すので、
        # 相対パスは kaggle_replays/ 基準に解決する(出力が提出ツリーに散らからないように)。
        out = Path(args.json)
        if not out.is_absolute():
            out = _HERE / out
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
