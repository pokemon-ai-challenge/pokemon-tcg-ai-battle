"""Kaggle の実戦リプレイを1手ずつ読み下し、こちらの判断を振り返るための道具。

`diagnose.py` は自己対戦を大量に回して統計を出すが、**実際のリーダーボードで
どんな相手にどう負けたか**は自己対戦からは分からない。このスクリプトはリプレイJSON
（`steps[i][player]["observation"]` に、そのときエージェントが受け取った obs が
そのまま入っている）を読み、

- 各選択で「何を聞かれ」「何を選んだか」を日本語で表示する
- 同じ obs を**現在の**エージェントに食わせ直し、当時と違う手を選ぶかを比べる

の2つを行う。2つ目は「その後の改良が、実際に負けた局面を変えるか」を確かめるためのもの。

使い方:

```bash
cd sample_submission
python3 ../opponents/rule_agents/replay_review.py <replay.json> --player 0
python3 ../opponents/rule_agents/replay_review.py <replay.json> --player 0 --diff-only
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import (  # noqa: E402
    AreaType,
    LogType,
    OptionType,
    SelectContext,
    all_attack,
    all_card_data,
    to_observation_class,
)

CARDS = {c.cardId: c for c in all_card_data()}
ATTACKS = {a.attackId: a for a in all_attack()}


def nm(cid: int | None) -> str:
    if not cid:
        return "?"
    c = CARDS.get(cid)
    return c.name if c else f"#{cid}"


def describe_option(obs, o) -> str:
    """選択肢1つを人間が読める文字列にする。"""
    s = obs.current

    def at(area, idx, pi):
        try:
            ps = s.players[pi if pi is not None else s.yourIndex]
            if area == AreaType.HAND:
                return ps.hand[idx]
            if area == AreaType.ACTIVE:
                return ps.active[idx]
            if area == AreaType.BENCH:
                return ps.bench[idx]
            if area == AreaType.DISCARD:
                return ps.discard[idx]
            if area == AreaType.DECK:
                return obs.select.deck[idx]
            if area == AreaType.STADIUM:
                return s.stadium[idx]
            if area == AreaType.LOOKING:
                return s.looking[idx]
            if area == AreaType.PRIZE:
                return ps.prize[idx]
        except Exception:
            return None
        return None

    t = o.type
    if t == OptionType.PLAY:
        c = at(AreaType.HAND, o.index, s.yourIndex)
        return f"出す:{nm(c.id) if c else '?'}"
    if t == OptionType.ATTACH:
        c = at(o.area, o.index, s.yourIndex)
        p = at(o.inPlayArea, o.inPlayIndex, s.yourIndex)
        return f"つける:{nm(c.id) if c else '?'}→{nm(p.id) if p else '?'}"
    if t == OptionType.EVOLVE:
        c = at(o.area, o.index, s.yourIndex)
        p = at(o.inPlayArea, o.inPlayIndex, s.yourIndex)
        return f"進化:{nm(c.id) if c else '?'}←{nm(p.id) if p else '?'}"
    if t == OptionType.ABILITY:
        c = at(o.area, o.index, o.playerIndex)
        return f"特性:{nm(c.id) if c else '?'}"
    if t == OptionType.ATTACK:
        a = ATTACKS.get(o.attackId)
        return f"ワザ:{a.name if a else o.attackId}({a.damage if a else '?'})"
    if t == OptionType.RETREAT:
        return "にげる"
    if t == OptionType.END:
        return "ターン終了"
    if t == OptionType.CARD:
        c = at(o.area, o.index, o.playerIndex)
        where = AreaType(o.area).name if o.area is not None else "?"
        who = "自" if o.playerIndex == s.yourIndex else "相"
        return f"札:{nm(c.id) if c else '?'}@{where}/{who}"
    if t in (OptionType.YES, OptionType.NO):
        return OptionType(t).name
    if t == OptionType.NUMBER:
        return f"数:{o.number}"
    if t in (OptionType.ENERGY, OptionType.ENERGY_CARD):
        c = at(o.area, o.index, o.playerIndex)
        who = "自" if o.playerIndex == s.yourIndex else "相"
        return f"エネ:{nm(c.id) if c else '?'}@{who}"
    return OptionType(t).name


def board_line(obs, me: int) -> str:
    s = obs.current
    mine = s.players[me]
    opp = s.players[1 - me]
    ap = mine.active[0] if mine.active and mine.active[0] else None
    op = opp.active[0] if opp.active and opp.active[0] else None

    def pk(p):
        if p is None:
            return "(伏)"
        tools = "+道具" if p.tools else ""
        return f"{nm(p.id)}/E{len(p.energies)}/hp{p.hp}{tools}"

    bench = [f"{nm(x.id)}/E{len(x.energies)}/hp{x.hp}" for x in mine.bench]
    obench = [f"{nm(x.id)}/hp{x.hp}" for x in opp.bench]
    stad = nm(s.stadium[0].id) if s.stadium else "-"
    return (f"T{s.turn} 自[{pk(ap)}] ベンチ{bench} 手{mine.handCount} サイド{len(mine.prize)}"
            f" | 相[{pk(op)}] ベンチ{obench} サイド{len(opp.prize)} | 場:{stad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", type=Path)
    ap.add_argument("--player", type=int, required=True, help="振り返る側の player_index")
    ap.add_argument("--agent", default="grimmsnarl",
                    choices=["grimmsnarl", "lucario", "archaludon"])
    ap.add_argument("--diff-only", action="store_true",
                    help="現在のエージェントと当時の選択が食い違った手だけ表示")
    ap.add_argument("--main-only", action="store_true", help="MAIN の選択だけ表示")
    args = ap.parse_args()

    import importlib
    agent_fn = importlib.import_module(f"opponents.rule_agents.{args.agent}").agent

    data = json.loads(args.replay.read_text(encoding="utf-8"))
    me = args.player
    names = data["info"].get("TeamNames", ["p0", "p1"])
    rewards = data.get("rewards")
    print(f"# {args.replay.name}  自分=p{me}({names[me]}) 相手={names[1-me]}  結果={rewards}")

    diffs = 0
    shown = 0
    steps = data["steps"]
    for i, step in enumerate(steps):
        if len(step) <= me:
            continue
        entry = step[me]
        # そのターン実際に手番だったかは status で判別する。INACTIVE の行にも
        # observation は入っているが、それは1つ前の状態の写しで action は常に空。
        if entry.get("status") != "ACTIVE":
            continue
        obs_dict = entry.get("observation")
        if not isinstance(obs_dict, dict) or obs_dict.get("select") is None:
            continue
        # **リプレイの action は1ステップ後ろにずれて記録されている。**
        # step0 の observation はデッキ選択(select=None)なのに、60枚のデッキが
        # action として現れるのは step1 の行。つまり observation[i] に対する
        # answer は steps[i+1] 側に入っている。
        if i + 1 >= len(steps) or len(steps[i + 1]) <= me:
            continue
        action = steps[i + 1][me].get("action")
        if not isinstance(action, list):
            continue
        obs = to_observation_class(obs_dict)
        if obs.current is None:
            continue
        ctx_name = SelectContext(obs.select.context).name
        if args.main_only and obs.select.context != SelectContext.MAIN:
            continue

        chosen = [describe_option(obs, obs.select.option[k])
                  for k in action if k < len(obs.select.option)]
        try:
            now = agent_fn(obs)
            now_desc = [describe_option(obs, obs.select.option[k])
                        for k in now if k < len(obs.select.option)]
        except Exception as e:  # noqa: BLE001
            now, now_desc = None, [f"EXC {e!r}"]

        differs = (now is not None and list(now) != list(action))
        if differs:
            diffs += 1
        if args.diff_only and not differs:
            continue

        shown += 1
        print(f"\n[step {i}] {board_line(obs, me)}")
        options = [describe_option(obs, o) for o in obs.select.option]
        print(f"  {ctx_name} 選択肢({obs.select.minCount}-{obs.select.maxCount}): {options}")
        print(f"  当時の選択: {chosen}")
        if differs:
            print(f"  → 現在なら: {now_desc}   ★食い違い")

    print(f"\n表示 {shown} 手 / 現在のエージェントと食い違った手 {diffs}")


if __name__ == "__main__":
    main()
