"""ルールベース3種の「どこで負けているか」を数字で出すための計測スクリプト。

勝率だけ見ても直しようがないので、1試合ずつ中身を記録して集計する。
`league/run_league.py` は勝敗しか返さないため、ここでは対戦ループを自前で回し、
エージェント関数を包んで「何を聞かれて何を返したか」まで拾う。

取っている指標（各エージェントについて）:

| 指標 | 意味・使いどころ |
|---|---|
| `win_rate` | 勝率（先攻・後攻を半々に入れ替えて計測） |
| `first_attack_turn` | 最初にワザを使えたターン。立ち上がりの速さ |
| `ace_turn` | 主役（オーロンゲex／メガルカリオex／ブリジュラスex）が場に出たターン |
| `attacks` | 使ったワザの内訳。想定した回し方になっているか |
| `damage_dealt` | 与えた総ダメージ |
| `prizes_taken` | 取ったサイド枚数（6で勝ち） |
| `pass_with_attack` | ワザを撃てるのに撃たずにターンを終えた回数。多ければ判断のバグ |
| `no_attack_option` | ワザの選択肢が無いままターンを終えた回数。エネ切れ・詰まりの指標 |
| `end_reason` | 決着理由（1=サイド, 2=山札切れ, 3=バトル場が空, 4=カードの効果） |

使い方:

```bash
cd sample_submission
python3 ../opponents/rule_agents/diagnose.py --a grimmsnarl_rule --b lucario_rule --games 100
python3 ../opponents/rule_agents/diagnose.py --round-robin --games 100
```
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import (  # noqa: E402
    AreaType,
    LogType,
    Observation,
    OptionType,
    SelectContext,
    all_attack,
    all_card_data,
    to_observation_class,
)
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

CARDS = {c.cardId: c for c in all_card_data()}
ATTACKS = {a.attackId: a for a in all_attack()}

# name -> (モジュール, デッキCSV, 主役のカードID)
AGENTS: dict[str, tuple[str, str, int]] = {
    "grimmsnarl_rule": ("opponents.rule_agents.grimmsnarl", "decks/marnie_grimmsnarl_ex.csv", 648),
    "lucario_rule": ("opponents.rule_agents.lucario", "decks/mega_lucario_ex.csv", 678),
    "archaludon_rule": ("opponents.rule_agents.archaludon", "decks/archaludon_ex.csv", 190),
    "dragapult_rule": ("opponents.dragapult_rule_agent", "../dragapult_ex_deck.csv", 121),
}

MAX_STEPS = 3000


def load_deck(name: str) -> list[int]:
    path = (_HERE / AGENTS[name][1]).resolve()
    text = path.read_text(encoding="utf-8")
    return [int(v) for v in text.replace(",", "\n").split() if v.strip()]


def load_agent(name: str):
    return importlib.import_module(AGENTS[name][0]).agent


class Probe:
    """エージェント関数を包んで、MAIN での「撃てたのに撃たなかった」を数える。"""

    def __init__(self, fn):
        self.fn = fn
        self.pass_with_attack = 0
        self.no_attack_option = 0
        self.turn_ends = 0

    def __call__(self, obs: Observation) -> list[int]:
        action = self.fn(obs)
        select = obs.select
        if select is not None and select.context == SelectContext.MAIN and action:
            chosen = select.option[action[0]] if action[0] < len(select.option) else None
            if chosen is not None and chosen.type == OptionType.END:
                self.turn_ends += 1
                if any(o.type == OptionType.ATTACK for o in select.option):
                    self.pass_with_attack += 1
                else:
                    self.no_attack_option += 1
        return action


def play_one(name0: str, name1: str, seed: int) -> dict:
    """1試合を回して、両者の中身つきの記録を返す。"""
    random.seed(seed)
    agents = [Probe(load_agent(name0)), Probe(load_agent(name1))]
    names = [name0, name1]
    aces = [AGENTS[name0][2], AGENTS[name1][2]]

    deck0, deck1 = load_deck(name0), load_deck(name1)
    obs_dict, start = battle_start(deck0, deck1)
    if start.errorType != 0:
        return {"error": f"battle_start errorType={start.errorType}"}

    logs: list = []
    steps = 0
    turn = 0
    winner: int | None = None
    reason: int | None = None
    final = None
    try:
        while True:
            obs = to_observation_class(obs_dict)
            logs.extend(obs.logs)
            if obs.current is None:
                return {"error": "current is None"}
            if obs.current.result != -1:
                winner = obs.current.result
                turn = obs.current.turn
                final = obs.current
                break
            if steps >= MAX_STEPS:
                return {"error": "max_steps"}
            action = agents[obs.current.yourIndex](obs)
            obs_dict = battle_select(action)
            steps += 1
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    finally:
        battle_finish()

    # --- ログを読んで指標を組み立てる ---
    cur_turn = 0
    first_attack = [None, None]
    ace_turn = [None, None]
    attacks: list[Counter] = [Counter(), Counter()]
    damage_taken = [0, 0]
    for log in logs:
        if log.type == LogType.TURN_START:
            cur_turn += 1
        elif log.type == LogType.ATTACK and log.playerIndex is not None:
            p = log.playerIndex
            attacks[p][log.attackId] += 1
            if first_attack[p] is None:
                first_attack[p] = cur_turn
        elif log.type == LogType.HP_CHANGE and log.playerIndex is not None and log.value:
            if log.value < 0:
                damage_taken[log.playerIndex] += -log.value
        elif log.type == LogType.EVOLVE and log.playerIndex is not None:
            # 「主役が場に出たターン」。手札に来ただけの MOVE_CARD は数えない
            # （立ち上がりの速さを測りたいので、場に立った瞬間だけを取る）。
            p = log.playerIndex
            if log.cardId == aces[p] and ace_turn[p] is None:
                ace_turn[p] = cur_turn
        elif log.type == LogType.MOVE_CARD and log.playerIndex is not None:
            p = log.playerIndex
            if (log.cardId == aces[p] and ace_turn[p] is None
                    and log.toArea in (AreaType.ACTIVE, AreaType.BENCH)):
                ace_turn[p] = cur_turn
        elif log.type == LogType.RESULT:
            reason = log.reason

    out = {
        "seed": seed,
        "winner": winner,
        "turns": turn,
        "steps": steps,
        "reason": reason,
        "players": [],
    }
    for p in (0, 1):
        out["players"].append({
            "name": names[p],
            "won": winner == p,
            "lost_by_reason": reason if winner is not None and winner != p else None,
            "prizes_taken": (6 - len(final.players[p].prize)) if final is not None else None,
            "first_attack_turn": first_attack[p],
            "ace_turn": ace_turn[p],
            "attacks": {str(k): v for k, v in attacks[p].items()},
            "damage_dealt": damage_taken[1 - p],
            "pass_with_attack": agents[p].pass_with_attack,
            "no_attack_option": agents[p].no_attack_option,
            "turn_ends": agents[p].turn_ends,
        })
    return out


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 2) if vals else None


def summarize(records: list[dict], name_a: str, name_b: str) -> dict:
    per: dict[str, dict] = {
        n: {"games": 0, "wins": 0, "first_attack_turn": [], "ace_turn": [],
            "attacks": Counter(), "damage_dealt": [], "pass_with_attack": 0,
            "no_attack_option": 0, "turn_ends": 0, "prizes_taken": [],
            "loss_reason": Counter()}
        for n in {name_a, name_b}
    }
    reasons: Counter = Counter()
    turns: list[int] = []
    errors = 0
    for r in records:
        if r.get("error"):
            errors += 1
            continue
        turns.append(r["turns"])
        reasons[r["reason"]] += 1
        for pl in r["players"]:
            slot = per[pl["name"]]
            slot["games"] += 1
            slot["wins"] += 1 if pl["won"] else 0
            slot["first_attack_turn"].append(pl["first_attack_turn"])
            slot["ace_turn"].append(pl["ace_turn"])
            slot["damage_dealt"].append(pl["damage_dealt"])
            slot["pass_with_attack"] += pl["pass_with_attack"]
            slot["no_attack_option"] += pl["no_attack_option"]
            slot["turn_ends"] += pl["turn_ends"]
            slot["prizes_taken"].append(pl["prizes_taken"])
            if pl["lost_by_reason"] is not None:
                slot["loss_reason"][pl["lost_by_reason"]] += 1
            for k, v in pl["attacks"].items():
                slot["attacks"][int(k)] += v

    summary = {"errors": errors, "avg_turns": _mean(turns),
               "end_reason": {str(k): v for k, v in reasons.items()}, "agents": {}}
    for name, slot in per.items():
        n = slot["games"]
        summary["agents"][name] = {
            "games": n,
            "win_rate": round(slot["wins"] / n, 3) if n else None,
            "first_attack_turn": _mean(slot["first_attack_turn"]),
            "ace_turn": _mean(slot["ace_turn"]),
            "ace_never_rate": round(
                sum(1 for v in slot["ace_turn"] if v is None) / n, 3) if n else None,
            "damage_dealt": _mean(slot["damage_dealt"]),
            "prizes_taken": _mean(slot["prizes_taken"]),
            "loss_reason": {str(k): v for k, v in slot["loss_reason"].items()},
            "pass_with_attack_per_game": round(slot["pass_with_attack"] / n, 2) if n else None,
            "no_attack_option_per_game": round(slot["no_attack_option"] / n, 2) if n else None,
            "attacks": {
                (ATTACKS[k].name if k in ATTACKS else str(k)): v
                for k, v in slot["attacks"].most_common()
            },
        }
    return summary


def run_pair(name_a: str, name_b: str, games: int, seed_start: int) -> dict:
    records = []
    for i in range(games):
        # 半分ずつ先攻/後攻を入れ替える（deck0 側が先攻）。
        if i % 2 == 0:
            records.append(play_one(name_a, name_b, seed_start + i))
        else:
            records.append(play_one(name_b, name_a, seed_start + i))
    return summarize(records, name_a, name_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="grimmsnarl_rule", choices=sorted(AGENTS))
    ap.add_argument("--b", default="lucario_rule", choices=sorted(AGENTS))
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--round-robin", action="store_true",
                    help="ルールベース3種の総当たり + ミラーを回す")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.round_robin:
        names = ["grimmsnarl_rule", "lucario_rule", "archaludon_rule"]
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i:]]
    else:
        pairs = [(args.a, args.b)]

    result = {}
    for a, b in pairs:
        key = f"{a}__vs__{b}"
        print(f"=== {key} ({args.games}試合) ===", flush=True)
        summary = run_pair(a, b, args.games, args.seed_start)
        result[key] = summary
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    print(f"\n所要 {time.time() - t0:.1f}s", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
