"""R1土台: 対Grimmsnarlで ml_policy エージェントが遭遇する『相手ポケモンを対象に選ぶ』選択の
実体(SelectContext / option の area・index・playerIndex・cardId)を観測する。

Boss's Orders(1182)の対象選択がどの context で、options が相手ベンチ(進化前 Impidimp646/
Morgrem647)をどう指すか(cardId 直付け or area/index 解決)を確定し、R1 の識別ロジックを固める。

使い方: python kaggle_replays/_diag_target_selects_vs_grimmsnarl.py --games 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_league  # noqa: E402
from cg.api import (AreaType, OptionType, SelectContext, SelectType,  # noqa: E402
                    to_observation_class)
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
PROD_DECK = _ROOT / "sample_submission" / "deck.csv"
MAX_STEPS = 3000
MARNIE_PREEVO = {646: "Impidimp", 647: "Morgrem", 648: "GrimmsnarlEx"}


def _name(enum_cls, val):
    try:
        return enum_cls(val).name
    except Exception:
        return str(val)


def _resolve_pokemon_id(opt, state):
    """option が area/index で場のポケモンを指すなら、その Pokemon.id を返す。"""
    area = opt.area
    idx = opt.index
    if area not in (AreaType.ACTIVE, AreaType.BENCH) or idx is None:
        return None
    pi = opt.playerIndex if opt.playerIndex is not None else state.yourIndex
    try:
        player = state.players[pi]
        zone = player.active if area == AreaType.ACTIVE else player.bench
        pk = zone[idx]
        return getattr(pk, "id", None) if pk is not None else None
    except (IndexError, AttributeError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--out", default=str(_HERE / "_diag_target_selects_vs_grimmsnarl_results.json"))
    args = ap.parse_args()

    os.chdir(_ROOT / "sample_submission")
    my = run_league.build_agent("ml_policy", None, "abl_5_full")
    opp = run_league.build_agent("ml_policy", str(WDIR / "policy_weights_marnie_grimmsnarl_ex.json"), "abl_5_full")
    deck0 = run_league.read_deck_csv_file(str(PROD_DECK))
    deck1 = run_league.read_deck_csv_file(str(DECKDIR / "marnie_grimmsnarl_ex" / "01.csv"))

    examples = {}  # (type,context) -> example dict
    boss_targets = []  # Boss(1182)直後の対象選択の記録

    for g in range(args.games):
        import random
        random.seed(g)
        obs_dict, sd = battle_start(deck0, deck1)
        if sd.errorType != 0:
            continue
        steps = 0
        just_played_boss = False
        try:
            while steps < MAX_STEPS:
                obs = to_observation_class(obs_dict)
                cur = obs.current
                if cur is None or cur.result != -1:
                    break
                sel = obs.select
                if cur.yourIndex == 0 and sel is not None and sel.context != SelectContext.MAIN:
                    key = (int(sel.type), int(sel.context))
                    opt_rows = []
                    for i, o in enumerate(sel.option):
                        opt_rows.append({
                            "i": i, "type": _name(OptionType, o.type),
                            "cardId": o.cardId,
                            "area": _name(AreaType, o.area) if o.area is not None else None,
                            "index": o.index, "playerIndex": o.playerIndex,
                            "resolved_pokemon_id": _resolve_pokemon_id(o, cur),
                        })
                    rec = {"selectType": _name(SelectType, sel.type),
                           "context": _name(SelectContext, sel.context),
                           "yourIndex": cur.yourIndex,
                           "options": opt_rows[:8]}
                    if key not in examples:
                        examples[key] = rec
                    if just_played_boss and len(boss_targets) < 6:
                        boss_targets.append(rec)
                if cur.yourIndex == 0:
                    action = my(obs)
                    # 直前に自分がBoss(1182)をplayしたか
                    just_played_boss = (
                        sel is not None and sel.context == SelectContext.MAIN
                        and any(0 <= i < len(sel.option) and sel.option[i].cardId == 1182
                                and sel.option[i].type == OptionType.PLAY for i in action)
                    )
                    obs_dict = battle_select(action)
                else:
                    obs_dict = battle_select(opp(obs))
                steps += 1
        except Exception as exc:  # noqa: BLE001
            print("game", g, "exception:", repr(exc))
        finally:
            battle_finish()

    out = {"distinct_target_contexts": list(examples.values()), "boss_target_examples": boss_targets}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"distinct non-MAIN target contexts seen: {len(examples)}")
    for (t, c), rec in examples.items():
        opp_targets = [(o["resolved_pokemon_id"], o["cardId"]) for o in rec["options"]
                       if o["playerIndex"] == 1 or (o["resolved_pokemon_id"] in MARNIE_PREEVO)]
        print(f"  ctx={rec['context']:<18} type={rec['selectType']:<8} nopt={len(rec['options'])} opp_targets={opp_targets[:5]}")
    print(f"boss_target_examples captured: {len(boss_targets)}")
    for r in boss_targets[:3]:
        print("  BOSS-TARGET ctx=", r["context"], "opts=", [(o["cardId"], o["resolved_pokemon_id"], o["playerIndex"]) for o in r["options"]])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
