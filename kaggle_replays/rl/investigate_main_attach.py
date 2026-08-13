"""MAIN/OptionType.ATTACH の raw option を正しく解決できるか調査する。

これまで SelectType.CARD の ATTACH_FROM/ATTACH_TO を「手貼り」だと思っていたが、
通常の手貼りは SelectType.MAIN 内の OptionType.ATTACH で、1つの option に
「付けるカード(area/index)」と「付与先ポケモン(inPlayArea/inPlayIndex)」の両方が
入っている可能性が高い(ユーザー指摘、公式API: https://matsuoinstitute.github.io/cabt/api.html)。

このスクリプトは:
  1. MAIN の raw option をそのまま(候補削減前)記録する
  2. OptionType.ATTACH を area/inPlayArea から解決する(AreaType を数値直書きしない)
  3. 選択直後の Observation.logs から LogType.ATTACH を拾い、
     解決した source/target が実際の付与と一致するか検証する
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from matchup_common import atomic_write_json, read_deck, discover_archetype_decks  # noqa: E402

OGERPON_EX = 96
TAPU_BULU = 920
GRASS_ENERGY_CARD_ID = 1   # 基本【草】エネルギー(data/JP_Card_Data.csv で確認済み)。


def resolve_source_card(state, me, area, index):
    """付けるカード(手札など)を area/index から解決する。"""
    from cg.api import AreaType

    player = state.players[me]
    if int(area) == int(AreaType.HAND):
        cards = player.hand or []
    elif int(area) == int(AreaType.DISCARD):
        cards = player.discard or []
    else:
        return None
    i = int(index)
    return cards[i] if 0 <= i < len(cards) else None


def resolve_target_pokemon(state, me, in_play_area, in_play_index):
    """付与先ポケモンを inPlayArea/inPlayIndex から解決する(AreaType を明示使用)。"""
    from cg.api import AreaType

    if in_play_area is None or in_play_index is None:
        return None
    player = state.players[me]
    a = int(in_play_area)
    i = int(in_play_index)
    if a == int(AreaType.ACTIVE):
        slots = player.active or []
    elif a == int(AreaType.BENCH):
        slots = player.bench or []
    else:
        return None
    return slots[i] if 0 <= i < len(slots) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--opponent-archetype", default="marnie_grimmsnarl_ex")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import (AreaType, LogType, OptionType, SelectContext, SelectType,
                        to_observation_class)
    from ptcg_ai.ml_policy import ogerpon_planner as P
    from ptcg_ai.shared import card_cache

    my_deck = read_deck(args.deck)
    opp_decks = discover_archetype_decks(args.opponent_archetype)
    opp_deck = next(d["deck"] for d in opp_decks if not d["errors"])

    raw_attach_examples = []
    verify_pairs = []      # (expected, actual) for source/target consistency check
    bulu_opportunity = []
    dist_counts = {"MAIN": 0, "MAIN_ATTACH_options_total": 0}
    game_log = []

    for gi in range(args.games):
        random.seed(args.seed + gi)
        controlled = gi % 2   # 自分をplayer 0/1で交互に(先後の偏りを避ける)
        d0, d1 = (my_deck, opp_deck) if controlled == 0 else (opp_deck, my_deck)
        obs_dict, sd = battle_start(d0, d1)
        turn_action_count = 0
        pending = None  # 直前に選んだ ATTACH option の期待値
        try:
            for step in range(500):
                obs = to_observation_class(obs_dict)
                cur, sel = obs.current, obs.select
                if cur is None or cur.result != -1:
                    break
                # --- 直前ターンで ATTACH を選んでいたら、このobsのlogsで検証する ---
                if pending is not None:
                    for log in (obs.logs or []):
                        if int(log.type) == int(LogType.ATTACH):
                            actual = {"actual_card_id": log.cardId, "actual_serial": log.serial,
                                     "actual_card_id_target": log.cardIdTarget,
                                     "actual_serial_target": log.serialTarget}
                            verify_pairs.append({"expected": pending, "actual": actual,
                                                 "match_source": (pending["expected_source_serial"]
                                                                  == actual["actual_serial"]),
                                                 "match_target": (pending["expected_target_serial"]
                                                                  == actual["actual_serial_target"])})
                            break
                    pending = None

                if sel is None or not sel.option:
                    obs_dict = battle_select([])
                    continue
                turn_action_count += 1
                is_mine = (cur.yourIndex == controlled)

                if is_mine and int(sel.type) == int(SelectType.MAIN):
                    dist_counts["MAIN"] += 1
                    mine = cur.players[controlled]
                    active = next((s for s in (mine.active or []) if s is not None), None)
                    bench = [s for s in (mine.bench or []) if s is not None]
                    bulu = next((b for b in bench if b.id == TAPU_BULU), None)
                    hand_energy = [c for c in (mine.hand or []) if c.id == GRASS_ENERGY_CARD_ID]

                    attach_options = []
                    for oi, o in enumerate(sel.option):
                        if int(o.type) != int(OptionType.ATTACH):
                            continue
                        dist_counts["MAIN_ATTACH_options_total"] += 1
                        src = resolve_source_card(cur, controlled, o.area, o.index)
                        tgt = resolve_target_pokemon(cur, controlled, o.inPlayArea, o.inPlayIndex)
                        rec = {
                            "option_index": oi, "area": int(o.area) if o.area is not None else None,
                            "index": o.index,
                            "inPlayArea": (int(o.inPlayArea) if o.inPlayArea is not None else None),
                            "inPlayIndex": o.inPlayIndex,
                            "source_card_id": (src.id if src else None),
                            "source_serial": (getattr(src, "serial", None) if src else None),
                            "target_card_id": (tgt.id if tgt else None),
                            "target_serial": (getattr(tgt, "serial", None) if tgt else None),
                            "target_is_ex": (P.is_ex(tgt) if tgt else None),
                            "target_hp": (tgt.hp if tgt else None),
                            "target_energy_count": (len(P.energies_of(tgt)) if tgt else None),
                        }
                        attach_options.append(rec)
                        if len(raw_attach_examples) < 8:
                            raw_attach_examples.append({
                                "game": gi, "turn": cur.turn, **rec,
                                "n_options_total": len(sel.option)})

                    # 中心命題: ブルルがベンチにいて、手札に草エネがあり、まだ今ターン
                    # 手貼りしていない局面で、rawにブルル向けATTACHがあるか。
                    if bulu is not None and hand_energy and not cur.energyAttached:
                        raw_has_bulu = any(a["target_serial"] == bulu.serial for a in attach_options)
                        bulu_opportunity.append({
                            "game": gi, "turn": cur.turn,
                            "bulu_serial": bulu.serial, "bulu_energy_before": len(P.energies_of(bulu)),
                            "raw_available": raw_has_bulu,
                            "n_attach_options": len(attach_options),
                            "attach_targets": [a["target_serial"] for a in attach_options],
                        })
                        if raw_has_bulu and len(game_log) < 5:
                            game_log.append({
                                "game": gi, "turn": cur.turn,
                                "active": {"id": active.id if active else None,
                                          "serial": getattr(active, "serial", None) if active else None},
                                "bench": [{"id": b.id, "serial": b.serial,
                                          "energy": len(P.energies_of(b))} for b in bench],
                                "hand_energy_ids": [c.id for c in hand_energy],
                                "attach_options": attach_options,
                            })

                    # 貪欲: 見えたブルル向けATTACHがあれば選ぶ(挙動確認用。実装は別途)
                    chosen = next((a["option_index"] for a in attach_options
                                  if a["target_serial"] == (bulu.serial if bulu else None)), None)
                    if chosen is None:
                        chosen = 0
                    o = sel.option[chosen]
                    if int(o.type) == int(OptionType.ATTACH):
                        src = resolve_source_card(cur, controlled, o.area, o.index)
                        tgt = resolve_target_pokemon(cur, controlled, o.inPlayArea, o.inPlayIndex)
                        pending = {"expected_source_card_id": (src.id if src else None),
                                  "expected_source_serial": (getattr(src, "serial", None) if src else None),
                                  "expected_target_card_id": (tgt.id if tgt else None),
                                  "expected_target_serial": (getattr(tgt, "serial", None) if tgt else None)}
                    action = [chosen]
                elif is_mine:
                    action = [0]
                else:
                    action = [0] if sel.minCount == 0 else list(range(sel.minCount))
                    n = len(sel.option)
                    action = list(range(max(sel.minCount, min(sel.maxCount, n))))
                obs_dict = battle_select(action)
        except Exception as exc:  # noqa: BLE001
            pass
        finally:
            battle_finish()
        if len(bulu_opportunity) >= 20 and gi >= args.games - 1:
            break

    out = {
        "games_run": args.games,
        "dist_counts": dist_counts,
        "raw_attach_examples": raw_attach_examples,
        "bulu_opportunity_count": len(bulu_opportunity),
        "bulu_opportunity_raw_available_count": sum(1 for b in bulu_opportunity if b["raw_available"]),
        "bulu_opportunity_samples": bulu_opportunity[:15],
        "verify_pairs_count": len(verify_pairs),
        "verify_pairs_all_match": all(v["match_source"] and v["match_target"] for v in verify_pairs),
        "verify_pairs_samples": verify_pairs[:10],
        "game_log_examples": game_log,
    }
    atomic_write_json(Path(args.output), out)
    print(json.dumps({k: out[k] for k in
                      ("dist_counts", "bulu_opportunity_count",
                       "bulu_opportunity_raw_available_count",
                       "verify_pairs_count", "verify_pairs_all_match")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
