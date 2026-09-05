"""Phase19.16-T0.5: MAIN root の source/target binding データセットを生成する。"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from _actionq_sampling import StratifiedQuota, cand_band, turn_band  # noqa: E402
from cg.api import AreaType, OptionType, SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.learning import encoder as ENC  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

GROUPS: list[dict] = []
STATS: Counter = Counter()
_ctx: dict = {"game": 0, "arch": "?", "first": True, "rec": False,
              "idx": 0, "taken": 0}
OPTS = {"top_k": 6, "max_cands": 8, "m": 8, "max_steps": 300,
        "assert_production_safe": True}
QUOTA: StratifiedQuota | None = None
EV = {"value": None}


def _enum_value(value):
    return int(value) if value is not None else None


def _option_raw(option) -> dict:
    """Option を engine enum に依存しない JSON 値へ写す。"""
    return {
        "type": _enum_value(getattr(option, "type", None)),
        "number": getattr(option, "number", None),
        "area": _enum_value(getattr(option, "area", None)),
        "index": getattr(option, "index", None),
        "playerIndex": getattr(option, "playerIndex", None),
        "toolIndex": getattr(option, "toolIndex", None),
        "energyIndex": getattr(option, "energyIndex", None),
        "count": getattr(option, "count", None),
        "inPlayArea": _enum_value(getattr(option, "inPlayArea", None)),
        "inPlayIndex": getattr(option, "inPlayIndex", None),
        "attackId": getattr(option, "attackId", None),
        "cardId": getattr(option, "cardId", None),
        "serial": getattr(option, "serial", None),
        "specialConditionType": _enum_value(getattr(option, "specialConditionType", None)),
    }


def _empty_binding() -> dict:
    return {"area": None, "index": None, "player_index": None,
            "serial": None, "card_id": None}


def _binding(option, state, *, in_play: bool = False) -> tuple[dict, bool]:
    """Option の source または in-play target を公開 card/Pokemon へ対応付ける。"""
    binding = _empty_binding()
    if state is None:
        return binding, False
    area = getattr(option, "inPlayArea", None) if in_play else getattr(option, "area", None)
    index = getattr(option, "inPlayIndex", None) if in_play else getattr(option, "index", None)
    if not in_play and area is None and getattr(option, "type", None) == OptionType.PLAY:
        area = AreaType.HAND
    if area is None or index is None:
        if not in_play:
            binding.update(player_index=state.yourIndex,
                           serial=getattr(option, "serial", None),
                           card_id=ENC._resolve_card_id(option, state))
        return binding, True
    option_player = getattr(option, "playerIndex", None)
    player_index = option_player if option_player is not None else state.yourIndex
    binding.update(area=_enum_value(area), index=index, player_index=player_index)
    if not (0 <= player_index < len(state.players)):
        return binding, False

    if in_play:
        pokemon = ENC._resolve_in_play_pokemon(option, state)
        if pokemon is None:
            return binding, False
        binding.update(serial=getattr(pokemon, "serial", None), card_id=getattr(pokemon, "id", None))
        return binding, binding["serial"] is not None or binding["card_id"] is not None

    tool_index = getattr(option, "toolIndex", None)
    energy_index = getattr(option, "energyIndex", None)
    pokemon = ENC._resolve_pokemon(option, state)
    if pokemon is not None and tool_index is None and energy_index is None:
        binding.update(serial=getattr(pokemon, "serial", None), card_id=getattr(pokemon, "id", None))
        return binding, binding["serial"] is not None or binding["card_id"] is not None

    card_id = ENC._resolve_card_id(option, state)
    serial = getattr(option, "serial", None)
    try:
        zone = ENC._zone_entries_for_option(area, state.players[player_index], state)
        card = zone[index] if zone is not None and 0 <= index < len(zone) else None
        if card is not None and tool_index is not None:
            card = card.tools[tool_index] if 0 <= tool_index < len(card.tools) else None
        elif card is not None and energy_index is not None:
            card = card.energyCards[energy_index] if 0 <= energy_index < len(card.energyCards) else None
        if card is not None:
            serial = getattr(card, "serial", serial)
    except (AttributeError, IndexError, TypeError):
        pass
    binding.update(serial=serial, card_id=card_id)
    return binding, serial is not None or card_id is not None


def _option_bindings(option, state) -> tuple[dict, dict, int]:
    source, source_ok = _binding(option, state)
    target, target_ok = _binding(option, state, in_play=True)
    has_target = (getattr(option, "inPlayArea", None) is not None
                  or getattr(option, "inPlayIndex", None) is not None)
    return source, target, int(not source_ok) + int(has_target and not target_ok)


def _entity(pokemon, owner: int, zone: str, slot_index: int) -> dict:
    return {
        "serial": getattr(pokemon, "serial", None),
        "card_id": getattr(pokemon, "id", None),
        "owner": owner,
        "zone": zone,
        "slot_index": slot_index,
        "hp": getattr(pokemon, "hp", None),
        "max_hp": getattr(pokemon, "maxHp", None),
        "energies": len(getattr(pokemon, "energies", None) or []),
        "energy_card_ids": [card.id for card in (getattr(pokemon, "energyCards", None) or [])],
        "tool_card_ids": [card.id for card in (getattr(pokemon, "tools", None) or [])],
        "pre_evolution_ids": [card.id for card in (getattr(pokemon, "preEvolution", None) or [])],
        "appear_this_turn": bool(getattr(pokemon, "appearThisTurn", False)),
    }


def snapshot_entities(state, me: int) -> list[dict]:
    """盤面上の Pokemon.serial をキーにした公開 entity snapshot を作る。"""
    if state is None:
        return []
    out = []
    for player_index, player in enumerate(state.players):
        owner = 0 if player_index == me else 1
        for i, pokemon in enumerate(getattr(player, "active", None) or []):
            if pokemon is not None:
                out.append(_entity(pokemon, owner, "active", i))
        for i, pokemon in enumerate(getattr(player, "bench", None) or []):
            if pokemon is not None:
                out.append(_entity(pokemon, owner, "bench", i))
    return out


def snapshot_global(state, me: int) -> dict:
    """production Observation に公開される global 情報だけを写す。"""
    mine, opp = state.players[me], state.players[1 - me]
    stadium = getattr(state, "stadium", None) or []
    return {
        "prize_self": len(getattr(mine, "prize", None) or []),
        "prize_opp": len(getattr(opp, "prize", None) or []),
        "hand_count_self": int(getattr(mine, "handCount", 0) or 0),
        "hand_count_opp": int(getattr(opp, "handCount", 0) or 0),
        "deck_self": int(getattr(mine, "deckCount", 0) or 0),
        "deck_opp": int(getattr(opp, "deckCount", 0) or 0),
        "discard_self_ids": [card.id for card in (getattr(mine, "discard", None) or [])],
        "discard_opp_ids": [card.id for card in (getattr(opp, "discard", None) or [])],
        "turn": int(getattr(state, "turn", 0)),
        "supporter_played": bool(getattr(state, "supporterPlayed", False)),
        "energy_attached": bool(getattr(state, "energyAttached", False)),
        "retreated": bool(getattr(state, "retreated", False)),
        "stadium_id": stadium[0].id if stadium else None,
    }


def snapshot_hand(state, me: int) -> list[int]:
    hand = state.players[me].hand
    return [card.id for card in (hand or [])]


def delta_by_serial(before: list[dict], after: list[dict]) -> dict:
    """serial を維持して before/after の HP・energy・zone 差分を表す。"""
    before_by = {row["serial"]: row for row in before if row["serial"] is not None}
    after_by = {row["serial"]: row for row in after if row["serial"] is not None}
    out = {}
    for serial in sorted(set(before_by) | set(after_by)):
        old, new = before_by.get(serial), after_by.get(serial)
        old_zone = old["zone"] if old is not None else None
        new_zone = new["zone"] if new is not None else None
        out[str(serial)] = {
            "hp_delta": (new["hp"] if new is not None else 0) - (old["hp"] if old is not None else 0),
            "energy_delta": (new["energies"] if new is not None else 0) - (old["energies"] if old is not None else 0),
            "zone_change": {"from": old_zone, "to": new_zone} if old_zone != new_zone else None,
            "appeared": old is None,
            "disappeared": new is None,
        }
    return out


def _trajectory_entry(step: int, actor: int, option_index: int, option, state) -> tuple[dict, int]:
    source, target, failed = _option_bindings(option, state)
    return {
        "step": step,
        "actor": actor,
        "option_index": option_index,
        "select_type": _enum_value(getattr(state, "select_type", None)),
        "option_raw": _option_raw(option),
        "action_card_id": ENC._resolve_card_id(option, state),
        "source": source,
        "target": target,
    }, failed


def _selection_trajectory_entries(step: int, actor: int, obs, selection: list[int]) -> tuple[list[dict], int]:
    entries, failed = [], 0
    for index in selection:
        if not (0 <= index < len(obs.select.option)):
            continue
        entry, failed_one = _trajectory_entry(step, actor, index, obs.select.option[index], obs.current)
        entry["select_type"] = _enum_value(obs.select.type)
        entries.append(entry)
        failed += failed_one
    return entries, failed


def _begin(obs, me: int, seed: int):
    hidden = search_adapter.to_search_begin_kwargs(
        match_context.get_own_state(me), match_context.get_opponent_state(me),
        obs, rng=random.Random(seed))
    return P._begin(obs, hidden)


def _roll_candidate(obs, me: int, action: int, seed: int, policy_model, before: list[dict]) -> dict | None:
    """root action の直後 snapshot と、自分の次 turn 開始までの policy trajectory を採取する。"""
    root = child = None
    select = obs.select
    option = select.option[action]
    root_entry, resolve_fail = _trajectory_entry(0, me, action, option, obs.current)
    root_entry["select_type"] = _enum_value(select.type)
    teacher = {"C1b": None, "C4": None}
    trajectory = [root_entry]
    try:
        root = _begin(obs, me, seed)
        child = P.cg_api.search_step(root.searchId, [action])
        node = child
        after_state = node.observation.current
        after = snapshot_entities(after_state, me)
        previous_actor = me
        for step in range(1, OPTS["max_steps"] + 1):
            current_obs = node.observation
            state = current_obs.current
            if state is None or state.result != -1:
                break
            actor = state.yourIndex
            if previous_actor == me and actor != me and teacher["C1b"] is None:
                teacher["C1b"] = float(EV["value"].evaluate(state, me))
            if previous_actor != me and actor == me:
                teacher["C4"] = float(EV["value"].evaluate(state, me))
                break
            if current_obs.select is None or not current_obs.select.option:
                break
            selected = P._greedy_selection(policy_model, current_obs)
            if not selected:
                break
            entries, failed = _selection_trajectory_entries(step, actor, current_obs, selected)
            trajectory.extend(entries)
            resolve_fail += failed
            try:
                node = P.cg_api.search_step(node.searchId, selected)
            except ValueError:
                break
            previous_actor = actor
        return {"after_entities": after, "delta_by_serial": delta_by_serial(before, after),
                "trajectory": trajectory, "traj_len": len(trajectory), "teacher": teacher,
                "resolve_fail": resolve_fail}
    except Exception as exc:  # noqa: BLE001
        STATS["roll_" + type(exc).__name__] += 1
        return None
    finally:
        for state_id in (child.searchId if child is not None else None,
                         root.searchId if root is not None else None):
            if state_id is not None:
                try:
                    P.cg_api.search_release(state_id)
                except Exception:  # noqa: BLE001
                    pass
        try:
            P.cg_api.search_end()
        except Exception:  # noqa: BLE001
            pass


def _assert_production_safe(row: dict) -> None:
    """非公開の相手 hand/deck/looking と terminal 結果を出力していないことを検査する。"""
    encoded = json.dumps(row, ensure_ascii=False)
    forbidden = ("looking", "opponent_hand", "opp_hand", "opponent_deck", "result")
    assert not any(token in encoded for token in forbidden)
    assert isinstance(row["before_hand"], list)


def audit(obs, me: int, policy_model, cfg_full, seed: int) -> dict | None:
    select, state = obs.select, obs.current
    if select is None or state is None:
        return None
    try:
        scores = policy_model.score_options(
            obs, MA._model_hidden_state_factory(obs, cfg_full), time.perf_counter() + 5.0)
    except Exception as exc:  # noqa: BLE001
        STATS["policy_" + type(exc).__name__] += 1
        return None
    if not scores or len(scores) != len(select.option):
        return None
    probs = P._softmax(scores)
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    candidates = ranked[:OPTS["top_k"]][:OPTS["max_cands"]]
    before = snapshot_entities(state, me)
    rows, resolve_fail = [], 0
    for ordinal, index in enumerate(candidates):
        option = select.option[index]
        source, target, failed = _option_bindings(option, state)
        rolls = []
        for sample in range(OPTS["m"]):
            rolled = _roll_candidate(obs, me, index, seed + sample, policy_model, before)
            if rolled is not None:
                rolls.append(rolled)
        if not rolls:
            continue
        rolled = rolls[0]
        resolve_fail += failed + sum(sample["resolve_fail"] for sample in rolls)
        teacher = {"C1b": [sample["teacher"]["C1b"] for sample in rolls
                            if sample["teacher"]["C1b"] is not None],
                   "C4": [sample["teacher"]["C4"] for sample in rolls
                          if sample["teacher"]["C4"] is not None]}
        if len(teacher["C1b"]) != OPTS["m"] or len(teacher["C4"]) != OPTS["m"]:
            STATS["incomplete_teacher"] += 1
            continue
        rows.append({
            "option_index": index,
            "option_raw": _option_raw(option),
            "action_card_id": ENC._resolve_card_id(option, state),
            "source": source,
            "target": target,
            "policy_score": float(scores[index]),
            "policy_prob": float(probs[index]),
            **{key: value for key, value in rolled.items()
               if key not in {"teacher", "resolve_fail"}},
            "teacher": teacher,
        })
    if not rows:
        return None
    row = {
        "n_legal_options": len(select.option),
        "me": me,
        "before_entities": before,
        "before_global": snapshot_global(state, me),
        "before_hand": snapshot_hand(state, me),
        "candidates": rows,
        "resolve_fail": resolve_fail,
    }
    if OPTS["assert_production_safe"]:
        _assert_production_safe(row)
    return row


def make_probe(inner, cfg_full, policy_model):
    def probe(obs):
        act = inner(obs)
        if obs.select is None or not _ctx["rec"] or QUOTA is None:
            return act
        _ctx["idx"] += 1
        select, state = obs.select, obs.current
        if (select.type != SelectType.MAIN or select.maxCount != 1 or not select.option
                or state is None or state.result != -1):
            return act
        turn = int(state.turn)
        if not QUOTA.accept(turn, len(select.option), _ctx["arch"], _ctx["taken"]):
            return act
        started = time.perf_counter()
        try:
            row = audit(obs, state.yourIndex, policy_model, cfg_full,
                        7_000_000 + _ctx["game"] * 997 + _ctx["idx"])
        except Exception as exc:  # noqa: BLE001
            STATS["audit_" + type(exc).__name__] += 1
            return act
        if row is None:
            return act
        row.update({"group_id": "g{}_{}".format(_ctx["game"], _ctx["idx"]),
                    "game": _ctx["game"], "arch": _ctx["arch"], "turn": turn,
                    "turn_band": turn_band(turn), "cand_band": cand_band(len(select.option)),
                    "me_first": _ctx["first"],
                    "audit_ms": round((time.perf_counter() - started) * 1000, 1)})
        GROUPS.append(row)
        QUOTA.commit(turn, len(select.option), _ctx["arch"])
        _ctx["taken"] += 1
        STATS["root"] += 1
        STATS["resolve_fail"] += row["resolve_fail"]
        return act
    return probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--per-game-cap", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-cands", type=int, default=8)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--offset", type=int, default=4_300_000)
    parser.add_argument("--opponents", default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,archaludon_ex,shirona_garchomp_ex")
    parser.add_argument("--tag", default="p0")
    parser.add_argument("--assert-production-safe", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.num_workers > 1:
        args.target = max(1, args.target // args.num_workers)
    OPTS.update(top_k=args.top_k, max_cands=args.max_cands, m=args.m,
                max_steps=args.max_steps, assert_production_safe=args.assert_production_safe)

    global QUOTA
    QUOTA = StratifiedQuota(args.target, per_game_cap=args.per_game_cap, cand_soft_cap=0.50)
    EV["value"] = leaf_eval_module.build_evaluator({"kind": "value"})
    cfg = agents.load_config_copy("abl_5_full")
    cfg["policy_weights_path"] = CLIMB
    climb = agents.make_ml_policy_agent(copy.deepcopy(cfg))
    policy_model = MA._get_model({"policy_weights_path": CLIMB})
    probe = make_probe(climb, cfg, policy_model)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    opponents = [name for name in args.opponents.split(",") if name]
    started = time.perf_counter()
    for game_index in range(args.max_games):
        if QUOTA.total >= args.target:
            break
        game = args.offset + args.worker_id + game_index * args.num_workers
        arch = opponents[game % len(opponents)]
        cfg_opp = agents.load_config_copy("abl_5_full")
        cfg_opp["policy_weights_path"] = str(_WDIR / "policy_weights_{}.json".format(arch))
        opponent = agents.make_ml_policy_agent(cfg_opp)
        first = game_index % 2 == 0
        _ctx.update(game=game, arch=arch, first=first, rec=True, idx=0, taken=0)
        deck_opp = runner.load_deck(_DECKDIR / arch / "01.csv")
        if first:
            runner.play_game(probe, opponent, deck_c, deck_opp)
        else:
            runner.play_game(opponent, probe, deck_opp, deck_c)
        _ctx["rec"] = False
        if game_index % 3 == 0:
            print("  [w{}] game#{} roots={}/{} {}min".format(
                args.worker_id, game, QUOTA.total, args.target,
                int((time.perf_counter() - started) / 60)), file=sys.stderr, flush=True)
    output = _HERE / "_p1916b_{}.jsonl.gz".format(args.tag)
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        for row in GROUPS:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"tag": args.tag, "n": len(GROUPS), "stats": dict(STATS),
                      "quota": QUOTA.report()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
