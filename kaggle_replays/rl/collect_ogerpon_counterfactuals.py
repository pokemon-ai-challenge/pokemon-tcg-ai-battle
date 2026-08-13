"""design.md(ogerpon_single_prize_mlp_design.md)§9(Phase2 item2): カプ・ブルル中継戦略の
反実仮想(counterfactual)paired rollout収集。

同じ公開盤面(Strategy Window Builderが発火した局面)から `EX_TEMPO` /
`SINGLE_PRIZE_ROTATION` を強制分岐させ、可能な限り同じ非公開情報仮説・相手Policyで
rolloutして勝敗等を記録する。将来のQ-critic(Phase3)学習の教師データを作るための
オフライン収集スクリプトであり、意思決定には一切使わない(既存の本番エージェント・
configには触れない)。

## 手順(design.md §9.3 に対応)

1. 通常のself-play(learner=既存Policy+lethal / opponent=archetype別Policy)を1本進める。
2. Strategy Window Builder(``ogerpon_strategy.detect_trigger``)が発火した局面ごとに
   公開状態(``Observation``)を保存する(通常の対戦はそのまま続ける)。
3. 保存した状態ごとに、``search_begin`` 用の hidden state を N 回サンプルする(決定化)。
4. 決定化ごとに、Optionごとに ``search_begin`` でrootを作り直す(design.mdの推奨どおり、
   1つのrootをsearch_stepで分岐させるのではなく、同じhidden state payloadから
   Optionごとに新しいrootを作る。速度よりペアの公平性を優先する)。
5. ``first_action`` を強制した後は、学習側は既存Policyのgreedy選択で継続する
   (``pipeline.py`` の内部rolloutと同じ簡略化。lethal探索・PIMC再帰は行わない)。
   ``SINGLE_PRIZE_ROTATION`` 側だけ、``ogerpon_planner`` の低位candidate resolverを
   Option Controllerとして働かせる。相手側は実際に学習済みのarchetype別Policyで進める。
6. 終局・最大ターン・最大stepまでrolloutし、勝敗・ターン数・KO数・ブルル攻撃数・
   取得サイド・ループ完遂・abort理由を記録する。
7. child/rootは必ず ``search_release`` し、1状態の処理が終わるごとに ``search_end`` する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from matchup_common import append_jsonl, atomic_write_json, git_commit_sha, read_deck  # noqa: E402
from eval_agent_field import build_field, make_tasks  # noqa: E402

SCHEMA_VERSION = 1
MAX_OUTER_STEPS = 3000       # 外側(state収集用)の1試合あたり上限
MAX_ROLLOUT_STEPS = 400      # 反実仮想rolloutの1本あたり上限
_W: dict = {}


def _init(weights_path, ml_config_name, workdir, opponents):
    import os
    os.environ["PTCG_AI_ML_CONFIG"] = ml_config_name
    os.chdir(workdir)
    from ptcg_ai.core.config import load_config
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.ml_policy import ml_policy_agent, ogerpon_option_state, ogerpon_planner, ogerpon_strategy

    config = dict(load_config(ml_config_name))
    config["policy_weights_path"] = weights_path
    _W["config"] = config
    _W["agent"] = ml_policy_agent
    _W["P"] = ogerpon_planner
    _W["OS"] = ogerpon_option_state
    _W["STRAT"] = ogerpon_strategy
    models = {}
    for _a, wp, _d in opponents:
        if wp not in models:
            models[wp] = PolicyModel(wp)
    _W["opp_models"] = models
    _W["opponents"] = opponents
    _W["learner_policy"] = PolicyModel(weights_path)
    _W["git_commit"] = git_commit_sha()


def state_id_of(state) -> str:
    """公開盤面(``obs.current``。logsは含まれない)のsha256。"""
    payload = json.dumps(asdict(state), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. 通常self-playを進めながらStrategy Windowの発火局面を集める
# ---------------------------------------------------------------------------

def _collect_states_from_one_game(task, max_states_per_game: int):
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    P, OS, STRAT = _W["P"], _W["OS"], _W["STRAT"]
    config = _W["config"]
    learner_index, seed, opp_idx = task
    random.seed(seed)
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    saved: list[dict] = []
    option_state = OS.OgerponOptionState()
    err = None
    n = 0
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"error": f"start {sd.errorType}", "saved": saved}
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur, sel = obs.current, obs.select
            if cur is None:
                err = "current None"; break
            if cur.result != -1:
                break
            if n >= MAX_OUTER_STEPS:
                err = "max_steps"; break

            if cur.yourIndex == learner_index and sel is not None:
                option_state = OS.advance_state(
                    option_state, obs, learner_index, config, force_start=True)
                baseline_action = _W["agent"].agent(obs, config)
                if len(saved) < max_states_per_game:
                    trigger = STRAT.detect_trigger(obs, learner_index, deck_l, option_state.mode)
                    if trigger is not None:
                        pair = STRAT.build_candidates(
                            obs, learner_index, config, deck_l, trigger, baseline_action)
                        if pair is not None:
                            saved.append({
                                "obs": obs, "pair": pair, "trigger_kind": trigger,
                                "opponent_archetype": arch, "opponent_weights_path": wpath,
                                "learner_index": learner_index, "match_seed": seed,
                                "turn": cur.turn,
                            })
                action = baseline_action
            else:
                if sel is None or not sel.option:
                    action = []
                elif sel.maxCount == 1:
                    oi = opp_pm.select_option(obs)
                    action = [oi if oi is not None else 0]
                else:
                    osc = opp_pm.score_options(obs)
                    nn = len(sel.option)
                    c = max(sel.minCount, min(sel.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: osc[i], reverse=True)[:c]
                              if osc else list(range(c)))
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    finally:
        battle_finish()

    return {"error": err, "saved": saved, "deck_l": deck_l}


# ---------------------------------------------------------------------------
# 2. 保存済み状態ごとに、Optionごとのpaired rolloutを実行する
# ---------------------------------------------------------------------------

def _learner_rollout_action(obs, option_name, deck_ids):
    """rollout中の学習側の一手。EX_TEMPOは素のPolicy greedy。SINGLE_PRIZE_ROTATIONは
    ogerpon_plannerのcandidate resolverを低位Option Controllerとして先に試し、
    無ければ素のPolicy greedyへフォールバックする(design.md §21の再利用方針)。
    """
    P = _W["P"]
    config = _W["config"]
    policy_model = _W["learner_policy"]
    select = obs.select
    if select is None or not select.option:
        return []
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))

    if option_name == "SINGLE_PRIZE_ROTATION" and select.maxCount == 1:
        forced = _single_prize_low_level_action(obs, P, config, deck_ids)
        if forced is not None:
            return forced

    try:
        scores = policy_model.score_options_from_state(obs.current, select)
    except Exception:  # noqa: BLE001
        scores = []
    if not scores or len(scores) != n:
        return list(range(count))
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return ranked[:count]


def _single_prize_low_level_action(obs, P, config, deck_ids):
    """SINGLE_PRIZE_ROTATION rollout用の低位Controller。安全に強制できる手があれば
    それを返す。無ければNone(呼び出し側は素のPolicy greedyへフォールバック)。

    「今ターンの攻撃を犠牲にしてまで強制しない」(design.md §5.4のハード中断条件と
    同じ考え方)。既存の候補resolverをそのまま再利用する(bonus値のスケール問題
    (design.md §3.2 item4)を避けるため、スコアを混ぜずに合法ならそのまま採用する
    forceスタイルにしている)。
    """
    from cg.api import SelectType

    select = obs.select
    if select.maxCount != 1:
        return None
    state = obs.current
    me = state.yourIndex
    stype = int(select.type)

    if stype == int(SelectType.MAIN):
        forced_cfg = _forced_planner_config(config)
        idx = P.select_strategic_attach_candidate(obs, me, forced_cfg, deck_ids)
        if idx is None:
            return None
        mine = state.players[me]
        active = next((s for s in (mine.active or []) if s is not None), None)
        if active is not None and not P.can_attack_now(active):
            return None
        return [idx]

    if stype == int(SelectType.CARD):
        forced_cfg = _forced_planner_config(config)
        adj = P.score_adjustments(obs, me, forced_cfg, deck_ids)
        if adj is None or not any(adj):
            return None
        best = max(range(len(adj)), key=lambda i: adj[i])
        if adj[best] <= 0:
            return None
        return [best]

    return None


def _forced_planner_config(config):
    base = dict((config or {}).get("ogerpon_planner") or {})
    base["enabled"] = True
    base["attach_enabled"] = True
    return {**(config or {}), "ogerpon_planner": base}


def _opponent_rollout_action(obs, opp_pm):
    select = obs.select
    if select is None or not select.option:
        return []
    if select.maxCount == 1:
        oi = opp_pm.select_option(obs)
        return [oi if oi is not None else 0]
    scores = opp_pm.score_options(obs)
    n = len(select.option)
    c = max(select.minCount, min(select.maxCount, n))
    return sorted(range(n), key=lambda i: scores[i], reverse=True)[:c] if scores else list(range(c))


def _rollout_option(root_obs, hidden_state, candidate, learner_index, opp_pm, deck_ids, option_cfg):
    """1Option・1決定化のrollout。design.md §9.5のoutcomeスキーマを返す。

    design.mdの推奨どおり、この呼び出しごとに新しい ``search_begin`` root を作る
    (同一rootからのsearch_step分岐は使わない。速度よりペアの公平性を優先する)。
    呼び出し側が ``search_end()`` を呼ぶまでメモリは解放されないため、root/leafの
    ``search_release`` はここで必ず行う。
    """
    from cg import api as cg_api
    from cg.api import OptionType

    P, OS = _W["P"], _W["OS"]
    outcome = {
        "win": None, "terminal_turns": None, "opponent_ko_count": 0,
        "bulu_attack_count": 0, "bulu_prizes_taken": 0, "loop_complete": False,
        "abort_reason": None, "error": None,
    }
    root = None
    try:
        root = cg_api.search_begin(
            root_obs, hidden_state["your_deck"], hidden_state["your_prize"],
            hidden_state["opponent_deck"], hidden_state["opponent_prize"],
            hidden_state["opponent_hand"], hidden_state["opponent_active"])
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = f"search_begin: {exc!r}"
        return outcome

    try:
        try:
            node = cg_api.search_step(root.searchId, candidate.first_action)
        except ValueError as exc:
            outcome["error"] = f"illegal_first_action: {exc!r}"
            return outcome

        option_state = OS.OgerponOptionState()
        option_resolved = False
        prev_opp_prize = None
        pending_ko_check = None  # {"serial": int} ブルル攻撃直後に相手個体の消失を確認する

        for _ in range(MAX_ROLLOUT_STEPS):
            obs = node.observation
            cur = obs.current
            if cur is None:
                outcome["error"] = "current None"
                break
            if cur.result != -1:
                outcome["win"] = 1 if cur.result == learner_index else 0
                outcome["terminal_turns"] = cur.turn
                break

            opp_now = cur.players[1 - learner_index]
            if prev_opp_prize is not None:
                taken = max(0, prev_opp_prize - len(opp_now.prize or []))
                if taken and pending_ko_check is not None:
                    outcome["bulu_prizes_taken"] += taken
                    pending_ko_check = None  # 一度計上したら次のブルル攻撃まで再計上しない
            prev_opp_prize = len(opp_now.prize or [])

            actor = cur.yourIndex
            if actor == learner_index and obs.select is not None:
                if candidate.option_name == "SINGLE_PRIZE_ROTATION":
                    option_state = OS.advance_state(
                        option_state, obs, learner_index, option_cfg, force_start=True)
                    if option_state.mode == OS.COMPLETE and not option_resolved:
                        option_resolved = True
                    elif option_state.mode == OS.ABORT and not option_resolved:
                        outcome["abort_reason"] = option_state.abort_reason
                        option_resolved = True

                select = obs.select
                if (select.maxCount == 1
                        and any(int(getattr(o, "type", -1)) == int(OptionType.ATTACK)
                               for o in select.option)):
                    mine = cur.players[learner_index]
                    active_now = next((s for s in (mine.active or []) if s is not None), None)
                    if active_now is not None and P.is_bulu(active_now):
                        chosen_is_attack = True
                    else:
                        chosen_is_attack = False
                else:
                    chosen_is_attack = False

                action = _learner_rollout_action(obs, candidate.option_name, deck_ids)
                if chosen_is_attack and action and 0 <= action[0] < len(select.option) \
                        and int(getattr(select.option[action[0]], "type", -1)) == int(OptionType.ATTACK):
                    outcome["bulu_attack_count"] += 1
                    opp_active_now = next((s for s in (opp_now.active or []) if s is not None), None)
                    pending_ko_check = {"serial": getattr(opp_active_now, "serial", None)}
                else:
                    pending_ko_check = None
            else:
                action = _opponent_rollout_action(obs, opp_pm)

            try:
                node = cg_api.search_step(node.searchId, action)
            except ValueError as exc:
                outcome["error"] = f"illegal_action: {exc!r}"
                break
        else:
            outcome["error"] = outcome["error"] or "max_rollout_steps"

        if candidate.option_name == "SINGLE_PRIZE_ROTATION":
            outcome["loop_complete"] = (option_state.mode == OS.COMPLETE)
        # 学習側の被弾KO数(= 相手が実際に行ったKO数)。開始時点との残サイド差で数える。
        try:
            start_prize = len(root_obs.current.players[learner_index].prize or [])
            end_prize = len(node.observation.current.players[learner_index].prize or []) \
                if node.observation.current is not None else start_prize
            outcome["opponent_ko_count"] = max(0, start_prize - end_prize)
        except Exception:  # noqa: BLE001
            pass
        return outcome
    finally:
        try:
            cg_api.search_release(root.searchId)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 3. 1状態 x N決定化 x 2Option を処理し、design.md §9.5 のレコードを組み立てる
# ---------------------------------------------------------------------------

def _process_saved_state(item: dict, determinizations: int, full_deck: list[int]) -> list[dict]:
    from cg import api as cg_api
    from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state

    obs = item["obs"]
    ex_candidate, single_candidate = item["pair"]
    learner_index = item["learner_index"]
    opp_pm = _W["opp_models"][item["opponent_weights_path"]]
    state_id = state_id_of(obs.current)
    rng = random.Random((hash((item["match_seed"], item["turn"], state_id))) & 0xFFFFFFFF)
    records: list[dict] = []
    try:
        for det_id in range(determinizations):
            hidden_state = build_dummy_search_state(obs, full_deck, rng=rng)
            if hidden_state is None:
                continue
            for candidate in (ex_candidate, single_candidate):
                outcome = _rollout_option(
                    obs, hidden_state, candidate, learner_index, opp_pm, full_deck, _W["config"])
                records.append({
                    "schema_version": SCHEMA_VERSION,
                    "git_commit": _W["git_commit"],
                    "state_id": state_id,
                    "pair_id": f"{state_id}:{det_id}",
                    "match_seed": item["match_seed"],
                    "determinization_id": det_id,
                    "opponent_archetype": item["opponent_archetype"],
                    "learner_index": learner_index,
                    "trigger_kind": item["trigger_kind"],
                    "option_name": candidate.option_name,
                    # Phase3のogerpon_strategy_encoderが実装されるまでは空配列のまま
                    # (design.md §9.5のスキーマ例自体も空配列で示されている)。
                    "continuous_features": [],
                    "slot_card_ids": [],
                    "option_features": [],
                    "first_action_identity": {
                        "first_action": candidate.first_action,
                        "target_serial": candidate.target_serial,
                        "target_card_id": candidate.target_card_id,
                        "turns_until_ready": candidate.turns_until_ready,
                        "required_ko_gain": candidate.required_ko_gain,
                        "safety_flags": candidate.safety_flags,
                    },
                    "outcome": outcome,
                })
    finally:
        try:
            cg_api.search_end()
        except Exception:  # noqa: BLE001
            pass
    return records


def _process_task(task, determinizations: int, max_states_per_game: int) -> dict:
    outer = _collect_states_from_one_game(task, max_states_per_game)
    saved = outer.get("saved") or []
    if not saved:
        return {"error": outer.get("error"), "records": [], "n_states": 0}
    full_deck = outer["deck_l"]
    records: list[dict] = []
    for item in saved:
        records.extend(_process_saved_state(item, determinizations, full_deck))
    return {"error": outer.get("error"), "records": records, "n_states": len(saved)}


def _process_task_star(packed):
    task, determinizations, max_states_per_game = packed
    return _process_task(task, determinizations, max_states_per_game)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ml-config", default="abl_5_full")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--determinizations", type=int, default=4)
    ap.add_argument("--max-states-per-game", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13571113)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from matchup_common import resolve_path

    deck = read_deck(resolve_path(args.deck))
    opponents, weights, _ = build_field(deck_gen="g2")
    tasks = make_tasks(weights, args.games, args.seed)

    workdir = tempfile.mkdtemp(prefix="ogercf_")
    Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")

    out_path = resolve_path(args.output)
    if out_path.exists():
        out_path.unlink()

    t0 = time.time()
    n_states = n_records = n_outer_errors = n_rollout_errors = 0
    packed_tasks = [(t, args.determinizations, args.max_states_per_game) for t in tasks]
    with Pool(processes=args.workers, initializer=_init,
              initargs=(str(resolve_path(args.weights)), args.ml_config, workdir, opponents)) as pool:
        for res in pool.imap_unordered(_process_task_star, packed_tasks):
            if res.get("error"):
                n_outer_errors += 1
            n_states += res.get("n_states", 0)
            for rec in res.get("records", []):
                append_jsonl(out_path, rec)
                n_records += 1
                if rec["outcome"].get("error"):
                    n_rollout_errors += 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "games": args.games, "determinizations": args.determinizations,
        "max_states_per_game": args.max_states_per_game,
        "n_states_collected": n_states, "n_rollout_records": n_records,
        "n_outer_game_errors": n_outer_errors, "n_rollout_errors": n_rollout_errors,
        "wall_seconds": time.time() - t0, "output": str(out_path),
    }
    atomic_write_json(Path(str(out_path) + ".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
