"""カプ・ブルル中継の戦略ループ(手貼り→交代→攻撃→次オーガポン準備→気絶→昇格)の完遂を計測する。

Phase 11.3 相当。試合数よりイベント数を優先する。

design.md(ogerpon_single_prize_mlp_design.md) Phase1 item4 でこのファイルを拡張し、
`ptcg_ai.ml_policy.ogerpon_option_state`(Phase1で実装済みの固定Option Controller、
純粋な状態遷移関数)を読み取り専用で1手ごとに進め、次を計測できるようにした。

- Option開始/BUILD→READY/READY→ACTIVE/ACTIVE→COMPLETE/ABORT の各遷移回数
- ACTIVE中の離脱理由の内訳(気絶=完遂 / 緊急交代 / 想定外の離脱)
- ループ完遂(`strategy_loop_complete`)を「気絶までACTIVEを継続し、その後に
  攻撃可能な次のオーガポンexへ接続できた場合」だけに厳格化(自発的な早期交代では増えない)
- `bulu_ko_opponent`(ブルルの攻撃で実際に相手をKOした回数)
- `bulu_died_or_lost_before_completion`(4エネ完成/攻撃可能に至る前にブルルを喪失した回数)

Option Controller はこの評価では**意思決定に一切介入しない**(既存 baseline agent の
選択をそのまま `battle_select` に渡す)。`advance_state()` の戻り値は telemetry の
記録にのみ使う。したがって `track_option_state=False` の arm では Option 関連の
telemetry が全て0になるだけで、既存 baseline の行動は変わらない。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from matchup_common import atomic_write_json, read_deck  # noqa: E402
from eval_agent_field import build_field, make_tasks  # noqa: E402

MAX_STEPS = 3000
_W: dict = {}

# Option Controllerが解決する1回の試行(IDLE->BUILD)は、必ずこの3つのいずれか
# ちょうど1つに帰着する(reset_local_option_stateで毎回IDLEへ戻すため)。
# main()の集計で option_start == option_complete + option_abort +
# option_in_progress_at_game_end を検算する。


def classify_emergency_retreat(bulu, opp_active, config) -> bool:
    """ACTIVE中のブルルが交代させられた直前の局面が、design.md 5.4のいう緊急
    (確定リーサル・攻撃不能・確定敗北回避)に近いと静的に判定できるか。

    `search_step` による仮実行(design.md item4のPhase2以降のスコープ)は行わず、
    「その時点で攻撃不能だったか」「確定KO圏内だったか」の2条件だけで近似する。
    確定リーサル・確定敗北回避の厳密な検出はこの関数の対象外(過大にemergency側へ
    倒すより、判定できないことを明示する方を優先する)。
    """
    from ptcg_ai.ml_policy import ogerpon_planner as P

    if bulu is None:
        return True  # 直前の対象を取得できていない = 判定不能。安全側でemergency扱いにする
    if not P.can_attack_now(bulu):
        return True
    cfg = P.main_config(config)
    risk, _ = P.ko_risk(bulu, opp_active, cfg)
    return risk == P.CERTAIN_KO


def detect_ko_after_bulu_attack(target_serial, opp_player) -> bool:
    """ブルルの攻撃直後、攻撃時点でバトル場にいた相手個体(target_serial)が
    相手の場(active+bench)のどこにも見つからなければKOとみなす。

    進化で同じserialのまま姿を変えるケースは未対応(見つからない=KOとして扱う、
    telemetry用の近似)。
    """
    if target_serial is None:
        return False
    for slot in list(opp_player.active or []) + list(opp_player.bench or []):
        if slot is not None and getattr(slot, "serial", None) == target_serial:
            return False
    return True


def _init(weights_path, ml_config_name, workdir, opponents):
    import os
    os.environ["PTCG_AI_ML_CONFIG"] = ml_config_name
    os.chdir(workdir)
    from ptcg_ai.core.config import load_config
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.ml_policy import ml_policy_agent, ogerpon_option_state, ogerpon_planner

    config = dict(load_config(ml_config_name))
    config["policy_weights_path"] = weights_path
    _W["config"] = config
    _W["agent"] = ml_policy_agent
    _W["P"] = ogerpon_planner
    _W["OS"] = ogerpon_option_state
    models = {}
    for _a, wp, _d in opponents:
        if wp not in models:
            models[wp] = PolicyModel(wp)
    _W["opp_models"] = models
    _W["opponents"] = opponents


def _empty_events() -> dict:
    return {
        # --- 既存(維持) ---
        "bulu_attach_opportunity": 0, "bulu_attach_actual": 0,
        "bulu_energy_0to1": 0, "bulu_energy_1to2": 0, "bulu_energy_2to3": 0,
        "bulu_energy_3to4": 0, "bulu_completions": 0,
        "bulu_ready_promotions": 0, "bulu_attacks_total": 0,
        "games_bulu_attacked": 0,
        "attack_tempo_loss_events": 0,
        "final_my_prize": None, "final_opp_prize": None,
        # --- item5: 未更新だった既存カウンタを実イベントへ接続 ---
        "bulu_ko_opponent": 0,
        "bulu_died_or_lost_before_completion": 0,
        # --- item1: Option Controller由来のtelemetry(読み取り専用) ---
        "option_start": 0,
        "option_build_to_ready": 0,
        "option_ready_to_active": 0,
        "option_complete": 0,
        "option_abort": 0,
        "option_abort_target_lost_or_replaced": 0,
        "option_abort_build_deadline_exceeded": 0,
        "option_abort_retreated_unexpectedly": 0,
        "option_in_progress_at_game_end": 0,
        # --- item3: ACTIVE中の離脱の内訳 ---
        "bulu_emergency_retreat": 0,
        "bulu_unexpected_departure": 0,
        # --- item4: ループ完遂(気絶ベースへ厳格化。option_completeと連動) ---
        "strategy_loop_complete": 0,
    }


def _play(task):
    from cg.api import OptionType, SelectType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    P = _W["P"]
    OS = _W["OS"]
    config = _W["config"]
    track_option_state = _W.get("track_option_state", True)
    learner_index, seed, opp_idx = task
    random.seed(seed)
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    ev = _empty_events()
    n = 0
    err = None
    illegal_action = False
    reward = 0.0
    bulu_attacked_this_game = False
    prev_active_serial = None
    saw_bulu_ready_active = False
    saw_bulu_attack = False
    saw_new_ogerpon_ready_after = False

    # Option Controller telemetry用のローカル状態(純粋関数 OS.advance_state を
    # 使う。モジュールグローバルは使わない = ワーカーがプロセス内で複数試合を
    # 処理しても試合間で状態が漏れない)。
    option_state = OS.OgerponOptionState()
    option_resolved_this_attempt = False  # 直近のBUILD開始をcomplete/abortのどちらかへ既に帰着させたか
    prev_bulu_obj = None       # 直前ループの「自分の場にいるブルル個体」(active/bench問わず)
    prev_opp_active_obj = None  # 直前ループの相手アクティブ(emergency判定用スナップショット)
    prev_bulu_ready = False    # 直前ループでのcan_attack_now(色込み)
    opportunity_turn_marker = None  # 直近でbulu_attach_opportunityを計上したcur.turn
    pending_ko_check = None    # {"serial": int} ブルル攻撃直後、次ループで相手個体の消失を確認する

    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"error": f"start {sd.errorType}", "events": ev}
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur, sel = obs.current, obs.select
            if cur is None:
                err = "current None"; break
            if cur.result != -1:
                reward = 1.0 if cur.result == learner_index else 0.0
                ev["final_my_prize"] = len(cur.players[learner_index].prize or [])
                ev["final_opp_prize"] = len(cur.players[1 - learner_index].prize or [])
                break
            if n >= MAX_STEPS:
                err = "max_steps"; break

            if cur.yourIndex == learner_index and sel is not None:
                mine = cur.players[learner_index]
                opp = cur.players[1 - learner_index]
                active = next((s for s in (mine.active or []) if s is not None), None)
                bench = [s for s in (mine.bench or []) if s is not None]
                opp_active = next((s for s in (opp.active or []) if s is not None), None)
                bulu = next((b for b in bench if P.is_bulu(b)), None)
                bulu_active_now = (active is not None and P.is_bulu(active))
                bulu_anywhere = (active if bulu_active_now else bulu)
                hand_energy = [c for c in (mine.hand or []) if c.id == 1]  # 基本【草】

                # --- KOチェック(直前ループでブルルの攻撃を選んでいた場合) ---
                if pending_ko_check is not None:
                    if detect_ko_after_bulu_attack(pending_ko_check["serial"], opp):
                        ev["bulu_ko_opponent"] += 1
                    pending_ko_check = None

                # --- Option Controller telemetry(読み取り専用、行動には使わない) ---
                if track_option_state:
                    prev_mode = option_state.mode
                    option_state = OS.advance_state(
                        option_state, obs, learner_index, config, force_start=True)
                    if prev_mode == OS.IDLE and option_state.mode == OS.BUILD:
                        ev["option_start"] += 1
                        option_resolved_this_attempt = False
                    if prev_mode == OS.BUILD and option_state.mode == OS.READY:
                        ev["option_build_to_ready"] += 1
                    if prev_mode == OS.READY and option_state.mode == OS.ACTIVE:
                        ev["option_ready_to_active"] += 1
                    if option_state.mode == OS.COMPLETE and not option_resolved_this_attempt:
                        ev["option_complete"] += 1
                        option_resolved_this_attempt = True
                        option_state.reset_in_place()
                    elif option_state.mode == OS.ABORT and not option_resolved_this_attempt:
                        ev["option_abort"] += 1
                        reason_key = f"option_abort_{option_state.abort_reason}"
                        if reason_key in ev:
                            ev[reason_key] += 1
                        if prev_mode in (OS.BUILD, OS.READY):
                            ev["bulu_died_or_lost_before_completion"] += 1
                        elif prev_mode == OS.ACTIVE:
                            if classify_emergency_retreat(prev_bulu_obj, prev_opp_active_obj, config):
                                ev["bulu_emergency_retreat"] += 1
                            else:
                                ev["bulu_unexpected_departure"] += 1
                        option_resolved_this_attempt = True
                        option_state.reset_in_place()

                if int(sel.type) == int(SelectType.MAIN):
                    if bulu is not None and hand_energy and not cur.energyAttached \
                            and cur.turn != opportunity_turn_marker:
                        ev["bulu_attach_opportunity"] += 1
                        opportunity_turn_marker = cur.turn

                    action = _W["agent"].agent(obs, config)
                    if action and 0 <= action[0] < len(sel.option):
                        opt = sel.option[action[0]]
                        ot = int(getattr(opt, "type", -1))
                        if ot == int(OptionType.ATTACH) and bulu is not None:
                            tgt = P._resolve_attach_target(cur, learner_index, opt)  # noqa: SLF001
                            if tgt is not None and P.is_bulu(tgt):
                                ev["bulu_attach_actual"] += 1
                                before = len(P.energies_of(bulu))
                                key = {0: "bulu_energy_0to1", 1: "bulu_energy_1to2",
                                      2: "bulu_energy_2to3", 3: "bulu_energy_3to4"}.get(before)
                                if key:
                                    ev[key] += 1
                            if tgt is not None and tgt.id == P.OGERPON_EX_CARD_ID and active is not None \
                                    and not P.can_attack_now(active):
                                ev["attack_tempo_loss_events"] += 1
                        if ot == int(OptionType.ATTACK) and bulu_active_now:
                            ev["bulu_attacks_total"] += 1
                            bulu_attacked_this_game = True
                            saw_bulu_attack = True
                            pending_ko_check = {"serial": getattr(opp_active, "serial", None)}
                else:
                    action = _W["agent"].agent(obs, config)

                # 色込みcan_attack_nowが0->1に変わった瞬間だけ「完成」を計上する
                # (attach総数ベースだと色が合わない手貼りでも完成扱いになってしまう)。
                if bulu_anywhere is not None:
                    ready_now = P.can_attack_now(bulu_anywhere)
                    if (not prev_bulu_ready and ready_now
                            and getattr(bulu_anywhere, "serial", None)
                            == getattr(prev_bulu_obj, "serial", None)):
                        ev["bulu_completions"] += 1
                    prev_bulu_ready = ready_now
                else:
                    prev_bulu_ready = False

                if bulu_active_now and P.can_attack_now(active):
                    if getattr(active, "serial", None) != prev_active_serial:
                        ev["bulu_ready_promotions"] += 1
                        saw_bulu_ready_active = True
                if saw_bulu_ready_active and not bulu_active_now and active is not None \
                        and P.is_ex(active) and P.can_attack_now(active):
                    saw_new_ogerpon_ready_after = True
                if active is not None:
                    prev_active_serial = getattr(active, "serial", None)

                prev_bulu_obj = bulu_anywhere
                prev_opp_active_obj = opp_active
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
            try:
                obs_dict = battle_select(action)
            except IndexError:
                illegal_action = True
                err = "illegal_action"
                break
            n += 1
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    finally:
        battle_finish()

    if track_option_state and option_state.mode in (OS.BUILD, OS.READY, OS.ACTIVE):
        ev["option_in_progress_at_game_end"] += 1
    if bulu_attacked_this_game:
        ev["games_bulu_attacked"] = 1
    # strategy_loop_complete: 気絶(option_complete)まで進み、かつ次のオーガポンexが
    # 攻撃可能な状態で昇格した場合だけ完遂とする。自発的に早期交代しただけでは
    # option_complete が増えないため、この条件では増えない。
    if track_option_state and ev["option_complete"] > 0 and saw_bulu_attack \
            and saw_new_ogerpon_ready_after:
        ev["strategy_loop_complete"] = 1
    return {"error": err, "illegal_action": illegal_action, "reward": reward,
           "archetype": arch, "events": ev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13571113)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from matchup_common import resolve_path

    deck = read_deck(resolve_path(args.deck))
    opponents, weights, _ = build_field(deck_gen="g2")
    tasks = make_tasks(weights, args.games, args.seed)

    _RATE_KEYS = (
        "option_start", "option_complete", "option_abort",
        "bulu_died_or_lost_before_completion", "bulu_emergency_retreat",
        "bulu_unexpected_departure", "bulu_ko_opponent",
        "games_bulu_attacked", "strategy_loop_complete",
    )

    out = {"games": args.games, "arms": {}}
    for a in (json.loads(x) for x in args.arm):
        name, ml_config = a["name"], a["ml_config"]
        track_option_state = bool(a.get("track_option_state", True))
        workdir = tempfile.mkdtemp(prefix=f"bululoop_{name}_")
        Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")
        t0 = time.time()
        with Pool(processes=args.workers, initializer=_init,
                  initargs=(str(resolve_path(args.weights)), ml_config, workdir, opponents)) as pool:
            # track_option_state は各workerプロセス内で_play_with_flag経由で_Wに設定する
            # (Poolはtaskごとに別プロセスで実行されるため、initializerではなくここで渡す)。
            res = pool.starmap(_play_with_flag, [(t, track_option_state) for t in tasks], chunksize=1)
        ok = [r for r in res if r.get("error") is None]
        illegal = sum(1 for r in res if r.get("illegal_action"))
        agg = {}
        for r in ok:
            for k, v in r["events"].items():
                if v is None:
                    continue
                agg[k] = agg.get(k, 0) + v
        wr = sum(1 for r in ok if r["reward"] >= 1.0) / (len(ok) or 1)
        n_valid = len(ok) or 1
        rates = {f"{k}_rate": agg.get(k, 0) / n_valid for k in _RATE_KEYS}
        # 検算: option_start == option_complete + option_abort + option_in_progress_at_game_end
        option_balance_ok = (agg.get("option_start", 0)
                             == agg.get("option_complete", 0) + agg.get("option_abort", 0)
                             + agg.get("option_in_progress_at_game_end", 0))
        out["arms"][name] = {
            "ml_config": ml_config, "track_option_state": track_option_state,
            "valid": len(ok), "errors": len(res) - len(ok), "illegal_actions": illegal,
            "winrate": wr, "wall_seconds": time.time() - t0, "events": agg,
            "rates_per_valid_game": rates, "option_balance_ok": option_balance_ok,
        }
        print(f"[{name}] wr={wr:.3f} valid={len(ok)} errors={len(res)-len(ok)} illegal={illegal} "
              f"手貼り機会={agg.get('bulu_attach_opportunity',0)} "
              f"実手貼り={agg.get('bulu_attach_actual',0)} "
              f"4エネ完成={agg.get('bulu_completions',0)} "
              f"攻撃可昇格={agg.get('bulu_ready_promotions',0)} "
              f"ブルル攻撃={agg.get('bulu_attacks_total',0)} "
              f"KO={agg.get('bulu_ko_opponent',0)} "
              f"option開始={agg.get('option_start',0)} "
              f"option完遂={agg.get('option_complete',0)} "
              f"optionアボート={agg.get('option_abort',0)} "
              f"balance_ok={option_balance_ok} "
              f"ループ完遂={agg.get('strategy_loop_complete',0)} "
              f"{time.time()-t0:.0f}s", flush=True)
    atomic_write_json(Path(resolve_path(args.output)), out)
    print(f"wrote {args.output}", flush=True)


def _play_with_flag(task, track_option_state):
    _W["track_option_state"] = track_option_state
    return _play(task)


if __name__ == "__main__":
    main()
