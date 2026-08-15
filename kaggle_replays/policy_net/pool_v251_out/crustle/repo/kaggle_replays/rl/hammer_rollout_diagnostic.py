"""run4: 「逃した」ハンマー決定点で、ロールアウトの中身を観測する。

背景
----
`hammer_diagnostic.py` の run3 は、tie_eps を 0.02 -> 0.010/0.005/0.002 に下げても
ハンマー使用率が改善しないことを示した(コーディネータ側で集計済み、標準誤差の範囲内で
tie_eps 仮説は棄却)。次の仮説は「ハンマーを撃った後のロールアウトが、そもそも
Powerful Hand を撃たない」というもの。

`pipeline.py` の Step4 (`_evaluate_candidate` -> `_rollout_and_eval`) は、候補 first move を
適用した後、自ターンの残りを `_greedy_selection`(Policy の argmax)で展開し、相手ターンを
`opponent_depth` 回展開してから末端評価する。この `_greedy_selection` はまさに「ハンマーより
引く・並べるを上位に置いた」当の方策なので、ロールアウトが方策の偏りを引き継いで確殺が
末端に現れない可能性がある。

このスクリプトは `pipeline.py` を一切変更せず、その module-level 関数
(`_greedy_selection`)をそのまま import して呼び、同じ手順を診断側で再現しながら
「ロールアウトの中で実際に何が起きたか」を観測する(`_rollout_and_eval` 自体は最終スコアしか
返さないため、これは診断専用の複製ループ)。

対象の決定点は `hammer_diagnostic.evaluate_decision` が qualifies=True(cond2〜5成立)と
判定し、かつ実際にはハンマーを使わなかった("逃した")もの(run3 の "missed" と同じ)。
各決定点について、`num_determinizations`(既定8)個の世界それぞれで:

  (a) ハンマーを first move としたときのロールアウト
  (b) 実際に選んだ手を first move としたときのロールアウト(比較対照)

の両方を観測し、
  1. 自分のターンの残りで ATTACK 選択肢が選ばれたか(Powerful Handを撃ったか)
  2. 撃った場合、相手のバトルポケモンが実際にKOされたか(State上でserial追跡)
  3. 撃たなかった場合、代わりに何が選ばれ続けたか(OptionTypeの列、先頭5手)
  4. 8世界のうち何世界で1が成立したか

を記録する。

`sample_submission/` 配下は一切変更しない(import して呼ぶだけ)。`hammer_diagnostic.py`
(このファイルと同じ `kaggle_replays/rl/` 配下、`sample_submission/` の外)から
`evaluate_decision` / `_resolve_one_action` / `_search_begin_from_hidden_state` 等を
再利用する(重複を避ける、run3 と同じ土台)。

使用例:
    python kaggle_replays/rl/hammer_rollout_diagnostic.py --games 60 \
        --opponents crustle,shirona_garchomp_ex --workers 0 \
        --out kaggle_replays/rl/hammer_diagnostic_out/run4_rollout.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # kaggle_replays/rl
_ROOT = _HERE.parents[1]                          # repo root
_LEAGUE_DIR = _ROOT / "league"
_SUB_DIR = _ROOT / "sample_submission"
for _p in (str(_HERE), str(_LEAGUE_DIR), str(_ROOT), str(_SUB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hammer_diagnostic as hd  # noqa: E402  (run3 の土台を再利用。sample_submission/ の外)
import run_league  # noqa: E402
from cg.api import OptionType, SelectType, search_step  # noqa: E402  (観測のみ、意思決定には使わない)
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402  (観測のみ)
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402  (観測のみ、agent本体は変更しない)
from ptcg_ai.search import pipeline as pipeline_search  # noqa: E402  (_greedy_selection をそのまま呼ぶ)

CONTENDER_CONFIG = hd.CONTENDER_CONFIG  # run3と同条件(既定 abl_5_full、PTCG_AI_ML_CONFIGで上書き可)
OPPONENT_CONFIG = hd.OPPONENT_CONFIG
OPPONENT_SPECS = hd.OPPONENT_SPECS
ALAKAZAM_DECK_PATH = hd.ALAKAZAM_DECK_PATH

MAX_TRACK_OPTIONS = 5  # 「撃たなかった場合に選ばれた手」を何手ぶん記録するか


# ---------------------------------------------------------------------------
# ロールアウト観測(pipeline._rollout_and_eval と同じ停止条件を診断側で複製)
# ---------------------------------------------------------------------------


def _pokemon_alive(state, opp_idx: int, serial: int) -> bool:
    """opp_idx側のアクティブ+ベンチに serial のポケモンが hp>0 で存在するか。"""
    player = state.players[opp_idx]
    for p in (player.active or []):
        if p is not None and p.serial == serial and p.hp > 0:
            return True
    for p in (player.bench or []):
        if p is not None and p.serial == serial and p.hp > 0:
            return True
    return False


def _capture_attack_snapshot(state, me: int, opp_idx: int, attack_id: int, defender_serial: int) -> dict | None:
    """ATTACK 選択肢が選ばれた**その瞬間**(まだ search_step で解決する前)の、攻撃側手札枚数と
    防御側HPから、`attack_features.resolve_damage`(evaluate_decision の cond4 と全く同じ計算)で
    「この時点の手札枚数で本当に倒せるか」を再計算する。

    背景: Powerful Hand は可変ダメージ技で、ダメージが攻撃側の手札枚数に比例する
    (`attack_features._estimate_variable_damage`: ダメージ = 手札1枚あたりのカウンター数 x 10 x
    手札枚数)。Enhanced Hammer は Item なのでプレイすると手札が1枚減る。cond4(hammer_diagnostic.
    evaluate_decision)の `hypothetical_damage` はハンマーを**プレイする前**の手札枚数で計算した
    予測であり、実際にハンマーをプレイしてから攻撃するまでの間に手札が変化すれば、その予測は
    もう成立しない可能性がある。ここでは事後にそれを検証する(判断には使わない、観測のみ)。
    """
    try:
        player = state.players[me]
        active = player.active[0] if (player.active and player.active[0] is not None) else None
        opponent = state.players[opp_idx]
        defender = None
        is_benched = False
        for p in (opponent.active or []):
            if p is not None and p.serial == defender_serial:
                defender = p
        if defender is None:
            for p in (opponent.bench or []):
                if p is not None and p.serial == defender_serial:
                    defender = p
                    is_benched = True
        if active is None or defender is None:
            return None
        attack = hd.card_cache.get_attack(attack_id)
        defender_card = hd.card_cache.get_card(defender.id)
        hand_count = player.handCount
        defender_side = [p for p in (opponent.active or []) if p is not None] + list(opponent.bench)
        damage_is_effect = hd.attack_features.damage_is_effect_based(attack)
        predicted = hd.attack_features.resolve_damage(
            attack, active, defender_card.weakness, defender_card.resistance, hand_count,
            defender=defender, defender_side_pokemon=defender_side,
            defender_is_benched=is_benched, damage_is_effect=damage_is_effect,
        )
        return {
            "attack_id": attack_id,
            "attacker_hand_count": hand_count,
            "defender_hp": defender.hp,
            "predicted_damage": predicted,
            "would_be_lethal": predicted >= defender.hp,
        }
    except Exception:  # noqa: BLE001
        return None


def _instrumented_rollout(node, me: int, opp_idx: int, defender_serial: int, config: dict, policy_model) -> dict:
    """`pipeline._rollout_and_eval` と全く同じ停止条件(opponent_depth / max_rollout_steps /
    決着 / 選択肢無し)で `_greedy_selection`(pipeline.py 本体をそのまま呼ぶ、複製しない)を
    使ってロールアウトを進めながら、以下を観測する:

      - attacked: 自分のターンの残り(相手ターンに切り替わる前)に ATTACK 型の選択肢が
        選ばれたか
      - ko: ロールアウト中のどこかの時点で defender_serial のポケモンが opp_idx 側の
        アクティブ/ベンチから居なくなった(hp<=0まで削られ場を離れた)か
      - option_type_seq: 自分のターンの残りで実際に選ばれた OptionType の列
        (最大 MAX_TRACK_OPTIONS 件。attacked=False のとき「代わりに何を選んだか」に使う)

    `_rollout_and_eval` 自体は最終スコアしか返さない設計のため、これは判断には一切
    介入しない観測専用の複製ループ(pipeline.py は変更しない)。中間ノードは
    `_rollout_and_eval` 本体と同じく明示的に release しない(`search_end()` でまとめて
    解放するチーム慣例、pipeline.py の実装と同じ)。
    """
    opponent_depth = max(1, int(config.get("opponent_depth", 1)))
    max_steps = int(config.get("max_rollout_steps", 40))
    prev_actor = me
    opp_turns = 0
    attacked = False
    ko = False
    option_type_seq: list[str] = []
    steps_taken = 0
    attack_snapshot: dict | None = None

    for _ in range(max_steps):
        obs = node.observation
        state = obs.current
        if state is None:
            break
        if state.result != -1:
            if not _pokemon_alive(state, opp_idx, defender_serial):
                ko = True
            break

        actor = state.yourIndex
        if prev_actor == me and actor != me:
            opp_turns += 1
        if actor == me and prev_actor != me and opp_turns >= opponent_depth:
            break

        if obs.select is None or not obs.select.option:
            break

        selection = pipeline_search._greedy_selection(policy_model, obs)
        if not selection:
            break

        if opp_turns == 0:
            option = obs.select.option[selection[0]]
            if len(option_type_seq) < MAX_TRACK_OPTIONS:
                option_type_seq.append(OptionType(option.type).name)
            if option.type == OptionType.ATTACK:
                attacked = True
                if attack_snapshot is None and option.attackId is not None:
                    attack_snapshot = _capture_attack_snapshot(
                        state, me, opp_idx, option.attackId, defender_serial,
                    )

        try:
            node = search_step(node.searchId, selection)
        except ValueError:
            break
        prev_actor = actor
        steps_taken += 1

        ns = node.observation.current
        if ns is not None and not _pokemon_alive(ns, opp_idx, defender_serial):
            ko = True
            break

    return {
        "attacked": attacked,
        "ko": ko,
        "option_type_seq": option_type_seq,
        "steps": steps_taken,
        "attack_snapshot": attack_snapshot,
    }


def compute_rollout_comparison(
    obs, info: dict, action: list[int], contender_cfg: dict, pipeline_cfg_merged: dict,
) -> dict:
    """1つの「逃した」決定点について、num_determinizations 個の世界それぞれで
    (a) ハンマー first move (b) 実際に選んだ手 first move のロールアウトを観測し、
    世界ごとの結果リストを返す。

    hidden_state の組み立ては `ml_policy_agent._try_pipeline` の estimated 分岐
    (`search_adapter.to_search_begin_kwargs`)と同じ factory を使い、pipeline.search() の
    Step3 と同様に世界の数だけ ``factory()`` を呼び直す(呼ぶたびに別の決定化になる、
    `search_adapter.to_search_begin_kwargs` の docstring 参照)。
    """
    result: dict = {"ok": False, "worlds_hammer": [], "worlds_actual": []}
    state = obs.current
    if state is None or obs.select is None:
        return result
    me = state.yourIndex
    opp_idx = 1 - me

    opponent = state.players[opp_idx]
    if not opponent.active or opponent.active[0] is None:
        return result
    defender_serial = opponent.active[0].serial

    hammer_idx = info["hammer_option_indices"][0]
    removed_energy_card = info["removed_energy_card"]
    target_serial = removed_energy_card.serial if removed_energy_card is not None else None

    policy_model = ml_policy_agent._get_model(contender_cfg)
    num_worlds = max(1, int(pipeline_cfg_merged.get("num_determinizations", 8)))

    factory = lambda: search_adapter.to_search_begin_kwargs(  # noqa: E731
        match_context.get_own_state(me), match_context.get_opponent_state(me), obs,
    )

    try:
        for _ in range(num_worlds):
            try:
                hidden_state = factory()
            except Exception:  # noqa: BLE001
                continue
            if hidden_state is None:
                continue
            try:
                root = hd._search_begin_from_hidden_state(obs, hidden_state)
            except Exception:  # noqa: BLE001
                continue
            try:
                child_hammer = hd._resolve_one_action(root, [hammer_idx], target_serial, obs)
                if child_hammer is not None:
                    try:
                        r = _instrumented_rollout(
                            child_hammer, me, opp_idx, defender_serial, pipeline_cfg_merged, policy_model,
                        )
                        result["worlds_hammer"].append(r)
                    finally:
                        try:
                            hd.search_release(child_hammer.searchId)
                        except Exception:  # noqa: BLE001
                            pass

                child_actual = hd._resolve_one_action(root, list(action), None, obs)
                if child_actual is not None:
                    try:
                        r2 = _instrumented_rollout(
                            child_actual, me, opp_idx, defender_serial, pipeline_cfg_merged, policy_model,
                        )
                        result["worlds_actual"].append(r2)
                    finally:
                        try:
                            hd.search_release(child_actual.searchId)
                        except Exception:  # noqa: BLE001
                            pass
            finally:
                try:
                    hd.search_release(root.searchId)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        try:
            hd.search_end()
        except Exception:  # noqa: BLE001
            pass

    result["ok"] = bool(result["worlds_hammer"]) or bool(result["worlds_actual"])
    return result


# ---------------------------------------------------------------------------
# 観測用ラッパー(hammer_diagnostic.instrument と同じ骨格、ロールアウト観測だけに絞る)
# ---------------------------------------------------------------------------


class DiagState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counts: Counter = Counter()
        self.missed_records: list[dict] = []


def instrument(agent_fn, state: DiagState, contender_cfg: dict, pipeline_cfg_merged: dict):
    """本番の agent(obs) をそのまま呼び、「逃した」決定点だけロールアウト比較を追加観測する。
    意思決定には一切介入しない(action はそのまま返す)。
    """

    def wrapped(obs) -> list[int]:
        info = None
        if obs.select is not None and obs.current is not None and obs.select.type == SelectType.MAIN:
            info = hd.evaluate_decision(obs)

        action = agent_fn(obs)

        if info is not None and info.get("qualifies"):
            state.counts["qualifying"] += 1
            used_hammer = bool(set(info["hammer_option_indices"]) & set(action))
            if used_hammer:
                state.counts["used_hammer"] += 1
            else:
                state.counts["missed_hammer"] += 1
                rollout = compute_rollout_comparison(obs, info, action, contender_cfg, pipeline_cfg_merged)
                if rollout["ok"]:
                    state.missed_records.append({
                        "defender_name": info["defender_name"],
                        "chosen_desc": hd.describe_action(obs, action),
                        "rollout": rollout,
                    })
                else:
                    state.counts["rollout_unobservable"] += 1

        return action

    return wrapped


# ---------------------------------------------------------------------------
# 対戦ドライバ(hammer_diagnostic.py と同じ構造)
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _build_contender_agent() -> tuple:
    diag_state = DiagState()
    contender_cfg = hd.load_ptcg_config(CONTENDER_CONFIG)
    pipeline_cfg_merged = {**pipeline_search.DEFAULTS, **(contender_cfg.get("pipeline") or {})}
    raw_agent = run_league.build_agent("ml_policy", None, CONTENDER_CONFIG)
    wrapped = instrument(raw_agent, diag_state, contender_cfg, pipeline_cfg_merged)
    return wrapped, diag_state


def _worker_init(opponent_names: list[str]) -> None:
    os.chdir(_SUB_DIR)
    agent_a, diag_state = _build_contender_agent()
    _WORKER_STATE["agent_a"] = agent_a
    _WORKER_STATE["diag_state"] = diag_state
    _WORKER_STATE["deck_a"] = run_league.read_deck_csv_file(str(ALAKAZAM_DECK_PATH))
    opponents = {}
    for name in opponent_names:
        deck_path, weights_path = OPPONENT_SPECS[name]
        agent_b = run_league.build_agent("ml_policy", str(weights_path), OPPONENT_CONFIG)
        deck_b = run_league.read_deck_csv_file(str(deck_path))
        opponents[name] = (agent_b, deck_b)
    _WORKER_STATE["opponents"] = opponents


def _play_one(task: tuple[str, int, int]) -> dict:
    opponent_name, game_index, seed = task
    s = _WORKER_STATE
    diag_state: DiagState = s["diag_state"]
    diag_state.reset()

    agent_b, deck_b = s["opponents"][opponent_name]
    a_is_player0 = (game_index % 2 == 0)
    if a_is_player0:
        agent0, agent1 = s["agent_a"], agent_b
        deck0, deck1 = s["deck_a"], deck_b
    else:
        agent0, agent1 = agent_b, s["agent_a"]
        deck0, deck1 = deck_b, s["deck_a"]

    result = run_league.play_match(agent0, agent1, deck0, deck1, seed=seed)

    return {
        "opponent": opponent_name,
        "game_index": game_index,
        "seed": seed,
        "error": result.error,
        "counts": dict(diag_state.counts),
        "missed_records": diag_state.missed_records,
    }


def run_diagnostic(
    games_per_opponent: int,
    opponent_names: list[str],
    seed_start: int,
    workers: int,
    log=lambda msg: print(msg, file=sys.stderr),
) -> dict:
    for name in opponent_names:
        if name not in OPPONENT_SPECS:
            raise ValueError(f"unknown opponent: {name!r} (choices: {sorted(OPPONENT_SPECS)})")

    tasks: list[tuple[str, int, int]] = []
    seed = seed_start
    for name in opponent_names:
        for gi in range(games_per_opponent):
            tasks.append((name, gi, seed))
            seed += 1

    n_workers = run_league.resolve_workers(workers)
    t0 = time.time()
    all_results: list[dict] = []

    if n_workers <= 1:
        os.chdir(_SUB_DIR)
        agent_a, diag_state = _build_contender_agent()
        deck_a = run_league.read_deck_csv_file(str(ALAKAZAM_DECK_PATH))
        opponents = {}
        for name in opponent_names:
            deck_path, weights_path = OPPONENT_SPECS[name]
            agent_b = run_league.build_agent("ml_policy", str(weights_path), OPPONENT_CONFIG)
            deck_b = run_league.read_deck_csv_file(str(deck_path))
            opponents[name] = (agent_b, deck_b)
        _WORKER_STATE["agent_a"] = agent_a
        _WORKER_STATE["diag_state"] = diag_state
        _WORKER_STATE["deck_a"] = deck_a
        _WORKER_STATE["opponents"] = opponents
        for i, task in enumerate(tasks, 1):
            all_results.append(_play_one(task))
            if i % 10 == 0 or i == len(tasks):
                log(f"[progress] {i}/{len(tasks)} games done ({time.time() - t0:.1f}s)")
    else:
        log(f"[info] running {len(tasks)} games across {n_workers} worker processes")
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_worker_init, initargs=(opponent_names,)
        ) as executor:
            futures = [executor.submit(_play_one, task) for task in tasks]
            done = 0
            for fut in as_completed(futures):
                all_results.append(fut.result())
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    log(f"[progress] {done}/{len(tasks)} games done ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0

    total_counts: Counter = Counter()
    all_missed: list[dict] = []
    errors = 0
    valid_games = 0
    for rec in all_results:
        if rec["error"] is not None:
            errors += 1
            continue
        valid_games += 1
        total_counts += Counter(rec["counts"])
        for r in rec["missed_records"]:
            r2 = dict(r)
            r2["opponent"] = rec["opponent"]
            r2["game_index"] = rec["game_index"]
            all_missed.append(r2)

    return {
        "contender_config": CONTENDER_CONFIG,
        "games_requested_per_opponent": games_per_opponent,
        "opponents": opponent_names,
        "games_total": len(tasks),
        "valid_games": valid_games,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "workers": n_workers,
        "total_counts": dict(total_counts),
        "missed_records": all_missed,
        "q3_rollout_analysis": analyze_rollouts(all_missed),
    }


# ---------------------------------------------------------------------------
# 集計 + レポート
# ---------------------------------------------------------------------------


def _rate(worlds: list[dict], key: str) -> float | None:
    if not worlds:
        return None
    return sum(1 for w in worlds if w[key]) / len(worlds)


def analyze_rollouts(missed_records: list[dict]) -> dict:
    """(a)ハンマー first move / (b)実際手 first move、それぞれについて
    「攻撃した世界の割合」「攻撃してKOに至った世界の割合」の decision 単位平均、
    および「攻撃しなかった世界」で選ばれた最初の手(OptionType)の内訳を集計する。
    """
    out: dict = {"n_missed_with_rollout": len(missed_records)}

    for branch, worlds_key in (("hammer", "worlds_hammer"), ("actual", "worlds_actual")):
        per_decision_attack_rates = []
        per_decision_ko_rates = []
        not_attacked_first_types: Counter = Counter()
        pooled_worlds = 0
        pooled_attacked = 0
        pooled_ko = 0
        # 攻撃はしたが手札枚数の変化により(cond4予測時点と比べ)その時点では既に非lethalに
        # なっていた世界を数える(hand_count可変ダメージの確認用、判断には使わない観測)。
        attacked_worlds_with_snapshot = 0
        attacked_not_lethal_at_attack_time = 0
        hand_count_deltas: list[int] = []
        for rec in missed_records:
            worlds = rec["rollout"].get(worlds_key) or []
            if not worlds:
                continue
            ar = _rate(worlds, "attacked")
            kr = _rate(worlds, "ko")
            if ar is not None:
                per_decision_attack_rates.append(ar)
            if kr is not None:
                per_decision_ko_rates.append(kr)
            pooled_worlds += len(worlds)
            pooled_attacked += sum(1 for w in worlds if w["attacked"])
            pooled_ko += sum(1 for w in worlds if w["ko"])
            for w in worlds:
                if not w["attacked"]:
                    first = w["option_type_seq"][0] if w["option_type_seq"] else "(no_option)"
                    not_attacked_first_types[first] += 1
                else:
                    snap = w.get("attack_snapshot")
                    if snap is not None:
                        attacked_worlds_with_snapshot += 1
                        if not snap["would_be_lethal"]:
                            attacked_not_lethal_at_attack_time += 1

        n = len(per_decision_attack_rates)
        out[branch] = {
            "n_decisions_with_worlds": n,
            "avg_attack_rate_per_decision": (
                sum(per_decision_attack_rates) / n if n else None
            ),
            "avg_ko_rate_per_decision": (
                sum(per_decision_ko_rates) / len(per_decision_ko_rates) if per_decision_ko_rates else None
            ),
            "pooled_worlds": pooled_worlds,
            "pooled_attack_rate": (pooled_attacked / pooled_worlds if pooled_worlds else None),
            "pooled_ko_rate": (pooled_ko / pooled_worlds if pooled_worlds else None),
            "not_attacked_first_option_type_counts": dict(not_attacked_first_types),
            "attacked_worlds_with_snapshot": attacked_worlds_with_snapshot,
            "attacked_not_lethal_at_attack_time": attacked_not_lethal_at_attack_time,
            "attacked_not_lethal_at_attack_time_pct": (
                attacked_not_lethal_at_attack_time / attacked_worlds_with_snapshot * 100.0
                if attacked_worlds_with_snapshot else None
            ),
        }

    return out


def print_report(report: dict, log=print) -> None:
    log(f"=== hammer rollout diagnostic (run4): alakazam(ml_policy/{report['contender_config']}) "
        f"vs {', '.join(report['opponents'])} ===")
    log(
        f"games: {report['valid_games']}/{report['games_total']} valid "
        f"({report['errors']} errors), elapsed {report['elapsed_seconds']:.1f}s, workers={report['workers']}"
    )
    tc = report["total_counts"]
    log("")
    log(f"qualifying = {tc.get('qualifying', 0)} 件  used_hammer = {tc.get('used_hammer', 0)} 件  "
        f"missed_hammer = {tc.get('missed_hammer', 0)} 件"
        + (f"  (rollout観測不能 {tc.get('rollout_unobservable', 0)} 件を除外)"
           if tc.get("rollout_unobservable") else ""))

    q3 = report["q3_rollout_analysis"]
    log("")
    log(f"=== 問い3(run4): ロールアウトの中身を観測する(逃した局面 {q3['n_missed_with_rollout']} 件) ===")
    for branch, label in (("hammer", "(a) ハンマーを first move にしたロールアウト"),
                          ("actual", "(b) 実際に選んだ手を first move にしたロールアウト(比較)")):
        b = q3[branch]
        log(f"  {label}")
        log(f"    観測できた決定点            {b['n_decisions_with_worlds']} 件 "
            f"(のべ世界数 {b['pooled_worlds']})")
        ar = b["avg_attack_rate_per_decision"]
        kr = b["avg_ko_rate_per_decision"]
        par = b["pooled_attack_rate"]
        pkr = b["pooled_ko_rate"]
        log(f"    ロールアウトで攻撃した世界の割合    決定点平均 "
            f"{ar * 100:.1f}%" if ar is not None else "    ロールアウトで攻撃した世界の割合    (計算不能)")
        if par is not None:
            log(f"                                     (プール平均 {par * 100:.1f}%、n={b['pooled_worlds']})")
        if kr is not None:
            log(f"    攻撃してKOまで至った世界の割合      決定点平均 {kr * 100:.1f}%")
        if pkr is not None:
            log(f"                                     (プール平均 {pkr * 100:.1f}%)")
        n_snap = b["attacked_worlds_with_snapshot"]
        pct_not_lethal = b["attacked_not_lethal_at_attack_time_pct"]
        if n_snap:
            log(f"    攻撃した世界のうち、攻撃時点で既に非lethalだった      "
                f"{b['attacked_not_lethal_at_attack_time']}/{n_snap} 件（{pct_not_lethal:.1f}%）"
                f"  (手札枚数依存ダメージの再計算。cond4予測=ハンマー着手前の手札枚数)")
        not_atk = b["not_attacked_first_option_type_counts"]
        total_not_atk = sum(not_atk.values())
        if total_not_atk:
            log(f"    攻撃しなかった世界で最初に選ばれた手の内訳(全 {total_not_atk} 世界):")
            for opt_type, cnt in sorted(not_atk.items(), key=lambda kv: -kv[1]):
                log(f"      {opt_type:12s} {cnt:4d} 件（{cnt / total_not_atk * 100:.1f}%）")
        log("")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="run4: 「逃した」ハンマー決定点でロールアウトの中身(攻撃したか/KOしたか/"
        "攻撃しなかった場合に何を選んだか)を観測する。pipeline.py は変更せず import して呼ぶだけ。"
    )
    ap.add_argument("--games", type=int, default=60, help="相手アーキタイプ1体あたりの試合数(default: 60)")
    ap.add_argument(
        "--opponents", default="crustle,shirona_garchomp_ex",
        help=f"カンマ区切りの相手アーキタイプ名(選択肢: {', '.join(sorted(OPPONENT_SPECS))})",
    )
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="0以下で自動(CPU数-1)")
    ap.add_argument("--out", default=None, help="結果JSONの出力先(default: 出力しない)")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    opponent_names = [x.strip() for x in args.opponents.split(",") if x.strip()]
    report = run_diagnostic(
        games_per_opponent=args.games,
        opponent_names=opponent_names,
        seed_start=args.seed_start,
        workers=args.workers,
    )
    print_report(report)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nresult written to: {out_path}")


if __name__ == "__main__":
    main()
