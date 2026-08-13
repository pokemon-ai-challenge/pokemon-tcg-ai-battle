"""design.md(ogerpon_single_prize_mlp_design.md)§9(Phase2 item2): カプ・ブルル中継戦略の
反実仮想(counterfactual)paired rollout収集。

同じ公開盤面(Strategy Window Builderが発火した局面)から `EX_TEMPO` /
`SINGLE_PRIZE_ROTATION` を強制分岐させ、同じ非公開情報仮説・相手Policyでrolloutして
勝敗等を記録する。将来のQ-critic(Phase3)学習の教師データを作るためのオフライン収集
スクリプトであり、意思決定には一切使わない(既存の本番エージェント・configには触れない)。

## 手順(design.md §9.3 に対応)

1. 通常のself-play(learner=既存Policy+lethal / opponent=archetype別Policy)を1本進める。
2. Strategy Window Builder(``ogerpon_strategy.detect_trigger``)が発火した局面ごとに、
   **その場で即座に**(公開状態を後回しで貯めず)反実仮想rolloutを実行する。理由: 非公開
   情報の推定(``match_context``)はターンを追うごとに更新される可変状態なので、発火時点の
   knowledgeを使うには「発火した瞬間に処理する」必要がある(後からまとめて処理すると、
   同じゲームの後の方の局面で得た情報が前の局面のhidden state生成に混入してしまう)。
3. hidden stateは ``ptcg_ai.hidden_information.search_adapter.to_search_begin_kwargs()``
   (``match_context`` が保持する ``OwnHiddenState``/``OpponentHiddenState`` の事後分布からの
   サンプリング)で作る。これは本番の ``ml_policy_agent._try_pipeline``(``abl_5_full`` の
   ``hidden_state_source: "estimated"``)が実際に使っているのと同じ仕組みであり、確定リーサル
   探索専用のダミースタブ(``build_dummy_search_state``、相手の山札・手札を汎用エネルギーで
   埋める)は**使わない**(相手の非公開領域がデッキ分布と無関係になり、数百手規模のrolloutでは
   勝敗ラベルの信頼性を大きく損なうため)。
4. 決定化ごとに、Optionごとに ``search_begin`` でrootを作り直す(design.mdの推奨どおり、
   1つのrootをsearch_stepで分岐させるのではなく、同じhidden state payloadからOptionごとに
   新しいrootを作る。速度よりペアの公平性を優先する)。同じdeterminizationは必ず両Optionで
   共有する。
5. ``first_action`` を強制した後は、学習側は既存Policyのgreedy選択で継続する
   (``pipeline.py`` の内部rolloutと同じ簡略化。lethal探索・PIMC再帰は行わない)。
   ``SINGLE_PRIZE_ROTATION`` 側は ``ogerpon_option_state`` の固定Option Controllerに
   実際に行動選択を拘束させる(BUILD/READY中は低位candidate resolver、ACTIVE中は原則攻撃・
   緊急時以外の交代拒否、COMPLETE/ABORT後は通常Policyへ戻す)。相手側は実際に学習済みの
   archetype別Policyで進める。
6. 終局・最大ターン・最大stepまでrolloutし、勝敗・(rootからの相対)ターン数・実KO数・
   ブルル攻撃数・取得サイド・ループ完遂(気絶までACTIVE継続+次アタッカー接続まで確認)・
   abort理由を記録する。
7. search nodeは経路上のものを含めて必ず ``search_release`` し、1状態の処理が終わるごとに
   ``search_end`` する。
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
from ogerpon_rollout_common import detect_vanished, is_emergency_retreat_allowed  # noqa: E402

SCHEMA_VERSION = 1
MAX_OUTER_STEPS = 3000       # 外側(state収集用)の1試合あたり上限
MAX_ROLLOUT_STEPS = 400      # 反実仮想rolloutの1本あたり上限
EX_TEMPO = "EX_TEMPO"
SINGLE_PRIZE_ROTATION = "SINGLE_PRIZE_ROTATION"
_W: dict = {}


def _init(weights_path, ml_config_name, workdir, opponents):
    import os
    os.environ["PTCG_AI_ML_CONFIG"] = ml_config_name
    os.chdir(workdir)
    from ptcg_ai.core.config import load_config
    from ptcg_ai.hidden_information import match_context, search_adapter
    from ptcg_ai.learning import ogerpon_strategy_encoder
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.ml_policy import ml_policy_agent, ogerpon_option_state, ogerpon_planner, ogerpon_strategy

    config = dict(load_config(ml_config_name))
    config["policy_weights_path"] = weights_path
    _W["config"] = config
    _W["agent"] = ml_policy_agent
    _W["P"] = ogerpon_planner
    _W["OS"] = ogerpon_option_state
    _W["STRAT"] = ogerpon_strategy
    _W["ENC"] = ogerpon_strategy_encoder
    _W["match_context"] = match_context
    _W["search_adapter"] = search_adapter
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


def determinization_seed(match_seed: int, turn: int, state_id: str, determinization_id: int) -> int:
    """決定化のRNG seedをプロセス非依存に導出する(``hash()`` は ``PYTHONHASHSEED`` に依存し
    プロセスをまたいで再現しないため使わない)。"""
    payload = f"{match_seed}:{turn}:{state_id}:{determinization_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def compute_loop_complete(option_mode, OS, bulu_attack_count: int, saw_next_attacker_ready: bool) -> bool:
    """design.md §8.2のloop_complete定義: 気絶(COMPLETE)まで進み、かつ実際に1回以上攻撃し、
    かつ気絶後に攻撃可能な次のオーガポンexへ接続できた場合だけTrue(外部レビュー指摘:
    以前はCOMPLETEだけで判定しており、ブルルが一度も攻撃せずに場を離れたケースを
    誤って完遂扱いする恐れがあった)。``eval_bulu_loop.py`` の ``strategy_loop_complete``
    と同じ3条件をここでも要求し、定義を揃える。
    """
    return bool(option_mode == OS.COMPLETE and bulu_attack_count > 0 and saw_next_attacker_ready)


# ---------------------------------------------------------------------------
# hidden state(design.md §9.3 item3): match_contextの推定を使う。ダミースタブは使わない。
# ---------------------------------------------------------------------------

def build_estimated_hidden_state(obs, learner_index: int, rng) -> dict:
    """``search_adapter.to_search_begin_kwargs`` を使い、``match_context`` が保持する
    推定(``OwnHiddenState``/``OpponentHiddenState``)からhidden stateをサンプルする。

    本番の ``ml_policy_agent._try_pipeline``(``hidden_state_source: "estimated"``)と同じ
    経路。生成できない場合は例外を送出する(呼び出し側が明示的にスキップ/エラー計上する。
    ダミーへの暗黙フォールバックはしない)。
    """
    match_context = _W["match_context"]
    search_adapter = _W["search_adapter"]
    own_state = match_context.get_own_state(learner_index)
    opponent_state = match_context.get_opponent_state(learner_index)
    return search_adapter.to_search_begin_kwargs(own_state, opponent_state, obs, rng)


# ---------------------------------------------------------------------------
# SINGLE_PRIZE_ROTATION rollout用の固定Option Controller(design.md §5.4 / §11.3)
# ---------------------------------------------------------------------------

def _single_prize_low_level_action(obs, P, OS, config, deck_ids, option_mode):
    """SINGLE_PRIZE_ROTATION rolloutの各局面で、Option Controllerが行動を拘束するか判定する。

    ``None`` を返した場合だけ、呼び出し側は素のPolicy greedyへフォールバックする
    (BUILD/READY中の「攻撃可能な手が無いなら通常Policyに任せる」局面など)。

    - COMPLETE/ABORT: 常に ``None``(通常Policyへ完全に戻す。低位Controllerは一切介入しない)。
    - ACTIVE: 気絶するまで攻撃を優先する。ATTACKが選べるなら必ずそれを選ぶ。にげる/交代は
      ``is_emergency_retreat_allowed`` が緊急と判定した場合以外は選ばせない
      (候補から除外し、除外後に残る最初の合法手を返す。他に選べる手が無ければNone)。
    - BUILD/READY/IDLE: 既存Plannerのcandidate resolverを低位Controllerとして使う
      (手貼り: ``select_strategic_attach_candidate``、昇格/交代: ``score_adjustments``)。
    """
    from cg.api import OptionType, SelectType

    if option_mode in (OS.COMPLETE, OS.ABORT):
        return None

    select = obs.select
    if select.maxCount != 1:
        return None
    state = obs.current
    me = state.yourIndex
    stype = int(select.type)
    mine = state.players[me]
    active = next((s for s in (mine.active or []) if s is not None), None)

    if option_mode == OS.ACTIVE:
        if stype == int(SelectType.MAIN):
            attack_indices = [i for i, o in enumerate(select.option)
                              if int(getattr(o, "type", -1)) == int(OptionType.ATTACK)]
            if attack_indices:
                return [attack_indices[0]]
            retreat_indices = {i for i, o in enumerate(select.option)
                              if int(getattr(o, "type", -1)) == int(OptionType.RETREAT)}
            if retreat_indices:
                opp = state.players[1 - me]
                opp_active = next((s for s in (opp.active or []) if s is not None), None)
                if not is_emergency_retreat_allowed(active, opp_active, config, P):
                    non_retreat = [i for i in range(len(select.option)) if i not in retreat_indices]
                    return [non_retreat[0]] if non_retreat else None
            return None
        if stype == int(SelectType.CARD):
            # ACTIVEのブルルを自発的に手放すTO_ACTIVE/SWITCHは、緊急以外は選ばせない。
            opp = state.players[1 - me]
            opp_active = next((s for s in (opp.active or []) if s is not None), None)
            if is_emergency_retreat_allowed(active, opp_active, config, P):
                return None  # 緊急時は通常Policyの判断に委ねる
            return None  # 緊急でなければCARD選択自体がこの局面では想定外(安全側でPolicyに委ねる)
        return None

    # BUILD/READY/IDLE
    if stype == int(SelectType.MAIN):
        forced_cfg = _forced_planner_config(config)
        idx = P.select_strategic_attach_candidate(obs, me, forced_cfg, deck_ids)
        if idx is None:
            return None
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


def _learner_rollout_action(obs, option_name, option_mode, deck_ids):
    """rollout中の学習側の一手。EX_TEMPOは常に素のPolicy greedy。SINGLE_PRIZE_ROTATIONは
    ``_single_prize_low_level_action``(Option Controller、現在のphaseに応じて行動を拘束する)を
    先に試し、Noneのときだけ素のPolicy greedyへフォールバックする。
    """
    P, OS = _W["P"], _W["OS"]
    policy_model = _W["learner_policy"]
    select = obs.select
    if select is None or not select.option:
        return []
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))

    if option_name == SINGLE_PRIZE_ROTATION and select.maxCount == 1:
        forced = _single_prize_low_level_action(obs, P, OS, _W["config"], deck_ids, option_mode)
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


# ---------------------------------------------------------------------------
# 1Option・1決定化のrollout
# ---------------------------------------------------------------------------

def _rollout_option(root_obs, hidden_state, candidate, learner_index, opp_pm, deck_ids, option_cfg):
    """design.md §9.5のoutcomeスキーマを返す。

    - ``terminal_turns``: rootからの相対ターン数(``root_obs`` 時点の ``turn`` との差分)。
    - ``opponent_ko_count``: サイド枚数の差ではなく、実際に相手の攻撃直後に自分の個体が
      場から消えた回数(KOイベント数)。
    - ``loop_complete``: Option Stateが ``COMPLETE`` に達し、かつ実際に1回以上攻撃し、
      かつ気絶後に攻撃可能な次のオーガポンexへ接続できた場合だけ ``True``。
    """
    from cg import api as cg_api
    from cg.api import OptionType

    P, OS = _W["P"], _W["OS"]
    outcome = {
        "win": None, "terminal_turns": None, "opponent_ko_count": 0,
        "bulu_attack_count": 0, "bulu_prizes_taken": 0, "loop_complete": False,
        "abort_reason": None, "error": None,
    }
    root_turn = getattr(root_obs.current, "turn", None)
    root = None
    try:
        root = cg_api.search_begin(
            root_obs, hidden_state["your_deck"], hidden_state["your_prize"],
            hidden_state["opponent_deck"], hidden_state["opponent_prize"],
            hidden_state["opponent_hand"], hidden_state["opponent_active"])
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = f"search_begin: {exc!r}"
        return outcome

    node = None
    try:
        try:
            node = cg_api.search_step(root.searchId, candidate.first_action)
        except ValueError as exc:
            outcome["error"] = f"illegal_first_action: {exc!r}"
            return outcome

        option_state = OS.OgerponOptionState()
        prev_opp_prize = None
        pending_bulu_ko_check = None   # ブルル攻撃直後、相手個体の消失を確認する
        pending_opp_ko_check = None    # 相手の攻撃直後、自分個体の消失を確認する
        saw_next_attacker_ready = False

        for _ in range(MAX_ROLLOUT_STEPS):
            obs = node.observation
            cur = obs.current
            if cur is None:
                outcome["error"] = "current None"
                break
            if cur.result != -1:
                outcome["win"] = 1 if cur.result == learner_index else 0
                outcome["terminal_turns"] = (
                    (cur.turn - root_turn) if (cur.turn is not None and root_turn is not None) else None)
                break

            mine_now = cur.players[learner_index]
            opp_now = cur.players[1 - learner_index]

            if pending_bulu_ko_check is not None:
                # 取得サイド枚数はKOイベント数と一致しない(ex=2枚/非ex=1枚)ため、
                # 実際のサイド差分(prize value)で数える。opponent_ko_countとは別の指標。
                taken = max(0, (prev_opp_prize or 0) - len(opp_now.prize or []))
                if taken > 0:
                    outcome["bulu_prizes_taken"] += taken
                pending_bulu_ko_check = None
            if pending_opp_ko_check is not None and detect_vanished(
                    pending_opp_ko_check["serial"], mine_now):
                outcome["opponent_ko_count"] += 1
                pending_opp_ko_check = None
            prev_opp_prize = len(opp_now.prize or [])

            actor = cur.yourIndex
            if actor == learner_index and obs.select is not None:
                if candidate.option_name == SINGLE_PRIZE_ROTATION:
                    prev_mode = option_state.mode
                    option_state = OS.advance_state(
                        option_state, obs, learner_index, option_cfg, force_start=True)
                    if option_state.mode == OS.ABORT and prev_mode != OS.ABORT and outcome["abort_reason"] is None:
                        outcome["abort_reason"] = option_state.abort_reason
                    active_now = next((s for s in (mine_now.active or []) if s is not None), None)
                    if (option_state.mode == OS.COMPLETE and outcome["bulu_attack_count"] > 0
                            and active_now is not None and P.is_ex(active_now)
                            and P.can_attack_now(active_now)):
                        saw_next_attacker_ready = True

                select = obs.select
                is_attack_decision = (
                    select.maxCount == 1
                    and any(int(getattr(o, "type", -1)) == int(OptionType.ATTACK) for o in select.option))
                active_now = next((s for s in (mine_now.active or []) if s is not None), None)
                bulu_active_now = active_now is not None and P.is_bulu(active_now)

                action = _learner_rollout_action(obs, candidate.option_name, option_state.mode, deck_ids)
                if (is_attack_decision and bulu_active_now and action
                        and 0 <= action[0] < len(select.option)
                        and int(getattr(select.option[action[0]], "type", -1)) == int(OptionType.ATTACK)):
                    outcome["bulu_attack_count"] += 1
                    opp_active_now = next((s for s in (opp_now.active or []) if s is not None), None)
                    pending_bulu_ko_check = {"serial": getattr(opp_active_now, "serial", None)}
            else:
                action = _opponent_rollout_action(obs, opp_pm)
                sel = obs.select
                if (sel is not None and sel.maxCount == 1 and action and 0 <= action[0] < len(sel.option)
                        and int(getattr(sel.option[action[0]], "type", -1)) == int(OptionType.ATTACK)):
                    my_active_now = next((s for s in (mine_now.active or []) if s is not None), None)
                    pending_opp_ko_check = {"serial": getattr(my_active_now, "serial", None)}

            try:
                next_node = cg_api.search_step(node.searchId, action)
            except ValueError as exc:
                outcome["error"] = f"illegal_action: {exc!r}"
                break
            prev_id = node.searchId
            node = next_node
            try:
                cg_api.search_release(prev_id)
            except Exception:  # noqa: BLE001
                pass
        else:
            outcome["error"] = outcome["error"] or "max_rollout_steps"

        if candidate.option_name == SINGLE_PRIZE_ROTATION:
            outcome["loop_complete"] = compute_loop_complete(
                option_state.mode, OS, outcome["bulu_attack_count"], saw_next_attacker_ready)
        return outcome
    finally:
        for sid in {getattr(root, "searchId", None), getattr(node, "searchId", None)}:
            if sid is None:
                continue
            try:
                cg_api.search_release(sid)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# 1発火局面 x N決定化 x 2Option を処理し、design.md §9.5 のレコードを組み立てる
# ---------------------------------------------------------------------------

def _process_triggered_state(obs, pair, trigger_kind, learner_index, match_seed, turn,
                             arch, opp_pm, deck_ids, determinizations: int) -> list[dict]:
    from cg import api as cg_api

    ENC = _W["ENC"]
    ex_candidate, single_candidate = pair
    state_id = state_id_of(obs.current)
    encoded = ENC.encode_strategy_pair(obs, learner_index, pair)

    records: list[dict] = []
    hidden_state_failures = 0
    try:
        for det_id in range(determinizations):
            seed = determinization_seed(match_seed, turn, state_id, det_id)
            rng = random.Random(seed)
            try:
                hidden_state = build_estimated_hidden_state(obs, learner_index, rng)
            except Exception as exc:  # noqa: BLE001
                hidden_state_failures += 1
                records.append({
                    "schema_version": SCHEMA_VERSION, "git_commit": _W["git_commit"],
                    "state_id": state_id, "pair_id": f"{state_id}:{det_id}",
                    "match_seed": match_seed, "determinization_id": det_id,
                    "opponent_archetype": arch, "learner_index": learner_index,
                    "trigger_kind": trigger_kind, "option_name": None,
                    "continuous_features": [], "slot_card_ids": [], "option_features": [],
                    "first_action_identity": {},
                    "outcome": {"win": None, "error": f"hidden_state_build_failed: {exc!r}"},
                })
                continue
            for candidate in (ex_candidate, single_candidate):
                outcome = _rollout_option(
                    obs, hidden_state, candidate, learner_index, opp_pm, deck_ids, _W["config"])
                records.append({
                    "schema_version": SCHEMA_VERSION,
                    "git_commit": _W["git_commit"],
                    "state_id": state_id,
                    "pair_id": f"{state_id}:{det_id}",
                    "match_seed": match_seed,
                    "determinization_id": det_id,
                    "opponent_archetype": arch,
                    "learner_index": learner_index,
                    "trigger_kind": trigger_kind,
                    "option_name": candidate.option_name,
                    "continuous_features": encoded["continuous_features"],
                    "slot_card_ids": encoded["slot_card_ids"],
                    "option_features": encoded["option_features"][candidate.option_name],
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


# ---------------------------------------------------------------------------
# 通常self-playを進めながら、発火局面をその場でrolloutする
# ---------------------------------------------------------------------------

def _process_task(task, determinizations: int, max_states_per_game: int) -> dict:
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    STRAT, OS = _W["STRAT"], _W["OS"]
    config = _W["config"]
    match_context = _W["match_context"]
    learner_index, seed, opp_idx = task
    random.seed(seed)
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    # match_contextはモジュールグローバル(player_indexキー)。1 workerプロセスは複数taskを
    # 順次処理するため、試合開始のたびに明示的にresetする(このタスクのagent()呼び出しは
    # obs.select is Noneのターンを一度も通らないため、agent()内部の暗黙resetに任せられない)。
    match_context.reset()

    all_records: list[dict] = []
    n_states = 0
    err = None
    n = 0
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"error": f"start {sd.errorType}", "records": [], "n_states": 0}
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
                baseline_action = _W["agent"].agent(obs, config)  # match_context.update()を内部で呼ぶ
                if n_states < max_states_per_game:
                    # 通常self-playはまだSINGLE_PRIZE_ROTATIONを選択する仕組みを持たない
                    # (Q-criticはPhase3、意思決定への配線はまだ無い)。したがって発火判定は
                    # 常にIDLEから行う。ここでOption Stateを強制開始してしまうと、ブルルが
                    # 盤面に出た瞬間からdetect_triggerが常にTRIGGER_CONTINUEを返すようになり、
                    # main_attach/promote/retreatの発火局面が事実上収集できなくなる
                    # (外部レビュー指摘、実コードで確認済みのバグ)。
                    trigger = STRAT.detect_trigger(obs, learner_index, deck_l, OS.IDLE)
                    if trigger is not None:
                        pair = STRAT.build_candidates(
                            obs, learner_index, config, deck_l, trigger, baseline_action)
                        if pair is not None:
                            n_states += 1
                            all_records.extend(_process_triggered_state(
                                obs, pair, trigger, learner_index, seed, cur.turn,
                                arch, opp_pm, deck_l, determinizations))
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

    return {"error": err, "records": all_records, "n_states": n_states}


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
    n_states = n_records = n_outer_errors = n_rollout_errors = n_hidden_state_failures = 0
    trigger_counts: dict[str, int] = {}
    option_counts: dict[str, int] = {}
    abort_reason_counts: dict[str, int] = {}
    pair_options: dict[str, set] = {}
    illegal_actions = 0

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
                trigger_counts[rec["trigger_kind"]] = trigger_counts.get(rec["trigger_kind"], 0) + 1
                if rec.get("option_name"):
                    option_counts[rec["option_name"]] = option_counts.get(rec["option_name"], 0) + 1
                    pair_options.setdefault(rec["pair_id"], set()).add(rec["option_name"])
                err = rec["outcome"].get("error")
                if err:
                    n_rollout_errors += 1
                    if "hidden_state_build_failed" in str(err):
                        n_hidden_state_failures += 1
                    if "illegal" in str(err):
                        illegal_actions += 1
                reason = rec["outcome"].get("abort_reason")
                if reason:
                    abort_reason_counts[reason] = abort_reason_counts.get(reason, 0) + 1

    complete_pairs = sum(1 for opts in pair_options.values() if len(opts) == 2)
    incomplete_pairs = sum(1 for opts in pair_options.values() if len(opts) != 2)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "games": args.games, "determinizations": args.determinizations,
        "max_states_per_game": args.max_states_per_game,
        "n_states_collected": n_states, "n_rollout_records": n_records,
        "n_outer_game_errors": n_outer_errors, "n_rollout_errors": n_rollout_errors,
        "n_hidden_state_failures": n_hidden_state_failures, "n_illegal_actions": illegal_actions,
        "trigger_kind_counts": trigger_counts, "option_name_counts": option_counts,
        "abort_reason_counts": abort_reason_counts,
        "complete_pairs": complete_pairs, "incomplete_pairs": incomplete_pairs,
        "wall_seconds": time.time() - t0, "output": str(out_path),
    }
    atomic_write_json(Path(str(out_path) + ".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
