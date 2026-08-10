"""オーロンゲ(marnie_grimmsnarl_ex)T1蒸留用のtoken shard収集。

教師は**ルールベース**(``opponents/rule_agents/grimmsnarl.py``、BC championは混ぜない、
ユーザー指示通り)。``framework.score_option``が既存の``choose()``内部で使っている
「全選択肢に点数をつける」ロジックそのものなので、これを``teacher_logits``として使う
(既存``collect_tokens.py``の``PolicyModel._forward``相当の役割)。

``collect_tokens.py``/``token_shard.py``は変更しない。board/option token生成
(board_tokens.py・encoder.py・legacy_feature_manifest.py)はdeck非依存であることを
事前監査で確認済みなのでそのまま再利用する。legacy_state_feat/legacy_global_featの
抽出だけは(プロファイル判定のためだけに)v40のPolicyModelを流用する
(fuudin_v4 profile自体はdeck非依存の機構であることを確認済み。カード知識は使わない)。
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission"), str(_HERE / "distributed"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as C  # noqa: E402
import token_shard as TS  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

MAX_STEPS = 3000
_FEATURE_PROFILE = "fuudin_v4"
_V40_TEACHER_FOR_FEATURES = (_ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models"
                            / "model_v40.json")

_W: dict = {}


def _init_worker(deck_l, deck_o, opp_weights_path, debug_pointers: bool) -> None:
    from ptcg_ai.learning.policy_model import PolicyModel
    from opponents.rule_agents import grimmsnarl as gm
    from opponents.rule_agents import framework as fw

    feat_pm = PolicyModel(str(_V40_TEACHER_FOR_FEATURES))
    if not feat_pm.is_ready:
        raise RuntimeError("特徴抽出用のv40を読み込めない")
    _W["feat_pm"] = feat_pm
    _W["strategy"] = gm.GrimmsnarlStrategy()
    _W["deck_l"] = deck_l
    _W["deck_o"] = deck_o
    _W["debug_pointers"] = debug_pointers
    if opp_weights_path is not None:
        opp_pm = PolicyModel(opp_weights_path)
        if not opp_pm.is_ready:
            raise RuntimeError(f"相手モデルを読み込めない: {opp_weights_path}")
        _W["opp_pm"] = opp_pm
    else:
        _W["opp_pm"] = None


def _teacher_scores(fw, ctx, st, select) -> list[float]:
    return [fw.score_option(ctx, st, o) for o in select.option]


def _build_decision(feat_pm, encoder, board_tokens_mod, manifest, cur, select,
                    teacher_logits: list[float], debug_pointers: bool) -> dict:
    legacy_state_feat = feat_pm.encode_state_features(cur)
    legacy_option_feats = encoder.encode_options_from_state(cur, select)
    option_card_ids = encoder.encode_option_card_ids(cur, select)
    legacy_global_feat = manifest.extract_global_features(legacy_state_feat, _FEATURE_PROFILE).tolist()

    tokens = board_tokens_mod.build_board_tokens(cur)
    if debug_pointers:
        debug = [board_tokens_mod.resolve_option_target_index_debug(opt, cur, tokens)
                for opt in select.option]
        option_target_indices = [d.target_index for d in debug]
    else:
        debug = None
        option_target_indices = [
            board_tokens_mod.resolve_option_target_index(opt, cur, tokens)
            for opt in select.option
        ]

    out = {
        "legacy_state_feat": legacy_state_feat,
        "legacy_global_feat": legacy_global_feat,
        "legacy_option_feats": legacy_option_feats,
        "option_card_ids": option_card_ids,
        "board_numeric_feats": tokens.numeric_features,
        "board_card_ids": tokens.card_ids,
        "board_zone_ids": tokens.zone_ids,
        "option_target_indices": option_target_indices,
        "teacher_logits": teacher_logits,
    }
    if debug is not None:
        out["board_serials"] = tokens.serials
        out["debug_target_serial"] = [d.target_serial for d in debug]
        out["debug_target_card_id"] = [d.target_card_id for d in debug]
        out["debug_resolver"] = [d.resolver for d in debug]
    return out


def _greedy_action_from_scores(scores: list[float], select) -> list[int]:
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order:
        if len(out) >= select.maxCount:
            break
        if scores[i] < 0 and len(out) >= select.minCount:
            break
        out.append(i)
    return out


def _play_one(task) -> dict:
    """1試合。オーロンゲ(ルールベース教師)がlearner_index側、相手はopp_pm(無ければ
    ルールベース自身とのミラー)。学習側の単一選択(maxCount==1)だけ記録する。"""
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.learning import board_tokens as board_tokens_mod
    from ptcg_ai.learning import encoder
    from ptcg_ai.learning import legacy_feature_manifest as manifest
    from opponents.rule_agents import framework as fw

    learner_index, seed = task
    feat_pm = _W["feat_pm"]
    st = _W["strategy"]
    opp_pm = _W["opp_pm"]
    debug_pointers = _W["debug_pointers"]
    random.seed(seed)

    deck0, deck1 = (_W["deck_l"], _W["deck_o"]) if learner_index == 0 else (_W["deck_o"], _W["deck_l"])
    decisions: list[dict] = []
    reward = 0.0
    winner = None
    error = None

    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        return {"steps": [], "reward": 0.0, "winner": None, "error": f"start {start_data.errorType}"}

    n = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                error = "current None"; break
            if cur.result != -1:
                winner = cur.result
                reward = 1.0 if cur.result == learner_index else 0.0
                break
            if n >= MAX_STEPS:
                error = "max_steps"; break

            select = obs.select
            if cur.yourIndex == learner_index and select is not None and select.option:
                ctx = fw.build_ctx(obs, _W["deck_l"])
                st.prepare(ctx)
                ctx.memo["attack_evals"] = fw.evaluate_attacks(ctx, bonus=st.attack_bonus(ctx))
                scores = _teacher_scores(fw, ctx, st, select)
                if select.maxCount == 1:
                    decision = _build_decision(feat_pm, encoder, board_tokens_mod, manifest,
                                               cur, select, scores, debug_pointers)
                    idx = max(range(len(scores)), key=lambda i: scores[i])
                    decision["chosen_idx"] = idx
                    decision["logprob"] = 0.0  # ルールベースは確率分布を持たない。蒸留はhard-top1のみ使う。
                    decisions.append(decision)
                    action = [idx]
                else:
                    action = _greedy_action_from_scores(scores, select)
            elif select is not None and select.option:
                if opp_pm is not None:
                    scores = opp_pm.score_options(obs)
                else:
                    ctx = fw.build_ctx(obs, _W["deck_l"])
                    st.prepare(ctx)
                    ctx.memo["attack_evals"] = fw.evaluate_attacks(ctx, bonus=st.attack_bonus(ctx))
                    scores = _teacher_scores(fw, ctx, st, select)
                nn = len(select.option)
                count = max(select.minCount, min(select.maxCount, nn))
                action = (sorted(range(nn), key=lambda i: scores[i], reverse=True)[:count]
                         if scores else list(range(count)))
            else:
                action = []
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        battle_finish()

    return {"steps": decisions, "reward": reward, "winner": winner, "error": error}


def parallel_collect_tokens(deck_l, deck_o, n_games, seed0, workers, opp_weights_path=None,
                            debug_pointers=False):
    tasks = [(g % 2, seed0 + g) for g in range(n_games)]
    with Pool(processes=workers, initializer=_init_worker,
              initargs=(deck_l, deck_o, opp_weights_path, debug_pointers)) as pool:
        results = pool.map(_play_one, tasks, chunksize=max(1, n_games // (workers * 4)))
    trajs, wins, valid, errors = [], 0, 0, 0
    for r in results:
        if r["error"] is not None:
            errors += 1
            continue
        valid += 1
        wins += 1 if r["reward"] >= 1.0 else 0
        if r["steps"]:
            trajs.append(r)
    return trajs, wins, valid, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck-l", required=True)
    ap.add_argument("--deck-o", required=True)
    ap.add_argument("--opponent-weights", default=None)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="grimmsnarl_rule_teacher")
    ap.add_argument("--generation", type=int, default=0)
    args = ap.parse_args()

    import os
    workers = args.workers or (os.cpu_count() or 2)
    deck_l = read_deck_csv_file(args.deck_l)
    deck_o = read_deck_csv_file(args.deck_o)

    t0 = time.time()
    trajs, wins, valid, errors = parallel_collect_tokens(
        deck_l, deck_o, args.games, args.seed0, workers,
        opp_weights_path=args.opponent_weights, debug_pointers=False)
    elapsed = time.time() - t0

    decisions = [step for tr in trajs for step in tr["steps"]]
    for tr in trajs:
        tr.setdefault("opp", 0)

    from ptcg_ai.learning import card_vocab
    from ptcg_ai.learning import legacy_feature_manifest as manifest
    vocab = card_vocab.load_vocab()

    meta = {
        "run_id": args.run_id,
        "generation": args.generation,
        "teacher_weights_path": "opponents/rule_agents/grimmsnarl.py (rule-based, hard-top1)",
        "teacher_model_sha256": C.sha256_file(Path(_ROOT / "opponents" / "rule_agents" / "grimmsnarl.py")),
        "extended_features_profile": _FEATURE_PROFILE,
        "vocabulary_version": vocab.version,
        "vocabulary_hash": vocab.hash,
        "card_vocab_size": vocab.size,
        "global_feature_manifest_hash": manifest.manifest_hash(_FEATURE_PROFILE),
        "games_requested": args.games,
        "temperature": 1.0,  # ルールベースはhard-top1のみ(温度概念なし、形式合わせの既定値)
        "action_selection_mode": "rule_based_hard_top1",
        "opponent_weights_path": str(args.opponent_weights) if args.opponent_weights else None,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "seconds": round(elapsed, 1),
        "wins": wins, "valid": valid, "errors": errors,
    }
    path = TS.write_token_shard(Path(args.out), decisions, trajs, meta)
    size_mb = path.stat().st_size / 1e6
    wr = wins / valid if valid else float("nan")
    print(f"完了 {elapsed:.0f}s  勝率 {wins}/{valid}={wr:.3f}  err={errors}  "
         f"決定点={len(decisions)}", flush=True)
    print(f"書き出し: {path}  ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
