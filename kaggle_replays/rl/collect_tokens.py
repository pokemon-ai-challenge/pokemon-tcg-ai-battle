"""T0: 既存MLP(教師)でself-playしながら、トークン形式のshardを収集する。

設計書: ``sample_submission/docs/plans/neural-agent/transformer-tokenized-encoder-design.md`` T0。

``kaggle_replays/rl/collect_parallel.py`` と同じ構造(pure-Python ``PolicyModel`` で
softmaxサンプリングしながらcgエンジンで対戦する、torch非依存のworker並列収集)を踏襲する。
違いは、各決定点で

1. 既存の集約特徴(``legacy_state_feat``/``legacy_option_feats``/``option_card_ids``)に加えて
   ``board_tokens.build_board_tokens()`` で盤面トークン列を作る、
2. 選択肢ごとに ``board_tokens.resolve_option_target_index()`` でポインタを解決する、
3. 教師モデル(``PolicyModel``)の生スコア(softmax前、温度適用前)を ``teacher_logits`` として
   全合法手ぶん保存する(``PolicyModel._forward`` はそもそも活性化なしの生スコアを返すので、
   追加の計算は不要。既存の ``score_options`` をそのまま使う)、

の3点をやりながら ``token_shard.write_token_shard()`` で書き出すこと。

Transformer自体・NumPy版Transformer推論はここでは一切扱わない(教師は既存のpure-Python MLP、
``sample_submission/ptcg_ai/learning/policy_model.py`` の ``PolicyModel``)。
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission"), str(_HERE / "distributed")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as C  # noqa: E402  (kaggle_replays/rl/distributed/common.py)
import token_shard as TS  # noqa: E402  (kaggle_replays/rl/distributed/token_shard.py)
from run_league import read_deck_csv_file  # noqa: E402

MAX_STEPS = 3000

_W: dict = {}


def _init_worker(weights_path: str, deck_l, deck_o, temperature: float, debug_pointers: bool,
                 opp_weights_path: str | None = None) -> None:
    """``opp_weights_path``省略時は既存動作(学習側と同じ重みで自己対戦)のまま。
    指定時は相手側の行動選択だけを別モデルで行う(相手の行動は教師ラベルにはしない、
    学習側(learner_index)の決定点のみ記録する既存の仕組みは変わらない)。"""
    from ptcg_ai.learning.policy_model import PolicyModel
    pm = PolicyModel(weights_path)
    if not pm.is_ready:
        raise RuntimeError(f"教師モデルを読み込めない: {weights_path}")
    opp_pm = pm
    if opp_weights_path is not None:
        opp_pm = PolicyModel(opp_weights_path)
        if not opp_pm.is_ready:
            raise RuntimeError(f"相手モデルを読み込めない: {opp_weights_path}")
    _W["pm"] = pm
    _W["opp_pm"] = opp_pm
    _W["deck_l"] = deck_l
    _W["deck_o"] = deck_o
    _W["temp"] = temperature
    _W["track_own_zone"] = bool(getattr(pm, "needs_own_zone_tracking", False))
    _W["debug_pointers"] = debug_pointers
    profile_name = getattr(pm, "_extended_profile", None) and pm._extended_profile.name
    _W["extended_features_profile"] = profile_name


def _softmax_sample(scores: list[float], temperature: float, rng: random.Random) -> tuple[int, float]:
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    z = sum(exps)
    probs = [e / z for e in exps]
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i, math.log(max(probs[i], 1e-12))
    return len(probs) - 1, math.log(max(probs[-1], 1e-12))


def _build_decision(pm, encoder, board_tokens_mod, manifest, profile_name, cur, select,
                    debug_pointers: bool) -> dict:
    """1決定点ぶんの legacy + token データを組み立てる(教師スコア込み)。"""
    legacy_state_feat = pm.encode_state_features(cur)
    legacy_option_feats = encoder.encode_options_from_state(cur, select)
    option_card_ids = encoder.encode_option_card_ids(cur, select)
    legacy_global_feat = manifest.extract_global_features(legacy_state_feat, profile_name).tolist()

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

    # 教師の生スコア(softmax前・温度適用前)。PolicyModel._forward は活性化なしの生値を
    # 返すので、これがそのまま raw logits になる(追加計算不要)。既存の状態/選択肢特徴を
    # 再計算せずそのまま使う(score_options_from_state は内部で encode_state_features を
    # もう一度呼んでしまい二重計算になるため使わない。consequence特徴は pool_v1 では
    # 使っていない(meta.consequence_fields 空)ため、_forward への直接依存で問題ない)。
    if pm._consequence_fields:
        raise RuntimeError(
            "consequence特徴を使う教師重みはT0の収集スクリプト未対応(pool_v1は空なので該当なし)")
    teacher_logits = [pm._forward(legacy_state_feat, of, ci)
                      for of, ci in zip(legacy_option_feats, option_card_ids)]

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
        out["board_serials"] = tokens.serials  # board tokenごとのPokemon.serial(監査用)
        out["debug_target_serial"] = [d.target_serial for d in debug]
        out["debug_target_card_id"] = [d.target_card_id for d in debug]
        out["debug_resolver"] = [d.resolver for d in debug]
    return out


def _play_one(task) -> dict:
    """1試合。教師MLP(自分・相手とも同一モデル)でsoftmaxサンプリングしながら、
    自分側(learner_index)の単一選択(maxCount==1)を記録する。"""
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.learning import board_tokens as board_tokens_mod
    from ptcg_ai.learning import encoder
    from ptcg_ai.learning import legacy_feature_manifest as manifest

    learner_index, seed = task
    pm = _W["pm"]
    opp_pm = _W["opp_pm"]
    temp = _W["temp"]
    profile_name = _W["extended_features_profile"]
    debug_pointers = _W["debug_pointers"]
    rng = random.Random(seed ^ 0x5DEECE66D)
    random.seed(seed)

    deck0, deck1 = (_W["deck_l"], _W["deck_o"]) if learner_index == 0 else (_W["deck_o"], _W["deck_l"])
    decisions: list[dict] = []
    reward = 0.0
    winner = None
    error = None

    if _W.get("track_own_zone", False):
        from ptcg_ai.learning import extended_features
        extended_features.begin_match(_W["deck_l"])

    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        return {"steps": [], "reward": 0.0, "winner": None,
                "error": f"start {start_data.errorType}"}

    n = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                error = "current None"; break
            if _W.get("track_own_zone", False) and cur.yourIndex == learner_index:
                from ptcg_ai.learning import extended_features
                extended_features.observe(obs)
            if cur.result != -1:
                winner = cur.result
                reward = 1.0 if cur.result == learner_index else 0.0
                break
            if n >= MAX_STEPS:
                error = "max_steps"; break

            select = obs.select
            if cur.yourIndex == learner_index and select is not None and select.option and select.maxCount == 1:
                decision = _build_decision(pm, encoder, board_tokens_mod, manifest, profile_name,
                                           cur, select, debug_pointers)
                idx, logp = _softmax_sample(decision["teacher_logits"], temp, rng)
                decision["chosen_idx"] = idx
                decision["logprob"] = logp
                decisions.append(decision)
                action = [idx]
            elif cur.yourIndex == learner_index and select is not None and select.option:
                # 学習側の複数選択(記録しない): 学習側自身のモデルでgreedy。
                scores = pm.score_options(obs)
                nn = len(select.option)
                count = max(select.minCount, min(select.maxCount, nn))
                action = (sorted(range(nn), key=lambda i: scores[i], reverse=True)[:count]
                         if scores else list(range(count)))
            elif select is not None and select.option:
                # 相手番(記録しない): 相手モデル(opp_pm。既定はpmと同一=自己対戦)でgreedy。
                scores = opp_pm.score_options(obs)
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


def parallel_collect_tokens(weights_path: str, deck_l, deck_o, n_games: int, seed0: int,
                            temperature: float, workers: int, debug_pointers: bool = False,
                            opp_weights_path: str | None = None):
    """ワーカー並列でn_games収集。戻り: (trajectories, wins, valid, errors)。

    ``trajectories`` の各要素は ``{"steps": [decision, ...], "reward": float}``
    (``token_shard.write_token_shard`` にそのまま渡せる形)。

    ``opp_weights_path``省略時は既存動作(自己対戦)のまま。指定時は相手側だけ
    別モデルで行動選択する(記録される決定点は引き続き学習側(learner_index)のみ)。
    """
    tasks = [(g % 2, seed0 + g) for g in range(n_games)]
    with Pool(processes=workers, initializer=_init_worker,
              initargs=(weights_path, deck_l, deck_o, temperature, debug_pointers,
                       opp_weights_path)) as pool:
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
    ap.add_argument("--teacher-weights", required=True,
                    help="教師MLPの重み(既存 policy_weights.json 形式)。")
    ap.add_argument("--deck-l", required=True, help="収集する側のデッキCSV")
    ap.add_argument("--deck-o", required=True, help="相手側のデッキCSV(教師どうしの自己対戦)")
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True, help="出力先 .npz パス")
    ap.add_argument("--run-id", default="token_t0")
    ap.add_argument("--generation", type=int, default=0)
    ap.add_argument("--debug-pointers", action="store_true",
                    help="pointer監査情報(target serial/card id/resolver種別)を追加保存する"
                         "(検証用。通常の学習経路では不要)。")
    ap.add_argument("--opponent-weights", default=None,
                    help="相手側の重み(省略時は--teacher-weightsと同じ=自己対戦)。"
                         "指定時、相手の行動はこのモデルで選ぶが教師ラベルとしては使わない"
                         "(記録される決定点は引き続き学習側=--teacher-weights側のみ)。")
    args = ap.parse_args()

    import os
    workers = args.workers or (os.cpu_count() or 2)
    deck_l = read_deck_csv_file(args.deck_l)
    deck_o = read_deck_csv_file(args.deck_o)

    t0 = time.time()
    trajs, wins, valid, errors = parallel_collect_tokens(
        args.teacher_weights, deck_l, deck_o, args.games, args.seed0,
        args.temperature, workers, debug_pointers=args.debug_pointers,
        opp_weights_path=args.opponent_weights)
    elapsed = time.time() - t0

    decisions = [step for tr in trajs for step in tr["steps"]]
    for tr in trajs:
        tr.setdefault("opp", 0)

    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.learning import card_vocab
    from ptcg_ai.learning import legacy_feature_manifest as manifest
    pm = PolicyModel(args.teacher_weights)
    vocab = card_vocab.load_vocab()
    profile_name = getattr(pm, "_extended_profile", None) and pm._extended_profile.name

    meta = {
        "run_id": args.run_id,
        "generation": args.generation,
        "teacher_weights_path": str(args.teacher_weights),
        "teacher_model_sha256": C.sha256_file(Path(args.teacher_weights)),
        "extended_features_profile": profile_name,
        "vocabulary_version": vocab.version,
        "vocabulary_hash": vocab.hash,
        "card_vocab_size": vocab.size,
        "global_feature_manifest_hash": manifest.manifest_hash(profile_name),
        "games_requested": args.games,
        "temperature": args.temperature,
        "action_selection_mode": "softmax_temperature",
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
