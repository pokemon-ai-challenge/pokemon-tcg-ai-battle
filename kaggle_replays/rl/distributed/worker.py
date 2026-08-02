"""worker: 配られた世代のモデルで Self-Play し、経験データを1ファイル書き出す。

Colab / Kaggle など試合生成だけを担当するマシンで動かす。**torch は使わない**
(収集は pure-Python 方策 + cg エンジンで完結する)ので、numpy さえあれば動く。

    python worker.py --run-dir <run_dir> --worker-id colab

試合数・温度・相手・デッキはすべて run.json から読む。コマンドラインでは変更できない。
これは「1回の収集量や設定を PC 間で揃える」ためで、うっかり片方だけ違う条件で回して
そのデータが PPO 更新に混ざるのを、仕組みとして防ぐ。
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import common as C
from collect_parallel import parallel_collect
from run_league import read_deck_csv_file


def verify_feature_dims(model_path: Path) -> int:
    """モデルが期待する盤面特徴の数と、実際に作れる数が一致するか確かめる。

    ここがずれても対戦は続いてしまい(内積が短い側で打ち切られる)、勝率だけが
    静かに崩れる。原因は「古いコードが Kaggle に配られた」「追加特徴のファイルが
    配布に入っていない」といった配布事故で、ログには何も出ない。実際に踏んだので、
    収集の前に必ず突き合わせる。
    """
    from ptcg_ai.learning.policy_model import PolicyModel

    pm = PolicyModel(str(model_path))
    if not pm.is_ready:
        raise SystemExit(f"モデルを読み込めない: {model_path}")
    expected = len(pm._state_mean)
    if not hasattr(pm, "encode_state_features"):
        raise SystemExit(
            "PolicyModel に encode_state_features が無い。古い版のコードが配られている。")
    actual = len(pm.encode_state_features(None))
    if actual != expected:
        raise SystemExit(
            f"盤面特徴の数が合わない: モデルの期待 {expected} / 実際に作れる数 {actual}。"
            "追加特徴のプロファイルやコードが配布に入っているか確認する。")
    return expected


def collect_generation(run: dict, run_dir: Path, gen: int, worker_index: int,
                       workers: int, quiet: bool = False):
    """run.json の設定どおりに1世代ぶん収集して (trajs, 統計) を返す。"""
    deck_l = read_deck_csv_file(str(C.resolve_deck(run["learner_deck"])))
    model = C.model_path(run_dir, gen)
    if not model.exists():
        raise SystemExit(f"モデルが無い: {model}\n  学習PCから models/model_v{gen}.json を持ってくる。")

    total = run["games_per_worker"]
    opponents = run["opponents"]
    counts = C.split_games(total, opponents)
    base = C.seed_base(gen, worker_index)

    all_trajs, wins, valid, errors = [], 0, 0, 0
    offset = 0
    per_opponent = []
    for opp_index, (opp, n_games) in enumerate(zip(opponents, counts)):
        if n_games == 0:
            continue
        opp_w = C.resolve_opponent_weights(run_dir, opp.get("weights"))
        # 相手はデッキも変えられる。方策の重みだけ差し替えて同じデッキを使わせると、
        # そのアーキタイプ用に学習した重みが噛み合わない相手になってしまう。
        deck_o = read_deck_csv_file(
            str(C.resolve_deck(opp.get("deck") or run["opponent_deck"])))
        t0 = time.time()
        trajs, w, v, e = parallel_collect(
            str(model), opp_w, deck_l, deck_o,
            n_games=n_games, seed0=base + offset,
            temperature=run["temperature"], workers=workers)
        dt = time.time() - t0
        # どの相手と戦った試合かを残す。learner が advantage を相手ごとに正規化するのに使う。
        # これが無いと「得意な相手では何をしても褒められ、苦手な相手では何をしても叱られる」
        # という、手の良し悪しと無関係な偏りがそのまま勾配に乗る(critic は相手をほとんど
        # 価値に反映していないことを実測済み: 相手119次元を消しても v(s) は 0.14 しか動かない)。
        for tr in trajs:
            tr["opp"] = opp_index
        all_trajs.extend(trajs)
        wins += w; valid += v; errors += e
        offset += n_games          # 相手ごとにシード範囲をずらす(重複させない)
        per_opponent.append({"id": opp["id"], "games": n_games, "wins": w,
                             "valid": v, "errors": e, "seconds": round(dt, 1)})
        if not quiet:
            wr = w / v if v else float("nan")
            print(f"  vs {opp['id']}: {n_games}試合 {dt:.0f}s 勝率 {w}/{v}={wr:.3f} err={e}",
                  flush=True)

    stats = {"wins": wins, "valid": valid, "errors": errors,
             "per_opponent": per_opponent, "seed_base": base,
             "seed_range": [base, base + total]}
    return all_trajs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--worker-id", required=True,
                    help="run.json の workers に登録した名前(例: colab / kaggle)")
    ap.add_argument("--workers", type=int, default=None,
                    help="このマシンで使う並列プロセス数。既定は CPU 数")
    ap.add_argument("--out", default=None,
                    help="シャードの書き出し先。既定 <run_dir>/shards/v<gen>/<worker_id>.npz")
    ap.add_argument("--start-method", default="spawn", choices=["spawn", "fork", "default"],
                    help="収集プロセスの起動方式。既定 spawn(Linux の fork だと親が読み込み済みの "
                         "cg エンジンを子が引き継ぐため)")
    args = ap.parse_args()

    start = C.set_collection_start_method(args.start_method)
    n_proc = args.workers or (os.cpu_count() or 2)

    run_dir = Path(args.run_dir)
    run = C.load_run(run_dir)
    gen = run["generation"]

    if args.worker_id not in run["workers"]:
        raise SystemExit(
            f"worker-id '{args.worker_id}' が run.json に無い(登録済み: {run['workers']})。\n"
            "  シードの割り当てが名前の順番で決まるので、勝手な名前は使えない。")
    worker_index = run["workers"].index(args.worker_id)

    model = C.model_path(run_dir, gen)
    model_sha = C.sha256_file(model)

    print(f"run={run['run_id']} 世代 v{gen} worker={args.worker_id}(#{worker_index}) "
          f"並列={n_proc}({start})", flush=True)
    n_state = verify_feature_dims(model)
    print(f"  model sha256={model_sha[:16]}...  盤面特徴={n_state}  "
          f"{run['games_per_worker']}試合 温度={run['temperature']}", flush=True)

    t0 = time.time()
    trajs, stats = collect_generation(run, run_dir, gen, worker_index, n_proc)
    elapsed = time.time() - t0

    meta = {
        "run_id": run["run_id"],
        "generation": gen,
        "model_sha256": model_sha,
        "worker_id": args.worker_id,
        "worker_index": worker_index,
        "games_requested": run["games_per_worker"],
        "temperature": run["temperature"],
        "opponents": run["opponents"],
        "learner_deck": run["learner_deck"],
        "opponent_deck": run["opponent_deck"],
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "seconds": round(elapsed, 1),
        **stats,
    }
    out = Path(args.out) if args.out else C.shard_dir(run_dir, gen) / f"{args.worker_id}.npz"
    path = C.write_shard(out, trajs, meta)
    size_mb = path.stat().st_size / 1e6
    n_steps = sum(len(tr["steps"]) for tr in trajs)
    wr = stats["wins"] / stats["valid"] if stats["valid"] else float("nan")
    print(f"\n完了 {elapsed:.0f}s  勝率 {stats['wins']}/{stats['valid']}={wr:.3f} "
          f"err={stats['errors']}  決定点={n_steps}")
    print(f"書き出し: {path}  ({size_mb:.1f} MB)")
    print("このファイルを学習PCの shards/v%d/ へ置く。" % gen)


if __name__ == "__main__":
    main()
