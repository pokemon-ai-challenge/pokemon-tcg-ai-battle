"""分散 Self-Play の run を新規作成する(学習PCで1回だけ実行)。

ここで決めた設定は run.json に固定され、以後 worker はすべてこのファイルを読む。
「収集量や温度を PC ごとに揃える」ためのやり方が、設定を worker のコマンドライン引数に
持たせず、モデルと一緒に配る run.json 側に持たせること。

例:
    python init_run.py --run-id dragapult_v1 \
        --learner-arch dragapult_ex --opponent-arch alakazam \
        --workers colab kaggle --games-per-worker 250
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="実験の名前。シャードの取り違え検出にも使う")
    ap.add_argument("--run-dir", default=None, help="既定 kaggle_replays/rl/runs/<run_id>")
    ap.add_argument("--learner-arch", default="dragapult_ex")
    ap.add_argument("--learner-weights", default=None,
                    help="初期重み。既定 policy_weights_<learner-arch>.json")
    ap.add_argument("--learner-deck", default=None, help="既定 <learner-arch>/01.csv")
    ap.add_argument("--opponent-arch", default="alakazam", help="相手のデッキ")
    ap.add_argument("--opponent-weights", default=None,
                    help="相手の重み。未指定なら production の既定 alakazam")
    ap.add_argument("--workers", nargs="+", required=True,
                    help="worker の名前を並べる(例: colab kaggle)。順番がシード割り当てを決めるので、"
                         "あとから並べ替えないこと")
    ap.add_argument("--games-per-worker", type=int, required=True,
                    help="1世代あたり1台が回す試合数。全 worker 共通")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="収集時の softmax 温度。learner の重要度比もこの値で計算される")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr-policy", type=float, default=3e-4)
    ap.add_argument("--lr-value", type=float, default=1e-3)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--minibatch-size", type=int, default=16384,
                    help="PPO更新のミニバッチ(決定点数)。worker を増やすとバッチが大きくなり"
                         "メモリを食うので既定で分割する。0 なら全件まとめて(train_v3 と同じ挙動)")
    args = ap.parse_args()

    if len(set(args.workers)) != len(args.workers):
        raise SystemExit("worker 名が重複している")
    if len(args.workers) > C.MAX_WORKERS:
        raise SystemExit(f"worker が多すぎる(最大 {C.MAX_WORKERS})")
    if not 0 < args.games_per_worker <= C.MAX_GAMES_PER_WORKER:
        raise SystemExit(f"--games-per-worker は 1..{C.MAX_GAMES_PER_WORKER}")

    run_dir = Path(args.run_dir) if args.run_dir else (
        C.REPO_ROOT / "kaggle_replays" / "rl" / "runs" / args.run_id)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"すでに中身がある: {run_dir}")

    src = Path(args.learner_weights) if args.learner_weights else (
        C.LEARNING_DIR / f"policy_weights_{args.learner_arch}.json")
    if not src.exists():
        src2 = C.LEARNING_DIR / src.name
        if not src2.exists():
            raise SystemExit(f"初期重みが見つからない: {src}")
        src = src2

    learner_deck = args.learner_deck or args.learner_arch
    opponent_deck = args.opponent_arch
    C.resolve_deck(learner_deck)      # 存在確認(無ければここで落とす)
    C.resolve_deck(opponent_deck)

    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "state").mkdir(parents=True, exist_ok=True)
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, C.model_path(run_dir, 0))

    # 相手の重みがローカルの実ファイルを指しているなら、model_v0.json と同じやり方で
    # run_dir 配下(opponents/)にコピーし、run.json には run_dir からの相対パスを書く。
    # 絶対パス(例: C:\Users\...\policy_weights_x.json)のまま run.json に残すと、
    # worker 側(Kaggle では run_dir が /kaggle/temp/run になる)でファイルが存在せず
    # 解決できない。未指定(None)や、実ファイルに解決できない指定(production の既定名など)
    # はそのまま素通しする(後方互換)。
    opponent_weights_rel = args.opponent_weights
    if args.opponent_weights:
        cand = Path(args.opponent_weights)
        if not cand.is_file():
            cand2 = C.LEARNING_DIR / cand.name
            if cand2.is_file():
                cand = cand2
        if cand.is_file():
            opp_dir = run_dir / "opponents"
            opp_dir.mkdir(parents=True, exist_ok=True)
            dest = opp_dir / cand.name
            shutil.copyfile(cand, dest)
            opponent_weights_rel = str(dest.relative_to(run_dir).as_posix())

    run = {
        "run_id": args.run_id,
        "shard_format": C.SHARD_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation": 0,                     # いま配布中の世代。learner が +1 する
        "workers": list(args.workers),       # 順番 = シード割り当て。並べ替え禁止
        "games_per_worker": args.games_per_worker,
        "temperature": args.temperature,
        "learner_arch": args.learner_arch,
        "learner_deck": learner_deck,
        "opponent_deck": opponent_deck,
        # 対戦相手プール。いまは1件だが、過去世代を混ぜたくなったら
        #   {"id": "v5", "weights": "models/model_v5.json", "share": 0.3}
        # のように足すだけでよい(worker 側は share の比で試合数を分ける)。
        "opponents": [
            {"id": args.opponent_weights or "production_default",
             "weights": opponent_weights_rel,
             "share": 1.0},
        ],
        "ppo": {
            "epochs": args.epochs, "clip": args.clip,
            "lr_policy": args.lr_policy, "lr_value": args.lr_value,
            "entropy": args.entropy, "gamma": args.gamma, "lam": args.lam,
            "minibatch_size": args.minibatch_size,
        },
        "base_weights": str(src.relative_to(C.REPO_ROOT)) if src.is_relative_to(C.REPO_ROOT) else str(src),
    }
    C.save_run(run_dir, run)

    print(f"run を作成: {run_dir}")
    print(f"  worker: {', '.join(args.workers)}  "
          f"(1世代あたり {args.games_per_worker}試合 × {len(args.workers)}台 = "
          f"{args.games_per_worker * len(args.workers)}試合)")
    print(f"  model_v0 = {src.name}  sha256={C.sha256_file(C.model_path(run_dir, 0))[:16]}...")
    print("\n次にやること: run.json と models/model_v0.json を各 worker へ配って、")
    print("  python worker.py --run-dir <run_dir> --worker-id <名前>")


if __name__ == "__main__":
    main()
