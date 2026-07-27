#!/usr/bin/env python3
"""世代ループ: 自己対戦で収集 -> 方策勾配で更新 -> 凍結した相手プールで評価 -> 採否判定。

設計書(``test_plan/ptcg_rl_design.md`` §4, §5, §6 row5-6)のとおり:

- 自分のデッキは archetype1(マリィのオーロンゲ)固定(``sample_submission/deck.csv``、
  これは既に archetype1 のデッキと一致している)。
- 相手プール(評価専用。**凍結し、学習中は一切更新しない**): archetype1-4 の専用BCモデル
  (``kaggle_replays/policy_prior/output/policy_weights_archetype{1..4}.json``)。
  現在の基準値(汎用BCモデル): 51.0%(±2.8pt、300試合×4相手=1,200試合)。
  この評価プロトコルとまったく同じもの(``parallel_eval.py --head-to-head
  --opponent-pool --opponents archetype1,archetype2,archetype3,archetype4``)を
  各世代で再利用する(既存インフラの再利用が望ましいという指示に従う)。
- 世代ごとに合計勝率(4相手合算。個別は参考値 — 設計書 §5.2 の多重比較への配慮)を測り、
  前世代を下回ったら巻き戻す(採用しない)。

出力: ``kaggle_replays/policy_prior/output/rl/genNNN.json``(各世代で採用された重み)、
``kaggle_replays/policy_prior/output/rl/genNNN_candidate.json``(採否に関わらず勾配更新
直後の重み)、``kaggle_replays/policy_prior/output/rl/history.json``(世代ごとの記録)。

**重要**: ``sample_submission/ptcg_ai/learning/policy_weights.json``(現在提出中のBC
モデル)は一切書き換えない。初期重みとして読み込むだけ。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_POLICY_PRIOR_DIR = _HERE.parent
_REPO_ROOT = _POLICY_PRIOR_DIR.parents[1]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
_BASE_BC_WEIGHTS = _SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "learning" / "policy_weights.json"
_PARALLEL_EVAL = _SAMPLE_SUBMISSION_DIR / "tests" / "local_sim" / "parallel_eval.py"
_DEFAULT_OUT_DIR = _POLICY_PRIOR_DIR / "output" / "rl"
_DEFAULT_OPPONENTS = "archetype1,archetype2,archetype3,archetype4"  # 凍結プール(主指標)

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_POLICY_PRIOR_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_PRIOR_DIR))

from rl.policy import LinearPolicy  # noqa: E402
from rl.reinforce import apply_update, compute_batch_gradient  # noqa: E402
from rl.selfplay import collect_selfplay, resolve_opponent_pool  # noqa: E402


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    import math

    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def evaluate_pool(weights_path: Path, games_per_opponent: int, workers: int,
                   opponents: str, json_out: Path, seed: int) -> dict:
    """``parallel_eval.py --head-to-head --opponent-pool`` を子プロセスで実行し、
    凍結プールに対する合計勝率を測る(既存インフラをそのまま再利用する)。
    """
    # サブプロセスは cwd=sample_submission で動くので、相対パスのまま渡すと
    # 呼び出し元(このプロセス)の cwd 基準の場所と食い違う。絶対パスへ解決する。
    weights_path = Path(weights_path).resolve()
    json_out = Path(json_out).resolve()
    cmd = [
        sys.executable, str(_PARALLEL_EVAL),
        "--games", str(games_per_opponent),
        "--head-to-head", "--opponent-pool",
        "--opponents", opponents,
        "--workers", str(workers),
        "--own-weights", str(weights_path),
        "--json-out", str(json_out),
        "--seed", str(seed),
    ]
    print(f"  評価コマンド: {' '.join(cmd)}", flush=True)
    # encoding を明示しないと Windows では既定の cp932 で復号され、子プロセスの
    # 日本語(UTF-8)出力で UnicodeDecodeError になる。結果自体は --json-out から
    # 読めるので実害が出にくく、代わりに「失敗したときに子プロセスのメッセージが
    # 読めない」という最も困る形で壊れる。errors="replace" も付けて、想定外の
    # バイト列が来ても例外にせず読めるところまでは読む。
    result = subprocess.run(
        cmd, cwd=str(_SAMPLE_SUBMISSION_DIR), capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("parallel_eval.py の実行に失敗しました")
    print(result.stdout)

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    combined = payload["combined"]["on"]
    win, loss, invalid = combined.get("win", 0), combined.get("loss", 0), combined.get("invalid", 0)
    n = win + loss
    lo, hi = wilson(win, n)
    return {
        "win": win, "loss": loss, "invalid": invalid, "n": n,
        "win_rate": win / n if n else float("nan"),
        "ci_lo": lo, "ci_hi": hi,
        "by_opponent": payload["by_arm_opponent"],
    }


def run_generations(
    n_generations: int,
    games_per_gen: int,
    eval_games_per_opponent: int,
    temperature: float,
    gamma: float,
    lr: float,
    selfplay_workers: int,
    eval_workers: int,
    opponents: str,
    out_dir: Path,
    seed: int,
    accept_margin: float = 0.0,
    max_relative_step: float = 0.02,
    selfplay_opponents: str | None = None,
    base_weights: str | None = None,
) -> None:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.json"
    history: list[dict] = []

    # 自己対戦の相手プール(評価用の凍結プールとは別物。学習時に相手を多様化する
    # ためのもので、既定(None)なら従来どおりの純粋な自己対戦)。
    # 専用モデルが見つからなければ resolve_opponent_pool が警告して None に
    # フォールバックする(例外にしない)。
    selfplay_opponent_pool = resolve_opponent_pool(selfplay_opponents)
    if selfplay_opponents is None:
        print("自己対戦: 純粋な自己対戦(同デッキ・同方策)")
    elif selfplay_opponent_pool is None:
        print(f"自己対戦: 相手プール({selfplay_opponents})を要求されたが専用モデルが"
              "揃っていないため純粋な自己対戦にフォールバック")
    else:
        print(f"自己対戦: 相手プール({[o['name'] for o in selfplay_opponent_pool]})と対戦"
              "(自分側の decision のみ記録)")

    # --- 世代0: 現行BC(汎用)モデルをそのままコピーして出発点にする(上書きしない) ---
    gen0_path = out_dir / "gen000.json"
    base = Path(base_weights) if base_weights else _BASE_BC_WEIGHTS
    base_policy = LinearPolicy.load(base)
    base_policy.meta["rl_generation"] = 0
    base_policy.meta["rl_parent"] = None
    base_policy.save(gen0_path)
    print(f"世代0(出発点): {base} をコピー -> {gen0_path}")

    print(f"世代0を凍結プール({opponents})に対して評価中 "
          f"({eval_games_per_opponent}試合×{len(opponents.split(','))}相手)...")
    gen0_eval = evaluate_pool(
        gen0_path, eval_games_per_opponent, eval_workers, opponents,
        out_dir / "gen000_eval.json", seed,
    )
    print(f"  世代0 プール勝率: {gen0_eval['win_rate']:.1%} "
          f"[{gen0_eval['ci_lo']:.1%}, {gen0_eval['ci_hi']:.1%}] ({gen0_eval['n']}試合)")
    history.append({
        "generation": 0, "weights_path": str(gen0_path), "accepted": True,
        "self_play": None, "pool_eval": gen0_eval,
    })
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    accepted_path = gen0_path
    accepted_eval = gen0_eval

    for gen in range(1, n_generations + 1):
        print(f"\n=== 世代 {gen} ===", flush=True)
        t0 = time.time()

        print(f"自己対戦収集中: {games_per_gen}試合 (T={temperature}, "
              f"{selfplay_workers}プロセス) 元重み={accepted_path.name}")
        episodes = collect_selfplay(
            weights_path=accepted_path,
            games=games_per_gen,
            temperature=temperature,
            workers=selfplay_workers,
            seed=seed + gen * 100003,
            opponent_pool=selfplay_opponent_pool,
        )
        n_decisions = sum(len(ep["decisions"]) for ep in episodes)
        wins = sum(1 for ep in episodes if ep["result"] == 1)
        win_note = (
            "自己対戦なので勝敗はほぼ50%になるはず" if selfplay_opponent_pool is None
            else "相手プールとの対戦なので50%からずれてよい"
        )
        print(f"  収集完了 ({time.time() - t0:.0f}s): エピソード{len(episodes)}件"
              f"({win_note}: {wins}/{len(episodes)})"
              f", decision数 {n_decisions}")

        policy = LinearPolicy.load(accepted_path)
        grad, diag = compute_batch_gradient(policy, episodes, temperature=temperature, gamma=gamma)
        print(f"  勾配診断: {diag}")
        _rel = min(lr, (max_relative_step * diag["weight_norm"] / diag["grad_norm"])
                   if (max_relative_step and diag["grad_norm"]) else lr)             * diag["grad_norm"] / diag["weight_norm"] if diag["weight_norm"] else 0.0
        print(f"  実効歩幅: ||Δw||/||w|| = {_rel:.2%}（上限 {max_relative_step:.1%}）")

        candidate = apply_update(policy, grad, lr=lr,
                                 max_relative_step=max_relative_step or None)
        candidate.meta["rl_generation"] = gen
        candidate.meta["rl_parent"] = str(accepted_path)
        candidate.meta["rl_temperature"] = temperature
        candidate.meta["rl_gamma"] = gamma
        candidate.meta["rl_lr"] = lr
        candidate_path = out_dir / f"gen{gen:03d}_candidate.json"
        candidate.save(candidate_path)

        print(f"  凍結プールで評価中: {eval_games_per_opponent}試合×"
              f"{len(opponents.split(','))}相手")
        cand_eval = evaluate_pool(
            candidate_path, eval_games_per_opponent, eval_workers, opponents,
            out_dir / f"gen{gen:03d}_eval.json", seed + gen,
        )
        print(f"  世代{gen} プール勝率: {cand_eval['win_rate']:.1%} "
              f"[{cand_eval['ci_lo']:.1%}, {cand_eval['ci_hi']:.1%}] ({cand_eval['n']}試合) "
              f"(前世代: {accepted_eval['win_rate']:.1%})")

        # 「前世代を1ポイントでも下回ったら却下」は、評価のノイズが大きいと機能しない。
        # 相手1体300試合(合計1,200)でも95%CIは±2.8pt。1世代の真の改善が+1〜2ptなら、
        # 採否はほぼコイン投げになり、良い世代を捨てて計算を無駄にする。
        # accept_margin を許容幅として引き、「明らかに悪化したときだけ却下」にする。
        # margin=0 なら従来どおり厳格。
        threshold = accepted_eval["win_rate"] - accept_margin
        accepted = cand_eval["win_rate"] >= threshold
        gen_path = out_dir / f"gen{gen:03d}.json"
        if accepted:
            candidate.save(gen_path)
            accepted_path = gen_path
            accepted_eval = cand_eval
            print(f"  -> 採用(しきい値 {threshold:.1%} 以上。許容幅 {accept_margin:.1%})")
        else:
            print(f"  -> 却下・巻き戻し(しきい値 {threshold:.1%} を下回った)。"
                  f"次世代も {accepted_path.name} から再開する")

        history.append({
            "generation": gen,
            "candidate_path": str(candidate_path),
            "accepted": accepted,
            "weights_path": str(gen_path) if accepted else str(accepted_path),
            "self_play": {
                "n_games_requested": games_per_gen,
                "n_episodes": len(episodes),
                "n_decisions": n_decisions,
                **diag,
            },
            "pool_eval": cand_eval,
            "elapsed_sec": time.time() - t0,
        })
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完了。履歴: {history_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--games-per-gen", type=int, default=200, help="1世代あたりの自己対戦試合数")
    ap.add_argument("--eval-games", type=int, default=75,
                    help="評価時の相手1体あたりの試合数(--opponents の相手数倍が総試合数)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=5.0,
                    help="学習率。勾配は decision 数で平均されるので、合計だった頃の"
                         "値(0.05)とは意味が違う。実効歩幅は --max-relative-step で頭打ちになる")
    ap.add_argument("--max-relative-step", type=float, default=0.02,
                    help="1世代あたりの ||Δw||/||w|| の上限。勾配の大きさは世代ごとに"
                         "変わるため lr だけでの制御は危うい。0 で無制限")
    ap.add_argument("--selfplay-workers", type=int, default=None)
    ap.add_argument("--eval-workers", type=int, default=None)
    ap.add_argument("--opponents", default=_DEFAULT_OPPONENTS)
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    ap.add_argument("--base-weights", default=None,
                    help="世代0の出発点にする重みJSON。省略時は提出中のBCモデル。"
                         "交互作用版など別の表現から始めるときに指定する")
    ap.add_argument("--accept-margin", type=float, default=0.0,
                    help="前世代の勝率からこの幅まで下がっても採用する(既定0=厳格)。"
                         "評価のノイズ(1,200試合で±2.8pt)で良い世代を捨てるのを防ぐ")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--selfplay-opponents", default=None,
                    help="自己対戦(学習用データ収集)の相手を多様化する。省略時(既定)は"
                         "従来どおり純粋な自己対戦(同デッキ・同方策)。カンマ区切りの"
                         "アーキタイプ名(例 'archetype1,archetype2')、または 'all' で "
                         "opponent_decks.json の全種。1試合ごとに巡回で選び、各相手が"
                         "ほぼ均等に当たる。相手専用モデル(policy_prior/output/"
                         "policy_weights_<name>.json)が無ければ警告して純粋な自己対戦に"
                         "フォールバックする(--opponents で使う評価用の凍結プールとは"
                         "別物)")
    args = ap.parse_args()

    import os
    selfplay_workers = args.selfplay_workers or max(1, (os.cpu_count() or 2) - 2)
    eval_workers = args.eval_workers or max(1, (os.cpu_count() or 2) - 2)

    run_generations(
        n_generations=args.generations,
        games_per_gen=args.games_per_gen,
        eval_games_per_opponent=args.eval_games,
        temperature=args.temperature,
        gamma=args.gamma,
        lr=args.lr,
        selfplay_workers=selfplay_workers,
        eval_workers=eval_workers,
        opponents=args.opponents,
        out_dir=args.out_dir,
        seed=args.seed,
        selfplay_opponents=args.selfplay_opponents,
        accept_margin=args.accept_margin,
        max_relative_step=args.max_relative_step,
        base_weights=args.base_weights,
    )


if __name__ == "__main__":
    main()
