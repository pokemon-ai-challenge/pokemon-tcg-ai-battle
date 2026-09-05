"""相互鍛錬ループ(段階III): モデル集合を巡回させ、互いを相手として PPO で鍛える。

段階II で「相手を鍛え上げればプール学習は固定相手学習と同等」まで確認できた。
本スクリプトはその先、複数モデルを世代ごとに巡回させて互いを強くしていくループを
1プロセス内(Kaggle の1ノートブックで完結)で回すドライバ。

    各世代 g:
      for 学習対象 in モデル集合(巡回):
          相手プール = 他モデルの現行版 + 過去世代のチェックポイント(全モデル分)
          収集 -> PPO 更新 -> チェックポイント保存

過去世代を相手プールに残すのは、最新版だけを相手にすると方策がじゃんけん状態になり
収束しないため(state.json / run-dir/gen<N>/ に全世代分のチェックポイントを残す)。

学習アルゴリズムやプール収集ロジックは一切再実装しない。train_pool.py を
「sys.argv を組み立てて import して呼ぶ」形で繰り返し呼び出すことで実現する。
train_pool.py / pools.py の既定挙動は変えていない(pools.load_extra_registry と
train_pool.py の --extra-registry は追加のみで、省略時は今までと同じ)。

評価用の相手プール(--eval-opponents)は常に固定(凍結・BC版4種)で、学習用プールが
世代ごとに変わっても比較可能な基準を保つ。これは train_pool.py の --eval-opponents と
同じ思想であり、意図的に train_league.py のトップレベル引数として独立させてある。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_policy import TorchOptionPolicy  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402
import pools  # noqa: E402
import train_pool  # noqa: E402

WDIR = pools.WDIR

DEFAULT_LEARNERS = "alakazam,marnie_grimmsnarl_ex,crustle,archaludon_ex"
DEFAULT_EVAL_OPPONENTS = "alakazam,crustle,marnie_grimmsnarl_ex,archaludon_ex"

GEN_EVAL_SEED0 = 900_000_000


def _parse_names(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def resolve_initial_weights(name: str, init_from: str) -> Path:
    """学習側の初期重みを解決する。

    init_from == "k60": 段階Iで自分のミラーで60イテレーション鍛えた版
    (pools.LEARNER_REGISTRY の "<name>_k60" キー)を使う。
    init_from == "bc": pools.LEARNER_REGISTRY の既定(BC版)を使う。alakazam の既定は
    weights_path=None(production の policy_weights.json)なので、train_pool.py の
    _resolve_learner_weights_path と同じ規則で解決する。
    """
    if init_from == "k60":
        k60_name = f"{name}_k60"
        if k60_name not in pools.LEARNER_REGISTRY:
            available = ", ".join(sorted(pools.LEARNER_REGISTRY))
            raise ValueError(
                f"--init-from k60 だが {k60_name!r} が pools.LEARNER_REGISTRY に無い"
                f"(学習側 {name!r})。利用可能なキー: {available}"
            )
        weights_path, _deck_csv = pools.resolve_learner(k60_name)
        return Path(weights_path)
    if init_from == "bc":
        weights_path, _deck_csv = pools.resolve_learner(name)
        if weights_path is None:
            return WDIR / "policy_weights.json"
        return Path(weights_path)
    raise ValueError(f"未知の --init-from: {init_from!r} (k60 か bc)")


def state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def history_path(run_dir: Path) -> Path:
    return run_dir / "history.json"


def save_state(run_dir: Path, state: dict) -> None:
    state_path(run_dir).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def save_history(run_dir: Path, history: list) -> None:
    history_path(run_dir).write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def run_generation_eval(state, learners, eval_opponents, eval_names, deck_cache, gen_dir, n_games, seed0, workers, device):
    """全学習側の現行版を、凍結された eval_opponents に対して評価する。

    train_pool.run_eval_pool をそのまま再利用する(評価ロジックを再実装しない)。
    """
    result = {}
    for learner in learners:
        wpath = Path(state["current_weights"][learner])
        arch = pools.LEARNER_REGISTRY[learner][1]
        deck_l = deck_cache[learner]
        base_payload = json.loads(wpath.read_text(encoding="utf-8"))
        policy = TorchOptionPolicy.from_json(wpath).float().to(device)
        tmp_path = gen_dir / f"_tmp_eval_{learner}.json"
        wr, w, v, per_opp = train_pool.run_eval_pool(
            policy, base_payload, tmp_path, eval_opponents, deck_l, n_games, seed0, workers,
        )
        result[learner] = {"winrate": wr, "wins": w, "valid": v, "per_opponent": per_opp}
        print(f"    eval[{learner}] (arch={arch}) {w}/{v} = {wr:.3f}", flush=True)
        tmp_path.unlink(missing_ok=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learners", default=DEFAULT_LEARNERS,
                     help=f"カンマ区切りの学習側名(pools.LEARNER_REGISTRY のキー)。既定: {DEFAULT_LEARNERS}")
    ap.add_argument("--init-from", default="k60", choices=["k60", "bc"],
                     help="各学習側の初期重み。k60 なら policy_weights_<arch>_pool_k60.json、"
                          "bc なら BC の既定重み。既定 k60。")
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--iters-per-turn", type=int, default=5,
                     help="1世代あたり1モデルが回す PPO イテレーション数。")
    ap.add_argument("--past-gens", type=int, default=2,
                     help="相手プールに含める過去世代の数(直近から)。")
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-opponents", default=DEFAULT_EVAL_OPPONENTS,
                     help="凍結。変更しないこと。学習用プールが世代ごとに変わっても比較可能な"
                          f"基準を保つための固定4種。既定: {DEFAULT_EVAL_OPPONENTS}")
    ap.add_argument("--eval-games", type=int, default=200,
                     help="世代の切れ目ごとの凍結プール評価に使う試合数。")
    ap.add_argument("--final-eval-games", type=int, default=1200,
                     help="最終世代(--generations に到達したとき)の凍結プール評価に使う試合数。")
    ap.add_argument("--run-dir", default=None,
                     help="既定: kaggle_replays/rl/league_runs/<tag>")
    ap.add_argument("--tag", default="league")
    ap.add_argument("--time-budget-hours", type=float, default=10.0,
                     help="これを超えそうなら世代の切れ目で打ち切って状態を保存する(Kaggleは12時間上限)。"
                          "判定は各世代が完了した直後に行う(世代の途中では止めない)。")
    ap.add_argument("--resume", action="store_true",
                     help="既存の --run-dir から続きを回す。")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    learners = _parse_names(args.learners)
    eval_names = _parse_names(args.eval_opponents)
    if len(learners) < 1:
        raise ValueError(f"--learners が空: {args.learners!r}")
    if not eval_names:
        raise ValueError(f"--eval-opponents が空: {args.eval_opponents!r}")

    run_dir = Path(args.run_dir) if args.run_dir else (_HERE / "league_runs" / args.tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not state_path(run_dir).exists():
            raise FileNotFoundError(f"--resume 指定だが state.json が無い: {state_path(run_dir)}")
        state = json.loads(state_path(run_dir).read_text(encoding="utf-8"))
        if state["learners"] != learners:
            print(f"[resume] WARNING: --learners({learners}) != state.json の learners"
                  f"({state['learners']})。state.json 側を優先する。", flush=True)
            learners = state["learners"]
        history = json.loads(history_path(run_dir).read_text(encoding="utf-8")) if history_path(run_dir).exists() else []
        start_gen = state["completed_generations"]
        print(f"[resume] run_dir={run_dir} completed_generations={start_gen} -> "
              f"--generations={args.generations} まで続行", flush=True)
    else:
        if state_path(run_dir).exists():
            raise FileExistsError(
                f"run-dir に既存の state.json がある(上書き事故防止のため中断): {state_path(run_dir)}\n"
                "続きから回すなら --resume を指定すること。"
            )
        current_weights = {}
        for name in learners:
            w = resolve_initial_weights(name, args.init_from)
            current_weights[name] = str(w)
        state = {
            "learners": learners,
            "init_from": args.init_from,
            "past_gens": args.past_gens,
            "completed_generations": 0,
            "current_weights": current_weights,
            "checkpoint_history": {name: [] for name in learners},
        }
        history = []
        save_state(run_dir, state)
        save_history(run_dir, history)
        start_gen = 0
        print(f"[init] run_dir={run_dir} learners={learners} init_from={args.init_from}", flush=True)
        for name, w in current_weights.items():
            print(f"    {name}: {w}", flush=True)

    deck_cache = {name: read_deck_csv_file(pools.resolve_learner(name)[1]) for name in learners}
    eval_opponents = pools.build_opponents(eval_names)  # 凍結。以後変更しない。

    t0 = time.time()
    for gen in range(start_gen, args.generations):
        gen_dir = run_dir / f"gen{gen}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== generation {gen} start ({time.time() - t0:.0f}s elapsed) ===", flush=True)

        for turn_idx, learner in enumerate(learners):
            arch = pools.LEARNER_REGISTRY[learner][1]

            # 相手プール = 他の学習側の現行版 + 過去 --past-gens 世代分のチェックポイント(全モデル分)。
            extra = {}
            for other in learners:
                if other == learner:
                    continue
                other_arch = pools.LEARNER_REGISTRY[other][1]
                extra[f"{other}_cur"] = [state["current_weights"][other], other_arch]
            recent_from = max(0, gen - args.past_gens)
            for m in learners:
                m_arch = pools.LEARNER_REGISTRY[m][1]
                for ck in state["checkpoint_history"][m]:
                    if recent_from <= ck["gen"] < gen:
                        extra[f"{m}_g{ck['gen']}"] = [ck["path"], m_arch]

            if not extra:
                raise RuntimeError(
                    f"学習側 {learner!r} の相手プールが空(世代 {gen}、他の学習側なし、過去世代なし)。"
                    "--learners を2体以上にするか、世代を進めて過去チェックポイントができるのを待つこと。"
                )

            extra_registry_path = gen_dir / f"extra_registry_{learner}.json"
            extra_registry_path.write_text(json.dumps(extra, indent=2, ensure_ascii=False), encoding="utf-8")
            train_opp_names = list(extra.keys())

            print(f"[gen {gen}] turn {turn_idx + 1}/{len(learners)}: learner={learner} arch={arch} "
                  f"opponents={train_opp_names}", flush=True)

            tag = f"{args.tag}_g{gen}_{learner}"
            argv = [
                "train_pool.py",
                "--learner", learner,
                "--learner-weights", state["current_weights"][learner],
                "--train-opponents", ",".join(train_opp_names),
                "--eval-opponents", args.eval_opponents,
                "--extra-registry", str(extra_registry_path),
                "--iters", str(args.iters_per_turn),
                "--games-per-iter", str(args.games_per_iter),
                "--eval-games", str(args.eval_games),
                "--eval-every", str(max(args.iters_per_turn, 1)),
                "--final-eval-games", str(args.eval_games),
                "--workers", str(args.workers),
                "--tag", tag,
                "--force",
            ]
            old_argv = sys.argv
            sys.argv = argv
            try:
                train_pool.main()
            finally:
                sys.argv = old_argv

            produced = WDIR / f"policy_weights_{learner}_pool_{tag}.json"
            if not produced.exists():
                raise RuntimeError(f"train_pool.main() が出力を作らなかった: {produced}")
            dest = gen_dir / f"policy_weights_{arch}.json"
            shutil.copyfile(produced, dest)

            state["current_weights"][learner] = str(dest)
            state["checkpoint_history"][learner].append({"gen": gen, "path": str(dest)})
            save_state(run_dir, state)
            print(f"[gen {gen}] {learner} done -> {dest} ({time.time() - t0:.0f}s elapsed)", flush=True)

        # 世代の終わりに、全学習側を凍結プールで評価する。
        is_final_gen = (gen == args.generations - 1)
        n_eval_games = args.final_eval_games if is_final_gen else args.eval_games
        print(f"[gen {gen}] frozen-pool eval (n_games={n_eval_games}, final={is_final_gen})", flush=True)
        gen_eval = run_generation_eval(
            state, learners, eval_opponents, eval_names, deck_cache, gen_dir,
            n_eval_games, GEN_EVAL_SEED0 + gen * 1000, args.workers, device,
        )

        history.append({"generation": gen, "elapsed_s": time.time() - t0, "eval": gen_eval})
        save_history(run_dir, history)

        state["completed_generations"] = gen + 1
        save_state(run_dir, state)

        elapsed_h = (time.time() - t0) / 3600
        if elapsed_h >= args.time_budget_hours and gen + 1 < args.generations:
            print(f"[time budget] {elapsed_h:.2f}h >= --time-budget-hours {args.time_budget_hours}h. "
                  f"世代 {gen} の切れ目で打ち切り。state.json は保存済み。"
                  f"--resume --run-dir {run_dir} --tag {args.tag} --generations {args.generations} "
                  "で続きから回せる。", flush=True)
            return

    print(f"done. completed_generations={state['completed_generations']} run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
