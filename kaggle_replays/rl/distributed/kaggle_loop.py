"""Kaggle経由で collect(push_kaggle.py) -> learn(learner.py) を目標世代まで無人で繰り返すドライバ。

2026-08-14、ogerpon/lucario の対面特化RL(step3)で使うために作成
(scratchpad の使い捨てスクリプトから昇格。同じパターンを次の対面でもそのまま使う想定なので
repo に置く)。ローカル収集版(worker.py を直接呼ぶだけの使い捨てループ)とは違い、
push_kaggle.py がKaggle側の完了を待ってから回収するぶん1イテレーションが長い
(データセットアップロード + 反映待ち90s + カーネル実行数分)。

**このスクリプト自身はローカルのPythonプロセスとして動き続ける必要がある**
(Kaggle側は試合生成だけをクラウドで行い、それを見張って回収・PPO更新・次世代の投入をするのは
このループ本体。PCをシャットダウンするとKaggle側の実行中カーネルはおそらくそのまま完走するが、
回収も次世代の投入も止まる。止まっても `run.json` の世代番号ベースで安全に再開できる——
このスクリプトを同じ引数でもう一度実行するだけでよい)。

並行して複数run(ogerpon/lucario等)を回すときは、run ごとに別の --dataset-slug/--kernel-slug を
push_kaggle.py に渡すこと(同じスラグのまま2本走らせるとKaggle上で同じDataset/Kernelを
取り合って競合する。push_kaggle.py の該当オプションのdocstring参照)。

使い方:
    python kaggle_loop.py <run_dir> <target_generation> [dataset_slug] [kernel_slug]

例:
    python kaggle_loop.py ../runs/kamitsuorochi_vs_ogerpon_teal_ex_v1 25
    python kaggle_loop.py ../runs/kamitsuorochi_vs_mega_lucario_ex_v1 25 \\
        ptcg-distributed-selfplay-lucario ptcg-worker-kaggle-lucario
"""
import json
import subprocess
import sys
from pathlib import Path


def main():
    run_dir = Path(sys.argv[1]).resolve()
    target = int(sys.argv[2])
    dataset_slug = sys.argv[3] if len(sys.argv) > 3 else None
    kernel_slug = sys.argv[4] if len(sys.argv) > 4 else None

    dist_dir = run_dir.parents[1] / "distributed"  # .../rl/runs/<id> -> .../rl/distributed
    notebooks_dir = dist_dir / "notebooks"

    def run_json():
        return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    def history_tail():
        p = run_dir / "history.jsonl"
        if not p.exists():
            return None
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        return json.loads(lines[-1]) if lines else None

    while True:
        gen = run_json()["generation"]
        if gen >= target:
            print(f"=== target generation {target} に到達。停止。 ===", flush=True)
            break

        print(f"\n=== gen {gen}: Kaggleへ収集投入 ===", flush=True)
        cmd = [sys.executable, "push_kaggle.py", "--run-dir", str(run_dir), "--allow-dirty"]
        if dataset_slug:
            cmd += ["--dataset-slug", dataset_slug]
        if kernel_slug:
            cmd += ["--kernel-slug", kernel_slug]
        r = subprocess.run(cmd, cwd=notebooks_dir, capture_output=True, text=True)
        print(r.stdout[-3000:])
        if r.returncode != 0:
            print("--- push_kaggle.py stderr ---")
            print(r.stderr[-3000:])
            print(f"=== gen {gen}: push_kaggle.py が失敗(exit {r.returncode})。停止。 ===", flush=True)
            break

        print(f"=== gen {gen}: 集約+PPO更新開始 ===", flush=True)
        r = subprocess.run([sys.executable, "learner.py", "--run-dir", str(run_dir)],
                           cwd=dist_dir, capture_output=True, text=True)
        print(r.stdout[-2000:])
        if r.returncode != 0:
            print("--- learner.py stderr ---")
            print(r.stderr[-3000:])
            print(f"=== gen {gen}: learner.py が失敗(exit {r.returncode})。停止。 ===", flush=True)
            break

        h = history_tail()
        wr = h.get("collect_winrate") if h else None
        print(f"=== gen {gen} 完了。collect_winrate={wr} ===", flush=True)

    print("=== ループ終了 ===", flush=True)


if __name__ == "__main__":
    main()
