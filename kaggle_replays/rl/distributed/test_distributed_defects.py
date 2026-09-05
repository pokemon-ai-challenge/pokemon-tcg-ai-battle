"""distributed/learner.py に移植した defect#1〜#7 修正の検証(2026-08-13)。

test_distributed.py が「配管」(世代管理・シャード検査・critic/optimizer の引き継ぎ)を
検証するのに対し、こちらは「train_pool.py/train_v3.py に入っている7件の修正が
learner.py 経由でも実際に効いているか」を、1台・少数試合の実サイクルで検証する。

学習側の初期重みには production の policy_weights.json(kamitsuorochi_ex、715次元)を使う。
test_distributed.py が使っている dragapult_ex/alakazam の素の重み(166次元、2026-08-08の
251/715次元拡張より前の旧いファイル)は Critic.from_value_net が前提とする
encoder.FEATURE_NAMES(715次元)と次元が合わず使えないため、ここでは明示的に
--learner-weights で現行の715次元重みを渡す。相手デッキは crustle にしてあり、これは
requirements-kamitsuorochi-2026-08-12.md が次の対面特化RLの対象として指定している
組み合わせそのもの。

確認する内容(すべて gen0->gen1、このrunで初めての更新 = critic warmup が発火する回):
  (a) critic warmup 後の critic がサイド差(prize margin)に単調に反応する
      (Critic.from_value_net による warm-start が壊れていないことの間接証拠。
      test_critic_warmstart.py の検証2と同じロジックを、実際に1世代分の
      warmup+joint PPO を経た後の critic に対して行う)
  (b) critic warmup が実際に発火し(5エポックぶんのログが出る)、learner.py 内の
      「warmup中は policy パラメータのnormが不変」という assert が例外を投げずに
      完走した(= 完走したこと自体が assert が通った証拠)
  (c) Elo 差(defect#7、train_v3.elo_diff)が learner.py の出力・history.jsonl に出る
  (d) KL(π‖π_BC) 項(defect#6)が0でない値としてログ・history.jsonl に残る

数分で終わる(GAMES を絞ってある)。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import common as C

HERE = Path(__file__).resolve().parent
REPO_ROOT = C.REPO_ROOT
LEARNER_WEIGHTS = REPO_ROOT / "sample_submission" / "ptcg_ai" / "learning" / "policy_weights.json"
GAMES = 16          # 1台あたり。速度優先で最小限
PROCS = 8           # 収集の並列プロセス数


def run(cmd, expect_ok=True):
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, capture_output=True, text=True)
    if expect_ok and r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(f"失敗するはずのないコマンドが失敗: {' '.join(cmd)}")
    return r


def main():
    assert LEARNER_WEIGHTS.is_file(), f"715次元の現行重みが無い: {LEARNER_WEIGHTS}"

    tmp = Path(tempfile.mkdtemp(prefix="ptcg_dist_defects_test_"))
    run_dir = tmp / "run"
    try:
        # ---------------------------------------------------------- 1. 初期化
        run(["init_run.py", "--run-id", "defects_test", "--run-dir", str(run_dir),
             "--workers", "alpha", "--games-per-worker", str(GAMES),
             "--learner-arch", "kamitsuorochi_ex", "--learner-weights", str(LEARNER_WEIGHTS),
             "--opponent-arch", "crustle"])
        assert C.model_path(run_dir, 0).exists(), "model_v0 が作られていない"
        print("1. init OK (learner=kamitsuorochi_ex opponent=crustle, 715次元BC)")

        # ---------------------------------------------------------- 2. 収集(1台)
        run(["worker.py", "--run-dir", str(run_dir), "--worker-id", "alpha",
             "--workers", str(PROCS)])
        shards = sorted(C.shard_dir(run_dir, 0).glob("*.npz"))
        assert len(shards) == 1, f"シャードが1件でない: {shards}"
        print("2. worker OK")

        # ---------------------------------------------------------- 3. learner(gen0->gen1、既定引数のまま)
        # 既定: --critic-init value_net --critic-warmup-epochs 5 --shaping-c 0.1 --kl-beta 0.02
        # このrunで初めての更新なので critic warmup が発火するはず。
        r = run(["learner.py", "--run-dir", str(run_dir), "--eval-games", "20"])
        out = r.stdout
        assert C.model_path(run_dir, 1).exists(), "model_v1 が作られていない"
        print("3. learner v0->v1 OK(既定引数、途中経過は以下)")

        hist_path = C.history_path(run_dir)
        hist = [json.loads(ln) for ln in hist_path.read_text(encoding="utf-8").splitlines()]
        assert len(hist) == 1, hist
        rec = hist[0]

        # ---------------------------------------------------------- (b) critic warmup が発火した
        warmup_lines = [ln for ln in out.splitlines() if "[critic warmup" in ln]
        assert len(warmup_lines) == 5, (
            f"warmupログが5件でない(--critic-warmup-epochs既定5): {warmup_lines}\n---stdout---\n{out}"
        )
        assert "critic warmup skipped" not in out, "初回更新なのにwarmupがskipされている"
        assert rec["critic_warmup_epochs"] == 5, rec
        print("(b) PASS: critic warmup が5エポックぶん発火し、learner.py内のassert"
              "(warmup中policyパラメータnorm不変)を例外なく通過した")
        for ln in warmup_lines:
            print("     " + ln)

        # ---------------------------------------------------------- (c) Elo差がログに出る
        assert re.search(r"Elo [+-]?\d+\.\d", out), f"Eloの出力が見つからない:\n{out}"
        assert "collect_elo_diff" in rec and isinstance(rec["collect_elo_diff"], float)
        assert "eval_elo_diff" in rec and isinstance(rec["eval_elo_diff"], float)
        print(f"(c) PASS: Elo差が標準出力とhistory.jsonlの両方に記録されている "
              f"(collect_elo_diff={rec['collect_elo_diff']:.1f} "
              f"eval_elo_diff={rec['eval_elo_diff']:.1f})")

        # ---------------------------------------------------------- (d) KLが非ゼロ
        assert rec["kl_beta"] == 0.02, rec["kl_beta"]
        assert rec["kl_to_bc"] > 0.0, f"kl_to_bc が0以下: {rec['kl_to_bc']}"
        print(f"(d) PASS: KL(π‖π_BC) が非ゼロ(kl_to_bc={rec['kl_to_bc']:.6f}, "
              f"beta={rec['kl_beta']})")

        # ---------------------------------------------------------- (a) warmup後のcriticがサイド差に単調反応
        import torch
        sys.path.insert(0, str(REPO_ROOT / "kaggle_replays" / "rl"))
        from train_v3 import Critic, OPP_PRIZE_REMAINING_IDX, SELF_PRIZE_REMAINING_IDX  # noqa: E402
        from ptcg_ai.learning import encoder as _enc  # noqa: E402

        assert rec["critic_init"] == "value_net", rec["critic_init"]
        base_payload = json.loads(LEARNER_WEIGHTS.read_text(encoding="utf-8"))
        std = base_payload["standardization"]
        st = torch.load(C.trainer_state_path(run_dir, 1), map_location="cpu", weights_only=True)
        # value_weights_v251.json は hidden=[64,16]・sigmoid出力(Critic.from_value_net が
        # そこから引き継ぐ形)なので、同じ形の Critic を作って state_dict をロードする。
        critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"],
                        hidden=(64, 16), output_activation="sigmoid")
        critic.load_state_dict(st["critic"])
        critic.eval()

        def row(self_prize, opp_prize):
            r = [0.0] * len(_enc.FEATURE_NAMES)
            r[SELF_PRIZE_REMAINING_IDX] = float(self_prize)
            r[OPP_PRIZE_REMAINING_IDX] = float(opp_prize)
            return r

        with torch.no_grad():
            v_ahead = critic(torch.tensor([row(1, 5)], dtype=torch.float32)).item()
            v_even = critic(torch.tensor([row(3, 3)], dtype=torch.float32)).item()
            v_behind = critic(torch.tensor([row(5, 1)], dtype=torch.float32)).item()
        print(f"(a) value(自分優勢 1-5)={v_ahead:.4f}  value(互角 3-3)={v_even:.4f}  "
              f"value(自分劣勢 5-1)={v_behind:.4f}")
        assert v_ahead > v_even > v_behind, (
            f"warmup後のcriticがサイド差の方向に反応していない: "
            f"ahead={v_ahead} even={v_even} behind={v_behind}"
        )
        print("(a) PASS: 1世代ぶんのwarmup+joint PPOを経た後もcriticはサイド差に単調反応する"
              "(value_net由来の事前知識が壊れていない)")

        print("\n全項目 PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
