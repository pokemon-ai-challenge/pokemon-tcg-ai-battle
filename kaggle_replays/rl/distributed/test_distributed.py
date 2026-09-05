"""分散 Self-Play の一巡を1台で検証する(worker 2台ぶんを順番に回す)。

確認する内容:
  1. init -> worker×2 -> learner -> 世代が進む、が通る
  2. 2台のシード範囲が重ならない
  3. 古い世代のシャードが必ず弾かれる(PPO に混ざらない)
  4. 同じ世代を2回更新できない
  5. critic / optimizer の状態が世代をまたいで引き継がれる
  6. 世代をまたいだシャードの取り違えを、モデルのハッシュで検出できる

試合数を絞ってあるので数分で終わる。学習の中身(強くなるか)ではなく、配管の検査。

2026-08-13 注記1: ここで使っていた dragapult_ex/alakazam の素の重み
(policy_weights_dragapult_ex.json / policy_weights_alakazam.json)は166次元で、
2026-08-08以降の251/715次元への拡張より前の旧いファイル。現行の715次元 encoder.FEATURE_NAMES
の下では ``PolicyModel.is_ready`` が False になり(次元ガード)、``collect_parallel._init_worker2``
が起動直後に例外を投げる。multiprocessing.Pool は initializer が例外を投げると failed worker を
無限に再spawnし続ける実装になっており、実行してもエラーが延々スパムされて実質ハングする
(このテストの範囲では新規に踏んだ挙動。7件の修正とは無関係の、コーパス側の旧ファイル起因の
既存の落とし穴)。そのためここでは production の現行715次元重み(PRODUCTION_WEIGHTS)を
明示的な --learner-weights として渡す。デッキ(アーキタイプ名)は dragapult_ex/alakazam の
ままで変えていない(配管テストなので重みとデッキの対応が実際のアーキタイプとズレていても
問題ない)。

2026-08-13 注記2: defect#3(Critic.from_value_net)と defect#5(報酬シェーピング)は
現行の715次元 encoder.FEATURE_NAMES を前提にしている。注記1の修正で715次元重みを使うように
なったため両方とも動くはずだが、この配管テストの目的はPPO配管の検証であって7件の修正の
検証ではないため、既定のまま(有効)にして「7件の修正が入っていてもこの配管テストが壊れない」
ことも合わせて確認する形にしている。7件の修正そのものの検証は test_distributed_defects.py
の役目。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import common as C

HERE = Path(__file__).resolve().parent
REPO_ROOT = C.REPO_ROOT
PRODUCTION_WEIGHTS = REPO_ROOT / "sample_submission" / "ptcg_ai" / "learning" / "policy_weights.json"
GAMES = 12          # 1台あたり。速度優先で最小限
PROCS = 8           # 収集の並列プロセス数


def check_onpolicy_consistency(run_dir: Path, gen: int):
    """収集時の方策と、更新側が再計算する方策が一致しているかを数値で確認する。

    PPO の重要度比は π_新(a) / π_収集(a)。更新前は両者が同じ方策のはずなので、比は
    ちょうど 1.0 でなければならない。ここがずれていると、1エポック目から誤った勾配が
    かかる(worker が softmax(scores/T) で行動したのに更新側が T を無視している、
    といった食い違いはこの検査で必ず出る)。
    """
    import torch
    from learner import build_batch, policy_logp_entropy
    from torch_policy import TorchOptionPolicy

    run_cfg = C.load_run(run_dir)
    shards = [C.read_shard(p) for p in sorted(C.shard_dir(run_dir, gen).glob("*.npz"))]
    batch = build_batch(shards, "cpu")
    policy = TorchOptionPolicy.from_json(C.model_path(run_dir, gen)).float()
    with torch.no_grad():
        logp, _ = policy_logp_entropy(policy, batch, float(run_cfg["temperature"]))
    max_logp_diff = float((logp - batch["old_logp"]).abs().max())
    max_ratio_dev = float((torch.exp(logp - batch["old_logp"]) - 1.0).abs().max())
    return max_logp_diff, max_ratio_dev


def run(cmd, expect_ok=True):
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, capture_output=True, text=True)
    if expect_ok and r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise AssertionError(f"失敗するはずのないコマンドが失敗: {' '.join(cmd)}")
    return r


def main():
    assert PRODUCTION_WEIGHTS.is_file(), f"715次元の現行重みが無い: {PRODUCTION_WEIGHTS}"
    tmp = Path(tempfile.mkdtemp(prefix="ptcg_dist_test_"))
    run_dir = tmp / "run"
    try:
        # ---------------------------------------------------------- 1. 初期化
        # --learner-weights で現行の715次元重みを明示する(上の注記1参照。dragapult_ex 自身の
        # 素の重みファイルは166次元で is_ready=False になり multiprocessing.Pool がハングする)。
        run(["init_run.py", "--run-id", "testrun", "--run-dir", str(run_dir),
             "--workers", "alpha", "beta", "--games-per-worker", str(GAMES),
             "--learner-arch", "dragapult_ex", "--learner-weights", str(PRODUCTION_WEIGHTS),
             "--opponent-arch", "alakazam"])
        assert C.model_path(run_dir, 0).exists(), "model_v0 が作られていない"
        print("1. init OK")

        # ---------------------------------------------------------- 2. 収集
        for wid in ("alpha", "beta"):
            run(["worker.py", "--run-dir", str(run_dir), "--worker-id", wid,
                 "--workers", str(PROCS)])
        shards = sorted(C.shard_dir(run_dir, 0).glob("*.npz"))
        assert len(shards) == 2, f"シャードが2件でない: {shards}"

        metas = {}
        for p in shards:
            _, m = C.read_shard(p)
            metas[m["worker_id"]] = m
        ra = metas["alpha"]["seed_range"]; rb = metas["beta"]["seed_range"]
        assert ra[1] <= rb[0] or rb[1] <= ra[0], f"シード範囲が重なっている: {ra} {rb}"
        assert metas["alpha"]["generation"] == 0 and metas["beta"]["generation"] == 0
        print(f"2. worker×2 OK (シード範囲 {ra} / {rb} 重なりなし)")

        # 未登録の worker-id は拒否される(シードの一意性が壊れるため)
        r = run(["worker.py", "--run-dir", str(run_dir), "--worker-id", "gamma"],
                expect_ok=False)
        assert r.returncode != 0, "未登録の worker-id が通ってしまった"
        print("3. 未登録 worker-id の拒否 OK")

        # -------------------------------------------- 3.5 収集時と更新時の方策の一致
        d, dev = check_onpolicy_consistency(run_dir, 0)
        assert d < 1e-3, f"収集時の logprob と更新側の再計算がずれている(最大 {d:.2e})"
        assert dev < 1e-3, f"重要度比が 1 から外れている(最大 {dev:.2e})"
        print(f"3.5 収集時と更新時の方策の一致 OK "
              f"(logp差 最大{d:.2e} / 重要度比の1からのずれ 最大{dev:.2e})")

        # ---------------------------------------------------------- 3. 世代0のシャードを退避(あとで古いデータとして使う)
        stale = tmp / "stale_v0_alpha.npz"
        shutil.copyfile(shards[0], stale)

        # ---------------------------------------------------------- 4. 集約 -> v1
        r = run(["learner.py", "--run-dir", str(run_dir)])
        assert C.model_path(run_dir, 1).exists(), "model_v1 が作られていない"
        assert C.trainer_state_path(run_dir, 1).exists(), "trainer_v1.pt が無い"
        assert C.load_run(run_dir)["generation"] == 1, "世代が進んでいない"
        assert not list(C.shard_dir(run_dir, 0).glob("*.npz")), "使ったシャードが残っている"
        assert list(C.consumed_dir(run_dir, 0).glob("*.npz")), "consumed へ退避されていない"
        print("4. learner v0->v1 OK(シャードは consumed へ退避)")

        # ---------------------------------------------------------- 5. 同じ世代の二重更新を拒否
        (C.shard_dir(run_dir, 0)).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stale, C.shard_dir(run_dir, 0) / "alpha.npz")
        run_json = C.load_run(run_dir)
        run_json["generation"] = 0                      # 世代を巻き戻してみる
        C.save_run(run_dir, run_json)
        r = run(["learner.py", "--run-dir", str(run_dir)], expect_ok=False)
        assert r.returncode != 0, "v1 があるのに v0 の更新が通ってしまった"
        assert "すでに存在する" in r.stdout + r.stderr
        run_json["generation"] = 1
        C.save_run(run_dir, run_json)
        print("5. 同一世代の二重更新の拒否 OK")

        # ---------------------------------------------------------- 6. 古い世代のシャードは弾かれる
        C.shard_dir(run_dir, 1).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stale, C.shard_dir(run_dir, 1) / "alpha.npz")   # 中身は v0 の経験
        r = run(["learner.py", "--run-dir", str(run_dir), "--status"])
        assert "NG" in r.stdout, "古いシャードが status で弾かれていない"
        r = run(["learner.py", "--run-dir", str(run_dir)], expect_ok=False)
        assert r.returncode != 0, "古い世代のデータだけで更新が通ってしまった"
        assert "世代が違う" in r.stdout + r.stderr, r.stdout + r.stderr
        print("6. 古い世代のシャードの除外 OK")

        # ---------------------------------------------------------- 7. 中身だけ差し替えたモデルの検出
        # 世代番号は v1 のまま、モデルの中身を別物にして収集したデータを混ぜる
        (C.shard_dir(run_dir, 1) / "alpha.npz").unlink()
        for wid in ("alpha", "beta"):
            run(["worker.py", "--run-dir", str(run_dir), "--worker-id", wid,
                 "--workers", str(PROCS)])
        # v1 のモデルを書き換える(= worker が使ったものと中身が違う状態)
        p1 = C.model_path(run_dir, 1)
        payload = json.loads(p1.read_text(encoding="utf-8"))
        payload["layers"][-1]["bias"][0] += 0.01
        p1.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        r = run(["learner.py", "--run-dir", str(run_dir)], expect_ok=False)
        assert "モデルの中身が違う" in r.stdout + r.stderr, r.stdout + r.stderr
        print("7. モデル中身の食い違い検出 OK(世代番号だけでは見抜けないケース)")

        # ---------------------------------------------------------- 8. 正しく戻して v1->v2、状態の引き継ぎを確認
        payload["layers"][-1]["bias"][0] -= 0.01
        p1.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        for wid in ("alpha", "beta"):     # ハッシュが変わったので収集し直す
            run(["worker.py", "--run-dir", str(run_dir), "--worker-id", wid,
                 "--workers", str(PROCS)])
        r = run(["learner.py", "--run-dir", str(run_dir)])
        assert "学習状態を引き継ぎ" in r.stdout, r.stdout
        assert C.model_path(run_dir, 2).exists()
        assert C.load_run(run_dir)["generation"] == 2
        print("8. v1->v2 OK(critic / optimizer の引き継ぎを確認)")

        hist = [json.loads(l) for l in C.history_path(run_dir).read_text(encoding="utf-8").splitlines()]
        assert len(hist) == 2, hist
        print(f"\n全項目 PASS  history={len(hist)}世代  "
              f"最終世代 v{C.load_run(run_dir)['generation']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
