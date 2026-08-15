#!/usr/bin/env python3
"""policy_positions.jsonl.gz を **エピソード単位**で間引き、BCのデータスケーリング曲線用の
縮小データセットを作る(bc-data-scaling-curve タスク1)。

なぜエピソード単位か: 同一試合内の決定点は強く相関している(同じ盤面展開・同じプレイヤーの
癖を反映)ため、決定点単位でランダムに間引いても「リプレイを何試合集めたか」の模擬にならない。
必ず episode_id 単位でサンプリングする。

なぜ test split のエピソードを常に全部残すか: train/val のエピソードだけを間引くことで、
Top-1 などの評価が全条件で同じ test set 上になり、fraction 間で直接比較できる
(test set 自体が条件ごとに変わると、data scaling の効果と test set の違いが混同される)。

split の判定式は build_features.py の split_for_episode と同一(md5(episode_id) % 100 ->
<80 train(0) / <90 val(1) / それ以外 test(2))。ここでは build_features.py を import せず
(cg エンジン等の重い依存を避けるため)、同じ式を独立に再実装している。

選び方: train/val エピソードのみを対象に、ソート済みID列に対して
``random.Random(seed).sample`` で選ぶ(決定的・再現可能)。

入出力は jsonl.gz。2パス構成:
  pass1: 全行を読み episode_id -> split を集計
  pass2: 残すエピソードに属する行だけを書き出す

使い方:
    python subsample_episodes.py --in <in.jsonl.gz> --out <out.jsonl.gz> --fraction 0.25 --seed 42
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_PROGRESS_EVERY = 100_000


def split_for_episode(episode_id: str) -> int:
    """episode_id の md5 ハッシュから train(0)/val(1)/test(2) を決める
    (build_features.py の split_for_episode と同一の固定式)。"""
    h = int(hashlib.md5(episode_id.encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return 0
    if h < 90:
        return 1
    return 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument(
        "--fraction", type=float, required=True,
        help="train/val エピソードのうち残す割合(0 < fraction <= 1)。test エピソードは常に全件残すので対象外。",
    )
    parser.add_argument("--seed", type=int, required=True, help="サンプリングの乱数シード(再現用)")
    args = parser.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise ValueError(f"--fraction は (0, 1] の範囲で指定してください: {args.fraction}")

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- pass1: 全行を読み、episode_id -> split を集計する ---
    print(f"pass1: episode_id を収集中: {in_path}", file=sys.stderr)
    t0 = time.time()
    episode_split: dict[str, int] = {}
    n_lines = 0
    with gzip.open(in_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            row = json.loads(line)
            ep = str(row["episode_id"])
            sp = split_for_episode(ep)
            prev = episode_split.get(ep)
            if prev is None:
                episode_split[ep] = sp
            elif prev != sp:
                # split は episode_id のみのハッシュで決まる純関数なので、同一エピソード内で
                # 行ごとに割れることは仕様上あり得ない。データ破損の兆候として止める(fail-fast)。
                raise ValueError(
                    f"episode_id={ep} の split が行によって食い違う: {prev} vs {sp}(データ破損の疑い)"
                )
            if n_lines % _PROGRESS_EVERY == 0:
                print(f"  pass1: {n_lines} 行 (経過 {time.time() - t0:.1f}s)", file=sys.stderr)
    print(
        f"pass1完了: {n_lines} 行 / {len(episode_split)} エピソード (経過 {time.time() - t0:.1f}s)",
        file=sys.stderr,
    )

    test_episodes = {ep for ep, sp in episode_split.items() if sp == 2}
    # sorted() でファイル内の出現順(≒非決定的な dict/set 順序)に依存しない決定的な列にしてから
    # サンプリングする(同じ seed なら OS・実行順によらず同じ結果になるように)。
    nontest_episodes = sorted(ep for ep, sp in episode_split.items() if sp != 2)

    n_sample = round(len(nontest_episodes) * args.fraction)
    n_sample = max(0, min(n_sample, len(nontest_episodes)))
    rng = random.Random(args.seed)
    sampled_nontest = set(rng.sample(nontest_episodes, n_sample))

    keep_episodes = test_episodes | sampled_nontest

    print(
        f"train/valエピソード間引き: {len(nontest_episodes)} -> {len(sampled_nontest)} "
        f"(fraction={args.fraction}, seed={args.seed}) / testエピソード {len(test_episodes)} は全件保持",
        file=sys.stderr,
    )
    print(f"元のエピソード数 {len(episode_split)} -> 残したエピソード数 {len(keep_episodes)}")

    # --- pass2: keep_episodes に含まれる行だけ書き出す ---
    print(f"pass2: 出力中: {out_path}", file=sys.stderr)
    t1 = time.time()
    n_kept_lines = 0
    n_seen2 = 0
    with gzip.open(in_path, "rt", encoding="utf-8") as f_in, gzip.open(
        out_path, "wt", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_seen2 += 1
            row = json.loads(line)
            ep = str(row["episode_id"])
            if ep in keep_episodes:
                f_out.write(line)
                f_out.write("\n")
                n_kept_lines += 1
            if n_seen2 % _PROGRESS_EVERY == 0:
                print(f"  pass2: {n_seen2} 行走査済み (経過 {time.time() - t1:.1f}s)", file=sys.stderr)

    if n_seen2 != n_lines:
        # pass1/pass2 は同じファイルを読んでいるので行数は必ず一致するはず。
        raise ValueError(f"pass1とpass2で行数が食い違う: {n_lines} vs {n_seen2}(ファイルが途中で変わった?)")

    elapsed2 = time.time() - t1
    print(f"pass2完了: 決定点数 {n_lines} -> {n_kept_lines} (経過 {elapsed2:.1f}s)", file=sys.stderr)
    print(f"決定点数: {n_lines} -> {n_kept_lines}")

    out_size_mb = out_path.stat().st_size / 1e6
    print(f"書き出し完了: {out_path} ({out_size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
