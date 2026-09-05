"""train_pool.looks_like_deckout(軌跡末尾からの近似デッキ切れ判定)の妥当性を検証する。

train_pool.py の reward シェーピング(--deckout-penalty)は looks_like_deckout の判定精度に
依存する。looks_like_deckout は「学習側の最後に記録された判断時点の self_deck_count」から
デッキ切れを推定する近似であり、正確な最終盤面を見ているわけではない(train_pool.py の
docstring 参照)。本スクリプトは、この近似がどれくらい当たるかを実測する。

比較対象(正解データ):
  blunder_metrics.py は cg のゲームループを自前で回し、試合終了時の盤面から直接
  ``final_deck_learner == 0`` を判定している(``learner_losses_deckout``)。これは近似ではなく
  厳密な判定である。

**制約(重要): 同一試合での1対1照合はできない。** cg 側の乱数(山札シャッフル・初手など)は
Python の seed では制御できない(collect_pool.py の docstring 参照。同一 seed0・同一コマンドを
複数回実行しても結果がばらつくことを実測済み)。そのため本スクリプトは
  - collect_pool.parallel_collect_pool で N 試合収集し、looks_like_deckout ベースの
    近似デッキ切れ率(学習側の負け試合のうち、山札切れとみなされた割合)を算出
  - 同じ設定(同じ学習側・同じ相手プール・同じ試合数)で blunder_metrics のロジックを使って
    厳密なデッキ切れ率を算出
という **2つの独立した収集を行い、集計された率同士を比較する** にとどめる。個々の試合を
対応づけて突き合わせることはできない。

閾値(自分の山札が何枚以下ならデッキ切れとみなすか)を 0/1/2/3 と変えたときに近似率が
どう動くかも出す(厳密判定にどの閾値が最も近いかを選ぶ材料にするため)。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pools  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402
from collect_pool import parallel_collect_pool  # noqa: E402
from blunder_metrics import parallel_collect_blunder, _sum_raw  # noqa: E402
from train_pool import looks_like_deckout, _resolve_learner_weights_path, DEFAULT_POOL  # noqa: E402

# approx(collect_pool)側と strict(blunder_metrics)側は同じ試合を再現するわけではない
# (上記 docstring の制約)ので、seed 空間を分けておく(衝突回避の意味のみ)。
APPROX_SEED0 = 700_000_000
STRICT_SEED0 = 710_000_000

THRESHOLDS = (0, 1, 2, 3)

# collect_pool の run_eval_pool と同じ「greedy 相当」温度(0.01)を使う。blunder_metrics 側は
# 常に argmax(PolicyModel.select_option)なので、これで両者の行動選択方針を揃える。
TEMPERATURE = 0.01


def approx_deckout_rates(trajs: list[dict], thresholds) -> dict:
    """collect_pool のトラジェクトリ群から、閾値ごとの近似デッキ切れ率を計算する。

    分母は「学習側が負けた軌跡数」、分子はそのうち looks_like_deckout が True の軌跡数。
    blunder_metrics.build_report の metric_c_learner_deckout(分母=学習側の負け試合数)と
    同じ定義に揃えてある。
    """
    losses = [tr for tr in trajs if tr["reward"] < 1.0]
    n_losses = len(losses)
    out = {}
    for th in thresholds:
        n_deckout = sum(1 for tr in losses if looks_like_deckout(tr, th))
        out[th] = {
            "deckout": n_deckout, "losses": n_losses,
            "rate": (n_deckout / n_losses) if n_losses else float("nan"),
        }
    return out


def strict_deckout_rate(weights_path, opponents, deck_l, per_opponent_games, seed0, workers) -> dict:
    """blunder_metrics のロジック(最終盤面を直接見る)で相手プール全体の厳密なデッキ切れ率を計算する。

    相手ごとに per_opponent_games 試合ずつ集め(collect_pool.build_tasks と同じ「相手あたり
    偶数試合」の分割に揃えるため、呼び出し側で build_tasks の per を渡す)、raw カウントを
    合算してから rate を出す。
    """
    raws = []
    for i, (name, opp_weights, deck_o) in enumerate(opponents):
        seed = seed0 + i * 1_000_000
        raw = parallel_collect_blunder(
            weights_path, opp_weights, deck_l, deck_o,
            n_games=per_opponent_games, seed0=seed, workers=workers,
        )
        raws.append(raw)
    total = _sum_raw(raws)
    n_losses = total["learner_losses"]
    n_deckout = total["learner_losses_deckout"]
    rate = (n_deckout / n_losses) if n_losses else float("nan")
    return {"deckout": n_deckout, "losses": n_losses, "rate": rate, "raw": total}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learner", default="alakazam", help="pools.LEARNER_REGISTRY のキー。既定 alakazam。")
    ap.add_argument("--opponents", default=DEFAULT_POOL,
                     help=f"カンマ区切りの相手名(pools.LEARNER_REGISTRY のキー)。既定: {DEFAULT_POOL}")
    ap.add_argument("--games", type=int, default=100, help="collect_pool 側の総試合数(既定100)。")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    opp_names = [x.strip() for x in args.opponents.split(",") if x.strip()]
    if not opp_names:
        raise ValueError(f"--opponents が空: {args.opponents!r}")

    learner_w = _resolve_learner_weights_path(args.learner, None)
    _weights_path_reg, learner_deck_csv = pools.resolve_learner(args.learner)
    deck_l = read_deck_csv_file(learner_deck_csv)
    opponents = pools.build_opponents(opp_names)

    print(f"learner={args.learner} weights={learner_w} opponents={opp_names} "
          f"games={args.games} workers={args.workers} temperature={TEMPERATURE}", flush=True)

    # --- approx: collect_pool + looks_like_deckout ---
    t0 = time.time()
    trajs, stats = parallel_collect_pool(
        str(learner_w), opponents, deck_l,
        n_games=args.games, seed0=APPROX_SEED0, temperature=TEMPERATURE, workers=args.workers,
    )
    per = stats["per"]
    approx_total_games = per * len(opponents)
    print(f"[approx/collect_pool] per_opponent={per} total_games={approx_total_games} "
          f"valid={stats['total']['valid']} errors={stats['total']['errors']} "
          f"trajs_with_steps={len(trajs)} ({time.time() - t0:.0f}s)", flush=True)

    approx = approx_deckout_rates(trajs, THRESHOLDS)
    for th in THRESHOLDS:
        r = approx[th]
        if r["losses"]:
            print(f"[approx threshold={th}] deckout {r['deckout']}/{r['losses']} = {r['rate']:.3f}",
                  flush=True)
        else:
            print(f"[approx threshold={th}] losses=0 (rate undefined)", flush=True)

    # --- strict: blunder_metrics(最終盤面を直接参照) ---
    # collect_pool と同じ「相手あたり試合数」(per)を使って母集団の規模を揃える。
    t1 = time.time()
    strict = strict_deckout_rate(str(learner_w), opponents, deck_l, per, STRICT_SEED0, args.workers)
    strict_total_games = per * len(opponents)
    print(f"[strict/blunder_metrics] per_opponent={per} total_games={strict_total_games} "
          f"valid={strict['raw']['valid']} errors={strict['raw']['errors']} "
          f"({time.time() - t1:.0f}s)", flush=True)
    if strict["losses"]:
        print(f"[strict] deckout {strict['deckout']}/{strict['losses']} = {strict['rate']:.3f}", flush=True)
    else:
        print("[strict] losses=0 (rate undefined)", flush=True)

    # --- 比較 ---
    print("", flush=True)
    print("閾値ごとの近似率 vs 厳密率(collect_pool と blunder_metrics は別試合の集計であり、"
          "対応なしの比較。docstring 参照):", flush=True)
    print(f"{'threshold':>9} | {'approx_rate':>11} | {'strict_rate':>11} | {'diff':>7}", flush=True)
    sr = strict["rate"]
    for th in THRESHOLDS:
        ar = approx[th]["rate"]
        if ar == ar and sr == sr:  # NaN チェック(NaN != NaN)
            diff_str = f"{ar - sr:+.3f}"
        else:
            diff_str = "n/a"
        ar_str = f"{ar:.3f}" if ar == ar else "n/a"
        sr_str = f"{sr:.3f}" if sr == sr else "n/a"
        print(f"{th:9d} | {ar_str:>11} | {sr_str:>11} | {diff_str:>7}", flush=True)

    print("", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
