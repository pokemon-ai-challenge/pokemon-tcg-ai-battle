"""デッキ候補どうしの直接対決(head-to-head)を大規模に回す測定本体(ローカル / Kaggle 共通)。

問い: 1枚差し替えたデッキ候補は、現行デッキ(baseline)に直接当てて勝ち越せるのか。

``kaggle_replays/_wall_deck_ab.py`` は「候補デッキ vs 各アーキタイプ」だったのに対し、
こちらは **候補デッキ vs baseline デッキのミラー**(エージェント・config・重みは完全に同一で、
違うのは60枚だけ)。両者が同じ思考をするので、勝率のズレはデッキ差だけに帰属する。

群(``--arms``)は候補デッキ名。相手は常に ``--baseline``(既定 baseline)。
``--arms`` に baseline 自身を入れると **self-mirror の対照群**(理論値 50%)になり、
ハーネスの健全性・ノイズ床の実測に使える。

先後は index の偶奇で半々。seed は ``seed_base + index`` で **群をまたいで同一列**なので、
候補どうし(aceburn vs bulu)は対応のある比較(McNemar)もできる
(ただし cg のシャッフルは開幕以降 seed 制御外なので完全な CRN ではない。参考値)。

出力は JSONL(1試合1行)を逐次 flush。カーネルが時間切れになってもそこまでは残る。

使い方(ローカル・スモーク):
    python kaggle_replays/kaggle_run/deck_h2h.py --games 4 --workers 2 \
        --arms aceburn --out C:/tmp/h2h_smoke.jsonl

使い方(Kaggle カーネル内。push_kernel_deck.py が生成するスクリプトが実行する):
    python <repo>/kaggle_replays/kaggle_run/deck_h2h.py --repo-root <repo> \
        --arms aceburn,bulu --games 1600 --workers 4 --seed-base 1300000 \
        --out /kaggle/working/h2h_s0.jsonl

注意(既知の罠):
* multiprocessing は **spawn 前提**。必ずスクリプトとして実行すること
  (``python -c`` やヒアドキュメントだとワーカーが ``__main__`` を再 import できず死ぬ)。
* ``league/run_match._begin_match_state`` の**デッキ注入**が入っていないと、deck.csv 以外を
  持つ側の探索(PIMC / ガード群)が黙って無効化される。データセットには必ず最新の
  ``league/`` と ``sample_submission/`` を同梱すること。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# kaggle_run -> kaggle_replays -> リポジトリルート
_DEFAULT_ROOT = _HERE.parent.parent

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - 端末によっては失敗するが致命的ではない
        pass

# ---------------------------------------------------------------------------
# 実験定義(_wall_deck_ab.py と同じ config / 重み / デッキ置き場)
# ---------------------------------------------------------------------------

CONFIG = "abl_5_full_og_r13"          # 現行最良(全10ガード)
WEIGHTS = "sample_submission/ptcg_ai/learning/policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"
DECK_DIR = "kaggle_replays/deck_search/candidates_wall"
BASELINE = "baseline"                  # og_v032 現行デッキ


# ---------------------------------------------------------------------------
# 統計ユーティリティ(drapa_ablation.py と同一実装。stdlib のみ)
# ---------------------------------------------------------------------------

def wilson(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 区間(既定 95%)。n=0 は [0,1]。"""
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def _norm_sf(z: float) -> float:
    """標準正規の上側確率 P(Z >= z)。"""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def binom_two_sided_p(wins: int, n: int, p0: float = 0.5) -> float | None:
    """H0: p = p0 の二項検定(両側)。p0=0.5 なので対称に足し上げれば厳密値になる。

    head-to-head の帰無仮説は「どちらのデッキも同じ強さ = 勝率 50%」なので、
    Wilson 区間が 0.5 をまたぐかどうかと合わせてこの p 値を見る。
    """
    if n <= 0:
        return None
    if p0 != 0.5:
        raise ValueError("p0=0.5 以外は未対応(この実験では常に 0.5)")
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2.0 ** n))


def two_proportion_z(w1: int, n1: int, w2: int, n2: int) -> dict:
    """2標本比率の片側 z 検定(H1: p1 > p2)。プールした標準誤差を使う。"""
    if n1 <= 0 or n2 <= 0:
        return {"z": None, "p_one_sided": None}
    p1, p2 = w1 / n1, w2 / n2
    pool = (w1 + w2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0.0:
        return {"z": None, "p_one_sided": None}
    z = (p1 - p2) / se
    return {"z": z, "p_one_sided": _norm_sf(z)}


def mcnemar_normal_greater(b: int, c: int) -> float | None:
    """McNemar の片側 p 値(正規近似)。b=「群1だけ勝ち」, c=「群2だけ勝ち」。

    n が数千になると exact(2**n の組合せ)は計算できないので正規近似を使う
    (b+c が数百以上あれば近似で十分)。
    """
    n = b + c
    if n == 0:
        return None
    z = (b - c) / math.sqrt(n)
    return _norm_sf(z)


def quantile(sorted_vals: list[float], q: float) -> float | None:
    """線形補間つき分位点(sorted_vals は昇順であること)。"""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# ワーカー(spawn 前提。重い import はすべてこの中で行う)
# ---------------------------------------------------------------------------

_W: dict = {}


def _worker_init(root_str: str, config: str, weights: str, deck_paths: dict) -> None:
    """各ワーカープロセスで1回だけ呼ばれ、パス・エージェント・デッキを準備する。"""
    root = Path(root_str)

    # sys.path: リポジトリルート・league/・sample_submission/(cg, ptcg_ai)。
    for cand in (str(root), str(root / "league"), str(root / "sample_submission")):
        if cand not in sys.path:
            sys.path.insert(0, cand)

    # ml_policy は cwd 相対で deck.csv 等を読むので cwd をそろえる
    # (spawn された子プロセスは親の cwd を継承しない)。
    os.chdir(root / "sample_submission")

    import run_league  # noqa: PLC0415 - ワーカー内で遅延 import する

    # 両陣営は **完全に同一のエージェント**(config も重みも同じ)。違うのはデッキだけ。
    agent = run_league.build_agent("ml_policy", weights, config)

    _W["run_league"] = run_league
    _W["agent"] = agent
    _W["decks"] = {name: run_league.read_deck_csv_file(p) for name, p in deck_paths.items()}


def _worker_play(task: tuple[str, str, int, int]) -> dict:
    """1試合を実行して JSON に書けるレコードを返す。"""
    arm, baseline, index, seed = task
    from run_match import play_match  # noqa: PLC0415 - ワーカー内で遅延 import

    agent = _W["agent"]
    decision_ms: list[float] = []

    def timed_agent(obs):
        t0 = time.perf_counter()
        action = agent(obs)
        decision_ms.append((time.perf_counter() - t0) * 1000.0)
        return action

    # 偶数 index は候補側が player0(先手扱い)。群をまたいで同一の割り当てになる。
    arm_is_player0 = (index % 2 == 0)
    arm_deck, base_deck = _W["decks"][arm], _W["decks"][baseline]
    if arm_is_player0:
        deck0, deck1, arm_index = arm_deck, base_deck, 0
        agent0, agent1 = timed_agent, agent
    else:
        deck0, deck1, arm_index = base_deck, arm_deck, 1
        agent0, agent1 = agent, timed_agent

    result = play_match(agent0, agent1, deck0, deck1, seed=seed)
    win = None if result.error is not None else bool(result.winner == arm_index)

    decision_ms.sort()
    return {
        "arm": arm,
        "index": index,
        "seed": seed,
        "arm_player_index": arm_index,
        "win": win,
        "turns": result.turns,
        "steps": result.steps,
        "seconds": round(result.seconds, 3),
        "error": result.error,
        # 決定時間は全件持たずに要約だけ(数千試合ぶんの JSONL を膨らませない)。
        "dec_n": len(decision_ms),
        "dec_mean_ms": round(sum(decision_ms) / len(decision_ms), 2) if decision_ms else None,
        "dec_p95_ms": round(quantile(decision_ms, 0.95), 2) if decision_ms else None,
        "dec_max_ms": round(decision_ms[-1], 2) if decision_ms else None,
    }


# ---------------------------------------------------------------------------
# 集計(shard をマージしても同じ関数で集計できるよう、レコード列だけに依存させる)
# ---------------------------------------------------------------------------

def summarize_arm(records: list[dict]) -> dict:
    """1群(=候補デッキ1つ)ぶんの勝率・先後別・エラー・時間を集計する。"""
    wins = valid = 0
    wins_p0 = n_p0 = wins_p1 = n_p1 = 0
    errors: dict[str, int] = {}
    turns_sum = turns_n = 0
    seconds_sum = 0.0
    dec_means: list[float] = []
    dec_max = 0.0

    for rec in records:
        seconds_sum += rec.get("seconds") or 0.0
        if rec.get("dec_mean_ms") is not None:
            dec_means.append(rec["dec_mean_ms"])
        if rec.get("dec_max_ms") is not None:
            dec_max = max(dec_max, rec["dec_max_ms"])
        if rec["error"] is not None:
            errors[rec["error"]] = errors.get(rec["error"], 0) + 1
            continue
        valid += 1
        won = bool(rec["win"])
        wins += int(won)
        if rec["arm_player_index"] == 0:
            n_p0 += 1
            wins_p0 += int(won)
        else:
            n_p1 += 1
            wins_p1 += int(won)
        if rec.get("turns") is not None:
            turns_sum += rec["turns"]
            turns_n += 1

    lo, hi = wilson(wins, valid)
    lo0, hi0 = wilson(wins_p0, n_p0)
    lo1, hi1 = wilson(wins_p1, n_p1)
    return {
        "wins": wins,
        "games": valid,
        "win_rate": (wins / valid) if valid else None,
        "wilson_95ci": [lo, hi],
        "p_two_sided_vs_50": binom_two_sided_p(wins, valid),
        "by_turn_order": {
            "arm_player0_first": {"wins": wins_p0, "games": n_p0,
                                  "win_rate": (wins_p0 / n_p0) if n_p0 else None,
                                  "wilson_95ci": [lo0, hi0]},
            "arm_player1_second": {"wins": wins_p1, "games": n_p1,
                                   "win_rate": (wins_p1 / n_p1) if n_p1 else None,
                                   "wilson_95ci": [lo1, hi1]},
        },
        "errors": {"count": sum(errors.values()), "by_reason": errors},
        "avg_turns": (turns_sum / turns_n) if turns_n else None,
        "avg_seconds": (seconds_sum / len(records)) if records else None,
        "decision_ms": {
            "mean_of_game_means": (sum(dec_means) / len(dec_means)) if dec_means else None,
            "max": dec_max or None,
        },
    }


def compare_arms(records: list[dict], treatment: str, control: str) -> dict:
    """候補どうしの比較(どちらも相手は baseline)。差(pt)と片側 p 値。"""
    t_recs = [r for r in records if r["arm"] == treatment and r["error"] is None]
    c_recs = [r for r in records if r["arm"] == control and r["error"] is None]
    tw, tn = sum(1 for r in t_recs if r["win"]), len(t_recs)
    cw, cn = sum(1 for r in c_recs if r["win"]), len(c_recs)

    t_map = {(r["index"], r["seed"]): r["win"] for r in t_recs}
    c_map = {(r["index"], r["seed"]): r["win"] for r in c_recs}
    common = set(t_map) & set(c_map)
    b = sum(1 for k in common if t_map[k] and not c_map[k])
    c = sum(1 for k in common if (not t_map[k]) and c_map[k])

    return {
        "treatment": treatment,
        "control": control,
        "treatment_rate": (tw / tn) if tn else None,
        "control_rate": (cw / cn) if cn else None,
        "diff_pt": (((tw / tn) - (cw / cn)) * 100) if (tn and cn) else None,
        "unpaired": {**two_proportion_z(tw, tn, cw, cn), "n_treatment": tn, "n_control": cn},
        "paired_mcnemar": {
            "n_pairs": len(common),
            "treatment_only_wins": b,
            "control_only_wins": c,
            "p_one_sided": mcnemar_normal_greater(b, c),
            "note": "cg のシャッフルは開幕以降 seed 制御外のため完全な CRN ではない。参考値。",
        },
    }


def load_records(path: str | Path) -> tuple[list[dict], list[dict]]:
    """JSONL を読み、``(試合レコード, メタ行)`` に分けて返す(shard マージ用)。"""
    records: list[dict] = []
    metas: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue    # 途中で切れた最終行(カーネル強制終了)を捨てる
            if "_meta" in obj:
                metas.append(obj["_meta"])
            elif "arm" in obj:
                records.append(obj)
    return records, metas


def aggregate(records: list[dict], meta: dict | None = None) -> dict:
    """全レコード(複数 shard をマージしたものでも可)から最終集計を作る。"""
    arms = sorted({r["arm"] for r in records})
    out: dict = {
        "meta": meta or {},
        "n_records": len(records),
        "arms": {a: summarize_arm([r for r in records if r["arm"] == a]) for a in arms},
        "comparisons": [],
    }
    # 候補どうしの比較(baseline 自身は self-mirror 対照群なので control 側に置く)。
    control = BASELINE if BASELINE in arms else None
    for arm in arms:
        if arm == control:
            continue
        if control:
            out["comparisons"].append(compare_arms(records, arm, control))
    others = [a for a in arms if a != control]
    if len(others) == 2:
        out["comparisons"].append(compare_arms(records, others[0], others[1]))
    return out


def print_summary(summary: dict) -> None:
    print("=" * 78)
    for arm, stat in summary["arms"].items():
        rate = stat["win_rate"]
        lo, hi = stat["wilson_95ci"]
        p = stat["p_two_sided_vs_50"]
        print(f"[{arm}] vs baseline: {stat['wins']}/{stat['games']} = "
              f"{(rate if rate is not None else float('nan')):.4f} "
              f"CI95[{lo:.4f},{hi:.4f}] p(2-sided vs 50%)="
              f"{'n/a' if p is None else format(p, '.4g')}  errors={stat['errors']['count']}")
        for key, slot in stat["by_turn_order"].items():
            r = slot["win_rate"]
            slo, shi = slot["wilson_95ci"]
            print(f"      {key:20s} {slot['wins']}/{slot['games']} = "
                  f"{(r if r is not None else float('nan')):.4f} CI95[{slo:.4f},{shi:.4f}]")
        dec = stat["decision_ms"]
        mean = dec["mean_of_game_means"]
        print(f"      avg_turns={stat['avg_turns']} avg_game_s={stat['avg_seconds']} "
              f"dec_mean_ms={'n/a' if mean is None else round(mean, 1)}")
    for cmp_ in summary["comparisons"]:
        d = cmp_["diff_pt"]
        z = cmp_["unpaired"]["z"]
        pz = cmp_["unpaired"]["p_one_sided"]
        mc = cmp_["paired_mcnemar"]
        print("-" * 78)
        print(f"{cmp_['treatment']} - {cmp_['control']} = "
              f"{(d if d is not None else float('nan')):+.2f}pt  "
              f"z={z if z is None else round(z, 3)} "
              f"p(one-sided,z)={pz if pz is None else round(pz, 5)}")
        print(f"  paired McNemar: b={mc['treatment_only_wins']} c={mc['control_only_wins']} "
              f"pairs={mc['n_pairs']} p={mc['p_one_sided']}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------

def build_tasks(arms: list[str], baseline: str, games: int,
                seed_base: int) -> list[tuple[str, str, int, int]]:
    """``(arm, baseline, index, seed)`` のタスク列。

    群を **インターリーブ** して並べる(index ごとに全群を回す)。カーネルが時間切れで
    打ち切られても、群ごとの試合数がほぼ等しくなり比較が壊れない
    (群を順番に流すと先頭の群だけ終わって後ろが0試合、という事故が起きる)。
    """
    tasks: list[tuple[str, str, int, int]] = []
    for index in range(games):
        for arm in arms:
            tasks.append((arm, baseline, index, seed_base + index))
    return tasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="aceburn,bulu",
                    help="候補デッキ名をカンマ区切り(<deck-dir>/<name>.csv)。"
                         f"{BASELINE!r} を含めると self-mirror 対照群になる")
    ap.add_argument("--baseline", default=BASELINE, help=f"相手デッキ名(default: {BASELINE})")
    ap.add_argument("--games", type=int, default=400, help="1群あたりの試合数(先後半々。偶数に丸める)")
    ap.add_argument("--workers", type=int, default=2, help="プロセス並列数(Kaggle CPU は 4)")
    ap.add_argument("--seed-base", type=int, default=1300000, help="seed の基点(shard ごとに変える)")
    ap.add_argument("--config", default=CONFIG, help=f"両陣営の config(default: {CONFIG})")
    ap.add_argument("--weights", default=WEIGHTS, help="両陣営の重み(リポジトリルート相対でも可)")
    ap.add_argument("--deck-dir", default=DECK_DIR, help="デッキCSVの置き場(リポジトリルート相対)")
    ap.add_argument("--repo-root", default=str(_DEFAULT_ROOT),
                    help="リポジトリルート(Kaggle では展開先を渡す)")
    ap.add_argument("--out", default=None, help="出力 JSONL(default: kaggle_run/out/h2h_<ts>.jsonl)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="この秒数を超えたら未実行タスクを打ち切る(0=無制限)")
    ap.add_argument("--progress-every", type=int, default=20, help="N試合ごとに進捗を出す")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import multiprocessing as mp

    args = parse_args(argv)
    root = Path(args.repo_root).resolve()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        raise SystemExit("--arms が空です")
    baseline = args.baseline

    deck_dir = Path(args.deck_dir)
    if not deck_dir.is_absolute():
        deck_dir = root / deck_dir
    deck_paths: dict[str, str] = {}
    for name in {*arms, baseline}:
        path = deck_dir / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"デッキCSVが見つかりません: {path}")
        deck_paths[name] = str(path)

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = root / weights
    if not weights.exists():
        raise SystemExit(f"重みが見つかりません: {weights}")

    games = args.games - (args.games % 2)   # 先後を必ず半々にする
    if games <= 0:
        raise SystemExit("--games は 2 以上の偶数に丸まる必要があります")

    out_path = Path(args.out) if args.out else (
        _HERE / "out" / f"h2h_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(arms, baseline, games, args.seed_base)
    meta = {
        "experiment": "deck_h2h",
        "arms": arms,
        "baseline": baseline,
        "config": args.config,
        "weights": str(weights),
        "deck_dir": str(deck_dir),
        "games_per_arm": games,
        "seed_base": args.seed_base,
        "workers": args.workers,
        "repo_root": str(root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    print(json.dumps({"meta": meta}, ensure_ascii=False), flush=True)

    deadline = (time.time() + args.max_seconds) if args.max_seconds > 0 else None
    records: list[dict] = []
    t0 = time.time()
    ctx = mp.get_context("spawn")   # Windows/Linux で挙動をそろえる

    with out_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        fp.flush()

        initargs = (str(root), args.config, str(weights), deck_paths)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                 initializer=_worker_init, initargs=initargs) as ex:
            futures = [ex.submit(_worker_play, t) for t in tasks]
            try:
                from concurrent.futures import as_completed
                for fut in as_completed(futures):
                    rec = fut.result()
                    records.append(rec)
                    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fp.flush()      # 途中で落ちてもそこまでの試合を残す
                    done = len(records)
                    if args.progress_every and (done % args.progress_every == 0
                                                or done == len(tasks)):
                        elapsed = time.time() - t0
                        parts = []
                        for arm in arms:
                            valid = [r for r in records if r["arm"] == arm and r["error"] is None]
                            w = sum(1 for r in valid if r["win"])
                            rate = w / len(valid) if valid else float("nan")
                            parts.append(f"{arm}={w}/{len(valid)}({rate:.3f})")
                        print(f"[{done}/{len(tasks)}] " + " ".join(parts)
                              + f"  elapsed={elapsed:.0f}s "
                                f"({elapsed / max(1, done):.2f}s/game)", flush=True)
                    if deadline is not None and time.time() > deadline:
                        print(f"[deadline] 時間予算に到達。未実行を打ち切ります "
                              f"({done}/{len(tasks)} 完了)", flush=True)
                        for f in futures:
                            f.cancel()
                        break
            except KeyboardInterrupt:   # 中断時もそこまでの結果を活かす
                for f in futures:
                    f.cancel()
                raise

    summary = aggregate(records, meta)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print_summary(summary)
    print(f"\nrecords -> {out_path}\nsummary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
