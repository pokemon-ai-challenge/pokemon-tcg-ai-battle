"""anytime探索 + 時間予算v2 の head-to-head(既定 mixogerpon ミラー)。

`abl_5_full_anytime`(A) vs `abl_5_full`(B)を **同一デッキ・同一重み** で対戦させ、
勝率とラウンド統計(1決定あたり何本の決定化 world を完走できたか)を測る。

最重要の落とし穴 — 予算グローバルの混線
------------------------------------------------
`ml_policy_agent` の時間予算は **モジュールグローバル**(`_match_start_perf` /
`_selects_seen` / `_expensive_roots_seen` / `_pipeline_breaker_tripped`)で持たれている。
本番は1プロセス1エージェントなので問題ないが、ローカル H2H は同一プロセスで2エージェントが
同じモジュールを共有するため、**素直に対戦させると両者の予算状態が混ざる**
(相手の select まで自分の分母に入り、両側とも本番と違う予算で走る)。

そこでここでは side ごとに4状態を保持し、agent 呼び出しの直前に復元・直後に退避する
(`_Side` / `_make_side_agent`)。各試合の開始時に両側とも独立に初期化するので、
**両側とも本番同様の動的予算が発火する**(B=従来式v1で約1350ms→、A=v2)。

計測
----
  * `pipeline.ANYTIME_STATS`: anytime パスのラウンド統計(A 側のみ anytime 有効なので
    A に帰属する)。試合ごとに `reset_anytime_stats()` してから読むので差分そのもの。
  * `pipeline.search` を probe でラップし、side 別に「探索呼び出し回数 / 決定化評価へ
    入った回数 / 実測 ms / 割当超過の最大値」を取る。probe が context へ差し込む
    `on_expensive_root` は production の同名コールバックに **チェーン** するので、
    B(mode 無し=production はカウントしない)側でも探索到達回数を観測できる。

再現性の注意: cg エンジンの内部シャッフルは seed を受け付けない(`battle_start(deck0, deck1)`)。
`--seed-base` は **Python 側 RNG**(決定化サンプリング等)のみを固定する。試合結果は完全再現
できないので、比較は必ず十分な n と CI で読むこと([[project_local_ab_noise_floor]])。

使用例:
    python kaggle_replays/challengers/anytime_h2h.py --games 4 --workers 2
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
_MEASUREMENT = _ROOT / "kaggle_replays" / "measurement"
for _p in (str(_SUB), str(_MEASUREMENT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_DECK = (_ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2"
                / "ogerpon_teal_ex" / "01.csv")
DEFAULT_WEIGHTS = (_SUB / "ptcg_ai" / "learning"
                   / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json")
DEFAULT_OUT = _HERE / "results" / "anytime_h2h.jsonl"

# worker プロセスごとの状態(spawn の子で `_init` が1回だけ埋める)。
_W: dict = {}


# ---------------------------------------------------------------------------
# side ごとの予算状態
# ---------------------------------------------------------------------------


class _Side:
    """1エージェント分の予算グローバル退避先 + 計測カウンタ。

    `ml_policy_agent` の4グローバルを side 単位で持ち、agent 呼び出しの前後で
    復元/退避することで、同一プロセス内の2エージェント間の混線を防ぐ。
    """

    __slots__ = ("label", "match_start", "selects", "roots", "breaker", "wall_s",
                 "search_calls", "search_roots", "search_ms", "max_overshoot_ms")

    def __init__(self, label: str) -> None:
        self.label = label
        # 試合開始時刻は side ごとに独立(= 両側とも本番同様の動的予算が発火する)。
        self.match_start = time.perf_counter()
        self.selects = 0
        self.roots = 0                  # production の _expensive_roots_seen(v2 の分母)
        self.breaker = False
        self.wall_s = 0.0               # この side の agent 呼び出し合計時間
        self.search_calls = 0           # pipeline.search を呼んだ回数(安い即 return 含む)
        self.search_roots = 0           # うち決定化評価(Step3)へ入った回数(probe 計測)
        self.search_ms = 0.0
        self.max_overshoot_ms = 0.0     # 実測 - 割当 の最大値(負なら余裕あり)

    def as_dict(self) -> dict:
        return {
            "selects": self.selects,
            "expensive_roots": self.roots,          # production カウンタ(v2 config のみ進む)
            "search_calls": self.search_calls,      # probe: search 呼び出し総数
            "search_roots": self.search_roots,      # probe: 実際に探索した回数(side 非依存)
            "breaker_tripped": self.breaker,
            "wall_s": round(self.wall_s, 2),
            "search_ms_total": round(self.search_ms, 1),
            "mean_search_ms": (round(self.search_ms / self.search_calls, 1)
                               if self.search_calls else None),
            "max_overshoot_ms": round(self.max_overshoot_ms, 1),
        }


def _make_side_agent(side: _Side, inner, MA):
    """side の予算状態を復元 → inner agent 実行 → 退避、で包んだ callable を返す。"""
    def _agent(obs):
        MA._match_start_perf = side.match_start
        MA._selects_seen = side.selects
        MA._expensive_roots_seen = side.roots
        MA._pipeline_breaker_tripped = side.breaker
        _W["cur"] = side                      # probe の帰属先
        t0 = time.perf_counter()
        try:
            return inner(obs)
        finally:
            side.wall_s += time.perf_counter() - t0
            # agent() は obs.select is None で _match_start_perf を再初期化しうるので戻す。
            side.match_start = MA._match_start_perf
            side.selects = MA._selects_seen
            side.roots = MA._expensive_roots_seen
            side.breaker = MA._pipeline_breaker_tripped
            _W["cur"] = None
    return _agent


def _install_search_probe(PL) -> None:
    """`pipeline.search` を side 帰属つきの計測ラッパで置き換える(worker 内だけ)。"""
    orig = PL.search

    def _probe(state, options, context):
        side = _W.get("cur")
        if side is not None:
            prev = context.get("on_expensive_root")

            def _hook(_prev=prev, _side=side) -> None:
                _side.search_roots += 1
                if _prev is not None:
                    _prev()               # production のカウンタ通知を潰さない

            context["on_expensive_root"] = _hook
        t0 = time.perf_counter()
        try:
            return orig(state, options, context)
        finally:
            if side is not None:
                ms = (time.perf_counter() - t0) * 1000.0
                side.search_calls += 1
                side.search_ms += ms
                try:
                    limit = float((context.get("config") or {}).get("time_limit_ms") or 0.0)
                except Exception:         # noqa: BLE001 - 計測が対戦を止めてはならない
                    limit = 0.0
                if limit > 0.0:
                    side.max_overshoot_ms = max(side.max_overshoot_ms, ms - limit)

    _W["orig_search"] = orig
    PL.search = _probe


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


def _init(cfg_a: str, cfg_b: str, deck_path: str, weights_path: str) -> None:
    """worker 起動時に1回。cwd 合わせ + 両 side の agent 構築 + probe 設置。"""
    import agents                        # noqa: PLC0415 - spawn 子でのみ import する
    import runner                        # noqa: PLC0415

    agents.ensure_production_cwd()
    from ptcg_ai.ml_policy import ml_policy_agent as MA   # noqa: PLC0415
    from ptcg_ai.search import pipeline as PL             # noqa: PLC0415

    def _mk(name: str):
        cfg = agents.load_config_copy(name)
        cfg["policy_weights_path"] = weights_path
        return agents.make_ml_policy_agent(cfg)

    _W["runner"] = runner
    _W["MA"] = MA
    _W["PL"] = PL
    _W["deck"] = runner.load_deck(deck_path)
    _W["inner"] = {"a": _mk(cfg_a), "b": _mk(cfg_b)}
    _install_search_probe(PL)


def _pct(values: list[int], p: float) -> float | None:
    """nearest-rank パーセンタイル(numpy 非依存)。空なら None。"""
    if not values:
        return None
    xs = sorted(values)
    k = math.ceil(p / 100.0 * len(xs)) - 1
    return float(xs[min(max(k, 0), len(xs) - 1)])


def _anytime_snapshot(PL) -> dict:
    """試合開始で reset した `ANYTIME_STATS` を読む(= その試合の差分そのもの)。"""
    s = PL.ANYTIME_STATS
    rounds = [int(v) for v in s["rounds_per_decision"]]
    return {
        "decisions": s["decisions"],
        "complete_rounds_total": s["complete_rounds_total"],
        "deadline_discard": s["deadline_discard"],
        "simulation_discard": s["simulation_discard"],
        "zero_complete_fallback": s["zero_complete_fallback"],
        "early_stops": s["early_stops"],
        "rounds_median": (float(statistics.median(rounds)) if rounds else None),
        "rounds_p10": _pct(rounds, 10),
        "rounds_p90": _pct(rounds, 90),
        "rounds_mean": (round(sum(rounds) / len(rounds), 3) if rounds else None),
        "rounds_per_decision": rounds,
    }


def _one(task: tuple[int, int, int]) -> dict:
    """1試合。task=(game index, A の player index, seed)。"""
    game, a_index, seed = task
    MA, PL, rn = _W["MA"], _W["PL"], _W["runner"]

    sides = {"a": _Side("a"), "b": _Side("b")}
    wrapped = {k: _make_side_agent(sides[k], _W["inner"][k], MA) for k in ("a", "b")}
    PL.reset_anytime_stats()
    random.seed(seed)                    # Python 側 RNG のみ(cg 内部は seed 不可)

    deck = list(_W["deck"])
    p0, p1 = ((wrapped["a"], wrapped["b"]) if a_index == 0 else (wrapped["b"], wrapped["a"]))
    t0 = time.perf_counter()
    try:
        r = rn.play_game(p0, p1, deck, list(deck))
    except Exception as exc:             # noqa: BLE001 - 1試合の事故で全体を落とさない
        return {"game": game, "seed": seed, "a_player_index": a_index,
                "error": f"{type(exc).__name__}: {exc}", "winner": None, "a_won": None}

    winner = getattr(r, "winner", None)
    return {
        "game": game, "seed": seed, "a_player_index": a_index,
        "winner": winner,
        "a_won": (None if winner is None else int(winner == a_index)),
        "primary": r.primary_win_condition,
        "turns": r.turns, "steps": r.steps,
        "error": r.error, "loser_by_error": r.loser_by_error,
        "game_wall_s": round(time.perf_counter() - t0, 2),
        "sides": {k: sides[k].as_dict() for k in ("a", "b")},
        # anytime が有効なのは A 側だけなので、この統計は A に帰属する。
        "anytime_a": _anytime_snapshot(PL),
    }


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------


def _wilson95(s: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    if n == 0:
        return (0.0, 1.0)
    phat = s / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4 * n)) / n)
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def _binom_p_one_sided(s: int, n: int) -> float:
    """H0: p <= 0.5 に対する片側 exact p 値 = P(X >= s | p = 0.5)。

    大きな n でも桁溢れしないよう対数(lgamma)で pmf を積む。
    """
    if n <= 0:
        return 1.0
    if s <= 0:
        return 1.0
    if s > n:
        return 0.0
    log2 = math.log(2.0)
    total = 0.0
    for k in range(s, n + 1):
        logp = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * log2)
        total += math.exp(logp)
    return min(1.0, total)


def _side_summary(rows: list[dict], side: str) -> dict:
    vals = [r["sides"][side] for r in rows if r.get("sides")]
    if not vals:
        return {}

    def _mean(key: str) -> float | None:
        xs = [v[key] for v in vals if v.get(key) is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    return {
        "games": len(vals),
        "mean_selects": _mean("selects"),
        "mean_expensive_roots": _mean("expensive_roots"),
        "mean_search_calls": _mean("search_calls"),
        "mean_search_roots": _mean("search_roots"),
        "mean_search_ms": _mean("mean_search_ms"),
        "mean_wall_s": _mean("wall_s"),
        "breaker_trips": sum(1 for v in vals if v.get("breaker_tripped")),
        "max_overshoot_ms": round(max(v["max_overshoot_ms"] for v in vals), 1),
    }


def summarize(rows: list[dict]) -> dict:
    """勝率(Wilson95 + 片側 exact p)/ side 別 / エラー / ラウンド統計。"""
    scored = [r for r in rows if r.get("a_won") is not None]
    n = len(scored)
    s = sum(int(r["a_won"]) for r in scored)
    lo, hi = _wilson95(s, n)

    def _wr(subset: list[dict]) -> dict:
        m = len(subset)
        w = sum(int(x["a_won"]) for x in subset)
        return {"games": m, "wins": w, "winrate": (round(w / m, 4) if m else None)}

    pooled: list[int] = []
    for r in rows:
        pooled.extend((r.get("anytime_a") or {}).get("rounds_per_decision") or [])

    return {
        "games_played": len(rows),
        "games_scored": n,
        "a_wins": s,
        "a_winrate": (round(s / n, 4) if n else None),
        "wilson95_ci": [round(lo, 4), round(hi, 4)],
        "p_one_sided_exact": round(_binom_p_one_sided(s, n), 5),
        "by_a_player_index": {
            "0": _wr([r for r in scored if r["a_player_index"] == 0]),
            "1": _wr([r for r in scored if r["a_player_index"] == 1]),
        },
        "errors": sum(1 for r in rows if r.get("error")),
        "error_samples": [r["error"] for r in rows if r.get("error")][:5],
        "side_a": _side_summary(rows, "a"),
        "side_b": _side_summary(rows, "b"),
        "anytime_a": {
            "decisions": sum((r.get("anytime_a") or {}).get("decisions", 0) for r in rows),
            "complete_rounds_total": sum((r.get("anytime_a") or {}).get("complete_rounds_total", 0)
                                         for r in rows),
            "deadline_discard": sum((r.get("anytime_a") or {}).get("deadline_discard", 0)
                                    for r in rows),
            "simulation_discard": sum((r.get("anytime_a") or {}).get("simulation_discard", 0)
                                      for r in rows),
            "zero_complete_fallback": sum((r.get("anytime_a") or {}).get("zero_complete_fallback", 0)
                                          for r in rows),
            "early_stops": sum((r.get("anytime_a") or {}).get("early_stops", 0) for r in rows),
            "rounds_per_decision": {
                "n": len(pooled),
                "median": (float(statistics.median(pooled)) if pooled else None),
                "mean": (round(sum(pooled) / len(pooled), 3) if pooled else None),
                "p10": _pct(pooled, 10), "p90": _pct(pooled, 90),
                "min": (min(pooled) if pooled else None),
                "max": (max(pooled) if pooled else None),
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="anytime + 時間予算v2 の H2H(既定 mixogerpon ミラー)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=20_260_813)
    ap.add_argument("--config-a", default="abl_5_full_anytime")
    ap.add_argument("--config-b", default="abl_5_full")
    ap.add_argument("--deck", default=str(DEFAULT_DECK))
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    deck_path = str(Path(args.deck).resolve())
    weights_path = str(Path(args.weights).resolve())
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()               # fresh run(追記汚染を避ける)

    # task: A の player index を交互に(偶数試合で層化)、seed は seed_base + game。
    tasks = [(g, g % 2, args.seed_base + g) for g in range(args.games)]

    print(f"=== anytime H2H: A={args.config_a} vs B={args.config_b} ===")
    print(f"  deck={deck_path}")
    print(f"  weights={weights_path}")
    print(f"  games={args.games} workers={args.workers} seed_base={args.seed_base}")

    rows: list[dict] = []
    t0 = time.perf_counter()
    with Pool(max(1, args.workers), initializer=_init,
              initargs=(args.config_a, args.config_b, deck_path, weights_path)) as pool:
        with out_path.open("a", encoding="utf-8") as fh:
            for rec in pool.imap(_one, tasks, chunksize=1):
                rows.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done = len(rows)
                wins = sum(int(r["a_won"]) for r in rows if r.get("a_won") is not None)
                el = time.perf_counter() - t0
                print(f"  [{done}/{args.games}] winner={rec.get('winner')} "
                      f"A={wins}/{done} err={sum(1 for r in rows if r.get('error'))} "
                      f"[{el:.0f}s]")

    summary = summarize(rows)
    summary["meta"] = {
        "config_a": args.config_a, "config_b": args.config_b,
        "deck": deck_path, "weights": weights_path,
        "seed_base": args.seed_base, "workers": args.workers,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nrows -> {out_path}\nsummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
