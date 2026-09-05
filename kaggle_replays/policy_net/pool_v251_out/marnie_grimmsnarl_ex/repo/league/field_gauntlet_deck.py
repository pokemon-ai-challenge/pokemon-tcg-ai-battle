#!/usr/bin/env python3
"""多様フィールド対戦(デッキ比較版): 契約者の config は abl_5_full に固定し、デッキだけを
新デッキ(experimental_decks/alakazam_xerosic) と 現行 deck.csv で振って、メタ各アーキタイプ相手の
対フィールド期待勝率を比較する。「最新フーディンデッキ(full)が他アーキにも強いか、現行デッキより
強いか」を local で測るのが目的。

field_gauntlet.py(config比較版)の姉妹。フィールド定義(deck/専用重み/メタシェア)は
round_robin.ARCHS を再利用。契約者=alakazam系(weights=None=production構成C)、config=abl_5_full。
opponent=各アーキ(deck+専用重み, config=v0only固定)。run_league/run_match は不変で流用。

使い方(デスクトップ):
  python field_gauntlet_deck.py --games 100 --workers 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_league  # noqa: E402
from round_robin import ARCHS  # noqa: E402

CONTENDER_CONFIG = "abl_5_full"
OPP_CONFIG = "ml_lethal_attackplan_v0only"
CONTENDER_DECKS = {
    "new_deck": "experimental_decks/alakazam_xerosic/deck.csv",
    "cur_deck": "sample_submission/deck.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    outdir = _ROOT / "results" / "field_gauntlet_deck"
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(_ROOT / "sample_submission")

    field = [a for a in ARCHS if a[0] != "alakazam"]
    results: dict[str, dict[str, dict]] = {k: {} for k in CONTENDER_DECKS}

    t0 = time.time()
    for dlabel, ddeck in CONTENDER_DECKS.items():
        print(f"\n=== contender deck={dlabel} ({ddeck}) config={CONTENDER_CONFIG} ===", flush=True)
        for (fn, fd, fw, share) in field:
            out = outdir / f"{dlabel}__vs__{fn}_{args.games}.json"
            if out.exists():
                summary = json.load(open(out, encoding="utf-8"))
                tag = "(既存再利用)"
            else:
                summary = run_league.run_league(
                    agent_a_name="ml_policy", agent_b_name="ml_policy", games=args.games,
                    deck_a_path=ddeck, deck_b_path=fd,
                    weights_a_path=None, weights_b_path=fw,
                    config_base_a=CONTENDER_CONFIG, config_base_b=OPP_CONFIG,
                    seed_start=args.seed_start, progress_every=0,
                    workers=args.workers, log=lambda m: None,
                )
                json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                tag = ""
            ov = summary["overall"]
            wr = ov["win_rate"]
            lo, hi = ov["wilson_95ci"]
            results[dlabel][fn] = {"wr": wr, "ci": [lo, hi], "share": share, "errors": summary["errors"]["count"]}
            wr_s = "n/a" if wr is None else f"{wr*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]"
            print(f"  vs {fn:22s} {wr_s} (err={summary['errors']['count']}) {tag}", flush=True)

    def field_expected(dlabel: str) -> float:
        num = den = 0.0
        for fn, r in results[dlabel].items():
            if r["wr"] is None:
                continue
            num += r["share"] * r["wr"]
            den += r["share"]
        return num / den if den else float("nan")

    fexp = {k: field_expected(k) for k in CONTENDER_DECKS}
    print("\n=== 対フィールド期待勝率(メタシェア加重, config=full固定) ===", flush=True)
    fields = list(results["new_deck"].keys())
    print(f"{'相手':22s}{'new_deck':>16s}{'cur_deck':>16s}{'Δ(new-cur)':>12s}")
    for fn in fields:
        wn = results["new_deck"][fn]["wr"]
        wc = results["cur_deck"][fn]["wr"]
        dd = (wn - wc) * 100 if (wn is not None and wc is not None) else float("nan")
        print(f"{fn:22s}{wn*100:14.1f}%{wc*100:14.1f}%{dd:+11.1f}", flush=True)
    for k in CONTENDER_DECKS:
        print(f"  対フィールド期待勝率 {k:9s} = {fexp[k]*100:.1f}%", flush=True)
    print(f"  delta(new - cur) = {(fexp['new_deck']-fexp['cur_deck'])*100:+.1f}pt", flush=True)

    matrix = {
        "games_per_pair": args.games,
        "contender_config": CONTENDER_CONFIG,
        "opponent_config": OPP_CONFIG,
        "contender_decks": CONTENDER_DECKS,
        "per_field": results,
        "field_expected_winrate": fexp,
        "delta_new_minus_cur": fexp["new_deck"] - fexp["cur_deck"],
    }
    mpath = outdir / "_gauntlet_deck_matrix.json"
    json.dump(matrix, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nmatrix -> {mpath}\n総経過 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
