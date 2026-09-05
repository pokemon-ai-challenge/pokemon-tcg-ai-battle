"""ドラパルトの「操縦の伸びしろ」上限測定。

問い: gen2 BC(模倣)のフィールド勝率 0.203 は、デッキが弱いからか、操縦が下手だからか。
方法: **同じデッキ・同じフィールド・同じ試合数配分**で、操縦者だけを差し替えて比較する。
  - bc    : ml_policy + policy_weights_dragapult_ex_g2.json(= train_league が測ったもの)
  - rule  : opponents/dragapult_rule_agent.py(kiyotah版の手練れヒューリスティック)
フィールドは train_league の FIELD_PRESETS["mix"](11アーキ、シェア比例で試合数配分)。
相手は各アーキの模倣重み + abl_5_full(= deck_search/stage1_screen.py と同条件)。

rule 側が大きく上回るなら「操縦の伸びしろ」が実在=計画アルゴリズムに投資する価値がある。
両方低いならデッキ自体が環境に合っていない=投資しない。

使い方:
  python kaggle_replays/_drapa_ceiling.py --games 400 --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "league"), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DRAPA_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2" / "dragapult_ex" / "01.csv"
CONFIG = "abl_5_full"

# train_league.FIELD_MIX と同一の11アーキ+シェア(stage1_screen.py と同じ定義)。
FIELD = [
    ("marnie_grimmsnarl_ex", 1082, "_g2"), ("alakazam", 891, "_g2"),
    ("mega_lucario_ex", 403, ""), ("dragapult_ex", 381, "_g2"),
    ("mega_froslass_ex", 353, "_g2"), ("ogerpon_teal_ex", 310, "_g2"),
    ("archaludon_ex", 287, "_g2"), ("crustle", 287, "_g2"),
    ("shirona_garchomp_ex", 158, "_g2"), ("omatsuri_ondo", 138, "_g2"),
    ("rocket_mewtwo_ex", 112, "_g2"),
]
_DECKDIR_OF = {"": "archetype_decks", "_g2": "archetype_decks_g2"}


def allocate(total: int) -> list[tuple[str, str, str, int]]:
    tot = sum(s for _, s, _ in FIELD)
    return [(a, g, _DECKDIR_OF[g], max(2, round(total * s / tot))) for a, s, g in FIELD]


def wilson(w: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, z = w / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_pilot(pilot: str, games: int, workers: int, seed_base: int) -> dict:
    """pilot='bc'(模倣) or 'rule'(kiyotah)。デッキ・相手・seedは完全に共通。"""
    import run_league
    rows, wins, tot = {}, 0, 0
    for arch, gen, deckdir, n in allocate(games):
        opp_w = str(WDIR / f"policy_weights_{arch}{gen}.json")
        kwargs = dict(
            agent_b_name="ml_policy", games=n,
            deck_a_path=str(DRAPA_DECK),
            deck_b_path=str(_ROOT / "kaggle_replays" / "meta_analysis" / deckdir / arch / "01.csv"),
            seed_start=seed_base, progress_every=0,
            weights_b_path=opp_w, config_base=CONFIG, workers=workers, log=lambda m: None,
        )
        if pilot == "rule":
            # 素の agent(obs) なので重み/config を取らない(PLAIN_AGENTS)
            summary = run_league.run_league(agent_a_name="dragapult_rule", **kwargs)
        else:
            summary = run_league.run_league(
                agent_a_name="ml_policy",
                weights_a_path=str(WDIR / "policy_weights_dragapult_ex_g2.json"), **kwargs)
        ov = summary["overall"]
        rows[arch] = {"wins": ov["wins"], "games": ov["games"]}
        wins += ov["wins"]
        tot += ov["games"]
        print(f"  [{pilot}] {arch:22s} {ov['wins']}/{ov['games']} = {ov['wins']/max(1,ov['games']):.3f}",
              flush=True)
    lo, hi = wilson(wins, tot)
    return {"pilot": pilot, "wins": wins, "games": tot, "winrate": wins / max(1, tot),
            "ci95": [lo, hi], "by_opp": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=400, help="1操縦者あたりの総試合数")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=960000)
    ap.add_argument("--out", default=str(_HERE / "_drapa_ceiling.json"))
    args = ap.parse_args()

    results = []
    for pilot in ("rule", "bc"):
        print(f"=== pilot={pilot} ===", flush=True)
        r = run_pilot(pilot, args.games, args.workers, args.seed_base)
        print(f"  -> {pilot}: {r['winrate']:.3f} "
              f"CI[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] n={r['games']}", flush=True)
        results.append(r)

    rule, bc = results[0], results[1]
    gap = (rule["winrate"] - bc["winrate"]) * 100
    out = {"field": "mix", "config": CONFIG, "deck": str(DRAPA_DECK),
           "seed_base": args.seed_base, "results": results, "pilot_gap_pt": gap}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n操縦者の差(rule − bc) = {gap:+.1f}pt")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
