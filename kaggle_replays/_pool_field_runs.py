"""複数 seed のフィールド走行を1つに合算する(アーム内でのみ合算)。

同一 config・同一相手・同一試合数で seed だけ変えた走行を足して n を増やし、
差の信頼区間を狭める。`_phase4_compare_field.py` にそのまま渡せる形式で出力する。

合算は **同じアーム内だけ**(baseline同士 / candidate同士)。異なる config を混ぜない。
探索失敗率が0でない走行が1つでも混ざったら、合算結果も無効として印を付ける。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.inputs]
    configs = {p.get("meta", {}).get("config") for p in payloads}
    if len(configs) != 1:
        raise SystemExit(f"config が混在している(合算不可): {configs}")

    acc: dict[str, dict] = {}
    clean = True
    for p in payloads:
        clean = clean and bool(p.get("ALL_SEARCH_CLEAN"))
        for r in p.get("per_archetype", []):
            a = r["archetype"]
            e = acc.setdefault(a, {"w": 0, "n": 0, "fail_rates": []})
            e["w"] += r["climb_wins"] or 0
            e["n"] += r["valid_games"] or 0
            e["fail_rates"].append(r["search"]["begin_fail_rate"])

    per = [{
        "archetype": a,
        "valid_games": e["n"],
        "climb_wins": e["w"],
        "win_rate": round(e["w"] / e["n"], 4) if e["n"] else None,
        "search": {"begin_fail_rate": (0.0 if all(f in (0, 0.0) for f in e["fail_rates"])
                                       else max(f for f in e["fail_rates"] if f is not None))},
    } for a, e in acc.items()]

    out = {
        "meta": {
            "config": configs.pop(),
            "pooled_from": [str(p) for p in args.inputs],
            "seeds": [p.get("meta", {}).get("seed_start") for p in payloads],
        },
        "per_archetype": per,
        "ALL_SEARCH_CLEAN": clean,
    }
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    total_n = sum(e["n"] for e in acc.values())
    total_w = sum(e["w"] for e in acc.values())
    print(f"pooled {len(payloads)} runs -> n={total_n} wins={total_w} "
          f"rate={total_w/total_n:.4f} clean={clean}")
    print(f"[written] {dest}")


if __name__ == "__main__":
    main()
