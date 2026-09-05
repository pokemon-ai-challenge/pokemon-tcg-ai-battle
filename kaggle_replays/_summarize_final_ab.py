"""最終A/B(og_r7_isorng vs og_r13)の集計。実ラダー68試合の対面出現数でメタ加重する。"""
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).parent
# 実ラダー(submission 55527580、68試合)の対面出現数
META_W = {
    "dragapult_ex": 14, "marnie_grimmsnarl_ex": 13, "kamitsuorochi_ex": 9,
    "mega_lucario_ex": 6, "lopunny_megafroslass": 6, "yadoking": 5,
    "ogerpon_teal_ex": 4, "alakazam": 4, "crustle": 3,
    "mega_froslass_ex": 2, "rocket_mewtwo_ex": 1, "shirona_garchomp_ex": 1,
}


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "_og_r13_final_ab_results.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    keys = list(d)
    base, cand = d[keys[0]], d[keys[1]]
    print(f"最終A/B  {keys[0]} → {keys[1]}")
    print(f"{'対面':24s} {'og_r7':>7s} {'og_r13':>7s} {'差':>9s}   n     メタ重み")
    tw = tb = tc = 0.0
    rows = []
    for arch in base:
        b, c = base[arch], cand.get(arch)
        if not c:
            print(f"  {arch:24s} (og_r13 未完)")
            continue
        diff = (c["wr"] - b["wr"]) * 100
        se = math.sqrt(b["wr"] * (1 - b["wr"]) / b["games"]
                       + c["wr"] * (1 - c["wr"]) / c["games"]) * 100
        w = META_W.get(arch, 1)
        tw += w
        tb += w * b["wr"]
        tc += w * c["wr"]
        sig = " *有意" if abs(diff) > se * 1.96 else ""
        rows.append((arch, b["wr"], c["wr"], diff, se * 1.96, b["games"], w, sig))
    for arch, bw, cw, diff, ci, n, w, sig in sorted(rows, key=lambda r: -r[6]):
        print(f"  {arch:24s} {bw:.3f}   {cw:.3f}  {diff:+6.1f}pt(±{ci:.1f}) n={n:<4d} w={w}{sig}")
    if tw:
        print(f"\n  メタ加重(実ラダー出現比): {tb/tw:.4f} → {tc/tw:.4f} = {(tc-tb)/tw*100:+.2f}pt")
        print(f"  単純平均: {sum(r[1] for r in rows)/len(rows):.4f} → "
              f"{sum(r[2] for r in rows)/len(rows):.4f}")


if __name__ == "__main__":
    main()
