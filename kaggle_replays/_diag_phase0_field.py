#!/usr/bin/env python3
"""Phase 0 / H1a: 学習分布 vs 配備分布のミスマッチ(read-only, 使い捨て診断)。

phase0-transfer-gap-diagnosis-implementation-plan.md §3.1。

policy は「上位プレイヤーのリプレイ」を模倣して学習している。上位プレイヤーの試合(=学習分布)で
遭遇するアーキタイプ構成と、自分のエージェントがフィールドで配備されて遭遇する構成(=配備分布)が
ズレていれば、模倣は「別の相手分布に最適な手」を学んでいることになる(転移ロスの一因)。

既存の `meta_analysis/output/meta_report.json` は archetype ごとに
- top_pct: 上位ランク帯での出現シェア(= 学習データの主戦場、上位ミラーの相手)
- field_pct: フィールドでの出現シェア(= 自エージェントが当たる相手の近似)
を **出現数ベースで(player_count とは別に)** 集計済み。混同([[project_meta_analysis_findings]] の
「出現数 vs 使用者数」)を避けるため、ゲーム頻度の代理として一貫して出現シェア(pct)を使う。

production / weights / config を一切変更しない。既存 JSON の読み取りのみ。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
META = HERE / "meta_analysis" / "output" / "meta_report.json"
TAXO = HERE / "_diag_loss_taxonomy_results.json"


def l1_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def kl(p: dict[str, float], q: dict[str, float], eps: float = 1e-6) -> float:
    """KL(p || q)。p=配備(field), q=学習(top) の向きで「配備で頻出なのに学習で稀」を検出。"""
    keys = set(p) | set(q)
    total = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        if pk <= 0:
            continue
        qk = max(q.get(k, 0.0), eps)
        total += pk * math.log(pk / qk)
    return total


def main() -> None:
    meta = json.loads(META.read_text(encoding="utf-8"))
    shares = meta["archetype_share"]

    # 出現シェア(%)を正規化して分布化
    top = {s["archetype"]: s.get("top_pct", 0.0) for s in shares}
    field = {s["archetype"]: s.get("field_pct", 0.0) for s in shares}
    top_sum = sum(top.values()) or 1.0
    field_sum = sum(field.values()) or 1.0
    top_n = {k: v / top_sum for k, v in top.items()}
    field_n = {k: v / field_sum for k, v in field.items()}

    rows = []
    for s in shares:
        a = s["archetype"]
        rows.append({
            "archetype": a,
            "top_pct": round(s.get("top_pct", 0.0), 2),
            "field_pct": round(s.get("field_pct", 0.0), 2),
            "top_minus_field": round(s.get("top_minus_field_pct", s.get("top_pct", 0.0) - s.get("field_pct", 0.0)), 2),
        })
    rows.sort(key=lambda r: r["top_minus_field"], reverse=True)

    # 自分の敗北の相手アーキタイプ(losses-only、参考)
    taxo = json.loads(TAXO.read_text(encoding="utf-8"))
    my_loss_opp = taxo.get("by_archetype_count", {})
    my_loss_total = sum(my_loss_opp.values()) or 1
    my_loss_share = {k: round(100 * v / my_loss_total, 2) for k, v in
                     sorted(my_loss_opp.items(), key=lambda kv: kv[1], reverse=True)}

    results = {
        "note": "top_pct=上位帯の出現シェア(学習データの相手), field_pct=フィールドの出現シェア(配備の相手). "
                "出現数ベース(player_count とは別軸). ゲーム頻度の代理.",
        "l1_distance_top_vs_field_normalized": round(l1_distance(top_n, field_n), 4),
        "kl_field_given_top": round(kl(field_n, top_n), 4),
        "rows_sorted_by_top_minus_field": rows,
        "my_loss_opponent_share_pct_losses_only": my_loss_share,
    }
    outp = HERE / "_diag_phase0_field_results.json"
    outp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== H1a: 学習分布(top)vs 配備分布(field)の出現シェア ==")
    print(f"{'archetype':<22} {'top%':>7} {'field%':>7} {'top-field':>10}")
    print("-" * 50)
    for r in rows:
        print(f"{r['archetype']:<22} {r['top_pct']:>7.2f} {r['field_pct']:>7.2f} {r['top_minus_field']:>10.2f}")
    print(f"\n正規化 L1 距離(top vs field): {results['l1_distance_top_vs_field_normalized']:.4f}  (0=一致, 2=完全不一致)")
    print(f"KL(field || top): {results['kl_field_given_top']:.4f}  (大きいほど『配備で頻出だが学習で稀』)")
    print("\n== 参考: 自分の敗北の相手アーキタイプ(losses-only) ==")
    for k, v in list(my_loss_share.items())[:10]:
        print(f"  {k:<22} {v:>6.2f}%")
    print(f"\n-> {outp.name} に保存")


if __name__ == "__main__":
    main()
