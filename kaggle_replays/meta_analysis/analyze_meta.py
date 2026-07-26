#!/usr/bin/env python3
"""環境メタ分析 (#1): アーキタイプ・シェアを「上位 vs 全体」で比較し、
`other` の中身を主軸カードで内訳分解し、アーキタイプごとのカード採用率を出す。

入力 (すべて deck_predictor/output/ 配下、既存):
  - deck_db.jsonl      : 各デッキ60枚 (deck_card_ids) + rank_at_fetch + team_name
  - deck_labels.jsonl  : (episode_id, player_index) -> archetype
  - ../index/episodes_master.jsonl : player 別 leaderboard_score / rank

結合キー: (episode_id, player_index)。名前列は文字化けがあるためカードIDを正とする。
カード名は data/JP_Card_Data.csv (id -> 名前) で後付けする。

出力: meta_analysis/output/meta_report.json (ビュアー用) と meta_report.md (人間用)。

使い方:
  python analyze_meta.py [--top-rank 100] [--top-decks 5]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent
_DP_OUT = _REPO / "kaggle_replays" / "deck_predictor" / "output"
_EPISODES = _REPO / "kaggle_replays" / "index" / "episodes_master.jsonl"
_CARD_CSV = _REPO / "data" / "JP_Card_Data.csv"
_OUT_DIR = _HERE / "output"

BASIC_ENERGY_IDS = set(range(1, 9))  # ID 1-8 = 基本エネルギー


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_card_names() -> dict[int, str]:
    names: dict[int, str] = {}
    if not _CARD_CSV.exists():
        return names
    with _CARD_CSV.open(encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if not row:
                continue
            try:
                cid = int(row[0])
            except (ValueError, IndexError):
                continue
            names[cid] = row[1] if len(row) > 1 else str(cid)
    return names


def load_labels() -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for d in _iter_jsonl(_DP_OUT / "deck_labels.jsonl"):
        out[(str(d["episode_id"]), int(d["player_index"]))] = d.get("archetype", "other")
    return out


def load_episode_scores() -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for ep in _iter_jsonl(_EPISODES):
        eid = str(ep.get("episode_id"))
        for p in ep.get("players", []):
            pi = p.get("player_index")
            if pi is None:
                continue
            out[(eid, int(pi))] = {
                "score": p.get("leaderboard_score_at_fetch"),
                "rank": p.get("rank_at_fetch"),
                "team": p.get("team_name"),
            }
    return out


def build_instances(labels, scores):
    """各デッキ出現を1レコードに束ねる。"""
    instances = []
    for d in _iter_jsonl(_DP_OUT / "deck_db.jsonl"):
        eid = str(d["episode_id"])
        pi = int(d["player_index"])
        key = (eid, pi)
        arche = labels.get(key, "other")
        ids = [int(x) for x in d.get("deck_card_ids", [])]
        ep = scores.get(key, {})
        # rank は deck_db を優先、無ければ episodes_master
        rank = d.get("rank_at_fetch")
        if rank is None:
            rank = ep.get("rank")
        instances.append(
            {
                "episode_id": eid,
                "player_index": pi,
                "team": d.get("team_name") or ep.get("team"),
                "rank": rank,
                "score": ep.get("score"),
                "archetype": arche,
                "card_ids": ids,
            }
        )
    return instances


def deck_signature(card_ids) -> tuple:
    return tuple(sorted(card_ids))


def compute_shares(instances, top_rank: int):
    archetypes = sorted({i["archetype"] for i in instances})
    overall = Counter()
    top = Counter()
    field = Counter()
    le100 = Counter()  # rank<=100 (超上位)
    le200 = Counter()  # rank<=200 (上位)
    players = defaultdict(set)  # archetype -> {team}
    n_top = n_field = n_unknown = 0
    n_le100 = n_le200 = 0

    for inst in instances:
        a = inst["archetype"]
        overall[a] += 1
        if inst["team"]:
            players[a].add(inst["team"])
        rank = inst["rank"]
        if rank is None:
            n_unknown += 1
        else:
            if rank <= 100:
                le100[a] += 1
                n_le100 += 1
            if rank <= 200:
                le200[a] += 1
                n_le200 += 1
            if rank <= top_rank:
                top[a] += 1
                n_top += 1
            else:
                field[a] += 1
                n_field += 1

    tot = sum(overall.values())
    rows = []
    for a in archetypes:
        o = overall[a]
        t = top[a]
        fd = field[a]
        top_pct = 100 * t / n_top if n_top else 0.0
        field_pct = 100 * fd / n_field if n_field else 0.0
        rows.append(
            {
                "archetype": a,
                "overall_appearances": o,
                "overall_pct": round(100 * o / tot, 2) if tot else 0.0,
                "player_count": len(players[a]),
                "top_appearances": t,
                "top_pct": round(top_pct, 2),
                "field_appearances": fd,
                "field_pct": round(field_pct, 2),
                "top_minus_field_pct": round(top_pct - field_pct, 2),
                # 円グラフのバンド別出現数(全体/上位<=200/超上位<=100)
                "band_all": o,
                "band_le200": le200[a],
                "band_le100": le100[a],
            }
        )
    rows.sort(key=lambda r: r["overall_appearances"], reverse=True)
    return rows, {
        "total_decks": tot,
        "top_rank_cutoff": top_rank,
        "n_top": n_top,
        "n_field": n_field,
        "n_unknown_rank": n_unknown,
        "n_le100": n_le100,
        "n_le200": n_le200,
    }


def compute_other_candidates(instances, names, all_freq):
    """other デッキの主軸カード(ex/最も特徴的なカード)を数える。"""
    def identity_card(ids):
        non_basic = [c for c in ids if c not in BASIC_ENERGY_IDS]
        if not non_basic:
            return None
        # ex/メガ カードを優先、その中で最もレア(全体頻度が低い)なもの
        cand = [c for c in set(non_basic) if "ex" in names.get(c, "").lower() or "ｅｘ" in names.get(c, "")]
        pool = cand if cand else list(set(non_basic))
        # 全体で珍しいカードほどそのデッキの identity 性が高い
        return min(pool, key=lambda c: (all_freq.get(c, 0), -c))

    counter = Counter()
    n_other = 0
    for inst in instances:
        if inst["archetype"] != "other":
            continue
        n_other += 1
        cid = identity_card(inst["card_ids"])
        if cid is not None:
            counter[cid] += 1
    out = [
        {
            "card_id": cid,
            "name": names.get(cid, str(cid)),
            "count": n,
            "pct_of_other": round(100 * n / n_other, 2) if n_other else 0.0,
        }
        for cid, n in counter.most_common(25)
    ]
    return out, n_other


def _adoption_from_builds(builds, names):
    """distinct build のリストから採用率カードテーブルを作る。"""
    n = len(builds)
    if n == 0:
        return []
    present = Counter()
    copies = defaultdict(int)
    for ids in builds:
        cc = Counter(ids)
        for cid, k in cc.items():
            present[cid] += 1
            copies[cid] += k
    cards = [
        {
            "card_id": cid,
            "name": names.get(cid, str(cid)),
            "adoption_pct": round(100 * present[cid] / n, 1),
            "avg_copies": round(copies[cid] / n, 2),
            "builds_with": present[cid],
        }
        for cid in present
    ]
    cards.sort(key=lambda c: (c["adoption_pct"], c["avg_copies"]), reverse=True)
    return cards


def compute_card_adoption(instances, names, min_builds=3, top_rank=100):
    """アーキタイプごと: 重複デッキを排除した distinct build 上でのカード採用率。

    全体 (cards) に加え、rank<=top_rank の上位デッキだけに絞った採用率 (cards_top) も出す。
    上位帯は標本が薄いので min_builds のフィルタは全体側にのみ適用する。
    """
    by_arch = defaultdict(list)
    by_arch_top = defaultdict(list)
    for inst in instances:
        by_arch[inst["archetype"]].append(inst["card_ids"])
        r = inst["rank"]
        if r is not None and r <= top_rank:
            by_arch_top[inst["archetype"]].append(inst["card_ids"])

    result = {}
    for a, decks in by_arch.items():
        seen = {}
        for ids in decks:
            seen[deck_signature(ids)] = ids  # 同一リストは1つに畳む
        builds = list(seen.values())
        if len(builds) < min_builds:
            continue
        # 上位帯 distinct build (無ければ空)
        seen_top = {}
        for ids in by_arch_top.get(a, []):
            seen_top[deck_signature(ids)] = ids
        builds_top = list(seen_top.values())
        result[a] = {
            "distinct_builds": len(builds),
            "total_decks": len(decks),
            "cards": _adoption_from_builds(builds, names),
            "top_rank": top_rank,
            "top_distinct_builds": len(builds_top),
            "top_total_decks": len(by_arch_top.get(a, [])),
            "cards_top": _adoption_from_builds(builds_top, names),
        }
    return result


def compute_top_decks(instances, names, top_n=5):
    """各アーキタイプで rank が最良(小さい)な distinct build を top_n 件抽出し、
    各カードの採用枚数を返す。ビュアーの『上位デッキ×カード枚数』マトリクス用。"""
    by_arch = defaultdict(list)
    for inst in instances:
        by_arch[inst["archetype"]].append(inst)

    result = {}
    for a, insts in by_arch.items():
        ranked = [i for i in insts if i["rank"] is not None]
        ranked.sort(key=lambda i: i["rank"])
        picked = []
        seen_sigs = set()
        for i in ranked:
            sig = deck_signature(i["card_ids"])
            if sig in seen_sigs:
                continue  # 同一構築は最良 rank の1件だけ残す
            seen_sigs.add(sig)
            cc = Counter(i["card_ids"])
            cards = [
                {"card_id": cid, "count": k}
                for cid, k in sorted(cc.items(), key=lambda x: (-x[1], x[0]))
            ]
            picked.append(
                {
                    "rank": i["rank"],
                    "team": i["team"],
                    "score": i["score"],
                    "episode_id": i["episode_id"],
                    "cards": cards,
                }
            )
            if len(picked) >= top_n:
                break
        if picked:
            result[a] = picked
    return result


def write_markdown(report, path: Path):
    m = report["meta"]
    lines = []
    lines.append("# 環境メタ分析レポート\n")
    lines.append(
        f"- 総デッキ出現: **{m['total_decks']}**  / うち rank 既知 top(≤{m['top_rank_cutoff']})={m['n_top']}, "
        f"field={m['n_field']}, rank不明={m['n_unknown_rank']}\n"
    )
    lines.append("\n## アーキタイプ・シェア (上位 vs 全体)\n")
    lines.append("| アーキタイプ | 全体% | 出現数 | 使用者数 | 上位% | field% | 上位-field |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for r in report["archetype_share"]:
        lines.append(
            f"| {r['archetype']} | {r['overall_pct']} | {r['overall_appearances']} | "
            f"{r['player_count']} | {r['top_pct']} | {r['field_pct']} | {r['top_minus_field_pct']:+} |"
        )
    lines.append(f"\n## `other` の主軸カード内訳 (未登録アーキタイプ候補, other総数={report['other_total']})\n")
    lines.append("| カード | ID | other内% | 出現数 |")
    lines.append("|---|--:|--:|--:|")
    for c in report["other_candidates"]:
        lines.append(f"| {c['name']} | {c['card_id']} | {c['pct_of_other']} | {c['count']} |")
    lines.append("\n## カード採用率 (各アーキタイプ上位10カード)\n")
    for a, d in report["card_adoption"].items():
        lines.append(f"\n### {a} (distinct builds={d['distinct_builds']})\n")
        lines.append("| カード | 採用% | 平均枚数 |")
        lines.append("|---|--:|--:|")
        for c in d["cards"][:10]:
            lines.append(f"| {c['name']} | {c['adoption_pct']} | {c['avg_copies']} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-rank", type=int, default=100, help="この rank 以下を『上位』とみなす")
    ap.add_argument("--top-decks", type=int, default=5, help="アーキタイプ別に抽出する上位デッキ数")
    args = ap.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = load_card_names()
    labels = load_labels()
    scores = load_episode_scores()
    instances = build_instances(labels, scores)

    all_freq = Counter()
    for inst in instances:
        for cid in set(inst["card_ids"]):
            all_freq[cid] += 1

    shares, meta = compute_shares(instances, args.top_rank)
    other_candidates, n_other = compute_other_candidates(instances, names, all_freq)
    adoption = compute_card_adoption(instances, names, top_rank=args.top_rank)
    top_decks = compute_top_decks(instances, names, top_n=args.top_decks)

    report = {
        "meta": meta,
        "archetype_share": shares,
        "other_candidates": other_candidates,
        "other_total": n_other,
        "card_adoption": adoption,
        "top_decks": top_decks,
    }
    (_OUT_DIR / "meta_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, _OUT_DIR / "meta_report.md")

    print(f"decks={meta['total_decks']} top={meta['n_top']} field={meta['n_field']} unknown={meta['n_unknown_rank']}")
    print(f"archetypes={len(shares)} other_candidates={len(other_candidates)} adoption_archetypes={len(adoption)} top_deck_archetypes={len(top_decks)}")
    print(f"wrote: {_OUT_DIR / 'meta_report.json'}")
    print(f"wrote: {_OUT_DIR / 'meta_report.md'}")


if __name__ == "__main__":
    main()
