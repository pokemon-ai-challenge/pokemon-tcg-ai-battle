#!/usr/bin/env python3
"""アーキタイプ定義の老化・誤判定を機械的に洗い出す監査ツール(読み取り専用)。

きっかけ: 2026-08 のデータで「オーガポン みどりのめんex を87デッキが採用しているのに
ogerpon_teal_ex とラベルされたのは4件」という状態が見つかった。原因は2種類あり、
どちらも他アーキタイプでも起こりうるので、同じ検査を全アーキタイプに機械適用する。

検出する不具合:
  [A] 取りこぼし (under-capture)
      アーキタイプAの固有カード(anchor / exclusive_core / signature /
      strong_evolution_line = そのデッキの主役そのもの)を持つのに、A以外の
      ラベルが付いたデッキ。相手側Bの固有カードも持っているなら正当な複合デッキなので
      除外し、「Bの固有カードを持たないのにBになっている」ものだけを疑わしいとして数える。
  [B] 他人の看板でゲートを通る (cross-gating)
      ラベルAが付いているのに、A自身の固有カードを1枚も持たないデッキ。
      label_decks.py は shared_anchor もゲート役として扱うため、
      「カミツオロチexが1枚も無いのに kamitsuorochi_ex」のような判定が起きる。
  [C] other クラスタ
      other に落ちたデッキで頻出するポケモンを列挙する。未登録アーキタイプの兆候。

使い方:
  python audit_archetype_labels.py --deck-db ./output/deck_db_g2.jsonl \\
      --deck-labels ./output/deck_labels_g2.jsonl \\
      --rough-predictor-json ./rough_predictor_g2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent

# 「そのアーキタイプ自身の看板」とみなす role。shared_anchor は複数アーキタイプで
# 共有される役なので、あえて含めない(含めると [B] を検出できない)。
_OWN_IDENTITY_ROLES = {"anchor", "exclusive_core", "signature", "strong_evolution_line"}


def _entry_names(entry: dict[str, Any]) -> list[str]:
    if "name" in entry:
        return [entry["name"]]
    return list(entry.get("names", []))


def own_identity_names(archetype: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for entry in archetype.get("cards", []):
        if entry.get("role") in _OWN_IDENTITY_ROLES:
            names.update(_entry_names(entry))
    for role, entries in archetype.get("role_cards", {}).items():
        for entry in entries:
            if entry.get("role", role) in _OWN_IDENTITY_ROLES:
                names.update(_entry_names(entry))
    return names


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck-db", default=str(_HERE / "output" / "deck_db_g2.jsonl"))
    parser.add_argument("--deck-labels", default=str(_HERE / "output" / "deck_labels_g2.jsonl"))
    parser.add_argument("--rough-predictor-json", default=str(_HERE / "rough_predictor_g2.json"))
    parser.add_argument("--out", default=None, help="Markdownレポートの出力先(省略時は標準出力のみ)")
    parser.add_argument("--min-report", type=int, default=3, help="この件数以上の不一致だけ報告する")
    args = parser.parse_args()

    config = json.loads(Path(args.rough_predictor_json).read_text(encoding="utf-8"))
    archetypes = config["archetypes"]
    identity = {name: own_identity_names(a) for name, a in archetypes.items()}

    decks = {(r["episode_id"], r["player_index"]): r["deck_card_names"] for r in load_jsonl(Path(args.deck_db))}
    labels = {(r["episode_id"], r["player_index"]): r["archetype"] for r in load_jsonl(Path(args.deck_labels))}

    def has_identity(deck: dict[str, int], arch: str) -> bool:
        return any(deck.get(n, 0) > 0 for n in identity.get(arch, ()))

    lines: list[str] = ["# アーキタイプ定義 監査レポート", ""]
    lines.append(f"- デッキ数: {len(decks)}")
    lines.append(f"- 定義: {args.rough_predictor_json}")
    lines.append("")

    # --- [A] 取りこぼし ---
    lines += ["## [A] 取りこぼし: 固有カードを持つのに別ラベル", "",
              "「奪った側の固有カードも持っている」複合デッキは正当なので除外している。", "",
              "| 本来 | 実際のラベル | 件数 | 代表的な固有カード |", "|---|---|---:|---|"]
    under = Counter()
    under_cards: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for key, deck in decks.items():
        actual = labels.get(key)
        for arch, names in identity.items():
            if arch == actual or not names:
                continue
            owned = [n for n in names if deck.get(n, 0) > 0]
            if not owned:
                continue
            if actual and actual != "other" and has_identity(deck, actual):
                continue  # 相手側の看板もある = 複合デッキ。誤判定とは言えない
            under[(arch, actual)] += 1
            for n in owned:
                under_cards[(arch, actual)][n] += 1
    for (arch, actual), count in under.most_common():
        if count < args.min_report:
            continue
        top = ", ".join(n for n, _ in under_cards[(arch, actual)].most_common(2))
        lines.append(f"| {arch} | {actual} | {count} | {top} |")
    if not any(c >= args.min_report for c in under.values()):
        lines.append("| (なし) | | | |")

    # --- [B] 他人の看板でゲートを通っている ---
    lines += ["", "## [B] 自分の固有カードを1枚も持たないラベル", "",
              "shared_anchor 経由でゲートを通っている疑い。定義側の役割を見直す候補。", "",
              "| ラベル | 固有カード無し | ラベル総数 | 比率 |", "|---|---:|---:|---:|"]
    total_by_label = Counter(labels.values())
    ghost = Counter()
    for key, deck in decks.items():
        actual = labels.get(key)
        if not actual or actual == "other" or not identity.get(actual):
            continue
        if not has_identity(deck, actual):
            ghost[actual] += 1
    for arch, count in ghost.most_common():
        total = total_by_label[arch]
        lines.append(f"| {arch} | {count} | {total} | {count / total * 100:.1f}% |")
    if not ghost:
        lines.append("| (なし) | | | |")

    # --- [C] other クラスタ ---
    other_keys = [k for k, v in labels.items() if v == "other"]
    lines += ["", f"## [C] other クラスタ ({len(other_keys)}件 / {len(decks)}件)", "",
              "未登録アーキタイプの兆候。上位のポケモンを列挙する。", "",
              "| カード | other内の採用デッキ数 | other内比率 |", "|---|---:|---:|"]
    pokemon_names: set[str] = set()
    for arch in archetypes.values():
        for entry in arch.get("cards", []):
            pokemon_names.update(_entry_names(entry))
        for entries in arch.get("role_cards", {}).values():
            for entry in entries:
                pokemon_names.update(_entry_names(entry))
    cc = Counter()
    for key in other_keys:
        for name, n in decks[key].items():
            if n > 0:
                cc[name] += 1
    for name, count in cc.most_common(25):
        if other_keys and count / len(other_keys) >= 0.25:
            known = " (定義済カード)" if name in pokemon_names else ""
            lines.append(f"| {name}{known} | {count} | {count / len(other_keys) * 100:.0f}% |")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nレポートを {args.out} に書き出しました")


if __name__ == "__main__":
    main()
