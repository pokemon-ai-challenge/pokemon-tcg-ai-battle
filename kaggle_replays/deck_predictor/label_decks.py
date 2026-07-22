#!/usr/bin/env python3
"""deck_db.jsonl の各60枚デッキを rough_predictor.json のキーカード定義で自動ラベリングする。

sample_submission/ptcg_ai/opponent_modeling/rough_predictor.py の実行時スコアリング
(_score_archetype 系)と同じ role_weights / combo_rules / evolution_candy_combo の考え方を、
「フルデッキリストが丸ごと見えている」前提向けに簡略化して再実装したもの。
実行時版は「今見えている公開カード」を対象にするのに対し、こちらは「デッキ60枚全部」を
対象にするので observed_card_ids のような部分観測の概念は不要で、名前ベースの単純な
枚数>0判定でよい。

ラベリング規則:
  各アーキタイプについて cards / role_cards / combo_rules の一致でスコアを積み上げ、
  最高スコアが LABEL_MIN_SCORE 以上ならそのアーキタイプ、未満なら "other"。
  LABEL_MIN_SCORE は「anchor 1枚(role_weights["anchor"]=10点)でほぼ確定」という感覚の
  初期値(=10)。 調整したい場合はこの定数を変更する。

使い方:
  python label_decks.py
  python label_decks.py --deck-db ./output/deck_db.jsonl --out ./output/deck_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_ROUGH_PREDICTOR_JSON = (
    _HERE.parent.parent / "sample_submission" / "ptcg_ai" / "opponent_modeling" / "rough_predictor.json"
)

# 「anchor 1枚(10点)で確定に近い」感覚のしきい値。調整可能な定数。
LABEL_MIN_SCORE = 10.0

# デッキ60枚が丸ごと見えている前提でのゲート条件。role_weights はもともと「部分観測からの
# 証拠の積み上げ」向けにチューニングされているため、フルデッキリストにそのまま適用すると
# core/flex/energy 止まりの汎用カード(ふしぎなアメ・シェイミ・ノコッチ・テレパス【超】等、
# 多くのデッキで共通に採用される)だけでもしきい値を超えてしまう
# (実測: alakazam が「フーディン本体なしで」50%近くにヒットする不具合が出た)。
# そこで「そのアーキタイプの主軸級カード(anchor/exclusive_core/shared_anchor/signature/
# strong_evolution_line)が実際に1枚もデッキに入っていない限り、そのアーキタイプでは
# ラベル付けしない」というゲートを追加する。フルデッキが見えている場合、主軸カードの
# 有無はほぼ決定的な証拠になるため、このゲートは妥当。
_GATE_ROLES = {"anchor", "exclusive_core", "shared_anchor", "signature", "strong_evolution_line"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _entry_names(entry: dict[str, Any]) -> list[str]:
    if "name" in entry:
        return [entry["name"]]
    return list(entry.get("names", []))


def _extract_generic_names(generic_cards: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for entry in generic_cards:
        names.update(_entry_names(entry))
    return names


def _evolution_line_names(archetype: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for entry in archetype.get("role_cards", {}).get("evolution_line", []):
        names.extend(_entry_names(entry))
    return names


def _archetype_has_any_name(archetype: dict[str, Any], names: list[str]) -> bool:
    target = set(names)
    for entry in archetype.get("cards", []):
        if target.intersection(_entry_names(entry)):
            return True
    for entries in archetype.get("role_cards", {}).values():
        for entry in entries:
            if target.intersection(_entry_names(entry)):
                return True
    return False


def _evolution_candy_combo_rule(archetype: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    combo_config = config.get("evolution_candy_combo", {})
    if not combo_config.get("enabled", False):
        return None
    candy_names = list(combo_config.get("candy_names", ["ふしぎなアメ"]))
    evolution_names = _evolution_line_names(archetype)
    if not candy_names or not evolution_names:
        return None
    if not _archetype_has_any_name(archetype, candy_names):
        return None
    return {
        "names": [candy_names[0]],
        "any_names": evolution_names,
        "bonus": float(combo_config.get("bonus", 0.0)),
    }


def _combo_rules_for_archetype(archetype: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    rules = list(archetype.get("combo_rules", []))
    generated = _evolution_candy_combo_rule(archetype, config)
    if generated is not None:
        rules.append(generated)
    return rules


def _match_entry(
    entry: dict[str, Any],
    role: str,
    deck_card_names: dict[str, int],
    role_weights: dict[str, float],
    ace_spec_bonus: float,
    generic_names: set[str],
    seen_line_keys: set[str],
    matched_cards: list[str],
) -> tuple[float, bool]:
    """1エントリを判定する。戻り値は (加点, ゲート条件[_GATE_ROLES]を満たしたか)。"""
    if role == "generic":
        return 0.0, False
    names = _entry_names(entry)
    if not names:
        return 0.0, False
    if any(name in generic_names for name in names):
        return 0.0, False

    matched_name = next((name for name in names if deck_card_names.get(name, 0) > 0), None)
    if matched_name is None:
        return 0.0, False

    line_key = entry.get("line_key")
    if line_key:
        if line_key in seen_line_keys:
            return 0.0, False
        seen_line_keys.add(line_key)

    weight = float(role_weights.get(role, 0.0))
    if weight <= 0:
        return 0.0, False
    if entry.get("ace_spec"):
        weight *= ace_spec_bonus

    matched_cards.append(matched_name)
    return weight, role in _GATE_ROLES


def score_archetype(
    deck_type: str,
    archetype: dict[str, Any],
    deck_card_names: dict[str, int],
    config: dict[str, Any],
    generic_names: set[str],
) -> tuple[float, list[str], bool]:
    """アーキタイプ1つ分をスコアリングする。戻り値は (score, matched_cards, gated)。

    ``gated`` は「主軸級カード(_GATE_ROLES)が実際に1枚でもデッキに入っていたか」。
    フルデッキ前提では、この条件を満たさない限りそのアーキタイプとしてラベル付けしない
    (score だけで判定すると汎用カードの積み上げで誤判定するため)。
    """
    role_weights = config.get("role_weights", {})
    ace_spec_bonus = float(config.get("ace_spec_bonus", 1.0))

    score = 0.0
    matched_cards: list[str] = []
    seen_line_keys: set[str] = set()
    gated = False

    for entry in archetype.get("cards", []):
        role = entry.get("role", "generic")
        weight, is_gate = _match_entry(
            entry, role, deck_card_names, role_weights, ace_spec_bonus, generic_names, seen_line_keys, matched_cards
        )
        score += weight
        gated = gated or is_gate

    for role, entries in archetype.get("role_cards", {}).items():
        for entry in entries:
            entry_role = entry.get("role", role)
            weight, is_gate = _match_entry(
                entry,
                entry_role,
                deck_card_names,
                role_weights,
                ace_spec_bonus,
                generic_names,
                seen_line_keys,
                matched_cards,
            )
            score += weight
            gated = gated or is_gate

    for combo_rule in _combo_rules_for_archetype(archetype, config):
        required_names = combo_rule.get("names", [])
        any_names = combo_rule.get("any_names", [])
        required_matched = all(deck_card_names.get(name, 0) > 0 for name in required_names)
        matched_any = next((name for name in any_names if deck_card_names.get(name, 0) > 0), None)
        any_matched = not any_names or matched_any is not None
        if required_names and required_matched and any_matched:
            bonus = float(combo_rule.get("bonus", 0.0))
            if bonus > 0:
                score += bonus
                matched_cards.append(f"combo:{'+'.join(required_names)}")

    return score, matched_cards, gated


def label_deck(
    deck_card_names: dict[str, int],
    config: dict[str, Any],
    generic_names: set[str],
    min_score: float = LABEL_MIN_SCORE,
) -> dict[str, Any]:
    best_deck_type = None
    best_score = 0.0
    best_matched: list[str] = []
    all_scores: dict[str, float] = {}

    for deck_type, archetype in sorted(config.get("archetypes", {}).items()):
        score, matched_cards, gated = score_archetype(deck_type, archetype, deck_card_names, config, generic_names)
        all_scores[deck_type] = score
        if not gated:
            # 主軸級カードが1枚も見えていないアーキタイプは、スコアがいくら高くても候補にしない。
            continue
        if score > best_score:
            best_score = score
            best_deck_type = deck_type
            best_matched = matched_cards

    if best_deck_type is not None and best_score >= min_score:
        return {"archetype": best_deck_type, "score": round(best_score, 2), "matched_cards": best_matched}
    return {
        "archetype": "other",
        "score": round(best_score, 2),
        "matched_cards": best_matched,
        "top_candidate": best_deck_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck-db", default=str(_HERE / "output" / "deck_db.jsonl"))
    parser.add_argument("--rough-predictor-json", default=str(_ROUGH_PREDICTOR_JSON))
    parser.add_argument("--out", default=str(_HERE / "output" / "deck_labels.jsonl"))
    parser.add_argument("--report", default=str(_HERE / "output" / "label_report.md"))
    parser.add_argument("--min-score", type=float, default=LABEL_MIN_SCORE)
    args = parser.parse_args()
    min_score = args.min_score

    config = load_config(Path(args.rough_predictor_json))
    generic_names = _extract_generic_names(config.get("generic_cards", []))

    deck_db_path = Path(args.deck_db)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    archetype_counts: Counter[str] = Counter()
    display_names = {dt: a.get("display_name", dt) for dt, a in config.get("archetypes", {}).items()}
    n_total = 0

    with deck_db_path.open(encoding="utf-8") as in_f, out_path.open("w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            deck_row = json.loads(line)
            result = label_deck(deck_row["deck_card_names"], config, generic_names, min_score)
            label_row = {
                "episode_id": deck_row["episode_id"],
                "player_index": deck_row["player_index"],
                "archetype": result["archetype"],
                "score": result["score"],
                "matched_cards": result["matched_cards"],
            }
            if "top_candidate" in result:
                label_row["top_candidate"] = result["top_candidate"]
            out_f.write(json.dumps(label_row, ensure_ascii=False) + "\n")
            archetype_counts[result["archetype"]] += 1
            n_total += 1

    report_lines = [
        "# デッキアーキタイプ分布レポート",
        "",
        f"- 総デッキ数: {n_total}",
        f"- しきい値 (LABEL_MIN_SCORE): {min_score}",
        "",
        "| アーキタイプ | 表示名 | 件数 | 割合 |",
        "|---|---|---:|---:|",
    ]
    print(f"総デッキ数: {n_total}  (しきい値={min_score})")
    print(f"{'アーキタイプ':30s} {'件数':>6s} {'割合':>8s}")
    for deck_type, count in archetype_counts.most_common():
        ratio = count / n_total if n_total else 0.0
        name = "other" if deck_type == "other" else display_names.get(deck_type, deck_type)
        print(f"{deck_type:30s} {count:6d} {ratio * 100:7.2f}%")
        report_lines.append(f"| {deck_type} | {name} | {count} | {ratio * 100:.2f}% |")

    report_lines.append("")
    Path(args.report).write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nラベル付き {n_total} 件を {out_path} に、分布レポートを {args.report} に書き出しました")


if __name__ == "__main__":
    main()
