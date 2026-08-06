#!/usr/bin/env python3
"""評価プールに使うアーキタイプのデッキが「健全」かどうかを機械的にチェックする。

## なぜこのチェックが要るのか

固定4アーキタイプ(alakazam, crustle, marnie_grimmsnarl_ex, archaludon_ex)で
評価プールを組んでいたところ、``archaludon_ex`` のデッキ(01.csv)は
たねポケモンが5枚しか入っておらず、初手7枚での**マリガン確率が52.5%**もある
壊れたデッキだったことが判明した。マリガンを繰り返すと場が作れないまま
デッキが尽きて「ベンチ切れ」で終局するケースが大半になり、実測でも
archaludon_ex 戦の終局理由の大半がベンチ切れ(6〜7ターンで終わる)だった。

これは相手にとって**ほぼ無料の勝ち筋**を供給してしまうということであり、
4アーキタイプ合計の勝率をアーキタイプ間の実力差とは無関係な要因で
不当に押し上げ、他の変化(学習側の強さの変化など)を薄めて見えなくしてしまう。
つまり壊れたデッキを評価プールに混ぜると、勝率という指標そのものが
何を測っているのか分からなくなる。

このスクリプトは、評価・学習プールに新しいアーキタイプを追加するたびに
同じ事故を繰り返さないよう、デッキ健全性(マリガン確率)と教師データ量
(模倣学習に使えるサンプル数)を機械的にチェックするための標準ツールとして
新規作成した。

## チェック内容

``kaggle_replays/meta_analysis/archetype_decks/*/01.csv`` を全て読み、
アーキタイプごとに以下を出す:

  - 総枚数(60枚のはず)
  - たねポケモン枚数(``card_cache.get_card(card_id).basic`` で判定)
  - 初手7枚でのマリガン確率(超幾何分布: ``comb(60-basics, 7) / comb(60, 7)``。
    「7枚の初手にたねポケモンが1枚も無い」確率で、そのたびに引き直しになる)
  - 教師データ量(``kaggle_replays/deck_predictor/output/deck_labels.jsonl`` で
    そのアーキタイプにラベル付けされた (episode_id, player_index) の件数。
    模倣学習(extract_policy_dataset.py)がそのアーキタイプの意思決定点を
    抽出できる元データ量に対応する)

判定は閾値ベース:
  - マリガン確率 35% 以上 -> 「デッキ不良」(評価・学習プールから除外を検討)
  - 教師データ量 300 件未満 -> 「教師不足」(模倣学習の質が低くなりうるが、
    除外理由にはしない。相手として次元と挙動が揃っていることが目的であって
    強さそのものは求めていないため)

使い方:
  python deck_health_check.py
  python deck_health_check.py --archetype alakazam crustle
  python deck_health_check.py --out health_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "sample_submission"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from ptcg_ai.shared.card_cache import get_card  # noqa: E402

DECK_DIR = _HERE / "meta_analysis" / "archetype_decks"
DECK_LABELS_PATH = _HERE / "deck_predictor" / "output" / "deck_labels.jsonl"

DECK_SIZE = 60
OPENING_HAND = 7

#: マリガン率がこれ以上ならデッキ不良と判定する(archaludon_ex の実測 52.5% を
#: 反面教師に、健全なデッキの実測(25.9%〜34.6%)より明確に高い水準として設定)。
MULLIGAN_THRESHOLD = 0.35

#: 教師データ件数がこれ未満なら教師不足と警告する(除外理由にはしない)。
TEACHER_THRESHOLD = 300


def read_deck_csv(path: Path) -> list[int]:
    """deck.csv 形式(改行区切り、カンマ区切りいずれも可)の60枚デッキを読む。

    league/run_league.py の read_deck_csv_file と同じ緩いパース規約
    (このスクリプトを kaggle_replays 単体で完結させるため独立実装。
    デッキCSVのパース自体は数行なので重複を許容する)。
    """
    text = path.read_text(encoding="utf-8")
    deck: list[int] = []
    for raw_value in text.replace(",", "\n").splitlines():
        value = raw_value.strip()
        if not value or value.startswith("#"):
            continue
        deck.append(int(value))
    return deck


def count_basics(deck: list[int]) -> int:
    """デッキ内のたねポケモン(card.basic == True)の枚数を数える。"""
    return sum(1 for card_id in deck if get_card(card_id).basic)


def mulligan_probability(basics: int, deck_size: int = DECK_SIZE, hand: int = OPENING_HAND) -> float:
    """初手 ``hand`` 枚にたねポケモンが1枚も無い(マリガンになる)確率。

    超幾何分布: 60枚からたね以外(60-basics枚)だけを7枚引く場合の数 / 60枚から7枚引く場合の数。
    """
    if basics <= 0:
        return 1.0
    return comb(deck_size - basics, hand) / comb(deck_size, hand)


def load_teacher_counts(path: Path) -> Counter:
    """deck_labels.jsonl からアーキタイプごとのラベル件数を数える。

    存在しない場合は警告して空の Counter を返す(extract_policy_dataset.py と
    同じ「エラーにせず空データ扱い」の方針を踏襲)。
    """
    if not path.exists():
        print(f"[警告] 教師データラベルが見つかりません: {path}", file=sys.stderr)
        return Counter()
    counts: Counter = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            archetype = row.get("archetype")
            if archetype:
                counts[archetype] += 1
    return counts


def check_deck(
    archetype: str,
    deck_csv: Path,
    teacher_counts: Counter,
    mulligan_threshold: float = MULLIGAN_THRESHOLD,
    teacher_threshold: int = TEACHER_THRESHOLD,
) -> dict:
    deck = read_deck_csv(deck_csv)
    basics = count_basics(deck)
    mulligan = mulligan_probability(basics, deck_size=len(deck))
    teacher = teacher_counts.get(archetype, 0)

    reasons = []
    if mulligan >= mulligan_threshold:
        reasons.append("デッキ不良")
    if teacher < teacher_threshold:
        reasons.append("教師不足")
    verdict = "・".join(reasons) if reasons else "OK"

    return {
        "archetype": archetype,
        "deck_csv": str(deck_csv),
        "total_cards": len(deck),
        "basics": basics,
        "mulligan_prob": mulligan,
        "teacher_count": teacher,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck-dir", default=str(DECK_DIR), help="archetype_decks/ のパス")
    parser.add_argument("--deck-labels-path", default=str(DECK_LABELS_PATH), help="deck_labels.jsonl のパス")
    parser.add_argument("--archetype", nargs="*", default=None,
                         help="チェック対象のアーキタイプ名(省略時は deck-dir 配下の全アーキタイプ)")
    parser.add_argument("--deck-file", default="01.csv", help="各アーキタイプフォルダ内で使うデッキCSVファイル名")
    parser.add_argument("--mulligan-threshold", type=float, default=MULLIGAN_THRESHOLD)
    parser.add_argument("--teacher-threshold", type=int, default=TEACHER_THRESHOLD)
    parser.add_argument("--out", default=None, help="結果をJSONで書き出すパス(省略時は標準出力の表のみ)")
    args = parser.parse_args()

    mulligan_threshold = args.mulligan_threshold
    teacher_threshold = args.teacher_threshold

    deck_dir = Path(args.deck_dir)
    if args.archetype:
        archetypes = args.archetype
    else:
        archetypes = sorted(p.name for p in deck_dir.iterdir() if p.is_dir())

    teacher_counts = load_teacher_counts(Path(args.deck_labels_path))

    results = []
    for archetype in archetypes:
        deck_csv = deck_dir / archetype / args.deck_file
        if not deck_csv.exists():
            print(f"[警告] デッキCSVが見つかりません: {deck_csv}", file=sys.stderr)
            continue
        results.append(check_deck(archetype, deck_csv, teacher_counts, mulligan_threshold, teacher_threshold))

    header = f"{'archetype':<24} {'枚数':>4} {'たね':>4} {'マリガン率':>10} {'教師':>7}  判定"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['archetype']:<24} {r['total_cards']:>4} {r['basics']:>4} "
            f"{r['mulligan_prob'] * 100:>9.1f}% {r['teacher_count']:>7}  {r['verdict']}"
        )

    n_bad_deck = sum(1 for r in results if "デッキ不良" in r["verdict"])
    n_low_teacher = sum(1 for r in results if "教師不足" in r["verdict"])
    print()
    print(f"合計 {len(results)} アーキタイプ: デッキ不良 {n_bad_deck} 件 / 教師不足 {n_low_teacher} 件"
          f"(閾値: マリガン率>={mulligan_threshold*100:.0f}%, 教師<{teacher_threshold})")

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n結果を書き出しました: {out_path}")


if __name__ == "__main__":
    main()
