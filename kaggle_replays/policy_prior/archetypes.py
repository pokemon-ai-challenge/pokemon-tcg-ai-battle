#!/usr/bin/env python3
"""リプレイのデッキをアーキタイプ別に貪欲クラスタリングする。

## デッキの取り出し

各リプレイの `steps[1][player_index]["action"]` が、そのプレイヤーが実際に使った
60枚のデッキ(card_id のリスト)。デッキ選択ターン(``steps[0]``)の次、対戦開始直前の
ステップに両プレイヤーの action として記録されている(build_dataset.py が decision を
数える ``steps[1:]`` の直前にあたる)。

## 貪欲クラスタリングの手順

1. リプレイをファイル名でソートし、各リプレイ内は player_index 0, 1 の順に処理する
   (実行するたびに同じ結果になるよう、順序を完全に決定的にする)。
2. デッキを1つずつ見ていき、既存クラスタを **生成された順** に走査する。
   そのクラスタの「シード」(そのクラスタに最初に割り当てられたデッキの多重集合)との
   **一致枚数**(``sum(min(自分の枚数, シードの枚数) for card_id in ...)``, 0〜60)が
   閾値(既定 50)以上になった **最初の** クラスタに追加する。
   どのクラスタとも閾値に届かなければ、このデッキ自身をシードとする新規クラスタを作る。
3. 全デッキを処理し終えたら、クラスタをサンプル数(所属デッキ数)の多い順に並べ替え、
   ``archetype1``, ``archetype2``, ... と採番する。
4. 各クラスタの **代表デッキ** は「クラスタ内で最も多く使われた完全一致のデッキリスト」
   (mode)。複数の完全一致デッキを平均する等の合成はしない
   ―― 合成すると誰も実戦で試したことのない構築になり、カード間のシナジーが壊れるため。

閾値・順序・「最初に見つかったクラスタに割り当てる」という3点はすべて、
300リプレイに対する実測 (32アーキタイプ、上位5クラスタの内訳) と一致することで
検証済み(``python archetypes.py`` で再検証できる)。

## 使い方

  python archetypes.py                       # 300リプレイに対する検証レポートを表示
  python archetypes.py --replays-dir ./replays --threshold 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_DEFAULT_REPLAYS_DIR = _HERE.parent / "replays"
_DEFAULT_THRESHOLD = 50


@dataclass
class DeckSample:
    episode_id: str
    player_index: int
    deck: tuple[int, ...]  # 60枚、card_id 昇順。カードの並び順に意味はなく多重集合として扱う
    # ため、ソート済みタプルを正準形にする(そうしないと同じ構築でも action の並びが
    # 対戦ごとに違うだけで「別の完全一致デッキ」に数えられてしまい、代表デッキの
    # 出現回数がずれる)。

    @property
    def counts(self) -> Counter:
        return Counter(self.deck)


@dataclass
class Cluster:
    """生成順に管理する1クラスタ。"""

    seed_counts: Counter  # このクラスタを開いたデッキの多重集合(比較用に固定。以後更新しない)。
    member_indices: list[int] = field(default_factory=list)  # samples 内のインデックス


def load_deck_samples(replays_dir: Path) -> list[DeckSample]:
    """全リプレイから (episode_id, player_index, デッキ) を決定的な順序で読み出す。"""
    samples: list[DeckSample] = []
    paths = sorted(replays_dir.glob("episode-*-replay.json"))
    for path in paths:
        episode_id = path.stem.split("-")[1]
        replay = json.loads(path.read_text(encoding="utf-8"))
        deck_step = replay["steps"][1]
        for player_index in (0, 1):
            action = deck_step[player_index].get("action")
            if not action or not isinstance(action, list):
                raise ValueError(
                    f"{path.name}: steps[1][{player_index}]['action'] がデッキ(60枚)ではありません: {action!r}"
                )
            samples.append(DeckSample(episode_id, player_index, tuple(sorted(action))))
    return samples


def _multiset_overlap(a: Counter, b: Counter) -> int:
    """2つの多重集合の共通枚数(枚数まで考慮)。0〜60。"""
    if len(a) > len(b):
        a, b = b, a
    return sum(min(count, b.get(card_id, 0)) for card_id, count in a.items())


def cluster_decks(samples: list[DeckSample], threshold: int = _DEFAULT_THRESHOLD) -> list[Cluster]:
    """貪欲クラスタリング本体。クラスタは **生成順**(まだ名前は付けない)で返す。"""
    clusters: list[Cluster] = []
    for idx, sample in enumerate(samples):
        deck_counts = sample.counts
        assigned = None
        for cluster in clusters:
            if _multiset_overlap(deck_counts, cluster.seed_counts) >= threshold:
                assigned = cluster
                break  # 最初に見つかった一致クラスタを採用する(貪欲)
        if assigned is not None:
            assigned.member_indices.append(idx)
        else:
            clusters.append(Cluster(seed_counts=deck_counts, member_indices=[idx]))
    return clusters


def representative_deck(cluster: Cluster, samples: list[DeckSample]) -> tuple[tuple[int, ...], int]:
    """クラスタ内で最も多く使われた完全一致のデッキリストと、その出現回数を返す。

    同数タイの場合はクラスタ内で先に現れた方を採る(``Counter.most_common`` は
    同数のとき挿入順を保つ)。
    """
    exact_counts: Counter[tuple[int, ...]] = Counter()
    for idx in cluster.member_indices:
        exact_counts[samples[idx].deck] += 1
    deck, count = exact_counts.most_common(1)[0]
    return deck, count


def name_and_summarize(
    clusters: list[Cluster], samples: list[DeckSample]
) -> tuple[dict[tuple[str, int], str], dict[str, dict]]:
    """クラスタをサンプル数の多い順に archetype1.. と採番し、
    (episode_id, player_index) -> archetype名 の対応表と、archetype毎のサマリを返す。
    """
    ordered = sorted(clusters, key=lambda c: len(c.member_indices), reverse=True)

    mapping: dict[tuple[str, int], str] = {}
    summary: dict[str, dict] = {}
    for rank, cluster in enumerate(ordered, start=1):
        name = f"archetype{rank}"
        rep_deck, rep_count = representative_deck(cluster, samples)
        exact_counts: Counter[tuple[int, ...]] = Counter()
        for idx in cluster.member_indices:
            sample = samples[idx]
            mapping[(sample.episode_id, sample.player_index)] = name
            exact_counts[sample.deck] += 1
        summary[name] = {
            "n_samples": len(cluster.member_indices),
            "representative_deck": list(rep_deck),
            "representative_count": rep_count,
            "n_distinct_exact_decklists": len(exact_counts),
            "n_unique_cards_in_representative": len(set(rep_deck)),
        }
    return mapping, summary


def build_archetype_map(
    replays_dir: Path = _DEFAULT_REPLAYS_DIR, threshold: int = _DEFAULT_THRESHOLD
) -> tuple[dict[tuple[str, int], str], dict[str, dict]]:
    """メイン関数: replays_dir から (episode_id, player_index) -> archetype名 の対応表を作る。

    戻り値は (mapping, summary)。summary は archetype名 -> {n_samples, representative_deck, ...}。
    """
    samples = load_deck_samples(replays_dir)
    clusters = cluster_decks(samples, threshold)
    return name_and_summarize(clusters, samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replays-dir", type=Path, default=_DEFAULT_REPLAYS_DIR)
    parser.add_argument("--threshold", type=int, default=_DEFAULT_THRESHOLD)
    parser.add_argument("--out", type=Path, default=None, help="対応表をJSONLで書き出す(任意)")
    args = parser.parse_args()

    mapping, summary = build_archetype_map(args.replays_dir, args.threshold)
    n_samples = len(mapping)
    n_archetypes = len(summary)
    print(f"標本{n_samples} / {n_archetypes}アーキタイプ  (threshold={args.threshold})")
    for name in sorted(summary, key=lambda n: int(n.replace("archetype", ""))):
        s = summary[name]
        print(
            f"{name}: クラスタ{s['n_samples']:>4}標本 / 最頻構築{s['representative_count']:>4}回 / "
            f"{s['n_unique_cards_in_representative']:>2}種"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for (episode_id, player_index), name in sorted(mapping.items()):
                f.write(
                    json.dumps(
                        {"episode_id": episode_id, "player_index": player_index, "archetype": name},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"対応表を書き出しました: {args.out}")


if __name__ == "__main__":
    main()
