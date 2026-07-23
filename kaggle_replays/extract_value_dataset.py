#!/usr/bin/env python3
"""ダウンロード済みのリプレイJSONから、勝敗ラベル付きの局面データセット(JSONL, gzip圧縮)を作る。

extract_training_data.py と同じリプレイ形式の規約を使う: steps[i][player]["observation"] が
そのプレイヤーに提示された観測、steps[i][player]["status"] が "ACTIVE" のときだけそのプレイヤーが
実際に意思決定を行っている(INACTIVE側は空アクションのプレースホルダ)。本スクリプトは行動
(action)ではなく観測(observation)そのものを、リプレイ末尾の勝敗と紐づけて1行1局面で書き出す。

勝敗ラベルはリプレイJSONの**トップレベル "rewards"** フィールド(player_index順のリスト。
例: [1, -1] は player_index 0 の勝ち・1 の負けを表す)から取得する。片方が正でもう片方が負の
組み合わせのみ採用し、それ以外(引き分け・rewards欠損・不正な値)は当該エピソードを丸ごと
スキップして理由別に件数を数える(design.md §2.2)。同一エピソード内の全局面には、その局面を
観測したプレイヤー自身の rewards[player_index] の符号から決まる同じラベル(勝ち=1/負け=0)が
付く。

局面は steps[i][player]["status"] == "ACTIVE" の steps[i][player]["observation"] を1件とする。
observation に現在盤面(current)が無いもの(初回デッキ選択など)はスキップする。

サイズ削減のため、observation の "logs" フィールド(イベント履歴。バリュー学習では現在盤面のみ
使うため不要)は書き出し前に削除する。出力はgzip圧縮する。

fetch_top_episodes.py が作る index/episodes_master.jsonl を結合し、extract_training_data.py と
同様に rank_at_fetch / leaderboard_score_at_fetch を各行に付与する(取得時点でそのプレイヤーが
何位・何点だったか。マスターインデックス未登録のエピソードは null になる)。

使い方:
  python extract_value_dataset.py
  python extract_value_dataset.py --limit 50
  python extract_value_dataset.py --audit --audit-out ./value_net/audit_report.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent

# deck_predictor/episode_window.py の rank_bucket() をそのまま再利用する
# (rank_at_fetch のバケット定義を重複させないため)。
sys.path.insert(0, str(_HERE / "deck_predictor"))
from episode_window import rank_bucket  # noqa: E402


TURN_BANDS = ["1-2", "3-5", "6-10", "11+", "不明"]


def turn_band(turn: int | None) -> str:
    if turn is None:
        return "不明"
    if turn <= 2:
        return "1-2"
    if turn <= 5:
        return "3-5"
    if turn <= 10:
        return "6-10"
    return "11+"


def load_master_index(master_path: Path) -> dict[str, dict]:
    if not master_path.exists():
        return {}
    index: dict[str, dict] = {}
    with master_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[row["episode_id"]] = row
    return index


def load_archetype_labels(deck_labels_path: Path) -> dict[tuple[str, int], str]:
    """deck_predictor/label_decks.py の出力 (episode_id, player_index) -> archetype を読む。

    存在しない場合は空辞書を返す(アーキタイプ分布はスキップされる)。
    """
    labels: dict[tuple[str, int], str] = {}
    if not deck_labels_path.exists():
        return labels
    with deck_labels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[(row["episode_id"], row["player_index"])] = row["archetype"]
    return labels


def classify_rewards(rewards: Any) -> tuple[str, dict[int, int] | None]:
    """rewards(トップレベル "rewards" フィールド)を分類する。

    戻り値は (status, labels)。status は以下のいずれか:
      - "ok": 片方が正・片方が負。labels は {player_index: 0/1}(勝ち=1/負け=0)。
      - "missing": rewards フィールド自体が存在しない/None。
      - "draw": 両者の値が等しい(引き分け相当)。
      - "malformed": 上記以外の不正な形(長さ不正・片方が0など符号が一意に決まらない)。
    """
    if rewards is None:
        return "missing", None
    if not isinstance(rewards, list) or len(rewards) != 2:
        return "malformed", None
    r0, r1 = rewards
    if r0 is None or r1 is None:
        return "malformed", None
    if r0 == r1:
        return "draw", None
    if r0 > 0 and r1 < 0:
        return "ok", {0: 1, 1: 0}
    if r0 < 0 and r1 > 0:
        return "ok", {0: 0, 1: 1}
    return "malformed", None


def iter_positions(replay: dict, episode_id: str, labels: dict[int, int], master_row: dict | None):
    """1リプレイから (player_index, step_index, turn, label, reduced_observation) を列挙する。"""
    steps = replay["steps"]
    players_meta = {p["player_index"]: p for p in master_row["players"]} if master_row else {}

    for player_index in (0, 1):
        label = labels[player_index]
        own_meta = players_meta.get(player_index, {})
        for i, step_pair in enumerate(steps):
            agent_step = step_pair[player_index]
            if agent_step.get("status") != "ACTIVE":
                continue
            obs = agent_step.get("observation")
            if not obs or obs.get("current") is None:
                # 初回デッキ選択など、盤面が存在しない局面はスキップ
                continue
            reduced_obs = {k: v for k, v in obs.items() if k != "logs"}
            turn = obs["current"].get("turn")
            yield {
                "episode_id": episode_id,
                "player_index": player_index,
                "step_index": i,
                "turn": turn,
                "label": label,
                "rank_at_fetch": own_meta.get("rank_at_fetch"),
                "leaderboard_score_at_fetch": own_meta.get("leaderboard_score_at_fetch"),
                "observation": reduced_obs,
            }


class Stats:
    """抽出中に集計する監査用の統計。--audit 指定時のみレポートに書き出す。"""

    def __init__(self) -> None:
        self.n_episodes_total = 0
        self.n_episodes_adopted = 0
        self.skip_reasons: Counter[str] = Counter()
        self.n_positions = 0
        self.turn_band_counts: Counter[str] = Counter()
        self.label_counts: Counter[int] = Counter()
        self.rank_present = 0
        self.rank_absent = 0
        self.rank_bucket_counts: Counter[str] = Counter()
        self.archetype_counts: Counter[str] = Counter()
        self.matchup_counts: Counter[tuple[str, str]] = Counter()
        self.n_episodes_without_master_row = 0

    def record_episode_skip(self, reason: str) -> None:
        self.skip_reasons[reason] += 1

    def record_episode_adopted(
        self,
        episode_id: str,
        master_row: dict | None,
        archetype_labels: dict[tuple[str, int], str],
    ) -> None:
        self.n_episodes_adopted += 1
        if master_row is None:
            self.n_episodes_without_master_row += 1
        a0 = archetype_labels.get((episode_id, 0))
        a1 = archetype_labels.get((episode_id, 1))
        if a0 is not None:
            self.archetype_counts[a0] += 1
        if a1 is not None:
            self.archetype_counts[a1] += 1
        if a0 is not None and a1 is not None:
            pair = tuple(sorted((a0, a1)))
            self.matchup_counts[pair] += 1

    def record_position(self, row: dict) -> None:
        self.n_positions += 1
        self.turn_band_counts[turn_band(row["turn"])] += 1
        self.label_counts[row["label"]] += 1
        rank = row["rank_at_fetch"]
        if rank is None:
            self.rank_absent += 1
        else:
            self.rank_present += 1
            self.rank_bucket_counts[rank_bucket(rank)] += 1


def process_replays(
    replays_dir: Path,
    master_index: dict[str, dict],
    archetype_labels: dict[tuple[str, int], str],
    out_f,
    limit: int | None,
    stats: Stats,
) -> None:
    replay_paths = sorted(replays_dir.glob("episode-*-replay.json"))
    if limit is not None:
        replay_paths = replay_paths[:limit]
    n_total = len(replay_paths)

    for idx, replay_path in enumerate(replay_paths, start=1):
        episode_id = replay_path.stem.split("-")[1]
        stats.n_episodes_total += 1
        try:
            with replay_path.open(encoding="utf-8") as rf:
                replay = json.load(rf)
            status, labels = classify_rewards(replay.get("rewards"))
        except Exception as exc:  # noqa: BLE001 - 1件の壊れたリプレイで全体を止めない
            print(f"  警告: {replay_path.name} の読み込みに失敗しました: {exc!r}", file=sys.stderr)
            stats.record_episode_skip("parse_error")
            continue

        if status != "ok":
            stats.record_episode_skip(status)
            continue

        master_row = master_index.get(episode_id)
        stats.record_episode_adopted(episode_id, master_row, archetype_labels)

        for row in iter_positions(replay, episode_id, labels, master_row):
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats.record_position(row)

        if idx % 100 == 0 or idx == n_total:
            print(f"  {idx}/{n_total} replays processed...", file=sys.stderr)


def write_audit_report(stats: Stats, out_path: Path, dataset_path: Path, audit_path: Path) -> None:
    lines: list[str] = []
    lines.append("# バリューネットワーク学習データ 監査レポート")
    lines.append("")
    lines.append(f"- 対象リプレイディレクトリ集計対象件数: {stats.n_episodes_total}")
    lines.append(f"- データセット出力: `{dataset_path}`")
    lines.append("")

    lines.append("## エピソード数")
    lines.append("")
    lines.append(f"- 総数: {stats.n_episodes_total}")
    lines.append(
        f"- 採用: {stats.n_episodes_adopted} "
        f"({stats.n_episodes_adopted / stats.n_episodes_total * 100:.2f}%)"
        if stats.n_episodes_total
        else "- 採用: 0"
    )
    n_skipped = sum(stats.skip_reasons.values())
    lines.append(f"- スキップ計: {n_skipped}")
    lines.append("")
    lines.append("| スキップ理由 | 件数 |")
    lines.append("|---|---:|")
    reason_labels = {
        "draw": "引き分け(rewards 同値)",
        "missing": "rewards 欠損",
        "malformed": "rewards 不正な形",
        "parse_error": "JSON パースエラー / 例外",
    }
    for reason in ("draw", "missing", "malformed", "parse_error"):
        count = stats.skip_reasons.get(reason, 0)
        lines.append(f"| {reason_labels[reason]} ({reason}) | {count} |")
    lines.append("")
    if stats.n_episodes_without_master_row:
        lines.append(
            f"- 注意: 採用エピソードのうち {stats.n_episodes_without_master_row} 件は "
            "episodes_master.jsonl に未登録のため rank_at_fetch 等が null です。"
        )
        lines.append("")

    lines.append("## 局面数")
    lines.append("")
    lines.append(f"- 総局面数: {stats.n_positions}")
    avg = stats.n_positions / stats.n_episodes_adopted if stats.n_episodes_adopted else 0.0
    lines.append(f"- 1エピソードあたり平均局面数: {avg:.2f}")
    lines.append("")

    lines.append("## ターン帯別の局面数分布")
    lines.append("")
    lines.append("| ターン帯 | 件数 | 割合 |")
    lines.append("|---|---:|---:|")
    for band in TURN_BANDS:
        count = stats.turn_band_counts.get(band, 0)
        ratio = count / stats.n_positions * 100 if stats.n_positions else 0.0
        lines.append(f"| {band} | {count} | {ratio:.2f}% |")
    lines.append("")

    lines.append("## ラベルバランス")
    lines.append("")
    lines.append(
        "両プレイヤー視点の局面を含むため、リプレイのペア構造上ほぼ50/50になるはず"
        "(大きくずれている場合はラベル付与ロジックのバグを疑う)。"
    )
    lines.append("")
    lines.append("| label | 件数 | 割合 |")
    lines.append("|---|---:|---:|")
    for label in (1, 0):
        count = stats.label_counts.get(label, 0)
        ratio = count / stats.n_positions * 100 if stats.n_positions else 0.0
        name = "勝ち(1)" if label == 1 else "負け(0)"
        lines.append(f"| {name} | {count} | {ratio:.2f}% |")
    lines.append("")

    lines.append("## rank_at_fetch の分布")
    lines.append("")
    lines.append(f"- あり: {stats.rank_present}")
    lines.append(f"- なし(null): {stats.rank_absent}")
    lines.append("")
    lines.append("| 順位帯 | 件数(rank_at_fetch ありのうち) | 割合 |")
    lines.append("|---|---:|---:|")
    for bucket in ("1-50", "51-200", "201-1000", "1001+"):
        count = stats.rank_bucket_counts.get(bucket, 0)
        ratio = count / stats.rank_present * 100 if stats.rank_present else 0.0
        lines.append(f"| {bucket} | {count} | {ratio:.2f}% |")
    lines.append("")

    lines.append("## アーキタイプ分布・マッチアップペア")
    lines.append("")
    if stats.archetype_counts:
        lines.append(
            "`kaggle_replays/deck_predictor/output/deck_labels.jsonl`(既存の "
            "`label_decks.py` 出力)を episode_id + player_index で結合して集計。"
        )
        lines.append("")
        lines.append("### アーキタイプ分布(採用エピソード内、両プレイヤー視点で計上)")
        lines.append("")
        lines.append("| アーキタイプ | 件数 | 割合 |")
        lines.append("|---|---:|---:|")
        total_archetype = sum(stats.archetype_counts.values())
        for archetype, count in stats.archetype_counts.most_common():
            ratio = count / total_archetype * 100 if total_archetype else 0.0
            lines.append(f"| {archetype} | {count} | {ratio:.2f}% |")
        lines.append("")
        lines.append("### マッチアップペア上位20(両プレイヤーのアーキタイプが判明しているエピソードのみ)")
        lines.append("")
        lines.append("| マッチアップ | 件数 |")
        lines.append("|---|---:|")
        for (a, b), count in stats.matchup_counts.most_common(20):
            lines.append(f"| {a} vs {b} | {count} |")
        lines.append("")
    else:
        lines.append(
            "TODO: アーキタイプ分布は deck_predictor 連携で別途"
            "(`kaggle_replays/deck_predictor/output/deck_labels.jsonl` が見つかりませんでした)。"
        )
        lines.append("")

    lines.append("## 出力ファイルサイズ")
    lines.append("")
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        lines.append(f"- `{out_path}`: {size_mb:.2f} MB")
    else:
        lines.append(f"- `{out_path}`: (未生成)")
    lines.append("")

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays-dir", default=str(_HERE / "replays"))
    parser.add_argument("--master-index-path", default=str(_HERE / "index" / "episodes_master.jsonl"))
    parser.add_argument(
        "--deck-labels-path",
        default=str(_HERE / "deck_predictor" / "output" / "deck_labels.jsonl"),
        help="アーキタイプ分布・マッチアップペア集計に使う label_decks.py の出力(--audit 時のみ使用)",
    )
    parser.add_argument("--out", default=str(_HERE / "training_data" / "value_positions.jsonl.gz"))
    parser.add_argument("--limit", type=int, default=None, help="先頭N件のリプレイのみ処理する(動作確認用)")
    parser.add_argument("--audit", action="store_true", help="監査レポート(Markdown)も書き出す")
    parser.add_argument("--audit-out", default=str(_HERE / "value_net" / "audit_report.md"))
    args = parser.parse_args()

    replays_dir = Path(args.replays_dir)
    master_index = load_master_index(Path(args.master_index_path))
    archetype_labels = load_archetype_labels(Path(args.deck_labels_path)) if args.audit else {}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    with gzip.open(out_path, "wt", encoding="utf-8") as out_f:
        process_replays(replays_dir, master_index, archetype_labels, out_f, args.limit, stats)

    n_skipped = sum(stats.skip_reasons.values())
    print(
        f"{stats.n_episodes_total}件のリプレイ中 {stats.n_episodes_adopted}件を採用し、"
        f"{stats.n_positions}件の局面を {out_path} に書き出しました "
        f"(スキップ {n_skipped}件: {dict(stats.skip_reasons)})"
    )
    if stats.n_episodes_without_master_row:
        print(
            f"注意: 採用エピソードのうち{stats.n_episodes_without_master_row}件はマスターインデックスに"
            "未登録のため rank_at_fetch 等が null になっています"
        )

    if args.audit:
        audit_path = Path(args.audit_out)
        write_audit_report(stats, out_path, out_path, audit_path)
        print(f"監査レポートを {audit_path} に書き出しました")


if __name__ == "__main__":
    main()
