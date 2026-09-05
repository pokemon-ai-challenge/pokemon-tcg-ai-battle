#!/usr/bin/env python3
"""ダウンロード済みのリプレイJSONから、模倣ポリシー学習用の意思決定点データセット
(JSONL, gzip圧縮)を作る。

extract_training_data.py と同じリプレイ形式の規約を使う: steps[i][player]["status"] が
"ACTIVE" のときだけそのプレイヤーが実際に意思決定を行っており、
steps[i][player]["observation"] がそのプレイヤーに提示された観測(選択肢を含む
SelectData)、steps[i+1][player]["action"] がその観測に対して実際に選んだ行動
(選択インデックス列)に対応する。i+1 が存在しない(i == len(steps)-1)場合は
そのステップを対象外とする。

Step2(模倣ポリシー)は design.md §8 / step2-design.md §2.1 の確定方針に従い、
1つのアーキタイプ使用プレイヤーの局面に限定する。対象アーキタイプは --archetype
で指定する(既定 alakazam、後方互換)。判定は
kaggle_replays/deck_predictor/output/deck_labels.jsonl で
(episode_id, player_index) -> archetype == 指定値 とラベル付けされた
プレイヤーのみ。deck_labels.jsonl が存在しない場合はエラーにせず、
「全件フィルタ対象外」として警告した上で空データセットを出力する。

さらに step2-design.md §2.4 の初期スコープ限定に従い、以下も満たす局面のみを採用する:
  - obs["current"] が存在する(初回デッキ選択などはスキップ)
  - obs["select"] が存在する
  - 実際に選ばれた action が空でない
  - action の各要素が obs["select"]["option"] の範囲内・重複なし・個数が
    minCount..maxCount の範囲内(外れる場合は異常データとしてスキップ)

2026-08-12: maxCount > 1(複数選択)は以前はここで一律スキップしていたが、現在は採用対象。
選ばれた全選択肢を chosen_indices に、後方互換用に先頭要素を chosen_index にも入れる
(下記参照)。

満たさない局面はスキップし、理由別に件数を集計して監査レポートに残す。

2026-08-12(requirements-kamitsuorochi-2026-08-12.md 「What to build」): maxCount > 1(複数選択)
は従来スキップしていたが、これを採用対象に変える。TO_HAND(ハイパーボール/むしとりセット等の
サーチ)・DISCARD(ハイパーボールの2枚トラッシュ等)がこの対象の大半を占め、これらは
このデッキ(進化ライン構築)の核心的な意思決定点のため、スキップしたままでは学習できない。
複数選択の行は選ばれた全選択肢を ``chosen_indices``(list[int])に持つ。既存の ``chosen_index``
(単一int)は後方互換のため引き続き全行に出力し、複数選択行では選ばれた集合の先頭要素を入れる
(build_features.py 側の記録と合わせる)。

サイズ削減のため、observation の "logs" フィールド(イベント履歴)は書き出し前に
削除する。extract_value_dataset.py と違い、選択肢特徴のエンコードに使うため
"select" フィールドは残す。出力はgzip圧縮する。

fetch_top_episodes.py が作る index/episodes_master.jsonl を結合し、
extract_value_dataset.py と同様に rank_at_fetch / leaderboard_score_at_fetch を
各行に付与する(取得時点でそのプレイヤーが何位・何点だったか。マスターインデックス
未登録のエピソードは null になる)。

使い方:
  python extract_policy_dataset.py
  python extract_policy_dataset.py --limit 50 --audit
  python extract_policy_dataset.py --audit --audit-out ./policy_net/audit_report.md
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
# (rank_at_fetch のバケット定義を重複させないため)。extract_value_dataset.py と同じ方法。
sys.path.insert(0, str(_HERE / "deck_predictor"))
from episode_window import rank_bucket  # noqa: E402


TURN_BANDS = ["1-2", "3-5", "6-10", "11+", "不明"]

SKIP_REASONS = (
    "not_alakazam",
    "no_current",
    "no_select",
    "bad_action_count",
    "malformed_action",
)


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

    存在しない場合は空辞書を返す(呼び出し側で「全件フィルタ対象外」の警告を出す)。
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


def iter_decision_points(
    replay: dict,
    episode_id: str,
    archetype_labels: dict[tuple[str, int], str],
    master_row: dict | None,
    stats: "Stats",
    target_archetype: str,
):
    """1リプレイから、target_archetype 使用プレイヤーの単純選択の意思決定点を列挙する。

    採用条件を満たさない局面はここで理由別にカウントしてスキップする。
    """
    steps = replay["steps"]
    players_meta = {p["player_index"]: p for p in master_row["players"]} if master_row else {}

    for player_index in (0, 1):
        if archetype_labels.get((episode_id, player_index)) != target_archetype:
            # 対象アーキタイプ以外(またはラベル欠損)は丸ごと対象外。
            # ステップ数分をまとめてカウントするより、局面単位のスキップ理由と
            # 同じ粒度で数えるため、あとで局面走査に混ぜてカウントする。
            for i in range(len(steps) - 1):
                agent_step = steps[i][player_index]
                if agent_step.get("status") != "ACTIVE":
                    continue
                stats.record_skip("not_alakazam")
            continue

        own_meta = players_meta.get(player_index, {})
        for i in range(len(steps) - 1):
            agent_step = steps[i][player_index]
            if agent_step.get("status") != "ACTIVE":
                continue

            obs = agent_step.get("observation")
            if not obs or obs.get("current") is None:
                stats.record_skip("no_current")
                continue

            select = obs.get("select")
            if not select:
                stats.record_skip("no_select")
                continue

            # action は None のこともある(次ステップでそのプレイヤーが行動を記録して
            # いない=対象外)。len(None) で落ちないよう None も bad_action_count 扱いで
            # スキップする(dragapult_ex のリプレイで実際に None が観測された)。
            # 2026-08-12: maxCount>1(複数選択)も採用対象にするため、action の長さは
            # 「非空」だけを条件にし、正確な件数チェック(minCount/maxCount範囲・重複無し)は
            # 下の malformed_action 判定にまとめる。
            action = steps[i + 1][player_index].get("action")
            if action is None or len(action) == 0:
                stats.record_skip("bad_action_count")
                continue

            options = select.get("option") or []
            min_count = select.get("minCount")
            max_count = select.get("maxCount")
            is_malformed = (
                len(set(action)) != len(action)  # 重複
                or not all(0 <= idx < len(options) for idx in action)  # 範囲外
                or (min_count is not None and len(action) < min_count)
                or (max_count is not None and len(action) > max_count)
            )
            if is_malformed:
                stats.record_skip("malformed_action")
                continue

            chosen_indices = list(action)
            chosen_index = chosen_indices[0]  # 後方互換: 既存の単一選択消費コードはこの列を読む

            reduced_obs = {k: v for k, v in obs.items() if k != "logs"}
            turn = obs["current"].get("turn")
            yield {
                "episode_id": episode_id,
                "player_index": player_index,
                "step_index": i,
                "turn": turn,
                "select_type": select.get("type"),
                "select_context": select.get("context"),
                "n_options": len(options),
                "chosen_index": chosen_index,
                "chosen_indices": chosen_indices,
                "rank_at_fetch": own_meta.get("rank_at_fetch"),
                "leaderboard_score_at_fetch": own_meta.get("leaderboard_score_at_fetch"),
                "observation": reduced_obs,
            }


class Stats:
    """抽出中に集計する監査用の統計。--audit 指定時のみレポートに書き出す。"""

    def __init__(self) -> None:
        self.n_replays_total = 0
        self.n_replays_parsed = 0
        self.n_alakazam_episode_players = 0
        self.n_records = 0
        self.n_multi_select_records = 0  # 2026-08-12: maxCount>1 で採用された行数(スキップではない)
        self.skip_reasons: Counter[str] = Counter()
        self.select_type_counts: Counter[int] = Counter()
        self.select_context_counts: Counter[int] = Counter()
        self.turn_band_counts: Counter[str] = Counter()
        self.rank_present = 0
        self.rank_absent = 0
        self.rank_bucket_counts: Counter[str] = Counter()

    def record_skip(self, reason: str) -> None:
        self.skip_reasons[reason] += 1

    def record_record(self, row: dict) -> None:
        self.n_records += 1
        if len(row["chosen_indices"]) > 1:
            self.n_multi_select_records += 1
        self.select_type_counts[row["select_type"]] += 1
        self.select_context_counts[row["select_context"]] += 1
        self.turn_band_counts[turn_band(row["turn"])] += 1
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
    target_archetype: str,
) -> None:
    replay_paths = sorted(replays_dir.glob("episode-*-replay.json"))
    if limit is not None:
        replay_paths = replay_paths[:limit]
    n_total = len(replay_paths)

    for idx, replay_path in enumerate(replay_paths, start=1):
        episode_id = replay_path.stem.split("-")[1]
        stats.n_replays_total += 1
        try:
            with replay_path.open(encoding="utf-8") as rf:
                replay = json.load(rf)
        except Exception as exc:  # noqa: BLE001 - 1件の壊れたリプレイで全体を止めない
            print(f"  警告: {replay_path.name} の読み込みに失敗しました: {exc!r}", file=sys.stderr)
            continue
        stats.n_replays_parsed += 1

        for player_index in (0, 1):
            if archetype_labels.get((episode_id, player_index)) == target_archetype:
                stats.n_alakazam_episode_players += 1

        master_row = master_index.get(episode_id)
        for row in iter_decision_points(
            replay, episode_id, archetype_labels, master_row, stats, target_archetype
        ):
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats.record_record(row)

        if idx % 100 == 0 or idx == n_total:
            print(f"  {idx}/{n_total} replays processed...", file=sys.stderr)


def write_audit_report(stats: Stats, out_path: Path, audit_path: Path) -> None:
    lines: list[str] = []
    lines.append("# 模倣ポリシー学習データ 監査レポート")
    lines.append("")
    lines.append(f"- 対象リプレイディレクトリ集計対象件数: {stats.n_replays_total}")
    lines.append(f"- JSON パース成功件数: {stats.n_replays_parsed}")
    lines.append(f"- データセット出力: `{out_path}`")
    lines.append("")

    lines.append("## フーディン(alakazam)該当 episode-player 数")
    lines.append("")
    lines.append(f"- 該当 (episode_id, player_index) 数: {stats.n_alakazam_episode_players}")
    lines.append("")

    lines.append("## 採用レコード数")
    lines.append("")
    lines.append(f"- 採用: {stats.n_records}")
    lines.append(
        f"- うち複数選択(maxCount>1、chosen_indices が2件以上): {stats.n_multi_select_records} "
        f"({stats.n_multi_select_records / stats.n_records * 100 if stats.n_records else 0.0:.2f}%)"
        "  ※ 2026-08-12以前はここがスキップ理由(multi_select)だったが、現在は採用対象"
    )
    n_skipped = sum(stats.skip_reasons.values())
    lines.append(f"- スキップ計(意思決定点単位): {n_skipped}")
    lines.append("")
    lines.append("| スキップ理由 | 件数 |")
    lines.append("|---|---:|")
    reason_labels = {
        "not_alakazam": "archetypeラベル不一致/欠損 (not_alakazam)",
        "no_current": "盤面(current)欠損 (no_current)",
        "no_select": "select欠損 (no_select)",
        "bad_action_count": "action空/欠損 (bad_action_count)",
        "malformed_action": "選択インデックスが範囲外/重複/個数がminCount..maxCount外 (malformed_action)",
    }
    for reason in SKIP_REASONS:
        count = stats.skip_reasons.get(reason, 0)
        lines.append(f"| {reason_labels[reason]} | {count} |")
    lines.append("")

    lines.append("## SelectType 別件数分布")
    lines.append("")
    lines.append("| select_type | 件数 | 割合 |")
    lines.append("|---|---:|---:|")
    for select_type, count in stats.select_type_counts.most_common():
        ratio = count / stats.n_records * 100 if stats.n_records else 0.0
        lines.append(f"| {select_type} | {count} | {ratio:.2f}% |")
    lines.append("")

    lines.append("## SelectContext 別件数分布(上位15件+その他)")
    lines.append("")
    lines.append("| select_context | 件数 | 割合 |")
    lines.append("|---|---:|---:|")
    top_contexts = stats.select_context_counts.most_common(15)
    top_context_keys = {c for c, _ in top_contexts}
    for select_context, count in top_contexts:
        ratio = count / stats.n_records * 100 if stats.n_records else 0.0
        lines.append(f"| {select_context} | {count} | {ratio:.2f}% |")
    other_count = sum(
        count for c, count in stats.select_context_counts.items() if c not in top_context_keys
    )
    if other_count:
        ratio = other_count / stats.n_records * 100 if stats.n_records else 0.0
        lines.append(f"| その他 | {other_count} | {ratio:.2f}% |")
    lines.append("")

    lines.append("## ターン帯別の局面数分布")
    lines.append("")
    lines.append("| ターン帯 | 件数 | 割合 |")
    lines.append("|---|---:|---:|")
    for band in TURN_BANDS:
        count = stats.turn_band_counts.get(band, 0)
        ratio = count / stats.n_records * 100 if stats.n_records else 0.0
        lines.append(f"| {band} | {count} | {ratio:.2f}% |")
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
        help="フーディン(alakazam)使用プレイヤーへの絞り込みに使う label_decks.py の出力",
    )
    parser.add_argument(
        "--archetype", default="alakazam",
        help="模倣学習の対象アーキタイプ(deck_labels.jsonl の archetype 値)。既定は alakazam"
        "(後方互換)。他アーキタイプを学習する場合は --out も専用パスを指定すること。",
    )
    parser.add_argument("--out", default=str(_HERE / "training_data" / "policy_positions.jsonl.gz"))
    parser.add_argument("--limit", type=int, default=None, help="先頭N件のリプレイのみ処理する(動作確認用)")
    parser.add_argument("--audit", action="store_true", help="監査レポート(Markdown)も書き出す")
    parser.add_argument("--audit-out", default=str(_HERE / "policy_net" / "audit_report.md"))
    args = parser.parse_args()

    replays_dir = Path(args.replays_dir)
    master_index = load_master_index(Path(args.master_index_path))
    deck_labels_path = Path(args.deck_labels_path)
    archetype_labels = load_archetype_labels(deck_labels_path)
    if not deck_labels_path.exists():
        print(
            f"警告: {deck_labels_path} が見つかりません。{args.archetype} への絞り込みが"
            "できないため、全件フィルタ対象外(空データセット)として出力します。",
            file=sys.stderr,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    with gzip.open(out_path, "wt", encoding="utf-8") as out_f:
        process_replays(
            replays_dir, master_index, archetype_labels, out_f, args.limit, stats, args.archetype
        )

    n_skipped = sum(stats.skip_reasons.values())
    print(
        f"{stats.n_replays_total}件のリプレイ中 {stats.n_alakazam_episode_players}件の "
        f"{args.archetype} episode-player を対象に {stats.n_records}件の意思決定点を {out_path} に "
        f"書き出しました(スキップ {n_skipped}件: {dict(stats.skip_reasons)})"
    )

    if args.audit:
        audit_path = Path(args.audit_out)
        write_audit_report(stats, out_path, audit_path)
        print(f"監査レポートを {audit_path} に書き出しました")


if __name__ == "__main__":
    main()
