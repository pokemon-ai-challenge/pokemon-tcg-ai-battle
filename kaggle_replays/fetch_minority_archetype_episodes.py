#!/usr/bin/env python3
"""相手デッキ予測器の少数クラス(アーキタイプ)を狙い撃ちしてリプレイを追加取得する。

`deck_predictor/output/deck_labels.jsonl` は各エピソードの各プレイヤーの60枚デッキを
アーキタイプでラベル付けした結果(episode_id, player_index, archetype)。ここから
クラス別のデッキ数(N)を集計し、N が少ないアーキタイプ(既定: N<50)を「少数クラス」と
判定する。1チームはほぼ同じデッキを使い続ける傾向があるため、少数クラスのデッキを
使っていた (episode_id, player_index) から team_id を逆引きし(`episodes_master.jsonl`
の players 情報を使う)、そのチームの他のエピソードを Kaggle API から探して未取得分を
`replays/` に追加ダウンロードする。

ダウンロード機構・レート制御・index 管理は `_common.py` を経由し、
`fetch_top_episodes.py` / `fetch_deep_decks.py` と同じ流儀(episodes_master.jsonl は
追記のみ・再実行は冪等)にそろえている。

使い方:
  python fetch_minority_archetype_episodes.py
  python fetch_minority_archetype_episodes.py --min-decks 50 --max-episodes 200

事前準備: `deck_predictor/extract_decks.py` と `deck_predictor/label_decks.py` を
先に実行して `deck_predictor/output/deck_labels.jsonl` を作っておくこと。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    append_master_rows,
    build_master_rows,
    download_replay,
    download_replays_parallel,
    fetch_episodes,
    load_existing_episode_ids,
    run_kaggle_json,
    set_min_interval,
)

_HERE = Path(__file__).parent


def load_deck_labels(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_master_index(master_path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    if not master_path.exists():
        return index
    with master_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[row["episode_id"]] = row
    return index


def fetch_team_submission_ids(team_id: int, limit: int) -> list[int]:
    rows = run_kaggle_json(["competitions", "team-submissions", str(team_id)])
    return [row["id"] for row in rows[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument(
        "--deck-labels",
        default=str(_HERE / "deck_predictor" / "output" / "deck_labels.jsonl"),
        help="label_decks.py の出力(クラス別デッキ数の集計元)",
    )
    parser.add_argument(
        "--min-decks", type=int, default=50,
        help="このデッキ数(N)未満のアーキタイプを少数クラスとみなす(既定50)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=200,
        help="このrunで新規ダウンロードするエピソード数の上限(全チーム合計、既定200)",
    )
    parser.add_argument(
        "--max-episodes-per-team", type=int, default=30,
        help="1チームあたり新規ダウンロードするエピソード数の上限"
        "(1チームが --max-episodes 全体を消費してしまうのを防ぐ、既定30)",
    )
    parser.add_argument(
        "--submissions-per-team", type=int, default=5,
        help="各チームにつき何個の提出(新しい順)からエピソードを探すか",
    )
    parser.add_argument("--out-dir", default=str(_HERE / "replays"))
    parser.add_argument(
        "--master-index-path",
        default=str(_HERE / "index" / "episodes_master.jsonl"),
    )
    parser.add_argument(
        "--sleep", type=float, default=0.3,
        help="API呼び出し間隔(秒)。fetch_top_episodes.py / fetch_deep_decks.py と同じ既定値",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="リプレイダウンロードの並列数(既定1=逐次、従来どおり)",
    )
    parser.add_argument(
        "--min-interval", type=float, default=None,
        help="kaggle CLI 呼び出しの最小間隔(秒、全スレッド共有)。並列時の429対策。"
        "既定は workers>1 のとき0.7、逐次のとき0",
    )
    args = parser.parse_args()

    min_interval = args.min_interval if args.min_interval is not None else (0.7 if args.workers > 1 else 0.0)
    set_min_interval(min_interval)

    deck_labels_path = Path(args.deck_labels)
    if not deck_labels_path.exists():
        print(
            f"エラー: {deck_labels_path} が見つかりません。"
            "先に deck_predictor/extract_decks.py と label_decks.py を実行してください",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_index_path = Path(args.master_index_path)
    master_index_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-minority"
    fetched_at = datetime.now(timezone.utc).isoformat()

    # --- [1/5] 少数クラスの判定 ---
    print(f"[1/5] {deck_labels_path} からクラス別デッキ数を集計")
    label_rows = load_deck_labels(deck_labels_path)
    counts: Counter[str] = Counter(row["archetype"] for row in label_rows)
    minority_archetypes = sorted(
        (dt for dt, n in counts.items() if dt != "other" and n < args.min_decks),
        key=lambda dt: counts[dt],
    )
    print(f"  総ラベル数: {len(label_rows)}")
    print(f"  しきい値 N < {args.min_decks} の少数クラス({len(minority_archetypes)}件、N昇順):")
    for dt in minority_archetypes:
        print(f"    {dt:24s} N={counts[dt]}")
    if not minority_archetypes:
        print("  少数クラスなし。追加取得は不要です。終了します。")
        return

    # --- [2/5] 少数クラスの (episode_id, player_index) -> team_id/team_name の逆引き ---
    print("[2/5] episodes_master.jsonl から team_id/team_name を逆引き")
    master_index = load_master_index(master_index_path)

    target_teams: dict[int, dict] = {}
    n_unresolved = 0
    unresolved_examples: list[str] = []
    for dt in minority_archetypes:  # レアなクラスから優先的に処理する順序を維持
        hits = [row for row in label_rows if row["archetype"] == dt]
        for row in hits:
            episode_id = row["episode_id"]
            player_index = row["player_index"]
            master_row = master_index.get(episode_id)
            if master_row is None:
                continue
            players = {p["player_index"]: p for p in master_row.get("players", [])}
            player = players.get(player_index)
            if player is None:
                continue
            team_id = player.get("team_id")
            team_name = player.get("team_name")
            if team_id is None:
                n_unresolved += 1
                if len(unresolved_examples) < 10:
                    unresolved_examples.append(f"{dt}: episode {episode_id} player {player_index} (team_name={team_name})")
                continue
            if team_id not in target_teams:
                target_teams[team_id] = {"team_name": team_name, "archetypes": set(), "hit_episode_ids": set()}
            target_teams[team_id]["archetypes"].add(dt)
            target_teams[team_id]["hit_episode_ids"].add(episode_id)

    print(f"  対象チーム数(team_id解決済み): {len(target_teams)}")
    print(f"  team_id 不明のため対象化できなかったヒット数: {n_unresolved}")
    for ex in unresolved_examples:
        print(f"    - {ex}")

    if not target_teams:
        print("  team_id を解決できた少数クラスのデッキがありません。追加取得できず終了します。")
        return

    # --- [3/5] チームごとにエピソードを探して未取得分をダウンロード ---
    print(
        f"[3/5] チームごとに未取得エピソードを探索・ダウンロード"
        f"(全体上限{args.max_episodes}件、チームあたり上限{args.max_episodes_per_team}件)"
    )
    already_indexed = load_existing_episode_ids(master_index_path)

    total_downloaded = 0
    total_new_master_rows = 0
    n_teams_processed = 0
    n_teams_skipped_budget = 0
    n_teams_failed = 0
    per_archetype_downloaded: Counter[str] = Counter()

    team_items = list(target_teams.items())
    for i, (team_id, info) in enumerate(team_items, 1):
        if total_downloaded >= args.max_episodes:
            n_teams_skipped_budget += len(team_items) - i + 1
            print(f"  全体上限({args.max_episodes}件)に達したため、残り{len(team_items) - i + 1}チームをスキップします")
            break

        team_name = info["team_name"]
        archetypes = sorted(info["archetypes"])
        prefix = f"  ({i}/{len(team_items)}) team={team_name} (teamId={team_id}) archetypes={archetypes}"

        try:
            submission_ids = fetch_team_submission_ids(team_id, args.submissions_per_team)
            time.sleep(args.sleep)
        except subprocess.CalledProcessError as e:
            n_teams_failed += 1
            print(f"{prefix}: 提出取得に失敗: {e.stderr}", file=sys.stderr)
            continue

        episode_meta: dict[str, dict] = {}
        for sub_id in submission_ids:
            try:
                episodes = fetch_episodes(sub_id)
            except subprocess.CalledProcessError as e:
                print(f"{prefix}: submission {sub_id} のエピソード取得に失敗: {e.stderr}", file=sys.stderr)
                continue
            for ep in episodes:
                episode_meta[str(ep["id"])] = ep
            time.sleep(args.sleep)

        # 未取得分だけを対象に、最新(episode_idが大きい)ものから優先してダウンロードする。
        new_episode_ids = sorted(
            (eid for eid in episode_meta if eid not in already_indexed),
            key=int,
            reverse=True,
        )
        remaining_overall = args.max_episodes - total_downloaded
        target_episode_ids = new_episode_ids[: min(args.max_episodes_per_team, remaining_overall)]

        if not target_episode_ids:
            print(f"{prefix}: 新規エピソードなし(既知{len(episode_meta) - len(new_episode_ids)}件、全て取得済み)")
            n_teams_processed += 1
            continue

        pending = [
            eid for eid in target_episode_ids if not (out_dir / f"episode-{eid}-replay.json").exists()
        ]
        # 既に手元にある分は「確認済み」として数える(target は全体上限で切ってあるので、
        # pending を全部落としても max_episodes を超えない)。
        downloaded_this_team = len(target_episode_ids) - len(pending)
        if args.workers > 1 and pending:
            ok, _failed = download_replays_parallel(pending, out_dir, args.workers)
            downloaded_this_team += len(ok)
            total_downloaded += len(ok)
        else:
            for eid in pending:
                try:
                    download_replay(int(eid), out_dir)
                    downloaded_this_team += 1
                    total_downloaded += 1
                except subprocess.CalledProcessError as e:
                    print(f"{prefix}: episode {eid} のリプレイ取得に失敗: {e.stderr}", file=sys.stderr)
                time.sleep(args.sleep)

        # このチーム自身の team_id は既に判明しているので、新規マスター行にも引き継ぐ
        # (相手側の team_id はこのrunではリーダーボードを引いていないため null のままになる。
        # 既存スクリプト群と同じ「解決できた範囲だけ埋める」方針)。
        name_to_context = {team_name: {"team_id": team_id, "rank": None, "score": None}}
        new_rows = build_master_rows(
            run_id=run_id,
            fetched_at=fetched_at,
            competition=args.competition,
            replays_dir=out_dir,
            episode_meta=episode_meta,
            name_to_context=name_to_context,
            already_indexed=already_indexed,
            episode_ids=target_episode_ids,
        )
        if new_rows:
            append_master_rows(master_index_path, new_rows)
            for r in new_rows:
                already_indexed.add(r["episode_id"])
            total_new_master_rows += len(new_rows)
            for dt in archetypes:
                per_archetype_downloaded[dt] += len(new_rows)

        n_teams_processed += 1
        print(
            f"{prefix}: 新規{len(target_episode_ids)}件対象"
            f"(ダウンロード{downloaded_this_team}件、マスターに{len(new_rows)}件追記)"
        )

    print("[4/5] 完了サマリ")
    print(f"  対象チーム数: {len(target_teams)}")
    print(f"  処理したチーム: {n_teams_processed}")
    print(f"  予算切れでスキップしたチーム: {n_teams_skipped_budget}")
    print(f"  失敗したチーム: {n_teams_failed}")
    print(f"  ダウンロードした新規リプレイ: {total_downloaded}")
    print(f"  マスターインデックスに追記した行数: {total_new_master_rows}")
    print("  少数クラス別の新規マスター行数(そのチームが関わった少数クラスへの寄与、重複カウントあり):")
    for dt in minority_archetypes:
        print(f"    {dt:24s} +{per_archetype_downloaded.get(dt, 0)}")

    print(
        "[5/5] 完了。deck_predictor/extract_decks.py と label_decks.py を再実行して、"
        "実際に少数クラスのデッキ数(N)が増えたか確認してください"
        "(増えなければ --min-decks や --max-episodes-per-team を調整して対象チームを広げ、再実行してください)"
    )


if __name__ == "__main__":
    main()
