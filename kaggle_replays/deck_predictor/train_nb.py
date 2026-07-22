#!/usr/bin/env python3
"""生成的ベイズ(Naive Bayes)版 相手デッキ予測器の学習スクリプト。

方針の詳細は
sample_submission/docs/plans/opponent-deck-predictor/early-confidence-improvement-plan.md
の「フェーズB」節(決定事項テーブル)を参照。LR(train.py + adjust_prior.py)と違い、こちらは
「P(カード採用 | アーキタイプ)」を直接、ラベル付きデッキから頻度推定する:

  log P(c | 観測) = log prior(c) + Σ_{観測カード n, 枚数 k} log P(deck が n を k 枚以上採用 | c)

- 汎用カード(全クラスでほぼ同じ採用率)→ 尤度がほぼ同じ → posterior は prior に留まる。
- 専用カード(そのクラスにしかほぼ入らない)→ 他クラスの尤度がほぼ0 → 1枚で確信してよい。

## カード名の対応付け

deck_db.jsonl の `deck_card_ids`(カードID)を、`cg.api.all_card_data()` の `cardId -> name`
で EN カード名に変換する。これは `sample_submission/ptcg_ai/opponent_modeling/opponent_knowledge.py`
の `OpponentKnowledge` がランタイムで observed_cards を作るときと同じ経路であり(cg.api の
`all_card_data()` は EN 名を返す)、train/serve skew を構造的に防ぐ。build_dataset.py が
sample_submission を sys.path に追加して cg.api / ptcg_ai を import するのと同じやり方を踏襲する。
`extract_decks.py` が deck_db.jsonl に書き込む `deck_card_names`(JP_Card_Data.csv 由来、JP名)は
使わない。

## 尤度

各 (カード名 n, クラス c, k=1..4) について P(そのクラスのデッキが n を k 枚以上採用 | c) を、
ラベル済みデッキ N_c 件から頻度推定する。ラプラス平滑化 (count + alpha) / (N_c + 2*alpha)、
alpha=1.0(0確率を作らない)。同名カード(再録違いの別ID)は名前単位で枚数合算。
デッキ内に5枚以上あっても k=4 の集計(≥4枚か)に丸める(基本エネルギー等)。

## prior

adjust_prior.py と同じ「直近 --recent-days 日(既定14) x 相手ランク上位 --top-rank 位(既定200)」
ウィンドウ内のデッキラベル分布(episode_window.py の関数を再利用)。ラプラス平滑化で0を作らない。
ウィンドウ内0件なら警告を出し、全体(ラベル付き全デッキ)の分布にフォールバックする。

使い方:
  python train_nb.py
  python train_nb.py --recent-days 30 --top-rank 500 --deploy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"

# sample_submission/ を sys.path に追加してから cg.api を import する(cg.dll がロードされる)。
# build_dataset.py / adjust_prior.py と同じやり方。
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from cg.api import all_card_data  # noqa: E402

from episode_window import EpisodeIndex, in_recent_top_rank_window  # noqa: E402

_DEPLOY_TARGET = (
    _REPO_ROOT / "sample_submission" / "ptcg_ai" / "opponent_modeling" / "deck_predictor_nb.json"
)
_DEFAULT_ALPHA = 1.0
_DEFAULT_RECENT_DAYS = 14
_DEFAULT_TOP_RANK = 200
_MAX_K = 4


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_id_to_name() -> dict[int, str]:
    """cg.api.all_card_data() から cardId -> name(EN)の対応表を作る。

    OpponentKnowledge._id_to_name と同じ作り方(train/serve skew を防ぐため必ずここ経由にする)。
    """
    return {c.cardId: c.name for c in all_card_data()}


def build_deck_name_counts(
    deck_db_rows: list[dict], id_to_name: dict[int, str]
) -> dict[tuple[str, int], Counter]:
    """(episode_id, player_index) -> {カード名: 枚数} の対応表を作る(名前単位で合算)。"""
    result: dict[tuple[str, int], Counter] = {}
    n_unknown_ids: Counter[int] = Counter()
    for row in deck_db_rows:
        key = (row["episode_id"], row["player_index"])
        counts: Counter[str] = Counter()
        for card_id in row["deck_card_ids"]:
            name = id_to_name.get(card_id)
            if name is None:
                # all_card_data() の語彙に無い card_id(通常は起きないはずだが、念のため記録する)。
                n_unknown_ids[card_id] += 1
                name = str(card_id)
            counts[name] += 1
        result[key] = counts
    if n_unknown_ids:
        total_unknown = sum(n_unknown_ids.values())
        print(
            f"警告: all_card_data() に無い card_id が {len(n_unknown_ids)} 種類、"
            f"のべ {total_unknown} 枚見つかりました(str(card_id) を仮の名前として使用): "
            f"{dict(list(n_unknown_ids.items())[:10])}{'...' if len(n_unknown_ids) > 10 else ''}",
            file=sys.stderr,
        )
    return result


def compute_likelihoods(
    deck_labels: list[dict],
    deck_name_counts: dict[tuple[str, int], Counter],
    classes: list[str],
    alpha: float,
) -> tuple[dict[str, dict[str, list[float]]], list[str], dict[str, int], int]:
    """クラスごとの尤度テーブルを頻度推定する。

    戻り値: (likelihoods, card_names(ソート済み語彙), n_decks_per_class, n_missing_decks)
      likelihoods[card_name][class] = [p_ge1, p_ge2, p_ge3, p_ge4]
    """
    # クラスごとに集めた「ラベル付きデッキの名前別枚数カウント」のリスト。
    decks_by_class: dict[str, list[Counter]] = defaultdict(list)
    n_missing_decks = 0
    for row in deck_labels:
        key = (row["episode_id"], row["player_index"])
        archetype = row["archetype"]
        name_counts = deck_name_counts.get(key)
        if name_counts is None:
            # deck_db.jsonl に対応するデッキが無い(通常は起きないはずだが、念のためスキップ)。
            n_missing_decks += 1
            continue
        decks_by_class[archetype].append(name_counts)

    n_decks_per_class = {c: len(decks_by_class.get(c, [])) for c in classes}

    # 語彙 = ラベル付きデッキに現れた全カード名。
    card_names: set[str] = set()
    for decks in decks_by_class.values():
        for name_counts in decks:
            card_names.update(name_counts.keys())
    sorted_card_names = sorted(card_names)

    likelihoods: dict[str, dict[str, list[float]]] = {
        name: {} for name in sorted_card_names
    }
    for c in classes:
        decks = decks_by_class.get(c, [])
        n_c = len(decks)
        # 各カード名について、このクラスのデッキが k 枚以上採用している件数を k=1..4 で数える。
        # (先に card_name -> [count_ge_1..4] を1パスで集計してから確率化する。)
        counts_ge_k: dict[str, list[int]] = {name: [0, 0, 0, 0] for name in sorted_card_names}
        for name_counts in decks:
            for name, count in name_counts.items():
                capped = min(count, _MAX_K)
                bucket = counts_ge_k[name]
                for k in range(1, capped + 1):
                    bucket[k - 1] += 1
        for name in sorted_card_names:
            bucket = counts_ge_k[name]
            probs = [(cnt + alpha) / (n_c + 2 * alpha) for cnt in bucket]
            likelihoods[name][c] = probs

    return likelihoods, sorted_card_names, n_decks_per_class, n_missing_decks


def compute_overall_distribution(deck_labels: list[dict], classes: list[str], alpha: float) -> dict[str, float]:
    counts: Counter[str] = Counter(row["archetype"] for row in deck_labels if row["archetype"] in classes)
    n = sum(counts.values())
    denom = n + alpha * len(classes)
    if denom <= 0:
        return {c: 1.0 / len(classes) for c in classes}
    return {c: (counts.get(c, 0) + alpha) / denom for c in classes}


def compute_prior(
    deck_labels: list[dict],
    episode_index: EpisodeIndex,
    classes: list[str],
    now: datetime,
    recent_days: int,
    top_rank: int,
    alpha: float,
) -> tuple[dict[str, float], dict]:
    """adjust_prior.py と同じウィンドウ定義でクラス事前分布を求める。ウィンドウ0件なら全体分布に
    フォールバックする(警告あり)。"""
    counts: Counter[str] = Counter()
    unknown_classes: Counter[str] = Counter()

    for row in deck_labels:
        episode_id = row["episode_id"]
        player_index = row["player_index"]
        archetype = row["archetype"]

        episode_time = episode_index.time_of(episode_id)
        rank = episode_index.rank_of(episode_id, player_index)
        if not in_recent_top_rank_window(episode_time, rank, now, recent_days, top_rank):
            continue

        if archetype not in classes:
            unknown_classes[archetype] += 1
            continue
        counts[archetype] += 1

    n_window = sum(counts.values())
    used_fallback = False
    if n_window == 0:
        print(
            f"警告: prior ウィンドウ(直近{recent_days}日 x 上位{top_rank}位以内)に該当するデッキが"
            "0件でした。全体(ラベル付き全デッキ)の分布にフォールバックします。",
            file=sys.stderr,
        )
        used_fallback = True
        pi = compute_overall_distribution(deck_labels, classes, alpha)
    else:
        denom = n_window + alpha * len(classes)
        pi = {c: (counts.get(c, 0) + alpha) / denom for c in classes}

    meta = {
        "recent_days": recent_days,
        "top_rank": top_rank,
        "now": now.isoformat(),
        "n_window_decks": n_window,
        "used_fallback": used_fallback,
        "alpha": alpha,
    }
    if unknown_classes:
        print(
            f"警告: prior ウィンドウ内に classes に無いアーキタイプが {sum(unknown_classes.values())} 件"
            f"ありました(母数から除外): {dict(unknown_classes)}",
            file=sys.stderr,
        )
    return pi, meta


def parse_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck-db", default=str(_HERE / "output" / "deck_db.jsonl"))
    parser.add_argument("--deck-labels", default=str(_HERE / "output" / "deck_labels.jsonl"))
    parser.add_argument(
        "--master-index", default=str(_REPO_ROOT / "kaggle_replays" / "index" / "episodes_master.jsonl")
    )
    parser.add_argument("--out", default=str(_HERE / "output" / "model" / "deck_predictor_nb.json"))
    parser.add_argument("--alpha", type=float, default=_DEFAULT_ALPHA, help="ラプラススムージングの加算値")
    parser.add_argument("--recent-days", type=int, default=_DEFAULT_RECENT_DAYS)
    parser.add_argument("--top-rank", type=int, default=_DEFAULT_TOP_RANK)
    parser.add_argument(
        "--now", default=None, help="基準時刻をISO8601で上書き(再現性確認用)。省略時は実行時のUTC時刻"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="sample_submission/ptcg_ai/opponent_modeling/deck_predictor_nb.json へコピーする",
    )
    args = parser.parse_args()

    now = parse_now(args.now)

    print(f"デッキDB読み込み中: {args.deck_db}")
    deck_db_rows = load_jsonl(Path(args.deck_db))
    print(f"  {len(deck_db_rows)} 件のデッキ")

    print(f"デッキラベル読み込み中: {args.deck_labels}")
    deck_labels = load_jsonl(Path(args.deck_labels))
    print(f"  {len(deck_labels)} 件のラベル")

    classes = sorted({row["archetype"] for row in deck_labels})
    print(f"  クラス数: {len(classes)} -> {classes}")

    print("cg.api.all_card_data() から cardId -> name(EN) の対応表を作成中...")
    id_to_name = build_id_to_name()
    print(f"  {len(id_to_name)} 件のカード定義")

    deck_name_counts = build_deck_name_counts(deck_db_rows, id_to_name)

    print("尤度テーブルを推定中...")
    likelihoods, card_names, n_decks_per_class, n_missing_decks = compute_likelihoods(
        deck_labels, deck_name_counts, classes, args.alpha
    )
    print(f"  語彙(カード名)数: {len(card_names)}")
    print(f"  クラスごとのデッキ数: {n_decks_per_class}")
    if n_missing_decks:
        print(
            f"警告: deck_labels.jsonl の {n_missing_decks} 件が deck_db.jsonl に見つからず"
            "尤度推定から除外されました。",
            file=sys.stderr,
        )
    small_classes = {c: n for c, n in n_decks_per_class.items() if n < 30}
    if small_classes:
        print(
            f"警告: デッキ数が30件未満のクラスがあります(尤度推定が不安定な可能性): {small_classes}",
            file=sys.stderr,
        )

    print(f"\nprior 計算中(直近{args.recent_days}日 x 上位{args.top_rank}位以内)...")
    episode_index = EpisodeIndex.load(Path(args.master_index))
    priors_by_class, prior_meta = compute_prior(
        deck_labels, episode_index, classes, now, args.recent_days, args.top_rank, args.alpha
    )
    priors = [priors_by_class[c] for c in classes]
    print(f"  prior: {dict(zip(classes, (round(p, 4) for p in priors)))}")

    payload = {
        "classes": classes,
        "priors": priors,
        "card_names": card_names,
        "likelihoods": likelihoods,
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_decks_total": len(deck_labels),
            "n_decks_per_class": n_decks_per_class,
            "n_missing_decks": n_missing_decks,
            "alpha": args.alpha,
            "max_k": _MAX_K,
            "prior_window": prior_meta,
            "version": 1,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nNB重みを {out_path} に書き出しました")

    if args.deploy:
        _DEPLOY_TARGET.parent.mkdir(parents=True, exist_ok=True)
        _DEPLOY_TARGET.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"デプロイ用に {_DEPLOY_TARGET} にもコピーしました")


if __name__ == "__main__":
    main()
