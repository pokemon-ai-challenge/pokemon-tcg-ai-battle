#!/usr/bin/env python3
"""非公開情報推定レイヤー(hidden_information)のキャリブレーションを、相手デッキの「ティア」別に層別集計する。

``evaluate_hidden_information.py`` は全リプレイをまとめて1本のreliabilityにするが、本スクリプトは
**相手が使っているアーキタイプのティア**で意思決定時点を振り分け、ティアごとに reliability/ECE を出す。
狙いは「推定器は出現頻度の高い上位デッキに引っ張られて学習されているため、中位・下位(=学習データ内で
少数)の相手を推定するとどれだけ精度が落ちるか」を定量化すること。

## ティア定義(このリポジトリの根拠)

この branch には明示的なティア表(meta_decks.py 等)が存在せず、推定プール(archetype_card_pool.json)は
21アーキタイプ全部をカバーしている。そのため**ティアは Kaggle リプレイ実データ(deck_labels.jsonl)の
出現頻度で定義する**。出現頻度 = 実戦で当たる頻度 ≈ 推定器の学習データ内での相対量、なので
「上位で学習している」という前提とも整合する。頻度の区切りは ``ARCHETYPE_TIER`` に固定表として持つ
(境界値は下記コメント参照。データ再取得で分布が動いた場合はここを更新する)。

ground truth の作り方・reliability/ECE の計算・ナイーブベースラインの定義は
``evaluate_hidden_information.py`` と完全に同一(低レベルヘルパをそのまま import して使う)。差分は
「サンプルを相手ティア別のバケツに入れる」ことだけ。

使い方:
  python evaluate_hidden_information_by_tier.py --replay-limit 1500
  python evaluate_hidden_information_by_tier.py --replay-limit 0   # 全件
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.hidden_information.opponent_hidden_state import OpponentHiddenState  # noqa: E402
from ptcg_ai.opponent_modeling.hybrid_predictor import HybridDeckPredictor  # noqa: E402
from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge  # noqa: E402

# 低レベルヘルパは全て sibling スクリプトから import(ground truth の定義を1箇所に保つ)。
import evaluate_hidden_information as base  # noqa: E402

_DEFAULT_POOL_PATH = _SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "hidden_information" / "archetype_card_pool.json"


# ----------------------------------------------------------------------
# ティア定義(deck_labels.jsonl の出現頻度に基づく。パーセントは全9396ラベル中の割合)

# 上位  : 各10%超(alakazam 29.9 / mega_lucario_ex 13.4 / archaludon_ex 11.5)
# 中の上: 5〜8%(crustle 7.8 / dragapult_ex 6.7 / marnie_grimmsnarl_ex 6.3 / mega_starmie_ex 5.2)
# 中の中: 1〜2%(shirona_garchomp_ex 1.9 / ogerpon_teal_ex 1.5 / mega_abomasnow_ex 1.5 /
#          kamitsuorochi_ex 1.5 / yadoking 1.3 / oliva_ex 1.1)
# 下位  : <1%(mega_froslass_ex / rocket_honchkrow / omatsuri_ondo / takeruraiko_ex /
#          n_zoroark_ex / toxtricity / gekkouga_ex)
# other : 分類不能ラベル(代表リスト無し。推定は「不明カード」扱いになるので別枠で見る)
ARCHETYPE_TIER: dict[str, str] = {
    "alakazam": "top",
    "mega_lucario_ex": "top",
    "archaludon_ex": "top",
    "crustle": "upper_mid",
    "dragapult_ex": "upper_mid",
    "marnie_grimmsnarl_ex": "upper_mid",
    "mega_starmie_ex": "upper_mid",
    "shirona_garchomp_ex": "mid",
    "ogerpon_teal_ex": "mid",
    "mega_abomasnow_ex": "mid",
    "kamitsuorochi_ex": "mid",
    "yadoking": "mid",
    "oliva_ex": "mid",
    "mega_froslass_ex": "low",
    "rocket_honchkrow": "low",
    "omatsuri_ondo": "low",
    "takeruraiko_ex": "low",
    "n_zoroark_ex": "low",
    "toxtricity": "low",
    "gekkouga_ex": "low",
    "other": "other",
}

TIER_ORDER = ["top", "upper_mid", "mid", "low", "other"]
TIER_LABEL_JA = {
    "top": "上位(Top)",
    "upper_mid": "中の上(Upper-mid)",
    "mid": "中の中(Mid)",
    "low": "下位(Low)",
    "other": "分類不能(Other)",
}


def load_archetype_map(path: Path) -> dict[tuple[str, int], str]:
    """(episode_id, player_index) -> archetype ラベル。"""
    result: dict[tuple[str, int], str] = {}
    for row in base.load_jsonl(path):
        result[(row["episode_id"], int(row["player_index"]))] = row["archetype"]
    return result


def _ece(pairs: list[tuple[float, bool]]) -> float:
    from evaluate_hidden_information import to_prediction_results  # noqa: PLC0415
    from evaluate import compute_ece  # noqa: PLC0415

    return compute_ece(to_prediction_results(pairs)) if pairs else float("nan")


def build_report_lines(
    n_replays: int,
    n_ok: int,
    n_errors: int,
    elapsed: float,
    predictor_mode: str,
    tier_samples: dict[str, base.Samples],
    tier_decks_seen: dict[str, set[tuple[str, int]]],
    partial: bool = False,
) -> list[str]:
    """現時点の tier_samples からレポート本文(markdown 行)を組み立てる。

    ``partial=True`` のときは「途中経過(チェックポイント)」であることを見出しに明記する。
    ループ内から定期的に呼べば、実行が途中で止まっても最新の集計が常にディスクに残る。
    """
    head = "(途中経過)" if partial else ""
    lines = [f"# 非公開情報推定レイヤー キャリブレーション ティア別検証レポート{head}", ""]
    lines.append(f"- 評価リプレイ数: {n_replays}件中 {n_ok}件成功({n_errors}件エラー)")
    lines.append(f"- 実行時間: {elapsed:.1f}秒" + ("  ※まだ実行中の途中集計" if partial else ""))
    lines.append(f"- predictor.mode: {predictor_mode}")
    lines.append("- ティア定義: deck_labels.jsonl の出現頻度(スクリプト冒頭 docstring 参照)")
    lines.append("- 軸: **相手が使っているアーキタイプのティア**で意思決定時点を層別")
    lines.append("")

    lines.append("## サマリ: ティア別 ECE(低いほど良い)")
    lines.append("")
    lines.append(
        "| ティア | 相手デッキ数 | 意思決定時点 | 手札ECE(実装/ナイーブ) | 山札ECE(実装/ナイーブ) |"
    )
    lines.append("|---|---:|---:|---|---|")
    for tier in TIER_ORDER:
        s = tier_samples[tier]
        if s.n_decision_points == 0:
            continue
        hr, hn = _ece(s.hand_real), _ece(s.hand_naive)
        dr, dn = _ece(s.deck_real), _ece(s.deck_naive)
        lines.append(
            f"| {TIER_LABEL_JA[tier]} | {len(tier_decks_seen[tier])} | {s.n_decision_points} | "
            f"{hr:.4f} / {hn:.4f} | {dr:.4f} / {dn:.4f} |"
        )
    lines.append("")

    for tier in TIER_ORDER:
        s = tier_samples[tier]
        if s.n_decision_points == 0:
            continue
        lines.append(f"## {TIER_LABEL_JA[tier]}")
        lines.append("")
        lines.append(
            f"- 相手デッキ数: {len(tier_decks_seen[tier])}  意思決定時点: {s.n_decision_points}  "
            f"山札GT不整合率: {s.n_deck_count_mismatch / s.n_decision_points * 100:.2f}%"
        )
        lines.append("")
        lines.append("### 手札(hand)")
        lines.append("")
        base.reliability_section(lines, "実装", s.hand_real)
        base.reliability_section(lines, "ナイーブ", s.hand_naive)
        lines.append("### 山札(deck)")
        lines.append("")
        base.reliability_section(lines, "実装", s.deck_real)
        base.reliability_section(lines, "ナイーブ", s.deck_naive)
    return lines


# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--replays-dir", default=str(_REPO_ROOT / "kaggle_replays" / "replays"))
    parser.add_argument("--deck-db", default=str(_HERE / "output" / "deck_db.jsonl"))
    parser.add_argument("--deck-labels", default=str(_HERE / "output" / "deck_labels.jsonl"))
    parser.add_argument("--pool-path", default=str(_DEFAULT_POOL_PATH))
    parser.add_argument("--report", default=str(_HERE / "output" / "hidden_info_eval_by_tier_report.md"))
    parser.add_argument("--replay-limit", type=int, default=1500, help="評価リプレイ数上限(0で全件)")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="このリプレイ数ごとに途中レポートを書き出す")
    args = parser.parse_args()

    predictor = HybridDeckPredictor()
    if not predictor.is_ready:
        print("エラー: HybridDeckPredictor 未ロード", file=sys.stderr)
        sys.exit(1)
    print(f"predictor.mode = {predictor.mode}")

    pool_path = Path(args.pool_path)
    opp_state = OpponentHiddenState(pool_path=pool_path)
    if not opp_state.is_ready:
        print(f"エラー: archetype_card_pool.json 未ロード: {pool_path}", file=sys.stderr)
        sys.exit(1)

    naive_posterior = base.load_naive_posterior(pool_path)
    deck_db = base.load_deck_db(Path(args.deck_db))
    archetype_map = load_archetype_map(Path(args.deck_labels))
    print(f"deck_db={len(deck_db)}  archetype_map={len(archetype_map)}")

    replay_paths = sorted(Path(args.replays_dir).glob("episode-*-replay.json"))
    if args.replay_limit > 0:
        replay_paths = replay_paths[: args.replay_limit]
    print(f"評価対象リプレイ数: {len(replay_paths)}")

    # tier -> Samples(base.Samples を流用)。相手アーキタイプ未知(deck_labels に無い)は集計から除外。
    tier_samples: dict[str, base.Samples] = {t: base.Samples() for t in TIER_ORDER}
    tier_decks_seen: dict[str, set[tuple[str, int]]] = {t: set() for t in TIER_ORDER}
    n_ok = n_errors = 0
    errors: list[str] = []
    t0 = time.time()

    for i, replay_path in enumerate(replay_paths, start=1):
        episode_id = replay_path.stem.split("-")[1]
        try:
            with replay_path.open(encoding="utf-8") as rf:
                replay = json.load(rf)
            records = base.load_replay_records(replay)

            for viewer in (0, 1):
                opponent = 1 - viewer
                full_deck = deck_db.get((episode_id, opponent))
                if full_deck is None:
                    continue
                archetype = archetype_map.get((episode_id, opponent))
                if archetype is None:
                    continue
                tier = ARCHETYPE_TIER.get(archetype, "other")
                samples = tier_samples[tier]
                tier_decks_seen[tier].add((episode_id, opponent))

                knowledge = OpponentKnowledge(opponent_index=opponent)
                for step_index, obs in records.records[viewer]:
                    knowledge.update_from_logs(obs.logs)
                    knowledge.update_from_state(obs.current)
                    features = knowledge.get_prediction_features()
                    turn = obs.current.turn
                    opponent_player_state = obs.current.players[opponent]

                    # --- ground truth(evaluate_hidden_information と同一定義)---
                    hand_gt = records.hand_ground_truth(opponent, step_index)
                    visible_gt = base.visible_opponent_cards(obs.current, opponent)
                    deck_gt = Counter(full_deck)
                    deck_gt.subtract(hand_gt)
                    deck_gt.subtract(visible_gt)
                    deck_gt = +deck_gt

                    samples.n_decision_points += 1
                    expected_hidden = opponent_player_state.deckCount + sum(
                        1 for c in opponent_player_state.prize if c is None
                    )
                    deck_accounting_ok = sum(deck_gt.values()) == expected_hidden
                    if not deck_accounting_ok:
                        samples.n_deck_count_mismatch += 1

                    # --- 実装(実posterior)---
                    archetype_posterior = predictor.predict(features["observed_cards"], turn)
                    opp_state.update(
                        archetype_posterior, features["observed_card_ids"], opponent_player_state
                    )
                    for card_id, probs in opp_state.marginals().items():
                        if card_id is None:
                            continue
                        samples.hand_real.append((probs["hand"], hand_gt.get(card_id, 0) > 0))
                        if deck_accounting_ok:
                            samples.deck_real.append((probs["deck"], deck_gt.get(card_id, 0) > 0))

                    # --- ナイーブベースライン ---
                    if naive_posterior:
                        opp_state.update(
                            naive_posterior, features["observed_card_ids"], opponent_player_state
                        )
                        for card_id, probs in opp_state.marginals().items():
                            if card_id is None:
                                continue
                            samples.hand_naive.append((probs["hand"], hand_gt.get(card_id, 0) > 0))
                            if deck_accounting_ok:
                                samples.deck_naive.append((probs["deck"], deck_gt.get(card_id, 0) > 0))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - 1件のエラーで全体を止めない
            n_errors += 1
            errors.append(f"{replay_path.name}: {exc!r}")

        if i % 20 == 0 or i == len(replay_paths):
            elapsed = time.time() - t0
            dp = sum(s.n_decision_points for s in tier_samples.values())
            print(f"  {i}/{len(replay_paths)} ({elapsed:.1f}s, 意思決定時点={dp})...", file=sys.stderr)

        # チェックポイント: 一定間隔で途中経過レポートをディスクに書き出す(実行が途中で
        # 止まっても最新の集計が残る。全件完走を待たずに部分結果を確認できる)。
        if i % args.checkpoint_every == 0 and i < len(replay_paths):
            lines = build_report_lines(
                len(replay_paths), n_ok, n_errors, time.time() - t0, predictor.mode,
                tier_samples, tier_decks_seen, partial=True,
            )
            Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  [checkpoint] {i}件時点のレポートを書き出し", file=sys.stderr)

    elapsed_total = time.time() - t0
    print(f"\n{n_ok}/{len(replay_paths)}件処理({elapsed_total:.1f}秒), エラー{n_errors}件")

    lines = build_report_lines(
        len(replay_paths), n_ok, n_errors, elapsed_total, predictor.mode,
        tier_samples, tier_decks_seen, partial=False,
    )
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nレポートを {args.report} に書き出しました")


if __name__ == "__main__":
    main()
