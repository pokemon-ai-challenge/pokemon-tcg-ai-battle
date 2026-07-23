#!/usr/bin/env python3
"""非公開情報推定レイヤー（hidden_information）のキャリブレーション検証。

sample_submission/docs/plans/hidden-information/implementation-plan.md の Phase 3 をそのまま実装した
評価スクリプト。各リプレイ・各意思決定時点で ``HybridDeckPredictor.predict()`` の出力を
``OpponentHiddenState.marginals()`` に通し、「相手の手札/山札にこの card_id が入っている確率」の
予測が実際どれだけ当たっているか（reliability・ECE）を測る。

## データソースと ground truth の作り方（design.md §10 / implementation-plan.md 3.1）

Kaggle リプレイ（``kaggle_replays/replays/*.json``）は両プレイヤー分の視点を含む神視点データ。
「手札」はカードの持ち主自身の視点では常に真値が入っている（``PlayerState.hand`` は自分視点では
``list[Card]``、非公開なのは相手から見た場合だけ）ので、同じ replay 内で相手プレイヤー自身の
アクティブステップを遡って読むだけで100%正解の手札を復元できる（追加シミュレーション不要）。

「山札」の ground truth は、プラン記載の定義をそのまま使う:
    山札の中身 = 全60枚(deck_db.jsonl) − 手札(上記) − 公開ゾーン(バトル場/ベンチ/トラッシュ/
                 スタジアム/一時公開/既に取得済みのサイド)
この定義は「まだ取得されていないサイドカード」を厳密には山札と区別できない（プラン 3.1 が明記する
既知の制約: 未取得サイドの中身は replay からは復元できない）。取得済みサイドは公開ゾーンの一部として
差し引かれるため、この定義による「山札」には微量の未取得サイドの混入が残りうる（deckCount 件に対し
未取得サイドは最大6件程度なので、大きな系統誤差にはならない想定）。取得済みサイドの的中率および
未取得サイドを含めた完全なキャリブレーションは、別スクリプト ``battle_review_viewer/hidden_info_diff.py``
（cg.game.visualize_data() の神視点を使うローカル対戦）で追加検証する（プラン 3.1 の2段構成）。

## ナイーブベースライン

``OpponentHiddenState`` はアーキタイプ事後分布で周辺化するが、この事後分布を
``HybridDeckPredictor.predict()`` の実際の推論結果ではなく「全アーキタイプ等重み」の一様分布に
差し替えた場合の marginals() も同じ decision point で計算し、reliability/ECE を並べて出す
（完了条件: 本実装が一様ベースラインより悪化していないことを確認する）。

## reliability/ECE の計算方法

evaluate.py が top-1 分類の確信度キャリブレーションに使っている ``PredictionResult`` /
``compute_ece`` / ``confidence_bin`` / ``_CONFIDENCE_BINS`` をそのまま再利用する（車輪の再発明を
避ける）。本スクリプトの「サンプル」は (予測確率, 実際にそのゾーンにcard_idが有ったか) のペアなので、
``true_label`` に "hit"/"miss"、``pred_label`` に常に "hit"、``top1_confidence`` に予測確率を
詰めた ``PredictionResult`` を作ることで、そのまま ``compute_ece`` のバケツ集計ロジックに乗せられる
（バケツ内の平均確信度 vs 実際の的中率、というreliabilityの定義がそのまま一致するため）。

使い方:
  python evaluate_hidden_information.py --replay-limit 100
  python evaluate_hidden_information.py --replay-limit 0   # 全件（0 は無制限の意味で使う）
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
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

from cg.api import Observation, Pokemon, State, to_observation_class  # noqa: E402
from ptcg_ai.hidden_information.opponent_hidden_state import OpponentHiddenState  # noqa: E402
from ptcg_ai.opponent_modeling.hybrid_predictor import HybridDeckPredictor  # noqa: E402
from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge  # noqa: E402

from evaluate import _CONFIDENCE_BINS, PredictionResult, compute_ece, confidence_bin  # noqa: E402

_DEFAULT_POOL_PATH = _SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "hidden_information" / "archetype_card_pool.json"


# ----------------------------------------------------------------------
# 入力読み込み


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_deck_db(path: Path) -> dict[tuple[str, int], Counter[int]]:
    """(episode_id, player_index) -> 実際の60枚デッキ(card_id の Counter)。"""
    result: dict[tuple[str, int], Counter[int]] = {}
    for row in load_jsonl(path):
        result[(row["episode_id"], row["player_index"])] = Counter(int(c) for c in row["deck_card_ids"])
    return result


def load_naive_posterior(pool_path: Path) -> dict[str, float]:
    """archetype_card_pool.json のキー一覧から「全アーキタイプ等重み」の一様事後分布を作る。

    OpponentHiddenState の内部状態(``_archetype_pool``)は private 属性なので触らず、同じ
    JSON ファイルを本スクリプト側で独立に読み直す(Phase 1/2 モジュール自体は変更しない制約)。
    """
    if not pool_path.exists():
        return {}
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    archetypes = list(payload.get("archetypes", {}).keys())
    if not archetypes:
        return {}
    weight = 1.0 / len(archetypes)
    return {a: weight for a in archetypes}


# ----------------------------------------------------------------------
# リプレイ1本分の下ごしらえ: 各プレイヤー自身の意思決定ステップ列 + 自分の手札スナップショット履歴


def _add_pokemon(counts: Counter[int], pokemon: Pokemon) -> None:
    counts[pokemon.id] += 1
    for card in pokemon.energyCards:
        counts[card.id] += 1
    for card in pokemon.tools:
        counts[card.id] += 1
    for card in pokemon.preEvolution:
        counts[card.id] += 1


def visible_opponent_cards(state: State, opponent_index: int) -> Counter[int]:
    """viewer 自身の obs.current から、相手(opponent_index)の公開ゾーンに見えている card_id を集計する。

    opponent_knowledge_diff.collect_ground_truth() の非公開情報レイヤー向け簡易版(serial単位ではなく
    card_id 単位の枚数カウント。ここでは「何枚見えているか」だけで十分)。バトル場/ベンチ(本体+付属)・
    トラッシュ・スタジアム・一時公開・**既に取得済みのサイド**(Card が非 None のもの)を対象にする。
    """
    counts: Counter[int] = Counter()
    player = state.players[opponent_index]
    for pokemon in player.active:
        if pokemon is not None:
            _add_pokemon(counts, pokemon)
    for pokemon in player.bench:
        _add_pokemon(counts, pokemon)
    for card in player.discard:
        counts[card.id] += 1
    for card in state.stadium:
        if card.playerIndex == opponent_index:
            counts[card.id] += 1
    if state.looking:
        for card in state.looking:
            if card is not None and card.playerIndex == opponent_index:
                counts[card.id] += 1
    for card in player.prize:
        if card is not None:
            counts[card.id] += 1
    return counts


class ReplayRecords:
    """1リプレイ分の下ごしらえ結果。"""

    def __init__(self) -> None:
        # player_index -> [(step_index, Observation), ...]（そのプレイヤー自身の意思決定ステップ）
        self.records: dict[int, list[tuple[int, Observation]]] = {0: [], 1: []}
        # player_index -> ([step_index, ...], [Counter(手札card_id), ...])（bisect用に分離）
        self.hand_steps: dict[int, list[int]] = {0: [], 1: []}
        self.hand_snapshots: dict[int, list[Counter[int]]] = {0: [], 1: []}

    def add(self, player_index: int, step_index: int, obs: Observation) -> None:
        self.records[player_index].append((step_index, obs))
        own_hand = obs.current.players[player_index].hand
        if own_hand is not None:
            self.hand_steps[player_index].append(step_index)
            self.hand_snapshots[player_index].append(Counter(c.id for c in own_hand))

    def hand_ground_truth(self, player_index: int, step_index: int) -> Counter[int]:
        """player_index 自身の、step_index 時点(以前で直近)の真の手札。

        手札は持ち主自身のターンでしか変化しないため、直近の「自分がACTIVEだった時点」の
        スナップショットがそのまま step_index 時点の真値になる(design.md/plan 3.1 の前提)。
        """
        steps = self.hand_steps[player_index]
        idx = bisect.bisect_right(steps, step_index) - 1
        if idx < 0:
            return Counter()
        return self.hand_snapshots[player_index][idx]


def load_replay_records(replay: dict) -> ReplayRecords:
    steps = replay["steps"]
    out = ReplayRecords()
    for i, step in enumerate(steps):
        for p in (0, 1):
            agent_step = step[p]
            if agent_step["status"] != "ACTIVE":
                continue
            obs_dict = agent_step["observation"]
            if obs_dict.get("select") is None:
                continue  # デッキ選択ステップ(current も None)
            obs = to_observation_class(obs_dict)
            if obs.current is None:
                continue
            out.add(p, i, obs)
    return out


# ----------------------------------------------------------------------
# 1リプレイの評価


class Samples:
    """reliability/ECE 集計用の (予測確率, 実際にそのゾーンにあったか) ペアを溜めるバケツ。"""

    def __init__(self) -> None:
        self.hand_real: list[tuple[float, bool]] = []
        self.deck_real: list[tuple[float, bool]] = []
        self.hand_naive: list[tuple[float, bool]] = []
        self.deck_naive: list[tuple[float, bool]] = []
        self.n_decision_points = 0
        self.n_deck_count_mismatch = 0

    def extend_from(self, other: "Samples") -> None:
        self.hand_real.extend(other.hand_real)
        self.deck_real.extend(other.deck_real)
        self.hand_naive.extend(other.hand_naive)
        self.deck_naive.extend(other.deck_naive)
        self.n_decision_points += other.n_decision_points
        self.n_deck_count_mismatch += other.n_deck_count_mismatch


def evaluate_replay(
    episode_id: str,
    replay: dict,
    deck_db: dict[tuple[str, int], Counter[int]],
    predictor: HybridDeckPredictor,
    naive_posterior: dict[str, float],
    opp_state: OpponentHiddenState,
) -> Samples:
    records = load_replay_records(replay)
    samples = Samples()

    for viewer in (0, 1):
        opponent = 1 - viewer
        full_deck = deck_db.get((episode_id, opponent))
        if full_deck is None:
            continue  # ラベル/デッキ抽出が無い相手(label_decks.py 対象外)はスキップ

        knowledge = OpponentKnowledge(opponent_index=opponent)

        for step_index, obs in records.records[viewer]:
            knowledge.update_from_logs(obs.logs)
            knowledge.update_from_state(obs.current)
            features = knowledge.get_prediction_features()
            turn = obs.current.turn
            opponent_player_state = obs.current.players[opponent]

            # --- ground truth ---
            hand_gt = records.hand_ground_truth(opponent, step_index)
            visible_gt = visible_opponent_cards(obs.current, opponent)
            deck_gt = Counter(full_deck)
            deck_gt.subtract(hand_gt)
            deck_gt.subtract(visible_gt)
            deck_gt = +deck_gt  # 単項+で負値を掃除(design.md/plan 3.1 の定義どおり)

            samples.n_decision_points += 1
            expected_hidden = opponent_player_state.deckCount + sum(
                1 for c in opponent_player_state.prize if c is None
            )
            # 山末 ground truth の会計整合性チェック。cg エンジンの「カードが手札を離れたがまだ
            # トラッシュ等に着地していない一瞬」(own_hidden_state.py の docstring が自分側について
            # 既に文書化している既知の遷移的不整合と同種の現象)により、稀に deck_gt の合計が
            # deckCount+未取得サイド数と食い違う。ここでは補正を試みず、食い違った時点は
            # 山末サンプルの追加をスキップする(正しいと確信できる時点だけで reliability を測るため。
            # 手札 ground truth は player.hand を直接読むだけなのでこの不整合の影響を受けない)。
            deck_accounting_ok = sum(deck_gt.values()) == expected_hidden
            if not deck_accounting_ok:
                samples.n_deck_count_mismatch += 1

            # --- 実装(HybridDeckPredictor の実posterior) ---
            archetype_posterior = predictor.predict(features["observed_cards"], turn)
            opp_state.update(archetype_posterior, features["observed_card_ids"], opponent_player_state)
            real_marginals = opp_state.marginals()
            for card_id, probs in real_marginals.items():
                if card_id is None:
                    continue
                samples.hand_real.append((probs["hand"], hand_gt.get(card_id, 0) > 0))
                if deck_accounting_ok:
                    samples.deck_real.append((probs["deck"], deck_gt.get(card_id, 0) > 0))

            # --- ナイーブベースライン(全アーキタイプ等重み) ---
            if naive_posterior:
                opp_state.update(naive_posterior, features["observed_card_ids"], opponent_player_state)
                naive_marginals = opp_state.marginals()
                for card_id, probs in naive_marginals.items():
                    if card_id is None:
                        continue
                    samples.hand_naive.append((probs["hand"], hand_gt.get(card_id, 0) > 0))
                    if deck_accounting_ok:
                        samples.deck_naive.append((probs["deck"], deck_gt.get(card_id, 0) > 0))

    return samples


# ----------------------------------------------------------------------
# reliability テーブル(evaluate.py の PredictionResult / compute_ece / confidence_bin を再利用)


def to_prediction_results(pairs: list[tuple[float, bool]]) -> list[PredictionResult]:
    results = []
    for prob, actual in pairs:
        true_label = "hit" if actual else "miss"
        p_true = prob if actual else (1.0 - prob)
        results.append(
            PredictionResult(
                true_label=true_label,
                pred_label="hit",
                log_loss=-math.log(max(p_true, 1e-12)),
                evidence_count=0,
                top1_confidence=prob,
                top3=[],
            )
        )
    return results


def reliability_section(lines: list[str], title: str, pairs: list[tuple[float, bool]]) -> float:
    """reliability テーブルを lines に追記し、ECE を返す。"""
    lines.append(f"### {title}")
    lines.append("")
    if not pairs:
        lines.append("(サンプルなし)")
        lines.append("")
        return float("nan")

    results = to_prediction_results(pairs)
    ece = compute_ece(results)
    n = len(results)
    positive_rate = sum(1 for _, actual in pairs if actual) / n

    lines.append(f"- サンプル数: {n}")
    lines.append(f"- 実際の陽性率(そのゾーンに実在した割合): {positive_rate * 100:.2f}%")
    lines.append(f"- ECE(expected calibration error): {ece:.4f}")
    lines.append("")
    lines.append("| 予測確率ビン | サンプル数 | 平均予測確率 | 実際の的中率 |")
    lines.append("|---|---:|---:|---:|")
    groups: dict[str, list[PredictionResult]] = {label: [] for _, _, label in _CONFIDENCE_BINS}
    for r in results:
        cbin = confidence_bin(r.top1_confidence)
        if cbin is not None:
            groups[cbin].append(r)
    for _, _, label in _CONFIDENCE_BINS:
        group = groups[label]
        if not group:
            continue
        gn = len(group)
        avg_conf = sum(r.top1_confidence for r in group) / gn
        hit_rate = sum(1 for r in group if r.true_label == "hit") / gn
        lines.append(f"| {label} | {gn} | {avg_conf * 100:.2f}% | {hit_rate * 100:.2f}% |")
    lines.append("")
    print(f"  [{title}] n={n} 陽性率={positive_rate * 100:.2f}% ECE={ece:.4f}")
    return ece


# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays-dir", default=str(_REPO_ROOT / "kaggle_replays" / "replays"))
    parser.add_argument("--deck-db", default=str(_HERE / "output" / "deck_db.jsonl"))
    parser.add_argument("--pool-path", default=str(_DEFAULT_POOL_PATH))
    parser.add_argument("--report", default=str(_HERE / "output" / "hidden_info_eval_report.md"))
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=100,
        help="評価するリプレイ数の上限(0で無制限=全件)。デフォルト100(まず動作確認する用)。",
    )
    args = parser.parse_args()

    predictor = HybridDeckPredictor()
    if not predictor.is_ready:
        print("エラー: HybridDeckPredictor が未ロード状態です(重みJSONを確認してください)", file=sys.stderr)
        sys.exit(1)
    print(f"predictor.mode = {predictor.mode}")

    pool_path = Path(args.pool_path)
    opp_state = OpponentHiddenState(pool_path=pool_path)
    if not opp_state.is_ready:
        print(f"エラー: archetype_card_pool.json を読み込めませんでした: {pool_path}", file=sys.stderr)
        sys.exit(1)

    naive_posterior = load_naive_posterior(pool_path)
    print(f"ナイーブベースライン(全アーキタイプ等重み)のアーキタイプ数: {len(naive_posterior)}")

    deck_db = load_deck_db(Path(args.deck_db))
    print(f"deck_db エントリ数: {len(deck_db)}")

    replay_paths = sorted(Path(args.replays_dir).glob("episode-*-replay.json"))
    if args.replay_limit > 0:
        replay_paths = replay_paths[: args.replay_limit]
    print(f"評価対象リプレイ数: {len(replay_paths)}")

    total = Samples()
    n_ok = 0
    n_errors = 0
    errors: list[str] = []
    t0 = time.time()

    for i, replay_path in enumerate(replay_paths, start=1):
        episode_id = replay_path.stem.split("-")[1]
        try:
            with replay_path.open(encoding="utf-8") as rf:
                replay = json.load(rf)
            samples = evaluate_replay(episode_id, replay, deck_db, predictor, naive_posterior, opp_state)
            total.extend_from(samples)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - 1件のエラーで全体を止めない
            n_errors += 1
            errors.append(f"{replay_path.name}: {exc!r}")

        if i % 20 == 0 or i == len(replay_paths):
            elapsed = time.time() - t0
            print(
                f"  {i}/{len(replay_paths)} replays processed ({elapsed:.1f}s経過, "
                f"意思決定時点={total.n_decision_points}, hand_realサンプル={len(total.hand_real)})...",
                file=sys.stderr,
            )

    elapsed_total = time.time() - t0
    print(f"\n{n_ok}/{len(replay_paths)}件のリプレイを処理しました({elapsed_total:.1f}秒)")
    if n_errors:
        print(f"エラー: {n_errors}件のリプレイでエラーが発生しました")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ...他 {len(errors) - 20} 件")

    mismatch_rate = (
        total.n_deck_count_mismatch / total.n_decision_points if total.n_decision_points else float("nan")
    )
    print(
        f"意思決定時点数={total.n_decision_points}  "
        f"山札ground truth件数不整合率={mismatch_rate * 100:.2f}%"
    )

    # --- レポート ---
    lines = ["# 非公開情報推定レイヤー(hidden_information) キャリブレーション検証レポート", ""]
    lines.append(f"- 評価対象リプレイ数: {len(replay_paths)}件中 {n_ok}件処理成功({n_errors}件エラー)")
    lines.append(f"- 実行時間: {elapsed_total:.1f}秒")
    lines.append(f"- predictor.mode: {predictor.mode}")
    lines.append(f"- 意思決定時点数(両プレイヤー×両視点の合計): {total.n_decision_points}")
    lines.append(
        f"- 山札ground truth件数不整合(deckCount+未取得サイド数 と 計算結果が食い違った時点)の割合: "
        f"{mismatch_rate * 100:.2f}%"
    )
    lines.append("")
    lines.append(
        "## 前提: 山札 ground truth の定義について\n\n"
        "本レポートの「山末」ground truth は `全60枚 - 手札(真値) - 公開ゾーン(取得済みサイド含む)` "
        "で計算しており、**未取得のサイドカードの中身がわずかに混入する**(design.md/plan 3.1 の既知の"
        "制約。取得済みサイドの的中率と、未取得サイドを含めた完全なキャリブレーションは "
        "`battle_review_viewer/hidden_info_diff.py` の神視点ローカル対戦で別途検証する)。\n"
    )

    lines.append("## 手札(hand) reliability")
    lines.append("")
    ece_hand_real = reliability_section(lines, "実装(HybridDeckPredictor の事後分布)", total.hand_real)
    ece_hand_naive = reliability_section(lines, "ナイーブベースライン(全アーキタイプ等重み)", total.hand_naive)

    lines.append("## 山末(deck) reliability")
    lines.append("")
    ece_deck_real = reliability_section(lines, "実装(HybridDeckPredictor の事後分布)", total.deck_real)
    ece_deck_naive = reliability_section(lines, "ナイーブベースライン(全アーキタイプ等重み)", total.deck_naive)

    lines.append("## 完了条件の判定")
    lines.append("")
    lines.append("| ゾーン | 実装ECE | ナイーブECE | 判定 |")
    lines.append("|---|---:|---:|---|")

    def verdict(real: float, naive: float) -> str:
        if math.isnan(real) or math.isnan(naive):
            return "判定不能(サンプル不足)"
        return "OK(悪化なし)" if real <= naive else "NG(悪化あり)"

    hand_verdict = verdict(ece_hand_real, ece_hand_naive)
    deck_verdict = verdict(ece_deck_real, ece_deck_naive)
    lines.append(f"| 手札 | {ece_hand_real:.4f} | {ece_hand_naive:.4f} | {hand_verdict} |")
    lines.append(f"| 山末 | {ece_deck_real:.4f} | {ece_deck_naive:.4f} | {deck_verdict} |")
    lines.append("")

    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nレポートを {args.report} に書き出しました")
    print(f"完了条件判定: 手札={hand_verdict}  山末={deck_verdict}")


if __name__ == "__main__":
    main()
