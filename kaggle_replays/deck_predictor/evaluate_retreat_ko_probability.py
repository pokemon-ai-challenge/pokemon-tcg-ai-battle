#!/usr/bin/env python3
"""board_evaluation.likely_ko_probability_next_turn（probabilistic_ko）のリプレイベースのバックテスト。

feature/bayesian-agent で追加した「相手が次ターンにこちらのポケモンをKOしてくる確率」の
ベイズ推定（``sample_submission/ptcg_ai/board_evaluation/board_features.py``）が実際どれだけ
当たっているかを、Kaggleリプレイ（``kaggle_replays/replays/*.json``）で検証する。
``evaluate_hidden_information.py``（hidden_information レイヤーのキャリブレーション検証）と
同じ設計方針を踏襲し、``evaluate.py`` の ``PredictionResult`` / ``compute_ece`` /
``confidence_bin`` をそのまま再利用する。

## ground truth（正解ラベル）の作り方

リプレイは両プレイヤー分の視点を含む神視点データ。各リプレイ・各視点(viewer)・その視点の
各ターンの最初の意思決定時点で、その時点の自分のアクティブポケモン（``Pokemon.serial`` で
一意識別）を対象に、次に自分が意思決定を求められるまでの間（＝実質的に相手の次ターン）で
そのポケモンが実際にきぜつしたかを、ログの
``LogType.MOVE_CARD``（``serial`` が一致し ``toArea == AreaType.DISCARD``）で判定する。
にげて場を離れただけ（ACTIVE→BENCH）なら DISCARD には移動しないため誤検知しない。

**既知の限界**: これは「実際の対局で起きたこと」を正解とするため、リプレイ内で当のプレイヤーが
（本スクリプトの対象とは別の判断で）実際に retreat していた場合、その「にげた」という事実込みの
結果が ground truth になる（＝「もし逃げていなかったら」という反実仮想ではない）。過去データを
使ったバックテスト一般に共通する制約であり、本スクリプトはこれを補正しない。

## 本番コードをそのまま使う

``hidden_information.match_context``（``OpponentHiddenState`` 等のモジュールレベルシングルトン）
を実際の対局と同じ手順（``match_context.update(obs)`` を各意思決定時点で順番に呼ぶ）で駆動し、
直後に ``board_features.likely_ko_probability_next_turn()`` を呼ぶ。内部ロジックを再実装せず
本番コードパスをそのまま検証するため、実装と評価のズレが生まれない。

視点(viewer)ごとに ``match_context.reset()`` してから、その視点の意思決定を古い順に処理する
（``match_context`` は「1試合を1つの視点から見た」状態しか保持できないシングルトンのため、
0視点・1視点を同時には処理できない）。

使い方:
  python evaluate_retreat_ko_probability.py --replay-limit 50
  python evaluate_retreat_ko_probability.py --replay-limit 0   # 全件
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))
sys.path.insert(0, str(_HERE))

# hidden_information.match_context は OwnHiddenState 経由で rule_based_agent.read_deck_csv() を
# 呼ぶが、これはカレントディレクトリからの相対パス "deck.csv" でファイルを開く実装になっている
# （Kaggle実行時のcwdがsample_submission/である前提）。このスクリプトを別ディレクトリ
# （kaggle_replays/deck_predictor/）から実行すると FileNotFoundError になり、
# match_context.update() 内の try/except に握りつぶされて「常にmarginals()が空」という
# サイレントな不具合を引き起こす。cwdを本番と合わせて明示的に固定する。
os.chdir(_SAMPLE_SUBMISSION_DIR)

from cg.api import AreaType, LogType, to_observation_class  # noqa: E402
from ptcg_ai.board_evaluation import board_features  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402

from evaluate import PredictionResult, compute_ece  # noqa: E402
from evaluate_hidden_information import ReplayRecords, load_replay_records  # noqa: E402

# 閾値スイープで並べる候補値。config既定の0.5を含める。
_THRESHOLD_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ----------------------------------------------------------------------
# ground truth: 対象serialのポケモンが指定ステップ区間内できぜつしたか


def scan_for_ko(steps: list, start_index: int, end_index: int | None, viewer: int, serial: int) -> bool:
    """steps[start_index+1 : end_index] の範囲で、serial のポケモンがDISCARDへ移動したかを見る。

    end_index が None なら試合終了まで走査する。
    """
    stop = end_index if end_index is not None else len(steps)
    for i in range(start_index + 1, stop):
        step = steps[i]
        for p in (0, 1):
            agent_step = step[p]
            if agent_step["status"] != "ACTIVE":
                continue
            obs_dict = agent_step["observation"]
            if obs_dict.get("select") is None:
                continue
            obs = to_observation_class(obs_dict)
            for log in obs.logs:
                if (
                    log.type == LogType.MOVE_CARD
                    and log.serial == serial
                    and log.toArea == AreaType.DISCARD
                    and log.playerIndex == viewer
                ):
                    return True
    return False


def scan_for_voluntary_retreat(steps: list, start_index: int, end_index: int | None, viewer: int, serial: int) -> bool:
    """steps[start_index+1 : end_index] の範囲で、serial のポケモンが自発的ににげた(SWITCH)かを見る。

    ``LogType.SWITCH`` の ``serialActive`` は「バトル場からベンチへ移動するポケモンのserial」
    (LogType定義コメント参照)。これが一致すれば、対象のポケモンが(きぜつではなく)にげて
    場を離れたことを意味する。これが起きたサンプルは「もし逃げていなかったら」の反実仮想が
    分からないため、"介入なし"サブセットの分析からは除外する対象としてマークする。
    """
    stop = end_index if end_index is not None else len(steps)
    for i in range(start_index + 1, stop):
        step = steps[i]
        for p in (0, 1):
            agent_step = step[p]
            if agent_step["status"] != "ACTIVE":
                continue
            obs_dict = agent_step["observation"]
            if obs_dict.get("select") is None:
                continue
            obs = to_observation_class(obs_dict)
            for log in obs.logs:
                if log.type == LogType.SWITCH and log.serialActive == serial and log.playerIndex == viewer:
                    return True
    return False


# ----------------------------------------------------------------------
# 1リプレイ・1視点ぶんの評価


class Samples:
    """(確率予測, 旧ブール判定, 実際にきぜつしたか, 自発的ににげたか) のタプルを溜めるバケツ。"""

    def __init__(self) -> None:
        self.rows: list[tuple[float, bool, bool, bool]] = []

    def add(self, prob: float, old_bool: bool, actual: bool, retreated_away: bool) -> None:
        self.rows.append((prob, old_bool, actual, retreated_away))

    def extend_from(self, other: "Samples") -> None:
        self.rows.extend(other.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def all_subset(self) -> list[tuple[float, bool, bool]]:
        """全サンプルの(確率, 旧ブール, 実際にきぜつしたか)のリスト(にげたかは無視)。"""
        return [(p, o, a) for p, o, a, _retreated in self.rows]

    def no_intervention_subset(self) -> list[tuple[float, bool, bool]]:
        """自発的ににげたサンプルを除いた(確率, 旧ブール, 実際にきぜつしたか)のリスト。

        にげたサンプルは「もし逃げていなかったら本当にきぜつしていたか」の反実仮想が
        分からないため除外する(スクリプト冒頭docstringの既知の限界への追加分析)。
        """
        return [(p, o, a) for p, o, a, retreated in self.rows if not retreated]

    def n_retreated_away(self) -> int:
        return sum(1 for *_rest, retreated in self.rows if retreated)


def evaluate_viewer(steps: list, records: ReplayRecords, viewer: int, samples: Samples) -> None:
    """viewer視点で match_context を駆動しつつ、ターンごとに1サンプル集める。"""
    match_context.reset()
    my_steps = records.records[viewer]

    pending: list[tuple[int, int, float, bool]] = []  # (step_index, serial, prob, old_bool)
    seen_turns: set[int] = set()

    for step_index, obs in my_steps:
        match_context.update(obs)  # 本番と同じ駆動(順序を欠かさず毎回呼ぶ)
        state = obs.current
        my_player = state.players[viewer]
        active = my_player.active[0] if my_player.active else None
        if active is None:
            continue
        turn = state.turn
        if turn in seen_turns:
            continue  # 同じターン内の重複サンプル(疑似反復)を避け、ターンにつき1件だけ取る
        seen_turns.add(turn)

        prob = board_features.likely_ko_probability_next_turn(active, state, viewer)
        old_bool = board_features.is_likely_ko_next_turn(active, state, viewer)
        pending.append((step_index, active.serial, prob, old_bool))

    for i, (step_index, serial, prob, old_bool) in enumerate(pending):
        end_index = pending[i + 1][0] if i + 1 < len(pending) else None
        actual = scan_for_ko(steps, step_index, end_index, viewer, serial)
        retreated_away = scan_for_voluntary_retreat(steps, step_index, end_index, viewer, serial)
        samples.add(prob, old_bool, actual, retreated_away)


# ----------------------------------------------------------------------
# レポート


def to_prediction_results(rows: list[tuple[float, bool, bool]]) -> list[PredictionResult]:
    results = []
    for prob, _old_bool, actual in rows:
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


# evaluate.py の _CONFIDENCE_BINS は「top-1分類confidence」向けで0.5未満を1つに束ねる設計のため、
# 0.5未満の分布(retreatの閾値検討でまさに重要な範囲)が粗くしか見えない。本スクリプト専用に、
# 0.0/1.0を特別扱いしつつ0.5未満も0.1刻みで細分化したビンを用意する。
_RETREAT_CONFIDENCE_BINS: list[tuple[float, float, str]] = [
    (0.0, 0.0 + 1e-12, "0.0(確実に安全)"),
    (0.0 + 1e-12, 0.1, "0.0<p<0.1"),
    (0.1, 0.2, "0.1-0.2"),
    (0.2, 0.3, "0.2-0.3"),
    (0.3, 0.4, "0.3-0.4"),
    (0.4, 0.5, "0.4-0.5"),
    (0.5, 0.6, "0.5-0.6"),
    (0.6, 0.7, "0.6-0.7"),
    (0.7, 0.8, "0.7-0.8"),
    (0.8, 0.9, "0.8-0.9"),
    (0.9, 1.0, "0.9<p<1.0"),
    (1.0, 1.0 + 1e-9, "1.0(確実にKO)"),
]


def _retreat_confidence_bin(prob: float) -> str | None:
    for lo, hi, label in _RETREAT_CONFIDENCE_BINS:
        if lo <= prob < hi:
            return label
    return None


def reliability_table(lines: list[str], rows: list[tuple[float, bool, bool]]) -> float:
    results = to_prediction_results(rows)
    if not results:
        lines.append("(サンプルなし)")
        lines.append("")
        return float("nan")
    ece = compute_ece(results)
    n = len(results)
    positive_rate = sum(1 for _p, _o, actual in rows if actual) / n
    lines.append(f"- サンプル数: {n}")
    lines.append(f"- 実際にきぜつした割合: {positive_rate * 100:.2f}%")
    lines.append(f"- ECE(expected calibration error): {ece:.4f}")
    lines.append("")
    lines.append("| 予測確率ビン | サンプル数 | 平均予測確率 | 実際の的中率 |")
    lines.append("|---|---:|---:|---:|")

    groups: dict[str, list[PredictionResult]] = {label: [] for _, _, label in _RETREAT_CONFIDENCE_BINS}
    for r in results:
        cbin = _retreat_confidence_bin(r.top1_confidence)
        if cbin is not None:
            groups[cbin].append(r)
    for _, _, label in _RETREAT_CONFIDENCE_BINS:
        group = groups[label]
        if not group:
            continue
        gn = len(group)
        avg_conf = sum(r.top1_confidence for r in group) / gn
        hit_rate = sum(1 for r in group if r.true_label == "hit") / gn
        lines.append(f"| {label} | {gn} | {avg_conf * 100:.4f}% | {hit_rate * 100:.2f}% |")
    lines.append("")
    return ece


def _prf(predicted_positive: list[bool], actual: list[bool]) -> tuple[float, float, float]:
    """(precision, recall, 陽性判定率) を返す。"""
    n = len(actual)
    tp = sum(1 for p, a in zip(predicted_positive, actual) if p and a)
    fp = sum(1 for p, a in zip(predicted_positive, actual) if p and not a)
    fn = sum(1 for p, a in zip(predicted_positive, actual) if not p and a)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    positive_rate = sum(predicted_positive) / n if n else float("nan")
    return precision, recall, positive_rate


def threshold_sweep_table(lines: list[str], rows: list[tuple[float, bool, bool]]) -> None:
    actual = [a for _p, _o, a in rows]
    lines.append("| 閾値 | 「危険」判定率 | precision(危険判定のうち実際にきぜつ) | recall(実際のきぜつのうち捕捉) |")
    lines.append("|---:|---:|---:|---:|")
    for threshold in _THRESHOLD_SWEEP:
        predicted = [prob >= threshold for prob, _o, _a in rows]
        precision, recall, positive_rate = _prf(predicted, actual)
        marker = " **(config既定)**" if threshold == 0.5 else ""
        lines.append(
            f"| {threshold:.1f}{marker} | {positive_rate * 100:.2f}% | {precision * 100:.2f}% | {recall * 100:.2f}% |"
        )
    lines.append("")
    lines.append("### 参考: 旧ブール判定(is_likely_ko_next_turn)")
    lines.append("")
    old_predicted = [old_bool for _p, old_bool, _a in rows]
    precision, recall, positive_rate = _prf(old_predicted, actual)
    lines.append("| 「危険」判定率 | precision | recall |")
    lines.append("|---:|---:|---:|")
    lines.append(f"| {positive_rate * 100:.2f}% | {precision * 100:.2f}% | {recall * 100:.2f}% |")
    lines.append("")


# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays-dir", default=str(_REPO_ROOT / "kaggle_replays" / "replays"))
    parser.add_argument("--report", default=str(_HERE / "output" / "retreat_ko_probability_eval_report.md"))
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=50,
        help="評価するリプレイ数の上限(0で無制限=全件)。デフォルト50(まず動作確認する用)。",
    )
    args = parser.parse_args()

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
        try:
            with replay_path.open(encoding="utf-8") as rf:
                replay = json.load(rf)
            steps = replay["steps"]
            records = load_replay_records(replay)
            for viewer in (0, 1):
                evaluate_viewer(steps, records, viewer, total)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - 1件のエラーで全体を止めない
            n_errors += 1
            errors.append(f"{replay_path.name}: {exc!r}")

        if i % 10 == 0 or i == len(replay_paths):
            elapsed = time.time() - t0
            print(
                f"  {i}/{len(replay_paths)} replays processed ({elapsed:.1f}s経過, サンプル数={len(total)})...",
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

    lines = ["# probabilistic_ko(likely_ko_probability_next_turn) リプレイバックテストレポート", ""]
    lines.append(f"- 評価対象リプレイ数: {len(replay_paths)}件中 {n_ok}件処理成功({n_errors}件エラー)")
    lines.append(f"- 実行時間: {elapsed_total:.1f}秒")
    lines.append(f"- サンプル数(視点×ターンの意思決定時点、両視点合計): {len(total)}")
    lines.append("")
    lines.append(
        "## 前提\n\n"
        "各リプレイ・各視点の各ターンの最初の意思決定時点で、その時点の自分のアクティブポケモンが"
        "「次に自分が意思決定するまでの間(実質的に相手の次ターン)」に実際にきぜつしたかを "
        "ground truth とする。詳細・既知の限界はスクリプト冒頭のdocstring参照。\n"
    )

    lines.append("## 全サンプル(にげて助かったケース込み)")
    lines.append("")
    lines.append("### reliability(予測確率 vs 実際のきぜつ率)")
    lines.append("")
    all_rows = total.all_subset()
    ece = reliability_table(lines, all_rows)
    print(f"ECE(全サンプル)={ece:.4f}" if not math.isnan(ece) else "ECE(全サンプル)=(サンプルなし)")

    lines.append("### 閾値スイープ(threshold候補ごとのprecision/recall)")
    lines.append("")
    threshold_sweep_table(lines, all_rows)

    n_retreated = total.n_retreated_away()
    no_intervention_rows = total.no_intervention_subset()
    lines.append("## 「介入なし」サブセット(自発的ににげたサンプルを除外)")
    lines.append("")
    lines.append(
        f"- 除外したサンプル数(判定後、次の自分の意思決定までの間ににげた): {n_retreated} / {len(total)}"
        f"({n_retreated / len(total) * 100:.2f}%)\n"
        "- ここでの「実際にきぜつしたか」は、プレイヤーが何もしなかった場合の結果に近いはずで、"
        "「にげて助かった」ケースによる交絡を除いた、より素直なcalibrationになる想定。\n"
    )
    lines.append("### reliability(予測確率 vs 実際のきぜつ率、介入なしサブセット)")
    lines.append("")
    ece_clean = reliability_table(lines, no_intervention_rows)
    print(
        f"ECE(介入なしサブセット)={ece_clean:.4f}" if not math.isnan(ece_clean) else "ECE(介入なしサブセット)=(サンプルなし)"
    )

    lines.append("### 閾値スイープ(threshold候補ごとのprecision/recall、介入なしサブセット)")
    lines.append("")
    threshold_sweep_table(lines, no_intervention_rows)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nレポートを {args.report} に書き出しました")


if __name__ == "__main__":
    main()
