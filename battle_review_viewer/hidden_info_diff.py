"""非公開情報推定レイヤー（hidden_information）の、神視点によるキャリブレーション検証ツール。

``opponent_knowledge_diff.py``（公開ゾーンの観測 vs 神視点の diff）と対になる位置づけで、
非公開ゾーン（山札・手札・サイド）について同じことを行う。``kaggle_replays/deck_predictor/
evaluate_hidden_information.py`` が Kaggle リプレイ(数千件規模、量はあるが手札は自己視点、
サイドは取得済み分しか真値が取れない)で検証するのに対し、本スクリプトはローカル自己対戦
（``cg.game.battle_start``/``battle_select``経由）を使い、``cg.game.visualize_data()`` の
神視点から**未取得のサイドカードの中身まで含めた完全な ground truth** を取る（試合数は少ないが、
非公開情報を一切欠かさず検証できる。implementation-plan.md Phase 3・3.1「データソースが2種類」の
2つ目に対応）。

各意思決定時点で ``HybridDeckPredictor.predict()`` -> ``OpponentHiddenState.marginals()`` を計算し、
同じ瞬間の神視点(``visualize_data()``)から相手の山札/手札/サイド(未取得分含む)の真の card_id 集合を
取り出して突き合わせる。reliability テーブル・ECE の定義は
``kaggle_replays/deck_predictor/evaluate.py`` の考え方をそのまま踏襲する
（本ファイルは battle_review_viewer 配下の他スクリプトと同じく自己完結にするため、コード自体は
複製している。ロジックの二重実装ではなく「同じ設計方針を別ディレクトリで再現している」点に注意）。

自動テスト対象外（implementation-plan.md Phase 3 のテスト方針: 本スクリプトは手動実行のレポート
ツール）。

使い方:
  python hidden_info_diff.py --matches 5
  python hidden_info_diff.py --matches 3 --player-deck sample_submission/deck.csv
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_SUBMISSION_DIR = ROOT_DIR / "sample_submission"
DEFAULT_DECK_PATH = SAMPLE_SUBMISSION_DIR / "deck.csv"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "output" / "hidden_info_diff_report.md"
DEFAULT_POOL_PATH = SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "hidden_information" / "archetype_card_pool.json"

if str(SAMPLE_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cg.api import Observation, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start, visualize_data  # noqa: E402
from main import agent  # noqa: E402
from ptcg_ai.hidden_information.opponent_hidden_state import OpponentHiddenState  # noqa: E402
from ptcg_ai.opponent_modeling.hybrid_predictor import HybridDeckPredictor  # noqa: E402
from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge  # noqa: E402

AgentFn = Any


# ----------------------------------------------------------------------
# デッキ読み込み・エージェント準備（export_replay.py と同じパターン）


def read_deck_csv_file(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    deck: list[int] = []
    for raw_value in text.replace(",", "\n").splitlines():
        value = raw_value.strip()
        if not value or value.startswith("#"):
            continue
        deck.append(int(value))
    if len(deck) != 60:
        raise ValueError(f"{path} must contain exactly 60 card IDs, but found {len(deck)}.")
    return deck


def resolve_deck_path(value: str | Path | None) -> Path:
    if value is None or str(value).strip() == "":
        return DEFAULT_DECK_PATH
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def with_initial_deck(base_agent: AgentFn, deck: list[int]) -> AgentFn:
    def wrapped(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck)
        return base_agent(obs_dict)

    return wrapped


def current_visual_frame() -> dict[str, Any]:
    visual_history = json.loads(visualize_data())
    return visual_history[-1]


# ----------------------------------------------------------------------
# 神視点(visualize_data)から、非公開ゾーンの ground truth を取り出す
# （opponent_knowledge_diff.collect_ground_truth() が公開ゾーンについてやっているのと対になる処理）


def collect_hidden_ground_truth(current: dict[str, Any] | None, opponent_index: int) -> dict[str, Counter[int]]:
    """神視点の盤面(current)から、opponent_index 側の 山札/手札/サイド の真の card_id 集合を返す。

    visualize_data() は ``deck``/``hand``/``prize`` を（本来非公開なものも含めて）フルに返す
    （実際のエージェント視点の Observation では ``prize`` は未取得分が None、``hand`` は相手なら
    None になる）。戻り値は {"deck": Counter, "hand": Counter, "prize": Counter}（card_id -> 枚数）。
    """
    empty = {"deck": Counter(), "hand": Counter(), "prize": Counter()}
    if current is None:
        return empty
    players = current.get("players") or []
    if not (0 <= opponent_index < len(players)):
        return empty
    player = players[opponent_index]

    result = {
        "deck": Counter(c["id"] for c in (player.get("deck") or []) if c is not None),
        "hand": Counter(c["id"] for c in (player.get("hand") or []) if c is not None),
        "prize": Counter(c["id"] for c in (player.get("prize") or []) if c is not None),
    }
    return result


# ----------------------------------------------------------------------
# reliability / ECE（kaggle_replays/deck_predictor/evaluate.py と同じ設計方針。自己完結のため複製）

_CONFIDENCE_BINS = [
    (0.0, 0.5, "0.0-0.5"),
    (0.5, 0.6, "0.5-0.6"),
    (0.6, 0.7, "0.6-0.7"),
    (0.7, 0.8, "0.7-0.8"),
    (0.8, 0.9, "0.8-0.9"),
    (0.9, 1.0 + 1e-9, "0.9-1.0"),
]
_ECE_BIN_COUNT = 10


def confidence_bin(prob: float) -> str | None:
    for lo, hi, label in _CONFIDENCE_BINS:
        if lo <= prob < hi:
            return label
    return None


def compute_ece(pairs: list[tuple[float, bool]], n_bins: int = _ECE_BIN_COUNT) -> float:
    if not pairs:
        return float("nan")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for prob, actual in pairs:
        idx = min(int(prob * n_bins), n_bins - 1)
        bins[idx].append((prob, actual))
    n = len(pairs)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        bn = len(bucket)
        avg_conf = sum(p for p, _ in bucket) / bn
        hit_rate = sum(1 for _, a in bucket if a) / bn
        ece += abs(avg_conf - hit_rate) * bn / n
    return ece


def reliability_section(lines: list[str], title: str, pairs: list[tuple[float, bool]]) -> float:
    lines.append(f"### {title}")
    lines.append("")
    if not pairs:
        lines.append("(サンプルなし)")
        lines.append("")
        return float("nan")

    n = len(pairs)
    ece = compute_ece(pairs)
    positive_rate = sum(1 for _, actual in pairs if actual) / n

    lines.append(f"- サンプル数: {n}")
    lines.append(f"- 実際の陽性率(そのゾーンに実在した割合): {positive_rate * 100:.2f}%")
    lines.append(f"- ECE(expected calibration error): {ece:.4f}")
    lines.append("")
    lines.append("| 予測確率ビン | サンプル数 | 平均予測確率 | 実際の的中率 |")
    lines.append("|---|---:|---:|---:|")
    groups: dict[str, list[tuple[float, bool]]] = {label: [] for _, _, label in _CONFIDENCE_BINS}
    for prob, actual in pairs:
        cbin = confidence_bin(prob)
        if cbin is not None:
            groups[cbin].append((prob, actual))
    for _, _, label in _CONFIDENCE_BINS:
        group = groups[label]
        if not group:
            continue
        gn = len(group)
        avg_conf = sum(p for p, _ in group) / gn
        hit_rate = sum(1 for _, a in group if a) / gn
        lines.append(f"| {label} | {gn} | {avg_conf * 100:.2f}% | {hit_rate * 100:.2f}% |")
    lines.append("")
    print(f"  [{title}] n={n} 陽性率={positive_rate * 100:.2f}% ECE={ece:.4f}")
    return ece


# ----------------------------------------------------------------------
# ナイーブベースライン(全アーキタイプ等重み)


def load_naive_posterior(pool_path: Path) -> dict[str, float]:
    if not pool_path.exists():
        return {}
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    archetypes = list(payload.get("archetypes", {}).keys())
    if not archetypes:
        return {}
    weight = 1.0 / len(archetypes)
    return {a: weight for a in archetypes}


# ----------------------------------------------------------------------
# 1試合分の実行


class MatchSamples:
    def __init__(self) -> None:
        self.deck_real: list[tuple[float, bool]] = []
        self.hand_real: list[tuple[float, bool]] = []
        self.prize_real: list[tuple[float, bool]] = []
        self.deck_naive: list[tuple[float, bool]] = []
        self.hand_naive: list[tuple[float, bool]] = []
        self.prize_naive: list[tuple[float, bool]] = []
        self.n_decision_points = 0

    def extend_from(self, other: "MatchSamples") -> None:
        self.deck_real.extend(other.deck_real)
        self.hand_real.extend(other.hand_real)
        self.prize_real.extend(other.prize_real)
        self.deck_naive.extend(other.deck_naive)
        self.hand_naive.extend(other.hand_naive)
        self.prize_naive.extend(other.prize_naive)
        self.n_decision_points += other.n_decision_points


def run_match(
    player_agents: dict[int, AgentFn],
    deck0: list[int],
    deck1: list[int],
    predictor: HybridDeckPredictor,
    naive_posterior: dict[str, float],
    max_steps: int,
) -> MatchSamples:
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        raise RuntimeError(f"battle_start failed with errorType={start_data.errorType}")

    # 両プレイヤー視点それぞれについて、「自分から見た相手」の観測蓄積器と非公開情報推定器を持つ
    # (evaluate_hidden_information.py と同じく、両視点を同時に検証してサンプルを稼ぐ)。
    knowledge = {0: OpponentKnowledge(opponent_index=1), 1: OpponentKnowledge(opponent_index=0)}
    opp_state = {0: OpponentHiddenState(), 1: OpponentHiddenState()}

    samples = MatchSamples()
    steps = 0
    result = None

    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                result = obs.current.result
                break
            if obs.current is None or obs.select is None:
                break

            viewer = obs.current.yourIndex
            opponent = 1 - viewer

            knowledge[viewer].update_from_logs(obs.logs)
            knowledge[viewer].update_from_state(obs.current)
            features = knowledge[viewer].get_prediction_features()
            archetype_posterior = predictor.predict(features["observed_cards"], obs.current.turn)
            opponent_player_state = obs.current.players[opponent]

            frame = current_visual_frame()
            ground_truth = collect_hidden_ground_truth(frame.get("current"), opponent)
            samples.n_decision_points += 1

            opp_state[viewer].update(archetype_posterior, features["observed_card_ids"], opponent_player_state)
            real_marginals = opp_state[viewer].marginals()
            for card_id, probs in real_marginals.items():
                if card_id is None:
                    continue
                samples.deck_real.append((probs["deck"], ground_truth["deck"].get(card_id, 0) > 0))
                samples.hand_real.append((probs["hand"], ground_truth["hand"].get(card_id, 0) > 0))
                samples.prize_real.append((probs["prize"], ground_truth["prize"].get(card_id, 0) > 0))

            if naive_posterior:
                opp_state[viewer].update(naive_posterior, features["observed_card_ids"], opponent_player_state)
                naive_marginals = opp_state[viewer].marginals()
                for card_id, probs in naive_marginals.items():
                    if card_id is None:
                        continue
                    samples.deck_naive.append((probs["deck"], ground_truth["deck"].get(card_id, 0) > 0))
                    samples.hand_naive.append((probs["hand"], ground_truth["hand"].get(card_id, 0) > 0))
                    samples.prize_naive.append((probs["prize"], ground_truth["prize"].get(card_id, 0) > 0))

            action = player_agents[viewer](obs_dict)
            obs_dict = battle_select(action)
            steps += 1
            if steps >= max_steps:
                raise RuntimeError(f"Reached max_steps={max_steps} before the match finished.")
    finally:
        battle_finish()

    print(f"  試合終了: result={result} steps={steps} 意思決定時点数={samples.n_decision_points}")
    return samples


# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches", type=int, default=5, help="ローカル自己対戦を何試合実行するか")
    parser.add_argument("--player-deck", default=None, help="player0 のデッキCSV(未指定なら sample_submission/deck.csv)")
    parser.add_argument("--opponent-deck", default=None, help="player1 のデッキCSV(未指定なら player0 と同じ)")
    parser.add_argument("--pool-path", default=str(DEFAULT_POOL_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1, help="1試合目のシード(以降は+1ずつずらす)")
    args = parser.parse_args()

    deck0_path = resolve_deck_path(args.player_deck)
    deck1_path = resolve_deck_path(args.opponent_deck) if args.opponent_deck else deck0_path
    deck0 = read_deck_csv_file(deck0_path)
    deck1 = read_deck_csv_file(deck1_path)
    print(f"player0 deck: {deck0_path}")
    print(f"player1 deck: {deck1_path}")

    predictor = HybridDeckPredictor()
    if not predictor.is_ready:
        print("エラー: HybridDeckPredictor が未ロード状態です", file=sys.stderr)
        sys.exit(1)
    print(f"predictor.mode = {predictor.mode}")

    pool_path = Path(args.pool_path)
    probe_state = OpponentHiddenState(pool_path=pool_path)
    if not probe_state.is_ready:
        print(f"エラー: archetype_card_pool.json を読み込めませんでした: {pool_path}", file=sys.stderr)
        sys.exit(1)
    naive_posterior = load_naive_posterior(pool_path)
    print(f"ナイーブベースライン(全アーキタイプ等重み)のアーキタイプ数: {len(naive_posterior)}")

    player_agents = {0: with_initial_deck(agent, deck0), 1: with_initial_deck(agent, deck1)}

    total = MatchSamples()
    n_ok = 0
    for i in range(args.matches):
        print(f"[{i + 1}/{args.matches}] 対戦開始(seed={args.seed + i})...")
        random.seed(args.seed + i)
        try:
            samples = run_match(player_agents, deck0, deck1, predictor, naive_posterior, args.max_steps)
            total.extend_from(samples)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 -- 1試合の失敗で全体を止めない
            print(f"  エラー: {exc!r}", file=sys.stderr)

    print(f"\n{n_ok}/{args.matches}試合が正常終了。意思決定時点合計={total.n_decision_points}")

    lines = ["# 非公開情報推定レイヤー(hidden_information) 神視点キャリブレーション検証レポート", ""]
    lines.append(f"- 対戦数: {args.matches}試合中 {n_ok}試合が正常終了")
    lines.append(f"- player0 deck: `{deck0_path}`")
    lines.append(f"- player1 deck: `{deck1_path}`")
    lines.append(f"- predictor.mode: {predictor.mode}")
    lines.append(f"- 意思決定時点数(両プレイヤー×両視点の合計): {total.n_decision_points}")
    lines.append("")
    lines.append(
        "## 前提\n\n"
        "本レポートの ground truth は `cg.game.visualize_data()` の神視点から取得しており、"
        "**未取得のサイドカードの中身も含めて100%正確**(Kaggle リプレイベースの "
        "`kaggle_replays/deck_predictor/evaluate_hidden_information.py` と異なり、サイドの近似が"
        "不要)。ただし試合数が少ないためサンプルの偏り(自己対戦・同一アーキタイプ同士)には注意。\n"
    )

    lines.append("## 山末(deck) reliability")
    lines.append("")
    ece_deck_real = reliability_section(lines, "実装(HybridDeckPredictor の事後分布)", total.deck_real)
    ece_deck_naive = reliability_section(lines, "ナイーブベースライン(全アーキタイプ等重み)", total.deck_naive)

    lines.append("## 手札(hand) reliability")
    lines.append("")
    ece_hand_real = reliability_section(lines, "実装(HybridDeckPredictor の事後分布)", total.hand_real)
    ece_hand_naive = reliability_section(lines, "ナイーブベースライン(全アーキタイプ等重み)", total.hand_naive)

    lines.append("## サイド(prize、未取得分含む)reliability")
    lines.append("")
    ece_prize_real = reliability_section(lines, "実装(HybridDeckPredictor の事後分布)", total.prize_real)
    ece_prize_naive = reliability_section(lines, "ナイーブベースライン(全アーキタイプ等重み)", total.prize_naive)

    lines.append("## ECE まとめ")
    lines.append("")
    lines.append("| ゾーン | 実装ECE | ナイーブECE | 差分(実装-ナイーブ) |")
    lines.append("|---|---:|---:|---:|")
    for name, real, naive in (
        ("山末", ece_deck_real, ece_deck_naive),
        ("手札", ece_hand_real, ece_hand_naive),
        ("サイド", ece_prize_real, ece_prize_naive),
    ):
        if real == real and naive == naive:  # NaN チェック
            lines.append(f"| {name} | {real:.4f} | {naive:.4f} | {real - naive:+.4f} |")
        else:
            lines.append(f"| {name} | {real} | {naive} | - |")
    lines.append("")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nレポートを {report_path} に書き出しました")


if __name__ == "__main__":
    main()
