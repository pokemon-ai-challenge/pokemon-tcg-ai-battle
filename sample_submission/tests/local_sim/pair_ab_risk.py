"""EXP-A44 P0: リスク調整Determinizationのペア A/B 計測ハーネス。

``docs/plans/search/EXP-A44-risk-adjusted-determinization.md`` §7.1 の方式（同一エージェント内で
``yourIndex`` ごとに設定（config）を切り替え、1ペア = 2試合で先後（player0/player1）を入れ替える）を
ローカルシミュレーションで実行する。既存の ``tests/local_sim/test_local_game_advanced.py``
（--opponent self/random の単純な複数戦実行）は変更せず、本スクリプトを新規に追加する。

## 設計判断（要件書に明記が無いため、このスクリプトの実装で決めた点）

- ``main.agent`` / ``rule_based_agent.agent`` は ``action_selection.selector.select_action`` に
  config を渡さず、内部で ``core.config.load_config()``（モジュールレベルキャッシュ）を使う。
  同一プロセス内で Control/Treatment 2つの config を同時に使い分けるには、このキャッシュを
  経由しない経路が要る。本スクリプトは ``rule_based_agent.agent`` と同じ処理順序
  （``match_context.update`` → 新規試合検知 → ``selector.select_action``）を
  ``make_agent_for_config()`` としてそのまま再現し、``config`` 引数だけを明示的に注入する
  （env var 切り替えや ``selector`` のモンキーパッチより副作用が少ないと判断した）。
- 自己対戦（ミラー）の性質上、``match_context`` / ``opponent_modeling.tracker`` / ``proposals`` の
  モジュールレベル状態は両陣営（player0/player1）で共有される。これは既存の
  ``test_local_game_advanced.py --opponent self`` と同じ制約であり、本スクリプト固有の問題では
  ないため踏襲する（Stage 2 以降で気になれば別プロセス化を検討する）。
- 「サイド差+2以上の優勢局面からの勝ち切り率」は、各試合中に一度でも
  「自分の残サイド枚数 <= 相手の残サイド枚数 - 2」になった役割（control/treatment）について、
  最終的にその試合に勝てたかを集計する（役割単位の集計。プレイヤーindexではなく
  Control/Treatmentのラベルで集計するのがペアA/Bの目的に合う）。
- 「例外」「不正Action」はいずれもゲームを止めずカウントだけして次の試合へ進む
  （1試合のクラッシュで測定全体を失わないため）。「ハング」は1手あたりの思考時間が
  ``--hang-threshold-ms``（既定5000ms）を超えた回数、および1試合の総手数が
  ``--max-steps-per-game``（既定500）を超えて打ち切られた回数として計測する。
- 非対称対面（要件書 §7.1 「最低限の代替策」）: このブランチには他アーキタイプの実デッキCSVが
  同梱されていない（``decks/`` は自分のデッキ専用プロファイルのみ）。そのため
  ``--vs-random``（相手は ``random_agent`` 固定・デッキは同一）を、現時点で用意できる
  最も近い非対称対面として提供する。別デッキCSVが手に入り次第 ``--opponent-deck`` で
  差し替えられるよう引数だけ用意しておく（要件書 §7.2 の測定限界の事前宣言どおり、
  真の非対称対面が無いことは既知の制約として報告する）。

## 使い方

```powershell
# Control(rule_lethal)を自分自身とペアA/Bで20戦（=10ペア）回し、50%±ノイズを確認する。
python tests/local_sim/pair_ab_risk.py --pairs 10 --control rule_lethal --treatment rule_lethal

# Control vs Treatment(rule_risk) を200戦（Stage 2 想定、P4以降で使用）
python tests/local_sim/pair_ab_risk.py --pairs 100 --control rule_lethal --treatment rule_risk

# 非対称対面（相手はランダム方策、Treatment側の機能確認用）
python tests/local_sim/pair_ab_risk.py --pairs 10 --treatment rule_risk --vs-random
```
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Observation, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from ptcg_ai.action_selection import selector  # noqa: E402
from ptcg_ai.core.config import load_config  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402
from ptcg_ai.opponent_modeling import tracker as opponent_tracker  # noqa: E402
from ptcg_ai.rule_based.main_turn_parts import proposals  # noqa: E402
from ptcg_ai.rule_based.rule_based_agent import read_deck_csv  # noqa: E402
from ptcg_ai.search import risk_determinization  # noqa: E402

AgentFn = Callable[[dict], list[int]]

_PRIZE_ADVANTAGE_MARGIN = 2  # サイド差+2以上を「優勢」とみなす（要件書 §5.4 / §6.2 と同じ閾値）。


def random_agent(obs_dict: dict) -> list[int]:
    """``test_local_game_advanced.py`` と同じ単純ランダム方策（非対称対面の相手用）。"""
    import random

    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)


def make_agent_for_config(config: dict) -> AgentFn:
    """``rule_based_agent.agent`` と同じ処理順序で、指定 config を注入した agent 関数を返す。

    ``core.config.load_config()`` のモジュールレベルキャッシュを経由しないため、同一プロセス内で
    Control/Treatment 2つの config を安全に使い分けられる。
    """

    def _agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        # rule_based_agent.agent() と同じ副作用の順序（match_context.update は判断に影響しない）。
        match_context.update(obs)

        if obs.select is None:
            opponent_tracker.reset()
            proposals.reset_turn_state()
            return read_deck_csv()

        return selector.select_action(obs, read_deck_csv(), config=config)

    return _agent


@dataclass
class RoleStats:
    """control / treatment いずれか1役割ぶんの集計。"""

    games: int = 0
    wins: int = 0
    illegal_actions: int = 0
    exceptions: int = 0
    hangs: int = 0
    decision_times_ms: list[float] = field(default_factory=list)
    dominant_games: int = 0  # サイド差+2以上に到達した試合数
    dominant_wins: int = 0  # そのうち勝ち切った試合数

    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def z_score_vs_half(self) -> float:
        """勝率50%に対する片側zスコア（正規近似）。"""
        if self.games == 0:
            return 0.0
        p = 0.5
        se = math.sqrt(p * (1 - p) / self.games)
        return (self.win_rate() - p) / se if se > 0 else 0.0

    def closing_rate(self) -> float:
        return self.dominant_wins / self.dominant_games if self.dominant_games else float("nan")

    def percentiles(self) -> dict[str, float]:
        if not self.decision_times_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        ordered = sorted(self.decision_times_ms)

        def _pct(p: float) -> float:
            if len(ordered) == 1:
                return ordered[0]
            k = (len(ordered) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return ordered[int(k)]
            return ordered[f] + (ordered[c] - ordered[f]) * (k - f)

        return {"p50": _pct(0.50), "p95": _pct(0.95), "p99": _pct(0.99)}


def validate_action(obs: Observation, action: list[int]) -> bool:
    """selector.is_valid_action 相当の外形チェック（ハーネス側でも独立に検証する）。"""
    if obs.select is None:
        return isinstance(action, list) and len(action) == 60
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        return False
    if not (obs.select.minCount <= len(action) <= obs.select.maxCount):
        return False
    if len(action) != len(set(action)):
        return False
    return all(0 <= i < len(obs.select.option) for i in action)


def _prize_diff_for(state, player_index: int) -> int:
    """player_index 視点のサイド差（自分の残サイド − 相手の残サイド）。負なら優勢。"""
    own = len(state.players[player_index].prize)
    opp = len(state.players[1 - player_index].prize)
    return own - opp


def play_one_game(
    role_by_index: dict[int, str],
    agent_by_index: dict[int, AgentFn],
    stats_by_role: dict[str, RoleStats],
    deck0: list[int],
    deck1: list[int],
    hang_threshold_ms: float,
    max_steps: int,
    verbose: bool,
) -> None:
    """1試合実行し、``role_by_index`` に基づいて ``stats_by_role`` へ集計する。"""
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        raise RuntimeError(f"battle_start failed with errorType={start_data.errorType}")

    reached_dominant: dict[str, bool] = {role: False for role in stats_by_role}
    steps = 0
    hung = False
    try:
        while True:
            obs: Observation = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                break
            if steps >= max_steps:
                hung = True
                break

            if obs.current is not None:
                for idx in (0, 1):
                    role = role_by_index[idx]
                    if _prize_diff_for(obs.current, idx) <= -_PRIZE_ADVANTAGE_MARGIN:
                        reached_dominant[role] = True

            acting_player = obs.current.yourIndex if obs.current is not None else 0
            role = role_by_index[acting_player]
            stats = stats_by_role[role]

            start = time.perf_counter()
            try:
                action = agent_by_index[acting_player](obs_dict)
            except Exception:
                stats.exceptions += 1
                break
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            stats.decision_times_ms.append(elapsed_ms)
            if elapsed_ms > hang_threshold_ms:
                stats.hangs += 1

            if not validate_action(obs, action):
                stats.illegal_actions += 1
                break

            if verbose and obs.select is not None:
                print(
                    f"turn={obs.current.turn if obs.current is not None else '?'} "
                    f"player={acting_player}({role}) context={obs.select.context} action={action}"
                )

            obs_dict = battle_select(action)
            steps += 1

        final_obs: Observation = to_observation_class(obs_dict)
        result = final_obs.current.result if final_obs.current is not None else -1
    finally:
        battle_finish()

    if hung:
        for role in stats_by_role:
            stats_by_role[role].hangs += 1

    for idx in (0, 1):
        role = role_by_index[idx]
        stats = stats_by_role[role]
        stats.games += 1
        if result == idx:
            stats.wins += 1
        if reached_dominant[role]:
            stats.dominant_games += 1
            if result == idx:
                stats.dominant_wins += 1


def run_pair(
    control_config: dict,
    treatment_config: dict,
    pair_index: int,
    deck: list[int],
    opponent_deck: list[int] | None,
    vs_random: bool,
    stats_by_role: dict[str, RoleStats],
    hang_threshold_ms: float,
    max_steps: int,
    verbose: bool,
) -> None:
    """1ペア=2試合。先攻/後攻（player0/player1）で control/treatment の座席を入れ替える。"""
    control_agent = make_agent_for_config(control_config)
    treatment_agent = make_agent_for_config(treatment_config)
    opp_deck = opponent_deck if opponent_deck is not None else deck

    if vs_random:
        # 非対称対面: treatment(またはcontrol) vs random_agent。役割は "treatment" と "random" とする。
        matchups = [
            ({0: "treatment", 1: "random"}, {0: treatment_agent, 1: random_agent}, deck, opp_deck),
            ({0: "random", 1: "treatment"}, {0: random_agent, 1: treatment_agent}, opp_deck, deck),
        ]
    else:
        matchups = [
            (
                {0: "control", 1: "treatment"},
                {0: control_agent, 1: treatment_agent},
                deck,
                opp_deck,
            ),
            (
                {0: "treatment", 1: "control"},
                {0: treatment_agent, 1: control_agent},
                opp_deck,
                deck,
            ),
        ]

    for role_by_index, agent_by_index, deck0, deck1 in matchups:
        play_one_game(
            role_by_index,
            agent_by_index,
            stats_by_role,
            deck0,
            deck1,
            hang_threshold_ms,
            max_steps,
            verbose,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP-A44 ペアA/B計測ハーネス（Stage 1/2）。")
    parser.add_argument("--pairs", type=int, default=10, help="ペア数（1ペア=2試合、先後交互）。")
    parser.add_argument("--control", default="rule_lethal", help="Control config名（configs/<name>.json）。")
    parser.add_argument("--treatment", default="rule_lethal", help="Treatment config名。")
    parser.add_argument("--vs-random", action="store_true", help="非対称対面: treatment vs random_agent。")
    parser.add_argument("--opponent-deck", default=None, help="非対称対面用の相手デッキCSV（任意）。")
    parser.add_argument("--hang-threshold-ms", type=float, default=5000.0, help="この時間を超えた1手をハング扱いにする。")
    parser.add_argument("--max-steps-per-game", type=int, default=500, help="この手数を超えたら試合を打ち切りハング扱いにする。")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--override-check",
        action="store_true",
        help=(
            "要件書 §5.10: 勝率A/Bを回す前に override率(overrides/fired)を実測するモード。"
            "--treatment のconfigをTreatment/Control両陣営に使い、risk_determinization.get_stats() "
            "を集計して report する（勝率は見ない）。"
        ),
    )
    return parser.parse_args()


def _load_opponent_deck(path: str | None) -> list[int] | None:
    if path is None:
        return None
    with open(path, "r") as fh:
        lines = fh.read().split("\n")
    return [int(lines[i]) for i in range(60)]


def run_override_check(args: argparse.Namespace) -> None:
    """要件書 §5.10: 勝率A/Bを回す前の override率(overrides/fired)チェック。

    ``--treatment`` の config を両陣営（player0/player1）に使い、``args.pairs`` ペア
    （= 2*pairs 試合）を自己対戦で回して ``risk_determinization.get_stats()`` を集計する。
    勝率は見ない（Controlを使わないので比較対象が無い）。tie_ratio 等のチューニングに使う。
    """
    deck = read_deck_csv()
    treatment_config = load_config(args.treatment)
    agent = make_agent_for_config(treatment_config)

    risk_determinization.reset_stats()

    illegal = 0
    exceptions = 0
    games = 0

    for pair_index in range(args.pairs):
        for deck0, deck1 in ((deck, deck), (deck, deck)):
            obs_dict, start_data = battle_start(deck0, deck1)
            if start_data.errorType != 0:
                raise RuntimeError(f"battle_start failed with errorType={start_data.errorType}")
            steps = 0
            try:
                while True:
                    obs: Observation = to_observation_class(obs_dict)
                    if obs.current is not None and obs.current.result != -1:
                        break
                    if steps >= args.max_steps_per_game:
                        break
                    try:
                        action = agent(obs_dict)
                    except Exception:
                        exceptions += 1
                        break
                    if not validate_action(obs, action):
                        illegal += 1
                        break
                    obs_dict = battle_select(action)
                    steps += 1
            finally:
                battle_finish()
            games += 1
        print(f"pair {pair_index + 1}/{args.pairs} done")

    stats = risk_determinization.get_stats()
    fired = stats["fired"]
    overrides = stats["overrides"]
    override_rate = (overrides / fired) if fired else float("nan")

    print()
    print("=== EXP-A44 override-rate check (要件書 §5.10) ===")
    print(f"config={args.treatment} games={games} illegal={illegal} exceptions={exceptions}")
    print(f"invocations={stats['invocations']} fired={fired} overrides={overrides} "
          f"override_rate={override_rate:.4f}")
    print(f"reject_reasons={stats['reject_reasons']}")
    print(f"tie_set_sizes(all)={stats['tie_set_sizes']}")
    print(f"tie_set_size_histogram={stats['tie_set_size_histogram']}")
    print(f"avg_tie_set_size={stats['avg_tie_set_size']:.3f}")
    print(f"truncated={stats['truncated']} evaluations_total={stats['evaluations_total']}")
    print(f"avg_override_score_delta={stats['avg_override_score_delta']:.4f}")
    print(f"decision_ms P50/P95/P99={stats['p50_ms']:.1f}/{stats['p95_ms']:.1f}/{stats['p99_ms']:.1f}")
    threshold = 0.05
    verdict = "PASS" if fired > 0 and override_rate >= threshold else "FAIL"
    print(f"criterion: overrides/fired >= {threshold} -> {verdict}")


def main() -> None:
    args = parse_args()
    if args.override_check:
        run_override_check(args)
        return

    deck = read_deck_csv()
    opponent_deck = _load_opponent_deck(args.opponent_deck)

    control_config = load_config(args.control)
    treatment_config = load_config(args.treatment)

    roles = ("treatment", "random") if args.vs_random else ("control", "treatment")
    stats_by_role: dict[str, RoleStats] = {role: RoleStats() for role in roles}

    for pair_index in range(args.pairs):
        run_pair(
            control_config,
            treatment_config,
            pair_index,
            deck,
            opponent_deck,
            args.vs_random,
            stats_by_role,
            args.hang_threshold_ms,
            args.max_steps_per_game,
            args.verbose,
        )
        print(f"pair {pair_index + 1}/{args.pairs} done")

    print()
    print("=== EXP-A44 pair A/B summary ===")
    for role, stats in stats_by_role.items():
        pct = stats.percentiles()
        print(
            f"[{role}] games={stats.games} wins={stats.wins} "
            f"win_rate={stats.win_rate():.3f} z_vs_50%={stats.z_score_vs_half():.2f} "
            f"illegal={stats.illegal_actions} exceptions={stats.exceptions} hangs={stats.hangs} "
            f"decision_ms(P50/P95/P99)={pct['p50']:.1f}/{pct['p95']:.1f}/{pct['p99']:.1f} "
            f"dominant_games={stats.dominant_games} closing_rate={stats.closing_rate():.3f}"
        )


if __name__ == "__main__":
    main()
