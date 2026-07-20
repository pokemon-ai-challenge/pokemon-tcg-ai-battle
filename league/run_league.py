"""対戦リーグ CLI: ``run_match.play_match`` を繰り返し呼び、勝率と信頼区間を算出する。

設計上の位置づけは ``sample_submission/docs/plans/ml-value-network/step2-design.md`` §5.2 /
``sample_submission/docs/plans/individual/shogo/ml-agent-plan.md`` の「足りない部品4:
評価基盤」に対応する。1試合の実行(``play_match``)には一切手を加えず、その上に
「N試合回して集計する」層だけを追加する。

先手/後手バイアスの排除
------------------------
``cg.game.battle_start`` は ``deck0``(player_index=0)を「先攻側」として明示的に扱う
(``battle_start`` の docstring: "deck0: List of card IDs included in the first player's
deck.")。CLAUDE.md にもある通り、このゲームには先手・後手の概念があり、player_index=0
に固定したエージェントが勝率で有利/不利になりうる。そのため必ず試合の半分は
(agent0=A, agent1=B)、残り半分は (agent0=B, agent1=A) で実行し、勝敗は player_index では
なく「エージェント名(A/B)」単位で集計する。

なお ``cg.api.State.firstPlayer``(実際にコイントスで先攻を得たプレイヤー)は
``MatchResult`` に含まれておらず、``run_match.py`` は変更禁止のためここから取得できない。
このスクリプトでの「先手/後手」は player_index=0/1 を指す(上記の通り deck0 側が
"first player" として API 上定義されているため、player_index をそのまま先手/後手の代理
指標として扱う)。
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

_LEAGUE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _LEAGUE_DIR.parent
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"

for _candidate in (str(_LEAGUE_DIR), str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from run_match import AgentFn, MatchResult, play_match  # noqa: E402

# エージェント名 -> agent(obs) 関数を提供するモジュールのレジストリ。
# 3つ目以降のエージェントが増えても、ここに1行追加するだけで --agent-a/--agent-b から
# 選べるようになる。
AGENT_REGISTRY: dict[str, str] = {
    "rule_based": "ptcg_ai.rule_based.rule_based_agent",
    "ml_policy": "ptcg_ai.ml_policy.ml_policy_agent",
}


def load_agent(name: str) -> AgentFn:
    """エージェント名から agent(obs) 関数を遅延 import で解決する。"""
    if name not in AGENT_REGISTRY:
        raise ValueError(f"unknown agent: {name!r} (choices: {sorted(AGENT_REGISTRY)})")
    module = importlib.import_module(AGENT_REGISTRY[name])
    return module.agent


def resolve_deck_path(value: str | Path | None) -> Path:
    """相対パスはリポジトリルート基準で解決する(battle_review_viewer/live_match.py の
    resolve_deck_path と同じ方針)。未指定なら sample_submission/deck.csv。
    """
    if value is None or str(value).strip() == "":
        return _SAMPLE_SUBMISSION_DIR / "deck.csv"
    path = Path(value)
    if not path.is_absolute():
        path = _ROOT_DIR / path
    return path.resolve()


def read_deck_csv_file(path: str | Path | None = None) -> list[int]:
    """deck.csv 形式(改行区切り、カンマ区切りいずれも可)の60枚デッキを読む。

    battle_review_viewer/live_match.py の read_deck_csv_file と同等ロジックの複製
    (この対戦リーグは battle_review_viewer に依存させない方針のため)。
    """
    resolved = resolve_deck_path(path)
    text = resolved.read_text(encoding="utf-8")
    deck: list[int] = []
    for raw_value in text.replace(",", "\n").splitlines():
        value = raw_value.strip()
        if not value or value.startswith("#"):
            continue
        deck.append(int(value))
    if len(deck) != 60:
        raise ValueError(f"Deck must contain exactly 60 card IDs, got {len(deck)}: {resolved}")
    return deck


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval(95%はデフォルトの z=1.95996...)。

    正規近似(Wald)区間ではなく Wilson 区間を使う(小標本・勝率が0/1に近いケースでも
    区間が [0, 1] の外に出ず安定するため)。n=0 の場合は情報が無いので [0, 1] を返す。
    """
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def _rate_stats(wins: int, n: int) -> dict:
    rate = wins / n if n else None
    lo, hi = wilson_interval(wins, n)
    return {
        "wins": wins,
        "games": n,
        "win_rate": rate,
        "wilson_95ci": [lo, hi],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="rule_based / ml_policy などのエージェント同士を自動対戦させ、"
        "先手/後手バイアスを排除した勝率とWilson信頼区間を算出する対戦リーグCLI。"
    )
    parser.add_argument(
        "--agent-a", choices=sorted(AGENT_REGISTRY), default="rule_based",
        help="エージェントA(default: rule_based)",
    )
    parser.add_argument(
        "--agent-b", choices=sorted(AGENT_REGISTRY), default="ml_policy",
        help="エージェントB(default: ml_policy)",
    )
    parser.add_argument("--games", type=int, default=500, help="総試合数(default: 500)")
    parser.add_argument(
        "--deck-a", default=None,
        help="エージェントAが使うデッキCSV(default: sample_submission/deck.csv)",
    )
    parser.add_argument(
        "--deck-b", default=None,
        help="エージェントBが使うデッキCSV(default: sample_submission/deck.csv)",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="試合iにはseed_start+iを渡す")
    parser.add_argument(
        "--out", default=None,
        help="結果JSONの出力先(default: league/results/<agent-a>_vs_<agent-b>_<timestamp>.json)",
    )
    parser.add_argument(
        "--progress-every", type=int, default=50,
        help="N試合ごとに進捗をstderrへ出力する(default: 50)",
    )
    return parser.parse_args(argv)


def resolve_out_path(value: str | None, agent_a: str, agent_b: str) -> Path:
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = _ROOT_DIR / path
        return path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return _LEAGUE_DIR / "results" / f"{agent_a}_vs_{agent_b}_{timestamp}.json"


def run_league(
    agent_a_name: str,
    agent_b_name: str,
    games: int,
    deck_a_path: str | Path | None,
    deck_b_path: str | Path | None,
    seed_start: int,
    progress_every: int,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> dict:
    """N試合を実行し、集計結果を dict(JSONにそのまま書ける形)で返す。"""

    requested_games = games
    games = requested_games - (requested_games % 2)
    if games != requested_games:
        log(
            f"[warn] --games {requested_games} は奇数のため、先手/後手の割り当てが偏らないよう "
            f"{games} 試合に切り詰めます(1試合分は実行しません)。"
        )
    if games <= 0:
        raise ValueError("--games must resolve to a positive even number (>= 2)")

    agent_a = load_agent(agent_a_name)
    agent_b = load_agent(agent_b_name)
    deck_a = read_deck_csv_file(deck_a_path)
    deck_b = read_deck_csv_file(deck_b_path)

    game_records: list[dict] = []
    error_reasons: Counter[str] = Counter()

    a_wins_total = 0
    b_wins_total = 0
    a_wins_first = 0  # A が player_index=0(先手扱い)だった試合でのA勝ち
    n_first = 0        # A が player_index=0 だった有効試合数
    a_wins_second = 0  # A が player_index=1(後手扱い)だった試合でのA勝ち
    n_second = 0
    turns_sum = 0
    steps_sum = 0
    valid_games = 0
    error_games = 0

    t_start = time.time()

    for i in range(games):
        seed = seed_start + i
        a_is_player0 = (i % 2 == 0)
        if a_is_player0:
            agent0, agent1 = agent_a, agent_b
            deck0, deck1 = deck_a, deck_b
            a_player_index = 0
        else:
            agent0, agent1 = agent_b, agent_a
            deck0, deck1 = deck_b, deck_a
            a_player_index = 1

        result: MatchResult = play_match(agent0, agent1, deck0, deck1, seed=seed)

        winner_agent: str | None
        if result.error is not None:
            error_games += 1
            error_reasons[result.error] += 1
            winner_agent = None
        else:
            valid_games += 1
            winner_agent = "A" if result.winner == a_player_index else "B"
            if winner_agent == "A":
                a_wins_total += 1
            else:
                b_wins_total += 1
            if a_is_player0:
                n_first += 1
                if winner_agent == "A":
                    a_wins_first += 1
            else:
                n_second += 1
                if winner_agent == "A":
                    a_wins_second += 1
            if result.turns is not None:
                turns_sum += result.turns
            steps_sum += result.steps

        game_records.append({
            "index": i,
            "seed": seed,
            "a_player_index": a_player_index,
            "winner_agent": winner_agent,
            "turns": result.turns,
            "steps": result.steps,
            "seconds": result.seconds,
            "error": result.error,
        })

        completed = i + 1
        if progress_every > 0 and (completed % progress_every == 0 or completed == games):
            elapsed = time.time() - t_start
            provisional_rate = a_wins_total / valid_games if valid_games else float("nan")
            log(
                f"[progress] {completed}/{games} games done "
                f"(A win rate so far: {provisional_rate:.3f} over {valid_games} valid games, "
                f"errors: {error_games}, elapsed: {elapsed:.1f}s)"
            )

    elapsed_total = time.time() - t_start
    error_rate = error_games / games if games else 0.0
    if error_rate > 0.05:
        log(
            f"[warn] error rate {error_rate:.1%} ({error_games}/{games}) exceeds the 5% threshold; "
            "win-rate estimates below exclude these games but treat them with caution."
        )

    avg_turns = turns_sum / valid_games if valid_games else None
    avg_steps = steps_sum / valid_games if valid_games else None

    summary = {
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "deck_a_path": str(resolve_deck_path(deck_a_path)),
        "deck_b_path": str(resolve_deck_path(deck_b_path)),
        "games_requested": requested_games,
        "games_run": games,
        "seed_start": seed_start,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": elapsed_total,
        "overall": {
            **_rate_stats(a_wins_total, valid_games),
            "b_wins": b_wins_total,
            "avg_turns": avg_turns,
            "avg_steps": avg_steps,
        },
        "by_turn_order": {
            "a_player0_first": _rate_stats(a_wins_first, n_first),
            "a_player1_second": _rate_stats(a_wins_second, n_second),
        },
        "errors": {
            "count": error_games,
            "rate": error_rate,
            "by_reason": dict(error_reasons),
        },
        "games": game_records,
    }
    return summary


def print_summary(summary: dict, log: Callable[[str], None] = print) -> None:
    overall = summary["overall"]
    by_order = summary["by_turn_order"]
    errors = summary["errors"]

    a = summary["agent_a"]
    b = summary["agent_b"]

    log(f"=== league result: {a} (A) vs {b} (B) ===")
    log(f"games requested/run: {summary['games_requested']}/{summary['games_run']}")
    log(f"elapsed: {summary['elapsed_seconds']:.1f}s")
    log("")
    if overall["games"]:
        lo, hi = overall["wilson_95ci"]
        log(
            f"overall: A {overall['wins']}/{overall['games']} "
            f"= {overall['win_rate']:.3f} (95% CI [{lo:.3f}, {hi:.3f}])"
        )
    else:
        log("overall: no valid games (all errored)")
    if overall["avg_turns"] is not None:
        log(f"avg turns: {overall['avg_turns']:.1f}, avg steps: {overall['avg_steps']:.1f}")
    log("")

    first = by_order["a_player0_first"]
    second = by_order["a_player1_second"]
    if first["games"]:
        lo, hi = first["wilson_95ci"]
        log(
            f"A as player0 (先手): {first['wins']}/{first['games']} "
            f"= {first['win_rate']:.3f} (95% CI [{lo:.3f}, {hi:.3f}])"
        )
    else:
        log("A as player0 (先手): no valid games")
    if second["games"]:
        lo, hi = second["wilson_95ci"]
        log(
            f"A as player1 (後手): {second['wins']}/{second['games']} "
            f"= {second['win_rate']:.3f} (95% CI [{lo:.3f}, {hi:.3f}])"
        )
    else:
        log("A as player1 (後手): no valid games")
    log("")

    log(f"errors: {errors['count']} ({errors['rate']:.1%})")
    for reason, count in sorted(errors["by_reason"].items(), key=lambda kv: -kv[1]):
        log(f"  - {reason}: {count}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_path = resolve_out_path(args.out, args.agent_a, args.agent_b)

    # deck.csv 等の相対パス依存(rule_based/ml_policy 双方が内部で使う)のため、
    # どのディレクトリから起動されても sample_submission/ を cwd にする。
    os.chdir(_SAMPLE_SUBMISSION_DIR)

    summary = run_league(
        agent_a_name=args.agent_a,
        agent_b_name=args.agent_b,
        games=args.games,
        deck_a_path=args.deck_a,
        deck_b_path=args.deck_b,
        seed_start=args.seed_start,
        progress_every=args.progress_every,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nresult written to: {out_path}")


if __name__ == "__main__":
    main()
