"""意思決定パイプライン アブレーション: 5構成を総当たりして勝率マトリクスを出す CLI。

`sample_submission/docs/plans/decision-pipeline/design-and-implementation-plan.md` の
「検証(アブレーション)」に対応。spec の5構成:

  1. rule_based           : ルールベースのみ(既存エージェント)
  2. policy_only          : Policy top1 のみ(abl_2_policy_only)
  3. policy_eval1         : Policy + 手作り評価, 探索なし1手読み(abl_3_policy_eval1)
  4. policy_searchN1      : Policy + 探索 N=1, belief なし(abl_4_policy_searchN1)
  5. full                 : フル構成 N=8, estimated belief(abl_5_full)

``run_match.play_match`` にも ``run_league`` にも手を加えず、``run_league`` の
``build_agent``(config 名で ml_policy に config を注入)/``_play_game``(先手後手バイアス
除去)/``wilson_interval`` をそのまま流用する。各(順序なし)ペアで ``--games`` 試合を回し、
「行の構成が列の構成に勝った率」の NxN マトリクスと Wilson 95% CI を JSON と表で出力する。

制約(局所計測のみに影響、本番提出には無関係):
    2つの ml_policy 構成が同一プロセスで対戦する場合、`ml_policy_agent` のモジュール
    グローバル(pipeline 動的時間予算のカウンタ)を共有する。時間予算は実経過時間 remaining_ms
    から per-move を算出し自己補正するため予算超過(=反則負け)は起きないが、abl_5 同士の
    per-move 予算配分の精度だけは理想化されない。本番では対戦相手は別プロセスの別提出物
    なのでこの共有は発生しない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_LEAGUE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _LEAGUE_DIR.parent
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"

for _candidate in (str(_LEAGUE_DIR), str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from run_league import (  # noqa: E402
    build_agent,
    read_deck_csv_file,
    wilson_interval,
    _play_game,
)

# (表示名, AGENT_REGISTRY 名, 注入する config 名 or None)
DEFAULT_COMPETITORS: list[tuple[str, str, str | None]] = [
    ("rule_based", "rule_based", None),
    ("policy_only", "ml_policy", "abl_2_policy_only"),
    ("policy_eval1", "ml_policy", "abl_3_policy_eval1"),
    ("policy_searchN1", "ml_policy", "abl_4_policy_searchN1"),
    ("full", "ml_policy", "abl_5_full"),
]


def _build(label_name_config: tuple[str, str, str | None]):
    _label, name, config_name = label_name_config
    # rule_based は config を取らない。ml_policy は config 名を base として注入する。
    return build_agent(name, None, config_name or "abl_2_policy_only")


def run_pair(comp_a, comp_b, deck, games: int, seed_start: int, progress_every: int) -> dict:
    """1ペアを games 試合(先手後手を交互に入れ替え)実行し、A視点の勝敗を集計する。"""
    agent_a = _build(comp_a)
    agent_b = _build(comp_b)
    a_wins = 0
    b_wins = 0
    errors = 0
    for i in range(games):
        rec = _play_game(agent_a, agent_b, deck, deck, index=i, seed=seed_start + i)
        if rec["error"] is not None:
            errors += 1
        elif rec["winner_agent"] == "A":
            a_wins += 1
        else:
            b_wins += 1
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {comp_a[0]} vs {comp_b[0]}: {i + 1}/{games}", file=sys.stderr)
    valid = a_wins + b_wins
    lo, hi = wilson_interval(a_wins, valid)
    return {
        "a": comp_a[0],
        "b": comp_b[0],
        "games": games,
        "valid": valid,
        "errors": errors,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "a_win_rate": (a_wins / valid) if valid else None,
        "a_win_rate_wilson95": [lo, hi],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100, help="各ペアの試合数(default: 100)")
    parser.add_argument("--deck", default=None, help="両者が使うデッキCSV(default: sample_submission/deck.csv)")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--out", default=None, help="結果JSONの出力先")
    parser.add_argument(
        "--only", default=None,
        help="カンマ区切りで対象構成を絞る(表示名。例: policy_only,full)。省略時は全5構成。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # ml_policy/rule_based は deck.csv 等を cwd 相対で参照するため cwd をそろえる。
    os.chdir(_SAMPLE_SUBMISSION_DIR)

    competitors = DEFAULT_COMPETITORS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        competitors = [c for c in competitors if c[0] in wanted]
        if len(competitors) < 2:
            print("error: --only は2構成以上を指定してください", file=sys.stderr)
            return 2

    deck = read_deck_csv_file(args.deck)
    labels = [c[0] for c in competitors]

    pairs = []
    for i in range(len(competitors)):
        for j in range(i + 1, len(competitors)):
            print(f"pair: {labels[i]} vs {labels[j]}", file=sys.stderr)
            pairs.append(run_pair(
                competitors[i], competitors[j], deck,
                args.games, args.seed_start, args.progress_every,
            ))

    # 勝率マトリクス matrix[row][col] = row が col に勝った率。
    matrix: dict[str, dict[str, float | None]] = {r: {c: None for c in labels} for r in labels}
    for p in pairs:
        rate = p["a_win_rate"]
        matrix[p["a"]][p["b"]] = rate
        matrix[p["b"]][p["a"]] = (1.0 - rate) if rate is not None else None

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "games_per_pair": args.games,
        "competitors": labels,
        "win_rate_matrix": matrix,
        "pairs": pairs,
    }

    out = args.out or str(_LEAGUE_DIR / "results" / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = _ROOT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_matrix(labels, matrix)
    print(f"\nsaved: {out_path}", file=sys.stderr)
    return 0


def _print_matrix(labels: list[str], matrix: dict) -> None:
    width = max(len(l) for l in labels) + 2
    header = " " * width + "".join(f"{l[:10]:>12}" for l in labels)
    print("\n勝率マトリクス(行が列に勝った率):")
    print(header)
    for r in labels:
        cells = ""
        for c in labels:
            v = matrix[r][c]
            cells += f"{'  -  ':>12}" if (r == c or v is None) else f"{v:>12.3f}"
        print(f"{r:<{width}}{cells}")


if __name__ == "__main__":
    raise SystemExit(main())
