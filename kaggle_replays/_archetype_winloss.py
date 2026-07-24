#!/usr/bin/env python3
"""指定した提出(submission ref)の実 Kaggle 対戦リプレイを、相手アーキタイプ別に
勝-負(W-L)で集計する。

- 対象エピソードは _submission_episode_map.json[ref]。
- 各リプレイで自チーム("MORIOKA Tsuoi")の player_index を特定し、rewards から勝敗を取る。
- 相手アーキタイプは production の rough_predictor.predict() に、こちら視点の最終盤面
  (最後に current が非Noneのステップ)を渡して deck_type を得る(公開カードからの推定なので
  終盤ほど確度が高い)。既存の submission 解析(_analyze_attackplan_submissions.py)と同じ
  リプレイ読み口を踏襲する throwaway 診断。

使い方:
  python _archetype_winloss.py --ref 54956037
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "sample_submission"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cg.api import to_observation_class  # noqa: E402
from ptcg_ai.opponent_modeling import rough_predictor as rp  # noqa: E402

OUR_TEAM = "MORIOKA Tsuoi"


def classify_opponent(data: dict, our_index: int) -> str:
    """全ステップでこちら視点の盤面を rough_predictor にかけ、最も証拠が多い
    (evidence_count 最大)予測の deck_type を採用する。相手のカードは進行に伴って
    公開が増えるため、単一の最終盤面より頑健(unknown を減らす)。"""
    steps = data["steps"]
    best_dt = "unknown"
    best_evidence = -1
    for k in range(len(steps)):
        obs = steps[k][our_index]["observation"]
        cur = obs.get("current")
        if cur is None:
            continue
        try:
            state = to_observation_class({"current": cur, "select": obs.get("select"), "logs": []}).current
            pred = rp.predict(state)
        except Exception:
            continue
        dt = pred.get("deck_type") or "unknown"
        ev = pred.get("evidence_count", 0) or 0
        if dt != "unknown" and ev > best_evidence:
            best_evidence = ev
            best_dt = dt
    return best_dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="54956037", help="submission ref")
    ap.add_argument("--map", default=str(_HERE / "_submission_episode_map.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
    if args.ref not in mapping:
        print(f"ref {args.ref} が map に無い(未fetch?)。map内: {list(mapping.keys())}")
        sys.exit(1)

    eids = mapping[args.ref]
    wl: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # arch -> [wins, losses]
    games = missing = 0
    per_game = []
    for eid in eids:
        path = _HERE / "replays" / f"episode-{eid}-replay.json"
        if not path.exists():
            missing += 1
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        tn = data["info"]["TeamNames"]
        if OUR_TEAM not in tn:
            continue
        our_index = tn.index(OUR_TEAM)
        reward = data["rewards"][our_index]
        won = reward is not None and reward > 0
        arch = classify_opponent(data, our_index)
        wl[arch][0 if won else 1] += 1
        games += 1
        per_game.append({"episode": eid, "opponent": arch, "won": won})

    total_w = sum(v[0] for v in wl.values())
    total_l = sum(v[1] for v in wl.values())
    print(f"=== submission {args.ref}: 相手アーキタイプ別 W-L ===")
    print(f"解析 {games} 試合 (未DL {missing})  総合 {total_w}-{total_l}"
          + (f" (勝率 {total_w/games:.3f})" if games else ""))
    print(f"{'相手アーキタイプ':24s}{'W-L':>8s}{'勝率':>8s}{'試合':>6s}")
    for arch, (w, l) in sorted(wl.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        n = w + l
        print(f"{arch:24s}{f'{w}-{l}':>8s}{(w/n if n else 0):>8.2f}{n:>6d}")

    result = {
        "ref": args.ref, "games": games, "missing": missing,
        "total": [total_w, total_l],
        "by_archetype": {a: {"win": v[0], "loss": v[1]} for a, v in wl.items()},
        "per_game": per_game,
    }
    out = args.out or str(_HERE / f"_archetype_winloss_{args.ref}.json")
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
