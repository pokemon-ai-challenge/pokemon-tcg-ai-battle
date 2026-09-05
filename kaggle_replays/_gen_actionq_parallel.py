"""Phase9B: 層化 MAIN データ生成の並列ドライバ(worker=別プロセス)。

`_gen_actionq_dataset_v2.py` を W 個のサブプロセスとして起動し、shard を結合する。
worker ごとに game index を stride で割り当てるので **試合の重複が無く**、
`split_of_game` の割当も worker 間で一貫する(= split leak が起きない)。

ProcessPoolExecutor を使わずサブプロセスにしているのは、生成側が
`ml_policy_agent._select_action` を monkeypatch し、モジュール global に状態を持つため
(pickle 経由で持ち回るより、プロセスを分けた方が安全で状態も混ざらない)。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-games", type=int, default=200)
    ap.add_argument("--per-game-cap", type=int, default=12)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    args = ap.parse_args()

    t0 = time.perf_counter()
    procs = []
    for w in range(args.workers):
        cmd = [sys.executable, str(_HERE / "_gen_actionq_dataset_v2.py"),
               "--target", str(args.target), "--max-games", str(args.max_games),
               "--per-game-cap", str(args.per_game_cap),
               "--opponents", args.opponents,
               "--tag", f"{args.tag}_w{w}",
               "--worker-id", str(w), "--num-workers", str(args.workers)]
        log = open(_HERE / f"_actionq_{args.tag}_w{w}.log", "w", encoding="utf-8")
        procs.append((w, subprocess.Popen(cmd, stdout=log, stderr=log), log))
        print(f"[launch] worker {w}", file=sys.stderr, flush=True)

    for w, p, log in procs:
        rc = p.wait()
        log.close()
        print(f"[done] worker {w} rc={rc}", file=sys.stderr, flush=True)

    # --- shard 結合 ---
    rows: list[dict] = []
    seen_gid: set[str] = set()
    dup = 0
    for w in range(args.workers):
        shard = _HERE / f"_actionq_{args.tag}_w{w}.jsonl.gz"
        if not shard.exists():
            print(f"[warn] shard missing: {shard}", file=sys.stderr)
            continue
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["group_id"] in seen_gid:
                    dup += 1
                    continue
                seen_gid.add(r["group_id"])
                rows.append(r)

    out = _HERE / f"_actionq_{args.tag}.jsonl.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 監査(§9) ---
    by_split = Counter(r["split"] for r in rows)
    games_per_split: dict[int, set] = {}
    for r in rows:
        games_per_split.setdefault(r["split"], set()).add(r["game"])
    leak = sum(1 for g in {r["game"] for r in rows}
               if len({r["split"] for r in rows if r["game"] == g}) > 1)
    test_rows = [r for r in rows if r["split"] == 2]
    summary = {
        "tag": args.tag, "workers": args.workers, "K": 6,
        "budget_regime": "research_accuracy",
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "n_groups": len(rows), "duplicate_group_ids_dropped": dup,
        "n_candidates": sum(len(r["candidates"]) for r in rows),
        "matches": len({r["game"] for r in rows}),
        "split_counts": {"train": by_split[0], "val": by_split[1], "test": by_split[2]},
        "games_per_split": {k: len(v) for k, v in sorted(games_per_split.items())},
        "SPLIT_LEAK_games_in_multiple_splits": leak,
        "turn_band": dict(Counter(r["turn_band"] for r in rows)),
        "turn_band_frac": {k: round(v / max(1, len(rows)), 4)
                           for k, v in Counter(r["turn_band"] for r in rows).items()},
        "cand_band": dict(Counter(r["cand_band"] for r in rows)),
        "cand_band_frac": {k: round(v / max(1, len(rows)), 4)
                           for k, v in Counter(r["cand_band"] for r in rows).items()},
        "archetype": dict(Counter(r["opponent_archetype"] for r in rows)),
        "archetype_frac": {k: round(v / max(1, len(rows)), 4) for k, v in
                           Counter(r["opponent_archetype"] for r in rows).items()},
        "option_type_top": dict(Counter(c["option_type"] for r in rows
                                        for c in r["candidates"]).most_common(12)),
        "mean_cands": round(statistics.mean([len(r["candidates"]) for r in rows]), 2)
        if rows else None,
        "outcome_rate": round(statistics.mean([r["outcome"] for r in rows]), 4)
        if rows else None,
        "test_groups_with_teacher_B": sum(
            1 for r in test_rows if r["candidates"][0].get("teacher_B") is not None),
        "test_groups": len(test_rows),
        "dataset_sha": hashlib.sha256(out.read_bytes()).hexdigest()[:16],
        "split_sha": hashlib.sha256(
            "".join(f"{r['group_id']}:{r['split']};" for r in rows).encode()).hexdigest()[:16],
        "out": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (_HERE / f"_actionq_{args.tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
