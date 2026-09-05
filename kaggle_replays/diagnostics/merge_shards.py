"""Merge 5 diag_crustle_shard.py shard JSON outputs into one 100-game summary.

Usage: python merge_shards.py runs/baseline_current/shard0.json ... runs/baseline_current/shard4.json
"""
import json
import sys
from collections import Counter


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    m = len(s)
    return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * ((phat * (1 - phat) + z2 / (4 * n)) / n) ** 0.5
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def main():
    paths = sys.argv[1:]
    all_records = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        all_records.extend(d["games"])

    valid = [r for r in all_records if not r.get("error")]
    errors = [r for r in all_records if r.get("error")]
    n_win = sum(1 for r in valid if r.get("win_reason") is not None)
    n_loss = sum(1 for r in valid if r.get("loss_reason") is not None)
    n_draw = len(valid) - n_win - n_loss
    lo, hi = wilson(n_win, len(valid))

    total_attacks = sum(r.get("our_attacks", 0) for r in valid)
    zero_attacks = sum(r.get("our_zero_damage_attacks", 0) for r in valid)
    turns = [r["turns"] for r in valid if r.get("turns") is not None]
    kanga_ko = [r.get("kangaskhan_ko_count", 0) for r in valid]
    kanga_dist = Counter(("0" if k == 0 else "1" if k == 1 else "2+") for k in kanga_ko)
    loss_reasons = Counter(r["loss_reason"] for r in valid if r.get("loss_reason") is not None)

    card_ids = set()
    for r in valid:
        card_ids.update(r.get("board_dev", {}).keys())
    board_dev = {}
    for cid in sorted(card_ids):
        vals = [r["board_dev"][cid] for r in valid if r.get("board_dev", {}).get(cid) is not None]
        board_dev[cid] = {
            "n_reached": len(vals),
            "n_games": len(valid),
            "reach_rate": len(vals) / len(valid) if valid else None,
            "median_turn": median(vals),
        }

    summary = {
        "n_shards": len(paths),
        "games_run": len(all_records),
        "valid_games": len(valid),
        "errors": len(errors),
        "error_details": [r["error"] for r in errors],
        "wins": n_win, "losses": n_loss, "draws": n_draw,
        "win_rate": n_win / len(valid) if valid else None,
        "wilson_95ci": [lo, hi],
        "avg_turns": sum(turns) / len(turns) if turns else None,
        "zero_damage_rate": zero_attacks / total_attacks if total_attacks else None,
        "kangaskhan_ko_distribution": dict(kanga_dist),
        "loss_reason_breakdown": dict(loss_reasons),
        "board_dev": board_dev,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
