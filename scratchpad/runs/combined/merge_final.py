import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

LOSS_REASON_LABELS = {
    1: "prizes_taken",
    2: "deckout",
    3: "no_active_pokemon",
    4: "card_effect",
}

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

all_records = []
wg_totals = Counter()
for i in range(5):
    d = json.loads((HERE / f"shard{i}.json").read_text(encoding="utf-8"))
    all_records.extend(d["games"])
    wg_totals.update(d["wall_guard_stats"])

valid = [r for r in all_records if not r.get("error")]
errors = [r for r in all_records if r.get("error")]
n_win = sum(1 for r in valid if r.get("win_reason") is not None)
n_loss = sum(1 for r in valid if r.get("loss_reason") is not None)
n_draw = len(valid) - n_win - n_loss
lo, hi = wilson(n_win, len(valid))

total_attacks = sum(r.get("our_attacks", 0) for r in valid)
zero_attacks = sum(r.get("our_zero_damage_attacks", 0) for r in valid)
ex_wall_zero = sum(r.get("ex_vs_wall_zero_damage", 0) for r in valid)
prizes = [r["prizes_taken"] for r in valid if r.get("prizes_taken") is not None]
kanga_ko = [r.get("kangaskhan_ko_count", 0) for r in valid]
kanga_dist = Counter(("0" if k == 0 else "1" if k == 1 else "2+") for k in kanga_ko)
loss_reasons = Counter(LOSS_REASON_LABELS.get(r["loss_reason"], f"unknown({r['loss_reason']})")
                        for r in valid if r.get("loss_reason") is not None)
turns = [r["turns"] for r in valid if r.get("turns") is not None]

card_ids = set()
for r in valid:
    card_ids.update(r.get("board_dev", {}).keys())
board_dev = {}
NAMES = {"150": "hydrapple_ex", "710": "meganium", "93": "dipplin", "96": "ogerpon_ex"}
for cid in sorted(card_ids):
    vals = [r["board_dev"][cid] for r in valid if r.get("board_dev", {}).get(cid) is not None]
    board_dev[cid] = {
        "name": NAMES.get(cid, cid),
        "n_reached": len(vals),
        "n_games": len(valid),
        "reach_rate": len(vals) / len(valid) if valid else None,
        "median_turn": median(vals),
    }

summary = {
    "n_shards": 5,
    "games_run": len(all_records),
    "valid_games": len(valid),
    "errors": len(errors),
    "error_details": [r["error"] for r in errors],
    "wins": n_win, "losses": n_loss, "draws": n_draw,
    "win_rate": n_win / len(valid) if valid else None,
    "wilson_95ci": [lo, hi],
    "avg_turns": sum(turns) / len(turns) if turns else None,
    "our_attacks_total": total_attacks,
    "zero_damage_attacks": zero_attacks,
    "zero_damage_rate": zero_attacks / total_attacks if total_attacks else None,
    "ex_vs_wall_zero_damage": ex_wall_zero,
    "mean_prizes_taken": sum(prizes) / len(prizes) if prizes else None,
    "kangaskhan_ko_distribution": dict(kanga_dist),
    "loss_reason_breakdown": dict(loss_reasons),
    "board_dev": board_dev,
    "wall_guard_stats": dict(wg_totals),
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
(HERE / "merged_final.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
