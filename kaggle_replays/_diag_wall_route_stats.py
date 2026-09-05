"""r14 `wall_attacker_route` の発火統計を実対戦で取る診断(原因特定用)。

`_diag_bulu_usage.py` と同じ対戦設定(bulu.csv vs crustle/01.csv)で走らせ、
自分側エージェントの `ml_policy_agent.get_guard_stats()` のうち wall_attacker_* を集計する。
「4エネには届いたのに撃たない」ときに、どの条件で弾かれているかを切り分けるために使う。
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT / "league", _ROOT / "sample_submission"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_match  # noqa: E402

BULU = 920
WALLS = (345, 117)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default=str(_ROOT / "kaggle_replays/deck_search/candidates_wall/bulu.csv"))
    ap.add_argument("--opp", default=str(_ROOT / "kaggle_replays/meta_analysis/archetype_decks_g2/crustle/01.csv"))
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--config", default="abl_5_full_og_r14")
    ap.add_argument("--out", default=str(_ROOT / "kaggle_replays/_diag_wall_route_stats.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
    import agents  # noqa: E402
    agents.ensure_production_cwd()
    from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402

    own_deck = [int(l.strip()) for l in open(args.deck, encoding="utf-8") if l.strip().isdigit()]
    opp_deck = [int(l.strip()) for l in open(args.opp, encoding="utf-8") if l.strip().isdigit()]
    wdir = _ROOT / "sample_submission/ptcg_ai/learning"
    own_cfg = agents.load_config_copy(args.config)
    own_cfg["policy_weights_path"] = str(wdir / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json")
    opp_cfg = agents.load_config_copy("abl_5_full")
    opp_cfg["policy_weights_path"] = str(wdir / "policy_weights_crustle_g2.json")

    # 「ブルルがバトル場に居た decision 数」「そのとき ATTACK 選択肢が出ていた数」も数える。
    obs_stats = Counter()

    _TYPE_NAME = {7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD",
                  12: "RETREAT", 13: "ATTACK", 14: "END", 3: "CARD"}

    def _wrap(agent, own_index: int):
        def probe(obs):
            classes: list[str] = []
            sel = obs.select
            try:
                cur = obs.current
                if cur is not None and sel is not None and cur.yourIndex == own_index:
                    me = cur.players[own_index]
                    act = (me.active or [None])[0]
                    opp = cur.players[1 - own_index]
                    opp_act = (opp.active or [None])[0]
                    wall_active = opp_act is not None and int(opp_act.id) in WALLS
                    wall_field = wall_active or any(
                        p is not None and int(p.id) in WALLS for p in (opp.bench or []))
                    bulu_bench = [p for p in (me.bench or [])
                                  if p is not None and int(p.id) == BULU]
                    if wall_field:
                        obs_stats["decision_wall_on_field"] += 1
                    if wall_active:
                        obs_stats["decision_wall_active"] += 1
                    if act is not None and int(act.id) == BULU:
                        obs_stats["decision_bulu_active"] += 1
                        if any(o.type == 13 for o in sel.option):
                            obs_stats["decision_bulu_active_can_attack"] += 1
                            classes.append("bulu_active_can_attack")
                    if bulu_bench and wall_active:
                        obs_stats["decision_bulu_bench_wall_active"] += 1
                        if max(len(p.energies or []) for p in bulu_bench) >= 4:
                            obs_stats["decision_bulu_bench_ready_wall_active"] += 1
                            if any(o.type == 12 for o in sel.option):
                                obs_stats["decision_bulu_bench_ready_retreat_legal"] += 1
                                classes.append("bulu_ready_retreat_legal")
            except Exception:  # noqa: BLE001
                pass
            action = agent(obs)
            # 実際に選ばれた手の種類(=どの条件で (b1)/(c) が弾かれたかの直接証拠)。
            try:
                if classes and action and sel is not None:
                    name = _TYPE_NAME.get(int(sel.option[action[0]].type), "OTHER")
                    for cls in classes:
                        obs_stats[f"chosen__{cls}__{name}"] += 1
            except Exception:  # noqa: BLE001
                pass
            return action
        return probe

    ml_policy_agent.reset_guard_stats()
    wins = 0
    for g in range(args.games):
        own_first = g % 2 == 0
        a_own = _wrap(agents.make_ml_policy_agent(dict(own_cfg)), 0 if own_first else 1)
        a_opp = agents.make_ml_policy_agent(dict(opp_cfg))
        r = (run_match.play_match(a_own, a_opp, own_deck, opp_deck, seed=1300000 + g) if own_first
             else run_match.play_match(a_opp, a_own, opp_deck, own_deck, seed=1300000 + g))
        wins += 1 if r.winner == (0 if own_first else 1) else 0

    stats = {k: v for k, v in ml_policy_agent.get_guard_stats().items()
             if k.startswith("wall_attacker")}
    out = {"config": args.config, "games": args.games, "wins": wins,
           "guard_stats": stats, "obs_stats": dict(obs_stats)}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
