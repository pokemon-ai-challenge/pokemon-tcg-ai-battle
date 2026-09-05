"""カプ・ブルルが実戦で「使えているか」を計測する診断。

+5.0pt(対crustle)が (a)ブルルが殴って稼いだ分 なのか
(b)ポケモンが5枚になった副次効果(ベンチ0事故の低減など)なのかを切り分ける。

計測: ブルルを場に出した試合の割合 / 最大到達エネ数 / ウッドハンマー使用回数 /
      壁(イワパレス345, いしずえのめんex117)を実際にKOした回数。
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
from cg.api import to_observation_class  # noqa: E402

BULU = 920
WALLS = {345: "イワパレス", 117: "いしずえのめんex"}
WOOD_HAMMER = 1326   # カプ・ブルルのワザ「ウッドハンマー」(all_attack() で実ID確認済み)
_LOG_ATTACK = 15     # LogType.ATTACK(attackId を持つ)
STATS = Counter()
PER_GAME: list[dict] = []


def _wrap(agent, own_index: int):
    """own側の各決定で盤面を覗き、ブルルの状態を記録する(意思決定は変更しない)。"""
    state = {"bulu_in_play": False, "bulu_max_energy": 0, "wood_hammer": 0,
             "bulu_active": False}

    def probe(obs):
        try:
            cur = obs.current
            if cur is not None and cur.yourIndex == own_index:
                me = cur.players[own_index]
                for pk in (me.active or []):
                    if pk is not None and getattr(pk, "id", None) == BULU:
                        state["bulu_active"] = True
                for pk in list(me.active or []) + list(me.bench or []):
                    if pk is None:
                        continue
                    if getattr(pk, "id", None) == BULU:
                        state["bulu_in_play"] = True
                        n = len(getattr(pk, "energies", []) or [])
                        state["bulu_max_energy"] = max(state["bulu_max_energy"], n)
            # `Log` に text フィールドは存在しない(cg/api.py の dataclass 参照)。
            # 以前はここで `getattr(lg, "text", "")` を走査していたため、この指標は
            # **構造的に常に0**で、「ウッドハンマー0.0回/試合」は計測不能を意味していた。
            # LogType.ATTACK(15)の attackId で数えるのが正しい。ブルルを持つのは自分だけ
            # なので attackId だけで一意に識別できる。
            for lg in (obs.logs or []):
                if int(getattr(lg, "type", -1)) == _LOG_ATTACK \
                        and getattr(lg, "attackId", None) == WOOD_HAMMER:
                    state["wood_hammer"] += 1
        except Exception:  # noqa: BLE001 - 観測失敗で対戦を止めない
            pass
        return agent(obs)

    probe._state = state  # type: ignore[attr-defined]
    return probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default=str(_ROOT / "kaggle_replays/deck_search/candidates_wall/bulu.csv"))
    ap.add_argument("--opp", default=str(_ROOT / "kaggle_replays/meta_analysis/archetype_decks_g2/crustle/01.csv"))
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--config", default="abl_5_full_og_r13")
    ap.add_argument("--out", default=str(_ROOT / "kaggle_replays/_diag_bulu_usage.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
    import agents  # noqa: E402
    agents.ensure_production_cwd()

    own_deck = [int(l.strip()) for l in open(args.deck, encoding="utf-8") if l.strip().isdigit()]
    opp_deck = [int(l.strip()) for l in open(args.opp, encoding="utf-8") if l.strip().isdigit()]
    wdir = _ROOT / "sample_submission/ptcg_ai/learning"
    own_cfg = agents.load_config_copy(args.config)
    own_cfg["policy_weights_path"] = str(wdir / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json")
    opp_cfg = agents.load_config_copy("abl_5_full")
    opp_cfg["policy_weights_path"] = str(wdir / "policy_weights_crustle_g2.json")

    for g in range(args.games):
        own_first = g % 2 == 0
        a_own = _wrap(agents.make_ml_policy_agent(dict(own_cfg)), 0 if own_first else 1)
        a_opp = agents.make_ml_policy_agent(dict(opp_cfg))
        r = (run_match.play_match(a_own, a_opp, own_deck, opp_deck, seed=1300000 + g) if own_first
             else run_match.play_match(a_opp, a_own, opp_deck, own_deck, seed=1300000 + g))
        st = a_own._state  # type: ignore[attr-defined]
        won = (r.winner == (0 if own_first else 1))
        PER_GAME.append({"game": g, "won": won, **st})
        STATS["games"] += 1
        STATS["wins"] += 1 if won else 0
        STATS["bulu_in_play"] += 1 if st["bulu_in_play"] else 0
        STATS["bulu_active"] += 1 if st["bulu_active"] else 0
        STATS["bulu_e4plus"] += 1 if st["bulu_max_energy"] >= 4 else 0
        STATS["bulu_e0"] += 1 if st["bulu_max_energy"] == 0 else 0
        STATS["wood_hammer_total"] += st["wood_hammer"]
        STATS["wood_hammer_games"] += 1 if st["wood_hammer"] > 0 else 0
        if r.error:
            STATS["errors"] += 1

    n = max(1, STATS["games"])
    summary = {
        "games": STATS["games"], "win_rate": round(STATS["wins"] / n, 4),
        "bulu_in_play_rate": round(STATS["bulu_in_play"] / n, 4),
        "bulu_became_active_rate": round(STATS["bulu_active"] / n, 4),
        "bulu_reached_4energy_rate": round(STATS["bulu_e4plus"] / n, 4),
        "bulu_zero_energy_rate": round(STATS["bulu_e0"] / n, 4),
        "wood_hammer_per_game": round(STATS["wood_hammer_total"] / n, 3),
        "wood_hammer_game_rate": round(STATS["wood_hammer_games"] / n, 4),
        "errors": STATS["errors"],
        "max_energy_hist": dict(Counter(p["bulu_max_energy"] for p in PER_GAME)),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "per_game": PER_GAME},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
