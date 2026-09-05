"""Phase19.15-R §25: candidate Policy と climb の Action disagreement(select type 別)。

production(abl_5_full + climb)を回し、到達した各 decision で **両 Policy のスコアを同一
observation 上で** 計算して argmax を比べる。ゲームを進めるのは climb 側(baseline 軌跡上の drift)。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"


def _abspath(p):
    """`ensure_production_cwd()` で cwd が sample_submission へ移るため、相対パスは
    repo root 基準で絶対化してから渡す(渡さないと PolicyModel が黙って未ロードになる)。"""
    q = Path(p)
    return str(q if q.is_absolute() else (_ROOT / q).resolve())
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import SelectType  # noqa: E402
from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: E402

_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
FIELD = ["mega_lucario_ex", "archaludon_ex", "crustle", "dragapult_ex",
         "marnie_grimmsnarl_ex", "rocket_mewtwo_ex", "shirona_garchomp_ex"]
TYPE = {0: "MAIN", 1: "CARD", 2: "ATTACHED_CARD", 3: "CARD_OR_ATTACHED", 4: "ENERGY",
        5: "SKILL", 6: "ATTACK", 7: "EVOLVE", 8: "COUNT", 9: "YES_NO",
        10: "SPECIAL_CONDITION"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(_WDIR / "policy_weights_alakazam_rl_climb.json"))
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--decisions", type=int, default=1200)
    ap.add_argument("--max-games", type=int, default=60)
    ap.add_argument("--tag", default="d1")
    args = ap.parse_args()

    args.baseline, args.candidate = _abspath(args.baseline), _abspath(args.candidate)
    a = PolicyModel(args.baseline)
    b = PolicyModel(args.candidate)
    if not (a.is_ready and b.is_ready):
        raise SystemExit("policy not ready")
    cfg = agents.load_config_copy("abl_5_full")
    cfg["policy_weights_path"] = args.baseline
    climb = agents.make_ml_policy_agent(copy.deepcopy(cfg))
    D = {"n": 0}
    tot: Counter = Counter()
    dis: Counter = Counter()
    maxdiff = [0.0]

    def probe(obs):
        act = climb(obs)
        if obs.select is None or not obs.select.option or D["n"] >= args.decisions:
            return act
        try:
            f = MA._model_hidden_state_factory(obs, cfg)
            dl = time.perf_counter() + 5.0
            sa, sb = a.score_options(obs, f, dl), b.score_options(obs, f, dl)
        except Exception:                                        # noqa: BLE001
            return act
        if not sa or not sb or len(sa) != len(sb):
            return act
        D["n"] += 1
        k = TYPE.get(int(obs.select.type), str(obs.select.type))
        tot[k] += 1
        tot["ALL"] += 1
        maxdiff[0] = max(maxdiff[0], max(abs(x - y) for x, y in zip(sa, sb)))
        if sa.index(max(sa)) != sb.index(max(sb)):
            dis[k] += 1
            dis["ALL"] += 1
        return act

    deck = runner.load_deck(_SUB / "deck.csv")
    for gi in range(args.max_games):
        if D["n"] >= args.decisions:
            break
        arch = FIELD[gi % len(FIELD)]
        co = agents.load_config_copy("abl_5_full")
        co["policy_weights_path"] = str(_WDIR / "policy_weights_{}.json".format(arch))
        opp = agents.make_ml_policy_agent(co)
        do = runner.load_deck(_DECKDIR / arch / "01.csv")
        (runner.play_game(probe, opp, list(deck), do) if gi % 2 == 0
         else runner.play_game(opp, probe, do, list(deck)))

    rep = {"tag": args.tag, "candidate": args.candidate, "decisions": D["n"],
           "max_score_diff": round(maxdiff[0], 5),
           "disagreement": {k: {"n": tot[k], "disagree": dis[k],
                                "rate": round(dis[k] / tot[k], 4)}
                            for k in sorted(tot, key=lambda x: -tot[x])}}
    (_HERE / "_p1915drift_{}.json".format(args.tag)).write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
