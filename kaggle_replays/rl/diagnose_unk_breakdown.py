"""option_unk_rateの内訳を切り分ける: card_id==0(識別なし、by design)と、
card_id!=0だがvocabに無い(真のvocab欠落)を分けて数える。"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TEACHER = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_RUN_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1"
_RUN_JSON = _RUN_DIR / "run.json"


def main(opp_id: str, n_games: int = 20, seed0: int = 700000) -> None:
    import json
    import common as C
    import t1_live_agent as la
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.learning.card_vocab import load_vocab, UNK_INDEX
    from run_league import read_deck_csv_file

    vocab = load_vocab()
    run_cfg = json.loads(_RUN_JSON.read_text(encoding="utf-8"))
    opp_cfg = next(o for o in run_cfg["opponents"] if o["id"] == opp_id)
    weights = C.resolve_opponent_weights(_RUN_DIR, opp_cfg.get("weights"))
    deck_o_path = C.resolve_deck(opp_cfg.get("deck") or run_cfg["opponent_deck"])
    pm_opp = PolicyModel(str(weights) if weights else None)
    deck_o = read_deck_csv_file(str(deck_o_path))
    deck_t1 = read_deck_csv_file(str(
        _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"))
    pm_teacher = PolicyModel(str(_TEACHER))

    n_zero = n_nonzero_unk = n_nonzero_known = 0
    unresolved_ids = set()

    for g in range(n_games):
        t1_index = g % 2
        deck0, deck1 = (deck_t1, deck_o) if t1_index == 0 else (deck_o, deck_t1)
        obs_dict, start_data = battle_start(deck0, deck1)
        if start_data.errorType != 0:
            continue
        n = 0
        try:
            while n < 3000:
                obs = to_observation_class(obs_dict)
                cur = obs.current
                if cur is None or cur.result != -1:
                    break
                select = obs.select
                if select is not None and select.option:
                    if cur.yourIndex == t1_index and select.maxCount == 1:
                        arrays = la.build_single_decision_arrays(pm_teacher, cur, select, "fuudin_v4")
                        for cid in arrays["option_card_ids"]:
                            cid = int(cid)
                            if cid == 0:
                                n_zero += 1
                            elif vocab.index_of(cid) == UNK_INDEX:
                                n_nonzero_unk += 1
                                unresolved_ids.add(cid)
                            else:
                                n_nonzero_known += 1
                        sc = pm_teacher.score_options_from_state(cur, select)
                        action = [int(max(range(len(select.option)), key=lambda i: sc[i]))]
                    elif cur.yourIndex == t1_index:
                        sc = pm_teacher.score_options_from_state(cur, select)
                        nn = len(select.option)
                        count = max(select.minCount, min(select.maxCount, nn))
                        action = sorted(range(nn), key=lambda i: sc[i], reverse=True)[:count]
                    else:
                        sc = pm_opp.score_options_from_state(cur, select)
                        nn = len(select.option)
                        count = max(select.minCount, min(select.maxCount, nn))
                        action = sorted(range(nn), key=lambda i: sc[i], reverse=True)[:count]
                else:
                    action = []
                obs_dict = battle_select(action)
                n += 1
        finally:
            battle_finish()

    total = n_zero + n_nonzero_unk + n_nonzero_known
    print(f"{opp_id}: total_options={total}")
    print(f"  card_id==0(識別なし、by design): {n_zero} ({n_zero/total:.1%})")
    print(f"  card_id!=0だがvocabに無い(真のvocab欠落): {n_nonzero_unk} ({n_nonzero_unk/total:.1%})")
    print(f"  vocabに解決できた: {n_nonzero_known} ({n_nonzero_known/total:.1%})")
    if unresolved_ids:
        print(f"  未解決id例: {sorted(unresolved_ids)[:20]}")


if __name__ == "__main__":
    for opp in ("crustle", "rocket_mewtwo_ex"):
        main(opp, n_games=20, seed0=700000 if opp == "crustle" else 800000)
