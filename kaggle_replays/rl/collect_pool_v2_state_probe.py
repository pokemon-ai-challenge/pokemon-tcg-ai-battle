"""pool_v2_validationの6相手との対局で実際に出現した決定点(状態)を収集し、
milestone_2000/2500/3000の3 checkpointを同じ状態へ入力して比較するための
probe setを作る(item3: 固定probe以外の状態でも一致率100%が保たれるかの検証)。

行動選択はmilestone_2000(既存のchampion, 現在はtransformer_ppo_reference)のargmaxで進める
(状態を集めるためだけで、収集自体は既存のt1_eval.play_one_gameと同じ経路)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CHAMPION_CKPT = _ROOT / "kaggle_replays" / "rl" / "runs" / "champions" / "milestone_2000" / "checkpoint_2000.pt"
_TEACHER = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_LEARNER_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"
_MANIFEST = _HERE / "pool_v2_validation_manifest.json"
_OUT = _HERE / "runs" / "pool_v2_validation_results" / "state_probe.npz"


def main(games_per_opponent: int = 6) -> None:
    import t1_live_agent as la
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    model, vocab, profile_name, _ = la.load_t1_for_inference(_CHAMPION_CKPT, device="cpu")
    t1_state_pm = PolicyModel(str(_TEACHER))
    deck_t1 = read_deck_csv_file(str(_LEARNER_DECK))

    decisions = []
    seed = 9700000
    for opp in manifest["opponents"]:
        pm = PolicyModel(opp["weights"])
        deck_o = read_deck_csv_file(str(_ROOT / opp["deck"]))
        for g in range(games_per_opponent):
            t1_index = g % 2
            deck0 = deck_t1 if t1_index == 0 else deck_o
            deck1 = deck_t1 if t1_index == 1 else deck_o
            obs_dict, start_data = battle_start(deck0, deck1)
            if start_data.errorType != 0:
                continue
            n = 0
            try:
                while n < 300:
                    obs = to_observation_class(obs_dict)
                    cur = obs.current
                    if cur is None or cur.result != -1:
                        break
                    select = obs.select
                    if select is not None and select.option:
                        if cur.yourIndex == t1_index and select.maxCount == 1:
                            arrays, scores = _forward(model, vocab, t1_state_pm, cur, select,
                                                      profile_name)
                            decisions.append(arrays)
                            action = [int(np.argmax(scores))]
                        elif cur.yourIndex == t1_index:
                            sc = t1_state_pm.score_options_from_state(cur, select)
                            nn = len(select.option)
                            count = max(select.minCount, min(select.maxCount, nn))
                            action = sorted(range(nn), key=lambda i: sc[i], reverse=True)[:count]
                        else:
                            sc = pm.score_options_from_state(cur, select)
                            nn = len(select.option)
                            count = max(select.minCount, min(select.maxCount, nn))
                            action = sorted(range(nn), key=lambda i: sc[i], reverse=True)[:count]
                    else:
                        action = []
                    obs_dict = battle_select(action)
                    n += 1
            finally:
                battle_finish()
            seed += 1

    print(f"収集した状態(決定点)数: {len(decisions)}")
    import t1_rollout as tr
    arrays = tr.concat_single_decision_arrays(decisions)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_OUT, **arrays)
    print(f"保存: {_OUT}")


def _forward(model, vocab, t1_state_pm, cur, select, profile_name):
    import t1_live_agent as la
    import token_batch as tb
    import torch
    arrays = la.build_single_decision_arrays(t1_state_pm, cur, select, profile_name)
    batch = tb.build_batch(arrays, vocab)
    inputs = la._to_tensors(batch, "cpu")
    model.eval()
    with torch.no_grad():
        scores = model(**inputs)[0]
    n_opt = int(arrays["counts"][0])
    return arrays, scores[:n_opt].numpy()


if __name__ == "__main__":
    main()
