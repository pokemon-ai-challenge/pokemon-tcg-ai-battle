"""M1: ロールアウトの健全性検証。

M1a: 実盤面で torch policy の argmax 選択が pure-Python PolicyModel と一致(encode->torch->action の
     end-to-end パリティ)。
M1b: 素policy(overlay なし)ベースライン勝率。torch dragapult(argmax) vs pure-Python alakazam。
     これが RL で伸ばす出発点。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402
from rollout import TorchActor, make_pure_policy_agent, play_and_collect, _encode_decision  # noqa: E402
from torch_policy import TorchOptionPolicy  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
DRAGAPULT_W = WDIR / "policy_weights_dragapult_ex.json"


def read_deck(path: Path) -> list[int]:
    sys.path.insert(0, str(_ROOT / "league"))
    from run_league import read_deck_csv_file
    return read_deck_csv_file(str(path))


def m1a_behavioral_parity(deck_learn, deck_opp, n_games=4):
    """torch argmax == pure-Python argmax を実盤面で確認(dragapult 側を検査)。"""
    tp = TorchOptionPolicy.from_json(DRAGAPULT_W).float().eval()
    pm = PolicyModel(DRAGAPULT_W)
    opp = make_pure_policy_agent(None)  # production alakazam

    checked = 0
    mismatch = 0
    for g in range(n_games):
        learner_index = g % 2
        deck0, deck1 = (deck_learn, deck_opp) if learner_index == 0 else (deck_opp, deck_learn)
        obs_dict, sd = battle_start(deck0, deck1)
        if sd.errorType != 0:
            battle_finish(); continue
        steps = 0
        try:
            while True:
                obs = to_observation_class(obs_dict)
                if obs.current is None or obs.current.result != -1 or steps >= 3000:
                    break
                tp_i = obs.current.yourIndex
                if tp_i == learner_index:
                    enc = _encode_decision(obs.current, obs.select)
                    if enc is not None:
                        sf, of, ci = enc
                        with torch.no_grad():
                            t_idx = int(tp.option_scores(
                                torch.tensor(sf), torch.tensor(of), torch.tensor(ci)).argmax().item())
                        p_idx = pm.select_option(obs)
                        checked += 1
                        if t_idx != p_idx:
                            mismatch += 1
                        action = [p_idx if p_idx is not None else 0]
                    else:
                        action = opp(obs) if False else _fallback_action(obs, pm)
                else:
                    action = opp(obs)
                obs_dict = battle_select(action)
                steps += 1
        finally:
            battle_finish()
    print(f"M1a: 検査 {checked} 決定点, 不一致 {mismatch}")
    assert checked > 0, "検査対象の単一選択決定が0件"
    assert mismatch == 0, f"torch と pure-Python の argmax が {mismatch} 件不一致"
    print("M1a PASS: 実盤面で torch argmax == pure-Python argmax")


def _fallback_action(obs, pm):
    select = obs.select
    if select is None or not select.option:
        return []
    scores = pm.score_options(obs)
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))
    if not scores:
        return list(range(count))
    return sorted(range(n), key=lambda i: scores[i], reverse=True)[:count]


def m1b_baseline(deck_learn, deck_opp, n_games=40):
    """素policy dragapult(argmax) vs alakazam の勝率。"""
    tp = TorchOptionPolicy.from_json(DRAGAPULT_W).float().eval()
    actor = TorchActor(tp, sample=False)  # greedy
    opp = make_pure_policy_agent(None)
    wins = 0
    valid = 0
    errors = 0
    for g in range(n_games):
        learner_index = g % 2
        traj = play_and_collect(actor, opp, deck_learn, deck_opp, learner_index, seed=1000 + g)
        if traj.error is not None:
            errors += 1
            continue
        valid += 1
        wins += 1 if traj.reward >= 1.0 else 0
    wr = wins / valid if valid else float("nan")
    print(f"M1b: dragapult(素policy,argmax) vs alakazam  勝率 {wins}/{valid} = {wr:.3f}  (errors={errors})")
    return wr


def main():
    deck_dra = read_deck(DECKDIR / "dragapult_ex" / "01.csv")
    deck_ala = read_deck(DECKDIR / "alakazam" / "01.csv")
    m1a_behavioral_parity(deck_dra, deck_ala, n_games=4)
    m1b_baseline(deck_dra, deck_ala, n_games=40)
    print("M1 done")


if __name__ == "__main__":
    main()
