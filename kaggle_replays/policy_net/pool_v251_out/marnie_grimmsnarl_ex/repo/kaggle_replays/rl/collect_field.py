"""field-sampling 並列収集: learner を「meta-share でサンプルした相手 mix」に対して回す。

単一相手RL の overfit(crustle 実験で確認、他マッチ劣化で field net ゼロ)を避けるため、毎ゲーム
相手アーキを meta-share でサンプルし、learner(alakazam)がフィールド全体に強くなるよう学習する。

collect_parallel.py と同じ torch非依存ワーカー(pure-Python PolicyModel、M0でtorch一致)。
"""

from __future__ import annotations

import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAX_STEPS = 3000
_W: dict = {}


def _init_field(learner_weights, opp_specs, learner_deck, temperature):
    """opp_specs: list[(weights_path, deck_ids)]。全相手PolicyModelをロード(is_ready検証)。"""
    from ptcg_ai.learning.policy_model import PolicyModel
    pm = PolicyModel(learner_weights)
    if not pm.is_ready:
        raise RuntimeError(f"learner not ready: {learner_weights}")
    _W["pm"] = pm
    opps = []
    for w, deck in opp_specs:
        m = PolicyModel(w)
        if not m.is_ready:
            raise RuntimeError(f"opponent not ready: {w}")
        opps.append((m, deck))
    _W["opps"] = opps
    _W["deck_l"] = learner_deck
    _W["temp"] = temperature


def _softmax_sample(scores, temperature, rng):
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    z = sum(exps)
    probs = [e / z for e in exps]
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i, math.log(max(probs[i], 1e-12))
    return len(probs) - 1, math.log(max(probs[-1], 1e-12))


def _play_field(task):
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.learning import encoder

    learner_index, seed, opp_idx = task
    pm = _W["pm"]
    opp_pm, deck_o = _W["opps"][opp_idx]
    deck_l = _W["deck_l"]
    temp = _W["temp"]
    rng = random.Random(seed ^ 0x9E3779B9)
    random.seed(seed)

    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)
    steps = []
    reward = 0.0
    error = None
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"steps": [], "reward": 0.0, "opp_idx": opp_idx, "error": f"start {sd.errorType}"}
    n = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                error = "current None"; break
            if cur.result != -1:
                reward = 1.0 if cur.result == learner_index else 0.0
                break
            if n >= MAX_STEPS:
                error = "max_steps"; break
            select = obs.select
            if cur.yourIndex == learner_index:
                if select is not None and select.option and select.maxCount == 1:
                    sf = encoder.encode_state_from_state(cur)
                    of = encoder.encode_options_from_state(cur, select)
                    ci = encoder.encode_option_card_ids(cur, select)
                    if of:
                        scores = [pm._forward(sf, of[i], ci[i]) for i in range(len(of))]
                        idx, logp = _softmax_sample(scores, temp, rng)
                        steps.append({"state_feat": sf, "option_feats": of, "card_ids": ci,
                                      "chosen_idx": idx, "logprob": logp})
                        action = [idx]
                    else:
                        action = [0]
                elif select is not None and select.option:
                    scores = pm.score_options(obs)
                    nn = len(select.option)
                    count = max(select.minCount, min(select.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: scores[i], reverse=True)[:count]
                              if scores else list(range(count)))
                else:
                    action = []
            else:
                if select is None or not select.option:
                    action = []
                elif select.maxCount == 1:
                    oi = opp_pm.select_option(obs)
                    action = [oi if oi is not None else 0]
                else:
                    osc = opp_pm.score_options(obs)
                    nn = len(select.option)
                    count = max(select.minCount, min(select.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: osc[i], reverse=True)[:count]
                              if osc else list(range(count)))
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        battle_finish()
    return {"steps": steps, "reward": reward, "opp_idx": opp_idx, "error": error}


def parallel_collect_field(learner_weights, opp_specs, shares, learner_deck,
                           n_games, seed0, temperature, workers):
    """相手を share で毎ゲームサンプルして並列収集。opp_specs と shares は同じ順序・長さ。
    戻り: (trajectories(list[dict]), wins, valid, errors, per_opp_winrate(dict idx->(w,n)))。"""
    rng = random.Random(seed0)
    total = sum(shares)
    cum = []
    acc = 0.0
    for s in shares:
        acc += s / total
        cum.append(acc)

    def sample_opp():
        r = rng.random()
        for i, c in enumerate(cum):
            if r <= c:
                return i
        return len(cum) - 1

    tasks = [(g % 2, seed0 + g, sample_opp()) for g in range(n_games)]
    with Pool(processes=workers, initializer=_init_field,
              initargs=(learner_weights, opp_specs, learner_deck, temperature)) as pool:
        results = pool.map(_play_field, tasks, chunksize=max(1, n_games // (workers * 4)))

    trajs, wins, valid, errors = [], 0, 0, 0
    per_opp: dict[int, list[int]] = {}
    for r in results:
        if r["error"] is not None:
            errors += 1
            continue
        valid += 1
        won = 1 if r["reward"] >= 1.0 else 0
        wins += won
        oi = r["opp_idx"]
        per_opp.setdefault(oi, [0, 0])
        per_opp[oi][0] += won
        per_opp[oi][1] += 1
        if r["steps"]:
            trajs.append(r)
    return trajs, wins, valid, errors, per_opp
