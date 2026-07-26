"""並列トラジェクトリ収集(multiprocessing、torch非依存ワーカー)。

cg 収集は CPU律速で単一プロセスだと1コアしか使わない。ワーカーを **pure-Python PolicyModel**
(M0 で torch と数値一致を保証済)にして softmax サンプリングで行動を選び、全コアで並列収集する。
torch は main の PPO 更新だけで使う(ワーカーは torch を import しないので起動が軽い)。

各イテレーション:
  main: 現在の torch policy を temp JSON にエクスポート
      -> parallel_collect(temp JSON) でワーカーが収集(現在の重みを読む)
      -> main が PPO 更新
temp JSON 経由の重み配布なので、ワーカーとの torch テンソル共有が要らず堅牢。
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

# ワーカーごとのグローバル状態(Pool initializer でセット)。
_W: dict = {}


def _init_worker(weights_path, deck_l, deck_o, temperature):
    from cg.api import to_observation_class  # noqa: F401 (import 確認)
    from ptcg_ai.learning.policy_model import PolicyModel
    _W["pm"] = PolicyModel(weights_path)
    _W["deck_l"] = deck_l
    _W["deck_o"] = deck_o
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


def _play_one(task):
    """1試合を pure-Python 方策(sampling)で。learner の単一選択を記録して dict で返す。"""
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.learning import encoder

    learner_index, seed = task
    pm = _W["pm"]
    temp = _W["temp"]
    rng = random.Random(seed ^ 0x5DEECE66D)
    random.seed(seed)

    deck0, deck1 = (_W["deck_l"], _W["deck_o"]) if learner_index == 0 else (_W["deck_o"], _W["deck_l"])
    steps = []
    reward = 0.0
    winner = None
    error = None

    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        return {"steps": [], "reward": 0.0, "winner": None, "error": f"start {start_data.errorType}"}
    n = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                error = "current None"; break
            if cur.result != -1:
                winner = cur.result
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
                        steps.append({"state_feat": sf, "option_feats": of,
                                      "card_ids": ci, "chosen_idx": idx, "logprob": logp})
                        action = [idx]
                    else:
                        action = [0]
                elif select is not None and select.option:
                    # 複数選択: greedy(記録しない)
                    scores = pm.score_options(obs)
                    nn = len(select.option)
                    count = max(select.minCount, min(select.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: scores[i], reverse=True)[:count]
                              if scores else list(range(count)))
                else:
                    action = []
            else:
                # 相手も同じ pure-Python 方策(argmax)。相手を production alakazam にするには
                # 別 weights を渡す設計にできるが、PoC は learner と同一方策プールで対戦させず
                # opponent 用 PolicyModel を別に持つ(下記 _W["opp"])。
                opp = _W.get("opp")
                if select is None or not select.option:
                    action = []
                elif select.maxCount == 1:
                    oi = opp.select_option(obs) if opp is not None else 0
                    action = [oi if oi is not None else 0]
                else:
                    osc = opp.score_options(obs) if opp is not None else []
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
    return {"steps": steps, "reward": reward, "winner": winner, "error": error}


def _init_worker2(weights_path, opp_weights, deck_l, deck_o, temperature):
    from ptcg_ai.learning.policy_model import PolicyModel
    pm = PolicyModel(weights_path)
    # 未ロード(パス誤り等)を黙って index0 固定の壊れた方策として使うと結果が偽陽性になる。
    # 明示的に fail-loud にする(learner は必ずロード済であるべき)。
    if not pm.is_ready:
        raise RuntimeError(f"learner policy not ready (path?): {weights_path}")
    _W["pm"] = pm
    opp = PolicyModel(opp_weights)  # opp_weights=None -> production alakazam(絶対デフォルトパス)
    if not opp.is_ready:
        raise RuntimeError(f"opponent policy not ready (path?): {opp_weights}")
    _W["opp"] = opp
    _W["deck_l"] = deck_l
    _W["deck_o"] = deck_o
    _W["temp"] = temperature


def parallel_collect(weights_path, opp_weights, deck_l, deck_o, n_games, seed0,
                     temperature, workers):
    """ワーカー並列で n_games 収集。learner は weights_path(temp JSON)、相手は opp_weights。
    戻り: (trajectories(list[dict]), wins, valid, errors)。"""
    tasks = [(g % 2, seed0 + g) for g in range(n_games)]
    with Pool(processes=workers, initializer=_init_worker2,
              initargs=(weights_path, opp_weights, deck_l, deck_o, temperature)) as pool:
        results = pool.map(_play_one, tasks, chunksize=max(1, n_games // (workers * 4)))
    trajs, wins, valid, errors = [], 0, 0, 0
    for r in results:
        if r["error"] is not None:
            errors += 1
            continue
        valid += 1
        wins += 1 if r["reward"] >= 1.0 else 0
        if r["steps"]:
            trajs.append(r)
    return trajs, wins, valid, errors
