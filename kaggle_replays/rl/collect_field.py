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


def _potential(state, learner_index) -> float:
    """Potential-Based Reward Shaping 用のポテンシャル Φ(s)∈[-1,1]。

    学習側視点の「サイド進捗差」。サイドは6枚から取り切ると勝ち(残 0)なので、
    進捗 = (6 - 残サイド)/6。Φ = 自分の進捗 − 相手の進捗。終局(勝ち)で自分が
    取り切れば Φ→1 付近になり、終局勝敗報酬と符号が一致する。デッキ切れ/場切れ勝ち
    はサイドに現れないが、PBRS はポテンシャルが何であれ最適方策を変えないため安全側。
    """
    me = state.players[learner_index]
    opp = state.players[1 - learner_index]
    my_prize = len(me.prize or [])
    opp_prize = len(opp.prize or [])
    my_prog = (6.0 - my_prize) / 6.0
    opp_prog = (6.0 - opp_prize) / 6.0
    phi = my_prog - opp_prog
    # 山札保全 PBRS 項(deck_coef>0 のとき)。危険域(<=15)で山が多いほど高い(飽和)。ループ/
    # せいなるはいで山を保つ/戻すと Φ が上がり、山を失うと下がる=密な信号。PBRSゆえ最適方策は
    # 不変(deck_coef=0=既定で従来と完全一致)。ユーザー方針: deckout管理を学習で獲得させる。
    deck_coef = _W.get("deck_coef", 0.0)
    if deck_coef:
        phi += deck_coef * (min(int(me.deckCount or 0), 15) / 15.0)
    # 価値ネット PBRS 項(value_coef>0 のとき)。学習した勝率予測(=将来価値の"直感")を [-1,1] 化して
    # ポテンシャルに足す。密な将来価値信号=方策が「誰に貼る/温存/将来の逃げ」等の長期価値を学ぶ。
    # PBRSゆえ最適方策は不変(安全)。side-progress(短期)+価値ネット(長期)=短期×長期の統合。
    value_coef = _W.get("value_coef", 0.0)
    vn = _W.get("value_net")
    if value_coef and vn is not None:
        try:
            phi += value_coef * (vn.predict_win_prob_from_state(state) - 0.5) * 2.0
        except Exception:  # noqa: BLE001 - 価値ネットの失敗が学習を止めてはならない
            pass
    return phi


def _init_field(learner_weights, opp_specs, learner_deck, temperature, deck_coef=0.0,
                value_coef=0.0, value_weights=None):
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
    _W["deck_coef"] = deck_coef
    _W["value_coef"] = value_coef
    _W["value_net"] = None
    if value_coef and value_weights:
        try:
            from ptcg_ai.learning.value_model import ValueModel
            vn = ValueModel(value_weights)
            _W["value_net"] = vn if getattr(vn, "is_ready", False) else None
        except Exception:  # noqa: BLE001
            _W["value_net"] = None


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


def _pl_sample(scores, count, temperature, rng):
    """Plackett-Luce: 逐次 softmax サンプル(重複なし)で count 個を順に選ぶ。

    返り: (chosen_seq(選んだ順のindex列), logprob(逐次log-probの和))。
    count==1 は _softmax_sample と数値一致(=従来の単一選択と後方互換)。学習側は
    この chosen_seq を action として engine に渡し、policy_logp_entropy 側で同じ PL
    log-prob を再計算して PPO 更新する。
    """
    n = len(scores)
    count = max(1, min(count, n))
    remaining = list(range(n))
    chosen: list[int] = []
    logp = 0.0
    for _ in range(count):
        sub = [scores[i] for i in remaining]
        m = max(sub)
        exps = [math.exp((s - m) / temperature) for s in sub]
        z = sum(exps)
        probs = [e / z for e in exps]
        r = rng.random()
        acc = 0.0
        pick = len(remaining) - 1
        for j, p in enumerate(probs):
            acc += p
            if r <= acc:
                pick = j
                break
        chosen.append(remaining[pick])
        logp += math.log(max(probs[pick], 1e-12))
        remaining.pop(pick)
    return chosen, logp


def _play_field(task):
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.learning import encoder
    from ptcg_ai.learning.extra_features import compute_extra_features

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
    # 入力拡張の predictor はゲーム内で不変なのでループ前に1回だけ用意(parity: 推論と同一ロード)。
    _ef = getattr(pm, "_extra_features", None)
    _predictor = None
    if _ef and "opp_belief" in _ef:
        from ptcg_ai.hidden_information import match_context
        _predictor = match_context._get_predictor()
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
                # 単一選択・多選択を PL に統一(count==1 は従来の単一選択と数値一致)。
                # 多選択も learner の学習対象=chosen_seq と PL log-prob を記録する。
                if select is not None and select.option:
                    # 入力拡張: 学習側/推論側で同一の compute_extra_features を通す(parity)。
                    _extra = compute_extra_features(cur, _ef, deck_l, _predictor)
                    sf = encoder.encode_state_from_state(cur, extra_features=_extra)
                    of = encoder.encode_options_from_state(cur, select)
                    ci = encoder.encode_option_card_ids(cur, select)
                    if of:
                        scores = [pm._forward(sf, of[i], ci[i]) for i in range(len(of))]
                        count = max(select.minCount, min(select.maxCount, len(of)))
                        chosen_seq, logp = _pl_sample(scores, count, temp, rng)
                        steps.append({"state_feat": sf, "option_feats": of, "card_ids": ci,
                                      "chosen_seq": chosen_seq, "logprob": logp,
                                      "phi": _potential(cur, learner_index)})
                        action = chosen_seq
                    else:
                        action = list(range(max(select.minCount, 1)))
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
                           n_games, seed0, temperature, workers, deck_coef=0.0,
                           value_coef=0.0, value_weights=None):
    """相手を share で毎ゲームサンプルして並列収集。opp_specs と shares は同じ順序・長さ。
    ``deck_coef``: 山札保全PBRS項の係数。``value_coef``/``value_weights``: 価値ネットPBRS項(0=不使用)。
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
              initargs=(learner_weights, opp_specs, learner_deck, temperature, deck_coef,
                        value_coef, value_weights)) as pool:
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
