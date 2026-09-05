"""ISMCTS v2.4 Phase C — rollout-state distillation dataset 収集(teacher=Policy 735dd38a)。

teacher が rollout 内で score_options を呼ぶ deployment distribution の state を中心に収集する:
self-play で局面を進め、各 MAIN state から determinize→teacher rollout を回し、rollout 中の各
score_options call について **PolicyModel と同一の 239 次元標準化入力 h**(= standardize(state166) ++
standardize(option65) ++ card_embedding(8))と **teacher raw score** を legal option ごとに記録。
game 単位で split(同一 game の state が train/val/test に跨らない)。

leakage: student 入力は teacher と完全同一(公開情報のみ、determinize は belief)。新規情報を足さない。
出力: npz(X[h], teacher_score, group_id, game_id, split, meta)。`python collect_rollout_dataset.py [n_games] [out.npz]`
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
for p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_ROOT / "kaggle_replays" / "challengers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cg.api import to_observation_class, SelectType  # noqa: E402
from cg import api as cg_api  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
import ismcts_v1_agent as A  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.hidden_information import match_context  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
from ptcg_ai.search import pipeline as _pl  # noqa: E402

_CFG = A.load_config()
_ROLLOUTS_PER_STATE = 2      # 各 MAIN state から回す teacher rollout 数(state 多様化)
_MAX_ROLLOUT_STEPS = 40


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def _build_h(model, state, select):
    """PolicyModel._forward と同一の 239 次元入力 h を legal option ごとに構築(teacher scores も返す)。"""
    sf = encoder.encode_state_from_state(state)
    orows = encoder.encode_options_from_state(state, select)
    cids = encoder.encode_option_card_ids(state, select)
    if not orows or len(orows) != len(select.option):
        return None, None
    sm, ss = model._state_mean, model._state_std
    om, os_ = model._option_mean, model._option_std
    state_std = [(sf[i] - sm[i]) / ss[i] if ss[i] else 0.0 for i in range(len(sf))]
    hs = []
    for orow, cid in zip(orows, cids):
        opt_std = [(orow[i] - om[i]) / os_[i] if os_[i] else 0.0 for i in range(len(orow))]
        h = state_std + opt_std + model._card_embedding(cid)
        hs.append(h)
    scores = model.score_options_from_state(state, select)
    if not scores or len(scores) != len(hs):
        return None, None
    return hs, scores


def _record(store, model, state, select, game_id, group_ctr, rollout_step, actor_is_ref):
    hs, scores = _build_h(model, state, select)
    if hs is None:
        return group_ctr
    gid = group_ctr
    for h, sc in zip(hs, scores):
        store["X"].append(np.asarray(h, dtype=np.float32))
        store["score"].append(np.float32(sc))
        store["group"].append(gid)
        store["game"].append(game_id)
        store["step"].append(rollout_step)
        store["actor_ref"].append(1 if actor_is_ref else 0)
        store["seltype"].append(int(select.type))
    return group_ctr + 1


def _one_game(game_id, deck, model, store):
    match_context.reset()
    obs_dict, sd = battle_start(deck, deck)
    if sd.errorType != 0:
        return
    group_ctr = store["_group_ctr"]
    for _ in range(400):
        obs = to_observation_class(obs_dict)
        cur = obs.current
        if cur is None or cur.result != -1:
            break
        sel = obs.select
        if sel is None:
            break
        try:
            match_context.update(obs)
        except Exception:
            pass
        factory = mpa._model_hidden_state_factory(obs, _CFG)
        if sel.type == SelectType.MAIN and sel.maxCount == 1 and len(sel.option) >= 2:
            det = A._determinize_factory(obs, _CFG)
            me = cur.yourIndex
            for _r in range(_ROLLOUTS_PER_STATE):
                world = None
                try:
                    world = det()
                except Exception:
                    world = None
                if world is None:
                    break
                try:
                    ncg = cg_api.search_begin(obs, world["your_deck"], world["your_prize"],
                                              world["opponent_deck"], world["opponent_prize"],
                                              world["opponent_hand"], world["opponent_active"])
                    for step in range(_MAX_ROLLOUT_STEPS):
                        o = ncg.observation; s = o.current
                        if s is None or s.result != -1 or o.select is None or not o.select.option:
                            break
                        if o.select.maxCount == 1 and len(o.select.option) >= 2:
                            group_ctr = _record(store, model, s, o.select, game_id, group_ctr,
                                                 step, s.yourIndex == me)
                        selection = _pl._greedy_selection(model, o)
                        if not selection:
                            break
                        try:
                            ncg = cg_api.search_step(ncg.searchId, selection)
                        except ValueError:
                            break
                finally:
                    try: cg_api.search_release(ncg.searchId)
                    except Exception: pass
                    try: cg_api.search_end()
                    except Exception: pass
        # advance game with teacher greedy
        if sel.maxCount == 1:
            idx = model.select_option(obs, factory, time.perf_counter() + 0.05)
            action = [idx if idx is not None else 0]
        else:
            action = list(range(sel.minCount))
        obs_dict = battle_select(action)
    battle_finish()
    store["_group_ctr"] = group_ctr


def main(n_games=40, out="rollout_dataset.npz"):
    os.chdir(_SUB)
    deck = _load_deck()
    model = mpa._get_model(_CFG)
    if not model.is_ready:
        print("teacher policy 未ロード"); return
    store = {k: [] for k in ("X", "score", "group", "game", "step", "actor_ref", "seltype")}
    store["_group_ctr"] = 0
    t0 = time.perf_counter()
    for g in range(n_games):
        _one_game(g, deck, model, store)
        if (g + 1) % 5 == 0:
            print(f"  game {g+1}/{n_games}  states={store['_group_ctr']}  rows={len(store['X'])}  "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    X = np.stack(store["X"]).astype(np.float32)
    score = np.asarray(store["score"], dtype=np.float32)
    group = np.asarray(store["group"], dtype=np.int64)
    game = np.asarray(store["game"], dtype=np.int64)
    step = np.asarray(store["step"], dtype=np.int32)
    actor_ref = np.asarray(store["actor_ref"], dtype=np.int8)
    seltype = np.asarray(store["seltype"], dtype=np.int32)
    # game 単位 split: 70/15/15
    ngame = int(game.max()) + 1 if len(game) else 0
    rng = np.random.default_rng(0)
    perm = rng.permutation(ngame)
    tr = set(perm[: int(0.70 * ngame)].tolist())
    va = set(perm[int(0.70 * ngame): int(0.85 * ngame)].tolist())
    split = np.array([0 if int(g) in tr else (1 if int(g) in va else 2) for g in game], dtype=np.int8)
    out_path = _HERE / out
    np.savez_compressed(out_path, X=X, score=score, group=group, game=game, step=step,
                        actor_ref=actor_ref, seltype=seltype, split=split)
    n_states = int(group.max()) + 1 if len(group) else 0
    print(f"\nsaved {out_path}")
    print(f"  states(groups)={n_states}  option-rows={len(X)}  games={ngame}  h_dim={X.shape[1]}")
    print(f"  split rows train/val/test = {(split==0).sum()}/{(split==1).sum()}/{(split==2).sum()}")
    print(f"  mean options/state = {len(X)/max(1,n_states):.2f}  [{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    outname = sys.argv[2] if len(sys.argv) > 2 else "rollout_dataset.npz"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main(n_games=n, out=outname)
