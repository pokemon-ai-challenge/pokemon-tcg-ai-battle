"""ISMCTS v2.4 Phase B/D1 — rollout Policy call の cost 分解(feature 構築 vs NN forward)。

Small rollout Policy が成立するのは「call 時間が NN forward 支配」の時のみ(feature 構築支配なら
小型化しても速くならない=Case D)。実 rollout 中に遭遇する state で score_options_from_state を
成分分解:encode_state / encode_options / encode_option_card_ids / NN _forward / 全体。
read-only。`python policy_cost_profile.py [n_states]`
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_SUB = _ROOT / "sample_submission"
for p in (str(_SUB), str(_ROOT / "kaggle_replays" / "search"), str(_HERE)):
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
_REPEAT = 20   # 各 state で成分を _REPEAT 回まわして平均(ms 単位の分解を安定化)


def _load_deck():
    lines = (_SUB / "deck.csv").read_text(encoding="utf-8").split("\n")
    return [int(lines[i]) for i in range(60)]


def _time(fn, repeat):
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat * 1000.0   # ms/call


def _profile_call(model, state, select):
    """score_options_from_state を成分分解。返り: dict(ms)。"""
    enc_state = lambda: encoder.encode_state_from_state(state)
    enc_opts = lambda: encoder.encode_options_from_state(state, select)
    enc_ids = lambda: encoder.encode_option_card_ids(state, select)
    full = lambda: model.score_options_from_state(state, select)
    # NN forward のみ: 事前に特徴を作っておき _forward を options 回まわす
    sf = encoder.encode_state_from_state(state)
    orows = encoder.encode_options_from_state(state, select)
    cids = encoder.encode_option_card_ids(state, select)
    def nn_only():
        for orow, cid in zip(orows, cids):
            model._forward(sf, orow, cid)
    return {
        "n_opt": len(select.option),
        "encode_state": _time(enc_state, _REPEAT),
        "encode_options": _time(enc_opts, _REPEAT),
        "encode_card_ids": _time(enc_ids, _REPEAT),
        "nn_forward": _time(nn_only, _REPEAT),
        "full": _time(full, _REPEAT),
    }


def main(n_states=16, max_steps=400):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(_SUB)
    match_context.reset()
    deck = _load_deck()
    obs_dict, sd = battle_start(deck, deck)
    if sd.errorType != 0:
        print("battle_start 失敗"); return
    model = mpa._get_model(_CFG)

    rows = []          # 各 profiled state の分解
    n_done = 0
    try:
        for _ in range(max_steps):
            if n_done >= n_states:
                break
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
                ok = False
                try:
                    ok = det() is not None
                except Exception:
                    ok = False
                if ok:
                    # この MAIN state を profile
                    rows.append(_profile_call(model, cur, sel))
                    # + rollout 中間 state も profile(determinize→search_begin→greedy 数手)
                    world = det()
                    if world is not None:
                        try:
                            ncg = cg_api.search_begin(obs, world["your_deck"], world["your_prize"],
                                                      world["opponent_deck"], world["opponent_prize"],
                                                      world["opponent_hand"], world["opponent_active"])
                            for _step in range(6):
                                o = ncg.observation; s = o.current
                                if s is None or s.result != -1 or o.select is None or not o.select.option:
                                    break
                                if o.select.maxCount == 1 and len(o.select.option) >= 2:
                                    rows.append(_profile_call(model, s, o.select))
                                selction = _pl._greedy_selection(model, o)
                                if not selction:
                                    break
                                try:
                                    ncg = cg_api.search_step(ncg.searchId, selction)
                                except ValueError:
                                    break
                        finally:
                            try: cg_api.search_release(ncg.searchId)
                            except Exception: pass
                            try: cg_api.search_end()
                            except Exception: pass
                    n_done += 1
            if sel.maxCount == 1:
                idx = model.select_option(obs, factory, time.perf_counter() + 0.05)
                action = [idx if idx is not None else 0]
            else:
                action = list(range(sel.minCount))
            obs_dict = battle_select(action)
    finally:
        battle_finish()

    if not rows:
        print("state 収集失敗"); return
    keys = ["encode_state", "encode_options", "encode_card_ids", "nn_forward", "full"]
    print(f"profiled {len(rows)} rollout-relevant states  (repeat={_REPEAT}/component)")
    print(f"  mean options/state = {statistics.mean(r['n_opt'] for r in rows):.1f}\n")
    means = {k: statistics.mean(r[k] for r in rows) for k in keys}
    full = means["full"]
    print(f"  {'component':16s} {'ms/call':>9s} {'% of full':>10s}")
    for k in keys:
        print(f"  {k:16s} {means[k]:9.3f} {100*means[k]/full:9.0f}%")
    feat = means["encode_state"] + means["encode_options"] + means["encode_card_ids"]
    print(f"\n  feature construction 合計 ≈ {feat:.3f}ms ({100*feat/full:.0f}% of full)")
    print(f"  NN forward           ≈ {means['nn_forward']:.3f}ms ({100*means['nn_forward']/full:.0f}% of full)")
    print("\n判定:")
    if means["nn_forward"] / full >= 0.5:
        print("  → NN forward 支配 = Small rollout Policy で高速化余地あり(distillation 進行可)。")
    elif means["nn_forward"] / full <= 0.25:
        print("  → NN forward は小割合 = feature 構築支配 = Case D。小型 net にしても rollout は速くならない。")
        print("    真の lever は feature 構築の軽量化(encoder caching / cheaper rollout features)。")
    else:
        print("  → 中間。small net で部分的高速化は可能だが上限は feature 構築時間。")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    main(n_states=n)
