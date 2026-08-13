"""full-agent を「実際の提出パイプライン」でメタフィールド全体に対して測るペア比較ツール。

既存 eval_field.py との違い(どちらも必要だが役割が違う):
  - eval_field.py の既定 --config-base は `ml_lethal_attackplan_v0only`(PIMC前読み **無し**)。
    実際の提出は `abl_5_full`(PIMC前読み **有り**)なので、あれは提出物とは別のパイプラインを
    測っている。本スクリプトは既定を abl_5_full にし、--ml-config で明示的に振る。
  - 「デッキ」「重み」「config」の3つを arm ごとに独立に差し替えられる。デッキ探索と
    探索configチューニングを同じ土俵で比較するため。
  - arm 間で対戦カード(相手アーキ・相手デッキ・先後・seed)を完全に揃えた paired 比較を行い、
    paired bootstrap で差の信頼区間を出す。

相手は11アーキの模倣Policy(g2優先)× そのアーキの全合法デッキ、gen2メタシェア加重。

使い方(arm は JSON で渡す):
    python kaggle_replays/rl/eval_agent_field.py --games 200 --workers 6 \
      --arm '{"name":"deck01_rl","deck":"kaggle_replays/meta_analysis/archetype_decks_g2/ogerpon_teal_ex/01.csv","weights":"sample_submission/ptcg_ai/learning/policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"}' \
      --arm '{"name":"deck03_rl","deck":"kaggle_replays/meta_analysis/archetype_decks_g2/ogerpon_teal_ex/03.csv","weights":"sample_submission/ptcg_ai/learning/policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"}' \
      --output runs/ogerpon/deck_ab.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from matchup_common import (  # noqa: E402
    GEN2_META_SHARE, TARGET_ARCHETYPES, atomic_write_json, discover_archetype_decks,
    read_deck, resolve_opponent_weights, resolve_path, sha256_file, validate_deck,
)
from train_v3 import wilson_lo  # noqa: E402

MAX_STEPS = 3000
_W: dict = {}


def _init(weights_path, ml_config_name, workdir, opponents):
    import os
    os.environ["PTCG_AI_ML_CONFIG"] = ml_config_name
    # 提出時と同じ状態(CWD直下の deck.csv = 自分のデッキ)にする。これが無いと
    # read_deck_csv() / match_context._load_own_deck_ids() が FileNotFoundError になり
    # lethal探索・PIMC の hidden_state_factory が全部落ちる。
    os.chdir(workdir)
    from ptcg_ai.core.config import load_config
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.ml_policy import ml_policy_agent

    config = dict(load_config(ml_config_name))
    config["policy_weights_path"] = weights_path
    _W["config"] = config
    _W["agent"] = ml_policy_agent
    # 相手は (archetype, weights_path, deck_ids) のリスト。Policyはパスごとに1個だけ持つ。
    models: dict[str, PolicyModel] = {}
    for _arch, wpath, _deck in opponents:
        if wpath not in models:
            m = PolicyModel(wpath)
            if not m.is_ready:
                raise RuntimeError(f"opponent policy not ready: {wpath}")
            models[wpath] = m
    _W["opp_models"] = models
    _W["opponents"] = opponents


def _play(task):
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    learner_index, seed, opp_idx = task
    random.seed(seed)
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))  # CWD = workdir(=learnerデッキ)
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"reward": 0.0, "error": f"start {sd.errorType}", "opp_idx": opp_idx,
                "steps": 0, "illegal": False}
    n = 0
    err = None
    illegal = False
    reward = 0.0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                err = "current None"; break
            if cur.result != -1:
                reward = 1.0 if cur.result == learner_index else 0.0
                break
            if n >= MAX_STEPS:
                err = "max_steps"; break
            sel = obs.select
            if cur.yourIndex == learner_index:
                action = _W["agent"].agent(obs, _W["config"])
            else:
                if sel is None or not sel.option:
                    action = []
                elif sel.maxCount == 1:
                    oi = opp_pm.select_option(obs)
                    action = [oi if oi is not None else 0]
                else:
                    osc = opp_pm.score_options(obs)
                    nn = len(sel.option)
                    count = max(sel.minCount, min(sel.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: osc[i], reverse=True)[:count]
                              if osc else list(range(count)))
            if sel is not None:
                if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
                    illegal = True; err = f"bad type {action}"; break
                if not (sel.minCount <= len(action) <= sel.maxCount):
                    illegal = True; err = f"bad count {action}"; break
                if len(action) != len(set(action)):
                    illegal = True; err = f"dup {action}"; break
                if not all(0 <= i < len(sel.option) for i in action):
                    illegal = True; err = f"oob {action}"; break
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    finally:
        battle_finish()
    return {"reward": reward if err is None else 0.0, "error": err, "opp_idx": opp_idx,
            "steps": n, "illegal": illegal}


def build_field(archetypes=None, shares=None, deck_gen="g2"):
    """[(archetype, weights_path, deck_ids)] と share のリストを返す(デッキ単位)。

    deck_gen: "g2" = archetype_decks_g2(2026-08取得の現行メタ。オーガポン等が現行型)。
              "july" = archetype_decks(7月プール、従来)。
    """
    archetypes = archetypes or list(TARGET_ARCHETYPES)
    shares = shares or GEN2_META_SHARE
    root = resolve_path("kaggle_replays/meta_analysis/"
                        + ("archetype_decks_g2" if deck_gen == "g2" else "archetype_decks"))
    opponents, weights = [], []
    skipped = []
    for arch in archetypes:
        wres = resolve_opponent_weights(arch)
        if wres["path"] is None:
            skipped.append({"archetype": arch, "reason": wres["error"]}); continue
        deck_paths = sorted(str(p) for p in (root / arch).glob("*.csv"))
        if not deck_paths:
            skipped.append({"archetype": arch, "reason": f"no deck under {root / arch}"}); continue
        decks = discover_archetype_decks(arch, explicit=deck_paths)
        usable = [d for d in decks if not d["errors"] and d["dup_of"] is None]
        if not usable:
            skipped.append({"archetype": arch, "reason": "no usable deck"}); continue
        s = float(shares.get(arch, 0.0))
        if s <= 0:
            skipped.append({"archetype": arch, "reason": "zero share"}); continue
        for d in usable:
            opponents.append((arch, str(wres["path"]), d["deck"]))
            weights.append(s / len(usable))
    return opponents, weights, skipped


def make_tasks(weights, n_games, seed0):
    """arm間で完全に同一の対戦カード列(相手index・先後・seed)を作る。"""
    rng = random.Random(seed0)
    total = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)

    def sample():
        r = rng.random()
        for i, c in enumerate(cum):
            if r <= c:
                return i
        return len(cum) - 1

    return [(g % 2, seed0 + g, sample()) for g in range(n_games)]


def summarize(results, tasks, opponents):
    wins = valid = errors = illegal = timeouts = 0
    per_arch: dict[str, list[int]] = {}
    per_side = {"first": [0, 0], "second": [0, 0]}
    error_counts: dict[str, int] = {}
    outcomes = []
    for i, r in enumerate(results):
        arch = opponents[r["opp_idx"]][0]
        if r["error"] is not None:
            errors += 1
            error_counts[r["error"]] = error_counts.get(r["error"], 0) + 1
            if r["error"] == "max_steps":
                timeouts += 1
            if r["illegal"]:
                illegal += 1
            outcomes.append(None)
            continue
        valid += 1
        won = 1 if r["reward"] >= 1.0 else 0
        wins += won
        outcomes.append(won)
        slot = per_arch.setdefault(arch, [0, 0])
        slot[0] += won; slot[1] += 1
        side = "first" if tasks[i][0] == 0 else "second"
        per_side[side][0] += won; per_side[side][1] += 1
    return {
        "wins": wins, "valid": valid, "errors": errors, "illegal_actions": illegal,
        "timeouts": timeouts,
        "winrate": wins / valid if valid else float("nan"),
        "wilson_lo": wilson_lo(wins, valid),
        "per_archetype": {k: {"wins": v[0], "games": v[1],
                              "winrate": v[0] / v[1] if v[1] else None}
                          for k, v in sorted(per_arch.items())},
        "per_side": {k: {"wins": v[0], "games": v[1],
                         "winrate": v[0] / v[1] if v[1] else None}
                     for k, v in per_side.items()},
        "error_breakdown": dict(sorted(error_counts.items(), key=lambda kv: -kv[1])[:5]),
        "_outcomes": outcomes,
    }


def paired_bootstrap(a_out, b_out, n_boot=2000, seed=0):
    """両方 valid なゲームだけでペア比較し、差(b-a)の95%CIを返す。"""
    pairs = [(a, b) for a, b in zip(a_out, b_out) if a is not None and b is not None]
    if not pairs:
        return (float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_boot):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(b for _, b in s) / n - sum(a for a, _ in s) / n)
    diffs.sort()
    return (diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1], n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help='JSON: {"name":..., "deck":..., "weights":..., "ml_config":"abl_5_full"}')
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777777)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ml-config", default="abl_5_full", help="arm 側で未指定のときの既定")
    ap.add_argument("--opponent-deck-gen", default="g2", choices=("g2", "july"),
                    help="相手デッキの世代。g2=2026-08取得の現行メタ(既定)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    arms = [json.loads(a) for a in args.arm]
    opponents, weights, skipped = build_field(deck_gen=args.opponent_deck_gen)
    tasks = make_tasks(weights, args.games, args.seed)
    print(f"field: {len(opponents)} (archetype x deck), skipped={skipped}", flush=True)

    results_by_arm = {}
    for arm in arms:
        name = arm["name"]
        deck_path = resolve_path(arm["deck"])
        deck = read_deck(deck_path)
        errs = validate_deck(deck)
        if errs:
            raise RuntimeError(f"arm {name}: illegal deck {errs}")
        weights_path = str(resolve_path(arm["weights"]))
        ml_config = arm.get("ml_config", args.ml_config)

        workdir = tempfile.mkdtemp(prefix=f"evalfield_{name}_")
        Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n",
                                             encoding="utf-8")
        t0 = time.time()
        with Pool(processes=args.workers, initializer=_init,
                  initargs=(weights_path, ml_config, workdir, opponents)) as pool:
            res = pool.map(_play, tasks, chunksize=1)
        elapsed = time.time() - t0
        s = summarize(res, tasks, opponents)
        s["deck"] = str(deck_path); s["deck_sha256"] = sha256_file(deck_path)
        s["weights"] = weights_path; s["weights_sha256"] = sha256_file(Path(weights_path))
        s["ml_config"] = ml_config; s["wall_seconds"] = elapsed
        results_by_arm[name] = s
        print(f"[{name}] wr={s['winrate']:.4f} ({s['wins']}/{s['valid']}) "
              f"errors={s['errors']} {elapsed:.0f}s {s['error_breakdown']}", flush=True)

    base_name = arms[0]["name"]
    comparisons = {}
    for arm in arms[1:]:
        lo, hi, npair = paired_bootstrap(results_by_arm[base_name]["_outcomes"],
                                         results_by_arm[arm["name"]]["_outcomes"],
                                         seed=args.seed)
        d = results_by_arm[arm["name"]]["winrate"] - results_by_arm[base_name]["winrate"]
        comparisons[f"{arm['name']} - {base_name}"] = {
            "delta_winrate": d, "paired_bootstrap_95ci": [lo, hi], "paired_games": npair,
            "significant": (lo > 0 or hi < 0),
        }
        print(f"[{arm['name']} - {base_name}] delta={d:+.4f} 95%CI=[{lo:+.4f},{hi:+.4f}] "
              f"{'SIGNIFICANT' if (lo > 0 or hi < 0) else 'not significant'}", flush=True)

    out = {"games": args.games, "seed": args.seed, "field_size": len(opponents),
           "opponent_deck_gen": args.opponent_deck_gen, "field_skipped": skipped,
           "arms": {k: {kk: vv for kk, vv in v.items() if kk != "_outcomes"}
                    for k, v in results_by_arm.items()},
           "comparisons": comparisons,
           "note": "full-agent, submission pipeline (default abl_5_full), gen2 meta-share weighted, "
                   "paired matchups across arms."}
    atomic_write_json(Path(resolve_path(args.output)), out)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
