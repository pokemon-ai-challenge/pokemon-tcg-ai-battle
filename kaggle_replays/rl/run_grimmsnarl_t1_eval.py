"""オーロンゲT1(蒸留、ルールベース教師)の評価。直接head-to-headと、
フーディンと同じ固定8相手プールでの評価の両方を行う(既存t1_eval.pyの
PolicyModel前提の実装とは相手の種類が違う(ルールベース)ため、専用に書く。
既存t1_eval.py/run_fixed_pool_eval.pyは変更しない)。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CKPT = _ROOT / "kaggle_replays" / "rl" / "runs" / "distill_grimmsnarl" / "train" / "rule_teacher_seed0" / "best.pt"
_V40_FOR_FEATURES = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_GRIMM_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "marnie_grimmsnarl_ex" / "01.csv"
MAX_STEPS = 3000


def wilson_lo(w: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = w / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (center - margin) / denom


def _rule_action(fw, st, obs, learner_deck) -> list[int]:
    ctx = fw.build_ctx(obs, learner_deck)
    st.prepare(ctx)
    ctx.memo["attack_evals"] = fw.evaluate_attacks(ctx, bonus=st.attack_bonus(ctx))
    return fw.choose(ctx, st)


def play_one_game(t1_model, vocab, t1_state_pm, opponent, deck_t1, deck_opp, t1_index: int,
                  seed: int, profile_name: str, device="cpu") -> dict:
    """opponent: {"type": "rule"} でルールベース教師自身、{"type": "pm", "pm": PolicyModel}で
    固定プールの相手。T1側の複数選択は同じルールベース教師のargmaxへフォールバック
    (蒸留・収集と同じ慣習)。

    ``seed``はPython側にのみ作用する(この対局専用の``random.Random(seed)``を1つ
    作るためだけに使う)。**cgエンジン(cg.dll、ctypes経由でロードするネイティブ
    ライブラリ)にはPythonから呼べるseed設定APIが無いため、この``seed``は
    デッキシャッフル・サイド配置などcg内部の乱数には一切影響しない**
    (``diagnose_seed_reproducibility.py``で確認済み、design.md §9.1.1)。
    同じseedを渡しても対局の実際の展開は再現しない——記録の一貫性(どのseed値で
    この対局を評価したかを追跡できる)のために保持している値であり、決定性の
    保証ではない。

    現状、T1のargmax行動選択・ルール教師のargmaxスコアリング・PM相手のsorted選択は
    いずれも決定的でPython randomを消費しないため、下の``episode_rng``は未使用。
    将来Python側で乱数が必要になった場合に備え、グローバルな``random.seed()``への
    依存(このプロセスの他のコードに影響する副作用がある)を避けて、呼び出し元ごとに
    独立したRNGインスタンスとして明示的に持たせている。
    """
    import t1_live_agent as la
    from opponents.rule_agents import grimmsnarl as gm
    from opponents.rule_agents import framework as fw
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    episode_rng = random.Random(seed)  # noqa: F841  (cg内部には影響しない、将来のPython側乱数用に予約)
    st = gm.GrimmsnarlStrategy()
    deck0 = deck_t1 if t1_index == 0 else deck_opp
    deck1 = deck_t1 if t1_index == 1 else deck_opp
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        return {"error": f"start {start_data.errorType}", "winner": None, "t1_win": None,
               "illegal_actions": 0}

    illegal = 0
    error = None
    winner = None
    n = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                error = "current None"; break
            if cur.result != -1:
                winner = cur.result; break
            if n >= MAX_STEPS:
                error = "max_steps"; break

            select = obs.select
            if select is not None and select.option:
                if cur.yourIndex == t1_index and select.maxCount == 1:
                    idx = la.t1_select_index(t1_model, vocab, t1_state_pm, cur, select,
                                             profile_name, device=device)
                    if not (0 <= idx < len(select.option)):
                        illegal += 1; idx = 0
                    action = [idx]
                elif cur.yourIndex == t1_index:
                    action = _rule_action(fw, st, obs, deck_t1)
                elif opponent["type"] == "rule":
                    action = _rule_action(fw, st, obs, deck_opp)
                else:
                    sc = opponent["pm"].score_options_from_state(cur, select)
                    nn = len(select.option)
                    count = max(select.minCount, min(select.maxCount, nn))
                    action = sorted(range(nn), key=lambda i: sc[i], reverse=True)[:count]
                if not (isinstance(action, list) and all(0 <= i < len(select.option) for i in action)):
                    illegal += 1; action = [0]
            else:
                action = []
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        battle_finish()

    t1_win = None if (error is not None or winner is None) else (winner == t1_index)
    return {"error": error, "winner": winner, "t1_win": t1_win, "illegal_actions": illegal}


def run_series(t1_model, vocab, t1_state_pm, opponent, deck_t1, deck_opp, n_games, seed0,
               profile_name, device="cpu") -> dict:
    results = []
    for g in range(n_games):
        t1_index = g % 2
        r = play_one_game(t1_model, vocab, t1_state_pm, opponent, deck_t1, deck_opp, t1_index,
                          seed0 + g, profile_name, device=device)
        results.append(r)
    valid = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    wins = sum(1 for r in valid if r["t1_win"] is True)
    illegal = sum(r["illegal_actions"] for r in results)
    return {"n_games": n_games, "n_valid": len(valid), "n_errors": len(errors),
           "n_draws": sum(1 for r in valid if r["t1_win"] is None),
           "t1_wins": wins, "t1_winrate": (wins / len(valid) if valid else float("nan")),
           "winrate_ci95_lo": wilson_lo(wins, len(valid)),
           "illegal_actions_total": illegal, "error_types": [r["error"] for r in errors]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2h-games", type=int, default=300)
    ap.add_argument("--pool-games-per-opponent", type=int, default=125)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import t1_live_agent as la
    import common as C
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    t1_model, vocab, profile_name, _ = la.load_t1_for_inference(_CKPT, device="cpu")
    t1_state_pm = PolicyModel(str(_V40_FOR_FEATURES))
    deck_t1 = read_deck_csv_file(str(_GRIMM_DECK))

    out = {}

    print("=== head-to-head vs ルールベース教師 ===", flush=True)
    h2h = run_series(t1_model, vocab, t1_state_pm, {"type": "rule"}, deck_t1, deck_t1,
                     args.h2h_games, seed0=7000000, profile_name=profile_name)
    out["h2h_teacher"] = h2h
    print(f"T1勝率={h2h['t1_winrate']:.4f}({h2h['t1_wins']}/{h2h['n_valid']}) "
         f"CIlo={h2h['winrate_ci95_lo']:.4f} err={h2h['n_errors']} illegal={h2h['illegal_actions_total']}",
         flush=True)

    print("=== 固定8相手プール ===", flush=True)
    run_cfg = json.loads((_ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "run.json")
                         .read_text(encoding="utf-8"))
    run_dir = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1"
    per_opp = []
    tot_w = tot_v = 0
    offset = 0
    for opp in run_cfg["opponents"]:
        w = C.resolve_opponent_weights(run_dir, opp.get("weights"))
        d = read_deck_csv_file(str(C.resolve_deck(opp.get("deck") or run_cfg["opponent_deck"])))
        pm = PolicyModel(str(w) if w else None)
        r = run_series(t1_model, vocab, t1_state_pm, {"type": "pm", "pm": pm}, deck_t1, d,
                       args.pool_games_per_opponent, seed0=7500000 + offset, profile_name=profile_name)
        offset += args.pool_games_per_opponent
        tot_w += r["t1_wins"]; tot_v += r["n_valid"]
        per_opp.append({"id": opp["id"], **r})
        print(f"{opp['id']}: {r['t1_winrate']:.3f}({r['t1_wins']}/{r['n_valid']}) "
             f"err={r['n_errors']} illegal={r['illegal_actions_total']}", flush=True)
    out["fixed_pool"] = {"overall_wins": tot_w, "overall_valid": tot_v,
                         "overall_winrate": tot_w / tot_v if tot_v else float("nan"),
                         "overall_winrate_ci95_lo": wilson_lo(tot_w, tot_v), "per_opponent": per_opp}
    print(f"固定プール全体: {out['fixed_pool']['overall_winrate']:.4f}({tot_w}/{tot_v})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
