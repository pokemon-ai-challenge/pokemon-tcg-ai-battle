"""PIMC 決定化がどこで落ちているかの内訳(climb v1.5 診断)。

`pipeline.search` は「factory が None」「search_begin が例外」「候補が全部違法」を
すべて黙って握りつぶし、`None` を返して Policy へフォールバックする。そのため
**探索が実際には走っていない**場合でも外からは見えない。内訳を数える。

計測点:
  - factory() が None を返した回数
  - _begin(search_begin) が例外を投げた回数(例外型別)
  - 世界を作れた回数 / そのうち候補評価が返った回数
  - search が最終的に action を返したか None だったか

出力: _diag_pimc_failure_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB_WEIGHTS = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

C = Counter()
BEGIN_ERRORS = Counter()
_cur = {"active": False, "worlds": 0, "evals": 0}
_gate = {"on": True}   # climb-only 計測用のゲート


def _install() -> None:
    orig_search = P.search
    orig_begin = P._begin
    orig_eval = P._evaluate_candidate
    orig_cand = P._select_candidate_indices
    orig_hsf = P._hidden_state_factory

    def search(state, legal_actions, context):
        if not _gate["on"]:
            return orig_search(state, legal_actions, context)
        _cur.update(active=True, worlds=0, evals=0)
        C["search_called"] += 1
        out = orig_search(state, legal_actions, context)
        if _cur.get("reached_candidates"):
            C["reached_candidate_stage"] += 1
            C["returned_action" if out is not None else "returned_none"] += 1
            if _cur["worlds"] == 0:
                C["zero_worlds_built"] += 1
            if _cur["evals"] == 0:
                C["zero_evals"] += 1
        _cur["active"] = False
        _cur["reached_candidates"] = False
        return out

    def cand(select, ranked, config, probs=None):
        if not _gate["on"]:
            return orig_cand(select, ranked, config, probs)
        _cur["reached_candidates"] = True
        return orig_cand(select, ranked, config, probs)

    def hidden_state_factory(context):
        if not _gate["on"]:
            return orig_hsf(context)
        f = orig_hsf(context)
        if f is None:
            C["factory_missing"] += 1
            return None

        def wrapped():
            try:
                hs = f()
            except Exception as exc:  # noqa: BLE001
                C["factory_raised"] += 1
                BEGIN_ERRORS[f"factory:{type(exc).__name__}:{str(exc)[:80]}"] += 1
                return None
            C["factory_none" if hs is None else "factory_ok"] += 1
            return hs

        return wrapped

    def begin(obs, hidden_state):
        if not _gate["on"]:
            return orig_begin(obs, hidden_state)
        try:
            out = orig_begin(obs, hidden_state)
        except Exception as exc:  # noqa: BLE001
            C["begin_raised"] += 1
            BEGIN_ERRORS[f"begin:{type(exc).__name__}:{str(exc)[:120]}"] += 1
            raise
        C["begin_ok"] += 1
        if _cur["active"]:
            _cur["worlds"] += 1
        return out

    def evaluate_candidate(root, candidate, me, config, deadline, evaluator, policy_model):
        if not _gate["on"]:
            return orig_eval(root, candidate, me, config, deadline, evaluator, policy_model)
        out = orig_eval(root, candidate, me, config, deadline, evaluator, policy_model)
        C["cand_eval_none" if out is None else "cand_eval_ok"] += 1
        if _cur["active"] and out is not None:
            _cur["evals"] += 1
        return out

    P.search = search
    P._begin = begin
    P._evaluate_candidate = evaluate_candidate
    P._select_candidate_indices = cand
    P._hidden_state_factory = hidden_state_factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--opponent", default="mega_lucario_ex")
    ap.add_argument("--time-limit-ms", type=float, default=None)
    ap.add_argument("--mirror", action="store_true",
                    help="相手も deck.csv(同一デッキ)。共有 _get_deck() 由来の不整合を切り分ける")
    ap.add_argument("--climb-only", action="store_true",
                    help="climb 側の select のみ計測(相手側の失敗を混ぜない)")
    ap.add_argument("--out", default=str(_HERE / "_diag_pimc_failure_results.json"))
    args = ap.parse_args()

    _install()
    cfg = agents.load_config_copy("abl_5_full")
    cfg["policy_weights_path"] = CLIMB_WEIGHTS
    if args.time_limit_ms is not None:
        cfg["pipeline"]["time_limit_ms"] = args.time_limit_ms
        cfg["pipeline"].pop("time_budget", None)
    _climb = agents.make_ml_policy_agent(cfg)
    cfg_o = agents.load_config_copy("abl_5_full")
    cfg_o["policy_weights_path"] = (
        CLIMB_WEIGHTS if args.mirror else str(_WDIR / f"policy_weights_{args.opponent}.json")
    )
    _opp = agents.make_ml_policy_agent(cfg_o)

    # climb-only: どちらのエージェントが今動いているかで計数を切り替える。
    def climb(obs):
        _gate["on"] = True
        return _climb(obs)

    def opp(obs):
        _gate["on"] = not args.climb_only
        return _opp(obs)

    deck_c = runner.load_deck(_SUB / "deck.csv")
    deck_o = deck_c if args.mirror else runner.load_deck(_DECKDIR / args.opponent / "01.csv")

    for g in range(args.games):
        if g % 2 == 0:
            runner.play_game(climb, opp, deck_c, deck_o)
        else:
            runner.play_game(opp, climb, deck_o, deck_c)
        print(f"  game {g+1}/{args.games}", file=sys.stderr)

    reached = C["reached_candidate_stage"] or 1
    out = {
        "time_limit_ms": cfg["pipeline"]["time_limit_ms"],
        "counters": dict(C),
        "rates": {
            "returned_action_of_reached": round(C["returned_action"] / reached, 4),
            "returned_none_of_reached": round(C["returned_none"] / reached, 4),
            "zero_worlds_of_reached": round(C["zero_worlds_built"] / reached, 4),
            "zero_evals_of_reached": round(C["zero_evals"] / reached, 4),
            "begin_fail_rate": round(
                C["begin_raised"] / max(1, C["begin_raised"] + C["begin_ok"]), 4),
            "cand_eval_none_rate": round(
                C["cand_eval_none"] / max(1, C["cand_eval_none"] + C["cand_eval_ok"]), 4),
        },
        "top_errors": BEGIN_ERRORS.most_common(10),
    }
    # 先に標準出力へ(ensure_production_cwd で CWD が変わるため、書き込み失敗で結果を失わない)。
    print(json.dumps(out, ensure_ascii=False, indent=2))
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
