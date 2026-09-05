"""Phase4C: CARD select(現状まったく探索されていない約35%の意思決定)の構造診断。

`pipeline` は `SelectType.MAIN` かつ `maxCount==1` にしか適用されないため、
「どのカードをサーチ/トラッシュ/回収するか」を決める **CARD select は Policy 貪欲のまま**。
ここにカード間の関係やターン内の使用順序が集中している可能性がある。

**本実装(CARD select 探索)は行わない**。次に何を作るべきかを決めるための計測だけを行う。

測るもの:
  - 全 select 中の CARD 比率 / 1試合あたりの CARD select 回数 / 候補数
  - Policy entropy、1位と2位の margin
  - 選択肢のカードID
  - **1段 lookahead**: 各 CARD 候補を1手だけ進めて末端評価し、
      * Policy top-1 と leaf-best が入れ替わる割合(= 順位反転率)
      * 反転したときの leaf value 差
  - 選択後に **増えた/消えた合法手**(カード選択が後続の選択肢集合をどれだけ変えるか)
  - 同一ターン内の後続 select 数 / ターン終了までの残り select 数
  - 分岐数の見積り: 1段 = 候補数、ターン終了まで = 候補数 × 後続 select の平均候補数^残り数

対戦挙動は変えない(観測のみ。`_select_action` の戻り値には触れない)。
出力: _diag_card_select_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

ROWS: list[dict] = []
TURN_TRACK: list[dict] = []          # 全 select の型/ターンを記録(比率・後続数の算出用)
_ctx = {"game": -1, "recording": False, "me": 0, "probed": 0, "seen_card": 0}
OPTS = {"every": 3, "worlds": 4, "max_candidates": 12, "time_ms": 6000, "max_per_game": 10}


def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def _legal_signature(obs) -> set:
    """その局面の合法手集合を (option type, card id, number) の集合で表す。"""
    sel = obs.select
    if sel is None or not sel.option:
        return set()
    sig = set()
    for o in sel.option:
        sig.add((getattr(o.type, "name", str(o.type)),
                 getattr(o, "cardId", None), getattr(o, "number", None)))
    return sig


def _probe_card_select(obs, model) -> dict | None:
    """CARD select で各候補を1手進め、末端評価と後続合法手の変化を測る。"""
    select = obs.select
    state = obs.current
    me = state.yourIndex
    try:
        scores = model.score_options(obs, None, None)
    except Exception:  # noqa: BLE001
        return None
    if not scores or len(scores) != len(select.option):
        return None

    probs = P._softmax(scores)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    cand = ranked[: OPTS["max_candidates"]]
    margin = (probs[ranked[0]] - probs[ranked[1]]) if len(ranked) > 1 else 1.0

    card_ids = []
    for i in cand:
        o = select.option[i]
        card_ids.append(getattr(o, "cardId", None))

    factory = lambda: search_adapter.to_search_begin_kwargs(  # noqa: E731
        match_context.get_own_state(me), match_context.get_opponent_state(me), obs)
    evaluator = leaf_eval_module.build_evaluator({"kind": "handcrafted"})
    deadline = time.perf_counter() + OPTS["time_ms"] / 1000.0

    # 2種類の末端評価を取る:
    #   immediate = カード選択直後の盤面をそのまま評価(= 現行 leaf をそのまま当てた場合)
    #   after_turn = 選択後、自ターンの残りを Policy 貪欲で進めてから評価(pipeline と同じ流儀)
    # 手作り leaf は手札の中身を見ないので immediate は原理的にほぼ動かないはず。
    # 「CARD select に浅い探索を足す価値」があるかは after_turn の差で判断する。
    agg: dict[int, list[float]] = {i: [] for i in cand}
    agg_turn: dict[int, list[float]] = {i: [] for i in cand}
    legal_after: dict[int, set] = {}
    n_after_options: dict[int, list[int]] = {i: [] for i in cand}

    try:
        for _ in range(OPTS["worlds"]):
            if time.perf_counter() > deadline:
                break
            hs = factory()
            if hs is None:
                continue
            try:
                root = P._begin(obs, hs)
            except Exception:  # noqa: BLE001
                continue
            try:
                for i in cand:
                    if time.perf_counter() > deadline:
                        break
                    try:
                        child = P.cg_api.search_step(root.searchId, [i])
                    except ValueError:
                        continue
                    try:
                        cobs = child.observation
                        cstate = cobs.current
                        if cstate is not None:
                            agg[i].append(evaluator.evaluate(cstate, me))
                        if i not in legal_after:
                            legal_after[i] = _legal_signature(cobs)
                        if cobs.select is not None and cobs.select.option:
                            n_after_options[i].append(len(cobs.select.option))
                        # 自ターンの残りを進めてから評価(pipeline の _rollout_and_eval を流用)
                        try:
                            v = P._rollout_and_eval(
                                child, me, {"opponent_depth": 1, "max_rollout_steps": 40},
                                deadline, evaluator, model)
                            if v is not None:
                                agg_turn[i].append(v)
                        except Exception:  # noqa: BLE001
                            pass
                    finally:
                        try:
                            P.cg_api.search_release(child.searchId)
                        except Exception:  # noqa: BLE001
                            pass
            finally:
                try:
                    P.cg_api.search_release(root.searchId)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        try:
            P.cg_api.search_end()
        except Exception:  # noqa: BLE001
            pass

    n_common = min((len(v) for v in agg.values()), default=0)
    if n_common < 1:
        return None
    leaf = {i: statistics.mean(agg[i][:n_common]) for i in cand if agg[i]}
    if not leaf:
        return None
    leaf_best = max(leaf, key=lambda i: leaf[i])
    top1 = ranked[0]

    tn = min((len(v) for v in agg_turn.values() if v), default=0)
    turn_leaf = {i: statistics.mean(agg_turn[i][:tn]) for i in cand if len(agg_turn[i]) >= tn > 0}
    turn_best = max(turn_leaf, key=lambda i: turn_leaf[i]) if turn_leaf else None

    # 後続合法手の変化: 候補ごとの合法手集合が互いにどれだけ違うか(和集合 - 積集合)。
    sigs = [s for s in legal_after.values() if s]
    changed_frac = None
    if len(sigs) >= 2:
        union = set().union(*sigs)
        inter = set(sigs[0]).intersection(*sigs[1:])
        changed_frac = (len(union) - len(inter)) / len(union) if union else 0.0

    return {
        "game": _ctx["game"],
        "turn": int(getattr(state, "turn", 0) or 0),
        "n_options": len(select.option),
        "n_candidates": len(cand),
        "policy_entropy": round(_entropy(probs), 4),
        "policy_margin_top1_top2": round(margin, 4),
        "top1_prob": round(probs[top1], 4),
        "card_ids": card_ids,
        "leaf_best_policy_rank": ranked.index(leaf_best) + 1,
        "rank_inverted": leaf_best != top1,
        "leaf_gap_best_minus_top1": round(leaf[leaf_best] - leaf.get(top1, leaf[leaf_best]), 5),
        "leaf_spread": round(max(leaf.values()) - min(leaf.values()), 5),
        # ターン終端まで進めてからの評価(こちらが「浅い探索の価値」の本体)
        "turn_best_policy_rank": (ranked.index(turn_best) + 1) if turn_best is not None else None,
        "turn_rank_inverted": (turn_best != top1) if turn_best is not None else None,
        "turn_gap_best_minus_top1": (
            round(turn_leaf[turn_best] - turn_leaf.get(top1, turn_leaf[turn_best]), 5)
            if turn_best is not None else None),
        "turn_leaf_spread": (round(max(turn_leaf.values()) - min(turn_leaf.values()), 5)
                             if turn_leaf else None),
        "turn_worlds_used": tn,
        "mean_next_n_options": (
            round(statistics.mean([v for lst in n_after_options.values() for v in lst]), 2)
            if any(n_after_options.values()) else None),
        "legal_set_changed_frac": round(changed_frac, 4) if changed_frac is not None else None,
        "worlds_used": n_common,
    }


def _install(model_getter) -> None:
    orig = ml_policy_agent._select_action

    def select_action(obs, config=None):
        if not _ctx["recording"] or obs.current is None:
            return orig(obs, config=config)
        sel = obs.select
        is_mine = obs.current.yourIndex == _ctx["me"]
        if is_mine and sel is not None:
            TURN_TRACK.append({
                "game": _ctx["game"],
                "turn": int(getattr(obs.current, "turn", 0) or 0),
                "type": (getattr(sel.type, "name", None)
                         or SelectType(int(sel.type)).name),
                "n_options": len(sel.option) if sel.option else 0,
            })
        if (is_mine and sel is not None and sel.option
                and (getattr(sel.type, "name", None) or SelectType(int(sel.type)).name) == "CARD"
                and len(sel.option) >= 2):
            _ctx["seen_card"] += 1
            if (_ctx["seen_card"] % OPTS["every"] == 0
                    and _ctx["probed"] < OPTS["max_per_game"]):
                try:
                    row = _probe_card_select(obs, model_getter(config))
                    if row is not None:
                        ROWS.append(row)
                        _ctx["probed"] += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[probe-err] {exc}", file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = select_action


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--opponent", default="mega_lucario_ex")
    ap.add_argument("--every", type=int, default=OPTS["every"])
    ap.add_argument("--worlds", type=int, default=OPTS["worlds"])
    ap.add_argument("--max-per-game", type=int, default=OPTS["max_per_game"])
    ap.add_argument("--out", default="_diag_card_select_results.json")
    args = ap.parse_args()
    OPTS.update(every=args.every, worlds=args.worlds, max_per_game=args.max_per_game)

    _install(lambda cfg: ml_policy_agent._get_model(cfg))
    cfg = agents.load_config_copy("climb_baseline")
    cfg["policy_weights_path"] = CLIMB
    climb = agents.make_ml_policy_agent(cfg)
    cfg_o = agents.load_config_copy("climb_baseline")
    cfg_o["policy_weights_path"] = str(_WDIR / f"policy_weights_{args.opponent}.json")
    opp = agents.make_ml_policy_agent(cfg_o)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    deck_o = runner.load_deck(_DECKDIR / args.opponent / "01.csv")

    t0 = time.perf_counter()
    outcomes = {}
    for g in range(args.games):
        p0 = (g % 2 == 0)
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, probed=0)
        res = (runner.play_game(climb, opp, deck_c, deck_o) if p0
               else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        outcomes[g] = (1 if res.winner == _ctx["me"] else 0) if not res.error else -1
        print(f"  game {g+1}/{args.games} probes={_ctx['probed']} total={len(ROWS)}",
              file=sys.stderr, flush=True)

    # --- select 構成比(自分の手番のみ) ---
    by_type = Counter(t["type"] for t in TURN_TRACK)
    total_sel = len(TURN_TRACK)
    card_rows = [t for t in TURN_TRACK if t["type"] == "CARD"]
    per_game_card = Counter(t["game"] for t in card_rows)

    n = len(ROWS)
    inverted = [r for r in ROWS if r["rank_inverted"]]
    ent = [r["policy_entropy"] for r in ROWS]
    branch1 = [r["n_options"] for r in ROWS]
    # ターン終了までの推定分岐: 候補数 × (後続selectの平均候補数)^(そのターンの残りselect数)
    turn_remaining = []
    for r in ROWS:
        same_turn = [t for t in TURN_TRACK
                     if t["game"] == r["game"] and t["turn"] == r["turn"]]
        turn_remaining.append(max(0, len(same_turn) - 1))
    est_turn_branch = []
    for r, rem in zip(ROWS, turn_remaining):
        nxt = r["mean_next_n_options"] or 1.0
        est_turn_branch.append(r["n_options"] * (nxt ** min(rem, 6)))

    out = {
        "note": "CARD select の構造診断。探索の本実装はしていない(観測のみ)。",
        "games": args.games, "opponent": args.opponent, "settings": dict(OPTS),
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "select_mix_own_turns": {
            "total_selects": total_sel,
            "by_type": dict(by_type.most_common()),
            "card_fraction": round(len(card_rows) / total_sel, 4) if total_sel else None,
            "card_selects_per_game": round(statistics.mean(per_game_card.values()), 2)
            if per_game_card else 0,
            "card_mean_n_options": round(statistics.mean([t["n_options"] for t in card_rows]), 2)
            if card_rows else None,
        },
        "probed": {
            "n_probes": n,
            "mean_n_options": round(statistics.mean(branch1), 2) if n else None,
            "mean_policy_entropy": round(statistics.mean(ent), 4) if n else None,
            "mean_margin_top1_top2": round(
                statistics.mean([r["policy_margin_top1_top2"] for r in ROWS]), 4) if n else None,
            "rank_inversion_rate": round(len(inverted) / n, 4) if n else None,
            "leaf_gap_when_inverted": {
                "mean": round(statistics.mean([r["leaf_gap_best_minus_top1"] for r in inverted]), 5)
                if inverted else None,
                "max": round(max([r["leaf_gap_best_minus_top1"] for r in inverted]), 5)
                if inverted else None,
                "n_gap_above_0.02": sum(
                    1 for r in inverted if r["leaf_gap_best_minus_top1"] > 0.02),
            },
            "mean_leaf_spread_immediate": round(
                statistics.mean([r["leaf_spread"] for r in ROWS]), 5) if n else None,
            "turn_rollout": {
                "n": sum(1 for r in ROWS if r.get("turn_rank_inverted") is not None),
                "rank_inversion_rate": (round(
                    sum(1 for r in ROWS if r.get("turn_rank_inverted")) /
                    sum(1 for r in ROWS if r.get("turn_rank_inverted") is not None), 4)
                    if any(r.get("turn_rank_inverted") is not None for r in ROWS) else None),
                "mean_spread": (round(statistics.mean(
                    [r["turn_leaf_spread"] for r in ROWS if r.get("turn_leaf_spread") is not None]), 5)
                    if any(r.get("turn_leaf_spread") is not None for r in ROWS) else None),
                "mean_gap_when_inverted": (round(statistics.mean(
                    [r["turn_gap_best_minus_top1"] for r in ROWS if r.get("turn_rank_inverted")]), 5)
                    if any(r.get("turn_rank_inverted") for r in ROWS) else None),
                "n_gap_above_0.02": sum(
                    1 for r in ROWS
                    if r.get("turn_rank_inverted") and (r.get("turn_gap_best_minus_top1") or 0) > 0.02),
            },
            "legal_set_changed_frac": {
                "mean": round(statistics.mean(
                    [r["legal_set_changed_frac"] for r in ROWS
                     if r["legal_set_changed_frac"] is not None]), 4)
                if any(r["legal_set_changed_frac"] is not None for r in ROWS) else None,
                "frac_probes_with_any_change": round(sum(
                    1 for r in ROWS
                    if (r["legal_set_changed_frac"] or 0) > 0) / n, 4) if n else None,
            },
            "mean_next_n_options": round(statistics.mean(
                [r["mean_next_n_options"] for r in ROWS if r["mean_next_n_options"]]), 2)
            if any(r["mean_next_n_options"] for r in ROWS) else None,
        },
        "branching_estimates": {
            "one_ply_mean": round(statistics.mean(branch1), 2) if n else None,
            "turn_end_mean": round(statistics.mean(est_turn_branch), 1) if est_turn_branch else None,
            "turn_end_median": round(statistics.median(est_turn_branch), 1)
            if est_turn_branch else None,
            "mean_remaining_selects_in_turn": round(statistics.mean(turn_remaining), 2)
            if turn_remaining else None,
        },
        "by_entropy_band": {},
        "rows": ROWS,
    }
    for lo, hi, name in ((0.0, 0.5, "low"), (0.5, 1.2, "mid"), (1.2, 99.0, "high")):
        sel = [r for r in ROWS if lo <= r["policy_entropy"] < hi]
        out["by_entropy_band"][name] = {
            "n": len(sel),
            "inversion_rate": round(sum(1 for r in sel if r["rank_inverted"]) / len(sel), 4)
            if sel else None,
        }

    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    print(json.dumps({k: v for k, v in out.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
