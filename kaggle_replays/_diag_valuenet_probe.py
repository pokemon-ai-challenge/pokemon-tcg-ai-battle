#!/usr/bin/env python3
"""読み取り専用の診断分析(コード変更なし・一時スクリプト)。

pimc-null-diagnosis-implementation-plan.md Step1: value網の摂動probe。

自チーム("MORIOKA Tsuoi")の既存リプレイ(kaggle_replays/replays/ に既に大量保存済み)から
中盤〜終盤の実局面を多数サンプルし、各局面で以下を行う:

1. 単一特徴の摂動(非現実的盤面になる前提で、純粋な単調性チェックとして):
   - self_deck_count を実測値 -> 0 へ下げる(期待: 勝率が下がる)
   - self_prize_remaining を実測値から減らす(期待: 上がる)
   - opp_prize_remaining を実測値から減らす(期待: 下がる)
   - prize_diff を「自分に有利な方向」(自分の残りサイド減 / 相手の残りサイド増と同じ符号)へ
     動かす(期待: 上がる)。prize_diff = your_prize_remaining - opponent_prize_remaining
     (board_features.prize_diff の実装)であり、残りサイドが少ない方が有利という通常のTCG
     ルールに従うと、この値が「有利な方向」に動くとは "減る" ことを意味する
     (board_features.prize_diff のdocstring "正なら自分が有利" とは符号が逆に見えるが、
     本probeでは実際のゲームルール(残りサイドが少ない=有利)を優先して判定する)。

2. 実トラジェクトリでの相関: 各対戦を通じて (turn, self_deck_count, predict_win_prob) を
   記録し、試合内で self_deck_count が減っていくにつれ predict_win_prob がどう動くかを見る
   (山札切れで負けた試合と、それ以外の試合を分けて集計)。

このスクリプトは ptcg_ai / league / configs を一切変更しない。分析専用。
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SUB = REPO_ROOT / "sample_submission"
KAGGLE_REPLAYS_DIR = Path(__file__).resolve().parent
REPLAYS_DIR = KAGGLE_REPLAYS_DIR / "replays"
MASTER_INDEX = KAGGLE_REPLAYS_DIR / "index" / "episodes_master.jsonl"

sys.path.insert(0, str(SAMPLE_SUB))
sys.path.insert(0, str(KAGGLE_REPLAYS_DIR))

from cg.api import to_observation_class  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
from ptcg_ai.learning.value_model import ValueModel  # noqa: E402

TEAM_NAME = "MORIOKA Tsuoi"

# 摂動対象の特徴とそのインデックス。
_IDX = {name: encoder.FEATURE_NAMES.index(name) for name in
        ("self_deck_count", "self_prize_remaining", "opp_prize_remaining", "prize_diff")}

MIN_TURN = 4  # 中盤〜終盤のみ対象(序盤は情報が乏しく摂動の意味が薄い)。
MAX_STATES = 200
MAX_EPISODES_SCANNED = 400


def find_own_episode_ids(limit: int) -> list[str]:
    """episodes_master.jsonl から自チームが参加したエピソードIDを集める。"""
    ids: list[str] = []
    seen: set[str] = set()
    with MASTER_INDEX.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            eid = row["episode_id"]
            if eid in seen:
                continue
            players = row.get("players", [])
            if any(p.get("team_name") == TEAM_NAME for p in players):
                path = REPLAYS_DIR / f"episode-{eid}-replay.json"
                if path.exists():
                    ids.append(eid)
                    seen.add(eid)
            if len(ids) >= limit:
                break
    return ids


def load_my_idx(replay: dict) -> int | None:
    team_names = replay.get("info", {}).get("TeamNames", [None, None])
    if TEAM_NAME not in team_names:
        return None
    return team_names.index(TEAM_NAME)


def iter_my_states(replay: dict, my_idx: int):
    """自分視点の (turn, State) を、obs.select is not None な決定点ごとに返す。"""
    steps = replay["steps"]
    for i in range(len(steps) - 1):
        agent_step = steps[i][my_idx]
        if agent_step.get("status") != "ACTIVE":
            continue
        obs_dict = agent_step.get("observation")
        if not obs_dict:
            continue
        try:
            obs = to_observation_class(obs_dict)
        except Exception:
            continue
        if obs.current is None:
            continue
        yield i, obs.current


def game_outcome(replay: dict, my_idx: int) -> str:
    rewards = replay.get("rewards", [None, None])
    opp_idx = 1 - my_idx
    if rewards[my_idx] is None or rewards[opp_idx] is None:
        return "unknown"
    if rewards[my_idx] > rewards[opp_idx]:
        return "win"
    if rewards[my_idx] < rewards[opp_idx]:
        return "loss"
    return "draw"


def loss_cause(replay: dict, my_idx: int) -> str | None:
    """自分が負けた試合の敗因を簡易分類する。

    注意: このリプレイ形式は最終手の"current"スナップショットで State.result が
    常に-1のまま(終局を反映しない既知の癖)であり、正確な最終盤面を厳密には
    再構成できない。そのため本関数は「自分(敗者)の deckCount が試合を通じて
    一度でも0を記録したか」で "deck_out" を判定する簡易版に留める
    (0を記録した = その時点でドローを試みれば山札切れが起きる状態に達していた
    ことの直接証拠。既知の癖の影響を受けない)。0を記録しなければ "not_deck_out"
    (bench壊滅・通常のサイド完投・その他はここでは区別しない。Step4はこの
    deck_out 判定を前提に、その原因の内訳を別途掘り下げる)。
    """
    my_deck_counts = []
    for i, state in iter_my_states(replay, my_idx):
        me = state.yourIndex
        my_deck_counts.append(state.players[me].deckCount)
    if not my_deck_counts:
        return None
    return "deck_out" if min(my_deck_counts) == 0 else "not_deck_out"


def perturb_and_score(model: ValueModel, state) -> dict:
    base_features = encoder.encode_state_from_state(state)
    base_prob = model.predict_win_prob_from_features(base_features, state.turn)

    results = {"base_prob": base_prob, "turn": state.turn}

    # self_deck_count: 実測値 -> 0
    f = list(base_features)
    f[_IDX["self_deck_count"]] = 0.0
    results["deck_to_zero_prob"] = model.predict_win_prob_from_features(f, state.turn)
    results["deck_to_zero_delta"] = results["deck_to_zero_prob"] - base_prob
    results["self_deck_count_actual"] = base_features[_IDX["self_deck_count"]]

    # self_prize_remaining: 実測値から2枚減(下限0)
    f = list(base_features)
    actual_self_prize = base_features[_IDX["self_prize_remaining"]]
    f[_IDX["self_prize_remaining"]] = max(0.0, actual_self_prize - 2.0)
    results["self_prize_down_prob"] = model.predict_win_prob_from_features(f, state.turn)
    results["self_prize_down_delta"] = results["self_prize_down_prob"] - base_prob
    results["self_prize_actual"] = actual_self_prize

    # opp_prize_remaining: 実測値から2枚減(下限0)
    f = list(base_features)
    actual_opp_prize = base_features[_IDX["opp_prize_remaining"]]
    f[_IDX["opp_prize_remaining"]] = max(0.0, actual_opp_prize - 2.0)
    results["opp_prize_down_prob"] = model.predict_win_prob_from_features(f, state.turn)
    results["opp_prize_down_delta"] = results["opp_prize_down_prob"] - base_prob
    results["opp_prize_actual"] = actual_opp_prize

    # prize_diff: 自分に有利な方向(値を下げる。上のprize_diff実装の符号に基づく)へ2動かす
    f = list(base_features)
    f[_IDX["prize_diff"]] = base_features[_IDX["prize_diff"]] - 2.0
    results["prize_diff_favorable_prob"] = model.predict_win_prob_from_features(f, state.turn)
    results["prize_diff_favorable_delta"] = results["prize_diff_favorable_prob"] - base_prob
    results["prize_diff_actual"] = base_features[_IDX["prize_diff"]]

    return results


def main() -> None:
    model = ValueModel()
    print(f"ValueModel.is_ready = {model.is_ready}")
    if not model.is_ready:
        print("警告: value_weights.json が読み込めていません。摂動probeは全て0.5になり無意味です。中断します。")
        return

    episode_ids = find_own_episode_ids(MAX_EPISODES_SCANNED)
    print(f"自チームエピソード候補: {len(episode_ids)}件(上限{MAX_EPISODES_SCANNED}件スキャン)")

    sampled_states = []  # list[(episode_id, turn, State)]
    trajectories = {}  # episode_id -> list[(turn, self_deck_count, win_prob)]
    outcomes = {}  # episode_id -> "win"/"loss"/"draw"/"unknown"
    causes = {}  # episode_id -> loss_cause (負け試合のみ)

    for eid in episode_ids:
        if len(sampled_states) >= MAX_STATES and len(trajectories) >= 150:
            break
        path = REPLAYS_DIR / f"episode-{eid}-replay.json"
        with path.open(encoding="utf-8") as f:
            replay = json.load(f)
        my_idx = load_my_idx(replay)
        if my_idx is None:
            continue

        outcome = game_outcome(replay, my_idx)
        outcomes[eid] = outcome
        if outcome == "loss":
            cause = loss_cause(replay, my_idx)
            if cause is not None:
                causes[eid] = cause

        traj = []
        states_this_game = list(iter_my_states(replay, my_idx))
        # 参照過多にならないよう、この試合からは最大2局面だけ摂動probe用にサンプル
        picked = 0
        for i, state in states_this_game:
            f_deck = encoder.encode_state_from_state(state)[_IDX["self_deck_count"]]
            prob = model.predict_win_prob_from_state(state)
            traj.append((state.turn, f_deck, prob))
            if state.turn >= MIN_TURN and picked < 2 and len(sampled_states) < MAX_STATES:
                sampled_states.append((eid, state.turn, state))
                picked += 1
        if traj:
            trajectories[eid] = traj

    print(f"摂動probe対象局面: {len(sampled_states)}件")
    print(f"トラジェクトリ記録試合数: {len(trajectories)}件")
    print(f"内訳: {json.dumps({k: sum(1 for v in outcomes.values() if v == k) for k in set(outcomes.values())}, ensure_ascii=False)}")
    print(f"負け試合の敗因分類(このスクリプト内の簡易版, 対象={len(causes)}件): "
          f"{json.dumps({k: sum(1 for v in causes.values() if v == k) for k in set(causes.values())}, ensure_ascii=False)}")

    all_results = [perturb_and_score(model, state) for (_eid, _turn, state) in sampled_states]

    def summarize(delta_key: str, expect_sign: str) -> dict:
        deltas = [r[delta_key] for r in all_results]
        mean_delta = statistics.mean(deltas) if deltas else float("nan")
        if expect_sign == "down":
            n_correct = sum(1 for d in deltas if d < -1e-9)
        else:
            n_correct = sum(1 for d in deltas if d > 1e-9)
        n_flat = sum(1 for d in deltas if abs(d) <= 1e-9)
        return {
            "mean_delta": mean_delta,
            "n": len(deltas),
            "n_correct_direction": n_correct,
            "n_flat": n_flat,
            "expected": expect_sign,
        }

    summary = {
        "self_deck_count(high->0), expect win_prob down": summarize("deck_to_zero_delta", "down"),
        "self_prize_remaining(-2), expect win_prob up": summarize("self_prize_down_delta", "up"),
        "opp_prize_remaining(-2), expect win_prob down": summarize("opp_prize_down_delta", "down"),
        "prize_diff(favorable direction, -2), expect win_prob up": summarize("prize_diff_favorable_delta", "up"),
    }

    print("\n=== 摂動probe サマリ ===")
    for key, s in summary.items():
        print(f"{key}:")
        print(f"  mean_delta={s['mean_delta']:+.4f}  n={s['n']}  "
              f"correct_direction={s['n_correct_direction']}/{s['n']}  flat={s['n_flat']}")

    # 実トラジェクトリでの相関(試合ごとの Pearson相関: self_deck_count vs win_prob)
    print("\n=== 実トラジェクトリ相関(試合ごと self_deck_count vs win_prob) ===")
    corr_by_outcome: dict[str, list[float]] = {"win": [], "loss": [], "draw": [], "unknown": []}
    corr_deckout: list[float] = []
    for eid, traj in trajectories.items():
        if len(traj) < 4:
            continue
        deck_counts = [t[1] for t in traj]
        probs = [t[2] for t in traj]
        if len(set(deck_counts)) < 2:
            continue
        try:
            corr = statistics.correlation(deck_counts, probs)
        except Exception:
            continue
        outcome = outcomes.get(eid, "unknown")
        corr_by_outcome.setdefault(outcome, []).append(corr)
        if causes.get(eid) in ("deck_out", "no_pokemon_and_deck_out"):
            corr_deckout.append(corr)

    for outcome, corrs in corr_by_outcome.items():
        if corrs:
            print(f"  outcome={outcome}: n={len(corrs)}, mean_corr(deck_count vs win_prob)={statistics.mean(corrs):+.3f}")
    if corr_deckout:
        print(f"  loss_cause=deck_out(のみ): n={len(corr_deckout)}, mean_corr={statistics.mean(corr_deckout):+.3f}")
    else:
        print("  loss_cause=deck_out(のみ): サンプルなし(このスキャン範囲内で山札切れ負けが見つからず)")

    out_path = KAGGLE_REPLAYS_DIR / "_diag_valuenet_probe_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "n_sampled_states": len(sampled_states),
            "n_trajectories": len(trajectories),
            "outcomes": {k: sum(1 for v in outcomes.values() if v == k) for k in set(outcomes.values())},
            "loss_causes": {k: sum(1 for v in causes.values() if v == k) for k in set(causes.values())},
            "corr_by_outcome": {k: v for k, v in corr_by_outcome.items() if v},
            "corr_deckout": corr_deckout,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nwrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
