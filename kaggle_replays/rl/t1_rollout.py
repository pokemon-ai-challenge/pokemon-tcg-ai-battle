"""T1(Transformer)自身によるon-policy rollout収集(T1-PPO用)。

**既存``collect_parallel.py``/``learner.py``/``PolicyModel``は変更しない。**
T1自身の分布からsoftmaxサンプリングして行動を選び、``old_logp``(サンプリング時の
T1自身の対数確率)を記録する。蒸留shardの``old_logp``(教師の分布)とは別物で、
これがPPOの重要度比に使える唯一の``old_logp``(design.md §8.3参照)。

mixed-opponent sampling: 重み付きで対戦相手を選ぶ(PPO pilotでは
{現在方策の凍結コピー, 初期/過去T1 snapshot, v40, rocket_mewtwo_ex, 残り7相手}の
指定比率)。相手側は``select_greedy``を持つ統一interfaceでPolicyModel/T1どちらも扱う
(``PMOpponent``/``T1Opponent``)。相手の方策はrollout中は凍結し、更新しない。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import t1_live_agent as la  # noqa: E402

MAX_STEPS = 3000


def _t1_forward_scores(model, vocab, t1_state_pm, cur, select, profile_name, device):
    """t1_live_agent.t1_scoresと同じだが、arraysも一緒に返す(rollout保存用)。"""
    arrays = la.build_single_decision_arrays(t1_state_pm, cur, select, profile_name)
    import token_batch as tb
    batch = tb.build_batch(arrays, vocab)
    inputs = la._to_tensors(batch, device)
    model.eval()
    with torch.no_grad():
        scores = model(**inputs)[0]
    n_opt = int(arrays["counts"][0])
    return arrays, scores[:n_opt].cpu().numpy()


def _sample_index(scores: np.ndarray, rng: random.Random) -> tuple[int, float]:
    m = scores.max()
    exps = np.exp(scores - m)
    probs = exps / exps.sum()
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i, float(np.log(max(probs[i], 1e-12)))
    return len(probs) - 1, float(np.log(max(probs[-1], 1e-12)))


def _greedy_action(pm, cur, select) -> list[int]:
    scores = pm.score_options_from_state(cur, select)
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))
    return sorted(range(n), key=lambda i: scores[i], reverse=True)[:count]


class PMOpponent:
    """既存PolicyModel(JSON重み)をそのまま相手として使う(argmax固定)。"""

    def __init__(self, pm, opp_id: str):
        self.pm = pm
        self.id = opp_id

    def select_greedy(self, cur, select) -> list[int]:
        return _greedy_action(self.pm, cur, select)


class RuleOpponent:
    """ルールベース戦略(``opponents.rule_agents``の``Strategy``、``framework.choose``で
    全選択肢に点数をつける方式)を相手として使う。オーロンゲ等、ルールベース教師を
    使うデッキ用(``framework.choose``自体が既にargmax、追加のsamplingは無い)。

    ``framework.build_ctx(obs, deck)``は``obs.current``/``obs.select``しか使わないため、
    既存の``select_greedy(cur, select)``インターフェース(PMOpponent/T1Opponentと共通)を
    変えずに、その場で最小限のnamespaceを組み立てて渡す。"""

    def __init__(self, strategy, deck, opp_id: str):
        self.strategy = strategy
        self.deck = deck
        self.id = opp_id

    def select_greedy(self, cur, select) -> list[int]:
        import types
        from opponents.rule_agents import framework as fw
        obs = types.SimpleNamespace(current=cur, select=select)
        ctx = fw.build_ctx(obs, self.deck)
        self.strategy.prepare(ctx)
        ctx.memo["attack_evals"] = fw.evaluate_attacks(ctx, bonus=self.strategy.attack_bonus(ctx))
        return fw.choose(ctx, self.strategy)


class T1Opponent:
    """凍結したT1モデル(現在方策のスナップショット、または初期/過去checkpoint)を相手として使う。
    ``model``は呼び出し側が凍結(``.eval()``・grad不要)を保証すること。argmax固定
    (既存PMOpponentと同じ「相手はgreedy」という慣習に揃える)。"""

    def __init__(self, model, vocab, t1_state_pm, profile_name: str, device: str, opp_id: str):
        self.model = model
        self.vocab = vocab
        self.t1_state_pm = t1_state_pm
        self.profile_name = profile_name
        self.device = device
        self.id = opp_id

    def select_greedy(self, cur, select) -> list[int]:
        if select.maxCount != 1:
            # 複数選択は蒸留と同じ慣習でt1_state_pm(v40)のgreedyにフォールバック
            # (T1は単一選択決定点しかモデル化していないため)。
            return _greedy_action(self.t1_state_pm, cur, select)
        _, scores = _t1_forward_scores(self.model, self.vocab, self.t1_state_pm, cur, select,
                                       self.profile_name, self.device)
        return [int(np.argmax(scores))]


def play_one_rollout_game(model, vocab, t1_state_pm, opponent, deck_t1, deck_opp,
                          t1_seats: set[int], seed: int, profile_name: str,
                          device: str = "cpu") -> dict:
    """1試合。``t1_seats``={0}または{1}なら対opponent、{0,1}ならT1自己対戦(両座席T1)。
    ``opponent``はT1が制御しない座席の行動選択に使う(``select_greedy(cur, select)``を
    持つオブジェクト、``PMOpponent``/``T1Opponent``)。``t1_seats=={0,1}``のときは未使用
    (``None``でよい)。

    Returns: {"decisions": [...], "error", "winner"}. 各decisionは
    {"arrays"(単一決定点のtoken batch用dict), "chosen": int, "old_logp": float, "seat": int}。
    """
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    rng = random.Random(seed)
    deck0 = deck_t1 if 0 in t1_seats else deck_opp
    deck1 = deck_t1 if 1 in t1_seats else deck_opp
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        return {"decisions": [], "error": f"start {start_data.errorType}", "winner": None}

    decisions = []
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
                winner = cur.result
                break
            if n >= MAX_STEPS:
                error = "max_steps"; break

            select = obs.select
            if select is not None and select.option:
                if cur.yourIndex in t1_seats and select.maxCount == 1:
                    arrays, scores = _t1_forward_scores(model, vocab, t1_state_pm, cur, select,
                                                        profile_name, device)
                    idx, logp = _sample_index(scores, rng)
                    decisions.append({"arrays": arrays, "chosen": idx, "old_logp": logp,
                                      "seat": cur.yourIndex})
                    action = [idx]
                elif cur.yourIndex in t1_seats:
                    action = _greedy_action(t1_state_pm, cur, select)
                else:
                    action = opponent.select_greedy(cur, select)
            else:
                action = []
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        battle_finish()

    return {"decisions": decisions, "error": error, "winner": winner}


def concat_single_decision_arrays(arrays_list: list[dict]) -> dict:
    """``build_single_decision_arrays``が返す単一決定点dict(n=1)のリストを、
    ``token_batch.build_batch``に渡せる複数決定点の``arrays``dict(n=len)にまとめる。"""
    def _cat1d(key):
        return np.concatenate([a[key] for a in arrays_list])

    def _catrows(key, empty_dim):
        rows = [a[key] for a in arrays_list if len(a[key])]
        if not rows:
            return np.zeros((0, empty_dim), dtype=np.float32)
        return np.concatenate(rows, axis=0)

    return {
        "board_counts": _cat1d("board_counts"),
        "counts": _cat1d("counts"),
        "board_token_numeric_features": _catrows("board_token_numeric_features", 11),
        "board_token_card_ids": _cat1d("board_token_card_ids"),
        "board_token_zone_ids": _cat1d("board_token_zone_ids"),
        "legacy_option_features": _catrows("legacy_option_features", 65),
        "option_card_ids": _cat1d("option_card_ids"),
        "teacher_logits": _cat1d("teacher_logits"),
        "option_target_token_indices": _cat1d("option_target_token_indices"),
        "chosen": _cat1d("chosen"),
        "legacy_global_features": np.concatenate(
            [a["legacy_global_features"] for a in arrays_list], axis=0),
    }


def collect_rollout(model, vocab, t1_state_pm, weighted_specs: list[dict], deck_t1, n_games: int,
                    seed0: int, profile_name: str, device: str = "cpu") -> tuple[list[dict], dict]:
    """重み付きmixed-opponent rollout収集。

    ``weighted_specs``: ``[{"id", "weight", "opponent", "deck"}, ...]``。
    ``opponent=None``は自己対戦(両座席T1、``deck``は無視)を意味する。それ以外は
    ``PMOpponent``/``T1Opponent``。試合ごとに``random.choices``で重みに従って1つ選ぶ。
    座席(先攻/後攻)は対opponent戦のみ``g % 2``で均等化(自己対戦は両座席がT1)。

    Returns: (trajectories, counts)。``trajectories``の各要素は
    ``{"decisions", "reward"(+1/-1), "opponent_id"}``。``counts``は
    ``{id: {"attempts": n, "valid": n}}``(エラー試合はvalidに含めない)。
    """
    rng = random.Random(seed0)
    trajectories = []
    counts = {s["id"]: {"attempts": 0, "valid": 0} for s in weighted_specs}
    weights = [s["weight"] for s in weighted_specs]
    for g in range(n_games):
        seed = seed0 + g
        spec = rng.choices(weighted_specs, weights=weights, k=1)[0]
        counts[spec["id"]]["attempts"] += 1
        if spec["opponent"] is None:
            r = play_one_rollout_game(model, vocab, t1_state_pm, None, deck_t1, deck_t1,
                                      {0, 1}, seed, profile_name, device=device)
            if r["error"] is not None or r["winner"] is None:
                continue
            counts[spec["id"]]["valid"] += 1
            for seat in (0, 1):
                seat_decisions = [d for d in r["decisions"] if d["seat"] == seat]
                if seat_decisions:
                    trajectories.append({"decisions": seat_decisions,
                                         "reward": 1.0 if r["winner"] == seat else -1.0,
                                         "opponent_id": spec["id"]})
        else:
            t1_index = g % 2
            r = play_one_rollout_game(model, vocab, t1_state_pm, spec["opponent"], deck_t1,
                                      spec["deck"], {t1_index}, seed, profile_name, device=device)
            if r["error"] is not None or r["winner"] is None or not r["decisions"]:
                continue
            counts[spec["id"]]["valid"] += 1
            trajectories.append({"decisions": r["decisions"],
                                 "reward": 1.0 if r["winner"] == t1_index else -1.0,
                                 "opponent_id": spec["id"]})
    return trajectories, counts
