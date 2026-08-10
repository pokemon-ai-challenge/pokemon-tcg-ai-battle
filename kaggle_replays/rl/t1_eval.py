"""T1(Transformer student)とv40教師(既存MLP)のhead-to-head対局ループ。

**既存``PolicyModel``/``collect_parallel.py``/``learner.py``/``evaluate_pool()``は
一切変更しない。** ここは並存する新規モジュール。両者とも明示的argmaxで行動を選ぶ
(既存``evaluate_pool``の``temperature=0.01``のnear-greedyとは違う、厳密なgreedy)。

学習(蒸留)がモデル化しているのは「自分の単一選択(maxCount==1)」の決定点だけなので、
それ以外(相手番・複数選択)は常に教師MLPのargmaxで進める
(``collect_tokens.py``がスコアの重み付き貪欲でこれらを進めるのと同じ設計思想。
ここは温度サンプリングではなく厳密argmaxにしている点だけが違う)。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import t1_live_agent as la  # noqa: E402

MAX_STEPS = 3000


def _teacher_argmax_action(pm, cur, select) -> list[int]:
    scores = pm.score_options_from_state(cur, select)
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))
    return sorted(range(n), key=lambda i: scores[i], reverse=True)[:count]


def play_one_game(model, vocab, t1_state_pm, teacher_pm, deck_t1, deck_teacher, t1_index: int,
                  seed: int, profile_name: str, device: str = "cpu") -> dict:
    """1試合。``t1_index``(0 or 1)がT1側のスロット
    (§9.1.1の通り、``battle_start(deck0, deck1)``のスロット0が確定的に先攻)。

    ``t1_state_pm``: T1自身の状態エンコード(``encode_state_features``)専用。**必ず
    T1と同じprofile(fuudin_v4)のPolicyModelを渡す**(相手のprofileが違っても
    T1側の特徴抽出には影響させないため)。``teacher_pm``: 相手側の行動選択専用
    (相手固有のprofile/weightsでよい、166次元の相手でも問題ない)。

    Returns: {"error", "winner", "t1_first", "t1_win", "inference_times", "illegal_actions"}
    """
    import random
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    random.seed(seed)
    deck0, deck1 = (deck_t1, deck_teacher) if t1_index == 0 else (deck_teacher, deck_t1)
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        return {"error": f"start {start_data.errorType}", "winner": None,
               "t1_first": None, "t1_win": None, "inference_times": [], "illegal_actions": 0}

    inference_times: list[float] = []
    illegal_actions = 0
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
                if cur.yourIndex == t1_index and select.maxCount == 1:
                    t0 = time.perf_counter()
                    idx = la.t1_select_index(model, vocab, t1_state_pm, cur, select, profile_name,
                                             device=device)
                    inference_times.append(time.perf_counter() - t0)
                    if not (0 <= idx < len(select.option)):
                        illegal_actions += 1
                        idx = 0
                    action = [idx]
                elif cur.yourIndex == t1_index:
                    # T1側の複数選択決定点(蒸留がモデル化していない)は、T1と同じ
                    # profileのt1_state_pm(v40)のargmaxで進める(相手の方策に依存させない)。
                    action = _teacher_argmax_action(t1_state_pm, cur, select)
                    if any(not (0 <= i < len(select.option)) for i in action):
                        illegal_actions += 1
                        action = [0]
                else:
                    action = _teacher_argmax_action(teacher_pm, cur, select)
                    if any(not (0 <= i < len(select.option)) for i in action):
                        illegal_actions += 1
                        action = [0]
            else:
                action = []
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        battle_finish()

    t1_first = (t1_index == 0)
    t1_win = None if (error is not None or winner is None) else (winner == t1_index)
    return {"error": error, "winner": winner, "t1_first": t1_first, "t1_win": t1_win,
           "inference_times": inference_times, "illegal_actions": illegal_actions}


def run_head_to_head(model, vocab, t1_state_pm, teacher_pm, deck_t1, deck_teacher, n_games: int,
                     seed0: int, profile_name: str, device: str = "cpu") -> dict:
    """``n_games``試合、T1の座席(t1_index)を``g % 2``で均等化して対局する
    (§9.1.1と同じ均等化方式)。``t1_state_pm``/``teacher_pm``は``play_one_game``参照。"""
    results = []
    for g in range(n_games):
        t1_index = g % 2
        r = play_one_game(model, vocab, t1_state_pm, teacher_pm, deck_t1, deck_teacher, t1_index,
                          seed0 + g, profile_name, device=device)
        results.append(r)

    valid = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    wins = sum(1 for r in valid if r["t1_win"] is True)
    losses = sum(1 for r in valid if r["t1_win"] is False)
    draws = sum(1 for r in valid if r["t1_win"] is None)  # error無しでwinner不定は理論上無いが念のため
    first_games = sum(1 for r in results if r["t1_first"])
    illegal_total = sum(r["illegal_actions"] for r in results)
    all_times = [t for r in results for t in r["inference_times"]]

    return {
        "n_games": n_games, "n_valid": len(valid), "n_errors": len(errors),
        "n_draws": draws, "t1_wins": wins, "t1_losses": losses,
        "t1_winrate": (wins / len(valid)) if valid else float("nan"),
        "t1_first_games": first_games, "t1_second_games": n_games - first_games,
        "illegal_actions_total": illegal_total,
        "error_types": [r["error"] for r in errors],
        "inference_time_mean": (sum(all_times) / len(all_times)) if all_times else float("nan"),
        "inference_time_max": max(all_times) if all_times else float("nan"),
        "n_inferences": len(all_times),
        "results": results,
    }
