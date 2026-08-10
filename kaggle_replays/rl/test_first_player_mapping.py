"""learner_indexと実際の先攻/後攻の対応関係を固定するテスト(D2.1 item5/6)。

23試行(同一デッキ×2回・異なるデッキ×2回のセット)の実測から、``firstPlayer``は
常にdeck slot0(``battle_start``の第1引数)になることを確認した。``collect_tokens.py``
の``deck0, deck1 = (deck_l, deck_o) if learner_index==0 else (deck_o, deck_l)``と
組み合わせると、**learner_index=0のとき常に学習側が先攻、learner_index=1のとき
常に学習側が後攻**になる(Pythonのrandom.seedを変えても、異なるデッキの組でも
この対応は崩れなかった)。

これは「試合indexの偶奇でランダムに50/50になる」のではなく、「learner_indexで
決定的に先攻/後攻が決まり、その学習側の座席(learner_index)自体を偶奇で
alternateさせることで結果的に50/50になる」という違いがある。評価件数から
実際の先攻試合数を計算する場合は、この決定的対応を前提にできる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def env():
    try:
        from cg.api import to_observation_class
        from cg.game import battle_start, battle_select, battle_finish
        from run_league import read_deck_csv_file
        from ptcg_ai.learning.policy_model import PolicyModel
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dependencies unavailable: {exc}")

    deck_l_path = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam" / "01.csv"
    deck_o_path = (_ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
                  / "dragapult_ex" / "01.csv")
    weights = _ROOT / "sample_submission" / "ptcg_ai" / "learning" / "policy_weights.json"
    if not (deck_l_path.exists() and deck_o_path.exists() and weights.exists()):
        pytest.skip("デッキ/教師重みが無い")

    pm = PolicyModel(str(weights))
    if not pm.is_ready:
        pytest.skip("production policy_weights.json が無い")

    return {
        "to_observation_class": to_observation_class, "battle_start": battle_start,
        "battle_select": battle_select, "battle_finish": battle_finish,
        "deck_l": read_deck_csv_file(str(deck_l_path)), "deck_o": read_deck_csv_file(str(deck_o_path)),
        "pm": pm,
    }


def _play_and_get_first_player(env, learner_index: int, seed: int) -> int:
    import random
    random.seed(seed)
    deck0, deck1 = (env["deck_l"], env["deck_o"]) if learner_index == 0 else (env["deck_o"], env["deck_l"])
    obs_dict, _ = env["battle_start"](deck0, deck1)
    n = 0
    first_seen = None
    cur = None
    try:
        while True:
            obs = env["to_observation_class"](obs_dict)
            cur = obs.current
            if cur is None or cur.result != -1:
                break
            if first_seen is None and cur.firstPlayer != -1:
                first_seen = cur.firstPlayer
            sel = obs.select
            if sel is None or not sel.option:
                action = []
            elif sel.maxCount == 1:
                idx = env["pm"].select_option(obs)
                action = [idx if idx is not None else 0]
            else:
                scores = env["pm"].score_options(obs)
                nn = len(sel.option)
                cnt = max(sel.minCount, min(sel.maxCount, nn))
                action = (sorted(range(nn), key=lambda i: scores[i], reverse=True)[:cnt]
                          if scores else list(range(cnt)))
            obs_dict = env["battle_select"](action)
            n += 1
            if n > 500:
                break
    finally:
        env["battle_finish"]()
    return first_seen


def test_learner_index_deterministically_maps_to_first_player(env):
    """learner_index=0なら常に先攻(firstPlayer==learner_index)、
    learner_index=1なら常に後攻になることを、複数試合で確認する。"""
    for g in range(6):
        learner_index = g % 2
        first_player = _play_and_get_first_player(env, learner_index, seed=7000 + g)
        assert first_player == 0, (
            f"firstPlayerが0以外だった(game={g}, learner_index={learner_index}, "
            f"firstPlayer={first_player})。先攻抽選の実装が変わった可能性がある。")
        learner_went_first = (first_player == learner_index)
        assert learner_went_first == (learner_index == 0), (
            f"learner_indexと先攻/後攻の対応が崩れた(game={g}, learner_index={learner_index})")
