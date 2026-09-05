"""Exploration RNG Non-Interference（Step 1-16 指示 5）。

本番接続の必須条件として追加された:

    search の有無だけで本番の random outcome が変化しない

これは **現行 API では検証できない**。その根拠をテストとして固定する。
検証できない以上、Phase 1 は `production candidate = NOT SAFE` として扱い、
既定 config では決して有効化しない。

Step 1-15 で「search の有無で 20/20 乖離した」と報告したが、これは誤りだった。
`BattleStart` 自体が再現しないので、control 同士でも 100% 乖離する。
その事実を `test_engine_is_not_reproducible` で固定し、同じ誤りを繰り返さないようにする。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import to_observation_class
from cg.game import _get_battle_data, battle_finish, battle_select, battle_start
from cg.sim import lib
from main import read_deck_csv
from ptcg_ai.core.config import load_config
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.search.lethal.engine import EngineError, HiddenState, SearchSession

# 本番 RNG に干渉しないことを保証できるまで、Phase 1 の本番 ON は禁止。
RNG_NON_INTERFERENCE_VERIFIED = False


def _fixed_action(observation) -> list[int]:
    """Python 乱数を一切使わない固定方策。"""
    select = observation.select
    count = max(select.minCount, 0)
    return list(range(min(count, len(select.option)))) if count else []


def _random_facts(observation) -> str:
    """観測のうち**乱数で決まる部分**だけを取り出す（手札の実体と山札枚数）。"""
    state = observation.current
    if state is None:
        return "none"
    return json.dumps([
        [
            [[c.id, c.serial] for c in player.hand] if getattr(player, "hand", None) else None,
            player.deckCount,
        ]
        for player in state.players
    ])


def _opening(deck, steps: int = 4) -> list[str]:
    obs_dict, start = battle_start(list(deck), list(deck))
    if start.errorType != 0:
        return []
    facts = []
    try:
        for _ in range(steps):
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                break
            if obs.select is None:
                obs_dict = battle_select(list(deck))
                continue
            obs_dict = battle_select(_fixed_action(obs))
            facts.append(_random_facts(to_observation_class(obs_dict)))
    finally:
        battle_finish()
    return facts


@pytest.fixture(scope="module")
def deck():
    return read_deck_csv()


def test_engine_has_no_seed_or_clone_entry_point():
    """seed 注入・状態複製・RNG 保存/復元の入口が **存在しない**ことを固定する。

    存在しないので、方法 A（探索専用 RNG の注入）・方法 C（RNG の保存と復元）は
    いずれも実装できない。将来 export が増えたらこのテストが落ちて気づける。
    """
    exported = {
        "GameInitialize", "BattleStart", "BattleFinish", "GetBattleData", "Select",
        "AgentStart", "VisualizeData", "AllCard", "AllAttack",
        "SearchBegin", "SearchStep", "SearchEnd", "SearchRelease",
    }
    for name in ("SetSeed", "Seed", "Clone", "CopyState", "SaveState", "RestoreState",
                 "GetRandomState", "SetRandomState"):
        assert not hasattr(lib, name), f"新しい入口 {name} が増えている。RNG 分離を再検討すること"
    for name in sorted(exported):
        assert hasattr(lib, name), f"既知の export {name} が消えた"


def test_engine_is_not_reproducible(deck):
    """同一デッキ・同一行動でも対局は再現しない。

    したがって **paired A/B は原理的に成立しない**。
    「search の有無で結果が変わった」ことを軌跡比較で示すこともできない。
    """
    runs = [_opening(deck) for _ in range(4)]
    runs = [r for r in runs if r]
    assert len(runs) >= 3
    assert any(r != runs[0] for r in runs[1:]), (
        "エンジンが再現するようになった。RNG 干渉を軌跡比較で直接検証できるので、"
        "Step 1-16 の判定をやり直すこと"
    )


def test_search_does_not_mutate_the_live_battle(deck):
    """探索を走らせても、本番対局の観測は変化しない（状態汚染は無い）。"""
    obs_dict, start = battle_start(list(deck), list(deck))
    if start.errorType != 0:
        pytest.skip("battle_start failed")
    checked = 0
    try:
        for _ in range(24):
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                break
            if obs.select is None:
                obs_dict = battle_select(list(deck))
                continue
            stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
            if stub is not None:
                before = _random_facts(to_observation_class(_get_battle_data()))
                with SearchSession(obs, HiddenState.from_stub(stub)) as session:
                    node = session.root
                    for _ in range(4):
                        if node.observation.select is None:
                            break
                        try:
                            node, _events = session.step(
                                node, _fixed_action(node.observation)
                            )
                        except EngineError:
                            break
                after = _random_facts(to_observation_class(_get_battle_data()))
                assert before == after, "探索が本番の観測を書き換えた"
                checked += 1
            obs_dict = battle_select(_fixed_action(obs))
    finally:
        battle_finish()
    assert checked > 0


def test_search_is_deterministic_unless_it_shuffles(deck):
    """探索は入力で決まる。ただし **SHUFFLE を含む場合だけ**結果が揺れる。

    実測（Step 1-16, 434 決定）: SHUFFLE 無し 179/179 一致、
    SHUFFLE 有り 172/255 一致（83 件が不一致）。
    つまり探索がエンジン側の乱数へ触れるのはシャッフル時に限られる。
    """
    obs_dict, start = battle_start(list(deck), list(deck))
    if start.errorType != 0:
        pytest.skip("battle_start failed")

    def run_once(observation, hidden):
        with SearchSession(observation, hidden) as session:
            node, shuffled = session.root, False
            for _ in range(6):
                if node.observation.select is None:
                    break
                try:
                    node, events = session.step(node, _fixed_action(node.observation))
                except EngineError:
                    break
                shuffled = shuffled or events.shuffled
            return _random_facts(node.observation), shuffled

    plain_checked = 0
    try:
        for _ in range(24):
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                break
            if obs.select is None:
                obs_dict = battle_select(list(deck))
                continue
            stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
            if stub is not None:
                first = run_once(obs, HiddenState.from_stub(stub))
                second = run_once(obs, HiddenState.from_stub(stub))
                if not first[1] and not second[1]:
                    assert first[0] == second[0], (
                        "SHUFFLE を含まない探索の結果が揺れた。"
                        "入力以外の乱数源が増えている"
                    )
                    plain_checked += 1
            obs_dict = battle_select(_fixed_action(obs))
    finally:
        battle_finish()
    assert plain_checked > 0


def test_phase1_stays_off_while_rng_non_interference_is_unverified():
    """RNG 非干渉が未検証である限り、既定 config で Phase 1 は動かない。"""
    assert RNG_NON_INTERFERENCE_VERIFIED is False, (
        "検証できたなら、根拠を step1-progress.md に書いた上でこの定数を更新すること"
    )
    lethal = load_config()["lethal_search"]
    assert lethal["module"] != "lethal_phase1", (
        "RNG 非干渉が未検証のまま Phase 1 が既定になっている"
    )
