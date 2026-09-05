"""league.run_match.play_match の単体テスト。

実 cg エンジン(cg.dll / libcg.so)を使う。DLL がロードできない環境ではスキップされる。
"""

import sys
from pathlib import Path

import pytest

_LEAGUE_DIR = Path(__file__).resolve().parents[1]
if str(_LEAGUE_DIR) not in sys.path:
    sys.path.insert(0, str(_LEAGUE_DIR))


@pytest.fixture(scope="module")
def rule_based_agent():
    try:
        from ptcg_ai.rule_based.rule_based_agent import agent, read_deck_csv
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"rule_based_agent / cg engine unavailable: {exc}")
    return agent, read_deck_csv


def test_rule_based_mirror_match_finishes_with_a_winner(rule_based_agent):
    """rule_based 同士のミラー戦(同一デッキ)が異常終了せず 0/1 いずれかの勝敗で終わること。"""
    from run_match import play_match

    agent, read_deck_csv = rule_based_agent
    deck = read_deck_csv()

    result = play_match(agent, agent, deck, deck, seed=0)

    assert result.error is None
    assert result.winner in (0, 1)
    assert result.turns is not None and result.turns > 0
    assert result.steps > 0


def test_different_agents_can_play_each_other(rule_based_agent):
    """異なるエージェント実装(rule_based vs ml_policy)同士でも正常に対戦できること。"""
    from run_match import play_match

    rb_agent, read_deck_csv = rule_based_agent
    try:
        from ptcg_ai.ml_policy.ml_policy_agent import agent as ml_policy_agent
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ml_policy_agent unavailable: {exc}")

    deck = read_deck_csv()
    result = play_match(rb_agent, ml_policy_agent, deck, deck, seed=1)

    assert result.error is None
    assert result.winner in (0, 1)


def test_invalid_action_is_reported_as_error_not_raised():
    """エージェントが不正な選択(範囲外インデックス)を返しても play_match 自体は例外を出さず
    error 付きの MatchResult を返すこと(対戦リーグ全体を1試合の異常で止めないため)。
    """
    from run_match import play_match

    try:
        from ptcg_ai.rule_based.rule_based_agent import read_deck_csv
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cg engine unavailable: {exc}")

    deck = read_deck_csv()

    def broken_agent(obs):
        if obs.select is None:
            return list(deck)
        return [999999]  # 常に範囲外インデックスを返す壊れたエージェント

    result = play_match(broken_agent, broken_agent, deck, deck, seed=2)

    assert result.winner is None
    assert result.error is not None
