"""レビュー指摘2 回帰テスト: proposals.py の下準備採用予算(_SetupTurnBudget)。

背景: 旧実装は `_setup_turn`/`_setup_count` をモジュール global で持ち、`decide()` 内で
`turn != _setup_turn` の時だけ自己修復していた。1プロセス内で複数ゲームを回す自己対戦・
テストでは、新しいゲームの1ターン目が前のゲームの最終ターン番号と偶然一致すると
自己修復に頼れず状態が汚染されうる（PR #90 レビュー指摘）。

修正後は状態を `_SetupTurnBudget` dataclass に封じ、新規ゲーム開始点
（`rule_based_agent.agent` が `obs.select is None` を受けた時）で
`proposals.reset_turn_state()` を呼んで明示リセットする。
"""

from ptcg_ai.rule_based.main_turn_parts import proposals


def test_take_resets_budget_on_turn_change():
    budget = proposals._SetupTurnBudget()
    assert budget.take(turn=1, cap=2) is True
    assert budget.take(turn=1, cap=2) is True
    assert budget.take(turn=1, cap=2) is False  # cap到達

    # ターンが変わると自己修復する。
    assert budget.take(turn=2, cap=2) is True
    assert budget.count == 1


def test_take_respects_cap_within_same_turn():
    budget = proposals._SetupTurnBudget()
    for _ in range(24):
        assert budget.take(turn=5, cap=24) is True
    assert budget.take(turn=5, cap=24) is False


def test_reset_turn_state_clears_module_level_budget():
    proposals._setup_budget.take(turn=3, cap=24)
    assert proposals._setup_budget.turn == 3
    assert proposals._setup_budget.count == 1

    proposals.reset_turn_state()

    assert proposals._setup_budget.turn is None
    assert proposals._setup_budget.count == 0


def test_stale_count_does_not_leak_across_games_with_same_turn_number():
    """ゲーム跨ぎでターン番号が偶然一致しても、reset_turn_state 済みなら予算は初期化されている。"""
    budget = proposals._setup_budget
    proposals.reset_turn_state()
    for _ in range(24):
        assert budget.take(turn=1, cap=24) is True
    assert budget.take(turn=1, cap=24) is False  # ゲーム1のターン1で予算を使い切った

    # 新規ゲーム開始（rule_based_agent.agent の obs.select is None 相当）。
    proposals.reset_turn_state()

    # ゲーム2のターン1（番号が偶然ゲーム1と同じ）でも予算はフルに使える。
    assert budget.take(turn=1, cap=24) is True
    assert budget.count == 1
