"""レビュー指摘1 回帰テスト: にげあしドロー等ドロー系特性の山札フロア設定。

背景: `priorities/ability.py` はかつて `_DRAW_ABILITY_MIN_DECK` を
`os.environ.get("PTCG_DRAW_ABILITY_MIN_DECK", "0")` から読んでいたため、実際に効く値が
env のデフォルト文字列に埋もれ、`grep` で追えず、Kaggle採点ハーネスがenvを設定しないため
出荷挙動が deck_plan.py に何も書かれていない状態だった（PR #90 レビュー指摘）。

修正後は decks/new_deck/deck_plan.py の DRAW_ABILITY_MIN_DECK / DRAW_ABILITY_MAX_HAND が
コミットされる既定値で、env はその上書き専用になる。この2点を固定する。
"""

import importlib

from decks import active
from ptcg_ai.shared import profile_registry


def test_deck_plan_commits_draw_ability_floor():
    # 提出物の実挙動は deck_plan.py に明示された値で決まる（2026-07-25、デッキ担当判断）。
    assert active.deck_plan.DRAW_ABILITY_MIN_DECK == 0
    assert active.deck_plan.DRAW_ABILITY_MAX_HAND == 999


def test_profile_registry_reads_deck_plan_floor():
    assert profile_registry.get_draw_ability_deck_floor() == 0
    assert profile_registry.get_draw_ability_max_hand() == 999


def test_ability_module_default_matches_deck_plan_without_env(monkeypatch):
    monkeypatch.delenv("PTCG_DRAW_ABILITY_MIN_DECK", raising=False)
    monkeypatch.delenv("PTCG_DRAW_ABILITY_MAX_HAND", raising=False)
    from ptcg_ai.rule_based.main_turn_parts.priorities import ability

    importlib.reload(ability)
    try:
        assert ability._DRAW_ABILITY_MIN_DECK == 0
        assert ability._DRAW_ABILITY_MAX_HAND == 999
    finally:
        importlib.reload(ability)  # 他テストへ影響しないよう env なし状態で戻す


def test_ability_module_env_overrides_deck_plan_floor(monkeypatch):
    monkeypatch.setenv("PTCG_DRAW_ABILITY_MIN_DECK", "14")
    monkeypatch.setenv("PTCG_DRAW_ABILITY_MAX_HAND", "6")
    from ptcg_ai.rule_based.main_turn_parts.priorities import ability

    importlib.reload(ability)
    try:
        assert ability._DRAW_ABILITY_MIN_DECK == 14
        assert ability._DRAW_ABILITY_MAX_HAND == 6
    finally:
        monkeypatch.delenv("PTCG_DRAW_ABILITY_MIN_DECK", raising=False)
        monkeypatch.delenv("PTCG_DRAW_ABILITY_MAX_HAND", raising=False)
        importlib.reload(ability)  # 既定値（env無し）に戻す
