"""``selector._build_candidate_provider`` が setup-first 不変条件を壊さないことの回帰テスト
（EXP-A44 §5.5 / §11: ``test_risk_candidates_respect_setup_first.py``）。

``proposals.SETUP_BEFORE_ATTACK``（既定 True）が有効なとき、「無コストで自己枯渇する下準備
（board/energy/draw/ability）が残っている間は attack 等を候補に混ぜない」という
``proposals.decide()`` の不変条件を、risk_determinization 用の候補プロバイダでも固定する。
"""

from __future__ import annotations

from ptcg_ai.action_selection import selector
from ptcg_ai.rule_based.main_turn_parts import proposals


def _proposal(category: str, select: list[int], score: float) -> proposals.ActionProposal:
    return proposals.ActionProposal(category=category, select=select, score=score, reason="test")


def test_setup_present_excludes_attack_from_candidates(monkeypatch):
    """下準備(board)が残っている間、attackはrisk層の候補集合に一切現れない。"""
    fixed_proposals = [
        _proposal("board", [0], score=1.0),
        _proposal("attack", [1], score=999.0),  # ルールスコアは圧倒的に高いが、setup-firstで除外される
        _proposal("end", [2], score=0.0),
    ]
    monkeypatch.setattr(proposals, "collect_proposals", lambda obs: fixed_proposals)
    monkeypatch.setattr(proposals, "SETUP_BEFORE_ATTACK", True)

    provider = selector._build_candidate_provider(obs=None)
    candidates = provider()

    assert len(candidates) == 1
    # _total_score = 提案自体のscore + weights.CATEGORY_BASE_WEIGHT["board"](=50.0)。
    assert candidates[0] == ([0], 1.0 + 50.0)


def test_setup_present_multiple_setup_candidates_kept_attack_excluded(monkeypatch):
    fixed_proposals = [
        _proposal("board", [0], score=1.0),
        _proposal("energy", [1], score=2.0),
        _proposal("attack", [2], score=999.0),
    ]
    monkeypatch.setattr(proposals, "collect_proposals", lambda obs: fixed_proposals)
    monkeypatch.setattr(proposals, "SETUP_BEFORE_ATTACK", True)

    provider = selector._build_candidate_provider(obs=None)
    candidates = provider()

    selects = [action for action, _score in candidates]
    assert [2] not in selects  # attack は候補に混じらない
    assert len(candidates) == 2


def test_no_setup_remaining_attack_is_a_candidate(monkeypatch):
    """下準備が無ければ、通常どおり attack も候補集合に入る。"""
    fixed_proposals = [
        _proposal("attack", [2], score=999.0),
        _proposal("end", [3], score=0.0),
    ]
    monkeypatch.setattr(proposals, "collect_proposals", lambda obs: fixed_proposals)
    monkeypatch.setattr(proposals, "SETUP_BEFORE_ATTACK", True)

    provider = selector._build_candidate_provider(obs=None)
    candidates = provider()

    selects = [action for action, _score in candidates]
    assert [2] in selects
    assert candidates[0][0] == [2]  # ルールスコア降順で先頭


def test_setup_before_attack_disabled_ignores_setup_first_filter(monkeypatch):
    """PTCG_SETUP_BEFORE_ATTACK=0相当(SETUP_BEFORE_ATTACK=False)では通常のスコア比較に戻る。"""
    fixed_proposals = [
        _proposal("board", [0], score=1.0),
        _proposal("attack", [1], score=999.0),
    ]
    monkeypatch.setattr(proposals, "collect_proposals", lambda obs: fixed_proposals)
    monkeypatch.setattr(proposals, "SETUP_BEFORE_ATTACK", False)

    provider = selector._build_candidate_provider(obs=None)
    candidates = provider()

    selects = [action for action, _score in candidates]
    assert [1] in selects
    assert candidates[0][0] == [1]  # スコア最大のattackが先頭に来る


def test_candidates_sorted_by_total_score_descending(monkeypatch):
    fixed_proposals = [
        _proposal("board", [0], score=1.0),
        _proposal("draw", [1], score=5.0),
        _proposal("ability", [2], score=3.0),
    ]
    monkeypatch.setattr(proposals, "collect_proposals", lambda obs: fixed_proposals)
    monkeypatch.setattr(proposals, "SETUP_BEFORE_ATTACK", True)

    provider = selector._build_candidate_provider(obs=None)
    candidates = provider()

    scores = [score for _action, score in candidates]
    assert scores == sorted(scores, reverse=True)
