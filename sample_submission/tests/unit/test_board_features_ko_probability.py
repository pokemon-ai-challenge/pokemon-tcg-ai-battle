"""board_features.likely_ko_probability_next_turn の単体テスト。

is_likely_ko_next_turn と同じワザ・ダメージ判定を共有するが、shortfall合計が1の
ケースを OpponentHiddenState.marginals()（超幾何ベースの較正済みベイズ推定）で
確率化する部分だけを検証する。card_cache / energy_requirements / attack_features /
hidden_information.match_context への依存はすべて monkeypatch で固定し、
likely_ko_probability_next_turn 自身の分岐ロジックだけを見る。
"""

from types import SimpleNamespace

from cg.api import EnergyType

from ptcg_ai.board_evaluation import attack_features, board_features, energy_requirements
from ptcg_ai.hidden_information import match_context
from ptcg_ai.shared import card_cache

_ATTACKER_CARD_ID = 1001
_DEFENDER_CARD_ID = 2002


class _StubOpponentState:
    def __init__(self, *, is_ready: bool, marginals: dict[int, dict[str, float]] | None = None):
        self.is_ready = is_ready
        self._marginals = marginals or {}

    def marginals(self):
        return self._marginals


def _attack(attack_id: int):
    return SimpleNamespace(attack_id=attack_id)


def _pokemon(card_id: int, hp: int = 100, energies=None):
    return SimpleNamespace(id=card_id, hp=hp, energies=list(energies or []))


def _state(*, active, hand_count: int = 5):
    opponent_player = SimpleNamespace(active=[active], handCount=hand_count)
    own_player = SimpleNamespace()
    return SimpleNamespace(players=[own_player, opponent_player])


def _patch_attacks(monkeypatch, attack_specs: list[tuple[int, dict, int]]):
    """attack_specs: [(attack_id, shortfall_dict, damage), ...]。

    card_cache.get_card/get_attack, energy_requirements.energy_shortfall,
    attack_features.resolve_damage をまとめて差し替える。
    """
    attack_ids = [attack_id for attack_id, _shortfall, _damage in attack_specs]
    shortfall_by_id = {attack_id: shortfall for attack_id, shortfall, _damage in attack_specs}
    damage_by_id = {attack_id: damage for attack_id, _shortfall, damage in attack_specs}

    attacker_card = SimpleNamespace(attacks=attack_ids)
    defender_card = SimpleNamespace(weakness=None, resistance=None)

    def fake_get_card(card_id):
        if card_id == _ATTACKER_CARD_ID:
            return attacker_card
        if card_id == _DEFENDER_CARD_ID:
            return defender_card
        raise KeyError(card_id)

    monkeypatch.setattr(card_cache, "get_card", fake_get_card)
    monkeypatch.setattr(card_cache, "get_attack", lambda attack_id: _attack(attack_id))
    monkeypatch.setattr(
        energy_requirements, "energy_shortfall", lambda attack, attached: shortfall_by_id[attack.attack_id]
    )
    monkeypatch.setattr(
        attack_features,
        "resolve_damage",
        lambda attack, attacker, weakness, resistance, hand_size: damage_by_id[attack.attack_id],
    )


def _patch_opponent_state(monkeypatch, *, is_ready: bool, marginals=None):
    monkeypatch.setattr(
        match_context, "get_opponent_state", lambda: _StubOpponentState(is_ready=is_ready, marginals=marginals)
    )


def test_certain_ko_returns_1_when_shortfall_zero(monkeypatch):
    """shortfall==0（今すぐ確実にKO可能）は match_context に触れず 1.0 を返す。"""
    _patch_attacks(monkeypatch, [(1, {}, 100)])
    active = _pokemon(_ATTACKER_CARD_ID)
    defender = _pokemon(_DEFENDER_CARD_ID, hp=100)
    state = _state(active=active)

    def fail_if_called():
        raise AssertionError("shortfall==0 では match_context.get_opponent_state を呼ばないはず")

    monkeypatch.setattr(match_context, "get_opponent_state", fail_if_called)

    assert board_features.likely_ko_probability_next_turn(defender, state, 0) == 1.0


def test_shortfall_one_uses_hand_marginal_probability(monkeypatch):
    """shortfall合計1は、必要エネルギーの手札marginal確率をそのまま使う（単一card_id）。"""
    energy_id = 20
    _patch_attacks(monkeypatch, [(1, {EnergyType.FIRE: 1}, 100)])
    monkeypatch.setattr(card_cache, "energy_card_ids_by_type", lambda: {EnergyType.FIRE: [energy_id]})
    _patch_opponent_state(monkeypatch, is_ready=True, marginals={energy_id: {"hand": 0.4}})

    active = _pokemon(_ATTACKER_CARD_ID)
    defender = _pokemon(_DEFENDER_CARD_ID, hp=100)
    state = _state(active=active)

    assert board_features.likely_ko_probability_next_turn(defender, state, 0) == 0.4


def test_shortfall_one_unions_multiple_reprint_card_ids(monkeypatch):
    """同タイプの再録違い(複数card_id)は 1-Π(1-p_i) で和集合として合成する。"""
    _patch_attacks(monkeypatch, [(1, {EnergyType.WATER: 1}, 100)])
    monkeypatch.setattr(card_cache, "energy_card_ids_by_type", lambda: {EnergyType.WATER: [21, 22]})
    _patch_opponent_state(monkeypatch, is_ready=True, marginals={21: {"hand": 0.3}, 22: {"hand": 0.2}})

    active = _pokemon(_ATTACKER_CARD_ID)
    defender = _pokemon(_DEFENDER_CARD_ID, hp=100)
    state = _state(active=active)

    result = board_features.likely_ko_probability_next_turn(defender, state, 0)
    assert result == 1.0 - (1.0 - 0.3) * (1.0 - 0.2)


def test_colorless_shortfall_unions_all_basic_types(monkeypatch):
    """COLORLESS不足は、タイプを問わず全基本エネルギーcard_idの和集合になる。"""
    _patch_attacks(monkeypatch, [(1, {EnergyType.COLORLESS: 1}, 100)])
    monkeypatch.setattr(
        card_cache, "energy_card_ids_by_type", lambda: {EnergyType.FIRE: [31], EnergyType.WATER: [32]}
    )
    _patch_opponent_state(monkeypatch, is_ready=True, marginals={31: {"hand": 0.3}, 32: {"hand": 0.5}})

    active = _pokemon(_ATTACKER_CARD_ID)
    defender = _pokemon(_DEFENDER_CARD_ID, hp=100)
    state = _state(active=active)

    result = board_features.likely_ko_probability_next_turn(defender, state, 0)
    assert result == 1.0 - (1.0 - 0.3) * (1.0 - 0.5)


def test_falls_back_to_1_when_not_ready(monkeypatch):
    """OpponentHiddenState未準備(is_ready=False)なら、今日の挙動(True相当)である1.0にフォールバックする。"""
    _patch_attacks(monkeypatch, [(1, {EnergyType.FIRE: 1}, 100)])
    monkeypatch.setattr(card_cache, "energy_card_ids_by_type", lambda: {EnergyType.FIRE: [20]})
    _patch_opponent_state(monkeypatch, is_ready=False)

    active = _pokemon(_ATTACKER_CARD_ID)
    defender = _pokemon(_DEFENDER_CARD_ID, hp=100)
    state = _state(active=active)

    assert board_features.likely_ko_probability_next_turn(defender, state, 0) == 1.0


def test_no_mapping_for_energy_type_falls_back_to_1(monkeypatch):
    """必要エネルギー種別がcard_cache側でマッピングできない場合も安全側の1.0にフォールバックする。"""
    _patch_attacks(monkeypatch, [(1, {EnergyType.FIRE: 1}, 100)])
    monkeypatch.setattr(card_cache, "energy_card_ids_by_type", lambda: {})
    _patch_opponent_state(monkeypatch, is_ready=True, marginals={99: {"hand": 0.9}})

    active = _pokemon(_ATTACKER_CARD_ID)
    defender = _pokemon(_DEFENDER_CARD_ID, hp=100)
    state = _state(active=active)

    assert board_features.likely_ko_probability_next_turn(defender, state, 0) == 1.0


def test_returns_0_when_no_attack_qualifies(monkeypatch):
    """shortfallが許容量を超える、またはダメージが足りないワザしか無ければ 0.0。"""
    _patch_attacks(monkeypatch, [(1, {EnergyType.FIRE: 2}, 999), (2, {}, 10)])
    active = _pokemon(_ATTACKER_CARD_ID)
    defender = _pokemon(_DEFENDER_CARD_ID, hp=100)
    state = _state(active=active)

    assert board_features.likely_ko_probability_next_turn(defender, state, 0) == 0.0


def test_multiple_attacks_take_max_probability(monkeypatch):
    """複数ワザが該当する場合はワザごとの確率の最大値を採る。"""
    _patch_attacks(
        monkeypatch,
        [
            (1, {EnergyType.FIRE: 1}, 100),
            (2, {EnergyType.WATER: 1}, 100),
        ],
    )
    monkeypatch.setattr(
        card_cache, "energy_card_ids_by_type", lambda: {EnergyType.FIRE: [20], EnergyType.WATER: [21]}
    )
    _patch_opponent_state(monkeypatch, is_ready=True, marginals={20: {"hand": 0.3}, 21: {"hand": 0.7}})

    active = _pokemon(_ATTACKER_CARD_ID)
    defender = _pokemon(_DEFENDER_CARD_ID, hp=100)
    state = _state(active=active)

    assert board_features.likely_ko_probability_next_turn(defender, state, 0) == 0.7


def test_no_active_opponent_returns_0(monkeypatch):
    """相手のバトル場が伏せ(None)なら 0.0。"""
    opponent_player = SimpleNamespace(active=[None], handCount=5)
    own_player = SimpleNamespace()
    state = SimpleNamespace(players=[own_player, opponent_player])
    defender = _pokemon(_DEFENDER_CARD_ID, hp=100)

    assert board_features.likely_ko_probability_next_turn(defender, state, 0) == 0.0
