"""card_cache.energy_card_ids_by_type() の単体テスト。

all_card_data() から BASIC_ENERGY カードだけを energyType ごとにグルーピングして
キャッシュする挙動、reset_cache() でキャッシュが破棄される挙動を確認する。
"""

from types import SimpleNamespace

from cg.api import CardType, EnergyType

from ptcg_ai.shared import card_cache


def _card(card_id: int, card_type: CardType, energy_type: EnergyType):
    return SimpleNamespace(cardId=card_id, cardType=card_type, energyType=energy_type)


def test_groups_basic_energy_by_type(monkeypatch):
    cards = [
        _card(10, CardType.BASIC_ENERGY, EnergyType.FIRE),
        _card(11, CardType.BASIC_ENERGY, EnergyType.WATER),
        _card(12, CardType.POKEMON, EnergyType.FIRE),  # ポケモンは無視される
    ]
    monkeypatch.setattr(card_cache, "all_card_data", lambda: cards)
    card_cache.reset_cache()

    result = card_cache.energy_card_ids_by_type()

    assert result == {EnergyType.FIRE: [10], EnergyType.WATER: [11]}


def test_excludes_special_energy(monkeypatch):
    cards = [
        _card(20, CardType.BASIC_ENERGY, EnergyType.PSYCHIC),
        _card(21, CardType.SPECIAL_ENERGY, EnergyType.PSYCHIC),
    ]
    monkeypatch.setattr(card_cache, "all_card_data", lambda: cards)
    card_cache.reset_cache()

    result = card_cache.energy_card_ids_by_type()

    assert result == {EnergyType.PSYCHIC: [20]}


def test_includes_multiple_reprints_of_same_type(monkeypatch):
    cards = [
        _card(30, CardType.BASIC_ENERGY, EnergyType.GRASS),
        _card(31, CardType.BASIC_ENERGY, EnergyType.GRASS),
    ]
    monkeypatch.setattr(card_cache, "all_card_data", lambda: cards)
    card_cache.reset_cache()

    result = card_cache.energy_card_ids_by_type()

    assert result == {EnergyType.GRASS: [30, 31]}


def test_result_is_cached_until_reset(monkeypatch):
    calls = {"count": 0}

    def fake_all_card_data():
        calls["count"] += 1
        return [_card(40, CardType.BASIC_ENERGY, EnergyType.METAL)]

    monkeypatch.setattr(card_cache, "all_card_data", fake_all_card_data)
    card_cache.reset_cache()

    card_cache.energy_card_ids_by_type()
    card_cache.energy_card_ids_by_type()

    assert calls["count"] == 1


def test_reset_cache_clears_energy_index(monkeypatch):
    monkeypatch.setattr(
        card_cache, "all_card_data", lambda: [_card(50, CardType.BASIC_ENERGY, EnergyType.LIGHTNING)]
    )
    card_cache.reset_cache()
    first = card_cache.energy_card_ids_by_type()
    assert first == {EnergyType.LIGHTNING: [50]}

    monkeypatch.setattr(
        card_cache, "all_card_data", lambda: [_card(51, CardType.BASIC_ENERGY, EnergyType.DARKNESS)]
    )
    card_cache.reset_cache()
    second = card_cache.energy_card_ids_by_type()

    assert second == {EnergyType.DARKNESS: [51]}
