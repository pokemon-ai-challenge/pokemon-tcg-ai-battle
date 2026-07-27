"""撤退判断の再設計（2026-07-23）の回帰テスト。

docs/plans/rule_based/retreat-margin-unification-2026-07-23.md の通り、retreat.propose() は
「逃げに具体的な必要性がある場合だけ」提案する設計になっている:

    トリガーA（回避）: 次ターンKOされそう、かつベンチの最良候補は次ターンKOされなさそう。
    トリガーB（後退での即きぜつ）: 現在のアクティブでは今ターン相手アクティブを倒せないが、
        ベンチ候補の中に、今持っているエネルギーだけで倒せるものがいる。

どちらにも該当しなければ逃げない（かつてのスコア比較+マージン方式では、脅威が無くても
交代先のスコアが少し高いだけで逃げてしまっていた）。

is_likely_ko_next_turn / can_ko_with_any_available_attack / best_switch_target は
それぞれ別のテストで検証済みの信頼できる依存として扱い、ここでは monkeypatch で
戻り値を固定して retreat.propose() 自身の分岐ロジックだけを検証する。
"""

from types import SimpleNamespace

from cg.api import OptionType

from ptcg_ai.rule_based.main_turn_parts import pokemon_value
from ptcg_ai.rule_based.main_turn_parts.priorities import retreat
from ptcg_ai.board_evaluation import attack_features, board_features, energy_requirements

_ALAKAZAM_ID = 743  # 実在カード。retreatCost=1（can_afford_retreatは別途モックするので値自体は使わない）


def _pokemon(card_id: int = _ALAKAZAM_ID, energies=None):
    return SimpleNamespace(id=card_id, energies=list(energies or []))


def _obs_with_retreat_option():
    option = SimpleNamespace(type=OptionType.RETREAT)
    select = SimpleNamespace(option=[option])
    return select


def _state(*, retreated: bool, active, bench):
    own_player = SimpleNamespace(active=[active], bench=bench, handCount=5)
    opponent_player = SimpleNamespace(active=[_pokemon(999)], bench=[], handCount=5)
    return SimpleNamespace(retreated=retreated, yourIndex=0, players=[own_player, opponent_player])


def _make_obs(*, active, bench, retreated=False):
    select = _obs_with_retreat_option()
    state = _state(retreated=retreated, active=active, bench=bench)
    return SimpleNamespace(select=select, current=state)


def _patch_common(monkeypatch, *, can_afford_retreat=True):
    monkeypatch.setattr(energy_requirements, "can_afford_retreat", lambda cost, energies: can_afford_retreat)
    # 既定では probabilistic_ko を無効化し、実ファイル(configs/rule_lethal.json)の値に関わらず
    # 従来のブール判定(is_likely_ko_next_turn)経路を決定的にテストする。確率経路自体のテストは
    # 下の「probabilistic_ko」セクションで個別に _CONFIG_CACHE を上書きする。
    monkeypatch.setattr(retreat, "_CONFIG_CACHE", {"probabilistic_ko": {"enabled": False}})


def test_no_retreat_when_no_threat_and_no_ko_opportunity(monkeypatch):
    """脅威も無く、交代してもKOできないなら逃げない（旧・スコア比較方式なら逃げていたケース）。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", lambda pokemon, state, your_index: False)
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: False)

    assert retreat.propose(obs) is None


def test_trigger_a_fires_when_urgent_and_candidate_is_safe(monkeypatch):
    """次ターンKOされそう、かつ候補は安全 → 逃げを提案する。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)

    def fake_ko_next_turn(pokemon, state, your_index):
        return pokemon is active  # active だけ危険、candidate は安全

    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", fake_ko_next_turn)
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: False)

    proposal = retreat.propose(obs)
    assert proposal is not None
    assert proposal.category == "retreat"
    assert proposal.select == [0]


def test_trigger_a_does_not_fire_when_candidate_is_also_in_danger(monkeypatch):
    """次ターンKOされそうでも、候補も同じく危険なら逃げても意味が無いので提案しない。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", lambda pokemon, state, your_index: True)
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: False)

    assert retreat.propose(obs) is None


def test_trigger_b_fires_when_bench_can_ko_but_active_cannot(monkeypatch):
    """現在のアクティブではKOできないが、ベンチ候補ならKOできる → 逃げを提案する。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", lambda pokemon, state, your_index: False)

    def fake_can_ko(pokemon, opponent, weakness, resistance, hand_size):
        return pokemon is candidate  # candidate だけKO可能

    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", fake_can_ko)

    proposal = retreat.propose(obs)
    assert proposal is not None
    assert proposal.category == "retreat"


def test_trigger_b_does_not_fire_when_active_can_already_ko(monkeypatch):
    """現在のアクティブで既にKOできるなら、attack.py 側で完結するので retreat は不要。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", lambda pokemon, state, your_index: False)
    # active・candidate 両方KO可能でも、active が既にKOできる時点でトリガーBは不成立。
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: True)

    assert retreat.propose(obs) is None


def test_trigger_b_does_not_fire_when_no_bench_candidate_can_ko(monkeypatch):
    """現在のアクティブでKOできず、ベンチ候補もどれもKOできないなら逃げない。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", lambda pokemon, state, your_index: False)
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: False)

    assert retreat.propose(obs) is None


def test_no_retreat_when_cannot_afford_retreat_cost(monkeypatch):
    """にげるコストを払えない場合は、トリガーが成立していても提案しない（既存の前提チェック）。"""
    _patch_common(monkeypatch, can_afford_retreat=False)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", lambda pokemon, state, your_index: True)
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: False)

    assert retreat.propose(obs) is None


def test_no_retreat_when_already_retreated_this_turn(monkeypatch):
    """State.retreated が True（今ターン既に交代済み）なら提案しない。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate], retreated=True)

    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", lambda pokemon, state, your_index: True)

    assert retreat.propose(obs) is None


def test_no_retreat_when_bench_is_empty(monkeypatch):
    """ベンチが空なら逃げ先が無いので提案しない。"""
    _patch_common(monkeypatch)
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[])

    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", lambda pokemon, state, your_index: True)

    assert retreat.propose(obs) is None


# --- probabilistic_ko (ベイズ推定によるトリガーAの確率化) -----------------------------------
#
# is_likely_ko_next_turn の代わりに board_features.likely_ko_probability_next_turn +
# config の threshold で判定する分岐。retreat._CONFIG_CACHE を直接差し替えて
# core.config.load_config() の実ファイル読み込みを経由せず決定的にテストする
# （monkeypatch はテスト終了時に元の値へ自動的に戻す）。


def test_trigger_a_probabilistic_fires_when_active_above_threshold_and_candidate_below(monkeypatch):
    """有効時: active確率が閾値以上・候補確率が閾値未満なら逃げを提案する。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(retreat, "_CONFIG_CACHE", {"probabilistic_ko": {"enabled": True, "threshold": 0.5}})
    monkeypatch.setattr(
        board_features,
        "likely_ko_probability_next_turn",
        lambda pokemon, state, your_index: 0.8 if pokemon is active else 0.2,
    )
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: False)

    proposal = retreat.propose(obs)
    assert proposal is not None
    assert proposal.category == "retreat"


def test_trigger_a_probabilistic_does_not_fire_when_active_below_threshold(monkeypatch):
    """active確率が閾値未満なら、旧ブール判定ならTrue相当のケースでも逃げない。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(retreat, "_CONFIG_CACHE", {"probabilistic_ko": {"enabled": True, "threshold": 0.5}})
    monkeypatch.setattr(board_features, "likely_ko_probability_next_turn", lambda pokemon, state, your_index: 0.3)
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: False)

    assert retreat.propose(obs) is None


def test_trigger_a_probabilistic_does_not_fire_when_candidate_also_above_threshold(monkeypatch):
    """active・候補どちらも閾値以上なら、逃げても意味が無いので提案しない。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(retreat, "_CONFIG_CACHE", {"probabilistic_ko": {"enabled": True, "threshold": 0.5}})
    monkeypatch.setattr(board_features, "likely_ko_probability_next_turn", lambda pokemon, state, your_index: 0.9)
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: False)

    assert retreat.propose(obs) is None


def test_trigger_a_falls_back_to_boolean_when_flag_disabled(monkeypatch):
    """probabilistic_ko.enabled が False なら、確率関数を呼ばず従来のブール判定を使う（非回帰）。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(retreat, "_CONFIG_CACHE", {"probabilistic_ko": {"enabled": False, "threshold": 0.5}})

    def fail_if_called(*a, **k):
        raise AssertionError("probabilistic_ko 無効時は likely_ko_probability_next_turn を呼ばないはず")

    monkeypatch.setattr(board_features, "likely_ko_probability_next_turn", fail_if_called)
    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", lambda pokemon, state, your_index: pokemon is active)
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: False)

    proposal = retreat.propose(obs)
    assert proposal is not None


def test_trigger_a_respects_custom_threshold(monkeypatch):
    """threshold をカスタム値にすると、その値で発火有無が変わる。"""
    _patch_common(monkeypatch)
    candidate = _pokemon(65, energies=[])
    active = _pokemon(_ALAKAZAM_ID, energies=[])
    obs = _make_obs(active=active, bench=[candidate])

    monkeypatch.setattr(pokemon_value, "best_switch_target", lambda bench, state, your_index: candidate)
    monkeypatch.setattr(retreat, "_CONFIG_CACHE", {"probabilistic_ko": {"enabled": True, "threshold": 0.9}})
    monkeypatch.setattr(
        board_features,
        "likely_ko_probability_next_turn",
        lambda pokemon, state, your_index: 0.8 if pokemon is active else 0.1,
    )
    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", lambda *a, **k: False)

    # active確率0.8 < 閾値0.9 なので発火しない（デフォルト閾値0.5なら発火するケース）。
    assert retreat.propose(obs) is None
