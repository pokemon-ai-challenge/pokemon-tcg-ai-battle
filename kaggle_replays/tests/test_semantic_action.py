"""semantic_action.py（要件定義 v2 §4.2 の解決規則）の検証。

このモジュールは方策学習・再マッピングの両方が乗る土台であり、壊れても例外を
投げず「解決できなかった」として静かに劣化する設計になっている。したがって
テストが無いと、精度が出なかったときに「モデルが悪いのか解決が壊れているのか」を
切り分けられない。

特に守るべき性質:
  - current.stadium は list である（単一オブジェクト扱いすると ABILITY の
    解決が 46% 失敗する。実際に踏んだ回帰）
  - 非公開領域・範囲外を引いても例外にせず None を返す
  - action_label に serial を含めない（試合ごとに振り直されるため、含めると
    別エピソード間で常に不一致になり、一致率の測定が無意味になる）
"""

from pathlib import Path
import sys

import pytest

# semantic_action.py は sample_submission 側(唯一の実装)へ移した。
# kaggle_replays/policy_prior/build_dataset.py と同じ配線パターンで import する。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"
if str(_SAMPLE_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from ptcg_ai.learning.semantic_action import (  # noqa: E402
    ABILITY,
    ATTACH,
    ATTACK,
    CARD,
    END,
    EVOLVE,
    NUMBER,
    PLAY,
    RETREAT,
    SKILL,
    YES,
    ResolutionStats,
    action_label,
    check_invariants,
    lookup_entity,
    resolve_option,
)

ME = 0
HAND, DISCARD, ACTIVE, BENCH, PRIZE, STADIUM = 2, 3, 4, 5, 6, 7
DECK = 1  # players 配下にリストを持たない（deckCount のみ公開）


def card(card_id: int, serial: int, owner: int = ME) -> dict:
    return {"id": card_id, "playerIndex": owner, "serial": serial}


def pokemon(card_id: int, serial: int, hp: int = 70) -> dict:
    return {
        "id": card_id, "serial": serial, "playerIndex": ME,
        "hp": hp, "maxHp": hp, "appearThisTurn": False,
        "energies": [], "energyCards": [], "tools": [], "preEvolution": [],
    }


@pytest.fixture
def current() -> dict:
    """自分視点の observation.current。相手の非公開領域は実データと同じく null。"""
    return {
        "yourIndex": ME,
        "turn": 5,
        "stadium": [card(900, 90)],
        "players": [
            {   # 自分
                "hand": [card(7, 3), card(19, 11), card(305, 42)],
                "active": [pokemon(646, 20)],
                "bench": [pokemon(112, 21, hp=110), pokemon(741, 22, hp=50)],
                "discard": [card(19, 12)],
                "prize": [card(66, 51)] * 3,
                "handCount": 3, "deckCount": 40,
            },
            {   # 相手: 手札とサイドは非公開
                "hand": None,
                "active": [pokemon(741, 78)],
                "bench": [],
                "discard": [card(743, 80)],
                "prize": [None] * 6,
                "handCount": 5, "deckCount": 44,
            },
        ],
    }


# --- lookup_entity ---------------------------------------------------------


def test_stadium_is_a_list_and_resolves_by_index(current):
    """★回帰テスト: current.stadium は単一オブジェクトではなく list。

    単一オブジェクトとして扱った初回実装では ABILITY の 46% が解決失敗した。
    """
    assert lookup_entity(current, ME, STADIUM, 0) == card(900, 90)


def test_empty_stadium_returns_none_without_raising(current):
    current["stadium"] = []
    assert lookup_entity(current, ME, STADIUM, 0) is None


@pytest.mark.parametrize(
    ("area", "index"),
    [
        (HAND, 99),      # 範囲外
        (HAND, -1),      # 負の index
        (HAND, None),    # index 欠落
        (None, 0),       # area 欠落
        (DECK, 0),       # players 配下にリストを持たない領域
        (999, 0),        # 未知の AreaType
    ],
)
def test_unresolvable_references_return_none(current, area, index):
    assert lookup_entity(current, ME, area, index) is None


def test_hidden_opponent_area_returns_none(current):
    """相手の手札は null として渡る。原理的に解決できない場合がある（v2 §4.2）。"""
    assert lookup_entity(current, 1, HAND, 0) is None


def test_null_entry_in_public_list_returns_none(current):
    """サイドは [null]*6 のように「枚数だけ公開」の形を取る。"""
    assert lookup_entity(current, 1, PRIZE, 0) is None


def test_out_of_range_player_returns_none(current):
    assert lookup_entity(current, 5, HAND, 0) is None


# --- resolve_option: OptionType ごとの解決 ---------------------------------


def test_play_resolves_from_hand_without_area_field(current):
    """PLAY は area を持たないが手札固定（api.py: "Index within the hand"）。"""
    stats = ResolutionStats()
    action = resolve_option({"type": PLAY, "index": 1}, current, ME, stats)
    assert action["card_id"] == 19
    assert action["serial"] == 11
    assert action["resolved"] is True
    assert stats.fail == {}


def test_attach_resolves_both_card_and_target(current):
    action = resolve_option(
        {"type": ATTACH, "area": HAND, "index": 0,
         "inPlayArea": BENCH, "inPlayIndex": 1},
        current, ME,
    )
    assert (action["card_id"], action["serial"]) == (7, 3)
    assert (action["target_card_id"], action["target_serial"]) == (741, 22)


def test_evolve_resolves_both_card_and_target(current):
    action = resolve_option(
        {"type": EVOLVE, "area": HAND, "index": 2,
         "inPlayArea": ACTIVE, "inPlayIndex": 0},
        current, ME,
    )
    assert action["card_id"] == 305
    assert action["target_card_id"] == 646


def test_ability_on_stadium_resolves(current):
    action = resolve_option({"type": ABILITY, "area": STADIUM, "index": 0}, current, ME)
    assert action["card_id"] == 900
    assert action["resolved"] is True


def test_ability_on_bench_resolves(current):
    action = resolve_option({"type": ABILITY, "area": BENCH, "index": 0}, current, ME)
    assert action["card_id"] == 112


def test_card_option_honours_player_index(current):
    """CARD は playerIndex を持つ。相手のトラッシュ（公開領域）を指せる。"""
    action = resolve_option(
        {"type": CARD, "area": DISCARD, "index": 0, "playerIndex": 1}, current, ME
    )
    assert action["card_id"] == 743


def test_attack_needs_no_resolution(current):
    stats = ResolutionStats()
    action = resolve_option({"type": ATTACK, "attackId": 1234}, current, ME, stats)
    assert action["attack_id"] == 1234
    assert action["card_id"] is None
    assert action["resolved"] is True
    assert stats.total == 0  # 解決を試みていないので成否に数えない


def test_skill_reads_identity_directly(current):
    """SKILL は Option 自身が cardId/serial を持つ唯一の型。"""
    action = resolve_option(
        {"type": SKILL, "cardId": 555, "serial": 66}, current, ME
    )
    assert (action["card_id"], action["serial"]) == (555, 66)
    assert action["resolved"] is True


@pytest.mark.parametrize("option_type", [END, RETREAT, YES, NUMBER])
def test_entityless_types_resolve_trivially(current, option_type):
    action = resolve_option({"type": option_type}, current, ME)
    assert action["resolved"] is True
    assert action["card_id"] is None


def test_number_keeps_its_value(current):
    action = resolve_option({"type": NUMBER, "number": 3}, current, ME)
    assert action["number"] == 3


# --- 失敗時の振る舞い ------------------------------------------------------


def test_unknown_option_type_is_recorded_not_raised(current):
    """v2 §4.7: Enum はコンペ期間中に要素追加され得る。例外で落ちてはならない。"""
    stats = ResolutionStats()
    action = resolve_option({"type": 99, "area": HAND, "index": 0}, current, ME, stats)
    assert action["resolved"] is False
    assert stats.fail[99] == 1


def test_failed_resolution_is_counted_by_option_type(current):
    stats = ResolutionStats()
    resolve_option({"type": PLAY, "index": 99}, current, ME, stats)  # 範囲外
    assert stats.fail[PLAY] == 1
    assert stats.failure_rate == 1.0


def test_failure_rate_is_zero_on_empty_stats():
    """0除算しないこと（サマリ生成が全件スキップ時に落ちないため）。"""
    assert ResolutionStats().failure_rate == 0.0


def test_unresolvable_target_does_not_fail_the_whole_action(current):
    """付け先だけ引けない場合、カード側の解決結果は保持する。"""
    action = resolve_option(
        {"type": ATTACH, "area": HAND, "index": 0,
         "inPlayArea": BENCH, "inPlayIndex": 99},
        current, ME,
    )
    assert action["card_id"] == 7
    assert action["target_card_id"] is None
    assert action["resolved"] is True


def test_resolve_does_not_mutate_inputs(current):
    import copy

    before_option = {"type": PLAY, "index": 0}
    before_current = copy.deepcopy(current)
    resolve_option(before_option, current, ME)
    assert before_option == {"type": PLAY, "index": 0}
    assert current == before_current


# --- action_label ----------------------------------------------------------


def test_label_ignores_serial(current):
    """serial は試合ごとに振り直される。含めると別エピソード間で常に不一致になり、
    上位エージェント間の一致率が測れなくなる。"""
    a = resolve_option({"type": PLAY, "index": 0}, current, ME)
    b = dict(a, serial=99999)
    assert action_label(a) == action_label(b)


def test_label_distinguishes_target(current):
    to_active = resolve_option(
        {"type": ATTACH, "area": HAND, "index": 0, "inPlayArea": ACTIVE, "inPlayIndex": 0},
        current, ME,
    )
    to_bench = resolve_option(
        {"type": ATTACH, "area": HAND, "index": 0, "inPlayArea": BENCH, "inPlayIndex": 0},
        current, ME,
    )
    assert action_label(to_active) != action_label(to_bench)


def test_label_distinguishes_option_type(current):
    play = resolve_option({"type": PLAY, "index": 0}, current, ME)
    attach = resolve_option({"type": ATTACH, "area": HAND, "index": 0}, current, ME)
    assert action_label(play) != action_label(attach)


# --- 不変条件（v2 §4.1 FR-ACT-005） ---------------------------------------


def test_main_with_exactly_one_end_is_clean():
    stats = ResolutionStats()
    select = {"context": 0, "option": [{"type": PLAY, "index": 0}, {"type": END}]}
    check_invariants(select, stats)
    assert stats.invariant_violations == {}


@pytest.mark.parametrize(
    "options",
    [
        [{"type": PLAY, "index": 0}],                     # END なし
        [{"type": END}, {"type": END}],                   # END が2つ
    ],
)
def test_main_without_exactly_one_end_is_flagged(options):
    stats = ResolutionStats()
    check_invariants({"context": 0, "option": options}, stats)
    assert stats.invariant_violations["main_has_exactly_one_end"] == 1


def test_end_invariant_applies_only_to_main():
    """MAIN 以外の context に END は無くてよい。"""
    stats = ResolutionStats()
    check_invariants({"context": 7, "option": [{"type": CARD, "area": HAND, "index": 0}]}, stats)
    assert stats.invariant_violations == {}


@pytest.mark.parametrize("option_type", [ATTACH, EVOLVE])
def test_attach_and_evolve_must_come_from_hand(option_type):
    stats = ResolutionStats()
    select = {"context": 7, "option": [{"type": option_type, "area": DISCARD, "index": 0}]}
    check_invariants(select, stats)
    assert stats.invariant_violations["attach_evolve_area_is_hand"] == 1


def test_ability_area_must_be_in_play():
    stats = ResolutionStats()
    check_invariants(
        {"context": 7, "option": [{"type": ABILITY, "area": HAND, "index": 0}]}, stats
    )
    assert stats.invariant_violations["ability_area_in_play"] == 1


@pytest.mark.parametrize("area", [ACTIVE, BENCH, STADIUM])
def test_ability_in_play_areas_are_clean(area):
    stats = ResolutionStats()
    check_invariants({"context": 7, "option": [{"type": ABILITY, "area": area, "index": 0}]}, stats)
    assert stats.invariant_violations == {}


def test_invariant_violations_never_raise():
    """エンジン更新で破れ得るため、例外ではなく件数集計に留めること。"""
    stats = ResolutionStats()
    check_invariants({"context": 0, "option": []}, stats)   # 空でも落ちない
    check_invariants({"context": 0}, stats)                 # option キーが無くても落ちない
    assert stats.invariant_violations["main_has_exactly_one_end"] == 2


# --- サマリ出力 ------------------------------------------------------------


def test_stats_as_dict_shape(current):
    stats = ResolutionStats()
    resolve_option({"type": PLAY, "index": 0}, current, ME, stats)
    resolve_option({"type": PLAY, "index": 99}, current, ME, stats)
    summary = stats.as_dict()
    assert summary["resolved"] == 1
    assert summary["failed"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["failed_by_option_type"] == {PLAY: 1}
