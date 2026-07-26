"""``ptcg_ai.learning.policy_features`` の特徴抽出を検証する。

最重要の回帰テストは FR-ACT-006(index / serial を特徴に使わない)。ここが壊れると
モデルが index の偽相関を学習してしまい、精度低下の原因が「モデルが悪いのか特徴抽出が
壊れているのか」切り分けられなくなる。
"""

from pathlib import Path
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from ptcg_ai.learning import policy_features
from ptcg_ai.learning.semantic_action import ATTACH, END, PLAY, resolve_option


def _state() -> dict:
    """``observable_state()`` の出力に相当する最小の状態 dict。"""
    return {
        "turn": 5,
        "turn_action_count": 2,
        "first_player": 0,
        "energy_attached": True,
        "retreated": False,
        "supporter_played": True,
        "stadium_played": False,
        "stadium": [900],
        "own": {
            "active": {
                "card_id": 646, "hp": 60, "max_hp": 90,
                "n_energy": 2, "n_tool": 0, "appear_this_turn": False,
            },
            "bench": [
                {"card_id": 112, "hp": 110, "max_hp": 110, "n_energy": 1, "n_tool": 0, "appear_this_turn": False},
            ],
            "n_prize": 3, "n_hand": 4, "n_deck": 40,
            "discard": [19],
            "asleep": False, "confused": False, "paralyzed": False,
            "poisoned": False, "burned": False,
        },
        "opponent": {
            "active": None,
            "bench": [],
            "n_prize": 6, "n_hand": 5, "n_deck": 44,
            "discard": [],
            "asleep": False, "confused": False, "paralyzed": False,
            "poisoned": False, "burned": False,
        },
    }


def _current() -> dict:
    return {
        "yourIndex": 0,
        "players": [
            {"hand": [{"id": 7, "serial": 3, "playerIndex": 0}], "active": [], "bench": [],
             "discard": [], "prize": [], "handCount": 1, "deckCount": 40},
            {"hand": None, "active": [], "bench": [], "discard": [], "prize": [None] * 6,
             "handCount": 5, "deckCount": 44},
        ],
        "stadium": [],
    }


# --- FR-ACT-006: index / serial を特徴に使わない ---------------------------


def test_features_never_contain_index_or_serial_keys():
    """action_features() が返す特徴名に index/serial 系のキーが一切含まれないこと。"""
    action = resolve_option(
        {"type": ATTACH, "area": 2, "index": 0, "inPlayArea": 5, "inPlayIndex": 1},
        _current(), 0,
    )
    features = policy_features.action_features(action)
    for name in features:
        assert "index" not in name.lower(), f"禁止された特徴名: {name}"
        assert "serial" not in name.lower(), f"禁止された特徴名: {name}"


def test_extract_features_never_contain_index_or_serial_keys():
    """state + action を合わせた最終出力でも同じ性質が保たれること(統合の回帰テスト)。"""
    action = resolve_option({"type": PLAY, "index": 2}, _current(), 0)
    features = policy_features.extract_features(_state(), action)
    for name in features:
        assert "index" not in name.lower()
        assert "serial" not in name.lower()


def test_two_actions_differing_only_by_index_produce_identical_features():
    """index だけが違う(= 同じカードを指す)2つの action は同一特徴になるべき。

    もし index を特徴に混入させていたら、この2つは別物として扱われてしまう。
    """
    current = _current()
    current["players"][0]["hand"] = [
        {"id": 7, "serial": 3, "playerIndex": 0},
        {"id": 7, "serial": 99, "playerIndex": 0},
    ]
    action_a = resolve_option({"type": PLAY, "index": 0}, current, 0)
    action_b = resolve_option({"type": PLAY, "index": 1}, current, 0)
    assert policy_features.action_features(action_a) == policy_features.action_features(action_b)


# --- 状態特徴 ---------------------------------------------------------------


def test_state_features_basic_fields():
    features = policy_features.state_features(_state())
    assert features["turn"] == 5.0
    assert features["energy_attached"] == 1.0
    assert features["retreated"] == 0.0
    assert features["own_active_present"] == 1.0
    assert features["own_active_hp"] == 60.0
    assert features["opp_active_present"] == 0.0
    assert features["own_n_hand"] == 4.0
    assert features["opp_n_prize"] == 6.0
    assert features["own_bench_count"] == 1.0


def test_state_features_tolerates_missing_fields():
    """状態が壊れていても例外を投げないこと(ターンを止めないため)。"""
    features = policy_features.state_features({})
    assert features["turn"] == 0.0
    assert features["own_active_present"] == 0.0


# --- 行動特徴: one-hot / 静的属性 / 交互作用 ---------------------------------


def test_option_type_one_hot():
    action = resolve_option({"type": END}, _current(), 0)
    features = policy_features.action_features(action)
    assert features[f"option_type_{END}"] == 1.0


def test_frequent_card_id_one_hot_only_for_listed_ids():
    action = resolve_option({"type": PLAY, "index": 0}, _current(), 0)  # card_id=7
    features_in = policy_features.action_features(action, frequent_card_ids=[7, 19])
    features_out = policy_features.action_features(action, frequent_card_ids=[19])
    assert features_in.get("card_id_7") == 1.0
    assert "card_id_7" not in features_out


def test_card_attribute_features_and_interaction():
    action = resolve_option({"type": PLAY, "index": 0}, _current(), 0)  # card_id=7
    card_attributes = {"7": {"hp": 60.0, "stage": 0.0}}
    features = policy_features.action_features(action, card_attributes=card_attributes)
    assert features["card_attr_hp"] == 60.0
    assert features["card_attr_stage"] == 0.0
    # option_type x card attribute の外積が明示的な線形項として存在すること。
    assert features[f"option_type_{PLAY}__attr_hp"] == 60.0


def test_unknown_card_id_has_no_attribute_features():
    action = resolve_option({"type": PLAY, "index": 0}, _current(), 0)  # card_id=7
    features = policy_features.action_features(action, card_attributes={"999": {"hp": 1.0}})
    assert "card_attr_hp" not in features
