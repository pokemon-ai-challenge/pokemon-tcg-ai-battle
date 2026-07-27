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
from ptcg_ai.learning.semantic_action import ATTACH, ATTACK, END, PLAY, resolve_option


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


# --- 状態×行動 交互作用 ------------------------------------------------------
#
# ここが本モジュールの中心的な回帰テスト。実測(200 decision)で state_features()
# の出力の61%が同一 decision 内の全選択肢で同じ値になっており、softmax/argmax は
# ``softmax(score + c) == softmax(score)`` の性質上、全選択肢に同じ値が乗る特徴を
# 完全に無視する。つまり状態特徴を単体で足しても方策は盤面を一切見ない。
# state_action_interaction_features() は状態を option_type と外積することで、
# 「同じ decision でも選択肢(option_type)が変われば値が変わる」特徴を作る。
# これらのテストは、その性質(=選択肢間で実際に変化すること)を直接検証する。
# 将来この関数が state 単体の特徴に戻されたり、外積が壊れたりした場合に検知する。


def _min_state(**overrides) -> dict:
    base = {
        "turn": 1,
        "energy_attached": False,
        "retreated": False,
        "supporter_played": False,
        "stadium_played": False,
        "own": {"active": None, "bench": [], "n_prize": 6, "n_hand": 0, "n_deck": 0, "discard": []},
        "opponent": {"active": None, "bench": [], "n_prize": 6, "n_hand": 0, "n_deck": 0, "discard": []},
    }
    base.update(overrides)
    return base


def test_state_features_alone_are_identical_across_options_but_interactions_differ():
    """同一 decision(同一 state)で option_type だけが違う2つの行動を比較する。

    state_features(state) は両者で完全に同一(状態は decision 内で不変なため)。
    これが「61%が定数だった」問題そのもの。extract_features() はこれに
    state_action_interaction_features() を足すことで、選択肢間で異なる特徴
    (option_type_{t}__turn_opening 等)を持つようになっていることを確認する。
    """
    state = _min_state(turn=1)
    action_play = resolve_option({"type": PLAY, "index": 0}, _current(), 0)
    action_end = resolve_option({"type": END}, _current(), 0)

    # state_features 単体は action に依存しないので当然同一(これは仕様どおり)。
    assert policy_features.state_features(state) == policy_features.state_features(state)

    feats_play = policy_features.extract_features(state, action_play)
    feats_end = policy_features.extract_features(state, action_end)

    # option_type ごとに別の特徴名になるため、交互作用特徴のキー自体が異なる
    # (= 選択肢間で「値が変化する」特徴として現れる)。
    assert f"option_type_{PLAY}__turn_opening" in feats_play
    assert f"option_type_{END}__turn_opening" in feats_end
    assert f"option_type_{PLAY}__turn_opening" not in feats_end
    assert f"option_type_{END}__turn_opening" not in feats_play


def test_interaction_own_and_opp_active_card_id():
    state = _min_state()
    state["own"]["active"] = {"card_id": 646, "hp": 90, "max_hp": 90}
    state["opponent"]["active"] = {"card_id": 743, "hp": 90, "max_hp": 90}
    action = resolve_option({"type": ATTACH, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0}, _current(), 0)

    interactions = policy_features.state_action_interaction_features(
        state, action, frequent_card_ids=[646, 743, 1]
    )
    assert interactions[f"option_type_{ATTACH}__own_active_card_646"] == 1.0
    assert interactions[f"option_type_{ATTACH}__opp_active_card_743"] == 1.0

    # 上位N(active_card_top_n)に入らないカードは one-hot にならない。
    interactions_excluded = policy_features.state_action_interaction_features(
        state, action, frequent_card_ids=[1], active_card_top_n=1
    )
    assert f"option_type_{ATTACH}__own_active_card_646" not in interactions_excluded


def test_interaction_prize_bucket_changes_with_prize_count():
    action = resolve_option({"type": END}, _current(), 0)

    state_early = _min_state()
    state_early["own"]["n_prize"] = 6
    state_late = _min_state()
    state_late["own"]["n_prize"] = 0

    feats_early = policy_features.state_action_interaction_features(state_early, action)
    feats_late = policy_features.state_action_interaction_features(state_late, action)

    assert feats_early[f"option_type_{END}__own_prize_early"] == 1.0
    assert feats_late[f"option_type_{END}__own_prize_late"] == 1.0
    # 同じ特徴名が両方には立たない(バケットが変わっている)。
    assert f"option_type_{END}__own_prize_early" not in feats_late


def test_interaction_hp_ratio_bucket():
    action = resolve_option({"type": ATTACK}, _current(), 0)

    state_low_hp = _min_state()
    state_low_hp["opponent"]["active"] = {"card_id": 1, "hp": 10, "max_hp": 100}
    feats = policy_features.state_action_interaction_features(state_low_hp, action)
    assert feats[f"option_type_{ATTACK}__opp_active_hp_low"] == 1.0

    state_high_hp = _min_state()
    state_high_hp["opponent"]["active"] = {"card_id": 1, "hp": 95, "max_hp": 100}
    feats_high = policy_features.state_action_interaction_features(state_high_hp, action)
    assert feats_high[f"option_type_{ATTACK}__opp_active_hp_high"] == 1.0


def test_interaction_turn_restriction_flags():
    action = resolve_option({"type": PLAY, "index": 0}, _current(), 0)
    state = _min_state(energy_attached=True, supporter_played=False)
    feats = policy_features.state_action_interaction_features(state, action)
    assert feats[f"option_type_{PLAY}__flag_energy_attached"] == 1.0
    assert f"option_type_{PLAY}__flag_supporter_played" not in feats


def test_interaction_features_absent_when_option_type_missing():
    """option_type が無い行動には交互作用特徴を作らない(_tag が組み立てられないため)。"""
    interactions = policy_features.state_action_interaction_features(_min_state(), {})
    assert interactions == {}
