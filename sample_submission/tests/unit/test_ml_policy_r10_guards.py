"""r10 の3ガード(Fix-F/G/H)のユニットテスト。

実ラダーのリプレイ分析で確定した「誤爆余地のほぼ無い無料改善」3種:

  Fix-F ``tool_stadium_guard``        どうぐ無効スタジアム(ジャミングタワー1246)下での
                                      どうぐ装着veto  — 実例 episode 93408551 T10 row129
  Fix-G ``energy_to_active_first``    攻撃コスト未充足のアクティブを差し置いてベンチへ
                                      エネを付ける手の振替 — 実例 93408551 T6 rows84/86/88
  Fix-H ``search_pick_pokemon_first`` 「山札の上からN枚見て手札に加える」でたねポケモンを
                                      拾う — 実例 93473767 T3 row37(むしとりセット)

いずれも独立 config キーで既定OFF。**キーが無ければ発火しない(本番不変)**ことを各ガードで
1本ずつ確認する。既存の `test_ml_policy_agent.py` は別担当が同時に編集しているため、
競合を避けてこの新ファイルに追加している(検証対象は同じ `ml_policy_agent`)。

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
"""

import copy
import json
import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_ENCODER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"

# 実カードID(data/JP_Card_Data.csv / all_card_data() で分類を確認済み)。
_JAMMING_TOWER = 1246   # スタジアム: おたがいのどうぐの効果をすべて無くす
_EXCITE_STADIUM = 1251  # スタジアム: どうぐを無効化しない(誤発火の対照)
_HERO_CAPE = 1159       # ポケモンのどうぐ(CardType.TOOL)
_BUG_CATCHING_SET = 1094  # グッズ(CardType.ITEM)
_GRASS_ENERGY = 1       # 基本【草】エネルギー(CardType.BASIC_ENERGY)
_OGERPON = 96           # オーガポン みどりのめん ex(たね。特性=みどりのまい)

_TOOL_GUARD_ON = {
    "lethal_search": {"enabled": False},
    "tool_stadium_guard": {"enabled": True, "nullifying_stadium_ids": [_JAMMING_TOWER]},
}
_ENERGY_FIRST_ON = {
    "lethal_search": {"enabled": False},
    "energy_to_active_first": {"enabled": True},
}
_SEARCH_PICK_ON = {
    "lethal_search": {"enabled": False},
    "search_pick_pokemon_first": {"enabled": True},
}
_OFF = {"lethal_search": {"enabled": False}}


@pytest.fixture(scope="module")
def ml_policy_agent():
    try:
        from ptcg_ai.ml_policy import ml_policy_agent as mod
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"ml_policy_agent / cg engine unavailable: {exc}")
    return mod


@pytest.fixture(scope="module")
def encoder_observations() -> dict:
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


class _StubScoreModel:
    """`score_options` が固定スコアを返すだけのモデル(差し替え先の決定を検証可能にする)。"""

    def __init__(self, scores):
        self._scores = list(scores)

    def score_options(self, obs, hidden_state_factory=None, deadline=None):
        return list(self._scores)

    def select_option(self, obs, hidden_state_factory=None, deadline=None):
        return max(range(len(self._scores)), key=lambda i: self._scores[i])


def _card(card_id: int, serial: int, player_index: int = 0) -> dict:
    return {"id": card_id, "playerIndex": player_index, "serial": serial}


def _build_obs(
    encoder_observations,
    options,
    *,
    hand_ids=None,
    stadium_id=None,
    active_id=None,
    bench_ids=None,
    bench_max=5,
    looking_ids=None,
    select_type=0,
    context=0,
    min_count=1,
    max_count=1,
):
    """mid_game を土台に、各ガードの判定材料だけを明示した合成 obs を作る。

    `options` は cg.api.Option 相当の dict のリスト(そのまま select.option に入る)。
    """
    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    state = base["current"]
    me = state["players"][0]
    if hand_ids is not None:
        me["hand"] = [_card(cid, 900 + i) for i, cid in enumerate(hand_ids)]
        me["handCount"] = len(hand_ids)
    if stadium_id is None:
        state["stadium"] = []
    else:
        state["stadium"] = [_card(stadium_id, 61)]
    if active_id is not None:
        me["active"][0]["id"] = active_id
    if bench_ids is not None:
        proto = copy.deepcopy(me["bench"][0])
        bench = []
        for i, cid in enumerate(bench_ids):
            slot = copy.deepcopy(proto)
            slot["id"] = cid
            slot["serial"] = 700 + i
            bench.append(slot)
        me["bench"] = bench
    me["benchMax"] = bench_max
    state["looking"] = (
        None if looking_ids is None
        else [_card(cid, 800 + i) for i, cid in enumerate(looking_ids)]
    )
    base["select"] = {
        "context": context,
        "contextCard": None,
        "deck": None,
        "effect": None,
        "maxCount": max_count,
        "minCount": min_count,
        "option": list(options),
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "type": select_type,
    }
    base["logs"] = []
    return to_observation_class(base)


# ---------------------------------------------------------------------------
# option ビルダ(cg/api.py の OptionType コメントのフィールド構成に合わせる)
# ---------------------------------------------------------------------------

def _play(hand_index: int) -> dict:
    return {"type": 7, "index": hand_index}


def _attach(hand_index: int, *, to_active: bool, in_play_index: int = 0) -> dict:
    """手札 `hand_index` のカードを場のポケモンに付ける ATTACH(area=HAND, inPlayArea=付け先)。"""
    return {
        "type": 8, "area": 2, "index": hand_index,
        "inPlayArea": 4 if to_active else 5, "inPlayIndex": in_play_index,
    }


def _ability(*, on_active: bool, index: int = 0) -> dict:
    return {"type": 10, "area": 4 if on_active else 5, "index": index}


def _attack(attack_id: int = 120) -> dict:
    return {"type": 13, "attackId": attack_id}


def _end() -> dict:
    return {"type": 14}


def _looking_option(looking_index: int) -> dict:
    return {"type": 3, "area": 12, "index": looking_index, "playerIndex": 0}


# ===========================================================================
# Fix-F: tool_stadium_guard
# ===========================================================================

def _tool_obs(encoder_observations, *, stadium_id=_JAMMING_TOWER, options=None):
    return _build_obs(
        encoder_observations,
        options if options is not None else [
            _attach(0, to_active=True),   # 0: ヒーローマント装着(無効スタジアム下=無駄撃ち)
            _play(1),                     # 1: むしとりセット(グッズ)
            _end(),                       # 2: ターン終了
        ],
        hand_ids=[_HERO_CAPE, _BUG_CATCHING_SET, _GRASS_ENERGY],
        stadium_id=stadium_id,
    )


def test_tool_stadium_guard_absent_key_is_inert(ml_policy_agent, encoder_observations, monkeypatch):
    """キー無し=本番configでは常に None(モデルも取りに行かない)。"""
    obs = _tool_obs(encoder_observations)

    def _must_not_be_called(config=None):
        raise AssertionError("tool_stadium_guard disabled なのにモデルを取得した")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_tool_stadium_guard(obs, [0], config=_OFF) is None
    assert ml_policy_agent._try_tool_stadium_guard(obs, [0], config={}) is None


def test_tool_stadium_guard_swaps_tool_attach_under_jamming_tower(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """ジャミングタワー下のどうぐ装着 → どうぐ以外で方策スコア最大の手へ差し替え。"""
    obs = _tool_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.4, 0.1]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_tool_stadium_guard(obs, [0], config=_TOOL_GUARD_ON) == [1]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["tool_stadium_guard_fired"] == 1
    assert stats["tool_stadium_guard_misfire_no_stadium"] == 0


def test_tool_stadium_guard_does_not_fire_without_nullifying_stadium(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """どうぐを無効化しないスタジアム(エキサイトスタジアム)/スタジアム無しでは不介入。"""
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.4, 0.1]))
    for stadium_id in (_EXCITE_STADIUM, None):
        obs = _tool_obs(encoder_observations, stadium_id=stadium_id)
        assert ml_policy_agent._try_tool_stadium_guard(obs, [0], config=_TOOL_GUARD_ON) is None


def test_tool_stadium_guard_ignores_non_tool_choices(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """無効スタジアム下でも、選んだ手がどうぐでなければ(グッズ/エネ付け)干渉しない。"""
    obs = _tool_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.4, 0.1]))
    assert ml_policy_agent._try_tool_stadium_guard(obs, [1], config=_TOOL_GUARD_ON) is None
    assert ml_policy_agent._try_tool_stadium_guard(obs, [2], config=_TOOL_GUARD_ON) is None


def test_tool_stadium_guard_no_alternative_does_not_intervene(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """選択肢がどうぐ装着しか無ければ介入しない(代替不在カウンタだけ進む)。"""
    obs = _tool_obs(encoder_observations, options=[
        _attach(0, to_active=True), _attach(0, to_active=False),
    ])
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.4]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_tool_stadium_guard(obs, [0], config=_TOOL_GUARD_ON) is None
    assert ml_policy_agent.get_guard_stats()["tool_stadium_guard_no_alternative"] == 1
    assert ml_policy_agent.get_guard_stats()["tool_stadium_guard_fired"] == 0


def test_tool_stadium_guard_wired_into_apply_action_vetoes(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """veto連鎖(実戦の呼ばれ方)経由でも差し替わること/キー無しでは不変であること。"""
    obs = _tool_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.4, 0.1]))
    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_TOOL_GUARD_ON) == [1]
    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_OFF) == [0]


# ===========================================================================
# Fix-G: energy_to_active_first
# ===========================================================================

def _energy_obs(encoder_observations, *, options=None, with_attack=False):
    """アクティブ/ベンチともオーガポン(96)、手札に基本草エネ1枚の MAIN 局面。"""
    if options is None:
        options = [
            _attach(0, to_active=False),  # 0: 手貼りをベンチへ
            _attach(0, to_active=True),   # 1: 手貼りをアクティブへ
            _ability(on_active=False),    # 2: みどりのまい(ベンチ個体)
            _ability(on_active=True),     # 3: みどりのまい(アクティブ)
            _end(),                       # 4
        ]
    if with_attack:
        options = [*options, _attack()]
    return _build_obs(
        encoder_observations, options,
        hand_ids=[_GRASS_ENERGY, _BUG_CATCHING_SET],
        active_id=_OGERPON, bench_ids=[_OGERPON],
    )


def test_energy_to_active_first_absent_key_is_inert(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """キー無し=本番configでは常に None(モデルも取りに行かない)。"""
    obs = _energy_obs(encoder_observations)

    def _must_not_be_called(config=None):
        raise AssertionError("energy_to_active_first disabled なのにモデルを取得した")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_energy_to_active_first(obs, [2], config=_OFF) is None
    assert ml_policy_agent._try_energy_to_active_first(obs, [2], config={}) is None


def test_energy_to_active_first_swaps_bench_ability_to_active(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """攻撃不能(ATTACK選択肢なし)でベンチにみどりのまい → アクティブの同特性へ振替。"""
    obs = _energy_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.1, 0.2, 1.0, 0.3, 0.0]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_energy_to_active_first(obs, [2], config=_ENERGY_FIRST_ON) == [3]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["energy_to_active_first_fired"] == 1
    assert stats["energy_to_active_first_misfire_can_attack"] == 0


def test_energy_to_active_first_swaps_bench_manual_attach_to_active(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """手貼りをベンチへ付ける手も、同じ種類(ATTACH)のアクティブ向けへ振り替える。"""
    obs = _energy_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.3, 0.9, 0.0]))

    # 同種(ATTACH)優先: 方策スコアが最大の ABILITY(idx3) ではなく ATTACH のアクティブ(idx1)。
    assert ml_policy_agent._try_energy_to_active_first(obs, [0], config=_ENERGY_FIRST_ON) == [1]


def test_energy_to_active_first_does_not_fire_when_attack_available(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """ATTACK 選択肢がある=既に攻撃コストを満たしている → 干渉しない(誤爆の対照)。"""
    obs = _energy_obs(encoder_observations, with_attack=True)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.1, 0.2, 1.0, 0.3, 0.0, 0.5]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_energy_to_active_first(obs, [2], config=_ENERGY_FIRST_ON) is None
    assert ml_policy_agent.get_guard_stats()["energy_to_active_first_fired"] == 0


def test_energy_to_active_first_ignores_active_target_and_other_options(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """既にアクティブへ付ける手/エネ付け以外(END)を選んでいるなら干渉しない。"""
    obs = _energy_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.1, 0.2, 1.0, 0.3, 0.0]))
    for chosen in ([1], [3], [4]):
        assert ml_policy_agent._try_energy_to_active_first(
            obs, chosen, config=_ENERGY_FIRST_ON) is None


def test_energy_to_active_first_no_active_option_does_not_intervene(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """アクティブへ付けられる選択肢が無ければ介入しない(ベンチ育成しか無い局面は尊重)。"""
    obs = _energy_obs(encoder_observations, options=[
        _attach(0, to_active=False), _ability(on_active=False), _end(),
    ])
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.5, 1.0, 0.0]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_energy_to_active_first(obs, [1], config=_ENERGY_FIRST_ON) is None
    assert ml_policy_agent.get_guard_stats()["energy_to_active_first_no_active_option"] == 1


def test_energy_to_active_first_ignores_unknown_ability(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """`energy_ability_card_ids` に無いポケモンの特性は「エネ付け」とみなさない(誤爆防止)。"""
    obs = _energy_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.1, 0.2, 1.0, 0.3, 0.0]))
    cfg = {"lethal_search": {"enabled": False},
           "energy_to_active_first": {"enabled": True, "energy_ability_card_ids": [12345]}}
    assert ml_policy_agent._try_energy_to_active_first(obs, [2], config=cfg) is None


def test_energy_to_active_first_wired_into_apply_action_vetoes(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """veto連鎖経由でも振り替わること/キー無しでは不変であること。"""
    obs = _energy_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.1, 0.2, 1.0, 0.3, 0.0]))
    assert ml_policy_agent._apply_action_vetoes(obs, [2], config=_ENERGY_FIRST_ON) == [3]
    assert ml_policy_agent._apply_action_vetoes(obs, [2], config=_OFF) == [2]


# ===========================================================================
# Fix-H: search_pick_pokemon_first
# ===========================================================================

def _search_obs(
    encoder_observations, *, looking_ids=(_GRASS_ENERGY, _OGERPON, _GRASS_ENERGY),
    option_indices=(0, 1, 2), bench_ids=(756,), bench_max=5, max_count=2, options=None,
):
    """むしとりセット相当の「山札の上から見た札を手札に加える」select(context=TO_HAND=7)。"""
    return _build_obs(
        encoder_observations,
        options if options is not None else [_looking_option(i) for i in option_indices],
        hand_ids=[_BUG_CATCHING_SET],
        looking_ids=list(looking_ids),
        bench_ids=list(bench_ids),
        bench_max=bench_max,
        select_type=1, context=7, min_count=0, max_count=max_count,
    )


def test_search_pick_pokemon_first_absent_key_is_inert(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """キー無し=本番configでは常に None(モデルも取りに行かない)。"""
    obs = _search_obs(encoder_observations)

    def _must_not_be_called(config=None):
        raise AssertionError("search_pick_pokemon_first disabled なのにモデルを取得した")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_search_pick_pokemon_first(obs, [0, 2], config=_OFF) is None
    assert ml_policy_agent._try_search_pick_pokemon_first(obs, [0, 2], config={}) is None


def test_search_pick_pokemon_first_replaces_worst_energy_with_pokemon(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """枠が埋まっている(2/2)ならスコア最小の選択を1つだけポケモンと入れ替える。"""
    obs = _search_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.9, 0.1, 0.2]))
    ml_policy_agent.reset_guard_stats()

    # 選択は [0, 2](エネ2枚)。スコア最小は idx2 なので、そこをポケモン(idx1)に入れ替える。
    assert ml_policy_agent._try_search_pick_pokemon_first(
        obs, [0, 2], config=_SEARCH_PICK_ON) == [0, 1]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["search_pick_pokemon_first_fired"] == 1
    assert stats["search_pick_pokemon_first_replaced"] == 1
    assert stats["search_pick_pokemon_first_misfire_bench_full"] == 0


def test_search_pick_pokemon_first_appends_when_slot_remains(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """枠が余っている(1/2)なら入れ替えずに「足す」。"""
    obs = _search_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.9, 0.1, 0.2]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_search_pick_pokemon_first(
        obs, [0], config=_SEARCH_PICK_ON) == [0, 1]
    assert ml_policy_agent.get_guard_stats()["search_pick_pokemon_first_appended"] == 1


def test_search_pick_pokemon_first_does_not_fire_when_pokemon_already_taken(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """既にポケモンを選んでいるなら不介入(2枚目を取りに行かない)。"""
    obs = _search_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.9, 0.1, 0.2]))
    assert ml_policy_agent._try_search_pick_pokemon_first(
        obs, [1, 2], config=_SEARCH_PICK_ON) is None


def test_search_pick_pokemon_first_does_not_fire_when_bench_full(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """ベンチ満杯=拾っても出せないので不介入(誤爆の対照)。"""
    obs = _search_obs(encoder_observations, bench_ids=(756,), bench_max=1)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.9, 0.1, 0.2]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_search_pick_pokemon_first(
        obs, [0, 2], config=_SEARCH_PICK_ON) is None
    stats = ml_policy_agent.get_guard_stats()
    assert stats["search_pick_pokemon_first_fired"] == 0
    assert stats["search_pick_pokemon_first_misfire_bench_full"] == 0


def test_search_pick_pokemon_first_does_not_fire_without_pokemon_candidate(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """looking にたねポケモンが無ければ不介入(93408551 T10 row131 と同じ形)。"""
    obs = _search_obs(
        encoder_observations,
        looking_ids=(_GRASS_ENERGY, _GRASS_ENERGY, _BUG_CATCHING_SET),
    )
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.9, 0.1, 0.2]))
    assert ml_policy_agent._try_search_pick_pokemon_first(
        obs, [0, 1], config=_SEARCH_PICK_ON) is None


def test_search_pick_pokemon_first_ignores_non_looking_selects(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """LOOKING 以外(実体が解決できない候補)が混ざる select には介入しない。"""
    obs = _search_obs(encoder_observations, options=[
        _looking_option(0), {"type": 3, "area": 1, "index": 0, "playerIndex": 0}, _looking_option(1),
    ])
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.9, 0.1, 0.2]))
    assert ml_policy_agent._try_search_pick_pokemon_first(
        obs, [0, 1], config=_SEARCH_PICK_ON) is None


def test_search_pick_pokemon_first_wired_into_apply_action_vetoes(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """veto連鎖経由でも差し替わること/キー無しでは不変であること。"""
    obs = _search_obs(encoder_observations)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([0.9, 0.1, 0.2]))
    assert ml_policy_agent._apply_action_vetoes(obs, [0, 2], config=_SEARCH_PICK_ON) == [0, 1]
    assert ml_policy_agent._apply_action_vetoes(obs, [0, 2], config=_OFF) == [0, 2]


# ===========================================================================
# 共通: 3ガードとも「新キーが無ければ何も起きない」(本番不変)
# ===========================================================================

def test_all_r10_guards_are_inert_without_keys(ml_policy_agent, encoder_observations, monkeypatch):
    """abl_5_full 相当(新キー無し)では3ガードとも None を返し、統計も動かない。"""
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.4, 0.1, 0.2, 0.0, 0.3]))
    ml_policy_agent.reset_guard_stats()
    before = ml_policy_agent.get_guard_stats()

    cases = [
        (ml_policy_agent._try_tool_stadium_guard, _tool_obs(encoder_observations), [0]),
        (ml_policy_agent._try_energy_to_active_first, _energy_obs(encoder_observations), [2]),
        (ml_policy_agent._try_search_pick_pokemon_first, _search_obs(encoder_observations), [0, 2]),
    ]
    for fn, obs, chosen in cases:
        assert fn(obs, chosen, config=_OFF) is None
        assert ml_policy_agent._apply_action_vetoes(obs, list(chosen), config=_OFF) == chosen
    assert ml_policy_agent.get_guard_stats() == before
