"""Unit tests for ptcg_ai.board_evaluation.consequence (Tier3 Stage3a/3b).

docs/plans/policy-feature-expansion/tier3-consequence-features-design-and-implementation-plan.md
Stage3a DoD: search_stepによる特徴生成が正しく動く(固定盤面を突合)。
Stage3b DoD: §6.1の受け入れテスト(ロック闘エネルギー+改造ハンマー)、transaction解決が
代表的な複数選択カードで正しく動く。

test_lethal_simple.py と同じ方針: 対戦エンジン(cg.api)を手作りの決定木(FakeEngine)で
差し替え、探索ロジック自体(候補ATTACKの列挙・最大値の取得・コイン除外・タイムアウト・
transaction解決)を実ゲームエンジン無しでテストする。カード効果の実解決(ロックエネルギー・
改造ハンマー等)そのものはfakeでは検証せず、状態遷移として手で与える。ただし
``card_cache``(``opp_special_energy_removed`` の判定に使う)は実カードデータ
(``cg.api.all_card_data()``)をそのまま使うため、テストのエネルギーカードには実在の
card_id(Rock Fighting Energy=20、Basic {W} Energy=3)を使う。
実エンジンでの盤面再現は results/2026-07-22_attackplan_rock_fighting_energy_reproduction.md 参照。

Run from sample_submission/:
    python -m pytest tests/unit/test_consequence.py -q
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import (
    Card,
    Log,
    LogType,
    Observation,
    Option,
    OptionType,
    Pokemon,
    PlayerState,
    SearchState,
    SelectContext,
    SelectData,
    SelectType,
    State,
)
from ptcg_ai.board_evaluation import consequence

# Rock Fighting Energy(SPECIAL_ENERGY) / Basic {W} Energy(BASIC_ENERGY)。
# data/EN_Card_Data.csv で実在確認済み(2026-07-22セッションのカード調査で確認)。
ROCK_FIGHTING_ENERGY_ID = 20
BASIC_WATER_ENERGY_ID = 3

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_energy_card(*, serial: int, card_id: int, player_index: int = 1) -> Card:
    return Card(id=card_id, serial=serial, playerIndex=player_index)


def make_pokemon(
    *, serial: int, hp: int, max_hp: int = 200, card_id: int = 1,
    energy_cards: list[Card] | None = None,
) -> Pokemon:
    return Pokemon(
        id=card_id, serial=serial, hp=hp, maxHp=max_hp, appearThisTurn=False,
        energies=[], energyCards=energy_cards or [], tools=[], preEvolution=[],
    )


def make_player(
    *, active_hp: int | None = None, active_serial: int = 1, active_card_id: int = 1,
    hand_count: int = 0, energy_cards: list[Card] | None = None,
) -> PlayerState:
    active = (
        [make_pokemon(serial=active_serial, hp=active_hp, card_id=active_card_id, energy_cards=energy_cards)]
        if active_hp is not None else [None]
    )
    return PlayerState(
        active=active, bench=[], benchMax=5, deckCount=10, discard=[],
        prize=[None], handCount=hand_count, hand=[], poisoned=False, burned=False,
        asleep=False, paralyzed=False, confused=False,
    )


def make_state(
    *, your_index: int = 0, my_active_hp: int = 200, opp_active_hp: int = 200,
    my_hand_count: int = 0, opp_energy_cards: list[Card] | None = None,
    stadium: list | None = None,
) -> State:
    mine = make_player(active_hp=my_active_hp, active_serial=1, hand_count=my_hand_count)
    theirs = make_player(active_hp=opp_active_hp, active_serial=2, energy_cards=opp_energy_cards)
    players = [mine, theirs] if your_index == 0 else [theirs, mine]
    return State(
        turn=3, turnActionCount=0, yourIndex=your_index, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
        retreated=False, result=-1, stadium=stadium or [], looking=None, players=players,
    )


def make_select(
    option_types: list[OptionType], *,
    select_type: SelectType = SelectType.MAIN, min_count: int = 1, max_count: int = 1,
) -> SelectData:
    return SelectData(
        type=select_type, context=SelectContext.MAIN, minCount=min_count, maxCount=max_count,
        remainDamageCounter=0, remainEnergyCost=0,
        option=[Option(type=t) for t in option_types],
        deck=None, contextCard=None, effect=None,
    )


def make_obs(state: State, select: SelectData | None, logs: list | None = None) -> Observation:
    return Observation(select=select, logs=logs or [], current=state, search_begin_input="{}")


DUMMY_HIDDEN = {
    "your_deck": [], "your_prize": [], "opponent_deck": [],
    "opponent_prize": [], "opponent_hand": [], "opponent_active": [],
}


def hidden_factory():
    return DUMMY_HIDDEN


# ---------------------------------------------------------------------------
# Fake search engine (test_lethal_simple.py と同じ設計)
# ---------------------------------------------------------------------------


class FakeNode:
    def __init__(self, obs: Observation, transitions: dict | None = None):
        self.obs = obs
        self.transitions = transitions or {}


class FakeEngine:
    """``root``(単一固定起点)または ``entry_points``(``id(obs) -> ノード名`` のマップ、
    ``option_consequence``のように複数の起点から仮実行し直すシナリオ向け)のどちらかで使う。
    両方省略時は ``root="root"`` を既定にする(Stage3aの既存テストとの後方互換)。
    """

    def __init__(
        self,
        nodes: dict[str, FakeNode],
        root: str | None = None,
        entry_points: dict[int, str] | None = None,
    ):
        self.nodes = nodes
        self.root = root if (root is not None or entry_points) else "root"
        self.entry_points = entry_points or {}
        self.next_id = 0
        self.id_to_name: dict[int, str] = {}
        self.released: set[int] = set()
        self.begin_count = 0

    def _make_state(self, name: str) -> SearchState:
        self.next_id += 1
        self.id_to_name[self.next_id] = name
        return SearchState(observation=self.nodes[name].obs, searchId=self.next_id)

    def search_begin(self, obs, *args, **kwargs) -> SearchState:
        self.begin_count += 1
        if self.entry_points:
            name = self.entry_points.get(id(obs))
            if name is None:
                raise AssertionError(f"no entry point registered for obs id={id(obs)}")
        else:
            name = self.root
        return self._make_state(name)

    def search_step(self, search_id: int, select: list[int]) -> SearchState:
        if search_id in self.released:
            raise ValueError("released")
        name = self.id_to_name[search_id]
        node = self.nodes[name]
        key = tuple(sorted(select))
        target = node.transitions.get(key)
        if target is None:
            raise ValueError(f"no transition from {name} for {key}")
        return self._make_state(target)

    def search_release(self, search_id: int) -> None:
        self.released.add(search_id)


@pytest.fixture
def install_engine(monkeypatch):
    def _install(engine: FakeEngine):
        monkeypatch.setattr(consequence, "cg_api", engine)
        return engine
    return _install


# ---------------------------------------------------------------------------
# best_effective_attack_damage
# ---------------------------------------------------------------------------


def test_returns_max_over_multiple_attacks(install_engine):
    root_select = make_select([OptionType.ATTACK, OptionType.ATTACK, OptionType.END])
    root_obs = make_obs(make_state(opp_active_hp=200), root_select)

    weak_obs = make_obs(make_state(opp_active_hp=200), None)   # 0ダメージ(locked相当)
    strong_obs = make_obs(make_state(opp_active_hp=80), None)  # 120ダメージ通った

    engine = install_engine(FakeEngine({
        "root": FakeNode(root_obs, {(0,): "weak", (1,): "strong"}),
        "weak": FakeNode(weak_obs),
        "strong": FakeNode(strong_obs),
    }))

    result = consequence.best_effective_attack_damage(root_obs, hidden_factory, time.perf_counter() + 1)
    assert result == 120
    # 2候補ぶん独立にsearch_beginしている(候補ごとに新規セッション)。
    assert engine.begin_count == 2


def test_zero_when_no_attack_options(install_engine):
    root_select = make_select([OptionType.END])
    root_obs = make_obs(make_state(), root_select)
    engine = install_engine(FakeEngine({"root": FakeNode(root_obs)}))

    result = consequence.best_effective_attack_damage(root_obs, hidden_factory, time.perf_counter() + 1)
    assert result == 0
    assert engine.begin_count == 0  # 候補が無ければ仮実行しない


def test_zero_when_locked(install_engine):
    """ロック闘エネルギー相当: 攻撃を仮実行しても相手HPが変化しない。"""
    root_select = make_select([OptionType.ATTACK])
    root_obs = make_obs(make_state(opp_active_hp=70), root_select)
    after_obs = make_obs(make_state(opp_active_hp=70), None)

    engine = install_engine(FakeEngine({
        "root": FakeNode(root_obs, {(0,): "after"}),
        "after": FakeNode(after_obs),
    }))

    assert consequence.best_effective_attack_damage(root_obs, hidden_factory, time.perf_counter() + 1) == 0


def test_after_hammer_damage_goes_through(install_engine):
    """§3.2.1受け入れ例の「後」半分: ロックが外れた状態でのbest_effective_attack_damage。"""
    root_select = make_select([OptionType.ATTACK])
    root_obs = make_obs(make_state(opp_active_hp=200), root_select)
    after_obs = make_obs(make_state(opp_active_hp=80), None)  # 120ダメージ通った

    engine = install_engine(FakeEngine({
        "root": FakeNode(root_obs, {(0,): "after"}),
        "after": FakeNode(after_obs),
    }))

    assert consequence.best_effective_attack_damage(root_obs, hidden_factory, time.perf_counter() + 1) == 120


def test_coin_attack_excluded(install_engine):
    root_select = make_select([OptionType.ATTACK])
    root_obs = make_obs(make_state(opp_active_hp=200), root_select)
    coin_obs = make_obs(make_state(opp_active_hp=80), None, logs=[Log(type=LogType.COIN)])

    engine = install_engine(FakeEngine({
        "root": FakeNode(root_obs, {(0,): "coin"}),
        "coin": FakeNode(coin_obs),
    }))

    # ダメージは通っているように見えるが、コイン絡みなので不採用(0)。
    assert consequence.best_effective_attack_damage(root_obs, hidden_factory, time.perf_counter() + 1) == 0


def test_hidden_state_unavailable_returns_zero(install_engine):
    root_select = make_select([OptionType.ATTACK])
    root_obs = make_obs(make_state(), root_select)
    engine = install_engine(FakeEngine({"root": FakeNode(root_obs)}))

    assert consequence.best_effective_attack_damage(root_obs, lambda: None, time.perf_counter() + 1) == 0
    assert engine.begin_count == 0


def test_timeout_stops_without_raising(install_engine):
    root_select = make_select([OptionType.ATTACK, OptionType.ATTACK])
    root_obs = make_obs(make_state(opp_active_hp=200), root_select)
    strong_obs = make_obs(make_state(opp_active_hp=80), None)

    engine = install_engine(FakeEngine({
        "root": FakeNode(root_obs, {(0,): "strong", (1,): "strong"}),
        "strong": FakeNode(strong_obs),
    }))

    # 既に過ぎている締め切りを渡す -> 例外を投げず0で返る。
    past_deadline = time.perf_counter() - 1
    assert consequence.best_effective_attack_damage(root_obs, hidden_factory, past_deadline) == 0


def test_no_select_returns_zero():
    obs = make_obs(make_state(), None)
    assert consequence.best_effective_attack_damage(obs, hidden_factory, time.perf_counter() + 1) == 0


# ---------------------------------------------------------------------------
# opp_hp_loss
# ---------------------------------------------------------------------------


def test_opp_hp_loss_basic(install_engine):
    root_select = make_select([OptionType.ATTACK])
    root_obs = make_obs(make_state(opp_active_hp=200), root_select)
    after_obs = make_obs(make_state(opp_active_hp=140), None)

    engine = install_engine(FakeEngine({
        "root": FakeNode(root_obs, {(0,): "after"}),
        "after": FakeNode(after_obs),
    }))

    assert consequence.opp_hp_loss(root_obs, 0, hidden_factory, time.perf_counter() + 1) == 60


def test_opp_hp_loss_none_when_coin(install_engine):
    root_select = make_select([OptionType.ATTACK])
    root_obs = make_obs(make_state(opp_active_hp=200), root_select)
    coin_obs = make_obs(make_state(opp_active_hp=140), None, logs=[Log(type=LogType.COIN)])

    engine = install_engine(FakeEngine({
        "root": FakeNode(root_obs, {(0,): "coin"}),
        "coin": FakeNode(coin_obs),
    }))

    assert consequence.opp_hp_loss(root_obs, 0, hidden_factory, time.perf_counter() + 1) is None


# ---------------------------------------------------------------------------
# option_consequence(Stage3b): 受け入れテスト §6.1
# ---------------------------------------------------------------------------


def test_hammer_removes_special_energy_and_unlocks_attack(install_engine):
    """Tier3方針書 §3.2.1/§6.1 の代表例そのもの。

    ロック闘エネルギー(Rock Fighting Energy)が付いた相手Activeに対し、
    (0) ATTACKは実効0(ロック中)。
    (1) 改造ハンマー(PLAY)→対象選択(相手の特殊エネルギー1枚)→MAIN復帰、を経由すると
        相手の特殊エネルギーが外れ、以後のATTACKが120ダメージ通るようになる。

    期待値: delta_best_effective_attack_damage == 120, opp_special_energy_removed is True。
    """
    rock_energy = make_energy_card(serial=101, card_id=ROCK_FIGHTING_ENERGY_ID)

    # --- root: MAIN選択(0=ATTACK, 1=改造ハンマーのPLAY) ---
    root_state = make_state(opp_active_hp=200, opp_energy_cards=[rock_energy])
    root_select = make_select([OptionType.ATTACK, OptionType.PLAY])
    root_obs = make_obs(root_state, root_select)

    # (0) ATTACKを今すぐ撃つと0ダメージ(ロック中)
    locked_attack_obs = make_obs(make_state(opp_active_hp=200, opp_energy_cards=[rock_energy]), None)

    # (1) 改造ハンマーPLAY -> 対象選択(相手の特殊エネルギーを1枚選ぶ、CARD型・1択)
    hammer_target_select = make_select(
        [OptionType.CARD], select_type=SelectType.ATTACHED_CARD, min_count=1, max_count=1,
    )
    hammer_target_state = make_state(opp_active_hp=200, opp_energy_cards=[rock_energy])
    hammer_target_obs = make_obs(hammer_target_state, hammer_target_select)

    # 対象選択後: MAIN復帰、相手の特殊エネルギーが外れている
    post_hammer_state = make_state(opp_active_hp=200, opp_energy_cards=[])
    post_hammer_select = make_select([OptionType.ATTACK, OptionType.END])
    post_hammer_obs = make_obs(post_hammer_state, post_hammer_select)

    # post_hammer_obsからさらにATTACKを仮実行 -> 120ダメージ通る
    unlocked_attack_obs = make_obs(make_state(opp_active_hp=80, opp_energy_cards=[]), None)

    engine = install_engine(FakeEngine(
        nodes={
            "root": FakeNode(root_obs, {(0,): "locked_attack", (1,): "hammer_target"}),
            "locked_attack": FakeNode(locked_attack_obs),
            "hammer_target": FakeNode(hammer_target_obs, {(0,): "post_hammer_main"}),
            "post_hammer_main": FakeNode(post_hammer_obs, {(0,): "unlocked_attack"}),
            "unlocked_attack": FakeNode(unlocked_attack_obs),
        },
        # post_hammer_obs(仮実行後のobs)はentry_pointsに登録しない: option_consequenceが
        # 「実行後」の評価のために新規search_beginを行っていたら(=修正前のバグが再発したら)
        # ここでAssertionErrorになり、テストが即座に失敗する(回帰防止)。
        entry_points={
            id(root_obs): "root",
        },
    ))

    result = consequence.option_consequence(root_obs, 1, hidden_factory, time.perf_counter() + 1)

    assert result is not None
    assert result.option_index == 1
    assert result.delta_best_effective_attack_damage == 120
    assert result.opp_special_energy_removed is True
    assert result.opp_energy_removed is True
    # コインは絡んでいないので候補は破棄されずに1件は必ず見つかっている。
    assert engine.begin_count >= 2  # option_consequence自身のroot探索 + before/after評価


def test_multiple_targets_picks_best_delta(install_engine):
    """改造ハンマーの対象が複数ある場合、deltaが最大になる解決を代表値として選ぶ(§4.1)。"""
    special_energy = make_energy_card(serial=101, card_id=ROCK_FIGHTING_ENERGY_ID)
    basic_energy = make_energy_card(serial=102, card_id=BASIC_WATER_ENERGY_ID)

    root_state = make_state(opp_active_hp=200, opp_energy_cards=[special_energy, basic_energy])
    root_select = make_select([OptionType.PLAY])
    root_obs = make_obs(root_state, root_select)

    # 対象選択: 0=特殊エネルギー(外すと解錠), 1=基本エネルギー(外しても無意味)
    target_select = make_select(
        [OptionType.CARD, OptionType.CARD], select_type=SelectType.ATTACHED_CARD,
        min_count=1, max_count=1,
    )
    target_obs = make_obs(make_state(opp_active_hp=200, opp_energy_cards=[special_energy, basic_energy]), target_select)

    good_main_obs = make_obs(
        make_state(opp_active_hp=200, opp_energy_cards=[basic_energy]),
        make_select([OptionType.ATTACK, OptionType.END]),
    )
    good_attack_obs = make_obs(make_state(opp_active_hp=80, opp_energy_cards=[basic_energy]), None)

    bad_main_obs = make_obs(
        make_state(opp_active_hp=200, opp_energy_cards=[special_energy]),
        make_select([OptionType.ATTACK, OptionType.END]),
    )
    bad_attack_obs = make_obs(make_state(opp_active_hp=200, opp_energy_cards=[special_energy]), None)

    engine = install_engine(FakeEngine(
        nodes={
            "root": FakeNode(root_obs, {(0,): "target"}),
            "target": FakeNode(target_obs, {(0,): "good_main", (1,): "bad_main"}),
            "good_main": FakeNode(good_main_obs, {(0,): "good_attack"}),
            "good_attack": FakeNode(good_attack_obs),
            "bad_main": FakeNode(bad_main_obs, {(0,): "bad_attack"}),
            "bad_attack": FakeNode(bad_attack_obs),
        },
        # good_main_obs/bad_main_obs(仮実行後)はentry_pointsに登録しない(回帰防止、上記参照)。
        entry_points={
            id(root_obs): "root",
        },
    ))

    result = consequence.option_consequence(root_obs, 0, hidden_factory, time.perf_counter() + 1)

    assert result is not None
    assert result.delta_best_effective_attack_damage == 120
    assert result.opp_special_energy_removed is True


def test_unresolved_returns_none_when_no_transition(install_engine):
    root_select = make_select([OptionType.PLAY])
    root_obs = make_obs(make_state(opp_active_hp=200), root_select)
    # "root" に選択(0,)に対応するtransitionが無い -> search_stepがValueErrorを投げてNone。
    engine = install_engine(FakeEngine({"root": FakeNode(root_obs, {})}, entry_points={id(root_obs): "root"}))

    result = consequence.option_consequence(root_obs, 0, hidden_factory, time.perf_counter() + 1)
    assert result is None


def test_coin_branch_excluded_from_candidates(install_engine):
    """対象選択の結果がコイン絡みなら、その解決は候補から除外される(§2.5)。"""
    root_select = make_select([OptionType.PLAY])
    root_obs = make_obs(make_state(opp_active_hp=200), root_select)

    target_select = make_select([OptionType.CARD], select_type=SelectType.ATTACHED_CARD)
    target_obs = make_obs(make_state(opp_active_hp=200), target_select)

    coin_main_obs = make_obs(
        make_state(opp_active_hp=200), make_select([OptionType.END]),
        logs=[Log(type=LogType.COIN)],
    )

    engine = install_engine(FakeEngine(
        nodes={
            "root": FakeNode(root_obs, {(0,): "target"}),
            "target": FakeNode(target_obs, {(0,): "coin_main"}),
            "coin_main": FakeNode(coin_main_obs),
        },
        entry_points={id(root_obs): "root"},
    ))

    result = consequence.option_consequence(root_obs, 0, hidden_factory, time.perf_counter() + 1)
    assert result is None  # 唯一の解決候補がコイン絡みで棄却され、他に候補が無い


def test_deck_touching_effect_is_not_excluded(install_engine):
    """attack_plan v0/v1と異なり、Tier3はdeckに触れる効果(ドロー等)を除外しない(§4)。"""
    root_select = make_select([OptionType.PLAY])
    root_obs = make_obs(make_state(opp_active_hp=200, my_hand_count=2), root_select)

    # ドロー(手札+1)が発生した上でMAIN復帰する解決。
    drew_state = make_state(opp_active_hp=200, my_hand_count=3)
    drew_main_obs = make_obs(drew_state, make_select([OptionType.END]), logs=[Log(type=LogType.DRAW)])

    engine = install_engine(FakeEngine(
        nodes={
            "root": FakeNode(root_obs, {(0,): "drew"}),
            "drew": FakeNode(drew_main_obs),
        },
        entry_points={id(root_obs): "root"},  # drew_main_obsは登録しない(回帰防止)
    ))

    result = consequence.option_consequence(root_obs, 0, hidden_factory, time.perf_counter() + 1)
    assert result is not None
    assert result.cards_drawn == 1


def test_combination_budget_limits_exploration(install_engine):
    """組み合わせ上限(max_combinations)を超える対象選択は打ち切られる(§4.1)。"""
    root_select = make_select([OptionType.PLAY])
    root_obs = make_obs(make_state(opp_active_hp=200), root_select)

    # 3択(0,1,2)から1つ選ぶ対象選択。max_combinations=1なら1つしか試されない。
    target_select = make_select(
        [OptionType.CARD, OptionType.CARD, OptionType.CARD],
        select_type=SelectType.ATTACHED_CARD, min_count=1, max_count=1,
    )
    target_obs = make_obs(make_state(opp_active_hp=200), target_select)

    main0 = make_obs(make_state(opp_active_hp=200), make_select([OptionType.END]))
    main1 = make_obs(make_state(opp_active_hp=200), make_select([OptionType.END]))
    main2 = make_obs(make_state(opp_active_hp=200), make_select([OptionType.END]))

    engine = install_engine(FakeEngine(
        nodes={
            "root": FakeNode(root_obs, {(0,): "target"}),
            "target": FakeNode(target_obs, {(0,): "main0", (1,): "main1", (2,): "main2"}),
            "main0": FakeNode(main0),
            "main1": FakeNode(main1),
            "main2": FakeNode(main2),
        },
        entry_points={id(root_obs): "root"},  # main0/1/2は登録しない(回帰防止)
    ))

    result = consequence.option_consequence(
        root_obs, 0, hidden_factory, time.perf_counter() + 1, max_combinations=1,
    )
    # 1候補しか試さないが、見つかった1件を結果として返す(例外にはならない)。
    assert result is not None


# ---------------------------------------------------------------------------
# delta_attack_ready / delta_energy_shortfall(静的計算、方針書 §5)
# ---------------------------------------------------------------------------


def test_attach_energy_makes_attack_ready(install_engine):
    """カダブラ(実card_id=742、技1071 Super Psy Bolt、コスト{P}=Psychic1個)に
    Psychicエネルギーを1枚ATTACHすると、has_ready_attackが0->1になり
    energy_shortfallが1->0(delta_attack_ready=True, delta_energy_shortfall=+1)。"""
    from cg.api import EnergyType

    KADABRA_ID = 742

    root_select = make_select([OptionType.ATTACH])
    root_state = make_state(opp_active_hp=200)
    # 自分のアクティブをカダブラ(エネルギー無し = 技が撃てない)に差し替える。
    root_state.players[0].active = [
        make_pokemon(serial=1, hp=80, max_hp=80, card_id=KADABRA_ID, energy_cards=[])
    ]
    root_obs = make_obs(root_state, root_select)

    after_state = make_state(opp_active_hp=200)
    after_pokemon = make_pokemon(serial=1, hp=80, max_hp=80, card_id=KADABRA_ID, energy_cards=[])
    after_pokemon.energies = [EnergyType.PSYCHIC]  # ATTACH後: Psychicエネルギー1個
    after_state.players[0].active = [after_pokemon]
    after_obs = make_obs(after_state, make_select([OptionType.ATTACK, OptionType.END]))

    engine = install_engine(FakeEngine(
        nodes={
            "root": FakeNode(root_obs, {(0,): "after"}),
            "after": FakeNode(after_obs),
        },
        entry_points={id(root_obs): "root"},
    ))

    result = consequence.option_consequence(root_obs, 0, hidden_factory, time.perf_counter() + 1)
    assert result is not None
    assert result.delta_attack_ready is True
    assert result.delta_energy_shortfall == 1.0


def test_no_energy_change_keeps_shortfall(install_engine):
    """技を持たない/対象外のポケモンではdelta_attack_ready=False, delta_energy_shortfall=0。"""
    root_select = make_select([OptionType.PLAY])
    root_obs = make_obs(make_state(opp_active_hp=200), root_select)
    after_obs = make_obs(make_state(opp_active_hp=200), make_select([OptionType.END]))

    engine = install_engine(FakeEngine(
        nodes={
            "root": FakeNode(root_obs, {(0,): "after"}),
            "after": FakeNode(after_obs),
        },
        entry_points={id(root_obs): "root"},
    ))

    result = consequence.option_consequence(root_obs, 0, hidden_factory, time.perf_counter() + 1)
    assert result is not None
    assert result.delta_attack_ready is False
    assert result.delta_energy_shortfall == 0.0
