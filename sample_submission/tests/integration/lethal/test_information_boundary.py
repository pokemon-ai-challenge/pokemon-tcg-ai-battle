"""情報境界・determinization・B1 の結合テスト(実エンジンを使う)。

設計 §9.1 の T1〜T5 と、規則 R3 / R4 を実際の ``cg`` エンジンに対して固定する。
**Phase 1/2 の探索本体はまだ無い**ので、T1 は「探索の代理(runner)」に対して回す:

- ``_info_set_runner``        : 新情報境界で止まる正しい実装の代理 → 一致すべき
- ``_scenario_peeking_runner``: 供給順序を直接覗く実装      → 検出されるべき
- ``_engine_lookahead_runner``: 引いた結果まで見て選ぶ実装   → 検出されるべき

後ろ2つが「検出される」ことまで確認するのがテストの検出力の確認(T5)である。
Phase 1/2 が実装されたら、同じハーネスを本体へ適用する。

実行:
    python -m pytest tests/integration/lethal/test_information_boundary.py -q
"""

from __future__ import annotations

import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import SelectType, to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from main import read_deck_csv
from ptcg_ai.action_selection import router
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.opponent_modeling import tracker as opponent_tracker
from ptcg_ai.search.lethal.chance import check_reveal_guard, classify_pending_chance
from ptcg_ai.search.lethal.engine import (
    EngineError,
    HiddenState,
    SearchSession,
    deck_revealed_at_root,
    is_replay_deterministic,
    iter_legal_single_selections,
)
from ptcg_ai.search.lethal.infokey import info_key
from ptcg_ai.search.lethal.scenario import Scenario
from ptcg_ai.search.lethal.types import ChanceClass, StopReason

MAIN_POSITIONS = 16
DECK_POSITIONS = 6
GAMES = 2
POOL_SIZE = 40          # 分類にかける候補数
CHANCE_QUOTA = 6        # うち「乱数事象を含む局面」を最低これだけ入れる
SEEDS = (0, 1, 2, 3)
WALK_DEPTH = 5


# --------------------------------------------------------------- 局面の採取


def _play_game(seed: int) -> tuple[list[dict], list[dict]]:
    """ランダム対戦を1試合最後まで回し、候補局面を全部集める。

    途中で打ち切ると序盤に偏り、ドローを伴う選択肢が出ない局面ばかりになる
    (Step 0 の実測でも、ドロー選択肢を持つのは全体の 3 割程度)。
    """
    rng = random.Random(seed)
    deck = read_deck_csv()
    main_positions: list[dict] = []
    deck_positions: list[dict] = []
    obs_dict, start = battle_start(list(deck), list(deck))
    assert start.errorType == 0
    try:
        for _ in range(4000):
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                break
            if obs.select is None:
                obs_dict = battle_select(list(deck))
                continue
            select = obs.select
            state = obs.current
            if (
                state is not None
                and state.yourIndex == 0
                and select.minCount <= 1 <= select.maxCount
            ):
                if select.deck is not None:
                    deck_positions.append(obs_dict)
                elif select.type == SelectType.MAIN and state.turn >= 3:
                    main_positions.append(obs_dict)
            n = len(select.option)
            count = rng.randint(select.minCount, min(select.maxCount, n))
            obs_dict = battle_select(rng.sample(range(n), count) if count else [])
    finally:
        battle_finish()
    return main_positions, deck_positions


def _subsample(items: list[dict], limit: int) -> list[dict]:
    """試合全体に散らばるように間引く(序盤への偏りを避ける)。"""
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def _collect_positions() -> tuple[list[dict], list[dict]]:
    """テストに使う局面を集める。

    エンジンの初期シャッフルはグローバル RNG で決まり、こちらから seed できない
    (試合の内容は実行のたびに変わる)。そのため「たまたま乱数事象を含む局面が
    採れなかった」でテストが落ちないよう、候補プールを分類してから選ぶ。
    """
    main_all: list[dict] = []
    deck_all: list[dict] = []
    for game in range(GAMES):
        main_positions, deck_positions = _play_game(20260812 + game)
        main_all += main_positions
        deck_all += deck_positions

    deck = read_deck_csv()
    pool = _subsample(main_all, POOL_SIZE)
    with_chance: list[dict] = []
    without_chance: list[dict] = []
    for obs_dict in pool:
        obs, variants = _hidden_variants(obs_dict, deck, seeds=(0,))
        if not variants:
            continue
        target = with_chance if _has_chance_option(obs, variants[0]) else without_chance
        target.append(obs_dict)
    selected = with_chance[:CHANCE_QUOTA]
    selected += without_chance[: max(0, MAIN_POSITIONS - len(selected))]
    selected += with_chance[CHANCE_QUOTA : MAIN_POSITIONS - len(selected) + CHANCE_QUOTA]
    return selected[:MAIN_POSITIONS], _subsample(deck_all, DECK_POSITIONS)


@pytest.fixture(scope="module")
def positions():
    main_positions, deck_positions = _collect_positions()
    assert main_positions, "自分の MAIN 局面が採取できなかった"
    return main_positions, deck_positions


# ------------------------------------------------- 同一 multiset の別順列を作る


def _hidden_variants(obs_dict: dict, deck: list[int], seeds=SEEDS):
    """同じ信念(multiset)で、山札の並びだけが違う HiddenState を作る。

    ``build_dummy_search_state`` は seed ごとに「どの未確認カードがサイドか」まで
    変えてしまう(= 信念そのものが変わる)。T1 が検証したいのは
    **信念が同じで並びだけ違う**場合の不変性なので、1つの stub を基準に
    順列だけ差し替える。
    """
    obs = to_observation_class(obs_dict)
    stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
    if stub is None:
        return obs, []
    base = HiddenState.from_stub(stub)
    variants = []
    for seed in seeds:
        order = list(base.scenario.order)
        random.Random(1000 + seed).shuffle(order)
        variant = base.with_scenario(Scenario(tuple(order)))
        assert variant.deck_multiset() == base.deck_multiset()
        variants.append(variant)
    return obs, variants


# ------------------------------------------------------------------ runners


def _is_information_boundary(child, events) -> bool:
    """新情報が公開された地点か(設計 §2.3)。"""
    if events.sequence:
        return True
    select = child.observation.select
    state = child.observation.current
    if select is not None and select.deck is not None:
        return True
    if state is not None and state.looking is not None:
        return True
    return False


def _choose(obs, hidden, score):
    """各選択肢を1手だけ試し、``score`` が最大のものを選ぶ。"""
    best = None
    with SearchSession(obs, hidden) as session:
        for selection in iter_legal_single_selections(obs):
            try:
                child, events = session.step(session.root, selection)
            except EngineError:
                continue
            value = score(session, child, events, hidden)
            if best is None or value > best[0]:
                best = (value, tuple(selection))
    return None if best is None else best[1]


def _info_set_runner(obs, hidden):
    """正しい実装の代理: 新情報境界を越えた先の中身を見ない。"""

    def score(session, child, events, _hidden):
        if _is_information_boundary(child, events):
            # 公開された中身(引いたカードの identity)は使わない。件数だけ。
            return (0, len(events.drawn), len(events.coin_heads), "")
        return (1, 0, 0, info_key(child.observation, session.me).digest())

    return _choose(obs, hidden, score)


def _scenario_peeking_runner(obs, hidden):
    """壊した実装その1: 供給した山札の一番上(=隠れた順序)を直接覗く。"""
    top = hidden.scenario.order[-1] if hidden.scenario.size() else 0

    def score(_session, _child, _events, _hidden):
        return (top % 7, 0, 0, "")

    # top に依存させるため、選択肢の順位付けを top でずらす
    best = None
    with SearchSession(obs, hidden) as session:
        selections = list(iter_legal_single_selections(obs))
        for offset, selection in enumerate(selections):
            try:
                child, events = session.step(session.root, selection)
            except EngineError:
                continue
            value = ((top + offset) % max(len(selections), 1), 0, 0, "")
            _ = score(session, child, events, hidden)
            if best is None or value > best[0]:
                best = (value, tuple(selection))
    return None if best is None else best[1]


def _engine_lookahead_runner(obs, hidden):
    """壊した実装その2: 1手先の完全な状態(引いたカードを含む)で選ぶ。

    「1つのシナリオで探索して、その結果の状態が最良の手を選ぶ」という
    determinization の典型形。引いたカードの identity が評価に入るので、
    山札の並びが変わると選ぶ手も変わる。
    """

    def score(session, child, _events, _hidden):
        fingerprint = SearchSession.fingerprint(child)
        digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
        return (0, 0, 0, digest)

    return _choose(obs, hidden, score)


def _actions_over_variants(runner, obs, variants):
    return [runner(obs, hidden) for hidden in variants]


# ------------------------------------------------------------------- T1 / T5


def _has_chance_option(obs, hidden) -> bool:
    """深さ1に「引く/コイン/シャッフル」を起こす選択肢があるか。"""
    with SearchSession(obs, hidden) as session:
        for selection in iter_legal_single_selections(obs):
            try:
                _child, events = session.step(session.root, selection)
            except EngineError:
                continue
            if events.sequence:
                return True
    return False


def test_info_set_runner_is_invariant_to_hidden_deck_order(positions):
    """T1: 隠れた山札順だけを変えても root action が一致する。

    「そもそも乱数事象が起きない局面ばかり」だと不変性は自明に成り立ってしまうので、
    乱数事象を含む局面が一定数含まれていることも同時に確認する(空振り防止)。
    """
    main_positions, _ = positions
    deck = read_deck_csv()
    checked = 0
    with_chance = 0
    for obs_dict in main_positions:
        obs, variants = _hidden_variants(obs_dict, deck)
        if len(variants) < 2:
            continue
        if _has_chance_option(obs, variants[0]):
            with_chance += 1
        actions = _actions_over_variants(_info_set_runner, obs, variants)
        assert len(set(actions)) == 1, (
            f"root action が山札順に依存した: {actions}"
        )
        checked += 1
    assert checked >= 5, f"検証できた局面が少なすぎる: {checked}"
    assert with_chance >= 3, (
        f"乱数事象を含む局面が少なすぎる({with_chance})。テストが空振りしている可能性がある"
    )


def test_scenario_peeking_runner_is_detected(positions):
    """T5: 順序を覗く実装を T1 のハーネスが検出できる。"""
    main_positions, _ = positions
    deck = read_deck_csv()
    diverged = 0
    for obs_dict in main_positions:
        obs, variants = _hidden_variants(obs_dict, deck)
        if len(variants) < 2:
            continue
        actions = _actions_over_variants(_scenario_peeking_runner, obs, variants)
        if len(set(actions)) > 1:
            diverged += 1
    assert diverged > 0, "順序を覗く実装を検出できなかった(テストの検出力が無い)"


def test_engine_lookahead_runner_is_detected(positions):
    """T5: 「1シナリオで先読みして選ぶ」determinization を検出できる。"""
    main_positions, _ = positions
    deck = read_deck_csv()
    diverged = 0
    for obs_dict in main_positions:
        obs, variants = _hidden_variants(obs_dict, deck)
        if len(variants) < 2:
            continue
        actions = _actions_over_variants(_engine_lookahead_runner, obs, variants)
        if len(set(actions)) > 1:
            diverged += 1
    assert diverged > 0, "1シナリオ先読み実装を検出できなかった"


def test_info_key_is_stable_across_deck_permutations(positions):
    """T2 の実データ版: 同じ局面・同じ multiset なら並びに関係なくキーが同じ。"""
    main_positions, _ = positions
    deck = read_deck_csv()
    for obs_dict in main_positions:
        obs, variants = _hidden_variants(obs_dict, deck)
        if len(variants) < 2:
            continue
        keys = {info_key(obs, obs.current.yourIndex, deck_multiset=h.deck_multiset())
                for h in variants}
        assert len(keys) == 1


# ---------------------------------------------------------------- B1 ガード


def _random_walk(session, rng, depth=WALK_DEPTH):
    """root から浅くランダムに歩き、各ステップの事象を集める。"""
    node = session.root
    events_seq = []
    for _ in range(depth):
        obs = node.observation
        state = obs.current
        if state is None or state.result != -1 or obs.select is None or not obs.select.option:
            break
        if state.yourIndex != session.me:
            break
        select = obs.select
        n = len(select.option)
        count = rng.randint(select.minCount, min(select.maxCount, n))
        selection = rng.sample(range(n), count) if count else []
        try:
            node, events = session.step(node, selection)
        except EngineError:
            break
        events_seq.append(events)
    return events_seq


def test_no_draw_before_shuffle_after_deck_reveal(positions):
    """B1 回帰: デッキ公開経路で SHUFFLE を挟まないドローが発生しないこと。

    Step 0 の観測(430 サンプルすべて SHUFFLE 後)を、実装後も継続して守るための
    回帰テスト。違反した場合はガードが ``UNSUPPORTED_EFFECT`` を返すので、
    ここでは「違反が起きていない」ことと「起きたら検出できる」ことの両方を見る。
    """
    main_positions, deck_positions = positions
    deck = read_deck_csv()
    walked = 0
    for obs_dict in deck_positions + main_positions:
        obs = to_observation_class(obs_dict)
        stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
        if stub is None:
            continue
        hidden = HiddenState.from_stub(stub)
        revealed = deck_revealed_at_root(obs)
        rng = random.Random(7)
        for _ in range(3):
            with SearchSession(obs, hidden) as session:
                events_seq = _random_walk(session, rng)
            violation = check_reveal_guard(events_seq, deck_revealed_at_root=revealed)
            assert violation is None, (
                f"デッキ公開後に SHUFFLE を挟まないドローを検出: revealed={revealed}"
            )
            walked += 1
    assert walked > 0


def test_deck_visible_root_is_never_classified_controllable(positions):
    """B1: 実デッキが使われる局面を「列挙可能」と判定しないこと。"""
    _, deck_positions = positions
    if not deck_positions:
        pytest.skip("デッキ公開局面が採取できなかった")
    deck = read_deck_csv()
    for obs_dict in deck_positions:
        obs = to_observation_class(obs_dict)
        assert deck_revealed_at_root(obs) is True
        stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
        if stub is None:
            continue
        result = classify_pending_chance(
            select=obs.select,
            manual_coin=False,
            deck_order_controlled=not deck_revealed_at_root(obs),
            shuffle_on_path=False,
            replay_deterministic=True,
        )
        assert result.chance_class is ChanceClass.ENGINE_RANDOM
        assert result.stop_reason is StopReason.DECK_REVEALED_AT_ROOT
        assert not result.is_enumerable


# ------------------------------------------------------- R3 再生一致検査


def _draw_kind(events, shuffle_before_step: bool) -> str | None:
    """このステップのドローが「供給順のまま」か「シャッフル後」かを返す。

    1ステップの中でも順序が意味を持つ(引いてからシャッフルする効果がある)ので、
    ログの並びを見る。
    """
    if not events.drew:
        return None
    shuffled = shuffle_before_step
    for kind in events.sequence:
        if kind == "shuffle":
            shuffled = True
        elif kind == "draw":
            return "after_shuffle" if shuffled else "supply_order"
    return None


def _find_paths(obs, hidden, *, depth_limit=2, max_breadth=6):
    """(供給順のままドローする経路, シャッフル後にドローする経路) を探す。

    深さ1だけを見ると、ドロー選択肢を持たない局面(Step 0 の実測で約7割)で
    何も見つからずテストが空振りするので、浅い幅優先で探す。
    """
    plain = shuffled = None
    with SearchSession(obs, hidden) as session:
        queue = [([], session.root, False)]
        while queue and (plain is None or shuffled is None):
            path, node, saw_shuffle = queue.pop(0)
            observation = node.observation
            state = observation.current
            if state is None or state.result != -1 or observation.select is None:
                continue
            if state.yourIndex != session.me:
                continue
            select = observation.select
            if not (select.minCount <= 1 <= select.maxCount):
                continue
            for index in range(min(len(select.option), max_breadth)):
                try:
                    child, events = session.step(node, [index])
                except EngineError:
                    continue
                new_path = path + [[index]]
                kind = _draw_kind(events, saw_shuffle)
                if kind == "supply_order" and plain is None:
                    plain = new_path
                elif kind == "after_shuffle" and shuffled is None:
                    shuffled = new_path
                if len(new_path) < depth_limit:
                    queue.append((new_path, child, saw_shuffle or events.shuffled))
    return plain, shuffled


def test_replay_check_separates_controlled_from_engine_random(positions):
    """R3: 再生一致検査の結果でのみ C を主張する。"""
    main_positions, _ = positions
    deck = read_deck_csv()
    checked_plain = checked_shuffled = 0
    for obs_dict in main_positions:
        obs = to_observation_class(obs_dict)
        stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
        if stub is None:
            continue
        hidden = HiddenState.from_stub(stub)
        plain, shuffled = _find_paths(obs, hidden)

        if plain is not None and checked_plain < 3:
            deterministic = is_replay_deterministic(obs, hidden, plain)
            assert deterministic is True, "供給順ドローが再生で一致しなかった"
            result = classify_pending_chance(
                select=obs.select,
                manual_coin=False,
                deck_order_controlled=True,
                shuffle_on_path=False,
                replay_deterministic=deterministic,
            )
            assert result.chance_class is ChanceClass.CONTROLLED_SUPPLY_ORDER
            checked_plain += 1

        if shuffled is not None and checked_shuffled < 3:
            deterministic = is_replay_deterministic(obs, hidden, shuffled)
            result = classify_pending_chance(
                select=obs.select,
                manual_coin=False,
                deck_order_controlled=True,
                shuffle_on_path=True,
                replay_deterministic=deterministic,
            )
            # シャッフル後は、再生が一致しようがしまいが列挙可能にはしない。
            assert result.chance_class is ChanceClass.ENGINE_RANDOM
            assert not result.is_enumerable
            checked_shuffled += 1

        if checked_plain >= 3 and checked_shuffled >= 3:
            break

    assert checked_plain > 0, "供給順ドローの経路が見つからなかった"


def test_sessions_release_resources_and_reject_nesting(positions):
    """資源管理: セッションは入れ子にできない(共有 agent_ptr を壊さない)。"""
    from ptcg_ai.search.lethal.engine import SessionNestingError

    main_positions, _ = positions
    deck = read_deck_csv()
    obs = to_observation_class(main_positions[0])
    stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
    hidden = HiddenState.from_stub(stub)
    with SearchSession(obs, hidden):
        with pytest.raises(SessionNestingError):
            with SearchSession(obs, hidden):
                pass
    # 例外後もセッションを開けること(状態が残っていない)
    with SearchSession(obs, hidden) as session:
        assert session.root.observation.current is not None


# ------------------------------------------- 通常方策・tracker への副作用なし


def _tracker_fingerprint() -> str:
    parts = []
    for name in dir(opponent_tracker):
        if name.startswith("_") and not name.startswith("__"):
            parts.append(f"{name}={getattr(opponent_tracker, name)!r}")
    return "|".join(parts)


def test_router_on_search_observation_does_not_mutate_tracker(positions):
    """探索用状態を通常方策へ渡しても、相手予測の状態を汚さないこと。"""
    main_positions, _ = positions
    deck = read_deck_csv()
    obs = to_observation_class(main_positions[0])
    stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
    hidden = HiddenState.from_stub(stub)

    before = _tracker_fingerprint()
    with SearchSession(obs, hidden) as session:
        node = session.root
        for _ in range(5):
            child_obs = node.observation
            state = child_obs.current
            if state is None or state.result != -1 or child_obs.select is None:
                break
            if state.yourIndex != session.me:
                break
            action = router.route(child_obs)
            assert action is not None
            select = child_obs.select
            assert select.minCount <= len(action) <= select.maxCount
            assert len(action) == len(set(action))
            assert all(0 <= i < len(select.option) for i in action)
            try:
                node, _events = session.step(node, action)
            except EngineError:
                pytest.fail("通常方策が探索中に違法手を返した")
    assert _tracker_fingerprint() == before


def test_deck_multiset_is_the_only_belief_exposed(positions):
    """HiddenState が決定側へ渡すのは multiset だけであること。"""
    main_positions, _ = positions
    deck = read_deck_csv()
    obs = to_observation_class(main_positions[0])
    stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
    hidden = HiddenState.from_stub(stub)
    assert isinstance(hidden.deck_multiset(), Counter)
    text = repr(hidden).replace(hidden.scenario.digest(), "")
    # 1〜2桁のIDは repr 中の枚数表示(size=45 など)と偶然一致するため、
    # 判定は3桁以上のカードIDに限る。
    for card_id in {c for c in stub["your_deck"] if c >= 100}:
        assert str(card_id) not in text
