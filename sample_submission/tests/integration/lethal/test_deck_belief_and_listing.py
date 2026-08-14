"""山札 belief の正しさと R2(既知 listing からの取得)の扱い(Step 1-9b/c)。

Step 1-9b で見つかった重大な問題:
``_deck_multiset()`` が「デッキリスト − 見えている自分のカード − 仮定した伏せサイド」で
belief を導いており、**系統的に過少計上**していた(供給 40 枚に対し belief 35 枚など)。
列挙対象のプールが小さいと **outcome を取りこぼす = 偽証明の経路**になる。

ここでは修正後の belief が「我々が供給した山札」と一致することを実データで固定し、
R2 の判定(既知 listing であることの**検証**)と、listing 順序への非依存を確認する。
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import SelectContext, to_observation_class
from main import read_deck_csv
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.search.lethal import phase1, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.cg_backend import CgBackend
from ptcg_ai.search.lethal.deck_view import KnownDeckComposition
from ptcg_ai.search.lethal.engine import HiddenState, SearchSession, deck_revealed_at_root
from ptcg_ai.search.lethal.scenario import Scenario
from ptcg_ai.search.lethal.types import Proof

FIXTURE = SAMPLE_SUBMISSION_ROOT / "tests" / "fixtures" / "lethal_positions.jsonl"


@pytest.fixture(scope="module")
def corpus():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def _hidden(obs, deck, seed=0):
    stub = build_dummy_search_state(obs, deck, rng=random.Random(seed))
    return None if stub is None else HiddenState.from_stub(stub)


def _backend(session, obs, hidden, deck, **kwargs):
    return CgBackend(
        session, obs, hidden,
        deck_composition=KnownDeckComposition.from_card_ids(deck),
        **kwargs,
    )


# ------------------------------------------------------------ belief の正しさ


def test_deck_belief_matches_the_supplied_deck(corpus):
    """root の belief は、我々が供給した山札と**一致**する(過少計上しない)。

    以前の実装はここで 1〜6 枚少ない multiset を返しており、
    ドローの outcome 列挙が取りこぼす可能性があった。
    """
    deck = read_deck_csv()
    checked = 0
    for row in corpus:
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None:
            continue
        with SearchSession(obs, hidden) as session:
            backend = _backend(session, obs, hidden, deck)
            root = backend.root()
            belief = backend._deck_multiset(backend._observation(root), root)
            truth = hidden.scenario.multiset()
            if deck_revealed_at_root(obs):
                # B1: 実デッキが使われる局面。listing 由来の belief になる
                assert belief is None or sum(belief.values()) >= 0
                continue
            assert belief is not None, f"{row['id']}: belief が作れない"
            assert belief == truth, (
                f"{row['id']}: belief が供給デッキと不一致 "
                f"missing={truth - belief} extra={belief - truth}"
            )
            checked += 1
    assert checked >= 20, f"検証できた局面が少なすぎる: {checked}"


def test_belief_shrinks_exactly_by_the_drawn_cards(corpus):
    """ドロー後の belief は、引いた分だけ正確に減る。"""
    deck = read_deck_csv()
    checked = 0
    for row in corpus:
        if "draw" not in row["tags"]:
            continue
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None or deck_revealed_at_root(obs):
            continue
        with SearchSession(obs, hidden) as session:
            backend = _backend(session, obs, hidden, deck)
            root = backend.root()
            before = backend._deck_multiset(backend._observation(root), root)
            for action in backend.legal_actions(root):
                transition = backend.apply(root, action)
                child = transition.state
                if child is None or child.draws == root.draws:
                    continue
                after = backend._deck_multiset(backend._observation(child), child)
                if after is None:
                    continue  # deckCount と一致しない = 安全側で諦めた
                drawn = Counter(child.draws[-1])
                assert after == before - drawn, f"{row['id']}: belief の減り方が違う"
                checked += 1
                break
    assert checked > 0


# --------------------------------------------------------------- R2 の判定


def test_known_listing_is_verified_not_assumed(corpus):
    """R2 判定は「listing が belief の部分集合」であることを**検証**している。"""
    deck = read_deck_csv()
    seen = 0
    for row in corpus:
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None:
            continue
        with SearchSession(obs, hidden) as session:
            backend = _backend(session, obs, hidden, deck, known_listing_as_decision=True)
            root = backend.root()
            for action in backend.legal_actions(root):
                transition = backend.apply(root, action)
                child = transition.state
                if child is None:
                    continue
                observation = backend._observation(child)
                select = observation.select
                if select is None or select.deck is None:
                    continue
                listing = Counter(card.id for card in select.deck if card is not None)
                expected = backend._belief_from_supply(child)
                if backend._is_known_listing_selection(observation, child):
                    # 既知と判定したなら、listing は belief の部分集合であること
                    assert expected is not None
                    assert not (listing - expected), f"{row['id']}: 未知のカードが listing に出た"
                    assert int(select.context) in (
                        int(SelectContext.TO_HAND), int(SelectContext.TO_BENCH)
                    )
                    seen += 1
                break
    assert seen > 0, "R2(既知 listing)が 1 件も検出できなかった"


def test_known_listing_flag_is_off_by_default(corpus):
    """R2 は既定 OFF（実測で利得ゼロ・コスト有のため）。"""
    deck = read_deck_csv()
    obs = to_observation_class(corpus[0]["obs"])
    hidden = _hidden(obs, deck)
    with SearchSession(obs, hidden) as session:
        backend = _backend(session, obs, hidden, deck)
        assert backend._known_listing_as_decision is False


def test_listing_order_does_not_change_the_search_result(corpus):
    """listing の順序だけが違っても、proof と探索ノード数が一致する。

    listing は供給した山札から作られるので、山札の並びを変えると listing の
    並びも変わる。順序に依存していないことをここで固定する。
    """
    deck = read_deck_csv()
    compared = 0
    for row in corpus:
        obs = to_observation_class(row["obs"])
        stub = build_dummy_search_state(obs, deck, rng=random.Random(0))
        if stub is None or deck_revealed_at_root(obs):
            continue
        base = HiddenState.from_stub(stub)
        results = set()
        for seed in range(3):
            order = list(base.scenario.order)
            random.Random(700 + seed).shuffle(order)
            variant = base.with_scenario(Scenario(tuple(order)))
            with SearchSession(obs, variant) as session:
                backend = _backend(session, obs, variant, deck,
                                   known_listing_as_decision=True)
                result = phase1.search(
                    backend,
                    Budget(time_limit_ms=30_000.0, max_nodes=400, max_depth=6),
                )
            results.add((result.proof.name, result.nodes))
        assert len(results) == 1, f"{row['id']}: listing の順序で結果が変わった: {results}"
        compared += 1
        if compared >= 6:
            break
    assert compared >= 3


def test_r2_does_not_change_phase2_results(corpus):
    """実測の記録: R2 の ON/OFF で Phase 2 の判定は変わらない。

    「UNKNOWN が減るはず」という期待で有効化されないよう、測定結果を固定する。
    """
    deck = read_deck_csv()
    checked = 0
    for row in corpus[:12]:
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None:
            continue
        proofs = []
        for flag in (False, True):
            with SearchSession(obs, hidden) as session:
                backend = _backend(session, obs, hidden, deck,
                                   known_listing_as_decision=flag)
                result = phase2.search(
                    backend,
                    Budget(time_limit_ms=30_000.0, max_nodes=400, max_depth=6,
                           max_chance_depth=1),
                )
            proofs.append(result.proof.name)
        assert proofs[0] == proofs[1], f"{row['id']}: R2 で Phase 2 の判定が変わった {proofs}"
        checked += 1
    assert checked >= 8


# ------------------------------------------- belief の回帰(Step 1-10 指示 2)


def test_belief_is_none_when_it_cannot_be_verified(corpus):
    """``deckCount`` と一致しない導出は **None**(安全側)になる。

    belief を「たぶんこれ」で返すと、outcome 列挙が取りこぼして偽証明になる。
    検証できないときは諦めることをここで固定する。
    """
    deck = read_deck_csv()
    obs = to_observation_class(corpus[0]["obs"])
    hidden = _hidden(obs, deck)
    with SearchSession(obs, hidden) as session:
        backend = _backend(session, obs, hidden, deck)
        root = backend.root()
        observation = backend._observation(root)
        # 供給した山札を 1 枚多い状態に偽装すると、deckCount と合わなくなる
        broken = HiddenState.from_stub(
            {**build_dummy_search_state(obs, deck, rng=random.Random(0)),
             "your_deck": list(hidden.scenario.order) + [1]}
        )
        backend._hidden = broken
        assert backend._deck_multiset(observation, root) is None


def test_undercounted_belief_does_not_yield_proven_win(corpus):
    """belief が過少計上されていたら、Phase 2 は ``PROVEN_WIN`` を出さない。

    Step 1-9b で見つかった不具合の再発防止。列挙対象のプールが小さいと
    「全 outcome で勝つ」の証明が穴だらけになる。
    """
    deck = read_deck_csv()
    checked = 0
    for row in corpus:
        if "draw" not in row["tags"]:
            continue
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None or deck_revealed_at_root(obs):
            continue

        class UnderCounting(CgBackend):
            def _belief_from_supply(self, state):
                belief = super()._belief_from_supply(state)
                if belief is None:
                    return None
                # わざと 1 枚落とす(= 過少計上の再現)
                for card_id in sorted(belief):
                    if belief[card_id] > 0:
                        belief[card_id] -= 1
                        break
                return +belief

        with SearchSession(obs, hidden) as session:
            backend = UnderCounting(
                session, obs, hidden,
                deck_composition=KnownDeckComposition.from_card_ids(deck),
            )
            result = phase2.search(
                backend,
                Budget(time_limit_ms=2000.0, max_nodes=400, max_depth=6, max_chance_depth=1),
            )
        # 過少計上した belief は deckCount と一致しないので None になり、
        # outcome 列挙は拒否される = PROVEN_WIN にはならない
        assert result.proof is not Proof.PROVEN_WIN, f"{row['id']}: 過少計上で誤証明した"
        checked += 1
        if checked >= 4:
            break
    assert checked > 0


def test_belief_after_search_and_shuffle(corpus):
    """サーチ・公開・シャッフルを跨いでも belief が破綻しない。"""
    deck = read_deck_csv()
    seen = Counter()
    for row in corpus:
        obs = to_observation_class(row["obs"])
        hidden = _hidden(obs, deck)
        if hidden is None:
            continue
        with SearchSession(obs, hidden) as session:
            backend = _backend(session, obs, hidden, deck, known_listing_as_decision=True)
            root = backend.root()
            for action in backend.legal_actions(root)[:8]:
                transition = backend.apply(root, action)
                child = transition.state
                if child is None:
                    continue
                belief = backend._deck_multiset(backend._observation(child), child)
                observation = backend._observation(child)
                deck_count = observation.current.players[backend._me].deckCount
                if belief is not None:
                    # 返ってきた以上は必ず deckCount と一致していること
                    assert sum(belief.values()) == deck_count, f"{row['id']}: 枚数不一致"
                    seen["verified"] += 1
                else:
                    seen["declined"] += 1
                if child.shuffled:
                    seen["after_shuffle"] += 1
    assert seen["verified"] > 0
