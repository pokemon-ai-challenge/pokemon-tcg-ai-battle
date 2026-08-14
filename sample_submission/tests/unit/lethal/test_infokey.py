"""InfoKey の情報境界テスト(設計 §9.1 T2 / 規則 R1・R2a)。

観点:
1. 山札の順序はキーに影響しない(multiset のみ)
2. 手札の並びが違うだけの局面は同一キー(意味へ解決してから正規化する)
3. 手札の中身が変わればキーは変わる
4. serial(物理個体)はキーに入らない
5. デッキ公開時、listing の**並び**はキーに影響しない(multiset のみ)
6. 選択肢を意味へ解決できない場合は、位置を保って**合流させない**
"""

from collections import Counter

from ptcg_ai.search.lethal.infokey import info_key

from tests.unit.lethal import builders as b


def test_deck_order_does_not_change_the_key():
    # 観点1: 同じ multiset の別順列を渡してもキーは同一。
    obs = b.simple_observation([1079, 1086, 1182])
    order_a = [5, 5, 743, 1182, 66]
    order_b = [66, 743, 5, 1182, 5]
    assert Counter(order_a) == Counter(order_b)

    key_a = info_key(obs, me=0, deck_multiset=Counter(order_a))
    key_b = info_key(obs, me=0, deck_multiset=Counter(order_b))
    assert key_a == key_b
    assert key_a.digest() == key_b.digest()


def test_deck_multiset_change_does_change_the_key():
    # 観点1の対: multiset そのものが違えば別局面として扱う。
    obs = b.simple_observation([1079])
    key_a = info_key(obs, me=0, deck_multiset=Counter([5, 5, 743]))
    key_b = info_key(obs, me=0, deck_multiset=Counter([5, 743, 743]))
    assert key_a != key_b


def test_hand_permutation_gives_the_same_key():
    # 観点2: 手札の並び順は意思決定に無関係。PLAY 選択肢の index も並び替わるが、
    # カードIDへ解決してからソートするので同一キーになる。
    obs_a = b.simple_observation([1079, 1086, 1182])
    obs_b = b.simple_observation([1182, 1079, 1086])
    deck = Counter([5, 743])
    assert info_key(obs_a, me=0, deck_multiset=deck) == info_key(obs_b, me=0, deck_multiset=deck)


def test_hand_content_change_changes_the_key():
    # 観点3.
    obs_a = b.simple_observation([1079, 1086])
    obs_b = b.simple_observation([1079, 1097])
    deck = Counter([5])
    assert info_key(obs_a, me=0, deck_multiset=deck) != info_key(obs_b, me=0, deck_multiset=deck)


def test_serials_do_not_enter_the_key():
    # 観点4: 同じカードIDなら物理個体が違ってもキーは同じ。
    obs_a = b.simple_observation([1079, 1086], serial_base=1)
    obs_b = b.simple_observation([1079, 1086], serial_base=500)
    deck = Counter([5])
    assert info_key(obs_a, me=0, deck_multiset=deck) == info_key(obs_b, me=0, deck_multiset=deck)


def test_deck_listing_order_does_not_change_the_key():
    # 観点5: サーチでデッキが公開されたとき、listing の並びは
    # 実際の山札順を反映しうるので条件付けに使わない(B1 の情報境界)。
    listing_a = [b.card(743, 10), b.card(1182, 11), b.card(5, 12)]
    listing_b = [b.card(5, 12), b.card(743, 10), b.card(1182, 11)]

    def build(listing):
        hand = [b.card(1079, 1)]
        me = b.player(hand=hand)
        opp = b.opponent()
        options = [
            b.Option(type=b.OptionType.CARD, area=b.AreaType.DECK, index=i, playerIndex=0)
            for i in range(len(listing))
        ]
        return b.observation(
            b.select(options, context=b.SelectContext.TO_HAND, deck=listing),
            b.state(me=me, opp=opp),
        )

    deck = Counter([743, 1182, 5])
    assert info_key(build(listing_a), me=0, deck_multiset=deck) == info_key(
        build(listing_b), me=0, deck_multiset=deck
    )


def test_unresolvable_options_stay_positional():
    # 観点6: 解決できない選択肢(参照先が観測に無い)は位置を保持し、
    # 並びが違えば別キーにする = 安全側(合流させない)。
    def build(indices):
        hand = [b.card(1079, 1)]
        me = b.player(hand=hand)
        opp = b.opponent()
        options = [
            # deck listing が無いのに DECK を指す = 解決不能
            b.Option(type=b.OptionType.CARD, area=b.AreaType.DECK, index=i, playerIndex=0)
            for i in indices
        ]
        return b.observation(
            b.select(options, context=b.SelectContext.TO_HAND), b.state(me=me, opp=opp)
        )

    deck = Counter([5])
    key_a = info_key(build([0, 1, 2]), me=0, deck_multiset=deck)
    key_b = info_key(build([2, 1, 0]), me=0, deck_multiset=deck)
    assert key_a != key_b


def test_opponent_hidden_hand_is_not_in_the_key():
    # 相手の手札は Observation 上 None。手札枚数だけがキーに入ることを確認する。
    obs = b.simple_observation([1079])
    opp = obs.current.players[1]
    assert opp.hand is None
    key_before = info_key(obs, me=0, deck_multiset=Counter([5]))
    opp.handCount += 1
    key_after = info_key(obs, me=0, deck_multiset=Counter([5]))
    assert key_before != key_after  # 枚数(公開情報)は反映される


def test_deck_multiset_none_falls_back_to_deck_count_only():
    # multiset を渡さない場合でも例外にせず、粗いキーになるだけ。
    obs = b.simple_observation([1079])
    key_a = info_key(obs, me=0)
    key_b = info_key(obs, me=0, deck_multiset=Counter([5, 5]))
    assert key_a != key_b
    assert info_key(obs, me=0) == key_a
