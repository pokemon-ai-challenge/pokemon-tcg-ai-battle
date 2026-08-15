"""自分の60枚から、公開済みゾーンを引いた「未確認プール」（山札∪サイド）を管理する。

design.md §4 の「自分側は組み合わせ論だけで済む」設計をそのままコード化したもの。
自分の60枚は100%既知なので、手札・場（本体・付属エネルギー・付属道具・進化元）・トラッシュという
全部見えているゾーンを引き算するだけで、「山札か、サイドか」だけが不確定な未確認プール
（``card_id -> 残り枚数`` の多重集合）が厳密に確定する。誤差の入り込む余地がないので、
毎ターン ``update()`` で ``State`` から作り直す（差分更新・状態の持ち越しはしない。§4.4）。

山札サーチ（``resolve_deck_search()``）だけは例外で、確率ではなく決定的な消し込みを行う。
"""

import random
from collections import Counter

from cg.api import Pokemon, SelectData, State

from .zone_math import prob_in_prize, prob_in_prize_exact


class OwnHiddenState:
    """自分の60枚のうち、まだ「山札かサイドか」が確定していないカードを管理する。

    - ``_pool``: 直近の ``update()`` 時点での「山札∪サイド」全体の多重集合
      （公開ゾーンを引き算しただけの生の集合。サーチによる確定情報は含まない）。
      不変条件 ``sum(_pool.values()) == deckCount + len(prize)`` は毎回 ``update()`` の中で検算する。
    - ``_confirmed_deck`` / ``_confirmed_prize``: ``_pool`` の部分集合で、山札サーチにより
      それぞれ「山札確定」「サイド確定」と分かった枚数（``card_id -> 枚数``）。
      山札サーチは自分の山札を丸ごと公開する（design.md §4.3、実データ検証で確認）ため、
      サーチが起きた瞬間、その時点のプール全体が2つに完全二分される（未確定分は残らない）。
      超幾何計算の対象（＝真に未確定なプール）からは、この2つの確定分を両方除外する。
    """

    def __init__(self, deck_card_ids: list[int]) -> None:
        self._full_deck: Counter[int] = Counter(deck_card_ids)
        self._pool: Counter[int] = Counter(deck_card_ids)
        self._confirmed_deck: Counter[int] = Counter()
        self._confirmed_prize: Counter[int] = Counter()
        # update() が一度も呼ばれる前は「今どれだけ山札/サイドに割れているか」自体が不明なので、
        # 両方 0 にしておく（marginals()/sample() は「情報なし」として振る舞う）。
        self._deck_count = 0
        self._prize_count = 0
        # 直近の update() 呼び出し時点での自分の手札（card_id -> 枚数）。セットアップ中の
        # 「バトル場ポケモンの同時公開待ち」不整合（update()のdocstring参照）を検知するためだけに
        # 使う補助情報で、_pool 自体の差分更新には使わない（§4.4 の原則は _pool に限る）。
        self._last_hand_ids: Counter[int] = Counter()
        # 上と同じ不整合の対処用: 一度「伏せられている自分のバトル場ポケモンはこのcard_id」と
        # 特定できたら、伏せが解除されるまで覚えておく。DrawCount（マリガンのボーナスドロー枚数
        # 選択）等の中間選択を挟んで同じ伏せが複数ターンにわたって続くことがあり、その間は手札に
        # 既に存在しない（＝前回との差分では二度と検知できない）ため、差分検知とは別に保持する。
        self._folded_own_active_id: int | None = None

    # ------------------------------------------------------------------
    # 更新

    def update(self, state: State, select: SelectData | None = None) -> None:
        """毎ターン呼ぶ。公開ゾーン（手札・場・トラッシュ・自分のスタジアム）を全60枚から引き、
        未確認プールを作り直す。

        ``select`` は省略可能（プラン記載のシグネチャは ``update(self, state)`` のみ）。ただし
        実データ検証で次の一時的不整合が判明したため、渡せる場合は渡すことを推奨する: グッズ/
        サポートが山札サーチ効果で ``TO_HAND``/``TO_BENCH`` 等の選択を発生させている最中、
        プレイされたカード自身は手札から既に消えているがトラッシュへはまだ移っていない
        （``SelectData.effect`` が指す一瞬だけの「解決中」状態）。``state`` だけではこの1枚を
        判別できず、公開ゾーンの合計が60枚に1枚足りなくなる。``select.effect`` が渡されていれば、
        「合計がちょうど1枚多い」かつ「そのカードがプール内に残っている」場合に限り、そのカードを
        観測済み（=トラッシュ行き確定と同等）として扱い、この一時的なズレを埋める
        （安全側の適用条件: 条件に合わない場合は何もせず、不変条件アサートに委ねる）。

        もう1つ、``select.effect`` では拾えない一時的不整合が実データ検証（``battle_review_viewer``
        を実際の対戦に接続）で見つかっている: セットアップ中（``state.turn == 0``）、両プレイヤーが
        バトル場を選び終えて同時公開されるまでの間、**自分で既に選んだはずの自分のバトル場ポケモン**
        が ``player.active`` 上でも ``None``（伏せ）のまま観測されることがある
        （``tests/local_sim/test_local_game_hidden_info.py`` が既に検知していた既知事象。従来は
        「1回だけ不変条件が破れて次ターンに自己修復する」を許容するだけで、本体側の修正はしていなかった）。
        この瞬間、選んだポケモンは手札からは既に消えているが、``player.active`` にも
        ``player.bench`` にも まだ現れないため、公開ゾーンの合計が60枚に1枚足りなくなる。
        ``select.effect`` に相当する「今解決中のカード」を直接教えてくれるフィールドが無いので、
        代わりに「前回の ``update()`` 時点の手札」と「今回の手札」の差分を使う: ``player.active`` に
        ``None`` が含まれ、かつ手札から消えたカードがちょうど1種類1枚だけ特定できる場合に限り、
        そのカードを観測済み扱いにする（``_last_hand_ids`` は ``_pool`` 本体の差分更新には使わない。
        あくまでこの一時的不整合を安全側に検知するためだけの補助情報）。

        なお、この伏せは ``SelectContext.DRAW_COUNT``（マリガンのボーナスドロー枚数選択）のような
        中間選択を挟んで、自分の複数回の手番にまたがって続くことがある（実データ検証で確認）。
        2回目以降の呼び出しでは、伏せられたカードは既に手札から消えて久しいため手札の差分では
        二度と検知できない。そのため一度特定できたカード ID は ``player.active`` の伏せが解ける
        まで ``_folded_own_active_id`` に覚えておき、以後の呼び出しでは差分検知をやり直さずに
        その ID をそのまま使う。
        """
        player = state.players[state.yourIndex]

        hand_ids: Counter[int] = Counter(card.id for card in player.hand) if player.hand else Counter()

        observed: Counter[int] = Counter()
        observed.update(hand_ids)
        for pokemon in player.active:
            if pokemon is not None:
                self._add_pokemon(observed, pokemon)
        for pokemon in player.bench:
            self._add_pokemon(observed, pokemon)
        for card in player.discard:
            observed[card.id] += 1
        # スタジアムは共有ゾーンだが、自分が出した1枚は自分の60枚から出ている（design.mdの§4.1に
        # 明記が無いが、実データ検証で欠落が判明。State.stadium は playerIndex で所有者が分かる）。
        for card in state.stadium:
            if card.playerIndex == state.yourIndex:
                observed[card.id] += 1

        pool = self._full_deck.copy()
        pool.subtract(observed)
        for card_id, count in pool.items():
            assert count >= 0, (
                f"card_id={card_id} が公開ゾーンで元の枚数({self._full_deck[card_id]})を"
                f"超えて観測された（二重カウントの疑い）"
            )
        pool = +pool  # 単項+でゼロ以下の要素を掃除

        new_deck_count = player.deckCount
        new_prize_count = len(player.prize)
        expected_total = new_deck_count + new_prize_count
        actual_total = sum(pool.values())

        if actual_total == expected_total + 1 and select is not None and select.effect is not None:
            # 上記docstring記載の既知の一時的不整合(解決中カード1枚)。プール内に対象カードの
            # 残数が有る場合に限り、観測済み扱いにして帳尻を合わせる。
            effect_id = select.effect.id
            if pool.get(effect_id, 0) > 0:
                pool[effect_id] -= 1
                actual_total -= 1

        active_folded = any(p is None for p in player.active)
        if not active_folded:
            # 伏せが解けた（=通常どおり player.active から観測できる）ので、次に別の伏せが
            # 起きたときに前回の特定結果を誤って使い回さないよう、記憶を破棄する。
            self._folded_own_active_id = None
        elif actual_total == expected_total + 1 and state.turn == 0:
            # 上記docstring記載の、セットアップ中の同時公開待ちによる一時的不整合。
            # まず、以前の呼び出しで既に特定済みのカード（伏せが複数ターン続いている場合）を
            # 優先して使う。未特定なら、前回の手札との差分で「手札から消えたが、まだ場にも
            # トラッシュにも現れていないカード」を特定する。候補が複数（または0）ある場合は
            # 取り違えのリスクがあるため補正せず、assertに委ねる（安全側の適用条件。既存の
            # select.effect ベースの補正と同じ思想）。
            folded_id = self._folded_own_active_id
            if folded_id is None:
                vanished_from_hand = self._last_hand_ids - hand_ids
                if sum(vanished_from_hand.values()) == 1:
                    (folded_id,) = vanished_from_hand.elements()
            if folded_id is not None and pool.get(folded_id, 0) > 0:
                pool[folded_id] -= 1
                actual_total -= 1
                self._folded_own_active_id = folded_id

        assert actual_total == expected_total, (
            f"不変条件違反: 未確認プール={actual_total} != "
            f"deckCount({new_deck_count}) + len(prize)({new_prize_count})"
        )

        self._pool = pool
        self._deck_count = new_deck_count
        self._prize_count = new_prize_count
        self._last_hand_ids = hand_ids

        # サーチで既に確定した枚数が、今回のプールに存在する枚数を超えないようクランプする
        # （確定した山札/サイドのカードが手札等へ移り公開ゾーンに含まれるようになった場合の整合を取る）。
        # プールが縮んだ場合、どちらの確定分が失われたかは State だけからは判別できないため、
        # サイド確定（希少で価値の高い情報）を優先して残し、山札確定側から先に削る。
        for card_id in set(self._confirmed_deck) | set(self._confirmed_prize):
            available = self._pool.get(card_id, 0)
            new_prize = min(self._confirmed_prize.get(card_id, 0), available)
            new_deck = min(self._confirmed_deck.get(card_id, 0), available - new_prize)
            if new_prize > 0:
                self._confirmed_prize[card_id] = new_prize
            elif card_id in self._confirmed_prize:
                del self._confirmed_prize[card_id]
            if new_deck > 0:
                self._confirmed_deck[card_id] = new_deck
            elif card_id in self._confirmed_deck:
                del self._confirmed_deck[card_id]

        # card_id 単位のクランプだけでは検知できない矛盾が残ることがある: 同じ card_id に
        # 山札確定分とサイド確定分が両方あるカードが、どちらか片方だけ公開ゾーンへ移動した場合、
        # State（card_id単位の集計）だけからは「どちらの確定分が減ったか」を区別できない
        # （実データ検証で発見。serial単位の追跡は本フェーズの設計外）。合計がありえない値
        # （サイド確定合計 > 実際のサイド枚数、等）になっていたら、安全側に倒して確定情報を
        # 丸ごと破棄する（以後は通常の超幾何計算に戻るだけで、不変条件そのものは壊れない。
        # 次に resolve_deck_search が呼ばれれば再び確定情報が張り直される）。
        if (
            sum(self._confirmed_prize.values()) > self._prize_count
            or sum(self._confirmed_deck.values()) > self._deck_count
        ):
            self._confirmed_deck = Counter()
            self._confirmed_prize = Counter()

    @staticmethod
    def _add_pokemon(observed: Counter[int], pokemon: Pokemon) -> None:
        observed[pokemon.id] += 1
        for card in pokemon.energyCards:
            observed[card.id] += 1
        for card in pokemon.tools:
            observed[card.id] += 1
        for card in pokemon.preEvolution:
            observed[card.id] += 1

    def resolve_deck_search(self, select: SelectData) -> None:
        """山札サーチで山札の中身が丸ごと公開された瞬間に呼ぶ。

        実データ検証結果（``tests/local_sim`` 相当のローカル対戦を10試合・149回の山札公開イベントで
        確認): ``select.deck`` は ``SelectContext.LOOK`` では一度も発火せず、``TO_HAND``（サーチ効果の
        対象選択）や ``TO_BENCH``（ベンチに出すポケモンをサーチする効果）等、選択肢の出所が山札の場合に
        ``select.context`` の値によらず埋まる。埋まる際は必ず「今の山札の中身全部」（
        ``len(select.deck) == player.deckCount``、伏せ無し）であり、部分的な公開は観測されなかった。
        そのため発火条件は ``select.context == SelectContext.LOOK`` ではなく ``select.deck is not None``
        のみとする（design.md/plan の前提から逸脱。詳細は実装報告を参照）。

        山札が丸ごと公開されるため、この瞬間プール全体が「山札確定」（``select.deck`` に写っている分、
        ``_confirmed_deck``）と「サイド確定」（写っていない分、``_confirmed_prize``）に完全に二分される。
        以後、この2つに含まれるカードは確率計算の対象から外れる（design.md §4.3）。
        """
        if select.deck is None:
            return

        seen_in_deck: Counter[int] = Counter(card.id for card in select.deck if card is not None)

        for card_id, total_count in self._pool.items():
            if total_count <= 0:
                continue
            in_deck_count = min(seen_in_deck.get(card_id, 0), total_count)
            prize_count = total_count - in_deck_count

            if in_deck_count > 0:
                self._confirmed_deck[card_id] = in_deck_count
            elif card_id in self._confirmed_deck:
                del self._confirmed_deck[card_id]

            if prize_count > 0:
                self._confirmed_prize[card_id] = prize_count
            elif card_id in self._confirmed_prize:
                del self._confirmed_prize[card_id]

    # ------------------------------------------------------------------
    # 参照

    def _undetermined_count(self, card_id: int, total_count: int) -> int:
        """この card_id のうち、山札確定・サイド確定のどちらでもない残り枚数。"""
        confirmed = self._confirmed_deck.get(card_id, 0) + self._confirmed_prize.get(card_id, 0)
        return total_count - confirmed

    def _undetermined_sizes(self) -> tuple[int, int]:
        """超幾何計算に使う (M, n) を返す。サーチで確定済みの分（山札・サイド両方）は除く。"""
        undetermined_deck = self._deck_count - sum(self._confirmed_deck.values())
        undetermined_prize = self._prize_count - sum(self._confirmed_prize.values())
        undetermined_pool = undetermined_deck + undetermined_prize
        return undetermined_pool, undetermined_prize

    def marginals(self) -> dict[int, dict[str, float]]:
        """card_id -> {"deck": p, "prize": p}。

        ``p`` はそれぞれ「この card_id の残り枚数のうち、少なくとも1枚がそのゾーンにある確率」。
        確定済み（サーチで判明済み）のカードは常に ``{"deck": 1.0, "prize": 0.0}`` または
        ``{"deck": 0.0, "prize": 1.0}``。
        """
        pool_size, prize_size = self._undetermined_sizes()
        result: dict[int, dict[str, float]] = {}

        for card_id, total_count in self._pool.items():
            if total_count <= 0:
                continue
            confirmed_deck = self._confirmed_deck.get(card_id, 0)
            confirmed_prize = self._confirmed_prize.get(card_id, 0)
            undetermined_count = total_count - confirmed_deck - confirmed_prize

            if confirmed_prize > 0:
                prize_p = 1.0
            elif undetermined_count > 0 and pool_size > 0:
                prize_p = prob_in_prize(pool_size, prize_size, undetermined_count, at_least=1)
            else:
                prize_p = 0.0

            if confirmed_deck > 0:
                deck_p = 1.0
            elif undetermined_count > 0 and pool_size > 0:
                # 「残っている未確定コピーが1枚も山札に無い（全部サイド）」の余事象。
                deck_p = 1.0 - prob_in_prize_exact(pool_size, prize_size, undetermined_count, undetermined_count)
            else:
                deck_p = 0.0

            result[card_id] = {"deck": deck_p, "prize": prize_p}

        return result

    def sample(self, rng: random.Random | None = None) -> tuple[list[int], list[int]]:
        """(deck_card_ids, prize_card_ids) を1組返す。多変量超幾何分布に従うサンプリング。

        未確認プールから ``prize`` の未確定分を非復元抽出すれば、カード種類ごとの確率は
        自動的に超幾何分布に従う。確定済み（``_confirmed_deck`` / ``_confirmed_prize``）は
        無条件でそれぞれ山札側・サイド側に入れる。
        """
        rng_obj: random.Random = rng if rng is not None else random  # type: ignore[assignment]
        _, undetermined_prize_slots = self._undetermined_sizes()

        undetermined_pool: list[int] = []
        for card_id, total_count in self._pool.items():
            undetermined_count = self._undetermined_count(card_id, total_count)
            undetermined_pool.extend([card_id] * undetermined_count)

        sampled_prize = (
            rng_obj.sample(undetermined_pool, undetermined_prize_slots)
            if undetermined_prize_slots > 0
            else []
        )

        remaining_deck = Counter(undetermined_pool)
        remaining_deck.subtract(Counter(sampled_prize))
        deck_card_ids: list[int] = []
        for card_id, count in (+remaining_deck).items():
            deck_card_ids.extend([card_id] * count)
        for card_id, count in self._confirmed_deck.items():
            deck_card_ids.extend([card_id] * count)

        prize_card_ids: list[int] = list(sampled_prize)
        for card_id, count in self._confirmed_prize.items():
            prize_card_ids.extend([card_id] * count)

        return deck_card_ids, prize_card_ids
