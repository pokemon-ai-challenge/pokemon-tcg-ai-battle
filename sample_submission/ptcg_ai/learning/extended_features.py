"""既存の盤面特徴(166次元)に足す追加特徴。

既存の encoder は「どのカードか」をほとんど見ていない。HP や打点といった数値は入るが、
バトル場にいるのがケーシィなのかフーディンなのかは区別されず、山札やトラッシュの中身も
枚数しか入らない。ここではその欠けを埋める。

追加する内容(設定 JSON の語彙サイズで決まる):
  A  自分の山札+サイドに残っているカードの構成   … デッキリストから見えているゾーンを引く
  B  自分のトラッシュの構成                     … discard の中身
  E1 自分のバトル場 / ベンチのポケモンの種類
  E2 相手のバトル場 / ベンチのポケモンの種類(語彙外は「その他」へ)
  D  相手のアーキタイプ予測(21クラスの確率)

**カードIDを数値のまま入れてはいけない。** ID 500 と 501 は近くない。どのカードかは
種類ごとのスイッチ(該当すれば1)で表す。

相手の非公開情報は使わない。相手の手札・山札の中身は参照せず、見えているゾーン
(バトル場・ベンチ・トラッシュ)だけから計算する。

設定は `extended_profile_*.json`。デッキを変えると語彙が変わるので、プロファイルも作り直す。
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# 相手アーキタイプ予測は毎決定ごとに呼ばれるので、予測器は1度だけ作って使い回す。
_predictor = None
_card_names: dict[int, str] | None = None


class Profile:
    """追加特徴の語彙。どの位置がどのカードかを固定する。"""

    def __init__(self, payload: dict):
        self.name: str = payload["name"]
        self.own_deck_ids: list[int] = [int(x) for x in payload["own_deck_ids"]]
        self.own_deck_counts: dict[int, int] = {
            int(k): int(v) for k, v in payload["own_deck_counts"].items()}
        self.own_poke_ids: list[int] = [int(x) for x in payload["own_poke_ids"]]
        self.opp_poke_vocab: list[int] = [int(x) for x in payload["opp_poke_vocab"]]
        self.archetypes: list[str] = list(payload["archetypes"])
        # A ブロックの作り方。
        #   "count"     … デッキリスト - 見えているゾーン。山札とサイドを区別しない(従来)
        #   "marginals" … OwnHiddenState を使い、山札にある確率とサイドにある確率を分ける。
        #                 山札サーチで中身を見た結果が確定情報として残るので、サイド落ちが
        #                 方策に見えるようになる。使う側は decide ごとに `observe()` を呼ぶこと
        self.own_zone: str = str(payload.get("own_zone", "count"))
        if self.own_zone not in ("count", "marginals"):
            raise ValueError(f"own_zone が不正: {self.own_zone}")

        self._deck_idx = {cid: i for i, cid in enumerate(self.own_deck_ids)}
        self._own_poke_idx = {cid: i for i, cid in enumerate(self.own_poke_ids)}
        self._opp_poke_idx = {cid: i for i, cid in enumerate(self.opp_poke_vocab)}

        d, p, o, a = (len(self.own_deck_ids), len(self.own_poke_ids),
                      len(self.opp_poke_vocab), len(self.archetypes))
        # A + (所在不明の枚数 [+ 推定が使えたか]) + B + E1(場+ベンチ) + E2(場+ベンチ、各+その他) + D
        a_block = (3 * d + 2) if self.own_zone == "marginals" else (d + 1)
        self.count = a_block + d + p + p + (o + 1) + (o + 1) + a

    @property
    def feature_names(self) -> list[str]:
        if self.own_zone == "marginals":
            n: list[str] = []
            for c in self.own_deck_ids:
                n += [f"remain_{c}", f"in_deck_p_{c}", f"in_prize_p_{c}"]
            n += ["own_unaccounted_count", "own_zone_known"]
        else:
            n = [f"deck_prize_remain_{c}" for c in self.own_deck_ids]
            n += ["own_unaccounted_count"]
        n += [f"self_discard_{c}" for c in self.own_deck_ids]
        n += [f"self_active_is_{c}" for c in self.own_poke_ids]
        n += [f"self_bench_has_{c}" for c in self.own_poke_ids]
        n += [f"opp_active_is_{c}" for c in self.opp_poke_vocab] + ["opp_active_is_other"]
        n += [f"opp_bench_has_{c}" for c in self.opp_poke_vocab] + ["opp_bench_has_other"]
        n += [f"opp_archetype_{a}" for a in self.archetypes]
        return n


def load_profile(name: str) -> Profile:
    path = _HERE / f"extended_profile_{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"追加特徴のプロファイルが無い: {path}")
    return Profile(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------- 自分の山札/サイドの追跡
# `own_zone="marginals"` のプロファイルで使う。`OwnHiddenState` は自分の60枚から見えている
# ゾーンを引いて「山札∪サイド」を出し、山札サーチが起きた瞬間に中身を確定情報として
# 覚えておく。ここが「サイド落ち」を知る唯一の経路で、盤面のスナップショットだけからは
# 決して作れない(サーチしたという過去の観測が要る)。
#
# 決定点ごとに `observe(obs)` を呼ぶ側の責任にしている。`encode()` は state しか
# 受け取らず、サーチの検知には `obs.select` が要るため。
_own_deck_ids: list[int] | None = None
_own_states: dict = {}
_observe_errors = 0


def begin_match(deck_ids: list[int]) -> None:
    """1試合の開始時に呼ぶ。前の試合の推定を捨てる。"""
    global _own_deck_ids, _observe_errors
    _own_deck_ids = [int(x) for x in deck_ids]
    _own_states.clear()
    _observe_errors = 0


def observe(obs) -> None:
    """決定点ごとに呼ぶ。自分の山札/サイドの推定を進める。

    `OwnHiddenState.update()` はセットアップ中の一時的な不整合で assert に落ちることが
    あり(伏せたバトル場が公開されるまでの1手など、本体側の docstring 参照)、次の手で
    自己修復する。対戦を止める理由にはならないので握るが、握った回数は数えておく
    (黙って壊れたまま学習が進むのを避けるため。`observe_errors()` で確認できる)。
    """
    global _observe_errors
    if _own_deck_ids is None:
        return
    state = getattr(obs, "current", None)
    if state is None:
        return
    from ptcg_ai.hidden_information.own_hidden_state import OwnHiddenState

    me = state.yourIndex
    st = _own_states.get(me)
    if st is None:
        st = OwnHiddenState(_own_deck_ids)
        _own_states[me] = st
    select = getattr(obs, "select", None)
    try:
        st.update(state, select)
        if select is not None and getattr(select, "deck", None) is not None:
            st.resolve_deck_search(select)
    except Exception:  # noqa: BLE001 -- 一時的な不整合。次の手で作り直される
        _observe_errors += 1


def observe_errors() -> int:
    return _observe_errors


def _resolve_own_state(player_index: int):
    """この決定点で使う `OwnHiddenState` を返す。無ければ None。

    経路が2つある。
      学習(collect_parallel): この模块の `begin_match()` / `observe()` が持つ
      本番(ml_policy_agent):  `hidden_information.match_context` が持つ
                              (`agent()` の冒頭で毎回 `update()` している)

    本番側は既に同じ推定を回しているので、二重に計算せずそちらを借りる。ここを繋がないと
    提出物では確率が全部 0 になり、学習したサイド落ちの判断がまるごと死ぬ。
    """
    st = _own_states.get(player_index)
    if st is not None:
        return st
    try:
        from ptcg_ai.hidden_information import match_context
        st = match_context.get_own_state(player_index)
    except Exception:  # noqa: BLE001
        return None
    # 一度も update() されていないと山札もサイドも 0 枚扱いで、確率が意味を持たない。
    if getattr(st, "_deck_count", 0) + getattr(st, "_prize_count", 0) <= 0:
        return None
    return st


def _get_card_names() -> dict[int, str]:
    """カードID -> 英語名。相手アーキタイプ予測の語彙が英語名で作られているため。
    エンジンから取れるので data/ の CSV には依存しない。"""
    global _card_names
    if _card_names is None:
        from cg.api import all_card_data
        _card_names = {c.cardId: c.name for c in all_card_data()}
    return _card_names


def _get_predictor():
    global _predictor
    if _predictor is None:
        from ptcg_ai.opponent_modeling.nb_predictor import NBDeckPredictor
        _predictor = NBDeckPredictor()
    return _predictor


def _board_pokemon(player) -> tuple[list[int], list[int]]:
    """(バトル場のカードID, ベンチのカードID)。伏せ中・不在は除く。"""
    act = [p.id for p in (player.active or []) if p is not None]
    ben = [p.id for p in (player.bench or []) if p is not None]
    return act, ben


def _visible_own_cards(me) -> list[int]:
    """自分のカードのうち、山札とサイド以外にあるものすべて。

    山札+サイドの残り = デッキリスト - ここで数えたもの。
    """
    out: list[int] = []
    for card in (me.hand or []):
        out.append(card.id)
    for card in (me.discard or []):
        out.append(card.id)
    for zone in (me.active or [], me.bench or []):
        for p in zone:
            if p is None:
                continue
            out.append(p.id)
            for c in (p.energyCards or []):
                out.append(c.id)
            for c in (p.tools or []):
                out.append(c.id)
            for c in (p.preEvolution or []):
                out.append(c.id)
    # 表になったサイドは中身が見えているので、残りから除く
    for card in (me.prize or []):
        if card is not None:
            out.append(card.id)
    return out


def encode(state, profile: Profile) -> list[float]:
    """追加特徴を返す。長さは profile.count。state が None なら全部 0。"""
    n = profile.count
    if state is None:
        return [0.0] * n

    feats: list[float] = []
    me = state.players[state.yourIndex]
    opp = state.players[1 - state.yourIndex]

    # --- A 山札+サイドの残り構成 ---
    # デッキリストから、中身が見えているゾーン(手札・場・トラッシュ・表のサイド)を引く。
    #
    # ただし所在の分からないカードが出る。準備フェーズでは自分のバトル場・ベンチも
    # 伏せられて None で返るのが典型例で、そのぶん残り枚数を多く見積もってしまう。
    # 「所在不明の枚数」を別の特徴として渡し、モデルに不確かさの量を伝える
    # (残り枚数の合計 = 山札 + 伏せサイド + 所在不明、が常に成り立つ)。
    visible = _visible_own_cards(me)
    seen: dict[int, int] = {}
    for cid in visible:
        seen[cid] = seen.get(cid, 0) + 1
    remain = [float(max(0, profile.own_deck_counts.get(cid, 0) - seen.get(cid, 0)))
              for cid in profile.own_deck_ids]
    hidden_prize = sum(1 for c in (me.prize or []) if c is None)
    unaccounted = (sum(profile.own_deck_counts.values()) - len(visible)
                   - me.deckCount - hidden_prize)

    if profile.own_zone == "marginals":
        # 残り枚数に加えて、それが山札にあるのかサイドにあるのかを確率で渡す。
        # 山札サーチで中身を見たあとは 1.0 / 0.0 に確定するので、そこが「サイド落ち」。
        # 推定が使えなかった決定点では確率を 0 にして、使えたかどうかを別の1本で伝える
        # (0 が「サイドに無い」なのか「分からない」なのかを取り違えないため)。
        st = _resolve_own_state(state.yourIndex)
        mg = None
        if st is not None:
            try:
                mg = st.marginals()
            except Exception:  # noqa: BLE001
                mg = None
        for i, cid in enumerate(profile.own_deck_ids):
            m = (mg or {}).get(cid)
            feats += [remain[i],
                      float(m["deck"]) if m else 0.0,
                      float(m["prize"]) if m else 0.0]
        feats.append(float(max(0, unaccounted)))
        feats.append(1.0 if mg else 0.0)
    else:
        feats += remain
        feats.append(float(max(0, unaccounted)))

    # --- B 自分のトラッシュの構成 ---
    disc: dict[int, int] = {}
    for card in (me.discard or []):
        disc[card.id] = disc.get(card.id, 0) + 1
    feats += [float(disc.get(cid, 0)) for cid in profile.own_deck_ids]

    # --- E1 自分の盤面のポケモン ---
    my_act, my_ben = _board_pokemon(me)
    a1 = [0.0] * len(profile.own_poke_ids)
    for cid in my_act:
        i = profile._own_poke_idx.get(cid)
        if i is not None:
            a1[i] = 1.0
    b1 = [0.0] * len(profile.own_poke_ids)
    for cid in my_ben:
        i = profile._own_poke_idx.get(cid)
        if i is not None:
            b1[i] += 1.0
    feats += a1 + b1

    # --- E2 相手の盤面のポケモン(語彙外は「その他」へ) ---
    op_act, op_ben = _board_pokemon(opp)
    a2 = [0.0] * (len(profile.opp_poke_vocab) + 1)
    for cid in op_act:
        i = profile._opp_poke_idx.get(cid)
        a2[i if i is not None else -1] = 1.0
    b2 = [0.0] * (len(profile.opp_poke_vocab) + 1)
    for cid in op_ben:
        i = profile._opp_poke_idx.get(cid)
        b2[i if i is not None else -1] += 1.0
    feats += a2 + b2

    # --- D 相手のアーキタイプ予測 ---
    feats += _archetype_probs(state, opp, profile)

    assert len(feats) == n, (len(feats), n)
    return feats


def _archetype_probs(state, opp, profile: Profile) -> list[float]:
    """相手の見えているカードから型を推定する。

    予測器そのものの用意に失敗した場合は**例外にする**。ここを黙って0で埋めると
    「特徴が全部0のまま学習が進む」という気づきにくい壊れ方をする(実際に一度踏んだ)。
    対戦中の想定外(未知のカード等)だけ0に倒す。
    """
    pred = _get_predictor()
    if not pred.is_ready:
        raise RuntimeError("相手アーキタイプ予測器を読み込めない "
                           "(deck_predictor_nb.json を確認する)")
    try:
        names = _get_card_names()
        observed: dict[str, int] = {}
        act, ben = _board_pokemon(opp)
        seen_ids = list(act) + list(ben)
        for card in (opp.discard or []):
            seen_ids.append(card.id)
        for zone in (opp.active or [], opp.bench or []):
            for p in zone:
                if p is None:
                    continue
                for c in list(p.energyCards or []) + list(p.tools or []) + \
                        list(p.preEvolution or []):
                    seen_ids.append(c.id)
        for cid in seen_ids:
            nm = names.get(cid)
            if nm:
                observed[nm] = observed.get(nm, 0) + 1
        if not observed:
            return [0.0] * len(profile.archetypes)
        post = pred.predict(observed, state.turn)
        return [float(post.get(a, 0.0)) for a in profile.archetypes]
    except Exception:
        # 予測が落ちても対戦は続ける(特徴が0になるだけ)
        return [0.0] * len(profile.archetypes)
