"""入力拡張(extra_features)。self-play 学習と実行時で同一に計算する parity-safe な特徴。

第一弾＝自分の未確認プール(山札∪サイド)から求める「自サイド構成リスク」。
OwnHiddenState は毎回 current State + 既知デッキから作り直す純関数(own_hidden_state §4.4、
差分更新しない)なので、サーチ確定(履歴依存)を使わずに marginals() だけ使えば、学習側
(リプレイの State)と実行側(生ゲームの State)で厳密に同一の出力になる=非転移リスク無し。

encode_state(obs, extra_features=own_resource_features(state, deck_ids)) の形で末尾接続する。
"""
from __future__ import annotations

from cg.api import CardType, State

from ptcg_ai.hidden_information.own_hidden_state import OwnHiddenState
from ptcg_ai.shared import card_cache

# 特徴名(この順で並ぶ)。学習側/実行側で共通。
OWN_RESOURCE_FEATURE_NAMES: list[str] = [
    "own_prize_risk_pokemon",   # 自ポケモンが少なくとも1枚サイドに埋もれている確率の総和
    "own_prize_risk_energy",    # エネルギー同上
    "own_prize_risk_trainer",   # トレーナーズ同上
    "own_prize_risk_basic",     # たねポケモン同上(展開の生命線=最重要)
]
OWN_RESOURCE_FEATURE_COUNT: int = len(OWN_RESOURCE_FEATURE_NAMES)

_ZERO = [0.0] * OWN_RESOURCE_FEATURE_COUNT

_TRAINER_TYPES = frozenset({CardType.ITEM, CardType.TOOL, CardType.SUPPORTER, CardType.STADIUM})
_ENERGY_TYPES = frozenset({CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY})


def _card_or_none(card_id: int):
    try:
        return card_cache.get_card(card_id)
    except Exception:  # noqa: BLE001 未知IDは寄与0
        return None


def own_resource_features(state: State | None, deck_card_ids: list[int] | None) -> list[float]:
    """自サイド構成リスクの固定長ベクトル。current State + 既知デッキの純関数(parity-safe)。

    OwnHiddenState をその場で作って1回だけ update(state)→marginals() を取る。サーチ確定は
    使わない(履歴非依存を保つため)。State/deck が無い・estimator が例外なら全0(安全側)。
    """
    if state is None or not deck_card_ids:
        return list(_ZERO)
    try:
        ohs = OwnHiddenState(list(deck_card_ids))
        ohs.update(state)
        marg = ohs.marginals()  # card_id -> {"deck": p, "prize": p}
    except Exception:  # noqa: BLE001 estimator の一時的不整合は安全側に0
        return list(_ZERO)

    pk = en = tr = basic = 0.0
    for card_id, probs in marg.items():
        p_prize = float(probs.get("prize", 0.0))
        if p_prize <= 0.0:
            continue
        card = _card_or_none(card_id)
        if card is None:
            continue
        ct = card.cardType
        if ct == CardType.POKEMON:
            pk += p_prize
            if getattr(card, "basic", False):
                basic += p_prize
        elif ct in _ENERGY_TYPES:
            en += p_prize
        elif ct in _TRAINER_TYPES:
            tr += p_prize
    return [pk, en, tr, basic]


def opponent_belief_features(state: State | None, predictor) -> list[float]:
    """相手アーキタイプ予測の固定長ベクトル(parity-safe, belief-conditioned RL 用)。

    設計: docs/plans/belief-conditioned-rl-plan.md。**現在 State の相手可視カードだけ**から
    予測する純関数にする(``tracker``/累積 knowledge は履歴依存なので使わない)。実装は
    ``OpponentKnowledge`` を毎回 fresh に作り ``update_from_state(state)`` で現在 State のみ
    observe → ``get_prediction_features()["observed_cards"]``(name単位)→ ``predictor.predict``。
    出力は ``predictor._classes`` の固定順の確率(長さ=クラス数, 決定的)。

    parity: 学習側(collect_field)と推論側(PolicyModel._compute_extra)で **同一 predictor・同一
    関数**を使うこと。predictor 未ロード/State無し/例外は全0(classes 数)を返す(安全側, own_resource と同じ)。
    """
    classes = list(getattr(predictor, "_classes", []) or [])
    n = len(classes)
    if n == 0 or predictor is None or not getattr(predictor, "is_ready", False):
        return [0.0] * n
    if state is None:
        return [0.0] * n
    try:
        from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge
        knowledge = OpponentKnowledge()
        knowledge.update_from_state(state)
        observed = knowledge.get_prediction_features().get("observed_cards", {})
        probs = predictor.predict(observed, int(getattr(state, "turn", 0) or 0))
    except Exception:  # noqa: BLE001 - 予測の失敗が意思決定/学習を止めてはならない
        return [0.0] * n
    return [float(probs.get(c, 0.0)) for c in classes]


# deck-sustain 特徴名(この順で並ぶ)。学習側/実行側で共通。
DECK_SUSTAIN_FEATURE_NAMES: list[str] = [
    "own_trash_pokemon_count",  # トラッシュのポケモン枚数(せいなるはいの回復量/ループ持続の鍵。base特徴に無い)
    "own_deck_danger",          # deckout危険度(1.0=山切れ間近, 0=安全)。危険域に焦点を当てた飽和信号
]
DECK_SUSTAIN_FEATURE_COUNT: int = len(DECK_SUSTAIN_FEATURE_NAMES)


def deck_sustain_features(state: State | None) -> list[float]:
    """deckout管理(ノコッチループ/せいなるはい)に必要だが方策に不足している特徴。parity-safe純関数。

    現状: 山残枚数は base 特徴にあるが、**トラッシュのポケモン枚数**(せいなるはいの回復量・ループの
    持続を左右)が無い(ユーザー指摘)。それと deckout 危険度(飽和)を足す。current State のみの決定的
    読み取りなので学習側/実行側で厳密一致。State 無し/例外は全0(安全側)。
    """
    if state is None:
        return [0.0] * DECK_SUSTAIN_FEATURE_COUNT
    try:
        me = state.yourIndex
        trash_pk = 0
        for c in (state.players[me].discard or []):
            if c is None:
                continue
            card = _card_or_none(c.id)
            if card is not None and card.cardType == CardType.POKEMON:
                trash_pk += 1
        deck = int(state.players[me].deckCount or 0)
        danger = 1.0 - min(deck, 15) / 15.0
        return [min(trash_pk, 10) / 10.0, danger]
    except Exception:  # noqa: BLE001 - 特徴計算の失敗が意思決定/学習を止めてはならない
        return [0.0] * DECK_SUSTAIN_FEATURE_COUNT


def compute_extra_features(state, extra_flag: str | None, deck_card_ids, predictor) -> list[float] | None:
    """``extra_features`` フラグに応じた入力拡張を **固定順(own_resource → opp_belief → deck_sustain)** で連結する。

    学習側(collect_field)と推論側(PolicyModel._compute_extra)が **この単一関数を呼ぶ**ことで、
    順序・計算・parity を一元的に保証する。フラグは部分文字列で判定するので "own_resource"、
    "opp_belief"、"deck_sustain"、それらの連結("own_resource+deck_sustain" 等)いずれも可。
    該当無し/フラグ無しは None(従来どおり)。
    """
    if not extra_flag:
        return None
    parts: list[float] = []
    if "own_resource" in extra_flag:
        parts += own_resource_features(state, deck_card_ids)
    if "opp_belief" in extra_flag:
        parts += opponent_belief_features(state, predictor)
    if "deck_sustain" in extra_flag:
        parts += deck_sustain_features(state)
    return parts or None
