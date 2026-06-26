"""Phase 5 Task 5-1: 相手デッキ推定モジュール。

バトル中の公開情報（相手のバトル場・ベンチ・トラッシュ）から
estimate_deck_candidates() を実行し、推定アーキタイプに基づく
ボスの指令ターゲット優先順位を返す。

利用箇所:
  - main.py: 毎ターン update(obs) を呼んで情報を蓄積
  - target_policy.choose_switch: SWITCH 選択時に boss_target_ids() を参照
  - main_policy._pick_best_play: estimate_hand_quality() でイオナ撃ちどき判断
"""
from __future__ import annotations
from cg.api import Observation, LogType
from src.knowledge.meta_decks import estimate_deck_candidates, DECK_RECIPES

IONO_ID = 1198  # イオナ / アカマツ（手札干渉サポーター）


# アーキタイプ別のボスの指令優先ターゲット（先頭ほど優先）
# 戦略根拠は docs/strategy-knowledge.md [META-002] 参照
ARCH_BOSS_TARGETS: dict[str, list[int]] = {
    # Meganium(710) がエネ加速役 → まず止める
    "hydrapple_ex":        [710, 709, 708, 96],
    # Meganium(710) が同様にエネ加速役
    "olivia_ex":           [710, 709, 708, 402, 96],
    # ドラパルトライン: Dreepy(119)→Drakloak(120)→Dragapult ex(121)。Blaziken(326)もエネ加速役
    "dragapult_ex":        [119, 120, 326, 121],
    # タケルライコ: ベンチのオーガポンex(96)・Iron Hands(75)を引き出す
    "raging_bolt_ex":      [96, 75, 62],
    # バレット系: マシマシラ(112)はドロー役なので先に処理
    "ogerpon_bullet":      [112, 184, 108, 96],
    "terasta_bullet":      [272, 112, 184, 108, 96],
    # ミラー: リオル(677)→ルカリオ進化を妨害、ルナトーン(675)ドロー役
    "mega_lucario_ex":     [677, 675, 676, 673],
    # イワパレス: イシズマイ(344)進化前を処理、メガガルーラ(756)も標的
    "crustle":             [344, 756],
    # マリィのオーロンゲ: ベロバー(646)→ギモー(647)の進化ラインを崩す。ユキワラシ(103)も標的
    "maries_obstagoon_ex": [646, 647, 103, 104],
    # フーディン: ケーシィ(741)→ユンゲラー(742)の進化を妨害
    "alakazam":            [741, 742, 65, 66],
}

# アーキタイプ推定の最低信頼度閾値（この値未満は「不明」扱い）
MIN_CONFIDENCE = 0.3


class OpponentModel:
    """相手デッキ推定のシングルトン。ゲームをまたいでリセットする。"""

    def __init__(self) -> None:
        self._revealed: list[int] = []
        self._last_turn: int = -1
        self._cached_arch: str | None = None
        self._cached_turn: int = -1

    def update(self, obs: Observation) -> None:
        """観測データから相手の公開カード ID を蓄積する。ターンが戻ったら新試合とみなしリセット。"""
        state = obs.current
        if state is None:
            return

        turn = state.turn
        if turn < self._last_turn:
            self._revealed = []
            self._cached_arch = None
            self._cached_turn = -1
        self._last_turn = turn

        opp_idx = 1 - state.yourIndex
        opp = state.players[opp_idx]

        for poke in opp.active:
            if poke:
                self._revealed.append(poke.id)
                for c in poke.energyCards:
                    self._revealed.append(c.id)
                for c in poke.tools:
                    self._revealed.append(c.id)
                for c in poke.preEvolution:
                    self._revealed.append(c.id)
        for poke in opp.bench:
            self._revealed.append(poke.id)
            for c in poke.energyCards:
                self._revealed.append(c.id)
            for c in poke.tools:
                self._revealed.append(c.id)
            for c in poke.preEvolution:
                self._revealed.append(c.id)
        for card in opp.discard:
            self._revealed.append(card.id)

    def estimate_arch(self) -> str | None:
        """最も可能性の高いアーキタイプ名を返す。信頼度が低い場合は None。"""
        if self._cached_turn == self._last_turn:
            return self._cached_arch  # 同ターン内はキャッシュ再利用

        candidates = estimate_deck_candidates(self._revealed)
        if not candidates or candidates[0][1] < MIN_CONFIDENCE:
            self._cached_arch = None
        else:
            self._cached_arch = candidates[0][0]
        self._cached_turn = self._last_turn
        return self._cached_arch

    def boss_target_ids(self) -> list[int]:
        """ボスの指令で狙うべき相手ポケモン ID のリストを返す（先頭ほど優先）。

        不明なアーキタイプの場合は空リスト（呼び出し側でフォールバック）。
        """
        arch = self.estimate_arch()
        if arch and arch in ARCH_BOSS_TARGETS:
            return ARCH_BOSS_TARGETS[arch]
        return []

    def estimate_remaining(self, card_id: int) -> float:
        """相手デッキに指定カードが何枚残っているかを推定する（ARC-3）。

        meta_decks のレシピ枚数 - revealed での確認枚数 = 推定残り枚数。
        アーキタイプ不明の場合は保守的に 2.0 を返す（ほぼ残っていると仮定）。
        """
        arch = self.estimate_arch()
        if arch is None or arch not in DECK_RECIPES:
            return 2.0
        recipe_count = sum(cnt for cid, cnt in DECK_RECIPES[arch] if cid == card_id)
        revealed_count = self._revealed.count(card_id)
        return max(0.0, float(recipe_count - revealed_count))

    def estimate_hand_quality(self, obs: Observation) -> float:
        """相手の手札の「質」を推定する（ARC-5）。高いほど干渉が刺さりやすい。

        計算要素:
          - opp.handCount が多い → 干渉が効果的（+3）
          - opp.handCount が少ない → 干渉すると逆に引かせてしまう（-5）
          - 直前ターンに相手が 2 回以上カードをプレイ → 次ターンへの準備完了（+2）
        """
        state = obs.current
        if state is None:
            return 0.0

        opp_idx = 1 - state.yourIndex
        opp = state.players[opp_idx]
        score = 0.0

        hand_count = opp.handCount
        if hand_count >= 5:
            score += 3.0
        elif hand_count <= 2:
            score -= 5.0
        elif hand_count >= 4:
            score += 1.0

        # 直前ターンの相手のカードプレイ数をカウント（準備完了の指標）
        opp_plays = sum(
            1 for log in obs.logs
            if log.type == LogType.PLAY and log.playerIndex == opp_idx
        )
        if opp_plays >= 2:
            score += 2.0
        elif opp_plays >= 1:
            score += 0.5

        return score


_model = OpponentModel()


def get_opponent_model() -> OpponentModel:
    return _model
