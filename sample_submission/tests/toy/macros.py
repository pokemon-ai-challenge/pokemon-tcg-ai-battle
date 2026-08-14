"""遅延生成マクロ層(Step 1-2f)。

設計原則(姉妹文書 lethal-macro-search §3.1 / 原設計 §6.1)をそのまま守る:

- **目的指向**: 打点差・必要札から候補を並べる
- **状態依存**: 順位は毎回その状態から計算する(固定優先順位ではない)
- **遅延生成**: その状態で1手分の候補を作るだけ。行動列は事前生成しない
- **新情報境界で終了**: ドローは1手で終わり、結果は探索側が outcome として展開する
- **枝刈りしない**: 順序を変えるだけで、合法手は全部残す

``lossy_macro_actions`` は「もっともらしいが健全でない枝刈り」を入れた版で、
マクロ化で勝ち筋を落としていないかを検出できるかの確認に使う。
"""

from __future__ import annotations

from tests.toy.toy_game import BACKEND, BOOST, DRAW2, END, STRIKE, STRIKE_DAMAGE, ToyState


def _goal(state: ToyState) -> dict:
    """状態からその場で導く目的分析(カード名のハードコードはするが toy なので可)。"""
    hand = list(state.hand)
    strikes = hand.count(STRIKE)
    boosts = hand.count(BOOST)
    damage_now = strikes * (STRIKE_DAMAGE + state.bonus)
    return {
        "gap": state.opponent_hp - damage_now,
        "strikes": strikes,
        "boosts": boosts,
        "can_kill_now": damage_now >= state.opponent_hp and strikes > 0,
        "actions_left": state.actions_left,
    }


def macro_actions(state: ToyState) -> tuple:
    """目的に沿って**並べ替えた**全合法手(枝刈りなし)。"""
    goal = _goal(state)
    actions = list(BACKEND.legal_actions(state))

    def rank(action) -> tuple:
        if action == END:
            return (9, 0)
        card_id = action[1]
        if goal["can_kill_now"]:
            # 今の打点で倒せるなら攻撃が最優先。
            return ({STRIKE: 0, BOOST: 2, DRAW2: 3}.get(card_id, 4), card_id)
        if goal["gap"] > 0 and goal["boosts"] > 0 and goal["strikes"] > 0:
            # 打点不足で、ブーストと攻撃札が揃っている: ブーストが先。
            return ({BOOST: 0, STRIKE: 1, DRAW2: 2}.get(card_id, 4), card_id)
        # 打点も札も足りない: まず引く。
        return ({DRAW2: 0, BOOST: 1, STRIKE: 2}.get(card_id, 4), card_id)

    return tuple(sorted(actions, key=rank))


def lossy_macro_actions(state: ToyState) -> tuple:
    """壊した版: 「攻撃札があるならブーストは無駄」という健全でない枝刈り。"""
    actions = [
        action
        for action in macro_actions(state)
        if not (action != END and action[1] == BOOST and STRIKE in state.hand)
    ]
    return tuple(actions)
