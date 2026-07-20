"""模倣ポリシー Agent の入口(Step2、イシュー未起票、ml-value-network の後続)。

`core.agent` から `AGENT_TYPE == "ml_policy"` のときに呼ばれる Agent 実装。
`ptcg_ai.learning.policy_model.PolicyModel`(選択肢スコアリング)で通常ターンの選択肢を
選ぶ。`rule_based/` `action_selection/` は一切変更・呼び出しせず独立に完結させる
(`sample_submission/docs/plans/ml-value-network/step2-design.md` §1 の接続点方針)。

デッキ選択(初回)は現行 `deck.csv`(フーディン)をそのまま使う。`rule_based_agent.py` の
`read_deck_csv()` は deck.csv を読むだけの共通ユーティリティ(rule_based 自体のゲーム判断
ロジックではない)であり、`main.py` 自身もここから re-export して使っている安定した部品
なので、そのまま再利用する(コード重複を避ける)。

選択肢が複数選択(`maxCount > 1`)の意思決定点は Step2 の学習スコープ外
(step2-design.md §2.4)。スコアの高い選択肢から `minCount`〜`maxCount` 件を独立に選ぶ
貪欲フォールバックで対応する。
"""

from cg.api import Observation, SelectData
from ptcg_ai.learning.policy_model import PolicyModel
from ptcg_ai.rule_based.rule_based_agent import read_deck_csv

_model: PolicyModel | None = None


def agent(obs: Observation) -> list[int]:
    """obs を見てデッキ返却 or 選択肢スコアリングによる選択を行う。

    Args:
        obs: core.agent から渡される Observation。

    Returns:
        list[int]: 初回はデッキの60枚のカードIDリスト。通常ターンは選択肢インデックスのリスト。
    """
    if obs.select is None:
        return read_deck_csv()
    return _select_action(obs)


def _get_model() -> PolicyModel:
    global _model
    if _model is None:
        _model = PolicyModel()
    return _model


def _select_action(obs: Observation) -> list[int]:
    select = obs.select
    model = _get_model()

    if select.maxCount == 1:
        idx = model.select_option(obs)
        return [idx if idx is not None else 0]

    return _greedy_multi_select(obs, model, select)


def _greedy_multi_select(obs: Observation, model: PolicyModel, select: SelectData) -> list[int]:
    """maxCount > 1(Step2 学習スコープ外)向けの貪欲フォールバック。

    各選択肢を独立にスコアリングし、上位から minCount〜maxCount 件を選ぶ。組み合わせの
    最適性は保証しない(step2-design.md §2.4 の既知の制約)。
    """
    n = len(select.option)
    count = max(select.minCount, min(select.maxCount, n))

    scores = model.score_options(obs)
    if not scores:
        return list(range(count))

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return ranked[:count]
