"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: AFFECT_SPECIAL_CONDITION, RECOVER_SPECIAL_CONDITION

どく・やけど・ねむり・まひ・こんらんなどの状態異常を、誰に付与するか/誰から回復するかを
選ぶ場面（SelectType.SPECIAL_CONDITION）。
"""

from cg.api import Observation


def handle(obs: Observation) -> list[int]:
    """AFFECT_SPECIAL_CONDITION / RECOVER_SPECIAL_CONDITION の選択肢を処理する。"""
    raise NotImplementedError
