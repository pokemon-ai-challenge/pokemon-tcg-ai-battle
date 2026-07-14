"""クラスタ⑨ 検証（共通基盤）／担当B

ptcg_ai/action_selection・ptcg_ai/rule_based 配下のテストが使う pytest fixture。実物の cg エンジンを起動せずに、
最小限の Observation/State/SelectData を組み立てて渡せるようにする。
"""

import pytest

from cg.api import Observation


@pytest.fixture
def empty_observation() -> Observation:
    """select・current ともに最小構成の Observation を返す（各テストで上書きして使う）。"""
    raise NotImplementedError
