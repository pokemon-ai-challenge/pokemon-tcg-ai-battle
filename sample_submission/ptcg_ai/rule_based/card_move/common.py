"""クラスタ④ カード移動・対象選択（共通ヘルパー）／担当B

decision.card_move 配下の各モジュールが共有するヘルパー。
Option 配列を area / playerIndex / type などでフィルタする汎用処理のみを置く。
カード名・カードIDのハードコードはしない（knowledge.profile_registry 経由で参照する）。
"""

from cg.api import AreaType, Option


def filter_by_area(options: list[Option], area: AreaType) -> list[Option]:
    """指定した AreaType の Option だけを抽出する。"""
    raise NotImplementedError


def filter_own(options: list[Option], your_index: int) -> list[Option]:
    """playerIndex が自分のものである Option だけを抽出する。"""
    raise NotImplementedError
