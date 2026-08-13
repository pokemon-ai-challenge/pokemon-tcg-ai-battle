"""``eval_bulu_loop.py`` と ``collect_ogerpon_counterfactuals.py`` が共有する、カプ・ブルル
中継戦略の判定ロジック(design.md 5.4)。2ファイルで定義が食い違わないよう、ここへ集約する
(外部レビュー指摘: collectorとevaluationで定義が食い違わないよう共有helperを検討する)。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def is_emergency_retreat_allowed(active, opp_active, config, P) -> bool:
    """design.md §5.4: ACTIVE中の交代は「確定リーサル・攻撃不能・確定敗北回避」だけ許可する。

    ``search_step`` による仮実行(確定リーサル・確定敗北回避の厳密な確認)はこの関数の
    対象外(Phase2/Phase4境界の簡略化。静的計算だけで判定できる「攻撃不能」「確実に
    今ターン中に倒される(CERTAIN_KO)」の2条件で近似する)。判定できない(``active`` が
    None)場合は安全側でTrue(緊急扱い)を返す。

    Args:
        active: 現在ACTIVEにいる対象(通常はカプ・ブルル)。
        opp_active: 相手のバトル場ポケモン。
        config: ``ogerpon_planner.ko_risk`` に渡すconfig(``main_config`` 経由で解決)。
        P: ``ptcg_ai.ml_policy.ogerpon_planner`` モジュール(呼び出し側が渡す。循環import回避)。
    """
    if active is None:
        return True
    if not P.can_attack_now(active):
        return True
    cfg = P.main_config(config)
    risk, _ = P.ko_risk(active, opp_active, cfg)
    return risk == P.CERTAIN_KO


def detect_vanished(target_serial, player) -> bool:
    """``target_serial`` が ``player``(``PlayerState``)の場(active+bench)のどこにも
    見つからなければTrue(気絶または捕捉していない離脱とみなす)。

    ブルルの攻撃直後に相手個体が消えたか(KO判定)、相手の攻撃直後に自分の個体が
    消えたか(被弾KO判定)の両方に使う汎用ヘルパー。進化で同じserialのまま姿を変える
    ケースは未対応(見つからない=消滅として扱う近似)。
    """
    if target_serial is None:
        return False
    for slot in list(player.active or []) + list(player.bench or []):
        if slot is not None and getattr(slot, "serial", None) == target_serial:
            return False
    return True
