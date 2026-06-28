from cg.api import Observation

from src.decision.main_turn_parts.energy_eval import choose_best_attach_option
from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS


def propose_energy_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """このターンで最もよいエネルギー貼りを候補として返す。"""
    if obs.current is None:
        return None
    if obs.current.energyAttached or not buckets.attach:
        return None

    # 貼り先の比較結果を見て、そもそも MAIN 候補として出してよいかを決める。
    best_attach = choose_best_attach_option(obs, buckets.attach)
    if not best_attach.should_offer:
        return None

    # 他カテゴリとの比較は、貼り先スコアではなくカテゴリ固定の重みで行う。
    return MainActionProposal(
        action=[best_attach.option_index],
        score=MAIN_ACTION_BASE_WEIGHTS["energy"],
        label="energy",
    )
