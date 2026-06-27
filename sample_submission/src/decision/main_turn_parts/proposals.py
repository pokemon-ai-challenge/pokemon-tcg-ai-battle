from dataclasses import dataclass


@dataclass
class MainActionProposal:
    """メインフェーズで取りたい行動候補。"""

    action: list[int]
    score: int
    label: str


def choose_best_proposal(
    proposals: list[MainActionProposal | None],
) -> MainActionProposal | None:
    """None を除いて、もっとも重みの高い候補を返す。"""
    valid_proposals = [proposal for proposal in proposals if proposal is not None]
    if not valid_proposals:
        return None
    return max(valid_proposals, key=lambda proposal: proposal.score)
