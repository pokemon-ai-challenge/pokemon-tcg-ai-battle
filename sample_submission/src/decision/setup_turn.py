from cg.api import Observation

from src.decision.fallback import choose_random_legal_action


def choose_setup_action(obs: Observation) -> list[int]:
    """Handle setup-phase selections such as active and bench placement."""
    # TODO:
    # - choose the best opening Basic for the active spot
    # - choose bench Basics with early-game stability in mind
    # - avoid exposing fragile or low-value cards too early
    return choose_random_legal_action(obs)
