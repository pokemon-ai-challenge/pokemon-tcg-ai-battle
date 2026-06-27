from cg.api import Observation, SelectContext

from src.decision.attack_turn import choose_attack_action
from src.decision.fallback import choose_random_legal_action
from src.decision.main_turn import choose_main_action
from src.decision.setup_turn import choose_setup_action


def choose_action(obs: Observation) -> list[int]:
    """通常ターンの処理をここから呼び分ける。"""
    if obs.select is None:
        raise ValueError("obs.select must not be None during turn decisions.")

    # 何を選ぶ場面かを先に確認する
    context = obs.select.context

    # メインフェーズの行動を決める
    if context == SelectContext.MAIN:
        return choose_main_action(obs)

    # セットアップ時の行動を決める
    if context in {
        SelectContext.SETUP_ACTIVE_POKEMON,
        SelectContext.SETUP_BENCH_POKEMON,
    }:
        return choose_setup_action(obs)

    # ワザ選択まわりの行動を決める
    if context == SelectContext.ATTACK:
        return choose_attack_action(obs)

    return choose_random_legal_action(obs)
