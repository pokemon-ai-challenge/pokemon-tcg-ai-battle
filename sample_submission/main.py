import os

from cg.api import Observation, to_observation_class


def read_deck_csv() -> list[int]:
    """Read deck.csv.
    
    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


def choose_minimum_required(obs: Observation) -> list[int]:
    """必要最低限の数を，候補の先頭から選ぶ。"""
    select = obs.select
    return list(range(select.minCount))


def validate_choice(obs: Observation, chosen: list[int]) -> None:
    """返すインデックスが選択ルールを満たしているか確認する。"""
    select = obs.select
    # 選択個数の確認
    if not (select.minCount <= len(chosen) <= select.maxCount):
        raise ValueError(
            f"Action length must be between {select.minCount} and {select.maxCount}, got {len(chosen)}."
        )

    # 同じインデックスを重複して選択していないか確認
    if len(chosen) != len(set(chosen)):
        raise ValueError("Duplicate select elements are not allowed.")

    if not all(0 <= i < len(select.option) for i in chosen):
        raise ValueError("Each selected index must be within the option range.")


def choose_main_action(obs: Observation) -> list[int]:
    """メインフェーズでの基本的な行動選択を行う。"""
    select = obs.select
    state = obs.current

    # 各行動タイプの候補番号を保存する。
    attach_options = []
    evolve_options = []
    ability_options = []
    attack_options = []
    end_options = []

    # 行動の優先順位を決めやすいよう、候補を種類ごとに分ける
    for i, option in enumerate(select.option):
        if option.type == 8:
            attach_options.append(i)
        elif option.type == 9:
            evolve_options.append(i)
        elif option.type == 10:
            ability_options.append(i)
        elif option.type == 13:
            attack_options.append(i)
        elif option.type == 14:
            end_options.append(i)

    if not state.energyAttached and attach_options:
        return [attach_options[0]]

    if evolve_options:
        return [evolve_options[0]]

    if ability_options:
        return [ability_options[0]]

    if attack_options:
        return [attack_options[0]]

    if end_options:
        return [end_options[0]]

    return choose_minimum_required(obs)


def choose_action(obs: Observation) -> list[int]:
    """状況に応じて行動選択処理を振り分ける。"""
    select = obs.select

    # メインフェーズでは専用の行動選択ロジックを使う
    if select.context == 0:
        chosen = choose_main_action(obs)
    else:
        # TODO:
        # ほかの場合分けの追加も行う

        # それ以外の選択では、まず必要最低限の合法手を探す
        chosen = choose_minimum_required(obs)

    validate_choice(obs, chosen)
    return chosen


def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return read_deck_csv()

    return choose_action(obs)
