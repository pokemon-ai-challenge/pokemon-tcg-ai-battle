from cg.api import Observation, SelectContext, all_card_data

from src.decision.fallback import choose_random_legal_action


_CARD_DATA = {c.cardId: c for c in all_card_data()}


        # ==========================================================
        # TODO: 初期配置の評価を改善する
        #
        # 今後追加したい評価項目
        # - HPが高いポケモンを優先する
        # - ワザがすぐ使えるポケモンを優先する
        # - 逃げエネルギーが少ないポケモンを優先する
        # - 手札のエネルギーとの相性を考慮する
        # - 先攻・後攻で評価を変更する
        # - ベンチとのシナジー（Solrock・Lunatone等）を追加する
        # - 将来的にはカードごとの評価値を辞書などで管理する
        # ==========================================================


def choose_setup_action(obs: Observation) -> list[int]:
    """初期配置でバトル場・ベンチに出すポケモンを選択する。"""

    if obs.current is None or obs.select is None:
        return choose_random_legal_action(obs)

    your_index = obs.current.yourIndex
    hand = obs.current.players[your_index].hand
    if hand is None:
        return choose_random_legal_action(obs)

    # 手札にあるカード名一覧
    hand_names = []
    for card in hand:
        card_data = _CARD_DATA.get(card.id)
        if card_data:
            hand_names.append(card_data.name)

    has_lunatone = "Lunatone" in hand_names
    has_solrock = "Solrock" in hand_names
    has_lucario = "Lucario" in hand_names

    best_option = None
    best_score = -9999

    for option_index, option in enumerate(obs.select.option):
        if option.index is None:
            continue

        hand_card = hand[option.index]
        card_data = _CARD_DATA.get(hand_card.id)
        if card_data is None:
            continue

        score = 0
        name = card_data.name

        # -------------------------
        # バトル場
        # -------------------------
        if obs.select.context == SelectContext.SETUP_ACTIVE_POKEMON:

            # Riolu + Lucario なら最優先
            if name == "Riolu" and has_lucario:
                score += 100

            # Solrock + Lunatone なら Solrock 優先
            elif name == "Solrock" and has_lunatone:
                score += 90

            # 通常優先度
            elif name == "Riolu":
                score += 80

            elif name == "Solrock":
                score += 70

            elif name == "Lunatone":
                score += 60

            elif name == "Makuhita":
                score -= 100

        # -------------------------
        # ベンチ
        # -------------------------
        elif obs.select.context == SelectContext.SETUP_BENCH_POKEMON:

            if has_solrock and name == "Lunatone":
                score += 100

            elif has_lunatone and name == "Solrock":
                score += 100

            elif name == "Riolu":
                score += 80

            elif name == "Makuhita":
                score -= 100

        if score > best_score:
            best_score = score
            best_option = option_index

    if best_option is None:
        return choose_random_legal_action(obs)

    return [best_option]