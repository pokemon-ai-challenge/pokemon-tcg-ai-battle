from cg.api import Attack, CardType, Observation, all_attack

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS
from src.knowledge.card_cache import load_card_data

# ---------------------------------------------------------------------------
# Skill.text のキーワードでサポーター・グッズの種別を判定する。
# ---------------------------------------------------------------------------

_DRAW_SCORE_BONUS = {
    "emergency": 30,   # 0〜2枚: 緊急 → 最優先でドロー
    "normal":    15,   # 3〜4枚: 普通 → 積極的に使う
    "enough":     0,   # 5〜6枚: 十分 → 使えるなら使う程度
    # 7枚以上はドロー目的では発動しない
}

_SUPPORTER_TYPE_BONUS = {
    "draw":                10,   # 純粋ドロー（シロナ・ホップ等）
    "search":               5,   # サーチ系（ネジキ・ポフィン等）
    "discard_draw":         0,   # 手札捨て/戻しドロー（博士の研究・マリィ等）
    "draw_to_x":            0,   # 手札をX枚まで引く（ポケモン研究者等）
    "discard_draw_penalty": -40, # 先に使えるカードがある場合の減点（discard_draw / draw_to_x 共通）
    "other":                0,
}

# 進化ペアが揃っているときの追加ペナルティ（手札から進化後を失うリスク）
_EVOLUTION_PAIR_DISCARD_PENALTY = -20

# ==============================
# エネルギー不足時のサポート優先加点
# ==============================
# 盤面の最高打点ポケモンがエネ不足のとき、サーチ系サポートのスコアを上乗せする量
_ENERGY_SEARCH_BONUS = 25

# ==============================
# 山札切れ防止ガード
# ==============================
# 自分の山札残り枚数がこの値以下なら、ドロー・サーチ系サポーターを一切使わない
_DECK_COUNT_SAFE_THRESHOLD = 10


def _classify_supporter_text(text: str) -> str:
    """サポーターの効果テキストから種別を返す。

    Returns:
        "discard_draw" : 手札を捨てて/山に戻してドロー（博士の研究・マリィ等）
        "draw_to_x"    : 手札が X 枚になるまでドロー（ポケモン研究者等）
        "draw"         : 純粋なドロー（シロナ・ホップ等）
        "search"       : 山札からサーチ（ネジキ・ポフィン等）
        "other"        : 上記以外（ボスの指令等）
    """
    t = text.lower()
    has_draw    = "draw" in t
    has_discard = "discard" in t or "shuffle" in t
    has_search  = "search" in t or "look at" in t

    has_draw_to_x = (
        ("until you have" in t or "so that you have" in t or "up to" in t)
        and "hand" in t
        and has_draw
    )

    if has_draw and has_discard:
        return "discard_draw"
    if has_draw_to_x:
        return "draw_to_x"
    if has_draw:
        return "draw"
    if has_search:
        return "search"
    return "other"


def _classify_item_text(text: str) -> str:
    """グッズの効果テキストから種別を返す。"""
    t = text.lower()
    has_draw = "draw" in t
    has_draw_to_x = (
        ("until you have" in t or "so that you have" in t or "up to" in t)
        and "hand" in t
        and has_draw
    )
    if "search" in t or "look at" in t:
        return "search"
    if has_draw_to_x:
        return "draw_to_x"
    if has_draw:
        return "draw"
    return "other"


def _get_card_skill_text(option_index: int, obs: Observation) -> str | None:
    """手札の option_index 番目のカードの最初の Skill テキストを返す。"""
    if obs.current is None or obs.select is None:
        return None

    option = obs.select.option[option_index]
    your_index = obs.current.yourIndex
    hand = obs.current.players[your_index].hand
    if hand is None or option.index is None or option.index >= len(hand):
        return None

    hand_card = hand[option.index]
    card_data_by_id = {card.cardId: card for card in load_card_data()}
    card_data = card_data_by_id.get(hand_card.id)
    if card_data is None or not card_data.skills:
        return None

    return card_data.skills[0].text


def _get_supporter_kind(option_index: int, obs: Observation) -> str:
    text = _get_card_skill_text(option_index, obs)
    if text is None:
        return "other"
    return _classify_supporter_text(text)


def _get_item_kind(option_index: int, obs: Observation) -> str:
    text = _get_card_skill_text(option_index, obs)
    if text is None:
        return "other"
    return _classify_item_text(text)


def _hand_bonus(hand_count: int) -> int | None:
    """手札枚数に応じた加点を返す。7枚以上は None でドロー不要を示す。"""
    if hand_count <= 2:
        return _DRAW_SCORE_BONUS["emergency"]
    if hand_count <= 4:
        return _DRAW_SCORE_BONUS["normal"]
    if hand_count <= 6:
        return _DRAW_SCORE_BONUS["enough"]
    return None


def _has_usable_non_draw_cards(obs: Observation, buckets: MainOptionBuckets) -> bool:
    """手札に「先に使うべきカード」があるか。"""
    has_non_draw_items = False
    for idx in buckets.item_play:
        kind = _get_item_kind(idx, obs)
        if kind not in ("draw", "draw_to_x", "search"):
            has_non_draw_items = True
            break

    return bool(
        buckets.pokemon_play
        or buckets.evolve
        or has_non_draw_items
        or buckets.tool_play
        or buckets.stadium_play
        or buckets.ability
        or buckets.attach
    )


def _deck_count_too_low(obs: Observation) -> bool:
    """自分の山札残り枚数が安全閾値以下かを判定する。

    山札が0枚の状態でドローフェーズを迎えると敗北になるため、
    残り枚数が少ないときはドロー・サーチ系サポーターの使用を避ける。
    エネルギー不足によるサーチ優先（needs_energy）よりもこのガードを優先する。
    """
    if obs.current is None:
        return False
    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    deck_count = getattr(player, "deckCount", None)
    if deck_count is None:
        return False
    return deck_count <= _DECK_COUNT_SAFE_THRESHOLD


def _count_evolution_pairs_on_bench(obs: Observation) -> int:
    """ベンチにいる進化前ポケモンに対して、手札に進化後カードが何枚あるかを返す。"""
    if obs.current is None:
        return 0

    card_data_list = load_card_data()
    card_data_by_id = {c.cardId: c for c in card_data_list}

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    hand = player.hand
    if hand is None:
        return 0

    bench_ids = {poke.id for poke in player.bench if poke is not None}
    if not bench_ids:
        return 0

    pair_count = 0
    for hand_card in hand:
        card = card_data_by_id.get(hand_card.id)
        if card is None:
            continue
        if hasattr(card, 'evolvesFrom') and card.evolvesFrom:
            for bench_id in bench_ids:
                bench_card = card_data_by_id.get(bench_id)
                if bench_card is None:
                    continue
                if bench_card.name and card.evolvesFrom.lower() in bench_card.name.lower():
                    pair_count += 1
                    break

    return pair_count


# ---------------------------------------------------------------------------
# エネルギー不足チェック（盤面の最高打点ポケモンがエネ不足かを判定）
# ---------------------------------------------------------------------------

def _count_hand_energies(obs: Observation) -> int:
    """手札にある基本エネルギーカードの枚数を返す。"""
    if obs.current is None:
        return 0
    your_index = obs.current.yourIndex
    hand = obs.current.players[your_index].hand
    if hand is None:
        return 0

    card_data_list = load_card_data()
    card_data_by_id = {c.cardId: c for c in card_data_list}

    count = 0
    for hand_card in hand:
        card = card_data_by_id.get(hand_card.id)
        if card is None:
            continue
        if card.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            count += 1
    return count


def _has_draw_supporter_available(buckets: MainOptionBuckets, obs: Observation) -> bool:
    """手札にドロー系サポーターがあるか（draw / search 種別）。"""
    for idx in buckets.supporter_play:
        kind = _get_supporter_kind(idx, obs)
        if kind in ("draw", "search"):
            return True
    return False


def _max_attack_damage_any_pokemon(obs: Observation) -> int:
    """自分の盤面（active + bench）のポケモンが現在のエネルギーで出せる最大打点を返す。

    この関数は board.py の _max_damage_of_pokemon と同じロジックだが、
    draw.py は board.py に依存しないためここに独立して実装する。
    """
    if obs.current is None:
        return 0

    card_data_list = load_card_data()
    card_data_by_id = {c.cardId: c for c in card_data_list}

    # attackId → damage の簡易マップ（cg.api の all_attack を使わず card_data から推定）
    # NOTE: 攻撃データは board.py 側の _get_attack_data() が持つが、
    #       draw.py では import を避けるため CardData.attacks が持つ damage を使う。
    #       CardData.attacks がリストの場合、各要素が attackId なので damage は取れない。
    #       ここでは「エネルギーが揃っているかどうか」だけを判定するため、
    #       ポケモンのエネルギー枚数と各技の要求エネ枚数の合計を比較する簡易版にする。

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]

    # active と bench をまとめて見る
    all_pokemon = []
    if player.active:
        for p in player.active:
            if p is not None:
                all_pokemon.append(p)
    all_pokemon.extend(p for p in player.bench if p is not None)

    max_dmg = 0
    for poke in all_pokemon:
        card = card_data_by_id.get(poke.id)
        if card is None:
            continue
        # エネルギー枚数（タイプ問わず合計）
        attached = len(poke.energies) if hasattr(poke, 'energies') and poke.energies else 0
        # カードが持つ技の中で、今のエネで使えそうな技の打点を取る
        # （詳細チェックは board.py に任せ、ここでは attached >= 技のエネ要求合計 で判定）
        for atk in (card.attacks or []):
            # atk が attackId（int）の場合、打点は取れないのでスキップ
            # atk が Attack オブジェクトの場合
            if hasattr(atk, 'damage') and hasattr(atk, 'energies'):
                req = len(atk.energies) if atk.energies else 0
                if attached >= req:
                    max_dmg = max(max_dmg, atk.damage)
    return max_dmg


def _best_pokemon_needs_energy(obs: Observation) -> bool:
    """盤面の最高打点ポケモンが、あと1枚以上エネルギーを必要としているか。

    判定ロジック:
    - active + bench の全ポケモンについて「現在エネで出せる最大打点」を取る
    - いずれかのポケモンで「エネを1枚追加すると使える技がある（かつ今は使えない）」なら True

    簡易版: 盤面に attach 候補（buckets.attach）があれば、
    エネを貼りたいポケモンがいる = まだエネが足りていないとみなす。
    """
    if obs.current is None:
        return False
    # attach 候補があるということは、まだエネを貼りたい状況
    # （energy_eval.py がスコア0以下と判定した場合は buckets.attach に入らないため
    #   ここでは buckets を参照しない。代わりに直接盤面を確認する）

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]

    card_data_list = load_card_data()
    card_data_by_id = {c.cardId: c for c in card_data_list}

    all_pokemon = []
    if player.active:
        for p in player.active:
            if p is not None:
                all_pokemon.append(p)
    all_pokemon.extend(p for p in player.bench if p is not None)

    for poke in all_pokemon:
        card = card_data_by_id.get(poke.id)
        if card is None:
            continue
        attached = len(poke.energies) if hasattr(poke, 'energies') and poke.energies else 0
        for atk in (card.attacks or []):
            if hasattr(atk, 'damage') and hasattr(atk, 'energies'):
                req = len(atk.energies) if atk.energies else 0
                # 今は使えないが、エネを1枚追加すれば使える技がある
                if req > attached and req <= attached + 1:
                    return True
    return False


# discard_draw / draw_to_x の両方に減点を適用するカテゴリ集合
_LOSS_ON_HAND_KINDS: frozenset[str] = frozenset({"discard_draw", "draw_to_x"})


# ---------------------------------------------------------------------------
# 「捨てても良い」例外条件（エネルギー枯渇ペナルティを解除する3条件）
# ---------------------------------------------------------------------------

_attack_cache: dict[int, Attack] | None = None


def _get_attack_data() -> dict[int, Attack]:
    global _attack_cache
    if _attack_cache is None:
        _attack_cache = {a.attackId: a for a in all_attack()}
    return _attack_cache


def _max_damage_with_energy_check(pokemon_id: int, energies: list, card_data_by_id: dict) -> int:
    """ポケモンが現在のエネルギーで出せる最大ダメージを返す（エネ充足チェック込み）。"""
    attack_data_by_id = _get_attack_data()
    card = card_data_by_id.get(pokemon_id)
    if card is None:
        return 0

    energy_counts: dict[int, int] = {}
    for e in energies:
        energy_counts[int(e)] = energy_counts.get(int(e), 0) + 1

    max_dmg = 0
    for attack_id in card.attacks:
        attack = attack_data_by_id.get(attack_id)
        if attack is None:
            continue
        required: dict[int, int] = {}
        colorless_needed = 0
        for e in attack.energies:
            if int(e) == 0:
                colorless_needed += 1
            else:
                required[int(e)] = required.get(int(e), 0) + 1

        can_use = True
        remaining = dict(energy_counts)
        for e_type, count in required.items():
            have = remaining.get(e_type, 0)
            if have < count:
                can_use = False
                break
            remaining[e_type] = have - count

        if can_use:
            total_remaining = sum(remaining.values())
            if total_remaining >= colorless_needed:
                max_dmg = max(max_dmg, attack.damage)

    return max_dmg


def _active_already_has_max_attack_energy(obs: Observation) -> bool:
    """例外1: 主力アタッカー（active）が既に最大火力の技を打てる状態か。"""
    if obs.current is None:
        return False
    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    if not player.active or player.active[0] is None:
        return False

    active = player.active[0]
    card_data_by_id = {c.cardId: c for c in load_card_data()}
    attack_data_by_id = _get_attack_data()

    card = card_data_by_id.get(active.id)
    if card is None or not card.attacks:
        return False

    attached = len(active.energies) if active.energies else 0

    max_required = 0
    for attack_id in card.attacks:
        attack = attack_data_by_id.get(attack_id)
        if attack is None:
            continue
        max_required = max(max_required, len(attack.energies) if attack.energies else 0)

    return attached >= max_required


def _energy_already_attached_this_turn(obs: Observation) -> bool:
    """例外2: このターン既にエネルギーを付け終わっているか。"""
    if obs.current is None:
        return False
    return bool(getattr(obs.current, "energyAttached", False))


def _can_ko_opponent_active_now(obs: Observation) -> bool:
    """例外3: 場のポケモン（active）が今すぐ攻撃すれば相手activeをKOできるか。"""
    if obs.current is None:
        return False
    your_index = obs.current.yourIndex
    opp_index = 1 - your_index
    player = obs.current.players[your_index]
    opp_player = obs.current.players[opp_index]

    if not player.active or player.active[0] is None:
        return False
    if not opp_player.active or opp_player.active[0] is None:
        return False

    active = player.active[0]
    opp_active = opp_player.active[0]

    card_data_by_id = {c.cardId: c for c in load_card_data()}
    my_max_dmg = _max_damage_with_energy_check(active.id, active.energies, card_data_by_id)

    opp_hp = getattr(opp_active, "remainingHp", None)
    if opp_hp is None:
        opp_hp = getattr(opp_active, "hp", None)
    if opp_hp is None:
        return False

    return my_max_dmg >= opp_hp


def _discard_draw_penalty_should_be_waived(obs: Observation) -> bool:
    """捨てドロー型サポートのエネルギー関連ペナルティを解除すべきかを判定する。

    例外1〜3のいずれか1つでも True なら、エネルギーを気にせず捨ててよい:
      例外1: 場のエネルギーが足りている（主力が既に最大技を打てる）
      例外2: このターンもうエネルギーを付けられない（energyAttached 済み）
      例外3: 場のポケモンが攻撃で相手を倒せる（今すぐKO可能）
    """
    if _active_already_has_max_attack_energy(obs):
        return True
    if _energy_already_attached_this_turn(obs):
        return True
    if _can_ko_opponent_active_now(obs):
        return True
    return False


def propose_draw_or_search_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """手札状況に応じて、ドロー・サーチ系カードを優先候補として返す。

    優先順位（スコアが高い順）:
        1. 純粋ドロー系サポーター（シロナ・ホップ等）
        2. サーチ系サポーター（ネジキ・ポフィン等）
           ★ 盤面のポケモンが最高打点を出すためにエネが不足している場合は
              手札枚数に関わらずサーチ系に _ENERGY_SEARCH_BONUS (+25) を加算する
        3. 手札捨て/戻し系・X枚までドロー系サポーター
           → 先に使えるカードがある場合はスコアを下げる
           → 進化ペアがある場合はさらにスコアを下げる
           ★ ただし以下いずれかに該当する場合は「先に使えるカードがある」ペナルティを解除する
              例外1: 場のエネルギーが足りている（主力が既に最大技を打てる）
              例外2: このターンもうエネルギーを付けられない
              例外3: 場のポケモンが攻撃で相手を倒せる（今すぐKO可能）
        4. ドロー・サーチ系グッズ

    手札 7 枚以上の場合は提案しない。
    ただしエネ不足補正の結果スコアが 0 以上になった場合は提案する
    （手札枚数制限を 「エネ不足なら免除」として扱う）。

    ★ 山札切れガード（最優先）:
        自分の山札残り枚数が 10 枚以下のときは、手札枚数やエネ不足の状況に
        かかわらずドロー・サーチ系サポーターを一切提案しない。
        山札が0枚でドローフェーズを迎えると敗北になるため。
    """
    if obs.current is None:
        return None

    # 山札切れガード: 残り枚数が少ないときは最優先でドロー系を提案しない
    if _deck_count_too_low(obs):
        return None

    your_index = obs.current.yourIndex
    hand_count = obs.current.players[your_index].handCount

    hand_bonus = _hand_bonus(hand_count)

    # エネルギー不足フラグ（手札枚数に関係なくサーチを優先するため事前計算）
    needs_energy = _best_pokemon_needs_energy(obs)

    # 手札7枚以上でもエネ不足ならサーチ系だけは提案候補に残す
    if hand_bonus is None and not needs_energy:
        return None

    # hand_bonus が None（手札7枚以上）でもエネ不足なら 0 で計算を続ける
    effective_hand_bonus = hand_bonus if hand_bonus is not None else 0

    base = MAIN_ACTION_BASE_WEIGHTS["draw_or_search"]

    KIND_PRIORITY = {"draw": 3, "search": 2, "discard_draw": 1, "draw_to_x": 1, "other": 0}

    best_supporter_index: int | None = None
    best_supporter_kind: str = "other"
    best_priority: int = 0

    for idx in buckets.supporter_play:
        kind = _get_supporter_kind(idx, obs)
        priority = KIND_PRIORITY.get(kind, 0)
        if priority > best_priority:
            best_supporter_index = idx
            best_supporter_kind = kind
            best_priority = priority
            if kind == "draw":
                break

    if best_supporter_index is not None and best_supporter_kind != "other":
        type_bonus = _SUPPORTER_TYPE_BONUS.get(best_supporter_kind, 0)

        # ★ エネルギー補充目的のサーチ系加点
        # 盤面のポケモンが最高打点を出すためにエネが足りないとき、
        # サーチ系サポート（山札からカードを持ってこられる）のスコアを上乗せする。
        # draw 系はランダムドローなのでエネが引ける保証がなく加点しない。
        # discard_draw 系は手札を一度捨てるのでエネを失うリスクがあり加点しない。
        if needs_energy and best_supporter_kind == "search":
            type_bonus += _ENERGY_SEARCH_BONUS

        if best_supporter_kind in _LOSS_ON_HAND_KINDS:
            # 例外1〜3のいずれかに該当するなら、エネルギー枯渇関連のペナルティのみ解除する
            # （進化ペアを失うリスクはエネルギーの話とは別問題なので、例外の対象にしない）
            waive_energy_penalty = _discard_draw_penalty_should_be_waived(obs)

            # 先に使えるカードがあるとペナルティ（エネルギー例外で解除されうる）
            if not waive_energy_penalty:
                if _has_usable_non_draw_cards(obs, buckets):
                    type_bonus += _SUPPORTER_TYPE_BONUS["discard_draw_penalty"]

            # 進化ペアがあるときの追加ペナルティ（エネルギー例外の影響を受けない）
            evolution_pairs = _count_evolution_pairs_on_bench(obs)
            if evolution_pairs > 0:
                type_bonus += _EVOLUTION_PAIR_DISCARD_PENALTY * min(evolution_pairs, 2)

        score = base + effective_hand_bonus + type_bonus

        # エネ不足補正でも最終スコアが draw_or_search 基本値を大きく下回るなら提案しない
        if score <= 0:
            return None

        return MainActionProposal(
            action=[best_supporter_index],
            score=score,
            label=f"draw_or_search_supporter_{best_supporter_kind}",
        )

    return None