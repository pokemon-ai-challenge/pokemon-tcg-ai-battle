"""propose_ability_action — MAIN フェーズでの特性発動を提案する。

ability.py は AreaType / OptionType で場のポケモンを正確に特定できる実装になっている。

スコア分類:
  draw 系特性（ドロー・サーチ）  → score = 85（draw_or_search 基本 + 15）
  捨てドロー型特性（ルナトーン等） → 通常 85 だが、以下の条件で減点する
  その他特性                      → score = 58（ability 基本）

捨てドロー型 特性のスコア減点ルール（2つ独立して適用）:

  ルール A — エネルギー枯渇リスク
    条件: 手札の基本エネルギーが 1 枚以下 かつ ドロー系サポートが手札にない
    効果: -80 → 実質発動しない（エネルギーを唯一の1枚捨てると次ターン攻撃不能）

  ルール B — 他に先にやることがある × エネ補充前に使う損
    条件: 「あと1エネで技が使えるポケモンがいる」かつ「手札エネルギーが0枚」
          かつ「手札に先に使えるカードがある（進化/展開/グッズ等）」
    効果: -40 → 他の行動を先に済ませてから使う

    意図: エネが0の状態で捨てドロー系特性を使ってもエネを引く保証がない。
         さらに今の手札にある別の有効なカードを先に使えばターンを有効活用できる。
         例: ベンチにリオルがいてルカリオが手札にある + エネ0 → まず進化を優先

捨てても良い（ペナルティを解除する）例外条件 — 以下いずれか1つでも True ならルール A/B 双方を解除:

  例外 1 — 場のエネルギーが足りている
    主力アタッカーが既に最大火力の技を打てる状態なら、
    今後さらにエネルギーを心配する必要がないため捨てて良い。

  例外 2 — このターンもうエネルギーを付けられない
    obs.current.energyAttached が True（このターン既にエネ貼り済み）なら、
    手札のエネルギーをこのターン使うことはできないので温存する意味が薄い。

  例外 3 — 場のポケモンが攻撃で相手を倒せる
    今すぐ攻撃すれば相手のバトルポケモンをKOできる状況なら、
    エネルギー不足の心配より先に勝負を決めるべきなので捨てて良い。

山札切れガード（最優先・例外条件より上位）:

  条件: 自分の山札残り枚数（deckCount）が 10 枚以下
  効果: ドロー・サーチ系特性（draw 系・捨てドロー型問わず）を一切提案しない
  意図: 山札が0枚になった状態で自分のドローフェーズを迎えると敗北になるため、
       残り枚数が少ない状況ではこれ以上山札を減らす行動を避ける。
       上記の例外1〜3より優先され、これらの例外条件が真であっても無効化しない。
"""

from cg.api import AreaType, Attack, CardData, CardType, Observation, OptionType, all_attack, all_card_data

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS
from src.knowledge.card_cache import load_card_data

# カードデータのキャッシュ（毎ターン呼び出しを避ける）
_card_data_cache: dict[int, CardData] | None = None
_attack_data_cache: dict[int, Attack] | None = None

# 山札切れ防止: 自分の山札残り枚数がこの値以下なら、ドロー・サーチ系特性を使わない
_DECK_COUNT_SAFE_THRESHOLD = 10


def _get_card_data_map() -> dict[int, CardData]:
    """cardId → CardData の辞書を返す（初回のみロード）。"""
    global _card_data_cache
    if _card_data_cache is None:
        _card_data_cache = {c.cardId: c for c in all_card_data()}
    return _card_data_cache


def _get_attack_data_map() -> dict[int, Attack]:
    """attackId → Attack の辞書を返す（初回のみロード）。"""
    global _attack_data_cache
    if _attack_data_cache is None:
        _attack_data_cache = {a.attackId: a for a in all_attack()}
    return _attack_data_cache


# ドロー・サーチ系とみなす英語キーワード（skillテキスト内を検索）
_DRAW_KEYWORDS = (
    "draw",
    "search",
    "look at",
    "put into your hand",
    "add to your hand",
    "into your hand",
)


def _is_draw_ability(card: CardData) -> bool:
    """特性テキストにドロー・サーチ系の記述があるか判定する。"""
    for skill in card.skills:
        text_lower = skill.text.lower()
        if any(kw in text_lower for kw in _DRAW_KEYWORDS):
            return True
    return False


def _is_discard_draw_ability(card: CardData) -> bool:
    """特性テキストが「手札を捨ててドロー」または「X枚までドロー」系かを判定する。

    ルナトーン「みちびき」のように discard（または shuffle）+ draw を含む特性、
    もしくは "until you have" / "up to" + draw + hand のパターン（draw_to_x型）。
    これらは手札からカードを失うリスクを持つため、特別扱いする。
    """
    for skill in card.skills:
        t = skill.text.lower()
        has_draw = "draw" in t
        if not has_draw:
            continue
        # 捨てドロー型: discard または shuffle + draw
        if "discard" in t or "shuffle" in t:
            return True
        # X枚までドロー型
        if (
            ("until you have" in t or "so that you have" in t or "up to" in t)
            and "hand" in t
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# 盤面のエネルギー状況チェック（ability.py 内で独立して持つ）
# ---------------------------------------------------------------------------

def _count_hand_basic_energies(obs: Observation) -> int:
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


def _has_draw_supporter_in_hand(obs: Observation) -> bool:
    """手札にドロー系またはサーチ系のサポーターがあるか（buckets 不使用版）。"""
    if obs.current is None:
        return False
    your_index = obs.current.yourIndex
    hand = obs.current.players[your_index].hand
    if hand is None:
        return False

    card_data_list = load_card_data()
    card_data_by_id = {c.cardId: c for c in card_data_list}

    for hand_card in hand:
        card = card_data_by_id.get(hand_card.id)
        if card is None or not card.skills:
            continue
        t = card.skills[0].text.lower()
        if ("draw" in t) or ("search" in t) or ("look at" in t):
            return True
    return False


def _pokemon_needs_one_more_energy(obs: Observation) -> bool:
    """盤面のいずれかのポケモンで「あと1エネで技が使える」状況かを判定する。

    active + bench の全ポケモンを確認し、
    「今のエネでは使えないが、1枚追加すれば使える技がある」なら True を返す。
    """
    if obs.current is None:
        return False
    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]

    card_data_list = load_card_data()
    card_data_by_id = {c.cardId: c for c in card_data_list}

    all_pokemon = []
    if player.active:
        all_pokemon.extend(p for p in player.active if p is not None)
    all_pokemon.extend(p for p in player.bench if p is not None)

    for poke in all_pokemon:
        card = card_data_by_id.get(poke.id)
        if card is None:
            continue
        attached = len(poke.energies) if hasattr(poke, 'energies') and poke.energies else 0
        for atk in (card.attacks or []):
            if hasattr(atk, 'energies') and hasattr(atk, 'damage'):
                req = len(atk.energies) if atk.energies else 0
                # あと1枚で使える（かつ今は使えない）
                if req == attached + 1:
                    return True
    return False


def _max_damage_of_pokemon(pokemon_id: int, energies: list, card_data_by_id: dict, attack_data_by_id: dict) -> int:
    """ポケモンが現在のエネルギーで出せる最大ダメージを返す（エネ充足チェック込み）。"""
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
            if int(e) == 0:  # COLORLESS は何でも代替可
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
    """例外1: 主力アタッカー（active）が既に最大火力の技を打てる状態か。

    active が持つ技のうち最も要求エネが多い技に、現在のエネルギーで届いているなら True。
    （＝ これ以上エネルギーを心配する必要がない盤面）
    """
    if obs.current is None:
        return False
    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    if not player.active or player.active[0] is None:
        return False

    active = player.active[0]
    card_data_by_id = _get_card_data_map()
    attack_data_by_id = _get_attack_data_map()

    card = card_data_by_id.get(active.id)
    if card is None or not card.attacks:
        return False

    attached = len(active.energies) if active.energies else 0

    # 全技の中で最大のエネ要求枚数
    max_required = 0
    for attack_id in card.attacks:
        attack = attack_data_by_id.get(attack_id)
        if attack is None:
            continue
        max_required = max(max_required, len(attack.energies) if attack.energies else 0)

    return attached >= max_required


def _energy_already_attached_this_turn(obs: Observation) -> bool:
    """例外2: このターン既にエネルギーを付け終わっているか。

    obs.current.energyAttached が True なら、このターンはもうエネを貼れないため、
    手札のエネルギーを今ターン中に活用する手段がない。
    """
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

    card_data_by_id = _get_card_data_map()
    attack_data_by_id = _get_attack_data_map()

    my_max_dmg = _max_damage_of_pokemon(active.id, active.energies, card_data_by_id, attack_data_by_id)

    opp_hp = getattr(opp_active, "remainingHp", None)
    if opp_hp is None:
        opp_hp = getattr(opp_active, "hp", None)
    if opp_hp is None:
        return False

    return my_max_dmg >= opp_hp


def _discard_draw_penalty_should_be_waived(obs: Observation) -> bool:
    """捨てドロー型特性のペナルティ（ルール A / B）を解除すべきかをまとめて判定する。

    例外1〜3のいずれか1つでも True なら、エネルギーを気にせず捨ててよいと判断する。
    """
    if _active_already_has_max_attack_energy(obs):
        return True
    if _energy_already_attached_this_turn(obs):
        return True
    if _can_ko_opponent_active_now(obs):
        return True
    return False


def _deck_count_too_low(obs: Observation) -> bool:
    """自分の山札残り枚数が安全閾値以下かを判定する。

    山札が0枚の状態でドローフェーズを迎えると敗北になるため、
    残り枚数が少ないときはドロー・サーチ系の行動を避けるべきという判断に使う。
    """
    if obs.current is None:
        return False
    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    deck_count = getattr(player, "deckCount", None)
    if deck_count is None:
        return False
    return deck_count <= _DECK_COUNT_SAFE_THRESHOLD


def _has_usable_non_draw_cards(obs: Observation, buckets: MainOptionBuckets) -> bool:
    """手札に「先に使うべきカード（ドロー系以外）」があるか。

    進化・ポケモン展開・非ドロー系グッズ・どうぐ・スタジアム・エネ貼りのいずれかがある場合 True。
    ドロー系グッズ（ボール・ポケギア等）は除外する。
    """
    if obs.current is None or obs.select is None:
        return False

    card_data_list = load_card_data()
    card_data_by_id = {c.cardId: c for c in card_data_list}

    your_index = obs.current.yourIndex
    hand = obs.current.players[your_index].hand

    has_non_draw_items = False
    if hand is not None:
        for idx in buckets.item_play:
            opt = obs.select.option[idx]
            if opt.index is None or hand is None or opt.index >= len(hand):
                continue
            item_card = card_data_by_id.get(hand[opt.index].id)
            if item_card is None or not item_card.skills:
                continue
            t = item_card.skills[0].text.lower()
            # ドロー・サーチ系グッズは除外
            if not (("draw" in t) or ("search" in t) or ("look at" in t)):
                has_non_draw_items = True
                break

    return bool(
        buckets.pokemon_play
        or buckets.evolve
        or has_non_draw_items
        or buckets.tool_play
        or buckets.stadium_play
        or buckets.attach
    )


# ---------------------------------------------------------------------------
# propose 関数
# ---------------------------------------------------------------------------

def propose_ability_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """特性を使う候補を返す。ドロー・サーチ系特性は最優先で使う。

    捨てドロー型特性（ルナトーン等）には2つのルールでスコアを減点する:
      ルール A: 手札エネ1枚以下 かつ ドロー系サポートなし → -80
      ルール B: あと1エネで技が使える × 手札エネ0枚 × 他にやることあり → -40
    """
    if not buckets.ability:
        return None
    if obs.current is None:
        return None

    # 山札切れガード: 山札が少ないときはドロー・サーチ系特性を提案候補から除外する
    # （通常型特性は対象外なので、後段の通常型判定には影響しない）
    deck_too_low = _deck_count_too_low(obs)

    card_map = _get_card_data_map()
    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]

    def get_card_id_from_option_idx(opt_idx: int) -> int | None:
        """optionインデックス → そのカードID を返す。

        AreaType.ACTIVE / BENCH を使って場のポケモンを特定する。
        これが ability.py の正しい参照方法（手札ではなく場のポケモン）。
        """
        if obs.select is None:
            return None
        opt = obs.select.option[opt_idx]
        if opt.type != OptionType.ABILITY:
            return None
        if opt.area == AreaType.ACTIVE:
            poke = player.active[opt.index] if player.active else None
        elif opt.area == AreaType.BENCH:
            poke = player.bench[opt.index] if opt.index < len(player.bench) else None
        else:
            return None
        return poke.id if poke is not None else None

    # 各 ability 候補を「捨てドロー型」「その他ドロー型」「通常型」に分類する
    discard_draw_candidates: list[int] = []  # 捨てドロー / draw_to_x 型
    draw_candidates: list[int] = []           # 上記以外のドロー・サーチ型
    normal_candidates: list[int] = []         # 非ドロー型

    for opt_idx in buckets.ability:
        card_id = get_card_id_from_option_idx(opt_idx)
        if card_id is not None and card_id in card_map:
            card = card_map[card_id]
            if _is_discard_draw_ability(card):
                discard_draw_candidates.append(opt_idx)
            elif _is_draw_ability(card):
                draw_candidates.append(opt_idx)
            else:
                normal_candidates.append(opt_idx)
        else:
            normal_candidates.append(opt_idx)

    # ----- 純粋ドロー・サーチ系特性（捨てリスクなし）-----
    # 条件なしで最優先（score = 85）。ただし山札が少ないときは提案しない。
    if draw_candidates and not deck_too_low:
        return MainActionProposal(
            action=[draw_candidates[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["draw_or_search"] + 15,  # 85
            label="ability_draw",
        )

    # ----- 捨てドロー型特性（ルナトーン等）-----
    # 山札が少ないときは最優先で提案しない（例外1〜3より優先する山札切れガード）
    if discard_draw_candidates and not deck_too_low:
        base_score = MAIN_ACTION_BASE_WEIGHTS["draw_or_search"] + 15  # 85

        penalty = 0

        # 例外1〜3のいずれかに該当するなら、エネルギー枯渇を気にせず捨ててよい
        # （場のエネが足りている / このターンもう貼れない / 今すぐ相手をKOできる）
        waive_penalty = _discard_draw_penalty_should_be_waived(obs)

        if not waive_penalty:
            # ルール A: 手札エネが1枚以下 かつ ドロー系サポートなし
            # → エネを捨てると次ターン攻撃できなくなる可能性が高い
            hand_energy = _count_hand_basic_energies(obs)
            has_draw_sup = _has_draw_supporter_in_hand(obs)
            if hand_energy <= 1 and not has_draw_sup:
                penalty += 80

            # ルール B: あと1エネで技が使える × 手札エネ0枚 × 他に使えるカードあり
            # → 先に他のカードを使ってからにする（エネが0なら捨てても意味が薄い）
            if hand_energy == 0 and _pokemon_needs_one_more_energy(obs):
                if _has_usable_non_draw_cards(obs, buckets):
                    penalty += 40

        final_score = base_score - penalty
        label = "ability_discard_draw"
        if waive_penalty:
            label = "ability_discard_draw_energy_ok"
        elif penalty >= 80:
            label = "ability_discard_draw_energy_risk"
        elif penalty >= 40:
            label = "ability_discard_draw_do_other_first"

        return MainActionProposal(
            action=[discard_draw_candidates[0]],
            score=final_score,
            label=label,
        )

    # ----- 通常型特性 -----
    if normal_candidates:
        return MainActionProposal(
            action=[normal_candidates[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["ability"],  # 58
            label="ability",
        )

    return None