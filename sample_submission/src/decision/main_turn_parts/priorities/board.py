from cg.api import Attack, CardData, CardType, LogType, Observation, all_attack, all_card_data

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS

# ---------------------------------------------------------------------------
# カードデータ・攻撃データのキャッシュ（呼び出しごとに毎回生成しない）
# ---------------------------------------------------------------------------

_card_cache: dict[int, CardData] | None = None
_attack_cache: dict[int, Attack] | None = None


def _get_card_data() -> dict[int, CardData]:
    global _card_cache
    if _card_cache is None:
        _card_cache = {c.cardId: c for c in all_card_data()}
    return _card_cache


def _get_attack_data() -> dict[int, Attack]:
    global _attack_cache
    if _attack_cache is None:
        _attack_cache = {a.attackId: a for a in all_attack()}
    return _attack_cache


# ---------------------------------------------------------------------------
# グッズ・どうぐのテキスト分類ユーティリティ
# ---------------------------------------------------------------------------

def _classify_item_text(text: str) -> str:
    """グッズの効果テキストから種別を返す。

    Returns:
        "search"       : 山札からポケモン等を持ってくる（ボール系等）
        "draw"         : 手札を増やす（ポケギア等）
        "draw_to_x"    : 手札がX枚になるまでドロー
        "other"        : 上記以外
    """
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
    card_data = _get_card_data().get(hand_card.id)
    if card_data is None or not card_data.skills:
        return None

    return card_data.skills[0].text


def _get_item_kind(option_index: int, obs: Observation) -> str:
    text = _get_card_skill_text(option_index, obs)
    if text is None:
        return "other"
    return _classify_item_text(text)


def _has_usable_non_draw_cards(obs: Observation, buckets: MainOptionBuckets) -> bool:
    """手札に「先に使うべきカード」があるか。

    discard_draw / draw_to_x を使う前に確認し、True ならスコアを下げる。
    item_play はドロー系グッズ（ボール・ポケギア等）も含むため、
    手札に「ドロー目的以外のグッズ」があるかをテキストで絞り込む。
    """
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


# ---------------------------------------------------------------------------
# 効果テキスト判定ユーティリティ
# ---------------------------------------------------------------------------

def _is_cant_use_next_turn(attack_text: str) -> bool:
    """「次のターン技が使えない」系の効果テキストか判定する。"""
    t = attack_text.lower()
    return ("can't use" in t or "cannot use" in t) and "next turn" in t


def _is_retreat_cost_reducer(skill_text: str) -> bool:
    """にげるコストを下げる/0にするどうぐのテキストか判定する。"""
    t = skill_text.lower()
    if "retreat cost" not in t:
        return False
    return any(kw in t for kw in ("less", " 0", "no ", "free", "reduc"))


def _tool_damage_bonus_amount(skill_text: str) -> int:
    """どうぐがワザのダメージを増やす量を返す（不明なら0）。

    "this pokemon's attacks do X more damage" のようなテキストを解析する。
    """
    import re
    t = skill_text.lower()
    # "do X more damage" パターン
    m = re.search(r'do\s+(\d+)\s+more\s+damage', t)
    if m:
        return int(m.group(1))
    # "deals X more damage" パターン
    m = re.search(r'deals?\s+(\d+)\s+more\s+damage', t)
    if m:
        return int(m.group(1))
    # "+X damage" パターン
    m = re.search(r'\+\s*(\d+)\s+damage', t)
    if m:
        return int(m.group(1))
    return 0


# ---------------------------------------------------------------------------
# 前ターンの攻撃ログから「次ターン技不可」状態かを判定
# ---------------------------------------------------------------------------

def _active_used_cant_use_attack(obs: Observation) -> bool:
    """バトルポケモンが直前ターンに「次のターン技が使えない」技を使ったか。"""
    if obs.current is None:
        return False

    your_index = obs.current.yourIndex
    attack_data = _get_attack_data()

    for log in obs.logs:
        if log.type == LogType.ATTACK and log.playerIndex == your_index:
            if log.attackId is None:
                continue
            attack = attack_data.get(log.attackId)
            if attack and _is_cant_use_next_turn(attack.text):
                return True
    return False


# ---------------------------------------------------------------------------
# ベンチの最大打点計算（エネルギー充足チェック付き）
# ---------------------------------------------------------------------------

def _max_damage_of_pokemon(pokemon_id: int, energies: list) -> int:
    """ポケモンが現在のエネルギーで出せる最大ダメージを返す。"""
    card_data = _get_card_data()
    attack_data = _get_attack_data()

    card = card_data.get(pokemon_id)
    if card is None:
        return 0

    energy_counts: dict[int, int] = {}
    for e in energies:
        energy_counts[int(e)] = energy_counts.get(int(e), 0) + 1

    max_dmg = 0
    for attack_id in card.attacks:
        attack = attack_data.get(attack_id)
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


def _bench_has_higher_damage_pokemon(obs: Observation) -> bool:
    """ベンチに、バトルポケモンより高い打点を出せるポケモンがいるか。"""
    if obs.current is None:
        return False

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]

    if not player.active or player.active[0] is None:
        return False

    active = player.active[0]
    active_dmg = _max_damage_of_pokemon(active.id, active.energies)

    for bench_poke in player.bench:
        bench_dmg = _max_damage_of_pokemon(bench_poke.id, bench_poke.energies)
        if bench_dmg > active_dmg:
            return True
    return False


# ---------------------------------------------------------------------------
# 進化ペア検出（ベンチの進化前 ↔ 手札の進化後）
# ---------------------------------------------------------------------------

def _get_hand_card_ids(obs: Observation) -> list[int]:
    """自分の手札にあるカードIDのリストを返す。"""
    if obs.current is None:
        return []
    your_index = obs.current.yourIndex
    hand = obs.current.players[your_index].hand
    if hand is None:
        return []
    return [card.id for card in hand]


def _count_evolution_pairs_on_bench(obs: Observation) -> int:
    """ベンチにいる進化前ポケモンに対して、手札に進化後カードが何枚あるかを返す。

    例: ベンチにリオル → 手札にルカリオEX があれば +1
        ベンチにマクノシタ → 手札にハリテヤマ があれば +1
        ベンチにソルロック & 手札にルナトーン → ペアとして +1（相互補完）

    CardData.evolvesFrom でつながりを確認する。
    """
    if obs.current is None:
        return 0

    card_data = _get_card_data()
    hand_ids = set(_get_hand_card_ids(obs))
    if not hand_ids:
        return 0

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]

    bench_ids = {poke.id for poke in player.bench if poke is not None}
    if not bench_ids:
        return 0

    # 手札の各カードについて「進化元がベンチにいるか」を確認
    pair_count = 0
    for hand_card_id in hand_ids:
        hand_card = card_data.get(hand_card_id)
        if hand_card is None:
            continue
        # evolvesFrom が設定されていれば、そのカードがベンチにいるかチェック
        if hasattr(hand_card, 'evolvesFrom') and hand_card.evolvesFrom:
            for bench_id in bench_ids:
                bench_card = card_data.get(bench_id)
                if bench_card is None:
                    continue
                # カード名で照合（evolvesFrom はカード名文字列）
                if bench_card.name and hand_card.evolvesFrom.lower() in bench_card.name.lower():
                    pair_count += 1
                    break  # 同一手札カードで二重カウントしない

    return pair_count


def _has_evolution_pair_at_risk(obs: Observation, buckets: MainOptionBuckets) -> bool:
    """手札に進化後カードがあり、かつ discard 系サポートを使うと失ってしまうリスクがあるか。

    draw 系でもサーチ系でもない supporter_play がある場合に呼ぶ想定。
    進化ペアが1つでも揃っていれば True を返す。
    """
    return _count_evolution_pairs_on_bench(obs) > 0


# ---------------------------------------------------------------------------
# どうぐのダメージ加算が KO に必要かを判定
# ---------------------------------------------------------------------------

def _opponent_active_remaining_hp(obs: Observation) -> int:
    """相手のバトルポケモンの残りHPを返す（不明なら9999）。"""
    if obs.current is None:
        return 9999
    your_index = obs.current.yourIndex
    opp_index = 1 - your_index
    opp_player = obs.current.players[opp_index]
    if not opp_player.active or opp_player.active[0] is None:
        return 9999
    active = opp_player.active[0]
    # remainingHp が存在する場合はそちらを優先
    if hasattr(active, 'remainingHp') and active.remainingHp is not None:
        return active.remainingHp
    # hp のみの場合（フル体力）
    if hasattr(active, 'hp') and active.hp is not None:
        return active.hp
    return 9999


def _my_active_best_attack_damage(obs: Observation) -> int:
    """自分のバトルポケモンが今出せる最大打点を返す。"""
    if obs.current is None:
        return 0
    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    if not player.active or player.active[0] is None:
        return 0
    active = player.active[0]
    return _max_damage_of_pokemon(active.id, active.energies)


def _tool_enables_ko(obs: Observation, tool_option_index: int) -> bool:
    """このどうぐを使うと、今ターンに相手activeをKOできるようになるか。

    ダメージ加算量を読み取り、「現在の打点 + 加算量 >= 相手の残りHP」なら True。
    ダメージ加算でないどうぐは False を返す。
    """
    text = _get_card_skill_text(tool_option_index, obs)
    if text is None:
        return False

    bonus = _tool_damage_bonus_amount(text)
    if bonus <= 0:
        return False

    current_dmg = _my_active_best_attack_damage(obs)
    opp_hp = _opponent_active_remaining_hp(obs)

    # KO ラインに届くかどうか
    return (current_dmg + bonus) >= opp_hp


def _tool_damage_is_wasted(obs: Observation, tool_option_index: int) -> bool:
    """このどうぐのダメージ加算が無駄打ちになるか。

    以下のいずれかなら True（今すぐ使う必要がない）:
    - 加算なしどうぐ → 判定対象外なので False
    - すでに今の打点だけで KO できる（加算が過剰）
    - 加算してもまだ KO できない（どうぐ単体では意味がない）
    """
    text = _get_card_skill_text(tool_option_index, obs)
    if text is None:
        return False

    bonus = _tool_damage_bonus_amount(text)
    if bonus <= 0:
        return False  # ダメージ系でないどうぐは別評価

    current_dmg = _my_active_best_attack_damage(obs)
    opp_hp = _opponent_active_remaining_hp(obs)

    already_ko = current_dmg >= opp_hp          # 加算なしでもKO可能
    still_not_ko = (current_dmg + bonus) < opp_hp  # 加算してもまだKO不可

    return already_ko or still_not_ko


# ---------------------------------------------------------------------------
# にげるコスト軽減どうぐを手札から探す
# ---------------------------------------------------------------------------

def _find_retreat_tool_index(obs: Observation, buckets: MainOptionBuckets) -> int | None:
    """tool_play の中から「にげるコスト軽減」どうぐの option_index を返す。"""
    if obs.current is None or obs.select is None:
        return None

    your_index = obs.current.yourIndex
    hand = obs.current.players[your_index].hand
    if hand is None:
        return None

    card_data = _get_card_data()

    for idx in buckets.tool_play:
        option = obs.select.option[idx]
        if option.index is None or option.index >= len(hand):
            continue
        hand_card = hand[option.index]
        card = card_data.get(hand_card.id)
        if card is None:
            continue
        for skill in card.skills:
            if _is_retreat_cost_reducer(skill.text):
                return idx
    return None


# ---------------------------------------------------------------------------
# propose 関数
# ---------------------------------------------------------------------------

def propose_pokemon_or_evolve_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """ポケモン展開や進化を候補として返す。

    - 進化を展開より優先する（進化できるなら即進化）
    - ベンチに空きがあればポケモンを展開する
    - ベンチに進化前がいて手札に進化後があるなら展開スコアを加算
      （例: リオルがベンチにいてルカリオEXが手札にある場合など）
    """
    if obs.current is None:
        return None

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    bench_space = player.benchMax - len(player.bench)

    # 進化を最優先（進化すれば即戦力が上がる）
    if buckets.evolve:
        return MainActionProposal(
            action=[buckets.evolve[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["evolve"] + 20,
            label="evolve",
        )

    if buckets.pokemon_play and bench_space > 0:
        base_score = MAIN_ACTION_BASE_WEIGHTS["pokemon_play"] + min(bench_space, 2) * 2 + 20

        # ベンチに進化前がいて手札に進化後がある → 早めに展開しておく価値がある
        # 手札から進化先を失う前にベンチに出しておく
        evolution_pairs = _count_evolution_pairs_on_bench(obs)
        # 展開するポケモン自体も将来の進化対象になりうる（サポート前に出す意義）
        # ペアがあるほど加点（最大+20）
        pair_bonus = min(evolution_pairs * 8, 20)

        return MainActionProposal(
            action=[buckets.pokemon_play[0]],
            score=base_score + pair_bonus,
            label="pokemon_play",
        )

    return None


def propose_board_item_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """グッズ、スタジアム、どうぐで盤面を整える候補を返す。

    優先順位:
        1. 「次ターン技不可」状態 + ベンチに高打点がいる → にげるコスト軽減どうぐ
        2. スタジアム（場に出ていなければ即置く）
        3. ドロー・サーチ系グッズ（ただし進化ペアがあるときは後回し）
        4. その他グッズ
        5. ダメージ加算どうぐ（KO に必要なときのみ優先、不要なら大幅減点）
        6. その他どうぐ
    """
    if obs.current is None:
        return None

    # --- 1. にげるコスト軽減どうぐ（メガブレイブ後の交代準備）---
    if (
        not obs.current.retreated
        and _active_used_cant_use_attack(obs)
        and _bench_has_higher_damage_pokemon(obs)
    ):
        retreat_tool_idx = _find_retreat_tool_index(obs, buckets)
        if retreat_tool_idx is not None:
            return MainActionProposal(
                action=[retreat_tool_idx],
                score=MAIN_ACTION_BASE_WEIGHTS["tool"] + 30,
                label="retreat_tool_for_swap",
            )

    # --- 2. スタジアム（場に出ていなければ置く）---
    if buckets.stadium_play and not obs.current.stadium:
        return MainActionProposal(
            action=[buckets.stadium_play[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["stadium"] + 10,
            label="stadium",
        )

    # 進化ペアの有無を先に確認（グッズ/どうぐの評価に使う）
    evolution_pairs = _count_evolution_pairs_on_bench(obs)

    # --- 3. ドロー・サーチ系グッズ ---
    for idx in buckets.item_play:
        kind = _get_item_kind(idx, obs)
        if kind in ("search", "draw", "draw_to_x"):
            score = MAIN_ACTION_BASE_WEIGHTS["board_item"] + 5

            # draw_to_x は他にやることがあると損
            if kind == "draw_to_x" and _has_usable_non_draw_cards(obs, buckets):
                score -= 40

            # 進化ペアが揃っている状態でドロー系グッズを先に使うと
            # 手札から進化後カードが流れるリスクがある（search はOK）
            if kind in ("draw", "draw_to_x") and evolution_pairs > 0:
                score -= evolution_pairs * 10

            return MainActionProposal(
                action=[idx],
                score=score,
                label=f"board_item_{kind}",
            )

    # --- 4. その他グッズ ---
    if buckets.item_play:
        return MainActionProposal(
            action=[buckets.item_play[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["board_item"],
            label="board_item",
        )

    # --- 5 & 6. どうぐ（ダメージ加算の必要性で評価を分岐）---
    if buckets.tool_play:
        # まずKOに必要などうぐを優先
        for idx in buckets.tool_play:
            if _tool_enables_ko(obs, idx):
                return MainActionProposal(
                    action=[idx],
                    score=MAIN_ACTION_BASE_WEIGHTS["tool"] + 20,
                    label="tool_enables_ko",
                )

        # KOに不要なダメージ加算どうぐは後回し（スコアを大幅に下げる）
        # 効果的に使えるどうぐ（ダメージ系でない）があればそちらを優先
        best_tool_idx: int | None = None
        best_tool_score: int = -9999
        for idx in buckets.tool_play:
            if _tool_damage_is_wasted(obs, idx):
                # ダメージ加算が無駄になる → 大幅減点
                s = MAIN_ACTION_BASE_WEIGHTS["tool"] - 35
            else:
                s = MAIN_ACTION_BASE_WEIGHTS["tool"]
            if s > best_tool_score:
                best_tool_score = s
                best_tool_idx = idx

        if best_tool_idx is not None and best_tool_score > 0:
            return MainActionProposal(
                action=[best_tool_idx],
                score=best_tool_score,
                label="tool",
            )

    return None