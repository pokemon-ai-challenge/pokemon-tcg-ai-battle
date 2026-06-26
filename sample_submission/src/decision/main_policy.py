from cg.api import Observation, OptionType, CardType, AreaType
from src.decision.utils import get_card_id_from_option, get_inplay_pokemon_id
from src.decision.evaluator import get_evaluator
from src.decision.risk_evaluator import will_be_ko_next_turn, energy_cap
from src.knowledge.card_database import get_attack_db, get_card_db
from src.knowledge.deck_plan import get_deck_plan
from src.knowledge.opponent_model import get_opponent_model, IONO_ID
from src.knowledge.matchup_policy import get_matchup_override
from src.knowledge.ex_immune import opponent_active_blocks_ex, my_active_is_ex
from src.knowledge.decision_trace import trace

BOSS_ORDERS_ID = 1182  # ボスの指令 (PAL 172)


def _classify_main_options(obs: Observation) -> dict[str, list[int]]:
    """MAIN フェーズの選択肢を種類ごとに分類する（分類責務）。"""
    classified: dict[str, list[int]] = {
        "play": [], "attach": [], "evolve": [], "ability": [],
        "discard": [], "retreat": [], "attack": [], "end": [],
    }
    for i, option in enumerate(obs.select.option):
        if option.type == OptionType.PLAY:
            classified["play"].append(i)
        elif option.type == OptionType.ATTACH:
            classified["attach"].append(i)
        elif option.type == OptionType.EVOLVE:
            classified["evolve"].append(i)
        elif option.type == OptionType.ABILITY:
            classified["ability"].append(i)
        elif option.type == OptionType.DISCARD:
            classified["discard"].append(i)
        elif option.type == OptionType.RETREAT:
            classified["retreat"].append(i)
        elif option.type == OptionType.ATTACK:
            classified["attack"].append(i)
        elif option.type == OptionType.END:
            classified["end"].append(i)
    return classified


def _pick_best_evolve(obs: Observation, evolve_options: list[int]) -> int:
    """EVOLVE 選択肢のうち evolution_priority が最高のものを選ぶ（優先順位判断責務）。"""
    plan = get_deck_plan()
    state = obs.current

    best_i = evolve_options[0]
    best_score = -1

    for i in evolve_options:
        opt = obs.select.option[i]
        # EVOLVE option: area/index = 進化後カード, inPlayArea/inPlayIndex = 進化元ポケモン
        evo_card_id = get_card_id_from_option(opt, state) if state else opt.cardId
        if evo_card_id in plan.evolution_priority:
            score = len(plan.evolution_priority) - plan.evolution_priority.index(evo_card_id)
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def _pick_best_attach(obs: Observation, attach_options: list[int]) -> int:
    """ATTACH 選択肢のうち energy_priority が最高のポケモン宛のものを選ぶ（優先順位判断責務）。"""
    plan = get_deck_plan()
    state = obs.current

    effective_priority = plan.energy_priority
    if state and opponent_active_blocks_ex(state) and plan.non_ex_energy_priority:
        effective_priority = plan.non_ex_energy_priority

    best_i = attach_options[0]
    best_score = -1

    for i in attach_options:
        opt = obs.select.option[i]
        # ATTACH option: inPlayArea/inPlayIndex = エネルギーを付けるポケモン
        target_id = get_inplay_pokemon_id(opt, state) if state else None

        if target_id in effective_priority:
            priority_score = len(effective_priority) - effective_priority.index(target_id)
            # エネルギーが少ないポケモンを優先（まだ攻撃できない方に付ける）
            energy_count = 0
            target_poke = None
            is_active = False
            if state:
                player = state.players[state.yourIndex]
                if opt.inPlayArea == AreaType.ACTIVE and player.active and opt.inPlayIndex is not None:
                    target_poke = player.active[opt.inPlayIndex] if opt.inPlayIndex < len(player.active) else None
                    energy_count = len(target_poke.energies) if target_poke else 0
                    is_active = True
                elif opt.inPlayArea == AreaType.BENCH and opt.inPlayIndex is not None:
                    if opt.inPlayIndex < len(player.bench):
                        target_poke = player.bench[opt.inPlayIndex]
                        energy_count = len(target_poke.energies)

            # エネルギー上限チェック: cap に達したポケモンへの付与はスキップ
            if target_id and energy_count >= energy_cap(target_id):
                score = 0
            # アクティブが次ターンKOされる場合はベンチを優先（エネルギー喪失を防ぐ）
            elif is_active and state and target_poke and will_be_ko_next_turn(state, target_poke):
                score = priority_score * 2  # 大幅減点してベンチへ回す
            else:
                score = priority_score * 10 - energy_count
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def _pick_best_attack(obs: Observation, attack_options: list[int]) -> int:
    """ATTACK 選択肢のうちダメージが最大（または一撃KO）のものを選ぶ（優先順位判断責務）。"""
    state = obs.current
    attack_db = get_attack_db()

    opp_hp: int | None = None
    if state:
        opp_idx = 1 - state.yourIndex
        opp_active = state.players[opp_idx].active
        if opp_active and opp_active[0]:
            opp_hp = opp_active[0].hp

    # まず一撃KOできる攻撃を探す
    if opp_hp is not None:
        for i in attack_options:
            opt = obs.select.option[i]
            if opt.attackId is not None:
                atk = attack_db.get(opt.attackId)
                if atk and atk.damage >= opp_hp:
                    return i

    # 最大ダメージの攻撃を選ぶ
    best_i = attack_options[0]
    best_dmg = -1
    for i in attack_options:
        opt = obs.select.option[i]
        if opt.attackId is not None:
            atk = attack_db.get(opt.attackId)
            if atk and atk.damage > best_dmg:
                best_dmg = atk.damage
                best_i = i

    return best_i


def _active_is_buffer(state, plan) -> bool:
    """アクティブが「バッファ」ポケモンで、ベンチに攻撃準備完了アタッカーがいるかチェック（ARC-7）。

    バッファ判定条件（いずれか）:
      - 逃げエネ 0（タダで引っ込める）
      - plan.buffer_pokemon_ids に含まれる（進化前壁役として明示指定）

    True の場合、Lazy Evaluation に従いリトリートして最適アタッカーに交代する。
    """
    card_db = get_card_db()
    my = state.players[state.yourIndex]
    active = my.active[0] if my.active else None
    if not active:
        return False

    # アタッカーはバッファではない
    if active.id in plan.main_attacker_ids or active.id in plan.sub_attacker_ids:
        return False

    # 逃げエネ0 OR buffer_pokemon_ids に含まれる
    active_card = card_db.get(active.id)
    is_free_retreat = active_card and active_card.retreatCost == 0
    is_declared_buffer = active.id in plan.buffer_pokemon_ids
    if not (is_free_retreat or is_declared_buffer):
        return False

    # ベンチに攻撃準備完了（エネルギー1枚以上）のアタッカーがいるか
    for bench_poke in my.bench:
        if bench_poke.id in plan.main_attacker_ids or bench_poke.id in plan.sub_attacker_ids:
            if len(bench_poke.energies) >= 1:
                return True

    return False


def _non_ex_bench_ready(state, plan) -> bool:
    """非EXアタッカーがベンチにいてエネルギーを 1 枚以上持っているか確認。"""
    if not plan.non_ex_active_priority:
        return False
    non_ex_ids = set(plan.non_ex_active_priority)
    bench = state.players[state.yourIndex].bench
    return any(poke and poke.id in non_ex_ids and len(poke.energies) >= 1 for poke in bench)


def _boss_target_on_bench(obs: Observation, boss_targets: list[int]) -> bool:
    """boss_rush ターゲットが相手ベンチに存在するかチェック。"""
    state = obs.current
    if not state or not boss_targets:
        return False
    opp_idx = 1 - state.yourIndex
    bench_ids = {poke.id for poke in state.players[opp_idx].bench if poke}
    return bool(bench_ids & set(boss_targets))


def _pick_best_play(obs: Observation, play_options: list[int]) -> int:
    """PLAY 選択肢のうち最も優先度の高いカードを選ぶ（カードタイプ・優先リスト基準）。

    優先度: ボスラッシュ時のボスの指令 > ベンチ展開ポケモン > サポーター > アイテム > どうぐ > スタジアム
    MatchupPolicy の max_bench_size に達している場合はベンチ展開ポケモンのスコアを 0 にする。
    boss_rush=True でボスターゲットがベンチにいる場合、ボスの指令を最優先にする。
    Iono タイミング（ARC-5）: 相手手札が少ない場合はイオナのスコアを大幅に下げる。
    """
    plan = get_deck_plan()
    card_db = get_card_db()
    state = obs.current
    opp_model = get_opponent_model()

    # MatchupPolicy: ベンチ上限・boss_rush 設定を取得
    bench_count = len(state.players[state.yourIndex].bench) if state else 0
    override = get_matchup_override(opp_model.estimate_arch())
    bench_full = bench_count >= override.max_bench_size

    # boss_rush: ターゲットがベンチにいるかどうかを事前確認（ループ外で1回だけ）
    boss_targets = opp_model.boss_target_ids() if override.boss_rush else []
    boss_rush_active = override.boss_rush and _boss_target_on_bench(obs, boss_targets)

    # Iono タイミング判断（ARC-5）: 相手手札の質を評価
    hand_quality = opp_model.estimate_hand_quality(obs)

    # カードタイプ別スコア（高いほど優先）
    SCORE = {
        CardType.SUPPORTER: 25,
        CardType.ITEM: 20,
        CardType.TOOL: 15,
        CardType.STADIUM: 10,
        CardType.POKEMON: 5,
    }
    BENCH_PRIORITY_POKEMON_SCORE = 30  # bench_priority ポケモンは最高優先（上限内の場合）
    BOSS_RUSH_SCORE = 40               # boss_rush 時のボスの指令: 全カードより最優先

    best_i = play_options[0]
    best_score = -1

    for i in play_options:
        opt = obs.select.option[i]
        score = 5

        # PLAY option の index は手札内インデックス (area フィールドは不定)
        if opt.index is not None and state:
            player = state.players[state.yourIndex]
            if player.hand and opt.index < len(player.hand):
                card_id = player.hand[opt.index].id
                card_data = card_db.get(card_id)
                if card_data:
                    if card_data.cardType == CardType.POKEMON:
                        if bench_full:
                            score = 0
                        elif card_id in plan.bench_priority:
                            score = BENCH_PRIORITY_POKEMON_SCORE
                        else:
                            score = 0  # bench_priority 外のポケモンは出さない
                    elif (card_data.cardType == CardType.SUPPORTER
                          and boss_rush_active
                          and card_id == BOSS_ORDERS_ID):
                        score = BOSS_RUSH_SCORE
                    elif card_data.cardType == CardType.SUPPORTER and card_id == IONO_ID:
                        # Iono タイミング調整（ARC-5）
                        # 相手手札が少ない（hand_quality < 0）: 使わない（score=3）
                        # 普通（hand_quality 0〜2）: やや控える（score=18）
                        # 相手が充実した準備完了（hand_quality >= 2）: 最優先（score=35）
                        if hand_quality < 0:
                            score = 3   # 使わない（他サポーターより大幅に低い）
                        elif hand_quality >= 2:
                            score = 35  # ボスより上に来るよう高スコア
                        else:
                            score = 18  # 通常サポーターより少し低い
                    else:
                        score = SCORE.get(card_data.cardType, 5)

        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def choose_main_action(obs: Observation) -> list[int]:
    """MAIN フェーズの行動を分類→優先順位判断で選ぶ。"""
    state = obs.current
    if state is None:
        return [0]

    classified = _classify_main_options(obs)
    plan = get_deck_plan()
    turn = state.turn

    # 評価関数によるリトリート判断: HP < 25% かつベンチに攻撃準備済みアタッカーがいるなら先に引く
    # 攻撃 > リトリートのデフォルト優先順位を逆転させる場合にのみ使う
    evaluator = get_evaluator()
    prefer_retreat = (
        classified["retreat"]
        and classified["attack"]
        and evaluator.should_retreat(state, plan)
    )
    if prefer_retreat:
        trace(turn, "MAIN", classified["retreat"][0], "evaluator_retreat_low_hp")
        return [classified["retreat"][0]]

    # 優先順位: 進化 > エネルギー付け > カードプレイ > アビリティ > 攻撃 > リトリート > 終了
    if classified["evolve"]:
        best = _pick_best_evolve(obs, classified["evolve"])
        trace(turn, "MAIN", best, "evolve_first")
        return [best]

    if not state.energyAttached and classified["attach"]:
        best = _pick_best_attach(obs, classified["attach"])
        trace(turn, "MAIN", best, "energy_attach")
        return [best]

    if classified["play"]:
        best = _pick_best_play(obs, classified["play"])
        trace(turn, "MAIN", best, "play_card")
        return [best]

    if classified["ability"]:
        trace(turn, "MAIN", classified["ability"][0], "use_ability")
        return [classified["ability"][0]]

    # Lazy Evaluation（ARC-7）: バッファポケモンがアクティブで、アタッカーが準備完了→交代
    if classified["retreat"] and _active_is_buffer(state, plan):
        trace(turn, "MAIN", classified["retreat"][0], "buffer_retreat_lazy_eval")
        return [classified["retreat"][0]]

    # Crustle 対面: 非EXアタッカーがベンチでエネルギーを持っていたら引っ込んで交代
    # boss_rush でイシズマイが引き出された場合は opponent_active_blocks_ex() = False → この分岐を通らない
    if (classified["retreat"]
            and opponent_active_blocks_ex(state)
            and my_active_is_ex(state)
            and _non_ex_bench_ready(state, plan)):
        trace(turn, "MAIN", classified["retreat"][0], "crustle_swap_to_non_ex")
        return [classified["retreat"][0]]

    if classified["attack"]:
        best = _pick_best_attack(obs, classified["attack"])
        trace(turn, "MAIN", best, "attack")
        return [best]

    if classified["retreat"]:
        trace(turn, "MAIN", classified["retreat"][0], "retreat")
        return [classified["retreat"][0]]

    if classified["end"]:
        trace(turn, "MAIN", classified["end"][0], "end_turn")
        return [classified["end"][0]]

    return [0]
