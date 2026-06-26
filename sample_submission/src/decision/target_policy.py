from cg.api import Observation, OptionType, AreaType, SelectContext
from src.decision.utils import get_card_id_from_option, get_inplay_pokemon_id
from src.knowledge.card_database import get_attack_db, get_card_db
from src.knowledge.deck_plan import get_deck_plan
from src.knowledge.opponent_model import get_opponent_model
from src.knowledge.matchup_policy import get_matchup_override
from src.knowledge.ex_immune import opponent_active_blocks_ex
from src.knowledge.decision_trace import trace


def _get_turn(obs: Observation) -> int:
    return obs.current.turn if obs.current else 0


# --- ATTACK context (35) ---

def choose_attack(obs: Observation) -> list[int]:
    """ATTACK: 一撃KOできる攻撃を優先し、なければ最大ダメージを選ぶ。"""
    state = obs.current
    attack_db = get_attack_db()
    options = obs.select.option
    turn = _get_turn(obs)

    opp_hp: int | None = None
    if state:
        opp_idx = 1 - state.yourIndex
        opp_active = state.players[opp_idx].active
        if opp_active and opp_active[0]:
            opp_hp = opp_active[0].hp

    if opp_hp is not None:
        for i, opt in enumerate(options):
            if opt.type == OptionType.ATTACK and opt.attackId is not None:
                atk = attack_db.get(opt.attackId)
                if atk and atk.damage >= opp_hp:
                    trace(turn, "ATTACK", i, f"lethal={atk.name}")
                    return [i]

    best_i = 0
    best_dmg = -1
    for i, opt in enumerate(options):
        if opt.type == OptionType.ATTACK and opt.attackId is not None:
            atk = attack_db.get(opt.attackId)
            if atk and atk.damage > best_dmg:
                best_dmg = atk.damage
                best_i = i

    trace(turn, "ATTACK", best_i, f"best_damage={best_dmg}")
    return [best_i]


# --- EVOLVES_FROM (18) / EVOLVES_TO (19) / EVOLVE (37) ---

def choose_evolves_from(obs: Observation) -> list[int]:
    """EVOLVES_FROM: 進化させるポケモンを evolution_priority で選ぶ。"""
    plan = get_deck_plan()
    state = obs.current
    options = obs.select.option
    turn = _get_turn(obs)

    best_i = 0
    best_score = -1

    for i, opt in enumerate(options):
        card_id = get_card_id_from_option(opt, state) if state else opt.cardId
        evo_target = plan.evolution_lines.get(card_id, None) if card_id else None
        if evo_target and evo_target in plan.evolution_priority:
            score = len(plan.evolution_priority) - plan.evolution_priority.index(evo_target)
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_i = i

    trace(turn, "EVOLVES_FROM", best_i, f"score={best_score}")
    return [best_i]


def choose_evolves_to(obs: Observation) -> list[int]:
    """EVOLVES_TO: 進化後カードを evolution_priority で選ぶ。"""
    plan = get_deck_plan()
    state = obs.current
    options = obs.select.option
    turn = _get_turn(obs)

    best_i = 0
    best_score = -1

    for i, opt in enumerate(options):
        card_id = get_card_id_from_option(opt, state) if state else opt.cardId
        if card_id in plan.evolution_priority:
            score = len(plan.evolution_priority) - plan.evolution_priority.index(card_id)
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_i = i

    trace(turn, "EVOLVES_TO", best_i, f"score={best_score}")
    return [best_i]


def choose_evolve(obs: Observation) -> list[int]:
    """EVOLVE context (37): 進化カードを evolution_priority で選ぶ。"""
    plan = get_deck_plan()
    state = obs.current
    options = obs.select.option
    turn = _get_turn(obs)

    best_i = 0
    best_score = -1

    for i, opt in enumerate(options):
        evo_card_id = get_card_id_from_option(opt, state) if state else opt.cardId
        if evo_card_id in plan.evolution_priority:
            score = len(plan.evolution_priority) - plan.evolution_priority.index(evo_card_id)
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_i = i

    trace(turn, "EVOLVE", best_i, f"score={best_score}")
    return [best_i]


# --- ATTACH_FROM (21) / ATTACH_TO (22) ---

def choose_attach_from(obs: Observation) -> list[int]:
    """ATTACH_FROM: エネルギーを付けるポケモンを energy_priority で選ぶ。
    EX 無効化アビリティ持ちが相手アクティブの場合は non_ex_energy_priority を使う。
    """
    plan = get_deck_plan()
    state = obs.current
    options = obs.select.option
    turn = _get_turn(obs)

    effective_priority = plan.energy_priority
    if state and opponent_active_blocks_ex(state) and plan.non_ex_energy_priority:
        effective_priority = plan.non_ex_energy_priority

    best_i = 0
    best_score = -1

    for i, opt in enumerate(options):
        card_id = get_card_id_from_option(opt, state) if state else opt.cardId
        if card_id in effective_priority:
            priority_score = len(effective_priority) - effective_priority.index(card_id)
            energy_count = 0
            if state:
                player_idx = opt.playerIndex if opt.playerIndex is not None else state.yourIndex
                player = state.players[player_idx]
                if opt.area == AreaType.ACTIVE and opt.index is not None:
                    poke = player.active[opt.index] if player.active and opt.index < len(player.active) else None
                    energy_count = len(poke.energies) if poke else 0
                elif opt.area == AreaType.BENCH and opt.index is not None:
                    if opt.index < len(player.bench):
                        energy_count = len(player.bench[opt.index].energies)
            # エネルギーが少ないほど付けたい（まだ攻撃できない方を優先）
            score = priority_score * 10 - energy_count
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_i = i

    trace(turn, "ATTACH_FROM", best_i, f"score={best_score}")
    return [best_i]


def choose_attach_to(obs: Observation) -> list[int]:
    """ATTACH_TO: 付けるカードを選ぶ。通常は先頭のみ（エネルギーは 1 種類）。"""
    trace(_get_turn(obs), "ATTACH_TO", 0, "default_first")
    return [0]


# --- SWITCH (3) ---

def _score_opponent_switch(opt, state, turn: int) -> float:
    """ボスの指令など相手ポケモンを引き出す際のスコア。

    優先順位:
      1. アーキタイプ別のボスターゲット（ARCH_BOSS_TARGETS順）
      2. HP が低い（ほぼ倒せる）ポケモン
    """
    card_id = get_card_id_from_option(opt, state) if state else opt.cardId
    score = 0.0

    # 1. アーキタイプ別優先ターゲット
    boss_targets = get_opponent_model().boss_target_ids()
    if card_id and card_id in boss_targets:
        score += 10000 - boss_targets.index(card_id) * 1000

    # 2. HP が低いポケモンを優先（KO しやすい）
    if state and opt.area == AreaType.BENCH and opt.index is not None:
        player_idx = opt.playerIndex if opt.playerIndex is not None else state.yourIndex
        bench = state.players[player_idx].bench
        if opt.index < len(bench):
            poke = bench[opt.index]
            if poke and poke.maxHp > 0:
                dmg_ratio = 1.0 - (poke.hp / poke.maxHp)
                score += dmg_ratio * 500

    return score


def _any_bench_attacker_ready(options, state, plan) -> bool:
    """ベンチに攻撃準備完了（エネルギー1枚以上）のメイン/サブアタッカーがいるかチェック。"""
    if state is None:
        return False
    all_attacker_ids = set(plan.main_attacker_ids) | set(plan.sub_attacker_ids)
    for opt in options:
        cid = get_card_id_from_option(opt, state)
        if cid in all_attacker_ids:
            if opt.area == AreaType.BENCH and opt.index is not None:
                bench = state.players[state.yourIndex].bench
                if opt.index < len(bench) and len(bench[opt.index].energies) >= 1:
                    return True
    return False


def choose_switch(obs: Observation) -> list[int]:
    """SWITCH: 自分切り替え時は energy_priority、ボスの指令時は相手デッキ推定を使う。

    Lazy Evaluation (ARC-7):
      TO_ACTIVE context（KO 後の強制アクティブ選択）でメインアタッカーが未準備の場合、
      逃げエネ0の非アタッカーポケモンを優先して前に出しバッファにする。
      その後 main_policy でドロー・サーチを済ませてから最適アタッカーに交代する。
    """
    plan = get_deck_plan()
    state = obs.current
    options = obs.select.option
    turn = _get_turn(obs)
    is_forced_active = (obs.select.context == SelectContext.TO_ACTIVE)

    best_i = 0
    best_score = float("-inf")

    # Lazy Evaluation: KO後の強制アクティブ選択でアタッカーが未準備なら逃げエネ0優先
    use_lazy_eval = (
        is_forced_active
        and state is not None
        and not _any_bench_attacker_ready(options, state, plan)
    )

    card_db = get_card_db() if use_lazy_eval else None

    for i, opt in enumerate(options):
        # ボスの指令判定: 相手のポケモンを選んでいるか
        is_opponent_target = (
            state is not None
            and opt.playerIndex is not None
            and opt.playerIndex != state.yourIndex
        )

        if is_opponent_target:
            score = _score_opponent_switch(opt, state, turn)
        elif use_lazy_eval:
            # Lazy Eval: 逃げエネ0の非アタッカーに最高スコアを付ける
            cid = get_card_id_from_option(opt, state)
            all_attacker_ids = set(plan.main_attacker_ids) | set(plan.sub_attacker_ids)
            data = card_db.get(cid) if cid and card_db else None
            if data and data.retreatCost == 0 and cid not in all_attacker_ids:
                score = 5000.0  # バッファとして前出し
            else:
                # フォールバック: 通常の energy_priority
                score = 0.0
                if cid in plan.energy_priority:
                    score += (len(plan.energy_priority) - plan.energy_priority.index(cid)) * 10
        else:
            # 自分切り替え: Crustle 対面は non_ex_active_priority を使う
            card_id = get_card_id_from_option(opt, state) if state else opt.cardId
            effective_switch_priority = plan.energy_priority
            if state and opponent_active_blocks_ex(state) and plan.non_ex_active_priority:
                effective_switch_priority = plan.non_ex_active_priority
            score = 0.0
            if card_id in effective_switch_priority:
                score += (len(effective_switch_priority) - effective_switch_priority.index(card_id)) * 10
            if state and opt.area == AreaType.BENCH and opt.index is not None:
                player_idx = opt.playerIndex if opt.playerIndex is not None else state.yourIndex
                bench = state.players[player_idx].bench
                if opt.index < len(bench):
                    score += len(bench[opt.index].energies) * 5

        if score > best_score:
            best_score = score
            best_i = i

    arch = get_opponent_model().estimate_arch() or "unknown"
    reason = "lazy_eval_buffer" if use_lazy_eval else f"score={best_score:.0f}"
    trace(turn, "SWITCH", best_i, f"{reason} opp_arch={arch}")
    return [best_i]


# --- TO_BENCH (5) ---

def choose_to_bench(obs: Observation) -> list[int]:
    """TO_BENCH: bench_priority に従ってベンチに出すポケモンを選ぶ。
    MatchupPolicy の max_bench_size を超えないよう展開数を制限する。
    """
    plan = get_deck_plan()
    state = obs.current
    options = obs.select.option
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    turn = _get_turn(obs)

    # MatchupPolicy: 現在のベンチ枚数と戦略上限を比較して展開数を決める
    bench_count = len(state.players[state.yourIndex].bench) if state else 0
    override = get_matchup_override(get_opponent_model().estimate_arch())
    available_slots = max(0, override.max_bench_size - bench_count)
    # 上限内で出せる最大数 (ただし min_count は必ず満たす)
    effective_max = max(min_count, min(max_count, available_slots))

    scored = []
    for i, opt in enumerate(options):
        card_id = get_card_id_from_option(opt, state) if state else opt.cardId
        if card_id in plan.bench_priority:
            score = 100 - plan.bench_priority.index(card_id)
        else:
            score = 0
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [i for _, i in scored[:effective_max]]

    trace(turn, "TO_BENCH", chosen, f"place_{len(chosen)} bench={bench_count}/{override.max_bench_size}")
    return chosen


# --- TO_HAND (7) ---

def choose_to_hand(obs: Observation) -> list[int]:
    """TO_HAND: do_not_discard_ids を優先して手札に加える。"""
    plan = get_deck_plan()
    state = obs.current
    options = obs.select.option
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    turn = _get_turn(obs)

    scored = []
    for i, opt in enumerate(options):
        card_id = get_card_id_from_option(opt, state) if state else opt.cardId
        if card_id in plan.bench_priority:
            score = 200 - plan.bench_priority.index(card_id)
        elif card_id in plan.do_not_discard_ids:
            score = 100
        else:
            score = 10
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [i for _, i in scored[:max_count]]
    if len(chosen) < min_count:
        all_indices = {i for _, i in scored}
        for _, i in scored:
            if i not in chosen:
                chosen.append(i)
                if len(chosen) >= min_count:
                    break

    trace(turn, "TO_HAND", chosen, f"pick_{len(chosen)}")
    return chosen


# --- DISCARD (8) ---

def choose_discard(obs: Observation) -> list[int]:
    """DISCARD: do_not_discard_ids を避けて最低限の枚数を捨てる。"""
    plan = get_deck_plan()
    state = obs.current
    options = obs.select.option
    min_count = obs.select.minCount
    turn = _get_turn(obs)

    scored = []
    for i, opt in enumerate(options):
        card_id = get_card_id_from_option(opt, state) if state else opt.cardId
        if card_id in plan.do_not_discard_ids:
            score = 100  # 捨てたくない（スコアが高い = 後回し）
        else:
            score = 10   # 捨ててもよい
        scored.append((score, i))

    scored.sort(key=lambda x: x[0])  # スコアが低い（= 捨ててよい）順
    chosen = [i for _, i in scored[:min_count]]

    trace(turn, "DISCARD", chosen, f"discard_{len(chosen)}")
    return chosen
