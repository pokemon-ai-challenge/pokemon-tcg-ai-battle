"""Phase 4: 複数手先探索によるアクション選択。

設計方針は docs/strategy-knowledge.md [SEARCH-001] / [SEARCH-002] を参照。

主なエントリポイント:
    choose_action_with_search(obs) -> list[int]

フォールバック: 推定失敗・例外時は choose_action(obs) に委ねる。
"""
import random
from cg.api import (
    Observation, SelectContext,
    search_begin, search_step, search_end, search_release,
)
from src.decision.router import choose_action
from src.decision.evaluator import get_evaluator
from src.knowledge.card_database import get_card_db
from src.knowledge.deck_plan import get_deck_plan
from src.knowledge.meta_decks import DECK_CATALOG, estimate_deck_candidates
from src.knowledge.decision_trace import trace

# --- 探索パラメータ ---
SEARCH_WIDTH = 5    # 候補アクション数
SEARCH_DEPTH = 8    # ロールアウト深さ

# --- モジュールレベル状態（ターンをまたいで相手の公開カードを蓄積する）---
# ゲームの各試合ごとにリセットが必要。_last_turn で検出。
_revealed_opp_card_ids: list[int] = []
_last_game_turn: int = -1

BASIC_FIGHT_ENERGY_ID = 6   # 基本格闘エネルギー（汎用パディング用）


def _reset_if_new_game(state) -> None:
    global _revealed_opp_card_ids, _last_game_turn
    turn = state.turn if state else 0
    if turn < _last_game_turn:  # ターンが戻った = 新しいゲーム
        _revealed_opp_card_ids = []
    _last_game_turn = turn


def _collect_revealed_opp_cards(state) -> list[int]:
    """相手の公開カード ID を収集してグローバルリストに蓄積する。"""
    global _revealed_opp_card_ids
    if not state:
        return _revealed_opp_card_ids

    opp_idx = 1 - state.yourIndex
    opp = state.players[opp_idx]

    newly_seen: set[int] = set()
    # バトル場（公開されているもの）
    for poke in opp.active:
        if poke:
            newly_seen.add(poke.id)
            for card in poke.energyCards:
                newly_seen.add(card.id)
            for card in poke.tools:
                newly_seen.add(card.id)
            for card in poke.preEvolution:
                newly_seen.add(card.id)
    # ベンチ
    for poke in opp.bench:
        newly_seen.add(poke.id)
        for card in poke.energyCards:
            newly_seen.add(card.id)
        for card in poke.tools:
            newly_seen.add(card.id)
        for card in poke.preEvolution:
            newly_seen.add(card.id)
    # トラッシュ（全て公開）
    for card in opp.discard:
        newly_seen.add(card.id)

    for cid in newly_seen:
        _revealed_opp_card_ids.append(cid)

    return _revealed_opp_card_ids


def _expand_recipe(recipe: list[tuple[int, int]]) -> list[int]:
    result = []
    for cid, cnt in recipe:
        result.extend([cid] * cnt)
    return result


def _estimate_opponent_state(state):
    """相手の非公開情報を推定して (opp_deck, opp_prize, opp_hand, opp_active) を返す。

    失敗時は None を返す（フォールバック用）。
    """
    opp_idx = 1 - state.yourIndex
    opp = state.players[opp_idx]

    # 相手デッキアーキタイプ推定
    candidates = estimate_deck_candidates(_revealed_opp_card_ids)
    best_arch = candidates[0][0] if candidates and candidates[0][1] > 0 else None

    if best_arch:
        base_recipe = list(_expand_recipe(DECK_CATALOG[best_arch].recipe))
    else:
        # フォールバック: 基本エネルギーだらけの汎用デッキ
        base_recipe = [BASIC_FIGHT_ENERGY_ID] * 60

    # 相手の公開済みカードをレシピから除去
    remaining = list(base_recipe)
    for cid in _revealed_opp_card_ids:
        if cid in remaining:
            remaining.remove(cid)

    # バトル場・ベンチに出ているポケモン本体 ID も除去済みと仮定
    # (energyCards / tools は _collect_revealed_opp_cards 内で収集済み)

    # 合計枚数を実際の枚数に合わせる
    deck_count = opp.deckCount
    prize_count = len([c for c in opp.prize if True])  # None も含む
    hand_count = opp.handCount
    total_needed = deck_count + prize_count + hand_count

    while len(remaining) < total_needed:
        remaining.append(BASIC_FIGHT_ENERGY_ID)
    if len(remaining) > total_needed:
        remaining = remaining[:total_needed]

    # 数の検証
    if len(remaining) < deck_count + prize_count + hand_count:
        return None  # 推定失敗

    # シャッフルして分割
    random.shuffle(remaining)
    opp_hand = remaining[:hand_count]
    opp_prize = remaining[hand_count:hand_count + prize_count]
    opp_deck = remaining[hand_count + prize_count:]

    # 相手アクティブが伏せの場合は推定
    opp_active_predicted: list[int] = []
    if opp.active and opp.active[0] is None:
        if best_arch:
            key_ids = DECK_CATALOG[best_arch].key_pokemon_ids
            # Basic ポケモンの ID を選ぶ
            card_db = get_card_db()
            for cid in key_ids:
                card = card_db.get(cid)
                if card and card.basic:
                    opp_active_predicted = [cid]
                    break
        if not opp_active_predicted:
            # フォールバック: アーキタイプ不明時はレシピ先頭の Basic
            card_db = get_card_db()
            for cid in base_recipe:
                card = card_db.get(cid)
                if card and card.basic:
                    opp_active_predicted = [cid]
                    break

    return opp_deck, opp_prize, opp_hand, opp_active_predicted


def _estimate_my_state(state):
    """自分のデッキ・プライズを推定して (my_deck, my_prize) を返す。"""
    from main import read_deck_csv
    my_idx = state.yourIndex
    my = state.players[my_idx]

    full_deck = read_deck_csv()  # 60枚のフル構成

    # 使用済みカード（バトル場・ベンチ・トラッシュ）をフルデッキから除去
    used: list[int] = []
    for poke in my.active:
        if poke:
            used.append(poke.id)
            used.extend(c.id for c in poke.energyCards)
            used.extend(c.id for c in poke.tools)
            used.extend(c.id for c in poke.preEvolution)
    for poke in my.bench:
        used.append(poke.id)
        used.extend(c.id for c in poke.energyCards)
        used.extend(c.id for c in poke.tools)
        used.extend(c.id for c in poke.preEvolution)
    for card in my.discard:
        used.append(card.id)
    if my.hand:
        used.extend(c.id for c in my.hand)

    remaining = list(full_deck)
    for cid in used:
        if cid in remaining:
            remaining.remove(cid)

    deck_count = my.deckCount
    prize_count = len([c for c in my.prize if True])
    total_needed = deck_count + prize_count

    while len(remaining) < total_needed:
        remaining.append(BASIC_FIGHT_ENERGY_ID)
    if len(remaining) > total_needed:
        remaining = remaining[:total_needed]

    random.shuffle(remaining)
    my_prize = remaining[:prize_count]
    my_deck = remaining[prize_count:]

    return my_deck, my_prize


def _get_top_candidates(obs: Observation, n: int = SEARCH_WIDTH) -> list[list[int]]:
    """評価関数で上位 n 候補アクションを生成する。

    現状は router.choose_action の結果を起点に、MAIN フェーズでのみ
    周辺候補（各オプションを単体で試す）を生成する。
    他コンテキストは候補1つ（通常通り）。
    """
    candidates = [choose_action(obs)]
    if obs.select is None:
        return candidates

    ctx = obs.select.context
    # MAIN フェーズのみ複数候補を生成（他は通常通り）
    if ctx != SelectContext.MAIN:
        return candidates

    # 各オプションを単体候補として追加（最大 n 個）
    for i in range(min(n, len(obs.select.option))):
        action = [i]
        if action not in candidates:
            candidates.append(action)
        if len(candidates) >= n:
            break

    return candidates[:n]


def choose_action_with_search(obs: Observation) -> list[int]:
    """探索を使って最善アクションを選ぶ。失敗時は choose_action にフォールバック。

    Phase 4 の主エントリポイント。
    MAIN フェーズ以外・デッキ選択・ゲーム未開始時はフォールバックする。
    """
    state = obs.current

    # フォールバック条件: 盤面情報なし / デッキ選択 / MAIN 以外
    if state is None or obs.select is None:
        return choose_action(obs)
    if obs.select.context != SelectContext.MAIN:
        return choose_action(obs)

    _reset_if_new_game(state)
    _collect_revealed_opp_cards(state)

    # 序盤(3ターン以内)で情報が少ない場合はフォールバック
    if state.turn <= 3 and not _revealed_opp_card_ids:
        return choose_action(obs)

    try:
        result = _search_best_action(obs)
        if result is not None:
            return result
    except Exception as e:
        trace(state.turn, "SEARCH_ERROR", -1, str(e)[:60])

    return choose_action(obs)


def _search_best_action(obs: Observation) -> list[int] | None:
    """search_begin + search_step で浅い探索を行い最善アクションを返す。

    失敗時は None を返す。
    """
    state = obs.current
    plan = get_deck_plan()
    evaluator = get_evaluator()

    # 相手状態推定
    opp_est = _estimate_opponent_state(state)
    if opp_est is None:
        return None
    opp_deck, opp_prize, opp_hand, opp_active = opp_est

    # 自分状態推定
    my_deck, my_prize = _estimate_my_state(state)

    # 候補アクション生成
    candidates = _get_top_candidates(obs, SEARCH_WIDTH)

    best_action = candidates[0]
    best_score = float("-inf")

    for candidate_action in candidates:
        root_state = None
        try:
            # 探索開始
            root_state = search_begin(
                agent_observation=obs,
                your_deck=my_deck,
                your_prize=my_prize,
                opponent_deck=opp_deck,
                opponent_prize=opp_prize,
                opponent_hand=opp_hand,
                opponent_active=opp_active,
            )

            # 候補アクションを1手進める（next_state は _rollout_score 内で解放）
            next_state = search_step(root_state.searchId, candidate_action)
            score = _rollout_score(next_state, plan, evaluator)

            if score > best_score:
                best_score = score
                best_action = candidate_action

        except Exception as e:
            # この候補だけ失敗 → スキップ
            trace(state.turn, "SEARCH_CAND_ERR", str(candidate_action), str(e)[:40])

        finally:
            # ROOT state は必ず解放（例外発生時も含む）
            if root_state is not None:
                try:
                    search_release(root_state.searchId)
                except Exception:
                    pass

    trace(state.turn, "SEARCH_BEST", best_action, f"score={best_score:.1f}")
    return best_action


def _rollout_score(search_state, plan, evaluator) -> float:
    """search_step を深さ分だけ進めて終端状態を評価する。

    各ステップはルールベースルーター（choose_action）を使ってロールアウトする。
    ランダム選択よりノイズが少なく、ルールベース戦略の質を評価できる。
    終端または最大深さに達したら BoardEvaluator でスコアを返す。
    メモリ管理: このメソッド内で生成したすべての searchId を finally で解放する。
    """
    current = search_state

    try:
        for _ in range(SEARCH_DEPTH):
            obs = current.observation
            if obs.current is not None and obs.current.result != -1:
                result = obs.current.result
                my_idx = obs.current.yourIndex
                return 1000.0 if result == my_idx else -1000.0

            if obs.select is None:
                break

            # ランダム選択で前進
            n_opts = len(obs.select.option)
            max_cnt = obs.select.maxCount
            action = random.sample(range(n_opts), min(max_cnt, n_opts))

            prev_id = current.searchId
            current = search_step(current.searchId, action)
            search_release(prev_id)

        final_obs = current.observation
        if final_obs.current:
            return evaluator.score(final_obs.current, plan)
        return 0.0

    finally:
        try:
            search_release(current.searchId)
        except Exception:
            pass
