# -*- coding: utf-8 -*-
"""
main.py - Pokemon TCG AI using Monte Carlo Simulation + Heuristics

強化学習的アプローチ:
  - 状態 (State): Observation（盤面・手札・ログ）
  - 行動 (Action): list[int]（選択肢インデックス）
  - 価値 (Value): サイド差分・HP比率などから計算
  - 探索: ゲームエンジンのsearch APIを使ったモンテカルロシミュレーション

手法: Flat Monte Carlo Search
  - 各合法手について複数回ランダムロールアウト
  - 期待価値が最大の行動を選択
  - 時間切れ時はヒューリスティックにフォールバック
"""

import os
import math
import time
import random
import logging
from typing import Optional

from cg.api import (
    Observation, SelectContext, OptionType, LogType,
    to_observation_class, all_card_data, all_attack, CardData, Attack,
    search_begin, search_step, search_end, search_release,
)

# ── Logging ───────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

# ── Hyperparameters ───────────────────────────────────────────────────────────

MCTS_TIME_BUDGET_SEC = 2.5   # 1ターンあたりの探索時間上限 (秒)
ROLLOUT_DEPTH       = 25     # ロールアウトの最大ステップ数
MAX_CANDIDATES      = 10     # 評価する候補行動の上限
ROLLOUT_ACTION_POOL = 3      # ロールアウト中に試す行動候補数

# ── Card database ─────────────────────────────────────────────────────────────

_CARD_DB:  dict[int, CardData] = {}
_ATTACK_DB: dict[int, Attack]  = {}
_DB_LOADED = False

def _load_databases() -> None:
    global _DB_LOADED
    if _DB_LOADED:
        return
    for c in all_card_data():
        _CARD_DB[c.cardId] = c
    for a in all_attack():
        _ATTACK_DB[a.attackId] = a
    _DB_LOADED = True

# ── Deck I/O ──────────────────────────────────────────────────────────────────

def read_deck_csv() -> list[int]:
    """deck.csv を読んで 60 枚のカードID リストを返す。"""
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as f:
        lines = f.read().split("\n")
    return [int(lines[i]) for i in range(60)]

# ── Game state helpers ────────────────────────────────────────────────────────

def _is_game_over(obs: Observation) -> bool:
    return obs.current is not None and obs.current.result != -1

def _game_result(obs: Observation, my_index: int) -> float:
    """ゲーム終了時の価値。勝ち=1.0, 負け=-1.0, 引き分け=0.0"""
    r = obs.current.result
    if r == my_index:
        return 1.0
    if r == 1 - my_index:
        return -1.0
    return 0.0

# ── Evaluation function ───────────────────────────────────────────────────────

def evaluate(obs: Observation, my_index: int) -> float:
    """
    盤面の価値を [-1, 1] で返す（my_index 視点）。

    考慮要素:
      - サイドカード残枚数の差（最重要）
      - バトルポケモンのHP残量比率
      - ベンチポケモン枚数差
      - バトルポケモンに付いているエネルギー数
    """
    if obs.current is None:
        return 0.0
    state = obs.current

    if state.result != -1:
        return _game_result(obs, my_index)

    me  = state.players[my_index]
    opp = state.players[1 - my_index]

    # サイド差分: 残り少ない方が有利
    prize_score = (len(opp.prize) - len(me.prize)) / 6.0

    # バトルポケモン HP 比率
    def hp_ratio(ps) -> float:
        if not ps.active or ps.active[0] is None:
            return 0.0
        p = ps.active[0]
        return p.hp / max(p.maxHp, 1)

    hp_score = (hp_ratio(me) - hp_ratio(opp)) * 0.15

    # ベンチ枚数（展開の進み具合）
    bench_score = (len(me.bench) - len(opp.bench)) * 0.04

    # バトルポケモンのエネルギー（攻撃準備）
    energy_score = 0.0
    if me.active and me.active[0] is not None:
        energy_score = min(len(me.active[0].energies), 3) * 0.02

    return max(-1.0, min(1.0,
        prize_score * 0.70
        + hp_score
        + bench_score
        + energy_score
    ))

# ── Candidate action generation ───────────────────────────────────────────────

def _legal_actions(obs: Observation, max_actions: int = MAX_CANDIDATES) -> list[list[int]]:
    """
    現在の選択状況に対して合法な行動候補を列挙する。
    各要素は list[int]（選択するオプションのインデックスリスト）。
    """
    select = obs.select
    n   = len(select.option)
    lo  = select.minCount
    hi  = select.maxCount

    if n == 0:
        return [[]] if lo == 0 else []

    candidates: list[list[int]] = []

    if hi <= 1:
        # 0 or 1 択: 各オプションを個別に試す
        for i in range(n):
            candidates.append([i])
        if lo == 0:
            candidates.append([])
    elif lo == hi:
        # 固定枚数選択
        if n <= hi:
            candidates.append(list(range(n)))
        else:
            candidates.append(list(range(hi)))  # 先頭 hi 枚
            for _ in range(min(max_actions - 1, 5)):
                candidates.append(random.sample(range(n), hi))
    else:
        # 可変枚数選択
        for count in range(lo, min(hi + 1, lo + 4)):
            if n >= count:
                candidates.append(list(range(count)))
                if n > count and count > 0:
                    candidates.append(random.sample(range(n), count))

    # 重複除去
    seen: set[tuple] = set()
    unique: list[list[int]] = []
    for a in candidates:
        key = tuple(sorted(a))
        if key not in seen:
            seen.add(key)
            unique.append(a)

    return unique[:max_actions] if unique else [list(range(lo))]


def _random_legal_action(obs: Observation) -> list[int]:
    """ロールアウト用: ランダムに合法手を1つ返す（高速版）。"""
    select = obs.select
    n  = len(select.option)
    lo = select.minCount
    hi = select.maxCount

    if n == 0:
        return []
    count = random.randint(lo, min(hi, n))
    if count == 0:
        return []
    return random.sample(range(n), count)

# ── Rollout ───────────────────────────────────────────────────────────────────

def _rollout(start_id: int, start_obs: Observation,
             my_index: int, depth: int, deadline: float) -> float:
    """
    start_id から始まるランダムロールアウト。
    start_id は呼び出し元が管理するため、このメソッド内では解放しない。
    作成した中間状態は終了時にすべて解放する。
    """
    cur_id   = start_id
    cur_obs  = start_obs
    created: list[int] = []  # このロールアウト内で作成した search_id

    try:
        for _ in range(depth):
            if time.time() > deadline:
                break
            if _is_game_over(cur_obs):
                return _game_result(cur_obs, my_index)

            action = _random_legal_action(cur_obs)
            if not action and cur_obs.select.minCount > 0:
                break

            try:
                ns = search_step(cur_id, action)
                created.append(ns.searchId)
                cur_id  = ns.searchId
                cur_obs = ns.observation
            except Exception:
                break
    finally:
        # 中間状態のメモリ解放（start_id は除く）
        for sid in created:
            try:
                search_release(sid)
            except Exception:
                pass

    if _is_game_over(cur_obs):
        return _game_result(cur_obs, my_index)
    return evaluate(cur_obs, my_index)

# ── Flat Monte Carlo Search ───────────────────────────────────────────────────

def _flat_mc_search(root_id: int, root_obs: Observation,
                    my_index: int, deadline: float) -> list[int]:
    """
    Flat Monte Carlo Search: 各即時行動をロールアウトで評価し、
    期待価値最大の行動を返す。

    RL的解釈:
      - Q(s, a) ≈ ロールアウトの平均報酬（モンテカルロ推定）
      - argmax_a Q(root, a) を選択
    """
    candidates = _legal_actions(root_obs)

    if len(candidates) == 1:
        return candidates[0]

    # 各候補の累積価値と評価回数
    values: dict[tuple, float] = {}
    counts: dict[tuple, int]   = {}
    for a in candidates:
        key = tuple(sorted(a))
        values[key] = 0.0
        counts[key] = 0

    # ラウンドロビンでロールアウトを繰り返す
    idx = 0
    while time.time() < deadline:
        action = candidates[idx % len(candidates)]
        key = tuple(sorted(action))
        idx += 1

        try:
            ns = search_step(root_id, action)
            v  = _rollout(ns.searchId, ns.observation, my_index,
                          ROLLOUT_DEPTH, deadline)
            search_release(ns.searchId)
            values[key] += v
            counts[key] += 1
        except Exception:
            pass

    # 最大期待価値の行動を選択
    best_action = candidates[0]
    best_value  = float('-inf')

    for action in candidates:
        key = tuple(sorted(action))
        if counts[key] > 0:
            avg = values[key] / counts[key]
            if avg > best_value:
                best_value  = avg
                best_action = action

    log.debug("MC search: %d rollouts, best_value=%.3f", idx, best_value)
    return best_action

# ── Opponent deck prediction ──────────────────────────────────────────────────

_MY_DECK:   list[int] = []
_OPP_SEEN:  list[int] = []  # 相手が公開したカードID


def _update_opp_seen(logs) -> None:
    """ログから相手の公開カードを記録する。"""
    for entry in logs:
        cid = getattr(entry, 'cardId', None)
        if cid is not None and cid not in _OPP_SEEN:
            _OPP_SEEN.append(cid)


def _predict_opp_deck(count: int) -> list[int]:
    """
    相手のデッキを予測する。
    公開済みカードを先頭に置き、残りは自デッキで補完する。
    count 枚必要な場合に必ず count 枚を返すよう保証する。
    """
    base = list(_OPP_SEEN)
    filler = _MY_DECK[:]
    random.shuffle(filler)
    combined = base + filler
    # 足りない場合は自デッキを繰り返して補完
    while len(combined) < count:
        combined.extend(_MY_DECK)
    return combined[:count]


def _predict_opp_hand(count: int, deck_count: int) -> list[int]:
    pool = _predict_opp_deck(count + deck_count)
    return pool[:count]


def _predict_opp_prize(count: int, deck_count: int, hand_count: int) -> list[int]:
    pool = _predict_opp_deck(count + deck_count + hand_count)
    return pool[:count]


def _predict_my_prize(me_state) -> list[int]:
    """自分のサイドカードを予測する（表向きは確定、裏向きは推測）。"""
    known        = [c.id for c in me_state.prize if c is not None]
    unknown_cnt  = len(me_state.prize) - len(known)
    if unknown_cnt <= 0:
        return known
    candidates = [c for c in _MY_DECK if c not in known]
    random.shuffle(candidates)
    return known + candidates[:unknown_cnt]

# ── Heuristic fallback ────────────────────────────────────────────────────────

def _heuristic(obs: Observation) -> list[int]:
    """
    ルールベースのフォールバック行動選択。
    MC探索が失敗した場合に使用。
    """
    select = obs.select
    ctx    = select.context

    if ctx == SelectContext.MAIN:
        return _heuristic_main(obs)
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        return _choose_setup_active(obs)
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        return _choose_setup_bench(obs)

    # デフォルト: 最低限必要な数だけ先頭から選択
    return list(range(select.minCount))


def _heuristic_main(obs: Observation) -> list[int]:
    """メインフェーズの行動を優先順位に従って選択する。"""
    select = obs.select
    state  = obs.current

    evolve_opts:  list[int] = []
    attach_opts:  list[int] = []
    ability_opts: list[int] = []
    attack_opts:  list[int] = []
    play_opts:    list[int] = []
    end_opts:     list[int] = []

    for i, opt in enumerate(select.option):
        t = opt.type
        if   t == OptionType.EVOLVE:   evolve_opts.append(i)
        elif t == OptionType.ATTACH:   attach_opts.append(i)
        elif t == OptionType.ABILITY:  ability_opts.append(i)
        elif t == OptionType.ATTACK:   attack_opts.append(i)
        elif t == OptionType.PLAY:     play_opts.append(i)
        elif t == OptionType.END:      end_opts.append(i)

    # 優先順位: 進化 > エネルギー付け > 特性 > 攻撃 > カードプレイ > ターン終了
    if evolve_opts:
        return [evolve_opts[0]]
    if not state.energyAttached and attach_opts:
        return [_pick_attach(obs, attach_opts)]
    if ability_opts:
        return [ability_opts[0]]
    if attack_opts:
        return [_pick_attack(obs, attack_opts)]
    if play_opts:
        return [play_opts[0]]
    if end_opts:
        return [end_opts[0]]
    return [0]


def _pick_attach(obs: Observation, opts: list[int]) -> int:
    """バトルポケモンへのエネルギー付けを優先する。"""
    select = obs.select
    for i in opts:
        opt = select.option[i]
        if getattr(opt, 'inPlayArea', None) == 4:  # ACTIVE = 4
            return i
    return opts[0]


def _pick_attack(obs: Observation, opts: list[int]) -> int:
    """最大ダメージ（できれば相手をきぜつ）の攻撃を選ぶ。"""
    select    = obs.select
    state     = obs.current
    my_index  = state.yourIndex
    opp       = state.players[1 - my_index]

    opp_hp = 0
    if opp.active and opp.active[0] is not None:
        opp_hp = opp.active[0].hp

    best_i, best_score = opts[0], -1

    for i in opts:
        opt = select.option[i]
        dmg = 0
        if opt.attackId is not None and opt.attackId in _ATTACK_DB:
            dmg = _ATTACK_DB[opt.attackId].damage
        # きぜつを取れる攻撃を最優先
        score = dmg + (1000 if opp_hp > 0 and dmg >= opp_hp else 0)
        if score > best_score:
            best_score, best_i = score, i

    return best_i


def _choose_setup_active(obs: Observation) -> list[int]:
    """セットアップ: バトルポケモンを HP が高い順に選ぶ。"""
    select = obs.select
    if not select.option:
        return []
    best_i, best_hp = 0, -1
    for i, opt in enumerate(select.option):
        cid = getattr(opt, 'cardId', None)
        if cid is not None and cid in _CARD_DB:
            hp = _CARD_DB[cid].hp
            if hp > best_hp:
                best_hp, best_i = hp, i
    return [best_i]


def _choose_setup_bench(obs: Observation) -> list[int]:
    """セットアップ: ベンチをできるだけ展開する。"""
    select = obs.select
    count  = min(select.maxCount, len(select.option))
    return list(range(count))

# ── Main agent ────────────────────────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    """
    Pokemon TCG AI エージェント。

    初回呼び出し: デッキ選択（60枚のカードIDリストを返す）
    通常ターン : Flat Monte Carlo Search でベスト行動を選択
                 失敗時はヒューリスティックにフォールバック

    Returns:
        list[int]: 選択するオプションのインデックスリスト
    """
    global _MY_DECK, _OPP_SEEN

    _load_databases()

    obs: Observation = to_observation_class(obs_dict)

    # ── 初回: デッキ選択 ──────────────────────────────────────────────────
    if obs.select is None:
        deck      = read_deck_csv()
        _MY_DECK  = deck[:]
        _OPP_SEEN = []
        return deck

    if not _MY_DECK:
        _MY_DECK = read_deck_csv()

    # 相手の公開カードをログから更新
    if obs.logs:
        _update_opp_seen(obs.logs)

    # 盤面情報がない場合はヒューリスティック
    if obs.current is None:
        return _heuristic(obs)

    state    = obs.current
    my_index = state.yourIndex
    me       = state.players[my_index]
    opp      = state.players[1 - my_index]

    # ── 相手の非公開情報を予測 ────────────────────────────────────────────
    opp_deck_count  = opp.deckCount
    opp_hand_count  = opp.handCount
    opp_prize_count = len(opp.prize)

    pred_opp_deck  = _predict_opp_deck(opp_deck_count)
    pred_opp_hand  = _predict_opp_hand(opp_hand_count, opp_deck_count)
    pred_opp_prize = _predict_opp_prize(opp_prize_count, opp_deck_count, opp_hand_count)
    pred_my_prize  = _predict_my_prize(me)
    pred_my_deck   = _MY_DECK[:]

    # 裏向きのバトルポケモンを予測
    pred_opp_active: list[int] = []
    if opp.active and len(opp.active) > 0 and opp.active[0] is None:
        basics = [c for c in pred_opp_deck if c in _CARD_DB and _CARD_DB[c].basic]
        pred_opp_active = [basics[0]] if basics else (pred_opp_deck[:1] if pred_opp_deck else [])

    # ── Monte Carlo 探索 ─────────────────────────────────────────────────
    try:
        root_state = search_begin(
            obs,
            pred_my_deck,
            pred_my_prize,
            pred_opp_deck,
            pred_opp_prize,
            pred_opp_hand,
            pred_opp_active,
        )

        deadline = time.time() + MCTS_TIME_BUDGET_SEC
        action   = _flat_mc_search(
            root_state.searchId,
            root_state.observation,
            my_index,
            deadline,
        )
        search_end()
        return action

    except Exception as e:
        log.warning("MC search failed (%s), falling back to heuristic.", e)
        try:
            search_end()
        except Exception:
            pass
        return _heuristic(obs)
