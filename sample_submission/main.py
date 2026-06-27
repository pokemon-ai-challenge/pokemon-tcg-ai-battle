# -*- coding: utf-8 -*-
"""
main.py  -  Dragapult ex Deck AI  (Flat Monte Carlo Search)

【強化学習的アプローチ】
  State  : Observation（盤面・手札・ログ）
  Action : list[int]（選択肢インデックスのリスト）
  Reward : +1 勝利 / -1 敗北 / 途中 = 評価関数値
  手法   : Flat Monte Carlo Search
             各合法手をロールアウトで評価 → 期待価値最大の手を選択

【時間制限】
  ゲーム全体 10 分 (600 秒) 以内。
  MAIN フェーズ 1 決定あたり最大 3 秒使用。
  約 20～30 MAIN 決定 × 3 秒 = 60～90 秒 ≪ 600 秒 なので余裕あり。

【デッキ戦略 (ドラパルト ex デッキ)】
  メインアタッカー : Dragapult ex (Dreepy→Drakloak→Dragapult ex)
  サブアタッカー   : Blaziken ex (Torchic→Combusken→Blaziken ex)
  サポート役       : Munkidori / Fezandipiti ex / Chi-Yu
  目標             : Dreepy をバトル場に置き、ベンチで Dragapult ex に育てて
                     Fantasy Ripper (200 ダメ + ベンチへのダメカン) で押し切る
"""

import os
import time
import random
import logging

from cg.api import (
    Observation, SelectContext, OptionType, AreaType,
    to_observation_class, all_card_data, all_attack, CardData, Attack,
    search_begin, search_step, search_end, search_release,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ハイパーパラメータ
# ─────────────────────────────────────────────────────────────────────────────

# MAIN フェーズ 1 決定あたりの探索時間上限 (秒)
# 10 分 / ゲーム内の MAIN 決定数 (~30) で割り算 → 余裕を持って 3 秒
TIME_BUDGET_SEC = 3.0

# ロールアウトの最大ステップ数（深くするほど精度が上がるが遅くなる）
ROLLOUT_DEPTH = 30

# 1 回の MAIN 選択で試す候補行動の上限
MAX_CANDIDATES = 12

# ─────────────────────────────────────────────────────────────────────────────
# デッキ構成定義 (Dragapult ex デッキ)
# ─────────────────────────────────────────────────────────────────────────────

# カード ID → 役割 マッピング
DREEPY_ID       = 119   # 基本ポケモン（進化元）
DRAKLOAK_ID     = 120   # 1 進化（進化中間体）
DRAGAPULT_EX_ID = 121   # 2 進化（メインアタッカー）
BLAZIKEN_EX_ID  = 326   # サブアタッカー
TORCHIC_ID      = 410   # Blaziken 進化元
COMBUSKEN_ID    = 411   # Blaziken 中間体
RARE_CANDY_ID   = 1079  # 進化スキップアイテム（Drakloak を飛ばす）
ULTRA_BALL_ID   = 1121  # ポケモンサーチ
BUDDY_POFFIN_ID = 1086  # 基本ポケモン 2 枚サーチ

# メインアタッカーに育てたいポケモン（進化順）
MAIN_EVOLINE  = [DREEPY_ID, DRAKLOAK_ID, DRAGAPULT_EX_ID]
SUB_EVOLINE   = [TORCHIC_ID, COMBUSKEN_ID, BLAZIKEN_EX_ID]

# ─────────────────────────────────────────────────────────────────────────────
# カード辞書（起動時に 1 度だけロード）
# ─────────────────────────────────────────────────────────────────────────────

_CARD_DB:   dict[int, CardData] = {}
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

# ─────────────────────────────────────────────────────────────────────────────
# デッキ読み込み
# ─────────────────────────────────────────────────────────────────────────────

def read_deck_csv() -> list[int]:
    """deck.csv から 60 枚のカード ID リストを返す。"""
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        lines = f.read().split("\n")
    return [int(lines[i]) for i in range(60)]

# ─────────────────────────────────────────────────────────────────────────────
# ゲーム状態ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def _is_over(obs: Observation) -> bool:
    return obs.current is not None and obs.current.result != -1


def _terminal_value(obs: Observation, my_idx: int) -> float:
    r = obs.current.result
    if r == my_idx:       return  1.0
    if r == 1 - my_idx:  return -1.0
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 評価関数
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(obs: Observation, my_idx: int) -> float:
    """
    盤面の価値を [-1, 1] で返す (my_idx 視点)。

    主要指標:
      ① サイドカード残枚数差  (比重 0.65)
           残り枚数が少ない = 勝利に近い
      ② バトルポケモン HP 比率差 (比重 0.15)
           自分のバトルポケモンが高 HP = 有利
      ③ 進化段階ボーナス (比重 0.12)
           Dragapult ex がバトル or ベンチにいると +ボーナス
      ④ エネルギー進捗 (比重 0.05)
           バトルポケモンのエネルギー数
      ⑤ ベンチ展開差 (比重 0.03)
    """
    if obs.current is None:
        return 0.0
    state = obs.current
    if state.result != -1:
        return _terminal_value(obs, my_idx)

    me  = state.players[my_idx]
    opp = state.players[1 - my_idx]

    # ① サイド差分
    prize = (len(opp.prize) - len(me.prize)) / 6.0

    # ② HP 比率
    def hp_r(ps) -> float:
        if not ps.active or ps.active[0] is None:
            return 0.0
        p = ps.active[0]
        return p.hp / max(p.maxHp, 1)
    hp = (hp_r(me) - hp_r(opp)) * 0.15

    # ③ 進化段階ボーナス (Dragapult ex が場にいると有利)
    evo_bonus = 0.0
    all_my = list(me.bench)
    if me.active and me.active[0] is not None:
        all_my.append(me.active[0])
    for p in all_my:
        if p.id == DRAGAPULT_EX_ID:
            evo_bonus += 0.06
        elif p.id == DRAKLOAK_ID:
            evo_bonus += 0.02
        elif p.id == BLAZIKEN_EX_ID:
            evo_bonus += 0.04

    # ④ エネルギー進捗
    energy = 0.0
    if me.active and me.active[0] is not None:
        energy = min(len(me.active[0].energies), 3) * 0.017

    # ⑤ ベンチ差
    bench = (len(me.bench) - len(opp.bench)) * 0.03

    return max(-1.0, min(1.0,
        prize  * 0.65
        + hp
        + evo_bonus
        + energy
        + bench
    ))

# ─────────────────────────────────────────────────────────────────────────────
# 合法手の列挙
# ─────────────────────────────────────────────────────────────────────────────

def _legal_actions(obs: Observation, max_n: int = MAX_CANDIDATES) -> list[list[int]]:
    """
    現在の選択に対して合法な候補行動リストを返す。
    各候補は list[int]（選択するインデックスのリスト）。
    """
    sel = obs.select
    n, lo, hi = len(sel.option), sel.minCount, sel.maxCount

    if n == 0:
        return [[]] if lo == 0 else []

    cands: list[list[int]] = []

    if hi <= 1:
        for i in range(n):
            cands.append([i])
        if lo == 0:
            cands.append([])
    elif lo == hi:
        if n <= hi:
            cands.append(list(range(n)))
        else:
            cands.append(list(range(hi)))
            for _ in range(min(max_n - 1, 5)):
                cands.append(random.sample(range(n), hi))
    else:
        for cnt in range(lo, min(hi + 1, lo + 4)):
            if n >= cnt:
                cands.append(list(range(cnt)))
                if n > cnt > 0:
                    cands.append(random.sample(range(n), cnt))

    seen: set[tuple] = set()
    unique: list[list[int]] = []
    for a in cands:
        k = tuple(sorted(a))
        if k not in seen:
            seen.add(k)
            unique.append(a)

    return unique[:max_n] if unique else [list(range(lo))]


def _rand_action(obs: Observation) -> list[int]:
    """ロールアウト用: 高速にランダム合法手を 1 つ返す。"""
    sel = obs.select
    n, lo, hi = len(sel.option), sel.minCount, sel.maxCount
    if n == 0:
        return []
    cnt = random.randint(lo, min(hi, n))
    return random.sample(range(n), cnt) if cnt > 0 else []

# ─────────────────────────────────────────────────────────────────────────────
# ランダムロールアウト
# ─────────────────────────────────────────────────────────────────────────────

def _rollout(start_id: int, start_obs: Observation,
             my_idx: int, depth: int, deadline: float) -> float:
    """
    start_id の状態から最大 depth ステップのランダムロールアウトを行い、
    終端の評価値を返す。
    作成した中間状態はすべて解放する（start_id は呼び出し元が管理）。
    """
    cur_id, cur_obs = start_id, start_obs
    created: list[int] = []
    try:
        for _ in range(depth):
            if time.time() > deadline or _is_over(cur_obs):
                break
            action = _rand_action(cur_obs)
            if not action and cur_obs.select.minCount > 0:
                break
            try:
                ns = search_step(cur_id, action)
                created.append(ns.searchId)
                cur_id, cur_obs = ns.searchId, ns.observation
            except Exception:
                break
    finally:
        for sid in created:
            try:
                search_release(sid)
            except Exception:
                pass

    if _is_over(cur_obs):
        return _terminal_value(cur_obs, my_idx)
    return evaluate(cur_obs, my_idx)

# ─────────────────────────────────────────────────────────────────────────────
# Flat Monte Carlo Search
# ─────────────────────────────────────────────────────────────────────────────

def _flat_mc(root_id: int, root_obs: Observation,
             my_idx: int, deadline: float) -> list[int]:
    """
    Flat Monte Carlo Search（平坦モンテカルロ探索）

    手順:
      1. 現在の手番から取れる合法手を列挙
      2. 各手をゲームエンジンで 1 手進める
      3. そこからランダムロールアウトで終局まで(or 深さ制限まで)シミュレート
      4. 期待報酬 Q(s, a) ≈ ロールアウト平均値 を推定
      5. Q が最大の手を返す

    RL との対応:
      Q(s, a) ≈ E[累積報酬 | 状態 s, 行動 a]  (モンテカルロ推定)
    """
    cands = _legal_actions(root_obs)
    if len(cands) == 1:
        return cands[0]

    vals:   dict[tuple, float] = {tuple(sorted(a)): 0.0 for a in cands}
    counts: dict[tuple, int]   = {tuple(sorted(a)): 0   for a in cands}

    idx = 0
    while time.time() < deadline:
        action = cands[idx % len(cands)]
        key    = tuple(sorted(action))
        idx   += 1
        try:
            ns = search_step(root_id, action)
            v  = _rollout(ns.searchId, ns.observation, my_idx,
                          ROLLOUT_DEPTH, deadline)
            search_release(ns.searchId)
            vals[key]   += v
            counts[key] += 1
        except Exception:
            pass

    best_a, best_v = cands[0], float('-inf')
    for a in cands:
        k = tuple(sorted(a))
        if counts[k] > 0 and (avg := vals[k] / counts[k]) > best_v:
            best_v, best_a = avg, a

    log.debug("MC: %d rollouts done, best Q=%.3f", idx, best_v)
    return best_a

# ─────────────────────────────────────────────────────────────────────────────
# 相手デッキ予測
# ─────────────────────────────────────────────────────────────────────────────

_MY_DECK:  list[int] = []
_OPP_SEEN: list[int] = []   # 対戦中に判明した相手カード ID


def _update_opp_seen(logs) -> None:
    for entry in logs:
        cid = getattr(entry, 'cardId', None)
        if cid is not None and cid not in _OPP_SEEN:
            _OPP_SEEN.append(cid)


def _pred_opp_deck(count: int) -> list[int]:
    """
    相手のデッキを予測する。
    既知の相手カード → 自デッキカードで補完 → 必要枚数を返す。
    """
    base   = list(_OPP_SEEN)
    filler = _MY_DECK[:]
    random.shuffle(filler)
    pool   = base + filler
    while len(pool) < count:
        pool.extend(_MY_DECK)
    return pool[:count]


def _pred_opp_hand(count: int, deck_count: int) -> list[int]:
    return _pred_opp_deck(count + deck_count)[:count]


def _pred_opp_prize(count: int, deck_count: int, hand_count: int) -> list[int]:
    return _pred_opp_deck(count + deck_count + hand_count)[:count]


def _pred_my_prize(me_state) -> list[int]:
    known = [c.id for c in me_state.prize if c is not None]
    unk   = len(me_state.prize) - len(known)
    if unk <= 0:
        return known
    rest = [c for c in _MY_DECK if c not in known]
    random.shuffle(rest)
    return known + rest[:unk]

# ─────────────────────────────────────────────────────────────────────────────
# ヒューリスティック（フォールバック）
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic(obs: Observation) -> list[int]:
    """
    MC 探索が失敗した場合に使うルールベースの行動選択。
    SelectContext ごとに最低限意味のある合法手を返す。
    """
    sel = obs.select
    ctx = sel.context

    if ctx == SelectContext.MAIN:
        return _h_main(obs)
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        return _h_setup_active(obs)
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        return _h_setup_bench(obs)
    if ctx in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        return _h_to_bench(obs)
    if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK_BOTTOM):
        return _h_discard(obs)

    # デフォルト: 必要最低数だけ先頭から選択
    return list(range(sel.minCount))


def _h_main(obs: Observation) -> list[int]:
    """
    メインフェーズの優先順位:
      進化 > エネルギー付け > 特性 > 攻撃 > カードプレイ > ターン終了

    ドラパルト ex デッキ特化:
      - Dreepy ではなく Dragapult ex / Drakloak を優先進化
      - エネルギーは Dragapult ex のいるポケモンに優先付け
    """
    sel, st = obs.select, obs.current
    evo, att, abi, atk, play, end = [], [], [], [], [], []

    for i, opt in enumerate(sel.option):
        t = opt.type
        if   t == OptionType.EVOLVE:   evo.append(i)
        elif t == OptionType.ATTACH:   att.append(i)
        elif t == OptionType.ABILITY:  abi.append(i)
        elif t == OptionType.ATTACK:   atk.append(i)
        elif t == OptionType.PLAY:     play.append(i)
        elif t == OptionType.END:      end.append(i)

    if evo:
        return [_best_evolve(obs, evo)]
    if not st.energyAttached and att:
        return [_best_attach(obs, att)]
    if abi:
        return [abi[0]]
    if atk:
        return [_best_attack(obs, atk)]
    if play:
        return [play[0]]
    if end:
        return [end[0]]
    return [0]


def _best_evolve(obs: Observation, opts: list[int]) -> int:
    """進化: Dragapult ex > Drakloak > Blaziken ex > その他 の優先順位。"""
    sel = obs.select
    priority = {DRAGAPULT_EX_ID: 100, DRAKLOAK_ID: 80, BLAZIKEN_EX_ID: 60, COMBUSKEN_ID: 40}
    best_i, best_p = opts[0], -1
    for i in opts:
        opt = sel.option[i]
        cid = getattr(opt, 'cardId', None)
        if cid is not None:
            p = priority.get(cid, 1)
            if p > best_p:
                best_p, best_i = p, i
    return best_i


def _best_attach(obs: Observation, opts: list[int]) -> int:
    """
    エネルギー付け先優先順位:
      Dragapult ex (バトル or ベンチ) > Drakloak > バトルポケモン > その他
    """
    sel = obs.select

    def priority(opt) -> int:
        area = getattr(opt, 'inPlayArea', None)
        idx  = getattr(opt, 'inPlayIndex', None)
        if area is None:
            return 0
        st = obs.current
        my = st.yourIndex
        ps = st.players[my]
        poke = None
        if area == AreaType.ACTIVE and ps.active:
            poke = ps.active[0]
        elif area == AreaType.BENCH and idx is not None and idx < len(ps.bench):
            poke = ps.bench[idx]
        if poke is None:
            return 1
        if poke.id == DRAGAPULT_EX_ID: return 100
        if poke.id == DRAKLOAK_ID:     return 80
        if poke.id == BLAZIKEN_EX_ID:  return 60
        if area == AreaType.ACTIVE:    return 40
        return 10

    return max(opts, key=lambda i: priority(sel.option[i]))


def _best_attack(obs: Observation, opts: list[int]) -> int:
    """最大ダメージの攻撃を選ぶ（きぜつ取れる攻撃を最優先）。"""
    sel    = obs.select
    st     = obs.current
    my     = st.yourIndex
    opp_hp = 0
    opp    = st.players[1 - my]
    if opp.active and opp.active[0] is not None:
        opp_hp = opp.active[0].hp

    best_i, best_s = opts[0], -1
    for i in opts:
        opt = sel.option[i]
        dmg = 0
        if opt.attackId is not None and opt.attackId in _ATTACK_DB:
            dmg = _ATTACK_DB[opt.attackId].damage
        s = dmg + (10000 if opp_hp > 0 and dmg >= opp_hp else 0)
        if s > best_s:
            best_s, best_i = s, i
    return best_i


def _h_setup_active(obs: Observation) -> list[int]:
    """
    セットアップ: バトルポケモン選択。
    ドラパルト ex デッキでは Dreepy を壁役にして HP 温存するのが基本。
    Dreepy がいれば Dreepy を選び、なければ HP 最低のポケモンを選ぶ
    （サブポケモンを温存するため）。
    """
    sel = obs.select
    if not sel.option:
        return []

    # Dreepy がいれば選ぶ（壁役）
    for i, opt in enumerate(sel.option):
        if getattr(opt, 'cardId', None) == DREEPY_ID:
            return [i]

    # なければ HP が最も低いポケモン（メインアタッカー候補を温存）
    best_i, best_hp = 0, float('inf')
    for i, opt in enumerate(sel.option):
        cid = getattr(opt, 'cardId', None)
        if cid is not None and cid in _CARD_DB:
            hp = _CARD_DB[cid].hp
            if hp < best_hp:
                best_hp, best_i = hp, i
    return [best_i]


def _h_setup_bench(obs: Observation) -> list[int]:
    """セットアップ: ベンチを最大まで展開する。"""
    sel   = obs.select
    count = min(sel.maxCount, len(sel.option))
    return list(range(count))


def _h_to_bench(obs: Observation) -> list[int]:
    """ベンチへの展開: Dreepy > Torchic > Munkidori > その他 の優先順位。"""
    sel   = obs.select
    count = min(sel.maxCount, len(sel.option))
    if count == 0:
        return []

    priority = {DREEPY_ID: 100, TORCHIC_ID: 80, 112: 60}   # 112 = Munkidori
    ranked = sorted(range(len(sel.option)),
                    key=lambda i: priority.get(getattr(sel.option[i], 'cardId', 0), 1),
                    reverse=True)
    return ranked[:count]


def _h_discard(obs: Observation) -> list[int]:
    """捨て札選択: 進化元を捨てないよう後回しにする。"""
    sel   = obs.select
    lo, hi = sel.minCount, sel.maxCount
    count = min(hi, len(sel.option))
    if count == 0:
        return []

    # 価値の低いカードを先に捨てる（エネルギー優先で捨てる）
    def discard_priority(i: int) -> int:
        cid = getattr(sel.option[i], 'cardId', None)
        if cid is None:
            return 50
        c = _CARD_DB.get(cid)
        if c is None:
            return 50
        if c.basic and not c.stage1 and not c.stage2:
            if cid in (DREEPY_ID, TORCHIC_ID):
                return 5   # 進化元は捨てにくい
            return 20      # 基本エネルギーなど
        return 50          # サポートやグッズは先に捨てる

    ranked = sorted(range(len(sel.option)), key=discard_priority, reverse=True)
    return ranked[:count]

# ─────────────────────────────────────────────────────────────────────────────
# メインエージェント
# ─────────────────────────────────────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    """
    ポケモンカードゲーム AI エージェント。

    呼び出しタイミングと返す値:
      初回 (obs.select is None) : 60 枚のカード ID リスト（デッキ宣言）
      通常ターン               : 選択肢インデックスのリスト

    処理フロー:
      1. ゲームエンジンの search_begin で探索ルートを初期化
      2. Flat MC Search で最良手を選択 (TIME_BUDGET_SEC 以内)
      3. 失敗・タイムアウト時はヒューリスティックにフォールバック
    """
    global _MY_DECK, _OPP_SEEN
    _load_databases()

    obs: Observation = to_observation_class(obs_dict)

    # ─── 初回: デッキ宣言 ────────────────────────────────────────────────
    if obs.select is None:
        deck     = read_deck_csv()
        _MY_DECK = deck[:]
        _OPP_SEEN = []
        return deck

    if not _MY_DECK:
        _MY_DECK = read_deck_csv()

    # 相手の公開カードをログから収集
    if obs.logs:
        _update_opp_seen(obs.logs)

    # 盤面情報がなければヒューリスティック
    if obs.current is None:
        return _heuristic(obs)

    st     = obs.current
    my_idx = st.yourIndex
    me     = st.players[my_idx]
    opp    = st.players[1 - my_idx]

    # ─── 相手の非公開情報を予測 ──────────────────────────────────────────
    opp_d = opp.deckCount
    opp_h = opp.handCount
    opp_p = len(opp.prize)

    pred_opp_deck  = _pred_opp_deck(opp_d)
    pred_opp_hand  = _pred_opp_hand(opp_h, opp_d)
    pred_opp_prize = _pred_opp_prize(opp_p, opp_d, opp_h)
    pred_my_prize  = _pred_my_prize(me)
    pred_my_deck   = _MY_DECK[:]

    # 裏向きバトルポケモンを予測（基本ポケモンから選ぶ）
    pred_opp_active: list[int] = []
    if opp.active and opp.active and opp.active[0] is None:
        basics = [c for c in pred_opp_deck if _CARD_DB.get(c, None) and _CARD_DB[c].basic]
        pred_opp_active = [basics[0]] if basics else pred_opp_deck[:1]

    # ─── Flat Monte Carlo Search ─────────────────────────────────────────
    try:
        root = search_begin(
            obs,
            pred_my_deck, pred_my_prize,
            pred_opp_deck, pred_opp_prize, pred_opp_hand,
            pred_opp_active,
        )
        deadline = time.time() + TIME_BUDGET_SEC
        action   = _flat_mc(root.searchId, root.observation, my_idx, deadline)
        search_end()
        return action

    except Exception as e:
        log.warning("MC failed (%s), using heuristic.", e)
        try:
            search_end()
        except Exception:
            pass
        return _heuristic(obs)
