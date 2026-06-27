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
  メインアタッカー : Dragapult ex
                    Fantasy Ripper: 200 ダメ + ベンチ全体に 20 ダメカン
  ダメカンサポート : Munkidori (特性: 悪いおくすり → 相手 1 体に 30 ダメカン)
                    Chi-Yu     (特性: 邪悪な炎 → 相手ベンチ全体に 10 ダメカン)
                    Fezandipiti ex (特性: 毒まねき → ダメカン配置 + ベンチ展開)
  勝ち筋          : 特性ダメカン + Dragapult ex 攻撃で相手を削り、
                     特に相手の ex ポケモン（サイド 2 枚）を狙い打ちにする
"""

import os
import time
import random
import logging

from cg.api import (
    Observation, SelectContext, SelectType, OptionType, AreaType,
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

# ── 時間管理 ──────────────────────────────────────────────────────────────
# 制限時間はゲーム全体で 10 分 (600 秒)。これを「我々が 1 ゲームで使ってよい
# 総計算時間」とみなし、その範囲内で 1 手あたりにできる限り時間をかける。
#
#   GAME_TIME_CAP : 1 ゲームで使う総時間の上限 (600秒 - 安全マージン60秒)
#   TIME_BUDGET_SEC: 1 手の探索時間の上限（以前の 3.0s から大幅増）
#   PER_MOVE_MIN  : 1 手の探索時間の下限（終盤でも最低限は探索する）
#   RESERVE_DIV   : 残り時間を「想定残り手数」で割って 1 手の予算を決める除数。
#                   大きいほど 1 手を節約し、長い試合でも枯渇しにくい。
#
# 計測（diag_timing.py）では 1 ゲームで探索が走る決定は対ランダムで約 33 回、
# 長い接戦の自己対戦では player あたり 70〜120 回程度。
# 下の動的予算は「累積使用時間が GAME_TIME_CAP を絶対に超えない」ことを
# 数学的に保証する（budget <= remaining/RESERVE_DIV < remaining のため）。
GAME_TIME_CAP   = 540.0   # 1 ゲームで使う総探索時間の上限 (秒)
TIME_BUDGET_SEC = 10.0    # 1 手あたりの探索時間の上限 (秒) ← 旧 3.0s の 3 倍以上
PER_MOVE_MIN    = 1.0     # 1 手あたりの探索時間の下限 (秒)
RESERVE_DIV     = 40.0    # 残り時間を割る想定残り手数

ROLLOUT_DEPTH   = 30    # ロールアウトの最大ステップ数
MAX_CANDIDATES  = 12    # 1 回の MAIN 選択で試す候補行動の上限

# 1 ゲームでこれまでに探索へ費やした累積時間（デッキ宣言時にリセット）
_GAME_TIME_USED = 0.0
# 直近に観測したターン数（新ゲーム検出用）。ターンが巻き戻ったらリセット。
_LAST_TURN = -1


def _decision_budget() -> float:
    """
    この 1 手に割り当てる探索時間（秒）を返す。

    残り時間を想定残り手数で割って 1 手の予算を決める。
    早い段階では TIME_BUDGET_SEC（上限）まで使い、ゲームが進んで
    残り時間が減ると自動的に 1 手の予算を絞る。これにより:
      - 序盤〜中盤は 1 手に最大限の時間をかけて精度を上げる
      - 終盤でも最低 PER_MOVE_MIN は確保し、手が枯渇しない
      - 累積時間が GAME_TIME_CAP を超えない（タイムアウト回避）
    """
    remaining = GAME_TIME_CAP - _GAME_TIME_USED
    if remaining <= PER_MOVE_MIN:
        return max(0.0, remaining)          # 残りわずか: 残り時間ちょうどまで
    budget = remaining / RESERVE_DIV
    budget = max(PER_MOVE_MIN, min(TIME_BUDGET_SEC, budget))
    return min(budget, remaining)           # 絶対に残りを超えない

# ─────────────────────────────────────────────────────────────────────────────
# デッキ構成定義 (Dragapult ex デッキ)
# ─────────────────────────────────────────────────────────────────────────────

DREEPY_ID        = 119
DRAKLOAK_ID      = 120
DRAGAPULT_EX_ID  = 121
BLAZIKEN_EX_ID   = 326
TORCHIC_ID       = 410
COMBUSKEN_ID     = 411
MUNKIDORI_ID     = 112   # 特性: 悪いおくすり (相手 1 体に 30 ダメカン)
CHI_YU_ID        = 31    # 特性: 邪悪な炎 (相手ベンチ全体に 10 ダメカン)
FEZANDIPITI_EX_ID = 140  # 特性: 毒まねき (ダメカン + ベンチ展開)
RARE_CANDY_ID    = 1079
ULTRA_BALL_ID    = 1121
BUDDY_POFFIN_ID  = 1086

MAIN_EVOLINE = [DREEPY_ID, DRAKLOAK_ID, DRAGAPULT_EX_ID]
SUB_EVOLINE  = [TORCHIC_ID, COMBUSKEN_ID, BLAZIKEN_EX_ID]

# ─────────────────────────────────────────────────────────────────────────────
# カード辞書（起動時 1 度だけロード）
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
    if r == my_idx:      return  1.0
    if r == 1 - my_idx: return -1.0
    return 0.0


def _prize_value(card_id: int) -> int:
    """きぜつ時に相手が取るサイド枚数（ex/Mega は 2〜3 枚）。"""
    c = _CARD_DB.get(card_id)
    if c is None:       return 1
    if c.megaEx:        return 3
    if c.ex:            return 2
    return 1


def _damage_progress(poke) -> float:
    """
    ポケモンがきぜつにどれだけ近いかを 0〜1 で返す。
    ex ポケモンは倒すとサイドが多いため重みを大きくする。
    """
    if poke is None:
        return 0.0
    hp_ratio = poke.hp / max(poke.maxHp, 1)
    pv = _prize_value(poke.id)
    return pv * (1.0 - hp_ratio)

# ─────────────────────────────────────────────────────────────────────────────
# 評価関数
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(obs: Observation, my_idx: int) -> float:
    """
    盤面の価値を [-1, 1] で返す (my_idx 視点)。

    指標と比重:
      ① サイドカード残枚数差      (60%)  ← 最重要
      ② 相手ポケモンのダメカン進捗 (18%)  ← 特性ダメカンの価値をここで評価
      ③ バトルポケモン HP 比率差   (10%)
      ④ 進化段階ボーナス           ( 8%)  Dragapult ex / Blaziken ex が場にいると+
      ⑤ エネルギー進捗             ( 2%)
      ⑥ ベンチ展開差               ( 2%)

    ②を追加したことで、Munkidori / Chi-Yu / Fezandipiti ex の特性によって
    相手にダメカンを乗せるほど評価値が上がる → MC 探索が特性使用を学習する。
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

    # ② ダメカン進捗差分
    #    相手のポケモンへのダメカン蓄積 - 自分のポケモンへのダメカン蓄積
    def total_damage_progress(ps) -> float:
        total = 0.0
        if ps.active and ps.active[0] is not None:
            total += _damage_progress(ps.active[0]) * 1.0   # バトルは重要度高め
        for p in ps.bench:
            total += _damage_progress(p) * 0.6              # ベンチは少し軽め
        return total

    damage = (total_damage_progress(opp) - total_damage_progress(me)) * 0.18

    # ③ HP 比率差分
    def hp_r(ps) -> float:
        if not ps.active or ps.active[0] is None:
            return 0.0
        p = ps.active[0]
        return p.hp / max(p.maxHp, 1)
    hp = (hp_r(me) - hp_r(opp)) * 0.10

    # ④ 進化段階ボーナス
    evo_bonus = 0.0
    all_my = list(me.bench)
    if me.active and me.active[0] is not None:
        all_my.append(me.active[0])
    for p in all_my:
        if p.id == DRAGAPULT_EX_ID:  evo_bonus += 0.05
        elif p.id == DRAKLOAK_ID:    evo_bonus += 0.02
        elif p.id == BLAZIKEN_EX_ID: evo_bonus += 0.03

    # ⑤ エネルギー進捗
    energy = 0.0
    if me.active and me.active[0] is not None:
        energy = min(len(me.active[0].energies), 3) * 0.007

    # ⑥ ベンチ差
    bench = (len(me.bench) - len(opp.bench)) * 0.02

    return max(-1.0, min(1.0,
        prize * 0.60
        + damage
        + hp
        + evo_bonus
        + energy
        + bench
    ))

# ─────────────────────────────────────────────────────────────────────────────
# ターゲット選択ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _get_poke_from_option(opt, state, my_idx: int):
    """
    Option（CARD タイプ）が指すポケモンを返す。
    相手のポケモンなら (poke, prize_value) を、
    自分のなら None を返す（ダメカンを乗せたくない）。
    """
    area  = getattr(opt, 'area',        None)
    idx   = getattr(opt, 'index',       None)
    pidx  = getattr(opt, 'playerIndex', None)

    if pidx is None or area is None:
        return None, 0

    # 自分のポケモンならスキップ
    if pidx == my_idx:
        return None, 0

    ps = state.players[pidx]
    poke = None
    if area == AreaType.ACTIVE:
        poke = ps.active[0] if ps.active else None
    elif area == AreaType.BENCH and idx is not None and idx < len(ps.bench):
        poke = ps.bench[idx]

    if poke is None:
        return None, 0

    return poke, _prize_value(poke.id)


def _rank_damage_targets(obs: Observation) -> list[int]:
    """
    ダメカン配置先の候補を「倒したときの価値が高い順」にソートして返す。

    優先順位:
      1. HP が残り少なく、きぜつさせられる相手のポケモン
         （特に ex ポケモンはサイド 2 枚なので最優先）
      2. HP がまだあっても ex ポケモン（ダメカンを積んでおく価値が高い）
      3. それ以外の相手ポケモン
    """
    sel   = obs.select
    state = obs.current
    my    = state.yourIndex

    def score(i: int) -> float:
        poke, pv = _get_poke_from_option(sel.option[i], state, my)
        if poke is None:
            return -9999.0
        hp_ratio = poke.hp / max(poke.maxHp, 1)
        # きぜつに近いほど高評価 + ex は特別ボーナス
        return pv * (1.0 - hp_ratio) + pv * 0.05

    return sorted(range(len(sel.option)), key=score, reverse=True)

# ─────────────────────────────────────────────────────────────────────────────
# 合法手の列挙
# ─────────────────────────────────────────────────────────────────────────────

def _legal_actions(obs: Observation, max_n: int = MAX_CANDIDATES) -> list[list[int]]:
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
    """ロールアウト用: 高速ランダム合法手。"""
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
    start_id の状態から最大 depth ステップのランダムロールアウトを行い
    終端の評価値を返す。作成した中間状態はすべて解放する。
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
      1. 合法手を列挙（最大 MAX_CANDIDATES 個）
      2. 各手を 1 つ試してロールアウト（ランダムプレイを ROLLOUT_DEPTH 手続ける）
      3. 平均報酬 Q(s,a) をモンテカルロ推定
      4. 時間切れになったら Q 最大の手を返す

    RL との対応:
      Q(s,a) ≈ E[累積報酬 | 状態 s, 行動 a]（モンテカルロ推定）
      evaluate() が改善されたことで、特性でダメカンを乗せた局面を
      高く評価するようになり、MC 探索が特性活用を学習する。
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
            v  = _rollout(ns.searchId, ns.observation, my_idx, ROLLOUT_DEPTH, deadline)
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

    log.debug("MC: %d rollouts, best Q=%.3f", idx, best_v)
    return best_a

# ─────────────────────────────────────────────────────────────────────────────
# 相手デッキ予測
# ─────────────────────────────────────────────────────────────────────────────

_MY_DECK:  list[int] = []
_OPP_SEEN: list[int] = []


def _update_opp_seen(logs) -> None:
    for entry in logs:
        cid = getattr(entry, 'cardId', None)
        if cid is not None and cid not in _OPP_SEEN:
            _OPP_SEEN.append(cid)


def _pred_opp_deck(count: int) -> list[int]:
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
    MC 探索が失敗した場合のルールベース行動選択。
    SelectContext ごとに専用ロジックを持つ。
    """
    sel = obs.select
    ctx = sel.context

    # メインフェーズ
    if ctx == SelectContext.MAIN:
        return _h_main(obs)

    # セットアップ
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        return _h_setup_active(obs)
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        return _h_setup_bench(obs)

    # ─── 特性・効果に関わるコンテキスト ───────────────────────────────────
    # ダメカン配置: Munkidori / Chi-Yu / Fezandipiti ex の特性発動時
    if ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
        return _h_damage_counter(obs)

    # ダメカン個数選択
    if ctx == SelectContext.DAMAGE_COUNTER_COUNT:
        return _h_count_max(obs)

    # 効果の対象選択（ダメカン以外の効果対象）
    if ctx == SelectContext.EFFECT_TARGET:
        return _h_effect_target(obs)

    # ─── その他の頻出コンテキスト ─────────────────────────────────────────
    if ctx in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        return _h_to_bench(obs)

    if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK_BOTTOM):
        return _h_discard(obs)

    # ダメカン除去（回復対象）: 自分で HP が低いポケモンを優先回復
    if ctx in (SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.HEAL):
        return _h_heal_target(obs)

    # YES/NO 選択: 基本的に YES（特性・効果を発動する）
    if ctx in (SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT,
               SelectContext.MULLIGAN, SelectContext.MORE_DEVOLVE):
        return _h_yes_no(obs, prefer_yes=True)

    # COIN: 表を選ぶ（有利な場合が多い）
    if ctx == SelectContext.COIN_HEAD:
        return _h_yes_no(obs, prefer_yes=True)

    # 手札に加える / デッキに戻す: 進化ラインを優先
    if ctx == SelectContext.TO_HAND:
        return _h_to_hand(obs)

    # 捨て札選択
    if ctx == SelectContext.DISCARD_ENERGY_CARD:
        return list(range(sel.minCount))

    # デフォルト: 必要最低数だけ先頭から
    return list(range(sel.minCount))

# ─────────────────────────────────────────────────────────────────────────────
# ヒューリスティック: 各コンテキストの実装
# ─────────────────────────────────────────────────────────────────────────────

def _h_main(obs: Observation) -> list[int]:
    """
    メインフェーズの優先順位:
      進化 > 特性 > エネルギー付け > 攻撃 > カードプレイ > ターン終了

    ポイント:
      - 特性 (ABILITY) を攻撃より先に使う
        → Munkidori でダメカンを乗せてから攻撃で確定 KO を狙う
      - ターン中に複数の特性が使えるなら、agent() が何度も呼ばれるので
        毎回 1 つずつ使っていく（最初に見つかった特性を使う）
    """
    sel, st = obs.select, obs.current
    evo, abi, att, atk, play, end = [], [], [], [], [], []

    for i, opt in enumerate(sel.option):
        t = opt.type
        if   t == OptionType.EVOLVE:  evo.append(i)
        elif t == OptionType.ABILITY: abi.append(i)
        elif t == OptionType.ATTACH:  att.append(i)
        elif t == OptionType.ATTACK:  atk.append(i)
        elif t == OptionType.PLAY:    play.append(i)
        elif t == OptionType.END:     end.append(i)

    if evo:
        return [_best_evolve(obs, evo)]
    if abi:
        # 特性を攻撃より先に使う（ダメカン先積み戦術）
        return [_best_ability(obs, abi)]
    if not st.energyAttached and att:
        return [_best_attach(obs, att)]
    if atk:
        return [_best_attack(obs, atk)]
    if play:
        return [play[0]]
    if end:
        return [end[0]]
    return [0]


def _best_evolve(obs: Observation, opts: list[int]) -> int:
    """進化優先順位: Dragapult ex > Drakloak > Blaziken ex > 他。"""
    sel      = obs.select
    priority = {DRAGAPULT_EX_ID: 100, DRAKLOAK_ID: 80,
                BLAZIKEN_EX_ID: 60,   COMBUSKEN_ID: 40}
    best_i, best_p = opts[0], -1
    for i in opts:
        cid = getattr(sel.option[i], 'cardId', None)
        if cid is not None:
            p = priority.get(cid, 1)
            if p > best_p:
                best_p, best_i = p, i
    return best_i


def _best_ability(obs: Observation, opts: list[int]) -> int:
    """
    特性の優先順位:
      Munkidori > Chi-Yu > Fezandipiti ex > 他
    （ダメカン系特性を優先して早めに発動する）
    """
    sel      = obs.select
    priority = {MUNKIDORI_ID: 100, CHI_YU_ID: 90, FEZANDIPITI_EX_ID: 80}
    best_i, best_p = opts[0], -1
    for i in opts:
        cid = getattr(sel.option[i], 'cardId', None)
        p   = priority.get(cid, 1) if cid else 1
        if p > best_p:
            best_p, best_i = p, i
    return best_i


def _best_attach(obs: Observation, opts: list[int]) -> int:
    """
    エネルギー付け先優先順位:
      Dragapult ex > Drakloak > Blaziken ex > バトルポケモン > 他
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
    sel   = obs.select
    st    = obs.current
    opp   = st.players[1 - st.yourIndex]
    opp_hp = (opp.active[0].hp
               if opp.active and opp.active[0] is not None else 0)

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


def _h_damage_counter(obs: Observation) -> list[int]:
    """
    ダメカン配置先を選ぶ（Munkidori・Chi-Yu・Fezandipiti ex の特性発動時など）。

    戦略:
      1. きぜつさせられる相手ポケモンを最優先（特に ex ポケモン）
      2. すでにダメカンが乗っている ex ポケモンを優先
      3. それ以外は HP が低い相手ポケモン

    「誰に乗せるか」は _rank_damage_targets() でスコア計算する。
    """
    sel   = obs.select
    lo    = sel.minCount
    hi    = sel.maxCount
    n     = len(sel.option)
    count = min(hi, n)

    if count == 0:
        return []

    ranked = _rank_damage_targets(obs)
    return ranked[:count]


def _h_count_max(obs: Observation) -> list[int]:
    """
    個数選択（DAMAGE_COUNTER_COUNT など）: 最大値を選ぶ。
    ダメカンはできるだけ多く乗せた方が有利なため。
    """
    sel = obs.select
    # NUMBER オプションの中から最大値を選ぶ
    if not sel.option:
        return [0]
    # NUMBER タイプはインデックスが 0..n-1 に対応
    return [len(sel.option) - 1]


def _h_effect_target(obs: Observation) -> list[int]:
    """
    効果の対象選択。
    ダメカン系効果なら _h_damage_counter と同じ戦略を使う。
    """
    sel   = obs.select
    lo    = sel.minCount
    count = min(sel.maxCount, len(sel.option))
    if count == 0:
        return []
    ranked = _rank_damage_targets(obs)
    result = ranked[:count]
    return result if len(result) >= lo else list(range(lo))


def _h_setup_active(obs: Observation) -> list[int]:
    """
    セットアップ: バトルポケモン選択。
    Dreepy を壁役にする（HP が低く、Dragapult ex の進化元としても使える）。
    Dreepy がなければ HP が最低のポケモン（メインアタッカーを温存）。
    """
    sel = obs.select
    if not sel.option:
        return []
    for i, opt in enumerate(sel.option):
        if getattr(opt, 'cardId', None) == DREEPY_ID:
            return [i]
    best_i, best_hp = 0, float('inf')
    for i, opt in enumerate(sel.option):
        cid = getattr(opt, 'cardId', None)
        if cid in _CARD_DB and _CARD_DB[cid].hp < best_hp:
            best_hp, best_i = _CARD_DB[cid].hp, i
    return [best_i]


def _h_setup_bench(obs: Observation) -> list[int]:
    """セットアップ: ベンチを最大まで展開する。"""
    sel   = obs.select
    count = min(sel.maxCount, len(sel.option))
    return list(range(count))


def _h_to_bench(obs: Observation) -> list[int]:
    """
    ベンチへの展開: Dreepy > Torchic > Munkidori > 他。
    デッキのパーツをベンチに揃えてから進化・特性を使う。
    """
    sel   = obs.select
    count = min(sel.maxCount, len(sel.option))
    if count == 0:
        return []
    prio = {DREEPY_ID: 100, TORCHIC_ID: 80, MUNKIDORI_ID: 60,
            CHI_YU_ID: 50, FEZANDIPITI_EX_ID: 40}
    ranked = sorted(range(len(sel.option)),
                    key=lambda i: prio.get(getattr(sel.option[i], 'cardId', 0), 1),
                    reverse=True)
    return ranked[:count]


def _h_discard(obs: Observation) -> list[int]:
    """
    捨て札選択: 進化元（Dreepy・Torchic）を最後まで温存する。
    価値の低いカード（エネルギー、グッズ）から捨てる。
    """
    sel   = obs.select
    count = min(sel.maxCount, len(sel.option))
    if count == 0:
        return []

    # 捨てにくさスコア（高いほど捨てたくない）
    def keep_score(i: int) -> int:
        cid = getattr(sel.option[i], 'cardId', None)
        if cid is None:
            return 10
        c = _CARD_DB.get(cid)
        if c is None:
            return 10
        if cid in (DREEPY_ID, TORCHIC_ID, MUNKIDORI_ID):
            return 100   # 進化元・特性持ちは温存
        if cid in (DRAGAPULT_EX_ID, DRAKLOAK_ID, BLAZIKEN_EX_ID):
            return 90    # 進化先も温存
        if c.basic and not (c.stage1 or c.stage2):
            return 20    # 基本エネルギー等はわりと捨てやすい
        return 30        # グッズ・サポートは中程度

    # keep_score が低い順（捨てやすいもの）から count 枚選ぶ
    ranked = sorted(range(len(sel.option)), key=keep_score)
    return ranked[:count]


def _h_heal_target(obs: Observation) -> list[int]:
    """
    回復対象の選択: HP 残量が最も少ない自分のポケモンを回復する。
    """
    sel   = obs.select
    count = min(sel.maxCount, len(sel.option))
    if count == 0:
        return []

    st  = obs.current
    my  = st.yourIndex
    ps  = st.players[my]

    def heal_score(i: int) -> float:
        opt  = sel.option[i]
        area = getattr(opt, 'area',  None)
        idx  = getattr(opt, 'index', None)
        pidx = getattr(opt, 'playerIndex', None)
        if pidx != my:
            return -1.0
        poke = None
        if area == AreaType.ACTIVE and ps.active:
            poke = ps.active[0]
        elif area == AreaType.BENCH and idx is not None and idx < len(ps.bench):
            poke = ps.bench[idx]
        if poke is None:
            return 0.0
        hp_ratio = poke.hp / max(poke.maxHp, 1)
        pv = _prize_value(poke.id)
        return pv * (1.0 - hp_ratio)   # HP が低いほど高スコア

    ranked = sorted(range(len(sel.option)), key=heal_score, reverse=True)
    return ranked[:count]


def _h_to_hand(obs: Observation) -> list[int]:
    """
    手札に加えるカードの選択: Dragapult ex / Drakloak / Munkidori を優先。
    """
    sel   = obs.select
    count = min(sel.maxCount, len(sel.option))
    if count == 0:
        return []
    prio = {DRAGAPULT_EX_ID: 100, DRAKLOAK_ID: 90, MUNKIDORI_ID: 80,
            DREEPY_ID: 70, BLAZIKEN_EX_ID: 60, TORCHIC_ID: 50}
    ranked = sorted(range(len(sel.option)),
                    key=lambda i: prio.get(getattr(sel.option[i], 'cardId', 0), 1),
                    reverse=True)
    return ranked[:count]


def _h_yes_no(obs: Observation, prefer_yes: bool) -> list[int]:
    """YES/NO 選択。prefer_yes=True なら YES (index 0 が YES の慣例)。"""
    sel = obs.select
    target_type = OptionType.YES if prefer_yes else OptionType.NO
    for i, opt in enumerate(sel.option):
        if opt.type == target_type:
            return [i]
    return [0]

# ─────────────────────────────────────────────────────────────────────────────
# メインエージェント
# ─────────────────────────────────────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    """
    ポケモンカードゲーム AI エージェント（ドラパルト ex デッキ特化版）。

    初回 (obs.select is None): デッキ 60 枚を返す
    通常ターン              : Flat MC Search で最良手を選択
                              失敗時はヒューリスティックにフォールバック

    特性への対応:
      - 評価関数が相手ポケモンへのダメカン蓄積を報酬として評価
        → MC 探索が「特性でダメカンを乗せる手」を自然に学習
      - ヒューリスティックが DAMAGE_COUNTER 系コンテキストに対応
        → Munkidori / Chi-Yu 発動時に最適なターゲットを選択
      - MAIN フェーズで特性を攻撃より先に使う（ダメカン先積み戦術）
    """
    global _MY_DECK, _OPP_SEEN, _GAME_TIME_USED, _LAST_TURN
    _load_databases()

    obs: Observation = to_observation_class(obs_dict)

    # ─── 初回: デッキ宣言 ────────────────────────────────────────────────
    if obs.select is None:
        deck      = read_deck_csv()
        _MY_DECK  = deck[:]
        _OPP_SEEN = []
        _GAME_TIME_USED = 0.0          # ゲーム開始: 時間予算をリセット
        _LAST_TURN = -1
        return deck

    # 新ゲーム検出: ターンが巻き戻ったら時間予算をリセット
    # （デッキを battle_start に直接渡すローカルテスト等で必要）
    if obs.current is not None:
        turn = obs.current.turn
        if turn < _LAST_TURN:
            _GAME_TIME_USED = 0.0
            _OPP_SEEN = []
        _LAST_TURN = turn

    if not _MY_DECK:
        _MY_DECK = read_deck_csv()

    if obs.logs:
        _update_opp_seen(obs.logs)

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

    pred_opp_active: list[int] = []
    if opp.active and opp.active[0] is None:
        basics = [c for c in pred_opp_deck
                  if c in _CARD_DB and _CARD_DB[c].basic]
        pred_opp_active = [basics[0]] if basics else pred_opp_deck[:1]

    # ─── Flat Monte Carlo Search ─────────────────────────────────────────
    # この 1 手に割り当てる時間を動的に決定（累積が GAME_TIME_CAP を超えない）
    budget = _decision_budget()
    t_start = time.time()
    try:
        root = search_begin(
            obs,
            pred_my_deck, pred_my_prize,
            pred_opp_deck, pred_opp_prize, pred_opp_hand,
            pred_opp_active,
        )
        deadline = t_start + budget
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

    finally:
        # 実際に消費した時間を累積（成功・失敗・例外いずれの経路でも必ず計上）
        _GAME_TIME_USED += time.time() - t_start
