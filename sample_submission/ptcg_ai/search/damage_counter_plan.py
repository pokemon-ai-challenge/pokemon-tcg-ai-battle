"""ファントムダイブのダメカン配置を「サイドを取り切るまでの攻撃回数の最小化」として解くプランナ。

背景
----
ドラパルトex のファントムダイブ(attackId 154)は「バトルポケモンに200 + 相手ベンチに
ダメカン6個を任意配分」。この配分は本番の探索パイプライン(``pipeline.py``)では扱えない。
パイプラインは ``SelectType.MAIN`` かつ ``maxCount == 1`` の意思決定にしか適用されず、
ダメカン配置は1個ずつの sub-select(``SelectContext.DAMAGE_COUNTER_ANY``)として降ってくる
ためである。つまり「探索が構造的に届かない領域」であり、明示的なプランナしか手段が無い。

既存の kiyotah 版ルールベース(``opponents/dragapult_rule_agent.py``)の配置ロジックは
「今すぐKOできる部分集合の列挙 + HP帯の固定ボーナス」で、複数ターン計画・サイド算術・
ボス連動が無い。本モジュールはそこを

    「残りサイドを取り切るまでに必要な自分の攻撃回数」を最小化する探索

として定式化し直す。HP<=60 や HP<=200 といった閾値は固定ボーナスとして与えない。
「その対象を倒すのに必要な資源(直接攻撃回数 / ダメカン個数)が減る」ことから
自然に価値が出る。

モデル(1「攻撃ターン」の抽象化)
--------------------------------
1ターンの行動 = (直接攻撃の対象 j, ダメカンの配分 x_i)。

* 直接攻撃は常にバトルポケモン(アクティブ)に当たる。``boss_budget`` があるとき限り、
  ベンチの1体をアクティブに引きずり出してから殴れる(ボスの指令)。
* ダメカンは **アクティブ以外**・``counter_protected`` でない・HPが残っている対象にのみ、
  1個(=10ダメージ)ずつ置ける。エンジンは置ける対象がある限り全部置かせるので、
  本モデルも「置ける対象がある限り置き切る」。
* HPが0以下になった対象は盤面から除去し、``remaining_prizes`` をその ``prize`` だけ減らす。
* アクティブがKOされた後の入れ替え先は相手が選ぶが、本モデルでは **楽観的に自分が選べる**
  ものとして扱う(緩和)。緩和は下界(LB)を過大評価しない方向なので安全側。

目的関数(辞書順)
------------------
1. horizon 内に勝てるか
2. 最短ターン数
3. LB(下界)の減少量
4. サイドを早く取れるか(ターン割引つきの取得サイド)
5. オーバーキルの小ささ
6. 除去した脅威度

決定性
------
探索・列挙はすべて ``key`` の昇順で行い、同点は先に評価した(=key が小さい)方を残す。
dict の反復順に依存する箇所は無いため、同一入力に対して常に同一出力になる。

仕様からの差分(意図的な追加。呼び出し側の既存コードは壊さない)
------------------------------------------------------------------
* ``TargetState.direct_immune``: ドラパルトの攻撃ダメージを受けない相手(Drednaw /
  Milotic ex / Sylveon / Crustle)がエンジンに実在するため、直接攻撃が無効な対象を
  表せないとモデルが嘘をつく。既定 False の追加フィールドなので既存の生成は不変。
* ``choose_next_counter(..., boss_budget=0)``: 既定0で仕様どおりの挙動。手札にボスの指令が
  あるときだけ呼び出し側が1を渡せるようにした(次ターン以降の直接攻撃対象の自由度)。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

# ---------------------------------------------------------------------------
# 公開データ構造
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetState:
    """相手の場のポケモン1体。盤面から抽出済みの純粋なデータ(cg の型に依存しない)。

    Attributes:
        key: 対象の安定した識別子(実戦では ``Pokemon.serial``)。
        hp10: **残り**HPを10単位(ダメカン1個分)で表した値。0以下はKO済み扱い。
        prize: この対象をKOしたときに取れるサイド枚数(ex=2, mega ex=3 など)。
        is_active: バトルポケモンなら True(ダメカンを置けない)。
        counter_protected: 特性/エネルギー等でダメカンを置けないなら True。
        threat: 除去価値(大きいほど倒したい)。最下位の同点処理にのみ使う。
        direct_immune: ドラパルトの攻撃ダメージを受け付けないなら True。
    """

    key: int
    hp10: int
    prize: int
    is_active: bool
    counter_protected: bool
    threat: float = 0.0
    direct_immune: bool = False


@dataclass(frozen=True)
class DamageState:
    """プランナが解く盤面。

    Attributes:
        targets: 相手の場(アクティブ+ベンチ)。
        remaining_prizes: 自分が勝つのに必要な残りサイド枚数(= 自分のサイドの残り枚数)。
        boss_budget: これから使えるボスの指令の枚数(ベンチをアクティブに引きずり出せる回数)。
    """

    targets: tuple[TargetState, ...]
    remaining_prizes: int
    boss_budget: int = 0


@dataclass(frozen=True)
class DamagePlanConfig:
    """探索パラメータ。

    Attributes:
        horizon: 先読みする攻撃ターン数(現在のターンを1と数える)。
        direct_damage10: 直接攻撃の打点(10単位)。ファントムダイブ200 → 20。
        counters: 1ターンに配置できるダメカン個数。
        max_states: 展開してよい状態数の上限。超えたら打ち切って LB 評価に落とす。
    """

    horizon: int = 2
    direct_damage10: int = 20
    counters: int = 6
    max_states: int = 200000


@dataclass(frozen=True)
class DamagePlan:
    """1ターン目の行動と、その先の見通し。

    Attributes:
        turns_to_win: horizon 内に取り切れるなら必要な攻撃ターン数。取り切れないなら None。
        lower_bound: 根の盤面の下界(楽観的に見積もった必要攻撃ターン数)。
        counter_alloc: 1ターン目のダメカン配分 ``((key, 個数), ...)``(key 昇順)。
        direct_target: 1ターン目の直接攻撃の対象 key(対象なしなら None)。
        projected_prizes: ターンごとに取れる見込みのサイド枚数。
    """

    turns_to_win: int | None
    lower_bound: int
    counter_alloc: tuple[tuple[int, int], ...]
    direct_target: int | None
    projected_prizes: tuple[int, ...]


DEFAULT_CONFIG = DamagePlanConfig()

# 取り切れないことを表す番兵(実盤面のサイドは最大6なので十分大きい)。
UNREACHABLE = 99

# 「サイドを早く取れるか」の割引率。後のターンの取得ほど価値を下げる。
_PRIZE_DISCOUNT = 0.9

# 内部表現(tuple)のフィールド位置。dataclass より軽く、そのままメモ化キーに使える。
_K, _HP, _PZ, _ACT, _PROT, _IMM, _THR = range(7)


# ---------------------------------------------------------------------------
# 内部: 探索の返り値
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Line:
    """ある局面から先の1本の読み筋。

    Attributes:
        solved: horizon 内にサイドを取り切れたか。
        turns: 取り切るまでの攻撃ターン数(未解決なら「使ったターン数 + LB」の見積り)。
        prizes: ターンごとの取得サイド枚数(index 0 = 現在のターン)。
        overkill: 無駄になった打点(KOに要らなかった直接打点 + 置けなかったダメカン)。
        threat: 除去した脅威度の合計。
        leaf_lb: 未解決で終わった葉の LB(解決済みなら0)。
    """

    solved: bool
    turns: int
    prizes: tuple[int, ...]
    overkill: int
    threat: float
    leaf_lb: int


_WIN_LINE = _Line(solved=True, turns=0, prizes=(), overkill=0, threat=0.0, leaf_lb=0)


def _discounted_prizes(prizes: tuple[int, ...]) -> float:
    """早く取ったサイドほど高く評価する割引和。"""
    total = 0.0
    factor = 1.0
    for p in prizes:
        total += p * factor
        factor *= _PRIZE_DISCOUNT
    return total


def _score(line: _Line) -> tuple:
    """辞書順の評価値。大きいほど良い。

    (1)horizon内に勝てるか (2)最短ターン数 (3)LBの減少量 (4)サイドを早く取れるか
    (5)オーバーキルの小ささ (6)除去した脅威度 の順。解決済みの読み筋は ``leaf_lb == 0``、
    未解決の読み筋は全て ``turns`` が見積り値なので、(1) で層が分かれた後の比較は
    それぞれの層の中で意味を持つ。
    """
    return (
        1 if line.solved else 0,
        -line.turns,
        -line.leaf_lb,
        round(_discounted_prizes(line.prizes), 6),
        -line.overkill,
        round(line.threat, 6),
    )


class _Context:
    """1回の求解で共有する状態(メモ・状態数予算)。"""

    __slots__ = ("config", "memo", "lb_memo", "states", "truncated")

    def __init__(self, config: DamagePlanConfig) -> None:
        self.config = config
        self.memo: dict[tuple, tuple[_Line, tuple | None]] = {}
        self.lb_memo: dict[tuple, int] = {}
        self.states = 0
        self.truncated = False


# ---------------------------------------------------------------------------
# 内部: 盤面(tuple 表現)の操作
# ---------------------------------------------------------------------------


def _hp_cap(config: DamagePlanConfig) -> int:
    """HPクリップの上限。

    horizon 内に与えられる最大ダメージ(毎ターン 直接打点 + ダメカン全部を1体に集中)を
    超えるHPは、horizon 内では絶対にKOできない点で等価なので同一視してよい。クリップは
    HPを下げる方向にしか働かないため LB を過大評価せず、下界としての健全性も保たれる。
    """
    return max(1, config.horizon * (config.direct_damage10 + config.counters) + 1)


def _to_node(state: DamageState, config: DamagePlanConfig) -> tuple:
    """``DamageState`` を内部 tuple 表現へ。key 昇順に正規化する(決定性のため)。"""
    cap = _hp_cap(config)
    targets = tuple(
        sorted(
            (
                int(t.key),
                min(int(t.hp10), cap),
                max(0, int(t.prize)),
                1 if t.is_active else 0,
                1 if t.counter_protected else 0,
                1 if getattr(t, "direct_immune", False) else 0,
                float(t.threat),
            )
            for t in state.targets
            if int(t.hp10) > 0
        )
    )
    return (targets, int(state.remaining_prizes), max(0, int(state.boss_budget)))


def _damage(node: tuple, key: int, amount: int) -> tuple[tuple, int, float]:
    """``key`` の対象に ``amount``(10単位)を与える。戻り値は (新 node, 取得サイド, 脅威度)。"""
    targets, remaining, boss = node
    new: list[tuple] = []
    prizes = 0
    threat = 0.0
    for t in targets:
        if t[_K] != key:
            new.append(t)
            continue
        hp = t[_HP] - amount
        if hp <= 0:
            prizes += t[_PZ]
            threat += t[_THR]
            continue  # KO: 盤面から除去
        new.append((t[_K], hp, t[_PZ], t[_ACT], t[_PROT], t[_IMM], t[_THR]))
    return (tuple(new), remaining - prizes, boss), prizes, threat


def _set_active(node: tuple, key: int, spend_boss: bool) -> tuple:
    """``key`` の対象をアクティブにする(ボスの指令 / KO後の入れ替え)。"""
    targets, remaining, boss = node
    if not spend_boss:
        cur = [t for t in targets if t[_ACT]]
        if cur and cur[0][_K] == key:
            return node
    new = tuple(
        (t[_K], t[_HP], t[_PZ], 1 if t[_K] == key else 0, t[_PROT], t[_IMM], t[_THR])
        for t in targets
    )
    return (new, remaining, boss - 1 if spend_boss else boss)


def _direct_choices(node: tuple, config: DamagePlanConfig) -> list[tuple[int, bool, int, int]]:
    """直接攻撃の候補 ``(key, ボスを使うか, 与ダメージ, 対象の残HP)`` を key 昇順で返す。"""
    targets, _remaining, boss = node
    out: list[tuple[int, bool, int, int]] = []
    actives = [t for t in targets if t[_ACT]]
    if actives:
        a = actives[0]
        out.append((a[_K], False, 0 if a[_IMM] else config.direct_damage10, a[_HP]))
        if boss > 0:
            for t in targets:
                if t[_ACT]:
                    continue
                out.append((t[_K], True, 0 if t[_IMM] else config.direct_damage10, t[_HP]))
    else:
        # アクティブが不在(前のターンにKO済み)。入れ替え先は相手が選ぶが、ここでは
        # 楽観的に自分が選べるものとして緩和する。
        for t in targets:
            out.append((t[_K], False, 0 if t[_IMM] else config.direct_damage10, t[_HP]))
    return out


def _counter_candidates(
    node: tuple, config: DamagePlanConfig, counters_left: int, turns_left: int
) -> list[int]:
    """ダメカンを置ける対象の key を昇順で返す(枝刈り込み)。

    枝刈りは4種類。いずれも「候補が全部消える場合は緩める」ので、置ける対象があるのに
    空リストを返すことは無い。

    1. 同型の対象(HP/サイド/属性が同じ)は1体に畳む(対称性)。
    2. horizon 内に到達できないHPの対象を落とす(そのターン数内ではKOに繋がらない)。
    3. サイドも脅威度も0の対象を落とす(倒しても評価が動かない)。
    4. **最終読みターン**(``turns_left <= 1``)では、このターン中にKOし切れる対象
       (``hp10 <= counters_left``)だけに絞る。ダメカンは減る一方なので「今KOできる」の
       判定はこの不等式と同値であり、最終ターンに達成可能なKO集合は変わらない(=最短
       ターン数の厳密性を壊さない)。落ちるのは horizon の外でしか実らない削りだけで、
       ここが状態数の主因なので効果が大きい。
    """
    targets = node[0]
    placeable = [t for t in targets if not t[_ACT] and not t[_PROT]]
    if not placeable:
        return []

    folded: list[tuple] = []
    seen: set[tuple] = set()
    for t in placeable:
        sig = (t[_HP], t[_PZ], t[_IMM], t[_THR])
        if sig in seen:
            continue
        seen.add(sig)
        folded.append(t)

    reach = counters_left + max(0, turns_left - 1) * config.counters + config.direct_damage10
    pruned = [t for t in folded if t[_HP] <= reach and (t[_PZ] > 0 or t[_THR] > 0.0)]
    use = pruned if pruned else folded
    if turns_left <= 1:
        finishable = [t for t in use if t[_HP] <= counters_left]
        if finishable:
            use = finishable
        else:
            # 最終ターンにKOできる対象が無い = 置いても horizon 内では実らない。
            # 「次のKOに一番近い対象」だけを残して状態爆発を止める(LB の同点処理用)。
            use = sorted(use, key=lambda t: (t[_HP], -t[_PZ], t[_K]))[:2]
    return [t[_K] for t in use]


# ---------------------------------------------------------------------------
# 内部: 下界(LB)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=65536)
def _lb_subset(items: tuple[tuple[int, int], ...], direct: int, counters: int) -> int:
    """KO対象集合 K を倒し切るのに必要な攻撃ターン数の下界。

    各対象 i に直接攻撃 ``a_i`` 回・ダメカン ``c_i`` 個を割り当て
    ``direct*a_i + c_i >= hp10_i`` を満たすとき、ターン数 T は
    ``T >= sum(a_i)`` かつ ``T >= ceil(sum(c_i)/counters)`` を満たす必要がある。
    ``max(sum a, ceil(sum c / counters))`` を (a, c) 全体で最小化した値を返す。

    直接攻撃の総数 A を固定すると、必要なダメカン総数は
    ``sum(hp10) - (A回の直接攻撃で削れる最大量)``。1回あたりの削り量は対象ごとに
    非増加(``min(direct, 残り)``)なので、全対象の限界削り量を降順に A 個取る貪欲が最適。

    同じ (HP, immune) の組み合わせは盤面をまたいで大量に再出現するので lru_cache する
    (葉の評価がプランナ全体のホットスポット)。

    Args:
        items: ``(hp10, direct_immune)`` のタプル(hashable)。immune な対象には直接攻撃を
            割り当てない。
        direct: 直接攻撃の打点(10単位)。
        counters: 1ターンあたりのダメカン個数。
    """
    total_hp = sum(hp for hp, _imm in items)
    if total_hp <= 0:
        return 0
    gains: list[int] = []
    for hp, imm in items:
        if imm or direct <= 0:
            continue
        rest = hp
        while rest > 0:
            g = min(direct, rest)
            gains.append(g)
            rest -= g
    gains.sort(reverse=True)

    best = UNREACHABLE
    prefix = 0
    for a in range(len(gains) + 1):
        if a > 0:
            prefix += gains[a - 1]
        need = total_hp - prefix
        if need <= 0:
            turns_by_counter = 0
        elif counters <= 0:
            turns_by_counter = UNREACHABLE  # ダメカンが無い設定では直接攻撃だけが頼り
        else:
            turns_by_counter = -(-need // counters)
        t = max(a, turns_by_counter)
        if t < best:
            best = t
        # A をこれ以上増やしても T >= A+1 >= best となり改善しない。
        if a + 1 >= best:
            break
    return best


@lru_cache(maxsize=65536)
def _lb_core(killable: tuple[tuple[int, int, int], ...], remaining: int, direct: int, counters: int) -> int:
    """LB の本体。``killable`` は ``(hp10, prize, direct_immune)`` の正規化済みタプル。

    盤面(node)そのものではなく「倒せる対象の多重集合」でキャッシュするため、探索中に
    大量に現れる同型局面で使い回せる(葉の評価がホットスポットなので効く)。
    """
    n = len(killable)
    prizes = [t[1] for t in killable]
    # マスクごとのサイド合計を差分で作る(内側の再ループを避ける)。
    prize_sum = [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        prize_sum[mask] = prize_sum[mask ^ low] + prizes[low.bit_length() - 1]

    best = UNREACHABLE
    for mask in range(1 << n):
        total = prize_sum[mask]
        if total < remaining:
            continue
        # 極小な集合だけ見れば十分。対象を足すと必要資源は増えこそすれ減らないので
        # LB は集合について単調(LB(K∪{x}) >= LB(K))であり、真部分集合で足りるなら
        # そちらの方が必ず小さい。
        minimal = True
        for i in range(n):
            if mask >> i & 1 and total - prizes[i] >= remaining:
                minimal = False
                break
        if not minimal:
            continue
        items = tuple((killable[i][0], killable[i][2]) for i in range(n) if mask >> i & 1)
        t = _lb_subset(items, direct, counters)
        if t < best:
            best = t
    return best


def _lower_bound(ctx: _Context, node: tuple) -> int:
    """盤面の下界。配置可能性(counter_protected)とボス入手を緩和した楽観的な値。

    ダメカンをアクティブに置けない制約・ボスの枚数制限は無視する(緩和)。緩和は必要
    ターン数を過小評価する方向にしか働かないので、下界として安全。逆に「絶対に倒せない
    対象(ダメカン不可 かつ 直接攻撃無効)」は KO 対象集合から外して締める。
    """
    cached = ctx.lb_memo.get(node)
    if cached is not None:
        return cached

    targets, remaining, _boss = node
    if remaining <= 0:
        ctx.lb_memo[node] = 0
        return 0

    killable = [t for t in targets if t[_PZ] > 0 and not (t[_PROT] and t[_IMM])]
    # 対象が多いときはサイド効率の良い順に絞る(2^n の爆発を防ぐ。実盤面は最大6体)。
    if len(killable) > 8:
        killable = sorted(killable, key=lambda t: (-t[_PZ], t[_HP], t[_K]))[:8]

    config = ctx.config
    # 順序は結果に影響しないので正規化してキャッシュ命中率を上げる。
    key = tuple(sorted((t[_HP], t[_PZ], t[_IMM]) for t in killable))
    best = _lb_core(key, remaining, config.direct_damage10, config.counters)
    ctx.lb_memo[node] = best
    return best


def _leaf_line(ctx: _Context, node: tuple) -> _Line:
    """horizon 端(または打ち切り)での評価。LB を残りターン数の見積りとして使う。"""
    lb = _lower_bound(ctx, node)
    return _Line(solved=False, turns=max(1, lb), prizes=(0,), overkill=0, threat=0.0, leaf_lb=lb)


# ---------------------------------------------------------------------------
# 内部: 探索
# ---------------------------------------------------------------------------


def _prepend(sub: _Line, prizes: int, overkill: int, threat: float) -> _Line:
    """同一ターン内の1手を読み筋の先頭に足す(ターン数は増えない)。"""
    if sub.turns == 0:
        # sub は「もう取り切っている」= このターンで勝ち切った。
        return _Line(True, 1, (prizes,), overkill, threat, 0)
    if not sub.prizes:
        new_prizes: tuple[int, ...] = (prizes,)
    else:
        new_prizes = (sub.prizes[0] + prizes,) + sub.prizes[1:]
    return _Line(
        sub.solved,
        sub.turns,
        new_prizes,
        sub.overkill + overkill,
        sub.threat + threat,
        sub.leaf_lb,
    )


def _advance_turn(sub: _Line) -> _Line:
    """ターン境界。現在のターンを1消費して次ターンの読み筋を接ぐ。"""
    return _Line(
        sub.solved,
        1 + sub.turns,
        (0,) + sub.prizes,
        sub.overkill,
        sub.threat,
        sub.leaf_lb,
    )


def _search(
    ctx: _Context, node: tuple, counters_left: int, direct_pending: bool, turns_left: int
) -> _Line:
    """局面 ``node`` からの最良の読み筋を返す。

    Args:
        node: 内部 tuple 表現の盤面。
        counters_left: このターンに残っているダメカン個数。
        direct_pending: このターンの直接攻撃がまだ済んでいないなら True。
        turns_left: 現在のターンを含めて、あと何ターン読むか。
    """
    remaining = node[1]
    if remaining <= 0:
        return _WIN_LINE
    if not node[0] or turns_left <= 0:
        return _leaf_line(ctx, node)

    memo_key = (node, counters_left, direct_pending, turns_left)
    hit = ctx.memo.get(memo_key)
    if hit is not None:
        return hit[0]
    if ctx.states >= ctx.config.max_states:
        # 予算切れ。例外は投げず LB ベースの評価に落とす(打ち切り後もメモ済みの値は
        # そのまま使うので、同じ局面が場所によって違う値になることはない)。
        ctx.truncated = True
        return _leaf_line(ctx, node)
    ctx.states += 1

    config = ctx.config
    best_line: _Line | None = None
    best_score: tuple | None = None
    best_action: tuple | None = None

    if direct_pending:
        for key, spend_boss, dmg, hp10 in _direct_choices(node, config):
            nd = _set_active(node, key, spend_boss)
            if dmg > 0:
                nd, prizes, threat = _damage(nd, key, dmg)
                overkill = max(0, dmg - hp10) if dmg >= hp10 else 0
            else:
                prizes, threat, overkill = 0, 0.0, 0
            sub = _search(ctx, nd, config.counters, False, turns_left)
            line = _prepend(sub, prizes, overkill, threat)
            score = _score(line)
            if best_score is None or score > best_score:
                best_score = score
                best_line = line
                best_action = ("direct", key, spend_boss)
    elif counters_left > 0:
        candidates = _counter_candidates(node, config, counters_left, turns_left)
        if candidates:
            for key in candidates:
                nd, prizes, threat = _damage(node, key, 1)
                sub = _search(ctx, nd, counters_left - 1, False, turns_left)
                line = _prepend(sub, prizes, 0, threat)
                score = _score(line)
                if best_score is None or score > best_score:
                    best_score = score
                    best_line = line
                    best_action = ("counter", key)
        else:
            # 置ける対象が無い = 残りのダメカンは無駄になる。
            sub = _search(ctx, node, 0, False, turns_left)
            best_line = _Line(
                sub.solved,
                sub.turns,
                sub.prizes,
                sub.overkill + counters_left,
                sub.threat,
                sub.leaf_lb,
            )
            best_action = ("waste", counters_left)
    else:
        sub = _search(ctx, node, config.counters, True, turns_left - 1)
        best_line = _advance_turn(sub)
        best_action = ("turn",)

    if best_line is None:  # 実行されない想定(候補ゼロは各分岐で吸収済み)
        best_line = _leaf_line(ctx, node)
        best_action = None
    ctx.memo[memo_key] = (best_line, best_action)
    return best_line


# ---------------------------------------------------------------------------
# 公開関数
# ---------------------------------------------------------------------------


def solve_damage_plan(
    state: DamageState, *, config: DamagePlanConfig = DEFAULT_CONFIG
) -> DamagePlan | None:
    """1ターン目の直接攻撃対象とダメカン配分、および先の見通しを求める。

    Args:
        state: 解く盤面。
        config: 探索パラメータ。

    Returns:
        DamagePlan: 対象が1体でもあれば必ず返る。対象が無いときのみ None。
    """
    node = _to_node(state, config)
    if not node[0]:
        return None

    ctx = _Context(config)
    lb = _lower_bound(ctx, node)
    line = _search(ctx, node, config.counters, True, max(1, config.horizon))

    # 1ターン目の行動をメモから復元する(探索と同じ選択がメモに入っている)。
    alloc: dict[int, int] = {}
    direct_target: int | None = None
    cur = node
    counters_left = config.counters
    direct_pending = True
    turns_left = max(1, config.horizon)
    while True:
        entry = ctx.memo.get((cur, counters_left, direct_pending, turns_left))
        if entry is None or entry[1] is None:
            break
        action = entry[1]
        kind = action[0]
        if kind == "direct":
            _, key, spend_boss = action
            direct_target = key
            cur = _set_active(cur, key, spend_boss)
            immune = any(t[_K] == key and t[_IMM] for t in cur[0])
            if not immune:
                cur, _p, _th = _damage(cur, key, config.direct_damage10)
            counters_left = config.counters
            direct_pending = False
        elif kind == "counter":
            _, key = action
            alloc[key] = alloc.get(key, 0) + 1
            cur, _p, _th = _damage(cur, key, 1)
            counters_left -= 1
        else:
            # "waste"(置き場が無い) / "turn"(ターン終了) = 1ターン目の行動は確定。
            break
        if cur[1] <= 0:
            break

    return DamagePlan(
        turns_to_win=line.turns if line.solved else None,
        lower_bound=lb,
        counter_alloc=tuple(sorted(alloc.items())),
        direct_target=direct_target,
        projected_prizes=tuple(line.prizes),
    )


def choose_next_counter(
    targets: Sequence[TargetState],
    remaining_counters: int,
    remaining_prizes: int,
    *,
    config: DamagePlanConfig = DEFAULT_CONFIG,
    boss_budget: int = 0,
) -> int | None:
    """次の1個を置くべき target key。毎回再計画する(計画キャッシュに依存しない)。

    このターンの直接攻撃はすでに済んでいる前提(ダメカン配置はファントムダイブの効果解決中に
    降ってくる sub-select なので、200ダメージはもう入っている)。したがって現在のターンは
    「残り ``remaining_counters`` 個を置くだけのターン」として扱い、次のターン以降は
    直接攻撃 + ダメカン full の通常ターンとして読む。

    Args:
        targets: 相手の場のポケモン(アクティブ含む)。
        remaining_counters: まだ置けるダメカンの個数(``select.remainDamageCounter``)。
        remaining_prizes: 自分が勝つのに必要な残りサイド枚数。
        config: 探索パラメータ。
        boss_budget: 次ターン以降に使えるボスの指令の枚数(既定0)。

    Returns:
        置くべき対象の key。解けない/対象なしは None(呼び出し側は従来ロジックへ)。
    """
    if remaining_counters < 1 or not targets:
        return None
    if remaining_prizes <= 0:
        # すでに取り切っている(このプランナが答えるべき状況ではない)。
        return None

    state = DamageState(
        targets=tuple(targets), remaining_prizes=remaining_prizes, boss_budget=boss_budget
    )
    node = _to_node(state, config)
    if not node[0]:
        return None

    candidates = [t[_K] for t in node[0] if not t[_ACT] and not t[_PROT]]
    if not candidates:
        return None

    ctx = _Context(config)
    turns_left = max(1, config.horizon)
    best_key: int | None = None
    best_score: tuple | None = None
    for key in candidates:  # node[0] は key 昇順 = 決定的な評価順
        nd, prizes, threat = _damage(node, key, 1)
        sub = _search(ctx, nd, remaining_counters - 1, False, turns_left)
        line = _prepend(sub, prizes, 0, threat)
        score = _score(line)
        if best_score is None or score > best_score:
            best_score = score
            best_key = key
    return best_key
