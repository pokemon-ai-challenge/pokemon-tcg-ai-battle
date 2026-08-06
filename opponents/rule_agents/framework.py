"""手書きルールベースエージェント3種（オーロンゲ／メガルカリオ／ジュラルドン）の共通土台。

## なぜ土台を分けるのか

`cg` エンジンは「今どの選択をさせているか」を `SelectContext` で細かく分けて聞いてくる
（48種類）。どのデッキを使っても答え方が変わらない機械的な部分（選択個数の下限・上限を
守る、コイントスに答える、ダメカンの置き先を選ぶ…）と、デッキごとに考えて決めるべき部分
（何を優先して進化させるか、どの技を撃つか、いつサポートを切るか）が混ざっている。

このモジュールは前者だけを引き受ける。デッキ固有の判断は `Strategy` を継承した
`grimmsnarl.py` / `lucario.py` / `archaludon.py` に手書きで置く。

## 判断の形

全部の選択肢に点数をつけ、点数が高い順に `minCount`〜`maxCount` 個返す、という形に統一した。
どの `SelectContext` でも「選択肢の配列から何個か選ぶ」という形は同じなので、この形なら
ルール違反（個数超過・重複・範囲外）を構造的に起こさない。

点数は下の `S_*` 定数を段階として使う。同じターン内で「進化 → エネルギー → 手札入れ替え →
攻撃」の順に処理させたいので、やりたい順に点数の帯を割り当ててある（1手選ぶたびに
エンジンが次の MAIN 選択を投げ直してくるため、点数の大小がそのまま行動の順番になる）。

## 用語

- **ワザのロック**: 「次の自分の番、このポケモンは○○を使えない」という制限。エンジン側が
  そもそも選択肢に出さないので、こちらで覚えておく必要はない。
- **サイド (prize)**: 相手を倒すと取れるカード。`len(player.prize)` は「あと何枚取れば
  勝ちか」を表す（0 になった側が勝ち）。
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# cg.api から取り込む名前は、実際に使うものだけに絞る。ここに書いた名前が
# 実行環境の cg.api に1つでも欠けていると、このモジュール全体の import が失敗し、
# 提出物は保険の行動しか返せなくなる（= 判断が丸ごと無効になる）。
from cg.api import (
    AreaType,
    Card,
    EnergyType,
    Observation,
    Option,
    OptionType,
    PlayerState,
    Pokemon,
    SelectContext,
    SelectData,
    State,
    all_attack,
    all_card_data,
    to_observation_class,
)

# --- 点数の帯 -------------------------------------------------------------------
# 大きいほど「先にやる」。1アクションごとにエンジンが MAIN を投げ直すので、
# ここの大小関係がそのまま1ターン内の行動順になる。
S_MUST = 90000      # これを逃すと負ける／勝ちを逃す手
S_KEY = 70000       # デッキの核。主軸への進化、ふしぎなアメ
S_TOOL = 60000      # 道具付け
S_DEVELOP = 50000   # たねポケモンをベンチに出す
S_SEARCH = 40000    # サーチ・ドローのアイテム
S_ENERGY = 30000    # エネルギー付け
S_RETREAT = 20000   # にげる／入れ替え
S_REFRESH = 15000   # 手札を丸ごと引き直すサポート（やることを済ませてから）
S_ATTACK = 10000    # 攻撃（＝ターン終了）
S_END = 100         # 何もせずターン終了
SKIP = -1.0         # 選ばない（minCount に足りないときだけ渋々選ばれる）

# ワザ ID のうち、弱点・抵抗力を無視するもの。
_IGNORE_WEAKNESS_ATTACKS = frozenset({
    980,  # Cosmic Beam（ソルロック）
    148,  # Demolish（イワオオギ／カウンターキャッチャー系ではなく "Demolish"）
})

_STADIUM_FULL_METAL_LAB = 1244
_STADIUM_GRAVITY_MOUNTAIN = 1252
_STADIUM_WATCHTOWER = 1256

class _LazyTable:
    """カード表・ワザ表を「最初に引かれたとき」に作る入れ物。

    中身は `all_card_data()` / `all_attack()` というネイティブライブラリ呼び出しで、
    これを import 時に実行すると、呼び出しが失敗した環境では
    **モジュールの import ごと落ちる**。提出物側は import に失敗しても例外を
    握りつぶして保険の行動に切り替える作りなので、そうなると「エラーは出ないのに
    判断が一切効いていない」状態のまま試合が進んでしまう(実際に Kaggle 上で
    これを踏み、118手すべてが保険の行動になっていた)。

    遅延化しておけば、少なくとも import は必ず通る。
    """

    def __init__(self, loader, key):
        self._loader = loader
        self._key = key
        self._data: dict | None = None

    def _ensure(self) -> dict:
        if self._data is None:
            self._data = {self._key(v): v for v in self._loader()}
        return self._data

    def get(self, k, default=None):
        return self._ensure().get(k, default)

    def __getitem__(self, k):
        return self._ensure()[k]

    def __contains__(self, k) -> bool:
        return k in self._ensure()

    def __len__(self) -> int:
        return len(self._ensure())

    def __iter__(self):
        return iter(self._ensure())

    def values(self):
        return self._ensure().values()

    def items(self):
        return self._ensure().items()


CARDS: _LazyTable = _LazyTable(all_card_data, lambda c: c.cardId)
ATTACKS: _LazyTable = _LazyTable(all_attack, lambda a: a.attackId)

_DEBUG = bool(os.environ.get("RULE_AGENT_DEBUG"))


def read_deck(path: str | Path) -> list[int]:
    """デッキ CSV（1行1カードID、カンマ区切りも可）を 60 枚のリストとして読む。"""
    text = Path(path).read_text(encoding="utf-8")
    deck = [int(v) for v in text.replace(",", "\n").split() if v.strip() and not v.startswith("#")]
    if len(deck) != 60:
        raise ValueError(f"デッキは60枚でなければならない: {len(deck)}枚 ({path})")
    return deck


# --- 盤面の読み取り -------------------------------------------------------------


@dataclass
class Ctx:
    """1回の選択のあいだ使い回す盤面の要約。

    `Observation` を毎回たどり直すと読みづらいので、よく使う値を先に取り出しておく。
    """

    obs: Observation
    state: State
    select: SelectData
    context: SelectContext
    me: int
    opp: int
    my: PlayerState
    op: PlayerState
    my_active: Pokemon | None
    op_active: Pokemon | None
    my_bench: list[Pokemon]
    op_bench: list[Pokemon]
    my_field: list[Pokemon]
    op_field: list[Pokemon]
    hand: list[Card]
    hand_counts: dict[int, int]
    field_counts: dict[int, int]
    discard_counts: dict[int, int]
    deck_counts: dict[int, int]
    stadium_id: int
    my_prize_left: int
    op_prize_left: int
    turn: int
    going_first: bool
    # 盤面の厚み。「バトル場のポケモンが倒されたとき、後がいない」＝即負けなので、
    # どのデッキでもここが薄いときはベンチ展開を最優先する必要がある。
    pokemon_in_play: int = 0
    bench_room: int = 0
    attack_options: list[Option] = field(default_factory=list)
    can_retreat: bool = False
    # 「このターンに使うと決めたサポート」。MAIN で1枚に絞ってから PLAY を評価する
    # （サポートは1ターン1枚なので、点数が近い2枚が交互に選ばれるのを避ける）。
    chosen_supporter: int = 0
    # デッキ固有の一時メモ（Strategy が自由に使う）。
    memo: dict = field(default_factory=dict)

    def card_at(self, area: AreaType | None, index: int | None, player_index: int | None) -> Card | Pokemon | None:
        """選択肢が指している場所から実体（Card / Pokemon）を取り出す。"""
        if area is None or index is None:
            return None
        pi = self.me if player_index is None else player_index
        ps = self.state.players[pi]
        try:
            if area == AreaType.DECK:
                return None if self.select.deck is None else self.select.deck[index]
            if area == AreaType.HAND:
                return None if ps.hand is None else ps.hand[index]
            if area == AreaType.DISCARD:
                return ps.discard[index]
            if area == AreaType.ACTIVE:
                return ps.active[index]
            if area == AreaType.BENCH:
                return ps.bench[index]
            if area == AreaType.PRIZE:
                return ps.prize[index]
            if area == AreaType.STADIUM:
                return self.state.stadium[index]
            if area == AreaType.LOOKING:
                return None if self.state.looking is None else self.state.looking[index]
        except (IndexError, TypeError):
            return None
        return None

    def card_id_of(self, o: Option) -> int:
        """選択肢が指しているカードの ID。分からなければ 0。"""
        if o.cardId:
            return o.cardId
        obj = self.card_at(o.area, o.index, o.playerIndex)
        return obj.id if obj is not None else 0


def _tally_known_cards(state: State, me: int, select: SelectData) -> dict[int, int]:
    """自分の 60 枚のうち「デッキ以外の場所で見えている」枚数を数える。"""
    seen: dict[int, int] = defaultdict(int)
    serials: set[int] = set()

    def note(card: Card | Pokemon | None) -> None:
        if card is None:
            return
        if getattr(card, "playerIndex", me) != me:
            return
        if card.serial not in serials:
            serials.add(card.serial)
            seen[card.id] += 1
        if isinstance(card, Pokemon):
            for c in card.energyCards:
                note(c)
            for c in card.tools:
                note(c)
            for c in card.preEvolution:
                note(c)

    ps = state.players[me]
    for c in ps.hand or []:
        note(c)
    for c in ps.discard:
        note(c)
    for p in ps.bench:
        note(p)
    for p in ps.active:
        note(p)
    for c in state.stadium:
        note(c)
    for c in state.looking or []:
        note(c)
    note(select.effect)
    return seen


def build_ctx(obs: Observation, deck: list[int]) -> Ctx:
    state = obs.current
    select = obs.select
    me = state.yourIndex
    opp = 1 - me
    my = state.players[me]
    op = state.players[opp]

    my_active = my.active[0] if my.active and my.active[0] is not None else None
    op_active = op.active[0] if op.active and op.active[0] is not None else None
    my_bench = list(my.bench)
    op_bench = list(op.bench)
    my_field = ([my_active] if my_active else []) + my_bench
    op_field = ([op_active] if op_active else []) + op_bench

    hand = list(my.hand or [])
    hand_counts: dict[int, int] = defaultdict(int)
    for c in hand:
        hand_counts[c.id] += 1

    field_counts: dict[int, int] = defaultdict(int)
    for p in my_field:
        field_counts[p.id] += 1
        for pre in p.preEvolution:
            field_counts[pre.id] += 1

    discard_counts: dict[int, int] = defaultdict(int)
    for c in my.discard:
        discard_counts[c.id] += 1

    # デッキ内訳の推定。サイド6枚は見えないのでデッキ側に数え続ける（多めに見積もる）。
    # サーチでデッキが開示されているとき（select.deck）はそちらが正確なのでそれを使う。
    deck_counts: dict[int, int] = defaultdict(int)
    if select.deck is not None:
        for c in select.deck:
            deck_counts[c.id] += 1
    else:
        seen = _tally_known_cards(state, me, select)
        for cid in deck:
            deck_counts[cid] += 1
        for cid, n in seen.items():
            deck_counts[cid] -= n

    stadium_id = state.stadium[0].id if state.stadium else 0

    ctx = Ctx(
        obs=obs,
        state=state,
        select=select,
        context=select.context,
        me=me,
        opp=opp,
        my=my,
        op=op,
        my_active=my_active,
        op_active=op_active,
        my_bench=my_bench,
        op_bench=op_bench,
        my_field=my_field,
        op_field=op_field,
        hand=hand,
        hand_counts=hand_counts,
        field_counts=field_counts,
        discard_counts=discard_counts,
        deck_counts=deck_counts,
        stadium_id=stadium_id,
        my_prize_left=len(my.prize),
        op_prize_left=len(op.prize),
        turn=state.turn,
        going_first=(state.firstPlayer == me),
        pokemon_in_play=len(my_field),
        bench_room=max(0, my.benchMax - len(my_bench)),
    )
    for o in select.option:
        if o.type == OptionType.ATTACK:
            ctx.attack_options.append(o)
        elif o.type == OptionType.RETREAT:
            ctx.can_retreat = True
    return ctx


# --- ダメージ計算 ---------------------------------------------------------------


def ko_prize(pokemon: Pokemon) -> int:
    """このポケモンを倒したとき、倒した側が取れるサイドの枚数。"""
    data = CARDS.get(pokemon.id)
    if data is None:
        return 1
    if data.megaEx:
        return 3
    if data.ex:
        return 2
    return 1


def has_ability(pokemon: Pokemon) -> bool:
    data = CARDS.get(pokemon.id)
    return bool(data and data.skills)


def damage_reduction(ctx: Ctx, defender: Pokemon) -> int:
    """スタジアムなど「受けるダメージを減らす」効果の合計。"""
    reduction = 0
    if ctx.stadium_id == _STADIUM_FULL_METAL_LAB:
        data = CARDS.get(defender.id)
        if data and data.energyType == EnergyType.METAL:
            reduction += 30
    return reduction


def is_pokemon_ex(pokemon: Pokemon | None) -> bool:
    """カードの `ex` フラグと `megaEx` フラグは排他だが、ルール文の"Pokémon ex"は
    両方を指す（メガ進化ポケモンexも Pokémon ex の一種）。判定はここに集約する。
    """
    if pokemon is None:
        return False
    data = CARDS.get(pokemon.id)
    return data is not None and (data.ex or data.megaEx)


# 「相手が Pokémon ex ならワザのダメージを完全に防ぐ」特性・効果を持つカード。
# クラスタゲが実測でこの壁を持ち、こちらのオーロンゲex/ブリジュラスex/メガルカリオex
# （全員 Pokémon ex）のワザを毎回0ダメージにしていた（実戦で見つけた）。
# キーが対象を限定しない（誰の攻撃も防ぐ）、値が True なら「たねポケモンの攻撃だけ」に限る
# （カビゴンフラダリ／ジラーチexの「よろいのおびれ」は "Basic Pokémon ex" 限定）。
_EX_WALL: dict[int, bool] = {
    345: False,  # Crustle「ふしぎな岩宿」
    330: False,  # Sylveon「まもるバリア」
    83: True,    # Farigiraf ex「アーマーテイル」(たね Pokémon ex 限定)
}

# スタジアム版の壁。ニュートラルゾーンは「ルールを持たないポケモンは、相手の
# Pokémon ex/V からのワザのダメージを受けない」を場に出ている間ずっと適用する
# （出した側だけでなく双方に効く。実戦(ep90534503)でこれを見落とし、9回連続で
# シャドーバレットを撃ってダメージ0が続いたまま倒された）。
_STADIUM_EX_WALL = 1247  # Neutralization Zone


def is_damage_immune(
    attacker: Pokemon | None, defender: Pokemon, on_bench: bool, stadium_id: int = 0,
) -> bool:
    """ワザのダメージが丸ごと通らない相手かどうか。

    - テラスポケモンはベンチにいる間ワザのダメージを受けない。
    - イワオオギは「特性を持つポケモンからのワザのダメージ」を全て防ぐ。
    - クラスタゲ／ニンフィア／キリンリキexは「Pokémon ex からのワザのダメージ」を
      全て防ぐ。こちらの主軸(オーロンゲex/ブリジュラスex/メガルカリオex)は全員
      Pokémon ex なので、これらが相手の場にいる間は攻撃しても何も減らせない。
    - ニュートラルゾーンが場にあると、ルールを持たない(exでもmegaExでもない)
      ポケモンは、こちらの Pokémon ex からのワザを一切受けない。
    """
    data = CARDS.get(defender.id)
    if data is None:
        return False
    if on_bench and data.tera:
        return True
    if defender.id == 117 and attacker is not None and has_ability(attacker):
        return True
    basic_only = _EX_WALL.get(defender.id)
    if basic_only is not None and is_pokemon_ex(attacker):
        attacker_data = CARDS.get(attacker.id) if attacker is not None else None
        if not basic_only or (attacker_data is not None and attacker_data.basic):
            return True
    if stadium_id == _STADIUM_EX_WALL and is_pokemon_ex(attacker) and not (data.ex or data.megaEx):
        return True
    return False


def estimate_damage(
    ctx: Ctx,
    attacker: Pokemon | None,
    attack_id: int,
    defender: Pokemon,
    *,
    bonus: int = 0,
    on_bench: bool = False,
) -> int:
    """このワザで相手に何ダメージ入るかの見積り。

    厳密な再現ではなく判断に足りる精度を狙っている。効果テキストで増えるダメージ
    （ジュラルドンの「あばれ叩き」など）は呼び出し側が `bonus` で足す。
    """
    atk = ATTACKS.get(attack_id)
    if atk is None:
        return 0
    base = atk.damage
    if base <= 0 and bonus <= 0:
        return 0
    if is_damage_immune(attacker, defender, on_bench, ctx.stadium_id):
        return 0

    dmg = base + bonus
    if not on_bench and attack_id not in _IGNORE_WEAKNESS_ATTACKS and attacker is not None:
        atk_data = CARDS.get(attacker.id)
        def_data = CARDS.get(defender.id)
        if atk_data is not None and def_data is not None:
            if def_data.weakness is not None and def_data.weakness == atk_data.energyType:
                dmg *= 2
            elif def_data.resistance is not None and def_data.resistance == atk_data.energyType:
                dmg -= 30
    dmg -= damage_reduction(ctx, defender)
    return max(0, dmg)


def retreat_blocked(ctx: Ctx) -> bool:
    """バトル場のポケモンが、エネルギー不足で逃げられない状態かどうか。

    にげるにはにげるエネルギーの数だけエネルギーをトラッシュする必要がある。
    ワザも撃てず、にげることもできないポケモンが前に居座ると、そのまま何ターンも
    棒に振る（実測で、マンキーが前に出た試合はこれで完全に止まっていた）。
    そういうときは「逃げるためだけに」エネルギーを1枚貼る価値がある。
    """
    active = ctx.my_active
    if active is None or ctx.state.retreated:
        return False
    data = CARDS.get(active.id)
    if data is None:
        return False
    return len(active.energies) < data.retreatCost


def can_attack_now(ctx: Ctx, pokemon: Pokemon) -> bool:
    """今ついているエネルギーで撃てるワザが1つでもあるか。"""
    data = CARDS.get(pokemon.id)
    if data is None:
        return False
    energy = len(pokemon.energies)
    return any(
        ATTACKS.get(aid) is not None and len(ATTACKS[aid].energies) <= energy
        for aid in data.attacks
    )


def active_is_stalled(ctx: Ctx) -> bool:
    """バトル場のポケモンが、このターン何もできない状態か。

    ワザも撃てず、控えと交代することもできない（にげるエネルギーが足りない、
    または控えがいない）。この状態のターンは丸ごと無駄になる。
    """
    active = ctx.my_active
    if active is None:
        return False
    if can_attack_now(ctx, active):
        return False
    return retreat_blocked(ctx) or not ctx.my_bench


def bench_can_attack(ctx: Ctx) -> bool:
    """控えに「今すぐワザを撃てる」ポケモンがいるか。"""
    return any(can_attack_now(ctx, p) for p in ctx.my_bench)


def incoming_damage(ctx: Ctx) -> int:
    """相手のバトルポケモンが次のターンに自分のバトルポケモンへ出せる最大ダメージの見積り。

    相手の手札は見えないので「今ついているエネルギーで撃てるワザ」だけを見る。
    自分のポケモンを逃がすか・回復するかの判断に使う。

    **表記ダメージが0のワザに注意。** 「〜1枚につき100ダメージ」のように、
    ダメージが状況で決まるワザは `Attack.damage` が 0 になっている。額面どおり
    0 と見なすと、こちらは「この相手は無害だ」と判断してしまう。実戦でメガユキノオーex
    の「ハンマーランチ」(山札の上6枚を落とし、その中の基本水エネルギー1枚につき100)
    がこれに当たり、水エネルギーを35枚積んだ相手の実質350ダメージを 0 と読んでいた。

    そこで、表記0でも効果文に「damage」を含むワザは、**必要エネルギー1個あたり90**
    という控えめな見積りに置き換える(ワザの打点はおおむねコストに比例するため)。
    守りの判断にしか使わない値なので、多めに見積もる側に倒しておく方が安全。
    """
    if ctx.op_active is None or ctx.my_active is None:
        return 0
    data = CARDS.get(ctx.op_active.id)
    if data is None:
        return 0
    energy = len(ctx.op_active.energies)
    best = 0
    for aid in data.attacks:
        atk = ATTACKS.get(aid)
        if atk is None or len(atk.energies) > energy:
            continue
        dmg = estimate_damage(ctx, ctx.op_active, aid, ctx.my_active)
        if dmg == 0 and atk.damage == 0 and "damage" in (atk.text or "").lower():
            dmg = 90 * max(1, len(atk.energies))
        best = max(best, dmg)
    return best


@dataclass
class AttackEval:
    option_index: int
    attack_id: int
    damage: int
    ko: bool
    prize: int


def evaluate_attacks(ctx: Ctx, *, bonus: int = 0) -> list[AttackEval]:
    """今の MAIN で撃てるワザそれぞれについて、相手バトルへの通り具合を出す。"""
    out: list[AttackEval] = []
    if ctx.op_active is None:
        return out
    for i, o in enumerate(ctx.select.option):
        if o.type != OptionType.ATTACK or o.attackId is None:
            continue
        dmg = estimate_damage(ctx, ctx.my_active, o.attackId, ctx.op_active, bonus=bonus)
        ko = dmg >= ctx.op_active.hp
        out.append(
            AttackEval(
                option_index=i,
                attack_id=o.attackId,
                damage=dmg,
                ko=ko,
                prize=ko_prize(ctx.op_active) if ko else 0,
            )
        )
    return out


def best_gust_target(ctx: Ctx, *, bonus: int = 0) -> tuple[int, float]:
    """ボスの指令などで相手のベンチから引きずり出す相手を選ぶ。

    Returns:
        (相手ベンチのインデックス, 評価値)。引きずり出す価値が無ければ (-1, 0)。
    """
    if not ctx.op_bench or ctx.my_active is None:
        return -1, 0.0
    data = CARDS.get(ctx.my_active.id)
    if data is None:
        return -1, 0.0
    energy = len(ctx.my_active.energies)
    usable = [
        aid for aid in data.attacks
        if (ATTACKS.get(aid) is not None and len(ATTACKS[aid].energies) <= energy)
    ]

    best_index, best_score = -1, 0.0
    for i, p in enumerate(ctx.op_bench):
        dmg = max((estimate_damage(ctx, ctx.my_active, aid, p, bonus=bonus) for aid in usable), default=0)
        score = 0.0
        if dmg >= p.hp:
            # 倒せるなら、取れるサイドの枚数がそのまま価値。
            score = 1000.0 * ko_prize(p)
            # 残りサイドがちょうど足りるなら最優先。
            if ko_prize(p) >= ctx.my_prize_left:
                score += 100000.0
        else:
            # 倒せなくても「育っていない／エネルギーが乗っていない」相手を前に出す価値はある。
            score = 120.0 - len(p.energies) * 40.0
            score += (1 - p.hp / max(1, p.maxHp)) * 100.0
        if score > best_score:
            best_index, best_score = i, score
    return best_index, best_score


# --- Strategy ------------------------------------------------------------------


class Strategy:
    """デッキ固有の判断を書く場所。必要なフックだけ上書きする。

    どのフックも「点数」を返す。負の点数は「選ばなくて済むなら選ばない」を意味する。
    """

    name: str = "unnamed"
    deck_path: str = ""

    # ---- ターンの方針 ----
    def prepare(self, ctx: Ctx) -> None:
        """点数付けを始める前に1回だけ呼ばれる。ターン全体の方針を決める場所。"""
        return None

    def attack_bonus(self, ctx: Ctx) -> int:
        """このターン既に確定しているダメージの上乗せ（プレミアムパワープロ等）。"""
        return 0

    # ---- MAIN（自分の番の行動選択） ----
    def play_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        """手札からカードを出す価値。"""
        return SKIP

    def attach_score(self, ctx: Ctx, card_id: int, target: Pokemon, is_active: bool) -> float:
        """エネルギー／道具を、この場のポケモンに付ける価値。"""
        return SKIP

    def evolve_score(self, ctx: Ctx, card_id: int, target: Pokemon, is_active: bool) -> float:
        """この場のポケモンを進化させる価値。"""
        return SKIP

    def ability_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        """特性を使う価値。"""
        return S_SEARCH

    def retreat_score(self, ctx: Ctx) -> float:
        """にげる価値。"""
        return SKIP

    def attack_score(self, ctx: Ctx, ev: AttackEval) -> float:
        """この技を撃つ価値。既定は「倒せるなら高く、それ以外はダメージ順」。"""
        if ev.ko:
            return S_ATTACK + 5000 + ev.prize * 500
        return S_ATTACK + min(ev.damage, 400)

    # ---- 場に出す／前に出す ----
    def active_pref(self, ctx: Ctx, card_id: int) -> float:
        """「この種類のポケモンをバトル場に置きたい度合い」。

        初期配置（手札から伏せる）のように、まだ場のポケモンとして存在しない段階でも
        判断できるよう、カードIDだけで答える。
        """
        return 0.0

    def active_score(self, ctx: Ctx, pokemon: Pokemon, o: Option) -> float:
        """既に場にいるポケモンをバトル場へ出す価値（入れ替え・戦闘不能後の選択）。"""
        return self.active_pref(ctx, pokemon.id) + len(pokemon.energies) * 100 + pokemon.hp

    def bench_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        """自分のポケモンをベンチに置く価値。"""
        return 100.0

    # ---- カードの欲しさ ----
    def want_in_hand(self, ctx: Ctx, card_id: int) -> float:
        """このカードを手札に加えたい度合い（サーチ・ドローの選択に使う）。

        捨てる選択（DISCARD）ではこの値の符号を反転して使う。
        """
        return 0.0

    # ---- その他の文脈 ----
    def yes_no_score(self, ctx: Ctx, is_yes: bool) -> float | None:
        """はい／いいえ。None を返すと共通の既定に任せる。"""
        return None

    def damage_counter_score(self, ctx: Ctx, pokemon: Pokemon, o: Option) -> float | None:
        """ダメカンの置き先／移動先。None で既定に任せる。"""
        return None

    def special_score(self, ctx: Ctx, o: Option) -> float | None:
        """上のどれにも当てはまらない文脈のための逃げ道。None で既定に任せる。"""
        return None


# --- 汎用のスコアリング ----------------------------------------------------------


def _generic_yes_no(ctx: Ctx, is_yes: bool) -> float:
    if ctx.context == SelectContext.IS_FIRST:
        # 先攻／後攻はデッキごとに考えが違うので Strategy 側で必ず上書きする。
        # ここに来た場合は「後攻（サポートが使えて、1ターン目から殴れる）」を既定にする。
        return 1.0 if not is_yes else 0.0
    if ctx.context == SelectContext.MULLIGAN:
        return 1.0 if is_yes else 0.0
    return 1.0 if is_yes else 0.0


def _generic_damage_counter(ctx: Ctx, pokemon: Pokemon | None, o: Option) -> float:
    """ダメカンを置く／移す先の既定。

    - 相手のポケモンなら「あと少しで倒せる」ものを最優先、次にサイドを多く取れるもの。
    - 自分のポケモンなら「一番減っていない（HPに余裕がある）」ものを選ぶ。
    """
    if pokemon is None:
        return 0.0
    mine = o.playerIndex == ctx.me
    remain = ctx.select.remainDamageCounter * 10 if ctx.select.remainDamageCounter else 0
    if mine:
        return -1000.0 + pokemon.hp  # 自分に置くのは最後の手段。減っていない子に。
    score = 1000.0 * ko_prize(pokemon)
    if 0 < pokemon.hp <= max(10, remain):
        score += 50000.0  # このダメカンで倒しきれる
    score += 2000.0 - pokemon.hp
    return score


def score_option(ctx: Ctx, st: Strategy, o: Option) -> float:
    """1つの選択肢に点数をつける。デッキ固有のフックを呼び、無ければ共通の既定に落とす。"""
    override = st.special_score(ctx, o)
    if override is not None:
        return override

    ctxt = ctx.context

    if o.type == OptionType.NUMBER:
        # 引く枚数・置くダメカンの数などは基本的に多いほど良い。
        return float(o.number or 0)

    if o.type in (OptionType.YES, OptionType.NO):
        is_yes = o.type == OptionType.YES
        s = st.yes_no_score(ctx, is_yes)
        return _generic_yes_no(ctx, is_yes) if s is None else s

    if o.type == OptionType.PLAY:
        card = ctx.card_at(AreaType.HAND, o.index, ctx.me)
        if card is None:
            return SKIP
        return st.play_score(ctx, card.id, o)

    if o.type == OptionType.ATTACH:
        src = ctx.card_at(o.area, o.index, ctx.me)
        target = ctx.card_at(o.inPlayArea, o.inPlayIndex, ctx.me)
        if src is None or not isinstance(target, Pokemon):
            return SKIP
        return st.attach_score(ctx, src.id, target, o.inPlayArea == AreaType.ACTIVE)

    if o.type == OptionType.EVOLVE:
        src = ctx.card_at(o.area, o.index, ctx.me)
        target = ctx.card_at(o.inPlayArea, o.inPlayIndex, ctx.me)
        if src is None or not isinstance(target, Pokemon):
            return SKIP
        return st.evolve_score(ctx, src.id, target, o.inPlayArea == AreaType.ACTIVE)

    if o.type == OptionType.ABILITY:
        card = ctx.card_at(o.area, o.index, ctx.me)
        return st.ability_score(ctx, card.id if card else 0, o)

    if o.type == OptionType.RETREAT:
        return st.retreat_score(ctx)

    if o.type == OptionType.ATTACK:
        for ev in ctx.memo.get("attack_evals", []):
            if ev.attack_id == o.attackId:
                return st.attack_score(ctx, ev)
        return S_ATTACK

    if o.type == OptionType.END:
        return S_END

    if o.type == OptionType.DISCARD:
        # 場のカードを（効果で）トラッシュする。相手のものなら歓迎、自分のものは避ける。
        return 100.0 if o.playerIndex != ctx.me else SKIP

    if o.type in (OptionType.ENERGY, OptionType.ENERGY_CARD):
        # エネルギーを剥がす／捨てる。相手のものならエネルギーが多く乗っている方を優先。
        if o.playerIndex != ctx.me:
            target = ctx.card_at(o.area, o.index, o.playerIndex)
            return 100.0 + (len(target.energies) if isinstance(target, Pokemon) else 0)
        # 自分のエネルギーを捨てる場面（にげる等）は、遊んでいるものから。
        target = ctx.card_at(o.area, o.index, ctx.me)
        return 10.0 - (len(target.energies) if isinstance(target, Pokemon) else 0)

    if o.type == OptionType.TOOL_CARD:
        return 100.0 if o.playerIndex != ctx.me else SKIP

    if o.type == OptionType.SKILL:
        return 1.0

    if o.type == OptionType.SPECIAL_CONDITION:
        return 1.0

    if o.type != OptionType.CARD:
        return 0.0

    # --- 以下 OptionType.CARD（文脈で意味がまるで変わる） ---
    obj = ctx.card_at(o.area, o.index, o.playerIndex)
    if obj is None:
        return 0.0
    mine = o.playerIndex == ctx.me

    if ctxt in (
        SelectContext.SETUP_ACTIVE_POKEMON,
        SelectContext.TO_ACTIVE,
        SelectContext.SWITCH,
    ):
        if mine and isinstance(obj, Pokemon):
            return st.active_score(ctx, obj, o)
        if mine:
            # 手札から直接バトル場に置く場合（Pokemon ではなく Card で来る）
            return st.active_pref(ctx, obj.id)
        # 相手のポケモンを前に出す＝ボスの指令など
        idx, score = best_gust_target(ctx)
        return 10000.0 + score if idx == o.index else score

    if ctxt in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
        return st.bench_score(ctx, obj.id, o)

    if ctxt in (SelectContext.TO_HAND, SelectContext.LOOK):
        return st.want_in_hand(ctx, obj.id)

    if ctxt in (SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM, SelectContext.TO_PRIZE):
        return -st.want_in_hand(ctx, obj.id)

    if ctxt == SelectContext.NOT_MOVE:
        # 「動かさないカード」を選ぶ＝残したいカードを選ぶ。
        return st.want_in_hand(ctx, obj.id)

    if ctxt in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY, SelectContext.DAMAGE):
        target = obj if isinstance(obj, Pokemon) else None
        s = st.damage_counter_score(ctx, target, o) if target is not None else None
        return _generic_damage_counter(ctx, target, o) if s is None else s

    if ctxt in (SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.HEAL):
        if not isinstance(obj, Pokemon):
            return 0.0
        # 傷が深く、かつ価値の高い（サイドを多く取られる）ポケモンから回復する。
        missing = obj.maxHp - obj.hp
        return (1000.0 if mine else -1000.0) + missing + ko_prize(obj) * 200.0

    if ctxt in (SelectContext.ATTACH_FROM, SelectContext.EFFECT_TARGET):
        # 「どのポケモンに付けるか」。contextCard が付ける側のカード。
        if isinstance(obj, Pokemon):
            src_id = ctx.select.contextCard.id if ctx.select.contextCard else 0
            return st.attach_score(ctx, src_id, obj, o.area == AreaType.ACTIVE)
        return 0.0

    if ctxt == SelectContext.ATTACH_TO:
        return st.want_in_hand(ctx, obj.id)

    if ctxt in (SelectContext.EVOLVES_FROM,):
        if isinstance(obj, Pokemon):
            return st.evolve_score(ctx, ctx.select.contextCard.id if ctx.select.contextCard else 0, obj,
                                   o.area == AreaType.ACTIVE)
        return 0.0

    if ctxt in (SelectContext.EVOLVES_TO,):
        return st.want_in_hand(ctx, obj.id)

    if ctxt == SelectContext.DEVOLVE:
        return 1000.0 if not mine else SKIP

    # 未知の文脈: 自分のものを優先し、あとは HP の高いものを選ぶ無難な既定。
    if _DEBUG:
        print(f"[rule_agent] 未処理の文脈 context={ctxt!r} option={o!r}", flush=True)
    return (100.0 if mine else 0.0) + (obj.hp if isinstance(obj, Pokemon) else 0)


def choose(ctx: Ctx, st: Strategy) -> list[int]:
    """全選択肢に点数をつけ、高い順に minCount〜maxCount 個返す。"""
    select = ctx.select
    scores = [score_option(ctx, st, o) for o in select.option]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    out: list[int] = []
    for i in order:
        if len(out) >= select.maxCount:
            break
        if scores[i] < 0 and len(out) >= select.minCount:
            break
        out.append(i)
    return out


class RuleAgent:
    """Strategy を `agent(obs) -> list[int]` に仕立てる。"""

    def __init__(self, strategy: Strategy):
        self.st = strategy
        self.deck = read_deck(strategy.deck_path)

    def __call__(self, obs_dict) -> list[int]:
        obs = obs_dict if isinstance(obs_dict, Observation) else to_observation_class(obs_dict)
        if obs.select is None:
            return list(self.deck)

        ctx = build_ctx(obs, self.deck)
        self.st.prepare(ctx)
        ctx.memo["attack_evals"] = evaluate_attacks(ctx, bonus=self.st.attack_bonus(ctx))
        return choose(ctx, self.st)
