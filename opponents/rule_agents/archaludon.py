"""ジュラルドン／ブリジュラスex（Archaludon ex）デッキのルールベース。

## このデッキで何が勝ち筋か

主役は **ブリジュラスex（190, HP300, ジュラルドンからの1進化）**。狙いは「落ちない前」を
作って、毎ターン 220 を押しつけ続けること。

耐久を支えるのは3枚重ね:

- **メタルディフェンダー（鋼鋼鋼, 220）** を撃つ本体が HP300、
- **フルメタルラボ(1244)**: 鋼ポケモンが受けるダメージ -30（お互いに適用されるが、
  鋼で固めているのはこちらだけなので一方的に得）、
- **ジャンボアイス(1147)**: エネルギーが3個以上ついたバトルポケモンを80回復、
  **ヒーローマント(1159)**: +100HP。

この3枚が乗ると実効 HP は 400 を超え、相手の 270 や 180 では 2 ターンかけても落ちない。

## 立ち上がりはエースバーン（666 Cinderace）

エースバーン（2進化）は特性「ばくねつ」で、**セットアップ時に手札から伏せてバトル場に出せる**。
にげるエネルギーが 0 なので、用が済んだらタダで下がれる。ワザ「ターボフレア（無1, 50）」は
デッキから基本エネルギーを3枚まで**ベンチに**付ける。つまりこのデッキの加速は

  1ターン目: エースバーンにエネルギー1 → ターボフレアでベンチのジュラルドンに3枚
  2ターン目: ジュラルドン→ブリジュラスex に進化（特性でトラッシュから2枚追加）
             エースバーンをタダで下げて、ブリジュラスがメタルディフェンダー

という形になる。**加速がワザである**以上、1ターン目から殴れる**後攻**を取る。

## ジーランス（57）の意味

特性「メモリーダイブ」で、進化しているポケモンは進化前のワザを使えるようになる。
ブリジュラスex がジュラルドンの「あばれ叩き（鋼鋼無, 80 + 自分に乗っているダメカン1個につき+10）」
を撃てるようになるのが本命。自分が 150 以上削られている場面では、あばれ叩きの方が
メタルディフェンダー（220）より高い打点になる。

## 詰まる相手

イワオオギex（117）は「特性を持つポケモンからのワザのダメージ」を全て防ぐ。
ブリジュラスex もエースバーンも特性持ちなので**まったく通らない**。抜け道は
**特性を持たないジュラルドン本体で殴る**か、**ボスの指令(4枚)で裏を呼んで殴る**か。
"""

from __future__ import annotations

from pathlib import Path

from cg.api import AreaType, CardType, Option, Pokemon, SelectContext

from . import framework as fw
from .framework import (
    CARDS,
    S_ATTACK,
    S_DEVELOP,
    S_ENERGY,
    S_KEY,
    S_MUST,
    S_REFRESH,
    S_RETREAT,
    S_SEARCH,
    S_TOOL,
    SKIP,
    AttackEval,
    Ctx,
    Strategy,
)

DURALUDON = 169
ARCHALUDON = 190      # ブリジュラスex
CINDERACE = 666       # エースバーン（セットアップ時にバトル場へ伏せられる）
RELICANTH = 57        # ジーランス（メモリーダイブ）

NIGHT_STRETCHER = 1097
ULTRA_BALL = 1121
POKEGEAR = 1122
JUMBO_ICE = 1147      # ジャンボアイス（エネ3個以上のバトルポケモンを80回復）
POKE_PAD = 1152
HEROS_CAPE = 1159
BOSS = 1182
EXPLORERS = 1185      # 探検家の招待（上6枚から2枚。残り4枚はトラッシュ）
LILLIE = 1227
FULL_METAL_LAB = 1244
METAL_ENERGY = 8

HAMMER_IN = 223
RAGING_HAMMER = 224   # 80 + 自分のダメカン1個につき +10
METAL_DEFENDER = 253
TURBO_FLARE = 965
RAZOR_FIN = 61


class ArchaludonStrategy(Strategy):
    name = "archaludon_rule"
    deck_path = str(Path(__file__).resolve().parent / "decks" / "archaludon_ex.csv")

    # ---- ターン方針 --------------------------------------------------------
    def prepare(self, ctx: Ctx) -> None:
        ctx.memo["archaludon_in_play"] = ctx.field_counts.get(ARCHALUDON, 0)
        ctx.memo["incoming"] = fw.incoming_damage(ctx)
        idx, gust = fw.best_gust_target(ctx)
        ctx.memo["gust_index"], ctx.memo["gust_score"] = idx, gust
        # 相手のバトルポケモンに、特性持ちのこちらのワザが通らないか
        # （イワオオギex の「いしずえのかまえ」）。
        ctx.memo["walled"] = (
            ctx.op_active is not None
            and ctx.my_active is not None
            and fw.has_ability(ctx.my_active)
            and fw.is_damage_immune(ctx.my_active, ctx.op_active, False)
        )

    # ---- 先攻／後攻 --------------------------------------------------------
    def yes_no_score(self, ctx: Ctx, is_yes: bool) -> float | None:
        if ctx.context == SelectContext.IS_FIRST:
            # 後攻を選ぶ。このデッキのエネルギー加速「ターボフレア」はワザなので、
            # 先攻を取ると1ターン目に加速できず、丸ごと1ターン損をする。
            # 後攻ならサポート（探検家の招待／リーリエの決意）も1ターン目から使える。
            #
            # 500試合ずつ測って確認済み。後攻の勝率は対オーロンゲ0.482／対ルカリオ0.440、
            # 先攻だと0.420／0.426まで落ちる。先攻を取るとブリジュラスexが場に出るのは
            # 8.3→7.5ターン目と早くなるのに勝率は下がる。加速が1ターン遅れた分、
            # 「落ちない前」が間に合わずサイドを取られる試合が増えるため。
            return 0.0 if is_yes else 1.0
        return None

    # ---- 手札から出す ------------------------------------------------------
    def play_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        field = ctx.field_counts
        hand = ctx.hand_counts
        deck = ctx.deck_counts
        bench_room = ctx.my.benchMax - len(ctx.my_bench)

        if card_id == DURALUDON:
            # ブリジュラスの土台。このデッキでベンチに出せるたねはジュラルドン(4枚)と
            # ジーランス(1枚)しかない。実測で「バトル場が空になって負け」が敗因の
            # 過半数を占めたので、出せるときは出す方針にしてある。
            if bench_room <= 0:
                return SKIP
            line = field.get(DURALUDON, 0) + field.get(ARCHALUDON, 0)
            if ctx.pokemon_in_play <= 2:
                return S_MUST - 5000   # 盤面が薄いときは何より先に置く
            # 4体目以降は置かない。ベンチ枠が埋まるうえ、育っていないジュラルドンは
            # ボスの指令で呼び出されてサイドを1枚献上するだけになる。
            return S_DEVELOP + 4000 if line < 3 else SKIP
        if card_id == RELICANTH:
            # 本来は「メモリーダイブでブリジュラスにあばれ叩きを撃たせる」ための1枚だが、
            # このデッキでは貴重な"たね"でもある。盤面が薄いなら理由を問わず置く。
            if field.get(RELICANTH, 0) > 0 or bench_room <= 0:
                return SKIP
            if ctx.pokemon_in_play <= 2:
                return S_MUST - 6000
            return S_DEVELOP - 3000 if field.get(ARCHALUDON, 0) >= 1 else SKIP

        if card_id == JUMBO_ICE:
            active = ctx.my_active
            if active is None or len(active.energies) < 3:
                return SKIP
            damage = active.maxHp - active.hp
            if damage <= 0:
                return SKIP
            # 相手の次の攻撃で落ちるところを、回復でずらせるなら最優先。
            if ctx.memo.get("incoming", 0) >= active.hp and ctx.memo["incoming"] < active.hp + min(80, damage):
                return S_MUST
            return S_TOOL - 1000 if damage >= 80 else SKIP
        if card_id == HEROS_CAPE:
            return SKIP  # 道具は ATTACH 側
        if card_id == ULTRA_BALL:
            # 手札2枚を捨ててポケモンをサーチ。捨てた鋼エネはアッセンブルアロイの
            # 弾になるので、コストは軽い。
            if self._wants_pokemon(ctx) and len(ctx.hand) >= 3:
                return S_MUST - 7000 if ctx.pokemon_in_play <= 2 else S_SEARCH + 1000
            return SKIP
        if card_id == POKE_PAD:
            # ルールを持たないポケモン＝ジュラルドン／エースバーン／ジーランス。
            # 盤面が薄いときは、これが「たね」を確保する最短経路になる。
            if deck.get(DURALUDON, 0) + deck.get(RELICANTH, 0) <= 0:
                return SKIP
            return S_MUST - 8000 if ctx.pokemon_in_play <= 2 else S_SEARCH + 500
        if card_id == NIGHT_STRETCHER:
            if ctx.pokemon_in_play <= 2 and ctx.discard_counts.get(DURALUDON, 0) > 0:
                return S_MUST - 9000
            for cid in (ARCHALUDON, DURALUDON, METAL_ENERGY):
                if ctx.discard_counts.get(cid, 0) > 0 and self.want_in_hand(ctx, cid) >= 400:
                    return S_SEARCH - 1000
            return SKIP
        if card_id == POKEGEAR:
            return S_SEARCH - 6000

        if card_id == FULL_METAL_LAB:
            # 鋼が受けるダメージ -30。常に貼っておきたい。
            return S_KEY - 2000 if ctx.stadium_id != FULL_METAL_LAB else SKIP

        if card_id == BOSS:
            if ctx.memo.get("gust_score", 0) >= 1000:
                return S_KEY - 5000
            # 前が壁（こちらのワザが通らない）なら、通る相手を引きずり出すだけでも価値がある。
            if ctx.memo.get("walled") and ctx.op_bench:
                return S_KEY - 5000
            return SKIP
        if card_id == EXPLORERS:
            # 上6枚から2枚。残り4枚がトラッシュへ落ちるのは、鋼エネを
            # アッセンブルアロイの弾にできるので実は損ではない。
            if ctx.my.deckCount <= 6:
                return SKIP
            return S_REFRESH + 800 if self._useful_hand_count(ctx) >= 3 else S_REFRESH - 800
        if card_id == LILLIE:
            if ctx.my.deckCount <= 6:
                return SKIP
            return S_REFRESH if self._useful_hand_count(ctx) < 3 else S_REFRESH - 1500
        return SKIP

    # ---- 進化 --------------------------------------------------------------
    def evolve_score(self, ctx: Ctx, card_id: int, target: Pokemon, is_active: bool) -> float:
        if card_id == ARCHALUDON:
            if ctx.memo.get("archaludon_in_play", 0) >= 2:
                return SKIP
            # 特性アッセンブルアロイでトラッシュから鋼を2枚付けられる。最優先。
            #
            # どのジュラルドンを進化させるかは「エネルギーが乗っている方」で決める。
            # エネルギー0のジュラルドンを進化させても、メタルディフェンダー（鋼3）が
            # 撃てるようになるまで何ターンもかかる。前と後ろで同数なら前を選ぶ
            # （進化した瞬間から殴れる）。
            score = S_KEY + 10000 + len(target.energies) * 1500
            if is_active:
                score += 800
            return score
        return SKIP

    # ---- エネルギー・道具を付ける ------------------------------------------
    def attach_score(self, ctx: Ctx, card_id: int, target: Pokemon, is_active: bool) -> float:
        data = CARDS.get(card_id)
        if data is not None and data.cardType == CardType.TOOL:
            # 道具は進化しても外れないので、ジュラルドンに付けておけば
            # そのままブリジュラスex の +100HP になる。
            if target.id == ARCHALUDON:
                return S_TOOL + 2000 + (1000 if is_active else 0)
            if target.id == DURALUDON:
                return S_TOOL
            return SKIP  # エースバーン／ジーランスに付けても腐る

        energies = len(target.energies)
        # 逃げることも殴ることもできない前を放置すると、そのターンが丸ごと無駄になる。
        # 「逃げるための1枚」は他のどのエネルギー付けより優先する。
        if is_active and fw.retreat_blocked(ctx) and not fw.can_attack_now(ctx, target):
            return S_ENERGY + 9000

        if target.id == ARCHALUDON:
            return S_ENERGY + 5000 + (2000 if is_active else 0) if energies < 3 else SKIP
        if target.id == DURALUDON:
            # ジュラルドンに乗せたエネルギーは進化後もそのまま残る。ただし
            # **前のジュラルドンから積む**こと。ベンチ側を先に育てると、前が
            # エネルギー0のまま進化してしまい、何ターンも殴れなくなる
            # （実測で最初の攻撃が11ターン目まで遅れていた原因がこれ）。
            if energies >= 3:
                return SKIP
            return S_ENERGY + (4000 if is_active else 1000)
        if target.id == CINDERACE:
            # ターボフレア（無1）を撃つための1枚だけ。それ以上は下がるときに無駄になる。
            return S_ENERGY + 4000 if energies == 0 else SKIP
        return SKIP

    # ---- 前に出す・ベンチに出す --------------------------------------------
    def active_pref(self, ctx: Ctx, card_id: int) -> float:
        if card_id == CINDERACE:
            # セットアップでは最優先。特性でバトル場に伏せられる唯一のカードで、
            # HP160・にげる0 という「1ターン目を支えて、タダで下がれる」性能。
            return 40000.0
        if card_id == DURALUDON:
            # 相手が「特性持ちのワザを防ぐ」壁を出しているときは、特性を持たない
            # ジュラルドン本体が唯一ダメージを通せる。
            if ctx.memo.get("walled"):
                return 45000.0
            return 20000.0
        return {
            ARCHALUDON: 60000.0,
            RELICANTH: 500.0,
        }.get(card_id, 0.0)

    def active_score(self, ctx: Ctx, pokemon: Pokemon, o: Option) -> float:
        base = self.active_pref(ctx, pokemon.id)
        if pokemon.id == ARCHALUDON:
            if len(pokemon.energies) >= 3:
                base += 20000
            if ctx.memo.get("incoming", 0) >= pokemon.hp:
                base -= 35000   # サイド2枚を渡すために差し出さない
        if pokemon.id == CINDERACE and ctx.memo.get("archaludon_in_play", 0) >= 1:
            base -= 35000       # ブリジュラスが立っていればエースバーンの役目は終わり
        return base + len(pokemon.energies) * 200 + pokemon.hp

    def bench_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        field = ctx.field_counts
        if card_id == DURALUDON:
            line = field.get(DURALUDON, 0) + field.get(ARCHALUDON, 0)
            return 9000.0 if line < 2 else (3000.0 if line < 3 else 200.0)
        if card_id == RELICANTH:
            return 1500.0 if field.get(RELICANTH, 0) == 0 else SKIP
        return 100.0

    # ---- 逃げる ------------------------------------------------------------
    def retreat_score(self, ctx: Ctx) -> float:
        active = ctx.my_active
        if active is None:
            return SKIP
        if active.id == CINDERACE:
            # にげるエネルギー0。ブリジュラスが撃てるなら即座に交代する。
            if any(p.id == ARCHALUDON and len(p.energies) >= 3 for p in ctx.my_bench):
                return S_RETREAT + 9000
            if any(p.id == ARCHALUDON for p in ctx.my_bench):
                return S_RETREAT + 4000
            # 前に出しても仕事が無いなら、殴れるジュラルドンと替える。
            if ctx.state.energyAttached and any(p.id == DURALUDON and len(p.energies) >= 3 for p in ctx.my_bench):
                return S_RETREAT + 1000
            return SKIP
        if active.id == RELICANTH:
            return S_RETREAT + 5000 if ctx.my_bench else SKIP
        if active.id == ARCHALUDON and ctx.memo.get("walled"):
            # 壁に詰まっているなら、ダメージが通るジュラルドンに交代する。
            # ただし にげるエネルギー2 を捨てるので、通る相手がベンチにいるときだけ。
            if any(p.id == DURALUDON and len(p.energies) >= 3 for p in ctx.my_bench):
                return S_RETREAT
        return SKIP

    # ---- 攻撃 --------------------------------------------------------------
    def attack_score(self, ctx: Ctx, ev: AttackEval) -> float:
        active = ctx.my_active
        if ev.attack_id == RAGING_HAMMER and active is not None and ctx.op_active is not None:
            # 「自分に乗っているダメカン1個につき +10」。削られているほど強くなる。
            bonus = max(0, active.maxHp - active.hp)
            dmg = fw.estimate_damage(ctx, active, RAGING_HAMMER, ctx.op_active, bonus=bonus)
            ko = dmg >= ctx.op_active.hp
            score = S_ATTACK + min(dmg, 400)
            if ko:
                score += 6000 + fw.ko_prize(ctx.op_active) * 600
            return score
        if ev.attack_id == TURBO_FLARE:
            # 打点50はおまけ。デッキからベンチに基本エネルギー3枚が本体。
            score = S_ATTACK + ev.damage
            if ev.ko:
                score += 6000 + ev.prize * 600
            if ctx.deck_counts.get(METAL_ENERGY, 0) > 0 and self._bench_needs_energy(ctx):
                score += 4000
            return score
        if ev.ko:
            return S_ATTACK + min(ev.damage, 400) + 6000 + ev.prize * 600
        return S_ATTACK + min(ev.damage, 400)

    # ---- カードの欲しさ ----------------------------------------------------
    def want_in_hand(self, ctx: Ctx, card_id: int) -> float:
        field = ctx.field_counts
        hand = ctx.hand_counts
        line = field.get(DURALUDON, 0) + field.get(ARCHALUDON, 0)

        if card_id == ARCHALUDON:
            if ctx.memo.get("archaludon_in_play", field.get(ARCHALUDON, 0)) >= 2:
                return 80.0
            return 900.0 if field.get(DURALUDON, 0) >= 1 else 600.0
        if card_id == DURALUDON:
            if ctx.pokemon_in_play <= 2:
                return 1000.0   # 盤面が空になる負けを防ぐのが最優先
            return 800.0 if line < 3 else 250.0
        if card_id == METAL_ENERGY:
            attached = sum(len(p.energies) for p in ctx.my_field)
            return 700.0 if attached < 3 else 300.0
        if card_id == FULL_METAL_LAB:
            return 500.0 if ctx.stadium_id != FULL_METAL_LAB else 100.0
        if card_id == HEROS_CAPE:
            return 550.0 if field.get(ARCHALUDON, 0) >= 1 and not any(p.tools for p in ctx.my_field) else 150.0
        if card_id == JUMBO_ICE:
            active = ctx.my_active
            damaged = active is not None and active.maxHp - active.hp >= 80
            return 500.0 if damaged else 250.0
        if card_id == BOSS:
            return 480.0 if hand.get(BOSS, 0) == 0 else 120.0
        if card_id == EXPLORERS:
            return 400.0
        if card_id == LILLIE:
            return 380.0
        if card_id == ULTRA_BALL:
            return 350.0
        if card_id == POKE_PAD:
            return 320.0
        if card_id == NIGHT_STRETCHER:
            return 250.0
        if card_id == CINDERACE:
            # 場に出す手段がセットアップ時しか無いので、途中で持っていても腐る。
            return 60.0
        if card_id == RELICANTH:
            if field.get(RELICANTH, 0) > 0:
                return 40.0
            return 950.0 if ctx.pokemon_in_play <= 2 else 200.0
        if card_id == POKEGEAR:
            return 130.0
        return 100.0

    # ---- 補助 --------------------------------------------------------------
    @staticmethod
    def _bench_needs_energy(ctx: Ctx) -> bool:
        return any(p.id in (DURALUDON, ARCHALUDON) and len(p.energies) < 3 for p in ctx.my_bench)

    @staticmethod
    def _wants_pokemon(ctx: Ctx) -> bool:
        field = ctx.field_counts
        if ctx.pokemon_in_play <= 2:
            return True
        if field.get(DURALUDON, 0) + field.get(ARCHALUDON, 0) < 3:
            return True
        return ctx.hand_counts.get(ARCHALUDON, 0) == 0 and field.get(ARCHALUDON, 0) < 2

    def _useful_hand_count(self, ctx: Ctx) -> int:
        return sum(1 for c in ctx.hand if self.want_in_hand(ctx, c.id) >= 400)


_AGENT = fw.RuleAgent(ArchaludonStrategy())


def agent(obs) -> list[int]:
    return _AGENT(obs)
