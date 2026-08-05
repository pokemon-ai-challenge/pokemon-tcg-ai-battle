"""メガルカリオex（Mega Lucario ex）デッキのルールベース。

リストは Kaggle エピソード 89945575 の player1（チーム Majkel1337, rank 2 / score 1194.3）。

## このデッキで何が勝ち筋か

主役は **メガルカリオex（678, HP340, リオルからの1進化）**。

- **メガブレイブ（闘闘, 270）**: ほぼ何でも一撃で落とすが、**次の自分の番は同じワザを
  使えない**。
- **オーラジャブ（闘, 130）**: 撃つとトラッシュから基本闘エネルギーを3枚まで
  **ベンチの**ポケモンに付けられる。

つまり「メガブレイブ → オーラジャブ → メガブレイブ …」と交互に撃つのが素の形で、
オーラジャブの番に次のアタッカー（ベンチのリオル）を仕上げる、というリズムで回る。
ワザのロック（次の番は使えない、という制限）はエンジンが選択肢から外してくれるので、
こちらで覚える必要はない。

## このリスト特有の事情

**リオルが3枚しか入っていないのに、メガルカリオexが4枚入っている。** つまり詰まる原因は
「進化先が引けない」ではなく「土台のリオルが足りない」側にある。リオルを持ってこられる
カードは ハイパーボール(1121, 何でも)、ファイティングゴング(1142, たね闘ポケモン)、
ポケパッド(1152, ルールを持たないポケモン) の3種10枚あるので、リオルが場に2体いない
うちはこれらを最優先で切る。

**スタジアムが1枚も入っていない。** 相手のスタジアム（スパイクタウンジム、フルメタルラボ）を
剥がす手段が無いので、相手のスタジアムは出たら出っぱなしになる前提で考える。

**ナンジャモ(1213 Judge)が4枚。** お互い手札を山に戻して4枚引く。こちらの手札が事故った
ときの立て直しと、相手が手札を抱えているときの妨害を兼ねる。ただし引ける枚数は
リーリエの決意(6枚)より少ないので、素の引き直しとしては後回しにする。

## サイド3枚を守る

メガ進化ポケモンexは**倒されるとサイドを3枚渡す**。相手のサイドは6枚なので、ルカリオを
2回倒されるとほぼ負ける。守る札は2枚:

- **ヒーローマント(1159)**: HP340→440。
- **ハクダンの想い(1229)**: ダメージを全回復する。付いていたエネルギーは手札に戻るが、
  戻ってきた1枚をその場で貼り直せばオーラジャブ（闘1）はまた撃てる。

どちらも「次の番に落とされる」ところまで削られてから使う。早く使うと手数を1回損する。

## 引き札

ルナトーン(675)「ムーンサイクル」は、ソルロック(676)が場にいるとき手札の基本闘エネルギーを
1枚トラッシュして3枚引く。**捨てた闘エネルギーはオーラジャブで回収できる**ので、実質
コスト無しのドロー3。ソルロックとルナトーンは1体ずつベンチに置く。
"""

from __future__ import annotations

from pathlib import Path

from cg.api import CardType, Option, Pokemon, SelectContext

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

RIOLU = 677           # 3枚しかない土台
LUCARIO = 678         # メガルカリオex（倒されるとサイド3枚）
MAKUHITA = 673
HARIYAMA = 674        # 進化時に相手ベンチを引きずり出す
LUNATONE = 675
SOLROCK = 676

ULTRA_BALL = 1121     # 手札2枚を捨てて、ポケモンを何でも1枚
SWITCH = 1123
POWER_PRO = 1141      # プレミアムパワープロ（このターン闘ポケモンのワザ+30）
GONG = 1142           # ファイティングゴング（基本闘エネ or たね闘ポケモン）
POKE_PAD = 1152
HEROS_CAPE = 1159     # ヒーローマント（+100HP）
BOSS = 1182
JUDGE = 1213          # ナンジャモ（お互い手札を戻して4枚）
LILLIE = 1227         # リーリエの決意（手札を戻して6枚）
WALLY = 1229          # ハクダンの想い（メガ進化exを全回復、エネは手札へ）
FIGHT_ENERGY = 6

AURA_JAB = 982
MEGA_BRAVE = 983
ACCEL_STAB = 981      # リオル「かそくづき」（次の番は同じワザを使えない）
WILD_PRESS = 978
COSMIC_BEAM = 980


class LucarioStrategy(Strategy):
    name = "lucario_rule"
    deck_path = str(Path(__file__).resolve().parent / "decks" / "mega_lucario_ex.csv")

    def __init__(self) -> None:
        # プレミアムパワープロを「このターンに何枚使ったか」を追うための記録。
        # 使ったアイテムはトラッシュに落ちるので、自分の番の開始時点との差で数える。
        self._turn_key: tuple[int, int] | None = None
        self._power_pro_base = 0

    # ---- ターン方針 --------------------------------------------------------
    def prepare(self, ctx: Ctx) -> None:
        key = (ctx.turn, ctx.me)
        if key != self._turn_key or ctx.turn <= 1:
            self._turn_key = key
            self._power_pro_base = ctx.discard_counts.get(POWER_PRO, 0)
        played = max(0, ctx.discard_counts.get(POWER_PRO, 0) - self._power_pro_base)
        ctx.memo["power_pro_played"] = played

        ctx.memo["lucario_in_play"] = ctx.field_counts.get(LUCARIO, 0)
        ctx.memo["incoming"] = fw.incoming_damage(ctx)
        idx, gust = fw.best_gust_target(ctx, bonus=30 * played)
        ctx.memo["gust_index"], ctx.memo["gust_score"] = idx, gust

        # 「あと何ダメージ足りないか」。プレミアムパワープロを切るかの判断に使う。
        shortfall = 0
        if ctx.op_active is not None:
            evals = fw.evaluate_attacks(ctx, bonus=30 * played)
            best = max((e.damage for e in evals), default=0)
            if evals and best < ctx.op_active.hp:
                shortfall = ctx.op_active.hp - best
        ctx.memo["shortfall"] = shortfall

    def attack_bonus(self, ctx: Ctx) -> int:
        return 30 * ctx.memo.get("power_pro_played", 0)

    # ---- 先攻／後攻 --------------------------------------------------------
    def yes_no_score(self, ctx: Ctx, is_yes: bool) -> float | None:
        if ctx.context == SelectContext.IS_FIRST:
            # 先攻を選ぶ。リオル→メガルカリオは1進化なので、先攻でも自分の2番目の
            # ターンにはメガブレイブが撃てる。先に殴り始められる方が、
            # 「サポートを1ターン我慢する」損より大きい。
            return 1.0 if is_yes else 0.0
        if ctx.context == SelectContext.ACTIVATE and ctx.select.contextCard is not None:
            if ctx.select.contextCard.id == LUNATONE:
                # 山札が細ったら引かない（山札切れは負け）。
                return (1.0 if is_yes else 0.0) if ctx.my.deckCount > 6 else (0.0 if is_yes else 1.0)
        return None

    # ---- 手札から出す ------------------------------------------------------
    def play_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        field = ctx.field_counts
        hand = ctx.hand_counts
        deck = ctx.deck_counts

        # --- たねポケモン ---
        if card_id == RIOLU:
            # このリストで一番数が少ない部品（3枚）。倒されると次のアタッカーが
            # 作れなくなるので、2体目までは必ず並べる。
            if ctx.bench_room <= 0:
                return SKIP
            line = field.get(RIOLU, 0) + field.get(LUCARIO, 0)
            if ctx.pokemon_in_play <= 2:
                return S_MUST - 5000
            return S_DEVELOP + 3000 if line < 2 else (S_DEVELOP - 2000 if line < 3 else SKIP)
        if card_id == SOLROCK:
            # ルナトーンの特性の条件。1体で足りる。
            if ctx.bench_room <= 0:
                return SKIP
            return S_DEVELOP + 2000 if field.get(SOLROCK, 0) == 0 else SKIP
        if card_id == LUNATONE:
            if field.get(LUNATONE, 0) > 0 or ctx.bench_room <= 0:
                return SKIP
            return S_DEVELOP + 1500 if field.get(SOLROCK, 0) > 0 or hand.get(SOLROCK, 0) > 0 else SKIP
        if card_id == MAKUHITA:
            # ハリテヤマの「がっぷりキャッチャー」を撃つための土台。
            if field.get(MAKUHITA, 0) + field.get(HARIYAMA, 0) > 0 or ctx.bench_room < 2:
                return SKIP
            return S_DEVELOP - 2000 if hand.get(HARIYAMA, 0) + deck.get(HARIYAMA, 0) > 0 else SKIP

        # --- アイテム ---
        if card_id == POWER_PRO:
            # 「あと30(60)足りない」ときにだけ切る。届かない上乗せは無駄打ち。
            short = ctx.memo.get("shortfall", 0)
            if 0 < short <= 30:
                return S_MUST
            if 30 < short <= 60 and hand.get(POWER_PRO, 0) >= 2:
                return S_MUST - 100
            return SKIP
        if card_id == ULTRA_BALL:
            # 手札2枚を捨ててポケモンを何でも1枚。捨てる闘エネはオーラジャブで
            # 回収できるので、コストは見た目より軽い。
            if len(ctx.hand) < 3 or not self._wants_pokemon(ctx):
                return SKIP
            return S_MUST - 6000 if ctx.pokemon_in_play <= 2 else S_SEARCH + 2000
        if card_id == GONG:
            # 基本闘エネ or たね闘ポケモン。リオルが足りない局面で最も強い。
            if deck.get(RIOLU, 0) > 0 or deck.get(FIGHT_ENERGY, 0) > 0:
                return S_SEARCH + 1500
            return SKIP
        if card_id == POKE_PAD:
            for cid in (RIOLU, SOLROCK, LUNATONE, MAKUHITA, HARIYAMA):
                if deck.get(cid, 0) > 0 and self.want_in_hand(ctx, cid) >= 400:
                    return S_SEARCH + 400
            return SKIP
        if card_id == SWITCH:
            # 前が詰まっている（殴れない子が前）ときの脱出用。
            return S_RETREAT + 3000 if self._stuck_active(ctx) else SKIP
        if card_id == HEROS_CAPE:
            return SKIP  # 道具は ATTACH 側で扱う

        # --- サポート ---
        if card_id == WALLY:
            # サイド3枚を守るための札。落とされる直前にだけ切る。
            return S_MUST if self._wally_target(ctx) else SKIP
        if card_id == BOSS:
            return S_KEY - 5000 if ctx.memo.get("gust_score", 0) >= 1000 else SKIP
        if card_id == LILLIE:
            # 手札を山に戻して6枚。やることを済ませてからの締め。
            return S_REFRESH if ctx.my.deckCount > 6 else SKIP
        if card_id == JUDGE:
            # 引ける枚数はリーリエ(6枚)より少ないので、素の引き直しとしては下。
            # 相手が手札を抱えているときは、妨害の価値がそれを上回る。
            if ctx.my.deckCount <= 6:
                return SKIP
            if ctx.op.handCount >= 7:
                return S_REFRESH + 1200
            if len(ctx.hand) <= 3:
                return S_REFRESH - 500
            return SKIP
        return SKIP

    # ---- 進化 --------------------------------------------------------------
    def evolve_score(self, ctx: Ctx, card_id: int, target: Pokemon, is_active: bool) -> float:
        if card_id == LUCARIO:
            if ctx.memo.get("lucario_in_play", 0) >= 2:
                return SKIP
            return S_KEY + 10000 + (3000 if is_active else 0)
        if card_id == HARIYAMA:
            # 進化時の特性で相手ベンチを引きずり出せる。倒せる的が奥にいるときに使う。
            if ctx.memo.get("gust_score", 0) >= 1000:
                return S_KEY - 4000
            return S_DEVELOP - 5000
        return SKIP

    # ---- エネルギー・道具を付ける ------------------------------------------
    def attach_score(self, ctx: Ctx, card_id: int, target: Pokemon, is_active: bool) -> float:
        data = CARDS.get(card_id)
        if data is not None and data.cardType == CardType.TOOL:
            # 道具は進化しても外れないので、リオルに付けても最終的にルカリオの +100HP になる。
            if target.id == LUCARIO:
                return S_TOOL + 2000 + (1000 if is_active else 0)
            if target.id == RIOLU:
                return S_TOOL
            return SKIP

        energies = len(target.energies)
        # ルナトーンやマクノシタが前に出てしまったとき、エネルギーが0だと
        # 「ワザも撃てない・にげることもできない」で盤面が止まる。逃げるための
        # 1枚を最優先で貼る。
        if is_active and fw.retreat_blocked(ctx) and not fw.can_attack_now(ctx, target):
            return S_ENERGY + 9000

        if target.id == LUCARIO:
            if energies >= 2:
                return SKIP  # メガブレイブに必要な2枚を超えて貼る意味は薄い
            return S_ENERGY + 5000 + (2000 if is_active else 0)
        if target.id == RIOLU:
            # リオルに貼ったエネルギーは進化してもそのまま残る。ベンチのリオルを
            # 育てておくと、ルカリオが倒された次の番からすぐ殴り返せる。
            if energies >= 2:
                return SKIP
            return S_ENERGY + (3000 if is_active else 1500)
        if target.id == SOLROCK:
            # コズミックビーム（闘1で70、弱点抵抗力を無視）は繋ぎになる。
            return S_ENERGY - 4000 if energies == 0 else SKIP
        if target.id == HARIYAMA:
            return S_ENERGY - 5000 if energies < 3 else SKIP
        return SKIP

    # ---- 特性 --------------------------------------------------------------
    def ability_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        if card_id == LUNATONE:
            # 手札の基本闘エネを1枚捨てて3枚引く。捨てた分はオーラジャブで回収できる。
            if ctx.hand_counts.get(FIGHT_ENERGY, 0) == 0 or ctx.my.deckCount <= 6:
                return SKIP
            # 今このエネルギーを場に貼りたいなら、捨てずに貼る方を優先する。
            if ctx.hand_counts.get(FIGHT_ENERGY, 0) == 1 and self._needs_energy_now(ctx):
                return SKIP
            return S_SEARCH + 2000
        return S_SEARCH

    # ---- 前に出す・ベンチに出す --------------------------------------------
    def active_pref(self, ctx: Ctx, card_id: int) -> float:
        return {
            LUCARIO: 60000.0,
            RIOLU: 20000.0,
            SOLROCK: 3000.0,
            HARIYAMA: 2500.0,
            MAKUHITA: 1200.0,
            LUNATONE: 800.0,     # 引き札要員。前に出したくない
        }.get(card_id, 0.0)

    def active_score(self, ctx: Ctx, pokemon: Pokemon, o: Option) -> float:
        base = self.active_pref(ctx, pokemon.id)
        if pokemon.id == LUCARIO:
            if len(pokemon.energies) >= 2:
                base += 20000
            # 傷んだルカリオを差し出してサイド3枚を渡すのは避ける。
            if ctx.memo.get("incoming", 0) >= pokemon.hp:
                base -= 45000
        return base + len(pokemon.energies) * 200 + pokemon.hp

    def bench_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        field = ctx.field_counts
        if card_id == RIOLU:
            line = field.get(RIOLU, 0) + field.get(LUCARIO, 0)
            return 8000.0 if line < 2 else 1000.0
        if card_id == SOLROCK:
            return 6000.0 if field.get(SOLROCK, 0) == 0 else SKIP
        if card_id == LUNATONE:
            return 5000.0 if field.get(LUNATONE, 0) == 0 else SKIP
        if card_id == MAKUHITA:
            return 1500.0 if field.get(MAKUHITA, 0) + field.get(HARIYAMA, 0) == 0 else SKIP
        return 100.0

    # ---- 逃げる ------------------------------------------------------------
    def retreat_score(self, ctx: Ctx) -> float:
        active = ctx.my_active
        if active is None:
            return SKIP
        if active.id in (LUNATONE, SOLROCK, MAKUHITA):
            if any(p.id == LUCARIO and len(p.energies) >= 1 for p in ctx.my_bench):
                return S_RETREAT + 5000
            if any(p.id == RIOLU for p in ctx.my_bench):
                return S_RETREAT
        return SKIP

    # ---- 攻撃 --------------------------------------------------------------
    def attack_score(self, ctx: Ctx, ev: AttackEval) -> float:
        score = S_ATTACK + min(ev.damage, 400)
        if ev.ko:
            score += 6000 + ev.prize * 600
        if ev.attack_id == AURA_JAB:
            # 倒しきれるならメガブレイブを温存してオーラジャブで倒す（次の番も
            # メガブレイブが撃てる）。
            if ev.ko:
                score += 900
            # トラッシュの闘エネをベンチに送れる分の上乗せ。ここを大きくすると
            # 「どちらでも倒せない場面」で 130 のオーラジャブが 270 のメガブレイブに
            # 勝ってしまう。打点差 140 を覆さない値にとどめる。
            if ctx.discard_counts.get(FIGHT_ENERGY, 0) > 0 and self._bench_needs_energy(ctx):
                score += 60
        if ev.attack_id == WILD_PRESS and not ev.ko:
            score -= 3000  # 自分に70。倒せないなら使わない
        if ev.attack_id == ACCEL_STAB:
            score += 40  # リオルで殴るのは繋ぎ。0ダメージの選択肢よりは上に
        return score

    # ---- カードの欲しさ ----------------------------------------------------
    def want_in_hand(self, ctx: Ctx, card_id: int) -> float:
        field = ctx.field_counts
        hand = ctx.hand_counts
        line = field.get(RIOLU, 0) + field.get(LUCARIO, 0)

        if card_id == RIOLU:
            # 3枚しかない土台。ここが切れると詰むので、ルカリオ本体より上に置く。
            if ctx.pokemon_in_play <= 2:
                return 1000.0
            return 850.0 if line < 2 else 200.0
        if card_id == LUCARIO:
            if ctx.memo.get("lucario_in_play", field.get(LUCARIO, 0)) >= 2:
                return 80.0
            return 880.0 if field.get(RIOLU, 0) >= 1 else 500.0
        if card_id == FIGHT_ENERGY:
            attached = sum(len(p.energies) for p in ctx.my_field)
            if attached < 2:
                return 700.0
            return 450.0 if hand.get(FIGHT_ENERGY, 0) <= 1 else 250.0
        if card_id == SOLROCK:
            return 500.0 if field.get(SOLROCK, 0) == 0 else 60.0
        if card_id == LUNATONE:
            return 480.0 if field.get(LUNATONE, 0) == 0 else 60.0
        if card_id == HEROS_CAPE:
            return 600.0 if line >= 1 and not any(p.tools for p in ctx.my_field) else 200.0
        if card_id == WALLY:
            return 550.0 if field.get(LUCARIO, 0) >= 1 else 120.0
        if card_id == ULTRA_BALL:
            return 450.0
        if card_id == POWER_PRO:
            return 420.0
        if card_id == GONG:
            return 400.0
        if card_id == BOSS:
            return 500.0 if hand.get(BOSS, 0) == 0 else 100.0
        if card_id == LILLIE:
            return 380.0
        if card_id == POKE_PAD:
            return 330.0
        if card_id == SWITCH:
            return 260.0
        if card_id == JUDGE:
            return 200.0
        if card_id in (MAKUHITA, HARIYAMA):
            return 180.0
        return 100.0

    # ---- 補助 --------------------------------------------------------------
    @staticmethod
    def _bench_needs_energy(ctx: Ctx) -> bool:
        return any(p.id in (RIOLU, LUCARIO) and len(p.energies) < 2 for p in ctx.my_bench)

    @staticmethod
    def _needs_energy_now(ctx: Ctx) -> bool:
        if ctx.state.energyAttached:
            return False
        return any(p.id in (RIOLU, LUCARIO) and len(p.energies) < 2 for p in ctx.my_field)

    @staticmethod
    def _stuck_active(ctx: Ctx) -> bool:
        active = ctx.my_active
        if active is None or active.id in (LUCARIO, RIOLU):
            return False
        return any(p.id in (LUCARIO, RIOLU) and len(p.energies) >= 1 for p in ctx.my_bench)

    @staticmethod
    def _wally_target(ctx: Ctx) -> bool:
        """ハクダンの想いを切る場面か。

        メガルカリオが「次の番に落ちる」ところまで削れているときだけ。回復しても
        エネルギーが手札に戻るので、まだ余裕があるうちに使うと手数を1回損する。
        """
        for p in ctx.my_field:
            if p.id == LUCARIO and p.maxHp - p.hp >= 120 and ctx.memo.get("incoming", 0) >= p.hp:
                return True
        return False

    @staticmethod
    def _wants_pokemon(ctx: Ctx) -> bool:
        field = ctx.field_counts
        if ctx.pokemon_in_play <= 2:
            return True
        if field.get(RIOLU, 0) + field.get(LUCARIO, 0) < 2:
            return True
        # リオルが立っているのにメガルカリオが手札にも場にも無い局面。
        if field.get(RIOLU, 0) >= 1 and field.get(LUCARIO, 0) == 0 and ctx.hand_counts.get(LUCARIO, 0) == 0:
            return True
        return field.get(SOLROCK, 0) == 0 or field.get(LUNATONE, 0) == 0


_AGENT = fw.RuleAgent(LucarioStrategy())


def agent(obs) -> list[int]:
    return _AGENT(obs)
