"""オーロンゲ（Marnie's Grimmsnarl ex）デッキのルールベース。

## このデッキで何が勝ち筋か

主役は **マーニーのオーロンゲex（648, HP320, たねから数えて2進化）**。強さの源は2つある。

1. **特性「パンクアップ」**: 手札から出して進化させたとき、デッキから基本悪エネルギーを
   最大5枚、自分の「マーニーの」ポケモンに付けられる。つまりこのデッキのエネルギー加速は
   「進化1回」に全部乗っている。手札からエネルギーを1枚ずつ貼る動きはおまけでしかない。
2. **HP320**。この環境でこれを1回で倒せるワザはほぼ無い。相手は2回殴る必要があるので、
   こちらは1体を立て続けに使い回すのが基本方針になる。

ワザは「シャドーバレット（悪悪, 180 + 相手ベンチ1体に30）」の1つだけ。180 は
メガルカリオex(340) もブリジュラスex(300) も一撃では落とせない。だから

- **ベンチへの30**と、
- **マンキー(112)「アドレナブレイン」でダメカンを押し付ける30**

の刻みをどれだけ積めるかが、実際の勝敗を分ける。

## 組み合わせて使う部品

- **ユキメノコ(104)「凍てつく霧」**: ポケモンチェックのたび、特性を持つポケモン全員に
  ダメカン1個。相手が特性デッキ（ブリジュラスex＝アッセンブル、メガルカリオの周辺、
  エースバーンなど）ならじわじわ効く。自分のオーロンゲexとマンキーにも乗るが、
  320 の器なら誤差で、しかも乗ったダメカンは次の項目で相手に返せる。
- **マンキー(112)「アドレナブレイン」**: 悪エネルギーが付いていれば、自分のポケモンの
  ダメカンを3個まで相手に移す。ユキメノコで自分に乗った分をそのまま相手に押し付けられる。
  「特性のダメカン」なのでワザのダメージを防ぐ効果（イワオオギなど）を貫通する。
  ここが、オーロンゲ側がイワオオギの壁に詰まないための唯一の抜け道。
- **スパイクタウンジム(1259)**: 毎ターン、デッキから「マーニーの」ポケモンを1枚サーチ。
  ベロバー／ギモー／オーロンゲの供給が安定するので、場に無ければ必ず貼る。

## 手なりの順番（点数の帯で表現している）

ふしぎなアメ → 進化 → ボスの指令 → ベンチ展開 → サーチ → エネルギー → 手札入れ替え → 攻撃。
1手打つたびにエンジンが選択をやり直させるので、点数の大小がそのまま行動順になる。
"""

from __future__ import annotations

from pathlib import Path

from cg.api import AreaType, CardType, Option, OptionType, Pokemon, SelectContext

from . import framework as fw
from .framework import (
    CARDS,
    S_ATTACK,
    S_DEVELOP,
    S_END,
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

IMPIDIMP = 646      # マーニーのベロバー
MORGREM = 647       # マーニーのギモー
GRIMMSNARL = 648    # マーニーのオーロンゲex
MUNKIDORI = 112     # マンキー（アドレナブレイン）
SNORUNT = 860       # ユキワラシ
FROSLASS = 104      # ユキメノコ（凍てつく霧）

RARE_CANDY = 1079
UNFAIR_STAMP = 1080
POFFIN = 1086       # なかよしポフィン
NIGHT_STRETCHER = 1097
POKEGEAR = 1122
TOOL_SCRAPPER = 1137
POKE_PAD = 1152
BOSS = 1182
PETREL = 1219       # ロケット団のペテル（トレーナーズサーチ）
LILLIE = 1227       # リーリエの決意（手札を戻して6枚ドロー）
DAWN = 1231         # ヒカリ（たね／1進化／2進化を1枚ずつサーチ）
SPIKEMUTH = 1259
DARK_ENERGY = 7

MARNIE_LINE = (IMPIDIMP, MORGREM, GRIMMSNARL)

SHADOW_BULLET = 937
FILCH = 934             # ベロバー「くすねる」: ドロー1
CORKSCREW_10 = 935
CORKSCREW_60 = 936


class GrimmsnarlStrategy(Strategy):
    name = "grimmsnarl_rule"
    deck_path = str(Path(__file__).resolve().parent / "decks" / "marnie_grimmsnarl_ex.csv")

    # ---- ターン方針 --------------------------------------------------------
    def prepare(self, ctx: Ctx) -> None:
        # 場のオーロンゲの本数。2体目までは保険として許すが、3体目は
        # 「倒されるとサイドを2枚渡す的」が増えるだけなので出さない。
        ctx.memo["grimmsnarl_in_play"] = ctx.field_counts.get(GRIMMSNARL, 0)
        ctx.memo["has_attacker"] = any(
            p.id == GRIMMSNARL and len(p.energies) >= 2 for p in ctx.my_field
        )
        # 自分の場に乗っているダメカンの総数。アドレナブレインを使う価値があるかの判断に使う。
        ctx.memo["my_damage"] = sum(p.maxHp - p.hp for p in ctx.my_field)
        # ふしぎなアメを「この番に実際に使えるか」。手札にあるかどうかではなく、
        # エンジンが選択肢として出しているかで判定する（最初の番や、その番に
        # 出したばかりのベロバーには使えないため）。
        ctx.memo["candy_available"] = any(
            o.type == OptionType.PLAY and ctx.card_id_of(o) == RARE_CANDY
            for o in ctx.select.option
        )
        idx, gust = fw.best_gust_target(ctx)
        ctx.memo["gust_index"], ctx.memo["gust_score"] = idx, gust

    # ---- 先攻／後攻 --------------------------------------------------------
    def yes_no_score(self, ctx: Ctx, is_yes: bool) -> float | None:
        if ctx.context == SelectContext.IS_FIRST:
            # 先攻を選ぶ。
            #
            # 当初は「先攻はサポートを使えないので、初動が全部サポートのこのデッキは
            # 後攻が良い」と考えて後攻にしていたが、実測で逆だった。メガルカリオ相手に
            # 200試合ずつ測ると、勝率は後攻 0.39〜0.40 に対して先攻 0.44〜0.46 で、
            # オーロンゲexが場に出るターンも 9.9 → 8.9 と1ターン早くなる。
            # ベロバー→ギモー→オーロンゲexの2段階を組むのに使える手番が1回増える方が、
            # サポート1枚を我慢する損より大きい。
            return 1.0 if is_yes else 0.0
        return None

    # ---- 手札から出す ------------------------------------------------------
    def play_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        deck = ctx.deck_counts
        hand = ctx.hand_counts
        field = ctx.field_counts
        grimm_in_play = ctx.memo["grimmsnarl_in_play"]

        # --- たねポケモン ---
        if card_id == IMPIDIMP:
            if ctx.bench_room <= 0:
                return SKIP
            # ベンチを埋めすぎるとボスの指令で弱いところを引きずり出されるが、
            # 「バトル場が空になって負け」の方がはるかに重い（実測でミラーの敗因の18%）。
            if ctx.pokemon_in_play <= 2:
                return S_MUST - 5000
            line_count = field.get(IMPIDIMP, 0) + field.get(MORGREM, 0) + field.get(GRIMMSNARL, 0)
            return S_DEVELOP + 500 if line_count < 3 else SKIP
        if card_id == MUNKIDORI:
            # アドレナブレイン要員は1体で足りる。
            return S_DEVELOP + 300 if field.get(MUNKIDORI, 0) == 0 else SKIP
        if card_id == SNORUNT:
            # ユキメノコにするための土台。相手に特性持ちがいるときだけ価値がある。
            if field.get(SNORUNT, 0) + field.get(FROSLASS, 0) > 0:
                return SKIP
            return S_DEVELOP if self._opponent_has_abilities(ctx) else SKIP
        if card_id == FROSLASS:
            return SKIP  # 進化なので EVOLVE 側で扱う

        # --- アイテム ---
        if card_id == RARE_CANDY:
            # ふしぎなアメはこのデッキの生命線。オーロンゲが手札にあり、進化先の
            # ベロバーが場にいるときだけ切る（エンジンが出す選択肢＝合法な場合のみ）。
            if hand.get(GRIMMSNARL, 0) >= 1 and grimm_in_play < 2:
                return S_KEY + 5000
            return SKIP
        if card_id == POFFIN:
            # HP70以下のたね2体をベンチへ。ベロバーとユキワラシが対象。
            if deck.get(IMPIDIMP, 0) + deck.get(SNORUNT, 0) <= 0:
                return SKIP
            line_count = field.get(IMPIDIMP, 0) + field.get(MORGREM, 0) + field.get(GRIMMSNARL, 0)
            return S_SEARCH + 2000 if line_count < 3 else SKIP
        if card_id == POKE_PAD:
            # ルールを持たないポケモン＝ベロバー／ギモー／マンキー／ユキワラシ。
            return S_SEARCH + 1000 if self._pad_target(ctx) else SKIP
        if card_id == NIGHT_STRETCHER:
            # トラッシュから拾い直す価値があるときだけ。
            for cid in (GRIMMSNARL, MORGREM, IMPIDIMP, MUNKIDORI, DARK_ENERGY):
                if ctx.discard_counts.get(cid, 0) > 0 and self.want_in_hand(ctx, cid) >= 400:
                    return S_SEARCH
            return SKIP
        if card_id == POKEGEAR:
            return S_SEARCH - 5000  # サポートを引きに行くだけなので他のサーチより後
        if card_id == TOOL_SCRAPPER:
            # ヒーローマント（+100HP）を剥がせるかどうかが全て。
            if any(p.tools for p in ctx.op_field):
                return S_TOOL
            return SKIP
        if card_id == UNFAIR_STAMP:
            # 倒された直後にしか使えない（エンジンが合法性を見ている）。手札が細いときに。
            return S_REFRESH + 2000 if len(ctx.hand) <= 4 else SKIP

        # --- スタジアム ---
        if card_id == SPIKEMUTH:
            # 毎ターン「マーニーの」ポケモンをサーチできる。相手のスタジアムを剥がす意味もある。
            return S_KEY - 1000 if ctx.stadium_id != SPIKEMUTH else SKIP

        # --- サポート ---
        if card_id == BOSS:
            if ctx.memo["gust_score"] >= 1000:
                return S_KEY - 5000  # 進化を済ませてから呼び出す
            return SKIP
        if card_id == DAWN:
            # たね／1進化／2進化を1枚ずつ。オーロンゲの通し牌をまとめて揃える最良の初動。
            if deck.get(GRIMMSNARL, 0) > 0 and grimm_in_play < 1:
                return S_SEARCH + 3000
            return SKIP
        if card_id == PETREL:
            # トレーナーズなら何でも。足りていない部品を1枚持ってこられる。
            if self._petrel_target(ctx):
                return S_SEARCH + 500
            return SKIP
        if card_id == LILLIE:
            # 手札を山に戻して6枚（サイド6枚残なら8枚）。やることを済ませた後の締め。
            return S_REFRESH

        # --- エネルギー ---
        if card_id == DARK_ENERGY:
            return SKIP  # 貼るのは ATTACH 側で扱う
        return SKIP

    # ---- 進化 --------------------------------------------------------------
    def evolve_score(self, ctx: Ctx, card_id: int, target: Pokemon, is_active: bool) -> float:
        if card_id == GRIMMSNARL:
            if ctx.memo["grimmsnarl_in_play"] >= 2:
                return SKIP
            # パンクアップでエネルギーが5枚飛んでくる。最優先。
            score = S_KEY + 10000
            if is_active:
                score += 2000  # 前のベロバー／ギモーから進化する方が1ターン早く殴れる
            return score
        if card_id == MORGREM:
            # アメが無いときの通常ルート。前を厚く（HP70→100）する意味もある。
            # 「この番にアメを実際に使える」ときだけギモーを飛ばす。手札にアメが
            # あるだけで飛ばすと、アメが使えない番（最初の番など）に何も進化せず
            # 1ターン丸ごと無駄になる。
            if (ctx.memo.get("candy_available")
                    and ctx.hand_counts.get(GRIMMSNARL, 0) >= 1
                    and ctx.memo["grimmsnarl_in_play"] < 2):
                return SKIP
            return S_KEY + (1000 if is_active else 0)
        if card_id == FROSLASS:
            return S_DEVELOP if self._opponent_has_abilities(ctx) else SKIP
        return SKIP

    # ---- エネルギー・道具を付ける ------------------------------------------
    def attach_score(self, ctx: Ctx, card_id: int, target: Pokemon, is_active: bool) -> float:
        data = CARDS.get(card_id)
        if data is not None and data.cardType == CardType.TOOL:
            return S_TOOL

        # ここは手札からの1枚貼りだけでなく、パンクアップ（デッキから最大5枚）の
        # 配り先選びにも使われる。5枚を1体に集めても2枚分しか使わないので、
        # 「前のオーロンゲに2枚 → 控えのベロバー／ギモーに2枚ずつ」の順で
        # 配られるように点数を組んである（控えが進化すれば、そのまま次のアタッカーになる）。
        energies = len(target.energies)
        # マンキーやユキワラシがバトル場に出てしまったとき、エネルギーが0だと
        # 「ワザも撃てない・にげることもできない」で盤面が完全に止まる。
        # 実測でこの形になった試合は5ターン以上何もできていなかったので、
        # 逃げるための1枚を最優先で貼る。
        if is_active and fw.retreat_blocked(ctx) and not fw.can_attack_now(ctx, target):
            return S_ENERGY + 9000

        if target.id == GRIMMSNARL:
            if energies >= 2:
                # 攻撃には2枚で足りる。3枚目以降は「にげる(2)」の保険程度。
                return S_ENERGY - 2000 if is_active else S_ENERGY - 4000
            return S_ENERGY + (3000 if is_active else 1000)
        if target.id == MUNKIDORI:
            # 悪エネが1枚付いて初めてアドレナブレインが使える。1枚だけ欲しい。
            return S_ENERGY + 2000 if energies == 0 else SKIP
        if target.id == MORGREM:
            # ギモーは進化してオーロンゲになる。乗せたエネルギーはそのまま引き継がれる。
            return S_ENERGY + (2000 if is_active else 500) if energies < 2 else SKIP
        if target.id == IMPIDIMP:
            # ベロバーもオーロンゲになる。前のオーロンゲが2枚持っている状態なら、
            # 余りは控えのベロバーに預けておく方が、次のアタッカーが1ターン早く立つ。
            if energies < 2:
                return S_ENERGY - 1000 - energies * 1500
            return SKIP
        return SKIP

    # ---- 特性 --------------------------------------------------------------
    def ability_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        if card_id == MUNKIDORI:
            # 自分の場にダメカンが乗っていて、かつ相手に押し付ける先があるときだけ。
            if ctx.memo["my_damage"] <= 0 or not ctx.op_field:
                return SKIP
            return S_SEARCH + 5000
        if card_id == SPIKEMUTH:
            # スタジアムの「1ターンに1回」サーチ。欲しい「マーニーの」ポケモンがいるとき。
            if any(ctx.deck_counts.get(cid, 0) > 0 for cid in MARNIE_LINE):
                return S_SEARCH + 3000
            return SKIP
        return S_SEARCH

    # ---- 前に出す・ベンチに出す --------------------------------------------
    def active_pref(self, ctx: Ctx, card_id: int) -> float:
        # 誰を前に置くか。殴れる子＞育てば殴れる子＞置物、の順。
        # 初期配置ではベロバーしか選べないことが多いが、マンキー／ユキワラシで
        # 始めると1ターン丸ごと無駄になるので明確に差をつけておく。
        return {
            GRIMMSNARL: 50000.0,
            MORGREM: 20000.0,
            IMPIDIMP: 12000.0,
            MUNKIDORI: 2000.0,     # 特性要員。前に出すと止まる
            FROSLASS: 1000.0,      # 特性要員
            SNORUNT: 500.0,
        }.get(card_id, 0.0)

    def active_score(self, ctx: Ctx, pokemon: Pokemon, o: Option) -> float:
        base = self.active_pref(ctx, pokemon.id)
        if pokemon.id == GRIMMSNARL and len(pokemon.energies) >= 2:
            base += 20000  # すぐ殴れる
        return base + len(pokemon.energies) * 300 + pokemon.hp

    def bench_score(self, ctx: Ctx, card_id: int, o: Option) -> float:
        field = ctx.field_counts
        line_count = field.get(IMPIDIMP, 0) + field.get(MORGREM, 0) + field.get(GRIMMSNARL, 0)
        if card_id == IMPIDIMP:
            return 5000.0 if line_count < 3 else 100.0
        if card_id == MUNKIDORI:
            return 4000.0 if field.get(MUNKIDORI, 0) == 0 else SKIP
        if card_id == SNORUNT:
            if field.get(SNORUNT, 0) + field.get(FROSLASS, 0) > 0:
                return SKIP
            return 3000.0 if self._opponent_has_abilities(ctx) else 200.0
        return 100.0

    # ---- 逃げる ------------------------------------------------------------
    def retreat_score(self, ctx: Ctx) -> float:
        active = ctx.my_active
        if active is None:
            return SKIP
        # 前がオーロンゲなら動かない（にげるエネルギー2は重く、320の器を下げる意味も無い）。
        if active.id == GRIMMSNARL:
            return SKIP

        # 前がこのターン殴れないなら、殴れる控えと替える。
        #
        # 当初は「前が置物（マンキー等）のとき」だけ交代していたが、実測すると
        # 「エネルギー1のギモー（コークスクリューパンチは悪2で足りない）」や
        # 「エネルギー1のマンキー」が前に残ったままターンを終える形が150試合で
        # 30回以上あった。判断の軸を種類ではなく「今殴れるかどうか」に変える。
        #
        # 「殴れるか」はエンジンがワザの選択肢を出しているかで見る。枚数だけで
        # 数えると、種類の合わないエネルギー（ユキワラシに悪エネルギー）や
        # 「次の番は使えない」制限を見落とす。
        if ctx.attack_options:
            return SKIP
        if any(p.id == GRIMMSNARL and len(p.energies) >= 2 for p in ctx.my_bench):
            return S_RETREAT + 5000
        if fw.bench_can_attack(ctx):
            return S_RETREAT + 2000
        # 今すぐ殴れる控えがいなくても、置物を前に置き続ける理由は無い。
        if active.id in (MUNKIDORI, FROSLASS, SNORUNT):
            if any(p.id in (GRIMMSNARL, MORGREM, IMPIDIMP) for p in ctx.my_bench):
                return S_RETREAT
        return SKIP

    # ---- 攻撃 --------------------------------------------------------------
    def attack_score(self, ctx: Ctx, ev: AttackEval) -> float:
        if ev.ko:
            return S_ATTACK + 5000 + ev.prize * 500
        if ev.attack_id == FILCH:
            # ベロバーの「くすねる」はダメージ0だが1枚引ける。10ダメージより価値がある。
            return S_ATTACK + 60
        if ev.attack_id == SHADOW_BULLET:
            # ベンチへの30も込みで評価する（相手ベンチを削るのがこのデッキの刻み方）。
            return S_ATTACK + ev.damage + 30
        return S_ATTACK + min(ev.damage, 400)

    # ---- ダメカン・ベンチ狙撃 ----------------------------------------------
    def damage_counter_score(self, ctx: Ctx, pokemon: Pokemon, o: Option) -> float | None:
        mine = o.playerIndex == ctx.me
        if mine:
            # アドレナブレインの移動元。傷が深い子から剥がす（＝オーロンゲを長持ちさせる）。
            return -1000.0 + (pokemon.maxHp - pokemon.hp)
        if fw.is_damage_immune(None, pokemon, o.area == AreaType.BENCH):
            return SKIP  # テラス（イワオオギ）などベンチで無敵の相手には無駄
        remain = (ctx.select.remainDamageCounter or 0) * 10
        # シャドーバレットのベンチ30 / アドレナブレインの30 をどこに置くか。
        step = max(remain, 30)
        score = 1000.0 * fw.ko_prize(pokemon)
        if pokemon.hp <= step:
            score += 50000.0  # これで倒しきれる
        else:
            # 「あと1回で落ちる」ところまで詰められる相手を優先する。
            score += (pokemon.maxHp - pokemon.hp) * 2.0 + len(pokemon.energies) * 100.0
        return score

    # ---- カードの欲しさ ----------------------------------------------------
    def want_in_hand(self, ctx: Ctx, card_id: int) -> float:
        field = ctx.field_counts
        hand = ctx.hand_counts
        grimm_in_play = ctx.memo.get("grimmsnarl_in_play", field.get(GRIMMSNARL, 0))
        line_count = field.get(IMPIDIMP, 0) + field.get(MORGREM, 0) + field.get(GRIMMSNARL, 0)

        if card_id == GRIMMSNARL:
            if grimm_in_play >= 2:
                return 50.0
            # アメが手札にあるなら即座に盤面が完成する。
            return 900.0 if hand.get(RARE_CANDY, 0) >= 1 or field.get(MORGREM, 0) >= 1 else 700.0
        if card_id == RARE_CANDY:
            if grimm_in_play >= 2:
                return 30.0
            return 850.0 if hand.get(GRIMMSNARL, 0) >= 1 else 600.0
        if card_id == MORGREM:
            return 400.0 if field.get(IMPIDIMP, 0) >= 1 and grimm_in_play == 0 else 120.0
        if card_id == IMPIDIMP:
            return 650.0 if line_count < 2 else 150.0
        if card_id == MUNKIDORI:
            return 450.0 if field.get(MUNKIDORI, 0) == 0 else 60.0
        if card_id == SNORUNT:
            if field.get(SNORUNT, 0) + field.get(FROSLASS, 0) > 0:
                return 40.0
            return 300.0 if self._opponent_has_abilities(ctx) else 80.0
        if card_id == FROSLASS:
            return 350.0 if field.get(SNORUNT, 0) >= 1 else 60.0
        if card_id == DAWN:
            return 700.0 if grimm_in_play == 0 else 100.0
        if card_id == SPIKEMUTH:
            return 300.0 if ctx.stadium_id != SPIKEMUTH else 80.0
        if card_id == BOSS:
            # 終盤の詰めに使う。手札に1枚は温存したい。
            return 500.0 if hand.get(BOSS, 0) == 0 else 100.0
        if card_id == PETREL:
            return 400.0
        if card_id == LILLIE:
            return 380.0
        if card_id == POFFIN:
            return 420.0 if line_count < 3 else 90.0
        if card_id == POKE_PAD:
            return 330.0
        if card_id == NIGHT_STRETCHER:
            return 250.0
        if card_id == DARK_ENERGY:
            # 原則としてパンクアップで一気に付くので、手札のエネルギーはさほど要らない。
            #
            # ただし「前がエネルギー0で、ワザも撃てず逃げることもできない」局面だけは別。
            # このとき必要なのは悪エネルギー1枚だけで、それが手札に無いために
            # 何ターンも止まる形が実測で最も多かった（150試合でベロバー83回、マンキー43回）。
            if fw.active_is_stalled(ctx) and hand.get(DARK_ENERGY, 0) == 0:
                return 950.0
            attached = sum(len(p.energies) for p in ctx.my_field)
            return 300.0 if attached < 2 else 90.0
        if card_id == UNFAIR_STAMP:
            return 150.0
        if card_id == POKEGEAR:
            return 140.0
        if card_id == TOOL_SCRAPPER:
            return 200.0 if any(p.tools for p in ctx.op_field) else 50.0
        return 100.0

    # ---- 補助 --------------------------------------------------------------
    @staticmethod
    def _opponent_has_abilities(ctx: Ctx) -> bool:
        """ユキメノコ（凍てつく霧）を立てる価値があるか。

        この特性は「特性を持つポケモン**全員**に」ダメカンを乗せる。自分の場にも
        オーロンゲex（パンクアップ）とマンキー（アドレナブレイン）がいるので、
        相手より自分の方が特性持ちが多いと、自分だけが削れていくことになる。

        実際、メガルカリオex は特性を持たない。ルカリオ相手にユキメノコを出すと、
        相手の主役には1ダメージも入らないまま、こちらのオーロンゲex とマンキーだけが
        毎ターン削られる。だから「頭数を比べて、相手の方が多いときだけ出す」。
        相手のバトルポケモンが特性持ちなら、そこを削れる価値が大きいので1体分上乗せする。
        """
        opp = sum(1 for p in ctx.op_field if fw.has_ability(p))
        if ctx.op_active is not None and fw.has_ability(ctx.op_active):
            opp += 1
        mine = sum(1 for p in ctx.my_field if fw.has_ability(p) and p.id != FROSLASS)
        return opp > mine

    @staticmethod
    def _pad_target(ctx: Ctx) -> bool:
        for cid in (IMPIDIMP, MORGREM, MUNKIDORI, SNORUNT):
            if ctx.deck_counts.get(cid, 0) > 0:
                return True
        return False

    @staticmethod
    def _petrel_target(ctx: Ctx) -> bool:
        for cid in (RARE_CANDY, SPIKEMUTH, POFFIN, BOSS, POKE_PAD, NIGHT_STRETCHER):
            if ctx.deck_counts.get(cid, 0) > 0:
                return True
        return False


_AGENT = fw.RuleAgent(GrimmsnarlStrategy())


def agent(obs) -> list[int]:
    """`league/run_match.py` と Kaggle 基盤の両方から呼べるエージェント関数。"""
    return _AGENT(obs)
