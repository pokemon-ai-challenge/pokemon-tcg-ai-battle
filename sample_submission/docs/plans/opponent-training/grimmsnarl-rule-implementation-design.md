# マリィのオーロンゲex ルールベース対戦相手 — 実装設計（context単位・実装契約）

- 作成日: 2026-07-30
- 位置づけ: [grimmsnarl-rule-opponent-plan.md](grimmsnarl-rule-opponent-plan.md)（戦略方針＋Phase 0 実測）の**実装契約**。rule-engine-expert が方針書＋dragapult 雛形＋Phase 0 実測(1055 decision)＋`all_attack()`/CSV 突合で確定。python-engineer はこの文書の粒度でそのまま実装する。
- 実装は `opponents/` 配下（対戦相手役）。production の `rule_based` には触れない。

## サマリ
- dragapult 骨格をほぼ流用可。**新規で必要なのは「アドレナブレイン込みの attack plan（`plan_attack`）」と「パンクアップ配分」の2箇所だけ**。他（`set_card_counts`/`get_card`/`hand_score`/`attach_score`/出力選択/no_draw/カウント系）はカード定数差し替えで流用。
- **Phase 0 §6.1 の実測修正（本文書が正）**: パンクアップは「悪エネ1枚ごとに ATTACH_TO→ATTACH_FROM」ではなく、**ATTACH_TO で悪エネをバッチ複数選択（min0/maxN, `select.deck` 提示, `effect`=648）→ その後 ATTACH_FROM が選んだ枚数ぶん連続（各 min1/max1, `contextCard`=悪エネ, options=マリィのポケモン）**。付け先の `energies` は各 ATTACH_FROM 間で更新されるので、`attach_score` を毎回読めば配分が自然に成立。

---

## 定数テーブル

### カードID（全5デッキの実カード）
```
# ポケモン
DARK_ENERGY      = 7     # 基本悪エネ（唯一のエネ。手貼り/パンクアップの弾）
IMPIDIMP         = 646   # マリィのベロバー(たね,HP70,弱点草) くすねる(934,draw)/どつく(935,悪,10)
MORGREM          = 647   # マリィのギモー(1進化,HP100,弱点草) どつく(936,悪悪,60)
GRIMMSNARL_EX    = 648   # マリィのオーロンゲex(2進化,HP320,弱点草) パンクアップ(特性)/シャドーバレット(937,悪悪,180+ベンチ30)
MORPEKO          = 649   # マリィのモルペコ(たね,HP70,弱点草) スパイクホイール(938,無無無,20+悪×40)
MUNKIDORI        = 112   # マシマシラ(たね,HP110,弱点【悪】) アドレナブレイン(特性)/技141は超●=撃てない
FROSLASS         = 104   # ユキメノコ(1進化,HP90,弱点鋼) いてつくとばり(特性,受動)/技131は水●=撃てない
SNORUNT          = 860   # ユキワラシ(たね,HP70,弱点鋼) → ユキメノコ進化元
BUDEW            = 235   # スボミー(たね,HP30,弱点炎) むずむずかふん(323,0コスト,グッズロック)
DUNSPARCE        = 305   # ノコッチ(たね,HP70) いれかわる(423)/ぶつかる(424) ※02のみ
DUDUNSPARCE      = 66    # ノココッチ(1進化,HP140) にげあしドロー(特性)/ランドクラッシュ(76) ※02のみ
# トレーナーズ
RARE_CANDY       = 1079  # ふしぎなアメ
UNFAIR_STAMP     = 1080  # アンフェアスタンプ(ACE SPEC,自5相手2)
BUDDY_POFFIN     = 1086  # なかよしポフィン(HP70以下たね2枚をベンチ; マシラ110は対象外)
NIGHT_STRETCHER  = 1097  # 夜のタンカ(トラッシュ→手札 ポケモン/基本エネ)
ENERGY_TRANSFER  = 1119  # エネルギー転送(山札→手札 基本エネ) ※02
POKEGEAR         = 1122  # ポケギア3.0(上7見てサポ1) ※01/04/05
TOOL_SCRAPPER    = 1137  # ツールスクラッパー(どうぐ2枚トラッシュ) ※01
ENERGY_RECYCLE   = 1139  # エネルギーリサイクル(トラッシュ悪エネ5枚→山札) ※02
POKE_PAD         = 1152  # ポケパッド(ルール無しポケモン1枚サーチ; オーロンゲexは不可)
HERO_MANTLE      = 1159  # ヒーローマント(ACE SPEC,最大HP+100) ※02
HANDY_CIRC       = 1161  # ハンディサーキュレーター(被弾時,殴った相手のエネ1個を相手ベンチへ) ※03
BALLOON          = 1174  # ふうせん(逃げ-2) ※05
BOSS_ORDERS      = 1182  # ボスの指令(相手ベンチ→バトル)
XEROSIC          = 1197  # クセロシキのたくらみ(相手手札を3枚に) ※02/04
ROCKET_LAMBDA    = 1219  # ロケット団のラムダ(山札→手札 トレーナーズ)
LILLIE_DETERM    = 1227  # リーリエの決心(手札山札戻し6ドロー,サイド6なら8)
DAWN_HIKARI      = 1231  # ヒカリ(山札→手札 たね/1進化/2進化 各1)
SPIKEMUTH_GYM    = 1259  # スパイクタウンジム(スタジアム;番1回マリィのポケモン1枚サーチ)
```

### 攻撃ID（`all_attack()` 実測）
```
SHADOW_BULLET = 937   # ← can_main_attack 判定（dragapult の Phantom Dive 154 に相当）
IMPIDIMP_FILCH = 934  # くすねる(draw)
IMPIDIMP_PUNCH = 935  # どつく10
MORGREM_PUNCH  = 936  # どつく60
MORPEKO_WHEEL  = 938  # スパイクホイール
BUDEW_ITCHY    = 323  # むずむずかふん(グッズロック; dragapult の no_item と同一ID)
```

### 弱点（attack plan に必須）
- **相手が弱点=悪(DARKNESS) のとき、シャドーバレット 180 は ×2=360**。相手マシマシラ(112,悪弱点)がいれば悪で×2で狙撃できるのが唯一の頻出ケース。
- 自軍: オーロンゲ/ベロバー/ギモー/モルペコ=草弱点、マシマシラ=悪弱点、ユキメノコ/ユキワラシ=鋼弱点（守備・被弾判断用）。

### 生int変換（Phase 0 必須注意・最大の初期バグ源）
`select.context` / `select.type` / `option.type` / `option.area` / `option.inPlayArea` は**生int**で来る。`SelectContext(int)`, `OptionType(int)`, `AreaType(int)` で包んで比較すること。dragapult は enum 比較なので**そのままだと int vs enum で全 else 落ち→全 context がデフォルト0点→不正手**。**受け取り直後に int→enum 変換を1箇所**入れる。

---

## A. 共通コア 意思決定表（context 単位）

`agent()` 骨格は dragapult と同一（初回は60枚 deck 返却／`state.turn==0` で global リセット／`select.deck!=None` で prize 推定／各 option をスコアリングして降順・negative skip）。「dragapult 流用」= 該当ブロックを定数差し替えでほぼそのまま。

| SelectContext | 何をする | スコア設計（大小関係） | 流用/新規 |
|---|---|---|---|
| **IS_FIRST**(41) | v1は後攻 | YES=-1（=後攻）。将来 eval で A/B | dragapult 流用 |
| **SETUP_ACTIVE**(1) | バトル場の基本 | スボミー(05,going-second時)=+120000 / ベロバー=+100000 / ユキワラシ=+30000 / モルペコ=+20000 / マシマシラ=**-5000**（悪弱点＋engine役割はベンチ）/ ノコッチ(02)=+40000。+energy×1000+hp | dragapult SETUP系流用 |
| **SETUP_BENCH**(2) | ベンチ展開 | ベロバー=+100000 / マシマシラ=+55000（engine最重要,ポフィン不可）/ ユキワラシ(メノコ型)=+40000 / モルペコ=+10000 / スボミー=**-1** / ノコッチ(02)=+45000。**dragapult の「先攻はDreepyだけ」制限は採用しない**（オーロンゲは展開速度が命） | dragapult 流用＋制限緩和 |
| **SWITCH/TO_ACTIVE**(3/4) | 前に出す | オーロンゲ(エネ≥2)=+50000 / オーロンゲ(エネ<2)=+15000 / ギモー(エネ≥2)=+20000 / ベロバー=+10000 / スボミー=**-8000** / マシマシラ=-10000。+energy×1000+hp。相手を出す系は plan.primary と一致するベンチに+100000 | dragapult 流用 |
| **EVOLVE**(37) | 進化 | §D。オーロンゲ化=最高優先（パンクアップ発火） | 一部流用＋新規ガード |
| **MAIN**(0) | 手番の親 | 各 OptionType を個別スコア。先頭で `plan_attack()` と `use_support` を計算 | dragapult 流用（本体） |
| **ACTIVATE**(43) | 特性YES/NO（=パンクアップ) | YES=+1（常にYES） | dragapult 流用 |
| **ATTACH_TO**(22) | パンクアップの弾（山札悪エネ）をバッチ選択 | 提示は山札の悪エネ。**maxCount ぶん全部選ぶ**（=min(5,残)全取り）。no_draw かつ active既に悪≥2 のときだけ活性化に必要な枚数に絞る | 新規（小） |
| **ATTACH_FROM**(21) | パンクアップ各弾の付け先 | `attach_score(DARK_ENERGY,pokemon,active,is_pankup=True)` を各回計算。activeオーロンゲ<2→最優先→ベンチのオーロンゲ/ギモー→モルペコ | 新規（attach_score拡張） |
| **ABILITY**(10) as MAIN | アドレナブレイン発動 | plan にアドレナ手が入っていれば +95000（攻撃より先に解決）。plan外の延命でも+40000。no_draw で下げない（山札を減らさない）。スタジアムABILITYはここに来ない（TO_HAND） | 新規（判定）＋dragapult枠 |
| **REMOVE_DAMAGE_COUNTER**(16) | アドレナの「元」（自分） | plan.adrena_source 一致=+100000。fallback `heal_value`（オーロンゲactive=+50000, KO危険域マシラ=+30000, 他=現ダメカン×10）。ダメカン0は選ばない | 新規 |
| **REMOVE_DAMAGE_COUNTER_COUNT**(40) | 剥がす個数1/2/3 | plan 要求個数。fallback=最大(NUMBER最大) | 新規（NUMBER最大＝dragapult流用） |
| **DAMAGE_COUNTER**(13) | アドレナの「先」（相手） | plan.adrena_target 一致=+100000。fallback: dragapult DAMAGE_COUNTER 式流用（`100000 -10*hp + pokemon_score`＋KO圏ボーナス）。**counterはattackダメでないのでTera benchも乗る** | dragapult 流用＋新規 |
| **DAMAGE**(15) | シャドーバレットのベンチ30 | plan.bench30_target=+100000。fallback: +30でKOのベンチ最優先→高価値system。**Tera bench=対象不可(attackダメ→skip)** | 新規 |
| **TO_HAND**(7) | サーチ/山札→手札 | `effect.id` で分岐（下記）。基本 hand_score 降順。スパイクタウンジム(1259,min0)は no_draw なら0枚見送り | 新規（effect分岐）＋dragapult hand_score |
| **TO_BENCH**(5) | ベンチに出す | hand_score 降順（負なら skip、min許す限り） | dragapult 流用 |
| **DISCARD**(8) | トラッシュ | `-hand_score`（低価値から） | dragapult 流用 |
| **DISCARD_ENERGY**(30) | 逃げ/効果のエネ切り | 自分は最少影響。相手のエネを切れる系(ハンディ)は相手ベンチ>active | dragapult 流用 |
| **DRAW_COUNT**(38) | 引く枚数 | NUMBER 最大 | dragapult 流用 |
| **SKILL_ORDER**(34) | ポケチェック効果順 | 効果同一で順不同→先頭から minCount 個 | 新規（自明） |
| **RETREAT**(12) | 逃げ | `do_switch` 真=+10000 / 偽=-1 | dragapult 流用 |
| **ATTACK**(35) as MAIN | 攻撃 | §B/§Fの gate。シャドーバレット: plan.prizes≥1→+90000 / prizes0→+1(成熟)or-1(未成熟) / サブ技: くすねる=3, どつく=2 | 新規（gate） |

**TO_HAND の effect 分岐**（`obs.select.effect.id`）:
- `1259`(スパイクタウンジム, min0/max1): 進化ライン不足分（オーロンゲ>ギモー>ベロバー、メノコ型はユキワラシ/ユキメノコ）を hand_score で1枚。no_draw なら **0枚見送り**。
- `1152`(ポケパッド): ルール無しポケモン。**マシマシラ最優先**（ポフィン不可＝主要確保手段）→ユキワラシ/ギモー/ベロバー。
- `1231`(ヒカリ): 来た option を hand_score 降順で minCount 満たす。
- `1219`(ラムダ): plan.prizes≥1 かつ手札にボス無→ボス、アメ不足→アメ、スタジアム未張→スパイクタウンジム、細い→リーリエ。
- `1097`(夜のタンカ): トラッシュから オーロンゲ系 or 枯れてる側の悪エネ。
- `1119`(エネ転送,02)/`1122`(ポケギア): 悪エネ / 最優先サポート。
- 既定: hand_score 降順。

---

## B. attack plan（`plan_attack` = main_option_proc 相当）

**目的**: 「シャドーバレット(180/360＋ベンチ30) ＋ アドレナブレイン(複数マシラ,最大9カウンター=90dmg,のせ替え先可変) ＋ ボスの指令(相手ベンチ→前)」を一括で解き、**このターンに取れるサイド数を最大化**。dragapult の `counter_indices`（部分集合列挙）を、アドレナ配分とのせ替え先を変数追加して拡張。

### B-1. 入力（毎 MAIN 冒頭で計算）
```
targets = [opp_active] + opp_bench          # index0 = 相手バトル
weak_mult(t) = 2 if card_table[t.id].weakness == DARKNESS else 1
can_main = (active.id==648) and (ATTACK option with attackId==937 exists)
primary_dmg_base = 180

masila_usable = [m for m in own_field
                 if m.id==112 and has_dark_energy(m) and ability_available(m)]  # ABILITY option の有無で判定
adrena_bins = len(masila_usable)                       # 各bin容量3カウンター
own_counters = sum((p.maxHp - p.hp)//10 for p in own_field)
adrena_total = min(adrena_bins*3, own_counters)        # 移せる総カウンター(×10=dmg)
bench30_available = can_main
boss_available = (BOSS_ORDERS in hand) and (not supporterPlayed) and profile.boss>0
```

### B-2. 探索（相手≤6・マシラ≤~3 で全列挙可）
```
best = None   # (prizes, tiebreak_score, plan)
primary_candidates = [0] + ([j for j in bench_idx] if boss_available else [])  # ボスで前に引く候補
for p in primary_candidates:
    needs_boss = (p != 0)
    rem = [t.hp for t in targets]
    prize_taken = 0; killed = set()
    dmg = primary_dmg_base * weak_mult(targets[p])
    if not no_damage_dex(targets[p].id):
        rem[p] -= dmg
        if rem[p] <= 0: prize_taken += prize_count(targets[p], True); killed.add(p)
    best_add = enumerate_finish(rem, killed, bench30_available, adrena_bins, adrena_total, needs_boss)
    total_prize = prize_taken + best_add.prize
    score = tiebreak(total_prize, killed|best_add.set, needs_boss)
    if best is None or (total_prize, score) > (best.prizes, best.score):
        best = Plan(prizes=total_prize, score=score, primary=p, needs_boss=needs_boss,
                    bench30_target=best_add.bench30, adrena_moves=best_add.moves, ...)
```
**`enumerate_finish`（軽量ビンパッキング）**:
```
n<=6。部分集合 S ⊆ (未KOの相手) を全列挙(<=2^6):
  各 t∈S の need_t = ceil(rem_t/10) カウンター
  - bench30 を「S内のベンチかつ need>0 の1匹」に割当(need_t -= 3, >=0)。どのベンチかも全通り(<=6)。
  - 残り need を容量3ビン(adrena_bins個)にパッキング可能か & Σ使用<=adrena_total。
    (need<=3=1ビン, 4..6=2ビン。need降順グリーディで可)
  可能なら prize=Σ prize_count(t)。最大 prize の S。同点は Σ pokemon_score(t)＋system除去で tiebreak。
```

### B-3. tiebreak（dragapult 準拠＋オーロンゲ流）
- サイド取れないなら `pokemon_score`（サイド×1000＋エネ×150＋どうぐ×100＋進化段階＋HP、マシマシラ+300、systemは減点）最大化。
- `remain_prize` が少ないとき prize1 偏重を補正（prize2以上のKO集中を+、prize1は-300 等、dragapult 同）。
- **ボスは「KO＋相手system除去」を両立時のみ加点**。needs_boss に小ペナルティ(-500)を入れ、同サイドなら非ボス案を優先（無駄打ち防止）。

### B-4. 出力（global 保存、sub-context が消費）
```
plan.prizes / plan.score / plan.use_shadow / plan.primary_target /
plan.needs_boss(+どのベンチ) / plan.bench30_target /
plan.adrena_moves=[(masila,count<=3,opp_target)] / plan.adrena_sources=[(own_pokemon)]
```
**いてつくとばり**: v1は lethal 本体に入れない（現HPベース）。「相手が特性ポケモンを場に持ち、かつ自分にユキメノコが居る」ときのみ次ポケチェックの相手-10を **tiebreak にだけ** 反映（過剰読み回避）。

---

## C. エネルギー付け `attach_score(attach_id, pokemon, active, is_pankup) -> int`
悪エネ以外の弾は無い。dragapult 構造（ベース＋補正／十分なら-1 skip）流用。
```
e = len(pokemon.energies)   # ATTACH_FROM 間で更新される

# 手貼り(is_pankup=False): マシマシラ最優先
if not is_pankup:
    if pokemon.id == MUNKIDORI:   return 30000 if e == 0 else -1   # 各マシラ悪1枚で十分。2枚目不要
    if pokemon.id == GRIMMSNARL_EX:
        if active: return 22000 if e < 2 else -1                   # 足りない分だけ。撃てるなら手貼りしない
        else:      return 15000 if e < 2 else -1                   # ベンチ攻撃予約
    if pokemon.id in (IMPIDIMP, MORGREM): return -1                # 進化前はパンクアップで賄う
    if pokemon.id == MORPEKO:  return 8000 + e*500                 # 悪の枚数=打点(02/05終盤)。控えめ
    return -1

# パンクアップ(is_pankup=True): マシラは options に来ない
if pokemon.id == GRIMMSNARL_EX:
    if active: return 40000 - e*1000 if e < 3 else 100             # active優先で2枚厚く
    else:      return 25000 - e*800  if e < 2 else 100             # ベンチ次点
if pokemon.id == MORGREM:  return 12000 - e*800                    # 次ターンの攻撃予約
if pokemon.id == MORPEKO:  return 6000 + e*300
if pokemon.id == IMPIDIMP: return 3000
return 500
```
e を毎回読むことで「activeが2枚で自動的にベンチ優先」に（状態機械不要）。**TOOL 装着**は dragapult 同様 `cardType==TOOL → 60000(+active 1000)`。ヒーローマント/ハンディはオーロンゲ active を +2000 で優先、ふうせんは逃げたい個体。

---

## D. 進化/アメ・サポ/グッズ・山札切れ・守備

### D-1. EVOLVE / ふしぎなアメ
```
EVOLVE:
  ギモー→オーロンゲ / ベロバー→(アメ)→オーロンゲ: base 70000
     過剰進化ガード: field_grimmsnarl>=2 なら -1（相手サイド<=2 は1体でも -1）  # dragapult dex過剰進化相当
     + len(target.energies)（エネ乗り個体を優先進化=パンクアップ後即戦力）
  ベロバー→ギモー: 30000（手札にオーロンゲ/アメがあれば+10000）
  ユキワラシ→ユキメノコ(メノコ型): 60000（engine始動,高優先）/ 02は該当なし
PLAY RARE_CANDY(1079):
  ベロバー in play(not appearThisTurn) and オーロンゲ in hand and not グッズロック → 75000
  else -1
  ※05(アメ4): 「ベロバーが場に1体でも」でアメ切る(他型は2ライン確保後)
```
turn-1 進化不可（§6.2）は engine が EVOLVE option を出さない→特別処理不要。

### D-2. サポート/グッズ（`use_support` 方式 = dragapult 流用）
MAIN 冒頭で supporterPlayed でなければ hand_score 最大のサポを use_support 確定→PLAY 時そのIDだけ高得点。
```
ボスの指令(1182):     plan.prizes>=1 → 60000 / else 0
リーリエの決心(1227):  手札<=4 / サイド6(8ドロー) → 45000。no_draw中 -1
ヒカリ(1231):         main_line<3 → 序盤 50000（02は主軸で常時高）
ラムダ(1219):         必要トレーナーズ欠 → 40000
クセロシキ(1197,02/04): 相手手札厚い/コンボ阻害 → 25000
--- グッズ(サポ枠外) ---
なかよしポフィン(1086): 山札にHP70以下たね>=2 → 35000（序盤最優先級）。no_draw -1
ポケパッド(1152):      マシラ/欠けパーツ要 → 45000（マシラ確保の背骨）。no_draw -1
アンフェアスタンプ(1080): pre_ko → 80000（捲り札,dragapult同）/ 通常の攻めターン合わせ 15000
夜のタンカ/エネ転送/エネリサイクル/ポケギア: 悪エネ・サポ再供給 5000〜。山札掘る系は no_draw で -1
ツールスクラッパー(1137,01): 相手に厄介などうぐ有→4000、無→5
```

### D-3. 山札切れガード `no_draw`（一般化・決定事項3）
```
no_draw = (my_state.deckCount <= 8)
no_draw で score=-1 にする対象を dragapult(引きグッズ)から一般化:
  リーリエ/ヒカリ/ラムダ/ポケギア/ポフィン/ポケパッド/夜のタンカ/エネ転送
  + スパイクタウンジム TO_HAND は 0枚見送り
  + パンクアップ ATTACH_TO は「活性化に必要な枚数」に絞る(全取りしない)
※ アドレナブレイン(ABILITY)・くすねる は山札を減らさない→抑制しない
```

### D-4. 守備（草弱点・ヒーローマント）
- 既知の草アタッカーは attack plan の狙う先＋守備で優先。v1 は「相手 active が草タイプ技持ちなら、可能なら耐える個体を前に」を SWITCH スコアで軽く反映（脅威テーブルは replay 後拡張）。
- ヒーローマント(1159,02): TOOL 装着で最大HP+100。付け先はオーロンゲ active 優先。

---

## E. デッキプロファイル `build_profile(deck)`
```
profile = {
  has_snow:   count(104)+count(860) > 0,      # 01/03/04/05=True, 02=False
  has_nokocchi: count(66)+count(305) > 0,     # 02=True
  has_subomi: count(235) > 0,                 # 05=True
  has_morpeko: count(649) > 0,                # 02/05=True
  rare_candy: count(1079),                    # 05=4, 他=3
  boss:       count(1182),                    # 02=0, 他=2
  xerosic:    count(1197),                    # 02=2, 04=1
  lambda:     count(1219),                    # 02=0, 他=4
  hikari:     count(1231),                    # 02=4, 05=0, 他=1
  has_balloon: count(1174)>0,                 # 05
  has_hero_mantle: count(1159)>0,             # 02
  has_handy_circ: count(1161)>0,              # 03
  dark_energy: count(7),                      # 05=9, 他=10
}
```
**効き方**:
- **02（非メノコ純ビート）**: `has_snow=False` → いてつくとばり自動ダメカン無し＝アドレナ資源は被弾ダメカンのみ（own_counters>0 のときだけ寄与）。ボス無し→needs_boss を生成しない。ヒカリ4・にげあしドロー(66)・クセロシキで回す。**プランB(ビート)固定**。にげあしドローは dragapult ドロー特性同様 +40000／no_draw -1、**ただしエネ/どうぐ載せた個体では撃たない**（山札に戻る）。ヒーローマントをオーロンゲに。
- **01/03/04（標準）**: メノコマシラ・エンジン有効。attack plan フル。03はハンディをオーロンゲ active に。04はクセロシキも use_support 候補。
- **05（テンポ）**: `has_subomi`→後攻1でスボミー active にしてむずむず(323)、撃ったら SWITCH で下げる。`rare_candy=4`→進化閾値↓。`has_balloon`→逃げに活用。モルペコ終盤フィニッシュ。

---

## F. 合法手フィルタ / fallback
- **出力ロジック**: dragapult 流用（scores 降順、negative は skip 可能なら選ばない）。**skip 可能条件**=「score<0 かつ i>=minCount かつ 0枚可の系(TO_BENCH/SETUP_BENCH等)」。それ以外(min>0 必須)は負でも minCount ぶん必ず埋める。
- **min/maxCount 遵守**: `for i in range(maxCount)`、`len(output)<minCount` の間は負でも追加。重複は index 一意で自然回避。
- **未知 context/option**: 変換不能値は score=0（無害側）でフォールバックし必ず有効 index を返す（不正手0件が完了条件）。
- **plan と option の不整合ガード**: sub-context で plan の target が options に無ければ fallback スコアで選ぶ（plan を盲信しない）。
- **攻撃成熟ゲート（dragapult 既知欠陥の非継承・§8）**:
  - シャドーバレット: `plan.prizes>=1`→+90000。`plan.prizes==0`→ 成熟(activeオーロンゲ悪≥2 かつ 自マリィ ライン≥2) のとき +1、未成熟なら **-1(撃たずEND/展開)**。
  - サブ技: くすねる=3, どつく=2（**展開系スコアは万単位で必ずこれを上回る**）。→ 攻撃はそのターンの展開を全部終えた後にだけ発火。`plan.prizes>=1` のサブ技KOなら +90000 に格上げ。
  - アドレナ ABILITY(+95000) > シャドーバレット(+90000) → **アドレナ→攻撃の順**で解決。
- **強さバンド上限ガード（§7.4）**: v1 では入れず、eval で対 production >65% に振れたら「needs_boss 無効化・アドレナ日和見のみ・探索深度↓」で弱める調整代を残す（evaluation-scientist 判定後）。

---

## G. python-engineer 実装ブリーフ

### G-1. ファイル構成（方針書§7.1）
```
opponents/
  grimmsnarl_core.py           # 全ロジック。agent(obs,*,deck,profile)
  grimmsnarl_rule_01.py ... _05.py   # DECK読み込み＋build_profile→core.agent の薄いラッパ
  grimmsnarl_ex_deck_01.csv ... _05.csv   # archetype_decks/marnie_grimmsnarl_ex/ からコピー
```
- **deck/profile 引き回し**: dragapult の module-level `my_deck` global を、core は `agent(obs, *, deck, profile)` 引数で受ける。`set_card_counts` 等 global 参照は deck を引数/クロージャで渡す形に。
- **global 状態 turn0 リセット**: `plan/use_support/pre_turn_log/current_turn_log/prize/card_counts/serial_set` は module global 可だが `state.turn==0` で全リセット（5ラッパ交互実行に備える）。deck/profile はゲーム跨ぎ不変でラッパ側 module 定数。

### G-2. 内部関数一覧
| 関数 | 入出力 | 説明 | 流用 |
|---|---|---|---|
| `build_profile(deck)->dict` | deck→フラグ | §E | 新規 |
| `set_card_counts(obs,my_index,deck)` | →card_counts | 山札/サイド推定 | dragapult(deck引数化) |
| `get_card(obs,area,index,player)` | →Card/Pokemon | **area は生int→AreaType 変換必須** | dragapult |
| `prize_count(pokemon,is_atk)->int` | →サイド | ex/megaEx 判定 | dragapult(簡略化可) |
| `pokemon_score(pokemon,is_atk)->int` | →価値 | 狙う先評価。マシラ+300, system減点 | dragapult |
| `attach_score(attach_id,pokemon,active,is_pankup)->int` | →付け値 | §C。**is_pankup 新規** | dragapult拡張 |
| `hand_score(id,ignore_count,ctx...)->int` | →手札価値 | §A/D テーブル | 新規(テーブル差替) |
| `heal_value(pokemon)->int` | →剥がす元価値 | REMOVE_DAMAGE_COUNTER 用 | 新規 |
| `plan_attack(obs,profile)->Plan` | →Plan(§B-4) | **核。アドレナ＋シャドバ＋ボス一括** | 新規(main_option_proc土台) |
| `enumerate_finish(rem,killed,bench30,bins,total,needs_boss)` | →最良追加KO | §B-2 | 新規 |
| `no_damage_dex(id)/no_damage_counter(p)` | →bool | 免疫。**初期は空/最小** | dragapult(中身空) |

### G-3. dragapult 流用可否
- **ほぼそのまま**: `agent`骨格・出力選択・`set_card_counts`/`add_card_count`・`get_card`・`prize_count`・`pokemon_score`・`no_draw`/`do_switch`・`use_support`機構・NUMBER/YES(IS_FIRST)/DISCARD/DISCARD_ENERGY/RETREAT スコア。
- **定数差し替えのみ**: `hand_score` の card 分岐、EVOLVE/SETUP/SWITCH の対象ID。
- **新規本体2箇所**: (1) `plan_attack`（§B）、(2) パンクアップ ATTACH_TO/ATTACH_FROM 配分（§A/C）。＋ REMOVE_DAMAGE_COUNTER(_COUNT)/DAMAGE_COUNTER/DAMAGE の sub-context、TO_HAND effect 分岐、profile 分岐。

### G-4. 実装上の落とし穴（必読）
1. **生int→enum 変換**を agent 冒頭で必ず（最大の初期バグ源。怠ると全 context 0点→不正手）。dragapult は enum 比較なので**そのままでは動かない**。
2. **パンクアップは ATTACH_TO(バッチ)→ATTACH_FROM×N**（方針書§6.1の「1枚ごと」は誤り。実ログで確定）。ATTACH_TO は maxCount 全選択、ATTACH_FROM は各回 `attach_score(...,is_pankup=True)`。付け先 `energies` は各回更新。
3. **アドレナは MAIN の ABILITY→REMOVE_DAMAGE_COUNTER→_COUNT→DAMAGE_COUNTER の4連**。plan を MAIN で確定・global 保存し sub-context が順に消費。マシラ複数で連続（`plan.adrena_moves` を pop）。
4. **can_main_attack = active.id==648 かつ attackId==937 の ATTACK option 存在**。bench_attacker = ベンチのオーロンゲが悪≥2。
5. **弱点×2** を lethal に必ず入れる（相手マシラ=悪弱点で×2が主発火。dragapult 未計上なので意図的に足す）。
6. **Tera bench**: シャドーバレットのベンチ30(attackダメ)は benched Tera に無効(skip)。アドレナ(counter)は Tera bench にも乗る。`CardData.tera` で判定。
7. **02 は has_snow=False**＝いてつくとばり自動無し。アドレナ資源は own_counters(被弾)のみ。ボス無し＝needs_boss 作らない。
8. **turn-1 進化/攻撃**は engine が option を出さない（§6.2 確定）→特別処理不要。

---

## Risks（実装時に特に注意）
- `plan_attack` のビンパッキングを甘く書くと取りこぼす（＝弱い相手）。「複数マシラで6カウンター1体集中」「bench30＋アドレナ合わせ」の feasibility を落とさないこと。
- アドレナ資源 `own_counters` の数え損ね（maxHp と hp の差。ヒーローマントで maxHp 変動に注意）。
- no_draw とパンクアップ全取りの相互作用で自滅 deckout（§D-3 のガード必須。記憶 `project_deckout_loss_cause`）。
- 生int変換の抜け（全 context 崩壊）。スモークで context 別カバレッジを確認。
- プロファイル検出のオフバイワン（05のアメ4/悪9）。build_profile を5デッキで単体テスト。

## 未確定（後工程）
- 勝率・強さバンド実測（対模倣>50% / 対production≲65%）→ experiment-planner → evaluation-scientist。スコア定数は「大小関係の目安」で、最終チューニングは replay 後。
- 先攻/後攻 A/B 最終決定（§8決定1）→ evaluation-scientist。v1は後攻既定のみ。
- lethal が pokemon.hp=現在HP で正しいか、実装時に1点だけ実測確認推奨（同一エンジンなので dragapult 前提）。
- アドレナ回復先（オーロンゲtank vs 危険マシラ）優先度、02のプランB固定の是非 → 必要なら battle-strategist に最終相談。
