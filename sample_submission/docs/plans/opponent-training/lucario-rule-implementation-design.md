> **【中止・2026-07-31】この設計は不採用・実装コード削除済み。** 自作ルールベースは3回試みて全て弱く（対模倣4.4% / 対rule_based 21.7→30%）、かつ模倣lucarioが既に有能な相手（対rule_based 75%）のため rule-base 自体が不要と判断。詳細と学びは memory `project_lucario_rule_opponent` 参照。resource として `opponents/lucario_ex_deck_official.csv`（公式デッキ）のみ残置。以下は記録として保存（特に §Phase 0 のエンジン実測はメガルカリオ機構の再利用可）。

# メガルカリオex ルールベース対戦相手 — 実装設計（実装契約）

- 作成日: 2026-07-31
- 位置づけ: `opponents/` 配下の対戦相手役。grimmsnarl_rule / dragapult_rule と同枠。production には触れない。
- 狙い: **模倣が下手な高メタ枠**（mega_lucario_ex: field≈0.39・share1257・専用相手なし）に dragapult_rule 型の強い専用相手を作る。grimmsnarl（模倣が強い枠）と違い、ここは**模倣を超えやすい**と期待。
- 実装は grimmsnarl_core.py / dragapult_rule_agent.py の骨格を流用（生int変換・出力選択・set_card_counts・get_card・no_draw・use_support）。

## アーキタイプ（OHKO/アグロ。ダメカン操作は無し＝grimmsnarl より単純）
勝ち筋: **メガルカリオex のメガブレイブ270 で毎ターン大型を1体ワンパン**。ハリテヤマの**どすこいキャッチャー（進化時gust）**で狙いを釣り、**ミツルの思いやり**で回復・エネ回収して使い回す。

### カードID
```
# ポケモン
RIOLU        = 677   # たねHP80 かそくづき闘●30(次番使用不可)
MEGA_LUCARIO = 678   # メガシンカex/1進化 HP340 弱点超 逃2
                     #   はどうづき[闘●]130 + トラッシュから基本闘エネ3枚までベンチに好きに付ける
                     #   メガブレイブ[闘闘]270（次の自分の番 使用不可＝反動）
MAKUNOSHITA  = 673   # たねHP80 → ハリテヤマ
HARIYAMA     = 674   # 1進化HP150 特性どすこいキャッチャー(進化時1回:相手ベンチ→バトル入替=gust内蔵)
                     #   ワイルドプレス[闘闘闘]210(自傷70)
LUNATONE     = 675   # たねHP110 特性ルナサイクル(ソルロック在+手札の基本闘エネ1枚トラッシュ→3ドロー,番1回)
SOLROCK      = 676   # たねHP110 コスモビーム[闘]70(ルナトーン在が条件)
OGERPON_EX   = 117   # たねHP210闘 テラスタル 特性いしずえのかまえ(相手の特性ポケからワザダメ受けない=壁)
                     #   ぶちやぶる[闘●●]140(弱点/抵抗/効果を計算しない)
# トレーナーズ
DARK_BALL    = 1102  # 山札下7枚からポケモン1枚サーチ
SWITCH_CART  = 1123  # ポケモンいれかえ
POWER_PROTEIN= 1141  # この番 闘ポケの相手バトルへのダメージ+30
FIGHT_GONG   = 1142  # 闘たね or 基本闘エネ サーチ
POKE_PAD     = 1152  # ルール無しポケモン サーチ
HERO_MANTLE  = 1159  # ACE SPEC 最大HP+100
BOSS_ORDERS  = 1182  # gust
ZEIYU        = 1192  # サポート(先攻1でも使える。ドロー系)
JUDGE        = 1213  # ジャッジマン 両者手札4枚に(手札干渉)
LILLIE       = 1227  # リーリエの決心 6ドロー(サイド6なら8)
MITSURU      = 1229  # ミツルの思いやり サポート: メガシンカexのHP全回復+エネを手札に戻す
GRAVITY_MT   = 1252  # スタジアム 2進化全員 最大HP-30
FIGHT_ENERGY = 6     # 基本闘エネ(deck01=13枚)
```
弱点: メガルカリオ/リオル=超, ハリテヤマ/マクノシタ=超, ルナトーン/ソルロック=草, オーガポン=草。攻撃側は相手が**闘弱点**なら×2。

## Phase 0 実測（`kaggle_replays/_phase0_lucario_probe.py`、8ゲーム・error0・確定）
> エンジンは context/type を**生int**で返す（要 `SelectContext(int)` 変換。grimmsnarl と同じ）。
- **はどうづき エネ加速（678）**: `ATTACH_TO`（type=CARD, **min0/max3**, effect=678, options=トラッシュの基本闘エネ）で最大3枚バッチ選択 → `ATTACH_FROM`（各 min1/max1, contextCard=闘エネ, options=ベンチ等）で付け先を選ぶ。**grimmsnarl パンクアップと同型**（ATTACH_TO batch→ATTACH_FROM×N、付け先 energies は各回更新）。
- **どすこいキャッチャー（674, 進化時）**: `ACTIVATE`（YES/NO, contextCard=674）→ **YES** → `SWITCH`（type=CARD, options=**相手のベンチ**）で釣る相手を選ぶ。＝進化時gust。
- **ミツルの思いやり（1229）**: サポートを PLAY → `HEAL`（type=CARD, min1/max1, effect=1229, options=自分のメガシンカex）で回復対象を選ぶ（エネ手札戻しは自動）。
- **ルナサイクル（675）**: `MAIN→ABILITY`（ルナトーン）→ `DISCARD`（type=CARD, effect=675, options=手札の基本闘エネ）を1枚捨てる → 3ドロー（自動）。**捨てた闘エネはトラッシュに乗る＝はどうづき加速の弾にもなる**（相乗効果）。
- **いしずえのかまえ（117）**: 受動（decision無し）。
- **turn-1 ルール**: EVOLVE/ATTACK が出る最速は **turn 3**。turn1（先攻初手）=ATTACH/END/PLAY、turn2（後攻初手）=END/PLAY。**進化・攻撃は先後とも不可（対称）**。lethal は `pokemon.hp`（現在HP）前提。
- **メガブレイブ反動**: 使った次の番はエンジンが ATTACK option を出さない想定（grimmsnarl の使用済み特性と同じ）。→ **「使用済み」を自前追跡しない。エンジンが提示した ATTACK/ABILITY option だけを信じる**（grimmsnarl #1 バグの教訓）。

## 意思決定表（context 単位。grimmsnarl 骨格を流用、OHKO 向けに単純化）
| Context | 方針 |
|---|---|
| **IS_FIRST** | v1 は後攻（YES=-1）。eval で先後 A/B は follow-up。 |
| **SETUP_ACTIVE** | リオル or マクノシタ（進化で価値化）。オーガポンは壁として可。単騎で晒したくないメガは避ける。 |
| **SETUP_BENCH** | リオル/マクノシタ/ルナトーン/ソルロック/オーガポンを展開。メガライン(リオル)最優先、ハリテヤマ用マクノシタ、ドローのルナ/ソル。 |
| **EVOLVE** | **リオル→メガルカリオex**=最優先（メインアタッカー）。**マクノシタ→ハリテヤマ**=良い gust 対象がいる/KOを作れるとき高優先（どすこい発火）。過剰進化は晒しKO注意。 |
| **ACTIVATE(674)** | YES（どすこいは強力）。 |
| **SWITCH(effect=674 / ボスの指令)** | 釣る相手＝**メガブレイブ270で落とせる高価値** or **相手システム/育成中**。相手ポケ評価は dragapult `pokemon_score`（サイド枚数×1000＋エネ＋道具＋進化段階＋HP）流用。落とせないHPの巨大exを釣らない。 |
| **ATTACH_TO(effect=678)** | はどうづき加速: トラッシュの闘エネを **max3 全取り**（no_draw等の制約なければ）。 |
| **ATTACH_FROM(effect=678)** | 付け先: **次に攻撃するベンチのメガルカリオ/リオル**を優先（2枚目育成）、無ければ将来の攻撃役。energies を毎回読んで配分。 |
| **ATTACH(手貼り 闘エネ)** | activeメガルカリオを**闘闘（メガブレイブ分=2）**にするのを最優先。届いたら次のアタッカー、次にオーガポン(闘●●)。 |
| **HEAL(effect=1229 ミツル)** | 回復対象＝**ダメージを受けた active メガルカリオ**（KO回避/使い回し）。健全なら撃たない。 |
| **ABILITY(ルナサイクル675)** | ソルロック在＋手札に闘エネ＋no_drawでない なら発動（3ドロー）。→続く DISCARD で余剰闘エネを1枚（トラッシュ=加速の弾）。 |
| **ATTACK** | 優先: **メガブレイブ270**（相手を落とせて反動許容なら）。撃てない/反動ターンは **はどうづき130+加速**（盤面育成）or **ワイルドプレス210**(ハリテヤマ) or オーガポン ぶちやぶる140。相手が闘弱点なら×2。**パワープロテイン(+30)で丁度KO**に届くなら使う。**盤面未展開で手なり攻撃しない**ゲート（grimmsnarl 同）。 |
| **PLAY サポート(use_support)** | ボス=KO確定時、リーリエ/ゼイユ=手札細い、ジャッジマン=相手手札厚い/干渉、ミツル=回復要時。1/ターン。 |
| **PLAY グッズ** | ファイトゴング/ダークボール/ポケパッド=不足パーツサーチ、いれかえ=逃げ代替、パワープロテイン=打点調整、ヒーローマント=メガ/主力に装着(+100)、グラビティーマウンテン=相手2進化を削るとき。 |
| **DISCARD/DISCARD_ENERGY/TO_HAND/番号** | dragapult/grimmsnarl 流用（hand_score 降順、番号は最大、逃げエネは最少影響）。 |
| **免疫/no_damage** | 最小から。オーガポンの「いしずえのかまえ」は相手側なので攻撃計算では無関係（我々が受ける側の話）。 |

## 山札切れガード / 合法手フィルタ
- `no_draw = deckCount<=8` を リーリエ/ゼイユ/ジャッジマン/ダークボール/ファイトゴング/ポケパッド/ルナサイクル にも適用（grimmsnarl §D-3 と同思想）。
- 出力選択・min/maxCount 遵守・重複禁止・未知値は無害フォールバックで必ず有効 index（grimmsnarl §F 流用）。
- グローバル状態は state.turn==0 で全リセット。deck/profile はラッパ側 module 定数。

## ファイル構成（grimmsnarl と同型）
```
opponents/
  lucario_core.py            # agent(obs, *, deck, profile)
  lucario_rule_01.py .. _05.py
  lucario_ex_deck_01.csv .. _05.csv  # archetype_decks/mega_lucario_ex/ からコピー
```
league/run_league.py の AGENT_REGISTRY / PLAIN_AGENTS に `lucario_rule_01..05` を5行、opponents/README.md 表に追記。

## build_profile（5デッキ差分。02–05 は実装時に decode して確定）
検出フラグ例: has_solrock_engine(675/676), ogerpon(117), power_protein数(1141), judge数(1213), zeiyu数, hero_mantle(1159), gravity_mountain(1252), fight_energy数。02–05 の枚数差で微調整（v1 は 01 基準でも可）。

## 実装ブリーフ（doer=Codex 想定）
- **本体の新規**は2つ:(1)はどうづき加速の ATTACH_TO/ATTACH_FROM 配分（grimmsnarl パンクアップとほぼ同じ）、(2)どすこい/ボスの SWITCH で釣る相手選択＋メガブレイブ lethal 判定（単純: 単体270、パワープロテインで+30、闘弱点×2）。ダメカン操作が無いので grimmsnarl の plan_attack より**大幅に単純**（複数KOビンパッキング不要、単体OHKO中心）。
- grimmsnarl_core.py / dragapult_rule_agent.py を土台に、カード定数と上記表を差し替え。
- 検証: pytest（grimmsnarl の test を雛形に build_profile/合法手不変条件/デッキ妥当性）→ 本体で実行。スモーク lucario_rule_01 vs rule_based / vs ml_policy(mega_lucario重み) error0。
- 完了条件: evaluation-scientist で **対模倣 CI下限>50%**（今回は模倣が弱いので到達期待）。

---

# REWORK v2: 公式デッキ版（2026-07-31）— 初版が対模倣4.4%/対rule_based21.7%で大敗したための作り直し

## なぜ作り直すか（診断）
初版(Kaggleデッキ+Codex実装)は **対 rule_based 21.7%（模倣は75%）・平均8.8ターンで決着＝立ち上がる前に轢かれる**＝根本的に弱い。コード診断で判明した主因:
1. **攻撃選択が雑（最重要）**: `OptionType.ATTACK` を全 option 一律 `90000/20000` にしていた（lucario_core.py:431）。→ **メガブレイブ270 を はどうづき130 より優先せず、選んだ技がKOに届くかも未検証**。
2. **公式デッキ未対応10枚**: 初版はKaggleデッキ前提でミツル回復軸。公式デッキは**ミツル不採用・ハイパーボール×4軸**で、未対応カードを捌けない。

## 公式デッキ（`opponents/lucario_ex_deck_official.csv`、master branch、60枚）
コア: リオル677×4 / メガルカリオex678×3 / マクノシタ673×2 / ハリテヤマ674×2 / ルナトーン675×2 / ソルロック676×2 / 基本闘エネ6×10。
トレーナーズ: リーリエ1227×4 / ファイトゴング1142×4 / パワープロテイン1141×4 / **ハイパーボール1121×4** / ジャッジマン1213×3 / ポケパッド1152×3 / ボス1182×2 / ゼイユ1192×2 / **ロック闘エネ20×2** / ポケモンいれかえ1123×1 / **ニャースex1071×1** / **マキシマムベルト1158×1** / ふうせん1174×1 / **アオキの手際1206×1** / **ロケット団の監視塔1256×1** / グラビティーマウンテン1252×1。
- **落ちた**: ミツル1229・オーガポン117・ヒーローマント1159・ダークボール1102（初版のこれら向けロジックは公式デッキでは死ぬ→build_profile で無効化）。

### 新カードID + 効果（実測）
```
HYPER_BALL   = 1121  # グッズ: 手札2枚トラッシュ→山札から好きなポケモン1枚サーチ（このデッキの手札/サーチエンジン）
ROCK_ENERGY  = 20    # 特殊エネ: 【闘】1個ぶんとして働く（メガブレイブ闘闘の支払いに使える。闘エネと同等に扱う）
NIASU_EX     = 1071  # たねHP170無 特性おくのてキャッチ=ベンチに出したとき1回:山札からサポート1枚を手札に / しっぽをまく●●●60(自分を手札に戻す)
MAX_BELT     = 1158  # ACE SPEC どうぐ: 装着ポケのワザの「相手バトル場のex」への+50（メガブレイブ270→320 vs ex）
BALLOON      = 1174  # どうぐ: 逃げ-2
AOKI         = 1206  # サポート: 手札全トラッシュ→山札からポケモン/サポート/基本エネを1枚ずつ（手札事故のリフレッシュ）
ROCKET_TOWER = 1256  # スタジアム: 両者の【無】ポケモンの特性を消す（相手の無系システムを妨害。自ニャースexの特性も消えるので出す順に注意）
```

## 最重要修正: 攻撃選択（per-attack lethality スコア）
ATTACK option を**技ごとに実ダメージとKO可否で評価**する。`attack_plan` は「どの技で・誰を落とせるか」を技別に持つ。
```
# 技別の基礎ダメージ（active 相手への)
MEGA_BRAVE(679系attackId) = 270   # 反動: 次番使用不可(engineがoption出さない→自前追跡しない)
HADOU(はどうづき)          = 130   # + トラッシュ闘エネ3枚をベンチ加速
WILD_PRESS(ハリテヤマ)     = 210   # 自傷70
NIASU しっぽ/サブ          = 小
# 補正
+30  if パワープロテイン(1141) を今ターン使える(闘ポケの相手バトルへ+30)
+50  if マキシマムベルト(1158) 装着 かつ 相手activeがex
×2   if 相手activeが闘弱点
# スコア
score(ATTACK opt) =
  95000  if その技の総ダメージ >= 相手active実HP（＝そのターンKO） and 高価値
  40000  if メガブレイブ以外で「盤面育成として撃つ価値」(はどうづき=加速目的, 相手を削れる)
  -1     if 盤面未成熟で手なり(=grimmsnarl §F 攻撃成熟ゲート)。ただしKO確定なら上記95000優先
```
**KO判定は必ずその技の実ダメージで**（初版の「全技同点」を廃止）。パワープロテイン/マキシマムベルト/闘弱点を必ず織り込む。ボス/どすこいで釣った相手にメガブレイブが届くかも同じ関数で判定。

## context 別スコア（初版の意思決定表を継承しつつ公式デッキ対応）
| Context | v2 方針（concrete） |
|---|---|
| EVOLVE | リオル→メガルカリオ=80000+付随エネ。マクノシタ→ハリテヤマ=55000（良gust対象がいる時+10000）。過剰進化は晒しKO注意で抑制。 |
| ATTACH(手貼り 闘/ロック闘エネ) | activeメガルカリオを**闘闘(=メガブレイブ2)**に最優先(40000, 2枚到達後は-1)。ロック闘エネ20も闘エネと同等に扱う(僅かに攻撃役へ)。次にベンチのメガ/リオル。 |
| ATTACH_TO/ATTACH_FROM(はどうづき加速) | ATTACH_TO=トラッシュ闘エネ max3 全取り。ATTACH_FROM=次に殴るベンチのメガ/リオル優先(2枚目育成)。 |
| ACTIVATE(674)→SWITCH | どすこいYES→釣る相手=**その場のメガブレイブ(補正込)で落とせる高価値** or 相手システム(pokemon_score)。落とせない巨大exは釣らない。ボスの指令も同じ選択。 |
| ABILITY | ハリテヤマどすこい=70000。ルナサイクル(675)=35000(no_drawで-1)。**ニャースex おくのてキャッチ**=ベンチ出し時に必要サポを取れるなら高(45000)。 |
| PLAY グッズ | **ハイパーボール(1121)=45000**（山札に必要ピースがあり、手札に捨てられる札≥2、no_drawでない時。捨てるのは hand_score 最低2枚）。ファイトゴング/ポケパッド=不足パーツ。パワープロテイン=KO丁度時。いれかえ/ふうせん=逃げ。 |
| PLAY サポート(use_support) | ボス=KO確定時65000。リーリエ/ゼイユ=手札細い。ジャッジマン=相手手札厚い/干渉。**アオキ(1206)=手札事故(打てる札が乏しい)時のリフレッシュ**。 |
| PLAY どうぐ | **マキシマムベルト(1158)=activeメガルカリオに装着優先(相手exに+50)**。ふうせん=逃げたい個体。 |
| スタジアム | グラビティ(1252)=相手2進化を削る時。**監視塔(1256)=相手が無系特性ポケモンを使う時（自ニャースexの特性を使い終えた後に出す）**。 |
| HEAL(ミツル) | **公式デッキはミツル不採用**→profileで無効(該当option来ない)。 |
| その他 | DISCARD/DISCARD_ENERGY/TO_HAND/番号/IS_FIRST(後攻)/no_draw ガード は初版・grimmsnarl 流用。 |

## build_profile（公式デッキ検出）
```
has_mitsuru      = count(1229)>0   # 官=False（回復軸ロジックを切る）
has_hyperball    = count(1121)>0   # 官=True（手札エンジン）
has_niasu        = count(1071)>0   # 官=True
has_maxbelt      = count(1158)>0   # 官=True（exに+50を lethal に織り込む）
has_rock_energy  = count(20)>0     # 官=True（闘エネ同等に扱う）
has_aoki         = count(1206)>0
has_two_stadiums = count(1252)+count(1256)>=2
has_ogerpon      = count(117)>0    # 官=False（Kaggleデッキ用ロジックはこれで分岐）
```
共通コア `lucario_core.py` が Kaggleデッキ(01-05)と公式デッキ両方を profile で捌く。

## ファイル/検証
- `lucario_core.py` を上記で**改修**（攻撃選択の per-attack 化＋新カード処理＋build_profile 拡張）。
- 新ラッパ `opponents/lucario_rule_official.py` → `lucario_ex_deck_official.csv`、league に登録。
- 既存 `test_lucario_opponents.py` を維持＋公式デッキ用に build_profile/合法手不変条件を1ケース追加。
- **教訓の再確認（初版失敗の芯）**: (1)攻撃は技別に実ダメージでKO判定（全技同点にしない）、(2)使用済み状態は自前追跡せずエンジンの option を信じる、(3)エネは確実に active メガルカリオを闘闘にする。
- 完了条件: **対 rule_based で模倣(75%)並みに勝てる**（サニティ）＋**対模倣(公式デッキ mirror)で改善**（初版4.4%からの大幅up、可能なら CI下限>50%）。
