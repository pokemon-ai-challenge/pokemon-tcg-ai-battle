# 診断(piloting) Step1: デッキの山札切れ関連カード棚卸し

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [deckout-piloting-diagnosis-implementation-plan.md](../plans/individual/shogo/deckout-piloting-diagnosis-implementation-plan.md) Step1
- コード変更: なし。`cg.api.all_card_data()` / `all_attack()` を直接呼んでカードの
  正式なスキル(`Skill.text`)・攻撃(`Attack.text`)テキストを取得(`data/EN_Card_Data.csv`の
  `Effect Explanation` 列はStage1以降のポケモンの特性(Ability)テキストを含んでいなかった
  ため、より正確なこちらを一次情報として採用)。

## 分類結果(現行 deck.csv、フーディン/Alakazam、60枚・22種)

### (i) サーチ/ドロー(山札消費)

| 枚数 | カード | 効果 |
|---|---|---|
| 4 | Buddy-Buddy Poffin | 山札から70HP以下のBasicポケモンを2枚までベンチに |
| 4 | Poké Pad | 山札からルール枠なしポケモン1枚を手札に |
| 4 | Hilda | 山札から進化ポケモン1枚+エネルギー1枚を手札に |
| 3 | Dawn | 山札からBasic/Stage1/Stage2を1枚ずつ手札に |
| 2 | Telepath Psychic Energy | 装着時、山札から{P}Basicポケモンを2枚までベンチに |
| 4 | Kadabra(特性 Psychic Draw) | 手札から進化させた時、2枚ドロー |
| 4 | Alakazam(特性 Psychic Draw) | 手札から進化させた時、3枚ドロー |
| 3 | Dudunsparce(特性 Run Away Draw) | 1ターン1回、3枚ドロー。ドローしたら自身と付属カードを山札に**シャッフルして戻す** |
| 1 | Enriching Energy | 装着時、4枚ドロー |
| 1 | Fezandipiti ex(特性 Flip the Script) | 相手ターンにポケモンがきぜつしていた場合、1ターン1回3枚ドロー |

**合計26枚(43%)がサーチ/ドロー系。** ただし Dudunsparce は自身を山札に戻す副作用があり
(ドロー3・自身+付属カードを山札へ+1で正味 山札-2)、純粋なドローカードよりは
「山札を掘って特定カードを探す」性格が強い(高速で同じ内容を何度も引き直す構築の一部)。

### (ii) 回復/山札戻し/循環(sustain)

| 枚数 | カード | 効果 | 山札への影響 |
|---|---|---|---|
| **1** | **Sacred Ash** | 捨札からポケモンを最大5枚**山札に**シャッフルして戻す | **唯一、山札を直接増やすカード** |
| 2 | Night Stretcher | 捨札からポケモンかエネルギー1枚を**手札に** | 山札には戻らない(手札止まり。以後の再サーチ回数を減らす間接効果のみ) |
| 1 | Lana's Aid | 捨札からポケモン/エネルギーを最大3枚**手札に** | 同上 |
| 2 | Wondrous Patch | 捨札の基本{P}エネルギーをベンチ1体に装着 | 山札に影響なし(エネルギー加速のみ) |

**「山札を直接増やす」カードは Sacred Ash 1枚のみ。** Night Stretcher/Lana's Aid は
捨札→手札の回収であり、山札消費そのものを相殺しない(再サーチの必要性を減らす間接効果に
留まる)。**このデッキには山札切れを直接止める手段がほぼ存在しない**ため、「良い回し方」は
sustainカードを駆使することではなく、**山札切れが起こる前に試合を終わらせる速さ
(詰めの速さ)にほぼ依存する**と考えられる。

### (iii) アタッカー/wincon/その他

| 枚数 | カード | 役割 |
|---|---|---|
| 4/4/4 | Abra/Kadabra/Alakazam | メインアタッカーライン(Alakazamの「Powerful Hand」は手札枚数×2ダメージ。ドローエンジンで貯めた手札がそのまま打点になる設計) |
| 3/3 | Dunsparce/Dudunsparce | サブアタッカー(Land Crush 90ダメージ)兼ドローエンジン |
| 1 | Fezandipiti ex | サブアタッカー(Cruel Arrow、ベンチ狙い100) |
| 1 | Shaymin | ベンチ保護(特性 Flower Curtain) |
| **3** | **Boss's Orders** | **相手のベンチを場に呼び出す(詰めの主力ツール)** |
| 3 | Enhanced Hammer | 妨害(相手の特殊エネルギー破棄) |
| 3 | Battle Cage | スタジアム、ベンチ保護 |

## 「良い回し方」の判定基準(Step3向け)

上記を踏まえ、Step3で検証すべき「山札を維持しつつ勝ち切るための正しい振る舞い」は
2つの仮説に整理できる:

1. **sustain札の活用(仮説A)**: Sacred Ash / Night Stretcher / Lana's Aid が手札にある
   ターンにそれを使わず山を削り続けていないか。ただし合計4枚(60枚中)しかなく、
   そもそも手札に来る頻度自体が低いため、**この仮説だけでは25.9%という高い山札切れ率を
   説明しきれない可能性が高い**(棚卸しの時点での予想)。
2. **詰めの速さ(仮説B、本命)**: **Boss's Orders(3枚)を持っているのに使わず、
   相手の場のアタッカーと正面から打ち合い続けていないか。** サイド獲得ペースが遅い
   (1サイドあたりのターン数が長い)ほど、山札を掘り進める(ドロー・サーチ)ターン数が
   増え、山札切れに近づく。Alakazamの打点(手札枚数依存)を活かせず長期戦に
   ズルズル入っていないかも合わせて見る。

Step3では、山札切れ負け52試合について (a) sustain札(Sacred Ash/Night
Stretcher/Lana's Aid)を手札に持ちながら未使用だったターン数の頻度と、
(b) サイド獲得ペース・Boss's Orders使用率・毎ターンの攻撃有無を、勝ち試合・
非山札切れ負け試合と比較する。
