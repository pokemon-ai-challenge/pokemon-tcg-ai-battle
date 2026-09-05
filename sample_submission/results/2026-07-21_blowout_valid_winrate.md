# 検証(blowout piloting) Step1: 第三者Alakazamパイロットの全体勝率

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [blowout-piloting-validation-implementation-plan.md](../plans/individual/shogo/blowout-piloting-validation-implementation-plan.md) Step1
- コード変更: なし。分析専用スクリプト `kaggle_replays/_diag_blowout_valid.py`
  (読み取り専用、既存リプレイ4763件のみ参照。追加ダウンロードなし)。

## 方法

自チーム(`MORIOKA Tsuoi`)以外の各プレイヤー側について、[[project_deckout_loss_cause|山札切れ検証]]
(`_diag_deckout_topplayer_rate.py`)と同一の判定(そのプレイヤー視点の観測に card_id 743
Alakazam + 742 Kadabra の両方が一度でも現れたら「同アーキタイプ」)でAlakazamパイロットを
特定し、勝敗が確定した試合(win/loss、draw・unknownは除外)の勝率を集計した。

## 結果

| 区分 | 試合数(決着) | 勝ち | 勝率 | 95% CI(Wilson) |
|---|---|---|---|---|
| 自チーム | 363 | 162 | **44.6%** | [39.6%, 49.8%] |
| 第三者Alakazamパイロット(695チーム) | 2515 | 1358 | **54.0%** | [52.0%, 55.9%] |

**両者の95%信頼区間は重ならず、統計的に有意な差(二標本比率検定 z=-3.34)。**
第三者は自チームより約9.4ポイント高い勝率で同アーキタイプを回している。

試合数上位20チームの内訳(133試合〜15試合、勝率52.2%〜81.2%)は`_diag_blowout_valid_run.log`
参照。特定の1チームに偏った結果ではなく、多数のチームにわたって自チームより高い勝率が
観測される。

## 解釈と注意

- **一次シグナル(補助指標)としては「piloting余地あり」を支持する。** ただし計画が
  明記する通り、第三者はマッチメイクが自チームと異なる(対戦相手の分布が違う)ため、
  この勝率差だけでは「同じ相手に対して回し方が上手いから勝てている」とまでは
  断定できない交絡が残る。
- 決定打は[Step2](./2026-07-21_blowout_valid_lossmix.md)の「負けのうちブローアウトが
  占める率」(マッチメイク差に対して頑健な指標)に置く。
- [Step3](./2026-07-21_blowout_valid_topsubset.md)で、この勝率差が特定の少数チーム
  (試合数上位)に牽引されていないか、また真に勝率の高い部分集団でどうなるかを補正する。

## 限界(既知・継承)

- 母集団はリーダーボード順位で厳密にフィルタしたものではない([[project_deckout_loss_cause]]
  の検証時と同様の限界)。
- 同一チームが複数試合を含む(最大133試合)ためクラスタ効果あり → Step3で対応。
