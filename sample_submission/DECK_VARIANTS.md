# デッキ変種の管理（いつでも入れ替え可能）

`deck.csv` が**実際に提出/使用されるデッキ**（`read_deck_csv()` が読む唯一のファイル）。
他の `deck_*.csv` は**保管された候補**。入れ替えは中身を `deck.csv` にコピーするだけ。

```bash
# 例: 過去デッキに戻す
cp sample_submission/deck_pre_planA.csv sample_submission/deck.csv
# 例: Plan A にする
cp sample_submission/deck_planA.csv sample_submission/deck.csv
```

（測定スクリプト `kaggle_replays/_measure_planA_matchups.py` は各候補を一時的に `deck.csv` へ
スワップして測り、必ず元へ復元する。）

## 変種一覧（2026-08-01 時点）

| ファイル | 内容 | 状態 |
|---|---|---|
| **`deck.csv`** | = **Plan A**（現在の live / 提出候補） | **使用中** |
| `deck_pre_planA.csv` | Plan A 以前の live デッキ（Alakazam「Powerful Hand」原型） | 保管（いつでも戻せる） |
| `deck_planA.csv` | Plan A（一貫性コア復元）＝ deck.csv と同一 | 保管 |
| `deck_planB.csv` | Plan B（速度全振り） | **棄却**（下記実測でAに劣後） |
| `experimental_decks/alakazam_xerosic/deck.csv` | 旧実験デッキ（クセロシキ型, [[project_alakazam_deck_v2]]） | 参考 |

## Plan A の中身（pre_planA からの差分, 60枚維持）

一貫性コアを上位標準へ戻し、名指し相手に効かない tech を削る:
- Kadabra(742) 3→4 ／ Telepath超エネ(19) 2→3 ／ Rare Candy(1079) 3→4
- Enhanced Hammer(1081) 3→1 ／ Wondrous Patch(1146) 2→1

## 採用根拠（実測, `_measure_planAB_matchups_results.json`）

現行 / Plan A / Plan B の対面別勝率（ローカル imitation 相手, config=abl_5_full）:

| 対面 | current | **Plan A** | Plan B |
|---|---|---|---|
| grimmsnarl (n=120) | 43.3% | **50.0%** | 44.2% |
| mirror (n=100) | 39.0% | **47.0%** | 43.0% |
| lucario (n=60) | 53.3% | **56.7%** | 48.3% |

Plan A が3対面すべてで最良（ΔA +6.7/+8.0/+3.3pt, 首尾一貫）。Plan B は削りすぎでAに劣り
lucario −5pt。個別は n では有意未満だが全対面プラスで一貫。詳細な方針は
`docs/plans/direction-and-rationale-2026-08.md`。
