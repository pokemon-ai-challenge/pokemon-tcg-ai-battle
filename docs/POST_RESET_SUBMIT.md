# リセット後に撃つ: v0only + 新デッキ(手札干渉型)

準備済みtarball: `build_ready/submission_v0only_newdeck.tar.gz`
中身 = 現行cg/ptcg_ai/configs/decks + main.py + deck.csv(=experimental_decks/alakazam_xerosic の新デッキ)、
かつ ml_policy_agent の既定config を `ml_lethal_attackplan_v0only`(v0only, パイプラインOFF)に差し替え済み。
検証: クリーン展開で agent _CONFIG_NAME=ml_lethal_attackplan_v0only を確認、フルゲーム完走・エラー0。

## submit コマンド(リセットで枠が回復したら)
```
cd /c/dev/pokemon-tcg-ai-battle/build_ready
PYTHONIOENCODING=utf-8 kaggle competitions submit pokemon-tcg-ai-battle \
  -f submission_v0only_newdeck.tar.gz \
  -m "リーサル+attack_plan v0only(土台=模倣構成C) フーディン手札干渉型デッキに変更(クセロシキ3枚・夜の鉱山) ※実績最高v0only×改善デッキ"
```
(cp932回避に PYTHONIOENCODING=utf-8 必須。認証は ~/.kaggle/access_token で通る。)

## 根拠
- 実フィールド: v0only(旧デッキ)717.2 > full(旧デッキ)702.8。
- 新デッキは旧デッキに自己対戦 0.710[0.644,0.768] で有意勝ち(experimental_decks/alakazam_xerosic)。
- => v0only × 新デッキ が理論上いちばん強い未提出の組合せ。

## もし作り直すなら(tarballを失った場合)
STAGE作業: sample_submission/{main.py,cg,configs,ptcg_ai,decks} をコピー → deck.csv を
experimental_decks/alakazam_xerosic/deck.csv に差し替え → staged ptcg_ai/ml_policy/ml_policy_agent.py の
既定を v0only に sed → tar -czf ... main.py deck.csv cg configs ptcg_ai decks(__pycache__除外)。
