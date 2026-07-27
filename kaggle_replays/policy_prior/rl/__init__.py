"""強化学習(自己対戦による方策勾配)の基盤。

設計書: ``test_plan/ptcg_rl_design.md``。

- ``policy.py``: 確率的方策(softmax + 温度)と対数尤度勾配
- ``shaping.py``: ポテンシャルベースの報酬シェーピング
- ``selfplay.py``: 自己対戦ログの収集(プロセス並列)
- ``reinforce.py``: REINFORCE + バッチ平均ベースラインの勾配計算
- ``generation.py``: 世代ループ(収集 -> 更新 -> 凍結プールで評価 -> 採否判定)

特徴抽出は ``ptcg_ai.learning.policy_features`` を import するだけで、
このパッケージ内では一切再実装しない。
"""
