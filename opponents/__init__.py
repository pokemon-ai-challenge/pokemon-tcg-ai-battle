"""対戦相手(sparring partner)エージェント置き場。

こちらの提出物(sample_submission/)とは別に、league から対戦相手として使う外部/
ベースラインのルールベースエージェントを収める。各エージェントは league の契約
``agent(obs: Observation) -> list[int]`` を満たし、``league/run_league.py`` の
``AGENT_REGISTRY`` に名前を登録して ``--agent-a/--agent-b`` から選ぶ。
"""
