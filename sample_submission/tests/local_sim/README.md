# local_sim

`local_sim/` には、`cg` を使ってローカル対戦を手動実行するスクリプトを置きます。

- [test_local_game.py](C:/dev/pokemon-tcg-ai-battle/sample_submission/tests/local_sim/test_local_game.py)
  1 試合だけ回す最小確認用です。
- [test_local_game_advanced.py](C:/dev/pokemon-tcg-ai-battle/sample_submission/tests/local_sim/test_local_game_advanced.py)
  試合数や相手方針を切り替えて確認できます。

実行例:

```powershell
cd C:\dev\pokemon-tcg-ai-battle\sample_submission
python .\tests\local_sim\test_local_game.py
python .\tests\local_sim\test_local_game_advanced.py --games 10 --opponent random
```
