"""相手プールに対して勝率が 50% 近辺になる学習側モデルを探す。

狙い: 「序盤の局面が勝敗を予測するか」を測るデータセットを作るとき、勝率が極端
(例: 15%)だと勝敗がターン0の相性でほぼ決まってしまい、局面の寄与と相性識別の寄与が
分離できない。実際 dragapult_ex は温度0.05でもこのプールに 3/48 しか勝てなかった。

そこで候補モデルごとに勝率を実測し、50% 近辺のものを選ぶ。

注意: 学習側だけが softmax サンプリング(温度あり)、相手は argmax 固定
(collect_parallel._play_one の仕様)。したがって温度を上げるほど学習側が不利になる。
両方の温度で測る。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "kaggle_replays" / "rl"),
           str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_pool import parallel_collect_pool  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

# (表示名, 重みパス or None(=production alakazam), デッキのアーキタイプ名)
CANDIDATES = [
    ("alakazam(production)", None, "alakazam"),
    ("dragapult_ex", "policy_weights_dragapult_ex.json", "dragapult_ex"),
    ("dragapult_ex_rl_v3", "policy_weights_dragapult_ex_rl_v3.json", "dragapult_ex"),
    ("crustle", "policy_weights_crustle.json", "crustle"),
    ("marnie_grimmsnarl_ex", "policy_weights_marnie_grimmsnarl_ex.json", "marnie_grimmsnarl_ex"),
    ("archaludon_ex", "policy_weights_archaludon_ex.json", "archaludon_ex"),
    ("archaludon_ex_rl_v2", "policy_weights_archaludon_ex_rl_v2.json", "archaludon_ex"),
    ("shirona_garchomp_ex", "policy_weights_shirona_garchomp_ex.json", "shirona_garchomp_ex"),
    ("mega_lucario_ex", "policy_weights_mega_lucario_ex.json", "mega_lucario_ex"),
    ("rocket_mewtwo_ex", "policy_weights_rocket_mewtwo_ex.json", "rocket_mewtwo_ex"),
]

POOL = [
    ("alakazam", None, "alakazam"),
    ("crustle", "policy_weights_crustle.json", "crustle"),
    ("marnie_grimmsnarl_ex", "policy_weights_marnie_grimmsnarl_ex.json", "marnie_grimmsnarl_ex"),
    ("archaludon_ex", "policy_weights_archaludon_ex.json", "archaludon_ex"),
]

GAMES = 48
WORKERS = 8


def main() -> None:
    opponents = [(name, str(WDIR / w) if w else None,
                  read_deck_csv_file(str(DECKDIR / arch / "01.csv")))
                 for name, w, arch in POOL]

    print(f"相手プール: {[o[0] for o in opponents]}  各 {GAMES // len(POOL)} 試合 / 温度ごとに {GAMES} 試合")
    print(f"{'学習側':<24} {'温度0.05':>10} {'温度1.0':>10}")
    print("-" * 48)

    t0 = time.time()
    for name, w, arch in CANDIDATES:
        wpath = str(WDIR / w) if w else None
        if wpath is not None and not Path(wpath).exists():
            print(f"{name:<24} {'(重み無し)':>21}")
            continue
        deck_l = read_deck_csv_file(str(DECKDIR / arch / "01.csv"))
        out = []
        for temp, seed0 in ((0.05, 31000), (1.0, 61000)):
            _, stats = parallel_collect_pool(wpath, opponents, deck_l, GAMES, seed0,
                                             temperature=temp, workers=WORKERS)
            t = stats["total"]
            out.append(t["wins"] / t["valid"] if t["valid"] else float("nan"))
        print(f"{name:<24} {out[0]:>10.3f} {out[1]:>10.3f}")

    print(f"\n所要 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
