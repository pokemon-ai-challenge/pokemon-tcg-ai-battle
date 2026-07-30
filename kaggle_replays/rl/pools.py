"""学習側レジストリと相手プールの定義(相互鍛錬 段階2)。

元は kaggle_replays/value_net_probe/selfplay_positions.py にあった
LEARNER_REGISTRY / OPPONENT_SPECS / resolve_learner をこちらへ移設したもの。
rl/ から value_net_probe/ を import するのは依存の向きとして不自然なため、定義は rl/ 側に置き、
selfplay_positions.py は本モジュールから import する(定義の二重管理はしない)。

このモジュール自体は cg/ 非依存(パスと文字列の組み立てのみ)。デッキ読込(read_deck_csv_file)は
league/run_league.py に依存するため、実際にファイルを読む build_opponents() 内で遅延 import する。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

# 学習側レジストリ: 表示名 -> (重みファイル名 or None(=production alakazam), デッキのアーキタイプ名)。
# calibrate_learner.py の CANDIDATES と同じ形式・同じ相手プールで実測した勝率(各48試合):
#   表示名                    温度0.05  温度1.0
#   alakazam(production)      0.521     0.271
#   marnie_grimmsnarl_ex      0.479     0.354
#   archaludon_ex             0.479     0.271
#   mega_lucario_ex           0.521     0.375
#   crustle                   0.604     0.667
#   dragapult_ex              0.167     0.062   (デッキ相性に極端に汚染されるため既定の
#                                                 --learners には含めない。指定は可能)
LEARNER_REGISTRY = {
    "alakazam": (None, "alakazam"),
    "marnie_grimmsnarl_ex": ("policy_weights_marnie_grimmsnarl_ex.json", "marnie_grimmsnarl_ex"),
    "archaludon_ex": ("policy_weights_archaludon_ex.json", "archaludon_ex"),
    "mega_lucario_ex": ("policy_weights_mega_lucario_ex.json", "mega_lucario_ex"),
    "crustle": ("policy_weights_crustle.json", "crustle"),
    "dragapult_ex": ("policy_weights_dragapult_ex.json", "dragapult_ex"),
}

DEFAULT_LEARNERS = "alakazam,marnie_grimmsnarl_ex,archaludon_ex,mega_lucario_ex"

# 固定の相手プール4種(kaggle_replays/rl/test_collect_pool.py と同一構成)。
# (name, weights_path_or_None, deck_csv_path) のリスト。デッキはまだ読み込んでいない
# (build_opponents() / selfplay_positions.build_opponents() が読み込む)。
OPPONENT_SPECS = [
    ("alakazam", None, str(DECKDIR / "alakazam" / "01.csv")),
    ("crustle", str(WDIR / "policy_weights_crustle.json"), str(DECKDIR / "crustle" / "01.csv")),
    ("marnie_grimmsnarl_ex", str(WDIR / "policy_weights_marnie_grimmsnarl_ex.json"),
     str(DECKDIR / "marnie_grimmsnarl_ex" / "01.csv")),
    ("archaludon_ex", str(WDIR / "policy_weights_archaludon_ex.json"),
     str(DECKDIR / "archaludon_ex" / "01.csv")),
]


def resolve_learner(name: str):
    """学習側名 -> (weights_path_or_None, deck_csv_path)。レジストリに無ければ ValueError。"""
    if name not in LEARNER_REGISTRY:
        available = ", ".join(sorted(LEARNER_REGISTRY))
        raise ValueError(f"未知の学習側名: {name!r}. 利用可能な名前: {available}")
    weights_file, arch = LEARNER_REGISTRY[name]
    weights_path = str(WDIR / weights_file) if weights_file else None
    deck_csv = str(DECKDIR / arch / "01.csv")
    return weights_path, deck_csv


def build_opponents(names):
    """名前のリスト(LEARNER_REGISTRY のキー) -> parallel_collect_pool 用の
    [(name, weights_path_or_None, deck), ...] を構築する(デッキ読込を含む、遅延 import で
    cg 依存を隔離)。

    train_pool.py の --train-opponents / --eval-opponents / --eval-fixed から共通に使う。
    """
    from run_league import read_deck_csv_file

    opponents = []
    for name in names:
        weights_path, deck_csv = resolve_learner(name)
        opponents.append((name, weights_path, read_deck_csv_file(deck_csv)))
    return opponents
