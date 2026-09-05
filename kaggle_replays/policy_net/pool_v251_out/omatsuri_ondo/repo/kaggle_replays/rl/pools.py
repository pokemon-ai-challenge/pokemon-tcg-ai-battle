"""学習側レジストリと相手プールの定義(相互鍛錬 段階2)。

元は kaggle_replays/value_net_probe/selfplay_positions.py にあった
LEARNER_REGISTRY / OPPONENT_SPECS / resolve_learner をこちらへ移設したもの。
rl/ から value_net_probe/ を import するのは依存の向きとして不自然なため、定義は rl/ 側に置き、
selfplay_positions.py は本モジュールから import する(定義の二重管理はしない)。

このモジュール自体は cg/ 非依存(パスと文字列の組み立てのみ)。デッキ読込(read_deck_csv_file)は
league/run_league.py に依存するため、実際にファイルを読む build_opponents() 内で遅延 import する。
"""

from __future__ import annotations

import json
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
    # 段階I(各アーキタイプを自分のミラーで60イテレーション鍛えたもの)の成果。
    # 全5体が BC 版の自分に勝ち越している(ミラー最終 0.599〜0.790、各1,200試合)。
    # 段階II「プール学習 vs 固定相手学習」を、相手が強い状態で測り直すために使う。
    # 前回の段階2は学習側だけが BC で相手が強く、crustle が壁になって
    # 学習予算の25%が情報の乏しい勾配に消えていた。
    "alakazam_k60": ("policy_weights_alakazam_pool_k60.json", "alakazam"),
    "crustle_k60": ("policy_weights_crustle_pool_k60.json", "crustle"),
    "marnie_grimmsnarl_ex_k60": ("policy_weights_marnie_grimmsnarl_ex_pool_k60.json",
                                 "marnie_grimmsnarl_ex"),
    "archaludon_ex_k60": ("policy_weights_archaludon_ex_pool_k60.json", "archaludon_ex"),
    "mega_lucario_ex_k60": ("policy_weights_mega_lucario_ex_pool_k60.json", "mega_lucario_ex"),
    # deck_health_check.py によるデッキ健全性チェックで健全と判定された5アーキタイプ
    # (マリガン率35%未満・デッキ枚数60枚)。run_archetype_pipeline.py の標準3ステップ
    # (extract_policy_dataset.py -> build_features.py --weight-scheme concentrated -> train.py)
    # でそのまま模倣学習した BC ポリシー。POOL8 の構成要素。
    "rocket_mewtwo_ex": ("policy_weights_rocket_mewtwo_ex.json", "rocket_mewtwo_ex"),
    "omatsuri_ondo": ("policy_weights_omatsuri_ondo.json", "omatsuri_ondo"),
    "shirona_garchomp_ex": ("policy_weights_shirona_garchomp_ex.json", "shirona_garchomp_ex"),
    "ogerpon_teal_ex": ("policy_weights_ogerpon_teal_ex.json", "ogerpon_teal_ex"),
    "dragapult_ex": ("policy_weights_dragapult_ex.json", "dragapult_ex"),
}

#: 段階I で鍛え上がった相手プール(段階II の学習用)。
STRONG_POOL = "alakazam_k60,crustle_k60,marnie_grimmsnarl_ex_k60,archaludon_ex_k60"

DEFAULT_LEARNERS = "alakazam,marnie_grimmsnarl_ex,archaludon_ex,mega_lucario_ex"

#: 評価・学習用の8アーキタイプ拡張プール。deck_health_check.py で
#: 「デッキ不良」(マリガン率35%以上)と判定された archaludon_ex を除外している。
#: archaludon_ex はたねポケモンが5枚しか無く初手マリガン率52.5%、実測でも終局理由の大半が
#: ベンチ切れという壊れたデッキで、これを評価プールに混ぜると相手にほぼ無料の勝ちを
#: 供給してしまい勝率という指標そのものを歪める(詳細は deck_health_check.py の docstring
#: および kaggle_replays/rl/train_pool.py の DEFAULT_POOL コメント参照)。
#: DEFAULT_POOL(train_pool.py)は過去の測定値との比較のため4体のまま変更しない。新規の
#: 評価・学習はこちらの POOL8 を使う。
POOL8 = ("alakazam,crustle,marnie_grimmsnarl_ex,rocket_mewtwo_ex,omatsuri_ondo,"
         "shirona_garchomp_ex,ogerpon_teal_ex,dragapult_ex")

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


def load_extra_registry(path) -> None:
    """{name: [weights_file_basename_or_abspath, archetype], ...} を LEARNER_REGISTRY に足す。

    段階III(相互鍛錬ループ、train_league.py)向け。過去世代のチェックポイントは名前もパスも
    実行時にしか決まらないため、LEARNER_REGISTRY(静的な辞書)に載せておくことができない。
    そこで JSON ファイル経由で動的に追加登録できるようにする。

    weights_file は resolve_learner() が ``WDIR / weights_file`` として解決するため、
    絶対パスを渡せば WDIR は無視されそのまま使われる(pathlib の仕様: 右辺が絶対パスなら
    左辺は捨てられる)。相対パス(ファイル名のみ)を渡した場合は WDIR 直下として扱われる
    (LEARNER_REGISTRY の既存エントリと同じ挙動)。

    既存キーと同名を指定した場合は上書きする(呼び出し側が意図して使う想定。例えば
    train_league.py は世代ごとに ``<name>_cur`` を最新の現行重みで上書きする)。

    このモジュールを import しただけでは何も起きない(呼ばれない限り LEARNER_REGISTRY は
    変わらない)。train_pool.py の既定挙動(--extra-registry 省略時)には一切影響しない。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, entry in data.items():
        weights_file, arch = entry
        LEARNER_REGISTRY[name] = (weights_file, arch)


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
