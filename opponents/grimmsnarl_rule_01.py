"""ルールベース マリィのオーロンゲex 対戦相手(01: 標準ユキメノコ型, Luca, LB 1242.3)。

出典: `kaggle_replays/meta_analysis/archetype_decks/marnie_grimmsnarl_ex/01.csv`
(LBスコア・チーム名は同ディレクトリの `manifest.json` 参照)。ロジック本体は
`opponents/grimmsnarl_core.py`(設計: `sample_submission/docs/plans/opponent-training/
grimmsnarl-rule-implementation-design.md`)。対戦相手(sparring partner)であり、
こちらの提出(`sample_submission/`)や production の rule_based とは無関係。

league から使うときは `--deck-b opponents/grimmsnarl_ex_deck_01.csv` を必ず対応させること
(`grimmsnarl_core.set_card_counts` が DECK 構成に依存するため)。
"""
import os

from opponents import grimmsnarl_core

_DECK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grimmsnarl_ex_deck_01.csv")
with open(_DECK_PATH, "r") as _file:
    _csv = _file.read().split("\n")
DECK = [int(_csv[i]) for i in range(60)]
PROFILE = grimmsnarl_core.build_profile(DECK)


def agent(obs) -> list[int]:
    """Kaggle 基盤(dict)/league(Observation)の両方の入力契約を満たす。"""
    return grimmsnarl_core.agent(obs, deck=DECK, profile=PROFILE)
