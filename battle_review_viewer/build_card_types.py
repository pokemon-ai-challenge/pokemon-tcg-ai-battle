"""cg エンジンの CardData から web/card_types.json（cardId -> CardType int）を生成するビルドスクリプト。

`web/app.js` の `cardTypeOf()` はこの JSON を見て、付属エネルギーが
「基本エネルギー(BASIC_ENERGY=5)」か「特殊エネルギー(SPECIAL_ENERGY=6)」かを判定している。
カード名の `{R}` のようなタイプ記号だけで判定すると、特殊エネルギーの多くは記号を持たないため
無色の基本エネルギーと同じ扱いになってしまう（実際には特殊効果を持つのに区別できない）。
`all_card_data()` は cg エンジン本体が返す正しい CardType を持っているので、これを静的ファイル化して使う。

カードが追加/変更されたとき（`cg.dll` / `libcg.so` が更新されたとき）は、このスクリプトを
再実行して `web/card_types.json` を更新すればよい。

実行例:
    python battle_review_viewer/build_card_types.py
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_SUBMISSION_DIR = ROOT_DIR / "sample_submission"
DEFAULT_OUT = Path(__file__).resolve().parent / "web" / "card_types.json"

if str(SAMPLE_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import all_card_data  # noqa: E402


def build_card_types() -> dict[str, int]:
    return {str(card.cardId): int(card.cardType) for card in all_card_data()}


def main() -> None:
    card_types = build_card_types()
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(card_types, ensure_ascii=False), encoding="utf-8")
    print(f"[build_card_types] {len(card_types)} 件のカードタイプを書き出しました -> {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
