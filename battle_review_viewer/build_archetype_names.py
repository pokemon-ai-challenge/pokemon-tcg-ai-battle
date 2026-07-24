"""web/archetype_display_names.json（デッキタイプID -> 日本語表示名）をビルドする。

rough_predictor.json の archetypes[*].display_name を単一の情報源（single source of truth）
として、ML版予測器（ml_predictor.py）が返す deck_type（例: "mega_lucario_ex"）を
ビュアー上で日本語のデッキ名（例: "メガルカリオex"）として表示するために使う。

rough_predictor.json に無いラベル（"other" = 未分類バケット）はここで補う。

再生成が必要になるとき: rough_predictor.json のアーキタイプ定義（表示名・追加/削除）が変わったとき。

使い方:
  python battle_review_viewer/build_archetype_names.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PREDICTOR_CONFIG_PATH = ROOT_DIR / "sample_submission" / "ptcg_ai" / "opponent_modeling" / "rough_predictor.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "web" / "archetype_display_names.json"

# rough_predictor.json のアーキタイプ定義に無い、ML予測器（21クラス）側だけの特殊ラベル。
EXTRA_LABELS_JA = {
    "other": "その他（未分類）",
}


def build_archetype_names() -> dict[str, str]:
    payload = json.loads(PREDICTOR_CONFIG_PATH.read_text(encoding="utf-8"))
    names = {
        deck_type: str(archetype.get("display_name", deck_type)).strip()
        for deck_type, archetype in payload.get("archetypes", {}).items()
    }
    names.update(EXTRA_LABELS_JA)
    return dict(sorted(names.items()))


def main() -> None:
    names = build_archetype_names()
    DEFAULT_OUT.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build_archetype_names] {len(names)} 件のアーキタイプ名を書き出しました -> {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
