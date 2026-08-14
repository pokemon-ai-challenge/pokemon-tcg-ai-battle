#!/usr/bin/env python3
"""rough_predictor.json に rocket_mewtwo_ex アーキタイプを追加する。

主軸(アンカー)は 431 ロケット団のミュウツーex(唯一のゲート=このカードが無いデッキは
このアーキタイプに分類しない)。ロケット団エンジン系カードは rocket_honchkrow と共有する
ため core に置き、区別は「891 ドンカラスを持つ=honchkrow / 431 ミュウツーex を持つ=本型」
という anchor の違いで行う。400/401 も honchkrow と共有するので signature にはしない。

カード名は data/JP_Card_Data.csv から card_id で解決(手打ちの文字化け回避)。
JSON はキー順・日本語をそのまま保つため ensure_ascii=False で書き戻す。
冪等: 既に rocket_mewtwo_ex があれば置換。

使い方: python register_rocket_mewtwo.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

_REPO = Path(__file__).parent.parent.parent
_RP = _REPO / "sample_submission" / "ptcg_ai" / "opponent_modeling" / "rough_predictor.json"
_CSV = _REPO / "data" / "JP_Card_Data.csv"

ANCHOR = 431
CORE = [414, 400, 401, 1216, 1220, 1218, 1134, 1257, 434, 1217, 15]
FLEX = [1227, 1094]


def load_names() -> dict[int, str]:
    names = {}
    for row in csv.reader(_CSV.open(encoding="utf-8", errors="replace")):
        if row and row[0].isdigit():
            names[int(row[0])] = row[1]
    return names


def main():
    names = load_names()

    def entry(cid, **extra):
        return {"name": names.get(cid, str(cid)), "card_id": [cid], **extra}

    arche = {
        "display_name": names.get(ANCHOR, str(ANCHOR)),
        "cards": [
            {
                "name": names.get(ANCHOR),
                "card_id": [ANCHOR],
                "role": "anchor",
                "reason": "デッキの主軸カード。このexが唯一の分類ゲート(rocket_honchkrowとはドンカラス891/ミュウツーex431の違いで区別)",
            }
        ],
        "role_cards": {
            "core": [entry(c) for c in CORE],
            "flex": [entry(c) for c in FLEX],
        },
        "energy_types": [],
        "combo_rules": [],
        "variants": {},
        "min_score": 6,
        "confident_score": 12,
    }

    config = json.loads(_RP.read_text(encoding="utf-8"))
    config["archetypes"]["rocket_mewtwo_ex"] = arche
    _RP.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("registered rocket_mewtwo_ex")
    print("  anchor:", names.get(ANCHOR))
    print("  core  :", [names.get(c) for c in CORE])
    print("  flex  :", [names.get(c) for c in FLEX])


if __name__ == "__main__":
    main()
