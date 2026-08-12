#!/usr/bin/env python3
"""gen2(2026-08取得)のデッキラベリング専用に rough_predictor.json のパッチ版を作る。

なぜ必要か:
  本番の sample_submission/ptcg_ai/opponent_modeling/rough_predictor.json は
  2026-07 時点の環境で作られており、オーガポンの定義が現行環境と合っていない。

  - `オーガポン みどりのめんex` の role が shared_anchor(=7点)で、label_decks.py の
    しきい値 LABEL_MIN_SCORE=10 に単体では届かない。
  - 加点対象の flex(シアノ/ニャースex/ラティアスex 等)は旧環境の相方で、現行の主流型が
    使う札(ヒーローマント/グロウ【草】エネルギー/テラスタルオーブ 等)が1枚も無い。

  実測(2026-08 上位200チーム 848デッキ): みどりのめんex は87デッキ(10.3%)に入っているのに
  ogerpon_teal_ex とラベル付けされたのは4件だけで、35件が other に落ちていた
  (other 56件の実に41件がオーガポン入り)。この状態で模倣学習のデータを作ると
  「オーガポン」の学習データがほぼ空になる。

パッチ内容:
  1. ogerpon_teal_ex に現行型の専用札を追加(グロウ【草】エネルギー/テラスタルオーブ/
     ヒーローマント)。これで shared_anchor 7点 + 4点 = 11点となり単体で threshold を超える。
     リーリエの決心/ボスの指令/ジャッジマン/ポケギア3.0 等の汎用トレーナーは他デッキにも
     広く入っており、足すと誤判定を招くので**あえて入れない**。
  2. kamitsuorochi_ex 側の オーガポン みどりのめんex を shared_anchor(7, ゲート役) から
     core(3, 非ゲート) に降格。label_decks.py は _GATE_ROLES のカードが1枚でもあれば
     そのアーキタイプの候補にするため、**カミツオロチexが1枚も入っていない純オーガポン
     デッキが kamitsuorochi_ex と判定される**バグがあった(実測21件)。降格後は
     カミツオロチex本体が無いとカミツオロチとは判定されない。

  なお「オーガポンを anchor(10) に昇格させる」案も試したが、イワパレス4枚入りの
  crustle デッキ(オーガポンをエネ加速に採用)を奪ってしまうため採らなかった。
  上の2点なら crustle 12点 > ogerpon 11点 で正しく crustle 側に残る。

本番ファイルは書き換えない。出力は deck_predictor/rough_predictor_g2.json で、
label_decks.py --rough-predictor-json で指定して使う。

使い方:
  python build_rough_predictor_g2.py
  python build_rough_predictor_g2.py --out ./rough_predictor_g2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_PROD_JSON = (
    _HERE.parent.parent / "sample_submission" / "ptcg_ai" / "opponent_modeling" / "rough_predictor.json"
)

# 現行オーガポン型の専用札。汎用トレーナーは含めない(誤判定防止)。
_OGERPON_ADDED_CARDS = [
    {"name": "グロウ【草】エネルギー", "role": "energy"},
    {"name": "テラスタルオーブ", "role": "flex"},
    {"name": "ヒーローマント", "role": "flex"},
]

# ogerpon_teal_ex の flex から外すカード。これらは takeruraiko_ex 自身の看板
# (anchor 相当)で、オーガポン側の加点対象にしたままだと
# 「タケルライコex + オーガポン」デッキをオーガポン側が競り勝って奪う
# (audit_archetype_labels.py の [A] で実測4件)。主役はタケルライコexなので外す。
_OGERPON_REMOVED_CARDS = ["タケルライコex", "ナゲツケサル"]


def _entry_names(entry: dict) -> list[str]:
    if "name" in entry:
        return [entry["name"]]
    return list(entry.get("names", []))


def patch(config: dict) -> tuple[dict, list[str]]:
    changes: list[str] = []
    ogerpon = config["archetypes"]["ogerpon_teal_ex"]

    # kamitsuorochi_ex が オーガポン単体でゲートを通ってしまう問題を塞ぐ
    kamitsu = config["archetypes"]["kamitsuorochi_ex"]
    for entry in kamitsu.get("cards", []):
        if entry.get("name") == "オーガポン みどりのめんex" and entry.get("role") == "shared_anchor":
            entry["role"] = "core"
            changes.append(
                "kamitsuorochi_ex: オーガポン みどりのめんex を shared_anchor -> core"
                "(カミツオロチex無しでカミツオロチ判定される問題を修正)"
            )

    existing = {
        e.get("name")
        for group in ([ogerpon.get("cards", [])] + list(ogerpon.get("role_cards", {}).values()))
        for e in group
    }
    role_cards = ogerpon.setdefault("role_cards", {})
    for card in _OGERPON_ADDED_CARDS:
        if card["name"] in existing:
            continue
        role_cards.setdefault(card["role"], []).append(dict(card))
        changes.append(f"ogerpon_teal_ex: {card['name']} を {card['role']} として追加")

    for role, entries in list(role_cards.items()):
        kept = [e for e in entries if not set(_entry_names(e)) & set(_OGERPON_REMOVED_CARDS)]
        removed = [n for e in entries for n in _entry_names(e) if n in _OGERPON_REMOVED_CARDS]
        if removed:
            role_cards[role] = kept
            for name in removed:
                changes.append(f"ogerpon_teal_ex: {name} を {role} から除外(takeruraiko_ex の看板のため)")

    return config, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=str(_PROD_JSON), help="元にする本番 rough_predictor.json(読むだけ)")
    parser.add_argument("--out", default=str(_HERE / "rough_predictor_g2.json"))
    args = parser.parse_args()

    src_path = Path(args.src)
    config = json.loads(src_path.read_text(encoding="utf-8"))
    config, changes = patch(config)
    config["_g2_patch_note"] = (
        "kaggle_replays/deck_predictor/build_rough_predictor_g2.py が生成。"
        "gen2ラベリング専用で、本番 rough_predictor.json は不変。"
    )
    config["_g2_patch_changes"] = changes

    out_path = Path(args.out)
    out_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"元: {src_path}")
    print(f"出力: {out_path}")
    print(f"パッチ {len(changes)}件:")
    for c in changes:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
