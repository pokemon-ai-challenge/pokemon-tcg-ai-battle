#!/usr/bin/env python3
"""``data/EN_Card_Data.csv`` からカードID -> 静的属性テーブルを作る。

出力は ``{str(card_id): {attr_name: value(float)}}`` の JSON。
``ptcg_ai.learning.policy_features.action_features()`` の ``card_attributes`` 引数と
``policy_model.py`` の重みJSON ``card_attributes`` フィールドにそのまま渡せる形。

属性名は固定でこのモジュールが決める（policy_features.py 側はキー名に一切依存しない
=> 学習側だけが列を追加・変更してよい）。

## 列の対応（実測: data/EN_Card_Data.csv, 2022行）

- `Category` 列は 1630/2022 行が `n/a` で、Pokemon/Trainer/Energy の分類には使えない。
  分類は `Stage (Pokémon)/Type (Energy and Trainer)` 列の値で判定する:
    - "Basic Pokémon" / "Stage 1 Pokémon" / "Stage 2 Pokémon" -> Pokemon
    - "Item" / "Supporter" / "Pokémon Tool" / "Stadium" -> Trainer
    - "Special Energy" / "Basic Energy" -> Energy
- `HP` / `Retreat` は Pokemon 以外では "n/a"。0埋め + missing フラグで対応する。
- `Type` / `Weakness` は `{X}` 形式のシンボル。中身（G/P/W/F/D/C/R/L/M/竜 等）を
  one-hot として展開する（複合シンボルはまれで無視できる規模のため単純化）。
- `Rule` 列は "Pokémon ex" / "Mega Pokémon ex" / "ACE SPEC" 等のフラグ。

## 正規化

HP は /340（実測最大値付近）、Retreat は /4 で [0, 1] 程度にスケールする。
正規化した値をそのままテーブルに埋め込むため、推論側（policy_features.py）は
正規化を意識する必要がない（生の値をそのまま特徴量として使うだけ）。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_DEFAULT_CSV = _REPO_ROOT / "data" / "EN_Card_Data.csv"
_DEFAULT_OUT = _HERE / "output" / "card_attributes.json"

_STAGE_COL = "Stage (Pokémon)/Type (Energy and Trainer)"

_POKEMON_STAGES = {"Basic Pokémon": 0.0, "Stage 1 Pokémon": 1.0, "Stage 2 Pokémon": 2.0}
_TRAINER_SUBTYPES = {"Item", "Supporter", "Pokémon Tool", "Stadium"}
_ENERGY_SUBTYPES = {"Basic Energy", "Special Energy"}

_HP_SCALE = 340.0  # 実測: EN_Card_Data.csv の HP 最大値付近(Mega/ex 込み)
_RETREAT_SCALE = 4.0  # 実測: Retreat の最大値


def _symbol(raw: str) -> str | None:
    """``"{G}"`` -> ``"G"``。複数シンボル(``"{C}{C}{C}"`` 等)は先頭のみ使う。
    "n/a" / 空文字列は None。
    """
    if not raw or raw == "n/a":
        return None
    m = re.search(r"\{([^}]+)\}", raw)
    if m:
        return m.group(1)
    return raw or None


def _to_float_or_none(raw: str) -> float | None:
    if raw is None or raw == "n/a" or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def build_attributes(csv_path: Path) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            card_id = row.get("Card ID")
            if not card_id:
                continue
            stage_raw = (row.get(_STAGE_COL) or "").strip()

            attrs: dict[str, float] = {}

            is_pokemon = stage_raw in _POKEMON_STAGES
            is_trainer = stage_raw in _TRAINER_SUBTYPES
            is_energy = stage_raw in _ENERGY_SUBTYPES

            attrs["category_pokemon"] = 1.0 if is_pokemon else 0.0
            attrs["category_trainer"] = 1.0 if is_trainer else 0.0
            attrs["category_energy"] = 1.0 if is_energy else 0.0

            # --- Pokemon 専用属性(Trainer/Energy は 0埋め + missing フラグ) ---
            stage_value = _POKEMON_STAGES.get(stage_raw)
            attrs["stage"] = stage_value if stage_value is not None else 0.0
            attrs["stage_missing"] = 0.0 if stage_value is not None else 1.0

            hp = _to_float_or_none(row.get("HP"))
            attrs["hp"] = (hp / _HP_SCALE) if hp is not None else 0.0
            attrs["hp_missing"] = 0.0 if hp is not None else 1.0

            retreat = _to_float_or_none(row.get("Retreat"))
            attrs["retreat_cost"] = (retreat / _RETREAT_SCALE) if retreat is not None else 0.0
            attrs["retreat_missing"] = 0.0 if retreat is not None else 1.0

            type_symbol = _symbol(row.get("Type") or "")
            if type_symbol is not None:
                attrs[f"type_{type_symbol}"] = 1.0

            weakness_symbol = _symbol(row.get("Weakness") or "")
            if weakness_symbol is not None:
                attrs[f"weakness_{weakness_symbol}"] = 1.0

            resistance_symbol = _symbol(row.get("Resistance (Type)") or "")
            if resistance_symbol is not None:
                attrs[f"resistance_{resistance_symbol}"] = 1.0

            # --- Trainer 専用属性(Item/Supporter/Tool/Stadium one-hot) ---
            if is_trainer:
                slug = stage_raw.lower().replace(" ", "_").replace("é", "e")
                attrs[f"trainer_subtype_{slug}"] = 1.0

            if is_energy:
                slug = stage_raw.lower().replace(" ", "_")
                attrs[f"energy_subtype_{slug}"] = 1.0

            # --- ルールフラグ ---
            rule = (row.get("Rule") or "").strip()
            attrs["is_ex"] = 1.0 if rule == "Pokémon ex" else 0.0
            attrs["is_mega_ex"] = 1.0 if rule == "Mega Pokémon ex" else 0.0
            attrs["is_ace_spec"] = 1.0 if rule == "ACE SPEC" else 0.0

            table[str(int(card_id))] = attrs
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    table = build_attributes(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=0), encoding="utf-8")

    n_attrs = len({name for attrs in table.values() for name in attrs})
    print(f"カード {len(table):,} 種 / 属性 {n_attrs} 種 -> {args.out}")
    n_pokemon = sum(1 for a in table.values() if a["category_pokemon"] == 1.0)
    n_trainer = sum(1 for a in table.values() if a["category_trainer"] == 1.0)
    n_energy = sum(1 for a in table.values() if a["category_energy"] == 1.0)
    print(f"  category: pokemon={n_pokemon} trainer={n_trainer} energy={n_energy} "
          f"other={len(table) - n_pokemon - n_trainer - n_energy}")


if __name__ == "__main__":
    main()
