"""Phase6B: Hand-aware Value 用のカード集合特徴を、既存 V0 特徴量と**同じ行**で作る。

既存 `features.npz` は state 166 次元しか持たない。本スクリプトは同じ
`value_positions.jsonl.gz` を走査し、各局面の

  - 自分の手札        (card_id の多重集合)
  - 自分のトラッシュ  (同上)
  - 相手の公開カード  (場のポケモン + トラッシュ。**手札・山札は入れない**)

を ragged 配列(flat ids + offsets)で保存する。V0 と **行の対応が取れること** が要件なので、
`(episode_id, player_index, step_index)` をキーに突き合わせ、一致率を報告する。

隠れ情報は入れない: 相手の hand は None のはずで、非 None なら例外にする(監査で0件を確認済み)。
出力: card_features.npz
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
RAW = _HERE.parent / "training_data" / "value_positions.jsonl.gz"
NPZ = _HERE / "features.npz"
OUT = _HERE / "card_features.npz"

MAX_HAND, MAX_DISCARD, MAX_OPP = 30, 48, 48


def _ids(cards, cap: int) -> list[int]:
    out = []
    for c in cards or []:
        if isinstance(c, dict):
            v = c.get("id")
            if v is not None:
                out.append(int(v))
        elif isinstance(c, int):
            out.append(c)
        if len(out) >= cap:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=全件")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    keys = {}
    eid, pi, si = d["episode_id"].tolist(), d["player_index"].tolist(), d["step_index"].tolist()
    for row, k in enumerate(zip(eid, pi, si)):
        keys[k] = row
    n_rows = len(eid)
    print(f"V0 rows={n_rows} unique_keys={len(keys)}", file=sys.stderr)

    hand_l: list[list[int]] = [[] for _ in range(n_rows)]
    disc_l: list[list[int]] = [[] for _ in range(n_rows)]
    opp_l: list[list[int]] = [[] for _ in range(n_rows)]
    matched = 0
    scanned = 0
    leak = 0

    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        for line in f:
            if args.limit and scanned >= args.limit:
                break
            r = json.loads(line)
            scanned += 1
            k = (r["episode_id"], r["player_index"], r["step_index"])
            row = keys.get(k)
            if row is None:
                continue
            cur = (r.get("observation") or {}).get("current") or {}
            players = cur.get("players") or []
            if len(players) != 2:
                continue
            yi = cur.get("yourIndex", 0)
            me, opp = players[yi], players[1 - yi]
            if opp.get("hand"):
                leak += 1          # 監査で0件。出たら止める。
                continue
            hand_l[row] = _ids(me.get("hand"), MAX_HAND)
            disc_l[row] = _ids(me.get("discard"), MAX_DISCARD)
            vis = _ids(opp.get("discard"), MAX_OPP)
            for slot in (opp.get("active") or []) + (opp.get("bench") or []):
                if isinstance(slot, dict) and slot.get("id") is not None and len(vis) < MAX_OPP:
                    vis.append(int(slot["id"]))
            opp_l[row] = vis
            matched += 1
            if matched % 100000 == 0:
                print(f"  matched {matched}", file=sys.stderr, flush=True)

    if leak:
        raise SystemExit(f"相手手札の中身が {leak} 件混入している。隠れ情報リークのため中止。")

    def pack(lists):
        off = np.zeros(len(lists) + 1, dtype=np.int64)
        for i, v in enumerate(lists):
            off[i + 1] = off[i] + len(v)
        flat = np.fromiter((x for v in lists for x in v), dtype=np.int32,
                           count=int(off[-1]))
        return flat, off

    h_flat, h_off = pack(hand_l)
    d_flat, d_off = pack(disc_l)
    o_flat, o_off = pack(opp_l)

    np.savez_compressed(
        args.out,
        hand_flat=h_flat, hand_off=h_off,
        discard_flat=d_flat, discard_off=d_off,
        opp_flat=o_flat, opp_off=o_off,
        matched=np.array([matched]), n_rows=np.array([n_rows]),
    )
    print(json.dumps({
        "v0_rows": n_rows, "scanned": scanned, "matched": matched,
        "match_rate": round(matched / n_rows, 4),
        "mean_hand": round(float(np.diff(h_off).mean()), 2),
        "mean_discard": round(float(np.diff(d_off).mean()), 2),
        "mean_opp_visible": round(float(np.diff(o_off).mean()), 2),
        "max_card_id": int(max(h_flat.max(initial=0), d_flat.max(initial=0),
                               o_flat.max(initial=0))),
        "opponent_hand_leak_rows": leak,
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
