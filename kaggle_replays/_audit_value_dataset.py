"""Phase6 §7.1: Hand-aware Value 用の学習データ監査。

「手札・トラッシュ・公開相手カードが **カードID単位で** 保存されているか」を確認する。
無ければ Hand-aware Value は作れない(推測補完は禁止 §7.2)。

併せて、既存 Value(V0)の split が **試合単位** で切られているか(= train/test リーク無し)、
ラベルの視点が正しいかも確認する。V1 は V0 と **同一 split** で学習する必要がある。

production 非改変・読み取り専用。出力: _audit_value_dataset_results.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RAW = _HERE / "training_data" / "value_positions.jsonl.gz"
NPZ = _HERE / "value_net" / "features.npz"


def split_for_episode(episode_id: str) -> int:
    """build_features.py と同一の split 規則(md5(episode_id) % 100)。"""
    h = int(hashlib.md5(episode_id.encode("utf-8")).hexdigest(), 16) % 100
    return 0 if h < 80 else 1 if h < 90 else 2


def _ids(cards) -> list[int]:
    """カードリストから card id を取り出す(dict でも int でも拾えるように)。"""
    out = []
    for c in cards or []:
        if isinstance(c, dict):
            v = c.get("id")
            if v is not None:
                out.append(int(v))
        elif isinstance(c, int):
            out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--out", default="_audit_value_dataset_results.json")
    args = ap.parse_args()

    n = 0
    episodes: set[str] = set()
    ep_split: dict[str, int] = {}
    hand_present = hand_nonempty = hand_ids_ok = 0
    disc_present = disc_nonempty = 0
    opp_hand_leak = 0          # 相手手札が中身付きで入っていたらリーク
    opp_visible_ok = 0
    deck_order_leak = 0        # 山札の順序が入っていたらリーク
    label_vals = Counter()
    hand_sizes, disc_sizes, opp_vis_sizes = [], [], []
    uniq_hand_ids: set[int] = set()
    turn_by_hand_empty = Counter()
    per_ep_labels: dict[str, set] = {}

    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        for line in f:
            if n >= args.limit:
                break
            r = json.loads(line)
            n += 1
            ep = r["episode_id"]
            episodes.add(ep)
            ep_split.setdefault(ep, split_for_episode(ep))
            label_vals[r.get("label")] += 1
            per_ep_labels.setdefault(ep, set()).add(
                (r.get("player_index"), r.get("label")))

            cur = (r.get("observation") or {}).get("current") or {}
            players = cur.get("players") or []
            if len(players) != 2:
                continue
            yi = cur.get("yourIndex", 0)
            me, opp = players[yi], players[1 - yi]

            if "hand" in me:
                hand_present += 1
                ids = _ids(me.get("hand"))
                if me.get("hand"):
                    hand_nonempty += 1
                    hand_sizes.append(len(me["hand"]))
                    if ids:
                        hand_ids_ok += 1
                        uniq_hand_ids.update(ids)
                else:
                    turn_by_hand_empty[r.get("turn")] += 1
            if "discard" in me:
                disc_present += 1
                if me.get("discard"):
                    disc_nonempty += 1
                    disc_sizes.append(len(me["discard"]))

            # 相手手札は None であるべき(中身があれば隠れ情報リーク)
            if opp.get("hand"):
                opp_hand_leak += 1
            # 相手の公開領域(場・トラッシュ)は取れるはず
            vis = _ids(opp.get("discard"))
            for slot in (opp.get("active") or []) + (opp.get("bench") or []):
                if isinstance(slot, dict) and slot.get("id") is not None:
                    vis.append(int(slot["id"]))
            if vis:
                opp_visible_ok += 1
                opp_vis_sizes.append(len(vis))
            # 山札の中身/順序が入っていないこと
            if me.get("deck") or opp.get("deck"):
                deck_order_leak += 1

    # ラベル視点の健全性: 同一エピソード内で player0 と player1 のラベルが逆であること
    perspective_ok = perspective_bad = 0
    for ep, pairs in per_ep_labels.items():
        by_p = {}
        for pi, lab in pairs:
            by_p.setdefault(pi, set()).add(lab)
        if len(by_p) == 2 and all(len(v) == 1 for v in by_p.values()):
            labs = [next(iter(v)) for v in by_p.values()]
            (perspective_ok if labs[0] != labs[1] else perspective_bad)
            if labs[0] != labs[1]:
                perspective_ok += 1
            else:
                perspective_bad += 1

    # npz 側の split がエピソード単位か(同一 episode_id が複数 split に跨がっていないか)
    npz_leak = None
    npz_info = {}
    try:
        import numpy as np

        d = np.load(NPZ, allow_pickle=True)
        eid, sp = d["episode_id"], d["split"]
        m: dict[str, set] = {}
        for e, s in zip(eid.tolist(), sp.tolist()):
            m.setdefault(e, set()).add(int(s))
        npz_leak = sum(1 for v in m.values() if len(v) > 1)
        npz_info = {
            "rows": int(eid.shape[0]), "episodes": len(m),
            "episodes_spanning_multiple_splits": npz_leak,
            "split_counts": {str(k): int((sp == k).sum()) for k in (0, 1, 2)},
            "feature_dim": int(d["X"].shape[1]),
        }
    except Exception as exc:  # noqa: BLE001
        npz_info = {"error": str(exc)}

    out = {
        "raw_dataset": str(RAW),
        "positions_scanned": n,
        "episodes_scanned": len(episodes),
        "split_by_episode_rule": "md5(episode_id) % 100 -> <80 train / <90 val / else test",
        "split_distribution_scanned": {
            "train": sum(1 for v in ep_split.values() if v == 0),
            "val": sum(1 for v in ep_split.values() if v == 1),
            "test": sum(1 for v in ep_split.values() if v == 2),
        },
        "hand": {
            "field_present_rate": round(hand_present / n, 4),
            "nonempty_rate": round(hand_nonempty / n, 4),
            "card_id_extractable_rate_of_nonempty": (
                round(hand_ids_ok / hand_nonempty, 4) if hand_nonempty else None),
            "mean_size": round(statistics.mean(hand_sizes), 2) if hand_sizes else None,
            "unique_card_ids_seen": len(uniq_hand_ids),
        },
        "discard": {
            "field_present_rate": round(disc_present / n, 4),
            "nonempty_rate": round(disc_nonempty / n, 4),
            "mean_size": round(statistics.mean(disc_sizes), 2) if disc_sizes else None,
        },
        "opponent_visible": {
            "any_visible_rate": round(opp_visible_ok / n, 4),
            "mean_visible_cards": round(statistics.mean(opp_vis_sizes), 2)
            if opp_vis_sizes else None,
        },
        "leak_checks": {
            "opponent_hand_contents_present": opp_hand_leak,
            "deck_contents_or_order_present": deck_order_leak,
            "PASS": opp_hand_leak == 0 and deck_order_leak == 0,
        },
        "label": {
            "values": dict(label_vals),
            "episodes_with_opposite_labels_per_player": perspective_ok,
            "episodes_with_same_label_both_players": perspective_bad,
            "perspective_PASS": perspective_bad == 0,
        },
        "npz_v0_featurestore": npz_info,
        "VERDICT_hand_aware_feasible": (
            hand_nonempty > 0 and hand_ids_ok > 0 and opp_hand_leak == 0
            and deck_order_leak == 0),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
