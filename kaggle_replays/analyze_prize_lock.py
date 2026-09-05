#!/usr/bin/env python3
"""サイド落ちの実頻度を上位リプレイから数える。

`kaggle_replays/docs/imitation/hidden-zone-features-design-2026-07-30.md` §8「先に測るべき安い前哨戦」の実装。
非公開ゾーン特徴（とくに `p_prized(X)` = カード X がサイド落ちしている確率）を
方策の入力に足す案に意味があるかを、試合を回さずに判定するための集計。

## 何を測るか

サイドは伏せ札（`prize` は `[null, ...]`）なので中身は直接読めない。ただし
**サイドを取ると、そのカードは手札に入る**。カードは試合内で一意な `serial` を
持つので、次の対応で実際のサイドの中身が復元できる:

    prize の枚数が k 減ったステップで、手札に新しく現れた serial k 個 = そのとき取ったサイド

取られなかったサイドは最後まで不明なので、**「取られたサイドだけ」が観測対象**になる。
勝者は6枚すべて取っているため、勝者側はサイド6枚が完全に判明する。

デッキ構成は「その試合でどこかのゾーンに現れた serial」を全部集めて近似する
（山札に埋もれたまま終わったカードは数えられないので、コピー枚数は**過小評価**になりうる。
このバイアスは「1枚しか入っていない」と誤判定する方向に働くため、
1-of の集計はやや多めに出る。結論に効く場合はその旨を明記すること）。

## 出力

- サイド落ちしたカードのコピー枚数分布（1-of が何割か）
- 試合あたり「1-of がサイドに落ちた」割合
- サイド落ち枚数と勝敗の関係

使い方:
    python kaggle_replays/analyze_prize_lock.py --limit 500
    python kaggle_replays/analyze_prize_lock.py            # 全件
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_REPLAYS = _HERE / "replays"


def load_archetype_decks(deck_dir: Path) -> dict[str, list[collections.Counter]]:
    """archetype -> [その型の実デッキリスト(card_id -> 枚数), ...] を読む。

    リプレイ内で見えたカードだけからコピー枚数を復元すると、山札に埋もれたまま
    終わった分を数え落として「1枚積み」を水増しする。実デッキリストがあるなら
    そちらを使うほうが正確なので、型ごとの実リスト（各5本）を読み込んでおく。
    """
    out: dict[str, list[collections.Counter]] = {}
    if not deck_dir.exists():
        return out
    for arch_dir in sorted(deck_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        lists = []
        for csv in sorted(arch_dir.glob("*.csv")):
            ids = []
            for line in csv.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.isdigit():
                    ids.append(int(line))
            if ids:
                lists.append(collections.Counter(ids))
        if lists:
            out[arch_dir.name] = lists
    return out


def load_deck_labels(path: Path) -> dict[tuple[str, int], str]:
    """(episode_id, player_index) -> archetype。"""
    out: dict[tuple[str, int], str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[(str(d.get("episode_id")), d.get("player_index"))] = d.get("archetype")
    return out


def pick_decklist(observed_ids: set[int], candidates: list[collections.Counter]):
    """観測されたカードidを最もよく含むデッキリストを選ぶ。候補が無ければ None。"""
    if not candidates:
        return None
    return max(candidates, key=lambda c: sum(1 for i in observed_ids if i in c))


def _iter_cards(node):
    """入れ子の任意構造から {'serial':..., 'id':...} を持つ dict を全部拾う。"""
    if isinstance(node, dict):
        if "serial" in node and "id" in node:
            yield node
        for v in node.values():
            yield from _iter_cards(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_cards(v)


def analyze_replay(doc) -> list[dict]:
    """1リプレイから player ごとの集計を返す。観測できなかった player は含めない。"""
    steps = doc.get("steps") or []
    rewards = doc.get("rewards") or []

    # player_index -> 直近の (prize枚数, 手札serial集合)
    prev_prize_n: dict[int, int] = {}
    prev_hand: dict[int, set] = {}
    prized_ids: dict[int, list] = collections.defaultdict(list)
    seen_serial_to_id: dict[int, dict[int, int]] = collections.defaultdict(dict)
    saw_player: set[int] = set()

    for step in steps:
        for entry in step:
            obs = entry.get("observation") or {}
            cur = obs.get("current")
            if not cur:
                continue
            me_idx = cur.get("yourIndex")
            if me_idx is None:
                continue
            players = cur.get("players") or []
            if me_idx >= len(players):
                continue
            me = players[me_idx]
            saw_player.add(me_idx)

            # 自分の全ゾーンに現れたカードを記録（デッキ構成の近似に使う）。
            for c in _iter_cards(me):
                seen_serial_to_id[me_idx][c["serial"]] = c["id"]

            hand = me.get("hand") or []
            hand_serials = {c["serial"] for c in _iter_cards(hand)}
            prize_n = len(me.get("prize") or [])

            if me_idx in prev_prize_n:
                drop = prev_prize_n[me_idx] - prize_n
                if drop > 0:
                    new_serials = hand_serials - prev_hand.get(me_idx, set())
                    # サイドを取った瞬間に手札へ増えた分を、その回のサイドとみなす。
                    # ドローと同時に起きると混ざりうるので、増分が drop と一致する場合のみ採用。
                    if len(new_serials) == drop:
                        for s in new_serials:
                            cid = seen_serial_to_id[me_idx].get(s)
                            if cid is not None:
                                prized_ids[me_idx].append(cid)

            prev_prize_n[me_idx] = prize_n
            prev_hand[me_idx] = hand_serials

    out = []
    for pi in sorted(saw_player):
        id_counts = collections.Counter(seen_serial_to_id[pi].values())
        pz = prized_ids.get(pi, [])
        won = None
        if len(rewards) > pi and rewards[pi] is not None:
            other = rewards[1 - pi] if len(rewards) > 1 - pi else None
            if other is not None:
                won = 1 if rewards[pi] > other else (0 if rewards[pi] < other else -1)
        out.append({
            "player_index": pi,
            "n_prized_observed": len(pz),
            "prized_card_ids": pz,
            "deck_id_counts": id_counts,
            "won": won,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays-dir", default=str(_REPLAYS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(Path(args.replays_dir).glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    decks = load_archetype_decks(_HERE / "meta_analysis" / "archetype_decks")
    labels = load_deck_labels(_HERE / "deck_predictor" / "output" / "deck_labels.jsonl")
    print(f"実デッキリスト: {len(decks)} 型  deck_labels: {len(labels)} 件\n")

    copies_of_prized = collections.Counter()   # コピー枚数 -> サイド落ちした枚数
    games_with_1of_prized = 0
    games_counted = 0
    full_prize_games = 0                        # サイド6枚すべて判明した player-game
    prized_n_by_outcome = {0: [], 1: []}
    n_players = 0
    n_used_real_decklist = 0

    for n, fp in enumerate(files, 1):
        if n % 500 == 0:
            print(f"  {n}/{len(files)} …", flush=True)
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        # doc["id"] は UUID であって episode_id ではない。deck_labels と突き合わせるのは
        # info.EpisodeId（ファイル名 episode-<id>-replay.json の <id> と同じ）。
        episode_id = str((doc.get("info") or {}).get("EpisodeId")
                         or fp.stem.replace("episode-", "").replace("-replay", ""))
        for rec in analyze_replay(doc):
            n_players += 1
            # コピー枚数は実デッキリスト優先。無ければ観測からの復元にフォールバックする
            # （復元は山札に埋もれた分を数え落とすため 1-of を水増しする方向に偏る）。
            arch = labels.get((episode_id, rec["player_index"]))
            counts = None
            if arch and arch in decks:
                counts = pick_decklist(set(rec["deck_id_counts"]), decks[arch])
                if counts is not None:
                    n_used_real_decklist += 1
            if counts is None:
                counts = rec["deck_id_counts"]
            pz = rec["prized_card_ids"]
            if rec["n_prized_observed"] >= 6:
                full_prize_games += 1
            if rec["won"] in (0, 1):
                prized_n_by_outcome[rec["won"]].append(rec["n_prized_observed"])
            if not pz:
                continue
            games_counted += 1
            has_1of = False
            for cid in pz:
                c = counts.get(cid, 0)
                copies_of_prized[c] += 1
                if c == 1:
                    has_1of = True
            if has_1of:
                games_with_1of_prized += 1

    print()
    print(f"リプレイ {len(files)} 件 / player-game {n_players} 件")
    print(f"実デッキリストで枚数を引けた player-game: {n_used_real_decklist} "
          f"({n_used_real_decklist/max(1,n_players):.1%})")
    print(f"サイドを1枚以上観測できた player-game: {games_counted}")
    print(f"うちサイド6枚すべて判明（＝勝者側）: {full_prize_games}")
    print()
    total_prized = sum(copies_of_prized.values())
    print(f"=== サイド落ちしたカード {total_prized} 枚のコピー枚数分布 ===")
    for c in sorted(copies_of_prized):
        n = copies_of_prized[c]
        label = f"{c}枚積み" if c else "不明(山札に埋没)"
        print(f"  {label:<16} {n:>7} 枚  ({n/max(1,total_prized):6.1%})")
    print()
    if games_counted:
        print(f"1-of がサイドに落ちた player-game: {games_with_1of_prized}/{games_counted} "
              f"({games_with_1of_prized/games_counted:.1%})")
    for w in (1, 0):
        arr = prized_n_by_outcome[w]
        if arr:
            print(f"  {'勝' if w else '敗'}: 観測できたサイド枚数 平均 {sum(arr)/len(arr):.2f} (n={len(arr)})")


if __name__ == "__main__":
    main()
