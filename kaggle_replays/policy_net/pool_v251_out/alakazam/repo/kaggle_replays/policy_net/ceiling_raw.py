#!/usr/bin/env python3
"""模倣学習(BC)の「教師一致率の天井」を、生の局面から測り直す。

背景（ロードマップ A4）:
    過去に報告した「天井 0.7685」（diagnose_errors.py の一貫性チェック）は、
    局面キーを 166 次元の状態特徴を丸めたハッシュで作っていたため、
    カードの同一性が入っておらず、別局面を同一視して天井を過小評価していた
    （＝一致率が本来より低く出ていた）可能性がある。

    本スクリプトは、生の observation（手札・場・トラッシュの card_id、
    デッキ残数、サイド残数、ターン、先攻/後攻、select_type/select_context）
    から局面キーを作り直し、選択された行動も「配列インデックス」ではなく
    「行動の意味」（option.type + 解決した card_id 等）に正規化して、
    教師一致率の天井を測る。

area / OptionType の対応は sample_submission/cg/api.py の AreaType /
OptionType / Option の docstring コメントを読んで確認した（推測していない）。
特に:
  - CARD(type=3) は area によって参照先が違う。
    area=DECK(1)  -> observation.select.deck[index]（deck 配列は探索時のみ非 null）
    area=LOOKING(12) -> observation.current.looking[index]
    area=STADIUM(7)  -> observation.current.stadium[index]
    area=HAND(2) / DISCARD(3) -> players[playerIndex] の該当ゾーン[index]
    area=ACTIVE(4) / BENCH(5) -> players[playerIndex] の該当ゾーン[index]（ポケモン参照）
    area=PRIZE(6)    -> 伏せられていて指し手本人にも中身が見えない
                        （players[i].prize は常に None で観測される）。
                        したがって card_id を解決できず、位置（area,index）を
                        そのまま行動識別子として使う（これは「推測」ではなく、
                        観測データそのものが伏せ札であることの反映）。
  - ATTACH(8) / EVOLVE(9) の area は常に HAND(2)（データで確認済み）、
    inPlayArea は自分の ACTIVE(4) / BENCH(5)。
  - ABILITY(10) の area は ACTIVE(4) / BENCH(5) / STADIUM(7)
    （スタジアムのアビリティ的効果も同じ OptionType で表現されている。
    実データで area=7 の例を確認し、current.stadium[index] と突き合わせて検証済み）。
  - TOOL_CARD(4) / ENERGY_CARD(5) / ENERGY(6) は area+index で対象ポケモンを
    特定し、toolIndex / energyIndex でポケモンに付いている tools / energyCards
    配列から card_id を引く。

実データ（このアーカイブの 818,350 行）で実際に出現する
(select_type, option keys, option.type) の組み合わせは全て列挙して確認済み
（DISCARD(11) / ENERGY_CARD(5) / CARD_OR_ATTACHED_CARD(select_type=3) /
SKILL(select_type=5) / SPECIAL_CONDITION(select_type=10) はこのファイルには
出現しない）。未知の組み合わせに遭遇した場合はクラッシュさせず
UNRESOLVED/UNKNOWN としてカウントし、最後に警告として件数を出す。

使い方:
    python ceiling_raw.py --limit 100000
    python ceiling_raw.py            # 全件
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

_DEFAULT_JSONL = _ROOT / "kaggle_replays" / "training_data" / "policy_positions_marnie_grimmsnarl_ex.jsonl.gz"
_DEFAULT_FEATURES = _HERE / "features_marnie_grimmsnarl_ex.npz"
_DEFAULT_META = _ROOT / "kaggle_replays" / "index" / "episodes_master.jsonl"
_DEFAULT_OUT = _HERE / "ceiling_raw_2026-08-07.json"

# --- sample_submission/cg/api.py の AreaType / OptionType 定義（コピーではなく値のみ）---
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_ENERGY = 8
AREA_TOOL = 9
AREA_PRE_EVOLUTION = 10
AREA_PLAYER = 11
AREA_LOOKING = 12

OT_NUMBER = 0
OT_YES = 1
OT_NO = 2
OT_CARD = 3
OT_TOOL_CARD = 4
OT_ENERGY_CARD = 5
OT_ENERGY = 6
OT_PLAY = 7
OT_ATTACH = 8
OT_EVOLVE = 9
OT_ABILITY = 10
OT_DISCARD = 11
OT_RETREAT = 12
OT_ATTACK = 13
OT_END = 14
OT_SKILL = 15
OT_SPECIAL_CONDITION = 16

_NOPT_BUCKETS = ("2", "3-4", "5-8", "9-16", "17+")


def _nopt_bucket(n: int) -> str:
    return ("2" if n <= 2 else "3-4" if n <= 4 else "5-8" if n <= 8
            else "9-16" if n <= 16 else "17+")


# ---------------------------------------------------------------------------
# 行動キー: option の意味を、配列インデックスではなく card_id / attackId 等で表す
# ---------------------------------------------------------------------------

class ResolveError(Exception):
    pass


def _resolve_card_ref(players, player_idx, area, index, deck, looking, stadium):
    """CARD(type=3) 系の area+index を解決する。
    戻り値は ('C', card_id) か ('P', card_id, player_idx, 'active'/'bench')（場のポケモン）
    か ('HIDDEN', ...) （伏せられていて中身不明）。
    """
    if area == AREA_DECK:
        if deck is None:
            raise ResolveError("area=DECK だが select.deck が None")
        return ("C", deck[index]["id"])
    if area == AREA_LOOKING:
        if looking is None:
            raise ResolveError("area=LOOKING だが current.looking が None")
        return ("C", looking[index]["id"])
    if area == AREA_STADIUM:
        return ("C", stadium[index]["id"])
    if area == AREA_HAND:
        hand = players[player_idx]["hand"]
        if hand is None:
            # 相手の手札は非公開。実データでは発生しない想定（HAND は自分の
            # 手札選択でのみ使われる）が、防御的に位置情報にフォールバックする。
            return ("HIDDEN", "hand", player_idx, index)
        return ("C", hand[index]["id"])
    if area == AREA_DISCARD:
        return ("C", players[player_idx]["discard"][index]["id"])
    if area == AREA_ACTIVE:
        pk = players[player_idx]["active"][index]
        return ("P", pk["id"], player_idx, "active")
    if area == AREA_BENCH:
        pk = players[player_idx]["bench"][index]
        return ("P", pk["id"], player_idx, "bench")
    if area == AREA_PRIZE:
        # 伏せ札。指し手本人にも card_id は見えない（観測データでも常に None）。
        return ("HIDDEN", "prize", player_idx, index)
    return ("UNRESOLVED_AREA", area, player_idx, index)


def _resolve_pokemon(players, player_idx, area, index):
    if area == AREA_ACTIVE:
        return players[player_idx]["active"][index], "active"
    if area == AREA_BENCH:
        return players[player_idx]["bench"][index], "bench"
    raise ResolveError(f"ポケモン参照の area が想定外: {area}")


def action_key(option, players, your_index, deck, looking, stadium):
    """選択された1件の option を、行動の意味を表すタプルに正規化する。"""
    t = option["type"]

    if t == OT_NUMBER:
        return ("NUMBER", option.get("number"))
    if t == OT_YES:
        return ("YES",)
    if t == OT_NO:
        return ("NO",)

    if t == OT_CARD:
        ref = _resolve_card_ref(players, option["playerIndex"], option["area"], option["index"],
                                 deck, looking, stadium)
        return ("CARD", ref)

    if t in (OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY):
        pk, zone = _resolve_pokemon(players, option["playerIndex"], option["area"], option["index"])
        if t == OT_TOOL_CARD:
            attach = pk["tools"][option["toolIndex"]]
            kind = "TOOL_CARD"
        else:
            attach = pk["energyCards"][option["energyIndex"]]
            kind = "ENERGY_CARD" if t == OT_ENERGY_CARD else "ENERGY"
        # 対象ポケモン + 付いているカードの card_id で正規化。
        # 同じ card_id の別コピー（同じポケモンに付いた基本エネ4枚など）は
        # energyIndex/toolIndex が違っても attach["id"] が同じなら同一行動になる。
        return (kind, attach["id"], pk["id"], option["playerIndex"], zone)

    if t == OT_PLAY:
        c = players[your_index]["hand"][option["index"]]
        return ("PLAY", c["id"])

    if t in (OT_ATTACH, OT_EVOLVE):
        src = players[your_index]["hand"][option["index"]]
        pk, zone = _resolve_pokemon(players, your_index, option["inPlayArea"], option["inPlayIndex"])
        kind = "ATTACH" if t == OT_ATTACH else "EVOLVE"
        return (kind, src["id"], pk["id"], zone)

    if t == OT_ABILITY:
        area = option["area"]
        index = option["index"]
        if area == AREA_STADIUM:
            return ("ABILITY", stadium[index]["id"])
        pk, _zone = _resolve_pokemon(players, your_index, area, index)
        return ("ABILITY", pk["id"])

    if t == OT_DISCARD:
        ref = _resolve_card_ref(players, your_index, option["area"], option["index"], deck, looking, stadium)
        return ("DISCARD", ref)

    if t == OT_RETREAT:
        return ("RETREAT",)
    if t == OT_ATTACK:
        return ("ATTACK", option.get("attackId"))
    if t == OT_END:
        return ("END",)
    if t == OT_SKILL:
        return ("SKILL", option.get("cardId"))
    if t == OT_SPECIAL_CONDITION:
        return ("SPECIAL_CONDITION", option.get("specialConditionType"))

    return ("UNKNOWN_OPTTYPE", t)


# ---------------------------------------------------------------------------
# 局面キー: 生の情報から作る（丸めない）
# ---------------------------------------------------------------------------

def _board_tuple(pokemon_list):
    out = []
    for pk in pokemon_list or []:
        if pk is None:
            continue
        energies = tuple(sorted(c["id"] for c in (pk.get("energyCards") or [])))
        out.append((pk["id"], pk["hp"], energies))
    return tuple(sorted(out))


def build_new_state_key(row):
    cur = row["observation"]["current"]
    players = cur["players"]
    yi = cur["yourIndex"]
    oi = 1 - yi
    me = players[yi]
    opp = players[oi]

    own_hand = tuple(sorted(c["id"] for c in (me.get("hand") or [])))
    own_active = _board_tuple(me.get("active"))
    own_bench = _board_tuple(me.get("bench"))
    opp_active = _board_tuple(opp.get("active"))
    opp_bench = _board_tuple(opp.get("bench"))
    own_discard = tuple(sorted(c["id"] for c in (me.get("discard") or [])))

    first_player = cur.get("firstPlayer")
    is_first = (first_player == yi)

    return (
        own_hand,
        own_active, own_bench,
        opp_active, opp_bench,
        own_discard,
        me.get("deckCount"), opp.get("deckCount"),
        len(me.get("prize") or []), len(opp.get("prize") or []),
        opp.get("handCount"),
        cur.get("turn"), is_first,
        # turnActionCount: ターン内で既に何回行動したか。手札/場/トラッシュが
        # 変化しない行動（例: 特定スタジアムのアビリティ使用）の直後にもう一度
        # MAIN選択が来ると、これを入れないと「同じ局面」に潰れてしまう
        # （実データで確認: episode 88057262 step134/135 など）。
        cur.get("turnActionCount"),
        row["select_type"], row["select_context"],
    )


def digest_of(key) -> bytes:
    return hashlib.blake2b(repr(key).encode("utf-8", "surrogatepass"), digest_size=16).digest()


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def load_team_meta(meta_path: Path) -> dict:
    meta = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            eid = obj["episode_id"]
            for p in obj["players"]:
                meta[(eid, p["player_index"])] = p.get("team_id")
    return meta


def ncr2(n: int) -> int:
    return n * (n - 1) // 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(_DEFAULT_JSONL))
    ap.add_argument("--features", default=str(_DEFAULT_FEATURES), help="旧方式比較用の166次元特徴 npz")
    ap.add_argument("--meta", default=str(_DEFAULT_META), help="episode_id -> team_id を引く episodes_master.jsonl")
    ap.add_argument("--output", default=str(_DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="先頭N行のみ処理")
    ap.add_argument("--progress-every", type=int, default=100000)
    args = ap.parse_args()

    t0 = time.time()

    print(f"[1/4] team_id メタデータ読み込み: {args.meta}")
    team_meta = load_team_meta(Path(args.meta))
    print(f"      {len(team_meta)} 件（episode_id, player_index）-> team_id")

    print(f"[2/4] 旧方式比較用 166次元特徴 読み込み: {args.features}")
    feat = np.load(args.features, allow_pickle=True)
    state_features = feat["state_features"]
    split_arr = feat["split"]
    n_feat = state_features.shape[0]
    print(f"      state_features shape={state_features.shape}")

    print(f"[3/4] 生ログを読みながら局面キー/行動キーを構築: {args.input}")

    groups_new: dict[bytes, collections.Counter] = collections.defaultdict(collections.Counter)
    groups_old: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    # 行ごとの軽量レコード（バケット別集計・(a)(b)判定に使う）
    row_digest = []
    row_old_key = []
    row_bucket = []
    row_new_action = []
    row_old_action = []
    row_split = []

    n_resolve_error = 0
    resolve_error_examples = []
    n_rows = 0

    opener = gzip.open(args.input, "rt", encoding="utf-8")
    with opener as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            row = json.loads(line)
            n_rows += 1

            cur = row["observation"]["current"]
            players = cur["players"]
            your_index = cur["yourIndex"]
            sel = row["observation"]["select"]
            deck = sel.get("deck")
            looking = cur.get("looking")
            stadium = cur.get("stadium")
            options = sel["option"]
            chosen = row["chosen_index"]
            n_options = row["n_options"]
            bucket = _nopt_bucket(n_options)

            try:
                new_action = action_key(options[chosen], players, your_index, deck, looking, stadium)
            except (ResolveError, IndexError, KeyError, TypeError) as e:
                n_resolve_error += 1
                if len(resolve_error_examples) < 10:
                    resolve_error_examples.append(
                        f"episode={row.get('episode_id')} step={row.get('step_index')} "
                        f"select_type={row.get('select_type')} select_context={row.get('select_context')} "
                        f"opt={options[chosen] if chosen < len(options) else None} err={e!r}"
                    )
                new_action = ("RESOLVE_ERROR",)

            new_state = build_new_state_key(row)
            digest = digest_of(new_state)

            eid = row["episode_id"]
            team_id = team_meta.get((eid, your_index))

            groups_new[digest][(team_id, new_action)] += 1

            old_key = (hash(np.round(state_features[i].astype(np.float64), 3).tobytes()), n_options)
            groups_old[old_key][chosen] += 1

            row_digest.append(digest)
            row_old_key.append(old_key)
            row_bucket.append(bucket)
            row_new_action.append(new_action)
            row_old_action.append(chosen)
            row_split.append(int(split_arr[i]))

            if (i + 1) % args.progress_every == 0:
                dt = time.time() - t0
                print(f"      {i + 1} 行処理済み ({dt:.1f}s, {(i + 1) / dt:.0f} rows/s)")

    dt = time.time() - t0
    print(f"      完了: {n_rows} 行 ({dt:.1f}s)")
    if n_resolve_error:
        print(f"      !! action_key 解決に失敗した行: {n_resolve_error} 件（RESOLVE_ERROR として集計に含めた)")
        for ex in resolve_error_examples:
            print(f"         例: {ex}")

    print("[4/4] 集計")

    # --- グループごとの多数派行動（新方式: team_id を周辺化） ---
    new_group_total = {}
    new_group_major = {}
    for digest, ctr in groups_new.items():
        marginal = collections.Counter()
        for (team_id, act), c in ctr.items():
            marginal[act] += c
        total = sum(marginal.values())
        major_act, major_n = marginal.most_common(1)[0]
        new_group_total[digest] = total
        new_group_major[digest] = major_act

    old_group_total = {}
    old_group_major = {}
    for old_key, ctr in groups_old.items():
        total = sum(ctr.values())
        major_act, _ = ctr.most_common(1)[0]
        old_group_total[old_key] = total
        old_group_major[old_key] = major_act

    def bucketed_ceiling(row_key_list, group_total, group_major, row_bucket_list, restrict_split=None,
                          row_split_list=None):
        overall = {"hit_a": 0, "n_a": 0, "hit_b": 0, "n_b": 0}
        by_bucket = {b: {"hit_a": 0, "n_a": 0, "hit_b": 0, "n_b": 0} for b in _NOPT_BUCKETS}
        for idx in range(len(row_key_list)):
            if restrict_split is not None and row_split_list[idx] != restrict_split:
                continue
            key = row_key_list[idx]
            act = row_new_action[idx] if group_major is new_group_major else row_old_action[idx]
            gtotal = group_total[key]
            is_major = int(act == group_major[key])
            b = row_bucket_list[idx]
            overall["hit_a"] += is_major
            overall["n_a"] += 1
            by_bucket[b]["hit_a"] += is_major
            by_bucket[b]["n_a"] += 1
            if gtotal >= 2:
                overall["hit_b"] += is_major
                overall["n_b"] += 1
                by_bucket[b]["hit_b"] += is_major
                by_bucket[b]["n_b"] += 1
        return overall, by_bucket

    def fmt(d):
        out = dict(d)
        out["ceiling_a"] = d["hit_a"] / d["n_a"] if d["n_a"] else None
        out["ceiling_b"] = d["hit_b"] / d["n_b"] if d["n_b"] else None
        return out

    new_overall, new_by_bucket = bucketed_ceiling(row_digest, new_group_total, new_group_major, row_bucket)
    old_overall, old_by_bucket = bucketed_ceiling(row_old_key, old_group_total, old_group_major, row_bucket)

    # 旧方式・test split のみ（元の 0.7685 の再現チェック）
    old_overall_test, old_by_bucket_test = bucketed_ceiling(
        row_old_key, old_group_total, old_group_major, row_bucket,
        restrict_split=2, row_split_list=row_split,
    )

    # --- (c) グループサイズ分布 ---
    def size_dist(group_total):
        sizes = list(group_total.values())
        n_groups = len(sizes)
        n1 = sum(1 for s in sizes if s == 1)
        n2p = n_groups - n1
        max_size = max(sizes) if sizes else 0
        dec_in_n1 = sum(s for s in sizes if s == 1)
        dec_in_n2p = sum(s for s in sizes if s >= 2)
        return {
            "n_groups": n_groups,
            "n_groups_size1": n1,
            "n_groups_size1_frac": n1 / n_groups if n_groups else None,
            "n_groups_size2plus": n2p,
            "n_groups_size2plus_frac": n2p / n_groups if n_groups else None,
            "max_group_size": max_size,
            "decisions_in_size1_groups": dec_in_n1,
            "decisions_in_size2plus_groups": dec_in_n2p,
        }

    new_dist = size_dist(new_group_total)
    old_dist = size_dist(old_group_total)

    # --- 4. 同一プレイヤー内 vs 異プレイヤー間（新方式のみ、team_id が既知の場合） ---
    # 巨大な1グループ（例: turn=0 の完全対称な初期局面）が cross_player_pairs を
    # 支配してしまうと「異プレイヤー間の一致率」が実質1グループの数値になり、
    # スタイル差の指標として意味をなさない。そのため max_group_size でグループの
    # 大きさに上限をかけた版も別途計算する。
    def compute_player_split(max_group_size=None):
        same_pairs = 0
        same_agree_pairs = 0
        cross_pairs = 0
        cross_agree_pairs = 0
        n_groups_used = 0
        n_rows_no_team = 0
        n_groups_excluded_for_size = 0
        for digest, ctr in groups_new.items():
            total = new_group_total[digest]
            if total < 2:
                continue
            if max_group_size is not None and total > max_group_size:
                n_groups_excluded_for_size += 1
                continue
            known = {(t, a): c for (t, a), c in ctr.items() if t is not None}
            n_known = sum(known.values())
            n_rows_no_team += total - n_known
            if n_known < 2:
                continue
            n_groups_used += 1
            total_pairs_known = ncr2(n_known)
            agree_pairs_known = 0
            act_marginal = collections.Counter()
            for (t, a), c in known.items():
                act_marginal[a] += c
            for a, c in act_marginal.items():
                agree_pairs_known += ncr2(c)

            team_marginal = collections.Counter()
            for (t, a), c in known.items():
                team_marginal[t] += c
            for t, c in team_marginal.items():
                same_pairs += ncr2(c)
            for (t, a), c in known.items():
                same_agree_pairs += ncr2(c)

            cross_pairs += total_pairs_known - sum(ncr2(c) for c in team_marginal.values())
            cross_agree_pairs += agree_pairs_known - sum(
                ncr2(c) for (t, a), c in known.items()
            )

        return {
            "max_group_size_cap": max_group_size,
            "usable": same_pairs + cross_pairs > 0,
            "n_repeated_groups_with_known_team": n_groups_used,
            "n_groups_excluded_for_size": n_groups_excluded_for_size,
            "decisions_with_unknown_team_in_repeated_groups": n_rows_no_team,
            "same_player_pairs": same_pairs,
            "same_player_agree_pairs": same_agree_pairs,
            "same_player_agreement_rate": (same_agree_pairs / same_pairs) if same_pairs else None,
            "cross_player_pairs": cross_pairs,
            "cross_player_agree_pairs": cross_agree_pairs,
            "cross_player_agreement_rate": (cross_agree_pairs / cross_pairs) if cross_pairs else None,
        }

    player_split_all = compute_player_split(max_group_size=None)
    player_split_capped = compute_player_split(max_group_size=50)
    player_split = {
        "all_repeated_groups": player_split_all,
        "excluding_groups_larger_than_50": player_split_capped,
        "note": "all_repeated_groups は巨大な退化グループ（例: turn=0 の完全対称な初期局面が"
                "4450件、ほぼ固定の1アクションに収束）に支配され、cross_player の一致率が"
                "ほぼ1.0になる。excluding_groups_larger_than_50 の方が「実質的にスタイル差が"
                "現れうる程度の繰り返し」に近い。",
    }

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": str(args.input),
        "n_rows": n_rows,
        "n_resolve_error": n_resolve_error,
        "resolve_error_examples": resolve_error_examples,
        "new_method": {
            "overall": fmt(new_overall),
            "by_n_options_bucket": {b: fmt(v) for b, v in new_by_bucket.items()},
            "group_size_distribution": new_dist,
        },
        "old_method_same_subset": {
            "overall": fmt(old_overall),
            "by_n_options_bucket": {b: fmt(v) for b, v in old_by_bucket.items()},
            "group_size_distribution": old_dist,
        },
        "old_method_test_split_only_sanity_check": {
            "note": "diagnose_errors.py が元々計算していたのと同じ split(=test) だけに絞った旧方式の値。"
                    "0.7685 の再現確認用。",
            "overall": fmt(old_overall_test),
            "by_n_options_bucket": {b: fmt(v) for b, v in old_by_bucket_test.items()},
        },
        "player_split": player_split,
        "elapsed_sec": time.time() - t0,
    }

    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {args.output}")

    print("\n=== 新方式 (a) 全決定点 ===")
    print(json.dumps(fmt(new_overall), ensure_ascii=False, indent=2))
    print("\n=== 旧方式 (a) 全決定点・同一サブセット ===")
    print(json.dumps(fmt(old_overall), ensure_ascii=False, indent=2))
    print("\n=== 旧方式 (a) test split のみ（0.7685 再現チェック）===")
    print(json.dumps(fmt(old_overall_test), ensure_ascii=False, indent=2))
    print("\n=== 新方式 グループサイズ分布 ===")
    print(json.dumps(new_dist, ensure_ascii=False, indent=2))
    print("\n=== 旧方式 グループサイズ分布 ===")
    print(json.dumps(old_dist, ensure_ascii=False, indent=2))
    print("\n=== プレイヤー分割 ===")
    print(json.dumps(player_split, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
