#!/usr/bin/env python3
"""E-Stage0: フーディン(alakazam)デッキ候補の系統的生成。

方針(Codex 5.6-sol と合意した選別プロトコルの入口。memory:
project_beyond_imitation_codex_plan 参照):
  - 種デッキ = 実績のあるリスト(Plan A / hammer4 / gen2上位パイロット5本)。
  - 候補 = 種そのもの + 種の近傍変形(1〜2スロットの入れ替え)で 64〜96本。
  - 変形は「実在する採用例があるカードプール内」でのみ行う(発明はしない)。
    プール = gen2 の alakazam ラベル891デッキで実際に使われたカードと採用率。
  - 幹(フーディンライン/アメ/ポフィン等のエンジン)は固定し、flex枠だけ動かす。

生成規則(1候補=種+差分):
  swap1: flexカードAを1枚減らし、プール頻出カードBを1枚足す
  swap2: swap1 を独立に2回(A,B が重複しないもの)
  ヒューリスティック制約: 60枚厳守 / 同名4枚以下(基本エネ除く) / たね>=そのまま /
  エンジンカード(下記 CORE)は減らさない。

出力: kaggle_replays/deck_search/candidates/<name>.csv と manifest.jsonl
(種名・差分・生成規則を記録。Stage1 の対戦割当とseedブロックは run 側で行う)。

使い方:
  python generate_candidates.py --max-candidates 96

alakazam 以外のアーキタイプにも使えるようパラメータ化してある(--arch / --seeds-dir /
--core-ids / --energy-floor / --interp-frac)。引数を渡さなければ従来通り alakazam の
挙動(SEEDS / CORE_IDS ハードコード・interp なし)のまま動く。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent

SEEDS = {
    # 実績のある種デッキ。名前 -> パス
    "planA": _ROOT / "sample_submission" / "deck.csv",
    "hammer4": _ROOT / "experimental_decks" / "foodin_hammer4_v1" / "deck.csv",
    "g2top1": _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2" / "alakazam" / "01.csv",
    "g2top2": _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2" / "alakazam" / "02.csv",
    "g2top3": _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2" / "alakazam" / "03.csv",
}

# エンジン(幹)。ここは減らさない。ID はプロジェクト既知の対応
# (741 ケーシィ / 742 ユンゲラー / 743 フーディン / 1079 ふしぎなアメ / 1086 なかよしポフィン)。
CORE_IDS = {741, 742, 743, 1079, 1086}

# 基本エネルギー(枚数制限なし帯)。JP_Card_Data の ID 1-8。
BASIC_ENERGY = set(range(1, 9))


def load_deck(path: Path) -> Counter:
    ids = [int(r[0]) for r in csv.reader(path.open(encoding="utf-8"))
           if r and r[0].strip().isdigit()]
    assert len(ids) == 60, f"{path}: {len(ids)}枚"
    return Counter(ids)


def load_seeds(seeds_dir: str | None) -> dict[str, Path]:
    """種デッキのパス一覧を返す。

    --seeds-dir 未指定なら現行の SEEDS(alakazam用ハードコード)をそのまま返す。
    指定時は <seeds_dir>/01.csv〜05.csv のうち存在するものを g2top1..g2top5 として使う。
    """
    if seeds_dir is None:
        return SEEDS
    d = Path(seeds_dir)
    out = {}
    for i in range(1, 6):
        p = d / f"0{i}.csv"
        if p.exists():
            out[f"g2top{i}"] = p
    return out


def load_pool(arch: str = "alakazam") -> tuple[Counter, dict[int, str]]:
    """gen2の指定アーキタイプのデッキ群から「実際に採用されているカード」の頻度プールを作る。"""
    db = {}
    for r in csv.DictReader((_ROOT / "data" / "JP_Card_Data.csv").open(encoding="utf-8")):
        db.setdefault(int(r["カード ID"]), r["カード名"])
    labels = {}
    lab_p = _ROOT / "kaggle_replays" / "deck_predictor" / "output" / "deck_labels_g2.jsonl"
    for line in lab_p.open(encoding="utf-8"):
        row = json.loads(line)
        labels[(row["episode_id"], row["player_index"])] = row["archetype"]
    pool: Counter = Counter()
    db_p = _ROOT / "kaggle_replays" / "deck_predictor" / "output" / "deck_db_g2.jsonl"
    for line in db_p.open(encoding="utf-8"):
        row = json.loads(line)
        if labels.get((row["episode_id"], row["player_index"])) != arch:
            continue
        for cid in set(int(x) for x in row.get("deck_card_ids", [])):
            pool[cid] += 1
    return pool, db


def reducible(deck: Counter, cid: int, core_ids: set[int], energy_floor: int | None) -> bool:
    """cid を1枚減らせるかどうか(CORE保護・基本エネのfloor制約を考慮)。

    energy_floor が None なら基本エネルギー(BASIC_ENERGY)は従来通り減らせない。
    """
    if cid in core_ids:
        return False
    if cid in BASIC_ENERGY:
        if energy_floor is None:
            return False
        basic_total = sum(n for c, n in deck.items() if c in BASIC_ENERGY)
        return basic_total - 1 >= energy_floor
    return True


def deck_to_rows(deck: Counter) -> list[int]:
    out = []
    for cid, n in sorted(deck.items()):
        out += [cid] * n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-candidates", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--out-dir", default=str(_HERE / "candidates"))
    ap.add_argument("--min-pool-count", type=int, default=30,
                    help="プール内でこのデッキ数以上に採用されているカードだけを追加候補にする")
    ap.add_argument("--arch", default="alakazam",
                    help="deck_labels_g2.jsonl のアーキタイプラベル(プール構築のフィルタ)")
    ap.add_argument("--seeds-dir", default=None,
                    help="種デッキのディレクトリ。指定時は <dir>/01.csv〜05.csv の"
                         "存在するものを種にする(g2top1..g2top5)。省略時は現行のSEEDS")
    ap.add_argument("--core-ids", default=None,
                    help="カンマ区切りのCORE ID(減らさないエンジンカード)。省略時は既定のCORE_IDS(alakazam用)")
    ap.add_argument("--energy-floor", type=int, default=None,
                    help="指定時、基本エネルギー(ID1-8)も「減らす」対象に含める。"
                         "デッキ内の基本エネルギー合計がこの枚数を下回る変形は生成しない。"
                         "省略時は現行挙動(基本エネは減らさない)")
    ap.add_argument("--interp-frac", type=float, default=0.0,
                    help="変形候補のうち interp(種2本の内挿)にする割合。既定0.0=従来通りswap1/swap2のみ")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    core_ids = {int(x) for x in args.core_ids.split(",") if x.strip()} if args.core_ids else CORE_IDS

    pool, names = load_pool(args.arch)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_paths = load_seeds(args.seeds_dir)
    seeds = {k: load_deck(p) for k, p in seed_paths.items() if p.exists()}
    print(f"種デッキ {len(seeds)} 本: {sorted(seeds)}")

    # 追加候補カード = 指定アーキタイプ(--arch)のプールで min_pool_count 以上採用されているカード
    addable = [cid for cid, n in pool.most_common() if n >= args.min_pool_count]
    print(f"追加候補プール: {len(addable)} 種(採用{args.min_pool_count}デッキ以上)")

    manifest = []
    made: set[tuple] = set()

    # ACE SPEC は**デッキ全体で合計1枚まで**(同名1枚ではない。ニュートラルセンター+
    # リッチエネルギーの2枚構成が battle_start errorType=4 で拒否されるのを実測)。
    ace_ids = set()
    for r in csv.DictReader((_ROOT / "data" / "JP_Card_Data.csv").open(encoding="utf-8")):
        if "ACE" in str(r.get("ルール", "")).upper():
            ace_ids.add(int(r["カード ID"]))

    def try_emit(name: str, deck: Counter, meta: dict) -> bool:
        key = tuple(sorted(deck.items()))
        if key in made:
            return False
        if sum(deck.values()) != 60:
            return False
        for cid, n in deck.items():
            if cid not in BASIC_ENERGY and n > 4:
                return False
            if n < 0:
                return False
        if sum(n for cid, n in deck.items() if cid in ace_ids) > 1:
            return False
        if args.energy_floor is not None:
            basic_total = sum(n for cid, n in deck.items() if cid in BASIC_ENERGY)
            if basic_total < args.energy_floor:
                return False
        made.add(key)
        path = out_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            for cid in deck_to_rows(deck):
                f.write(f"{cid}\n")
        manifest.append({"name": name, **meta})
        return True

    # 1) 種そのものを候補に含める(Stage1で全候補と同条件比較するため)
    for sname, deck in seeds.items():
        try_emit(f"seed_{sname}", deck, {"kind": "seed", "base": sname, "diff": []})

    # 2) 近傍変形
    n_target = args.max_candidates
    attempts = 0
    while len(manifest) < n_target and attempts < 20000:
        attempts += 1
        # interp: 種2本 A,B の内挿(seedsが2本以上あるときだけ発生しうる)。
        # interp_frac<=0(既定)のときは rng.random() すら呼ばず、既存の乱数消費順序を変えない
        # (これにより --interp-frac 未指定時は従来と完全に同じ出力になる)。
        if len(seeds) >= 2 and args.interp_frac > 0 and rng.random() < args.interp_frac:
            sname_a, sname_b = rng.sample(list(seeds), 2)
            deck = Counter(seeds[sname_a])
            target = seeds[sname_b]
            k = rng.choice([2, 3])
            diffs = []
            ok = True
            for _ in range(k):
                # 減らす: A(現在の deck)にあって B より多いカードから、CORE保護/energy-floorを尊重
                dec_cands = [cid for cid, n in deck.items()
                            if n > target.get(cid, 0) and reducible(deck, cid, core_ids, args.energy_floor)]
                if not dec_cands:
                    ok = False
                    break
                # 足す: B にあって現在の deck より多いカードから(同名4枚制限内)
                inc_cands = [cid for cid, n in target.items()
                            if n > deck.get(cid, 0) and (cid in BASIC_ENERGY or deck.get(cid, 0) < 4)]
                if not inc_cands:
                    ok = False
                    break
                dec = rng.choice(dec_cands)
                inc = rng.choice(inc_cands)
                deck[dec] -= 1
                if deck[dec] == 0:
                    del deck[dec]
                deck[inc] = deck.get(inc, 0) + 1
                diffs.append({"out": dec, "out_name": names.get(dec, "?"),
                              "in": inc, "in_name": names.get(inc, "?")})
            if not ok:
                continue
            nm = f"{sname_a}_to_{sname_b}_v{len(manifest):03d}"
            try_emit(nm, deck, {"kind": f"interp{k}", "base": sname_a, "target": sname_b, "diff": diffs})
            continue

        sname = rng.choice(list(seeds))
        base = seeds[sname]
        deck = Counter(base)
        n_swaps = 1 if rng.random() < 0.6 else 2
        diffs = []
        ok = True
        for _ in range(n_swaps):
            # 減らす: flex(非CORE・非基本エネ。energy-floor指定時は基本エネもfloorを守る範囲で対象)
            flex = [cid for cid, n in deck.items()
                    if n > 0 and reducible(deck, cid, core_ids, args.energy_floor)]
            if not flex:
                ok = False
                break
            dec = rng.choice(flex)
            # 足す: プールから(同名4枚制限内・減らしたカードと別)
            cands = [c for c in addable if c != dec and (c in BASIC_ENERGY or deck.get(c, 0) < 4)]
            if not cands:
                ok = False
                break
            # 採用率に比例した重みでサンプル(頻出カードを優先しつつ多様性を残す)
            weights = [pool[c] for c in cands]
            inc = rng.choices(cands, weights=weights, k=1)[0]
            deck[dec] -= 1
            if deck[dec] == 0:
                del deck[dec]
            deck[inc] = deck.get(inc, 0) + 1
            diffs.append({"out": dec, "out_name": names.get(dec, "?"),
                          "in": inc, "in_name": names.get(inc, "?")})
        if not ok:
            continue
        nm = f"{sname}_v{len(manifest):03d}"
        try_emit(nm, deck, {"kind": f"swap{n_swaps}", "base": sname, "diff": diffs})

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    kinds = Counter(m["kind"] for m in manifest)
    bases = Counter(m["base"] for m in manifest)
    print(f"生成 {len(manifest)} 候補 -> {out_dir}")
    print(f"  内訳 kind={dict(kinds)}  base={dict(bases)}")


if __name__ == "__main__":
    main()
