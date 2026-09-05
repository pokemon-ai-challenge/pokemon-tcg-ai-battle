#!/usr/bin/env python3
"""E-Stage1: デッキ候補の粗選別(各候補200試合、ブロック化)。

プロトコル(project_beyond_imitation_codex_plan):
  - 全候補 + 現行(planA種)を、同一の相手ブロックで対戦させ「対現行差」で順位付け。
  - 相手 = mix フィールド(gen2実測11アーキ、シェア比例で試合数を配分)。
  - 方策は全候補共通で climb 重み(デッキだけを動かす。方策適応は採用後の別段)。
  - seed はブロック化: 候補 i の試合 j は seed = base + j(全候補で同じ列)。
    先後はrun_league側の交互割当に従う。
  - Stage1 は粗選別(200試合 → 95%CI±7pt)。上位16本だけが Stage2(新seed 1000試合)へ。

使い方:
  python stage1_screen.py --games 200 --workers 14
  python stage1_screen.py --analyze          # 結果集計と上位16の出力

alakazam 以外のアーキタイプにも使えるようパラメータ化してある(--policy-weights /
--learner-arch / --candidates-dir / --results-file / --baseline)。引数を渡さなければ
従来通り alakazam・climb重み・candidates/・seed_planA基準のまま動く。

対戦 config は --config-base で指定する(既定は従来の ml_lethal_attackplan_v0only)。提出構成は
探索ONの abl_5_full なので、デッキ×探索の相互作用まで見るなら採用予定 config を明示すること
(Stage2 以降の方針)。--analyze は「対現行差」の95%CIを併記する(SE_diff = sqrt(SE_c^2+SE_b^2))。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT / "league"), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
POLICY = "policy_weights_alakazam_rl_climb.json"  # 全候補共通(デッキだけ動かす)
CONFIG = "ml_lethal_attackplan_v0only"

# mix フィールド(train_league.FIELD_MIX と同一の11アーキ+シェア)
FIELD = [
    ("marnie_grimmsnarl_ex", 1082, "_g2"), ("alakazam", 891, "_g2"),
    ("mega_lucario_ex", 403, ""), ("dragapult_ex", 381, "_g2"),
    ("mega_froslass_ex", 353, "_g2"), ("ogerpon_teal_ex", 310, "_g2"),
    ("archaludon_ex", 287, "_g2"), ("crustle", 287, "_g2"),
    ("shirona_garchomp_ex", 158, "_g2"), ("omatsuri_ondo", 138, "_g2"),
    ("rocket_mewtwo_ex", 112, "_g2"),
]
_DECKDIR_OF = {"": "archetype_decks", "_g2": "archetype_decks_g2"}


def allocate_games(total: int) -> list[tuple[str, str, str, int]]:
    """シェア比例で相手別試合数を割り当てる(最小2試合)。"""
    tot = sum(s for _, s, _ in FIELD)
    out = []
    for arch, share, gen in FIELD:
        n = max(2, round(total * share / tot))
        deckdir = _DECKDIR_OF[gen]
        out.append((arch, gen, deckdir, n))
    return out


def run_candidate(name: str, deck_path: Path, games: int, workers: int, seed_base: int,
                  policy_weights_path: str, learner_arch: str, log,
                  config_base: str = CONFIG) -> dict:
    """1候補を FIELD 全アーキと対戦させ、合計勝率を返す。

    ``config_base`` は両陣営に渡す agent config 名(既定 ``CONFIG`` = 従来と同じ v0only)。
    提出構成は探索ON(``abl_5_full``)なので、デッキ×探索の相互作用まで測りたい場合は
    採用予定の config 名を渡す(Stage2 以降の方針)。
    """
    import run_league
    rows = {}
    wins = tot = 0
    for arch, gen, deckdir, n in allocate_games(games):
        summary = run_league.run_league(
            agent_a_name="ml_policy", agent_b_name="ml_policy", games=n,
            deck_a_path=str(deck_path),
            deck_b_path=str(_ROOT / "kaggle_replays" / "meta_analysis" / deckdir / arch / "01.csv"),
            seed_start=seed_base, progress_every=0,
            weights_a_path=policy_weights_path,
            weights_b_path=str(WDIR / f"policy_weights_{arch}{gen}.json") if arch != learner_arch
            else policy_weights_path,  # ミラー(相手arch==learner_arch)は学習側と同じ重みを使う
            # (alakazamの既定では「ミラーは強いclimb」というリーグ学習と同じ想定を一般化したもの)
            config_base=config_base, workers=workers, log=lambda m: None,
        )
        ov = summary["overall"]
        rows[arch] = {"wins": ov["wins"], "games": ov["games"]}
        wins += ov["wins"]
        tot += ov["games"]
    if tot == 0:
        # 全試合エラー(不正デッキ等)。クラッシュさせず失敗として記録して次へ。
        log(f"  {name}: 全試合エラー(スキップ)")
        return {"name": name, "wins": 0, "games": 0, "failed": True, "by_opp": rows,
                "config": config_base}
    log(f"  {name}: {wins}/{tot} = {wins / tot:.3f}")
    # config はプロベナンス用(どの構成で測ったか)。--analyze は参照しない。
    return {"name": name, "wins": wins, "games": tot, "by_opp": rows, "config": config_base}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=910000)
    ap.add_argument("--candidates-dir", default=str(_HERE / "candidates"))
    ap.add_argument("--results-file", default=str(_HERE / "stage1_results.jsonl"))
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--top-n", type=int, default=16)
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="候補ディレクトリの先頭からこの本数だけ処理する(スモーク検証用。省略時は全件)")
    ap.add_argument("--policy-weights", default=str(WDIR / POLICY),
                    help="全候補共通の学習側重みパス(既定: climb重み)")
    ap.add_argument("--learner-arch", default="alakazam",
                    help="ミラー特例判定用のアーキタイプ名(既定alakazam。相手archがこれと一致するとき"
                         "相手にも --policy-weights と同じ重みを使う)")
    ap.add_argument("--baseline", default="seed_planA",
                    help="「現行デッキ」として対現行差を計算する候補名")
    ap.add_argument("--config-base", default=CONFIG,
                    help="両陣営に渡す agent config 名(既定 ml_lethal_attackplan_v0only=従来挙動)。"
                         "提出構成は探索ONの abl_5_full なので、デッキ×探索の相互作用まで測るなら "
                         "採用予定の config 名を指定する")
    args = ap.parse_args()

    out_path = Path(args.results_file)

    if args.analyze:
        rows = [json.loads(l) for l in out_path.open(encoding="utf-8")]
        rows = [r for r in rows if not r.get("failed") and r.get("games", 0) > 0]
        base = next((r for r in rows if r["name"] == args.baseline), None)
        base_wr = base["wins"] / base["games"] if base else 0.5
        # baseline も有限試合の推定値なので、順位付けに使うのは「差」の不確実性。
        #   SE_diff = sqrt(SE_候補^2 + SE_baseline^2)
        # ブロック化(同一seed列・同一相手配分)で両者は正の相関を持つため、この独立仮定は
        # 差のCIを**広め**に見積もる保守側の近似(過小表示にはならない)。
        base_se = math.sqrt(base_wr * (1 - base_wr) / base["games"]) if base else None
        print(f"現行({args.baseline}): {base_wr:.3f}  候補数 {len(rows)}")
        if base is None:
            print(f"[warn] baseline '{args.baseline}' が結果ファイルに無いため、"
                  f"対現行差は 0.5 基準・CIは候補単体のSEのみ(差の不確実性は表示できません)")
        scored = []
        for r in rows:
            wr = r["wins"] / r["games"]
            se = math.sqrt(wr * (1 - wr) / r["games"])
            se_diff = math.sqrt(se * se + base_se * base_se) if base_se is not None else None
            scored.append((wr - base_wr, wr, se, r["name"], se_diff))
        scored.sort(reverse=True)
        if base_se is None:
            print(f"{'順位':<4}{'候補':<22}{'勝率':>8}{'対現行差':>10}{'95%CI半幅':>10}")
            for i, (d, wr, se, nm, _sd) in enumerate(scored[: args.top_n], 1):
                print(f"{i:<4}{nm:<22}{wr * 100:7.1f}%{d * 100:+9.1f}pt{1.96 * se * 100:9.1f}pt")
        else:
            print(f"{'順位':<4}{'候補':<22}{'勝率':>8}{'対現行差':>10}{'候補CI半幅':>11}{'差CI半幅':>11}")
            for i, (d, wr, se, nm, sd) in enumerate(scored[: args.top_n], 1):
                # baseline 自身の行は「自分との差」なので常に 0pt。CI は意味を持たないため '-'。
                dci = "-" if nm == args.baseline else f"{1.96 * sd * 100:.1f}pt"
                print(f"{i:<4}{nm:<22}{wr * 100:7.1f}%{d * 100:+9.1f}pt"
                      f"{1.96 * se * 100:10.1f}pt{dci:>11}")
        top = [nm for _, _, _, nm, _sd in scored[: args.top_n]]
        (Path(args.candidates_dir).parent / "stage1_top.json").write_text(
            json.dumps(top, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n上位{args.top_n}を stage1_top.json に書き出し")
        return

    os.chdir(_ROOT / "sample_submission")
    cand_dir = Path(args.candidates_dir)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["name"] for l in out_path.open(encoding="utf-8")}
        print(f"再開: {len(done)} 件は処理済み")

    cands = sorted(cand_dir.glob("*.csv"))
    if args.max_candidates is not None:
        cands = cands[: args.max_candidates]
    print(f"候補 {len(cands)} 本 × {args.games} 試合 (workers={args.workers}, config={args.config_base})")
    with out_path.open("a", encoding="utf-8") as f:
        for i, p in enumerate(cands, 1):
            name = p.stem
            if name in done:
                continue
            res = run_candidate(name, p, args.games, args.workers, args.seed_base,
                                args.policy_weights, args.learner_arch,
                                log=lambda m: print(f"[{i}/{len(cands)}]{m}", flush=True),
                                config_base=args.config_base)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
    print("Stage1 完了")


if __name__ == "__main__":
    main()
