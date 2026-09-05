"""モデル x 相手 の勝率行列を測定し、事前登録された担当表(オラクル振り分け)の成績を評価する。

目的: 「相手のデッキを推定してモデルごと切り替える」方式に効果があるかを判定するための
測定専用スクリプト。振り分けの推論機構(相手アーキタイプ推定器)はまだ作らない。評価では
相手のアーキタイプを我々が指定しているので、「推定が当たった場合」の振り分け性能は
モデル x 相手の勝率行列から直接計算できる(オラクル振り分け)。同時に、専門家が担当外の
相手でどれだけ落ちるかも同じ行列から読める。

**極めて重要**: 列(相手)ごとに事後的に最良のモデルを選んではいけない。必ず --routing で
事前に与えられた担当表に従うこと。最良を選ぶと選択バイアスが入る(実測で 200 試合選抜の
best が再評価で 5〜7pt 下振れした事例がある)。

相手の重みとデッキの定義は kaggle_replays/rl/pools.py の LEARNER_REGISTRY /
build_opponents() をそのまま使う(定義をコピーしない)。学習側(--models)は pools.py の
レジストリに無い任意の重みファイルを指せるようにするため、こちらは basename 指定 +
sample_submission/ptcg_ai/learning/ 直下の存在確認を自前で行う。

各セルは collect_pool.parallel_collect_pool(weights, [その相手1体], deck_l, games_per_cell,
seed, temperature=0.01, workers) を呼ぶだけ。温度は 0.01(ほぼ argmax)固定
(train_pool.py の評価 run_eval_pool と揃える)。

出力は kaggle_replays/value_net_probe/ 配下(測定結果の置き場を揃えるため。--out-prefix に
ディレクトリが含まれていなければ自動でそこに前置する)。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_pool import parallel_collect_pool, SEED_STRIDE  # noqa: E402
import pools  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

OUT_DIR = _ROOT / "kaggle_replays" / "value_net_probe"

TEMPERATURE = 0.01  # train_pool.run_eval_pool と同じ(ほぼ argmax)。


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (両側)。train_v3.wilson_lo と同じ式を上下限に一般化したもの。"""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = max(0.0, (c - m) / d)
    hi = min(1.0, (c + m) / d)
    return lo, hi


def _parse_kv_list(s: str, label: str) -> list[tuple[str, str]]:
    """"a=b,c=d" -> [(a,b),(c,d)]。順序を保持。"""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--{label} は 'key=value' 形式のカンマ区切りでなければならない: {part!r}")
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not v:
            raise ValueError(f"--{label} のキーまたは値が空: {part!r}")
        out.append((k, v))
    return out


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def resolve_models(models_arg: str) -> list[tuple[str, str]]:
    """--models 文字列 -> [(表示名, 重みの絶対パス), ...]。

    重みファイルが sample_submission/ptcg_ai/learning/ に存在しない場合は、黙って
    production 既定にフォールバックせず、利用可能な一覧を添えて ValueError で落とす。
    """
    pairs = _parse_kv_list(models_arg, "models")
    if not pairs:
        raise ValueError(f"--models が空: {models_arg!r}")
    names = [n for n, _ in pairs]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"--models の表示名に重複がある(統計が上書きされる): {dupes}")

    missing = []
    resolved = []
    for name, fname in pairs:
        p = pools.WDIR / fname
        if not p.exists():
            missing.append((name, fname))
        else:
            resolved.append((name, str(p)))
    if missing:
        available = sorted(p.name for p in pools.WDIR.glob("*.json"))
        missing_str = ", ".join(f"{n}={f}" for n, f in missing)
        raise ValueError(
            f"重みファイルが見つからない(存在確認に失敗): {missing_str}\n"
            f"探索先: {pools.WDIR}\n"
            "利用可能なファイル一覧:\n  " + "\n  ".join(available)
        )
    return resolved


def resolve_opponents(names: list[str]):
    """相手名リスト -> pools.build_opponents() の戻り値(name, weights_path_or_None, deck)。

    pools.LEARNER_REGISTRY / build_opponents() をそのまま使う(定義をコピーしない)。
    戻り値の順序は names と一致する。
    """
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"--opponents に重複がある: {dupes}")
    unknown = [n for n in names if n not in pools.LEARNER_REGISTRY]
    if unknown:
        available = ", ".join(sorted(pools.LEARNER_REGISTRY))
        raise ValueError(f"未知の相手名: {unknown}. 利用可能な名前(pools.LEARNER_REGISTRY): {available}")
    return pools.build_opponents(names)


def summarize_cell(stats: dict) -> dict:
    total = stats["total"]
    wins, valid, games, errors = total["wins"], total["valid"], total["games"], total["errors"]
    wr = (wins / valid) if valid else float("nan")
    lo, hi = wilson_interval(wins, valid)
    return {
        "wins": wins, "valid": valid, "games": games, "errors": errors,
        "winrate": wr, "ci95": [lo, hi],
    }


def render_markdown(results: dict) -> str:
    models = results["models"]
    opponents = results["opponents"]
    matrix = results["matrix"]
    routing = results["routing"]
    pool_avg = results["pool_avg_equal_weight"]
    oracle = results["oracle_routing"]
    degradation = results["degradation"]
    args = results["args"]

    lines = []
    lines.append("# eval_matrix 測定結果")
    lines.append("")
    lines.append(
        "モデル x 相手の勝率行列。**すべて等重み平均**(相手4種を単純平均)であり、"
        "実際のメタ分布(alakazam 30%, mega_lucario_ex 13%, archaludon_ex 11%, "
        "crustle 8%, marnie_grimmsnarl_ex 6%)とは異なる点に注意。"
    )
    lines.append("")
    lines.append(
        f"- 学習側デッキ: `{args['learner_deck']}`(全モデル共通)\n"
        f"- games_per_cell: {args['games_per_cell']}\n"
        f"- temperature: {TEMPERATURE}\n"
        f"- workers: {args['workers']}\n"
        f"- seed0: {args['seed0']}\n"
        f"- 所要時間: {results['elapsed_sec']:.0f}s"
    )
    lines.append("")

    lines.append("## 1. 勝率行列(モデル x 相手)")
    lines.append("")
    header = "| モデル | " + " | ".join(opponents) + " | プール平均(等重み) |"
    sep = "|---|" + "---|" * (len(opponents) + 1)
    lines.append(header)
    lines.append(sep)
    for m in models:
        cells = []
        for o in opponents:
            c = matrix[m][o]
            lo, hi = c["ci95"]
            cells.append(f"{c['wins']}/{c['valid']} = {c['winrate']:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {pool_avg[m]:.3f} |")
    lines.append("")

    lines.append("## 2. オラクル振り分け(事前登録の担当表)")
    lines.append("")
    lines.append("担当表(--routing): " + ", ".join(f"{o}→{routing[o]}" for o in opponents))
    lines.append("")
    lines.append("列ごとに事後的に最良モデルを選ぶ選択バイアスを避けるため、担当は事前登録のみを使う。")
    lines.append("")
    header2 = "| 相手 | 担当モデル | 勝数/試合数 | 勝率 |"
    lines.append(header2)
    lines.append("|---|---|---|---|")
    for o in opponents:
        m = routing[o]
        c = matrix[m][o]
        lines.append(f"| {o} | {m} | {c['wins']}/{c['valid']} | {c['winrate']:.3f} |")
    lo, hi = oracle["pooled_ci95"]
    lines.append("")
    lines.append(
        f"- オラクル振り分け 等重み平均勝率: **{oracle['equal_weight_avg']:.3f}**\n"
        f"- オラクル振り分け 合算(試合数等しいため単一二項比率として計算)勝率: "
        f"{oracle['pooled_winrate']:.3f}({oracle['pooled_wins']}/{oracle['pooled_valid']}) "
        f"Wilson95%区間 [{lo:.3f}, {hi:.3f}]"
    )
    lines.append("")

    lines.append("## 3. モデル別プール平均(等重み、参考: 単一モデルで全相手をカバーする場合)")
    lines.append("")
    lines.append("| モデル | プール平均勝率(等重み) |")
    lines.append("|---|---|")
    for m in models:
        lines.append(f"| {m} | {pool_avg[m]:.3f} |")
    lines.append("")

    lines.append("## 4. 担当外での劣化(専門家ごと)")
    lines.append("")
    n_off_target = len(opponents) - 1
    off_label = f"担当外{n_off_target}相手平均勝率" if n_off_target > 0 else "担当外相手平均勝率(該当なし)"
    has_bc = any("bc_own_winrate" in degradation[o] for o in opponents)
    if has_bc:
        header3 = (f"| 相手 | 専門家 | 担当相手での勝率 | {off_label} | 差(担当-担当外) | "
                    "BC 担当相手勝率 | BC 担当外平均勝率 | 専門家-BC(担当) | 専門家-BC(担当外) |")
        lines.append(header3)
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for o in opponents:
            e = degradation[o]
            lines.append(
                f"| {o} | {e['specialist']} | {e['own_winrate']:.3f} | "
                f"{e['off_target_avg_winrate']:.3f} | {e['own_minus_off_target']:.3f} | "
                f"{e['bc_own_winrate']:.3f} | {e['bc_off_target_avg_winrate']:.3f} | "
                f"{e['specialist_minus_bc_own']:+.3f} | {e['specialist_minus_bc_off_target']:+.3f} |"
            )
    else:
        header3 = f"| 相手 | 専門家 | 担当相手での勝率 | {off_label} | 差(担当-担当外) |"
        lines.append(header3)
        lines.append("|---|---|---|---|---|")
        for o in opponents:
            e = degradation[o]
            lines.append(
                f"| {o} | {e['specialist']} | {e['own_winrate']:.3f} | "
                f"{e['off_target_avg_winrate']:.3f} | {e['own_minus_off_target']:.3f} |"
            )
    lines.append("")
    lines.append(
        "「有意差がある」は区間が重ならない場合のみ主張できる。上の表の値には区間を付けていないので、"
        "優劣の主張には第1節の行列の Wilson95%区間を参照すること。"
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True,
                     help="'表示名=重みファイル名' のカンマ区切り。重みファイル名は "
                          "sample_submission/ptcg_ai/learning/ 配下の basename。"
                          "例: BC=policy_weights.json,SPEC_marnie=policy_weights_alakazam_pool_spec_marnie.json")
    ap.add_argument("--learner-deck", default="alakazam",
                     help="学習側デッキのアーキタイプ名(pools.LEARNER_REGISTRY のキー)。既定 alakazam。"
                          "全モデルが同じデッキの方策である前提のため共通指定。")
    ap.add_argument("--opponents", default="alakazam,crustle,marnie_grimmsnarl_ex,archaludon_ex",
                     help="カンマ区切りの相手名(pools.LEARNER_REGISTRY のキー)。")
    ap.add_argument("--games-per-cell", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=500_000)
    ap.add_argument("--routing", required=True,
                     help="'相手=モデル表示名' のカンマ区切り(事前登録の担当表)。--opponents の"
                          "全相手に対して過不足なく指定する必要がある。")
    ap.add_argument("--out-prefix", default="eval_matrix",
                     help="出力ファイルの prefix。ディレクトリを含まなければ "
                          "kaggle_replays/value_net_probe/ を自動で前置する。")
    args = ap.parse_args()

    models = resolve_models(args.models)
    model_names = [m for m, _ in models]

    opponent_names = _parse_csv_list(args.opponents)
    if not opponent_names:
        raise ValueError(f"--opponents が空: {args.opponents!r}")
    opponents_built = resolve_opponents(opponent_names)
    assert [o[0] for o in opponents_built] == opponent_names

    _weights_path, learner_deck_csv = pools.resolve_learner(args.learner_deck)
    deck_l = read_deck_csv_file(learner_deck_csv)

    routing_pairs = _parse_kv_list(args.routing, "routing")
    routing_opp_names = [o for o, _ in routing_pairs]
    if len(set(routing_opp_names)) != len(routing_opp_names):
        dupes = sorted({o for o in routing_opp_names if routing_opp_names.count(o) > 1})
        raise ValueError(f"--routing に相手名の重複がある: {dupes}")
    routing = dict(routing_pairs)
    for opp, model in routing.items():
        if opp not in opponent_names:
            raise ValueError(f"--routing の相手 {opp!r} が --opponents に無い: {opponent_names}")
        if model not in model_names:
            raise ValueError(f"--routing のモデル {model!r} が --models に無い: {model_names}")
    missing_routing = [o for o in opponent_names if o not in routing]
    if missing_routing:
        raise ValueError(f"--routing に相手 {missing_routing} の担当が指定されていない(全相手に必須)")

    out_prefix = args.out_prefix
    if "/" not in out_prefix and "\\" not in out_prefix:
        out_prefix = str(OUT_DIR / out_prefix)
    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"models={model_names}", flush=True)
    print(f"opponents={opponent_names}", flush=True)
    print(f"learner_deck={args.learner_deck} deck_csv={learner_deck_csv}", flush=True)
    print(f"games_per_cell={args.games_per_cell} workers={args.workers} temperature={TEMPERATURE} "
          f"seed0={args.seed0}", flush=True)
    print(f"routing={routing}", flush=True)

    n_models = len(models)
    n_opp = len(opponent_names)
    # セル(i, j) の seed 間隔。各セルは k=1(相手1体)で parallel_collect_pool を呼ぶので、
    # collect_pool.build_tasks が使う seed 空間は [seed_cell, seed_cell + games_per_cell) に
    # 収まる(games_per_cell < SEED_STRIDE=100000 が前提。実際に既定400・スモーク16は十分小さい)。
    # SEED_STRIDE * 2 間隔を空ければ、セル間で games_per_cell が SEED_STRIDE に近い極端な値でも
    # 衝突しない安全マージンになる。
    cell_seed_gap = SEED_STRIDE * 2

    matrix: dict[str, dict[str, dict]] = {}
    t0 = time.time()
    total_cells = n_models * n_opp
    cell_i = 0
    for i, (model_name, weights_path) in enumerate(models):
        matrix[model_name] = {}
        for j, opp_name in enumerate(opponent_names):
            seed = args.seed0 + (i * n_opp + j) * cell_seed_gap
            single_opp = [opponents_built[j]]
            t_cell = time.time()
            _, stats = parallel_collect_pool(
                weights_path, single_opp, deck_l,
                n_games=args.games_per_cell, seed0=seed,
                temperature=TEMPERATURE, workers=args.workers,
            )
            cell = summarize_cell(stats)
            cell["seed0"] = seed
            matrix[model_name][opp_name] = cell
            cell_i += 1
            lo, hi = cell["ci95"]
            print(f"[{cell_i}/{total_cells}] {model_name} vs {opp_name}: "
                  f"{cell['wins']}/{cell['valid']} = {cell['winrate']:.3f} "
                  f"CI[{lo:.3f},{hi:.3f}] err{cell['errors']} seed0={seed} "
                  f"({time.time() - t_cell:.0f}s)", flush=True)

    # 2. プール平均(等重み)
    pool_avg = {m: sum(matrix[m][o]["winrate"] for o in opponent_names) / n_opp for m in model_names}

    # 3. オラクル振り分け
    oracle_wins = oracle_valid = 0
    oracle_per_opp_wr = {}
    for opp_name in opponent_names:
        model_name = routing[opp_name]
        c = matrix[model_name][opp_name]
        oracle_wins += c["wins"]
        oracle_valid += c["valid"]
        oracle_per_opp_wr[opp_name] = c["winrate"]
    oracle_equal_weight_avg = sum(oracle_per_opp_wr.values()) / n_opp
    oracle_pooled_wr = (oracle_wins / oracle_valid) if oracle_valid else float("nan")
    oracle_lo, oracle_hi = wilson_interval(oracle_wins, oracle_valid)

    # 4. 担当外での劣化
    bc_name = "BC" if "BC" in matrix else None
    degradation = {}
    for opp_name, model_name in routing.items():
        others = [o for o in opponent_names if o != opp_name]
        own_wr = matrix[model_name][opp_name]["winrate"]
        other_wr = (sum(matrix[model_name][o]["winrate"] for o in others) / len(others)
                    if others else float("nan"))
        entry = {
            "specialist": model_name, "own_opponent": opp_name,
            "own_winrate": own_wr, "off_target_avg_winrate": other_wr,
            "own_minus_off_target": own_wr - other_wr if others else float("nan"),
        }
        if bc_name is not None:
            bc_own = matrix[bc_name][opp_name]["winrate"]
            bc_other = (sum(matrix[bc_name][o]["winrate"] for o in others) / len(others)
                        if others else float("nan"))
            entry["bc_own_winrate"] = bc_own
            entry["bc_off_target_avg_winrate"] = bc_other
            entry["specialist_minus_bc_own"] = own_wr - bc_own
            entry["specialist_minus_bc_off_target"] = other_wr - bc_other
        degradation[opp_name] = entry

    elapsed = time.time() - t0
    print(f"total elapsed {elapsed:.0f}s", flush=True)

    results = {
        "args": vars(args),
        "models": model_names,
        "opponents": opponent_names,
        "routing": routing,
        "matrix": matrix,
        "pool_avg_equal_weight": pool_avg,
        "oracle_routing": {
            "per_opponent_winrate": oracle_per_opp_wr,
            "equal_weight_avg": oracle_equal_weight_avg,
            "pooled_wins": oracle_wins, "pooled_valid": oracle_valid,
            "pooled_winrate": oracle_pooled_wr, "pooled_ci95": [oracle_lo, oracle_hi],
        },
        "degradation": degradation,
        "elapsed_sec": elapsed,
    }

    json_path = Path(f"{out_prefix}_results.json")
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = Path(f"{out_prefix}.md")
    md_path.write_text(render_markdown(results), encoding="utf-8")

    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
