"""value_variation.py

序盤の critic(価値関数)予測 Phi(s) が「盤面の質」を読んでいるのか、それとも
「どのデッキ同士の対戦か(マッチアップ)」という試合を通じて不変な情報を読んで
いるだけなのかを、試合内変動(within-game variation)の大きさで検証するスクリプト。

マッチアップは1試合の中で変化しないため、もし critic がマッチアップだけを見て
いるなら、序盤の各決定に対する予測値はほぼ一定になり、advantage shaping 項
gamma * Phi(s') - Phi(s) はゼロに近づき、どの手が良かったかを区別できなくなる。
これを直接測るために、
  (A) 試合内分散 / 試合間分散の分解 (ICC)
  (B) 同一試合内で隣り合う決定間の Phi の差分
をターン帯ごとに計算する。

このスクリプトは既存の npz (自己対戦ログから抽出した特徴量) を読むだけで、
試合シミュレーションは一切行わない。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

BANDS = [("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, None)]


def band_of(turn: np.ndarray) -> np.ndarray:
    """turn (int array) -> band ラベル文字列の配列。"""
    labels = np.empty(turn.shape[0], dtype=object)
    for name, lo, hi in BANDS:
        if hi is None:
            mask = turn >= lo
        else:
            mask = (turn >= lo) & (turn <= hi)
        labels[mask] = name
    return labels


def check_turn_order(turn: np.ndarray, game_id: np.ndarray) -> tuple[int, list[int]]:
    """各 game_id 内で turn が非減少であることを検証する。

    行の順序 = 試合内の決定の順序という前提(selfplay_positions.py が順に
    append している)に依存しているため、これが崩れていたら以降の連続差分
    解析(隣接行の差分)は意味を持たない。ここでは並べ替えは一切行わない。
    """
    bad_game_ids: list[int] = []
    n = len(game_id)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and game_id[j + 1] == game_id[i]:
            j += 1
        seg = turn[i : j + 1]
        if seg.shape[0] >= 2 and np.any(np.diff(seg) < 0):
            bad_game_ids.append(int(game_id[i]))
        i = j + 1
    return len(bad_game_ids), bad_game_ids


def oof_predict_gbm(X: np.ndarray, y: np.ndarray, game_id: np.ndarray,
                     n_splits: int, seed: int) -> np.ndarray:
    """GroupKFold で out-of-fold 予測を作る。

    GBM (HistGradientBoostingClassifier) を使うのは、本番の critic(MLP)
    より表現力が高い可能性が高く、この強いモデルですら試合内変動が
    出なければ、それは「達成可能な試合内変動の上界が小さい」ことの
    決定的な証拠になるから。逆に GBM で試合内変動が出れば、少なくとも
    166特徴から序盤の局面差を読み取ること自体は可能だと分かる。
    """
    oof = np.full(y.shape[0], np.nan, dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(X, y, groups=game_id):
        model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    assert not np.any(np.isnan(oof)), "oof に埋まっていない行がある"
    return oof


def oof_predict_matchup_onehot(matchup: np.ndarray, y: np.ndarray,
                                game_id: np.ndarray, n_splits: int,
                                seed: int) -> np.ndarray:
    """マッチアップ one-hot だけを特徴にした LogisticRegression の OOF 予測。

    同一試合内では matchup は不変なので、この予測値も試合内では定数になる
    はず(=試合内分散・連続差分は定義上ゼロ)。これは実装の検算用の参照値。
    GroupKFold の分割は oof_predict_gbm と同一のグループ配列・n_splits・
    (シャッフルなしの)決定的なアルゴリズムから作られるため、両モデルで
    fold の切り方は一致する。
    """
    oof = np.full(y.shape[0], np.nan, dtype=np.float64)
    matchup_2d = matchup.reshape(-1, 1)
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(matchup_2d, y, groups=game_id):
        enc = OneHotEncoder(handle_unknown="ignore")
        X_train = enc.fit_transform(matchup_2d[train_idx])
        X_test = enc.transform(matchup_2d[test_idx])
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(X_train, y[train_idx])
        oof[test_idx] = model.predict_proba(X_test)[:, 1]
    assert not np.any(np.isnan(oof)), "oof に埋まっていない行がある"
    return oof


def variance_decomposition(phi: np.ndarray, band_label: np.ndarray,
                            game_id: np.ndarray) -> dict:
    """(A) ターン帯ごとの between / within / total / ICC を計算する。

    between: その帯における「試合ごとの Phi の平均」の分散(試合を単位とする分散)
    within : その帯における「試合ごとの Phi の分散」の平均
             (その帯に2行以上ある試合だけを対象にする)
    分散は numpy 既定の母分散(ddof=0)を用いる。
    """
    result = {}
    for name, _lo, _hi in BANDS:
        mask = band_label == name
        gids = game_id[mask]
        vals = phi[mask]
        if gids.shape[0] == 0:
            result[name] = {
                "n_rows": 0, "n_games_between": 0, "n_games_within": 0,
                "between": None, "within": None, "total": None, "icc": None,
            }
            continue

        uniq_games, inverse = np.unique(gids, return_inverse=True)
        # 試合ごとの平均 (between 用)
        sums = np.bincount(inverse, weights=vals, minlength=uniq_games.shape[0])
        counts = np.bincount(inverse, minlength=uniq_games.shape[0])
        means = sums / counts
        between = float(np.var(means, ddof=0)) if uniq_games.shape[0] >= 1 else None
        n_games_between = int(uniq_games.shape[0])

        # 試合ごとの分散 (within 用、2行以上の試合のみ)
        within_vars = []
        for g_idx in range(uniq_games.shape[0]):
            if counts[g_idx] >= 2:
                game_vals = vals[inverse == g_idx]
                within_vars.append(np.var(game_vals, ddof=0))
        n_games_within = len(within_vars)
        within = float(np.mean(within_vars)) if n_games_within > 0 else None

        if between is not None and within is not None:
            total = between + within
            icc = between / total if total > 0 else None
        else:
            total = None
            icc = None

        result[name] = {
            "n_rows": int(gids.shape[0]),
            "n_games_between": n_games_between,
            "n_games_within": n_games_within,
            "between": between,
            "within": within,
            "total": total,
            "icc": icc,
        }
    return result


def consecutive_diff(phi: np.ndarray, turn: np.ndarray,
                      game_id: np.ndarray) -> dict:
    """(B) 同一 game_id 内で隣り合う行の Phi(次) - Phi(今) を帯ごとに集計する。

    帯の判定は「今」(diff の前側の行)の turn で行う。game_id が変わる境界
    (試合をまたぐ差分)は除外する。
    """
    same_game = game_id[1:] == game_id[:-1]
    delta = phi[1:] - phi[:-1]
    delta = delta[same_game]
    turn_now = turn[:-1][same_game]
    band_now = band_of(turn_now)

    # 帯自体の Phi の標準偏差 (Δ の大きさと比較する参考値)
    band_all = band_of(turn)

    result = {}
    for name, _lo, _hi in BANDS:
        m = band_now == name
        d = delta[m]
        n_pairs = int(d.shape[0])
        if n_pairs > 0:
            abs_d = np.abs(d)
            entry = {
                "n_pairs": n_pairs,
                "mean_abs_delta": float(np.mean(abs_d)),
                "sd_delta": float(np.std(d, ddof=0)),
                "median_abs_delta": float(np.median(abs_d)),
                "p90_abs_delta": float(np.percentile(abs_d, 90)),
            }
        else:
            entry = {
                "n_pairs": 0,
                "mean_abs_delta": None,
                "sd_delta": None,
                "median_abs_delta": None,
                "p90_abs_delta": None,
            }
        phi_band = phi[band_all == name]
        entry["phi_sd_in_band"] = float(np.std(phi_band, ddof=0)) if phi_band.shape[0] > 0 else None
        entry["n_rows_in_band"] = int(phi_band.shape[0])
        result[name] = entry
    return result


def render_markdown(npz_path: str, n_rows_total: int, n_rows_used: int,
                     n_games: int, n_splits: int, seed: int,
                     order_bad_count: int, order_bad_ids: list[int],
                     var_decomp: dict, cons_diff: dict,
                     skip_consecutive: bool, sanity: dict) -> str:
    lines = []
    lines.append(f"# value_variation 結果: `{npz_path}`\n")
    lines.append(f"- 全行数: {n_rows_total}")
    lines.append(f"- 使用行数 (turn >= 1): {n_rows_used}")
    lines.append(f"- 試合数: {n_games}")
    lines.append(f"- GroupKFold n_splits: {n_splits}, seed: {seed}\n")

    lines.append("## turn 非減少チェック\n")
    if order_bad_count == 0:
        lines.append("すべての game_id で turn は非減少。順序は健全。\n")
    else:
        lines.append(f"**{order_bad_count} 件の game_id で turn の減少を検出。** "
                      "行の順序が崩れている可能性があるため、(B) 連続差分解析は行っていない。")
        shown = order_bad_ids[:20]
        lines.append(f"該当 game_id (先頭最大20件): {shown}\n")

    lines.append("## (A) 分散分解 (between / within / total / ICC)\n")
    for model_key, model_name in [("gbm", "GBM (166特徴)"), ("matchup_onehot", "マッチアップ one-hot (参照)")]:
        lines.append(f"### {model_name}\n")
        lines.append("| 帯 | n_rows | n_games(between) | n_games(within) | between | within | total | ICC |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, _lo, _hi in BANDS:
            e = var_decomp[model_key][name]
            def fmt(x):
                return f"{x:.6f}" if x is not None else "-"
            lines.append(
                f"| {name} | {e['n_rows']} | {e['n_games_between']} | {e['n_games_within']} | "
                f"{fmt(e['between'])} | {fmt(e['within'])} | {fmt(e['total'])} | {fmt(e['icc'])} |"
            )
        lines.append("")

    lines.append("## (B) 連続する決定間の差分\n")
    if skip_consecutive:
        lines.append("turn 非減少チェックに失敗したため、この解析はスキップした。\n")
    else:
        for model_key, model_name in [("gbm", "GBM (166特徴)"), ("matchup_onehot", "マッチアップ one-hot (参照)")]:
            lines.append(f"### {model_name}\n")
            lines.append("| 帯 | n_pairs | mean(|Δ|) | sd(Δ) | median(|Δ|) | p90(|Δ|) | Φ自体のsd(帯内, n_rows) |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for name, _lo, _hi in BANDS:
                e = cons_diff[model_key][name]
                def fmt(x):
                    return f"{x:.6f}" if x is not None else "-"
                phi_sd_str = f"{fmt(e['phi_sd_in_band'])} (n={e['n_rows_in_band']})"
                lines.append(
                    f"| {name} | {e['n_pairs']} | {fmt(e['mean_abs_delta'])} | {fmt(e['sd_delta'])} | "
                    f"{fmt(e['median_abs_delta'])} | {fmt(e['p90_abs_delta'])} | {phi_sd_str} |"
                )
            lines.append("")

    lines.append("## 検算: マッチアップ one-hot の within / mean(|Δ|) はゼロか\n")
    lines.append(f"- within が全帯でゼロ: {sanity['within_all_zero']}")
    if skip_consecutive:
        lines.append("- mean(|Δ|): (B) をスキップしたため未評価")
    else:
        lines.append(f"- mean(|Δ|) が全帯でゼロ: {sanity['mean_abs_delta_all_zero']}")
    if not sanity["within_all_zero"] or (not skip_consecutive and not sanity["mean_abs_delta_all_zero"]):
        lines.append("\n**警告: マッチアップ one-hot の試合内変動がゼロになっていない。実装のバグの可能性があるため確認が必要。**\n")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out-prefix", default="value_variation")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X_all = data["X"]
    y_all = data["y"].astype(np.int64)
    turn_all = data["turn"].astype(np.int64)
    game_id_all = data["game_id"].astype(np.int64)
    matchup_all = data["matchup"]

    n_rows_total = X_all.shape[0]

    # turn 非減少チェック(全データ、フィルタ前)。並べ替えは一切行わない。
    order_bad_count, order_bad_ids = check_turn_order(turn_all, game_id_all)
    skip_consecutive = order_bad_count > 0

    # 手順1: turn >= 1 の行だけ使う
    mask = turn_all >= 1
    X = X_all[mask]
    y = y_all[mask]
    turn = turn_all[mask]
    game_id = game_id_all[mask]
    matchup = matchup_all[mask]

    n_rows_used = X.shape[0]
    n_games = int(np.unique(game_id).shape[0])

    print(f"[{args.npz}] 全行数={n_rows_total} 使用行数(turn>=1)={n_rows_used} 試合数={n_games}")
    print(f"turn非減少チェック: 違反game_id数={order_bad_count}")
    if skip_consecutive:
        print(f"  違反game_id (先頭20件): {order_bad_ids[:20]}")
        print("  -> (B) 連続差分解析はスキップします。")

    # 手順2: GBM の OOF 予測 (joint, 全ターン込みで学習)
    print("GBM の GroupKFold OOF 予測を計算中...")
    phi_gbm = oof_predict_gbm(X, y, game_id, args.n_splits, args.seed)

    # 手順3: マッチアップ one-hot の OOF 予測 (参照)
    print("マッチアップ one-hot LogisticRegression の GroupKFold OOF 予測を計算中...")
    phi_matchup = oof_predict_matchup_onehot(matchup, y, game_id, args.n_splits, args.seed)

    band_label = band_of(turn)

    var_decomp = {
        "gbm": variance_decomposition(phi_gbm, band_label, game_id),
        "matchup_onehot": variance_decomposition(phi_matchup, band_label, game_id),
    }

    if skip_consecutive:
        cons_diff = {"gbm": {}, "matchup_onehot": {}}
    else:
        cons_diff = {
            "gbm": consecutive_diff(phi_gbm, turn, game_id),
            "matchup_onehot": consecutive_diff(phi_matchup, turn, game_id),
        }

    # 検算: マッチアップ one-hot の within / mean(|Δ|) はゼロか
    EPS = 1e-9
    within_vals = [e["within"] for e in var_decomp["matchup_onehot"].values() if e["within"] is not None]
    within_all_zero = all(abs(v) < EPS for v in within_vals) if within_vals else True
    if skip_consecutive:
        mean_abs_delta_all_zero = None
    else:
        mad_vals = [e["mean_abs_delta"] for e in cons_diff["matchup_onehot"].values() if e["mean_abs_delta"] is not None]
        mean_abs_delta_all_zero = all(v < EPS for v in mad_vals) if mad_vals else True

    sanity = {
        "within_all_zero": within_all_zero,
        "mean_abs_delta_all_zero": mean_abs_delta_all_zero,
    }

    print(f"検算: matchup_onehot within が全帯ゼロ = {within_all_zero}")
    if not skip_consecutive:
        print(f"検算: matchup_onehot mean(|Δ|) が全帯ゼロ = {mean_abs_delta_all_zero}")
        if not within_all_zero or not mean_abs_delta_all_zero:
            print("  警告: ゼロになっていない箇所がある。実装バグの可能性。")
    elif not within_all_zero:
        print("  警告: ゼロになっていない箇所がある。実装バグの可能性。")

    results = {
        "npz": args.npz,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "n_rows_total": n_rows_total,
        "n_rows_used": n_rows_used,
        "n_games": n_games,
        "order_check": {
            "n_bad_games": order_bad_count,
            "bad_game_ids": order_bad_ids,
        },
        "skip_consecutive": skip_consecutive,
        "variance_decomposition": var_decomp,
        "consecutive_diff": cons_diff,
        "sanity_check": sanity,
    }

    out_json = Path(f"{args.out_prefix}_results.json")
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    md = render_markdown(
        args.npz, n_rows_total, n_rows_used, n_games, args.n_splits, args.seed,
        order_bad_count, order_bad_ids, var_decomp, cons_diff, skip_consecutive, sanity,
    )
    out_md = Path(f"{args.out_prefix}.md")
    out_md.write_text(md, encoding="utf-8")

    print(f"書き出し: {out_json}")
    print(f"書き出し: {out_md}")


if __name__ == "__main__":
    main()
