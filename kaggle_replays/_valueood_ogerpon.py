"""Frozen Value の ogerpon OOD gate(Codex 5.6-sol 設計)。

TES(Turn-End Scorer, `kaggle_replays/_gen_tes_data.py` 等)は葉評価に **Frozen Value**
(`sample_submission/ptcg_ai/learning/value_weights.json`。読み込みは
`ptcg_ai/learning/value_model.ValueModel`、探索への差し込みは
`ptcg_ai/search/leaf_eval.build_evaluator({"kind": "value"})` → `ValueModelEvaluator`)を使う。
この Value は alakazam 分布のリプレイで学習・較正されたもの(オフライン AUC 0.740〜0.753、
`kaggle_replays/value_net/train.py` の学習パイプライン。メモリ参照:
「Hand-aware Value と CARD教師のノイズ」)であり、**ogerpon(mixogerpon)側の局面**へそのまま
転用してよいかは未検証。本スクリプトはその事前判定(OOD gate)専用の計測ハーネス。

やること:
  1. mixogerpon(デッキ `archetype_decks_g2/ogerpon_teal_ex/01.csv`、重み
     `policy_weights_ogerpon_teal_ex_rl_mixogerpon.json`、config `abl_5_full`)を、
     `kaggle_replays/rl/train_league.FIELD_PRESETS["mix"]`(各アーキ模倣重み + abl_5_full)
     と対戦させる。
  2. ogerpon 側の自分の意思決定点(SelectType.MAIN の主選択のみ。サブ選択は対象外)ごとに、
     Frozen Value の予測 P(win)・turn・サイド差・deckCount を記録し、試合終了後に最終勝敗を
     全行へ付与して JSONL(gzip)へ保存する。
  3. `--analyze` で turn帯別(1-4 / 5-10 / 11+ / overall)に AUC・logloss・Brier・較正(10分位
     ECE)・サイド差単変量ロジスティック baseline・試合クラスタブートストラップ AUC 95%CI を出す。

production(`sample_submission/`)・`cg/`・`data/` は一切変更しない(読み取り・import のみ)。
`kaggle_replays/measurement/` の `agents.py`/`runner.py` の作法(`_c0_headroom.py` を参考)を
踏襲する。外部統計ライブラリ(scikit-learn 等)は使わず、AUC/ロジスティック回帰も含めて
標準ライブラリのみで実装する。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    # Windows コンソールの既定コードページ(cp932等)で日本語JSON出力が文字化けするのを防ぐ
    # (kaggle_replays/_diag_valuenet_probe.py と同じ手口)。
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import SelectType  # noqa: E402
from ptcg_ai.board_evaluation.board_features import prize_diff  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402

_WDIR = _SUB / "ptcg_ai" / "learning"
_META_DIR = _ROOT / "kaggle_replays" / "meta_analysis"
_CONFIG_NAME = "abl_5_full"

_OGERPON_DECK = _META_DIR / "archetype_decks_g2" / "ogerpon_teal_ex" / "01.csv"
_OGERPON_WEIGHTS = _WDIR / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"

_GAME_OFFSET = 4_700_000

# 参考値: alakazam 分布で計測済みの Value オフライン AUC レンジ(較正・容量ablation時点)。
# メモリ「Hand-aware Value と CARD教師のノイズ」。ogerpon 側でこのレンジから大きく落ちる場合、
# TES を ogerpon にそのまま転用するのは OOD リスクありと判断する材料になる(閾値判定はしない、
# 参考コメントのみ)。
_REFERENCE_ALAKAZAM_AUC_RANGE = (0.740, 0.753)

ROWS: list[dict] = []
STATS: dict[str, int] = defaultdict(int)
CTX: dict = {"game_id": 0, "opp_arch": "?", "record": False, "step": 0}


def _load_mix_field() -> list[tuple[str, float, str, str]]:
    """`kaggle_replays/rl/train_league.py` の `FIELD_PRESETS["mix"]` をそのまま使う。

    torch 依存で重いため import は呼び出し時まで遅延する(`_c0_headroom.py` の
    `_resolve_field` と同じ手口)。train_league.py 自体は変更しない。
    """
    _rl_dir = _ROOT / "kaggle_replays" / "rl"
    _league_dir = _ROOT / "league"
    for _p in (str(_rl_dir), str(_league_dir)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import train_league  # noqa: E402 - 遅延import

    return list(train_league.FIELD_PRESETS["mix"])


def _pick_opponent(
    field: list[tuple[str, float, str, str]], game_id: int, rng: random.Random,
) -> tuple[str, str, Path]:
    """フィールド定義(アーキ, share, 重み接尾辞, デッキdir)から相手を1体選ぶ。

    `_c0_headroom.py` の `_pick_opponent` と同じ、累積share の重み付き抽出。
    """
    shares = [share for _arch, share, _gen, _deckdir in field]
    total = sum(shares)
    draw = rng.random() * total
    acc = 0.0
    idx = len(field) - 1
    for i, share in enumerate(shares):
        acc += share
        if draw <= acc:
            idx = i
            break
    arch, _share, gen, deckdir = field[idx]
    weights_path = str(_WDIR / f"policy_weights_{arch}{gen}.json")
    deck_path = _META_DIR / deckdir / arch / "01.csv"
    return arch, weights_path, deck_path


def make_probe(inner, evaluator):
    """production agent(ogerpon 側)に、Frozen Value の予測観測を重ねる。

    `inner` の返す行動はそのまま使う(観測は副作用として記録するだけで意思決定には使わない)。
    記録対象は「主選択」(SelectType.MAIN・maxCount==1・進行中局面)のみ(`_c0_headroom.py` の
    pipeline 観測フィルタと同じ考え方: サブ選択の重複は state の変化が薄く、ラベル(勝敗)は
    試合単位で共有されるため相関ノイズを増やすだけになる)。
    """

    def probe(obs):
        action = inner(obs)
        sel = obs.select
        if sel is None or not CTX["record"]:
            return action
        cur = obs.current
        eligible = (
            sel.type == SelectType.MAIN
            and sel.maxCount == 1
            and bool(sel.option)
            and cur is not None
            and cur.result == -1
        )
        if not eligible:
            return action
        me = cur.yourIndex
        try:
            value_pred = float(evaluator.evaluate(cur, me))
            mine = cur.players[me]
            row = {
                "game_id": CTX["game_id"],
                "step": CTX["step"],
                "opp_arch": CTX["opp_arch"],
                "own_index": me,
                "turn": int(getattr(cur, "turn", 0) or 0),
                "value_pred": round(value_pred, 6),
                # 自分残りprize - 相手残りprize(board_features.prize_diff と同一定義。
                # 正=自分の残りサイドの方が多い=取り切りに遠い=不利寄り)。
                "side_diff": int(prize_diff(cur, me)),
                "deck_count": int(getattr(mine, "deckCount", 0) or 0),
            }
        except Exception as exc:  # noqa: BLE001 - 観測失敗は対戦を止めない
            STATS["observe_error_" + type(exc).__name__] += 1
            return action
        CTX["step"] += 1
        ROWS.append(row)
        STATS["recorded"] += 1
        return action

    return probe


def _read_rows(paths: list[str]) -> list[dict]:
    """gzip JSONL または通常 JSONL を複数ファイルから読む。"""
    rows: list[dict] = []
    for name in paths:
        path = Path(name)
        if not path.is_absolute() and not path.exists():
            path = _ROOT / path
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: JSON の解析に失敗しました") from exc
    return rows


def collect(args) -> None:
    """production 対戦(mixogerpon vs mixフィールド)を回し、Value 予測を JSONL(gzip)へ保存する。"""
    num_workers = args.num_workers if args.num_workers is not None else args.workers
    if args.games < 1 or num_workers < 1 or not 0 <= args.worker_id < num_workers:
        raise ValueError("--games は1以上、--worker-id は 0以上 --num-workers(--workers)未満にしてください")
    local_games = max(1, args.games // num_workers)

    # Value ネットが未ロードだと ValueModelEvaluator は常に 0.5 を返し続ける(leaf_eval.py の
    # value_model_ready() docstring 参照)。計測が定数になっている気付きにくい失敗を避けるため、
    # 開始前に必ず確認する。
    if not leaf_eval_module.value_model_ready():
        raise RuntimeError(
            "Frozen Value(value_weights.json)が未ロードです。全予測が0.5に落ちるため測定を中止します。"
        )
    evaluator = leaf_eval_module.build_evaluator({"kind": "value"})

    own_cfg = agents.load_config_copy(_CONFIG_NAME)
    own_cfg["policy_weights_path"] = str(_OGERPON_WEIGHTS)
    own_inner = agents.make_ml_policy_agent(own_cfg)
    probe = make_probe(own_inner, evaluator)
    own_deck = runner.load_deck(_OGERPON_DECK)

    field = _load_mix_field()
    field_rng = random.Random(0xBADA55 ^ args.worker_id)

    started = time.perf_counter()
    for game_index in range(local_games):
        game_id = _GAME_OFFSET + args.worker_id + game_index * num_workers
        arch, opp_weights_path, opp_deck_path = _pick_opponent(field, game_id, field_rng)
        opp_cfg = agents.load_config_copy(_CONFIG_NAME)
        opp_cfg["policy_weights_path"] = opp_weights_path
        opponent = agents.make_ml_policy_agent(opp_cfg)
        opp_deck = runner.load_deck(opp_deck_path)

        own_first = game_index % 2 == 0
        own_index = 0 if own_first else 1
        CTX.update(game_id=game_id, opp_arch=arch, record=True, step=0)
        start_idx = len(ROWS)
        if own_first:
            result = runner.play_game(probe, opponent, own_deck, opp_deck)
        else:
            result = runner.play_game(opponent, probe, opp_deck, own_deck)
        CTX["record"] = False

        own_win = (result.winner == own_index) if result.winner is not None else None
        for row in ROWS[start_idx:]:
            row["own_win"] = own_win
            row["game_error"] = result.error is not None
            row["primary_win_condition"] = result.primary_win_condition
        STATS["games"] += 1
        if result.error is not None:
            STATS["games_error"] += 1
        if own_win is None:
            STATS["games_no_label"] += 1

        print(
            "  [w{}] game#{} opp={} rows_total={} tagged={} own_win={} {:.1f}分".format(
                args.worker_id, game_id, arch, len(ROWS), len(ROWS) - start_idx, own_win,
                (time.perf_counter() - started) / 60.0,
            ),
            file=sys.stderr,
            flush=True,
        )

    output = _HERE / f"_valueood_ogerpon_{args.tag}_w{args.worker_id}.jsonl.gz"
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        for row in ROWS:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "tag": args.tag,
        "worker_id": args.worker_id,
        "num_workers": num_workers,
        "output": str(output),
        "rows": len(ROWS),
        "games": local_games,
        "stats": dict(STATS),
    }, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 分析(外部統計ライブラリ不使用): AUC / logloss / Brier / 較正 / 単変量ロジスティック / bootstrap
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _turn_band(turn: int) -> str:
    if turn <= 4:
        return "1-4"
    if turn <= 10:
        return "5-10"
    return "11+"


def _percentile(values: list[float], percent: float) -> float | None:
    """線形補間 percentile。"""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percent / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _auc(preds: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney U 統計量による AUC(タイは平均順位で処理、外部ライブラリ不使用)。"""
    n = len(preds)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=lambda i: preds[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and preds[order[j + 1]] == preds[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i in range(n) if labels[i] == 1)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(auc, 6)


def _logloss(preds: list[float], labels: list[int], eps: float = 1e-12) -> float | None:
    if not preds:
        return None
    total = 0.0
    for p, y in zip(preds, labels):
        p = min(max(p, eps), 1.0 - eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return round(total / len(preds), 6)


def _brier(preds: list[float], labels: list[int]) -> float | None:
    if not preds:
        return None
    return round(sum((p - y) ** 2 for p, y in zip(preds, labels)) / len(preds), 6)


def _calibration(preds: list[float], labels: list[int], n_bins: int = 10) -> dict:
    """予測を等頻度(quantile)で最大 n_bins 分位ビンへ分け、ビンごとの予測平均 vs 実勝率 と ECE。"""
    n = len(preds)
    if n == 0:
        return {"bins": [], "ece": None}
    order = sorted(range(n), key=lambda i: preds[i])
    k = min(n_bins, n)
    boundaries = [round(i * n / k) for i in range(k + 1)]
    bins = []
    ece = 0.0
    for b in range(k):
        idx = order[boundaries[b]:boundaries[b + 1]]
        if not idx:
            continue
        p_mean = sum(preds[i] for i in idx) / len(idx)
        y_mean = sum(labels[i] for i in idx) / len(idx)
        gap = abs(p_mean - y_mean)
        bins.append({
            "n": len(idx),
            "pred_mean": round(p_mean, 6),
            "actual_win_rate": round(y_mean, 6),
            "abs_gap": round(gap, 6),
        })
        ece += (len(idx) / n) * gap
    return {"bins": bins, "ece": round(ece, 6)}


def _fit_logistic_1d(
    x: list[float], y: list[int], lr: float = 0.3, iters: int = 800,
) -> tuple[float, float, float, float] | None:
    """単変量ロジスティック回帰(全バッチ勾配降下、標準化込み、外部ライブラリ不使用)。

    戻り値 (w, b, mean, std)。予測は sigmoid(w * (v - mean) / std + b)。
    """
    n = len(x)
    if n < 2 or len(set(y)) < 2:
        return None
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    std = math.sqrt(var) if var > 0 else 1.0
    xs = [(v - mean) / std for v in x]
    w, b = 0.0, 0.0
    for _ in range(iters):
        grad_w = 0.0
        grad_b = 0.0
        for xi, yi in zip(xs, y):
            err = _sigmoid(w * xi + b) - yi
            grad_w += err * xi
            grad_b += err
        w -= lr * grad_w / n
        b -= lr * grad_b / n
    return w, b, mean, std


def _baseline_side_diff_probs(side_diff: list[int], labels: list[int]) -> list[float] | None:
    """baseline1: サイド差(自分残り-相手残り)のみの単変量ロジスティック予測確率。"""
    fit = _fit_logistic_1d([float(v) for v in side_diff], labels)
    if fit is None:
        return None
    w, b, mean, std = fit
    return [_sigmoid(w * ((float(v) - mean) / std) + b) for v in side_diff]


def _bootstrap_auc(members: list[dict], seed: int, resamples: int = 1000) -> dict:
    """試合(game_id)をクラスタとして復元抽出し、AUC の95%CIを作る。"""
    by_game: dict = defaultdict(list)
    for row in members:
        by_game[row.get("game_id")].append(row)
    games = list(by_game)
    point = _auc(
        [float(r["value_pred"]) for r in members],
        [1 if r["own_win"] else 0 for r in members],
    )
    if len(games) < 2:
        return {
            "estimate": point, "ci95": [None, None], "resamples": 0,
            "games": len(games), "note": "game数<2のためbootstrap不可(参考値扱い)",
        }
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        picked = [rng.choice(games) for _ in games]
        sampled_rows: list[dict] = []
        for g in picked:
            sampled_rows.extend(by_game[g])
        auc = _auc(
            [float(r["value_pred"]) for r in sampled_rows],
            [1 if r["own_win"] else 0 for r in sampled_rows],
        )
        if auc is not None:
            samples.append(auc)
    lo = _percentile(samples, 2.5)
    hi = _percentile(samples, 97.5)
    return {
        "estimate": point,
        "ci95": [round(lo, 6) if lo is not None else None, round(hi, 6) if hi is not None else None],
        "resamples": len(samples),
        "games": len(games),
    }


def _band_report(members: list[dict]) -> dict:
    n = len(members)
    if n == 0:
        return {"n": 0, "games": 0}
    preds = [float(row["value_pred"]) for row in members]
    labels = [1 if row["own_win"] else 0 for row in members]
    side_diff = [int(row["side_diff"]) for row in members]

    baseline_probs = _baseline_side_diff_probs(side_diff, labels)

    return {
        "n": n,
        "games": len({row.get("game_id") for row in members}),
        "value_net": {
            "auc": _auc(preds, labels),
            "logloss": _logloss(preds, labels),
            "brier": _brier(preds, labels),
            "calibration": _calibration(preds, labels),
        },
        "baseline_side_diff_logistic": {
            "auc": _auc(baseline_probs, labels) if baseline_probs is not None else None,
            "logloss": _logloss(baseline_probs, labels) if baseline_probs is not None else None,
        },
        "value_auc_cluster_bootstrap_ci95": _bootstrap_auc(members, seed=20260813),
    }


def analyze(paths: list[str]) -> None:
    """収集済み JSONL(gzip可・複数ファイル可)を turn帯別に集計し、JSON レポートを標準出力へ出す。"""
    rows = _read_rows(paths)
    valid = [row for row in rows if row.get("own_win") is not None]
    dropped = len(rows) - len(valid)

    bands: dict[str, list[dict]] = {"1-4": [], "5-10": [], "11+": []}
    for row in valid:
        bands[_turn_band(int(row.get("turn", 0)))].append(row)

    report = {
        "files": paths,
        "rows_total": len(rows),
        "rows_dropped_no_label": dropped,
        "games": len({row.get("game_id") for row in valid}),
        "note": "n が小さい(スモーク相当)場合は参考値扱い。閾値判定には使わない。",
        "reference_alakazam_value_auc_range": list(_REFERENCE_ALAKAZAM_AUC_RANGE),
        "reference_note": (
            "alakazam分布でのオフラインValue AUCは0.740〜0.753(較正済み、Hand-aware Value)。"
            "ここから大きく落ちる場合はogerpon側でOOD(=TES流用に懸念あり)の可能性。"
        ),
        "bands": {
            "overall": _band_report(valid),
            "1-4": _band_report(bands["1-4"]),
            "5-10": _band_report(bands["5-10"]),
            "11+": _band_report(bands["11+"]),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen Value(TES用 value_weights.json)の ogerpon OOD gate を測定します。"
    )
    parser.add_argument("--games", type=int, default=200, help="全体の試合数(--num-workersで頭割り)")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="ワーカー総数の既定値(--num-workers省略時にこの値を使う。利便のための別名)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="頭割りに実際に使うワーカー総数(省略時=--workersの値)",
    )
    parser.add_argument("--worker-id", type=int, default=0, help="このワーカーの番号(0始まり)")
    parser.add_argument(
        "--tag", default="run",
        help="出力ファイル識別子(実際のファイル名には worker-id が自動付与される: "
             "_valueood_ogerpon_<tag>_w<worker-id>.jsonl.gz)",
    )
    parser.add_argument(
        "--analyze", nargs="+", metavar="JSONL",
        help="収集せず、指定したJSONL(gzip可・複数ファイル可)を集計する",
    )
    args = parser.parse_args()
    if args.analyze:
        analyze(args.analyze)
    else:
        collect(args)


if __name__ == "__main__":
    main()
