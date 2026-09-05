"""ドラパルト操縦アルゴリズムの ablation 測定(ローカル / Kaggle Notebook 共通の本体)。

問い: ダメカン配置プランナ(``ptcg_ai/search/damage_counter_plan.py``)を挟むと、
kiyotah 版のヒューリスティック操縦より強くなるのか。

群(``--groups``):
  - ``rule``     : ``opponents/dragapult_rule_agent.py``(対照群・凍結。プランナ無し)
  - ``plan``     : ``opponents/dragapult_planning_agent.py`` + ``DRAGAPULT_PLANNER_ENABLED=1``
  - ``plan_off`` : 同モジュール + ``DRAGAPULT_PLANNER_ENABLED=0``(= rule と同一挙動になるはず。
                   パイプラインの健全性チェック用。既定では走らせない)

相手フィールドは ``kaggle_replays/_drapa_ceiling.py`` と同一の mix 11アーキ(シェア比例配分)。
相手は各アーキの模倣重み + config ``abl_5_full``。**群を替えても seed・デッキ・配分・先後の
割り当ては完全に同一**にしてあるので、差分は操縦アルゴリズムだけになる。

出力は JSONL(1試合1行)を逐次 flush しながら書く。途中でカーネルが時間切れになっても
そこまでの試合は残り、``--merge``(push_kernel.py)で集計できる。

使い方(ローカル):
    python kaggle_replays/kaggle_run/drapa_ablation.py --games 20 --workers 2 \
        --out C:/tmp/drapa_smoke.jsonl

使い方(Kaggle カーネル内。push_kernel.py が生成するスクリプトが実行する):
    python <repo>/kaggle_replays/kaggle_run/drapa_ablation.py \
        --repo-root <repo> --games 400 --workers 4 --seed-base 970000 \
        --out /kaggle/working/drapa_s0.jsonl

注意(既知の罠): multiprocessing は **spawn 前提**で書いてある。ヒアドキュメントや
`python -c` で実行するとワーカーが ``__main__`` を再 import できず BrokenProcessPool に
なるので、必ずこのファイルをスクリプトとして実行すること(``if __name__ == "__main__"`` 必須)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# kaggle_run -> kaggle_replays -> リポジトリルート
_DEFAULT_ROOT = _HERE.parent.parent

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - 端末によっては失敗するが致命的ではない
        pass

# ---------------------------------------------------------------------------
# 実験定義(_drapa_ceiling.py と同一のフィールド)
# ---------------------------------------------------------------------------

CONFIG = "abl_5_full"

# train_league.FIELD_MIX と同一の11アーキ + シェア(stage1_screen.py / _drapa_ceiling.py と同じ)。
FIELD: list[tuple[str, int, str]] = [
    ("marnie_grimmsnarl_ex", 1082, "_g2"), ("alakazam", 891, "_g2"),
    ("mega_lucario_ex", 403, ""), ("dragapult_ex", 381, "_g2"),
    ("mega_froslass_ex", 353, "_g2"), ("ogerpon_teal_ex", 310, "_g2"),
    ("archaludon_ex", 287, "_g2"), ("crustle", 287, "_g2"),
    ("shirona_garchomp_ex", 158, "_g2"), ("omatsuri_ondo", 138, "_g2"),
    ("rocket_mewtwo_ex", 112, "_g2"),
]
_DECKDIR_OF = {"": "archetype_decks", "_g2": "archetype_decks_g2"}

# 群名 -> (エージェントのモジュール, 環境変数)。環境変数はワーカーが**モジュール import 前**に
# 設定する(dragapult_planning_agent は import 時に PLANNER_ENABLED を確定させるため)。
GROUPS: dict[str, dict] = {
    "rule": {"module": "opponents.dragapult_rule_agent", "env": {}},
    "plan": {"module": "opponents.dragapult_planning_agent",
             "env": {"DRAGAPULT_PLANNER_ENABLED": "1"}},
    "plan_off": {"module": "opponents.dragapult_planning_agent",
                 "env": {"DRAGAPULT_PLANNER_ENABLED": "0"}},
}


def field_entries(only: set[str] | None = None) -> list[tuple[str, int, str]]:
    """測定対象アーキの一覧。``only`` はデバッグ用の絞り込み(本測定では使わない)。"""
    if not only:
        return list(FIELD)
    entries = [e for e in FIELD if e[0] in only]
    unknown = only - {e[0] for e in FIELD}
    if unknown:
        raise SystemExit(f"unknown archetype(s): {sorted(unknown)}")
    return entries


def allocate(total: int, only: set[str] | None = None) -> list[tuple[str, str, str, int]]:
    """総試合数をシェア比例で各アーキに配分する。

    先手/後手を必ず半々にするため、各アーキの試合数は偶数(最低2)に丸める。
    戻り値は ``(arch, gen, deckdir, games)``。
    """
    entries = field_entries(only)
    tot = sum(s for _, s, _ in entries)
    out: list[tuple[str, str, str, int]] = []
    for arch, share, gen in entries:
        n = int(round(total * share / tot))
        n -= n % 2
        out.append((arch, gen, _DECKDIR_OF[gen], max(2, n)))
    return out


def build_tasks(games: int, seed_base: int, only: set[str] | None = None) -> list[tuple[str, int, int]]:
    """``(arch, index, seed)`` のタスク列。群をまたいで完全に同一の列になる。

    seed は「アーキごとに1万ずつずらした基点 + 通し番号」。群を替えても同じ seed 列に
    なるので、対応のある比較(McNemar)ができる。
    """
    tasks: list[tuple[str, int, int]] = []
    for arch_i, (arch, _gen, _dd, n) in enumerate(allocate(games, only)):
        for i in range(n):
            tasks.append((arch, i, seed_base + arch_i * 10_000 + i))
    return tasks


# ---------------------------------------------------------------------------
# 統計ユーティリティ
# ---------------------------------------------------------------------------

def wilson(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 区間(既定 95%)。n=0 は [0,1]。"""
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def _norm_sf(z: float) -> float:
    """標準正規の上側確率 P(Z >= z)。"""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_proportion_z(w1: int, n1: int, w2: int, n2: int) -> dict:
    """2標本比率の片側 z 検定(H1: p1 > p2)。プールした標準誤差を使う。"""
    if n1 <= 0 or n2 <= 0:
        return {"z": None, "p_one_sided": None}
    p1, p2 = w1 / n1, w2 / n2
    pool = (w1 + w2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0.0:
        return {"z": None, "p_one_sided": None}
    z = (p1 - p2) / se
    return {"z": z, "p_one_sided": _norm_sf(z)}


def fisher_exact_greater(w1: int, n1: int, w2: int, n2: int) -> float | None:
    """Fisher 正確検定の片側 p 値(H1: 群1の勝率 > 群2の勝率)。

    2x2 表 [[w1, n1-w1], [w2, n2-w2]] を周辺度数固定の超幾何分布として、
    P(X >= w1) を厳密に足し上げる(n が数百程度なので実用上十分速い)。
    """
    if n1 <= 0 or n2 <= 0:
        return None
    total = n1 + n2
    wins = w1 + w2
    hi = min(n1, wins)  # 群1が取りうる勝ち数の上限(周辺度数固定)
    denom = math.comb(total, wins)
    if denom == 0:
        return None
    num = sum(math.comb(n1, k) * math.comb(n2, wins - k) for k in range(w1, hi + 1))
    return num / denom


def mcnemar_exact_greater(b: int, c: int) -> float | None:
    """McNemar の片側 exact 検定。b=「群1だけ勝ち」, c=「群2だけ勝ち」の件数。

    seed/デッキ/先後をそろえてあるため対応のある比較ができる(ただし cg エンジンの
    シャッフルは開幕以降 seed 制御外なので **完全な CRN ではない**。参考値として扱う)。
    """
    n = b + c
    if n == 0:
        return None
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def quantile(sorted_vals: list[float], q: float) -> float | None:
    """線形補間つき分位点(sorted_vals は昇順であること)。"""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# ワーカー(spawn 前提。重い import はすべてこの中で行う)
# ---------------------------------------------------------------------------

_W: dict = {}


def _worker_init(root_str: str, group: str, horizon: int, config: str,
                 self_deck_path: str, opp_deck_paths: dict, opp_weight_paths: dict) -> None:
    """各ワーカープロセスで1回だけ呼ばれ、パス・環境変数・エージェント・デッキを準備する。"""
    root = Path(root_str)

    # 1) 環境変数は **エージェント module の import より前** に確定させる
    #    (dragapult_planning_agent は import 時に PLANNER_ENABLED を評価するため)。
    for key, value in GROUPS[group]["env"].items():
        os.environ[key] = value
    if group != "rule":
        os.environ["DRAGAPULT_PLANNER_HORIZON"] = str(horizon)

    # 2) sys.path: リポジトリルート(opponents/)・league/・sample_submission/(cg, ptcg_ai)。
    for cand in (str(root), str(root / "league"), str(root / "sample_submission")):
        if cand not in sys.path:
            sys.path.insert(0, cand)

    # 3) ml_policy / rule_based は cwd 相対で deck.csv 等を読むので cwd をそろえる
    #    (spawn された子プロセスは親の cwd を継承しない)。
    os.chdir(root / "sample_submission")

    import importlib

    import run_league  # noqa: PLC0415 - ワーカー内で遅延 import する

    agent_module = importlib.import_module(GROUPS[group]["module"])

    _W["group"] = group
    _W["config"] = config
    _W["run_league"] = run_league
    _W["self_agent"] = agent_module.agent
    _W["self_deck"] = run_league.read_deck_csv_file(self_deck_path)
    _W["opp_decks"] = {a: run_league.read_deck_csv_file(p) for a, p in opp_deck_paths.items()}
    _W["opp_weights"] = dict(opp_weight_paths)
    _W["opp_agents"] = {}


def _opponent_agent(arch: str):
    """相手(ml_policy + アーキタイプ別模倣重み + abl_5_full)をワーカー内でキャッシュ生成。"""
    cache = _W["opp_agents"]
    if arch not in cache:
        cache[arch] = _W["run_league"].build_agent("ml_policy", _W["opp_weights"][arch], _W["config"])
    return cache[arch]


def _worker_play(task: tuple[str, int, int]) -> dict:
    """1試合を実行して JSON に書けるレコードを返す。"""
    arch, index, seed = task
    from run_match import play_match  # noqa: PLC0415 - ワーカー内で遅延 import

    opponent = _opponent_agent(arch)
    decision_ms: list[float] = []

    self_agent_raw = _W["self_agent"]

    def timed_self_agent(obs):
        t0 = time.perf_counter()
        action = self_agent_raw(obs)
        decision_ms.append((time.perf_counter() - t0) * 1000.0)
        return action

    # 偶数 index はドラパルト側が player0(先手扱い)。群をまたいで同一の割り当てになる。
    self_is_player0 = (index % 2 == 0)
    self_deck, opp_deck = _W["self_deck"], _W["opp_decks"][arch]
    if self_is_player0:
        agent0, agent1, deck0, deck1, self_index = timed_self_agent, opponent, self_deck, opp_deck, 0
    else:
        agent0, agent1, deck0, deck1, self_index = opponent, timed_self_agent, opp_deck, self_deck, 1

    result = play_match(agent0, agent1, deck0, deck1, seed=seed)
    win = None if result.error is not None else bool(result.winner == self_index)

    return {
        "group": _W["group"],
        "arch": arch,
        "index": index,
        "seed": seed,
        "self_player_index": self_index,
        "win": win,
        "turns": result.turns,
        "steps": result.steps,
        "seconds": round(result.seconds, 3),
        "error": result.error,
        "dec_ms": [round(v, 3) for v in decision_ms],
    }


# ---------------------------------------------------------------------------
# 集計(shard をマージしても同じ関数で集計できるよう、レコード列だけに依存させる)
# ---------------------------------------------------------------------------

def summarize_group(records: list[dict]) -> dict:
    """1群ぶんのレコードから勝率・対面別内訳・エラー・1決定あたり時間を集計する。"""
    wins = 0
    valid = 0
    errors: dict[str, int] = {}
    by_opp: dict[str, dict] = {}
    dec_all: list[float] = []
    turns_sum = 0
    turns_n = 0
    seconds_sum = 0.0

    for rec in records:
        dec_all.extend(rec.get("dec_ms", []))
        seconds_sum += rec.get("seconds") or 0.0
        arch = rec["arch"]
        slot = by_opp.setdefault(arch, {"wins": 0, "games": 0, "errors": 0})
        if rec["error"] is not None:
            errors[rec["error"]] = errors.get(rec["error"], 0) + 1
            slot["errors"] += 1
            continue
        valid += 1
        slot["games"] += 1
        if rec["win"]:
            wins += 1
            slot["wins"] += 1
        if rec.get("turns") is not None:
            turns_sum += rec["turns"]
            turns_n += 1

    for slot in by_opp.values():
        n = slot["games"]
        slot["win_rate"] = (slot["wins"] / n) if n else None
        lo, hi = wilson(slot["wins"], n)
        slot["wilson_95ci"] = [lo, hi]

    dec_all.sort()
    lo, hi = wilson(wins, valid)
    return {
        "wins": wins,
        "games": valid,
        "win_rate": (wins / valid) if valid else None,
        "wilson_95ci": [lo, hi],
        "errors": {"count": sum(errors.values()), "by_reason": errors},
        "avg_turns": (turns_sum / turns_n) if turns_n else None,
        "total_seconds": round(seconds_sum, 1),
        "decision_ms": {
            "n": len(dec_all),
            "p50": quantile(dec_all, 0.50),
            "p95": quantile(dec_all, 0.95),
            "mean": (sum(dec_all) / len(dec_all)) if dec_all else None,
            "max": dec_all[-1] if dec_all else None,
        },
        "by_opponent": dict(sorted(by_opp.items())),
    }


def compare_groups(records: list[dict], treatment: str, control: str) -> dict:
    """treatment − control の差(pt)と片側 p 値。対応のある McNemar も参考として付ける。"""
    t_recs = [r for r in records if r["group"] == treatment and r["error"] is None]
    c_recs = [r for r in records if r["group"] == control and r["error"] is None]
    tw, tn = sum(1 for r in t_recs if r["win"]), len(t_recs)
    cw, cn = sum(1 for r in c_recs if r["win"]), len(c_recs)

    # 対応のある比較: (arch, index, seed) が一致する試合どうしを突き合わせる。
    t_map = {(r["arch"], r["index"], r["seed"]): r["win"] for r in t_recs}
    c_map = {(r["arch"], r["index"], r["seed"]): r["win"] for r in c_recs}
    common = sorted(set(t_map) & set(c_map))
    b = sum(1 for k in common if t_map[k] and not c_map[k])
    c = sum(1 for k in common if (not t_map[k]) and c_map[k])

    diff_pt = ((tw / tn) - (cw / cn)) * 100 if (tn and cn) else None
    return {
        "treatment": treatment,
        "control": control,
        "treatment_rate": (tw / tn) if tn else None,
        "control_rate": (cw / cn) if cn else None,
        "diff_pt": diff_pt,
        "unpaired": {
            **two_proportion_z(tw, tn, cw, cn),
            "p_fisher_one_sided": fisher_exact_greater(tw, tn, cw, cn),
            "n_treatment": tn, "n_control": cn,
        },
        "paired_mcnemar": {
            "n_pairs": len(common),
            "treatment_only_wins": b,
            "control_only_wins": c,
            "p_one_sided": mcnemar_exact_greater(b, c),
            "note": "cg のシャッフルは開幕以降 seed 制御外のため完全な CRN ではない。参考値。",
        },
    }


def load_records(path: str | Path) -> tuple[list[dict], list[dict]]:
    """JSONL を読み、``(試合レコード, メタ行)`` に分けて返す(shard マージ用)。"""
    records: list[dict] = []
    metas: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                metas.append(obj["_meta"])
            elif "group" in obj:
                records.append(obj)
    return records, metas


def aggregate(records: list[dict], meta: dict | None = None) -> dict:
    """全レコード(複数 shard をマージしたものでも可)から最終集計を作る。"""
    groups = sorted({r["group"] for r in records})
    out: dict = {
        "meta": meta or {},
        "n_records": len(records),
        "groups": {g: summarize_group([r for r in records if r["group"] == g]) for g in groups},
        "comparisons": [],
    }
    for treatment in ("plan", "plan_off"):
        if treatment in groups and "rule" in groups:
            out["comparisons"].append(compare_groups(records, treatment, "rule"))
    return out


def print_summary(summary: dict) -> None:
    print("=" * 72)
    for group, stat in summary["groups"].items():
        rate = stat["win_rate"]
        lo, hi = stat["wilson_95ci"]
        dec = stat["decision_ms"]
        print(f"[{group}] {stat['wins']}/{stat['games']} = "
              f"{(rate if rate is not None else float('nan')):.3f} "
              f"CI95[{lo:.3f},{hi:.3f}]  errors={stat['errors']['count']}")
        p50 = dec["p50"] if dec["p50"] is not None else float("nan")
        p95 = dec["p95"] if dec["p95"] is not None else float("nan")
        print(f"    decision_ms p50={p50:.2f} p95={p95:.2f} (n={dec['n']}) "
              f"avg_turns={stat['avg_turns']}")
        for arch, slot in stat["by_opponent"].items():
            r = slot["win_rate"]
            print(f"      {arch:22s} {slot['wins']}/{slot['games']} = "
                  f"{(r if r is not None else float('nan')):.3f}")
    for cmp_ in summary["comparisons"]:
        d = cmp_["diff_pt"]
        z = cmp_["unpaired"]["z"]
        pz = cmp_["unpaired"]["p_one_sided"]
        pf = cmp_["unpaired"]["p_fisher_one_sided"]
        mc = cmp_["paired_mcnemar"]
        print("-" * 72)
        print(f"{cmp_['treatment']} − {cmp_['control']} = "
              f"{(d if d is not None else float('nan')):+.1f}pt  "
              f"z={z if z is None else round(z, 3)} "
              f"p(one-sided,z)={pz if pz is None else round(pz, 4)} "
              f"p(fisher)={pf if pf is None else round(pf, 4)}")
        print(f"  paired McNemar: b={mc['treatment_only_wins']} c={mc['control_only_wins']} "
              f"pairs={mc['n_pairs']} p={mc['p_one_sided']}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------

def run_group(group: str, root: Path, tasks: list[tuple[str, int, int]], workers: int,
              horizon: int, config: str, self_deck: Path, sink, deadline: float | None,
              progress_every: int) -> list[dict]:
    """1群ぶんを ProcessPoolExecutor(spawn)で実行し、完了ごとに sink へ書き出す。"""
    import multiprocessing as mp

    archs = {a for a, _i, _s in tasks}
    opp_decks = {}
    opp_weights = {}
    for arch, _share, gen in FIELD:
        if arch not in archs:
            continue
        deckdir = _DECKDIR_OF[gen]
        opp_decks[arch] = str(root / "kaggle_replays" / "meta_analysis" / deckdir / arch / "01.csv")
        opp_weights[arch] = str(root / "sample_submission" / "ptcg_ai" / "learning"
                                / f"policy_weights_{arch}{gen}.json")

    initargs = (str(root), group, horizon, config, str(self_deck), opp_decks, opp_weights)
    records: list[dict] = []
    t0 = time.time()
    ctx = mp.get_context("spawn")  # Windows/Linux で挙動をそろえる(fork 依存にしない)

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_worker_init, initargs=initargs) as ex:
        futures = {ex.submit(_worker_play, t): t for t in tasks}
        try:
            for fut in _as_completed(futures):
                rec = fut.result()
                records.append(rec)
                sink(rec)
                done = len(records)
                if progress_every and (done % progress_every == 0 or done == len(tasks)):
                    valid = [r for r in records if r["error"] is None]
                    wins = sum(1 for r in valid if r["win"])
                    rate = wins / len(valid) if valid else float("nan")
                    print(f"[{group}] {done}/{len(tasks)} games  win_rate={rate:.3f} "
                          f"({wins}/{len(valid)})  elapsed={time.time() - t0:.0f}s", flush=True)
                if deadline is not None and time.time() > deadline:
                    print(f"[{group}] 時間予算に到達。未実行のタスクを打ち切ります "
                          f"({done}/{len(tasks)} 完了)", flush=True)
                    for f in futures:
                        f.cancel()
                    break
        except KeyboardInterrupt:  # noqa: PERF203 - 中断時もそこまでの結果を活かす
            for f in futures:
                f.cancel()
            raise
    return records


def _as_completed(futures):
    from concurrent.futures import as_completed
    return as_completed(list(futures))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", default="rule,plan",
                    help=f"比較する群をカンマ区切りで({'/'.join(GROUPS)})。default: rule,plan")
    ap.add_argument("--games", type=int, default=400, help="1群あたりの総試合数(default: 400)")
    ap.add_argument("--workers", type=int, default=2,
                    help="プロセス並列数(default: 2。Kaggle CPU は 4)")
    ap.add_argument("--seed-base", type=int, default=970000, help="seed の基点(shard ごとに変える)")
    ap.add_argument("--horizon", type=int, default=2, help="DRAGAPULT_PLANNER_HORIZON(default: 2)")
    ap.add_argument("--config", default=CONFIG, help=f"相手 ml_policy の config(default: {CONFIG})")
    ap.add_argument("--repo-root", default=str(_DEFAULT_ROOT),
                    help="リポジトリルート(Kaggle では展開先を渡す)")
    ap.add_argument("--self-deck", default=None,
                    help="ドラパルト側のデッキCSV(default: opponents/dragapult_ex_deck.csv。"
                         "rule/planning agent の set_card_counts がこのデッキ構成を前提とするため、"
                         "別デッキを使う場合は結果に必ず記録する)")
    ap.add_argument("--only-archs", default=None,
                    help="デバッグ用: フィールドを指定アーキだけに絞る(カンマ区切り)。"
                         "本測定では使わないこと(フィールド構成が変わる)")
    ap.add_argument("--out", default=None, help="出力 JSONL(default: kaggle_run/out/drapa_<ts>.jsonl)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="この秒数を超えたら未実行タスクを打ち切る(0=無制限)")
    ap.add_argument("--progress-every", type=int, default=10, help="N試合ごとに進捗を出す")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.repo_root).resolve()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group: {g!r} (choices: {sorted(GROUPS)})")

    self_deck = Path(args.self_deck).resolve() if args.self_deck else (
        root / "opponents" / "dragapult_ex_deck.csv")
    if not self_deck.exists():
        raise SystemExit(f"self deck not found: {self_deck}")

    out_path = Path(args.out) if args.out else (
        _HERE / "out" / f"drapa_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    only = {a.strip() for a in args.only_archs.split(",") if a.strip()} if args.only_archs else None
    tasks = build_tasks(args.games, args.seed_base, only)
    alloc = {a: n for a, _g, _d, n in allocate(args.games, only)}
    meta = {
        "config": args.config,
        "only_archs": sorted(only) if only else None,
        "self_deck": str(self_deck),
        "self_deck_is_default": self_deck.name == "dragapult_ex_deck.csv",
        "groups": groups,
        "games_per_group": len(tasks),
        "seed_base": args.seed_base,
        "horizon": args.horizon,
        "allocation": alloc,
        "workers": args.workers,
        "repo_root": str(root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    print(json.dumps({"meta": meta}, ensure_ascii=False), flush=True)

    hard_deadline = (time.time() + args.max_seconds) if args.max_seconds > 0 else None
    all_records: list[dict] = []
    with out_path.open("w", encoding="utf-8") as fp:
        def sink(rec: dict) -> None:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()  # 途中で落ちてもそこまでの試合を残す

        fp.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        fp.flush()
        for i, group in enumerate(groups):
            # 時間予算は残り群で等分する(先頭の群が全部使い切って対照群が0試合、を防ぐ)。
            deadline = None
            if hard_deadline is not None:
                deadline = time.time() + max(0.0, hard_deadline - time.time()) / (len(groups) - i)
            print(f"=== group={group} ({len(tasks)} games, workers={args.workers}) ===", flush=True)
            all_records.extend(run_group(
                group, root, tasks, args.workers, args.horizon, args.config,
                self_deck, sink, deadline, args.progress_every))

    summary = aggregate(all_records, meta)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print_summary(summary)
    print(f"\nrecords -> {out_path}\nsummary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
