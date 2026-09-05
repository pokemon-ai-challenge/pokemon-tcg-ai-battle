"""C0: 決定化本数による探索ヘッドルームを固定 root で測る。

production(`abl_5_full` + climb 重み)のローカル対戦中に、pipeline が実際に
決定化探索へ入った decision root だけを観測する。同一 root・同一 top-k 候補を
独立 seed の low(N=2) / ref(N=32) / ref2(N=32) で再評価し、壁時計予算ではなく
完走本数を固定して比較する。

production / cg / config / 重み / deck は変更せず、実行時の関数 wrap と既存の
探索ヘルパーだけを使う。
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import SelectContext, SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.learning import encoder as ENC  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402


_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
_CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")
_OPPONENTS = (
    "mega_lucario_ex",
    "dragapult_ex",
    "crustle",
    "marnie_grimmsnarl_ex",
    "archaludon_ex",
    "shirona_garchomp_ex",
)
_GAME_OFFSET = 4_600_000
_N_LOW = 2
_N_REF = 32
_BOOTSTRAP_RESAMPLES = 1000
# --field-preset の choices。"july7" は本ファイル既存の _OPPONENTS(現行挙動)専用で、
# train_league.FIELD_PRESETS には無いキーなので別扱いする。
_FIELD_PRESET_CHOICES = ("july7", "g2", "mix")


def _resolve_path(raw: str) -> Path:
    """CLI で渡された相対パスを repo ルート基準で解決する。

    `agents.ensure_production_cwd()` が import 時点で cwd を `sample_submission/` に
    変えているため、呼び出し側の cwd を基準にした素朴な相対パス解決はできない。
    絶対パス・既存の相対パス(cwd=sample_submission基準で解決できる場合)はそのまま使う。
    """
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return path
    return _ROOT / path


def _resolve_field(preset: str) -> list[tuple[str, float, str, str]]:
    """--field-preset を (arch, share, 重みファイルgen接尾辞, デッキdir名) のリストに解決する。

    "july7"(既定)は本ファイル既存の _OPPONENTS(6アーキ・接尾辞なし・archetype_decks)を
    そのまま使い、現行挙動を一切変えない。"g2"/"mix" は kaggle_replays/rl/train_league.py の
    FIELD_PRESETS を都度 import して使う(kaggle_replays/rl/eval_field.py と同じ手口)。
    train_league.py 自体は変更しない。実行系(torch 等)が重いため、july7 では import しない。
    """
    if preset == "july7":
        return [(arch, 1.0, "", "archetype_decks") for arch in _OPPONENTS]
    _rl_dir = _ROOT / "kaggle_replays" / "rl"
    _league_dir = _ROOT / "league"
    for _p in (str(_rl_dir), str(_league_dir)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import train_league  # noqa: E402 - g2/mix 指定時のみの遅延import

    return list(train_league.FIELD_PRESETS[preset])


def _pick_opponent(
    field: list[tuple[str, float, str, str]], game_id: int, rng: random.Random,
) -> tuple[str, str, Path]:
    """フィールド定義から相手アーキを1体選び、(アーキ名, 重みパス, デッキパス) を返す。

    share が全て一様(="july7"の現行挙動)なら従来どおり game_id の巡回で決定論的に選ぶ
    (_OPPONENTS[game_id % len(_OPPONENTS)] と完全一致)。share にばらつきがあれば
    kaggle_replays/rl/collect_field.py と同じ累積share の重み付き抽出で選ぶ。
    """
    shares = [share for _arch, share, _gen, _deckdir in field]
    if len(set(shares)) <= 1:
        idx = game_id % len(field)
    else:
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
    deck_path = _DECKDIR.parent / deckdir / arch / "01.csv"
    return arch, weights_path, deck_path


ROWS: list[dict] = []
STATS: Counter = Counter()
CTX: dict = {
    "game_id": 0,
    "arch": "?",
    "record": False,
    "step": 0,
    "taken": 0,
    "target": 0,
    "per_game_cap": 0,
}


def _enum_name(value, enum_cls) -> str:
    """未知の Enum 値も落とさず、既知値は名前で返す。"""
    if value is None:
        return "NONE"
    raw = int(value)
    member = enum_cls._value2member_map_.get(raw)
    return member.name if member is not None else str(raw)


def _policy_view(obs, cfg_full: dict, policy_model):
    """production と同じ Policy スコア、確率、順位を作る。"""
    deadline = time.perf_counter() + 10.0
    factory = MA._model_hidden_state_factory(obs, cfg_full)
    scores = policy_model.score_options(obs, factory, deadline)
    if not scores or len(scores) != len(obs.select.option):
        return None
    probs = P._softmax(scores)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return probs, ranked


def _begin(obs, seed: int):
    """seed を明示した決定化で search root を作る。"""
    me = obs.current.yourIndex
    hidden = search_adapter.to_search_begin_kwargs(
        match_context.get_own_state(me),
        match_context.get_opponent_state(me),
        obs,
        rng=random.Random(seed),
    )
    return P._begin(obs, hidden)


def _run_search(
    obs,
    me: int,
    candidates: list[int],
    probs: list[float],
    pcfg: dict,
    evaluator,
    policy_model,
    n_det: int,
    seed: int,
) -> tuple[int, list[float | None], list[int]] | None:
    """候補集合を N 決定化で再評価し、production と同じ tie 処理で選ぶ。"""
    aggregate: dict[int, list[float]] = {i: [] for i in candidates}
    try:
        for det in range(n_det):
            root = None
            try:
                root = _begin(obs, seed + det * 1009)
            except Exception:  # noqa: BLE001 - 失敗 root は集計せず呼び出し側で棄却
                STATS["begin_fail"] += 1
                continue
            try:
                for idx in candidates:
                    score = P._evaluate_candidate(
                        root,
                        [idx],
                        me,
                        pcfg,
                        float("inf"),
                        evaluator,
                        policy_model,
                    )
                    if score is not None:
                        aggregate[idx].append(float(score))
            finally:
                try:
                    P.cg_api.search_release(root.searchId)
                except Exception:  # noqa: BLE001 - 解放失敗で観測全体を止めない
                    pass
    finally:
        try:
            P.cg_api.search_end()
        except Exception:  # noqa: BLE001 - 同上
            pass

    # 本数固定の比較なので、全候補が指定本数を完走した root だけを採用する。
    counts = [len(aggregate[i]) for i in candidates]
    if any(n != n_det for n in counts):
        STATS["incomplete_search"] += 1
        return None

    means = {i: st.mean(aggregate[i]) for i in candidates}
    best_mean = max(means.values())
    tie_eps = float(pcfg["tie_eps"])
    contenders = [i for i, value in means.items() if best_mean - value <= tie_eps]
    action = max(contenders, key=lambda i: probs[i])
    dense: list[float | None] = [None] * len(obs.select.option)
    for idx, value in means.items():
        dense[idx] = round(value, 6)
    return action, dense, counts


def _pipeline_started(inner, obs) -> tuple[list[int], int]:
    """production 呼び出し中の pipeline root 作成回数だけを観測する。"""
    original = P._begin
    starts = 0

    def traced_begin(*args, **kwargs):
        nonlocal starts
        starts += 1
        return original(*args, **kwargs)

    P._begin = traced_begin
    try:
        action = inner(obs)
    finally:
        P._begin = original
    return action, starts


def _candidate_card_ids(obs, candidates: list[int]) -> list[int | None]:
    """Observation から解決できる候補 cardId を返す。"""
    state = obs.current
    return [ENC._resolve_card_id(obs.select.option[i], state) for i in candidates]


def _gap(values: list[float | None], candidates: list[int]) -> float | None:
    """候補価値の1位と2位の差。"""
    ranked = sorted((values[i] for i in candidates if values[i] is not None), reverse=True)
    return round(ranked[0] - ranked[1], 6) if len(ranked) >= 2 else None


def _audit_root(obs, cfg_full: dict, policy_model, pcfg: dict) -> dict | None:
    """同一 root を low / ref / ref2 の独立 seed で固定本数再評価する。"""
    view = _policy_view(obs, cfg_full, policy_model)
    if view is None:
        STATS["policy_view_fail"] += 1
        return None
    probs, ranked = view

    # C0 は production の固定 top_k=4 を比較対象とする。
    candidate_cfg = dict(pcfg)
    candidate_cfg["top_k"] = 4
    candidates = P._select_candidate_indices(obs.select, ranked, candidate_cfg, probs)
    if len(candidates) < 2:
        STATS["too_few_candidates"] += 1
        return None

    me = obs.current.yourIndex
    evaluator = leaf_eval_module.build_evaluator(pcfg.get("leaf_eval"))
    base = 9_000_000 + CTX["game_id"] * 100_003 + CTX["step"] * 101
    low = _run_search(
        obs, me, candidates, probs, pcfg, evaluator, policy_model,
        _N_LOW, base + 1_000_000,
    )
    ref = _run_search(
        obs, me, candidates, probs, pcfg, evaluator, policy_model,
        _N_REF, base + 2_000_000,
    )
    ref2 = _run_search(
        obs, me, candidates, probs, pcfg, evaluator, policy_model,
        _N_REF, base + 3_000_000,
    )
    if low is None or ref is None or ref2 is None:
        return None

    a_low, v_low, n_low = low
    a_ref, v_ref, n_ref = ref
    a_ref2, _v_ref2, n_ref2 = ref2
    card_ids = _candidate_card_ids(obs, candidates)
    return {
        "candidate_count": len(candidates),
        "candidate_indices": candidates,
        "candidate_card_ids": card_ids,
        "identifiable": all(card_id is not None for card_id in card_ids),
        "a_low": a_low,
        "a_ref": a_ref,
        "a_ref2": a_ref2,
        "v_ref": v_ref,
        "v_low": v_low,
        "top1top2gap_ref": _gap(v_ref, candidates),
        "n_completed": {"low": n_low, "ref": n_ref, "ref2": n_ref2},
    }


def _audit_root_sweep(
    obs, cfg_full: dict, policy_model, pcfg: dict, arm_ns: list[int],
) -> dict | None:
    """同一 root を任意の決定化数リスト arm_ns で再評価し、限界効用スイープ用の行を作る。

    各 N を独立 seed で1回ずつ評価し、さらに最大 N を別 seed でもう一度評価した ref2 を
    null 対照(探索分散だけで生じる regret の床)として必ず添える。
    """
    view = _policy_view(obs, cfg_full, policy_model)
    if view is None:
        STATS["policy_view_fail"] += 1
        return None
    probs, ranked = view

    # C0 は production の固定 top_k=4 を比較対象とする(_audit_root と同じ)。
    candidate_cfg = dict(pcfg)
    candidate_cfg["top_k"] = 4
    candidates = P._select_candidate_indices(obs.select, ranked, candidate_cfg, probs)
    if len(candidates) < 2:
        STATS["too_few_candidates"] += 1
        return None

    me = obs.current.yourIndex
    evaluator = leaf_eval_module.build_evaluator(pcfg.get("leaf_eval"))
    base = 9_000_000 + CTX["game_id"] * 100_003 + CTX["step"] * 101

    arm_results: dict[str, dict] = {}
    for offset, n_det in enumerate(arm_ns):
        result = _run_search(
            obs, me, candidates, probs, pcfg, evaluator, policy_model,
            n_det, base + (offset + 1) * 1_000_000,
        )
        if result is None:
            return None
        action, values, counts = result
        arm_results[str(n_det)] = {"action": action, "values": values, "n_completed": counts}

    max_n = max(arm_ns)
    ref2 = _run_search(
        obs, me, candidates, probs, pcfg, evaluator, policy_model,
        max_n, base + (len(arm_ns) + 1) * 1_000_000,
    )
    if ref2 is None:
        return None
    a_ref2, _v_ref2, n_ref2 = ref2

    ref_values = arm_results[str(max_n)]["values"]
    card_ids = _candidate_card_ids(obs, candidates)
    return {
        "candidate_count": len(candidates),
        "candidate_indices": candidates,
        "candidate_card_ids": card_ids,
        "identifiable": all(card_id is not None for card_id in card_ids),
        "arms_n": arm_ns,
        "max_n": max_n,
        "arm_results": arm_results,
        "ref2": {"action": a_ref2, "n_completed": n_ref2},
        "top1top2gap_ref": _gap(ref_values, candidates),
    }


def make_probe(inner, cfg_full: dict, policy_model, target: int, per_game_cap: int, audit_fn=_audit_root):
    """production agent に pipeline 実走観測と固定 root 再評価を重ねる。

    ``audit_fn`` は既定で従来の low/ref/ref2 3-arm 比較(_audit_root)。--arms 指定時は
    呼び出し側が _audit_root_sweep を束縛したものを渡す。
    """
    pcfg = {**P.DEFAULTS, **cfg_full["pipeline"]}

    def probe(obs):
        sel = obs.select
        eligible = (
            sel is not None
            and CTX["record"]
            and sel.type == SelectType.MAIN
            and sel.maxCount == 1
            and bool(sel.option)
            and obs.current is not None
            and obs.current.result == -1
        )
        if eligible:
            action, pipeline_starts = _pipeline_started(inner, obs)
        else:
            action = inner(obs)
            pipeline_starts = 0
        if sel is None:
            return action

        CTX["step"] += 1
        if not eligible or pipeline_starts == 0:
            STATS["pipeline_not_started"] += 1
            return action
        STATS["pipeline_started"] += 1
        # --min-turn: 序盤rootを収集対象から外す(per-game-cap が序盤を先取りして
        # 中終盤が未サンプルになる偏りへの対処。既定0=従来挙動)。
        if int(getattr(obs.current, "turn", 0) or 0) < CTX.get("min_turn", 0):
            STATS["min_turn_skip"] = STATS.get("min_turn_skip", 0) + 1
            return action
        if len(ROWS) >= target or CTX["taken"] >= per_game_cap:
            STATS["cap_skip"] += 1
            return action

        started = time.perf_counter()
        try:
            row = audit_fn(obs, cfg_full, policy_model, pcfg)
        except Exception as exc:  # noqa: BLE001 - 観測失敗は対戦を止めない
            STATS["audit_error_" + type(exc).__name__] += 1
            return action
        if row is None:
            STATS["audit_none"] += 1
            return action

        row.update({
            "game_id": CTX["game_id"],
            "step": CTX["step"],
            "turn": int(getattr(obs.current, "turn", 0)),
            "select_type": _enum_name(sel.type, SelectType),
            "select_context": _enum_name(sel.context, SelectContext),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "arch": CTX["arch"],
            "production_action": action[0] if action else None,
            "production_determinizations_started": pipeline_starts,
        })
        ROWS.append(row)
        CTX["taken"] += 1
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


def _regret(row: dict) -> float | None:
    """ref 価値上の方向付き regret を返す。"""
    values = row.get("v_ref") or []
    a_ref, a_low = row.get("a_ref"), row.get("a_low")
    if not isinstance(a_ref, int) or not isinstance(a_low, int):
        return None
    if not (0 <= a_ref < len(values) and 0 <= a_low < len(values)):
        return None
    if values[a_ref] is None or values[a_low] is None:
        return None
    return float(values[a_ref]) - float(values[a_low])


def _summary(rows: list[dict]) -> dict:
    """不一致率・参照自己不一致率・regret 要約を作る。"""
    regrets = [value for row in rows if (value := _regret(row)) is not None]
    n = len(rows)
    mismatch = sum(row.get("a_low") != row.get("a_ref") for row in rows)
    ref_self = sum(row.get("a_ref") != row.get("a_ref2") for row in rows)
    return {
        "roots": n,
        "games": len({row.get("game_id") for row in rows}),
        "mismatch_rate": round(mismatch / n, 6) if n else None,
        "ref_self_mismatch_rate": round(ref_self / n, 6) if n else None,
        "regret": {
            "n": len(regrets),
            "mean": round(st.mean(regrets), 6) if regrets else None,
            "median": round(st.median(regrets), 6) if regrets else None,
            "p95": round(_percentile(regrets, 95), 6) if regrets else None,
        },
    }


def _turn_band(turn: int) -> str:
    if turn <= 4:
        return "1-4"
    if turn <= 10:
        return "5-10"
    return "11+"


def _gap_band(gap: float | None) -> str:
    if gap is None:
        return "unknown"
    if gap < 0.01:
        return "gap<0.01"
    if gap <= 0.05:
        return "0.01-0.05"
    return "gap>0.05"


def _strata(rows: list[dict], population: str) -> list[dict]:
    """指定された turn帯 × select_type × 接戦度で集計する。"""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            _turn_band(int(row.get("turn", 0))),
            str(row.get("select_type", "UNKNOWN")),
            _gap_band(row.get("top1top2gap_ref")),
        )
        groups[key].append(row)
    result = []
    for (turn, select_type, gap), members in sorted(groups.items()):
        result.append({
            "population": population,
            "turn_band": turn,
            "select_type": select_type,
            "gap_band": gap,
            **_summary(members),
        })
    return result


def _bootstrap(rows: list[dict], seed: int) -> dict:
    """試合をクラスタとして復元抽出し、全体指標の95% CIを作る。"""
    by_game: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        by_game[row.get("game_id")].append(row)
    games = list(by_game)
    if not games:
        return {
            "resamples": _BOOTSTRAP_RESAMPLES,
            "games": 0,
            "mismatch_rate": {"estimate": None, "ci95": [None, None]},
            "mean_regret": {"estimate": None, "ci95": [None, None]},
        }

    rng = random.Random(seed)
    mismatch_samples: list[float] = []
    regret_samples: list[float] = []
    for _ in range(_BOOTSTRAP_RESAMPLES):
        sampled: list[dict] = []
        for _game in games:
            sampled.extend(by_game[rng.choice(games)])
        if sampled:
            mismatch_samples.append(
                sum(row.get("a_low") != row.get("a_ref") for row in sampled)
                / len(sampled)
            )
        regrets = [value for row in sampled if (value := _regret(row)) is not None]
        if regrets:
            regret_samples.append(st.mean(regrets))

    observed = _summary(rows)
    mismatch_lo = _percentile(mismatch_samples, 2.5)
    mismatch_hi = _percentile(mismatch_samples, 97.5)
    regret_lo = _percentile(regret_samples, 2.5)
    regret_hi = _percentile(regret_samples, 97.5)
    return {
        "resamples": _BOOTSTRAP_RESAMPLES,
        "games": len(games),
        "mismatch_rate": {
            "estimate": observed["mismatch_rate"],
            "ci95": [
                round(mismatch_lo, 6) if mismatch_lo is not None else None,
                round(mismatch_hi, 6) if mismatch_hi is not None else None,
            ],
        },
        "mean_regret": {
            "estimate": observed["regret"]["mean"],
            "ci95": [
                round(regret_lo, 6) if regret_lo is not None else None,
                round(regret_hi, 6) if regret_hi is not None else None,
            ],
        },
    }


def _is_sweep_row(row: dict) -> bool:
    """--arms で収集した行かどうか(従来 low/ref/ref2 行と区別する)。"""
    return "arm_results" in row


def _sweep_regret(row: dict, n: int) -> float | None:
    """決定化数 n の regret。最大N(=row["max_n"])の値ベクトルで、最大N選択とn選択の値差を測る。"""
    max_n = row.get("max_n")
    arms = row.get("arm_results") or {}
    ref_arm = arms.get(str(max_n))
    arm = arms.get(str(n))
    if ref_arm is None or arm is None:
        return None
    values = ref_arm.get("values") or []
    a_ref, a_n = ref_arm.get("action"), arm.get("action")
    if not isinstance(a_ref, int) or not isinstance(a_n, int):
        return None
    if not (0 <= a_ref < len(values) and 0 <= a_n < len(values)):
        return None
    if values[a_ref] is None or values[a_n] is None:
        return None
    return float(values[a_ref]) - float(values[a_n])


def _sweep_null_regret(row: dict) -> float | None:
    """ref2(最大N・別seed)の null regret。探索分散だけで生じる regret の床を測る。"""
    max_n = row.get("max_n")
    arms = row.get("arm_results") or {}
    ref_arm = arms.get(str(max_n))
    ref2 = row.get("ref2") or {}
    if ref_arm is None:
        return None
    values = ref_arm.get("values") or []
    a_ref, a_ref2 = ref_arm.get("action"), ref2.get("action")
    if not isinstance(a_ref, int) or not isinstance(a_ref2, int):
        return None
    if not (0 <= a_ref < len(values) and 0 <= a_ref2 < len(values)):
        return None
    if values[a_ref] is None or values[a_ref2] is None:
        return None
    return float(values[a_ref]) - float(values[a_ref2])


def _analyze_sweep(rows: list[dict]) -> dict:
    """決定化数 N の限界効用カーブ(最大N参照へのregret・不一致率)と null 対照を作る。"""
    all_ns: set[int] = set()
    for row in rows:
        all_ns.update(int(n) for n in row.get("arms_n", []))
    curve = []
    for n in sorted(all_ns):
        applicable = [row for row in rows if n in (row.get("arms_n") or [])]
        regrets = [value for row in applicable if (value := _sweep_regret(row, n)) is not None]
        mismatches = 0
        for row in applicable:
            max_n = row.get("max_n")
            arms = row.get("arm_results") or {}
            ref_action = (arms.get(str(max_n)) or {}).get("action")
            n_action = (arms.get(str(n)) or {}).get("action")
            if ref_action != n_action:
                mismatches += 1
        curve.append({
            "n": n,
            "roots": len(applicable),
            "mismatch_rate": round(mismatches / len(applicable), 6) if applicable else None,
            "regret_mean": round(st.mean(regrets), 6) if regrets else None,
            "regret_median": round(st.median(regrets), 6) if regrets else None,
            "regret_p95": round(_percentile(regrets, 95), 6) if regrets else None,
        })

    null_regrets = [value for row in rows if (value := _sweep_null_regret(row)) is not None]
    null_mismatches = 0
    for row in rows:
        max_n = row.get("max_n")
        arms = row.get("arm_results") or {}
        ref_action = (arms.get(str(max_n)) or {}).get("action")
        ref2_action = (row.get("ref2") or {}).get("action")
        if ref_action != ref2_action:
            null_mismatches += 1

    return {
        "roots": len(rows),
        "games": len({row.get("game_id") for row in rows}),
        "curve": curve,
        "ref2_null": {
            "mismatch_rate": round(null_mismatches / len(rows), 6) if rows else None,
            "regret_mean": round(st.mean(null_regrets), 6) if null_regrets else None,
            "regret_median": round(st.median(null_regrets), 6) if null_regrets else None,
            "regret_p95": round(_percentile(null_regrets, 95), 6) if null_regrets else None,
        },
    }


def analyze(paths: list[str]) -> None:
    """収集済み JSONL を集計し、JSON レポートを標準出力へ出す。

    従来行(low/ref/ref2)と --arms スイープ行が混在していても、それぞれ別集計にする。
    ファイル群が全て従来行なら出力形は変更前と完全に同じ(sweepキーが増えない)。
    """
    rows = _read_rows(paths)
    sweep_rows = [row for row in rows if _is_sweep_row(row)]
    legacy_rows = [row for row in rows if not _is_sweep_row(row)]

    report: dict = {"files": paths}
    if legacy_rows:
        identifiable = [row for row in legacy_rows if row.get("identifiable") is True]
        report.update({
            "all": _summary(legacy_rows),
            "identifiable": _summary(identifiable),
            "strata": _strata(legacy_rows, "all") + _strata(identifiable, "identifiable"),
            "cluster_bootstrap": {
                "all": _bootstrap(legacy_rows, 20260813),
                "identifiable": _bootstrap(identifiable, 20260814),
            },
        })
    if sweep_rows:
        report["sweep"] = _analyze_sweep(sweep_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def collect(args) -> None:
    """production 対戦を回して固定 root 比較を JSONL(gzip)へ保存する。

    --deck/--weights/--field-preset/--arms を省略すると、従来どおり
    climb 重み・sample_submission/deck.csv・_OPPONENTS 6アーキ巡回・low/ref/ref2 3-arm になる。
    """
    if args.target < 1 or args.max_games < 1 or args.per_game_cap < 1:
        raise ValueError("--target/--max-games/--per-game-cap は1以上が必要です")
    if args.num_workers < 1 or not 0 <= args.worker_id < args.num_workers:
        raise ValueError("--worker-id は 0 以上 --num-workers 未満にしてください")

    arm_ns: list[int] | None = None
    if args.arms:
        arm_ns = sorted({int(token) for token in args.arms.split(",") if token.strip()})
        if not arm_ns:
            raise ValueError("--arms の指定が空です")

    local_target = args.target
    if args.num_workers > 1:
        local_target = max(1, args.target // args.num_workers)
    CTX.update(target=local_target, per_game_cap=args.per_game_cap,
               min_turn=getattr(args, "min_turn", 0))

    own_weights = str(_resolve_path(args.weights)) if args.weights else _CLIMB
    own_deck_path = _resolve_path(args.deck) if args.deck else (_SUB / "deck.csv")
    field = _resolve_field(args.field_preset)
    field_rng = random.Random(0xC0FFEE ^ args.worker_id)

    cfg = agents.load_config_copy("abl_5_full")
    cfg["policy_weights_path"] = own_weights
    production = agents.make_ml_policy_agent(copy.deepcopy(cfg))
    policy_model = MA._get_model({"policy_weights_path": own_weights})
    if arm_ns is not None:
        def audit_fn(obs, cfg_full, model, pcfg, _arm_ns=arm_ns):
            return _audit_root_sweep(obs, cfg_full, model, pcfg, _arm_ns)
    else:
        audit_fn = _audit_root
    probe = make_probe(production, cfg, policy_model, local_target, args.per_game_cap, audit_fn)
    own_deck = runner.load_deck(own_deck_path)

    started = time.perf_counter()
    for game_index in range(args.max_games):
        if len(ROWS) >= local_target:
            break
        game_id = _GAME_OFFSET + args.worker_id + game_index * args.num_workers
        arch, opp_weights_path, opp_deck_path = _pick_opponent(field, game_id, field_rng)
        opp_cfg = agents.load_config_copy("abl_5_full")
        opp_cfg["policy_weights_path"] = opp_weights_path
        opponent = agents.make_ml_policy_agent(opp_cfg)
        opp_deck = runner.load_deck(opp_deck_path)
        own_first = game_index % 2 == 0
        CTX.update(
            game_id=game_id,
            arch=arch,
            record=True,
            step=0,
            taken=0,
        )
        if own_first:
            runner.play_game(probe, opponent, own_deck, opp_deck)
        else:
            runner.play_game(opponent, probe, opp_deck, own_deck)
        CTX["record"] = False
        print(
            "  [w{}] game#{} roots={}/{} pipeline={} {:.1f}分".format(
                args.worker_id,
                game_id,
                len(ROWS),
                local_target,
                STATS["pipeline_started"],
                (time.perf_counter() - started) / 60.0,
            ),
            file=sys.stderr,
            flush=True,
        )

    output = _HERE / f"_c0_headroom_{args.tag}.jsonl.gz"
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        for row in ROWS:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "tag": args.tag,
        "output": str(output),
        "roots": len(ROWS),
        "target": local_target,
        "stats": dict(STATS),
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C0探索ヘッドルームを固定root比較で測定します。"
    )
    parser.add_argument("--target", type=int, default=2000, help="収集root数")
    parser.add_argument("--max-games", type=int, default=400, help="最大試合数")
    parser.add_argument("--per-game-cap", type=int, default=6, help="1試合の収集上限")
    parser.add_argument("--num-workers", type=int, default=1, help="並列worker総数")
    parser.add_argument("--worker-id", type=int, default=0, help="このworkerの番号")
    parser.add_argument("--tag", default="run", help="出力ファイル識別子")
    parser.add_argument("--min-turn", type=int, default=0,
                        help="このターン未満のrootは収集しない(序盤先取り偏りの対処。既定0=従来)")
    parser.add_argument(
        "--deck", default=None,
        help="自分側デッキCSVパス(省略時=sample_submission/deck.csv、現行挙動維持)",
    )
    parser.add_argument(
        "--weights", default=None,
        help="自分側 policy 重みJSONパス(省略時=climb重み、現行挙動維持)",
    )
    parser.add_argument(
        "--field-preset", default="july7", choices=_FIELD_PRESET_CHOICES,
        help="相手フィールド。july7=本ファイル既存の6アーキ巡回(既定・現行挙動維持)、"
             "g2/mix=kaggle_replays/rl/train_league.FIELD_PRESETSをshare比例抽出で使う",
    )
    parser.add_argument(
        "--arms", default=None,
        help="決定化数スイープをカンマ区切りNで指定(例 '2,4,8,16,32')。"
             "指定した全Nを個別評価し、最大Nをref2(別seed)でnull対照する。"
             "省略時=従来のlow(N=2)/ref(N=32)/ref2(N=32,別seed) 3-arm",
    )
    parser.add_argument(
        "--analyze",
        nargs="+",
        metavar="JSONL",
        help="収集せず、指定したJSONL(gzip可)を集計する",
    )
    args = parser.parse_args()
    if args.analyze:
        analyze(args.analyze)
    else:
        collect(args)


if __name__ == "__main__":
    main()
