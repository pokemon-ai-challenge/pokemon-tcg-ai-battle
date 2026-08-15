"""og_v032(クラッシュハンマー4枚)のハンマー浪費veto診断ランナー。

`ml_policy_agent._try_hammer_veto`(改造ハンマー1081専用)を一般化した
`hammer_veto.card_ids`(sample_submission/configs/abl_5_full_og_hammer.json で [1120] を指定)の
効果を測る前段として、まず**現状の浪費頻度**(shadowモード)と、config を切り替えたときの
**veto発火回数**を計測する。あわせて `runner.py` の `primary_win_condition` から
自分/相手それぞれの deckout(山札切れ)負け率も集計する。

やること:
  1. og_v032(デッキ `kaggle_replays/deck_search/candidates_ogerpon_stage3/g2top2_v032.csv`、
     重み `policy_weights_ogerpon_teal_ex_rl_mixogerpon.json`)を、
     `kaggle_replays/rl/train_league.FIELD_PRESETS["mix"]` と対戦させる。
  2. **shadowモード(既定 `--config abl_5_full`)**: veto は一切発火させない(config に
     `hammer_veto` キーが無い=production は完全に従来どおり動く)。その代わり
     `ml_policy_agent._apply_action_vetoes` を record-only にラップし、veto適用**前**の
     選択がクラッシュハンマー(1120)PLAY かつ `ko_search.can_ko_this_turn`=真 かつ
     ベンチに他の正当なエネ対象が無い(=`_hammer_veto_guard_blocks` が False)場合を
     「浪費」として1試合ごとにカウントする。ko_search呼び出しは記録専用で、返り値は
     一切書き換えず意思決定に影響させない。
  3. **veto ONモード(`--config abl_5_full_og_hammer`)**: production 自体が veto を適用する。
     `ml_policy_agent._try_hammer_veto` を record-only にラップし、実際に発火(None以外を
     返した)回数も追加で数える。同様に `_try_hammer_redirect`(ターゲットをアクティブ→
     ベンチへ振り替える `hammer_veto.redirect_target`)の発火回数も `redirect_fired` に数える。
     浪費観測(2.と同じロジック)は config に関わらず常に行う
     (shadow/ON の両方で同じ「浪費の発生頻度」を横並びで見れるようにするため)。
  4. 試合終了後、実際に打たれたクラッシュハンマーの回数(`hammer_played`。veto後の最終行動を
     観測。ONモードでは浪費分が差し替わって減るはず)、勝敗、`win_condition` を1行のJSONLに
     まとめる。`--analyze` で複数ワーカーの出力をまとめて集計できる。

production(`sample_submission/`)・`cg/`・`data/` は一切変更しない(モジュール属性の実行時
差し替えのみ。ファイルは無傷)。`kaggle_replays/measurement/` の `agents.py`/`runner.py` の
作法(`_c0_headroom.py`/`_valueood_ogerpon.py` を参考)を踏襲する。
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    # Windows コンソールの既定コードページ(cp932等)で日本語JSON出力が文字化けするのを防ぐ。
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import AreaType, SelectType  # noqa: E402
from ptcg_ai.learning import encoder as ENC  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: E402
from ptcg_ai.search import ko_search, lethal_simple  # noqa: E402

_WDIR = _SUB / "ptcg_ai" / "learning"
_META_DIR = _ROOT / "kaggle_replays" / "meta_analysis"
_OG_DECK = _ROOT / "kaggle_replays" / "deck_search" / "candidates_ogerpon_stage3" / "g2top2_v032.csv"
_OG_WEIGHTS = _WDIR / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"

_GAME_OFFSET = 4_800_000
_KO_SEARCH_TIME_LIMIT_MS = 100  # production の hammer_veto 既定(time_limit_ms=100)と同じ

ROWS: list[dict] = []
STATS: dict[str, int] = {}
CTX: dict = {
    "game_id": 0, "opp_arch": "?", "own_index": 0, "record": False,
    "hammer_played": 0, "waste": 0, "veto_fired": 0, "redirect_fired": 0,
    "target_selects": 0, "active_target_waste": 0, "shadow_cached": 0,
    # --lethal-stats 用: 1試合分の lethal_simple.search 壁時計(ms、実際に探索した回のみ)、
    # ゲートで即 None になった呼び出し回数、自分側だけの件数カウンタ。
    "lethal_ms": [], "lethal_gated": 0, "lethal_counts": {},
}

# lethal 統計(--lethal-stats)で集計する lethal_simple.get_stats() のカウンタ名。
_LETHAL_COUNTERS = ("searches", "found", "timeouts", "node_limit_hits", "verify_rejects")


def _bump(key: str) -> None:
    STATS[key] = STATS.get(key, 0) + 1


def _percentile(values: list[float], q: float) -> float | None:
    """ソート済み最近傍法の分位点(numpy 非依存)。空なら None。"""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 1)


# ---------------------------------------------------------------------------
# フィールド(kaggle_replays/rl/train_league.FIELD_PRESETS["mix"] をそのまま使う)
# ---------------------------------------------------------------------------

def _load_mix_field() -> list[tuple[str, float, str, str]]:
    """torch 依存で重いため import は呼び出し時まで遅延する(`_c0_headroom.py` と同じ手口)。"""
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
    """フィールド定義(アーキ, share, 重み接尾辞, デッキdir)から相手を1体選ぶ(share比例抽出)。"""
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


# ---------------------------------------------------------------------------
# record-only 観測フック(意思決定には一切影響させない)
# ---------------------------------------------------------------------------

def _is_own_main_decision(obs, action) -> tuple | None:
    """自分(og_v032)の MAIN・単一選択の決定でなければ None、そうなら (sel, cur, idx) を返す。"""
    sel, cur = obs.select, obs.current
    if sel is None or cur is None or cur.result != -1:
        return None
    if cur.yourIndex != CTX["own_index"] or not CTX["record"]:
        return None
    if sel.type != SelectType.MAIN or len(action) != 1:
        return None
    idx = action[0]
    if not (0 <= idx < len(sel.option)):
        return None
    return sel, cur, idx


def _resolved_card_id(sel, cur, idx) -> int | None:
    try:
        return ENC._resolve_card_id(sel.option[idx], cur)
    except Exception:  # noqa: BLE001 - 解決失敗は記録スキップ
        return None


def _observe_waste(obs, action, config) -> None:
    """veto適用"前"の選択がクラッシュハンマーの浪費条件を満たすかを record-only で判定する。

    判定は production の `_try_hammer_veto`(1120 対応版)と同じ3条件
    (対象カード・KO可能・guardが掛からない)。ko_search の呼び出しは記録専用で、
    ここでの結果は行動選択に一切反映しない(呼び出し側は常に action をそのまま使う)。
    """
    hit = _is_own_main_decision(obs, action)
    if hit is None:
        return
    sel, cur, idx = hit
    card_id = _resolved_card_id(sel, cur, idx)
    if card_id != MA._CRUSHING_HAMMER_ID:
        return
    try:
        factory = MA._model_hidden_state_factory(obs, config)
        deadline = time.perf_counter() + _KO_SEARCH_TIME_LIMIT_MS / 1000.0
        ko = ko_search.can_ko_this_turn(obs, factory, {}, deadline)
    except Exception:  # noqa: BLE001 - 観測失敗は対戦を止めない
        _bump("ko_search_error")
        return
    if not ko:
        return
    try:
        blocked = MA._hammer_veto_guard_blocks(cur, MA._CRUSHING_HAMMER_ID)
    except Exception:  # noqa: BLE001
        _bump("guard_error")
        return
    if blocked:
        return  # ベンチに正当な対象あり=真の浪費ではない
    CTX["waste"] += 1


def _observe_active_target_waste(obs, final_action, config) -> None:
    """最終行動が「このターン倒せる相手アクティブからエネを剥がす」ままかを record-only で数える。

    実ラダー監査(`_audit_hammer_replay.py`)の"無駄撃ち"と同じ定義(同ターンに自分の攻撃で
    KOする相手からエネを剥がす)を、ターゲット選択の**確定後**に適用する。`waste`(播き時観測)
    と違い、veto/redirect 適用**後**の結果を見るので、shadow と redirect ON を横並びで比較して
    「無駄撃ちが減ったか」を直接見られる。ko_search 呼び出しは記録専用(意思決定に影響しない)。
    """
    sel, cur = obs.select, obs.current
    if sel is None or cur is None or cur.result != -1 or not CTX["record"]:
        return
    if cur.yourIndex != CTX["own_index"] or len(final_action) != 1:
        return
    try:
        card_id = MA._hammer_target_select_card_id(sel, [MA._CRUSHING_HAMMER_ID])
    except Exception:  # noqa: BLE001 - 観測失敗は対戦を止めない
        _bump("target_select_error")
        return
    if card_id is None:
        return  # クラッシュハンマーの「剥がす対象を選ぶ select」ではない
    idx = final_action[0]
    if not (0 <= idx < len(sel.option)):
        return
    CTX["target_selects"] += 1
    if getattr(sel.option[idx], "area", None) != AreaType.ACTIVE:
        return  # ベンチを狙った=無駄撃ちではない
    try:
        hv_config = dict((config or {}).get("hammer_veto") or {})
        hv_config.setdefault("time_limit_ms", _KO_SEARCH_TIME_LIMIT_MS)
        if MA._hammer_redirect_can_ko(obs, card_id, hv_config, config):
            CTX["active_target_waste"] += 1
    except Exception:  # noqa: BLE001
        _bump("target_ko_error")


def _count_hammer_played(obs, action) -> None:
    """veto適用"後"(=実際にゲーム内で起きた)クラッシュハンマーPLAYの回数を数える。"""
    hit = _is_own_main_decision(obs, action)
    if hit is None:
        return
    sel, cur, idx = hit
    if _resolved_card_id(sel, cur, idx) == MA._CRUSHING_HAMMER_ID:
        CTX["hammer_played"] += 1


def _install_lethal_probe() -> None:
    """`lethal_simple.search` を record-only にラップし、**自分側だけ**のリーサル探索統計を集める。

    件数系(searches / found / timeouts / node_limit_hits / verify_rejects)は production の
    `lethal_simple.get_stats()` を呼び出しの前後で引き算して、その1回分の差分だけを足す。
    こうする理由は2つ:

    - `get_stats()` はプロセス共有の累積カウンタなので、**同じプロセスで動く相手エージェント**
      (abl_5_full 固定、time_limit_ms=100)の探索が混ざってしまう。差分方式なら
      `obs.current.yourIndex` で自分側の呼び出しだけを選んで積める。
    - 壁時計の分布(p50/p95)は `get_stats()` が持たない(total/max のみ)ので自前で計る。

    ゲート(サイド残・自ターン・隠れ状態)で即 None になる呼び出しは探索していない(0ms)ので、
    `searches` が増えなかった回は `lethal_gated` に分けて時間分布からは外す。
    返り値は素通しなので意思決定には一切影響しない。
    """
    original_search = lethal_simple.search

    def _is_own_call(context) -> bool:
        try:
            obs = (context or {}).get("observation")
            return obs is not None and obs.current is not None and obs.current.yourIndex == CTX["own_index"]
        except Exception:  # noqa: BLE001 - 観測失敗は計測スキップ(対戦は止めない)
            return False

    def wrapped_search(state, legal_actions, context):
        if not (CTX["record"] and _is_own_call(context)):
            return original_search(state, legal_actions, context)
        before = lethal_simple.get_stats()
        started = time.perf_counter()
        try:
            return original_search(state, legal_actions, context)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            after = lethal_simple.get_stats()
            if after["searches"] > before["searches"]:
                CTX["lethal_ms"].append(round(elapsed_ms, 1))
                for key in _LETHAL_COUNTERS:
                    CTX["lethal_counts"][key] += int(after[key]) - int(before[key])
            else:
                CTX["lethal_gated"] += 1

    # `ml_policy_agent._try_lethal` は `_SEARCH_MODULES` 経由でモジュール属性を都度引くので、
    # モジュール属性の差し替えだけでラップが効く(production ファイルは無傷)。
    lethal_simple.search = wrapped_search


def _install_diagnostic_hooks(lethal_stats: bool = False) -> None:
    """production の内部関数3つを record-only にラップする(意思決定は一切変えない)。

    1) `_apply_action_vetoes`: veto適用"前"の行動を横取りし `_observe_waste` を呼ぶ
       (config に関わらず常に計測=shadow/ON 両方で横並びの「浪費の発生頻度」が取れる)。
    2) `_try_hammer_veto`: 実際に veto が発火した(None以外を返した)回数を数える
       (hammer_veto 無効の config、または `veto_mode="shadow"` では常に None なので0になる)。
    3) `_make_hammer_ko_cache`: `veto_mode="shadow"`(redirect単独検証)でも KO判定キャッシュの
       構築自体は行われることを見える化する record-only カウンタ(`shadow_cached`、任意)。
       veto発火とは独立に「播き時のキャッシュ構築が起きた回数」を数えるので、
       `veto_fired=0` かつ `shadow_cached>0` なら「shadowが意図どおり動いている」ことの
       直接証拠になる。
    4) `lethal_stats=True`(--lethal-stats)のときだけ `lethal_simple.search` も
       record-only にラップして壁時計を集める(`_install_lethal_probe`)。
    プロセス内で一度だけ呼ぶ。production コード自体は書き換えない
    (モジュール属性の実行時差し替えのみ、ファイルは無傷)。
    """
    original_apply = MA._apply_action_vetoes
    original_veto = MA._try_hammer_veto
    original_redirect = getattr(MA, "_try_hammer_redirect", None)
    original_make_cache = getattr(MA, "_make_hammer_ko_cache", None)

    def wrapped_apply(obs, action, config=None):
        if CTX["record"]:
            _observe_waste(obs, action, config)
        final_action = original_apply(obs, action, config=config)
        if CTX["record"]:
            _observe_active_target_waste(obs, final_action, config)
        return final_action

    def wrapped_veto(obs, chosen_action, config=None):
        result = original_veto(obs, chosen_action, config=config)
        if CTX["record"] and result is not None:
            cur = obs.current
            if cur is not None and cur.yourIndex == CTX["own_index"]:
                CTX["veto_fired"] += 1
        return result

    def wrapped_redirect(obs, chosen_action, config=None):
        result = original_redirect(obs, chosen_action, config=config)
        if CTX["record"] and result is not None:
            cur = obs.current
            if cur is not None and cur.yourIndex == CTX["own_index"]:
                CTX["redirect_fired"] += 1
        return result

    def wrapped_make_cache(obs, card_id, can_ko):
        result = original_make_cache(obs, card_id, can_ko)
        if CTX["record"] and result is not None:
            cur = obs.current
            if cur is not None and cur.yourIndex == CTX["own_index"]:
                CTX["shadow_cached"] += 1
        return result

    MA._apply_action_vetoes = wrapped_apply
    MA._try_hammer_veto = wrapped_veto
    if lethal_stats:
        _install_lethal_probe()
    if original_redirect is not None:
        # ターゲット・リダイレクト(hammer_veto.redirect_target)の発火回数。
        # `_apply_action_vetoes` はモジュール属性を都度参照するのでラップが効く。
        MA._try_hammer_redirect = wrapped_redirect
    if original_make_cache is not None:
        # `_try_hammer_veto` はモジュールのグローバル名として `_make_hammer_ko_cache` を呼ぶため、
        # モジュール属性を差し替えれば(`_try_hammer_veto`/`_apply_action_vetoes` と同じ手口)
        # 呼び出し側のコードを変えずにラップが効く。
        MA._make_hammer_ko_cache = wrapped_make_cache


def make_probe(inner):
    """production agent(og_v032側)に、実際に打たれたハンマー回数の観測を重ねる。

    action はそのまま返す(記録は副作用のみ、意思決定には一切影響しない)。
    """
    def probe(obs):
        action = inner(obs)
        if CTX["record"]:
            _count_hammer_played(obs, action)
        return action

    return probe


# ---------------------------------------------------------------------------
# 収集
# ---------------------------------------------------------------------------

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


def _summarize(rows: list[dict]) -> dict:
    """勝率・ハンマー使用/浪費/veto発火の頻度・deckout負け率(自分/相手別)を集計する。"""
    n = len(rows)
    if n == 0:
        return {
            "games": 0, "own_win_rate": None,
            "hammer_played_total": 0, "hammer_played_per_game": None,
            "waste_detected_total": 0, "waste_detected_per_game": None,
            "games_with_waste_frac": None,
            "veto_fired_total": 0, "veto_fired_per_game": None,
            "redirect_fired_total": 0, "redirect_fired_per_game": None,
            "shadow_cached_total": 0, "shadow_cached_per_game": None,
            "target_selects_total": 0, "active_target_waste_total": 0,
            "active_target_waste_frac": None,
            "own_deckout_loss_rate": None, "opp_deckout_loss_rate": None,
            "error_rate": None,
            "lethal": None,
        }
    wins = [r for r in rows if r.get("own_win") is True]
    losses = [r for r in rows if r.get("own_win") is False]
    own_deckout = sum(1 for r in losses if r.get("win_condition") == "deckout")
    opp_deckout = sum(1 for r in wins if r.get("win_condition") == "deckout")
    hammer_total = sum(int(r.get("hammer_played", 0)) for r in rows)
    waste_total = sum(int(r.get("waste_detected", 0)) for r in rows)
    veto_total = sum(int(r.get("veto_fired", 0)) for r in rows)
    redirect_total = sum(int(r.get("redirect_fired", 0)) for r in rows)
    shadow_cached_total = sum(int(r.get("shadow_cached", 0)) for r in rows)
    target_total = sum(int(r.get("target_selects", 0)) for r in rows)
    active_waste_total = sum(int(r.get("active_target_waste", 0)) for r in rows)
    return {
        "games": n,
        "own_win_rate": round(len(wins) / n, 4),
        "hammer_played_total": hammer_total,
        "hammer_played_per_game": round(hammer_total / n, 4),
        "waste_detected_total": waste_total,
        "waste_detected_per_game": round(waste_total / n, 4),
        "games_with_waste_frac": round(
            sum(1 for r in rows if int(r.get("waste_detected", 0)) > 0) / n, 4
        ),
        "veto_fired_total": veto_total,
        "veto_fired_per_game": round(veto_total / n, 4),
        # ターゲット・リダイレクト(アクティブ→ベンチ)が実際に差し替えた回数。
        "redirect_fired_total": redirect_total,
        "redirect_fired_per_game": round(redirect_total / n, 4),
        # `veto_mode="shadow"` でも KO判定キャッシュの構築自体は起きたことの見える化(任意)。
        # veto_fired=0 でも shadow_cached>0 なら「shadowが播き時の判定を止めていない」証拠。
        "shadow_cached_total": shadow_cached_total,
        "shadow_cached_per_game": round(shadow_cached_total / n, 4),
        # コイン成功でターゲットを選んだ回数(=剥がしが実際に起きた回数)。
        "target_selects_total": target_total,
        # そのうち「このターン倒せる相手アクティブ」を狙ったまま確定した回数(実ラダー監査と同じ定義)。
        "active_target_waste_total": active_waste_total,
        "active_target_waste_frac": round(active_waste_total / target_total, 4) if target_total else None,
        # 自分が負け かつ 自分の山札切れ(win_condition='deckout') / 全試合。
        "own_deckout_loss_rate": round(own_deckout / n, 4),
        # 自分が勝ち かつ 相手の山札切れ / 全試合。
        "opp_deckout_loss_rate": round(opp_deckout / n, 4),
        "error_rate": round(sum(1 for r in rows if r.get("error")) / n, 4),
        # --lethal-stats を付けて収集した行がある場合のみ、リーサル探索の内訳を出す。
        "lethal": _summarize_lethal(rows),
    }


def _summarize_lethal(rows: list[dict]) -> dict | None:
    """ワーカー横断で lethal_simple の統計を合算し、壁時計の p50/p95 も出す。

    `--lethal-stats` を付けずに収集した行には lethal_* が無いので None を返す
    (既存の集計出力を壊さない)。timeout_rate / verify_reject_rate は「探索1回あたり」。
    """
    lethal_rows = [r for r in rows if "lethal_searches" in r]
    if not lethal_rows:
        return None
    totals = {k: sum(int(r.get(f"lethal_{k}", 0)) for r in lethal_rows) for k in _LETHAL_COUNTERS}
    times: list[float] = []
    for row in lethal_rows:
        times.extend(float(x) for x in (row.get("lethal_ms") or []))
    searches = totals["searches"]
    return {
        "games": len(lethal_rows),
        **{f"{k}_total": v for k, v in totals.items()},
        # ゲート(サイド残・自ターン・隠れ状態)で即 None になり探索に入らなかった呼び出し。
        "gated_total": sum(int(r.get("lethal_gated", 0)) for r in lethal_rows),
        "searches_per_game": round(searches / len(lethal_rows), 3),
        "found_rate": round(totals["found"] / searches, 4) if searches else None,
        "timeout_rate": round(totals["timeouts"] / searches, 4) if searches else None,
        "node_limit_rate": round(totals["node_limit_hits"] / searches, 4) if searches else None,
        "verify_reject_rate": round(totals["verify_rejects"] / searches, 4) if searches else None,
        "ms_calls": len(times),
        "ms_p50": _percentile(times, 0.50),
        "ms_p95": _percentile(times, 0.95),
        "ms_max": round(max(times), 1) if times else None,
    }


def collect(args) -> None:
    """og_v032(mixogerpon重み) vs mixフィールドを回し、JSONL(gzip)+サマリを出す。"""
    num_workers = args.num_workers if args.num_workers is not None else args.workers
    if args.games < 1 or num_workers < 1 or not 0 <= args.worker_id < num_workers:
        raise ValueError("--games は1以上、--worker-id は0以上 --num-workers(--workers)未満にしてください")
    local_games = max(1, args.games // num_workers)

    _install_diagnostic_hooks(lethal_stats=args.lethal_stats)

    own_cfg = agents.load_config_copy(args.config)
    hv_own = own_cfg.get("hammer_veto") or {}
    # `veto_mode`(既定 "swap")を config からそのまま透過する。ここでの `mode` ラベルは
    # 従来どおり「hammer_veto.enabled の有無」だけを見る(過去ログとの互換のため意味は変えない)。
    # `veto_mode="shadow"`(redirect単独検証)の場合でも hammer_veto.enabled=True なので
    # mode="veto_on" になる点に注意(実際には veto は発火しない=veto_fired は常に0になるはず)。
    # 区別したい場合は row/summary の "veto_mode" フィールドを見る。
    mode = "veto_on" if hv_own.get("enabled") else "shadow"
    veto_mode = hv_own.get("veto_mode", "swap")
    # `apply_veto=false` は veto_mode="shadow" の別名(R7 の abl_5_full_og_r7 が使う)。
    # veto_mode ラベルだけでは "swap" に見えてしまうので、実効値を別フィールドで残す。
    apply_veto = bool(hv_own.get("apply_veto", True))
    own_cfg["policy_weights_path"] = str(_OG_WEIGHTS)
    own_inner = agents.make_ml_policy_agent(own_cfg)
    probe = make_probe(own_inner)
    own_deck = runner.load_deck(_OG_DECK)

    # ローカル評価専用の是正: `ml_policy_agent._get_deck()` は CWD の deck.csv(=Plan A の
    # フーディン)を読むが、ここで実際に使うのは og_v032。両者が食い違うと
    # `build_dummy_search_state` が「観測とデッキが不整合」で必ず None を返し、
    # `_model_hidden_state_factory`(既定 dummy)を使う **ko_search が常に False** になる
    # =hammer_veto も redirect も一度も発火しない(実測: 8試合の全ターゲット選択で ko=False)。
    # Kaggle 本番は 1プロセス1エージェントで deck.csv = 実際に使うデッキなので、この差は
    # ハーネス側の欠陥。`runner._declare_own_decks`(match_context 経由の estimated 側)と
    # 同じ趣旨の宣言を dummy 側にも与える。production のファイルは変更しない。
    MA._deck_cache = list(own_deck)

    field = _load_mix_field()
    field_rng = random.Random(0xC0FFEE ^ args.worker_id)

    started = time.perf_counter()
    for game_index in range(local_games):
        game_id = _GAME_OFFSET + args.worker_id + game_index * num_workers
        arch, opp_weights_path, opp_deck_path = _pick_opponent(field, game_id, field_rng)
        opp_cfg = agents.load_config_copy("abl_5_full")
        opp_cfg["policy_weights_path"] = opp_weights_path
        opponent = agents.make_ml_policy_agent(opp_cfg)
        opp_deck = runner.load_deck(opp_deck_path)

        own_first = game_index % 2 == 0
        own_index = 0 if own_first else 1
        CTX.update(
            game_id=game_id, opp_arch=arch, own_index=own_index, record=True,
            hammer_played=0, waste=0, veto_fired=0, redirect_fired=0,
            target_selects=0, active_target_waste=0, shadow_cached=0,
            lethal_ms=[], lethal_gated=0,
            lethal_counts={k: 0 for k in _LETHAL_COUNTERS},
        )
        if own_first:
            result = runner.play_game(probe, opponent, own_deck, opp_deck)
        else:
            result = runner.play_game(opponent, probe, opp_deck, own_deck)
        CTX["record"] = False

        own_win = (result.winner == own_index) if result.winner is not None else None
        row = {
            "game_id": game_id,
            "worker_id": args.worker_id,
            "mode": mode,
            "veto_mode": veto_mode,
            "apply_veto": apply_veto,
            "config": args.config,
            "opp_arch": arch,
            "own_index": own_index,
            "own_win": own_win,
            "win_condition": result.primary_win_condition,
            "turns": result.turns,
            "error": result.error,
            "hammer_played": CTX["hammer_played"],
            "waste_detected": CTX["waste"],
            "veto_fired": CTX["veto_fired"],
            "redirect_fired": CTX["redirect_fired"],
            "target_selects": CTX["target_selects"],
            "active_target_waste": CTX["active_target_waste"],
            "shadow_cached": CTX["shadow_cached"],
        }
        if args.lethal_stats:
            row.update({f"lethal_{k}": int(CTX["lethal_counts"].get(k, 0)) for k in _LETHAL_COUNTERS})
            row["lethal_ms"] = list(CTX["lethal_ms"])
            row["lethal_gated"] = CTX["lethal_gated"]
        ROWS.append(row)
        _bump("games")
        if result.error is not None:
            _bump("games_error")
        if own_win is None:
            _bump("games_no_label")

        print(
            "  [w{}] game#{} opp={} own_win={} cond={} hammer={} waste={} veto={} redirect={} "
            "cached={} target={} act_waste={} {:.1f}分".format(
                args.worker_id, game_id, arch, own_win, result.primary_win_condition,
                CTX["hammer_played"], CTX["waste"], CTX["veto_fired"], CTX["redirect_fired"],
                CTX["shadow_cached"], CTX["target_selects"], CTX["active_target_waste"],
                (time.perf_counter() - started) / 60.0,
            ),
            file=sys.stderr, flush=True,
        )

    output = _HERE / f"_diag_og_hammer_{args.tag}_w{args.worker_id}.jsonl.gz"
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        for row in ROWS:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summarize(ROWS)
    summary.update({
        "tag": args.tag, "worker_id": args.worker_id, "num_workers": num_workers,
        "config": args.config, "mode": mode, "veto_mode": veto_mode,
        "apply_veto": apply_veto, "output": str(output),
        "target_games": local_games, "stats": dict(STATS),
    })
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="og_v032(クラッシュハンマー4枚)のハンマー浪費veto診断(発火率+deckout率)を測定します。"
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
        help="出力ファイル識別子(実ファイル名: _diag_og_hammer_<tag>_w<worker-id>.jsonl.gz)",
    )
    parser.add_argument(
        "--config", default="abl_5_full",
        help="自分側(og_v032)の config名。既定 abl_5_full=shadowモード(veto無効・現行挙動)。"
             "abl_5_full_og_hammer を指定すると veto ON側を計測できる"
             "(config の hammer_veto.enabled で自動判定し、結果の mode フィールドに記録する)。",
    )
    parser.add_argument(
        "--lethal-stats", action="store_true",
        help="lethal_simple の統計(searches/found/timeouts/node_limit_hits/verify_rejects と"
             "壁時計 p50/p95)を1試合ごとに記録する。--analyze 時はワーカー横断で合算される。"
             "record-only のラップのみで意思決定は変えない",
    )
    parser.add_argument(
        "--analyze", nargs="+", metavar="JSONL",
        help="収集せず、指定したJSONL(gzip可・複数ファイル可)を集計する",
    )
    args = parser.parse_args()
    if args.analyze:
        rows = _read_rows(args.analyze)
        summary = _summarize(rows)
        summary["files"] = args.analyze
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        collect(args)


if __name__ == "__main__":
    main()
