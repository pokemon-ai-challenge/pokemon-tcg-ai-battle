"""指定モデルで対戦させ、勝敗を使わない密な診断指標(「機械的に判定できる明確なミス」)を測る。

競技ポケカのトップ1%プレイヤーが Kaggle Discussion で挙げた3つのミス categories を計測する:

- **指標A: KO機会の逸失** — 相手のバトル場を倒せるワザが選択肢に出たターンに、一度も攻撃しなかった回数。
- **指標B: 過剰エネ付け**(ミスと断定しない、率として報告) — 既にそのポケモンの最大コストのワザを
  打てるだけのエネルギーが付いているのに、さらにエネルギーを付けた回数。
- **指標C: デッキ切れ負け** — 負けた試合のうち、deckCount==0 で終わった割合。

``eval_matrix.py``(モデル x 相手の勝率行列)の相手解決・存在確認・並列化の作法を踏襲するが、
``collect_pool.parallel_collect_pool`` は特徴量しか記録せず observation を見られないため使えない。
本モジュールは cg.game.battle_start/battle_select/battle_finish を直接叩く自前のゲームループを持つ
(``collect_parallel._play_one`` のロジックはコピーしていない。判断ロジックは同型だが、盤面から
ミス指標を計測するための追加フックが本質的に必要なため独立実装)。

学習側・相手ともに ``PolicyModel.select_option``(argmax)を使う。温度サンプリングは行わない
(argmax は決定的なので、``--seed0`` は cg エンジン内部のシャッフル乱数を制御しない
= ``collect_pool.py`` の docstring と同じ注意が当てはまる。実質的には試合を識別するための
メタデータ以上の意味を持たない)。

並列化はプロセスのみ(cg はプロセス全体に1つの対局状態を持つため、スレッドでは共有できない)。

出力は kaggle_replays/value_net_probe/ 配下(測定結果の置き場を eval_matrix.py と揃える)。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pools  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

OUT_DIR = _ROOT / "kaggle_replays" / "value_net_probe"
MAX_STEPS = 3000  # collect_parallel.MAX_STEPS と同じ値(異常終局の安全弁)。

# セル(モデル x 相手)ごとの seed 空間の間隔。argmax のみを使う本モジュールでは python 側の乱数は
# 使わない(方策はサンプリングしない)ため、この値は衝突回避の意味しか持たない
# (cg エンジン内部のシャッフル乱数は python の seed では制御できない。collect_pool.py の
# docstring 参照)。
CELL_SEED_GAP = 1_000_000


# ----------------------------------------------------------------------
# Wilson区間 / CLI引数パース(eval_matrix.py と同じ作法を踏襲した独立実装。
# pools.py のレジストリ定義そのものは絶対にコピーしない)。
# ----------------------------------------------------------------------
def wilson_interval(num: int, den: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval(両側)。"""
    if den == 0:
        return 0.0, 1.0
    p = num / den
    d = 1 + z * z / den
    c = p + z * z / (2 * den)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * den)) / den)
    lo = max(0.0, (c - m) / d)
    hi = min(1.0, (c + m) / d)
    return lo, hi


def rate_ci(num: int, den: int) -> dict:
    """num/den の生カウント・率・Wilson95%区間をまとめる。den==0 なら率は NaN。"""
    if den == 0:
        return {"num": num, "den": den, "rate": float("nan"), "ci95": [float("nan"), float("nan")]}
    lo, hi = wilson_interval(num, den)
    return {"num": num, "den": den, "rate": num / den, "ci95": [lo, hi]}


def _parse_kv_list(s: str, label: str) -> list[tuple[str, str]]:
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
    """相手名リスト -> pools.build_opponents() の戻り値(name, weights_path_or_None, deck)。"""
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"--opponents に重複がある: {dupes}")
    unknown = [n for n in names if n not in pools.LEARNER_REGISTRY]
    if unknown:
        available = ", ".join(sorted(pools.LEARNER_REGISTRY))
        raise ValueError(f"未知の相手名: {unknown}. 利用可能な名前(pools.LEARNER_REGISTRY): {available}")
    return pools.build_opponents(names)


# ----------------------------------------------------------------------
# ワーカー: 自前のゲームループ(collect_parallel._play_one はコピーしない、独立実装)。
# ----------------------------------------------------------------------
_W: dict = {}


def _init_worker(weights_path, opp_weights_path, deck_l, deck_o):
    from ptcg_ai.learning.policy_model import PolicyModel

    pm = PolicyModel(weights_path)
    if not pm.is_ready:
        raise RuntimeError(f"learner policy not ready (path?): {weights_path}")
    opp = PolicyModel(opp_weights_path)
    if not opp.is_ready:
        raise RuntimeError(f"opponent policy not ready (path?): {opp_weights_path}")
    _W["pm"] = pm
    _W["opp"] = opp
    _W["deck_l"] = deck_l
    _W["deck_o"] = deck_o


def _is_relevant_select(select, SelectType, SelectContext) -> bool:
    """MAIN(通常ターンの混合選択肢)、または独立した ATTACK 選択(context ATTACK。
    context DISABLE_ATTACK は相手のワザを無効化する対象選択であり、学習側が攻撃する
    選択ではないため除外する)。ATTACK/ATTACH は必ずこのいずれかの select にしか
    出現しない(cg/api.py の SelectType docstring 参照)。
    """
    if select.type == SelectType.MAIN:
        return True
    if select.type == SelectType.ATTACK and select.context == SelectContext.ATTACK:
        return True
    return False


def _play_one_blunder(task):
    """1試合。学習側・相手とも PolicyModel.select_option(argmax)。学習側の判断のみ計測する。"""
    from cg.api import AreaType, CardType, OptionType, SelectContext, SelectType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.board_evaluation import attack_features, energy_requirements
    from ptcg_ai.shared import card_cache

    learner_index, seed = task
    pm = _W["pm"]
    opp = _W["opp"]
    deck0, deck1 = (_W["deck_l"], _W["deck_o"]) if learner_index == 0 else (_W["deck_o"], _W["deck_l"])

    turn_stats: dict[int, dict] = {}
    energy_attach_total = 0
    energy_attach_over = 0
    winner = None
    error = None
    game_length = 0
    final_deck_learner = None
    final_deck_opp = None

    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        return {"error": f"start {start_data.errorType}", "learner_index": learner_index}

    n = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                error = "current None"
                break
            if cur.result != -1:
                winner = cur.result
                game_length = cur.turn
                final_deck_learner = cur.players[learner_index].deckCount
                final_deck_opp = cur.players[1 - learner_index].deckCount
                break
            if n >= MAX_STEPS:
                error = "max_steps"
                break

            select = obs.select
            is_learner_turn = cur.yourIndex == learner_index
            model = pm if is_learner_turn else opp

            if select is None or not select.option:
                action = []
            elif select.maxCount == 1:
                idx = model.select_option(obs)
                idx = idx if idx is not None else 0
                action = [idx]

                if is_learner_turn and _is_relevant_select(select, SelectType, SelectContext):
                    turn = cur.turn
                    entry = turn_stats.setdefault(
                        turn, {"lethal_offered": False, "attacked": False, "energy_attached": False}
                    )
                    learner_state = cur.players[learner_index]
                    opp_state = cur.players[1 - learner_index]
                    attacker = (
                        learner_state.active[0]
                        if learner_state.active and learner_state.active[0] is not None
                        else None
                    )
                    defender = (
                        opp_state.active[0]
                        if opp_state.active and opp_state.active[0] is not None
                        else None
                    )

                    # 指標A: 倒せるワザが選択肢に出ていたか(相手のバトル場が不在なら判定しない)。
                    if attacker is not None and defender is not None:
                        defender_card = card_cache.get_card(defender.id)
                        hand_size = learner_state.handCount
                        for opt in select.option:
                            if opt.attackId is None:
                                continue
                            attack = card_cache.get_attack(opt.attackId)
                            if attack_features.can_ko(
                                attack, attacker, defender,
                                defender_card.weakness, defender_card.resistance, hand_size,
                            ):
                                entry["lethal_offered"] = True
                                break

                    chosen = select.option[idx]
                    if chosen.type == OptionType.ATTACK:
                        entry["attacked"] = True
                    elif chosen.type == OptionType.ATTACH:
                        hand = learner_state.hand or []
                        if chosen.index is not None and 0 <= chosen.index < len(hand):
                            hand_card = hand[chosen.index]
                            hand_card_data = card_cache.get_card(hand_card.id)
                            if hand_card_data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
                                entry["energy_attached"] = True
                                energy_attach_total += 1

                                # 指標B: 付ける「前」の状態で、最もコストが高いワザが既に足りていたか。
                                target = None
                                if chosen.inPlayArea == AreaType.ACTIVE:
                                    target = (
                                        learner_state.active[0]
                                        if learner_state.active and learner_state.active[0] is not None
                                        else None
                                    )
                                elif chosen.inPlayArea == AreaType.BENCH:
                                    bidx = chosen.inPlayIndex
                                    if bidx is not None and 0 <= bidx < len(learner_state.bench):
                                        target = learner_state.bench[bidx]
                                if target is not None:
                                    target_card = card_cache.get_card(target.id)
                                    atk_list = [
                                        card_cache.get_attack(aid) for aid in (target_card.attacks or [])
                                    ]
                                    if atk_list:
                                        max_cost = max(len(a.energies) for a in atk_list)
                                        already_sufficient = any(
                                            len(a.energies) == max_cost
                                            and energy_requirements.is_energy_sufficient(a, target.energies or [])
                                            for a in atk_list
                                        )
                                        if already_sufficient:
                                            energy_attach_over += 1
            else:
                scores = model.score_options(obs)
                nn = len(select.option)
                count = max(select.minCount, min(select.maxCount, nn))
                action = (
                    sorted(range(nn), key=lambda i: scores[i], reverse=True)[:count]
                    if scores else list(range(count))
                )

            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        battle_finish()

    if error is not None:
        return {"error": error, "learner_index": learner_index}

    turns_total = len(turn_stats)
    turns_attacked = sum(1 for v in turn_stats.values() if v["attacked"])
    turns_energy_attached = sum(1 for v in turn_stats.values() if v["energy_attached"])
    lethal_offered_turns = sum(1 for v in turn_stats.values() if v["lethal_offered"])
    lethal_missed_turns = sum(1 for v in turn_stats.values() if v["lethal_offered"] and not v["attacked"])

    return {
        "error": None,
        "learner_index": learner_index,
        "winner": winner,
        "game_length": game_length,
        "final_deck_learner": final_deck_learner,
        "final_deck_opp": final_deck_opp,
        "turns_total": turns_total,
        "turns_attacked": turns_attacked,
        "turns_energy_attached": turns_energy_attached,
        "lethal_offered_turns": lethal_offered_turns,
        "lethal_missed_turns": lethal_missed_turns,
        "energy_attach_total": energy_attach_total,
        "energy_attach_over": energy_attach_over,
    }


def parallel_collect_blunder(weights_path, opp_weights_path, deck_l, deck_o, n_games, seed0, workers):
    """1セル(モデル x 相手)分の試合を集める。戻り値は raw カウントの dict。"""
    if n_games <= 0:
        return _empty_raw()

    tasks = [(g % 2, seed0 + g) for g in range(n_games)]
    with Pool(processes=workers, initializer=_init_worker,
              initargs=(weights_path, opp_weights_path, deck_l, deck_o)) as pool:
        results = pool.map(_play_one_blunder, tasks, chunksize=max(1, n_games // (workers * 4)))
    return _aggregate_raw(results)


def _empty_raw() -> dict:
    return {
        "games": 0, "errors": 0, "valid": 0, "seat0": 0, "seat1": 0,
        "learner_wins": 0, "learner_losses": 0, "draws": 0,
        "turns_total": 0, "turns_attacked": 0, "turns_energy_attached": 0,
        "lethal_offered_turns": 0, "lethal_missed_turns": 0,
        "energy_attach_total": 0, "energy_attach_over": 0,
        "learner_losses_deckout": 0, "opp_losses_deckout": 0,
        "game_length_sum": 0,
    }


def _aggregate_raw(results: list[dict]) -> dict:
    games = len(results)
    errors = sum(1 for r in results if r.get("error") is not None)
    valid_results = [r for r in results if r.get("error") is None]
    valid = len(valid_results)
    seat0 = sum(1 for r in valid_results if r["learner_index"] == 0)
    seat1 = valid - seat0

    learner_wins = sum(1 for r in valid_results if r["winner"] == r["learner_index"])
    learner_losses = sum(1 for r in valid_results if r["winner"] == (1 - r["learner_index"]))
    draws = valid - learner_wins - learner_losses

    learner_losses_deckout = sum(
        1 for r in valid_results
        if r["winner"] == (1 - r["learner_index"]) and r["final_deck_learner"] == 0
    )
    opp_losses_deckout = sum(
        1 for r in valid_results
        if r["winner"] == r["learner_index"] and r["final_deck_opp"] == 0
    )

    return {
        "games": games, "errors": errors, "valid": valid, "seat0": seat0, "seat1": seat1,
        "learner_wins": learner_wins, "learner_losses": learner_losses, "draws": draws,
        "turns_total": sum(r["turns_total"] for r in valid_results),
        "turns_attacked": sum(r["turns_attacked"] for r in valid_results),
        "turns_energy_attached": sum(r["turns_energy_attached"] for r in valid_results),
        "lethal_offered_turns": sum(r["lethal_offered_turns"] for r in valid_results),
        "lethal_missed_turns": sum(r["lethal_missed_turns"] for r in valid_results),
        "energy_attach_total": sum(r["energy_attach_total"] for r in valid_results),
        "energy_attach_over": sum(r["energy_attach_over"] for r in valid_results),
        "learner_losses_deckout": learner_losses_deckout,
        "opp_losses_deckout": opp_losses_deckout,
        "game_length_sum": sum(r["game_length"] for r in valid_results),
    }


def _sum_raw(raws: list[dict]) -> dict:
    """複数セルの raw カウントをフィールドごとに合算する(モデル別集計用)。"""
    out = _empty_raw()
    for raw in raws:
        for k in out:
            out[k] += raw[k]
    return out


def build_report(raw: dict) -> dict:
    """raw カウント -> 指標A/B/C + 補助指標(すべて Wilson95%区間つき)。"""
    valid = raw["valid"]
    avg_game_length = (raw["game_length_sum"] / valid) if valid else float("nan")
    return {
        "raw": raw,
        "metric_a_ko_missed": rate_ci(raw["lethal_missed_turns"], raw["lethal_offered_turns"]),
        "metric_b_overattach": rate_ci(raw["energy_attach_over"], raw["energy_attach_total"]),
        "metric_c_learner_deckout": rate_ci(raw["learner_losses_deckout"], raw["learner_losses"]),
        "metric_c_opponent_deckout_ref": rate_ci(raw["opp_losses_deckout"], raw["learner_wins"]),
        "attacked_turn_rate": rate_ci(raw["turns_attacked"], raw["turns_total"]),
        "energy_attached_turn_rate": rate_ci(raw["turns_energy_attached"], raw["turns_total"]),
        "avg_game_length": avg_game_length,
    }


# ----------------------------------------------------------------------
# Markdown レンダリング
# ----------------------------------------------------------------------
def _fmt_rate(entry: dict) -> str:
    if entry["den"] == 0:
        return f"{entry['num']}/{entry['den']} (分母0)"
    lo, hi = entry["ci95"]
    return f"{entry['num']}/{entry['den']} = {entry['rate']:.3f} [{lo:.3f}, {hi:.3f}]"


def render_markdown(results: dict) -> str:
    models = results["models"]
    opponents = results["opponents"]
    matrix = results["matrix"]
    model_totals = results["model_totals"]
    args = results["args"]

    lines = []
    lines.append("# blunder_metrics 測定結果")
    lines.append("")
    lines.append(
        "勝敗を使わない密な診断指標。「機械的に判定できる明確なミス」として、KO機会の逸失(指標A)・"
        "過剰エネ付け(指標B、率のみ報告・ミスとは断定しない)・デッキ切れ負け(指標C)を計測する。"
        "学習側・相手ともに PolicyModel.select_option(argmax、温度サンプリングなし)。"
    )
    lines.append("")
    lines.append(
        f"- 学習側デッキ: `{args['learner_deck']}`(全モデル共通)\n"
        f"- games_per_cell: {args['games_per_cell']}\n"
        f"- workers: {args['workers']}\n"
        f"- seed0: {args['seed0']}(cg エンジン内部のシャッフル乱数は制御できないため参考値)\n"
        f"- 所要時間: {results['elapsed_sec']:.0f}s"
    )
    lines.append("")
    lines.append(
        "**区間が重なっているものについて優劣を主張しないこと。** すべての率に Wilson 95%区間を付けている。"
    )
    lines.append("")

    lines.append("## 1. 指標A: KO機会の逸失")
    lines.append("")
    lines.append(
        "分母: そのターン中に相手のバトル場を倒せるワザが選択肢に出たターンの数"
        "(相手のバトル場が不在のターンは除外)。分子: そのうち一度も攻撃しなかったターンの数。"
    )
    lines.append("")
    header = "| モデル | " + " | ".join(opponents) + " |"
    sep = "|---|" + "---|" * len(opponents)
    lines.append(header)
    lines.append(sep)
    for m in models:
        cells = [_fmt_rate(matrix[m][o]["metric_a_ko_missed"]) for o in opponents]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("モデル別集計(全相手を合算):")
    lines.append("")
    lines.append("| モデル | 指標A(逸失/機会) |")
    lines.append("|---|---|")
    for m in models:
        lines.append(f"| {m} | {_fmt_rate(model_totals[m]['metric_a_ko_missed'])} |")
    lines.append("")

    lines.append("## 2. 指標B: 過剰エネ付け(率として報告、ミスとは断定しない)")
    lines.append("")
    lines.append(
        "分母: エネルギー付けを選んだ回数。分子: そのうち、既にそのポケモンの最大コストのワザを"
        "打てるだけのエネルギーが付いていた回数。逃げる予定や特定のメタ対策として正当なケースが"
        "あるため、この数値だけでミスと判定しないこと。"
    )
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for m in models:
        cells = [_fmt_rate(matrix[m][o]["metric_b_overattach"]) for o in opponents]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("モデル別集計(全相手を合算):")
    lines.append("")
    lines.append("| モデル | 指標B(過剰付け/エネ付け回数) |")
    lines.append("|---|---|")
    for m in models:
        lines.append(f"| {m} | {_fmt_rate(model_totals[m]['metric_b_overattach'])} |")
    lines.append("")

    lines.append("## 3. 指標C: デッキ切れ負け")
    lines.append("")
    lines.append(
        "分母: 学習側が負けた試合数。分子: そのうち学習側の deckCount==0 だった試合数。"
        "参考として相手側のデッキ切れ負け率(学習側が勝った試合のうち相手 deckCount==0)も示す。"
    )
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for m in models:
        cells = [_fmt_rate(matrix[m][o]["metric_c_learner_deckout"]) for o in opponents]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("参考: 相手側のデッキ切れ負け率(学習側視点、モデル別集計):")
    lines.append("")
    lines.append("| モデル | 学習側デッキ切れ負け率 | 相手デッキ切れ負け率(参考) |")
    lines.append("|---|---|---|")
    for m in models:
        t = model_totals[m]
        lines.append(
            f"| {m} | {_fmt_rate(t['metric_c_learner_deckout'])} | "
            f"{_fmt_rate(t['metric_c_opponent_deckout_ref'])} |"
        )
    lines.append("")

    lines.append("## 4. 補助指標(分母の妥当性確認・行動分布分析との接続用)")
    lines.append("")
    lines.append(
        "| モデル | 相手 | 総ターン数 | 攻撃したターン割合 | エネルギーを付けたターン割合 | "
        "平均試合長(ターン) | 試合数 | エラー試合数 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in models:
        for o in opponents:
            c = matrix[m][o]
            raw = c["raw"]
            lines.append(
                f"| {m} | {o} | {raw['turns_total']} | {_fmt_rate(c['attacked_turn_rate'])} | "
                f"{_fmt_rate(c['energy_attached_turn_rate'])} | {c['avg_game_length']:.1f} | "
                f"{raw['games']} | {raw['errors']} |"
            )
    lines.append("")

    lines.append("## 5. 実行情報")
    lines.append("")
    total_games = sum(matrix[m][o]["raw"]["games"] for m in models for o in opponents)
    total_errors = sum(matrix[m][o]["raw"]["errors"] for m in models for o in opponents)
    lines.append(f"- 総試合数: {total_games}\n- 総エラー試合数: {total_errors}")
    lines.append("")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True,
                     help="'表示名=重みファイル名' のカンマ区切り。重みファイル名は "
                          "sample_submission/ptcg_ai/learning/ 配下の basename。")
    ap.add_argument("--learner-deck", default="alakazam",
                     help="学習側デッキのアーキタイプ名(pools.LEARNER_REGISTRY のキー)。既定 alakazam。")
    ap.add_argument("--opponents", default="alakazam,crustle,marnie_grimmsnarl_ex,archaludon_ex",
                     help="カンマ区切りの相手名(pools.LEARNER_REGISTRY のキー)。")
    ap.add_argument("--games-per-cell", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--out-prefix", default="blunder_metrics",
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

    out_prefix = args.out_prefix
    if "/" not in out_prefix and "\\" not in out_prefix:
        out_prefix = str(OUT_DIR / out_prefix)
    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"models={model_names}", flush=True)
    print(f"opponents={opponent_names}", flush=True)
    print(f"learner_deck={args.learner_deck} deck_csv={learner_deck_csv}", flush=True)
    print(f"games_per_cell={args.games_per_cell} workers={args.workers} seed0={args.seed0}", flush=True)

    n_models = len(models)
    n_opp = len(opponent_names)

    matrix: dict[str, dict[str, dict]] = {}
    t0 = time.time()
    total_cells = n_models * n_opp
    cell_i = 0
    for i, (model_name, weights_path) in enumerate(models):
        matrix[model_name] = {}
        for j, (opp_name, opp_weights_path, deck_o) in enumerate(opponents_built):
            seed = args.seed0 + (i * n_opp + j) * CELL_SEED_GAP
            t_cell = time.time()
            raw = parallel_collect_blunder(
                weights_path, opp_weights_path, deck_l, deck_o,
                n_games=args.games_per_cell, seed0=seed, workers=args.workers,
            )
            report = build_report(raw)
            matrix[model_name][opp_name] = report
            cell_i += 1
            print(
                f"[{cell_i}/{total_cells}] {model_name} vs {opp_name}: "
                f"games={raw['games']} err={raw['errors']} seat0={raw['seat0']} seat1={raw['seat1']} "
                f"metricA={_fmt_rate(report['metric_a_ko_missed'])} "
                f"metricB={_fmt_rate(report['metric_b_overattach'])} "
                f"metricC={_fmt_rate(report['metric_c_learner_deckout'])} "
                f"({time.time() - t_cell:.0f}s)",
                flush=True,
            )

    model_totals = {}
    for model_name, _ in models:
        raws = [matrix[model_name][o]["raw"] for o in opponent_names]
        model_totals[model_name] = build_report(_sum_raw(raws))

    elapsed = time.time() - t0
    print(f"total elapsed {elapsed:.0f}s", flush=True)

    results = {
        "args": vars(args),
        "models": model_names,
        "opponents": opponent_names,
        "matrix": matrix,
        "model_totals": model_totals,
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
