"""hand_damage_guard の Iteration 3(``substitute_only``)向け shadow スモーク。

`_diag_og_hammer.py` の shadow 計測(production は新config=I3 で実際に駆動しつつ、
旧挙動=I1 を record-only で並行評価する手口)を流用する。production コード/`cg/`/`data/` は
一切変更しない(モジュール属性の実行時差し替えのみ)。

やること:
  1. own agent は og_r8(I3、``hand_damage_guard.substitute_only: true``、production 既定重み
     climb)で実際に対戦を駆動する。
  2. 相手は `hand_damage_guard` を持たない `abl_5_full`(=対戦全体で `_GUARD_STATS` を own 側の
     判定だけに保つための選択)。トリガー確率を上げるため、own_hand 型(861=メガユキメノコex)
     と opp_hand 型(743、フーディンのハンドパワー=ミラー)を交互に相手アーキとして選ぶ。
  3. `_apply_action_vetoes` を record-only にラップし、own 側の決定ごとに **I1
     (``substitute_only`` 無し、他は同一 sources)なら何を選んだか**を並行評価する
     (`_GUARD_STATS` への副作用は呼び出し前後で保存/復元して打ち消す=production の I3 統計を
     汚さない)。I1 の差し替え結果を次のいずれかに分類する:
       - "none": 発火しなかった(閾値未達 / 相手の場に対象が居ない)
       - "judge" / "boss": 代替サポートへの差し替え(I3 でも起きるはずの正当な差し替え)
       - "pure_block": 代替でも何でもない手(END・攻撃など)への差し替え=「封じるだけ封じて
         ドローエンジンを止める」退行の有害枝そのもの
  4. production(I3)の実際の介入内訳は `ml_policy_agent.get_guard_stats()` をゲームごとに
     reset/読み取りして集計する(``hand_damage_guard_no_substitute`` が I1 の "pure_block" に
     相当する決定点で不介入になったことを示す)。

期待される確認事項(このスクリプトの目的):
  I3(production)側の実際の介入(``hand_damage_guard_to_judge`` + ``hand_damage_guard_to_other``)
  のうち、ジャッジマン/ボスの指令以外への差し替えは **常にゼロ**(``substitute_only`` の実装上
  の不変条件だが、実対戦の decision 経路でも壊れていないことを確認する)。あわせて I1 shadow の
  "pure_block" 件数(=退行の有害枝だったはずの決定点)を報告する。

使い方:
    python kaggle_replays/_diag_hand_guard_i3.py --games 8
    python kaggle_replays/_diag_hand_guard_i3.py --games 8 --tag smoke
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import OptionType, SelectType  # noqa: E402
from ptcg_ai.learning import encoder as ENC  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: E402

_WDIR = _SUB / "ptcg_ai" / "learning"
_META_DIR = _ROOT / "kaggle_replays" / "meta_analysis"
_OWN_DECK = _SUB / "deck.csv"                              # production 実使用デッキ(Plan A)
_OWN_WEIGHTS = _WDIR / "policy_weights_alakazam_rl_climb.json"  # production 既定(climb)

# own_hand 型(861, メガユキメノコex)/opp_hand 型(743, フーディンのハンドパワー=ミラー)を
# 交互に踏ませるための相手アーキ2種。相手側 config は `hand_damage_guard` を持たない
# `abl_5_full` に固定し、`_GUARD_STATS` が own 側の判定だけを反映するようにする。
_OPPONENTS = [
    ("mega_froslass_ex", _WDIR / "policy_weights_mega_froslass_ex_g2.json",
     _META_DIR / "archetype_decks" / "mega_froslass_ex" / "01.csv"),
    ("alakazam_mirror", _WDIR / "policy_weights_alakazam_g2.json",
     _META_DIR / "archetype_decks" / "alakazam" / "01.csv"),
]

_GAME_OFFSET = 4_900_000

CTX: dict = {"own_index": 0, "record": False}
ROWS: list[dict] = []

_SHADOW_KINDS = ("none", "judge", "boss", "pure_block")


def _fresh_shadow_counts() -> dict[str, int]:
    return {k: 0 for k in _SHADOW_KINDS}


def load_i1_config(r8_config: dict) -> dict:
    """og_r8 の sources を引き継ぎつつ ``substitute_only`` を外した I1 相当の設定を作る。

    og_r8 自体(設定ファイル)には触れず、メモリ上でコピーして差分だけ変更する。
    """
    import copy

    guard = copy.deepcopy(r8_config["hand_damage_guard"])
    guard.pop("substitute_only", None)
    guard.pop("_comment", None)
    return {"hand_damage_guard": guard}


def _classify_replacement(obs, action: list[int] | None) -> str:
    """`_try_hand_damage_guard` の戻り値を "none"/"judge"/"boss"/"pure_block" に分類する。"""
    if action is None:
        return "none"
    sel = obs.select
    if sel is None or len(action) != 1 or not (0 <= action[0] < len(sel.option)):
        return "pure_block"  # 想定外の形=安全側で「代替ではない」扱い
    opt = sel.option[action[0]]
    if opt.type != OptionType.PLAY:
        return "pure_block"
    card_id = ENC._resolve_card_id(opt, obs.current)
    if card_id == MA._JUDGE_ID:
        return "judge"
    if card_id == MA._BOSS_ORDER_ID:
        return "boss"
    return "pure_block"


def _install_shadow_hook(i1_config: dict) -> None:
    """`_apply_action_vetoes` を record-only にラップし、I1 shadow 判定を並行実行する。

    production(I3, 呼び出し側が渡す config)の意思決定には一切影響しない。I1 shadow 呼び出しの
    `_GUARD_STATS` への副作用は呼び出し前後で保存/復元して打ち消す(production の I3 統計を
    このラップで汚さないため)。
    """
    original_apply = MA._apply_action_vetoes

    def wrapped_apply(obs, action, config=None):
        if CTX["record"] and obs.current is not None and obs.current.yourIndex == CTX["own_index"]:
            try:
                before = dict(MA._GUARD_STATS)
                shadow = MA._try_hand_damage_guard(obs, list(action), config=i1_config)
                after_fired = MA._GUARD_STATS.get("hand_damage_guard_fired", 0)
                # 呼び出し前後で副作用を打ち消す(I1 shadow の計測を production 統計に混ぜない)。
                MA._GUARD_STATS.clear()
                MA._GUARD_STATS.update(before)
                if after_fired - before.get("hand_damage_guard_fired", 0) > 0:
                    CTX["i1_shadow_fired"] += 1
                    kind = _classify_replacement(obs, shadow)
                    CTX["i1_shadow_kinds"][kind] += 1
            except Exception:  # noqa: BLE001 - 観測失敗で対戦を止めない
                CTX["i1_shadow_errors"] += 1
        return original_apply(obs, action, config=config)

    MA._apply_action_vetoes = wrapped_apply


def collect(args) -> None:
    r8_config = agents.load_config_copy("abl_5_full_og_r8")
    assert r8_config.get("hand_damage_guard", {}).get("substitute_only") is True, (
        "abl_5_full_og_r8.json に substitute_only=true が無い(config が想定と違う)"
    )
    i1_config = load_i1_config(r8_config)
    r8_config["policy_weights_path"] = str(_OWN_WEIGHTS)
    own_inner = agents.make_ml_policy_agent(r8_config)
    own_deck = runner.load_deck(_OWN_DECK)
    # `_get_deck()`(pipeline/lethal_search の隠れ状態スタブが参照する)を own_deck に固定。
    MA._deck_cache = list(own_deck)

    _install_shadow_hook(i1_config)

    started = time.perf_counter()
    for game_index in range(args.games):
        game_id = _GAME_OFFSET + game_index
        arch, opp_weights_path, opp_deck_path = _OPPONENTS[game_index % len(_OPPONENTS)]
        opp_cfg = agents.load_config_copy("abl_5_full")  # hand_damage_guard キー無し=不発
        opp_cfg["policy_weights_path"] = str(opp_weights_path)
        opponent = agents.make_ml_policy_agent(opp_cfg)
        opp_deck = runner.load_deck(opp_deck_path)

        own_first = game_index % 2 == 0
        own_index = 0 if own_first else 1
        CTX.update(
            own_index=own_index, record=True,
            i1_shadow_fired=0, i1_shadow_kinds=_fresh_shadow_counts(), i1_shadow_errors=0,
        )
        MA.reset_guard_stats()

        if own_first:
            result = runner.play_game(own_inner, opponent, own_deck, opp_deck)
        else:
            result = runner.play_game(opponent, own_inner, opp_deck, own_deck)
        CTX["record"] = False

        i3_stats = MA.get_guard_stats()
        own_win = (result.winner == own_index) if result.winner is not None else None
        row = {
            "game_id": game_id,
            "opp_arch": arch,
            "own_first": own_first,
            "own_win": own_win,
            "win_condition": result.primary_win_condition,
            "turns": result.turns,
            "error": result.error,
            "i3_fired": i3_stats["hand_damage_guard_fired"],
            "i3_to_judge": i3_stats["hand_damage_guard_to_judge"],
            "i3_to_other": i3_stats["hand_damage_guard_to_other"],
            "i3_to_boss": i3_stats["hand_damage_guard_to_boss"],
            "i3_no_substitute": i3_stats["hand_damage_guard_no_substitute"],
            "i1_shadow_fired": CTX["i1_shadow_fired"],
            "i1_shadow_judge": CTX["i1_shadow_kinds"]["judge"],
            "i1_shadow_boss": CTX["i1_shadow_kinds"]["boss"],
            "i1_shadow_pure_block": CTX["i1_shadow_kinds"]["pure_block"],
            "i1_shadow_errors": CTX["i1_shadow_errors"],
        }
        ROWS.append(row)
        print(
            "  game#{} opp={} own_first={} own_win={} cond={} "
            "i3(fired={},judge={},other={},boss={},no_sub={}) "
            "i1_shadow(fired={},judge={},boss={},pure_block={}) {:.1f}分".format(
                game_id, arch, own_first, own_win, result.primary_win_condition,
                row["i3_fired"], row["i3_to_judge"], row["i3_to_other"], row["i3_to_boss"],
                row["i3_no_substitute"], row["i1_shadow_fired"], row["i1_shadow_judge"],
                row["i1_shadow_boss"], row["i1_shadow_pure_block"],
                (time.perf_counter() - started) / 60.0,
            ),
            file=sys.stderr, flush=True,
        )

    output = _HERE / f"_diag_hand_guard_i3_{args.tag}.jsonl.gz"
    with gzip.open(output, "wt", encoding="utf-8") as stream:
        for row in ROWS:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    i3_pure_block_events = sum(r["i3_to_other"] - r["i3_to_boss"] for r in ROWS)
    # i3_to_judge はジャッジ以外にはなり得ない(step1)ので pure_block 判定は to_other 側だけでよい。
    summary = {
        "games": len(ROWS),
        "output": str(output),
        "i3_fired_total": sum(r["i3_fired"] for r in ROWS),
        "i3_to_judge_total": sum(r["i3_to_judge"] for r in ROWS),
        "i3_to_boss_total": sum(r["i3_to_boss"] for r in ROWS),
        "i3_no_substitute_total": sum(r["i3_no_substitute"] for r in ROWS),
        "i3_non_substitute_intervention_total": i3_pure_block_events,  # これが目標値=0
        "i1_shadow_fired_total": sum(r["i1_shadow_fired"] for r in ROWS),
        "i1_shadow_judge_total": sum(r["i1_shadow_judge"] for r in ROWS),
        "i1_shadow_boss_total": sum(r["i1_shadow_boss"] for r in ROWS),
        "i1_shadow_pure_block_total": sum(r["i1_shadow_pure_block"] for r in ROWS),
        "i1_shadow_errors_total": sum(r["i1_shadow_errors"] for r in ROWS),
        "errors": sum(1 for r in ROWS if r["error"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hand_damage_guard I3(substitute_only)の shadow スモーク"
                     "(I1比で「代替なしで封じるだけ」介入がゼロになることを確認する)。"
    )
    parser.add_argument("--games", type=int, default=8, help="総試合数(既定8=4試合ペア)")
    parser.add_argument("--tag", default="smoke", help="出力ファイル識別子")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
