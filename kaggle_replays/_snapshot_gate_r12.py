#!/usr/bin/env python3
"""og_r12(Fix-J ``terastal_rotation`` = 「テラスタル退避」ゲート)の再現ゲート: リプレイの
実局面を固定入力にした PASS/FAIL 判定。

相手方策を回さない**主判定**。実ラダーの実局面をそのまま観測として食わせ、og_r11 と og_r12 で
挙動がどう変わるかを固定する。対戦を回さないので分散が無く、リグレッションの検出に使える
(`_snapshot_gate_r10.py`/`_snapshot_gate_r11.py` と同じ流儀・同じユーティリティ再利用。
あちらは書き換えずに別ファイル)。

**背景**: オーガポンex系の特性「テラスタル」=「このポケモンは、ベンチにいるかぎり、ワザの
ダメージを受けない」(``CardData.tera``)。傷ついたアクティブを、より健康な同族のベンチ個体と
入れ替えるのは、にげるコストが軽い限り原則ノーリスク。

対象(すべて `kaggle_replays/replays/episode-*-replay.json`。デッキは `steps[1]` の60枚回答=
その試合で実際に使ったデッキを使う):

  J1 fires (qualified) 93408551 row134 (T10、決定的)
      アクティブ HP10/エネ4、ベンチに HP150/エネ4 の健康な個体。ブライア+KOで3サイド
      取ったのは正しいが、「にげる(コスト1)→HP150の個体で攻撃」でも同じKO・同じ3サイド
      が取れた上に、HP10の個体をベンチへ退避できた。実際はHP10のまま残し、次ターンに
      70ダメージで落とされて敗北した。

      **重要な発見(仕様の矛盾)**: この局面の相手アクティブは**ドラパルトex(121)本人**
      (実測: 同ターンに実際にファントムダイブでベンチのオーガポンへ6個のダメカン=60ダメージ
      を置いている。ワザ本文の「テラスタルを貫通する」反例が同じエピソード内で実証されている)。
      仕様の条件4(「相手の場にドラパルトex等が居る場合は発火しない」)を字面どおり実装すると、
      「決定的」と明記されたこの局面自体が**既定configでは発火しない**という仕様間の矛盾が
      生じる。ここでは条件4を字面どおり実装した上で、(a) 既定config(skip適用)では不発火
      であること、(b) skipを外した診断用configでは条件1-3だけで正しくRETREATへ差し替わる
      こと、の両方を固定して矛盾を可視化する(詳細はレポート参照)。

  J2 fires 93473767 row129 (T11)
      アクティブHP60/260、ベンチHP240/240・エネ3。KOは元々不可能なので、傷んだ個体を
      ベンチに逃がすのが明確に上。相手アクティブは Mega Froslass ex(861、スキップ対象外)
      なので、条件4との矛盾は無く既定configでそのままRETREATへ差し替わる。

  J3 control (dragapult対面) 93408551 row134 を既定configで実行
      J1の(a)がそのままこの対照を兼ねる(相手がドラパルトexのため発火しない)。

  J4 control (健康なアクティブ) 93408551 row134 の合成コピー(アクティブHPだけ300に引き上げ)
      条件1(想定1発打点以下)を満たさないため発火しない。

  J5 parity  上記局面で og_r11(新キー無し)が従来挙動のままであること。

使い方:
    python kaggle_replays/_snapshot_gate_r12.py
    python kaggle_replays/_snapshot_gate_r12.py --json out.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# 読み出し/表示ユーティリティは既存診断ハーネスをそのまま使う(sys.path と cwd の設定も
# 向こうで完結している)。重複実装を作らない。
sys.path.insert(0, str(_HERE))
import _probe_missed_lethal as pm  # noqa: E402

from cg.api import OptionType, to_observation_class  # noqa: E402
from ptcg_ai.learning import encoder as _enc  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402

_CONFIG_DIR = _ROOT / "sample_submission" / "configs"


def load_agent_config(name: str) -> dict[str, Any]:
    return json.loads((_CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8"))


R11 = load_agent_config("abl_5_full_og_r11")
R12 = load_agent_config("abl_5_full_og_r12")

# J1の診断用: skip_vs_bench_damage_archetypes を外し、条件1-3だけを見る変種(committされた
# 本番configではない。仕様矛盾の切り分け専用)。
#
# 注意: `_try_terastal_rotation` 本体は既存ガード群(`ability_draw_brake` の `card_ids` 等)
# と同じ ``guard.get(key) or DEFAULT`` の作法で既定値を持つため、空リスト ``[]`` を渡しても
# 「未指定」と区別されずデフォルト([121])にフォールバックしてしまう(この作法は本ファイルの
# 他ガードにも共通しており、terastal_rotation 側だけ ``.get(key, DEFAULT)`` に変えると流儀が
# 割れるため踏襲する)。ここでは実在しないダミーIDを指定して「実質的にスキップが発火しない」
# 状態を作る。
R12_NO_SKIP = copy.deepcopy(R12)
R12_NO_SKIP["name"] = "abl_5_full_og_r12__diag_no_dragapult_skip"
R12_NO_SKIP["terastal_rotation"]["skip_vs_bench_damage_archetypes"] = [999999]

_DRAGAPULT = 121


# --------------------------------------------------------------------------
# 局面の取り出し(`_snapshot_gate_r10.py` / `_snapshot_gate_r11.py` と同じ流儀)
# --------------------------------------------------------------------------

_replay_cache: dict[int, tuple[dict[int, dict], list[int]]] = {}


def episode_rows(episode_id: int) -> tuple[dict[int, dict], list[int]]:
    cached = _replay_cache.get(episode_id)
    if cached is None:
        replay = pm.load_replay(episode_id)
        own = pm.own_index_of(replay)
        deck = pm.own_deck_of(replay, own)
        rows = {d["row"]: d for d in pm.own_decisions(replay, own)}
        cached = (rows, deck)
        _replay_cache[episode_id] = cached
    return cached


def use_deck(deck: list[int]) -> None:
    """ml_policy_agent が読むデッキを、その試合で実際に使ったデッキに差し替える。"""
    mpa._deck_cache = list(deck)
    mpa._model = None
    mpa._model_cache_by_weights_path.clear()


def get_case(episode_id: int, row: int):
    rows, deck = episode_rows(episode_id)
    dec = rows[row]
    obs = to_observation_class(dec["obs_dict"])
    use_deck(deck)
    return obs, deck, list(dec["answer"] or [])


def _option_card_ids(obs) -> list[int | None]:
    return [_enc._resolve_card_id(o, obs.current) for o in obs.select.option]


def describe_action(obs, action: list[int] | None) -> str:
    if action is None:
        return "None"
    parts = []
    for i in action:
        opt_type = pm.enum_name(OptionType, obs.select.option[i].type)
        parts.append(f"{i}:{opt_type}")
    return "[" + ", ".join(parts) + "]"


class Gate:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, bool(ok), detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
        return bool(ok)

    def summary(self) -> int:
        failed = [r for r in self.results if not r[1]]
        print("\n" + "=" * 100)
        print(f"SNAPSHOT GATE r12: {len(self.results) - len(failed)}/{len(self.results)} PASS")
        for name, _ok, detail in failed:
            print(f"  FAIL: {name}" + (f"  — {detail}" if detail else ""))
        return 1 if failed else 0


def _retreat_index(obs) -> int:
    return next(i for i, o in enumerate(obs.select.option) if o.type == OptionType.RETREAT)


# --------------------------------------------------------------------------
# J1: episode 93408551 T10 row134(決定的、ただし相手がドラパルトex本人)
# --------------------------------------------------------------------------

J1_CASE = (93408551, 134, "T10 アクティブHP10/エネ4、ベンチHP150/エネ4。相手アクティブはドラパルトex")


def run_j1(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("J1 terastal_rotation: 93408551 T10 row134(決定的だが相手がドラパルトex本人)")
    episode_id, row, note = J1_CASE
    obs, _deck, answer = get_case(episode_id, row)
    state = obs.current
    me = state.yourIndex
    opp_active = state.players[1 - me].active[0]
    print(f"\n  {episode_id} row{row} (T{state.turn}): {note}")
    print(f"    自アクティブHP{state.players[me].active[0].hp} "
          f"相手アクティブ={opp_active.id}(hp={opp_active.hp}) 実戦={describe_action(obs, answer)}")
    gate.check("J1-0 前提: 実戦の選択がATTACK", len(answer) == 1
               and obs.select.option[answer[0]].type == OptionType.ATTACK,
               describe_action(obs, answer))
    gate.check("J1-0' 前提: 相手アクティブがドラパルトex(121)本人", opp_active.id == _DRAGAPULT,
               f"id={opp_active.id}")

    retreat_idx = _retreat_index(obs)

    # (a) 既定config(skipあり)では、条件4により発火しない。
    mpa.reset_guard_stats()
    r12_default = mpa._try_terastal_rotation(obs, list(answer), config=R12)
    print(f"    og_r12(既定/skipあり) -> {describe_action(obs, r12_default)}")
    gate.check("J1-a 既定config(ドラパルトexスキップ有効)では不発火", r12_default is None,
               describe_action(obs, r12_default))
    stats = mpa.get_guard_stats()
    gate.check("J1-b 不発火の理由がドラパルトexスキップである",
               stats.get("terastal_rotation_skipped_bench_damage_archetype", 0) == 1,
               str({k: v for k, v in stats.items() if k.startswith("terastal_rotation")}))

    # (b) skipを外した診断用configでは、条件1-3だけでRETREATへ差し替わる
    #     (=「にげてもKOが変わらない」ことをエンジンの実探索で確認できている)。
    mpa.reset_guard_stats()
    r12_noskip = mpa._try_terastal_rotation(obs, list(answer), config=R12_NO_SKIP)
    print(f"    og_r12(diag: skipなし) -> {describe_action(obs, r12_noskip)}")
    gate.check("J1-c skipを外せば条件1-3だけでRETREATへ差し替わる(仕様の核=決定的局面)",
               r12_noskip == [retreat_idx], describe_action(obs, r12_noskip))
    stats2 = mpa.get_guard_stats()
    gate.check("J1-d (b)の発火理由がKO成否維持(見送りではない)",
               stats2.get("terastal_rotation_fired", 0) == 1,
               str({k: v for k, v in stats2.items() if k.startswith("terastal_rotation")}))


# --------------------------------------------------------------------------
# J2: episode 93473767 T11 row129
# --------------------------------------------------------------------------

J2_CASE = (93473767, 129, "T11 アクティブHP60/260、ベンチHP240/240・エネ3。KOは元々不可能")


def run_j2(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("J2 terastal_rotation: 93473767 T11 row129")
    episode_id, row, note = J2_CASE
    obs, _deck, answer = get_case(episode_id, row)
    state = obs.current
    me = state.yourIndex
    opp_active = state.players[1 - me].active[0]
    print(f"\n  {episode_id} row{row} (T{state.turn}): {note}")
    print(f"    自アクティブHP{state.players[me].active[0].hp} "
          f"相手アクティブ={opp_active.id}(hp={opp_active.hp}) 実戦={describe_action(obs, answer)}")
    gate.check("J2-0 前提: 実戦の選択がATTACK", len(answer) == 1
               and obs.select.option[answer[0]].type == OptionType.ATTACK,
               describe_action(obs, answer))
    gate.check("J2-0' 前提: 相手アクティブはスキップ対象外(ドラパルトexでない)",
               opp_active.id != _DRAGAPULT, f"id={opp_active.id}")

    retreat_idx = _retreat_index(obs)

    mpa.reset_guard_stats()
    r11_out = mpa._try_terastal_rotation(obs, list(answer), config=R11)
    r12_out = mpa._try_terastal_rotation(obs, list(answer), config=R12)
    print(f"    og_r11 -> {describe_action(obs, r11_out)}")
    print(f"    og_r12 -> {describe_action(obs, r12_out)}")
    gate.check("J2-a og_r11 は不介入(新キー無し)", r11_out is None, describe_action(obs, r11_out))
    gate.check("J2-b og_r12(既定config)がRETREATへ差し替え", r12_out == [retreat_idx],
               describe_action(obs, r12_out))
    stats = mpa.get_guard_stats()
    gate.check("J2-c 発火理由がKOの元々の不在(current_ko=False→parity自動成立)",
               stats.get("terastal_rotation_fired", 0) == 1,
               str({k: v for k, v in stats.items() if k.startswith("terastal_rotation")}))


# --------------------------------------------------------------------------
# J4: 対照(健康なアクティブ) — 93408551 row134 の合成コピー(アクティブHPのみ300に引上げ)
# --------------------------------------------------------------------------

def run_j4(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("J4 control: 健康なアクティブ(想定1発打点を上回るHP)では不発火")
    episode_id, row, _note = J1_CASE
    obs, _deck, answer = get_case(episode_id, row)
    obs.current.players[obs.current.yourIndex].active[0].hp = 300
    obs.current.players[obs.current.yourIndex].active[0].maxHp = 300
    print(f"    {episode_id} row{row}(合成): 自アクティブHPを300に引上げ 実戦想定={describe_action(obs, answer)}")

    mpa.reset_guard_stats()
    out = mpa._try_terastal_rotation(obs, list(answer), config=R12_NO_SKIP)
    print(f"    og_r12(diag: skipなし, HP300) -> {describe_action(obs, out)}")
    gate.check("J4-a 健康なアクティブ(HP300)では不発火(条件1で弾かれる)", out is None,
               describe_action(obs, out))


# --------------------------------------------------------------------------
# J5: og_r11(新キー無し)の従来挙動が保たれていること
# --------------------------------------------------------------------------

_J5_CASES = [(93408551, 134), (93473767, 129)]


def run_j5(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("J5 parity: og_r11(新キー無し)は全局面で terastal_rotation が一切発火しない")
    ok = True
    detail: list[str] = []
    for episode_id, row in _J5_CASES:
        obs, _deck, answer = get_case(episode_id, row)
        if not answer:
            continue
        out = mpa._try_terastal_rotation(obs, list(answer), config=R11)
        if out is not None:
            ok = False
            detail.append(f"{episode_id} row{row} terastal_rotation -> {out}")
    gate.check("J5-a og_r11 では terastal_rotation が一切発火しない", ok, "; ".join(detail))


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="og_r12 の実局面スナップショットゲート")
    ap.add_argument("--json", default=None, help="判定結果のJSON出力先")
    args = ap.parse_args()

    print(f"configs: og_r11={_CONFIG_DIR / 'abl_5_full_og_r11.json'}")
    print(f"         og_r12={_CONFIG_DIR / 'abl_5_full_og_r12.json'}")
    gate = Gate()
    mpa.reset_guard_stats()
    run_j1(gate)
    run_j2(gate)
    run_j4(gate)
    run_j5(gate)
    print("\nguard stats(このゲート全体の発火数):",
          json.dumps({k: v for k, v in mpa.get_guard_stats().items() if v}, ensure_ascii=False))
    code = gate.summary()
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"results": [{"name": n, "pass": ok, "detail": d} for n, ok, d in gate.results],
             "guard_stats": mpa.get_guard_stats()},
            ensure_ascii=False, indent=1), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
