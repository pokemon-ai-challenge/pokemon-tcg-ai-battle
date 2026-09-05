#!/usr/bin/env python3
"""og_r8(Fix-A/B/C)の再現ゲート: リプレイの実局面を固定入力にした PASS/FAIL 判定。

相手方策を回さない**主判定**。実ラダーの負け局面をそのまま観測として食わせ、og_r7 と og_r8 で
挙動がどう変わるかを固定する。対戦を回さないので分散がなく、リグレッションの検出に使える。

対象(すべて `kaggle_replays/replays/episode-*-replay.json`。抽出は `_probe_missed_lethal` の
ユーティリティを再利用し、デッキは `steps[1]` の60枚回答=その試合で実際に使ったデッキを使う。
リポジトリの `deck.csv` は別デッキに変わっているため、そちらを読むと隠れ状態が組めない):

  G1 lethal   93335550 rows 162-165 (T15)
      「ボスの指令 → 相手ベンチの弱いポケモン(HP100)を引きずり出す → エネ付与 → 攻撃で勝ち」
      の検証済みリーサル。og_r8 は終盤予算(2500ms/3万node)+ 相手対象のHP昇順ソートで
      到達できるか。4決定は**同一ターン内の連続決定**なので、どれか1つで見つかれば
      その場で試合に勝つ(= 主判定は「行数」ではなく「そのターンを勝てたか」)。

  G2 regression 93322802 row186 (T15)
      ATTACK を選ぶのが正しい局面(相手HP310に最大打点240で届かず、lethal は正しく None)。
      og_r8 でも ATTACK を選び続けること = 新ガードが攻撃を取り上げないことの対照。

  G3 hand guard 93335550 row168 / 93324616 row132
      相手メガユキメノコex(861、うらみぶし=相手の手札×50)が場にいるのに
      「リーリエの決心」で手札を6枚に増やして即死した局面。og_r8 が veto するか。

  G4 briar/chain 93309236 row130 (対 フーディン743、相手手札15枚=被弾300 vs 240HP)
      ブライアではなくジャッジマン(手札4枚=被弾80)を選べたはずの局面。事後veto連鎖の結果。
      93326434 row125 は**誤発火の対照**: ブライアが正当(このターンKOできる)なので触らないこと。

  G5 parity  上記すべての局面で og_r7(新キー無し)が従来挙動のままであること。

  G6 deck brake (r9 Fix-D) 93526443 row147 / 93517227 rows 104,124,139,161
      「山札僅少なのに山札を減らすドローサポートを撃つ」自滅の再現ゲート。
      正例=93526443 T21(山5・手札3 でリーリエの決心 → 山2 → T23 山0 で敗北)。
      対照=93517227 の T7-T13 窓(ジャッジマンは山札を**増やす**ので止めない/
      山10・手札9 のリーリエは山が**増える**ので止めない)。og_r8 は全局面で不介入。

  G7 boss gate (r9 Fix-E) 93503044 row108/row109 / 93528233 row91
      「今ターン倒せない相手をボスの指令で引きずり出す」の再現ゲート。
      93503044 T13 はメガガルーラex(HP330)を 8エネで倒せないのに引きずり出した局面
      (実測: 倒せる対象は bench4 のシェイミ HP110 のみ)。r9 は対象を KO 可能な相手へ
      振り替える。93528233 T6 は成功例(8エネのオーガポン=相手ベンチ最多エネ)で、
      エネ除去例外により**触らない**ことを固定する。og_r8 は全局面で不介入。

使い方:
    python kaggle_replays/_snapshot_gate_r8.py
    python kaggle_replays/_snapshot_gate_r8.py --repeat 5 --min-lethal-rows 3 --explain
    python kaggle_replays/_snapshot_gate_r8.py --skip-g1        # 重い G1 を飛ばす

`--min-lethal-rows` は G1 の合格条件(og_r8 が勝ちリーサルを返すべき行数)。既定 1 =
「このターンに勝てる」。

実測(2026-08-16、repeat=5)の内訳 — 判定条件を「4行中3行」にしていない理由:

  row165  og_r7 2/5 → og_r8 5/5   (og_r8 で確実に発見。med 659ms)
  row164  og_r7 0/5 → og_r8 0/5   (単独計測では 1331ms/1625node で到達するが分散が大きく、
                                   2500ms では取りこぼす回数の方が多い)
  row163  og_r7 0/5 → og_r8 0/5   (緩和予算 20秒/200万node でも未到達=深さ7が要る)
  row162  og_r7 0/5 → og_r8 0/5   (緩和予算で 8.8秒 / 1.2万node かかる。2500ms では届かない)

つまり 2500ms 予算で安定して届くのは row165 のみ。ただし4決定は同一ターンの連続決定なので、
row165 で見つかれば実戦ではその時点で勝つ(og_r7 は 2/5 でしか勝てない)。
なお `endgame_max_nodes`(3万)は実質バインドしない: この局面のノード単価は約 1ms/node で、
2500ms では 2500 ノード程度しか踏めないため、律速は常に時間側。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# 読み出し/表示/探索ユーティリティは既存診断ハーネスをそのまま使う(sys.path と cwd の設定も
# 向こうで完結している)。重複実装を作らない。
sys.path.insert(0, str(_HERE))
import _probe_missed_lethal as pm  # noqa: E402

from cg.api import OptionType, SelectType, to_observation_class  # noqa: E402
from ptcg_ai.learning import encoder as _enc  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as mpa  # noqa: E402
from ptcg_ai.search import lethal_simple  # noqa: E402

_CONFIG_DIR = _ROOT / "sample_submission" / "configs"


def load_agent_config(name: str) -> dict[str, Any]:
    return json.loads((_CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8"))


R7 = load_agent_config("abl_5_full_og_r7")
R8 = load_agent_config("abl_5_full_og_r8")
# r9(Fix-D 山札僅少ブレーキ + Fix-E ボスのKOゲート)。独立ablation用の片側ONも読む。
R9 = load_agent_config("abl_5_full_og_r9")
R9_BRAKE = load_agent_config("abl_5_full_og_r9_deckbrake")
R9_BOSS = load_agent_config("abl_5_full_og_r9_bossgate")

# 「手札を増やすサポート」= 生存ガードが避けるべき札(判定の期待値として明示する)。
_REFILL_IDS = {1227, 1223}
_JUDGE_ID = 1213
_BRIAR_ID = 1201
_BOSS_ID = 1182


# --------------------------------------------------------------------------
# 局面の取り出し
# --------------------------------------------------------------------------

_replay_cache: dict[int, tuple[dict[int, dict], list[int]]] = {}


def episode_rows(episode_id: int) -> tuple[dict[int, dict], list[int]]:
    """(row -> decision, その試合で実際に使った60枚デッキ) を返す(episode 単位でキャッシュ)。"""
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
    """ml_policy_agent が読むデッキを、その試合で実際に使ったデッキに差し替える。

    `_get_deck()` は `deck.csv` をキャッシュするだけなので、ここを差し替えると
    隠れ状態スタブ(`build_dummy_search_state`)も PolicyModel の deck 基準特徴も揃う。
    PolicyModel はデッキを見てロードされるので、デッキを変えたらモデルキャッシュも捨てる。
    """
    mpa._deck_cache = list(deck)
    mpa._model = None
    mpa._model_cache_by_weights_path.clear()


def get_case(episode_id: int, row: int):
    rows, deck = episode_rows(episode_id)
    dec = rows[row]
    obs = to_observation_class(dec["obs_dict"])
    use_deck(deck)
    return obs, deck, list(dec["answer"] or [])


def option_ids(obs) -> list[int | None]:
    return [_enc._resolve_card_id(o, obs.current) for o in obs.select.option]


def describe_action(obs, action: list[int] | None) -> str:
    if action is None:
        return "None"
    ids = option_ids(obs)
    parts = []
    for i in action:
        opt_type = OptionType(obs.select.option[i].type).name
        card = pm.card_name(ids[i]) if ids[i] is not None else "-"
        parts.append(f"{i}:{opt_type}/{card}")
    return "[" + ", ".join(parts) + "]"


# --------------------------------------------------------------------------
# 判定結果の器
# --------------------------------------------------------------------------

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
        print(f"SNAPSHOT GATE r8: {len(self.results) - len(failed)}/{len(self.results)} PASS")
        for name, _ok, detail in failed:
            print(f"  FAIL: {name}" + (f"  — {detail}" if detail else ""))
        return 1 if failed else 0


# --------------------------------------------------------------------------
# G1: 終盤リーサル(Fix-A)
# --------------------------------------------------------------------------

G1_EPISODE = 93335550
G1_ROWS = [162, 163, 164, 165]


def run_g1(gate: Gate, repeat: int, min_rows: int, explain: bool) -> None:
    print("\n" + "=" * 100)
    print(f"G1 lethal: episode {G1_EPISODE} rows {G1_ROWS} (T15) / 各 {repeat} 回")
    found: dict[str, dict[int, int]] = {"og_r7": {}, "og_r8": {}}
    per_repeat: dict[str, list[set[int]]] = {
        "og_r7": [set() for _ in range(repeat)],
        "og_r8": [set() for _ in range(repeat)],
    }

    for row in G1_ROWS:
        obs, deck, answer = get_case(G1_EPISODE, row)
        state, select = obs.current, obs.select
        print(f"\n  row {row}: {pm.enum_name(SelectType, select.type)} "
              f"選択肢{len(select.option)}件 実戦の回答={describe_action(obs, answer)}")
        for label, cfg in (("og_r7", R7), ("og_r8", R8)):
            hits, ms_list = 0, []
            for rep in range(repeat):
                lethal_simple.reset_stats()
                t0 = time.perf_counter()
                action = lethal_simple.search(state, select.option, {
                    "observation": obs,
                    "config": cfg["lethal_search"],
                    "hidden_state_factory": lambda: pm.build_dummy_search_state(obs, deck),
                })
                ms_list.append((time.perf_counter() - t0) * 1000.0)
                if action is not None:
                    hits += 1
                    per_repeat[label][rep].add(row)
                    if explain and label == "og_r8" and hits == 1:
                        print(f"      勝ち手: {describe_action(obs, action)}")
            found[label][row] = hits
            print(f"    {label}: 発見 {hits}/{repeat}  "
                  f"時間 min={min(ms_list):.0f} / med={sorted(ms_list)[len(ms_list) // 2]:.0f} / "
                  f"max={max(ms_list):.0f} ms")

    # 隠れ状態スタブ(`build_dummy_search_state`)は毎回シャッフルするので、同じ局面でも
    # 探索木の大きさが変わる = 発見は確率的。og_r7 でも row165 がまれに 400ms 内に収まる。
    # よって「og_r7 は必ず見逃す」ではなく **発見率で厳密に上回る** ことを判定条件にする。
    r7_total = sum(found["og_r7"].values())
    r8_total = sum(found["og_r8"].values())
    n_cells = repeat * len(G1_ROWS)
    print(f"\n  発見率(行×試行 {n_cells} セル中): og_r7={r7_total} / og_r8={r8_total}")
    gate.check(
        "G1-a og_r8 の発見数が og_r7 を厳密に上回る",
        r8_total > r7_total,
        f"og_r7={r7_total} vs og_r8={r8_total} / 行別 r7={dict(found['og_r7'])} "
        f"r8={dict(found['og_r8'])}",
    )
    won = {label: sum(1 for s in per_repeat[label] if len(s) >= min_rows) for label in per_repeat}
    worst = min(len(s) for s in per_repeat["og_r8"])
    rows_hit = sorted(set().union(*per_repeat["og_r8"]))
    print(f"  「このターンに勝てた」試行数(>= {min_rows} 行で発見): "
          f"og_r7={won['og_r7']}/{repeat} / og_r8={won['og_r8']}/{repeat}")
    gate.check(
        f"G1-b og_r8 が同一ターン内で >= {min_rows} 行の勝ちリーサルを毎回返す",
        worst >= min_rows,
        f"各試行の発見行数 min={worst} / 発見された行={rows_hit} / 行別 {dict(found['og_r8'])}",
    )
    gate.check(
        "G1-c og_r7 は同じ条件でこのターンを勝ちきれない(修正前の再現)",
        won["og_r7"] < repeat,
        f"og_r7 が勝てた試行={won['og_r7']}/{repeat}",
    )


# --------------------------------------------------------------------------
# G2: ATTACK が退行しないこと
# --------------------------------------------------------------------------

def run_g2(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("G2 regression: episode 93322802 row186 (T15) — ATTACK を選び続けるか")
    obs, _deck, answer = get_case(93322802, 186)
    picked: dict[str, list[int]] = {}
    for label, cfg in (("og_r7", R7), ("og_r8", R8)):
        t0 = time.perf_counter()
        action = mpa._select_action(obs, config=cfg)
        ms = (time.perf_counter() - t0) * 1000.0
        picked[label] = action
        print(f"    {label}: {describe_action(obs, action)} ({ms:.0f}ms) "
              f"実戦={describe_action(obs, answer)}")

    def is_attack(action: list[int]) -> bool:
        return len(action) == 1 and obs.select.option[action[0]].type == OptionType.ATTACK

    gate.check("G2-a og_r7 は ATTACK を選ぶ(基準)", is_attack(picked["og_r7"]),
               describe_action(obs, picked["og_r7"]))
    gate.check("G2-b og_r8 も ATTACK を選ぶ(退行なし)", is_attack(picked["og_r8"]),
               describe_action(obs, picked["og_r8"]))
    gate.check("G2-c og_r7 と og_r8 の選択が一致", picked["og_r7"] == picked["og_r8"],
               f"{picked['og_r7']} vs {picked['og_r8']}")


# --------------------------------------------------------------------------
# G3: 手札連動ダメージへの生存ガード(Fix-B)
# --------------------------------------------------------------------------

G3_CASES = [
    (93335550, 168, "相手メガユキメノコex(861) / 自HP240 / リーリエ後6枚×50=300 で即死"),
    (93324616, 132, "同型・ジャッジ不在(手札を増やさない手へ逃がすルート)"),
]


def run_g3(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("G3 hand guard: リーリエの決心で自滅する局面を veto するか")
    for episode_id, row, note in G3_CASES:
        obs, _deck, answer = get_case(episode_id, row)
        ids = option_ids(obs)
        print(f"\n  {episode_id} row{row}: {note}")
        print(f"    実戦の回答={describe_action(obs, answer)}")
        r7_out = mpa._try_hand_damage_guard(obs, list(answer), config=R7)
        r8_out = mpa._try_hand_damage_guard(obs, list(answer), config=R8)
        print(f"    og_r7 -> {describe_action(obs, r7_out)}")
        print(f"    og_r8 -> {describe_action(obs, r8_out)}")
        gate.check(f"G3 og_r7 は不介入 ({episode_id} row{row})", r7_out is None,
                   describe_action(obs, r7_out))
        replaced = r8_out is not None and r8_out != answer
        safe = replaced and ids[r8_out[0]] not in _REFILL_IDS
        gate.check(f"G3 og_r8 がリーリエを手札非増加の手へ差し替え ({episode_id} row{row})",
                   bool(safe), describe_action(obs, r8_out))
        # `_apply_action_vetoes` 経由(実戦の呼ばれ方)でも同じ結論になること。
        chain = mpa._apply_action_vetoes(obs, list(answer), config=R8)
        gate.check(f"G3 veto連鎖でも同じ結論 ({episode_id} row{row})", chain == r8_out,
                   f"chain={describe_action(obs, chain)}")


# --------------------------------------------------------------------------
# G4: ブライア(Fix-C)と veto 連鎖
# --------------------------------------------------------------------------

def run_g4(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("G4 briar / veto chain")

    # (a) 93309236 row130: KOはできる(=ブライア自体は空撃ちでない)が、相手フーディンの
    #     ハンドパワー(相手手札15×20=300)で 240HP が落ちる。ジャッジマンへ回るのが正解。
    obs, _deck, answer = get_case(93309236, 130)
    ids = option_ids(obs)
    briar_only = mpa._try_briar_gate(obs, list(answer), config=R8)
    chain_r7 = mpa._apply_action_vetoes(obs, list(answer), config=R7)
    chain_r8 = mpa._apply_action_vetoes(obs, list(answer), config=R8)
    print(f"\n  93309236 row130: 実戦={describe_action(obs, answer)}")
    print(f"    briar_gate 単体 -> {describe_action(obs, briar_only)} "
          f"(このターンKO可能なのでブライア自体は正当=不介入が正しい)")
    print(f"    og_r7 veto連鎖 -> {describe_action(obs, chain_r7)}")
    print(f"    og_r8 veto連鎖 -> {describe_action(obs, chain_r8)}")
    gate.check("G4-a og_r7 は実戦どおりブライアのまま", chain_r7 == answer,
               describe_action(obs, chain_r7))
    gate.check("G4-b og_r8 はジャッジマンへ差し替え",
               len(chain_r8) == 1 and ids[chain_r8[0]] == _JUDGE_ID,
               describe_action(obs, chain_r8))

    # (b) 誤発火の対照: ブライアが正当(このターンKOできる)な局面では触らないこと。
    obs2, _deck2, answer2 = get_case(93326434, 125)
    ids2 = option_ids(obs2)
    gate_out = mpa._try_briar_gate(obs2, list(answer2), config=R8)
    print(f"\n  93326434 row125 (誤発火の対照): 実戦={describe_action(obs2, answer2)}")
    print(f"    briar_gate -> {describe_action(obs2, gate_out)}")
    gate.check("G4-c 正当なブライア(KO可能)には介入しない", gate_out is None,
               describe_action(obs2, gate_out))
    gate.check("G4-c' 対象局面がブライアであることの確認",
               ids2[answer2[0]] == _BRIAR_ID, f"cardId={ids2[answer2[0]]}")


# --------------------------------------------------------------------------
# G5: og_r7 の従来挙動が保たれていること
# --------------------------------------------------------------------------

G5_CASES = [
    (G1_EPISODE, 162), (G1_EPISODE, 163), (G1_EPISODE, 165), (G1_EPISODE, 168),
    (93322802, 186), (93324616, 132), (93309236, 130), (93326434, 125),
]


def run_g5(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("G5 parity: og_r7(新キー無し)は全局面で従来挙動")
    ok = True
    detail: list[str] = []
    for episode_id, row in G5_CASES:
        obs, _deck, answer = get_case(episode_id, row)
        if not answer:
            continue
        # 事後veto群は全て None(=手を変えない)であること。
        for name, fn in (("hand_damage_guard", mpa._try_hand_damage_guard),
                         ("briar_gate", mpa._try_briar_gate)):
            out = fn(obs, list(answer), config=R7)
            if out is not None:
                ok = False
                detail.append(f"{episode_id} row{row} {name} -> {out}")
        chain = mpa._apply_action_vetoes(obs, list(answer), config=R7)
        if chain != answer:
            ok = False
            detail.append(f"{episode_id} row{row} chain {answer} -> {chain}")
    gate.check("G5-a og_r7 では新ガードが一切発火しない", ok, "; ".join(detail))

    # 探索順序: og_r7 の config では `_candidate_selections` の生成順が新キー導入前と同一。
    import itertools

    def legacy_order(select, config) -> list[list[int]]:
        order = list(range(len(select.option)))
        if select.type == SelectType.MAIN:
            order = [i for i in order if select.option[i].type != OptionType.END]
            priority_ids = lethal_simple._priority_play_ids(config, select_state[0])

            def _rank(i: int) -> float:
                option = select.option[i]
                if priority_ids is not None and option.type == OptionType.PLAY:
                    card_id = lethal_simple._play_option_card_id(option, select_state[0])
                    if card_id is not None and card_id in priority_ids:
                        return lethal_simple._PRIORITY_PLAY_RANK
                return float(lethal_simple._MAIN_OPTION_PRIORITY.get(
                    option.type, lethal_simple._DEFAULT_PRIORITY))

            order.sort(key=_rank)
        out: list[list[int]] = []
        limit = int(config["max_combinations_per_select"])
        for count in range(max(select.minCount, 0), min(select.maxCount, len(order)) + 1):
            for combo in itertools.combinations(order, count):
                out.append(list(combo))
                if len(out) >= limit:
                    return out
        return out

    order_ok = True
    order_detail: list[str] = []
    cfg = {**lethal_simple.DEFAULTS, **R7["lethal_search"]}
    for episode_id, row in G5_CASES:
        obs, _deck, _answer = get_case(episode_id, row)
        select_state = [obs.current]
        got = list(lethal_simple._candidate_selections(obs.select, cfg, obs.current))
        want = legacy_order(obs.select, cfg)
        if got != want:
            order_ok = False
            order_detail.append(f"{episode_id} row{row}: {got} != {want}")
    gate.check("G5-b og_r7 の探索候補生成順が新キー導入前と一致", order_ok,
               "; ".join(order_detail))


# --------------------------------------------------------------------------
# G6: 山札僅少ブレーキ(r9 Fix-D)
# --------------------------------------------------------------------------

# 正例: 93526443 T21 = 山5・手札3 でリーリエの決心(山に戻るのは2枚、引くのは6枚 = 山-3)。
# 実戦はこれで山2 → T23 に山0 になり、相手の山は24枚残っているのに山札切れで敗北した。
G6_BRAKE_CASE = (93526443, 147)

# 対照: 「止めてはいけない」局面。ここで発火したら過剰抑制(ドローエンジンの停止)。
G6_CONTROL_CASES = [
    (93517227, 104, "T7 山10・手札9 のリーリエ = 山に戻る8 - 引く6 で山札は**増える**"),
    (93517227, 124, "T9 山4 のジャッジマン = 手札7 → 山+2(ジャッジは対象外)"),
    (93517227, 139, "T11 山2 のジャッジマン = 手札8 → 山+3(同上)"),
    (93517227, 161, "T13 山0 のジャッジマン = 手札6 → 山+1(同上)"),
]


def run_g6(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("G6 deck brake: 山札僅少で山札を減らすドローを止め、増やす/ジャッジは止めないか")

    episode_id, row = G6_BRAKE_CASE
    obs, _deck, answer = get_case(episode_id, row)
    ids = option_ids(obs)
    state = obs.current
    me = state.yourIndex
    print(f"\n  {episode_id} row{row} (T{state.turn}): 山{state.players[me].deckCount} "
          f"手札{len(state.players[me].hand or [])} 実戦={describe_action(obs, answer)}")
    r8_out = mpa._try_low_deck_draw_brake(obs, list(answer), config=R8)
    r9_out = mpa._try_low_deck_draw_brake(obs, list(answer), config=R9)
    brake_only = mpa._try_low_deck_draw_brake(obs, list(answer), config=R9_BRAKE)
    boss_only = mpa._try_low_deck_draw_brake(obs, list(answer), config=R9_BOSS)
    print(f"    og_r8      -> {describe_action(obs, r8_out)}")
    print(f"    og_r9      -> {describe_action(obs, r9_out)}")
    print(f"    deckbrake  -> {describe_action(obs, brake_only)}")
    print(f"    bossgate   -> {describe_action(obs, boss_only)}")
    gate.check(f"G6-a og_r8 は不介入 ({episode_id} row{row})", r8_out is None,
               describe_action(obs, r8_out))
    replaced = (r9_out is not None and r9_out != answer
                and ids[r9_out[0]] not in _REFILL_IDS)
    gate.check(f"G6-b og_r9 が山札を減らすリーリエを別の手へ差し替え ({episode_id} row{row})",
               bool(replaced), describe_action(obs, r9_out))
    gate.check("G6-c deckbrake 単独 config でも同じ結論(独立ablation)",
               brake_only == r9_out, describe_action(obs, brake_only))
    gate.check("G6-d bossgate 単独 config では発火しない(キー独立)", boss_only is None,
               describe_action(obs, boss_only))

    ok = True
    detail: list[str] = []
    for episode_id, row, note in G6_CONTROL_CASES:
        obs, _deck, answer = get_case(episode_id, row)
        if not answer:
            continue
        out = mpa._try_low_deck_draw_brake(obs, list(answer), config=R9)
        chain_r8 = mpa._apply_action_vetoes(obs, list(answer), config=R8)
        chain_r9 = mpa._apply_action_vetoes(obs, list(answer), config=R9)
        state = obs.current
        me = state.yourIndex
        print(f"\n  対照 {episode_id} row{row}: {note}")
        print(f"    山{state.players[me].deckCount} 手札{len(state.players[me].hand or [])} "
              f"実戦={describe_action(obs, answer)} brake -> {describe_action(obs, out)}")
        if out is not None:
            ok = False
            detail.append(f"{episode_id} row{row} brake -> {out}")
        if chain_r8 != chain_r9:
            ok = False
            detail.append(f"{episode_id} row{row} chain r8={chain_r8} != r9={chain_r9}")
    gate.check("G6-e 対照局面(山札が増える/ジャッジ)では一切介入しない", ok, "; ".join(detail))


# --------------------------------------------------------------------------
# G7: ボスのKOゲート(r9 Fix-E)
# --------------------------------------------------------------------------

# 93503044 T13: 相手ベンチは Crustle×3(HP150)/ メガガルーラex(HP330)/ シェイミ(HP110)。
# 自分は8エネのオーガポンだが、対象別のエンジン評価では**シェイミ(bench4)だけ**が今ターンKO可能
# (Crustle は倒せない)。実戦は KO 不能なメガガルーラex(bench3)を引きずり出して失敗した。
G7_BOSS_PLAY = (93503044, 108)
G7_BOSS_TARGET = (93503044, 109)
G7_KANGASKHAN_OPTION = 3  # 実戦が選んだ対象(メガガルーラex)
# 93528233 T6: 相手ベンチ最多エネ(8個)のオーガポンを引きずり出した成功例。KO はできないが
# エネ除去例外(allow_energy_denial)で許可されるべき局面。
G7_DENIAL_PLAY = (93528233, 91)
G7_DENIAL_TARGET = (93528233, 92)


def run_g7(gate: Gate) -> None:
    print("\n" + "=" * 100)
    print("G7 boss gate: 倒せない相手を引きずり出さない / 成功例には触らない")

    # (a) 倒せない対象(メガガルーラex)を選んでいる局面を、KO可能な対象へ振り替える。
    ep, row = G7_BOSS_PLAY
    play_obs, _deck, play_answer = get_case(ep, row)
    mpa._selects_seen = 0
    mpa._boss_target_cache = None
    r8_play = mpa._try_boss_lethal_gate(play_obs, list(play_answer), config=R8)
    mpa._selects_seen = 0
    mpa._boss_target_cache = None
    r9_play = mpa._try_boss_lethal_gate(play_obs, list(play_answer), config=R9)
    cache = mpa._boss_target_cache
    print(f"\n  {ep} row{row} (ボスPLAY): 実戦={describe_action(play_obs, play_answer)}")
    print(f"    og_r8 -> {describe_action(play_obs, r8_play)} / "
          f"og_r9 -> {describe_action(play_obs, r9_play)}")
    print(f"    許可対象(area,index,player)={sorted(cache['allowed']) if cache else None}")
    gate.check(f"G7-a og_r8 は不介入 ({ep} row{row})", r8_play is None,
               describe_action(play_obs, r8_play))
    gate.check("G7-b og_r9 はボスPLAY自体は通す(倒せる対象が居るため)", r9_play is None,
               describe_action(play_obs, r9_play))
    gate.check("G7-c 許可対象が1件以上ある(=対象選択で振り替えられる)",
               bool(cache and cache.get("allowed")), str(cache and cache.get("allowed")))

    ep_t, row_t = G7_BOSS_TARGET
    tgt_obs, _deck_t, tgt_answer = get_case(ep_t, row_t)
    mpa._selects_seen = 1  # ボスPLAYの**次の** decision
    r9_tgt = mpa._try_boss_target_redirect(tgt_obs, list(tgt_answer), config=R9)
    print(f"  {ep_t} row{row_t} (対象選択): 実戦={describe_action(tgt_obs, tgt_answer)} "
          f"-> og_r9 {describe_action(tgt_obs, r9_tgt)}")
    gate.check("G7-d og_r9 がKO不能なメガガルーラexから対象を振り替える",
               r9_tgt is not None and r9_tgt != tgt_answer
               and r9_tgt[0] != G7_KANGASKHAN_OPTION,
               describe_action(tgt_obs, r9_tgt))
    # og_r8 はキーが無いので同じ手順でも一切触らない。
    mpa._selects_seen = 0
    mpa._boss_target_cache = None
    mpa._try_boss_lethal_gate(play_obs, list(play_answer), config=R8)
    mpa._selects_seen = 1
    r8_tgt = mpa._try_boss_target_redirect(tgt_obs, list(tgt_answer), config=R8)
    gate.check("G7-e og_r8 は対象選択にも不介入", r8_tgt is None, describe_action(tgt_obs, r8_tgt))

    # (b) 成功例(エネ除去): 相手ベンチ最多エネの相手なら KO 不能でも許可する。
    ep_d, row_d = G7_DENIAL_PLAY
    d_obs, _deck_d, d_answer = get_case(ep_d, row_d)
    mpa._selects_seen = 0
    mpa._boss_target_cache = None
    d_play = mpa._try_boss_lethal_gate(d_obs, list(d_answer), config=R9)
    d_cache = mpa._boss_target_cache
    print(f"\n  {ep_d} row{row_d} (成功例のボスPLAY): 実戦={describe_action(d_obs, d_answer)} "
          f"-> og_r9 {describe_action(d_obs, d_play)}")
    print(f"    許可={sorted(d_cache['allowed']) if d_cache else None} "
          f"エネ除去例外={sorted(d_cache['denial']) if d_cache else None}")
    gate.check("G7-f 成功例のボスPLAYは通す(KO不能でも最多エネ=エネ除去例外)",
               d_play is None and bool(d_cache and d_cache.get("denial")),
               describe_action(d_obs, d_play))

    ep_dt, row_dt = G7_DENIAL_TARGET
    dt_obs, _deck_dt, dt_answer = get_case(ep_dt, row_dt)
    mpa._selects_seen = 1
    dt_out = mpa._try_boss_target_redirect(dt_obs, list(dt_answer), config=R9)
    print(f"  {ep_dt} row{row_dt} (成功例の対象選択): 実戦={describe_action(dt_obs, dt_answer)} "
          f"-> og_r9 {describe_action(dt_obs, dt_out)}")
    gate.check("G7-g 成功例の対象(最多エネ)には触らない", dt_out is None,
               describe_action(dt_obs, dt_out))

    # (c) キー独立: deckbrake 単独 config ではボスゲートは一切動かない。
    mpa._selects_seen = 0
    mpa._boss_target_cache = None
    brake_only = mpa._try_boss_lethal_gate(play_obs, list(play_answer), config=R9_BRAKE)
    gate.check("G7-h deckbrake 単独 config ではボスゲートが発火しない(キー独立)",
               brake_only is None and mpa._boss_target_cache is None,
               describe_action(play_obs, brake_only))
    mpa._boss_target_cache = None


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="og_r8 / og_r9 の実局面スナップショットゲート")
    ap.add_argument("--repeat", type=int, default=5,
                    help="G1 の繰り返し回数(隠れ状態スタブは毎回シャッフルするため分散がある)")
    ap.add_argument("--min-lethal-rows", type=int, default=1,
                    help="G1 の合格条件: og_r8 が勝ちリーサルを返すべき行数(既定1=このターンに勝てる)")
    ap.add_argument("--explain", action="store_true", help="見つかった勝ち手を表示する")
    ap.add_argument("--skip-g1", action="store_true", help="重い G1 を飛ばす")
    ap.add_argument("--only", default=None,
                    help="実行するゲートをカンマ区切りで指定(例: g6,g7)。既定は全部。")
    args = ap.parse_args()

    print(f"configs: og_r7={_CONFIG_DIR / 'abl_5_full_og_r7.json'}")
    print(f"         og_r8={_CONFIG_DIR / 'abl_5_full_og_r8.json'}")
    print(f"         og_r9={_CONFIG_DIR / 'abl_5_full_og_r9.json'} "
          f"(+ deckbrake / bossgate の片側ON)")
    only = {s.strip().lower() for s in args.only.split(",")} if args.only else None
    gate = Gate()
    if not args.skip_g1 and (only is None or "g1" in only):
        run_g1(gate, args.repeat, args.min_lethal_rows, args.explain)
    for name, fn in (("g2", run_g2), ("g3", run_g3), ("g4", run_g4), ("g5", run_g5),
                     ("g6", run_g6), ("g7", run_g7)):
        if only is None or name in only:
            fn(gate)
    return gate.summary()


if __name__ == "__main__":
    raise SystemExit(main())
