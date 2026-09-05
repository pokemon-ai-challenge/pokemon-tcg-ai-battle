#!/usr/bin/env python3
"""yadoking / lopunny_megafroslass BC(policy_weights_<arch>_g2.json)の品質ゲート。

学習に一切使っていない実ラダーリプレイ(submission 55527580, kaggle_replays/replays/ 配下)を
対象に、対面が yadoking / lopunny_megafroslass だった試合の「相手側」の意思決定点を
extract_policy_dataset.iter_decision_points() と同じ抽出基準で取り出し、学習済み重みで
top-1 一致率を測る。読み取り専用(cg/ data/ sample_submission/ は変更しない)。

対象試合(kaggle_replays/_submission_episode_map.json の "55527580" 41試合を
_audit_hammer_replay._classify_opponent() で分類し、対面アーキタイプが一致した試合を採用。
分類には sample_submission/ptcg_ai/opponent_modeling/rough_predictor.json を使う。
この抽出時点(2026-08)で該当したのは以下の7試合):
  yadoking: 93320069(負け,対ANDPAD kaggler team) / 93330074(負け,対nasuo445) /
            93331988(勝ち,対nasuo445)
  lopunny_megafroslass: 93322802(負け,対NIWATORI) / 93324616(負け,対Urazalinov Baurzhan) /
            93326434(負け,対stnick) / 93335550(負け,対Rikito Kanda)
上記いずれの episode_id も training pool(kaggle_replays/replays_g2/)には存在しない
(episode単位のリークなし)。ただし一部の対戦相手チーム名(nasuo445 等)は同名チームの
別試合が少数クラス追加取得で training pool に混入している可能性がある(パイロット単位の
重複はゼロではない、下記出力で teams_overlap として明示する)。

「キー手順」の操作的定義(実測トランスクリプト kaggle_replays/transcripts/55527580/*.md の
記述に基づく。厳密なゲーム内メカニズムの完全一致ではなくベストエフォート):
  - yadoking: EVOLVE(→ヤドキング cardId=163) / ATTACH(対象がヤドキング) /
    RETREAT(アクティブがヤドキング) / CARD選択(SWITCH文脈でヤドキングを場に戻す) /
    ATTACK(「ひらめきチャレンジ」)
  - lopunny_megafroslass: 同様に EVOLVE/ATTACH/RETREAT/SWITCH の対象が
    メガミミロップex(849) or メガユキメノコex(861) / ATTACK(「しっぷうづき」「うらみぶし」)

使い方:
  PYTHONIOENCODING=utf-8 python kaggle_replays/_yadoking_lopunny_quality_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
for p in (_SUB, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import _audit_hammer_replay as _audit  # noqa: E402  own_index解決/相手アーキ分類の再利用
import _render_replay_transcript as _transcript  # noqa: E402  日本語カード名/技名辞書の再利用
import extract_policy_dataset as _extract  # noqa: E402  意思決定点抽出ロジックの再利用

from cg.api import OptionType, SelectContext, to_observation_class  # noqa: E402
from ptcg_ai.learning import encoder as _enc  # noqa: E402
from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402

REF_DECK_PATH = _HERE / "deck_search" / "candidates_ogerpon_stage3" / "g2top2_v032.csv"
LEARNING_DIR = _SUB / "ptcg_ai" / "learning"

EVAL_CASES: dict[str, list[str]] = {
    "yadoking": ["93320069", "93330074", "93331988"],
    "lopunny_megafroslass": ["93322802", "93324616", "93326434", "93335550"],
}

KEY_CARD_IDS: dict[str, set[int]] = {
    "yadoking": {163},
    "lopunny_megafroslass": {849, 861},
}

KEY_ATTACK_NAMES: dict[str, set[str]] = {
    "yadoking": {"ひらめきチャレンジ"},
    "lopunny_megafroslass": {"しっぷうづき", "うらみぶし"},
}

TRAINING_REPLAYS_DIR = _HERE / "replays_g2"
TRAINING_LABELS_PATH = _HERE / "deck_predictor" / "output" / "deck_labels_g2_v2.jsonl"
TRAINING_MASTER_PATH = _HERE / "index_g2" / "episodes_master.jsonl"


def training_teams_for(arch: str) -> set[str]:
    """training pool(g2)でこのアーキタイプを使っていたチーム名の集合(リーク確認用)。"""
    master: dict[str, dict] = {}
    with TRAINING_MASTER_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            master[row["episode_id"]] = row
    teams: set[str] = set()
    with TRAINING_LABELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("archetype") != arch:
                continue
            m = master.get(d["episode_id"])
            if not m:
                continue
            players = {p["player_index"]: p for p in m.get("players", [])}
            p = players.get(d["player_index"])
            if p and p.get("team_name"):
                teams.add(p["team_name"])
    return teams


def classify_key(
    row: dict, jp_attacks: dict[int, str], key_card_ids: set[int], key_attack_names: set[str]
) -> str | None:
    """ground-truth(実際に選ばれた選択肢)がこのアーキタイプの「キー手順」に該当するかを判定する。

    該当すれば人間可読の短い理由文字列、非該当なら None を返す。
    """
    obs = to_observation_class({**row["observation"], "logs": []})
    state = obs.current
    opt = obs.select.option[row["chosen_index"]]
    t = opt.type

    if t == OptionType.ATTACK:
        name = jp_attacks.get(opt.attackId, f"attackId={opt.attackId}")
        if name in key_attack_names:
            return f"ATTACK:{name}"
        return None

    if t == OptionType.EVOLVE:
        cid = _enc._resolve_card_id(opt, state)
        if cid in key_card_ids:
            return f"EVOLVE:cardId={cid}"
        return None

    if t == OptionType.ATTACH:
        tgt = _enc._resolve_in_play_pokemon(opt, state)
        if tgt is not None and tgt.id in key_card_ids:
            return f"ATTACH:targetId={tgt.id}"
        return None

    if t == OptionType.RETREAT:
        active = state.players[state.yourIndex].active
        if active and active[0] is not None and active[0].id in key_card_ids:
            return f"RETREAT:activeId={active[0].id}"
        return None

    if obs.select.context == SelectContext.SWITCH:
        cid = _enc._resolve_card_id(opt, state)
        if cid in key_card_ids:
            return f"SWITCH_IN:cardId={cid}"
        return None

    return None


def evaluate_archetype(arch: str) -> dict:
    print(f"\n{'=' * 100}\n{arch}\n{'=' * 100}")
    ref_deck = _audit.load_ref_deck(REF_DECK_PATH)
    weights_path = LEARNING_DIR / f"policy_weights_{arch}_g2.json"
    model = PolicyModel(weights_path=weights_path)
    if not model.is_ready:
        print(f"  警告: {weights_path} が未生成またはロード失敗")
        return {"archetype": arch, "error": "weights_not_ready"}

    jp_attacks = _transcript.build_jp_attack_names()

    all_rows: list[dict] = []
    per_episode: list[dict] = []
    for eid in EVAL_CASES[arch]:
        data = _audit._load_episode(eid)
        if data is None:
            print(f"  episode {eid}: リプレイファイルが無い、skip")
            continue
        own_index, reason = _audit._resolve_own_index(data, ref_deck)
        if own_index is None:
            print(f"  episode {eid}: own_index解決失敗({reason})、skip")
            continue
        opp_index = 1 - own_index
        team_names = data.get("info", {}).get("TeamNames") or ["?", "?"]
        archetype_labels = {(eid, opp_index): arch}
        stats = _extract.Stats()
        rows = list(
            _extract.iter_decision_points(data, eid, archetype_labels, None, stats, arch)
        )
        rewards = data.get("rewards") or [None, None]
        opp_reward = rewards[opp_index] if opp_index < len(rewards) else None
        opp_won = opp_reward is not None and opp_reward > 0
        print(
            f"  episode {eid}: 相手={team_names[opp_index]}  相手視点の結果="
            f"{'勝ち' if opp_won else '負け'}  抽出決定点={len(rows)}件"
        )
        per_episode.append({"episode_id": eid, "opponent_team": team_names[opp_index], "n_rows": len(rows)})
        all_rows.extend(rows)

    n = len(all_rows)
    if n == 0:
        print("  抽出できた決定点が0件")
        return {"archetype": arch, "n": 0}

    n_match = 0
    n_nonforced = 0
    n_nonforced_match = 0
    n_key = 0
    n_key_match = 0
    detail_key: list[dict] = []

    for row in all_rows:
        obs = to_observation_class({**row["observation"], "logs": []})
        pred = model.select_option(obs)
        actual = row["chosen_index"]
        match = pred == actual
        n_match += int(match)

        n_options = row["n_options"]
        forced = n_options < 2
        if not forced:
            n_nonforced += 1
            n_nonforced_match += int(match)

        key_reason = classify_key(row, jp_attacks, KEY_CARD_IDS[arch], KEY_ATTACK_NAMES[arch])
        if key_reason is not None:
            n_key += 1
            n_key_match += int(match)
            detail_key.append(
                {
                    "episode_id": row["episode_id"],
                    "turn": row["turn"],
                    "n_options": n_options,
                    "reason": key_reason,
                    "match": bool(match),
                }
            )

    top1 = n_match / n
    nonforced_top1 = n_nonforced_match / n_nonforced if n_nonforced else None
    key_top1 = n_key_match / n_key if n_key else None

    print(f"\n  全決定点: n={n}  top1={top1:.4f}")
    print(f"  非強制選択(選択肢>=2): n={n_nonforced}  top1={nonforced_top1:.4f}" if n_nonforced else "  非強制選択: 0件")
    print(f"  キー手順: n={n_key}  top1={key_top1:.4f}" if n_key else "  キー手順: 0件")
    if detail_key:
        print("  キー手順の内訳:")
        for d in detail_key:
            mark = "OK" if d["match"] else "NG"
            print(f"    [{mark}] episode={d['episode_id']} turn={d['turn']} n_opt={d['n_options']} {d['reason']}")

    training_teams = training_teams_for(arch)
    eval_teams = {e["opponent_team"] for e in per_episode}
    overlap = sorted(eval_teams & training_teams)
    if overlap:
        print(f"  注意(パイロット単位の重複、episode単位のリークではない): {overlap}")

    return {
        "archetype": arch,
        "n_episodes": len(per_episode),
        "per_episode": per_episode,
        "n": n,
        "top1": top1,
        "n_nonforced": n_nonforced,
        "nonforced_top1": nonforced_top1,
        "n_key": n_key,
        "key_top1": key_top1,
        "key_detail": detail_key,
        "pilot_overlap_with_training": overlap,
    }


def main() -> None:
    results = {arch: evaluate_archetype(arch) for arch in EVAL_CASES}
    out_path = _HERE / "_yadoking_lopunny_quality_gate_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果を書き出しました: {out_path}")


if __name__ == "__main__":
    main()
