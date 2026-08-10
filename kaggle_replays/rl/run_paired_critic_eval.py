"""critic無し(``runs/ppo_pilot_grimmsnarl/``)/critic+GAE有り(``runs/ppo_pilot_grimmsnarl_critic/``)
のオーロンゲT1 checkpointを、共通episode manifest上でpaired evaluationする。

前提(既知の制約、本スクリプト実行前に再確認済み):
cg.dll(ネイティブゲームエンジン、変更禁止)にPythonから呼べるseed設定APIが無いため、
同一Pythonシードでもデッキシャッフル・サイド配置は再現しない
(``diagnose_seed_reproducibility.py``を再実行して再確認: 同じ(learner_index=0, seed=555)を
5回投げても決定点数が[92, 76, 10, 75, 98]と毎回異なった。design.md §9.1.1で既に
「マルチプロセスのタスク割当順序が原因」という仮説は単一プロセス再現テストで排除済み、
原因はcgエンジン内部の非公開乱数と特定されている)。

このため「盤面内容が完全一致するpaired比較」はできない。本スクリプトが担保するのは、
制御可能な要素——対戦相手の割当・先攻後攻・試合数・試合順序・Python側seed値——を
全8 checkpoint(critic無し/あり x checkpoint_0/500/1000/2000)で完全に一致させること
(1回だけepisode manifestを構築・保存し、全checkpointの評価で使い回す)。

checkpoint_0はcritic無し/あり両armとも同じ元ファイル(``runs/distill_grimmsnarl/train/
rule_teacher_seed0/best.pt``)を指すため、この2つのcheckpoint_0評価run自体が
「同一checkpointを2回評価した場合の結果」の再確認になる(要求4/5)。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_grimmsnarl_t1_eval as ge  # noqa: E402

_INITIAL_CKPT = (_ROOT / "kaggle_replays" / "rl" / "runs" / "distill_grimmsnarl" / "train"
                / "rule_teacher_seed0" / "best.pt")
_CRITIC_FREE_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_pilot_grimmsnarl"
_CRITIC_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "ppo_pilot_grimmsnarl_critic"
_OUT_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "paired_critic_eval"
_V40_FOR_FEATURES = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_GRIMM_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "marnie_grimmsnarl_ex" / "01.csv"

# 要求どおり: crustle/mega_lucario_exは200試合以上、その他の固定プール相手は100試合以上。
# teacher_rule(直接対戦)は既存H2H_GAMES=300を踏襲(固定プールの一部ではないが、
# 「相手別」の1つとしてmanifestに含める)。
OPPONENT_GAME_COUNTS = [
    ("teacher_rule", 300),
    ("crustle", 200),
    ("mega_lucario_ex", 200),
    ("alakazam", 100),
    ("archaludon_ex", 100),
    ("marnie_grimmsnarl_ex", 100),
    ("rocket_mewtwo_ex", 100),
    ("shirona_garchomp_ex", 100),
    ("dragapult_ex", 100),
]

MANIFEST_SEED_BASE = 9_000_000

CHECKPOINT_ENTRIES = [
    ("critic_free", "checkpoint_0", _INITIAL_CKPT),
    ("critic_free", "checkpoint_500", _CRITIC_FREE_DIR / "checkpoint_500.pt"),
    ("critic_free", "checkpoint_1000", _CRITIC_FREE_DIR / "checkpoint_1000.pt"),
    ("critic_free", "checkpoint_2000", _CRITIC_FREE_DIR / "checkpoint_2000.pt"),
    ("critic", "checkpoint_0", _INITIAL_CKPT),
    ("critic", "checkpoint_500", _CRITIC_DIR / "checkpoint_500.pt"),
    ("critic", "checkpoint_1000", _CRITIC_DIR / "checkpoint_1000.pt"),
    ("critic", "checkpoint_2000", _CRITIC_DIR / "checkpoint_2000.pt"),
]


def build_manifest() -> list[dict]:
    manifest = []
    idx = 0
    seed_counter = MANIFEST_SEED_BASE
    for opp_id, n_games in OPPONENT_GAME_COUNTS:
        for g in range(n_games):
            manifest.append({
                "idx": idx, "opponent_id": opp_id, "t1_index": g % 2, "seed": seed_counter,
            })
            idx += 1
            seed_counter += 1
    return manifest


def load_manifest_or_build(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def _opponent_spec(opp_id: str, run_cfg: dict, run_dir: Path):
    """manifestのopponent_idから {"type":...} と相手デッキを作る(既存run_diagnostic_eval/
    run_grimmsnarl_t1_eval.mainと同じ解決ロジックを再利用、opponent_id="teacher_rule"だけ特別扱い)。"""
    import common as C
    from opponents.rule_agents import grimmsnarl as gm  # noqa: F401  (play_one_game内でimport済み)
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    if opp_id == "teacher_rule":
        deck = read_deck_csv_file(str(_GRIMM_DECK))
        return {"type": "rule"}, deck

    opp_cfg = next(o for o in run_cfg["opponents"] if o["id"] == opp_id)
    w = C.resolve_opponent_weights(run_dir, opp_cfg.get("weights"))
    d = read_deck_csv_file(str(C.resolve_deck(opp_cfg.get("deck") or run_cfg["opponent_deck"])))
    pm = PolicyModel(str(w) if w else None)
    return {"type": "pm", "pm": pm}, d


def evaluate_checkpoint(entry_id: str, ckpt_path: Path, manifest: list[dict], device="cpu") -> list[dict]:
    import t1_live_agent as la
    from ptcg_ai.learning.policy_model import PolicyModel

    run_cfg = json.loads((_ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "run.json")
                         .read_text(encoding="utf-8"))
    run_dir = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1"

    model, vocab, profile_name, _ = la.load_t1_for_inference(ckpt_path, device=device,
                                                              meta_source_checkpoint=_INITIAL_CKPT)
    t1_state_pm = PolicyModel(str(_V40_FOR_FEATURES))
    deck_t1 = None

    # opponent仕様は相手idごとに1回だけ構築(PolicyModel構築コストを節約)。
    opp_cache: dict[str, tuple[dict, list[int]]] = {}
    results = []
    for row in manifest:
        opp_id = row["opponent_id"]
        if opp_id not in opp_cache:
            if deck_t1 is None:
                from run_league import read_deck_csv_file
                deck_t1 = read_deck_csv_file(str(_GRIMM_DECK))
            opp_cache[opp_id] = _opponent_spec(opp_id, run_cfg, run_dir)
        opponent, deck_opp = opp_cache[opp_id]

        r = ge.play_one_game(model, vocab, t1_state_pm, opponent, deck_t1, deck_opp,
                             row["t1_index"], row["seed"], profile_name, device=device)
        results.append({"entry_id": entry_id, "idx": row["idx"], "opponent_id": opp_id,
                        "t1_index": row["t1_index"], "seed": row["seed"], **r})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-games", type=int, default=None,
                    help="デバッグ用: manifest先頭N件だけ評価する")
    ap.add_argument("--entries", default=None,
                    help="デバッグ用: 'critic_free:checkpoint_0,critic:checkpoint_0' のように絞り込む")
    args = ap.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = _OUT_DIR / "episode_manifest.json"
    manifest = load_manifest_or_build(manifest_path)
    print(f"episode manifest: {len(manifest)}試合、{manifest_path}", flush=True)
    if args.limit_games:
        manifest = manifest[:args.limit_games]
        print(f"--limit-gamesにより先頭{len(manifest)}件のみ使用", flush=True)

    entries = CHECKPOINT_ENTRIES
    if args.entries:
        wanted = set(args.entries.split(","))
        entries = [e for e in entries if f"{e[0]}:{e[1]}" in wanted]

    raw_path = _OUT_DIR / "raw_results.jsonl"
    for arm, milestone, ckpt_path in entries:
        if not ckpt_path.exists():
            print(f"skip {arm}/{milestone}: {ckpt_path} が無い", flush=True)
            continue
        entry_id = f"{arm}/{milestone}"
        t0 = time.time()
        results = evaluate_checkpoint(entry_id, ckpt_path, manifest)
        dt = time.time() - t0
        wins = sum(1 for r in results if r["t1_win"] is True)
        errs = sum(1 for r in results if r["error"] is not None)
        print(f"{entry_id}: {wins}/{len(results)}勝 (err={errs}) [{dt:.1f}s, "
             f"{dt/max(len(results),1)*1000:.1f}ms/game]", flush=True)
        with open(raw_path, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("完了", flush=True)


if __name__ == "__main__":
    main()
