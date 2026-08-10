"""crustle・rocket_mewtwo_exでのT1崩壊を診断するスクリプト(mirror-distilled T1
baselineがfixed pool主判定に不合格だったことを受けての原因切り分け)。

確認する項目:
- teacher(v40)とT1の最初の行動不一致率(top-1一致率)
- T1が選んだ行動にteacherが割り当てていた確率(教師確率)
- 盤面/選択肢card idのUNK率(vocabに無いカードをUNKにフォールバックしている割合)
- 盤面トークン数(truncationの兆候が無いか)
- mask/argmaxが常に合法手範囲内であること
- ライブ経路(t1_live_agent)とoffline経路(collect_tokens._build_decision)の
  token一致(このデッキでもparityが崩れていないか)

pytestの回帰テストではなく、手動実行して結果を確認する診断スクリプト。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CKPT = (_ROOT / "kaggle_replays" / "rl" / "runs" / "distill_v40" / "train"
        / "stage100k_teacher_t1_seed1" / "best.pt")
_TEACHER = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_LEARNER_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"
_RUN_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1"
_RUN_JSON = _RUN_DIR / "run.json"


def diagnose_opponent(opp_id: str, n_games: int, seed0: int, model, vocab, t1_state_pm,
                      profile_name: str) -> dict:
    import json
    import common as C
    import collect_tokens as ct
    import t1_live_agent as la
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.learning import board_tokens as board_tokens_mod
    from ptcg_ai.learning import encoder
    from ptcg_ai.learning import legacy_feature_manifest as manifest
    from ptcg_ai.learning.card_vocab import UNK_INDEX
    from run_league import read_deck_csv_file

    run_cfg = json.loads(_RUN_JSON.read_text(encoding="utf-8"))
    opp_cfg = next(o for o in run_cfg["opponents"] if o["id"] == opp_id)
    weights = C.resolve_opponent_weights(_RUN_DIR, opp_cfg.get("weights"))
    deck_o_path = C.resolve_deck(opp_cfg.get("deck") or run_cfg["opponent_deck"])
    pm_opp = PolicyModel(str(weights) if weights else None)
    deck_o = read_deck_csv_file(str(deck_o_path))
    deck_t1 = read_deck_csv_file(str(_LEARNER_DECK))

    n_decisions = 0
    n_match = 0
    teacher_prob_sum = 0.0
    board_unk = board_total = 0
    option_unk = option_total = 0
    board_token_counts = []
    parity_checked = 0
    parity_ok = 0

    for g in range(n_games):
        t1_index = g % 2
        deck0, deck1 = (deck_t1, deck_o) if t1_index == 0 else (deck_o, deck_t1)
        obs_dict, start_data = battle_start(deck0, deck1)
        if start_data.errorType != 0:
            continue
        n = 0
        try:
            while n < 3000:
                obs = to_observation_class(obs_dict)
                cur = obs.current
                if cur is None or cur.result != -1:
                    break
                select = obs.select
                if select is not None and select.option:
                    if cur.yourIndex == t1_index and select.maxCount == 1:
                        arrays = la.build_single_decision_arrays(t1_state_pm, cur, select, profile_name)
                        board_unk += sum(1 for cid in arrays["board_token_card_ids"]
                                        if vocab.index_of(int(cid)) == UNK_INDEX)
                        board_total += len(arrays["board_token_card_ids"])
                        option_unk += sum(1 for cid in arrays["option_card_ids"] if vocab.index_of(int(cid)) == UNK_INDEX)
                        option_total += len(arrays["option_card_ids"])
                        board_token_counts.append(len(arrays["board_token_card_ids"]))

                        t1_scores = la.t1_scores(model, vocab, t1_state_pm, cur, select, profile_name)
                        t1_idx = int(np.argmax(t1_scores))

                        teacher_scores = np.asarray(t1_state_pm.score_options_from_state(cur, select))
                        teacher_idx = int(np.argmax(teacher_scores))
                        teacher_probs = np.exp(teacher_scores - teacher_scores.max())
                        teacher_probs = teacher_probs / teacher_probs.sum()

                        n_decisions += 1
                        n_match += int(t1_idx == teacher_idx)
                        teacher_prob_sum += float(teacher_probs[t1_idx])

                        if parity_checked < 20:
                            offline = ct._build_decision(t1_state_pm, encoder, board_tokens_mod, manifest,
                                                          profile_name, cur, select, debug_pointers=False)
                            ok = (list(offline["option_card_ids"]) == list(arrays["option_card_ids"])
                                 and list(offline["board_card_ids"]) == list(arrays["board_token_card_ids"]))
                            parity_checked += 1
                            parity_ok += int(ok)

                        action = [t1_idx]
                    elif cur.yourIndex == t1_index:
                        sc = t1_state_pm.score_options_from_state(cur, select)
                        nn = len(select.option)
                        count = max(select.minCount, min(select.maxCount, nn))
                        action = sorted(range(nn), key=lambda i: sc[i], reverse=True)[:count]
                    else:
                        sc = pm_opp.score_options_from_state(cur, select)
                        nn = len(select.option)
                        count = max(select.minCount, min(select.maxCount, nn))
                        action = sorted(range(nn), key=lambda i: sc[i], reverse=True)[:count]
                else:
                    action = []
                obs_dict = battle_select(action)
                n += 1
        finally:
            battle_finish()

    return {
        "opponent": opp_id, "n_games": n_games, "n_decisions": n_decisions,
        "top1_agreement_vs_v40": (n_match / n_decisions) if n_decisions else float("nan"),
        "teacher_prob_of_t1_choice_mean": (teacher_prob_sum / n_decisions) if n_decisions else float("nan"),
        "board_unk_rate": (board_unk / board_total) if board_total else float("nan"),
        "option_unk_rate": (option_unk / option_total) if option_total else float("nan"),
        "board_token_count_mean": (sum(board_token_counts) / len(board_token_counts)) if board_token_counts else float("nan"),
        "board_token_count_max": max(board_token_counts) if board_token_counts else 0,
        "live_offline_parity": f"{parity_ok}/{parity_checked}",
    }


def main(n_games_per_opponent: int = 80) -> None:
    import t1_live_agent as la
    from ptcg_ai.learning.policy_model import PolicyModel

    model, vocab, profile_name, _ = la.load_t1_for_inference(_CKPT, device="cpu")
    t1_state_pm = PolicyModel(str(_TEACHER))
    assert t1_state_pm.is_ready

    for opp_id, seed0 in (("crustle", 500000), ("rocket_mewtwo_ex", 600000)):
        r = diagnose_opponent(opp_id, n_games_per_opponent, seed0, model, vocab, t1_state_pm, profile_name)
        print(f"\n=== {opp_id} ===")
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
