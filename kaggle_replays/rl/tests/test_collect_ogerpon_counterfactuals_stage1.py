"""collect_ogerpon_counterfactuals.py のtrigger別quota・sample_id対応(Stage1)のテスト。

ユーザー指摘(外部レビュー後の追加指摘)対応の回帰テスト:
- 単純な全trigger共通の先着上限をやめ、trigger別quotaにする。
- state_id(盤面内容ハッシュ)だけでは異なる試合間で衝突しうるため、match_seed +
  learner_index + opponent_archetype + turn + state_id + 試合内発火連番から
  一意な sample_id を作る。pair_id は sample_id から作る。
- hidden stateの生データではなくhashだけを記録し、同一pairの2 Optionが同じ
  determinizationを共有しているかを検証できるようにする。

自己対戦を伴う本体(_process_task)の統合的な疎通確認は、50-100ゲームのpreflight
スクリプト実行で別途行う(このテストはpure functionのみを対象にする)。
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_RL_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = _RL_DIR.parent.parent
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"
for _p in (str(_RL_DIR), str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def mod():
    try:
        import collect_ogerpon_counterfactuals as m
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"collect_ogerpon_counterfactuals / cg engine unavailable: {exc}")
    return m


# ===========================================================================
# sample_id_of: 試合をまたいだ衝突安全性
# ===========================================================================

def test_sample_id_is_deterministic(mod):
    a = mod.sample_id_of(1, 0, "alakazam", 5, "stateabc", 1)
    b = mod.sample_id_of(1, 0, "alakazam", 5, "stateabc", 1)
    assert a == b


def test_sample_id_is_sha256_hex_digest(mod):
    sid = mod.sample_id_of(1, 0, "alakazam", 5, "stateabc", 1)
    assert isinstance(sid, str) and len(sid) == 64
    int(sid, 16)


def test_sample_id_differs_when_match_seed_differs_despite_same_state_id(mod):
    """異なる試合が偶然同じ盤面内容(state_id)に達しても、sample_idは衝突しない。"""
    a = mod.sample_id_of(match_seed=1, learner_index=0, opponent_archetype="alakazam",
                         turn=5, state_id="same_content_hash", fire_seq=1)
    b = mod.sample_id_of(match_seed=2, learner_index=0, opponent_archetype="alakazam",
                         turn=5, state_id="same_content_hash", fire_seq=1)
    assert a != b


def test_sample_id_differs_when_fire_seq_differs(mod):
    """同じ試合内で同じstate_idが複数回発火しても(理論上)、発火連番で区別される。"""
    a = mod.sample_id_of(1, 0, "alakazam", 5, "same_content_hash", fire_seq=1)
    b = mod.sample_id_of(1, 0, "alakazam", 5, "same_content_hash", fire_seq=2)
    assert a != b


def test_sample_id_differs_when_learner_index_or_archetype_differs(mod):
    base = mod.sample_id_of(1, 0, "alakazam", 5, "s", 1)
    diff_side = mod.sample_id_of(1, 1, "alakazam", 5, "s", 1)
    diff_arch = mod.sample_id_of(1, 0, "ogerpon_teal_ex", 5, "s", 1)
    assert base != diff_side
    assert base != diff_arch


# ===========================================================================
# hidden_state_hash_of: 生データを晒さずdeterminization共有を検証できる
# ===========================================================================

def test_hidden_state_hash_is_deterministic(mod):
    hs = {"opponent_hand": [1, 2, 3], "opponent_deck": [4, 5]}
    assert mod.hidden_state_hash_of(hs) == mod.hidden_state_hash_of(hs)


def test_hidden_state_hash_differs_for_different_hidden_state(mod):
    a = mod.hidden_state_hash_of({"opponent_hand": [1, 2, 3]})
    b = mod.hidden_state_hash_of({"opponent_hand": [1, 2, 4]})
    assert a != b


def test_hidden_state_hash_is_sha256_hex_digest(mod):
    h = mod.hidden_state_hash_of({"a": 1})
    assert isinstance(h, str) and len(h) == 64
    int(h, 16)


# ===========================================================================
# _merge_trigger_diag: worker結果をtrigger別に正しく合算する
# ===========================================================================

def test_merge_trigger_diag_sums_counts_and_failure_reasons(mod):
    total: dict = {}
    part1 = {
        "main_attach": {"fired": 3, "quota_skipped": 1, "duplicate_state_skipped": 0,
                        "build_attempted": 2, "build_failed": 1,
                        "build_failure_reasons": {"no_index_resolved": 1}, "saved": 1},
    }
    part2 = {
        "main_attach": {"fired": 2, "quota_skipped": 0, "duplicate_state_skipped": 1,
                        "build_attempted": 1, "build_failed": 0,
                        "build_failure_reasons": {}, "saved": 1},
        "promote": {"fired": 1, "quota_skipped": 0, "duplicate_state_skipped": 0,
                   "build_attempted": 1, "build_failed": 1,
                   "build_failure_reasons": {"no_target_resolved": 1}, "saved": 0},
    }
    mod._merge_trigger_diag(total, part1)
    mod._merge_trigger_diag(total, part2)

    assert total["main_attach"]["fired"] == 5
    assert total["main_attach"]["saved"] == 2
    assert total["main_attach"]["duplicate_state_skipped"] == 1
    assert total["main_attach"]["build_failure_reasons"] == {"no_index_resolved": 1}
    assert total["promote"]["build_failure_reasons"] == {"no_target_resolved": 1}


# ===========================================================================
# _diagnose_candidate_build_failure: 候補構築失敗理由の粗い分類
# ===========================================================================

def test_diagnose_candidate_build_failure_no_index_resolved(mod, monkeypatch):
    from ptcg_ai.ml_policy import ogerpon_strategy as STRAT

    monkeypatch.setattr(STRAT, "_resolve_single_prize_first_action",
                        lambda *a, **k: (None, None))
    mod._W["STRAT"] = STRAT
    reason = mod._diagnose_candidate_build_failure(
        obs=None, learner_index=0, config={}, deck_l=[], trigger="main_attach",
        select=SimpleNamespace(option=[1, 2]), state=None)
    assert reason == "no_index_resolved"


def test_diagnose_candidate_build_failure_no_target_resolved(mod, monkeypatch):
    from ptcg_ai.ml_policy import ogerpon_strategy as STRAT

    monkeypatch.setattr(STRAT, "_resolve_single_prize_first_action",
                        lambda *a, **k: (0, None))
    mod._W["STRAT"] = STRAT
    reason = mod._diagnose_candidate_build_failure(
        obs=None, learner_index=0, config={}, deck_l=[], trigger="promote",
        select=SimpleNamespace(option=[1, 2]), state=None)
    assert reason == "no_target_resolved"


def test_diagnose_candidate_build_failure_index_out_of_range(mod, monkeypatch):
    from ptcg_ai.ml_policy import ogerpon_strategy as STRAT

    monkeypatch.setattr(STRAT, "_resolve_single_prize_first_action",
                        lambda *a, **k: (99, object()))
    mod._W["STRAT"] = STRAT
    reason = mod._diagnose_candidate_build_failure(
        obs=None, learner_index=0, config={}, deck_l=[], trigger="retreat",
        select=SimpleNamespace(option=[1, 2]), state=None)
    assert reason == "index_out_of_range"


# ===========================================================================
# _process_task: trigger別quotaが単純な共通上限に退行していないこと(source inspection)
# ===========================================================================

def test_process_task_uses_per_trigger_quota_not_shared_counter(mod):
    """外部レビュー後の追加指摘への回帰: trigger種別ごとの保存数を、trigger別の独立した
    カウンタ(``trigger_saved_count[trigger]``)で判定すること。``max_states_per_game``の
    単純な共通カウンタ1つだけがtrigger検出のゲートになっている(=main_attachが独占し、
    promote/retreatが収集できなくなる)状態に戻っていないことを確認する。
    ``max_states_per_game``自体は全trigger合算の安全上限として残ってよい。
    """
    src = inspect.getsource(mod._process_task)
    assert "trigger_saved_count[trigger]" in src
    assert "max_states_per_trigger_per_game" in src
    assert "trigger_saved_count[trigger] >= max_states_per_trigger_per_game" in src, (
        "trigger別の保存数がtrigger別quotaと比較されていること")


def test_process_task_deduplicates_same_state_id_within_game(mod):
    src = inspect.getsource(mod._process_task)
    assert "seen_state_ids_this_game" in src


def test_process_task_signature_has_trigger_quota_param(mod):
    sig = inspect.signature(mod._process_task)
    assert "max_states_per_trigger_per_game" in sig.parameters
    assert "max_states_per_game" in sig.parameters


def test_process_triggered_state_records_sample_id_and_hidden_state_hash(mod):
    src = inspect.getsource(mod._process_triggered_state)
    assert '"sample_id": sample_id' in src
    assert '"hidden_state_hash"' in src
    assert 'f"{sample_id}:{det_id}"' in src, "pair_idはsample_idから作ること"
