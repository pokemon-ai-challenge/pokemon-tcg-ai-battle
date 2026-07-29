"""ptcg_ai.learning.encoder.encode_obs_dict が学習時と同じ特徴ベクトルを出すことを直接確認する。

背景(2026-07-29 の価値関数再学習時の事故): 学習側 build_features.py と実行側 value_model.py は
どちらも同じ encode_obs_dict を呼ぶが、encode_obs_dict は
ptcg_ai/board_evaluation/attack_features.py の resolve_damage() 等に依存している。
モデル学習後にこの依存先だけが変更され、encode_obs_dict の出力(特徴の値そのもの)が学習時と
実行時で食い違った(例: opp_active_best_attack_damage が 26 -> 260 に変化、
*_is_likely_ko_next_turn が軒並み反転)ことがあった。

このとき既存の golden test (test_value_model.py::test_golden_predictions_match) は「最終予測確率」
の一致しか見ていないため、原因が (a) 特徴抽出そのものが変わったのか (b) フォワードパス
(標準化/重み/較正)の実装がズレたのか、切り分けに時間がかかった。

本テストは kaggle_replays/value_net/sample_predictions.json に保存済みの feature_vector
(学習時に encode_obs_dict(observation) が実際に出した値、build_features.py が features.npz に
書き出したもの)と、同じ observation から「今の」encode_obs_dict が出す値を直接突き合わせる。
これが失敗すれば原因は特徴抽出側(encoder / その依存先)に絞り込める。golden test が失敗して
本テストが通れば、原因はフォワードパス/較正側に絞り込める。

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
"""

import json
import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_REPO_ROOT = SAMPLE_SUBMISSION_ROOT.parent
_SAMPLE_PREDICTIONS = _REPO_ROOT / "kaggle_replays" / "value_net" / "sample_predictions.json"

# feature_vector は features.npz 保存時に float32 へ丸められている(build_features.py の
# `X = np.asarray(X_rows, dtype=np.float32)`)。今 encode_obs_dict を呼んだ結果は float64 精度の
# python float なので、float32 の丸め誤差ぶんだけ緩めた許容差で比較する。
_ABS_TOL = 1e-3
_REL_TOL = 1e-4


@pytest.fixture(scope="module")
def sample_predictions() -> list[dict]:
    if not _SAMPLE_PREDICTIONS.exists():
        pytest.skip(f"sample_predictions.json not found at {_SAMPLE_PREDICTIONS}")
    with _SAMPLE_PREDICTIONS.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def encode_obs_dict():
    try:
        from ptcg_ai.learning.encoder import encode_obs_dict as _encode
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"encoder / cg engine unavailable: {exc}")
    return _encode


def test_encoder_output_matches_saved_feature_vector(encode_obs_dict, sample_predictions):
    """保存済み feature_vector == 今の encode_obs_dict(observation) の出力(要素ごと)。"""
    assert len(sample_predictions) > 0
    mismatches = []
    for row in sample_predictions:
        saved = row["feature_vector"]
        got = encode_obs_dict(row["observation"])
        assert len(got) == len(saved), (
            f"特徴ベクトル長が変わった: episode_id={row.get('episode_id')} "
            f"saved={len(saved)} got={len(got)}(encoder に列の追加/削除があった可能性)"
        )
        for i, (g, s) in enumerate(zip(got, saved)):
            if abs(g - s) > max(_ABS_TOL, _REL_TOL * abs(s)):
                mismatches.append(
                    {
                        "episode_id": row.get("episode_id"),
                        "feature_index": i,
                        "saved": s,
                        "got": g,
                        "diff": abs(g - s),
                    }
                )
    assert not mismatches, (
        f"{len(mismatches)} 個の特徴値が保存済み feature_vector と食い違う "
        "(encode_obs_dict の依存先が学習後に変更された可能性が高い): "
        + json.dumps(mismatches[:10], ensure_ascii=False)
    )
