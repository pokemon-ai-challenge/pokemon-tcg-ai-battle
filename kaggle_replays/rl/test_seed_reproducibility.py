"""seed再現性の調査結果を固定するテスト(T1.1補修 item 4、D2.1で範囲を訂正)。

**cgエンジン(cg.dll/libcg.so)にはPythonから呼べるseed設定関数が無い。**
``cg.api.lib`` に ``seed``/``rand`` を含む関数名が存在しないことを確認する
(``all_card_data``等と同様、ctypes経由でロードした関数一覧から確認できる)。
これは自分たちのコード(cg.api)に対する確定的な検査なので、pytestの
回帰テストとして残す。

**[D2.1] 「同じseedを複数回投げると盤面展開が毎回変わる」という確率的な観察は
pytestから外し、`diagnose_seed_reproducibility.py` の診断スクリプトへ移した**
(one-shotで偶然一致する確率がゼロではない主張をpass/fail条件にしない)。
その観察から言えるのは「マルチプロセスのタスク割当順序だけが原因、という仮説は
排除できる」ことまでで、根本原因(cgエンジン内部の乱数かどうか)を断定するもの
ではない(design.md §9.1.1参照)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def cg_lib():
    try:
        from cg.api import lib
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cg engine unavailable: {exc}")
    return lib


def test_cg_engine_exposes_no_seed_function(cg_lib):
    """cg.dll/libcg.soにPythonから呼べるseed設定APIが無いことを確認する
    (これが「同じseedでも試合展開が再現しない」の根本原因)。"""
    names = [n.lower() for n in dir(cg_lib)]
    assert not any("seed" in n for n in names), (
        "cgエンジンにseed関連の関数が見つかった。再現性の前提が変わった可能性がある。")


# 「同じseedを複数回投げると盤面展開が毎回変わる」ことの確率的な確認は
# diagnose_seed_reproducibility.py に移した(D2.1)。ここには置かない。
