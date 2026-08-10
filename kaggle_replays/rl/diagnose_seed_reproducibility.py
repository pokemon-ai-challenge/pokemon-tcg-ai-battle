"""seed再現性の調査スクリプト(D2.1: 確率的pytestから診断スクリプトへ移動)。

**これはpytestの回帰テストではない。** 「同じ(learner_index, seed)を複数回
投げたら毎回結果が変わる」という主張はcgエンジン(外部・変更不可)の挙動に
関する経験的観察であり、run毎に理論上one-shotで一致する確率がゼロではない
(=厳密なpass/fail条件としてpytestに置くべき性質の主張ではない)。
再現性の前提を確認したいときに手動で実行する診断スクリプトとして置く。

単一プロセス・マルチプロセス無しで、同じ(learner_index, seed)タプルを
``collect_tokens._play_one`` に複数回渡し、決定点数が毎回変わることを観察する。
**この観察から言えるのは「マルチプロセスのタスク割当順序だけが原因、という
仮説は排除できる」ことまで**(単一プロセス・逐次呼び出しでも変動するため)。
「Pythonから制御できないネイティブ側(cgエンジン内部)の乱数が原因」は
有力な仮説だが、cgのソース自体を確認していないため断定はしない
(design.md §9.1.1参照)。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(n_trials: int = 5) -> None:
    import collect_tokens as ct
    from run_league import read_deck_csv_file

    deck_path = (_ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
                / "alakazam" / "01.csv").resolve()
    weights = (_ROOT / "sample_submission" / "ptcg_ai" / "learning" / "policy_weights.json").resolve()
    if not deck_path.exists() or not weights.exists():
        print(f"デッキ/教師重みが無い: {deck_path} / {weights}")
        return
    deck = read_deck_csv_file(str(deck_path))

    ct._init_worker(str(weights), deck, deck, 1.0, False)
    decision_counts = []
    for _ in range(n_trials):
        r = ct._play_one((0, 555))  # 毎回同じ(learner_index, seed)
        assert r["error"] is None
        decision_counts.append(len(r["steps"]))

    print(f"同じ(learner_index=0, seed=555)を{n_trials}回: 決定点数={decision_counts}")
    if len(set(decision_counts)) > 1:
        print("→ 単一プロセス・逐次呼び出しでも結果が変動する。"
              "マルチプロセスのタスク割当順序が『唯一の』原因ではないことが分かる"
              "(根本原因の断定はできない。design.md §9.1.1参照)。")
    else:
        print(f"→ {n_trials}回とも同じ決定点数だった。再現性の前提が変わった"
              "可能性がある(cgエンジンがseedを尊重するようになった等)。要再調査。")


if __name__ == "__main__":
    main()
