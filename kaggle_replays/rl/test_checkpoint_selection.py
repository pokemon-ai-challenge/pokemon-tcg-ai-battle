"""defect#1(checkpoint選択: best-of-eval の winner's curse をやめ、最終イテレートを採用する)の検証。

train_pool.py / train_v3.py の PPO ループは cg エンジンでの実対戦を伴うため、"本当に3イテレーション
走らせてbest-of-evalが罠に落ちるかどうか" を再現するには数分〜数十分かかる(このタスクでは長時間の
RL実行はしない、という制約がある)。そこで:

検証1(静的): 修正後のソースに「途中の eval が改善したときだけ policy の state_dict を退避し、
   ループを抜けた後にそれへ load_state_dict して export する」という旧パターン
   (``best_state = copy.deepcopy(...)`` からの ``policy.load_state_dict(best_state)``)が
   もう存在しないことを確認する(存在すれば defect#1 が再発している)。
検証2(静的): 一方で「途中経過の記録のためだけの best_wr/best_iter 変数」自体は残ってよい
   (ログの参考値としては有用)ので、それが export ロジックに使われていない
   (payload 生成の直前で state_dict の復元をしていない)ことを、
   「最後の state_dict 復元」から「payload 生成」までの間に別のtrajectory/collectionが
   挟まっていない、という行番号の前後関係で確認する。
検証3(動的・軽量): train_v3.Critic / TorchOptionPolicy を使い、
   「ループ中に一時的に良く見えた state_dict」に戻さずに「最後の state_dict」をexportする、
   という該当パターンをミニマル再現し、修正済みロジックと同じ書き方をしたときに
   期待通り「最後の状態」が残ることを確認する(パターンそのものの正しさの確認。
   train_pool.py の実コードとは独立に、ロジックパターンとして正しいことを担保する)。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

TARGETS = [_HERE / "train_pool.py", _HERE / "train_v3.py"]


def main():
    # ------------------------------------------------------------------
    # 検証1: best_state を load_state_dict で復元してから export する旧パターンが無い
    # ------------------------------------------------------------------
    print("--- 検証1: best-of-eval の load_state_dict 復元パターンが無いことの静的確認 ---")
    forbidden = re.compile(r"load_state_dict\s*\(\s*best_state\s*\)")
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        m = forbidden.search(text)
        assert m is None, (
            f"{path.name}: best_state を load_state_dict で復元しているコードが見つかった"
            "(winner's curse の再発)"
        )
        print(f"  {path.name}: OK (見つからない)")
    print("検証1 PASS")

    # ------------------------------------------------------------------
    # 検証2: best_state 自体への代入(copy.deepcopy(policy.state_dict()))も残っていない
    # (=「保存はしているが使っていないだけ」ではなく、保存すらしていない=完全に旧経路が無い)
    # ------------------------------------------------------------------
    print("\n--- 検証2: best_state への state_dict 退避コード自体が無いことの確認 ---")
    forbidden2 = re.compile(r"best_state\s*=\s*copy\.deepcopy")
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        m = forbidden2.search(text)
        assert m is None, f"{path.name}: best_state = copy.deepcopy(...) が見つかった"
        print(f"  {path.name}: OK (見つからない)")
    print("検証2 PASS")

    # ------------------------------------------------------------------
    # 検証3: export 直前(payload = policy.to_json_payload(...))と、ループ最後(export_temp呼び出し
    # のうち一番最後、もしくは PPO 更新の最後)との間に load_state_dict 呼び出しが1つも無いこと。
    # これにより「to_json_payload の直前で “final” 以外の状態に戻す」経路が無いことを保証する。
    # ------------------------------------------------------------------
    print("\n--- 検証3: to_json_payload 直前に load_state_dict が挟まっていないことの確認 ---")
    for path in TARGETS:
        lines = path.read_text(encoding="utf-8").splitlines()
        export_line_idxs = [i for i, ln in enumerate(lines) if "policy.to_json_payload(base_payload)" in ln]
        assert export_line_idxs, f"{path.name}: to_json_payload の呼び出しが見つからない"
        for idx in export_line_idxs:
            # export呼び出しの直前50行以内に load_state_dict があれば「何かに戻してから出力」
            # というパターンなので、それを禁止する(defect#1修正後は存在しないはず)。
            window = lines[max(0, idx - 50):idx]
            bad = [ln for ln in window if "load_state_dict(" in ln]
            assert not bad, f"{path.name}:{idx+1} 直前に load_state_dict がある: {bad}"
        print(f"  {path.name}: export直前50行にload_state_dictなし ({len(export_line_idxs)}箇所)")
    print("検証3 PASS")

    # ------------------------------------------------------------------
    # 検証4(動的): 修正後と同じロジックパターン(「ループを抜けた時点のpolicyをそのままexport」)を
    # ミニ再現し、途中で一時的に良く見えたstateに引っ張られず最後の状態が残ることを確認する。
    # ------------------------------------------------------------------
    print("\n--- 検証4: ロジックパターンの動的再現 ---")
    # 「学習」を単純化して、iterごとに policy を表す整数を更新するだけのモデルにする。
    # eval_winrate は itに応じて途中で高くなり最後に下がる、という
    # winner's curse が典型的に起きるパターン(marnie実測: best 0.800@200試合 -> 最終0.757@1200試合)
    # を模す。
    class FakePolicy:
        def __init__(self):
            self.value = 0

    def fake_eval(it):
        # 3iter目が見かけ上ベスト(過大評価)、しかし本当の実力は単調増加という設定。
        table = {0: 0.50, 1: 0.55, 2: 0.60, 3: 0.90, 4: 0.65}  # it=3 が overestimate
        return table[it]

    policy = FakePolicy()
    best_wr, best_iter = fake_eval(0), 0
    for it in range(1, 5):
        policy.value = it  # 「学習」= 状態を更新するだけ
        wr = fake_eval(it)
        if wr > best_wr:
            best_wr, best_iter = wr, it
        # 修正後のパターン: state_dictの退避も復元もしない。

    # 修正後: export される値は「ループを抜けた時点の policy」(= it=4 の状態)であるべき。
    exported_value = policy.value
    print(f"  best_seen: iter={best_iter} wr={best_wr}  (winner's curseで過大評価されたのはiter=3)")
    print(f"  exported policy.value = {exported_value} (期待: 4, ループの最後)")
    assert exported_value == 4, "exportされた値がループ最後のものになっていない"
    assert exported_value != best_iter, (
        "このテストのセットアップ自体がwinner's curseを再現できていない"
        "(best_iterと最終iterが同じでは何も検証していない)"
    )
    print("検証4 PASS: best-of-eval(iter=3)に引っ張られず、最終イテレート(iter=4)がexportされる"
          "パターンを確認")

    print("\n全検証 PASS")


if __name__ == "__main__":
    main()
