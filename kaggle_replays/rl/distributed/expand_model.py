"""学習済みモデルの入力特徴を増やす(学習をやり直さずに)。

新しい特徴に対応する重みを **0 で初期化**すれば、拡張後のモデルの出力は拡張前と
完全に一致する。0 を掛けた項は何であれ寄与しないため。そこから学習を続ければ、
ゼロだった重みが育って新しい情報を使い始める。模倣学習からやり直す必要はない。

中間層を広げる場合も同様に、増やしたユニットの**出力側**の重みを 0 にすれば
出力は変わらない(入力側はランダムに散らす。全部同じ値だと対称性が崩れず学習が進まない)。

critic と Adam の内部状態も同じ形に拡張する。ここを揃えないと学習を再開できない。

    python expand_model.py --run-dir ../runs/fuudin_v1 --profile fuudin_v2 [--hidden 128]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import common as C  # noqa: E402

sys.path.insert(0, str(C.REPO_ROOT / "sample_submission"))
sys.stdout.reconfigure(encoding="utf-8")


def expand_policy(payload: dict, n_new: int, new_hidden: int | None,
                  rng: np.random.Generator) -> dict:
    out = copy.deepcopy(payload)
    std = out["standardization"]
    n_state = len(std["state_mean"])

    # 新特徴の標準化は平均0・標準偏差1。std=0 にすると encoder 側で 0 に潰される。
    std["state_mean"] = list(std["state_mean"]) + [0.0] * n_new
    std["state_std"] = list(std["state_std"]) + [1.0] * n_new

    # 第1層の重みは [盤面 | 選択肢 | カード埋め込み] の順。盤面の直後に 0 列を挿す。
    w0 = np.asarray(out["layers"][0]["weight"], dtype=np.float64)
    left, right = w0[:, :n_state], w0[:, n_state:]
    w0 = np.concatenate([left, np.zeros((w0.shape[0], n_new)), right], axis=1)
    b0 = np.asarray(out["layers"][0]["bias"], dtype=np.float64)
    w1 = np.asarray(out["layers"][1]["weight"], dtype=np.float64)   # (1, hidden)
    b1 = np.asarray(out["layers"][1]["bias"], dtype=np.float64)

    if new_hidden is not None:
        h = w0.shape[0]
        if new_hidden < h:
            raise SystemExit(f"中間層は減らせない({h} -> {new_hidden})")
        add = new_hidden - h
        if add:
            # 入力側はランダム(対称性を崩す)、出力側は 0(出力を変えない)
            scale = 1.0 / np.sqrt(w0.shape[1])
            w0 = np.concatenate([w0, rng.normal(0.0, scale, (add, w0.shape[1]))], axis=0)
            b0 = np.concatenate([b0, np.zeros(add)])
            w1 = np.concatenate([w1, np.zeros((w1.shape[0], add))], axis=1)

    out["layers"][0] = {"weight": w0.tolist(), "bias": b0.tolist()}
    out["layers"][1] = {"weight": w1.tolist(), "bias": b1.tolist()}
    return out


def expand_trainer(state: dict, n_state_old: int, n_new: int) -> dict:
    """critic(盤面特徴だけを見る)と、両方の Adam の内部状態を拡張する。"""
    st = copy.deepcopy(state)

    def grow_in(t: torch.Tensor) -> torch.Tensor:
        """(out, in) の in 側を末尾に n_new 列足す。"""
        return torch.cat([t, torch.zeros(t.shape[0], n_new, dtype=t.dtype)], dim=1)

    ck = st["critic"]
    for key in ("mean", "std"):
        # critic の標準化バッファ。std は 0 にすると 0 で潰れるので 1 を足す。
        pad = torch.zeros(n_new) if key == "mean" else torch.ones(n_new)
        ck[key] = torch.cat([ck[key], pad.to(ck[key].dtype)])
    first = "net.0.weight"
    if ck[first].shape[1] != n_state_old:
        raise SystemExit(f"critic の入力次元が想定と違う: {ck[first].shape[1]} != {n_state_old}")
    ck[first] = grow_in(ck[first])

    # Adam: 対応するパラメータの moment も同じ形にする。
    #   opt_policy の param 0 = card_embedding, 1 = _linears.0.weight ... 順序は
    #   policy.parameters() の順。形で判定するほうが取り違えにくい。
    for opt_key, targets in (("opt_policy", None), ("opt_value", None)):
        opt = st[opt_key]
        for pid, slot in opt["state"].items():
            for mk in ("exp_avg", "exp_avg_sq"):
                t = slot.get(mk)
                if t is None or t.dim() != 2:
                    continue
                # 第1層の重みだけ列が増える。入力次元が拡張前と一致するものを対象にする。
                if opt_key == "opt_value" and t.shape[1] == n_state_old:
                    slot[mk] = grow_in(t)
                elif opt_key == "opt_policy" and t.shape[1] >= n_state_old:
                    # policy 第1層は (hidden, 盤面+選択肢+埋め込み)。盤面の直後に挿す。
                    left, right = t[:, :n_state_old], t[:, n_state_old:]
                    slot[mk] = torch.cat(
                        [left, torch.zeros(t.shape[0], n_new, dtype=t.dtype), right], dim=1)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--profile", required=True, help="extended_profile_<名前>.json の <名前>")
    ap.add_argument("--hidden", type=int, default=None, help="中間層をこの数まで広げる")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ptcg_ai.learning import encoder, extended_features

    run_dir = Path(args.run_dir).resolve()
    run = C.load_run(run_dir)
    gen = run["generation"]
    prof = extended_features.load_profile(args.profile)

    mp = C.model_path(run_dir, gen)
    payload = json.loads(mp.read_text(encoding="utf-8"))
    n_state_old = len(payload["standardization"]["state_mean"])
    if n_state_old != encoder.BASE_FEATURE_COUNT:
        raise SystemExit(f"すでに拡張済みに見える(盤面特徴 {n_state_old})")

    print(f"run={run['run_id']} v{gen}")
    print(f"  プロファイル {prof.name}: 追加 {prof.count} 次元 "
          f"({n_state_old} -> {n_state_old + prof.count})")
    hidden_old = len(payload["layers"][0]["bias"])
    if args.hidden:
        print(f"  中間層: {hidden_old} -> {args.hidden}")

    rng = np.random.default_rng(args.seed)
    grown = expand_policy(payload, prof.count, args.hidden, rng)
    grown.setdefault("meta", {}).update({
        "extended_features_profile": prof.name,
        "expanded_from_generation": gen,
        "state_feature_count": n_state_old + prof.count,
        "hidden_size": len(grown["layers"][0]["bias"]),
    })

    # バックアップを取ってから同じ世代のファイルを差し替える。世代番号は変えない
    # (収集していない世代を作ると履歴と噛み合わなくなる)。
    bak = mp.with_suffix(".json.pre_expand")
    if not bak.exists():
        shutil.copyfile(mp, bak)
    tmp = mp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(grown, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, mp)
    print(f"  モデルを差し替え: {mp.name}(元は {bak.name})")

    sp = C.trainer_state_path(run_dir, gen)
    if sp.exists():
        stt = torch.load(sp, map_location="cpu", weights_only=True)
        grown_st = expand_trainer(stt, n_state_old, prof.count)
        sbak = sp.with_suffix(".pt.pre_expand")
        if not sbak.exists():
            shutil.copyfile(sp, sbak)
        tmp = sp.with_suffix(".pt.tmp")
        torch.save(grown_st, tmp)
        os.replace(tmp, sp)
        print(f"  学習状態を差し替え: {sp.name}(元は {sbak.name})")
        if args.hidden:
            # 中間層を広げると policy のパラメータの形が変わるので、Adam の moment は
            # 使えない。learner 側が形を検査して作り直す(critic 側は引き継がれる)。
            # Adam の moment は数ステップで温まり直すので実害は小さい。
            print("  注意: 中間層を広げたので policy 側の Adam は learner が作り直します"
                  "(critic 側は引き継がれます)")
    else:
        print("  学習状態が無いのでスキップ")

    print("\n拡張しました。次の世代からこの特徴で収集・学習されます。")


if __name__ == "__main__":
    main()
