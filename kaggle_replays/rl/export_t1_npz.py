"""T1 checkpoint(torch)をnumpy推論用npzへ変換する(ローカル実行専用、torch必須)。
提出物(sample_submission/)自体はtorchに依存しない
(``sample_submission/ptcg_ai/learning/t1_numpy_policy.py``参照)。

state_dictのキーの``.``は npz のキーとして安全な``__``に置き換えて保存する
(``t1_numpy_policy.load_t1_numpy_policy``で読み込み時に戻す)。
正規化統計(``*_mean``/``*_std``)もstate_dictのバッファとして含まれているため、
同じnpzに一緒に入る(別ファイル管理不要)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="train_distill_t1.pyのcheckpoint、またはtrain_ppo_t1.save_ppo_checkpointの"
                         "checkpoint(model_state_dictのみ持つ。model_configはbuffer込みの"
                         "state_dictから復元できる部分以外は既知の値をハードコード)。")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"]
    arrays = {k.replace(".", "__"): v.cpu().numpy() for k, v in state_dict.items()}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    print(f"書き出し: {args.out}")
    if "model_config" in payload:
        print(f"model_config: {payload['model_config']}")
        print(f"feature_profile: {payload['feature_profile']}")
        print(f"normalization_mode: {payload['normalization_mode']}")
    else:
        print("PPO checkpoint形式(model_config等のメタ情報は無し。"
             "seed1蒸留checkpointのmodel_config/profileを流用する想定)。")
        print(f"payload keys: {list(payload.keys())}")


if __name__ == "__main__":
    main()
