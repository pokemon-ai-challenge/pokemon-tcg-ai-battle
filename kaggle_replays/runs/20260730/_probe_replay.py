#!/usr/bin/env python3
"""1エピソードのリプレイDLを試み、結果トークンを1行で標準出力する。

driver から直接 kaggle.exe を spawn すると WinError 4551(アプリケーション制御
ポリシーでブロック)になる環境があるため、実績のある _common.download_replay
(python→kaggle の入れ子)を経由して叩く。

出力: SUCCESS / RATELIMITED / UNAVAILABLE / ERROR のいずれか1語。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE_REPLAYS = HERE.parent.parent
sys.path.insert(0, str(KAGGLE_REPLAYS))
from _common import download_replay  # noqa: E402

REPLAYS = HERE / "replays"


def main() -> None:
    eid = sys.argv[1]
    dest = REPLAYS / f"episode-{eid}-replay.json"
    if dest.exists():
        print("SUCCESS")
        return
    try:
        download_replay(int(eid), REPLAYS)
        print("SUCCESS")
    except subprocess.CalledProcessError as e:
        out = (e.stderr or "") + (e.stdout or "")
        if "429" in out or "Too Many Requests" in out:
            print("RATELIMITED")
        elif "404" in out or "Not Found" in out or "403" in out or "Forbidden" in out:
            print("UNAVAILABLE")
        else:
            print("ERROR")
    except Exception:  # noqa: BLE001
        print("ERROR")


if __name__ == "__main__":
    main()
