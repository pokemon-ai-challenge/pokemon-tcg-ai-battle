"""baseline package の policy_weights.json(と任意で deck.csv)だけを差し替えた候補 tarball。

baseline tarball は上書きしない。他のファイルは1バイトも触らず、差分が指定したものだけで
あることを機械的に検証する(§28/§39/§41)。

--deck を付けるとデッキも差し替える(既定=なし=従来どおり policy weights のみ)。
デッキを替える候補では「差分は weights と deck.csv の2つ」であることを検証する。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import pathlib
import tarfile

TARGET = "ptcg_ai/learning/policy_weights.json"
DECK_TARGET = "deck.csv"


def norm(name: str) -> str:
    return name.lstrip("./").replace(chr(92), "/")


def inventory(path) -> dict:
    d = {}
    with tarfile.open(path) as t:
        for m in t.getmembers():
            if m.isfile():
                d[norm(m.name)] = hashlib.sha256(t.extractfile(m).read()).hexdigest()[:16]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="build_ready/submission_climb.tar.gz")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--deck", default=None, help="差し替える deck.csv(既定=差し替えない)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base, out = pathlib.Path(args.base), pathlib.Path(args.out)
    replace = {TARGET: pathlib.Path(args.weights).read_bytes()}
    if args.deck:
        replace[DECK_TARGET] = pathlib.Path(args.deck).read_bytes()

    with tarfile.open(base) as tin, tarfile.open(out, "w:gz") as tout:
        for m in tin.getmembers():
            new_bytes = replace.get(norm(m.name)) if m.isfile() else None
            if new_bytes is not None:
                info = tarfile.TarInfo(m.name)
                info.size, info.mode, info.mtime = len(new_bytes), m.mode, m.mtime
                info.uid, info.gid = m.uid, m.gid
                tout.addfile(info, io.BytesIO(new_bytes))
            else:
                tout.addfile(m, tin.extractfile(m) if m.isfile() else None)

    a, b = inventory(base), inventory(out)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    print("package sha256[:16] =", hashlib.sha256(out.read_bytes()).hexdigest()[:16])
    print("size =", out.stat().st_size, "bytes, files =", len(b))
    print("differing files vs baseline package:", diff)
    expected = sorted(replace)
    print("expected diff:", expected)
    print("diff matches expected:", diff == expected)
    for k in (TARGET, "deck.csv", "configs/abl_5_full.json", "main.py"):
        print("  ", b.get(k), k)


if __name__ == "__main__":
    main()
