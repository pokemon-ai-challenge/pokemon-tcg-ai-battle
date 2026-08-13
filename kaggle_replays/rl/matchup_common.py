"""アーキタイプ別 climb 専門Policy学習の共通ヘルパー。

train_matchup_specialist.py / collect_matchup_specialist.py / eval_matchup_specialist.py /
run_specialist_batch.py / overnight_specialists.py から共有する。

既存 train_field.py / train_v3.py の「相対パスを Path.name に落として WDIR 直下に
決め打ちする」パターンはバグ(指定サブディレクトリが無視される)なので、ここでは
一切使わない。相対パスは常にリポジトリルート基準で解決する。
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
WDIR = ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

for _p in (str(_HERE), str(ROOT), str(ROOT / "sample_submission"), str(ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# train_field.py の FIELD (share-sampled anchor 相手) をそのまま anchor "auto" 設定に流用する。
FIELD_ANCHOR = [
    ("mega_lucario_ex", 1257), ("archaludon_ex", 1078), ("crustle", 737),
    ("dragapult_ex", 625), ("marnie_grimmsnarl_ex", 591),
    ("rocket_mewtwo_ex", 247), ("shirona_garchomp_ex", 181),
]

# 汎化(generalist)学習で使う11アーキタイプのメタシェア推定。
# 出典: gen2 BC 学習時の「学習デッキ数」(commit d5b0453、2026-08-11 取得の 2402試合/4804デッキ)。
# ただし archaludon_ex / rocket_mewtwo_ex は少数クラス狙い撃ちの追加取得で件数が水増しされて
# いるため、水増し前の自然出現数(57 / 26)を採用する。これが素直なメタ分布の推定になる。
GEN2_META_SHARE = {
    "marnie_grimmsnarl_ex": 1082,
    "alakazam": 891,
    "mega_lucario_ex": 403,
    "dragapult_ex": 381,
    "mega_froslass_ex": 353,
    "ogerpon_teal_ex": 310,
    "crustle": 287,
    "shirona_garchomp_ex": 158,
    "omatsuri_ondo": 138,
    "archaludon_ex": 57,
    "rocket_mewtwo_ex": 26,
}

TARGET_ARCHETYPES = [
    "alakazam", "marnie_grimmsnarl_ex", "ogerpon_teal_ex", "shirona_garchomp_ex",
    "omatsuri_ondo", "mega_froslass_ex", "mega_lucario_ex", "rocket_mewtwo_ex",
    "dragapult_ex", "archaludon_ex", "crustle",
]

FROZEN_CLIMB_WEIGHTS = WDIR / "policy_weights_alakazam_rl_climb.json"
FROZEN_CLIMB_SHA256 = "395b02486f851be9c668e560449da5624f501b3b3131612689a6c8e2876f5196"

# 823.5 提出時の元祖 Plan A(キチキギスex無し)。学習は当初これを learner デッキとして開始したが、
# 2026-08-13 01:xx にユーザー指示でキチキギスex採用の新デッキへ切り替えた
# (このファイル自体は変更していない、識別・比較用の参照として残す)。
ORIGINAL_PLAN_A_DECK = ROOT / "sample_submission" / "models" / "climb_lb823" / "deck.csv"
ORIGINAL_PLAN_A_SHA256 = "8ae7a618b2655669e45eeb296df2d9271a38cf4977fd4d7dbf75bfaf386a84c2"

# 現在の learner デッキ(キチキギスex採用版、sample_submission/deck.csv と同一内容)。
# train_matchup_specialist.py の --learner-deck 既定値・manifest の "matches" 判定はこちらを見る。
FROZEN_PLAN_A_DECK = ROOT / "sample_submission" / "deck.csv"
FROZEN_PLAN_A_SHA256 = "90b5ed2d1572bd9cc51f1f1d7625740a353c1bbb8921fa7644b7e4cbc360ec45"

# 今回の初期値・学習に使ってはいけないもの(絶対条件、混同防止用)。
FORBIDDEN_INITIAL_WEIGHTS_NAMES = {"policy_weights.json", "policy_weights_alakazam_rl_vscrustle.json"}


def resolve_path(value: str | os.PathLike | None, default: Path | None = None) -> Path | None:
    """相対パスは常にリポジトリルート基準で解決する(Path.name への切り詰めは絶対にしない)。"""
    if value is None:
        return default
    p = Path(value)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def append_jsonl(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc!r})"


def env_info() -> dict:
    import torch
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


# ---------------------------------------------------------------------------
# デッキ検証
# ---------------------------------------------------------------------------

_ACE_SPEC_IDS: set[int] | None = None
_BASIC_ENERGY_IDS: set[int] | None = None


def _load_card_rule_sets() -> tuple[set[int], set[int]]:
    """(ace_spec_ids, basic_energy_ids) を返す(基本エネルギーは CLAUDE.md により無制限)。"""
    global _ACE_SPEC_IDS, _BASIC_ENERGY_IDS
    if _ACE_SPEC_IDS is not None and _BASIC_ENERGY_IDS is not None:
        return _ACE_SPEC_IDS, _BASIC_ENERGY_IDS
    ace_ids: set[int] = set()
    basic_energy_ids: set[int] = set()
    csv_path = ROOT / "data" / "JP_Card_Data.csv"
    try:
        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    cid = int(row["カード ID"])
                except (KeyError, ValueError):
                    continue
                rule = (row.get("ルール") or "").strip()
                if "ACE SPEC" in rule:
                    ace_ids.add(cid)
                # 実体は「カテゴリ」列ではなく「ポケモンの進化の段階/エネルギー・トレーナーズの
                # 種類」列に "基本エネルギー" が入る(CSVヘッダ参照)。
                kind = (row.get("ポケモンの進化の段階/エネルギー・トレーナーズの種類") or "").strip()
                if kind == "基本エネルギー":
                    basic_energy_ids.add(cid)
    except OSError:
        pass
    _ACE_SPEC_IDS, _BASIC_ENERGY_IDS = ace_ids, basic_energy_ids
    return ace_ids, basic_energy_ids


def validate_deck(deck_ids: list[int]) -> list[str]:
    """デッキルール違反のリストを返す(空なら合法)。基本エネルギーは枚数無制限。"""
    errors = []
    if len(deck_ids) != 60:
        errors.append(f"deck must have 60 cards, got {len(deck_ids)}")
    ace_ids, basic_energy_ids = _load_card_rule_sets()
    from collections import Counter
    counts = Counter(deck_ids)
    for cid, n in counts.items():
        if cid in basic_energy_ids:
            continue
        limit = 1 if cid in ace_ids else 4
        if n > limit:
            errors.append(f"card {cid} appears {n} times (limit {limit})")
    return errors


def read_deck(path: Path) -> list[int]:
    text = Path(path).read_text(encoding="utf-8")
    ids = []
    for raw in text.replace(",", "\n").splitlines():
        v = raw.strip()
        if not v or v.startswith("#"):
            continue
        ids.append(int(v))
    return ids


def deck_multiset_jaccard(a: list[int], b: list[int]) -> float:
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


def discover_archetype_decks(archetype: str, explicit: list[str] | None = None) -> list[dict]:
    """[{"path": Path, "deck": list[int], "errors": [...], "dup_of": Path|None}] を返す。

    同一アーキタイプ内のデッキ間で multiset Jaccard を計算し、完全一致(Jaccard==1.0)は
    "dup_of" に最初に見つかった方のパスをセットして二重サンプリングを避ける。
    """
    if explicit:
        paths = [resolve_path(p) for p in explicit]
    else:
        d = DECKDIR / archetype
        paths = sorted(Path(p) for p in glob.glob(str(d / "*.csv")))
    decks = []
    for p in paths:
        try:
            ids = read_deck(p)
            errs = validate_deck(ids)
        except Exception as exc:  # noqa: BLE001
            ids, errs = [], [f"failed to read: {exc!r}"]
        decks.append({"path": p, "deck": ids, "errors": errs, "dup_of": None,
                      "sha256": sha256_file(p) if p.exists() else None})
    # 完全一致(重複)検出。
    for i, di in enumerate(decks):
        if di["dup_of"] is not None or di["errors"]:
            continue
        for dj in decks[:i]:
            if dj["dup_of"] is not None or dj["errors"]:
                continue
            if deck_multiset_jaccard(di["deck"], dj["deck"]) >= 0.999:
                di["dup_of"] = str(dj["path"])
                break
    return decks


def near_duplicate_report(decks: list[dict], threshold: float = 0.85) -> list[dict]:
    report = []
    n = len(decks)
    for i in range(n):
        for j in range(i + 1, n):
            if decks[i]["errors"] or decks[j]["errors"]:
                continue
            sim = deck_multiset_jaccard(decks[i]["deck"], decks[j]["deck"])
            if sim >= threshold:
                report.append({"a": str(decks[i]["path"]), "b": str(decks[j]["path"]),
                               "jaccard": sim, "exact": sim >= 0.999})
    return report


# ---------------------------------------------------------------------------
# 相手Policy解決
# ---------------------------------------------------------------------------

def resolve_opponent_weights(archetype: str) -> dict:
    """優先順位: policy_weights_<arch>_g2.json -> policy_weights_<arch>.json -> 代替なし。

    戻り: {"path": Path|None, "tier": str, "sha256": str|None, "error": str|None}
    """
    from ptcg_ai.learning.policy_model import PolicyModel  # noqa: E402

    candidates = [
        (WDIR / f"policy_weights_{archetype}_g2.json", "g2"),
        (WDIR / f"policy_weights_{archetype}.json", "plain"),
    ]
    for path, tier in candidates:
        if not path.exists():
            continue
        try:
            pm = PolicyModel(path)
        except Exception as exc:  # noqa: BLE001
            continue
        if pm.is_ready:
            return {"path": path, "tier": tier, "sha256": sha256_file(path), "error": None}
    return {"path": None, "tier": "missing", "sha256": None,
            "error": f"no usable opponent policy for archetype={archetype!r}"}


def resolve_alakazam_mirror_candidates() -> list[dict]:
    """フーディンミラー用の相手候補(仕様で明示された2つを優先)。"""
    out = []
    for path in (WDIR / "policy_weights_alakazam_g2.json", FROZEN_CLIMB_WEIGHTS):
        if path.exists():
            out.append({"path": path, "sha256": sha256_file(path)})
    return out


def build_generalist_opp_specs(archetypes: list[str] | None = None,
                               shares: dict[str, float] | None = None) -> tuple[list, list, list]:
    """汎化学習用: 全11アーキタイプの「模倣Policy × そのアーキの全合法デッキ」を相手にする。

    1アーキタイプが複数デッキ(01..05.csv)を持つので、そのアーキのメタシェアを
    デッキ数で等分して各デッキへ配る(層化サンプリング。デッキ数の多いアーキが
    不当に重くならない)。完全重複デッキは除外して二重重み付けを避ける。

    戻り: (opp_specs [(weights_path_str, deck_ids)], shares [float], meta [dict])
    """
    archetypes = archetypes or list(TARGET_ARCHETYPES)
    shares = shares or GEN2_META_SHARE
    opp_specs: list = []
    out_shares: list[float] = []
    meta: list[dict] = []
    for arch in archetypes:
        wres = resolve_opponent_weights(arch)
        if wres["path"] is None:
            meta.append({"archetype": arch, "status": "skipped", "reason": wres["error"]})
            continue
        decks = discover_archetype_decks(arch)
        usable = [d for d in decks if not d["errors"] and d["dup_of"] is None]
        if not usable:
            meta.append({"archetype": arch, "status": "skipped", "reason": "no usable deck"})
            continue
        arch_share = float(shares.get(arch, 0.0))
        if arch_share <= 0:
            meta.append({"archetype": arch, "status": "skipped", "reason": "zero meta share"})
            continue
        per_deck = arch_share / len(usable)
        for d in usable:
            opp_specs.append((str(wres["path"]), d["deck"]))
            out_shares.append(per_deck)
        meta.append({"archetype": arch, "status": "ok", "weights": str(wres["path"]),
                     "weights_tier": wres["tier"], "weights_sha256": wres["sha256"],
                     "meta_share": arch_share, "n_decks": len(usable),
                     "share_per_deck": per_deck,
                     "decks": [{"path": str(d["path"]), "sha256": d["sha256"]} for d in usable],
                     "skipped_decks": [{"path": str(d["path"]), "errors": d["errors"],
                                        "dup_of": d["dup_of"]}
                                       for d in decks if d["errors"] or d["dup_of"]]})
    return opp_specs, out_shares, meta


def build_anchor_opp_specs(target_archetype: str, exclude: set[str] | None = None) -> tuple[list, list, list]:
    """train_field.FIELD ベースの anchor 相手一覧を構築する。

    戻り: (opp_specs [(weights_path_str, deck_ids)], shares [float], meta [dict])
    ロード不能な anchor アーキは警告付きでスキップする(全体は止めない)。
    """
    exclude = exclude or set()
    opp_specs, shares, meta = [], [], []
    for arch, share in FIELD_ANCHOR:
        if arch == target_archetype or arch in exclude:
            continue
        wres = resolve_opponent_weights(arch)
        if wres["path"] is None:
            meta.append({"archetype": arch, "status": "skipped", "reason": wres["error"]})
            continue
        deck_path = DECKDIR / arch / "01.csv"
        if not deck_path.exists():
            meta.append({"archetype": arch, "status": "skipped", "reason": f"no deck at {deck_path}"})
            continue
        deck_ids = read_deck(deck_path)
        errs = validate_deck(deck_ids)
        if errs:
            meta.append({"archetype": arch, "status": "skipped", "reason": f"invalid deck: {errs}"})
            continue
        opp_specs.append((str(wres["path"]), deck_ids))
        shares.append(float(share))
        meta.append({"archetype": arch, "status": "ok", "weights": str(wres["path"]),
                    "weights_tier": wres["tier"], "deck": str(deck_path)})
    return opp_specs, shares, meta
