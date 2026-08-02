"""分散 Self-Play の共通部品(世代管理・シャード入出力・整合性検査)。

**このモジュールは torch を import しない。** worker 側(Colab / Kaggle)は torch 無しで
動く必要があるため。torch を使うのは learner.py だけ。

用語:
  世代 (generation) — 「この重みで集めた経験」を識別する通し番号。model_v0, model_v1, ...
  シャード (shard)  — 1台の worker が1世代ぶんに集めた経験データ1ファイル。

ディレクトリ構成(run_dir):

    runs/<run_id>/
    ├── run.json                    実験設定 + 現在の世代。**全 worker に配る唯一の設定源**
    ├── models/model_v0.json ...    各世代のモデル重み(配布物)
    ├── shards/v0/<worker_id>.npz   worker が書き出す経験データ
    ├── shards/consumed/v0/...      PPO 更新に使い終わったシャードの退避先
    ├── state/trainer_v1.pt         critic + optimizer の状態(学習PCのみ・torch形式)
    └── history.jsonl               世代ごとの記録(1行1世代)

worker と学習PCの間でやりとりするのは以下だけ:
  学習PC -> worker : run.json と models/model_v<gen>.json (合計 約1MB)
  worker -> 学習PC : shards/v<gen>/<worker_id>.npz (250試合で 約1.2MB)
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent.parent
for _p in (str(_HERE.parent), str(REPO_ROOT), str(REPO_ROOT / "sample_submission"),
           str(REPO_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# シャード形式のバージョン。中身の並びを変えたら上げる(古いシャードは弾かれる)。
# /2 で試合ごとの相手番号(opp)を追加した。形式を上げないと、相手番号の無い古いシャードが
# 「全部同じ相手」として黙って混ざり、相手ごとの正規化が効かないまま学習が進んでしまう。
SHARD_FORMAT = "ptcg-rl-shard/2"

# 乱数シードの割り当て。世代 g・worker 番号 i の試合は
#   [g*SEED_STRIDE_GEN + i*SEED_STRIDE_WORKER, ... + games) を使う。
# ストライドが試合数・worker 数より十分大きいので、世代間でも worker 間でも範囲が重ならない。
SEED_STRIDE_GEN = 100_000_000
SEED_STRIDE_WORKER = 1_000_000
MAX_WORKERS = SEED_STRIDE_GEN // SEED_STRIDE_WORKER      # 100
MAX_GAMES_PER_WORKER = SEED_STRIDE_WORKER                # 1,000,000

# 評価用の試合は、どの世代の収集ともシード範囲が被らないところから取る。
EVAL_SEED_BASE = 1_000_000_000_000

DECK_ROOT = REPO_ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
LEARNING_DIR = REPO_ROOT / "sample_submission" / "ptcg_ai" / "learning"


# ---------------------------------------------------------------- パス規約
def run_json_path(run_dir: Path) -> Path:
    return Path(run_dir) / "run.json"


def model_path(run_dir: Path, gen: int) -> Path:
    return Path(run_dir) / "models" / f"model_v{gen}.json"


def shard_dir(run_dir: Path, gen: int) -> Path:
    return Path(run_dir) / "shards" / f"v{gen}"


def consumed_dir(run_dir: Path, gen: int) -> Path:
    return Path(run_dir) / "shards" / "consumed" / f"v{gen}"


def trainer_state_path(run_dir: Path, gen: int) -> Path:
    """世代 gen のモデルと対になる critic / optimizer の状態。"""
    return Path(run_dir) / "state" / f"trainer_v{gen}.pt"


def history_path(run_dir: Path) -> Path:
    return Path(run_dir) / "history.jsonl"


# ---------------------------------------------------------------- run.json
def load_run(run_dir: Path) -> dict:
    p = run_json_path(run_dir)
    if not p.exists():
        raise SystemExit(f"run.json が無い: {p}\n  init_run.py で作るか、--run-dir を見直す。")
    run = json.loads(p.read_text(encoding="utf-8"))
    if run.get("shard_format") != SHARD_FORMAT:
        raise SystemExit(
            f"run.json のシャード形式が古い({run.get('shard_format')} != {SHARD_FORMAT})。"
            "この run は現在のコードでは扱えない。")
    return run


def save_run(run_dir: Path, run: dict) -> None:
    run_json_path(run_dir).write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")


def append_history(run_dir: Path, record: dict) -> None:
    with history_path(run_dir).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- ハッシュ
def sha256_file(path: Path) -> str:
    """モデルファイルの中身のハッシュ。世代番号だけだと『番号は合っているが中身が違う』
    (配布ミス・転送途中の取り違え)を検出できないので、必ずこれで照合する。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- シード
def seed_base(generation: int, worker_index: int) -> int:
    if not 0 <= worker_index < MAX_WORKERS:
        raise SystemExit(f"worker 番号が範囲外: {worker_index} (0..{MAX_WORKERS-1})")
    return generation * SEED_STRIDE_GEN + worker_index * SEED_STRIDE_WORKER


# ---------------------------------------------------------------- デッキ / 相手
def resolve_deck(spec: str) -> Path:
    """run.json のデッキ指定を実ファイルへ。リポジトリからの相対パスか、
    アーキタイプ名(その場合 <archetype>/01.csv)を受け付ける。"""
    # アーキタイプ名で渡されると DECK_ROOT/<名前> はディレクトリなので、必ず is_file() で見る。
    for cand in (REPO_ROOT / spec, DECK_ROOT / spec, DECK_ROOT / spec / "01.csv"):
        if cand.is_file():
            return cand
    raise SystemExit(f"デッキが見つからない: {spec}")


def resolve_opponent_weights(run_dir: Path, spec: str | None) -> str | None:
    """相手の重み指定を実パスへ。None は「production の既定 alakazam」を意味する
    (collect_parallel が PolicyModel(None) でその重みを読む)。"""
    if spec is None:
        return None
    for cand in (Path(run_dir) / spec, REPO_ROOT / spec, LEARNING_DIR / spec, Path(spec)):
        if cand.is_file():
            return str(cand)
    raise SystemExit(f"相手の重みが見つからない: {spec}")


def split_games(total: int, opponents: list[dict]) -> list[int]:
    """対戦相手プールへ試合数を配分する。share の比で割り、端数は先頭から1つずつ足す。
    全 worker が同じ run.json を読むので、どの worker でも同じ配分になる。"""
    shares = [float(o.get("share", 1.0)) for o in opponents]
    s = sum(shares)
    if s <= 0:
        raise SystemExit("対戦相手の share の合計が 0")
    counts = [int(total * x / s) for x in shares]
    i = 0
    while sum(counts) < total:
        counts[i % len(counts)] += 1
        i += 1
    return counts


# ---------------------------------------------------------------- シャード入出力
def write_shard(path: Path, trajs: list[dict], meta: dict) -> Path:
    """parallel_collect の戻り値(軌跡のリスト)を平坦化して圧縮保存する。

    選択肢の数は決定点ごとに違うので、`opts` / `cids` は全決定点ぶんを縦に連結し、
    各決定点が何行使うかを `counts` に持たせる(パディングは学習側で行う)。
    """
    steps = [s for tr in trajs for s in tr["steps"]]
    if not steps:
        raise SystemExit("収集できた決定点が0件。重みのパスや試合数を確認する。")

    counts = np.array([len(s["option_feats"]) for s in steps], dtype=np.int32)
    arrays = {
        "state": np.array([s["state_feat"] for s in steps], dtype=np.float32),
        "opts": np.concatenate([np.asarray(s["option_feats"], dtype=np.float32) for s in steps]),
        "cids": np.concatenate([np.asarray(s["card_ids"], dtype=np.int32) for s in steps]),
        "counts": counts,
        "chosen": np.array([s["chosen_idx"] for s in steps], dtype=np.int32),
        "logp": np.array([s["logprob"] for s in steps], dtype=np.float32),
        "lengths": np.array([len(tr["steps"]) for tr in trajs], dtype=np.int32),
        "rewards": np.array([tr["reward"] for tr in trajs], dtype=np.float32),
        # 試合ごとの相手番号(run.json の opponents の並び順)。learner が advantage を
        # 相手ごとに正規化するのに使う。収集側が付けていなければ全部 0 = 1グループ扱い。
        "opp": np.array([int(tr.get("opp", 0)) for tr in trajs], dtype=np.int16),
    }
    meta = dict(meta)
    meta.update({
        "shard_format": SHARD_FORMAT,
        "n_steps": int(len(steps)),
        "n_trajectories": int(len(trajs)),
        "state_dim": int(arrays["state"].shape[1]),
        "option_dim": int(arrays["opts"].shape[1]),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.part")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    tmp.replace(path)   # 書き込み途中のファイルを learner に拾わせない
    return path


def read_shard(path: Path) -> tuple[dict, dict]:
    """(arrays, meta) を返す。"""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"].item()))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    return arrays, meta


class ShardRejected(Exception):
    """整合性検査に落ちたシャード。"""


def check_shard(meta: dict, run: dict, generation: int, model_sha: str) -> None:
    """『この経験を今回の PPO 更新に混ぜてよいか』の検査。

    PPO は収集時の方策と更新対象の方策が一致している前提の手法なので、1つでも違う世代の
    データが混ざると更新が壊れる。ここは黙って通さず、必ず例外にする。
    """
    if meta.get("shard_format") != SHARD_FORMAT:
        raise ShardRejected(f"シャード形式が違う({meta.get('shard_format')})")
    if meta.get("run_id") != run["run_id"]:
        raise ShardRejected(f"別の run のデータ(run_id={meta.get('run_id')})")
    if meta.get("generation") != generation:
        raise ShardRejected(f"世代が違う(v{meta.get('generation')} != v{generation})")
    if meta.get("model_sha256") != model_sha:
        raise ShardRejected("モデルの中身が違う(世代番号は同じだが重みが別物)")
    if meta.get("games_requested") != run["games_per_worker"]:
        raise ShardRejected(
            f"収集量が違う({meta.get('games_requested')} != {run['games_per_worker']})")
    if abs(float(meta.get("temperature", -1)) - float(run["temperature"])) > 1e-12:
        raise ShardRejected(f"温度が違う({meta.get('temperature')} != {run['temperature']})")
    if meta.get("opponents") != run["opponents"]:
        raise ShardRejected("対戦相手の設定が違う")


def set_collection_start_method(method: str = "spawn") -> str:
    """収集プロセスの起動方式を明示する。**Pool を作る前に、__main__ から呼ぶこと。**

    Linux の既定は fork で、親プロセスをコピーして子を作る。ところが親は
    run_league -> run_match 経由で cg エンジン(ネイティブライブラリ)を読み込み済みなので、
    子は初期化済みのエンジン状態を引き継ぐことになる。Windows の既定(spawn)ではこれが
    起きないため、開発環境では出ないのに Colab / Kaggle でだけ壊れる、という形になりうる。

    そこで既定を spawn に揃える。子はまっさらな状態から import し直すので確実。
    その代わり起動が数秒重くなるが、Pool は1回の収集につき1度しか作らないので影響は小さい。
    """
    import multiprocessing as mp
    if method == "default":
        return mp.get_start_method()
    mp.set_start_method(method, force=True)
    return method


def move_consumed(run_dir: Path, generation: int, paths: list[Path]) -> None:
    """使い終わったシャードを consumed/ へ退避する。同じ経験を2回 PPO に食わせないため。"""
    dest = consumed_dir(run_dir, generation)
    dest.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.move(str(p), str(dest / p.name))
