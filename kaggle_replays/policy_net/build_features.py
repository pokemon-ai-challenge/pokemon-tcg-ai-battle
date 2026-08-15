#!/usr/bin/env python3
"""policy_positions.jsonl.gz を読み、模倣ポリシー学習用の特徴量アレイ(features.npz)を作る。

各行(1意思決定点)について:
  - obs = to_observation_class({**row["observation"], "logs": []})
  - state_features = encoder.encode_state(obs)  (長さ BASE_FEATURE_COUNT)
  - option_features = encoder.encode_options(obs)  (長さ n_options のリスト。各要素は
    長さ OPTION_FEATURE_COUNT)
  - option_card_ids = encoder.encode_option_card_ids(obs.current, obs.select)  (長さ n_options の
    int リスト。card_embedding 用の生の card_id キー列。2026-07-20 追加、policy_model.py の
    card_embedding 対応)
  - split = split_for_episode(row["episode_id"])  (value_net/build_features.py と同じ md5 式)
  - weight = weight_for_rank(row["rank_at_fetch"])  (value_net と同じ rank_bucket テーブル)

2026-08-12(requirements-kamitsuorochi-2026-08-12.md「What to build」2): row["chosen_indices"]
(複数選択の場合は2件以上、単一選択なら1件)を選択肢と同じ n_options 長の bool 配列
``chosen_mask``(選ばれた位置が True)へ変換して持ち回す。option_features/option_card_ids と
同じ ragged 規約(dtype=object の配列、各要素が長さ n_options の numpy 配列)に揃える
(新しい規約を作らない)。既存の ``chosen_index``(単一 int のスカラー配列)は後方互換のため
そのまま残す(evaluate.py 等の既存読み手はこれを読む。複数選択行では選ばれた集合の先頭要素)。
row["chosen_indices"] が無い旧形式の jsonl(2026-08-12より前に作った他アーキタイプの
policy_positions_*.jsonl.gz)は [row["chosen_index"]] とみなして後方互換に処理する。

card_id_max は ``cg.api.all_card_data()`` から動的に計算し(ハードコードしない。デッキ・
カードデータが変わっても再学習だけで使い回せるという policy_model.py 側の設計要件)、
features.npz にスカラーとして保存する。card_embedding テーブルのサイズ(card_id_max + 1)を
train.py 側が決めるのに使う。

設計は固定(このファイル単体の都合で変更しないこと。step2-design.md §4 参照):
- サンプル重み: rank_bucket() の結果 -> {1-50: 1.5, 51-200: 1.3, 201-1000: 1.1, 1001+: 1.0, 不明: 1.0}
- split: h = md5(episode_id) % 100 -> h<80: train(0) / h<90: val(1) / else: test(2)

重要な制約: 行の順序は変えない(シャッフルしない)。features.npz の行 i は
policy_positions.jsonl.gz の i 行目(0-indexed、空行を除く)に対応する。evaluate.py が
後で rule_based_agent の再実行のために元の observation(同じ行番号)を突き合わせるのに
必要。したがって、value_net/build_features.py と異なり **encode 失敗行を黙ってスキップ
しない**(スキップすると row_index の対応関係が崩れる)。encode_state/encode_options は
仕様上「完成済みで正しく動作確認済み」の前提のため、失敗した場合は詳細をログした上で
例外を再送出しビルドを止める(fail-fast)。

使い方:
  python build_features.py                  # 全件処理
  python build_features.py --limit 2000      # 先頭2000行のみ(動作確認用)
  python build_features.py --in ... --out ... --limit ...

skill-concentration実験用フラグ(policymodel-skill-concentration-implementation-plan.md Step2。
既定値は上記の固定設計のまま = フラグを付けなければ既存の挙動と完全に同じ):
  --rank-max N        rank_at_fetch が N を超える行(不明=Noneも含む)を除外するハードフィルタ。
                       既定は None(フィルタなし)。
  --weight-scheme {default,concentrated}
                       default(既定)は上記の _WEIGHT_BY_RANK_BUCKET をそのまま使う。
                       concentrated は rank_at_fetch の生値に基づく専用スキーム
                       (weight_for_rank_concentrated、rank<=20:8.0 / <=50:3.0 / <=200:1.5 /
                       <=1000:1.0 / 1000超:0.5 / 不明:0.2)。既存の rank_bucket() より粒度が
                       細かく、上位への勾配集中を強める実験用(design.md §4 の固定表は変更しない
                       別関数として追加)。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"

# 既存の repo -> sample_submission import パターン(kaggle_replays/value_net/build_features.py 踏襲)。
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))
sys.path.insert(0, str(_HERE.parent / "deck_predictor"))

from cg.api import all_card_data, to_observation_class  # noqa: E402
from ptcg_ai.learning.encoder import (  # noqa: E402
    BASE_FEATURE_COUNT,
    BOARD_SLOTS,
    CONSEQUENCE_FEATURE_COUNT,
    HAND_CARD_SLOTS,
    OPP_CARD_SLOTS,
    OPTION_FEATURE_COUNT,
    POOL8_ARCHETYPES,
    encode_board_card_ids,
    encode_option_card_ids,
    encode_option_consequence_features,
    encode_options,
    encode_state,
)
from episode_window import rank_bucket  # noqa: E402

# Tier3 Stage3c: consequence特徴(--with-consequence-features 有効時のみ)。既定は付けない
# (フラグ無しなら features.npz に consequence_features キー自体が入らず、既存の
# build_features.py の出力と完全に同一。tier3-consequence-features-design-and-
# implementation-plan.md §6 Stage3c / §10 可逆性)。
_CONSEQUENCE_TIME_BUDGET_MS = 100.0

_DEFAULT_IN = _HERE.parent / "training_data" / "policy_positions.jsonl.gz"
_DEFAULT_OUT = _HERE / "features.npz"
# Phase1(案C, beyond-bc): 勝敗ラベルの取得元。value_positions は position 単位で label
# (win=1/loss=0)を持ち、(episode_id, player_index) では試合×プレイヤーで一意
# (実測: 不整合0)。policy_positions と同じ replay 由来なので episode_id/player_index で join できる。
_DEFAULT_OUTCOME_SOURCE = _HERE.parent / "training_data" / "value_positions.jsonl.gz"

# rank_bucket() の出力(日本語バケット名)->サンプル重み。value_net と同じ対応表(固定)。
_WEIGHT_BY_RANK_BUCKET: dict[str, float] = {
    "1-50": 1.5,
    "51-200": 1.3,
    "201-1000": 1.1,
    "1001+": 1.0,
    "不明": 1.0,
}

_PROGRESS_EVERY = 5_000


def split_for_episode(episode_id: str) -> int:
    """episode_id の md5 ハッシュから train(0)/val(1)/test(2) を決める(value_net と同じ固定式)。"""
    h = int(hashlib.md5(episode_id.encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return 0
    if h < 90:
        return 1
    return 2


def _read_deck_csv_ids(path: Path) -> list[int]:
    """1つのデッキCSV(改行区切り、read_deck_csv() と同じ形式)から card_id のリストを読む。"""
    with path.open(encoding="utf-8") as fh:
        return [int(v) for v in fh.read().split("\n") if v.strip()]


def build_hand_card_vocab(deck_csv_arg: str) -> list[int]:
    """``--deck-csv`` の値(ファイル or ディレクトリ)から手札 card_id 語彙(昇順)を作る。

    ファイルならそのファイルの card_id、ディレクトリなら配下の全 ``*.csv`` の card_id の
    和集合を使う(アーキタイプ内の deck variant 違いのテックカードを漏らさないため。
    build_features.py の --deck-csv ヘルプ参照)。
    """
    path = Path(deck_csv_arg)
    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        if not csv_paths:
            raise ValueError(f"--deck-csv にディレクトリを指定しましたが *.csv がありません: {path}")
        ids: set[int] = set()
        for csv_path in csv_paths:
            ids.update(_read_deck_csv_ids(csv_path))
        print(
            f"hand_card_vocab: ディレクトリ {path} 配下の {len(csv_paths)} 個のCSVの和集合から作成 "
            f"({[p.name for p in csv_paths]})",
            file=sys.stderr,
        )
    else:
        ids = set(_read_deck_csv_ids(path))
    return sorted(ids)


def build_opponent_card_vocab(archetype_decks_root: str) -> list[int]:
    """T1残り(design-transformer-representation-2026-08-08.md §9論点11): 相手の場・
    トラッシュ card_id カウント特徴に使う語彙(昇順)を、POOL8(encoder.POOL8_ARCHETYPES)
    全アーキタイプの代表デッキの和集合から作る。

    自分の手札語彙(build_hand_card_vocab)と異なり「使用中のデッキ」ではなく「相手として
    出現しうるメタ8アーキタイプ全部」から作る点が違う。``archetype_decks_root`` 配下に
    POOL8 の各アーキタイプ名のディレクトリがあることを前提とする
    (kaggle_replays/meta_analysis/archetype_decks/ の構造)。アーキタイプディレクトリが
    見つからない場合は警告のみで処理は続行する(見つかった分だけで語彙を作る)。
    """
    root = Path(archetype_decks_root)
    ids: set[int] = set()
    missing: list[str] = []
    for arch in POOL8_ARCHETYPES:
        arch_dir = root / arch
        if not arch_dir.is_dir():
            missing.append(arch)
            continue
        csv_paths = sorted(arch_dir.glob("*.csv"))
        for csv_path in csv_paths:
            ids.update(_read_deck_csv_ids(csv_path))
    if missing:
        print(
            f"[build_opponent_card_vocab] 警告: 見つからないアーキタイプディレクトリ: {missing}"
            f"(root={root})。見つかった分だけで語彙を作ります。",
            file=sys.stderr,
        )
    return sorted(ids)


def build_card_id_permutation(seed: int) -> dict[int, int]:
    """Negative Control(design-transformer-representation-2026-08-08.md §4原則9・§8):
    ゲーム全体の card_id 空間(``all_card_data()``)に対する固定のランダム全単射を返す。

    ``--shuffle-card-ids`` 指定時、盤面/選択肢の card_id(embedding 用、値そのものが
    識別子として使われる)をこの写像で置換してから学習する。0("識別なし/範囲外"の
    予約枠)は常に0に固定する(埋め込みテーブルの index 0 の意味を壊さないため)。

    本物の card_id 版と比べて性能が変わらなければ、改善が card_id という情報そのものの
    おかげではなく、次元が増えたこと自体の副作用である可能性を示唆する。
    """
    ids = sorted(c.cardId for c in all_card_data())
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    perm = dict(zip(ids, shuffled))
    perm[0] = 0
    return perm


def shuffle_card_id_list(ids: list[int], seed: int) -> list[int]:
    """Negative Control: カウント特徴の語彙(hand_card_vocab/opponent_card_vocab)の
    **並び順**をシャッフルする(値そのものは変えない)。

    語彙は「その語彙に含まれる card_id 全部が必ずどこかのスロットにカウントされる」
    という閉じた集合なので、``build_card_id_permutation``(ゲーム全体1,267種に対する
    全単射)をそのまま適用すると、語彙外の card_id にマップされてカウントが失われる
    (情報が単に消えるだけで対照群として無意味になる)。そのため語彙は要素の**順序**だけを
    シャッフルし、「どのスロットがどの card_id を指すか」の対応だけを壊す
    (カウントされる/されないという情報量は保つ)。
    """
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def weight_for_rank(rank_at_fetch: int | None) -> float:
    return _WEIGHT_BY_RANK_BUCKET[rank_bucket(rank_at_fetch)]


def weight_for_rank_concentrated(rank_at_fetch: int | None) -> float:
    """--weight-scheme concentrated 用。rank_bucket() より粒度の細かい専用テーブル
    (skill-concentration実験専用。既定の weight_for_rank は変更しない)。"""
    if rank_at_fetch is None:
        return 0.2
    if rank_at_fetch <= 20:
        return 8.0
    if rank_at_fetch <= 50:
        return 3.0
    if rank_at_fetch <= 200:
        return 1.5
    if rank_at_fetch <= 1000:
        return 1.0
    return 0.5


def load_outcome_map(path: Path) -> dict[tuple[str, int], int]:
    """value_positions.jsonl.gz から ``(episode_id, player_index) -> won(0/1)`` を作る。

    label は position 単位だが試合×プレイヤーで一定のはず。同一キーで label が食い違う場合は
    データ健全性の問題として ValueError を投げる(将来データで壊れたら気付けるように)。
    """
    outcome: dict[tuple[str, int], int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (str(r["episode_id"]), int(r["player_index"]))
            label = int(r["label"])
            prev = outcome.get(key)
            if prev is None:
                outcome[key] = label
            elif prev != label:
                raise ValueError(
                    f"outcome ラベル不整合: {key} に label {prev} と {label} が混在"
                )
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", default=str(_DEFAULT_IN))
    parser.add_argument("--out", dest="out_path", default=str(_DEFAULT_OUT))
    parser.add_argument(
        "--limit", type=int, default=None, help="先頭N行のみ処理する(動作確認モード)"
    )
    parser.add_argument(
        "--rank-max", type=int, default=None,
        help="rank_at_fetch がこの値を超える行(不明含む)を除外するハードフィルタ(既定: フィルタなし)",
    )
    parser.add_argument(
        "--weight-scheme", choices=["default", "concentrated"], default="default",
        help="サンプル重みスキーム(既定: default = 固定の rank_bucket テーブル)",
    )
    parser.add_argument(
        "--with-consequence-features", action="store_true",
        help="Tier3 Stage3c: 選択肢のconsequence特徴(仮実行、CONSEQUENCE_FEATURE_NAMES)を"
        "追加で計算し features.npz に consequence_features として保存する(既定OFF。"
        "付けない場合は既存の出力と完全に同一)。学習時もdummy隠れ状態のみを使う"
        "(tier3-consequence-features-design-and-implementation-plan.md §2.3のtrain/"
        "runtime parity要件。--with-consequence-features-deck で使うデッキを指定、"
        "既定は sample_submission/deck.csv)。仮実行を伴うため大幅に遅くなる。",
    )
    parser.add_argument(
        "--consequence-deck", default=None,
        help="--with-consequence-features 用の仮実行に使う自分のデッキCSV(既定: "
        "sample_submission/deck.csv、read_deck_csv() と同じ解決規則)。",
    )
    parser.add_argument(
        "--with-outcome", action="store_true",
        help="Phase1(案C, beyond-bc): 各行に won(勝=1/負=0/不明=-1)を追加して features.npz に "
        "保存する(既定OFF。付けない場合は won キー自体が入らず既存の出力と完全に同一)。"
        "勝敗は --outcome-source(value_positions)から (episode_id, player_index) で引く。",
    )
    parser.add_argument(
        "--outcome-source", default=str(_DEFAULT_OUTCOME_SOURCE),
        help="--with-outcome 用の勝敗ラベル源(既定: training_data/value_positions.jsonl.gz)。",
    )
    parser.add_argument(
        "--drop-unknown-outcome", action="store_true",
        help="--with-outcome 時、勝敗を引けない行(won=-1)を出力から除く(既定: 残す)。"
        "除くと row_index の元ファイル対応は崩れる(このnpzはoutcome学習専用で evaluate.py の"
        "行突き合わせには使わないため許容)。",
    )
    parser.add_argument(
        "--deck-csv", default=None,
        help="C2 第一段階(roadmap-2026-08-05.md): 自分の手札 card_id カウント特徴"
        "(encoder.encode_state の hand_card_vocab)の語彙をここから作る(既定: 指定なし = "
        "語彙なし、追加 HAND_CARD_SLOTS 次元は全て0で従来の出力と完全に同一)。"
        "ファイルまたはディレクトリを受け付ける: "
        "  - ファイル(改行区切り60行、read_deck_csv() と同じ形式。例: "
        "archetype_decks/<arch>/01.csv)を渡すと、そのファイルの card_id だけを使う。"
        "  - ディレクトリ(例: archetype_decks/<arch>/)を渡すと、配下の全 *.csv の card_id "
        "の**和集合**を使う。同一アーキタイプでも player ごとにテックカード違いの variant が"
        "あり、学習データ(policy_positions_*.jsonl.gz)には複数 variant のリプレイが"
        "混ざっているため、**アーキタイプ単位で学習するならディレクトリを渡すことを推奨する**"
        "(単一CSVだと他 variant だけが使うテックカードが語彙から漏れ、そのカードが手札に"
        "あるときだけカウントが handCount と食い違う)。"
        "学習データに出現した card_id からではなくデッキリストから語彙を作るのは、"
        "たまたま学習データに出現しなかったカードが欠けて推論時とズレるのを避けるため。"
        "異なる card_id を昇順に並べ、"
        f"{HAND_CARD_SLOTS}種を超える場合は警告して先頭{HAND_CARD_SLOTS}種のみ使う"
        "(encoder.HAND_CARD_SLOTS の固定長スロットに合わせる)。",
    )
    parser.add_argument(
        "--opponent-vocab-dir", default=None,
        help="T1残り(design-transformer-representation-2026-08-08.md §9論点11): 相手の場・"
        "トラッシュ card_id カウント特徴(encoder.encode_state の opponent_card_vocab)の語彙を、"
        "POOL8(encoder.POOL8_ARCHETYPES)全アーキタイプの代表デッキの和集合から作る。"
        "kaggle_replays/meta_analysis/archetype_decks のようなディレクトリを指定する想定"
        "(配下に POOL8 各アーキタイプ名のサブディレクトリがあること)。"
        "既定は指定なし = 語彙なし、追加 OPP_CARD_SLOTS*2 次元は全て0で従来の出力と完全に同一。"
        f"{OPP_CARD_SLOTS}種を超える場合は警告して先頭{OPP_CARD_SLOTS}種のみ使う。",
    )
    parser.add_argument(
        "--shuffle-card-ids", type=int, default=None,
        help="Negative Control(design-transformer-representation-2026-08-08.md §4原則9・§8): "
        "指定した seed で card_id の意味を壊した対照群を作る。手札/場/トラッシュのカウント"
        "特徴の語彙(hand_card_vocab/opponent_card_vocab)は要素の並び順を、盤面/選択肢の "
        "card_id(embedding 用、board_card_ids/option_card_ids)は値そのものをゲーム全体の "
        "card_id 空間に対する固定のランダム全単射で置換する。既定は None(シャッフルしない、"
        "既存の挙動と完全に同一)。本物の card_id 版と比べて性能が変わらなければ、改善が "
        "card_id という情報自体ではなく次元増加等の副作用による可能性を示唆する。",
    )
    args = parser.parse_args()
    weight_fn = weight_for_rank_concentrated if args.weight_scheme == "concentrated" else weight_for_rank

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    card_id_max = max(c.cardId for c in all_card_data())
    print(f"card_id_max = {card_id_max}(all_card_data() から動的に計算)", file=sys.stderr)

    # C2 第一段階: 手札 card_id カウント特徴の語彙(--deck-csv 指定時のみ)。
    hand_card_vocab: list[int] | None = None
    if args.deck_csv:
        hand_card_vocab = build_hand_card_vocab(args.deck_csv)
        if len(hand_card_vocab) > HAND_CARD_SLOTS:
            print(
                f"警告: --deck-csv の異なる card_id 数({len(hand_card_vocab)})が "
                f"HAND_CARD_SLOTS({HAND_CARD_SLOTS})を超えています。先頭 {HAND_CARD_SLOTS} 種"
                "のみを語彙として使用します(encoder.HAND_CARD_SLOTS 参照)。",
                file=sys.stderr,
            )
            hand_card_vocab = hand_card_vocab[:HAND_CARD_SLOTS]
        print(
            f"hand_card_vocab = {hand_card_vocab}(--deck-csv {args.deck_csv} から、"
            f"{len(hand_card_vocab)}種)",
            file=sys.stderr,
        )

    # T1残り: 相手の場・トラッシュ card_id カウント特徴の語彙(--opponent-vocab-dir 指定時のみ)。
    opponent_card_vocab: list[int] | None = None
    if args.opponent_vocab_dir:
        opponent_card_vocab = build_opponent_card_vocab(args.opponent_vocab_dir)
        if len(opponent_card_vocab) > OPP_CARD_SLOTS:
            print(
                f"警告: --opponent-vocab-dir の異なる card_id 数({len(opponent_card_vocab)})が "
                f"OPP_CARD_SLOTS({OPP_CARD_SLOTS})を超えています。先頭 {OPP_CARD_SLOTS} 種"
                "のみを語彙として使用します(encoder.OPP_CARD_SLOTS 参照)。",
                file=sys.stderr,
            )
            opponent_card_vocab = opponent_card_vocab[:OPP_CARD_SLOTS]
        print(
            f"opponent_card_vocab = {opponent_card_vocab}(--opponent-vocab-dir "
            f"{args.opponent_vocab_dir} から POOL8 の和集合、{len(opponent_card_vocab)}種)",
            file=sys.stderr,
        )

    # Negative Control(design-transformer-representation-2026-08-08.md §4原則9・§8):
    # --shuffle-card-ids 指定時、語彙は要素の並び順を、board/option の card_id は値そのものを
    # 固定のランダム全単射で置換する(理由は各関数のdocstring参照)。
    card_id_perm: dict[int, int] | None = None
    if args.shuffle_card_ids is not None:
        card_id_perm = build_card_id_permutation(args.shuffle_card_ids)
        if hand_card_vocab:
            hand_card_vocab = shuffle_card_id_list(hand_card_vocab, args.shuffle_card_ids)
            print(f"[shuffle-card-ids] hand_card_vocab の並び順をシャッフル: {hand_card_vocab}", file=sys.stderr)
        if opponent_card_vocab:
            # hand と同じ seed で shuffle するとゾーン間で同じ並べ替えパターンが出て
            # シャッフルの意味が薄まるため、seed をずらす(語彙が別空間なので混同はしない)。
            opponent_card_vocab = shuffle_card_id_list(opponent_card_vocab, args.shuffle_card_ids + 1)
            print(
                f"[shuffle-card-ids] opponent_card_vocab の並び順をシャッフル: {opponent_card_vocab}",
                file=sys.stderr,
            )
        print(
            f"[shuffle-card-ids] seed={args.shuffle_card_ids}: board_card_ids/option_card_ids は "
            "card_id 空間全体の固定ランダム置換で値そのものを変換します(Negative Control)。",
            file=sys.stderr,
        )

    outcome_map: dict[tuple[str, int], int] | None = None
    if args.with_outcome:
        outcome_source = Path(args.outcome_source)
        print(f"勝敗ラベルを読み込み(--with-outcome): {outcome_source}", file=sys.stderr)
        outcome_map = load_outcome_map(outcome_source)
        print(f"  (episode_id, player_index) 対 = {len(outcome_map)}(label不整合なし)", file=sys.stderr)

    consequence_deck: list[int] | None = None
    if args.with_consequence_features:
        import time as _time_mod

        from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
        from ptcg_ai.rule_based.rule_based_agent import read_deck_csv

        if args.consequence_deck:
            with open(args.consequence_deck, encoding="utf-8") as fh:
                consequence_deck = [int(v) for v in fh.read().split("\n") if v.strip()][:60]
        else:
            consequence_deck = read_deck_csv()
        print(
            f"consequence特徴を計算します(--with-consequence-features、deck枚数="
            f"{len(consequence_deck)}、1行あたり時間予算={_CONSEQUENCE_TIME_BUDGET_MS}ms)",
            file=sys.stderr,
        )

    state_feature_rows: list[list[float]] = []
    board_card_id_rows: list[list[int]] = []
    option_feature_rows: list[np.ndarray] = []
    option_card_id_rows: list[np.ndarray] = []
    consequence_feature_rows: list[np.ndarray] = []
    n_consequence_errors = 0
    chosen_index_rows: list[int] = []
    chosen_mask_rows: list[np.ndarray] = []  # 2026-08-12: 複数選択対応、ragged bool配列(長さn_options)
    n_multi_select_rows = 0
    split_rows: list[int] = []
    weight_rows: list[float] = []
    turn_rows: list[int] = []
    select_type_rows: list[int] = []
    select_context_rows: list[int] = []
    rank_rows: list[int] = []
    row_index_rows: list[int] = []
    won_rows: list[int] = []  # --with-outcome 時のみ使用(勝=1/負=0/不明=-1)
    n_outcome_unknown = 0

    n_total = 0  # フィルタ後の採用行数(= 出力行数)
    n_seen_lines = 0
    line_no = -1  # 空行を除いた0-indexed行番号(evaluate.pyの数え方と同じ。フィルタの影響を受けない)
    t0 = time.time()

    print(f"読み込み開始: {in_path}", file=sys.stderr)
    if args.limit is not None:
        print(f"動作確認モード: 先頭 {args.limit} 行のみ処理", file=sys.stderr)
    if args.rank_max is not None:
        print(f"rank フィルタ: rank_at_fetch<= {args.rank_max}(不明含め超過行は除外)", file=sys.stderr)
    if args.weight_scheme != "default":
        print(f"weight スキーム: {args.weight_scheme}", file=sys.stderr)

    with gzip.open(in_path, "rt", encoding="utf-8") as f:
        for line in f:
            n_seen_lines += 1
            line = line.strip()
            if not line:
                continue
            line_no += 1
            if args.limit is not None and n_total >= args.limit:
                break

            try:
                row = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"エラー: JSON パース失敗(生ファイル行 {n_seen_lines}): {exc!r}",
                    file=sys.stderr,
                )
                raise

            rank_at_fetch = row.get("rank_at_fetch")
            if args.rank_max is not None and (rank_at_fetch is None or rank_at_fetch > args.rank_max):
                continue  # フィルタで除外(line_no は既にインクリメント済みなので元ファイルの行番号との対応は崩れない)

            row_index = line_no  # 元ファイル(policy_positions.jsonl.gz)での0-indexed行番号

            try:
                obs_dict = {**row["observation"], "logs": []}
                obs = to_observation_class(obs_dict)
                state_feats = encode_state(
                    obs, hand_card_vocab=hand_card_vocab, opponent_card_vocab=opponent_card_vocab,
                )
                board_card_ids = encode_board_card_ids(obs.current)
                option_feats = encode_options(obs)
                option_card_ids = encode_option_card_ids(obs.current, obs.select)
                if card_id_perm is not None:
                    # Negative Control: embedding に使う値そのものを固定置換で変換する
                    # (0 = 識別なし/範囲外は perm[0] = 0 のまま変わらない)。
                    board_card_ids = [card_id_perm.get(cid, cid) for cid in board_card_ids]
                    option_card_ids = [card_id_perm.get(cid, cid) for cid in option_card_ids]
            except Exception as exc:  # noqa: BLE001 - fail-fast(row_index 対応関係を壊さない)
                print(
                    f"エラー: encode 失敗(row_index={row_index}, "
                    f"episode_id={row.get('episode_id')!r}, step_index={row.get('step_index')!r}): "
                    f"{exc!r}",
                    file=sys.stderr,
                )
                raise

            if len(state_feats) != BASE_FEATURE_COUNT:
                raise ValueError(
                    f"state特徴ベクトル長不正(row_index={row_index}): "
                    f"{len(state_feats)} != {BASE_FEATURE_COUNT}"
                )
            if len(board_card_ids) != BOARD_SLOTS:
                raise ValueError(
                    f"board_card_ids長不正(row_index={row_index}): "
                    f"{len(board_card_ids)} != {BOARD_SLOTS}"
                )
            n_options = row["n_options"]
            if len(option_feats) != n_options:
                raise ValueError(
                    f"選択肢数不一致(row_index={row_index}): "
                    f"encode_options={len(option_feats)} != row.n_options={n_options}"
                )
            for opt_vec in option_feats:
                if len(opt_vec) != OPTION_FEATURE_COUNT:
                    raise ValueError(
                        f"option特徴ベクトル長不正(row_index={row_index}): "
                        f"{len(opt_vec)} != {OPTION_FEATURE_COUNT}"
                    )
            if len(option_card_ids) != n_options:
                raise ValueError(
                    f"option_card_ids 長不正(row_index={row_index}): "
                    f"{len(option_card_ids)} != n_options={n_options}"
                )

            chosen_index = int(row["chosen_index"])
            if not (0 <= chosen_index < n_options):
                raise ValueError(
                    f"chosen_index が範囲外(row_index={row_index}): "
                    f"{chosen_index} not in [0, {n_options})"
                )

            # 2026-08-12: 複数選択対応。chosen_indices が無い旧形式の行は [chosen_index] とみなす。
            raw_chosen_indices = row.get("chosen_indices", [chosen_index])
            chosen_indices_this_row = [int(v) for v in raw_chosen_indices]
            if not chosen_indices_this_row:
                raise ValueError(f"chosen_indices が空(row_index={row_index})")
            if len(set(chosen_indices_this_row)) != len(chosen_indices_this_row):
                raise ValueError(f"chosen_indices に重複があります(row_index={row_index}): {chosen_indices_this_row}")
            for idx in chosen_indices_this_row:
                if not (0 <= idx < n_options):
                    raise ValueError(
                        f"chosen_indices の要素が範囲外(row_index={row_index}): "
                        f"{idx} not in [0, {n_options}) (chosen_indices={chosen_indices_this_row})"
                    )
            chosen_mask_row = np.zeros(n_options, dtype=bool)
            chosen_mask_row[chosen_indices_this_row] = True

            # Phase1(案C): 勝敗ラベル。行の player_index(= obs.current.yourIndex、決定者)で引く。
            # --drop-unknown-outcome 時は引けない行を早期スキップ(consequence 追加より前で行い、
            # 各 *_rows の対応を崩さない)。
            won = -1
            if outcome_map is not None:
                won = outcome_map.get((str(row["episode_id"]), int(row["player_index"])), -1)
                if won == -1:
                    if args.drop_unknown_outcome:
                        continue
                    n_outcome_unknown += 1  # 出力に残す不明行だけ数える

            if args.with_consequence_features:
                # 仮実行(cg.api.search_step)を伴うため、既存のfail-fast方針の例外として
                # fail-soft(失敗した行は0埋め、ビルド全体は止めない。方針書 §6 Stage3c)。
                deadline = _time_mod.perf_counter() + _CONSEQUENCE_TIME_BUDGET_MS / 1000
                factory = lambda: build_dummy_search_state(obs, consequence_deck)  # noqa: B023
                try:
                    consequence_feats = encode_option_consequence_features(obs, factory, deadline)
                    if len(consequence_feats) != n_options:
                        raise ValueError(
                            f"consequence特徴の選択肢数不一致: "
                            f"{len(consequence_feats)} != n_options={n_options}"
                        )
                except Exception as exc:  # noqa: BLE001
                    n_consequence_errors += 1
                    print(
                        f"警告: consequence特徴の計算に失敗(row_index={row_index}): {exc!r}"
                        f"(0埋めで続行)",
                        file=sys.stderr,
                    )
                    consequence_feats = [[0.0] * CONSEQUENCE_FEATURE_COUNT for _ in range(n_options)]
                consequence_feature_rows.append(np.asarray(consequence_feats, dtype=np.float32))

            episode_id = str(row["episode_id"])

            state_feature_rows.append(state_feats)
            board_card_id_rows.append(board_card_ids)
            option_feature_rows.append(np.asarray(option_feats, dtype=np.float32))
            option_card_id_rows.append(np.asarray(option_card_ids, dtype=np.int32))
            chosen_index_rows.append(chosen_index)
            chosen_mask_rows.append(chosen_mask_row)
            if len(chosen_indices_this_row) > 1:
                n_multi_select_rows += 1
            split_rows.append(split_for_episode(episode_id))
            weight_rows.append(weight_fn(rank_at_fetch))
            turn_rows.append(int(row.get("turn", -1)))
            select_type_rows.append(int(row["select_type"]))
            select_context_rows.append(int(row["select_context"]))
            rank_rows.append(rank_at_fetch if rank_at_fetch is not None else -1)
            row_index_rows.append(row_index)
            if outcome_map is not None:
                won_rows.append(won)

            n_total += 1
            if n_total % _PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed if elapsed > 0 else 0.0
                print(
                    f"  {n_total} 件処理済み(経過 {elapsed:.1f}s、{rate:.1f} rows/sec)",
                    file=sys.stderr,
                )

    elapsed = time.time() - t0
    print(f"完了: 採用 {n_total} 件(経過 {elapsed:.1f}s)", file=sys.stderr)

    state_features = np.asarray(state_feature_rows, dtype=np.float32)
    # T2(design-transformer-representation-2026-08-08.md §5.2): 盤面12スロットの card_id 列。
    # 選択肢と違って固定長(BOARD_SLOTS)なので ragged にならず、通常のint配列でよい。
    board_card_ids_arr = np.asarray(board_card_id_rows, dtype=np.int32)
    # ragged なので object 配列(dtype=object)にする。読み込み側は allow_pickle=True が必要。
    option_features = np.empty(n_total, dtype=object)
    for i, arr in enumerate(option_feature_rows):
        option_features[i] = arr
    option_card_ids_arr = np.empty(n_total, dtype=object)
    for i, arr in enumerate(option_card_id_rows):
        option_card_ids_arr[i] = arr
    chosen_index = np.asarray(chosen_index_rows, dtype=np.int32)
    # 2026-08-12: chosen_mask も option_features/option_card_ids と同じ ragged 規約(dtype=object、
    # 各要素が長さ n_options の numpy 配列)。train.py の多正解listwise損失はこれを直接使う。
    chosen_mask_arr = np.empty(n_total, dtype=object)
    for i, arr in enumerate(chosen_mask_rows):
        chosen_mask_arr[i] = arr
    split = np.asarray(split_rows, dtype=np.int8)
    weight = np.asarray(weight_rows, dtype=np.float32)
    turn = np.asarray(turn_rows, dtype=np.int32)
    select_type = np.asarray(select_type_rows, dtype=np.int32)
    select_context = np.asarray(select_context_rows, dtype=np.int32)
    rank_at_fetch_arr = np.asarray(rank_rows, dtype=np.int32)  # -1 = 不明(rank_at_fetch is None)
    row_index = np.asarray(row_index_rows, dtype=np.int32)  # 元ファイルでの0-indexed行番号(rank-maxフィルタ時はarangeと異なる)

    print(f"state_features shape={state_features.shape} dtype={state_features.dtype}", file=sys.stderr)
    print(f"board_card_ids shape={board_card_ids_arr.shape} dtype={board_card_ids_arr.dtype}", file=sys.stderr)
    print(
        f"split 内訳: train={int((split == 0).sum())} "
        f"val={int((split == 1).sum())} test={int((split == 2).sum())}",
        file=sys.stderr,
    )
    print(
        f"複数選択(chosen_indices>=2件)行: {n_multi_select_rows}/{n_total} "
        f"({n_multi_select_rows / max(n_total, 1) * 100:.2f}%)",
        file=sys.stderr,
    )

    extra_arrays = {}
    if outcome_map is not None:
        extra_arrays["won"] = np.asarray(won_rows, dtype=np.int8)
        n_known = n_total - n_outcome_unknown
        print(
            f"outcome: {n_total}行中 勝敗既知={n_known} ({n_known / max(n_total,1):.3f}) "
            f"不明(won=-1)={n_outcome_unknown}",
            file=sys.stderr,
        )
    if args.with_consequence_features:
        consequence_features_arr = np.empty(n_total, dtype=object)
        for i, arr in enumerate(consequence_feature_rows):
            consequence_features_arr[i] = arr
        extra_arrays["consequence_features"] = consequence_features_arr
        extra_arrays["consequence_feature_count"] = np.array(CONSEQUENCE_FEATURE_COUNT)
        print(
            f"consequence特徴: {n_total}行中 {n_consequence_errors}行で計算失敗(0埋め)",
            file=sys.stderr,
        )

    # hand_card_vocab は train.py が重みJSONの meta.hand_card_vocab へそのまま書き出す
    # (C2 第一段階)。--deck-csv 未指定時は空配列(train.py 側は「語彙なし」として扱う)。
    extra_arrays["hand_card_vocab"] = np.asarray(hand_card_vocab or [], dtype=np.int32)
    # opponent_card_vocab も同様(T1残り)。--opponent-vocab-dir 未指定時は空配列。
    extra_arrays["opponent_card_vocab"] = np.asarray(opponent_card_vocab or [], dtype=np.int32)

    np.savez_compressed(
        out_path,
        state_features=state_features,
        board_card_ids=board_card_ids_arr,
        option_features=option_features,
        option_card_ids=option_card_ids_arr,
        card_id_max=np.array(card_id_max),
        chosen_index=chosen_index,
        chosen_mask=chosen_mask_arr,
        split=split,
        weight=weight,
        turn=turn,
        select_type=select_type,
        select_context=select_context,
        row_index=row_index,
        rank_at_fetch=rank_at_fetch_arr,
        **extra_arrays,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"書き出し完了: {out_path} ({size_mb:.1f} MB)", file=sys.stderr)
    print(
        "注意: option_features/option_card_ids/chosen_mask は object 配列(ragged、各行 "
        "shape=(n_options,) または (n_options, OPTION_FEATURE_COUNT))。読み込みは "
        "np.load(path, allow_pickle=True) を使うこと。card_id_max はスカラー"
        "(data['card_id_max'].item() で取り出す)。chosen_index(単一int)は複数選択行では "
        "選ばれた集合の先頭要素、選ばれた全集合は chosen_mask(bool、True=選択された)を使うこと。",
        file=sys.stderr,
    )
    if args.with_consequence_features:
        print(
            "consequence_features も object 配列(ragged、各行 shape=(n_options, "
            f"{CONSEQUENCE_FEATURE_COUNT}))。特徴の並びは "
            "ptcg_ai.learning.encoder.CONSEQUENCE_FEATURE_NAMES と同じ。",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
