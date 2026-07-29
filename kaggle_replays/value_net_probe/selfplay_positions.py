"""自己対戦(学習側 複数アーキタイプ vs 相手プール4種)で局面データセットを収集し npz に落とす。

目的: 「序盤(ターン1-5)で局面から勝敗が予測できないのは、(a)特徴量に情報が無いからか、
(c)本質的に予測不能だからか」を、試合数を自由に増やせる自己対戦データで決着させるため。
上位リプレイは300試合しか無く、決着しなかった(kaggle_replays/value_net_probe/early_game_signal.py
参照)。強化学習の critic は本来 on-policy であるべきなので、自己対戦データの方が RL 用
critic としても正しい分布になる、という副次的な狙いもある。

学習側を単一アーキタイプ(dragapult_ex)に固定していたところ、
kaggle_replays/value_net_probe/calibrate_learner.py で較正したところ相手プールに対し
勝率 0.167(温度0.05) / 0.062(温度1.0) しか無く、勝敗がデッキ相性でほぼ決まってしまい
「序盤の局面が勝敗を予測するか」という問いが相性識別に汚染される問題があった。そこで
較正で50%近辺だった複数の学習側(LEARNER_REGISTRY)を --learners で切り替えられるようにし、
デッキの多様性を確保しつつ全体の勝率を50%付近に近づける。

相手プールの構成は kaggle_replays/rl/test_collect_pool.py と同一:
  - 相手4種: alakazam(weights=None, production既定) / crustle / marnie_grimmsnarl_ex /
    archaludon_ex、各デッキは archetype_decks/<name>/01.csv

重要な設計上の注意:
  - 収集は kaggle_replays/rl/collect_pool.py の parallel_collect_pool を使う。
    このモジュール自体・cg/・data/ は一切変更しない。
  - 各 trajectory の step には option_feats(選択肢数 x 約100次元)が入っており、
    state_feat の何倍もサイズが大きい。数千試合分を一度にメモリ上に持つと圧迫するため、
    --batch-games 単位で「収集 -> 必要な列だけ即座に抜き出す -> 軌跡本体(trajs)は破棄」を
    繰り返す。軌跡全体を蓄積することはしない。この抜き出しは学習側ごと・バッチごとに行う。
  - 記録されるのは学習側(learner)の decision のみ(collect_parallel._play_one の仕様。
    「相手も同じ pure-Python 方策」のコメントはあるが、記録(steps.append)は
    cur.yourIndex == learner_index の分岐でのみ行われる)。したがって同一局面が両視点で
    重複して入る、といった漏洩は起きない。ただし同一試合内の複数 step は明らかに相関する
    (同じゲームの経過)ので、学習/評価分割は必ず game_id 単位で行うこと(行単位の
    分割はリークになる)。game_id は学習側をまたいでも一意(連番)。

seed 設計(バッチ間・学習側間の衝突回避):
  collect_pool.build_tasks は相手 i (0-indexed, k=len(opponents)) に対して
  seed = seed0 + i * SEED_STRIDE + g (g は 0..per-1) を割り当てる。
  相手は4種(k=4)なので、1バッチが使う seed 空間は
  [seed0, seed0 + (k-1)*SEED_STRIDE + per) にほぼ収まり、常に
  seed0 + k*SEED_STRIDE より小さい(per < SEED_STRIDE である限り)。
  したがってバッチ b (0-indexed) の seed0 を
      batch_seed0 = learner_seed0 + b * k * SEED_STRIDE
  とすれば、バッチ b の使用域 [batch_seed0, batch_seed0 + k*SEED_STRIDE) は
  バッチ b+1 の使用域と重ならず、同一学習側内の全バッチ・全相手を通じて seed が衝突しない。

  学習側をまたぐ場合はさらに、学習側 j (0-indexed, --learners で指定した順) に
      learner_seed0 = args.seed0 + j * learner_seed_slot
      learner_seed_slot = n_batches * k * SEED_STRIDE
  という大きなオフセットを与える(main() 内)。--games は --learners の人数で等分するため
  games_per_learner、したがって n_batches は全学習側で共通の値になる。ゆえに学習側 j の
  seed 空間 [learner_seed0, learner_seed0 + learner_seed_slot) は、上記のバッチ間衝突回避の
  議論よりその内部で(バッチ間・相手間とも)衝突せず、かつ学習側 j+1 の開始位置
  learner_seed0 + learner_seed_slot ちょうどで終わるので学習側間でも重ならない。

  なお parallel_collect_pool / _play_one_pool は trajectory の戻り値に seed を含めない
  (collect_parallel._play_one が返す dict は {"steps","reward","winner","error"} のみで、
  _play_one_pool が追加するのも "opponent"/"opp_idx"/"learner_index" だけ)。collect_pool.py
  は変更禁止のため、npz に seed 列を実測値として保存することはできない。代わりに上記の
  オフセット設計により衝突しないことを算術的に保証する。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_RL_DIR = _ROOT / "kaggle_replays" / "rl"
for _p in (str(_RL_DIR), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

# 学習側レジストリ: 表示名 -> (重みファイル名 or None(=production alakazam), デッキのアーキタイプ名)。
# calibrate_learner.py の CANDIDATES と同じ形式・同じ相手プールで実測した勝率(各48試合):
#   表示名                    温度0.05  温度1.0
#   alakazam(production)      0.521     0.271
#   marnie_grimmsnarl_ex      0.479     0.354
#   archaludon_ex             0.479     0.271
#   mega_lucario_ex           0.521     0.375
#   crustle                   0.604     0.667
#   dragapult_ex              0.167     0.062   (デッキ相性に極端に汚染されるため既定の
#                                                 --learners には含めない。指定は可能)
LEARNER_REGISTRY = {
    "alakazam": (None, "alakazam"),
    "marnie_grimmsnarl_ex": ("policy_weights_marnie_grimmsnarl_ex.json", "marnie_grimmsnarl_ex"),
    "archaludon_ex": ("policy_weights_archaludon_ex.json", "archaludon_ex"),
    "mega_lucario_ex": ("policy_weights_mega_lucario_ex.json", "mega_lucario_ex"),
    "crustle": ("policy_weights_crustle.json", "crustle"),
    "dragapult_ex": ("policy_weights_dragapult_ex.json", "dragapult_ex"),
}

DEFAULT_LEARNERS = "alakazam,marnie_grimmsnarl_ex,archaludon_ex,mega_lucario_ex"

OPPONENT_SPECS = [
    ("alakazam", None, str(DECKDIR / "alakazam" / "01.csv")),
    ("crustle", str(WDIR / "policy_weights_crustle.json"), str(DECKDIR / "crustle" / "01.csv")),
    ("marnie_grimmsnarl_ex", str(WDIR / "policy_weights_marnie_grimmsnarl_ex.json"),
     str(DECKDIR / "marnie_grimmsnarl_ex" / "01.csv")),
    ("archaludon_ex", str(WDIR / "policy_weights_archaludon_ex.json"),
     str(DECKDIR / "archaludon_ex" / "01.csv")),
]

DEFAULT_OUT = _HERE / "selfplay_positions.npz"


def build_opponents():
    """(name, weights_path_or_None, deck_o) のリストを構築する(遅延 import で cg 依存を隔離)。"""
    from run_league import read_deck_csv_file

    return [(name, wpath, read_deck_csv_file(deck_csv)) for name, wpath, deck_csv in OPPONENT_SPECS]


def resolve_learner(name: str):
    """学習側名 -> (weights_path_or_None, deck_csv_path)。レジストリに無ければ ValueError。"""
    if name not in LEARNER_REGISTRY:
        available = ", ".join(sorted(LEARNER_REGISTRY))
        raise ValueError(f"未知の学習側名: {name!r}. 利用可能な名前: {available}")
    weights_file, arch = LEARNER_REGISTRY[name]
    weights_path = str(WDIR / weights_file) if weights_file else None
    deck_csv = str(DECKDIR / arch / "01.csv")
    return weights_path, deck_csv


def extract_batch_arrays(trajs, game_id_start, turn_feat_idx, learner_name):
    """1バッチ分の trajectories から npz 用の列を抜き出す(軌跡本体はここで使い切って捨てる)。

    戻り値: dict of numpy 配列(このバッチ分のみ)。game_id はバッチ・学習側をまたいで一意になるよう
    game_id_start から連番を振る(呼び出し側が学習側をまたいでも連続した game_id_start を渡すこと)。
    """
    X_list = []
    y_list = []
    turn_list = []
    game_id_list = []
    opponent_list = []
    learner_index_list = []
    n_options_list = []
    learner_list = []

    gid = game_id_start
    for traj in trajs:
        won = 1 if traj["reward"] >= 1.0 else 0
        opponent = traj["opponent"]
        learner_index = traj["learner_index"]
        for step in traj["steps"]:
            sf = step["state_feat"]
            X_list.append(sf)
            y_list.append(won)
            turn_list.append(int(sf[turn_feat_idx]))
            game_id_list.append(gid)
            opponent_list.append(opponent)
            learner_index_list.append(learner_index)
            n_options_list.append(len(step["option_feats"]))
            learner_list.append(learner_name)
        gid += 1

    n_games_in_batch = gid - game_id_start

    return {
        "X": np.asarray(X_list, dtype=np.float32),
        "y": np.asarray(y_list, dtype=np.int8),
        "turn": np.asarray(turn_list, dtype=np.int16),
        "game_id": np.asarray(game_id_list, dtype=np.int32),
        "opponent": np.asarray(opponent_list, dtype="<U24"),
        "learner_index": np.asarray(learner_index_list, dtype=np.int8),
        "n_options": np.asarray(n_options_list, dtype=np.int16),
        "learner": np.asarray(learner_list, dtype="<U24"),
    }, n_games_in_batch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--games", type=int, required=True,
        help="収集する総試合数。--learners の人数で等分する(端数切り捨て、捨てた数は報告する)。"
             "parallel_collect_pool は相手ごとにさらに等分し per を偶数に丸めるので、"
             "無駄なく使うには --games を「学習側の数 x 相手4種 x 2(先攻/後攻)」の倍数にすること。",
    )
    ap.add_argument(
        "--learners", type=str, default=DEFAULT_LEARNERS,
        help=f"カンマ区切りの学習側名(LEARNER_REGISTRY のキー)。既定は較正で50%%近辺だった4種: "
             f"{DEFAULT_LEARNERS}",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed0", type=int, default=11000)
    ap.add_argument("--batch-games", type=int, default=400)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if args.games <= 0:
        raise ValueError(f"--games は正の整数でなければならない: {args.games}")
    if args.batch_games <= 0:
        raise ValueError(f"--batch-games は正の整数でなければならない: {args.batch_games}")

    learner_names = [s.strip() for s in args.learners.split(",") if s.strip()]
    if not learner_names:
        raise ValueError(f"--learners が空: {args.learners!r}")
    for name in learner_names:
        if name not in LEARNER_REGISTRY:
            available = ", ".join(sorted(LEARNER_REGISTRY))
            raise ValueError(f"未知の学習側名: {name!r}. 利用可能な名前: {available}")

    from collect_pool import SEED_STRIDE, parallel_collect_pool
    from ptcg_ai.learning import encoder
    from run_league import read_deck_csv_file

    turn_feat_idx = encoder.FEATURE_NAMES.index("turn")

    opponents = build_opponents()
    k = len(opponents)
    n_learners = len(learner_names)

    games_per_learner = args.games // n_learners
    dropped_games_learner_split = args.games - games_per_learner * n_learners
    if games_per_learner <= 0:
        raise ValueError(
            f"--games({args.games}) を --learners の人数({n_learners})で割ると0になる。"
            "--games を増やすか --learners を減らすこと。"
        )

    # 学習側ごとのバッチ数(games_per_learner は全学習側で同一なので n_batches も全学習側で同一)。
    n_batches = (games_per_learner + args.batch_games - 1) // args.batch_games

    # 学習側間の seed 衝突回避:
    # 学習側1つが使う seed 空間は、バッチ間衝突回避と同じ理屈で
    # [args.seed0, args.seed0 + n_batches*k*SEED_STRIDE) に収まる(docstring 参照)。
    # したがって学習側 j (0-indexed) に
    #     learner_seed0 = args.seed0 + j * n_batches * k * SEED_STRIDE
    # を与えれば、学習側 j の空間 [learner_seed0, learner_seed0 + n_batches*k*SEED_STRIDE) は
    # 学習側 j+1 の開始位置 learner_seed0 (j+1) ちょうどまでで終わり、重ならない。
    learner_seed_slot = n_batches * k * SEED_STRIDE

    X_parts, y_parts, turn_parts, gid_parts, opp_parts, li_parts, nopt_parts, learner_parts = (
        [], [], [], [], [], [], [], []
    )

    total_games = 0
    total_steps = 0
    next_game_id = 0
    t_start = time.time()

    for j, learner_name in enumerate(learner_names):
        weights_path, deck_csv = resolve_learner(learner_name)
        deck_l = read_deck_csv_file(deck_csv)
        learner_seed0 = args.seed0 + j * learner_seed_slot

        learner_games = 0
        learner_steps = 0
        games_remaining = games_per_learner
        for b in range(n_batches):
            batch_games = min(args.batch_games, games_remaining)
            games_remaining -= batch_games
            batch_seed0 = learner_seed0 + b * k * SEED_STRIDE

            trajs, stats = parallel_collect_pool(
                weights_path, opponents, deck_l,
                n_games=batch_games, seed0=batch_seed0,
                temperature=args.temperature, workers=args.workers,
            )

            arrays, n_games_in_batch = extract_batch_arrays(
                trajs, next_game_id, turn_feat_idx, learner_name
            )
            next_game_id += n_games_in_batch
            total_games += n_games_in_batch
            total_steps += len(arrays["y"])
            learner_games += n_games_in_batch
            learner_steps += len(arrays["y"])

            X_parts.append(arrays["X"])
            y_parts.append(arrays["y"])
            turn_parts.append(arrays["turn"])
            gid_parts.append(arrays["game_id"])
            opp_parts.append(arrays["opponent"])
            li_parts.append(arrays["learner_index"])
            nopt_parts.append(arrays["n_options"])
            learner_parts.append(arrays["learner"])

            # 軌跡本体(option_feats等を含む重いデータ)はここで参照を切って GC 対象にする。
            del trajs

            elapsed = time.time() - t_start
            print(
                f"[learner {j + 1}/{n_learners}={learner_name} batch {b + 1}/{n_batches}] "
                f"batch_games={batch_games} errors={stats['total']['errors']} "
                f"累計(learner内) n_games={learner_games} n_steps={learner_steps} "
                f"累計(全体) n_games={total_games} n_steps={total_steps} 経過={elapsed:.1f}s",
                flush=True,
            )

    X = np.concatenate(X_parts, axis=0) if X_parts else np.zeros((0, len(encoder.FEATURE_NAMES)), dtype=np.float32)
    y = np.concatenate(y_parts, axis=0) if y_parts else np.zeros((0,), dtype=np.int8)
    turn = np.concatenate(turn_parts, axis=0) if turn_parts else np.zeros((0,), dtype=np.int16)
    game_id = np.concatenate(gid_parts, axis=0) if gid_parts else np.zeros((0,), dtype=np.int32)
    opponent = np.concatenate(opp_parts, axis=0) if opp_parts else np.zeros((0,), dtype="<U24")
    learner_index = np.concatenate(li_parts, axis=0) if li_parts else np.zeros((0,), dtype=np.int8)
    n_options = np.concatenate(nopt_parts, axis=0) if nopt_parts else np.zeros((0,), dtype=np.int16)
    learner = np.concatenate(learner_parts, axis=0) if learner_parts else np.zeros((0,), dtype="<U24")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X, y=y, turn=turn, game_id=game_id,
        opponent=opponent, learner_index=learner_index, n_options=n_options,
        learner=learner,
    )

    total_elapsed = time.time() - t_start
    print(f"\n--games({args.games}) を学習側{n_learners}種で等分: "
          f"games_per_learner={games_per_learner} 切り捨て={dropped_games_learner_split}")
    print("学習側ごとの内訳(n_games, n_steps, win_rate=yの平均):")
    for name in learner_names:
        m = learner == name
        n_games_l = int(len(np.unique(game_id[m]))) if m.any() else 0
        n_steps_l = int(m.sum())
        wr = float(y[m].mean()) if m.any() else float("nan")
        print(f"  {name:24s} n_games={n_games_l:5d} n_steps={n_steps_l:7d} win_rate={wr:.3f}")
    overall_win_rate = float(np.mean(y)) if len(y) else float("nan")
    print(f"\n完了: total_games={total_games} total_steps={total_steps} "
          f"overall_y_mean={overall_win_rate:.4f} "
          f"elapsed={total_elapsed:.1f}s out={out_path}")


if __name__ == "__main__":
    main()
