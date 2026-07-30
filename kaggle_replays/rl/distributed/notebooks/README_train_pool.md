# Kaggle Notebook で train_pool.py を回す

ローカルは8コアで20イテレーション51〜65分。Kaggle は**4コア**なので概ね2倍かかるが、
PC を占有せず、複数を並行できる。60イテレーションで5〜7時間、12時間のセッション上限には収まる。

## 前提

- `kaggle` CLI が入っていること
- 認証が済んでいること（`~/.kaggle/credentials.json` か `kaggle.json`）。**このファイルは読まない**。
  ユーザー名は `kaggle config view` から取る（OAuth ログインだと `kaggle.json` は作られない）

## 手順

```bash
# 1. まず送らずに中身を確認する
python kaggle_replays/rl/distributed/notebooks/push_train_pool.py \
    --train-args "--learner alakazam --train-opponents alakazam --iters 60 --tag k60" \
    --dry-run

# 2. 送る（非公開 Dataset + 非公開 Notebook）
python kaggle_replays/rl/distributed/notebooks/push_train_pool.py \
    --train-args "--learner alakazam --train-opponents alakazam --eval-fixed alakazam --iters 60 --eval-every 5 --tag k60" \
    --kernel-slug ptcg-train-alakazam-k60 \
    --dataset-wait 600 --no-wait

# 3. 状態を見る
kaggle kernels status koshin953/ptcg-train-alakazam-k60

# 4. 終わったら回収する（重み JSON と学習ログが /kaggle/working に入っている）
kaggle kernels output koshin953/ptcg-train-alakazam-k60 -p <保存先>
```

`--no-wait` を付けないと完走まで待つ。長い run は投げっぱなしにして後から回収する方がよい。

## バンドルについて

`git archive HEAD` は使わない。**作業ツリーから明示的なリストで zip を作る**（未コミットのファイルを
含める必要があるため）。含めるもの: `sample_submission/cg/` `sample_submission/ptcg_ai/` `league/`
`kaggle_replays/rl/` `kaggle_replays/meta_analysis/archetype_decks/`。約20MB → zip 8MB。

重みは `--weights` で絞れる（既定は全部）。

**クリーンルーム検証を省かないこと。** zip をリポジトリ外に展開し、そこから `train_pool.py` を
極小設定で実行する。過去にこの検証で「デッキが入っていない」「`__main__` ガード欠落で
BrokenProcessPool」「glob が特定の重みを除外」の3件を実際に見つけている。

## 踏んだ落とし穴

### 1. Kaggle は Dataset の zip を既定で展開する

`ptcg_bundle.zip` をアップロードしても、`/kaggle/input/<slug>/` には zip ではなく
**展開されたディレクトリ木**が置かれる。zip 前提でノートブックを書いて失敗した。

対策: ノートブックは**両方の形に対応**する。zip があれば展開、無ければ `libcg.so` を目印に
ツリーを探して `/kaggle/working/repo/` へ複製する（`/kaggle/input` は読み取り専用で、
`train_pool.py` は重みを書き出すため writable な場所が要る）。複製後に `libcg.so` へ実行権限を付ける。

### 2. Dataset の処理完了を待たずに Notebook を push すると /kaggle/input が空になる

固定秒数の sleep（90秒）で済ませたら、`/kaggle/input/` が空のまま実行され、
原因の分かりにくい失敗になった。`kaggle datasets status <slug>` が `ready` を返すまで待つこと。

**注意**: `datasets status` は Dataset 全体の状態を返すため、**新バージョンの処理中でも既存版が
ready なら ready を返す可能性がある**（実際、版を更新した直後に 2秒で ready が返った）。
上記1の「両方の形に対応」があるので実害は出にくいが、この待ち合わせは完全ではない。

### 3. Notebook のスラッグは title から決まる

Kaggle は Notebook を**新規作成**するとき、`kernel-metadata.json` の `id` ではなく
**title を slug 化**して ref を決める。title を固定にすると `--kernel-slug` を変えても
同じ Notebook を上書きしてしまう。`--kernel-slug ptcg-train-alakazam-k60` で送ったのに
ref が `ptcg-train-pool-run` になった。現在は title を `--kernel-slug` から生成している。

### 4. Windows の文字コード

- `kaggle` CLI の出力を読むときは `encoding="utf-8"` を明示する（Windows の既定は cp932）
- Notebook JSON は `ensure_ascii=True` で書く。Kaggle CLI は push するファイルを
  システム既定文字コードで読むため、日本語が生で入っていると失敗する
- `kaggle kernels output` が落とすログ JSON は UTF-8 で読めないバイトを含むことがある。
  `open(..., 'rb').read().decode('utf-8', errors='replace')` で読む

## 設定

Dataset・Notebook とも**非公開**（`isPrivate` / `is_private`）。GPU 無し、インターネット無し。
