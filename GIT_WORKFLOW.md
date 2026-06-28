# Git / GitHub Workflow

このリポジトリ向けの、最小限で安全な Git / GitHub 手順メモです。

このリポジトリでは通常、`master` は直接作業しません。作業は必ず別ブランチで進めます。

---

## 基本ルール

- `master` で直接編集しない
- 新しい作業は `feature/...` や `experiment/...` ブランチで行う
- 共有ブランチの最新化は `git fetch` + `git merge --ff-only` を優先する
- 取り込み前に `git status` で未コミット変更を確認する
- 未コミット変更がある状態で無理に `git switch` しない

---

## まず確認するコマンド

```powershell
git status --short --branch
git branch --show-current
git branch -a -vv
```

見方:

- `git status --short --branch` で現在ブランチと作業ツリーの汚れを確認します
- `git branch --show-current` で今いるブランチを確認します
- `git branch -a -vv` でローカルとリモートの対応関係を確認します

---

## `master` ベースで新しい作業ブランチを作る

`master` を親にしたいときは、先にリモートの最新を取り込んでからブランチを切ります。

```powershell
git switch master
git fetch origin
git merge --ff-only origin/master
git switch -c feature/branch-name
```

例:

```powershell
git switch -c feature/improve-agent-logic
```

補足:

- `git pull origin master` でも更新できますが、このリポジトリでは `fetch` と `merge` を分けた方が確認しやすいです
- `origin/master` を明示すると、古いローカル `master` をうっかり基準にしにくくなります

---

## 共有ブランチを親にして新しい作業ブランチを作る

チームで `develop/nagata` のような共有ブランチを使う場合は、そのブランチを最新化してから子ブランチを切ります。

```powershell
git switch develop/nagata
git fetch origin
git merge --ff-only origin/develop/nagata
git switch -c feature/nagata-improve-search
```

補足:

- `develop/nagata` は共有の親ブランチ例です
- この場合の feature ブランチは `master` ではなく `develop/nagata` から派生します
- Pull Request の取り込み先も通常は `master` ではなく `develop/nagata` です

---

## 共有ブランチの更新を自分の feature ブランチへ取り込む

共有ブランチが進んだら、まず親ブランチを更新してから、自分の feature ブランチへ merge します。

```powershell
git switch develop/nagata
git fetch origin
git merge --ff-only origin/develop/nagata
git switch feature/nagata-improve-search
git merge develop/nagata
```

確認用:

```powershell
git branch --contains origin/develop/nagata
git rev-list --left-right --count HEAD...origin/develop/nagata
```

見方:

- `git branch --contains origin/develop/nagata` で現在ブランチが共有ブランチ先端を含むか確認できます
- `git rev-list --left-right --count HEAD...origin/develop/nagata` の右側が `0` なら未取り込みはありません

---

## 未コミット変更があるとき

`git switch` や `git merge` の前に、まず現在の変更を安全に退避します。

```powershell
git status --short --branch
git stash push --include-untracked -m "wip before sync"
```

そのあと同期作業をして、最後に戻します。

```powershell
git stash pop
```

補足:

- 競合しそうな大きな作業中は、`stash` ではなく別 worktree を使う方が安全です
- 取り込み対象が多いときほど、今の作業ツリーを汚したまま branch switch しない方が安心です

---

## 変更をコミットする

```powershell
git status
git add .
git commit -m "feat: improve main action selection"
```

コミットメッセージ例:

- `feat: improve main action selection`
- `fix: correct deck export behavior`
- `docs: update git workflow guide`

---

## GitHub に push する

最初の push:

```powershell
git push -u origin feature/branch-name
```

2 回目以降:

```powershell
git push
```

補足:

- `-u` を付けると次回から `git push` だけで済みます
- 追跡先がずれているときは `git branch -vv` で upstream を確認します

---

## Pull Request の考え方

- `master` ベースの作業なら PR の取り込み先は `master`
- `develop/nagata` ベースの作業なら PR の取り込み先は `develop/nagata`
- どの親ブランチから切ったかと、どこへ戻すかを揃えます

例:

- 取り込み先: `develop/nagata`
- 作業ブランチ: `feature/nagata-improve-search`

```powershell
git switch feature/nagata-improve-search
git push -u origin feature/nagata-improve-search
```

---

## 困ったときの確認コマンド

```powershell
git status --short --branch
git branch --show-current
git branch -a -vv
git log --oneline --decorate --graph --max-count 10
```

追加で便利な確認:

```powershell
git rev-list --left-right --count HEAD...origin/master
git rev-list --left-right --count HEAD...origin/develop/nagata
```

右側が `0` なら、その相手ブランチに対して behind していません。

---

## このリポジトリでよくある注意

- デフォルトは `main` ではなく `master`
- `sample_submission/` は提出物に関わるので、作業ブランチを分けて慎重に触る
- `cardlist_referenced/` のローカルツール作業は、提出物変更と混ぜない方が安全
- ローカル `master` が古いことがあるので、基準確認には `origin/master` を優先する

---

## 提出コードまわりの主要ファイル

- エージェント本体: `sample_submission/main.py`
- 使用デッキ: `sample_submission/deck.csv`
- 提出コードの説明: `sample_submission/README.md`

`cg/` はコンペ提供のゲームエンジンなので、通常は編集しません。
