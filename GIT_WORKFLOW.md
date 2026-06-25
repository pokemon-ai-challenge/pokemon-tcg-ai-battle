# Git / GitHub 作業メモ

このファイルは、初心者向けの Git / GitHub 作業手順メモです。

このリポジトリでは、通常は `master` を元本ブランチとして扱い、作業は別ブランチで進めます。

---

## まず覚える方針

- `master` で直接作業しない
- 作業前に `master` を最新にする
- 自分の作業は `feature/...` ブランチで行う
- GitHub に上げるときは `git push` まで行う

---

## いまの状態を確認する

最初に、今どのブランチにいるかと、未保存の変更があるかを確認します。

```powershell
git status
git branch --show-current
```

見方:

- `git status` は、変更されたファイルやコミット待ちのファイルを表示します
- `git branch --show-current` は、今いるブランチ名を表示します

---

## 新しく作業を始める

`master` を最新にしてから、新しい作業ブランチを作ります。

```powershell
git switch master
git pull origin master
git switch -c feature/branch-name
```

例:

```powershell
git switch -c feature/improve-agent-logic
```

見方:

- `git switch master` で `master` に移動します
- `git pull origin master` で GitHub 上の最新状態を取り込みます
- `git switch -c ...` で新しいブランチを作って、そのまま移動します

---

## ファイルを編集したあと

編集したら、変更を確認してコミットします。

```powershell
git status
git add .
git commit -m "変更内容を書く"
```

コミットメッセージ例:

```powershell
git commit -m "feat: improve main action selection"
git commit -m "docs: add git workflow guide"
git commit -m "fix: correct deck export behavior"
```

見方:

- `git add .` は、今のフォルダ以下の変更をステージします
- `git commit -m "..."` は、ステージした変更を1つの履歴として保存します

---

## GitHub に上げる

初回の push では、`-u` を付けます。

```powershell
git push -u origin feature/branch-name
```

2回目以降は、同じブランチなら次で大丈夫です。

```powershell
git push
```

見方:

- `origin` は GitHub 側のリモート名です
- `-u` を付けると、そのブランチの送り先を覚えてくれます
- `git push` まで終わって初めて GitHub 側に反映されます

---

## 作業の基本セット

最小の流れだけ覚えるなら、まずはこれで十分です。

```powershell
git status
git branch --show-current
git switch master
git pull origin master
git switch -c feature/branch-name
```

編集後:

```powershell
git status
git add .
git commit -m "変更内容を書く"
git push -u origin feature/branch-name
```

---

## `master` が更新されたあと、自分のブランチに取り込む

他の人の変更が `master` に入ったあと、自分の作業ブランチにも最新を取り込みたいことがあります。

そのときは、まず `master` を更新してから、自分のブランチに戻ってマージします。

```powershell
git switch master
git fetch origin
git merge --ff-only origin/master
git switch feature/branch-name
git merge master
```

見方:

- 先に `master` を最新にします
- そのあと `git merge master` で、自分のブランチへ最新を取り込みます
- コンフリクトが出たら、該当ファイルを直してから改めてコミットします

---

## ブランチ名の付け方

このリポジトリでは、作業ブランチは `feature/...` を基本にすると分かりやすいです。

例:

- `feature/improve-agent-logic`
- `feature/update-readme`
- `feature/deck-adjustment`

試行錯誤や学習用として分けたいだけなら、`experiment/...` でも構いません。

---

## よくある注意

- `master` のまま編集を始めない
- `git push` を忘れると GitHub には反映されない
- `main` ではなく `master` が基準のリポジトリもあるので、毎回決め打ちしない
- ブランチを切る前に `git status` を見て、前の作業が混ざっていないか確認する

---

## 困ったときに見るコマンド

```powershell
git status
git branch --show-current
git branch -a -vv
git log --oneline --decorate -10
```

見方:

- `git branch -a -vv` は、ローカルとリモートのブランチ一覧を見たいときに便利です
- `git log --oneline --decorate -10` は、最近のコミット履歴を短く確認できます

---

## このリポジトリでよく触る場所

- 提出エージェント本体: `sample_submission/main.py`
- 使用デッキ: `sample_submission/deck.csv`
- 提出コードの説明: `sample_submission/README.md`

`cg/` はコンペ提供のゲームエンジンなので、通常は変更しません。
