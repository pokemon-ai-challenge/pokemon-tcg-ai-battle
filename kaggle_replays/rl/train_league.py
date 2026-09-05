"""champion-league RL: 固定フィールド + 昇格したチャンピオン(過去の強い自分)プールに対して学習。

train_field(固定7アーキ相手)の拡張。相手プール = 固定フィールド(grounding) + チャンピオン
スナップショット(学習側の過去版, learner deck)。field-eval が margin 超えて改善したら現方策を
チャンピオンとしてプールに追加する(昇格ゲートを field 強度に紐付け=ミラー過学習を防ぐ)。

非転移対策(本プロジェクトの『ミラー勝ちはLB転移しない』教訓): 純ミラーleagueにせず
--field-frac で固定フィールドの割合を確保する。密報酬(PBRS)+多選択PL は既定ON
(collect_field/train_v3 の既定)。collect_field は無改変=毎iterプールを動的に組み替えるだけ。

すべて kaggle_replays/rl/ 内・torch は main のみ・production 無変更。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_field import parallel_collect_field  # noqa: E402
from torch_policy import TorchOptionPolicy  # noqa: E402
from train_v3 import Critic, wilson_lo, build_padded, compute_gae, policy_logp_entropy, export_temp  # noqa: E402
from train_field import dense_step_rewards, compute_gae_dense  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

# 現在メタ(2026-08-01, 全体1474分類, current_meta_shares_2026-08-01.json)加重。旧メタ(marnie低)から
# 更新: Grimmsnarl が現環境の非ミラー最大勢力。ミラー(alakazam)は champion スナップショット経由なので
# FIELD には含めない。local-Kaggle 差の"メタ不一致"分を縮めるための変更。
# (旧: lucario1257/archaludon1078/crustle737/dragapult625/marnie591/rocket247/shirona181)
# 実メタ(2026-08-03, climb 55186283 の実ラダー24試合リプレイ実測)。旧想定はミラー0%だったが
# 実際は33%がミラー(自分と同じフーディン)=最大の対面。lucario/dragapultは過大、garchomp/rocketは
# 実在せず。この乖離が local↔LB 非転移の元凶だった。alakazam(ミラー)は field_specs で強いclimbを使う。
# (旧: grimmsnarl199/lucario149/crustle102/archaludon96/dragapult64/garchomp24/rocket23, mirror無し)
FIELD = [
    ("alakazam", 33), ("marnie_grimmsnarl_ex", 21), ("archaludon_ex", 12),
    ("crustle", 12), ("mega_lucario_ex", 12), ("dragapult_ex", 4),
]

# gen2(2026-08-11取得, 2402試合/4804デッキ)の実測フィールド。上の FIELD は climb 提出の
# 実ラダー24試合から起こした推定だったのに対し、こちらは同じ日に取り直した上位200チームの
# 実測分布。オーガポン/メガユキメノコ/おまつりおんどが新規に入り、相手ポリシーも
# policy_weights_<arch>_g2.json(2026-08データで学習し直した模倣)、デッキも
# archetype_decks_g2/(オーガポンは旧型でなく現行型)を使う。
FIELD_G2 = [
    ("marnie_grimmsnarl_ex", 1082), ("alakazam", 891), ("mega_lucario_ex", 403),
    ("dragapult_ex", 381), ("mega_froslass_ex", 353), ("ogerpon_teal_ex", 310),
    ("archaludon_ex", 287), ("crustle", 287), ("shirona_garchomp_ex", 158),
    ("omatsuri_ondo", 138), ("rocket_mewtwo_ex", 112),
]

# アーキタイプごとに世代を選んだ混成フィールド。gen2版と7月版の相手モデルを
# 直接対決させ(compare_opponents.py)、**強い方**を採用した。デッキも採用世代に合わせる。
# 強い相手ほど grounding として価値が高い(弱い相手は勝率を水増しして判断を歪める)。
#   gen2採用(有意に強い): alakazam 67.3%(n300) / rocket_mewtwo 68.3% / crustle 65.8%
#                          / marnie 61.7% / dragapult 60.0%
#   7月採用(有意に強い) : mega_lucario 33.0%(n300) ← gen2は1257->403デッキと減り弱かった
#   差なし -> gen2       : archaludon 47.3%(n300) / shirona 52.0%(n300)
#                          (デッキ一致57/60・52/60でほぼ同一構築。現行メタ側にそろえる)
#   gen2のみ            : ogerpon_teal_ex / mega_froslass_ex / omatsuri_ondo
# share は gen2 実測(現行メタの分布)。
# 注: データ量と相手モデルの強さは対応しない(rocket 112デッキでも有意に強い)。実測必須。
FIELD_MIX = [
    ("marnie_grimmsnarl_ex", 1082, "_g2"), ("alakazam", 891, "_g2"),
    ("mega_lucario_ex", 403, ""), ("dragapult_ex", 381, "_g2"),
    ("mega_froslass_ex", 353, "_g2"), ("ogerpon_teal_ex", 310, "_g2"),
    ("archaludon_ex", 287, "_g2"), ("crustle", 287, "_g2"),
    ("shirona_garchomp_ex", 158, "_g2"), ("omatsuri_ondo", 138, "_g2"),
    ("rocket_mewtwo_ex", 112, "_g2"),
]

_DECKDIR_OF = {"": "archetype_decks", "_g2": "archetype_decks_g2"}


def _expand(rows, suffix=None):
    """(arch, share) か (arch, share, gen) を (arch, share, 接尾辞, デッキdir) に正規化する。"""
    out = []
    for row in rows:
        arch, share = row[0], row[1]
        gen = row[2] if len(row) > 2 else suffix
        out.append((arch, float(share), gen, _DECKDIR_OF[gen]))
    return out


# hard-tail混合(Codex 5.6-sol R4合意、2026-08-14)。g2top2_v032デッキ用のRL短分岐向け。
# 「元フィールド67% + hard-tail 33%」= mix の正規化シェアを0.67倍したものに、
# 実ラダーで比重が大きい/苦手な3対面(ミラーのogerpon_teal_ex, alakazam, crustle)へ
# 0.33を1/3ずつ均等配分して上乗せする。mix の定義(FIELD_MIX)は変えず、mix の展開結果
# から計算で導出する(mix が更新されたら自動追従し、二重管理でずれるのを防ぐ)。
_HARD_TAIL_ARCHS = ("ogerpon_teal_ex", "alakazam", "crustle")
_HARD_TAIL_FRAC = 0.33


def _mix_hard_tail(mix_rows, hard_archs=_HARD_TAIL_ARCHS, hard_frac=_HARD_TAIL_FRAC):
    """mix(_expand済み)から hard-tail 混合を計算で導出する。
    絶対スケールは mix の合計に合わせて維持する(share の桁を他プリセットと揃えて可読にする)。
    出力の各アーキ比率 = share_i/total = 0.67*(元のmix比率) + (hard_archsなら0.33/3を加算)。
    """
    total = sum(r[1] for r in mix_rows)
    names = [r[0] for r in mix_rows]
    missing = [a for a in hard_archs if a not in names]
    if missing:
        raise ValueError(f"mixhard: mix に無いアーキ指定: {missing}")
    base_scale = 1.0 - hard_frac
    bonus = total * hard_frac / len(hard_archs)
    out = []
    for arch, share, gen, dd in mix_rows:
        new_share = share * base_scale
        if arch in hard_archs:
            new_share += bonus
        out.append((arch, new_share, gen, dd))
    return out


FIELD_PRESETS = {
    "realmeta": _expand(FIELD, ""),
    "g2": _expand(FIELD_G2, "_g2"),
    "mix": _expand(FIELD_MIX),
}
FIELD_PRESETS["mixhard"] = _mix_hard_tail(FIELD_PRESETS["mix"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner-arch", default="alakazam")
    ap.add_argument("--learner-weights", default="policy_weights.json")
    ap.add_argument("--learner-deck-path", default=None,
                    help="学習側デッキの任意パス(既定=DECKDIR/<arch>/01.csv)。提出デッキ(xerosic等)で学習する用。")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr-policy", type=float, default=3e-4)
    ap.add_argument("--lr-value", type=float, default=1e-3)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--shaping-coef", type=float, default=1.0)
    ap.add_argument("--sparse", action="store_true", help="密報酬OFF(既定=PBRS密報酬ON)")
    # --- champion-league ---
    ap.add_argument("--deck-potential-coef", type=float, default=0.0,
                    help="山札保全PBRS項の係数(Φに min(deckCount,15)/15 を加算)。0=既定=従来不変。"
                         "deckout管理(ノコッチループ/せいなるはい)を学習で獲得させる用。")
    ap.add_argument("--value-potential-coef", type=float, default=0.0,
                    help="価値ネットPBRS項の係数(Φに (win_prob-0.5)*2 を加算)。0=既定=従来不変。"
                         "学習した勝率予測(将来価値の直感)を密な信号にして方策に長期価値を学ばせる。")
    ap.add_argument("--value-weights", default="ptcg_ai/learning/value_weights.json",
                    help="価値ネット重みJSON(value_potential_coef>0 のとき使用)。")
    ap.add_argument("--field-frac", type=float, default=0.5,
                    help="チャンピオン存在時、相手サンプルに占める固定フィールドの割合(残りがチャンピオン)。"
                         "1.0=従来のfield only、0.0=チャンピオンのみ(非推奨:ミラー過学習)。")
    ap.add_argument("--promote-margin", type=float, default=0.03,
                    help="field-eval が前チャンピオンから何pt改善したら昇格(スナップショットをプール追加)。")
    ap.add_argument("--max-champions", type=int, default=8, help="プール内チャンピオン上限(超過は最古を破棄)。")
    ap.add_argument("--field-preset", default="realmeta", choices=sorted(FIELD_PRESETS),
                    help="固定フィールドの世代。realmeta=従来(既定で挙動不変)、"
                         "g2=2026-08取得の実測11アーキ(全部gen2)、"
                         "mix=アーキごとに強い方を実測で選んだ混成(推奨)、"
                         "mixhard=mixの67%%+hard-tail3対面(ogerpon_teal_ex/alakazam/crustle)33%%")
    ap.add_argument("--mirror-weights", default=None,
                    help="ミラー(alakazam)相手の重み。既定=policy_weights_alakazam_rl_climb.json")
    ap.add_argument("--field-only", default=None, metavar="ARCH",
                    help="指定アーキ以外の share を0にして単一相手の専用モデルを作る。"
                         "--field-frac 1.0 と併用すると champion も混ざらない純粋な対策学習になる。")
    ap.add_argument("--field-weight", action="append", default=None, metavar="ARCH=W",
                    help="FIELD の特定アーキ share を上書き(例 --field-weight crustle=350)。"
                         "偏重診断用の opt-in。未指定なら FIELD の既定比率(本番不変)。")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="league")
    ap.add_argument("--resume", default=None,
                    help="チェックポイント(.pt)から再開。存在すれば途中状態をロードしbaselineをskip。")
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="N iterごとに全状態チェックポイント保存(中断再開/Kaggle 12h制限対策)。")
    args = ap.parse_args()

    device = args.device
    arch = args.learner_arch
    dense = not args.sparse
    lw = Path(args.learner_weights)
    learner_w = lw if lw.is_absolute() else WDIR / lw.name
    out = WDIR / f"policy_weights_{arch}_rl_{args.tag}.json"
    tmp = _HERE / f"_tmp_policy_{arch}_{args.tag}.json"
    logpath = _HERE / f"_train_{arch}_{args.tag}.log"
    champ_dir = _HERE / "league_champions" / f"{arch}_{args.tag}"
    champ_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _HERE / f"_ckpt_{arch}_{args.tag}.pt"

    field = FIELD_PRESETS[args.field_preset]
    _default_deckdir = DECKDIR.parent / field[0][3]

    learner_deck = read_deck_csv_file(args.learner_deck_path if args.learner_deck_path
                                      else str(_default_deckdir / arch / "01.csv"))
    # 固定フィールド(grounding)。alakazam(ミラー)は強いclimbを相手にする(実ラダーのミラーは強い)。
    # --mirror-weights で差し替え可(gen2版で「相手も模倣ポリシー」にそろえたい場合など)。
    _mirror_w = args.mirror_weights or "policy_weights_alakazam_rl_climb.json"
    _mirror_w = _mirror_w if Path(_mirror_w).is_absolute() else str(WDIR / _mirror_w)
    field_specs = [((_mirror_w if a == "alakazam" else str(WDIR / f"policy_weights_{a}{gen}.json")),
                    read_deck_csv_file(str(DECKDIR.parent / dd / a / "01.csv")))
                   for a, _, gen, dd in field]
    _missing = [p for p, _ in field_specs if not Path(p).exists()]
    if _missing:
        raise SystemExit("相手重みが無い: " + ", ".join(Path(m).name for m in _missing))
    field_shares_raw = [s for _, s, _, _ in field]
    if args.field_only:
        names = [a for a, _, _, _ in field]
        if args.field_only not in names:
            raise SystemExit(f"--field-only {args.field_only} は preset に無い: {names}")
        field_shares_raw = [(1.0 if a == args.field_only else 0.0) for a in names]
        print(f"[field-only] {args.field_only} 以外の share を0にしました", flush=True)
    if args.field_weight:  # ARCH=W 上書き(偏重診断用、未指定なら本番不変)。
        _ov = {}
        for kv in args.field_weight:
            a, _, w = kv.partition("=")
            _ov[a.strip()] = float(w)
        _names = [a for a, _, _, _ in field]
        field_shares_raw = [(_ov.get(a, s)) for a, s in zip(_names, field_shares_raw)]
        print(f"[field-weight override] {_ov} -> shares={dict(zip(_names, field_shares_raw))}", flush=True)
    print(f"[field-preset={args.field_preset}] {len(field)}arch mirror={Path(_mirror_w).name}", flush=True)
    for a, sh, gen, dd in field:
        print(f"    {a:<22} share={sh:>6.0f}  weights={'climb(mirror)' if a=='alakazam' else 'policy_weights_'+a+gen+'.json'}  deck={dd}", flush=True)
    field_total = sum(field_shares_raw)
    # チャンピオンプール(学習側の過去スナップショット, learner deck)。(weights_path, deck, wr)
    champions: list[tuple[str, list[int], float]] = []

    def build_pool():
        """毎iterの相手プールを (opp_specs, shares) で返す。field_frac で固定フィールド割合を確保。"""
        if not champions:
            return field_specs, field_shares_raw
        f_w = [(s / field_total) * args.field_frac for s in field_shares_raw]
        c_each = (1.0 - args.field_frac) / len(champions)
        c_specs = [(w, d) for (w, d, _) in champions]
        c_w = [c_each] * len(champions)
        return field_specs + c_specs, f_w + c_w

    base_payload = json.loads(learner_w.read_text(encoding="utf-8"))
    policy = TorchOptionPolicy.from_json(learner_w).float().to(device)
    std = base_payload["standardization"]
    critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr_value)

    def do_eval(seed, ngames):
        """評価は固定フィールドに対して行う(グラウンディング指標=昇格ゲート/best選択の基準)。"""
        export_temp(policy, base_payload, tmp)
        _, w, v, _, _ = parallel_collect_field(str(tmp), field_specs, field_shares_raw, learner_deck,
                                               ngames, seed, temperature=0.01, workers=args.workers)
        return (w / v if v else float("nan")), w, v

    def save_ckpt(it):
        """全学習状態を保存(中断再開用)。champions は deck を learner_deck から再構成できるので (path,wr) のみ。"""
        torch.save({
            "iter": it, "policy": policy.state_dict(), "critic": critic.state_dict(),
            "opt_p": opt_p.state_dict(), "opt_v": opt_v.state_dict(),
            "best_wr": best_wr, "best_iter": best_iter, "best_state": best_state,
            "last_promoted_wr": last_promoted_wr, "wr0": wr0,
            "champions": [(w, wr) for (w, _d, wr) in champions], "history": history,
        }, ckpt_path)

    print(f"device={device} learner={arch} LEAGUE(field_frac={args.field_frac} promote_margin={args.promote_margin} "
          f"max_champ={args.max_champions}) dense={dense} workers={args.workers} -> {out.name}", flush=True)

    history = []
    resume_from = Path(args.resume) if args.resume else None
    if resume_from and resume_from.exists():
        ck = torch.load(resume_from, map_location=device)
        policy.load_state_dict(ck["policy"]); critic.load_state_dict(ck["critic"])
        opt_p.load_state_dict(ck["opt_p"]); opt_v.load_state_dict(ck["opt_v"])
        best_wr = ck["best_wr"]; best_iter = ck["best_iter"]; best_state = ck["best_state"]
        last_promoted_wr = ck["last_promoted_wr"]; wr0 = ck["wr0"]
        champions = [(w, learner_deck, wr) for (w, wr) in ck["champions"]]
        history = ck.get("history", [])
        start_iter = int(ck["iter"]) + 1
        print(f"[resume] {resume_from} から iter {start_iter} 再開 (best {best_wr:.3f}@{best_iter}, champ {len(champions)})", flush=True)
    else:
        wr0, w0, v0 = do_eval(900000, args.eval_games)
        print(f"[iter 0] baseline field-weighted greedy {w0}/{v0} = {wr0:.3f} (CI_lo {wilson_lo(w0,v0):.3f})", flush=True)
        history.append({"iter": 0, "eval_winrate": wr0, "eval_wins": w0, "eval_valid": v0, "champions": 0})
        best_wr, best_iter = wr0, 0
        best_state = copy.deepcopy(policy.state_dict())
        last_promoted_wr = wr0
        start_iter = 1

    t0 = time.time()
    for it in range(start_iter, args.iters + 1):
        export_temp(policy, base_payload, tmp)
        opp_specs, shares = build_pool()
        trajs, wins, valid, errors, per_opp = parallel_collect_field(
            str(tmp), opp_specs, shares, learner_deck, args.games_per_iter,
            seed0=it * 100000, temperature=args.temperature, workers=args.workers,
            deck_coef=args.deck_potential_coef,
            value_coef=args.value_potential_coef, value_weights=args.value_weights)
        if not trajs:
            print(f"[iter {it}] no trajs", flush=True); continue
        batch = build_padded(trajs, device)
        with torch.no_grad():
            values = critic(batch["state_rows"])
        if dense:
            step_rews = dense_step_rewards(trajs, args.gamma, args.shaping_coef)
            adv, vtarget = compute_gae_dense(batch["lengths"], step_rews, values, args.gamma, args.lam, device)
        else:
            adv, vtarget = compute_gae(batch["lengths"], batch["rewards"], values, args.gamma, args.lam, device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pl = vl = en = 0.0
        for _ in range(args.epochs):
            new_logp, ent = policy_logp_entropy(policy, batch)
            ratio = torch.exp(new_logp - batch["old_logp"])
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
            pol_loss = -torch.min(s1, s2).mean() - args.entropy * ent.mean()
            opt_p.zero_grad(); pol_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt_p.step()
            v_pred = critic(batch["state_rows"])
            val_loss = ((v_pred - vtarget) ** 2).mean()
            opt_v.zero_grad(); val_loss.backward(); opt_v.step()
            pl, vl, en = pol_loss.item(), val_loss.item(), ent.mean().item()

        train_wr = wins / valid if valid else float("nan")
        print(f"[iter {it}] train_wr {train_wr:.3f} ({wins}/{valid} err{errors}) champ={len(champions)} "
              f"steps {batch['n']} pol {pl:.4f} val {vl:.4f} ent {en:.3f} {time.time()-t0:.0f}s", flush=True)
        rec = {"iter": it, "train_winrate": train_wr, "steps": batch["n"], "pol_loss": pl, "val_loss": vl,
               "entropy": en, "champions": len(champions)}
        if it % args.eval_every == 0 or it == args.iters:
            wr, w, v = do_eval(900000, args.eval_games)
            rec.update({"eval_winrate": wr, "eval_wins": w, "eval_valid": v})
            star = ""
            if wr > best_wr:
                best_wr, best_iter = wr, it
                best_state = copy.deepcopy(policy.state_dict()); star = " *BEST*"
                # 途中保存(長時間run/中断対策): 新bestが出た時点で即 out へ書き出す。
                # この時点の policy が best なので export_temp(policy,...) でよい。
                export_temp(policy, base_payload, out)
            # 昇格ゲート: field-eval が前チャンピオン(なければbaseline)から margin 超えて改善したら追加。
            promo = ""
            if wr >= last_promoted_wr + args.promote_margin:
                snap = champ_dir / f"champ_it{it}_wr{wr:.3f}.json"
                export_temp(policy, base_payload, snap)
                champions.append((str(snap), learner_deck, wr))
                if len(champions) > args.max_champions:
                    champions.pop(0)  # 最古を破棄
                last_promoted_wr = wr
                promo = f" PROMOTE->champ#{len(champions)}(wr{wr:.3f})"
                rec["promoted_iter"] = it; rec["promoted_wr"] = wr
            print(f"    [eval] field {w}/{v} = {wr:.3f} (CI_lo {wilson_lo(w,v):.3f}) "
                  f"best={best_wr:.3f}@{best_iter} champ={len(champions)}{star}{promo}", flush=True)
        history.append(rec)
        logpath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        if it % args.ckpt_every == 0 or it == args.iters:
            save_ckpt(it)

    policy.load_state_dict(best_state)
    payload = policy.to_json_payload(base_payload)
    payload.setdefault("meta", {}).update({"rl_finetuned": True, "rl_league": True, "rl_best_iter": best_iter,
                                           "rl_best_eval_winrate": best_wr, "rl_baseline": wr0,
                                           "rl_champions_final": len(champions), "rl_field_frac": args.field_frac})
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"exported BEST (iter {best_iter}, field {best_wr:.3f}, baseline {wr0:.3f}, champions {len(champions)}) -> {out}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
