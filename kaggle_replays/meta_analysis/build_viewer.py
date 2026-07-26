#!/usr/bin/env python3
"""meta_report.json を読み、自己完結の分析ビュアー(meta_viewer.html)を生成する。

チャート:
  - アーキタイプ・シェア: ランキング棒(出現数 / 使用者数トグル) + 円グラフ(上位8+other)
  - 上位-field ダイバージングバー(上位偏重 / field偏重)
  - other 主軸カード内訳
  - アーキタイプ別カード採用率(セレクタ切替, 横棒)

配色は dataviz スキルの検証済みパレット。ブラウザでファイルを開くだけで動く。
使い方: python build_viewer.py   (先に analyze_meta.py を実行しておく)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent
_JSON = _HERE / "output" / "meta_report.json"
_OUT = _HERE / "output" / "meta_viewer.html"
_JP_CSV = _REPO / "data" / "JP_Card_Data.csv"
_EN_CSV = _REPO / "data" / "EN_Card_Data.csv"

MAX_CARDS_PER_ARCH = 60


def _load_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    if not path.exists():
        return names
    for row in csv.reader(path.open(encoding="utf-8", errors="replace")):
        if row and row[0].isdigit():
            names[int(row[0])] = row[1] if len(row) > 1 else row[0]
    return names


def _collect_card_ids(report) -> set[int]:
    ids: set[int] = set()
    for c in report.get("other_candidates", []):
        ids.add(c["card_id"])
    for d in report.get("card_adoption", {}).values():
        for key in ("cards", "cards_top"):
            for c in d.get(key, []):
                ids.add(c["card_id"])
    for decks in report.get("top_decks", {}).values():
        for deck in decks:
            for c in deck["cards"]:
                ids.add(c["card_id"])
    return ids


def main():
    report = json.loads(_JSON.read_text(encoding="utf-8"))
    # 採用率は上位カードだけに絞ってサイズ削減
    for a, d in report["card_adoption"].items():
        d["cards"] = d["cards"][:MAX_CARDS_PER_ARCH]
        if "cards_top" in d:
            d["cards_top"] = d["cards_top"][:MAX_CARDS_PER_ARCH]
    data_json = json.dumps(report, ensure_ascii=False)

    # 日英カード名(ビュアーで表示中のカードIDだけ)
    jp = _load_names(_JP_CSV)
    en = _load_names(_EN_CSV)
    used = _collect_card_ids(report)
    names = {str(cid): {"ja": jp.get(cid, str(cid)), "en": en.get(cid, jp.get(cid, str(cid)))} for cid in used}
    names_json = json.dumps(names, ensure_ascii=False)

    # アーキタイプ名: ja=rough_predictor.json の display_name / en=slug
    rp_path = _REPO / "sample_submission" / "ptcg_ai" / "opponent_modeling" / "rough_predictor.json"
    rp = json.loads(rp_path.read_text(encoding="utf-8")) if rp_path.exists() else {"archetypes": {}}
    arch_names = {"other": {"ja": "その他", "en": "other"}}
    for slug, a in rp.get("archetypes", {}).items():
        arch_names[slug] = {"ja": a.get("display_name") or slug, "en": slug}
    arch_json = json.dumps(arch_names, ensure_ascii=False)

    html = (
        _TEMPLATE.replace("__DATA__", data_json)
        .replace("__NAMES__", names_json)
        .replace("__ARCHNAMES__", arch_json)
    )
    _OUT.write_text(html, encoding="utf-8")
    print(f"wrote: {_OUT}  (cards with names: {len(names)})")


_TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>環境メタ分析ビュアー</title>
<style>
  :root{
    color-scheme: light;
    --surface-1:#fcfcfb; --page:#f9f9f7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
    --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,0.10);
    --seq:#256abf; --seq-soft:#9ec5f4;
    --pole-top:#2a78d6; --pole-field:#e34948; --neutral:#f0efec;
    --c1:#2a78d6;--c2:#008300;--c3:#e87ba4;--c4:#eda100;--c5:#1baf7a;--c6:#eb6834;--c7:#4a3aa7;--c8:#e34948;--c9:#898781;
  }
  @media (prefers-color-scheme: dark){
    :root:where(:not([data-theme="light"])){
      color-scheme: dark;
      --surface-1:#1a1a19; --page:#0d0d0d;
      --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
      --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,0.10);
      --seq:#3987e5; --seq-soft:#184f95;
      --pole-top:#3987e5; --pole-field:#e66767; --neutral:#383835;
      --c1:#3987e5;--c2:#008300;--c3:#d55181;--c4:#c98500;--c5:#199e70;--c6:#d95926;--c7:#9085e9;--c8:#e66767;--c9:#898781;
    }
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--page);color:var(--text-primary);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;line-height:1.5}
  .wrap{max-width:1100px;margin:0 auto;padding:24px 20px 80px}
  h1{font-size:22px;margin:0 0 4px}
  h2{font-size:16px;margin:0 0 12px}
  .sub{color:var(--text-secondary);margin:0 0 24px}
  .card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;
    padding:18px 20px;margin:0 0 20px}
  .row{display:flex;gap:20px;flex-wrap:wrap}
  .row > .card{flex:1;min-width:320px}
  .bars{display:flex;flex-direction:column;gap:6px}
  .bar-row{display:grid;grid-template-columns:150px 1fr 64px;align-items:center;gap:10px}
  .bar-label{color:var(--text-secondary);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .bar-track{background:var(--grid);border-radius:4px;height:16px;position:relative;overflow:hidden}
  .bar-fill{height:100%;border-radius:4px;background:var(--seq)}
  .bar-val{font-variant-numeric:tabular-nums;color:var(--text-primary);text-align:right}
  .div-track{position:relative;height:16px;background:transparent}
  .div-mid{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--baseline)}
  .div-fill{position:absolute;top:0;height:100%;border-radius:4px}
  .toggle{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:12px}
  .toggle button{background:transparent;border:0;color:var(--text-secondary);padding:6px 12px;cursor:pointer;font:inherit}
  .toggle button.on{background:var(--seq);color:#fff}
  select{font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--border);
    background:var(--surface-1);color:var(--text-primary)}
  .legend{display:flex;flex-wrap:wrap;gap:10px 16px;margin-top:12px}
  .legend span{display:inline-flex;align-items:center;gap:6px;color:var(--text-secondary);font-size:12px}
  .dot{width:10px;height:10px;border-radius:2px;display:inline-block}
  .tip{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--surface-1);
    padding:6px 9px;border-radius:6px;font-size:12px;opacity:0;transition:opacity .08s;z-index:10;white-space:nowrap}
  .muted{color:var(--muted);font-size:12px}
  .pie-wrap{display:flex;gap:20px;align-items:center;flex-wrap:wrap}
  svg{display:block}
  .matrix-wrap{overflow-x:auto}
  table.matrix{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
  table.matrix th,table.matrix td{border:1px solid var(--border);padding:5px 9px;text-align:center;white-space:nowrap}
  table.matrix th.cardname,table.matrix td.cardname{text-align:left;color:var(--text-secondary);position:sticky;left:0;background:var(--surface-1);z-index:1}
  table.matrix thead th{background:var(--neutral);color:var(--text-primary);font-weight:600}
  table.matrix td.n0{color:var(--muted)}
  table.matrix td.hit{color:var(--text-primary);font-weight:600}
  table.matrix tbody tr:hover td{background:rgba(37,106,191,0.06)}
  .deckhead{font-size:11px;color:var(--text-secondary);font-weight:400}
</style>
</head>
<body>
<div class="wrap">
  <h1>環境メタ分析ビュアー</h1>
  <div class="toggle" id="langToggle" style="margin-bottom:10px">
    <button data-lang="ja" class="on">日本語</button>
    <button data-lang="en">English</button>
  </div>
  <p class="sub" id="summary"></p>

  <div class="row">
    <div class="card">
      <h2>アーキタイプ・シェア</h2>
      <div class="toggle" id="shareToggle">
        <button data-k="appear" class="on">出現数シェア</button>
        <button data-k="player">使用者数シェア</button>
      </div>
      <div class="bars" id="shareBars"></div>
      <p class="muted">出現数=試合単位の露出（連投で膨らむ）／使用者数=distinct player。self-play重み付けは使用者数側が歪みにくい。</p>
    </div>
    <div class="card">
      <h2>シェア円グラフ（上位8＋その他）</h2>
      <div class="toggle" id="pieBandToggle">
        <button data-band="band_all" class="on">全体</button>
        <button data-band="band_le200">上位(rank≤200)</button>
        <button data-band="band_le100">超上位(rank≤100)</button>
      </div>
      <p class="muted" id="pieN"></p>
      <div class="pie-wrap">
        <div id="pie"></div>
        <div class="legend" id="pieLegend"></div>
      </div>
      <p class="muted">rank≤100(602件)とrank≤200(610件)はこのデータでは差が僅少（101–200位帯にほぼ標本なし）。実質は上位 vs 全体。</p>
    </div>
  </div>

  <div class="card">
    <h2>上位 vs field 偏り（上位%−field%）</h2>
    <div class="bars" id="divBars"></div>
    <div class="legend">
      <span><i class="dot" style="background:var(--pole-top)"></i>上位に偏重（上位が好んで握る）</span>
      <span><i class="dot" style="background:var(--pole-field)"></i>field に偏重（下位に多く上位は避ける）</span>
    </div>
    <p class="muted">出現数100件未満のアーキタイプは偏りが不安定なため非表示。ブリジュラスex(ルカリオexと並ぶfield偏重)はここに表示。</p>
  </div>

  <div class="card">
    <h2><code>other</code> の主軸カード内訳 — 未登録アーキタイプ候補</h2>
    <div class="bars" id="otherBars"></div>
    <p class="muted" id="otherNote"></p>
  </div>

  <div class="card">
    <h2>アーキタイプ別 カード採用率</h2>
    <div style="margin-bottom:12px">
      <select id="archSel"></select>
      <span class="muted" id="archMeta"></span>
    </div>
    <div class="toggle" id="adoptScopeToggle">
      <button data-scope="all" class="on">全体</button>
      <button data-scope="top">上位(rank≤100)</button>
    </div>
    <div class="bars" id="adoptBars"></div>
    <p class="muted" id="adoptScopeNote"></p>
  </div>

  <div class="card">
    <h2>アーキタイプ別 上位デッキのカード構成</h2>
    <div style="margin-bottom:12px">
      <select id="deckArchSel"></select>
      <span class="muted" id="deckArchMeta"></span>
    </div>
    <div class="matrix-wrap"><div id="deckMatrix"></div></div>
    <p class="muted">rank が最も良い順に distinct な構築を最大5件抽出。各セルはそのデッキに入っている枚数（基本エネルギーは末尾に集約）。列見出しは rank と使用者名。</p>
  </div>
</div>
<div class="tip" id="tip"></div>

<script>
const DATA = __DATA__;
const NAMES = __NAMES__;
const ARCHNAMES = __ARCHNAMES__;
let LANG = 'ja';
function nameOf(id, fallback){ const n = NAMES[id]; return n ? (n[LANG] || fallback) : fallback; }
function archName(slug){ const a = ARCHNAMES[slug]; return a ? (a[LANG] || slug) : slug; }
let shareKind = 'appear';
const tip = document.getElementById('tip');
function showTip(html, e){ tip.innerHTML=html; tip.style.opacity=1;
  tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px'; }
function hideTip(){ tip.style.opacity=0; }
const CAT = ['--c1','--c2','--c3','--c4','--c5','--c6','--c7','--c8','--c9'].map(v=>`var(${v})`);

// summary
const m = DATA.meta;
document.getElementById('summary').textContent =
  `総デッキ出現 ${m.total_decks.toLocaleString()} 件 ／ 上位(rank≤${m.top_rank_cutoff}) ${m.n_top}・field ${m.n_field}・rank不明 ${m.n_unknown_rank}`;

// ---- share bars ----
const share = DATA.archetype_share;
const totAppear = share.reduce((s,r)=>s+r.overall_appearances,0);
const totPlayer = share.reduce((s,r)=>s+r.player_count,0);
function renderShare(kind){
  const rows = share.map(r=>({
    name:r.archetype,
    val: kind==='appear'? r.overall_appearances : r.player_count,
    pct: kind==='appear'? 100*r.overall_appearances/totAppear : 100*r.player_count/totPlayer,
  })).sort((a,b)=>b.val-a.val);
  const max = Math.max(...rows.map(r=>r.pct));
  const el = document.getElementById('shareBars'); el.innerHTML='';
  rows.forEach(r=>{
    const nm=archName(r.name);
    const d=document.createElement('div'); d.className='bar-row';
    d.innerHTML=`<div class="bar-label" title="${nm}">${nm}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(r.pct/max*100).toFixed(1)}%"></div></div>
      <div class="bar-val">${r.pct.toFixed(1)}%</div>`;
    d.querySelector('.bar-track').addEventListener('mousemove',e=>
      showTip(`<b>${nm}</b><br>${r.pct.toFixed(1)}% ・ ${r.val.toLocaleString()}${kind==='appear'?'出現':'人'}`,e));
    d.querySelector('.bar-track').addEventListener('mouseleave',hideTip);
    el.appendChild(d);
  });
}
document.querySelectorAll('#shareToggle button').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('#shareToggle button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); shareKind=b.dataset.k; renderShare(shareKind);
}));
renderShare(shareKind);

// ---- pie (top8 + その他), band-aware, re-renders on language/band toggle ----
let pieBand = 'band_all';
function renderPie(){
  document.getElementById('pie').innerHTML='';
  document.getElementById('pieLegend').innerHTML='';
  const band=pieBand;
  const total=share.reduce((s,r)=>s+(r[band]||0),0) || 1;
  // 'other'(その他)は集約スライスに畳むので上位8の対象から除外(その他の二重表示を防ぐ)
  const real=share.filter(r=>r.archetype!=='other').sort((a,b)=>(b[band]||0)-(a[band]||0));
  const top=real.slice(0,8);
  const topSum=top.reduce((s,r)=>s+(r[band]||0),0);
  const restCount=total-topSum;  // 残り実アーキタイプ + other をまとめて『その他』
  const slices=top.map((r,i)=>({name:archName(r.archetype),pct:100*(r[band]||0)/total,color:CAT[i]}));
  slices.push({name:archName('other'),pct:100*restCount/total,color:CAT[8]});
  const bandLabel={band_all:'全体',band_le200:'上位 rank≤200',band_le100:'超上位 rank≤100'}[band];
  document.getElementById('pieN').textContent=`${bandLabel} ・ n=${total.toLocaleString()} 出現`;
  const R=90,C=110,cx=C,cy=C; let ang=-Math.PI/2;
  const ns='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(ns,'svg'); svg.setAttribute('width',C*2); svg.setAttribute('height',C*2);
  slices.forEach(s=>{
    const a2=ang+2*Math.PI*(s.pct/100);
    const x1=cx+R*Math.cos(ang),y1=cy+R*Math.sin(ang),x2=cx+R*Math.cos(a2),y2=cy+R*Math.sin(a2);
    const large=(a2-ang)>Math.PI?1:0;
    const p=document.createElementNS(ns,'path');
    p.setAttribute('d',`M${cx},${cy} L${x1},${y1} A${R},${R} 0 ${large} 1 ${x2},${y2} Z`);
    p.setAttribute('fill',s.color); p.setAttribute('stroke','var(--surface-1)'); p.setAttribute('stroke-width','2');
    p.addEventListener('mousemove',e=>showTip(`<b>${s.name}</b><br>${s.pct.toFixed(1)}%`,e));
    p.addEventListener('mouseleave',hideTip);
    // direct label (relief for light-mode contrast WARN)
    if(s.pct>=4){const mid=(ang+a2)/2,lx=cx+R*0.62*Math.cos(mid),ly=cy+R*0.62*Math.sin(mid);
      const t=document.createElementNS(ns,'text');t.setAttribute('x',lx);t.setAttribute('y',ly);
      t.setAttribute('fill','#fff');t.setAttribute('font-size','11');t.setAttribute('text-anchor','middle');
      t.setAttribute('font-weight','600');t.textContent=s.pct.toFixed(0)+'%';svg.appendChild(p);svg.appendChild(t);}
    else svg.appendChild(p);
    ang=a2;
  });
  document.getElementById('pie').appendChild(svg);
  const lg=document.getElementById('pieLegend');
  slices.forEach(s=>{const sp=document.createElement('span');
    sp.innerHTML=`<i class="dot" style="background:${s.color}"></i>${s.name} ${s.pct.toFixed(1)}%`;lg.appendChild(sp);});
}
document.querySelectorAll('#pieBandToggle button').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('#pieBandToggle button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); pieBand=b.dataset.band; renderPie();
}));
renderPie();

// ---- diverging top vs field, re-renders on language toggle ----
// 出現数が少ないアーキタイプ(±0.1程度の偏りしか出ず意味が薄い)は除外し、
// 実在感のある型だけ表示する。閾値未満はノイズとして中央に溜まり大バーを埋もれさせるため。
const DIV_MIN_OVERALL = 100;
function renderDiv(){
  const rows=[...share].filter(r=>r.overall_appearances>=DIV_MIN_OVERALL
      && r.top_appearances+r.field_appearances>0)
    .sort((a,b)=>b.top_minus_field_pct-a.top_minus_field_pct);
  const maxAbs=Math.max(...rows.map(r=>Math.abs(r.top_minus_field_pct)),1);
  const el=document.getElementById('divBars'); el.innerHTML='';
  rows.forEach(r=>{
    const nm=archName(r.archetype);
    const v=r.top_minus_field_pct, w=Math.abs(v)/maxAbs*50;
    const d=document.createElement('div'); d.className='bar-row';
    d.innerHTML=`<div class="bar-label" title="${nm}">${nm}</div>
      <div class="div-track"><div class="div-mid"></div>
        <div class="div-fill" style="${v>=0?`left:50%;background:var(--pole-top)`:`right:50%;background:var(--pole-field)`};width:${w}%"></div>
      </div>
      <div class="bar-val">${v>=0?'+':''}${v.toFixed(1)}</div>`;
    const tr=d.querySelector('.div-track');
    tr.addEventListener('mousemove',e=>showTip(`<b>${nm}</b><br>上位 ${r.top_pct}% ／ field ${r.field_pct}%`,e));
    tr.addEventListener('mouseleave',hideTip);
    el.appendChild(d);
  });
}
renderDiv();

// ---- other candidates (re-renders on language toggle) ----
function renderOther(){
  const rows=DATA.other_candidates.slice(0,15);
  const max=Math.max(...rows.map(r=>r.pct_of_other));
  const el=document.getElementById('otherBars'); el.innerHTML='';
  rows.forEach(r=>{
    const nm=nameOf(r.card_id, r.name);
    const d=document.createElement('div'); d.className='bar-row';
    d.innerHTML=`<div class="bar-label" title="${nm}">${nm}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(r.pct_of_other/max*100).toFixed(1)}%;background:var(--seq)"></div></div>
      <div class="bar-val">${r.pct_of_other}%</div>`;
    const tr=d.querySelector('.bar-track');
    tr.addEventListener('mousemove',e=>showTip(`<b>${nm}</b> (ID ${r.card_id})<br>${r.count} decks ・ other内 ${r.pct_of_other}%`,e));
    tr.addEventListener('mouseleave',hideTip);
    el.appendChild(d);
  });
  document.getElementById('otherNote').textContent=`other 総数 ${DATA.other_total} デッキ。シェア上位は独立アーキタイプ化の候補。`;
}
renderOther();

// ---- adoption by archetype ----
const archSel=document.getElementById('archSel');
(function(){
  const arches=Object.keys(DATA.card_adoption)
    .sort((a,b)=>DATA.card_adoption[b].distinct_builds-DATA.card_adoption[a].distinct_builds);
  arches.forEach(a=>{const o=document.createElement('option');o.value=a;o.textContent=archName(a);archSel.appendChild(o);});
})();
function refreshArchOptions(){
  [...archSel.options].forEach(o=>{o.textContent=archName(o.value);});
}
let adoptScope='all';
function renderAdopt(a){
  const d=DATA.card_adoption[a]; if(!d) return;
  const note=document.getElementById('adoptScopeNote');
  const topBtn=document.querySelector('#adoptScopeToggle button[data-scope="top"]');
  const topN=d.top_distinct_builds||0;
  topBtn.disabled = topN===0;
  topBtn.style.opacity = topN===0?0.4:1;
  const useTop = adoptScope==='top' && topN>0;
  const cards = useTop ? (d.cards_top||[]) : d.cards;
  if(useTop){
    document.getElementById('archMeta').textContent=` 上位 distinct builds=${topN} / 上位デッキ=${d.top_total_decks}`;
    note.textContent=`rank≤${d.top_rank} のデッキだけに絞った採用率。標本が薄いので参考値。`;
  }else{
    document.getElementById('archMeta').textContent=` distinct builds=${d.distinct_builds} / 総デッキ=${d.total_decks}`;
    note.textContent = topN===0 ? '上位(rank≤100)のデッキがこのアーキタイプには無いため上位ビューは非表示。' : '';
  }
  const rows=cards.slice(0,40);
  const el=document.getElementById('adoptBars'); el.innerHTML='';
  rows.forEach(c=>{
    const nm=nameOf(c.card_id, c.name);
    const dv=document.createElement('div'); dv.className='bar-row';
    const col=c.adoption_pct>=50?'var(--seq)':'var(--seq-soft)';
    dv.innerHTML=`<div class="bar-label" title="${nm}">${nm}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${c.adoption_pct}%;background:${col}"></div></div>
      <div class="bar-val">${c.adoption_pct}%</div>`;
    const tr=dv.querySelector('.bar-track');
    tr.addEventListener('mousemove',e=>showTip(`<b>${nm}</b><br>採用 ${c.adoption_pct}% ・ 平均 ${c.avg_copies}枚`,e));
    tr.addEventListener('mouseleave',hideTip);
    el.appendChild(dv);
  });
}
document.querySelectorAll('#adoptScopeToggle button').forEach(b=>b.addEventListener('click',()=>{
  if(b.disabled) return;
  document.querySelectorAll('#adoptScopeToggle button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); adoptScope=b.dataset.scope; renderAdopt(archSel.value);
}));
archSel.addEventListener('change',()=>renderAdopt(archSel.value));
renderAdopt(archSel.value);

// ---- top decks card-count matrix ----
const BASIC_ENERGY=new Set([1,2,3,4,5,6,7,8]);
const deckArchSel=document.getElementById('deckArchSel');
(function(){
  const arches=Object.keys(DATA.top_decks||{})
    .sort((a,b)=>(DATA.card_adoption[b]?.distinct_builds||0)-(DATA.card_adoption[a]?.distinct_builds||0));
  arches.forEach(a=>{const o=document.createElement('option');o.value=a;o.textContent=archName(a);deckArchSel.appendChild(o);});
})();
function refreshDeckArchOptions(){ [...deckArchSel.options].forEach(o=>{o.textContent=archName(o.value);}); }
function renderMatrix(a){
  const decks=(DATA.top_decks||{})[a]; const host=document.getElementById('deckMatrix');
  const meta=document.getElementById('deckArchMeta');
  host.innerHTML='';
  if(!decks || !decks.length){ meta.textContent=' 上位デッキなし'; return; }
  meta.textContent=` ${decks.length}件の上位デッキ`;
  // 各デッキの card_id -> count
  const maps=decks.map(dk=>{const m=new Map();dk.cards.forEach(c=>m.set(c.card_id,c.count));return m;});
  // カード集合。基本エネルギー以外を先に(全デッキ合計枚数の降順)、基本エネは末尾。
  const totals=new Map();
  decks.forEach(dk=>dk.cards.forEach(c=>totals.set(c.card_id,(totals.get(c.card_id)||0)+c.count)));
  const ids=[...totals.keys()];
  const rank=id=>BASIC_ENERGY.has(id)?1:0;
  ids.sort((x,y)=> rank(x)-rank(y) || totals.get(y)-totals.get(x) || x-y);
  // テーブル構築
  const tbl=document.createElement('table'); tbl.className='matrix';
  const thead=document.createElement('thead'); const htr=document.createElement('tr');
  const corner=document.createElement('th'); corner.className='cardname'; corner.textContent='カード'; htr.appendChild(corner);
  decks.forEach((dk,i)=>{const th=document.createElement('th');
    const team=dk.team?String(dk.team):'—';
    th.innerHTML=`#${dk.rank}<br><span class="deckhead" title="${team}">${team.length>10?team.slice(0,10)+'…':team}</span>`;
    htr.appendChild(th);});
  thead.appendChild(htr); tbl.appendChild(thead);
  const tbody=document.createElement('tbody');
  ids.forEach(id=>{
    const tr=document.createElement('tr');
    const nm=nameOf(id, String(id));
    const c0=document.createElement('td'); c0.className='cardname'; c0.title=nm; c0.textContent=nm; tr.appendChild(c0);
    maps.forEach(m=>{const td=document.createElement('td');const v=m.get(id)||0;
      td.textContent=v||'·'; td.className=v?'hit':'n0'; tr.appendChild(td);});
    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody); host.appendChild(tbl);
}
deckArchSel.addEventListener('change',()=>renderMatrix(deckArchSel.value));
if(deckArchSel.options.length) renderMatrix(deckArchSel.value);

// ---- language toggle (JP default, EN switchable) ----
document.querySelectorAll('#langToggle button').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('#langToggle button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); LANG=b.dataset.lang;
  renderShare(shareKind); renderPie(); renderDiv();
  refreshArchOptions(); renderOther(); renderAdopt(archSel.value);
  refreshDeckArchOptions(); if(deckArchSel.options.length) renderMatrix(deckArchSel.value);
}));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
