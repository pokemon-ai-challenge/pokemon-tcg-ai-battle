"use strict";
const $ = (id) => document.getElementById(id);

const S = {
  cards: {}, attacks: {},
  replay: null, frames: [], meta: {}, name: "",
  idx: 0, timer: null,
  showImages: true, showOppHand: true, showPrize: true,
  lang: "jp",
};

// AreaType ints (from cg.api)
const AREA = { DECK: 1, HAND: 2, DISCARD: 3, ACTIVE: 4, BENCH: 5, PRIZE: 6, STADIUM: 7 };

const CONTEXT_JP = {
  Main: "メインフェーズ", SetupActivePokemon: "バトル場を選ぶ", SetupBenchPokemon: "ベンチを選ぶ",
  Switch: "入れ替え先を選ぶ", ToActive: "バトル場へ", ToBench: "ベンチへ", ToField: "場に出す",
  ToHand: "手札に加える", Discard: "トラッシュする", ToDeck: "山札に戻す", DamageCounter: "ダメカンを乗せる",
  DamageCounterAny: "ダメカンを好きに乗せる", Damage: "ダメージを与える", Heal: "回復する",
  EvolvesFrom: "進化元を選ぶ", EvolvesTo: "進化先を選ぶ", AttachFrom: "付ける先を選ぶ", AttachTo: "付けるカードを選ぶ",
  Attack: "ワザを選ぶ", Evolve: "進化", IsFirst: "先攻/後攻", Mulligan: "マリガン", Activate: "効果を使う?",
  CoinHead: "コイン", DrawCount: "引く枚数", DamageCounterCount: "ダメカン数", ToHandEnergy: "エネを手札へ",
};

// ---------------------------------------------------------------- helpers
function cardName(id) {
  const c = S.cards[id];
  if (!c) return "ID:" + id;
  if (S.lang === "jp") return c.nameJp || c.name || ("ID:" + id);
  return c.name || c.nameJp || ("ID:" + id);
}
function contextLabel(ctx) {
  return (S.lang === "jp" && CONTEXT_JP[ctx]) ? CONTEXT_JP[ctx] : ctx;
}
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

// ---------------------------------------------------------------- card DOM
function fallbackCard(id) {
  const d = document.createElement("div");
  d.className = "fallback";
  d.textContent = cardName(id);
  return d;
}
function cardEl(id, cls) {
  const wrap = document.createElement("div");
  wrap.className = "card" + (cls ? " " + cls : "");
  if (S.showImages) {
    const img = document.createElement("img");
    img.src = "/cards/" + id;
    img.alt = cardName(id);
    img.addEventListener("error", () => { img.remove(); wrap.appendChild(fallbackCard(id)); });
    wrap.appendChild(img);
  } else {
    wrap.appendChild(fallbackCard(id));
  }
  wrap.title = cardName(id);
  wrap.addEventListener("mouseenter", (e) => showPop(id, e));
  wrap.addEventListener("mousemove", movePop);
  wrap.addEventListener("mouseleave", hidePop);
  return wrap;
}
function monEl(poke, big) {
  const m = document.createElement("div");
  m.className = "mon" + (big ? " big" : "");
  const hp = document.createElement("div");
  hp.className = "hp";
  hp.textContent = `${poke.hp}/${poke.maxHp}`;
  m.appendChild(hp);
  m.appendChild(cardEl(poke.id));
  const en = document.createElement("div");
  en.className = "energy";
  const n = (poke.energies || []).length;
  for (let i = 0; i < Math.min(n, 6); i++) {
    const e = document.createElement("div"); e.className = "e"; en.appendChild(e);
  }
  m.appendChild(en);
  return m;
}

// hover zoom popup
function showPop(id, e) {
  if (!S.showImages) return;
  const pop = $("card-pop"), img = $("card-pop-img");
  img.src = "/cards/" + id;
  pop.hidden = false;
  movePop(e);
}
function movePop(e) {
  const pop = $("card-pop");
  if (pop.hidden) return;
  const x = Math.min(e.clientX + 16, window.innerWidth - 280);
  const y = Math.min(e.clientY + 16, window.innerHeight - 380);
  pop.style.left = x + "px";
  pop.style.top = y + "px";
}
function hidePop() { $("card-pop").hidden = true; }

// ---------------------------------------------------------------- resolve
function cardIdAt(state, sel, pi, area, index) {
  if (area === AREA.DECK && sel && sel.deck) {
    const it = sel.deck[index == null ? 0 : index];
    return it ? it.id : null;
  }
  const p = state.players[pi];
  if (!p) return null;
  let list = null;
  if (area === AREA.HAND) list = p.hand;
  else if (area === AREA.ACTIVE) list = p.active;
  else if (area === AREA.BENCH) list = p.bench;
  else if (area === AREA.DISCARD) list = p.discard;
  else if (area === AREA.PRIZE) list = p.prize;
  if (!list) return null;
  const item = list[index == null ? 0 : index];
  return item ? item.id : null;
}
function nm(state, sel, pi, area, index) {
  const id = cardIdAt(state, sel, pi, area, index);
  return id == null ? "?" : cardName(id);
}

function describeOption(opt, state, actor, sel) {
  const pi = (opt.playerIndex == null) ? actor : opt.playerIndex;
  switch (opt.type) {
    case "Play": return "手札から出す: " + nm(state, sel, actor, AREA.HAND, opt.index);
    case "Attach":
      return nm(state, sel, actor, opt.area, opt.index) + " を " +
             nm(state, sel, actor, opt.inPlayArea, opt.inPlayIndex) + " につける";
    case "Evolve":
      return nm(state, sel, actor, opt.area, opt.index) + " に進化（" +
             nm(state, sel, actor, opt.inPlayArea, opt.inPlayIndex) + "）";
    case "Ability": return "特性を使う: " + nm(state, sel, actor, opt.area, opt.index);
    case "Attack": {
      const a = S.attacks[opt.attackId];
      return "ワザ: " + (a ? a.name + (a.damage ? "（" + a.damage + "）" : "") : "attack " + opt.attackId);
    }
    case "Retreat": return "にげる";
    case "End": return "ターン終了";
    case "Card": return "選ぶ: " + nm(state, sel, pi, opt.area, opt.index);
    case "ToolCard": return "道具を選ぶ: " + nm(state, sel, pi, opt.area, opt.index);
    case "EnergyCard": return "エネルギーを選ぶ: " + nm(state, sel, pi, opt.area, opt.index);
    case "Energy": return "エネルギーを選ぶ";
    case "Yes": return "はい";
    case "No": return "いいえ";
    case "Number": return "数を選ぶ: " + opt.number;
    case "SpecialCondition": return "特殊状態: " + opt.specialConditionType;
    default: return opt.type + " " + JSON.stringify(opt);
  }
}

// ---------------------------------------------------------------- render
function renderHand(elId, hand, reveal) {
  const el = $(elId);
  el.innerHTML = "";
  const list = hand || [];
  if (!reveal) {
    for (let i = 0; i < list.length; i++) {
      const b = document.createElement("div"); b.className = "card hand"; b.appendChild(fallbackHidden()); el.appendChild(b);
    }
    return;
  }
  list.forEach((c) => el.appendChild(cardEl(c.id, "hand")));
}
function fallbackHidden() {
  const d = document.createElement("div"); d.className = "fallback"; d.textContent = "🂠"; return d;
}
function renderPrize(elId, prize) {
  const el = $(elId); el.innerHTML = "";
  (prize || []).forEach((c) => {
    const p = document.createElement("div"); p.className = "p";
    if (S.showPrize && c) {
      const img = document.createElement("img"); img.src = "/cards/" + c.id;
      img.addEventListener("error", () => img.remove());
      p.appendChild(img);
    }
    el.appendChild(p);
  });
}
function renderSide(prefix, player, isActor) {
  $(prefix + "-deck").textContent = player.deckCount;
  $(prefix + "-discard").textContent = (player.discard || []).length;
  const active = $(prefix + "-active"); active.innerHTML = "";
  if (player.active && player.active[0]) active.appendChild(monEl(player.active[0], true));
  const bench = $(prefix + "-bench"); bench.innerHTML = "";
  (player.bench || []).forEach((b) => bench.appendChild(monEl(b, false)));
  renderPrize(prefix + "-prize", player.prize);
  $("field-" + prefix).classList.toggle("acting", isActor);
}

function render() {
  if (!S.frames.length) return;
  S.idx = Math.max(0, Math.min(S.idx, S.frames.length - 1));
  const fr = S.frames[S.idx];
  const st = fr.current;
  const sel = fr.select;
  const actor = st.yourIndex;

  renderSide("opp", st.players[1], actor === 1);
  renderSide("me", st.players[0], actor === 0);
  renderHand("opp-hand", st.players[1].hand, S.showOppHand);
  renderHand("me-hand", st.players[0].hand, true);
  $("opp-hand-count").textContent = st.players[1].handCount;
  $("me-hand-count").textContent = st.players[0].handCount;

  // first/result badges
  setBadge("opp-first", st.firstPlayer === 1);
  setBadge("me-first", st.firstPlayer === 0);
  renderResultBadge(st.result);

  // stadium
  $("stadium-card").textContent = (st.stadium && st.stadium[0]) ? cardName(st.stadium[0].id) : "なし";

  // center
  $("turn-n").textContent = "ターン " + st.turn;
  $("turn-phase").textContent = sel ? contextLabel(sel.context) : "-";

  // summary
  const sm = $("summary");
  sm.innerHTML = "";
  const rows = [
    ["リプレイ", S.name],
    ["相手", S.meta.opponent || "-"],
    ["勝者", st.result === -1 ? "対戦中" : "Player " + st.result],
    ["ターン", st.turn],
    ["手番", "Player " + actor],
    ["先攻プレイヤー", st.firstPlayer < 0 ? "-" : "Player " + st.firstPlayer],
    ["コンテキスト", sel ? contextLabel(sel.context) : "-"],
    ["選択インデックス", JSON.stringify(fr.selected || [])],
  ];
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    sm.appendChild(dt); sm.appendChild(dd);
  }

  // chosen action
  const chosen = $("chosen");
  const selected = fr.selected || [];
  if (!sel || selected.length === 0) {
    chosen.textContent = sel ? "（選択なし / パス）" : "-";
  } else {
    chosen.textContent = selected.map((i) =>
      `${i}: ${sel.option[i] ? describeOption(sel.option[i], st, actor, sel) : "?"}`).join(" ／ ");
  }

  // options list (fixed-size scroll)
  const optEl = $("options");
  optEl.innerHTML = "";
  if (sel) {
    sel.option.forEach((o, i) => {
      const row = document.createElement("div");
      row.className = "opt" + (selected.includes(i) ? " selected" : "");
      const idx = document.createElement("span"); idx.className = "idx"; idx.textContent = i;
      const txt = document.createElement("span"); txt.textContent = describeOption(o, st, actor, sel);
      row.appendChild(idx); row.appendChild(txt);
      optEl.appendChild(row);
    });
  }

  // log
  const log = $("log"); log.innerHTML = "";
  (fr.logs || []).forEach((entry) => {
    const row = document.createElement("div"); row.className = "row";
    row.textContent = JSON.stringify(entry);
    log.appendChild(row);
  });

  // transport
  $("scrubber").max = S.frames.length - 1;
  $("scrubber").value = S.idx;
  $("frame-info").textContent = `${S.idx + 1} / ${S.frames.length}`;
  $("btn-prev").disabled = S.idx === 0;
  $("btn-next").disabled = S.idx === S.frames.length - 1;
}

function setBadge(id, on) { $(id).hidden = !on; }
function renderResultBadge(result) {
  for (const [pi, id] of [[0, "me-result"], [1, "opp-result"]]) {
    const el = $(id);
    if (result === -1 || result == null) { el.hidden = true; continue; }
    el.hidden = false;
    const win = result === pi;
    el.textContent = win ? "勝ち" : (result === 2 ? "引分" : "負け");
    el.classList.toggle("win", win);
    el.classList.toggle("lose", !win && result !== 2);
  }
}

// ---------------------------------------------------------------- playback
function go(i) { stopAuto(); S.idx = i; render(); }
function step(d) { go(Math.max(0, Math.min(S.frames.length - 1, S.idx + d))); }
function toggleAuto() {
  if (S.timer) { stopAuto(); return; }
  $("btn-auto").textContent = "停止";
  S.timer = setInterval(() => {
    if (S.idx >= S.frames.length - 1) { stopAuto(); return; }
    S.idx++; render();
  }, 550);
}
function stopAuto() { if (S.timer) { clearInterval(S.timer); S.timer = null; } $("btn-auto").textContent = "自動再生"; }

// ---------------------------------------------------------------- load
async function loadReplays(selectLatest = true) {
  const list = await getJSON("/api/replays");
  const sel = $("replay-select");
  sel.innerHTML = "";
  list.forEach((r, i) => {
    const o = document.createElement("option");
    o.value = r.name;
    const fc = r.meta && r.meta.frameCount ? ` · ${r.meta.frameCount}手` : "";
    o.textContent = r.name + (i === 0 ? " (latest)" : "") + fc;
    sel.appendChild(o);
  });
  if (list.length && selectLatest) { sel.value = list[0].name; await loadReplay(list[0].name); }
  else if (!list.length) { $("frame-info").textContent = "リプレイなし"; }
}
async function loadReplay(name) {
  stopAuto();
  const data = await getJSON("/api/replay?name=" + encodeURIComponent(name));
  S.name = name; S.meta = data.meta || {}; S.frames = data.frames || []; S.idx = 0;
  render();
}

// ---------------------------------------------------------------- new match
function fillSelect(el, items, mk) {
  el.innerHTML = "";
  items.forEach((it) => {
    const [value, label, disabled] = mk(it);
    const o = document.createElement("option");
    o.value = value; o.textContent = label; o.disabled = !!disabled;
    el.appendChild(o);
  });
}
function setVal(id, v) {
  const el = $(id);
  if ([...el.options].some((o) => o.value === v && !o.disabled)) el.value = v;
}
async function loadMatchOptions() {
  const [agentsList, decksList] = await Promise.all([getJSON("/api/agents"), getJSON("/api/decks")]);
  const mkAgent = (a) => [a.id, a.label + (a.slow ? " (遅い)" : ""), false];
  const mkDeck = (d) => [d.id, d.label + (d.valid ? "" : ` ⚠${d.count}枚`), !d.valid];
  fillSelect($("nm-p0-ai"), agentsList, mkAgent);
  fillSelect($("nm-p1-ai"), agentsList, mkAgent);
  fillSelect($("nm-p0-deck"), decksList, mkDeck);
  fillSelect($("nm-p1-deck"), decksList, mkDeck);
  setVal("nm-p0-ai", "mppo");
  setVal("nm-p1-ai", "random");
}
async function runMatch() {
  const body = {
    p0: $("nm-p0-ai").value, p1: $("nm-p1-ai").value,
    deck0: $("nm-p0-deck").value, deck1: $("nm-p1-deck").value,
  };
  const status = $("nm-status"), btn = $("nm-run");
  status.className = "nm-status busy";
  status.textContent = "対戦を実行中…（モンテカルロ等の遅いAIは時間がかかります）";
  btn.disabled = true;
  try {
    const r = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || ("HTTP " + r.status));
    status.className = "nm-status ok";
    status.textContent = `完了: ${data.name}（${data.meta.frameCount}手 / 勝者 Player ${data.meta.result}）`;
    await loadReplays(false);
    setVal("replay-select", data.name);
    await loadReplay(data.name);
  } catch (e) {
    status.className = "nm-status err";
    status.textContent = "失敗: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function init() {
  const cardsData = await getJSON("/api/cards");
  S.cards = cardsData.cards; S.attacks = cardsData.attacks;
  await loadMatchOptions();
  $("nm-run").addEventListener("click", runMatch);

  $("btn-prev").addEventListener("click", () => step(-1));
  $("btn-next").addEventListener("click", () => step(1));
  $("btn-auto").addEventListener("click", toggleAuto);
  $("scrubber").addEventListener("input", (e) => go(+e.target.value));
  $("btn-reload").addEventListener("click", () => loadReplays(false));
  $("replay-select").addEventListener("change", (e) => loadReplay(e.target.value));
  $("btn-lang").addEventListener("click", () => {
    S.lang = S.lang === "jp" ? "en" : "jp";
    $("btn-lang").textContent = S.lang === "jp" ? "EN" : "日本語";
    render();
  });
  $("tg-images").addEventListener("change", (e) => { S.showImages = e.target.checked; render(); });
  $("tg-opphand").addEventListener("change", (e) => { S.showOppHand = e.target.checked; render(); });
  $("tg-prize").addEventListener("change", (e) => { S.showPrize = e.target.checked; render(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
    else if (e.key === " ") { e.preventDefault(); toggleAuto(); }
  });

  await loadReplays(true);
}

init().catch((e) => { $("frame-info").textContent = "エラー: " + e.message; console.error(e); });
