"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const S = {
  meta: null,
  mode: "ai",          // "ai" | "human"
  replay: null,        // AI vs AI: array of views
  idx: 0,
  playing: false,
  timer: null,
  picked: [],          // human selection in progress
};

const ENERGY = {
  0: ["無", "e0"], 1: ["草", "e1"], 2: ["炎", "e2"], 3: ["水", "e3"],
  4: ["雷", "e4"], 5: ["超", "e5"], 6: ["闘", "e6"], 7: ["悪", "e7"],
  8: ["鋼", "e8"], 9: ["竜", "e9"], 10: ["虹", "e10"], 11: ["R", "e11"],
};
const STAGE_JP = { basic: "たね", stage1: "1進化", stage2: "2進化" };

// ---------------------------------------------------------------------------
// Tiny DOM helper
// ---------------------------------------------------------------------------
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function $(id) { return document.getElementById(id); }
async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  const data = await r.json();
  if (!r.ok || data.error) throw new Error(data.error || ("HTTP " + r.status));
  return data;
}
function showLoading(text) { $("loading-text").textContent = text || "実行中…"; $("loading").hidden = false; }
function hideLoading() { $("loading").hidden = true; }

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  try {
    S.meta = await api("/api/meta");
  } catch (e) {
    alert("メタ情報の取得に失敗しました: " + e.message);
    return;
  }
  populateSelects();
  wireEvents();
}

function agentOptions(sel) {
  sel.innerHTML = "";
  for (const a of S.meta.agents) {
    const o = el("option", null, a.label + (a.slow ? "（遅い）" : ""));
    o.value = a.id;
    sel.appendChild(o);
  }
}
function deckOptions(sel) {
  sel.innerHTML = "";
  for (const d of S.meta.decks) {
    const o = el("option", null, d.label + (d.valid ? "" : "（無効:" + d.count + "枚）"));
    o.value = d.id;
    if (!d.valid) o.disabled = true;
    sel.appendChild(o);
  }
}
function populateSelects() {
  ["ai0", "ai1", "human-ai"].forEach((id) => agentOptions($(id)));
  ["deck0", "deck1", "human-deck", "ai-deck"].forEach((id) => deckOptions($(id)));
  if ($("ai1").options.length > 1) $("ai1").selectedIndex = Math.min(1, $("ai1").options.length - 1);
}

function wireEvents() {
  document.querySelectorAll(".mode-tab").forEach((t) =>
    t.addEventListener("click", () => switchMode(t.dataset.mode)));
  $("btn-start-ai").addEventListener("click", startAiMatch);
  $("btn-start-human").addEventListener("click", startHuman);
  $("btn-home").addEventListener("click", goHome);
  $("btn-submit").addEventListener("click", submitHuman);

  $("replay-controls").querySelectorAll("button[data-act]").forEach((b) =>
    b.addEventListener("click", () => replayControl(b.dataset.act)));
  $("scrubber").addEventListener("input", (e) => { stopPlay(); S.idx = +e.target.value; renderReplay(); });
}

function switchMode(mode) {
  S.mode = mode;
  document.querySelectorAll(".mode-tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
  $("setup-ai").hidden = mode !== "ai";
  $("setup-human").hidden = mode !== "human";
}

function goHome() {
  stopPlay();
  $("game").hidden = true;
  $("setup").hidden = false;
  $("btn-home").hidden = true;
}

// ---------------------------------------------------------------------------
// AI vs AI
// ---------------------------------------------------------------------------
async function startAiMatch() {
  const body = { ai0: $("ai0").value, deck0: $("deck0").value, ai1: $("ai1").value, deck1: $("deck1").value };
  showLoading("対戦を実行中…（AIによっては時間がかかります）");
  try {
    const out = await api("/api/match", body);
    S.replay = out.replay;
    S.idx = 0;
    enterGame();
    $("replay-controls").hidden = false;
    $("human-controls").hidden = true;
    $("ai-moves").hidden = true;
    $("scrubber").max = S.replay.length - 1;
    renderReplay();
  } catch (e) {
    alert("対戦の実行に失敗しました: " + e.message);
  } finally {
    hideLoading();
  }
}

function renderReplay() {
  const v = S.replay[S.idx];
  $("scrubber").value = S.idx;
  $("step-label").textContent = (S.idx + 1) + " / " + S.replay.length;
  renderView(v, { interactive: false });
}

function replayControl(act) {
  if (act === "play") { S.playing ? stopPlay() : startPlay(); return; }
  stopPlay();
  if (act === "first") S.idx = 0;
  else if (act === "last") S.idx = S.replay.length - 1;
  else if (act === "prev") S.idx = Math.max(0, S.idx - 1);
  else if (act === "next") S.idx = Math.min(S.replay.length - 1, S.idx + 1);
  renderReplay();
}
function startPlay() {
  S.playing = true;
  $("btn-play").textContent = "⏸ 停止";
  const tick = () => {
    if (!S.playing) return;
    if (S.idx >= S.replay.length - 1) { stopPlay(); return; }
    S.idx++; renderReplay();
    S.timer = setTimeout(tick, +$("speed").value);
  };
  S.timer = setTimeout(tick, +$("speed").value);
}
function stopPlay() {
  S.playing = false;
  if (S.timer) { clearTimeout(S.timer); S.timer = null; }
  $("btn-play").textContent = "▶ 再生";
}

// ---------------------------------------------------------------------------
// Human vs AI
// ---------------------------------------------------------------------------
async function startHuman() {
  const body = {
    humanSide: +$("human-side").value,
    ai: $("human-ai").value,
    humanDeck: $("human-deck").value,
    aiDeck: $("ai-deck").value,
  };
  showLoading("対戦を準備中…");
  try {
    const view = await api("/api/human/new", body);
    enterGame();
    $("replay-controls").hidden = true;
    renderView(view, { interactive: true });
  } catch (e) {
    alert("対戦の開始に失敗しました: " + e.message);
  } finally {
    hideLoading();
  }
}

async function submitHuman() {
  const action = S.picked.slice();
  showLoading("AIが考えています…");
  try {
    const view = await api("/api/human/act", { action });
    renderView(view, { interactive: true });
  } catch (e) {
    alert("送信に失敗しました: " + e.message);
  } finally {
    hideLoading();
  }
}

// ---------------------------------------------------------------------------
// Shared rendering
// ---------------------------------------------------------------------------
function enterGame() {
  $("setup").hidden = true;
  $("game").hidden = false;
  $("btn-home").hidden = false;
}

function renderView(v, opts) {
  const bottom = v.bottomIndex ?? 0;
  const top = 1 - bottom;
  $("player-top").replaceChildren(renderPlayer(v, top, true));
  $("player-bottom").replaceChildren(renderPlayer(v, bottom, false));
  renderMidline(v);
  renderResult(v);
  renderOptions(v, opts);
  renderAiMoves(v);
  renderLogs(v);
}

function renderMidline(v) {
  const slot = $("stadium-slot");
  slot.replaceChildren();
  if (v.stadium) {
    const c = el("div", "stadium-card", "🏟 " + v.stadium.name);
    slot.appendChild(c);
  } else {
    slot.appendChild(el("span", null, "スタジアム なし"));
  }
  const ti = $("turn-info");
  ti.replaceChildren();
  ti.appendChild(el("span", null, "ターン "));
  const tb = el("b", null, String(v.turn)); ti.appendChild(tb);
  if (v.actingIndex != null) {
    ti.appendChild(el("span", null, "　手番: "));
    ti.appendChild(el("b", null, v.players[v.actingIndex].label));
  }
}

function renderResult(v) {
  const banner = $("result-banner");
  if (v.result === -1 || v.result == null) { banner.hidden = true; return; }
  banner.hidden = false;
  if (v.result === 2) banner.textContent = "引き分け";
  else banner.textContent = "勝者: " + v.players[v.result].label + " 🏆";
}

function renderPlayer(v, idx, isOpp) {
  const p = v.players[idx];
  const area = el("div", "player-area " + (isOpp ? "opponent" : "self"));

  const header = el("div", "area-header" + (v.actingIndex === idx ? " acting" : ""));
  header.appendChild(el("span", "pname", p.label + (v.actingIndex === idx ? " ◀手番" : "")));
  // prize pips
  const prizes = el("span", "prizes");
  const taken = 6 - p.prizeTotal;
  for (let i = 0; i < 6; i++) {
    prizes.appendChild(el("span", "prize-pip" + (i < taken ? " taken" : "")));
  }
  const pwrap = el("span", "chip");
  pwrap.append("サイド残 " + p.prizeTotal + " ");
  pwrap.appendChild(prizes);
  header.appendChild(pwrap);
  header.appendChild(el("span", "chip", "山札 " + p.deckCount));
  header.appendChild(el("span", "chip", "トラッシュ " + p.discardCount));
  header.appendChild(el("span", "chip", "手札 " + p.handCount));

  const bench = el("div", "row bench");
  if (!p.bench.length) bench.appendChild(emptyPoke("ベンチなし"));
  p.bench.forEach((b) => bench.appendChild(renderPoke(b, false)));

  const activeRow = el("div", "row active-row");
  activeRow.appendChild(p.active ? renderPoke(p.active, true) : emptyPoke("バトル場 空"));

  // hand bar (shown when available)
  const hand = renderHand(p);

  if (isOpp) {
    area.append(header, hand, bench, activeRow);
  } else {
    area.append(activeRow, bench, hand, header);
  }
  return area;
}

function renderHand(p) {
  const bar = el("div", "handbar");
  if (p.hand == null) {
    bar.appendChild(el("span", "hand-label", "手札"));
    for (let i = 0; i < Math.min(p.handCount, 10); i++) bar.appendChild(el("span", "hand-card back", "・"));
    return bar;
  }
  bar.appendChild(el("span", "hand-label", "手札"));
  p.hand.forEach((c) => bar.appendChild(el("span", "hand-card", c.name)));
  return bar;
}

function emptyPoke(text) { return el("div", "poke empty", text); }

function renderPoke(poke, isActive) {
  const c = el("div", "poke" + (isActive ? " active" : ""));
  // badges
  const badges = el("div", "badges");
  badges.appendChild(el("span", "badge stage", STAGE_JP[poke.stage] || ""));
  if (poke.megaEx) badges.appendChild(el("span", "badge mega", "M-ex"));
  else if (poke.ex) badges.appendChild(el("span", "badge ex", "ex"));
  if (poke.appearThisTurn) badges.appendChild(el("span", "badge new", "NEW"));
  c.appendChild(badges);

  c.appendChild(el("div", "pname", poke.name));

  // HP
  const hpRow = el("div", "hp-row");
  const bar = el("div", "hp-bar");
  const fill = el("div", "hp-fill");
  const ratio = poke.maxHp ? Math.max(0, Math.min(1, poke.hp / poke.maxHp)) : 0;
  fill.style.width = (ratio * 100) + "%";
  fill.style.background = ratio > 0.5 ? "var(--hp-hi)" : ratio > 0.2 ? "var(--hp-mid)" : "var(--hp-lo)";
  bar.appendChild(fill);
  hpRow.appendChild(bar);
  hpRow.appendChild(el("span", "hp-text", poke.hp + "/" + poke.maxHp));
  c.appendChild(hpRow);

  // energies
  const en = el("div", "energies");
  (poke.energies || []).forEach((e) => {
    const [letter, cls] = ENERGY[e] || ["?", "e0"];
    const pip = el("span", "epip " + cls, letter);
    pip.style.background = "var(--" + cls + ")";
    en.appendChild(pip);
  });
  c.appendChild(en);

  // tools
  if (poke.toolCount) {
    c.appendChild(el("div", "tool-badge", "🔧 " + poke.tools.join(", ")));
  }

  // status (active only meaningful but render anyway if present on poke owner)
  return c;
}

// ---------------------------------------------------------------------------
// Options panel
// ---------------------------------------------------------------------------
function renderOptions(v, opts) {
  const interactive = opts && opts.interactive;
  const list = $("options-list");
  list.replaceChildren();
  const title = $("options-title");
  const sel = v.select;
  title.textContent = sel ? "選択肢 — " + sel.contextLabel : "選択肢";

  // status badges of acting player's active (show conditions near options)
  const yourTurn = interactive && v.yourTurn;
  S.picked = [];

  if (!v.options || !v.options.length) {
    list.appendChild(el("div", "select-hint", v.finished ? "対戦終了" : "—"));
    $("human-controls").hidden = true;
    return;
  }

  v.options.forEach((o) => {
    const row = el("div", "option" + (o.selected ? " selected" : "") + (yourTurn ? " clickable" : ""));
    row.appendChild(el("span", "otag", kindTag(o.kind)));
    row.appendChild(el("span", "olabel", o.label));
    if (o.selected && !interactive) row.appendChild(el("span", "check", "✓ 選択"));
    if (yourTurn) {
      row.dataset.idx = o.idx;
      row.addEventListener("click", () => togglePick(v, o.idx, row));
    }
    list.appendChild(row);
  });

  // human controls
  const hc = $("human-controls");
  if (yourTurn) {
    hc.hidden = false;
    updateSelectHint(sel);
    $("btn-submit").disabled = !pickValid(sel);
    // single-choice convenience handled in togglePick
  } else {
    hc.hidden = true;
  }
}

function kindTag(kind) {
  const m = {
    PLAY: "出す", ATTACH: "つける", EVOLVE: "進化", ABILITY: "特性",
    ATTACK: "ワザ", RETREAT: "にげる", END: "終了", YES: "はい", NO: "いいえ",
    CARD: "選択", ENERGY: "エネ", ENERGY_CARD: "エネ", TOOL_CARD: "どうぐ",
    NUMBER: "数", SKILL: "効果", SPECIAL_CONDITION: "状態",
  };
  return m[kind] || kind;
}

function togglePick(v, idx, row) {
  const sel = v.select;
  const single = sel.minCount === 1 && sel.maxCount === 1;
  if (single) {
    S.picked = [idx];
    submitHuman();
    return;
  }
  const at = S.picked.indexOf(idx);
  if (at >= 0) { S.picked.splice(at, 1); row.classList.remove("picked"); }
  else {
    if (S.picked.length >= sel.maxCount) return;
    S.picked.push(idx); row.classList.add("picked");
  }
  updateSelectHint(sel);
  $("btn-submit").disabled = !pickValid(sel);
}

function pickValid(sel) {
  return S.picked.length >= sel.minCount && S.picked.length <= sel.maxCount;
}
function updateSelectHint(sel) {
  const h = $("select-hint");
  if (sel.minCount === sel.maxCount) h.textContent = `${sel.maxCount}個選択（${S.picked.length}）`;
  else h.textContent = `${sel.minCount}〜${sel.maxCount}個選択（${S.picked.length}）`;
  if (sel.minCount === 0) $("btn-submit").textContent = S.picked.length ? "決定" : "選ばない";
  else $("btn-submit").textContent = "決定";
}

// ---------------------------------------------------------------------------
// AI moves (human mode) + logs
// ---------------------------------------------------------------------------
function renderAiMoves(v) {
  const panel = $("ai-moves");
  if (!v.aiMoves) { panel.hidden = true; return; }
  panel.hidden = false;
  const list = $("ai-moves-list");
  list.replaceChildren();
  if (!v.aiMoves.length) { list.appendChild(el("div", "ai-move", "（直近の相手の行動はありません）")); return; }
  v.aiMoves.forEach((m) => {
    const box = el("div", "ai-move");
    box.appendChild(el("div", "am-ctx", "T" + m.turn + "・" + m.context));
    const chosen = (m.options || []).filter((o) => o.selected).map((o) => o.label).join(" / ") || "—";
    box.appendChild(el("div", "am-chosen", "▶ " + chosen));
    const others = (m.options || []).filter((o) => !o.selected).map((o) => o.label);
    if (others.length) box.appendChild(el("div", "am-opts", "候補: " + others.join("、")));
    list.appendChild(box);
  });
}

function renderLogs(v) {
  const list = $("log-list");
  list.replaceChildren();
  (v.logs || []).forEach((line) => {
    let cls = "log-line";
    if (line.startsWith("---")) cls += " turn";
    if (line.startsWith("===")) cls += " result";
    list.appendChild(el("div", cls, line));
  });
  list.scrollTop = list.scrollHeight;
}

init();
