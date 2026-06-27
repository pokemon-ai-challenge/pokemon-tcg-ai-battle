const replaySelect = document.getElementById("replaySelect");
const reloadButton = document.getElementById("reloadButton");
const prevButton = document.getElementById("prevButton");
const nextButton = document.getElementById("nextButton");
const autoplayButton = document.getElementById("autoplayButton");
const frameSlider = document.getElementById("frameSlider");
const frameMeta = document.getElementById("frameMeta");

let replayData = null;
let frameIndex = 0;
let autoplayTimer = null;

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

async function loadReplayList() {
  const replayFiles = await fetchJson("/api/replays");
  replaySelect.innerHTML = "";

  if (!replayFiles.length) {
    const option = document.createElement("option");
    option.textContent = "No replay files found";
    option.value = "";
    replaySelect.append(option);
    replaySelect.disabled = true;
    return;
  }

  replaySelect.disabled = false;
  replayFiles.forEach((file, index) => {
    const option = document.createElement("option");
    option.value = file.name;
    option.textContent = index === replayFiles.length - 1 ? `${file.name} (latest)` : file.name;
    replaySelect.append(option);
  });
  replaySelect.value = replayFiles[replayFiles.length - 1].name;
}

async function loadSelectedReplay() {
  if (!replaySelect.value) {
    return;
  }
  replayData = await fetchJson(`/api/replays/${encodeURIComponent(replaySelect.value)}`);
  frameIndex = 0;
  frameSlider.min = "0";
  frameSlider.max = String(Math.max((replayData.frames?.length || 1) - 1, 0));
  frameSlider.value = "0";
  render();
}

function render() {
  if (!replayData || !replayData.frames || !replayData.frames.length) {
    return;
  }

  const frame = replayData.frames[frameIndex];
  const visual = frame.visual;
  const current = visual.current || {};
  const players = current.players || [{}, {}];
  const actingPlayer = frame.actingPlayer ?? 0;

  frameMeta.textContent = `frame ${frameIndex + 1} / ${replayData.frames.length}`;
  document.getElementById("turnBadge").textContent = `Turn ${frame.turn ?? "?"}`;
  document.getElementById("contextBadge").textContent = frame.context || "Terminal";
  document.getElementById("handCountBadge").textContent = `${(players[0]?.hand || []).length} cards`;

  renderSummary(replayData.metadata, frame);
  renderAction(frame);
  renderOptions(frame);
  renderLogs(visual.logs || []);

  renderPlayer(players[1], {
    activeId: "opponentActive",
    benchId: "opponentBench",
    prizeId: "opponentPrize",
    deckCountId: "opponentDeckCount",
    discardCountId: "opponentDiscardCount",
  });
  renderPlayer(players[0], {
    activeId: "selfActive",
    benchId: "selfBench",
    prizeId: "selfPrize",
    deckCountId: "selfDeckCount",
    discardCountId: "selfDiscardCount",
  });
  renderHand(players[0]?.hand || [], frame.actionLabels || []);
}

function renderSummary(metadata, frame) {
  const summaryList = document.getElementById("summaryList");
  const entries = [
    ["replay", replaySelect.value],
    ["opponent", metadata.opponent],
    ["result", metadata.result],
    ["steps", metadata.steps],
    ["actingPlayer", frame.actingPlayer],
    ["selected", JSON.stringify(frame.action || [])],
  ];

  summaryList.innerHTML = entries
    .map(([key, value]) => `<dt>${escapeHtml(String(key))}</dt><dd>${escapeHtml(String(value))}</dd>`)
    .join("");
}

function renderAction(frame) {
  const chosenAction = document.getElementById("chosenAction");
  if (!frame.action || !frame.action.length) {
    chosenAction.textContent = "No action recorded for this frame.";
    return;
  }

  chosenAction.innerHTML = frame.actionLabels
    .map((label, index) => `<div>${frame.action[index]}: ${escapeHtml(label)}</div>`)
    .join("");
}

function renderOptions(frame) {
  const optionsList = document.getElementById("optionsList");
  const selected = new Set(frame.action || []);

  optionsList.innerHTML = (frame.options || [])
    .map((option) => {
      const className = selected.has(option.index) ? "selected-option" : "";
      return `<li class="${className}"><strong>${option.index}</strong> ${escapeHtml(option.label)}</li>`;
    })
    .join("");
}

function renderLogs(logs) {
  const logsList = document.getElementById("logsList");
  if (!logs.length) {
    logsList.innerHTML = `<div class="log-entry">No logs on this frame.</div>`;
    return;
  }
  logsList.innerHTML = logs
    .map((log) => `<div class="log-entry">${escapeHtml(JSON.stringify(log))}</div>`)
    .join("");
}

function renderPlayer(player, targets) {
  document.getElementById(targets.deckCountId).textContent = String(player?.deckCount ?? 0);
  document.getElementById(targets.discardCountId).textContent = String((player?.discard || []).length);

  renderCardList(document.getElementById(targets.activeId), player?.active || [], {
    active: true,
    emptyLabel: "No active Pokemon",
  });
  renderCardList(document.getElementById(targets.benchId), player?.bench || [], {
    active: false,
    emptyLabel: "No bench",
  });
  renderCardList(document.getElementById(targets.prizeId), player?.prize || [], {
    active: false,
    emptyLabel: "No prize",
    maxSlots: 6,
  });
}

function renderHand(cards, actionLabels) {
  const hand = document.getElementById("selfHand");
  renderCardList(hand, cards, {
    active: false,
    hand: true,
    emptyLabel: "No hand cards",
    highlightLabels: actionLabels,
  });
}

function renderCardList(container, cards, config) {
  const normalized = Array.isArray(cards) ? cards : [];
  if (!normalized.length) {
    container.innerHTML = `<div class="empty-slot">${escapeHtml(config.emptyLabel)}</div>`;
    return;
  }

  const html = normalized.map((card) => renderCard(card, config)).join("");
  if (config.maxSlots && normalized.length < config.maxSlots) {
    const empties = new Array(config.maxSlots - normalized.length)
      .fill(0)
      .map(() => `<div class="empty-slot">empty</div>`)
      .join("");
    container.innerHTML = html + empties;
    return;
  }
  container.innerHTML = html;
}

function renderCard(card, config) {
  if (!card) {
    return `<div class="empty-slot">hidden</div>`;
  }

  const classes = ["card-chip"];
  if (config.active) {
    classes.push("active-card");
  }
  if (config.hand) {
    classes.push("hand-card");
  }

  const tags = [];
  if (card.hp !== undefined && card.hp !== null) {
    tags.push(`HP ${card.hp}/${card.maxHp ?? card.hp}`);
  }
  if (Array.isArray(card.energies) && card.energies.length) {
    tags.push(`Energy ${card.energies.length}`);
  }
  if (Array.isArray(card.tools) && card.tools.length) {
    tags.push(`Tools ${card.tools.length}`);
  }
  if (Array.isArray(card.preEvolution) && card.preEvolution.length) {
    tags.push(`Prev ${card.preEvolution.length}`);
  }

  return `
    <article class="${classes.join(" ")}">
      <div class="card-name">${escapeHtml(card.name || `card #${card.id ?? "?"}`)}</div>
      <div class="card-meta">id=${escapeHtml(String(card.id ?? "?"))}</div>
      <div class="card-meta">serial=${escapeHtml(String(card.serial ?? "?"))}</div>
      <div class="card-tags">
        ${tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}
      </div>
    </article>
  `;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function stepFrame(delta) {
  if (!replayData) {
    return;
  }
  frameIndex = Math.min(Math.max(frameIndex + delta, 0), replayData.frames.length - 1);
  frameSlider.value = String(frameIndex);
  render();
}

function toggleAutoplay() {
  if (autoplayTimer) {
    window.clearInterval(autoplayTimer);
    autoplayTimer = null;
    autoplayButton.textContent = "Autoplay";
    return;
  }

  autoplayButton.textContent = "Pause";
  autoplayTimer = window.setInterval(() => {
    if (!replayData || frameIndex >= replayData.frames.length - 1) {
      toggleAutoplay();
      return;
    }
    stepFrame(1);
  }, 1200);
}

reloadButton.addEventListener("click", async () => {
  await loadReplayList();
  await loadSelectedReplay();
});

replaySelect.addEventListener("change", async () => {
  await loadSelectedReplay();
});

prevButton.addEventListener("click", () => stepFrame(-1));
nextButton.addEventListener("click", () => stepFrame(1));
autoplayButton.addEventListener("click", toggleAutoplay);
frameSlider.addEventListener("input", (event) => {
  frameIndex = Number(event.target.value);
  render();
});

async function boot() {
  await loadReplayList();
  await loadSelectedReplay();
}

boot().catch((error) => {
  frameMeta.textContent = error.message;
  console.error(error);
});
