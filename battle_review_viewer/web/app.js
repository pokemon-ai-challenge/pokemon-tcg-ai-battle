const modeSelect = document.getElementById("modeSelect");
const replaySelect = document.getElementById("replaySelect");
const liveCpuSelect = document.getElementById("liveCpuSelect");
const livePlayerDeckSelect = document.getElementById("livePlayerDeckSelect");
const liveOpponentDeckSelect = document.getElementById("liveOpponentDeckSelect");
const liveStartButton = document.getElementById("liveStartButton");
const reloadButton = document.getElementById("reloadButton");
const prevButton = document.getElementById("prevButton");
const nextButton = document.getElementById("nextButton");
const autoplayButton = document.getElementById("autoplayButton");
const speedSelect = document.getElementById("speedSelect");
const liveUndoButton = document.getElementById("liveUndoButton");
const liveConfirmButton = document.getElementById("liveConfirmButton");
const liveClearButton = document.getElementById("liveClearButton");
const liveStatusBox = document.getElementById("liveStatusBox");
const liveSelectionMeta = document.getElementById("liveSelectionMeta");
const frameSlider = document.getElementById("frameSlider");
const frameMeta = document.getElementById("frameMeta");
const toggleOpponentHand = document.getElementById("toggleOpponentHand");
const togglePrize = document.getElementById("togglePrize");
const toggleCardImages = document.getElementById("toggleCardImages");
const toggleAnimation = document.getElementById("toggleAnimation");
const drawer = document.getElementById("drawer");
const drawerTitle = document.getElementById("drawerTitle");
const drawerClose = document.getElementById("drawerClose");
const panelButtons = Array.from(document.querySelectorAll("[data-panel-btn]"));
const panelSections = Array.from(document.querySelectorAll(".panel[data-panel]"));
const cardDetailContent = document.getElementById("cardDetailContent");
const activeEffectsShelf = document.getElementById("activeEffectsShelf");

const SELF_INDEX = 0;
const OPPONENT_INDEX = 1;
const AREA = { DECK: 1, HAND: 2, DISCARD: 3, ACTIVE: 4, BENCH: 5, PRIZE: 6, STADIUM: 7 };
const AREA_ZONE = { 1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize", 7: "stadium" };
const CONDITION_FLAGS = [
  ["poisoned", "Poisoned"],
  ["burned", "Burned"],
  ["asleep", "Asleep"],
  ["paralyzed", "Paralyzed"],
  ["confused", "Confused"],
];

// card_types.json の値（cg エンジンの CardType enum と対応）。
const CARD_TYPE = { POKEMON: 0, ITEM: 1, TOOL: 2, SUPPORTER: 3, STADIUM: 4, BASIC_ENERGY: 5, SPECIAL_ENERGY: 6 };
const ACTIVE_EFFECT_PROFILES = {
  1141: {
    label: "Power Protein",
    badge: "+30",
    expire: "Turn end",
    detail: "This turn, attack damage is boosted by this played card.",
  },
};

const ENERGY_TYPE = {
  G: { ja: "草", en: "G", bg: "#3AAB3A", text: "#fff" },
  R: { ja: "炎", en: "R", bg: "#E8401A", text: "#fff" },
  W: { ja: "水", en: "W", bg: "#2780E0", text: "#fff" },
  L: { ja: "雷", en: "L", bg: "#F0B800", text: "#333" },
  P: { ja: "超", en: "P", bg: "#9640C8", text: "#fff" },
  F: { ja: "闘", en: "F", bg: "#C07020", text: "#fff" },
  D: { ja: "悪", en: "D", bg: "#384050", text: "#fff" },
  M: { ja: "鋼", en: "M", bg: "#7090A8", text: "#fff" },
  N: { ja: "竜", en: "N", bg: "#3CB8C8", text: "#fff" },
  C: { ja: "無", en: "C", bg: "#A0A0A0", text: "#fff" },
  Y: { ja: "妖", en: "Y", bg: "#D03080", text: "#fff" },
};

// cg/api.py の SelectContext 全49種に対応する表示名。ここに無いものは ctxName() が
// enum名をそのまま出す（例: "DrawCount"）ので、cg/api.py に新しい SelectContext が
// 追加されたときはここにも追記すること（api.py のコメントに元の意味が書いてある）。
const CONTEXT_MAP = {
  Main: { ja: "メイン", en: "Main" },
  SetupActivePokemon: { ja: "バトル場セット", en: "Setup Active" },
  SetupBenchPokemon: { ja: "ベンチセット", en: "Setup Bench" },
  Switch: { ja: "にげる/交代", en: "Switch" },
  ToActive: { ja: "バトル場へ", en: "To Active" },
  ToBench: { ja: "ベンチへ", en: "To Bench" },
  ToField: { ja: "場へ", en: "To Field" },
  ToHand: { ja: "手札へ", en: "To Hand" },
  Discard: { ja: "トラッシュ", en: "Discard" },
  ToDeck: { ja: "山札へ", en: "To Deck" },
  ToDeckBottom: { ja: "山札の下へ", en: "To Deck Bottom" },
  ToPrize: { ja: "サイドへ", en: "To Prize" },
  NotMove: { ja: "動かさない", en: "Not Move" },
  DamageCounter: { ja: "ダメカン配置", en: "Damage Counter" },
  DamageCounterAny: { ja: "ダメカン配置（自由）", en: "Damage Counter (Any)" },
  Damage: { ja: "ダメージ対象", en: "Damage Target" },
  RemoveDamageCounter: { ja: "ダメカン除去", en: "Remove Damage Counter" },
  Heal: { ja: "回復", en: "Heal" },
  EvolvesFrom: { ja: "進化元", en: "Evolves From" },
  EvolvesTo: { ja: "進化先", en: "Evolves To" },
  Devolve: { ja: "退化", en: "Devolve" },
  AttachFrom: { ja: "つける道具/エネルギー", en: "Attach From" },
  AttachTo: { ja: "つける先のポケモン", en: "Attach To" },
  DetachFrom: { ja: "外す対象", en: "Detach From" },
  Look: { ja: "確認", en: "Look" },
  EffectTarget: { ja: "効果の対象", en: "Effect Target" },
  DiscardEnergyCard: { ja: "エネルギーをトラッシュ", en: "Discard Energy Card" },
  DiscardToolCard: { ja: "どうぐをトラッシュ", en: "Discard Tool Card" },
  SwitchEnergyCard: { ja: "エネルギーの付け替え", en: "Switch Energy Card" },
  DiscardCardOrAttachedCard: { ja: "トラッシュ対象", en: "Discard Card / Attached Card" },
  DiscardEnergy: { ja: "エネルギーをトラッシュ", en: "Discard Energy" },
  ToHandEnergy: { ja: "エネルギーを手札へ", en: "Energy To Hand" },
  ToDeckEnergy: { ja: "エネルギーを山札へ", en: "Energy To Deck" },
  SwitchEnergy: { ja: "エネルギーの入れ替え", en: "Switch Energy" },
  SkillOrder: { ja: "効果の発動順", en: "Skill Order" },
  Attack: { ja: "ワザ", en: "Attack" },
  DisableAttack: { ja: "ワザを使用不可に", en: "Disable Attack" },
  Evolve: { ja: "進化", en: "Evolve" },
  DrawCount: { ja: "追加ドロー枚数", en: "Draw Count" },
  DamageCounterCount: { ja: "ダメカン配置枚数", en: "Damage Counter Count" },
  RemoveDamageCounterCount: { ja: "ダメカン除去枚数", en: "Remove Damage Counter Count" },
  IsFirst: { ja: "先攻後攻", en: "First / Second" },
  Mulligan: { ja: "マリガン（引き直し）", en: "Mulligan" },
  Activate: { ja: "効果の発動確認", en: "Activate" },
  FirstEffect: { ja: "最初の効果の選択", en: "First Effect" },
  MoreDevolve: { ja: "追加の退化確認", en: "More Devolve" },
  CoinHead: { ja: "コインの表裏", en: "Coin Head" },
  AffectSpecialCondition: { ja: "特殊状態の付与対象", en: "Affect Special Condition" },
  RecoverSpecialCondition: { ja: "特殊状態の回復対象", en: "Recover Special Condition" },
};

// context ごとの Yes/No 質問文。cg/api.py の SelectContext コメント（"Would you like to..."）を
// 日本語化したもの。ここに無い context の Yes/No はそのまま「はい」「いいえ」を表示する。
const YES_NO_QUESTION_MAP = {
  IsFirst: { yes: { ja: "先攻を選ぶ", en: "Go first" }, no: { ja: "後攻を選ぶ", en: "Go second" } },
  Mulligan: { yes: { ja: "引き直す（マリガン）", en: "Redraw (mulligan)" }, no: { ja: "引き直さない", en: "Don't redraw" } },
  Activate: { yes: { ja: "効果を発動する", en: "Activate the effect" }, no: { ja: "発動しない", en: "Don't activate" } },
  FirstEffect: { yes: { ja: "最初の効果を選ぶ", en: "Select the first effect" }, no: { ja: "選ばない", en: "Don't select" } },
  MoreDevolve: { yes: { ja: "さらに退化させる", en: "Devolve further" }, no: { ja: "ここで止める", en: "Stop here" } },
  CoinHead: { yes: { ja: "表を選ぶ", en: "Choose heads" }, no: { ja: "裏を選ぶ", en: "Choose tails" } },
};

let lang = "ja";
let viewerMode = "replay";
let replayData = null;
let frameIndex = 0;
let autoplayTimer = null;
let livePlaybackTimer = null;
let replaySpeed = 1;
let cardManifest = {};
let cardNamesJp = {};
let attackNamesJp = { byId: {}, byName: {} };
let cardTypes = {}; // { card_id: CardType int } — POKEMON0 ITEM1 TOOL2 SUPPORTER3 STADIUM4 BASIC_ENERGY5 SPECIAL_ENERGY6
let currentPlayers = [{}, {}];
let cardRegistry = {};
let cardSelectionRefs = {};
let cardKeyCounter = 0;
let attachedPreviewRegistry = {};
let attachedPreviewKeyCounter = 0;
let effectPreviewRegistry = {};
let effectPreviewKeyCounter = 0;
let previewPinned = false;
let pinnedKey = null;
let selectedCardDetailRef = null;
let effectOverlayTimer = null;
let lastEffectOverlayToken = null;
let liveSelection = [];
let liveFilterRef = null;
let liveBusyMessage = null;
const urlParams = new URLSearchParams(window.location.search);
const ASSET_VERSION = "20260713f";

const UI = {
  ja: {
    eyebrow: "対戦ビューアー",
    modeLabel: "モード",
    replayLabel: "リプレイ",
    liveCpuLabel: "CPU",
    livePanelHeading: "人間 vs CPU",
    liveStart: "LIVE開始",
    liveConfirm: "これで決定",
    liveClear: "選択クリア",
    reloadList: "更新",
    prevButton: "前へ",
    nextButton: "次へ",
    autoplay: "自動再生",
    pause: "一時停止",
    frameLabel: "フレーム",
    summaryHeading: "対戦サマリー",
    moveSummaryHeading: "このフレームの動き",
    actionHeading: "選択内容",
    reasonTraceHeading: "行動の理由",
    searchTraceHeading: "探索レビュー",
    optionsHeading: "合法手一覧",
    logsHeading: "ログ",
    animToggleLabel: "アニメーション",
    langToggle: "EN",
    turnLabel: "ターン",
    selfNameDefault: "Player 0 (あなた)",
    opponentNameDefault: "Player 1",
    first: "先攻",
    second: "後攻",
    win: "勝ち",
    lose: "負け",
    draw: "引き分け",
    terminalContext: "終端",
    hiddenCard: "非公開",
    pinned: "固定中（外側クリックで解除）",
    clickToPin: "クリックで固定",
    noCardImages: "card_images/manifest.json がありません。build_card_assets.py を実行してください。",
    noAction: "このフレームに action はありません。",
    noSearchTrace: "このフレームに探索ログはありません。",
    noLogs: "このフレームにログはありません。",
    noMoves: "移動なし",
    liveInactive: "live 対戦はまだ始まっていません。",
    liveReady: "カードか合法手を選んでから決定してください。",
    liveWaitingCpu: "CPU の手番です。",
    liveFinished: "対戦終了です。",
    liveSelection: "選択",
    liveFilter: "フィルタ",
    liveNoFilter: "なし",
    liveConfirmHint: "合法な index の組み合わせになったら決定できます。",
    sk_replay: "replay",
    sk_opponent: "opponent",
    sk_result: "result",
    sk_turn: "turn",
    sk_actingPlayer: "actingPlayer",
    sk_firstPlayer: "firstPlayer",
    sk_context: "context",
    sk_selected: "selected",
  },
  en: {
    eyebrow: "Battle Review",
    modeLabel: "Mode",
    replayLabel: "Replay",
    liveCpuLabel: "CPU",
    livePanelHeading: "Human vs CPU",
    liveStart: "Start Live",
    liveConfirm: "Confirm",
    liveClear: "Clear",
    reloadList: "Refresh",
    prevButton: "Prev",
    nextButton: "Next",
    autoplay: "Autoplay",
    pause: "Pause",
    frameLabel: "Frame",
    summaryHeading: "Replay Summary",
    moveSummaryHeading: "Frame Moves",
    actionHeading: "Chosen Action",
    reasonTraceHeading: "Why This Action",
    searchTraceHeading: "Search Review",
    optionsHeading: "Options",
    logsHeading: "Logs",
    animToggleLabel: "Animation",
    langToggle: "JP",
    turnLabel: "Turn",
    selfNameDefault: "Player 0 (you)",
    opponentNameDefault: "Player 1",
    first: "First",
    second: "Second",
    win: "WIN",
    lose: "LOSE",
    draw: "DRAW",
    terminalContext: "Terminal",
    hiddenCard: "hidden",
    pinned: "Pinned (click outside to unpin)",
    clickToPin: "Click to pin",
    noCardImages: "Disabled: card_images/manifest.json not found (run build_card_assets.py)",
    noAction: "No action on this frame.",
    noSearchTrace: "No search trace on this frame.",
    noLogs: "No logs on this frame.",
    noMoves: "No moves",
    liveInactive: "No active live match. Start one from the top controls.",
    liveReady: "Click cards or options, then confirm the legal action.",
    liveWaitingCpu: "CPU is taking its turn.",
    liveFinished: "Match finished.",
    liveSelection: "Selection",
    liveFilter: "Filter",
    liveNoFilter: "none",
    liveConfirmHint: "Confirm once the selection is a legal list of option indexes.",
    sk_replay: "replay",
    sk_opponent: "opponent",
    sk_result: "result",
    sk_turn: "turn",
    sk_actingPlayer: "actingPlayer",
    sk_firstPlayer: "firstPlayer",
    sk_context: "context",
    sk_selected: "selected",
  },
};

UI.ja.effectSourceHeading = "使用カード / 効果元";
UI.ja.noEffectSource = "このフレームでは効果元カードを特定できませんでした。";
UI.en.effectSourceHeading = "Effect Source";
UI.en.noEffectSource = "No effect source card could be identified on this frame.";

UI.ja.reasonTraceHeading = "行動の理由";
UI.ja.noReasonTrace = "理由(詳細): このエージェントは未提供です。";
UI.ja.reasonChosen = "採用";
UI.ja.reasonAlternatives = "不採用候補";
UI.en.reasonTraceHeading = "Why This Action";
UI.en.noReasonTrace = "Reason (detail): not provided by this agent.";
UI.en.reasonChosen = "Chosen";
UI.en.reasonAlternatives = "Not chosen";

UI.ja.debugHeading = "デバッグ";
UI.ja.debugViewOpponentKnowledge = "相手の公開情報";
UI.ja.debugViewDeckPredictor = "デッキ予測";
UI.ja.observedCardsHeading = "観測済みカード（名前別）";
UI.ja.currentZonesHeading = "現在のゾーン";
UI.ja.groundTruthDiffHeading = "神視点との差分";
UI.ja.noObservationYet = "まだ観測がありません（player0 の最初の選択待ちです）。";
UI.ja.noCardsObservedYet = "まだ観測されたカードはありません";
UI.ja.noCardsVisible = "現在見えているカードはありません";
UI.ja.noMismatchAtStep = "この時点では差分はありません。";
UI.en.debugHeading = "Debug";
UI.en.debugViewOpponentKnowledge = "Opponent Knowledge";
UI.en.debugViewDeckPredictor = "Deck Predictor";
UI.en.observedCardsHeading = "Observed cards (by name)";
UI.en.currentZonesHeading = "Current zones";
UI.en.groundTruthDiffHeading = "Ground-truth diff";
UI.en.noObservationYet = "No observation yet (waiting for player0's first decision point).";
UI.en.noCardsObservedYet = "No cards observed yet";
UI.en.noCardsVisible = "No cards currently visible";
UI.en.noMismatchAtStep = "No mismatch at this step.";

const ZONE_LABEL = {
  active: { ja: "バトル場", en: "active" },
  bench: { ja: "ベンチ", en: "bench" },
  discard: { ja: "トラッシュ", en: "discard" },
  energy: { ja: "付属エネルギー", en: "energy" },
  tool: { ja: "付属どうぐ", en: "tool" },
  pre_evolution: { ja: "進化元", en: "pre_evolution" },
  stadium: { ja: "スタジアム", en: "stadium" },
  revealed: { ja: "一時公開", en: "revealed" },
};

function zoneLabel(zone) {
  const info = ZONE_LABEL[zone];
  if (!info) return zone;
  return lang === "ja" ? info.ja : info.en;
}

// observed_cards/zone_cards/diff は OpponentKnowledge 側の英語名(name)しか持っていないので、
// nameToCardIds（name -> {card_id: count}）を経由して card_id を逆引きし、cardDisplayName と
// 同じ card_names_jp.json ルックアップに繋ぐ。該当する card_id が見つからなければ英語名のまま。
function localizedObservedName(name, cardId) {
  if (lang !== "ja") return name;
  if (cardId != null) {
    const jp = cardNamesJp[String(cardId)];
    if (jp) return jp;
  }
  return translateEnergyName(name);
}

function localizedZoneName(name, nameToCardIds) {
  if (lang !== "ja") return name;
  const idCounts = nameToCardIds?.[name];
  const cardId = idCounts ? Number(Object.keys(idCounts)[0]) : null;
  return localizedObservedName(name, cardId);
}

function localizedAttackName(option) {
  const fallback = (option?.label || "").replace(/^Use attack:\s*/i, "");
  if (lang !== "ja") return fallback;

  const raw = option?.raw || {};
  const byId = attackNamesJp?.byId || {};
  const byName = attackNamesJp?.byName || {};
  const attackId = raw.attackId == null ? null : String(raw.attackId);
  return (attackId && byId[attackId]) || byName[fallback] || fallback;
}

function t(key) {
  return UI[lang][key] ?? key;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function parseEnergyCode(name) {
  const m = String(name || "").match(/\{(\w+)\}/);
  return m ? m[1].toUpperCase() : "C";
}

function energyDotHtml(energyCard) {
  // 特殊エネルギーは名前に {R} のようなタイプ記号を持たないことが多く、それだけで判定すると
  // 無色の基本エネルギーと見分けが付かなくなる（= 特殊効果があるのに気付けない）。
  // card_types.json（cg エンジン由来の正しい CardType）で判定できる場合はそちらを優先する。
  if (cardTypeOf(energyCard?.id) === CARD_TYPE.SPECIAL_ENERGY) {
    const name = cardDisplayName(energyCard);
    const title = lang === "ja" ? `特殊エネルギー: ${name}` : `Special Energy: ${name}`;
    const label = lang === "ja" ? "特" : "S";
    return `<span class="en-dot en-dot-special" title="${escapeHtml(title)}">${escapeHtml(label)}</span>`;
  }
  const code = parseEnergyCode(energyCard?.name);
  const info = ENERGY_TYPE[code] || ENERGY_TYPE.C;
  const label = lang === "ja" ? info.ja : info.en;
  const title = lang === "ja" ? `${info.ja}エネルギー` : `${info.en} Energy`;
  return `<span class="en-dot" style="background:${info.bg};color:${info.text}" title="${escapeHtml(title)}">${escapeHtml(label)}</span>`;
}

function energyDotsHtml(card) {
  if (Array.isArray(card?.energyCards) && card.energyCards.length) {
    return card.energyCards.map(energyDotHtml).join("");
  }
  if (Array.isArray(card?.energies) && card.energies.length) {
    return card.energies.map(() => energyDotHtml(null)).join("");
  }
  return "";
}

function translateEnergyName(name) {
  return String(name || "").replace(/Basic \{(\w+)\} Energy/i, (_, code) => {
    const info = ENERGY_TYPE[code.toUpperCase()];
    return info ? `基本${info.ja}エネルギー` : name;
  });
}

function cardDisplayName(card) {
  if (!card) return t("hiddenCard");
  if (lang === "ja" && card.id != null) {
    const jpName = cardNamesJp[String(card.id)];
    if (jpName) return jpName;
    if (card.name) return translateEnergyName(card.name);
  }
  return card.name || `card #${card.id ?? "?"}`;
}

function hpColor(hp, maxHp) {
  if (hp == null) return "";
  const max = maxHp != null && maxHp > 0 ? maxHp : hp;
  if (max <= 0 || hp >= max) return "hp-full";
  const ratio = hp / max;
  if (ratio > 0.5) return "hp-full";
  if (ratio > 0.25) return "hp-mid";
  return "hp-low";
}

function ctxName(ctx) {
  if (ctx == null) return t("terminalContext");
  const entry = CONTEXT_MAP[String(ctx)] || CONTEXT_MAP[ctx];
  if (entry) return entry[lang] || String(ctx);
  return String(ctx);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Failed to post ${url}: ${response.status}`);
  }
  return body;
}

function isLiveMode() {
  return viewerMode === "live";
}

function liveLatestFrameIndex() {
  if (!isLiveMode()) return replayData?.frames?.length ? replayData.frames.length - 1 : 0;
  return replayData?.live?.latestFrameIndex ?? Math.max((replayData?.frames?.length || 1) - 1, 0);
}

function isLatestLiveFrame() {
  return frameIndex === liveLatestFrameIndex();
}

function currentFrame() {
  if (!replayData?.frames?.length) return null;
  const base = replayData.frames[Math.min(Math.max(frameIndex, 0), replayData.frames.length - 1)];
  if (!isLiveMode() || !isLatestLiveFrame()) return base;
  return { ...base, action: [...liveSelection] };
}

function liveInfo() {
  return replayData?.live || { active: false, humanTurn: false, minCount: 0, maxCount: 0 };
}

function getCardFromVisual(visual, area, index, playerIndex) {
  const current = visual?.current;
  if (!current) return null;
  const player = (current.players || [])[playerIndex];
  if (!player) return null;
  const zoneName = AREA_ZONE[area];
  if (!zoneName) return null;
  const zone = player[zoneName];
  if (!zone) return null;
  if (zoneName === "stadium") return Array.isArray(zone) ? (zone[0] ?? null) : null;
  if (!Array.isArray(zone) || index < 0 || index >= zone.length) return null;
  const card = zone[index];
  if (zoneName === "active" && Array.isArray(card)) return card[0] ?? null;
  return card;
}

function cardStub(cardId, serial, playerIndex, name = null) {
  const fallbackName = name || cardNamesJp[String(cardId)] || `card #${cardId ?? "?"}`;
  return { id: cardId, serial, playerIndex, name: fallbackName };
}

function findCardByIdentity(visual, ref = {}) {
  const current = visual?.current;
  if (!current) return null;
  const wantedSerial = ref.serial ?? null;
  const wantedId = ref.id ?? ref.cardId ?? null;
  const wantedPlayer = ref.playerIndex ?? null;

  const inspectPokemon = (pokemon) => {
    if (!pokemon) return null;
    if ((wantedSerial == null || pokemon.serial === wantedSerial) && (wantedId == null || pokemon.id === wantedId)) {
      return pokemon;
    }
    for (const child of [...(pokemon.energyCards || []), ...(pokemon.tools || []), ...(pokemon.preEvolution || [])]) {
      if ((wantedSerial == null || child.serial === wantedSerial) && (wantedId == null || child.id === wantedId)) {
        return child;
      }
    }
    return null;
  };

  for (const [playerIndex, player] of (current.players || []).entries()) {
    if (wantedPlayer != null && playerIndex !== wantedPlayer) continue;
    for (const zoneName of ["hand", "discard", "prize", "deck", "stadium"]) {
      for (const card of player?.[zoneName] || []) {
        if (!card) continue;
        if ((wantedSerial == null || card.serial === wantedSerial) && (wantedId == null || card.id === wantedId)) {
          return card;
        }
      }
    }
    for (const slot of player?.active || []) {
      const found = inspectPokemon(Array.isArray(slot) ? slot[0] : slot);
      if (found) return found;
    }
    for (const pokemon of player?.bench || []) {
      const found = inspectPokemon(pokemon);
      if (found) return found;
    }
  }

  if (wantedId != null || wantedSerial != null) {
    return cardStub(wantedId, wantedSerial, wantedPlayer);
  }
  return null;
}

function primarySelectedOption(frame) {
  if (!frame?.action?.length) return null;
  const selected = new Set(frame.action);
  return (frame.options || []).find((option) => selected.has(option.index)) || null;
}

function resolveEffectSource(frame) {
  const visual = frame?.visual || {};
  const logs = visual.logs || [];
  const select = visual.select || {};
  const current = visual.current || {};
  const actingPlayer = current.yourIndex ?? frame?.actingPlayer ?? SELF_INDEX;
  const selectedOption = primarySelectedOption(frame);
  const raw = selectedOption?.raw || {};

  // グッズ/サポートが使われたフレーム（ワザ=攻撃は対象外）。
  const playLog = effectPlayLog(frame);
  if (playLog) {
    return {
      card: findCardByIdentity(visual, { id: playLog.cardId, serial: playLog.serial, playerIndex: playLog.playerIndex }),
      kind: lang === "ja" ? "このフレームで使われたカード" : "Played card on this frame",
      detail: lang === "ja"
        ? "このカードの使用結果として、右下のログや移動が発生しています。"
        : "The logs and card movements on this frame are resolving this played card.",
    };
  }

  if (select.effect) {
    return {
      card: findCardByIdentity(visual, select.effect),
      kind: lang === "ja" ? "いま解決中の効果元カード" : "Card whose effect is resolving",
      detail: selectedOption ? describeOptionLabel(selectedOption, frame) : (lang === "ja" ? "このカードの効果処理中です。" : "This card effect is currently being resolved."),
    };
  }

  if (raw.type === "Ability") {
    return {
      card: getCardFromVisual(visual, raw.area, raw.index, raw.playerIndex ?? actingPlayer),
      kind: lang === "ja" ? "選択中の特性" : "Selected ability",
      detail: describeOptionLabel(selectedOption, frame),
    };
  }

  // Activate プロンプト（例: フーディン）で「はい」を選んだ発動カード。
  if (select.context === "Activate" && select.contextCard && raw.type === "Yes") {
    return {
      card: findCardByIdentity(visual, select.contextCard),
      kind: lang === "ja" ? "発動した特性/効果" : "Activated ability/effect",
      detail: lang === "ja" ? "このカードの効果を発動しました。" : "Activated this card's effect.",
    };
  }

  if (raw.type === "Play") {
    return {
      card: getCardFromVisual(visual, AREA.HAND, raw.index, actingPlayer),
      kind: lang === "ja" ? "手札から使用するカード" : "Card being played from hand",
      detail: describeOptionLabel(selectedOption, frame),
    };
  }

  if (select.contextCard) {
    return {
      card: findCardByIdentity(visual, select.contextCard),
      kind: lang === "ja" ? "この選択に関連するカード" : "Card related to this prompt",
      detail: lang === "ja" ? "このカードに対する選択を待っています。" : "Waiting for a choice related to this card.",
    };
  }

  return null;
}

// このカードタイプの「使用」だけオーバーレイに出す: グッズ(ITEM=1) と サポート(SUPPORTER=3)。
// ポケモン/エネルギー/ワザ(攻撃) は出さない。特性(Ability)は別枠で出す。
const EFFECT_PLAY_TYPES = new Set([1, 3]);

function cardTypeOf(cardId) {
  if (cardId == null) return undefined;
  const t = cardTypes[String(cardId)];
  return typeof t === "number" ? t : undefined;
}

function activeEffectProfile(cardId) {
  return ACTIVE_EFFECT_PROFILES[String(cardId)] || null;
}

function activeEffectKey(playerIndex, cardId, serial) {
  return `${playerIndex ?? "?"}:${cardId ?? "?"}:${serial ?? "?"}`;
}

function buildActiveEffects(currentIndex) {
  const effects = new Map();
  const frames = replayData?.frames || [];
  const last = Math.min(currentIndex, frames.length - 1);
  for (let i = 0; i <= last; i += 1) {
    const frame = frames[i];
    for (const log of frame?.visual?.logs || []) {
      if (!log) continue;
      if ((log.type === "TurnEnd" || log.type === "TurnStart") && log.playerIndex != null) {
        for (const [key, effect] of Array.from(effects.entries())) {
          if (effect.playerIndex === log.playerIndex) effects.delete(key);
        }
      }
      if (log.type !== "Play") continue;
      const profile = activeEffectProfile(log.cardId);
      if (!profile) continue;
      const card = findCardByIdentity(frame.visual, { id: log.cardId, serial: log.serial, playerIndex: log.playerIndex })
        || cardStub(log.cardId, log.serial, log.playerIndex, profile.label);
      const key = activeEffectKey(log.playerIndex, log.cardId, log.serial);
      effects.set(key, {
        key,
        card,
        profile,
        playerIndex: log.playerIndex,
        startedFrame: i,
      });
    }
  }
  return Array.from(effects.values());
}

function registerEffectPreviewCard(card) {
  const key = String(effectPreviewKeyCounter++);
  effectPreviewRegistry[key] = card;
  return key;
}

function renderActiveEffectsShelf() {
  if (!activeEffectsShelf) return;
  if (previewPinned && String(pinnedKey || "").startsWith("effect:")) unpinPreview();
  effectPreviewRegistry = {};
  effectPreviewKeyCounter = 0;
  const effects = buildActiveEffects(frameIndex);
  if (!effects.length) {
    activeEffectsShelf.hidden = true;
    activeEffectsShelf.innerHTML = "";
    return;
  }
  activeEffectsShelf.hidden = false;
  activeEffectsShelf.innerHTML = effects.map((effect) => {
    const previewKey = registerEffectPreviewCard(effect.card);
    const player = effect.playerIndex === SELF_INDEX ? "P0" : effect.playerIndex === OPPONENT_INDEX ? "P1" : `P${effect.playerIndex ?? "?"}`;
    return `
      <button class="active-effect-card" type="button" data-effect-card-key="${escapeHtml(previewKey)}" title="${escapeHtml(effect.profile.detail)}">
        <span class="active-effect-owner">${escapeHtml(player)}</span>
        ${cardImageHtml(effect.card, "active-effect-img")}
        <span class="active-effect-badge">${escapeHtml(effect.profile.badge)}</span>
        <span class="active-effect-expire">${escapeHtml(effect.profile.expire)}</span>
      </button>`;
  }).join("");
}

function effectPlayLog(frame) {
  // このフレームで解決された Play のうち、グッズ/サポートのものを返す。
  const logs = frame?.visual?.logs || [];
  return logs.find((log) => log?.type === "Play" && EFFECT_PLAY_TYPES.has(cardTypeOf(log.cardId))) || null;
}

function activatedAbilityCard(frame) {
  // 特性の発動には2通りの出方があるので両方拾う:
  //   (A) Ability メイン選択肢（例: ノココッチ）→ raw.type === "Ability"
  //   (B) 「効果を発動しますか？」の Activate プロンプトで「はい」を選んだ場合（例: フーディン, ユンゲラー）
  //       → select.context === "Activate" かつ contextCard が発動カード
  const selectedOption = primarySelectedOption(frame);
  const raw = selectedOption?.raw || {};
  if (raw.type === "Ability") {
    const actingPlayer = frame?.visual?.current?.yourIndex ?? frame?.actingPlayer ?? SELF_INDEX;
    return getCardFromVisual(frame?.visual, raw.area, raw.index, raw.playerIndex ?? actingPlayer) || true;
  }
  const select = frame?.visual?.select || {};
  if (select.context === "Activate" && select.contextCard && raw.type === "Yes") {
    return select.contextCard;
  }
  return null;
}

function shouldShowEffectOverlay(frame) {
  // 直前アクションの結果ログ: グッズ/サポートが使われた。
  if (effectPlayLog(frame)) return true;

  // 特性の発動（Ability 選択肢 or Activate プロンプトで「はい」）。
  if (activatedAbilityCard(frame)) return true;

  // 手札からグッズ/サポートを使おうとしている（選択時点）。
  const raw = primarySelectedOption(frame)?.raw || {};
  if (raw.type === "Play") {
    const actingPlayer = frame?.visual?.current?.yourIndex ?? frame?.actingPlayer ?? SELF_INDEX;
    const card = getCardFromVisual(frame?.visual, AREA.HAND, raw.index, actingPlayer);
    if (card && EFFECT_PLAY_TYPES.has(cardTypeOf(card.id))) return true;
  }
  return false;
}

function cardDisambiguator(card) {
  // 同名カード（同じ名前・別インスタンス）が同時に選択肢に並ぶことがあるため
  // （例: セットアップ中に手札の2匹目の同名ポケモンを選ぶ場合）、id/serial を必ず添えて
  // 「カード選択: リオル」が複数並んで区別できない、という事態を避ける。
  if (!card) return "";
  const bits = [];
  if (card.id != null) bits.push(`id=${card.id}`);
  if (card.serial != null) bits.push(`serial=${card.serial}`);
  return bits.length ? ` (${bits.join(", ")})` : "";
}

function namedCard(card) {
  return `${cardDisplayName(card)}${cardDisambiguator(card)}`;
}

function describeOptionLabel(option, frame) {
  if (lang === "en") return option.label;

  const raw = option.raw || {};
  const visual = frame.visual || {};
  const actingPlayer = visual.current?.yourIndex ?? frame.actingPlayer ?? SELF_INDEX;
  const type = raw.type;
  const context = visual.select?.context;

  if (type === "Yes" || type === "No") {
    const question = YES_NO_QUESTION_MAP[context];
    if (question) return (type === "Yes" ? question.yes : question.no)[lang] || (type === "Yes" ? "はい" : "いいえ");
    return type === "Yes" ? "はい" : "いいえ";
  }
  if (type === "Number") return `数字 ${raw.number ?? "?"}`;
  if (type === "Attack") return `ワザ: ${localizedAttackName(option)}`;
  if (type === "Play") {
    const card = getCardFromVisual(visual, AREA.HAND, raw.index, actingPlayer);
    return `手札から出す: ${namedCard(card)}`;
  }
  if (type === "Card") {
    const owner = raw.playerIndex ?? actingPlayer;
    const card = getCardFromVisual(visual, raw.area, raw.index, owner);
    return `カード選択: ${namedCard(card)}`;
  }
  if (type === "Ability") {
    const owner = raw.playerIndex ?? actingPlayer;
    const card = getCardFromVisual(visual, raw.area, raw.index, owner);
    return `特性: ${namedCard(card)}`;
  }
  if (type === "Attach") {
    const source = getCardFromVisual(visual, raw.area, raw.index, actingPlayer);
    const target = getCardFromVisual(visual, raw.inPlayArea, raw.inPlayIndex, actingPlayer);
    return `${namedCard(source)} を ${namedCard(target)} につける`;
  }
  if (type === "Evolve") {
    const evolved = getCardFromVisual(visual, raw.area, raw.index, actingPlayer);
    const base = getCardFromVisual(visual, raw.inPlayArea, raw.inPlayIndex, actingPlayer);
    return `${namedCard(base)} を ${namedCard(evolved)} に進化`;
  }
  if (type === "Energy") return `エネルギー ${raw.energyIndex ?? "?"}`;
  if (type === "EnergyCard") return `ついているエネルギー ${raw.energyIndex ?? "?"}`;
  if (type === "ToolCard") return `ついているどうぐ ${raw.toolIndex ?? "?"}`;
  if (type === "Retreat") return "にげる";
  if (type === "End") return "ターン終了";
  return option.label || JSON.stringify(raw);
}

function applyLang() {
  const set = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };

  set("eyebrowLabel", t("eyebrow"));
  set("modeLabel", t("modeLabel"));
  set("replayLabel", t("replayLabel"));
  set("speedLabel", lang === "ja" ? "速度" : "Speed");
  set("liveCpuLabel", t("liveCpuLabel"));
  set("livePlayerDeckLabel", lang === "ja" ? "自分デッキ" : "Your deck");
  set("liveOpponentDeckLabel", lang === "ja" ? "CPUデッキ" : "CPU deck");
  set("livePanelHeading", t("livePanelHeading"));
  set("reloadButton", t("reloadList"));
  set("prevButton", t("prevButton"));
  set("nextButton", t("nextButton"));
  set("frameSectionLabel", t("frameLabel"));
  set("summaryHeading", t("summaryHeading"));
  set("moveSummaryHeading", t("moveSummaryHeading"));
  set("actionHeading", t("actionHeading"));
  set("reasonTraceHeading", t("reasonTraceHeading"));
  set("effectSourceHeading", t("effectSourceHeading"));
  set("searchTraceHeading", t("searchTraceHeading"));
  set("optionsHeading", t("optionsHeading"));
  set("logsHeading", t("logsHeading"));
  set("animToggleLabel", t("animToggleLabel"));
  set("langToggleButton", t("langToggle"));
  set("liveStartButton", t("liveStart"));
  set("liveUndoButton", lang === "ja" ? "1手戻る" : "Undo");
  set("liveConfirmButton", t("liveConfirm"));
  set("liveClearButton", t("liveClear"));
  set("debugHeading", t("debugHeading"));
  set("observedCardsHeading", t("observedCardsHeading"));
  set("currentZonesHeading", t("currentZonesHeading"));
  set("groundTruthDiffHeading", t("groundTruthDiffHeading"));
  set("generatePlayerPolicyLabel", lang === "ja" ? "自分のCPU" : "Your CPU");
  set("generateOpponentLabel", lang === "ja" ? "相手のCPU" : "Opponent CPU");
  set("generatePlayerDeckLabel", lang === "ja" ? "自分のデッキ" : "Your deck");
  set("generateOpponentDeckLabel", lang === "ja" ? "相手のデッキ" : "Opponent deck");
  buildDebugSubtabs();

  autoplayButton.textContent = (autoplayTimer || livePlaybackTimer) ? t("pause") : t("autoplay");
  updateModeUI();
  if (replayData?.frames?.length) render();
  else renderLivePanel();
}

function updateModeUI() {
  document.body.classList.toggle("live-mode", isLiveMode());
  document.body.classList.toggle("replay-mode", !isLiveMode());
  document.querySelector(".topbar")?.classList.toggle("is-disabled", false);
  replaySelect.disabled = isLiveMode() || replaySelect.options.length === 0;
  prevButton.disabled = !replayData?.frames?.length || frameIndex <= 0;
  nextButton.disabled = !replayData?.frames?.length || frameIndex >= Math.max((replayData?.frames?.length || 1) - 1, 0);
  autoplayButton.disabled = !replayData?.frames?.length;
  if (speedSelect) speedSelect.disabled = false;
  frameSlider.disabled = !replayData?.frames?.length;

  const showLiveConfig = isLiveMode();
  document.querySelectorAll(".live-config").forEach((el) => {
    el.classList.toggle("live-config-hidden", !showLiveConfig);
  });
}

async function loadReplayList() {
  const replayFiles = await fetchJson("/api/replays");
  replaySelect.innerHTML = "";
  if (!replayFiles.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No replay files found";
    replaySelect.append(option);
    replaySelect.disabled = true;
    return;
  }

  const requestedReplay = requestedReplayName();
    replayFiles.forEach((file, index) => {
    const option = document.createElement("option");
    option.value = file.name;
    option.textContent = index === 0 ? `${file.name} (latest)` : file.name;
    replaySelect.append(option);
  });
  replaySelect.disabled = false;
  replaySelect.value = replayFiles.some((file) => file.name === requestedReplay)
    ? requestedReplay
    : replayFiles[0].name;
}

async function loadSelectedReplay() {
  if (!replaySelect.value) return;
  hideEffectOverlay();
  lastEffectOverlayToken = null;
  replayData = await fetchJson(`/api/replays/${encodeURIComponent(replaySelect.value)}`);
  const lastIndex = Math.max((replayData.frames?.length || 1) - 1, 0);
  const requested = Number(new URLSearchParams(window.location.search).get("frame"));
  frameIndex = Number.isFinite(requested) ? Math.min(Math.max(requested, 0), lastIndex) : 0;
  frameSlider.min = "0";
  frameSlider.max = String(lastIndex);
  frameSlider.value = String(frameIndex);
  liveSelection = [];
  liveFilterRef = null;
  syncReplayUrl();
  render();
}

async function loadLiveState() {
  hideEffectOverlay();
  lastEffectOverlayToken = null;
  replayData = await fetchJson("/api/live/state");
  frameIndex = Math.min(liveLatestFrameIndex(), Math.max((replayData.frames?.length || 1) - 1, 0));
  frameSlider.min = "0";
  frameSlider.max = String(Math.max((replayData.frames?.length || 1) - 1, 0));
  frameSlider.value = String(frameIndex);
  render();
}

async function ensureLiveMatchStarted() {
  await loadLiveState();
  if (replayData?.live?.active && replayData?.live?.humanTurn && activePanel === null) {
    activePanel = "action";
    applyPanel();
  }
}

function optionRefs(raw, actingPlayer) {
  const refs = [];
  if (raw.type === "Play" && raw.index != null) {
    refs.push(`${AREA.HAND}:${actingPlayer}:${raw.index}`);
  } else if ((raw.type === "Card" || raw.type === "Ability") && raw.area != null && raw.index != null) {
    refs.push(`${raw.area}:${raw.playerIndex ?? actingPlayer}:${raw.index}`);
  } else if (raw.type === "Attach" || raw.type === "Evolve") {
    if (raw.area != null && raw.index != null) refs.push(`${raw.area}:${actingPlayer}:${raw.index}`);
    if (raw.inPlayArea != null && raw.inPlayIndex != null) refs.push(`${raw.inPlayArea}:${actingPlayer}:${raw.inPlayIndex}`);
  }
  return refs;
}

function actionableOptionsForRef(frame, refKey) {
  if (!frame || !refKey) return [];
  const acting = frame.actingPlayer ?? SELF_INDEX;
  return (frame.options || []).filter((option) => optionRefs(option.raw || {}, acting).includes(refKey));
}

function getActionableCardRefs(frame) {
  const refs = new Set();
  const live = liveInfo();
  if (!isLiveMode() || !live.active || !live.humanTurn || !isLatestLiveFrame() || liveBusyMessage) return refs;
  const acting = frame?.actingPlayer ?? SELF_INDEX;
  (frame?.options || []).forEach((option) => {
    optionRefs(option.raw || {}, acting).forEach((ref) => refs.add(ref));
  });
  return refs;
}

function optionMatchesFilter(option, frame, refKey) {
  if (!refKey) return true;
  const raw = option.raw || {};
  if (raw.type === "End" || raw.type === "Retreat" || raw.type === "Yes" || raw.type === "No" || raw.type === "Number" || raw.type === "Attack") {
    return true;
  }
  const acting = frame.actingPlayer ?? SELF_INDEX;
  return optionRefs(raw, acting).includes(refKey);
}

function visibleOptions(frame) {
  const options = frame?.options || [];
  if (!isLiveMode() || !liveFilterRef) return options;
  return options.filter((option) => optionMatchesFilter(option, frame, liveFilterRef));
}

function getSelectedCardRefs(frame) {
  const refs = new Set();
  const acting = frame.actingPlayer ?? SELF_INDEX;
  const selected = new Set(frame.action || []);
  (frame.options || []).forEach((option) => {
    if (!selected.has(option.index)) return;
    optionRefs(option.raw || {}, acting).forEach((ref) => refs.add(ref));
  });
  return refs;
}

async function handleLiveBoardSelection(refKey) {
  if (!isLiveMode()) return false;
  const frame = currentFrame();
  const live = liveInfo();
  if (!frame || !live.active || !live.humanTurn || !isLatestLiveFrame() || liveBusyMessage) return false;

  const matches = actionableOptionsForRef(frame, refKey);
  if (!matches.length) return false;

  if (matches.length === 1) {
    liveFilterRef = null;
    await toggleLiveOption(matches[0].index);
    return true;
  }

  liveFilterRef = liveFilterRef === refKey ? null : refKey;
  activePanel = "action";
  applyPanel();
  render();
  return true;
}

async function toggleLiveOption(optionIndex) {
  if (!isLiveMode()) return;
  const frame = currentFrame();
  const live = liveInfo();
  if (!frame || !live.active || !live.humanTurn || !isLatestLiveFrame() || liveBusyMessage) return;
  const option = (frame.options || []).find((item) => item.index === optionIndex);
  const rawType = option?.raw?.type;

  if (rawType === "End") {
    liveSelection = [optionIndex];
    render();
    await submitLiveAction();
    return;
  }

  if (live.minCount === 1 && live.maxCount === 1) {
    liveSelection = [optionIndex];
    render();
    await submitLiveAction();
    return;
  }

  const next = [...liveSelection];
  const pos = next.indexOf(optionIndex);
  if (pos >= 0) {
    next.splice(pos, 1);
  } else if (next.length < live.maxCount) {
    next.push(optionIndex);
  }
  liveSelection = next.sort((a, b) => a - b);
  render();
}

async function startLiveMatch() {
  stopLivePlayback();
  liveBusyMessage = lang === "ja" ? "対戦を開始しています..." : "Starting live match...";
  render();
  try {
    hideEffectOverlay();
    lastEffectOverlayToken = null;
    replayData = await postJson("/api/live/start", {
      cpuPolicy: liveCpuSelect.value,
      playerDeck: livePlayerDeckSelect?.value || "",
      opponentDeck: liveOpponentDeckSelect?.value || "",
    });
    liveSelection = [];
    liveFilterRef = null;
    frameIndex = liveLatestFrameIndex();
    activePanel = "action";
    applyPanel();
    document.querySelectorAll(".live-config").forEach((el) => {
      el.classList.add("live-config-hidden");
    });
  } finally {
    liveBusyMessage = null;
    render();
  }
}

async function submitLiveAction() {
  const live = liveInfo();
  if (!live.active || !live.humanTurn || !isLatestLiveFrame()) return;
  if (liveSelection.length < live.minCount || liveSelection.length > live.maxCount) return;
  const previousLatestFrame = liveLatestFrameIndex();
  stopLivePlayback();
  liveBusyMessage = lang === "ja" ? "ターンを反映して CPU の応答を待っています..." : "Applying your turn and waiting for CPU...";
  render();
  try {
    replayData = await postJson("/api/live/action", { action: liveSelection });
    liveSelection = [];
    liveFilterRef = null;
    frameIndex = Math.min(previousLatestFrame, Math.max((replayData.frames?.length || 1) - 1, 0));
  } finally {
    liveBusyMessage = null;
    render();
  }
  playLiveSequenceToLatest(true);
}

async function undoLiveTurn() {
  const live = liveInfo();
  if (!live.active || !live.canUndo) return;
  stopLivePlayback();
  liveBusyMessage = lang === "ja" ? "1手戻しています..." : "Undoing last human turn...";
  render();
  try {
    hideEffectOverlay();
    lastEffectOverlayToken = null;
    replayData = await postJson("/api/live/undo", {});
    liveSelection = [];
    liveFilterRef = null;
    frameIndex = liveLatestFrameIndex();
  } finally {
    liveBusyMessage = null;
    render();
  }
}

function clearLiveSelection() {
  liveSelection = [];
  liveFilterRef = null;
  render();
}

function render() {
  syncReplayUrl();
  updateModeUI();
  if (!replayData?.frames?.length) {
    hideEffectOverlay();
    if (activeEffectsShelf) {
      activeEffectsShelf.hidden = true;
      activeEffectsShelf.innerHTML = "";
    }
    frameMeta.textContent = isLiveMode() ? "live" : "0 / 0";
    renderLivePanel();
    return;
  }

  const frame = currentFrame();
  const visual = frame.visual || {};
  const current = visual.current || {};
  const players = current.players || [{}, {}];
  const self = players[SELF_INDEX] || {};
  const opponent = players[OPPONENT_INDEX] || {};
  const actingPlayer = current.yourIndex ?? frame.actingPlayer ?? SELF_INDEX;
  const selectedRefs = getSelectedCardRefs(frame);
  const actionableRefs = getActionableCardRefs(frame);

  cardRegistry = {};
  cardSelectionRefs = {};
  cardKeyCounter = 0;
  if (!previewPinned) cardPreview.hidden = true;
  closeDiscardModal();
  currentPlayers = players;

  frameMeta.textContent = isLiveMode() ? `live ${frameIndex + 1} / ${replayData.frames.length}` : `${frameIndex + 1} / ${replayData.frames.length}`;
  frameSlider.value = String(frameIndex);
  frameSlider.max = String(Math.max((replayData.frames?.length || 1) - 1, 0));
  document.getElementById("turnBadge").textContent = `${t("turnLabel")} ${frame.turn ?? "?"}`;
  document.getElementById("contextBadge").textContent = ctxName(frame.context);
  renderActiveEffectsShelf();

  renderSummary(replayData.metadata || {}, frame, current);
  renderResultBanner(current, replayData.metadata || {});
  renderMeta(current, replayData.metadata || {}, actingPlayer);
  renderMoveSummary([]);
  renderAction(frame);
  renderReasonTrace(frame);
  renderEffectSource(frame);
  showEffectOverlay(frame);
  renderSearchTrace(frame);
  renderOptions(frame);
  renderLogs(visual.logs || []);
  const debugEntry = findLatestOpponentKnowledgeDebug(replayData.frames, frameIndex);
  renderOpponentKnowledge(debugEntry);
  renderDeckPredictor(debugEntry);
  renderStadium(current.stadium || [], selectedRefs, actionableRefs);

  renderPlayer(opponent, OPPONENT_INDEX, selectedRefs, actionableRefs, {
    activeId: "opponentActive",
    benchId: "opponentBench",
    prizeId: "opponentPrize",
    deckCountId: "opponentDeckCount",
    discardCountId: "opponentDiscardCount",
    conditionsId: "opponentConditions",
  });
  renderPlayer(self, SELF_INDEX, selectedRefs, actionableRefs, {
    activeId: "selfActive",
    benchId: "selfBench",
    prizeId: "selfPrize",
    deckCountId: "selfDeckCount",
    discardCountId: "selfDiscardCount",
    conditionsId: "selfConditions",
  });

  renderHand("opponentHand", "opponentHandCount", opponent.hand || [], {
    hidden: !toggleOpponentHand.checked,
    selectedRefs,
    actionableRefs,
    playerIndex: OPPONENT_INDEX,
  });
  renderHand("selfHand", "selfHandCount", self.hand || [], {
    hidden: false,
    selectedRefs,
    actionableRefs,
    playerIndex: SELF_INDEX,
  });

  refreshOpenCardDetail(frame.visual);
  renderLivePanel();
  if (isLiveMode() && !isLatestLiveFrame()) {
    liveConfirmButton.disabled = true;
  }
}

function describeResult(result, players = {}) {
  if (result == null || result < 0) return "-";
  if (result === 2) {
    return t("draw");
  }
  const playerName = players[result]?.name || (result === SELF_INDEX ? t("selfNameDefault") : t("opponentNameDefault"));
  const label = result === SELF_INDEX ? t("win") : t("lose");
  return `${playerName} ${label}`;
}

function renderSummary(metadata, frame, current) {
  const summaryList = document.getElementById("summaryList");
  const entries = [
    [t("sk_replay"), isLiveMode() ? "live" : replaySelect.value],
    [t("sk_opponent"), metadata.opponent ?? "-"],
    [t("sk_result"), describeResult(metadata.result, metadata.players || {})],
    [t("sk_turn"), frame.turn ?? "-"],
    [t("sk_actingPlayer"), current.yourIndex ?? frame.actingPlayer ?? "-"],
    [t("sk_firstPlayer"), current.firstPlayer ?? "-"],
    [t("sk_context"), ctxName(frame.context)],
    [t("sk_selected"), JSON.stringify(frame.action || [])],
  ];
  summaryList.innerHTML = entries
    .map(([key, value]) => `<dt>${escapeHtml(String(key))}</dt><dd>${escapeHtml(String(value))}</dd>`)
    .join("");
}

function renderResultBanner(current, metadata) {
  const banner = document.getElementById("resultBanner");
  if (!banner) return;
  const result = current?.result ?? metadata?.result ?? -1;
  if (result == null || result < 0) {
    banner.hidden = true;
    banner.className = "result-banner";
    banner.innerHTML = "";
    return;
  }

  const names = metadata?.players || {};
  const selfName = names[SELF_INDEX]?.name || t("selfNameDefault");
  const opponentName = names[OPPONENT_INDEX]?.name || t("opponentNameDefault");
  let title = "";
  let winnerName = "";
  let tone = "draw";

  if (result === 2) {
    title = lang === "ja" ? "引き分け" : "Draw";
    winnerName = lang === "ja" ? "引き分けです" : "It is a draw";
    tone = "draw";
  } else if (result === SELF_INDEX) {
    title = lang === "ja" ? "勝ち" : "Win";
    winnerName = selfName;
    tone = "win";
  } else if (result === OPPONENT_INDEX) {
    title = lang === "ja" ? "負け" : "Loss";
    winnerName = opponentName;
    tone = "loss";
  } else {
    banner.hidden = true;
    banner.className = "result-banner";
    banner.innerHTML = "";
    return;
  }

  banner.className = `result-banner ${tone}`;
  banner.innerHTML = `
    <div class="result-banner-title">${escapeHtml(title)}</div>
    <div class="result-banner-name">${escapeHtml(winnerName)}</div>
    <div class="result-banner-subtitle">${escapeHtml(lang === "ja" ? "対戦終了" : "Match finished")}</div>
  `;
  banner.hidden = false;
}

function renderMeta(current, metadata, actingPlayer) {
  document.getElementById("selfStrip").classList.toggle("acting", actingPlayer === SELF_INDEX);
  document.getElementById("opponentStrip").classList.toggle("acting", actingPlayer === OPPONENT_INDEX);

  const names = metadata.players || {};
  document.getElementById("selfName").textContent = names[SELF_INDEX]?.name || t("selfNameDefault");
  document.getElementById("opponentName").textContent = names[OPPONENT_INDEX]?.name || t("opponentNameDefault");

  setOrderBadge("selfOrder", current.firstPlayer, SELF_INDEX);
  setOrderBadge("opponentOrder", current.firstPlayer, OPPONENT_INDEX);
  setResultBadge("selfResult", metadata.result, SELF_INDEX);
  setResultBadge("opponentResult", metadata.result, OPPONENT_INDEX);
}

function setOrderBadge(id, firstPlayer, playerIndex) {
  const el = document.getElementById(id);
  if (firstPlayer == null || firstPlayer < 0) {
    el.textContent = "";
    el.className = "order-badge";
    return;
  }
  const isFirst = firstPlayer === playerIndex;
  el.textContent = isFirst ? t("first") : t("second");
  el.className = isFirst ? "order-badge first" : "order-badge";
}

function setResultBadge(id, result, playerIndex) {
  const el = document.getElementById(id);
  if (result == null || result < 0) {
    el.className = "result-badge";
    el.textContent = "";
    return;
  }
  if (result === 2) {
    el.className = "result-badge draw";
    el.textContent = t("draw");
    return;
  }
  const win = result === playerIndex;
  el.className = win ? "result-badge win" : "result-badge loss";
  el.textContent = win ? t("win") : t("lose");
}

/* ================================================================
   フェーズ2.6: カード移動アニメーション（差分エンジン + fly 演出）
   ライブモード版へ再統合。既存の hpColor / SELF_INDEX 等に依存。
   ================================================================ */
let animEnabled = true;
let prevCardRects = new Map(); // serial(string) -> ページ座標矩形（render 前スナップショット）
const ANIM_MS = 360;

const MOVE_TYPES = {
  play_active:     { ja: 'バトル場へ',           en: 'To Active',      cls: 'mt-play'    },
  play_bench:      { ja: 'ベンチへ',              en: 'To Bench',       cls: 'mt-play'    },
  play_stadium:    { ja: 'スタジアムへ',          en: 'Stadium',        cls: 'mt-play'    },
  attach_energy:   { ja: 'エネルギー付与',        en: 'Attach Energy',  cls: 'mt-energy'  },
  attach_tool:     { ja: 'どうぐ付与',            en: 'Attach Tool',    cls: 'mt-tool'    },
  discard:         { ja: 'トラッシュ',            en: 'Discard',        cls: 'mt-discard' },
  ko:              { ja: 'きぜつ',               en: 'Knocked Out',    cls: 'mt-ko'      },
  retreat_to_bench:{ ja: 'にげる（ベンチへ）',   en: 'Retreat',        cls: 'mt-retreat' },
  come_active:     { ja: 'バトル場へ（入替）',    en: 'Switch In',      cls: 'mt-retreat' },
  take_prize:      { ja: 'サイドを取る',          en: 'Take Prize',     cls: 'mt-prize'   },
  search_hand:     { ja: '手札へ（サーチ）',      en: 'Search to Hand', cls: 'mt-draw'    },
  draw:            { ja: 'ドロー',               en: 'Draw',           cls: 'mt-draw'    },
  deck_decrease:   { ja: '山札から',             en: 'From Deck',      cls: 'mt-draw'    },
  deck_increase:   { ja: '山札へ戻す',           en: 'To Deck',        cls: 'mt-draw'    },
  move:            { ja: '移動',                 en: 'Move',           cls: 'mt-move'    },
};

/** 盤面の全ゾーンからカードを収集: serial -> {card, zone, playerIdx, idx, parentSerial} */
function collectVisibleCards(visual) {
  const map = new Map();
  function add(card, zone, pi, idx, parentSerial) {
    if (!card || card.serial == null) return;
    if (map.has(card.serial)) return;
    map.set(card.serial, { card, zone, playerIdx: pi, idx, parentSerial: parentSerial ?? null });
  }
  (visual?.players || []).forEach((player, pi) => {
    (player.hand    || []).forEach((c, i) => add(c, 'hand',    pi, i));
    (player.discard || []).forEach((c, i) => add(c, 'discard', pi, i));
    (player.prize   || []).forEach((c, i) => add(c, 'prize',   pi, i));
    const stadium = Array.isArray(player.stadium) ? player.stadium : (player.stadium ? [player.stadium] : []);
    stadium.forEach((c, i) => add(c, 'stadium', pi, i));

    (player.active || []).forEach((slot, i) => {
      const poke = Array.isArray(slot) ? slot[0] : slot;
      if (!poke) return;
      add(poke, 'active', pi, i);
      (poke.energyCards || []).forEach((ec, j) => add(ec, 'active_energy', pi, i * 100 + j, poke.serial));
      (poke.tools       || []).forEach((tl, j) => add(tl, 'active_tool',   pi, i * 100 + j, poke.serial));
    });
    (player.bench || []).forEach((poke, i) => {
      if (!poke) return;
      add(poke, 'bench', pi, i);
      (poke.energyCards || []).forEach((ec, j) => add(ec, 'bench_energy', pi, i * 100 + j, poke.serial));
      (poke.tools       || []).forEach((tl, j) => add(tl, 'bench_tool',   pi, i * 100 + j, poke.serial));
    });
  });
  return map;
}

function getMoveType(fromZone, toZone) {
  if (fromZone === 'hand' && toZone === 'active')   return 'play_active';
  if (fromZone === 'hand' && toZone === 'bench')    return 'play_bench';
  if (fromZone === 'hand' && toZone === 'stadium')  return 'play_stadium';
  if (fromZone === 'hand' && (toZone === 'active_energy' || toZone === 'bench_energy')) return 'attach_energy';
  if (fromZone === 'hand' && (toZone === 'active_tool'   || toZone === 'bench_tool'))   return 'attach_tool';
  if (fromZone === 'hand' && toZone === 'discard')  return 'discard';
  if ((fromZone === 'active' || fromZone === 'bench') && toZone === 'discard') return 'ko';
  if (fromZone === 'active' && toZone === 'bench')  return 'retreat_to_bench';
  if (fromZone === 'bench'  && toZone === 'active') return 'come_active';
  if (fromZone === 'prize'  && toZone === 'hand')   return 'take_prize';
  if ((fromZone === 'discard' || fromZone === 'prize') && toZone === 'hand') return 'search_hand';
  return 'move';
}

/** 2フレーム間の差分から移動リストを返す */
function diffFrames(prevVisual, currVisual) {
  if (!prevVisual || !currVisual) return [];
  const moves = [];
  const prevCards = collectVisibleCards(prevVisual);
  const currCards = collectVisibleCards(currVisual);

  for (const [serial, prev] of prevCards) {
    const curr = currCards.get(serial);
    if (!curr) continue;
    if (curr.zone === prev.zone && curr.playerIdx === prev.playerIdx) continue;
    moves.push({ type: getMoveType(prev.zone, curr.zone), card: curr.card, from: prev, to: curr });
  }
  for (const [serial, curr] of currCards) {
    if (!prevCards.has(serial)) {
      moves.push({ type: 'draw', card: curr.card, from: null, to: curr });
    }
  }
  for (const [serial, prev] of prevCards) {
    if (!currCards.has(serial)) {
      moves.push({ type: 'vanish', card: prev.card, from: prev, to: null });
    }
  }
  for (let pi = 0; pi < 2; pi++) {
    const prevDeck = prevVisual.players?.[pi]?.deckCount ?? 0;
    const currDeck = currVisual.players?.[pi]?.deckCount ?? 0;
    const delta = currDeck - prevDeck;
    if (delta !== 0) {
      moves.push({ type: delta < 0 ? 'deck_decrease' : 'deck_increase', card: null, playerIdx: pi, delta: Math.abs(delta) });
    }
  }
  return moves;
}

/** 要素の矩形をページ座標（スクロール込み）で返す。 */
function pageRectOf(el) {
  const r = el.getBoundingClientRect();
  return { left: r.left + window.scrollX, top: r.top + window.scrollY, width: r.width, height: r.height };
}
/** ページ座標 → 現在のビューポート座標へ変換。 */
function toViewportRect(pr) {
  if (!pr) return null;
  return { left: pr.left - window.scrollX, top: pr.top - window.scrollY, width: pr.width, height: pr.height };
}
/** render() 直前に全カードチップの座標をページ座標で記録 */
function snapshotCardPositions() {
  prevCardRects = new Map();
  document.querySelectorAll('.card-chip[data-serial]').forEach(el => {
    const s = el.dataset.serial;
    if (s) prevCardRects.set(s, pageRectOf(el));
  });
}
/** 山札・トラッシュゾーン要素の矩形（ビューポート）を返す */
function zoneRectFor(kind, playerIdx) {
  const idMap = {
    deck:    playerIdx === SELF_INDEX ? 'selfDeckZone'    : 'opponentDeckZone',
    discard: playerIdx === SELF_INDEX ? 'selfDiscardCard' : 'opponentDiscardCard',
  };
  const el = document.getElementById(idMap[kind]);
  return el ? el.getBoundingClientRect() : null;
}
/** serial からポケモンの盤面チップ要素を返す */
function pokemonElBySerial(serial) {
  if (serial == null) return null;
  return document.querySelector(`.card-chip[data-serial="${serial}"]`);
}

/** 飛ぶカード（ゴースト）を fromRect → toRect で生成・再生。後で自動削除。 */
function spawnFlyGhost(fromRect, toRect, opts = {}) {
  if (!fromRect || !toRect) return;
  const ghost = document.createElement('div');
  ghost.className = 'fly-ghost' + (opts.cls ? ' ' + opts.cls : '');

  if (opts.faceDown) {
    ghost.classList.add('fly-back');
  } else if (opts.card) {
    const imageFile = toggleCardImages.checked ? cardManifest[opts.card.id] : null;
    if (imageFile) {
      ghost.style.backgroundImage = `url(./card_images/${encodeURIComponent(imageFile)})`;
      ghost.classList.add('fly-img');
    } else {
      ghost.classList.add('fly-text');
      ghost.textContent = cardDisplayName(opts.card);
    }
  }

  const sizeRect = opts.sizeFromTo ? toRect : fromRect;
  const w = sizeRect.width  || 56;
  const h = sizeRect.height || 78;
  const startLeft = fromRect.left + fromRect.width  / 2 - w / 2;
  const startTop  = fromRect.top  + fromRect.height / 2 - h / 2;
  ghost.style.left   = `${startLeft}px`;
  ghost.style.top    = `${startTop}px`;
  ghost.style.width  = `${w}px`;
  ghost.style.height = `${h}px`;

  const fromScale = opts.fromScale ?? 1;
  const toScale   = opts.toScale ?? 0.5;
  ghost.style.transform = `translate(0,0) scale(${fromScale})`;
  document.body.appendChild(ghost);

  const dx = (toRect.left + toRect.width  / 2) - (startLeft + w / 2);
  const dy = (toRect.top  + toRect.height / 2) - (startTop  + h / 2);
  const dur = opts.duration ?? ANIM_MS;

  requestAnimationFrame(() => {
    ghost.style.transition = `transform ${dur}ms cubic-bezier(0.4,0,0.2,1), opacity ${dur}ms ease-out`;
    ghost.style.transform = `translate(${dx}px,${dy}px) scale(${toScale})`;
    if (opts.fadeOut) ghost.style.opacity = '0';
  });
  if (opts.onArrive) setTimeout(opts.onArrive, dur);
  setTimeout(() => ghost.remove(), dur + 150);
}

/** カウントバッジのパルス */
function pulseCount(countId) {
  const el = document.getElementById(countId);
  if (!el) return;
  el.classList.remove('count-pulse');
  el.offsetHeight;
  el.classList.add('count-pulse');
  setTimeout(() => el.classList.remove('count-pulse'), 700);
}

/** 山札シャッフル演出: 山札ゾーンを振動 + リフルする小カード + バッジ */
function playShuffleAnimation(playerIdx) {
  const zoneId = playerIdx === SELF_INDEX ? 'selfDeckZone' : 'opponentDeckZone';
  const zone = document.getElementById(zoneId);
  if (!zone) return;

  // 振動
  zone.classList.remove('deck-shuffle');
  zone.offsetHeight;
  zone.classList.add('deck-shuffle');
  setTimeout(() => zone.classList.remove('deck-shuffle'), 700);

  // リフル小カード群
  const pileEl = zone.querySelector('.deck-pile') || zone;
  const r = pileEl.getBoundingClientRect();
  const fx = document.createElement('div');
  fx.className = 'shuffle-fx';
  fx.style.left = `${r.left + r.width / 2}px`;
  fx.style.top  = `${r.top + r.height / 2}px`;
  for (let i = 0; i < 6; i++) {
    const c = document.createElement('span');
    c.className = 'shuffle-card ' + (i % 2 === 0 ? 'sc-left' : 'sc-right');
    c.style.animationDelay = `${i * 55}ms`;
    fx.appendChild(c);
  }
  const badge = document.createElement('span');
  badge.className = 'shuffle-badge';
  badge.textContent = lang === 'ja' ? '🔀 シャッフル' : '🔀 Shuffle';
  fx.appendChild(badge);
  document.body.appendChild(fx);
  setTimeout(() => fx.remove(), 950);
}

/** render() 直後に各種アニメーションを再生 */
function playMoveAnimations(moves, logs) {
  if (!animEnabled) return;

  const flipSerials = new Set(
    moves.filter(m => m.card?.serial != null && m.from != null && m.to != null)
         .map(m => String(m.card.serial))
  );
  // ドロー（山札 → 手札）: serial -> move
  const drawMoves = new Map(
    moves.filter(m => m.type === 'draw' && m.card?.serial != null)
         .map(m => [String(m.card.serial), m])
  );

  // --- 1. 盤面内の見える移動は FLIP / ドローは山札から飛ばす / それ以外は出現フェード ---
  document.querySelectorAll('.card-chip[data-serial]').forEach(newEl => {
    const s = newEl.dataset.serial;
    if (!s) return;
    const prevRect = prevCardRects.get(s);

    if (prevRect && flipSerials.has(s)) {
      const newRect = pageRectOf(newEl);
      const dx = prevRect.left - newRect.left;
      const dy = prevRect.top  - newRect.top;
      if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
        newEl.style.transition = 'none';
        newEl.style.transform = `translate(${dx}px,${dy}px)`;
        newEl.offsetHeight;
        newEl.style.transition = `transform ${ANIM_MS}ms cubic-bezier(0.25,0.46,0.45,0.94)`;
        newEl.style.transform = '';
        newEl.addEventListener('transitionend', () => {
          newEl.style.transition = '';
          newEl.style.transform  = '';
        }, { once: true });
      }
    } else if (!prevRect && drawMoves.has(s)) {
      // ドロー: 山札ゾーン → 手札のカード位置へ飛ばす
      const m = drawMoves.get(s);
      const pi = m.to?.playerIdx ?? SELF_INDEX;
      const deckRect = zoneRectFor('deck', pi);
      const toRect = newEl.getBoundingClientRect();
      if (deckRect) {
        // 相手の手札が非公開なら裏向きで飛ばす（中身を明かさない）
        const reveal = (pi === SELF_INDEX) || toggleOpponentHand.checked;
        newEl.style.visibility = 'hidden';
        spawnFlyGhost(deckRect, toRect, {
          card: reveal ? m.card : null,
          faceDown: !reveal,
          sizeFromTo: true,
          fromScale: 0.34,
          toScale: 1,
          duration: ANIM_MS + 60,
          onArrive: () => {
            newEl.style.visibility = '';
            newEl.classList.add('card-arrive');
            setTimeout(() => newEl.classList.remove('card-arrive'), ANIM_MS);
          },
        });
        pulseCount(pi === SELF_INDEX ? 'selfDeckCount' : 'opponentDeckCount');
      } else {
        newEl.classList.add('card-arrive');
        setTimeout(() => newEl.classList.remove('card-arrive'), ANIM_MS + 100);
      }
    } else if (!prevRect && !flipSerials.has(s)) {
      newEl.classList.add('card-arrive');
      setTimeout(() => newEl.classList.remove('card-arrive'), ANIM_MS + 100);
    }
  });

  // --- 2. ゾーンへの出入りは fly ゴースト ---
  moves.forEach(m => {
    const s = m.card?.serial != null ? String(m.card.serial) : null;
    const fromRect = toViewportRect(s ? prevCardRects.get(s) : null);

    if (m.type === 'discard' || m.type === 'ko') {
      const pi = m.to?.playerIdx ?? m.from?.playerIdx ?? SELF_INDEX;
      spawnFlyGhost(fromRect, zoneRectFor('discard', pi), { card: m.card, fadeOut: true, toScale: 0.55 });
      pulseCount(pi === SELF_INDEX ? 'selfDiscardCount' : 'opponentDiscardCount');

    } else if (m.type === 'vanish') {
      const pi = m.from?.playerIdx ?? SELF_INDEX;
      const deckUp = moves.some(x => x.type === 'deck_increase' && x.playerIdx === pi);
      if (deckUp) {
        spawnFlyGhost(fromRect, zoneRectFor('deck', pi), { faceDown: true, fadeOut: true, toScale: 0.5 });
      }

    } else if (m.type === 'attach_energy' || m.type === 'attach_tool') {
      const targetEl = pokemonElBySerial(m.to?.parentSerial);
      const toRect = targetEl ? targetEl.getBoundingClientRect() : null;
      spawnFlyGhost(fromRect, toRect, {
        card: m.card,
        cls: m.type === 'attach_energy' ? 'fly-energy' : 'fly-tool',
        toScale: 0.4,
        fadeOut: true,
      });
      if (targetEl) {
        const cls = m.type === 'attach_energy' ? 'attach-glow' : 'attach-glow-tool';
        targetEl.classList.add(cls);
        setTimeout(() => targetEl.classList.remove(cls), ANIM_MS + 400);
      }
    }
  });

  // --- 3. 山札カウントのパルス（ドロー分は上で処理済み。戻し等をここで）---
  moves.forEach(m => {
    if (m.type === 'deck_increase') {
      pulseCount(m.playerIdx === SELF_INDEX ? 'selfDeckCount' : 'opponentDeckCount');
    }
  });

  // --- 4. シャッフル演出（logs から検出）---
  (logs || []).forEach(log => {
    if (log && log.type === 'Shuffle' && log.playerIndex != null) {
      playShuffleAnimation(log.playerIndex);
    }
  });
}

/** インスペクタの「このフレームの動き」セクションを更新 */
function renderMoveSummary(moves) {
  const el = document.getElementById("moveSummaryList");
  if (!el) return;

  const cardMoves  = moves.filter(m => m && m.card && m.type !== 'vanish');
  const countMoves = moves.filter(m => m && !m.card && m.delta);

  if (!cardMoves.length && !countMoves.length) {
    el.innerHTML = `<li class="no-moves">${escapeHtml(t("noMoves"))}</li>`;
    return;
  }

  const items = [];
  cardMoves.forEach(m => {
    const info = MOVE_TYPES[m.type] || MOVE_TYPES.move;
    const typeLabel = lang === 'ja' ? info.ja : info.en;
    const cardName  = cardDisplayName(m.card);
    const pi = m.to?.playerIdx ?? m.from?.playerIdx ?? SELF_INDEX;
    const playerLabel = pi === SELF_INDEX
      ? (lang === 'ja' ? '自分' : 'P0')
      : (lang === 'ja' ? '相手' : 'P1');
    items.push(`<li class="move-item ${info.cls}">
      <span class="move-type-badge">${escapeHtml(typeLabel)}</span>
      <span class="move-card-name">${escapeHtml(cardName)}</span>
      <span class="move-player">${escapeHtml(playerLabel)}</span>
    </li>`);
  });
  countMoves.forEach(m => {
    const info = MOVE_TYPES[m.type] || MOVE_TYPES.move;
    const typeLabel = lang === 'ja' ? info.ja : info.en;
    const pi = m.playerIdx ?? SELF_INDEX;
    const playerLabel = pi === SELF_INDEX
      ? (lang === 'ja' ? '自分' : 'P0')
      : (lang === 'ja' ? '相手' : 'P1');
    items.push(`<li class="move-item ${info.cls}">
      <span class="move-type-badge">${escapeHtml(typeLabel)}</span>
      <span class="move-card-name">×${m.delta}</span>
      <span class="move-player">${escapeHtml(playerLabel)}</span>
    </li>`);
  });
  el.innerHTML = items.join("");
}

function renderAction(frame) {
  const chosenAction = document.getElementById("chosenAction");
  if (!frame.action?.length) {
    chosenAction.textContent = t("noAction");
    return;
  }
  const optionMap = {};
  (frame.options || []).forEach((option) => { optionMap[option.index] = option; });
  chosenAction.innerHTML = frame.action
    .map((optionIndex) => {
      const option = optionMap[optionIndex];
      const label = option ? describeOptionLabel(option, frame) : "?";
      return `<div><strong>${optionIndex}</strong>: ${escapeHtml(label)}</div>`;
    })
    .join("");
}

function renderEffectSource(frame) {
  const box = document.getElementById("effectSourceBox");
  if (!box) return;

  const source = resolveEffectSource(frame);
  if (!source?.card) {
    box.textContent = t("noEffectSource");
    return;
  }

  box.innerHTML = `
    <div class="effect-source-copy">
      <div class="effect-source-kind">${escapeHtml(source.kind || "")}</div>
      <div class="effect-source-detail">${escapeHtml(source.detail || "")}</div>
    </div>
    <div class="effect-source-card-wrap">
      ${renderCard(source.card, { cardClass: "effect-source-card" })}
    </div>`;
}

function hideEffectOverlay() {
  const overlay = document.getElementById("effectOverlay");
  if (!overlay) return;
  if (effectOverlayTimer) {
    window.clearTimeout(effectOverlayTimer);
    effectOverlayTimer = null;
  }
  overlay.hidden = true;
  overlay.classList.remove("is-visible");
}

function showEffectOverlay(frame) {
  const overlay = document.getElementById("effectOverlay");
  if (!overlay) return;

  if (!shouldShowEffectOverlay(frame)) {
    lastEffectOverlayToken = `${viewerMode}:${frameIndex}:suppressed`;
    hideEffectOverlay();
    return;
  }

  const source = resolveEffectSource(frame);
  const sourceCard = source?.card;
  const token = sourceCard
    ? [
        viewerMode,
        frameIndex,
        frame?.actingPlayer ?? "?",
        sourceCard.serial ?? sourceCard.id ?? "?",
        JSON.stringify(frame?.action || []),
      ].join(":")
    : `${viewerMode}:${frameIndex}:none`;

  if (token === lastEffectOverlayToken) return;
  lastEffectOverlayToken = token;

  if (!sourceCard) {
    hideEffectOverlay();
    return;
  }

  overlay.innerHTML = `
    <div class="effect-overlay-copy">
      <div class="effect-overlay-kind">${escapeHtml(source.kind || "")}</div>
      <div class="effect-overlay-detail">${escapeHtml(source.detail || "")}</div>
    </div>
    <div class="effect-overlay-card-wrap">
      ${renderCard(sourceCard, { cardClass: "effect-overlay-card" })}
    </div>`;

  overlay.hidden = false;
  overlay.classList.remove("is-visible");
  void overlay.offsetWidth;
  overlay.classList.add("is-visible");

  if (effectOverlayTimer) window.clearTimeout(effectOverlayTimer);
  effectOverlayTimer = window.setTimeout(() => {
    overlay.hidden = true;
    overlay.classList.remove("is-visible");
    effectOverlayTimer = null;
  }, 2800);
}

// proposal の内部ラベル → 日本語の平易な説明。未知ラベルは生の文字列にフォールバック。
const PROPOSAL_LABELS = {
  ability: "特性を使う",
  ability_draw: "特性でドロー",
  attack: "ワザで攻撃",
  board_item: "グッズを使う",
  board_item_search: "グッズでサーチ",
  end: "ターンを終了",
  energy: "エネルギーをつける",
  evolve: "進化する",
  evolve_hariyama_gust: "ハリテヤマに進化（呼び出し）",
  pokemon_play: "ポケモンを出す",
  retreat: "にげる",
  retreat_tool_for_swap: "入れ替えのためにげる",
  stadium: "スタジアムを貼る",
  tool: "どうぐをつける",
  tool_enables_ko: "どうぐでKO圏内に",
  draw_or_search_supporter_search: "サポートでサーチ",
  draw_or_search_supporter_discard_draw: "サポートで引き直し",
  合法手フォールバック: "合法手フォールバック",
};

/** proposal ラベルを表示用に整形。JP は辞書引き、未知/EN は生ラベル。 */
function reasonLabel(label) {
  if (lang === "ja" && PROPOSAL_LABELS[label]) return PROPOSAL_LABELS[label];
  return String(label ?? "?");
}

/** B層: 意思決定理由（frame.trace）を表示。無ければ「未提供」とだけ出す。 */
function renderReasonTrace(frame) {
  const box = document.getElementById("reasonTraceBox");
  if (!box) return;

  const records = Array.isArray(frame.trace) ? frame.trace : [];
  if (!records.length) {
    box.innerHTML = `<div class="reason-empty">${escapeHtml(t("noReasonTrace"))}</div>`;
    return;
  }

  const fmtScore = (s) => (s == null ? "" : `<span class="reason-score">${escapeHtml(String(s))}</span>`);

  box.innerHTML = records.map((rec) => {
    const alts = Array.isArray(rec.alternatives) ? rec.alternatives : [];
    const altHtml = alts.length
      ? `<div class="reason-alts">
           <div class="reason-alts-label">${escapeHtml(t("reasonAlternatives"))}</div>
           ${alts.map((a) => `<div class="reason-alt">
              <span class="reason-alt-label">${escapeHtml(reasonLabel(a.label))}</span>
              ${fmtScore(a.score)}
            </div>`).join("")}
         </div>`
      : "";
    const notesHtml = rec.notes
      ? `<div class="reason-notes">${escapeHtml(String(rec.notes))}</div>`
      : "";
    const ctxLabel = ctxName(rec.context);
    return `<div class="reason-record">
      <div class="reason-ctx">${escapeHtml(String(ctxLabel))}</div>
      <div class="reason-chosen">
        <span class="reason-badge">${escapeHtml(t("reasonChosen"))}</span>
        <span class="reason-chosen-label">${escapeHtml(reasonLabel(rec.chosenLabel))}</span>
        ${fmtScore(rec.chosenScore)}
      </div>
      ${notesHtml}
      ${altHtml}
    </div>`;
  }).join("");
}

function renderSearchTrace(frame) {
  const box = document.getElementById("searchTraceBox");
  const trace = frame.searchTrace;
  if (!trace) {
    box.textContent = t("noSearchTrace");
    return;
  }
  const rows = [
    ["changed", trace.changed],
    ["fallback", JSON.stringify(trace.fallback ?? [])],
    ["result", JSON.stringify(trace.result ?? [])],
    ["iterations", trace.iterations ?? "?"],
    ["elapsed", `${trace.elapsedSec ?? "?"}s / ${trace.budgetSec ?? "?"}s`],
    ["candidates", trace.candidateCount ?? "?"],
    ["reason", trace.reason ?? "?"],
  ];
  box.innerHTML = rows
    .map(([key, value]) => `<div class="trace-row"><span>${escapeHtml(String(key))}</span><strong>${escapeHtml(String(value))}</strong></div>`)
    .join("");
}

function renderOptions(frame) {
  const optionsList = document.getElementById("optionsList");
  const selected = new Set(frame.action || []);
  const visible = visibleOptions(frame);
  optionsList.innerHTML = visible
    .map((option) => {
      const classes = [];
      if (selected.has(option.index)) classes.push("selected-option");
      if (isLiveMode()) classes.push("live-option");
      return `<li class="${classes.join(" ")}" data-option-index="${option.index}"><strong>${option.index}</strong> ${escapeHtml(describeOptionLabel(option, frame))}</li>`;
    })
    .join("");

  if (!visible.length) {
    optionsList.innerHTML = `<li>${escapeHtml(lang === "ja" ? "この絞り込みでは候補がありません。" : "No options match the current filter.")}</li>`;
  }
}

function renderLogs(logs) {
  const logsList = document.getElementById("logsList");
  if (!logs.length) {
    logsList.innerHTML = `<div class="log-entry">${escapeHtml(t("noLogs"))}</div>`;
    return;
  }
  logsList.innerHTML = logs
    .map((log) => `<div class="log-entry">${escapeHtml(JSON.stringify(log))}</div>`)
    .join("");
}

// player1(相手)の手番のフレームには opponentKnowledgeDebug が無い(null)ため、
// 現在フレームより前方向に遡って直近の値を探す（Prev/Next やスライダーでの飛び移動でも
// 表示が空白にならないようにするため）。ライブモードのフレームには常に null（未対応）。
function findLatestOpponentKnowledgeDebug(frames, index) {
  if (!Array.isArray(frames)) return null;
  for (let i = Math.min(index, frames.length - 1); i >= 0; i -= 1) {
    const debug = frames[i]?.opponentKnowledgeDebug;
    if (debug) {
      return { debug, atStep: frames[i].stepIndex };
    }
  }
  return null;
}

function renderOpponentKnowledge(entry) {
  const statusEl = document.getElementById("opponentKnowledgeStatus");
  const observedEl = document.getElementById("opponentKnowledgeObserved");
  const zonesEl = document.getElementById("opponentKnowledgeZones");
  const diffEl = document.getElementById("opponentKnowledgeDiff");
  if (!statusEl || !observedEl || !zonesEl || !diffEl) return;

  if (!entry) {
    statusEl.className = "diagnostic-status";
    statusEl.textContent = t("noObservationYet");
    observedEl.innerHTML = "";
    zonesEl.innerHTML = "";
    diffEl.innerHTML = "";
    return;
  }

  const { debug, atStep } = entry;
  const { features, diff } = debug;
  const nameToCardIds = features.name_to_card_ids || {};
  const mismatchCount = diff.missing.length + diff.extra.length + diff.mismatched.length;

  statusEl.className = `diagnostic-status ${mismatchCount === 0 ? "diagnostic-ok" : "diagnostic-bad"}`;
  statusEl.textContent =
    mismatchCount === 0
      ? (lang === "ja" ? `${atStep} 手目時点: 神視点と一致（差分なし）` : `As of step ${atStep}: matches ground truth (no mismatch).`)
      : (lang === "ja" ? `${atStep} 手目時点: 神視点との差分が ${mismatchCount} 件` : `As of step ${atStep}: ${mismatchCount} mismatch(es) vs ground truth.`);

  const observedEntries = Object.entries(features.observed_cards || {});
  observedEl.innerHTML = observedEntries.length
    ? observedEntries
        .map(([name, count]) => `<span class="chip">${escapeHtml(localizedZoneName(name, nameToCardIds))} &times;${count}</span>`)
        .join("")
    : `<span class="chip chip-empty">${escapeHtml(t("noCardsObservedYet"))}</span>`;

  const zoneEntries = Object.entries(features.zone_cards || {});
  zonesEl.innerHTML = zoneEntries.length
    ? zoneEntries
        .map(
          ([zone, names]) => `
        <div class="zone-block">
          <div class="zone-block-title">${escapeHtml(zoneLabel(zone))}</div>
          <div class="chip-list">${names.map((name) => `<span class="chip">${escapeHtml(localizedZoneName(name, nameToCardIds))}</span>`).join("")}</div>
        </div>`
        )
        .join("")
    : `<div class="zone-block-title">${escapeHtml(t("noCardsVisible"))}</div>`;

  const MISMATCH_KIND_LABEL = { card_id: { ja: "カードID", en: "card_id" }, zone: { ja: "ゾーン", en: "zone" } };
  const formatMismatchValue = (kind, value) => (kind === "zone" ? zoneLabel(value) : value);
  const diagnosticRows = [
    ...diff.missing.map((item) => ({
      kind: "missing",
      text: lang === "ja"
        ? `観測漏れ: ${localizedObservedName(item.name, item.card_id)} (id=${item.card_id}, serial=${item.serial}) は本来「${zoneLabel(item.zone)}」で見えているはずですが観測できていません`
        : `missing: ${item.name} (id=${item.card_id}, serial=${item.serial}) expected in "${item.zone}" but was not observed`,
    })),
    ...diff.extra.map((item) => ({
      kind: "extra",
      text: lang === "ja"
        ? `過剰観測: ${localizedObservedName(item.name, item.card_id)} (id=${item.card_id}, serial=${item.serial}) を「${zoneLabel(item.zone)}」として観測済みですが、神視点では一致しません`
        : `extra: ${item.name} (id=${item.card_id}, serial=${item.serial}) tracked as "${item.zone}" but ground truth disagrees`,
    })),
    ...diff.mismatched.map((item) => {
      const kindLabel = MISMATCH_KIND_LABEL[item.kind] || { ja: item.kind, en: item.kind };
      return {
        kind: "mismatched",
        text: lang === "ja"
          ? `${lang === "ja" ? kindLabel.ja : kindLabel.en}の不一致: serial=${item.serial} 期待値「${formatMismatchValue(item.kind, item.expected)}」に対して実際は「${formatMismatchValue(item.kind, item.actual)}」`
          : `${kindLabel.en} mismatch: serial=${item.serial} expected "${formatMismatchValue(item.kind, item.expected)}" but got "${formatMismatchValue(item.kind, item.actual)}"`,
      };
    }),
  ];
  diffEl.innerHTML = diagnosticRows.length
    ? diagnosticRows.map((row) => `<div class="diagnostic-row diagnostic-${row.kind}">${escapeHtml(row.text)}</div>`).join("")
    : `<div class="diagnostic-row diagnostic-ok">${escapeHtml(t("noMismatchAtStep"))}</div>`;
}

function renderDeckPredictor(entry) {
  const topEl = document.getElementById("deckPredictorTop");
  const evidenceEl = document.getElementById("deckPredictorEvidence");
  const candidatesEl = document.getElementById("deckPredictorCandidates");
  if (!topEl || !evidenceEl || !candidatesEl) return;

  const prediction = entry?.debug?.prediction;
  if (!prediction) {
    topEl.className = "predictor-top";
    topEl.innerHTML = `<div class="diagnostic-status">この replay には予測結果がありません（古い replay か、予測器が無効）。新しく生成すると出ます。</div>`;
    evidenceEl.innerHTML = "";
    candidatesEl.innerHTML = "";
    return;
  }
  if (prediction.error) {
    topEl.innerHTML = `<div class="diagnostic-status diagnostic-bad">予測器エラー: ${escapeHtml(String(prediction.error))}</div>`;
    evidenceEl.innerHTML = "";
    candidatesEl.innerHTML = "";
    return;
  }

  const statusLabels = {
    confident: "確定",
    insufficient_evidence: "情報不足",
    ambiguous: "候補を絞り切れない",
    no_candidate: "候補なし",
  };
  const status = prediction.status || null;
  const isUnknown = prediction.deck_type === "unknown";
  const headlineName = isUnknown && prediction.top_candidate
    ? prediction.top_candidate
    : (prediction.display_name || prediction.deck_type || "unknown");
  const headlineSub = isUnknown && prediction.top_candidate ? "最有力候補" : null;
  const matchRateValue = prediction.match_rate ?? prediction.confidence ?? 0;
  const matchRatePct = Math.round(matchRateValue * 100);
  const confidentScore = prediction.confident_score;
  const scoreText = confidentScore != null
    ? `score ${prediction.score ?? 0} / 基準 ${confidentScore}`
    : `score ${prediction.score ?? 0}`;
  const normalizedScoreText = prediction.normalized_score != null
    ? `normalized ${Number(prediction.normalized_score).toFixed(2)}`
    : null;
  const marginText = prediction.margin != null
    ? `差分 ${Number(prediction.margin).toFixed(2)}`
    : null;
  const evidenceCountText = prediction.evidence_count != null
    ? `根拠数 ${prediction.evidence_count}`
    : null;
  const headlineMeta = [
    status ? `判定状態: ${statusLabels[status] || status}` : null,
    `判定基準到達率: ${matchRatePct}%`,
    scoreText,
    normalizedScoreText,
    marginText,
    evidenceCountText,
  ].filter(Boolean).join(" / ");
  topEl.className = "predictor-top";
  topEl.innerHTML = `
    <div class="predictor-headline ${isUnknown ? "is-unknown" : "is-known"}">
      <span class="predictor-name">${escapeHtml(headlineName)}</span>
      <span class="predictor-conf">${escapeHtml(headlineMeta)}</span>
    </div>`;
  if (headlineSub) {
    topEl.innerHTML += `<div class="predictor-subhead">${escapeHtml(headlineSub)}</div>`;
  }

  const evidence = prediction.evidence || [];
  const selectedCandidateKey = topEl.dataset.selectedCandidateKey || null;
  const candidateItems = (prediction.candidates || []).map((c, i) => ({
    ...c,
    key: c.deck_type || c.display_name || `candidate-${i}`,
    evidence: Array.isArray(c.evidence) ? c.evidence : [],
    index: i,
  }));
  const defaultCandidate = candidateItems[0] || null;
  const selectedCandidate = candidateItems.find((item) => item.key === selectedCandidateKey) || defaultCandidate;
  const selectedEvidence = selectedCandidate?.evidence?.length ? selectedCandidate.evidence : evidence;
  const evidenceTitle = selectedCandidate
    ? `${escapeHtml(selectedCandidate.display_name || selectedCandidate.deck_type || "candidate")} の根拠`
    : "根拠カード";
  evidenceEl.innerHTML = selectedEvidence.length
    ? `
        <div class="predictor-ev-title">${evidenceTitle}</div>
        ${selectedEvidence.map((e) => `
          <div class="predictor-ev-row">
            <span class="predictor-ev-w">+${e.weight}</span>
            <span class="predictor-ev-card">${escapeHtml(e.card || "")}</span>
            <span class="predictor-ev-role">${escapeHtml(e.role || "")} / ${escapeHtml(e.zone || "")}</span>
          </div>`).join("")}
      `
    : `<div class="chip chip-empty">根拠カードなし</div>`;

  const candidateRows = candidateItems.map((c, i) => {
    const rowPct = Math.round(((c.match_rate ?? c.confidence ?? 0)) * 100);
    const rowMeta = `score ${c.score} / 基準 ${c.confident_score ?? "—"} ・ 到達率 ${rowPct}%`;
    const isSelected = selectedCandidate && selectedCandidate.key === c.key;
    return `
        <button class="predictor-cand-row ${isSelected ? "is-selected" : ""} ${i === 0 && !isUnknown ? "is-top" : ""}" data-candidate-key="${escapeHtml(c.key)}" type="button">
          <span class="predictor-cand-name">${escapeHtml(c.display_name || c.deck_type)}</span>
          <span class="predictor-cand-score">${escapeHtml(rowMeta)}</span>
        </button>`;
  });
  candidatesEl.innerHTML = candidateRows.length
    ? candidateRows.join("")
    : `<div class="chip chip-empty">候補なし</div>`;
  candidatesEl.querySelectorAll(".predictor-cand-row").forEach((btn) => {
    btn.addEventListener("click", () => {
      topEl.dataset.selectedCandidateKey = btn.dataset.candidateKey;
      renderDeckPredictor(entry);
    });
  });
  if (!selectedCandidateKey && defaultCandidate) {
    topEl.dataset.selectedCandidateKey = defaultCandidate.key;
  }
}

function renderStadium(stadium, selectedRefs, actionableRefs) {
  const container = document.getElementById("stadiumSlot");
  if (!stadium.length) {
    container.innerHTML = `<div class="empty-slot">Stadium<br>empty</div>`;
    return;
  }
  container.innerHTML = stadium
    .map((card, index) => {
      const key = `${AREA.STADIUM}:${SELF_INDEX}:${index}`;
      return renderCard(card, {
        selectionRef: key,
        highlighted: selectedRefs.has(key),
        selectable: actionableRefs?.has(key),
      });
    })
    .join("");
}

function renderPlayer(player, playerIndex, selectedRefs, actionableRefs, targets) {
  document.getElementById(targets.deckCountId).textContent = String(player?.deckCount ?? 0);
  document.getElementById(targets.discardCountId).textContent = String((player?.discard || []).length);
  renderConditions(targets.conditionsId, player);

  renderZone(document.getElementById(targets.activeId), player?.active || [], {
    area: AREA.ACTIVE,
    playerIndex,
    selectedRefs,
    actionableRefs,
    cardClass: "active-card",
    emptyLabel: "Active\nempty",
    flip: playerIndex === OPPONENT_INDEX,
  });
  renderZone(document.getElementById(targets.benchId), player?.bench || [], {
    area: AREA.BENCH,
    playerIndex,
    selectedRefs,
    actionableRefs,
    emptyLabel: "Bench\nempty",
    flip: playerIndex === OPPONENT_INDEX,
  });
  renderPrize(document.getElementById(targets.prizeId), player?.prize || [], playerIndex);
}

function renderConditions(id, player) {
  const el = document.getElementById(id);
  el.innerHTML = CONDITION_FLAGS.filter(([flag]) => player?.[flag])
    .map(([, label]) => `<span class="condition-pill">${escapeHtml(label)}</span>`)
    .join("");
}

function renderPrize(container, cards, playerIndex) {
  const showContents = togglePrize.checked;
  const slots = [];
  for (let i = 0; i < 6; i += 1) {
    const card = cards[i];
    if (!card || !showContents) {
      slots.push(`<div class="card-back prize-card"></div>`);
    } else {
      slots.push(renderCard(card, { prize: true, selectionRef: `${AREA.PRIZE}:${playerIndex}:${i}` }));
    }
  }
  container.innerHTML = slots.join("");
}

function renderZone(container, cards, config) {
  const normalized = Array.isArray(cards) ? cards : [];
  if (!normalized.length) {
    container.innerHTML = `<div class="empty-slot">${escapeHtml(config.emptyLabel).replace(/\n/g, "<br>")}</div>`;
    return;
  }
  container.innerHTML = normalized.map((card, index) => {
    const key = `${config.area}:${config.playerIndex}:${index}`;
    return renderCard(card, {
      highlighted: config.selectedRefs?.has(key),
      selectable: config.actionableRefs?.has(key),
      cardClass: config.cardClass,
      flip: config.flip,
      selectionRef: key,
    });
  }).join("");
}

function renderHand(containerId, countId, cards, config) {
  const container = document.getElementById(containerId);
  const normalized = Array.isArray(cards) ? cards : [];
  document.getElementById(countId).textContent = String(normalized.length);
  if (!normalized.length) {
    container.innerHTML = `<div class="empty-slot">Hand empty</div>`;
    return;
  }
  if (config.hidden) {
    container.innerHTML = normalized.map(() => `<div class="card-back hand-card"></div>`).join("");
    return;
  }
  container.innerHTML = normalized.map((card, index) => {
    const key = `${AREA.HAND}:${config.playerIndex}:${index}`;
    return renderCard(card, {
      highlighted: config.selectedRefs?.has(key),
      selectable: config.actionableRefs?.has(key),
      cardClass: "hand-card",
      selectionRef: key,
    });
  }).join("");
}

function renderCard(card, config = {}) {
  if (!card) {
    return `<div class="empty-slot">${escapeHtml(t("hiddenCard"))}</div>`;
  }

  const cardKey = String(cardKeyCounter++);
  cardRegistry[cardKey] = card;
  if (config.selectionRef) cardSelectionRefs[cardKey] = config.selectionRef;

  const classes = ["card-chip"];
  if (config.cardClass) classes.push(config.cardClass);
  if (config.prize) classes.push("prize-card");
  if (config.highlighted) classes.push("highlight");
  if (config.selectable) classes.push("selectable-live");
  if (config.selectionRef && liveFilterRef === config.selectionRef) classes.push("filtered-live");
  if (config.flip) classes.push("flip");

  const attrs = [
    `data-card-key="${escapeHtml(cardKey)}"`,
    `data-serial="${escapeHtml(String(card.serial ?? ""))}"`,
  ];
  if (config.selectionRef) attrs.push(`data-select-ref="${escapeHtml(config.selectionRef)}"`);

  const tags = [];
  if (card.hp != null) tags.push(`<span class="tag hp ${hpColor(card.hp, card.maxHp)}">HP ${card.hp}/${card.maxHp ?? card.hp}</span>`);
  const dots = energyDotsHtml(card);
  if (dots) tags.push(`<span class="tag energy">${dots}</span>`);
  if (Array.isArray(card.tools) && card.tools.length) tags.push(`<span class="tag">Tool ${card.tools.length}</span>`);
  if (Array.isArray(card.preEvolution) && card.preEvolution.length) tags.push(`<span class="tag">Prev ${card.preEvolution.length}</span>`);

  const textInner = `
    <div class="card-name">${escapeHtml(cardDisplayName(card))}</div>
    <div class="card-meta">id=${escapeHtml(String(card.id ?? "?"))}</div>
    <div class="card-tags">${tags.join("")}</div>`;

  const imageFile = toggleCardImages.checked ? cardManifest[card.id] : null;
  if (imageFile) {
    classes.push("has-img");
    const overlay = [];
    if (card.hp != null) overlay.push(`<span class="ov ov-hp ${hpColor(card.hp, card.maxHp)}">${card.hp}/${card.maxHp ?? card.hp}</span>`);
    if (dots) overlay.push(`<span class="ov ov-energies">${dots}</span>`);
    return `
      <article class="${classes.join(" ")}" ${attrs.join(" ")}>
        <img class="card-img" src="./card_images/${encodeURIComponent(imageFile)}?v=${ASSET_VERSION}" alt="${escapeHtml(card.name || "")}" loading="lazy" onerror="this.closest('.card-chip').classList.add('img-failed')" />
        <div class="card-overlay">${overlay.join("")}</div>
        <div class="card-fallback">${textInner}</div>
      </article>`;
  }

  return `<article class="${classes.join(" ")}" ${attrs.join(" ")}>${textInner}</article>`;
}

function renderLivePanel() {
  const live = liveInfo();
  const panel = document.getElementById("livePanel");
  if (!panel) return;
  panel.style.display = isLiveMode() ? "" : "none";
  if (!isLiveMode()) return;

  let status = t("liveInactive");
  if (liveBusyMessage) status = liveBusyMessage;
  else if (livePlaybackTimer) status = lang === "ja" ? "相手ターンを再生中です。" : "Replaying opponent turn.";
  else if (live.active && live.result != null) status = t("liveFinished");
  else if (live.active && live.humanTurn && live.minCount === 1 && live.maxCount === 1) status = lang === "ja"
    ? "候補を1回クリックするとそのまま実行されます。"
    : "Single-choice prompts execute immediately when clicked.";
  else if (live.active && live.humanTurn) status = t("liveReady");
  else if (live.active) status = t("liveWaitingCpu");
  liveStatusBox.textContent = status;

  liveSelectionMeta.innerHTML = [
    `${t("liveSelection")}: ${escapeHtml(JSON.stringify(liveSelection))}`,
    `${t("liveFilter")}: ${escapeHtml(liveFilterRef || t("liveNoFilter"))}`,
    `${escapeHtml(lang === "ja" ? "ターン終了や Yes/No などの全体操作は常に表示されます。" : "Global actions like End Turn and Yes/No always stay visible.")}`,
    `${escapeHtml(t("liveConfirmHint"))}`,
  ].join("<br>");

  liveConfirmButton.disabled = !live.active
    || !!liveBusyMessage
    || !live.humanTurn
    || (live.minCount === 1 && live.maxCount === 1)
    || liveSelection.length < live.minCount
    || liveSelection.length > live.maxCount;
  liveUndoButton.hidden = true;
  liveUndoButton.disabled = true;
  liveClearButton.disabled = !!liveBusyMessage || (!liveSelection.length && !liveFilterRef);
}

function openDiscardModal(playerIndex) {
  const player = currentPlayers[playerIndex] || {};
  const discard = player.discard || [];
  const modal = document.getElementById("discardModal");
  const title = document.getElementById("discardModalTitle");
  const content = document.getElementById("discardModalContent");
  title.textContent = `${playerIndex === SELF_INDEX ? "Player 0" : "Player 1"} Discard (${discard.length})`;
  if (!discard.length) {
    content.innerHTML = `<div class="empty-slot" style="width:100%;text-align:center;padding:32px;">Discard pile is empty</div>`;
  } else {
    content.innerHTML = discard.map((card) => renderCard(card, {})).join("");
  }
  modal.hidden = false;
}

function closeDiscardModal() {
  const modal = document.getElementById("discardModal");
  if (modal) modal.hidden = true;
}

function cardImageHtml(card, className = "card-detail-img") {
  const imageFile = cardManifest[card?.id];
  const name = cardDisplayName(card);
  if (imageFile) {
    return `<img class="${escapeHtml(className)}" src="./card_images/${encodeURIComponent(imageFile)}?v=${ASSET_VERSION}" alt="${escapeHtml(name)}" loading="lazy" />`;
  }
  return `
    <div class="${escapeHtml(className)} card-detail-img-missing">
      <span>${escapeHtml(name)}</span>
      <small>id=${escapeHtml(String(card?.id ?? "?"))}</small>
    </div>`;
}

function registerAttachedPreviewCard(card) {
  const key = String(attachedPreviewKeyCounter++);
  attachedPreviewRegistry[key] = card;
  return key;
}

function attachedKindLabel(card, fallback) {
  const type = cardTypeOf(card?.id);
  if (type === CARD_TYPE.SPECIAL_ENERGY) return lang === "ja" ? "特殊エネルギー" : "Special Energy";
  if (type === CARD_TYPE.BASIC_ENERGY) {
    const code = parseEnergyCode(card?.name);
    const info = ENERGY_TYPE[code] || ENERGY_TYPE.C;
    return lang === "ja" ? `基本${info.ja}エネルギー` : `Basic ${info.en} Energy`;
  }
  if (type === CARD_TYPE.TOOL) return lang === "ja" ? "ポケモンのどうぐ" : "Pokemon Tool";
  return fallback;
}

function attachedCardTile(card, fallbackKind) {
  const kind = attachedKindLabel(card, fallbackKind);
  const special = cardTypeOf(card?.id) === CARD_TYPE.SPECIAL_ENERGY ? " special-energy" : "";
  const previewKey = registerAttachedPreviewCard(card);
  return `
    <div class="attached-card-tile${special}" data-attached-card-key="${escapeHtml(previewKey)}" role="button" tabindex="0">
      ${cardImageHtml(card, "attached-card-img")}
      <div class="attached-card-info">
        <span class="attached-kind">${escapeHtml(kind)}</span>
        <strong>${escapeHtml(cardDisplayName(card))}</strong>
        <small>id=${escapeHtml(String(card?.id ?? "?"))}</small>
      </div>
    </div>`;
}

function attachedSectionHtml(title, cards, fallbackKind) {
  const normalized = Array.isArray(cards) ? cards.filter(Boolean) : [];
  if (!normalized.length) return "";
  return `
    <section class="attached-section">
      <h3>${escapeHtml(title)}</h3>
      <div class="attached-card-grid">
        ${normalized.map((card) => attachedCardTile(card, fallbackKind)).join("")}
      </div>
    </section>`;
}

function renderCardDetail(card) {
  if (!cardDetailContent) return;
  if (previewPinned && String(pinnedKey || "").startsWith("attached:")) unpinPreview();
  attachedPreviewRegistry = {};
  attachedPreviewKeyCounter = 0;
  if (!card) {
    cardDetailContent.innerHTML = `<div class="card-detail-empty">Click a card on the board.</div>`;
    return;
  }

  const tools = Array.isArray(card.tools) ? card.tools : [];
  const energyCards = Array.isArray(card.energyCards) ? card.energyCards : [];
  const preEvolution = Array.isArray(card.preEvolution) ? card.preEvolution : [];
  const rows = [
    ["name", cardDisplayName(card)],
    ["id", card.id ?? "?"],
  ];
  if (card.hp != null) rows.push(["HP", `${card.hp} / ${card.maxHp ?? card.hp}`]);
  if (energyCards.length) rows.push(["energy", `${energyCards.length}`]);
  if (tools.length) rows.push(["tools", `${tools.length}`]);

  const rowHtml = rows
    .map(([k, v]) => `<div class="card-detail-row"><span>${escapeHtml(k)}</span><strong>${escapeHtml(String(v))}</strong></div>`)
    .join("");
  const energyFallback = !energyCards.length && Array.isArray(card.energies) && card.energies.length
    ? `<section class="attached-section"><h3>${escapeHtml(lang === "ja" ? "付属エネルギー" : "Attached Energy")}</h3><div class="energy-summary">${energyDotsHtml(card)}<span>${escapeHtml(String(card.energies.length))}</span></div></section>`
    : "";
  const attachedHtml = [
    attachedSectionHtml(lang === "ja" ? "ポケモンのどうぐ" : "Pokemon Tools", tools, lang === "ja" ? "ポケモンのどうぐ" : "Pokemon Tool"),
    attachedSectionHtml(lang === "ja" ? "付属エネルギー" : "Attached Energy", energyCards, lang === "ja" ? "エネルギー" : "Energy"),
    energyFallback,
    attachedSectionHtml(lang === "ja" ? "進化元" : "Pre-evolution", preEvolution, lang === "ja" ? "進化元" : "Pre-evolution"),
  ].filter(Boolean).join("");
  const emptyAttached = attachedHtml
    ? ""
    : `<div class="card-detail-empty compact">${escapeHtml(lang === "ja" ? "ついているどうぐ・エネルギーはありません。" : "No attached tools or energy cards.")}</div>`;

  cardDetailContent.innerHTML = `
    <div class="card-detail-main">
      <div class="card-detail-photo">${cardImageHtml(card)}</div>
      <div class="card-detail-meta">
        <h2>${escapeHtml(cardDisplayName(card))}</h2>
        <div class="card-detail-rows">${rowHtml}</div>
      </div>
    </div>
    <div class="card-detail-attached">
      ${attachedHtml || emptyAttached}
    </div>`;
}

function getCardBySelectionRef(selectionRef, visual) {
  const parts = String(selectionRef || "").split(":").map((part) => Number(part));
  if (parts.length !== 3 || parts.some((part) => Number.isNaN(part))) return null;
  const [area, playerIndex, index] = parts;
  return getCardFromVisual(visual, area, index, playerIndex);
}

function refreshOpenCardDetail(visual) {
  if (activePanel !== "cardDetail" || !selectedCardDetailRef) return;
  const card = getCardBySelectionRef(selectedCardDetailRef, visual);
  renderCardDetail(card);
}

function openCardDetail(card, selectionRef = null) {
  selectedCardDetailRef = selectionRef;
  renderCardDetail(card);
  previewPinned = false;
  pinnedKey = null;
  cardPreview?.classList.remove("pinned");
  if (cardPreview) cardPreview.hidden = true;
  activePanel = "cardDetail";
  applyPanel();
}

function buildPreviewHTML(card) {
  const imageFile = cardManifest[card.id];
  const rows = [];
  rows.push(["name", cardDisplayName(card)]);
  rows.push(["id", card.id ?? "?"]);
  if (card.hp != null) rows.push(["HP", `${card.hp} / ${card.maxHp ?? card.hp}`]);
  if (Array.isArray(card.tools) && card.tools.length) rows.push(["tools", card.tools.map((tool) => cardDisplayName(tool)).join(", ")]);
  const img = imageFile ? `<img class="preview-img" src="./card_images/${encodeURIComponent(imageFile)}?v=${ASSET_VERSION}" alt="${escapeHtml(card.name || "")}" />` : "";
  const info = rows.map(([k, v]) => `<div class="preview-row"><span class="pk">${escapeHtml(k)}</span><span class="pv">${escapeHtml(String(v))}</span></div>`).join("");
  const hintText = lang === "ja" ? "クリックで詳細" : "Click for detail";
  const hint = `<div class="preview-hint">${escapeHtml(hintText)}</div>`;
  return `${img}<div class="preview-info">${info}</div>${hint}`;
}

const cardPreview = document.getElementById("cardPreview");

function chipFromEvent(event) {
  const target = event.target;
  return target?.closest ? target.closest(".card-chip[data-card-key]") : null;
}

function attachedTileFromEvent(event) {
  const target = event.target;
  return target?.closest ? target.closest("[data-attached-card-key]") : null;
}

function effectTileFromEvent(event) {
  const target = event.target;
  return target?.closest ? target.closest("[data-effect-card-key]") : null;
}

function positionPreview(x, y) {
  const pad = 18;
  const w = cardPreview.offsetWidth;
  const h = cardPreview.offsetHeight;
  let left = x + pad;
  let top = y + pad;
  if (left + w > window.innerWidth - 8) left = x - w - pad;
  if (left < 8) left = 8;
  if (top + h > window.innerHeight - 8) top = window.innerHeight - h - 8;
  if (top < 8) top = 8;
  cardPreview.style.left = `${left}px`;
  cardPreview.style.top = `${top}px`;
}

function showPreview(card, x, y, large = false) {
  if (!card) return;
  cardPreview.classList.toggle("large", large);
  cardPreview.innerHTML = buildPreviewHTML(card);
  cardPreview.hidden = false;
  positionPreview(x, y);
}

function unpinPreview() {
  previewPinned = false;
  pinnedKey = null;
  cardPreview.classList.remove("pinned", "large");
  cardPreview.hidden = true;
}

function setupCardPreview() {
  const boardEl = document.querySelector(".board");
  if (!boardEl) return;

  boardEl.addEventListener("mouseover", (event) => {
    if (previewPinned) return;
    const chip = chipFromEvent(event);
    if (chip) showPreview(cardRegistry[chip.dataset.cardKey], event.clientX, event.clientY);
  });

  boardEl.addEventListener("mousemove", (event) => {
    if (previewPinned || cardPreview.hidden) return;
    if (chipFromEvent(event)) positionPreview(event.clientX, event.clientY);
  });

  boardEl.addEventListener("mouseout", (event) => {
    if (previewPinned) return;
    const to = event.relatedTarget;
    if (!to || !(to.closest && to.closest(".card-chip[data-card-key]"))) {
      cardPreview.hidden = true;
    }
  });

  cardDetailContent?.addEventListener("mouseover", (event) => {
    if (previewPinned) return;
    const tile = attachedTileFromEvent(event);
    if (tile) showPreview(attachedPreviewRegistry[tile.dataset.attachedCardKey], event.clientX, event.clientY, true);
  });

  cardDetailContent?.addEventListener("mousemove", (event) => {
    if (previewPinned || cardPreview.hidden) return;
    if (attachedTileFromEvent(event)) positionPreview(event.clientX, event.clientY);
  });

  cardDetailContent?.addEventListener("mouseout", (event) => {
    if (previewPinned) return;
    const to = event.relatedTarget;
    if (!to || !(to.closest && to.closest("[data-attached-card-key]"))) {
      cardPreview.classList.remove("large");
      cardPreview.hidden = true;
    }
  });

  cardDetailContent?.addEventListener("click", (event) => {
    const tile = attachedTileFromEvent(event);
    if (!tile) return;
    event.stopPropagation();
    previewPinned = true;
    pinnedKey = `attached:${tile.dataset.attachedCardKey}`;
    cardPreview.classList.add("pinned");
    showPreview(attachedPreviewRegistry[tile.dataset.attachedCardKey], event.clientX, event.clientY, true);
  });

  cardDetailContent?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const tile = attachedTileFromEvent(event);
    if (!tile) return;
    event.preventDefault();
    event.stopPropagation();
    const rect = tile.getBoundingClientRect();
    previewPinned = true;
    pinnedKey = `attached:${tile.dataset.attachedCardKey}`;
    cardPreview.classList.add("pinned");
    showPreview(attachedPreviewRegistry[tile.dataset.attachedCardKey], rect.right, rect.top, true);
  });

  activeEffectsShelf?.addEventListener("mouseover", (event) => {
    if (previewPinned) return;
    const tile = effectTileFromEvent(event);
    if (tile) showPreview(effectPreviewRegistry[tile.dataset.effectCardKey], event.clientX, event.clientY, true);
  });

  activeEffectsShelf?.addEventListener("mousemove", (event) => {
    if (previewPinned || cardPreview.hidden) return;
    if (effectTileFromEvent(event)) positionPreview(event.clientX, event.clientY);
  });

  activeEffectsShelf?.addEventListener("mouseout", (event) => {
    if (previewPinned) return;
    const to = event.relatedTarget;
    if (!to || !(to.closest && to.closest("[data-effect-card-key]"))) {
      cardPreview.classList.remove("large");
      cardPreview.hidden = true;
    }
  });

  activeEffectsShelf?.addEventListener("click", (event) => {
    const tile = effectTileFromEvent(event);
    if (!tile) return;
    event.stopPropagation();
    previewPinned = true;
    pinnedKey = `effect:${tile.dataset.effectCardKey}`;
    cardPreview.classList.add("pinned");
    showPreview(effectPreviewRegistry[tile.dataset.effectCardKey], event.clientX, event.clientY, true);
  });

  activeEffectsShelf?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const tile = effectTileFromEvent(event);
    if (!tile) return;
    event.preventDefault();
    event.stopPropagation();
    const rect = tile.getBoundingClientRect();
    previewPinned = true;
    pinnedKey = `effect:${tile.dataset.effectCardKey}`;
    cardPreview.classList.add("pinned");
    showPreview(effectPreviewRegistry[tile.dataset.effectCardKey], rect.right, rect.top, true);
  });

  document.addEventListener("click", (event) => {
    const chip = chipFromEvent(event);
    if (chip) {
      const key = chip.dataset.cardKey;
      openCardDetail(cardRegistry[key], cardSelectionRefs[key] || null);
      return;
    }
    if (previewPinned && !event.target.closest?.("#cardPreview")) {
      unpinPreview();
    }
  });
}

function setupDiscardModal() {
  document.getElementById("opponentDiscardCard")?.addEventListener("click", () => openDiscardModal(OPPONENT_INDEX));
  document.getElementById("selfDiscardCard")?.addEventListener("click", () => openDiscardModal(SELF_INDEX));
  document.getElementById("discardModalClose")?.addEventListener("click", closeDiscardModal);
  document.querySelector(".discard-modal-backdrop")?.addEventListener("click", closeDiscardModal);
}

function setupInteractiveSelection() {
  document.getElementById("optionsList")?.addEventListener("click", async (event) => {
    const item = event.target.closest?.("[data-option-index]");
    if (!item) return;
    await toggleLiveOption(Number(item.dataset.optionIndex));
  });

  document.querySelector(".board")?.addEventListener("click", async (event) => {
    if (!isLiveMode()) return;
    const chip = event.target.closest?.("[data-select-ref]");
    if (!chip) return;
    const ref = chip.dataset.selectRef;
    if (actionableOptionsForRef(currentFrame(), ref).length) {
      event.preventDefault();
      event.stopPropagation();
      await handleLiveBoardSelection(ref);
    }
  });
}

function stepFrame(delta) {
  if (!replayData?.frames?.length) return;

  // 順方向1フレームのみアニメーション（逆・スライダーは即時描画）
  const doAnim = animEnabled && delta === 1;
  const prevVisual = doAnim ? (replayData.frames[frameIndex]?.visual?.current ?? null) : null;
  if (doAnim) snapshotCardPositions();

  frameIndex = Math.min(Math.max(frameIndex + delta, 0), replayData.frames.length - 1);
  render();

  if (doAnim && prevVisual) {
    const currFrame = replayData.frames[frameIndex];
    const currVisual = currFrame?.visual?.current ?? null;
    const logs = currFrame?.visual?.logs ?? [];
    const moves = diffFrames(prevVisual, currVisual);
    playMoveAnimations(moves, logs);
    renderMoveSummary(moves);
  }
}

function stopLivePlayback() {
  if (!livePlaybackTimer) return;
  window.clearInterval(livePlaybackTimer);
  livePlaybackTimer = null;
  autoplayButton.textContent = t("autoplay");
}

function playLiveSequenceToLatest(immediate = false) {
  if (!isLiveMode()) return;
  const latest = liveLatestFrameIndex();
  if (frameIndex >= latest) {
    frameIndex = latest;
    render();
    return;
  }
  stopLivePlayback();
  autoplayButton.textContent = t("pause");
  if (immediate && frameIndex < latest) {
    stepFrame(1);
  }
  if (frameIndex >= liveLatestFrameIndex()) {
    stopLivePlayback();
    render();
    return;
  }
  livePlaybackTimer = window.setInterval(() => {
    const target = liveLatestFrameIndex();
    if (frameIndex >= target) {
      stopLivePlayback();
      render();
      return;
    }
    stepFrame(1);
  }, Math.max(120, Math.round(1200 / replaySpeed)));
}

function toggleAutoplay() {
  if (isLiveMode()) {
    if (livePlaybackTimer) {
      stopLivePlayback();
      return;
    }
    playLiveSequenceToLatest();
    return;
  }
  if (autoplayTimer) {
    window.clearInterval(autoplayTimer);
    autoplayTimer = null;
    autoplayButton.textContent = t("autoplay");
    return;
  }
  autoplayButton.textContent = t("pause");
  autoplayTimer = window.setInterval(() => {
    if (!replayData || frameIndex >= replayData.frames.length - 1) {
      toggleAutoplay();
      return;
    }
    stepFrame(1);
  }, Math.max(120, Math.round(1200 / replaySpeed)));
}

async function loadCardNamesJp() {
  try {
    const response = await fetch(`./card_names_jp.json?v=${ASSET_VERSION}`);
    if (response.ok) cardNamesJp = await response.json();
  } catch (_) {}
}

async function loadAttackNamesJp() {
  try {
    const response = await fetch(`./attack_names_jp.json?v=${ASSET_VERSION}`);
    if (response.ok) attackNamesJp = await response.json();
  } catch (_) {
    attackNamesJp = { byId: {}, byName: {} };
  }
}

async function loadCardTypes() {
  try {
    const response = await fetch(`./card_types.json?v=${ASSET_VERSION}`);
    if (response.ok) cardTypes = await response.json();
  } catch (_) { cardTypes = {}; }
}

async function loadCardManifest() {
  try {
    const response = await fetch(`./card_images/manifest.json?v=${ASSET_VERSION}`);
    if (response.ok) cardManifest = await response.json();
  } catch (_) {
    cardManifest = {};
  }
  if (!Object.keys(cardManifest).length) {
    toggleCardImages.checked = false;
    toggleCardImages.disabled = true;
    toggleCardImages.closest(".toggle")?.setAttribute("title", t("noCardImages"));
  }
}

async function switchMode(nextMode) {
  viewerMode = nextMode;
  liveSelection = [];
  liveFilterRef = null;
  stopLivePlayback();
  if (autoplayTimer) toggleAutoplay();
  if (isLiveMode()) {
    await ensureLiveMatchStarted();
  } else {
    await loadSelectedReplay();
  }
}

function requestedInitialMode() {
  const mode = urlParams.get("mode");
  return mode === "live" ? "live" : "replay";
}

function requestedInitialCpu() {
  const cpu = urlParams.get("cpu");
  return cpu === "random" ? "random" : "self";
}

function requestedReplayName() {
  return urlParams.get("replay") || "";
}

function syncReplayUrl() {
  const next = new URL(window.location.href);
  if (viewerMode === "replay" && replaySelect.value) {
    next.searchParams.set("replay", replaySelect.value);
    next.searchParams.set("frame", String(frameIndex));
  } else {
    next.searchParams.delete("replay");
    next.searchParams.delete("frame");
  }
  window.history.replaceState({}, "", next);
}

reloadButton.addEventListener("click", async () => {
  if (isLiveMode()) {
    stopLivePlayback();
    await loadLiveState();
    return;
  }
  await loadReplayList();
  await loadSelectedReplay();
});

modeSelect.addEventListener("change", async (event) => {
  stopLivePlayback();
  await switchMode(event.target.value);
});

replaySelect.addEventListener("change", async () => {
  await loadSelectedReplay();
});

liveStartButton.addEventListener("click", async () => {
  await startLiveMatch();
});

liveUndoButton.addEventListener("click", async () => {
  await undoLiveTurn();
});

liveConfirmButton.addEventListener("click", async () => {
  await submitLiveAction();
});

liveClearButton.addEventListener("click", clearLiveSelection);

prevButton.addEventListener("click", () => stepFrame(-1));
nextButton.addEventListener("click", () => stepFrame(1));
autoplayButton.addEventListener("click", toggleAutoplay);
speedSelect?.addEventListener("change", () => {
  replaySpeed = Number(speedSelect.value || "1") || 1;
  if (livePlaybackTimer) {
    stopLivePlayback();
    playLiveSequenceToLatest();
  }
  if (autoplayTimer) {
    window.clearInterval(autoplayTimer);
    autoplayTimer = null;
    toggleAutoplay();
  }
});
frameSlider.addEventListener("input", (event) => {
  frameIndex = Number(event.target.value);
  render();
});
toggleOpponentHand.addEventListener("change", render);
togglePrize.addEventListener("change", render);
toggleCardImages.addEventListener("change", render);
toggleAnimation?.addEventListener("change", (e) => { animEnabled = e.target.checked; });

// ---------- On-demand drawer (tabbed panels) ----------
// 盤面を主役にするため、盤面以外の情報はデフォルト非表示。タブを押した時だけ
// 右からドロワーを開き、1パネルだけ大きく表示する。同じタブを再度押すと閉じる。
const PANEL_TITLES = {
  summary: "Info",
  action: "Action / Options",
  logs: "Logs",
  debug: "Debug",
  live: "Human vs CPU",
  settings: "Settings",
  cardDetail: "Card Detail",
};
let activePanel = null;

function applyPanel() {
  const open = activePanel !== null;
  // ドロワーをトップバーの真下に配置し、開いている間は stage を縮めて盤面と重ならないようにする。
  const topbar = document.querySelector(".topbar");
  const topH = topbar ? Math.round(topbar.getBoundingClientRect().height) : 56;
  drawer.style.top = `${topH}px`;
  drawer.style.height = `calc(100vh - ${topH}px)`;
  document.body.classList.toggle("drawer-open", open);
  drawer.classList.toggle("open", open);
  if (open && drawerTitle) drawerTitle.textContent = PANEL_TITLES[activePanel] || "";
  panelSections.forEach((sec) => sec.classList.toggle("panel-active", sec.dataset.panel === activePanel));
  panelButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.panelBtn === activePanel));
}

function setPanel(name) {
  activePanel = activePanel === name ? null : name;
  applyPanel();
}

panelButtons.forEach((btn) => btn.addEventListener("click", () => setPanel(btn.dataset.panelBtn)));
drawerClose?.addEventListener("click", () => { activePanel = null; applyPanel(); });

// ---------- Debug sub-views (extensible) ----------
// Debug パネル内の切り替え。新しいデバッグビューを足すときは、この配列に1行と、
// index.html の .debug-views に対応する <div class="debug-view" data-debug-view="<id>"> を足すだけ。
// 表示内容の描画が要るビューは render() 側でそのビューの関数を呼ぶ（未実装ビューは静的で可）。
const DEBUG_VIEWS = [
  { id: "opponentKnowledge", labelKey: "debugViewOpponentKnowledge" },
  { id: "deckPredictor", labelKey: "debugViewDeckPredictor" },
];
let activeDebugView = DEBUG_VIEWS[0].id;

function applyDebugView() {
  document.querySelectorAll(".debug-view").forEach((view) => {
    view.classList.toggle("debug-view-active", view.dataset.debugView === activeDebugView);
  });
  document.querySelectorAll("#debugSubtabs .subtab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.debugView === activeDebugView);
  });
}

function buildDebugSubtabs() {
  const bar = document.getElementById("debugSubtabs");
  if (!bar) return;
  bar.innerHTML = DEBUG_VIEWS
    .map((v) => `<button class="subtab-btn" data-debug-view="${v.id}">${escapeHtml(t(v.labelKey))}</button>`)
    .join("");
  bar.querySelectorAll(".subtab-btn").forEach((btn) => {
    btn.addEventListener("click", () => { activeDebugView = btn.dataset.debugView; applyDebugView(); });
  });
  applyDebugView();
}

// ---------- First-run guided tour (mobile-style coachmarks) ----------
// 各 UI 要素をスポットライトして「これは何をするボタンか」を1つずつ説明する。
// ステップを増やすときは selector と ja/en の文言を1行足すだけ。
const ONBOARDING_KEY = "brv_onboarded_v1";
const TOUR_STEPS = [
  { selector: ".topbar-group.center", ja: "再生コントロール。前へ／次へやスライダーでフレームを1手ずつ動かします。自動再生も。", en: "Playback controls. Step through frames with Prev/Next or the slider (or Autoplay)." },
  { selector: "#replaySelect", ja: "見たいリプレイをここで選びます。", en: "Pick which replay to watch here." },
  { selector: '[data-panel-btn="summary"]', ja: "盤面以外の情報（サマリー・行動・選択肢・ログ）はこれらのタブから、必要な時だけ開きます。", en: "Open info panels (summary, action, options, logs) from these tabs — only when you need them." },
  { selector: '[data-panel-btn="debug"]', ja: "Debug は相手の観測情報やデッキ予測などの検証用ビュー。中はサブタブで切り替えます。", en: "Debug holds verification views (opponent knowledge, deck predictor). Switch with its sub-tabs." },
  { selector: '[data-panel-btn="settings"]', ja: "⚙ は表示設定と、リプレイの新規生成（コマンド不要）。", en: "⚙ has display settings and one-click replay generation (no command line)." },
  { selector: "#helpButton", ja: "この案内は ? からいつでも見返せます。", en: "Replay this tour anytime from the ? button." },
];
const tourEl = document.getElementById("tour");
const tourHighlight = document.getElementById("tourHighlight");
const tourTip = document.getElementById("tourTip");
const tourTextEl = document.getElementById("tourText");
const tourProgress = document.getElementById("tourProgress");
const tourPrevBtn = document.getElementById("tourPrev");
const tourNextBtn = document.getElementById("tourNext");
const tourSkipBtn = document.getElementById("tourSkip");
let tourIndex = 0;

function positionTour(step) {
  const el = document.querySelector(step.selector);
  if (!el) return false;
  const r = el.getBoundingClientRect();
  const pad = 6;
  tourHighlight.style.left = `${r.left - pad}px`;
  tourHighlight.style.top = `${r.top - pad}px`;
  tourHighlight.style.width = `${r.width + pad * 2}px`;
  tourHighlight.style.height = `${r.height + pad * 2}px`;
  // place the tooltip below the target if it fits, otherwise above.
  const tipW = tourTip.offsetWidth || 280;
  const tipH = tourTip.offsetHeight || 130;
  let top = r.bottom + 12;
  if (top + tipH > window.innerHeight - 12) top = Math.max(12, r.top - tipH - 12);
  const left = Math.min(Math.max(r.left, 12), window.innerWidth - tipW - 12);
  tourTip.style.top = `${top}px`;
  tourTip.style.left = `${left}px`;
  return true;
}

function showTourStep(i) {
  if (i < 0 || i >= TOUR_STEPS.length) return endTour(true);
  // skip steps whose target is not present (e.g. hidden in a mode)
  if (!document.querySelector(TOUR_STEPS[i].selector)) {
    return showTourStep(i > tourIndex ? i + 1 : i - 1);
  }
  tourIndex = i;
  const step = TOUR_STEPS[i];
  tourTextEl.textContent = lang === "ja" ? step.ja : step.en;
  tourProgress.textContent = `${i + 1} / ${TOUR_STEPS.length}`;
  tourPrevBtn.style.visibility = i === 0 ? "hidden" : "visible";
  const last = i === TOUR_STEPS.length - 1;
  tourNextBtn.textContent = last ? (lang === "ja" ? "完了" : "Done") : (lang === "ja" ? "次へ" : "Next");
  tourSkipBtn.textContent = lang === "ja" ? "スキップ" : "Skip";
  positionTour(step);
}

function startTour() {
  if (!tourEl) return;
  tourEl.hidden = false;
  showTourStep(0);
}
function endTour(markSeen) {
  if (tourEl) tourEl.hidden = true;
  if (markSeen) { try { localStorage.setItem(ONBOARDING_KEY, "1"); } catch (_) { /* ignore */ } }
}

tourNextBtn?.addEventListener("click", () => {
  if (tourIndex >= TOUR_STEPS.length - 1) endTour(true);
  else showTourStep(tourIndex + 1);
});
tourPrevBtn?.addEventListener("click", () => showTourStep(tourIndex - 1));
tourSkipBtn?.addEventListener("click", () => endTour(true));
window.addEventListener("resize", () => {
  if (tourEl && !tourEl.hidden) positionTour(TOUR_STEPS[tourIndex]);
});
document.getElementById("helpButton")?.addEventListener("click", startTour);

// ---------- Generate a replay from the UI (no command line needed) ----------
const generateButton = document.getElementById("generateButton");
const generatePlayerPolicy = document.getElementById("generatePlayerPolicy");
const generateOpponent = document.getElementById("generateOpponent");
const generateSeed = document.getElementById("generateSeed");
const generateStatus = document.getElementById("generateStatus");
const generatePlayerDeck = document.getElementById("generatePlayerDeck");
const generateOpponentDeck = document.getElementById("generateOpponentDeck");
let deckOptionsLoaded = false;

function populateDeckSelect(select, decks) {
  if (!select) return;
  const current = select.value;
  select.innerHTML = "";
  decks.forEach((deck) => {
    const option = document.createElement("option");
    option.value = deck.value;
    option.textContent = deck.label || deck.value;
    select.append(option);
  });
  if (current && decks.some((deck) => deck.value === current)) {
    select.value = current;
  }
}

async function loadDeckOptions() {
  const decks = await fetchJson("/api/decks");
  populateDeckSelect(livePlayerDeckSelect, decks);
  populateDeckSelect(liveOpponentDeckSelect, decks);
  populateDeckSelect(generatePlayerDeck, decks);
  populateDeckSelect(generateOpponentDeck, decks);
  deckOptionsLoaded = true;
}

async function generateReplay() {
  if (!generateButton) return;
  const playerPolicy = generatePlayerPolicy?.value || "self";
  const opponent = generateOpponent.value;
  const seed = generateSeed.value;
  const playerDeck = generatePlayerDeck?.value || "";
  const opponentDeck = generateOpponentDeck?.value || "";
  generateButton.disabled = true;
  generateStatus.className = "generate-status busy";
  generateStatus.textContent = lang === "ja"
    ? "生成中… 1試合を回しています（数十秒かかることがあります）。"
    : "Generating… running one match (this can take a while).";
  try {
    const res = await fetch("/api/replays/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ playerPolicy, opponent, seed: Number(seed), playerDeck, opponentDeck }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
    // 生成後は一覧を更新し、新しい replay（最新＝先頭）を自動で開く。
    await loadReplayList();
    if (data.name) replaySelect.value = data.name;
    await loadSelectedReplay();
    generateStatus.className = "generate-status ok";
    generateStatus.textContent = (lang === "ja" ? "完了: " : "Done: ") + (data.name || "");
    closeNewReplayModal();
    // 新しいリプレイに未生成のカードがあれば裏で画像を作る。
    ensureImagesForCurrentReplay();
  } catch (err) {
    generateStatus.className = "generate-status err";
    generateStatus.textContent = (lang === "ja" ? "失敗: " : "Failed: ") + err.message;
  } finally {
    generateButton.disabled = false;
  }
}

generateButton?.addEventListener("click", generateReplay);

// ---------- New-replay modal (opened from the "＋ 新規" button by the Replay picker) ----------
const newReplayModal = document.getElementById("newReplayModal");
function openNewReplayModal() {
  if (!newReplayModal) return;
  if (generateStatus) { generateStatus.textContent = ""; generateStatus.className = "generate-status"; }
  newReplayModal.hidden = false;
  if (!deckOptionsLoaded) {
    loadDeckOptions().catch((error) => {
      if (generateStatus) {
        generateStatus.className = "generate-status err";
        generateStatus.textContent = (lang === "ja" ? "Deck list failed: " : "Deck list failed: ") + error.message;
      }
    });
  }
}
function closeNewReplayModal() {
  if (newReplayModal) newReplayModal.hidden = true;
}
document.getElementById("newReplayButton")?.addEventListener("click", openNewReplayModal);
document.getElementById("newReplayClose")?.addEventListener("click", closeNewReplayModal);
document.getElementById("newReplayBackdrop")?.addEventListener("click", closeNewReplayModal);

// ---------- Card images: generate in the background when missing ----------
// card_images/ は gitignore 済みで、初回クローン時は空。無ければ裏で build_card_assets.py を回し、
// 「画像を作成中です…」を出す。生成中もテキスト表示で使えるので非ブロッキング。
const imageBuildBanner = document.getElementById("imageBuildBanner");
const imageBuildText = document.getElementById("imageBuildText");
const imageBuildSpinner = document.getElementById("imageBuildSpinner");
const imageBuildDismiss = document.getElementById("imageBuildDismiss");
const rebuildImagesButton = document.getElementById("rebuildImagesButton");
const cardImageInfo = document.getElementById("cardImageInfo");
let cardImageBuilding = false;

imageBuildDismiss?.addEventListener("click", () => { if (imageBuildBanner) imageBuildBanner.hidden = true; });

async function reloadCardManifest() {
  try {
    const r = await fetch(`./card_images/manifest.json?v=${Date.now()}`);
    cardManifest = r.ok ? await r.json() : {};
  } catch (_) { cardManifest = {}; }
  const has = Object.keys(cardManifest).length > 0;
  toggleCardImages.disabled = !has;
  if (has) toggleCardImages.closest(".toggle")?.removeAttribute("title");
  if (cardImageInfo) {
    cardImageInfo.textContent = has
      ? (lang === "ja" ? `カード画像: ${Object.keys(cardManifest).length} 枚` : `Card images: ${Object.keys(cardManifest).length}`)
      : (lang === "ja" ? "カード画像なし（テキスト表示）" : "No card images (text mode)");
  }
}

async function runCardImageBuild(mode) {
  if (cardImageBuilding) return;
  cardImageBuilding = true;
  if (imageBuildBanner) imageBuildBanner.hidden = false;
  if (imageBuildSpinner) imageBuildSpinner.hidden = false;
  if (imageBuildDismiss) imageBuildDismiss.hidden = true;
  if (imageBuildText) {
    imageBuildText.textContent = lang === "ja"
      ? (mode === "all" ? "全カード画像を作成中です。お待ちください…（数分かかります）" : "画像を作成中です。お待ちください…（初回のみ）")
      : "Generating card images… please wait.";
  }
  try {
    await fetch("/api/card_images/build", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    for (let i = 0; i < 900; i += 1) {
      await new Promise((r) => setTimeout(r, 2000));
      const st = await (await fetch("/api/card_images/status")).json();
      if (st.running && st.stage === "venv_setup" && imageBuildText) {
        imageBuildText.textContent = lang === "ja"
          ? "初回セットアップ中です（画像抽出用の環境を準備しています）。数分かかることがあります…"
          : "First-time setup in progress (preparing the image-extraction environment). This can take a few minutes…";
      }
      if (!st.running) {
        if (st.error && !st.count) {
          if (imageBuildText) imageBuildText.textContent = lang === "ja"
            ? `画像は用意できませんでした（${st.error}）。README の「カード画像を用意する」の手動手順を`
              + "試すか、data/ に PDF があるか確認してください。画像なしでもテキスト表示で問題なく使えます。"
            : `Could not build images (${st.error}). Try the manual steps in the README section `
              + "\"Prepare card images\", or check that the PDF exists under data/. "
              + "The viewer works fine in text mode without them.";
          // 自動で消さない: 原因(venv未セットアップ)に気づいてもらう必要があるため、
          // 「画像を作成中です…」のように数秒で自動的に消える通知にはしない。× で手動で閉じる。
          if (imageBuildSpinner) imageBuildSpinner.hidden = true;
          if (imageBuildDismiss) imageBuildDismiss.hidden = false;
          return;
        }
        await reloadCardManifest();
        if (toggleCardImages.checked && replayData) render();
        if (imageBuildBanner) imageBuildBanner.hidden = true;
        return;
      }
    }
    if (imageBuildBanner) imageBuildBanner.hidden = true;
  } catch (_) {
    if (imageBuildBanner) imageBuildBanner.hidden = true;
  } finally {
    cardImageBuilding = false;
  }
}

function replayHasMissingImages() {
  if (!replayData || toggleCardImages?.disabled) return false;
  const found = new Set();
  const walk = (o) => {
    if (Array.isArray(o)) { o.forEach(walk); return; }
    if (o && typeof o === "object") {
      if (typeof o.id === "number" && ("name" in o || "serial" in o)) found.add(o.id);
      for (const k in o) walk(o[k]);
    }
  };
  walk(replayData.frames || []);
  for (const id of found) if (!cardManifest[String(id)]) return true;
  return false;
}

// 画像が1枚も無い、または今のリプレイに画像未生成のカードがあれば、裏で生成する。
async function ensureImagesForCurrentReplay() {
  if (cardImageBuilding) return;
  if (!Object.keys(cardManifest).length || replayHasMissingImages()) {
    await runCardImageBuild("deck");
  } else {
    reloadCardManifest(); // info テキスト更新のみ
  }
}

rebuildImagesButton?.addEventListener("click", () => runCardImageBuild("all"));

document.getElementById("langToggleButton").addEventListener("click", () => {
  lang = lang === "ja" ? "en" : "ja";
  applyLang();
});

async function boot() {
  document.body.classList.add("replay-mode");
  setupCardPreview();
  setupDiscardModal();
  setupInteractiveSelection();
  buildDebugSubtabs();
  applyLang();
  // 初回起動時だけガイドツアーを開始する（既読は localStorage に記録）。
  let seenOnboarding = false;
  try { seenOnboarding = !!localStorage.getItem(ONBOARDING_KEY); } catch (_) { seenOnboarding = false; }
  if (!seenOnboarding) setTimeout(startTour, 300);
  await Promise.all([loadCardManifest(), loadCardNamesJp(), loadAttackNamesJp(), loadCardTypes(), loadDeckOptions()]);
  replaySpeed = Number(speedSelect?.value || "1") || 1;
  liveCpuSelect.value = requestedInitialCpu();
  await loadReplayList();
  viewerMode = requestedInitialMode();
  modeSelect.value = viewerMode;
  if (viewerMode === "live") {
    await ensureLiveMatchStarted();
  } else {
    await loadSelectedReplay();
  }
  updateModeUI();
  // 画像が無ければ裏で生成（初回のみ・非ブロッキング）。
  ensureImagesForCurrentReplay();
}

boot().catch((error) => {
  frameMeta.textContent = error.message;
  console.error(error);
});
