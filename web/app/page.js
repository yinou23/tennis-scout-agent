"use client";
import { Fragment, useEffect, useState } from "react";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// Week 9 Step 3: gauge dashboard, side list, bilingual UI, player
// and surface selectors, foldables, market section inside the gauge
// card. Presentation only.
//
// Frontend computes NO tennis numbers. Probability, Elo, win rates,
// surface record, and head to head are read from the API summary
// fields and shown exactly as returned. The only arithmetic allowed
// is a display transform, documented where it is used:
//   gauge green = the favorite's share of the arc (display only)
//   percentage text = rate * 100, rounded to the nearest whole
//   percent (display only); Elo shows as a whole number (display
//   only). In both cases the original number is shown in the
//   report text, and the transform never changes the number
//   itself. The report text is never parsed. If the summary field
//   is missing (old API), the gauge, side list, and short summary
//   are hidden, and only the verbatim report is shown. The page
//   never crashes on a missing summary or a missing market field.
//
// Expected summary shape (from the backend Step 3 agent):
//   report.summary = {
//     prob_a_calibrated, prob_b_calibrated, prob_a_blended,
//     prob_b_blended, favorite, player_a, player_b, surface,
//     data_as_of, h2h: { total, wins_a, wins_b },
//     player_a_obj / players: per player overall_elo, surface_elo,
//     blended_elo, surface record, form }
// The normalizer below also accepts the report dict itself, a
// players array, and the tools dict from build_scout_report, so
// a slightly different backend shape still renders instead of
// crashing. Fields that are absent are shown as a dash, never
// invented.
//
// Expected market shape (GET /market?player_a=..&player_b=..):
//   { polymarket: {...} | null, kalshi: {...} | null }
// or { markets: [...] } or a list. Each market entry may carry
// prob_a / prob_b (or prob / price), fetched_at, and volume.
// Market probability is shown separately, never mixed with the
// model probability, and never shown as the main probability.

const PLAYERS = [
  "Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", "Roger Federer",
  "Rafael Nadal", "Alexander Zverev", "Daniil Medvedev", "Andy Murray",
  "Stan Wawrinka", "Dominic Thiem", "Stefanos Tsitsipas", "Andrey Rublev",
  "Casper Ruud", "Taylor Fritz", "Ben Shelton", "Frances Tiafoe",
];

// Low-poly avatars in web/public/players/. All 16 files face right;
// the right-side player is flipped with CSS so the pair face each
// other. Lookup is by surname, so "Jannik Sinner", "Sinner", and
// "J. Sinner" all resolve. A name with no file falls back to the
// generic wireframe portrait, never a broken image. Avatars are
// static files, not database content: the database holds tennis
// numbers only.
const AVATARS = {
  Sinner: "/players/sinner-white.webp",
  Alcaraz: "/players/alcaraz-white.webp",
  Djokovic: "/players/djokovic-white.webp",
  Federer: "/players/federer-white.webp",
  Nadal: "/players/nadal-white.webp",
  Zverev: "/players/zverev-white.webp",
  Medvedev: "/players/medvedev-white.webp",
  Murray: "/players/murray-white.webp",
  Wawrinka: "/players/wawrinka-white.webp",
  Thiem: "/players/thiem-white.webp",
  Tsitsipas: "/players/tsitsipas-white.webp",
  Rublev: "/players/rublev-white.webp",
  Ruud: "/players/ruud-white.webp",
  Fritz: "/players/fritz-white.webp",
  Shelton: "/players/shelton-white.webp",
  Tiafoe: "/players/tiafoe-white.webp",
};

function avatarSrc(name) {
  if (!name) return null;
  const last = String(name).trim().split(/\s+/).pop();
  return AVATARS[last] || "/players/generic-white.webp";
}

const STRINGS = {
  zh: {
    title: "Tennis Scout Agent",
    dataCutoff: "数据截至 2026-05-25，排名截至 2026-06-08",
    datasetLine: "比赛数据：Jeff Sackmann ATP 数据集 · 数据截至 2026-05-25，排名截至 2026-06-08",
    connected: "已连接",
    notConnected: "未连接",
    playerA: "球员 A",
    playerB: "球员 B",
    surface: "场地",
    surfaceAny: "不限",
    advancedLabel: "高级：用自然语言提问（填写后优先使用）",
    advancedPlaceholder: "例如：辛纳对阿尔卡拉斯硬地谁是热门",
    generate: "预测",
    generating: "预测中",
    dashboard: "概率仪表盘",
    favorite: "热门",
    probability: "概率",
    matchesLoading: "正在拉取赛程",
    matchesEmpty: "暂无即将开打的比赛",
    matchesUnavailable: "赛程暂时不可用",
    noQuote: "暂无报价",
    probInfo: "这是用 Elo 算出来的校准概率，0.5 是五五开，绿色弧线是热门一方的占比。混合概率是校准前的数，放在右边数据一览里。",
    mixedProb: "混合概率",
    mixedProbInfo: "混合概率是校准前的概率，先把场地 Elo 和总 Elo 混合，再校准成最终概率。",
    elo: "Elo",
    eloInfo: "Elo 是从历史比赛算出来的评分，分数越高，历史表现越强。",
    blendInfo: "场地混合是把该场地的 Elo 和总 Elo 混合起来算的，用的先验是 100，场地比赛越多，场地 Elo 的权重越大。",
    overallElo: "Elo",
    surfaceElo: "Elo",
    surfaceRecord: "战绩",
    surfaceRecordInfo: "该球员在这个场地上的胜率。",
    last10: "最近胜率",
    last10Info: "最近 2 个月内的胜率；如果超过 10 场，只算最近 10 场。",
    h2h: "H2H",
    h2hInfo: "两人历史交手的胜负场次。",
    marketNote: "市场概率是投注市场投出来的，和模型概率是两回事，只做对照，不混在一起。只读行情，不下单。蓝色胶囊是 Polymarket，橙色胶囊是 Kalshi；差值是市场均价减模型概率。",
    marketEmpty: "Polymarket、Kalshi 暂无市场",
    marketPolymarket: "Polymarket",
    marketKalshi: "Kalshi",
    marketUnavailable: "市场数据暂时不可用",
    marketLoading: "正在拉取市场数据",
    marketUpdated: "上次更新",
    updatedJustNow: "刚刚",
    updatedMinAgo: "分钟前",
    updatedHrAgo: "小时前",
    marketUpdateFailed: "更新失败，显示旧数据",
    marketAutoRefresh: "市场自动刷新",
    fullReport: "完整报告",
    sources: "来源与核查",
    verificationPassed: "核查通过",
    verificationFailed: "核查未通过",
    model: "模型",
    plannerSource: "规划器来源",
    plan: "计划",
    langGraph: "是否用了真 LangGraph",
    yes: "是",
    no: "否",
    extraNumbers: "多余数字",
    missingNumbers: "缺失必需数字",
    tagIssue: "标签问题",
    tagOk: "标签无问题",
    errorTitle: "生成失败，没有报告",
    errorNote: "失败时不显示没过的报告，这是特意这样做的。",
    emptyQuestion: "问题不能为空",
    networkError: "网络错误，请确认 API 已启动。",
    footnote: "数据截至 2026-05-25，不是实时数据。报告里的数字来自数据库查询，不是模型凭空写出。状态、局数、退赛、交手只做描述，不进概率。",
    vs: "对阵",
  },
  en: {
    title: "Tennis Scout Agent",
    dataCutoff: "Data as of 2026-05-25, rankings as of 2026-06-08",
    datasetLine: "Match data: Jeff Sackmann ATP dataset · Data as of 2026-05-25, rankings as of 2026-06-08",
    connected: "Connected",
    notConnected: "Not connected",
    playerA: "Player A",
    playerB: "Player B",
    surface: "Surface",
    surfaceAny: "Any",
    advancedLabel: "Advanced: ask in your own words (used first when filled)",
    advancedPlaceholder: "Example: Who is the favorite, Sinner or Alcaraz on Hard?",
    generate: "Predict",
    generating: "Predicting",
    dashboard: "Probability gauge",
    favorite: "Favorite",
    probability: "Probability",
    matchesLoading: "Loading matches",
    matchesEmpty: "No upcoming matches",
    matchesUnavailable: "Schedule unavailable",
    noQuote: "No quote",
    probInfo: "This is the calibrated probability from Elo. 0.5 is even, and the green arc is the favorite's share. The blended probability is the pre-calibration number, shown in the side list.",
    mixedProb: "Blended probability",
    mixedProbInfo: "Blended probability is the pre-calibration number: surface Elo and overall Elo are blended first, then calibrated into the final probability.",
    elo: "Elo",
    eloInfo: "Elo is a rating computed from match history. A higher rating means stronger past results.",
    blendInfo: "Surface blend mixes the surface Elo with the overall Elo, with prior 100. More matches on that surface give the surface Elo more weight.",
    overallElo: "Elo",
    surfaceElo: "Elo",
    surfaceRecord: "record",
    surfaceRecordInfo: "This player's win rate on this surface.",
    last10: "Recent win rate",
    last10Info: "Win rate in the last 2 months; if more than 10 matches, only the last 10 count.",
    h2h: "H2H",
    h2hInfo: "Head-to-head wins for each side.",
    marketNote: "Market probabilities come from betting markets. They are separate from the model probability, shown only for comparison. Read-only prices, no trading. Blue pill is Polymarket, orange pill is Kalshi; the diff is the market average minus the model probability.",
    marketEmpty: "No markets on Polymarket or Kalshi",
    marketPolymarket: "Polymarket",
    marketKalshi: "Kalshi",
    marketUnavailable: "Market data unavailable",
    marketLoading: "Loading market data",
    marketUpdated: "Last updated",
    updatedJustNow: "just now",
    updatedMinAgo: "min ago",
    updatedHrAgo: "hr ago",
    marketUpdateFailed: "Refresh failed, showing older data",
    marketAutoRefresh: "Market Auto Refresh",
    fullReport: "Full report",
    sources: "Sources and verification",
    verificationPassed: "Verification passed",
    verificationFailed: "Verification failed",
    model: "Model",
    plannerSource: "Planner source",
    plan: "Plan",
    langGraph: "Used real LangGraph",
    yes: "Yes",
    no: "No",
    extraNumbers: "Unexpected numbers",
    missingNumbers: "Missing required numbers",
    tagIssue: "Tag problem",
    tagOk: "Tags OK",
    errorTitle: "Failed, no report",
    errorNote: "A failed report is never shown as if it had passed. That is deliberate.",
    emptyQuestion: "Question cannot be empty",
    networkError: "Network error. Please check that the API is running.",
    footnote: "Data as of 2026-05-25, not live. Numbers in the report come from database queries, not model text. Form, games, retirement, and head to head are descriptors only, not probability inputs.",
    vs: "vs",
  },
};

// Surface names for the dashboard chip: the preview shows
// "HARD · 硬地". The chip is shown only when a surface is set.
const SURFACE_ZH = { Hard: "硬地", Clay: "红土", Grass: "草地", Carpet: "地毯" };

// Chip text is one language only: English in English mode, Chinese in
// Chinese mode. The chip color is per surface: Hard blue, Grass green,
// Clay brown-red, Carpet gray.
function surfaceChip(surface, lang) {
  if (!surface) return null;
  if (lang === "zh") return SURFACE_ZH[surface] || surface;
  return String(surface).toUpperCase();
}

function surfaceChipClass(surface) {
  const key = String(surface || "").toLowerCase();
  if (key === "hard") return "surface-chip-hard";
  if (key === "clay") return "surface-chip-clay";
  if (key === "grass") return "surface-chip-grass";
  return "surface-chip-carpet";
}

function pick(obj, keys) {
  if (!obj || typeof obj !== "object") return undefined;
  for (const key of keys) {
    const value = obj[key];
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
}

function asText(value) {
  if (value === undefined || value === null || value === "") return null;
  return String(value);
}

function asNumber(value) {
  if (value === undefined || value === null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function asElo(value) {
  // Elo is stored rounded to 1 decimal (2236.0, 2283.1). String(2236.0)
  // would drop the trailing .0, so keep one decimal for numbers.
  if (value === undefined || value === null || value === "") return null;
  const n = Number(value);
  if (Number.isFinite(n)) return n.toFixed(1);
  return String(value);
}

function asPct(value) {
  // Display transform only: rate 0.8149 shows as 81 percent.
  // The original rate is kept for the bar width.
  if (value === undefined || value === null || value === "") return null;
  const n = Number(value);
  if (Number.isFinite(n)) return `${Math.round(n * 100)}%`;
  return String(value);
}

// Relative update time: "just now", "2 min ago", "1 hr ago".
// Takes the strings object so both languages work.
function relativeUpdatedAt(date, t) {
  if (!date) return "";
  const seconds = Math.max(0, Math.floor((Date.now() - date.getTime()) / 1000));
  if (seconds < 60) return t.updatedJustNow;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} ${t.updatedMinAgo}`;
  const hours = Math.floor(minutes / 60);
  return `${hours} ${t.updatedHrAgo}`;
}

function asInt(value) {
  // Display transform only: Elo 2283.1 shows as 2283.
  if (value === undefined || value === null || value === "") return null;
  const n = Number(value);
  if (Number.isFinite(n)) return `${Math.round(n)}`;
  return String(value);
}

// Leading value in a comparison row: the larger number wins the
// green mark. Ties and unparseable values get no mark. Used for
// the side list only; it never changes a number, only its colour.
function leadSide(a, b) {
  const na = Number(a);
  const nb = Number(b);
  if (Number.isNaN(na) || Number.isNaN(nb) || na === nb) return null;
  return na > nb ? "a" : "b";
}

function normalizePlayer(raw, fallbackName) {
  if (typeof raw === "string") return { name: raw };
  if (!raw || typeof raw !== "object") return { name: fallbackName || null };
  // Backend summary shape (scripts/summary.py): per player block
  // is { name, elo: {overall, surface_elo, blended, surface_matches,
  // matches_played}, record: {surface: {wins, losses, total, win_pct}},
  // form: {win10_pct, win52_pct, ...} }. The nested blocks below
  // read those first, then fall back to the flat tool shapes.
  const elo = raw.elo || {};
  const record = raw.record || raw.surface_record || {};
  const surfaceEntry = (record && typeof record.surface === "object" && record.surface)
    || raw.surface_entry || record;
  const form = raw.form || raw.recent_form || {};
  const win = form.win_form || form.win || form;
  return {
    name: asText(pick(raw, ["name", "player", "full_name"])) || fallbackName || null,
    overall_elo: asElo(pick(elo, ["overall", "overall_elo"])) || asElo(pick(raw, ["overall_elo", "overall", "elo"])),
    surface_elo: asElo(pick(elo, ["surface_elo", "surfaceElo"])) || asElo(pick(raw, ["surface_elo", "surfaceElo"])),
    blended_elo: asElo(pick(elo, ["blended", "blended_elo"])) || asElo(pick(raw, ["blended_elo", "blended"])),
    matches_played: asText(pick(elo, ["matches_played"])) || asText(pick(raw, ["matches_played", "matches", "total_matches"])),
    surface_matches: asText(pick(elo, ["surface_matches", "surface_n"])) || asText(pick(raw, ["surface_matches", "surface_n"])),
    surface_wins: asText(pick(surfaceEntry, ["wins", "surface_wins"])),
    surface_losses: asText(pick(surfaceEntry, ["losses", "surface_losses"])),
    surface_total: asText(pick(surfaceEntry, ["total", "surface_total"])),
    surface_win_rate: asText(pick(surfaceEntry, ["win_pct", "win_rate", "winPct", "surface_win_rate"])),
    last10_win_rate: asText(pick(win, ["last10_pct", "last10_win_rate", "win10_pct", "win10", "last10"])) || asText(pick(raw, ["last10_pct", "last10_win_rate", "win10"])),
    last10_wins: asText(pick(win, ["last10_wins", "win10_wins"])),
    last10_n: asText(pick(win, ["last10_n", "win10_n"])),
    recent_win_rate: asText(pick(win, ["recent2m_pct", "recent2m_win_rate", "win2m_pct", "win2m"])),
    last52_win_rate: asText(pick(win, ["last52_pct", "last52_win_rate", "win52_pct", "win52"])),
  };
}

function normalizeSummary(report) {
  if (!report || typeof report !== "object") return null;
  const s = report.summary || report.report_summary || null;
  const base = s || report;
  const hasProb = pick(base, ["prob_a_calibrated", "probA_calibrated"]) !== undefined;
  const hasPlayers = base.player_a !== undefined || base.players !== undefined || base.tools !== undefined;
  if (!s && !hasProb && !hasPlayers) return null;
  const tools = base.tools || report.tools || {};
  const matchTool = tools.match || base.match || {};
  // Backend summary shape: per player blocks live under a and b,
  // player_a / player_b are plain name strings. Do not pass the
  // name string in as the player object, it carries no Elo.
  const rawA = base.a || base.player_a_obj || (Array.isArray(base.players) ? base.players[0] : base.players?.a) || matchTool.player_a || base.elo_a || null;
  const rawB = base.b || base.player_b_obj || (Array.isArray(base.players) ? base.players[1] : base.players?.b) || matchTool.player_b || base.elo_b || null;
  const plan = report.plan || {};
  const nameA = asText(pick(base, ["player_a_name"])) || (typeof base.player_a === "string" ? base.player_a : null) || (base.a && base.a.name) || (typeof rawA === "string" ? rawA : null) || plan.player_a || null;
  const nameB = asText(pick(base, ["player_b_name"])) || (typeof base.player_b === "string" ? base.player_b : null) || (base.b && base.b.name) || (typeof rawB === "string" ? rawB : null) || plan.player_b || null;
  const playerA = normalizePlayer(rawA, nameA);
  const playerB = normalizePlayer(rawB, nameB);
  // If the per player object came from the tools dict, its Elo fields
  // live under overall_elo / blended_elo / surface_elo already.
  if (!playerA.overall_elo && matchTool.player_a) {
    playerA.overall_elo = asText(matchTool.player_a.overall_elo);
    playerA.blended_elo = asText(matchTool.player_a.blended_elo);
  }
  if (!playerB.overall_elo && matchTool.player_b) {
    playerB.overall_elo = asText(matchTool.player_b.overall_elo);
    playerB.blended_elo = asText(matchTool.player_b.blended_elo);
  }
  const h2hRaw = base.h2h || base.head_to_head || tools.h2h || null;
  const h2h = h2hRaw && typeof h2hRaw === "object" ? {
    total: asText(pick(h2hRaw, ["total"])),
    wins_a: asText(pick(h2hRaw, ["wins_a", "winsA"])),
    wins_b: asText(pick(h2hRaw, ["wins_b", "winsB"])),
  } : {
    total: asText(pick(base, ["h2h_total"])),
    wins_a: asText(pick(base, ["h2h_wins_a"])),
    wins_b: asText(pick(base, ["h2h_wins_b"])),
  };
  return {
    prob_a_calibrated: asText(pick(base, ["prob_a_calibrated", "probA_calibrated"])),
    prob_b_calibrated: asText(pick(base, ["prob_b_calibrated", "probB_calibrated"])),
    prob_a_blended: asText(pick(base, ["prob_a_blended", "probA_blended"])),
    prob_b_blended: asText(pick(base, ["prob_b_blended", "probB_blended"])),
    favorite: asText(pick(base, ["favorite"])),
    surface: asText(pick(base, ["surface"])) || asText(plan.surface),
    data_as_of: asText(pick(base, ["data_as_of", "dataAsOf", "data_asof"])) || asText(report.data_as_of) || "2026-05-25",
    player_a: playerA,
    player_b: playerB,
    h2h,
  };
}

function normalizeMarket(data) {
  if (!data) return [];
  let entries = [];
  if (Array.isArray(data)) entries = data;
  else if (Array.isArray(data.markets)) entries = data.markets;
  else if (data.polymarket !== undefined || data.kalshi !== undefined) {
    // Always show both venues. A venue with no market renders as
    // "no quote" (or "no market" when neither has one), never hidden.
    entries.push(data.polymarket ? { source: "Polymarket", ...data.polymarket } : { source: "Polymarket" });
    entries.push(data.kalshi ? { source: "Kalshi", ...data.kalshi } : { source: "Kalshi" });
  } else if (typeof data === "object") entries = [data];
  return entries.map((entry) => {
    if (!entry || typeof entry !== "object") return null;
    return {
      source: asText(pick(entry, ["source", "venue", "market"])) || "market",
      prob_a: asText(pick(entry, ["prob_a", "player_a_probability", "probability_a", "prob", "price", "yes_price", "implied_probability", "implied_prob_a"])),
      prob_b: asText(pick(entry, ["prob_b", "player_b_probability", "probability_b", "implied_prob_b"])),
      fetched_at: asText(pick(entry, ["fetched_at", "fetched_time", "timestamp", "as_of", "fetchedAt"])),
      volume: asText(pick(entry, ["volume", "total_volume", "volume_usd", "liquidity"])),
      question: asText(pick(entry, ["question", "title", "slug", "url"])),
    };
  }).filter(Boolean);
}

function planText(plan, t) {
  if (!plan || typeof plan !== "object") return "-";
  if (plan.tool === "match") {
    return `${t.vs}: ${plan.player_a} / ${plan.player_b}, ${t.surface}: ${plan.surface || t.surfaceAny}`;
  }
  if (plan.tool === "h2h") {
    return `${t.h2h}: ${plan.player_a} / ${plan.player_b}`;
  }
  if (plan.tool === "elo" || plan.tool === "form" || plan.tool === "surface") {
    return `${plan.tool}: ${plan.player}`;
  }
  return JSON.stringify(plan);
}

function verificationItems(verification, t) {
  if (!verification || typeof verification !== "object") return [];
  const items = [];
  items.push(`${t.extraNumbers}: ${(verification.unexpected_numbers || []).length}`);
  items.push(`${t.missingNumbers}: ${(verification.missing_required || []).length}`);
  if (verification.tag_error) items.push(`${t.tagIssue}: ${verification.tag_error}`);
  else items.push(`${t.tagOk}`);
  return items;
}

// Avatar: circular low-poly portrait above the player name.
// flip mirrors the image so the right-side player faces the left
// one. A missing file hides itself instead of showing a broken
// image icon.
function Avatar({ name, flip, large }) {
  const src = avatarSrc(name);
  if (!src) return null;
  const cls = `avatar${large ? " avatar-lg" : ""}${flip ? " avatar-flip" : ""}`;
  return (
    <img
      src={src}
      alt={name}
      className={cls}
      onError={(e) => { e.currentTarget.style.display = "none"; }}
    />
  );
}

// Gauge: semicircular SVG, no needle. Display transform only.
// Model: the left end is player A (probA = 1), the right end is
// player B (probA = 0), the centre is 50/50. The green fills the
// favoured side's share of the arc, anchored at the favoured end,
// so the green reads as the favourite's share. The favourite's
// calibrated probability is printed over the arc as HTML, so the
// number itself can carry a hover tooltip; there is no separate
// label or info button. No 0/100 end labels: they read backwards
// against the left-is-A scale. The original probability is shown
// as returned, and the fill never changes the number.
function Gauge({ summary, t, favProb }) {
  const probA = asNumber(summary.prob_a_calibrated);
  if (probA === null) return null;
  const arcLen = 251.3;
  const greenLen = probA >= 0.5 ? probA * arcLen : (1 - probA) * arcLen;
  const greenStart = probA >= 0.5 ? 0 : probA * arcLen;
  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 200 115" width="100%" role="img" aria-label={t.probability}>
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e5e7eb" strokeWidth="10" strokeLinecap="round" />
        {greenLen > 0 && (
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#15803d" strokeWidth="10" strokeLinecap="round" strokeDasharray={`${greenLen} ${arcLen}`} strokeDashoffset={-greenStart} />
        )}
      </svg>
      <div className="gauge-number">
        <Tip text={t.probInfo}>{asPct(favProb) || "-"}</Tip>
      </div>
    </div>
  );
}

function Tip({ text, children }) {
  return (
    <span className="has-tip" tabIndex={0}>
      {children}
      <span className="info-tip">{text}</span>
    </span>
  );
}

export default function Home() {
  const [lang, setLang] = useState("en");
  const t = STRINGS[lang];
  const [playerA, setPlayerA] = useState("Jannik Sinner");
  const [playerB, setPlayerB] = useState("Carlos Alcaraz");
  const [surface, setSurface] = useState("Hard");
  const [advanced, setAdvanced] = useState("");
  const [connected, setConnected] = useState(null);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [answer, setAnswer] = useState(null);
  const [error, setError] = useState(null);
  const [market, setMarket] = useState({ status: "idle", rows: [] });
  const [marketPair, setMarketPair] = useState(null);
  const [marketUpdatedAt, setMarketUpdatedAt] = useState(null);
  // Re-render every 30s so the relative "updated X min ago" stays fresh
  // even when auto refresh is off and no new market data arrives.
  const [, setNowTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setNowTick((v) => v + 1), 30 * 1000);
    return () => clearInterval(id);
  }, []);
  const [marketAuto, setMarketAuto] = useState(true);
  const [marketRefreshFailed, setMarketRefreshFailed] = useState(false);
  const [matches, setMatches] = useState({ status: "idle", rows: [] });

  // Upcoming-matches strip: fetched once when the page loads,
  // from the live Polymarket feed, never from the preview's
  // stale snapshot. A tap loads that matchup into the form
  // and runs the report. The initial page is the soonest
  // upcoming match from this feed when there is one, else the
  // Sinner / Hard / Alcaraz default; it generates at once so
  // the page opens on data instead of an empty dashboard.
  // Matches carry no surface, so the surface stays on the
  // default (Hard), same as tapping a fixture.
  useEffect(() => {
    setMatches({ status: "loading", rows: [] });
    fetch(`${API_URL}/matches`)
      .then((res) => {
        if (!res.ok) throw new Error(`matches ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const rows = Array.isArray(data.matches) ? data.matches : [];
        setMatches({ status: "done", rows });
        if (rows.length > 0) {
          setPlayerA(rows[0].player_a);
          setPlayerB(rows[0].player_b);
          setAdvanced("");
          onGenerate({ a: rows[0].player_a, b: rows[0].player_b }, { scroll: false });
        } else {
          // No upcoming match: the default is Sinner / Hard /
          // Alcaraz, which is pre-generated, so load the cached
          // report instead of running /report again.
          loadCachedDefault();
        }
      })
      .catch(() => {
        setMatches({ status: "unavailable", rows: [] });
        loadCachedDefault();
      });
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((res) => res.json())
      .then((data) => setConnected(data && data.status === "ok"))
      .catch(() => setConnected(false));
  }, []);

  // Auto refresh: the market section is the only part of the page
  // that changes during the day. The model numbers come from
  // the database as of 2026-05-25 and never move, so refresh
  // only re-fetches /market and only rewrites the market rows,
  // never the whole page, never the gauge or the side list.
  // The timer is 60 seconds after the last successful update.
  // A hidden tab pauses, and coming back refreshes at once if
  // the last fetch is stale. 60 seconds matches the backend
  // /market cache, so a faster interval would only re-fetch
  // the same cached numbers.
  useEffect(() => {
    if (!marketPair || !marketAuto) return undefined;
    const tick = () => {
      if (!document.hidden) fetchMarket(marketPair.a, marketPair.b, { silent: true });
    };
    const id = setInterval(tick, 60 * 1000);
    const onVisible = () => {
      if (!document.hidden && marketUpdatedAt && Date.now() - marketUpdatedAt.getTime() > 60 * 1000) {
        fetchMarket(marketPair.a, marketPair.b, { silent: true });
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      clearInterval(id);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [marketPair, marketAuto, marketUpdatedAt]);

  async function fetchMarket(a, b, opts) {
    const silent = opts && opts.silent;
    // Silent refresh keeps the old rows on screen and only swaps
    // them when the new fetch succeeds. A failed refresh keeps
    // the old numbers and says so, instead of blanking the card.
    if (!silent) setMarket({ status: "loading", rows: [] });
    setMarketPair({ a, b });
    try {
      const url = `${API_URL}/market?player_a=${encodeURIComponent(a)}&player_b=${encodeURIComponent(b)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`market ${res.status}`);
      const data = await res.json();
      const rows = normalizeMarket(data);
      setMarket({ status: "done", rows });
      setMarketUpdatedAt(new Date());
      setMarketRefreshFailed(false);
    } catch (e) {
      if (silent) {
        setMarketRefreshFailed(true);
      } else {
        setMarket({ status: "unavailable", rows: [] });
      }
    }
  }

  // Apply a report response (from /report or from the pre-generated
  // default-report.json) to state. Shared by onGenerate and
  // loadCachedDefault so both paths show the same dashboard.
  function applyReport(data, a, b, opts) {
    setReport(data);
    setAnswer(null);
    // Market is separate and non blocking. Use the player names
    // from the report plan when present, else the selectors.
    const pa = (data.plan && data.plan.player_a) || a;
    const pb = (data.plan && data.plan.player_b) || b;
    fetchMarket(pa, pb);
    // The dashboard only exists once a report is set, so
    // scroll to it here, after the report lands. The initial
    // auto-generate on page load does not scroll: the user
    // opened the page at the top.
    if (!opts || opts.scroll !== false) {
      const el = document.getElementById("dashboard");
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  // Load the pre-generated default report (Sinner / Hard /
  // Alcaraz) from the static file instead of calling /report.
  // The numbers come from the database as of 2026-05-25 and
  // never move, so a cached copy is correct until the database
  // is updated, at which point default-report.json is
  // regenerated. If the file is missing, fall back to /report.
  async function loadCachedDefault() {
    setLoading(true);
    setError(null);
    setReport(null);
    setAnswer(null);
    setMarket({ status: "idle", rows: [] });
    try {
      const res = await fetch("/default-report.json");
      if (!res.ok) throw new Error(`default report ${res.status}`);
      const data = await res.json();
      if (!data || data.error || !data.summary) throw new Error("default report invalid");
      applyReport(data, "Jannik Sinner", "Carlos Alcaraz", { scroll: false });
      setLoading(false);
    } catch (e) {
      // Fall back to /report. onGenerate manages loading itself,
      // so do not clear it here.
      onGenerate(null, { scroll: false });
    }
  }

  async function onGenerate(override, opts) {
    const a = (override && override.a ? override.a : playerA).trim();
    const b = (override && override.b ? override.b : playerB).trim();
    const adv = advanced.trim();
    let q = adv;
    if (!q) {
      if (!a || !b) {
        setError({ message: t.emptyQuestion });
        setReport(null);
        setAnswer(null);
        return;
      }
      q = `${a} vs ${b}`;
      if (surface) q = `${q} ${surface}`;
    }
    setLoading(true);
    setError(null);
    setReport(null);
    setAnswer(null);
    setMarket({ status: "idle", rows: [] });
    try {
      const res = await fetch(`${API_URL}/report`, {
        body: JSON.stringify({ question: q, player_a: a || null, player_b: b || null, surface: surface || null }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        setError({
          message: detail.error || "Request failed",
          plannerSource: detail.planner_source,
          plan: detail.plan,
          verification: detail.verification,
        });
        return;
      }
      // Open tennis question or out of scope: show the text
      // answer (or the friendly scope message) as a card, not
      // as an error and not as a match dashboard.
      if (data.kind === "answer" || data.kind === "out_of_scope") {
        setAnswer({ text: data.text || data.message || "", question: q });
        return;
      }
      applyReport(data, a, b, opts);
    } catch (e) {
      setError({ message: `${t.networkError} ${String(e)}` });
    } finally {
      setLoading(false);
    }
  }

  // Strip tap: load the matchup into the form, keep the
  // currently selected surface, and run the report. State
  // updates are async, so the names are passed to onGenerate
  // directly instead of read back from state.
  function onMatchClick(match) {
    setPlayerA(match.player_a);
    setPlayerB(match.player_b);
    setAdvanced("");
    onGenerate({ a: match.player_a, b: match.player_b });
  }

  const summary = report ? normalizeSummary(report) : null;
  const favProb = summary && summary.favorite
    ? (summary.favorite === summary.player_a.name ? summary.prob_a_calibrated
      : summary.favorite === summary.player_b.name ? summary.prob_b_calibrated
      : summary.prob_a_calibrated)
    : null;
  const verOk = report && report.verification && report.verification.ok;
  const surfaceLabel = summary && summary.surface ? summary.surface : null;
  // Side list leading marks: the larger number in each row wins
  // the green mark. Computed from the raw summary values, never
  // from the formatted display text.
  const leadBlended = summary ? leadSide(summary.prob_a_blended, summary.prob_b_blended) : null;
  const leadElo = summary ? leadSide(summary.player_a.overall_elo, summary.player_b.overall_elo) : null;
  const leadSurfaceElo = summary ? leadSide(summary.player_a.surface_elo, summary.player_b.surface_elo) : null;
  const leadRecord = summary ? leadSide(summary.player_a.surface_win_rate, summary.player_b.surface_win_rate) : null;
  const leadRecent = summary ? leadSide(summary.player_a.recent_win_rate, summary.player_b.recent_win_rate) : null;
  const leadH2h = summary ? leadSide(summary.h2h.wins_a, summary.h2h.wins_b) : null;
  // Market rows: a venue with no quote shows "no quote" when the
  // other venue has one, and "no market" when neither has one.
  const marketHasAny = market.rows.some((row) => row.prob_a && row.prob_b);
  const marketNameA = summary && summary.player_a.name ? summary.player_a.name : "A";
  const marketNameB = summary && summary.player_b.name ? summary.player_b.name : "B";

  return (
    <main>
      <header className="topbar">
        <h1>Tennis Scout <span className="title-agent">Agent</span></h1>
        <div className="topbar-right">
          <span className="status-pill">
            <span className={`dot ${connected ? "dot-ok" : "dot-bad"}`} />
            {connected === null ? "..." : connected ? t.connected : t.notConnected}
          </span>
          <button type="button" className={`header-auto-btn${marketAuto ? " on" : ""}`} aria-pressed={marketAuto} onClick={() => setMarketAuto(!marketAuto)}>
            {t.marketAutoRefresh}
          </button>
          <div className="lang-switch" role="group" aria-label="Language">
            <button type="button" className={lang === "en" ? "lang-opt active" : "lang-opt"} onClick={() => setLang("en")}>EN</button>
            <button type="button" className={lang === "zh" ? "lang-opt active" : "lang-opt"} onClick={() => setLang("zh")}>CN</button>
          </div>
        </div>
      </header>

      <section className="matches">
        {matches.status === "loading" && <p className="matches-note">{t.matchesLoading}</p>}
        {matches.status === "unavailable" && <p className="matches-note">{t.matchesUnavailable}</p>}
        {matches.status === "done" && matches.rows.length === 0 && <p className="matches-note">{t.matchesEmpty}</p>}
        {matches.status === "done" && matches.rows.length > 0 && (
          <div className="matches-grid">
            {matches.rows.map((m) => {
              const poly = m.polymarket || null;
              const active = playerA.trim() === m.player_a && playerB.trim() === m.player_b;
              const when = m.start_time ? new Date(m.start_time).toLocaleString(lang === "zh" ? "zh-CN" : "en-US", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }) : null;
              const surname = (name) => String(name).trim().split(/\s+/).pop();
              return (
                <button
                  key={`${m.player_a}-${m.player_b}-${m.start_time || ""}`}
                  type="button"
                  className={`fixture-card${active ? " active" : ""}`}
                  onClick={() => onMatchClick(m)}
                >
                  {(when || m.tournament) && (
                    <div className="fixture-time">
                      {when}
                      {m.tournament && <span className="fixture-event">{when ? ` · ${m.tournament}` : m.tournament}</span>}
                    </div>
                  )}
                  <div className="fixture-players">
                    <span className="fixture-player">
                      <Avatar name={m.player_a} />
                      {surname(m.player_a)}
                      {poly && poly.prob_a !== undefined ? (
                        <span className="fixture-pill-inline">{Math.round(Number(poly.prob_a) * 100)}%</span>
                      ) : (
                        <span className="fixture-pill-inline fixture-pill-none">-</span>
                      )}
                    </span>
                    <span className="fixture-vs">vs</span>
                    <span className="fixture-player">
                      <Avatar name={m.player_b} flip />
                      {surname(m.player_b)}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </section>

      <div className="form-row">
        <input id="player-a" type="text" list="player-list" value={playerA} onChange={(e) => setPlayerA(e.target.value)} aria-label={t.playerA} />
        <input id="player-b" type="text" list="player-list" value={playerB} onChange={(e) => setPlayerB(e.target.value)} aria-label={t.playerB} />
        <select
          className="surface-select"
          value={surface}
          onChange={(e) => setSurface(e.target.value)}
          aria-label={t.surface}
        >
          <option value="">{t.surfaceAny}</option>
          <option value="Hard">Hard</option>
          <option value="Clay">Clay</option>
          <option value="Grass">Grass</option>
          <option value="Carpet">Carpet</option>
        </select>
        <button className="generate-btn" onClick={() => onGenerate()} disabled={loading}>{loading ? t.generating : t.generate}</button>
      </div>
      <datalist id="player-list">
        {PLAYERS.map((name) => (
          <option key={name} value={name} />
        ))}
      </datalist>

      {error && (
        <section className="error">
          <h2>{t.errorTitle}</h2>
          <p>{error.message}</p>
          {error.plannerSource && <p>{t.plannerSource}: {error.plannerSource}</p>}
          {error.plan && <p>{t.plan}: {planText(error.plan, t)}</p>}
          {error.verification && (
            <ul>
              {verificationItems(error.verification, t).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
          <p>{t.errorNote}</p>
        </section>
      )}

      {answer && (
        <section className="card answer-card" id="answer">
          <p className="answer-text">{answer.text}</p>
        </section>
      )}

      {report && summary && (
        <div className="dashboard-grid" id="dashboard">
          <section className="card gauge-card">
            <div className="gauge-zone">
              <span className={`gauge-name${summary.favorite === summary.player_a.name ? " fav-name" : ""}`}>{summary.player_a.name || "-"}</span>
              {surfaceChip(summary.surface, lang) ? (
                <span className={`surface-chip ${surfaceChipClass(summary.surface)}`}>{surfaceChip(summary.surface, lang)}</span>
              ) : (
                <span aria-hidden="true" />
              )}
              <span className={`gauge-name${summary.favorite === summary.player_b.name ? " fav-name" : ""}`}>{summary.player_b.name || "-"}</span>
              <Avatar name={summary.player_a.name} large />
              <Gauge summary={summary} t={t} favProb={favProb} />
              <Avatar name={summary.player_b.name} flip large />
            </div>
            {market.status === "done" && !marketHasAny ? (
              <p className="market-empty">{t.marketEmpty}</p>
            ) : (
            <div className="market-in-gauge">
              <div className="market-table">
                <span aria-hidden="true" />
                <span className="market-col-label">{t.marketPolymarket}</span>
                <span className="market-col-label">{t.marketKalshi}</span>
                <div className="market-controls">
                  {marketUpdatedAt && (
                    <span>{t.marketUpdated} {relativeUpdatedAt(marketUpdatedAt, t)}</span>
                  )}
                </div>
                {marketRefreshFailed && <p className="market-status">{t.marketUpdateFailed}</p>}
                {market.status === "loading" && <p className="market-status">{t.marketLoading}</p>}
                {market.status === "unavailable" && <p className="market-status">{t.marketUnavailable}</p>}
                {market.status === "done" && (() => {
                  const lastName = (name) => String(name || "").trim().split(/\s+/).pop() || "-";
                  const polyRow = market.rows.find((r) => /polymarket/i.test(r.source || ""));
                  const kalshiRow = market.rows.find((r) => /kalshi/i.test(r.source || ""));
                  const players = [
                    { name: marketNameA, model: asNumber(summary.prob_a_calibrated), poly: polyRow ? asNumber(polyRow.prob_a) : null, kalshi: kalshiRow ? asNumber(kalshiRow.prob_a) : null },
                    { name: marketNameB, model: asNumber(summary.prob_b_calibrated), poly: polyRow ? asNumber(polyRow.prob_b) : null, kalshi: kalshiRow ? asNumber(kalshiRow.prob_b) : null },
                  ];
                  return players.map((p) => {
                    const markets = [p.poly, p.kalshi].filter((v) => v !== null && v !== undefined);
                    const avg = markets.length ? markets.reduce((s, v) => s + v, 0) / markets.length : null;
                    const diff = avg !== null && p.model !== null && p.model !== undefined ? Math.round((avg - p.model) * 100) : null;
                    const notable = diff !== null && Math.abs(diff) > 5;
                    return (
                      <Fragment key={p.name}>
                        <span className="market-player">{lastName(p.name)}</span>
                        <span className="market-cell"><span className={`market-pill market-pill-poly${p.poly === null || p.poly === undefined ? " market-pill-none" : ""}`}>{p.poly !== null && p.poly !== undefined ? `${Math.round(p.poly * 100)}%` : "-"}</span></span>
                        <span className="market-cell"><span className={`market-pill market-pill-kalshi${p.kalshi === null || p.kalshi === undefined ? " market-pill-none" : ""}`}>{p.kalshi !== null && p.kalshi !== undefined ? `${Math.round(p.kalshi * 100)}%` : "-"}</span></span>
                        <span className={`market-diff${notable ? " market-diff-notable" : ""}`}>{diff !== null ? `${diff > 0 ? "+" : ""}${diff}%` : "-"}</span>
                      </Fragment>
                    );
                  });
                })()}
              </div>
            </div>
            )}
          </section>
          <section className="card">
            <table className="side-table">
              <thead>
                <tr>
                  <th className={summary.favorite === summary.player_a.name ? "fav-name" : ""}>{summary.player_a.name || "-"}</th>
                  <th>{""}</th>
                  <th className={summary.favorite === summary.player_b.name ? "fav-name" : ""}>{summary.player_b.name || "-"}</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className={`num${leadBlended === "a" ? " lead" : ""}`}>{asPct(summary.prob_a_blended) ?? "-"}</td>
                  <td>
                    <Tip text={t.mixedProbInfo}>{t.mixedProb}</Tip>
                  </td>
                  <td className={`num${leadBlended === "b" ? " lead" : ""}`}>{asPct(summary.prob_b_blended) ?? "-"}</td>
                </tr>
                <tr>
                  <td className={`num${leadElo === "a" ? " lead" : ""}`}>{asInt(summary.player_a.overall_elo) ?? "-"}</td>
                  <td>
                    <Tip text={t.eloInfo}>{t.overallElo}</Tip>
                  </td>
                  <td className={`num${leadElo === "b" ? " lead" : ""}`}>{asInt(summary.player_b.overall_elo) ?? "-"}</td>
                </tr>
                <tr>
                  <td className={`num${leadSurfaceElo === "a" ? " lead" : ""}`}>{asInt(summary.player_a.surface_elo) ?? "-"}</td>
                  <td>
                    <Tip text={t.blendInfo}>{surfaceLabel ? `${surfaceLabel} ${t.surfaceElo}` : t.surfaceElo}</Tip>
                  </td>
                  <td className={`num${leadSurfaceElo === "b" ? " lead" : ""}`}>{asInt(summary.player_b.surface_elo) ?? "-"}</td>
                </tr>
                <tr>
                  <td className={`num${leadRecord === "a" ? " lead" : ""}`}>
                    {summary.player_a.surface_win_rate ? asPct(summary.player_a.surface_win_rate) : "-"}
                  </td>
                  <td>
                    <Tip text={t.surfaceRecordInfo}>{surfaceLabel ? `${surfaceLabel} ${t.surfaceRecord}` : t.surfaceRecord}</Tip>
                  </td>
                  <td className={`num${leadRecord === "b" ? " lead" : ""}`}>
                    {summary.player_b.surface_win_rate ? asPct(summary.player_b.surface_win_rate) : "-"}
                  </td>
                </tr>
                <tr>
                  <td className={`num${leadRecent === "a" ? " lead" : ""}`}>
                    {asPct(summary.player_a.recent_win_rate) ?? "-"}
                  </td>
                  <td>
                    <Tip text={t.last10Info}>{t.last10}</Tip>
                  </td>
                  <td className={`num${leadRecent === "b" ? " lead" : ""}`}>
                    {asPct(summary.player_b.recent_win_rate) ?? "-"}
                  </td>
                </tr>
                <tr>
                  <td className={`num${leadH2h === "a" ? " lead" : ""}`}>
                    {summary.h2h.wins_a ? `${summary.h2h.wins_a}` : "-"}
                  </td>
                  <td>
                    <Tip text={t.h2hInfo}>{t.h2h}</Tip>
                  </td>
                  <td className={`num${leadH2h === "b" ? " lead" : ""}`}>
                    {summary.h2h.wins_b ? `${summary.h2h.wins_b}` : "-"}
                  </td>
                </tr>
              </tbody>
            </table>
          </section>
        </div>
      )}

      {report && (
        <details className="card">
          <summary>{t.fullReport}</summary>
          <pre className="report">{report.final_text}</pre>
        </details>
      )}

      {report && (
        <details className="card">
          <summary>{t.sources}</summary>
          <p>{verOk ? t.verificationPassed : t.verificationFailed}</p>
          <p>{t.dataCutoff}</p>
          <p>
            {t.plannerSource}: {report.planner_source}
          </p>
          <p>{t.plan}: {planText(report.plan, t)}</p>
          <p>{t.model}: {report.llm_model}</p>
          <p>{t.langGraph}: {report.graph_used_langgraph ? t.yes : t.no}</p>
          <p>{t.datasetLine}</p>
          <p>{t.footnote}</p>
          <p>{t.marketNote}</p>
          {report.verification && (
            <ul>
              {verificationItems(report.verification, t).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
        </details>
      )}

      <section className="card advanced-bottom">
        <label htmlFor="advanced">{t.advancedLabel}</label>
        <textarea id="advanced" rows={2} value={advanced} onChange={(e) => setAdvanced(e.target.value)} placeholder={t.advancedPlaceholder} />
      </section>

      <footer className="site-footer">
        <a href="https://github.com/yinou23/tennis-scout-agent" target="_blank" rel="noreferrer">GitHub</a>
      </footer>
    </main>
  );
}
