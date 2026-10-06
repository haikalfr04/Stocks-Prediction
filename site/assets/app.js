"use strict";

const state = { summary: null, cache: {}, current: null };

const pct = (v, digits = 1) => (v == null ? "–" : `${(v * 100).toFixed(digits)}%`);
const num = (v, digits = 2) => (v == null ? "–" : v.toFixed(digits));
const signedPct = (v, digits = 2) => (v == null ? "–" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`);
const idr = (v) => `Rp${Math.round(v).toLocaleString("id-ID")}`;

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of [].concat(children)) node.append(child);
  return node;
}

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function baseLayout(extra = {}) {
  const text = token("--text-secondary");
  const grid = token("--grid");
  const axis = { gridcolor: grid, zerolinecolor: token("--border"), linecolor: grid, tickfont: { color: text, size: 12 } };
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", color: text, size: 12 },
    margin: { l: 56, r: 16, t: 8, b: 36 },
    hovermode: "x unified",
    hoverlabel: { bgcolor: token("--surface"), bordercolor: token("--border"), font: { color: token("--text-primary") } },
    legend: { orientation: "h", x: 0, y: 1.12, font: { color: token("--text-primary") } },
    ...extra,
    xaxis: { ...axis, ...(extra.xaxis || {}) },
    yaxis: { ...axis, ...(extra.yaxis || {}) },
  };
}

const plotConfig = { displayModeBar: false, responsive: true };

async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

async function init() {
  try {
    state.summary = await loadJSON("data/summary.json");
  } catch (err) {
    document.getElementById("meta").textContent =
      "No results found. Run `python -m stockpred.pipeline --out site/data` first.";
    return;
  }
  const { summary } = state;
  const updated = new Date(summary.generated_at);
  document.getElementById("meta").textContent =
    `Updated ${updated.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })}`;
  document.getElementById("synthetic-banner").hidden = !summary.synthetic;

  const tabs = document.getElementById("tabs");
  for (const t of summary.tickers) {
    const btn = el("button", { class: "tab", role: "tab", "aria-selected": "false", "data-symbol": t.symbol }, [
      t.symbol,
      el("small", { text: t.name }),
    ]);
    btn.addEventListener("click", () => select(t.symbol));
    tabs.append(btn);
  }
  const fromHash = location.hash.slice(1).toUpperCase();
  const first = summary.tickers.find((t) => t.symbol === fromHash) || summary.tickers[0];
  select(first.symbol);

  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.current && render(state.current));
}

async function select(symbol) {
  for (const btn of document.querySelectorAll(".tab")) {
    btn.setAttribute("aria-selected", String(btn.dataset.symbol === symbol));
  }
  history.replaceState(null, "", `#${symbol}`);
  if (!state.cache[symbol]) state.cache[symbol] = await loadJSON(`data/${symbol}.json`);
  state.current = state.cache[symbol];
  render(state.current);
}

function render(d) {
  renderKpis(d);
  renderPrice(d);
  renderEquity(d);
  renderRecent(d);
  renderHitRate(d);
  renderTable(d);
  renderImportance(d);
}

function model(d, key) {
  return d.models.find((m) => m.key === key);
}

function kpi(label, value, sub) {
  return el("div", { class: "kpi" }, [
    el("p", { class: "kpi-label", text: label }),
    value,
    el("p", { class: "kpi-sub", text: sub }),
  ]);
}

function renderKpis(d) {
  const lgbm = model(d, "lightgbm");
  const baselines = d.models.filter((m) => m.baseline);
  const bestBaseline = baselines.reduce((a, b) =>
    (b.forecast.directional_accuracy ?? 0) > (a.forecast.directional_accuracy ?? 0) ? b : a,
  );
  const f = d.forecast;
  const long = f.signal === "LONG";

  const signal = el("p", { class: "kpi-value" }, [
    signedPct(Math.expm1(f.predicted_log_return)),
    el("span", { class: `pill ${long ? "up" : "down"}`, text: long ? "▲ LONG" : "▼ CASH" }),
  ]);

  const kpis = document.getElementById("kpis");
  kpis.replaceChildren(
    kpi("Next-day forecast (LightGBM)", signal, `Predicted return after close of ${f.as_of} · last close ${idr(d.last_close)}`),
    kpi(
      "Directional accuracy",
      el("p", { class: "kpi-value", text: pct(lgbm.forecast.directional_accuracy) }),
      `Best baseline: ${pct(bestBaseline.forecast.directional_accuracy)} (${bestBaseline.label})`,
    ),
    kpi(
      "Sharpe ratio, strategy",
      el("p", { class: "kpi-value", text: num(lgbm.strategy.sharpe) }),
      `Buy & hold: ${num(d.buy_hold.sharpe)}`,
    ),
    kpi(
      "Total return, test period",
      el("p", { class: "kpi-value", text: signedPct(lgbm.strategy.total_return, 0) }),
      `Buy & hold: ${signedPct(d.buy_hold.total_return, 0)} · ${d.n_test_days} trading days`,
    ),
  );
}

function renderPrice(d) {
  const trace = {
    x: d.prices.dates,
    y: d.prices.close,
    type: "scatter",
    mode: "lines",
    name: `${d.symbol} close`,
    line: { color: token("--series-1"), width: 2 },
    hovertemplate: "Rp%{y:,.0f}<extra></extra>",
  };
  const layout = baseLayout({
    showlegend: false,
    yaxis: { tickprefix: "Rp", tickformat: ",.0f" },
    shapes: [{
      type: "rect", xref: "x", yref: "paper", x0: d.test_start, x1: d.data_end, y0: 0, y1: 1,
      fillcolor: token("--shade"), line: { width: 0 }, layer: "below",
    }],
    annotations: [{
      x: d.test_start, y: 1, xref: "x", yref: "paper", xanchor: "left", yanchor: "top",
      text: " Test period", showarrow: false, font: { color: token("--text-muted"), size: 12 },
    }],
  });
  Plotly.react("chart-price", [trace], layout, plotConfig);
}

function renderEquity(d) {
  const e = d.equity;
  const line = (y, name, color, extra = {}) => ({
    x: e.dates, y, name, type: "scatter", mode: "lines",
    line: { color, width: 2, ...extra }, hovertemplate: `${name}: Rp%{y:.2f}<extra></extra>`,
  });
  const traces = [
    line(e.buy_hold, "Buy & hold", token("--series-benchmark"), { dash: "dot" }),
    line(e.ridge, "Ridge strategy", token("--series-2")),
    line(e.lightgbm, "LightGBM strategy", token("--series-1")),
  ];
  const values = [...e.buy_hold, ...e.ridge, ...e.lightgbm].filter((v) => v != null);
  const [lo, hi] = [Math.min(...values), Math.max(...values)];
  const ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16].filter((t) => t >= lo / 1.2 && t <= hi * 1.2);
  const yaxis = { type: "log", tickvals: ticks, ticktext: ticks.map((t) => `Rp${t}`) };
  Plotly.react("chart-equity", traces, baseLayout({ yaxis }), plotConfig);
}

function renderRecent(d) {
  const r = d.recent;
  const traces = [
    {
      x: r.dates, y: r.actual, type: "bar", name: "Actual",
      marker: { color: token("--bar-neutral") }, hovertemplate: "Actual: %{y:.2%}<extra></extra>",
    },
    {
      x: r.dates, y: r.lightgbm, type: "scatter", mode: "lines", name: "LightGBM",
      line: { color: token("--series-1"), width: 2 }, hovertemplate: "LightGBM: %{y:.2%}<extra></extra>",
    },
  ];
  Plotly.react("chart-recent", traces, baseLayout({ bargap: 0.25, yaxis: { tickformat: ".1%" } }), plotConfig);
}

function renderHitRate(d) {
  const h = d.hit_rate;
  const trace = {
    x: h.dates, y: h.lightgbm, type: "scatter", mode: "lines", name: "LightGBM",
    line: { color: token("--series-1"), width: 2 }, hovertemplate: "Hit rate: %{y:.1%}<extra></extra>",
  };
  const layout = baseLayout({
    showlegend: false,
    yaxis: { tickformat: ".0%" },
    shapes: [{
      type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5,
      line: { color: token("--text-muted"), width: 1, dash: "dash" },
    }],
    annotations: [{
      xref: "paper", x: 0.01, y: 0.5, xanchor: "left", yanchor: "middle",
      text: "50% = coin flip", showarrow: false, bgcolor: token("--surface"),
      font: { color: token("--text-muted"), size: 12 },
    }],
  });
  Plotly.react("chart-hit", [trace], layout, plotConfig);
}

function renderImportance(d) {
  const items = [...d.feature_importance].reverse();
  const trace = {
    x: items.map((i) => i.gain), y: items.map((i) => i.feature), type: "bar", orientation: "h",
    marker: { color: token("--series-1") }, hovertemplate: "%{y}: %{x:.1%}<extra></extra>",
  };
  const layout = baseLayout({
    showlegend: false, hovermode: "closest", bargap: 0.3,
    margin: { l: 120, r: 16, t: 8, b: 36 },
    xaxis: { tickformat: ".0%" },
    yaxis: { automargin: true },
  });
  Plotly.react("chart-importance", [trace], layout, plotConfig);
}

function renderTable(d) {
  const cols = [
    ["Model", (m) => m.label],
    ["Dir. accuracy", (m) => pct(m.forecast?.directional_accuracy)],
    ["R² vs random walk", (m) => (m.forecast ? signedPct(m.forecast.r2_vs_random_walk, 2) : "–")],
    ["Info. coef.", (m) => num(m.forecast?.information_coefficient, 3)],
    ["MAE", (m) => pct(m.forecast?.mae, 2)],
    ["Sharpe", (m) => num(m.strategy.sharpe)],
    ["CAGR", (m) => pct(m.strategy.cagr)],
    ["Max drawdown", (m) => pct(m.strategy.max_drawdown)],
    ["Time invested", (m) => pct(m.strategy.exposure, 0)],
    ["Trades", (m) => String(m.strategy.trades)],
  ];
  const rows = [
    ...d.models.map((m) => ({ ...m, cls: m.baseline ? "baseline" : m.key === "lightgbm" ? "highlight" : "" })),
    { label: "Buy & hold (benchmark)", strategy: d.buy_hold, cls: "benchmark" },
  ];
  const table = document.getElementById("table-models");
  table.replaceChildren(
    el("thead", {}, el("tr", {}, cols.map(([h]) => el("th", { scope: "col", text: h })))),
    el("tbody", {}, rows.map((m) => el("tr", { class: m.cls }, cols.map(([, f]) => el("td", { text: f(m) }))))),
  );
}

init();
