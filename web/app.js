"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
};
const vclass = (v) => "v-" + String(v).split(" ")[0];

let STATUS = null;
let LAST_CLAIM = null;
let LAST_PARAMS = {};
const charts = {};

/* ------------------------------------------------------------------ theme + tabs */
(function theme() {
  let saved = null;
  try { saved = localStorage.getItem("rf-theme"); } catch (e) {}
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-btn").onclick = () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("rf-theme", next); } catch (e) {}
    redrawCharts();
  };
})();

function showTab(name) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$("section.tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + name));
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  if (name === "corpus") loadCorpus();
  if (name === "bench") loadLastBench();
  if (name === "eval") loadEval();
  if (name === "history") loadHistory();
  if (name === "compare") { drawWeightChart(); loadCmpBench(); }
}
$$("#tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));

/* ------------------------------------------------------------------ status */
async function pollStatus() {
  try {
    STATUS = await api("/api/status");
    $("#st-dot").className = "dot " + (STATUS.ingesting ? "busy" : "ok");
    $("#st-text").textContent = `${STATUS.corpus_size.toLocaleString()} articles · ${STATUS.feeds} feeds · ${STATUS.max_threads} OpenMP threads` +
      (STATUS.ingesting ? " · ingesting…" : "");
  } catch (e) {
    $("#st-dot").className = "dot";
    $("#st-text").textContent = "server offline";
  }
}

/* ------------------------------------------------------------------ verify */
const PARAM_META = [
  ["M", "M: retrieve-wide pool", "number"], ["K", "K: clusters kept", "number"],
  ["tau", "τ: duplicate threshold", "number", 0.01], ["lam", "λ: independence weight", "number", 0.05],
  ["theta", "θ: evidence sufficiency", "number", 0.05], ["delta", "δ: disagreement band", "number", 0.01],
  ["min_rel", "relevance floor (cos)", "number", 0.01], ["threads", "threads (0 = auto)", "number"],
  ["schedule", "OpenMP schedule", "select"],
];

function buildParams(defaults) {
  $("#param-grid").innerHTML = PARAM_META.map(([k, label, type, step]) => {
    if (type === "select")
      return `<label class="field"><span>${label}</span><select data-p="${k}">${["static", "dynamic", "guided"].map((o) => `<option ${o === defaults[k] ? "selected" : ""}>${o}</option>`).join("")}</select></label>`;
    return `<label class="field"><span>${label}</span><input type="number" data-p="${k}" value="${defaults[k]}" ${step ? `step="${step}"` : ""}></label>`;
  }).join("");
}
const readParams = () => Object.fromEntries($$("[data-p]").map((el) => [el.dataset.p, el.value]));

async function loadExamples() {
  const fixed = [
    "Russian strike on a Ukrainian train station killed at least 13 people",
    "The Earth is flat",
    "Scientists confirmed that global warming is accelerating",
  ];
  let live = [];
  try {
    const c = await api("/api/corpus");
    const seen = new Set();
    live = c.recent.filter((r) => r.origin !== "web" && !seen.has(r.domain) && seen.add(r.domain)).slice(0, 4).map((r) => r.title);
  } catch (e) {}
  $("#examples").innerHTML = `<span class="tiny muted" style="align-self:center">Try:</span>` +
    [...live, ...fixed].map((t) => `<button class="chip" title="${esc(t)}">${esc(t)}</button>`).join("");
  $$("#examples .chip").forEach((b) => (b.onclick = () => { $("#claim").value = b.title; runVerify(); }));
}

async function runVerify(extra = {}) {
  const claim = $("#claim").value.trim();
  if (!claim) return;
  const btn = $("#verify-btn");
  btn.disabled = true;
  const web = CORPUS === "live" && $("#web-search").checked && !extra.syndicate_id;
  btn.innerHTML = `<span class="spinner"></span> ${web ? "Searching web" : "Verifying"}`;
  if (CORPUS === "fever") $("#verify-out").innerHTML = `<div class="card small muted"><span class="spinner"></span> Verifying against the FEVER Wikipedia subset…</div>`;
  else if (web) $("#verify-out").innerHTML = `<div class="card small muted"><span class="spinner"></span> Searching Google News and Wikipedia for this claim, then verifying…</div>`;
  LAST_CLAIM = claim;
  LAST_PARAMS = { ...readParams(), ...extra };
  try {
    const r = await api("/api/verify", { method: "POST", body: JSON.stringify({ claim, params: LAST_PARAMS, corpus: CORPUS, web }) });
    renderVerify(r);
  } catch (e) {
    $("#verify-out").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Verify";
  }
}
$("#verify-btn").onclick = () => runVerify();
try { const w = localStorage.getItem("rf-web"); if (w !== null) $("#web-search").checked = w === "1"; } catch (e) {}
$("#web-search").onchange = (e) => { try { localStorage.setItem("rf-web", e.target.checked ? "1" : "0"); } catch (err) {} };

/* ------------------------------------------------------------------ corpus switch + FEVER demo */
let CORPUS = "live";
let FEVER = null;
let FEVER_FILTER = "all";

function corpusDefaults(c) {
  const base = STATUS ? STATUS.defaults : {};
  return { ...base, ...((STATUS && STATUS.corpus_defaults && STATUS.corpus_defaults[c]) || {}) };
}

async function setCorpus(c) {
  if (c === "fever" && !(STATUS && STATUS.fever && STATUS.fever.available)) {
    $("#verify-out").innerHTML = `<div class="err">The FEVER demo corpus isn't built yet. Run <code>.venv/bin/python -m scripts.build_fever_demo</code> and reload.</div>`;
    return;
  }
  CORPUS = c;
  for (const [id, on] of [["#corpus-live", c === "live"], ["#corpus-fever", c === "fever"]]) {
    $(id).classList.toggle("active", on);
    $(id).setAttribute("aria-selected", on);
  }
  $("#live-input").hidden = c !== "live";
  $("#fever-input").hidden = c !== "fever";
  buildParams(corpusDefaults(c));
  $("#verify-out").innerHTML = "";
  $("#claim").value = "";
  if (c === "fever") {
    if (!FEVER) FEVER = await api("/api/fever/claims");
    const m = FEVER.meta;
    $("#fever-note").innerHTML = `Claims and pages come from the FEVER benchmark (Thorne et al., 2018). The evidence pool is <b>${m.pages.toLocaleString()} Wikipedia pages</b> from the 2017 dump: the gold pages for these ${m.claims} claims, ${m.hard_distractors.toLocaleString()} pages cited by other FEVER claims as hard distractors, and ${m.random_pages.toLocaleString()} random pages. Claims were chosen so their evidence sits in a page's first ${m.gold_max_line + 1} sentences, which a lead-only index can reach. Pick a claim to verify it; the result shows FEVER's gold answer next to RealFact's.`;
    renderFeverList();
  }
}
$("#corpus-live").onclick = () => setCorpus("live");
$("#corpus-fever").onclick = () => setCorpus("fever");

const lblClass = (l) => "lbl lbl-" + l.split(" ")[0];
const lblName = (l) => ({ SUPPORTS: "Supports", REFUTES: "Refutes", "NOT ENOUGH INFO": "Not enough info" }[l] || l);

function renderFeverList() {
  const q = $("#fever-search").value.trim().toLowerCase();
  const items = FEVER.claims.filter((c) => (FEVER_FILTER === "all" || c.label === FEVER_FILTER) && (!q || c.claim.toLowerCase().includes(q)));
  $("#fever-list").innerHTML = items.length
    ? items.map((c) => `<button class="fever-item ${c.claim === LAST_CLAIM && CORPUS === "fever" ? "current" : ""}" role="option" data-claim="${esc(c.claim)}"><span class="txt">${esc(c.claim)}</span><span class="${lblClass(c.label)}" title="FEVER gold label">${lblName(c.label)}</span></button>`).join("")
    : `<div class="empty small">No claims match.</div>`;
  $$("#fever-list .fever-item").forEach((b) => (b.onclick = () => pickFever(b.dataset.claim)));
}
function pickFever(claim) {
  $("#claim").value = claim;
  runVerify().then(renderFeverList);
}
$("#fever-search").addEventListener("input", () => FEVER && renderFeverList());
$$("#fever-filter button").forEach((b) => (b.onclick = () => {
  FEVER_FILTER = b.dataset.f;
  $$("#fever-filter button").forEach((x) => x.classList.toggle("active", x === b));
  renderFeverList();
}));
$("#fever-random").onclick = () => {
  const pool = FEVER.claims.filter((c) => FEVER_FILTER === "all" || c.label === FEVER_FILTER);
  pickFever(pool[Math.floor(Math.random() * pool.length)].claim);
};

function feverPanel(f) {
  const rank = f.gold_retrieved_rank;
  return `<div class="gold" style="margin-bottom:16px">
    <div class="gold-row">
      <div><div class="eyebrow">FEVER gold label</div><div style="margin-top:6px"><span class="${lblClass(f.label)}" style="font-size:12px">${lblName(f.label)}</span></div></div>
      <div><div class="eyebrow">RealFact answer, mapped</div><div style="margin-top:6px"><span class="${lblClass(f.system_label)}" style="font-size:12px">${lblName(f.system_label)}</span></div></div>
      <div><div class="eyebrow">Match</div><div style="margin-top:4px" class="${f.correct ? "yes" : "no"}">${f.correct ? "✓ agrees with FEVER" : "✗ differs from FEVER"}</div></div>
      ${f.gold_pages.length ? `<div><div class="eyebrow">Gold page retrieved?</div><div style="margin-top:4px" class="${rank ? "yes" : "no"}">${rank ? `✓ cluster #${rank}` : "✗ not in top-K"}</div></div>` : ""}
    </div>
    ${f.evidence.length ? `<div class="small" style="margin-top:12px"><b>FEVER's annotated evidence</b><ul>${f.evidence.map((e) => `<li><i>${esc(e.page)}</i>, sentence ${e.line}: ${esc(e.sentence)}</li>`).join("")}</ul></div>`
      : `<p class="small muted" style="margin:10px 0 0">FEVER marks this claim as having no supporting or refuting evidence in Wikipedia, so the correct answer is <b>Unverified</b>.</p>`}
    <p class="tiny muted" style="margin:10px 0 0">Mapping: Supported → SUPPORTS, Contradicted → REFUTES, Unverified and Conflicting Evidence → NOT ENOUGH INFO (methodology §I-B).</p>
  </div>`;
}
$("#claim").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runVerify(); } });

function probsBar(pr) {
  return `<div class="probs" title="SUPPORT ${fmt(pr.SUPPORT)} · NEUTRAL ${fmt(pr.NEUTRAL)} · CONTRADICT ${fmt(pr.CONTRADICT)}">
    <i style="width:${pr.SUPPORT * 100}%;background:var(--sup)"></i><i style="width:${pr.NEUTRAL * 100}%;background:var(--neu-soft)"></i><i style="width:${pr.CONTRADICT * 100}%;background:var(--con)"></i></div>`;
}

function renderVerify(r) {
  const p = r.params, a = r.aggregate, rw = r.retrieve_wide, par = r.parallel, t = r.timings_ms;
  const needle = ((r.S + 1) / 2) * 100;
  const sim = r.syndication_test;
  const ablOrder = ["independence_log", "one_vote", "naive_count", "filter_off"];

  const stages = [
    ...(r.web ? [["Web search", `Google News + Wikipedia · +${r.web.added} new`, t.web_search, false]] : []),
    ["Embed claim", "BGE-small · 384-d", t.embed_claim, false],
    ["OpenMP exact scan", `top-M · ${par.threads}T${par.threads_auto ? " (auto)" : ""} ${par.schedule}`, t.retrieval_kernel, true],
    ["Cluster duplicates", `τ = ${p.tau} → top-${p.K}`, t.clustering, true],
    ["NLI per cluster", "SUPPORT / CONTRADICT / NEUTRAL", t.nli, false],
    ["Weighted aggregation", `w = 1 + ${p.lam}·log₂n`, t.aggregation, true],
  ];

  $("#verify-out").innerHTML = `
  ${r.saved ? historyBanner(r.saved) : ""}
  ${r.fever ? feverPanel(r.fever) : ""}
  ${r.web ? webPanel(r.web) : ""}
  ${sim ? `<div class="callout warn" style="margin-bottom:16px"><b>Syndication stress test active:</b> article #${sim.article_id} has been republished ×${sim.copies} under simulated outlets (<code>wire-copy-N.simulated</code>), for this query only. Nothing is written to the corpus. Compare the verdict table below. <button class="btn sm" id="clear-sim" style="margin-left:8px">Clear simulation</button></div>` : ""}
  <div class="verdict-wrap">
    <div class="verdict ${vclass(r.verdict)}">
      <div class="eyebrow" style="color:inherit;opacity:.8">RealFact verdict</div>
      <div class="v-label">${esc(r.verdict)}</div>
      <div class="claim-echo">“${esc(r.claim)}”</div>
      <div class="meter"><div class="bar"><div class="needle" style="left:${needle}%"></div></div>
        <div class="labels"><span>contradicted −1</span><span>S = ${fmt(r.S)}</span><span>+1 supported</span></div></div>
      <dl class="kv" style="margin-top:14px;color:var(--ink)">
        <dt>D (support share)</dt><dd>${r.D === null ? "— (no polar evidence)" : fmt(r.D)}</dd>
        <dt>Polar mass Σw·p</dt><dd>${fmt(a.polar_mass)} ${a.polar_mass < p.theta ? "&lt;" : "≥"} θ = ${p.theta}</dd>
        <dt>Disagreement band</dt><dd>${p.delta} &lt; D &lt; ${fmt(1 - p.delta)}</dd>
        <dt>Independent clusters</dt><dd>${r.evidence.length}</dd>
      </dl>
    </div>
    <div class="stack">
      <div class="grid g3">
        <div class="card stat"><div class="n">${rw.corpus_size.toLocaleString()}</div><div class="l">articles scanned exactly (N)</div></div>
        <div class="card stat"><div class="n">${rw.M_returned.toLocaleString()}</div><div class="l">top-M returned · ${rw.above_relevance_floor} above relevance floor ${p.min_rel}</div></div>
        <div class="card stat novel"><div class="n">${rw.clusters_kept} <span class="small muted">from ${rw.articles_in_kept_clusters}</span></div><div class="l">independent clusters from articles · ${rw.duplicates_collapsed} copies collapsed</div></div>
      </div>
      <div class="card">
        <div class="eyebrow" style="margin-bottom:10px">Pipeline stages for this claim (ms) · blue = novel stage</div>
        <div class="pipe">${stages.map(([nm, sub, ms, nov], i) => `${i ? '<span class="arrow">→</span>' : ""}<div class="stage ${nov ? "novel" : ""}">${nov ? '<span class="tag">novel</span>' : ""}<div class="nm">${nm}</div><div class="tiny muted">${sub}</div><div class="t">${fmt(ms, ms < 10 ? 3 : 1)} ms</div></div>`).join("")}</div>
        <div class="small" style="margin-top:12px;display:flex;gap:18px;flex-wrap:wrap">
          <span>Scan: OpenMP <b class="mono">${fmt(par.kernel_ms, 3)} ms</b> vs serial build <b class="mono">${fmt(par.serial_ms, 3)} ms</b> (×${fmt(par.speedup)} for this one query)</span>
          <span class="${par.ids_match_serial ? "yes" : "no"}">${par.ids_match_serial ? "✓ IDs identical to serial build" : "✗ IDs differ from serial"}</span>
          <span class="muted">Retrieve-then-truncate would have kept only <b>${rw.truncate_then_cluster_independent}</b> independent stories in K = ${p.K} (${rw.truncate_then_cluster_slots_wasted} slots wasted)</span>
        </div>
        ${rw.corpus_size < 50000 ? `<p class="tiny muted" style="margin:8px 0 0">With a corpus this small (${rw.corpus_size.toLocaleString()} ${r.corpus === "fever" ? "pages" : "articles"}), OpenMP start-up costs about as much as the scan itself. See the Parallel benchmark tab for 10K–500K corpora.</p>` : ""}
      </div>
    </div>
  </div>

  <div class="section-head"><h2>Existing approaches vs RealFact on this same evidence</h2><p>Same retrieval and NLI; only the counting rule changes</p></div>
  <div class="card tbl-wrap"><table>
    <thead><tr><th>Aggregation rule</th><th>Verdict</th><th class="num">S</th><th class="num">D</th><th class="num">Σ<sub>σ≠0</sub> w·p</th><th class="num">votes Σw</th></tr></thead>
    <tbody>${ablOrder.map((k) => { const m = r.ablations[k]; return `<tr class="${k === "independence_log" ? "ours" : ""}"><td>${esc(m.label)}</td><td><span class="vchip ${vclass(m.verdict)}">${esc(m.verdict)}</span></td><td class="num">${fmt(m.S)}</td><td class="num">${fmt(m.D)}</td><td class="num">${fmt(m.polar_mass)}</td><td class="num">${fmt(m.n_votes, 1)}</td></tr>`; }).join("")}</tbody>
  </table>
  <p class="tiny muted" style="margin:8px 0 0">“Filter off” is the usual retrieve-then-classify setup: raw top-K articles, one vote each. “Naive counting” gives every syndicated copy a vote. With no syndication in the pool all rules agree, which is the expected control result (methodology §III-D). Press “Simulate syndication” on a card to see them diverge.</p></div>

  <div class="section-head"><h2>Cited evidence</h2><p>One NLI call per independent cluster, on its highest-scoring representative</p></div>
  <div id="ev-list">${r.evidence.length ? r.evidence.map((e) => evidenceCard(e, p)).join("") : `<div class="card empty">No article cleared the relevance floor (cos ≥ ${p.min_rel}). The corpus has no evidence for this claim, so the verdict is <b>Unverified</b> rather than a guess.</div>`}</div>`;

  $$(".syn-btn").forEach((b) => (b.onclick = () => runVerify({ syndicate_id: +b.dataset.id, syndicate_n: 10 })));
  $$(".mem-btn").forEach((b) => (b.onclick = () => { const m = b.closest(".ev").querySelector(".members"); m.hidden = !m.hidden; }));
  const cs = $("#clear-sim");
  if (cs) cs.onclick = () => runVerify();
  const rr = $("#rerun-btn");
  if (rr) rr.onclick = () => rerun(r.saved);
}

function webPanel(w) {
  const old = w.found - w.added;
  return `<div class="card" style="margin-bottom:16px">
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <div><div class="eyebrow">Web search for this claim</div>
        <div class="small" style="margin-top:4px">Query <code>${esc(w.query)}</code> · ${w.found} results · <b>${w.added} new</b> added to the corpus${old ? ` · ${old} already there` : ""} · ${fmt(w.seconds, 1)} s</div></div>
      ${w.items.length ? `<button class="btn sm" id="web-items-btn" style="margin-left:auto" onclick="this.closest('.card').querySelector('.web-items').hidden^=1">Show results</button>` : ""}
    </div>
    ${w.errors.length ? `<div class="err small" style="margin-top:8px">${w.errors.map(esc).join("<br>")}</div>` : ""}
    <ul class="clean small web-items" hidden>${w.items.map((i) => `<li><span class="muted tiny">${esc(i.source)} · ${esc(i.domain)}</span>${i.new ? "" : ' <span class="tiny muted">(already in corpus)</span>'} · <a href="${esc(i.url)}" target="_blank" rel="noopener">${esc(i.title)}</a></li>`).join("")}</ul>
    <p class="tiny muted" style="margin:8px 0 0">Web results are ordinary corpus entries from here on: the verdict below still comes only from what clears the relevance floor, after duplicate clustering. Google News often returns several outlets carrying one wire story, which collapse into a single cluster.</p>
  </div>`;
}

function evidenceCard(e, p) {
  const r = e.rep;
  const date = r.published_at ? new Date(r.published_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "undated";
  const copies = Math.min(e.n_c, 24);
  return `<div class="ev">
    <div class="ev-head">
      <div class="ev-rank">#${e.rank}</div>
      <div style="min-width:0">
        <div class="ev-title"><a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)}</a></div>
        <div class="ev-meta"><span>${esc(r.domain)}</span><span>${esc(date)}</span><span class="mono">cos ${fmt(r.score, 3)}</span>${r.origin === "web" ? '<span class="web-tag">WEB SEARCH</span>' : ""}${r.simulated ? '<span class="sim-tag">SIMULATED COPY</span>' : ""}</div>
        ${r.summary ? `<div class="ev-sum">${esc(r.summary.slice(0, 320))}${r.summary.length > 320 ? "…" : ""}</div>` : ""}
      </div>
      <div class="ev-right"><span class="stance s-${e.label}">${e.label}</span><div class="tiny muted" style="margin-top:4px">p = ${fmt(e.p)}</div>${probsBar(e.probs)}</div>
    </div>
    <div class="ev-foot">
      <span title="cluster size n_c"><span class="copies">${Array.from({ length: copies }, (_, i) => `<i class="${i === 0 ? "rep" : ""}"></i>`).join("")}</span> n<sub>c</sub> = ${e.n_c}${e.domains.length > 1 ? ` across ${e.domains.length} outlets` : ""}</span>
      <span class="w">w<sub>c</sub> = 1 + ${p.lam}·log₂${e.n_c} = ${fmt(e.w_log)}</span>
      <span class="w muted">naive w = ${e.n_c}</span>
      <span class="w">contribution w·p·σ = ${e.contribution >= 0 ? "+" : ""}${fmt(e.contribution)}</span>
      <span style="margin-left:auto;display:flex;gap:6px">
        ${e.n_c > 1 ? `<button class="btn sm mem-btn">Show ${e.n_c} copies</button>` : ""}
        ${r.simulated ? "" : `<button class="btn sm syn-btn" data-id="${r.id}" title="Republish this article ×10 under simulated outlets and re-run the pipeline">Simulate syndication ×10</button>`}
      </span>
    </div>
    ${e.n_c > 1 ? `<ul class="members clean" hidden>${e.members.map((m) => `<li><span class="mono tiny">${fmt(m.score, 3)}</span> · ${esc(m.domain)} ${m.simulated ? '<span class="sim-tag">SIM</span>' : ""} · <a href="${esc(m.url)}" target="_blank" rel="noopener">${esc(m.title)}</a></li>`).join("")}</ul>` : ""}
  </div>`;
}

/* ------------------------------------------------------------------ diagrams */
function retrieveDiagram() {
  const pal = ["var(--con)", "var(--sup)", "var(--conf)", "var(--accent)", "#8e5bd3", "#2aa3a3", "#c4572b", "#6b8e23", "#d14d8a", "#5a6acf"];
  const box = (x, y, c, op = 1) => `<rect x="${x}" y="${y}" width="26" height="20" rx="4" fill="${c}" opacity="${op}"/>`;
  const row = (y, colors, ops) => colors.map((c, i) => box(128 + i * 30, y, c, ops ? ops[i] : 1)).join("");
  const exist = ["var(--con)", "var(--con)", "var(--con)", "var(--con)", "var(--con)", "var(--con)", "var(--con)", "var(--sup)", "var(--conf)", "var(--accent)"];
  const existAfter = ["var(--con)", "var(--sup)", "var(--conf)", "var(--accent)", "var(--line)", "var(--line)", "var(--line)", "var(--line)", "var(--line)", "var(--line)"];
  // Labels sit in their own 128-unit column; boxes run from x = 128 to 424; captions go under their row.
  return `<svg viewBox="0 0 430 156" role="img" aria-label="Truncate-then-cluster vs retrieve-wide">
    <text x="0" y="22" font-size="11" fill="var(--ink-2)">Existing</text><text x="0" y="35" font-size="9" fill="var(--ink-3)">top-K articles</text>
    ${row(12, exist)}
    <text x="0" y="64" font-size="9" fill="var(--ink-3)">then cluster</text>${row(50, existAfter)}
    <text x="424" y="84" font-size="10" fill="var(--con)" text-anchor="end">6 slots empty</text>
    <text x="0" y="114" font-size="11" fill="var(--accent)" font-weight="600">RealFact</text><text x="0" y="127" font-size="9" fill="var(--ink-3)">top-M → cluster → top-K</text>
    ${row(104, pal)}
    <text x="424" y="140" font-size="10" fill="var(--sup)" text-anchor="end">10 independent stories</text>
  </svg>`;
}
$("#diag-retrieve").innerHTML = retrieveDiagram();


/* ------------------------------------------------------------------ charts */
function baseOpts(xTitle, yTitle, extra = {}) {
  const ink = css("--ink-2"), line = css("--line");
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { labels: { color: ink, boxWidth: 12, font: { size: 11 } } }, tooltip: { mode: "index", intersect: false } },
    scales: {
      x: { title: { display: true, text: xTitle, color: ink }, ticks: { color: ink }, grid: { color: line }, ...(extra.x || {}) },
      y: { title: { display: true, text: yTitle, color: ink }, ticks: { color: ink }, grid: { color: line }, beginAtZero: true, ...(extra.y || {}) },
    },
  };
}
function mkChart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  const el = document.getElementById(id);
  if (!el) return;
  charts[id] = new Chart(el, cfg);
  charts[id]._cfgFn = null;
}
let redrawers = [];
function redrawCharts() { setTimeout(() => redrawers.forEach((f) => f()), 30); }

function drawWeightChart() {
  const ns = [1, 2, 4, 8, 16, 32];
  const draw = () => mkChart("weight-chart", {
    type: "line",
    data: {
      labels: ns,
      datasets: [
        { label: "naive (w = n)", data: ns, borderColor: css("--con"), backgroundColor: css("--con"), tension: 0 },
        { label: "RealFact λ = 0.5", data: ns.map((n) => 1 + 0.5 * Math.log2(n)), borderColor: css("--accent"), backgroundColor: css("--accent"), borderWidth: 3 },
        { label: "one vote (λ = 0)", data: ns.map(() => 1), borderColor: css("--ink-3"), backgroundColor: css("--ink-3"), borderDash: [5, 4] },
      ],
    },
    options: baseOpts("syndicated copies n", "votes w"),
  });
  draw();
  if (!redrawers.includes(drawWeightChart)) redrawers.push(drawWeightChart);
}

/* ------------------------------------------------------------------ benchmark */
const SCHED_COLORS = () => ({ static: css("--accent"), dynamic: css("--con"), guided: css("--sup") });
const listOf = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);

$("#bench-btn").onclick = async () => {
  const body = {
    sizes: listOf($("#b-sizes").value).map((x) => (x === "live" ? "live" : parseInt(x, 10))),
    threads: listOf($("#b-threads").value).map(Number),
    chunks: listOf($("#b-chunks").value).map(Number),
    reps: +$("#b-reps").value, M: +$("#b-M").value,
  };
  const btn = $("#bench-btn");
  btn.disabled = true;
  $("#bench-prog").hidden = false;
  try {
    const job = await api("/api/bench", { method: "POST", body: JSON.stringify(body) });
    const res = await waitJob(job.id, (j) => { $("#bench-prog i").style.width = j.progress * 100 + "%"; $("#bench-msg").textContent = j.message; });
    renderBench(res);
    $("#bench-msg").textContent = "Finished.";
  } catch (e) {
    $("#bench-msg").textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
    $("#bench-prog").hidden = true;
  }
};

async function waitJob(id, onTick) {
  for (;;) {
    const j = await api("/api/jobs/" + id);
    onTick && onTick(j);
    if (j.state === "done") return j.result;
    if (j.state === "error") throw new Error(j.message);
    await new Promise((r) => setTimeout(r, 700));
  }
}

let BENCH = null;
async function loadLastBench() {
  if (BENCH) return;
  const r = await api("/api/bench/last");
  if (r) renderBench(r);
  else $("#bench-out").innerHTML = `<div class="card empty">No benchmark yet. Press <b>Run benchmark</b>: the default sweep (5 sizes × 3 schedules × 5 thread counts) takes about a minute.</div>`;
}

const bytes = (b) => (b >= 1e9 ? (b / 1e9).toFixed(2) + " GB" : b >= 1e6 ? (b / 1e6).toFixed(1) + " MB" : (b / 1e3).toFixed(0) + " KB");

function renderBench(res) {
  BENCH = res;
  const pf = res.platform;
  const l2 = +pf.l2_perf_bytes || 0;
  const threads = res.params.threads;
  const pmax = Math.max(...threads);
  const sizes = res.sizes;

  // analysis text derived from the data
  const big = sizes[sizes.length - 1];
  const best = (s, p) => s.configs.filter((c) => c.threads === p).sort((a, b) => a.median_ms - b.median_ms)[0];
  const allMatch = sizes.every((s) => s.configs.every((c) => c.ids_match_serial));
  const bestBig = best(big, pmax);
  const staticRow = (s, p) => s.configs.find((c) => c.schedule === "static" && c.threads === p);
  const sp = threads.map((p) => staticRow(big, p)?.speedup || 0);
  let knee = threads[threads.length - 1];
  for (let i = 1; i < threads.length; i++) if (sp[i] / sp[i - 1] < 1 + 0.35 * (threads[i] / threads[i - 1] - 1)) { knee = threads[i - 1]; break; }

  $("#bench-out").innerHTML = `
  <div class="grid g4">
    <div class="card stat"><div class="n">${fmt(Math.max(...big.configs.map((c) => c.speedup)))}×</div><div class="l">best speedup at N = ${big.N.toLocaleString()} vs serial build</div></div>
    <div class="card stat"><div class="n">${bestBig.schedule}</div><div class="l">fastest schedule at ${pmax} threads (median ${fmt(bestBig.median_ms, 2)} ms)</div></div>
    <div class="card stat"><div class="n">${fmt(Math.max(...big.configs.map((c) => c.throughput_qps)), 0)}</div><div class="l">peak throughput, queries/s at N = ${big.N.toLocaleString()}</div></div>
    <div class="card stat"><div class="n ${allMatch ? "yes" : "no"}">${allMatch ? "✓ identical" : "✗ mismatch"}</div><div class="l">retrieved IDs, serial vs every parallel run</div></div>
  </div>

  <div class="section-head"><h2>Expected vs observed</h2><p>Methodology §III-D predictions checked against this run</p></div>
  <div class="card small tbl-wrap"><table>
    <thead><tr><th>Prediction</th><th>Observed on ${esc(pf.cpu)}</th></tr></thead><tbody>
    <tr><td>Speedup near-linear at 1–4 threads, then flattens at the memory-bandwidth ceiling (0.5 flop/byte) before cores run out</td><td>Static speedup at N = ${big.N.toLocaleString()}: ${threads.map((p, i) => `${p}T ×${fmt(sp[i])}`).join(", ")}. Gains slow after ~${knee} threads on ${pf.physical_cores} cores (${pf.perf_cores} performance + ${pf.efficiency_cores} efficiency).</td></tr>
    <tr><td>Static wins, because every article costs the same 384-d dot product</td><td>Fastest at ${pmax} threads: <b>${bestBig.schedule}</b>. ${bestBig.schedule === "static" ? "Matches the prediction." : "Differs from the prediction. The methodology attributes this to an external imbalance source, and Apple Silicon's mix of fast performance cores and slower efficiency cores is exactly that: static gives each core an equal share, so the slow cores finish last."} Dynamic with the default chunk (1) pays an atomic counter increment per article.</td></tr>
    <tr><td>Working set crossing the cache causes a visible throughput step</td><td>${sizes.map((s) => `${s.label}: ${bytes(s.working_set_bytes)}${l2 && s.working_set_bytes > l2 ? " (&gt; L2)" : ""}`).join(" · ")}. Performance-cluster L2 = ${l2 ? bytes(l2) : "n/a"}.</td></tr>
    <tr><td>OpenMP entry cost is charged separately from T<sub>1</sub></td><td>${sizes.map((s) => `${s.label}: ${fmt(s.openmp_entry_cost_ms, 3)} ms`).join(" · ")} (1-thread OpenMP build minus non-OpenMP serial build)</td></tr>
    </tbody></table></div>

  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h3>Speedup S<sub>p</sub> = T<sub>1</sub>/T<sub>p</sub> · N = ${big.N.toLocaleString()}</h3><div class="chart-box"><canvas id="ch-speedup"></canvas></div></div>
    <div class="card"><h3>Parallel efficiency E<sub>p</sub> = S<sub>p</sub>/p · N = ${big.N.toLocaleString()}</h3><div class="chart-box"><canvas id="ch-eff"></canvas></div></div>
    <div class="card"><h3>Corpus scaling at ${pmax} threads: median latency</h3><div class="chart-box"><canvas id="ch-scale"></canvas></div></div>
    <div class="card"><h3>Corpus scaling: throughput (queries/s)</h3><div class="chart-box"><canvas id="ch-thru"></canvas></div></div>
  </div>

  <div class="section-head"><h2>Per-size results</h2><p>median ± IQR over ${res.params.reps} runs after ${res.params.warmup} warm-ups · M = ${res.params.M.toLocaleString()}</p></div>
  ${sizes.map((s) => `
    <div class="card tbl-wrap">
      <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:baseline;margin-bottom:8px">
        <h3>${esc(s.label)}</h3>
        <span class="small muted">working set ${bytes(s.working_set_bytes)} · serial T₁ ${fmt(s.serial.median_ms, 3)} ms (±${fmt(s.serial.iqr_ms, 3)}) · OpenMP-1T ${fmt(s.omp1.median_ms, 3)} ms</span>
      </div>
      <table><thead><tr><th>Schedule</th><th class="num">Threads</th><th class="num">Median ms</th><th class="num">IQR</th><th class="num">p95</th><th class="num">Speedup</th><th class="num">Efficiency</th><th class="num">q/s</th><th class="num">CPU util.</th><th class="num" title="Karp–Flatt experimentally determined serial fraction">K–F e</th><th>IDs = serial</th></tr></thead>
      <tbody>${s.configs.map((c) => `<tr><td>${c.schedule}${c.chunk ? ` (chunk ${c.chunk})` : ""}</td><td class="num">${c.threads}${c.oversubscribed ? "*" : ""}</td><td class="num">${fmt(c.median_ms, 3)}</td><td class="num">${fmt(c.iqr_ms, 3)}</td><td class="num">${fmt(c.p95_ms, 3)}</td><td class="num">${fmt(c.speedup)}</td><td class="num">${fmt(c.efficiency * 100, 0)}%</td><td class="num">${fmt(c.throughput_qps, 0)}</td><td class="num">${fmt(c.cpu_util * 100, 0)}%</td><td class="num">${c.karp_flatt == null ? "—" : fmt(c.karp_flatt, 3)}</td><td class="${c.ids_match_serial ? "yes" : "no"}">${c.ids_match_serial ? "✓" : "✗"}</td></tr>`).join("")}</tbody></table>
      ${s.wilcoxon.length ? `<p class="small" style="margin:10px 0 0"><b>Wilcoxon signed-rank</b> on paired per-query latencies: ${s.wilcoxon.map((w) => `${w.a} vs ${w.b}: W = ${fmt(w.W, 1)}, p = ${w.p < 0.001 ? w.p.toExponential(1) : fmt(w.p, 3)}, effect r = ${fmt(w.effect_r)}`).join(" · ")}</p>` : ""}
    </div>`).join("")}
  <p class="tiny muted">* oversubscribed: more threads than physical cores.</p>

  ${courseSection(res)}
  <div class="section-head"><h2>Platform disclosure</h2><p>§III-A</p></div>
  <div class="card"><dl class="kv">
    <dt>CPU</dt><dd>${esc(pf.cpu)} · ${pf.physical_cores} physical / ${pf.logical_cores} logical (${pf.perf_cores} performance + ${pf.efficiency_cores} efficiency cores)</dd>
    <dt>Caches</dt><dd>L1d ${bytes(+pf.l1d_bytes)} · L2 ${bytes(+pf.l2_perf_bytes)} (perf cluster) / ${bytes(+pf.l2_eff_bytes)} (eff cluster)</dd>
    <dt>Memory</dt><dd>${bytes(+pf.mem_bytes)}</dd>
    <dt>Compiler</dt><dd>${esc(pf.compiler)} · ${esc(pf.flags)}</dd>
    <dt>OpenMP</dt><dd>${esc(pf.openmp_runtime)} · max threads ${pf.omp_max_threads}</dd>
    <dt>Pinning</dt><dd>OMP_PROC_BIND=${esc(pf.OMP_PROC_BIND)}, OMP_PLACES=${esc(pf.OMP_PLACES)}. ${esc(pf.pinning_note)}</dd>
    <dt>Run</dt><dd>${esc(res.finished_at || "")}</dd>
  </dl></div>`;

  const draw = () => {
    drawCourseCharts(res);
    const col = SCHED_COLORS();
    const series = (s, key) => {
      const groups = {};
      s.configs.forEach((c) => { const k = c.schedule + (c.chunk ? `/${c.chunk}` : ""); (groups[k] = groups[k] || []).push(c); });
      return Object.entries(groups).map(([k, cs]) => ({
        label: k, data: threads.map((p) => (cs.find((c) => c.threads === p) || {})[key] ?? null),
        borderColor: col[k.split("/")[0]], backgroundColor: col[k.split("/")[0]], borderDash: k.includes("/") ? [4, 3] : [], tension: 0.15,
      }));
    };
    const xy = (ds, f = (v) => v) => ds.map((d) => ({ ...d, data: d.data.map((v, i) => ({ x: threads[i], y: v === null ? null : f(v) })) }));
    const linX = { x: { type: "linear", min: 0, max: pmax, ticks: { stepSize: 2 } } };
    mkChart("ch-speedup", { type: "line", data: { datasets: [...xy(series(big, "speedup")), { label: "ideal", data: threads.map((p) => ({ x: p, y: p })), borderColor: css("--ink-3"), borderDash: [6, 4], pointRadius: 0, borderWidth: 1 }] }, options: baseOpts("threads p", "speedup", linX) });
    mkChart("ch-eff", { type: "line", data: { datasets: xy(series(big, "efficiency"), (v) => v * 100) }, options: baseOpts("threads p", "efficiency %", { ...linX, y: { max: 110 } }) });
    const at = (s, sch) => s.configs.find((c) => c.schedule === sch && c.threads === pmax && !c.chunk);
    const labels = sizes.map((s) => s.N.toLocaleString());
    const mk = (key, scale = 1) => ["static", "dynamic", "guided"].filter((sch) => sizes[0].configs.some((c) => c.schedule === sch))
      .map((sch) => ({ label: `${sch} ×${pmax}`, data: sizes.map((s) => (at(s, sch) || {})[key] * scale), borderColor: col[sch], backgroundColor: col[sch] }));
    mkChart("ch-scale", { type: "line", data: { labels, datasets: [...mk("median_ms"), { label: "serial T₁", data: sizes.map((s) => s.serial.median_ms), borderColor: css("--ink-3"), borderDash: [6, 4] }] }, options: baseOpts("corpus size N", "median ms") });
    mkChart("ch-thru", { type: "bar", data: { labels, datasets: mk("throughput_qps") }, options: baseOpts("corpus size N", "queries / s", { y: { type: "logarithmic", beginAtZero: false } }) });
  };
  draw();
  loadBaselineCompare(res);
  redrawers = redrawers.filter((f) => f.name !== "benchDraw");
  Object.defineProperty(draw, "name", { value: "benchDraw" });
  redrawers.push(draw);
}

/* ------------------------------------------------------------------ course analyses */
function courseSection(res) {
  if (!res.load_balance) return "";
  const lb = res.load_balance, cs = res.chunk_sweep, ws = res.weak_scaling, bt = res.batched;
  const big = res.sizes[res.sizes.length - 1];
  const kf = big.configs.filter((c) => c.schedule === "guided" && c.karp_flatt != null);
  return `
  <div class="section-head"><h2>Parallel Computing analyses</h2><p>Course concepts applied to the retrieval kernel</p></div>
  <div id="v1v2"></div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h3>Load balance: per-thread busy time (${lb[0].threads} threads, N = ${cs.N.toLocaleString()})</h3>
      <div class="chart-box"><canvas id="ch-lb"></canvas></div>
      <div class="tbl-wrap" style="margin-top:8px"><table class="small"><thead><tr><th>Schedule</th><th class="num">kernel ms</th><th class="num">max/mean busy</th><th class="num">idle %</th><th class="num">rows min–max</th></tr></thead>
      <tbody>${lb.map((r) => `<tr><td>${r.schedule}${r.chunk ? ` (chunk ${r.chunk})` : ""}</td><td class="num">${fmt(r.kernel_ms, 2)}</td><td class="num">${fmt(r.imbalance, 2)}</td><td class="num">${fmt(r.idle_pct, 0)}%</td><td class="num">${Math.min(...r.rows).toLocaleString()}–${Math.max(...r.rows).toLocaleString()}</td></tr>`).join("")}</tbody></table></div>
      <p class="tiny muted" style="margin:8px 0 0">The chart leaves out configurations more than 3× slower than the best (dynamic, chunk 1). Static gives every thread the same number of rows, so threads on slower efficiency cores finish last. Guided hands out shrinking chunks, so fast cores take more rows (see the rows column) and finish together. Dynamic with chunk 1 balances well, but every row costs an atomic counter update.</p></div>
    <div class="card"><h3>Chunk-size sweep (${cs.threads} threads, N = ${cs.N.toLocaleString()})</h3>
      <div class="chart-box"><canvas id="ch-chunk"></canvas></div>
      <p class="tiny muted" style="margin:8px 0 0">Small dynamic chunks pay scheduling overhead on every chunk; very large chunks bring back static's imbalance. The sweet spot lies in between. Dashed line = static.</p></div>
    <div class="card"><h3>Weak scaling (Gustafson): ${ws[0].n_per_thread.toLocaleString()} rows per thread</h3>
      <div class="chart-box"><canvas id="ch-weak"></canvas></div>
      <p class="tiny muted" style="margin:8px 0 0">Work grows with p, so ideal weak efficiency stays at 100%. It falls once total bandwidth demand exceeds the memory system, which is the same ceiling that limits strong scaling.</p></div>
    <div class="card"><h3>Karp–Flatt serial fraction (guided, N = ${big.N.toLocaleString()})</h3>
      <div class="chart-box"><canvas id="ch-kf"></canvas></div>
      <p class="tiny muted" style="margin:8px 0 0">e = (1/S − 1/p)/(1 − 1/p). A constant e would mean a fixed serial part (Amdahl). An e that <b>rises</b> with p, as here, points to an overhead that grows with p: memory-bandwidth contention, plus the P-heap merge. The merge is serial but small.</p></div>
    <div class="card" style="grid-column:1/-1"><h3>Roofline: multi-query blocking (${bt.threads} threads, N = ${bt.N.toLocaleString()})</h3>
      <div class="grid g2"><div class="chart-box"><canvas id="ch-roof"></canvas></div><div class="chart-box"><canvas id="ch-batch"></canvas></div></div>
      <p class="tiny muted" style="margin:8px 0 0">One query reads 768 MB to do 2 flops per 4-byte float: 0.5 flop/byte, deep in the memory-bound region. Scoring Q queries per row load raises intensity to 0.5·Q flop/byte, so throughput climbs until the kernel becomes compute-bound. With M = 10,000, each thread keeps Q heaps of 10K entries, and heap maintenance eats most of the gain. That trade-off is why batching suits evaluation runs with a small M.</p></div>
  </div>`;
}

function drawCourseCharts(res) {
  if (!res.load_balance) return;
  const col = SCHED_COLORS();
  const lbAll = res.load_balance;
  const fastest = Math.min(...lbAll.map((r) => r.kernel_ms));
  const lb = lbAll.filter((r) => r.kernel_ms < 3 * fastest);  // chunk-1 dynamic stays in the table only
  const lbColor = (r) => (r.schedule === "dynamic" && r.chunk ? css("--conf") : col[r.schedule]);
  mkChart("ch-lb", { type: "bar", data: { labels: lb[0].busy_ms.map((_, i) => "T" + i),
    datasets: lb.map((r) => ({ label: `${r.schedule}${r.chunk ? "/" + r.chunk : ""}`, data: r.busy_ms, backgroundColor: lbColor(r) })) },
    options: baseOpts("thread", "busy ms in loop") });
  const cs = res.chunk_sweep.rows;
  const chunks = [...new Set(cs.filter((r) => r.chunk).map((r) => r.chunk))];
  const st = cs.find((r) => r.schedule === "static").median_ms;
  mkChart("ch-chunk", { type: "line", data: { labels: chunks.map(String), datasets: [
    ...["dynamic", "guided"].map((sch) => ({ label: sch, data: chunks.map((c) => cs.find((r) => r.schedule === sch && r.chunk === c).median_ms), borderColor: col[sch], backgroundColor: col[sch] })),
    { label: "static", data: chunks.map(() => st), borderColor: col.static, borderDash: [6, 4], pointRadius: 0 }] },
    options: baseOpts("chunk size (rows)", "median ms", { y: { type: "logarithmic", beginAtZero: false } }) });
  const ws = res.weak_scaling;
  mkChart("ch-weak", { type: "line", data: { datasets: [
    ...ws.map((w) => ({ label: w.schedule, data: w.points.map((pt) => ({ x: pt.threads, y: pt.weak_efficiency * 100 })), borderColor: col[w.schedule], backgroundColor: col[w.schedule] })),
    { label: "ideal", data: [{ x: 1, y: 100 }, { x: Math.max(...res.params.threads), y: 100 }], borderColor: css("--ink-3"), borderDash: [6, 4], pointRadius: 0 }] },
    options: baseOpts("threads p", "weak efficiency %", { x: { type: "linear" }, y: { max: 110 } }) });
  const big = res.sizes[res.sizes.length - 1];
  mkChart("ch-kf", { type: "line", data: { datasets: ["static", "guided"].map((sch) => ({ label: sch,
    data: big.configs.filter((c) => c.schedule === sch && c.karp_flatt != null && !c.chunk).map((c) => ({ x: c.threads, y: c.karp_flatt })), borderColor: col[sch], backgroundColor: col[sch] })) },
    options: baseOpts("threads p", "serial fraction e", { x: { type: "linear" } }) });
  const bt = res.batched.rows;
  const Ms = [...new Set(bt.map((r) => r.M))];
  const mcol = [css("--accent"), css("--con")];
  const bw = Math.max(...bt.map((r) => r.gbytes_per_s));
  const ais = [0.25, 0.5, 1, 2, 4, 8, 16, 32];
  mkChart("ch-roof", { type: "scatter", data: { datasets: [
    { label: `bandwidth roof (${fmt(bw, 0)} GB/s measured)`, type: "line", data: ais.map((a) => ({ x: a, y: a * bw })), borderColor: css("--ink-3"), borderDash: [6, 4], pointRadius: 0 },
    ...Ms.map((M, i) => ({ label: `attained, M = ${M.toLocaleString()}`, data: bt.filter((r) => r.M === M).map((r) => ({ x: r.flop_per_byte, y: r.gflops })), borderColor: mcol[i], backgroundColor: mcol[i], showLine: true }))] },
    options: baseOpts("arithmetic intensity (flop/byte)", "GFLOP/s", { x: { type: "logarithmic" }, y: { type: "logarithmic", beginAtZero: false } }) });
  mkChart("ch-batch", { type: "bar", data: { labels: [...new Set(bt.map((r) => "Q=" + r.Q))],
    datasets: Ms.map((M, i) => ({ label: `M = ${M.toLocaleString()}`, data: bt.filter((r) => r.M === M).map((r) => r.throughput_qps), backgroundColor: mcol[i] })) },
    options: baseOpts("queries per pass", "queries / s") });
}

async function loadBaselineCompare(res) {
  const el = document.getElementById("v1v2");
  if (!el) return;
  const v1 = await api("/api/bench/baseline");
  if (!v1) { el.innerHTML = ""; return; }
  const pick = (r) => r.sizes[r.sizes.length - 1];
  const a = pick(v1), b = pick(res);
  if (a.N !== b.N) { el.innerHTML = ""; return; }
  const best = (s) => s.configs.slice().sort((x, y) => x.median_ms - y.median_ms)[0];
  const ba = best(a), bb = best(b);
  el.innerHTML = `<div class="card tbl-wrap"><h3>Kernel v1 → v2 (N = ${b.N.toLocaleString()})</h3>
    <table style="margin-top:8px"><thead><tr><th></th><th class="num">v1: single accumulator</th><th class="num">v2: 16-way ILP</th><th class="num">change</th></tr></thead><tbody>
    <tr><td>Serial T₁ (no OpenMP)</td><td class="num">${fmt(a.serial.median_ms, 2)} ms</td><td class="num">${fmt(b.serial.median_ms, 2)} ms</td><td class="num yes">${fmt(a.serial.median_ms / b.serial.median_ms, 2)}× faster</td></tr>
    <tr><td>Single-core bandwidth</td><td class="num">${fmt(b.working_set_bytes / a.serial.median_ms / 1e6, 1)} GB/s</td><td class="num">${fmt(b.working_set_bytes / b.serial.median_ms / 1e6, 1)} GB/s</td><td></td></tr>
    <tr><td>Best parallel (${esc(ba.schedule)} ${ba.threads}T → ${esc(bb.schedule)} ${bb.threads}T)</td><td class="num">${fmt(ba.median_ms, 2)} ms</td><td class="num">${fmt(bb.median_ms, 2)} ms</td><td class="num yes">${fmt(ba.median_ms / bb.median_ms, 2)}× faster</td></tr>
    <tr><td>Speedup over its own T₁</td><td class="num">${fmt(ba.speedup, 2)}×</td><td class="num">${fmt(bb.speedup, 2)}×</td><td class="num muted">lower</td></tr>
    <tr><td>Threads needed to reach ~90% of best</td><td class="num">${a.configs.filter((c) => c.median_ms <= ba.median_ms / 0.9).sort((x, y) => x.threads - y.threads)[0].threads}</td><td class="num">${b.configs.filter((c) => c.median_ms <= bb.median_ms / 0.9).sort((x, y) => x.threads - y.threads)[0].threads}</td><td></td></tr>
    </tbody></table>
    <p class="tiny muted" style="margin:8px 0 0">v1 accumulated every row into one vector register, so each add waited for the one before it (latency-bound). v2 uses four independent registers, with the summation order written out so results stay bit-exact. Absolute time improves, but speedup <em>drops</em>: a faster core reaches the shared memory-bandwidth ceiling with fewer threads. That is the roofline in practice. Speedup only means something next to T₁.</p></div>`;
}

/* ------------------------------------------------------------------ self-test */
$("#selftest-btn").onclick = async () => {
  const b = $("#selftest-btn");
  b.disabled = true;
  b.innerHTML = `<span class="spinner"></span> Running`;
  try {
    const r = await api("/api/selftest", { method: "POST" });
    $("#selftest-out").innerHTML = `<p class="small ${r.passed === r.total ? "yes" : "no"}" style="margin:12px 0 6px">${r.passed}/${r.total} checks passed</p>
      <div class="tbl-wrap"><table class="small"><tbody>${r.results.map((t) => `<tr><td class="${t.skipped ? "muted" : t.ok ? "yes" : "no"}">${t.skipped ? "skip" : t.ok ? "✓" : "✗"}</td><td>${esc(t.doc || t.name)}</td><td class="muted">${esc(t.detail)}</td><td class="num">${fmt(t.ms, 0)} ms</td></tr>`).join("")}</tbody></table></div>`;
  } catch (e) {
    $("#selftest-out").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  } finally {
    b.disabled = false;
    b.textContent = "Run self-test";
  }
};

async function loadCmpBench() {
  const r = BENCH || (await api("/api/bench/last"));
  if (!r) { $("#cmp-bench").innerHTML = `<span class="muted">Run the Parallel benchmark tab to fill this panel with measured numbers.</span>`; return; }
  const big = r.sizes[r.sizes.length - 1];
  const s1 = big.serial.median_ms;
  const bestc = big.configs.slice().sort((a, b) => a.median_ms - b.median_ms)[0];
  $("#cmp-bench").innerHTML = `<table>
    <tr><td>Corpus</td><td class="num">${esc(big.label)}</td></tr>
    <tr><td>Serial scan (no OpenMP)</td><td class="num">${fmt(s1, 2)} ms</td></tr>
    <tr><td>RealFact best (${bestc.schedule}, ${bestc.threads}T)</td><td class="num"><b>${fmt(bestc.median_ms, 2)} ms</b></td></tr>
    <tr><td>Speedup / throughput</td><td class="num">×${fmt(bestc.speedup)} · ${fmt(bestc.throughput_qps, 0)} q/s</td></tr>
    <tr><td>IDs identical to serial</td><td class="num ${big.configs.every((c) => c.ids_match_serial) ? "yes" : "no"}">${big.configs.every((c) => c.ids_match_serial) ? "✓ all configs" : "✗"}</td></tr></table>`;
}

/* ------------------------------------------------------------------ corpus */
$("#ingest-btn").onclick = async () => {
  const b = $("#ingest-btn");
  b.disabled = true;
  b.innerHTML = `<span class="spinner"></span> Fetching`;
  try {
    const job = await api("/api/ingest", { method: "POST" });
    const r = await waitJob(job.id);
    b.textContent = `+${r.added} new`;
    loadCorpus();
    pollStatus();
  } catch (e) {
    b.textContent = "Error";
  } finally {
    setTimeout(() => { b.disabled = false; b.textContent = "Fetch feeds now"; }, 2500);
  }
};

async function loadCorpus() {
  const c = await api("/api/corpus");
  $("#c-title").textContent = `${c.articles.toLocaleString()} articles from ${c.domains.length} outlets`;
  $("#c-sub").textContent = `Embedding matrix ${bytes(c.matrix_bytes)} (N × 384 float32, contiguous) · headline + lead from permitted RSS only`;
  const maxN = Math.max(1, ...c.domains.map((d) => d.n));
  $("#corpus-out").innerHTML = `
  <div class="grid g2">
    <div class="card"><h3>Articles per outlet</h3>
      <div style="margin-top:10px">${c.domains.map((d) => `<div style="display:grid;grid-template-columns:140px 1fr 50px;gap:8px;align-items:center;font-size:12.5px;margin:3px 0"><span>${esc(d.domain)}</span><span style="background:var(--surface-2);border-radius:4px;height:10px"><i style="display:block;height:100%;width:${(d.n / maxN) * 100}%;background:var(--accent);border-radius:4px"></i></span><span class="mono tiny" style="text-align:right">${d.n}</span></div>`).join("")}</div></div>
    <div class="stack">
      <div class="card tbl-wrap"><h3>Ingestion log</h3><table style="margin-top:8px"><thead><tr><th>When (UTC)</th><th class="num">Feeds ok</th><th class="num">Seen</th><th class="num">Exact dups</th><th class="num">Added</th><th class="num">s</th></tr></thead>
        <tbody>${c.ingest_log.map((l) => `<tr><td class="tiny">${esc(l.at.slice(0, 19).replace("T", " "))}</td><td class="num">${l.feeds_ok}${l.feeds_failed ? `/${l.feeds_ok + l.feeds_failed}` : ""}</td><td class="num">${l.seen}</td><td class="num">${l.exact_dups}</td><td class="num">${l.added}</td><td class="num">${l.seconds}</td></tr>`).join("")}</tbody></table>
        <p class="tiny muted" style="margin:8px 0 0">“Exact dups” are dropped by the (domain, normalised headline) hash before embedding. Near-duplicates across outlets are kept and handled by clustering at query time.</p></div>
      <div class="card"><h3>Most recent</h3><ul class="clean small" style="margin-top:8px">${c.recent.slice(0, 12).map((r) => `<li><span class="muted tiny">${esc(r.domain)}</span>${r.origin === "web" ? ' <span class="web-tag">WEB</span>' : ""} · <a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)}</a></li>`).join("")}</ul></div>
    </div>
  </div>`;
}

/* ------------------------------------------------------------------ evaluation */
async function loadEval() {
  const r = await api("/api/eval/last");
  const howto = `
    <div class="card">
      <h3>How to run the verification-quality evaluation</h3>
      <p class="small" style="margin-top:6px">The benchmark datasets aren't bundled, because of licences and size. Put them under <code>data/datasets/</code> and run:</p>
      <div class="formula">.venv/bin/python -m eval.run_eval --dataset liar --path data/datasets/liar/test.tsv --limit 300</div>
      <div class="formula" style="margin-top:6px">.venv/bin/python -m eval.run_eval --dataset fever --path data/datasets/fever/shared_task_dev.jsonl</div>
      <div class="formula" style="margin-top:6px">.venv/bin/python -m eval.run_eval --dataset fakenewsnet --path data/datasets/fakenewsnet</div>
      <p class="small muted" style="margin-top:10px">The harness applies the label mapping from §I-B and leakage control, and scores all four aggregation rules (log-weighted, λ = 0, naive, filter off) from the same retrieval. It reports accuracy, macro and per-class P/R/F1, Unverified/Conflicting counts, LIAR's middle band separately, and FakeNewsNet's merged not-supported class.</p>
      <div class="callout warn small" style="margin-top:10px">Honest caveat: FEVER's gold evidence is Wikipedia, and LIAR's claims date from 2007–2016, while this prototype's evidence pool is <em>current news</em>. On those datasets most claims will correctly come back <b>Unverified</b>. Reproducing FEVER-style accuracy requires indexing the FEVER Wikipedia dump (≈5.4M documents) into the same engine: <code>store.add()</code> takes any texts.</div>
    </div>`;
  if (!r) { $("#eval-out").innerHTML = `<div class="card empty" style="margin-bottom:16px">No evaluation results yet.</div>` + howto; return; }
  const modes = ["independence_log", "one_vote", "naive_count", "filter_off"];
  const names = { independence_log: "RealFact log-weighted", one_vote: "λ = 0", naive_count: "naive counting", filter_off: "filter off" };
  $("#eval-out").innerHTML = Object.values(r).map((d) => `
    <div class="card tbl-wrap">
      <h3>${d.dataset === "fever-demo" ? "FEVER demo subset" : esc(d.dataset.toUpperCase())}: ${d.n} claims · corpus ${d.corpus_size.toLocaleString()} ${d.dataset === "fever-demo" ? "Wikipedia pages" : "articles"} · ${esc(d.finished_at)}</h3>
      <p class="small muted">median latency ${fmt(d.median_latency_ms, 1)} ms · p95 ${fmt(d.p95_latency_ms, 1)} ms · Recall@K (gold page in kept clusters): ${typeof d.recall_at_k === "object" ? Object.entries(d.recall_at_k).map(([k, v]) => `${k} ${fmt(v * 100, 1)}%`).join(" · ") : esc(d.recall_at_k)}${d.params && d.params.nli_model ? ` · NLI: <code>${esc(d.params.nli_model)}</code>` : ""}</p>
      ${d.dataset === "fever-demo" ? `<p class="small" style="margin:0 0 8px">Curated FEVER dev claims whose evidence is in the first sentences of one page, verified against a 25K-page Wikipedia subset. This is a demo slice, not the full FEVER benchmark, so it is not comparable to published FEVER scores.</p>` : ""}
      ${(() => { const pc = d.modes.independence_log.per_class; const maj = Object.entries(pc).sort((a, b) => b[1].support - a[1].support)[0]; const unv = (d.modes.independence_log.verdict_counts.Unverified || 0) / d.n;
        return `<p class="small">Majority-class baseline (always ${maj[0]}): <b>${fmt((maj[1].support / d.n) * 100, 1)}%</b> · claims answered Unverified: <b>${fmt(unv * 100, 1)}%</b>${unv > 0.8 ? ". Most claims found no evidence in the current news corpus, so accuracy here measures corpus coverage, not reasoning quality." : ""}</p>`; })()}
      <table><thead><tr><th>Mode</th><th class="num">Accuracy</th><th class="num">Macro F1</th><th class="num">F1 SUP</th><th class="num">F1 CON</th><th class="num">F1 NEU</th><th>Verdict counts</th></tr></thead>
      <tbody>${modes.map((m) => { const x = d.modes[m]; return `<tr class="${m === "independence_log" ? "ours" : ""}"><td>${names[m]}</td><td class="num">${fmt(x.accuracy * 100, 1)}%</td><td class="num">${fmt(x.macro_f1, 3)}</td><td class="num">${fmt(x.per_class.SUPPORT.f1, 3)}</td><td class="num">${fmt(x.per_class.CONTRADICT.f1, 3)}</td><td class="num">${fmt(x.per_class.NEUTRAL.f1, 3)}</td><td class="tiny">${Object.entries(x.verdict_counts).map(([k, v]) => `${k}: ${v}`).join(" · ")}</td></tr>`; }).join("")}</tbody></table>
      ${d.modes.independence_log.nei_band_verdicts ? `<p class="small" style="margin:8px 0 0">Verdicts on NOT ENOUGH INFO claims: ${Object.entries(d.modes.independence_log.nei_band_verdicts).map(([k, v]) => `${k}: ${v}`).join(" · ")}</p>` : ""}
      ${d.modes.independence_log.middle_band_verdicts ? `<p class="small" style="margin:8px 0 0">LIAR middle band (half-true / barely-true) verdicts: ${Object.entries(d.modes.independence_log.middle_band_verdicts).map(([k, v]) => `${k}: ${v}`).join(" · ")} · two-sided band accuracy ${fmt(d.modes.independence_log.two_sided_band * 100, 1)}%</p>` : ""}
      ${d.modes.independence_log.merged_not_supported ? `<p class="small" style="margin:8px 0 0">Merged not-supported accuracy: ${fmt(d.modes.independence_log.merged_not_supported.accuracy * 100, 1)}%</p>` : ""}
    </div>`).join("") + `<div style="margin-top:16px">${howto}</div>`;
}

/* ------------------------------------------------------------------ history */
const when = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

function historyBanner(h) {
  return `<div class="callout hist-banner"><span>Saved result from <b>${esc(when(h.at))}</b>, shown exactly as it was then (corpus of ${h.corpus_size.toLocaleString()} ${h.corpus === "fever" ? "pages" : "articles"}).</span>
    <button class="btn sm primary" id="rerun-btn">Re-run now</button></div>`;
}

async function loadHistory() {
  const q = $("#h-search").value.trim();
  const h = await api("/api/history?q=" + encodeURIComponent(q));
  $("#h-title").textContent = `${h.total.toLocaleString()} saved verification${h.total === 1 ? "" : "s"}`;
  $("#h-clear").disabled = !h.total;
  const counts = Object.entries(h.by_verdict).map(([k, v]) => `<span class="vchip ${vclass(k)}">${esc(k)} · ${v}</span>`).join(" ");
  if (!h.items.length) {
    $("#history-out").innerHTML = `<div class="card empty">${q ? "No saved verifications match this search." : "No verifications yet. Verify a claim and it will appear here."}</div>`;
    return;
  }
  $("#history-out").innerHTML = `
    ${counts ? `<div class="chips" style="margin:0 0 12px">${counts}</div>` : ""}
    <div class="card tbl-wrap"><table>
      <thead><tr><th>When</th><th>Corpus</th><th>Claim</th><th>Verdict</th><th class="num">S</th><th class="num">Clusters</th><th class="num">ms</th><th></th></tr></thead>
      <tbody>${h.items.map((i) => `<tr class="hist-row" data-id="${i.id}">
        <td class="tiny" style="white-space:nowrap">${esc(when(i.at))}</td>
        <td class="tiny">${i.corpus === "fever" ? "FEVER" : "Live"}${i.web_added !== null ? ` <span class="web-tag" title="web search added ${i.web_added} new items">WEB</span>` : ""}${i.fever_correct !== null ? ` <span class="${i.fever_correct ? "yes" : "no"}">${i.fever_correct ? "✓" : "✗"}</span>` : ""}</td>
        <td class="hist-claim">${esc(i.claim)}</td>
        <td><span class="vchip ${vclass(i.verdict)}">${esc(i.verdict)}</span></td>
        <td class="num">${fmt(i.S)}</td><td class="num">${i.clusters}</td><td class="num">${fmt(i.latency_ms, 0)}</td>
        <td class="hist-actions"><button class="btn sm h-open">Open</button> <button class="btn sm h-rerun">Re-run</button> <button class="btn sm h-del" title="Delete this entry">✕</button></td>
      </tr>`).join("")}</tbody>
    </table>
    <p class="tiny muted" style="margin:8px 0 0">Stored locally in <code>data/history.sqlite</code>. Syndication stress-test runs are simulations and aren't saved. ✓/✗ marks whether a FEVER demo answer matched the gold label.</p></div>`;
  $$("#history-out .hist-row").forEach((tr) => {
    const id = +tr.dataset.id;
    tr.onclick = (e) => { if (!e.target.closest("button") || e.target.closest(".h-open")) openHistory(id); };
    $(".h-rerun", tr).onclick = async (e) => { e.stopPropagation(); rerun((await api("/api/history/" + id))); };
    $(".h-del", tr).onclick = async (e) => { e.stopPropagation(); await api("/api/history/" + id, { method: "DELETE" }); loadHistory(); };
  });
}

async function openHistory(id) {
  const h = await api("/api/history/" + id);
  await setCorpus(h.corpus);
  $("#claim").value = h.claim;
  showTab("verify");
  renderVerify({ ...h.result, saved: h });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function rerun(h) {
  await setCorpus(h.corpus);
  $("#claim").value = h.claim;
  showTab("verify");
  runVerify();
}

let hTimer = null;
$("#h-search").addEventListener("input", () => { clearTimeout(hTimer); hTimer = setTimeout(loadHistory, 200); });
$("#h-clear").onclick = async () => {
  if (!confirm("Delete all saved verifications? This can't be undone.")) return;
  await api("/api/history", { method: "DELETE" });
  loadHistory();
};

/* ------------------------------------------------------------------ boot */
(async function boot() {
  await pollStatus();
  setInterval(pollStatus, 6000);
  buildParams(STATUS ? STATUS.defaults : { M: 10000, K: 10, tau: 0.9, lam: 0.5, theta: 0.6, delta: 0.25, min_rel: 0.68, threads: 0, schedule: "guided" });
  loadExamples();
  const h = location.hash.slice(1);
  if (h && $("#tab-" + h)) showTab(h);
})();
