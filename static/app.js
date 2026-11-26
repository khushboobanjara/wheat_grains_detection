/* Wheat Grain Analysis Dashboard - front end (vanilla JS, no external libraries) */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const n0 = (v) => (v == null || isNaN(v) ? "-" : Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 }));
const n1 = (v) => (v == null || isNaN(v) ? "-" : Number(v).toLocaleString(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 }));
const n2 = (v) => (v == null || isNaN(v) ? "-" : Number(v).toFixed(2));

/* ------------------------------------------------------------------ state */
const S = {
  cfg: null,                 // /api/status
  files: [], cur: 0,         // uploaded images
  originals: {},             // sid -> data URL
  tab: "overview",
  set: null,                 // detection settings
  det: null,                 // last detection result
  view: "side", metric: "length", onlyOut: false, sort: { key: "id", dir: 1 },
  cls: null, clsRunning: false, clsSub: 0, clsShown: null, clsOnlyFlag: false,
  ai: null, aiKey: "", aiRunning: false,
  batch: null, batchKey: "",
};
const TITLES = { overview: "Overview", detection: "Detection", quality: "Quality classification", ai: "AI report", export: "Export", batch: "Batch comparison" };
const ctrls = {};
let detectToken = 0, detectTimer = null, clsTimer = null, busyCount = 0;

/* ---------------------------------------------------------------- helpers */
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  let j;
  try { j = await r.json(); } catch { throw new Error("The server returned an unexpected response."); }
  if (!r.ok || j.error) throw new Error(j.error || "Request failed.");
  return j;
}
function busy(on) {
  busyCount = Math.max(0, busyCount + (on ? 1 : -1));
  let bar = $("#busyBar");
  if (busyCount && !bar) { bar = document.createElement("div"); bar.id = "busyBar"; bar.className = "busy-bar"; document.body.appendChild(bar); }
  if (!busyCount && bar) bar.remove();
}
function toast(msg, ok = false) {
  const t = document.createElement("div");
  t.className = "toast" + (ok ? " ok" : "");
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 6000);
}
const curFile = () => S.files[S.cur];
function payload() {
  const s = S.set;
  return { hue: s.hue, sat: s.sat, val: s.val, auto: s.auto, blur: s.blur, open: s.open, close: s.close, area: s.area,
           aspect: s.aspect, solidity: s.solidity, rel: s.rel, split: s.split, level: s.level, ids: s.ids, tilted: s.tilted };
}
const detKey = () => { const p = payload(); delete p.ids; delete p.tilted; return JSON.stringify(p); };
function dlUrl(kind) {
  const p = encodeURIComponent(JSON.stringify({ params: payload(), threshold: S.set.thr }));
  return `/api/download/${kind}?sid=${curFile().sid}&p=${p}`;
}
function kpi(label, value, sub = "", tone = "") {
  return `<div class="card kpi ${tone}"><div class="kpi-label">${esc(label)}</div><div class="kpi-value">${esc(value)}</div><div class="kpi-sub">${esc(sub)}</div></div>`;
}
function callout(kind, text) { return `<div class="callout ${kind}">${esc(text)}</div>`; }
function head(title, sub = "", right = "") {
  return `<div class="card-head"><div><h3>${esc(title)}</h3>${sub ? `<p>${esc(sub)}</p>` : ""}</div>${right}</div>`;
}
function imgBox(src, cap = "", zoom = true) {
  return `<figure class="imgbox ${zoom ? "zoomable" : ""}" style="margin:0"><img src="${src}" alt="${esc(cap)}">${cap ? `<div class="cap">${cap}</div>` : ""}</figure>`;
}
const dlIcon = `<svg class="ico" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5M12 15V3"/></svg>`;

/* --------------------------------------------------------------- markdown */
function md(src) {
  const inline = (t) => esc(t).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>");
  const lines = String(src || "").split(/\r?\n/);
  let out = "", list = null, i = 0;
  const close = () => { if (list) { out += `</${list}>`; list = null; } };
  while (i < lines.length) {
    const L = lines[i], t = L.trim();
    if (!t) { close(); i++; continue; }
    if (/^\|.*\|$/.test(t)) {                                  // table
      close();
      const rows = [];
      while (i < lines.length && /^\|.*\|$/.test(lines[i].trim())) { rows.push(lines[i].trim()); i++; }
      const cells = (r) => r.slice(1, -1).split("|").map((c) => c.trim());
      const body = rows.filter((r) => !/^\|[\s:|-]+\|$/.test(r));
      out += `<div class="table-wrap"><table><thead><tr>${cells(body[0]).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>`
        + body.slice(1).map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>";
      continue;
    }
    let m;
    if ((m = t.match(/^(#{1,4})\s+(.*)$/))) { close(); out += `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`; }
    else if (/^(-{3,}|\*{3,})$/.test(t)) { close(); out += "<hr>"; }
    else if ((m = t.match(/^[-*•]\s+(.*)$/))) { if (list !== "ul") { close(); out += "<ul>"; list = "ul"; } out += `<li>${inline(m[1])}</li>`; }
    else if ((m = t.match(/^\d+[.)]\s+(.*)$/))) { if (list !== "ol") { close(); out += "<ol>"; list = "ol"; } out += `<li>${inline(m[1])}</li>`; }
    else { close(); out += `<p>${inline(t)}</p>`; }
    i++;
  }
  close();
  return `<div class="md">${out}</div>`;
}

/* ----------------------------------------------------------------- charts */
function niceMax(v) { return Math.max(4, Math.ceil((v * 1.05) / 4) * 4); }   // 4 even gridline steps, integer ticks
function histSvg(values, label) {
  const W = 560, H = 300, L = 46, R = 12, T = 12, B = 48;
  const lo = Math.min(...values), hi = Math.max(...values), nb = Math.min(20, Math.max(5, Math.round(Math.sqrt(values.length) * 1.6)));
  const span = hi - lo || 1, bins = new Array(nb).fill(0);
  values.forEach((v) => { bins[Math.min(nb - 1, Math.floor(((v - lo) / span) * nb))]++; });
  const ymax = niceMax(Math.max(...bins)), iw = W - L - R, ih = H - T - B, bw = iw / nb;
  const med = [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
  const mx = L + ((med - lo) / span) * iw;
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const y = T + ih - (k / 4) * ih;
    g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="var(--border)"/><text x="${L - 8}" y="${y + 4}" text-anchor="end">${n0((k / 4) * ymax)}</text>`;
  }
  bins.forEach((c, k) => {
    const h = (c / ymax) * ih, x = L + k * bw + 1.5;
    const a = lo + (k / nb) * span, b = lo + ((k + 1) / nb) * span;
    g += `<rect x="${x}" y="${T + ih - h}" width="${bw - 3}" height="${h}" rx="3" fill="var(--primary)" opacity=".9"><title>${n0(a)} - ${n0(b)}: ${c} grains</title></rect>`;
  });
  for (let k = 0; k <= 5; k++) g += `<text x="${L + (k / 5) * iw}" y="${H - 28}" text-anchor="middle">${n0(lo + (k / 5) * span)}</text>`;
  g += `<line x1="${mx}" x2="${mx}" y1="${T}" y2="${T + ih}" stroke="var(--text)" stroke-width="2" stroke-dasharray="5 4"/>`;
  g += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">${esc(label)}</text>`;
  g += `<text x="12" y="${T + ih / 2}" transform="rotate(-90 12 ${T + ih / 2})" text-anchor="middle">Grains</text>`;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of ${esc(label)}">${g}</svg>`;
}
function barSvg(items, opt = {}) {
  const W = opt.w || 520, H = opt.h || 280, L = 44, R = 10, T = 14, B = opt.b || 70;
  const ymax = niceMax(Math.max(1, ...items.map((i) => i.value))), iw = W - L - R, ih = H - T - B, bw = iw / items.length;
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const y = T + ih - (k / 4) * ih;
    g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="var(--border)"/><text x="${L - 8}" y="${y + 4}" text-anchor="end">${n0((k / 4) * ymax)}</text>`;
  }
  items.forEach((it, k) => {
    const h = (it.value / ymax) * ih, x = L + k * bw + bw * 0.18, w = bw * 0.64, cx = x + w / 2;
    g += `<rect x="${x}" y="${T + ih - h}" width="${w}" height="${h}" rx="4" fill="${it.color || "var(--primary)"}"><title>${esc(it.label)}: ${it.value}</title></rect>`;
    g += `<text x="${cx}" y="${T + ih - h - 5}" text-anchor="middle" style="fill:var(--text);font-weight:600">${it.value}</text>`;
    const lab = it.label.length > 14 ? it.label.slice(0, 13) + "…" : it.label;
    g += `<text x="${cx}" y="${T + ih + 16}" text-anchor="end" transform="rotate(-28 ${cx} ${T + ih + 16})">${esc(lab)}</text>`;
  });
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">${g}</svg>`;
}
function donutSvg(items) {
  const total = items.reduce((a, i) => a + i.value, 0), R = 70, C = 2 * Math.PI * R;
  let off = 0, g = "";
  items.filter((i) => i.value > 0).forEach((it) => {
    const len = (it.value / total) * C;
    g += `<circle r="${R}" cx="110" cy="110" fill="none" stroke="${it.color}" stroke-width="30" stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-off}" transform="rotate(-90 110 110)"><title>${esc(it.label)}: ${it.value} (${((it.value / total) * 100).toFixed(1)}%)</title></circle>`;
    off += len;
  });
  g += `<text x="110" y="108" text-anchor="middle" style="fill:var(--text);font-size:26px;font-weight:750">${total}</text><text x="110" y="128" text-anchor="middle">classified</text>`;
  return `<svg class="chart" viewBox="0 0 220 220" style="max-width:280px;margin:0 auto" role="img">${g}</svg>`;
}
const legend = (m) => `<div class="legend">${Object.entries(m).map(([k, c]) => `<span><i style="background:${c}"></i>${esc(k)}</span>`).join("")}</div>`;

/* -------------------------------------------------------- settings panel */
const GROUPS = [
  { title: "Colour range (HSV)", open: true, items: [
    { t: "preset" },
    { t: "dual", k: "hue", min: 0, max: 179, step: 1, label: "Hue" },
    { t: "dual", k: "sat", min: 0, max: 255, step: 1, label: "Saturation", hint: "Raise the minimum to ignore white/grey paper, black text and shadows." },
    { t: "dual", k: "val", min: 0, max: 255, step: 1, label: "Value (brightness)" },
    { t: "check", k: "auto", label: "Auto-tune min saturation (Otsu)" } ] },
  { title: "Mask clean-up", items: [
    { t: "range", k: "blur", min: 1, max: 15, step: 2, label: "Blur kernel" },
    { t: "range", k: "open", min: 0, max: 5, step: 1, label: "Remove specks (opening)" },
    { t: "range", k: "close", min: 0, max: 5, step: 1, label: "Fill gaps (closing)" } ] },
  { title: "Shape filters", open: true, items: [
    { t: "dual", k: "area", min: 20, max: 100000, step: 10, label: "Grain area (px)", hint: "Measured on the processed image (long side ≤ 1600 px)." },
    { t: "dual", k: "aspect", min: 1, max: 8, step: 0.05, label: "Aspect ratio (length / width)" },
    { t: "range", k: "solidity", min: 0.3, max: 1, step: 0.01, label: "Min solidity", hint: "Rejects ragged shapes such as letters and digits." },
    { t: "range", k: "rel", min: 0, max: 80, step: 1, label: "Min size vs median grain (%)", hint: "Drops fragments much smaller than a typical grain. 0 disables." } ] },
  { title: "Touching grains", items: [
    { t: "check", k: "split", label: "Split touching grains (watershed)" },
    { t: "range", k: "level", min: 0.3, max: 0.9, step: 0.05, label: "Split sensitivity", hint: "Lower = splits more aggressively, higher = splits less." } ] },
  { title: "Display", items: [
    { t: "check", k: "ids", label: "Number the grains" },
    { t: "check", k: "tilted", label: "Tilted boxes (rotated rectangles)" } ] },
  { title: "Classification (optional)", items: [
    { t: "range", k: "thr", min: 0, max: 1, step: 0.05, label: "Manual-inspection threshold", pct: true, hint: "Grains classified below this confidence are flagged. Updates instantly; no new Groq calls." },
    { t: "range", k: "cbatch", min: 1, max: 5, step: 1, label: "Grains per Groq request", hint: "1 = most faithful but most API calls. Higher = fewer calls." } ] },
];
const DEFAULT_EXTRA = { auto: false, ids: false, tilted: false, thr: 0.7, cbatch: 1, preset: "" };
const fmtV = (def, v) => (def.pct ? Math.round(v * 100) + "%" : def.step < 1 ? Number(v).toFixed(2) : n0(v));

function resetSettings() {
  const d = JSON.parse(JSON.stringify(S.cfg.defaults));
  const match = Object.entries(S.cfg.presets).find(([, p]) => JSON.stringify(p.hue) === JSON.stringify(d.hue) && JSON.stringify(p.sat) === JSON.stringify(d.sat) && JSON.stringify(p.val) === JSON.stringify(d.val));
  S.set = Object.assign(d, DEFAULT_EXTRA, { preset: match ? match[0] : Object.keys(S.cfg.presets)[0] });
  syncControls();
}
function syncControls() { Object.values(ctrls).forEach((c) => c.update()); }
function changed(key) {
  if (key === "thr") { clearTimeout(clsTimer); clsTimer = setTimeout(refreshCls, 250); return; }
  if (key === "cbatch") { if (S.tab === "quality") render(); return; }
  clearTimeout(detectTimer);
  detectTimer = setTimeout(runDetect, 300);
}
function buildSettings() {
  const body = $("#settingsBody");
  body.innerHTML = "";
  GROUPS.forEach((grp) => {
    const d = document.createElement("details");
    d.className = "group"; d.open = !!grp.open;
    d.innerHTML = `<summary>${esc(grp.title)}</summary><div class="group-body"></div>`;
    const gb = $(".group-body", d);
    grp.items.forEach((def) => gb.appendChild(makeControl(def)));
    body.appendChild(d);
  });
}
function makeControl(def) {
  const w = document.createElement("div");
  w.className = "field";
  const hint = def.hint ? `<div class="hint">${esc(def.hint)}</div>` : "";
  if (def.t === "preset") {
    w.innerHTML = `<label for="presetSel"><span>Colour preset</span></label><select id="presetSel" style="width:100%">${Object.keys(S.cfg.presets).map((p) => `<option>${esc(p)}</option>`).join("")}</select><div class="hint">Starting point for the HSV colour range. Changing it resets the three colour sliders.</div>`;
    const sel = $("select", w);
    sel.onchange = () => { const p = S.cfg.presets[sel.value]; S.set.preset = sel.value; S.set.hue = [...p.hue]; S.set.sat = [...p.sat]; S.set.val = [...p.val]; syncControls(); changed("preset"); };
    ctrls.preset = { update: () => { sel.value = S.set.preset; } };
  } else if (def.t === "check") {
    w.innerHTML = `<label class="check"><input type="checkbox"><span>${esc(def.label)}</span></label>`;
    const cb = $("input", w);
    cb.onchange = () => { S.set[def.k] = cb.checked; changed(def.k); };
    ctrls[def.k] = { update: () => { cb.checked = !!S.set[def.k]; } };
  } else if (def.t === "range") {
    w.innerHTML = `<label><span>${esc(def.label)}</span><span class="val"></span></label><input type="range" min="${def.min}" max="${def.max}" step="${def.step}">${hint}`;
    const r = $("input", w), v = $(".val", w);
    r.oninput = () => { S.set[def.k] = Number(r.value); v.textContent = fmtV(def, r.value); changed(def.k); };
    ctrls[def.k] = { update: () => { r.value = S.set[def.k]; v.textContent = fmtV(def, S.set[def.k]); } };
  } else if (def.t === "dual") {
    w.innerHTML = `<div class="lbl"><span>${esc(def.label)}</span><span class="val"></span></div><div class="dual"><div class="track"></div><div class="fill"></div><input type="range" aria-label="${esc(def.label)} minimum"><input type="range" aria-label="${esc(def.label)} maximum"></div>${hint}`;
    const [a, b] = $$("input", w), fill = $(".fill", w), v = $(".val", w);
    [a, b].forEach((x) => { x.min = def.min; x.max = def.max; x.step = def.step; });
    const draw = () => {
      const [lo, hi] = S.set[def.k], span = def.max - def.min;
      a.value = lo; b.value = hi;
      fill.style.left = `calc(${((lo - def.min) / span) * 100}% )`;
      fill.style.width = `${((hi - lo) / span) * 100}%`;
      v.textContent = `${fmtV(def, lo)} – ${fmtV(def, hi)}`;
    };
    a.oninput = () => { let lo = Number(a.value); const hi = S.set[def.k][1]; if (lo > hi) lo = hi; S.set[def.k] = [lo, hi]; draw(); changed(def.k); };
    b.oninput = () => { let hi = Number(b.value); const lo = S.set[def.k][0]; if (hi < lo) hi = lo; S.set[def.k] = [lo, hi]; draw(); changed(def.k); };
    ctrls[def.k] = { update: draw };
  }
  return w;
}

/* ------------------------------------------------------------- upload flow */
async function uploadFiles(list) {
  const files = [...list];
  if (!files.length) return;
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  busy(true);
  try {
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || "Upload failed.");
    j.files.filter((f) => f.error).forEach((f) => toast(`${f.name}: ${f.error}`));
    const ok = j.files.filter((f) => f.sid);
    if (!ok.length) return;
    S.files = ok; S.cur = 0; S.originals = {};
    S.batch = null; S.batchKey = "";
    afterFileChange();
    toast(`${ok.length} image${ok.length > 1 ? "s" : ""} ready.`, true);
    await selectFile(0);
  } catch (e) { toast(e.message); }
  finally { busy(false); }
}
function afterFileChange() {
  const has = S.files.length > 0;
  if (!has) $("#settings").classList.remove("open");
  $("#settingsToggle").classList.toggle("hidden", !has);
  $("#navBatch").classList.toggle("hidden", S.files.length < 2);
  $("#batchBadge").textContent = S.files.length;
  const sel = $("#fileSelect");
  sel.classList.toggle("hidden", S.files.length < 2);
  sel.innerHTML = S.files.map((f, i) => `<option value="${i}">${i + 1}. ${esc(f.name)}</option>`).join("");
  sel.value = S.cur;
  if (S.tab === "batch" && S.files.length < 2) S.tab = "overview";
}
async function selectFile(i) {
  S.cur = i; S.det = null; S.cls = null; S.ai = null; S.clsSub = 0; S.clsShown = null;
  $("#fileSelect").value = i;
  render();
  const f = curFile();
  if (!S.originals[f.sid]) {
    try { S.originals[f.sid] = (await api(`/api/original/${f.sid}`)).image; } catch (e) { toast(e.message); }
  }
  await runDetect();
}

/* --------------------------------------------------------------- detection */
async function runDetect() {
  const f = curFile();
  if (!f) return;
  const token = ++detectToken;
  busy(true);
  try {
    const d = await api("/api/detect", { sid: f.sid, params: payload() });
    if (token !== detectToken) return;
    S.det = d;
    if (["overview", "quality", "export"].includes(S.tab)) await refreshCls(true);
    if (S.tab === "batch") S.batchKey = "";
  } catch (e) { if (token === detectToken) toast(e.message); }
  finally { busy(false); }
  if (token === detectToken) render();
}
async function refreshCls(silent) {
  const f = curFile();
  if (!f || !S.cfg.ai_available) return;
  try {
    S.cls = await api("/api/classification", { sid: f.sid, params: payload(), threshold: S.set.thr });
    if (!S.clsShown && S.cls.state === "ok") S.clsShown = Object.keys(S.cfg.categories).filter((c) => c !== "Unclassified");
  } catch (e) { if (!silent) toast(e.message); }
  if (!silent) render();
}

/* ------------------------------------------------------------------ views */
function landing() {
  return `
  <div class="hero" id="heroDrop">
    <div class="big">🌾</div>
    <h2>Analyse your wheat photos</h2>
    <p>Drop one or more photos here (JPG, PNG, TIFF, WebP…) to detect, count and measure every grain. Everything runs locally on your computer.</p>
    <button class="btn primary" id="heroBtn">${`<svg class="ico" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5M12 3v12"/></svg>`} Choose photos</button>
  </div>
  <div class="grid-3">
    ${[["1", "Upload", "Drop one or more photos. Several images can be compared in the Batch view."],
       ["2", "Detect & tune", "Grains are found locally with OpenCV. Adjust the colour range and filters until the count looks right."],
       ["3", "Analyse & export", "Review size statistics, optionally grade each grain with Groq, then download images, CSVs and a PDF."]]
      .map(([n, t, d]) => `<div class="card step"><div class="n">${n}</div><h4>${t}</h4><p>${d}</p></div>`).join("")}
  </div>
  <div class="card card-pad">
    ${head("Tips for the best results")}
    <div class="grid-2">
      <ul class="tips"><li>Spread grains out on a <b>plain, contrasting</b> background</li><li>Use <b>even lighting</b>; avoid strong shadows and glare</li><li>Shoot from directly above, with the grains in focus</li></ul>
      <ul class="tips"><li>Text, numbers and labels are ignored automatically (low colour saturation)</li><li>Touching grains are split with a watershed step</li><li>Heavily stacked grains still need manual slider tuning</li></ul>
    </div>
  </div>`;
}
function loadingView() { return `<div class="card card-pad empty"><div class="spin" style="margin:0 auto 12px"></div>Detecting grains…</div>`; }


function conditionCard() {
  const cfg = S.cfg, n = S.det.stats.count, c = S.cls, colors = cfg.categories, defs = cfg.category_defs || {};
  const batch = S.set.cbatch, calls = Math.ceil(n / batch) || 0;
  const done = c && c.state === "ok";
  let body = "";
  if (!cfg.ai_available) body += callout("info", "Grain classification is unavailable in this installation.");
  else if (!cfg.groq) body += callout("info", "Add your Groq API key to the .env file and reload the page to grade each grain (Sound, Shrivelled, Immature, Broken, Damaged/Discoloured).");
  if (c && c.state === "stale") body += callout("warn", "Detection settings changed since the last analysis, so the grain IDs no longer match. Run the analysis again.");
  if (c && c.state === "error") body += callout("bad", c.error);
  if (done) {
    const s = c.summary;
    if (c.fatal_error) body += callout("bad", c.fatal_error) + (c.hint ? `<div class="small muted">${esc(c.hint)}</div>` : "");
    body += `<div class="cond-grid">${s.categories.map((x) => `<div class="cond" style="--c:${colors[x.category]}" title="${esc(defs[x.category] || "")}">
      <div class="nm">${esc(x.category)}</div><div class="ct">${x.count}</div><div class="pc">${x.percent.toFixed(1)}% of classified</div></div>`).join("")}</div>`;
    const items = s.categories.map((x) => ({ label: x.category, value: x.count, color: colors[x.category] }));
    if (s.classified) body += `<div class="grid-2"><div>${barSvg(items)}</div><div>${donutSvg(items)}</div></div>`;
    body += `<div class="small muted" style="margin:10px 0">${s.classified} of ${s.total_detected} grains classified${s.unclassified ? `, ${s.unclassified} could not be classified` : ""}; ${s.flagged_total} flagged for manual inspection. Per-grain results and the colour-coded image are in the Quality tab.</div>`;
  }
  body += `<div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:6px">
    <button class="btn primary" id="clsBtn" ${!cfg.ai_available || !cfg.groq || !n || S.clsRunning ? "disabled" : ""}>${S.clsRunning ? "Analysing…" : done ? "Re-run analysis" : "Analyse grain condition"}</button>
    <span class="small muted">${n ? `${n} grains will be sent to Groq in ${calls} request${calls !== 1 ? "s" : ""}. Nothing is sent until you click.` : "No grains detected yet."}</span></div>
    <div class="progress ${S.clsRunning ? "" : "hidden"}" style="margin-top:12px"><div id="clsBar"></div></div><div class="small muted ${S.clsRunning ? "" : "hidden"}" id="clsTxt" style="margin-top:6px"></div>`;
  return `<div class="card card-pad">${head("Grain condition", "Each grain is graded as Sound, Shrivelled, Immature, Broken or Damaged/Discoloured. OpenCV does the counting; Groq judges each grain's close-up.")}${body}</div>`;
}

function viewOverview() {
  const d = S.det, st = d.stats, n = st.count;
  const orig = S.originals[curFile().sid];
  const tone = n < 2 ? "" : st.variation < 15 ? "good" : st.variation < 30 ? "neutral" : "warn";
  let html = `<div class="kpis">
    ${kpi("Grains detected", n0(n), `${d.rejected} blobs filtered out`, n ? "good" : "bad")}
    ${kpi("Avg grain size", n ? `${n0(st.avg_length)} × ${n0(st.avg_width)} px` : "-", n ? `length × width · median ${n0(st.med_length)} × ${n0(st.med_width)}` : "", "")}
    ${kpi("Size variation", n > 1 ? n0(st.variation) + "%" : "-", "area CV, lower = more uniform", tone)}
    ${kpi("Processing", d.elapsed.toFixed(2) + " s", `${d.width} × ${d.height} px image`, "neutral")}
  </div>`;
  if (!n) html += callout("warn", "No grains found. Try another colour preset, widen the Hue/Saturation range, or relax the shape filters (see the Detection tab for the mask).");

  const seg = `<div class="seg" id="viewSeg">${[["side", "Side by side"], ["processed", "Processed"], ["original", "Original"], ["mask", "Mask"]]
    .map(([k, l]) => `<button data-v="${k}" class="${S.view === k ? "on" : ""}">${l}</button>`).join("")}</div>`;
  let imgs;
  if (S.view === "side") imgs = `<div class="grid-2">${imgBox(orig, "Original")}${imgBox(d.processed, `Processed · ${n} grains`)}</div>`;
  else if (S.view === "processed") imgs = imgBox(d.processed, `Processed · ${n} grains`);
  else if (S.view === "original") imgs = imgBox(orig, "Original");
  else imgs = imgBox(d.mask, "White = pixels inside the HSV range after clean-up.");
  html += conditionCard();
  html += `<div class="card card-pad">${head("Image preview", "Click an image to enlarge it.", seg)}${imgs}</div>`;

  if (n) {
    const metrics = [["length", "Length (px)"], ["width", "Width (px)"], ["area", "Area (px²)"]];
    const tbl = [["Smallest length", n0(st.min_length) + " px"], ["Median length", n0(st.med_length) + " px"], ["Largest length", n0(st.max_length) + " px"],
      ["Avg aspect ratio", n2(st.avg_aspect)], ["Avg solidity", n2(st.avg_solidity)], ["Small grains (<65% median area)", st.small], ["Large grains (>150%)", st.large]];
    html += `<div class="grid-21">
      <div class="card card-pad">${head("Grain size distribution", "Dashed line = median.",
        `<select id="metricSel" aria-label="Measure">${metrics.map(([k, l]) => `<option value="${k}" ${S.metric === k ? "selected" : ""}>${l}</option>`).join("")}</select>`)}
        <div id="histBox"></div></div>
      <div class="card card-pad">${head("Summary statistics")}
        <div class="table-wrap table-plain"><table><tbody>${tbl.map(([a, b]) => `<tr><td>${a}</td><td class="num"><b>${b}</b></td></tr>`).join("")}</tbody></table></div></div>
    </div>`;
  }
  return html;
}
function bindOverview() {
  const cb = $("#clsBtn");
  if (cb) cb.onclick = runClassify;
  $$("#viewSeg button").forEach((b) => (b.onclick = () => { S.view = b.dataset.v; render(); }));
  const sel = $("#metricSel");
  const draw = () => {
    const m = [["length", "Length (px)"], ["width", "Width (px)"], ["area", "Area (px²)"]].find((x) => x[0] === S.metric);
    $("#histBox").innerHTML = histSvg(S.det.grains.map((g) => g[S.metric]), m[1]);
  };
  if (sel) { sel.onchange = () => { S.metric = sel.value; draw(); }; draw(); }
}

const COLS = [["id", "ID"], ["x", "X"], ["y", "Y"], ["box_w", "Box W"], ["box_h", "Box H"], ["length", "Length (px)"], ["width", "Width (px)"],
  ["aspect", "Aspect"], ["solidity", "Solidity"], ["area", "Area (px²)"], ["size_flag", "Size"]];
function viewDetection() {
  const d = S.det, p = S.set;
  const act = [["Hue", p.hue.join(" – ")], ["Saturation", p.sat.join(" – ") + (p.auto ? " (auto min)" : "")], ["Value", p.val.join(" – ")],
    ["Area (px)", p.area.join(" – ")], ["Aspect ratio", p.aspect.join(" – ")], ["Min solidity", p.solidity], ["Min size vs median", p.rel + "%"],
    ["Split touching", p.split ? "yes" : "no"], ["Split sensitivity", p.level]];
  return `
  <div class="card card-pad">${head("Segmentation mask", "White pixels passed the colour range. If grains are missing here, adjust the HSV sliders.")}
    <div class="grid-21">${imgBox(d.mask, "")}
      <div><div class="small muted" style="font-weight:700;margin-bottom:6px">ACTIVE SETTINGS</div>
        <div class="table-wrap table-plain"><table><tbody>${act.map(([a, b]) => `<tr><td>${a}</td><td class="num"><b>${esc(b)}</b></td></tr>`).join("")}</tbody></table></div></div>
    </div></div>
  <div class="card card-pad">${head("Per-grain measurements", "Pixels of the original image. Click a column header to sort.",
      `<label class="check"><input type="checkbox" id="onlyOut" ${S.onlyOut ? "checked" : ""}>Only size outliers</label>`)}
    <div id="grainTable"></div><div class="small muted" id="grainCount" style="margin-top:8px"></div></div>`;
}
function drawGrainTable() {
  const g = S.det.grains;
  if (!g.length) { $("#grainTable").innerHTML = `<div class="empty">No grains to list.</div>`; return; }
  let rows = S.onlyOut ? g.filter((r) => r.size_flag !== "Normal") : [...g];
  const { key, dir } = S.sort;
  rows.sort((a, b) => (a[key] > b[key] ? 1 : a[key] < b[key] ? -1 : 0) * dir);
  const fmt = { length: n1, width: n1, aspect: n2, solidity: n2, area: n0 };
  $("#grainTable").innerHTML = `<div class="table-wrap"><table><thead><tr>${COLS.map(([k, l]) =>
    `<th class="sortable ${k === "size_flag" ? "" : "num"}" data-k="${k}">${l}${S.sort.key === k ? (dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}</tr></thead><tbody>
    ${rows.map((r) => `<tr>${COLS.map(([k]) => k === "size_flag" ? `<td><span class="tag ${r[k]}">${r[k]}</span></td>` : `<td class="num">${(fmt[k] || ((v) => v))(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  $("#grainCount").textContent = `Showing ${rows.length} of ${g.length} grains.`;
  $$("#grainTable th.sortable").forEach((th) => (th.onclick = () => {
    S.sort = { key: th.dataset.k, dir: S.sort.key === th.dataset.k ? -S.sort.dir : 1 };
    drawGrainTable();
  }));
}
function bindDetection() {
  $("#onlyOut").onchange = (e) => { S.onlyOut = e.target.checked; drawGrainTable(); };
  drawGrainTable();
}

/* ---- quality (Groq classification) ---- */
function viewQuality() {
  const n = S.det.stats.count, cfg = S.cfg, c = S.cls;
  const batch = S.set.cbatch, calls = Math.ceil(n / batch) || 0;
  let html = `<div class="card card-pad">${head("Grain quality classification",
    "OpenCV detects, counts, boxes and measures every grain. Groq then looks at each grain's own crop and assigns one category: Sound, Shrivelled, Immature, Broken or Damaged/Discoloured. Uncertain or failed grains are flagged for manual inspection, never guessed.")}`;
  if (!cfg.ai_available) html += callout("info", "Grain classification is unavailable in this installation.");
  else {
    if (!cfg.groq) html += callout("info", "Optional feature: add your Groq API key to the .env file (see .env.example) and reload the page to enable it. Everything else works without it.");
    html += `<div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:12px">
      <button class="btn primary" id="clsBtn" ${!cfg.groq || !n || S.clsRunning ? "disabled" : ""}>Classify grains with Groq</button>
      <span class="small muted">${n ? `${n} grains will be sent in ${calls} request${calls !== 1 ? "s" : ""} (${batch} per request). Clicking the button uploads the grain crops to Groq. Free-tier rate limits may slow this down.` : "No grains detected yet, so there is nothing to classify."}</span></div>
      <div class="progress ${S.clsRunning ? "" : "hidden"}" style="margin-top:12px"><div id="clsBar"></div></div><div class="small muted ${S.clsRunning ? "" : "hidden"}" id="clsTxt" style="margin-top:6px"></div>`;
  }
  html += `</div>`;
  if (!c || c.state === "none") return html;
  if (c.state === "stale") return html + callout("warn", "These classification results belong to a different image or detection settings, so the grain IDs no longer match. Click 'Classify grains with Groq' to classify the current detection.");
  if (c.state === "error") return html + callout("bad", c.error) + `<div class="small muted">Detection results are unaffected.</div>`;

  const s = c.summary;
  if (c.fatal_error) html += callout("bad", c.fatal_error) + (c.hint ? `<div class="small muted">${esc(c.hint)}</div>` : "");
  html += `<div class="kpis">
    ${kpi("Grains detected", s.total_detected, "", "neutral")}${kpi("Classified", s.classified, "", "good")}
    ${kpi("Not classified", s.unclassified, "", s.unclassified ? "bad" : "neutral")}
    ${kpi("Manual inspection", s.flagged_total, `threshold ${Math.round(c.threshold * 100)}%`, s.flagged_total ? "warn" : "good")}</div>`;
  html += callout(s.reconciled ? "ok" : "bad", (s.reconciled ? "Counts reconcile: " : "Counts do not reconcile: ") + c.reconciliation);

  const tabs = ["Summary & charts", "Per-grain results", "Observations & recommendations", "Detailed report"];
  html += `<div class="card card-pad"><div class="subtabs" id="clsTabs">${tabs.map((t, i) => `<button data-i="${i}" class="${S.clsSub === i ? "on" : ""}">${t}</button>`).join("")}</div><div id="clsPane"></div></div>`;
  return html;
}
function drawClsPane() {
  const c = S.cls, pane = $("#clsPane");
  if (!pane || !c || c.state !== "ok") return;
  const s = c.summary, colors = S.cfg.categories;
  if (S.clsSub === 0) {
    const rows = s.categories.map((x) => `<tr><td><span class="tag cat" style="background:${colors[x.category]}">${esc(x.category)}</span></td><td class="num">${x.count}</td><td class="num">${x.percent.toFixed(1)}%</td><td class="num">${x.flagged}</td></tr>`);
    rows.push(`<tr><td><b>Total classified</b></td><td class="num"><b>${s.classified}</b></td><td class="num">${s.classified ? "100.0%" : "0.0%"}</td><td class="num">${s.flagged_total - s.flagged_unclassified}</td></tr>`);
    if (s.unclassified) rows.push(`<tr><td>Unclassified (excluded from %)</td><td class="num">${s.unclassified}</td><td class="num">-</td><td class="num">${s.flagged_unclassified}</td></tr>`);
    const items = s.categories.map((x) => ({ label: x.category, value: x.count, color: colors[x.category] }));
    pane.innerHTML = `<div class="table-wrap"><table><thead><tr><th>Category</th><th class="num">Count</th><th class="num">% of classified</th><th class="num">Flagged for inspection</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>
      <div class="small muted" style="margin:8px 0 16px">Percentages are shares of classified grains. Unclassified grains are never assigned a category.</div>
      ${s.classified ? `<div class="grid-2"><div><b>Grains per category</b>${barSvg(items)}</div><div><b>Share of classified grains</b>${donutSvg(items)}</div></div>` : `<div class="empty">No classified grains to chart.</div>`}
      <div style="margin-top:16px"><b>Classified overview</b> <span class="small muted">(box colour = category, * after the ID = manual inspection)</span></div>
      <div style="margin-top:8px">${imgBox(c.overview, "", true)}</div>${legend(colors)}`;
  } else if (S.clsSub === 1) {
    const cats = Object.keys(colors);
    pane.innerHTML = `<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:12px">
      ${cats.map((k) => `<label class="check"><input type="checkbox" data-cat="${esc(k)}" ${S.clsShown.includes(k) ? "checked" : ""}><i style="width:10px;height:10px;border-radius:3px;background:${colors[k]};display:inline-block"></i>${esc(k)}</label>`).join("")}
      <label class="check" style="margin-left:auto"><input type="checkbox" id="onlyFlag" ${S.clsOnlyFlag ? "checked" : ""}>Only manual inspection</label></div><div id="clsTable"></div>`;
    $$("[data-cat]", pane).forEach((cb) => (cb.onchange = () => {
      S.clsShown = $$("[data-cat]", pane).filter((x) => x.checked).map((x) => x.dataset.cat); drawClsTable();
    }));
    $("#onlyFlag").onchange = (e) => { S.clsOnlyFlag = e.target.checked; drawClsTable(); };
    drawClsTable();
  } else if (S.clsSub === 2) {
    pane.innerHTML = `<h4>Observations</h4><ul class="tips" style="margin:8px 0 18px">${c.observations.map((o) => `<li>${esc(o)}</li>`).join("")}</ul>
      <h4>Recommendations</h4><ul class="tips" style="margin-top:8px">${c.recommendations.map((o) => `<li>${esc(o)}</li>`).join("")}</ul>`;
  } else {
    pane.innerHTML = md(c.report);
  }
}
function drawClsTable() {
  const c = S.cls, colors = S.cfg.categories;
  const rows = c.rows.filter((r) => S.clsShown.includes(r.category) || (r.category === "Unclassified" && S.clsShown.includes("Unclassified")))
    .filter((r) => !S.clsOnlyFlag || r.manual_inspection === "Required");
  $("#clsTable").innerHTML = `<div class="table-wrap"><table><thead><tr><th>ID</th><th>Category</th><th class="num">Confidence</th><th>Manual inspection</th><th>Why / notes</th><th>Model's visual reason</th>
    <th class="num">Length</th><th class="num">Width</th><th class="num">Aspect</th><th class="num">Solidity</th><th class="num">Area</th></tr></thead><tbody>
    ${rows.map((r) => `<tr><td class="num">${r.id}</td><td><span class="tag cat" style="background:${colors[r.category] || "#7f7f7f"}">${esc(r.category)}</span></td>
      <td class="num">${r.confidence == null ? "-" : Math.round(r.confidence * 100) + "%"}</td>
      <td><span class="tag ${r.manual_inspection === "Required" ? "Required" : "Normal"}">${esc(r.manual_inspection)}</span></td>
      <td style="white-space:normal;min-width:220px">${esc(r.review_reasons)}</td><td style="white-space:normal;min-width:220px">${esc(r.model_reason)}</td>
      <td class="num">${n1(r.length_px)}</td><td class="num">${n1(r.width_px)}</td><td class="num">${n2(r.aspect_ratio)}</td><td class="num">${n2(r.solidity)}</td><td class="num">${n0(r.area_px)}</td></tr>`).join("")}
    </tbody></table></div><div class="small muted" style="margin-top:8px">Showing ${rows.length} of ${c.rows.length} grains. Confidence is reported by the model; it is not a calibrated probability. Measurements are in pixels.</div>`;
}
function bindQuality() {
  const b = $("#clsBtn");
  if (b) b.onclick = runClassify;
  $$("#clsTabs button").forEach((t) => (t.onclick = () => { S.clsSub = +t.dataset.i; $$("#clsTabs button").forEach((x) => x.classList.toggle("on", x === t)); drawClsPane(); }));
  drawClsPane();
}
async function runClassify() {
  const f = curFile();
  S.clsRunning = true; render();
  const poll = setInterval(async () => {
    try {
      const p = await api(`/api/progress/${f.sid}`);
      const bar = $("#clsBar"), txt = $("#clsTxt");
      if (bar && p.total) { bar.style.width = (p.done / p.total) * 100 + "%"; txt.textContent = `Classified ${p.done} of ${p.total} grains`; }
    } catch { /* ignore */ }
  }, 700);
  try {
    S.cls = await api("/api/classify", { sid: f.sid, params: payload(), threshold: S.set.thr, batch: S.set.cbatch });
    if (S.cls.state === "ok") S.clsShown = Object.keys(S.cfg.categories).filter((c) => c !== "Unclassified");
  } catch (e) { toast(e.message); }
  finally { clearInterval(poll); S.clsRunning = false; render(); }
}

/* ---- AI report ---- */
function viewAI() {
  const n = S.det.stats.count, cfg = S.cfg;
  let html = `<div class="card card-pad">${head("AI wheat analysis",
    "Groq describes grain appearance, colour, shape and possible damage. OpenCV still does all detection and counting. Clicking the button sends the numbered overview image and a grid of grain close-ups to Groq.")}`;
  if (!cfg.ai_available) html += callout("info", "AI analysis is unavailable in this installation.");
  else {
    if (!cfg.groq) html += callout("info", "Optional feature: add your Groq API key to the .env file to enable this section, then reload the page.");
    if (!n) html += `<div class="small muted">No grains detected yet, so there is nothing to analyse.</div>`;
    html += `<div style="margin-top:12px"><button class="btn primary" id="aiBtn" ${!cfg.groq || !n || S.aiRunning ? "disabled" : ""}>${S.aiRunning ? "Asking Groq…" : "Generate AI report"}</button></div>`;
  }
  html += `</div>`;
  if (S.aiRunning) html += `<div class="card card-pad empty"><div class="spin" style="margin:0 auto 12px"></div>Asking Groq to analyse the grains…</div>`;
  else if (S.ai) {
    if (S.aiKey !== detKey()) html += callout("warn", "This report was generated for different detection settings. Click 'Generate AI report' to refresh it.");
    html += S.ai.ok ? `<div class="card card-pad">${md(S.ai.report)}</div>`
      : callout("bad", S.ai.error) + (S.ai.hint ? `<div class="small muted">${esc(S.ai.hint)}</div>` : "") + `<div class="small muted">Detection results are unaffected.</div>`;
  }
  return html;
}
async function runAI() {
  const f = curFile();
  S.aiRunning = true; render();
  try { S.ai = await api("/api/ai", { sid: f.sid, params: payload() }); S.aiKey = detKey(); }
  catch (e) { S.ai = { ok: false, error: e.message, hint: "" }; S.aiKey = detKey(); }
  finally { S.aiRunning = false; render(); }
}

/* ---- export ---- */
function dlBtn(label, kind, enabled = true) {
  return enabled ? `<a class="btn" href="${dlUrl(kind)}" download>${dlIcon} ${label}</a>` : `<span class="btn" style="opacity:.45;cursor:not-allowed">${dlIcon} ${label}</span>`;
}
function viewExport() {
  const clsOk = S.cls && S.cls.state === "ok", aiOk = S.ai && S.ai.ok && S.aiKey === detKey();
  return `
  <div class="card card-pad">${head("Detection files")}<div class="dl-grid">
    ${dlBtn("Processed image (PNG)", "processed")}${dlBtn("Segmentation mask (PNG)", "mask")}${dlBtn("Grain data (CSV)", "csv")}</div></div>
  <div class="card card-pad">${head("Classification results", clsOk ? "" : "Run “Classify grains with Groq” in the Quality tab to enable these downloads.")}<div class="dl-grid">
    ${dlBtn("Classification report (PDF)", "pdf", clsOk)}${dlBtn("Per-grain results (CSV)", "cls_csv", clsOk)}${dlBtn("Category summary (CSV)", "cls_summary", clsOk)}</div></div>
  <div class="card card-pad">${head("AI report", aiOk ? "" : "Generate an AI report in the AI report tab to enable these downloads.")}<div class="dl-grid">
    ${dlBtn("Report (Markdown)", "md", aiOk)}${dlBtn("Report (Text)", "txt", aiOk)}</div></div>`;
}

/* ---- batch ---- */
async function loadBatch() {
  const key = detKey() + "|" + S.files.map((f) => f.sid).join();
  if (S.batchKey === key) return;            // already loaded or currently loading
  S.batchKey = key; S.batch = null;
  busy(true);
  try { S.batch = (await api("/api/batch", { sids: S.files.map((f) => f.sid), params: payload() })).rows; }
  catch (e) { toast(e.message); S.batchKey = ""; }
  finally { busy(false); }
  if (S.tab === "batch") render();
}
function viewBatch() {
  if (!S.batch) return `<div class="card card-pad empty"><div class="spin" style="margin:0 auto 12px"></div>Processing ${S.files.length} images…</div>`;
  const ok = S.batch.filter((r) => r.grains != null), total = ok.reduce((a, r) => a + r.grains, 0);
  let html = "";
  if (ok.length) {
    const g = ok.map((r) => r.grains);
    html += `<div class="kpis">${kpi("Images", S.batch.length, "", "neutral")}${kpi("Total grains", n0(total), "", "good")}
      ${kpi("Mean grains / image", n0(total / ok.length), "", "")}${kpi("Range", `${Math.min(...g)} – ${Math.max(...g)}`, "grains per image", "neutral")}</div>
      <div class="card card-pad">${head("Grains per image")}${barSvg(ok.map((r) => ({ label: r.name, value: r.grains })), { w: 760, h: 300, b: 90 })}</div>`;
  }
  html += `<div class="card card-pad">${head("Batch comparison", "All uploaded images processed with the current detection settings.",
    `<button class="btn" id="batchCsv">${dlIcon} Batch summary (CSV)</button>`)}
    <div class="table-wrap"><table><thead><tr><th>Image</th><th class="num">Grains</th><th class="num">Avg length (px)</th><th class="num">Avg width (px)</th><th class="num">Size variation (%)</th><th class="num">Filtered blobs</th><th>Resolution</th></tr></thead><tbody>
    ${S.batch.map((r) => `<tr><td>${esc(r.name)}</td><td class="num">${r.grains ?? "-"}</td><td class="num">${r.grains != null ? n1(r.avg_length) : "-"}</td><td class="num">${r.grains != null ? n1(r.avg_width) : "-"}</td>
      <td class="num">${r.grains != null ? n1(r.variation) : "-"}</td><td class="num">${r.rejected ?? "-"}</td><td>${esc(r.resolution)}</td></tr>`).join("")}</tbody></table></div></div>`;
  return html;
}
function bindBatch() {
  const b = $("#batchCsv");
  if (!b) return;
  b.onclick = () => {
    const head = ["Image", "Grains", "Avg length (px)", "Avg width (px)", "Size variation (%)", "Filtered blobs", "Resolution"];
    const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const csv = [head.map(q).join(",")].concat(S.batch.map((r) => [r.name, r.grains, r.avg_length, r.avg_width, r.variation, r.rejected, r.resolution].map(q).join(","))).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "wheat_batch_summary.csv"; a.click(); URL.revokeObjectURL(a.href);
  };
}

/* ----------------------------------------------------------------- router */
function render() {
  $$(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === S.tab));
  $("#pageTitle").textContent = S.files.length ? TITLES[S.tab] : "Welcome";
  const page = $("#page");
  if (!S.files.length) { page.innerHTML = landing(); bindLanding(); return; }
  if (!S.det && S.tab !== "batch") { page.innerHTML = loadingView(); return; }
  const drop = `<div class="dropzone" id="dropzone"><span class="small muted">Drag more photos here to replace the current set, or</span><button class="btn sm" id="dzBtn">Choose photos</button><span class="small muted" style="margin-left:auto">${esc(curFile().name)} · ${curFile().width} × ${curFile().height}px</span></div>`;
  const views = { overview: viewOverview, detection: viewDetection, quality: viewQuality, ai: viewAI, export: viewExport, batch: viewBatch };
  page.innerHTML = drop + views[S.tab]();
  $("#dzBtn").onclick = () => $("#fileInput").click();
  const binds = { overview: bindOverview, detection: bindDetection, quality: bindQuality, batch: bindBatch };
  if (binds[S.tab]) binds[S.tab]();
  if (S.tab === "ai" && $("#aiBtn")) $("#aiBtn").onclick = runAI;
  $$(".zoomable img", page).forEach((im) => (im.onclick = () => zoom(im.src)));
  if (S.tab === "batch") loadBatch();
}
function bindLanding() {
  const b = $("#heroBtn");
  if (b) b.onclick = () => $("#fileInput").click();
}
function zoom(src) {
  const o = document.createElement("div");
  o.className = "lightbox"; o.innerHTML = `<img src="${src}" alt="Enlarged image">`;
  o.onclick = () => o.remove();
  document.addEventListener("keydown", function esc_(e) { if (e.key === "Escape") { o.remove(); document.removeEventListener("keydown", esc_); } });
  document.body.appendChild(o);
}
async function setTab(t) {
  S.tab = t;
  $("#sidebar").classList.remove("open");
  render();
  if (S.det && ["overview", "quality", "export"].includes(t)) { await refreshCls(true); render(); }
}

/* ------------------------------------------------------------------- init */
function initTheme() {
  let t = null;
  try { t = localStorage.getItem("wheat-theme"); } catch { /* storage unavailable */ }
  if (!t) t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = t;
  $("#themeBtn").onclick = () => {
    const n = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = n;
    try { localStorage.setItem("wheat-theme", n); } catch { /* ignore */ }
  };
}
async function init() {
  initTheme();
  try { S.cfg = await api("/api/status"); }
  catch (e) { $("#page").innerHTML = callout("bad", "Could not reach the server: " + e.message); return; }
  $("#groqDot").className = "dot " + (S.cfg.groq ? "ok" : "off");
  $("#groqText").textContent = S.cfg.groq ? "Groq connected" : "Groq key not set (optional)";
  $("#modelText").textContent = S.cfg.groq ? S.cfg.model : "";
  resetSettings(); buildSettings(); syncControls();

  $$(".nav-btn").forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));
  $("#fileInput").onchange = (e) => { uploadFiles(e.target.files); e.target.value = ""; };
  $("#uploadBtn").onclick = () => $("#fileInput").click();
  $("#fileSelect").onchange = (e) => selectFile(+e.target.value);
  $("#resetBtn").onclick = () => { resetSettings(); runDetect(); if (S.tab === "quality") refreshCls(); };
  $("#settingsToggle").onclick = () => $("#settings").classList.toggle("open");
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#settings").classList.remove("open"); });
  $("#settingsClose").onclick = () => $("#settings").classList.remove("open");
  $("#menuBtn").onclick = () => $("#sidebar").classList.toggle("open");

  ["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); $$(".hero,.dropzone").forEach((x) => x.classList.add("drag")); }));
  ["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => { e.preventDefault(); if (ev === "drop" || e.target === document.documentElement || !e.relatedTarget) $$(".hero,.dropzone").forEach((x) => x.classList.remove("drag")); }));
  document.addEventListener("drop", (e) => { if (e.dataTransfer && e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files); });
  render();
}
init();
