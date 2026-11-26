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
  model: null,               // /api/model (the model trained in the terminal with train.py)
  pred: {},                  // sid -> model predictions
  sel: null, zoom: 1, showIds: true,
  qsrc: "model", qShown: null, qOnlyReview: false,
  ai: null, aiKey: "", aiRunning: false,
  batch: null, batchKey: "",
};
const TITLES = { overview: "Overview", detection: "Detection", predict: "Predict grain types", quality: "Grain quality report", ai: "AI report", export: "Export", batch: "Batch comparison" };
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
  { title: "Quality grading", items: [
    { t: "range", k: "thr", min: 0, max: 1, step: 0.05, label: "Manual-inspection threshold", pct: true, hint: "Grains graded below this confidence are flagged for review. Updates instantly; no new Groq calls." },
    { t: "range", k: "cbatch", min: 1, max: 5, step: 1, label: "Grains per Groq request (Groq only)", hint: "1 = most faithful but most API calls. Higher = fewer calls." } ] },
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
  if (key === "thr") { clearTimeout(clsTimer); clsTimer = setTimeout(async () => { await refreshCls(true); if (S.det && ["overview", "quality"].includes(S.tab)) render(); }, 250); return; }
  if (key === "cbatch") { if (S.tab === "quality") render(); return; }
  if (S.det) S.sel = null;
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
    S.pred = {}; S.sel = null;
    Object.keys(cropImgs).forEach((k) => delete cropImgs[k]);
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
  S.cur = i; S.det = null; S.cls = null; S.ai = null; S.clsSub = 0; S.clsShown = null; S.sel = null;
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
  const n = S.det.stats.count, tr = S.model, qd = qualityData(), defs = S.cfg.class_defs || {};
  let body = "";
  if (qd) {
    const q = qualityTable(qd.rows);
    body += `<div class="cond-grid">${q.cats.filter((x) => x.cat !== "Unclassified" || x.count).map((x) => `<div class="cond" style="--c:${colorOf(x.cat)}" title="${esc(defs[x.cat] || "")}">
      <div class="nm">${esc(x.cat)}</div><div class="ct">${x.count}</div><div class="pc">${pc1(x.pct)} of graded</div></div>`).join("")}</div>
      <div class="small muted" style="margin-bottom:10px">${q.total.count} grains graded by ${esc(qd.source)}; ${q.total.review} flagged for manual review.</div>`;
  } else if (S.qsrc === "model") {
    if (!tr || !tr.available) body += callout("info", (tr && tr.reason) || "The model module is unavailable in this installation.");
    else if (!tr.model.trained) body += callout("info", "No trained model found. Train one in the terminal (python train.py) to grade grains as Sound, Shrivelled, Immature, Broken or Damaged/Discoloured.");
    else if (n) body += `<div class="small muted" style="margin-bottom:10px">Grading ${n} grains…</div>`;
  } else body += callout("info", "Run the Groq classification in the Quality tab to see the grain types here.");
  body += `<div style="display:flex;gap:10px;flex-wrap:wrap"><button class="btn primary" id="goQuality">Open the full quality table</button><button class="btn" id="goTrain">${tr && tr.available && tr.model.trained ? "Model details &amp; prediction" : "How to train the model"}</button></div>`;
  return `<div class="card card-pad">${head("Grain condition", "Grain types counted from the photo. The detailed table, with one row per grain type, is in the Quality tab.")}${body}</div>`;
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
  const gq = $("#goQuality"), gtr = $("#goTrain");
  if (gq) gq.onclick = () => setTab("quality");
  if (gtr) gtr.onclick = () => setTab("predict");
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

/* ======================================================================== */
/*  Shared helpers: model predictions, image overlay                         */
/* ======================================================================== */
const modelTrained = () => !!(S.model && S.model.available && S.model.model.trained);
const modelVer = () => (S.model && S.model.model && S.model.model.version) || "";
const predKey = () => detKey() + "|" + modelVer();
const gCenter = (g) => [g.x + g.box_w / 2, g.y + g.box_h / 2];
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
const colorOf = (c) => S.cfg.colors[c] || "#7f7f7f";

/* Predictions of your trained model, cached per image + detection settings + model version. */
const predRows = (sid = curFile().sid) => { const p = S.pred[sid]; return p && p.key === predKey() ? p.rows : null; };
function predMap() { const m = {}; (predRows() || []).forEach((r) => (m[r.id] = r)); return m; }
async function ensurePred(sids) {
  if (!modelTrained()) return false;
  const key = predKey(), need = sids.filter((s) => !(S.pred[s] && S.pred[s].key === key));
  if (!need.length) return false;
  busy(true);
  try {
    const r = await api("/api/predict", { sids: need, params: payload() });
    if (key !== predKey()) return false;                       // settings or model changed meanwhile
    need.forEach((s) => { const x = r.results[s] || {}; S.pred[s] = { key, rows: x.error ? [] : x.rows || [], error: x.error || "" }; });
    return true;
  } catch (e) {
    need.forEach((s) => (S.pred[s] = { key, rows: [], error: e.message }));   // stop retrying until something changes
    toast(e.message);
    return true;
  } finally { busy(false); }
}
async function autoPred() {
  if (!S.det || !modelTrained()) return;
  const sid = curFile().sid, tab = S.tab;
  if (tab === "predict") {
    if ((await ensurePred([sid])) && S.tab === "predict") render();
  } else if (tab === "quality" && S.qsrc === "model") {
    if (await ensurePred([sid])) render();
    const others = S.files.map((f) => f.sid).filter((s) => s !== sid);
    if (others.length && (await ensurePred(others)) && S.tab === "quality") render();
  } else if (["overview", "export"].includes(tab) && S.qsrc === "model") {
    if ((await ensurePred([sid])) && S.tab === tab) render();
  }
}

/* The image with one clickable box per grain (SVG drawn on top of the photo). */
function stageShell(id, zoom = 1) {
  const src = S.originals[curFile().sid];
  return `<div class="stage-wrap" id="${id}Wrap"><div class="stage" id="${id}Stage" style="width:${zoom * 100}%"><img src="${src}" alt="Grain photo" draggable="false"><svg class="ov" id="${id}Svg"></svg></div></div>`;
}
function fillOverlay(svg, styleFn, selId, showIds) {
  const d = S.det, W = d.width, H = d.height, fs = Math.max(10, W / 80);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const gs = [...d.grains].sort((a, b) => b.area - a.area);        // big first, small grains stay clickable on top
  let boxes = "", txt = "", sel = "";
  gs.forEach((g) => {
    const s = styleFn(g);
    boxes += `<rect data-id="${g.id}" x="${g.x}" y="${g.y}" width="${g.box_w}" height="${g.box_h}" rx="3" fill="${s.color}" fill-opacity="${s.fill}" stroke="${s.color}" stroke-width="${s.w || 2}"${s.dash ? ' stroke-dasharray="5 3"' : ""}><title>${esc(s.tip || "")}</title></rect>`;
    if (showIds) txt += `<text x="${g.x}" y="${g.y - 3}" font-size="${fs}" style="stroke-width:${(fs * 0.22).toFixed(1)}px">${g.id}${s.star ? "*" : ""}</text>`;
    if (g.id === selId) sel = `<rect x="${g.x - 3}" y="${g.y - 3}" width="${g.box_w + 6}" height="${g.box_h + 6}" rx="5" fill="none" stroke="#111" stroke-width="5" pointer-events="none"/><rect x="${g.x - 3}" y="${g.y - 3}" width="${g.box_w + 6}" height="${g.box_h + 6}" rx="5" fill="none" stroke="#fff" stroke-width="2" pointer-events="none"/>`;
  });
  svg.innerHTML = boxes + sel + txt;
}

/* ======================================================================== */
/*  Quality report: grain types as parameters, shown as a table              */
/* ======================================================================== */
function qualityData() {
  if (S.qsrc === "groq") {
    const c = S.cls;
    if (!c || c.state !== "ok") return null;
    return { source: `Groq (${c.model || "vision model"})`,
      rows: c.rows.map((r) => ({ id: r.id, category: r.category, confidence: r.confidence, length: r.length_px, width: r.width_px, area: r.area_px,
        review: r.manual_inspection === "Required", reason: r.review_reasons || "" })) };
  }
  const rows = predRows();
  if (!rows) return null;
  const thr = S.set.thr;
  return { source: `Your trained model (${S.model.model.n_samples} training grains)`,
    rows: rows.map((r) => ({ id: r.id, category: r.label, confidence: r.confidence, length: r.length, width: r.width, area: r.area,
      review: r.confidence < thr, reason: r.confidence < thr ? `Confidence below ${Math.round(thr * 100)}%` : "" })) };
}
function qualityTable(rows) {
  const cats = [...S.cfg.classes];
  const unc = rows.filter((r) => r.category === "Unclassified");
  if (unc.length) cats.push("Unclassified");
  const graded = rows.filter((r) => r.category !== "Unclassified"), tot = graded.length, totArea = graded.reduce((a, r) => a + r.area, 0);
  const agg = (items, isUnc) => {
    const conf = items.filter((r) => r.confidence != null).map((r) => r.confidence), a = items.reduce((s, r) => s + r.area, 0);
    return { count: items.length, pct: isUnc ? null : tot ? (items.length / tot) * 100 : 0, areaPct: isUnc ? null : totArea ? (a / totArea) * 100 : 0,
      avgL: mean(items.map((r) => r.length)), avgW: mean(items.map((r) => r.width)), avgA: mean(items.map((r) => r.area)),
      conf: mean(conf), review: items.filter((r) => r.review).length };
  };
  return { cats: cats.map((c) => ({ cat: c, ...agg(rows.filter((r) => r.category === c), c === "Unclassified") })),
    total: agg(graded), defective: agg(graded.filter((r) => r.category !== "Sound")), unclassified: unc.length };
}
const pc1 = (v) => (v == null ? "-" : v.toFixed(1) + "%");
const dash = (v, f) => (v == null ? "-" : f(v));
function qualityTableHtml(q) {
  const defs = S.cfg.class_defs || {};
  const row = (name, x, o = {}) => `<tr class="${o.cls || ""}">
    <td class="pname">${o.color ? `<i class="sw" style="background:${o.color}"></i>` : ""}<div><b>${esc(name)}</b>${o.sub ? `<div class="psub">${esc(o.sub)}</div>` : ""}</div></td>
    <td class="num"><b>${x.count}</b></td>
    <td class="num">${x.pct == null ? "-" : `<div class="pcell"><span>${pc1(x.pct)}</span><div class="pbar"><i style="width:${Math.min(100, x.pct)}%;background:${o.color || "var(--primary)"}"></i></div></div>`}</td>
    <td class="num">${pc1(x.areaPct)}</td><td class="num">${dash(x.avgL, n1)}</td><td class="num">${dash(x.avgW, n1)}</td><td class="num">${dash(x.avgA, n0)}</td>
    <td class="num">${x.conf == null ? "-" : Math.round(x.conf * 100) + "%"}</td><td class="num">${x.review ? `<span class="tag Required">${x.review}</span>` : "0"}</td></tr>`;
  return `<div class="table-wrap qwrap"><table class="qtable"><thead><tr>
    <th>Parameter (grain type)</th><th class="num">Grains</th><th class="num">% by count</th><th class="num" title="Share of the total grain pixel area, a stand-in for weight share">% by area*</th>
    <th class="num">Avg length (px)</th><th class="num">Avg width (px)</th><th class="num">Avg area (px²)</th><th class="num">Avg confidence</th><th class="num">Needs review</th></tr></thead><tbody>
    ${q.cats.map((x) => row(x.cat, x, { color: colorOf(x.cat), sub: x.cat === "Unclassified" ? "Could not be graded" : defs[x.cat] })).join("")}
    ${row("Total graded", q.total, { cls: "qtotal" })}
    ${row("Defective grains (all types except Sound)", q.defective, { cls: "qdef" })}</tbody></table></div>`;
}
function csvDownload(name, matrix) {
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + matrix.map((r) => r.map(q).join(",")).join("\n")], { type: "text/csv" }));
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function qualityCsv() {
  const qd = qualityData(); if (!qd) return;
  const q = qualityTable(qd.rows), r1 = (v, k) => (v == null ? "" : +v.toFixed(k));
  const line = (name, x) => [name, x.count, r1(x.pct, 1), r1(x.areaPct, 1), r1(x.avgL, 1), r1(x.avgW, 1), r1(x.avgA, 0), x.conf == null ? "" : r1(x.conf * 100, 0), x.review];
  csvDownload(`${curFile().name.replace(/\.[^.]+$/, "")}_quality_table.csv`, [
    ["Parameter (grain type)", "Grains", "% by count", "% by area", "Avg length (px)", "Avg width (px)", "Avg area (px2)", "Avg confidence (%)", "Needs review"],
    ...q.cats.map((x) => line(x.cat, x)), line("Total graded", q.total), line("Defective grains (all except Sound)", q.defective)]);
}
function qualityGrainCsv() {
  const qd = qualityData(); if (!qd) return;
  csvDownload(`${curFile().name.replace(/\.[^.]+$/, "")}_quality_per_grain.csv`, [
    ["Grain ID", "Grain type", "Confidence (%)", "Needs review", "Length (px)", "Width (px)", "Area (px2)"],
    ...qd.rows.map((r) => [r.id, r.category, r.confidence == null ? "" : Math.round(r.confidence * 100), r.review ? "Yes" : "No", r.length, r.width, r.area])]);
}

function viewQuality() {
  const cfg = S.cfg, n = S.det.stats.count, tr = S.model, sid = curFile().sid;
  const seg = `<div class="seg" id="qSrc">${[["model", "My trained model"], ["groq", "Groq (optional)"]].map(([k, l]) => `<button data-v="${k}" class="${S.qsrc === k ? "on" : ""}">${l}</button>`).join("")}</div>`;
  let html = `<div class="card card-pad">${head("Grain quality report", "Every detected grain is graded and the results are summed up per grain type. Uncertain grains are flagged for manual inspection, never hidden.", seg)}`;
  let waiting = false;

  if (S.qsrc === "model") {
    if (!tr || !tr.available) html += callout("info", (tr && tr.reason) || "The model module is unavailable in this installation.");
    else if (!tr.model.trained) html += callout("info", "No trained model found. Train one in the terminal with  python train.py  and this table fills in automatically.") +
      `<div style="margin-top:12px"><button class="btn primary" id="goTrain">How to train the model</button></div>`;
    else if (!n) html += callout("warn", "No grains detected, so there is nothing to grade.");
    else if (!predRows(sid)) { waiting = true; html += `<div class="empty"><div class="spin" style="margin:0 auto 12px"></div>Grading ${n} grains with your model…</div>`; }
  } else {
    const batch = S.set.cbatch, calls = Math.ceil(n / batch) || 0;
    if (!cfg.ai_available) html += callout("info", "Groq classification is unavailable in this installation.");
    else {
      if (!cfg.groq) html += callout("info", "Optional: add your Groq API key to the .env file (see .env.example) and reload the page to grade grains with Groq instead of your own model.");
      html += `<div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:6px">
        <button class="btn primary" id="clsBtn" ${!cfg.groq || !n || S.clsRunning ? "disabled" : ""}>${S.clsRunning ? "Classifying…" : "Classify grains with Groq"}</button>
        <span class="small muted">${n ? `${n} grains will be sent in ${calls} request${calls !== 1 ? "s" : ""}. Nothing is sent until you click.` : "No grains detected yet."}</span></div>
        <div class="progress ${S.clsRunning ? "" : "hidden"}" style="margin-top:12px"><div id="clsBar"></div></div><div class="small muted ${S.clsRunning ? "" : "hidden"}" id="clsTxt" style="margin-top:6px"></div>`;
      if (S.cls && S.cls.state === "stale") html += `<div style="margin-top:10px">${callout("warn", "Detection settings changed since the last Groq run, so the grain IDs no longer match. Classify again.")}</div>`;
      if (S.cls && S.cls.state === "error") html += `<div style="margin-top:10px">${callout("bad", S.cls.error)}</div>`;
      if (S.cls && S.cls.state === "ok" && S.cls.fatal_error) html += `<div style="margin-top:10px">${callout("bad", S.cls.fatal_error)}</div>`;
    }
  }

  const qd = waiting ? null : qualityData();
  if (!qd) return html + `</div>`;
  const q = qualityTable(qd.rows);
  html += `<div class="rep-meta"><span><b>Sample:</b> ${esc(curFile().name)}</span><span><b>Grains graded:</b> ${q.total.count}</span><span><b>Graded by:</b> ${esc(qd.source)}</span>
    <span class="grow"></span><button class="btn sm" id="qCsv">${dlIcon} Table (CSV)</button></div>`;
  html += qualityTableHtml(q);
  html += `<div class="small muted" style="margin-top:8px">*% by area is each type's share of the total grain pixel area. It approximates the share by weight but is not a measured weight. Lengths and widths are in pixels of the original photo.
    ${q.unclassified ? `${q.unclassified} grain${q.unclassified > 1 ? "s" : ""} could not be graded and ${q.unclassified > 1 ? "are" : "is"} excluded from the percentages.` : ""}</div></div>`;

  const sound = q.cats.find((x) => x.cat === "Sound");
  html += `<div class="kpis">${kpi("Grains graded", q.total.count, qd.source.split(" (")[0], "neutral")}${kpi("Sound", pc1(sound.pct), `${sound.count} grains`, sound.pct >= 85 ? "good" : sound.pct >= 60 ? "neutral" : "warn")}
    ${kpi("Defective", pc1(q.defective.pct), `${q.defective.count} grains`, q.defective.pct <= 10 ? "good" : q.defective.pct <= 30 ? "warn" : "bad")}
    ${kpi("Needs manual review", q.total.review, `confidence below ${Math.round(S.set.thr * 100)}%`, q.total.review ? "warn" : "good")}</div>`;

  /* all images, grain types as rows */
  if (S.qsrc === "model" && S.files.length > 1) html += `<div class="card card-pad" id="qMatrix">${matrixHtml()}</div>`;

  html += `<div class="card card-pad">${head("Graded photo", "Box colour = grain type. A dashed box with * is flagged for manual inspection. Hover a grain for details.")}
    <div class="grid-21"><div>${stageShell("q")}</div><div>
      <b>Grains per type</b>${barSvg(q.cats.map((x) => ({ label: x.cat, value: x.count, color: colorOf(x.cat) })), { w: 420, h: 250, b: 80 })}
      <b>Share by count</b>${q.total.count ? donutSvg(q.cats.filter((x) => x.cat !== "Unclassified").map((x) => ({ label: x.cat, value: x.count, color: colorOf(x.cat) }))) : ""}</div></div>
    ${legend(Object.fromEntries(Object.entries(S.cfg.colors).filter(([k]) => k !== "Unclassified" || q.unclassified)))}</div>`;

  html += `<div class="card card-pad">${head("Per-grain results", "Filter by grain type or show only the grains that need a second look.",
    `<button class="btn sm" id="qCsv2">${dlIcon} Per-grain (CSV)</button>`)}
    <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:12px">
      ${q.cats.map((x) => `<label class="check"><input type="checkbox" data-cat="${esc(x.cat)}" ${S.qShown.includes(x.cat) ? "checked" : ""}><i class="sw" style="background:${colorOf(x.cat)}"></i>${esc(x.cat)}</label>`).join("")}
      <label class="check" style="margin-left:auto"><input type="checkbox" id="qOnlyRev" ${S.qOnlyReview ? "checked" : ""}>Only needs review</label></div><div id="qGrainTable"></div></div>`;
  return html;
}
function matrixHtml() {
  const files = S.files, cols = files.map((f) => ({ f, rows: predRows(f.sid) }));
  const qs = cols.map((c) => (c.rows ? qualityTable(c.rows.map((r) => ({ category: r.label, confidence: r.confidence, length: r.length, width: r.width, area: r.area, review: r.confidence < S.set.thr }))) : null));
  const cell = (x) => (x ? `${x.count} <span class="muted">(${x.pct.toFixed(1)}%)</span>` : `<span class="muted">…</span>`);
  const tot = qs.reduce((a, q) => a + (q ? q.total.count : 0), 0);
  const cats = S.cfg.classes;
  return `${head("All images · grain types as parameters", "Count and share of each grain type for every uploaded photo.", `<button class="btn sm" id="qCsvAll">${dlIcon} All images (CSV)</button>`)}
    <div class="table-wrap qwrap"><table class="qtable"><thead><tr><th>Parameter (grain type)</th>${files.map((f, i) => `<th class="num ${i === S.cur ? "curcol" : ""}" title="${esc(f.name)}">${esc(f.name.length > 18 ? f.name.slice(0, 17) + "…" : f.name)}</th>`).join("")}<th class="num">All images</th></tr></thead><tbody>
    ${cats.map((c) => { const all = qs.reduce((a, q) => a + (q ? q.cats.find((x) => x.cat === c).count : 0), 0);
      return `<tr><td class="pname"><i class="sw" style="background:${colorOf(c)}"></i><b>${esc(c)}</b></td>${qs.map((q) => `<td class="num">${cell(q && q.cats.find((x) => x.cat === c))}</td>`).join("")}<td class="num"><b>${all}</b> <span class="muted">(${tot ? ((all / tot) * 100).toFixed(1) : "0.0"}%)</span></td></tr>`; }).join("")}
    <tr class="qtotal"><td class="pname"><b>Total graded</b></td>${qs.map((q) => `<td class="num"><b>${q ? q.total.count : "…"}</b></td>`).join("")}<td class="num"><b>${tot}</b></td></tr>
    <tr class="qdef"><td class="pname"><b>Defective (all except Sound)</b></td>${qs.map((q) => `<td class="num">${q ? `${q.defective.count} <span class="muted">(${q.defective.pct.toFixed(1)}%)</span>` : "…"}</td>`).join("")}<td class="num">${(() => { const d = qs.reduce((a, q) => a + (q ? q.defective.count : 0), 0); return `${d} <span class="muted">(${tot ? ((d / tot) * 100).toFixed(1) : "0.0"}%)</span>`; })()}</td></tr>
    </tbody></table></div>`;
}
function drawQGrainTable() {
  const qd = qualityData(); if (!qd) return;
  const rows = qd.rows.filter((r) => S.qShown.includes(r.category) && (!S.qOnlyReview || r.review));
  $("#qGrainTable").innerHTML = `<div class="table-wrap"><table><thead><tr><th class="num">ID</th><th>Grain type</th><th class="num">Confidence</th><th>Manual inspection</th><th>Note</th><th class="num">Length (px)</th><th class="num">Width (px)</th><th class="num">Area (px²)</th></tr></thead><tbody>
    ${rows.map((r) => `<tr><td class="num">${r.id}</td><td><span class="tag cat" style="background:${colorOf(r.category)}">${esc(r.category)}</span></td>
      <td class="num">${r.confidence == null ? "-" : Math.round(r.confidence * 100) + "%"}</td><td><span class="tag ${r.review ? "Required" : "Normal"}">${r.review ? "Required" : "Not required"}</span></td>
      <td style="white-space:normal;min-width:180px">${esc(r.reason)}</td><td class="num">${n1(r.length)}</td><td class="num">${n1(r.width)}</td><td class="num">${n0(r.area)}</td></tr>`).join("")}
    </tbody></table></div><div class="small muted" style="margin-top:8px">Showing ${rows.length} of ${qd.rows.length} grains. Confidence is ${S.qsrc === "model" ? "the share of the model's decision trees that voted for the type; it is not a calibrated probability" : "reported by the Groq model; it is not a calibrated probability"}.</div>`;
}
function bindQuality() {
  $$("#qSrc button").forEach((b) => (b.onclick = () => { S.qsrc = b.dataset.v; S.qShown = [...S.cfg.classes, "Unclassified"]; render(); }));
  const gt = $("#goTrain"); if (gt) gt.onclick = () => setTab("predict");
  const cb = $("#clsBtn"); if (cb) cb.onclick = runClassify;
  const qd = qualityData();
  if (!qd) return;
  $("#qCsv").onclick = qualityCsv;
  $("#qCsv2").onclick = qualityGrainCsv;
  const all = $("#qCsvAll");
  if (all) all.onclick = () => {
    const rows = [["Parameter (grain type)", ...S.files.map((f) => f.name + " (count)"), ...S.files.map((f) => f.name + " (%)")]];
    const qs = S.files.map((f) => { const r = predRows(f.sid); return r ? qualityTable(r.map((x) => ({ category: x.label, confidence: x.confidence, length: x.length, width: x.width, area: x.area, review: false }))) : null; });
    S.cfg.classes.forEach((c) => rows.push([c, ...qs.map((q) => (q ? q.cats.find((x) => x.cat === c).count : "")), ...qs.map((q) => (q ? +q.cats.find((x) => x.cat === c).pct.toFixed(1) : ""))]));
    rows.push(["Total graded", ...qs.map((q) => (q ? q.total.count : "")), ...qs.map(() => "")]);
    csvDownload("quality_all_images.csv", rows);
  };
  const byId = {}; qd.rows.forEach((r) => (byId[r.id] = r));
  fillOverlay($("#qSvg"), (g) => { const r = byId[g.id]; const c = r ? colorOf(r.category) : "#7f7f7f";
    return { color: c, fill: 0.18, dash: r && r.review, star: r && r.review, w: 2.5, tip: r ? `#${g.id} · ${r.category}${r.confidence != null ? " · " + Math.round(r.confidence * 100) + "%" : ""}${r.review ? " · needs review" : ""}` : `#${g.id}` }; }, null, true);
  $$("[data-cat]").forEach((c) => (c.onchange = () => { S.qShown = $$("[data-cat]").filter((x) => x.checked).map((x) => x.dataset.cat); drawQGrainTable(); }));
  $("#qOnlyRev").onchange = (e) => { S.qOnlyReview = e.target.checked; drawQGrainTable(); };
  drawQGrainTable();
}

/* ======================================================================== */
/*  Predict: the model is TRAINED in the terminal (python train.py);         */
/*  this tab only shows the model and grades the uploaded photo with it.     */
/* ======================================================================== */
const cropImgs = {};                                                     // sid -> Image (for the zoomed crop)
function getCropImg(sid) {
  let im = cropImgs[sid];
  if (!im) {
    im = cropImgs[sid] = new Image();
    im.onload = () => { const g = S.sel && S.det ? grainById(S.sel) : null; if (g && S.tab === "predict") drawCrop(g); };
    im.src = S.originals[sid];
  }
  return im;
}
const grainById = (id) => S.det.grains.find((g) => g.id === id);
const pct0 = (v) => Math.round(v * 100) + "%";

function noModelHtml(why) {
  return `<div class="card card-pad">${head("No trained model yet", "The model is trained in the terminal, not in the browser. Do it once, then come back here.")}
    ${why ? callout("warn", why) + `<div style="height:12px"></div>` : ""}
    <div class="mini-steps">
      <div class="mini-step"><b class="n">1</b><div><b>Sort your photos</b><div class="small muted">One folder per grain type inside <code>dataset/</code>, each holding photos of that type only. <code>python train.py --make-folders</code> creates the folders.</div></div></div>
      <div class="mini-step"><b class="n">2</b><div><b>Train in the terminal</b><pre class="cmd">python train.py</pre><div class="small muted">Prints the accuracy on photos it did not train on, then saves <code>models/grain_model.joblib</code>.</div></div></div>
      <div class="mini-step"><b class="n">3</b><div><b>Predict here</b><div class="small muted">Press <i>Reload model</i> (or switch tabs), then upload any new photo.</div></div></div>
    </div>
    <button class="btn primary" id="reloadModel">Reload model</button></div>`;
}

function modelCard(m) {
  const ev = m.evaluation, cats = S.cfg.classes, d = m.detect;
  let html = `<div class="card card-pad">${head("Trained model", "Trained in the terminal with python train.py. Retrain there, then press Reload model.",
    `<button class="btn sm" id="reloadModel">Reload model</button>`)}
    <div class="kpis">
      ${kpi("Trained", (m.trained_at || "-").split(" ")[0], `${(m.trained_at || "").split(" ")[1] || ""} · ${m.n_samples} grains${m.n_photos ? " · " + m.n_photos + " photos" : ""}`, "neutral")}
      ${ev ? kpi("Test accuracy", (ev.accuracy * 100).toFixed(1) + "%", ev.label, ev.accuracy >= 0.9 ? "good" : ev.accuracy >= 0.75 ? "neutral" : "warn") : kpi("Test accuracy", "-", "not evaluated", "neutral")}
      ${ev ? kpi("Macro F1", (ev.macro_f1 * 100).toFixed(1) + "%", "average over grain types", "neutral") : ""}
      ${kpi("Size reference", m.normalize === "reference" ? "Fixed" : "Per photo", m.normalize === "reference" ? "photos need the training camera setup" : "median grain of each photo", "neutral")}
    </div>`;
  const missing = cats.filter((c) => !(m.counts && m.counts[c]));
  if (missing.length) html += `<div style="margin-top:10px">${callout("warn", `The model has no training grains for: ${missing.join(", ")}. It can never predict these types.`)}</div>`;
  if (ev && ev.note) html += `<div style="margin-top:10px">${callout("warn", ev.note)}</div>`;
  if (m.normalize === "reference") html += `<div class="small muted" style="margin-top:10px">Grain sizes are judged against a fixed reference grain from training, so photos should be taken at the same distance and in similar light as the training photos.</div>`;
  if (d) html += `<div class="small muted" style="margin-top:6px">Detection settings used in training: hue ${d.h_min}–${d.h_max}, saturation ${d.s_min}–${d.s_max}, value ${d.v_min}–${d.v_max}, grain area ${d.min_area}–${d.max_area} px.</div>`;

  const pc = ev ? ev.per_class : {};
  const rows = cats.map((c) => { const e = pc[c];
    return `<tr><td class="pname"><i class="sw" style="background:${colorOf(c)}"></i><b>${esc(c)}</b></td><td class="num">${(m.counts && m.counts[c]) || 0}</td>
      <td class="num">${e ? pct0(e.precision) : "-"}</td><td class="num">${e ? pct0(e.recall) : "-"}</td><td class="num">${e ? pct0(e.f1) : "-"}</td><td class="num">${e ? e.support : "-"}</td></tr>`; }).join("");
  let conf = "";
  if (ev && ev.confusion) {
    const L = ev.confusion.labels;
    conf = `<div style="margin-top:16px"><b>Confusion matrix</b> <span class="small muted">rows = true type, columns = predicted (green = correct)</span></div>
      <div class="table-wrap table-plain" style="margin-top:8px"><table class="qtable"><thead><tr><th>True ↓ / Predicted →</th>${L.map((c) => `<th class="num">${esc(c.length > 12 ? c.slice(0, 11) + "…" : c)}</th>`).join("")}</tr></thead><tbody>
      ${ev.confusion.matrix.map((row, i) => `<tr><td class="pname"><b>${esc(L[i])}</b></td>${row.map((v, j) => `<td class="num" style="${i === j && v ? "background:var(--ok-soft);font-weight:700" : ""}">${v}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  const top = m.top_features && m.top_features.length ? `<div class="small muted" style="margin-top:12px">Features the model relies on most: ${m.top_features.slice(0, 6).map(([k, v]) => `${esc(k)} (${v.toFixed(2)})`).join(", ")}.</div>` : "";
  html += `<details style="margin-top:14px"><summary class="small" style="cursor:pointer;font-weight:650">Per-type scores${ev ? " and confusion matrix" : ""}</summary>
    <div class="table-wrap table-plain" style="margin-top:8px"><table class="qtable"><thead><tr><th>Grain type</th><th class="num">Training grains</th>
      <th class="num" title="Of the grains the model called this type, how many really are">Precision</th><th class="num" title="Of the grains that really are this type, how many the model found">Recall</th><th class="num">F1</th><th class="num">Tested grains</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${conf}${top}
    <div class="small muted" style="margin-top:12px">To improve the model: add more photos to <code>dataset/&lt;type&gt;/</code> (especially for the weakest types above), run <code>python train.py</code> again, then press Reload model. Use <code>--test-dir</code> to score it on separate photos.</div></details></div>`;
  return html;
}

function viewPredict() {
  const ms = S.model, n = S.det.stats.count, sid = curFile().sid;
  if (!ms || !ms.available) return callout("info", (ms && ms.reason) || "The model module is unavailable in this installation.");
  const m = ms.model;
  if (!m.trained) return noModelHtml(m.error);
  let html = modelCard(m);
  const ctl = `<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><label class="check"><input type="checkbox" id="pIds" ${S.showIds ? "checked" : ""}>Show IDs</label>
    <div class="seg" id="pZoom">${[[1, "Fit"], [1.6, "160%"], [2.4, "240%"]].map(([z, l]) => `<button data-z="${z}" class="${S.zoom === z ? "on" : ""}">${l}</button>`).join("")}</div></div>`;
  html += `<div class="card card-pad">${head("Predict this photo", "Upload any new photo: every detected grain is graded by the trained model. Click a grain for details. Dashed boxes with * are low-confidence.", ctl)}`;
  const pr = S.pred[sid];
  if (!n) html += callout("warn", "No grains detected in this photo, so there is nothing to grade. Adjust the detection settings (button at the top right) until every grain has a box.") + `</div>`;
  else if (pr && pr.error && pr.key === predKey()) html += callout("bad", pr.error) + `</div>`;
  else if (!predRows(sid)) html += `<div class="empty"><div class="spin" style="margin:0 auto 12px"></div>Grading ${n} grains…</div></div>`;
  else html += `<div class="grid-21 train-grid"><div>${stageShell("p", S.zoom)}
      <div class="legend">${S.cfg.classes.map((c) => `<span><i style="background:${colorOf(c)}"></i>${esc(c)}</span>`).join("")}</div></div>
      <div class="train-side" id="pSide"></div></div></div>`;
  return html;
}

function predictStyle() {
  const pm = predMap(), thr = S.set.thr;
  return (g) => {
    const p = pm[g.id];
    if (!p) return { color: "#64748b", fill: 0.04, w: 1.8, tip: `#${g.id}` };
    const low = p.confidence < thr;
    return { color: colorOf(p.label), fill: 0.2, dash: low, star: low, w: 2.5,
      tip: `#${g.id} · ${p.label} ${pct0(p.confidence)}${low ? " · low confidence, check by eye" : ""}` };
  };
}

function fillPredictSide() {
  const el = $("#pSide"), rows = predRows(); if (!el || !rows) return;
  const thr = S.set.thr, tot = rows.length, low = rows.filter((r) => r.confidence < thr).length;
  const counts = S.cfg.classes.map((c) => ({ cat: c, n: rows.filter((r) => r.label === c).length }));
  const sel = S.sel ? rows.find((r) => r.id === S.sel) : null, g = S.sel ? grainById(S.sel) : null;
  let insp;
  if (sel && g) {
    insp = `${head(`Grain #${g.id}`, `${n0(g.length)} × ${n0(g.width)} px · area ${n0(g.area)} px²`)}<canvas id="cropCv" width="280" height="280"></canvas>
      <div class="predline"><span>Model says</span> <span class="tag cat" style="background:${colorOf(sel.label)}">${esc(sel.label)}</span> <b>${pct0(sel.confidence)}</b>
      ${sel.confidence < thr ? `<span class="tag Required">check by eye</span>` : ""}
      <div class="small muted" style="margin-top:3px">${Object.entries(sel.probs).slice(0, 3).map(([k, v]) => `${esc(k)} ${pct0(v)}`).join(" · ")}</div></div>`;
  } else insp = `${head("Grain inspector")}<div class="small muted">Click a grain on the photo to see it close up with the model's confidence for each type.</div>`;
  el.innerHTML = `<div class="card card-pad">${head("Result", `${tot} grains graded`)}
      <div class="table-wrap table-plain"><table class="qtable"><tbody>${counts.map((x) => `<tr><td class="pname"><i class="sw" style="background:${colorOf(x.cat)}"></i><b>${esc(x.cat)}</b></td><td class="num"><b>${x.n}</b></td><td class="num">${tot ? ((x.n / tot) * 100).toFixed(1) : "0.0"}%</td></tr>`).join("")}</tbody></table></div>
      <div class="small muted" style="margin-top:8px">${low ? `${low} grain${low > 1 ? "s" : ""} below ${pct0(thr)} confidence (dashed, *). Check ${low > 1 ? "them" : "it"} by eye.` : `All grains are above the ${pct0(thr)} confidence threshold.`}</div>
      <div style="margin-top:12px"><button class="btn primary" id="goQuality">Open the full quality report</button></div></div>
    <div class="card card-pad" id="inspector">${insp}</div>`;
  const gq = $("#goQuality"); if (gq) gq.onclick = () => setTab("quality");
  if (sel && g) { const im = getCropImg(curFile().sid); if (im.complete && im.naturalWidth) drawCrop(g); }
}
function drawCrop(g) {
  const cv = $("#cropCv"), im = cropImgs[curFile().sid];
  if (!cv || !im || !im.naturalWidth || !S.sel || S.sel !== g.id) return;
  const k = im.naturalWidth / S.det.width, cx = g.x + g.box_w / 2, cy = g.y + g.box_h / 2;
  const side = Math.max(g.box_w, g.box_h) * 1.9 + 16, ctx = cv.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#e5e7eb"; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.drawImage(im, (cx - side / 2) * k, (cy - side / 2) * k, side * k, side * k, 0, 0, cv.width, cv.height);
  const s = cv.width / side;                                              // outline of the grain's box
  ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
  ctx.strokeRect((g.x - (cx - side / 2)) * s - 2, (g.y - (cy - side / 2)) * s - 2, g.box_w * s + 4, g.box_h * s + 4);
}
function refreshPredict() {
  const svg = $("#pSvg"); if (S.tab !== "predict" || !svg) return;
  fillOverlay(svg, predictStyle(), S.sel, S.showIds);
  fillPredictSide();
}
async function reloadModel() {
  busy(true);
  try {
    const before = modelVer();
    S.model = await api("/api/model/reload", {});
    S.pred = {};
    const m = S.model.model;
    if (m.trained) toast(m.version === before ? "Model reloaded (no newer model found)." : `Model loaded: trained ${m.trained_at}.`, true);
    else toast(m.error || "No model file found yet. Run  python train.py  in the terminal first.");
  } catch (e) { toast(e.message); }
  finally { busy(false); }
  render();
}
function bindPredict() {
  const rb = $("#reloadModel"); if (rb) rb.onclick = reloadModel;
  if (!$("#pSvg")) return;
  $("#pSvg").onclick = (e) => { const r = e.target.closest("[data-id]"); if (r) { S.sel = +r.dataset.id; refreshPredict(); } };
  $("#pIds").onchange = (e) => { S.showIds = e.target.checked; refreshPredict(); };
  $$("#pZoom button").forEach((b) => (b.onclick = () => { S.zoom = +b.dataset.z; $$("#pZoom button").forEach((x) => x.classList.toggle("on", x === b)); $("#pStage").style.width = S.zoom * 100 + "%"; }));
  refreshPredict();
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
  const clsOk = S.cls && S.cls.state === "ok", aiOk = S.ai && S.ai.ok && S.aiKey === detKey(), qOk = !!qualityData();
  return `
  <div class="card card-pad">${head("Detection files")}<div class="dl-grid">
    ${dlBtn("Processed image (PNG)", "processed")}${dlBtn("Segmentation mask (PNG)", "mask")}${dlBtn("Grain data (CSV)", "csv")}</div></div>
  <div class="card card-pad">${head("Quality table", qOk ? "Grain types as parameters, for the current photo." : "Train a model in the terminal (python train.py) or run the Groq classification (Quality tab) to enable these downloads.")}<div class="dl-grid">
    ${qOk ? `<button class="btn" id="exQ1">${dlIcon} Quality table (CSV)</button><button class="btn" id="exQ2">${dlIcon} Per-grain quality (CSV)</button>` : `<span class="btn" style="opacity:.45;cursor:not-allowed">${dlIcon} Quality table (CSV)</span><span class="btn" style="opacity:.45;cursor:not-allowed">${dlIcon} Per-grain quality (CSV)</span>`}</div></div>
  <div class="card card-pad">${head("Groq classification report", clsOk ? "" : "Run “Classify grains with Groq” in the Quality tab (Groq source) to enable these downloads.")}<div class="dl-grid">
    ${dlBtn("Classification report (PDF)", "pdf", clsOk)}${dlBtn("Per-grain results (CSV)", "cls_csv", clsOk)}${dlBtn("Category summary (CSV)", "cls_summary", clsOk)}</div></div>
  <div class="card card-pad">${head("AI report", aiOk ? "" : "Generate an AI report in the AI report tab to enable these downloads.")}<div class="dl-grid">
    ${dlBtn("Report (Markdown)", "md", aiOk)}${dlBtn("Report (Text)", "txt", aiOk)}</div></div>`;
}

function bindExport() {
  const a = $("#exQ1"), b = $("#exQ2");
  if (a) a.onclick = qualityCsv;
  if (b) b.onclick = qualityGrainCsv;
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
  const views = { overview: viewOverview, detection: viewDetection, predict: viewPredict, quality: viewQuality, ai: viewAI, export: viewExport, batch: viewBatch };
  page.innerHTML = drop + views[S.tab]();
  $("#dzBtn").onclick = () => $("#fileInput").click();
  const binds = { overview: bindOverview, detection: bindDetection, predict: bindPredict, quality: bindQuality, export: bindExport, batch: bindBatch };
  if (binds[S.tab]) binds[S.tab]();
  if (S.tab === "ai" && $("#aiBtn")) $("#aiBtn").onclick = runAI;
  $$(".zoomable img", page).forEach((im) => (im.onclick = () => zoom(im.src)));
  if (S.tab === "batch") loadBatch();
  autoPred();
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
async function refreshModel() {                       // did train.py save a new model since the page loaded?
  try {
    const before = modelVer();
    S.model = await api("/api/model");
    if (modelVer() !== before) S.pred = {};
  } catch { /* keep the previous state */ }
}
async function setTab(t) {
  S.tab = t;
  $("#sidebar").classList.remove("open");
  if (["predict", "quality", "overview", "export"].includes(t)) await refreshModel();
  render();
  if (S.det && ["overview", "quality", "export"].includes(t)) { await refreshCls(true); if (S.tab === t) render(); }
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
  S.model = S.cfg.model;
  S.qShown = [...S.cfg.classes, "Unclassified"];
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
