"""Visual theme and small HTML components for the dashboard (no detection logic here)."""
from __future__ import annotations

import html
import inspect

import streamlit as st

PRIMARY = "#B45309"
GOLD = "#D69E2E"
INK = "#1F2937"
MUTED = "#6B7280"

CSS = """
<style>
:root {
  --w-primary: #B45309; --w-gold: #D69E2E; --w-ink: #1F2937; --w-muted: #6B7280;
  --w-card: #FFFFFF; --w-border: #E7E2D6; --w-ok: #15803D; --w-warn: #B45309; --w-bad: #B91C1C;
}
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }
footer, #MainMenu { visibility: hidden; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }

/* ---------- header ---------- */
.w-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  padding:1.25rem 1.5rem; border-radius:18px; margin-bottom:1.1rem; color:#fff;
  background: linear-gradient(120deg, #2E2412 0%, #5E4514 55%, #A9741B 100%);
  box-shadow: 0 6px 20px rgba(60,40,10,.18); }
.w-brand { display:flex; align-items:center; gap:.95rem; }
.w-logo { width:52px; height:52px; border-radius:14px; display:flex; align-items:center; justify-content:center;
  font-size:28px; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25); }
.w-title { font-size:1.55rem; font-weight:750; line-height:1.15; margin:0; color:#fff; }
.w-sub { font-size:.9rem; opacity:.85; margin-top:.15rem; color:#fff; }
.w-chips { display:flex; gap:.5rem; flex-wrap:wrap; }
.w-chip { font-size:.78rem; font-weight:600; padding:.3rem .7rem; border-radius:999px; white-space:nowrap;
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28); color:#fff; }
.w-chip.ok::before { content:"● "; color:#86EFAC; } .w-chip.off::before { content:"○ "; color:#FDE68A; }

/* ---------- cards / KPIs ---------- */
.w-kpi { background:var(--w-card); border:1px solid var(--w-border); border-radius:14px; padding:.95rem 1.1rem;
  height:100%; box-shadow:0 1px 2px rgba(0,0,0,.03); border-top:3px solid var(--w-gold); }
.w-kpi.good { border-top-color:var(--w-ok); } .w-kpi.warn { border-top-color:#D97706; }
.w-kpi.bad { border-top-color:var(--w-bad); } .w-kpi.neutral { border-top-color:#9CA3AF; }
.w-kpi-label { font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:var(--w-muted); font-weight:650; }
.w-kpi-value { font-size:1.9rem; font-weight:760; color:var(--w-ink); line-height:1.15; margin-top:.2rem; }
.w-kpi-sub { font-size:.78rem; color:var(--w-muted); margin-top:.15rem; }

.w-section { margin:1.2rem 0 .5rem; }
.w-section h3 { margin:0; font-size:1.12rem; color:var(--w-ink); }
.w-section p { margin:.1rem 0 0; font-size:.86rem; color:var(--w-muted); }

.w-callout { border-radius:12px; padding:.8rem 1rem; margin:.5rem 0; font-size:.9rem; border:1px solid; }
.w-callout.info { background:#FFF8E8; border-color:#F3DDA8; color:#5B4410; }
.w-callout.ok { background:#ECFDF3; border-color:#B7EBC6; color:#14532D; }
.w-callout.warn { background:#FFF4E5; border-color:#FBD38D; color:#7B341E; }
.w-callout.bad { background:#FEF2F2; border-color:#FECACA; color:#7F1D1D; }

.w-step { background:var(--w-card); border:1px solid var(--w-border); border-radius:16px; padding:1.2rem 1.3rem; height:100%; }
.w-step .n { width:30px; height:30px; border-radius:50%; background:var(--w-primary); color:#fff; font-weight:700;
  display:flex; align-items:center; justify-content:center; margin-bottom:.6rem; }
.w-step h4 { margin:0 0 .25rem; font-size:1.02rem; } .w-step p { margin:0; font-size:.88rem; color:var(--w-muted); }

.w-legend span { display:inline-block; margin-right:1rem; font-size:.85rem; }

/* ---------- native widgets ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] { background:#FFFFFF; border-radius:14px; border-color:var(--w-border); }
.stTabs [data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid var(--w-border); }
.stTabs [data-baseweb="tab"] { height:2.8rem; padding:0 1rem; font-weight:600; border-radius:10px 10px 0 0; }
.stTabs [aria-selected="true"] { color:var(--w-primary) !important; }
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:600; }
section[data-testid="stSidebar"] { border-right:1px solid var(--w-border); }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-size:1rem; }
div[data-testid="stFileUploader"] section { border-radius:14px; border:2px dashed #D9CFB6; background:#FFFDF8; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def stretch(fn) -> dict:
    """Full-width kwargs: width="stretch" on current Streamlit, use_container_width on older releases."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        if (major, minor) >= (1, 52):
            return {"width": "stretch"}
    except Exception:
        pass
    try:
        if "use_container_width" in inspect.signature(fn).parameters:
            return {"use_container_width": True}
    except (TypeError, ValueError):
        pass
    return {"width": "stretch"}


def header(engine_ok: bool, groq_ok: bool, model: str = "") -> None:
    groq_chip = (f'<span class="w-chip ok">Groq AI · {html.escape(model)}</span>' if groq_ok
                 else '<span class="w-chip off">Groq AI · not configured (optional)</span>')
    st.markdown(f"""
<div class="w-header">
  <div class="w-brand"><div class="w-logo">🌾</div>
    <div><p class="w-title">Wheat Grain Analysis Dashboard</p>
    <div class="w-sub">Detect, count, measure and grade wheat grains from a photo</div></div></div>
  <div class="w-chips">
    <span class="w-chip {'ok' if engine_ok else 'off'}">OpenCV engine · local</span>{groq_chip}
  </div>
</div>""", unsafe_allow_html=True)


def kpi(label: str, value, sub: str = "", tone: str = "") -> str:
    return (f'<div class="w-kpi {tone}"><div class="w-kpi-label">{html.escape(str(label))}</div>'
            f'<div class="w-kpi-value">{html.escape(str(value))}</div>'
            f'<div class="w-kpi-sub">{html.escape(str(sub))}</div></div>')


def kpi_row(items) -> None:
    """items: list of (label, value, sub, tone)."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="w-section"><h3>{html.escape(title)}</h3>{sub}</div>', unsafe_allow_html=True)


def callout(kind: str, text: str) -> None:
    """kind: info | ok | warn | bad. text is plain text (escaped)."""
    st.markdown(f'<div class="w-callout {kind}">{html.escape(text)}</div>', unsafe_allow_html=True)


def legend(colors: dict) -> None:
    st.markdown('<div class="w-legend">' + "".join(
        f"<span><b style='color:{c}'>■</b> {html.escape(n)}</span>" for n, c in colors.items()) + "</div>",
        unsafe_allow_html=True)
