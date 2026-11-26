"""
Wheat Grain Analysis Dashboard  -  Streamlit front end.

Run with:   streamlit run app.py

Detection, counting, boxes and measurements are computed locally with OpenCV.
Grain classification and the written report are OPTIONAL and use the Groq Vision API
only when a key is present in .env and you click the button.
"""
from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from ui_theme import (PRIMARY, callout, header, inject_css, kpi_row, legend, section, stretch)
from wheat_detector import (PRESETS, Params, annotate, detect, encode_png, grains_to_csv,
                            resize_for_processing)

try:                       # optional AI modules: detection must work without them
    import groq_analyzer as gz
except Exception:
    gz = None
try:
    import grain_report as gr
except Exception:
    gr = None

st.set_page_config(page_title="Wheat Grain Analysis Dashboard", page_icon="🌾", layout="wide",
                   initial_sidebar_state="expanded")

ACCEPTED = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]


# =========================================================================== #
# Helpers
# =========================================================================== #
def load_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR array, honouring EXIF rotation."""
    import io
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def grain_frame(grains, scale: float) -> pd.DataFrame:
    """One row per grain, measured in pixels of the ORIGINAL image."""
    rows = []
    for i, g in enumerate(grains, 1):
        x, y, w, h = [int(round(v * scale)) for v in g.box]
        rows.append({"id": i, "x": x, "y": y, "box_w": w, "box_h": h,
                     "length": max(g.rect[1]) * scale, "width": min(g.rect[1]) * scale,
                     "aspect": g.aspect, "solidity": g.solidity, "area": g.area * scale * scale})
    df = pd.DataFrame(rows, columns=["id", "x", "y", "box_w", "box_h", "length", "width",
                                     "aspect", "solidity", "area"])
    if len(df):
        med = df["area"].median()
        df["size_flag"] = np.where(df["area"] < 0.65 * med, "Small",
                                   np.where(df["area"] > 1.5 * med, "Large", "Normal"))
    else:
        df["size_flag"] = []
    return df


def hist_chart(df: pd.DataFrame, col: str, label: str):
    import altair as alt
    bars = alt.Chart(df).mark_bar(color=PRIMARY, opacity=0.9, cornerRadiusTopLeft=3,
                                  cornerRadiusTopRight=3).encode(
        x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=20), title=label),
        y=alt.Y("count()", title="Grains"),
        tooltip=[alt.Tooltip("count()", title="Grains")])
    rule = alt.Chart(pd.DataFrame({"m": [float(df[col].median())]})).mark_rule(
        color="#1F2937", strokeDash=[5, 4], size=2).encode(x="m:Q")
    return (bars + rule).properties(height=280)


@st.cache_data(show_spinner=False, max_entries=64)
def batch_stats(data: bytes, params_key: str, _params: Params) -> dict:
    """Detect one image for the batch table (cached per image + settings)."""
    original = load_image(data)
    work, scale = resize_for_processing(original)
    res = detect(work, _params)
    df = grain_frame(res.grains, scale)
    n = len(df)
    return {"grains": n, "avg_length": float(df["length"].mean()) if n else 0.0,
            "avg_width": float(df["width"].mean()) if n else 0.0,
            "variation": float(df["area"].std(ddof=0) / df["area"].mean() * 100) if n > 1 else 0.0,
            "rejected": int(res.rejected), "size": f"{original.shape[1]}x{original.shape[0]}"}


# =========================================================================== #
# Sidebar
# =========================================================================== #
def sidebar():
    sb = st.sidebar
    sb.markdown("## ⚙️ Detection settings")
    preset = sb.selectbox("Colour preset", list(PRESETS), key="cfg_preset",
                          help="Starting point for the HSV colour range. Changing it resets the sliders.")
    ph = PRESETS[preset]

    with sb.expander("🎨 Colour range (HSV)", expanded=True):
        h = st.slider("Hue", 0, 179, ph["h"], key=f"cfg_h_{preset}")
        s = st.slider("Saturation", 0, 255, ph["s"], key=f"cfg_s_{preset}",
                      help="Raise the minimum to ignore white/grey paper, black text and shadows.")
        v = st.slider("Value (brightness)", 0, 255, ph["v"], key=f"cfg_v_{preset}")
        auto_s = st.checkbox("Auto-tune min saturation (Otsu)", value=False, key="cfg_auto",
                             help="Picks the saturation threshold automatically inside the hue band.")

    with sb.expander("🧹 Mask clean-up"):
        blur = st.slider("Blur kernel", 1, 15, 5, step=2, key="cfg_blur")
        open_it = st.slider("Remove specks (opening)", 0, 5, 1, key="cfg_open")
        close_it = st.slider("Fill gaps (closing)", 0, 5, 1, key="cfg_close")

    with sb.expander("📐 Shape filters", expanded=True):
        min_area, max_area = st.slider("Grain area (px)", 20, 100000, (250, 30000), step=10, key="cfg_area",
                                       help="Measured on the processed image (long side <= 1600 px).")
        min_asp, max_asp = st.slider("Aspect ratio (length / width)", 1.0, 8.0, (1.1, 4.5), step=0.05,
                                     key="cfg_asp")
        min_sol = st.slider("Min solidity", 0.3, 1.0, 0.80, step=0.01, key="cfg_sol",
                            help="Rejects ragged shapes such as letters and digits.")
        rel_min = st.slider("Min size vs median grain (%)", 0, 80, 35, key="cfg_rel",
                            help="Drops fragments much smaller than a typical grain. 0 disables.")

    with sb.expander("🔗 Touching grains"):
        split = st.checkbox("Split touching grains (watershed)", value=True, key="cfg_split")
        level = st.slider("Split sensitivity", 0.3, 0.9, 0.5, step=0.05, key="cfg_level",
                          help="Lower = splits more aggressively, higher = splits less.")

    with sb.expander("🖼️ Display"):
        ids = st.checkbox("Number the grains", value=False, key="cfg_ids")
        tilted = st.checkbox("Tilted boxes (rotated rectangles)", value=False, key="cfg_tilt")

    sb.markdown("## 🔬 Classification")
    thr = sb.slider("Manual-inspection threshold", 0.0, 1.0, 0.70, step=0.05, key="cfg_thr",
                    help="Grains classified with confidence below this are flagged for manual inspection. "
                         "Changing it updates results instantly; no new Groq calls.")
    batch = sb.slider("Grains per Groq request", 1, 5, 1, key="cfg_batch",
                      help="1 = one grain per request (most faithful, most API calls). Higher = fewer calls; "
                           "each grain still gets its own crop and answer.")

    sb.divider()
    if sb.button("↺ Reset all settings", **stretch(st.button)):
        for k in [k for k in st.session_state if str(k).startswith("cfg_")]:
            del st.session_state[k]
        st.rerun()
    sb.caption("Tip: if grains are missing, check the *Detection* tab mask first, then widen Hue / lower "
               "minimum Saturation.")

    params = Params(h_min=h[0], h_max=h[1], s_min=s[0], s_max=s[1], v_min=v[0], v_max=v[1],
                    auto_saturation=auto_s, blur=blur, open_iter=open_it, close_iter=close_it,
                    min_area=min_area, max_area=max_area, min_aspect=min_asp, max_aspect=max_asp,
                    min_solidity=min_sol, rel_min_pct=rel_min, split=split, split_level=level)
    return params, ids, tilted, thr, batch


# =========================================================================== #
# Landing page
# =========================================================================== #
def landing():
    c1, c2, c3 = st.columns(3)
    steps = [("1", "Upload", "Drop one or more photos (JPG, PNG, JPEG, ...). Several images can be compared in the Batch tab."),
             ("2", "Detect & tune", "Grains are found locally with OpenCV. Adjust colour range and filters until the count looks right."),
             ("3", "Analyse & export", "Review size statistics, optionally grade each grain with Groq, then download images, CSVs and a PDF.")]
    for col, (n, t, d) in zip((c1, c2, c3), steps):
        col.markdown(f'<div class="w-step"><div class="n">{n}</div><h4>{t}</h4><p>{d}</p></div>',
                     unsafe_allow_html=True)
    section("Tips for the best results")
    a, b = st.columns(2)
    a.markdown("- Spread grains out on a **plain, contrasting** background\n"
               "- Use **even lighting**; avoid strong shadows and glare\n"
               "- Shoot from directly above, with the grains in focus")
    b.markdown("- Text, numbers and labels are ignored automatically (low colour saturation)\n"
               "- Touching grains are split with a watershed step\n"
               "- Heavily stacked grains still need manual slider tuning")


# =========================================================================== #
# Tabs
# =========================================================================== #
def tab_overview(original, annotated, res, df, scale, elapsed):
    n = len(df)
    h_, w_ = original.shape[:2]
    cv = float(df["area"].std(ddof=0) / df["area"].mean() * 100) if n > 1 else 0.0
    var_tone = "" if n < 2 else ("good" if cv < 15 else "neutral" if cv < 30 else "warn")
    kpi_row([
        ("Grains detected", n, f"{res.rejected} blobs filtered out", "good" if n else "bad"),
        ("Avg length", f"{df['length'].mean():.0f} px" if n else "-", f"median {df['length'].median():.0f} px" if n else ""),
        ("Avg width", f"{df['width'].mean():.0f} px" if n else "-", f"median {df['width'].median():.0f} px" if n else ""),
        ("Size variation", f"{cv:.0f}%" if n > 1 else "-", "area CV, lower = more uniform", var_tone),
        ("Processing", f"{elapsed:.2f} s", f"{w_} x {h_} px image", "neutral"),
    ])
    if n == 0:
        callout("warn", "No grains found. Try another colour preset, widen the Hue/Saturation range, or relax "
                        "the shape filters (see the Detection tab for the mask).")

    section("Image preview")
    view = st.radio("View", ["Side by side", "Processed", "Original", "Mask"], horizontal=True,
                    label_visibility="collapsed", key="view_mode")
    if view == "Side by side":
        l, r = st.columns(2)
        l.markdown("**Original**")
        l.image(rgb(original), **stretch(st.image))
        r.markdown(f"**Processed** · {n} grains")
        r.image(rgb(annotated), **stretch(st.image))
    elif view == "Processed":
        st.image(rgb(annotated), **stretch(st.image))
    elif view == "Original":
        st.image(rgb(original), **stretch(st.image))
    else:
        st.image(res.mask, caption="White = pixels inside the HSV range after clean-up.", **stretch(st.image))

    if n:
        section("Grain size distribution", "Dashed line = median.")
        c1, c2 = st.columns([2, 1])
        with c1:
            metric = st.selectbox("Measure", ["Length (px)", "Width (px)", "Area (px²)"], key="hist_metric",
                                  label_visibility="collapsed")
            col = {"Length (px)": "length", "Width (px)": "width", "Area (px²)": "area"}[metric]
            st.altair_chart(hist_chart(df, col, metric), **stretch(st.altair_chart))
        with c2:
            counts = df["size_flag"].value_counts()
            summary = pd.DataFrame({
                "Metric": ["Smallest length", "Median length", "Largest length", "Avg aspect ratio",
                           "Avg solidity", "Small grains (<65% median area)", "Large grains (>150%)"],
                "Value": [f"{df['length'].min():.0f} px", f"{df['length'].median():.0f} px",
                          f"{df['length'].max():.0f} px", f"{df['aspect'].mean():.2f}",
                          f"{df['solidity'].mean():.2f}", str(int(counts.get("Small", 0))),
                          str(int(counts.get("Large", 0)))]})
            st.markdown("**Summary statistics**")
            st.dataframe(summary, hide_index=True, **stretch(st.dataframe))


def tab_detection(res, df, params, scale):
    section("Segmentation mask", "White pixels passed the colour range. If grains are missing here, adjust the "
                                 "HSV sliders in the sidebar.")
    c1, c2 = st.columns([3, 2])
    c1.image(res.mask, **stretch(st.image))
    with c2:
        st.markdown("**Active settings**")
        st.dataframe(pd.DataFrame({
            "Setting": ["Hue", "Saturation", "Value", "Area (px)", "Aspect ratio", "Min solidity",
                        "Min size vs median", "Split touching", "Split sensitivity"],
            "Value": [f"{params.h_min} - {params.h_max}", f"{params.s_min} - {params.s_max}"
                      + (" (auto min)" if params.auto_saturation else ""), f"{params.v_min} - {params.v_max}",
                      f"{params.min_area} - {params.max_area}", f"{params.min_aspect} - {params.max_aspect}",
                      str(params.min_solidity), f"{params.rel_min_pct}%", "yes" if params.split else "no",
                      str(params.split_level)]}), hide_index=True, **stretch(st.dataframe))

    section("Per-grain measurements", "Pixels of the original image. Click a column header to sort.")
    if df.empty:
        st.caption("No grains to list.")
        return
    f1, f2 = st.columns([1, 3])
    only_out = f1.checkbox("Only size outliers", value=False, key="only_outliers")
    show = df[df["size_flag"] != "Normal"] if only_out else df
    disp = show.rename(columns={"id": "ID", "x": "X", "y": "Y", "box_w": "Box W", "box_h": "Box H",
                                "length": "Length (px)", "width": "Width (px)", "aspect": "Aspect",
                                "solidity": "Solidity", "area": "Area (px²)", "size_flag": "Size"})
    st.dataframe(disp, hide_index=True, **stretch(st.dataframe), column_config={
        "Length (px)": st.column_config.NumberColumn(format="%.1f"),
        "Width (px)": st.column_config.NumberColumn(format="%.1f"),
        "Aspect": st.column_config.NumberColumn(format="%.2f"),
        "Solidity": st.column_config.NumberColumn(format="%.2f"),
        "Area (px²)": st.column_config.NumberColumn(format="%.0f")})
    f2.caption(f"Showing {len(show)} of {len(df)} grains.")


def _classification_context(sig, n, threshold, uploaded, original, res, scale, tilted):
    """Return (context, state). state is one of none / stale / error / ok."""
    saved = st.session_state.get("cls_result")
    if not saved:
        return None, "none"
    if saved["sig"] != sig:
        return None, "stale"
    if saved.get("error"):
        return saved, "error"
    run, meas = saved["run"], saved["meas"]
    if len(meas) != n:
        return None, "stale"
    rows = gr.build_rows(meas, run.results, threshold)
    summary = gr.summarize(rows)
    meta = {"image": uploaded.name, "model": run.model, "batch": run.batch_size,
            "threshold": threshold, "generated": saved["time"]}
    overview = gr.annotate_classified(original, res.grains, scale, rows, tilted)
    return {"run": run, "rows": rows, "summary": summary, "meta": meta, "overview": overview}, "ok"


def _category_frame(summary):
    rows = [{"Category": c["category"], "Count": c["count"], "% of classified": f"{c['percent']:.1f}%",
             "Flagged for manual inspection": c["flagged"]} for c in summary["categories"]]
    rows.append({"Category": "Total classified", "Count": summary["classified"],
                 "% of classified": "100.0%" if summary["classified"] else "0.0%",
                 "Flagged for manual inspection": summary["flagged_total"] - summary["flagged_unclassified"]})
    if summary["unclassified"]:
        rows.append({"Category": "Unclassified (excluded from %)", "Count": summary["unclassified"],
                     "% of classified": "-", "Flagged for manual inspection": summary["flagged_unclassified"]})
    return pd.DataFrame(rows)


def _draw_charts(summary):
    if summary["classified"] == 0:
        st.caption("No classified grains to chart.")
        return
    df = pd.DataFrame([{"Category": c["category"], "Count": c["count"], "Percent": c["percent"]}
                       for c in summary["categories"]])
    names = list(df["Category"])
    colours = [gr.CATEGORY_COLORS[n] for n in names]
    try:
        import altair as alt
        scale = alt.Scale(domain=names, range=colours)
        tip = ["Category", "Count", alt.Tooltip("Percent:Q", format=".1f", title="% of classified")]
        c1, c2 = st.columns(2)
        c1.markdown("**Grains per category**")
        c1.altair_chart(alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Category:N", sort=names, title=None, axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("Count:Q", title="Grains"),
            color=alt.Color("Category:N", scale=scale, legend=None), tooltip=tip), **stretch(st.altair_chart))
        c2.markdown("**Share of classified grains**")
        c2.altair_chart(alt.Chart(df[df["Count"] > 0]).mark_arc(innerRadius=55).encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Category:N", scale=scale, legend=alt.Legend(title=None)), tooltip=tip),
            **stretch(st.altair_chart))
    except Exception:                                    # charts are a convenience only
        st.bar_chart(df.set_index("Category")["Count"])


def tab_classification(cfg, original, work, scale, res, tilted, uploaded, params, threshold, batch):
    """Five-category Groq classification. Returns the context (or None) for the export tab."""
    section("Grain quality classification",
            "OpenCV detects, counts, boxes and measures every grain. Groq then looks at each grain's own crop and "
            "assigns one category: Sound, Shrivelled, Immature, Broken or Damaged/Discoloured. Uncertain or failed "
            "grains are flagged for manual inspection, never guessed.")
    n = len(res.grains)
    sig = f"{uploaded.name}|{uploaded.size}|{params}"

    if gz is None or gr is None or cfg is None:
        callout("info", "Grain classification is unavailable in this installation.")
        return None
    if not cfg.configured:
        callout("info", "Optional feature: add your Groq API key to the .env file (see .env.example) and reload "
                        "the page to enable it. Everything else works without it.")
    if n == 0:
        st.caption("No grains detected yet, so there is nothing to classify.")
    else:
        calls = -(-n // batch)
        st.caption(f"{n} grains will be sent in {calls} request{'s' if calls != 1 else ''} "
                   f"({batch} per request). Clicking the button uploads the grain crops to Groq. "
                   "Free-tier rate limits may slow this down.")

    if st.button("Classify grains with Groq", type="primary", disabled=(not cfg.configured or n == 0),
                 key="btn_classify"):
        bar = st.progress(0.0, text="Starting classification...")
        try:
            meas = gz.measure_grains(work, res.grains, scale)

            def on_progress(done, total):
                bar.progress(min(1.0, done / max(total, 1)), text=f"Classified {done} of {total} grains")

            run = gz.classify_grains(original, res.grains, meas, scale, cfg, batch, on_progress)
            st.session_state["cls_result"] = {"sig": sig, "meas": meas, "run": run, "error": "",
                                              "time": datetime.now().strftime("%Y-%m-%d %H:%M")}
        except Exception as e:          # never let classification break the detection page
            st.session_state["cls_result"] = {"sig": sig, "error": f"Classification failed: {type(e).__name__}"}
        finally:
            bar.empty()

    ctx, state = _classification_context(sig, n, threshold, uploaded, original, res, scale, tilted)
    if state == "none":
        return None
    if state == "stale":
        callout("warn", "These classification results belong to a different image or detection settings, so the "
                        "grain IDs no longer match. Click 'Classify grains with Groq' to classify the current detection.")
        return None
    if state == "error":
        callout("bad", ctx["error"])
        st.caption("Detection results are unaffected.")
        return None

    run, rows, summary = ctx["run"], ctx["rows"], ctx["summary"]
    if run.fatal_error:
        callout("bad", run.fatal_error)
        if run.hint:
            st.caption(run.hint)

    kpi_row([("Grains detected", summary["total_detected"], "", "neutral"),
             ("Classified", summary["classified"], "", "good"),
             ("Not classified", summary["unclassified"], "", "bad" if summary["unclassified"] else "neutral"),
             ("Manual inspection", summary["flagged_total"], f"threshold {threshold:.0%}",
              "warn" if summary["flagged_total"] else "good")])
    callout("ok" if summary["reconciled"] else "bad",
            ("Counts reconcile: " if summary["reconciled"] else "Counts do not reconcile: ")
            + gr.reconciliation_text(summary))

    t_sum, t_grain, t_obs, t_rep = st.tabs(["Summary & charts", "Per-grain results",
                                             "Observations & recommendations", "Detailed report"])
    with t_sum:
        st.dataframe(_category_frame(summary), hide_index=True, **stretch(st.dataframe))
        st.caption("Percentages are shares of classified grains. Unclassified grains are never assigned a category.")
        _draw_charts(summary)
        st.markdown("**Classified overview**  (box colour = category, `*` after the ID = manual inspection)")
        st.image(rgb(ctx["overview"]), **stretch(st.image))
        legend(gr.CATEGORY_COLORS)
    with t_grain:
        f1, f2 = st.columns([3, 1])
        shown = f1.multiselect("Categories shown", list(gr.CATEGORY_COLORS), default=list(gr.CATEGORY_COLORS))
        only_flagged = f2.checkbox("Only manual inspection", value=False, key="cls_only_flagged")
        df = pd.DataFrame([{
            "ID": r["id"], "Category": r["category"],
            "Confidence": None if r["confidence"] is None else round(r["confidence"] * 100),
            "Manual inspection": r["manual_inspection"], "Why / notes": r["review_reasons"],
            "Model's visual reason": r["model_reason"], "Length (px)": r["length_px"],
            "Width (px)": r["width_px"], "Aspect": r["aspect_ratio"], "Solidity": r["solidity"],
            "Area (px)": r["area_px"]} for r in rows
            if r["category"] in shown and (not only_flagged or r["manual_inspection"] == "Required")])
        st.dataframe(df, hide_index=True, **stretch(st.dataframe), column_config={
            "Confidence": st.column_config.NumberColumn("Confidence", format="%d%%",
                                                        help="Reported by the model; not a calibrated probability.")})
        st.caption(f"Showing {len(df)} of {len(rows)} grains. Measurements are in pixels (no physical calibration).")
    with t_obs:
        st.markdown("**Observations**")
        for o in gr.observations(summary):
            st.markdown(f"- {o}")
        st.markdown("**Recommendations**")
        for r in gr.recommendations(summary):
            st.markdown(f"- {r}")
    with t_rep:
        st.markdown(gr.build_report_markdown(summary, rows, ctx["meta"]))
    st.caption("PDF and CSV downloads for these results are in the Export tab.")
    return ctx


def tab_ai(cfg, original, work, scale, res, tilted, uploaded, params):
    """Optional Groq written report. OpenCV results are never changed by anything in here."""
    section("AI wheat analysis",
            "Groq describes grain appearance, colour, shape and possible damage. OpenCV still does all detection "
            "and counting. Clicking the button sends the numbered overview image and a grid of grain close-ups to Groq.")
    n = len(res.grains)
    sig = f"{uploaded.name}|{uploaded.size}|{params}|{tilted}"
    if gz is None or cfg is None:
        callout("info", "AI analysis is unavailable in this installation.")
        return
    if not cfg.configured:
        callout("info", "Optional feature: add your Groq API key to the .env file to enable this section, "
                        "then reload the page.")
    if n == 0:
        st.caption("No grains detected yet, so there is nothing to analyse.")

    if st.button("Generate AI report", type="primary", disabled=(not cfg.configured or n == 0), key="btn_ai"):
        try:
            with st.spinner("Asking Groq to analyse the grains..."):
                stats = gz.summarize_grains(work, res.grains, scale)
                numbered = annotate(original, res.grains, scale, True, tilted)
                montage = gz.make_montage(original, res.grains, scale, [s["id"] for s in stats["suspects"]])
                out = gz.analyze(numbered, montage, stats, cfg)
            st.session_state["groq_result"] = {
                "sig": sig, "ok": out.ok, "error": out.error, "hint": out.hint, "report": out.report,
                "doc": gz.build_report_document(out.report, stats, out.model, uploaded.name) if out.ok else ""}
        except Exception as e:
            st.session_state["groq_result"] = {"sig": sig, "ok": False, "report": "", "doc": "",
                                               "error": f"AI analysis failed: {type(e).__name__}", "hint": ""}

    saved = st.session_state.get("groq_result")
    if not saved:
        return
    if saved["sig"] != sig:
        callout("warn", "This report was generated for a different image or detection settings. "
                        "Click 'Generate AI report' to refresh it.")
    if saved["ok"]:
        with st.container(border=True):
            st.markdown(saved["report"])
    else:
        callout("bad", saved["error"])
        if saved["hint"]:
            st.caption(saved["hint"])
        st.caption("Detection results are unaffected.")


def tab_export(annotated, res, scale, uploaded, cls_ctx, params, tilted):
    stem = uploaded.name.rsplit(".", 1)[0]
    section("Detection files")
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ Processed image (PNG)", encode_png(annotated), file_name=f"{stem}_detected.png",
                       mime="image/png", key="dl_img", **stretch(st.download_button))
    c2.download_button("⬇ Segmentation mask (PNG)", encode_png(res.mask), file_name=f"{stem}_mask.png",
                       mime="image/png", key="dl_mask", **stretch(st.download_button))
    c3.download_button("⬇ Grain data (CSV)", grains_to_csv(res.grains, scale), file_name=f"{stem}_grains.csv",
                       mime="text/csv", key="dl_csv", **stretch(st.download_button))

    section("Classification results")
    if cls_ctx is None:
        st.caption("Run 'Classify grains with Groq' in the Quality classification tab to enable these downloads.")
    else:
        d1, d2, d3 = st.columns(3)
        try:
            d1.download_button("⬇ Classification report (PDF)",
                               gr.build_pdf(cls_ctx["summary"], cls_ctx["rows"], cls_ctx["overview"], cls_ctx["meta"]),
                               file_name=f"{stem}_classification_report.pdf", mime="application/pdf",
                               key="dl_pdf", **stretch(st.download_button))
        except ImportError:
            d1.warning("PDF export needs reportlab: pip install reportlab")
        except Exception as e:
            d1.warning(f"PDF could not be created ({type(e).__name__}).")
        d2.download_button("⬇ Per-grain results (CSV)", gr.rows_to_csv(cls_ctx["rows"]),
                           file_name=f"{stem}_grain_classification.csv", mime="text/csv", key="dl_cls_rows",
                           **stretch(st.download_button))
        d3.download_button("⬇ Category summary (CSV)", gr.summary_to_csv(cls_ctx["summary"]),
                           file_name=f"{stem}_category_summary.csv", mime="text/csv", key="dl_cls_sum",
                           **stretch(st.download_button))

    section("AI report")
    saved = st.session_state.get("groq_result")
    sig = f"{uploaded.name}|{uploaded.size}|{params}|{tilted}"
    if not saved or not saved.get("ok") or saved["sig"] != sig:
        st.caption("Generate an AI report in the AI report tab to enable these downloads.")
    else:
        e1, e2, _ = st.columns(3)
        e1.download_button("⬇ Report (Markdown)", saved["doc"], file_name=f"{stem}_analysis.md",
                           mime="text/markdown", key="dl_md", **stretch(st.download_button))
        e2.download_button("⬇ Report (Text)", saved["doc"], file_name=f"{stem}_analysis.txt",
                           mime="text/plain", key="dl_txt", **stretch(st.download_button))


def tab_batch(uploads, params):
    section("Batch comparison", "All uploaded images processed with the current detection settings.")
    rows, bar = [], st.progress(0.0, text="Processing images...")
    for i, f in enumerate(uploads):
        try:
            s = batch_stats(f.getvalue(), repr(params), params)
            rows.append({"Image": f.name, "Grains": s["grains"], "Avg length (px)": round(s["avg_length"], 1),
                         "Avg width (px)": round(s["avg_width"], 1), "Size variation (%)": round(s["variation"], 1),
                         "Filtered blobs": s["rejected"], "Resolution": s["size"]})
        except Exception as e:
            rows.append({"Image": f.name, "Grains": None, "Resolution": f"error: {type(e).__name__}"})
        bar.progress((i + 1) / len(uploads), text=f"Processed {i + 1} of {len(uploads)}")
    bar.empty()
    df = pd.DataFrame(rows)
    ok = df.dropna(subset=["Grains"])
    if len(ok):
        kpi_row([("Images", len(df), "", "neutral"), ("Total grains", int(ok["Grains"].sum()), "", "good"),
                 ("Mean grains / image", f"{ok['Grains'].mean():.0f}", "", ""),
                 ("Range", f"{int(ok['Grains'].min())} - {int(ok['Grains'].max())}", "grains per image", "neutral")])
        st.bar_chart(ok.set_index("Image")["Grains"], color=PRIMARY)
    st.dataframe(df, hide_index=True, **stretch(st.dataframe))
    st.download_button("⬇ Batch summary (CSV)", df.to_csv(index=False), file_name="wheat_batch_summary.csv",
                       mime="text/csv", key="dl_batch")


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    inject_css()
    cfg = gz.load_config() if gz is not None else None
    header(True, bool(cfg and cfg.configured), cfg.model if cfg else "")
    params, show_ids, tilted, cls_thr, cls_batch = sidebar()

    uploads = st.file_uploader("Upload wheat photos", type=ACCEPTED, accept_multiple_files=True,
                               help="JPG, PNG, JPEG and other common formats. Max 50 MB each.")
    if not uploads:
        landing()
        return

    idx = 0
    if len(uploads) > 1:
        idx = st.selectbox("Image to analyse", range(len(uploads)),
                           format_func=lambda i: f"{i + 1}. {uploads[i].name}")
    uploaded = uploads[idx]

    try:
        original = load_image(uploaded.getvalue())
    except Exception:
        callout("bad", f"Could not read '{uploaded.name}'. Please upload a valid image file.")
        return
    work, scale = resize_for_processing(original)

    t0 = time.perf_counter()
    with st.spinner("Detecting grains..."):
        res = detect(work, params)
        annotated = annotate(original, res.grains, scale, show_ids, tilted)
    elapsed = time.perf_counter() - t0
    df = grain_frame(res.grains, scale)

    names = ["📊 Overview", "🔍 Detection", "🔬 Quality classification", "🧠 AI report", "📦 Export"]
    if len(uploads) > 1:
        names.append("🗂 Batch")
    tabs = st.tabs(names)

    with tabs[0]:
        tab_overview(original, annotated, res, df, scale, elapsed)
    with tabs[1]:
        tab_detection(res, df, params, scale)
    with tabs[2]:
        cls_ctx = tab_classification(cfg, original, work, scale, res, tilted, uploaded, params, cls_thr, cls_batch)
    with tabs[3]:
        tab_ai(cfg, original, work, scale, res, tilted, uploaded, params)
    with tabs[4]:
        tab_export(annotated, res, scale, uploaded, cls_ctx, params, tilted)
    if len(uploads) > 1:
        with tabs[5]:
            tab_batch(uploads, params)

    st.caption("Detection runs locally. Groq is optional and only receives data when you click its buttons.")


if __name__ == "__main__":
    main()
