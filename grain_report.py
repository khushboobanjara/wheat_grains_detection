"""
Results, reporting and exports for the five-category grain classification.

Everything here is derived ONLY from (a) the OpenCV measurements and (b) the classifications
returned by Groq (see groq_analyzer.classify_grains). Nothing is estimated or invented:
observations and recommendations are rule-based sentences filled with the actual counts.

Public functions
----------------
build_rows         merge measurements + classifications, apply manual-inspection rules
summarize          category counts / percentages and the reconciliation check
observations       list of factual statements about the summary
recommendations    list of next steps driven by the summary
annotate_classified  overview image with colour-coded boxes and ids
rows_to_csv / summary_to_csv
build_report_markdown  the detailed report shown in the app
build_pdf          the same report as a PDF (needs reportlab)
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

from groq_analyzer import CATEGORIES, MIN_RELIABLE_LENGTH_PX, UNCLASSIFIED

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
FLAG_PREVIEW_LIMIT = 40                     # ids listed inline before "and N more"

CATEGORY_DEFINITIONS = {
    "Sound": "Plump, intact grain with normal uniform colour and no visible defect.",
    "Shrivelled": "Whole grain that is visibly shrunken, wrinkled, thin or flattened.",
    "Immature": "Underdeveloped grain: small, greenish, pale or chalky, poorly filled.",
    "Broken": "Fragmented, cracked, chipped or with a visibly missing part.",
    "Damaged/Discoloured": "Insect holes, sprouting, mould-like patches, heat/weather damage or stained colour.",
}

# One colour per category, shared by the app charts, the overlay image and the PDF (RGB hex).
CATEGORY_COLORS = {
    "Sound": "#2e9e4f",
    "Shrivelled": "#f28c28",
    "Immature": "#e0c000",
    "Broken": "#8e44ad",
    "Damaged/Discoloured": "#d62728",
    UNCLASSIFIED: "#7f7f7f",
}


def _hex_to_bgr(h: str):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


# --------------------------------------------------------------------------- #
# Rows: measurements + classification + manual-inspection status
# --------------------------------------------------------------------------- #
def build_rows(measurements: List[dict], classifications, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
               ) -> List[dict]:
    """One row per detected grain. Manual-inspection rules (any one triggers it):

    * the grain could not be classified (API failure, invalid or missing reply)
    * confidence missing, or below ``threshold``
    * the model itself marked the grain as uncertain
    * the grain touches the image edge (it may be cut off)
    * the grain is very small in the photo (low resolution)
    """
    by_id = {c.id: c for c in classifications}
    rows = []
    for m in measurements:
        c = by_id.get(m["id"])
        reasons = []
        classified = bool(c and c.classified)
        if not classified:
            reasons.append((c.error if c and c.error else "Not classified") + " Classify manually.")
        else:
            if c.confidence is None:
                reasons.append("No valid confidence returned")
            elif c.confidence < threshold:
                reasons.append(f"Confidence {c.confidence:.0%} is below the {threshold:.0%} threshold")
            if c.uncertain:
                reasons.append("Model marked this grain as uncertain")
        if m.get("touches_edge"):
            reasons.append("Grain touches the image edge and may be cut off")
        if m.get("length_px", 0) < MIN_RELIABLE_LENGTH_PX:
            reasons.append(f"Grain is under {MIN_RELIABLE_LENGTH_PX}px long in the photo (low resolution)")
        rows.append({
            "id": m["id"],
            "category": c.category if classified else UNCLASSIFIED,
            "classified": classified,
            "confidence": c.confidence if classified else None,
            "manual_inspection": "Required" if reasons else "Not required",
            "review_reasons": "; ".join(reasons),
            "model_reason": (c.reason if c else ""),
            "touches_edge": bool(m.get("touches_edge")),
            "low_res": m.get("length_px", 0) < MIN_RELIABLE_LENGTH_PX,
            **{k: m.get(k) for k in ("x", "y", "width", "height", "area_px", "length_px", "width_px",
                                     "aspect_ratio", "solidity", "mean_colour_hex", "hue_deg",
                                     "saturation_pct", "brightness_pct")},
        })
    return rows


# --------------------------------------------------------------------------- #
# Counts, percentages, reconciliation
# --------------------------------------------------------------------------- #
def _percentages(counts: List[int], total: int) -> List[float]:
    """Share of ``total`` per count, rounded to 1 decimal (equal counts always give equal percentages)."""
    if total <= 0:
        return [0.0] * len(counts)
    return [round(c * 100.0 / total, 1) for c in counts]


def summarize(rows: List[dict]) -> dict:
    """Category table and the reconciliation check.

    Percentages are shares of the CLASSIFIED grains. Unclassified grains are never assigned
    to a category; they are reported separately so that
        sum(category counts) == classified   and   classified + unclassified == detected.
    """
    total = len(rows)
    counts = [sum(1 for r in rows if r["category"] == c) for c in CATEGORIES]
    classified = sum(1 for r in rows if r["classified"])
    unclassified = total - classified
    pcts = _percentages(counts, classified)
    flagged = [r for r in rows if r["manual_inspection"] == "Required"]
    cats = []
    for c, n, p in zip(CATEGORIES, counts, pcts):
        cats.append({"category": c, "count": n, "percent": p,
                     "flagged": sum(1 for r in flagged if r["category"] == c)})
    confs = [r["confidence"] for r in rows if r["classified"] and r["confidence"] is not None]
    return {
        "total_detected": total,
        "classified": classified,
        "unclassified": unclassified,
        "categories": cats,
        "flagged_total": len(flagged),
        "flagged_unclassified": sum(1 for r in flagged if not r["classified"]),
        "flagged_ids": [r["id"] for r in flagged],
        "edge_count": sum(1 for r in rows if r["touches_edge"]),
        "low_res_count": sum(1 for r in rows if r["low_res"]),
        "mean_confidence": (sum(confs) / len(confs)) if confs else None,
        "reconciled": (sum(counts) == classified and classified + unclassified == total),
    }


def _ids_text(ids: List[int]) -> str:
    shown = ", ".join(str(i) for i in ids[:FLAG_PREVIEW_LIMIT])
    return shown + (f" and {len(ids) - FLAG_PREVIEW_LIMIT} more" if len(ids) > FLAG_PREVIEW_LIMIT else "")


def reconciliation_text(s: dict) -> str:
    parts = " + ".join(f"{c['count']}" for c in s["categories"])
    base = f"{parts} = {s['classified']} classified"
    if s["unclassified"]:
        base += f"; {s['classified']} classified + {s['unclassified']} unclassified = {s['total_detected']} detected"
    else:
        base += f" = {s['total_detected']} detected"
    return base


# --------------------------------------------------------------------------- #
# Observations and recommendations (rule-based, built only from the actual numbers)
# --------------------------------------------------------------------------- #
def observations(s: dict) -> List[str]:
    out = []
    t, k, u = s["total_detected"], s["classified"], s["unclassified"]
    if t == 0:
        return ["No grains were detected, so there is nothing to classify."]
    out.append(f"OpenCV detected {t} grains; {k} were classified by Groq"
               + (f" and {u} could not be classified." if u else "."))
    if k == 0:
        return out
    cats = s["categories"]
    top = max(c["count"] for c in cats)
    leaders = [c for c in cats if c["count"] == top]
    if len(leaders) == 1:
        out.append(f"{leaders[0]['category']} is the largest category: {top} of {k} classified grains "
                   f"({leaders[0]['percent']:.1f}%).")
    else:
        out.append("Largest categories are tied at " + f"{top} grains each: "
                   + ", ".join(c["category"] for c in leaders) + ".")
    sound = next(c for c in cats if c["category"] == "Sound")
    others = [c for c in cats if c["category"] != "Sound" and c["count"] > 0]
    n_other = sum(c["count"] for c in others)
    if others:
        out.append(f"{n_other} grains ({n_other / k * 100:.1f}% of classified) were placed in a category other "
                   "than Sound: " + ", ".join(f"{c['category']} {c['count']} ({c['percent']:.1f}%)" for c in others) + ".")
    elif sound["count"] == k:
        out.append("Every classified grain was labelled Sound.")
    empty = [c["category"] for c in cats if c["count"] == 0]
    if empty and others:
        out.append("No grains were classified as: " + ", ".join(empty) + ".")
    if s["flagged_total"]:
        out.append(f"{s['flagged_total']} of {t} grains ({s['flagged_total'] / t * 100:.1f}%) are flagged for manual "
                   f"inspection (IDs: {_ids_text(s['flagged_ids'])}).")
    else:
        out.append("No grains were flagged for manual inspection.")
    if s["mean_confidence"] is not None:
        out.append(f"Mean model-reported confidence is {s['mean_confidence']:.0%}. This is the model's own "
                   "estimate, not a measured accuracy.")
    if k < 100:
        out.append(f"With {k} classified grains, each grain moves the percentages by {100 / k:.1f} points.")
    return out


def recommendations(s: dict) -> List[str]:
    out = []
    if s["total_detected"] == 0:
        return ["Adjust the detection settings (colour preset, HSV range, shape filters) until grains are "
                "detected, then classify again."]
    if s["classified"] == 0:
        return ["No grain could be classified. Check the Groq key, model and connection shown above, then "
                "classify again. Until then, inspect the grains manually."]
    if s["flagged_total"]:
        out.append("Inspect the grains flagged for manual inspection by hand and correct their category before "
                   "relying on the percentages. They are listed in the per-grain table and the PDF.")
    if s["unclassified"]:
        out.append(f"{s['unclassified']} grains have no category and are excluded from the percentages. "
                   "Run the classification again to retry them, or classify them manually.")
    if s["edge_count"]:
        out.append(f"{s['edge_count']} grains touch the image edge. Re-photograph with every grain fully inside "
                   "the frame so partial grains are not mistaken for broken ones.")
    if s["low_res_count"]:
        out.append(f"{s['low_res_count']} grains are very small in the photo. Move the camera closer or use a "
                   "higher resolution so each grain is at least a few dozen pixels long.")
    non_sound = sum(c["count"] for c in s["categories"] if c["category"] != "Sound")
    if non_sound:
        out.append("Compare the non-Sound percentages with the grain specification that applies to your purpose. "
                   "This tool reports what is visible and does not apply grading limits.")
    dd = next(c for c in s["categories"] if c["category"] == "Damaged/Discoloured")
    if dd["count"]:
        out.append("Confirm Damaged/Discoloured grains by hand. A photo cannot reliably separate insect damage, "
                   "mould, sprouting and staining, and cannot test for moisture, protein or mycotoxins.")
    out.append("If the lot is larger than the photographed sample, repeat with several representative samples "
               "before drawing conclusions about the whole lot.")
    return out


# --------------------------------------------------------------------------- #
# Overview image with colour-coded boxes
# --------------------------------------------------------------------------- #
def annotate_classified(original_bgr: np.ndarray, grains, scale: float, rows: List[dict],
                        tilted: bool = False) -> np.ndarray:
    """Boxes coloured by category, labelled with the grain id ('*' = manual inspection)."""
    out = original_bgr.copy()
    th = max(2, int(round(max(out.shape[:2]) / 500)))
    fs = max(0.5, th * 0.35)
    for g, r in zip(grains, rows):
        col = _hex_to_bgr(CATEGORY_COLORS.get(r["category"], CATEGORY_COLORS[UNCLASSIFIED]))
        if tilted:
            pts = (cv2.boxPoints(g.rect) * scale).astype(np.int32)
            cv2.polylines(out, [pts], True, col, th, cv2.LINE_AA)
            tx, ty = int(pts[:, 0].min()), int(pts[:, 1].min())
        else:
            x, y, w, h = [int(round(v * scale)) for v in g.box]
            cv2.rectangle(out, (x, y), (x + w, y + h), col, th)
            tx, ty = x, y
        label = f"{r['id']}{'*' if r['manual_inspection'] == 'Required' else ''}"
        pos = (tx, max(ty - 4, 14))
        cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(2, th), cv2.LINE_AA)
        cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, col, max(1, th // 2), cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def _safe(v):
    """Neutralise spreadsheet formulas in text that came from the model."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def rows_to_csv(rows: List[dict]) -> str:
    cols = [("id", "grain_id"), ("category", "category"), ("confidence", "confidence"),
            ("manual_inspection", "manual_inspection"), ("review_reasons", "manual_inspection_reasons"),
            ("model_reason", "model_reason"), ("x", "x_px"), ("y", "y_px"), ("width", "box_width_px"),
            ("height", "box_height_px"), ("area_px", "area_px"), ("length_px", "length_px"),
            ("width_px", "width_px"), ("aspect_ratio", "aspect_ratio"), ("solidity", "solidity"),
            ("mean_colour_hex", "mean_colour_hex"), ("hue_deg", "hue_deg"),
            ("saturation_pct", "saturation_pct"), ("brightness_pct", "brightness_pct")]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([c[1] for c in cols])
    for r in rows:
        w.writerow([_safe("" if r.get(k) is None else r[k]) for k, _ in cols])
    return buf.getvalue()


def summary_to_csv(s: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["category", "count", "percent_of_classified", "flagged_for_manual_inspection"])
    for c in s["categories"]:
        w.writerow([c["category"], c["count"], f"{c['percent']:.1f}", c["flagged"]])
    w.writerow(["Total classified", s["classified"], "100.0" if s["classified"] else "0.0",
                sum(c["flagged"] for c in s["categories"])])
    w.writerow([UNCLASSIFIED, s["unclassified"], "", s["flagged_unclassified"]])
    w.writerow(["Total detected", s["total_detected"], "", s["flagged_total"]])
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Detailed report (Markdown, shown in the app)
# --------------------------------------------------------------------------- #
def build_report_markdown(s: dict, rows: List[dict], meta: Dict[str, str]) -> str:
    L = ["# Wheat Grain Classification Report", ""]
    L += [f"- **Generated:** {meta.get('generated', datetime.now().strftime('%Y-%m-%d %H:%M'))}",
          f"- **Image:** {meta.get('image', 'uploaded image')}",
          "- **Detection and measurements:** OpenCV (HSV segmentation, contours, shape filtering)",
          f"- **Classification:** Groq Vision (`{meta.get('model', '-')}`), "
          f"{meta.get('batch', 1)} grain(s) per request",
          f"- **Manual-inspection confidence threshold:** {meta.get('threshold', DEFAULT_CONFIDENCE_THRESHOLD):.0%}", ""]
    L += ["## 1. Totals", "",
          "| Measure | Count |", "|---|---|",
          f"| Grains detected (OpenCV) | {s['total_detected']} |",
          f"| Grains classified (Groq) | {s['classified']} |",
          f"| Not classified | {s['unclassified']} |",
          f"| Flagged for manual inspection | {s['flagged_total']} |", "",
          f"Reconciliation: {reconciliation_text(s)}. "
          + ("All counts reconcile." if s["reconciled"] else "**Counts do not reconcile - check the data.**"), ""]
    L += ["## 2. Category counts", "",
          "| Category | Count | % of classified | Flagged for manual inspection |", "|---|---|---|---|"]
    for c in s["categories"]:
        L.append(f"| {c['category']} | {c['count']} | {c['percent']:.1f}% | {c['flagged']} |")
    L += [f"| **Total classified** | **{s['classified']}** | **{'100.0%' if s['classified'] else '0.0%'}** | "
          f"**{s['flagged_total'] - s['flagged_unclassified']}** |"]
    if s["unclassified"]:
        L.append(f"| Unclassified (excluded from percentages) | {s['unclassified']} | - | {s['flagged_unclassified']} |")
    L += ["", "Percentages are shares of classified grains, rounded to one decimal, so they may differ from 100.0% by rounding.", ""]
    L += ["## 3. Category definitions used", ""]
    L += [f"- **{c}**: {d}" for c, d in CATEGORY_DEFINITIONS.items()]
    L += ["", "## 4. Observations", ""] + [f"- {o}" for o in observations(s)]
    L += ["", "## 5. Recommendations", ""] + [f"- {r}" for r in recommendations(s)]
    flagged = [r for r in rows if r["manual_inspection"] == "Required"]
    L += ["", "## 6. Grains flagged for manual inspection", ""]
    if flagged:
        L += ["| ID | Category | Confidence | Why |", "|---|---|---|---|"]
        for r in flagged:
            conf = "-" if r["confidence"] is None else f"{r['confidence']:.0%}"
            L.append(f"| {r['id']} | {r['category']} | {conf} | {r['review_reasons'].replace('|', '/')} |")
    else:
        L.append("None.")
    L += ["", "## 7. Method and limitations", "",
          "- OpenCV alone detects grains, draws the boxes, counts them and measures them. Groq never changes "
          "the grain count.",
          "- Groq classifies each grain from its own cropped image and that grain's OpenCV measurements, "
          "returning one of the five categories, a confidence and a short visual reason.",
          "- Confidence is reported by the model itself and is not a calibrated probability.",
          "- Measurements are in pixels of the original image; there is no physical scale or calibration.",
          "- Grains the model could not classify are shown as Unclassified, never guessed.",
          "- This is a visual screening from a photograph. It is not an official grade or laboratory result and "
          "cannot assess moisture, protein, variety or mycotoxins."]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# PDF (reportlab is imported lazily so the rest of the app never depends on it)
# --------------------------------------------------------------------------- #
def _pdf_text(v) -> str:
    """Escape for reportlab Paragraph and keep to the Latin-1 range of the built-in fonts."""
    from xml.sax.saxutils import escape
    return escape(str(v)).encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: dict, rows: List[dict], overview_bgr: Optional[np.ndarray], meta: Dict[str, str]) -> bytes:
    from reportlab.graphics.charts.barcharts import HorizontalBarChart
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, LongTable, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=9.5)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, leading=24, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=5, keepWithNext=1,
                        textColor=colors.HexColor("#1f3d2b"))
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)
    grid = colors.HexColor("#b8b8b8")
    head_bg = colors.HexColor("#1f3d2b")

    def table(data, widths, repeat=1, font=8.5, long=False, extra=None):
        cls = LongTable if long else Table
        t = cls(data, colWidths=widths, repeatRows=repeat)
        st = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.4, grid),
              ("BACKGROUND", (0, 0), (-1, 0), head_bg), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
              ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f4")]),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        t.setStyle(TableStyle(st + (extra or [])))
        return t

    def page_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.5 * cm, 0.9 * cm, "Wheat Grain Classification Report - visual screening, not an official grade")
        canvas.drawRightString(A4[0] - 1.5 * cm, 0.9 * cm, f"Page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                            title="Wheat Grain Classification Report", author="Wheat Grain Analysis Tool")
    W = A4[0] - 3 * cm
    story = [Paragraph("Wheat Grain Classification Report", h1)]
    story.append(Paragraph(
        f"<b>Generated:</b> {_pdf_text(meta.get('generated', datetime.now().strftime('%Y-%m-%d %H:%M')))}<br/>"
        f"<b>Image:</b> {_pdf_text(meta.get('image', 'uploaded image'))}<br/>"
        "<b>Detection and measurements:</b> OpenCV (HSV segmentation, contours, shape filtering)<br/>"
        f"<b>Classification:</b> Groq Vision ({_pdf_text(meta.get('model', '-'))}), "
        f"{_pdf_text(meta.get('batch', 1))} grain(s) per request<br/>"
        f"<b>Manual-inspection confidence threshold:</b> {meta.get('threshold', DEFAULT_CONFIDENCE_THRESHOLD):.0%}",
        body))

    # 1. totals
    story.append(Paragraph("1. Totals", h2))
    story.append(table([["Measure", "Count"],
                        ["Grains detected (OpenCV)", s["total_detected"]],
                        ["Grains classified (Groq)", s["classified"]],
                        ["Not classified", s["unclassified"]],
                        ["Flagged for manual inspection", s["flagged_total"]]], [W * 0.7, W * 0.3]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        _pdf_text("Reconciliation: " + reconciliation_text(s) + ". ")
        + ("All counts reconcile." if s["reconciled"] else "<b>Counts do not reconcile.</b>"), body))

    # 2. category table
    story.append(Paragraph("2. Category counts and percentages", h2))
    data = [["Category", "Count", "% of classified", "Flagged for manual inspection"]]
    extra = []
    for i, c in enumerate(s["categories"], 1):
        data.append([c["category"], c["count"], f"{c['percent']:.1f}%", c["flagged"]])
        extra.append(("LINEBEFORE", (0, i), (0, i), 6, colors.HexColor(CATEGORY_COLORS[c["category"]])))
    data.append(["Total classified", s["classified"], "100.0%" if s["classified"] else "0.0%",
                 s["flagged_total"] - s["flagged_unclassified"]])
    n = len(data) - 1
    extra += [("FONTNAME", (0, n), (-1, n), "Helvetica-Bold"), ("LEFTPADDING", (0, 1), (0, -1), 9)]
    if s["unclassified"]:
        data.append(["Unclassified (excluded from %)", s["unclassified"], "-", s["flagged_unclassified"]])
    story.append(table(data, [W * 0.38, W * 0.14, W * 0.2, W * 0.28], extra=extra))
    story.append(Paragraph("Percentages are shares of classified grains, rounded to one decimal, so they may differ from 100.0% by rounding.",
                           small))

    # charts
    if s["classified"]:
        names = [c["category"] for c in s["categories"]]
        vals = [c["count"] for c in s["categories"]]
        d1 = Drawing(W * 0.56, 175)
        d1.add(String(0, 163, "Grains per category", fontName="Helvetica-Bold", fontSize=9))
        bc = HorizontalBarChart()
        bc.x, bc.y, bc.width, bc.height = 100, 10, W * 0.56 - 125, 140
        bc.data = [list(reversed(vals))]
        bc.categoryAxis.categoryNames = list(reversed(names))
        bc.categoryAxis.labels.fontSize = 8
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = max(vals) * 1.2 + 1
        bc.valueAxis.valueStep = None
        bc.valueAxis.labels.fontSize = 7
        bc.bars.strokeWidth = 0
        bc.barLabelFormat = "%d"
        bc.barLabels.fontSize = 8
        bc.barLabels.nudge = 8
        for idx, nm in enumerate(reversed(names)):
            bc.bars[(0, idx)].fillColor = colors.HexColor(CATEGORY_COLORS[nm])
        d1.add(bc)

        d2 = Drawing(W * 0.44, 175)
        d2.add(String(0, 163, "Share of classified grains", fontName="Helvetica-Bold", fontSize=9))
        nz = [(nm, v) for nm, v in zip(names, vals) if v > 0]
        pie = Pie()
        pie.x, pie.y, pie.width, pie.height = 5, 30, 100, 100
        pie.data = [v for _, v in nz]
        pie.labels = None
        pie.sideLabels = False
        pie.slices.strokeWidth = 0.5
        pie.slices.strokeColor = colors.white
        for idx, (nm, _) in enumerate(nz):
            pie.slices[idx].fillColor = colors.HexColor(CATEGORY_COLORS[nm])
        d2.add(pie)
        lg = Legend()
        lg.x, lg.y = 115, 120
        lg.fontName, lg.fontSize = "Helvetica", 7.5
        lg.alignment = "right"
        lg.deltay, lg.dxTextSpace, lg.columnMaximum = 11, 5, 6
        pct_of = {c["category"]: c["percent"] for c in s["categories"]}
        lg.colorNamePairs = [(colors.HexColor(CATEGORY_COLORS[nm]), f"{nm} {pct_of[nm]:.1f}%") for nm, _ in nz]
        d2.add(lg)
        story.append(Spacer(1, 6))
        story.append(Table([[d1, d2]], colWidths=[W * 0.56, W * 0.44]))

    # 3. overview image
    if overview_bgr is not None:
        ok, jpg = cv2.imencode(".jpg", _fit(overview_bgr, 1600), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            ih, iw = overview_bgr.shape[:2]
            wpt = min(W, 11.5 * cm * iw / ih)
            story.append(KeepTogether([
                Paragraph("3. Classified overview", h2),
                Image(io.BytesIO(jpg.tobytes()), width=wpt, height=wpt * ih / iw),
                Paragraph("Box colour = category; a * after the id = manual inspection required. Legend: "
                          + ", ".join(f"<font color='{CATEGORY_COLORS[k]}'><b>{_pdf_text(k)}</b></font>"
                                      for k in CATEGORY_COLORS), small)]))

    # 4 / 5. observations and recommendations
    story.append(Paragraph("4. Observations", h2))
    story += [Paragraph(_pdf_text(o), bullet, bulletText="\u2022") for o in observations(s)]
    story.append(Paragraph("5. Recommendations", h2))
    story += [Paragraph(_pdf_text(r), bullet, bulletText="\u2022") for r in recommendations(s)]

    # 6. manual inspection list
    story.append(Paragraph("6. Grains flagged for manual inspection", h2))
    flagged = [r for r in rows if r["manual_inspection"] == "Required"]
    if flagged:
        data = [["ID", "Category", "Conf.", "Why"]]
        for r in flagged:
            data.append([r["id"], Paragraph(_pdf_text(r["category"]), small),
                         "-" if r["confidence"] is None else f"{r['confidence']:.0%}",
                         Paragraph(_pdf_text(r["review_reasons"]), small)])
        story.append(table(data, [W * 0.07, W * 0.2, W * 0.08, W * 0.65], font=7.5, long=True))
    else:
        story.append(Paragraph("None.", body))

    # 7. per-grain table
    story.append(PageBreak())
    story.append(Paragraph("7. Per-grain results", h2))
    story.append(Paragraph("Lengths, widths and areas are in pixels of the original image (no physical calibration).",
                           small))
    story.append(Spacer(1, 3))
    data = [["ID", "Category", "Conf.", "Manual insp.", "Length", "Width", "Aspect", "Solidity", "Area"]]
    extra = []
    for i, r in enumerate(rows, 1):
        data.append([r["id"], r["category"], "-" if r["confidence"] is None else f"{r['confidence']:.0%}",
                     r["manual_inspection"], f"{r['length_px']:.0f}", f"{r['width_px']:.0f}",
                     f"{r['aspect_ratio']:.2f}", f"{r['solidity']:.2f}", f"{r['area_px']:.0f}"])
        if r["manual_inspection"] == "Required":
            extra.append(("TEXTCOLOR", (3, i), (3, i), colors.HexColor("#b00020")))
    story.append(table(data, [W * 0.07, W * 0.23, W * 0.08, W * 0.14, W * 0.1, W * 0.09, W * 0.09,
                              W * 0.1, W * 0.1], font=7.5, long=True, extra=extra))

    # 8. method
    method = [Paragraph("8. Category definitions, method and limitations", h2)]
    method += [Paragraph(f"<b>{_pdf_text(c)}</b>: {_pdf_text(d)}", bullet, bulletText="\u2022")
               for c, d in CATEGORY_DEFINITIONS.items()]
    for t in ["OpenCV alone detects grains, draws the boxes, counts them and measures them. Groq never changes the grain count.",
              "Groq classifies each grain from its own cropped image and that grain's OpenCV measurements, returning "
              "one of the five categories, a confidence and a short visual reason.",
              "Confidence is reported by the model itself and is not a calibrated probability.",
              "Grains the model could not classify are shown as Unclassified, never guessed.",
              "This is a visual screening from a photograph. It is not an official grade or laboratory result and cannot "
              "assess moisture, protein, variety or mycotoxins."]:
        method.append(Paragraph(_pdf_text(t), bullet, bulletText="\u2022"))
    story.append(KeepTogether(method))

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return buf.getvalue()


def _fit(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    f = max_side / float(max(h, w))
    return cv2.resize(bgr, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
