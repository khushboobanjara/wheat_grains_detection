"""
Wheat Grain Analysis Dashboard - Flask backend.

Run with:   python server.py        (opens http://localhost:8000)

The detection pipeline (wheat_detector.py) and the optional Groq modules
(groq_analyzer.py, grain_report.py) are used exactly as before. This file only
exposes them through a small JSON API for the HTML/CSS/JS front end in /static.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps

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
try:                       # your own trainable classifier (needs scikit-learn)
    import grain_trainer as gt
except Exception:
    gt = None

BASE = Path(__file__).resolve().parent
ACCEPTED = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
MAX_SESSIONS = 40

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024          # 300 MB per request
app.json.sort_keys = False                                    # keep preset / category order

SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
LOCK = threading.Lock()
STORE = gt.TrainingStore() if gt else None          # labelled grains saved on disk
MODEL = gt.GrainModel() if gt else None             # the trained classifier


# =========================================================================== #
# Helpers
# =========================================================================== #
def load_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR array, honouring EXIF rotation."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def data_url(bgr: np.ndarray, fmt: str = "jpg", max_side: int = 1800, quality: int = 86) -> str:
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        f = max_side / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    if fmt == "png":
        ok, buf = cv2.imencode(".png", bgr)
        mime = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime = "image/jpeg"
    if not ok:
        raise RuntimeError("image encoding failed")
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode()


def get_session(sid: str) -> dict:
    with LOCK:
        s = SESSIONS.get(sid)
    if s is None:
        raise KeyError("This image is no longer available. Please upload it again.")
    return s


def parse_params(raw: dict | None) -> tuple[Params, bool, bool]:
    """Turn the JSON settings sent by the browser into Params (+ display options)."""
    raw = raw or {}
    d = Params()

    def pair(key, lo, hi, cast):
        v = raw.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return cast(v[0]), cast(v[1])
        return lo, hi

    h = pair("hue", d.h_min, d.h_max, int)
    s = pair("sat", d.s_min, d.s_max, int)
    v = pair("val", d.v_min, d.v_max, int)
    area = pair("area", d.min_area, d.max_area, int)
    asp = pair("aspect", d.min_aspect, d.max_aspect, float)

    p = Params(
        h_min=h[0], h_max=h[1], s_min=s[0], s_max=s[1], v_min=v[0], v_max=v[1],
        auto_saturation=bool(raw.get("auto", d.auto_saturation)),
        blur=int(raw.get("blur", d.blur)), open_iter=int(raw.get("open", d.open_iter)),
        close_iter=int(raw.get("close", d.close_iter)),
        min_area=area[0], max_area=area[1], min_aspect=asp[0], max_aspect=asp[1],
        min_solidity=float(raw.get("solidity", d.min_solidity)),
        rel_min_pct=int(raw.get("rel", d.rel_min_pct)),
        split=bool(raw.get("split", d.split)), split_level=float(raw.get("level", d.split_level)),
    )
    return p, bool(raw.get("ids", False)), bool(raw.get("tilted", False))


def get_detection(sess: dict, params: Params, ids: bool, tilted: bool) -> dict:
    """Run (or reuse) the detection for this image + settings."""
    key = f"{params!r}|{ids}|{tilted}"
    cached = sess.get("det")
    if cached and cached["key"] == key:
        return cached
    if "orig" not in sess:
        sess["orig"] = load_image(sess["data"])
    original = sess["orig"]
    work, scale = resize_for_processing(original)
    t0 = time.perf_counter()
    res = detect(work, params)
    annotated = annotate(original, res.grains, scale, ids, tilted)
    elapsed = time.perf_counter() - t0
    det = {"key": key, "work": work, "scale": scale, "res": res, "annotated": annotated,
           "elapsed": elapsed, "original": original}
    sess["det"] = det
    return det


def grain_rows(grains, scale: float) -> list[dict]:
    rows = []
    for i, g in enumerate(grains, 1):
        x, y, w, h = [int(round(v * scale)) for v in g.box]
        rows.append({"id": i, "x": x, "y": y, "box_w": w, "box_h": h,
                     "length": round(float(max(g.rect[1]) * scale), 2),
                     "width": round(float(min(g.rect[1]) * scale), 2),
                     "aspect": round(float(g.aspect), 3), "solidity": round(float(g.solidity), 3),
                     "area": round(float(g.area * scale * scale), 1)})
    if rows:
        med = float(np.median([r["area"] for r in rows]))
        for r in rows:
            r["size_flag"] = "Small" if r["area"] < 0.65 * med else ("Large" if r["area"] > 1.5 * med else "Normal")
    return rows


def summary_stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"count": 0}
    L = np.array([r["length"] for r in rows]); W = np.array([r["width"] for r in rows])
    A = np.array([r["area"] for r in rows]); asp = np.array([r["aspect"] for r in rows])
    sol = np.array([r["solidity"] for r in rows])
    cv = float(A.std() / A.mean() * 100) if n > 1 else 0.0
    return {"count": n, "avg_length": float(L.mean()), "med_length": float(np.median(L)),
            "min_length": float(L.min()), "max_length": float(L.max()),
            "avg_width": float(W.mean()), "med_width": float(np.median(W)),
            "avg_aspect": float(asp.mean()), "avg_solidity": float(sol.mean()),
            "variation": cv,
            "small": sum(1 for r in rows if r["size_flag"] == "Small"),
            "large": sum(1 for r in rows if r["size_flag"] == "Large")}


def err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


@app.errorhandler(KeyError)
def _key_error(e):
    return err(str(e.args[0]) if e.args else "Not found", 404)


@app.errorhandler(ValueError)
def _value_error(e):                                  # friendly, expected problems (not enough labels, ...)
    return err(str(e), 400)


@app.errorhandler(Exception)
def _any_error(e):
    if hasattr(e, "code") and isinstance(e.code, int):          # werkzeug HTTP errors
        return err(getattr(e, "description", "Request failed"), e.code)
    return err(f"{type(e).__name__}: {e}", 500)


# =========================================================================== #
# Pages and config
# =========================================================================== #
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    cfg = gz.load_config() if gz is not None else None
    defaults = Params()
    return jsonify({
        "groq": bool(cfg and cfg.configured), "model": cfg.model if cfg else "",
        "ai_available": gz is not None and gr is not None,
        "presets": {k: {"hue": list(v["h"]), "sat": list(v["s"]), "val": list(v["v"])} for k, v in PRESETS.items()},
        "defaults": {"hue": [defaults.h_min, defaults.h_max], "sat": [defaults.s_min, defaults.s_max],
                     "val": [defaults.v_min, defaults.v_max], "blur": defaults.blur, "open": defaults.open_iter,
                     "close": defaults.close_iter, "area": [defaults.min_area, defaults.max_area],
                     "aspect": [defaults.min_aspect, defaults.max_aspect], "solidity": defaults.min_solidity,
                     "rel": defaults.rel_min_pct, "split": defaults.split, "level": defaults.split_level},
        "categories": gr.CATEGORY_COLORS if gr else {},
        "category_defs": gr.CATEGORY_DEFINITIONS if gr else {},
        "classes": gt.CLASSES if gt else [], "colors": gt.COLORS if gt else {},
        "class_defs": gt.DEFINITIONS if gt else {},
        "train": train_state(),
    })


# =========================================================================== #
# Upload / detection
# =========================================================================== #
@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return err("No files received.")
    out = []
    for f in files:
        ext = (f.filename or "").rsplit(".", 1)[-1].lower()
        if ext not in ACCEPTED:
            out.append({"name": f.filename, "error": "Unsupported file type."})
            continue
        data = f.read()
        try:
            img = load_image(data)
        except Exception:
            out.append({"name": f.filename, "error": "Could not read this image."})
            continue
        sid = uuid.uuid4().hex
        with LOCK:
            SESSIONS[sid] = {"name": f.filename, "data": data, "orig": img, "size": len(data)}
            while len(SESSIONS) > MAX_SESSIONS:
                SESSIONS.popitem(last=False)
        out.append({"sid": sid, "name": f.filename, "width": int(img.shape[1]), "height": int(img.shape[0]),
                    "size": len(data)})
    return jsonify({"files": out})


@app.get("/api/original/<sid>")
def original(sid):
    s = get_session(sid)
    return jsonify({"image": data_url(s["orig"])})


@app.post("/api/detect")
def api_detect():
    body = request.get_json(force=True)
    sess = get_session(body["sid"])
    params, ids, tilted = parse_params(body.get("params"))
    det = get_detection(sess, params, ids, tilted)
    rows = grain_rows(det["res"].grains, det["scale"])
    h, w = det["original"].shape[:2]
    return jsonify({
        "width": int(w), "height": int(h), "elapsed": round(det["elapsed"], 3),
        "rejected": int(det["res"].rejected), "grains": rows, "stats": summary_stats(rows),
        "processed": data_url(det["annotated"]), "mask": data_url(det["res"].mask, "png", 1400),
    })


@app.post("/api/batch")
def api_batch():
    body = request.get_json(force=True)
    params, _, _ = parse_params(body.get("params"))
    rows = []
    for sid in body.get("sids", []):
        try:
            s = get_session(sid)
            original_img = s["orig"]
            work, scale = resize_for_processing(original_img)
            res = detect(work, params)
            gr_rows = grain_rows(res.grains, scale)
            st = summary_stats(gr_rows)
            rows.append({"name": s["name"], "grains": st["count"],
                         "avg_length": round(st.get("avg_length", 0), 1),
                         "avg_width": round(st.get("avg_width", 0), 1),
                         "variation": round(st.get("variation", 0), 1), "rejected": int(res.rejected),
                         "resolution": f"{original_img.shape[1]}x{original_img.shape[0]}"})
        except Exception as e:
            rows.append({"name": sid, "grains": None, "resolution": f"error: {type(e).__name__}"})
    return jsonify({"rows": rows})


# =========================================================================== #
# Optional Groq classification
# =========================================================================== #
def _sig(sess: dict, params: Params) -> str:
    return f"{sess['name']}|{sess['size']}|{params}"


def classification_view(sess: dict, params: Params, threshold: float, tilted: bool) -> dict:
    saved = sess.get("cls")
    if not saved:
        return {"state": "none"}
    if saved["sig"] != _sig(sess, params):
        return {"state": "stale"}
    if saved.get("error"):
        return {"state": "error", "error": saved["error"]}
    det = get_detection(sess, params, False, tilted)
    run, meas = saved["run"], saved["meas"]
    if len(meas) != len(det["res"].grains):
        return {"state": "stale"}
    rows = gr.build_rows(meas, run.results, threshold)
    summary = gr.summarize(rows)
    overview = gr.annotate_classified(det["original"], det["res"].grains, det["scale"], rows, tilted)
    sess["cls_ctx"] = {"rows": rows, "summary": summary, "overview": overview,
                       "meta": {"image": sess["name"], "model": run.model, "batch": run.batch_size,
                                "threshold": threshold, "generated": saved["time"]}}
    return {
        "state": "ok", "rows": rows, "summary": summary, "overview": data_url(overview),
        "fatal_error": run.fatal_error, "hint": run.hint, "model": run.model,
        "reconciliation": gr.reconciliation_text(summary),
        "observations": gr.observations(summary), "recommendations": gr.recommendations(summary),
        "report": gr.build_report_markdown(summary, rows, sess["cls_ctx"]["meta"]),
        "threshold": threshold,
    }


@app.post("/api/classify")
def api_classify():
    if gz is None or gr is None:
        return err("Grain classification is unavailable in this installation.")
    body = request.get_json(force=True)
    sess = get_session(body["sid"])
    params, _, tilted = parse_params(body.get("params"))
    threshold = float(body.get("threshold", 0.7))
    batch = int(body.get("batch", 1))
    cfg = gz.load_config()
    if not cfg.configured:
        return err("No Groq API key configured. Add GROQ_API_KEY to the .env file and reload.")
    det = get_detection(sess, params, False, tilted)
    n = len(det["res"].grains)
    if n == 0:
        return err("No grains detected, so there is nothing to classify.")
    sess["progress"] = {"done": 0, "total": n}

    def on_progress(done, total):
        sess["progress"] = {"done": done, "total": total}

    sig = _sig(sess, params)
    try:
        meas = gz.measure_grains(det["work"], det["res"].grains, det["scale"])
        run = gz.classify_grains(det["original"], det["res"].grains, meas, det["scale"], cfg, batch, on_progress)
        sess["cls"] = {"sig": sig, "meas": meas, "run": run, "error": "",
                       "time": datetime.now().strftime("%Y-%m-%d %H:%M")}
    except Exception as e:                                   # never break the detection page
        sess["cls"] = {"sig": sig, "error": f"Classification failed: {type(e).__name__}"}
    finally:
        sess["progress"] = None
    return jsonify(classification_view(sess, params, threshold, tilted))


@app.get("/api/progress/<sid>")
def progress(sid):
    s = get_session(sid)
    return jsonify(s.get("progress") or {"done": 0, "total": 0})


@app.post("/api/classification")
def api_classification():
    """Re-read stored classification results (e.g. after changing the inspection threshold)."""
    if gr is None:
        return jsonify({"state": "none"})
    body = request.get_json(force=True)
    sess = get_session(body["sid"])
    params, _, tilted = parse_params(body.get("params"))
    return jsonify(classification_view(sess, params, float(body.get("threshold", 0.7)), tilted))


# =========================================================================== #
# Optional Groq written report
# =========================================================================== #
@app.post("/api/ai")
def api_ai():
    if gz is None:
        return err("AI analysis is unavailable in this installation.")
    body = request.get_json(force=True)
    sess = get_session(body["sid"])
    params, _, tilted = parse_params(body.get("params"))
    cfg = gz.load_config()
    if not cfg.configured:
        return err("No Groq API key configured. Add GROQ_API_KEY to the .env file and reload.")
    det = get_detection(sess, params, False, tilted)
    grains = det["res"].grains
    if not grains:
        return err("No grains detected, so there is nothing to analyse.")
    try:
        stats = gz.summarize_grains(det["work"], grains, det["scale"])
        numbered = annotate(det["original"], grains, det["scale"], True, tilted)
        montage = gz.make_montage(det["original"], grains, det["scale"], [s["id"] for s in stats["suspects"]])
        out = gz.analyze(numbered, montage, stats, cfg)
        sess["ai"] = {"sig": _sig(sess, params), "ok": out.ok, "error": out.error, "hint": out.hint,
                      "report": out.report,
                      "doc": gz.build_report_document(out.report, stats, out.model, sess["name"]) if out.ok else ""}
    except Exception as e:
        sess["ai"] = {"sig": _sig(sess, params), "ok": False, "report": "", "doc": "",
                      "error": f"AI analysis failed: {type(e).__name__}", "hint": ""}
    a = sess["ai"]
    return jsonify({"ok": a["ok"], "error": a["error"], "hint": a["hint"], "report": a["report"]})


# =========================================================================== #
# Train your own classifier (local, scikit-learn)
# =========================================================================== #
def train_state() -> dict:
    if gt is None:
        return {"available": False, "reason": "scikit-learn is not installed. Run: pip install scikit-learn",
                "model": {"trained": False, "version": None}, "store": {"total": 0, "counts": {}, "sources": []}}
    return {"available": True, "model": MODEL.info(), "store": STORE.summary()}


def _need_trainer():
    if gt is None:
        raise ValueError("Training needs scikit-learn. Run:  pip install scikit-learn   then restart the server.")


def _source_key(sess: dict) -> str:
    return f"{sess['name']}|{sess['size']}"


def _detect_light(sess: dict, params: Params, ids: bool, tilted: bool):
    """Detection for this image + settings, reusing the cached result when it matches (no annotation)."""
    cached = sess.get("det")
    if cached and cached["key"] == f"{params!r}|{ids}|{tilted}":
        return cached["original"], cached["res"].grains, cached["scale"]
    original = sess.get("orig")
    if original is None:
        original = sess["orig"] = load_image(sess["data"])
    work, scale = resize_for_processing(original)
    return original, detect(work, params).grains, scale


def _features(sess: dict, params: Params, ids: bool, tilted: bool):
    """Feature matrix for the grains of this image (cached per image + detection settings)."""
    key = repr(params)
    f = sess.get("feat")
    if f and f["key"] == key:
        return f["X"], f["grains"], f["scale"]
    original, grains, scale = _detect_light(sess, params, ids, tilted)
    X = gt.extract_features(original, grains, scale)
    sess["feat"] = {"key": key, "X": X, "grains": grains, "scale": scale}
    return X, grains, scale


@app.get("/api/train/state")
def api_train_state():
    return jsonify(train_state())


@app.post("/api/train/commit")
def api_train_commit():
    """Save the labels you gave one image to the training set (replaces earlier labels of that image)."""
    _need_trainer()
    body = request.get_json(force=True)
    sess = get_session(body["sid"])
    params, ids, tilted = parse_params(body.get("params"))
    labels = body.get("labels") or {}
    X, grains, scale = _features(sess, params, ids, tilted)
    gids, labs = [], []
    for k, v in labels.items():
        i = int(k)
        if v in gt.CLASSES and 1 <= i <= len(grains):
            gids.append(i); labs.append(v)
    if not gids:
        return err("Label at least one grain first (click a grain, then choose its type).")
    STORE.add_image(_source_key(sess), sess["name"], gids, labs, X[[i - 1 for i in gids]])
    return jsonify({"saved": len(gids), "state": train_state()})


@app.post("/api/train/fit")
def api_train_fit():
    _need_trainer()
    X, y = STORE.matrix()
    MODEL.fit(X, y)
    return jsonify(train_state())


@app.post("/api/train/predict")
def api_train_predict():
    """Grade the grains of one or more images with your trained model."""
    _need_trainer()
    if not MODEL.trained:
        return err("Train a model first (Train tab).")
    body = request.get_json(force=True)
    params, ids, tilted = parse_params(body.get("params"))
    out = {}
    for sid in body.get("sids", []):
        try:
            sess = get_session(sid)
            X, grains, scale = _features(sess, params, ids, tilted)
            preds = MODEL.predict(X)
            rows = grain_rows(grains, scale)
            for r, p in zip(rows, preds):
                r.update(label=p["label"], confidence=round(p["confidence"], 3), probs=p["probs"])
            out[sid] = {"name": sess["name"], "rows": rows}
        except KeyError as e:
            out[sid] = {"name": sid, "rows": [], "error": str(e.args[0])}
    return jsonify({"version": MODEL.meta.get("version"), "results": out})


@app.post("/api/train/delete")
def api_train_delete():
    _need_trainer()
    body = request.get_json(force=True)
    if body.get("all"):
        STORE.clear(); MODEL.reset()
    elif body.get("source"):
        STORE.remove_source(body["source"])
    return jsonify(train_state())


# =========================================================================== #
# Downloads
# =========================================================================== #
def _send(data, name: str, mime: str):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return send_file(io.BytesIO(data), mimetype=mime, as_attachment=True, download_name=name)


@app.get("/api/download/<kind>")
def download(kind):
    sid = request.args.get("sid", "")
    sess = get_session(sid)
    raw = json.loads(request.args.get("p", "{}") or "{}")
    params, ids, tilted = parse_params(raw.get("params"))
    threshold = float(raw.get("threshold", 0.7))
    stem = sess["name"].rsplit(".", 1)[0]
    det = get_detection(sess, params, ids, tilted)

    if kind == "processed":
        return _send(encode_png(det["annotated"]), f"{stem}_detected.png", "image/png")
    if kind == "mask":
        return _send(encode_png(det["res"].mask), f"{stem}_mask.png", "image/png")
    if kind == "csv":
        return _send(grains_to_csv(det["res"].grains, det["scale"]), f"{stem}_grains.csv", "text/csv")

    if kind in ("pdf", "cls_csv", "cls_summary"):
        view = classification_view(sess, params, threshold, tilted)
        if view.get("state") != "ok":
            return err("Run the classification first.")
        ctx = sess["cls_ctx"]
        if kind == "pdf":
            try:
                pdf = gr.build_pdf(ctx["summary"], ctx["rows"], ctx["overview"], ctx["meta"])
            except ImportError:
                return err("PDF export needs reportlab: pip install reportlab", 500)
            return _send(pdf, f"{stem}_classification_report.pdf", "application/pdf")
        if kind == "cls_csv":
            return _send(gr.rows_to_csv(ctx["rows"]), f"{stem}_grain_classification.csv", "text/csv")
        return _send(gr.summary_to_csv(ctx["summary"]), f"{stem}_category_summary.csv", "text/csv")

    if kind in ("md", "txt"):
        a = sess.get("ai")
        if not a or not a.get("ok") or a["sig"] != _sig(sess, params):
            return err("Generate the AI report first.")
        if kind == "md":
            return _send(a["doc"], f"{stem}_analysis.md", "text/markdown")
        return _send(a["doc"], f"{stem}_analysis.txt", "text/plain")

    return err("Unknown download.", 404)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n  Wheat Grain Analysis Dashboard  ->  http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
