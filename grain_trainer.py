"""
Train your own grain-quality classifier (runs 100 % locally, no API key, no internet).

Workflow
--------
1. Upload a photo; OpenCV detects the grains (wheat_detector.py).
2. You label grains one by one (Sound / Shrivelled / Immature / Broken / Damaged/Discoloured).
3. ``TrainingStore.add_image`` turns every labelled grain into a feature vector and saves it.
4. ``fit`` trains a Random Forest on everything saved so far.
5. ``predict`` grades the grains of any other photo; the browser compares the predictions with
   the labels you give that photo to report the real accuracy.

Features (per grain, all rotation-invariant and scale-free where possible)
-------------------------------------------------------------------------
* shape  : aspect ratio, solidity, extent, circularity, length / width / area relative to the
           median grain of the SAME photo (so a fragment looks small whatever the zoom is)
* colour : mean + spread of H, S, V and L*a*b* inside the grain outline, plus how far each
           mean is from the photo's median grain (cancels most lighting differences)
* defect : share of unusually dark / pale / off-hue pixels (rot, stains, chalky or green grain)
* texture: grey-level spread, Laplacian roughness and edge density (wrinkles, cracks, holes)

Files written next to this module (git-ignored):
    training_data/samples.json   labelled grains (features only, no images)
    models/grain_model.joblib    the trained model
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "training_data"
MODEL_DIR = BASE / "models"
SAMPLES_FILE = DATA_DIR / "samples.json"
MODEL_FILE = MODEL_DIR / "grain_model.joblib"

# Same names / colours as grain_report.py so both sources look identical in the dashboard.
CLASSES = ["Sound", "Shrivelled", "Immature", "Broken", "Damaged/Discoloured"]
COLORS = {
    "Sound": "#2e9e4f",
    "Shrivelled": "#f28c28",
    "Immature": "#e0c000",
    "Broken": "#8e44ad",
    "Damaged/Discoloured": "#d62728",
    "Unclassified": "#7f7f7f",
}
DEFINITIONS = {
    "Sound": "Plump, intact grain with normal uniform colour and no visible defect.",
    "Shrivelled": "Whole grain that is visibly shrunken, wrinkled, thin or flattened.",
    "Immature": "Underdeveloped grain: small, greenish, pale or chalky, poorly filled.",
    "Broken": "Fragmented, cracked, chipped or with a visibly missing part.",
    "Damaged/Discoloured": "Insect holes, sprouting, mould-like patches, heat/weather damage or stained colour.",
}

MIN_SAMPLES = 4          # smallest training set we accept
MIN_CLASSES = 2          # a classifier needs at least two different labels
FEATURE_VERSION = 1

FEATURE_NAMES = [
    # shape
    "aspect", "solidity", "extent", "circularity", "rel_length", "rel_width", "rel_area",
    # colour (absolute)
    "h_mean", "s_mean", "v_mean", "L_mean", "a_mean", "b_mean", "h_std", "s_std", "v_std",
    # colour (relative to the photo's median grain)
    "d_h", "d_s", "d_v", "d_L", "d_a", "d_b",
    # defect fractions
    "dark_frac", "pale_frac", "offhue_frac",
    # texture
    "gray_std", "lap_mean", "edge_density",
]


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def _patch(orig: np.ndarray, contour_work: np.ndarray, scale: float, pad: int = 2):
    """Crop of the original image around one grain + a mask of its outline (original pixels)."""
    c = np.round(contour_work.astype(np.float32) * scale).astype(np.int32)
    x, y, w, h = cv2.boundingRect(c)
    H, W = orig.shape[:2]
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    crop = orig[y0:y1, x0:x1]
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.drawContours(mask, [c - np.array([x0, y0])], -1, 255, thickness=-1)
    inner = cv2.erode(mask, np.ones((3, 3), np.uint8))          # drop blended edge pixels
    if int(inner.sum()) // 255 >= 20:
        mask = inner
    return crop, mask


def extract_features(orig_bgr: np.ndarray, grains: Sequence, scale: float) -> np.ndarray:
    """Feature matrix (n_grains x len(FEATURE_NAMES)) for the grains found in ONE photo."""
    n = len(grains)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)))

    lengths = np.array([max(g.rect[1]) * scale for g in grains], float)
    widths = np.array([min(g.rect[1]) * scale for g in grains], float)
    areas = np.array([g.area * scale * scale for g in grains], float)
    med_l, med_w, med_a = (max(float(np.median(v)), 1e-6) for v in (lengths, widths, areas))

    stats, patches = [], []
    for g in grains:
        crop, mask = _patch(orig_bgr, g.contour, scale)
        sel = mask > 0
        if not sel.any():
            sel = np.ones(mask.shape, bool)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv_px, lab_px = hsv[sel], lab[sel]
        stats.append((hsv_px.mean(0), lab_px.mean(0), hsv_px.std(0)))
        patches.append((crop, sel, hsv_px))

    hsv_med = np.median([s[0] for s in stats], axis=0)
    lab_med = np.median([s[1] for s in stats], axis=0)

    rows = []
    for i, g in enumerate(grains):
        crop, sel, hsv_px = patches[i]
        hsv_m, lab_m, hsv_s = stats[i]
        per = max(cv2.arcLength(g.contour, True) * scale, 1e-6)
        area = areas[i]
        rect_area = max(lengths[i] * widths[i], 1e-6)

        dark = float((hsv_px[:, 2] < 0.70 * hsv_med[2]).mean())
        pale = float((hsv_px[:, 1] < 0.60 * hsv_med[1]).mean())
        offhue = float((np.abs(hsv_px[:, 0] - hsv_med[0]) > 6).mean())

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        edges = cv2.Canny(gray, 60, 140) > 0
        rows.append([
            g.aspect, g.solidity, min(area / rect_area, 1.5), min(4 * np.pi * area / per ** 2, 1.5),
            lengths[i] / med_l, widths[i] / med_w, areas[i] / med_a,
            hsv_m[0], hsv_m[1], hsv_m[2], lab_m[0], lab_m[1], lab_m[2], hsv_s[0], hsv_s[1], hsv_s[2],
            hsv_m[0] - hsv_med[0], hsv_m[1] - hsv_med[1], hsv_m[2] - hsv_med[2],
            lab_m[0] - lab_med[0], lab_m[1] - lab_med[1], lab_m[2] - lab_med[2],
            dark, pale, offhue,
            float(gray[sel].std()), float(lap[sel].mean()), float(edges[sel].mean()),
        ])
    return np.nan_to_num(np.array(rows, float), nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Training-set store (JSON on disk)
# --------------------------------------------------------------------------- #
class TrainingStore:
    """Labelled grains collected from one or more photos, persisted in training_data/samples.json."""

    def __init__(self, path: Path = SAMPLES_FILE):
        self.path = path
        self.lock = threading.RLock()
        self.samples: List[dict] = []
        self._load()

    def _load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("feature_version") == FEATURE_VERSION and data.get("feature_names") == FEATURE_NAMES:
                self.samples = [s for s in data.get("samples", []) if s.get("label") in CLASSES]
        except Exception:
            self.samples = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
                                   "saved": datetime.now().isoformat(timespec="seconds"),
                                   "samples": self.samples}), encoding="utf-8")
        tmp.replace(self.path)

    def add_image(self, source: str, name: str, ids: List[int], labels: List[str], feats: np.ndarray):
        """Store the labelled grains of one photo. Earlier labels from the same photo are replaced."""
        with self.lock:
            self.samples = [s for s in self.samples if s["source"] != source]
            for gid, lab, f in zip(ids, labels, feats):
                self.samples.append({"source": source, "name": name, "id": int(gid), "label": lab,
                                     "x": [round(float(v), 5) for v in f]})
            self._save()

    def remove_source(self, source: str):
        with self.lock:
            self.samples = [s for s in self.samples if s["source"] != source]
            self._save()

    def clear(self):
        with self.lock:
            self.samples = []
            self._save()

    def matrix(self):
        with self.lock:
            X = np.array([s["x"] for s in self.samples], float) if self.samples else np.zeros((0, len(FEATURE_NAMES)))
            y = [s["label"] for s in self.samples]
        return X, y

    def summary(self) -> dict:
        with self.lock:
            counts = Counter(s["label"] for s in self.samples)
            srcs: Dict[str, dict] = {}
            for s in self.samples:
                d = srcs.setdefault(s["source"], {"source": s["source"], "name": s["name"], "count": 0})
                d["count"] += 1
            return {"total": len(self.samples), "counts": {c: counts.get(c, 0) for c in CLASSES},
                    "sources": list(srcs.values())}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GrainModel:
    """Random Forest wrapper: fit(), predict(), and cached metadata for the dashboard."""

    def __init__(self, path: Path = MODEL_FILE):
        self.path = path
        self.lock = threading.RLock()
        self.clf = None
        self.meta: dict = {}
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self):
        try:
            import joblib
            if self.path.is_file():
                blob = joblib.load(self.path)
                if blob.get("feature_names") == FEATURE_NAMES:
                    self.clf, self.meta = blob["clf"], blob["meta"]
        except Exception:
            self.clf, self.meta = None, {}

    def _save(self):
        import joblib
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "meta": self.meta, "feature_names": FEATURE_NAMES}, self.path)

    # -- training -----------------------------------------------------------
    def fit(self, X: np.ndarray, y: List[str]) -> dict:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
        except ImportError as e:                                           # pragma: no cover
            raise RuntimeError("Training needs scikit-learn:  pip install scikit-learn") from e

        counts = Counter(y)
        if len(y) < MIN_SAMPLES or len(counts) < MIN_CLASSES:
            raise ValueError(f"Label at least {MIN_SAMPLES} grains covering at least {MIN_CLASSES} different "
                             f"types before training (you have {len(y)} grains, {len(counts)} type"
                             f"{'' if len(counts) == 1 else 's'}).")

        def make():
            return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                          min_samples_leaf=1, random_state=0, n_jobs=1)

        # Cross-validation: grains from the same photo are similar, so this is only a sanity check.
        cv = {"accuracy": None, "folds": 0, "per_class": {}, "note": ""}
        ya = np.array(y)
        min_c = min(counts.values())
        try:
            if min_c >= 2:
                k = int(min(5, min_c))
                splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
            elif len(y) >= 6:
                k = int(min(5, len(y)))
                splitter = KFold(n_splits=k, shuffle=True, random_state=0)
                cv["note"] = "A type has only one example, so the check is rough."
            else:
                splitter, k = None, 0
                cv["note"] = "Too few labelled grains for a cross-check."
            if splitter is not None:
                pred = cross_val_predict(make(), X, ya, cv=splitter)
                cv["accuracy"] = float(accuracy_score(ya, pred))
                cv["folds"] = k
                for c in counts:
                    m = ya == c
                    cv["per_class"][c] = {"n": int(m.sum()), "correct": int((pred[m] == c).sum())}
        except Exception as e:                                             # never block training on the check
            cv["note"] = f"Cross-check skipped ({type(e).__name__})."

        clf = make().fit(X, ya)
        meta = {"version": datetime.now().strftime("%Y%m%d%H%M%S"),
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "n_samples": int(len(y)), "counts": {c: counts.get(c, 0) for c in CLASSES},
                "classes": [str(c) for c in clf.classes_], "cv": cv}
        with self.lock:
            self.clf, self.meta = clf, meta
            self._save()
        return meta

    def reset(self):
        with self.lock:
            self.clf, self.meta = None, {}
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    # -- inference ----------------------------------------------------------
    @property
    def trained(self) -> bool:
        return self.clf is not None

    def predict(self, X: np.ndarray) -> List[dict]:
        if not self.trained:
            raise RuntimeError("No trained model yet.")
        if len(X) == 0:
            return []
        with self.lock:
            proba = self.clf.predict_proba(X)
            classes = [str(c) for c in self.clf.classes_]
        out = []
        for p in proba:
            order = np.argsort(p)[::-1]
            out.append({"label": classes[order[0]], "confidence": float(p[order[0]]),
                        "probs": {classes[j]: round(float(p[j]), 3) for j in order if p[j] > 0}})
        return out

    def info(self) -> dict:
        if not self.trained:
            return {"trained": False, "version": None}
        return {"trained": True, "version": self.meta.get("version"), "trained_at": self.meta.get("trained_at"),
                "n_samples": self.meta.get("n_samples"), "counts": self.meta.get("counts"),
                "classes": self.meta.get("classes"), "cv": self.meta.get("cv")}
