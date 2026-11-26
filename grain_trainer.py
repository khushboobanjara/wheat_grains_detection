"""
Grain-quality classifier: feature extraction + model wrapper (runs 100 % locally, no API key).

How the pieces fit together
---------------------------
    train.py   (terminal)   photos sorted into one folder per grain type
                            -> OpenCV finds the grains -> features -> Random Forest
                            -> models/grain_model.joblib   (+ held-out evaluation)
    server.py  (dashboard)  loads that file and only PREDICTS the grains of uploaded photos.

Nothing is trained inside the browser any more.

Features (per grain)
--------------------
* shape  : aspect ratio, solidity, extent, circularity, length / width / area relative to a REFERENCE grain
* colour : mean + spread of H, S, V and L*a*b* inside the grain outline, plus offset from the reference colour
* defect : share of unusually dark / pale / off-hue pixels (rot, stains, chalky or green grain)
* texture: grey-level spread, Laplacian roughness and edge density (wrinkles, cracks, holes)

Reference grain
---------------
"Relative" features need a yardstick:

* ``normalize = "reference"`` (default when training from sorted folders): a fixed reference (median Sound
  grain of the training photos) is stored inside the model and reused for every prediction. Needed because a
  photo that contains ONLY broken grains has no "normal" grain to compare against.
  -> take all photos with the same camera distance / lighting.
* ``normalize = "photo"``: the median grain of each photo is the yardstick (needs photos that mix good and bad
  grains, e.g. the legacy labels in training_data/samples.json).

The mode and the reference are saved in the model file, so the dashboard always applies the same one.
"""

from __future__ import annotations

import json
import re
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
LEGACY_SAMPLES = DATA_DIR / "samples.json"          # labelled grains saved by the old in-browser trainer
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
# Folder names are matched loosely (case, spaces, "_" and "-" are ignored; British/US spelling both work).
FOLDER_ALIASES = [("sound", "Sound"), ("shriv", "Shrivelled"), ("shrunk", "Shrivelled"), ("immat", "Immature"),
                  ("broken", "Broken"), ("damag", "Damaged/Discoloured"), ("discolo", "Damaged/Discoloured")]
FOLDER_NAMES = {"Sound": "Sound", "Shrivelled": "Shrivelled", "Immature": "Immature", "Broken": "Broken",
                "Damaged/Discoloured": "Damaged_Discoloured"}      # what `train.py --make-folders` creates

MIN_SAMPLES = 4          # smallest training set we accept
MIN_CLASSES = 2          # a classifier needs at least two different labels
FEATURE_VERSION = 1

FEATURE_NAMES = [
    # shape
    "aspect", "solidity", "extent", "circularity", "rel_length", "rel_width", "rel_area",
    # colour (absolute)
    "h_mean", "s_mean", "v_mean", "L_mean", "a_mean", "b_mean", "h_std", "s_std", "v_std",
    # colour (relative to the reference grain)
    "d_h", "d_s", "d_v", "d_L", "d_a", "d_b",
    # defect fractions
    "dark_frac", "pale_frac", "offhue_frac",
    # texture
    "gray_std", "lap_mean", "edge_density",
]


def class_from_folder(name: str) -> Optional[str]:
    """'Damaged_Discoloured' / 'damaged-discoloured' / 'Shriveled' ... -> canonical class name (or None)."""
    key = re.sub(r"[^a-z]", "", name.lower())
    for prefix, cls in FOLDER_ALIASES:
        if key.startswith(prefix):
            return cls
    return None


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


def collect(orig_bgr: np.ndarray, grains: Sequence, scale: float) -> dict:
    """Raw per-grain measurements of ONE photo (no reference needed yet)."""
    lengths = np.array([max(g.rect[1]) * scale for g in grains], float)
    widths = np.array([min(g.rect[1]) * scale for g in grains], float)
    areas = np.array([g.area * scale * scale for g in grains], float)
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
    return {"lengths": lengths, "widths": widths, "areas": areas, "stats": stats, "patches": patches}


def photo_reference(raw: dict) -> dict:
    """The median grain of one photo (JSON-friendly)."""
    return {"length": max(float(np.median(raw["lengths"])), 1e-6),
            "width": max(float(np.median(raw["widths"])), 1e-6),
            "area": max(float(np.median(raw["areas"])), 1e-6),
            "hsv": np.median([s[0] for s in raw["stats"]], axis=0).tolist(),
            "lab": np.median([s[1] for s in raw["stats"]], axis=0).tolist()}


def merge_references(refs: List[dict]) -> dict:
    """One reference out of several photos (median of the per-photo medians)."""
    return {"length": float(np.median([r["length"] for r in refs])),
            "width": float(np.median([r["width"] for r in refs])),
            "area": float(np.median([r["area"] for r in refs])),
            "hsv": np.median([r["hsv"] for r in refs], axis=0).tolist(),
            "lab": np.median([r["lab"] for r in refs], axis=0).tolist()}


def features_from(raw: dict, grains: Sequence, scale: float, ref: dict) -> np.ndarray:
    """Feature matrix (n_grains x len(FEATURE_NAMES)) measured against the reference grain `ref`."""
    n = len(grains)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)))
    lengths, widths, areas, stats, patches = (raw[k] for k in ("lengths", "widths", "areas", "stats", "patches"))
    med_l, med_w, med_a = ref["length"], ref["width"], ref["area"]
    hsv_med, lab_med = np.array(ref["hsv"], float), np.array(ref["lab"], float)

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


def extract_features(orig_bgr: np.ndarray, grains: Sequence, scale: float, ref: Optional[dict] = None) -> np.ndarray:
    """Features for the grains of one photo. ``ref`` = the model's stored reference; None = this photo's median."""
    if len(grains) == 0:
        return np.zeros((0, len(FEATURE_NAMES)))
    raw = collect(orig_bgr, grains, scale)
    return features_from(raw, grains, scale, ref if ref else photo_reference(raw))


def load_legacy_samples(path: Path = LEGACY_SAMPLES):
    """Labelled grains saved by the old in-browser trainer (photo-normalised features). Returns (X, y, source_per_grain)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("feature_names") != FEATURE_NAMES:
        raise ValueError(f"{path} was written with different features and cannot be used.")
    samples = [s for s in data.get("samples", []) if s.get("label") in CLASSES]
    X = np.array([s["x"] for s in samples], float) if samples else np.zeros((0, len(FEATURE_NAMES)))
    return X, [s["label"] for s in samples], [s["source"] for s in samples]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_classifier(trees: int = 300, seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=trees, class_weight="balanced_subsample",
                                  min_samples_leaf=1, random_state=seed, n_jobs=-1)


class GrainModel:
    """Loads a model trained by train.py. The dashboard only calls load() / predict() / info()."""

    def __init__(self, path: Path = MODEL_FILE):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.clf = None
        self.meta: dict = {}
        self.error = ""
        self._stamp = None                 # (mtime, size) of the model file when it was last loaded
        self.load()

    # -- persistence --------------------------------------------------------
    def load(self) -> bool:
        """(Re)load the model file. Safe to call while the server is running (e.g. after re-training)."""
        clf, meta, error = None, {}, ""
        if self.path.is_file():
            try:
                import joblib
                blob = joblib.load(self.path)
                if blob.get("feature_names") == FEATURE_NAMES:
                    clf, meta = blob["clf"], blob["meta"]
                else:
                    error = "The saved model uses a different feature set. Train it again with: python train.py"
            except Exception as e:
                error = f"The model file could not be read ({type(e).__name__}). Train it again with: python train.py"
        with self.lock:
            self.clf, self.meta, self.error, self._stamp = clf, meta, error, self._file_stamp()
        return clf is not None

    def _file_stamp(self):
        try:
            st = self.path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def reload_if_changed(self) -> bool:
        """Re-read the model file when train.py has replaced (or removed) it since the last load."""
        if self._file_stamp() != self._stamp:
            self.load()
            return True
        return False

    def save(self):
        import joblib
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "meta": self.meta, "feature_names": FEATURE_NAMES}, self.path)

    # -- training (called from train.py, never from the dashboard) ---------
    def fit(self, X: np.ndarray, y: List[str], *, ref: Optional[dict], normalize: str, detect: dict,
            evaluation: Optional[dict], trees: int = 300, seed: int = 0, extra: Optional[dict] = None) -> dict:
        counts = Counter(y)
        if len(y) < MIN_SAMPLES or len(counts) < MIN_CLASSES:
            raise ValueError(f"Need at least {MIN_SAMPLES} grains covering at least {MIN_CLASSES} different types "
                             f"(found {len(y)} grains, {len(counts)} type{'' if len(counts) == 1 else 's'}).")
        clf = make_classifier(trees, seed).fit(X, np.array(y))
        order = np.argsort(clf.feature_importances_)[::-1][:10]
        meta = {"version": datetime.now().strftime("%Y%m%d%H%M%S"),
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "n_samples": int(len(y)), "counts": {c: counts.get(c, 0) for c in CLASSES},
                "classes": [str(c) for c in clf.classes_], "trees": trees,
                "normalize": normalize, "reference": ref, "detect": detect,
                "evaluation": evaluation,
                "top_features": [[FEATURE_NAMES[i], round(float(clf.feature_importances_[i]), 4)] for i in order],
                **(extra or {})}
        with self.lock:
            self.clf, self.meta, self.error = clf, meta, ""
            self.save()
            self._stamp = self._file_stamp()
        return meta

    # -- inference ----------------------------------------------------------
    @property
    def trained(self) -> bool:
        return self.clf is not None

    @property
    def reference(self) -> Optional[dict]:
        """Reference grain stored in the model (None = per-photo normalisation)."""
        return self.meta.get("reference") if self.meta.get("normalize") == "reference" else None

    @property
    def version(self) -> str:
        return self.meta.get("version") or ""

    def predict(self, X: np.ndarray) -> List[dict]:
        if not self.trained:
            raise RuntimeError("No trained model found. Run  python train.py  first.")
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
        base = {"file": str(self.path.relative_to(BASE)) if self.path.is_relative_to(BASE) else str(self.path),
                "file_exists": self.path.is_file(), "error": self.error}
        if not self.trained:
            return {**base, "trained": False, "version": None}
        m = self.meta
        return {**base, "trained": True, "version": m.get("version"), "trained_at": m.get("trained_at"),
                "n_samples": m.get("n_samples"), "counts": m.get("counts"), "classes": m.get("classes"),
                "normalize": m.get("normalize", "photo"), "detect": m.get("detect"),
                "evaluation": m.get("evaluation"), "top_features": m.get("top_features"),
                "n_photos": m.get("n_photos"), "source": m.get("source", "terminal")}
