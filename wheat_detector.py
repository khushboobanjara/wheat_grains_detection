"""
Core wheat-grain detection logic (classical computer vision only).

Pipeline
--------
1. Gaussian blur                     -> suppress sensor noise
2. BGR -> HSV, cv2.inRange           -> colour thresholding (wheat is golden/tan,
                                        text/numbers are black/white/low-saturation)
3. Morphology (open / close / hole fill) -> clean the binary mask
4. Contour detection                 -> candidate blobs
5. Watershed splitting               -> separate grains that touch each other
6. Shape filtering                   -> area, aspect ratio, solidity, size vs median
                                        (removes text, specks and unrelated objects)

No datasets, no deep learning, no pretrained models, no network access.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

MAX_SIDE = 1600            # images are processed at most this many pixels on the long side
GREEN = (0, 255, 0)        # BGR

# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255
PRESETS = {
    "Golden / amber wheat": dict(h=(8, 32), s=(50, 255), v=(80, 255)),
    "Pale / light wheat": dict(h=(8, 35), s=(25, 255), v=(110, 255)),
    "Dark / red wheat": dict(h=(0, 25), s=(60, 255), v=(40, 230)),
}


@dataclass
class Params:
    # HSV thresholds
    h_min: int = 8
    h_max: int = 32
    s_min: int = 50
    s_max: int = 255
    v_min: int = 80
    v_max: int = 255
    auto_saturation: bool = False   # Otsu threshold on S channel inside the hue band
    # Pre/post processing
    blur: int = 5                   # Gaussian kernel (odd; 1 disables)
    open_iter: int = 1
    close_iter: int = 1
    # Shape filters (pixels are measured on the processed, possibly down-scaled, image)
    min_area: int = 250
    max_area: int = 30000
    min_aspect: float = 1.1         # long side / short side of min-area rectangle
    max_aspect: float = 4.5
    min_solidity: float = 0.80      # contour area / convex hull area
    rel_min_pct: int = 35           # drop blobs smaller than this % of the median grain area
    # Touching-grain splitting
    split: bool = True
    split_level: float = 0.5        # fraction of the distance-transform peak kept as a seed


@dataclass
class Grain:
    contour: np.ndarray
    box: Tuple[int, int, int, int]                       # x, y, w, h (processed image)
    rect: Tuple[Tuple[float, float], Tuple[float, float], float]  # cv2.minAreaRect
    area: float
    aspect: float
    solidity: float


@dataclass
class Result:
    mask: np.ndarray
    grains: List[Grain] = field(default_factory=list)
    rejected: int = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resize_for_processing(bgr: np.ndarray, max_side: int = MAX_SIDE):
    """Down-scale large images. Returns (resized_image, scale_back_to_original)."""
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr, 1.0
    f = max_side / float(longest)
    small = cv2.resize(bgr, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
    return small, 1.0 / f


def _fill_small_holes(mask: np.ndarray, max_hole: float) -> np.ndarray:
    """Fill tiny holes inside grains (glare, creases) without bridging gaps between grains."""
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return mask
    out = mask.copy()
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 and cv2.contourArea(c) <= max_hole:
            cv2.drawContours(out, [c], -1, 255, thickness=-1)
    return out


def build_mask(bgr: np.ndarray, p: Params) -> np.ndarray:
    """HSV segmentation + thresholding + morphological clean-up -> binary mask (0/255)."""
    img = bgr
    if p.blur and p.blur > 1:
        k = p.blur if p.blur % 2 == 1 else p.blur + 1
        img = cv2.GaussianBlur(bgr, (k, k), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    s_min = p.s_min
    if p.auto_saturation:
        hue_ok = cv2.inRange(hsv[:, :, 0], p.h_min, p.h_max)
        vals = hsv[:, :, 1][hue_ok > 0]
        if vals.size > 200:
            t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            s_min = int(np.clip(t, 20, 150))

    lo = np.array([p.h_min, s_min, p.v_min], dtype=np.uint8)
    hi = np.array([p.h_max, p.s_max, p.v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if p.open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=p.open_iter)
    if p.close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=p.close_iter)
    mask = _fill_small_holes(mask, max_hole=max(30.0, p.min_area * 0.5))
    return mask


def _solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return float(area / hull_area) if hull_area > 0 else 0.0


def _split_contour(contour: np.ndarray, level: float) -> List[np.ndarray]:
    """Split a blob of touching grains with a distance-transform seeded watershed."""
    x, y, w, h = cv2.boundingRect(contour)
    pad = 3
    off = np.array([x - pad, y - pad])
    roi = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(roi, [contour - off], -1, 255, thickness=-1)

    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return [contour]
    # Start at the requested level; if the seeds are still joined (grains that overlap
    # along their length), raise the level so the seeds shrink until they separate.
    n, fg, markers = 0, None, None
    for lv in np.arange(level, 0.96, 0.1):
        _, fg = cv2.threshold(dist, float(lv) * dist.max(), 255, cv2.THRESH_BINARY)
        fg = fg.astype(np.uint8)
        n, markers = cv2.connectedComponents(fg)
        if n > 2:
            break
    if n <= 2:                       # still one seed -> nothing to split
        return [contour]

    markers = markers + 1            # background of ROI = 1, seeds = 2..n
    markers[(roi > 0) & (fg == 0)] = 0   # unknown region to be flooded
    markers = cv2.watershed(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))

    parts = []
    for lab in range(2, n + 1):
        part = ((markers == lab) * 255).astype(np.uint8)
        cs, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            parts.append(max(cs, key=cv2.contourArea) + off)
    return parts or [contour]


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def detect(bgr: np.ndarray, p: Params) -> Result:
    mask = build_mask(bgr, p)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    big = [c for c in cnts if cv2.contourArea(c) >= p.min_area]
    ref = float(np.median([cv2.contourArea(c) for c in big])) if big else 0.0

    # --- split touching grains ------------------------------------------------
    pieces: List[np.ndarray] = []
    for c in big:
        a = cv2.contourArea(c)
        if p.split and (a > 1.6 * ref or a > p.max_area or _solidity(c) < 0.88):
            pieces.extend(_split_contour(c, p.split_level))
        else:
            pieces.append(c)

    # --- shape filtering ------------------------------------------------------
    grains: List[Grain] = []
    rejected = len(cnts) - len(big)
    for c in pieces:
        a = cv2.contourArea(c)
        if a < p.min_area or a > p.max_area or len(c) < 5:
            rejected += 1
            continue
        rect = cv2.minAreaRect(c)
        long_s, short_s = max(rect[1]), min(rect[1])
        if short_s < 2:
            rejected += 1
            continue
        aspect = long_s / short_s
        sol = _solidity(c)
        if not (p.min_aspect <= aspect <= p.max_aspect) or sol < p.min_solidity:
            rejected += 1
            continue
        grains.append(Grain(c, cv2.boundingRect(c), rect, float(a), float(aspect), sol))

    # --- size consistency: grains of one sample have similar areas ------------
    if len(grains) >= 5 and p.rel_min_pct > 0:
        med = float(np.median([g.area for g in grains]))
        kept = [g for g in grains if p.rel_min_pct / 100.0 * med <= g.area <= 2.5 * med]
        rejected += len(grains) - len(kept)
        grains = kept

    grains.sort(key=lambda g: (g.box[1] + g.box[3] / 2, g.box[0] + g.box[2] / 2))
    return Result(mask=mask, grains=grains, rejected=rejected)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def annotate(original_bgr: np.ndarray, grains: List[Grain], scale: float = 1.0,
             show_ids: bool = False, tilted: bool = False) -> np.ndarray:
    """Draw one green box per grain on a copy of the (full resolution) original."""
    out = original_bgr.copy()
    th = max(2, int(round(max(out.shape[:2]) / 500)))
    fs = max(0.5, th * 0.35)
    for i, g in enumerate(grains, 1):
        if tilted:
            pts = (cv2.boxPoints(g.rect) * scale).astype(np.int32)
            cv2.polylines(out, [pts], True, GREEN, th, cv2.LINE_AA)
            tx, ty = int(pts[:, 0].min()), int(pts[:, 1].min())
        else:
            x, y, w, h = [int(round(v * scale)) for v in g.box]
            cv2.rectangle(out, (x, y), (x + w, y + h), GREEN, th)
            tx, ty = x, y
        if show_ids:
            cv2.putText(out, str(i), (tx, max(ty - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, GREEN, max(1, th // 2), cv2.LINE_AA)
    return out


def grains_to_csv(grains: List[Grain], scale: float = 1.0) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "x", "y", "width", "height", "area_px", "aspect_ratio", "solidity"])
    for i, g in enumerate(grains, 1):
        x, y, bw, bh = [int(round(v * scale)) for v in g.box]
        w.writerow([i, x, y, bw, bh, round(g.area * scale * scale, 1),
                    round(g.aspect, 3), round(g.solidity, 3)])
    return buf.getvalue()


def encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
