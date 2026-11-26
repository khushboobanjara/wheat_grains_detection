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
    # Segmentation method: "background" finds ANYTHING that differs from the background
    # (every grain colour: green, golden, dark brown, pale, broken). "colour" uses the HSV
    # range above and only finds grains whose colour falls inside it.
    method: str = "background"
    bg_sens: float = 0.7            # background mode: lower = more sensitive (finds fainter grains)
    # Pre/post processing
    blur: int = 5                   # Gaussian kernel (odd; 1 disables)
    open_iter: int = 1
    close_iter: int = 1
    # Shape filters (pixels are measured on the processed, possibly down-scaled, image)
    min_area: int = 120
    max_area: int = 30000
    min_aspect: float = 1.0         # long side / short side of min-area rectangle
    max_aspect: float = 4.5
    min_solidity: float = 0.70      # contour area / convex hull area
    rel_min_pct: int = 15           # drop blobs smaller than this % of the median grain area
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


def build_mask_colour(bgr: np.ndarray, p: Params) -> np.ndarray:
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


def _background_score(bgr: np.ndarray, blur: int) -> np.ndarray:
    """Per-pixel 'how different from the background' score (colour independent)."""
    img = bgr
    if blur and blur > 1:
        k = blur if blur % 2 == 1 else blur + 1
        img = cv2.GaussianBlur(bgr, (k, k), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    # Estimate the background (paper / table, incl. uneven lighting) with a big median
    # filter on a down-scaled copy; grains are small compared with the kernel.
    f = 4 if max(h, w) > 600 else 2
    small = cv2.resize(lab, (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    k = int(max(15, min(small.shape[:2]) // 3)) | 1
    k = min(k, 101)
    bg = np.stack([cv2.medianBlur(np.clip(small[:, :, i], 0, 255).astype(np.uint8), k)
                   for i in range(3)], -1).astype(np.float32)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    d = lab - bg
    chroma = np.sqrt(d[:, :, 1] ** 2 + d[:, :, 2] ** 2)     # colour shift: grains are tan/green/brown
    dark = np.clip(-d[:, :, 0], 0, None)                    # darker than the background
    # chroma weighs more than darkness so soft shadows are not mistaken for grains
    return np.sqrt((1.6 * chroma) ** 2 + (0.8 * dark) ** 2)


def _refine_blobs(mask: np.ndarray, s8: np.ndarray, min_area: int) -> np.ndarray:
    """Tighten oversized blobs (several grains + shadow/halo fused together).

    Inside each blob that is much larger than a typical blob, a local Otsu threshold on the
    difference score keeps the strongly different pixels (grain bodies) and drops the faint
    ones (soft shadows, halos) that glue neighbouring grains together.
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]
    good = areas[areas >= min_area]
    if good.size == 0:
        return mask
    ref = float(np.median(good))
    out = mask.copy()
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 1.6 * ref:
            continue
        x, y, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        blob = lab[y:y + h, x:x + w] == i
        vals = s8[y:y + h, x:x + w][blob]
        t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        core = ((s8[y:y + h, x:x + w] > t) & blob).astype(np.uint8) * 255
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, k3)
        if core.sum() / 255 >= 0.45 * a:             # only accept if most of the blob survives
            out[y:y + h, x:x + w][blob] = core[blob]
    return out


def build_mask_background(bgr: np.ndarray, p: Params) -> np.ndarray:
    score = _background_score(bgr, p.blur)
    s8 = np.clip(score * 4, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(t * p.bg_sens, 16, 200))           # never below noise level
    mask = (s8 > thr).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if p.open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=p.open_iter)
    if p.close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=p.close_iter)
    mask = _fill_small_holes(mask, max_hole=max(30.0, p.min_area * 0.5))
    if p.split:
        mask = _refine_blobs(mask, s8, p.min_area)
    return mask


def build_mask(bgr: np.ndarray, p: Params) -> np.ndarray:
    """Dispatch to the chosen segmentation method."""
    if p.method == "background":
        return build_mask_background(bgr, p)
    return build_mask_colour(bgr, p)


def _solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return float(area / hull_area) if hull_area > 0 else 0.0


def _split_contour(contour: np.ndarray, level: float, grain_w: float = 0.0,
                   min_piece: float = 0.0) -> List[np.ndarray]:
    """Split a blob of touching grains with a seeded watershed.

    Seeds are local maxima of the smoothed distance transform that are at least about one
    grain-width apart (``grain_w`` = typical grain short side measured from isolated grains).
    This separates dense piles that a single global threshold on the distance map cannot.
    """
    x, y, w, h = cv2.boundingRect(contour)
    pad = 3
    off = np.array([x - pad, y - pad])
    roi = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(roi, [contour - off], -1, 255, thickness=-1)

    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return [contour]
    sm = cv2.GaussianBlur(dist, (0, 0), max(1.0, (grain_w or 12.0) * 0.08))

    if grain_w <= 0:
        grain_w = 2.0 * float(dist.max())
    sep = max(3, int(round(grain_w * (0.35 + 0.5 * level))))      # min distance between seeds
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sep + 1, 2 * sep + 1))
    peaks = (sm >= cv2.dilate(sm, k) - 1e-6) & (sm >= 0.30 * grain_w / 2.0) & (roi > 0)
    n, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    if n <= 2:                       # one seed -> nothing to split
        return [contour]

    markers = markers + 1            # ROI background = 1, seeds = 2..n+1
    markers[roi == 0] = 1
    markers[(roi > 0) & (peaks == 0)] = 0
    # Height map: low in the grain centres, rising towards the edges, and a hard wall (255)
    # outside the blob so the background marker cannot flood into the grains.
    height = (1.0 - sm / max(float(sm.max()), 1e-6)) * 150.0
    height[roi == 0] = 255.0
    flood = cv2.cvtColor(height.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(flood, markers.astype(np.int32))

    # Merge slivers (pieces far smaller than a grain) back into the neighbour they touch most.
    if min_piece > 0:
        for _ in range(3):
            changed = False
            for lab in range(2, n + 1):
                m = markers == lab
                a = int(m.sum())
                if a == 0 or a >= min_piece:
                    continue
                ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                neigh = markers[ring & ~m]
                neigh = neigh[(neigh >= 2) & (neigh != lab)]
                if neigh.size:
                    markers[m] = int(np.bincount(neigh).argmax())
                    changed = True
            if not changed:
                break
        # watershed lines (-1) belong to the blob too: give them to a neighbouring label
        lines = (markers == -1) & (roi > 0)
        if lines.any():
            filled = cv2.dilate(np.where(markers >= 2, markers, 0).astype(np.float32), np.ones((3, 3), np.uint8))
            markers[lines] = filled[lines].astype(np.int32)

    parts = []
    for lab in range(2, n + 1):
        part = ((markers == lab) * 255).astype(np.uint8)
        cs, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            parts.append(max(cs, key=cv2.contourArea) + off)
    return parts or [contour]


def _typical_grain(cnts: List[np.ndarray], min_area: float) -> Tuple[float, float]:
    """(typical area, typical short side) from the blobs that look like single grains."""
    areas, shorts = [], []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area * 2 or len(c) < 5:
            continue
        if _solidity(c) < 0.9:
            continue
        r = cv2.minAreaRect(c)
        if min(r[1]) < 2 or max(r[1]) / min(r[1]) > 4.5:
            continue
        areas.append(a)
        shorts.append(min(r[1]))
    if len(areas) >= 3:
        return float(np.median(areas)), float(np.median(shorts))
    return 0.0, 0.0


def _drop_duplicates(grains: List[Grain], contain: float = 0.8) -> List[Grain]:
    """Remove a grain whose box lies (almost) entirely inside a larger grain's box."""
    order = sorted(range(len(grains)), key=lambda i: -grains[i].area)
    keep: List[int] = []
    for i in order:
        x, y, w, h = grains[i].box
        dup = False
        for j in keep:
            X, Y, W, H = grains[j].box
            iw = min(x + w, X + W) - max(x, X)
            ih = min(y + h, Y + H) - max(y, Y)
            if iw > 0 and ih > 0 and iw * ih >= contain * w * h:
                dup = True
                break
        if not dup:
            keep.append(i)
    return [grains[i] for i in sorted(keep)]


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def detect(bgr: np.ndarray, p: Params) -> Result:
    mask = build_mask(bgr, p)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    big = [c for c in cnts if cv2.contourArea(c) >= p.min_area]
    ref, grain_w = _typical_grain(big, p.min_area)
    if ref <= 0:                                   # no isolated grain to learn from
        ref = float(np.median([cv2.contourArea(c) for c in big])) if big else 0.0
        grain_w = 0.0

    # --- split touching grains (twice: a pile may need a second pass) ----------
    def needs_split(c) -> bool:
        a = cv2.contourArea(c)
        return p.split and (a > 1.5 * ref or a > p.max_area or _solidity(c) < 0.88)

    def split_recursive(c, level: float, depth: int = 0) -> List[np.ndarray]:
        """Split, then re-split any piece that is still too big with more sensitive seeds."""
        parts = _split_contour(c, level, grain_w, 0.3 * ref)
        if len(parts) <= 1 or depth >= 3:
            return parts
        out: List[np.ndarray] = []
        for part in parts:
            if needs_split(part) and level > 0.05:
                out.extend(split_recursive(part, level - 0.2, depth + 1))
            else:
                out.append(part)
        return out

    pieces: List[np.ndarray] = []
    for c in big:
        if needs_split(c):
            parts = split_recursive(c, p.split_level)
            if len(parts) <= 1 and p.split_level > 0.05:      # first try found nothing: be more sensitive
                parts = split_recursive(c, max(0.0, p.split_level - 0.3))
            pieces.extend(parts)
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
        if not (p.min_aspect <= aspect <= p.max_aspect):
            rejected += 1
            continue
        # Ragged shapes (letters, digits, dirt) are rejected, but a grain-sized piece that is only
        # slightly ragged (cut by the watershed, creased, glare) is still a grain: keep it.
        grain_sized = ref > 0 and 0.4 * ref <= a <= 1.8 * ref
        if sol < p.min_solidity and not (grain_sized and sol >= 0.5):
            rejected += 1
            continue
        grains.append(Grain(c, cv2.boundingRect(c), rect, float(a), float(aspect), sol))

    # --- size consistency: drop specks much smaller than a normal grain ------------
    # (no upper limit: a big blob is more likely several grains we failed to split than
    #  a non-grain, and silently losing grains is worse than a slightly large box)
    if len(grains) >= 5 and p.rel_min_pct > 0:
        full = [g.area for g in grains if g.area >= 0.5 * (ref or np.median([x.area for x in grains]))]
        med = float(np.median(full)) if full else float(np.median([g.area for g in grains]))
        kept = [g for g in grains if g.area >= p.rel_min_pct / 100.0 * med]
        rejected += len(grains) - len(kept)
        grains = kept

    grains = _drop_duplicates(grains)
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