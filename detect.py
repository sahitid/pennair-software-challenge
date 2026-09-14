"""
Shape detection for the PennAiR software challenge.

The backgrounds in this challenge (grass, gravel, whatever) are textured.
The shapes are not — even the ones with color gradients. So instead of
trying to name the background color, I score each pixel by how much
high-frequency texture is around it, and keep the smooth blobs.
"""

import math

import cv2
import numpy as np


def _odd(n):
    n = max(3, int(round(n)))
    return n if n % 2 == 1 else n + 1


def smoothness_map(bgr, sigma=None, win=None):
    """Per-pixel high-frequency energy. Textured bg is high, smooth shapes are low.

    I first tried local color stddev, which worked on the solid grass image
    but punched holes in the gradient shapes on the hard video (a ramp
    looks 'varied' to a std filter). Subtracting a Gaussian blur leaves
    only the grain, and a smooth ramp survives that.
    """
    h = bgr.shape[0]
    if sigma is None:
        sigma = 2.0 * (h / 1080.0)
        sigma = max(1.0, sigma)
    if win is None:
        win = _odd(15 * (h / 1080.0))
    acc = None
    for ch in cv2.split(bgr):
        f = ch.astype(np.float32)
        blur = cv2.GaussianBlur(f, (0, 0), sigma)
        d = f - blur
        energy = cv2.blur(d * d, (win, win))
        acc = energy if acc is None else acc + energy
    return acc / 3.0


def shape_mask(bgr):
    energy = smoothness_map(bgr)
    med = float(np.median(energy))
    # fraction of the background's typical energy. 0.42 was a compromise:
    # lower undershoots the real edge, higher lets gravel specks in.
    cut = max(0.42 * med, 3.0)
    mask = (energy < cut).astype(np.uint8) * 255

    s = bgr.shape[0] / 1080.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(11 * s), _odd(11 * s)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # residual energy spikes right on the edge, so the mask is a bit small.
    d = _odd(9 * s)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))

    # fill holes inside shapes (the white trapezoid's gradient fooled the
    # energy map in the middle a couple times)
    h, w = mask.shape
    flooded = mask.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flooded[y, x] == 0:
            cv2.floodFill(flooded, ff, (x, y), 128)
    mask[flooded == 0] = 255
    return mask


def _label(contour):
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if peri < 1 or area < 1:
        return "shape"
    circ = 4 * math.pi * area / (peri * peri)
    if circ > 0.85:
        return "circle"
    # hull first so grass-jagged outlines don't invent extra corners
    hull = cv2.convexHull(contour)
    hperi = cv2.arcLength(hull, True)
    n = None
    for eps in (0.04, 0.03, 0.05, 0.02):
        cand = len(cv2.approxPolyDP(hull, eps * hperi, True))
        if cand in (3, 4, 5):
            n = cand
            break
    if n == 3:
        return "triangle"
    if n == 4:
        pts = cv2.approxPolyDP(hull, 0.04 * hperi, True).reshape(-1, 2).astype(np.float32)
        if len(pts) != 4:
            return "quad"
        # rectangles have ~90 degree corners; trapezoids don't
        dots = []
        for i in range(4):
            v1 = pts[i] - pts[(i - 1) % 4]
            v2 = pts[(i + 1) % 4] - pts[i]
            n1 = np.linalg.norm(v1) + 1e-6
            n2 = np.linalg.norm(v2) + 1e-6
            dots.append(abs(np.dot(v1, v2) / (n1 * n2)))
        return "rectangle" if float(np.mean(dots)) < 0.22 else "trapezoid"
    if n == 5:
        return "pentagon"
    if circ > 0.80:
        return "circle"
    return "shape"


def _split_if_overlap(bgr, contour, min_area):
    """Two overlapping shapes make a concave blob. Gradient shapes stay convex.

    Only try to split when the outline caves in, otherwise k-means would
    cut the blue/yellow pentagon in half.
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1 or area / hull_area > 0.90:
        return [contour]

    m = np.zeros(bgr.shape[:2], np.uint8)
    cv2.drawContours(m, [contour], -1, 255, -1)
    ys, xs = np.where(m > 0)
    if len(xs) < 500:
        return [contour]

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    pts = lab[ys, xs].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _compact, labels, centers = cv2.kmeans(
        pts, 2, None, crit, 4, cv2.KMEANS_PP_CENTERS
    )
    sep = np.linalg.norm(centers[0] - centers[1])
    counts = np.bincount(labels.flatten(), minlength=2)
    if sep < 22 or counts.min() < 0.06 * counts.sum():
        return [contour]

    pieces = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for i in range(2):
        part = np.zeros_like(m)
        sel = labels.flatten() == i
        part[ys[sel], xs[sel]] = 255
        part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > min_area:
            pieces.append(c)
    return pieces if len(pieces) >= 2 else [contour]


def find_shapes(bgr):
    """Return a list of dicts: contour, center, name, area, radius, circularity."""
    h0, w0 = bgr.shape[:2]
    work = bgr
    up = 1.0
    # 1080p is overkill for finding blobs. half-res is about 4x cheaper
    # and I just scale the contours back up. (part 2 asked about cheaper ops)
    if w0 > 1400:
        up = 2.0
        work = cv2.resize(bgr, (w0 // 2, h0 // 2), interpolation=cv2.INTER_AREA)

    mask = shape_mask(work)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    min_area = 0.0018 * h * w
    max_area = 0.22 * h * w

    raw = []
    for c in cnts:
        a = cv2.contourArea(c)
        if min_area < a < max_area:
            raw.extend(_split_if_overlap(work, c, min_area))

    shapes = []
    for c in raw:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        peri = cv2.arcLength(c, True)
        circ = 4 * math.pi * a / (peri * peri + 1e-6)
        (_x, _y), r = cv2.minEnclosingCircle(c)
        if up != 1.0:
            c = (c.astype(np.float32) * up).astype(np.int32)
            cx *= up
            cy *= up
            r *= up
            a *= up * up
        shapes.append(
            {
                "contour": c,
                "center": (cx, cy),
                "name": _label(c),
                "area": a,
                "radius": r,
                "circ": circ,
            }
        )
    return shapes, mask


def draw(bgr, shapes):
    out = bgr.copy()
    h, w = out.shape[:2]
    for s in shapes:
        cnt = s["contour"]
        hull = cv2.convexHull(cnt)
        cv2.drawContours(out, [hull], -1, (0, 255, 255), 2)
        cx, cy = int(round(s["center"][0])), int(round(s["center"][1]))
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        label = f"{s['name']}  ({cx}, {cy})"
        tx = min(max(cx + 8, 4), w - 8)
        ty = max(cy - 8, 16)
        cv2.putText(
            out,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out
