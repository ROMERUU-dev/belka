"""Geometry of the develop module: lens corrections, straighten and perspective.

Two stages, applied in this order to the oriented camera image:

* ``lens_correct``: radial distortion and lens vignetting about the centre
  (section "lens").
* ``warp``: the Transform panel (keystone, rotate, aspect, scale, offsets)
  followed by the crop's straighten ``angle``, as one homography (section
  "transform"). The angle goes last, as Lightroom's crop does, so adding the
  result of ``angle_from_line`` to it levels a line drawn on the shown image.

Coordinates. Normalised points are (u, v) in [0, 1] across width and height,
y down. The maths runs in continuous pixel coordinates (pixel i spans
[i, i + 1]); ``homography`` converts to OpenCV's convention (pixel centres on
integers) so it goes straight into ``cv2.warpPerspective``. "Source" is the
oriented image before the lens correction, where ``pipeline.analyze`` measures
the film; "output" is the transformed image the crop lives in.

Keystone is a virtual camera turned about the picture centre (tilt for
Vertical, pan for Horizontal) with a fixed focal length: that is what makes
converging lines parallel. The result is re-centred and re-scaled so the
middle of the picture stays put and keeps its width (Vertical) or height
(Horizontal) while the sliders move; only the foreshortening is undone.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import cv2
import numpy as np

from belka.core.film import FilmProfile
from belka.core.pipeline import DevelopSettings

FOCAL = 1.0  # virtual focal length / long side: a 36 mm lens on 35 mm film
MAX_TILT_TAN = 1.0  # tan of the virtual tilt (or pan) at persp_vertical = ±1
HORIZON_MARGIN = 1.1  # the vanishing line stays this far beyond the corners
MAX_ASPECT = 1.5  # stretch at persp_aspect = ±1
MAX_K1 = 0.25  # |k1| at lens_distortion = ±1, with r = 1 at the corners
VIGNETTE_STOPS = 1.5  # corner exposure change of the print at lens_vignette = ±1

UPRIGHT_SIDE = 1200  # long side the line detection runs at
LINE_TOL = 30.0  # degrees a line may lean and still count as horizontal/vertical
MIN_LINE_LENGTH = 0.1  # total line length needed, as a fraction of the long side
INLIER_DEG = 1.0  # a line points at a vanishing point within this angle
VP_SHARE = 0.25  # ...and the lines doing so carry at least this share of the length
MIN_LINE_SPREAD = 0.03  # distinct lines lie this far apart (fraction of the long side)
MIN_VP_LINES = 4  # distinct lines a vanishing point needs: two or three short edges meet anywhere
PEAK_RATIO = 4.0  # how far the agreed lean must stand above an even spread
AUTO_MAX_TILT = 0.6  # "auto" keeps its keystone below this slider value...
AUTO_MAX_ANGLE = 15.0  # ...and trusts no rotation larger than this

GUIDE_MIN_LENGTH = 0.02  # shorter guides (fraction of the long side) are ignored, like one just started
GUIDE_SAME_LINE = 0.01  # guides whose ends all lie this close to each other's line are one line

_OFF = {"angle": 0.0, "persp_vertical": 0.0, "persp_horizontal": 0.0}


# ---------------------------------------------------------------- state

def _perspective_on(s: DevelopSettings) -> bool:
    return s.section_on("transform")


def _transform_on(s: DevelopSettings) -> bool:
    # The straighten angle belongs to the crop, as in Lightroom: switching the
    # Transform panel off drops the perspective sliders, not the angle.
    if abs(float(s.angle)) > 1e-9:
        return True
    if not _perspective_on(s):
        return False
    values = (s.persp_vertical, s.persp_horizontal, s.persp_rotate, s.persp_aspect,
              s.persp_scale - 1.0, s.persp_x, s.persp_y)
    return any(abs(float(v)) > 1e-9 for v in values)


def _k1(s: DevelopSettings) -> float:
    return -MAX_K1 * float(s.lens_distortion) if s.section_on("lens") else 0.0


def _vignette(s: DevelopSettings) -> float:
    return float(s.lens_vignette) if s.section_on("lens") else 0.0


def is_identity(s: DevelopSettings) -> bool:
    return not _transform_on(s) and _k1(s) == 0.0 and _vignette(s) == 0.0


# ---------------------------------------------------------------- homography

def _translate(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _scale(sx: float, sy: float) -> np.ndarray:
    return np.diag([sx, sy, 1.0])


def _rotate(degrees: float) -> np.ndarray:
    """Positive turns clockwise on screen (y points down)."""
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    xy = pts @ m[:2, :2].T + m[:2, 2]
    return xy / (pts @ m[2, :2] + m[2, 2])[..., None]


def _keystone_tangents(pv: float, ph: float, w: int, h: int) -> tuple[float, float]:
    """tan of the virtual tilt and pan for the Vertical/Horizontal sliders.

    Both near their ends at once would bring the vanishing line into the
    picture (points behind the virtual camera), so they are scaled down
    together until it stays HORIZON_MARGIN beyond the corners.
    """
    tv, th = float(pv) * MAX_TILT_TAN, float(ph) * MAX_TILT_TAN
    f = FOCAL * max(w, h)
    limit = (f / (HORIZON_MARGIN * 0.5 * np.hypot(w, h))) ** 2
    # Distance of the vanishing line from the centre is f / sqrt(load).
    load = th * th + tv * tv * (1.0 + th * th)
    if load > limit:
        a, b = (th * tv) ** 2, th * th + tv * tv
        u = limit / b if a < 1e-12 else (np.sqrt(b * b + 4.0 * a * limit) - b) / (2.0 * a)
        tv, th = tv * float(np.sqrt(u)), th * float(np.sqrt(u))
    return tv, th


def _keystone(tv: float, th: float, f: float) -> np.ndarray:
    """Virtual tilt (tv > 0 widens the top) and pan (th > 0 enlarges the right side).

    Centred coordinates. The centre is kept in place. A turned camera
    magnifies the centre by 1/cos along both axes and once more along the
    turn's own axis (the foreshortening); dividing by the cube root of the
    local area takes out the first part only, so the middle row (tilt) or
    column (pan) keeps its length and the foreshortening is undone.
    """
    if tv == 0.0 and th == 0.0:
        return np.eye(3)
    ct, st = 1.0 / np.hypot(1.0, tv), tv / np.hypot(1.0, tv)
    cp, sp = 1.0 / np.hypot(1.0, th), th / np.hypot(1.0, th)
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    pan = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    k = np.diag([f, f, 1.0])
    m = k @ tilt @ pan @ np.linalg.inv(k)
    m = _translate(-m[0, 2] / m[2, 2], -m[1, 2] / m[2, 2]) @ m
    jacobian = m[:2, :2] / m[2, 2]
    g = abs(np.linalg.det(jacobian)) ** (-1.0 / 3.0)
    return _scale(g, g) @ m


def _forward(s: DevelopSettings, w: int, h: int) -> np.ndarray | None:
    """Source → output homography in continuous pixel coordinates (None = identity)."""
    if not _transform_on(s):
        return None
    if not _perspective_on(s):
        s = replace(s, persp_vertical=0.0, persp_horizontal=0.0, persp_rotate=0.0, persp_aspect=0.0,
                    persp_scale=1.0, persp_x=0.0, persp_y=0.0)
    tv, th = _keystone_tangents(s.persp_vertical, s.persp_horizontal, w, h)
    a = float(s.persp_aspect)
    # As in Lightroom, positive makes the picture taller and negative wider.
    sx, sy = (1.0, MAX_ASPECT ** a) if a >= 0 else (MAX_ASPECT ** -a, 1.0)
    k = max(float(s.persp_scale), 0.05)
    centred = (_rotate(s.angle) @ _translate(s.persp_x * w, -s.persp_y * h) @ _scale(k * sx, k * sy)
               @ _rotate(s.persp_rotate) @ _keystone(tv, th, FOCAL * max(w, h)))
    c = _translate(w / 2.0, h / 2.0)
    return c @ centred @ np.linalg.inv(c)


def homography(s: DevelopSettings, width: int, height: int) -> np.ndarray:
    """3×3 from an output pixel to the pixel of ``warp``'s input (OpenCV centres).

    Use with ``cv2.WARP_INVERSE_MAP``. The input is the lens-corrected image:
    the lens stage is not part of this matrix.
    """
    fwd = _forward(s, width, height)
    if fwd is None:
        return np.eye(3)
    m = _translate(-0.5, -0.5) @ np.linalg.inv(fwd) @ _translate(0.5, 0.5)
    return m / m[2, 2]


# ---------------------------------------------------------------- lens

def _distort(pts: np.ndarray, k1: float, w: int, h: int) -> np.ndarray:
    """Corrected → source: r_src = r (1 + k1 r²), r = 1 at the corners."""
    c = np.array([w / 2.0, h / 2.0])
    radius = 0.5 * np.hypot(w, h)
    q = (pts - c) / radius
    r2 = np.sum(q * q, axis=-1, keepdims=True)
    return c + q * (1.0 + k1 * r2) * radius


def _undistort(pts: np.ndarray, k1: float, w: int, h: int) -> np.ndarray:
    """Source → corrected, by Newton on the radius.

    With k1 < 0 the model folds back past r = sqrt(-1 / 3 k1); source points
    beyond the fold have no corrected position and are left on it, which is
    outside the frame anyway.
    """
    c = np.array([w / 2.0, h / 2.0])
    radius = 0.5 * np.hypot(w, h)
    q = (pts - c) / radius
    rho = np.sqrt(np.sum(q * q, axis=-1))
    r_max = np.sqrt(-1.0 / (3.0 * k1)) if k1 < 0 else np.inf
    r = rho.copy()
    for _ in range(20):
        slope = np.maximum(1.0 + 3.0 * k1 * r * r, 1e-6)
        r = np.clip(r - (r + k1 * r ** 3 - rho) / slope, 0.0, r_max)
    ratio = np.divide(r, rho, out=np.ones_like(rho), where=rho > 0)
    return c + q * ratio[..., None] * radius


def _lens_maps(k1: float, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    f = 0.5 * np.hypot(w, h)
    cam = np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])
    return cv2.initUndistortRectifyMap(cam, np.array([k1, 0.0, 0.0, 0.0]), None, cam, (w, h), cv2.CV_32FC1)


def _vignette_gain(amount: float, w: int, h: int, profile: FilmProfile | None) -> np.ndarray:
    """Per-pixel gain, smooth enough to compute at 1/8 size and upsample."""
    stops = VIGNETTE_STOPS * amount
    if profile is not None:
        # The gain acts on the light through the film, before the inversion.
        # A print gains one stop when the capture changes by gamma stops,
        # the other way round on a negative (thinner film prints darker).
        gamma = profile.gamma[1] or 0.6
        stops *= -gamma if profile.is_negative else gamma
    sw, sh = max(w // 8, 2), max(h // 8, 2)
    radius = 0.5 * np.hypot(w, h)
    xs = ((np.arange(sw) + 0.5) * (w / sw) - w / 2.0) / radius
    ys = ((np.arange(sh) + 0.5) * (h / sh) - h / 2.0) / radius
    r2 = ys[:, None] ** 2 + xs[None, :] ** 2
    small = np.exp2(stops * (0.5 * r2 + 0.5 * r2 * r2)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _resampled(img: np.ndarray, out: np.ndarray) -> np.ndarray:
    """OpenCV drops a trailing 1-channel axis; cubic overshoot must not go negative."""
    out = out.reshape(img.shape)
    if out.dtype.kind == "f":
        np.maximum(out, 0.0, out=out)
    return out


def lens_correct(img: np.ndarray, s: DevelopSettings, profile: FilmProfile | None = None) -> np.ndarray:
    """Radial distortion and lens vignetting; returns ``img`` itself when off.

    ``lens_vignette`` > 0 brightens the corners of the final print. Pass the
    film ``profile`` when ``img`` is the capture of a negative or slide so
    the gain is converted through the film (for a negative it darkens the
    capture's corners); without it the gain brightens ``img`` directly.
    Outside the picture (pincushion correction) the result is black.
    """
    k1, amount = _k1(s), _vignette(s)
    if k1 == 0.0 and amount == 0.0:
        return img
    h, w = img.shape[:2]
    out = img
    if k1 != 0.0:
        mx, my = _lens_maps(k1, w, h)
        out = _resampled(img, cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0))
    if amount != 0.0:
        gain = _vignette_gain(amount, w, h, profile)
        if out.ndim == 3:
            gain = gain[..., None]
        out = out * gain if out is img else np.multiply(out, gain, out=out)
    return out


# ---------------------------------------------------------------- warp

def _valid_mask(s: DevelopSettings, w: int, h: int) -> np.ndarray:
    """Fraction of each output pixel that comes from inside the picture."""
    k1 = _k1(s)
    if not _transform_on(s) and k1 <= 0.0:
        return np.ones((h, w), dtype=np.float32)
    mask = np.full((h, w), 255, dtype=np.uint8)
    if k1 > 0.0:
        mask = cv2.remap(mask, *_lens_maps(k1, w, h), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    if _transform_on(s):
        mask = cv2.warpPerspective(mask, homography(s, w, h), (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return mask.astype(np.float32) / 255.0


def warp(img: np.ndarray, s: DevelopSettings, with_mask: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Straighten and perspective, same size; black outside the picture.

    Returns ``img`` itself when the transform is off. The optional mask
    (float32, 1 inside, 0 outside, soft at the edge) also covers what the lens
    stage left black, so it describes the whole geometry.
    """
    h, w = img.shape[:2]
    out = img
    if _transform_on(s):
        out = _resampled(img, cv2.warpPerspective(img, homography(s, w, h), (w, h),
                                                  flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    if not with_mask:
        return out
    return out, _valid_mask(s, w, h)


# ---------------------------------------------------------------- mapping

def map_points_to_source(pts_norm, s: DevelopSettings, w: int, h: int) -> np.ndarray:
    """Output (transformed) normalised points → source normalised points, shape (..., 2)."""
    p = np.asarray(pts_norm, dtype=np.float64) * (w, h)
    fwd = _forward(s, w, h)
    if fwd is not None:
        p = _apply(np.linalg.inv(fwd), p)
    k1 = _k1(s)
    if k1 != 0.0:
        p = _distort(p, k1, w, h)
    return p / (w, h)


def map_points_from_source(pts_norm, s: DevelopSettings, w: int, h: int) -> np.ndarray:
    """Source normalised points → output (transformed) normalised points, shape (..., 2)."""
    p = np.asarray(pts_norm, dtype=np.float64) * (w, h)
    k1 = _k1(s)
    if k1 != 0.0:
        p = _undistort(p, k1, w, h)
    fwd = _forward(s, w, h)
    if fwd is not None:
        p = _apply(fwd, p)
    return p / (w, h)


def _turn_point(x: float, y: float, op: str) -> tuple[float, float]:
    """A normalised point under a screen rotation ("cw", "ccw") or mirror ("flip_h", "flip_v")."""
    return {"cw": (1.0 - y, x), "ccw": (y, 1.0 - x), "flip_h": (1.0 - x, y), "flip_v": (x, 1.0 - y)}[op]


def _turn_rect(rect, op: str):
    if rect is None:
        return None
    (ax, ay), (bx, by) = _turn_point(rect[0], rect[1], op), _turn_point(rect[2], rect[3], op)
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def reorient(s: DevelopSettings, op: str) -> dict:
    """Settings changes that turn the developed picture with the image when it is
    rotated 90° ("cw", "ccw") or mirrored ("flip_h", "flip_v") on screen.

    Orientation is applied before the warp, so the straighten and perspective
    sliders must follow it: the new transform is the old one conjugated by
    the turn (F' = Q F Q⁻¹). Mirrors are exact. A quarter turn swaps the
    Vertical and Horizontal keystones, which is exact when only one of them is
    set (the keystone applies tilt before pan, and the turn reverses that
    order). The crop and the Upright guides turn with the picture.
    """
    changes: dict = {"crop": _turn_rect(s.crop, op),
                     "upright_guides": tuple(_turn_point(g[0], g[1], op) + _turn_point(g[2], g[3], op)
                                             for g in s.upright_guides)}
    if op == "flip_h":
        changes.update(angle=-s.angle, persp_rotate=-s.persp_rotate, persp_horizontal=-s.persp_horizontal,
                       persp_x=-s.persp_x)
    elif op == "flip_v":
        changes.update(angle=-s.angle, persp_rotate=-s.persp_rotate, persp_vertical=-s.persp_vertical,
                       persp_y=-s.persp_y)
    elif op == "cw":
        changes.update(persp_vertical=-s.persp_horizontal, persp_horizontal=s.persp_vertical,
                       persp_x=s.persp_y, persp_y=-s.persp_x, persp_aspect=-s.persp_aspect)
    elif op == "ccw":
        changes.update(persp_vertical=s.persp_horizontal, persp_horizontal=-s.persp_vertical,
                       persp_x=-s.persp_y, persp_y=s.persp_x, persp_aspect=-s.persp_aspect)
    else:
        raise ValueError(op)
    # -0.0 would show as "-0" on the sliders.
    return {k: (v + 0.0 if isinstance(v, float) else v) for k, v in changes.items()}


def valid_polygon(s: DevelopSettings, w: int, h: int, samples: int = 48) -> np.ndarray:
    """Outline of the area that holds picture, (N, 2) normalised output points in order.

    Convex, and not clipped to the frame (for drawing the image edge while
    cropping). Its edges are straight unless a pincushion correction curves
    them, then each side is sampled ``samples`` times.
    """
    k1 = _k1(s)
    if k1 > 0.0:
        t = np.arange(samples) / samples
        zero, one = np.zeros_like(t), np.ones_like(t)
        sides = [(t, zero), (one, t), (1.0 - t, one), (zero, 1.0 - t)]
        pts = np.concatenate([np.stack([x * w, y * h], axis=1) for x, y in sides])
        pts = _undistort(pts, k1, w, h)
    else:
        pts = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    fwd = _forward(s, w, h)
    if fwd is not None:
        pts = _apply(fwd, pts)
    return pts / (w, h)


def _sides(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Start points and vectors of a polygon's non-empty edges, and twice its signed area (the winding)."""
    d = np.roll(poly, -1, axis=0) - poly
    keep = np.hypot(d[:, 0], d[:, 1]) > 1e-12
    pts, d = poly[keep], d[keep]
    if len(pts) < 3:
        return None
    area = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    return pts, d, area


def _inward(d: np.ndarray, area: float) -> np.ndarray:
    normals = np.sign(area) * np.stack([-d[:, 1], d[:, 0]], axis=1)
    return normals / np.hypot(normals[:, 0], normals[:, 1])[:, None]


def _halfplanes(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Inward unit normals and offsets (inside: n · p >= offset) of a convex polygon's edges."""
    found = _sides(poly)
    if found is None:
        return None
    pts, d, area = found
    normals = _inward(d, area)
    return normals, np.sum(normals * pts, axis=1)


def _fit_rect(planes: tuple[np.ndarray, np.ndarray] | None, w: int, h: int, aspect: float | None,
              box: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """Largest axis-aligned rectangle inside the half-planes ``planes`` and ``box``.

    For a fixed aspect this is a linear programme in (centre, size); a tiny
    pull towards the box centre picks the centred one among equals. A free
    aspect is searched for the largest area.
    """
    if planes is None:
        return None
    normals, offsets = planes
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def solve(ratio: float) -> tuple[float, tuple[float, float, float, float]] | None:
        # Variables (px, py, t, u, v): centre, height, distances to the box centre.
        a, b = 0.5 * ratio * h / w, 0.5
        m = np.abs(normals[:, 0]) * a + np.abs(normals[:, 1]) * b
        rows = [np.column_stack([-normals, m, np.zeros((len(m), 2)), -offsets])]
        rows.append(np.array([
            [-1, 0, a, 0, 0, -x0], [1, 0, a, 0, 0, x1], [0, -1, b, 0, 0, -y0], [0, 1, b, 0, 0, y1],
            [1, 0, 0, -1, 0, cx], [-1, 0, 0, -1, 0, -cx], [0, 1, 0, 0, -1, cy], [0, -1, 0, 0, -1, -cy],
        ], dtype=np.float64))
        status, z = cv2.solveLP(np.array([[0.0, 0.0, 1.0, -1e-3, -1e-3]]), np.vstack(rows))
        if status < 0:
            return None
        px, py, t = z.ravel()[:3]
        if t <= 1e-9:
            return None
        return ratio * t * t, (px - a * t, py - b * t, px + a * t, py + b * t)

    if aspect is not None:
        found = solve(float(aspect))
    else:
        # Area against log aspect: a coarse scan, then golden-section refinement.
        base = np.log(w / h)
        grid = base + np.linspace(-2.5, 2.5, 21)
        scores = [solve(float(np.exp(g))) for g in grid]
        areas = [sc[0] if sc else -1.0 for sc in scores]
        best = int(np.argmax(areas))
        if areas[best] <= 0:
            return None
        lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
        gold = (np.sqrt(5.0) - 1.0) / 2.0
        found = scores[best]
        for _ in range(24):
            c1, c2 = hi - gold * (hi - lo), lo + gold * (hi - lo)
            s1, s2 = solve(float(np.exp(c1))), solve(float(np.exp(c2)))
            a1, a2 = (s1[0] if s1 else -1.0), (s2[0] if s2 else -1.0)
            if a1 >= a2:
                hi = c2
            else:
                lo = c1
            for cand in (s1, s2):
                if cand and cand[0] > found[0]:
                    found = cand
    if found is None:
        return None
    rx0, ry0, rx1, ry1 = found[1]
    return (float(max(rx0, x0)), float(max(ry0, y0)), float(min(rx1, x1)), float(min(ry1, y1)))


def largest_valid_rect(s: DevelopSettings, w: int, h: int, aspect: float | None = None,
                       within: tuple[float, float, float, float] | None = None) -> tuple[float, float, float, float]:
    """Largest crop (x0, y0, x1, y1, normalised) with picture everywhere: Lightroom's "Constrain crop".

    ``aspect`` is the crop's width / height in pixels (None: the largest area
    of any shape). ``within`` limits the search to a box. When nothing of the
    picture is left in view, a zero-size rectangle at the box centre.
    """
    box = (0.0, 0.0, 1.0, 1.0) if within is None else tuple(float(np.clip(v, 0.0, 1.0)) for v in within)
    rect = _fit_rect(_halfplanes(valid_polygon(s, w, h)), w, h, aspect, box)
    if rect is None:
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        return (cx, cy, cx, cy)
    return rect


def _inner_halfplanes(poly: np.ndarray, per_side: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Half-planes of a convex region inside ``poly``, four sides of ``per_side`` points each.

    A convex outline is used as it is. A barrel correction bows a rectangle's
    sides inwards; each side then becomes the line along its chord through
    its innermost point, so the region never reaches past the curve.
    """
    found = _sides(poly)
    if found is None:
        return None
    pts, d, area = found
    turns = d[:, 0] * np.roll(d[:, 1], -1) - d[:, 1] * np.roll(d[:, 0], -1)
    if np.all(turns * np.sign(area) >= -1e-12 * np.abs(area)):
        return _halfplanes(poly)
    sides = poly.reshape(4, per_side, 2)
    chords = np.roll(sides[:, 0], -1, axis=0) - sides[:, 0]
    normals = _inward(chords, area)
    return normals, np.max(np.einsum("sk,snk->sn", normals, sides), axis=1)


def map_rect_from_source(rect_norm, s: DevelopSettings, w: int, h: int,
                         aspect: float | str | None = "frame") -> tuple[float, float, float, float] | None:
    """Largest rectangle inside a source rectangle (the detected frame) once it is transformed.

    A straightened or keystoned frame is a tilted quadrilateral; only the
    rectangle inscribed in it is free of the film rebate around the picture.
    ``aspect``: "frame" keeps the source rectangle's own shape, a number is
    width / height in pixels, None the largest area of any shape. None when
    nothing of the rectangle is left in view.
    """
    if rect_norm is None:
        return None
    x0, x1 = sorted(min(max(float(v), 0.0), 1.0) for v in (rect_norm[0], rect_norm[2]))
    y0, y1 = sorted(min(max(float(v), 0.0), 1.0) for v in (rect_norm[1], rect_norm[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    k1 = _k1(s)
    if not _transform_on(s) and k1 == 0.0:
        return (x0, y0, x1, y1)
    if aspect == "frame":
        aspect = (x1 - x0) * w / ((y1 - y0) * h)
    # Straight sides stay straight under the homography; the lens curves them.
    per_side = 1 if k1 == 0.0 else 32
    t = np.arange(per_side) / per_side
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    outline = np.concatenate([np.column_stack([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
                              for a, b in zip(corners, corners[1:] + corners[:1])])
    poly = map_points_from_source(outline, s, w, h)
    return _fit_rect(_inner_halfplanes(poly, per_side), w, h, aspect, (0.0, 0.0, 1.0, 1.0))


# ---------------------------------------------------------------- upright

def _lean(dx, dy, vertical) -> np.ndarray:
    """Clockwise lean in degrees of directions (dx, dy) from horizontal, or from vertical where ``vertical``."""
    a = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 180.0 - 90.0
    return np.where(vertical, np.where(a > 0, a - 90.0, a + 90.0), a)


def angle_from_line(p0, p1) -> float:
    """Degrees to add to ``angle`` so the line p0–p1 becomes level or plumb, whichever is nearer.

    Points in pixels (or any units with the same scale on both axes), y down.
    """
    dx, dy = float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])
    if dx == 0.0 and dy == 0.0:
        return 0.0
    vertical = abs(dy) > abs(dx)
    return float(-_lean(dx, dy, vertical))


def _line_segments(img: np.ndarray, region) -> np.ndarray:
    """Straight edges as (N, 4) x0, y0, x1, y1 in continuous pixels of ``img``.

    Canny and probabilistic Hough on log luminance at UPRIGHT_SIDE, then each
    segment refitted to its own edge pixels, which is far finer than the
    Hough angle step.
    """
    h, w = img.shape[:2]
    k = min(1.0, UPRIGHT_SIDE / max(h, w))
    sw, sh = max(int(round(w * k)), 1), max(int(round(h * k)), 1)
    small = img.astype(np.float32, copy=False)
    if k < 1.0:
        small = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_AREA)
    lum = small.reshape(sh, sw, -1).mean(axis=2)
    peak = float(np.percentile(lum, 99.5))
    if peak <= 0:
        return np.zeros((0, 4))
    log = np.log2(np.maximum(lum, peak * 2.0 ** -12))
    lo, hi = np.percentile(log, [0.5, 99.5])
    gray = np.clip((log - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    # Edges are what stands well above both the busiest texture and the noise.
    mag = np.hypot(gx, gy)
    strong = max(float(np.percentile(mag, 92)), 6.0 * float(np.median(mag)), 24.0)
    edges = cv2.Canny(gray, 0.4 * strong, strong, L2gradient=True)
    if region is not None:
        x0, y0, x1, y1 = region
        ix, iy = 0.02 * (x1 - x0), 0.02 * (y1 - y0)  # stay off the frame's own border
        keep = np.zeros_like(edges)
        keep[int((y0 + iy) * sh):int(np.ceil((y1 - iy) * sh)), int((x0 + ix) * sw):int(np.ceil((x1 - ix) * sw))] = 1
        edges *= keep
    side = max(sw, sh)
    found = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=30, minLineLength=0.04 * side, maxLineGap=0.005 * side)
    if found is None:
        return np.zeros((0, 4))
    segs = found.reshape(-1, 4).astype(np.float64)
    segs = segs[np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))[:600]]
    refined = np.array([_refit(edges, seg) for seg in segs])
    return (refined + 0.5) * np.array([w / sw, h / sh, w / sw, h / sh])


def _refit(edges: np.ndarray, seg: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = seg
    xa, xb = int(max(min(x0, x1) - 2, 0)), int(min(max(x0, x1) + 3, edges.shape[1]))
    ya, yb = int(max(min(y0, y1) - 2, 0)), int(min(max(y0, y1) + 3, edges.shape[0]))
    ys, xs = np.nonzero(edges[ya:yb, xa:xb])
    pts = np.stack([xs + xa, ys + ya], axis=1).astype(np.float64)
    p0, d = np.array([x0, y0]), np.array([x1 - x0, y1 - y0])
    length = np.hypot(*d)
    if length == 0 or len(pts) < 8:
        return seg
    d /= length
    rel = pts - p0
    along = rel @ d
    across = rel @ np.array([-d[1], d[0]])
    near = (np.abs(across) < 1.5) & (along > -2) & (along < length + 2)
    if np.count_nonzero(near) < 8:
        return seg
    pts = pts[near]
    centre = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    u = vecs[:, 1]
    ends = centre + np.outer([(p0 - centre) @ u, (p0 + d * length - centre) @ u], u)
    return ends.ravel()


def _robust_lean(lean: np.ndarray, weight: np.ndarray, min_weight: float) -> float | None:
    """The lean most line length agrees on: kernel mode, then a trimmed mean.

    None when the lines are too few or do not agree (the mode must stand
    PEAK_RATIO times above an even spread), as with texture or noise.
    """
    if weight.sum() < min_weight:
        return None
    grid = np.arange(-LINE_TOL, LINE_TOL + 1e-9, 0.05)
    density = (weight[:, None] * np.exp(-0.5 * ((grid[None, :] - lean[:, None]) / 0.35) ** 2)).sum(axis=0)
    if density.max() < PEAK_RATIO * density.mean():
        return None
    est = float(grid[np.argmax(density)])
    for tol in (1.0, 0.5):
        near = np.abs(lean - est) < tol
        if weight[near].sum() < 0.5 * min_weight:
            return None
        est = float(np.average(lean[near], weights=weight[near]))
    return est


def _distinct_lines(offsets: np.ndarray) -> int:
    """How many separate lines a set of (normalised) line offsets holds; pieces of one line count once."""
    if len(offsets) == 0:
        return 0
    return 1 + int(np.sum(np.diff(np.sort(offsets)) > MIN_LINE_SPREAD))


def _vanishing_point(segs: np.ndarray, weight: np.ndarray, scale: float,
                     min_weight: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Common point of the lines (homogeneous, centred pixels) and its inlier mask, or None.

    Candidates are the crossings of pairs of the longest distinct lines; the
    one most line length points at (within INLIER_DEG) wins and is refined by
    least squares on those lines. A point at infinity means the lines are
    parallel. Pieces of a single line (a row of window sills) fix no point,
    so the inliers must spread across the picture, and they must be at least
    MIN_VP_LINES separate lines: the few short edges of sprocket holes or a
    film border meet at a spurious nearby point that would pull the keystone
    to its limit.
    """
    if len(segs) < 2 or weight.sum() < min_weight:
        return None
    p0 = np.column_stack([segs[:, :2] / scale, np.ones(len(segs))])
    p1 = np.column_stack([segs[:, 2:] / scale, np.ones(len(segs))])
    lines = np.cross(p0, p1)
    lines /= np.hypot(lines[:, 0], lines[:, 1])[:, None]
    lines *= np.where(lines[:, 0] + lines[:, 1] < 0, -1.0, 1.0)[:, None]  # same side for near-parallel lines
    mid = (p0[:, :2] + p1[:, :2]) / 2.0
    direction = p1[:, :2] - p0[:, :2]
    direction /= np.hypot(direction[:, 0], direction[:, 1])[:, None]
    tol = np.sin(np.radians(INLIER_DEG))

    def misfit(v: np.ndarray) -> np.ndarray:
        """Sine of the angle between each line and the way to v, for (M, 3) candidates."""
        to_v = v[:, None, :2] - mid[None, :, :] * v[:, None, 2:3]
        norm = np.hypot(to_v[..., 0], to_v[..., 1])
        cross = np.abs(direction[None, :, 0] * to_v[..., 1] - direction[None, :, 1] * to_v[..., 0])
        return np.where(norm > 1e-12, cross / np.maximum(norm, 1e-12), 1.0)

    top = np.argsort(-weight)[:30]
    pairs = np.array(list(combinations(top, 2)))
    li, lj = lines[pairs[:, 0]], lines[pairs[:, 1]]
    distinct = np.abs(li[:, 2] - lj[:, 2]) > MIN_LINE_SPREAD
    cand = np.cross(li[distinct], lj[distinct])
    norm = np.linalg.norm(cand, axis=1)
    cand = cand[norm > 1e-12] / norm[norm > 1e-12, None]
    if len(cand) == 0:
        return None
    scores = ((misfit(cand) < tol) * weight[None, :]).sum(axis=1)
    v = cand[int(np.argmax(scores))]
    for _ in range(3):
        inl = misfit(v[None])[0] < tol
        if (weight[inl].sum() < max(min_weight, VP_SHARE * weight.sum())
                or _distinct_lines(lines[inl, 2]) < MIN_VP_LINES):
            return None
        lw = lines[inl] * np.sqrt(weight[inl])[:, None]
        _, vecs = np.linalg.eigh(lw.T @ lw)
        v = vecs[:, 0]
    return np.array([v[0] * scale, v[1] * scale, v[2]]), misfit(v[None])[0] < tol


def _tilt_pan_through(vp_v: np.ndarray, vp_h: np.ndarray, f: float) -> tuple[float, float] | None:
    """Keystone tangents whose vanishing line runs through both vanishing points.

    The keystone sends the line -tan(pan) x + tan(tilt)/cos(pan) y + f = 0 to
    infinity, so matching it to the line through the two points makes both
    families of lines parallel.
    """
    a, b, d = np.cross(vp_v, vp_h)
    if abs(d) < 1e-9 * f * max(np.hypot(a, b), 1e-12):
        return None
    th = -f * a / d
    tv = f * b / d / np.hypot(1.0, th)
    return float(tv), float(th)


def _plumb_angle(key: np.ndarray, vp_v: np.ndarray) -> float:
    """Straighten angle that turns the verticals' direction after ``key`` upright."""
    v = key @ vp_v
    return -float(_lean(v[0], v[1], True))


def _full_tangents(vp_v: np.ndarray, segs: np.ndarray, weight: np.ndarray, w: int, h: int) -> tuple[float, float] | None:
    """Tilt and pan for "full": verticals parallel and plumb, horizontals as level as possible.

    With the virtual focal length equal to the lens's, one keystone makes both
    families parallel and square. Otherwise forcing both parallel leaves a
    shear, so only the verticals are kept exact (the tilt follows from the
    pan) and the pan minimises the length-weighted squared lean of the
    horizontal lines ``segs`` (centred pixels): exact when the lens matches,
    balanced when it does not.
    """
    f = FOCAL * max(w, h)
    if abs(vp_v[1]) < 1e-12:
        return None

    def tilt(th: float) -> float:
        # Keeps the vertical vanishing point on the keystone's vanishing line.
        return float((th * vp_v[0] - f * vp_v[2]) / vp_v[1] / np.hypot(1.0, th))

    def cost(th: float) -> float:
        key = _keystone(*_keystone_tangents(tilt(th) / MAX_TILT_TAN, th / MAX_TILT_TAN, w, h), f)
        m = _rotate(_plumb_angle(key, vp_v)) @ key
        d = _apply(m, segs[:, 2:]) - _apply(m, segs[:, :2])
        return float(np.sum(weight * _lean(d[:, 0], d[:, 1], False) ** 2))

    grid = np.linspace(-MAX_TILT_TAN, MAX_TILT_TAN, 101)
    best = int(np.argmin([cost(th) for th in grid]))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    gold = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(40):
        c1, c2 = hi - gold * (hi - lo), lo + gold * (hi - lo)
        if cost(c1) <= cost(c2):
            hi = c2
        else:
            lo = c1
    th = 0.5 * (lo + hi)
    return tilt(th), float(th)


def _upright_result(angle: float, pv: float, ph: float) -> dict:
    # Adding 0.0 turns a rounded -0.0 into 0.0, which the sliders would show as "-0".
    return {"angle": round(float(angle), 2) + 0.0, "persp_vertical": round(float(pv), 3) + 0.0,
            "persp_horizontal": round(float(ph), 3) + 0.0}


def auto_upright(img_linear: np.ndarray, mode: str, region=None) -> dict:
    """Lightroom's Upright: {angle, persp_vertical, persp_horizontal}, or {} without enough lines.

    ``img_linear`` is the oriented, lens-corrected image with no transform;
    the result replaces those three settings and assumes the other Transform
    sliders (rotate, aspect) at zero. ``region`` (x0, y0, x1, y1, normalised)
    limits the lines to the picture, e.g. the detected frame, so the film
    edges and sprocket holes do not vote. Modes: "level" (rotation only),
    "vertical" (+ vertical keystone), "full" (+ horizontal keystone), "auto"
    (level and vertical within moderate limits), "off" (all zero).

    "full" squares both families only when the taking lens matches the
    virtual one (FOCAL); otherwise a shear is left, a few degrees of lean on
    the horizontals. Taking it out would need Rotate near 45° (the shear's
    principal axes) or, within Rotate's ±10°, a 10-20 % Aspect change, so
    the verticals are kept exact and the horizontals as level as possible.
    """
    if mode == "off":
        return dict(_OFF)
    h, w = img_linear.shape[:2]
    if min(h, w) < 32:
        return {}
    segs = _line_segments(img_linear, region)
    if len(segs) == 0:
        return {}
    segs = segs - np.array([w, h, w, h]) / 2.0
    long_side = float(max(w, h))
    f = FOCAL * long_side
    min_weight = MIN_LINE_LENGTH * long_side
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    a = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 180.0 - 90.0
    horizontal = np.abs(a) <= LINE_TOL
    vertical = np.abs(a) >= 90.0 - LINE_TOL
    lean = _lean(dx, dy, vertical)

    # Level: converging verticals each lean their own way, so the ones that
    # meet at a vanishing point vote with the direction to it from the
    # picture centre instead (that is the camera's roll).
    found_v = _vanishing_point(segs[vertical], length[vertical], f, min_weight)
    votes, voters = lean.copy(), horizontal | vertical
    if found_v is not None:
        vp_v, inliers = found_v
        cx, cy = 0.0, 0.0
        if region is not None:
            cx, cy = (region[0] + region[2] - 1.0) * w / 2.0, (region[1] + region[3] - 1.0) * h / 2.0
        idx = np.nonzero(vertical)[0]
        votes[idx[inliers]] = _lean(vp_v[0] - cx * vp_v[2], vp_v[1] - cy * vp_v[2], True)
        voters[idx[~inliers]] = False
    level = _robust_lean(votes[voters], length[voters], min_weight)

    def only_level() -> dict:
        if level is None or (mode == "auto" and abs(level) > AUTO_MAX_ANGLE):
            return {}
        return _upright_result(-level, 0.0, 0.0)

    if mode == "level" or found_v is None:
        return only_level()
    tangents = _tilt_pan_through(vp_v, np.array([1.0, 0.0, 0.0]), f)
    if mode == "full":
        found_h = _vanishing_point(segs[horizontal], length[horizontal], f, min_weight)
        if found_h is not None:
            family = found_h[1]
            tangents = _full_tangents(vp_v, segs[horizontal][family], length[horizontal][family], w, h) or tangents
    if tangents is None:
        return only_level()
    limit = AUTO_MAX_TILT if mode == "auto" else 1.0
    pv = round(float(np.clip(tangents[0] / MAX_TILT_TAN, -limit, limit)), 3)
    ph = round(float(np.clip(tangents[1] / MAX_TILT_TAN, -1.0, 1.0)), 3) if mode == "full" else 0.0
    # Whatever lean the keystone leaves is taken out by the rotation, measured
    # through the same matrix the warp will use.
    angle = _plumb_angle(_keystone(*_keystone_tangents(pv, ph, w, h), f), vp_v)
    if mode == "auto" and abs(angle) > AUTO_MAX_ANGLE:
        return only_level()
    return _upright_result(angle, pv, ph)


# ---------------------------------------------------------------- guided upright

def _guide_leans(pts: np.ndarray, vertical: np.ndarray, pv: float, ph: float, w: int, h: int) -> np.ndarray:
    """Lean in degrees of each guide ((N, 2, 2) centred corrected pixels) after the keystone for (pv, ph)."""
    q = _apply(_keystone(*_keystone_tangents(pv, ph, w, h), FOCAL * max(w, h)), pts)
    d = q[:, 1] - q[:, 0]
    return _lean(d[:, 0], d[:, 1], vertical)


def _one_line(pts: np.ndarray, tol: float) -> bool:
    """Whether the guides ((N, 2, 2) pixels) lie on one line: every end within ``tol`` of every guide's line."""
    d = pts[:, 1] - pts[:, 0]
    normals = np.stack([-d[:, 1], d[:, 0]], axis=1) / np.hypot(d[:, 0], d[:, 1])[:, None]
    off = np.einsum("ik,jek->ije", normals, pts) - np.sum(normals * pts[:, 0], axis=1)[:, None, None]
    return bool(np.abs(off).max() < tol)


def _meeting_point(pts: np.ndarray, f: float) -> np.ndarray:
    """Least-squares common point (homogeneous, centred pixels) of the lines through the guides."""
    p = np.concatenate([pts / f, np.ones(pts.shape[:2] + (1,))], axis=2)
    lines = np.cross(p[:, 0], p[:, 1])
    lines /= np.hypot(lines[:, 0], lines[:, 1])[:, None]
    _, vecs = np.linalg.eigh(lines.T @ lines)
    v = vecs[:, 0]
    return np.array([v[0] * f, v[1] * f, v[2]])


def _settle(residuals, p: np.ndarray, free: list[int]) -> np.ndarray:
    """Least squares over the ``free`` entries of ``p``, kept within [-1, 1].

    Gauss-Newton with a numeric Jacobian, halving steps that do not help.
    Where the residuals leave a direction free (one guide of each kind), the
    pseudo-inverse steps only across it, so from no keystone it reaches the
    nearest keystone that works rather than wandering along the free one.
    """
    r, eps = residuals(p), 1e-6
    for _ in range(30):
        jac = np.empty((len(r), len(free)))
        for j, k in enumerate(free):
            q = p.copy()
            q[k] += eps
            jac[:, j] = (residuals(q) - r) / eps
        step = np.linalg.lstsq(jac, -r, rcond=None)[0]
        t = 1.0
        while t > 1e-3:
            q = p.copy()
            q[free] = np.clip(p[free] + t * step, -1.0, 1.0)
            rq = residuals(q)
            if rq @ rq < r @ r:
                break
            t *= 0.5
        else:
            break
        done = np.abs(q - p).max() < 1e-4  # well below the 0.001 the sliders keep
        p, r = q, rq
        if done:
            break
    return p


def guided_upright(guides, s: DevelopSettings, w: int, h: int) -> dict:
    """Lightroom's Guided Upright: {angle, persp_vertical, persp_horizontal} that make the guides plumb or level.

    ``guides``: up to four lines (x0, y0, x1, y1) normalised in source
    coordinates (as ``map_points_to_source``), drawn along edges that should
    stand straight up or run straight across; each one is whichever of the two
    it is nearer to as shown under ``s``. ``w``, ``h``: the oriented image
    (only its shape matters). One guide only rotates. Guides of one kind add
    that kind's keystone. Both kinds use both keystones: by least squares when
    the guides ask more than the three settings can give (two of each square
    up exactly only when the taking lens matches the virtual one, as in
    ``auto_upright``'s "full"), and one of each with the keystone that squares
    them most directly. Lens correction, scale and offsets stay as in ``s``;
    rotate and aspect are assumed zero, as for ``auto_upright``. {} when no
    guide is long enough, or when two or more guides of a kind lie on one line.
    """
    size = np.array([w, h], dtype=np.float64)
    long_side = float(max(w, h))
    ends = np.asarray(guides, dtype=np.float64).reshape(-1, 2, 2)
    # Straight on the shown image means straight once the lens is corrected.
    k1 = _k1(s)
    pts = (_undistort(ends * size, k1, w, h) if k1 != 0.0 else ends * size) - size / 2.0
    span = np.hypot(*(pts[:, 1] - pts[:, 0]).T)
    keep = np.isfinite(span) & (span >= GUIDE_MIN_LENGTH * long_side)
    if not keep.any():
        return {}
    ends, pts = ends[keep], pts[keep]
    shown = map_points_from_source(ends, s, w, h) * size
    d = shown[:, 1] - shown[:, 0]
    vertical = np.abs(d[:, 1]) > np.abs(d[:, 0])
    nv, nh = int(vertical.sum()), int((~vertical).sum())
    for n, kind in ((nv, vertical), (nh, ~vertical)):
        if n >= 2 and _one_line(pts[kind], GUIDE_SAME_LINE * long_side):
            return {}

    p = np.zeros(2)
    mixed = nv > 0 and nh > 0
    free = [k for k, on in ((0, nv >= 2 or mixed), (1, nh >= 2 or mixed)) if on]
    if free:
        # Start from the keystone that sends where the guides of each kind
        # meet to infinity; the solver then takes care of one-axis keystones,
        # clamping and a taking lens unlike the virtual one.
        f = FOCAL * long_side
        vp_v = _meeting_point(pts[vertical], f) if nv >= 2 else np.array([0.0, 1.0, 0.0])
        vp_h = _meeting_point(pts[~vertical], f) if nh >= 2 else np.array([1.0, 0.0, 0.0])
        start = np.array(_tilt_pan_through(vp_v, vp_h, f) or (0.0, 0.0)) / MAX_TILT_TAN
        p[free] = np.clip(start[free], -1.0, 1.0)

        def residuals(q: np.ndarray) -> np.ndarray:
            # The best rotation for a keystone is minus the mean lean, so what
            # is left is the spread of the leans about their mean.
            lean = _guide_leans(pts, vertical, q[0], q[1], w, h)
            return lean - lean.mean()

        p = _settle(residuals, p, free)
    pv, ph = (round(float(v), 3) for v in p)
    # The rotation is measured through the rounded keystone the warp will use.
    angle = -float(np.mean(_guide_leans(pts, vertical, pv, ph, w, h)))
    return _upright_result(np.clip(angle, -45.0, 45.0), pv, ph)
