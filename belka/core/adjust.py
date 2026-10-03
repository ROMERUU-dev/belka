"""Lightroom-style adjustments on the rendered image (NumPy + OpenCV, no Qt).

``pipeline.render`` hands over display-referred, sRGB-encoded RGB; everything
a Lightroom develop panel does after the basic tone happens here, in the
order the contract fixes. Colour work is done in OKLab/OKLCh (Ottosson 2020),
whose lightness, chroma and hue are perceptually independent: a tone move
does not shift hue and a hue move does not change lightness.

Speed: per-pixel work runs in horizontal strips on a thread pool (NumPy and
OpenCV release the GIL, and a strip's working set stays in cache), while
the independent whole-image filters of a stage run side by side on the same
pool. Wide filters work on a subsampled copy, and every per-pixel curve is a
table read with ``cv2.remap``. Tables depend only on settings and are
cached, so dragging one slider rebuilds only its own. Radii are given in
pixels of the full-resolution image and multiplied by ``scale``, so a
preview looks like the downsampled export.
"""

from __future__ import annotations

import colorsys
import functools
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from belka.core.pipeline import DevelopSettings

# (name, OKLCh hue in degrees). The centres are the OKLab hues of moderately
# saturated sRGB colours at Lightroom's band hues, so each band grabs what a
# Lightroom user expects (skin falls between red and orange, mostly orange).
HSL_BANDS: list[tuple[str, float]] = [
    ("Rojo", 20.0),
    ("Naranja", 58.0),
    ("Amarillo", 105.0),
    ("Verde", 143.0),
    ("Aguamarina", 195.0),
    ("Azul", 262.0),
    ("Púrpura", 300.0),
    ("Magenta", 335.0),
]
HSL_HUE_DEGREES = 30.0  # hue shift of a band at ±1
SKIN_HUE = 50.0  # OKLCh hue of skin, which vibrance leaves alone

HS_SIGMA_FRACTION = 0.02  # highlights/shadows adapt over 2 % of the long side
HS_RANGE = 0.05  # ...and only to neighbours within about this OKLab lightness
TEXTURE_SIGMAS = (1.5, 6.0)  # band-pass, full-resolution pixels
TEXTURE_FINE_MIN = 0.6  # pixels of the image at hand: a narrower sampled Gaussian is nearly all centre tap
TEXTURE_EPS = 0.03 ** 2
CLARITY_SIGMA = 50.0
CLARITY_EPS = 0.005
SHARPEN_MIN_SIGMA = 0.55  # pixels of the image at hand, as TEXTURE_FINE_MIN
GRADE_CHROMA = 0.08  # OKLab chroma of a colour-grading tint at saturation 1
GRADE_LUM = 0.12
GRAIN_STRENGTH = 0.045
# Colour noise reduction keeps a pixel whose colour is this many noise
# levels away from the local mean.
_NR_STANDOUT = 2.0

_EPS = 1e-6
# Pixels per strip task. Every NumPy call takes the GIL to start, so with
# small strips the threads mostly queue for it; with much larger ones a
# strip no longer stays in cache. Fixed, so results are bit-identical
# whatever the CPU count.
_STRIP_PIXELS = 90000
# Fine 1-D tables are read at the nearest entry: with this many, rounding
# moves a value by less than 1e-4, and a nearest read costs a third of a
# bilinear one. Tables over OKLab lightness are smooth and can be shorter.
_N = 16384
_N_L = 4096
_REMAP_MAX = 32000  # remap wants maps and tables narrower than 32767 pixels
_REMAP_BLOCK = 1 << 16  # remap parallelises calls larger than this
_LMS_FLOOR = np.float32(1e-30)  # cv2.log of 0 is -inf, and very slow

# Linear sRGB -> LMS and LMS^(1/3) -> OKLab.
_M1_64 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2_64 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M1 = _M1_64.astype(np.float32)
_M2_ROWS = [np.ascontiguousarray(_M2_64[i:i + 1], dtype=np.float32) for i in range(3)]
_M1_INV = np.linalg.inv(_M1_64).astype(np.float32)
_M2_INV = np.linalg.inv(_M2_64).astype(np.float32)

# Threads start on first use; shared by every call.
_POOL_NAME = "belka-adjust"
_POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1), thread_name_prefix=_POOL_NAME)


# ---------------------------------------------------------------- helpers

def _srgb_decode64(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb_encode64(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def _oklab64(rgb: np.ndarray) -> np.ndarray:
    """OKLab of sRGB-encoded colours (..., 3), in float64: for building tables."""
    lms = _srgb_decode64(np.asarray(rgb, dtype=np.float64)) @ _M1_64.T
    return np.cbrt(lms) @ _M2_64.T


def _in_pool() -> bool:
    return threading.current_thread().name.startswith(_POOL_NAME)


def _rows(fn, h: int, w: int) -> None:
    """Run ``fn(slice)`` over horizontal strips of an (h, w) image in parallel
    (one after another inside a pool job, which must not wait on the pool)."""
    step = max(1, _STRIP_PIXELS // max(w, 1))
    strips = [slice(y, min(y + step, h)) for y in range(0, h, step)]
    if len(strips) == 1 or _in_pool():
        for sl in strips:
            fn(sl)
        return
    for future in [_POOL.submit(fn, sl) for sl in strips]:
        future.result()


def _together(*jobs: Callable[[], object]) -> list:
    """Results of independent whole-image jobs, run at once: the first on
    this thread (its strips can still use the pool), the rest as pool jobs."""
    if _in_pool():
        return [job() for job in jobs]
    futures = [_POOL.submit(job) for job in jobs[1:]]
    results = [jobs[0]()] if jobs else []
    return results + [f.result() for f in futures]


def _smoothstep(e0: float, e1: float, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------- table reads

def _frozen(table: np.ndarray) -> np.ndarray:
    """Tables are cached and shared between calls and threads: read-only."""
    table.flags.writeable = False
    return table


def _tables(*rows) -> np.ndarray:
    """1-D tables as one remap source, a row each."""
    return _frozen(np.ascontiguousarray(np.stack(rows), dtype=np.float32))


def _remap(table: np.ndarray, x: np.ndarray, y: np.ndarray, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    """``table`` read at (x, y) for every pixel; positions off the edges clamp.

    Called in blocks of at most _REMAP_BLOCK pixels, which remap runs on the
    calling thread instead of waking OpenCV's pool against ours.
    """
    h, w = x.shape
    if w > _REMAP_MAX:
        return np.concatenate([_remap(table, np.ascontiguousarray(x[:, i:i + _REMAP_MAX]),
                                      np.ascontiguousarray(y[:, i:i + _REMAP_MAX]), interpolation)
                               for i in range(0, w, _REMAP_MAX)], axis=1)
    step = max(1, _REMAP_BLOCK // w)
    if h <= step:
        return cv2.remap(table, x, y, interpolation, borderMode=cv2.BORDER_REPLICATE)
    out = np.empty((h, w) + table.shape[2:], dtype=np.float32)
    for r in range(0, h, step):
        cv2.remap(table, x[r:r + step], y[r:r + step], interpolation, dst=out[r:r + step],
                  borderMode=cv2.BORDER_REPLICATE)
    return out


@functools.lru_cache(maxsize=16)
def _row_index(h: int, w: int, rows: int) -> np.ndarray:
    """Shared, read-only y map sending channel c of each pixel to table row c
    (a strip has few distinct shapes)."""
    y = np.tile(np.arange(rows, dtype=np.float32), (h, w))
    y.flags.writeable = False
    return y


def _lut(coord: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Nearest entry of ``table`` at ``coord`` (in entries), channel by channel.

    ``table`` has a row per channel of ``coord`` (h, w, c), or one row that
    serves every channel; out-of-range positions take the end entries.
    """
    h = coord.shape[0]
    flat = coord.reshape(h, -1)
    rows = table.shape[0]
    return _remap(table, flat, _row_index(h, flat.shape[1] // rows, rows)).reshape(coord.shape)


_GRID = np.linspace(0.0, 1.0, _N)
_DECODE = _tables(_srgb_decode64(_GRID))
# Linear light is looked up by its square root: that spends the table's
# resolution where sRGB encoding is steepest, near black.
_ENCODE = _tables(_srgb_encode64(_GRID * _GRID))


def _decode(enc: np.ndarray) -> np.ndarray:
    return _lut(enc * np.float32(_N - 1), _DECODE)


def _sqrt_index(lin: np.ndarray) -> np.ndarray:
    coord = np.maximum(lin, np.float32(0.0))
    cv2.sqrt(coord, dst=coord)
    coord *= np.float32(_N - 1)
    return coord


def _encode(lin: np.ndarray) -> np.ndarray:
    return _lut(_sqrt_index(lin), _ENCODE)


def _lin_to_lab(lin: np.ndarray, L: np.ndarray, a: np.ndarray, b: np.ndarray, sl: slice) -> None:
    lms = cv2.transform(lin, _M1)
    np.maximum(lms, _LMS_FLOOR, out=lms)
    # Cube root as exp(log(x) / 3): three times faster than cv2.pow.
    cv2.log(lms, dst=lms)
    lms *= np.float32(1.0 / 3.0)
    cv2.exp(lms, dst=lms)
    for plane, row in zip((L, a, b), _M2_ROWS):
        cv2.transform(lms, row, dst=plane[sl])


def _lab_to_lin(L: np.ndarray, a: np.ndarray, b: np.ndarray, sl: slice) -> np.ndarray:
    lms = cv2.transform(cv2.merge([L[sl], a[sl], b[sl]]), _M2_INV)
    cv2.pow(lms, 3.0, dst=lms)
    return cv2.transform(lms, _M1_INV)


# ---------------------------------------------------------------- filters

def _shrink(x: np.ndarray, f: int) -> np.ndarray:
    """Means of f x f blocks; a last partial block row or column is left out.

    Whole blocks keep OpenCV on its fast integer-factor path (four times
    quicker) and put block i's centre exactly at (i + 1/2) f.
    """
    if f == 1:
        return x
    h, w = x.shape[0] // f, x.shape[1] // f
    return cv2.resize(x[:h * f, :w * f], (w, h), interpolation=cv2.INTER_AREA)


def _expand(x: np.ndarray, f: int, h: int, w: int) -> np.ndarray:
    """Back from :func:`_shrink` to (h, w): bilinear between block centres."""
    if f == 1:
        return x
    up = cv2.resize(x, (x.shape[1] * f, x.shape[0] * f), interpolation=cv2.INTER_LINEAR)
    if up.shape != (h, w):
        up = cv2.copyMakeBorder(up, 0, h - up.shape[0], 0, w - up.shape[1], cv2.BORDER_REPLICATE)
    return up


def _expand_rows(x: np.ndarray, f: int, sl: slice, h: int, w: int) -> np.ndarray:
    """Rows ``sl`` of ``_expand(x, f, h, w)``, without expanding the others."""
    if f == 1:
        return x[sl]
    # One block row of margin each side, so the bilinear weights of every
    # wanted row come out as in the whole expansion.
    g0 = max(0, sl.start // f - 1)
    g1 = min(x.shape[0], (sl.stop - 1) // f + 2)
    up = cv2.resize(x[g0:g1], (x.shape[1] * f, (g1 - g0) * f), interpolation=cv2.INTER_LINEAR)
    top, bottom = sl.start - g0 * f, sl.stop - g0 * f
    if bottom > up.shape[0] or up.shape[1] != w:
        up = cv2.copyMakeBorder(up, 0, max(0, bottom - up.shape[0]), 0, w - up.shape[1], cv2.BORDER_REPLICATE)
    return up[top:bottom]


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; wide ones run on a subsampled copy (same look, far cheaper)."""
    if sigma < 0.3:
        return x
    h, w = x.shape
    f = int(sigma // 4)
    if f >= 2 and min(h, w) >= 4 * f:
        return _expand(cv2.GaussianBlur(_shrink(x, f), (0, 0), sigma / f), f, h, w)
    return cv2.GaussianBlur(x, (0, 0), sigma)


def _guided(I: np.ndarray, sigma: float, eps: float, step: float = 4.0,
            min_factor: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Self-guided filter (He et al.) as full-size maps (A, B): smoothed I = A*I + B.

    Flat areas get the local mean (A ~ 0) while edges stronger than
    sqrt(eps) are kept (A ~ 1), so subtracting it never makes halos at hard
    edges. The coefficients are smooth, so they are solved on a grid
    ``sigma / step`` (at least ``min_factor``) times coarser and
    interpolated ("fast guided filter"); the edges still come from the
    full-size I.
    """
    h, w = I.shape
    f = max(min_factor, int(sigma // step))
    if min(h, w) < 4 * f:
        f = 1
    s = max(sigma / f, 0.3)
    gi = _shrink(I, f)
    mean = cv2.GaussianBlur(gi, (0, 0), s)
    corr = cv2.GaussianBlur(gi * gi, (0, 0), s)
    eps32 = np.float32(eps)
    A = np.empty_like(mean)
    B = np.empty_like(mean)

    def solve(sl: slice) -> None:
        m = mean[sl]
        var = corr[sl] - m * m
        np.maximum(var, 0.0, out=var)
        gain = var + eps32
        np.divide(var, gain, out=gain)
        A[sl] = gain
        gain *= m
        np.subtract(m, gain, out=B[sl])

    _rows(solve, *mean.shape)
    return tuple(_expand(cv2.GaussianBlur(m, (0, 0), s), f, h, w) for m in (A, B))


@functools.lru_cache(maxsize=8)
def _grid_cells(h: int, w: int, c: int, gw: int) -> np.ndarray:
    """Grid cell of each pixel of an (h, w) image cut into c x c cells (flat, read-only)."""
    cells = ((np.arange(h) // c)[:, None] * gw + (np.arange(w) // c)[None, :]).ravel()
    cells.flags.writeable = False
    return cells


@functools.lru_cache(maxsize=16)
def _strip_cells(rows: int, w: int, cell: int, gw: int) -> np.ndarray:
    """y map for :meth:`_BilateralGrid.smooth`: row r of a strip, column x
    reads table row r * gw + (x's grid column), clamped so a bilinear read
    never runs into the next image row's part of the table (read-only)."""
    gx = np.clip((np.arange(w, dtype=np.float32) + 0.5) / cell - 0.5, 0.0, gw - 1.0)
    y = np.arange(rows, dtype=np.float32)[:, None] * np.float32(gw) + gx
    y.flags.writeable = False
    return y


class _BilateralGrid:
    """Bilateral filter of a lightness plane (Chen, Paris & Durand 2007).

    Pixels are splatted into a coarse (y, x, lightness) grid, blurred there
    and read back at their own lightness, so each pixel gets the mean of
    neighbours of similar lightness only: a flat patch keeps one value up to
    its outline, whatever lies beyond it, while texture finer than the
    lightness window still averages out. The grid is built from a copy
    shrunk to about eight pixels per sigma.

    The grid is interpolated to every image row up front; a strip then
    reads it with one bilinear remap whose x is the lightness bin and whose
    y walks the grid columns, which interpolates in both at once.
    """

    _CELL = 4  # grid cell, in pixels of the shrunk copy
    # Weight of a bin's own lightness in its mean: where no neighbour has a
    # pixel's lightness, the filter gives back that lightness.
    _REGULAR = 0.01

    def __init__(self, L: np.ndarray, sigma: float, sigma_r: float) -> None:
        h, w = L.shape
        c = self._CELL
        f = max(1, min(round(sigma / (2 * c)), min(h, w)))
        small = _shrink(L, f).ravel()
        hs, ws = h // f, w // f
        gh, gw = -(-hs // c), -(-ws // c)
        # Tent splat, [1 2 1] blur and linear read-back make the lightness
        # window 0.91 bins wide (standard deviation).
        spacing = sigma_r * 1.1
        d = int(np.ceil(1.0 / spacing)) + 2
        z = np.clip(small / np.float32(spacing), 0.0, d - 1.001)
        k = z.astype(np.int64)
        t = z - k
        idx = _grid_cells(hs, ws, c, gw) * d + k
        n = gh * gw * d
        lo, hi = 1.0 - t, t
        grid = np.empty((gh, gw, 2, d), dtype=np.float32)
        grid[:, :, 0] = (np.bincount(idx, lo * small, n) + np.bincount(idx + 1, hi * small, n)).reshape(gh, gw, d)
        grid[:, :, 1] = (np.bincount(idx, lo, n) + np.bincount(idx + 1, hi, n)).reshape(gh, gw, d)
        grid *= np.float32(1.0 / (c * c))
        grid = grid.reshape(gh, gw, 2 * d)
        sg = sigma / (f * c)
        if sg >= 0.3:
            grid = cv2.GaussianBlur(grid, (0, 0), sg, borderType=cv2.BORDER_REPLICATE)
        grid = grid.reshape(gh, gw, 2, d)
        grid[..., 1:-1] = 0.5 * grid[..., 1:-1] + 0.25 * (grid[..., :-2] + grid[..., 2:])
        reg = np.float32(self._REGULAR)
        mean = grid[:, :, 0] + reg * spacing * np.arange(d, dtype=np.float32)
        mean /= grid[:, :, 1] + reg
        # Rows of the grid at every image row (one extra grid row covers the
        # rows the shrink left out): (h, gw * d), bins innermost.
        cell = f * c
        mean = cv2.copyMakeBorder(mean.reshape(gh, gw * d), 0, 1, 0, 0, cv2.BORDER_REPLICATE)
        rows = cv2.resize(mean, (gw * d, (gh + 1) * cell), interpolation=cv2.INTER_LINEAR)[:h]
        self._rows = rows.reshape(h * gw, d)
        self._gw, self._cell = gw, cell
        self._inv_spacing = np.float32(1.0 / spacing)
        self._chunk = max(1, _REMAP_MAX // gw)  # image rows per remap table

    def smooth(self, L0: np.ndarray, sl: slice) -> np.ndarray:
        """The filtered lightness of rows ``sl``, whose lightness is ``L0``."""
        z = L0 * self._inv_spacing
        out = np.empty_like(z)
        for r in range(0, z.shape[0], self._chunk):
            y0 = sl.start + r
            n = min(self._chunk, z.shape[0] - r)
            table = self._rows[y0 * self._gw:(y0 + n) * self._gw]
            out[r:r + n] = _remap(table, z[r:r + n], _strip_cells(n, z.shape[1], self._cell, self._gw),
                                  cv2.INTER_LINEAR)
        return out


# ---------------------------------------------------------------- curves

def curve_lut(points, size: int = 1024) -> np.ndarray:
    """Monotone cubic (Fritsch-Carlson) through ``points``, sampled over [0, 1].

    Between points the curve never overshoots (no wiggles, no reversal on
    monotone data); beyond the first and last points it stays flat, as in
    Lightroom when an end point is moved inwards.
    """
    pts = sorted((min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)) for x, y in points)
    xs: list[float] = []
    ys: list[float] = []
    for x, y in pts:
        if xs and x - xs[-1] < 1e-6:
            ys[-1] = y
        else:
            xs.append(x)
            ys.append(y)
    grid = np.linspace(0.0, 1.0, size)
    if not xs:
        return grid.astype(np.float32)
    if len(xs) == 1:
        return np.full(size, ys[0], dtype=np.float32)
    x = np.array(xs)
    y = np.array(ys)
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.empty_like(x)
    m[0], m[-1] = delta[0], delta[-1]
    m[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    m[1:-1][delta[:-1] * delta[1:] <= 0] = 0.0
    for k in range(len(delta)):
        if delta[k] == 0.0:
            m[k] = m[k + 1] = 0.0
            continue
        alpha, beta = m[k] / delta[k], m[k + 1] / delta[k]
        alpha, beta = max(alpha, 0.0), max(beta, 0.0)
        r = alpha * alpha + beta * beta
        if r > 9.0:
            tau = 3.0 / np.sqrt(r)
            alpha, beta = tau * alpha, tau * beta
        m[k], m[k + 1] = alpha * delta[k], beta * delta[k]
    g = np.clip(grid, x[0], x[-1])
    k = np.clip(np.searchsorted(x, g, side="right") - 1, 0, len(x) - 2)
    hk = h[k]
    t = (g - x[k]) / hk
    t2, t3 = t * t, t * t * t
    out = ((2 * t3 - 3 * t2 + 1) * y[k] + (t3 - 2 * t2 + t) * hk * m[k]
           + (-2 * t3 + 3 * t2) * y[k + 1] + (t3 - t2) * hk * m[k + 1])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _clean_splits(splits) -> tuple[float, float, float]:
    a, b, c = sorted(min(max(float(v), 0.05), 0.95) for v in splits)
    b = max(b, a + 0.02)
    c = max(c, b + 0.02)
    return a, b, c


def _parametric(values, splits, size: int) -> np.ndarray:
    s1, s2, s3 = _clean_splits(splits)
    edges = [0.0, s1, s2, s3, 1.0]
    lift = [0.45 * float(np.clip(v, -1.0, 1.0)) * (edges[i + 1] - edges[i]) for i, v in enumerate(values)]
    xs = [0.0]
    ds = [0.0]
    for i in range(4):
        xs.append(0.5 * (edges[i] + edges[i + 1]))
        ds.append(lift[i])
        xs.append(edges[i + 1])
        ds.append(0.5 * (lift[i] + lift[i + 1]) if i < 3 else 0.0)
    ys = np.maximum.accumulate(np.clip(np.array(xs) + np.array(ds), 0.0, 1.0))
    return curve_lut(list(zip(xs, ys)), size)


def parametric_lut(s: DevelopSettings, size: int = 1024) -> np.ndarray:
    """Lightroom's region curve: Shadows/Darks/Lights/Highlights between the splits.

    Each slider lifts or lowers the middle of its region by up to 45 % of the
    region's width; neighbouring regions meet half-way at their split, and a
    monotone spline joins the knots, so the curve stays smooth and never
    reverses. The end points stay at 0 and 1.
    """
    return _parametric((s.curve_shadows, s.curve_darks, s.curve_lights, s.curve_highlights), s.curve_splits, size)


def _curve_is_identity(points) -> bool:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2 or abs(min(p[0] for p in pts)) > _EPS or abs(max(p[0] for p in pts) - 1.0) > _EPS:
        return False
    return all(abs(x - y) <= _EPS for x, y in pts)


def _curve_key(s: DevelopSettings) -> tuple | None:
    """Everything the tone curves depend on (hashable), or None when they do nothing."""
    if not s.section_on("curve"):
        return None
    params = tuple(float(v) for v in (s.curve_shadows, s.curve_darks, s.curve_lights, s.curve_highlights))
    curves = tuple(tuple((float(x), float(y)) for x, y in p)
                   for p in (s.curve_rgb, s.curve_red, s.curve_green, s.curve_blue))
    if all(abs(v) <= _EPS for v in params) and all(_curve_is_identity(p) for p in curves):
        return None
    return params, tuple(float(v) for v in s.curve_splits), curves


@functools.lru_cache(maxsize=8)
def _curve_table(key: tuple, src: str, dst: str) -> np.ndarray:
    """Per-channel display-space curve (parametric, then RGB, then R/G/B) as
    three table rows taking each channel from ``src`` to ``dst`` ("enc" | "lin").
    A "lin" source is indexed by the square root of linear light."""
    params, splits, (rgb, *channels) = key
    n = 4096
    x = np.linspace(0.0, 1.0, n)
    base = _parametric(params, splits, n).astype(np.float64) if any(abs(v) > _EPS for v in params) else x
    if not _curve_is_identity(rgb):
        base = np.interp(base, x, curve_lut(rgb, n))
    enc = _srgb_encode64(_GRID * _GRID) if src == "lin" else _GRID
    rows = []
    for pts in channels:
        lut = base if _curve_is_identity(pts) else np.interp(base, x, curve_lut(pts, n))
        y = np.interp(enc, x, lut)
        rows.append(_srgb_decode64(y) if dst == "lin" else y)
    return _tables(*rows)


# ---------------------------------------------------------------- tables for colour

def _band_weights(hue: np.ndarray) -> np.ndarray:
    """(len(hue), 8) weights: each band rises from its neighbours' centres
    to 1 at its own as cos², so neighbours always sum to one."""
    centres = np.array([c for _, c in HSL_BANDS])
    ext = np.concatenate([[centres[-1] - 360.0], centres, [centres[0] + 360.0]])
    hue = np.mod(hue, 360.0)
    idx = np.clip(np.searchsorted(ext, hue, side="right") - 1, 0, len(ext) - 2)
    t = (hue - ext[idx]) / (ext[idx + 1] - ext[idx])
    n = len(centres)
    w = np.zeros((len(hue), n))
    rows = np.arange(len(hue))
    w[rows, (idx - 1) % n] += np.cos(0.5 * np.pi * t) ** 2
    w[rows, idx % n] += np.sin(0.5 * np.pi * t) ** 2
    return w


def _hsl_active(s: DevelopSettings) -> bool:
    return s.section_on("hsl") and any(abs(v) > _EPS for v in (*s.hsl_hue, *s.hsl_sat, *s.hsl_lum))


# Chroma moves (vibrance, HSL) depend only on a pixel's (a, b), so they are
# tabulated over that plane and applied with one bilinear read instead of a
# polar round trip per pixel. The tables have four channels: new a, new b,
# and a lightness scale and lift (L' = L * scale + lift).
_AB_RANGE = 0.5  # covers the sRGB gamut (|a|, |b| < 0.33) with room to spare
_AB_N = 257
_AB_SCALE = np.float32((_AB_N - 1) / (2 * _AB_RANGE))


@functools.cache
def _ab_plane() -> tuple[np.ndarray, ...]:
    """a, b, chroma, hue (degrees) and the HSL band weights over the table grid."""
    axis = np.linspace(-_AB_RANGE, _AB_RANGE, _AB_N)
    a, b = np.meshgrid(axis, axis)  # [row = b, column = a]
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    weights = _band_weights(hue.ravel()).reshape(*hue.shape, -1)
    planes = (a, b, chroma, hue, weights)
    for p in planes:
        p.flags.writeable = False
    return planes


def _ab_table(new_a: np.ndarray, new_b: np.ndarray, scale: np.ndarray | float = 1.0,
              lift: np.ndarray | float = 0.0) -> np.ndarray:
    out = np.empty((_AB_N, _AB_N, 4), dtype=np.float32)
    out[..., 0] = new_a
    out[..., 1] = new_b
    out[..., 2] = scale
    out[..., 3] = lift
    return _frozen(out)


def _ab_lookup(a0: np.ndarray, b0: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Bilinear read of a four-channel a/b table: (h, w, 4)."""
    x = a0 + np.float32(_AB_RANGE)
    x *= _AB_SCALE
    y = b0 + np.float32(_AB_RANGE)
    y *= _AB_SCALE
    return _remap(table, x, y, cv2.INTER_LINEAR)


@functools.lru_cache(maxsize=8)
def _vibrance_table(vibrance: float) -> np.ndarray:
    """New (a, b) for vibrance: low chroma gains most, skin hues are spared."""
    a, b, chroma, hue, _ = _ab_plane()
    norm = np.minimum(chroma / 0.16, 1.0)
    if vibrance > 0:
        off_skin = np.abs((hue - SKIN_HUE + 180.0) % 360.0 - 180.0)
        protect = (1.0 - _smoothstep(0.0, 35.0, off_skin)) * 0.7
        gain = 1.0 + vibrance * (1.0 - norm) ** 2 * (1.0 - protect)
    else:
        gain = 1.0 + 0.75 * vibrance * (1.0 - 0.3 * norm)
    return _ab_table(a * gain, b * gain)


@functools.lru_cache(maxsize=8)
def _hsl_table(hue_shift: tuple, sat: tuple, lum: tuple) -> np.ndarray:
    """New a, b and the lightness scale and lift for the eight HSL bands.

    Hue moves rotate by up to HSL_HUE_DEGREES, saturation scales chroma
    (0 at -1, double at +1). Luminance fades out towards grey, which has no
    hue. Darkening scales L, a and b alike, by up to 0.75 stop: an exposure
    change, which keeps the colour and stays in gamut. Lightening blends
    towards white by as much (1 - L shrinks as L does when darkening), which
    keeps the hue and stays in gamut too; scaling L alone would push bright
    colours past the gamut, where they clip to white through other hues.
    """
    _, _, chroma, hue, w = _ab_plane()
    shift = w @ (np.clip(np.asarray(hue_shift, dtype=np.float64), -1, 1) * HSL_HUE_DEGREES)
    gain = np.maximum(1.0 + w @ np.clip(np.asarray(sat, dtype=np.float64), -1, 1), 0.0)
    lum_shift = (w @ np.clip(np.asarray(lum, dtype=np.float64), -1, 1)) * np.minimum(chroma / 0.07, 1.0)
    scale = np.exp2(-0.75 * np.abs(lum_shift))
    new_chroma = chroma * gain * scale
    new_hue = np.radians(hue + shift)
    return _ab_table(new_chroma * np.cos(new_hue), new_chroma * np.sin(new_hue), scale,
                     np.where(lum_shift > 0, 1.0 - scale, 0.0))


def _hue_direction(hue: float) -> np.ndarray:
    """Unit OKLab (a, b) of a colour-wheel hue (HSV degrees: 0 red, 120 green, 240 blue)."""
    rgb = colorsys.hsv_to_rgb((float(hue) % 360.0) / 360.0, 0.35, 0.75)
    ab = _oklab64(np.array(rgb))[1:]
    return ab / max(float(np.hypot(*ab)), _EPS)


def _grading_active(s: DevelopSettings) -> bool:
    wheels = (s.grade_shadows, s.grade_midtones, s.grade_highlights, s.grade_global)
    return s.section_on("grading") and any(w[1] > _EPS or abs(w[2]) > _EPS for w in wheels)


@functools.lru_cache(maxsize=8)
def _grading_table(wheels: tuple, blending: float, balance: float) -> np.ndarray:
    """Δa, Δb, ΔL over OKLab L for the shadows/midtones/highlights/global wheels.

    Range weights are normalised Gaussians at 0, ½ and 1 on a lightness
    axis bent by ``grade_balance`` (positive gives the highlights more of
    the range); ``grade_blending`` widens them so the ranges overlap more.
    Tints fade out at pure black and pure white so those stay neutral.
    Returned as a (1, n, 4) table read at the nearest entry.
    """
    x = np.linspace(0.0, 1.0, _N_L)
    pivot = 0.5 - 0.3 * float(np.clip(balance, -1.0, 1.0))
    t = x ** (np.log(0.5) / np.log(pivot))
    sigma = 0.10 + 0.25 * float(np.clip(blending, 0.0, 1.0))
    weights = np.stack([np.exp(-((t - c) ** 2) / (2 * sigma * sigma)) for c in (0.0, 0.5, 1.0)])
    weights /= weights.sum(axis=0)
    taper = _smoothstep(0.0, 0.12, x) * (1.0 - _smoothstep(0.92, 1.0, x))
    out = np.zeros((1, _N_L, 4), dtype=np.float32)
    delta = np.zeros((3, _N_L))
    for i, (hue, sat, lum) in enumerate(wheels):
        weight = weights[i] if i < 3 else np.ones(_N_L)
        tint = GRADE_CHROMA * float(np.clip(sat, 0.0, 1.0)) * _hue_direction(hue)
        delta[0] += weight * taper * tint[0]
        delta[1] += weight * taper * tint[1]
        delta[2] += weight * GRADE_LUM * float(np.clip(lum, -1.0, 1.0))
    out[0, :, :3] = delta.T
    return _frozen(out)


@functools.lru_cache(maxsize=8)
def _hs_table(highlights: float, shadows: float) -> np.ndarray:
    """ΔL as a function of the local mean lightness (of similar tones only).

    In display terms the shadows bump covers values up to ~0.6 and peaks
    near 0.18; the highlights bump starts at ~0.3 and peaks near 0.74, so
    neither reaches the far end of the range. The direction that expands
    tones (lifting shadows, pulling highlights down) may go further than
    the one that compresses them, which stops before the curve reverses.
    """
    x = np.linspace(0.0, 1.0, _N_L)
    u = np.clip(x / 0.7, 0.0, 1.0)
    bump_s = u * (1 - u) ** 1.5  # peak at L 0.28
    u = np.clip((x - 0.4) / 0.6, 0.0, 1.0)
    bump_h = u * u * (1 - u)  # peak at L 0.8
    sh = float(np.clip(shadows, -1, 1))
    hi = float(np.clip(highlights, -1, 1))
    lift = sh * (0.2 if sh > 0 else 0.11) * bump_s / bump_s.max()
    pull = hi * (0.15 if hi < 0 else 0.08) * bump_h / bump_h.max()
    # Both expanding at once overlap in the midtones, where their slopes add
    # up and would flatten different tones into one. Keep the curve at least
    # as steep as either alone (or 1/4), and take what that adds back from
    # the steeper parts in proportion, so the ends stay at 0 and 1.
    slope = np.diff(x + lift + pull)
    floor = min(np.diff(x + lift).min(), np.diff(x + pull).min(), 0.25 / (_N_L - 1))
    slope = np.maximum(slope, floor)
    extra = slope - floor
    slope = floor + extra * ((1.0 - floor * (_N_L - 1)) / extra.sum())
    return _tables(np.concatenate([[0.0], np.cumsum(slope)]) - x)


@functools.lru_cache(maxsize=8)
def _clarity_table(clarity: float) -> np.ndarray:
    """Clarity's gain over the local mean lightness: full in the midtones,
    none in deep shadows and bright highlights."""
    x = np.linspace(0.0, 1.0, _N_L)
    return _tables(clarity * np.maximum(1.0 - ((x - 0.55) / 0.45) ** 2, 0.0))


# ---------------------------------------------------------------- stages
#
# Each stage does its whole-image work (blurs, filter coefficients, tables)
# up front and returns a kernel for one strip of rows, or None when it has
# nothing to do. Kernels between two whole-image steps run fused, so a strip
# goes through all of them while it is still in cache.

Kernel = Callable[[slice], None]


def _tone_active(s: DevelopSettings) -> bool:
    return any(abs(v) > _EPS for v in (s.highlights, s.shadows, s.texture, s.clarity, s.vibrance))


def _detail_active(s: DevelopSettings) -> bool:
    return s.section_on("detail") and (s.sharpen_amount > _EPS or s.nr_luma > _EPS or s.nr_color > _EPS)


def _effects_active(s: DevelopSettings) -> bool:
    return s.section_on("effects") and (abs(s.vignette_amount) > _EPS or s.grain_amount > _EPS)


def _sharpen_on(s: DevelopSettings) -> bool:
    return s.section_on("detail") and s.sharpen_amount > _EPS


def is_identity(s: DevelopSettings) -> bool:
    """True when no adjustment would change a pixel (neutral or switched off)."""
    return not (_tone_active(s) or _curve_key(s) is not None or _hsl_active(s) or _grading_active(s)
                or _detail_active(s) or _effects_active(s))


def _tone_stage(L: np.ndarray, a: np.ndarray, b: np.ndarray, s: DevelopSettings, px: float) -> Kernel:
    """Highlights/shadows, texture, clarity and vibrance.

    The local operators are all measured on the incoming lightness and
    their changes added together, which keeps each slider's look
    independent of the others.
    """
    h, w = L.shape
    hs = abs(s.highlights) > _EPS or abs(s.shadows) > _EPS
    texture = float(np.clip(s.texture, -1, 1))
    clarity = float(np.clip(s.clarity, -1, 1))
    jobs = {}
    if abs(texture) > _EPS:
        # Edge-preserving coarse level: next to a hard edge the band is ~0,
        # so texture never draws halos along outlines.
        jobs["tx"] = lambda: _guided(L, TEXTURE_SIGMAS[1] * px, TEXTURE_EPS, step=1.5, min_factor=2)
        # Floored so a preview keeps the share of detail the downsampled
        # export has in this band, not all of its own finest detail.
        jobs["fine"] = lambda: _blur(L, max(TEXTURE_SIGMAS[0] * px, TEXTURE_FINE_MIN))
        texture = np.float32(texture * (2.0 if texture > 0 else 1.5))
    if abs(clarity) > _EPS:
        jobs["cl"] = lambda: _guided(L, CLARITY_SIGMA * px, CLARITY_EPS, step=2.5)
        cl_table = _clarity_table(clarity * (1.0 if clarity > 0 else 0.8))
    if hs:
        # A bilateral mean: each area moves by the change for its own tone,
        # so flat areas stay flat and outlines get no halos, while texture
        # finer than HS_RANGE rides along and keeps its contrast.
        jobs["hs"] = lambda: _BilateralGrid(L, HS_SIGMA_FRACTION * max(h, w), HS_RANGE)
        hs_table = _hs_table(float(s.highlights), float(s.shadows))
    done = dict(zip(jobs, _together(*jobs.values())))
    if hs:
        hs_grid = done["hs"]
    if abs(texture) > _EPS:
        tx_a, tx_b = done["tx"]
        fine = done["fine"]
    if abs(clarity) > _EPS:
        cl_a, cl_b = done["cl"]
    local = hs or abs(texture) > _EPS or abs(clarity) > _EPS
    vibrance = float(np.clip(s.vibrance, -1, 1))
    vib_table = _vibrance_table(vibrance) if abs(vibrance) > _EPS else None
    to_entry = np.float32(_N_L - 1)

    def local_tone(sl: slice) -> None:
        # Every change is measured on the incoming L0 and added at the end.
        L0 = L[sl]
        changes = []
        if hs:
            base = hs_grid.smooth(L0, sl)
            base *= to_entry
            change = _lut(base, hs_table)
            # Scale chroma with lightness: like an exposure change, a lifted
            # shadow keeps its colour instead of turning grey.
            ratio = L0 + np.float32(0.03)
            np.divide(change, ratio, out=ratio)
            ratio += np.float32(1.0)
            np.clip(ratio, 0.5, 2.0, out=ratio)
            a[sl] *= ratio
            b[sl] *= ratio
            changes.append(change)
        if abs(texture) > _EPS:
            # The coarse level is A*I + B with I the fine level, not L0: in
            # busy texture A ~ 1, and A*L0 would carry L0's finest detail
            # into the band with a minus sign, so texture would soften it.
            f0 = fine[sl]
            band = tx_a[sl] * f0
            band += tx_b[sl]
            np.subtract(f0, band, out=band)
            soft = np.abs(band)
            soft *= np.float32(1 / 0.05)
            soft += np.float32(1.0)
            band /= soft
            band *= texture
            changes.append(band)
        if abs(clarity) > _EPS:
            base = cl_a[sl] * L0
            base += cl_b[sl]
            detail = L0 - base
            base *= to_entry
            detail *= _lut(base, cl_table)
            changes.append(detail)
        for change in changes:
            L0 += change

    def kernel(sl: slice) -> None:
        if local:
            local_tone(sl)
        if vib_table is not None:
            ab = _ab_lookup(a[sl], b[sl], vib_table)
            a[sl] = ab[..., 0]
            b[sl] = ab[..., 1]

    return kernel


def _colour_stage(L: np.ndarray, a: np.ndarray, b: np.ndarray, s: DevelopSettings) -> Kernel | None:
    """HSL in OKLCh, then colour grading. Tables only: no whole-image work."""
    hsl = _hsl_active(s)
    grading = _grading_active(s)
    if not (hsl or grading):
        return None
    if hsl:
        hsl_table = _hsl_table(tuple(map(float, s.hsl_hue)), tuple(map(float, s.hsl_sat)), tuple(map(float, s.hsl_lum)))
        hsl_lum = any(abs(v) > _EPS for v in s.hsl_lum)
    if grading:
        wheels = tuple(tuple(map(float, w))
                       for w in (s.grade_shadows, s.grade_midtones, s.grade_highlights, s.grade_global))
        grade_table = _grading_table(wheels, float(s.grade_blending), float(s.grade_balance))

    def kernel(sl: slice) -> None:
        L0, a0, b0 = L[sl], a[sl], b[sl]
        if hsl:
            new = _ab_lookup(a0, b0, hsl_table)
            a0[...] = new[..., 0]
            b0[...] = new[..., 1]
            if hsl_lum:
                L0 *= new[..., 2]
                L0 += new[..., 3]
        if grading:
            coord = L0 * np.float32(_N_L - 1)
            delta = _remap(grade_table, coord, _row_index(*coord.shape, 1))
            a0 += delta[..., 0]
            b0 += delta[..., 1]
            L0 += delta[..., 2]

    return kernel


class _LocalStats:
    """Local mean of x and of x² (Gaussian window), for the Lee filters.

    Worked out on a copy f = sigma // ``step`` times smaller (none when that
    is under 2) and expanded strip by strip as the filter needs them: the
    statistics are smooth, and image-sized maps are slow to make.
    """

    def __init__(self, x: np.ndarray, sigma: float, step: float = 4.0) -> None:
        self.h, self.w = x.shape
        f = int(sigma // step)
        self.f = f if f >= 2 and min(self.h, self.w) >= 4 * f else 1
        self.mean = cv2.GaussianBlur(_shrink(x, self.f), (0, 0), sigma / self.f)
        self.corr = cv2.GaussianBlur(_shrink(x * x, self.f), (0, 0), sigma / self.f)

    def rows(self, sl: slice) -> tuple[np.ndarray, np.ndarray]:
        return (_expand_rows(self.mean, self.f, sl, self.h, self.w),
                _expand_rows(self.corr, self.f, sl, self.h, self.w))


def _lee(x: np.ndarray, mean: np.ndarray, corr: np.ndarray, eps: np.float32, k: float) -> None:
    """Move x a fraction k towards Lee's local-statistics filter, in place.

    The filter is the local mean where the local variance is no more than
    the noise (eps), and the pixel itself where it is larger: across a real
    edge the variance is large, so edges are kept and nothing bleeds over.
    """
    gain = mean * mean
    np.subtract(corr, gain, out=gain)
    np.maximum(gain, 0.0, out=gain)
    gain /= gain + eps
    smooth = x - mean
    smooth *= gain
    smooth += mean
    cv2.accumulateWeighted(smooth, x, k)


def _lee_colour(a: np.ndarray, b: np.ndarray, stats: list[tuple[np.ndarray, np.ndarray]],
                eps: np.float32, k: float) -> None:
    """:func:`_lee` on a and b together, in place, with one gain for both so
    the hue does not drift.

    The window is wide, so the gain is sharper than Lee's: squared, it stops
    the smoothing quickly once the local variance passes the noise, which
    keeps colour from bleeding far across an edge. It also looks at how far
    the pixel's colour is from the local mean: a wide window holds so little
    of a colour detail a few pixels across that its variance alone would not
    save it, but such a pixel stands out from the mean by more than noise.
    """
    (ma, ca), (mb, cb) = stats
    da = a - ma
    db = b - mb
    noise = np.float32(2.0) * eps  # two channels
    var = ma * ma
    np.subtract(ca, var, out=var)
    var += cb
    var -= mb * mb
    np.maximum(var, 0.0, out=var)
    var += noise
    move = np.divide(noise, var)
    move *= move
    move *= np.float32(k)
    dist = da * da
    dist += db * db
    far = np.float32(_NR_STANDOUT ** 2) * noise
    dist += far
    move *= far
    move /= dist
    da *= move
    db *= move
    a -= da
    b -= db


def _noise_stage(L: np.ndarray, a: np.ndarray, b: np.ndarray, s: DevelopSettings, px: float) -> Kernel | None:
    """Noise reduction: Lee's filter on lightness, and on a and b for colour.

    Filtering a and b by their own statistics keeps colour edges wherever
    they are; a filter guided by lightness smears colour across every edge
    without a lightness step, and between colours of similar lightness.
    """
    if not s.section_on("detail"):
        return None
    luma = float(np.clip(s.nr_luma, 0, 1))
    colour = float(np.clip(s.nr_color, 0, 1))
    jobs = []
    sigma = (0.6 + 2.4 * luma) * px
    do_luma = luma > _EPS and sigma >= 0.3
    if do_luma:
        jobs.append(functools.partial(_LocalStats, L, sigma))
        luma_eps, luma_k = np.float32((0.008 + 0.04 * luma) ** 2), min(1.0, 1.4 * luma)
    # Colour noise from film grain and demosaicing comes in blotches several
    # pixels wide, so the colour window is wide. Its statistics vary slowly:
    # a copy shrunk to about 1.25 pixels per sigma serves.
    sigma = (2.0 + 18.0 * colour) * px
    do_colour = colour > _EPS and sigma >= 0.3
    if do_colour:
        jobs += [functools.partial(_LocalStats, x, sigma, 1.25) for x in (a, b)]
        colour_eps, colour_k = np.float32((0.015 + 0.025 * colour) ** 2), min(1.0, 4.0 * colour)
    if not jobs:
        return None
    stats = _together(*jobs)
    luma_stats = stats.pop(0) if do_luma else None

    def kernel(sl: slice) -> None:
        if do_luma:
            _lee(L[sl], *luma_stats.rows(sl), luma_eps, luma_k)
        if do_colour:
            _lee_colour(a[sl], b[sl], [st.rows(sl) for st in stats], colour_eps, colour_k)

    return kernel


def _edge_strength(L: np.ndarray, px: float) -> np.ndarray:
    """Gradient magnitude of the softened lightness, per full-resolution pixel."""
    soft = _blur(L, max(1.0 * px, 0.5))
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)
    # Sobel gives 8x the slope per pixel; times px makes it per full-res pixel.
    return cv2.magnitude(gx, gy) * np.float32(px / 8.0)


def _edge_mask(grad: np.ndarray, masking: float) -> np.ndarray:
    """Lightroom's sharpening mask: 1 on edges, 0 on smooth areas."""
    threshold = 0.04 * masking
    return _smoothstep(0.5 * threshold, 1.5 * threshold, grad)


def _vignette(h: int, w: int, s: DevelopSettings) -> tuple[np.ndarray, np.ndarray | None]:
    """Post-crop vignette as (gain, lift) maps: L' = L * gain + lift, a/b * gain.

    Darkening scales L, a and b together, an exposure change (-1 takes the
    corners down three stops); lightening blends towards white. The maps are
    smooth at the scale of the feather (3 % of the frame at least), so they
    are worked out on a grid about 256 samples across and interpolated.
    """
    f = max(1, round(max(h, w) / 256))
    gh, gw = max(1, round(h / f)), max(1, round(w / f))
    # Sample centres in full-size pixels, where cv2.resize expects them.
    x = ((np.arange(gw) + 0.5) * (w / gw) - w / 2) / (w / 2)
    y = ((np.arange(gh) + 0.5) * (h / gh) - h / 2) / (h / 2)
    r = float(np.clip(s.vignette_roundness, -1, 1))
    if r > 0:
        # Towards a circle in pixels: equal scales on both axes.
        g = np.sqrt(w * h)
        x *= 1 + r * (w / g - 1)
        y *= 1 + r * (h / g - 1)
    p = 2.0 + 6.0 * max(-r, 0.0)  # negative roundness: towards a rounded rectangle
    # Distance 1 is the middle of an edge, sqrt(2) a corner (ellipse).
    dist = (np.abs(y)[:, None] ** p + np.abs(x)[None, :] ** p) ** (1.0 / p)
    mid = 0.5 + 0.8 * float(np.clip(s.vignette_midpoint, 0, 1))
    half = 0.03 + 0.45 * float(np.clip(s.vignette_feather, 0, 1))
    t = _smoothstep(mid - half, mid + half, dist)
    amount = float(np.clip(s.vignette_amount, -1, 1))
    if amount < 0:
        maps = [np.exp2(amount * t)]
    else:
        lift = 0.9 * amount * t
        maps = [1.0 - lift, lift]

    def full(m: np.ndarray) -> np.ndarray:
        m = m.astype(np.float32)
        return m if (gh, gw) == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    gain, *lift_map = (full(m) for m in maps)
    return gain, (lift_map[0] if lift_map else None)


def _white_noise(shape: tuple[int, int], key: tuple[int, ...]) -> np.ndarray:
    """Zero-mean, unit-std uniform noise, deterministic in ``key``.

    Sixteen raw bits per value from SFC64: three times quicker than
    ``Generator.random``, and still 65536 levels.
    """
    n = shape[0] * shape[1]
    bits = np.random.SFC64(np.random.SeedSequence(key)).random_raw((n + 3) // 4).view(np.uint16)[:n]
    noise = bits.astype(np.float32).reshape(shape)
    noise -= np.float32(32767.5)
    noise *= np.float32(np.sqrt(12.0) / 65536.0)
    return noise


def _grain_field(h: int, w: int, sigma: float, strength: float, key: tuple[int, ...]) -> np.ndarray:
    """Grain in round clumps about 2 * ``sigma`` pixels across, with the std
    its pixels show times ``strength``.

    A pixel shows the grain averaged over its area, taken as a Gaussian of
    variance 1/12 (that of a box one pixel wide). So the field is white
    noise through a Gaussian of sigma² + 1/12, and pixels show
    sigma / sqrt(sigma² + 1/12) of the grain's strength: grain finer than a
    pixel averages out, by as much in a preview as in the downsampled
    export. Filtering white noise keeps the clumps round, where
    interpolating a coarse noise grid lines them up with the grid.
    """
    s = float(np.sqrt(sigma * sigma + 1.0 / 12.0))
    k = cv2.getGaussianKernel(2 * int(np.ceil(3.0 * s)) + 1, s, cv2.CV_32F)
    # Unit white noise comes out of k kᵀ with std sum(k²); the kernel is
    # scaled to give the std wanted instead (it is applied twice).
    k *= np.float32(np.sqrt(strength * (sigma / s) / float((k * k).sum())))
    field = _white_noise((h, w), key)
    cv2.sepFilter2D(field, -1, k, k, dst=field, borderType=cv2.BORDER_REFLECT)
    return field


class _Effects:
    """Vignette maps and grain noise. They depend only on the frame size and
    the settings, so they are made in the background while the rest runs."""

    def __init__(self, h: int, w: int, s: DevelopSettings, px: float, seed: int) -> None:
        self.vignette = abs(s.vignette_amount) > _EPS
        if self.vignette:
            self.gain, self.lift = _vignette(h, w, s)
        self.grain = s.grain_amount > _EPS
        if self.grain:
            self.amp = np.float32(GRAIN_STRENGTH * float(np.clip(s.grain_amount, 0, 1)))
            sigma = 0.5 * (1.0 + 5.0 * float(np.clip(s.grain_size, 0, 1))) * px
            rough = float(np.clip(s.grain_roughness, 0, 1))
            key = int(seed) & 0xFFFFFFFF
            # Roughness mixes in clumps a few grains wide; at full size the
            # mix keeps unit std. Each part fades by its own size when it is
            # finer than the pixels, the fine grain long before the clumps.
            norm = 1.0 / np.hypot(1.0 - 0.5 * rough, 0.6 * rough)
            self.noise = _grain_field(h, w, sigma, (1.0 - 0.5 * rough) * norm, (key, 1))
            if rough > _EPS:
                self.noise += _grain_field(h, w, 2.5 * sigma, 0.6 * rough * norm, (key, 3))


def _finish_stage(L: np.ndarray, a: np.ndarray, b: np.ndarray, s: DevelopSettings, px: float,
                  effects: _Effects | None) -> Kernel | None:
    """Sharpening, post-crop vignette and grain."""
    sharpen = _sharpen_on(s)
    if sharpen:
        masking = float(np.clip(s.sharpen_masking, 0, 1))
        # A sampled Gaussian narrower than SHARPEN_MIN_SIGMA is nearly all
        # centre tap, so a preview would show next to no sharpening. A wider
        # one with the amount cut by (radius / sigma)² keeps the unsharp
        # mask's gain on the coarser detail a preview has, which is what
        # the downsampled export shows.
        radius = max(float(s.sharpen_radius), 0.0) * px
        sigma = max(radius, SHARPEN_MIN_SIGMA)
        jobs = [lambda: _blur(L, sigma)]
        if masking > _EPS:
            jobs.append(lambda: _edge_strength(L, px))
        soft, *grad = _together(*jobs)
        grad = grad[0] if grad else None
        # Detail sets the halo limit: low values cap the overshoot at strong
        # edges, 1 is a plain unsharp mask.
        inv_limit = np.float32(1.0 / (0.02 + 0.4 * float(np.clip(s.sharpen_detail, 0, 1)) ** 2))
        amount = np.float32(np.clip(s.sharpen_amount, 0, 1.5) * (radius / sigma) ** 2)
    vignette = effects is not None and effects.vignette
    grain = effects is not None and effects.grain
    if grain:
        g_amp = effects.amp
    if not (sharpen or vignette or grain):
        return None

    def kernel(sl: slice) -> None:
        L0 = L[sl]
        if sharpen:
            d = L0 - soft[sl]
            limit = np.abs(d)
            limit *= inv_limit
            limit += np.float32(1.0)
            d /= limit
            if grad is not None:
                d *= _edge_mask(grad[sl], masking)
            d *= amount
            L0 += d
        if vignette:
            gain = effects.gain[sl]
            L0 *= gain
            if effects.lift is not None:
                L0 += effects.lift[sl]
            a[sl] *= gain
            b[sl] *= gain
        if grain:
            # Strongest in the midtones, like grain seen through a print.
            mids = np.float32(1.0) - L0
            mids *= L0
            mids *= np.float32(4.0) * g_amp
            np.clip(mids, 0.15 * g_amp, g_amp, out=mids)
            mids *= effects.noise[sl]
            L0 += mids

    return kernel


# ---------------------------------------------------------------- entry points

def apply_adjustments(rgb: np.ndarray, s: DevelopSettings, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Apply every Lightroom-style adjustment to sRGB-encoded ``rgb`` (H, W, 3).

    ``scale`` is this image's width over the full-resolution width; ``seed``
    fixes the grain pattern. The input is never modified, and with nothing
    to do it is returned as is.
    """
    if is_identity(s):
        return rgb
    src = np.ascontiguousarray(rgb, dtype=np.float32)
    h, w = src.shape[:2]
    px = float(max(scale, 1e-3))
    out = np.empty((h, w, 3), dtype=np.float32)
    curve = _curve_key(s)
    tone_on = _tone_active(s)
    colour_on = _hsl_active(s) or _grading_active(s)
    later = _detail_active(s) or _effects_active(s)
    if not (tone_on or colour_on or later):

        def curves_only(sl: slice) -> None:
            out[sl] = _lut(src[sl] * np.float32(_N - 1), table)

        table = _curve_table(curve, "enc", "enc")
        _rows(curves_only, h, w)
        return out

    effects = _POOL.submit(_Effects, h, w, s, px, seed) if _effects_active(s) else None
    L, a, b = (np.empty((h, w), dtype=np.float32) for _ in range(3))
    tone = None
    if tone_on:
        _rows(lambda sl: _lin_to_lab(_decode(src[sl]), L, a, b, sl), h, w)
        tone = _tone_stage(L, a, b, s, px)
    colour = _colour_stage(L, a, b, s)
    ends_in_curve = curve is not None and tone_on and colour is None and not later
    if curve is not None:
        table = _curve_table(curve, "lin" if tone_on else "enc", "enc" if ends_in_curve else "lin")

    def first(sl: slice) -> None:
        if tone is not None:
            tone(sl)
            if curve is not None:
                mapped = _lut(_sqrt_index(_lab_to_lin(L, a, b, sl)), table)
                if ends_in_curve:
                    out[sl] = mapped
                    return
                _lin_to_lab(mapped, L, a, b, sl)
        else:
            lin = _decode(src[sl]) if curve is None else _lut(src[sl] * np.float32(_N - 1), table)
            _lin_to_lab(lin, L, a, b, sl)
        if colour is not None:
            colour(sl)
        if not later:
            out[sl] = _encode(_lab_to_lin(L, a, b, sl))

    _rows(first, h, w)
    if not later:
        return out
    tone = None  # frees its filter maps (up to seven image-sized planes) before the next stage makes its own
    noise = _noise_stage(L, a, b, s, px)
    if noise is not None and _sharpen_on(s):
        _rows(noise, h, w)  # sharpening must see the denoised image
        noise = None
    finish = _finish_stage(L, a, b, s, px, effects.result() if effects is not None else None)

    def last(sl: slice) -> None:
        if noise is not None:
            noise(sl)
        if finish is not None:
            finish(sl)
        out[sl] = _encode(_lab_to_lin(L, a, b, sl))

    _rows(last, h, w)
    return out


def clipping_masks(rgb: np.ndarray, low: float = 0.5 / 255, high: float = 254.5 / 255) -> tuple[np.ndarray, np.ndarray]:
    """(shadows, highlights) boolean maps of pixels with any channel clipped."""
    return rgb.min(axis=-1) <= low, rgb.max(axis=-1) >= high
