"""Reading the DX film edge barcode on the rebate of 35 mm film.

Since 1983 most 135 film carries, between the sprocket holes and the film
edge on one side, a latent-image barcode exposed at the factory and repeated
every half frame (19 mm, four perforations). It has two parallel tracks of
equal length, made of modules about 0.4 mm wide:

* the clock track, next to the perforations: a 5-module bar, single-module
  spaces and bars alternating, and a 3-module bar. 31 modules long on film
  that also encodes the frame number (since the 1990s), 23 on older film;
* the data track, next to the film edge, one bit per clock module (1 = bar):

  ======  ======================================================
  module  31-module code (23-module code)
  ======  ======================================================
  0-5     start ``101010``
  6-12    DX number part 1, the product (7 bits, most significant first)
  13      unassigned, always 0
  14-17   DX number part 2, the specifier (4 bits)
  18-23   frame number, 6 bits             (18: parity; 19-22: stop ``0101``)
  24      half frame: 1 for "21A", 0 for "21"
  25      unassigned, always 0
  26      parity: the bits of parts 1 and 2, frame and half frame plus
          this one add up to an even number
  27-30   stop ``0101``
  ======  ======================================================

The DX number "part 1-part 2" (e.g. 95-7) names the emulsion; the six-digit
cartridge code (PHHHHE) carries it hashed as 16 * part1 + part2 in its four
middle digits, though the two do not always agree. The eye-readable frame
number ("21", "21A") is printed just before the code it encodes, in reading
order. A bar is film exposed at the factory: dense on a negative, clear on
a slide.

Nothing about the strip needs to be known first. Runs of equal alternating
bars (the clock) are looked for along the rows and the columns of a small
image pyramid, so any scale and either direction is found, the ones with
few others around them first: a regular texture in a picture makes many,
crowded together. The strip's local direction comes from the edges around
them; a patch resampled along the strip is searched for the whole clock
pattern, whose 5-module end says where the code starts, and the data track
is sampled at the clock's module centres on whichever side of it decodes.
That side gives the film's face: seen from the face where the edge print
reads normally, the data track lies below a code read left to right, to the
right of the reading direction on screen; on its left, the film was turned
over and the picture is mirrored. Every rule of the format must hold (start,
stop and empty bits, parity), the instances along the strip vote on the
film, and the numbered ones on the frame number. The capture shows a bar as
denser than the base (checked on a Nikon D780 capture of Kodak film: bars
about 0.45 D above the base), but the camera blurs light, not density, so
bars are told apart half way in light.

Sources:

* ZXing-CPP, ``core/src/oned/ODDXFilmEdgeReader.cpp`` (A. Mérino and A.
  Waggershauser, 2023): clock patterns, field positions and the parity
  rule, as decoded from real film. https://github.com/zxing-cpp/zxing-cpp
* Wikipedia, "DX encoding" and "DX number": track layout, field lengths,
  the cartridge hash; after US patent 4,965,628 (Eastman Kodak, 1990) and
  I3A, "DX Codes for 135-Size Film" (2008).
* The film names in ``belka/data/dx_codes.json`` list their own sources.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from belka import paths

# ---------------------------------------------------------------- the format
# One code every 19 mm: 47 modules of about 0.403 mm (measured on a Kodak
# strip against its 4.75 mm perforation pitch).
HALF_FRAME_MODULES = 47.0
# The printed number's centre lies this far before the code's start (same capture).
LABEL_GAP_MODULES = 4.8


@dataclass(frozen=True)
class _Layout:
    modules: int
    clock: tuple[int, ...]  # bar/space boundaries of the clock track, in modules from the start
    known: tuple[tuple[int, int], ...]  # (module, bit) fixed on the data track
    parity: int  # module of the parity bit
    numbered: bool  # carries a frame number


def _layout(modules: int) -> _Layout:
    known = [(0, 1), (1, 0), (2, 1), (3, 0), (4, 1), (5, 0), (13, 0),
             (modules - 4, 0), (modules - 3, 1), (modules - 2, 0), (modules - 1, 1)]
    if modules == 31:
        known.append((25, 0))
    return _Layout(modules, (0, *range(5, modules - 2), modules), tuple(sorted(known)),
                   26 if modules == 31 else 18, modules == 31)


LAYOUTS = (_layout(31), _layout(23))


@dataclass
class EdgeInfo:
    """What the edge barcode says about a frame (kept in ``Frame.edge``)."""

    dx: str  # DX number "part1-part2", e.g. "95-7"
    film: str  # film name from dx_codes.json, else the maker and the number
    profile_id: str | None  # matching Belka profile, if any
    frame: str  # printed frame number nearest the frame centre ("21A"), "" if not encoded
    confidence: float  # 0..1: how strongly and how far the instances agree, on the film and the number
    mirrored: bool = False  # film seen from its other face: the picture is mirrored too


@dataclass(frozen=True)
class Barcode:
    """One decoded instance and where it lies, in pixels of the input image."""

    part1: int
    part2: int
    number: int | None  # frame number, None on 23-module codes
    half: bool  # the "A" half frame
    start: tuple[float, float]  # start of module 0, on the clock track
    end: tuple[float, float]  # end of the last module: start -> end is the reading direction
    mirrored: bool
    quality: float  # 0..1, the weakest bit's margin

    @property
    def dx(self) -> str:
        return f"{self.part1}-{self.part2}"

    @property
    def label(self) -> str:
        return "" if self.number is None else f"{self.number}{'A' if self.half else ''}"

    @property
    def module(self) -> float:
        """Module width in pixels."""
        return math.dist(self.start, self.end) / (31 if self.number is not None else 23)


# ---------------------------------------------------------------- finding clocks
WORK_SIDE = 2000  # long side the search runs at
LEVEL_MODULES = (2.6, 5.6)  # module widths (pixels) looked for on each pyramid level
SEED_RUNS = 9  # alternating runs of one width that make a candidate clock
MIN_CONTRAST = 0.045  # density between bar and space
NOISE_MARGIN = 6.0  # ... and at least this many times the image noise, to look for bars
DENSITY_RANGE = 2.2  # above the bare light; denser is unlit surround, flattened
MAX_RUNS = 33  # longer stretches of equal runs lie off any code (a data track has 31 modules)
TEXTURE_MODULES = 10.0  # how far to each side of a seed its crowding is counted
CROWDING_STEPS = 4  # seeds are ranked by crowding in quarters
MAX_CANDIDATES = 32
PATCH_PX = 5.0  # resampled patch: pixels per module
PATCH_SIZE = (84, 16)  # modules along and across the strip
FINE_SHARE = 0.3  # of the bar-space range that single modules keep through blur, to binarise them alone
MIN_MARGIN = 0.3  # of every data bit, in half the bar-space contrast
MIN_ROWS = 3  # patch rows (0.2 module apart) that must read the same data track


def _density(linear: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Log density of the mean channel at the working size, that size over the
    input's, and the smallest bar contrast that stands out of the noise."""
    img = np.asarray(linear, dtype=np.float32)
    h, w = img.shape[:2]
    scale = min(1.0, WORK_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    if img.ndim == 3:
        img = cv2.transform(img, np.full((1, img.shape[2]), 1.0 / img.shape[2], np.float32))
    dens = cv2.log(np.maximum(img, np.float32(1e-5)))
    dens *= np.float32(-1.0 / math.log(10.0))
    lo = float(np.percentile(dens[::4, ::4], 0.5))
    np.clip(dens, lo, lo + DENSITY_RANGE, out=dens)
    # Noise from neighbouring pixels (robust: edges are few), off the flattened surround.
    left, right = dens[::4, :-1:3], dens[::4, 1::3]
    lit = (left < lo + DENSITY_RANGE) & (right < lo + DENSITY_RANGE)
    step = np.abs(right - left)[lit]
    sigma = 1.4826 / math.sqrt(2.0) * float(np.median(step)) if step.size else 0.0
    return dens, scale, max(MIN_CONTRAST, NOISE_MARGIN * sigma)


def _runs(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs along every row of a 0/1 image: (row, start, width, value).

    Each row ends with a run of width 0 and value 2, so a window of runs that
    crosses rows always fails a width test.
    """
    r, w = binary.shape
    padded = np.empty((r, w + 1), np.int8)
    padded[:, :w] = binary
    padded[:, w] = 2
    change = np.empty((r, w + 1), bool)
    change[:, 0] = True
    np.not_equal(padded[:, 1:], padded[:, :-1], out=change[:, 1:])
    rows, starts = np.nonzero(change)
    widths = np.empty_like(starts)
    widths[:-1] = starts[1:] - starts[:-1]
    widths[-1] = 0
    widths[starts == w] = 0
    return rows, starts, widths, padded[rows, starts]


def _binarize(light: np.ndarray, window: int, floor: float, vertical: bool = False, fine: int = 0) -> np.ndarray:
    """1 where denser than half way, in light, between the lightest and the
    densest of ``window`` pixels along the rows (columns when ``vertical``);
    0 there and where they differ by less than ``floor`` in density. Unlike a
    local mean, the midpoint does not drift on a wide bar or next to a
    printed number; and as camera blur mixes light, not density, a blurred
    bar keeps its width at a midpoint in light even when it is dense (black
    and white) or clear (slides).

    With ``fine``, the midpoint of that many pixels counts where they differ
    by much of the wide window's range: a blurred single bar next to a wide
    one misses the wide window's midpoint, not its own neighbours'; inside a
    wide bar, where they differ by the grain only, the wide window counts.
    """
    kernel = np.ones((window, 1) if vertical else (1, window), np.uint8)
    bright, dark = cv2.dilate(light, kernel), cv2.erode(light, kernel)
    level = bright + dark
    least = np.float32(10.0 ** floor)  # the floor as a ratio of light
    if fine:
        kernel = np.ones((fine, 1) if vertical else (1, fine), np.uint8)
        bright_f, dark_f = cv2.dilate(light, kernel), cv2.erode(light, kernel)
        enough = np.maximum(least, cv2.pow(bright / dark, FINE_SHARE))
        level = np.where(bright_f > dark_f * enough, bright_f + dark_f, level)
    return ((light + light < level) & (bright > dark * least)).astype(np.uint8)


def _seeds(light: np.ndarray, vertical: bool, floor: float) -> np.ndarray:
    """Stretches of equal alternating runs along the rows (columns when
    ``vertical``) that could lie on a code, as rows of (line across, start,
    end, module, runs, crowding).

    A track is at most 31 modules long and about two modules wide, so a longer
    stretch is something else. Crowding is the share of the lines up to
    TEXTURE_MODULES to one side that have runs as short as a seed's near it:
    beside a clock only its own track and the data track do, on the plain
    rebate; inside, or on the border of, a picture with fine detail or a
    regular texture (blinds, a fence, fabric) most of them.
    """
    m_lo, m_hi = LEVEL_MODULES
    window = 2 * round(1.25 * (m_lo + m_hi)) + 1
    h, w = light.shape
    none = np.empty((0, 6), np.float32)
    # Lines two pixels thick: less noise, half the work.
    if vertical:
        binary = np.ascontiguousarray(_binarize(cv2.resize(light, (w // 2, h), interpolation=cv2.INTER_AREA),
                                                window, floor, vertical=True).T)
    else:
        binary = _binarize(cv2.resize(light, (w, h // 2), interpolation=cv2.INTER_AREA), window, floor)
    rows, starts, widths, _ = _runs(binary)
    n = SEED_RUNS
    if len(widths) < n:
        return none
    # On a clock every bar and every space is one module wide: n runs span
    # n / 2 periods, and each bar + space pair is one period.
    width = widths.astype(np.float32)
    span = np.cumsum(np.concatenate(([0.0], width)))
    period = 2 * (span[n:] - span[:-n]) / n
    idx = np.nonzero((period >= 2 * m_lo) & (period < 2 * m_hi))[0]
    if not len(idx):
        return none
    win = np.lib.stride_tricks.sliding_window_view(width, n)[idx]
    pairs = win[:, :-1] + win[:, 1:]
    p = period[idx][:, None]
    ok = np.all(np.abs(pairs - p) <= np.maximum(1.5, 0.25 * p), axis=1)
    ok &= np.all((win >= np.maximum(1.0, 0.3 * p)) & (win <= 0.7 * p), axis=1)
    idx = idx[ok]
    if not len(idx):
        return none
    # Overlapping windows of one line make one stretch.
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    first, last = np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]] + n - 1
    runs = last + 1 - first
    keep = runs <= MAX_RUNS
    first, last, runs = first[keep], last[keep], runs[keep]
    if not len(first):
        return none
    line, a0, a1 = rows[first], starts[first], starts[last] + widths[last]
    module = (a1 - a0) / runs
    # Lines with runs as short as a seed's within a few modules along, counted to each side.
    lines, length = binary.shape
    kernel = np.ones((1, int(1.4 * m_hi) + 1), np.uint8)
    busy = ((binary ^ cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel))
            | (binary ^ cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)))
    slack = round(1.5 * (m_lo + m_hi))
    covered = cv2.dilate(busy, np.ones((1, 2 * slack + 1), np.uint8))
    columns, column = np.unique((a0 + a1) // 2, return_inverse=True)
    count = np.zeros((lines + 1, len(columns)), np.int32)
    np.cumsum(covered[:, columns], axis=0, out=count[1:])
    reach = np.maximum(2, np.round(TEXTURE_MODULES * module / 2)).astype(np.int64)  # lines are two pixels apart
    before = count[line, column] - count[np.maximum(0, line - reach), column]
    after = count[np.minimum(lines, line + 1 + reach), column] - count[line + 1, column]
    crowding = np.maximum(before, after) / reach
    return np.stack([2.0 * line + 1.0, a0, a1, module, runs, crowding], axis=1).astype(np.float32)


def _pyramid(light: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """(light, its pixels per working pixel), halving down to a few clock lengths."""
    levels = [(light, 1.0)]
    while min(levels[-1][0].shape) >= 80:
        img, factor = levels[-1]
        levels.append((cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA),
                       factor / 2))
    return levels


def _candidates(levels: list[tuple[np.ndarray, float]], vertical: bool, floor: float,
                skip: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> list[tuple[int, float, float, float]]:
    """Likely clocks: (level, cx, cy, module) in that level's pixels, leaving
    out those at working-image points where ``skip`` says so.

    The least crowded seeds come first, in quarters, and the longest of
    those: a texture can make many seeds longer than a clock's 23 runs, but
    not uncrowded ones.
    """
    found = []
    for k, (img, factor) in enumerate(levels):
        seeds = _seeds(img, vertical, floor)
        line, mid = seeds[:, 0], (seeds[:, 1] + seeds[:, 2]) / 2
        cx, cy = (line, mid) if vertical else (mid, line)
        keep = ~skip(cx / factor, cy / factor)
        found.append(np.stack([np.full(np.count_nonzero(keep), k), cx[keep], cy[keep], seeds[keep, 3],
                               np.minimum(CROWDING_STEPS - 1, np.floor(CROWDING_STEPS * seeds[keep, 5])),
                               -seeds[keep, 4]], axis=1))
    found = np.concatenate(found)
    found = found[np.lexsort((found[:, 5], found[:, 4]))]
    reach_x, reach_y = (4, 12) if vertical else (12, 4)  # in modules
    picked: list[tuple[int, float, float, float]] = []
    for k, cx, cy, module in found[:, :4].tolist():
        k = int(k)
        f, m = levels[k][1], module / levels[k][1]
        # Another stretch of the same clock: same module, a few modules along it
        # (on another line when tilted), but not as far across as a texture's border.
        if any(abs(cx / f - x / levels[j][1]) < reach_x * m and abs(cy / f - y / levels[j][1]) < reach_y * m
               and 0.7 < m * levels[j][1] / mj < 1.4 for j, x, y, mj in picked):
            continue
        picked.append((k, cx, cy, module))
        if len(picked) >= MAX_CANDIDATES:
            break
    return picked


# ---------------------------------------------------------------- decoding

def _direction(light: np.ndarray, cx: float, cy: float, module: float, vertical: bool) -> float:
    """Angle (radians) of the strip around a candidate.

    Every edge there runs along or across the strip (bars, holes, the film
    edge), so the gradients are averaged at four times their angle, where
    both families agree instead of cancelling.
    """
    r = round(10 * module)
    rx, ry = (r, 2 * r) if vertical else (2 * r, r)
    crop = light[max(0, int(cy) - ry):int(cy) + ry + 1, max(0, int(cx) - rx):int(cx) + rx + 1]
    base = math.pi / 2 if vertical else 0.0
    if min(crop.shape) < 3:
        return base
    crop = cv2.log(crop)  # in density, edges weigh by their contrast, not by how bright they are
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    xx, yy, xy = gx * gx, gy * gy, gx * gy
    energy = np.maximum(xx + yy, np.float32(1e-12))
    c4 = float((((xx - yy) ** 2 - 4 * xy * xy) / energy).sum())
    s4 = float((4 * xy * (xx - yy) / energy).sum())
    return base + 0.25 * math.atan2(s4, c4)


class _Patch:
    """The level's light resampled along the strip around a candidate, and its
    density: PATCH_PX pixels per module, columns along the strip, rows across it."""

    def __init__(self, light: np.ndarray, cx: float, cy: float, angle: float, module: float):
        self.s = module / PATCH_PX
        self.u = (math.cos(angle), math.sin(angle))
        self.v = (-self.u[1], self.u[0])
        self.w, self.h = int(PATCH_SIZE[0] * PATCH_PX), int(PATCH_SIZE[1] * PATCH_PX)
        self.cx, self.cy = cx, cy
        (ux, uy), (vx, vy), s = self.u, self.v, self.s
        m = np.array([[ux * s, vx * s, cx - (ux * self.w + vx * self.h) * s / 2],
                      [uy * s, vy * s, cy - (uy * self.w + vy * self.h) * s / 2]], np.float32)
        self.light = cv2.warpAffine(light, m, (self.w, self.h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REPLICATE)
        self.density = cv2.log(self.light) * np.float32(-1.0 / math.log(10.0))

    def to_level(self, px: float, py: float) -> tuple[float, float]:
        dx, dy = (px - self.w / 2) * self.s, (py - self.h / 2) * self.s
        return self.cx + dx * self.u[0] + dy * self.v[0], self.cy + dx * self.u[1] + dy * self.v[1]


def _clocks(light: np.ndarray, floor: float) -> list[tuple[_Layout, int, float, float, float]]:
    """Whole clock tracks in a patch's light: (layout, bar value, centre row, a, b).

    Module k, in reading order, starts at column a + b * k: ``b`` is negative
    when the code reads right to left in the patch.
    """
    # Each row averages a module across the track; along it, blur is left to the fine window.
    binary = _binarize(cv2.blur(light, (1, int(PATCH_PX))), int(6 * PATCH_PX) | 1, floor,
                       fine=int(2 * PATCH_PX) | 1)[::2]
    rows, starts, widths, values = _runs(binary)
    width = widths.astype(np.float32)
    span = np.cumsum(np.concatenate(([0.0], width)))
    hits: list[tuple[_Layout, int, int, float, float]] = []
    for layout in LAYOUTS:
        n = len(layout.clock) - 1
        if len(width) < n:
            continue
        first, last = width[:1 - n], width[n - 1:]
        u = np.maximum((span[n:] - span[:-n] - first - last) / (n - 2), 1e-3)
        forward = (first > 3.8 * u) & (first < 6.4 * u) & (last > 2.2 * u) & (last < 3.9 * u)
        backward = (last > 3.8 * u) & (last < 6.4 * u) & (first > 2.2 * u) & (first < 3.9 * u)
        idx = np.nonzero(forward | backward)[0]
        if not len(idx):
            continue
        singles = np.lib.stride_tricks.sliding_window_view(width, n)[idx, 1:-1]
        ok = np.all((singles > 0.45 * u[idx, None]) & (singles < 1.7 * u[idx, None]), axis=1)
        ks = np.asarray(layout.clock, np.float32)
        for i in idx[ok]:
            bounds = np.append(starts[i:i + n], starts[i + n - 1] + widths[i + n - 1]).astype(np.float32)
            if backward[i]:
                bounds = bounds[::-1]  # module 0 at the right: b comes out negative
            b, a = np.polyfit(ks, bounds, 1)
            if np.sqrt(np.mean((a + b * ks - bounds) ** 2)) < 0.3 * abs(b):
                # The first run is a bar either way, which gives the bars' value.
                hits.append((layout, int(values[i]), 2 * int(rows[i]), float(a), float(b)))
    # One clock reads on several neighbouring rows: keep it once, at its middle row.
    clocks = []
    for hit in hits:
        group = [h for h in hits if h[:2] == hit[:2] and abs(h[3] - hit[3]) < PATCH_PX and h[4] * hit[4] > 0]
        if hit is group[0]:
            centre = (min(h[2] for h in group) + max(h[2] for h in group)) / 2
            clocks.append((*hit[:2], centre, float(np.mean([h[3] for h in group])),
                           float(np.mean([h[4] for h in group]))))
    return clocks


@functools.cache
def _level_fit(layout: _Layout) -> tuple[np.ndarray, np.ndarray]:
    """Matrices that fit a straight line through the known 1s and the known 0s of a track."""
    def fit(ks):
        design = np.stack([np.asarray(ks, np.float64), np.ones(len(ks))], axis=1)
        full = np.stack([np.arange(layout.modules, dtype=np.float64), np.ones(layout.modules)], axis=1)
        return (full @ np.linalg.pinv(design)).astype(np.float32)  # (modules, len(ks))
    return (fit([k for k, bit in layout.known if bit]), fit([k for k, bit in layout.known if not bit]))


def _data(density: np.ndarray, layout: _Layout, rows: np.ndarray, a: float, b: float, bar: int):
    """The data track read along each of ``rows`` of a patch's density; the best
    that keeps every rule of the format: ((part1, part2, number, half), margin) or None."""
    h, w = density.shape
    xs = a + b * (np.arange(layout.modules, dtype=np.float32) + 0.5)
    rows = rows[(rows >= 2) & (rows <= h - 3)]
    if not len(rows) or xs.min() < 1 or xs.max() > w - 2:
        return None
    # Each sample averages half a module along the track and a module across it.
    smooth = cv2.blur(density, (max(1, int(abs(b) / 2)) | 1, int(PATCH_PX)))
    mx = np.broadcast_to(xs, (len(rows), layout.modules)).astype(np.float32)
    my = np.broadcast_to(rows[:, None], mx.shape).astype(np.float32)
    v = cv2.remap(smooth, mx, my, cv2.INTER_LINEAR)
    fit1, fit0 = _level_fit(layout)
    ones = [k for k, bit in layout.known if bit]
    zeros = [k for k, bit in layout.known if not bit]
    sign = 1.0 if bar else -1.0

    def scaled(signal: np.ndarray) -> np.ndarray:
        """Each sample against the straight lines through the known bits: +-1 at them, 0 half way."""
        hi, lo = signal[:, ones] @ fit1.T, signal[:, zeros] @ fit0.T
        return (signal - (hi + lo) / 2) / np.maximum((hi - lo) / 2, 1e-6)

    good = sign * ((v[:, ones] @ fit1.T) - (v[:, zeros] @ fit0.T)).min(axis=1) >= MIN_CONTRAST
    # Camera blur mixes light evenly, which density does not show on dense black and white or
    # slide bars; grain and shot noise are more even in density: each row is read in whichever
    # its weakest bit stands out more.
    q, q_light = scaled(sign * v), scaled(-sign * np.power(np.float32(10.0), -v))
    q = np.where((np.abs(q_light).min(axis=1) > np.abs(q).min(axis=1))[:, None], q_light, q)
    bits = (q > 0).astype(np.int64)
    for k, bit in layout.known:
        good &= bits[:, k] == bit
    last = 25 if layout.numbered else 18
    good &= (bits[:, 6:last].sum(axis=1) - bits[:, 13]) % 2 == bits[:, layout.parity]
    part1 = bits[:, 6:13] @ (1 << np.arange(6, -1, -1))
    margin = np.abs(q).min(axis=1)
    good &= (part1 > 0) & (margin >= MIN_MARGIN)
    if not good.any():
        return None
    # A real track reads the same across much of its height; a single row on
    # its blurred border can pass every rule by chance.
    packed = bits @ (1 << np.arange(layout.modules, dtype=np.int64))
    agree = np.array([np.count_nonzero(good & (packed == p)) for p in packed])
    good &= agree >= MIN_ROWS
    if not good.any():
        return None
    i = int(np.argmax(np.where(good, margin, -1.0)))
    row_bits = bits[i]
    number = int(row_bits[18:24] @ (1 << np.arange(5, -1, -1))) if layout.numbered else None
    fields = (int(part1[i]), int(row_bits[14:18] @ (1 << np.arange(3, -1, -1))), number,
              bool(layout.numbered and row_bits[24]))
    return fields, float(margin[i])


def _read(light: np.ndarray, cx: float, cy: float, module: float, vertical: bool, floor: float) -> list:
    """Codes around one candidate in a level's light: (fields, margin, start, end, mirrored), in its pixels."""
    patch = _Patch(light, cx, cy, _direction(light, cx, cy, module, vertical), module)
    found = []
    offsets = np.arange(1.2 * PATCH_PX, 3.6 * PATCH_PX + 0.5, 1.0, dtype=np.float32)
    for layout, bar, row, a, b in _clocks(patch.light, floor):
        best = None
        for side in (-1.0, 1.0):
            got = _data(patch.density, layout, row + side * offsets, a, b, bar)
            if got and (best is None or got[1] > best[1]):
                best = (*got, side)
        if best is None:
            continue
        fields, margin, side = best
        start, end = patch.to_level(a, row), patch.to_level(a + b * layout.modules, row)
        read = (end[0] - start[0], end[1] - start[1])
        data = (side * patch.v[0], side * patch.v[1])
        # From its usual face the data track lies to the right of the reading
        # direction as seen on screen: below a code read left to right.
        found.append((fields, margin, start, end, bool(read[0] * data[1] - read[1] * data[0] < 0)))
    return found


def find_barcodes(linear: np.ndarray, analysis=None) -> list[Barcode]:
    """Every DX code instance that reads in the image, each once.

    With an ``analysis`` whose ``extra["frame"]`` is known, the picture is
    skipped and the strip is looked for first along the frame's long side.
    """
    dens, scale, floor = _density(linear)
    h, w = dens.shape
    if min(h, w) < 40:
        return []
    frame = analysis.extra.get("frame") if analysis is not None else None
    order = (False, True)
    if frame is not None and (frame[3] - frame[1]) * h > (frame[2] - frame[0]) * w:
        order = (True, False)

    def in_picture(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if frame is None:
            return np.zeros(x.shape, bool)
        return (frame[0] * w < x) & (x < frame[2] * w) & (frame[1] * h < y) & (y < frame[3] * h)

    levels = _pyramid(cv2.exp(dens * np.float32(-math.log(10.0))))
    codes: list[Barcode] = []
    for vertical in order:
        for k, cx, cy, module in _candidates(levels, vertical, floor, in_picture):
            img, factor = levels[k]
            to_input = factor * scale
            if any(_near(c, cx / to_input, cy / to_input) for c in codes):
                continue  # a stretch of an instance already read
            for (part1, part2, number, half), margin, start, end, mirrored in _read(img, cx, cy, module, vertical,
                                                                                    floor):
                code = Barcode(part1, part2, number, half, (float(start[0] / to_input), float(start[1] / to_input)),
                               (float(end[0] / to_input), float(end[1] / to_input)), mirrored, min(1.0, margin))
                if not any(math.dist(code.start, c.start) < 3 * c.module for c in codes):
                    codes.append(code)
        if codes:
            break
    return codes


def _near(code: Barcode, x: float, y: float) -> bool:
    """Whether (x, y) lies on the code (with a few modules around it)."""
    sx, sy = code.start
    ux, uy = code.end[0] - sx, code.end[1] - sy
    length = math.hypot(ux, uy)
    along = ((x - sx) * ux + (y - sy) * uy) / length
    across = abs((x - sx) * uy - (y - sy) * ux) / length
    m = code.module
    return -3 * m < along < length + 3 * m and across < 6 * m


# ---------------------------------------------------------------- meaning

@functools.cache
def _table() -> dict:
    return json.loads((paths.data_dir() / "dx_codes.json").read_text(encoding="utf-8"))


def lookup(dx: str) -> tuple[str, str | None]:
    """Film name and Belka profile id for a DX number "part1-part2"."""
    table = _table()
    entry = table["films"].get(dx)
    if entry:
        return entry["film"], entry.get("profile")
    maker = table["makers"].get(dx.split("-")[0])
    return (f"{maker} (DX {dx})" if maker else f"DX {dx}"), None


def frame_label(codes: list[Barcode], point: tuple[float, float]) -> tuple[str, float]:
    """The printed frame number nearest ``point`` (input pixels), read from the
    codes, and the share of the numbered codes' quality that agrees with it.

    Numbers run half a frame apart along the reading direction, each printed
    just before its code, so every numbered code says which half frame lies
    at the point. Dust can turn two bits of one code's number and keep its
    parity: codes that place the sequence elsewhere than most do not count,
    and without a majority there is no number.
    """
    at = []  # (half-frame index at the point, quality)
    for c in codes:
        if c.number is None:
            continue
        length = math.dist(c.start, c.end)
        ux, uy = (c.end[0] - c.start[0]) / length, (c.end[1] - c.start[1]) / length
        gap = LABEL_GAP_MODULES * c.module
        x, y = c.start[0] - ux * gap, c.start[1] - uy * gap
        steps = ((point[0] - x) * ux + (point[1] - y) * uy) / (HALF_FRAME_MODULES * c.module)
        at.append((2 * c.number + int(c.half) + steps, c.quality))
    if not at:
        return "", 1.0
    # Codes of one sequence place the point within a fraction of a half frame of each other.
    anchor = max((v for v, _ in at), key=lambda v: sum(q for u, q in at if abs(u - v) < 0.5))
    group = [(v, q) for v, q in at if abs(v - anchor) < 0.5]
    weight = sum(q for _, q in group)
    share = weight / sum(q for _, q in at)
    value = round(sum(v * q for v, q in group) / weight)
    if share <= 0.5 or value < 0:
        return "", share
    return f"{value // 2}{'A' if value % 2 else ''}", share


def read_edge(linear: np.ndarray, analysis=None) -> EdgeInfo | None:
    """Film and frame number from the DX edge code, or None when no code reads.

    ``linear`` is the camera image the ``analysis`` was measured on, in any
    orientation and from either face of the film; the frame number is the one
    printed nearest the detected frame's centre (the image centre without it),
    "" when the numbered instances do not mostly agree on it.
    """
    codes = find_barcodes(linear, analysis)
    if not codes:
        return None
    votes: dict[str, float] = {}
    for c in codes:
        votes[c.dx] = votes.get(c.dx, 0.0) + c.quality
    dx = max(votes, key=votes.__getitem__)
    agree = [c for c in codes if c.dx == dx]
    # Each instance passed every rule of the format; agreeing ones add up.
    confidence = (1.0 - math.prod(1.0 - 0.7 * c.quality for c in agree)) * votes[dx] / sum(votes.values())
    h, w = np.asarray(linear).shape[:2]
    frame = analysis.extra.get("frame") if analysis is not None else None
    centre = ((frame[0] + frame[2]) / 2 * w, (frame[1] + frame[3]) / 2 * h) if frame else (w / 2, h / 2)
    number, share = frame_label(agree, centre)
    film, profile_id = lookup(dx)
    mirrored = sum(c.quality if c.mirrored else -c.quality for c in agree) > 0
    return EdgeInfo(dx, film, profile_id, number, round(confidence * share, 3), mirrored)
