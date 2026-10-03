"""Dust, fibre and scratch removal on the linear camera image.

Two ways to find the dirt:

* With a dark-field shot. For it the screen goes black under the film and
  lights a ring around it, so light only reaches the film obliquely, through
  the diffuser. Unscattered light misses the lens; what scatters it (dust,
  fibres, hairs, scratches) glows on a nearly black frame, while the film's own
  image shows only faintly, through the screen's black level and the glow that
  spreads in the diffuser. The shot is several stops slower and not
  flat-fielded, so it is normalised here: divided by its own smooth
  background, the faint film image (predicted from the bright shot) taken
  away, and the rest measured in units of its local noise. The two shots are
  taken one after the other, so the dark field is first moved onto the bright
  one (phase correlation of their edges), and the faint image is predicted at
  its brightest within a pixel: what is left of the misalignment, or a
  slightly softer focus, does not glow at every edge of the picture. A glow
  too small to be anything but a hot pixel of the long exposure counts only
  where the bright shot shows a shadow under it. A dark field that does not
  line up with the bright shot is not used (``darkfield_matches``): the bright
  shot is searched alone then.
* Without one, on the bright shot itself. Dust blocks the light from below,
  so on a capture of a negative it is a small, sharp, dark speck or line,
  darker than its surroundings by the same factor in every channel (it is
  grey: opaque, or nearly), unlike grain (one dye layer at a time) and most
  picture detail. A morphological closing of the log image measures how far
  each pixel dips below its neighbourhood; a dip counts only when it stands
  far out of the local spread of such dips and is neutral. But a point
  highlight of the picture (a star, a catchlight, a glint) is exactly that
  on a negative too, only bounded by the density the film can reach, while
  dust is nearly opaque. So a dip also has to be denser than the picture
  itself gets: than its densest large areas plus the headroom of a point
  highlight, and than the densest a negative reads in a capture. Lower
  strengths ask for more, higher ones less (at full strength small
  highlights can go). Half-transparent fibres and scratches mostly stay:
  this mode is deliberately conservative, and only looks inside the picture
  (edge print, frame numbers and the DX code are dark marks on the rebate too).

The repair works in log space (density): Telea inpainting carries edges and
gradients into each hole (colour, which changes slowly, comes from around
it), and the fine texture it smears (grain, or the
screen's pixel grid when it shows through) is copied from clean film next to
each piece of the dirt (a speck, or a short stretch of a long fibre or
scratch), or made up with the local grain's strength and colour where no
clean film is near, so a repair does not show as a smooth patch. Pixels
outside the mask are never touched.

Scales are written for a 2000 px preview and grow with the image, but the
same dirt covers fewer pixels on a smaller image, so a preview and a larger
image of the same frame can disagree on the smallest and faintest specks.
Images much larger than the half-size decode are searched at about that size
(a full-size export as the half-size detail view), smaller ones as they are.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from belka.core import pipeline as pl
from belka.i18n import _

UNIT_SIDE = 2000.0  # long side the pixel scales below are written for
DETECT_SIDE = 3100  # larger images are searched downsampled to about this
LOG_FLOOR = 1e-5  # linear floor before taking logs (no light at all)
DIP_CAP = 3.0  # ln: dips are measured up to this (95 % of the light blocked)
OPAQUE = 0.7  # ln: past this every channel counts the same, whatever its noise floor
SPECK_Z = 16.0  # how far out of the local spread of dips a speck must stand (at strength 0)
GLOW_Z = 10.0  # noise units a glow must reach on the dark-field shot (at strength 0)
# Px at the preview scale grown around what is found, plus the strength: the
# dark-field hysteresis already reaches into a speck's soft edge, the closing
# of the bright shot stops short of it.
GLOW_HALO = 0.5
SPECK_HALO = 1.0
SAMPLES = 20_000  # pixels the global statistics are measured on
# How much denser than the film base (ln) a dip on the bright shot alone must
# be at the default strength: than the picture's densest large areas plus the
# headroom of a point highlight, and than the densest a negative reads in a
# camera capture (about 1.45 in log10 density). Strength moves both by SLACK.
HEADROOM = 0.7
DENSE = 3.3
SLACK = 3.4
# A glow no larger than a hot pixel spread by demosaicing, downsampling and
# the search's blur ((HOT_SIDE + HOT_SPREAD * unit) px across) counts only
# where the bright shot shows a shadow (ln) under it.
HOT_SIDE = 3.0
HOT_SPREAD = 2.5
SHADOW = 0.12
REGISTER_SIDE = 384  # px of the window the two shots are registered on
MIN_RESPONSE = 0.15  # phase correlation peak below this: the shots show different things
MAX_SHIFT = 0.01  # of the long side: a dark field further off is not this frame's
MAX_GLOW = 0.05  # share of the film that may glow: more is a dark field that does not line up
TILE = 24.0  # px at the preview scale: long fibres and scratches borrow their grain piece by piece


def _unit(shape: tuple[int, ...]) -> float:
    return max(shape[0], shape[1]) / UNIT_SIDE


def _odd(n: float) -> int:
    return 2 * max(int(round(n)), 1) + 1


_MEAN3 = np.full((1, 3), 1.0 / 3.0, dtype=np.float32)
_SUM3 = np.ones((1, 3), dtype=np.float32)


def _grey(img: np.ndarray) -> np.ndarray:
    """Mean of the three channels (NumPy's mean over a last axis of 3 is ~10x slower)."""
    return cv2.transform(img, _MEAN3)


def _shrink(img: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter downsample by an integer factor: ``pipeline.downsample``, but
    through OpenCV's area resize, which is much faster on large images."""
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(w // factor, 1), max(h // factor, 1)), interpolation=cv2.INTER_AREA)


def _grow(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), se).astype(bool)


def _digest(img: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(img).tobytes(), digest_size=16).digest()


class _Cache:
    """A few recent results: dragging the strength re-measures nothing."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.items: dict = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            return self.items.get(key)

    def put(self, key, value) -> None:
        with self.lock:
            self.items[key] = value
            while len(self.items) > self.size:
                del self.items[next(iter(self.items))]

    def clear(self) -> None:
        with self.lock:
            self.items.clear()


# ---------------------------------------------------------------- film area

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """``mask`` with every enclosed gap filled (gaps touching the border stay)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~mask).view(np.uint8), connectivity=4)
    h, w = mask.shape
    x, y, bw, bh = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    enclosed = (x > 0) & (y > 0) & (x + bw < w) & (y + bh < h)
    enclosed[0] = False  # label 0 is the mask itself
    return mask | enclosed[labels]


def _light(small: np.ndarray) -> dict[str, np.ndarray | None]:
    """Bare light, unlit surround and film on the analysis image.

    ``pipeline.light_masks`` also calls the thinnest film "light" when its
    colour stands apart from the rest (tight framing, a carrier: no bare light
    in view), and dirt on the shadows of the negative would never be searched.
    Here bare light is only what clips, or a separate brighter population with
    a gap of density below it: any film base is denser than the light.
    """
    h, w, c = small.shape
    lit = _grey(small) > pl.NOISE_FLOOR
    clipped = np.maximum(np.maximum(small[..., 0], small[..., 1]), small[..., 2]) >= 0.98
    cand = lit & ~clipped
    bare = clipped.copy()
    n = np.count_nonzero(cand)
    if n >= 64:
        pick = np.flatnonzero(cand)[:: max(n // SAMPLES, 1)]
        peak = np.percentile(small.reshape(-1, c)[pick], 99.7, axis=0).astype(np.float32)
        dm = -_grey(cv2.log(np.clip(small / np.maximum(peak, 1e-6), 1e-6, None))) / np.float32(np.log(10.0))
        top = cand & (dm < 0.08)
        below = cand & (dm >= 0.08) & (dm < 0.25)
        if (np.count_nonzero(top) > 0.002 * n and np.count_nonzero(below) < 0.25 * np.count_nonzero(top)
                and np.count_nonzero(cand & (dm >= 0.25)) > 0.01 * n):
            bare |= top
    se = np.ones((2 * pl.EDGE_PX + 1,) * 2, np.uint8)
    light = cv2.dilate(bare.view(np.uint8), se).view(bool)
    unlit = cv2.dilate((~lit).view(np.uint8), se).view(bool)
    film = ~light & ~unlit
    if np.count_nonzero(bare) > 0.002 * h * w:
        # As in pipeline.light_masks: the film that matters lies on the light;
        # outside the box the bare light spans there is only glare.
        ys, xs = np.nonzero(bare)
        y0, y1 = np.percentile(ys, [0.5, 99.5]).astype(int)
        x0, x1 = np.percentile(xs, [0.5, 99.5]).astype(int)
        if (y1 - y0) > 0.15 * h and (x1 - x0) > 0.15 * w:
            inside = np.zeros((h, w), dtype=bool)
            inside[y0:y1 + 1, x0:x1 + 1] = True
            film &= inside
    field = pl.illumination_field(small, bare & ~clipped)
    return {"film": film, "light": light, "bare": bare, "unlit": unlit, "field": field}


@dataclass
class _Film:
    area: np.ndarray  # uint8 0/1 at the analysis size: film away from bare light, holes and cut edges
    picture: np.ndarray  # the same, inside the frame (the whole area when no frame is found)
    base: np.ndarray  # film base, camera RGB


_FILMS = _Cache(4)


def _film(img: np.ndarray, analysis: pl.Analysis | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(area, picture, base): where dirt is looked for with and without a dark
    field, at the size of ``img``, and the film base.

    Measured on ``img`` itself at the analysis resolution: ``pipeline.analyze``
    works on the oriented image, so only the film base is taken from
    ``analysis``.
    """
    h, w = img.shape[:2]
    small = _shrink(img, int(np.ceil(max(h, w) / pl.ANALYSIS_SIDE)))
    base = None
    if analysis is not None and np.asarray(analysis.base).size == 3:
        base = np.asarray(analysis.base, dtype=np.float32)
    key = (small.shape, None if base is None else base.tobytes(), _digest(small[::2, ::2]))
    film = _FILMS.get(key)
    if film is None:
        film = _measure_film(small, base)
        _FILMS.put(key, film)

    def full(m: np.ndarray) -> np.ndarray:
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).view(bool)

    return full(film.area), full(film.picture), film.base


def _measure_film(small: np.ndarray, base: np.ndarray | None) -> _Film:
    """``_film`` at the analysis resolution.

    Opaque dust reads as "no light" and would punch holes in the film mask,
    so enclosed gaps are filled back; sprocket holes are bare light, and stay out.
    """
    masks = _light(small)
    if base is None:
        base = pl.estimate_base(small, None, masks)
    area = _fill_holes(masks["film"]) & ~masks["light"]
    picture = area
    frame = pl.detect_frame(small, base, masks)
    if frame is not None:
        picture = np.zeros_like(area)
        picture[pl.crop_slices(small.shape, frame)] = True
        picture &= ~masks["light"]
    # One analysis pixel more off every edge: the blur of a hole or of the
    # strip's cut edge, which glows in the dark field.
    one = np.ones((3, 3), np.uint8)
    return _Film(cv2.erode(area.view(np.uint8), one), cv2.erode(picture.view(np.uint8), one),
                 np.asarray(base, dtype=np.float32))


# ---------------------------------------------------------------- detection helpers

def _local_mean(values: np.ndarray, weight: np.ndarray, radius: float,
                at: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Weighted box mean over about ``radius`` px, computed on a coarse grid;
    everywhere, or only at the pixels ``at`` (ys, xs)."""
    h, w = values.shape[:2]
    f = max(int(radius // 4), 1)
    size = (max(w // f, 1), max(h // f, 1))
    num = cv2.resize(values * weight, size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
    k = _odd(radius / f)
    mean = cv2.blur(num, (k, k)) / np.maximum(cv2.blur(den, (k, k)), 1e-6)
    if at is not None:
        return mean[np.minimum(at[0] // f, size[1] - 1), np.minimum(at[1] // f, size[0] - 1)]
    return cv2.resize(mean, (w, h), interpolation=cv2.INTER_LINEAR)


def _smooth_background(values: np.ndarray, region: np.ndarray, unit: float) -> np.ndarray:
    """Large-scale level of ``values`` over ``region``, unmoved by small bright spots.

    Block means of the region's pixels on a coarse grid, a median across
    blocks (a speck or a hair raises a block or two, not the median), then a
    soft blur; blocks with little film take their neighbours' values.
    """
    h, w = values.shape[:2]
    f = max(int(round(12 * unit)), 2)
    size = (max(w // f, 3), max(h // f, 3))
    weight = region.astype(np.float32)
    num = cv2.resize(values * weight, size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
    good = (den > 0.2).astype(np.float32)
    low = num / np.maximum(den, 1e-6)
    filled = cv2.GaussianBlur(low * good, (0, 0), 3.0) / np.maximum(cv2.GaussianBlur(good, (0, 0), 3.0), 1e-6)
    low = np.where(good > 0, low, filled).astype(np.float32)
    low = cv2.GaussianBlur(cv2.medianBlur(low, 5), (0, 0), 1.0)
    return cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)


def _rms(values: np.ndarray, weight: np.ndarray, radius: float) -> np.ndarray:
    """Local root mean square (interpolation can dip a hair below zero: clamped)."""
    return np.sqrt(np.maximum(_local_mean(values * values, weight, radius), 0.0))


def _sample(mask: np.ndarray) -> np.ndarray:
    """Flat indices of about ``SAMPLES`` pixels of ``mask``, on a regular grid
    (cheaper than listing them all)."""
    h, w = mask.shape
    step = max(int(np.sqrt(np.count_nonzero(mask) / SAMPLES)), 1)
    ys, xs = np.nonzero(mask[::step, ::step])
    return ys * (step * w) + xs * step


Spots = tuple[np.ndarray, np.ndarray, np.ndarray, int]  # ys, xs, component of each pixel, number of components


def _spots(score: np.ndarray, high: float, low: float, allowed: np.ndarray) -> Spots:
    """The pixels of the components of ``allowed & score > low`` that reach
    ``high`` somewhere (hysteresis). Sparse: the dirt is a tiny part of the image."""
    cand = allowed & (score > low)
    n, labels = cv2.connectedComponents(cand.view(np.uint8), connectivity=8)
    ys, xs = np.nonzero(cand)
    lab = labels[ys, xs]
    keep = np.zeros(n, dtype=bool)
    keep[lab[score[ys, xs] > high]] = True
    return _select(ys, xs, lab, keep)


def _select(ys: np.ndarray, xs: np.ndarray, lab: np.ndarray, keep: np.ndarray) -> Spots:
    """Only the components where ``keep[component]``, renumbered from 0."""
    ids = np.cumsum(keep) - 1
    sel = keep[lab]
    return ys[sel], xs[sel], ids[lab[sel]], int(np.count_nonzero(keep))


def _with(spots: Spots, where: np.ndarray) -> Spots:
    """Only the components with at least one pixel ``where`` (a test passed by
    some pixel: a component only grows with the strength, so it stays in)."""
    ys, xs, lab, count = spots
    keep = np.zeros(count, dtype=bool)
    keep[lab[where[ys, xs]]] = True
    return _select(ys, xs, lab, keep)


def _paint(shape: tuple[int, int], ys: np.ndarray, xs: np.ndarray, grow: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[ys, xs] = True
    return _grow(mask, grow)


# ---------------------------------------------------------------- the dark field

_RING8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)  # a pixel's 8 neighbours


def _jump_noise(img: np.ndarray) -> np.ndarray:
    """Pixel noise per channel, from the differences of sampled neighbours (robust)."""
    step = max(int(np.sqrt(img.shape[0] * img.shape[1] / SAMPLES)), 1)
    a = img[::step, : img.shape[1] - 1 : step]
    b = img[::step, 1::step][:, : a.shape[1]]
    jump = (a - b).reshape(a.shape[0] * a.shape[1], -1)
    return 1.4826 / np.sqrt(2.0) * np.median(np.abs(jump), axis=0) + 1e-9


def _without_hot_pixels(img: np.ndarray) -> np.ndarray:
    """``img`` with lone spikes in a single channel (hot pixels of a long
    exposure, as a half-size decode shows them) put back to the level of
    their neighbours, so they move neither the registration nor the noise.

    Dirt is never that: it scatters the screen's light into every channel,
    and even a speck smaller than a pixel spreads into the next ones. A
    demosaiced full-size decode spreads a hot photosite over several pixels
    and channels; ``_find_glow`` sorts those out by their missing shadow.
    """
    grey = _grey(img)
    ys, xs = np.nonzero(grey - cv2.dilate(grey, _RING8) > 2.5 * float(_jump_noise(grey[..., None])[0]))
    if ys.size == 0:
        return img
    noise = _jump_noise(img)
    h, w = grey.shape
    around = np.stack([img[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]).max(axis=0)
    spike = img[ys, xs] - around > 8.0 * noise
    alone = np.count_nonzero(spike, axis=1) == 1
    out = img.copy()
    c = np.argmax(spike, axis=1)[alone]
    out[ys[alone], xs[alone], c] = around[alone, c]
    return out


def _edges(img: np.ndarray, unit: float) -> np.ndarray:
    """Gradient magnitude of the log image: the same edges whatever their sign."""
    g = cv2.GaussianBlur(cv2.log(np.maximum(_grey(img), LOG_FLOOR)), (0, 0), max(1.0, unit))
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def _offset(bright: np.ndarray, dark: np.ndarray, region: np.ndarray, unit: float) -> tuple[float, float] | None:
    """How far (dx, dy) to move the dark-field shot onto the bright one; None
    when they do not show the same picture.

    Phase correlation of their edges in a window on the picture (``region``):
    its edges and the dirt's show in both shots, with different signs and
    strengths, which edge magnitudes ignore. Holes and the strip's edges are
    left out: every frame of the roll has them in the same places. Good to a
    few tenths of a pixel; ``_prepare_glow`` tolerates the rest.
    """
    h, w = region.shape
    m = cv2.moments(region.view(np.uint8), binaryImage=True)
    if m["m00"] <= 0:
        return None
    sh, sw = min(REGISTER_SIDE, h) & ~1, min(REGISTER_SIDE, w) & ~1
    y0 = int(np.clip(m["m01"] / m["m00"] - sh / 2, 0, h - sh))
    x0 = int(np.clip(m["m10"] / m["m00"] - sw / 2, 0, w - sw))
    win = (slice(y0, y0 + sh), slice(x0, x0 + sw))
    # Inside the frame's edge too: every frame has one in the same place.
    film = cv2.erode(region[win].view(np.uint8), np.ones((_odd(4 * unit),) * 2, np.uint8)).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(_edges(bright[win], unit) * film, _edges(dark[win], unit) * film,
                                            cv2.createHanningWindow((sw, sh), cv2.CV_32F))
    if response < MIN_RESPONSE or max(abs(dx), abs(dy)) > MAX_SHIFT * max(h, w):
        return None
    return -dx, -dy


@dataclass
class _Prepared:
    """What does not depend on the strength, for one image (and dark field)."""

    window: tuple[slice, slice]  # the part of the (shrunk) image worked on
    allowed: np.ndarray  # where a spot may lie
    z: np.ndarray  # how far each pixel stands out of the noise or the grain (the maps are float16: a few are cached)
    unit: float
    darkfield: bool  # found on a dark field (else on the bright shot alone)
    shadow: np.ndarray | None = None  # dark field: where the bright shot dips under the dark field's glow
    dip: np.ndarray | None = None  # bright shot: how far it dips (ln, mean of the channels)
    chroma: np.ndarray | None = None  # bright shot: how unequal the dip is across channels
    density: np.ndarray | None = None  # bright shot: ln below the film base
    top: float = 0.0  # bright shot: the picture's densest large areas (ln below the base)


def _prepare_glow(bright: np.ndarray, darkfield: np.ndarray, region: np.ndarray, picture: np.ndarray, unit: float,
                  window: tuple[slice, slice]) -> _Prepared | None:
    """Dirt on a dark-field shot: what glows above the background and the
    film's faint image, anywhere on the film (``region``). None when the dark
    field does not match the bright shot."""
    dark = _without_hot_pixels(darkfield)
    offset = _offset(bright, dark, picture, unit)
    if offset is None:
        return None
    h, w = region.shape
    if max(abs(offset[0]), abs(offset[1])) > 0.05:
        move = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
        dark = cv2.warpAffine(dark, move, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sigma = 0.6 * unit  # about a speck: averages the noise down, not the dirt
    d = cv2.GaussianBlur(_grey(dark), (0, 0), sigma)
    b = cv2.GaussianBlur(_grey(bright), (0, 0), sigma)
    d_bg = _smooth_background(d, region, unit)
    b_bg = _smooth_background(b, region, unit)
    rel = d / np.maximum(d_bg, 1e-9) - 1.0
    sel = _sample(region & (b_bg > LOG_FLOOR) & (d_bg > 0))
    if sel.size < 100:
        return None
    # The film's own picture shows in the dark field the way it shows in the
    # bright shot: as transmitted light, and (silver) as scatter that follows
    # its density. Dirt is dark in the bright shot too, so the predictors stop
    # at the darkest film: dirt cannot pass for a dense patch of picture.
    ratio = np.maximum(b, LOG_FLOOR) / np.maximum(b_bg, LOG_FLOOR)
    preds = [ratio - 1.0, np.log(ratio)]
    preds = [np.maximum(p, np.percentile(p.reshape(-1)[sel], 0.5)) for p in preds]
    fit = sel[::4]  # two numbers: a few thousand pixels pin them down
    x = np.stack([p.reshape(-1)[fit] for p in preds], axis=1).astype(np.float64)
    y = rel.reshape(-1)[fit].astype(np.float64)
    use = np.ones(y.size, dtype=bool)
    coef = np.zeros(2)
    for _ in range(3):  # least squares, dropping what does not fit (the dirt)
        xu = x[use]
        coef = np.linalg.lstsq(xu.T @ xu, xu.T @ y[use], rcond=None)[0]
        res = y - x @ coef
        spread = 1.4826 * np.median(np.abs(res[use] - np.median(res[use]))) + 1e-9
        use = np.abs(res) < 4.0 * spread
    if np.count_nonzero(use) < 0.5 * use.size:
        return None  # it disagrees with the bright shot all over
    # The prediction at its brightest within a pixel: a film edge a fraction
    # of a pixel off, or a little softer, is not dirt.
    pred = np.float32(coef[0]) * preds[0] + np.float32(coef[1]) * preds[1]
    resid = rel - cv2.dilate(pred, np.ones((3, 3), np.uint8))
    # Noise grows where the shot is darker: a local RMS, clipped so the dirt
    # itself does not raise it.
    spread = 1.4826 * float(np.median(np.abs(resid.reshape(-1)[sel]))) + 1e-9
    noise = _rms(np.clip(resid, -4.0 * spread, 4.0 * spread), region.astype(np.float32), 24 * unit)
    z = resid / np.maximum(noise, 0.25 * spread)
    if np.count_nonzero(z.reshape(-1)[sel] > 0.5 * GLOW_Z) > MAX_GLOW * sel.size:
        return None  # glowing all over: the film moved or turned between the shots
    # How far the bright shot dips under each pixel, for telling hot pixels from specks.
    log_b = cv2.log(np.maximum(b, LOG_FLOOR))
    k = _odd(2 * unit)
    shadow = cv2.morphologyEx(log_b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))) - log_b
    return _Prepared(window, region, z.astype(np.float16), unit, True, shadow=shadow > SHADOW)


def _find_glow(prep: _Prepared, strength: float) -> Spots:
    high = GLOW_Z * 2.0 ** (-1.15 * strength)
    spots = _spots(prep.z, high, 0.5 * high, prep.allowed)
    ys, xs, lab, count = spots
    if not count:
        return spots
    # A hot photosite that demosaicing spread over a few pixels and channels
    # glows like a tiny speck, but a speck also shades the bright shot.
    small = np.bincount(lab, minlength=count) <= (HOT_SIDE + HOT_SPREAD * prep.unit) ** 2
    shaded = np.zeros(count, dtype=bool)
    shaded[lab[prep.shadow[ys, xs]]] = True
    return _select(ys, xs, lab, ~small | shaded)


# ---------------------------------------------------------------- the bright shot alone

def _prepare_specks(bright: np.ndarray, region: np.ndarray, unit: float, base: np.ndarray,
                    window: tuple[slice, slice]) -> _Prepared | None:
    """Dirt on the bright shot alone: sharp, neutral, dense dips well beyond the grain."""
    log = cv2.GaussianBlur(cv2.log(np.maximum(bright, LOG_FLOOR)), (0, 0), 0.6 * unit)
    k = _odd(5 * unit)
    closed = cv2.morphologyEx(log, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    dips = closed - log
    dip = _grey(np.minimum(dips, DIP_CAP))
    # Half a closing away from the edge of the search: there the closing also
    # fills the corners the picture makes with the brighter rebate.
    inside = cv2.erode(region.view(np.uint8), np.ones((k, k), np.uint8)).view(bool)
    sel = _sample(inside)
    if sel.size < 100:
        return None
    flat = dip.reshape(-1)[sel]
    med = float(np.median(flat))
    mad = 1.4826 * float(np.median(np.abs(flat - med))) + 1e-6
    # How deep dips usually are around here (grain, texture, the screen's
    # pixel grid showing through): a local mean and spread, clipped so the
    # dirt itself does not raise them.
    capped = np.minimum(dip, med + 6.0 * mad)
    weight = region.astype(np.float32)
    mean = _local_mean(capped, weight, 24 * unit)
    spread = np.sqrt(np.maximum(_local_mean(capped * capped, weight, 24 * unit) - mean * mean, 0.0))
    z = (dip - mean) / np.maximum(spread, 0.25 * mad)
    # Dust dims every channel by the same factor; grain and most picture detail do not.
    r, g, b = cv2.split(np.minimum(dips, OPAQUE))
    spread_rgb = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    chroma = spread_rgb / np.maximum((r + g + b) * np.float32(1.0 / 3.0), 1e-6)
    level = np.float32(np.mean(np.log(np.maximum(base, LOG_FLOOR))))
    density = level - _grey(log)
    top = float(np.percentile((level - _grey(closed)).reshape(-1)[sel], 99.5))
    half = np.float16
    return _Prepared(window, inside, z.astype(half), unit, False, dip=dip.astype(half), chroma=chroma.astype(half),
                     density=density.astype(half), top=top)


def _find_specks(prep: _Prepared, strength: float) -> Spots:
    high = SPECK_Z * 2.0 ** (-1.2 * strength)
    depth = 0.25 - 0.15 * strength
    need = max(prep.top + HEADROOM, DENSE) + SLACK * (0.5 - strength)
    spots = _spots(prep.z, high, 0.5 * high, prep.allowed & (prep.dip > 0.5 * depth))
    # Deep, grey and denser than the picture gets, all at one pixel (the core of a speck).
    return _with(spots, (prep.dip > depth) & (prep.chroma < 0.12 + 0.3 * strength) & (prep.density > need))


# ---------------------------------------------------------------- detection

_PREPARED = _Cache(3)


def _forget() -> None:
    """Drop every cached measurement (tests time cold calls with it)."""
    _FILMS.clear()
    _PREPARED.clear()


def _check(bright: np.ndarray, darkfield: np.ndarray | None) -> None:
    if darkfield is not None and darkfield.shape != bright.shape:
        # A real file can do this (the camera's image size changed between the two shots).
        raise ValueError(_("La toma de campo oscuro mide {dw}×{dh} px y el fotograma {fw}×{fh} px.").format(
            dw=darkfield.shape[1], dh=darkfield.shape[0], fw=bright.shape[1], fh=bright.shape[0]))


def _prepare(bright: np.ndarray, darkfield: np.ndarray | None, analysis: pl.Analysis | None
             ) -> tuple[_Prepared | None, int]:
    """The strength-independent part of ``dust_mask``, cached, and the shrink factor."""
    h, w = bright.shape[:2]
    _check(bright, darkfield)
    f = int(np.ceil(max(h, w) / DETECT_SIDE))
    img = _shrink(np.asarray(bright, dtype=np.float32), f)
    dark = None if darkfield is None else _shrink(np.asarray(darkfield, dtype=np.float32), f)
    base = None if analysis is None else np.asarray(analysis.base, dtype=np.float32).tobytes()
    key = (img.shape, _digest(img[::13, ::13]), None if dark is None else _digest(dark[::13, ::13]), base)
    prep = _PREPARED.get(key)
    if prep is None:
        prep = _measure(img, dark, analysis)
        _PREPARED.put(key, prep)
    return prep, f


def _measure(img: np.ndarray, dark: np.ndarray | None, analysis: pl.Analysis | None) -> _Prepared | None:
    area, picture, base = _film(img, analysis)
    unit = _unit(img.shape)

    def around(region: np.ndarray) -> tuple[slice, slice]:
        # Only the film's bounding box is worked on (the strip is often a
        # small part of the camera's view), with a margin for the filters.
        x, y, bw, bh = cv2.boundingRect(region.view(np.uint8))
        m = int(8 * unit) + 2
        return (slice(max(y - m, 0), min(y + bh + m, img.shape[0])),
                slice(max(x - m, 0), min(x + bw + m, img.shape[1])))

    if dark is not None and area.any():
        win = around(area)
        prep = _prepare_glow(img[win], dark[win], area[win], picture[win], unit, win)
        if prep is not None:
            return prep
    if not picture.any():
        return None
    win = around(picture)
    return _prepare_specks(img[win], picture[win], unit, base, win)


def dust_mask(
    bright: np.ndarray,
    darkfield: np.ndarray | None = None,
    strength: float = 0.5,
    analysis: pl.Analysis | None = None,
) -> np.ndarray:
    """Pixels covered by dust, fibres or scratches (bool H x W).

    ``bright`` and ``darkfield`` are linear camera RGB as loaded: unoriented and
    the same size. ``strength`` 0..1: 0 finds nothing, 0.5 is a sensible
    default, higher is more sensitive (and, without a dark field, riskier for
    small highlights of the picture). ``analysis`` lends its film base; the
    film area and the frame are found on ``bright`` itself, since
    ``pipeline.analyze`` measures the oriented image.
    """
    h, w = bright.shape[:2]
    s = float(np.clip(np.nan_to_num(strength), 0.0, 1.0))
    _check(bright, darkfield)
    if s <= 0.0 or h < 16 or w < 16:
        return np.zeros((h, w), dtype=bool)
    prep, f = _prepare(bright, darkfield, analysis)
    if prep is None:
        return np.zeros((h, w), dtype=bool)
    shape = prep.allowed.shape
    if prep.darkfield:
        ys, xs, _, _ = _find_glow(prep, s)
        found = _paint(shape, ys, xs, int(prep.unit * (GLOW_HALO + s) + 0.5))
    else:
        ys, xs, _, _ = _find_specks(prep, s)
        found = _paint(shape, ys, xs, int(prep.unit * (SPECK_HALO + s) + 0.5))
    small = np.zeros((max(h // f, 1), max(w // f, 1)), dtype=bool)
    small[prep.window] = found
    if f == 1:
        return small
    return cv2.resize(small.view(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).view(bool)


def darkfield_matches(bright: np.ndarray, darkfield: np.ndarray, analysis: pl.Analysis | None = None) -> bool:
    """Whether ``dust_mask`` can use this dark-field shot: it lines up with the
    bright one and shows the same film. When not, the bright shot is searched
    alone (cached: cheap right after ``dust_mask`` on the same pair)."""
    prep, _f = _prepare(bright, darkfield, analysis)
    return prep is not None and prep.darkfield


# ---------------------------------------------------------------- repair

def _inpaint(log: np.ndarray, hole: np.ndarray, radius: float, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """``log`` (3 channels) with the holes, at (ys, xs), filled from around them.

    Telea inpainting carries the edges and gradients of the grey (the mean of
    the channels) into each hole; colour changes slowly, and is taken from
    what surrounds the hole (``_around``), at a third of the cost of
    inpainting every channel. OpenCV 5.0's float path of INPAINT_TELEA is
    broken (it rings far outside the input's range); its 16-bit path is right
    and, spread over the box's range of log values, finer than any grain.
    """
    grey = _grey(log)
    lo, hi = float(grey.min()), float(grey.max())
    scale = 65535.0 / max(hi - lo, 1e-6)
    words = np.round((grey - lo) * scale).astype(np.uint16)
    filled = cv2.inpaint(words, hole.view(np.uint8), radius, cv2.INPAINT_TELEA)[ys, xs]
    around = _around(log, hole, ys, xs)
    out = log.copy()
    out[ys, xs] = around - around.mean(axis=1, keepdims=True) + (
        filled.astype(np.float32) * np.float32(1.0 / scale) + np.float32(lo))[:, None]
    return out


def _around(img: np.ndarray, hole: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """The average of ``img`` around each hole pixel (ys, xs): from the finest
    level of a pyramid of the known pixels where enough of them back it."""
    weight = (~hole).astype(np.float32)
    num = img.copy()
    num[ys, xs] = 0.0
    out = np.zeros((ys.size, img.shape[-1]), dtype=np.float32)
    todo = np.arange(ys.size)
    level = 0
    while todo.size and min(weight.shape) > 1:
        num, weight, level = cv2.pyrDown(num), cv2.pyrDown(weight), level + 1
        y = np.minimum(ys[todo] >> level, weight.shape[0] - 1)
        x = np.minimum(xs[todo] >> level, weight.shape[1] - 1)
        support = weight[y, x]
        ok = support > 0.25
        out[todo[ok]] = num[y[ok], x[ok]] / support[ok, None]
        todo = todo[~ok]
    return out


def _texture(detail: np.ndarray, hole: np.ndarray, clean: np.ndarray, energy: np.ndarray, level: np.ndarray,
             rms: np.ndarray, sample: np.ndarray, unit: float, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Fine texture (grain, or whatever pattern the film shows) for the hole
    pixels (ys, xs), per channel. ``energy`` is the fine detail's strength at
    each pixel; ``level`` and ``rms`` are the grain's strength around each hole
    pixel, in all and per channel; ``sample`` is some clean detail (its colour).

    Each piece of the dirt (a speck, or a stretch of a long fibre or scratch)
    copies it from a clean place next to it: real grain, with its colour and
    coarseness. Candidates sit a little more than the piece's thickness away
    in eight directions; the one whose texture is as strong as the film's
    around it wins (an edge nearby would be stronger, a smoother patch
    weaker). A piece with no clean place near gets noise with the local
    grain's strength and the film's grain colour.
    """
    h, w = hole.shape
    _n, labels = cv2.connectedComponents(hole.view(np.uint8), connectivity=8)
    t = max(int(round(TILE * unit)), 8)
    tiles_x = -(-w // t)
    key = labels[ys, xs].astype(np.int64) * (-(-h // t) * tiles_x) + (ys // t) * tiles_x + xs // t
    _keys, piece = np.unique(key, return_inverse=True)
    count = int(piece.max()) + 1
    size = np.bincount(piece, minlength=count).astype(np.float64)
    thick = np.zeros(count, dtype=np.float32)
    np.maximum.at(thick, piece, cv2.distanceTransform(hole.view(np.uint8), cv2.DIST_L2, 3)[ys, xs])
    thick *= 2.0
    target = np.bincount(piece, level, count) / size
    cap = 9.0 * target  # one edge pixel must not outweigh the rest
    # Padded and flattened, so a candidate is one gather; off the box counts as dirty.
    pad = int(3.0 * (float(thick.max()) + 2.0)) + 1
    width = w + 2 * pad
    blocked = np.pad(~clean, pad, constant_values=True).reshape(-1)
    energy = np.pad(energy, pad).reshape(-1)
    at = (ys + pad).astype(np.int64) * width + (xs + pad)
    best = np.full(count, np.inf)
    dy_best = np.zeros(count, dtype=np.int64)
    dx_best = np.zeros(count, dtype=np.int64)
    settled = np.zeros(count, dtype=bool)
    active = np.arange(ys.size)  # pixels of the pieces still looking
    for reach in (1.2, 2.0, 3.0):
        own = piece[active]
        step = reach * (thick + 2.0)
        for angle in np.arange(8) * (np.pi / 4):
            dy, dx = np.rint(step * np.sin(angle)).astype(np.int64), np.rint(step * np.cos(angle)).astype(np.int64)
            there = at[active] + (dy * width + dx)[own]
            bad = np.bincount(own, blocked[there], count)
            strength = np.bincount(own, np.minimum(energy[there], cap[own]), count) / size
            score = np.abs(np.log((strength + 1e-12) / target))
            score[bad > 0.02 * size] = np.inf
            better = (score < best) & ~settled
            best[better], dy_best[better], dx_best[better] = score[better], dy[better], dx[better]
        settled |= best < 0.5
        active = active[~settled[piece[active]]]
        if not active.size:
            break
    found = np.isfinite(best)[piece]
    out = np.empty((ys.size, detail.shape[-1]), dtype=np.float32)
    out[found] = detail[np.clip(ys[found] + dy_best[piece[found]], 0, h - 1),
                        np.clip(xs[found] + dx_best[piece[found]], 0, w - 1)]
    if not found.all():
        c = detail.shape[-1]
        corr = np.corrcoef(sample.astype(np.float64), rowvar=False) if sample.shape[0] > 10 else np.eye(c)
        mix = np.linalg.cholesky(np.nan_to_num(np.atleast_2d(corr)) + 1e-6 * np.eye(c))
        noise = np.random.default_rng(0).standard_normal((int(np.count_nonzero(~found)), c)) @ mix.T
        out[~found] = noise * rms[~found]
    return out


def repair(bright: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``bright`` with the masked pixels filled in; every other pixel unchanged.

    Worked on the box around the dirt, wide enough to borrow texture from
    beside each piece of it (``_texture``).
    """
    out = bright.copy()
    if mask is None or not np.any(mask):
        return out
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != bright.shape[:2]:
        raise ValueError(f"mask is {mask.shape}, the image {bright.shape[:2]}")
    h, w = mask.shape
    unit = _unit(mask.shape)
    sigma = max(1.5, 1.5 * unit)  # grain lives below this scale: it is borrowed, the rest inpainted
    x, y, bw, bh = cv2.boundingRect(mask.view(np.uint8))
    inner = mask[y:y + bh, x:x + bw]
    thick = 2.0 * float(cv2.distanceTransform(np.pad(inner, 1).view(np.uint8), cv2.DIST_L2, 3).max())
    m = int(3.0 * (thick + 2.0) + 4 * sigma + 12 * unit) + 2
    box = (slice(max(y - m, 0), min(y + bh + m, h)), slice(max(x - m, 0), min(x + bw + m, w)))
    hole = mask[box]
    ys, xs = np.nonzero(hole)
    log = cv2.log(np.maximum(bright[box], LOG_FLOOR).astype(np.float32))
    log = _inpaint(log, hole, max(2.0, unit), ys, xs)
    low = cv2.GaussianBlur(log, (0, 0), sigma)
    detail = log - low
    # Inpainted holes have no grain to lend, nor has the blur around them.
    clean = ~_grow(hole, int(np.ceil(2 * sigma)))
    energy = cv2.transform(detail * detail, _SUM3)
    level = np.maximum(_local_mean(energy, clean.astype(np.float32), 12 * unit, at=(ys, xs)), 1e-12)
    # The grain's colour: how its strength splits between the channels.
    sample = detail.reshape(-1, detail.shape[-1])[_sample(clean)]
    share = np.mean(sample * sample, axis=0) if sample.size else np.ones(detail.shape[-1])
    rms = np.sqrt(level[:, None] * (share / max(float(share.sum()), 1e-12))).astype(np.float32)
    texture = _texture(detail, hole, clean, energy, level, rms, sample, unit, ys, xs)
    # Fine detail up to the grain's size comes from the borrowed patch, so an
    # edge it happens to cross is not pasted in; what the inpainting carries
    # beyond that size is an edge running through the hole, and stays.
    grain = 3.0 * rms
    inpainted = detail[ys, xs]
    edges = inpainted - np.clip(inpainted, -grain, grain)
    values = np.exp(low[ys, xs] + edges + np.clip(texture, -grain, grain))
    out[box][ys, xs] = values.astype(out.dtype)
    return out


def remove_dust(
    bright: np.ndarray,
    darkfield: np.ndarray | None,
    strength: float,
    analysis: pl.Analysis | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(repaired image, mask): ``dust_mask`` then ``repair``."""
    mask = dust_mask(bright, darkfield, strength, analysis)
    return repair(bright, mask), mask
