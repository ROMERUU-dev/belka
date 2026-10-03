"""Automatic tone, like the Auto button of Lightroom's Tone panel.

Auto starts from the scene's plain print (no stretch, contrast 1, exposed
so its brightest 0.1 % just reach white) and takes a view on it: the
median moves half way to middle grey, unless the scene has no room on that
side (a night keeps its dark median, snow its light one); the darkest 0.1 %
go most of the way to black; Shadows opens a lower quarter sunk deep in the
dark and Highlights holds back an upper quarter spreading into the light;
Contrast moves a little. Nothing past the extreme 0.1 % clips.

Exposure, whites and blacks all act on the film's log-exposure scale before
the paper curve, where together they are one straight line: three sliders
for two degrees of freedom. So everything is fitted at once, as a small
least-squares problem whose regularisation shares any stretch equally
between whites and blacks and keeps contrast near 1 and highlights and
shadows at 0 unless their own targets need them. A single slider's Auto
answers for that slider's own target.

Rendering every candidate would be slow, so the fit runs on a model of the
print stage (``pipeline.PaperCurve``, saturation, sRGB, and the
highlights/shadows curves read off ``adjust`` on a grey ramp) fed with a few
hundred pixels picked at quantiles of the picture. The fit is then checked
on the real chain (``develop.invert`` + ``develop.finish``) over the whole
reduced picture and redone with the model's error added, so the targets are
met on what the user will see.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable

import cv2
import numpy as np

from belka.core import develop as dv
from belka.core import pipeline as pl
from belka.core.film import FilmProfile

TONE_FIELDS = ("exposure", "contrast", "highlights", "shadows", "white", "black")
NEUTRAL = {"exposure": 0.0, "contrast": 1.0, "highlights": 0.0, "shadows": 0.0, "white": 0.0, "black": 0.0}

MAX_SIDE = 320
TAIL = 0.001  # share of the picture allowed past each end's target
WHITE_LEVEL = 0.978  # the brightest channel's top 0.1 % at 249/255: under the clipping warning (254.5/255) with room for the grain and detail a reduced picture smooths away
BLACK_LEVEL = 0.012  # the darkest 0.1 % (luminance) from 3/255, over it (0.5/255)
MID_GREY = 0.46  # sRGB of 18 % grey: where an average scene's median goes
MID_PULL = 0.5  # share of the way from the scene's own median to MID_GREY
BLACK_PULL = 0.7  # share of the way from the scene's darkest to BLACK_LEVEL (in log)
QUARTILES = (0.25, 0.75)  # the parts of the picture Shadows and Highlights look after
QUARTILE_SLACK = 0.25  # stops the quartiles may spread beyond the scene's own
LOW_DEPTH = 1.5  # stops under middle grey the lower quartile may sink before Shadows opens it
FLAT_SPREAD = 2.0  # stops of scene range below which the ends count for less
ROOM = 2.0  # stops between the median and the end it moves toward for the full MID_PULL
CONTRAST_RANGE = (0.75, 1.5)  # Auto only nudges contrast
HS_LIMIT = 0.7

# How far each target may miss for the same cost in the least-squares fit.
_SIGMA = {"mid": 0.02, "white_over": 0.003, "white_under": 0.015, "black_under": 0.1, "black_over": 0.3,
          "low": 0.04, "high": 0.02,
          # and how far each slider may go for that cost: equal for whites and
          # blacks, which then share any stretch between them
          "exposure": 6.0, "contrast": 0.15, "white": 0.18, "black": 0.18, "highlights": 0.2, "shadows": 0.2}
_BLACK_EPS = 0.005  # blacks are compared as log(level + eps): equal steps look alike near black
_CLIP_HI, _CLIP_LO = 254.5 / 255, 0.5 / 255  # adjust.clipping_masks
_CLIP_SLOPE = 3.0  # past clipping, the miss keeps growing with the clipped share

_STEPS = {"exposure": 0.02, "white": 0.004, "black": 0.004, "contrast": 0.01, "highlights": 0.02, "shadows": 0.02}
_BOUNDS = {"exposure": (-5.0, 5.0), "white": (-0.3, 0.3), "black": (-0.3, 0.3), "contrast": CONTRAST_RANGE,
           "highlights": (-HS_LIMIT, 0.0), "shadows": (0.0, HS_LIMIT)}
_LUT_T = np.linspace(-20.0, 6.0, 2049)  # log2 of scene light, over the paper's whole range
_LUT_SCALE = (len(_LUT_T) - 1) / (_LUT_T[-1] - _LUT_T[0])
_LUT_CONTRASTS = np.geomspace(*CONTRAST_RANGE, 7)
_HS_L = np.linspace(0.0, 1.0, 257)  # OKLab lightness of the highlights/shadows curves
_DECODE = pl.srgb_decode(np.linspace(0.0, 1.0, 4096, dtype=np.float32))
_EXACT_TAIL = 0.03
# Quantiles of the pixels the fit runs on: luminance over the whole range,
# densest at the dark end (Blacks), and the top of the brightest channel (Whites).
_LUM_LEVELS = np.concatenate([np.geomspace(0.0002, _EXACT_TAIL, 24), np.linspace(0.04, 0.99, 96)])
_TOP_LEVELS = 1.0 - np.geomspace(0.6, 0.0002, 48)


def _encode(v: float) -> float:
    return 12.92 * v if v <= 0.0031308 else 1.055 * max(v, 0.0) ** (1 / 2.4) - 0.055


def _decode(v: float) -> float:
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


_CLIP_HI_LIN, _CLIP_LO_LIN = _decode(_CLIP_HI), _decode(_CLIP_LO)


def _lookup(table: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Linear interpolation in a table sampled at 0, 1, 2...; ``pos`` beyond the ends reads the ends."""
    pos = np.clip(pos, 0.0, len(table) - 1.0001)
    i = pos.astype(np.intp)
    lo = table[i]
    return lo + (pos - i) * (table[i + 1] - lo)


@functools.lru_cache(maxsize=32)
def _paper(gamma: float) -> np.ndarray:
    """Display (linear) of a grey on a paper of ``gamma``, over _LUT_T (read-only)."""
    table = pl.PaperCurve(gamma=gamma).apply(np.exp2(_LUT_T).astype(np.float32)).astype(np.float64)
    table.flags.writeable = False
    return table


@functools.cache
def _hs_curves() -> tuple[np.ndarray, np.ndarray]:
    """Lightness change of Shadows +1 and Highlights −1 over _HS_L, read off adjust on a grey ramp.

    The adjustment works on the local mean, which on a bare ramp's last
    steps is one-sided: the ends are read in flat black and white margins
    instead, so pure white stays white as it does in a blown sky.
    """
    from belka.core import adjust

    pad = 64
    light = np.concatenate([np.zeros(pad), _HS_L, np.ones(pad)])
    ramp = pl.srgb_encode((light ** 3).astype(np.float32))
    ramp = np.ascontiguousarray(np.repeat(ramp[None, :, None], 3, axis=2))
    curves = []
    for change in ({"shadows": 1.0}, {"highlights": -1.0}):
        out = adjust.apply_adjustments(ramp, pl.DevelopSettings(**change))[0, :, 1]
        out = np.concatenate([out[pad // 2:pad // 2 + 1], out[pad + 1:-pad - 1], out[-pad // 2:-pad // 2 + 1]])
        curve = np.cbrt(pl.srgb_decode(out).astype(np.float64)) - _HS_L
        curve.flags.writeable = False
        curves.append(curve)
    return curves[0], curves[1]


def _columns(a: np.ndarray, op: np.ufunc) -> np.ndarray:
    """``op`` reduced over the columns of (n, c): numpy's axis=1 reductions are slow on so few."""
    out = a[:, 0]
    for i in range(1, a.shape[1]):
        out = op(out, a[:, i])
    return out


def _brightest(rgb: np.ndarray) -> np.ndarray:
    return _columns(rgb, np.maximum)


def _share_below(levels: np.ndarray, values: np.ndarray, v: float) -> float:
    """Share of the picture under ``v``, from sorted ``values`` at quantile ``levels``."""
    i = int(np.searchsorted(values, v))
    if i == 0:
        return 0.0
    if i == len(values):
        return float(levels[-1])
    a, b = values[i - 1], values[i]
    return float(levels[i - 1] + (levels[i] - levels[i - 1]) * (v - a) / max(b - a, 1e-12))


def _statistics(lum_levels: np.ndarray, lum: np.ndarray, top_levels: np.ndarray, top: np.ndarray) -> np.ndarray:
    """(median, white end, log black end, lower quartile, upper quartile), sRGB.

    From sorted linear luminance and brightest-channel values at their
    quantile levels. Past clipping the ends keep moving with the clipped
    share, so the fit still sees which way to go.
    """
    clipped = 1.0 - _share_below(top_levels, top, _CLIP_HI_LIN)
    white = (_CLIP_HI + _CLIP_SLOPE * (clipped - TAIL) if clipped > TAIL
             else _encode(np.interp(1 - TAIL, top_levels, top)))
    crushed = _share_below(lum_levels, lum, _CLIP_LO_LIN)
    black = (np.log(_CLIP_LO + _BLACK_EPS) - 10 * _CLIP_SLOPE * (crushed - TAIL) if crushed > TAIL
             else np.log(_encode(np.interp(TAIL, lum_levels, lum)) + _BLACK_EPS))
    mid, low, high = (_encode(v) for v in np.interp((0.5, *QUARTILES), lum_levels, lum))
    return np.array([mid, white, black, low, high])


def _pick(key: np.ndarray, levels: np.ndarray, tail: str) -> np.ndarray:
    """Indices of the pixels at quantile ``levels`` of ``key``.

    Ranked on a subsample of about 6 000 pixels (sorting them all costs
    more than the fit), and exactly within the 3 % ``tail`` ("top" or
    "bottom") whose 0.1 % the clipping targets look at.
    """
    n = len(key)
    sub = np.arange(0, n, max(1, n // 6000))
    sub = sub[np.argsort(key[sub])]
    idx = sub[np.round(levels * (len(sub) - 1)).astype(int)]
    k = min(n, int(np.ceil(_EXACT_TAIL * n)) + 1)
    if tail == "top":
        part, inside, offset = np.argpartition(key, n - k)[n - k:], levels >= 1 - _EXACT_TAIL, n - k
    else:
        part, inside, offset = np.argpartition(key, k - 1)[:k], levels <= _EXACT_TAIL, 0
    part = part[np.argsort(key[part])]
    idx[inside] = part[np.clip(np.round(levels[inside] * (n - 1)).astype(int) - offset, 0, k - 1)]
    return idx


class _Picture:
    """A reduced picture, its film pixels and the quantile pixels the fit runs on."""

    def __init__(self, geo: np.ndarray, analysis: pl.Analysis, settings: pl.DevelopSettings, profile: FilmProfile,
                 camera_matrix: np.ndarray | None, rgb_sequential: bool) -> None:
        self.analysis, self.profile = analysis, profile
        self.camera_matrix, self.rgb_sequential = camera_matrix, rgb_sequential
        h, w = geo.shape[:2]
        f = -(-max(h, w) // MAX_SIDE)
        if f > 1:
            # By a whole factor, where OpenCV's area resize is a plain (fast) box filter.
            geo = cv2.resize(geo[:(h // f) * f or h, :(w // f) * f or w], (max(1, w // f), max(1, h // f)),
                             interpolation=cv2.INTER_AREA)
        self.small = np.ascontiguousarray(geo, dtype=np.float32)
        self.scale = self.small.shape[1] / w
        self.stops = pl.scene_stops(analysis, profile)
        self.shift = (pl.white_balance_shift(settings.temperature + profile.temperature, settings.tint)
                      if analysis.channels == 3 else np.zeros(3, dtype=np.float32)).astype(np.float64)
        self.saturation = 1.0 if profile.is_bw else profile.saturation * float(settings.saturation)
        self._gamma = 1.6 * profile.paper_contrast
        self._blend: tuple[float, np.ndarray | None] = (np.nan, None)

        # Every pixel's place on the film's log-exposure scale (0 shadow, 1
        # white), colour corrections included: what exposure, white and black act on.
        x = pl.normalised_log_exposure(self.small, analysis, settings.copy(white=0.0, black=0.0), profile,
                                       camera_matrix, rgb_sequential).reshape(-1, analysis.channels)
        # Bare light (thinner than the film base: sprocket holes, the screen
        # around the strip) and unlit glass are not picture: they print pure
        # black or white whatever the sliders.
        dens = pl.densities(pl.to_working(self.small, profile), analysis.base).reshape(-1, analysis.channels)
        max_d = pl.MAX_FILM_DENSITY + (0.0 if profile.is_negative else 1.5)
        finite = np.isfinite(_columns(x, np.add))
        valid = finite & (_columns(dens, np.minimum) > -0.05) & (_columns(dens, np.maximum) < max_d)
        if np.count_nonzero(valid) < min(64, len(x)):
            valid = finite
        self.valid = valid
        self.x = x[valid].astype(np.float64)

        # Stops of scene luminance from the median down to the darkest and up
        # to the brightest: how much room each side has before Auto moves it.
        lum = np.exp2((self.x - 1.0) * self.stops + self.shift) @ pl.REC709_Y
        darkest, median, brightest = np.maximum(np.percentile(lum, (100 * TAIL, 50, 100 * (1 - TAIL))), 1e-30)
        self.below, self.above = float(np.log2(median / darkest)), float(np.log2(brightest / median))
        self.pick(NEUTRAL)

    def pick(self, values: dict[str, float]) -> None:
        """Choose the quantile pixels as they print with ``values``.

        Picked again as the fit moves: saturation can take a channel of a
        bright colour past the others, so the order of the brightest
        channel depends on the settings.
        """
        ref = self._print(self.x, values, clip=False)
        idx = np.concatenate([_pick(ref @ pl.REC709_Y, _LUM_LEVELS, "bottom"),
                              _pick(_brightest(ref), _TOP_LEVELS, "top")])
        self.reps_x = self.x[idx]

    def exposure_for(self, mid: float, values: dict[str, float]) -> float:
        """Exposure that puts the median at ``mid`` with the other ``values``: where the fit
        starts, since a median lost in clipping gives it no slope to follow."""
        lo, hi = _BOUNDS["exposure"]
        for _ in range(16):
            ev = 0.5 * (lo + hi)
            if self.model(dict(values, exposure=ev))[0] > mid:
                hi = ev
            else:
                lo = ev
        return 0.5 * (lo + hi)

    def natural(self) -> np.ndarray:
        """Statistics of the plain print (no stretch, contrast 1) exposed so its whites
        just reach WHITE_LEVEL: how the scene looks before Auto takes a view on it."""
        lo, hi = _BOUNDS["exposure"]
        for _ in range(20):
            ev = 0.5 * (lo + hi)
            if self.model(dict(NEUTRAL, exposure=ev))[1] > WHITE_LEVEL:
                hi = ev
            else:
                lo = ev
        return self.model(dict(NEUTRAL, exposure=0.5 * (lo + hi)))

    # ------------------------------------------------------------ the model

    def _paper_table(self, contrast: float) -> np.ndarray:
        """Display (linear) of a grey on the paper over _LUT_T: precomputed papers
        blended, since solving a new one costs a millisecond or two."""
        if self._blend[0] == contrast:
            return self._blend[1]
        if CONTRAST_RANGE[0] <= contrast <= CONTRAST_RANGE[1]:
            i = int(np.clip(np.searchsorted(_LUT_CONTRASTS, contrast) - 1, 0, len(_LUT_CONTRASTS) - 2))
            c0, c1 = float(_LUT_CONTRASTS[i]), float(_LUT_CONTRASTS[i + 1])
            a = np.log(contrast / c0) / np.log(c1 / c0)
            table = (1 - a) * _paper(self._gamma * c0) + a * _paper(self._gamma * c1)
        else:
            table = _paper(self._gamma * contrast)
        self._blend = (contrast, table)
        return table

    def _print(self, x: np.ndarray, values: dict[str, float], clip: bool = True) -> np.ndarray:
        """Linear display of pixels at log exposure ``x``: the rest of pipeline.render."""
        b, w = values["black"], values["white"]
        t = ((x - b) / max(1.0 - b - w, 0.05) - 1.0) * self.stops + values["exposure"] + self.shift
        lin = _lookup(self._paper_table(values["contrast"]), (t - _LUT_T[0]) * _LUT_SCALE)
        if abs(self.saturation - 1.0) > 1e-3:
            y = (lin @ pl.REC709_Y)[:, None]
            lin = y + self.saturation * (lin - y)
            if clip:
                lin = np.clip(lin, 0.0, 1.0)
        return lin

    def model(self, values: dict[str, float]) -> np.ndarray:
        """The statistics the quantile pixels would print with (local adjustments as tone curves)."""
        lin = self._print(self.reps_x, values)
        if values["shadows"] or values["highlights"]:
            lift, pull = _hs_curves()
            light = np.cbrt(np.maximum(lin @ pl.REC709_Y, 1e-9))
            pos = light * (len(_HS_L) - 1)
            new = light + values["shadows"] * _lookup(lift, pos) - values["highlights"] * _lookup(pull, pos)
            lin = np.clip(lin * ((np.maximum(new, 0.0) / light) ** 3)[:, None], 0.0, 1.0)
        return self._reps_statistics(lin)

    def real(self, settings: pl.DevelopSettings) -> np.ndarray:
        """The whole reduced picture through the real chain."""
        display = dv.finish(dv.invert(self.small, self.analysis, settings, self.profile, self.camera_matrix,
                                      self.rgb_sequential), settings, self.scale)
        lin = _DECODE[np.rint(display.reshape(-1, 3)[self.valid] * 4095).astype(np.intp)]
        levels = np.linspace(0.0, 1.0, len(lin))
        return _statistics(levels, np.sort(lin @ pl.REC709_Y), levels, np.sort(_brightest(lin)))

    def _reps_statistics(self, lin: np.ndarray) -> np.ndarray:
        n = len(_LUM_LEVELS)
        return _statistics(_LUM_LEVELS, np.sort(lin[:n] @ pl.REC709_Y), _TOP_LEVELS, np.sort(_brightest(lin[n:])))


# ---------------------------------------------------------------- fitting

class _Targets:
    """Where each statistic should land, and the weighted misses the requested fields answer for.

    Relative to the scene's plain print: the median moves part of the way
    to middle grey (a night stays dark, snow stays light), the blacks part
    of the way to BLACK_LEVEL, and the quartiles may spread a little more
    than the scene's own before Shadows and Highlights hold them back;
    Shadows also opens a lower quartile sunk deep under middle grey.
    """

    def __init__(self, fields: tuple[str, ...], natural: np.ndarray, below: float, above: float) -> None:
        self.fields = fields
        self.full = len(fields) > 1
        mid, _, black, low, high = natural
        # The ends weigh in with the scene's range: a flat picture only gets its exposure.
        self.ends = float(np.clip((below + above) / FLAT_SPREAD, 0.0, 1.0))
        # Brightening a median that sits on the darkest tones (a night) or
        # darkening one on the brightest (snow) would only grey them.
        room = np.clip((below if mid < MID_GREY else above) / ROOM, 0.0, 1.0)
        self.mid = mid + (MID_GREY - mid) * (1.0 - self.ends * (1.0 - MID_PULL * room))
        ratio = _decode(self.mid) / max(_decode(mid), 1e-9)
        # A lower quartile far under middle grey (or a darker median) is a crowd of shadows to open up.
        deep = _decode(min(self.mid, MID_GREY)) * 2.0 ** -LOW_DEPTH
        self.low = _encode(max(_decode(low) * ratio * 2.0 ** -QUARTILE_SLACK, deep))
        self.high = _encode(min(_decode(high) * ratio * 2.0 ** QUARTILE_SLACK, 1.0))
        # Likewise the darkest tones only go to black when they lie well under the median.
        pull = BLACK_PULL * np.clip(below / ROOM, 0.0, 1.0)
        self.black = black + pull * (np.log(BLACK_LEVEL + _BLACK_EPS) - black)

    def residuals(self, stats: np.ndarray, values: dict[str, float]) -> np.ndarray:
        """All the targets for the full Auto; a single slider's Auto (Shift + double
        click) answers for its own target only, contrast for the whole tonal range."""
        mid, white, black, low, high = stats
        f, ends = self.fields, self.ends
        full = self.full or "contrast" in f
        out = []
        if full or "exposure" in f:
            out.append((mid - self.mid) / _SIGMA["mid"])
        if full or "white" in f:
            out.append(ends * (white - WHITE_LEVEL) / _SIGMA["white_over" if white > WHITE_LEVEL else "white_under"])
        if full or "black" in f:
            out.append(ends * (black - self.black) / _SIGMA["black_under" if black < self.black else "black_over"])
        if self.full or "shadows" in f:
            out.append(ends * min(low - self.low, 0.0) / _SIGMA["low"])
        if self.full or "highlights" in f:
            out.append(ends * max(high - self.high, 0.0) / _SIGMA["high"])
        for name in f:
            value = np.log(values[name]) if name == "contrast" else values[name]
            out.append(value / _SIGMA[name])
        return np.asarray(out, dtype=np.float64)


def _with(values: dict[str, float], params: tuple[str, ...], p: np.ndarray) -> dict[str, float]:
    return {**values, **{name: float(v) for name, v in zip(params, p)}}


def _fit(evaluate: Callable[[dict[str, float]], np.ndarray], values: dict[str, float], targets: _Targets,
         iterations: int = 12) -> dict[str, float]:
    """Levenberg-Marquardt over the free parameters, within their ranges.

    It stops once an iteration gains less than 3 %: the trade-offs between
    targets leave long, nearly flat valleys where further steps change the
    picture by less than the sliders' resolution.
    """
    params = targets.fields
    lo = np.array([_BOUNDS[n][0] for n in params])
    hi = np.array([_BOUNDS[n][1] for n in params])
    p = np.clip([values[n] for n in params], lo, hi)

    def residuals(q: np.ndarray) -> tuple[np.ndarray, float]:
        v = _with(values, params, q)
        r = targets.residuals(evaluate(v), v)
        return r, float(r @ r)

    r, cost = residuals(p)
    damping = 1e-3
    for _ in range(iterations):
        jac = np.empty((len(r), len(p)))
        for i, name in enumerate(params):
            h = _STEPS[name] if p[i] + _STEPS[name] <= hi[i] else -_STEPS[name]
            q = p.copy()
            q[i] += h
            jac[:, i] = (residuals(q)[0] - r) / h
        a, g = jac.T @ jac, jac.T @ r
        while True:
            step = np.linalg.solve(a + damping * np.diag(np.diag(a) + 1e-9), -g)
            q = np.clip(p + step, lo, hi)
            r_new, cost_new = residuals(q)
            if cost_new < cost:
                break
            damping *= 4.0
            if damping > 1e6:
                return _with(values, params, p)
        gain = (cost - cost_new) / max(cost, 1e-12)
        p, r, cost = q, r_new, cost_new
        damping = max(damping / 3.0, 1e-6)
        if gain < 0.03:
            break
    return _with(values, params, p)


def auto_tone(geo: np.ndarray, analysis: pl.Analysis, settings: pl.DevelopSettings, profile: FilmProfile,
              camera_matrix: np.ndarray | None, rgb_sequential: bool,
              fields: Iterable[str] | None = None) -> dict[str, float]:
    """Absolute model values for ``fields`` (all of TONE_FIELDS by default).

    ``geo`` is the oriented, transformed and cropped linear camera image.
    Every requested field starts from neutral, so the answer does not depend
    on its current value and Auto twice gives the same picture; the fields
    not requested keep theirs. Measured on tone alone: the Saturation
    slider (whose stronger colours can push one channel past white without
    any tone being blown), effects and detail (vignette, grain, sharpening,
    noise reduction) are left out, and a flat (for editing) output is
    measured as the print it would make.
    """
    fields = TONE_FIELDS if fields is None else tuple(dict.fromkeys(fields))
    unknown = set(fields) - set(TONE_FIELDS)
    if unknown:
        raise ValueError(f"not tone fields: {sorted(unknown)}")
    values = {f: NEUTRAL[f] if f in fields else float(getattr(settings, f)) for f in TONE_FIELDS}
    if geo.size == 0 or not fields or analysis.extra.get("no_light"):
        return {f: values[f] for f in fields}
    measured = settings.copy(output="print", saturation=1.0,
                             disabled=tuple(sorted(set(settings.disabled) | {"detail", "effects"})))
    picture = _Picture(geo, analysis, measured, profile, camera_matrix, rgb_sequential)
    targets = _Targets(fields, picture.natural(), picture.below, picture.above)

    # Fit on the model, check on the real chain, where the local
    # adjustments act as they really do, and fit again with the model's
    # error added.
    if "exposure" in fields:
        values["exposure"] = picture.exposure_for(targets.mid, values)
    values = _rounded(_fit(picture.model, values, targets), fields)
    picture.pick(values)
    error = picture.real(measured.copy(**values)) - picture.model(values)
    values = _rounded(_fit(lambda v: picture.model(v) + error, values, targets), fields)
    return {f: values[f] for f in fields}


def _rounded(values: dict[str, float], fields: tuple[str, ...]) -> dict[str, float]:
    """``fields`` to about the sliders' resolution: the history reads "Exposición +0,35", and
    contrast on a 0.01 grid often reuses the paper table pipeline built for the check."""
    return {f: round(v, 2 if f in ("exposure", "contrast", "highlights", "shadows") else 3) if f in fields else v
            for f, v in values.items()}
