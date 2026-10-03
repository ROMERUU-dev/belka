"""Negative to positive inversion (NumPy and OpenCV, no Qt).

Per channel c, the model is::

    T_c = I_c / F_c                    flat-field corrected camera signal (linear)
    D_c = -log10(T_c / B_c)            density above the film base B (D-min)
    x_c = (D_c - lo_c) / (hi_c - lo_c) normalised log exposure: 0 shadow, 1 white
    x  <- M x                          density-domain unmixing (dye crosstalk)
    L_c = 2 ** ((x_c - 1) * S + ev)    scene-linear light, white = 1
    y  = paper(L)                      print curve, then sRGB encoding

``lo``/``hi`` come from blending two estimates: a physical one (the film base
is black and the layers keep the profile's gamma ratios, anchored on the green
channel) and per-channel auto levels from the image. The blend is
``auto_balance``. Because the film records log exposure linearly, every colour
correction (white balance, neutral picker) is an additive shift of ``x``.

Speed: up to the print, the chain is an affine map of ln(T) followed by one
curve shared by every channel, so ``render`` does a log, a 3x3 affine and a
table read per pixel, in horizontal strips on a thread pool (NumPy and
OpenCV release the GIL). ``_render_reference`` keeps the model written out
step by step; the tests hold ``render`` to it.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace

import cv2
import numpy as np

from belka.core.film import FilmProfile

EPS = 1e-6
LOG10_2 = float(np.log10(2.0))
MID_GRAY = 0.18
REC709_Y = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Camera -> sRGB matrix typical of a modern DSLR, used when the input file
# carries no colour matrix (TIFF/JPEG scans, the simulated camera). Rows sum
# to one so neutrals stay neutral.
GENERIC_CAMERA_MATRIX = np.array(
    [
        [1.85, -0.75, -0.10],
        [-0.20, 1.55, -0.35],
        [0.05, -0.55, 1.50],
    ],
    dtype=np.float64,
)

# A frame captured with sequential red/green/blue light has far less channel
# crosstalk, so it needs a weaker unmixing.
RGB_SEQUENTIAL_SEPARATION = 0.4


@dataclass
class DevelopSettings:
    profile_id: str = "generic-c41"
    base: tuple[float, float, float] | None = None
    auto_balance: float = 0.6
    exposure: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    contrast: float = 1.0
    black: float = 0.0
    white: float = 0.0
    saturation: float = 1.0
    separation: float | None = None
    neutral: tuple[float, float, float] = (0.0, 0.0, 0.0)
    output: str = "print"
    crop: tuple[float, float, float, float] | None = None
    rotation: int = 0
    flip_h: bool = False
    flip_v: bool = False
    auto_crop: bool = True
    analysis_margin: float = 0.08

    # ---- Lightroom-style adjustments, applied after the inversion on the
    # display-referred (sRGB-encoded) image by belka.core.adjust. Every
    # default is neutral, so rolls from 0.1 render exactly as before.
    # Básico · tono y presencia (-1..1 unless noted)
    highlights: float = 0.0
    shadows: float = 0.0
    texture: float = 0.0
    clarity: float = 0.0
    vibrance: float = 0.0
    # Curva de tonos: parametric regions (-1..1) with their three split points,
    # then point curves in display space, ((x, y), ...) with x, y in [0, 1].
    curve_shadows: float = 0.0
    curve_darks: float = 0.0
    curve_lights: float = 0.0
    curve_highlights: float = 0.0
    curve_splits: tuple[float, float, float] = (0.25, 0.5, 0.75)
    curve_rgb: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
    curve_red: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
    curve_green: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
    curve_blue: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
    # HSL: eight bands (rojo, naranja, amarillo, verde, aguamarina, azul,
    # púrpura, magenta), each -1..1.
    hsl_hue: tuple[float, ...] = (0.0,) * 8
    hsl_sat: tuple[float, ...] = (0.0,) * 8
    hsl_lum: tuple[float, ...] = (0.0,) * 8
    # Gradación de color: (hue in degrees 0..360, saturation 0..1, luminance -1..1)
    grade_shadows: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grade_midtones: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grade_highlights: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grade_global: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grade_blending: float = 0.5  # 0..1
    grade_balance: float = 0.0  # -1..1
    # Detalle (radius in pixels of the full-resolution image)
    sharpen_amount: float = 0.0  # 0..1.5
    sharpen_radius: float = 1.0  # 0.5..3
    sharpen_detail: float = 0.25  # 0..1
    sharpen_masking: float = 0.0  # 0..1
    nr_luma: float = 0.0  # 0..1
    nr_color: float = 0.0  # 0..1
    # Efectos
    vignette_amount: float = 0.0  # -1..1, after the crop
    vignette_midpoint: float = 0.5
    vignette_roundness: float = 0.0  # -1..1
    vignette_feather: float = 0.5
    grain_amount: float = 0.0  # 0..1
    grain_size: float = 0.25  # 0..1
    grain_roughness: float = 0.5  # 0..1
    # ---- Geometry (belka.core.transform), applied to the oriented image
    # before the crop; ``crop`` is in the coordinates of the transformed image.
    angle: float = 0.0  # straighten, degrees, -45..45 (positive = clockwise)
    persp_vertical: float = 0.0  # -1..1 keystone
    persp_horizontal: float = 0.0  # -1..1
    persp_rotate: float = 0.0  # degrees, -10..10
    persp_aspect: float = 0.0  # -1..1
    persp_scale: float = 1.0  # 0.5..1.5
    persp_x: float = 0.0  # -1..1, fraction of the width
    persp_y: float = 0.0
    lens_distortion: float = 0.0  # -1..1 (barrel + / pincushion -)
    lens_vignette: float = 0.0  # -1..1, brightens (+) the corners before the crop
    constrain_crop: bool = True
    crop_aspect: str = "original"  # "free", "original", "1:1", "4:5", "3:2", "16:9", ...
    # The Upright mode that produced angle/persp_* ("", "auto", "level",
    # "vertical", "full" or "guided"), so its button shows as active.
    upright_mode: str = ""
    # Guided Upright: up to 4 lines (x0, y0, x1, y1) in normalised coordinates
    # of the oriented image before lens correction and warp, so they stay on
    # the same picture features whatever the transform.
    upright_guides: tuple[tuple[float, float, float, float], ...] = ()
    # Sections switched off with their panel toggle ("curve", "hsl", "grading",
    # "detail", "effects", "transform", "lens").
    disabled: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict | None) -> "DevelopSettings":
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        for key in ("base", "neutral", "crop", "curve_splits", "hsl_hue", "hsl_sat", "hsl_lum",
                    "grade_shadows", "grade_midtones", "grade_highlights", "grade_global"):
            if clean.get(key) is not None:
                clean[key] = tuple(float(v) for v in clean[key])
        for key in ("curve_rgb", "curve_red", "curve_green", "curve_blue"):
            if clean.get(key) is not None:
                clean[key] = tuple((float(x), float(y)) for x, y in clean[key])
        if clean.get("disabled") is not None:
            clean["disabled"] = tuple(str(v) for v in clean["disabled"])
        if clean.get("upright_guides") is not None:
            clean["upright_guides"] = tuple(tuple(float(v) for v in g) for g in clean["upright_guides"])
        return cls(**clean)

    def section_on(self, key: str) -> bool:
        return key not in self.disabled

    def geometry_key(self) -> tuple:
        """Everything that changes where pixels land (cache key for the warp)."""
        return (self.rotation, self.flip_h, self.flip_v, self.angle, self.persp_vertical, self.persp_horizontal,
                self.persp_rotate, self.persp_aspect, self.persp_scale, self.persp_x, self.persp_y,
                self.lens_distortion, self.lens_vignette, self.section_on("transform"), self.section_on("lens"))

    def copy(self, **changes) -> "DevelopSettings":
        return replace(self, **changes)

    # Settings that change what is measured, as opposed to how it is rendered.
    def analysis_key(self) -> tuple:
        return (self.profile_id, self.base, self.crop, self.auto_crop, self.rotation, self.flip_h, self.flip_v,
                self.analysis_margin)


@dataclass
class Analysis:
    base: np.ndarray
    base_is_auto: bool
    lo: np.ndarray
    hi: np.ndarray
    channels: int = 3
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- geometry

def orient(img: np.ndarray, rotation: int = 0, flip_h: bool = False, flip_v: bool = False) -> np.ndarray:
    out = img
    if flip_h:
        out = out[:, ::-1]
    if flip_v:
        out = out[::-1]
    k = (rotation // 90) % 4
    if k:
        # np.rot90 turns counter-clockwise; the UI rotates clockwise.
        out = np.rot90(out, k=-k)
    return out


def crop_slices(shape: tuple[int, ...], crop: tuple[float, float, float, float] | None, inset: float = 0.0):
    h, w = shape[:2]
    if crop is None:
        x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
    else:
        x0, y0, x1, y1 = crop
        x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
        y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    dx, dy = (x1 - x0) * inset, (y1 - y0) * inset
    xs = slice(int(round((x0 + dx) * w)), max(int(round((x1 - dx) * w)), int(round((x0 + dx) * w)) + 1))
    ys = slice(int(round((y0 + dy) * h)), max(int(round((y1 - dy) * h)), int(round((y0 + dy) * h)) + 1))
    return ys, xs


def effective_crop(settings: DevelopSettings, analysis: "Analysis | None" = None):
    """The user's crop, else the automatically detected frame, else None."""
    if settings.crop is not None:
        return settings.crop
    if settings.auto_crop and analysis is not None:
        return analysis.extra.get("frame")
    return None


def apply_geometry(img: np.ndarray, settings: DevelopSettings, analysis: "Analysis | None" = None) -> np.ndarray:
    out = orient(img, settings.rotation, settings.flip_h, settings.flip_v)
    crop = effective_crop(settings, analysis)
    if crop is not None:
        ys, xs = crop_slices(out.shape, crop)
        out = out[ys, xs]
    return out


def downsample(img: np.ndarray, max_side: int) -> np.ndarray:
    """Box-filter downsample by an integer factor so the long side fits."""
    h, w = img.shape[:2]
    factor = int(np.ceil(max(h, w) / max_side)) if max(h, w) > max_side else 1
    if factor <= 1:
        return img
    h2, w2 = h // factor, w // factor
    trimmed = img[: h2 * factor, : w2 * factor]
    shape = (h2, factor, w2, factor) + img.shape[2:]
    return trimmed.reshape(shape).mean(axis=(1, 3), dtype=np.float32)


# ---------------------------------------------------------------- measurement

def to_working(img: np.ndarray, profile: FilmProfile) -> np.ndarray:
    """B&W stocks are measured on one channel mixed from the camera RGB."""
    if profile.is_bw and img.ndim == 3:
        mix = np.asarray(profile.bw_mix, dtype=np.float32)
        mix = mix / mix.sum()
        return (img @ mix)[..., None]
    return img


# A camera frame of a backlit strip holds more than film: bare light around
# the strip and through the sprocket holes, and a dark surround outside the
# lit area. None of it is film, and both fool a naive "brightest = base,
# darkest = highlight" analysis, so they are masked out first.
NOISE_FLOOR = 0.0015  # mean linear signal below this is "no light at all"
MAX_FILM_DENSITY = 2.4  # above the base; anything denser is unlit, not film
ORANGE_MIN = 0.15  # D_blue - D_red that marks the colour-negative orange mask
LIGHT_CHROMA_TOL = 0.08
LIGHT_GAP = 0.25  # bare light sits at least this much (mean D) above the thinnest film
ANALYSIS_SIDE = 640
EDGE_PX = 3  # pixels around bare light treated as blur, at ANALYSIS_SIDE


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a square, via shifted ORs (fine at analysis size)."""
    if radius <= 0 or not mask.any():
        return mask
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            ys = slice(max(dy, 0), h + min(dy, 0))
            yd = slice(max(-dy, 0), h + min(-dy, 0))
            xs = slice(max(dx, 0), w + min(dx, 0))
            xd = slice(max(-dx, 0), w + min(-dx, 0))
            out[yd, xd] |= mask[ys, xs]
    return out


def light_masks(img: np.ndarray, profile: FilmProfile | None = None) -> dict[str, np.ndarray]:
    """Classify an (H, W, C) linear image into lit/unlit and bare light vs film.

    Clipped pixels are always bare light. Unclipped bare light is only
    accepted when it really is a separate, brighter population: light that
    went through no film keeps the light's chroma and sits well above the
    thinnest film, while film carries the orange mask. Without that gap the
    brightest unclipped pixels are the film base itself (tight framing, a
    film carrier, clipped light or a tint-calibrated light), and calling them
    "light" would push the base estimate into the picture.
    """
    h, w, c = img.shape
    flat = img.reshape(-1, c)
    lum = flat.mean(axis=1)
    clipped = np.any(flat >= 0.98, axis=1)
    lit = lum > NOISE_FLOOR
    cand = lit & ~clipped
    light = clipped.copy()
    n = np.count_nonzero(cand)
    if n >= 64:
        peak = np.percentile(flat[cand], 99.7, axis=0)
        d = -np.log10(np.clip(flat / np.maximum(peak, EPS), EPS, None))
        dm = d.mean(axis=1)
        if c == 3 and (profile is None or profile.type == "color_negative"):
            orange = d[:, 2] - d[:, 0]
            film = cand & (orange > ORANGE_MIN)
            light_like = cand & (np.abs(orange) < LIGHT_CHROMA_TOL) & (dm < 0.6)
            if np.count_nonzero(film) > 0.01 * n and np.count_nonzero(light_like) > 0.002 * n:
                gap = np.percentile(dm[film], 2) - np.median(dm[light_like])
                if gap > LIGHT_GAP:
                    light |= light_like
        else:
            # B&W and slides: only brightness. Require a valley under the top
            # band, i.e. a bright population that the film does not continue.
            top = cand & (dm < 0.08)
            below = cand & (dm >= 0.08) & (dm < 0.25)
            rest = cand & (dm >= 0.25)
            if (np.count_nonzero(top) > 0.002 * n and np.count_nonzero(below) < 0.25 * np.count_nonzero(top)
                    and np.count_nonzero(rest) > 0.01 * n):
                light |= top
    light2d = _dilate(light.reshape(h, w), EDGE_PX)
    unlit2d = _dilate(~lit.reshape(h, w), EDGE_PX)
    film2d = ~light2d & ~unlit2d
    # The film that matters lies on the light: inside the box spanned by the
    # bare light around the strip and through its holes. Outside it there is
    # only glare from the panel, dim enough to pass for dense film.
    bare = light.reshape(h, w)
    if np.count_nonzero(bare) > 0.002 * h * w:
        ys, xs = np.nonzero(bare)
        y0, y1 = np.percentile(ys, [0.5, 99.5]).astype(int)
        x0, x1 = np.percentile(xs, [0.5, 99.5]).astype(int)
        if (y1 - y0) > 0.15 * h and (x1 - x0) > 0.15 * w:
            inside = np.zeros((h, w), dtype=bool)
            inside[y0:y1 + 1, x0:x1 + 1] = True
            film2d &= inside
    return {"light": light2d, "unlit": unlit2d, "film": film2d, "bare": bare,
            "lit_fraction": float(np.mean(lit)),
            "field": illumination_field(img, bare & ~clipped.reshape(h, w))}


def illumination_field(img: np.ndarray, samples: np.ndarray) -> np.ndarray | None:
    """Smooth light falloff fitted to the unclipped bare light, normalised to 1.

    Without a flat-field the screen and the lens leave the corners darker (and
    tinted by the viewing angle); bare light around the strip and through its
    holes samples that falloff, and a quadratic surface in log space through
    those samples is enough to make the film base read uniform for analysis.
    None when there is too little unclipped bare light to fit.
    """
    h, w, c = img.shape
    pts = samples & ~_dilate(~samples, 2)  # away from blurred edges
    ys, xs = np.nonzero(pts)
    if len(ys) < 300 or (ys.max() - ys.min()) < 0.3 * h:
        return None
    step = max(1, len(ys) // 4000)
    ys, xs = ys[::step], xs[::step]
    u, v = xs / w - 0.5, ys / h - 0.5
    basis = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)
    values = np.log(np.maximum(img[ys, xs], EPS))
    coef, *_ = np.linalg.lstsq(basis, values, rcond=None)
    gu, gv = np.meshgrid(np.arange(w) / w - 0.5, np.arange(h) / h - 0.5)
    grid = np.stack([np.ones_like(gu), gu, gv, gu * gu, gu * gv, gv * gv], axis=-1)
    field = np.exp(grid @ coef)
    field /= np.maximum(field.reshape(-1, c).max(axis=0), EPS)
    return np.clip(field, 0.2, 1.0).astype(np.float32)


def estimate_base(img: np.ndarray, profile: FilmProfile | None = None, masks: dict | None = None) -> np.ndarray:
    """Film base colour: the least dense film in view.

    For a negative that is the rebate between frames or the deepest shadows;
    for a slide it is the clearest highlight. Bare light (sprocket holes, the
    screen around the strip), its blurred edges and unlit pixels are excluded
    because they show the light or nothing, not the film.
    """
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim == 3 and max(img.shape[:2]) > ANALYSIS_SIDE + 64:
        # The masks' edge margins are sized for the analysis resolution.
        img = downsample(img, ANALYSIS_SIDE)
        masks = None
    if img.ndim == 3 and img.shape[0] > 1 and img.shape[1] > 1:
        masks = masks or light_masks(img, profile)
        pool = masks["film"].reshape(-1)
        work = img.reshape(-1, img.shape[-1]).astype(np.float32)
    else:
        work = img.reshape(-1, img.shape[-1]).astype(np.float32)
        pool = np.ones(work.shape[0], dtype=bool)
    lum = work.mean(axis=1)
    pool = pool & np.all(work < 0.98, axis=1) & (lum > EPS)
    if np.count_nonzero(pool) < 64:
        pool = np.all(work < 0.98, axis=1) & (lum > EPS)
        if np.count_nonzero(pool) < 8:
            pool = lum > EPS
            if not np.any(pool):
                return np.ones(img.shape[-1], dtype=np.float32)
    lo, hi = np.percentile(lum[pool], [98.5, 99.7])
    band = pool & (lum >= lo) & (lum <= hi)
    if np.count_nonzero(band) < 8:
        band = pool & (lum >= lo)
    base = np.median(work[band], axis=0)
    return np.maximum(base, EPS).astype(np.float32)


def densities(img: np.ndarray, base: np.ndarray) -> np.ndarray:
    ratio = img / np.maximum(base, EPS)
    return -np.log10(np.clip(ratio, EPS, None))


def _runs(above: np.ndarray, bridge: int = 0) -> list[tuple[int, int]]:
    """Runs of True, joining runs separated by gaps of at most ``bridge``."""
    runs: list[list[int]] = []
    start = None
    for i, on in enumerate(np.append(above, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if runs and start - runs[-1][1] <= bridge:
                runs[-1][1] = i
            else:
                runs.append([start, i])
            start = None
    return [(a, b) for a, b in runs]


def _longest_run(values: np.ndarray, threshold: float, bridge: int = 0) -> tuple[int, int] | None:
    runs = _runs(values > threshold, bridge)
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def _smooth(values: np.ndarray, radius: int = 2) -> np.ndarray:
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    return np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")


def _smooth2d(a: np.ndarray, radius: int) -> np.ndarray:
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis, dtype=np.float64)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        size = out.shape[axis]
        hi = np.take(c, np.arange(2 * radius + 1, 2 * radius + 1 + size), axis=axis)
        lo = np.take(c, np.arange(0, size), axis=axis)
        out = ((hi - lo) / (2 * radius + 1)).astype(np.float32)
    return out


def detect_frame(img: np.ndarray, base: np.ndarray, masks: dict) -> tuple[float, float, float, float] | None:
    """Find the exposed picture inside a strip: (x0, y0, x1, y1) normalised.

    Across the strip the picture lies between the two rows of sprocket holes
    (when they are in view) less the thin rebate; along it, frames are
    separated by gaps of bare base that run the whole height of the picture
    band. A column belongs to the picture when *any* part of it is denser than
    the base, so a night sky or a black background does not cut the frame.
    """
    h, w = img.shape[:2]
    film = masks["film"]
    if np.count_nonzero(film) < 0.05 * h * w:
        return None
    field = masks.get("field")
    if field is not None:
        # Uneven light makes a uniform base look denser away from the bright
        # spot; measure against the base as it would read in the bright spot.
        flat = img / field
        work = flat.reshape(-1, flat.shape[-1])[masks["film"].reshape(-1)]
        lum = work.mean(axis=1)
        top = lum >= np.percentile(lum, 98.5)
        local_base = np.median(work[top], axis=0) if np.count_nonzero(top) >= 8 else base
        dens = _smooth2d(densities(flat, local_base).max(axis=-1), 1)
    else:
        dens = _smooth2d(densities(img, base).max(axis=-1), 1)
    exposed = film & (dens > 0.06) & (dens < MAX_FILM_DENSITY)
    near_base = film & (dens <= 0.08)
    bare = masks.get("bare", np.zeros_like(film))
    ys, xs = np.nonzero(film)
    fy0, fy1 = (int(v) for v in np.percentile(ys, [1, 99]))
    fx0, fx1 = (int(v) for v in np.percentile(xs, [1, 99]))
    transposed = (fy1 - fy0) > (fx1 - fx0)  # strip running vertically
    if transposed:
        film, exposed, near_base, bare = film.T, exposed.T, near_base.T, bare.T
        fy0, fy1, fx0, fx1 = fx0, fx1, fy0, fy1
    H, W = film.shape

    # Across the strip: between the sprocket-hole bands when both are visible.
    band = None
    row_light = bare[:, fx0:fx1 + 1].mean(axis=1)
    hole_rows = np.nonzero(row_light > 0.08)[0]
    hole_rows = hole_rows[(hole_rows >= fy0 - 3) & (hole_rows <= fy1 + 3)]
    mid = (fy0 + fy1) / 2
    upper, lower = hole_rows[hole_rows < mid], hole_rows[hole_rows > mid]
    if len(upper) and len(lower):
        a, b = int(upper.max()) + 1, int(lower.min()) - 1
        if b - a > 0.2 * H:
            rebate = int(round((b - a) * 0.03))  # the thin rebate (a slight tilt eats the rest)
            band = (a + rebate, b - rebate)
    if band is None:
        film_rows = film.mean(axis=1)
        rows = np.nonzero(film_rows > 0.5 * film_rows.max())[0]
        if len(rows) < 0.1 * H:
            return None
        band = (int(rows.min()), int(rows.max()) + 1)

    # Along the strip: a gap between frames is base-level over most of the
    # picture band; a picture column almost never is (a night sky covers part
    # of it). Real gaps read 0.99+; light smoothing only: they can be 4 px wide.
    rows = slice(band[0], band[1])
    on_film = film[rows].mean(axis=0)
    gapness = _smooth(near_base[rows].mean(axis=0), 1)
    picture = (gapness < 0.85) & (on_film > 0.5) & (_smooth(exposed[rows].mean(axis=0), 1) > 0.05)
    run_x = _longest_run(picture.astype(float), 0.5, bridge=1)
    if run_x is None:
        return None
    # Refine across the strip from the picture itself when no holes framed it.
    if not (len(upper) and len(lower)):
        rows_profile = _smooth(exposed[:, run_x[0]:run_x[1]].mean(axis=1))
        run_y = _longest_run(rows_profile, max(0.05, 0.15 * rows_profile.max()), bridge=int(0.03 * H))
        if run_y is None:
            return None
        band = (max(band[0], run_y[0]), min(band[1], run_y[1]))
    x0, x1 = run_x[0] / W, run_x[1] / W
    y0, y1 = band[0] / H, band[1] / H
    if transposed:
        x0, y0, x1, y1 = y0, x0, y1, x1
    fw, fh = x1 - x0, y1 - y0
    if fw <= 0 or fh <= 0 or fw * fh < 0.04 or fw * fh > 0.97:
        return None
    aspect = max(fw * w, fh * h) / max(min(fw * w, fh * h), 1)
    if aspect > 3.2:
        return None
    # A little inside the detected edge, which is blurred by the lens.
    ix, iy = fw * 0.01, fh * 0.01
    return (round(x0 + ix, 4), round(y0 + iy, 4), round(x1 - ix, 4), round(y1 - iy, 4))


def analyze(img: np.ndarray, settings: DevelopSettings, profile: FilmProfile) -> Analysis:
    oriented = orient(img, settings.rotation, settings.flip_h, settings.flip_v)
    full = to_working(downsample(oriented, ANALYSIS_SIDE), profile)
    masks = light_masks(full, profile)
    if settings.base is not None:
        base = np.asarray(settings.base, dtype=np.float32)
        if profile.is_bw:
            mix = np.asarray(profile.bw_mix, dtype=np.float32)
            base = np.array([float(base @ (mix / mix.sum()))], dtype=np.float32)
        base_is_auto = False
    else:
        # Measured on the whole view: the rebate lies outside any crop.
        base = estimate_base(full, profile, masks)
        base_is_auto = True

    frame = None
    if settings.crop is not None:
        box, inset = settings.crop, 0.02
    else:
        if settings.auto_crop:
            frame = detect_frame(full, base, masks)
        if frame is not None:
            box, inset = frame, 0.02
        else:
            m = min(max(settings.analysis_margin, 0.0), 0.4)
            box, inset = (m, m, 1 - m, 1 - m), 0.0
    ys, xs = crop_slices(full.shape, box, inset=inset)
    region = full[ys, xs]
    film = masks["film"][ys, xs].reshape(-1)

    dens = densities(region, base).reshape(-1, region.shape[-1])
    max_d = MAX_FILM_DENSITY + (1.5 if not profile.is_negative else 0.0)
    valid = film & np.all(dens > -0.05, axis=1) & np.all(dens < max_d, axis=1) & np.all(np.isfinite(dens), axis=1)
    if np.count_nonzero(valid) < 64:
        valid = np.all(np.isfinite(dens), axis=1) & np.all(dens > -0.05, axis=1)
        if np.count_nonzero(valid) < 64:
            valid = np.all(np.isfinite(dens), axis=1)
    sample = dens[valid]
    lo = np.percentile(sample, 0.5, axis=0).astype(np.float32)
    hi = np.percentile(sample, 99.7, axis=0).astype(np.float32)
    hi = np.maximum(hi, lo + 0.05)
    # Clipping inside the picture means the base (thinner still) clipped too:
    # measured on the camera channels, so a B&W mix cannot hide it.
    camera_full = downsample(oriented, ANALYSIS_SIDE)
    clipped = 0.0
    if settings.crop is not None or frame is not None:
        clipped = float(np.mean(np.any(camera_full[ys, xs] >= 0.98, axis=-1)))
    # Film whose thinnest channel clips while the others do not: the orange
    # mask's red goes first. Bare light clips in every channel and is skipped,
    # with the blurred edge around it.
    any_c = np.any(camera_full >= 0.98, axis=-1)
    all_c = np.all(camera_full >= 0.98, axis=-1)
    partial = any_c & ~_dilate(all_c, EDGE_PX) & ~masks["unlit"]
    lit_px = max(float(np.count_nonzero(~masks["unlit"])), 1.0)
    clipped = max(clipped, float(np.count_nonzero(partial)) / lit_px)
    extra = {"frame": frame, "base_level": float(np.max(base)), "clipped": clipped,
             "no_light": masks["lit_fraction"] < 0.005 or float(np.percentile(full, 99.9)) < 0.01}
    return Analysis(base=base, base_is_auto=base_is_auto, lo=lo, hi=hi, channels=region.shape[-1], extra=extra)


def exposure_advice(analysis: Analysis, target: float = 0.75) -> float:
    """Stops to add (+) or remove (−) so the film base sits near ``target``.

    Scanning a negative is "expose to the right" on the film base: the bare
    light may clip freely, but the densest highlights of the negative live
    well below the base, so every stop of underexposure costs one bit there.
    Returns 0 when the base is already within about a stop of the target.
    """
    if analysis.extra.get("no_light"):
        return 0.0  # nothing lit: the light is off, not the exposure short
    if analysis.extra.get("clipped", 0.0) > 0.003:
        return -1.0
    level = float(analysis.extra.get("base_level", 0.0))
    if level <= 0:
        return 0.0
    stops = float(np.log2(target / level))
    return stops if abs(stops) >= 1.0 else 0.0


def levels(analysis: Analysis, profile: FilmProfile, auto_balance: float) -> tuple[np.ndarray, np.ndarray]:
    """Blend physical (base + profile gamma) and per-channel automatic levels."""
    a = float(np.clip(auto_balance, 0.0, 1.0))
    if analysis.channels == 1:
        return analysis.lo.copy(), analysis.hi.copy()
    rel = np.asarray(profile.relative_gamma(), dtype=np.float32)
    # Physical estimate: a given scene exposure gives D_c / gamma_c equal on
    # every layer, so the green measurements fix the others.
    phys_lo = analysis.lo[1] * rel
    phys_hi = analysis.hi[1] * rel
    lo = (1 - a) * phys_lo + a * analysis.lo
    hi = (1 - a) * phys_hi + a * analysis.hi
    hi = np.maximum(hi, lo + 0.05)
    return lo, hi


def scene_stops(analysis: Analysis, profile: FilmProfile) -> float:
    """How many stops of scene light the measured density range represents."""
    idx = 1 if analysis.channels == 3 else 0
    span = float(analysis.hi[idx] - analysis.lo[idx])
    gamma = profile.gamma[1] if profile.gamma[1] > 0 else 0.6
    return float(np.clip(span / gamma / LOG10_2, 3.0, 16.0))


def separation_matrix(strength: float, camera_matrix: np.ndarray | None) -> np.ndarray:
    base = GENERIC_CAMERA_MATRIX if camera_matrix is None else np.asarray(camera_matrix, dtype=np.float64)
    m = np.eye(3) + float(strength) * (base - np.eye(3))
    # Re-normalise rows so a neutral stays exactly neutral.
    m = m / m.sum(axis=1, keepdims=True)
    return m.astype(np.float32)


# ---------------------------------------------------------------- rendering

def white_balance_shift(temperature: float, tint: float) -> np.ndarray:
    """Per-channel exposure shift in stops (R, G, B)."""
    t, m = float(temperature), float(tint)
    return np.array([0.5 * t + 0.25 * m, -0.5 * m, -0.5 * t + 0.25 * m], dtype=np.float32)


def srgb_encode(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055).astype(np.float32)


def srgb_decode(enc: np.ndarray) -> np.ndarray:
    enc = np.clip(enc, 0.0, 1.0)
    return np.where(enc <= 0.04045, enc / 12.92, np.power((enc + 0.055) / 1.055, 2.4)).astype(np.float32)


SCENE_FLOOR = 1e-5  # the paper prints any darker scene light like this


@dataclass(frozen=True)
class PaperCurve:
    """Print paper: density falls along a logistic curve as scene light grows.

    ``gamma`` is the paper slope (print density per log10 of scene light) at
    the curve's centre and ``dmax`` its maximum black. The centre is solved so
    that middle grey prints at 18 % and the analysed white at ``white``.
    """

    gamma: float = 1.6
    dmax: float = 2.1
    white: float = 0.95

    def _reflectance(self, z: np.ndarray, z0: float) -> np.ndarray:
        k = 4.0 * self.gamma / self.dmax
        dens = self.dmax / (1.0 + np.exp(k * (z - z0)))
        return np.power(10.0, -dens)

    def _display(self, z: np.ndarray, z0: float) -> np.ndarray:
        r_min = 10.0 ** (-self.dmax)
        r_white = self._reflectance(np.float64(0.0), z0)
        return (self._reflectance(z, z0) - r_min) / max(r_white - r_min, EPS) * self.white

    def solve(self) -> tuple[float, float]:
        """Curve centre and white level that print middle grey at 18 %.

        As a function of the centre, middle grey's display value is not
        monotonic: it falls to a minimum and rises again (the normalisation to
        the white point takes over). Only the falling branch is meaningful, so
        the minimum is located first and the bisection stays left of it. A
        very soft paper cannot reach 18 % with its white at ``white``; like a
        real soft paper it then prints whites a little grey instead, which
        keeps middle grey in place and the result continuous as contrast moves.
        """
        z_mid = np.float64(np.log10(MID_GRAY))
        f = lambda z0: float(self._display(z_mid, z0))
        a, b = -3.0, 3.0
        g = (np.sqrt(5.0) - 1.0) / 2.0
        c, d = b - g * (b - a), a + g * (b - a)
        for _ in range(80):  # golden-section search, f is unimodal
            if f(c) < f(d):
                b = d
            else:
                a = c
            c, d = b - g * (b - a), a + g * (b - a)
        z_min = 0.5 * (a + b)
        lowest = f(z_min)
        if lowest > MID_GRAY:
            return z_min, self.white * MID_GRAY / lowest
        lo, hi = -3.0, z_min
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(mid) > MID_GRAY:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), self.white

    def centre(self) -> float:
        return self.solve()[0]

    def apply(self, scene_linear: np.ndarray) -> np.ndarray:
        z0, white = self.solve()
        z = np.log10(np.maximum(scene_linear, SCENE_FLOOR))
        out = self._display(z, z0) * (white / self.white)
        return np.clip(out, 0.0, 1.0).astype(np.float32)


def normalised_log_exposure(
    img: np.ndarray,
    analysis: Analysis,
    settings: DevelopSettings,
    profile: FilmProfile,
    camera_matrix: np.ndarray | None = None,
    rgb_sequential: bool = False,
) -> np.ndarray:
    work = to_working(img, profile)
    dens = densities(work, analysis.base)
    lo, hi = levels(analysis, profile, settings.auto_balance)
    x = (dens - lo) / (hi - lo)
    if not profile.is_negative:
        x = 1.0 - x
    if analysis.channels == 3:
        strength = profile.separation if settings.separation is None else settings.separation
        if rgb_sequential:
            strength *= RGB_SEQUENTIAL_SEPARATION
        if strength > 0:
            m = separation_matrix(strength, camera_matrix)
            x = x @ m.T
        # After the separation: the matrix keeps neutrals neutral but would
        # bend an offset that is not neutral itself.
        x = x + np.asarray(settings.neutral, dtype=np.float32)
    b = float(settings.black)
    w = float(settings.white)
    span = max(1.0 - b - w, 0.05)
    return ((x - b) / span).astype(np.float32)


# ---- the fast path of render
# Per strip: ln of the camera signal, one affine map (cv2.transform) straight
# to positions in a table of the print curve, and a table read (cv2.remap,
# whose linear interpolation splits a step in 32: with these tables and their
# ranges that stays within about 3e-5 of the curve). The table depends only
# on the paper contrast, so dragging exposure, white balance, levels or
# geometry reuses it.
_TABLE_SIZE = 32000  # remap wants tables and maps narrower than 32767 entries
_TABLE_TOL = 1e-7  # tables stop where the print is this close to its end values
_REMAP_BLOCK = 1 << 16  # remap runs larger calls on OpenCV's own pool, against ours
_STRIP_PIXELS = 1 << 16  # per pool task; the strip's buffers stay in cache
_POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="belka-render")


def _frozen(values: np.ndarray) -> np.ndarray:
    """A 1-D table as a read-only remap source: two equal rows and the last
    entry repeated, so that no read's 2x2 neighbourhood leaves the table
    (remap's border handling makes a read 40 % slower)."""
    row = np.append(np.asarray(values, dtype=np.float32), np.float32(values[-1]))
    table = np.ascontiguousarray(np.stack([row, row]))
    table.flags.writeable = False
    return table


@functools.lru_cache(maxsize=16)
def _print_table(gamma: float, encode: bool) -> tuple[np.ndarray, float, float]:
    """``PaperCurve(gamma).apply`` (then sRGB encoding when ``encode``) as a
    table over the scene exponent in stops (white = 0): (table, first, step).

    The table only spans the range where the print still changes, so reads
    beyond its ends clamp to values within _TABLE_TOL of the true ones. When
    the print clips to white the table ends exactly there: a step across that
    kink would round the shoulder off by up to 1e-4.
    """
    paper = PaperCurve(gamma=gamma)
    z0, white = paper.solve()

    def curve(stops: np.ndarray) -> np.ndarray:
        z = np.maximum(stops * LOG10_2, np.log10(SCENE_FLOOR))
        with np.errstate(over="ignore"):  # far above white exp() overflows: no density, as it should
            lin = paper._display(z, z0) * (white / paper.white)
        return np.clip(lin, 0.0, 1.0)

    coarse = np.log2(SCENE_FLOOR) + np.arange(16 * 1024 + 1) / 16.0
    lin = curve(coarse)
    moving = np.nonzero((np.abs(lin - lin[0]) > _TABLE_TOL) & (np.abs(lin - lin[-1]) > _TABLE_TOL))[0]
    first, last = coarse[0], coarse[1]
    if moving.size:
        first, last = coarse[max(moving[0] - 1, 0)], coarse[min(moving[-1] + 1, coarse.size - 1)]
        if lin[-1] >= 1.0:
            j = int(np.argmax(lin >= 1.0))
            lo, last = coarse[j - 1], coarse[j]
            for _ in range(60):
                mid = 0.5 * (lo + last)
                lo, last = (mid, last) if curve(mid) < 1.0 else (lo, mid)
    step = (last - first) / (_TABLE_SIZE - 1)
    lin = curve(first + step * np.arange(_TABLE_SIZE))
    return _frozen(srgb_encode(lin) if encode else lin), float(first), float(step)


_ENCODE_TABLE = _frozen(srgb_encode(np.linspace(0.0, 1.0, _TABLE_SIZE)))


def _exponent_affine(
    analysis: Analysis,
    settings: DevelopSettings,
    profile: FilmProfile,
    camera_matrix: np.ndarray | None,
    rgb_sequential: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(floor, A, c) with the scene exponent in stops (white = 0, before the
    paper) ``e = A @ ln(max(work, floor)) + c`` per pixel.

    ``normalised_log_exposure`` and the exposure step of ``_render_reference``
    folded into one affine map, in float64.
    """
    base = np.maximum(analysis.base, EPS).astype(np.float64)
    lo, hi = (v.astype(np.float64) for v in levels(analysis, profile, settings.auto_balance))
    # x = (-log10(T / B) - lo) / (hi - lo), with ln T as the variable.
    scale = -1.0 / (np.log(10.0) * (hi - lo))
    a = np.diag(scale)
    c = (np.log10(base) - lo) / (hi - lo)
    if not profile.is_negative:
        a, c = -a, 1.0 - c
    if analysis.channels == 3:
        strength = profile.separation if settings.separation is None else settings.separation
        if rgb_sequential:
            strength *= RGB_SEQUENTIAL_SEPARATION
        if strength > 0:
            m = separation_matrix(strength, camera_matrix).astype(np.float64)
            a, c = m @ a, m @ c
        c = c + np.asarray(settings.neutral, dtype=np.float32)
        shift = white_balance_shift(settings.temperature + profile.temperature, settings.tint)
    else:
        shift = np.zeros(1, dtype=np.float32)
    b = float(settings.black)
    span = max(1.0 - b - float(settings.white), 0.05)
    stops = scene_stops(analysis, profile)
    a = a * (stops / span)
    c = ((c - b) / span - 1.0) * stops + float(settings.exposure) + shift.astype(np.float64)
    return EPS * base, a, c


def _fast_path(img: np.ndarray, analysis: Analysis, profile: FilmProfile) -> bool:
    """Whether ``render`` can use the tables: a 3-channel image whose analysis
    matches the profile (one channel for B&W, three otherwise)."""
    n = 1 if profile.is_bw else 3
    return (img.ndim == 3 and img.shape[-1] == 3 and img.shape[0] > 0 and 0 < img.shape[1] <= _TABLE_SIZE
            and analysis.channels == n and analysis.base.size == n and analysis.lo.size == n
            and analysis.hi.size == n)


def _strips(fn: Callable[[slice], None], h: int, w: int) -> None:
    """``fn(rows)`` over horizontal strips, on the pool when there are several."""
    step = max(1, _STRIP_PIXELS // max(w, 1))
    strips = [slice(y, min(y + step, h)) for y in range(0, h, step)]
    if len(strips) == 1:
        fn(strips[0])
        return
    for future in [_POOL.submit(fn, sl) for sl in strips]:
        future.result()


@functools.lru_cache(maxsize=8)
def _zeros(rows: int, w: int) -> np.ndarray:
    zeros = np.zeros((rows, w), dtype=np.float32)
    zeros.flags.writeable = False
    return zeros


def _read(table: np.ndarray, pos: np.ndarray, out: np.ndarray) -> None:
    """``out = table[pos]`` with linear interpolation, ``pos`` in entries and
    already clamped to the table; both contiguous, of any shape.

    Blocks of at most _REMAP_BLOCK values keep remap on the calling thread.
    """
    w = pos.shape[1]
    pos2, out2 = pos.reshape(-1, w), out.reshape(-1, w)
    step = max(1, _REMAP_BLOCK // w)
    for r in range(0, pos2.shape[0], step):
        block = pos2[r:r + step]
        cv2.remap(table, block, _zeros(*block.shape), cv2.INTER_LINEAR, dst=out2[r:r + step],
                  borderMode=cv2.BORDER_REPLICATE)


def render(
    img: np.ndarray,
    analysis: Analysis,
    settings: DevelopSettings,
    profile: FilmProfile,
    camera_matrix: np.ndarray | None = None,
    rgb_sequential: bool = False,
) -> np.ndarray:
    """Return an RGB float32 image in [0, 1].

    ``output == "print"`` gives sRGB-encoded values for display or export;
    ``output == "flat"`` gives scene-linear light (white at 0.8) in linear
    sRGB primaries, for further editing elsewhere.

    The result of ``_render_reference`` to within 1e-4 (a few 1e-6 at the
    usual settings), some twenty times faster.
    """
    if not _fast_path(img, analysis, profile):
        return _render_reference(img, analysis, settings, profile, camera_matrix, rgb_sequential)
    floor, a, c = _exponent_affine(analysis, settings, profile, camera_matrix, rgb_sequential)
    bw = analysis.channels == 1
    sat = profile.saturation * float(settings.saturation)
    saturate = settings.output != "flat" and not bw and abs(sat - 1.0) > 1e-3
    if settings.output == "flat":
        # 0.8 * 2 ** e as one exponential: exp(e ln 2 + ln 0.8).
        table = None
        a, c = a * np.log(2.0), c * np.log(2.0) + np.log(0.8)
    else:
        table, first, step = _print_table(1.6 * profile.paper_contrast * float(settings.contrast), not saturate)
        a, c = a / step, (c - first) / step
    affine = np.hstack([a, c[:, None]]).astype(np.float32)
    mix = np.asarray(profile.bw_mix, dtype=np.float32)
    mix = (mix / mix.sum()).reshape(1, 3)
    # lum + sat * (lin - lum) as a matrix, scaled to positions in the encoding table.
    sat_m = ((sat * np.eye(3) + (1.0 - sat) * REC709_Y.astype(np.float64)) * (_TABLE_SIZE - 1)).astype(np.float32)
    top = np.float32(_TABLE_SIZE - 1)
    h, w = img.shape[:2]
    # The per-channel floor along whole rows: a long inner loop, unlike a (3,) broadcast.
    floor_row = np.tile(floor.astype(np.float32), 1 if bw else w)
    out = np.empty((h, w, 3), dtype=np.float32)

    def rows(sl: slice) -> None:
        src = img[sl]
        n = src.shape[0]
        work = cv2.transform(src, mix) if bw else src.reshape(n, -1)
        pos = np.maximum(work, floor_row, dtype=np.float32).reshape((n, w) if bw else (n, w, 3))
        cv2.log(pos, dst=pos)
        cv2.transform(pos, affine, dst=pos)
        dst = np.empty_like(pos) if bw else out[sl]
        if table is None:
            cv2.exp(pos, dst=pos)
            np.minimum(pos, 1.0, out=dst)
        else:
            np.clip(pos, 0.0, top, out=pos)
            if saturate:
                lin = np.empty_like(pos)
                _read(table, pos, lin)
                cv2.transform(lin, sat_m, dst=pos)
                np.clip(pos, 0.0, top, out=pos)
                _read(_ENCODE_TABLE, pos, dst)
            else:
                _read(table, pos, dst)
        if bw:
            cv2.merge([dst, dst, dst], dst=out[sl])

    _strips(rows, h, w)
    return out


def _render_reference(
    img: np.ndarray,
    analysis: Analysis,
    settings: DevelopSettings,
    profile: FilmProfile,
    camera_matrix: np.ndarray | None = None,
    rgb_sequential: bool = False,
) -> np.ndarray:
    """``render`` written out step by step on whole arrays: the reference the
    tests hold the fast path to, and the fallback for images it does not take
    (empty, not RGB, extreme widths, an analysis that does not match the profile)."""
    x = normalised_log_exposure(img, analysis, settings, profile, camera_matrix, rgb_sequential)
    stops = scene_stops(analysis, profile)
    ev = float(settings.exposure)
    if analysis.channels == 3:
        shift = white_balance_shift(settings.temperature + profile.temperature, settings.tint)
    else:
        shift = np.zeros(1, dtype=np.float32)
    scene = np.exp2((x - 1.0) * stops + ev + shift).astype(np.float32)
    if scene.shape[-1] == 1:
        scene = np.repeat(scene, 3, axis=-1)

    if settings.output == "flat":
        return np.clip(scene * 0.8, 0.0, 1.0).astype(np.float32)

    paper = PaperCurve(gamma=1.6 * profile.paper_contrast * float(settings.contrast))
    lin = paper.apply(scene)
    sat = profile.saturation * float(settings.saturation)
    if not profile.is_bw and abs(sat - 1.0) > 1e-3:
        lum = (lin @ REC709_Y)[..., None]
        lin = np.clip(lum + sat * (lin - lum), 0.0, 1.0)
    return srgb_encode(lin)


def neutral_offsets(
    patch: np.ndarray,
    analysis: Analysis,
    settings: DevelopSettings,
    profile: FilmProfile,
    camera_matrix: np.ndarray | None = None,
    rgb_sequential: bool = False,
) -> tuple[float, float, float]:
    """Offsets that make ``patch`` (raw camera pixels) neutral grey."""
    if analysis.channels != 3:
        return (0.0, 0.0, 0.0)
    probe = settings.copy(neutral=(0.0, 0.0, 0.0))
    x = normalised_log_exposure(patch.reshape(-1, 1, patch.shape[-1]), analysis, probe, profile,
                                camera_matrix, rgb_sequential)
    # The render adds the white-balance shift (in stops) after this point, so
    # fold it in: the patch must come out grey with temperature/tint applied.
    shift = white_balance_shift(settings.temperature + profile.temperature, settings.tint) / scene_stops(analysis, profile)
    value = np.median(x.reshape(-1, 3), axis=0) + shift
    target = float(value.mean())
    # Offsets live before the black/white remap, which divides by its span.
    span = max(1.0 - float(settings.black) - float(settings.white), 0.05)
    return tuple(float((target - v) * span) for v in value)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_uint16(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
