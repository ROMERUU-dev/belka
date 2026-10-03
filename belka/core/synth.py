"""Synthetic negatives: a forward model of film + light + camera.

Used by the simulated camera (so the whole app can be tried without hardware)
and by the tests, which check that inverting a synthetic negative gives back
the scene it was made from.
"""

from __future__ import annotations

import numpy as np

from belka.core.film import FilmProfile
from belka.core.pipeline import srgb_decode

# Classic 24-patch ColorChecker, 8-bit sRGB.
COLORCHECKER_SRGB = np.array(
    [
        [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
        [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
        [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
        [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
    ],
    dtype=np.float32,
)

# How the camera's three channels see the three dye layers (rows: camera R,
# G, B; columns: cyan, magenta, yellow dye density). Broad camera filters
# overlap, which is what the separation matrix later undoes.
CAMERA_DYE_CROSSTALK = np.array(
    [
        [0.86, 0.11, 0.03],
        [0.10, 0.80, 0.10],
        [0.02, 0.14, 0.84],
    ],
    dtype=np.float32,
)

# Camera raw response to a screen's red, green and blue primaries (columns).
# The green channel is the most sensitive, as on a real Bayer sensor.
SCREEN_TO_CAMERA = np.array(
    [
        [0.55, 0.08, 0.02],
        [0.10, 0.95, 0.12],
        [0.02, 0.12, 0.62],
    ],
    dtype=np.float32,
)


def colorchecker_scene(width: int = 900, height: int = 600) -> np.ndarray:
    """Scene-linear sRGB image: ColorChecker, grey ramp and a colour sweep."""
    scene = np.zeros((height, width, 3), dtype=np.float32)
    patches = srgb_decode(COLORCHECKER_SRGB / 255.0)
    chart_h = int(height * 0.62)
    cell_w, cell_h = width / 6, chart_h / 4
    for i, rgb in enumerate(patches):
        r, c = divmod(i, 6)
        y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
        x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
        scene[y0:y1, x0:x1] = rgb
        # Thin dark gutters, like the real chart.
        scene[y0:y0 + 3, x0:x1] = 0.02
        scene[y0:y1, x0:x0 + 3] = 0.02
    ramp_y0, ramp_y1 = chart_h, int(height * 0.80)
    ramp = np.power(2.0, np.linspace(-8.0, 0.4, width, dtype=np.float32))
    scene[ramp_y0:ramp_y1] = ramp[None, :, None]
    hue = np.linspace(0.0, 1.0, width, dtype=np.float32)
    sweep = np.stack(
        [
            0.5 + 0.5 * np.cos(2 * np.pi * (hue - 0.0)),
            0.5 + 0.5 * np.cos(2 * np.pi * (hue - 1 / 3)),
            0.5 + 0.5 * np.cos(2 * np.pi * (hue - 2 / 3)),
        ],
        axis=-1,
    )
    shade = np.linspace(1.0, 0.15, height - ramp_y1, dtype=np.float32)[:, None, None]
    scene[ramp_y1:] = 0.04 + 0.7 * sweep[None] * shade
    return scene


def characteristic_curve(log_h: np.ndarray, gamma: float, toe: float = 0.18, dmax: float = 2.6) -> np.ndarray:
    """Density above base for a log10 exposure: soft toe, linear, soft shoulder."""
    straight = toe * np.logaddexp(0.0, log_h / toe)
    density = gamma * straight
    return dmax * np.tanh(density / dmax)


def expose_film(scene: np.ndarray, profile: FilmProfile, exposure_ev: float = 0.0) -> np.ndarray:
    """Total dye density (including the base) of each layer."""
    # Exposed so that middle grey sits about 1 log10 above the toe.
    log_h = np.log10(np.maximum(scene, 1e-6)) + exposure_ev * np.log10(2.0) + 1.75
    gamma = np.asarray(profile.gamma, dtype=np.float32)
    base = np.asarray(profile.base_density, dtype=np.float32)
    if profile.is_bw:
        lum = scene @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        log_h = np.log10(np.maximum(lum, 1e-6))[..., None] + exposure_ev * np.log10(2.0) + 1.75
        log_h = np.repeat(log_h, 3, axis=-1)
    if profile.is_negative:
        return base + characteristic_curve(log_h, 1.0) * gamma
    # Reversal film: more light, less dye.
    return base + np.clip(2.9 - characteristic_curve(log_h, 1.0) * gamma, 0.0, 3.2)


def add_rebate(density: np.ndarray, profile: FilmProfile, border: float = 0.07) -> tuple[np.ndarray, np.ndarray]:
    """Surround the frame with unexposed film and sprocket holes.

    Returns the enlarged density map and a mask of the holes (bare light).
    """
    h, w = density.shape[:2]
    by, bx = int(h * border * 1.8), int(w * border * 0.6)
    out = np.empty((h + 2 * by, w + 2 * bx, 3), dtype=np.float32)
    out[:] = np.asarray(profile.base_density, dtype=np.float32)
    out[by:by + h, bx:bx + w] = density
    holes = np.zeros(out.shape[:2], dtype=bool)
    hole_h, hole_w = int(by * 0.45), max(int(w * 0.03), 4)
    pitch = hole_w * 2.2
    for i in range(int(out.shape[1] / pitch) + 1):
        x0 = int(i * pitch + pitch * 0.3)
        holes[int(by * 0.2):int(by * 0.2) + hole_h, x0:x0 + hole_w] = True
        holes[-int(by * 0.2) - hole_h:-int(by * 0.2), x0:x0 + hole_w] = True
    return out, holes


def photograph(
    density: np.ndarray,
    light_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    holes: np.ndarray | None = None,
    exposure: float = 1.0,
    noise: float = 0.002,
    vignette: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Linear camera raw RGB of the film backlit by a screen."""
    rng = rng or np.random.default_rng(1)
    cam_density = density @ CAMERA_DYE_CROSSTALK.T
    transmittance = np.power(10.0, -cam_density)
    if holes is not None:
        transmittance[holes] = 1.0
    light = SCREEN_TO_CAMERA @ np.asarray(light_rgb, dtype=np.float32)
    h, w = density.shape[:2]
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    falloff = 1.0 - vignette * (xx**2 + yy**2) / 2.0
    signal = transmittance * light * exposure * falloff[..., None]
    signal = signal + rng.normal(0.0, noise, signal.shape) * np.sqrt(np.maximum(signal, 0.0) + 0.01)
    return np.clip(signal, 0.0, 1.0).astype(np.float32)


def synthetic_negative(
    profile: FilmProfile,
    scene: np.ndarray | None = None,
    light_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    exposure_ev: float = 0.0,
    with_rebate: bool = True,
    rng: np.random.Generator | None = None,
    camera_ev: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (camera raw, scene) for a frame of ``profile`` film.

    ``light_rgb`` is the linear emission of the screen's red, green and blue
    primaries; ``camera_ev`` over- or under-exposes the camera relative to a
    careful exposure that puts the film base just under clipping.
    """
    scene = colorchecker_scene() if scene is None else scene
    density = expose_film(scene, profile, exposure_ev)
    holes = None
    if with_rebate:
        density, holes = add_rebate(density, profile)
    base_t = np.power(10.0, -(np.asarray(profile.base_density, dtype=np.float32) @ CAMERA_DYE_CROSSTALK.T))
    exposure = 0.85 / float(np.max(base_t * (SCREEN_TO_CAMERA @ np.ones(3, dtype=np.float32))))
    raw = photograph(density, light_rgb, holes, exposure=exposure * 2.0**camera_ev, rng=rng)
    return raw, scene


def landscape_scene(width: int = 900, height: int = 600, seed: int = 0) -> np.ndarray:
    """A simple outdoor scene: sky, sun, hills, a red barn and some foliage."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    v = yy / height
    sky = np.stack([0.25 + 0.35 * v, 0.40 + 0.30 * v, 0.85 - 0.10 * v], axis=-1) * 0.9
    scene = sky.astype(np.float32)
    sun_x, sun_y = width * (0.2 + 0.6 * rng.random()), height * 0.18
    sun = np.exp(-(((xx - sun_x) ** 2 + (yy - sun_y) ** 2) / (2 * (height * 0.05) ** 2)))
    scene += (sun * 3.0)[..., None] * np.array([1.0, 0.95, 0.8], dtype=np.float32)
    horizon = height * (0.55 + 0.05 * np.sin(xx / width * 6.0 + rng.random() * 6))
    ground = yy > horizon
    texture = 0.75 + 0.25 * np.sin(xx * 0.09 + yy * 0.05) * np.sin(yy * 0.13)
    grass = np.stack([0.08 * texture, 0.20 * texture, 0.05 * texture], axis=-1)
    scene[ground] = grass[ground]
    bx0, by0 = int(width * 0.55), int(height * 0.45)
    scene[by0:by0 + int(height * 0.2), bx0:bx0 + int(width * 0.18)] = (0.30, 0.04, 0.03)
    scene[by0 + int(height * 0.08):by0 + int(height * 0.2), bx0 + int(width * 0.07):bx0 + int(width * 0.11)] = (0.02, 0.015, 0.01)
    for _ in range(14):
        cx, cy = rng.random() * width, height * (0.6 + 0.35 * rng.random())
        r = height * (0.03 + 0.05 * rng.random())
        blob = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        scene[blob] = (0.03 + 0.04 * rng.random(), 0.09 + 0.06 * rng.random(), 0.02)
    sweater = ((xx - width * 0.3) ** 2 / (width * 0.06) ** 2 + (yy - height * 0.78) ** 2 / (height * 0.12) ** 2) < 1
    scene[sweater] = (0.45, 0.28, 0.05)
    face = ((xx - width * 0.3) ** 2 + (yy - height * 0.60) ** 2) < (height * 0.06) ** 2
    scene[face] = (0.42, 0.26, 0.19)
    return np.clip(scene, 0.0, 8.0).astype(np.float32)


def backlit_scan(
    profile: FilmProfile,
    scene: np.ndarray | None = None,
    light_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    camera_ev: float = 0.0,
    film_ev: float = 0.0,
    rng: np.random.Generator | None = None,
    glow: float = 0.004,
) -> tuple[np.ndarray, dict]:
    """A camera view like a real copy-stand shot on a screen.

    The strip (frame, rebate, sprocket holes) lies across a lit rectangle
    narrower than the strip, with bare light above and below it, inside a dark
    surround that only receives some glare from the panel. ``camera_ev`` = 0
    puts the film base just under clipping; negative values underexpose, and
    then the bare light no longer clips, as in a careless first test shot.

    Returns the camera raw and the geometry: normalised ``frame`` rectangle
    and the camera-space ``base`` colour at the centre.
    """
    rng = rng or np.random.default_rng(3)
    scene = colorchecker_scene(600, 400) if scene is None else scene
    density = expose_film(scene, profile, film_ev)
    sh, sw = density.shape[:2]
    strip, holes = add_rebate(density, profile)
    h, w = strip.shape[:2]
    by, bx = (h - sh) // 2, (w - sw) // 2
    canvas_h, canvas_w = int(h * 2.2), int(w * 1.3)
    oy, ox = (canvas_h - h) // 2, (canvas_w - w) // 2
    cam_density = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    bare = np.ones((canvas_h, canvas_w), dtype=bool)
    cam_density[oy:oy + h, ox:ox + w] = strip @ CAMERA_DYE_CROSSTALK.T
    bare[oy:oy + h, ox:ox + w] = holes
    lit = np.zeros((canvas_h, canvas_w), dtype=bool)
    lh, lw = int(h * 1.7), int(w * 0.92)
    ly, lx = (canvas_h - lh) // 2, (canvas_w - lw) // 2
    lit[ly:ly + lh, lx:lx + lw] = True
    transmittance = np.where(bare[..., None], 1.0, np.power(10.0, -cam_density))
    light = SCREEN_TO_CAMERA @ np.asarray(light_rgb, dtype=np.float32)
    base_t = np.power(10.0, -(np.asarray(profile.base_density, dtype=np.float32) @ CAMERA_DYE_CROSSTALK.T))
    exposure = 0.85 / float(np.max(base_t * (SCREEN_TO_CAMERA @ np.ones(3, dtype=np.float32)))) * 2.0 ** camera_ev
    yy, xx = np.mgrid[-1:1:complex(canvas_h), -1:1:complex(canvas_w)]
    falloff = (1.0 - 0.15 * (xx**2 + yy**2) / 2.0)[..., None]
    signal = transmittance * lit[..., None] * light * exposure * falloff
    # Glare: a soft halo of the lit area spilling into the surround.
    from belka.core.flatfield import _box_blur

    halo = _box_blur(_box_blur(lit[..., None].astype(np.float32), 25), 25)
    signal = signal + halo * glow * light * 2.0 ** camera_ev
    signal = signal + rng.normal(0.0, 0.0006, signal.shape)
    raw = np.clip(signal, 0.0, 1.0).astype(np.float32)
    frame = ((ox + bx) / canvas_w, (oy + by) / canvas_h, (ox + bx + sw) / canvas_w, (oy + by + sh) / canvas_h)
    return raw, {"frame": frame, "base": base_t * light * exposure}
