"""Synthetic negatives: a forward model of film + light + camera.

Used by the simulated camera (so the whole app can be tried without hardware)
and by the tests, which check that inverting a synthetic negative gives back
the scene it was made from.
"""

from __future__ import annotations

import cv2
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


_SUPERSAMPLE = 4  # defects are drawn this much finer, then averaged: soft, sub-pixel edges


def _defect_maps(
    shape: tuple[int, int],
    picture: tuple[int, int, int, int],
    film: np.ndarray,
    rng: np.random.Generator,
    specks: int,
    fibres: int,
    scratches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(coverage, opacity, albedo) of dust specks, fibres and scratches on the film.

    Opacity is what a defect blocks of the light from below, albedo how much
    oblique light it scatters into the lens. Specks are nearly opaque; fibres
    let some light through; scratches in the base barely darken the bright
    shot but glow in the dark-field one. Most defects land on the picture, the
    rest anywhere on ``film``, and none off it.
    """
    h, w = shape
    ss = _SUPERSAMPLE
    unit = max(h, w) / 2000.0
    canvases = [np.zeros((h * ss, w * ss), dtype=np.uint8) for _ in range(3)]
    px0, py0, px1, py1 = picture
    film_ys, film_xs = np.nonzero(film)

    def spot() -> np.ndarray:
        if rng.random() < 0.75:
            return np.array([rng.uniform(px0, px1), rng.uniform(py0, py1)])
        i = rng.integers(len(film_ys))
        return np.array([film_xs[i], film_ys[i]], dtype=float)

    def draw(paint, opacity: float, albedo: float) -> None:
        for canvas, value in zip(canvases, (1.0, opacity, albedo)):
            paint(canvas, int(round(255 * value)))

    def sub(p) -> tuple[int, int]:
        return int(p[0] * ss), int(p[1] * ss)

    for _ in range(specks):
        centre = spot()
        radius = float(np.clip(np.exp(rng.normal(np.log(2.2 * unit), 0.6)), 0.7 * unit, 9.0 * unit))
        lobes = [(centre + rng.normal(0.0, 0.5 * radius, 2), radius * rng.uniform(0.5, 1.0),
                  radius * rng.uniform(0.4, 1.0), rng.uniform(0, 180)) for _ in range(rng.integers(1, 4))]

        def blob(canvas: np.ndarray, value: int, lobes=lobes) -> None:
            for p, a, b, angle in lobes:
                cv2.ellipse(canvas, sub(p), (max(int(a * ss), 1), max(int(b * ss), 1)), angle, 0, 360, value, -1)

        draw(blob, rng.uniform(0.75, 1.0), rng.uniform(0.4, 1.0))
    for _ in range(fibres):
        p, angle = spot(), rng.uniform(0, 2 * np.pi)
        points = [p]
        for _ in range(int(rng.uniform(40, 220) * unit / 4)):
            angle += rng.normal(0.0, 0.25)
            points.append(points[-1] + 4.0 * np.array([np.cos(angle), np.sin(angle)]))
        line = np.array([sub(q) for q in points], dtype=np.int32)
        width = max(int(rng.uniform(1.0, 2.2) * unit * ss), 1)
        draw(lambda c, v, line=line, width=width: cv2.polylines(c, [line], False, v, width),
             rng.uniform(0.5, 0.85), rng.uniform(0.4, 0.8))
    for _ in range(scratches):
        length = rng.uniform(0.3, 0.8) * (px1 - px0)
        x0, y0 = rng.uniform(px0, px1 - length), rng.uniform(py0, py1)
        slope = np.tan(np.radians(rng.normal(0.0, 1.5)))
        xs = np.linspace(x0, x0 + length, 24)
        ys = y0 + slope * (xs - x0) + 2.0 * unit * np.sin(np.linspace(0, np.pi, 24))
        line = np.array([sub(q) for q in zip(xs, ys)], dtype=np.int32)
        width = max(int(rng.uniform(0.8, 1.4) * unit * ss), 1)
        draw(lambda c, v, line=line, width=width: cv2.polylines(c, [line], False, v, width),
             rng.uniform(0.1, 0.2), rng.uniform(0.6, 1.0))
    # Averaged down, then softened a little by the copy lens.
    blur = 0.5 * unit
    maps = [cv2.GaussianBlur(cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0,
                             (0, 0), blur) * film for c in canvases]
    coverage = np.maximum(maps[0], 1e-6)
    return maps[0], np.clip(maps[1] / coverage, 0, 1) * maps[0], np.clip(maps[2] / coverage, 0, 1) * maps[0]


def dusty_scan(
    profile: FilmProfile,
    size: tuple[int, int] = (1500, 1000),
    specks: int = 40,
    fibres: int = 5,
    scratches: int = 2,
    darkfield_stops: float = 4.0,
    grain: float = 0.02,
    scene: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A dusty strip on a copy stand, its dark-field shot and where the dirt is.

    The strip runs across a ``size`` (width, height) camera view, lit from
    below by a rectangle of the screen with bare light above and below it and
    sprocket holes along both edges, the film grainy (``grain`` = RMS density
    of each dye layer). For the dark-field shot the rectangle goes black and a
    ring around it lights up; the camera exposes ``darkfield_stops`` longer.
    Light spreading in the diffuser leaves a glow under the film (strongest
    near the ring), the screen's black level shows the film faintly, and
    what scatters the oblique light glows: dust, fibres, scratches, and the
    film's own cut edges and sprocket holes. A long exposure also brings out a
    few hot pixels. ``scene`` replaces the default landscape in the picture.

    Returns (bright, darkfield, truth): both linear camera raws and a mask of
    the pixels a defect noticeably covers.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    unit = max(h, w) / 2000.0
    # The picture with its rebate fills the middle of the strip, which runs
    # past both ends of the lit rectangle as on a real copy stand.
    ph = int(0.46 * h / (1 + 2 * 0.126))
    pw = int(ph * 1.5)
    scene = landscape_scene(pw, ph, seed=seed) if scene is None else scene
    ph, pw = scene.shape[:2]
    density = expose_film(scene, profile)
    base = np.asarray(profile.base_density, dtype=np.float32)
    film_w = int(w / (1 + 2 * 0.07 * 0.6))
    roll = np.empty((ph, film_w, 3), dtype=np.float32)
    roll[:] = base
    fx = (film_w - pw) // 2
    roll[:, fx:fx + pw] = density
    strip, holes = add_rebate(roll, profile)
    sh, sw = strip.shape[:2]
    oy, ox = (h - sh) // 2, (w - sw) // 2
    dens = np.zeros((h, w, 3), dtype=np.float32)
    film = np.zeros((h, w), dtype=bool)
    ys, xs = slice(max(oy, 0), min(oy + sh, h)), slice(max(ox, 0), min(ox + sw, w))
    cut = strip[max(-oy, 0):, max(-ox, 0):][:ys.stop - ys.start, :xs.stop - xs.start]
    dens[ys, xs] = cut
    film[ys, xs] = ~holes[max(-oy, 0):, max(-ox, 0):][:ys.stop - ys.start, :xs.stop - xs.start]
    # Grain: dye clouds, partly shared by the three layers, about a pixel across.
    noise = rng.standard_normal((h, w, 4), dtype=np.float32)
    noise = cv2.GaussianBlur(0.6 * noise[..., :3] + 0.8 * noise[..., 3:], (0, 0), 0.6)
    noise *= grain / max(float(noise[film].std()), 1e-6)
    dens = np.where(film[..., None], dens + noise, 0.0)
    px0, py0 = ox + (sw - film_w) // 2 + fx, oy + (sh - ph) // 2
    picture = (px0, py0, px0 + pw, py0 + ph)
    rect = np.zeros((h, w), dtype=np.float32)
    rect[int(0.14 * h):int(0.86 * h), int(0.18 * w):int(0.82 * w)] = 1.0
    # Dirt on the film beyond the lit rectangle shows in neither shot that matters.
    coverage, opacity, albedo = _defect_maps((h, w), picture, film & (rect > 0), rng, specks, fibres, scratches)
    transmittance = np.where(film[..., None], np.power(10.0, -(dens @ CAMERA_DYE_CROSSTALK.T)), 1.0)
    transmittance *= (1.0 - opacity)[..., None]
    light = SCREEN_TO_CAMERA @ np.ones(3, dtype=np.float32)
    base_t = np.power(10.0, -(base @ CAMERA_DYE_CROSSTALK.T))
    exposure = 0.85 / float(np.max(base_t * light))
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    falloff = (1.0 - 0.15 * (xx**2 + yy**2) / 2.0)[..., None]

    def shoot(illumination: np.ndarray, scattered: np.ndarray, ev: float) -> np.ndarray:
        gain = light * np.float32(exposure * 2.0**ev)
        signal = (transmittance * illumination[..., None] + scattered[..., None]) * gain * falloff
        shot_noise = np.float32(0.002) * np.sqrt(np.maximum(signal, 0.0) + np.float32(0.01))
        return signal + rng.standard_normal(signal.shape, dtype=np.float32) * shot_noise

    box = 2 * int(25 * unit) + 1
    glare = cv2.blur(cv2.blur(rect, (box, box), borderType=cv2.BORDER_REPLICATE), (box, box),
                     borderType=cv2.BORDER_REPLICATE)
    bright = shoot(rect, 0.004 * glare, 0.0)

    ring_w = int(0.04 * w)
    ring = cv2.dilate(rect, np.ones((2 * ring_w + 1, 2 * ring_w + 1), np.uint8)) - rect
    small = cv2.resize(ring, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 0.06 * w / 8), (w, h), interpolation=cv2.INTER_LINEAR)
    oblique = 0.004 + 0.15 * glow
    edges = cv2.morphologyEx(film.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edges = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 0.7 * unit)
    film_scatter = np.where(film, 0.004 * (dens.mean(axis=-1) - float(base.mean())), 0.0).astype(np.float32)
    scattered = oblique * (albedo + 0.4 * edges + film_scatter)
    darkfield = shoot(ring + 0.002 + 0.01 * glow, scattered, darkfield_stops)
    hot = rng.integers(0, h * w, 25)
    darkfield.reshape(-1, 3)[hot, rng.integers(0, 3, 25)] += rng.uniform(0.03, 0.4, 25)

    truth = coverage > 0.2
    return (np.clip(bright, 0.0, 1.0).astype(np.float32), np.clip(darkfield, 0.0, 1.0).astype(np.float32), truth)


def night_sky_scene(width: int = 900, height: int = 600, stars: int = 60, brightest: float = 4.0,
                    warm: bool = False, seed: int = 0) -> np.ndarray:
    """A night sky full of stars over dark ground; ``warm`` makes them the lights of a town.

    Point highlights: on the negative each is a small, sharp, dense dot, as a
    speck of dust is on the capture. ``brightest`` is the peak of the
    brightest one (scene-linear, the sky is 0.01-0.03).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    scene = np.empty((height, width, 3), dtype=np.float32)
    scene[:] = (0.01, 0.015, 0.03)
    scene[yy > height * 0.75] = (0.03, 0.03, 0.025)
    colour = np.array((1.0, 0.85, 0.6) if warm else (1.0, 1.0, 1.0), dtype=np.float32)
    for _ in range(stars):
        x, y = rng.uniform(10, width - 10), rng.uniform(10, height * 0.7)
        r, peak = rng.uniform(0.6, 2.0), rng.uniform(0.075, 1.0) * brightest
        scene += (peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * r * r)))[..., None] * colour
    return scene


def catchlight_scene(width: int = 900, height: int = 600, eyes: int = 30, seed: int = 0) -> np.ndarray:
    """Skin with dark eyes, each with the tiny specular reflection of a light in it."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    scene = np.empty((height, width, 3), dtype=np.float32)
    scene[:] = (0.12, 0.08, 0.06)
    for _ in range(eyes):
        x, y = rng.uniform(10, width - 10), rng.uniform(10, height - 10)
        scene[(xx - x) ** 2 + (yy - y) ** 2 < 25] = (0.02, 0.015, 0.01)
        scene[(xx - x - 1) ** 2 + (yy - y + 1) ** 2 < rng.uniform(1.0, 3.0) ** 2] = 1.5
    return scene


def opaque_specks(raw: np.ndarray, where: np.ndarray, count: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``raw`` with ``count`` opaque specks of dust dropped inside ``where``, and
    the mask of the pixels they noticeably cover.

    Mineral dust and soot block nearly all the light; ``dusty_scan``'s specks
    let up to a quarter through. Same sizes and soft edges; where a speck
    covers a pixel it shows the lens's veiling glare (half a percent of the
    brightest light) instead of the film.
    """
    rng = np.random.default_rng(seed)
    h, w = where.shape
    ss = _SUPERSAMPLE
    unit = max(h, w) / 2000.0
    canvas = np.zeros((h * ss, w * ss), dtype=np.uint8)
    ys, xs = np.nonzero(where)
    for i in rng.integers(0, ys.size, count):
        centre = np.array([xs[i], ys[i]], dtype=float)
        radius = float(np.clip(np.exp(rng.normal(np.log(2.2 * unit), 0.6)), 0.7 * unit, 9.0 * unit))
        for _ in range(rng.integers(1, 4)):
            p = (centre + rng.normal(0.0, 0.5 * radius, 2)) * ss
            axes = (max(int(radius * rng.uniform(0.5, 1.0) * ss), 1), max(int(radius * rng.uniform(0.4, 1.0) * ss), 1))
            cv2.ellipse(canvas, (int(p[0]), int(p[1])), axes, rng.uniform(0, 180), 0, 360, 255, -1)
    cover = cv2.GaussianBlur(cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0,
                             (0, 0), 0.5 * unit)
    glare = 0.005 * float(np.percentile(raw, 99.5))
    soiled = raw * (1.0 - cover[..., None]) + glare * cover[..., None]
    return soiled.astype(raw.dtype), cover > 0.2
