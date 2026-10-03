"""Synthetic 35 mm strips with DX edge barcodes drawn to the format, for the edgeprint tests.

A strip is described in millimetres (x along it, y across from the edge
without the code) and photographed by mapping every camera pixel back onto
the film: perforations (KS, 4.75 mm pitch), 36 x 24 mm frames every 38 mm,
the eye-readable frame numbers and, along the far edge, one code every half
frame with its clock track next to the perforations and its data track next
to the edge. The camera sees it backlit, like on the screen: bare light
around the strip and through the holes, each pixel averaging the light it
gets, not the density.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FILM_MM = 35.0
MODULE_MM = 0.4025  # measured on a Kodak strip against the perforation pitch
HALF_FRAME_MM = 19.0
FRAME_PITCH_MM = 38.0
PICTURE_MM = (36.0, 24.0)
PERF_PITCH_MM, PERF_W_MM, PERF_H_MM, PERF_EDGE_MM = 4.75, 2.794, 1.981, 2.01
CLOCK_Y_MM = (33.10, 33.98)  # the clock track, next to the perforations
DATA_Y_MM = (33.98, 34.86)  # the data track, next to the film edge
LABEL_GAP_MM = 4.8 * MODULE_MM  # printed number's centre before the code start


def code_bits(part1: int, part2: int, number: int | None = None, half: bool = False,
              parity_error: bool = False) -> tuple[list[int], list[int]]:
    """Clock and data modules (1 = bar) of one code; 31 modules with a frame
    number, 23 without. ``parity_error`` flips the parity bit."""
    assert 1 <= part1 < 128 and 0 <= part2 < 16
    p1 = [int(b) for b in f"{part1:07b}"]
    p2 = [int(b) for b in f"{part2:04b}"]
    if number is None:
        payload = p1 + p2
        data = [1, 0, 1, 0, 1, 0] + p1 + [0] + p2 + [(sum(payload) + parity_error) % 2] + [0, 1, 0, 1]
    else:
        fn = [int(b) for b in f"{number:06b}"] + [int(half)]
        payload = p1 + p2 + fn
        data = [1, 0, 1, 0, 1, 0] + p1 + [0] + p2 + fn + [0, (sum(payload) + parity_error) % 2, 0, 1, 0, 1]
    n = len(data)
    clock = [1] * 5 + [(k - 5) % 2 for k in range(5, n - 3)] + [1] * 3
    return clock, data


@dataclass
class Strip:
    """What to draw: the film and how the camera sees it."""

    part1: int = 95
    part2: int = 7
    first: int = 20  # frame number of the code that starts code_offset_mm into frame 0
    numbered: bool = True  # 31-module codes with frame numbers (else 23-module)
    code_offset_mm: float = 7.0  # where the first code starts, from the first frame's left edge
    kind: str = "negative"  # "negative", "bw" or "slide"
    contrast: float = 1.0  # scales the code's density
    parity_error: bool = False
    damage: tuple[str, tuple[int, ...]] | None = None  # (label, data modules) turned over on that one code
    codes: bool = True
    stripes_mm: float | None = None  # a picket fence in the pictures, this period
    # camera
    px_per_mm: float = 12.0
    view_mm: tuple[float, float] = (100.0, 50.0)  # along, across
    angle: float = 0.0  # degrees counter-clockwise; 90 makes the strip vertical
    mirrored: bool = False  # the film turned over
    shift_mm: float = 0.0  # the strip slid along under the camera
    blur: float = 0.0  # Gaussian sigma, camera pixels
    noise: float = 0.0  # standard deviation, linear camera units
    seed: int = 0


def label(value: int) -> str:
    """Printed label of a half-frame index (2 * number + half)."""
    return f"{value // 2}{'A' if value % 2 else ''}"


def _codes(strip: Strip, x0: float, x1: float) -> list[tuple[int, float, np.ndarray, np.ndarray]]:
    """(half-frame index, start mm, clock, data) of every code starting in [x0 - 19, x1]."""
    out = []
    first_start = 1.0 + strip.code_offset_mm  # frame 0's picture starts at x = 1 mm
    for j in range(int(np.floor((x0 - HALF_FRAME_MM - first_start) / HALF_FRAME_MM)),
                   int(np.ceil((x1 - first_start) / HALF_FRAME_MM)) + 1):
        value = 2 * strip.first + j
        if value < 0:
            continue
        clock, data = code_bits(strip.part1, strip.part2, value // 2 if strip.numbered else None,
                                bool(value % 2), strip.parity_error)
        data = np.array(data)
        if strip.damage and strip.damage[0] == label(value):
            data[list(strip.damage[1])] ^= 1  # specks of dust, dense in transmission, or scratches
        out.append((value, first_start + j * HALF_FRAME_MM, np.array(clock), data))
    return out


def _film(strip: Strip, x0: float, x1: float, res: float) -> tuple[np.ndarray, ...]:
    """The film between x0 and x1 mm at ``res`` pixels per mm, as uint8 maps:
    film (0 on holes), picture, scene brightness and printed ink."""
    w, h = int(np.ceil((x1 - x0) * res)), int(np.ceil(FILM_MM * res))
    xs = x0 + (np.arange(w) + 0.5) / res
    ys = (np.arange(h) + 0.5) / res
    film = np.full((h, w), 255, np.uint8)
    picture = np.zeros((h, w), np.uint8)
    scene = np.zeros((h, w), np.uint8)
    ink = np.zeros((h, w), np.uint8)
    # perforations, rounded corners
    phase = np.mod(xs, PERF_PITCH_MM)[None, :]
    for cy in (PERF_EDGE_MM + PERF_H_MM / 2, FILM_MM - PERF_EDGE_MM - PERF_H_MM / 2):
        dx = np.maximum(np.abs(phase - PERF_PITCH_MM / 2) - (PERF_W_MM / 2 - 0.5), 0.0)
        dy = np.maximum(np.abs(ys[:, None] - cy) - (PERF_H_MM / 2 - 0.5), 0.0)
        film[(dx * dx + dy * dy <= 0.25) & (np.abs(phase - PERF_PITCH_MM / 2) <= PERF_W_MM / 2)
             & (np.abs(ys[:, None] - cy) <= PERF_H_MM / 2)] = 0
    # pictures: soft blobs and a ramp (drawn coarse, they are smooth) and, if asked, a picket fence
    blobs = np.random.default_rng(strip.seed + 1).random((12, 4))
    rows = np.nonzero((ys >= 5.5) & (ys < 5.5 + PICTURE_MM[1]))[0]
    fi = np.floor((xs - 1.0) / FRAME_PITCH_MM)
    u = (xs - 1.0 - fi * FRAME_PITCH_MM) / PICTURE_MM[0]
    cols = np.nonzero(u < 1.0)[0]
    uu, vv = np.meshgrid(np.linspace(0.0, 1.0, 145), np.linspace(0.0, 1.0, 97))
    coarse = 0.3 + 0.5 * uu
    for bx, by, br, bv in blobs:
        coarse += (bv - 0.4) * np.exp(-((uu - bx) ** 2 + (vv - by) ** 2) / (0.02 + 0.1 * br))
    value = cv2.resize(coarse.astype(np.float32), (round(PICTURE_MM[0] * res), len(rows)),
                       interpolation=cv2.INTER_LINEAR)
    ix = np.minimum((u[cols] * value.shape[1]).astype(int), value.shape[1] - 1)
    value = value[:, ix]
    if strip.stripes_mm:
        value = value + 0.5 * (np.sin(2 * np.pi * u[cols] * PICTURE_MM[0] / strip.stripes_mm) > 0)
    block = np.ix_(rows, cols)
    picture[block] = 255
    scene[block] = np.round(np.clip(value / 1.6, 0.0, 1.0) * 255)
    # edge print: the name on the near edge, and every code with its number on the far one
    for i in range(int(fi.min()), int(fi.max()) + 1):
        _text(ink, "KODAK GC 400", (1.0 + i * FRAME_PITCH_MM + 14.0 - x0) * res, 1.0 * res, 1.1 * res)
    if strip.codes:
        clock_rows = slice(round(CLOCK_Y_MM[0] * res), round(CLOCK_Y_MM[1] * res))
        data_rows = slice(round(DATA_Y_MM[0] * res), round(DATA_Y_MM[1] * res))
        for value, start, clock, data in _codes(strip, x0, x1):
            k = np.floor((xs - start) / MODULE_MM).astype(int)
            on = (k >= 0) & (k < len(clock))
            kk = np.clip(k, 0, len(clock) - 1)
            ink[clock_rows, on & (clock[kk] == 1)] = 255
            ink[data_rows, on & (data[kk] == 1)] = 255
            if strip.numbered:
                _text(ink, label(value), (start - LABEL_GAP_MM - x0) * res, 34.0 * res, 1.0 * res)
    return film, picture, scene, ink


def _text(canvas: np.ndarray, text: str, cx: float, cy: float, height: float) -> None:
    """Draws ``text`` centred on (cx, cy), ``height`` pixels tall."""
    scale = height / 22.0
    thick = max(1, round(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(canvas, text, (round(cx - tw / 2), round(cy + th / 2)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255,
                thick, cv2.LINE_AA)


def render(strip: Strip) -> tuple[np.ndarray, dict]:
    """The camera image (linear, float32 H x W x 3) and the geometry drawn:
    ``frame``, the normalised rectangle of the frame nearest the centre (the
    part in view); ``codes`` [(label, start px, end px)] of the codes wholly
    in view; and the numbers printed nearest the frame's centre and nearest
    the image's centre."""
    s = strip.px_per_mm
    along, across = strip.view_mm
    theta = np.deg2rad(strip.angle)
    c, sn = np.cos(theta), np.sin(theta)
    w = int(round((abs(along * c) + abs(across * sn)) * s))
    h = int(round((abs(along * sn) + abs(across * c)) * s))
    x_centre = 60.0 + strip.shift_mm

    def to_px(x, y):
        yy = FILM_MM - y if strip.mirrored else y
        dx, dy = x - x_centre, yy - FILM_MM / 2
        return w / 2 + s * (dx * c + dy * sn), h / 2 + s * (-dx * sn + dy * c)

    def to_film(px, py):
        dx, dy = (px - w / 2) / s, (py - h / 2) / s
        x, yy = x_centre + dx * c - dy * sn, FILM_MM / 2 + dx * sn + dy * c
        return x, (FILM_MM - yy if strip.mirrored else yy)

    reach = np.hypot(w, h) / s / 2 + 2.0
    x0, x1 = x_centre - reach, x_centre + reach
    res = min(max(16.0, 2.0 * s), 64.0)
    film, picture, scene, ink = (m.astype(np.float32) / np.float32(255.0) for m in _film(strip, x0, x1, res))
    scene = scene * 1.6
    # Density on the film above its base, the same in every channel (a slide's is all of it).
    if strip.kind == "slide":
        base = np.zeros(3, np.float32)
        dens = 2.9 + picture * (2.6 - 1.6 * scene - 2.9) - 2.4 * strip.contrast * ink
    else:
        base = np.array([0.22, 0.22, 0.22] if strip.kind == "bw" else [0.25, 0.62, 0.92], np.float32)
        gain, bar = (0.9, 1.2) if strip.kind == "bw" else (0.7, 0.45)
        dens = picture * scene * gain + bar * strip.contrast * ink
    # Each camera pixel averages the light it gets, through the film and through the holes:
    # blur both to its size on the film, then sample.
    box = max(1, round(res / s))
    through = cv2.blur(film * np.power(np.float32(10.0), -dens), (box, box))
    holes = cv2.blur(np.float32(1.0) - film, (box, box))
    py, px = np.mgrid[0:h, 0:w].astype(np.float32)
    fx, fy = to_film(px + 0.5, py + 0.5)
    mx, my = ((fx - x0) * res - 0.5).astype(np.float32), (fy * res - 0.5).astype(np.float32)
    through, holes = (cv2.remap(m, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
                      for m, fill in ((through, 0.0), (holes, 1.0)))
    trans = through[..., None] * np.power(np.float32(10.0), -base) + holes[..., None]
    # The screen behind: brightest at the centre, exposed to sit just under clipping.
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    light = (0.9 * (1.0 - 0.12 * (xx ** 2 + yy ** 2) / 2.0))[..., None] * np.array([0.92, 1.0, 0.96])
    img = trans * light.astype(np.float32)
    if strip.blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), strip.blur)
    if strip.noise > 0:
        rng = np.random.default_rng(strip.seed)
        img = img + rng.normal(0.0, strip.noise, img.shape) * np.sqrt(np.maximum(img, 1e-4) / 0.5)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # What the tests check against.
    centre_x = to_film(w / 2, h / 2)[0]
    frame_x = 1.0 + np.round((centre_x - 1.0 - PICTURE_MM[0] / 2) / FRAME_PITCH_MM) * FRAME_PITCH_MM
    corners = [to_px(frame_x + a, 5.5 + b) for a in (0, PICTURE_MM[0]) for b in (0, PICTURE_MM[1])]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    # As an analysis reports it: the part of the frame in view.
    frame = (max(min(xs), 0) / w, max(min(ys), 0) / h, min(max(xs), w) / w, min(max(ys), h) / h)
    codes = _codes(strip, x0 - 40.0, x1 + 40.0)
    in_view = []
    for value, start, clock, _ in codes:
        end = start + len(clock) * MODULE_MM
        corners = [to_px(x, y) for x in (start, end) for y in (CLOCK_Y_MM[0], DATA_Y_MM[1])]
        if all(0 <= p[0] < w and 0 <= p[1] < h for p in corners):
            in_view.append((label(value) if strip.numbered else "", to_px(start, CLOCK_Y_MM[0]),
                            to_px(end, CLOCK_Y_MM[0])))
    labels = [(label(value), to_px(start - LABEL_GAP_MM, 34.0)) for value, start, *_ in codes] if strip.numbered else []
    centre = ((frame[0] + frame[2]) / 2 * w, (frame[1] + frame[3]) / 2 * h)
    return img, {"frame": frame, "codes": in_view, "label_at_frame": _nearest_label(labels, centre),
                 "label_at_centre": _nearest_label(labels, (w / 2, h / 2))}


def _nearest_label(labels: list[tuple[str, tuple[float, float]]], point: tuple[float, float]) -> str:
    """The printed number nearest ``point`` (camera pixels), "" without numbers."""
    if not labels:
        return ""
    return min(labels, key=lambda lb: np.hypot(lb[1][0] - point[0], lb[1][1] - point[1]))[0]
