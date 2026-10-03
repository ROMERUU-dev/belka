"""Flat-field correction.

A capture of the light source with no film in the holder records how uneven
the screen and the lens vignetting are. Dividing every frame by that map makes
the film base read the same in the corners as in the centre, which matters
because the inversion measures densities against the base.

Only the shape of the flat is kept (a smooth, low resolution map normalised to
1), so it can be shot at a lower exposure than the frames to avoid clipping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from belka.core.pipeline import downsample
from belka.i18n import _

FLAT_SIDE = 96


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return img
    out = img
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        csum = np.cumsum(padded, axis=axis, dtype=np.float64)
        zero = np.zeros_like(np.take(csum, [0], axis=axis))
        csum = np.concatenate([zero, csum], axis=axis)
        size = out.shape[axis]
        hi = np.take(csum, np.arange(2 * radius + 1, 2 * radius + 1 + size), axis=axis)
        lo = np.take(csum, np.arange(0, size), axis=axis)
        out = ((hi - lo) / (2 * radius + 1)).astype(np.float32)
    return out


def build_flat(rgb: np.ndarray) -> np.ndarray:
    small = downsample(rgb.astype(np.float32), FLAT_SIDE)
    smooth = _box_blur(_box_blur(small, 2), 2)
    peak = np.percentile(smooth.reshape(-1, smooth.shape[-1]), 99.0, axis=0)
    return (smooth / np.maximum(peak, 1e-6)).astype(np.float32)


def flat_warnings(rgb: np.ndarray) -> list[str]:
    warnings = []
    clipped = float(np.mean(np.any(rgb >= 0.98, axis=-1)))
    if clipped > 0.01:
        warnings.append(_("{pct:.0%} del flat está saturado: baja la exposición 1-2 pasos y repítelo.").format(pct=clipped))
    if float(rgb.max()) < 0.1:
        warnings.append(_("El flat está muy oscuro: sube la exposición."))
    flat = build_flat(rgb)
    spread = float(flat[..., 1].min())
    if spread < 0.5:
        warnings.append(_("La luz es muy desigual (más de 1 paso entre centro y borde): revisa el difusor."))
    return warnings


def resize_bilinear(small: np.ndarray, height: int, width: int) -> np.ndarray:
    sh, sw = small.shape[:2]
    ys = np.clip((np.arange(height) + 0.5) * sh / height - 0.5, 0, sh - 1)
    xs = np.clip((np.arange(width) + 0.5) * sw / width - 0.5, 0, sw - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (ys - y0)[:, None, None].astype(np.float32)
    wx = (xs - x0)[None, :, None].astype(np.float32)
    top = small[y0][:, x0] * (1 - wx) + small[y0][:, x1] * wx
    bottom = small[y1][:, x0] * (1 - wx) + small[y1][:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def apply_flat(rgb: np.ndarray, flat: np.ndarray | None) -> np.ndarray:
    if flat is None:
        return rgb
    full = resize_bilinear(flat, rgb.shape[0], rgb.shape[1])
    return (rgb / np.maximum(full, 0.05)).astype(np.float32)


def save_flat(path: Path, flat: np.ndarray) -> Path:
    np.save(path, flat.astype(np.float32))
    return path


def load_flat(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except (OSError, ValueError):
        return None
