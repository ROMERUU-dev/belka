"""The full develop chain, shared by the preview worker and the exporter.

    linear camera RGB ─► orient ─► lens / perspective warp ─► crop
                     ─► inversion (pipeline.render) ─► Lightroom-style adjustments

The film analysis (base, levels, detected frame) is measured on the oriented
image *before* any warp, so straightening a frame never changes its colour.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from belka.core import pipeline as pl
from belka.core.film import FilmProfile

# Fields applied by belka.core.adjust after the inversion. Changing only
# these lets the preview reuse the cached inverted image.
ADJUST_FIELDS = (
    "highlights", "shadows", "texture", "clarity", "vibrance",
    "curve_shadows", "curve_darks", "curve_lights", "curve_highlights", "curve_splits",
    "curve_rgb", "curve_red", "curve_green", "curve_blue",
    "hsl_hue", "hsl_sat", "hsl_lum",
    "grade_shadows", "grade_midtones", "grade_highlights", "grade_global", "grade_blending", "grade_balance",
    "sharpen_amount", "sharpen_radius", "sharpen_detail", "sharpen_masking", "nr_luma", "nr_color",
    "vignette_amount", "vignette_midpoint", "vignette_roundness", "vignette_feather",
    "grain_amount", "grain_size", "grain_roughness",
)
_DEFAULTS = pl.DevelopSettings()


def adjust_free(settings: pl.DevelopSettings) -> pl.DevelopSettings:
    """The settings with every adjustment at its neutral value (the inversion's cache key)."""
    return settings.copy(**{name: getattr(_DEFAULTS, name) for name in ADJUST_FIELDS},
                         disabled=tuple(d for d in settings.disabled if d in ("transform", "lens")))


def _transform():
    from belka.core import transform

    return transform


def _adjust():
    from belka.core import adjust

    return adjust


def geometry_is_identity(settings: pl.DevelopSettings) -> bool:
    return _transform().is_identity(settings)


def warp_oriented(oriented: np.ndarray, settings: pl.DevelopSettings,
                  profile: FilmProfile | None = None) -> np.ndarray:
    """Lens correction and perspective/straighten warp of an oriented image.

    The film ``profile`` lets the lens vignetting be undone through the film:
    brightening the print's corners means darkening a negative's.
    """
    tr = _transform()
    if tr.is_identity(settings):
        return oriented
    out = oriented
    if settings.section_on("lens") and (settings.lens_distortion or settings.lens_vignette):
        out = tr.lens_correct(out, settings, profile)
    out = tr.warp(out, settings)  # a switched-off Transform panel still straightens
    return out


def crop_ratio(key: str, width: int, height: int) -> float | None:
    """A crop_aspect key as a width/height ratio in pixels; None for "free"."""
    if key == "free":
        return None
    if key == "original":
        return width / max(height, 1)
    try:
        a, b = (float(v) for v in key.split(":"))
        return a / b
    except ValueError:
        return width / max(height, 1)


def effective_crop(settings: pl.DevelopSettings, analysis: pl.Analysis | None, width: int, height: int):
    """Crop in the coordinates of the warped image.

    The user's crop wins. Otherwise the automatically detected frame (found
    on the unwarped image) is carried through the warp: the largest rectangle
    inside the tilted frame, so no rebate or sprocket hole shows. With
    "Restringir recorte" and no frame, the largest rectangle with picture
    everywhere (no empty corners).
    """
    tr = _transform()
    if settings.crop is not None:
        if not settings.constrain_crop or tr.is_identity(settings):
            return settings.crop
        # Lightroom's "Constrain to image": after straightening or a perspective
        # change the user's crop shrinks, keeping its shape, until it holds
        # picture everywhere (no empty wedges, no film rebate brought in).
        x0, y0, x1, y1 = settings.crop
        aspect = (x1 - x0) * width / max((y1 - y0) * height, 1e-9)
        fitted = tr.largest_valid_rect(settings, width, height, aspect=aspect, within=settings.crop)
        return fitted if fitted[2] > fitted[0] and fitted[3] > fitted[1] else settings.crop
    if tr.is_identity(settings):
        frame = analysis.extra.get("frame") if settings.auto_crop and analysis is not None else None
        return frame
    # Automatic crops keep their own shape (the crop tool's aspect lock only
    # applies to crops drawn with it): the detected frame's, or the picture's.
    if settings.auto_crop and analysis is not None and analysis.extra.get("frame"):
        # The mapped frame already holds picture everywhere.
        return tr.map_rect_from_source(analysis.extra["frame"], settings, width, height, aspect="frame")
    if settings.constrain_crop:
        return tr.largest_valid_rect(settings, width, height, aspect=width / max(height, 1))
    return None


def geometry(img: np.ndarray, settings: pl.DevelopSettings, analysis: pl.Analysis | None,
             apply_crop: bool = True, profile: FilmProfile | None = None) -> np.ndarray:
    oriented = pl.orient(img, settings.rotation, settings.flip_h, settings.flip_v)
    warped = warp_oriented(oriented, settings, profile)
    if not apply_crop:
        return warped
    crop = effective_crop(settings, analysis, warped.shape[1], warped.shape[0])
    if crop is None:
        return warped
    ys, xs = pl.crop_slices(warped.shape, crop)
    return warped[ys, xs]


def invert(geo: np.ndarray, analysis: pl.Analysis, settings: pl.DevelopSettings, profile: FilmProfile,
           camera_matrix: np.ndarray | None, rgb_sequential: bool, strip_rows: int | None = None) -> np.ndarray:
    if strip_rows is None or geo.shape[0] <= strip_rows:
        return pl.render(geo, analysis, settings, profile, camera_matrix, rgb_sequential)
    # Per-pixel, so strips keep the temporaries of a 24 MP frame small.
    out = np.empty(geo.shape[:2] + (3,), dtype=np.float32)
    for y in range(0, geo.shape[0], strip_rows):
        out[y:y + strip_rows] = pl.render(geo[y:y + strip_rows], analysis, settings, profile, camera_matrix, rgb_sequential)
    return out


def finish(display: np.ndarray, settings: pl.DevelopSettings, scale: float, seed: int = 0) -> np.ndarray:
    """Lightroom-style adjustments; skipped for the flat output (meant for another program)."""
    if settings.output == "flat":
        return display
    adjust = _adjust()
    if adjust.is_identity(settings):
        return display
    return adjust.apply_adjustments(display, settings, scale=scale, seed=seed)


def develop(img: np.ndarray, analysis: pl.Analysis, settings: pl.DevelopSettings, profile: FilmProfile,
            camera_matrix: np.ndarray | None, rgb_sequential: bool, full_width: int | None,
            seed: int = 0, strip_rows: int | None = None) -> np.ndarray:
    geo = geometry(img, settings, analysis, profile=profile)
    display = invert(geo, analysis, settings, profile, camera_matrix, rgb_sequential, strip_rows)
    scale = img.shape[1] / full_width if full_width else 1.0
    return finish(display, settings, scale, seed)


def match_size(img: np.ndarray, shape: tuple) -> np.ndarray:
    """``img`` resized to ``shape``'s height and width (a dark-field shot decoded at another size)."""
    if img.shape[:2] == tuple(shape[:2]):
        return img
    import cv2

    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def frame_seed(frame_id: str) -> int:
    """Grain must look the same in the preview and in every export of a frame."""
    return sum(ord(c) * 131 ** i for i, c in enumerate(frame_id)) % (2**31)


def known_fields() -> set[str]:
    return {f.name for f in fields(pl.DevelopSettings)}
