"""Writing positives: 16-bit TIFF and JPEG, both with an embedded ICC profile."""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from belka import __version__
from belka.core import develop as dv
from belka.core import pipeline as pl
from belka.core.film import ProfileLibrary
from belka.core.session import Frame, Session, load_frame

STRIP_ROWS = 1024  # the inversion keeps its own temporaries small; strips only bound the output pass

FORMATS = {
    "tiff16": ("TIFF 16 bits", ".tif"),
    "jpeg": ("JPEG", ".jpg"),
}


@dataclass
class ExportOptions:
    formats: tuple[str, ...] = ("tiff16", "jpeg")
    jpeg_quality: int = 95
    max_side: int | None = None  # None = full resolution
    folder: Path | None = None


def icc_profile(linear: bool) -> bytes:
    from PySide6.QtGui import QColorSpace

    name = QColorSpace.NamedColorSpace.SRgbLinear if linear else QColorSpace.NamedColorSpace.SRgb
    return bytes(QColorSpace(name).iccProfile())


def develop_full(session: Session, frame: Frame, library: ProfileLibrary, max_side: int | None = None) -> tuple[np.ndarray, pl.DevelopSettings]:
    """The developed frame; with ``max_side`` exactly that long on its long side.

    The size applies to the final, cropped picture, so the capture is decoded
    at a resolution where the crop still has at least ``max_side`` pixels
    (half-size raw when it does, full otherwise) and then resized down.
    """
    settings = session.settings_for(frame)
    profile = library.get(settings.profile_id)
    image = load_frame(session, frame, half_size=max_side is not None)
    analysis = pl.analyze(image.rgb, settings, profile)
    if max_side is not None and _cropped_side(image, analysis, settings) < max_side:
        image = load_frame(session, frame, half_size=False)
        analysis = pl.analyze(image.rgb, settings, profile)
    full_w = (image.meta.get("full_size") or (image.rgb.shape[1], 0))[0]
    out = dv.develop(
        image.rgb, analysis, settings, profile, image.camera_matrix, bool(image.meta.get("rgb_sequential")),
        full_width=full_w, seed=dv.frame_seed(frame.id), strip_rows=STRIP_ROWS,
    )
    if max_side is not None and max(out.shape[:2]) > max_side:
        import cv2

        h, w = out.shape[:2]
        k = max_side / max(h, w)
        out = cv2.resize(out, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    return out, settings


def _cropped_side(image, analysis: pl.Analysis, settings: pl.DevelopSettings) -> float:
    h, w = image.rgb.shape[:2]
    if settings.rotation % 180:
        w, h = h, w
    crop = dv.effective_crop(settings, analysis, w, h) or (0.0, 0.0, 1.0, 1.0)
    return max((crop[2] - crop[0]) * w, (crop[3] - crop[1]) * h)


def write_tiff16(path: Path, rgb: np.ndarray, linear: bool, description: str) -> Path:
    import tifffile

    with _atomic(path) as tmp:
        tifffile.imwrite(
            str(tmp),
            pl.to_uint16(rgb),
            photometric="rgb",
            compression="zlib",
            description=description,
            metadata=None,
            iccprofile=icc_profile(linear),
            software=f"Belka {__version__}",
        )
    return path


def write_jpeg(path: Path, rgb: np.ndarray, quality: int, linear: bool) -> Path:
    from PySide6.QtGui import QColorSpace, QImage

    data = pl.to_uint8(pl.srgb_encode(rgb) if linear else rgb)
    h, w = data.shape[:2]
    image = QImage(data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
    image.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    with _atomic(path) as tmp:
        if not image.save(str(tmp), "JPG", quality):
            raise OSError(f"No se pudo escribir {path}")
    return path


@contextmanager
def _atomic(path: Path):
    """Write to a sibling temp file and move it into place only on success,
    so a failed export never leaves a stub or clobbers an earlier positive."""
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=f".part{path.suffix}", dir=path.parent)
    os.close(fd)
    tmp = Path(name)  # unique: a second export of the same frame never shares it
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def export_stem(session: Session, frame: Frame, folder: Path) -> str:
    """File name for a frame's positive.

    Inside the roll's own folder the roll name is enough; anywhere else the
    roll's dated folder name is used, so two rolls both called "Portra 400"
    exporting to the same place cannot overwrite each other.
    """
    if folder.resolve().is_relative_to(session.path.resolve()):
        return session.new_capture_basename(frame.id)
    safe = re.sub(r"[^\w\-]+", "_", session.path.name)
    return f"{safe}_{frame.id}"


def export_frame(session: Session, frame: Frame, library: ProfileLibrary, options: ExportOptions) -> list[Path]:
    rgb, settings = develop_full(session, frame, library, options.max_side)
    profile = library.get(settings.profile_id)
    linear = settings.output == "flat"
    folder = options.folder or session.export_dir
    folder.mkdir(parents=True, exist_ok=True)
    stem = export_stem(session, frame, folder)
    # TIFF text tags must be 7-bit ASCII ("Genérico C-41" broke every export):
    # \uXXXX escapes keep the JSON exact.
    description = json.dumps(
        {"app": "Belka", "version": __version__, "film": profile.name, "settings": settings.to_json()},
        ensure_ascii=True,
    )
    written, errors = [], []
    for fmt in options.formats:
        _, ext = FORMATS[fmt]
        path = folder / f"{stem}{ext}"
        try:
            if fmt == "tiff16":
                written.append(write_tiff16(path, rgb, linear, description))
            elif fmt == "jpeg":
                written.append(write_jpeg(path, rgb, options.jpeg_quality, linear))
        except Exception as exc:  # one format failing must not cost the other
            errors.append(f"{FORMATS[fmt][0]}: {exc}")
    if written:
        frame.exported = [str(p.relative_to(session.path)) if p.is_relative_to(session.path) else str(p) for p in written]
        session.save()
    if errors:
        raise OSError("; ".join(errors))
    return written
