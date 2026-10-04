"""Loading captures as linear camera RGB.

* Camera raw files (NEF, CR2/CR3, ARW, RAF, DNG, ...) go through LibRaw with
  no white balance, no colour matrix and no gamma: the film base, not the
  camera, decides the white balance later.
* Belka's own linear TIFFs (simulated camera, RGB-sequential composites)
  are tagged in their ImageDescription and read as-is.
* Any other TIFF/JPEG/PNG is assumed to be sRGB-encoded and is linearised.
"""

from __future__ import annotations

import errno
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from belka.core.pipeline import downsample, srgb_decode

RAW_EXTENSIONS = {
    ".nef", ".nrw", ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2", ".raf", ".rw2", ".orf",
    ".pef", ".dng", ".3fr", ".iiq", ".srw", ".x3f", ".erf", ".kdc", ".mef", ".mos",
}
IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | IMAGE_EXTENSIONS
LINEAR_TAG = "belka:linear-camera-rgb"

SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


@dataclass
class LinearImage:
    rgb: np.ndarray  # float32, HxWx3, linear, 1.0 = sensor clipping
    camera_matrix: np.ndarray | None = None  # camera RGB -> linear sRGB, rows sum to 1
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def camera_to_srgb(rgb_xyz_matrix: np.ndarray) -> np.ndarray | None:
    """dcraw's recipe: XYZ->camera matrix to a normalised camera->sRGB one."""
    xyz_cam = np.asarray(rgb_xyz_matrix, dtype=np.float64)[:3]
    if not np.any(xyz_cam):
        return None
    cam_rgb = xyz_cam @ SRGB_TO_XYZ
    sums = cam_rgb.sum(axis=1, keepdims=True)
    if np.any(np.abs(sums) < 1e-9):
        return None
    cam_rgb = cam_rgb / sums
    try:
        return np.linalg.inv(cam_rgb)
    except np.linalg.LinAlgError:
        return None


def load_linear(path: str | Path, half_size: bool = True, max_side: int | None = None) -> LinearImage:
    path = Path(path)
    if not path.is_file():
        # LibRaw would only say "Input/output error".
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        image = _load_raw(path, half_size)
    elif suffix in {".tif", ".tiff"}:
        image = _load_tiff(path)
    elif suffix in IMAGE_EXTENSIONS:
        image = _load_qt_image(path)
    else:
        raise ValueError(f"Formato no soportado: {path.name}")
    if "full_size" not in image.meta:
        image.meta["full_size"] = (int(image.rgb.shape[1]), int(image.rgb.shape[0]))
    if max_side:
        image.rgb = downsample(image.rgb, max_side)
    return image


def read_exif(path: str | Path) -> dict:
    """ISO, shutter, aperture and focal length from TIFF-based raws (NEF, DNG, ...)."""
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            tags = tif.pages[0].tags
            exif = tags["ExifTag"].value if "ExifTag" in tags else {}
            model = tags["Model"].value if "Model" in tags else ""
    except Exception:
        return {}

    def ratio(v):
        if isinstance(v, tuple) and len(v) == 2 and v[1]:
            return v[0] / v[1]
        return v

    out = {"model": str(model).strip()}
    if exif.get("ISOSpeedRatings"):
        out["iso"] = int(exif["ISOSpeedRatings"] if not isinstance(exif["ISOSpeedRatings"], tuple) else exif["ISOSpeedRatings"][0])
    if exif.get("ExposureTime"):
        out["exposure"] = float(ratio(exif["ExposureTime"]))
    if exif.get("FNumber"):
        out["fnumber"] = float(ratio(exif["FNumber"]))
    if exif.get("FocalLength"):
        out["focal"] = float(ratio(exif["FocalLength"]))
    return out


def exif_summary(info: dict) -> str:
    parts = []
    if info.get("iso"):
        parts.append(f"ISO {info['iso']}")
    if info.get("focal"):
        parts.append(f"{info['focal']:.0f} mm")
    if info.get("fnumber"):
        parts.append(f"f/{info['fnumber']:.1f}".replace(".0", ""))
    if info.get("exposure"):
        t = info["exposure"]
        parts.append(f"1/{round(1 / t)} s" if t < 0.5 else f"{t:.1f} s".replace(".0 s", " s"))
    return "   ".join(parts)


def _load_raw(path: Path, half_size: bool) -> LinearImage:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        black = float(np.mean(raw.black_level_per_channel))
        saturation = _camera_saturation(raw) or int(raw.white_level)
        # No LibRaw scaling at all: it would normalise to the 14-bit ceiling
        # (16383) rather than where the sensor really clips (often lower,
        # e.g. 15311), and rescale each image to its own brightest pixel. Unscaled,
        # the output is exactly raw minus black, so dividing by the usable
        # range makes 1.0 mean "clipped" in every frame of a roll.
        rgb16 = raw.postprocess(
            half_size=half_size,
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
            output_color=rawpy.ColorSpace.raw,
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            no_auto_scale=True,
            adjust_maximum_thr=0.0,
            highlight_mode=rawpy.HighlightMode.Clip,
            # Sensor orientation always: on a copy stand the camera's tilt
            # sensor is arbitrary (prueba_001.nef came out as flip 5), and
            # flats, RGB parts and frames must line up. Belka rotates itself.
            user_flip=0,
        )
        matrix = camera_to_srgb(raw.rgb_xyz_matrix)
        rgb16 = _crop_margins(rgb16, raw.sizes, half_size)
        sizes = raw.sizes
        full_size = (int(sizes.crop_width or sizes.width), int(sizes.crop_height or sizes.height))
        meta = {
            "source": "raw",
            "camera_white_balance": list(map(float, raw.camera_whitebalance)),
            "raw_type": str(raw.raw_type),
            "black": black,
            "saturation": saturation,
            "full_size": full_size,
            "exif": read_exif(path),
        }
    rgb = np.minimum(rgb16.astype(np.float32) / max(saturation - black, 1.0), 1.0)
    return LinearImage(rgb=rgb, camera_matrix=matrix, meta=meta)


def _camera_saturation(raw) -> int | None:
    levels = getattr(raw, "camera_white_level_per_channel", None)
    valid = [int(v) for v in (levels if levels is not None else []) if v and int(v) > 0]
    return min(valid) if valid else None


def _crop_margins(rgb: np.ndarray, sizes, half_size: bool) -> np.ndarray:
    """Drop the masked border LibRaw leaves around some sensors (8 px on some Nikon NEFs)."""
    try:
        left, top = int(sizes.crop_left_margin), int(sizes.crop_top_margin)
        width, height = int(sizes.crop_width), int(sizes.crop_height)
    except AttributeError:
        return rgb
    if width <= 0 or height <= 0:
        return rgb
    step = 2 if half_size else 1
    y0, x0 = top // step, left // step
    y1, x1 = y0 + height // step, x0 + width // step
    if y1 > rgb.shape[0] or x1 > rgb.shape[1]:
        return rgb
    return rgb[y0:y1, x0:x1]


def _load_tiff(path: Path) -> LinearImage:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        data = page.asarray()
        description = page.description or ""
    linear = LINEAR_TAG in description
    meta: dict = {"source": "tiff"}
    matrix = None
    if linear:
        try:
            info = json.loads(description)
            meta.update(info.get("meta", {}))
            if info.get("camera_matrix"):
                matrix = np.asarray(info["camera_matrix"], dtype=np.float64)
        except json.JSONDecodeError:
            pass
    rgb = _normalise(data)
    if not linear:
        rgb = srgb_decode(rgb)
    return LinearImage(rgb=rgb, camera_matrix=matrix, meta=meta)


def _load_qt_image(path: Path) -> LinearImage:
    from PySide6.QtGui import QImage

    image = QImage(str(path))
    if image.isNull():
        raise ValueError(f"No se pudo leer {path.name}")
    w, h = image.width(), image.height()
    if image.depth() > 32 or image.format() == QImage.Format.Format_Grayscale16:
        # 16-bit PNG scans: keep all 16 bits, the density step needs them in
        # the dense (dark) parts of the negative.
        image = image.convertToFormat(QImage.Format.Format_RGBX64)
        buf = np.frombuffer(image.constBits(), dtype=np.uint16, count=image.sizeInBytes() // 2)
        rgb = buf.reshape(h, image.bytesPerLine() // 2)[:, : w * 4].reshape(h, w, 4)[..., :3]
        return LinearImage(rgb=srgb_decode(rgb.astype(np.float32) / 65535.0), meta={"source": "image"})
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    buf = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    rgb = buf.reshape(h, image.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
    return LinearImage(rgb=srgb_decode(rgb.astype(np.float32) / 255.0), meta={"source": "image"})


def decode_jpeg_preview(data: bytes) -> np.ndarray | None:
    """Camera live-view JPEG to linear float RGB (for the live inversion)."""
    from PySide6.QtGui import QImage

    image = QImage.fromData(data)
    if image.isNull():
        return None
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    buf = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    rgb = buf.reshape(h, image.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
    return srgb_decode(rgb.astype(np.float32) / 255.0)


def _normalise(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=-1)
    data = data[..., :3]
    if data.dtype == np.uint8:
        return data.astype(np.float32) / 255.0
    if data.dtype == np.uint16:
        return data.astype(np.float32) / 65535.0
    return np.clip(data.astype(np.float32), 0.0, None)


def save_linear_tiff(path: str | Path, rgb: np.ndarray, camera_matrix: np.ndarray | None = None, meta: dict | None = None) -> Path:
    """Write linear camera RGB losslessly, tagged so it is read back linear."""
    import tifffile

    path = Path(path)
    description = json.dumps(
        {
            "tag": LINEAR_TAG,
            "camera_matrix": None if camera_matrix is None else np.asarray(camera_matrix).tolist(),
            "meta": meta or {},
        }
    )
    data = (np.clip(rgb, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    tifffile.imwrite(str(path), data, photometric="rgb", description=description, metadata=None, compression="zlib")
    return path
