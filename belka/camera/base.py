"""Camera abstraction: what the capture worker needs from a backend.

Backends are synchronous and not thread-safe; the app drives them from a
single worker thread (see ``belka.camera.worker``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from belka.i18n import _

# Labels of the logical settings; the libgphoto2 widget names behind each
# one, per brand, are in belka/data/cameras.json.
SETTING_LABELS = {
    "iso": "ISO",
    "iso_auto": "ISO automático",
    "liveview_size": "Tamaño vista en vivo",
    "shutter": "Velocidad",
    "aperture": "Diafragma",
    "exposure_comp": "Comp. exposición",
    "quality": "Calidad",
    "target": "Guardar en",
    "whitebalance": "Balance de blancos",
    "focusmode": "Modo de enfoque",
    "program": "Modo",
    "battery": "Batería",
}


class CameraError(Exception):
    pass


class CameraBusyError(CameraError):
    """The USB device is claimed by another process (usually GNOME's gvfs)."""


@dataclass(frozen=True)
class CameraInfo:
    model: str
    port: str
    backend: str = "gphoto2"  # key in belka.camera.worker.BACKENDS


@dataclass
class CameraSetting:
    key: str
    name: str
    label: str
    kind: str  # "choice", "range", "toggle" or "text"
    value: object
    choices: list[str] = field(default_factory=list)
    range: tuple[float, float, float] | None = None
    readonly: bool = False


@dataclass
class CaptureHint:
    """What the app is about to shoot, for backends that can use it."""

    kind: str = "frame"  # "frame", "flat" or "calibration"
    light: tuple[float, float, float] = (1.0, 1.0, 1.0)
    part: int = 0  # index within an RGB-sequential frame
    parts: int = 1


class CameraBackend(ABC):
    capabilities: frozenset[str] = frozenset()

    def __init__(self, info: CameraInfo):
        self.info = info

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def capture(self, dest_dir: Path, basename: str, hint: CaptureHint | None = None) -> list[Path]:
        """Shoot one image and download every file it produced."""

    def preview(self) -> bytes | None:
        """One live-view JPEG, or None if unsupported."""
        return None

    def stop_preview(self) -> None:
        pass

    def summary(self) -> str:
        return self.info.model

    def settings(self) -> list[CameraSetting]:
        return []

    def set_setting(self, name: str, value: object) -> None:
        raise CameraError(_("Esta cámara no permite cambiar ajustes"))

    def focus_step(self, steps: int) -> None:
        raise CameraError(_("Esta cámara no permite mover el foco"))

    def autofocus(self) -> None:
        raise CameraError(_("Esta cámara no permite autoenfoque remoto"))
