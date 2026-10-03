"""A camera backend for tests: photographs synthetic negatives and never touches USB.

It answers its shutter speed like a real camera on a copy stand (a slower
speed gives a brighter raw), so the capture flow and the exposure advice can
be exercised end to end. ``install`` makes the app detect and open it.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from belka.camera.base import CameraBackend, CameraError, CameraInfo, CameraSetting, CaptureHint
from belka.core import synth
from belka.core.film import ProfileLibrary
from belka.core.pipeline import srgb_encode, to_uint8
from belka.core.rawio import save_linear_tiff

FAKE = CameraInfo(model="Fake Camera", port="usb:fake", backend="fake")
SHUTTERS = ["1/250", "1/125", "1/60", "1/30", "1/15", "1/8", "1/4", "1/2", "1"]
REFERENCE_SHUTTER = "1/30"  # puts the film base just under clipping


def _seconds(text: str) -> float:
    num, _slash, den = text.partition("/")
    return float(num) / float(den or 1)


class FakeBackend(CameraBackend):
    capabilities = frozenset({"preview", "settings", "focus"})

    def __init__(self, info: CameraInfo = FAKE, shutter: str = REFERENCE_SHUTTER, film: str = "kodak-portra-400"):
        super().__init__(info)
        self._settings = {
            "shutter": CameraSetting("shutter", "shutterspeed2", "Velocidad", "choice", shutter, list(SHUTTERS)),
            "iso": CameraSetting("iso", "iso", "ISO", "choice", "100", ["100", "200", "400"]),
            "aperture": CameraSetting("aperture", "f-number", "Diafragma", "choice", "f/8", ["f/5.6", "f/8", "f/11"]),
            "quality": CameraSetting("quality", "imagequality", "Calidad", "choice", "NEF (Raw)", ["NEF (Raw)"]),
        }
        self._film = film
        self.opened = False
        self.focus = 0
        self.captures: list[CaptureHint] = []
        self._rng = np.random.default_rng(7)

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def settings(self) -> list[CameraSetting]:
        return list(self._settings.values())

    def set_setting(self, name: str, value: object) -> None:
        setting = next((s for s in self._settings.values() if s.name == name), None)
        if setting is None:
            raise CameraError(f"no such setting: {name}")
        setting.value = str(value)

    def focus_step(self, steps: int) -> None:
        self.focus += steps

    def autofocus(self) -> None:
        self.focus = 0

    def camera_ev(self) -> float:
        shutter = _seconds(str(self._settings["shutter"].value)) / _seconds(REFERENCE_SHUTTER)
        return math.log2(shutter * float(self._settings["iso"].value) / 100.0)

    def capture(self, dest_dir: Path, basename: str, hint: CaptureHint | None = None) -> list[Path]:
        hint = hint or CaptureHint()
        self.captures.append(hint)
        emission = tuple(float(v) ** 2.2 for v in hint.light)
        if hint.kind == "flat":
            raw = synth.photograph(np.zeros((400, 630, 3), np.float32), emission,
                                   exposure=0.5 * 2.0 ** self.camera_ev(), rng=self._rng)
        else:
            profile = ProfileLibrary(user_dir=Path("/nonexistent-belka")).get(self._film)
            raw, _truth = synth.backlit_scan(profile, light_rgb=emission, camera_ev=self.camera_ev(), rng=self._rng)
        dest_dir.mkdir(parents=True, exist_ok=True)
        return [save_linear_tiff(dest_dir / f"{basename}.tif", raw, meta={"camera": self.info.model})]

    def preview(self) -> bytes | None:
        image = to_uint8(srgb_encode(np.full((120, 180, 3), 0.4, np.float32)))
        ok, jpeg = cv2.imencode(".jpg", image[..., ::-1])
        return bytes(jpeg) if ok else None


def install(monkeypatch, cameras: tuple[CameraInfo, ...] = (FAKE,)) -> None:
    """Make camera detection find ``cameras`` and open them with FakeBackend."""
    from belka.camera import gphoto, worker

    monkeypatch.setitem(worker.BACKENDS, "fake", FakeBackend)
    monkeypatch.setattr(gphoto, "detect", lambda: list(cameras))
