"""A camera backend for tests: photographs synthetic negatives and never touches USB.

It answers its shutter speed like a real camera on a copy stand (a slower
speed gives a brighter raw), so the capture flow and the exposure advice can
be exercised end to end. A dark-field shot (``hint.pattern == "darkfield"``)
is ``synth.dusty_scan``'s: dust glowing on black over a faint film, as bright
as the shutter and the ring's brightness make it. Its dirt does not match
the bright frames (``synth.backlit_scan``); a test that needs a matching pair
takes both from ``synth.dusty_scan``. ``install`` makes the app detect and
open it.
"""

from __future__ import annotations

import functools
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
FRAME_SIZE = (845, 1100)  # (width, height) of synth.backlit_scan's view, which dark-field shots share


def _profile(film: str):
    return ProfileLibrary(user_dir=Path("/nonexistent-belka")).get(film)


@functools.lru_cache(maxsize=8)
def _darkfield(film: str, stops: float) -> tuple[np.ndarray, np.ndarray]:
    """(dark-field raw, dirt mask) ``stops`` over the reference exposure; deterministic, so cached."""
    _bright, darkfield, truth = synth.dusty_scan(_profile(film), size=FRAME_SIZE, darkfield_stops=stops, seed=7)
    return darkfield, truth


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
        self.shutters: list[str] = []  # the speed each capture was taken at
        self.darkfield_truth: np.ndarray | None = None  # where the last dark-field shot has dirt
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
        self.shutters.append(str(self._settings["shutter"].value))
        emission = tuple(float(v) ** 2.2 for v in hint.light)
        if hint.kind == "flat":
            raw = synth.photograph(np.zeros((400, 630, 3), np.float32), emission,
                                   exposure=0.5 * 2.0 ** self.camera_ev(), rng=self._rng)
        elif hint.pattern == "darkfield":
            # The ring's brightness scales the scattered light just as the shutter does.
            stops = self.camera_ev() + math.log2(max(max(emission), 1e-3))
            raw, self.darkfield_truth = _darkfield(self._film, round(stops, 2))
        else:
            raw, _truth = synth.backlit_scan(_profile(self._film), light_rgb=emission, camera_ev=self.camera_ev(),
                                             rng=self._rng)
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
