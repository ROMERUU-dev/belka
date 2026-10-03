"""The dust shot: an extra dark-field exposure after each frame, a few stops slower.

Driven end to end through the camera thread with the fake camera; the light
panel is replaced by callables that record what the flow asks of it.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from belka.camera.base import CameraError
from belka.core.rawio import load_linear
from belka.core.session import Session, load_frame
from belka.light.geometry import LightSettings, display_value
from tests.fakes import FAKE, FakeBackend, install


def _wait(condition, timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("timed out")


def _glow(dark: np.ndarray, truth: np.ndarray) -> float:
    """How much brighter the dirt is than the film right around it."""
    def grow(mask, size):
        return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)

    near = grow(truth, 9) & ~grow(truth, 3)
    return float(dark[truth].mean() / dark[near].mean())


class Camera(FakeBackend):
    """The fake camera, able to fail or dawdle on one kind of shot."""

    fail = ""  # pattern whose capture raises
    slow = ""  # pattern whose capture takes a while
    no_shutter = False

    def capture(self, dest_dir, basename, hint=None):
        if hint is not None and hint.pattern == self.slow:
            time.sleep(0.4)
        if hint is not None and hint.pattern == self.fail:
            self.shutters.append(str(self._settings["shutter"].value))
            raise CameraError("el obturador se atascó")
        return super().capture(dest_dir, basename, hint)

    def settings(self):
        return [s for s in super().settings() if not (self.no_shutter and s.key == "shutter")]


class Rig:
    def __init__(self, tmp_path: Path, monkeypatch):
        from belka.camera import worker
        from belka.camera.worker import CameraController
        from belka.ui.capture_flow import CaptureFlow

        install(monkeypatch)
        self.cameras: list[Camera] = []
        monkeypatch.setitem(worker.BACKENDS, "fake", lambda info: self.cameras.append(Camera(info)) or self.cameras[-1])
        self.controller = CameraController()
        self.light = LightSettings(settle_ms=0, mode="tint", tint=(0.4, 0.7, 1.0), brightness=0.8)
        self.session = Session.create(tmp_path / "rollos", "prueba")
        self.visible = True
        self.stops = 3
        self.patterns: list[str] = []
        self.colors: list[tuple] = []
        self.flow = CaptureFlow(
            self.controller, self.light, lambda: self.session, self.colors.append, lambda: self.visible,
            lambda _capturing: None, set_pattern=self._pattern, darkfield_stops=lambda: self.stops,
        )
        self.frames, self.errors, self.messages = [], [], []
        self.flow.frameCaptured.connect(lambda frame, _session: self.frames.append(frame))
        self.flow.error.connect(self.errors.append)
        self.flow.message.connect(self.messages.append)
        self.on_pattern = None
        self.controller.request_open.emit(FAKE)
        _wait(lambda: self.controller.connected and self.flow._camera_settings)

    @property
    def camera(self) -> Camera:
        return self.cameras[-1]

    def _pattern(self, pattern: str) -> None:
        self.patterns.append(pattern)
        if self.on_pattern is not None:
            self.on_pattern(pattern)

    def shoot(self) -> None:
        assert self.flow.capture_frame()
        _wait(lambda: not self.flow.busy)

    def shutter_settles_at(self, speed: str) -> None:
        """The camera thread has applied every queued change and the flow saw the result."""
        _wait(lambda: self.camera._settings["shutter"].value == speed
              and str(self.flow._camera_settings["shutter"].value) == speed)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    yield rig
    rig.controller.shutdown()


def test_frame_gets_a_darkfield_shot_three_stops_slower(rig):
    rig.shoot()
    assert not rig.errors
    hints = rig.camera.captures
    assert [h.pattern for h in hints] == ["normal", "darkfield"]
    assert [(h.part, h.parts) for h in hints] == [(0, 2), (1, 2)]
    assert rig.camera.shutters == ["1/30", "1/4"]  # 1/30 × 2³ ≈ 0.27 s: the closest the camera has
    rig.shutter_settles_at("1/30")
    assert rig.patterns == ["normal", "darkfield", "normal"]
    # The ring shines with the frame's brightness and tint.
    tinted = tuple(display_value(0.8 * v) for v in (0.4, 0.7, 1.0))
    assert hints[1].light == pytest.approx(tinted) and hints[0].light == pytest.approx(rig.light.display_rgb())

    frame = rig.frames[0]
    assert frame.files == ["raw/prueba_001.tif"] and frame.darkfield == "raw/prueba_001_df.tif"
    assert Session.load(rig.session.path).frame(frame.id).darkfield == frame.darkfield
    # The dark-field file is the camera's raw: dust glowing over a dim film, the size of the frame.
    dark = load_linear(rig.session.resolve(frame.darkfield)).rgb
    assert np.median(dark) < 0.1 and _glow(dark, rig.camera.darkfield_truth) > 1.4
    assert dark.shape == load_frame(rig.session, frame).rgb.shape


def test_darkfield_follows_the_rgb_parts_and_the_chosen_stops(rig):
    rig.light.mode = "rgb"
    rig.stops = 1
    rig.shoot()
    hints = rig.camera.captures
    assert [h.pattern for h in hints] == ["normal"] * 3 + ["darkfield"]
    assert [h.part for h in hints] == [0, 1, 2, 3] and {h.parts for h in hints} == {4}
    assert rig.camera.shutters == ["1/30"] * 3 + ["1/15"]
    # All three primaries at their calibrated levels at once.
    assert hints[3].light == pytest.approx(tuple(display_value(0.8 * v) for v in (0.4, 0.7, 1.0)))
    frame = rig.frames[0]
    assert frame.mode == "rgb" and len(frame.files) == 3 and len(frame.lights) == 3
    assert frame.files[0].endswith("_R.tif") and frame.darkfield.endswith("_df.tif")
    assert rig.session.resolve(frame.darkfield).exists()
    rig.shutter_settles_at("1/30")


def test_failed_darkfield_keeps_the_frame_and_restores_the_camera(rig):
    rig.cameras[-1].fail = "darkfield"
    rig.shoot()
    assert rig.camera.shutters == ["1/30", "1/4"]
    rig.shutter_settles_at("1/30")
    assert rig.patterns[-1] == "normal"
    frame = rig.frames[0]
    assert frame.darkfield == "" and rig.session.resolve(frame.files[0]).exists()
    assert len(rig.errors) == 1 and "antipolvo" in rig.errors[0] and "atascó" in rig.errors[0]


def test_failed_frame_never_touches_the_shutter(rig):
    rig.cameras[-1].fail = "normal"
    rig.shoot()
    assert rig.camera.shutters == ["1/30"] and not rig.frames
    assert rig.patterns == ["normal", "normal"] and "atascó" in rig.errors[0]
    assert rig.flow.capture_frame()  # the next frame starts clean
    rig.flow.cancel()


def test_cancel_mid_darkfield_restores_after_the_shot_in_flight(rig):
    rig.cameras[-1].slow = "darkfield"
    rig.on_pattern = lambda p: p == "darkfield" and QTimer.singleShot(150, rig.flow.cancel)
    rig.shoot()
    # The restore was queued behind the slow exposure, which kept its speed.
    rig.shutter_settles_at("1/30")
    assert rig.camera.shutters == ["1/30", "1/4"]
    assert rig.patterns[-1] == "normal" and not rig.frames


def test_light_turned_off_before_the_darkfield_cancels_and_restores(rig):
    def lights_out(pattern):
        if pattern == "darkfield":
            rig.visible = False

    rig.on_pattern = lights_out
    rig.shoot()
    rig.shutter_settles_at("1/30")
    assert [h.pattern for h in rig.camera.captures] == ["normal"]
    assert rig.patterns[-1] == "normal" and not rig.frames and "panel de luz" in rig.errors[0]


def test_darkfield_is_skipped_without_the_light_panel(rig):
    rig.visible = False  # an external light source: the screen cannot draw the ring
    rig.shoot()
    assert [h.pattern for h in rig.camera.captures] == ["normal"]
    assert rig.camera.shutters == ["1/30"]
    assert rig.frames[0].darkfield == "" and any("antipolvo" in m for m in rig.messages)
    assert not rig.errors


def test_darkfield_off_takes_one_shot(rig):
    rig.stops = 0
    rig.shoot()
    assert [h.pattern for h in rig.camera.captures] == ["normal"] and rig.frames[0].darkfield == ""
    assert not rig.messages and rig.patterns == ["normal", "normal"]


def test_darkfield_without_a_shutter_control_keeps_the_speed(rig):
    rig.camera.no_shutter = True
    rig.controller.request_refresh.emit()
    _wait(lambda: "shutter" not in rig.flow._camera_settings)
    rig.shoot()
    assert rig.camera.shutters == ["1/30", "1/30"]
    assert rig.frames[0].darkfield and any("velocidad" in m for m in rig.messages)


def test_fake_darkfield_answers_the_shutter(tmp_path):
    from belka.camera.base import CaptureHint

    cam = FakeBackend()
    hint = CaptureHint(pattern="darkfield")
    base = load_linear(cam.capture(tmp_path, "a", hint)[0]).rgb
    cam.set_setting("shutterspeed2", "1/4")
    slow = load_linear(cam.capture(tmp_path, "b", hint)[0]).rgb
    dirt = cam.darkfield_truth & (slow.max(-1) < 0.95)
    assert 6 < slow[dirt].mean() / base[dirt].mean() < 9  # three stops
    assert _glow(slow, cam.darkfield_truth) > 1.4
    dimmer = load_linear(cam.capture(tmp_path, "c", CaptureHint(pattern="darkfield", light=(0.5,) * 3))[0]).rgb
    assert dimmer[dirt].mean() < 0.3 * slow[dirt].mean()  # a dimmer ring, less scattered light


# ---------------------------------------------------------------- capture panel

@pytest.fixture
def panel():
    from belka.light.geometry import load_adapters
    from belka.ui.capture_panel import CapturePanel

    p = CapturePanel(LightSettings(), load_adapters())
    yield p
    p.deleteLater()


def test_panel_darkfield_option_and_its_settings(panel):
    assert not panel.darkfield_enabled() and panel.darkfield_stops() == 3
    assert not panel.darkfield_spin.isEnabled()
    assert "polvo" in panel.darkfield_check.toolTip()
    seen = []
    panel.captureOptionsChanged.connect(seen.append)
    panel.darkfield_check.setChecked(True)
    assert panel.darkfield_spin.isEnabled()
    panel.darkfield_spin.setValue(9)
    assert panel.darkfield_stops() == 6  # 1–6 stops
    assert seen[-1] == {"darkfield": True, "darkfield_stops": 6}
    panel.set_busy(True)
    assert not panel.darkfield_check.isEnabled() and not panel.darkfield_spin.isEnabled()
    panel.set_busy(False)
    assert panel.darkfield_spin.isEnabled()
    count = len(seen)
    panel.load_capture_options({"darkfield": False, "darkfield_stops": 2})
    assert len(seen) == count  # loading does not echo back to be saved again
    assert not panel.darkfield_enabled() and panel.darkfield_stops() == 2 and not panel.darkfield_spin.isEnabled()
    panel.load_capture_options(None)
    assert panel.capture_options() == {"darkfield": False, "darkfield_stops": 3}


def test_panel_with_the_darkfield_option_fits_the_left_column(panel):
    panel.darkfield_check.setChecked(True)
    scrollbar = panel.verticalScrollBar().sizeHint().width()
    assert panel.widget().minimumSizeHint().width() + scrollbar <= 270
    assert panel.darkfield_check.sizeHint().width() <= 270 - scrollbar - 40


def test_closest_shutter_matches_in_stops():
    from belka.ui.capture_panel import closest_shutter

    speeds = ["1/250", "1/30", "1/4", "1", "Bulb"]
    assert closest_shutter(speeds, 1 / 30 * 8) == "1/4"
    assert closest_shutter(speeds, 20.0) == "1"
    assert closest_shutter(["Bulb", "Time"], 1.0) is None
