import numpy as np
import pytest

from belka.core import synth
from belka.light.geometry import LightSettings, display_value, load_adapters, tint_for_base


def test_adapters_load():
    adapters = {a.id: a for a in load_adapters()}
    assert adapters["35mm"].frame_mm == (36, 24)
    assert adapters["free"].light_mm is None
    for a in adapters.values():
        if a.light_mm and a.frame_mm:
            assert a.light_mm[0] >= a.frame_mm[0] and a.light_mm[1] >= a.frame_mm[1]


def test_light_modes():
    light = LightSettings(brightness=0.5, tint=(0.3, 0.4, 1.0))
    assert light.emission() == (0.5, 0.5, 0.5)
    assert light.display_rgb()[0] == pytest.approx(0.5 ** (1 / 2.2))
    light.mode = "tint"
    assert light.emission() == pytest.approx((0.15, 0.2, 0.5))
    light.mode = "rgb"
    assert light.capture_lights()[1] == pytest.approx((0.0, display_value(0.2), 0.0))
    assert len(light.capture_lights()) == 3


def test_light_settings_json():
    light = LightSettings(rect_mm=(10, 20, 56, 40), mode="tint", tint=(0.2, 0.3, 1.0), scale_correction={"x": 1.01})
    assert LightSettings.from_json(light.to_json()) == light
    assert LightSettings.from_json({"mode": "laser"}).mode == "white"


def test_tint_calibration_balances_the_film_base():
    """Iterating the calibration through the simulated screen and camera converges."""
    density = np.asarray([0.23, 0.63, 0.88], np.float32) @ synth.CAMERA_DYE_CROSSTALK.T
    transmittance = 10.0 ** -density

    def measure(emission):
        return synth.SCREEN_TO_CAMERA @ np.asarray(emission) * transmittance

    white = measure((1, 1, 1))
    assert white.max() / white.min() > 2.5  # the orange mask under white light
    emission = (1.0, 1.0, 1.0)
    for _ in range(4):
        emission = tint_for_base(measure(emission), emission)
    balanced = measure(emission)
    assert balanced.max() / balanced.min() < 1.1
    assert max(emission) == 1.0 and emission[2] == 1.0


# ---------------------------------------------------------------- dark-field pattern

def _render(panel) -> np.ndarray:
    from PySide6.QtGui import QImage

    image = panel.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    return np.frombuffer(image.constBits(), np.uint8).reshape(h, image.bytesPerLine())[:, :w * 3].reshape(h, w, 3).copy()


@pytest.fixture
def light_panel():
    from belka.light.panel import LightPanel

    adapters = {a.id: a for a in load_adapters()}
    panel = LightPanel(LightSettings(), adapters["35mm"])
    panel.resize(900, 700)
    panel.set_display_color((1.0, 0.8, 0.6))
    yield panel
    panel.deleteLater()


def test_darkfield_darkens_the_film_and_lights_a_ring_around_it(light_panel):
    panel = light_panel
    panel.set_capturing(True)  # as the camera sees it: no HUD, guides or marks
    normal = _render(panel)
    light = panel.light_rect()
    ppm = panel.ppm()
    panel.set_pattern("darkfield")
    assert panel.pattern == "darkfield"
    image = _render(panel)
    film, outer = panel.darkfield_rects()
    assert film == light
    l, t, r, b = (int(round(v)) for v in (light.left(), light.top(), light.right(), light.bottom()))
    assert normal[t + 2:b - 2, l + 2:r - 2].min() > 150
    assert image[t - 1:b + 1, l - 1:r + 1].max() == 0  # the film area and a little more are black
    # Across the middle: 2 mm dark, then a 15 mm ring at the panel's colour, then black.
    for row, start, step in ((image[int(light.center().y())], l, -1), (image[int(light.center().y())], r, 1)):
        lit = np.flatnonzero(row[:, 0] > 0)
        side = lit[lit < start] if step < 0 else lit[lit > start]
        assert abs(len(side) / ppm - 15.0) < 0.6
        assert abs(abs(start - (side.max() if step < 0 else side.min())) / ppm - 2.0) < 0.6
        assert tuple(row[side[len(side) // 2]]) == (255, 204, 153)
    column = image[:, int(light.center().x())]
    lit = np.flatnonzero(column[:, 0] > 0)
    assert abs(len(lit) / ppm - 30.0) < 1.2  # above and below
    assert image[: int(outer.top()) - 1].max() == 0 and image[int(outer.bottom()) + 1:].max() == 0
    panel.set_pattern("normal")
    assert np.array_equal(_render(panel), normal)


def test_darkfield_ring_width_and_margin_are_configurable(light_panel):
    panel = light_panel
    panel.set_capturing(True)
    panel.set_pattern("darkfield", ring_mm=8.0, margin_mm=4.0)
    image = _render(panel)
    light, ppm = panel.light_rect(), panel.ppm()
    row = image[int(light.center().y())]
    lit = np.flatnonzero(row[:, 0] > 0)
    left = lit[lit < light.left()]
    assert abs(len(left) / ppm - 8.0) < 0.6 and abs((light.left() - left.max()) / ppm - 4.0) < 0.6
    with pytest.raises(ValueError):
        panel.set_pattern("strobe")


def test_darkfield_in_a_window_puts_the_ring_on_its_border(light_panel):
    panel = light_panel
    panel.settings.fullscreen = False  # the whole window is the light: no room around it
    panel.set_capturing(True)
    panel.set_pattern("darkfield")
    image = _render(panel)
    film, outer = panel.darkfield_rects()
    assert outer == panel.rect().toRectF() and film.width() > 0
    ring = (15.0 + 2.0) * panel.ppm()
    assert film.left() == pytest.approx(ring) and film.bottom() == pytest.approx(panel.height() - ring)
    assert image[2:8, 2:-2].min() > 100  # the border is lit
    c = film.center()
    assert image[int(c.y()) - 20:int(c.y()) + 20, int(c.x()) - 20:int(c.x()) + 20].max() == 0


def test_hud_stays_clear_of_the_darkfield_ring(light_panel):
    panel = light_panel
    panel.resize(1600, 1000)
    panel.set_pattern("darkfield")
    panel.set_capturing(True)
    _render(panel)
    assert panel._buttons == {}  # hidden while exposing, as in the normal pattern
    panel.set_capturing(False)
    _render(panel)
    _film, outer = panel.darkfield_rects()
    assert panel._buttons and not any(r.intersects(outer) for r in panel._buttons.values())
