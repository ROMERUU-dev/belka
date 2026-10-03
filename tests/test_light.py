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
