"""Automatic tone (Lightroom's Tone "Auto") on synthetic negatives."""

import time

import cv2
import numpy as np
import pytest

from belka.core import adjust, autotone, synth
from belka.core import develop as dv
from belka.core import pipeline as pl

RANGES = {"exposure": (-5.0, 5.0), "contrast": (0.4, 2.0), "highlights": (-1.0, 1.0), "shadows": (-1.0, 1.0),
          "white": (-0.3, 0.3), "black": (-0.3, 0.3)}


# ---------------------------------------------------------------- scenes

def _night(w=600, h=400):
    """Low key: dark sky and ground, a few lit windows and the moon."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    scene = np.full((h, w, 3), 0.012, np.float32) * (1 + 0.5 * yy / h)[..., None]
    scene[int(h * 0.55):] = (0.02, 0.018, 0.015)
    for _ in range(25):
        x0, y0 = int(rng.random() * w * 0.9), int(h * (0.2 + 0.3 * rng.random()))
        scene[y0:y0 + 12, x0:x0 + 8] = (0.9, 0.7, 0.35)
    scene[((xx - w * 0.8) ** 2 + (yy - h * 0.15) ** 2) < (h * 0.05) ** 2] = (1.2, 1.2, 1.1)
    return scene


def _snow(w=600, h=400):
    """High key: textured snow, a dark tree and a red jacket."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    scene = np.full((h, w, 3), 0.75, np.float32) * np.array([0.97, 0.99, 1.0], np.float32)
    scene *= (0.9 + 0.1 * np.sin(xx / 37.0) * np.sin(yy / 23.0))[..., None]
    tree = (np.abs(xx - w * 0.3) < (yy - h * 0.2) * 0.3) & (yy < h * 0.8) & (yy > h * 0.2)
    scene[tree] = (0.04, 0.07, 0.04)
    scene[((xx - w * 0.7) ** 2 + (yy - h * 0.6) ** 2) < (h * 0.06) ** 2] = (0.5, 0.06, 0.05)
    return scene


SCENES = {
    "carta": lambda: synth.colorchecker_scene(600, 400),
    "paisaje": lambda: synth.landscape_scene(600, 400),
    "noche": _night,
    "nieve": _snow,
}


class Scan:
    """A frame of a synthetic backlit strip, cropped to its detected frame like the preview."""

    def __init__(self, library, scene: str, film_ev: float = 0.0, profile_id: str = "generic-c41",
                 camera_ev: float = 0.0):
        self.profile = library.get(profile_id)
        self.raw, _ = synth.backlit_scan(self.profile, SCENES[scene](), film_ev=film_ev, camera_ev=camera_ev)
        self.settings = pl.DevelopSettings(profile_id=profile_id)
        self.analysis = pl.analyze(self.raw, self.settings, self.profile)
        self.geo = dv.geometry(self.raw, self.settings, self.analysis, profile=self.profile)

    def auto(self, settings=None, fields=None, geo=None) -> dict[str, float]:
        return autotone.auto_tone(self.geo if geo is None else geo, self.analysis, settings or self.settings,
                                  self.profile, None, False, fields=fields)

    def print(self, values: dict[str, float] | None = None, settings=None) -> np.ndarray:
        s = (settings or self.settings).copy(**(values or {}))
        return dv.finish(dv.invert(self.geo, self.analysis, s, self.profile, None, False), s, 1.0)


@pytest.fixture(scope="module")
def scan(library):
    """Scans are slow to make and never modified: share them across the module."""
    made: dict[tuple, Scan] = {}

    def make(scene: str = "carta", **kw) -> Scan:
        key = (scene, *sorted(kw.items()))
        if key not in made:
            made[key] = Scan(library, scene, **kw)
        return made[key]

    return make


def _median(rgb: np.ndarray) -> float:
    return float(np.median(_lum(rgb)))


def _lum(rgb: np.ndarray) -> np.ndarray:
    return pl.srgb_encode(pl.srgb_decode(rgb) @ pl.REC709_Y)


def _clipped(rgb: np.ndarray) -> tuple[float, float]:
    """Shares of the picture in the clipping warnings (shadows, highlights)."""
    shadows, highlights = adjust.clipping_masks(rgb)
    return float(shadows.mean()), float(highlights.mean())


# ---------------------------------------------------------------- the answer

def test_every_tone_field_comes_back_as_an_absolute_value_in_its_slider_range(scan):
    values = scan().auto()
    assert tuple(values) == autotone.TONE_FIELDS
    for field, (lo, hi) in RANGES.items():
        assert lo <= values[field] <= hi, (field, values[field])


def test_requested_fields_start_from_neutral(scan):
    """The answer is absolute: it does not depend on where the sliders were."""
    s = scan()
    moved = s.settings.copy(exposure=2.5, contrast=1.8, highlights=0.6, shadows=-0.5, white=-0.2, black=0.25)
    assert s.auto(moved) == s.auto()


def test_deterministic(scan):
    s = scan("paisaje")
    assert s.auto() == s.auto()


def test_unknown_field_is_refused(scan):
    with pytest.raises(ValueError):
        scan().auto(fields=("exposure", "saturation"))


# ---------------------------------------------------------------- the look

@pytest.mark.parametrize("scene", ["carta", "paisaje"])
def test_under_and_overexposed_negatives_print_alike(scan, scene):
    """Thin and dense negatives of an average scene come out with a middle tone
    near middle grey, much closer together than their plain prints."""
    before, after = [], []
    for film_ev in (-2.0, 0.0, 2.0):
        s = scan(scene, film_ev=film_ev)
        before.append(_median(s.print()))
        after.append(_median(s.print(s.auto())))
    assert all(0.44 <= m <= 0.67 for m in after), after
    assert np.mean(np.abs(np.array(after) - autotone.MID_GREY)) < np.mean(np.abs(np.array(before) - autotone.MID_GREY))
    assert max(after) - min(after) < max(before) - min(before)


@pytest.mark.parametrize("scene,film_ev,profile_id,camera_ev", [
    ("carta", -2.0, "generic-c41", 0.0),
    ("carta", 2.0, "kodak-portra-400", 0.0),
    ("carta", 0.0, "kodak-portra-400", -2.0),
    ("paisaje", 0.0, "generic-c41", 0.0),
    ("paisaje", 2.0, "ilford-hp5-plus", 0.0),
    ("nieve", 0.0, "generic-c41", 0.0),
    ("noche", 0.0, "generic-c41", 0.0),
])
def test_nothing_clips_past_the_extreme_tenth_of_a_percent(scan, scene, film_ev, profile_id, camera_ev):
    s = scan(scene, film_ev=film_ev, profile_id=profile_id, camera_ev=camera_ev)
    shadows, highlights = _clipped(s.print(s.auto()))
    assert shadows < 0.001 and highlights < 0.001, (shadows, highlights)


def test_whites_reach_the_edge_of_clipping(scan):
    s = scan("paisaje")
    out = s.print(s.auto())
    top = np.percentile(out.max(axis=-1), 99.9)
    assert autotone.WHITE_LEVEL - 0.02 < top < 254.5 / 255, top


def test_a_flat_negative_gets_deeper_blacks_and_more_contrast(scan):
    s = scan("carta", film_ev=-2.0)
    values = s.auto()
    assert values["contrast"] > 1.0 and values["black"] > 0.0
    assert np.percentile(_lum(s.print(values)), 0.5) < np.percentile(_lum(s.print()), 0.5) - 0.05


def test_a_night_stays_dark_and_snow_stays_light(scan):
    """Low- and high-key scenes keep their key instead of being dragged to grey."""
    night, snow = scan("noche"), scan("nieve")
    night_mid = _median(night.print(night.auto()))
    snow_mid = _median(snow.print(snow.auto()))
    assert night_mid < 0.2, night_mid
    assert snow_mid > 0.85, snow_mid


def test_black_and_white_film(scan):
    s = scan("paisaje", profile_id="ilford-hp5-plus")
    assert s.analysis.channels == 1
    out = s.print(s.auto())
    assert 0.44 <= _median(out) <= 0.67
    assert max(_clipped(out)) < 0.001


def test_slide_film(scan):
    s = scan("carta", profile_id="fujifilm-provia-100f")
    values = s.auto()
    out = s.print(values)
    assert 0.4 <= _median(out) <= 0.67
    # The grey ramp's end (0.3 % of the frame) is clear film, saturated by the
    # slide itself: at full resolution its grain may poke past the warning.
    assert max(_clipped(out)) < 0.002


def test_rebate_and_sprocket_holes_in_the_crop(scan):
    """A crop wider than the frame, with rebate and holes, as for a contact-sheet look:
    the holes are bare light (not picture), and the frame itself still prints well."""
    s = scan("carta")
    x0, y0, x1, y1 = s.analysis.extra["frame"]
    wide = s.settings.copy(crop=(x0 - 0.04, y0 - 0.12, x1 + 0.04, y1 + 0.12))
    geo = dv.geometry(s.raw, wide, s.analysis, profile=s.profile)
    values = autotone.auto_tone(geo, s.analysis, wide, s.profile, None, False)
    out = dv.finish(dv.invert(geo, s.analysis, wide.copy(**values), s.profile, None, False), wide.copy(**values), 1.0)
    h, w = out.shape[:2]
    fx0, fy0 = int(0.04 / (x1 - x0 + 0.08) * w) + 4, int(0.12 / (y1 - y0 + 0.24) * h) + 4
    film = out[fy0:h - fy0, fx0:w - fx0]
    assert 0.44 <= _median(film) <= 0.67
    assert max(_clipped(film)) < 0.001


def test_the_saturation_slider_does_not_change_the_tone(scan):
    """Stronger colours can push one channel past white with no tone blown."""
    s = scan("paisaje")
    assert s.auto(s.settings.copy(saturation=1.7)) == s.auto()


def test_local_adjustments_are_measured_as_they_act(scan):
    """Clarity is not a tone curve: the fit is checked on the real chain."""
    s = scan("paisaje")
    settings = s.settings.copy(clarity=0.8, texture=0.5)
    out = s.print(s.auto(settings), settings)
    assert max(_clipped(out)) < 0.002


# ---------------------------------------------------------------- one field at a time

@pytest.mark.parametrize("field", autotone.TONE_FIELDS)
def test_one_field_leaves_the_others_alone(scan, field):
    s = scan("paisaje")
    current = s.settings.copy(exposure=0.4, contrast=1.2, highlights=-0.3, shadows=0.2, white=0.05, black=-0.05)
    values = s.auto(current, fields=(field,))
    assert list(values) == [field]
    lo, hi = RANGES[field]
    assert lo <= values[field] <= hi


def test_auto_white_takes_the_brightest_to_the_edge_whatever_the_rest(scan):
    s = scan("carta")
    for exposure in (-0.6, 0.0, 0.6):
        current = s.settings.copy(exposure=exposure)
        white = s.auto(current, fields=("white",))["white"]
        top = np.percentile(s.print({"white": white}, current).max(axis=-1), 99.9)
        assert abs(top - autotone.WHITE_LEVEL) < 0.012, (exposure, white, top)


def test_auto_exposure_alone_moves_the_middle_tone(scan):
    s = scan("carta", film_ev=2.0)
    ev = s.auto(fields=("exposure",))["exposure"]
    assert ev < 0.0
    assert abs(_median(s.print({"exposure": ev})) - autotone.MID_GREY) < abs(_median(s.print()) - autotone.MID_GREY)


# ---------------------------------------------------------------- stability

@pytest.mark.parametrize("scene", ["carta", "paisaje", "nieve"])
def test_auto_on_an_auto_picture_changes_nothing(scan, scene):
    s = scan(scene)
    values = s.auto()
    again = s.auto(s.settings.copy(**values))
    assert again == values
    # Each slider's own Auto (Shift + double click) barely moves it either.
    # Blacks alone moves most: the full Auto shares the stretch with Whites
    # and makes up the darker middle with Exposure, Blacks alone does neither.
    tuned = s.settings.copy(**values)
    tolerance = {"exposure": 0.12, "contrast": 0.08, "highlights": 0.15, "shadows": 0.15, "white": 0.03,
                 "black": 0.06}
    for field in autotone.TONE_FIELDS:
        single = s.auto(tuned, fields=(field,))[field]
        assert abs(single - values[field]) <= tolerance[field], (field, values[field], single)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (1, 300), (300, 1)])
def test_tiny_pictures(scan, shape):
    s = scan()
    h, w = shape
    y, x = s.geo.shape[0] // 3, s.geo.shape[1] // 4
    values = s.auto(geo=np.ascontiguousarray(s.geo[y:y + h, x:x + w]))
    for field, (lo, hi) in RANGES.items():
        assert np.isfinite(values[field]) and lo <= values[field] <= hi, (field, values[field])


def test_a_flat_picture_only_gets_its_exposure(scan):
    s = scan()
    patch = np.empty((60, 90, 3), np.float32)
    patch[:] = s.geo[s.geo.shape[0] // 3, s.geo.shape[1] // 4]
    values = s.auto(geo=patch)
    assert values["contrast"] == pytest.approx(1.0, abs=0.02)
    assert abs(values["white"]) < 0.01 and abs(values["highlights"]) < 0.02 and abs(values["shadows"]) < 0.02
    out = dv.finish(dv.invert(patch, s.analysis, s.settings.copy(**values), s.profile, None, False),
                    s.settings.copy(**values), 1.0)
    assert abs(_median(out) - autotone.MID_GREY) < 0.03


def test_nothing_to_measure_leaves_neutral_values(scan):
    s = scan()
    assert s.auto(geo=s.geo[:0]) == autotone.NEUTRAL
    unlit = pl.Analysis(base=s.analysis.base, base_is_auto=True, lo=s.analysis.lo, hi=s.analysis.hi,
                        extra={**s.analysis.extra, "no_light": True})
    values = autotone.auto_tone(s.geo, unlit, s.settings, s.profile, None, False, fields=("exposure", "white"))
    assert values == {"exposure": 0.0, "white": 0.0}


# ---------------------------------------------------------------- speed

def test_speed_on_a_preview_sized_frame(scan):
    s = scan("paisaje")
    geo = cv2.resize(s.geo, (1800, 1200), interpolation=cv2.INTER_LINEAR)
    s.auto(geo=geo)  # first-call caches (paper tables, highlights/shadows curves)

    def timed(i: int) -> float:
        settings = s.settings.copy(temperature=0.01 * i)  # a different picture each time
        t0 = time.perf_counter()
        s.auto(settings, geo=geo)
        return time.perf_counter() - t0

    best = min(timed(i) for i in range(4))
    assert best < 0.15, f"{best * 1000:.0f} ms"
