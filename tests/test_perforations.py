"""Sprocket holes as the strip's ruler: frame, film and lighting in real copy-stand views."""

import cv2
import numpy as np
import pytest

from belka.core import pipeline as pl
from belka.core import synth
from tests import dx_synth as ds


@pytest.fixture(scope="module")
def portra(library):
    return library.get("kodak-portra-400")


def _small_light_and_paper(strip: ds.Strip, paper: bool = True) -> tuple[np.ndarray, dict]:
    """As on the copy stand that broke 0.4: the lit zone smaller than the frame
    (holes and rebate only get spill), and paper over the neighbouring frames
    lit from the front by the room during a long exposure."""
    img, geo = ds.render(strip)
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = geo["frame"]
    lit = np.full((h, w), 0.12, np.float32)
    lit[int((y0 + 0.02) * h):int((y1 - 0.02) * h), int((x0 + 0.06) * w):int((x1 - 0.06) * w)] = 1.0
    img = img * cv2.GaussianBlur(lit, (0, 0), 0.02 * w)[..., None]
    if paper:
        face = np.array([0.6, 0.55, 0.4], np.float32) * float(np.percentile(img, 99.5))
        img[:, : max(int((x0 - 0.04) * w), 0)] = face
        img[:, int((x1 + 0.04) * w):] = face
    return img, geo


@pytest.mark.parametrize("angle", [0, 90, 3])
def test_both_rows_are_found_at_the_135_pitch(angle):
    strip = ds.Strip(angle=angle)
    img, _ = ds.render(strip)
    small = pl.downsample(img.astype(np.float32), pl.ANALYSIS_SIDE)
    scale = small.shape[1] / img.shape[1]
    perfs = pl.find_perforations(small)
    assert perfs is not None and perfs.vertical == (angle == 90)
    assert perfs.pitch == pytest.approx(pl.PERF_PITCH_MM * strip.px_per_mm * scale, rel=0.03)
    assert (perfs.rows[1] - perfs.rows[0]) / perfs.pitch == pytest.approx(29.0 / 4.75, rel=0.06)
    assert perfs.holes.any() and perfs.rebate.any() and not (perfs.holes & perfs.rebate).any()


def test_views_without_135_holes_have_none(portra):
    assert pl.find_perforations(synth.landscape_scene(600, 400, seed=2)) is None
    raw, _ = synth.backlit_scan(portra, rng=np.random.default_rng(3))
    assert pl.find_perforations(pl.downsample(raw, pl.ANALYSIS_SIDE)) is None


def test_tight_framing_under_a_small_light_finds_the_frame_beside_the_paper(portra):
    img, geo = _small_light_and_paper(ds.Strip(view_mm=(50.0, 33.0), shift_mm=-4.0))
    analysis = pl.analyze(img, pl.DevelopSettings(profile_id=portra.id), portra)
    frame = analysis.extra["frame"]
    assert frame is not None and np.abs(np.asarray(frame) - np.asarray(geo["frame"])).max() < 0.03
    # ...and says why the inversion cannot be trusted.
    assert analysis.extra["holes_unlit"]
    # The paper is not film: the base is not taken from it.
    masks = pl.light_masks(pl.downsample(img, pl.ANALYSIS_SIDE), portra)
    w = masks["film"].shape[1]
    assert not masks["film"][:, : int((geo["frame"][0] - 0.06) * w)].any()


def test_a_lit_strip_is_not_flagged(portra):
    img, geo = ds.render(ds.Strip(view_mm=(50.0, 33.0), shift_mm=-4.0))
    analysis = pl.analyze(img.astype(np.float32), pl.DevelopSettings(profile_id=portra.id), portra)
    assert not analysis.extra["holes_unlit"]
    assert np.abs(np.asarray(analysis.extra["frame"]) - np.asarray(geo["frame"])).max() < 0.03


def test_exposure_advice_measures_this_shot_even_with_a_fixed_base(portra):
    """A base locked for the roll was measured on another shot: the advice must not use it."""
    img, _ = ds.render(ds.Strip(view_mm=(60.0, 44.0)))
    img = img.astype(np.float32)
    auto = pl.analyze(img, pl.DevelopSettings(profile_id=portra.id), portra)
    fixed = pl.analyze(img, pl.DevelopSettings(profile_id=portra.id, base=(0.03, 0.03, 0.02)), portra)
    assert fixed.extra["base_level"] == pytest.approx(auto.extra["base_level"], rel=1e-3)
    assert pl.exposure_advice(fixed) == pl.exposure_advice(auto)
