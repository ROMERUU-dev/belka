import os
import threading
import time

import cv2
import numpy as np
import pytest

from belka.core import adjust, synth
from belka.core import pipeline as pl
from belka.core.pipeline import DevelopSettings

# Ottosson's OKLab, written out again so the tests check the module against
# an independent conversion.
M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])


def oklab(rgb):
    lin = pl.srgb_decode(np.asarray(rgb, dtype=np.float32)).astype(np.float64)
    return np.cbrt(lin @ M1.T) @ M2.T


def oklch_to_srgb(L, C, hue):
    h = np.radians(hue)
    lab = np.array([L, C * np.cos(h), C * np.sin(h)])
    lin = (np.linalg.inv(M2) @ lab) ** 3 @ np.linalg.inv(M1).T
    assert lin.min() > -1e-4 and lin.max() < 1 + 1e-4, "test colour out of gamut"
    return pl.srgb_encode(lin.astype(np.float32))


def chroma_hue(rgb):
    lab = oklab(rgb)
    return np.hypot(lab[..., 1], lab[..., 2]), np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360


def swatches(colours, size=40):
    """A row of flat patches; ``patch(img, i)`` reads back the middle of one."""
    img = np.zeros((size, size * len(colours), 3), dtype=np.float32)
    for i, c in enumerate(colours):
        img[:, i * size:(i + 1) * size] = c
    return img


def patch(img, i, size=40):
    m = size // 4
    return img[m:size - m, i * size + m:(i + 1) * size - m].reshape(-1, 3).mean(0)


@pytest.fixture(scope="module")
def landscape():
    rng = np.random.default_rng(7)
    img = pl.srgb_encode(synth.landscape_scene(900, 600, seed=2) * 0.9)
    return np.clip(img + rng.normal(0, 0.01, img.shape).astype(np.float32), 0, 1)


SKY = (slice(20, 120), slice(500, 880))  # bright, smooth
GRASS = (slice(470, 590), slice(560, 880))  # dark, textured


def light(img, region=None):
    lum = img.mean(-1)
    return lum if region is None else lum[region]


def everything(**changes):
    s = DevelopSettings(
        highlights=-0.5, shadows=0.4, texture=0.3, clarity=0.3, vibrance=0.3,
        curve_shadows=-0.2, curve_lights=0.2, curve_rgb=((0, 0), (0.5, 0.55), (1, 1)), curve_blue=((0, 0), (0.5, 0.47), (1, 1)),
        hsl_hue=(0.2,) * 8, hsl_sat=(0.1, -0.2, 0, 0, 0, 0.3, 0, 0), hsl_lum=(0, 0, 0, 0, 0, -0.4, 0, 0),
        grade_shadows=(220, 0.4, 0.1), grade_midtones=(60, 0.1, 0), grade_highlights=(40, 0.3, 0), grade_global=(0, 0.1, 0.05),
        sharpen_amount=0.6, sharpen_masking=0.3, nr_luma=0.4, nr_color=0.4,
        vignette_amount=-0.4, grain_amount=0.3,
    )
    return s.copy(**changes)


# ---------------------------------------------------------------- basics

def test_neutral_settings_return_the_input_object(landscape):
    s = DevelopSettings()
    assert adjust.is_identity(s)
    assert adjust.apply_adjustments(landscape, s) is landscape


def test_controls_only_in_disabled_sections_are_identity():
    s = DevelopSettings(curve_darks=0.5, hsl_sat=(1.0,) * 8, grade_global=(30, 0.5, 0), sharpen_amount=1,
                        grain_amount=1, disabled=("curve", "hsl", "grading", "detail", "effects"))
    assert adjust.is_identity(s)
    # Values that do nothing by themselves are still identity.
    assert adjust.is_identity(DevelopSettings(curve_splits=(0.3, 0.5, 0.7), grade_shadows=(200, 0, 0), grade_balance=0.4))
    assert not adjust.is_identity(DevelopSettings(vibrance=0.1))
    assert not adjust.is_identity(DevelopSettings(curve_red=((0, 0), (0.5, 0.6), (1, 1))))


def test_input_is_never_modified_and_output_is_well_formed(landscape):
    before = landscape.copy()
    out = adjust.apply_adjustments(landscape, everything(), scale=0.5, seed=3)
    np.testing.assert_array_equal(landscape, before)
    assert out.dtype == np.float32 and out.shape == landscape.shape
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_colour_round_trip_is_lossless(landscape):
    # A vignette this faint changes nothing, so only the trip through OKLab
    # and back is left.
    out = adjust.apply_adjustments(landscape, DevelopSettings(vignette_amount=-1e-5))
    assert np.abs(out - landscape).max() < 2e-4


def test_same_call_gives_the_same_pixels(landscape):
    s = everything()
    np.testing.assert_array_equal(adjust.apply_adjustments(landscape, s, seed=5),
                                  adjust.apply_adjustments(landscape, s, seed=5))


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (200, 7), (2, 33000)])
def test_odd_shapes(shape):
    rng = np.random.default_rng(1)
    img = rng.random((*shape, 3), dtype=np.float32)
    out = adjust.apply_adjustments(img, everything(), scale=1.0)
    assert out.shape == img.shape and np.isfinite(out).all()
    # The curve-only path has its own strip loop.
    curve = adjust.apply_adjustments(img, DevelopSettings(curve_rgb=((0, 0), (0.5, 0.7), (1, 1))))
    np.testing.assert_allclose(curve, np.interp(img, np.linspace(0, 1, 1024),
                                                adjust.curve_lut(((0, 0), (0.5, 0.7), (1, 1)))), atol=2e-3)


def test_hsl_bands_contract():
    names = [n for n, _ in adjust.HSL_BANDS]
    centres = [c for _, c in adjust.HSL_BANDS]
    assert names == ["Rojo", "Naranja", "Amarillo", "Verde", "Aguamarina", "Azul", "Púrpura", "Magenta"]
    assert centres == sorted(centres) and 0 <= centres[0] and centres[-1] < 360


# ---------------------------------------------------------------- tone

def test_highlights_work_on_bright_areas(landscape):
    down = adjust.apply_adjustments(landscape, DevelopSettings(highlights=-1))
    up = adjust.apply_adjustments(landscape, DevelopSettings(highlights=1))
    sky0, grass0 = light(landscape, SKY).mean(), light(landscape, GRASS).mean()
    assert light(down, SKY).mean() < sky0 - 0.08
    assert light(up, SKY).mean() > sky0 + 0.03
    assert abs(light(down, GRASS).mean() - grass0) < 0.02


def test_shadows_work_on_dark_areas(landscape):
    up = adjust.apply_adjustments(landscape, DevelopSettings(shadows=1))
    down = adjust.apply_adjustments(landscape, DevelopSettings(shadows=-1))
    grass0 = light(landscape, GRASS).mean()
    assert light(up, GRASS).mean() > grass0 + 0.06
    assert light(down, GRASS).mean() < grass0 - 0.02
    assert abs(light(up, SKY).mean() - light(landscape, SKY).mean()) < 0.03
    # A lifted shadow keeps its colour instead of going grey.
    c0, _ = chroma_hue(landscape[GRASS].reshape(-1, 3).mean(0))
    c1, _ = chroma_hue(up[GRASS].reshape(-1, 3).mean(0))
    assert c1 > c0


def test_shadows_keep_local_contrast(landscape):
    """Driven by a smoothed lightness, the lift moves the grass as a whole and
    keeps its texture, where a plain curve with this lift would flatten it."""
    up = adjust.apply_adjustments(landscape, DevelopSettings(shadows=1))

    def detail(img):
        lum = light(img)
        return (lum - cv2.GaussianBlur(lum, (0, 0), 4))[GRASS].std()

    assert detail(up) > 0.9 * detail(landscape)


@pytest.mark.parametrize("background, values", [(0.2, (0.33, 0.47, 0.62, 0.8, 0.95)),
                                               (0.85, (0.05, 0.2, 0.33, 0.47, 0.62))])
def test_highlights_and_shadows_keep_flat_areas_flat(background, values):
    """Flat patches on a darker or brighter ground: each moves as a whole,
    with no dark centre, bright rim or halo in the ground around it. (Tones
    within about HS_RANGE of their ground count partly as the same area.)"""
    img = np.full((400, 600, 3), background, dtype=np.float32)
    for i, v in enumerate(values):
        img[130:270, 20 + i * 95:100 + i * 95] = v
    for s in (DevelopSettings(shadows=1), DevelopSettings(shadows=-1), DevelopSettings(highlights=1),
              DevelopSettings(highlights=-1), DevelopSettings(highlights=-1, shadows=1)):
        out = light(adjust.apply_adjustments(img, s, scale=0.3))
        for i in range(len(values)):
            inside = out[133:267, 23 + i * 95:97 + i * 95]
            assert inside.max() - inside.min() < 0.01, (s, values[i])
        ground = np.concatenate([out[5:120].ravel(), out[280:395].ravel()])
        assert ground.max() - ground.min() < 0.01, s
        # Tones keep their order, even with both sliders at their ends.
        means = [out[200, 60 + i * 95] for i in range(len(values))]
        assert np.all(np.diff(means) > 0.01), (s, means)


def test_highlights_reach_up_to_an_edge():
    """Sky above a dark hill: darkened as much next to the horizon as far
    from it, and the hill not darkened along its top."""
    img = np.full((300, 450, 3), 0.85, dtype=np.float32)
    img[150:] = 0.25
    out = light(adjust.apply_adjustments(img, DevelopSettings(highlights=-1)))
    change = out - light(img)
    assert change[20, 225] < -0.05
    assert abs(change[147, 225] - change[20, 225]) < 0.01
    assert abs(change[153, 225] - change[280, 225]) < 0.01


def _band_energy(img, small, large):
    lum = light(img)
    return (cv2.GaussianBlur(lum, (0, 0), small) - cv2.GaussianBlur(lum, (0, 0), large))[GRASS].std()


@pytest.mark.parametrize("field, scales", [("texture", (0.7, 4.0)), ("clarity", (6.0, 40.0))])
def test_local_contrast_controls(landscape, field, scales):
    base = _band_energy(landscape, *scales)
    more = _band_energy(adjust.apply_adjustments(landscape, DevelopSettings(**{field: 1.0})), *scales)
    less = _band_energy(adjust.apply_adjustments(landscape, DevelopSettings(**{field: -1.0})), *scales)
    assert more > 1.15 * base
    assert less < 0.9 * base


def test_texture_and_clarity_leave_flat_areas_and_mean_alone(landscape):
    for s in (DevelopSettings(texture=1), DevelopSettings(clarity=1)):
        out = adjust.apply_adjustments(landscape, s)
        assert abs(light(out).mean() - light(landscape).mean()) < 0.01
        assert np.abs(out[SKY] - landscape[SKY]).mean() < 0.01


# ---------------------------------------------------------------- vibrance

def test_vibrance_favours_muted_colours_and_spares_skin():
    muted_blue = oklch_to_srgb(0.6, 0.04, 250)
    vivid_blue = oklch_to_srgb(0.5, 0.20, 262)
    muted_skin = oklch_to_srgb(0.7, 0.04, 50)
    img = swatches([muted_blue, vivid_blue, muted_skin])
    out = adjust.apply_adjustments(img, DevelopSettings(vibrance=1))
    gain = [chroma_hue(patch(out, i))[0] / chroma_hue(patch(img, i))[0] for i in range(3)]
    assert gain[0] > 1.5  # muted colours get much more
    assert gain[1] < 1.1  # than colours already saturated
    assert 1.0 < gain[2] < 0.5 + 0.5 * gain[0]  # skin is protected
    # Unlike saturation (pipeline.render), which scales every chroma alike.
    hues = [chroma_hue(patch(out, i))[1] - chroma_hue(patch(img, i))[1] for i in range(3)]
    assert max(abs(h) for h in hues) < 2


def test_negative_vibrance_desaturates_without_going_grey():
    img = swatches([oklch_to_srgb(0.6, 0.06, 150), oklch_to_srgb(0.6, 0.15, 30)])
    out = adjust.apply_adjustments(img, DevelopSettings(vibrance=-1))
    for i in range(2):
        c0, c1 = chroma_hue(patch(img, i))[0], chroma_hue(patch(out, i))[0]
        assert 0.1 * c0 < c1 < 0.7 * c0


# ---------------------------------------------------------------- curves

def test_curve_lut_identity_and_endpoints():
    x = np.linspace(0, 1, 1024)
    np.testing.assert_allclose(adjust.curve_lut(((0, 0), (1, 1))), x, atol=1e-6)
    np.testing.assert_allclose(adjust.curve_lut(((0, 0), (0.3, 0.3), (0.8, 0.8), (1, 1)), 256), np.linspace(0, 1, 256), atol=1e-6)
    lut = adjust.curve_lut(((0, 0.1), (0.5, 0.6), (1, 0.9)))
    assert lut[0] == pytest.approx(0.1) and lut[-1] == pytest.approx(0.9)
    assert lut.dtype == np.float32 and lut.shape == (1024,)


def test_curve_lut_passes_through_points_and_is_monotone():
    pts = ((0, 0), (0.2, 0.05), (0.25, 0.5), (0.7, 0.55), (1, 1))  # steep then flat: splines overshoot here
    lut = adjust.curve_lut(pts, 1001)
    for x, y in pts:
        assert lut[int(round(x * 1000))] == pytest.approx(y, abs=1e-5)
    assert np.all(np.diff(lut) >= -1e-7)
    rng = np.random.default_rng(3)
    for _ in range(50):
        xs = np.sort(rng.random(rng.integers(2, 8)))
        ys = np.sort(rng.random(len(xs)))
        assert np.all(np.diff(adjust.curve_lut(list(zip(xs, ys)))) >= -1e-7)


def test_curve_lut_flat_beyond_end_points_and_no_overshoot():
    lut = adjust.curve_lut(((0.6, 0.8), (0.2, 0.1), (0.4, 0.9)), 1001)  # unsorted, non-monotone
    assert np.all(lut[:200] == pytest.approx(0.1)) and np.all(lut[601:] == pytest.approx(0.8))
    assert lut.max() <= 0.9 + 1e-6 and lut.min() >= 0.1 - 1e-6


def test_parametric_lut_regions():
    x = np.linspace(0, 1, 1024)
    np.testing.assert_allclose(adjust.parametric_lut(DevelopSettings()), x, atol=1e-6)
    at = lambda lut, v: lut[int(round(v * 1023))]  # noqa: E731
    shadows = adjust.parametric_lut(DevelopSettings(curve_shadows=1))
    assert at(shadows, 0.125) > 0.2 and abs(at(shadows, 0.75) - 0.75) < 0.01
    highlights = adjust.parametric_lut(DevelopSettings(curve_highlights=-1))
    assert at(highlights, 0.875) < 0.8 and abs(at(highlights, 0.25) - 0.25) < 0.01
    # Moving the splits moves where a region acts.
    wide = adjust.parametric_lut(DevelopSettings(curve_shadows=1, curve_splits=(0.45, 0.6, 0.8)))
    assert at(wide, 0.35) - 0.35 > at(shadows, 0.35) - 0.35 + 0.05
    rng = np.random.default_rng(4)
    for _ in range(40):
        v = rng.uniform(-1, 1, 4)
        s = DevelopSettings(curve_shadows=v[0], curve_darks=v[1], curve_lights=v[2], curve_highlights=v[3],
                            curve_splits=tuple(np.sort(rng.uniform(0.1, 0.9, 3))))
        lut = adjust.parametric_lut(s)
        assert lut[0] == pytest.approx(0, abs=1e-6) and lut[-1] == pytest.approx(1, abs=1e-6)
        assert np.all(np.diff(lut) >= -1e-7)


def test_tone_curves_apply_in_display_space():
    greys = swatches([(v, v, v) for v in (0.1, 0.3, 0.5, 0.7, 0.9)])
    lifted = adjust.apply_adjustments(greys, DevelopSettings(curve_rgb=((0, 0), (0.5, 0.65), (1, 1))))
    lut = adjust.curve_lut(((0, 0), (0.5, 0.65), (1, 1)), 1001)
    for i, v in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
        np.testing.assert_allclose(patch(lifted, i), lut[int(v * 1000)], atol=2e-3)
    red = adjust.apply_adjustments(greys, DevelopSettings(curve_red=((0, 0), (0.5, 0.6), (1, 1))))
    r, g, b = patch(red, 2)
    assert r > 0.58 and abs(g - 0.5) < 1e-3 and abs(b - 0.5) < 1e-3
    # Parametric first, then the RGB curve on its result.
    both = DevelopSettings(curve_lights=1, curve_rgb=((0, 0), (0.5, 0.65), (1, 1)))
    out = adjust.apply_adjustments(greys, both)
    expected = lut[int(round(adjust.parametric_lut(both, 1001)[700] * 1000))]
    assert patch(out, 3)[0] == pytest.approx(expected, abs=3e-3)


def test_curves_after_tone_use_the_same_tables(landscape):
    """The curve runs after the OKLab stage in that branch; same result as chaining."""
    tone = DevelopSettings(shadows=0.6)
    curve = ((0, 0.05), (0.5, 0.6), (1, 0.95))
    chained = adjust.apply_adjustments(adjust.apply_adjustments(landscape, tone), DevelopSettings(curve_rgb=curve))
    together = adjust.apply_adjustments(landscape, tone.copy(curve_rgb=curve))
    assert np.abs(chained - together).max() < 3e-3


# ---------------------------------------------------------------- HSL

def band_swatches():
    return swatches([oklch_to_srgb(0.68, 0.07, c) for _, c in adjust.HSL_BANDS])


@pytest.mark.parametrize("band", range(8))
def test_hsl_saturation_isolates_its_band(band):
    img = band_swatches()
    sat = [0.0] * 8
    sat[band] = -1.0
    out = adjust.apply_adjustments(img, DevelopSettings(hsl_sat=tuple(sat)))
    for i in range(8):
        c0, c1 = chroma_hue(patch(img, i))[0], chroma_hue(patch(out, i))[0]
        if i == band:
            assert c1 < 0.05 * c0
        else:
            assert abs(c1 - c0) < 0.02 * c0, (i, c0, c1)


@pytest.mark.parametrize("band", [0, 3, 5])
def test_hsl_hue_and_luminance(band):
    img = band_swatches()
    shift = [0.0] * 8
    shift[band] = 1.0
    hue_out = adjust.apply_adjustments(img, DevelopSettings(hsl_hue=tuple(shift)))
    h0, h1 = chroma_hue(patch(img, band))[1], chroma_hue(patch(hue_out, band))[1]
    assert (h1 - h0) % 360 == pytest.approx(adjust.HSL_HUE_DEGREES, abs=2.0)
    lum = [0.0] * 8
    lum[band] = -1.0
    dark = adjust.apply_adjustments(img, DevelopSettings(hsl_lum=tuple(lum)))
    assert oklab(patch(dark, band))[0] < oklab(patch(img, band))[0] - 0.1
    other = (band + 4) % 8
    assert abs(oklab(patch(dark, other))[0] - oklab(patch(img, other))[0]) < 1e-3


def test_hsl_luminance_leaves_greys_alone():
    img = swatches([(0.5, 0.5, 0.5), (0.2, 0.2, 0.2)])
    out = adjust.apply_adjustments(img, DevelopSettings(hsl_lum=(1.0,) * 8, hsl_sat=(1.0,) * 8))
    assert np.abs(out - img).max() < 2e-3


@pytest.mark.parametrize("colour, band, amount", [
    ((115, 166, 235), 5, 1.0),  # sky, Blue +100
    ((193, 227, 255), 5, 0.6),  # pale sky
    ((230, 184, 153), 1, 0.6),  # light skin, Orange +60
    ((242, 204, 38), 2, 1.0),  # yellow flower
])
def test_hsl_lightening_keeps_the_colour(colour, band, amount):
    """Brighter, same hue, and short of white: lightening must not push
    bright colours out of gamut, where they clip through other hues."""
    img = swatches([np.array(colour, dtype=np.float32) / 255])
    lum = [0.0] * 8
    lum[band] = amount
    out = adjust.apply_adjustments(img, DevelopSettings(hsl_lum=tuple(lum)))
    before, after = patch(img, 0), patch(out, 0)
    lightness = oklab(before)[0]
    assert oklab(after)[0] > lightness + 0.25 * amount * (1 - lightness)
    c0, h0 = chroma_hue(before)
    c1, h1 = chroma_hue(after)
    assert abs((h1 - h0 + 180) % 360 - 180) < 3
    assert c1 > 0.4 * c0
    assert after.min() < 0.97  # not clipped to white


# ---------------------------------------------------------------- colour grading

def grade_swatches():
    return swatches([oklch_to_srgb(L, 0.0, 0) for L in (0.25, 0.55, 0.85)])


def test_grading_tints_each_range():
    img = grade_swatches()
    blue_shadows = adjust.apply_adjustments(img, DevelopSettings(grade_shadows=(240, 1.0, 0)))
    b = [oklab(patch(blue_shadows, i))[2] for i in range(3)]
    assert b[0] < -0.03 and abs(b[2]) < 0.01  # shadows blue, highlights untouched
    warm_highlights = adjust.apply_adjustments(img, DevelopSettings(grade_highlights=(40, 1.0, 0)))
    lab = [oklab(patch(warm_highlights, i)) for i in range(3)]
    assert lab[2][2] > 0.03 and abs(lab[0][2]) < 0.01
    _, hue = chroma_hue(patch(warm_highlights, 2))
    assert 20 < hue < 90  # orange, where the colour wheel puts 40°
    lifted = adjust.apply_adjustments(img, DevelopSettings(grade_midtones=(0, 0, 1.0)))
    assert oklab(patch(lifted, 1))[0] > oklab(patch(img, 1))[0] + 0.05
    every = adjust.apply_adjustments(img, DevelopSettings(grade_global=(120, 0.5, 0)))
    assert all(oklab(patch(every, i))[1] < -0.01 for i in range(3))  # greener everywhere


def test_grading_balance_and_blending():
    img = grade_swatches()
    tint = dict(grade_highlights=(40, 1.0, 0))
    mid_b = lambda s: oklab(patch(adjust.apply_adjustments(img, s), 1))[2]  # noqa: E731
    neutral = mid_b(DevelopSettings(**tint))
    assert mid_b(DevelopSettings(grade_balance=1.0, **tint)) > neutral + 0.01
    assert mid_b(DevelopSettings(grade_balance=-1.0, **tint)) < neutral
    assert mid_b(DevelopSettings(grade_blending=1.0, **tint)) > mid_b(DevelopSettings(grade_blending=0.0, **tint))


# ---------------------------------------------------------------- detail

def test_noise_reduction_smooths_noise_and_keeps_edges():
    rng = np.random.default_rng(2)
    img = np.full((120, 240, 3), 0.3, dtype=np.float32)
    img[:, 120:] = 0.7
    noisy = np.clip(img + rng.normal(0, 0.03, img.shape).astype(np.float32), 0, 1)
    out = adjust.apply_adjustments(noisy, DevelopSettings(nr_luma=1.0, nr_color=1.0))
    flat = (slice(20, 100), slice(20, 100))
    assert light(out)[flat].std() < 0.5 * light(noisy)[flat].std()
    # Colour noise (channel differences) goes too.
    assert np.std(out[flat][..., 0] - out[flat][..., 2]) < 0.5 * np.std(noisy[flat][..., 0] - noisy[flat][..., 2])
    step = light(out)[60, 124] - light(out)[60, 115]
    assert step > 0.9 * 0.4


def test_colour_noise_reduction_keeps_colour_edges():
    """Red beside a grey of the same lightness: no lightness edge to follow,
    yet the grey must not turn pink while its colour noise goes."""
    red = oklch_to_srgb(0.6, 0.15, 25)
    grey = oklch_to_srgb(0.6, 0.0, 0)
    img = np.empty((120, 240, 3), dtype=np.float32)
    img[:, :120] = red
    img[:, 120:] = grey
    rng = np.random.default_rng(4)
    noisy = np.clip(img + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
    out = adjust.apply_adjustments(noisy, DevelopSettings(nr_color=1.0))
    chroma_out, _ = chroma_hue(out)
    chroma_in, _ = chroma_hue(noisy)
    assert np.median(chroma_out[:, 124:132]) < 0.02  # 4-12 px from the edge: still grey
    assert np.median(chroma_out[:, 100:116]) > 0.13  # and the red stays red
    assert chroma_out[:, 160:].mean() < 0.6 * chroma_in[:, 160:].mean()  # colour speckle reduced


def _colour_blotches(blob, size=(300, 450), seed=6):
    """Mid grey with colour-only noise in blotches about ``blob`` pixels wide."""
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(size=(*size, 3)).astype(np.float32), (0, 0), blob)
    noise -= noise.mean(-1, keepdims=True)
    return np.clip(0.45 + noise * (0.03 / noise.std()), 0, 1)


def test_colour_noise_reduction_removes_blotches():
    """Film scans show colour mottle several pixels wide, not just per pixel."""
    img = _colour_blotches(6.0)
    chroma = lambda im: chroma_hue(im)[0].mean()  # noqa: E731
    left = {v: chroma(adjust.apply_adjustments(img, DevelopSettings(nr_color=v))) / chroma(img) for v in (0.25, 1.0)}
    assert left[1.0] < 0.55  # mostly gone at full strength
    assert left[0.25] < 0.9  # and already reduced at Lightroom's default


def test_colour_noise_reduction_keeps_small_colour_details():
    """A wide window must not wash out colourful dots a few pixels across."""
    img = np.full((240, 240, 3), 0.45, dtype=np.float32)
    yy, xx = np.mgrid[:240, :240]
    dots = np.zeros((240, 240), bool)
    for cy in range(30, 240, 60):
        for cx in range(30, 240, 60):
            dots |= (yy - cy) ** 2 + (xx - cx) ** 2 <= 9
    img[dots] = oklch_to_srgb(0.6, 0.1, 140)
    rng = np.random.default_rng(8)
    noisy = np.clip(img + rng.normal(0, 0.01, img.shape).astype(np.float32), 0, 1)
    out = adjust.apply_adjustments(noisy, DevelopSettings(nr_color=0.25))
    assert chroma_hue(out)[0][dots].mean() > 0.85 * chroma_hue(img)[0][dots].mean()


def _soft_edge():
    x = np.linspace(-1, 1, 240, dtype=np.float32)
    ramp = 0.3 + 0.2 * (1 + np.tanh(x * 60))
    return np.repeat(np.repeat(ramp[None, :, None], 120, 0), 3, 2)


def test_sharpening_adds_edge_contrast():
    img = _soft_edge()
    out = adjust.apply_adjustments(img, DevelopSettings(sharpen_amount=1.0, sharpen_radius=1.5, sharpen_detail=1.0))
    grad = lambda im: np.abs(np.diff(light(im)[60])).max()  # noqa: E731
    assert grad(out) > 1.2 * grad(img)
    # Detail low caps the halo at a strong edge.
    soft = adjust.apply_adjustments(img, DevelopSettings(sharpen_amount=1.0, sharpen_radius=1.5, sharpen_detail=0.0))
    assert light(soft)[60].max() - 0.7 < light(out)[60].max() - 0.7


def test_sharpening_radius_sets_the_halo_width():
    img = _soft_edge()
    narrow = adjust.apply_adjustments(img, DevelopSettings(sharpen_amount=1.0, sharpen_radius=0.5, sharpen_detail=1.0))
    wide = adjust.apply_adjustments(img, DevelopSettings(sharpen_amount=1.0, sharpen_radius=3.0, sharpen_detail=1.0))
    away = (slice(10, 110), np.r_[110:115, 125:130])  # 5-10 px from the edge
    assert np.abs(wide[away] - img[away]).mean() > 3 * np.abs(narrow[away] - img[away]).mean()


def _textured(size=1200, seed=0):
    """Detail at every scale (1/f), as a photograph has."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(size, size)).astype(np.float32)
    lum = sum(cv2.GaussianBlur(noise, (0, 0), s) * s for s in (0.7, 1.5, 3, 6, 12))
    lum = 0.45 + 0.12 * lum / lum.std()
    return np.repeat(np.clip(lum, 0.02, 0.98)[..., None], 3, 2)


def _high_pass(img):
    lum = light(img)
    return (lum - cv2.GaussianBlur(lum, (0, 0), 2.0)).std()


@pytest.mark.parametrize("settings", [
    DevelopSettings(sharpen_amount=1.0),  # default radius and detail
    DevelopSettings(sharpen_amount=1.5, sharpen_detail=1.0),
    DevelopSettings(texture=1.0),
    DevelopSettings(texture=-1.0),
], ids=["sharpen", "sharpen-strong", "texture+", "texture-"])
def test_fine_detail_controls_match_the_export_in_a_quarter_size_preview(settings):
    """The app previews at about a quarter of the camera's width: a fine
    radius there is under a pixel. The preview must still show the change
    in fine detail the downsampled export gets, neither nothing nor double."""
    full = _textured()
    small = pl.downsample(full, 300)
    base = _high_pass(small)
    exported = _high_pass(pl.downsample(adjust.apply_adjustments(full, settings, scale=1.0), 300)) / base - 1
    preview = _high_pass(adjust.apply_adjustments(small, settings, scale=0.25)) / base - 1
    assert abs(exported) > 0.03
    assert 0.5 < preview / exported < 1.35, (exported, preview)


def test_sharpening_responds_smoothly_to_the_radius():
    """No radius at which the preview's sharpening switches on with a jump."""
    small = pl.downsample(_textured(), 300)
    effect = [_high_pass(adjust.apply_adjustments(small, DevelopSettings(sharpen_amount=1.0, sharpen_radius=r), scale=0.25))
              for r in np.linspace(0.5, 3.0, 11)]
    steps = np.diff(effect)
    assert steps[0] > 0.0005  # already acting at half the default radius
    assert np.all(steps > 0) and np.all(np.abs(np.diff(np.log(steps))) < np.log(1.7))


def test_sharpening_masking_spares_smooth_areas():
    rng = np.random.default_rng(5)
    img = _soft_edge()
    img[:, :60] += rng.normal(0, 0.01, (120, 60, 3)).astype(np.float32)  # faint texture, no edge
    plain = adjust.apply_adjustments(img, DevelopSettings(sharpen_amount=1.0))
    masked = adjust.apply_adjustments(img, DevelopSettings(sharpen_amount=1.0, sharpen_masking=1.0))
    flat = (slice(10, 110), slice(5, 55))
    change = lambda out: np.abs(out[flat] - img[flat]).mean()  # noqa: E731
    assert change(masked) < 0.3 * change(plain)
    edge = (slice(10, 110), slice(110, 130))
    assert np.abs(masked[edge] - img[edge]).mean() > 0.5 * np.abs(plain[edge] - img[edge]).mean()


# ---------------------------------------------------------------- effects

def test_vignette_darkens_or_lightens_the_corners():
    img = np.full((300, 450, 3), 0.5, dtype=np.float32)
    dark = adjust.apply_adjustments(img, DevelopSettings(vignette_amount=-1))
    light_ = adjust.apply_adjustments(img, DevelopSettings(vignette_amount=1))
    centre = (slice(140, 160), slice(215, 235))
    corner = (slice(0, 10), slice(0, 10))
    assert np.abs(dark[centre] - 0.5).max() < 1e-3
    assert dark[corner].mean() < 0.25 and light_[corner].mean() > 0.85
    # Symmetric about the centre of the (cropped) frame.
    np.testing.assert_allclose(dark, dark[::-1, ::-1], atol=1e-4)
    # A larger midpoint keeps more of the frame clear.
    wide = adjust.apply_adjustments(img, DevelopSettings(vignette_amount=-1, vignette_midpoint=1.0))
    assert wide.mean() > dark.mean() + 0.02
    # Round: same falloff along both axes in pixels; rectangle-ish: edges darken like corners.
    round_ = adjust.apply_adjustments(img, DevelopSettings(vignette_amount=-1, vignette_roundness=1, vignette_feather=0.2))
    assert abs(round_[150, 225 - 140, 0] - round_[150 - 140, 225, 0]) < 0.02
    rect = adjust.apply_adjustments(img, DevelopSettings(vignette_amount=-1, vignette_roundness=-1, vignette_feather=0))
    assert rect[150, 2, 0] < 0.3 and rect[150, 60, 0] > 0.49


def test_vignette_feather_softens_the_transition():
    img = np.full((300, 450, 3), 0.5, dtype=np.float32)
    steepest = lambda s: np.abs(np.diff(adjust.apply_adjustments(img, s)[150, :, 1])).max()  # noqa: E731
    hard = DevelopSettings(vignette_amount=-1, vignette_feather=0.0)
    assert steepest(hard) > 2 * steepest(hard.copy(vignette_feather=1.0))


def test_grain_roughness_adds_clumps():
    img = np.full((200, 200, 3), 0.5, dtype=np.float32)

    def corr(rough: float) -> float:
        s = DevelopSettings(grain_amount=1.0, grain_roughness=rough)
        g = (adjust.apply_adjustments(img, s, seed=2) - img)[..., 1]
        return float(np.corrcoef(g[:, :-3].ravel(), g[:, 3:].ravel())[0, 1])

    assert corr(1.0) > corr(0.0) + 0.1


def test_grain_is_deterministic_and_lives_in_the_midtones():
    img = swatches([(0.03, 0.03, 0.03), (0.5, 0.5, 0.5), (0.97, 0.97, 0.97)], size=120)
    s = DevelopSettings(grain_amount=1.0)
    a = adjust.apply_adjustments(img, s, seed=11)
    np.testing.assert_array_equal(a, adjust.apply_adjustments(img, s, seed=11))
    assert not np.array_equal(a, adjust.apply_adjustments(img, s, seed=12))
    noise = [np.std((a - img)[20:100, i * 120 + 20:i * 120 + 100].mean(-1)) for i in range(3)]
    assert noise[1] > 1.5 * noise[0] and noise[1] > 1.5 * noise[2]
    assert abs(patch(a, 1, 120).mean() - 0.5) < 0.01  # grain does not shift the tone
    # Bigger grain makes bigger clumps: neighbours are more alike.
    big = adjust.apply_adjustments(img, s.copy(grain_size=1.0), seed=11)
    corr = lambda im: np.corrcoef((im - img)[20:100, 140:220, 1][:, :-2].ravel(),  # noqa: E731
                                  (im - img)[20:100, 140:220, 1][:, 2:].ravel())[0, 1]
    assert corr(big) > corr(a) + 0.1


@pytest.mark.parametrize("size, rough", [(0.0, 1.0), (0.25, 1.0), (0.0, 0.0), (1.0, 0.5)])
def test_grain_preview_is_as_strong_as_the_export(size, rough):
    """Both the fine grain and the roughness clumps fade with their own
    size in a small preview, as they do when the export is downsampled."""
    full = np.full((1200, 1800, 3), 0.5, dtype=np.float32)
    small = pl.downsample(full, 450)
    s = DevelopSettings(grain_amount=1.0, grain_size=size, grain_roughness=rough)
    exported = (pl.downsample(adjust.apply_adjustments(full, s, scale=1.0, seed=3), 450) - small).mean(-1).std()
    preview = (adjust.apply_adjustments(small, s, scale=0.25, seed=3) - small).mean(-1).std()
    assert 0.85 < preview / exported < 1.2


def test_grain_clumps_are_round():
    """Same correlation at the same distance along an axis and on a slant:
    no grid-aligned lattice or maze."""
    img = np.full((400, 400, 3), 0.5, dtype=np.float32)
    g = (adjust.apply_adjustments(img, DevelopSettings(grain_amount=0.6, grain_size=1.0, grain_roughness=0.0), seed=5)
         - img)[..., 1]

    def corr(dy, dx):
        return np.corrcoef(g[20:-20, 20:-20].ravel(), g[20 + dy:380 + dy, 20 + dx:380 + dx].ravel())[0, 1]

    assert corr(0, 5) > 0.2  # clumps a few pixels wide
    assert abs(corr(0, 5) - corr(3, 4)) < 0.05
    assert abs(corr(5, 0) - corr(4, 3)) < 0.05


# ---------------------------------------------------------------- sections, scale, speed

SECTIONS = {
    "curve": dict(curve_darks=0.6, curve_rgb=((0, 0), (0.5, 0.6), (1, 1))),
    "hsl": dict(hsl_sat=(0.8,) * 8, hsl_hue=(0.5,) * 8),
    "grading": dict(grade_shadows=(220, 0.8, 0.3)),
    "detail": dict(sharpen_amount=1.0, nr_luma=0.8, nr_color=0.8),
    "effects": dict(vignette_amount=-0.8, grain_amount=0.8),
}


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_disabled_sections_are_skipped(landscape, section):
    base = DevelopSettings(shadows=0.5)
    reference = adjust.apply_adjustments(landscape, base, seed=1)
    off = base.copy(disabled=(section,), **SECTIONS[section])
    np.testing.assert_array_equal(adjust.apply_adjustments(landscape, off, seed=1), reference)
    on = base.copy(**SECTIONS[section])
    assert np.abs(adjust.apply_adjustments(landscape, on, seed=1) - reference).mean() > 1e-3


def test_result_does_not_depend_on_how_the_work_is_split(landscape, monkeypatch):
    """Grain and filters are whole-image, so strips only change where OpenCV's
    vector loops end: a last-bit difference that can move a table read by
    one entry, never more."""
    s = everything(sharpen_radius=2.0)
    reference = adjust.apply_adjustments(landscape, s, scale=0.5, seed=4)
    monkeypatch.setattr(adjust, "_STRIP_PIXELS", 7000)
    assert np.abs(adjust.apply_adjustments(landscape, s, scale=0.5, seed=4) - reference).max() < 2e-4


def test_concurrent_calls_are_safe(landscape):
    """The preview worker and an export may run at once on the shared pool."""
    settings = [everything(shadows=v) for v in (0.1, 0.5, -0.4)]
    expected = [adjust.apply_adjustments(landscape, s, seed=1) for s in settings]
    results: dict[int, np.ndarray] = {}

    def run(i: int) -> None:
        results[i] = adjust.apply_adjustments(landscape, settings[i], seed=1)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(settings))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    for i, e in enumerate(expected):
        np.testing.assert_array_equal(results[i], e)


def test_preview_matches_the_downsampled_export():
    rng = np.random.default_rng(0)
    full = pl.srgb_encode(synth.landscape_scene(1200, 800, seed=3) * 0.9)
    full = np.clip(full + rng.normal(0, 0.02, full.shape).astype(np.float32), 0, 1)
    small = pl.downsample(full, 300)
    s = everything(grain_amount=0.0)
    exported = pl.downsample(adjust.apply_adjustments(full, s, scale=1.0), 300)
    preview = adjust.apply_adjustments(small, s, scale=0.25)
    diff = np.abs(exported - preview)
    assert diff.mean() < 0.006 and np.percentile(diff, 99) < 0.04
    assert np.abs(preview - small).mean() > 0.05  # the settings do change the picture
    # Grain is random per pixel, but just as strong in both.
    g = DevelopSettings(grain_amount=1.0)
    grain_full = pl.downsample(adjust.apply_adjustments(full, g, scale=1.0) - full, 300).std()
    grain_prev = (adjust.apply_adjustments(small, g, scale=0.25) - small).std()
    assert 0.7 < grain_prev / grain_full < 1.4


def test_clipping_masks():
    img = np.array([[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0.5, 0.5, 0.5], [0.002, 0.5, 0.998]]], dtype=np.float32)
    shadows, highlights = adjust.clipping_masks(img)
    assert shadows.dtype == bool and shadows.shape == (1, 5)
    assert shadows.tolist() == [[True, False, True, False, False]]
    assert highlights.tolist() == [[False, True, True, False, False]]


def test_everything_on_preview_size_is_fast():
    """Contract: 1800x1200 with everything on in under 250 ms. Measured as
    while dragging Sombras: that slider's table is rebuilt on every frame,
    and the sharpening radius is wide enough to act in this preview."""
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    if load > (os.cpu_count() or 1) / 3:
        pytest.skip(f"machine busy (load {load:.1f}), timing would say nothing")
    rng = np.random.default_rng(0)
    img = pl.srgb_encode(synth.landscape_scene(1800, 1200, seed=1) * 0.9)
    img = np.clip(img + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
    s = everything(sharpen_radius=1.5)
    adjust.apply_adjustments(img, s, scale=0.3)  # start the threads
    best = min(_timed(lambda: adjust.apply_adjustments(img, s.copy(shadows=0.3 + 0.01 * i), scale=0.3))
               for i in range(6))
    assert best < 0.25, f"{best * 1000:.0f} ms"


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
