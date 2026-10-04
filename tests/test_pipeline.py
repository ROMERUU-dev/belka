import os
import time

import cv2
import numpy as np
import pytest

from belka.core import pipeline as pl
from belka.core import synth


def _patches(image, scene_shape):
    """Mean of each ColorChecker patch, from an image with the rebate border."""
    h, w = scene_shape[:2]
    H, W = image.shape[:2]
    by, bx = (H - h) // 2, (W - w) // 2
    inner = image[by:by + h, bx:bx + w]
    chart_h = int(h * 0.62)
    cw, ch = w / 6, chart_h / 4
    out = []
    for i in range(24):
        r, c = divmod(i, 6)
        out.append(inner[int(r * ch) + 10:int((r + 1) * ch) - 10, int(c * cw) + 10:int((c + 1) * cw) - 10].reshape(-1, 3).mean(0))
    return np.array(out)


def _develop(raw, settings, profile):
    analysis = pl.analyze(raw, settings, profile)
    return pl.srgb_decode(pl.render(raw, analysis, settings, profile)), analysis


@pytest.mark.parametrize("film", ["generic-c41", "kodak-portra-400", "kodak-ektar-100", "fujifilm-superia-xtra-400", "cinestill-800t"])
def test_colour_negative_greys_come_out_neutral_and_ordered(library, film):
    profile = library.get(film)
    raw, scene = synth.synthetic_negative(profile)
    out, _ = _develop(raw, pl.DevelopSettings(profile_id=film), profile)
    greys = _patches(out, scene.shape)[18:]
    chroma = (greys.max(1) - greys.min(1)) / greys.mean(1)
    assert chroma.max() < 0.06, chroma
    lum = greys.mean(1)
    assert np.all(np.diff(lum) < 0), "white to black patches must get darker"
    assert lum[0] > 0.6 and lum[-1] < 0.08


def test_colour_negative_hues_survive_inversion(library):
    profile = library.get("kodak-portra-400")
    raw, scene = synth.synthetic_negative(profile)
    out, _ = _develop(raw, pl.DevelopSettings(profile_id=profile.id), profile)
    src = _patches(scene, scene.shape)[:18]
    got = _patches(out, scene.shape)[:18]

    def hue(rgb):
        lg = np.log(np.maximum(rgb, 1e-4))
        return np.degrees(np.arctan2((lg[:, 0] + lg[:, 1]) / 2 - lg[:, 2], lg[:, 0] - lg[:, 1]))

    err = np.abs((hue(got) - hue(src) + 180) % 360 - 180)
    assert err.mean() < 8 and err.max() < 20, err


def test_full_auto_balance_does_not_depend_on_the_base(library):
    profile = library.get("kodak-portra-400")
    raw, _ = synth.synthetic_negative(profile)
    # Same analysis region either way: frame detection itself measures
    # densities against the base, so it is pinned here.
    settings = pl.DevelopSettings(profile_id=profile.id, auto_balance=1.0, crop=(0.1, 0.15, 0.9, 0.85))
    auto, _ = _develop(raw, settings, profile)
    forced, _ = _develop(raw, settings.copy(base=(0.9, 0.7, 0.4)), profile)
    assert np.allclose(auto, forced, atol=2e-3)


def test_light_colour_is_cancelled_by_the_base(library):
    """A tinted backlight changes the raw file but not the developed positive."""
    profile = library.get("generic-c41")
    white, scene = synth.synthetic_negative(profile, light_rgb=(1.0, 1.0, 1.0))
    tinted, _ = synth.synthetic_negative(profile, light_rgb=(0.3, 0.4, 1.0))
    settings = pl.DevelopSettings(profile_id=profile.id)
    a = _patches(_develop(white, settings, profile)[0], scene.shape)
    b = _patches(_develop(tinted, settings, profile)[0], scene.shape)
    assert np.abs(a - b).max() < 0.05


def test_bw_negative_is_monochrome(library):
    profile = library.get("ilford-hp5-plus")
    raw, scene = synth.synthetic_negative(profile)
    out, analysis = _develop(raw, pl.DevelopSettings(profile_id=profile.id), profile)
    assert analysis.channels == 1
    assert np.allclose(out[..., 0], out[..., 1]) and np.allclose(out[..., 1], out[..., 2])
    greys = _patches(out, scene.shape)[18:, 0]
    assert np.all(np.diff(greys) < 0)


def test_slide_is_not_inverted(library):
    profile = library.get("fujifilm-provia-100f")
    raw, scene = synth.synthetic_negative(profile)
    out, _ = _develop(raw, pl.DevelopSettings(profile_id=profile.id), profile)
    greys = _patches(out, scene.shape)[18:].mean(1)
    assert greys[0] > greys[-1] + 0.4


def test_neutral_picker_neutralises_the_patch(library):
    profile = library.get("kodak-portra-400")
    raw, scene = synth.synthetic_negative(profile)
    settings = pl.DevelopSettings(profile_id=profile.id, auto_balance=0.0, temperature=0.8, black=0.05, white=-0.05)
    analysis = pl.analyze(raw, settings, profile)
    # Patch 20 (neutral 6.5) in raw-image coordinates.
    h, w = scene.shape[:2]
    H, W = raw.shape[:2]
    by, bx = (H - h) // 2, (W - w) // 2
    cw, ch = w / 6, int(h * 0.62) / 4
    ys = slice(by + int(3 * ch) + 10, by + int(4 * ch) - 10)
    xs = slice(bx + int(2 * cw) + 10, bx + int(3 * cw) - 10)
    offsets = pl.neutral_offsets(raw[ys, xs], analysis, settings, profile)
    fixed = settings.copy(neutral=offsets)  # still with temperature 0.8
    out = pl.render(raw[ys, xs], analysis, fixed, profile)
    mean = out.reshape(-1, 3).mean(0)
    assert (mean.max() - mean.min()) / mean.mean() < 0.03


def test_exposure_and_contrast_move_the_image(library):
    profile = library.get("generic-c41")
    raw, _ = synth.synthetic_negative(profile)
    base = pl.DevelopSettings(profile_id=profile.id)
    analysis = pl.analyze(raw, base, profile)
    mid = lambda s: float(pl.render(raw, analysis, s, profile).mean())
    assert mid(base.copy(exposure=1.0)) > mid(base) > mid(base.copy(exposure=-1.0))


def test_flat_output_is_linear_and_in_range(library):
    profile = library.get("generic-c41")
    raw, _ = synth.synthetic_negative(profile)
    settings = pl.DevelopSettings(profile_id=profile.id, output="flat")
    analysis = pl.analyze(raw, settings, profile)
    out = pl.render(raw, analysis, settings, profile)
    assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1


def test_paper_curve_hits_its_targets():
    paper = pl.PaperCurve()
    out = paper.apply(np.array([pl.MID_GRAY, 1.0, 2.0 ** -9], dtype=np.float32))
    assert out[0] == pytest.approx(pl.MID_GRAY, abs=1e-3)
    assert out[1] == pytest.approx(paper.white, abs=1e-3)
    assert out[2] < 0.005


def test_separation_matrix_keeps_neutrals():
    m = pl.separation_matrix(0.7, None)
    assert np.allclose(m @ np.ones(3), np.ones(3), atol=1e-6)
    assert np.allclose(pl.separation_matrix(0.0, None), np.eye(3))


def test_orient_and_crop():
    img = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    rotated = pl.orient(img, 90)
    assert rotated.shape == (3, 2, 3)
    # Clockwise: the bottom-left pixel becomes the top-left one.
    assert np.array_equal(rotated[0, 0], img[1, 0])
    assert np.array_equal(pl.orient(img, flip_h=True)[0, 0], img[0, 2])
    ys, xs = pl.crop_slices((100, 200), (0.25, 0.5, 0.75, 1.0))
    assert (ys.start, ys.stop, xs.start, xs.stop) == (50, 100, 50, 150)


def test_downsample_box_filter():
    img = np.ones((1000, 600, 3), dtype=np.float32)
    small = pl.downsample(img, 250)
    assert max(small.shape[:2]) <= 250
    assert np.allclose(small, 1.0)


def test_settings_json_round_trip():
    s = pl.DevelopSettings(base=(0.5, 0.4, 0.2), crop=(0.1, 0.1, 0.9, 0.9), rotation=90, separation=None)
    again = pl.DevelopSettings.from_json(s.to_json())
    assert again == s
    assert pl.DevelopSettings.from_json({"bogus": 1}).profile_id == "generic-c41"


@pytest.mark.parametrize("camera_ev", [-3.5, -2.0, 0.0])
def test_real_world_scan_layout(library, camera_ev):
    """Bare light around the strip, holes, a dark surround with glare.

    Regression for the first real scans: with the light unclipped the
    old analysis took the bare light for the film base and the surround for
    the densest highlight, and the inversion came out wrong.
    """
    profile = library.get("kodak-gold-200")
    raw, truth = synth.backlit_scan(profile, camera_ev=camera_ev)
    settings = pl.DevelopSettings(profile_id=profile.id)
    analysis = pl.analyze(raw, settings, profile)
    err = np.abs(np.log10(analysis.base / truth["base"]))
    assert err.max() < 0.06, (analysis.base, truth["base"])
    frame = analysis.extra["frame"]
    assert frame is not None
    assert np.abs(np.asarray(frame) - np.asarray(truth["frame"])).max() < 0.03, (frame, truth["frame"])
    reference = pl.analyze(raw, settings.copy(crop=truth["frame"], base=tuple(truth["base"])), profile)
    assert np.abs(analysis.lo - reference.lo).max() < 0.06
    assert np.abs(analysis.hi - reference.hi).max() < 0.06
    advice = pl.exposure_advice(analysis)
    if camera_ev <= -2:
        assert advice == pytest.approx(-camera_ev + np.log2(0.75 / 0.85), abs=0.3)
    else:
        assert advice == 0.0
    # The developed picture is the cropped frame, with neutral greys.
    out = pl.render(pl.apply_geometry(raw, settings, analysis), analysis, settings, profile)
    h, w = out.shape[:2]
    cw, ch = w / 6, h * 0.62 / 4
    greys = np.array([out[int(3 * ch) + 6:int(4 * ch) - 6, int(c * cw) + 6:int((c + 1) * cw) - 6].reshape(-1, 3).mean(0)
                      for c in range(6)])
    lin = pl.srgb_decode(greys)
    chroma = (lin.max(1) - lin.min(1)) / lin.mean(1)
    # 3.5 stops under, sensor noise biases the per-channel highlight levels a
    # little; that is what the exposure warning is for.
    assert chroma[:5].max() < (0.2 if camera_ev < -3 else 0.12), chroma
    assert np.all(np.diff(lin.mean(1)) < 0)


def test_frame_detection_gives_up_on_a_blank_view(library):
    profile = library.get("generic-c41")
    blank = np.full((300, 450, 3), 0.4, np.float32)
    analysis = pl.analyze(blank, pl.DevelopSettings(), profile)
    assert analysis.extra["frame"] is None
    assert pl.effective_crop(pl.DevelopSettings(), analysis) is None


@pytest.mark.parametrize("gamma", [0.25, 0.5, 0.64, 0.8, 0.88, 0.9, 1.2, 1.6, 2.4, 3.2, 4.0])
def test_paper_curve_is_continuous_and_keeps_middle_grey(gamma):
    """Regression: soft papers (Velvia, low contrast) used to jump to the +3 bound."""
    paper = pl.PaperCurve(gamma=gamma)
    centre = paper.centre()
    assert -3.0 < centre < 3.0
    mid = float(paper.apply(np.array([pl.MID_GRAY], np.float32))[0])
    assert mid == pytest.approx(pl.MID_GRAY, abs=0.005)
    ramp = paper.apply(np.power(2.0, np.linspace(-10, 1, 200)).astype(np.float32))
    assert np.all(np.diff(ramp) >= -1e-6)
    near = pl.PaperCurve(gamma=gamma * 1.01)
    assert abs(near.centre() - centre) < 0.2
    probe = np.array([0.02, 0.18, 1.0], np.float32)
    assert np.abs(near.apply(probe) - paper.apply(probe)).max() < 0.02


def _night_scene():
    scene = synth.colorchecker_scene(600, 400)
    scene[: int(400 * 0.45)] = 0.0004  # a black night sky over the top 45 %
    return scene


def test_frame_detection_keeps_a_night_sky(library):
    """Regression: base-level picture areas used to be cut off by auto crop."""
    profile = library.get("kodak-portra-400")
    raw, truth = synth.backlit_scan(profile, scene=_night_scene(), camera_ev=-1.0)
    frame = pl.analyze(raw, pl.DevelopSettings(profile_id=profile.id), profile).extra["frame"]
    assert frame is not None
    assert np.abs(np.asarray(frame) - np.asarray(truth["frame"])).max() < 0.04, (frame, truth["frame"])


def test_base_without_bare_light_in_view(library):
    """Regression: with no bare light in view the base was taken for light."""
    profile = library.get("kodak-gold-200")
    raw, truth = synth.backlit_scan(profile, camera_ev=-2.0)
    fx0, fy0, fx1, fy1 = truth["frame"]
    h, w = raw.shape[:2]
    # Tight framing: picture plus a little rebate, no holes, no bare light.
    pad_y, pad_x = int((fy1 - fy0) * h * 0.06), int((fx1 - fx0) * w * 0.06)
    tight = raw[int(fy0 * h) - pad_y:int(fy1 * h) + pad_y, int(fx0 * w) - pad_x:int(fx1 * w) + pad_x]
    base = pl.analyze(tight, pl.DevelopSettings(profile_id=profile.id), profile).base
    assert np.abs(np.log10(base / truth["base"])).max() < 0.08, (base, truth["base"])


def test_base_with_tint_calibrated_light(library):
    profile = library.get("kodak-portra-400")
    raw, truth = synth.backlit_scan(profile, light_rgb=(0.28, 0.31, 1.0), camera_ev=0.5)
    base = pl.analyze(raw, pl.DevelopSettings(profile_id=profile.id), profile).base
    assert np.abs(np.log10(base / truth["base"])).max() < 0.08, (base, truth["base"])


def test_exposure_advice_reports_clipped_base_and_missing_light(library):
    profile = library.get("kodak-portra-400")
    over, _t = synth.backlit_scan(profile, camera_ev=1.5)
    analysis = pl.analyze(over, pl.DevelopSettings(profile_id=profile.id), profile)
    assert pl.exposure_advice(analysis) == -1.0
    dark = np.abs(np.random.default_rng(0).normal(0, 0.0004, (300, 450, 3))).astype(np.float32)
    analysis = pl.analyze(dark, pl.DevelopSettings(profile_id=profile.id), profile)
    assert analysis.extra["no_light"] and pl.exposure_advice(analysis) == 0.0


# Every slider of the inversion at its ends, alone and combined.
RENDER_CASES = [
    {}, {"output": "flat"}, {"exposure": 5.0}, {"exposure": -5.0}, {"temperature": 1.5, "tint": -1.5},
    {"temperature": -1.5, "tint": 1.5}, {"contrast": 0.4}, {"contrast": 2.0}, {"black": 0.3, "white": 0.3},
    {"black": -0.3, "white": -0.3}, {"separation": 0.0}, {"separation": 1.5, "neutral": (0.05, -0.03, 0.02)},
    {"auto_balance": 0.0}, {"auto_balance": 1.0}, {"saturation": 0.0}, {"saturation": 2.0, "contrast": 2.0},
    {"output": "flat", "exposure": 5.0, "temperature": -1.5}, {"output": "flat", "exposure": -5.0},
]
CAMERA = np.array([[1.9, -0.8, -0.1], [-0.25, 1.6, -0.35], [0.02, -0.6, 1.58]])


@pytest.mark.parametrize("film", ["kodak-portra-400", "ilford-hp5-plus", "fujifilm-velvia-50", "kodak-gold-200"])
def test_render_matches_the_reference(library, film):
    """The fast path (tables read on a thread pool) against the model written
    out step by step: clipped light, bare base and dense film included."""
    profile = library.get(film)
    if film == "kodak-gold-200":
        raw, _ = synth.backlit_scan(profile, camera_ev=1.5)
    else:
        raw, _ = synth.synthetic_negative(profile, scene=synth.landscape_scene(300, 200, seed=1))
    base = pl.DevelopSettings(profile_id=film)
    analysis = pl.analyze(raw, base, profile)
    img = pl.downsample(raw, 400)  # the reference is slow
    for changes in RENDER_CASES:
        settings = base.copy(**changes)
        for camera, sequential in ((None, False), (CAMERA, True)):
            got = pl.render(img, analysis, settings, profile, camera, sequential)
            with np.errstate(over="ignore"):
                want = pl._render_reference(img, analysis, settings, profile, camera, sequential)
            assert got.dtype == np.float32 and got.shape == want.shape
            assert np.abs(got - want).max() < 2e-4, (changes, camera is None)


def test_render_takes_views_and_odd_shapes(library):
    profile = library.get("kodak-portra-400")
    raw, _ = synth.synthetic_negative(profile, scene=synth.landscape_scene(150, 100, seed=2))
    settings = pl.DevelopSettings(profile_id=profile.id, saturation=1.3)
    analysis = pl.analyze(raw, settings, profile)
    bw = library.get("ilford-hp5-plus")
    bw_analysis = pl.analyze(raw, pl.DevelopSettings(profile_id=bw.id), bw)
    for img in (np.rot90(raw), raw[::-1, ::2], raw.astype(np.float64), raw[40:41], raw[:, 70:71], raw[3:4, 4:5]):
        for prof, an, s in ((profile, analysis, settings), (bw, bw_analysis, pl.DevelopSettings(profile_id=bw.id))):
            for output in ("print", "flat"):
                got = pl.render(img, an, s.copy(output=output), prof)
                want = pl._render_reference(img, an, s.copy(output=output), prof)
                assert got.shape == want.shape and np.abs(got - want).max() < 2e-4
    assert pl.render(raw[:0], analysis, settings, profile).shape == (0, raw.shape[1], 3)


def test_render_does_not_depend_on_how_the_work_is_split(library, monkeypatch):
    profile = library.get("kodak-ektar-100")
    raw, _ = synth.synthetic_negative(profile, scene=synth.landscape_scene(300, 200, seed=3))
    settings = pl.DevelopSettings(profile_id=profile.id, contrast=1.4)
    analysis = pl.analyze(raw, settings, profile)
    whole = pl.render(raw, analysis, settings, profile)
    monkeypatch.setattr(pl, "_STRIP_PIXELS", 3000)
    monkeypatch.setattr(pl, "_REMAP_BLOCK", 5000)
    assert np.abs(pl.render(raw, analysis, settings, profile) - whole).max() < 1e-6


def test_render_preview_size_is_fast(library):
    """Contract: a 1512x1006 preview inverts in under 150 ms (550 ms on whole
    float arrays), so every inversion slider keeps up with the mouse. Measured
    while dragging Contraste, which rebuilds the print table on every step."""
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    if load > (os.cpu_count() or 1) / 3:
        pytest.skip(f"machine busy (load {load:.1f}), timing would say nothing")
    profile = library.get("kodak-portra-400")
    small, _ = synth.synthetic_negative(profile, scene=synth.landscape_scene(450, 300, seed=1))
    settings = pl.DevelopSettings(profile_id=profile.id)
    analysis = pl.analyze(small, settings, profile)
    raw = cv2.resize(small, (1512, 1006), interpolation=cv2.INTER_LINEAR)
    pl.render(raw, analysis, settings, profile)  # start the threads
    best = min(_timed(lambda: pl.render(raw, analysis, settings.copy(contrast=1.0 + 0.01 * i), profile))
               for i in range(1, 7))
    assert best < 0.15, f"{best * 1000:.0f} ms"


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
