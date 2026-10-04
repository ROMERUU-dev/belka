import os
import time
import warnings

import cv2
import numpy as np
import pytest

from belka.core import dust, synth
from belka.core import pipeline as pl


def frame_box(bright, library, profile_id):
    """The detected picture, as a mask, slightly inside its edges."""
    profile = library.get(profile_id)
    analysis = pl.analyze(bright, pl.DevelopSettings(profile_id=profile_id), profile)
    x0, y0, x1, y1 = analysis.extra["frame"]
    box = np.zeros(bright.shape[:2], dtype=bool)
    box[pl.crop_slices(bright.shape, (x0, y0, x1, y1), inset=0.02)] = True
    return box


def objects(truth, where):
    """Labelled defects that lie wholly inside ``where``."""
    n, labels = cv2.connectedComponents(truth.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n) if where[labels == i].all()]


def recall(mask, truth, where):
    found = [obj for obj in objects(truth, where) if mask[obj].mean() > 0.5]
    return len(found), len(objects(truth, where))


def false_spots(mask, truth, slack=4):
    """Components of the mask that touch no defect (give or take ``slack`` px)."""
    near = cv2.dilate(truth.astype(np.uint8), np.ones((2 * slack + 1,) * 2, np.uint8)).astype(bool)
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return sum(1 for i in range(1, n) if not near[labels == i].any())


def spots(mask):
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1


def shifted(img, dx, dy):
    """``img`` moved by a (sub-)pixel offset, as the camera can between two shots."""
    move = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, move, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def picture_size(height=1000):
    """The picture's size inside a ``dusty_scan`` of this height (for scenes made to fit it)."""
    ph = int(0.46 * height / (1 + 2 * 0.126))
    return int(ph * 1.5), ph


def busy() -> str | None:
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    return f"machine busy (load {load:.1f}), timing would say nothing" if load > (os.cpu_count() or 1) / 3 else None


@pytest.fixture(scope="module")
def dusty(library):
    bright, dark, truth = synth.dusty_scan(library.get("kodak-portra-400"), seed=1)
    clean, clean_dark, _ = synth.dusty_scan(library.get("kodak-portra-400"), specks=0, fibres=0, scratches=0, seed=1)
    return {"bright": bright, "dark": dark, "truth": truth, "clean": clean, "clean_dark": clean_dark,
            "frame": frame_box(bright, library, "kodak-portra-400")}


@pytest.fixture(scope="module")
def silver(library):
    bright, dark, truth = synth.dusty_scan(library.get("ilford-hp5-plus"), seed=2)
    return {"bright": bright, "dark": dark, "truth": truth, "frame": frame_box(bright, library, "ilford-hp5-plus")}


@pytest.fixture(scope="module")
def opaque(dusty):
    """Opaque dust (as most real dust is) on the clean scan's picture."""
    bright, truth = synth.opaque_specks(dusty["clean"], dusty["frame"], 40, seed=2)
    return {"bright": bright, "truth": truth, "frame": dusty["frame"], "clean": dusty["clean"]}


# ---------------------------------------------------------------- the synthetic pair

def test_dusty_scan_soils_where_it_says(dusty):
    """Same scene and grain with and without the dirt: the dirt darkens the
    bright shot and glows in the dark field, exactly at the truth mask."""
    truth, bright, clean = dusty["truth"], dusty["bright"], dusty["clean"]
    assert truth.dtype == bool and truth.shape == bright.shape[:2] and 0.0005 < truth.mean() < 0.02
    assert bright.dtype == np.float32 and dusty["dark"].shape == bright.shape
    core = cv2.erode(truth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & dusty["frame"]
    darker = np.log(clean[core] + 1e-5) - np.log(bright[core] + 1e-5)
    assert np.median(darker) > 0.5
    away = ~cv2.dilate(truth.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & dusty["frame"]
    assert np.median(dusty["dark"][core].mean(axis=-1)) > 3 * np.median(dusty["dark"][away].mean(axis=-1))


def test_opaque_specks_block_the_light(opaque, dusty):
    truth, frame = opaque["truth"], opaque["frame"]
    assert truth.any() and (truth & frame).sum() > 0.95 * truth.sum()
    core = cv2.erode(truth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert np.median(opaque["bright"][core] / dusty["clean"][core]) < 0.1
    changed = np.abs(np.log((opaque["bright"] + 1e-5) / (dusty["clean"] + 1e-5))).max(axis=-1) > 0.25
    assert not (changed & ~cv2.dilate(truth.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)).any()


# ---------------------------------------------------------------- with a dark-field shot

@pytest.mark.parametrize("pair", ["dusty", "silver"])
def test_darkfield_finds_dust_fibres_and_scratches(pair, request):
    data = request.getfixturevalue(pair)
    mask = dust.dust_mask(data["bright"], data["dark"], 0.5)
    found, total = recall(mask, data["truth"], data["frame"])
    assert total >= 20 and found >= 0.95 * total, f"found {found} of {total}"
    assert false_spots(mask, data["truth"]) == 0
    near = cv2.dilate(data["truth"].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    assert (mask & near).sum() / mask.sum() > 0.97  # tight around the dirt, halo included


@pytest.mark.parametrize("profile_id", ["kodak-portra-400", "ilford-hp5-plus", "fujifilm-superia-xtra-400"])
def test_darkfield_of_a_clean_strip_finds_nothing(library, profile_id):
    """The faint film image, the glow near the ring, the glowing cut edges and
    sprocket holes and the long exposure's hot pixels are not dirt."""
    bright, dark, _ = synth.dusty_scan(library.get(profile_id), specks=0, fibres=0, scratches=0, seed=11)
    assert not dust.dust_mask(bright, dark, 0.5).any()


@pytest.mark.parametrize("dx, dy", [(0.3, 0.4), (-0.5, 0.25), (2.6, -1.7)])
def test_darkfield_tolerates_the_camera_moving_between_shots(dusty, dx, dy):
    """The dark field is a second, slower exposure: a fraction of a pixel or a
    few pixels off, a little softer. Film edges must not glow, and the dirt
    must still be found where it is in the bright shot."""
    soft = cv2.blur(shifted(dusty["clean_dark"], dx, dy), (3, 3))
    for strength in (0.5, 1.0):
        assert spots(dust.dust_mask(dusty["clean"], soft, strength)) == 0
    mask = dust.dust_mask(dusty["bright"], shifted(dusty["dark"], dx, dy), 0.5)
    found, total = recall(mask, dusty["truth"], dusty["frame"])
    assert found >= 0.95 * total, f"found {found} of {total}"
    assert false_spots(mask, dusty["truth"]) == 0


def test_darkfield_of_another_picture_is_not_used(library, dusty):
    """A dark field that shows another picture (or does not line up) would
    put that picture's dirt here: the bright shot is searched alone instead."""
    width, height = picture_size()
    _, other, _ = synth.dusty_scan(library.get("kodak-portra-400"), scene=synth.colorchecker_scene(width, height),
                                   seed=7)
    assert dust.darkfield_matches(dusty["bright"], dusty["dark"])
    assert not dust.darkfield_matches(dusty["bright"], other)
    assert np.array_equal(dust.dust_mask(dusty["bright"], other, 0.5), dust.dust_mask(dusty["bright"], None, 0.5))


# What LibRaw's AHD demosaic makes of one hot photosite (measured on a real
# NEF): a red or blue one spreads over 3x3 pixels of its channel, a green one
# into all three channels.
HOT_RB = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], np.float32)
HOT_G = [np.array([[0, 0, 0], [0.5, 1.0, -0.2], [0, 0, 0]], np.float32),
         np.array([[0, 0, 0], [0.5, 1.0, 0], [0, 0, 0]], np.float32),
         np.array([[-0.13, -0.13, -0.12], [0, 0.5, 0], [-0.12, -0.12, -0.12]], np.float32)]


def test_hot_pixels_of_a_full_size_decode_are_not_dust(dusty):
    """On a demosaiced full-size dark field a hot photosite covers a few pixels
    and channels; dirt is told from it by the shadow it casts on the bright shot."""
    size = (3400, 2267)
    bright, dark, clean, clean_dark = (cv2.resize(dusty[k], size, interpolation=cv2.INTER_LINEAR)
                                       for k in ("bright", "dark", "clean", "clean_dark"))
    truth = cv2.resize(dusty["truth"].view(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    frame = cv2.resize(dusty["frame"].view(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    rng = np.random.default_rng(5)
    ys, xs = np.nonzero(cv2.erode(frame.view(np.uint8), np.ones((9, 9), np.uint8)))
    for i in rng.choice(ys.size, 40, replace=False):
        win = (slice(ys[i] - 1, ys[i] + 2), slice(xs[i] - 1, xs[i] + 2))
        amp, kind = rng.uniform(0.02, 0.3), rng.integers(3)
        for d in (dark, clean_dark):
            if kind < 2:
                d[win + (2 * kind,)] += amp * HOT_RB
            else:
                for c in range(3):
                    d[win + (c,)] += amp * HOT_G[c]
    for strength in (0.5, 1.0):
        assert spots(dust.dust_mask(clean, clean_dark, strength)) == 0
    mask = dust.dust_mask(bright, dark, 0.5)
    found, total = recall(mask, truth, frame)
    assert found >= 0.95 * total, f"found {found} of {total}"
    assert false_spots(mask, truth) == 0


def test_darkfield_must_match_the_frame(dusty):
    with pytest.raises(ValueError):
        dust.dust_mask(dusty["bright"], dusty["dark"][:-1], 0.5)


# ---------------------------------------------------------------- the bright shot alone

def test_image_only_finds_opaque_specks(opaque):
    """Without a dark field only what is denser than the picture can be counts:
    every opaque speck that covers a few pixels, and nothing else."""
    mask = dust.dust_mask(opaque["bright"], None, 0.5)
    n, labels = cv2.connectedComponents(opaque["truth"].astype(np.uint8), connectivity=8)
    sizable = [labels == i for i in range(1, n) if (labels == i).sum() >= 10]
    assert len(sizable) >= 15 and all(mask[obj].mean() > 0.5 for obj in sizable)
    assert false_spots(mask, opaque["truth"]) == 0


def test_image_only_at_full_strength_takes_half_transparent_dirt(library):
    bright, _, truth = synth.dusty_scan(library.get("kodak-portra-400"), specks=40, fibres=5, scratches=0, seed=2)
    frame = frame_box(bright, library, "kodak-portra-400")
    mask = dust.dust_mask(bright, None, 1.0)
    found, total = recall(mask, truth, frame)
    assert total >= 25 and found >= 0.8 * total, f"found {found} of {total}"
    assert false_spots(mask, truth) <= 1


@pytest.mark.parametrize("profile_id", ["kodak-portra-400", "ilford-hp5-plus", "fujifilm-superia-xtra-400"])
def test_image_only_leaves_grain_and_picture_alone(library, profile_id):
    bright, _, _ = synth.dusty_scan(library.get(profile_id), specks=0, fibres=0, scratches=0, seed=3)
    assert not dust.dust_mask(bright, None, 0.5).any()
    frame = frame_box(bright, library, profile_id)
    assert dust.dust_mask(bright, None, 1.0).sum() < 0.002 * frame.sum()


@pytest.mark.parametrize("profile_id", ["kodak-portra-400", "ilford-hp5-plus"])
@pytest.mark.parametrize("scene", ["stars", "bright stars", "town lights", "catchlights"])
def test_image_only_leaves_point_highlights_alone(library, profile_id, scene):
    """A star, a catchlight or a glint is a small, sharp, neutral, dense dot on
    the negative, like a speck of dust on the capture; only its density, which
    the film bounds, tells them apart."""
    width, height = picture_size()
    picture = {"stars": lambda: synth.night_sky_scene(width, height),
               "bright stars": lambda: synth.night_sky_scene(width, height, brightest=40.0, seed=1),
               "town lights": lambda: synth.night_sky_scene(width, height, warm=True, seed=2),
               "catchlights": lambda: synth.catchlight_scene(width, height)}[scene]()
    bright, _, _ = synth.dusty_scan(library.get(profile_id), scene=picture, specks=0, fibres=0, scratches=0, seed=3)
    assert not dust.dust_mask(bright, None, 0.5).any()


def test_image_only_stays_off_the_rebate(dusty):
    """Edge print, frame numbers and DX code are dark marks on the rebate too,
    so without a dark field only the picture is searched."""
    mask = dust.dust_mask(dusty["bright"], None, 1.0)
    outside = dusty["truth"] & ~cv2.dilate(dusty["frame"].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    assert outside.sum() > 50
    assert not (mask & outside).any()


@pytest.mark.parametrize("pad", [0.06, 0.065])
def test_tight_framing_is_searched_all_over(library, dusty, pad):
    """Picture plus a little rebate and no bare light in view (a carrier): the
    thin parts of the negative are film, not light, and are searched too."""
    x0, y0, x1, y1 = pl.analyze(dusty["bright"], pl.DevelopSettings(profile_id="kodak-portra-400"),
                                library.get("kodak-portra-400")).extra["frame"]
    h, w = dusty["bright"].shape[:2]
    py, px = int((y1 - y0) * h * pad), int((x1 - x0) * w * pad)
    crop = (slice(int(y0 * h) - py, int(y1 * h) + py), slice(int(x0 * w) - px, int(x1 * w) + px))
    k = 2000 / (crop[1].stop - crop[1].start)

    def tight(img, interpolation=cv2.INTER_LINEAR):
        return cv2.resize(np.ascontiguousarray(img[crop]), None, fx=k, fy=k, interpolation=interpolation)

    picture = tight(dusty["frame"].view(np.uint8), cv2.INTER_NEAREST).astype(bool)
    truth = tight(dusty["truth"].view(np.uint8), cv2.INTER_NEAREST).astype(bool)
    mask = dust.dust_mask(tight(dusty["bright"]), tight(dusty["dark"]), 0.5)
    found, total = recall(mask, truth, picture)
    assert total >= 30 and found >= 0.9 * total, f"found {found} of {total}"
    soiled, specks = synth.opaque_specks(tight(dusty["clean"]), picture, 30, seed=4)
    mask = dust.dust_mask(soiled, None, 0.5)
    lower = np.zeros_like(picture)
    lower[picture.shape[0] // 2:] = True  # grass and shadows: the thin half of the negative
    sizable = [obj for obj in objects(specks, picture & lower) if obj.sum() >= 20]
    assert len(sizable) >= 5 and all(mask[obj].mean() > 0.5 for obj in sizable)


@pytest.mark.parametrize("mode", ["darkfield", "image"])
def test_strength_only_adds(library, mode):
    """Every step up the slider keeps what the step below found and adds to it."""
    bright, dark, _ = synth.dusty_scan(library.get("fujifilm-superia-xtra-400"), seed=5)
    dark = dark if mode == "darkfield" else None
    masks = [dust.dust_mask(bright, dark, s) for s in np.round(np.arange(0.0, 1.0001, 0.02), 2)]
    assert not masks[0].any() and masks[-1].sum() > masks[25].sum() > 0
    for step, (weaker, stronger) in enumerate(zip(masks, masks[1:])):
        assert not (weaker & ~stronger).any(), f"lost pixels going to {0.02 * (step + 1):.2f}"


def test_local_statistics_never_go_negative(dusty):
    """Interpolating a local mean of squares can dip a hair below zero; its
    square root must not turn a pixel into NaN (and so never into dirt)."""
    crop = (slice(308, 688), slice(468, 1028))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bright, dark = (np.ascontiguousarray(dusty[k][crop]) for k in ("bright", "dark"))
        mask = dust.dust_mask(bright, dark, 0.5)
    truth = dusty["truth"][crop]
    found, total = recall(mask, truth, dusty["frame"][crop])
    assert found >= 0.9 * total
    values = np.zeros((40, 40), np.float32)
    values[20:, :] = 1e-3
    assert np.all(dust._rms(values, np.ones_like(values), 24.0) >= 0)


# ---------------------------------------------------------------- repair

def test_repair_touches_only_the_mask(dusty):
    bright = dusty["bright"]
    before = bright.copy()
    rng = np.random.default_rng(0)
    mask = np.zeros(bright.shape[:2], dtype=bool)
    for y, x in rng.integers(0, bright.shape[:2], (40, 2)):
        cv2.circle(mask.view(np.uint8), (int(x), int(y)), int(rng.integers(1, 6)), 1, -1)
    out = dust.repair(bright, mask)
    assert np.array_equal(bright, before)  # the input is never written
    assert out.dtype == bright.dtype and out.shape == bright.shape
    assert np.array_equal(out[~mask], bright[~mask])
    assert np.all(np.isfinite(out)) and not np.array_equal(out[mask], bright[mask])
    assert np.array_equal(dust.repair(bright, np.zeros_like(mask)), bright)


def fine(img):
    """The grain (and any edge) of an image: its log minus a blur."""
    log = np.log(img + 1e-5)
    return log - cv2.GaussianBlur(log, (0, 0), 1.5)


def test_repair_looks_like_the_film_underneath(dusty):
    """Holes punched in a clean scan come back at the right level and as
    grainy as the film around them: no smooth patches."""
    clean, frame = dusty["clean"], dusty["frame"]
    rng = np.random.default_rng(4)
    ys, xs = np.nonzero(cv2.erode(frame.astype(np.uint8), np.ones((41, 41), np.uint8)))
    mask = np.zeros(clean.shape[:2], dtype=bool)
    for i in rng.choice(ys.size, 60, replace=False):
        cv2.circle(mask.view(np.uint8), (int(xs[i]), int(ys[i])), int(rng.integers(3, 8)), 1, -1)
    out = dust.repair(clean, mask)
    log_in, log_out = np.log(clean + 1e-5), np.log(out + 1e-5)
    fine_in, fine_out = fine(clean), fine(out)
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    level, texture = [], []
    for i in range(1, n):
        spot = labels == i
        level.append(np.abs(log_out[spot].mean(axis=0) - log_in[spot].mean(axis=0)).max())
        texture.append(fine_out[spot].std() / fine_in[spot].std())
    assert np.median(level) < 0.05 and np.percentile(level, 90) < 0.15
    assert 0.75 < np.median(texture) < 1.3, np.median(texture)


def test_repair_keeps_grain_along_long_dirt(dusty):
    """What the dark field finds here includes a fibre and two scratches that
    run into one another: repaired on the clean scan, every stretch of them
    is as grainy as the film (stretches across a strong edge of the picture
    aside, where the clean film's fine detail is the edge itself)."""
    mask = dust.dust_mask(dusty["bright"], dusty["dark"], 0.5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask.view(np.uint8), np.ones((9, 9), np.uint8)))
    assert stats[1:, cv2.CC_STAT_AREA].max() > 3000  # the tangle
    fine_in, fine_out = fine(dusty["clean"]), fine(dust.repair(dusty["clean"], mask))
    ratios, grain = [], []
    for ty in range(0, mask.shape[0], 24):
        for tx in range(0, mask.shape[1], 24):
            tile = (slice(ty, ty + 24), slice(tx, tx + 24))
            piece = mask[tile]
            if piece.sum() >= 30:
                ratios.append(fine_out[tile][piece].std() / fine_in[tile][piece].std())
                grain.append(fine_in[tile][piece].std())
    flat = np.array(ratios)[np.array(grain) < 1.5 * np.median(grain)]
    assert flat.size > 40
    assert np.median(flat) > 0.85 and flat.min() > 0.6, (np.median(flat), flat.min())


def test_remove_dust_gives_back_the_clean_film(dusty, opaque):
    for bright, dark, truth, clean, gain in ((dusty["bright"], dusty["dark"], dusty["truth"], dusty["clean"], 0.25),
                                             (opaque["bright"], None, opaque["truth"], opaque["clean"], 0.35)):
        soiled = truth & dusty["frame"]
        repaired, mask = dust.remove_dust(bright, dark, 0.5)
        assert mask.dtype == bool and mask.shape == truth.shape
        assert np.array_equal(repaired[~mask], bright[~mask])
        error_before = np.abs(np.log(bright[soiled] + 1e-5) - np.log(clean[soiled] + 1e-5)).mean()
        error_after = np.abs(np.log(repaired[soiled] + 1e-5) - np.log(clean[soiled] + 1e-5)).mean()
        assert error_after < gain * error_before


def test_strength_zero_is_a_copy(dusty):
    repaired, mask = dust.remove_dust(dusty["bright"], dusty["dark"], 0.0)
    assert not mask.any()
    assert repaired is not dusty["bright"] and np.array_equal(repaired, dusty["bright"])


def test_large_frames_are_searched_smaller(dusty):
    """Beyond the half-size decode the search runs downsampled; the mask
    still covers the dirt at full size."""
    size = (3400, int(3400 * dusty["bright"].shape[0] / dusty["bright"].shape[1]))
    bright = cv2.resize(dusty["bright"], size, interpolation=cv2.INTER_LINEAR)
    dark = cv2.resize(dusty["dark"], size, interpolation=cv2.INTER_LINEAR)
    truth = cv2.resize(dusty["truth"].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    frame = cv2.resize(dusty["frame"].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    mask = dust.dust_mask(bright, dark, 0.5)
    assert mask.shape == truth.shape
    found, total = recall(mask, truth, frame)
    assert found >= 0.95 * total


# ---------------------------------------------------------------- speed

def _best(fn, runs: int = 4, cold: bool = True) -> float:
    """Best of a few runs; cold ones reuse nothing measured on the frame before."""
    times = []
    for _ in range(runs):
        if cold:
            dust._forget()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


@pytest.mark.parametrize("dirt", ["typical", "heavy"])
def test_preview_is_fast(library, dirt):
    """Contract on a 2000 px preview, mask and repair together: a step of the
    strength slider in under 150 ms, even on a heavily soiled frame; the first
    call on a frame, which also measures the film, in under 200 ms (typical
    dirt) or 250 ms (heavy)."""
    if reason := busy():
        pytest.skip(reason)
    specks, fibres, scratches = (25, 2, 1) if dirt == "typical" else (80, 8, 2)
    profile = library.get("kodak-portra-400")
    bright, dark, _ = synth.dusty_scan(profile, size=(2000, 1333), specks=specks, fibres=fibres, scratches=scratches,
                                       seed=8)
    analysis = pl.analyze(bright, pl.DevelopSettings(profile_id=profile.id), profile)  # the app has it already
    first = 0.2 if dirt == "typical" else 0.25
    for darkfield in (dark, None):
        cold = _best(lambda: dust.remove_dust(bright, darkfield, 0.5, analysis))
        step = _best(lambda: dust.remove_dust(bright, darkfield, 0.6, analysis), cold=False)
        assert cold < first, f"first call {cold * 1000:.0f} ms"
        assert step < 0.15, f"strength step {step * 1000:.0f} ms"


def test_24_megapixels_in_under_3_seconds(library):
    """A full-size export of a heavily soiled frame, first call."""
    if reason := busy():
        pytest.skip(reason)
    bright, dark, _ = synth.dusty_scan(library.get("kodak-portra-400"), size=(2000, 1333), specks=150, fibres=15,
                                       scratches=4, seed=8)
    bright = cv2.resize(bright, (6048, 4024), interpolation=cv2.INTER_LINEAR)
    dark = cv2.resize(dark, (6048, 4024), interpolation=cv2.INTER_LINEAR)
    assert _best(lambda: dust.remove_dust(bright, dark, 0.5), runs=1) < 3.0
