"""DX film edge barcodes: synthetic strips drawn to the format, decoded exactly."""

import dataclasses
import json
import re
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from belka import paths
from belka.core import edgeprint as ep
from tests import dx_synth as ds

# The data track of the code after "21A" on a real Kodak UltraMax 400 strip
# (a real capture), as Belka read it: DX 95-7, frame 21, half frame.
REAL_21A = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]


def _analysis(geo: dict) -> SimpleNamespace:
    """What read_edge uses of a pipeline.Analysis: the detected frame."""
    return SimpleNamespace(extra={"frame": geo["frame"]})


def _check(strip: ds.Strip, use_frame: bool = True) -> ep.EdgeInfo:
    """Every code wholly in view reads exactly; returns read_edge's answer."""
    img, geo = ds.render(strip)
    analysis = _analysis(geo) if use_frame else None
    codes = ep.find_barcodes(img, analysis)
    dx = f"{strip.part1}-{strip.part2}"
    assert {c.dx for c in codes} == {dx}
    assert {c.mirrored for c in codes} == {strip.mirrored}
    labels = [c.label for c in codes]
    assert set(g[0] for g in geo["codes"]) <= set(labels)
    if strip.numbered:
        assert len(labels) == len(set(labels))  # each instance once
    info = ep.read_edge(img, analysis)
    assert info is not None and info.dx == dx and info.mirrored == strip.mirrored
    assert info.frame == geo["label_at_frame" if use_frame else "label_at_centre"]
    return info


# ---------------------------------------------------------------- the format

def test_synthetic_codes_follow_the_real_strip():
    clock, data = ds.code_bits(95, 7, 21, True)
    assert data == REAL_21A
    assert clock == [1] * 5 + [0, 1] * 11 + [0] + [1] * 3
    clock, data = ds.code_bits(64, 5)
    assert len(clock) == len(data) == 23 and data[-4:] == [0, 1, 0, 1]


def test_layouts_agree_with_the_drawn_codes():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p1, p2, n, half = int(rng.integers(1, 128)), int(rng.integers(0, 16)), int(rng.integers(0, 64)), bool(
            rng.integers(0, 2))
        for layout, number in ((ep.LAYOUTS[0], n), (ep.LAYOUTS[1], None)):
            clock, data = ds.code_bits(p1, p2, number, half)
            assert len(data) == layout.modules
            assert all(data[k] == bit for k, bit in layout.known)
            edges = [0] + [k for k in range(1, len(clock)) if clock[k] != clock[k - 1]] + [len(clock)]
            assert tuple(edges) == layout.clock


# ---------------------------------------------------------------- reading

@pytest.mark.parametrize("angle, mirrored, use_frame", [
    (0, False, True), (0, True, True), (90, False, True), (90, True, False), (180, False, False),
    (270, True, True), (2.5, False, True), (-93, True, False),
])
def test_reads_any_orientation_and_face(angle, mirrored, use_frame):
    _check(ds.Strip(angle=angle, mirrored=mirrored, seed=int(angle) % 7), use_frame)


@pytest.mark.parametrize("px_per_mm, view", [(7.0, (130, 50)), (12.0, (100, 50)), (30.0, (60, 45)), (60.0, (40, 44))])
def test_reads_from_small_to_large_modules(px_per_mm, view):
    info = _check(ds.Strip(px_per_mm=px_per_mm, view_mm=view, shift_mm=4.0))
    assert info.confidence > 0.5


@pytest.mark.parametrize("strip", [
    ds.Strip(blur=1.5, noise=0.03, angle=1.0),
    ds.Strip(contrast=0.2, noise=0.01, blur=0.8, angle=-1.0),  # bars 0.09 D above the base
    ds.Strip(angle=12.0, noise=0.005, blur=0.7),
], ids=["blur-noise", "low-contrast", "tilted"])
def test_reads_through_blur_noise_and_low_contrast(strip):
    _check(strip)


@pytest.mark.parametrize("kind, blur", [("negative", 0.4), ("bw", 0.4), ("slide", 0.35)])
def test_reads_through_defocus(kind, blur):
    # Gaussian blur of that much of a module (8 px here): a single bar next to
    # the 3-module one, and dense or clear bars blurred in light, stay readable.
    strip = ds.Strip(kind=kind, px_per_mm=20.0, view_mm=(100, 66.6), blur=blur * ds.MODULE_MM * 20.0, noise=0.01,
                     angle=1.5)
    _check(strip)


@pytest.mark.parametrize("strip, use_frame", [
    (ds.Strip(stripes_mm=0.6, noise=0.005), False),
    (ds.Strip(stripes_mm=0.5, px_per_mm=20.0, view_mm=(100, 66.6), noise=0.01, blur=0.8), False),
    (ds.Strip(stripes_mm=0.66, kind="bw", px_per_mm=20.0, view_mm=(60, 50), angle=7.0, noise=0.02, blur=2.0), False),
    (ds.Strip(stripes_mm=0.8, px_per_mm=8.7, view_mm=(230, 40)), True),  # six frames, twelve codes
], ids=["fence", "fence-preview", "fence-tilted-noisy", "whole-strip"])
def test_a_regular_texture_in_the_pictures_does_not_hide_the_codes(strip, use_frame):
    # A picket fence about as fine as the clock makes many more and longer
    # runs of equal bars than the codes, all over the pictures.
    _check(strip, use_frame)


def test_without_the_frame_the_image_centre_picks_the_number():
    for shift in (0.0, 7.0, 12.0):
        img, geo = ds.render(ds.Strip(angle=90, shift_mm=shift))
        assert ep.read_edge(img).frame == geo["label_at_centre"]


@pytest.mark.parametrize("shift", [0.0, 9.5, 19.0, -14.0])
def test_frame_number_follows_the_strip(shift):
    _check(ds.Strip(shift_mm=shift, angle=180 if shift < 0 else 0))


def test_black_and_white_slide_and_old_codes():
    info = _check(ds.Strip(kind="bw", part1=109, part2=9))
    assert (info.film, info.profile_id) == ("Ilford HP5 Plus", "ilford-hp5-plus")
    info = _check(ds.Strip(kind="slide", part1=34, part2=13, angle=-90))  # clear bars on a black rebate
    assert info.profile_id == "fujifilm-provia-100f"
    info = _check(ds.Strip(numbered=False, part1=64, part2=5))
    assert (info.film, info.frame) == ("Kodak Tri-X 400", "")


# ---------------------------------------------------------------- rejection

def test_nothing_reads_without_a_code():
    img, _ = ds.render(ds.Strip(codes=False, stripes_mm=0.8))  # holes, edge print and a picket fence
    assert ep.read_edge(img) is None
    rng = np.random.default_rng(1)
    assert ep.read_edge(rng.random((800, 1200, 3)).astype(np.float32) * 0.5) is None
    assert ep.read_edge(np.zeros((600, 900, 3), np.float32)) is None
    assert ep.read_edge(np.ones((12, 30), np.float32)) is None


def test_a_wrong_parity_is_rejected():
    img, _ = ds.render(ds.Strip(parity_error=True))
    assert ep.find_barcodes(img) == []


def test_a_broken_start_pattern_is_rejected():
    # Whole clock tracks, but ink over the start of every data track.
    img, _ = ds.render(ds.Strip(part1=1, part2=0))
    good = ep.find_barcodes(img)
    assert good and {c.dx for c in good} == {"1-0"}
    bad = img.copy()
    for c in good:
        start, m = np.array(c.start), c.module
        u = (np.array(c.end) - start) / np.linalg.norm(np.array(c.end) - start)
        v = np.array([-u[1], u[0]]) * (-1 if c.mirrored else 1)  # towards the data track
        corners = [start + u * t + v * s for t, s in ((-m, m), (6 * m, m), (6 * m, 3.4 * m), (-m, 3.4 * m))]
        cv2.fillConvexPoly(bad, np.round(corners).astype(np.int32), (0.0, 0.0, 0.0))
    assert ep.find_barcodes(bad) == []


# ---------------------------------------------------------------- frame number

def _code(number, half, start, end) -> ep.Barcode:
    return ep.Barcode(95, 7, number, half, start, end, False, 1.0)


def test_frame_label_steps_half_frames_from_the_codes():
    module = 10.0
    pitch = ep.HALF_FRAME_MODULES * module
    codes = [_code(21, False, (100.0, 50.0), (100.0 + 31 * module, 50.0)),
             _code(21, True, (100.0 + pitch, 50.0), (100.0 + pitch + 31 * module, 50.0))]
    label = 100.0 - ep.LABEL_GAP_MODULES * module  # where "21" is printed
    assert ep.frame_label(codes, (label, 300.0)) == ("21", 1.0)
    assert ep.frame_label(codes, (label + 0.6 * pitch, 300.0))[0] == "21A"
    assert ep.frame_label(codes, (label + 2.4 * pitch, 300.0))[0] == "22"  # beyond the last code
    assert ep.frame_label(codes, (label + 2.6 * pitch, 300.0))[0] == "22A"
    assert ep.frame_label(codes, (label - 1.2 * pitch, 300.0))[0] == "20A"
    # Read the other way (strip turned round), numbers grow to the left.
    back = [_code(5, False, (500.0, 50.0), (500.0 - 31 * module, 50.0))]
    assert ep.frame_label(back, (500.0 + ep.LABEL_GAP_MODULES * module - pitch, 0.0))[0] == "5A"
    assert ep.frame_label([_code(0, False, (0.0, 0.0), (310.0, 0.0))], (-3 * pitch, 0.0))[0] == ""
    unnumbered = ep.Barcode(64, 5, None, False, (0.0, 0.0), (230.0, 0.0), False, 1.0)
    assert ep.frame_label([unnumbered], (0.0, 0.0)) == ("", 1.0)


def test_frame_label_needs_most_codes_to_agree():
    module = 10.0
    pitch = ep.HALF_FRAME_MODULES * module
    at = [(100.0 + j * pitch, 50.0) for j in range(4)]
    codes = [_code(n, half, start, (start[0] + 31 * module, 50.0))
             for (n, half), start in zip([(20, True), (21, False), (37, True), (22, False)], at)]
    point = (at[2][0] - ep.LABEL_GAP_MODULES * module, 300.0)  # where "21A" is printed
    # The third code read 37A for 21A (two bits of its number turned over): the others outvote it.
    assert ep.frame_label(codes, point) == ("21A", 0.75)
    assert ep.frame_label(codes[1:3], point) == ("", 0.5)  # one against one: no number
    assert ep.frame_label(codes[2:3], point) == ("37A", 1.0)  # nothing to tell it by


def test_a_damaged_frame_number_is_outvoted():
    # Dust over two bits of the frame number keeps the parity: that instance reads 37A.
    img, geo = ds.render(ds.Strip(damage=("21A", (18, 19))))
    analysis = _analysis(geo)
    assert "37A" in {c.label for c in ep.find_barcodes(img, analysis)}
    info = ep.read_edge(img, analysis)
    assert info.frame == geo["label_at_frame"] == "21A"
    img, _ = ds.render(ds.Strip())
    assert info.confidence < 0.8 * ep.read_edge(img, analysis).confidence


def test_confidence_grows_with_agreeing_instances():
    img, geo = ds.render(ds.Strip(px_per_mm=60.0, view_mm=(40, 44), shift_mm=4.0))
    one = ep.read_edge(img)
    img, geo = ds.render(ds.Strip())
    many = ep.read_edge(img, _analysis(geo))
    assert len(geo["codes"]) >= 3
    assert 0.5 < one.confidence < many.confidence <= 1.0


def test_edge_info_is_plain_json():
    img, geo = ds.render(ds.Strip(mirrored=True))
    info = ep.read_edge(img, _analysis(geo))
    data = json.loads(json.dumps(dataclasses.asdict(info)))
    assert data == {"dx": "95-7", "film": "Kodak UltraMax 400", "profile_id": "kodak-ultramax-400",
                    "frame": geo["label_at_frame"], "confidence": info.confidence, "mirrored": True}


def test_preview_speed():
    img, geo = ds.render(ds.Strip(px_per_mm=20.0, view_mm=(100, 66.6), noise=0.01, blur=0.8))
    assert max(img.shape) == 2000
    for analysis in (None, _analysis(geo)):
        times = []
        for _ in range(3):
            start = time.perf_counter()
            assert ep.read_edge(img, analysis).dx == "95-7"
            times.append(time.perf_counter() - start)
        assert min(times) < 0.2


# ---------------------------------------------------------------- the film table

def test_lookup_names_the_film_and_its_profile():
    assert ep.lookup("95-7") == ("Kodak UltraMax 400", "kodak-ultramax-400")
    assert ep.lookup("64-5") == ("Kodak Tri-X 400", "kodak-tri-x-400")
    assert ep.lookup("96-7") == ("Kodak (DX 96-7)", None)  # Kodak's number, film not listed
    assert ep.lookup("126-3") == ("DX 126-3", None)
    assert ep.lookup("79-15") == ("Kodak BW400CN", None)
    assert ep.lookup("78-13") == ("Kodak Black & White+ 400 (BWC)", None)
    assert ep.lookup("95-12") == ("Kodak Ektar 100 / ProFoto XL 100", None)  # shared: no profile


def test_film_table_is_consistent(library):
    table = json.loads((paths.data_dir() / "dx_codes.json").read_text(encoding="utf-8"))
    sources = set(table["_sources"])
    for dx, entry in table["films"].items():
        part1, part2 = (int(v) for v in re.fullmatch(r"(\d+)-(\d+)", dx).groups())
        assert 1 <= part1 < 128 and 0 <= part2 < 16, dx
        assert str(part1) in table["makers"], dx
        assert entry["film"] and set(entry["sources"]) <= sources, dx
        if entry["profile"] is not None:
            assert entry["profile"] in library, dx
            assert not entry.get("uncertain"), dx  # a guess never picks the profile
    for part1 in table["makers"]:
        assert 1 <= int(part1) < 128
