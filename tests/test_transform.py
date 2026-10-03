import time

import cv2
import numpy as np
import pytest

from belka.core import transform as tf
from belka.core.film import FilmProfile
from belka.core.pipeline import DevelopSettings

W, H = 1500, 1000


def _lean_deg(d):
    """Clockwise lean of a direction from horizontal or vertical, whichever is nearer."""
    a = (np.degrees(np.arctan2(d[1], d[0])) + 90.0) % 180.0 - 90.0
    if abs(a) > 45:
        a = a - 90.0 if a > 0 else a + 90.0
    return a


def _homography(m3, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(m3).T
    return p[:, :2] / p[:, 2:]


# ---------------------------------------------------------------- synthetic scenes

def _horizon(tilt_deg, w=1800, h=1200, seed=0):
    """Sky over ground, the horizon leaning clockwise by ``tilt_deg``, with diagonal clutter and grain."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    a = np.radians(tilt_deg)
    below = np.clip(0.5 - (xx - w / 2) * np.sin(a) + (yy - 0.55 * h) * np.cos(a), 0, 1)
    img = (0.55 - 0.15 * yy / h) * (1 - below) + 0.12 * below
    clutter = np.zeros((h, w), np.float32)
    for _ in range(25):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang = np.radians(rng.uniform(35, 55) * rng.choice([-1, 1]))
        length = rng.uniform(40, 160)
        cv2.line(clutter, (int(x0), int(y0)), (int(x0 + length * np.cos(ang)), int(y0 + length * np.sin(ang))),
                 0.2, 3, cv2.LINE_AA)
    img = (img + clutter) * np.exp(rng.normal(0, 0.03, img.shape))
    return np.repeat(img[..., None], 3, axis=2).astype(np.float32)


def _facade(w=W, h=H):
    """A building seen straight on, and its vertical and horizontal edges in continuous pixels."""
    img = np.full((h, w), 0.5, np.float32)
    bx0, bx1, by0 = 350, 1150, 80
    img[by0:, bx0:bx1] = 0.12
    verticals = [((bx0, by0), (bx0, h)), ((bx1, by0), (bx1, h))]
    horizontals = [((bx0, by0), (bx1, by0))]
    for c in range(6):
        for r in range(5):
            x0, y0 = bx0 + 50 + c * 125, by0 + 60 + r * 170
            img[y0:y0 + 110, x0:x0 + 70] = 0.4
            verticals += [((x0, y0), (x0, y0 + 110)), ((x0 + 70, y0), (x0 + 70, y0 + 110))]
            horizontals += [((x0, y0), (x0 + 70, y0)), ((x0, y0 + 110), (x0 + 70, y0 + 110))]
    return img, verticals, horizontals


def _camera(img, tilt_deg, pan_deg=0.0, roll_deg=0.0, focal=0.9, zoom=0.8):
    """Photograph ``img`` with a camera tilted up, panned and rolled: returns the photo and the map.

    A pinhole turned about its centre, focal length as a fraction of the long
    side, then zoomed and rolled about the principal point.
    """
    h, w = img.shape[:2]
    f = focal * max(w, h)
    t, p, r = np.radians([tilt_deg, pan_deg, roll_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]])
    ry = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    k = np.diag([f, f, 1.0])
    roll = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    centre = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1.0]])
    g = centre @ roll @ np.diag([zoom, zoom, 1.0]) @ k @ rx @ ry @ np.linalg.inv(k) @ np.linalg.inv(centre)
    shift = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1.0]])
    gpix = shift @ g @ np.linalg.inv(shift)
    photo = cv2.warpPerspective(img, gpix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0.5)
    return np.repeat(photo[..., None], 3, axis=2), g


def _leans_after(edges, g, s, w=W, h=H):
    """Lean of each scene edge in the transformed photo."""
    out = []
    for p0, p1 in edges:
        q = _homography(g, np.array([p0, p1], float))
        o = tf.map_points_from_source(q / (w, h), s, w, h) * (w, h)
        out.append(_lean_deg(o[1] - o[0]))
    return np.array(out)


# ---------------------------------------------------------------- identity and switches

def test_defaults_are_the_identity():
    s = DevelopSettings()
    img = np.random.default_rng(0).random((60, 90, 3)).astype(np.float32)
    assert tf.is_identity(s)
    assert np.array_equal(tf.homography(s, 90, 60), np.eye(3))
    assert tf.lens_correct(img, s) is img
    assert tf.warp(img, s) is img
    out, mask = tf.warp(img, s, with_mask=True)
    assert out is img and mask.shape == (60, 90) and np.all(mask == 1)
    pts = np.array([[0.1, 0.2], [0.9, 0.7]])
    assert np.allclose(tf.map_points_to_source(pts, s, 90, 60), pts)
    assert np.allclose(tf.map_points_from_source(pts, s, 90, 60), pts)
    assert np.allclose(tf.largest_valid_rect(s, 90, 60), (0, 0, 1, 1), atol=1e-6)
    assert tf.map_rect_from_source((0.1, 0.2, 0.8, 0.9), s, 90, 60) == (0.1, 0.2, 0.8, 0.9)


def test_switched_off_sections_mean_identity():
    # Without an angle: the straighten angle survives the Transform switch (see below).
    busy = dict(persp_vertical=0.5, persp_x=0.2, lens_distortion=0.5, lens_vignette=0.5)
    img = np.random.default_rng(1).random((40, 60, 3)).astype(np.float32)
    off = DevelopSettings(**busy, disabled=("transform", "lens"))
    assert tf.is_identity(off)
    assert tf.warp(img, off) is img and tf.lens_correct(img, off) is img
    assert np.array_equal(tf.homography(off, 60, 40), np.eye(3))
    lens_only = DevelopSettings(**busy, disabled=("transform",))
    assert not tf.is_identity(lens_only)
    assert tf.warp(img, lens_only) is img
    assert tf.lens_correct(img, lens_only) is not img


# ---------------------------------------------------------------- directions

def test_positive_angle_turns_the_picture_clockwise():
    s = DevelopSettings(angle=10)
    right_of_centre = tf.map_points_from_source([(0.75, 0.5)], s, W, H)[0]
    assert right_of_centre[1] > 0.5 and right_of_centre[0] < 0.75
    # A level line on screen leaning 10° clockwise needs -10°, and that undoes it.
    p0, p1 = tf.map_points_from_source([(0.2, 0.5), (0.8, 0.5)], s, W, H) * (W, H)
    assert tf.angle_from_line(p0, p1) == pytest.approx(-10.0, abs=1e-6)


def test_keystone_and_aspect_directions():
    top = tf.map_points_from_source([(0.25, 0.1), (0.75, 0.1)], DevelopSettings(persp_vertical=0.5), W, H)
    bottom = tf.map_points_from_source([(0.25, 0.9), (0.75, 0.9)], DevelopSettings(persp_vertical=0.5), W, H)
    assert np.ptp(top[:, 0]) > 1.2 * np.ptp(bottom[:, 0]), "positive Vertical widens the top"
    s = DevelopSettings(persp_horizontal=0.5)
    left = tf.map_points_from_source([(0.1, 0.25), (0.1, 0.75)], s, W, H)
    right = tf.map_points_from_source([(0.9, 0.25), (0.9, 0.75)], s, W, H)
    assert np.ptp(right[:, 1]) > 1.2 * np.ptp(left[:, 1]), "positive Horizontal enlarges the right side"
    for key in ("persp_vertical", "persp_horizontal"):
        centre = tf.map_points_from_source([(0.5, 0.5)], DevelopSettings(**{key: 0.8}), W, H)[0]
        assert np.allclose(centre, (0.5, 0.5))
    square = [(0.25, 0.25), (0.75, 0.75)]
    tall = tf.map_points_from_source(square, DevelopSettings(persp_aspect=1.0), W, H)
    assert np.ptp(tall[:, 0]) == pytest.approx(0.5) and np.ptp(tall[:, 1]) == pytest.approx(0.75), "Lightroom: + is taller"
    wide = tf.map_points_from_source(square, DevelopSettings(persp_aspect=-1.0), W, H)
    assert np.ptp(wide[:, 0]) == pytest.approx(0.75) and np.ptp(wide[:, 1]) == pytest.approx(0.5)
    moved = tf.map_points_from_source([(0.5, 0.5)], DevelopSettings(persp_x=0.1, persp_y=0.1), W, H)[0]
    assert moved == pytest.approx((0.6, 0.4)), "positive X moves right, positive Y moves up"


def test_extreme_keystone_keeps_the_mapping_valid():
    s = DevelopSettings(persp_vertical=1.0, persp_horizontal=-1.0)
    for w, h in ((1000, 1000), (1500, 1000), (1000, 1500)):
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        fwd = tf.map_points_from_source(corners, s, w, h)
        assert np.all(np.isfinite(fwd))
        assert np.allclose(tf.map_points_to_source(fwd, s, w, h), corners, atol=1e-9)


# ---------------------------------------------------------------- mapping and resampling

MIXED = [
    dict(angle=7),
    dict(persp_vertical=0.8, persp_horizontal=-0.6, angle=-3, persp_rotate=4, persp_aspect=0.4, persp_scale=1.2,
         persp_x=0.1, persp_y=-0.05),
    dict(lens_distortion=0.7),
    dict(lens_distortion=-0.9, angle=2, persp_vertical=0.3),
    dict(persp_aspect=-0.7, persp_scale=0.7),
]


@pytest.mark.parametrize("kw", MIXED)
def test_mappings_invert_each_other(kw):
    s = DevelopSettings(**kw)
    pts = np.random.default_rng(2).uniform(0.05, 0.95, (300, 2))
    there = tf.map_points_from_source(pts, s, W, H)
    assert np.abs(tf.map_points_to_source(there, s, W, H) - pts).max() < 1e-9
    back = tf.map_points_to_source(pts, s, W, H)
    assert np.abs(tf.map_points_from_source(back, s, W, H) - pts).max() < 1e-9


def test_homography_uses_opencv_pixel_centres():
    s = DevelopSettings(angle=5, persp_vertical=0.3, persp_x=0.05)
    pix = np.array([[0.0, 0.0], [100.0, 200.0], [1499.0, 999.0]])
    expected = tf.map_points_to_source((pix + 0.5) / (W, H), s, W, H) * (W, H) - 0.5
    assert np.allclose(_homography(tf.homography(s, W, H), pix), expected)


def _centroid(lum, x, y, r=12):
    """Brightness centroid near (x, y), continuous pixels."""
    x0, y0 = int(x) - r, int(y) - r
    patch = lum[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
    ys, xs = np.mgrid[0:2 * r + 1, 0:2 * r + 1]
    return np.array([(patch * xs).sum() / patch.sum() + x0 + 0.5, (patch * ys).sum() / patch.sum() + y0 + 0.5])


def test_warp_puts_pixels_where_the_mapping_says():
    s = DevelopSettings(angle=4, persp_vertical=0.35, persp_horizontal=-0.2, lens_distortion=0.5, lens_vignette=0.3)
    w, h = 600, 400
    img = np.zeros((h, w, 3), np.float32)
    spots = np.array([[0.3, 0.3], [0.7, 0.35], [0.5, 0.6], [0.25, 0.7], [0.72, 0.68]]) * (w, h)
    for x, y in spots - 0.5:  # OpenCV draws in pixel-index coordinates
        cv2.circle(img, (round(x * 16), round(y * 16)), 3 * 16, (1, 1, 1), -1, cv2.LINE_AA, shift=4)
    measured = np.array([_centroid(img.mean(axis=2), x, y) for x, y in spots])
    out = tf.warp(tf.lens_correct(img, s), s)
    expected = tf.map_points_from_source(measured / (w, h), s, w, h) * (w, h)
    for ex, ey in expected:
        assert np.hypot(*(_centroid(out.mean(axis=2), ex, ey) - (ex, ey))) < 0.1


def test_warp_keeps_shape_dtype_and_blacks_out_the_outside():
    s = DevelopSettings(angle=20)
    for shape in ((80, 120), (80, 120, 1), (80, 120, 3)):
        img = np.full(shape, 0.5, np.float32)
        out, mask = tf.warp(img, s, with_mask=True)
        assert out.shape == img.shape and out.dtype == np.float32
        assert mask.shape == (80, 120) and mask.dtype == np.float32
        assert out[0, 0].max() == 0.0 and mask[0, 0] == 0.0
        assert out[40, 60].min() == pytest.approx(0.5) and mask[40, 60] == 1.0
        assert out.min() >= 0.0


def test_lens_distortion_and_vignetting():
    w, h = 300, 200
    corner = [(0.999, 0.999)]
    barrel_fix = tf.map_points_to_source(corner, DevelopSettings(lens_distortion=0.6), w, h)[0]
    pincushion_fix = tf.map_points_to_source(corner, DevelopSettings(lens_distortion=-0.6), w, h)[0]
    assert barrel_fix[0] < 0.999 and pincushion_fix[0] > 1.0
    img = np.full((h, w, 3), 0.4, np.float32)
    pinched = tf.lens_correct(img, DevelopSettings(lens_distortion=-0.6))
    _, mask = tf.warp(pinched, DevelopSettings(lens_distortion=-0.6), with_mask=True)
    assert pinched[0, 0].max() == 0.0 and mask[0, 0] == 0.0 and mask[h // 2, w // 2] == 1.0

    brighten = DevelopSettings(lens_vignette=0.8)
    plain = tf.lens_correct(img, brighten)
    assert plain[0, 0, 0] > 1.5 * plain[h // 2, w // 2, 0] and plain[h // 2, w // 2, 0] == pytest.approx(0.4, abs=0.01)
    negative = tf.lens_correct(img, brighten, FilmProfile(id="n", name="n"))
    slide = tf.lens_correct(img, brighten, FilmProfile(id="s", name="s", type="slide"))
    assert negative[0, 0, 0] < 0.4 < slide[0, 0, 0], "a brighter print corner is a denser negative corner"
    assert np.all(img == 0.4), "the input is never modified"


# ---------------------------------------------------------------- constrain crop

RECT_CASES = [
    dict(angle=7),
    dict(angle=-4, persp_vertical=0.4, persp_horizontal=-0.3, lens_distortion=-0.6),
    dict(lens_distortion=-1.0),
    dict(lens_distortion=0.8, persp_horizontal=0.5),
    dict(persp_rotate=5, persp_scale=0.8, persp_x=0.1, persp_y=0.1),
]


@pytest.mark.parametrize("kw", RECT_CASES)
@pytest.mark.parametrize("aspect", [None, 1.5, 1.0, 0.8])
def test_largest_valid_rect_holds_only_picture(kw, aspect):
    s = DevelopSettings(**kw)
    x0, y0, x1, y1 = tf.largest_valid_rect(s, W, H, aspect=aspect)
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    if aspect is not None:
        assert (x1 - x0) * W / ((y1 - y0) * H) == pytest.approx(aspect, rel=1e-4)

    def outline(scale):
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * scale, (y1 - y0) / 2 * scale
        t = np.linspace(0, 1, 200)
        return np.concatenate([np.column_stack([cx - hw + 2 * hw * t, np.full_like(t, cy - hh)]),
                               np.column_stack([cx - hw + 2 * hw * t, np.full_like(t, cy + hh)]),
                               np.column_stack([np.full_like(t, cx - hw), cy - hh + 2 * hh * t]),
                               np.column_stack([np.full_like(t, cx + hw), cy - hh + 2 * hh * t])])

    def inside(pts):
        src = tf.map_points_to_source(pts, s, W, H)
        corrected = src if tf._k1(s) == 0 else tf.map_points_to_source(pts, s.copy(lens_distortion=0.0), W, H)
        ok = np.all((src > -1e-6) & (src < 1 + 1e-6), axis=1)
        return ok & np.all((corrected > -1e-6) & (corrected < 1 + 1e-6), axis=1)

    assert inside(outline(1.0)).all()
    grown = outline(1.01)
    in_frame = np.all((grown >= 0) & (grown <= 1), axis=1)
    assert not (inside(grown) & in_frame).all() or not in_frame.all(), "a 1 % larger crop must touch the edge"

    _, mask = tf.warp(np.ones((H, W, 3), np.float32), s, with_mask=True)
    ix0, iy0, ix1, iy1 = int(np.ceil(x0 * W)) + 1, int(np.ceil(y0 * H)) + 1, int(x1 * W) - 1, int(y1 * H) - 1
    assert mask[iy0:iy1, ix0:ix1].min() > 0.99


FRAME = (0.1, 0.3, 0.85, 0.7)  # the picture, with film rebate and sprocket holes above and below
FRAME_CASES = [
    dict(angle=3),
    dict(angle=-6),
    dict(persp_vertical=0.3),
    dict(persp_horizontal=-0.4, persp_rotate=3),
    dict(lens_distortion=0.8),
    dict(lens_distortion=-0.8),
    dict(lens_distortion=0.6, angle=4, persp_horizontal=-0.4),
]


def _grid(rect, scale=1.0, n=81):
    """Points over a rectangle grown by ``scale`` about its centre, its edges included."""
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    t = np.linspace(-scale, scale, n)
    xs, ys = np.meshgrid(cx + t * (rect[2] - rect[0]) / 2, cy + t * (rect[3] - rect[1]) / 2)
    return np.column_stack([xs.ravel(), ys.ravel()])


def _inside(pts, rect, tol=1e-9):
    return ((pts[:, 0] >= rect[0] - tol) & (pts[:, 0] <= rect[2] + tol)
            & (pts[:, 1] >= rect[1] - tol) & (pts[:, 1] <= rect[3] + tol))


@pytest.mark.parametrize("kw", FRAME_CASES)
@pytest.mark.parametrize("aspect", ["frame", None, 1.0])
def test_map_rect_from_source_stays_inside_the_frame(kw, aspect):
    s = DevelopSettings(**kw)
    rect = tf.map_rect_from_source(FRAME, s, W, H, aspect=aspect)
    assert 0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1
    ratio = (FRAME[2] - FRAME[0]) * W / ((FRAME[3] - FRAME[1]) * H) if aspect == "frame" else aspect
    if ratio is not None:
        assert (rect[2] - rect[0]) * W / ((rect[3] - rect[1]) * H) == pytest.approx(ratio, rel=1e-4)
    # The crop shows only the picture: none of the rebate around the tilted frame.
    assert _inside(tf.map_points_to_source(_grid(rect), s, W, H), FRAME).all()
    # And it is the largest one: 1 % larger pokes out of the frame or the image.
    grown = _grid(rect, 1.01)
    in_view = np.all((grown >= 0) & (grown <= 1), axis=1)
    assert not (in_view.all() and _inside(tf.map_points_to_source(grown, s, W, H), FRAME).all())


def test_straightened_frame_crop_has_no_rebate_pixels():
    w, h = 900, 600
    img = np.zeros((h, w, 3), np.float32)  # black rebate around a grey picture
    img[round(FRAME[1] * h):round(FRAME[3] * h), round(FRAME[0] * w):round(FRAME[2] * w)] = 0.5
    for kw in FRAME_CASES:
        s = DevelopSettings(**kw)
        out = tf.warp(tf.lens_correct(img, s), s)
        x0, y0, x1, y1 = tf.map_rect_from_source(FRAME, s, w, h)
        # Three pixels in: the two cubic resamplings soften the frame's edge.
        crop = out[int(y0 * h) + 3:int(np.ceil(y1 * h)) - 3, int(x0 * w) + 3:int(np.ceil(x1 * w)) - 3]
        assert crop.min() > 0.45, kw


def test_map_rect_from_source_edge_cases():
    s = DevelopSettings(angle=3)
    # Unchanged without geometry, clipped to the image.
    assert tf.map_rect_from_source((-0.1, 0.2, 0.8, 1.2), DevelopSettings(lens_vignette=0.5), W, H) == (0.0, 0.2, 0.8, 1.0)
    # The whole image as the frame: the same crop as Constrain crop.
    whole = (0.0, 0.0, 1.0, 1.0)
    assert tf.map_rect_from_source(whole, DevelopSettings(angle=8), W, H) == pytest.approx(
        tf.largest_valid_rect(DevelopSettings(angle=8), W, H, aspect=W / H), abs=1e-6)
    assert tf.map_rect_from_source(whole, DevelopSettings(angle=8), W, H, aspect=None) == pytest.approx(
        tf.largest_valid_rect(DevelopSettings(angle=8), W, H), abs=1e-4)
    assert tf.map_rect_from_source(None, s, W, H) is None
    assert tf.map_rect_from_source((0.5, 0.2, 0.5, 0.6), s, W, H) is None
    assert tf.map_rect_from_source((0.2, 0.2, 0.6, 0.6), DevelopSettings(persp_x=1.0), W, H) is None


# ---------------------------------------------------------------- upright

@pytest.mark.parametrize("tilt", [3.0, -3.0])
def test_level_recovers_a_tilted_horizon(tilt):
    result = tf.auto_upright(_horizon(tilt), "level")
    assert result["angle"] == pytest.approx(-tilt, abs=0.3)
    assert result["persp_vertical"] == 0.0 and result["persp_horizontal"] == 0.0


def test_vertical_makes_converging_verticals_parallel():
    facade, verticals, horizontals = _facade()
    photo, g = _camera(facade, tilt_deg=15, roll_deg=1.5)
    before = _leans_after(verticals, g, DevelopSettings())
    assert np.ptp(before) > 6, "the scene must really converge"
    result = tf.auto_upright(photo, "vertical")
    assert result["persp_vertical"] > 0, "shot from below: positive Vertical"
    leans = _leans_after(verticals, g, DevelopSettings(**result))
    assert np.abs(leans).max() < 1.0 and np.ptp(leans) < 1.0
    # Level alone takes out the roll, read from where the verticals meet.
    assert tf.auto_upright(photo, "level")["angle"] == pytest.approx(-1.5, abs=0.3)


def test_vertical_handles_a_camera_pointing_down():
    facade, verticals, _ = _facade()
    photo, g = _camera(facade, tilt_deg=-12, roll_deg=-1.0)
    result = tf.auto_upright(photo, "vertical")
    assert result["persp_vertical"] < 0
    leans = _leans_after(verticals, g, DevelopSettings(**result))
    assert np.abs(leans).max() < 1.0


def test_full_squares_both_families_when_the_lens_matches():
    facade, verticals, horizontals = _facade()
    # zoom * focal = 1.0: the virtual camera's focal length.
    photo, g = _camera(facade, tilt_deg=14, pan_deg=18, roll_deg=1.0, focal=1.25)
    result = tf.auto_upright(photo, "full")
    s = DevelopSettings(**result)
    assert np.abs(_leans_after(verticals, g, s)).max() < 1.0
    assert np.abs(_leans_after(horizontals, g, s)).max() < 1.0
    vertical_only = DevelopSettings(**tf.auto_upright(photo, "vertical"))
    assert np.ptp(_leans_after(horizontals, g, vertical_only)) > 5


def test_auto_limits_a_strong_keystone():
    facade, _, _ = _facade()
    photo, _ = _camera(facade, tilt_deg=25, roll_deg=-2.0)
    full = tf.auto_upright(photo, "vertical")
    auto = tf.auto_upright(photo, "auto")
    assert full["persp_vertical"] > tf.AUTO_MAX_TILT
    assert auto["persp_vertical"] == pytest.approx(tf.AUTO_MAX_TILT)
    assert auto["angle"] == pytest.approx(2.0, abs=0.5)


def test_upright_needs_lines():
    rng = np.random.default_rng(3)
    noise = rng.uniform(0.1, 0.2, (800, 1200, 3)).astype(np.float32)
    flat = np.full((800, 1200, 3), 0.3, np.float32)
    for mode in ("level", "vertical", "auto", "full"):
        assert tf.auto_upright(noise, mode) == {}
        assert tf.auto_upright(flat, mode) == {}
    assert tf.auto_upright(noise, "off") == {"angle": 0.0, "persp_vertical": 0.0, "persp_horizontal": 0.0}
    # Lines outside the region do not count: the sky above the horizon has none.
    assert tf.auto_upright(_horizon(3.0), "level", region=(0.0, 0.0, 1.0, 0.4)) == {}


def test_angle_from_line_picks_the_nearer_axis():
    assert tf.angle_from_line((0, 0), (100, 100 * np.tan(np.radians(5)))) == pytest.approx(-5)
    assert tf.angle_from_line((100, 0), (0, -100 * np.tan(np.radians(5)))) == pytest.approx(-5)
    assert tf.angle_from_line((0, 0), (-100 * np.tan(np.radians(3)), 100)) == pytest.approx(-3)
    assert tf.angle_from_line((0, 0), (100 * np.tan(np.radians(3)), 100)) == pytest.approx(3)
    assert tf.angle_from_line((5, 5), (5, 5)) == 0.0


# ---------------------------------------------------------------- speed

def test_full_resolution_speed():
    img = np.random.default_rng(4).random((4000, 6000, 3), dtype=np.float32)
    s = DevelopSettings(angle=2.5, persp_vertical=0.3, persp_horizontal=-0.1, lens_distortion=0.3, lens_vignette=0.4)
    t = time.perf_counter()
    corrected = tf.lens_correct(img, s)
    t_lens = time.perf_counter() - t
    t = time.perf_counter()
    out, mask = tf.warp(corrected, s, with_mask=True)
    t_warp = time.perf_counter() - t
    t = time.perf_counter()
    tf.largest_valid_rect(s, 6000, 4000)
    tf.largest_valid_rect(s, 6000, 4000, aspect=1.5)
    t_rect = time.perf_counter() - t
    t = time.perf_counter()
    tf.auto_upright(img, "full")
    t_upright = time.perf_counter() - t
    assert out.shape == img.shape and mask.shape == img.shape[:2]
    assert t_lens < 1.0, t_lens
    assert t_warp < 1.0, t_warp
    assert t_rect < 0.2, t_rect
    assert t_upright < 2.0, t_upright


def test_a_few_short_edges_fix_no_vanishing_point():
    """Regression: the two or three short edges of sprocket holes in a crop met at a
    spurious point and pulled Vertical to its limit; they may only level."""
    img = np.full((800, 1200, 3), 0.04, np.float32)
    for x, lean in ((450, -2.0), (600, 0.0), (750, 2.0)):  # three short edges, converging
        dx = np.tan(np.radians(lean)) * 60
        cv2.line(img, (int(x - dx), 340), (int(x + dx), 460), (0.8, 0.8, 0.8), 6)
    cv2.line(img, (200, 600), (1000, 600), (0.8, 0.8, 0.8), 6)
    for mode in ("vertical", "auto", "full"):
        out = tf.auto_upright(img, mode)
        assert out.get("persp_vertical", 0.0) == 0.0, (mode, out)
        assert out.get("persp_horizontal", 0.0) == 0.0, (mode, out)


# ---------------------------------------------------------------- guided upright

def _guides(edges, g, k1=0.0, w=W, h=H):
    """Scene edges drawn as guides: normalised source points of the photo, through a lens with ``k1``."""
    out = []
    for p0, p1 in edges:
        q = _homography(g, np.array([p0, p1], float))
        if k1:
            q = tf._distort(q, k1, w, h)
        out.append(tuple((q / (w, h)).ravel()))
    return out


def _settled_cost(guides, s, w=W, h=H):
    """Sum of squared leans of the guides under ``s`` with its rotation at its best."""
    leans = []
    for x0, y0, x1, y1 in guides:
        o = tf.map_points_from_source([(x0, y0), (x1, y1)], s, w, h) * (w, h)
        leans.append(_lean_deg(o[1] - o[0]))
    leans = np.array(leans)
    return float(np.sum((leans - leans.mean()) ** 2))


@pytest.mark.parametrize("tilt, roll", [(15.0, 1.5), (-12.0, -1.0)])
def test_guided_two_verticals_make_the_building_plumb(tilt, roll):
    facade, verticals, _ = _facade()
    _, g = _camera(facade, tilt_deg=tilt, roll_deg=roll)
    result = tf.guided_upright(_guides(verticals[:2], g), DevelopSettings(), W, H)
    assert set(result) == {"angle", "persp_vertical", "persp_horizontal"}
    assert np.sign(result["persp_vertical"]) == np.sign(tilt) and result["persp_horizontal"] == 0.0
    s = DevelopSettings(**result)
    assert np.abs(_leans_after(verticals[:2], g, s)).max() < 0.1
    # The building's other verticals meet at the same point, so they come out plumb too.
    assert np.abs(_leans_after(verticals, g, s)).max() < 0.1


def test_guided_two_horizontals_use_only_the_horizontal_keystone():
    facade, _, horizontals = _facade()
    _, g = _camera(facade, tilt_deg=0.0, pan_deg=15.0, roll_deg=-1.0)
    result = tf.guided_upright(_guides([horizontals[0], horizontals[-1]], g), DevelopSettings(), W, H)
    assert result["persp_vertical"] == 0.0 and result["persp_horizontal"] > 0.1
    assert np.abs(_leans_after(horizontals, g, DevelopSettings(**result))).max() < 0.1


def test_guided_four_guides_square_both_families_when_the_lens_matches():
    facade, verticals, horizontals = _facade()
    # zoom * focal = 1.0: the virtual camera's focal length.
    _, g = _camera(facade, tilt_deg=14, pan_deg=18, roll_deg=1.0, focal=1.25)
    edges = [verticals[0], verticals[1], horizontals[0], horizontals[-1]]
    s = DevelopSettings(**tf.guided_upright(_guides(edges, g), DevelopSettings(), W, H))
    assert np.abs(_leans_after(verticals, g, s)).max() < 0.1
    assert np.abs(_leans_after(horizontals, g, s)).max() < 0.1


def test_guided_three_guides_are_exact_and_four_a_least_squares_compromise():
    facade, verticals, horizontals = _facade()
    _, g = _camera(facade, tilt_deg=14, pan_deg=18, roll_deg=1.0)  # a wider lens than the virtual one
    three = _guides([verticals[0], verticals[1], horizontals[0]], g)
    s3 = DevelopSettings(**tf.guided_upright(three, DevelopSettings(), W, H))
    assert np.abs(_leans_after([verticals[0], verticals[1], horizontals[0]], g, s3)).max() < 0.1
    four = three + _guides([horizontals[-1]], g)
    result = tf.guided_upright(four, DevelopSettings(), W, H)
    s4 = DevelopSettings(**result)
    assert np.abs(_leans_after([verticals[0], verticals[1], horizontals[0], horizontals[-1]], g, s4)).max() < 3.0
    best = _settled_cost(four, s4)
    assert best < _settled_cost(four, s3)
    for dv, dh in ((0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)):
        nearby = s4.copy(persp_vertical=result["persp_vertical"] + dv, persp_horizontal=result["persp_horizontal"] + dh)
        assert best <= _settled_cost(four, nearby) + 1e-3


def test_guided_one_guide_only_rotates():
    t3, t4 = np.tan(np.radians(3.0)), np.tan(np.radians(4.0))
    level = (0.2, 0.5, 0.8, 0.5 + 0.6 * W * t3 / H)  # 3° clockwise of level
    plumb = (0.5, 0.1, 0.5 + 0.8 * H * t4 / W, 0.9)  # 4° anticlockwise of plumb
    assert tf.guided_upright([level], DevelopSettings(), W, H) == {
        "angle": -3.0, "persp_vertical": 0.0, "persp_horizontal": 0.0}
    assert tf.guided_upright([plumb], DevelopSettings(), W, H)["angle"] == pytest.approx(4.0)
    # The result replaces the current straighten and keystone...
    busy = DevelopSettings(angle=2.0, persp_vertical=0.4, persp_horizontal=-0.2)
    assert tf.guided_upright([level], busy, W, H) == tf.guided_upright([level], DevelopSettings(), W, H)
    # ...which only decide whether a guide reads as level or plumb.
    a = np.radians(40.0)
    steep = (0.4, 0.3, 0.4 + 400 * np.cos(a) / W, 0.3 + 400 * np.sin(a) / H)
    assert tf.guided_upright([steep], DevelopSettings(), W, H)["angle"] == pytest.approx(-40.0)
    assert tf.guided_upright([steep], DevelopSettings(angle=10.0), W, H)["angle"] == 45.0  # 50°, clamped


def test_guided_one_of_each_squares_them():
    facade, verticals, horizontals = _facade()
    _, g = _camera(facade, tilt_deg=15, roll_deg=1.5)
    edges = [verticals[0], horizontals[0]]
    s = DevelopSettings(**tf.guided_upright(_guides(edges, g), DevelopSettings(), W, H))
    assert np.abs(_leans_after(edges, g, s)).max() < 0.1
    # Already square to each other: rotation alone does it.
    _, rolled = _camera(facade, 0.0, roll_deg=2.0)
    assert tf.guided_upright(_guides(edges, rolled), DevelopSettings(), W, H) == {
        "angle": -2.0, "persp_vertical": 0.0, "persp_horizontal": 0.0}


def test_guided_follows_the_lens_correction_and_keeps_scale_and_offsets():
    facade, verticals, horizontals = _facade()
    _, g = _camera(facade, tilt_deg=12, roll_deg=-1.0)
    s = DevelopSettings(lens_distortion=0.4, persp_scale=1.2, persp_x=0.1, persp_y=-0.05)
    k1 = tf._k1(s)
    edges = [verticals[0], verticals[1], horizontals[0]]
    result = tf.guided_upright(_guides(edges, g, k1), s, W, H)
    assert result != tf.guided_upright(_guides(edges, g, k1), DevelopSettings(), W, H)
    out = s.copy(**result)

    def leans(scene_edges):
        found = []
        for p0, p1 in scene_edges:
            src = tf._distort(_homography(g, np.array([p0, p1], float)), k1, W, H) / (W, H)
            o = tf.map_points_from_source(src, out, W, H) * (W, H)
            found.append(_lean_deg(o[1] - o[0]))
        return np.abs(found)

    assert leans(verticals).max() < 0.1
    assert leans(horizontals).max() < 0.1


def test_guided_degenerate_guides():
    facade, verticals, horizontals = _facade()
    _, g = _camera(facade, tilt_deg=15)
    two = _guides(verticals[:2], g)
    s = DevelopSettings()
    assert tf.guided_upright([], s, W, H) == {}
    assert tf.guided_upright([(0.5, 0.5, 0.505, 0.51)], s, W, H) == {}  # 1.3 % of the long side
    # One just started, or broken, is left out.
    assert tf.guided_upright(two + [(0.3, 0.3, 0.3, 0.3)], s, W, H) == tf.guided_upright(two, s, W, H)
    assert tf.guided_upright(two + [(np.nan, 0.1, 0.2, 0.3)], s, W, H) == tf.guided_upright(two, s, W, H)
    # Two guides along one edge fix no vanishing point, whatever the other guides say.
    x0, y0, x1, y1 = two[0]
    upper = (x0, y0, x0 + 0.4 * (x1 - x0), y0 + 0.4 * (y1 - y0))
    lower = (x0 + 0.6 * (x1 - x0), y0 + 0.6 * (y1 - y0) + 0.002, x1, y1)
    assert tf.guided_upright([two[0], two[0]], s, W, H) == {}
    assert tf.guided_upright([upper, lower], s, W, H) == {}
    assert tf.guided_upright([upper, lower] + _guides([horizontals[0], horizontals[-1]], g), s, W, H) == {}


def test_guided_clamps_to_the_slider_ranges():
    converging = [(0.2, 0.9, 0.45, 0.1), (0.8, 0.9, 0.55, 0.1)]
    assert tf.guided_upright(converging, DevelopSettings(), W, H) == {
        "angle": 0.0, "persp_vertical": 1.0, "persp_horizontal": 0.0}
    result = tf.guided_upright(converging + [(0.1, 0.2, 0.9, 0.35)], DevelopSettings(), W, H)
    assert -1.0 <= result["persp_vertical"] <= 1.0 and -1.0 <= result["persp_horizontal"] <= 1.0
    assert -45.0 <= result["angle"] <= 45.0


def test_guided_speed():
    facade, verticals, horizontals = _facade()
    _, g = _camera(facade, tilt_deg=25, pan_deg=-20, roll_deg=3.0, focal=0.6)
    s = DevelopSettings(lens_distortion=0.3)
    guides = _guides([verticals[0], verticals[1], horizontals[0], horizontals[-1]], g, tf._k1(s))
    times = []
    for _ in range(5):
        t = time.perf_counter()
        result = tf.guided_upright(guides, s, 6000, 4000)
        times.append(time.perf_counter() - t)
    assert result
    # Called on every move of a guide's end: the best of a few runs keeps a busy machine out of it.
    assert min(times) < 0.02, times


@pytest.mark.parametrize("op", ["flip_h", "flip_v", "cw", "ccw"])
@pytest.mark.parametrize("params", [
    dict(angle=4.0, persp_vertical=0.5, persp_x=0.1),
    dict(angle=-7.0, persp_horizontal=-0.4, persp_y=0.08, persp_rotate=3.0),
    dict(angle=2.0, persp_vertical=0.3, persp_aspect=0.4, persp_scale=1.2),
])
def test_reorient_turns_the_developed_picture_with_the_image(op, params):
    """Regression: flipping or rotating 90° after straightening doubled the tilt or put
    the keystone on the wrong axis, because orientation is applied before the warp."""
    w, h = 1500, 1000
    s = DevelopSettings(**params)
    turned = s.copy(**tf.reorient(s, op))
    tw, th = (h, w) if op in ("cw", "ccw") else (w, h)
    rng = np.random.default_rng(0)
    src = rng.uniform(0.15, 0.85, size=(40, 2))
    # The picture before, turned on screen ...
    expected = np.array([tf._turn_point(x, y, op) for x, y in tf.map_points_from_source(src, s, w, h)])
    # ... must be what the turned settings make of the turned source.
    turned_src = np.array([tf._turn_point(x, y, op) for x, y in src])
    got = tf.map_points_from_source(turned_src, turned, tw, th)
    assert np.abs(got - expected).max() < 1e-9


def test_reorient_moves_crop_and_guides():
    s = DevelopSettings(crop=(0.1, 0.2, 0.5, 0.6), upright_guides=((0.2, 0.1, 0.25, 0.9),))
    cw = tf.reorient(s, "cw")
    assert cw["crop"] == pytest.approx((0.4, 0.1, 0.8, 0.5))
    assert cw["upright_guides"][0] == pytest.approx((0.9, 0.2, 0.1, 0.25))
    back = s.copy(**cw)
    assert back.copy(**tf.reorient(back, "ccw")).crop == pytest.approx(s.crop)


def test_switching_transform_off_keeps_the_straighten_angle():
    s = DevelopSettings(angle=5.0, persp_vertical=0.6, disabled=("transform",))
    only_angle = DevelopSettings(angle=5.0)
    pts = [(0.2, 0.3), (0.8, 0.7)]
    assert not tf.is_identity(s)
    assert np.allclose(tf.map_points_from_source(pts, s, 1200, 800), tf.map_points_from_source(pts, only_angle, 1200, 800))
