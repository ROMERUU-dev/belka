import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from belka import i18n
from belka.core.adjust import clipping_masks
from belka.ui.histogram import InteractiveHistogram, clipped_channels, zone_label

LEFT, WIDTH = 8.0, 300.0  # graph area of a 316 px wide widget


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def flat_histogram() -> np.ndarray:
    h = np.zeros((3, 256))
    h[:, 40:220] = 300.0
    return h


@pytest.fixture
def hist(app):
    w = InteractiveHistogram()
    w.resize(int(WIDTH + 2 * LEFT), 120)
    w.set_histogram(flat_histogram())
    w.adjusts, w.finished, w.toggles = [], [], []
    w.adjustRequested.connect(lambda f, d: w.adjusts.append((f, d)))
    w.dragFinished.connect(w.finished.append)
    w.clippingToggled.connect(lambda which, on: w.toggles.append((which, on)))
    return w


def x_at(fraction: float) -> float:
    return LEFT + WIDTH * fraction


def send(w, kind, x, y=60.0, button=Qt.MouseButton.NoButton, buttons=Qt.MouseButton.NoButton):
    event = QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(w, event)


def press(w, x, y=60.0):
    send(w, QEvent.Type.MouseButtonPress, x, y, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)


def move(w, x, y=60.0, held=True):
    send(w, QEvent.Type.MouseMove, x, y, buttons=Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton)


def release(w, x, y=60.0):
    send(w, QEvent.Type.MouseButtonRelease, x, y, Qt.MouseButton.LeftButton)


def drag(w, start: float, end: float, steps: int = 6) -> None:
    press(w, x_at(start))
    for f in np.linspace(start, end, steps)[1:]:
        move(w, x_at(f))
    release(w, x_at(end))


def total(w, field: str) -> float:
    return sum(d for f, d in w.adjusts if f == field)


def test_zones_map_position_to_fields(hist):
    cases = [(0.0, "black"), (0.05, "black"), (0.099, "black"), (0.1, "shadows"), (0.29, "shadows"),
             (0.31, "exposure"), (0.69, "exposure"), (0.71, "highlights"), (0.89, "highlights"),
             (0.91, "white"), (1.0, "white")]
    for fraction, field in cases:
        assert hist.zone_at(x_at(fraction)) == field, fraction
    # The margins beside the graph belong to the outer zones.
    assert hist.zone_at(0.0) == "black"
    assert hist.zone_at(hist.width()) == "white"


def test_drag_right_raises_exposure_four_ev_per_width(hist):
    hist.set_values({"exposure": 0.5, "contrast": 1.3})  # unknown keys are ignored
    drag(hist, 0.5, 0.75)
    assert {f for f, _d in hist.adjusts} == {"exposure"}
    assert len(hist.adjusts) > 1  # continuous while dragging, not only on release
    assert all(d > 0 for _f, d in hist.adjusts)
    assert total(hist, "exposure") == pytest.approx(1.0, abs=0.011)
    assert hist.finished == ["exposure"]


def test_drag_left_lowers_tone_fields_with_their_scale(hist):
    drag(hist, 0.2, 0.1)  # shadows: 2.0 per width
    assert all(d < 0 for _f, d in hist.adjusts)
    assert total(hist, "shadows") == pytest.approx(-0.2, abs=0.011)
    drag(hist, 0.8, 0.9)  # starts in highlights, so it stays highlights past the zone edge
    assert total(hist, "highlights") == pytest.approx(0.2, abs=0.011)
    drag(hist, 0.95, 0.85)  # whites: the whole -0.3..0.3 range per width
    assert total(hist, "white") == pytest.approx(-0.06, abs=0.004)
    assert hist.finished == ["shadows", "highlights", "white"]


def test_dragging_the_blacks_right_lifts_them(hist):
    # A positive model ``black`` darkens, so lifting the blacks (the left foot
    # of the histogram following the mouse right) is a negative model delta.
    drag(hist, 0.05, 0.15)
    assert hist.adjusts and all(f == "black" and d < 0 for f, d in hist.adjusts)
    assert total(hist, "black") == pytest.approx(-0.06, abs=0.004)
    hist.adjusts.clear()
    hist.set_values({"black": 0.0})
    drag(hist, 0.05, 0.0)
    assert total(hist, "black") == pytest.approx(0.03, abs=0.004)
    # Clamped in model units: Negros +100 is the model's -0.3.
    hist.adjusts.clear()
    hist.set_values({"black": -0.28})
    drag(hist, 0.05, 0.5)
    assert total(hist, "black") == pytest.approx(-0.02)


def test_drag_is_clamped_to_the_field_range(hist):
    hist.set_values({"white": 0.25, "exposure": -2.9})
    drag(hist, 0.95, 1.6)
    assert total(hist, "white") == pytest.approx(0.05)
    assert hist.cursor().shape() == Qt.CursorShape.ArrowCursor  # released outside: no hover left behind
    drag(hist, 0.5, -0.3)
    assert total(hist, "exposure") == pytest.approx(-0.1)
    # Coming back after hitting the limit undoes only what was applied.
    hist.adjusts.clear()
    press(hist, x_at(0.95))
    move(hist, x_at(1.6))
    move(hist, x_at(0.95))
    release(hist, x_at(0.95))
    assert total(hist, "white") == pytest.approx(0.0, abs=1e-9)


def test_a_value_past_the_range_does_not_jump_to_the_limit(hist):
    # The exposure slider reaches ±5 EV; a small drag must stay a small change.
    hist.set_values({"exposure": 4.0})
    drag(hist, 0.5, 0.48)
    assert total(hist, "exposure") == pytest.approx(-0.08, abs=0.011)
    hist.adjusts.clear()
    hist.set_values({"exposure": 4.0})
    drag(hist, 0.5, 0.9)  # cannot grow further, but nothing is taken away either
    assert hist.adjusts == [] and hist._values["exposure"] == 4.0


def test_click_without_moving_changes_nothing(hist):
    press(hist, x_at(0.5))
    move(hist, x_at(0.5) + 0.2)  # below one step
    release(hist, x_at(0.5) + 0.2)
    assert hist.adjusts == [] and hist.finished == []


def test_double_click_resets_the_zone(hist):
    hist.set_values({"shadows": 0.4})
    send(hist, QEvent.Type.MouseButtonDblClick, x_at(0.2), 60.0, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    assert hist.adjusts == [("shadows", pytest.approx(-0.4))]
    assert hist.finished == ["shadows"]


def test_clipping_detection_uses_the_end_bins():
    h = np.zeros((3, 256))
    h[:, 100:150] = 1000.0  # 50 000 pixels per channel
    assert clipped_channels(h) == ((False,) * 3, (False,) * 3)
    h[2, 0] = 50  # 0.1 % of blue is black
    h[:, 255] = 2  # 0.004 %: noise, not clipping
    assert clipped_channels(h) == ((False, False, True), (False, False, False))
    h[0, 255] = 500
    assert clipped_channels(h)[1] == (True, False, False)


def test_widget_reports_clipping_and_lights_the_triangle(hist):
    h = np.zeros((3, 256))
    h[:, 60:200] = 500.0
    hist.set_histogram(h)
    assert hist.clipping() == (False, False)
    off = hist.grab().toImage().pixelColor(301, 10)
    h[:, 255] = 4000.0
    hist.set_histogram(h)
    assert hist.clipping() == (False, True)
    lit = hist.grab().toImage().pixelColor(301, 10)
    assert lit.lightness() > 200 > 100 > off.lightness()
    hist.set_histogram(None)
    assert hist.clipping() == (False, False)


def test_triangles_toggle_the_clipping_overlay(hist):
    shadows, highlights = (14.0, 10.0), (hist.width() - 14.0, 10.0)
    for x, y in (highlights, highlights, shadows):
        press(hist, x, y)
        release(hist, x, y)
    assert hist.toggles == [("highlights", True), ("highlights", False), ("shadows", True)]
    hist.set_clipping(False, True)  # e.g. the J shortcut: mirrored, not echoed
    assert len(hist.toggles) == 3
    press(hist, *highlights)
    release(hist, *highlights)
    assert hist.toggles[-1] == ("highlights", False)
    assert hist.adjusts == [] and hist.finished == []


def test_cursor_shows_what_a_drag_would_do(hist):
    move(hist, x_at(0.5), held=False)
    assert hist.cursor().shape() == Qt.CursorShape.SizeHorCursor
    move(hist, 14.0, 10.0, held=False)
    assert hist.cursor().shape() == Qt.CursorShape.PointingHandCursor


def test_labels_read_like_the_sliders():
    assert zone_label("exposure", 0.35) == "Exposición +0,35"
    assert zone_label("exposure", -0.001) == "Exposición 0,00"
    assert zone_label("shadows", -0.25) == "Sombras −25"
    assert zone_label("white", 0.15) == "Blancos +50"
    assert zone_label("highlights", 0.0) == "Altas luces 0"
    try:
        i18n.set_language("en")
        assert zone_label("exposure", -1.2) == "Exposure −1.20"
        assert zone_label("black", -0.3) == "Blacks +100"
    finally:
        i18n.set_language("es")
    # Negros reads like its slider, which runs opposite to the model value.
    assert zone_label("black", -0.15) == "Negros +50"
    assert zone_label("black", 0.3) == "Negros −100"


def test_a_lost_release_ends_the_drag_instead_of_editing_on_hover(hist):
    hist.show()
    press(hist, x_at(0.5))
    move(hist, x_at(0.6))
    assert total(hist, "exposure") > 0
    hist.hide()  # e.g. a module switch: the release never arrives
    assert hist.finished == ["exposure"]
    hist.show()
    hist.adjusts.clear()
    move(hist, x_at(0.8), held=False)
    assert hist.adjusts == [] and hist.finished == ["exposure"]
    # Disabled mid-drag (a modal dialog): the same.
    press(hist, x_at(0.5))
    move(hist, x_at(0.4))
    hist.setEnabled(False)
    hist.setEnabled(True)
    hist.adjusts.clear()
    move(hist, x_at(0.2), held=False)
    assert hist.adjusts == [] and hist.finished == ["exposure", "exposure"]
    # The release went to another widget and the pointer comes back unpressed.
    press(hist, x_at(0.5))
    move(hist, x_at(0.55))
    hist.adjusts.clear()
    move(hist, x_at(0.9), held=False)
    assert hist.adjusts == [] and hist.finished == ["exposure"] * 3
    assert hist.cursor().shape() == Qt.CursorShape.SizeHorCursor  # back to plain hover
    hist.hide()


def test_renders_arriving_during_a_drag_leave_the_dragged_field_alone(hist):
    press(hist, x_at(0.5))
    move(hist, x_at(0.6))
    dragged = total(hist, "exposure")
    hist.set_values({"exposure": 0.1, "shadows": 0.3})  # an older render's settings
    assert hist._values["exposure"] == pytest.approx(dragged)
    assert hist._values["shadows"] == 0.3
    move(hist, x_at(0.65))
    release(hist, x_at(0.65))
    hist.set_values({"exposure": 0.7})
    assert hist._values["exposure"] == 0.7


def test_without_a_photo_the_zones_are_inert(hist):
    hist.set_histogram(None)
    move(hist, x_at(0.5), held=False)
    assert hist.cursor().shape() == Qt.CursorShape.ArrowCursor
    drag(hist, 0.5, 0.8)
    send(hist, QEvent.Type.MouseButtonDblClick, x_at(0.5), 60.0, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    assert hist.adjusts == [] and hist.finished == []
    assert hist._values["exposure"] == 0.0
    # The triangles still mirror the clipping view, which J toggles anyway.
    press(hist, 14.0, 10.0)
    release(hist, 14.0, 10.0)
    assert hist.toggles == [("shadows", True)]


def test_inert_zones_neither_drag_nor_reset(hist):
    """The flat output has no Sombras or Altas luces: their zones say so and do nothing."""
    hist.set_values({"shadows": 0.4})
    hist.set_inert_zones(("shadows", "highlights"))
    move(hist, x_at(0.2), held=False)
    assert hist.cursor().shape() == Qt.CursorShape.ArrowCursor
    hist.grab()  # paints the "not applied" footer
    drag(hist, 0.2, 0.25)
    drag(hist, 0.8, 0.85)
    send(hist, QEvent.Type.MouseButtonDblClick, x_at(0.2), 60.0, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    assert hist.adjusts == [] and hist.finished == []
    drag(hist, 0.5, 0.6)
    assert total(hist, "exposure") > 0
    hist.set_inert_zones(())
    move(hist, x_at(0.21), held=False)
    assert hist.cursor().shape() == Qt.CursorShape.SizeHorCursor


def test_losing_the_photo_mid_drag_closes_the_drag(hist):
    press(hist, x_at(0.5))
    move(hist, x_at(0.6))
    hist.set_histogram(None)
    hist.adjusts.clear()
    move(hist, x_at(0.9))
    assert hist.adjusts == [] and hist.finished == ["exposure"]


def test_triangle_rule_matches_the_clipping_overlay():
    # Deep film shadows often clip blue alone: the triangle and the overlay
    # (adjust.clipping_masks) must agree that the pixel clips.
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    img[:10, :10, 2] = 0
    img[50:60, :10, 0] = 255
    counts = np.stack([np.bincount(img[..., c].ravel(), minlength=256) for c in range(3)])
    low, high = clipped_channels(counts)
    shadows, highlights = clipping_masks(img.astype(np.float32) / 255.0)
    assert low == (False, False, True) and shadows.sum() == 100
    assert high == (True, False, False) and highlights.sum() == 100
    img[..., 2] = np.where(img[..., 2] == 0, 1, img[..., 2])  # 1/255 is not clipped for either
    img[..., 0] = np.where(img[..., 0] == 255, 254, img[..., 0])
    counts = np.stack([np.bincount(img[..., c].ravel(), minlength=256) for c in range(3)])
    shadows, highlights = clipping_masks(img.astype(np.float32) / 255.0)
    assert clipped_channels(counts) == ((False,) * 3, (False,) * 3)
    assert not shadows.any() and not highlights.any()


def test_paints_every_state(hist):
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal([90, 120, 150], 40, (200, 300, 3)), 0, 255).astype(np.uint8)
    hist.set_histogram(np.stack([np.bincount(img[..., c].ravel(), minlength=256) for c in range(3)]))
    hist.set_info("ISO 100  ·  f/8  ·  1/2 s")
    hist.set_readout("R 52 %  G 48 %  B 41 %")
    assert not hist.grab().isNull()
    move(hist, x_at(0.5), held=False)
    press(hist, x_at(0.5))
    move(hist, x_at(0.6))
    assert not hist.grab().isNull()
    release(hist, x_at(0.6))
    hist.setEnabled(False)
    assert not hist.grab().isNull()
