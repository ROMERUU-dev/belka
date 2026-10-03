"""ImageView: crop frame, straighten line, Guided Upright, grid, context menu, before/after, zoom and hover."""

import math

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QAction, QColor, QContextMenuEvent, QImage, QKeyEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QVBoxLayout, QWidget

W, H = 1000, 500  # image pixels: 2:1 in an 800x600 view, so fit leaves bands above and below
VIEW_W, VIEW_H = 800, 600

# Other test files create a bare QGuiApplication when they run first, and widgets
# need a QApplication: make it at collection time, before any test runs.
APP = QApplication.instance() or QApplication([])


def solid(color: str, w: int = W, h: int = H) -> QImage:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(color))
    return img


@pytest.fixture
def view():
    from belka.ui.image_view import ImageView

    v = ImageView()
    v.resize(VIEW_W, VIEW_H)
    v.show()
    v.set_image(solid("#406080"))
    yield v
    v.close()
    v.deleteLater()


def vp(v, x: float, y: float) -> QPoint:
    """Viewport position of a normalised image point."""
    r = v.sceneRect()
    return v.viewportTransform().map(QPointF(x * r.width(), y * r.height())).toPoint()


def drag(v, a: QPoint, b: QPoint, modifiers=Qt.KeyboardModifier.NoModifier, steps: int = 4) -> None:
    port = v.viewport()
    QTest.mousePress(port, Qt.MouseButton.LeftButton, modifiers, a)
    for i in range(1, steps + 1):
        QTest.mouseMove(port, a + (b - a) * (i / steps))
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, modifiers, b)


def quick_double_click(v, pos: QPoint, then_drag_to: QPoint | None = None) -> None:
    """What the platform delivers for two quick clicks: the second press arrives only as a DblClick."""
    port = v.viewport()
    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton

    def send(kind, at: QPoint, buttons) -> None:
        p = QPointF(at)
        QApplication.sendEvent(port, QMouseEvent(kind, p, port.mapToGlobal(p), left if kind != QEvent.Type.MouseMove
                                                 else none, buttons, Qt.KeyboardModifier.NoModifier))

    send(QEvent.Type.MouseButtonPress, pos, left)
    send(QEvent.Type.MouseButtonRelease, pos, none)
    send(QEvent.Type.MouseButtonDblClick, pos, left)
    end = pos
    if then_drag_to is not None:
        for i in range(1, 5):
            end = pos + (then_drag_to - pos) * (i / 4)
            send(QEvent.Type.MouseMove, end, left)
    send(QEvent.Type.MouseButtonRelease, end, none)


def approx_rect(rect, expected, tol=0.006):
    assert rect == pytest.approx(expected, abs=tol)


def pixel_aspect(rect) -> float:
    x0, y0, x1, y1 = rect
    return (x1 - x0) * W / ((y1 - y0) * H)


# ---------------------------------------------------------------- crop
def test_crop_tool_leaves_room_around_the_image(view):
    fit = view.zoom()
    assert fit == pytest.approx(0.8)
    view.set_tool("crop")
    assert view.zoom() < fit  # margin to grab the handles and rotate outside the frame
    assert view.is_fit()


def test_crop_corner_drag_free(view):
    view.set_tool("crop")
    view.set_crop_rect((0.1, 0.1, 0.9, 0.9))
    changes = []
    view.cropChanged.connect(changes.append)
    drag(view, vp(view, 0.9, 0.9), vp(view, 0.7, 0.6))
    approx_rect(view.crop_rect(), (0.1, 0.1, 0.7, 0.6))
    assert changes and changes[-1] == view.crop_rect()


def test_crop_hit_parts():
    from belka.ui.crop_overlay import CropOverlay

    overlay = CropOverlay()
    overlay.set_image_size(W, H)
    overlay.set_rect((0.2, 0.2, 0.8, 0.8))  # 200..800 x 100..400 px
    zoom = 1.0  # tolerance 8 screen px = 8 image px
    assert overlay.hit(QPointF(200, 100), zoom) == "tl"
    assert overlay.hit(QPointF(810, 411), zoom) == "br"  # corners reach a little further
    assert overlay.hit(QPointF(805, 250), zoom) == "r"
    assert overlay.hit(QPointF(500, 96), zoom) == "t"
    assert overlay.hit(QPointF(500, 250), zoom) == "move"
    assert overlay.hit(QPointF(812, 250), zoom) == "rotate"
    assert overlay.hit(QPointF(100, 30), zoom) == "rotate"


def test_the_whole_corner_bracket_grabs_the_corner():
    from belka.ui.crop_overlay import CropOverlay

    overlay = CropOverlay()
    overlay.set_image_size(W, H)
    overlay.set_rect((0.2, 0.2, 0.8, 0.8))  # 600 x 300 px: brackets with 22 px arms
    for along in (2, 12, 18, 21):  # on the arms, 2 px inside the frame
        assert overlay.hit(QPointF(200 + along, 102), 1.0) == "tl"
        assert overlay.hit(QPointF(798, 400 - along), 1.0) == "br"
    assert overlay.hit(QPointF(230, 102), 1.0) == "t"  # past the bracket: the edge
    # Measured on screen: at 2x the arm covers 11 image px.
    assert overlay.hit(QPointF(209, 101), 2.0) == "tl"
    assert overlay.hit(QPointF(214, 101), 2.0) == "t"


def test_crop_edge_drag_moves_one_side(view):
    view.set_tool("crop")
    view.set_crop_rect((0.1, 0.1, 0.9, 0.9))
    drag(view, vp(view, 0.1, 0.5), vp(view, 0.3, 0.2))
    approx_rect(view.crop_rect(), (0.3, 0.1, 0.9, 0.9))


def test_crop_move_inside_is_clamped_to_the_image(view):
    view.set_tool("crop")
    view.set_crop_rect((0.2, 0.2, 0.6, 0.6))
    drag(view, vp(view, 0.4, 0.4), vp(view, 0.95, 0.5))
    approx_rect(view.crop_rect(), (0.6, 0.3, 1.0, 0.7))


def test_crop_resize_stops_at_the_bounds(view):
    view.set_tool("crop")
    view.set_crop_rect((0.3, 0.3, 0.6, 0.6))
    view.set_crop_bounds((0.1, 0.1, 0.8, 0.9))
    drag(view, vp(view, 0.6, 0.6), vp(view, 0.99, 0.99))
    approx_rect(view.crop_rect(), (0.3, 0.3, 0.8, 0.9))


def test_bounds_shrink_the_frame_keeping_its_shape(view):
    view.set_crop_rect(None)
    view.set_crop_bounds((0.2, 0.25, 0.8, 0.75))
    rect = view.crop_rect()
    assert rect[0] >= 0.2 - 1e-9 and rect[2] <= 0.8 + 1e-9 and rect[1] >= 0.25 - 1e-9 and rect[3] <= 0.75 + 1e-9
    assert pixel_aspect(rect) == pytest.approx(W / H, rel=1e-6)


def test_aspect_lock_holds_while_dragging(view):
    view.set_tool("crop")
    view.set_crop_rect((0.1, 0.1, 0.9, 0.9))
    view.set_crop_aspect(1.0)
    assert pixel_aspect(view.crop_rect()) == pytest.approx(1.0, rel=1e-6)
    x0, y0, x1, y1 = view.crop_rect()
    for target in ((0.5, 0.95), (0.98, 0.3), (0.55, 0.55)):
        x0, y0, x1, y1 = view.crop_rect()
        drag(view, vp(view, x1, y1), vp(view, *target))
        assert pixel_aspect(view.crop_rect()) == pytest.approx(1.0, rel=1e-6)
    x0, y0, x1, y1 = view.crop_rect()
    drag(view, vp(view, x1, (y0 + y1) / 2), vp(view, 0.99, 0.5))  # edge handle
    rect = view.crop_rect()
    assert pixel_aspect(rect) == pytest.approx(1.0, rel=1e-6)
    assert min(rect) >= -1e-9 and max(rect) <= 1 + 1e-9


def test_aspect_follows_the_frame_orientation(view):
    view.set_crop_rect((0.4, 0.1, 0.6, 0.9))  # 200 x 400 px: portrait
    view.set_crop_aspect(3 / 2)
    assert view.crop_aspect() == pytest.approx(2 / 3)
    assert pixel_aspect(view.crop_rect()) == pytest.approx(2 / 3, rel=1e-6)
    view.set_crop_aspect(None)
    assert view.crop_aspect() is None


def test_x_swaps_orientation(view):
    view.set_tool("crop")
    view.set_crop_rect((0.3, 0.2, 0.5, 0.8))  # 200 x 300 px
    view.set_crop_aspect(2 / 3)
    QTest.keyClick(view, Qt.Key.Key_X)
    assert view.crop_aspect() == pytest.approx(3 / 2)
    assert pixel_aspect(view.crop_rect()) == pytest.approx(3 / 2, rel=1e-6)


def test_enter_commits_and_escape_cancels(view):
    view.set_tool("crop")
    view.set_crop_rect((0.2, 0.2, 0.7, 0.8))
    committed, cancelled = [], []
    view.cropCommitted.connect(committed.append)
    view.cropCancelled.connect(lambda: cancelled.append(True))
    QTest.keyClick(view, Qt.Key.Key_Return)
    QTest.keyClick(view, Qt.Key.Key_Enter, Qt.KeyboardModifier.KeypadModifier)
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert len(committed) == 2 and cancelled == [True]
    approx_rect(committed[0], (0.2, 0.2, 0.7, 0.8), 1e-9)
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(view, 0.45, 0.5))
    assert len(committed) == 3


def test_crop_keys_override_window_shortcuts(view):
    view.set_tool("crop")
    event = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(view, event)
    assert event.isAccepted()
    view.set_tool("none")
    event = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    event.ignore()
    QApplication.sendEvent(view, event)
    assert not event.isAccepted()


def test_dragging_outside_the_frame_rotates(view):
    view.set_tool("crop")
    view.set_crop_rect((0.25, 0.25, 0.75, 0.75))
    deltas, finished = [], []
    view.angleDelta.connect(deltas.append)
    view.rotateFinished.connect(lambda: finished.append(True))
    centre = view.viewportTransform().map(QPointF(0.5 * W, 0.5 * H))
    start = QPoint(round(centre.x() + 300), round(centre.y()))
    end = QPoint(round(centre.x() + 300), round(centre.y() + 120))  # clockwise on screen
    before = view.crop_rect()
    drag(view, start, end, steps=6)
    expected = math.degrees(math.atan2(end.y() - centre.y(), end.x() - centre.x())
                            - math.atan2(start.y() - centre.y(), start.x() - centre.x()))
    assert len(deltas) == 6
    assert sum(deltas) == pytest.approx(expected, abs=0.3)
    assert sum(deltas) > 0
    assert finished == [True]
    assert view.crop_rect() == before


def test_rotating_out_and_back_gives_the_frame_back(view):
    """The integrator's loop: each angle step shrinks the valid area, and the bounds follow it."""
    view.set_tool("crop")
    view.set_crop_rect(None)
    angle = {"now": 0.0}

    def started() -> None:
        angle["start"], angle["sum"] = angle["now"], 0.0

    def stepped(delta: float) -> None:
        angle["sum"] += delta
        angle["now"] = max(-45.0, min(45.0, angle["start"] + angle["sum"]))
        m = abs(angle["now"]) / 100  # a stand-in for transform.largest_valid_rect
        view.set_crop_bounds((m, m, 1 - m, 1 - m))

    view.rotateStarted.connect(started)
    view.angleDelta.connect(stepped)
    centre = view.viewportTransform().map(QPointF(0.5 * W, 0.5 * H))

    def at(deg: float) -> QPoint:  # above the frame, on a circle around its centre
        a = math.radians(deg - 90)
        return QPoint(round(centre.x() + 250 * math.cos(a)), round(centre.y() + 250 * math.sin(a)))

    port = view.viewport()
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at(0))
    for deg in range(1, 11):
        QTest.mouseMove(port, at(deg))
    assert angle["now"] == pytest.approx(10, abs=0.3)
    approx_rect(view.crop_rect(), (0.1, 0.1, 0.9, 0.9))
    for deg in range(9, -1, -1):
        QTest.mouseMove(port, at(deg))
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at(0))
    assert angle["now"] == pytest.approx(0, abs=1e-6)
    approx_rect(view.crop_rect(), (0, 0, 1, 1), 1e-9)

    # A render that arrives after the release still gives the frame back.
    view.set_crop_bounds((0.1, 0.1, 0.9, 0.9))
    view.set_crop_bounds(None)
    approx_rect(view.crop_rect(), (0, 0, 1, 1), 1e-9)
    # What the user drags is the new frame to keep.
    view.set_crop_bounds((0.1, 0.1, 0.9, 0.9))
    drag(view, vp(view, 0.9, 0.9), vp(view, 0.6, 0.7))
    view.set_crop_bounds(None)
    approx_rect(view.crop_rect(), (0.1, 0.1, 0.6, 0.7))


def test_commit_crop_hands_over_the_frame(view):
    committed = []
    view.cropCommitted.connect(committed.append)
    view.commit_crop()  # not cropping: nothing to hand over
    assert committed == []
    view.set_tool("crop")
    view.set_crop_rect((0.2, 0.3, 0.6, 0.9))
    port = view.viewport()
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(view, 0.6, 0.9))
    QTest.mouseMove(port, vp(view, 0.5, 0.8))
    view.commit_crop()  # the window's Done button, or R again, in the middle of a drag
    QTest.mouseMove(port, vp(view, 0.4, 0.7))
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(view, 0.4, 0.7))
    assert len(committed) == 1
    approx_rect(committed[0], (0.2, 0.3, 0.5, 0.8))
    assert view.crop_rect() == committed[0]  # the drag ended with the commit


def test_click_then_quick_drag_on_a_handle_resizes(view):
    view.set_tool("crop")
    view.set_crop_rect((0.1, 0.1, 0.9, 0.9))
    committed = []
    view.cropCommitted.connect(committed.append)
    corner = vp(view, 0.9, 0.9)
    quick_double_click(view, corner, then_drag_to=vp(view, 0.7, 0.6))
    approx_rect(view.crop_rect(), (0.1, 0.1, 0.7, 0.6))
    assert committed == []  # only a double click inside the frame commits
    quick_double_click(view, vp(view, 0.4, 0.4))
    assert len(committed) == 1


# ---------------------------------------------------------------- straighten
def test_straighten_line_in_normalised_coordinates(view):
    view.set_tool("straighten")
    lines = []
    view.straightenLine.connect(lambda a, b: lines.append((a, b)))
    drag(view, vp(view, 0.1, 0.5), vp(view, 0.9, 0.42))
    assert len(lines) == 1
    a, b = lines[0]
    assert (a.x(), a.y()) == pytest.approx((0.1, 0.5), abs=0.004)
    assert (b.x(), b.y()) == pytest.approx((0.9, 0.42), abs=0.004)


def test_straighten_click_emits_nothing(view):
    view.set_tool("straighten")
    lines = []
    view.straightenLine.connect(lambda a, b: lines.append((a, b)))
    drag(view, vp(view, 0.5, 0.5), vp(view, 0.5, 0.5) + QPoint(2, 1))
    assert lines == []


def test_ctrl_drag_in_crop_straightens(view):
    view.set_tool("crop")
    view.set_crop_rect((0.1, 0.1, 0.9, 0.9))
    lines = []
    view.straightenLine.connect(lambda a, b: lines.append((a, b)))
    drag(view, vp(view, 0.2, 0.3), vp(view, 0.7, 0.35), Qt.KeyboardModifier.ControlModifier)
    assert len(lines) == 1
    approx_rect(view.crop_rect(), (0.1, 0.1, 0.9, 0.9), 1e-9)


# ---------------------------------------------------------------- compare
def color_at(v, x: int, y: int) -> QColor:
    return v.grab().toImage().pixelColor(x, y)


def test_split_compare_shows_before_left_and_follows_the_divider(view):
    view.set_image(solid("#2040ff"))
    view.set_compare("split", solid("#ff2020"))
    assert view.compare_mode == "split"
    y = VIEW_H // 2 + 60
    assert color_at(view, 200, y).red() > 200  # before on the left
    assert color_at(view, 600, y).blue() > 200  # after on the right
    drag(view, QPoint(VIEW_W // 2, y), QPoint(150, y))
    assert color_at(view, 250, y).blue() > 200
    assert color_at(view, 100, y).red() > 200


def test_side_by_side_uses_two_panes(view):
    view.set_compare("side", solid("#ff2020"))
    assert view.compare_mode == "side"
    assert view.viewportMargins().left() == VIEW_W // 2
    assert view.viewport().width() == VIEW_W - VIEW_W // 2
    assert view.is_fit() and view.zoom() < 0.8  # each half holds the whole image
    shot = view.grab().toImage()
    assert shot.pixelColor(VIEW_W // 4, VIEW_H // 2).red() > 200
    assert shot.pixelColor(3 * VIEW_W // 4, VIEW_H // 2).blue() > 100
    view.set_compare("off")
    assert view.viewportMargins().left() == 0
    assert view.zoom() == pytest.approx(0.8)


def test_unknown_compare_mode_is_off(view):
    view.set_compare("wipe", None)
    assert view.compare_mode == "off"


def test_frame_tools_hide_the_split(view):
    view.set_image(solid("#2040ff"))
    view.set_compare("split", solid("#ff2020"))
    view.set_tool("crop")
    view.set_crop_rect((0.3, 0.2, 0.7, 0.8))
    assert view.compare_mode == "split"  # still requested, for when the tool closes
    inside = vp(view, 0.4, 0.5)
    assert color_at(view, inside.x(), inside.y()).blue() > 200  # the after image fills the frame
    drag(view, vp(view, 0.5, 0.2), vp(view, 0.5, 0.1))  # the top handle, right where the divider was
    approx_rect(view.crop_rect(), (0.3, 0.1, 0.7, 0.8))
    view.set_tool("none")
    assert color_at(view, 200, VIEW_H // 2 + 60).red() > 200


def test_frame_tools_hide_the_side_pane(view):
    view.set_compare("side", solid("#ff2020"))
    view.set_tool("straighten")
    assert view.viewportMargins().left() == 0
    assert color_at(view, VIEW_W // 4, VIEW_H // 2).red() < 100
    view.set_tool("none")
    assert view.viewportMargins().left() == VIEW_W // 2


# ---------------------------------------------------------------- zoom, hover, overlays
def test_toggle_zoom_keeps_the_point_under_the_pointer(view):
    zooms = []
    view.zoomChanged.connect(zooms.append)
    pos = QPointF(600, 250)
    point = view.mapToScene(pos.toPoint())
    view.toggle_zoom(factor=2.0, pos=pos)  # at 2:1 the image overflows both ways, so both axes can follow
    assert view.zoom() == pytest.approx(2.0) and not view.is_fit()
    assert (view.mapFromScene(point) - pos.toPoint()).manhattanLength() <= 2
    view.toggle_zoom()
    assert view.is_fit() and view.zoom() == pytest.approx(0.8)
    view.toggle_zoom(pos=pos)
    assert view.zoom() == pytest.approx(1.0)
    assert zooms == [pytest.approx(2.0), pytest.approx(0.8), pytest.approx(1.0)]


def test_toggle_zoom_connects_to_an_action(view):
    action = QAction(view)  # as the window wires Z: triggered(checked) must not become the factor
    action.triggered.connect(view.toggle_zoom)
    action.trigger()
    assert view.zoom() == pytest.approx(1.0) and not view.is_fit()
    action.trigger()
    assert view.is_fit()


def test_two_quick_clicks_zoom_in_and_back_out(view):
    quick_double_click(view, QPoint(400, 300))
    assert view.is_fit() and view.zoom() == pytest.approx(0.8)


def test_frame_tools_fit_the_view(view):
    view.zoom_to(2.0)
    view.center_on(0.2, 0.3)
    view.set_tool("crop")  # the uncropped render follows: a kept zoom would land elsewhere
    assert view.is_fit() and view.zoom() < 0.8
    view.zoom_to(1.5)
    view.set_tool("none")
    assert view.is_fit() and view.zoom() == pytest.approx(0.8)
    view.zoom_to(2.0)
    view.set_tool("base")  # the samplers keep the zoom: they need detail
    assert view.zoom() == pytest.approx(2.0)


def test_click_toggles_zoom_and_drag_pans(view):
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(400, 300))
    assert view.zoom() == pytest.approx(1.0)
    visible = view.visible_rect()
    drag(view, QPoint(400, 300), QPoint(300, 250))
    assert view.zoom() == pytest.approx(1.0)  # a drag is not a click
    moved = view.visible_rect()
    assert moved.x() * W == pytest.approx(visible.x() * W + 100, abs=1)
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(400, 300))
    assert view.is_fit()


def test_zoom_to_center_on_and_viewport_signal(view):
    rects = []
    view.viewportChanged.connect(rects.append)
    view.zoom_to(2.0)
    vis = view.visible_rect()
    assert vis.width() == pytest.approx(VIEW_W / 2 / W, abs=0.002)
    assert vis.height() == pytest.approx(VIEW_H / 2 / H, abs=0.002)
    view.center_on(0.25, 0.4)
    vis = view.visible_rect()
    assert vis.center().x() == pytest.approx(0.25, abs=0.002)
    assert vis.center().y() == pytest.approx(0.4, abs=0.002)
    assert rects and rects[-1] == vis
    view.fit()
    assert view.visible_rect().width() == pytest.approx(1.0)


def test_fixed_zoom_survives_an_image_of_the_same_shape(view):
    view.zoom_to(2.0)
    view.center_on(0.3, 0.4)
    visible = view.visible_rect()
    view.set_image(solid("#806040", 2 * W, 2 * H))  # sharper render of the same frame
    assert view.zoom() == pytest.approx(1.0) and not view.is_fit()
    assert view.visible_rect().center().x() == pytest.approx(visible.center().x(), abs=0.002)
    assert view.visible_rect().center().y() == pytest.approx(visible.center().y(), abs=0.002)
    assert view.visible_rect().width() == pytest.approx(visible.width(), abs=0.002)
    view.set_image(solid("#806040", H, W))  # portrait: a different picture, start from fit
    assert view.is_fit()


def test_fill_covers_the_view(view):
    view.fill()
    assert view.zoom() == pytest.approx(VIEW_H / H)
    view.resize(900, 600)
    assert view.zoom() == pytest.approx(VIEW_H / H)


def test_pixel_hovered(view):
    seen = []
    view.pixelHovered.connect(lambda x, y: seen.append((x, y)))
    QTest.mouseMove(view.viewport(), QPoint(3, 3))  # Qt drops a move to where the pointer already is
    QTest.mouseMove(view.viewport(), vp(view, 0.25, 0.75))
    assert seen[-1] == pytest.approx((0.25, 0.75), abs=0.003)
    QTest.mouseMove(view.viewport(), QPoint(400, 20))  # band above the image
    assert seen[-1] == (-1.0, -1.0)
    QTest.mouseMove(view.viewport(), vp(view, 0.5, 0.5))
    QApplication.sendEvent(view.viewport(), QEvent(QEvent.Type.Leave))
    assert seen[-1] == (-1.0, -1.0)


def test_clipping_overlay_is_stretched_over_the_image(view):
    mask = QImage(W // 4, H // 4, QImage.Format.Format_ARGB32)
    mask.fill(QColor(0, 0, 0, 0))
    for x in range(W // 8):
        for y in range(H // 4):
            mask.setPixelColor(x, y, QColor(255, 0, 0))
    view.set_clipping_overlay(mask)
    shot = view.grab().toImage()
    assert shot.pixelColor(vp(view, 0.25, 0.5)).red() > 240
    assert shot.pixelColor(vp(view, 0.75, 0.5)).red() < 100
    view.set_clipping_overlay(None)
    assert view.grab().toImage().pixelColor(vp(view, 0.25, 0.5)).red() < 100


def test_samplers_still_select_regions(view):
    regions = []
    view.regionSelected.connect(lambda tool, rect: regions.append((tool, rect)))
    view.set_tool("base")
    assert view.viewport().cursor().shape() == Qt.CursorShape.BitmapCursor  # eyedropper
    drag(view, vp(view, 0.1, 0.2), vp(view, 0.3, 0.4))
    view.set_tool("neutral")
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(view, 0.5, 0.5))
    assert regions[0][0] == "base"
    approx_rect(regions[0][1], (0.1, 0.2, 0.3, 0.4), 0.003)
    tool, (x0, y0, x1, y1) = regions[1]
    assert tool == "neutral" and x0 < 0.5 < x1 and y0 < 0.5 < y1
    assert view.zoom() == pytest.approx(0.8)  # sampling clicks do not zoom


def test_unknown_tool_falls_back_to_none(view):
    view.set_tool("lasso")
    assert view.tool == "none"


def test_set_image_none_clears(view):
    view.set_image(None)
    assert not view.has_image()
    assert view.visible_rect().isEmpty()
    view.toggle_zoom()  # no image: nothing to do, no error
    assert view.is_fit()


# ---------------------------------------------------------------- gestures
@pytest.mark.parametrize("tool", ["straighten", "base"])
@pytest.mark.parametrize("replacement", [None, "smaller"])
def test_a_new_image_mid_drag_cancels_pixel_gestures(view, tool, replacement):
    emitted = []
    view.straightenLine.connect(lambda a, b: emitted.append((a, b)))
    view.regionSelected.connect(lambda t, r: emitted.append(r))
    view.set_tool(tool)
    port = view.viewport()
    a, b = vp(view, 0.1, 0.5), vp(view, 0.8, 0.45)
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    QTest.mouseMove(port, b)
    view.set_image(None if replacement is None else solid("#806040", W // 2, H // 2))
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    assert emitted == []


def test_another_button_does_not_end_the_drag(view):
    view.set_tool("crop")
    view.set_crop_rect((0.25, 0.25, 0.75, 0.75))
    deltas, finished = [], []
    view.angleDelta.connect(deltas.append)
    view.rotateFinished.connect(lambda: finished.append(True))
    port = view.viewport()
    centre = view.viewportTransform().map(QPointF(0.5 * W, 0.5 * H))
    start = QPoint(round(centre.x() + 320), round(centre.y()))
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(port, start + QPoint(0, 40))
    for button in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
        QTest.mousePress(port, button, Qt.KeyboardModifier.NoModifier, start + QPoint(0, 40))
        QTest.mouseRelease(port, button, Qt.KeyboardModifier.NoModifier, start + QPoint(0, 40))
    assert finished == []
    QTest.mouseMove(port, start + QPoint(0, 80))
    assert len(deltas) == 2  # still rotating
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(0, 80))
    assert finished == [True]
    QTest.mouseMove(port, start + QPoint(0, 120))
    assert len(deltas) == 2


# ---------------------------------------------------------------- guided upright
GUIDE = (0.2, 0.2, 0.3, 0.8)


@pytest.fixture
def guided(view):
    view.set_tool("guided")
    return view


def recorder(signal) -> list:
    seen = []
    signal.connect(lambda *args: seen.append(args[0] if len(args) == 1 else args))
    return seen


def move(widget: QWidget, pos: QPoint, buttons=Qt.MouseButton.NoButton) -> None:
    """A move straight to ``widget``: QTest's would go to whichever window is on top at that
    screen position (other tests leave some) and is dropped if the pointer does not move."""
    p = QPointF(pos)
    QApplication.sendEvent(widget, QMouseEvent(QEvent.Type.MouseMove, p, widget.mapToGlobal(p), Qt.MouseButton.NoButton,
                                               buttons, Qt.KeyboardModifier.NoModifier))


def hover(v, pos: QPoint) -> None:
    move(v.viewport(), pos)


def click(v, pos: QPoint) -> None:
    QTest.mouseClick(v.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)


def right_click(v, pos: QPoint, widget: QWidget | None = None, release: bool = True) -> None:
    """Through the window, as the platform delivers it: the press, then a context menu event."""
    widget = widget or v.viewport()
    window = v.window()
    QTest.qWaitForWindowExposed(window)
    at = widget.mapTo(window, pos)
    QTest.mousePress(window.windowHandle(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, at)
    if release:
        QTest.mouseRelease(window.windowHandle(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, at)


def claims(v, key) -> bool:
    event = QKeyEvent(QEvent.Type.ShortcutOverride, key, Qt.KeyboardModifier.NoModifier)
    event.ignore()
    QApplication.sendEvent(v, event)
    return event.isAccepted()


def test_guided_is_a_frame_tool(view):
    view.set_compare("split", solid("#ff2020"))
    view.set_tool("guided")
    assert view.zoom() < 0.8 and view.is_fit()  # room to grab endpoints on the image edges
    assert color_at(view, 200, VIEW_H // 2).red() < 100  # no comparison while placing guides


def test_dragging_draws_a_guide_reported_once(guided):
    changes, dragging = recorder(guided.guidesChanged), recorder(guided.guideDragging)
    drag(guided, vp(guided, 0.2, 0.8), vp(guided, 0.25, 0.1), steps=6)
    assert len(changes) == 1  # at the release, not on every move
    assert changes[0][0] == pytest.approx((0.2, 0.8, 0.25, 0.1), abs=0.004)
    assert guided.guides() == changes[0]
    assert dragging == [True, False]


def test_a_click_draws_no_guide(guided):
    changes = recorder(guided.guidesChanged)
    click(guided, vp(guided, 0.5, 0.5))
    drag(guided, vp(guided, 0.5, 0.5), vp(guided, 0.5, 0.5) + QPoint(3, 2))
    assert changes == [] and guided.guides() == []
    assert guided.zoom() < 0.8  # nor zooms


def test_at_most_four_guides(guided):
    limit = recorder(guided.guideLimitReached)
    for i in range(4):
        drag(guided, vp(guided, 0.1 + 0.2 * i, 0.2), vp(guided, 0.15 + 0.2 * i, 0.8))
    assert len(guided.guides()) == 4 and limit == []
    changes = recorder(guided.guidesChanged)
    click(guided, vp(guided, 0.5, 0.9))  # a click only deselects
    assert limit == []
    drag(guided, vp(guided, 0.5, 0.9), vp(guided, 0.9, 0.95))
    assert len(limit) == 1 and changes == [] and len(guided.guides()) == 4
    hover(guided, vp(guided, 0.9, 0.5))
    assert guided.viewport().cursor().shape() == Qt.CursorShape.ArrowCursor  # nothing more to draw


def test_dragging_an_endpoint_moves_only_that_end(guided):
    guided.set_guides([GUIDE])
    changes = recorder(guided.guidesChanged)
    port, zoom = guided.viewport(), guided.zoom()
    grab = vp(guided, 0.3, 0.8) + QPoint(3, 2)  # off-centre: the end must not jump to the pointer
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, grab)
    QTest.mouseMove(port, grab + QPoint(20, 0))
    QTest.mouseMove(port, grab + QPoint(36, -18))
    assert changes == []
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, grab + QPoint(36, -18))
    assert changes[0][0] == pytest.approx((0.2, 0.2, 0.3 + 36 / (zoom * W), 0.8 - 18 / (zoom * H)), abs=1e-6)


def test_dragging_the_line_moves_the_guide_up_to_the_edge(guided):
    guided.set_guides([(0.2, 0.3, 0.4, 0.7)])
    mid = vp(guided, 0.3, 0.5)
    hover(guided, mid)
    assert guided.viewport().cursor().shape() == Qt.CursorShape.OpenHandCursor
    drag(guided, mid, mid + QPoint(-500, 0))
    assert guided.guides()[0] == pytest.approx((0.0, 0.3, 0.2, 0.7), abs=1e-9)


def test_a_guide_past_the_edge_is_not_pulled_in():
    from belka.ui.guide_overlay import GuideEditor

    editor = GuideEditor()
    editor.set([(-0.1, 0.2, 0.3, 0.8)])  # remapped by a warp: one end now outside the photo
    editor.begin((0, None), (0.1, 0.5))
    editor.drag((0.05, 0.5))  # further out: refused
    assert editor.guides[0] == pytest.approx((-0.1, 0.2, 0.3, 0.8))
    editor.drag((0.15, 0.5))  # inwards: follows
    assert editor.guides[0] == pytest.approx((-0.05, 0.2, 0.35, 0.8))
    editor.cancel()
    assert editor.guides[0] == pytest.approx((-0.1, 0.2, 0.3, 0.8))


def test_hover_lights_the_guide_and_tells_the_part(guided):
    guided.set_guides([GUIDE])
    port = guided.viewport()
    hover(guided, vp(guided, 0.8, 0.5))
    assert port.cursor().shape() == Qt.CursorShape.CrossCursor
    plain = guided.grab().toImage()
    hover(guided, vp(guided, 0.25, 0.5))
    assert port.cursor().shape() == Qt.CursorShape.OpenHandCursor
    assert guided.grab().toImage() != plain
    hover(guided, vp(guided, 0.3, 0.8))
    assert port.cursor().shape() == Qt.CursorShape.SizeAllCursor
    hover(guided, vp(guided, 0.8, 0.5))
    assert guided.grab().toImage() == plain


def test_delete_removes_the_selected_guide(guided):
    guided.set_guides([GUIDE, (0.6, 0.2, 0.7, 0.8)])
    changes = recorder(guided.guidesChanged)
    assert not claims(guided, Qt.Key.Key_Delete)  # nothing selected: the window's Delete
    click(guided, vp(guided, 0.65, 0.5))
    assert changes == []  # selecting does not move it
    assert claims(guided, Qt.Key.Key_Delete) and claims(guided, Qt.Key.Key_Backspace)
    QTest.keyClick(guided, Qt.Key.Key_Delete)
    assert changes == [[GUIDE]]
    click(guided, vp(guided, 0.2, 0.2))
    QTest.keyClick(guided, Qt.Key.Key_Backspace)
    assert changes[-1] == [] and guided.guides() == []
    assert not claims(guided, Qt.Key.Key_Delete)


def test_a_click_beside_the_guides_deselects(guided):
    guided.set_guides([GUIDE])
    click(guided, vp(guided, 0.25, 0.5))
    assert claims(guided, Qt.Key.Key_Delete)
    click(guided, vp(guided, 0.7, 0.5))
    assert not claims(guided, Qt.Key.Key_Delete)


def test_right_click_on_a_guide_removes_it(guided):
    guided.set_guides([GUIDE])
    changes, menus = recorder(guided.guidesChanged), recorder(guided.contextMenuRequested)
    right_click(guided, vp(guided, 0.25, 0.5))
    assert changes == [[]] and menus == []
    right_click(guided, vp(guided, 0.6, 0.4))  # not on a guide: the window's menu
    (pos, x, y), = menus
    assert (x, y) == pytest.approx((0.6, 0.4), abs=0.004)
    assert pos == guided.viewport().mapToGlobal(vp(guided, 0.6, 0.4))


def test_escape_drops_a_drag_then_leaves_the_tool(guided):
    guided.set_guides([GUIDE])
    changes, dragging = recorder(guided.guidesChanged), recorder(guided.guideDragging)
    cancelled, finished = recorder(guided.toolCancelled), recorder(guided.toolFinished)
    port = guided.viewport()
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(guided, 0.3, 0.8))
    QTest.mouseMove(port, vp(guided, 0.5, 0.6))
    assert claims(guided, Qt.Key.Key_Escape)
    QTest.keyClick(guided, Qt.Key.Key_Escape)
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(guided, 0.5, 0.6))
    assert guided.guides() == [GUIDE] and changes == [] and dragging == [True, False]
    assert cancelled == [] and guided.tool == "guided"
    QTest.keyClick(guided, Qt.Key.Key_Escape)
    assert cancelled == ["guided"]
    assert claims(guided, Qt.Key.Key_Return)
    QTest.keyClick(guided, Qt.Key.Key_Return)
    assert finished == ["guided"]


def test_enter_in_mid_drag_keeps_the_guide(guided):
    changes, finished = recorder(guided.guidesChanged), recorder(guided.toolFinished)
    port = guided.viewport()
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(guided, 0.4, 0.2))
    QTest.mouseMove(port, vp(guided, 0.45, 0.7))
    QTest.keyClick(guided, Qt.Key.Key_Return)
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(guided, 0.45, 0.7))
    assert len(changes) == 1 and changes[0][0] == pytest.approx((0.4, 0.2, 0.45, 0.7), abs=0.004)
    assert finished == ["guided"]


def test_guides_show_only_in_the_tool(view):
    view.set_guides([GUIDE])
    with_guides = view.grab().toImage()
    view.set_guides([])
    assert view.grab().toImage() == with_guides  # not drawn outside the tool
    view.set_tool("guided")
    empty = view.grab().toImage()
    view.set_guides([GUIDE])
    assert view.grab().toImage() != empty
    view.set_tool("none")
    assert view.guides() == [GUIDE]  # kept for the next time


def test_guides_stay_put_across_renders(guided):
    guided.set_guides([GUIDE])
    guided.set_image(solid("#806040", 2 * W, 2 * H))  # sharper render: normalised, so nothing to redo
    assert guided.guides() == [GUIDE]
    changes = recorder(guided.guidesChanged)
    drag(guided, vp(guided, 0.3, 0.8), vp(guided, 0.5, 0.8))
    assert changes[0][0] == pytest.approx((0.2, 0.2, 0.5, 0.8), abs=0.004)


def test_a_remapped_list_mid_drag_carries_the_drag_on(guided):
    guided.set_guides([GUIDE])
    changes, dragging = recorder(guided.guidesChanged), recorder(guided.guideDragging)
    port, zoom = guided.viewport(), guided.zoom()
    start = vp(guided, 0.3, 0.8)
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(port, start + QPoint(20, 0))
    guided.set_guides([(0.25, 0.2, 0.35, 0.8)])  # the render answering the previous edit
    QTest.mouseMove(port, start + QPoint(40, 0))
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(40, 0))
    assert changes[0][0] == pytest.approx((0.25, 0.2, 0.35 + 40 / (zoom * W), 0.8), abs=1e-6)
    # A list without the guide being dragged ends the drag.
    x0, y0, x1, y1 = guided.guides()[0]
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     vp(guided, (x0 + x1) / 2, (y0 + y1) / 2))
    guided.set_guides([])
    QTest.mouseMove(port, vp(guided, 0.5, 0.5))
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(guided, 0.5, 0.5))
    assert len(changes) == 1 and guided.guides() == []
    assert dragging == [True, False, True, False]


def red_pixels(v) -> int:
    import numpy as np

    img = v.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    px = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return int(((px[..., 2] > 200) & (px[..., 1] < 60) & (px[..., 0] < 60)).sum())


def test_loupe_magnifies_the_endpoint_while_dragging(guided):
    img = solid("#406080")
    for x in range(495, 505):
        for y in range(245, 255):
            img.setPixelColor(x, y, QColor(255, 0, 0))
    guided.set_image(img)
    guided.set_guides([(0.2, 0.9, 0.5, 0.5)])
    before = red_pixels(guided)
    port = guided.viewport()
    end = vp(guided, 0.5, 0.5)
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    QTest.mouseMove(port, end + QPoint(1, 0))
    assert red_pixels(guided) > 4 * before  # the red square again, magnified beside the pointer
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end + QPoint(1, 0))
    assert red_pixels(guided) == pytest.approx(before, rel=0.3)


def test_loupe_keeps_off_the_line_and_inside_the_view():
    from PySide6.QtCore import QRectF

    from belka.ui.guide_overlay import loupe_rect

    bounds = QRectF(0, 0, 800, 600)
    at = QPointF(400, 300)
    assert loupe_rect(at, QPointF(400, 500), bounds).bottom() < 300  # line goes down: loupe above
    assert loupe_rect(at, QPointF(400, 100), bounds).top() > 300
    assert loupe_rect(at, QPointF(100, 300), bounds).left() > 400
    corner = loupe_rect(QPointF(10, 10), QPointF(10, 10), bounds)
    assert bounds.contains(corner) and not corner.contains(QPointF(10, 10))


# ---------------------------------------------------------------- grid
def test_grid_draws_lines_over_the_photo(view):
    background = QColor("#406080")
    cell = W / 12  # image px: twelve square cells along the long side
    y = vp(view, 0.5, (H / 2 + cell / 2) / H).y()  # between two horizontal lines
    on_line = vp(view, 0.5, 0.5).x()  # the vertical line through the centre
    between = vp(view, (W / 2 + cell / 2) / W, 0.5).x()
    assert view.grid_mode == "off"
    view.set_grid("grid")
    assert view.grid_mode == "grid"
    shot = view.grab().toImage()
    assert shot.pixelColor(on_line, y).lightness() > background.lightness() + 15
    assert shot.pixelColor(between, y) == background
    view.set_tool("guided")  # independent of the tool
    x = vp(view, 0.5, 0.5).x()
    assert color_at(view, x, vp(view, 0.5, 0.1).y()).lightness() > background.lightness() + 15
    view.set_grid("dots")
    assert view.grid_mode == "off"
    assert color_at(view, x, vp(view, 0.5, 0.1).y()) == background


# ---------------------------------------------------------------- context menu
@pytest.mark.parametrize("tool", ["none", "base", "neutral"])
def test_right_click_asks_for_the_menu(view, tool):
    view.set_tool(tool)
    menus = recorder(view.contextMenuRequested)
    regions = recorder(view.regionSelected)
    right_click(view, vp(view, 0.25, 0.75))
    right_click(view, QPoint(400, 20))  # band above the image
    assert [(x, y) for _pos, x, y in menus] == [pytest.approx((0.25, 0.75), abs=0.004), (-1.0, -1.0)]
    assert menus[0][0] == view.viewport().mapToGlobal(vp(view, 0.25, 0.75))
    assert regions == [] and view.zoom() == pytest.approx(0.8)  # neither samples nor zooms


def test_menu_key_opens_the_menu_at_the_pointer(view):
    menus = recorder(view.contextMenuRequested)
    hover(view, vp(view, 0.4, 0.6))
    QApplication.sendEvent(view, QContextMenuEvent(QContextMenuEvent.Reason.Keyboard, QPoint(0, 0), QPoint(0, 0)))
    assert (menus[0][1], menus[0][2]) == pytest.approx((0.4, 0.6), abs=0.004)


def test_no_menu_in_the_middle_of_a_drag(view):
    menus = recorder(view.contextMenuRequested)
    port = view.viewport()
    QTest.mousePress(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(view, 0.5, 0.5))
    right_click(view, vp(view, 0.5, 0.5))
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp(view, 0.5, 0.5))
    assert menus == []


def test_right_click_on_the_before_pane(view):
    view.set_compare("side", solid("#ff2020"))
    view.zoom_to(2.0)
    visible = view.visible_rect()
    menus = recorder(view.contextMenuRequested)
    pane_pos = vp(view, 0.4, 0.5)  # both halves are as wide, and the pane sits at the view's origin
    right_click(view, pane_pos, widget=view, release=False)
    assert (menus[0][1], menus[0][2]) == pytest.approx((0.4, 0.5), abs=0.004)
    move(view.childAt(pane_pos), pane_pos + QPoint(-60, -40), Qt.MouseButton.RightButton)
    assert view.visible_rect() == visible  # the right button is the menu's, not a pan
    QTest.mouseRelease(view.windowHandle(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, pane_pos)


# ---------------------------------------------------------------- keyboard focus
@pytest.fixture
def window():
    """The view in a window with a combo, a text field and a window shortcut on X (flag as rejected)."""
    from belka.ui.image_view import ImageView

    win = QWidget()
    layout = QVBoxLayout(win)
    view, combo, field = ImageView(), QComboBox(), QLineEdit()
    combo.addItems(["2:3", "1:1"])
    for widget in (view, combo, field):
        layout.addWidget(widget)
    rejected = []
    action = QAction(win)
    action.setShortcut("X")
    action.triggered.connect(lambda: rejected.append(True))
    win.addAction(action)
    win.resize(VIEW_W, VIEW_H + 80)
    win.show()
    win.activateWindow()
    assert QTest.qWaitForWindowActive(win)
    view.set_image(solid("#406080"))
    yield view, combo, field, rejected
    win.close()
    win.deleteLater()


def test_crop_keys_work_from_a_combo_just_used(window):
    view, combo, field, rejected = window
    committed, cancelled = recorder(view.cropCommitted), recorder(view.cropCancelled)
    view.set_tool("crop")
    assert view.hasFocus()
    view.set_crop_rect((0.3, 0.2, 0.5, 0.8))
    view.set_crop_aspect(2 / 3)
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key.Key_X)
    assert view.crop_aspect() == pytest.approx(3 / 2) and rejected == []
    QTest.keyClick(combo, Qt.Key.Key_Return)
    QTest.keyClick(combo, Qt.Key.Key_Escape)
    assert len(committed) == 1 and len(cancelled) == 1
    field.setFocus()  # a text field keeps its keys
    QTest.keyClick(field, Qt.Key.Key_X)
    assert field.text() == "x" and view.crop_aspect() == pytest.approx(3 / 2)
    view.set_tool("none")  # not cropping: X is the window's again
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key.Key_X)
    assert rejected == [True]


def test_guided_keys_work_from_a_combo_but_not_delete(window):
    view, combo, _field, _rejected = window
    view.set_tool("guided")
    view.set_guides([GUIDE])
    click(view, vp(view, 0.25, 0.5))
    finished, changes = recorder(view.toolFinished), recorder(view.guidesChanged)
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key.Key_Delete)  # deleting needs the view's own focus
    QTest.keyClick(combo, Qt.Key.Key_Return)
    assert changes == [] and finished == ["guided"]


def test_a_click_on_the_view_takes_the_focus(window):
    view, combo, _field, _rejected = window
    combo.setFocus()
    QTest.mouseClick(view.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(400, 300))
    assert view.hasFocus()


def test_eyedroppers_hide_the_before_half():
    """Regression: with split compare on, the eyedroppers showed the cropped 'before' over
    the uncropped tool view, so a pick there sampled other pixels."""
    from belka.ui.image_view import ImageView

    view = ImageView()
    view.resize(600, 400)
    image = QImage(300, 200, QImage.Format.Format_RGB888)
    image.fill(0x808080)
    view.set_image(image)
    view.set_compare("split", image)
    for tool in ("base", "neutral"):
        view.set_tool(tool)
        assert view._shown_compare() == "off" and view.compare_mode == "split"
    view.set_tool("none")
    assert view._shown_compare() == "split"
