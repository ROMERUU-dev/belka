import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QContextMenuEvent, QKeySequence, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QMainWindow,
    QMenu,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from belka import i18n
from belka.core import adjust
from belka.core.pipeline import DevelopSettings
from belka.ui import develop_panel as dp
from belka.ui.curve_editor import IDENTITY, ToneCurveEditor
from belka.ui.develop_panel import DevelopPanel
from belka.ui.widgets import SliderRow, TrackSlider, WheelGuard, format_value, parse_value

EVENTS = {
    "press": QEvent.Type.MouseButtonPress,
    "move": QEvent.Type.MouseMove,
    "release": QEvent.Type.MouseButtonRelease,
    "dclick": QEvent.Type.MouseButtonDblClick,
}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def spanish(monkeypatch):
    monkeypatch.setattr(i18n, "_language", "es")


def mouse(widget, kind: str, x: float, y: float, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    pos = QPointF(x, y)
    button = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
    held = Qt.MouseButton.NoButton if kind == "release" else Qt.MouseButton.LeftButton
    event = QMouseEvent(EVENTS[kind], pos, widget.mapToGlobal(pos), button, held, modifiers)
    QApplication.sendEvent(widget, event)


def double_click(widget, x: float, y: float, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    for kind in ("press", "release", "dclick", "release"):
        mouse(widget, kind, x, y, modifiers)


SHIFT = Qt.KeyboardModifier.ShiftModifier


def thumb_x(slider: TrackSlider) -> float:
    lo, hi = slider._min, slider._max
    return TrackSlider.PAD + (slider.value() - lo) / (hi - lo) * (slider.width() - 2 * TrackSlider.PAD)


# QTest's mouse and key functions take the window-system path of a real mouse
# and keyboard: focus moves on press, and the window's shortcuts get the
# ShortcutOverride first. QApplication.sendEvent (mouse() above) skips both.
LEFT, NONE = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier


def click(widget, x: float, y: float) -> None:
    QTest.mouseClick(widget, LEFT, NONE, QPoint(round(x), round(y)))


def key(window, k, modifiers=NONE) -> None:
    """A key on the window, delivered to whatever has the focus, as typed."""
    QTest.keyClick(window.windowHandle(), k, modifiers)


def type_text(window, text: str) -> None:
    for ch in text:
        QTest.keyClick(window.windowHandle(), ch)


@pytest.fixture
def panel(app, library):
    p = DevelopPanel(library)
    p.load(DevelopSettings(), lock_base=False)
    p.setEnabled(True)
    p.sections.set_all_expanded(True)
    p.resize(320, 900)
    p.show()
    app.processEvents()
    p.changes, p.commits, p.signals = [], [], []
    p.settingsChanged.connect(p.changes.append)
    p.editCommitted.connect(p.commits.append)
    for name in ("toolRequested", "uprightRequested", "lockBaseChanged", "autoFieldRequested", "gridToggled",
                 "transformDragging"):
        getattr(p, name).connect(lambda v, n=name: p.signals.append((n, v)))
    for name in ("applyAllRequested", "saveProfileRequested", "resetRequested", "previousRequested",
                 "autoToneRequested"):
        getattr(p, name).connect(lambda n=name: p.signals.append((n, None)))
    yield p
    p.close()


@pytest.fixture
def window(app, library):
    """The panel inside an active window that carries the main window's plain-key shortcuts."""
    win = QMainWindow()
    win.fired = []
    for name, shortcut in (("prev", "Left"), ("next", "Right"), ("delete", QKeySequence.StandardKey.Delete),
                           ("cancel", "Escape"), ("panels", "Tab")):
        action = QAction(name, win)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _c=False, n=name: win.fired.append(n))
        win.addAction(action)
    p = DevelopPanel(library)
    p.load(DevelopSettings(), lock_base=False)
    p.setEnabled(True)
    p.sections.set_all_expanded(True)
    win.setCentralWidget(p)
    win.resize(340, 900)
    win.show()
    win.activateWindow()
    assert QTest.qWaitForWindowActive(win, 2000)
    p.commits = []
    p.editCommitted.connect(p.commits.append)
    win.panel = p
    yield win
    win.close()


@pytest.fixture
def desk(app, library):
    """Photo and panel side by side, with the main window's shortcut contexts.

    As in MainWindow, the frame keys (arrows, 0-5) only fire while the photo
    has the focus, and Tab (hide the panels) anywhere in the window.
    """
    win = QMainWindow()
    win.fired = []
    photo = QWidget()
    photo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    photo.setMinimumWidth(200)
    keys = [("prev", "Left"), ("next", "Right")] + [(f"rate{n}", str(n)) for n in range(6)]
    for name, shortcut in keys:
        action = QAction(name, photo)
        action.setShortcut(QKeySequence(shortcut))
        action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        action.triggered.connect(lambda _c=False, n=name: win.fired.append(n))
        photo.addAction(action)
    panels = QAction("panels", win)
    panels.setShortcut(QKeySequence("Tab"))
    panels.triggered.connect(lambda: win.fired.append("panels"))
    win.addAction(panels)
    p = DevelopPanel(library)
    p.load(DevelopSettings(), lock_base=False)
    p.setEnabled(True)
    p.sections.set_all_expanded(True)
    split = QSplitter()
    split.addWidget(photo)
    split.addWidget(p)
    win.setCentralWidget(split)
    win.resize(560, 900)
    win.show()
    win.activateWindow()
    assert QTest.qWaitForWindowActive(win, 2000)
    photo.setFocus()
    p.commits = []
    p.editCommitted.connect(p.commits.append)
    win.panel, win.photo = p, photo
    yield win
    win.close()


# ---------------------------------------------------------------- units

def test_mappings_use_lightroom_units():
    assert dp.EXPOSURE.display_range() == (-5.0, 5.0)
    assert dp.EXPOSURE.to_model(0.35) == pytest.approx(0.35)
    assert dp.BIPOLAR.to_model(-40) == pytest.approx(-0.4)
    assert dp.BIPOLAR.to_model(250) == pytest.approx(1.0)  # clamped to the range
    assert dp.UNIT.to_model(25) == pytest.approx(0.25)
    # Contrast is a 0.4..2.0 multiplier with 1 neutral, halves interpolated in log space.
    assert dp.CONTRAST.to_model(-100) == pytest.approx(0.4)
    assert dp.CONTRAST.to_model(0) == pytest.approx(1.0)
    assert dp.CONTRAST.to_model(100) == pytest.approx(2.0)
    assert dp.CONTRAST.to_model(50) == pytest.approx(2 ** 0.5)
    assert dp.CONTRAST.to_display(0.4) == pytest.approx(-100)
    # Saturation is a 0..2 multiplier with 1 neutral.
    assert dp.SATURATION.to_model(-100) == pytest.approx(0.0)
    assert dp.SATURATION.to_model(50) == pytest.approx(1.5)
    assert dp.SATURATION.to_display(1.0) == pytest.approx(0.0)
    # Temperature and tint keep the model's -1.5..1.5.
    assert dp.WHITE_BALANCE.to_model(100) == pytest.approx(1.5)
    assert dp.WHITE_BALANCE.to_display(-0.75) == pytest.approx(-50)
    # Negros +100 lifts the blacks, which is a negative model black.
    assert dp.BLACKS.to_model(100) == pytest.approx(-0.3)
    assert dp.BLACKS.to_display(0.15) == pytest.approx(-50)
    assert dp.WHITES.to_model(100) == pytest.approx(0.3)
    assert dp.SHARPEN.to_model(150) == pytest.approx(1.5)
    assert dp.SCALE.to_model(120) == pytest.approx(1.2)
    assert dp.DEGREES.to_model(-2.5) == pytest.approx(-2.5)


def test_numbers_are_formatted_and_parsed_like_lightroom(monkeypatch):
    assert format_value(0.35, 2, True) == "+0,35"
    assert format_value(-0.004, 2, True) == "0,00"
    assert format_value(-12.4, 0, True) == "-12"
    assert format_value(25, 0, False) == "25"
    assert format_value(-1.5, 1, True, "°") == "-1,5°"
    assert parse_value("1,25") == pytest.approx(1.25)
    assert parse_value("+20") == pytest.approx(20)
    assert parse_value("−3,5°") == pytest.approx(-3.5)
    assert parse_value("abc") is None
    monkeypatch.setattr(i18n, "_language", "en")
    assert format_value(0.35, 2, True) == "+0.35"


def test_rows_show_display_units_and_store_model_units(panel):
    panel.rows["contrast"].set_display(-100)
    panel.rows["black"].set_display(50)
    panel.rows["saturation"].set_display(-50)
    panel.rows["temperature"].set_display(20)
    panel.rows["sharpen_amount"].set_display(40)
    s = panel.settings
    assert s.contrast == pytest.approx(0.4)
    assert s.black == pytest.approx(-0.15)
    assert s.saturation == pytest.approx(0.5)
    assert s.temperature == pytest.approx(0.3)
    assert s.sharpen_amount == pytest.approx(0.4)
    assert panel.rows["black"].text() == "+50"
    assert len(panel.changes) == 5 and panel.commits == []


# ---------------------------------------------------------------- gestures

def test_dragging_a_slider_emits_live_and_commits_once(panel):
    slider = panel.rows["exposure"].slider
    x = thumb_x(slider)
    mouse(slider, "press", x, 12)
    mouse(slider, "move", x + 20, 12)
    mouse(slider, "move", x + 40, 12)
    assert panel.commits == []
    mouse(slider, "release", x + 40, 12)
    assert panel.settings.exposure > 0.5
    assert len(panel.changes) == 2
    assert panel.commits == [f"Exposición {panel.rows['exposure'].text()}"]
    assert panel.commits[0].startswith("Exposición +")


def test_clicking_the_thumb_without_moving_changes_nothing(panel):
    slider = panel.rows["clarity"].slider
    x = thumb_x(slider)
    mouse(slider, "press", x + 2, 12)
    mouse(slider, "release", x + 2, 12)
    assert panel.changes == [] and panel.commits == []


def test_typing_a_value_commits_it(window):
    panel = window.panel
    field = panel.rows["exposure"].field
    assert not field.hasFocus()
    click(field, 10, 8)
    # A click opens the field with the number selected, so typing replaces it.
    assert field.hasFocus() and field.selectedText() == "0,00"
    type_text(window, "1,25")
    key(window, Qt.Key.Key_Return)
    assert panel.settings.exposure == pytest.approx(1.25)
    assert panel.commits == ["Exposición +1,25"]
    assert not field.hasFocus()
    # Out of range is clamped, garbage is ignored; Tab confirms like Return.
    click(field, 10, 8)
    type_text(window, "9")
    key(window, Qt.Key.Key_Tab)
    assert panel.settings.exposure == pytest.approx(5.0)
    click(field, 10, 8)
    type_text(window, "xyz")
    key(window, Qt.Key.Key_Return)
    assert panel.settings.exposure == pytest.approx(5.0)
    assert field.text() == "+5,00"
    assert panel.commits == ["Exposición +1,25", "Exposición +5,00"]
    # While typing, Up/Down nudge (committed when the field closes).
    click(field, 10, 8)
    key(window, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    assert field.text() == "+4,90"
    key(window, Qt.Key.Key_Return)
    assert panel.settings.exposure == pytest.approx(4.9)
    assert panel.commits[-1] == "Exposición +4,90"
    assert window.fired == []


def test_pressing_another_value_commits_the_open_one(window):
    panel = window.panel
    exposure, shadows = panel.rows["exposure"].field, panel.rows["shadows"].field
    click(exposure, 10, 8)
    type_text(window, "1")
    QTest.mousePress(shadows, LEFT, NONE, QPoint(20, 8))
    assert panel.settings.exposure == pytest.approx(1.0) and not exposure.hasFocus()
    assert panel.commits == ["Exposición +1,00"]
    for x in (30, 40):
        QTest.mouseMove(shadows, QPoint(x, 8))
    QTest.mouseRelease(shadows, LEFT, NONE, QPoint(40, 8))
    assert panel.commits == ["Exposición +1,00", "Sombras +10"]


def test_escape_cancels_typing(window):
    panel = window.panel
    field = panel.rows["exposure"].field
    click(field, 10, 8)
    type_text(window, "20")
    key(window, Qt.Key.Key_Escape)
    assert field.text() == "0,00" and not field.hasFocus()
    assert panel.settings.exposure == 0.0 and panel.commits == []
    assert window.fired == []  # the window's Escape (cancel tool) did not fire
    # Once closed, the field leaves Escape to the window.
    key(window, Qt.Key.Key_Escape)
    assert window.fired == ["cancel"]


def test_scrubbing_the_value_changes_it(window):
    panel = window.panel
    field = panel.rows["shadows"].field
    QTest.mousePress(field, LEFT, NONE, QPoint(20, 8))
    assert not field.hasFocus()  # a press alone does not open the field for typing
    for x in (25, 30, 35, 40):
        QTest.mouseMove(field, QPoint(x, 8))
    assert panel.settings.shadows == pytest.approx(0.10)  # 20 px at 2 px per step, live
    assert panel.commits == []
    QTest.mouseRelease(field, LEFT, NONE, QPoint(40, 8))
    assert panel.commits == ["Sombras +10"]
    assert not field.hasFocus() and field.text() == "+10"


def test_double_clicking_the_value_resets_it(window):
    panel = window.panel
    panel.apply_external(highlights=-0.6)
    field = panel.rows["highlights"].field
    click(field, 20, 8)
    QTest.mouseDClick(field, LEFT, NONE, QPoint(20, 8))
    assert panel.settings.highlights == 0.0 and field.text() == "0"
    assert panel.commits == ["Altas luces 0"] and not field.hasFocus()


def test_double_click_resets_label_value_and_thumb(panel):
    row = panel.rows["highlights"]
    for target in (row.label, row.field, row.slider):
        row.set_display(-60)
        panel.commits.clear()
        double_click(target, 6, 6)
        assert panel.settings.highlights == 0.0
        assert row.text() == "0"
        assert panel.commits[-1] == "Altas luces 0"


def test_dragging_a_slider_leaves_the_keys_with_the_photo(desk):
    panel = desk.panel
    slider = panel.rows["texture"].slider
    x = thumb_x(slider)
    QTest.mousePress(slider, LEFT, NONE, QPoint(round(x), 12))
    QTest.mouseMove(slider, QPoint(round(x) + 30, 12))
    QTest.mouseRelease(slider, LEFT, NONE, QPoint(round(x) + 30, 12))
    texture = panel.settings.texture
    assert texture > 0 and len(panel.commits) == 1
    assert desk.photo.hasFocus() and not slider.hasFocus()
    # The arrows and digits step through and rate frames; the slider stays put.
    key(desk, Qt.Key.Key_Right)
    key(desk, Qt.Key.Key_Left)
    key(desk, Qt.Key.Key_3)
    assert desk.fired == ["next", "prev", "rate3"]
    assert panel.settings.texture == texture and len(panel.commits) == 1


def test_panel_buttons_leave_the_keys_with_the_photo(desk):
    panel = desk.panel
    for button in (panel.auto_tone, panel.show_grid, panel.upright_buttons["auto"], panel.apply_all_btn,
                   panel.sections.section("detail").header.switch):
        click(button, button.width() / 2, button.height() / 2)
        assert desk.photo.hasFocus(), button
    key(desk, Qt.Key.Key_5)
    assert desk.fired == ["rate5"]


def test_picking_from_a_list_leaves_the_keys_with_the_photo(desk):
    panel = desk.panel
    combo = panel.output_combo
    click(combo, 20, combo.height() / 2)
    assert combo.view().isVisible()
    QTest.keyClick(combo.view(), Qt.Key.Key_Down)
    QTest.keyClick(combo.view(), Qt.Key.Key_Return)
    assert panel.settings.output == "flat" and panel.commits == ["Salida: Plana lineal (para editar)"]
    assert desk.photo.hasFocus()
    # The offscreen platform leaves its hidden list window active; a window manager would not.
    desk.activateWindow()
    assert QTest.qWaitForWindowActive(desk, 2000) and desk.photo.hasFocus()
    key(desk, Qt.Key.Key_Down)
    key(desk, Qt.Key.Key_Right)
    assert desk.fired == ["next"] and panel.settings.output == "flat"


def test_closing_a_typed_value_gives_the_keys_back_to_the_photo(desk):
    panel = desk.panel
    field = panel.rows["exposure"].field
    click(field, 10, 8)
    assert field.hasFocus()
    type_text(desk, "1")
    key(desk, Qt.Key.Key_Right)  # while typing, the arrows are the text cursor's
    key(desk, Qt.Key.Key_Return)
    assert panel.settings.exposure == pytest.approx(1.0) and desk.photo.hasFocus()
    for close in (Qt.Key.Key_Escape, Qt.Key.Key_Tab):
        click(field, 10, 8)
        key(desk, close)
        assert desk.photo.hasFocus() and not field.hasFocus()
    key(desk, Qt.Key.Key_Right)
    assert desk.fired == ["next"]


def test_pressing_a_slider_commits_an_open_value(desk):
    panel = desk.panel
    field = panel.rows["exposure"].field
    click(field, 10, 8)
    type_text(desk, "0,5")
    slider = panel.rows["shadows"].slider
    QTest.mousePress(slider, LEFT, NONE, QPoint(round(thumb_x(slider)), 12))
    assert panel.settings.exposure == pytest.approx(0.5) and not field.hasFocus()
    assert panel.commits == ["Exposición +0,50"]
    QTest.mouseRelease(slider, LEFT, NONE, QPoint(round(thumb_x(slider)), 12))


def test_pressing_a_button_or_the_bare_panel_commits_an_open_value(desk):
    panel = desk.panel
    seen = []
    panel.autoToneRequested.connect(lambda: seen.append(panel.settings.exposure))
    field = panel.rows["exposure"].field
    click(field, 10, 8)
    type_text(desk, "2")
    click(panel.auto_tone, 10, 8)
    assert seen == [pytest.approx(2.0)]  # applied before the button acts
    click(field, 10, 8)
    type_text(desk, "-1")
    body = panel.sections.section("basic").body
    click(body, body.width() / 2, 4)
    assert panel.settings.exposure == pytest.approx(-1.0) and not field.hasFocus()
    assert panel.commits == ["Exposición +2,00", "Exposición -1,00"] and desk.photo.hasFocus()


def test_tab_in_an_open_value_does_not_hide_the_panels(desk):
    panel = desk.panel
    field = panel.rows["contrast"].field
    click(field, 10, 8)
    type_text(desk, "20")
    key(desk, Qt.Key.Key_Tab)
    click(field, 10, 8)
    type_text(desk, "30")
    key(desk, Qt.Key.Key_Backtab, SHIFT)
    assert desk.fired == []
    assert panel.commits == ["Contraste +20", "Contraste +30"]
    key(desk, Qt.Key.Key_Tab)  # with no field open, Tab is the window's
    assert desk.fired == ["panels"]


def test_curve_keys_are_not_taken_by_the_window(window):
    panel = window.panel
    plot = panel.curve_editor.plot
    p = plot.to_px(0.5, 0.6)
    click(plot, p.x(), p.y())
    assert plot.hasFocus() and plot.selected == 1
    x0 = panel.settings.curve_rgb[1][0]
    key(window, Qt.Key.Key_Left)
    assert panel.settings.curve_rgb[1][0] == pytest.approx(x0 - 1 / 255, abs=1e-3)
    key(window, Qt.Key.Key_Delete)
    assert panel.settings.curve_rgb == IDENTITY
    assert window.fired == []
    # With no point selected, the keys are the window's again.
    key(window, Qt.Key.Key_Delete)
    key(window, Qt.Key.Key_Right)
    assert window.fired == ["delete", "next"]


def test_history_names_disambiguate_repeated_labels(panel):
    panel.rows["grain_amount"].reset()
    panel.rows["grain_amount"].set_display(30)
    panel.rows["grain_amount"].reset()
    assert panel.commits == ["Grano: Cantidad 0"]
    panel.rows["hsl_sat.5"].set_display(20)
    panel.rows["hsl_sat.5"].reset()
    assert panel.settings.hsl_sat == (0.0,) * 8
    assert panel.commits[-1] == "Saturación: Azul 0"
    # Hue is "Matiz" (Lightroom's Spanish term); "Tono" is Básico's tone group.
    panel.rows["hsl_hue.0"].slider.pressed.emit()
    panel.rows["hsl_hue.0"].set_display(10)
    panel.rows["hsl_hue.0"].slider.released.emit()
    assert panel.commits[-1] == "Matiz: Rojo +10"
    assert panel.hsl_tabs.button("hue").text() == "Matiz"


# ---------------------------------------------------------------- sections

def test_section_switch_writes_disabled(panel):
    switch = panel.sections.section("curve").header.switch
    switch.click()
    assert "curve" in panel.settings.disabled
    assert panel.commits == ["Desactivar Curva de tonos"]
    switch.click()
    assert "curve" not in panel.settings.disabled
    assert panel.commits[-1] == "Activar Curva de tonos"
    assert panel.sections.section("basic").header.switch is None


def test_loading_shows_disabled_sections(panel):
    panel.load(DevelopSettings(disabled=("hsl", "effects")), lock_base=False)
    assert not panel.sections.section("hsl").is_active()
    assert not panel.sections.section("effects").is_active()
    assert panel.sections.section("detail").is_active()
    assert panel.changes == []


def test_reset_section_restores_only_its_fields(panel):
    panel.apply_external(exposure=1.0, texture=0.5, temperature=0.4, neutral=(0.01, 0.0, -0.01),
                         hsl_hue=(0.5,) * 8)
    panel.reset_section("basic")
    s = panel.settings
    assert (s.exposure, s.texture, s.temperature, s.neutral) == (0.0, 0.0, 0.0, (0.0, 0.0, 0.0))
    assert s.hsl_hue == (0.5,) * 8
    assert panel.rows["exposure"].text() == "0,00"
    assert panel.commits == ["Restablecer Básico"]
    panel.reset_section("basic")  # nothing left to reset: no empty history step
    assert panel.commits == ["Restablecer Básico"]


def test_double_click_on_title_resets_and_keeps_the_panel_open(panel):
    section = panel.sections.section("detail")
    panel.apply_external(sharpen_amount=0.8, nr_color=0.3)
    title = section.header.title.geometry().center()
    double_click(section.header, title.x(), title.y())
    assert section.is_expanded()
    assert (panel.settings.sharpen_amount, panel.settings.nr_color) == (0.0, 0.0)
    assert panel.commits == ["Restablecer Detalle"]
    mouse(section.header, "press", title.x(), title.y())
    mouse(section.header, "release", title.x(), title.y())
    assert not section.is_expanded()


def test_double_click_elsewhere_on_the_header_only_folds_and_unfolds(panel):
    section = panel.sections.section("basic")
    panel.apply_external(exposure=1.2, contrast=1.4, shadows=0.3)
    header = section.header
    for x in (header.width() - 13, 60):  # the disclosure triangle, the empty middle
        double_click(header, x, 15)
        assert section.is_expanded()
    assert (panel.settings.exposure, panel.settings.contrast, panel.settings.shadows) == (1.2, 1.4, 0.3)
    assert panel.commits == []


def test_group_heading_double_click_resets_the_group(panel):
    panel.apply_external(exposure=0.7, black=0.1, texture=0.3)
    heading = next(h for h in panel.sections.section("basic").body.findChildren(dp.SubHeading)
                   if h.label.text() == "Tono")
    double_click(heading.label, 4, 4)
    assert (panel.settings.exposure, panel.settings.black, panel.settings.texture) == (0.0, 0.0, 0.3)
    assert panel.commits == ["Restablecer Tono"]


def test_state_round_trip_and_solo_mode(panel):
    panel.set_state({"expanded": {"film": False, "basic": False, "curve": True, "hsl": False, "grading": False,
                                  "detail": False, "lens": False, "transform": False, "effects": False},
                     "hsl_tab": "lum", "grade_region": "global", "curve_channel": "blue", "solo": True})
    state = panel.get_state()
    assert state["expanded"]["curve"] and not state["expanded"]["basic"]
    assert (state["hsl_tab"], state["grade_region"], state["curve_channel"], state["solo"]) == (
        "lum", "global", "blue", True)
    panel.sections.section("detail").set_expanded(True)
    expanded = panel.get_state()["expanded"]
    assert [k for k, on in expanded.items() if on] == ["detail"]


def test_show_section_opens_and_scrolls_to_it(app, panel):
    panel.sections.set_all_expanded(False)
    panel.sections.section("basic").set_expanded(True)
    panel.resize(320, 400)
    app.processEvents()
    panel.show_section("effects")
    app.processEvents()
    section = panel.sections.section("effects")
    assert section.is_expanded()
    bar = panel.verticalScrollBar()
    assert bar.value() == min(section.y(), bar.maximum()) and bar.value() > 0


def test_right_click_menus_do_not_pile_up(app, panel):
    def close_popup() -> None:
        popup = QApplication.activePopupWidget()
        if popup is not None:
            popup.close()

    header = panel.sections.section("basic").header
    plot = panel.curve_editor.plot
    for _ in range(3):
        for widget in (header, plot):
            QTimer.singleShot(0, close_popup)
            event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(20, 10), widget.mapToGlobal(QPoint(20, 10)))
            QApplication.sendEvent(widget, event)
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert panel.findChildren(QMenu) == []


def test_context_menus(panel):
    def trigger(menu, text: str) -> None:
        next(a for a in menu.actions() if a.text() == text).trigger()

    panel.apply_external(vignette_amount=-0.4)
    trigger(panel.sections.panel_menu("effects"), "Restablecer Efectos")
    assert panel.settings.vignette_amount == 0.0 and panel.commits == ["Restablecer Efectos"]
    trigger(panel.sections.panel_menu("effects"), "Modo individual")
    assert panel.sections.solo
    assert sum(on for on in panel.get_state()["expanded"].values()) == 1
    trigger(panel.sections.panel_menu("effects"), "Contraer todos los paneles")
    assert not any(panel.get_state()["expanded"].values())

    panel.apply_external(curve_rgb=((0.0, 0.0), (0.5, 0.6), (1.0, 1.0)), curve_red=((0.0, 0.1), (1.0, 1.0)))
    trigger(panel.curve_editor.context_menu(), "Restablecer todas las curvas")
    assert panel.settings.curve_rgb == IDENTITY and panel.settings.curve_red == IDENTITY
    assert panel.commits[-1] == "Restablecer curva de tonos"


def test_hsl_tabs_show_one_group_or_all(panel):
    groups = {k: g for k, (g, _h) in panel.hsl_groups.items()}
    panel.hsl_tabs.button("sat").click()
    assert [k for k, g in groups.items() if not g.isHidden()] == ["sat"]
    panel.hsl_tabs.button("all").click()
    assert all(not g.isHidden() for g in groups.values())
    assert all(not h.isHidden() for _g, h in panel.hsl_groups.values())


# ---------------------------------------------------------------- curve and wheel

def test_curve_add_drag_and_remove_points(app, panel):
    plot = panel.curve_editor.plot
    p = plot.to_px(0.5, 0.7)
    mouse(plot, "press", p.x(), p.y())
    q = plot.to_px(0.5, 0.8)
    mouse(plot, "move", q.x(), q.y())
    mouse(plot, "release", q.x(), q.y())
    curve = panel.settings.curve_rgb
    assert len(curve) == 3 and curve[1][0] == pytest.approx(0.5, abs=0.01)
    assert curve[1][1] == pytest.approx(0.8, abs=0.01)
    assert panel.commits == ["Curva de tonos RGB"]
    # Dragging it off the graph removes it.
    mouse(plot, "press", q.x(), q.y())
    mouse(plot, "move", q.x(), plot.plot_rect().top() - 60)
    assert panel.settings.curve_rgb == IDENTITY  # gone already while outside
    mouse(plot, "release", q.x(), plot.plot_rect().top() - 60)
    assert panel.settings.curve_rgb == IDENTITY
    assert panel.commits[-1] == "Curva de tonos RGB" and len(panel.commits) == 2
    # Double-click adds and removes in one gesture: nothing changes.
    r = plot.to_px(0.3, 0.3)
    double_click(plot, r.x(), r.y())
    assert panel.settings.curve_rgb == IDENTITY


def test_curve_end_points_move_but_stay(app, panel):
    plot = panel.curve_editor.plot
    p = plot.to_px(0.0, 0.0)
    mouse(plot, "press", p.x(), p.y())
    q = plot.to_px(0.1, 0.05)
    mouse(plot, "move", q.x(), q.y())
    mouse(plot, "move", q.x(), plot.plot_rect().bottom() + 80)  # far out: end points are never removed
    mouse(plot, "release", q.x(), plot.plot_rect().bottom() + 80)
    assert panel.settings.curve_rgb[0] == pytest.approx((0.1, 0.0), abs=0.01)
    double_click(plot, *plot.to_px(*panel.settings.curve_rgb[0]).toTuple())
    assert panel.settings.curve_rgb == IDENTITY


def test_curve_channels_keys_and_splits(app, panel):
    editor = panel.curve_editor
    plot = editor.plot
    editor.channels.button("red").click()
    p = plot.to_px(0.5, 0.6)
    mouse(plot, "press", p.x(), p.y())
    mouse(plot, "release", p.x(), p.y())
    assert len(panel.settings.curve_red) == 3 and panel.settings.curve_rgb == IDENTITY
    assert panel.commits == ["Curva de tonos Rojo"]
    y0 = panel.settings.curve_red[1][1]
    QTest.keyClick(plot, Qt.Key.Key_Up, Qt.KeyboardModifier.ShiftModifier)
    assert panel.settings.curve_red[1][1] == pytest.approx(y0 + 10 / 255, abs=1e-3)
    QTest.keyClick(plot, Qt.Key.Key_Delete)
    assert panel.settings.curve_red == IDENTITY
    # Split handles exist on the RGB curve only.
    editor.channels.button("rgb").click()
    x = plot.to_px(0.25, 0).x()
    y = plot.plot_rect().bottom() + 9
    mouse(plot, "press", x, y)
    mouse(plot, "move", plot.to_px(0.35, 0).x(), y)
    mouse(plot, "release", plot.to_px(0.35, 0).x(), y)
    assert panel.settings.curve_splits[0] == pytest.approx(0.35, abs=0.01)
    assert panel.commits[-1] == "Divisiones de la curva"
    # Splits cannot cross.
    mouse(plot, "press", plot.to_px(0.35, 0).x(), y)
    mouse(plot, "move", plot.to_px(0.9, 0).x(), y)
    mouse(plot, "release", plot.to_px(0.9, 0).x(), y)
    assert panel.settings.curve_splits[0] == pytest.approx(0.45)


def test_parametric_sliders_draw_their_curve(panel):
    plot = panel.curve_editor.plot
    assert plot.param_lut is None
    panel.rows["curve_lights"].set_display(50)
    assert panel.settings.curve_lights == pytest.approx(0.5)
    lut = plot.param_lut
    assert lut is not None and lut[len(lut) * 5 // 8] > 5 / 8
    np.testing.assert_allclose(plot.rendered_curve(), lut, atol=1e-6)


def test_the_rgb_line_is_the_curve_the_pipeline_applies(app, panel):
    """Region sliders and a point curve together: the bright line is point(parametric(x))."""
    panel.apply_external(curve_lights=0.6, curve_shadows=-0.5, curve_highlights=0.3,
                         curve_rgb=((0.0, 0.0), (0.5, 0.4), (1.0, 1.0)))
    s = panel.settings
    x = np.linspace(0.0, 1.0, 256)
    expected = np.interp(adjust.parametric_lut(s, 256), x, adjust.curve_lut(s.curve_rgb, 256))
    plot = panel.curve_editor.plot
    np.testing.assert_allclose(plot.rendered_curve(), expected, atol=1e-6)
    # And that is what is painted in the channel colour, at x = 0.625 (Claros).
    image = plot.grab().toImage()
    ratio = image.width() / plot.width()

    def brightness_near(px: QPointF) -> int:
        return max(QColor(image.pixel(round(px.x() * ratio) + dx, round(px.y() * ratio) + dy)).lightness()
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1))

    on_curve = plot.to_px(0.625, float(np.interp(0.625, x, expected)))
    on_point_curve = plot.to_px(0.625, float(np.interp(0.625, x, adjust.curve_lut(s.curve_rgb, 256))))
    assert abs(on_curve.y() - on_point_curve.y()) > 8
    assert brightness_near(on_curve) > 170
    assert brightness_near(on_point_curve) < 140  # the point curve only shows faintly
    # On R/G/B the line is that channel's own point curve.
    panel.curve_editor.set_channel("red")
    np.testing.assert_allclose(plot.rendered_curve(), x, atol=1e-6)


def test_tone_curve_editor_alone(app, request):
    editor = ToneCurveEditor()
    request.addfinalizer(editor.close)  # a window left open takes later tests' hover events
    editor.resize(280, 330)
    editor.show()
    app.processEvents()
    seen = []
    editor.curveChanged.connect(lambda ch, pts: seen.append((ch, pts)))
    editor.set_curves({"rgb": ((0, 0), (0.5, 0.6), (1, 1))})
    editor.set_channel("green")
    assert editor.curve("rgb")[1] == (0.5, 0.6) and seen == []
    assert editor.grab().width() == 280


def test_color_wheel_sets_hue_and_saturation(app, panel):
    wheel = panel.grade_wheel
    c = QPointF(wheel.width() / 2, wheel.height() / 2)
    r = min(wheel.width(), wheel.height()) / 2 - 6
    mouse(wheel, "press", c.x() + r * 0.5, c.y())
    mouse(wheel, "move", c.x(), c.y() - r * 0.5)
    mouse(wheel, "release", c.x(), c.y() - r * 0.5)
    hue, sat, lum = panel.settings.grade_shadows
    assert hue == pytest.approx(90, abs=1) and sat == pytest.approx(0.5, abs=0.01) and lum == 0.0
    assert panel.rows["grade_shadows.0"].text() == "90"
    assert panel.commits == ["Gradación de color: Sombras"]
    # The other regions are untouched and the wheel follows the chosen region.
    panel.grade_tabs.button("highlights").click()
    assert panel.grade_wheel.value() == (0.0, 0.0)
    panel.rows["grade_highlights.0"].set_display(200)
    panel.rows["grade_highlights.1"].set_display(30)
    assert panel.grade_wheel.value() == pytest.approx((200.0, 0.3))
    assert panel.settings.grade_shadows[:2] == pytest.approx((hue, sat))
    # Double-clicking the wheel clears hue and saturation but keeps the luminance.
    panel.rows["grade_highlights.2"].set_display(-20)
    double_click(wheel, c.x() + r * 0.3, c.y())
    assert panel.settings.grade_highlights == (0.0, 0.0, pytest.approx(-0.2))
    assert panel.rows["grade_highlights.1"].text() == "0"
    assert panel.commits[-1] == "Gradación de color: Luces altas"


# ---------------------------------------------------------------- loading and signals

def test_load_and_apply_external_round_trip(panel, library):
    s = DevelopSettings(
        profile_id="kodak-portra-400", exposure=1.234, temperature=0.123, tint=-0.4, contrast=0.7, black=0.05,
        white=-0.02, saturation=1.3, highlights=-0.5, shadows=0.4, texture=0.1, clarity=-0.2, vibrance=0.3,
        curve_rgb=((0.0, 0.0), (0.4, 0.5), (1.0, 1.0)), curve_blue=((0.0, 0.1), (1.0, 0.9)), curve_lights=0.2,
        curve_splits=(0.2, 0.5, 0.8), hsl_hue=(0.1, 0, 0, 0, 0, 0, 0, -0.3), hsl_lum=(0, 0, 0.25, 0, 0, 0, 0, 0),
        grade_midtones=(120.0, 0.3, 0.1), grade_blending=0.7, sharpen_amount=0.6, sharpen_radius=1.5,
        lens_distortion=0.2, persp_vertical=-0.3, persp_rotate=1.5, persp_scale=1.1, vignette_amount=-0.3,
        grain_amount=0.2, disabled=("hsl", "lens"), constrain_crop=False, separation=0.5, output="flat",
        auto_crop=False,
    )
    panel.load(s, lock_base=True)
    assert panel.settings == s
    assert panel.changes == [] and panel.commits == []
    assert panel.rows["exposure"].text() == "+1,23"
    assert panel.rows["black"].text() == "-17"
    assert panel.rows["contrast"].display_value() == pytest.approx(dp.CONTRAST.to_display(0.7))
    assert panel.rows["persp_rotate"].text() == "+1,5°"
    assert panel.rows["persp_scale"].text() == "110"
    assert panel.rows["hsl_hue.7"].text() == "-30"
    assert panel.curve_editor.curve("rgb") == s.curve_rgb and panel.curve_editor.splits() == s.curve_splits
    assert not panel.sections.section("hsl").is_active() and not panel.sections.section("lens").is_active()
    assert panel.output_combo.currentData() == "flat" and not panel.auto_crop.isChecked()
    assert panel.profile_combo.currentData() == "kodak-portra-400"
    assert panel.base_lock.isChecked() and not panel.base_auto.isEnabled()
    assert not panel.separation_profile.isChecked() and panel.separation.isEnabled()
    assert panel.separation.value() == pytest.approx(0.5)
    assert not panel.constrain_crop.isChecked()
    # Touching one control leaves every other value exactly as loaded.
    panel.rows["exposure"].set_display(0.5)
    assert panel.settings == s.copy(exposure=0.5)
    panel.apply_external(angle=2.5, persp_vertical=0.4, crop=(0.1, 0.1, 0.9, 0.9))
    assert panel.changes[-1] == s.copy(exposure=0.5, angle=2.5, persp_vertical=0.4, crop=(0.1, 0.1, 0.9, 0.9))
    assert panel.rows["persp_vertical"].text() == "+40"
    assert panel.commits == []
    # A DevelopSettings copy, not the panel's own object.
    assert panel.changes[-1] is not panel.settings


def test_film_names_carry_the_brand_once(panel):
    combo = panel.profile_combo
    # No leading spaces (the list indents through the delegate) and no "CineStill CineStill 800T".
    assert combo.itemText(combo.findData("kodak-portra-400")) == "Kodak Portra 400"
    assert combo.itemText(combo.findData("cinestill-800t")) == "CineStill 800T"
    assert combo.itemText(combo.findData("generic-c41")) == "Genérico C-41"
    assert all(combo.itemText(i) == combo.itemText(i).strip() for i in range(combo.count()))
    index = combo.findData("cinestill-400d")
    combo.setCurrentIndex(index)  # what picking it in the list does, before ``activated``
    combo.activated.emit(index)
    assert panel.commits == ["Película: CineStill 400D"]
    assert combo.currentText() == "CineStill 400D"


def test_profile_output_and_crop_controls(panel, library):
    combo = panel.profile_combo
    combo.activated.emit(combo.findData("ilford-hp5-plus"))
    assert panel.settings.profile_id == "ilford-hp5-plus"
    assert panel.commits == ["Película: Ilford HP5 Plus"]
    # Black and white: the colour controls are off.
    assert not panel.rows["temperature"].isEnabled() and not panel.rows["saturation"].isEnabled()
    assert not panel.sections.section("hsl").body.isEnabled()
    assert panel.rows["exposure"].isEnabled()
    combo.activated.emit(combo.findData("kodak-portra-400"))
    assert panel.rows["temperature"].isEnabled()
    panel.output_combo.activated.emit(panel.output_combo.findData("flat"))
    assert panel.settings.output == "flat"
    panel.apply_external(crop=(0.1, 0.1, 0.8, 0.8))
    assert not panel.auto_crop.isChecked()
    panel.auto_crop.click()
    assert panel.settings.auto_crop and panel.settings.crop is None


def test_separation_follows_the_profile_or_its_slider(panel):
    profile = panel.current_profile()
    panel.separation_profile.click()
    assert panel.settings.separation == pytest.approx(profile.separation)
    assert panel.separation.isEnabled()
    panel.separation.set_display(70)
    assert panel.settings.separation == pytest.approx(0.7)
    panel.separation_profile.click()
    assert panel.settings.separation is None and not panel.separation.isEnabled()
    assert panel.commits == ["Separación propia", "Separación del perfil"]


def test_buttons_emit_their_requests(panel):
    panel.neutral_pick.click()
    panel.base_pick.click()
    panel.upright_buttons["auto"].click()
    panel.upright_buttons["off"].click()
    panel.base_lock.click()
    panel.apply_all_btn.click()
    panel.save_profile_btn.click()
    panel.previous_btn.click()
    panel.reset_btn.click()
    assert panel.signals == [
        ("toolRequested", "neutral"), ("toolRequested", "base"), ("uprightRequested", "auto"),
        ("uprightRequested", "off"), ("lockBaseChanged", True), ("applyAllRequested", None),
        ("saveProfileRequested", None), ("previousRequested", None), ("resetRequested", None),
    ]
    assert not panel.base_auto.isEnabled()


def test_neutral_and_base_can_be_cleared(panel):
    panel.apply_external(neutral=(0.02, -0.01, 0.0), base=(0.5, 0.3, 0.2))
    assert not panel.neutral_clear.isHidden()
    assert panel.base_label.text() == "Medida en la imagen"
    panel.neutral_clear.click()
    panel.base_auto.click()
    assert panel.settings.neutral == (0.0, 0.0, 0.0) and panel.settings.base is None
    assert panel.neutral_clear.isHidden()
    assert panel.commits == ["Quitar gris neutro", "Base automática"]


def test_panel_renders_with_everything_open(app, panel):
    panel.set_exposure_note("⚠ Subexpuesta")
    import numpy as np

    panel.set_histogram(np.ones((3, 256)) * 50)
    panel.apply_external(curve_lights=0.4, grade_shadows=(200.0, 0.4, 0.0))
    panel.resize(320, 4000)
    app.processEvents()
    image = panel.grab().toImage()
    assert image.width() == 320
    assert not panel.exposure_note.isHidden()
    # The scrolled content ends above the fixed bottom bar.
    assert panel.viewport().geometry().bottom() < panel.bottom_bar.geometry().top()


def test_scrollbar_stops_above_the_bottom_bar(app, panel):
    bar = panel.bottom_bar
    scroll = panel.verticalScrollBar()
    for height in (700, 520, 860):
        panel.resize(300, height)
        app.processEvents()
        assert scroll.isVisible()
        bottom = scroll.mapTo(panel, QPoint(0, scroll.height())).y()
        assert bottom <= bar.geometry().top() == height - panel.BAR_HEIGHT
    # Qt lays the scrollbar out again when it hides and shows with the content height.
    panel.sections.set_all_expanded(False)
    app.processEvents()
    panel.sections.set_all_expanded(True)
    app.processEvents()
    assert scroll.mapTo(panel, QPoint(0, scroll.height())).y() <= bar.geometry().top()
    scroll.setValue(scroll.maximum())
    image = panel.grab().toImage()
    # The bar's own background, not the scrollbar handle, fills its right edge.
    ratio = image.width() / panel.width()
    corner = QColor(image.pixel(round((panel.width() - 5) * ratio), round((panel.height() - 4) * ratio)))
    assert corner == QColor(dp.GROUND)


def test_disabled_colour_tracks_look_like_plain_ones(app):
    plain, colour = TrackSlider(-100, 100, 1), TrackSlider(-100, 100, 1)
    colour.set_gradient(dp.TEMP_TRACK)
    for slider in (plain, colour):
        slider.resize(160, 18)
        slider.set_value(30)
        slider.setEnabled(False)
    assert plain.grab().toImage() == colour.grab().toImage()
    colour.setEnabled(True)
    assert plain.grab().toImage() != colour.grab().toImage()


def test_slider_row_alone_reports_model_values(app):
    row = SliderRow("Contraste", dp.CONTRAST, 1.0)
    seen, done = [], []
    row.valueChanged.connect(seen.append)
    row.editFinished.connect(done.append)
    row.set_value(2.0)
    assert row.text() == "+100" and seen == []
    row.reset()
    assert seen == [1.0] and done == ["Contraste 0"]


# ---------------------------------------------------------------- 0.2.1: auto tone, Upright, grid, wheel

def test_auto_button_beside_tono_asks_for_auto_tone(panel):
    heading = next(h for h in panel.sections.section("basic").body.findChildren(dp.SubHeading)
                   if h.label.text() == "Tono")
    assert panel.auto_tone.parent() is heading and panel.auto_tone.text() == "Auto"
    # At the right end of the heading, like Lightroom's.
    assert panel.auto_tone.geometry().left() > heading.width() / 2
    panel.auto_tone.click()
    assert panel.signals == [("autoToneRequested", None)]
    assert panel.changes == [] and panel.commits == []


def test_shift_double_click_asks_for_one_automatic_value(panel):
    panel.apply_external(shadows=0.4, texture=0.3)
    panel.changes.clear()
    for field in dp.TONE_FIELDS:
        row = panel.rows[field]
        for target in (row.label, row.field, row.slider):
            panel.signals.clear()
            x = thumb_x(row.slider) if target is row.slider else 6
            double_click(target, x, 6, SHIFT)
            assert panel.signals == [("autoFieldRequested", field)], (field, target)
        assert not row.field.hasFocus() and not row.field._editing  # the click before it opened the field
    # The panel only asks: the window computes and applies the value.
    assert panel.settings.shadows == pytest.approx(0.4) and panel.changes == [] and panel.commits == []
    assert "Mayús+doble clic" in panel.rows["exposure"].label.toolTip()
    # A plain double-click still resets; elsewhere Shift+double-click resets too.
    double_click(panel.rows["shadows"].label, 6, 6)
    double_click(panel.rows["texture"].label, 6, 6, SHIFT)
    assert (panel.settings.shadows, panel.settings.texture) == (0.0, 0.0)
    assert panel.commits == ["Sombras 0", "Textura 0"]
    assert "Mayús+doble clic" not in panel.rows["texture"].label.toolTip()


def test_shift_double_click_on_a_group_heading_still_resets_it(panel):
    panel.apply_external(exposure=0.7)
    heading = next(h for h in panel.sections.section("basic").body.findChildren(dp.SubHeading)
                   if h.label.text() == "Tono")
    double_click(heading.label, 4, 4, SHIFT)
    assert panel.settings.exposure == 0.0 and panel.commits == ["Restablecer Tono"]


def test_upright_buttons_show_the_applied_mode(panel):
    buttons = panel.upright_buttons
    assert list(buttons) == ["off", "auto", "guided", "level", "vertical", "full"]
    assert [b.text() for b in buttons.values()] == ["Desactivado", "Auto", "Guiada", "Nivel", "Vertical",
                                                    "Completo"]
    assert "Mayús+T" in buttons["guided"].toolTip()

    def lit() -> list[str]:
        return [mode for mode, b in buttons.items() if b.isChecked()]

    assert lit() == []  # upright_mode "": nothing applied, nothing lit
    buttons["guided"].click()
    buttons["auto"].click()
    assert panel.signals == [("uprightRequested", "guided"), ("uprightRequested", "auto")]
    assert lit() == []  # a click only asks; the window applies the mode
    panel.apply_external(upright_mode="auto", persp_vertical=0.2)
    assert lit() == ["auto"]
    panel.load(DevelopSettings(upright_mode="guided"), lock_base=False)
    assert lit() == ["guided"]
    buttons["full"].click()
    assert lit() == ["guided"] and panel.signals[-1] == ("uprightRequested", "full")
    panel.apply_external(upright_mode="")
    assert lit() == []
    assert panel.commits == []


def test_grid_checkbox_is_view_state(panel):
    assert not panel.show_grid.isChecked()
    panel.show_grid.click()
    assert panel.signals == [("gridToggled", True)]
    panel.show_grid.click()
    assert panel.signals[-1] == ("gridToggled", False)
    panel.set_grid_checked(True)
    assert panel.show_grid.isChecked() and len(panel.signals) == 2
    panel.load(DevelopSettings(), lock_base=False)
    assert panel.show_grid.isChecked()
    assert panel.changes == [] and panel.commits == []


def test_transform_and_lens_sliders_report_drags(panel):
    for field in ("angle", "persp_vertical", "persp_scale", "lens_distortion", "lens_vignette"):
        slider = panel.rows[field].slider
        x = thumb_x(slider)
        panel.signals.clear()
        mouse(slider, "press", x, 12)
        assert panel.signals == [("transformDragging", True)]
        mouse(slider, "move", x + 15, 12)
        mouse(slider, "release", x + 15, 12)
        assert panel.signals == [("transformDragging", True), ("transformDragging", False)], field
    # Scrubbing the number is a drag too.
    field = panel.rows["persp_horizontal"].field
    panel.signals.clear()
    mouse(field, "press", 20, 8)
    mouse(field, "move", 30, 8)
    assert panel.signals == [("transformDragging", True)]
    mouse(field, "release", 30, 8)
    assert panel.signals == [("transformDragging", True), ("transformDragging", False)]
    # Other panels do not.
    panel.signals.clear()
    slider = panel.rows["texture"].slider
    mouse(slider, "press", thumb_x(slider), 12)
    mouse(slider, "release", thumb_x(slider), 12)
    assert panel.signals == []


def test_angle_slider_straightens(panel):
    row = panel.rows["angle"]
    transform = panel.sections.section("transform").body
    rows = [r for r in transform.findChildren(dp.SliderRow)]
    assert rows[0] is row and row.label.text() == "Ángulo"
    assert row.mapping.display_range() == (-45.0, 45.0)
    row.slider.pressed.emit()
    row.set_display(12.34)
    row.slider.released.emit()
    assert panel.settings.angle == pytest.approx(12.3)
    assert panel.commits == ["Ángulo +12,3°"]
    panel.apply_external(angle=-3.5)  # from the crop tool or the straighten line
    assert row.text() == "-3,5°"


def test_resetting_transform_forgets_the_upright(panel):
    guides = ((0.1, 0.1, 0.12, 0.9), (0.8, 0.1, 0.78, 0.9))
    panel.apply_external(angle=2.0, persp_vertical=0.3, persp_scale=1.2, upright_mode="guided",
                         upright_guides=guides, constrain_crop=False, lens_distortion=0.2)
    heading = next(h for h in panel.sections.section("transform").body.findChildren(dp.SubHeading)
                   if h.label.text() == "Transformar")
    double_click(heading.label, 4, 4)
    s = panel.settings
    assert (s.angle, s.persp_vertical, s.persp_scale, s.upright_mode, s.upright_guides) == (0.0, 0.0, 1.0, "", ())
    assert not s.constrain_crop and s.lens_distortion == pytest.approx(0.2)
    assert not any(b.isChecked() for b in panel.upright_buttons.values())
    panel.reset_section("transform")
    assert panel.settings.constrain_crop
    assert panel.commits == ["Restablecer Transformar", "Restablecer Transformar"]


def wheel(widget, dy: int = -120) -> None:
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(), QPoint(0, dy), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


def test_wheel_scrolls_past_combos_and_spin_boxes_even_focused(app, request):
    area = QScrollArea()
    request.addfinalizer(area.close)
    area.setWidgetResizable(True)
    body = QWidget()
    layout = QVBoxLayout(body)
    combo, spin, slider = QComboBox(), QDoubleSpinBox(), QSlider(Qt.Orientation.Horizontal)
    combo.addItems(["a", "b", "c"])
    spin.setValue(5.0)
    slider.setRange(0, 100)
    slider.setValue(50)
    for widget in (combo, spin, slider):
        layout.addWidget(widget)
    layout.addSpacing(2000)
    area.setWidget(body)
    area.resize(200, 300)
    WheelGuard(area).guard_children(body)
    area.show()
    area.activateWindow()
    assert QTest.qWaitForWindowActive(area, 2000)
    bar = area.verticalScrollBar()
    for widget in (combo, spin):
        widget.setFocus()
        assert widget.hasFocus()
        before = bar.value()
        wheel(widget)
        assert bar.value() > before, widget
    assert combo.currentIndex() == 0 and spin.value() == 5.0
    # An open list scrolls itself, not the panel.
    combo.showPopup()
    before = bar.value()
    wheel(combo)
    assert bar.value() == before
    combo.hidePopup()
    # A slider takes the wheel only while it has the focus.
    before = bar.value()
    slider.clearFocus()
    wheel(slider)
    assert bar.value() > before and slider.value() == 50
    slider.setFocus()
    assert slider.hasFocus()
    before = bar.value()
    wheel(slider)
    assert bar.value() == before and slider.value() != 50
