"""Develop history (undo, redo, coalescing) and the Develop module's left panels."""

import itertools

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QFocusEvent, QImage, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow

from belka.core.film import FilmProfile, ProfileLibrary
from belka.core.history import COALESCE_SECONDS, MAX_ENTRIES, History, HistoryStore, split_label
from belka.core.pipeline import DevelopSettings


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def history(clock):
    return History(DevelopSettings(), "Importar", clock=clock)


def exposures(h: History) -> list[float]:
    return [e.settings.exposure for e in h.entries()]


# ---------------------------------------------------------------- History

def test_undo_redo_walk_the_steps(history, clock):
    for ev in (0.5, 1.0, 1.5):
        clock.advance(5)
        assert history.push(f"Exposición +{ev}", DevelopSettings(exposure=ev))
    assert history.index == 3 and history.can_undo and not history.can_redo
    assert history.undo().exposure == 1.0
    assert history.undo().exposure == 0.5
    assert history.can_redo
    assert history.redo().exposure == 1.0
    assert history.current.exposure == 1.0
    assert history.undo().exposure == 0.5
    assert history.undo().exposure == 0.0
    assert history.undo() is None and not history.can_undo
    assert history.index == 0


def test_redo_at_the_end_does_nothing(history):
    assert history.redo() is None


def test_new_edit_after_undo_discards_the_undone_steps(history, clock):
    for ev in (0.5, 1.0):
        clock.advance(5)
        history.push("Exposición", DevelopSettings(exposure=ev))
    history.undo()
    clock.advance(5)
    history.push("Contraste", DevelopSettings(exposure=0.5, contrast=1.2))
    assert [e.label for e in history.entries()] == ["Importar", "Exposición", "Contraste"]
    assert not history.can_redo


def test_jump_moves_anywhere_and_rejects_bad_indices(history, clock):
    for ev in (0.5, 1.0, 1.5):
        clock.advance(5)
        history.push(f"Exposición {ev}", DevelopSettings(exposure=ev))
    assert history.jump(1).exposure == 0.5
    assert history.index == 1 and history.can_undo and history.can_redo
    assert len(history) == 4  # jumping keeps the later steps until a new edit
    with pytest.raises(IndexError):
        history.jump(4)
    with pytest.raises(IndexError):
        history.jump(-1)


def test_returned_settings_are_copies(history, clock):
    clock.advance(5)
    pushed = DevelopSettings(exposure=0.5)
    history.push("Exposición", pushed)
    pushed.exposure = 9.0  # the caller keeps mutating its own object
    undone = history.undo()
    undone.exposure = 7.0
    assert exposures(history) == [0.0, 0.5]
    assert history.jump(1).exposure == 0.5


def test_push_without_change_is_ignored(history, clock):
    clock.advance(5)
    assert not history.push("Exposición +0,00", DevelopSettings())
    assert len(history) == 1


def test_same_control_within_a_second_folds_into_one_step(history, clock):
    clock.advance(5)
    history.push("Exposición +0,10", DevelopSettings(exposure=0.1))
    # Wheel nudges 0.4 s apart: a sliding window, so all of them fold.
    for ev in (0.2, 0.3, 0.4):
        clock.advance(0.4)
        history.push(f"Exposición +0,{int(ev * 10)}0", DevelopSettings(exposure=ev))
    assert [e.label for e in history.entries()] == ["Importar", "Exposición +0,40"]
    assert history.current.exposure == 0.4
    assert history.undo().exposure == 0.0


def test_coalescing_stops_after_the_window(history, clock):
    clock.advance(5)
    history.push("Exposición +0,10", DevelopSettings(exposure=0.1))
    clock.advance(COALESCE_SECONDS + 0.01)
    history.push("Exposición +0,20", DevelopSettings(exposure=0.2))
    assert exposures(history) == [0.0, 0.1, 0.2]


def test_coalescing_needs_the_same_control(history, clock):
    clock.advance(5)
    history.push("Exposición +0,10", DevelopSettings(exposure=0.1))
    clock.advance(0.2)
    history.push("Contraste +10", DevelopSettings(exposure=0.1, contrast=1.1))
    assert len(history) == 3


def test_coalescing_needs_the_same_step_name(history, clock):
    clock.advance(5)
    history.push("Enderezar −1,0°", DevelopSettings(angle=-1.0))
    clock.advance(0.2)
    history.push("Rotar +2,0°", DevelopSettings(angle=2.0))  # same field, another control
    assert [e.label for e in history.entries()] == ["Importar", "Enderezar −1,0°", "Rotar +2,0°"]


def test_sliders_of_one_section_stay_separate_steps(history, clock):
    s = DevelopSettings()
    for label, change in (("Enfoque: Cantidad 40", {"sharpen_amount": 0.4}),
                          ("Enfoque: Radio 1,5", {"sharpen_radius": 1.5}),
                          ("Sombras 10", {"shadows": 0.1}),
                          ("Sombras: Matiz 210", {"grade_shadows": (210.0, 0.0, 0.0)})):
        clock.advance(0.5)
        s = s.copy(**change)
        history.push(label, s)
    assert [(e.name, e.value) for e in history.entries()[1:]] == [
        ("Enfoque: Cantidad", "40"), ("Enfoque: Radio", "1,5"), ("Sombras", "10"), ("Sombras: Matiz", "210")]
    history.undo()
    history.undo()
    undone = history.undo()
    assert (undone.sharpen_amount, undone.sharpen_radius) == (0.4, 1.0)


def test_coalescing_needs_the_same_fields_changed(history, clock):
    # A label can be shared once translated; the band that moved tells the controls apart.
    def sat(*bands: float) -> DevelopSettings:
        return DevelopSettings(hsl_sat=bands + (0.0,) * (8 - len(bands)))

    clock.advance(5)
    history.push("Saturación 20", sat(0.2))
    clock.advance(0.3)
    history.push("Saturación 30", sat(0.2, 0.3))
    assert len(history) == 3
    clock.advance(0.3)
    history.push("Saturación 40", sat(0.2, 0.4))  # the same band again: one step
    assert len(history) == 3 and history.current.hsl_sat[1] == 0.4


def test_no_two_develop_sliders_fold_together(qapp, library, clock):
    """Every slider of the real DevelopPanel, nudged right after another, is a step of its own."""
    from belka.ui.develop_panel import DevelopPanel

    panel = DevelopPanel(library)
    start = DevelopSettings()

    def nudged(settings: DevelopSettings, key: str) -> DevelopSettings:
        field, index = panel._bindings[key]
        value = getattr(settings, field)
        if index is None:
            return settings.copy(**{field: value + 0.125})
        return settings.copy(**{field: value[:index] + (value[index] + 0.125,) + value[index + 1:]})

    folded = []
    for first, second in itertools.permutations(panel.rows, 2):
        h = History(start, "Abrir", clock=clock)
        clock.advance(5)
        once = nudged(start, first)
        h.push(panel.rows[first].history_label(), once)
        clock.advance(0.3)
        h.push(panel.rows[second].history_label(), nudged(once, second))
        if len(h) != 3:
            folded.append((first, second))
    assert folded == []
    names = [split_label(row.history_label())[0] for row in panel.rows.values()]
    assert len(set(names)) == len(names)  # and the History panel tells them apart


def test_nudging_back_to_the_start_cancels_the_step(history, clock):
    clock.advance(5)
    history.push("Contraste +10", DevelopSettings(contrast=1.1))
    clock.advance(5)
    history.push("Exposición +0,10", DevelopSettings(contrast=1.1, exposure=0.1))
    clock.advance(0.3)
    assert history.push("Exposición 0,00", DevelopSettings(contrast=1.1))
    assert [e.label for e in history.entries()] == ["Importar", "Contraste +10"]
    assert history.index == 1 and not history.can_redo
    # The next nudge starts a step of its own instead of rewriting "Contraste".
    clock.advance(0.3)
    history.push("Exposición +0,20", DevelopSettings(contrast=1.1, exposure=0.2))
    assert exposures(history) == [0.0, 0.0, 0.2]


def test_coalescing_never_rewrites_the_first_step_or_a_step_jumped_to(clock):
    h = History(DevelopSettings(), "Exposición", clock=clock)
    clock.advance(0.1)
    h.push("Exposición +0,10", DevelopSettings(exposure=0.1))
    assert len(h) == 2
    clock.advance(0.1)
    h.undo()
    clock.advance(0.1)
    h.push("Exposición +0,20", DevelopSettings(exposure=0.2))
    assert exposures(h) == [0.0, 0.2]
    clock.advance(0.1)
    h.jump(0)
    clock.advance(0.1)
    h.push("Exposición +0,30", DevelopSettings(exposure=0.3))
    assert exposures(h) == [0.0, 0.3]


def test_history_keeps_the_start_and_the_newest_steps(history, clock):
    for i in range(1, MAX_ENTRIES + 50):
        clock.advance(5)
        history.push(f"Exposición {i}", DevelopSettings(exposure=float(i)))
    assert len(history) == MAX_ENTRIES
    assert history.index == MAX_ENTRIES - 1
    assert history.current.exposure == MAX_ENTRIES + 49
    # The starting step stays: Before / After compares against it.
    assert history.entries()[0].label == "Importar"
    assert history.entries()[1].settings.exposure == 51.0


def test_clear_keeps_only_the_current_state(history, clock):
    for ev in (0.5, 1.0, 1.5):
        clock.advance(5)
        history.push("Exposición", DevelopSettings(exposure=ev))
    history.undo()
    history.clear("Historial borrado")
    assert len(history) == 1 and history.index == 0
    assert history.entries()[0].label == "Historial borrado"
    assert history.current.exposure == 1.0
    assert not history.can_undo and not history.can_redo


def test_empty_history_takes_its_first_push_as_start(clock):
    h = History(clock=clock)
    assert h.current is None and h.index == -1 and not h.can_undo
    assert h.push("Abrir", DevelopSettings())
    assert h.index == 0 and not h.can_undo
    clock.advance(0.1)
    h.push("Abrir", DevelopSettings(exposure=0.2))  # same name at once: still a new step
    assert exposures(h) == [0.0, 0.2]


def test_entries_unpack_as_label_and_settings(history, clock):
    clock.advance(5)
    history.push("Película: Portra 400", DevelopSettings(profile_id="kodak-portra-400"))
    (first, _s), (label, settings) = history.entries()
    assert first == "Importar" and label == "Película: Portra 400"
    assert settings.profile_id == "kodak-portra-400"
    assert (history.entries()[1].name, history.entries()[1].value) == ("Película", "Portra 400")


@pytest.mark.parametrize("label, expected", [
    ("Exposición +0,35", ("Exposición", "+0,35")),
    ("Temperatura −12", ("Temperatura", "−12")),
    ("Enderezar -1.5°", ("Enderezar", "-1.5°")),
    ("Grano 30 %", ("Grano", "30 %")),
    ("Enfoque: Cantidad 40", ("Enfoque: Cantidad", "40")),
    ("Película: Portra 400", ("Película", "Portra 400")),
    ("Upright: auto", ("Upright", "auto")),
    ("Gradación de color: Sombras", ("Gradación de color", "Sombras")),
    ("Recortar", ("Recortar", "")),
    ("Perfil «Mi Portra»", ("Perfil «Mi Portra»", "")),
    ("400", ("400", "")),
])
def test_split_label(label, expected):
    assert split_label(label) == expected


def test_store_keeps_one_history_per_frame(clock):
    store = HistoryStore(clock=clock)
    a = store.get("001", DevelopSettings(), "Importar")
    b = store.get("002", DevelopSettings(exposure=1.0), "Importar")
    clock.advance(5)
    a.push("Exposición", DevelopSettings(exposure=0.5))
    assert store.get("001") is a and len(a) == 2
    assert store.get("002") is b and len(b) == 1 and b.current.exposure == 1.0
    # An existing history ignores a new starting state.
    assert store.get("001", DevelopSettings(exposure=3.0)).current.exposure == 0.5
    assert "001" in store
    store.discard("001")
    assert "001" not in store
    store.clear()
    assert "002" not in store


def test_store_seeds_a_history_created_empty(clock):
    store = HistoryStore(clock=clock)
    assert len(store.get("001")) == 0
    h = store.get("001", DevelopSettings(exposure=0.3), "Importar")
    assert len(h) == 1 and h.current.exposure == 0.3


# ---------------------------------------------------------------- panels

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance()  # made by conftest before any test module


def recorder(signal) -> list:
    calls = []
    signal.connect(lambda *args: calls.append(args if len(args) != 1 else args[0]))
    return calls


@pytest.fixture
def shown(qapp):
    """Show widgets offscreen and close them afterwards, so no stray window
    catches the synthetic mouse events of later test modules."""
    widgets = []

    def show(widget, width: int = 260, height: int = 400):
        widget.resize(width, height)
        widget.show()
        QApplication.processEvents()
        widgets.append(widget)
        return widget

    yield show
    for widget in widgets:
        widget.close()
        widget.deleteLater()
    QApplication.processEvents()


def test_block_header_click_collapses(shown):
    from belka.ui.left_panels import HistoryPanel

    panel = shown(HistoryPanel())
    toggled = recorder(panel.toggled)
    QTest.mouseClick(panel.header, Qt.MouseButton.LeftButton, pos=QPoint(40, 10))
    assert toggled == [False] and not panel.body.isVisible()
    QTest.mouseClick(panel.header, Qt.MouseButton.LeftButton, pos=QPoint(40, 10))
    assert toggled == [False, True] and panel.body.isVisible()


def test_navigator_zoom_buttons(shown):
    from belka.ui.left_panels import NavigatorPanel

    nav = shown(NavigatorPanel())
    zooms = recorder(nav.zoomRequested)
    assert nav.zoom_buttons["fit"].isChecked()
    nav.zoom_buttons["100"].click()
    nav.zoom_buttons["200"].click()
    assert zooms == ["100", "200"]
    assert nav.zoom_buttons["200"].isChecked() and not nav.zoom_buttons["fit"].isChecked()
    nav.set_zoom(None)  # a wheel zoom matches no preset
    assert not any(b.isChecked() for b in nav.zoom_buttons.values())


def test_navigator_click_and_drag_pan(shown):
    from belka.ui.left_panels import NavigatorPanel

    nav = shown(NavigatorPanel())
    image = QImage(300, 200, QImage.Format.Format_RGB32)
    image.fill(QColor(90, 90, 90))
    nav.set_image(image)
    pans = recorder(nav.panRequested)
    nav.set_viewport((0.0, 0.0, 0.5, 0.5))
    target = nav.view.image_rect()
    center = target.center().toPoint()
    QTest.mouseClick(nav.view, Qt.MouseButton.LeftButton, pos=center)
    x, y = pans[-1]
    assert x == pytest.approx(0.5, abs=0.02) and y == pytest.approx(0.5, abs=0.02)
    x0, y0, x1, y1 = nav.view.view_rect()
    assert (x1 - x0, y1 - y0) == pytest.approx((0.5, 0.5))
    # Dragging to the corner keeps the visible area inside the image.
    QTest.mousePress(nav.view, Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseMove(nav.view, target.topLeft().toPoint())
    QTest.mouseRelease(nav.view, Qt.MouseButton.LeftButton, pos=target.topLeft().toPoint())
    assert pans[-1] == pytest.approx((0.25, 0.25), abs=0.02)
    assert nav.view.view_rect() == pytest.approx((0.0, 0.0, 0.5, 0.5), abs=0.02)


def test_navigator_whole_image_has_no_outline(qapp):
    from belka.ui.left_panels import NavigatorPanel

    nav = NavigatorPanel()
    nav.set_viewport((0.0, 0.0, 1.0, 1.0))
    assert nav.view.view_rect() is None
    nav.set_viewport((0.2, 0.2, 0.6, 0.6))
    assert nav.view.view_rect() == pytest.approx((0.2, 0.2, 0.6, 0.6))
    nav.set_viewport(QRectF(0.25, 0.5, 0.5, 0.25))  # as ImageView.viewportChanged sends it
    assert nav.view.view_rect() == pytest.approx((0.25, 0.5, 0.75, 0.75))
    nav.set_viewport(None)
    assert nav.view.view_rect() is None
    # What ImageView reports once it has no image: no outline, not a dot over a dimmed thumbnail.
    nav.set_viewport(QRectF(0.2, 0.2, 0.6, 0.6))
    nav.set_viewport(QRectF())
    assert nav.view.view_rect() is None
    nav.set_viewport((0.3, 0.3, 0.3, 0.6))
    assert nav.view.view_rect() is None


class ResizeCounter(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def eventFilter(self, _obj, event) -> bool:
        if event.type() == QEvent.Type.Resize:
            self.count += 1
        return False


def test_navigator_settles_whatever_the_column_height(library, shown):
    """Its 3:2 height must not follow the scroll bar, or bar and well chase each other forever."""
    from belka.ui.left_panels import LeftColumn

    column = shown(LeftColumn(library), 270, 500)
    counter = ResizeCounter()
    column.navigator.view.installEventFilter(counter)
    restless = []
    for height in range(500, 700):
        column.resize(270, height)
        for _ in range(6):
            QApplication.processEvents()
        counter.count = 0
        for _ in range(6):
            QApplication.processEvents()
        if counter.count:
            restless.append(height)
    assert restless == []
    # The well still follows the column: no scroll bar at full height, 3:2 of the width.
    column.resize(270, 1000)
    QApplication.processEvents()
    assert column.navigator.view.height() == round(column.navigator.view.width() * 2 / 3)


def test_navigator_keeps_its_own_small_copy(qapp):
    from belka.ui.left_panels import NAVIGATOR_SOURCE, NavigatorPanel

    nav = NavigatorPanel()
    big = QImage(3000, 2000, QImage.Format.Format_RGB32)
    big.fill(QColor(10, 20, 30))
    nav.set_image(big)
    big.fill(QColor(200, 0, 0))  # the caller reuses its buffer
    kept = nav.view._image
    assert max(kept.width(), kept.height()) == NAVIGATOR_SOURCE
    assert kept.pixelColor(5, 5) == QColor(10, 20, 30)


@pytest.fixture
def user_library(tmp_path):
    lib = ProfileLibrary(user_dir=tmp_path / "perfiles")
    lib.save_user_profile(FilmProfile(id="mi-portra", name="Portra 400 (rollo 12)", brand="Kodak"))
    return lib


def top_level(tree) -> list[str]:
    root = tree.invisibleRootItem()
    return [root.child(i).text(0) for i in range(root.childCount())]


def test_profile_browser_groups_mine_first_then_type_and_brand(qapp, user_library):
    from belka.ui.left_panels import ProfileBrowserPanel

    panel = ProfileBrowserPanel(user_library)
    assert top_level(panel.tree) == ["Mis perfiles", "Negativo color", "Blanco y negro", "Diapositiva"]
    color = panel.tree.invisibleRootItem().child(1)
    children = [color.child(i).text(0) for i in range(color.childCount())]
    assert children[:2] == ["Genérico C-41", "Genérico ECN-2 (cine)"]  # generics before brand folders
    assert "Kodak" in children and "Fujifilm" in children
    kodak = next(color.child(i) for i in range(color.childCount()) if color.child(i).text(0) == "Kodak")
    assert kodak.childCount() > 3 and kodak.text(1) == str(kodak.childCount())


def test_profile_browser_without_user_profiles(qapp, library):
    from belka.ui.left_panels import ProfileBrowserPanel

    assert top_level(ProfileBrowserPanel(library).tree)[0] == "Negativo color"


def test_profile_click_emits_and_group_click_toggles(library, shown):
    from belka.ui.left_panels import ProfileBrowserPanel

    panel = shown(ProfileBrowserPanel(library), height=500)
    chosen = recorder(panel.profileChosen)
    panel.set_current("kodak-portra-400")
    assert chosen == []  # programmatic selection does not emit
    item = panel.tree.currentItem()
    assert item.data(0, Qt.ItemDataRole.UserRole) == "kodak-portra-400"
    assert item.parent().isExpanded() and item.parent().parent().isExpanded()
    rect = panel.tree.visualItemRect(item)
    QTest.mouseClick(panel.tree.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert chosen == ["kodak-portra-400"]
    # The activation that follows the click of a double-click does not repeat it.
    panel.tree.itemActivated.emit(item, 0)
    assert chosen == ["kodak-portra-400"]
    other = item.parent().child(0)
    panel.tree.itemActivated.emit(other, 0)  # Enter on another profile
    assert chosen == ["kodak-portra-400", other.data(0, Qt.ItemDataRole.UserRole)]
    group = item.parent()
    rect = panel.tree.visualItemRect(group)
    QTest.mouseClick(panel.tree.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert not group.isExpanded() and len(chosen) == 2
    # Moving to a frame shot on another film opens its folder and leaves the rest alone.
    group.setExpanded(True)
    panel.set_current("ilford-hp5-plus")
    assert panel.tree.currentItem().parent().isExpanded() and group.isExpanded()
    # Repeated after every edit, it leaves a folder the user closed alone.
    hp5 = panel.tree.currentItem()
    hp5.parent().setExpanded(False)
    panel.set_current("ilford-hp5-plus")
    assert not hp5.parent().isExpanded() and chosen == ["kodak-portra-400", other.data(0, Qt.ItemDataRole.UserRole)]
    panel.reload()  # a rebuilt tree shows it again
    panel.set_current("ilford-hp5-plus")
    assert panel.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "ilford-hp5-plus"


def visible_leaves(panel) -> list[str]:
    return [it.data(0, Qt.ItemDataRole.UserRole) for it in panel._items()
            if it.childCount() == 0 and not it.isHidden()]


def test_profile_search_filters_by_words_without_accents(qapp, user_library):
    from belka.ui.left_panels import ProfileBrowserPanel

    panel = ProfileBrowserPanel(user_library)
    panel.search.setText("portra 400")
    assert sorted(visible_leaves(panel)) == ["kodak-portra-400", "mi-portra"]
    assert panel.tree.invisibleRootItem().child(2).isHidden()  # Blanco y negro has no match
    panel.search.setText("GENERICO")
    assert {"generic-c41", "generic-ecn2"} <= set(visible_leaves(panel))
    panel.search.setText("zzz")
    assert visible_leaves(panel) == [] and panel.tree.visible_rows() == 0
    panel.search.clear()
    assert len(visible_leaves(panel)) == len(user_library.all())


def test_snapshots_create_apply_delete(shown):
    from belka.ui.left_panels import SnapshotsPanel

    panel = shown(SnapshotsPanel())
    created, applied, deleted = (recorder(s) for s in (panel.snapshotCreate, panel.snapshotApply, panel.snapshotDelete))
    panel.set_snapshots(["Solo nombre"])
    assert panel.list.item(0).text() == "Solo nombre"
    panel.set_snapshots([{"name": "Base neutra", "created": "2026-10-01T20:11:05"}, {"name": "B/N"}])
    assert [panel.list.item(i).text() for i in range(panel.list.count())] == ["Base neutra", "B/N"]
    assert not panel.remove_button.isEnabled()

    panel.add_button.click()
    assert panel.name_edit.isVisible() and panel.name_edit.text()  # date-and-time default
    panel.name_edit.setText("Antes de la curva")
    QTest.keyClick(panel.name_edit, Qt.Key.Key_Return)
    assert created == ["Antes de la curva"] and not panel.name_edit.isVisible()

    panel.begin_create("Descartada")
    QTest.keyClick(panel.name_edit, Qt.Key.Key_Escape)
    assert created == ["Antes de la curva"] and not panel.name_edit.isVisible()

    # Clicking elsewhere keeps the name as typed (or the default), once.
    panel.begin_create("Al salir")
    QApplication.sendEvent(panel.name_edit, QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.MouseFocusReason))
    QApplication.sendEvent(panel.name_edit, QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.MouseFocusReason))
    assert created == ["Antes de la curva", "Al salir"]

    rect = panel.list.visualItemRect(panel.list.item(1))
    QTest.mouseClick(panel.list.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert applied == [1]
    assert panel.remove_button.isEnabled()
    panel.remove_button.click()
    QTest.keyClick(panel.list, Qt.Key.Key_Delete)
    menu = panel.context_menu(panel.list.item(0))
    next(a for a in menu.actions() if a.text() == "Eliminar").trigger()
    assert deleted == [1, 1, 0]


def test_snapshot_selection_survives_the_refresh_after_applying(shown):
    """The main window rebuilds the list (names only) after every edit, the click's included."""
    from belka.ui.left_panels import SnapshotsPanel

    panel = shown(SnapshotsPanel())
    names = ["Base neutra", "B/N", "Cálida"]
    deleted = recorder(panel.snapshotDelete)

    def refresh(*_args) -> None:
        panel.set_snapshots(list(names))

    panel.snapshotApply.connect(refresh)
    refresh()
    rect = panel.list.visualItemRect(panel.list.item(1))
    QTest.mouseClick(panel.list.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert [i.text() for i in panel.list.selectedItems()] == ["B/N"]
    assert panel.remove_button.isEnabled()
    panel.remove_button.click()
    assert deleted == [1]
    # Another snapshot added, or one before it deleted: the same snapshot stays selected.
    names.append("Nueva")
    refresh()
    del names[0]
    refresh()
    assert [i.text() for i in panel.list.selectedItems()] == ["B/N"]
    QTest.keyClick(panel.list, Qt.Key.Key_Delete)
    assert deleted == [1, 0]
    del names[0]
    refresh()
    assert panel.list.selectedItems() == [] and not panel.remove_button.isEnabled()


def test_snapshot_name_field_keeps_esc_and_enter_from_the_window(shown):
    """The main window binds Esc (cancel tool); in the name field it must cancel the snapshot."""
    from belka.ui.left_panels import SnapshotsPanel

    window = QMainWindow()
    fired = []
    for key in ("Escape", "Return"):
        action = QAction(key, window)
        action.setShortcut(QKeySequence(key))
        action.triggered.connect(lambda _checked=False, k=key: fired.append(k))
        window.addAction(action)
    panel = SnapshotsPanel()
    window.setCentralWidget(panel)
    shown(window)
    window.activateWindow()
    created = recorder(panel.snapshotCreate)
    panel.begin_create("Descartada")
    QTest.keyClick(panel.name_edit, Qt.Key.Key_Escape)
    assert fired == [] and not panel.name_edit.isVisible()
    panel.list.setFocus()  # leaving afterwards must not create the cancelled one
    QApplication.processEvents()
    assert created == []
    panel.begin_create("Guardada")
    QTest.keyClick(panel.name_edit, Qt.Key.Key_Return)
    assert fired == [] and created == ["Guardada"]
    # Outside the field the window keeps its shortcut.
    panel.list.setFocus()
    QTest.keyClick(panel.list, Qt.Key.Key_Escape)
    assert fired == ["Escape"]


def test_history_panel_newest_first_and_jumps(clock, shown):
    from belka.ui.left_panels import HistoryPanel

    h = History(DevelopSettings(), "Importar", clock=clock)
    for label, ev in (("Exposición +0,50", 0.5), ("Contraste +10", 0.5), ("Recortar", 0.5)):
        clock.advance(5)
        h.push(label, DevelopSettings(exposure=ev, contrast=1.0 + len(h) / 10))
    h.jump(2)
    panel = shown(HistoryPanel())
    jumps, clears = recorder(panel.historyJump), recorder(panel.historyClear)
    panel.set_history(h)
    labels = [panel.list.item(i).text() for i in range(panel.list.count())]
    assert labels == ["Recortar", "Contraste +10", "Exposición +0,50", "Importar"]
    assert panel.list.currentItem().text() == "Contraste +10"
    assert panel.clear_button.isEnabled()
    rect = panel.list.visualItemRect(panel.list.item(3))
    QTest.mouseClick(panel.list.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert jumps == [0]
    panel.clear_button.click()
    assert clears == [()]
    h.clear()
    panel.set_history(h)
    assert panel.list.count() == 1 and not panel.clear_button.isEnabled()
    panel.set_history(None)
    assert panel.list.count() == 0


def test_left_column_renders(library, shown):
    from belka.ui.left_panels import LeftColumn

    column = shown(LeftColumn(library), 260, 900)
    column.history.set_entries(["Importar", "Exposición +0,35"], 1)
    pix = column.grab()
    assert pix.width() == 260 and pix.height() == 900
    # History soaks up the spare height; the blocks above keep their size.
    assert column.history.height() > column.snapshots.height()
