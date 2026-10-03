"""The main window wired to the develop modules: offscreen, on a synthetic roll."""

import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QMessageBox

from belka.core import synth
from belka.core.rawio import save_linear_tiff
from belka.core.session import Session


def _pump(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def _wait_render(win, timeout: float = 60.0) -> None:
    """Until the newest render request has been answered."""
    start = win._last_result
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        dev = win.develop
        if win._last_result is not start and not dev._busy and dev._pending_render is None and not dev._queue:
            _pump(0.05)
            return
        time.sleep(0.005)
    raise AssertionError("no render arrived")


@pytest.fixture
def window(tmp_path, monkeypatch, library):
    import belka.camera.gphoto as gphoto

    monkeypatch.setattr(gphoto, "detect", lambda: [])  # never the USB camera
    errors: list = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: errors.append(a[2:])))
    from belka.settings import Settings
    from belka.ui.main_window import MainWindow

    win = MainWindow(Settings())
    win.develop.failed.connect(errors.append)
    win.resize(1400, 900)
    win.show()
    _pump(0.1)
    profile = library.get("kodak-portra-400")
    raw, truth = synth.backlit_scan(profile, rng=np.random.default_rng(3))
    scan = tmp_path / "scan.tif"
    save_linear_tiff(scan, raw)
    win._load_session(Session.create(tmp_path / "rolls", "prueba", profile.id))
    win.import_files([str(scan)])
    _wait_render(win)
    yield win, errors, truth
    win.shutdown()


def test_opens_and_develops_the_detected_frame(window):
    win, errors, truth = window
    result = win._last_result
    assert result.image is not None and result.histogram is not None
    # The loupe shows the detected frame, not the whole capture.
    assert np.abs(np.asarray(result.crop) - np.asarray(truth["frame"])).max() < 0.03
    assert win.history.get(win.frame.id).entries()[0].label == "Abrir"
    assert not errors


def test_histogram_drag_records_one_step_and_undo_restores(window):
    win, errors, _truth = window
    before = win._current_settings().exposure
    for _ in range(3):
        win.histogram.adjustRequested.emit("exposure", 0.25)
    win.histogram.dragFinished.emit("exposure")
    _wait_render(win)
    assert win._current_settings().exposure == pytest.approx(before + 0.75)
    hist = win.history.get(win.frame.id)
    assert hist.entries()[-1].label.startswith("Exposición")
    win.undo()
    assert win._current_settings().exposure == pytest.approx(before)
    win.redo()
    assert win._current_settings().exposure == pytest.approx(before + 0.75)
    assert not errors


def test_crop_tool_rotate_commit_and_cancel(window):
    win, errors, _truth = window
    win.act_crop.trigger()
    _wait_render(win)
    assert win.view.tool == "crop" and win._last_result.job.ignore_crop
    start = win.view.crop_rect()
    win.view.angleDelta.emit(2.0)
    _wait_render(win)
    win.view.rotateFinished.emit()
    assert win._current_settings().angle == pytest.approx(2.0)
    # The frame stays inside the rotated picture.
    valid = win._last_result.valid
    rect = win.view.crop_rect()
    assert rect[0] >= valid[0] - 1e-3 and rect[2] <= valid[2] + 1e-3
    win.commit_crop()
    _wait_render(win)
    s = win._current_settings()
    # Only rotated: the detected frame stays automatic and follows the straightening.
    assert win.view.tool == "none" and s.crop is None and s.auto_crop
    labels = [e.label for e in win.history.get(win.frame.id).entries()]
    assert labels[-1].startswith("Ángulo")
    # A frame the user changed is stored.
    win.act_crop.trigger()
    _wait_render(win)
    win.view.set_crop_rect((0.3, 0.3, 0.7, 0.7))
    win.commit_crop()
    _wait_render(win)
    s = win._current_settings()
    assert s.crop is not None and not s.auto_crop
    assert [e.label for e in win.history.get(win.frame.id).entries()][-1] == "Recortar"
    # Cancelling a second session restores everything, angle included.
    committed = win._current_settings().copy()
    win.act_crop.trigger()
    _wait_render(win)
    win.view.angleDelta.emit(5.0)
    _wait_render(win)
    win.view.cropCancelled.emit()
    _wait_render(win)
    assert win._current_settings() == committed
    assert start is not None and not errors


def test_straighten_line_adds_to_the_angle(window):
    win, errors, _truth = window
    win.start_tool("straighten")
    _wait_render(win)
    img = win._last_result.image
    # A line rising 2 degrees to the right on screen.
    dy = np.tan(np.radians(2.0)) * 0.6 * img.width() / img.height()
    win.view.straightenLine.emit(QPointF(0.2, 0.5), QPointF(0.8, 0.5 - dy))
    _wait_render(win)
    assert abs(win._current_settings().angle) == pytest.approx(2.0, abs=0.05)
    assert win.view.tool == "none" and not errors


def test_copy_paste_keeps_geometry_and_snapshots_apply(window):
    win, errors, _truth = window
    win.develop_panel.apply_external(saturation=1.4, crop=(0.3, 0.3, 0.7, 0.7))
    win.commit_history("test")
    win.create_snapshot("uno")
    win.copy_settings()
    win.develop_panel.apply_external(saturation=1.0, crop=(0.1, 0.1, 0.9, 0.9))
    win.paste_settings()
    s = win._current_settings()
    assert s.saturation == pytest.approx(1.4) and s.crop == (0.1, 0.1, 0.9, 0.9)
    win.apply_snapshot(0)
    assert win._current_settings().crop == (0.3, 0.3, 0.7, 0.7)
    assert win.frame.snapshots[0]["name"] == "uno" and not errors


def test_clipping_overlay_and_compare(window):
    win, errors, _truth = window
    win.develop_panel.apply_external(exposure=3.0)
    _wait_render(win)
    win.act_clipping.trigger()
    assert win.view._clipping.isVisible()
    win.toggle_compare("split")
    _pump(0.1)
    end = time.monotonic() + 30
    while win.view.compare_mode != "split" and time.monotonic() < end:
        _pump(0.05)
    assert win.view.compare_mode == "split"
    win.toggle_compare("split")
    assert win.view.compare_mode == "off" and not errors


def test_modules_switch_and_empty_roll(window, tmp_path):
    win, errors, _truth = window
    for module in ("library", "capture", "develop"):
        win.set_module(module)
        _pump(0.05)
        assert win.module == module
    win._load_session(Session.create(tmp_path / "rolls", "vacío", "generic-c41"))
    _pump(0.1)
    assert win.frame is None and not win.view.has_image() and not errors


def _second_frame(win, tmp_path, library, seed=7):
    profile = library.get("kodak-portra-400")
    raw, _truth = synth.backlit_scan(profile, rng=np.random.default_rng(seed))
    scan = tmp_path / f"scan{seed}.tif"
    save_linear_tiff(scan, raw)
    win.import_files([str(scan)])
    _wait_render(win)


def test_delete_to_trash_and_a_reused_id_starts_a_clean_history(window, tmp_path, library, monkeypatch):
    from PySide6.QtCore import QFile

    win, errors, _truth = window
    _second_frame(win, tmp_path, library)
    last = win.session.frames[-1]
    win.filmstrip.select_frame(last.id)
    win.select_frame(last.id)
    win.develop_panel.apply_external(exposure=1.0)
    win.commit_history("Exposición +1,00")
    trashed = []

    def to_trash(path):  # like PySide's: a bare bool
        trashed.append(path)
        Path(path).unlink()
        return True

    monkeypatch.setattr(QFile, "moveToTrash", staticmethod(to_trash))
    monkeypatch.setattr(win, "_ask_delete", lambda mode, n: "trash")
    files = [win.session.resolve(f) for f in last.files]
    win.delete_frames(ids=[last.id])
    assert win.session.frame(last.id) is None and trashed and not any(f.exists() for f in files)
    # A reshoot gets the same id: it must not undo into the deleted photo's edits.
    _second_frame(win, tmp_path, library, seed=8)
    again = win.session.frames[-1]
    assert again.id == last.id
    assert [e.label for e in win.history.get(again.id).entries()] == ["Abrir"]
    assert not errors


def test_delete_rejected_keeps_the_files(window, tmp_path, library, monkeypatch):
    win, errors, _truth = window
    _second_frame(win, tmp_path, library)
    first = win.session.frames[0]
    first.flag = -1
    monkeypatch.setattr(win, "_ask_delete", lambda mode, n: "remove")
    win.delete_rejected()
    assert win.session.frame(first.id) is None and len(win.session.frames) == 1
    assert all(win.session.resolve(f).exists() for f in first.files)
    assert not errors


def test_space_zooms_outside_capture(window):
    win, errors, _truth = window
    win.set_module("develop")
    win.view.fit()
    assert win.view.is_fit()
    win.act_space.trigger()
    assert not win.view.is_fit()
    win.act_space.trigger()
    assert win.view.is_fit() and not errors


def test_guides_drive_guided_upright(window):
    win, errors, _truth = window
    win.request_upright("guided")
    _wait_render(win)
    assert win.view.tool == "guided"
    # Two lines that converge upwards, as verticals shot from below do.
    win.view.guidesChanged.emit([(0.3, 0.2, 0.25, 0.8), (0.7, 0.2, 0.75, 0.8)])
    _wait_render(win)
    s = win._current_settings()
    assert s.upright_mode == "guided" and len(s.upright_guides) == 2 and s.persp_vertical != 0.0
    assert win.history.get(win.frame.id).entries()[-1].label == "Upright guiado"
    # The guides follow the transformed picture on screen.
    assert len(win.view.guides()) == 2
    win.view.toolCancelled.emit("guided")
    assert win.view.tool == "none" and not errors


def test_esc_in_the_crop_tool_keeps_panel_edits(window):
    win, errors, _truth = window
    win.act_crop.trigger()
    _wait_render(win)
    win.develop_panel.apply_external(exposure=1.0)
    win.view.angleDelta.emit(4.0)
    _wait_render(win)
    win.view.cropCancelled.emit()
    _wait_render(win)
    s = win._current_settings()
    assert s.exposure == pytest.approx(1.0) and s.angle == 0.0 and not errors


def test_tab_in_a_text_field_does_not_hide_the_panels(window):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    win, errors, _truth = window
    win.set_module("develop")
    field = win.profiles_panel.search
    event = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(field, event)
    assert event.isAccepted()  # the field keeps Tab: the window's "Ocultar paneles" must not fire
    assert win.left_stack.isVisible() and not errors


def test_a_failed_trash_still_takes_the_frame_out_and_keeps_the_ui_alive(window, tmp_path, library, monkeypatch):
    """Regression: an error after trashing left the frame in the roll and the filmstrip
    with its signals blocked, so the whole window seemed frozen."""
    from PySide6.QtCore import QFile

    win, errors, _truth = window
    _second_frame(win, tmp_path, library)
    first = win.session.frames[0]
    monkeypatch.setattr(QFile, "moveToTrash", staticmethod(lambda path: False))
    monkeypatch.setattr(win, "_ask_delete", lambda mode, n: "trash")
    win.delete_frames(ids=[first.id])
    from belka.core.session import Session as S

    assert first.id not in [f.id for f in S.load(win.session.path).frames]  # saved without it
    assert errors and "papelera" in str(errors[-1])  # the file that stayed is reported
    assert not win.filmstrip.signalsBlocked() and not win.grid.signalsBlocked()
    remaining = win.session.frames[0].id
    assert win.frame is not None and win.frame.id == remaining


def test_a_frame_whose_file_vanished_reports_once(window, tmp_path, library):
    win, errors, _truth = window
    _second_frame(win, tmp_path, library)
    gone = win.session.frames[0]
    for rel in gone.files:
        win.session.resolve(rel).unlink()
    win.filmstrip.select_frame(gone.id)
    win.select_frame(gone.id)
    for _ in range(3):
        win.develop_panel.apply_external(exposure=0.1 * (_ + 1))
        _pump(0.5)
    dialogs = [e for e in errors if isinstance(e, tuple)]  # QMessageBox.warning, not the raw signal
    assert len(dialogs) == 1 and "No se encuentra el archivo" in str(dialogs[0])
