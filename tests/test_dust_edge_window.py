"""0.4 wired into the app: dust removal with the dark-field shot, and the film edge read on import."""

import time

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from belka.core import synth
from belka.core.rawio import save_linear_tiff
from belka.core.session import Session
from tests import dx_synth as ds
from tests.test_main_window import _pump, _wait_render


def _open(win, tmp_path, profile_id: str, image: np.ndarray, name: str = "scan.tif") -> None:
    scan = tmp_path / name
    save_linear_tiff(scan, image)
    win._load_session(Session.create(tmp_path / "rolls", "prueba", profile_id))
    win.import_files([str(scan)])
    _wait_render(win)


@pytest.fixture
def win(monkeypatch):
    import belka.camera.gphoto as gphoto

    monkeypatch.setattr(gphoto, "detect", lambda: [])  # never the USB camera
    errors: list = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: errors.append(a[2:])))
    from belka.settings import Settings
    from belka.ui.main_window import MainWindow

    w = MainWindow(Settings())
    w.develop.failed.connect(errors.append)
    w.errors = errors
    w.resize(1400, 900)
    w.show()
    _pump(0.1)
    yield w
    w.shutdown()
    # Gone for good: a window left on screen takes later tests' mouse moves.
    w.close()
    w.deleteLater()
    _pump(0.05)


def _with_darkfield(win, tmp_path, darkfield: np.ndarray) -> None:
    """Give the open frame a dark-field shot and show it again, as after a capture."""
    path = win.session.path / "raw" / "scan_df.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_linear_tiff(path, darkfield)
    win.frame.darkfield = str(path.relative_to(win.session.path))
    win.session.save()
    win._show_frame_extras(win.frame)


def test_dust_is_removed_with_the_dark_field_and_shown_on_request(win, tmp_path, library):
    profile = library.get("kodak-portra-400")
    bright, darkfield, truth = synth.dusty_scan(profile, size=(1500, 1000), seed=1)
    _open(win, tmp_path, profile.id, bright)
    _with_darkfield(win, tmp_path, darkfield)
    dp = win.develop_panel
    assert dp.dust_source.text() == "Con toma de campo oscuro"
    before = win._last_result.rgb8.astype(np.float32)
    dp.show_dust.click()
    dp.rows["dust_strength"].set_display(50)
    _wait_render(win)
    result = win._last_result
    assert result.darkfield_ok is True
    assert result.dust is not None and result.dust.shape == result.rgb8.shape[:2] and result.dust.any()
    assert np.abs(result.rgb8.astype(np.float32) - before).max() > 30  # specks repaired
    assert dp.dust_source.text() == "Con toma de campo oscuro"
    # Off again: no mask comes back.
    dp.show_dust.click()
    _wait_render(win)
    assert win._last_result.dust is None and not win.errors


def test_a_dark_field_from_another_frame_is_reported(win, tmp_path, library):
    profile = library.get("kodak-portra-400")
    bright, _darkfield, _truth = synth.dusty_scan(profile, size=(1500, 1000), seed=1)
    _other_bright, other_darkfield, _ = synth.dusty_scan(profile, size=(1500, 1000), seed=5,
                                                         scene=synth.landscape_scene(600, 400, seed=9))
    _open(win, tmp_path, profile.id, bright)
    _with_darkfield(win, tmp_path, other_darkfield)
    win.develop_panel.rows["dust_strength"].set_display(50)
    _wait_render(win)
    assert win._last_result.darkfield_ok is False
    assert "no coincide" in win.develop_panel.dust_source.text()
    assert not win.errors


def test_the_edge_is_read_once_on_import_and_names_the_frame(win, tmp_path, library):
    img, geo = ds.render(ds.Strip())
    _open(win, tmp_path, "generic-c41", img.astype(np.float32), "strip.tif")
    frame = win.frame
    end = time.monotonic() + 30
    while not frame.edge and time.monotonic() < end:
        _pump(0.02)
    assert frame.edge.get("dx") == "95-7"
    assert frame.edge.get("frame") and frame.label.startswith(f"#{frame.id} · ")
    assert win.develop_panel._edge_profile() is not None  # the edge names another film: "Usar" offers it
    assert not win.errors


@pytest.mark.parametrize("rotation", [90, 180])
def test_the_edge_reads_the_same_number_on_a_rotated_frame(win, tmp_path, rotation):
    # One whole frame in view, off the image centre: read on the unrotated
    # capture with the rotated frame, 180 degrees gave the next number ("22").
    img, geo = ds.render(ds.Strip(shift_mm=8.0, view_mm=(70.0, 50.0)))
    _open(win, tmp_path, "generic-c41", img.astype(np.float32), "strip.tif")
    frame = win.frame
    end = time.monotonic() + 30
    while not frame.edge and time.monotonic() < end:
        _pump(0.02)
    upright = dict(frame.edge)
    win.develop_panel.apply_external(rotation=rotation)
    win._save_frame_settings()
    frame.edge = {}
    win._show_frame_extras(frame)
    end = time.monotonic() + 30
    while not frame.edge and time.monotonic() < end:
        _pump(0.02)
    assert upright.get("frame") == geo["label_at_frame"] == "21A"
    assert frame.edge.get("frame") == "21A" and frame.edge.get("dx") == upright.get("dx")
    assert not win.errors


def test_deleting_a_frame_trashes_its_dark_field_too(tmp_path, library, monkeypatch):
    from PySide6.QtCore import QFile

    session = Session.create(tmp_path / "rolls", "prueba", "generic-c41")
    raw = session.path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    shot, dust_shot = raw / "prueba_001.tif", raw / "prueba_001_df.tif"
    for p in (shot, dust_shot):
        p.write_bytes(b"x")
    frame = session.add_frame([shot])
    frame.darkfield = str(dust_shot.relative_to(session.path))
    trashed = []

    def to_trash(path):
        trashed.append(path)
        from pathlib import Path

        Path(path).unlink()
        return True

    monkeypatch.setattr(QFile, "moveToTrash", staticmethod(to_trash))
    assert session.remove_frame(frame, delete_files=True) == []
    assert sorted(trashed) == sorted([str(shot), str(dust_shot)])


def test_export_removes_the_same_dust(tmp_path, library):
    from belka.core.export import develop_full

    profile = library.get("kodak-portra-400")
    bright, darkfield, truth = synth.dusty_scan(profile, size=(1500, 1000), seed=1)
    session = Session.create(tmp_path / "rolls", "prueba", profile.id)
    raw = session.path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    save_linear_tiff(raw / "scan.tif", bright)
    save_linear_tiff(raw / "scan_df.tif", darkfield)
    frame = session.add_frame([raw / "scan.tif"])
    frame.darkfield = "raw/scan_df.tif"
    clean_settings = session.settings_for(frame).copy(dust_strength=0.5, auto_crop=False)
    frame.settings = clean_settings.to_json()
    cleaned, _ = develop_full(session, frame, library)
    frame.settings = clean_settings.copy(dust_strength=0.0).to_json()
    dusty, _ = develop_full(session, frame, library)
    changed = np.abs(cleaned - dusty).max(axis=-1) > 0.02
    # The repairs are where the dirt is, and nowhere much else.
    assert changed[truth].mean() > 0.3 and changed[~truth].mean() < 0.01
