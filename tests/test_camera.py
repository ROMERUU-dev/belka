"""Camera backends, the per-brand data in cameras.json and the capture panel.

Nothing here touches USB: libgphoto2 is driven through fake config widgets,
the worker through tests/fakes.py, and support flags come from libgphoto2's
static model list.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from belka import paths
from belka.camera import models
from belka.camera.base import CameraError, CameraInfo, CameraSetting, CaptureHint
from belka.core.rawio import load_linear
from tests.fakes import FAKE, FakeBackend, install


class FakeWidget:
    def __init__(self, name, wtype, value, choices=(), rng=None):
        self.name, self.wtype, self.value, self.choices, self.rng = name, wtype, value, list(choices), rng

    def get_type(self): return self.wtype
    def get_value(self): return self.value
    def set_value(self, v): self.value = v
    def get_name(self): return self.name
    def get_label(self): return self.name
    def get_readonly(self): return 0
    def count_choices(self): return len(self.choices)
    def get_choice(self, i): return self.choices[i]
    def get_range(self): return self.rng


def _backend(model, widgets):
    """A GPhotoBackend whose camera is a dict of fake widgets; returns it and the writes it makes."""
    import gphoto2 as gp

    from belka.camera import gphoto

    class Config:
        def get_child_by_name(self, name):
            if name not in widgets:
                raise gp.GPhoto2Error(gp.GP_ERROR_BAD_PARAMETERS)
            return widgets[name]

    applied = []

    class Cam:
        def get_config(self): return Config()
        def set_single_config(self, name, widget): applied.append((name, widget.value))

    backend = gphoto.GPhotoBackend(CameraInfo(model, "usb:001,005"))
    backend._camera = Cam()
    backend._resolve_names()
    return backend, applied


def _radio(name, value, choices):
    import gphoto2 as gp

    return FakeWidget(name, gp.GP_WIDGET_RADIO, value, choices)


def _range(name, lo, hi):
    import gphoto2 as gp

    return FakeWidget(name, gp.GP_WIDGET_RANGE, 0, rng=(lo, hi, 1))


def _toggle(name):
    import gphoto2 as gp

    return FakeWidget(name, gp.GP_WIDGET_TOGGLE, 0)


# ---------------------------------------------------------------- libgphoto2 backend

def test_gphoto_backend_maps_brand_specific_widgets():
    backend, applied = _backend("Nikon DSC D850", {
        "iso": _radio("iso", "100", ["100", "200"]),
        "shutterspeed": _radio("shutterspeed", "1/30", ["1/30", "1/15"]),
        "f-number": _radio("f-number", "f/8", ["f/8"]),
        "imagequality": _radio("imagequality", "NEF (Raw)", ["NEF (Raw)", "JPEG Fine"]),
        "manualfocusdrive": _range("manualfocusdrive", -32767, 32767),
    })
    settings = {s.key: s for s in backend.settings()}
    assert settings["shutter"].name == "shutterspeed"
    assert settings["aperture"].name == "f-number"
    assert settings["quality"].choices == ["NEF (Raw)", "JPEG Fine"]
    backend.set_setting("iso", "200")
    backend.focus_step(40)
    assert applied == [("iso", "200"), ("manualfocusdrive", 40.0)]


def test_nikon_dslr_resolves_the_same_controls_as_before():
    """Checked against a real Nikon: its control names must not move."""
    names = ["iso", "autoiso", "isoauto", "shutterspeed", "shutterspeed2", "f-number", "exposurecompensation",
             "imagequality", "capturetarget", "whitebalance", "liveviewsize", "focusmode", "expprogram",
             "batterylevel", "manualfocusdrive", "autofocusdrive", "viewfinder"]
    widgets = {n: _radio(n, "x", ["x"]) for n in names}
    widgets["manualfocusdrive"] = _range("manualfocusdrive", -32767, 32767)
    widgets["autofocusdrive"] = _toggle("autofocusdrive")
    backend, applied = _backend("Nikon DSC D850", widgets)
    assert backend._names == {
        "iso": "iso", "iso_auto": "autoiso", "shutter": "shutterspeed2", "aperture": "f-number",
        "exposure_comp": "exposurecompensation", "quality": "imagequality", "target": "capturetarget",
        "whitebalance": "whitebalance", "liveview_size": "liveviewsize", "focusmode": "focusmode",
        "program": "expprogram", "battery": "batterylevel",
    }
    assert backend.capabilities >= {"preview", "settings", "focus"}
    backend.focus_step(-400)
    backend.focus_step(40000)
    backend.autofocus()
    backend._liveview = True
    backend.stop_preview()
    assert applied == [("manualfocusdrive", -400.0), ("manualfocusdrive", 32767.0), ("autofocusdrive", 1),
                       ("viewfinder", 0)]


def test_canon_uses_its_own_names_and_near_far_focus():
    near_far = ["Near 1", "Near 2", "Near 3", "None", "Far 1", "Far 2", "Far 3"]
    backend, applied = _backend("Canon EOS R6", {
        "iso": _radio("iso", "100", ["100"]),
        "shutterspeed": _radio("shutterspeed", "1/30", ["1/30"]),
        "aperture": _radio("aperture", "8", ["8"]),
        "imageformat": _radio("imageformat", "RAW", ["RAW"]),
        "imagequality": _radio("imagequality", "x", ["x"]),
        "autoexposuremode": _radio("autoexposuremode", "Manual", ["Manual"]),
        "manualfocusdrive": _radio("manualfocusdrive", "None", near_far),
        "autofocusdrive": _toggle("autofocusdrive"),
        "viewfinder": _toggle("viewfinder"),
    })
    assert backend._names["quality"] == "imageformat"  # Canon's RAW/JPEG control, before the generic one
    assert backend._names["aperture"] == "aperture" and backend._names["program"] == "autoexposuremode"
    backend.focus_step(40)
    backend.focus_step(-400)
    assert applied == [("manualfocusdrive", "Far 1"), ("manualfocusdrive", "Near 3")]


def test_canon_focus_falls_back_to_the_fixed_order_when_translated():
    labels = ["Cerca 1", "Cerca 2", "Cerca 3", "Ninguno", "Lejos 1", "Lejos 2", "Lejos 3"]
    backend, applied = _backend("Canon EOS 5D Mark IV", {"manualfocusdrive": _radio("manualfocusdrive", "", labels)})
    backend.focus_step(100)
    backend.focus_step(-40)
    assert applied == [("manualfocusdrive", "Lejos 2"), ("manualfocusdrive", "Cerca 1")]


def test_sony_focus_is_a_step_size_and_af_is_released(monkeypatch):
    from belka.camera import gphoto

    monkeypatch.setattr(gphoto.time, "sleep", lambda _s: None)
    backend, applied = _backend("Sony Alpha-A7 III (PC Control)", {
        "manualfocus": _range("manualfocus", -7, 7),
        "autofocus": _toggle("autofocus"),
    })
    backend.focus_step(400)
    backend.focus_step(-40)
    backend.autofocus()
    backend._liveview = True
    backend.stop_preview()  # Sony has no live-view switch: nothing to write
    assert applied == [("manualfocus", 3.0), ("manualfocus", -1.0), ("autofocus", 1), ("autofocus", 0)]


def test_unknown_camera_guesses_the_focus_units_and_reports_missing_controls():
    backend, applied = _backend("USB PTP Class Camera", {"manualfocusdrive": _range("manualfocusdrive", -3, 3)})
    backend.focus_step(400)
    assert applied == [("manualfocusdrive", 3.0)]
    with pytest.raises(CameraError):
        backend.autofocus()
    pentax, _applied = _backend("Pentax K1", {})
    with pytest.raises(CameraError, match="foco"):
        pentax.focus_step(40)


def test_model_without_preview_drops_the_capability():
    leica, _applied = _backend("Leica M9", {})
    assert "preview" not in leica.capabilities


def test_gphoto_claim_error_is_explained():
    import gphoto2 as gp

    from belka.camera import gphoto
    from belka.camera.base import CameraBusyError

    err = gphoto._wrap(SimpleNamespace(code=gp.GP_ERROR_IO_USB_CLAIM))
    assert isinstance(err, CameraBusyError)


def test_simulated_camera_is_gone():
    from belka.camera import worker

    assert not (Path(worker.__file__).parent / "simulated.py").exists()
    assert set(worker.BACKENDS) == {"gphoto2"}


# ---------------------------------------------------------------- cameras.json

def test_cameras_json_is_well_formed():
    data = json.loads((paths.data_dir() / "cameras.json").read_text(encoding="utf-8"))
    keys = set(data["generic"]["widgets"])
    assert keys >= {"iso", "shutter", "aperture", "quality", "target"}
    for entry in [data["generic"], *data["brands"]]:
        assert {"es", "en"} <= set(entry["usb_mode"])
        assert set(entry.get("widgets", {})) <= keys, entry.get("id")
        if "focus" in entry:
            assert entry["focus"]["type"] in ("range", "choice")
            assert entry["focus"].get("units", "") in ("", "steps", "size")
        for note in entry.get("notes", []):
            assert {"es", "en"} <= set(note)
        for series in entry.get("series", []):
            re.compile(series["match"])


@pytest.mark.parametrize("model, brand, name", [
    ("Nikon DSC D850", "nikon", "Nikon réflex (D)"),
    ("Nikon Z6_2", "nikon", "Nikon Z (sin espejo)"),
    ("Canon EOS R5", "canon", "Canon EOS R (sin espejo)"),
    ("Canon EOS Rebel T7i", "canon", "Canon EOS réflex"),
    ("Sony ILCE-7RM5 (PC Control)", "sony", "Sony Alpha"),
    ("Fuji Fujifilm X-T4", "fujifilm", "Fujifilm X / GFX"),
    ("Fuji GFX 50 S", "fujifilm", "Fujifilm X / GFX"),
    ("Panasonic DC-GH5", "panasonic", "Panasonic Lumix"),
    ("Olympus OM-1", "olympus", "OM System / Olympus"),
    ("Pentax K3II", "pentax", "Pentax / Ricoh"),
    ("Sigma fp L", "sigma", "Sigma"),
    ("Leica SL3", "leica", "Leica"),
    ("USB PTP Class Camera", "generic", "Otra cámara PTP"),
])
def test_profile_matches_brand_and_series(model, brand, name):
    profile = models.profile_for(model)
    assert (profile.brand, profile.name) == (brand, name)
    assert profile.usb_mode


def test_series_and_model_refine_the_brand():
    z = models.profile_for("Nikon Z8")
    assert "Conectar a PC" in z.usb_mode
    assert not any("espejo" in n for n in z.notes)
    d70 = models.profile_for("Nikon DSC D70 (PTP mode)")
    assert "PTP" in d70.usb_mode and any("espejo" in n for n in d70.notes)
    sony = models.profile_for("Sony Alpha-A7 IV (PC Control)")
    assert sony.focus == models.FocusDrive(("manualfocus",), "range", "size")
    assert models.profile_for("Canon EOS RP").focus.kind == "choice"
    assert "PC Remote" in models.profile_for("Sony ZV-E10 (Control)").usb_mode


def test_an_exact_model_adds_its_own_notes(monkeypatch):
    """A brand's "models" entry refines its profile for that model only."""
    import copy

    catalog = copy.deepcopy(models._catalog())
    nikon = next(b for b in catalog["brands"] if b["id"] == "nikon")
    nikon["models"] = {"Nikon DSC D850": {"notes": [{"es": "Nota de este modelo.", "en": "This model's note."}]}}
    monkeypatch.setattr(models, "_catalog", lambda: catalog)
    assert "Nota de este modelo." in models.profile_for("Nikon DSC D850").notes
    assert "Nota de este modelo." not in models.profile_for("Nikon DSC D750").notes


def test_profile_text_follows_the_language():
    from belka.i18n import set_language

    try:
        set_language("en")
        profile = models.profile_for("Fuji Fujifilm X-H2")
        assert "USB TETHER SHOOTING AUTO" in profile.usb_mode and "Menú" not in profile.usb_mode
        assert models.profile_for("Nikon Z9").name == "Nikon Z (mirrorless)"
    finally:
        set_language("es")


def test_libgphoto2_support_flags_come_from_the_static_list():
    d850 = models.support("Nikon DSC D850")
    assert d850.capture and d850.preview and d850.status == "production"
    m9 = models.support("Leica M9")
    assert m9.capture and not m9.preview
    assert models.support("Cámara inventada 3000") is None
    listed = models.capture_models()
    assert len(listed) > 300 and all(m.capture for m in listed)
    assert [m.model.lower() for m in listed] == sorted(m.model.lower() for m in listed)
    assert not any("WLAN" in m.model for m in listed)  # network-only entries are not USB cameras


def test_every_modern_brand_in_libgphoto2_has_a_profile():
    prefixes = ("Nikon Z", "Nikon DSC D", "Canon EOS", "Sony ", "Fuji ", "Panasonic ", "Olympus E-", "Olympus OM",
                "Pentax K", "Sigma ", "Leica ")
    for m in models.capture_models():
        if m.model.startswith(prefixes):
            assert models.brand_for(m.model) is not None, m.model


# ---------------------------------------------------------------- worker

def _collect(signal):
    seen = []
    signal.connect(lambda *args: seen.append(args))
    return seen


def test_worker_detects_opens_and_captures_with_a_backend(tmp_path, monkeypatch):
    from belka.camera.worker import CameraWorker

    install(monkeypatch)
    worker = CameraWorker()
    found, opened, settings, captured = (_collect(s) for s in (
        worker.cameras_found, worker.opened, worker.settings_ready, worker.captured))
    worker.detect()
    assert found == [([FAKE],)]
    worker.open_camera(FAKE)
    assert opened[0][0] == FAKE
    assert {s.key for s in settings[0][0]} >= {"shutter", "iso"}
    worker.capture(tmp_path, "f1", CaptureHint(), "tag")
    files, tag = captured[0]
    assert tag == "tag" and files[0].exists()
    worker.set_setting("shutterspeed2", "1/8")
    assert {s.key: s.value for s in settings[-1][0]}["shutter"] == "1/8"
    worker.close_camera()


def test_worker_reports_why_detection_failed(monkeypatch):
    from belka.camera import gphoto
    from belka.camera.worker import CameraWorker

    def missing():
        raise CameraError("Falta el módulo gphoto2")

    monkeypatch.setattr(gphoto, "detect", missing)
    worker = CameraWorker()
    found, failed = _collect(worker.cameras_found), _collect(worker.failed)
    worker.detect()
    assert found == [([],)] and "gphoto2" in failed[0][0]
    worker.open_camera(CameraInfo("X", "x:", backend="nonexistent"))
    assert "desconocido" in failed[-1][0]


def test_fake_camera_answers_the_shutter_speed(tmp_path):
    cam = FakeBackend()
    normal = load_linear(cam.capture(tmp_path, "a")[0]).rgb
    cam.set_setting("shutterspeed2", "1/125")
    darker = load_linear(cam.capture(tmp_path, "b")[0]).rgb
    ok = np.all(normal < 0.98, axis=-1) & (normal.mean(-1) > 0.01)
    assert darker[ok].mean() < normal[ok].mean() * 0.4
    assert cam.preview()[:2] == b"\xff\xd8"


# ---------------------------------------------------------------- capture panel

@pytest.fixture
def panel():
    from belka.light.geometry import LightSettings, load_adapters
    from belka.ui.capture_panel import CapturePanel

    p = CapturePanel(LightSettings(), load_adapters())
    yield p
    p.deleteLater()


NIKON = CameraInfo("Nikon DSC D850", "usb:001,005")
SETTINGS = [
    CameraSetting("shutter", "shutterspeed2", "Velocidad", "choice", "1/30", ["1/60", "1/30", "1/15", "1/8"]),
    CameraSetting("quality", "imagequality", "Calidad", "choice", "NEF (Raw) + JPEG Fine (Star)",
                  ["NEF (Raw) + JPEG Fine (Star)", "NEF (Raw)"]),
    CameraSetting("liveview_size", "liveviewsize", "Tamaño vista en vivo", "choice", "XGA 1024x768", ["XGA 1024x768"]),
    CameraSetting("battery", "batterylevel", "Batería", "text", "80%", readonly=True),
]


def test_panel_says_when_no_camera_is_detected(panel):
    panel.set_cameras([])
    assert panel.no_camera_label.isVisibleTo(panel) and panel.no_camera_hint.isVisibleTo(panel)
    assert "No se ha detectado ninguna cámara" in panel.no_camera_label.text()
    assert "USB" in panel.no_camera_hint.text()
    assert panel.detect_btn.isVisibleTo(panel)
    assert not panel.camera_combo.isVisibleTo(panel) and not panel.connect_btn.isVisibleTo(panel)
    requested = _collect(panel.detectRequested)
    panel.detect_btn.click()
    assert requested and panel.camera_status.isVisibleTo(panel)


def test_panel_lists_cameras_with_their_support(panel):
    twin = CameraInfo("Nikon DSC D850", "usb:001,009")
    leica = CameraInfo("Leica M9", "usb:002,003")
    panel.set_cameras([NIKON, twin, leica], keep="Leica M9")
    assert not panel.no_camera_label.isVisibleTo(panel) and panel.camera_combo.isVisibleTo(panel)
    assert [panel.camera_combo.itemText(i) for i in range(3)] == [
        "Nikon DSC D850 (usb:001,005)", "Nikon DSC D850 (usb:001,009)", "Leica M9"]
    assert "Vista en vivo: no" in panel.support_label.text()
    assert "PTP" in panel.tips_label.text()
    panel.camera_combo.setCurrentIndex(0)
    assert panel.support_label.text() == "Captura remota: sí · Vista en vivo: sí"
    assert panel.support_label.property("warn") is False
    connect = _collect(panel.connectRequested)
    panel.connect_btn.click()
    assert connect == [(NIKON,)]
    panel.set_cameras([CameraInfo("Cámara inventada 3000", "usb:9,9")])
    assert panel.support_label.property("warn") is True


def test_panel_connected_state(panel):
    panel.set_cameras([CameraInfo("Leica M9", "usb:002,003")])
    assert not panel.focus_row.isVisibleTo(panel)
    panel.set_connected(True)
    assert panel.focus_row.isVisibleTo(panel) and not panel.tips_label.isVisibleTo(panel)
    assert not panel.live_check.isEnabled()  # libgphoto2 has no live view for the M9
    panel.set_connected(False)
    assert panel.tips_label.isVisibleTo(panel) and not panel.focus_row.isVisibleTo(panel)


@pytest.mark.parametrize("connected", [False, True])
def test_panel_fits_lightrooms_left_column(panel, connected):
    panel.set_cameras([CameraInfo("Sony Alpha-A7r III (PC Control)", "usb:001,005")])
    if connected:
        panel.set_connected(True, "Sony Alpha-A7r III (PC Control)")
        panel.show_settings(SETTINGS)
        panel.set_exposure_suggestion("⚠ Subexpuesta: expón ~1.4 pasos más.", "shutter", "1/8")
    scrollbar = panel.verticalScrollBar().sizeHint().width()
    assert panel.widget().minimumSizeHint().width() + scrollbar <= 270
    assert panel.minimumWidth() <= 270


def test_exposure_suggestion_applies_the_shutter_speed(panel):
    panel.set_cameras([NIKON])
    panel.set_connected(True)
    panel.show_settings(SETTINGS)
    changed = _collect(panel.settingChanged)
    panel.set_exposure_suggestion("Expón 2 pasos más. Velocidad: 1/30 → 1/8.", "shutter", "1/8")
    assert panel.exposure_box.isVisibleTo(panel) and panel.exposure_apply.isVisibleTo(panel)
    panel.exposure_apply.click()
    assert changed == [("shutterspeed2", "1/8")]
    assert not panel.exposure_apply.isEnabled()
    panel.set_exposure_suggestion("Bien expuesta.", None, None)
    assert panel.exposure_box.isVisibleTo(panel) and not panel.exposure_apply.isVisibleTo(panel)
    panel.set_exposure_suggestion("", None, None)
    assert not panel.exposure_box.isVisibleTo(panel)
    panel.set_exposure_suggestion("Expón más.", "shutterspeed2", "1/15")  # a control name works too
    panel.exposure_apply.click()
    assert changed[-1] == ("shutterspeed2", "1/15")
    panel.set_exposure_suggestion("Expón más.", "shutter", "1/8")
    panel.set_connected(False)
    assert not panel.exposure_apply.isVisibleTo(panel)


def test_shutter_suggestion_uses_the_camera_choices(panel):
    panel.show_settings(SETTINGS)
    assert panel.setting_name("shutter") == "shutterspeed2"
    assert panel.shutter_suggestion(2.0) == ("1/30", "1/8")
    assert panel.shutter_suggestion(-1.0, from_seconds=1 / 30) == ("1/30", "1/60")


def test_compatible_cameras_dialog_filters_and_explains(panel):
    from belka.ui.capture_panel import CompatibleCamerasDialog

    dialog = CompatibleCamerasDialog(panel)
    total = len(dialog.visible_models())
    assert total == len(models.capture_models())
    dialog.search.setText("nikon z6")
    shown = dialog.visible_models()
    assert "Nikon Z6 III" in shown and all("Z6" in m for m in shown)
    assert f"{len(shown)} de {total}" in dialog.count_label.text()
    dialog.search.setText("reflex d850")  # accents do not matter: the series is "Nikon réflex (D)"
    assert dialog.visible_models() == ["Nikon DSC D850"]
    dialog.select_model("Sony Alpha-A7 IV (PC Control)")
    assert dialog.tree.currentItem().text(0) == "Sony Alpha-A7 IV (PC Control)"
    assert dialog.search.text() == ""
    assert "PC Remote" in dialog.details.text() and "Vista en vivo: sí" in dialog.details.text()
    dialog.deleteLater()


def test_compatible_cameras_button_opens_the_list_on_the_detected_model(panel):
    panel.set_cameras([NIKON])
    panel.compat_btn.click()
    dialog = panel._compat_dialog
    assert dialog.isVisible() and dialog.tree.currentItem().text(0) == "Nikon DSC D850"
    dialog.close()


def test_suggested_shutter_fixes_an_underexposed_capture(panel, tmp_path, library):
    """The loop the main window runs after each capture, on the fake camera."""
    from belka.core import pipeline as pl

    cam = FakeBackend(shutter="1/250")  # 3 stops under
    profile = library.get("kodak-portra-400")

    def advice(name):
        raw = load_linear(cam.capture(tmp_path, name)[0]).rgb
        return pl.exposure_advice(pl.analyze(raw, pl.DevelopSettings(profile_id=profile.id), profile))

    stops = advice("under")
    assert stops > 2
    panel.set_cameras([FAKE])
    panel.set_connected(True)
    panel.show_settings(cam.settings())
    shot_at, suggested = panel.shutter_suggestion(stops)
    assert (shot_at, suggested) == ("1/250", "1/30")
    panel.settingChanged.connect(cam.set_setting)
    panel.set_exposure_suggestion("Subexpuesta", "shutter", suggested)
    panel.exposure_apply.click()
    assert advice("fixed") == 0.0
