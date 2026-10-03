"""Left-hand panel: camera, live view and light source.

Laid out for Lightroom's narrow left column (270 px at 1366×768): one control
per row where a pair would not fit, combo texts elided instead of clipped.
"""

from __future__ import annotations

import html
import math
import unicodedata

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from belka.camera import models
from belka.camera.base import CameraInfo, CameraSetting
from belka.i18n import _
from belka.light.geometry import Adapter, LightSettings
from belka.ui.widgets import ColorSwatch, LabeledSlider, WheelGuard

LIGHT_MODES = (
    ("white", "Blanca"),
    ("tint", "Tinte calibrado (neutraliza la máscara)"),
    ("rgb", "RGB secuencial (3 tomas por fotograma)"),
)
SETTING_ORDER = ("program", "shutter", "aperture", "iso", "iso_auto", "exposure_comp", "quality", "target",
                 "whitebalance", "liveview_size", "focusmode", "battery")
DRIVER_STATUS_LABELS = {
    "production": "Estable",
    "testing": "En pruebas",
    "experimental": "Experimental",
    "deprecated": "Obsoleto",
}

DARKFIELD_STOPS = 3  # scattered light is faint: ~8× the frame's exposure

DIM = "#8a8a8a"
WARN = "#e0a050"
_QSS = f"""
QLabel#Hint {{ color: {DIM}; font-size: 11px; }}
QLabel#NoCamera {{ color: #d6d6d6; font-weight: 600; }}
QLabel#Support {{ color: {DIM}; font-size: 11px; }}
QLabel#Support[warn="true"] {{ color: {WARN}; }}
QPushButton#Link {{ background: transparent; border: none; padding: 0; color: #a8a8a8; text-align: left;
                    font-size: 11px; }}
QPushButton#Link:hover {{ color: #ececec; text-decoration: underline; }}
QFrame#ExposureNote {{ background: #2b2b2d; border: none; border-left: 2px solid #d67828; border-radius: 2px; }}
QFrame#ExposureNote QLabel {{ color: #d0d0d0; font-size: 11px; }}
"""


def shutter_seconds(text: str) -> float | None:
    """Exposure time from a camera's label: "1/30", "10/25", "2", "0.0333s", "1/2,5"."""
    t = text.strip().lower().rstrip("s").replace(",", ".").replace('"', "")
    try:
        if "/" in t:
            num, den = t.split("/", 1)
            value = float(num) / float(den)
        else:
            value = float(t)
    except ValueError:
        return None
    return value if value > 0 else None


def closest_shutter(choices: list[str], seconds: float) -> str | None:
    """The camera's speed nearest to ``seconds``, in stops; None if no label reads as a time."""
    options = [(c, v) for c in choices if (v := shutter_seconds(c))]
    if not options:
        return None
    return min(options, key=lambda cv: abs(math.log2(cv[1] / seconds)))[0]


def support_text(model: str) -> tuple[str, bool]:
    """One line on what libgphoto2 can do with ``model``, and whether it is a warning."""
    support = models.support(model)
    if support is None:
        return _("libgphoto2 no conoce este modelo: puede funcionar como cámara PTP genérica."), True
    if not support.capture:
        return _("⚠ libgphoto2 solo puede descargar fotos de este modelo, no dispararlo."), True
    text = _("Captura remota: sí · Vista en vivo: {live}").format(live=_("sí") if support.preview else _("no"))
    if support.status != "production":
        text += " · " + _(DRIVER_STATUS_LABELS[support.status]).lower()
    return text, False


class _ElidedCombo(QComboBox):
    """Ends a long entry with "…" instead of clipping it in a narrow column."""

    def paintEvent(self, _event) -> None:
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        field = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt,
                                            QStyle.SubControl.SC_ComboBoxEditField, self)
        opt.currentText = self.fontMetrics().elidedText(opt.currentText, Qt.TextElideMode.ElideRight, field.width() - 4)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)


def tips_html(profile: models.CameraProfile) -> str:
    """How to set the camera up (USB mode) and the brand's quirks, as rich text."""
    parts = []
    if profile.usb_mode:
        label = html.escape(_("Conexión:"))
        parts.append(f"<span style='color:#c8c8c8'>{label}</span> {html.escape(profile.usb_mode)}")
    parts += [f"• {html.escape(note)}" for note in profile.notes]
    return "<br>".join(parts)


def _hint(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("Hint")
    label.setWordWrap(True)
    return label


class CapturePanel(QScrollArea):
    detectRequested = Signal()
    connectRequested = Signal(object)
    disconnectRequested = Signal()
    settingChanged = Signal(str, object)
    focusRequested = Signal(int)
    autofocusRequested = Signal()
    liveviewToggled = Signal(bool)
    captureRequested = Signal()
    lightToggled = Signal(bool)
    lightChanged = Signal()
    calibrateTintRequested = Signal()
    flatRequested = Signal()
    clearFlatsRequested = Signal()
    scaleRequested = Signal()
    captureOptionsChanged = Signal(dict)  # capture_options(), to store in the app settings

    def __init__(self, light: LightSettings, adapters: list[Adapter], parent: QWidget | None = None):
        super().__init__(parent)
        self.light = light
        self.adapters = adapters
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setStyleSheet(_QSS)
        self.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(8, 8, 8, 8)
        self._cameras: list[CameraInfo] = []
        self._connected = False
        self._setting_widgets: dict[str, QWidget] = {}
        self._setting_keys: dict[str, CameraSetting] = {}
        self._suggestion: tuple[str, str] | None = None
        self._compat_dialog: CompatibleCamerasDialog | None = None
        self._busy = False

        # ---------------------------------------------------- camera
        cam_box = QGroupBox(_("Cámara"))
        cam = QVBoxLayout(cam_box)
        row = QHBoxLayout()
        self.camera_combo = _ElidedCombo()
        self.camera_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.no_camera_label = QLabel(_("No se ha detectado ninguna cámara"))
        self.no_camera_label.setObjectName("NoCamera")
        self.no_camera_label.setWordWrap(True)
        self.detect_btn = QToolButton()
        self.detect_btn.setText("⟳")
        self.detect_btn.setToolTip(_("Buscar cámaras USB"))
        row.addWidget(self.camera_combo, 1)
        row.addWidget(self.no_camera_label, 1)
        row.addWidget(self.detect_btn, 0, Qt.AlignmentFlag.AlignTop)
        cam.addLayout(row)
        self.no_camera_hint = _hint(_("Comprueba que esté conectada por USB, encendida y en modo PTP o control "
                                      "remoto. Si otro programa la está usando, ciérralo y pulsa ⟳."))
        cam.addWidget(self.no_camera_hint)
        self.connect_btn = QPushButton(_("Conectar"))
        cam.addWidget(self.connect_btn)
        self.camera_status = _hint(_("Sin conectar"))
        cam.addWidget(self.camera_status)
        self.support_label = QLabel()
        self.support_label.setObjectName("Support")
        self.support_label.setWordWrap(True)
        cam.addWidget(self.support_label)
        self.tips_label = _hint()
        self.tips_label.setTextFormat(Qt.TextFormat.RichText)
        cam.addWidget(self.tips_label)
        self.compat_btn = QPushButton(_("Cámaras compatibles…"))
        self.compat_btn.setObjectName("Link")
        self.compat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.compat_btn.setToolTip(_("Modelos que libgphoto2 puede disparar por USB, con los ajustes de cada marca"))
        cam.addWidget(self.compat_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.settings_form = QFormLayout()
        self.settings_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        cam.addLayout(self.settings_form)
        self.focus_row = QWidget()
        focus_row = QHBoxLayout(self.focus_row)
        focus_row.setContentsMargins(0, 0, 0, 0)
        focus_row.addWidget(QLabel(_("Foco")))
        self.focus_buttons = []
        for text, steps, tip in (("«", -400, _("Más cerca (grueso)")), ("‹", -40, _("Más cerca (fino)")),
                                 ("›", 40, _("Más lejos (fino)")), ("»", 400, _("Más lejos (grueso)"))):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, s=steps: self.focusRequested.emit(s))
            focus_row.addWidget(b)
            self.focus_buttons.append(b)
        self.af_btn = QToolButton()
        self.af_btn.setText("AF")
        self.af_btn.setToolTip(_("Autoenfoque"))
        focus_row.addWidget(self.af_btn)
        focus_row.addStretch(1)
        cam.addWidget(self.focus_row)
        root.addWidget(cam_box)

        # ---------------------------------------------------- live view
        live_box = QGroupBox(_("Vista en vivo"))
        live = QVBoxLayout(live_box)
        self.live_label = QLabel(_("Conecta la cámara y activa la vista en vivo para encuadrar y enfocar."))
        self.live_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_label.setWordWrap(True)
        self.live_label.setMinimumHeight(170)
        self.live_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.live_label.setStyleSheet("background: #1c1c1e; color: #888; border-radius: 4px; padding: 8px;")
        live.addWidget(self.live_label)
        row = QHBoxLayout()
        self.live_check = QCheckBox(_("Activa"))
        self.invert_check = QCheckBox(_("Invertida"))
        self.invert_check.setChecked(True)
        self.zoom_combo = QComboBox()
        for z in (1, 2, 4, 8):
            self.zoom_combo.addItem(f"{z}×", z)
        self.zoom_combo.setToolTip(_("Ampliación del centro, para enfocar el grano"))
        row.addWidget(self.live_check)
        row.addWidget(self.invert_check)
        row.addStretch(1)
        row.addWidget(self.zoom_combo)
        live.addLayout(row)
        root.addWidget(live_box)

        # ---------------------------------------------------- light
        light_box = QGroupBox(_("Fuente de luz (pantalla)"))
        lf = QFormLayout(light_box)
        lf.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.screen_combo = _ElidedCombo()
        lf.addRow(_("Pantalla"), self.screen_combo)
        self.adapter_combo = _ElidedCombo()
        for a in adapters:
            self.adapter_combo.addItem(_(a.name), a.id)
        lf.addRow(_("Formato"), self.adapter_combo)
        self.mode_combo = _ElidedCombo()
        for key, label in LIGHT_MODES:
            self.mode_combo.addItem(_(label), key)
        lf.addRow(_("Luz"), self.mode_combo)
        self.brightness = LabeledSlider(_("Brillo"), 0.05, 1.0, 1.0)
        self.brightness.label.setMinimumWidth(0)
        self.brightness.label.hide()
        self.brightness.box.setFixedWidth(62)
        lf.addRow(_("Brillo"), self.brightness)
        tint_row = QHBoxLayout()
        self.tint_swatch = ColorSwatch()
        self.tint_label = QLabel()
        tint_row.addWidget(self.tint_swatch)
        tint_row.addWidget(self.tint_label, 1)
        lf.addRow(_("Tinte"), tint_row)
        self.calibrate_btn = QPushButton(_("Calibrar tinte"))
        self.calibrate_btn.setToolTip(_("Con el negativo puesto: toma una foto y ajusta el color de la luz para que la máscara naranja quede neutra."))
        lf.addRow(self.calibrate_btn)
        self.fullscreen_check = QCheckBox(_("Pantalla completa"))
        self.fullscreen_check.setToolTip(_("Panel de luz a pantalla completa, con la zona iluminada medida en mm. "
                                           "Sin marcar: una ventana iluminada entera."))
        lf.addRow(self.fullscreen_check)
        row = QHBoxLayout()
        self.light_btn = QPushButton(_("Encender luz"))
        self.light_btn.setCheckable(True)
        self.light_btn.setToolTip(_("Muestra el panel de luz. Dentro: Espacio captura, Esc vuelve."))
        self.scale_btn = QPushButton(_("Escala…"))
        self.scale_btn.setToolTip(_("Calibra los milímetros de la pantalla con una regla"))
        row.addWidget(self.light_btn, 1)
        row.addWidget(self.scale_btn)
        lf.addRow(row)
        self.flat_btn = QPushButton(_("Capturar flat…"))
        self.flat_btn.setToolTip(_("Foto de la luz SIN película: corrige viñeteo y luz desigual"))
        lf.addRow(self.flat_btn)
        self.flat_clear_btn = QPushButton(_("Borrar flats"))
        lf.addRow(self.flat_clear_btn)
        self.flat_label = _hint()
        lf.addRow(self.flat_label)
        self.darkfield_check = QCheckBox(_("Toma antipolvo (campo oscuro)"))
        self.darkfield_check.setToolTip(_("Tras cada fotograma, otra toma con la película a oscuras y un anillo de luz "
                                          "alrededor: solo brillan el polvo y los rayones, y el revelado los borra. "
                                          "Necesita el panel de luz."))
        lf.addRow(self.darkfield_check)
        self.darkfield_spin = QSpinBox()
        self.darkfield_spin.setRange(1, 6)
        self.darkfield_spin.setValue(DARKFIELD_STOPS)
        self.darkfield_spin.setPrefix("+")
        self.darkfield_spin.setSuffix(" " + _("pasos"))
        self.darkfield_spin.setToolTip(_("Cuánto más lenta que la del fotograma es la velocidad de la toma antipolvo"))
        self.darkfield_spin.setEnabled(False)
        lf.addRow(_("Exposición"), self.darkfield_spin)
        root.addWidget(light_box)

        # ---------------------------------------------------- capture
        self.exposure_box = QFrame()
        self.exposure_box.setObjectName("ExposureNote")
        note = QVBoxLayout(self.exposure_box)
        note.setContentsMargins(8, 6, 6, 6)
        self.exposure_note = QLabel()
        self.exposure_note.setWordWrap(True)
        note.addWidget(self.exposure_note)
        self.exposure_apply = QPushButton(_("Aplicar"))
        note.addWidget(self.exposure_apply, 0, Qt.AlignmentFlag.AlignRight)
        self.exposure_box.hide()
        root.addWidget(self.exposure_box)
        self.capture_btn = QPushButton(_("Capturar fotograma  (Espacio)"))
        self.capture_btn.setMinimumHeight(40)
        self.capture_btn.setStyleSheet("font-weight: 600; font-size: 13px;")
        root.addWidget(self.capture_btn)
        self.capture_status = QLabel()
        self.capture_status.setWordWrap(True)
        root.addWidget(self.capture_status)
        root.addStretch(1)

        self.refresh_screens()
        self.load_light(light)
        self._wire()
        # Long entries must not force the column wider than Lightroom's 270 px.
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(4)
        self.zoom_combo.setMinimumContentsLength(2)
        self.wheel_guard = WheelGuard(self)
        self.wheel_guard.guard_children(self.widget())
        self.set_cameras([])
        self.set_connected(False)

    # ------------------------------------------------------------ wiring
    def _wire(self) -> None:
        self.detect_btn.clicked.connect(self._on_detect)
        self.connect_btn.clicked.connect(self._on_connect)
        self.camera_combo.currentIndexChanged.connect(lambda _i: self._show_camera_info())
        self.compat_btn.clicked.connect(self.show_compatible_cameras)
        self.af_btn.clicked.connect(self.autofocusRequested.emit)
        self.live_check.toggled.connect(self.liveviewToggled.emit)
        self.exposure_apply.clicked.connect(self._apply_suggestion)
        self.capture_btn.clicked.connect(self.captureRequested.emit)
        self.light_btn.toggled.connect(self._on_light_toggled)
        self.calibrate_btn.clicked.connect(self.calibrateTintRequested.emit)
        self.flat_btn.clicked.connect(self.flatRequested.emit)
        self.flat_clear_btn.clicked.connect(self.clearFlatsRequested.emit)
        self.scale_btn.clicked.connect(self.scaleRequested.emit)
        self.screen_combo.currentIndexChanged.connect(self._on_light_ui)
        self.adapter_combo.currentIndexChanged.connect(self._on_adapter)
        self.mode_combo.currentIndexChanged.connect(self._on_light_ui)
        self.brightness.valueChanged.connect(lambda _v: self._on_light_ui())
        self.fullscreen_check.toggled.connect(lambda _v: self._on_light_ui())
        self.darkfield_check.toggled.connect(self._on_capture_options)
        self.darkfield_spin.valueChanged.connect(self._on_capture_options)

    def _on_detect(self) -> None:
        self.camera_status.setText(_("Buscando cámaras…"))
        self.camera_status.show()
        self.detectRequested.emit()

    def _on_connect(self) -> None:
        if self._connected:
            self.disconnectRequested.emit()
            return
        camera = self.selected_camera()
        if camera is not None:
            self.connectRequested.emit(camera)

    def _on_light_toggled(self, on: bool) -> None:
        self.light_btn.setText(_("Apagar luz") if on else _("Encender luz"))
        self.lightToggled.emit(on)

    def _on_capture_options(self, *_args) -> None:
        self._update_darkfield_spin()
        self.captureOptionsChanged.emit(self.capture_options())

    def _update_darkfield_spin(self) -> None:
        self.darkfield_spin.setEnabled(self.darkfield_check.isChecked() and not self._busy)

    def _on_adapter(self, _i: int) -> None:
        self.light.adapter = self.adapter_combo.currentData()
        self.light.rect_mm = None
        self.lightChanged.emit()

    def _on_light_ui(self, *_args) -> None:
        screen = self.screen_combo.currentData()
        names = [s.name() for s in QGuiApplication.screens()]
        current = self.light.screen if self.light.screen in names else QGuiApplication.primaryScreen().name()
        if screen and screen != self.light.screen:
            # Millimetres from another screen's corner mean nothing there; but
            # naming the screen the light is already on must not recentre it.
            if screen != current:
                self.light.rect_mm = None
            self.light.screen = screen
        self.light.mode = self.mode_combo.currentData()
        self.light.brightness = self.brightness.value()
        self.light.fullscreen = self.fullscreen_check.isChecked()
        self.lightChanged.emit()

    # ------------------------------------------------------------ state
    def refresh_screens(self) -> None:
        self.screen_combo.blockSignals(True)
        self.screen_combo.clear()
        for screen in QGuiApplication.screens():
            size = screen.physicalSize()
            label = f"{screen.name()} · {screen.model() or screen.manufacturer()} · {size.width():.0f}×{size.height():.0f} mm"
            self.screen_combo.addItem(label, screen.name())
            self.screen_combo.setItemData(self.screen_combo.count() - 1, label, Qt.ItemDataRole.ToolTipRole)
        idx = self.screen_combo.findData(self.light.screen)
        self.screen_combo.setCurrentIndex(max(idx, 0))
        self.screen_combo.blockSignals(False)

    def load_light(self, light: LightSettings) -> None:
        widgets = (self.screen_combo, self.adapter_combo, self.mode_combo, self.fullscreen_check)
        for w in widgets:
            w.blockSignals(True)
        self.adapter_combo.setCurrentIndex(max(0, self.adapter_combo.findData(light.adapter)))
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(light.mode)))
        self.fullscreen_check.setChecked(light.fullscreen)
        self.brightness.set_value(light.brightness)
        for w in widgets:
            w.blockSignals(False)
        self.show_tint(light.tint)

    def darkfield_enabled(self) -> bool:
        return self.darkfield_check.isChecked()

    def darkfield_stops(self) -> int:
        return self.darkfield_spin.value()

    def capture_options(self) -> dict:
        return {"darkfield": self.darkfield_enabled(), "darkfield_stops": self.darkfield_stops()}

    def load_capture_options(self, data: dict | None) -> None:
        """Restore what ``capture_options`` returned, without echoing it back."""
        data = data or {}
        for w in (self.darkfield_check, self.darkfield_spin):
            w.blockSignals(True)
        self.darkfield_check.setChecked(bool(data.get("darkfield", False)))
        self.darkfield_spin.setValue(int(data.get("darkfield_stops", DARKFIELD_STOPS)))
        for w in (self.darkfield_check, self.darkfield_spin):
            w.blockSignals(False)
        self._update_darkfield_spin()

    def show_tint(self, tint) -> None:
        self.tint_swatch.set_color(tint)
        self.tint_label.setText("R {:.2f} · G {:.2f} · B {:.2f}".format(*tint))

    def set_flat_info(self, count: int) -> None:
        self.flat_label.setText(_("Flats en este rollo: {n}").format(n=count) if count else _("Sin flat-field (opcional)"))

    def set_cameras(self, cameras: list[CameraInfo], keep: str = "") -> None:
        """Fill the camera list; with none, say so instead of showing an empty list."""
        self._cameras = list(cameras)
        models_seen = [c.model for c in cameras]
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for cam in cameras:
            # The USB port only tells two identical bodies apart.
            text = cam.model if models_seen.count(cam.model) == 1 else f"{cam.model} ({cam.port})"
            self.camera_combo.addItem(text)
            self.camera_combo.setItemData(self.camera_combo.count() - 1, f"{cam.model} · {cam.port}",
                                          Qt.ItemDataRole.ToolTipRole)
        self.camera_combo.setCurrentIndex(next((i for i, c in enumerate(cameras) if keep and c.model == keep), 0))
        self.camera_combo.blockSignals(False)
        found = bool(cameras)
        self.camera_combo.setVisible(found)
        self.connect_btn.setVisible(found)
        self.no_camera_label.setVisible(not found)
        self.no_camera_hint.setVisible(not found)
        self.camera_status.setVisible(found)
        if self.camera_status.text() == _("Buscando cámaras…"):
            self.camera_status.setText(_("Sin conectar"))
        self._show_camera_info()

    def selected_camera(self) -> CameraInfo | None:
        idx = self.camera_combo.currentIndex()
        return self._cameras[idx] if 0 <= idx < len(self._cameras) else None

    def _show_camera_info(self) -> None:
        """What libgphoto2 can do with the chosen model, and how to set the camera up."""
        camera = self.selected_camera()
        self.support_label.setVisible(camera is not None)
        self.tips_label.setVisible(camera is not None and not self._connected)
        if camera is None:
            return
        text, warn = support_text(camera.model)
        self.support_label.setText(text)
        self.support_label.setProperty("warn", warn)
        self.support_label.style().polish(self.support_label)
        self.tips_label.setText(tips_html(models.profile_for(camera.model)))
        support = models.support(camera.model)
        if self._connected and support is not None and not support.preview:
            self.live_check.setEnabled(False)
            self.live_check.setToolTip(_("libgphoto2 no ofrece vista en vivo para este modelo"))

    def set_connected(self, connected: bool, text: str = "") -> None:
        self._connected = connected
        self.connect_btn.setText(_("Desconectar") if connected else _("Conectar"))
        self.camera_combo.setEnabled(not connected)
        self.detect_btn.setEnabled(not connected)
        self.focus_row.setVisible(connected)
        self.live_check.setEnabled(connected)
        self.live_check.setToolTip("")
        if not connected:
            self.live_check.setChecked(False)
            self._clear_settings()
            self.live_label.setPixmap(QPixmap())
            self.live_label.setText(_("Conecta la cámara y activa la vista en vivo para encuadrar y enfocar."))
        self.camera_status.setText(text or (_("Conectada") if connected else _("Sin conectar")))
        self.exposure_apply.setVisible(connected and self._suggestion is not None)
        self._show_camera_info()

    def set_busy(self, busy: bool) -> None:
        # The light used for an exposure must be the one the job recorded.
        for w in (self.capture_btn, self.calibrate_btn, self.flat_btn, self.flat_clear_btn, self.mode_combo,
                  self.brightness, self.screen_combo, self.adapter_combo, self.fullscreen_check, self.scale_btn,
                  self.darkfield_check):
            w.setEnabled(not busy)
        self._busy = busy
        self._update_darkfield_spin()

    def set_liveview_checked(self, on: bool) -> None:
        self.live_check.blockSignals(True)
        self.live_check.setChecked(on)
        self.live_check.blockSignals(False)

    def show_compatible_cameras(self) -> None:
        camera = self.selected_camera()
        if self._compat_dialog is None:
            self._compat_dialog = CompatibleCamerasDialog(self)
        self._compat_dialog.select_model(camera.model if camera else "")
        self._compat_dialog.show()
        self._compat_dialog.raise_()

    # ------------------------------------------------------------ settings
    def _clear_settings(self) -> None:
        while self.settings_form.rowCount():
            self.settings_form.removeRow(0)
        self._setting_widgets.clear()
        self._setting_keys.clear()

    def show_settings(self, settings: list[CameraSetting]) -> None:
        self._clear_settings()
        order = {k: i for i, k in enumerate(SETTING_ORDER)}
        for s in sorted(settings, key=lambda s: order.get(s.key, 99)):
            widget = self._make_setting_widget(s)
            self.settings_form.addRow(_(s.label), widget)
            self._setting_widgets[s.name] = widget
            self._setting_keys[s.key] = s

    def setting_name(self, key: str) -> str | None:
        """The camera's control name behind a logical setting ("shutter" → "shutterspeed2" on Nikon)."""
        setting = self._setting_keys.get(key)
        return setting.name if setting else None

    def shutter_suggestion(self, stops: float, from_seconds: float | None = None) -> tuple[str, str] | None:
        """(shot at, suggested) shutter speed for an exposure change of ``stops``.

        ``from_seconds`` is the speed the frame was taken at (EXIF); without it
        the camera's current speed is used.
        """
        setting = self._setting_keys.get("shutter")
        if setting is None or setting.kind != "choice":
            return None
        if from_seconds:
            seconds = from_seconds
            current = closest_shutter(setting.choices, seconds)
        else:
            current = str(setting.value)
            seconds = shutter_seconds(current)
        if current is None or seconds is None:
            return None
        best = closest_shutter(setting.choices, seconds * 2.0 ** stops)
        return (current, best) if best else None

    def set_exposure_suggestion(self, text: str, setting: str | None = None, value: str | None = None) -> None:
        """Advice after a capture; with a value, an "Aplicar" button sets it on the camera.

        ``setting`` is a logical key ("shutter") or the camera's control name.
        An empty ``text`` hides the note.
        """
        name = (self.setting_name(setting) or setting) if setting else None
        self._suggestion = (name, value) if name and value else None
        self.exposure_note.setText(text)
        self.exposure_apply.setText(_("Aplicar"))
        self.exposure_apply.setEnabled(True)
        tip = _("Pone la velocidad de la cámara en {value}").format(value=value) if value else ""
        self.exposure_apply.setToolTip(tip)
        self.exposure_apply.setVisible(self._connected and self._suggestion is not None)
        self.exposure_box.setVisible(bool(text))

    def _apply_suggestion(self) -> None:
        if self._suggestion is None:
            return
        self.settingChanged.emit(*self._suggestion)
        self.exposure_apply.setText(_("✓ Aplicado"))
        self.exposure_apply.setEnabled(False)

    def _make_setting_widget(self, s: CameraSetting) -> QWidget:
        if s.kind == "choice" and not s.readonly:
            combo = _ElidedCombo()
            self.wheel_guard.guard(combo)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(4)
            combo.addItems(s.choices)
            combo.setCurrentText(str(s.value))
            combo.setToolTip(str(s.value))
            combo.setMaxVisibleItems(20)
            combo.textActivated.connect(lambda text, name=s.name: self.settingChanged.emit(name, text))
            return combo
        if s.kind == "range" and not s.readonly and s.range:
            box = QDoubleSpinBox()
            self.wheel_guard.guard(box)
            lo, hi, step = s.range
            box.setRange(lo, hi)
            box.setSingleStep(step or 1)
            box.setValue(float(s.value))
            box.setKeyboardTracking(False)
            box.valueChanged.connect(lambda v, name=s.name: self.settingChanged.emit(name, v))
            return box
        if s.kind == "toggle" and not s.readonly:
            check = QCheckBox()
            check.setChecked(bool(s.value))
            check.toggled.connect(lambda v, name=s.name: self.settingChanged.emit(name, int(v)))
            return check
        label = QLabel(str(s.value))
        label.setStyleSheet(f"color: {DIM};")
        return label

    def show_live(self, image: QImage, note: str = "") -> None:
        zoom = self.zoom_combo.currentData() or 1
        if zoom > 1:
            w, h = image.width() // zoom, image.height() // zoom
            image = image.copy((image.width() - w) // 2, (image.height() - h) // 2, w, h)
        pix = QPixmap.fromImage(image).scaled(
            self.live_label.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if note:
            painter = QPainter(pix)
            painter.fillRect(0, pix.height() - 22, pix.width(), 22, QColor(0, 0, 0, 170))
            painter.setPen(QColor(230, 230, 230))
            painter.drawText(pix.rect().adjusted(6, 0, -6, -4), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, note)
            painter.end()
        self.live_label.setPixmap(pix)


def _fold(text: str) -> str:
    """Lower case without accents, so "camara" finds "Cámara"."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if not unicodedata.combining(c))


class CompatibleCamerasDialog(QDialog):
    """Searchable list of the USB cameras libgphoto2 can fire, with Belka's tips per brand."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(_("Cámaras compatibles"))
        self.resize(760, 560)
        self.setStyleSheet(f"""
            QTreeWidget {{ background: #1e1e1e; color: #c8c8c8; border: 1px solid #141414; outline: 0; }}
            QTreeWidget::item {{ height: 22px; }}
            QTreeWidget::item:hover {{ background: #2c2c2c; }}
            QTreeWidget::item:selected {{ background: #4a4a4a; color: #ececec; }}
            QHeaderView::section {{ background: #262626; color: {DIM}; border: none;
                                    border-bottom: 1px solid #141414; padding: 4px 6px; }}
            QLabel#Hint {{ color: {DIM}; }}
            QLabel#Details {{ background: #262626; color: #c8c8c8; border-radius: 3px; padding: 8px 10px; }}
        """)
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText(_("Buscar modelo o marca (p. ej. «nikon z6», «a7 iv», «eos r»)…"))
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setHeaderLabels([_("Modelo"), _("Vista en vivo"), _("Soporte"), _("Marca")])
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.tree, 1)
        self.count_label = _hint()
        layout.addWidget(self.count_label)
        self.details = QLabel()
        self.details.setObjectName("Details")
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.RichText)
        self.details.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.details.setMinimumHeight(110)
        layout.addWidget(self.details)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._rows: list[tuple[QTreeWidgetItem, str]] = []
        for m in models.capture_models():
            profile = models.profile_for(m.model)
            item = QTreeWidgetItem([m.model, _("sí") if m.preview else "—",
                                    _(DRIVER_STATUS_LABELS[m.status]), profile.name])
            item.setData(0, Qt.ItemDataRole.UserRole, m.model)
            if m.status != "production":
                item.setForeground(2, QColor(WARN))
            self.tree.addTopLevelItem(item)
            self._rows.append((item, _fold(f"{m.model} {profile.name}")))
        self.search.textChanged.connect(self.filter)
        self.tree.currentItemChanged.connect(lambda item, _prev: self._show_details(item))
        self.filter("")
        self._show_details(None)

    def filter(self, text: str) -> None:
        """Keep the rows that contain every word of ``text``."""
        words = _fold(text).split()
        shown = 0
        for item, haystack in self._rows:
            hidden = not all(w in haystack for w in words)
            item.setHidden(hidden)
            shown += not hidden
        self.count_label.setText(_("{n} de {total} modelos con disparo remoto por USB").format(
            n=shown, total=len(self._rows)))

    def visible_models(self) -> list[str]:
        return [item.text(0) for item, _h in self._rows if not item.isHidden()]

    def select_model(self, model: str) -> None:
        """Highlight the detected camera, if libgphoto2 lists it."""
        item = next((i for i, _h in self._rows if i.text(0) == model), None)
        if item is not None:
            self.search.clear()
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item, QTreeWidget.ScrollHint.PositionAtCenter)

    def _show_details(self, item: QTreeWidgetItem | None) -> None:
        if item is None:
            self.details.setText(_("Elige un modelo para ver cómo conectarlo."))
            return
        model = item.text(0)
        profile = models.profile_for(model)
        support, warn = support_text(model)
        color = WARN if warn else DIM
        parts = [f"<b>{html.escape(model)}</b> &nbsp;<span style='color:{DIM}'>{html.escape(profile.name)}</span>",
                 f"<span style='color:{color}'>{html.escape(support)}</span>"]
        self.details.setText("<br>".join(p for p in (*parts, tips_html(profile)) if p))
