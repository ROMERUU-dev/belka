"""Main window, laid out like Lightroom Classic.

    ┌ module bar: Belka · quick actions ············ Biblioteca | Captura | Revelado ┐
    │ left panel │ tools │ loupe / grid                         │ histogram + panels  │
    │            │       │ ─ bottom bar (stars, flag, crop opts) │                     │
    └ filmstrip ─────────────────────────────────────────────────────────────────────┘

The left panel depends on the module (camera and light in Captura; navigator,
film profiles, snapshots and history in Revelado; roll info in Biblioteca).
The right panel (histogram and develop sections) is shared by Captura and
Revelado so a fresh capture can be adjusted on the spot.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QByteArray, QEvent, QObject, QPointF, QRectF, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QImage, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from belka import APP_NAME, __version__
from belka.camera.base import CameraInfo
from belka.camera.worker import CameraController
from belka.core import pipeline as pl
from belka.core.film import FilmProfile, ProfileLibrary, display_name, film_name, slugify
from belka.core.history import HistoryStore
from belka.core.rawio import SUPPORTED_EXTENSIONS, exif_summary
from belka.core.session import Frame, Session
from belka.i18n import _
from belka.light.geometry import LightSettings, load_adapters
from belka.light.panel import LightPanel
from belka.settings import Settings
from belka.system import IdleInhibitor, night_light_enabled, night_light_paused, pause_night_light, resume_night_light
from belka.ui.capture_flow import CaptureFlow
from belka.ui.capture_panel import CapturePanel
from belka.ui.develop_panel import FLAT_IGNORED_FIELDS, TONE_FIELDS, DevelopPanel
from belka.ui.dialogs import ExportDialog, ExportRunner, HelpDialog, NewRollDialog, SaveProfileDialog, default_root
from belka.ui.filmstrip import Filmstrip
from belka.ui.histogram import InteractiveHistogram, zone_label
from belka.ui.icons import icon
from belka.ui.image_view import ImageView
from belka.ui.left_panels import LeftColumn
from belka.ui.live import LiveInverter
from belka.ui.processing import DevelopController, DevelopJob
from belka.ui.toolstrip import ModuleBar, ToolStrip

MODULES = ("library", "capture", "develop")
# Geometry is per frame: copying or syncing settings never carries it over.
GEOMETRY_FIELDS = ("crop", "rotation", "flip_h", "flip_v", "angle", "persp_vertical", "persp_horizontal",
                   "persp_rotate", "persp_aspect", "persp_scale", "persp_x", "persp_y", "crop_aspect",
                   "constrain_crop", "upright_mode", "upright_guides", "base")
# Sections whose on/off switch is part of a frame's geometry, not of its look.
GEOMETRY_SECTIONS = ("transform", "lens")
CROP_ASPECTS = (("original", "Original"), ("free", "Libre"), ("1:1", "1 : 1"), ("4:5", "4 : 5"),
                ("5:7", "5 : 7"), ("2:3", "2 : 3"), ("3:2", "3 : 2"), ("16:9", "16 : 9"))
PANEL_STYLE = "background: #232324;"
# Same ranges as the Básico sliders, so histogram drags and sliders agree.
HIST_RANGES = {"black": (-0.3, 0.3), "white": (-0.3, 0.3), "shadows": (-1.0, 1.0),
               "highlights": (-1.0, 1.0), "exposure": (-5.0, 5.0)}


def _file_filter() -> str:
    return _("Imágenes") + " (" + " ".join(f"*{e} *{e.upper()}" for e in sorted(SUPPORTED_EXTENSIONS)) + ")"


class _KeyGuard(QObject):
    """Lightroom's one-key shortcuts (arrows, digits, Delete, Tab) are window-wide,
    but a text field, a list, a combo or a slider with focus keeps its own keys."""

    NAV = {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Home, Qt.Key.Key_End,
           Qt.Key.Key_PageUp, Qt.Key.Key_PageDown, Qt.Key.Key_Delete, Qt.Key.Key_Backspace}
    DIGITS = {getattr(Qt.Key, f"Key_{n}") for n in range(10)}

    def __init__(self, parent: QObject, own_views: tuple) -> None:
        super().__init__(parent)
        self._own_views = own_views

    def eventFilter(self, obj, event) -> bool:
        if event.type() != QEvent.Type.ShortcutOverride or not isinstance(obj, QWidget):
            return False
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
                                | Qt.KeyboardModifier.MetaModifier):
            return False
        key = Qt.Key(event.key())
        text_input = isinstance(obj, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit))
        if key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab) and text_input:
            event.accept()  # moves to the next field instead of hiding the panels
        elif isinstance(obj, QAbstractItemView) and obj not in self._own_views and key in self.NAV | self.DIGITS:
            event.accept()  # profile tree, history, snapshots: their own navigation and type-ahead
        elif isinstance(obj, (QComboBox, QAbstractSlider)) and key in self.NAV:
            event.accept()
        return False


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.library = ProfileLibrary()
        self.adapters = load_adapters()
        self.light = LightSettings.from_json(settings.get("light"))
        self.session: Session | None = None
        self.frame: Frame | None = None
        self.module = "develop"
        self.view_mode = "positive"
        self.panel: LightPanel | None = None
        self.inhibitor = IdleInhibitor()
        self.history = HistoryStore()
        self._night_light_asked = False
        self._night_light_paused = False
        self._shut_down = False
        self._last_analysis: pl.Analysis | None = None
        self._last_result = None
        self._last_rgb8: np.ndarray | None = None
        self._last_live = 0.0
        self._last_capture_id: str | None = None
        self._noted_capture: str | None = None  # the capture whose exposure note was already posted
        self._exports: list[ExportRunner] = []
        self._clipboard: pl.DevelopSettings | None = None
        self._clip_shadows = False
        self._clip_highlights = False
        self._compare = "off"
        self._crop_backup: pl.DevelopSettings | None = None
        self._crop_fresh = False  # the next crop-mode render places the frame
        self._crop_placed: tuple | None = None  # the frame as placed on entering the tool
        self._requested_ratio: float | None = None  # crop lock the last render's bounds were sized for
        self._rotate_from: float | None = None  # angle when the rotate drag started
        self._grid_on = False
        self._before_key: tuple | None = None
        self._last_failure: tuple[str, float] = ("", 0.0)
        self._show_dust = False
        self._last_dust: np.ndarray | None = None
        self._transform_dragging = False
        self._rotate_total = 0.0
        self._applying_history = False

        self.setWindowTitle(APP_NAME)
        self.camera = CameraController(self)
        self.develop = DevelopController(self)
        self.live = LiveInverter(self)

        self._build_widgets()
        self.capture_panel.load_capture_options(self.settings.get("capture"))
        self.flow = CaptureFlow(
            self.camera, self.light, lambda: self.session, self._show_light_color,
            self._light_visible, self._set_capturing, self, panel_ready=self._panel_ready,
            set_pattern=self._set_light_pattern, darkfield_stops=self._darkfield_stops,
        )
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)
        self._save_timer.timeout.connect(self._save_frame_settings)
        self._build_actions()
        self._build_layout()
        self._connect()
        self._restore_window()
        self.set_module(self.settings.get("module", "develop"))
        QTimer.singleShot(0, self._startup)

    # ================================================================ widgets
    def _build_widgets(self) -> None:
        self.view = ImageView()
        self.grid = Filmstrip(QSize(210, 140), grid=True)
        self.filmstrip = Filmstrip()
        self.histogram = InteractiveHistogram()
        self.develop_panel = DevelopPanel(self.library)
        self.capture_panel = CapturePanel(self.light, self.adapters)
        self.left_column = LeftColumn(self.library)
        self.navigator = self.left_column.navigator
        self.profiles_panel = self.left_column.profiles
        self.snapshots_panel = self.left_column.snapshots
        self.history_panel = self.left_column.history
        self.roll_info = QLabel()
        self.roll_info.setWordWrap(True)
        self.roll_info.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.roll_info.setStyleSheet("color: #b8b8b8; padding: 10px; font-size: 12px;")
        self.recent_list = QListWidget()
        self.recent_list.setStyleSheet("QListWidget { background: #232324; border: none; color: #b8b8b8; }"
                                       "QListWidget::item { padding: 3px 10px; }")
        self.metadata = QLabel()
        self.metadata.setWordWrap(True)
        self.metadata.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.metadata.setStyleSheet("color: #b8b8b8; padding: 10px; font-size: 12px;")
        self.tools = ToolStrip()
        self.module_bar = ModuleBar([("library", _("Biblioteca")), ("capture", _("Captura")), ("develop", _("Revelado"))])

    def _build_layout(self) -> None:
        # Left: one page per module.
        self.left_stack = QStackedWidget()
        library_left = QWidget()
        ll = QVBoxLayout(library_left)
        ll.setContentsMargins(0, 0, 0, 0)
        title = QLabel(_("Rollos recientes"))
        title.setStyleSheet("color: #8a8a8a; padding: 8px 10px 2px; font-size: 11px; text-transform: uppercase;")
        ll.addWidget(self.roll_info)
        ll.addWidget(title)
        ll.addWidget(self.recent_list, 1)
        self.left_stack.addWidget(library_left)
        self.left_stack.addWidget(self.capture_panel)
        self.left_stack.addWidget(self.left_column)
        self.left_stack.setStyleSheet(PANEL_STYLE)

        # Centre: tool strip, loupe or grid, bottom bar.
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self.view)
        self.center_stack.addWidget(self.grid)
        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.tools)
        row.addWidget(self.center_stack, 1)
        cl.addLayout(row, 1)
        cl.addWidget(self._build_bottom_bar())

        # Right: histogram on top, then develop sections or metadata.
        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(self.develop_panel)
        self.right_stack.addWidget(self.metadata)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self.histogram)
        rl.addWidget(self.right_stack, 1)
        right.setMinimumWidth(330)
        right.setStyleSheet(PANEL_STYLE)
        self.right_panel = right

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.left_stack)
        self.splitter.addWidget(centre)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes([300, 1000, 360])
        self.splitter.setHandleWidth(1)
        self.splitter.setStyleSheet("QSplitter::handle { background: #121213; }")

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.module_bar)
        layout.addWidget(self.splitter, 1)
        layout.addWidget(self.filmstrip)
        self.setCentralWidget(root)
        self.camera_label = QLabel(_("Sin cámara"))
        self.night_label = QLabel()
        self.night_label.setStyleSheet("color: #d08a00;")
        self.statusBar().addPermanentWidget(self.night_label)
        self.statusBar().addPermanentWidget(self.camera_label)
        self.statusBar().setStyleSheet("QStatusBar { background: #161617; color: #9a9a9a; }")

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("bottomBar")
        bar.setFixedHeight(38)
        bar.setStyleSheet("""
            #bottomBar { background: #1d1d1e; border-top: 1px solid #2c2c2e; }
            QToolButton { border: none; color: #b8b8b8; padding: 3px; border-radius: 3px; }
            QToolButton:hover { background: #333335; }
            QToolButton:checked { color: #f0a050; }
            QLabel { color: #9a9a9a; }
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(4)
        self.star_buttons = []
        for n in range(1, 6):
            b = QToolButton()
            b.setText("★")
            b.setToolTip(_("{n} estrellas  ({n})").format(n=n))
            b.clicked.connect(lambda _c=False, k=n: self.set_rating(k))
            layout.addWidget(b)
            self.star_buttons.append(b)
        layout.addSpacing(10)
        self.pick_button = QToolButton()
        self.pick_button.setIcon(icon("flag"))
        self.pick_button.setToolTip(_("Marcar como elegida  (P)"))
        self.pick_button.clicked.connect(lambda: self.set_flag(1))
        self.reject_button = QToolButton()
        self.reject_button.setIcon(icon("reject"))
        self.reject_button.setToolTip(_("Marcar como rechazada  (X)"))
        self.reject_button.clicked.connect(lambda: self.set_flag(-1))
        layout.addWidget(self.pick_button)
        layout.addWidget(self.reject_button)
        layout.addSpacing(16)
        # Crop options, shown only while the crop tool is active.
        self.crop_options = QWidget()
        co = QHBoxLayout(self.crop_options)
        co.setContentsMargins(0, 0, 0, 0)
        co.setSpacing(6)
        co.addWidget(QLabel(_("Aspecto")))
        self.aspect_combo = QComboBox()
        for key, label in CROP_ASPECTS:
            self.aspect_combo.addItem(_(label), key)
        self.aspect_combo.currentIndexChanged.connect(self._on_aspect)
        co.addWidget(self.aspect_combo)
        self.crop_done = QPushButton(_("Listo"))
        self.crop_done.clicked.connect(self.commit_crop)
        self.crop_reset = QPushButton(_("Restablecer"))
        self.crop_reset.clicked.connect(self.reset_crop)
        co.addWidget(self.crop_reset)
        co.addWidget(self.crop_done)
        self.crop_options.hide()
        layout.addWidget(self.crop_options)
        layout.addStretch(1)
        self.zoom_label = QLabel("")
        layout.addWidget(self.zoom_label)
        layout.addSpacing(10)
        self.capture_button = QPushButton(icon("shutter"), _("Capturar"))
        self.capture_button.setToolTip(_("Capturar fotograma  (Espacio)"))
        self.capture_button.setStyleSheet(
            "QPushButton { background: #3a3a3c; color: #eee; border: none; border-radius: 4px; padding: 5px 16px; font-weight: 600; }"
            "QPushButton:hover { background: #4a4a4e; }")
        self.capture_button.clicked.connect(self.capture_frame)
        layout.addWidget(self.capture_button)
        return bar

    def _action(self, text: str, slot, shortcut: str | QKeySequence | None = None, icon_name: str | None = None,
                tip: str = "", checkable: bool = False) -> QAction:
        action = QAction(text, self)
        if icon_name:
            action.setIcon(icon(icon_name))
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.setToolTip(tip or text.replace("&", "").rstrip("…"))
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        self.addAction(action)
        return action

    def _build_actions(self) -> None:
        a = self._action
        self.act_new = a(_("Nuevo rollo…"), self.new_roll, QKeySequence.StandardKey.New, "new_roll")
        self.act_open = a(_("Abrir rollo…"), self.open_roll_dialog, QKeySequence.StandardKey.Open, "open_roll")
        self.act_import = a(_("Importar archivos…"), self.import_dialog, "Ctrl+Shift+I", "import",
                            _("Agrega fotos RAW/TIFF/JPEG ya tomadas"))
        self.act_export = a(_("Exportar…"), self.export_dialog, "Ctrl+Shift+E", "export")
        self.act_quit = a(_("Salir"), self.close, QKeySequence.StandardKey.Quit)
        self.act_light = a(_("Panel de luz"), lambda on: self.capture_panel.light_btn.setChecked(on), "L", "light",
                           checkable=True)
        self.act_capture = a(_("Capturar fotograma"), self.capture_frame, None, "shutter")
        # Space captures in Captura (or with the light on) and is Lightroom's
        # zoom key everywhere else, so it has its own always-enabled action.
        self.act_space = a(_("Capturar / zoom"), self.capture_frame, "Space")
        # Modules
        self.act_library = a(_("Biblioteca"), lambda: self.set_module("library"), "G", "library")
        self.act_capture_module = a(_("Captura"), lambda: self.set_module("capture"), "Ctrl+Alt+2", "capture")
        self.act_develop = a(_("Revelado"), lambda: self.set_module("develop"), "D", "develop")
        # Tools (left strip)
        self.act_crop = a(_("Recortar y enderezar"), lambda on: self.toggle_crop(on), "R", "crop", checkable=True)
        self.act_straighten = a(_("Enderezar con una línea"), lambda: self.start_tool("straighten"), "S", "straighten")
        self.act_rot_ccw = a(_("Rotar a la izquierda"), lambda: self.rotate(-90), "Ctrl+[", "rotate_left")
        self.act_rot_cw = a(_("Rotar a la derecha"), lambda: self.rotate(90), "Ctrl+]", "rotate_right")
        self.act_flip_h = a(_("Voltear horizontal"), lambda: self.flip("h"), "Shift+H", "flip_horizontal",
                            _("Voltear horizontal") + "\n"
                            + _("Si escaneaste por el lado de la emulsión, la imagen sale en espejo"))
        self.act_flip_v = a(_("Voltear vertical"), lambda: self.flip("v"), "Shift+V", "flip_vertical")
        self.act_perspective = a(_("Upright guiado (reglas)"), lambda: self.request_upright("guided"), "Shift+T",
                                 "perspective",
                                 _("Upright guiado") + "\n"
                                 + _("Traza de 2 a 4 líneas sobre bordes que deban quedar verticales u horizontales"))
        self.act_transform = a(_("Panel Transformar"), self.show_transform)
        self.act_upright = a(_("Enderezar automáticamente"), lambda: self.request_upright("auto"), "Shift+U",
                             "upright_auto")
        self.act_base = a(_("Medir la base de la película"), lambda: self.start_tool("base"), "B", "eyedropper_base")
        self.act_wb = a(_("Balance de blancos (gris neutro)"), lambda: self.start_tool("neutral"), "W", "eyedropper_wb")
        self.act_auto_tone = a(_("Tono automático"), lambda: self.auto_tone(), "Ctrl+U")
        self.act_grid = a(_("Mostrar cuadrícula"), self.toggle_grid, "Ctrl+Alt+O", "grid_overlay", checkable=True)
        self.act_negative = a(_("Ver negativo"), self.toggle_negative, "N", "negative", checkable=True)
        self.act_compare = a(_("Antes / después"), lambda: self.toggle_compare("split"), "\\", "compare_split")
        self.act_compare_side = a(_("Antes / después lado a lado"), lambda: self.toggle_compare("side"), "Y",
                                  "compare")
        self.act_clipping = a(_("Mostrar recorte de tonos"), self.toggle_clipping, "J", "clipping", checkable=True)
        self.act_zoom = a(_("Zoom 1:1 / ajustar"), lambda: self.view.toggle_zoom(), "Z", "zoom_100")
        self.act_fit = a(_("Ajustar a la ventana"), lambda: self.view.fit(), "Ctrl+0", "zoom_fit")
        self.act_undo = a(_("Deshacer"), self.undo, QKeySequence.StandardKey.Undo, "undo")
        self.act_redo = a(_("Rehacer"), self.redo, QKeySequence.StandardKey.Redo, "redo")
        self.act_cancel = a(_("Cancelar herramienta"), self.cancel_tool, "Escape")
        # Settings
        self.act_copy = a(_("Copiar ajustes"), self.copy_settings, "Ctrl+Shift+C", "copy_settings")
        self.act_paste = a(_("Pegar ajustes"), self.paste_settings, "Ctrl+Shift+V", "paste_settings")
        self.act_previous = a(_("Ajustes del anterior"), self.paste_previous, "Ctrl+Alt+V")
        self.act_sync = a(_("Sincronizar ajustes con la selección"), self.sync_settings, "Ctrl+Shift+S", "sync")
        self.act_reset = a(_("Restablecer ajustes"), self.reset_settings, "Ctrl+Shift+R", "reset")
        self.act_snapshot = a(_("Nueva instantánea…"), self.create_snapshot_dialog, "Ctrl+N", "snapshot")
        # Frames
        self.act_prev = a(_("Fotograma anterior"), lambda: self.step_frame(-1), "Left")
        self.act_next = a(_("Fotograma siguiente"), lambda: self.step_frame(1), "Right")
        self.act_delete = a(_("Quitar o eliminar…"), lambda: self.delete_frames(), QKeySequence.StandardKey.Delete)
        self.act_remove = a(_("Quitar del rollo…"), lambda: self.delete_frames("remove"))
        self.act_trash = a(_("Eliminar del disco…"), lambda: self.delete_frames("trash"))
        self.act_delete_rejected = a(_("Eliminar fotos rechazadas…"), self.delete_rejected, "Ctrl+Backspace")
        self.act_show_folder = a(_("Mostrar en la carpeta"), self.show_in_folder)
        self.act_pick = a(_("Elegida"), lambda: self.set_flag(1), "P", "flag")
        self.act_reject = a(_("Rechazada"), lambda: self.set_flag(-1), "X", "reject")
        self.act_unflag = a(_("Sin marca"), lambda: self.set_flag(0), "U")
        self.rating_actions = [a(_("{n} estrellas").format(n=n), lambda _c=False, k=n: self.set_rating(k), str(n))
                               for n in range(0, 6)]
        # Panels
        self.act_left = a(_("Panel izquierdo"), lambda: self._toggle_panel(0), "F7")
        self.act_right = a(_("Panel derecho"), lambda: self._toggle_panel(2), "F8")
        self.act_strip = a(_("Tira de fotogramas"), lambda: self.filmstrip.setVisible(not self.filmstrip.isVisible()), "F6")
        self.act_panels = a(_("Ocultar paneles"), self._toggle_side_panels, "Tab")
        self.act_help = a(_("Guía rápida"), lambda: HelpDialog(self).exec(), "F1", "help")
        self.act_about = a(_("Acerca de Belka"), self.about, None, "info")
        # New-roll shortcut belongs to the roll; Ctrl+N is the Lightroom snapshot key.
        self.act_new.setShortcut(QKeySequence("Ctrl+Shift+N"))
        # Arrows, digits, Delete and Tab are window-wide, as in Lightroom; text
        # fields, lists and combos keep them through _KeyGuard.
        self._key_guard = _KeyGuard(self, (self.grid, self.filmstrip))
        QApplication.instance().installEventFilter(self._key_guard)

        bar = self.menuBar()
        bar.setStyleSheet("QMenuBar { background: #161617; color: #b0b0b0; } QMenuBar::item:selected { background: #2e2e30; }")
        m = bar.addMenu(_("&Archivo"))
        for act in (self.act_new, self.act_open):
            m.addAction(act)
        self.recent_menu = m.addMenu(_("Rollos recientes"))
        m.addSeparator()
        for act in (self.act_import, self.act_export):
            m.addAction(act)
        m.addSeparator()
        m.addAction(self.act_quit)
        m = bar.addMenu(_("&Editar"))
        for act in (self.act_undo, self.act_redo, None, self.act_copy, self.act_paste, self.act_previous,
                    self.act_sync, None, self.act_auto_tone, self.act_reset, self.act_snapshot):
            m.addSeparator() if act is None else m.addAction(act)
        m = bar.addMenu(_("&Foto"))
        for act in (self.act_rot_ccw, self.act_rot_cw, self.act_flip_h, self.act_flip_v, None, self.act_crop,
                    self.act_straighten, self.act_upright, self.act_perspective, self.act_transform, None,
                    self.act_pick, self.act_reject, self.act_unflag, None, self.act_prev, self.act_next, None,
                    self.act_show_folder, self.act_remove, self.act_trash, self.act_delete_rejected):
            m.addSeparator() if act is None else m.addAction(act)
        m = bar.addMenu(_("&Vista"))
        for act in (self.act_library, self.act_capture_module, self.act_develop, None, self.act_negative,
                    self.act_compare, self.act_compare_side, self.act_clipping, self.act_grid, self.act_zoom,
                    self.act_fit, None, self.act_left, self.act_right, self.act_strip, self.act_panels):
            m.addSeparator() if act is None else m.addAction(act)
        m = bar.addMenu(_("&Captura"))
        for act in (self.act_capture, self.act_light):
            m.addAction(act)
        m = bar.addMenu(_("A&yuda"))
        m.addAction(self.act_help)
        m.addAction(self.act_about)

        # What fits a 768 px screen; undo/redo and zoom live in the menus and the Navigator.
        for act in (self.act_crop, self.act_straighten, self.act_upright, self.act_perspective, None,
                    self.act_rot_ccw, self.act_rot_cw, self.act_flip_h, self.act_flip_v, None,
                    self.act_base, self.act_wb, None, self.act_negative, self.act_compare, self.act_compare_side,
                    self.act_clipping, self.act_grid):
            self.tools.addSeparator() if act is None else self.tools.add(act)
        for act in (self.act_new, self.act_open, self.act_import, self.act_export, self.act_light):
            self.module_bar.add_quick(act)

    def _connect(self) -> None:
        cp, dp, w = self.capture_panel, self.develop_panel, self.camera.worker
        cp.detectRequested.connect(self.camera.request_detect.emit)
        cp.connectRequested.connect(self._connect_camera)
        cp.disconnectRequested.connect(self.camera.request_close.emit)
        cp.settingChanged.connect(self.camera.request_setting.emit)
        cp.focusRequested.connect(self.camera.request_focus.emit)
        cp.autofocusRequested.connect(self.camera.request_autofocus.emit)
        cp.liveviewToggled.connect(self.camera.request_liveview.emit)
        cp.captureRequested.connect(self.capture_frame)
        cp.lightToggled.connect(self.set_light_on)
        cp.lightChanged.connect(self._on_light_settings)
        cp.calibrateTintRequested.connect(self.flow.calibrate_tint)
        cp.flatRequested.connect(self.capture_flat)
        cp.clearFlatsRequested.connect(self.clear_flats)
        cp.scaleRequested.connect(self.calibrate_scale)
        cp.captureOptionsChanged.connect(lambda options: self.settings.set("capture", options))
        # Worker signals must reach bound methods of GUI objects: a lambda has no
        # thread affinity and would run on the camera thread.
        w.cameras_found.connect(self._on_cameras)
        w.opened.connect(self._on_camera_opened)
        w.closed.connect(self._on_camera_closed)
        w.settings_ready.connect(cp.show_settings)
        w.preview_frame.connect(self._on_live)
        w.failed.connect(self._camera_error)
        w.busy_changed.connect(self._on_camera_busy)
        w.liveview_stopped.connect(self._on_liveview_stopped)
        self.flow.busyChanged.connect(cp.set_busy)
        self.flow.frameCaptured.connect(self._on_frame_captured)
        self.flow.flatsChanged.connect(self._on_flats_changed)
        self.flow.tintCalibrated.connect(self._on_tint)
        self.flow.message.connect(self._message)
        self.flow.error.connect(self._error)
        dp.settingsChanged.connect(self._on_settings)
        dp.editCommitted.connect(self.commit_history)
        dp.toolRequested.connect(self.start_tool)
        dp.uprightRequested.connect(self.request_upright)
        dp.lockBaseChanged.connect(self._on_lock_base)
        dp.applyAllRequested.connect(self.apply_to_roll)
        dp.saveProfileRequested.connect(self.save_profile)
        dp.resetRequested.connect(self.reset_settings)
        dp.previousRequested.connect(self.paste_previous)
        dp.autoToneRequested.connect(lambda: self.auto_tone())
        dp.autoFieldRequested.connect(lambda field: self.auto_tone((field,)))
        dp.gridToggled.connect(self._on_grid_toggled)
        dp.dustMaskToggled.connect(self._on_dust_mask_toggled)
        dp.edgeProfileRequested.connect(self._on_profile_chosen)
        dp.transformDragging.connect(self._on_transform_dragging)
        self.histogram.adjustRequested.connect(self._on_histogram_drag)
        self.histogram.dragFinished.connect(self._on_histogram_drag_finished)
        self.histogram.clippingToggled.connect(self._on_clipping_toggled)
        self.view.regionSelected.connect(self._on_region)
        self.view.filesDropped.connect(self.import_files)
        self.view.zoomChanged.connect(self._on_zoom_changed)
        self.view.rotateStarted.connect(self._on_rotate_started)
        self.view.angleDelta.connect(self._on_angle_delta)
        self.view.rotateFinished.connect(self._on_rotate_finished)
        self.view.cropCommitted.connect(self._on_crop_committed)
        self.view.cropChanged.connect(self._on_crop_changed)
        self.view.cropCancelled.connect(self.cancel_crop)
        self.view.straightenLine.connect(self._on_straighten_line)
        self.view.pixelHovered.connect(self._on_pixel_hovered)
        self.view.guidesChanged.connect(self._on_guides_changed)
        self.view.guideLimitReached.connect(
            lambda: self._message(_("Máximo 4 guías: quita una con clic derecho o Supr.")))
        self.view.toolFinished.connect(lambda _tool: self.cancel_tool())
        self.view.toolCancelled.connect(lambda _tool: self.cancel_tool())
        self.view.contextMenuRequested.connect(self._photo_menu)
        self.filmstrip.customContextMenuRequested.connect(lambda pos: self._strip_menu(self.filmstrip, pos))
        self.grid.customContextMenuRequested.connect(lambda pos: self._strip_menu(self.grid, pos))
        # The crop keys (Enter, Esc, X) belong to the view again once an aspect is picked.
        self.aspect_combo.activated.connect(lambda _i: self.view.setFocus())
        self.view.viewportChanged.connect(self._on_viewport)
        self.navigator.panRequested.connect(self.view.center_on)
        self.navigator.zoomRequested.connect(self._on_navigator_zoom)
        self.profiles_panel.profileChosen.connect(self._on_profile_chosen)
        self.snapshots_panel.snapshotCreate.connect(self.create_snapshot)
        self.snapshots_panel.snapshotApply.connect(self.apply_snapshot)
        self.snapshots_panel.snapshotDelete.connect(self.delete_snapshot)
        self.history_panel.historyJump.connect(self.jump_history)
        self.history_panel.historyClear.connect(self.clear_history)
        self.filmstrip.frameSelected.connect(self.select_frame)
        self.grid.frameSelected.connect(self.select_frame)
        self.grid.frameActivated.connect(lambda fid: (self.select_frame(fid), self.set_module("develop")))
        self.filmstrip.frameActivated.connect(lambda fid: self.set_module("develop"))
        self.module_bar.moduleChanged.connect(self.set_module)
        self.recent_list.itemActivated.connect(lambda item: self.open_roll(Path(item.data(Qt.ItemDataRole.UserRole))))
        self.develop.rendered.connect(self._on_rendered)
        self.live.ready.connect(self._on_live_image)
        self.develop.before.connect(self._on_before)
        self.develop.thumbnail.connect(self._on_thumbnail)
        self.develop.sampled.connect(self._on_sampled)
        self.develop.failed.connect(self._on_develop_failed)
        app = QGuiApplication.instance()
        for signal in (app.screenAdded, app.screenRemoved, app.primaryScreenChanged):
            signal.connect(lambda *_a: QTimer.singleShot(0, self._on_screens_changed))

    # ================================================================ modules and panels
    def set_module(self, module: str) -> None:
        if module not in MODULES:
            module = "develop"
        if self.module == "capture" and module != "capture" and self.capture_panel.live_check.isChecked():
            self.capture_panel.live_check.setChecked(False)
        self.module = module
        self.module_bar.set_module(module)
        self.settings.data["module"] = module
        self.left_stack.setCurrentIndex(MODULES.index(module))
        self.center_stack.setCurrentIndex(1 if module == "library" else 0)
        self.right_stack.setCurrentIndex(1 if module == "library" else 0)
        self.tools.setVisible(module != "library")
        self.filmstrip.setVisible(module != "library")
        self.capture_button.setVisible(module == "capture")
        self.zoom_label.setVisible(module != "library")
        self.act_capture.setEnabled(module == "capture" or self._light_visible())
        if module != "develop":
            self._close_frame_tools()
        self._update_metadata()

    def _toggle_panel(self, index: int) -> None:
        widget = self.splitter.widget(index)
        widget.setVisible(not widget.isVisible())

    def _toggle_side_panels(self) -> None:
        visible = self.left_stack.isVisible() or self.right_panel.isVisible()
        self.left_stack.setVisible(not visible)
        self.right_panel.setVisible(not visible)

    def show_transform(self) -> None:
        self.set_module("develop")
        self.develop_panel.show_section("transform")

    # ================================================================ startup / shutdown
    def _restore_window(self) -> None:
        geo = self.settings.get("window_geometry")
        if geo:
            self.restoreGeometry(QByteArray.fromBase64(geo.encode()))
        else:
            self.resize(1600, 960)
        sizes = self.settings.get("splitter_sizes")
        if sizes:
            self.splitter.setSizes(sizes)
        state = self.settings.get("develop_panel_state")
        if state:
            self.develop_panel.set_state(state)

    def _startup(self) -> None:
        self._refresh_recent()
        self.camera.request_detect.emit()
        recent = self.settings.recent_sessions()
        if recent:
            try:
                self.open_roll(Path(recent[0]))
            except (OSError, ValueError) as exc:
                self._message(_("No se pudo abrir el último rollo: {msg}").format(msg=exc))
        self._update_night_light_label()
        self.capture_panel.set_flat_info(len(self.session.flats) if self.session else 0)

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        """Leave the desktop as it was. Also runs on SIGTERM and logout (aboutToQuit)."""
        if self._shut_down:
            return
        self._shut_down = True
        self.flow.cancel()
        for runner in list(self._exports):
            # The current frame finishes writing; then the thread can stop.
            runner.cancel()
            runner.thread.quit()
            runner.thread.wait()
        try:
            self._close_frame_tools()
            self._save_frame_settings()
            self.settings.data["window_geometry"] = bytes(self.saveGeometry().toBase64()).decode()
            self.settings.data["splitter_sizes"] = self.splitter.sizes()
            self.settings.data["develop_panel_state"] = self.develop_panel.get_state()
            self.settings.data["light"] = self.light.to_json()
            self.settings.save()
        except OSError as exc:
            # A full or vanished disk must not keep the camera, the inhibitor
            # and Night Light from being released below.
            import sys

            print(f"Belka: no se pudieron guardar los ajustes al salir: {exc}", file=sys.stderr)
        if self.panel is not None:
            try:
                self.panel.closed.disconnect()
            except (RuntimeError, TypeError):
                pass
            self.panel.close()
        self.inhibitor.stop()
        if self._night_light_paused:
            resume_night_light()
            self._night_light_paused = False
        self.camera.shutdown()
        self.develop.shutdown()
        self.live.shutdown()

    # ================================================================ messages
    def _message(self, text: str) -> None:
        self.statusBar().showMessage(text, 8000)
        if self.panel is not None and self.panel.isVisible():
            self.panel.set_status(text)

    def _error(self, text: str) -> None:
        self._message("⚠ " + text)
        if self.panel is None or not self.panel.isVisible() or self.panel.screen() != self.screen():
            QMessageBox.warning(self, APP_NAME, text)

    def _camera_error(self, text: str) -> None:
        self._error(text)

    def about(self) -> None:
        from PySide6.QtGui import QIcon

        from belka import paths

        box = QMessageBox(self)
        box.setWindowTitle(_("Acerca de Belka"))
        box.setIconPixmap(QIcon(str(paths.data_dir() / "icons" / "belka-badge.svg")).pixmap(112, 112))
        box.setText(
            f"<h3>Belka {__version__}</h3>"
            + _("<p>Digitalización de negativos con cámara y la pantalla como fuente de luz.</p>"
                "<p>Captura con libgphoto2, revelado con LibRaw e inversión por densidad con perfiles de película.</p>")
            + "<p>© 2026 Juvenal Romero Pedraza (ROMERUU-dev) · MIT</p>"
        )
        box.exec()

    # ================================================================ rolls
    def _refresh_recent(self) -> None:
        self.recent_menu.clear()
        self.recent_list.clear()
        for path in self.settings.recent_sessions():
            act = self.recent_menu.addAction(Path(path).name)
            act.triggered.connect(lambda _c=False, p=path: self.open_roll(Path(p)))
            self.recent_list.addItem(Path(path).name)
            self.recent_list.item(self.recent_list.count() - 1).setData(Qt.ItemDataRole.UserRole, path)
        self.recent_menu.setEnabled(bool(self.recent_menu.actions()))

    def new_roll(self) -> bool:
        if self.flow.busy:
            self._message(_("Espera: hay una captura en curso."))
            return False
        current = self.develop_panel.settings.profile_id if self.session else "generic-c41"
        dialog = NewRollDialog(self.library, Path(self.settings.get("sessions_root", str(default_root()))), current, self)
        if dialog.exec() != NewRollDialog.DialogCode.Accepted:
            return False
        name, profile_id, root = dialog.values()
        try:
            session = Session.create(root, name, profile_id)
        except OSError as exc:
            self._error(_("No se pudo crear el rollo: {msg}").format(msg=exc))
            return False
        self.settings.set("sessions_root", str(root))
        self._load_session(session)
        return True

    def open_roll_dialog(self) -> None:
        start = self.settings.get("sessions_root", str(default_root()))
        folder = QFileDialog.getExistingDirectory(self, _("Abrir rollo"), start)
        if folder:
            try:
                self.open_roll(Path(folder))
            except (OSError, ValueError) as exc:
                self._error(_("Esa carpeta no es un rollo de Belka: {msg}").format(msg=exc))

    def open_roll(self, path: Path) -> None:
        self._load_session(Session.load(path))

    def _load_session(self, session: Session) -> None:
        if self.flow.busy:
            self._message(_("Espera: hay una captura en curso."))
            return
        self._close_frame_tools()
        self._save_frame_settings()
        self.session = session
        self.frame = None
        self.history.clear()
        self.settings.add_recent_session(session.path)
        self._refresh_recent()
        self.setWindowTitle(f"{APP_NAME} — {session.name}")
        self.module_bar.set_subtitle(session.name)
        self.develop.cancel_thumbnails()
        for strip in (self.filmstrip, self.grid):
            strip.set_frames(session.frames)
        self.capture_panel.set_flat_info(len(session.flats))
        self._clear_loupe()
        for frame in session.frames:
            self._queue_thumbnail(frame)
        if session.frames:
            self.filmstrip.select_frame(session.frames[-1].id)
        else:
            self.develop_panel.load(session.roll_defaults(), session.lock_base)
            self.develop_panel.setEnabled(False)
        self._update_metadata()
        self._message(_("Rollo «{name}»: {n} fotogramas").format(name=session.name, n=len(session.frames)))

    def _ensure_session(self) -> bool:
        return self.session is not None or self.new_roll()

    def _update_metadata(self) -> None:
        if self.session is None:
            self.roll_info.setText(_("Sin rollo abierto.\n\nArchivo → Nuevo rollo (Ctrl+Mayús+N)"))
            self.metadata.setText("")
            return
        s = self.session
        profile = self.library.get(s.film_profile)
        picks = sum(1 for f in s.frames if f.flag == 1)
        rejects = sum(1 for f in s.frames if f.flag == -1)
        self.roll_info.setText(
            f"<b style='color:#e0e0e0;font-size:14px'>{s.name}</b><br>"
            + _("Película: {film}").format(film=film_name(profile)) + "<br>"
            + _("Fotogramas: {n}").format(n=len(s.frames)) + "<br>"
            + _("Elegidas: {p} · Rechazadas: {r}").format(p=picks, r=rejects) + "<br>"
            + (_("Cámara: {cam}").format(cam=s.camera) + "<br>" if s.camera else "")
            + f"<span style='color:#7a7a7a'>{s.path}</span>")
        if self.frame is not None:
            f = self.frame
            exif = (self._last_result.meta.get("exif") if self._last_result else None) or {}
            rows = [(_("Fotograma"), f.label), (_("Capturado"), f.captured.replace("T", " ")),
                    (_("Archivo"), Path(f.files[0]).name), (_("Valoración"), "★" * f.rating or "—")]
            if exif:
                rows.append((_("Cámara"), exif.get("model", "")))
                rows.append((_("Exposición"), exif_summary(exif)))
            edge = f.edge if isinstance(f.edge, dict) else {}
            if edge.get("film") or edge.get("frame"):
                rows.append((_("Borde"), " · ".join(v for v in (edge.get("film", ""), edge.get("frame", "")) if v)))
            self.metadata.setText("<table cellspacing=4>" + "".join(
                f"<tr><td style='color:#7a7a7a'>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>")

    # ================================================================ import / export
    def import_dialog(self) -> None:
        start = self.settings.get("import_dir", str(Path.home()))
        files, _f = QFileDialog.getOpenFileNames(self, _("Importar negativos"), start, _file_filter())
        if files:
            self.settings.set("import_dir", str(Path(files[0]).parent))
            self.import_files(files)

    def import_files(self, files: list[str]) -> None:
        paths_ = [Path(f) for f in files if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS]
        if not paths_:
            self._message(_("Ningún archivo compatible."))
            return
        if not self._ensure_session():
            return
        assert self.session is not None
        frames, errors = self.session.import_files(paths_, copy=True)
        for frame in frames:
            self.history.discard(frame.id)
            for strip in (self.filmstrip, self.grid):
                strip.add_frame(frame, select=False)
            self._queue_thumbnail(frame)
        if frames:
            self.filmstrip.select_frame(frames[0].id)
        if errors:
            self._error(_("Se importaron {n} de {total}. No se pudo importar: {msg}").format(
                n=len(frames), total=len(paths_), msg="\n".join(errors)))
        else:
            self._message(_("{n} archivos importados.").format(n=len(frames)))
        self._update_metadata()

    def export_dialog(self) -> None:
        if self.session is None or not self.session.frames:
            self._message(_("No hay fotogramas que exportar."))
            return
        self._save_frame_settings()
        selected = self._selected_ids()
        dialog = ExportDialog(self.session, len(selected), len(self.session.frames), self)
        if dialog.exec() != ExportDialog.DialogCode.Accepted:
            return
        scope, options = dialog.options()
        frames = [f for f in self.session.frames if scope == "all" or f.id in selected]
        progress = QProgressDialog(_("Exportando…"), _("Cancelar"), 0, len(frames), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        runner = ExportRunner(self.session, frames, self.library, options, self)
        runner.progress.connect(lambda i, n, label: (progress.setValue(i), progress.setLabelText(
            _("Exportando {label} ({i}/{n})…").format(label=label, i=i + 1, n=n))))
        progress.canceled.connect(runner.cancel)

        def done(written: list, errors: list) -> None:
            progress.setValue(len(frames))
            if runner in self._exports:
                self._exports.remove(runner)
            if self._shut_down:
                return
            if errors:
                QMessageBox.warning(self, APP_NAME, "\n".join(errors))
            self._message(_("Exportados {n} archivos en {folder}").format(n=len(written), folder=options.folder))

        runner.finished.connect(done)
        self._exports.append(runner)
        runner.start()

    # ================================================================ frames
    def _selected_ids(self) -> list[str]:
        strip = self.grid if self.module == "library" else self.filmstrip
        return strip.selected_ids()

    def _queue_thumbnail(self, frame: Frame) -> None:
        if self.session is None:
            return
        settings = self.session.settings_for(frame)
        self.develop.thumbnail_for(DevelopJob(
            kind="thumb", session=self.session, frame=frame, settings=settings,
            profile=self.library.get(settings.profile_id), max_side=420,
        ))

    def _on_thumbnail(self, result) -> None:
        if self.session is None:
            return
        frame = self.session.frame(result.job.frame.id)
        if frame is None:
            return
        current = (self._current_settings() if self.frame is not None and frame.id == self.frame.id
                   else self.session.settings_for(frame))
        if result.job.settings != current:
            # Queued before an edit: ask again with today's settings (one job
            # per frame, so this converges).
            self._queue_thumbnail(frame)
            return
        for strip in (self.filmstrip, self.grid):
            strip.set_thumbnail(frame.id, result.image)

    def select_frame(self, frame_id: str) -> None:
        if self.session is None:
            return
        frame = self.session.frame(frame_id)
        if frame is None or (self.frame is not None and frame.id == self.frame.id):
            return
        self._close_frame_tools()
        self._save_frame_settings()
        self.frame = frame
        self._last_analysis = None
        for strip in (self.filmstrip, self.grid):
            strip.blockSignals(True)
            strip.select_frame(frame_id, keep_selection=True)
            strip.mark_current(frame_id)
            strip.blockSignals(False)
        settings = self.session.settings_for(frame)
        self.develop_panel.setEnabled(True)
        self.develop_panel.load(settings, self.session.lock_base)
        self._show_frame_extras(frame)
        self.history.get(frame.id, settings, _("Abrir"))
        # Another frame's snapshot of the same name must not stay selected.
        self.snapshots_panel.list.clearSelection()
        self._refresh_history_panels()
        self._refresh_rating_bar()
        self.request_render()
        if self._compare != "off":
            self._request_before()
        self._update_metadata()

    def _close_frame_tools(self) -> None:
        """Leaving a photo or Develop applies a crop in progress, as Lightroom does;
        other tools just close (their edits are already applied)."""
        if self.view.tool == "crop":
            self.commit_crop()
        else:
            self.cancel_tool()

    def step_frame(self, delta: int) -> None:
        if self.session is None or self.frame is None:
            return
        ids = [f.id for f in self.session.frames]
        idx = ids.index(self.frame.id) + delta
        if 0 <= idx < len(ids):
            self.filmstrip.select_frame(ids[idx])
            self.select_frame(ids[idx])

    def delete_frames(self, mode: str | None = None, ids: list[str] | None = None) -> None:
        """Remove frames from the roll ("remove") or also send their files to the trash ("trash").

        Without a mode it asks, like Lightroom's Delete key. The roll's other
        frames, its flats and the exported positives are never touched.
        """
        if self.session is None:
            return
        if ids is None:
            ids = self._selected_ids() or ([self.frame.id] if self.frame else [])
        if not ids:
            return
        n = len(ids)
        choice = self._ask_delete(mode, n)
        if choice is None:
            return
        self._close_frame_tools()
        self._save_frame_settings()
        # Detach first: removing items moves the strips' current item and
        # fires frameSelected mid-loop, which must not save into a dying frame.
        self.frame = None
        kept: list[Path] = []
        failure = ""
        for strip in (self.filmstrip, self.grid):
            strip.blockSignals(True)
        try:
            for frame_id in ids:
                frame = self.session.frame(frame_id)
                if frame is None:
                    continue
                for strip in (self.filmstrip, self.grid):
                    strip.remove_frame(frame_id)
                # The next capture may reuse the id: it must not inherit this history.
                self.history.discard(frame_id)
                kept += self.session.remove_frame(frame, delete_files=choice == "trash")
        except OSError as exc:
            failure = str(exc)
        finally:
            # Whatever happened, the strips must answer clicks again.
            for strip in (self.filmstrip, self.grid):
                strip.blockSignals(False)
        if failure:
            self._error(_("No se pudo terminar de quitar los fotogramas: {msg}").format(msg=failure))
        if self.session.frames:
            last = self.session.frames[-1].id
            self.filmstrip.select_frame(last)
            self.select_frame(last)
        else:
            self._clear_loupe()
            self.develop_panel.setEnabled(False)
        self._update_metadata()
        if kept:
            self._error(_("No se pudieron mandar a la papelera; siguen en el disco:\n{files}").format(
                files="\n".join(str(p) for p in kept)))
        elif choice == "trash":
            self._message(_("{n} fotograma(s) eliminados (los archivos están en la papelera).").format(n=n))
        else:
            self._message(_("{n} fotograma(s) quitados del rollo.").format(n=n))

    def _ask_delete(self, mode: str | None, n: int) -> str | None:
        """Asks; returns "trash", "remove" or None when cancelled."""
        box = QMessageBox(self)
        box.setWindowTitle(_("Quitar o eliminar"))
        box.setIcon(QMessageBox.Icon.Question)
        trash = remove = None
        if mode in (None, "trash"):
            box.setText(_("¿Eliminar {n} fotograma(s) del disco?").format(n=n) if mode == "trash"
                        else _("¿Qué hacer con {n} fotograma(s)?").format(n=n))
            box.setInformativeText(_("«Eliminar del disco» manda los archivos de la captura a la papelera y los "
                                     "quita del rollo. «Quitar del rollo» deja los archivos en la carpeta."))
            trash = box.addButton(_("Eliminar del disco"), QMessageBox.ButtonRole.DestructiveRole)
        if mode in (None, "remove"):
            if mode == "remove":
                box.setText(_("¿Quitar {n} fotograma(s) del rollo? Los archivos se quedan en la carpeta.").format(n=n))
            remove = box.addButton(_("Quitar del rollo"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(remove or trash)
        box.exec()
        clicked = box.clickedButton()
        if clicked is not None and clicked is trash:
            return "trash"
        if clicked is not None and clicked is remove:
            return "remove"
        return None

    def delete_rejected(self) -> None:
        """Lightroom's "Delete Rejected Photos": for the failed captures marked with X."""
        if self.session is None:
            return
        ids = [f.id for f in self.session.frames if f.flag == -1]
        if not ids:
            self._message(_("No hay fotos marcadas como rechazadas (X)."))
            return
        self.delete_frames(None, ids)

    def show_in_folder(self) -> None:
        if self.session is None:
            return
        target = self.session.path
        if self.frame is not None:
            target = self.session.resolve(self.frame.files[0]).parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def set_rating(self, rating: int) -> None:
        for frame in self._frames_for_marks():
            frame.rating = max(0, min(5, rating))
            for strip in (self.filmstrip, self.grid):
                strip.update_frame(frame)
        if self.session is not None:
            self.session.save()
        self._refresh_rating_bar()
        self._update_metadata()

    def set_flag(self, flag: int) -> None:
        for frame in self._frames_for_marks():
            frame.flag = 0 if frame.flag == flag and flag != 0 else flag
            for strip in (self.filmstrip, self.grid):
                strip.update_frame(frame)
        if self.session is not None:
            self.session.save()
        self._refresh_rating_bar()
        self._update_metadata()

    def _frames_for_marks(self) -> list[Frame]:
        if self.session is None:
            return []
        ids = self._selected_ids() or ([self.frame.id] if self.frame else [])
        return [f for f in self.session.frames if f.id in ids]

    def _refresh_rating_bar(self) -> None:
        rating = self.frame.rating if self.frame else 0
        flag = self.frame.flag if self.frame else 0
        for i, b in enumerate(self.star_buttons):
            b.setStyleSheet("color: #f0f0f0;" if i < rating else "color: #5a5a5c;")
        self.pick_button.setStyleSheet("color: #f0f0f0; background: #3a3a3c;" if flag == 1 else "")
        self.reject_button.setStyleSheet("color: #e05050; background: #3a2a2a;" if flag == -1 else "")

    # ================================================================ developing
    def _clear_loupe(self) -> None:
        self.view.set_image(None)
        self.histogram.set_histogram(None)
        self.histogram.set_info("")
        self.navigator.set_image(None)
        self._last_result = None
        self._last_rgb8 = None
        self._update_clipping_overlay()

    def _current_settings(self) -> pl.DevelopSettings:
        return self.develop_panel.settings

    def request_render(self) -> None:
        if self.session is None or self.frame is None:
            return
        settings = self._current_settings()
        tool = self.view.tool
        view = "negative" if (self.view_mode == "negative" or tool == "base") else "positive"
        self._requested_ratio = self.view.crop_aspect() if tool == "crop" else None
        self.develop.render(DevelopJob(
            kind="render", session=self.session, frame=self.frame, settings=settings,
            profile=self.library.get(settings.profile_id), view=view,
            ignore_crop=tool in ("base", "neutral", "crop", "straighten", "guided"), max_side=1800,
            crop_ratio=self._requested_ratio, show_dust=self._show_dust,
        ))

    def _before_settings(self) -> pl.DevelopSettings:
        """Lightroom's "before": the frame as imported (roll defaults with its film and
        base), in the current geometry, so the halves line up and only the look differs."""
        current = self._current_settings()
        roll = self.session.roll_defaults() if self.session else pl.DevelopSettings()
        before = roll.copy(profile_id=current.profile_id, separation=None, auto_crop=current.auto_crop,
                           lens_distortion=current.lens_distortion, lens_vignette=current.lens_vignette,
                           disabled=tuple(d for d in current.disabled if d in GEOMETRY_SECTIONS),
                           **{k: getattr(current, k) for k in GEOMETRY_FIELDS})
        return before

    def _request_before(self) -> None:
        if self.session is None or self.frame is None:
            return
        before = self._before_settings()
        key = (str(self.session.path), self.frame.id, before)
        if key == self._before_key:
            return  # unchanged: slider drags only touch the "after"
        self._before_key = key
        self.develop.render_before(DevelopJob(
            kind="before", session=self.session, frame=self.frame, settings=before,
            profile=self.library.get(before.profile_id), max_side=1800,
        ))

    def _on_rendered(self, result) -> None:
        if self.frame is None or result.job.frame.id != self.frame.id:
            return
        self._last_result = result
        self._last_rgb8 = result.rgb8
        self._last_dust = result.dust
        if result.darkfield_ok is False:
            self.develop_panel.set_dust_source(
                _("La toma de campo oscuro no coincide con este fotograma; el polvo se busca solo en la imagen."))
        self.view.set_image(result.image)
        self._last_analysis = result.analysis
        self.develop_panel.show_analysis(result.analysis)
        if result.histogram is not None and result.job.view == "positive":
            self.histogram.set_histogram(result.histogram)
            self.develop_panel.set_histogram(result.histogram)
        self._show_tone_values(result.job.settings)
        self.histogram.set_info(exif_summary(result.meta.get("exif") or {}))
        if self.view.tool == "crop" and result.job.ignore_crop:
            # Constrain to the picture, as Lightroom's crop tool does: after
            # straightening the frame never shows the empty corners.
            self.view.set_crop_bounds(result.valid)
            if self._crop_fresh:
                # Only when entering the tool: later renders (while rotating)
                # must keep the frame the user is dragging.
                self._crop_fresh = False
                ratio = self._crop_aspect_ratio(self.aspect_combo.currentData())
                if ratio is not None and not self._crop_matches(result, ratio):
                    # An existing crop of another shape (drawn freely, or the
                    # detected frame) is shown as it is, not forced to the ratio.
                    self.aspect_combo.blockSignals(True)
                    self.aspect_combo.setCurrentIndex(self.aspect_combo.findData("free"))
                    self.aspect_combo.blockSignals(False)
                    ratio = None
                self.view.set_crop_aspect(ratio)
                self.view.set_crop_rect(result.crop)
                self._crop_placed = tuple(self.view.crop_rect())
        if self.view.tool == "guided" and result.job.ignore_crop:
            self.view.set_guides(self._guides_on_screen(result))
        self._update_clipping_overlay()
        self._show_exposure_advice(result)
        self.navigator.set_image(result.image)
        if result.job.view == "positive" and not result.job.ignore_crop:
            for strip in (self.filmstrip, self.grid):
                strip.set_thumbnail(result.job.frame.id, result.image)
            if self.panel is not None and self.panel.isVisible() and result.job.frame.id == self._last_capture_id:
                self.panel.set_thumbnail(result.image)
        self._update_metadata()

    def _on_develop_failed(self, msg: str) -> None:
        """A frame that cannot be developed (e.g. its file was deleted outside Belka):
        say so once, not in a dialog on every render."""
        text = _("Error al revelar: {msg}").format(msg=msg)
        if "FileNotFoundError" in msg or "No such file" in msg:
            text = _("No se encuentra el archivo de este fotograma (¿se borró o movió?). "
                     "Quítalo del rollo con Supr.")
            self._clear_loupe()
        now = time.monotonic()
        if text == self._last_failure[0] and now - self._last_failure[1] < 30.0:
            self._message(text)
            return
        self._last_failure = (text, now)
        self._error(text)

    def _on_before(self, result) -> None:
        if self.frame is None or result.job.frame.id != self.frame.id or self._compare == "off":
            return
        self.view.set_compare(self._compare, result.image)

    def _show_exposure_advice(self, result) -> None:
        """Warn when the film base sits far below the sensor's range (or clips)."""
        note = ""
        analysis = result.analysis
        if analysis is not None and analysis.extra.get("no_light"):
            note = _("⚠ No se ve luz en esta foto: ¿estaba encendido el panel de luz?")
        elif analysis is not None and result.meta.get("source") == "raw" and analysis.channels == 3:
            stops = pl.exposure_advice(analysis)
            level = analysis.extra.get("base_level", 0.0)
            shot_at = (result.meta.get("exif") or {}).get("exposure")
            suggestion = None
            if stops >= 1.0 or stops < 0:
                suggestion = self.capture_panel.shutter_suggestion(stops, shot_at) if self.camera.connected else None
            if stops >= 1.0:
                note = _("⚠ Subexpuesta: la base está al {pct:.0%} del sensor. Expón ~{st:.1f} pasos más.").format(
                    pct=level, st=stops)
                if suggestion and suggestion[0] != suggestion[1]:
                    note += " " + _("Velocidad: {cur} → {new}.").format(cur=suggestion[0], new=suggestion[1])
                note += " " + _("Que la luz alrededor de la tira se sature no importa.")
            elif stops < 0:
                note = _("⚠ La base de la película está saturada: baja la exposición 1 paso.")
            if result.job.frame.id == self._last_capture_id:
                # Capture-time auto exposure: one click sets the suggested speed on the camera.
                apply = suggestion[1] if suggestion and suggestion[0] != suggestion[1] else None
                self.capture_panel.set_exposure_suggestion(note, "shutter" if apply else None, apply)
        if (self.session is not None and self.session.flats and result.meta
                and result.meta.get("flat_applied") is False):
            extra = _("Este fotograma no tiene flat-field: los del rollo son de otro modo de luz (blanca/tinte o RGB).")
            note = f"{note}\n{extra}" if note else extra
        self.develop_panel.set_exposure_note(note)
        # Once per capture, and never over a tool's hint.
        if (note and result.job.frame.id == self._last_capture_id and self._noted_capture != self._last_capture_id
                and self.view.tool == "none"):
            self._noted_capture = self._last_capture_id
            self._message(note)

    def _on_settings(self, settings: pl.DevelopSettings) -> None:
        self._save_timer.start()
        self._show_tone_values(settings)
        self.request_render()
        if self._compare != "off":
            self._request_before()

    def _apply(self, label: str | None = None, **changes) -> None:
        """Change settings from a tool or shortcut, render and (optionally) record a history step."""
        self.develop_panel.apply_external(**changes)
        if label:
            self.commit_history(label)

    def _save_frame_settings(self) -> None:
        self._save_timer.stop()
        if self.session is None or self.frame is None:
            return
        settings = self._current_settings()
        self.session.set_frame_settings(self.frame, settings)
        if self.session.lock_base and settings.base is not None:
            roll = self.session.roll_defaults()
            if roll.base != settings.base:
                self.session.roll_settings = roll.copy(base=settings.base).to_json()
                self.session.save()

    # ---------------------------------------------------------------- history
    def commit_history(self, label: str) -> None:
        if self.frame is None or self._applying_history:
            return
        self.history.get(self.frame.id).push(label, self._current_settings())
        self._refresh_history_panels()

    def _load_settings(self, settings: pl.DevelopSettings) -> None:
        """Every "load these settings" path (history, snapshots, paste, reset): the panel,
        a render, and a crop tool that is open follows the new crop."""
        if self.view.tool == "crop":
            self._crop_fresh = True
            self._crop_backup = settings.copy()
        self.develop_panel.load(settings, self.session.lock_base if self.session else False)
        self._on_settings(settings)

    def _load_from_history(self, settings: pl.DevelopSettings | None) -> None:
        if settings is None or self.frame is None:
            return
        self._applying_history = True
        try:
            self._load_settings(settings)
        finally:
            self._applying_history = False
        self._refresh_history_panels()

    def undo(self) -> None:
        if self.frame is not None:
            self._load_from_history(self.history.get(self.frame.id).undo())

    def redo(self) -> None:
        if self.frame is not None:
            self._load_from_history(self.history.get(self.frame.id).redo())

    def jump_history(self, index: int) -> None:
        if self.frame is not None:
            self._load_from_history(self.history.get(self.frame.id).jump(index))

    def clear_history(self) -> None:
        if self.frame is not None:
            self.history.get(self.frame.id).clear(_("Abrir"))
            self._refresh_history_panels()

    def _refresh_history_panels(self) -> None:
        if self.frame is None:
            return
        hist = self.history.get(self.frame.id)
        self.history_panel.set_entries([label for label, _s in hist.entries()], hist.index)
        self.snapshots_panel.set_snapshots([s.get("name", "") for s in self.frame.snapshots])
        self.profiles_panel.set_current(self._current_settings().profile_id)
        self.act_undo.setEnabled(hist.can_undo)
        self.act_redo.setEnabled(hist.can_redo)

    # ---------------------------------------------------------------- snapshots
    def create_snapshot_dialog(self) -> None:
        if self.frame is None:
            return
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, _("Nueva instantánea"), _("Nombre"),
                                        text=datetime.now().strftime("%d/%m %H:%M"))
        if ok and name.strip():
            self.create_snapshot(name.strip())

    def create_snapshot(self, name: str) -> None:
        if self.frame is None or self.session is None:
            return
        self.frame.snapshots.append({"name": name, "created": datetime.now().isoformat(timespec="seconds"),
                                     "settings": self._current_settings().to_json()})
        self.session.save()
        self._refresh_history_panels()

    def apply_snapshot(self, index: int) -> None:
        if self.frame is None or not (0 <= index < len(self.frame.snapshots)):
            return
        snap = self.frame.snapshots[index]
        settings = pl.DevelopSettings.from_json(snap["settings"])
        self._load_settings(settings)
        self.commit_history(_("Instantánea: {name}").format(name=snap.get("name", "")))

    def delete_snapshot(self, index: int) -> None:
        if self.frame is None or self.session is None or not (0 <= index < len(self.frame.snapshots)):
            return
        del self.frame.snapshots[index]
        self.session.save()
        self._refresh_history_panels()

    # ---------------------------------------------------------------- copy / paste / sync
    def copy_settings(self) -> None:
        if self.frame is None:
            return
        self._clipboard = self._current_settings().copy()
        self._message(_("Ajustes copiados (sin recorte, enderezado ni perspectiva)."))

    def _paste_onto(self, source: pl.DevelopSettings, target: pl.DevelopSettings) -> pl.DevelopSettings:
        """The look of ``source`` on ``target``'s own geometry (crop, straightening,
        perspective, base and the Transform/Óptica switches). The lens corrections
        travel: same lens on the same copy stand for the whole roll."""
        keep = {k: getattr(target, k) for k in GEOMETRY_FIELDS}
        disabled = tuple(d for d in source.disabled if d not in GEOMETRY_SECTIONS) + tuple(
            d for d in target.disabled if d in GEOMETRY_SECTIONS)
        return source.copy(**keep, auto_crop=target.auto_crop, disabled=disabled)

    def paste_settings(self) -> None:
        if self.frame is None or self._clipboard is None:
            return
        pasted = self._paste_onto(self._clipboard, self._current_settings())
        self._load_settings(pasted)
        self.commit_history(_("Pegar ajustes"))

    def paste_previous(self) -> None:
        if self.session is None or self.frame is None:
            return
        ids = [f.id for f in self.session.frames]
        idx = ids.index(self.frame.id)
        if idx == 0:
            return
        previous = self.session.settings_for(self.session.frames[idx - 1])
        pasted = self._paste_onto(previous, self._current_settings())
        self._load_settings(pasted)
        self.commit_history(_("Ajustes del anterior"))

    def sync_settings(self) -> None:
        if self.session is None or self.frame is None:
            return
        source = self._current_settings()
        targets = [f for f in self.session.frames if f.id in self._selected_ids() and f.id != self.frame.id]
        if not targets:
            self._message(_("Selecciona en la tira los fotogramas que quieres sincronizar."))
            return
        for frame in targets:
            before = self.session.settings_for(frame)
            # Seed the history first, or the sync would become its starting step.
            self.history.get(frame.id, before, _("Abrir"))
            frame.settings = self._paste_onto(source, before).to_json()
            self.history.get(frame.id).push(_("Sincronizar"), pl.DevelopSettings.from_json(frame.settings))
            self._queue_thumbnail(frame)
        self.session.save()
        self._message(_("Ajustes sincronizados con {n} fotogramas.").format(n=len(targets)))

    def reset_settings(self) -> None:
        """Back to the defaults, keeping what is not a look: the film, its measured base
        and the frame's geometry (crop with the straightening it was drawn in)."""
        if self.session is None or self.frame is None:
            return
        current = self._current_settings()
        fresh = pl.DevelopSettings(
            profile_id=current.profile_id, auto_crop=current.auto_crop,
            lens_distortion=current.lens_distortion, lens_vignette=current.lens_vignette,
            disabled=tuple(d for d in current.disabled if d in GEOMETRY_SECTIONS),
            **{k: getattr(current, k) for k in GEOMETRY_FIELDS},
        )
        self._load_settings(fresh)
        self.commit_history(_("Restablecer"))

    def apply_to_roll(self) -> None:
        if self.session is None or self.frame is None:
            return
        settings = self._current_settings()
        if QMessageBox.question(
            self, APP_NAME,
            _("¿Aplicar película y ajustes de este fotograma a los {n} fotogramas del rollo? (Se conservan recortes y rotaciones.)").format(n=len(self.session.frames)),
        ) != QMessageBox.StandardButton.Yes:
            return
        for frame in self.session.frames:
            before = self.session.settings_for(frame)
            # Seed first, or this step would become the frame's starting point.
            self.history.get(frame.id, before, _("Abrir"))
            after = self._paste_onto(settings, before)
            frame.settings = after.to_json()
            self.history.get(frame.id).push(_("Aplicar al rollo"), after)
        # New captures inherit the look, not this frame's straightening or base
        # (the base only when it is locked for the roll).
        roll = self.session.roll_defaults()
        self.session.roll_settings = self._paste_onto(settings, roll).copy(crop=None).to_json()
        self.session.film_profile = settings.profile_id
        self.session.save()
        for frame in self.session.frames:
            self._queue_thumbnail(frame)
        self._refresh_history_panels()
        self._message(_("Ajustes aplicados al rollo."))

    def _on_lock_base(self, locked: bool) -> None:
        if self.session is None:
            return
        self.session.lock_base = locked
        settings = self._current_settings()
        if locked:
            base = settings.base
            if base is None and self._last_analysis is not None and self._last_analysis.channels == 3:
                base = tuple(float(v) for v in self._last_analysis.base)
            if base is not None:
                self.session.roll_settings = self.session.roll_defaults().copy(base=base).to_json()
                self.develop_panel.apply_external(base=base)
            self._message(_("La base de este fotograma se usará en todo el rollo."))
        self.session.save()
        for frame in self.session.frames:
            self._queue_thumbnail(frame)

    def _on_profile_chosen(self, profile_id: str) -> None:
        if self.frame is not None:
            self._apply(_("Película: {name}").format(name=film_name(self.library.get(profile_id))),
                        profile_id=profile_id)

    def save_profile(self) -> None:
        if self.frame is None or self._last_analysis is None:
            self._message(_("Revela un fotograma antes de guardar un perfil."))
            return
        settings = self._current_settings()
        base = self.library.get(settings.profile_id)
        dialog = SaveProfileDialog(display_name(base), self)
        if dialog.exec() != SaveProfileDialog.DialogCode.Accepted:
            return
        name = dialog.name.text().strip() or display_name(base)
        profile_id = "user-" + slugify(name)
        if profile_id in self.library:
            if QMessageBox.question(
                self, APP_NAME,
                _("Ya existe el perfil «{name}». ¿Reemplazarlo? Los rollos que lo usan cambiarán.").format(
                    name=self.library.get(profile_id).name),
            ) != QMessageBox.StandardButton.Yes:
                return
        analysis = self._last_analysis
        gamma = base.gamma
        if analysis.channels == 3:
            lo, hi = pl.levels(analysis, base, settings.auto_balance)
            span = hi - lo
            gamma = tuple(round(float(base.gamma[1] * s / span[1]), 4) for s in span)
        profile = FilmProfile(
            id=profile_id, name=name, brand=_("Propio"), type=base.type, process=base.process,
            iso=base.iso, gamma=gamma, base_density=base.base_density,
            separation=round(base.separation if settings.separation is None else settings.separation, 3),
            paper_contrast=round(base.paper_contrast * settings.contrast, 3),
            saturation=round(base.saturation * settings.saturation, 3),
            temperature=round(base.temperature + settings.temperature, 3),
            bw_mix=base.bw_mix,
            notes=_("Creado desde «{roll}» #{frame} a partir de {film}.").format(
                roll=self.session.name if self.session else "", frame=self.frame.id, film=base.name),
            builtin=False,
        )
        self.library.save_user_profile(profile)
        self.develop_panel.reload_profiles()
        self.profiles_panel.reload()
        new = settings.copy(profile_id=profile.id, auto_balance=0.2, contrast=1.0, saturation=1.0,
                            temperature=0.0, separation=None)
        self._load_settings(new)
        self.commit_history(_("Perfil «{name}»").format(name=name))
        self._message(_("Perfil «{name}» guardado.").format(name=name))

    # ---------------------------------------------------------------- geometry
    def toggle_negative(self, on: bool) -> None:
        self.view_mode = "negative" if on else "positive"
        self.request_render()

    def _reoriented(self, apply) -> None:
        """Rotate or flip; with the crop tool open the frame being edited is applied
        first and the tool reopens on the turned picture (Esc and undo stay simple)."""
        if self.frame is None:
            return
        cropping = self.view.tool == "crop"
        if cropping:
            self.commit_crop()
        apply(self._current_settings())
        if cropping:
            self.start_tool("crop")

    def rotate(self, degrees: int) -> None:
        def apply(s: pl.DevelopSettings) -> None:
            from belka.core import transform

            changes = transform.reorient(s, "cw" if degrees > 0 else "ccw")
            self._apply(_("Rotar"), rotation=(s.rotation + degrees) % 360, **changes)

        self._reoriented(apply)

    def flip(self, axis: str) -> None:
        def apply(s: pl.DevelopSettings) -> None:
            from belka.core import transform

            # Flips act on the sensor image before the rotation; at 90/270 degrees
            # a mirror across the screen's vertical axis is the other flag.
            screen_horizontal = axis == "h"
            sensor_h = screen_horizontal if s.rotation % 180 == 0 else not screen_horizontal
            changes = transform.reorient(s, "flip_h" if screen_horizontal else "flip_v")
            flag = {"flip_h": not s.flip_h} if sensor_h else {"flip_v": not s.flip_v}
            self._apply(_("Voltear"), **flag, **changes)

        self._reoriented(apply)

    def request_upright(self, mode: str) -> None:
        if self.session is None or self.frame is None:
            return
        if mode == "off":
            self._apply(_("Upright: desactivado"), angle=0.0, persp_vertical=0.0, persp_horizontal=0.0,
                        upright_mode="", upright_guides=())
            return
        if mode == "guided":
            if self.view.tool == "guided":
                self.cancel_tool()
            else:
                self.start_tool("guided")
            return
        settings = self._current_settings()
        self.develop.request(DevelopJob(
            kind="upright", session=self.session, frame=self.frame, settings=settings,
            profile=self.library.get(settings.profile_id), mode=mode,
        ))
        self._message(_("Buscando líneas para enderezar…"))

    # ---------------------------------------------------------------- dust and edge print
    def _show_frame_extras(self, frame: Frame) -> None:
        """What the panel says about this frame's dark-field shot and film edge; reads the edge once."""
        self.develop_panel.set_dust_source(
            _("Con toma de campo oscuro") if frame.darkfield
            else _("Detección en la imagen (sin toma de campo oscuro)"))
        edge = frame.edge if isinstance(frame.edge, dict) else {}
        self.develop_panel.set_edge_info(edge if edge.get("film") or edge.get("frame") else None)
        if not edge and self.session is not None:
            settings = self.session.settings_for(frame)
            self.develop.request(DevelopJob(
                kind="edge", session=self.session, frame=frame, settings=settings,
                profile=self.library.get(settings.profile_id),
            ))

    def _on_edge_read(self, result) -> None:
        if self.session is None:
            return
        frame = self.session.frame(result.job.frame.id)
        if frame is None:
            return
        frame.edge = result.value or {"found": False}  # read once per frame
        self.session.save()
        for strip in (self.filmstrip, self.grid):
            strip.update_frame(frame)
        if self.frame is not None and frame.id == self.frame.id:
            self._show_frame_extras(frame)
            self._update_metadata()
            info = result.value or {}
            if info.get("film") and info.get("profile_id") and info["profile_id"] != self._current_settings().profile_id:
                self._message(_("En el borde de la película dice {film}: «Usar» en Perfil de película la aplica.").format(
                    film=info["film"]))

    def _on_dust_mask_toggled(self, on: bool) -> None:
        self._show_dust = bool(on)
        if not on:
            self._last_dust = None
            self._update_clipping_overlay()
        self.request_render()

    # ---------------------------------------------------------------- context menus
    def _marks_menu(self, menu: QMenu) -> None:
        rating = menu.addMenu(_("Calificación"))
        for act in self.rating_actions:
            rating.addAction(act)
        flag = menu.addMenu(_("Marca"))
        for act in (self.act_pick, self.act_reject, self.act_unflag):
            flag.addAction(act)

    def _delete_menu(self, menu: QMenu) -> None:
        menu.addAction(self.act_show_folder)
        menu.addSeparator()
        for act in (self.act_remove, self.act_trash, self.act_delete_rejected):
            menu.addAction(act)

    def _exec_menu(self, menu: QMenu, pos, sync: bool = False) -> None:
        """Show a context menu with the actions that cannot apply right now greyed out."""
        rejected = any(f.flag == -1 for f in (self.session.frames if self.session else []))
        states = {self.act_paste: self._clipboard is not None, self.act_delete_rejected: rejected,
                  self.act_sync: sync}
        for act, on in states.items():
            act.setEnabled(on)
        try:
            menu.exec(pos)
        finally:
            for act in states:
                act.setEnabled(True)

    def _photo_menu(self, global_pos, _x: float = -1.0, _y: float = -1.0) -> None:
        """Right-click on the photo: the develop tools and the frame's marks, as in Lightroom."""
        if self.frame is None or self.view.tool == "crop":
            return
        menu = QMenu(self)
        upright = QMenu(_("Upright"), menu)
        for mode, label in (("auto", _("Auto")), ("guided", _("Guiada…")), ("level", _("Nivel")),
                            ("vertical", _("Vertical")), ("full", _("Completo")), ("off", _("Desactivado"))):
            upright.addAction(label, lambda m=mode: self.request_upright(m))
        for act in (self.act_crop, self.act_straighten):
            menu.addAction(act)
        menu.addMenu(upright)
        transform = menu.addMenu(_("Rotar y voltear"))
        for act in (self.act_rot_ccw, self.act_rot_cw, self.act_flip_h, self.act_flip_v):
            transform.addAction(act)
        menu.addSeparator()
        for act in (self.act_auto_tone, self.act_wb, self.act_base):
            menu.addAction(act)
        settings = menu.addMenu(_("Ajustes"))
        for act in (self.act_copy, self.act_paste, self.act_previous, None, self.act_reset, self.act_snapshot):
            settings.addSeparator() if act is None else settings.addAction(act)
        view = menu.addMenu(_("Vista"))
        for act in (self.act_negative, self.act_compare, self.act_compare_side, self.act_clipping, self.act_grid,
                    self.act_zoom):
            view.addAction(act)
        menu.addSeparator()
        self._marks_menu(menu)
        menu.addAction(self.act_export)
        self._delete_menu(menu)
        self._exec_menu(menu, global_pos)

    def _strip_menu(self, strip, pos) -> None:
        """Right-click in the filmstrip or the grid acts on the selection; a frame
        outside it becomes the selection first, as in Lightroom."""
        frame_id = strip.item_id_at(pos)
        if frame_id is None or self.session is None:
            return
        if frame_id not in strip.selected_ids():
            strip.select_frame(frame_id)
            self.select_frame(frame_id)
        n = len(strip.selected_ids())
        menu = QMenu(self)
        open_develop = menu.addAction(_("Abrir en Revelado"), lambda: self.set_module("develop"))
        open_develop.setEnabled(self.module != "develop")
        menu.addSeparator()
        self._marks_menu(menu)
        menu.addSeparator()
        for act in (self.act_copy, self.act_paste, self.act_previous, self.act_sync):
            menu.addAction(act)
        menu.addAction(self.act_export)
        self._delete_menu(menu)
        self._exec_menu(menu, strip.viewport().mapToGlobal(pos), sync=n > 1)

    # ---------------------------------------------------------------- guided upright and grid
    def _guides_on_screen(self, result) -> list[tuple]:
        """The stored guides (source coordinates) on the rendered, transformed image."""
        from belka.core import transform

        s = result.job.settings
        w, h = result.image.width(), result.image.height()
        out = []
        for x0, y0, x1, y1 in s.upright_guides:
            (a, b), (c, d) = transform.map_points_from_source([(x0, y0), (x1, y1)], s, w, h)
            out.append((float(a), float(b), float(c), float(d)))
        return out

    def _on_guides_changed(self, guides: list) -> None:
        """Guides drawn on screen → source coordinates → the transform that makes them plumb or level."""
        from belka.core import transform

        res = self._last_result
        if self.frame is None or res is None or res.image is None:
            return
        shown = res.job.settings  # the image the guides were drawn on
        w, h = res.image.width(), res.image.height()
        source = []
        for x0, y0, x1, y1 in guides[:4]:
            (a, b), (c, d) = transform.map_points_to_source([(x0, y0), (x1, y1)], shown, w, h)
            source.append((float(a), float(b), float(c), float(d)))
        current = self._current_settings().copy(upright_guides=tuple(source))
        solved = transform.guided_upright(source, current, w, h) if source else {}
        if not source:
            solved = {"angle": 0.0, "persp_vertical": 0.0, "persp_horizontal": 0.0}
        # A switched-off Transform panel would hide the correction the guides ask for.
        disabled = tuple(d for d in current.disabled if d != "transform")
        self._apply(_("Upright guiado"), upright_guides=tuple(source), upright_mode="guided" if source else "",
                    persp_rotate=0.0, persp_aspect=0.0, disabled=disabled, **solved)

    def toggle_grid(self, on: bool) -> None:
        self._grid_on = bool(on)
        self.develop_panel.set_grid_checked(self._grid_on)
        self._update_grid()

    def _on_grid_toggled(self, on: bool) -> None:
        self.act_grid.blockSignals(True)
        self.act_grid.setChecked(on)
        self.act_grid.blockSignals(False)
        self._grid_on = bool(on)
        self._update_grid()

    def _on_transform_dragging(self, dragging: bool) -> None:
        # Lightroom shows the grid while a Transform slider is held.
        self._transform_dragging = dragging
        self._update_grid()

    def _update_grid(self) -> None:
        self.view.set_grid("grid" if self._grid_on or self._transform_dragging or self.view.tool == "guided" else "off")

    # ---------------------------------------------------------------- auto tone
    def auto_tone(self, fields: tuple | None = None) -> None:
        """Lightroom's Tone "Auto" (Ctrl+U), or one slider with Shift+double-click."""
        if self.session is None or self.frame is None:
            return
        settings = self._current_settings()
        if settings.output == "flat":
            # Contrast, highlights and shadows do nothing on the flat output.
            fields = tuple(f for f in (fields or TONE_FIELDS) if f not in FLAT_IGNORED_FIELDS)
            if not fields:
                return
        self.develop.request(DevelopJob(
            kind="auto_tone", session=self.session, frame=self.frame, settings=settings,
            profile=self.library.get(settings.profile_id), fields=fields,
        ))

    # ---------------------------------------------------------------- crop and straighten
    def toggle_crop(self, on: bool) -> None:
        if on:
            self.start_tool("crop")
        else:
            self.commit_crop()

    def _crop_aspect_ratio(self, key: str) -> float | None:
        if key == "free":
            return None
        if key == "original":
            # The crop tool shows the whole (uncropped) picture.
            res = self._last_result
            if res is not None and res.image is not None and res.job.ignore_crop:
                return res.image.width() / max(res.image.height(), 1)
            return 3 / 2
        w, h = key.split(":")
        return float(w) / float(h)

    @staticmethod
    def _crop_matches(result, ratio: float) -> bool:
        if result.crop is None or result.image is None:
            return True
        x0, y0, x1, y1 = result.crop
        shape = (x1 - x0) * result.image.width() / max((y1 - y0) * result.image.height(), 1e-6)
        # Either orientation: the frame can be swapped to portrait.
        return min(abs(shape / ratio - 1.0), abs(shape * ratio - 1.0)) < 0.01

    def _on_aspect(self, _index: int) -> None:
        key = self.aspect_combo.currentData()
        # The view first: the render this triggers sizes the bounds for the new ratio.
        self.view.set_crop_aspect(self._crop_aspect_ratio(key))
        if self.frame is not None:
            self.develop_panel.apply_external(crop_aspect=key)

    def _on_crop_changed(self, _rect: tuple) -> None:
        # X swaps the frame to portrait: its bounds depend on the locked ratio.
        if self.view.tool == "crop" and self.view.crop_aspect() != self._requested_ratio:
            self.request_render()

    def _on_rotate_started(self) -> None:
        self._rotate_from = self._current_settings().angle
        self._rotate_total = 0.0

    def _on_angle_delta(self, delta: float) -> None:
        # Accumulate from the start of the drag: clipping each step at ±45°
        # would let the pointer and the angle drift apart.
        if self._rotate_from is None:
            self._on_rotate_started()
        self._rotate_total += delta
        self.develop_panel.apply_external(angle=float(np.clip(self._rotate_from + self._rotate_total, -45.0, 45.0)))

    def _on_rotate_finished(self) -> None:
        self._rotate_from = None
        from belka.ui.widgets import format_value

        self.commit_history(_("Ángulo") + " " + format_value(self._current_settings().angle, 1, True, "°"))

    def _on_crop_committed(self, rect: tuple) -> None:
        rect = tuple(float(v) for v in rect)
        self._crop_backup = None
        self._finish_crop_ui()
        untouched = self._crop_placed is not None and max(abs(a - b) for a, b in zip(rect, self._crop_placed)) < 1e-4
        if untouched and self._current_settings().crop is None:
            return  # the automatic frame stays automatic (it follows later straightening)
        self._apply(_("Recortar"), crop=rect, auto_crop=False)

    def commit_crop(self) -> None:
        if self.view.tool == "crop" and self.view.has_image() and not self._crop_fresh:
            self.view.commit_crop()  # hands the frame over through cropCommitted
        else:
            # Before the uncropped render placed the frame the overlay holds a stale
            # one: leave the stored crop as it is.
            self._crop_backup = None
            self._finish_crop_ui()

    def cancel_crop(self) -> None:
        """Esc: undo only what the crop tool did (frame, angle, aspect); edits made
        in the panels meanwhile stay."""
        backup = self._crop_backup
        self._crop_backup = None
        if backup is not None and self.frame is not None:
            fields = ("crop", "auto_crop", "angle", "crop_aspect")
            restored = self._current_settings().copy(**{k: getattr(backup, k) for k in fields})
            self.develop_panel.load(restored, self.session.lock_base if self.session else False)
            self._on_settings(restored)
            self.commit_history(_("Cancelar recorte"))
        self._finish_crop_ui()

    def reset_crop(self) -> None:
        self.develop_panel.apply_external(crop=None, auto_crop=True, angle=0.0, crop_aspect="original")
        self.commit_history(_("Restablecer recorte"))
        self._finish_crop_ui()

    def _finish_crop_ui(self) -> None:
        self.act_crop.blockSignals(True)
        self.act_crop.setChecked(False)
        self.act_crop.blockSignals(False)
        self.crop_options.hide()
        if self.view.tool in ("crop", "straighten"):
            self.view.set_tool("none")
        self.statusBar().clearMessage()
        self.request_render()

    def _on_straighten_line(self, p0: QPointF, p1: QPointF) -> None:
        from belka.core import transform

        res = self._last_result
        w = res.image.width() if res and res.image else 1
        h = res.image.height() if res and res.image else 1
        delta = transform.angle_from_line((p0.x() * w, p0.y() * h), (p1.x() * w, p1.y() * h))
        s = self._current_settings()
        if self.view.tool != "crop":
            # Ctrl+drag in the crop tool straightens and stays in the tool, as in Lightroom.
            self.view.set_tool("none")
            self.statusBar().clearMessage()
        self._apply(_("Enderezar"), angle=float(np.clip(s.angle + delta, -45.0, 45.0)))

    # ================================================================ tools
    def start_tool(self, tool: str) -> None:
        if self.frame is None:
            return
        if self.module != "develop" and tool in ("crop", "straighten", "guided"):
            self.set_module("develop")
        if self.view.tool not in ("none", tool):
            self._close_frame_tools()
        if tool == "crop":
            self._crop_backup = self._current_settings().copy()
            self.crop_options.show()
            self.act_crop.blockSignals(True)
            self.act_crop.setChecked(True)
            self.act_crop.blockSignals(False)
            idx = self.aspect_combo.findData(self._current_settings().crop_aspect)
            self.aspect_combo.blockSignals(True)
            self.aspect_combo.setCurrentIndex(max(idx, 0))
            self.aspect_combo.blockSignals(False)
            # Aspect and frame are placed once the uncropped render arrives.
            self._crop_fresh = True
        self.view.set_tool(tool)
        self._update_grid()
        hints = {
            "base": _("Arrastra un rectángulo sobre película SIN exponer (borde o espacio entre fotogramas). Esc cancela."),
            "neutral": _("Haz clic o arrastra sobre algo que deba ser gris neutro. Esc cancela."),
            "crop": _("Arrastra las esquinas para recortar; arrastra fuera del recuadro para girar. Enter aplica, Esc cancela."),
            "straighten": _("Traza una línea sobre algo que deba quedar horizontal o vertical. Esc cancela."),
            "guided": _("Upright guiado: traza de 2 a 4 líneas sobre bordes que deban quedar verticales u "
                        "horizontales; arrastra sus extremos para afinar, clic derecho o Supr quita una. "
                        "Esc o Mayús+T terminan."),
        }
        self.statusBar().showMessage(hints.get(tool, ""))
        self.request_render()

    def cancel_tool(self) -> None:
        if self.view.tool == "crop":
            self.cancel_crop()
            return
        if self.view.tool != "none":
            self.view.set_tool("none")
            self.view.set_overlay(None)
            self._update_grid()
            self.statusBar().clearMessage()
            self.request_render()

    def _on_region(self, tool: str, rect: tuple) -> None:
        if self.session is None or self.frame is None:
            return
        settings = self._current_settings()
        self.view.set_tool("none")
        self.view.set_overlay(None)
        self.statusBar().clearMessage()
        if tool not in ("base", "neutral"):
            return
        kind = "sample_base" if tool == "base" else "neutral"
        self.develop.request(DevelopJob(
            kind=kind, session=self.session, frame=self.frame, settings=settings,
            profile=self.library.get(settings.profile_id), rect=rect,
        ))

    def _on_sampled(self, result) -> None:
        if result.job.kind == "edge":  # stored for any frame, current or not
            self._on_edge_read(result)
            return
        if self.frame is None or result.job.frame.id != self.frame.id:
            return
        if result.job.kind == "sample_base":
            self._apply(_("Medir base"), base=result.value)
            if self.session is not None and self.session.lock_base:
                self.session.roll_settings = self.session.roll_defaults().copy(base=result.value).to_json()
                self.session.save()
            self._message(_("Base medida: R {:.3f} · G {:.3f} · B {:.3f}").format(*result.value))
        elif result.job.kind == "neutral":
            self._apply(_("Balance de blancos"), neutral=result.value)
        elif result.job.kind == "upright":
            if not result.value:
                self._message(_("No hay suficientes líneas rectas para enderezar esta foto."))
                return
            # Upright's values are absolute and assume no manual rotate or aspect.
            names = {"auto": _("Auto"), "level": _("Nivel"), "vertical": _("Vertical"), "full": _("Completo")}
            disabled = tuple(d for d in self._current_settings().disabled if d != "transform")
            self._apply(_("Upright: {mode}").format(mode=names.get(result.job.mode, result.job.mode)),
                        persp_rotate=0.0, persp_aspect=0.0, upright_mode=result.job.mode, disabled=disabled,
                        **result.value)
        elif result.job.kind == "auto_tone":
            if self._last_analysis is not None and self._last_analysis.extra.get("no_light"):
                self._message(_("No hay luz en esta foto: no hay tono que ajustar."))
                return
            fields = result.job.fields
            self.develop_panel.apply_external(**result.value)
            if fields and len(fields) == 1 and fields[0] in self.develop_panel.rows:
                label = _("Automático: {name}").format(name=self.develop_panel.rows[fields[0]].history_label())
            else:
                label = _("Tono automático")
            self.commit_history(label)

    # ---------------------------------------------------------------- histogram, compare, clipping
    def _show_tone_values(self, settings: pl.DevelopSettings) -> None:
        self.histogram.set_values({k: float(getattr(settings, k)) for k in HIST_RANGES})
        # The flat output has no Sombras or Altas luces: their zones would drag nothing.
        self.histogram.set_inert_zones(FLAT_IGNORED_FIELDS if settings.output == "flat" else ())

    def _on_histogram_drag(self, field: str, delta: float) -> None:
        if self.frame is None or field not in HIST_RANGES:
            return
        lo, hi = HIST_RANGES[field]
        current = float(getattr(self._current_settings(), field))
        self.develop_panel.apply_external(**{field: float(np.clip(current + delta, lo, hi))})

    def _on_histogram_drag_finished(self, field: str) -> None:
        if field in HIST_RANGES:
            self.commit_history(zone_label(field, float(getattr(self._current_settings(), field))))

    def toggle_compare(self, mode: str) -> None:
        self._compare = "off" if self._compare == mode else mode
        self._before_key = None
        if self._compare == "off":
            self.view.set_compare("off", None)
        else:
            self._request_before()

    def toggle_clipping(self, on: bool) -> None:
        self._clip_shadows = self._clip_highlights = bool(on)
        self.histogram.set_clipping(self._clip_shadows, self._clip_highlights)
        self._update_clipping_overlay()

    def _on_clipping_toggled(self, which: str, on: bool) -> None:
        if which == "shadows":
            self._clip_shadows = on
        else:
            self._clip_highlights = on
        self.act_clipping.blockSignals(True)
        self.act_clipping.setChecked(self._clip_shadows or self._clip_highlights)
        self.act_clipping.blockSignals(False)
        self._update_clipping_overlay()

    def _update_clipping_overlay(self) -> None:
        rgb8 = self._last_rgb8
        # The negative view is normalised to its own peak: it has no tonal clipping to show.
        negative = self._last_result is not None and self._last_result.job.view == "negative"
        dust = self._last_dust if self._show_dust else None
        if rgb8 is None or negative or not (self._clip_shadows or self._clip_highlights or dust is not None):
            self.view.set_clipping_overlay(None)
            return
        from belka.core import adjust

        # The same rule as the histogram's triangles.
        shadows, highlights = adjust.clipping_masks(rgb8.astype(np.float32) / 255.0)
        overlay = np.zeros(rgb8.shape[:2] + (4,), dtype=np.uint8)
        if self._clip_shadows:
            overlay[shadows] = (40, 110, 255, 230)
        if self._clip_highlights:
            overlay[highlights] = (255, 50, 40, 230)
        if dust is not None and dust.shape == overlay.shape[:2]:
            overlay[dust] = (255, 40, 200, 210)  # what the dust removal repairs
        h, w = overlay.shape[:2]
        image = QImage(np.ascontiguousarray(overlay).data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        self.view.set_clipping_overlay(image)

    def _on_pixel_hovered(self, x: float, y: float) -> None:
        rgb8 = self._last_rgb8
        if rgb8 is None or x < 0 or y < 0:
            self.histogram.set_readout("")
            return
        h, w = rgb8.shape[:2]
        r, g, b = rgb8[min(int(y * h), h - 1), min(int(x * w), w - 1)]
        self.histogram.set_readout(f"R {r / 2.55:.0f} %   G {g / 2.55:.0f} %   B {b / 2.55:.0f} %")

    def _on_viewport(self, rect: QRectF) -> None:
        self.navigator.set_viewport(rect)

    def _on_zoom_changed(self, zoom: float) -> None:
        self.zoom_label.setText(f"{zoom * 100:.0f} %")
        if self.view.is_fit():
            self.navigator.set_zoom("fit")
        else:
            self.navigator.set_zoom({100: "100", 200: "200"}.get(round(zoom * 100)))

    def _on_navigator_zoom(self, key: str) -> None:
        if key == "fit":
            self.view.fit()
        else:
            self.view.zoom_to(2.0 if key == "200" else 1.0)

    # ================================================================ camera
    def _on_cameras(self, cameras: list[CameraInfo]) -> None:
        last = self.settings.get("last_camera", "")
        self.capture_panel.set_cameras(cameras, keep=last)
        if cameras:
            self._message(_("Cámara detectada: {model}").format(model=cameras[0].model))
            self.camera_label.setText(_("Cámara: {model} (sin conectar)").format(model=cameras[0].model)
                                      if not self.camera.connected else self.camera_label.text())
        else:
            self._message(_("No se ha detectado ninguna cámara."))
            if not self.camera.connected:
                self.camera_label.setText(_("Sin cámara"))

    def _connect_camera(self, info: CameraInfo) -> None:
        self.capture_panel.camera_status.setText(_("Conectando con {model}…").format(model=info.model))
        self.camera.request_open.emit(info)

    def _on_camera_opened(self, info: CameraInfo, summary: str) -> None:
        self.settings.set("last_camera", info.model)
        self.capture_panel.set_connected(True)
        self.camera_label.setText("📷 " + info.model)
        if self.session is not None and not self.session.camera:
            self.session.camera = info.model
            self.session.save()

    def _on_camera_busy(self, busy: bool) -> None:
        busy_text = _("Cámara ocupada…")
        if busy:
            self.statusBar().showMessage(busy_text)
        elif self.statusBar().currentMessage() == busy_text:
            # Only our own text: a capture's result message arrives just before.
            self.statusBar().clearMessage()

    def _on_camera_closed(self) -> None:
        self.capture_panel.set_connected(False)
        self.camera_label.setText(_("Sin cámara"))

    def _on_liveview_stopped(self) -> None:
        self.capture_panel.set_liveview_checked(False)

    def _on_live(self, data: bytes) -> None:
        if not self.capture_panel.invert_check.isChecked():
            now = time.monotonic()
            if now - self._last_live < 0.05:
                return
            self._last_live = now
            image = QImage.fromData(data)
            if not image.isNull():
                self.capture_panel.show_live(image)
            return
        settings = self._current_settings()
        self.live.submit(data, settings, self.library.get(settings.profile_id))

    def _on_live_image(self, image: QImage, note: str) -> None:
        if self.capture_panel.live_check.isChecked():
            self.capture_panel.show_live(image, note) if note else self.capture_panel.show_live(image)

    # ================================================================ capture
    def capture_frame(self) -> None:
        focus = QApplication.focusWidget()
        if focus is not None and focus.inherits("QLineEdit"):
            return
        if self.module != "capture" and not self._light_visible():
            # Space is Lightroom's zoom key everywhere else.
            self.view.toggle_zoom()
            return
        if not self._ensure_session():
            return
        self.flow.capture_frame()

    def capture_flat(self) -> None:
        if not self._ensure_session():
            return
        # Clicking the dock button means the user is in the main window: always
        # ask (the panel's own F, F path confirms inside the panel).
        if QMessageBox.information(
            self, _("Flat-field"),
            _("Retira la película del soporte (deja el difusor) y pulsa Aceptar.\n"
              "Usa la misma luz, enfoque y diafragma que para los negativos; si la imagen se satura, "
              "baja la exposición 1–2 pasos: solo importa la forma de la iluminación."),
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Ok:
            return
        self.flow.capture_flat()

    def clear_flats(self) -> None:
        if self.session is None:
            return
        self.session.clear_flats()
        self._on_flats_changed()

    def _on_flats_changed(self) -> None:
        if self.session is None:
            return
        self.capture_panel.set_flat_info(len(self.session.flats))
        self.request_render()
        for frame in self.session.frames:
            self._queue_thumbnail(frame)

    def _on_frame_captured(self, frame: Frame, session: Session) -> None:
        if session is not self.session:
            self._message(_("Fotograma {label} guardado en el rollo «{name}».").format(label=frame.label, name=session.name))
            return
        self._last_capture_id = frame.id
        self._noted_capture = None
        # A reshoot can reuse a deleted frame's id: start a clean history.
        self.history.discard(frame.id)
        self.grid.add_frame(frame, select=False)
        self.filmstrip.add_frame(frame, select=False)
        self._select_when_idle(frame.id)
        self._queue_thumbnail(frame)
        self._message(_("Fotograma {label} capturado.").format(label=frame.label))
        self._update_metadata()

    def _select_when_idle(self, frame_id: str) -> None:
        """Show a new capture, but not in the middle of a drag: the rest of the
        gesture would land on the new frame."""
        if QApplication.mouseButtons() != Qt.MouseButton.NoButton:
            QTimer.singleShot(100, lambda: self._select_when_idle(frame_id))
            return
        if self.session is not None and self.session.frame(frame_id) is not None:
            self.filmstrip.select_frame(frame_id)
            self.select_frame(frame_id)

    def _on_tint(self, tint: tuple) -> None:
        self.capture_panel.load_light(self.light)
        self._on_light_settings()
        self._message(_("Tinte calibrado: R {:.2f} · G {:.2f} · B {:.2f}. Repite para afinar.").format(*tint))

    # ================================================================ light
    def _light_visible(self) -> bool:
        return self.panel is not None and self.panel.isVisible()

    def _adapter(self):
        return next((a for a in self.adapters if a.id == self.light.adapter), None)

    def _screen_by_name(self, name: str):
        for screen in QGuiApplication.screens():
            if screen.name() == name:
                return screen
        return QGuiApplication.primaryScreen()

    def set_light_on(self, on: bool) -> None:
        self.act_light.blockSignals(True)
        self.act_light.setChecked(on)
        self.act_light.blockSignals(False)
        self.act_capture.setEnabled(self.module == "capture" or on)
        if not on:
            if self.flow.busy:
                self.flow.cancel(_("Se apagó el panel de luz durante la captura: captura cancelada."))
            if self.panel is not None:
                self.panel.hide()
            self.inhibitor.stop()
            return
        self._check_night_light()
        if self.panel is None:
            self.panel = LightPanel(self.light, self._adapter())
            self.panel.captureRequested.connect(self.flow.capture_frame)
            self.panel.flatRequested.connect(self.flow.capture_flat)
            self.panel.calibrateRequested.connect(self.flow.calibrate_tint)
            self.panel.closed.connect(lambda: self.capture_panel.light_btn.setChecked(False))
            self.panel.settingsChanged.connect(self._on_panel_settings)
        if self.panel.adapter != self._adapter():
            self.panel.set_adapter(self._adapter())
        self.panel.set_capturing(self.flow.busy)
        if not self.flow.busy:
            self.panel.set_status("")
            self.panel.set_display_color(self.light.display_rgb())
        self.panel.show_on(self._screen_by_name(self.light.screen))
        self.inhibitor.start()

    def _set_light_pattern(self, pattern: str) -> None:
        """The light panel's backlight, or the dark-field ring of the dust shot (called on every step)."""
        if self.panel is not None:
            self.panel.set_pattern(pattern)

    def _darkfield_stops(self) -> int:
        """How much slower the dust shot is than the frame; 0 when it is off."""
        cp = self.capture_panel
        return cp.darkfield_stops() if cp.darkfield_enabled() else 0

    def _panel_ready(self) -> bool:
        """Safe to expose: the panel is in front, or alone on its own screen."""
        if self.panel is None or not self.panel.isVisible():
            return False
        return self.panel.is_in_front() or self.panel.screen() is not self.screen()

    def _show_light_color(self, rgb: tuple) -> None:
        if self.panel is not None and self.panel.isVisible():
            self.panel.set_display_color(rgb)

    def _set_capturing(self, capturing: bool) -> None:
        if self.panel is None:
            return
        # Always, visible or not: a panel hidden mid-capture must not come back
        # stuck in capture mode (no HUD, guides or ruler).
        self.panel.set_capturing(capturing)
        if not capturing:
            self.panel.set_status("")
        elif self.panel.isVisible():
            # Wayland ignores raise_(); the flow waits until this activation
            # has actually brought the panel in front before exposing.
            self.panel.bring_to_front()
            self.panel.set_status(_("Capturando…"))

    def _on_light_settings(self) -> None:
        self.settings.data["light"] = self.light.to_json()
        self.settings.save()
        self.capture_panel.show_tint(self.light.tint)
        if self.panel is not None:
            if self.panel.adapter != self._adapter():
                self.panel.set_adapter(self._adapter())
            if not self.flow.busy:  # mid-capture the panel shows the step's colour
                self.panel.set_display_color(self.light.display_rgb())
            if self.panel.isVisible():
                target = self._screen_by_name(self.light.screen)
                if self.panel.screen() != target or self.panel.isFullScreen() != self.light.fullscreen:
                    self.panel.show_on(target)

    def _on_panel_settings(self) -> None:
        self.capture_panel.load_light(self.light)
        if self.panel is not None and not self.flow.busy:
            self.panel.set_display_color(self.light.display_rgb())
        self.settings.data["light"] = self.light.to_json()
        self.settings.save()

    def calibrate_scale(self) -> None:
        if not self._light_visible():
            self.capture_panel.light_btn.setChecked(True)
        if self.panel is not None:
            self.panel.show_ruler(True)

    def _check_night_light(self) -> None:
        if self._night_light_asked:
            return
        self._night_light_asked = True
        if night_light_enabled() and not night_light_paused():
            answer = QMessageBox.question(
                self, _("Luz nocturna"),
                _("La luz nocturna de GNOME está activada: vuelve más cálida la pantalla, que ahora es tu fuente "
                  "de luz, y cambia de color según la hora.\n\n¿Pausarla mientras Belka esté abierto? "
                  "Se reanuda al salir (y GNOME la reanuda solo mañana aunque algo falle)."),
            )
            if answer == QMessageBox.StandardButton.Yes:
                # A temporary pause in the daemon, never the saved setting.
                self._night_light_paused = pause_night_light()
            self._update_night_light_label()

    def _update_night_light_label(self) -> None:
        on = night_light_enabled() and not night_light_paused()
        self.night_label.setText(_("⚠ Luz nocturna activa") if on else "")

    def _on_screens_changed(self) -> None:
        self.capture_panel.refresh_screens()
        if self.panel is not None and self.panel.isVisible():
            names = [sc.name() for sc in QGuiApplication.screens()]
            if self.light.screen and self.light.screen not in names:
                self._message(_("La pantalla de la luz se desconectó; el panel pasó a la principal."))
            self.panel.show_on(self._screen_by_name(self.light.screen))
