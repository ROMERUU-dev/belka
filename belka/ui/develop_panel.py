"""Right-hand develop panel, in the manner of Lightroom Classic.

Sections, top to bottom: Perfil de película (Belka's own: the inversion),
Básico, Curva de tonos, HSL / Color, Gradación de color, Detalle, Óptica,
Transformar and Efectos, then Lightroom's "Anterior · Restablecer" bar.

Sliders show Lightroom units and store the model's (the ``Mapping``
constants below say how); every change emits ``settingsChanged`` and every
finished gesture ``editCommitted`` with a history label such as
"Exposición +0,35". Changes made from outside through :meth:`apply_external`
or :meth:`load` never emit ``editCommitted``: the caller names those steps.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from belka.core.film import FilmProfile, ProfileLibrary, film_name
from belka.core.pipeline import Analysis, DevelopSettings
from belka.i18n import _
from belka.ui.curve_editor import CHANNELS, ToneCurveEditor
from belka.ui.icons import icon
from belka.ui.sections import GROUND, PANEL_STYLE, Section, SectionList, SubHeading
from belka.ui.widgets import CheckBox, ColorSwatch, ColorWheel, GlyphButton, Mapping, Segmented, SliderRow, WheelGuard

DEFAULTS = DevelopSettings()

GROUPS = {
    "color_negative": "Negativo color",
    "bw_negative": "Blanco y negro",
    "slide": "Diapositiva (positivo)",
}
# The film list indents its films under the group titles. The indent is item
# padding (a styled delegate draws it), not spaces in the text, so the closed
# combo shows the name flush left like the other combos.
FILM_LIST_STYLE = """
QComboBox#filmCombo QAbstractItemView::item { padding: 3px 8px 3px 18px; color: #cfcfcf; }
QComboBox#filmCombo QAbstractItemView::item:disabled { padding-left: 8px; color: #7a7a7a; }
QComboBox#filmCombo QAbstractItemView::item:selected { background: #4a4a4a; color: #f0f0f0; }
"""

# ---- Display units -> model units (see DevelopSettings for the model ranges).
BIPOLAR = Mapping.linear((-100, 100), (-1.0, 1.0))
UNIT = Mapping.linear((0, 100), (0.0, 1.0))
WHITE_BALANCE = Mapping.linear((-100, 100), (-1.5, 1.5))
EXPOSURE = Mapping.identity(-5.0, 5.0)
# Contrast multiplies the paper gamma: halving and doubling are equally far
# from neutral, so the two halves of the slider are interpolated in log space.
CONTRAST = Mapping([(-100, 0.4), (0, 1.0), (100, 2.0)], log=True)
SATURATION = Mapping.linear((-100, 100), (0.0, 2.0))
WHITES = Mapping.linear((-100, 100), (-0.3, 0.3))
# Lightroom's Negros lift the blacks to the right; the model's ``black`` raises
# the black point (darker) when positive, so the scale runs backwards.
BLACKS = Mapping.linear((-100, 100), (0.3, -0.3))
SHARPEN = Mapping.linear((0, 150), (0.0, 1.5))
RADIUS = Mapping.identity(0.5, 3.0)
DEGREES = Mapping.identity(-10.0, 10.0)
ANGLE = Mapping.identity(-45.0, 45.0)
SCALE = Mapping.linear((50, 150), (0.5, 1.5))
HUE = Mapping.identity(0, 360)

# Básico's tone group: its "Auto" button and the sliders with an automatic value (Shift+double-click).
TONE_FIELDS = ("exposure", "contrast", "highlights", "shadows", "white", "black")
# The Transformar sliders; resetting them also forgets the Upright mode and guides that set them.
TRANSFORM_FIELDS = ("angle", "persp_vertical", "persp_horizontal", "persp_rotate", "persp_aspect", "persp_scale",
                    "persp_x", "persp_y", "upright_mode", "upright_guides")

# Fields each panel resets (double-click on its title).
SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "film": ("auto_balance", "separation"),
    "basic": ("temperature", "tint", "neutral", *TONE_FIELDS, "texture", "clarity", "vibrance", "saturation"),
    "curve": ("curve_highlights", "curve_lights", "curve_darks", "curve_shadows", "curve_splits",
              "curve_rgb", "curve_red", "curve_green", "curve_blue"),
    "hsl": ("hsl_hue", "hsl_sat", "hsl_lum"),
    "grading": ("grade_shadows", "grade_midtones", "grade_highlights", "grade_global", "grade_blending",
                "grade_balance"),
    "detail": ("sharpen_amount", "sharpen_radius", "sharpen_detail", "sharpen_masking", "nr_luma", "nr_color"),
    "lens": ("lens_distortion", "lens_vignette"),
    "transform": (*TRANSFORM_FIELDS, "constrain_crop"),
    "effects": ("vignette_amount", "vignette_midpoint", "vignette_roundness", "vignette_feather", "grain_amount",
                "grain_size", "grain_roughness"),
}
SECTION_TITLES = {
    "film": "Perfil de película", "basic": "Básico", "curve": "Curva de tonos", "hsl": "HSL / Color",
    "grading": "Gradación de color", "detail": "Detalle", "lens": "Óptica", "transform": "Transformar",
    "effects": "Efectos",
}
# Panels with an on/off switch; the key is what goes in ``DevelopSettings.disabled``.
SWITCHED = ("curve", "hsl", "grading", "detail", "lens", "transform", "effects")
EXPANDED_AT_START = ("film", "basic")
PARAMETRIC = ("curve_highlights", "curve_lights", "curve_darks", "curve_shadows", "curve_splits")

HSL_BANDS = ("Rojo", "Naranja", "Amarillo", "Verde", "Aguamarina", "Azul", "Púrpura", "Magenta")
BAND_HUES = (0, 30, 55, 120, 180, 220, 275, 315)  # display hue of each band, degrees
# "Matiz", as in Lightroom's Spanish UI: "Tono" is Básico's tone group (and translates as "Tone").
HSL_TABS = (("hue", "Matiz", "hsl_hue"), ("sat", "Saturación", "hsl_sat"), ("lum", "Luminancia", "hsl_lum"))
GRADE_REGIONS = (
    ("shadows", "Sombras", "dot", "#2c2c2c"),
    ("midtones", "Medios tonos", "half", "#808080"),
    ("highlights", "Luces altas", "dot", "#d8d8d8"),
    ("global", "Global", "ring", "#a6a6a6"),
)
UPRIGHT = (("off", "Desactivado"), ("auto", "Auto"), ("guided", "Guiada"), ("level", "Nivel"),
           ("vertical", "Vertical"), ("full", "Completo"))


def _hsv(hue: float, sat: float = 0.6, val: float = 0.78) -> QColor:
    return QColor.fromHsvF((hue % 360.0) / 360.0, sat, val)


NEUTRAL_TRACK = QColor("#9a9a9a")
TEMP_TRACK = [QColor("#4a7fd4"), NEUTRAL_TRACK, QColor("#dcc04c")]
TINT_TRACK = [QColor("#4fae58"), NEUTRAL_TRACK, QColor("#c757c2")]
HUE_TRACK = [_hsv(h, 0.65, 0.8) for h in range(0, 361, 30)]
LUM_TRACK = [QColor("#151515"), QColor("#e6e6e6")]


def _band_tracks(i: int) -> dict[str, list[QColor]]:
    h = BAND_HUES[i]
    before, after = BAND_HUES[i - 1], BAND_HUES[(i + 1) % len(BAND_HUES)]
    return {
        "hsl_hue": [_hsv(before), _hsv(h), _hsv(after)],
        "hsl_sat": [_hsv(h, 0.0, 0.55), _hsv(h, 0.75, 0.82)],
        "hsl_lum": [_hsv(h, 0.75, 0.22), _hsv(h, 0.45, 0.96)],
    }


class DevelopPanel(QScrollArea):
    settingsChanged = Signal(object)  # DevelopSettings, on every change (drags included)
    editCommitted = Signal(str)  # history label when a gesture ends ("Exposición +0,35")
    toolRequested = Signal(str)  # "base" | "neutral"
    uprightRequested = Signal(str)  # "off" | "auto" | "guided" | "level" | "vertical" | "full"
    autoToneRequested = Signal()  # "Auto" beside Tono
    autoFieldRequested = Signal(str)  # Shift+double-click on a TONE_FIELDS row
    gridToggled = Signal(bool)  # "Mostrar cuadrícula" (view state, not a setting)
    transformDragging = Signal(bool)  # a Transformar or Óptica slider is held (True) or let go
    lockBaseChanged = Signal(bool)
    applyAllRequested = Signal()
    saveProfileRequested = Signal()
    resetRequested = Signal()
    previousRequested = Signal()

    BAR_HEIGHT = 40
    # Qt filters the scrollbar's events through eventFilter() while the base class is being built.
    _scrollbar_box: QWidget | None = None

    def __init__(self, library: ProfileLibrary, parent: QWidget | None = None):
        super().__init__(parent)
        self._scrollbar_box = self.verticalScrollBar().parentWidget()
        self.library = library
        self._settings = DevelopSettings()
        self._analysis: Analysis | None = None
        self.rows: dict[str, SliderRow] = {}
        self._bindings: dict[str, tuple[str, int | None]] = {}
        self._grade_start: tuple | None = None
        self.setObjectName("developPanel")
        self.setStyleSheet(PANEL_STYLE + FILM_LIST_STYLE
                           + f"QScrollArea#developPanel {{ background: {GROUND}; border: none; }}")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(300)

        self.sections = SectionList()
        for key in SECTION_TITLES:
            section = self.sections.add(key, Section(_(SECTION_TITLES[key]), switchable=key in SWITCHED))
            section.resetRequested.connect(lambda k=key: self.reset_section(k))
            if key in SWITCHED:
                section.toggled.connect(lambda on, k=key: self._on_switch(k, on))
        self.sections.add_stretch()
        self.setWidget(self.sections)

        self._build_film(self.sections.section("film"))
        self._build_basic(self.sections.section("basic"))
        self._build_curve(self.sections.section("curve"))
        self._build_hsl(self.sections.section("hsl"))
        self._build_grading(self.sections.section("grading"))
        self._build_detail(self.sections.section("detail"))
        self._build_lens(self.sections.section("lens"))
        self._build_transform(self.sections.section("transform"))
        self._build_effects(self.sections.section("effects"))
        self._build_bottom_bar()
        for key in SECTION_TITLES:
            self._align_labels(self.sections.section(key))
            self.sections.section(key).set_expanded(key in EXPANDED_AT_START)

        self.reload_profiles()
        # Long entries must not force the dock wider than the screen allows.
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        WheelGuard(self).guard_children(self.widget())
        # Clicking a button, a switch, a list or the bare panel (a scroll area
        # takes click focus) must leave the keyboard with the photo: the arrows
        # and 0-5 step through frames and rate them. An open list still has
        # the keyboard, and on closing hands it back.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for widget in [*self.findChildren(QAbstractButton), *self.findChildren(QComboBox)]:
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._sync()
        self.setEnabled(False)

    # ================================================================ building
    @staticmethod
    def _place(target: Section | QWidget, widget: QWidget) -> None:
        """Into a panel, or into one of its switchable groups (HSL tab, grading region)."""
        if isinstance(target, Section):
            target.add(widget)
        else:
            target.layout().addWidget(widget)

    def _slider(self, target: Section | QWidget, field: str, label: str, mapping: Mapping, *,
                index: int | None = None, step: float = 1.0, decimals: int = 0, gradient: list[QColor] | None = None,
                suffix: str = "", tooltip: str = "", history: str = "", auto: bool = False) -> SliderRow:
        value = getattr(DEFAULTS, field)
        row = SliderRow(label, mapping, value if index is None else value[index], step=step, decimals=decimals,
                        gradient=gradient, suffix=suffix, tooltip=tooltip, history=history, auto=auto)
        key = field if index is None else f"{field}.{index}"
        self.rows[key] = row
        self._bindings[key] = (field, index)
        row.valueChanged.connect(lambda v, f=field, i=index: self._on_row(f, i, v))
        row.editFinished.connect(self.editCommitted)
        if auto:
            row.autoRequested.connect(lambda f=field: self.autoFieldRequested.emit(f))
        self._place(target, row)
        return row

    def _heading(self, target: Section | QWidget, text: str, fields: tuple[str, ...] = ()) -> SubHeading:
        """A group title; with ``fields``, double-clicking it resets them."""
        heading = SubHeading(text)
        if fields:
            heading.label.setToolTip(_("Doble clic: restablecer {group}").format(group=text))
            heading.doubleClicked.connect(lambda: self._reset_fields(fields, _("Restablecer {group}").format(group=text)))
        self._place(target, heading)
        return heading

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    @staticmethod
    def _icon_button(name: str, text: str, tooltip: str, beside: bool = False) -> QToolButton:
        """A small tool button with its line icon; the text stands in until the icon exists."""
        button = QToolButton()
        button.setObjectName("iconButton")
        button.setText(text)
        button.setToolTip(tooltip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        glyph = icon(name)
        if glyph.isNull():
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        else:
            button.setIcon(glyph)
            button.setIconSize(QSize(16, 16))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon if beside
                                      else Qt.ToolButtonStyle.ToolButtonIconOnly)
        return button

    @staticmethod
    def _group() -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        return widget

    def _build_film(self, sec: Section) -> None:
        self.profile_combo = QComboBox()
        self.profile_combo.setObjectName("filmCombo")
        # Item stylesheets (FILM_LIST_STYLE) only reach a styled delegate.
        self.profile_combo.setItemDelegate(QStyledItemDelegate(self.profile_combo))
        self.profile_combo.setMaxVisibleItems(24)
        self.profile_combo.setToolTip(_("Perfil de la película: densidades, contraste y color de la emulsión"))
        sec.add(self._form_row(_("Película"), self.profile_combo))
        self.profile_notes = QLabel()
        self.profile_notes.setObjectName("note")
        self.profile_notes.setWordWrap(True)
        self.profile_notes.setIndent(2)
        sec.add(self.profile_notes)
        self.output_combo = QComboBox()
        self.output_combo.addItem(_("Impresión (curva de papel)"), "print")
        self.output_combo.addItem(_("Plana lineal (para editar)"), "flat")
        sec.add(self._form_row(_("Salida"), self.output_combo))
        self.auto_crop = CheckBox(_("Encuadre automático del fotograma"))
        self.auto_crop.setToolTip(_("Recorta solo la imagen, sin perforaciones, bordes ni luz alrededor."))
        sec.add(self.auto_crop)

        self._heading(sec, _("Base de la película"))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.base_swatch = ColorSwatch()
        self.base_label = QLabel(_("Automática"))
        self.base_pick = self._icon_button(
            "eyedropper_base", _("Medir"),
            _("Medir la base: arrastra un rectángulo sobre película sin exponer (entre fotogramas o junto a las perforaciones)."),
            beside=True)
        self.base_auto = QPushButton(_("Auto"))
        self.base_auto.setToolTip(_("Estimar la base automáticamente"))
        row.addWidget(self.base_swatch)
        row.addWidget(self.base_label, 1)
        row.addWidget(self.base_pick)
        row.addWidget(self.base_auto)
        sec.add(row)
        self.base_lock = CheckBox(_("Usar esta base en todo el rollo"))
        sec.add(self.base_lock)
        self.exposure_note = QLabel()
        self.exposure_note.setObjectName("warning")
        self.exposure_note.setWordWrap(True)
        self.exposure_note.hide()
        sec.add(self.exposure_note)

        self._heading(sec, _("Inversión"), ("auto_balance", "separation"))
        self._slider(sec, "auto_balance", _("Auto-balance"), UNIT, tooltip=_(
            "0 = curvas del perfil (consistente en todo el rollo)\n100 = niveles automáticos por canal (cada foto)"))
        self.separation = SliderRow(_("Separación"), UNIT, 0.35,
                                    tooltip=_("Deshace la mezcla entre canales de los tintes de la película"))
        self.separation.valueChanged.connect(self._on_separation)
        self.separation.editFinished.connect(self.editCommitted)
        sec.add(self.separation)
        self.separation_profile = CheckBox(_("Separación del perfil"))
        sec.add(self.separation_profile)

        sec.add_spacing(6)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.save_profile_btn = QPushButton(_("Guardar perfil…"))
        self.save_profile_btn.setToolTip(_("Guardar como perfil de película propio"))
        self.apply_all_btn = QPushButton(_("Aplicar al rollo…"))
        self.apply_all_btn.setToolTip(_("Copia estos ajustes (salvo recorte y rotación) a todos los fotogramas"))
        row.addWidget(self.save_profile_btn)
        row.addWidget(self.apply_all_btn)
        sec.add(row)

        self.profile_combo.activated.connect(self._on_profile)
        self.output_combo.activated.connect(self._on_output)
        self.auto_crop.clicked.connect(self._on_auto_crop)
        self.base_pick.clicked.connect(lambda: self.toolRequested.emit("base"))
        self.base_auto.clicked.connect(lambda: self._change(_("Base automática"), base=None))
        self.base_lock.clicked.connect(self._on_lock_base)
        self.separation_profile.clicked.connect(self._on_separation_profile)
        self.save_profile_btn.clicked.connect(self.saveProfileRequested)
        self.apply_all_btn.clicked.connect(self.applyAllRequested)

    def _form_row(self, text: str, widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._field_label(text))
        row.addWidget(widget, 1)
        return row

    def _build_basic(self, sec: Section) -> None:
        wb = self._heading(sec, _("Balance de blancos"), ("temperature", "tint", "neutral"))
        self.neutral_state = QLabel()
        self.neutral_state.setObjectName("note")
        self.neutral_clear = QToolButton()
        self.neutral_clear.setObjectName("iconButton")
        self.neutral_clear.setText(_("Quitar"))
        self.neutral_clear.setToolTip(_("Quitar la corrección de gris neutro"))
        self.neutral_pick = self._icon_button(
            "eyedropper_wb", _("Gris neutro"),
            _("Selector de balance de blancos (W): haz clic o arrastra sobre algo que deba ser gris"))
        wb.add_widget(self.neutral_state)
        wb.add_widget(self.neutral_clear)
        wb.add_widget(self.neutral_pick)
        self._slider(sec, "temperature", _("Temperatura"), WHITE_BALANCE, gradient=TEMP_TRACK)
        self._slider(sec, "tint", _("Tinte"), WHITE_BALANCE, gradient=TINT_TRACK)

        tone = self._heading(sec, _("Tono"), TONE_FIELDS)
        self.auto_tone = QToolButton()
        self.auto_tone.setObjectName("textButton")
        self.auto_tone.setText(_("Auto"))
        self.auto_tone.setToolTip(_("Ajusta exposición, contraste, altas luces, sombras, blancos y negros "
                                    "para esta foto\nMayús+doble clic en un deslizador: solo ese"))
        self.auto_tone.setCursor(Qt.CursorShape.PointingHandCursor)
        tone.add_widget(self.auto_tone)
        self._slider(sec, "exposure", _("Exposición"), EXPOSURE, step=0.01, decimals=2, auto=True)
        self._slider(sec, "contrast", _("Contraste"), CONTRAST, auto=True)
        self._slider(sec, "highlights", _("Altas luces"), BIPOLAR, auto=True)
        self._slider(sec, "shadows", _("Sombras"), BIPOLAR, auto=True)
        self._slider(sec, "white", _("Blancos"), WHITES, auto=True)
        self._slider(sec, "black", _("Negros"), BLACKS, auto=True)

        self._heading(sec, _("Presencia"), ("texture", "clarity", "vibrance", "saturation"))
        self._slider(sec, "texture", _("Textura"), BIPOLAR)
        self._slider(sec, "clarity", _("Claridad"), BIPOLAR)
        self._slider(sec, "vibrance", _("Intensidad"), BIPOLAR)
        self._slider(sec, "saturation", _("Saturación"), SATURATION)

        self.neutral_pick.clicked.connect(lambda: self.toolRequested.emit("neutral"))
        self.neutral_clear.clicked.connect(lambda: self._change(_("Quitar gris neutro"), neutral=(0.0, 0.0, 0.0)))
        self.auto_tone.clicked.connect(self.autoToneRequested)

    def _build_curve(self, sec: Section) -> None:
        self.curve_editor = ToneCurveEditor()
        sec.add(self.curve_editor)
        self.curve_editor.curveChanged.connect(lambda ch, pts: self._change(None, **{f"curve_{ch}": tuple(pts)}))
        self.curve_editor.splitsChanged.connect(self._on_splits)
        self.curve_editor.editFinished.connect(self.editCommitted)
        self._heading(sec, _("Región"), ("curve_highlights", "curve_lights", "curve_darks", "curve_shadows"))
        for field, label in (("curve_highlights", _("Altas luces")), ("curve_lights", _("Claros")),
                             ("curve_darks", _("Oscuros")), ("curve_shadows", _("Sombras"))):
            self._slider(sec, field, label, BIPOLAR, history=_("Curva: {name}").format(name=label))

    def _build_hsl(self, sec: Section) -> None:
        self.hsl_tabs = Segmented([(key, _(text)) for key, text, _field in HSL_TABS] + [("all", _("Todo"))])
        sec.add(self.hsl_tabs)
        self.hsl_groups: dict[str, tuple[QWidget, SubHeading]] = {}
        for key, text, field in HSL_TABS:
            group = self._group()
            heading = self._heading(group, _(text), (field,))
            for i, band in enumerate(HSL_BANDS):
                name = _(band)
                self._slider(group, field, name, BIPOLAR, index=i, gradient=_band_tracks(i)[field],
                             history=f"{_(text)}: {name}")
            sec.add(group)
            self.hsl_groups[key] = (group, heading)
        self.hsl_tabs.changed.connect(self._show_hsl_tab)
        self._show_hsl_tab("hue")

    def _show_hsl_tab(self, tab: str) -> None:
        self.hsl_tabs.set_current(tab)
        for key, (group, heading) in self.hsl_groups.items():
            group.setVisible(tab in (key, "all"))
            heading.setVisible(tab == "all")

    def _build_grading(self, sec: Section) -> None:
        self.grade_tabs = Segmented([
            (key, GlyphButton(glyph, QColor(color), _(name))) for key, name, glyph, color in GRADE_REGIONS
        ])
        sec.add(self.grade_tabs)
        self.grade_title = QLabel()
        self.grade_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grade_title.setObjectName("subHeading")
        sec.add(self.grade_title)
        self.grade_wheel = ColorWheel()
        sec.add(self.grade_wheel)
        self.grade_groups: dict[str, QWidget] = {}
        for key, name, _glyph, _color in GRADE_REGIONS:
            group = self._group()
            field = f"grade_{key}"
            region = _(name)
            self._slider(group, field, _("Matiz"), HUE, index=0, gradient=HUE_TRACK,
                         history=f"{region}: {_('Matiz')}")
            self._slider(group, field, _("Saturación"), UNIT, index=1, history=f"{region}: {_('Saturación')}")
            self._slider(group, field, _("Luminancia"), BIPOLAR, index=2, gradient=LUM_TRACK,
                         history=f"{region}: {_('Luminancia')}")
            sec.add(group)
            self.grade_groups[key] = group
        sec.add_spacing(4)
        self._slider(sec, "grade_blending", _("Fusión"), UNIT, history=_("Gradación: fusión"))
        self._slider(sec, "grade_balance", _("Equilibrio"), BIPOLAR, history=_("Gradación: equilibrio"))
        self.grade_tabs.changed.connect(self._show_grade_region)
        self.grade_wheel.pressed.connect(self._on_wheel_pressed)
        self.grade_wheel.changed.connect(self._on_wheel)
        self.grade_wheel.released.connect(self._on_wheel_released)
        self.grade_wheel.resetRequested.connect(self._on_wheel_reset)
        self._show_grade_region("shadows")

    def _show_grade_region(self, region: str) -> None:
        self.grade_tabs.set_current(region)
        for key, group in self.grade_groups.items():
            group.setVisible(key == region)
        self.grade_title.setText(next(_(name) for key, name, _g, _c in GRADE_REGIONS if key == region))
        self._sync_wheel()

    def _build_detail(self, sec: Section) -> None:
        sharpen = _("Enfoque")
        self._heading(sec, sharpen, ("sharpen_amount", "sharpen_radius", "sharpen_detail", "sharpen_masking"))
        self._slider(sec, "sharpen_amount", _("Cantidad"), SHARPEN, history=f"{sharpen}: {_('Cantidad')}")
        self._slider(sec, "sharpen_radius", _("Radio"), RADIUS, step=0.1, decimals=1,
                     history=f"{sharpen}: {_('Radio')}", tooltip=_("En píxeles de la imagen a resolución completa"))
        self._slider(sec, "sharpen_detail", _("Detalle"), UNIT, history=f"{sharpen}: {_('Detalle')}")
        self._slider(sec, "sharpen_masking", _("Máscara"), UNIT, history=f"{sharpen}: {_('Máscara')}",
                     tooltip=_("Limita el enfoque a los bordes; las zonas lisas quedan sin enfocar"))
        noise = _("Reducción de ruido")
        self._heading(sec, noise, ("nr_luma", "nr_color"))
        self._slider(sec, "nr_luma", _("Luminancia"), UNIT, history=f"{noise}: {_('Luminancia')}")
        self._slider(sec, "nr_color", _("Color"), UNIT, history=f"{noise}: {_('Color')}")

    def _build_lens(self, sec: Section) -> None:
        self._slider(sec, "lens_distortion", _("Distorsión"), BIPOLAR,
                     tooltip=_("Positivo corrige el barril; negativo, el cojín"))
        self._slider(sec, "lens_vignette", _("Viñeteo de lente"), BIPOLAR,
                     tooltip=_("Positivo aclara las esquinas oscurecidas por el objetivo o la luz"))
        self._report_drags(sec)

    def _report_drags(self, sec: Section) -> None:
        """While one of these sliders is held the window shows a grid, as Lightroom does."""
        for row in sec.body.findChildren(SliderRow):
            row.dragging.connect(self.transformDragging)

    def _build_transform(self, sec: Section) -> None:
        self._heading(sec, "Upright")
        # Two rows of three: six Spanish labels in one row would not fit the narrowest dock.
        grid = QGridLayout()
        grid.setSpacing(0)
        tips = {
            "off": _("Quita la corrección de perspectiva"),
            "auto": _("Nivela y corrige la perspectiva de forma equilibrada"),
            "guided": _("Dibuja de 2 a 4 líneas sobre bordes que deban quedar verticales u horizontales "
                        "(Mayús+T)"),
            "level": _("Solo nivela el horizonte"),
            "vertical": _("Nivela y endereza las verticales"),
            "full": _("Endereza verticales y horizontales; si el objetivo no coincide con el de referencia, "
                      "las horizontales pueden quedar ligeramente inclinadas"),
        }
        # Exclusive, but the panel never latches a button itself: it lights
        # the mode in ``upright_mode`` once the window has applied it.
        self.upright_group = QButtonGroup(self)
        self.upright_buttons: dict[str, QToolButton] = {}
        for i, (mode, text) in enumerate(UPRIGHT):
            button = QToolButton()
            button.setObjectName("uprightButton")
            button.setText(_(text))
            button.setToolTip(tips[mode])
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _c=False, m=mode: self._on_upright(m))
            self.upright_group.addButton(button)
            grid.addWidget(button, i // 3, i % 3)
            self.upright_buttons[mode] = button
        for column in range(3):
            grid.setColumnStretch(column, 1)
        sec.add(grid)
        sec.add_spacing(4)
        checks = QHBoxLayout()
        checks.setSpacing(12)
        self.constrain_crop = CheckBox(_("Restringir recorte"))
        self.constrain_crop.setToolTip(_("Recorta lo justo para que no queden bordes vacíos tras la corrección"))
        self.show_grid = CheckBox(_("Mostrar cuadrícula"))
        self.show_grid.setToolTip(_("Cuadrícula fina sobre la foto para alinear verticales y horizontales; "
                                    "aparece sola mientras arrastras un deslizador de este panel"))
        checks.addWidget(self.constrain_crop)
        checks.addWidget(self.show_grid)
        checks.addStretch(1)
        sec.add(checks)
        self._heading(sec, _("Transformar"), TRANSFORM_FIELDS)
        self._slider(sec, "angle", _("Ángulo"), ANGLE, step=0.1, decimals=1, suffix="°", tooltip=_(
            "Enderezar: el mismo ángulo que fijan la herramienta Recortar (R) y la línea de enderezar (S)"))
        self._slider(sec, "persp_vertical", _("Vertical"), BIPOLAR, history=_("Perspectiva vertical"))
        self._slider(sec, "persp_horizontal", _("Horizontal"), BIPOLAR, history=_("Perspectiva horizontal"))
        self._slider(sec, "persp_rotate", _("Rotar"), DEGREES, step=0.1, decimals=1, suffix="°")
        self._slider(sec, "persp_aspect", _("Aspecto"), BIPOLAR)
        self._slider(sec, "persp_scale", _("Escala"), SCALE)
        self._slider(sec, "persp_x", _("Desplazamiento X"), BIPOLAR)
        self._slider(sec, "persp_y", _("Desplazamiento Y"), BIPOLAR)
        self._report_drags(sec)
        self.constrain_crop.clicked.connect(lambda on: self._change(_("Restringir recorte"), constrain_crop=on))
        self.show_grid.clicked.connect(self.gridToggled)

    def _build_effects(self, sec: Section) -> None:
        vignette = _("Viñeta post-recorte")
        self._heading(sec, vignette, SECTION_FIELDS["effects"][:4])
        for field, label, mapping in (("vignette_amount", _("Cantidad"), BIPOLAR),
                                      ("vignette_midpoint", _("Punto medio"), UNIT),
                                      ("vignette_roundness", _("Redondez"), BIPOLAR),
                                      ("vignette_feather", _("Difuminar"), UNIT)):
            self._slider(sec, field, label, mapping, history=f"{_('Viñeta')}: {label}")
        grain = _("Grano")
        self._heading(sec, grain, SECTION_FIELDS["effects"][4:])
        for field, label in (("grain_amount", _("Cantidad")), ("grain_size", _("Tamaño")),
                             ("grain_roughness", _("Aspereza"))):
            self._slider(sec, field, label, UNIT, history=f"{grain}: {label}")

    def _build_bottom_bar(self) -> None:
        """Lightroom's "Anterior · Restablecer", fixed under the scrolling panels."""
        self.bottom_bar = QWidget(self)
        self.bottom_bar.setObjectName("bottomBar")
        self.bottom_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.bottom_bar.setStyleSheet(
            f"QWidget#bottomBar {{ background: {GROUND}; border-top: 1px solid #141414; }}"
            "QPushButton { padding: 4px 0; }")
        layout = QHBoxLayout(self.bottom_bar)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(8)
        self.previous_btn = QPushButton(_("Anterior"))
        self.previous_btn.setToolTip(_("Aplicar los ajustes del fotograma anterior"))
        self.reset_btn = QPushButton(_("Restablecer"))
        self.reset_btn.setToolTip(_("Volver a los ajustes por defecto (conserva película, base y recorte)"))
        layout.addWidget(self.previous_btn, 1)
        layout.addWidget(self.reset_btn, 1)
        self.previous_btn.clicked.connect(self.previousRequested)
        self.reset_btn.clicked.connect(self.resetRequested)
        self.setViewportMargins(0, 0, 0, self.BAR_HEIGHT)
        # The margins shrink only the viewport; the scrollbar would still run
        # down over the bar, so its container is shortened whenever Qt lays it out.
        self._scrollbar_box.installEventFilter(self)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.bottom_bar.setGeometry(0, self.height() - self.BAR_HEIGHT, self.width(), self.BAR_HEIGHT)
        self._fit_scrollbar()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._scrollbar_box and event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
            self._fit_scrollbar()
        return super().eventFilter(obj, event)

    def _fit_scrollbar(self) -> None:
        box = self._scrollbar_box.geometry()
        bottom = self.height() - self.BAR_HEIGHT  # first row of the bar
        if box.y() + box.height() > bottom:
            self._scrollbar_box.setGeometry(box.x(), box.y(), box.width(), max(bottom - box.y(), 0))

    def _align_labels(self, sec: Section) -> None:
        """One label column per panel, as wide as its longest label."""
        labels = [row.label for row in sec.body.findChildren(SliderRow)]
        labels += sec.body.findChildren(QLabel, "fieldLabel")
        if labels:
            width = max(label.sizeHint().width() for label in labels)
            for label in labels:
                label.setFixedWidth(width)

    # ================================================================ profiles
    def reload_profiles(self) -> None:
        current = self.profile_combo.currentData()
        model = QStandardItemModel(self.profile_combo)
        user = [p for p in self.library.all() if not p.builtin]
        groups: list[tuple[str, list[FilmProfile]]] = []
        if user:
            groups.append((_("Mis perfiles"), user))
        for key, title in GROUPS.items():
            groups.append((_(title), [p for p in self.library.all() if p.builtin and p.type == key]))
        for title, profiles in groups:
            header = QStandardItem(title.upper())
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            model.appendRow(header)
            for p in profiles:
                item = QStandardItem(film_name(p))
                item.setData(p.id, Qt.ItemDataRole.UserRole)
                model.appendRow(item)
        self.profile_combo.setModel(model)
        self._select_profile(current or self._settings.profile_id)

    def _select_profile(self, profile_id: str) -> None:
        idx = self.profile_combo.findData(profile_id)
        if idx < 0:
            idx = self.profile_combo.findData("generic-c41")
        self.profile_combo.setCurrentIndex(max(idx, 0))

    def current_profile(self) -> FilmProfile:
        return self.library.get(self._settings.profile_id)

    # ================================================================ state
    @property
    def settings(self) -> DevelopSettings:
        return self._settings

    def load(self, settings: DevelopSettings, lock_base: bool) -> None:
        """Show a frame's settings without emitting anything."""
        self._settings = settings.copy()
        self.base_lock.setChecked(lock_base)
        self._analysis = None
        self._sync()

    def apply_external(self, **changes) -> None:
        """Change settings from outside (tools, rotation, Upright…) and notify.

        Only ``settingsChanged`` is emitted: the caller records the history step.
        """
        self._settings = self._settings.copy(**changes)
        self._sync()
        self.settingsChanged.emit(self._settings.copy())

    def show_analysis(self, analysis: Analysis | None) -> None:
        self._analysis = analysis
        self._refresh_base()

    def set_exposure_note(self, text: str) -> None:
        self.exposure_note.setText(text)
        self.exposure_note.setVisible(bool(text))

    def set_histogram(self, hist) -> None:
        """Feeds the faint histogram behind the tone curve."""
        self.curve_editor.set_histogram(hist)

    def set_grid_checked(self, on: bool) -> None:
        """Show the grid state the window keeps, without emitting ``gridToggled``."""
        self.show_grid.setChecked(on)

    def get_state(self) -> dict:
        """What is folded and which tabs are shown, for the window to remember."""
        state = self.sections.get_state()
        state.update(hsl_tab=self.hsl_tabs.current(), grade_region=self.grade_tabs.current(),
                     curve_channel=self.curve_editor.channel())
        return state

    def set_state(self, state: dict | None) -> None:
        if not state:
            return
        self.sections.set_state(state)
        if state.get("hsl_tab") in ("hue", "sat", "lum", "all"):
            self._show_hsl_tab(state["hsl_tab"])
        if state.get("grade_region") in self.grade_groups:
            self._show_grade_region(state["grade_region"])
        if state.get("curve_channel") in CHANNELS:
            self.curve_editor.set_channel(state["curve_channel"])

    def show_section(self, key: str) -> None:
        """Open a panel and scroll its header to the top (the Transformar tool opens its panel)."""
        section = self.sections.section(key)
        section.set_expanded(True)
        # The scroll range only grows once the layout has run.
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(section.y()))

    def reset_section(self, key: str) -> None:
        self._reset_fields(SECTION_FIELDS[key], _("Restablecer {panel}").format(panel=_(SECTION_TITLES[key])))

    # ================================================================ model -> widgets
    def _sync(self) -> None:
        """Show ``self._settings`` everywhere. Setters here never emit signals."""
        s = self._settings
        self._select_profile(s.profile_id)
        self.output_combo.setCurrentIndex(max(0, self.output_combo.findData(s.output)))
        self.auto_crop.setChecked(s.crop is None and s.auto_crop)
        self.base_auto.setEnabled(not self.base_lock.isChecked())
        for key, (field, index) in self._bindings.items():
            value = getattr(s, field)
            self.rows[key].set_value(value if index is None else value[index])
        profile = self.current_profile()
        use_profile = s.separation is None
        self.separation_profile.setChecked(use_profile)
        self.separation.default = profile.separation
        self.separation.set_value(profile.separation if use_profile else float(s.separation))
        self.curve_editor.set_curves({c: getattr(s, f"curve_{c}") for c in CHANNELS})
        self.curve_editor.set_splits(s.curve_splits)
        self._refresh_parametric()
        self._sync_wheel()
        for key in SWITCHED:
            self.sections.section(key).set_active(s.section_on(key))
        self.constrain_crop.setChecked(s.constrain_crop)
        self._sync_upright()
        self.neutral_state.setText(_("Gris medido") if any(s.neutral) else "")
        self.neutral_clear.setVisible(any(s.neutral))
        self._refresh_profile_info()
        self._refresh_base()

    def _sync_upright(self) -> None:
        """Light the button of ``upright_mode``; none while no Upright is applied ("")."""
        self.upright_group.setExclusive(False)  # an exclusive group cannot uncheck its last button
        for mode, button in self.upright_buttons.items():
            button.setChecked(mode == self._settings.upright_mode)
        self.upright_group.setExclusive(True)

    def _refresh_base(self) -> None:
        analysis = self._analysis
        if self._settings.base is not None:
            self.base_swatch.set_color(self._settings.base)
            self.base_label.setText(_("Medida en la imagen"))
        elif analysis is not None and analysis.channels == 3:
            self.base_swatch.set_color(tuple(float(v) for v in analysis.base))
            self.base_label.setText(_("Automática (estimada)"))
        else:
            self.base_swatch.set_color(None)
            self.base_label.setText(_("Automática"))

    def _refresh_profile_info(self) -> None:
        profile = self.current_profile()
        parts = [profile.process]
        if profile.iso:
            parts.append(f"ISO {profile.iso}")
        if not profile.builtin:
            parts.append(_("perfil propio"))
        text = " · ".join(parts)
        if profile.notes:
            text += "\n" + profile.notes
        self.profile_notes.setText(text)
        color = not profile.is_bw
        for widget in (self.rows["temperature"], self.rows["tint"], self.rows["vibrance"], self.rows["saturation"],
                       self.neutral_pick, self.neutral_clear, self.neutral_state, self.separation_profile,
                       self.sections.section("hsl").body):
            widget.setEnabled(color)
        self.separation.setEnabled(color and not self.separation_profile.isChecked())

    def _refresh_parametric(self) -> None:
        s = self._settings
        flat = all(getattr(s, f) == 0.0 for f in PARAMETRIC[:4])
        if flat:
            self.curve_editor.set_parametric_lut(None)
            return
        from belka.core import adjust

        self.curve_editor.set_parametric_lut(adjust.parametric_lut(s, 256))

    def _grade_value(self) -> tuple[float, float, float]:
        return getattr(self._settings, f"grade_{self.grade_tabs.current()}")

    def _sync_wheel(self) -> None:
        hue, sat, _lum = self._grade_value()
        self.grade_wheel.set_value(hue, sat)
        region = self.grade_tabs.current()
        self.rows[f"grade_{region}.1"].set_gradient([_hsv(hue, 0.0, 0.55), _hsv(hue, 0.75, 0.85)])

    # ================================================================ widgets -> model
    def _change(self, label: str | None, **changes) -> None:
        """Apply a change made in the panel; ``label`` closes a history step."""
        self._settings = self._settings.copy(**changes)
        if "base" in changes:
            self._refresh_base()
        if "neutral" in changes:
            self.neutral_state.setText(_("Gris medido") if any(self._settings.neutral) else "")
            self.neutral_clear.setVisible(any(self._settings.neutral))
        self.settingsChanged.emit(self._settings.copy())
        if label:
            self.editCommitted.emit(label)

    def _on_row(self, field: str, index: int | None, value: float) -> None:
        if index is None:
            self._change(None, **{field: value})
        else:
            values = list(getattr(self._settings, field))
            values[index] = value
            self._change(None, **{field: tuple(values)})
        if field in PARAMETRIC:
            self._refresh_parametric()
        elif field.startswith("grade_") and index is not None:
            self._sync_wheel()

    def _on_splits(self, splits) -> None:
        self._change(None, curve_splits=tuple(splits))
        self._refresh_parametric()

    def _set_wheel(self, hue: float, sat: float, label: str | None) -> None:
        """Hue and saturation of the shown region from its wheel; its rows follow."""
        region = self.grade_tabs.current()
        lum = self._grade_value()[2]
        self._change(label, **{f"grade_{region}": (hue, sat, lum)})
        self.rows[f"grade_{region}.0"].set_value(hue)
        self.rows[f"grade_{region}.1"].set_value(sat)
        self._sync_wheel()

    def _on_wheel(self, hue: float, sat: float) -> None:
        self._set_wheel(hue, sat, None)

    def _on_wheel_pressed(self) -> None:
        self._grade_start = self._grade_value()

    def _wheel_label(self) -> str:
        return _("Gradación de color: {region}").format(region=self.grade_title.text())

    def _on_wheel_released(self) -> None:
        if self._grade_start is not None and self._grade_start != self._grade_value():
            self.editCommitted.emit(self._wheel_label())
        self._grade_start = None

    def _on_wheel_reset(self) -> None:
        if self._grade_value()[:2] != (0.0, 0.0):
            self._set_wheel(0.0, 0.0, self._wheel_label())

    def _on_upright(self, mode: str) -> None:
        self.uprightRequested.emit(mode)
        self._sync_upright()  # undo the click's own check: the button follows the settings

    def _on_switch(self, key: str, on: bool) -> None:
        disabled = tuple(d for d in self._settings.disabled if d != key) + (() if on else (key,))
        self.sections.section(key).set_active(on)
        title = _(SECTION_TITLES[key])
        self._change(_("Activar {panel}").format(panel=title) if on else _("Desactivar {panel}").format(panel=title),
                     disabled=disabled)

    def _reset_fields(self, fields: tuple[str, ...], label: str) -> None:
        changes = {f: getattr(DEFAULTS, f) for f in fields if getattr(self._settings, f) != getattr(DEFAULTS, f)}
        if not changes:
            return
        self._settings = self._settings.copy(**changes)
        self._sync()
        self.settingsChanged.emit(self._settings.copy())
        self.editCommitted.emit(label)

    def _on_profile(self, index: int) -> None:
        profile_id = self.profile_combo.itemData(index)
        if not profile_id or profile_id == self._settings.profile_id:
            return
        self._settings = self._settings.copy(profile_id=profile_id)
        profile = self.current_profile()
        self._refresh_profile_info()
        self.separation.default = profile.separation
        if self.separation_profile.isChecked():
            self.separation.set_value(profile.separation)
        self._change(_("Película: {name}").format(name=film_name(profile)))

    def _on_output(self, index: int) -> None:
        output = self.output_combo.itemData(index)
        if output != self._settings.output:
            self._change(_("Salida: {name}").format(name=self.output_combo.itemText(index)), output=output)

    def _on_auto_crop(self, on: bool) -> None:
        # Turning it on replaces a manual crop; turning it off shows everything.
        self._change(_("Encuadre automático") if on else _("Sin encuadre automático"), auto_crop=on, crop=None)

    def _on_lock_base(self, on: bool) -> None:
        # With the roll base locked, export uses it whatever the panel says, so
        # "Auto" would only change the preview.
        self.base_auto.setEnabled(not on)
        self.lockBaseChanged.emit(on)

    def _on_separation(self, value: float) -> None:
        if not self.separation_profile.isChecked():
            self._change(None, separation=value)

    def _on_separation_profile(self, use_profile: bool) -> None:
        self.separation.setEnabled(not use_profile and not self.current_profile().is_bw)
        if use_profile:
            self.separation.set_value(self.current_profile().separation)
            self._change(_("Separación del perfil"), separation=None)
        else:
            self._change(_("Separación propia"), separation=self.separation.value())
