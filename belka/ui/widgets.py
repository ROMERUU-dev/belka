"""Small reusable widgets, and the Lightroom-style slider rows of the develop panel."""

from __future__ import annotations

import math
from collections.abc import Container

import numpy as np
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QWidget,
)

from belka.i18n import _, language


class LabeledSlider(QWidget):
    """Slider with a numeric box; double-click the label to reset."""

    valueChanged = Signal(float)

    def __init__(self, label: str, minimum: float, maximum: float, default: float, step: float = 0.01,
                 decimals: int = 2, suffix: str = "", tooltip: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._default = default
        self._scale = 1.0 / step
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.label = QLabel(label)
        self.label.setMinimumWidth(104)
        self.label.setToolTip((tooltip + "\n" if tooltip else "") + "Doble clic: valor por defecto")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(round(minimum * self._scale)), int(round(maximum * self._scale)))
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.box = QDoubleSpinBox()
        self.box.setRange(minimum, maximum)
        self.box.setDecimals(decimals)
        self.box.setSingleStep(step)
        self.box.setSuffix(suffix)
        self.box.setFixedWidth(74)
        self.box.setKeyboardTracking(False)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.box)
        self.slider.valueChanged.connect(self._from_slider)
        self.box.valueChanged.connect(self._from_box)
        self.label.mouseDoubleClickEvent = lambda _e: self.reset()
        self.set_value(default, emit=False)

    def value(self) -> float:
        return float(self.box.value())

    def set_value(self, value: float, emit: bool = False) -> None:
        for w in (self.slider, self.box):
            w.blockSignals(True)
        self.slider.setValue(int(round(value * self._scale)))
        self.box.setValue(value)
        for w in (self.slider, self.box):
            w.blockSignals(False)
        if emit:
            self.valueChanged.emit(self.value())

    def reset(self) -> None:
        self.set_value(self._default, emit=True)

    def _from_slider(self, raw: int) -> None:
        self.box.blockSignals(True)
        self.box.setValue(raw / self._scale)
        self.box.blockSignals(False)
        self.valueChanged.emit(self.value())

    def _from_box(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(value * self._scale)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(self.value())


class HistogramWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(90)
        self._hist: np.ndarray | None = None

    def set_histogram(self, hist: np.ndarray | None) -> None:
        self._hist = hist
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.fillRect(rect, self.palette().base())
        if self._hist is None:
            return
        hist = np.log1p(self._hist.astype(np.float64))
        peak = hist[:, 2:-2].max() or 1.0
        colors = [QColor(230, 60, 60, 110), QColor(60, 200, 80, 110), QColor(70, 120, 240, 110)]
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for c in range(3):
            path = QPainterPath(QPointF(rect.left(), rect.bottom()))
            for i, v in enumerate(hist[c]):
                x = rect.left() + rect.width() * i / 255.0
                y = rect.bottom() - rect.height() * min(v / peak, 1.0)
                path.lineTo(QPointF(x, y))
            path.lineTo(QPointF(rect.right(), rect.bottom()))
            painter.fillPath(path, colors[c])
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(self.palette().mid().color()))
        painter.drawRect(rect)


class ColorSwatch(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(28, 18)
        self.setFrameShape(QFrame.Shape.Box)
        self.set_color(None)

    def set_color(self, rgb: tuple[float, float, float] | None) -> None:
        if rgb is None:
            self.setStyleSheet("background: transparent;")
            return
        peak = max(max(rgb), 1e-6)
        r, g, b = (int(255 * min(1.0, (v / peak) ** (1 / 2.2))) for v in rgb)
        self.setStyleSheet(f"background: rgb({r},{g},{b});")


class WheelGuard(QObject):
    """A wheel over an input scrolls the panel instead of changing the input.

    Without this, scrolling the capture dock past "Velocidad" or "Calidad"
    silently changed the camera's shutter speed or switched it to JPEG. Combo
    and spin boxes never take the wheel, focused or not (a combo keeps the
    focus after a pick, and the next scroll changed the film); only an open
    combo list scrolls itself. A focused slider still takes it.
    """

    def __init__(self, area: QScrollArea):
        super().__init__(area)
        self.area = area

    def guard(self, widget: QWidget) -> None:
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        widget.installEventFilter(self)

    def guard_children(self, root: QWidget) -> None:
        for kind in (QComboBox, QAbstractSpinBox, QSlider):
            for widget in root.findChildren(kind):
                self.guard(widget)

    @staticmethod
    def _scrolls_panel(widget: QWidget) -> bool:
        if isinstance(widget, QComboBox):
            return not widget.view().isVisible()
        return isinstance(widget, QAbstractSpinBox) or not widget.hasFocus()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget) and self._scrolls_panel(obj):
            QCoreApplication.sendEvent(self.area.verticalScrollBar(), event)
            return True
        return False


# ---------------------------------------------------------------- develop panel controls

def format_value(value: float, decimals: int, signed: bool, suffix: str = "") -> str:
    """'+0,35' in Spanish, '+0.35' in English, like Lightroom; zero carries no sign."""
    value = round(float(value), decimals) + 0.0  # + 0.0 turns -0.0 into 0.0
    text = f"{value:+.{decimals}f}" if signed and value != 0 else f"{value:.{decimals}f}"
    if language() == "es":
        text = text.replace(".", ",")
    return text + suffix


def parse_value(text: str) -> float | None:
    """Read a typed number in either decimal convention, ignoring sign and unit decorations."""
    clean = text.strip().replace("−", "-").replace(",", ".").replace("+", "").rstrip("°% ").strip()
    try:
        value = float(clean)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def claim_keys(event, keys: Container[Qt.Key]) -> bool:
    """Take ``keys`` (bare or with Shift) from the window's shortcuts, for ``event()``.

    The main window binds plain keys (Escape cancels a tool, Tab hides the
    panels, the arrows and Delete step through and remove frames). A focused
    control that uses one must accept the ShortcutOverride sent before the
    key press, or the window's action fires instead.
    """
    others = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier
    if event.type() == QEvent.Type.ShortcutOverride and event.key() in keys and not (event.modifiers() & others):
        event.accept()
        return True
    return False


class Mapping:
    """Piecewise-linear map between what a slider shows and what the model stores.

    ``points`` are (display, model) pairs and may run backwards (Negros: +100
    lifts the blacks, which is a negative model ``black``). ``log`` interpolates
    the model in log space, for multipliers such as contrast where 0.5× and 2×
    are equally far from 1×.
    """

    def __init__(self, points: list[tuple[float, float]], log: bool = False):
        pts = sorted(points)
        self._display = np.array([p[0] for p in pts], dtype=np.float64)
        model = np.array([p[1] for p in pts], dtype=np.float64)
        self._log = log
        self._model = np.log(model) if log else model
        self._order = np.argsort(self._model)

    @classmethod
    def linear(cls, display: tuple[float, float], model: tuple[float, float]) -> "Mapping":
        return cls([(display[0], model[0]), (display[1], model[1])])

    @classmethod
    def identity(cls, lo: float, hi: float) -> "Mapping":
        return cls([(lo, lo), (hi, hi)])

    def display_range(self) -> tuple[float, float]:
        return float(self._display[0]), float(self._display[-1])

    def to_model(self, display: float) -> float:
        value = float(np.interp(display, self._display, self._model))
        return float(np.exp(value)) if self._log else value

    def to_display(self, model: float) -> float:
        value = math.log(max(model, 1e-9)) if self._log else float(model)
        return float(np.interp(value, self._model[self._order], self._display[self._order]))


class CheckBox(QCheckBox):
    """Lightroom's small square checkbox with a drawn tick.

    A stylesheet can only draw a tick from an image file, and Fusion's box
    disappears against the dark panels, so the indicator is painted here; the
    text keeps the stylesheet's colour through the palette.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        enabled = self.isEnabled()
        box = QRectF(0.5, (self.height() - 12) / 2 + 0.5, 12, 12)
        border = "#9a9a9a" if self.underMouse() and enabled else ("#6c6c6c" if enabled else "#444444")
        p.setPen(QPen(QColor(border), 1.0))
        p.setBrush(QColor("#191919"))
        p.drawRoundedRect(box, 2, 2)
        if self.isChecked():
            pen = QPen(QColor("#e4e4e4" if enabled else "#5c5c5c"), 1.7)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            x, y = box.left(), box.top()
            p.drawPolyline(QPolygonF([QPointF(x + 3, y + 6.2), QPointF(x + 5.2, y + 8.6), QPointF(x + 9.2, y + 3.6)]))
        group = QPalette.ColorGroup.Active if enabled else QPalette.ColorGroup.Disabled
        p.setPen(self.palette().color(group, QPalette.ColorRole.WindowText))
        text_rect = QRectF(box.right() + 7, 0, self.width() - box.right() - 7, self.height())
        p.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())


def _shift(event) -> bool:
    return bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)


class ClickLabel(QLabel):
    doubleClicked = Signal()
    shiftDoubleClicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:
        (self.shiftDoubleClicked if _shift(event) else self.doubleClicked).emit()


class TrackSlider(QWidget):
    """Thin Lightroom track with a triangular thumb and an optional colour gradient.

    Works in display units. Clicking the track jumps there; dragging the thumb
    keeps the grab offset so picking it up never nudges the value. The wheel
    is left alone so it scrolls the panel.

    It never takes the keyboard focus: after a drag, the arrows and 0-5 must
    still step through frames and rate them, not nudge the slider. Typing and
    nudging live in the row's value field.
    """

    valueChanged = Signal(float)
    pressed = Signal()
    released = Signal()
    resetRequested = Signal()  # double-click
    autoRequested = Signal()  # Shift+double-click

    PAD = 6  # half the thumb: its centre reaches both ends without clipping
    TRACK_Y = 7.5

    def __init__(self, minimum: float, maximum: float, step: float, parent: QWidget | None = None):
        super().__init__(parent)
        self._min, self._max, self._step = float(minimum), float(maximum), float(step)
        self._value = self._min
        self._gradient: list[QColor] = []
        self._grab: float | None = None
        self.setFixedHeight(18)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        self._value = min(max(float(value), self._min), self._max)
        self.update()

    def set_gradient(self, colors: list[QColor]) -> None:
        self._gradient = list(colors)
        self.update()

    def snap(self, value: float) -> float:
        value = round(value / self._step) * self._step
        return min(max(value, self._min), self._max)

    def _x_for(self, value: float) -> float:
        span = self._max - self._min
        t = (value - self._min) / span if span else 0.0
        return self.PAD + t * (self.width() - 2 * self.PAD)

    def _value_at(self, x: float) -> float:
        t = (x - self.PAD) / max(self.width() - 2 * self.PAD, 1)
        return self.snap(self._min + t * (self._max - self._min))

    def _move_to(self, value: float) -> None:
        value = self.snap(value)
        if value != self._value:
            self._value = value
            self.update()
            self.valueChanged.emit(value)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.pressed.emit()
        x = event.position().x()
        if abs(x - self._x_for(self._value)) > self.PAD + 1:
            self._move_to(self._value_at(x))
        self._grab = x - self._x_for(self._value)

    def mouseMoveEvent(self, event) -> None:
        if self._grab is not None:
            self._move_to(self._value_at(event.position().x() - self._grab))

    def mouseReleaseEvent(self, event) -> None:
        if self._grab is not None:
            self._grab = None
            self.update()
            self.released.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        (self.autoRequested if _shift(event) else self.resetRequested).emit()

    def wheelEvent(self, event) -> None:
        event.ignore()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        enabled = self.isEnabled()
        x0, x1, y = float(self.PAD), float(self.width() - self.PAD), self.TRACK_Y
        p.setPen(Qt.PenStyle.NoPen)
        # A disabled colour track is drawn like any disabled track: dimmed
        # greys of the gradient still stood out over the active sliders.
        if self._gradient and enabled:
            grad = QLinearGradient(x0, 0, x1, 0)
            last = max(len(self._gradient) - 1, 1)
            for i, color in enumerate(self._gradient):
                grad.setColorAt(i / last, QColor(color))
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(QRectF(x0 - 1, y - 2, x1 - x0 + 2, 4), 2, 2)
        else:
            p.setBrush(QColor("#555555" if enabled else "#3a3a3a"))
            p.drawRoundedRect(QRectF(x0 - 1, y - 1, x1 - x0 + 2, 2), 1, 1)
        x = self._x_for(self._value)
        if not enabled:
            fill = QColor("#555555")
        elif self._grab is not None:
            fill = QColor("#ffffff")
        elif self.underMouse():
            fill = QColor("#e6e6e6")
        else:
            fill = QColor("#b4b4b4")
        thumb = QPolygonF([QPointF(x, y + 1.0), QPointF(x - 5.0, y + 9.0), QPointF(x + 5.0, y + 9.0)])
        p.setPen(QPen(QColor(0, 0, 0, 140), 1.0))
        p.setBrush(fill)
        p.drawPolygon(thumb)


class ValueField(QLineEdit):
    """The number closing a slider row: click to type, drag sideways to scrub,
    double-click to reset (Shift: automatic value); while typing, Up/Down
    nudge (Shift: ten steps), Return or Tab confirm and Escape cancels.

    Idle, the field takes no focus: with click focus Qt would hand it the
    focus before the press arrived, and every press would look like a click
    into the text, so nothing could be scrubbed or selected. A click without
    a drag opens it for typing with the number selected; losing the focus or
    a press anywhere else commits and closes it (sliders, buttons and the
    bare panel take no focus, so they would leave it open with its number
    unapplied). Closing hands the focus back to whatever had it (the photo),
    so the arrows step through frames again.
    """

    edited = Signal(float)  # display units, live while scrubbing or once typed
    started = Signal()  # a gesture begins (the row notes the value before it)
    finished = Signal()  # it ends: typing confirmed, scrub released
    resetRequested = Signal()  # double-click
    autoRequested = Signal()  # Shift+double-click

    SCRUB_PX = 2  # mouse pixels per step
    EDIT_KEYS = (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Up, Qt.Key.Key_Down,
                 Qt.Key.Key_Tab, Qt.Key.Key_Backtab)

    def __init__(self, step: float, parent: QWidget | None = None):
        super().__init__(parent)
        self._step = step
        self._value = 0.0
        self._shown = ""
        self._editing = False
        self._return_to: QWidget | None = None
        self._press_x: float | None = None
        self._scrub_from: float | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setFrame(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setToolTip(_("Clic: escribir · Arrastrar: ajustar · Doble clic: restablecer"))
        self.returnPressed.connect(self.leave)
        self._close()

    def show_value(self, value: float, text: str) -> None:
        self._value, self._shown = value, text
        if not (self._editing and self.isModified()):
            self.setText(text)

    def _open(self) -> None:
        self._editing = True
        previous = QApplication.focusWidget()
        self._return_to = None if isinstance(previous, ValueField) else previous
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.started.emit()
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.selectAll()
        QApplication.instance().installEventFilter(self)

    def _close(self) -> None:
        self._editing = False
        self._return_to = None
        QApplication.instance().removeEventFilter(self)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.deselect()

    def eventFilter(self, obj, event) -> bool:
        """While open: a press on any other widget closes the field first."""
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(obj, QWidget) and obj is not self:
            self.leave()
        return False

    def leave(self) -> None:
        """Close (committing a typed number) and give the focus back to whatever had it."""
        back = self._return_to
        self.clearFocus()
        if (back is not None and shiboken6.isValid(back) and QApplication.focusWidget() is None
                and back.isVisible() and back.isEnabled()):
            back.setFocus(Qt.FocusReason.OtherFocusReason)

    def mousePressEvent(self, event) -> None:
        if self._editing:
            super().mousePressEvent(event)
        elif event.button() == Qt.MouseButton.LeftButton:
            self._press_x = event.position().x()

    def mouseMoveEvent(self, event) -> None:
        if self._press_x is None:
            if self._editing:
                super().mouseMoveEvent(event)
            return
        dx = event.position().x() - self._press_x
        if self._scrub_from is None and abs(dx) > 3:
            self._scrub_from = self._value
            self.started.emit()
        if self._scrub_from is not None:
            self.edited.emit(self._scrub_from + round(dx / self.SCRUB_PX) * self._step)

    def mouseReleaseEvent(self, event) -> None:
        if self._press_x is None:
            if self._editing:
                super().mouseReleaseEvent(event)
            return
        self._press_x = None
        if self._scrub_from is not None:
            self._scrub_from = None
            self.finished.emit()
        else:
            self._open()

    def mouseDoubleClickEvent(self, event) -> None:
        # The click before it opened the field: close it unchanged, then reset
        # (or, with Shift, ask for the automatic value).
        self._press_x = None
        self.setModified(False)
        self.leave()
        if self._editing:  # the focus never arrived (inactive window)
            self._close()
            self.finished.emit()
        (self.autoRequested if _shift(event) else self.resetRequested).emit()

    def event(self, event) -> bool:
        if self._editing and claim_keys(event, self.EDIT_KEYS):
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            steps = 1 if event.key() == Qt.Key.Key_Up else -1
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                steps *= 10
            self.setModified(False)
            self.edited.emit(self._value + steps * self._step)
            self.selectAll()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.setModified(False)
            self.setText(self._shown)
            self.leave()
            return
        super().keyPressEvent(event)

    def focusNextPrevChild(self, _next: bool) -> bool:
        # Tab confirms the number, as Return does, instead of wandering off to another control.
        self.leave()
        return True

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        if not self._editing or event.reason() in (Qt.FocusReason.ActiveWindowFocusReason,
                                                   Qt.FocusReason.PopupFocusReason):
            return  # switching windows only borrows the focus; typing resumes on return
        value = parse_value(self.text()) if self.isModified() else None
        self.setModified(False)
        self._close()
        if value is None:
            self.setText(self._shown)
        else:
            self.edited.emit(value)
        self.finished.emit()

    def wheelEvent(self, event) -> None:
        event.ignore()


class SliderRow(QWidget):
    """Label · track · value, showing display units over a model value.

    Every change emits ``valueChanged`` in model units; ``editFinished`` comes
    once per gesture (slider released, typing confirmed, reset) with a history
    label such as "Exposición +0,35", and only if the value really changed.
    Double-clicking the label, the thumb or the number resets it; with
    ``auto``, Shift+double-click asks for the automatic value instead
    (Lightroom's per-slider Auto), which the caller computes.
    """

    valueChanged = Signal(float)
    editFinished = Signal(str)
    autoRequested = Signal()
    dragging = Signal(bool)  # a gesture starts (thumb held, number scrubbed or typed) or ends

    def __init__(self, label: str, mapping: Mapping, default: float, *, step: float = 1.0, decimals: int = 0,
                 signed: bool | None = None, suffix: str = "", gradient: list[QColor] | None = None,
                 tooltip: str = "", history: str = "", auto: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.mapping = mapping
        self._history = history
        self.auto = auto
        self.default = float(default)
        lo, hi = mapping.display_range()
        self._decimals, self._suffix = decimals, suffix
        self._signed = lo < 0 if signed is None else signed
        self._model = self.default
        self._start: float | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.label = ClickLabel(label)
        self.label.setObjectName("rowLabel")
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        auto_tip = "\n" + _("Mayús+doble clic: automático") if auto else ""
        self.label.setToolTip((tooltip + "\n" if tooltip else "") + _("Doble clic: valor por defecto") + auto_tip)
        self.slider = TrackSlider(lo, hi, step)
        self.field = ValueField(step)
        self.field.setFixedWidth(42)
        self.field.setToolTip(self.field.toolTip() + auto_tip)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.field)
        self.setFixedHeight(21)
        if gradient:
            self.slider.set_gradient(gradient)

        self.slider.pressed.connect(self._begin)
        self.slider.valueChanged.connect(self.set_display)
        self.slider.released.connect(self._finish)
        self.slider.resetRequested.connect(self.reset)
        self.slider.autoRequested.connect(self._shift_double_click)
        self.field.started.connect(self._begin)
        self.field.edited.connect(self.set_display)
        self.field.finished.connect(self._finish)
        self.field.resetRequested.connect(self.reset)
        self.field.autoRequested.connect(self._shift_double_click)
        self.label.doubleClicked.connect(self.reset)
        self.label.shiftDoubleClicked.connect(self._shift_double_click)
        self.set_value(self.default)

    def value(self) -> float:
        return self._model

    def display_value(self) -> float:
        return self.slider.value()

    def text(self) -> str:
        return format_value(self.slider.value(), self._decimals, self._signed, self._suffix)

    def history_label(self) -> str:
        """``history`` names the control where the label alone is ambiguous ("Grano: Cantidad")."""
        return f"{self._history or self.label.text()} {self.text()}"

    def set_value(self, model: float) -> None:
        """Show a model value without emitting anything (loading a frame)."""
        self._model = float(model)
        self.slider.set_value(self.mapping.to_display(self._model))
        self.field.show_value(self.slider.value(), self.text())

    def set_gradient(self, colors: list[QColor]) -> None:
        self.slider.set_gradient(colors)

    def reset(self) -> None:
        changed = abs(self._model - self.default) > 1e-9
        self._start = None
        self.set_value(self.default)
        if changed:
            self.valueChanged.emit(self._model)
            self.editFinished.emit(self.history_label())

    def set_display(self, display: float) -> None:
        """Set from display units as the user does, emitting ``valueChanged`` if it changes."""
        display = self.slider.snap(display)
        model = self.mapping.to_model(display)
        self.slider.set_value(display)
        self.field.show_value(display, self.text())
        if abs(model - self._model) > 1e-12:
            self._model = model
            self.valueChanged.emit(model)

    def _shift_double_click(self) -> None:
        if self.auto:
            self.autoRequested.emit()
        else:
            self.reset()

    def _begin(self) -> None:
        self._start = self._model
        self.dragging.emit(True)

    def _finish(self) -> None:
        if self._start is not None and abs(self._model - self._start) > 1e-9:
            self.editFinished.emit(self.history_label())
        self._start = None
        self.dragging.emit(False)


class GlyphButton(QAbstractButton):
    """A small checkable circle in Lightroom's manner: the curve channel dots and
    the colour-grading regions ("dot", "half" for midtones, "ring" for global)."""

    def __init__(self, glyph: str, color: QColor, tooltip: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.glyph, self.color = glyph, QColor(color)
        self.setCheckable(True)
        self.setToolTip(tooltip)
        self.setFixedSize(24, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    def sizeHint(self) -> QSize:
        return QSize(24, 24)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        r = 6.0
        if self.isChecked() or self.underMouse():
            p.setPen(QPen(QColor("#e0e0e0" if self.isChecked() else "#6a6a6a"), 1.3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, r + 3.5, r + 3.5)
        color = self.color if self.isEnabled() else QColor("#4a4a4a")
        outline = QPen(QColor(0, 0, 0, 160), 1.0)
        if self.glyph == "half":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#2e2e2e"))
            p.drawPie(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), 90 * 16, 180 * 16)
            p.setBrush(QColor("#bdbdbd"))
            p.drawPie(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), -90 * 16, 180 * 16)
            p.setPen(QPen(QColor("#8a8a8a"), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, r, r)
        elif self.glyph == "ring":
            p.setPen(QPen(color, 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, r - 0.5, r - 0.5)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawEllipse(c, 2.5, 2.5)
        else:
            p.setPen(outline if color.lightness() > 90 else QPen(QColor("#8a8a8a"), 1.0))
            p.setBrush(color)
            p.drawEllipse(c, r, r)


class Segmented(QWidget):
    """Mutually exclusive choices in a row: text tabs ("Tono  Saturación…") or
    glyph buttons. ``changed`` carries the key of the chosen item."""

    changed = Signal(str)

    def __init__(self, items: list[tuple[str, str | QAbstractButton]], parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QAbstractButton] = {}
        layout.addStretch(1)
        for key, item in items:
            if isinstance(item, str):
                button = QToolButton()
                button.setText(item)
                button.setCheckable(True)
                button.setAutoRaise(True)
                button.setProperty("segment", True)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                button = item
            self._group.addButton(button)
            self._buttons[key] = button
            layout.addWidget(button)
            button.clicked.connect(lambda _c=False, k=key: self.changed.emit(k))
        layout.addStretch(1)
        if items:
            self.set_current(items[0][0])

    def button(self, key: str) -> QAbstractButton:
        return self._buttons[key]

    def current(self) -> str:
        return next((k for k, b in self._buttons.items() if b.isChecked()), "")

    def set_current(self, key: str) -> None:
        if key in self._buttons:
            self._buttons[key].setChecked(True)


class ColorWheel(QWidget):
    """Hue around, saturation outwards, as in Lightroom's colour grading.

    Hue follows the Lightroom/HSV convention: 0° red on the right, counter-
    clockwise through yellow (60°), green (120°) and blue (240°). Click or drag
    to place the puck (Shift keeps the hue and changes only the saturation),
    double-click to clear it.
    """

    changed = Signal(float, float)  # hue in degrees, saturation 0..1
    pressed = Signal()
    released = Signal()
    resetRequested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._hue, self._sat = 0.0, 0.0
        self._dragging = False
        self._cache: QPixmap | None = None
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(150)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def value(self) -> tuple[float, float]:
        return self._hue, self._sat

    def set_value(self, hue: float, sat: float) -> None:
        self._hue, self._sat = float(hue) % 360.0, min(max(float(sat), 0.0), 1.0)
        self.update()

    def _geometry(self) -> tuple[QPointF, float]:
        r = min(self.width(), self.height()) / 2 - 6
        return QPointF(self.width() / 2, self.height() / 2), r

    def _wheel_image(self, radius: int) -> QPixmap:
        n = 2 * radius + 1
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
        rho = np.hypot(xx, yy) / radius
        hue = (np.degrees(np.arctan2(-yy, xx)) % 360.0) / 60.0
        sat = np.clip(rho, 0.0, 1.0) * 0.78
        val = 0.72
        # HSV to RGB, vectorised.
        i = np.floor(hue).astype(int) % 6
        f = hue - np.floor(hue)
        pv, qv, tv = val * (1 - sat), val * (1 - sat * f), val * (1 - sat * (1 - f))
        r = np.choose(i, [val * np.ones_like(f), qv, pv, pv, tv, val * np.ones_like(f)])
        g = np.choose(i, [tv, val * np.ones_like(f), val * np.ones_like(f), qv, pv, pv])
        b = np.choose(i, [pv, pv, tv, val * np.ones_like(f), val * np.ones_like(f), qv])
        alpha = np.clip((1.0 - rho) * radius + 0.5, 0.0, 1.0)  # one-pixel antialiased rim
        rgba = np.dstack([r * alpha, g * alpha, b * alpha, alpha])  # premultiplied
        data = np.ascontiguousarray((rgba * 255 + 0.5).astype(np.uint8))
        image = QImage(data.data, n, n, 4 * n, QImage.Format.Format_RGBA8888_Premultiplied).copy()
        return QPixmap.fromImage(image)

    def _set_from(self, pos: QPointF, keep_hue: bool) -> None:
        c, r = self._geometry()
        dx, dy = pos.x() - c.x(), c.y() - pos.y()
        sat = min(math.hypot(dx, dy) / max(r, 1.0), 1.0)
        hue = self._hue if keep_hue else math.degrees(math.atan2(dy, dx)) % 360.0
        if (round(hue, 1), round(sat, 3)) != (round(self._hue, 1), round(self._sat, 3)):
            self._hue, self._sat = round(hue, 1) % 360.0, round(sat, 3)
            self.update()
            self.changed.emit(self._hue, self._sat)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.pressed.emit()
            self._set_from(event.position(), bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._set_from(event.position(), bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging:
            self._dragging = False
            self.released.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        self.resetRequested.emit()

    def wheelEvent(self, event) -> None:
        event.ignore()

    def resizeEvent(self, event) -> None:
        self._cache = None
        super().resizeEvent(event)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, r = self._geometry()
        radius = max(int(r), 4)
        if self._cache is None or self._cache.width() != 2 * radius + 1:
            self._cache = self._wheel_image(radius)
        if not self.isEnabled():
            p.setOpacity(0.35)
        p.drawPixmap(int(c.x()) - radius, int(c.y()) - radius, self._cache)
        p.setOpacity(1.0)
        p.setPen(QPen(QColor(0, 0, 0, 120), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(int(c.x()) + 0.5, int(c.y()) + 0.5), radius + 0.5, radius + 0.5)
        # Faint crosshair through the neutral centre.
        p.setPen(QPen(QColor(255, 255, 255, 40), 1.0))
        p.drawLine(QPointF(c.x() - 5, c.y()), QPointF(c.x() + 5, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - 5), QPointF(c.x(), c.y() + 5))
        angle = math.radians(self._hue)
        puck = QPointF(c.x() + math.cos(angle) * self._sat * r, c.y() - math.sin(angle) * self._sat * r)
        if self._sat > 0.0:
            p.setPen(QPen(QColor(255, 255, 255, 90), 1.0))
            p.drawLine(c, puck)
        p.setPen(QPen(QColor(0, 0, 0, 170), 3.0))
        p.drawEllipse(puck, 5.0, 5.0)
        p.setPen(QPen(QColor("#f2f2f2"), 1.6))
        p.setBrush(QColor.fromHsvF(self._hue / 360.0, self._sat * 0.78, 0.72))
        p.drawEllipse(puck, 5.0, 5.0)
