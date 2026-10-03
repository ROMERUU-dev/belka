"""Tone curve editor: point curves per channel over the image's histogram.

Like Lightroom Classic's point curve: click to add a point, drag it, drag it
off the graph or double-click it to remove it; arrows move the selected point
(Shift: ten times more) and Delete removes it. The three handles under the
graph set where the parametric regions (sombras, oscuros, claros, altas
luces) meet. Curves are sampled with :func:`belka.core.adjust.curve_lut`,
the spline the pipeline uses, and on the RGB channel the bright line is the
point curve applied after the region curve, as the pipeline does; the point
curve the handles sit on shows faintly beneath it once the two differ.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QMenu, QSizePolicy, QVBoxLayout, QWidget

from belka.core.pipeline import DevelopSettings
from belka.i18n import _
from belka.ui.widgets import GlyphButton, Segmented, claim_keys

CHANNELS = ("rgb", "red", "green", "blue")
CHANNEL_COLORS = {"rgb": "#d2d2d2", "red": "#e2605a", "green": "#5cbf6a", "blue": "#5f8fe8"}
IDENTITY: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
DEFAULT_SPLITS = DevelopSettings().curve_splits

MIN_GAP = 0.01  # closest two points may come along x
MAX_POINTS = 16
HIT_PX = 7.0
REMOVE_PX = 24.0  # this far outside the graph, a dragged point is removed
SPLIT_GAP = 0.05
SPLIT_RANGE = (0.05, 0.95)
POINT_KEYS = (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Delete,
              Qt.Key.Key_Backspace)


def channel_name(channel: str) -> str:
    return {"rgb": "RGB", "red": _("Rojo"), "green": _("Verde"), "blue": _("Azul")}[channel]


@lru_cache(maxsize=64)
def _curve_samples(points: tuple[tuple[float, float], ...], size: int = 256) -> np.ndarray:
    from belka.core import adjust

    return np.asarray(adjust.curve_lut(points, size), dtype=np.float64)


class CurvePlot(QWidget):
    """The square graph plus the strip of split handles beneath it."""

    curveEdited = Signal(object)  # points of the shown channel, live
    splitsEdited = Signal(object)
    finished = Signal(str)  # "curve" or "splits", once per gesture that changed something

    STRIP = 16
    INSET = 5

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.channel = "rgb"
        self.curves: dict[str, list[tuple[float, float]]] = {c: list(IDENTITY) for c in CHANNELS}
        self.splits = list(DEFAULT_SPLITS)
        self.param_lut: np.ndarray | None = None
        self.hist: np.ndarray | None = None
        self.selected: int | None = None
        self._drag: int | None = None
        self._drag_off = False  # the dragged point is outside: it goes on release
        self._split_drag: int | None = None
        self._before: tuple | None = None
        self._hover: QPointF | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumSize(160, 160 + self.STRIP)

    # ------------------------------------------------------------ geometry
    def hasHeightForWidth(self) -> bool:
        return True

    def sizeHint(self) -> QSize:
        return QSize(256, 256 + self.STRIP)

    def heightForWidth(self, width: int) -> int:
        return width + self.STRIP

    def plot_rect(self) -> QRectF:
        # Inset so the end points, drawn on the border, are never clipped.
        side = min(self.width(), self.height() - self.STRIP) - 2 * self.INSET
        return QRectF((self.width() - side) / 2, self.INSET, side, side)

    def to_px(self, x: float, y: float) -> QPointF:
        r = self.plot_rect()
        return QPointF(r.left() + x * r.width(), r.bottom() - y * r.height())

    def from_px(self, pos: QPointF) -> tuple[float, float]:
        r = self.plot_rect()
        return (pos.x() - r.left()) / r.width(), (r.bottom() - pos.y()) / r.height()

    def points(self) -> list[tuple[float, float]]:
        pts = self.curves[self.channel]
        if self._drag is not None and self._drag_off:
            return [p for i, p in enumerate(pts) if i != self._drag]
        return pts

    def rendered_curve(self) -> np.ndarray:
        """The tone curve the pipeline applies for the shown channel's line.

        On RGB it is the point curve applied after the region curve
        (``adjust._curve_luts``); on R/G/B, the channel's own point curve.
        """
        samples = _curve_samples(tuple(self.points()))
        if self.channel != "rgb" or self.param_lut is None:
            return samples
        lut = np.asarray(self.param_lut, dtype=np.float64)
        return np.interp(lut, np.linspace(0.0, 1.0, len(samples)), samples)

    def _has_selection(self) -> bool:
        return self.selected is not None and self.selected < len(self.curves[self.channel])

    def _hit(self, pos: QPointF) -> int | None:
        best, best_d = None, HIT_PX
        for i, (x, y) in enumerate(self.curves[self.channel]):
            d = (self.to_px(x, y) - pos).manhattanLength()
            if d <= best_d:
                best, best_d = i, d
        return best

    def _split_hit(self, pos: QPointF) -> int | None:
        if self.channel != "rgb" or pos.y() < self.plot_rect().bottom() + 1:
            return None
        xs = [abs(self.to_px(s, 0).x() - pos.x()) for s in self.splits]
        i = int(np.argmin(xs))
        return i if xs[i] <= 8 else None

    # ------------------------------------------------------------ editing
    def _state(self) -> tuple:
        return tuple(self.curves[self.channel]), tuple(self.splits)

    def _emit_curve(self) -> None:
        self.update()
        self.curveEdited.emit(tuple(self.points()))

    def _finish(self, kind: str) -> None:
        if self._before is not None and self._before != self._state():
            self.finished.emit(kind)
        self._before = None

    def _constrain(self, i: int, x: float, y: float) -> tuple[float, float]:
        pts = self.curves[self.channel]
        lo = pts[i - 1][0] + MIN_GAP if i > 0 else 0.0
        hi = pts[i + 1][0] - MIN_GAP if i < len(pts) - 1 else 1.0
        return round(min(max(x, lo), hi), 4), round(min(max(y, 0.0), 1.0), 4)

    def _move_point(self, i: int, x: float, y: float) -> None:
        self.curves[self.channel][i] = self._constrain(i, x, y)

    def remove_point(self, i: int) -> None:
        pts = self.curves[self.channel]
        if 0 < i < len(pts) - 1:
            del pts[i]
        else:
            # The end points stay; removing one puts it back in its corner.
            pts[i] = IDENTITY[0] if i == 0 else IDENTITY[1]
        self.selected = None

    def reset_channel(self, channel: str) -> None:
        self.curves[channel] = list(IDENTITY)
        self.selected = None

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        self._before = self._state()
        split = self._split_hit(pos)
        if split is not None:
            self._split_drag = split
            return
        if not self.plot_rect().adjusted(-HIT_PX, -HIT_PX, HIT_PX, HIT_PX).contains(pos):
            return
        i = self._hit(pos)
        if i is None:
            pts = self.curves[self.channel]
            x, y = self.from_px(pos)
            x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
            if len(pts) >= MAX_POINTS or any(abs(px - x) < MIN_GAP for px, _py in pts):
                return
            i = int(np.searchsorted([p[0] for p in pts], x))
            if i == 0 or i == len(pts):
                return  # outside the end points: nothing to bend there
            pts.insert(i, (round(x, 4), round(y, 4)))
            self._emit_curve()
        self._drag, self._drag_off, self.selected = i, False, i
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        self._hover = pos
        if self._split_drag is not None:
            k = self._split_drag
            x, _y = self.from_px(pos)
            lo = self.splits[k - 1] + SPLIT_GAP if k > 0 else SPLIT_RANGE[0]
            hi = self.splits[k + 1] - SPLIT_GAP if k < 2 else SPLIT_RANGE[1]
            value = round(min(max(x, lo), hi), 3)
            if value != self.splits[k]:
                self.splits[k] = value
                self.update()
                self.splitsEdited.emit(tuple(self.splits))
            return
        if self._drag is not None:
            pts = self.curves[self.channel]
            interior = 0 < self._drag < len(pts) - 1
            outside = not self.plot_rect().adjusted(-REMOVE_PX, -REMOVE_PX, REMOVE_PX, REMOVE_PX).contains(pos)
            self._drag_off = interior and outside
            if not self._drag_off:
                self._move_point(self._drag, *self.from_px(pos))
            self._emit_curve()
            return
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._split_drag is not None:
            self._split_drag = None
            self._finish("splits")
            return
        if self._drag is not None:
            if self._drag_off:
                self.remove_point(self._drag)
            self._drag, self._drag_off = None, False
            self.update()
            self._finish("curve")

    def mouseDoubleClickEvent(self, event) -> None:
        pos = event.position()
        self._before = self._state()
        split = self._split_hit(pos)
        if split is not None:
            self.splits[split] = DEFAULT_SPLITS[split]
            self.update()
            self.splitsEdited.emit(tuple(self.splits))
            self._finish("splits")
            return
        i = self._hit(pos)
        if i is not None:
            self.remove_point(i)
            self._emit_curve()
            self._finish("curve")
        else:
            self._before = None

    def leaveEvent(self, _event) -> None:
        self._hover = None
        self.update()

    def event(self, event) -> bool:
        # With a point selected, arrows and Delete edit it instead of changing or removing frames.
        if self._has_selection() and claim_keys(event, POINT_KEYS):
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        pts = self.curves[self.channel]
        if not self._has_selection():
            super().keyPressEvent(event)
            return
        step = (10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1) / 255.0
        moves = {Qt.Key.Key_Up: (0, step), Qt.Key.Key_Down: (0, -step),
                 Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0)}
        self._before = self._state()
        if event.key() in moves:
            dx, dy = moves[event.key()]
            x, y = pts[self.selected]
            self._move_point(self.selected, x + dx, y + dy)
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_point(self.selected)
        else:
            self._before = None
            super().keyPressEvent(event)
            return
        self._emit_curve()
        self._finish("curve")

    def wheelEvent(self, event) -> None:
        event.ignore()

    # ------------------------------------------------------------ painting
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.plot_rect()
        enabled = self.isEnabled()
        p.fillRect(r, QColor("#1c1c1c"))
        self._paint_histogram(p, r)
        grid = QPen(QColor(255, 255, 255, 20), 1.0)
        p.setPen(grid)
        for k in (1, 2, 3):
            x = round(r.left() + r.width() * k / 4) + 0.5
            y = round(r.top() + r.height() * k / 4) + 0.5
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(QPen(QColor(255, 255, 255, 34), 1.0))
        p.drawLine(self.to_px(0, 0), self.to_px(1, 1))
        if self.channel == "rgb":
            p.setPen(QPen(QColor(255, 255, 255, 16), 1.0, Qt.PenStyle.DashLine))
            for s in self.splits:
                x = self.to_px(s, 0).x()
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        color = QColor(CHANNEL_COLORS[self.channel] if enabled else "#5a5a5a")
        pts = tuple(self.points())
        curve = self.rendered_curve()
        if self.channel == "rgb" and self.param_lut is not None and pts != IDENTITY:
            faint = QColor(color)
            faint.setAlpha(100)
            self._paint_samples(p, _curve_samples(pts), QPen(faint, 1.0))
        self._paint_samples(p, curve, QPen(color, 1.6))
        for i, (x, y) in enumerate(self.curves[self.channel]):
            if i == self._drag and self._drag_off:
                continue
            c = self.to_px(x, y)
            active = i == self._drag or i == self.selected
            p.setPen(QPen(color, 1.4))
            p.setBrush(color if active else QColor("#1c1c1c"))
            p.drawEllipse(c, 3.6, 3.6)
        p.setPen(QPen(QColor("#3a3a3a"), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r.adjusted(-0.5, -0.5, 0.5, 0.5))
        self._paint_readout(p, r, curve)
        self._paint_strip(p, r)

    def _paint_samples(self, p: QPainter, lut: np.ndarray, pen: QPen) -> None:
        r = self.plot_rect()
        n = len(lut)
        poly = QPolygonF([QPointF(r.left() + r.width() * i / (n - 1), r.bottom() - r.height() * float(np.clip(v, 0, 1)))
                          for i, v in enumerate(lut)])
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolyline(poly)

    def _paint_histogram(self, p: QPainter, r: QRectF) -> None:
        if self.hist is None:
            return
        hist = np.asarray(self.hist, dtype=np.float64)
        values = hist.sum(axis=0) if self.channel == "rgb" else hist[CHANNELS.index(self.channel) - 1]
        values = np.sqrt(values)
        peak = float(np.percentile(values[2:-2], 99.5)) if values.size > 8 else float(values.max())
        if peak <= 0:
            return
        path = QPainterPath(QPointF(r.left(), r.bottom()))
        n = len(values)
        for i, v in enumerate(values):
            path.lineTo(QPointF(r.left() + r.width() * i / (n - 1), r.bottom() - r.height() * 0.9 * min(v / peak, 1.0)))
        path.lineTo(QPointF(r.right(), r.bottom()))
        path.closeSubpath()
        tint = QColor(CHANNEL_COLORS[self.channel])
        tint.setAlpha(26 if self.channel == "rgb" else 40)
        p.fillPath(path, tint)

    def _paint_readout(self, p: QPainter, r: QRectF, curve: np.ndarray) -> None:
        if self._drag is not None and not self._drag_off:
            x, y = self.curves[self.channel][self._drag]
        elif self._hover is not None and r.contains(self._hover):
            x = min(max(self.from_px(self._hover)[0], 0.0), 1.0)
            y = float(np.interp(x, np.linspace(0, 1, len(curve)), curve))
        else:
            return
        font = QFont(self.font())
        font.setPixelSize(10)
        p.setFont(font)
        p.setPen(QColor("#9a9a9a"))
        text = _("Entrada {i} %  ·  Salida {o} %").format(i=round(x * 100), o=round(y * 100))
        p.drawText(r.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, text)

    def _paint_strip(self, p: QPainter, r: QRectF) -> None:
        top = r.bottom() + 3
        bar = QRectF(r.left(), top, r.width(), 3)
        grad = QLinearGradient(bar.left(), 0, bar.right(), 0)
        end = QColor("#d8d8d8") if self.channel == "rgb" else QColor(CHANNEL_COLORS[self.channel])
        grad.setColorAt(0.0, QColor("#0c0c0c"))
        grad.setColorAt(1.0, end)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawRect(bar)
        if self.channel != "rgb":
            return
        hover = self._split_hit(self._hover) if self._hover is not None else None
        for k, s in enumerate(self.splits):
            x = self.to_px(s, 0).x()
            lit = k == self._split_drag or k == hover
            p.setPen(QPen(QColor(0, 0, 0, 150), 1.0))
            p.setBrush(QColor("#f0f0f0" if lit else "#a6a6a6") if self.isEnabled() else QColor("#555555"))
            p.drawPolygon(QPolygonF([QPointF(x, top + 3), QPointF(x - 4.5, top + 11), QPointF(x + 4.5, top + 11)]))


class ToneCurveEditor(QWidget):
    """Channel dots (RGB, R, G, B) above the point-curve graph.

    ``curveChanged`` and ``splitsChanged`` fire live while dragging;
    ``editFinished`` once per gesture with a label for the history.
    """

    curveChanged = Signal(str, object)  # channel, ((x, y), ...)
    splitsChanged = Signal(object)  # (sombras|oscuros, oscuros|claros, claros|altas luces)
    editFinished = Signal(str)
    channelChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(4)
        self.channels = Segmented([
            (c, GlyphButton("dot", QColor(CHANNEL_COLORS[c]),
                            _("Curva {channel}").format(channel=channel_name(c))))
            for c in CHANNELS
        ])
        layout.addWidget(self.channels)
        self.plot = CurvePlot()
        layout.addWidget(self.plot)
        self.channels.changed.connect(self.set_channel)
        self.plot.curveEdited.connect(lambda pts: self.curveChanged.emit(self.plot.channel, pts))
        self.plot.splitsEdited.connect(self.splitsChanged)
        self.plot.finished.connect(self._on_finished)
        self.plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.plot.customContextMenuRequested.connect(self._menu)

    def channel(self) -> str:
        return self.plot.channel

    def set_channel(self, channel: str) -> None:
        if channel not in CHANNELS:
            return
        self.channels.set_current(channel)
        if channel != self.plot.channel:
            self.plot.channel = channel
            self.plot.selected = None
            self.plot.update()
            self.channelChanged.emit(channel)

    def curve(self, channel: str) -> tuple[tuple[float, float], ...]:
        return tuple(self.plot.curves[channel])

    def set_curves(self, curves: dict[str, tuple]) -> None:
        """Show stored curves without emitting anything (loading a frame)."""
        for channel, points in curves.items():
            self.plot.curves[channel] = [(float(x), float(y)) for x, y in points]
        self.plot.selected = None
        self.plot.update()

    def splits(self) -> tuple[float, float, float]:
        return tuple(self.plot.splits)

    def set_splits(self, splits: tuple[float, float, float]) -> None:
        self.plot.splits = [float(s) for s in splits]
        self.plot.update()

    def set_parametric_lut(self, lut: np.ndarray | None) -> None:
        """The region curve applied before the RGB point curve (None: flat); the RGB line shows both."""
        self.plot.param_lut = lut
        self.plot.update()

    def set_histogram(self, hist: np.ndarray | None) -> None:
        self.plot.hist = hist
        self.plot.update()

    def _on_finished(self, kind: str) -> None:
        if kind == "splits":
            self.editFinished.emit(_("Divisiones de la curva"))
        else:
            self.editFinished.emit(_("Curva de tonos {channel}").format(channel=channel_name(self.plot.channel)))

    def _menu(self, pos) -> None:
        menu = self.context_menu()
        menu.exec(self.plot.mapToGlobal(pos))
        menu.deleteLater()  # a new one is built on every right-click

    def context_menu(self) -> QMenu:
        channel = self.plot.channel
        menu = QMenu(self)
        menu.addAction(_("Restablecer curva {channel}").format(channel=channel_name(channel)),
                       lambda: self._reset([channel]))
        menu.addAction(_("Restablecer todas las curvas"), lambda: self._reset(list(CHANNELS)))
        return menu

    def _reset(self, channels: list[str]) -> None:
        changed = [c for c in channels if tuple(self.plot.curves[c]) != IDENTITY]
        for c in changed:
            self.plot.reset_channel(c)
            self.curveChanged.emit(c, IDENTITY)
        self.plot.update()
        if changed:
            self.editFinished.emit(_("Restablecer curva de tonos"))
