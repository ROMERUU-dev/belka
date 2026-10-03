"""Lightroom-style histogram that doubles as a tone control.

The width is split in the five zones Lightroom uses (blacks, shadows,
exposure, highlights, whites); dragging sideways over a zone moves that
slider, so the photographer can push the tones where they see them. The
triangles in the top corners warn about clipping and switch the clipping
overlay of the image view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from belka.i18n import _, language

# Fraction of a channel's pixels that must sit in the end bin before the
# triangle lights up: enough to ignore a few noisy pixels of a sampled
# histogram, small enough to warn about a clipped specular highlight.
CLIP_FRACTION = 1e-4

ZONES: tuple[tuple[str, float, float], ...] = (
    ("black", 0.0, 0.1),
    ("shadows", 0.1, 0.3),
    ("exposure", 0.3, 0.7),
    ("highlights", 0.7, 0.9),
    ("white", 0.9, 1.0),
)


@dataclass(frozen=True)
class FieldSpec:
    minimum: float
    maximum: float
    per_width: float  # change produced by a drag across the whole histogram
    step: float  # drag resolution: one unit of the slider's display
    sign: float = 1.0  # display = sign * model: -1 where the slider runs backwards


# Highlights/shadows sweep their whole -100..100 range across the width, and
# blacks/whites (stored as -0.3..0.3, shown as -100..100) do the same, so all
# four feel alike; exposure moves 4 EV per width. A positive model ``black``
# raises the black point (darker) while Lightroom's Negros +100 lifts the
# blacks, so its display and its drag run backwards, as on the slider.
FIELDS: dict[str, FieldSpec] = {
    "black": FieldSpec(-0.3, 0.3, 0.6, 0.003, sign=-1.0),
    "shadows": FieldSpec(-1.0, 1.0, 2.0, 0.01),
    "exposure": FieldSpec(-3.0, 3.0, 4.0, 0.01),
    "highlights": FieldSpec(-1.0, 1.0, 2.0, 0.01),
    "white": FieldSpec(-0.3, 0.3, 0.6, 0.003),
}

# Channel fills are added together, so overlaps turn yellow/cyan/magenta and
# the part where all three agree a light neutral grey, as in Lightroom.
_CHANNEL_FILLS = (QColor(122, 32, 32), QColor(32, 110, 38), QColor(34, 46, 124))
_CHANNEL_EDGES = (QColor(84, 18, 18), QColor(18, 74, 22), QColor(20, 32, 96))
_CLIP_LIT = (np.array([235, 64, 56]), np.array([70, 215, 80]), np.array([80, 120, 255]))
_WELL = QColor(22, 22, 22)
_WELL_EDGE = QColor(50, 50, 50)
_BASELINE = QColor(78, 78, 78)
_BAND = QColor(255, 255, 255, 13)
_BAND_DRAG = QColor(255, 255, 255, 22)
_TEXT = QColor(184, 184, 184)
_TEXT_DIM = QColor(128, 128, 128)
_TEXT_BRIGHT = QColor(236, 236, 236)
_CLIP_OFF = QColor(74, 74, 74)
_CLIP_HOVER = QColor(120, 120, 120)
_CLIP_FRAME = QColor(200, 200, 200)

_SIDE = 8.0
_TOP = 4.0
_FOOTER = 20.0
_CLIP_BOX = QSize(16, 13)


def field_label(field: str) -> str:
    return {
        "black": _("Negros"),
        "shadows": _("Sombras"),
        "exposure": _("Exposición"),
        "highlights": _("Altas luces"),
        "white": _("Blancos"),
    }[field]


def format_value(field: str, value: float) -> str:
    """The value as its slider shows it: EV with two decimals, others -100..100."""
    if field == "exposure":
        text = f"{value:+.2f}"
    else:
        spec = FIELDS[field]
        text = f"{round(spec.sign * value / spec.maximum * 100):+d}"
    if float(text) == 0.0:
        text = text[1:]
    text = text.replace("-", "\u2212")
    return text.replace(".", ",") if language() == "es" else text


def zone_label(field: str, value: float) -> str:
    """'Exposición +0,35': the hover text, also usable as the history step name."""
    return f"{field_label(field)} {format_value(field, value)}"


def zone_for(fraction: float) -> str:
    """Zone under a horizontal position given as a fraction of the histogram width."""
    for name, _lo, hi in ZONES:
        if fraction < hi:
            return name
    return ZONES[-1][0]


def clipped_channels(hist: np.ndarray) -> tuple[tuple[bool, bool, bool], tuple[bool, bool, bool]]:
    """(shadows, highlights) flags per R, G, B from the end bins of a (3, 256) histogram.

    A pixel clips when any one channel reaches 0 or 255, the rule of
    ``adjust.clipping_masks``: the overlay a lit triangle switches on must
    use it too, or a blue-only shadow clip lights a triangle with no overlay.
    """
    h = np.asarray(hist, dtype=np.float64)
    total = np.maximum(h.sum(axis=1), 1.0)
    low = h[:, 0] / total > CLIP_FRACTION
    high = h[:, -1] / total > CLIP_FRACTION
    return tuple(bool(v) for v in low), tuple(bool(v) for v in high)


def _display_curves(hist: np.ndarray) -> np.ndarray:
    """Bar heights in [0, 1]: lightly smoothed, scaled to the tallest interior bin.

    The end bins are left out of the scale (a clipped sky would flatten the
    rest) and out of the smoothing (so a clipping spike stays at the edge);
    the smoothing hides the comb an 8-bit tone curve leaves in the counts.
    """
    h = np.asarray(hist, dtype=np.float64)
    taps = np.exp(-0.5 * (np.arange(-3, 4) / 1.1) ** 2)
    taps /= taps.sum()
    inner = np.stack([np.convolve(np.pad(row, 3, mode="edge"), taps, mode="valid") for row in h[:, 1:-1]])
    out = np.empty_like(h)
    out[:, 1:-1] = inner
    out[:, 0], out[:, -1] = h[:, 0], h[:, -1]
    peak = inner.max()
    return np.clip(out / peak, 0.0, 1.0) if peak > 0 else np.zeros_like(h)


@dataclass
class _Drag:
    field: str
    x0: float
    start: float
    last: float


class InteractiveHistogram(QWidget):
    adjustRequested = Signal(str, float)  # (field, delta in model units)
    dragFinished = Signal(str)  # field, once per drag that changed the value
    clippingToggled = Signal(str, bool)  # "shadows" | "highlights", new state

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._curves: np.ndarray | None = None
        self._clipped: tuple[tuple[bool, ...], tuple[bool, ...]] = ((False,) * 3, (False,) * 3)
        self._show_clipping = {"shadows": False, "highlights": False}
        self._values = {name: 0.0 for name in FIELDS}
        self._inert: frozenset[str] = frozenset()
        self._readout = ""
        self._info = ""
        self._hover_zone: str | None = None
        self._hover_clip: str | None = None
        self._drag: _Drag | None = None
        self._font = QFont(self.font())
        self._font.setPixelSize(11)

    def sizeHint(self) -> QSize:
        return QSize(280, 120)

    def minimumSizeHint(self) -> QSize:
        return QSize(160, 120)

    # ------------------------------------------------------------ public API

    def set_histogram(self, hist: np.ndarray | None) -> None:
        """(3, 256) counts; None (no photo) also makes the zones inert, as in Lightroom."""
        counts = None if hist is None else np.asarray(hist)
        if counts is None or counts.shape != (3, 256) or not counts.any():
            self._curves = None
            self._clipped = ((False,) * 3, (False,) * 3)
            self._end_drag()
            self._hover_zone = None
            if self._hover_clip is None:
                self.unsetCursor()
        else:
            self._curves = _display_curves(counts)
            self._clipped = clipped_channels(counts)
        self.update()

    def set_values(self, values: dict[str, float]) -> None:
        """Model values to show; the dragged field keeps its own, since renders lag the drag."""
        dragged = self._drag.field if self._drag else None
        for name in FIELDS:
            if name in values and name != dragged:
                self._values[name] = float(values[name])
        self.update()

    def set_inert_zones(self, zones) -> None:
        """Zones whose slider does nothing for this photo (the flat output has no
        Sombras or Altas luces): hovering says so, and they do not drag."""
        zones = frozenset(zones)
        if zones != self._inert:
            self._inert = zones
            if self._drag is not None and self._drag.field in zones:
                self._end_drag()
            self._hover_zone = self._hover_clip = None
            self.update()

    def set_readout(self, text: str) -> None:
        self._readout = text
        self.update()

    def set_info(self, text: str) -> None:
        self._info = text
        self.update()

    def set_clipping(self, shadows: bool, highlights: bool) -> None:
        """Mirror the overlay state (e.g. after the J shortcut) without emitting."""
        self._show_clipping = {"shadows": bool(shadows), "highlights": bool(highlights)}
        self.update()

    def clipping(self) -> tuple[bool, bool]:
        """Whether the image clips in the (shadows, highlights), any channel."""
        return any(self._clipped[0]), any(self._clipped[1])

    def zone_at(self, x: float) -> str:
        graph = self._graph_rect()
        return zone_for((x - graph.left()) / max(graph.width(), 1.0))

    # ------------------------------------------------------------ geometry

    def _graph_rect(self) -> QRectF:
        return QRectF(_SIDE, _TOP, max(self.width() - 2 * _SIDE, 1.0), max(self.height() - _TOP - _FOOTER, 1.0))

    def _clip_rect(self, which: str) -> QRectF:
        graph = self._graph_rect()
        w, h = _CLIP_BOX.width(), _CLIP_BOX.height()
        x = graph.left() + 2 if which == "shadows" else graph.right() - 2 - w
        return QRectF(x, graph.top() + 2, w, h)

    def _clip_at(self, pos: QPointF) -> str | None:
        for which in ("shadows", "highlights"):
            if self._clip_rect(which).adjusted(-2, -2, 2, 2).contains(pos):
                return which
        return None

    # ------------------------------------------------------------ mouse

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        pos = event.position()
        clip = self._clip_at(pos)
        if clip:
            self._show_clipping[clip] = not self._show_clipping[clip]
            self.update()
            self.clippingToggled.emit(clip, self._show_clipping[clip])
            return
        self._end_drag()
        if self._curves is None:
            return
        field = self.zone_at(pos.x())
        if field in self._inert:
            return
        value = self._values[field]
        self._drag = _Drag(field, pos.x(), value, value)
        self._hover_zone = field
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._drag_to(event.position().x())
                return
            # The release went elsewhere (a dialog, a lost grab): close the
            # drag rather than keep editing on a plain hover.
            self._end_drag()
        self._update_hover(event.position())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._drag is None:
            super().mouseReleaseEvent(event)
            return
        self._end_drag()
        self._update_hover(event.position())

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click on a zone resets that slider, like double-clicking its label."""
        pos = event.position()
        if event.button() != Qt.MouseButton.LeftButton or self._clip_at(pos) or self._curves is None:
            super().mouseDoubleClickEvent(event)
            return
        field = self.zone_at(pos.x())
        if field in self._inert:
            return
        value = self._values[field]
        if value != 0.0:
            self._values[field] = 0.0
            self.update()
            self.adjustRequested.emit(field, -value)
            self.dragFinished.emit(field)

    def leaveEvent(self, event) -> None:
        if self._drag is None:
            self._hover_zone = self._hover_clip = None
            self.unsetCursor()
            self.update()
        super().leaveEvent(event)

    def hideEvent(self, event) -> None:
        # A hidden widget never sees the release; nor does a disabled one.
        self._end_drag()
        self._hover_zone = self._hover_clip = None
        self.unsetCursor()
        super().hideEvent(event)

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.Type.EnabledChange and not self.isEnabled():
            self._end_drag()
        super().changeEvent(event)

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            clip = self._clip_at(QPointF(event.pos()))
            if clip == "shadows":
                QToolTip.showText(event.globalPos(), _("Mostrar recorte de sombras"), self)
            elif clip == "highlights":
                QToolTip.showText(event.globalPos(), _("Mostrar recorte de altas luces"), self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)

    def _end_drag(self) -> None:
        """Close the drag, as one history step if it changed the value."""
        drag, self._drag = self._drag, None
        if drag is None:
            return
        self.update()
        if drag.last != drag.start:
            self.dragFinished.emit(drag.field)

    def _drag_to(self, x: float) -> None:
        drag = self._drag
        spec = FIELDS[drag.field]
        steps = round((x - drag.x0) / self._graph_rect().width() * spec.per_width / spec.step)
        # A value already past the range (the exposure slider reaches ±5 EV)
        # is not pulled to the limit by the first pixel of the drag.
        lo, hi = min(spec.minimum, drag.start), max(spec.maximum, drag.start)
        target = round(min(max(drag.start + spec.sign * steps * spec.step, lo), hi), 6)
        delta = target - drag.last
        if abs(delta) < 1e-9:
            return
        drag.last = target
        self._values[drag.field] = target
        self.update()
        self.adjustRequested.emit(drag.field, delta)

    def _update_hover(self, pos: QPointF) -> None:
        inside = QRectF(self.rect()).contains(pos)  # a drag can end outside the widget
        clip = self._clip_at(pos)
        zone = self.zone_at(pos.x()) if inside and not clip and self._curves is not None else None
        if (clip, zone) == (self._hover_clip, self._hover_zone):
            return
        self._hover_clip, self._hover_zone = clip, zone
        if clip:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif zone and zone not in self._inert:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.unsetCursor()
        self.update()

    # ------------------------------------------------------------ painting

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)
        graph = self._graph_rect()
        painter.setPen(QPen(_WELL_EDGE, 1.0))
        painter.setBrush(_WELL)
        painter.drawRoundedRect(graph.adjusted(-0.5, -0.5, 0.5, 0.5), 2.0, 2.0)
        zone = self._drag.field if self._drag else self._hover_zone
        if zone and self.isEnabled():
            lo, hi = next((lo, hi) for name, lo, hi in ZONES if name == zone)
            band = QRectF(graph.left() + lo * graph.width(), graph.top(), (hi - lo) * graph.width(), graph.height())
            painter.fillRect(band, _BAND_DRAG if self._drag else _BAND)
        if self._curves is not None:
            self._paint_curves(painter, graph)
        painter.setPen(QPen(_BASELINE, 1.0))
        painter.drawLine(QPointF(graph.left(), graph.bottom() - 0.5), QPointF(graph.right(), graph.bottom() - 0.5))
        for which, flags in (("shadows", self._clipped[0]), ("highlights", self._clipped[1])):
            self._paint_clip(painter, which, flags)
        self._paint_text(painter, graph, zone)

    def _paint_curves(self, painter: QPainter, graph: QRectF) -> None:
        bins = self._curves.shape[1]
        xs = graph.left() + graph.width() * np.arange(bins) / (bins - 1)
        # Headroom keeps the tallest peak clear of the triangles and the info line.
        height = (graph.height() - 2.0) * 0.86
        bottom = graph.bottom() - 1.0
        painter.save()
        painter.setClipRect(graph)
        if not self.isEnabled():
            painter.setOpacity(0.45)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for curve, fill, edge in zip(self._curves, _CHANNEL_FILLS, _CHANNEL_EDGES):
            ys = bottom - height * curve
            top = QPainterPath(QPointF(xs[0], ys[0]))
            for x, y in zip(xs[1:], ys[1:]):
                top.lineTo(QPointF(x, y))
            area = QPainterPath(top)
            area.lineTo(QPointF(graph.right(), bottom))
            area.lineTo(QPointF(graph.left(), bottom))
            area.closeSubpath()
            painter.fillPath(area, fill)
            painter.strokePath(top, QPen(edge, 1.0))
        painter.restore()

    def _paint_clip(self, painter: QPainter, which: str, flags: tuple[bool, ...]) -> None:
        box = self._clip_rect(which)
        if any(flags):
            rgb = np.minimum(sum(c for c, on in zip(_CLIP_LIT, flags) if on), 255)
            color = QColor(*(int(v) for v in rgb))
        else:
            color = _CLIP_HOVER if self._hover_clip == which else _CLIP_OFF
        if not self.isEnabled():
            color = _CLIP_OFF
        if self._show_clipping[which]:
            painter.setPen(QPen(_CLIP_FRAME, 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), 2.0, 2.0)
        # A corner triangle whose long side faces the graph.
        size = 7.0
        cy = box.top() + 3.0
        if which == "shadows":
            cx = box.left() + 3.0
            points = (QPointF(cx, cy), QPointF(cx + size, cy), QPointF(cx, cy + size))
        else:
            cx = box.right() - 3.0
            points = (QPointF(cx, cy), QPointF(cx - size, cy), QPointF(cx, cy + size))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF(points))

    def _paint_text(self, painter: QPainter, graph: QRectF, zone: str | None) -> None:
        metrics = QFontMetricsF(self._font)
        if self._info:
            left = self._clip_rect("shadows").right() + 4
            room = self._clip_rect("highlights").left() - 4 - left
            text = metrics.elidedText(self._info, Qt.TextElideMode.ElideRight, room)
            base = graph.top() + 2 + metrics.ascent()
            # A dark halo keeps the line legible over a tall histogram.
            painter.setPen(QColor(0, 0, 0, 170))
            painter.drawText(QPointF(left + 1, base + 1), text)
            painter.setPen(_TEXT_DIM)
            painter.drawText(QPointF(left, base), text)
        footer = QRectF(graph.left(), graph.bottom(), graph.width(), self.height() - graph.bottom())
        base = footer.center().y() + (metrics.ascent() - metrics.descent()) / 2
        if zone and self.isEnabled():
            name = field_label(zone) + "  "
            inert = zone in self._inert
            value = _("no se aplica a la salida plana") if inert else format_value(zone, self._values[zone])
            x = footer.center().x() - (metrics.horizontalAdvance(name) + metrics.horizontalAdvance(value)) / 2
            painter.setPen(_TEXT)
            painter.drawText(QPointF(x, base), name)
            painter.setPen(_TEXT_DIM if inert else _TEXT_BRIGHT)
            painter.drawText(QPointF(x + metrics.horizontalAdvance(name), base), value)
        elif self._readout:
            text = metrics.elidedText(self._readout, Qt.TextElideMode.ElideRight, footer.width())
            painter.setPen(_TEXT)
            painter.drawText(QPointF(footer.center().x() - metrics.horizontalAdvance(text) / 2, base), text)
