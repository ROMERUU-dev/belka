"""Guided Upright: the guide lines and their editing, the loupe over a dragged endpoint, the transform grid.

Guides are kept normalised to the displayed image, like the crop frame, so they stay on the
same picture features when a render of another size arrives. Hit-testing and painting work in
viewport pixels, where tolerances and handle sizes mean what the user sees.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap

from belka.ui.crop_overlay import LINE

Guide = tuple[float, float, float, float]  # (x0, y0, x1, y1), normalised to the displayed image
Hit = tuple[int, int | None]  # (guide index, endpoint 0/1, or None for the line itself)
Segment = tuple[QPointF, QPointF]

MAX_GUIDES = 4
END_TOLERANCE = 8.0  # screen px around an endpoint
LINE_TOLERANCE = 5.0  # screen px either side of a line
MIN_LENGTH = 6.0  # screen px: a shorter new guide was a click
LOUPE_SIZE = 132.0  # screen px, square
LOUPE_GAP = 20.0  # screen px between the endpoint and the loupe's corner
LOUPE_FACTOR = 3.0  # magnification over the view...
LOUPE_MIN_SCALE = 2.0  # ...but always at least this many screen px per image px
GRID_CELLS = 12  # square cells along the long side

GUIDE = QColor(246, 200, 40)
GUIDE_LIT = QColor(255, 228, 112)
HALO = QColor(0, 0, 0, 165)
HANDLE_HOLE = QColor(24, 24, 24, 230)
GRID_LIGHT = QColor(255, 255, 255, 72)
GRID_DARK = QColor(0, 0, 0, 52)


def _guide(values: Sequence[float]) -> Guide:
    x0, y0, x1, y1 = (float(v) for v in values)
    return x0, y0, x1, y1


def _shift(d: float, lo: float, hi: float) -> float:
    """``d`` limited so coordinates spanning lo..hi stay inside 0..1; ones already outside (a guide
    remapped past the edge) are never pushed further out, nor jump in."""
    return min(max(d, min(0.0, -lo)), max(0.0, 1.0 - hi))


class GuideEditor:
    """The guides and the drag that edits them, with no widget attached.

    A drag is replayed from the guides it started on and the pointer's travel, so a remapped
    list arriving mid-drag (:meth:`set`, after a render) moves the start without losing the drag,
    and grabbing a handle off-centre never makes it jump to the pointer.
    """

    def __init__(self) -> None:
        self.guides: list[Guide] = []
        self.selected: int | None = None
        self._before: list[Guide] = []
        self._edit: tuple[int | None, int | None] | None = None  # (index or None for a new guide, end or None)
        self._anchor = (0.0, 0.0)  # pointer at the press; a new guide starts there
        self._pointer = (0.0, 0.0)

    def set(self, guides: Iterable[Sequence[float]]) -> bool:
        """Replace the guides. A drag carries on from its guide's new place; False when it had to end."""
        new = [_guide(g) for g in guides][:MAX_GUIDES]
        alive = True
        if self._edit is not None:
            index = self._edit[0]
            alive = len(new) < MAX_GUIDES if index is None else index < len(new)
            if not alive:
                self._edit = None
        if self.selected is not None and self.selected >= len(new):
            self.selected = None
        self._before = new
        if self._edit is None:
            self.guides = list(new)
        else:
            self._replay()
        return alive

    @property
    def dragging(self) -> bool:
        return self._edit is not None

    @property
    def creating(self) -> bool:
        return self._edit is not None and self._edit[0] is None

    def active(self) -> Hit | None:
        """(index in :attr:`guides`, end) of what is being dragged; end None moves the whole line."""
        if self._edit is None:
            return None
        index, end = self._edit
        return (len(self._before) if index is None else index), end

    def begin(self, hit: Hit | None, point: tuple[float, float]) -> bool:
        """Drag the ``hit`` part, or draw a new guide from ``point`` (normalised, clamped to the image).

        False, changing nothing but the selection, when there is no room for another guide."""
        if hit is None and len(self.guides) >= MAX_GUIDES:
            self.selected = None
            return False
        if hit is None:
            point = (min(max(point[0], 0.0), 1.0), min(max(point[1], 0.0), 1.0))
        self._before = list(self.guides)
        self._edit = (None, 1) if hit is None else hit
        self._anchor = self._pointer = point
        self.selected = None if hit is None else hit[0]
        self._replay()
        return True

    def drag(self, point: tuple[float, float]) -> None:
        if self._edit is not None:
            self._pointer = point
            self._replay()

    def finish(self) -> bool:
        """End the drag keeping its result (and selecting its guide); True when the guides changed."""
        active = self.active()
        self._edit = None
        if active is not None:
            self.selected = active[0]
        return self.guides != self._before

    def cancel(self) -> None:
        """End the drag putting the guides back as it found them (Esc, or a new guide that was a click)."""
        self._edit = None
        self.guides = list(self._before)
        if self.selected is not None and self.selected >= len(self.guides):
            self.selected = None

    def remove(self, index: int) -> None:
        del self.guides[index]
        self.selected = None

    def _replay(self) -> None:
        index, end = self._edit
        ax, ay = self._anchor
        dx, dy = self._pointer[0] - ax, self._pointer[1] - ay
        guides = list(self._before)
        if index is None:
            guides.append((ax, ay, ax + _shift(dx, ax, ax), ay + _shift(dy, ay, ay)))
        elif end is None:
            x0, y0, x1, y1 = guides[index]
            dx, dy = _shift(dx, min(x0, x1), max(x0, x1)), _shift(dy, min(y0, y1), max(y0, y1))
            guides[index] = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        else:
            g = list(guides[index])
            x, y = g[2 * end], g[2 * end + 1]
            g[2 * end:2 * end + 2] = (x + _shift(dx, x, x), y + _shift(dy, y, y))
            guides[index] = _guide(g)
        self.guides = guides


def _distance_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
    d = b - a
    length2 = QPointF.dotProduct(d, d)
    t = 0.0 if length2 == 0 else min(max(QPointF.dotProduct(p - a, d) / length2, 0.0), 1.0)
    return QLineF(p, a + d * t).length()


def hit_test(segments: Sequence[Segment], pos: QPointF) -> Hit | None:
    """Guide part under ``pos`` (viewport px): the nearest endpoint within reach wins over any line,
    so two guides meeting at a corner can still be pulled apart."""
    best: tuple[float, int, int | None] | None = None
    for i, (a, b) in enumerate(segments):
        for end, q in enumerate((a, b)):
            d = QLineF(q, pos).length()
            if d <= END_TOLERANCE and (best is None or d <= best[0]):
                best = (d, i, end)
    if best is None:
        for i, (a, b) in enumerate(segments):
            d = _distance_to_segment(pos, a, b)
            if d <= LINE_TOLERANCE and (best is None or d <= best[0]):
                best = (d, i, None)
    return None if best is None else (best[1], best[2])


def _stroke(p: QPainter, path: QPainterPath, colour: QColor, width: float) -> None:
    cap, join = Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
    p.strokePath(path, QPen(HALO, width + 2.0, Qt.PenStyle.SolidLine, cap, join))
    p.strokePath(path, QPen(colour, width, Qt.PenStyle.SolidLine, cap, join))


def paint_guides(p: QPainter, segments: Sequence[Segment], selected: int | None, hot: Hit | None) -> None:
    """Yellow lines with round handles. ``hot`` is the part under the pointer or being dragged: its
    guide is lit and its endpoint grows. The selected guide's handles are rings (Delete removes it)."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    hot_index = None if hot is None else hot[0]
    for i in sorted(range(len(segments)), key=lambda k: (k == selected, k == hot_index)):  # lit ones on top
        a, b = segments[i]
        lit = i in (selected, hot_index)
        colour = GUIDE_LIT if lit else GUIDE
        line = QPainterPath(a)
        line.lineTo(b)
        _stroke(p, line, colour, 2.2 if lit else 1.5)
        for end, q in enumerate((a, b)):
            r = 5.0 if hot == (i, end) else 3.6
            dot = QPainterPath()
            dot.addEllipse(q, r, r)
            p.strokePath(dot, QPen(HALO, 2.4))
            if i == selected:
                p.fillPath(dot, HANDLE_HOLE)
                p.strokePath(dot, QPen(colour, 1.6))
            else:
                p.fillPath(dot, colour)
    p.restore()


def loupe_scale(zoom: float) -> float:
    """Screen px per image px inside the loupe."""
    return max(LOUPE_FACTOR * zoom, LOUPE_MIN_SCALE)


def loupe_rect(at: QPointF, other: QPointF, bounds: QRectF) -> QRectF:
    """Square beside the endpoint ``at`` on the diagonal facing away from ``other`` (the line's other
    end), so it keeps off the line being placed where there is room; inside ``bounds`` whenever it fits."""
    half = LOUPE_SIZE / 2
    reach = LOUPE_GAP + half
    toward = other - at
    best: tuple[tuple[bool, float], QRectF] | None = None
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):  # up-left when the line has no length yet
        c = at + QPointF(sx * reach, sy * reach)
        r = QRectF(c.x() - half, c.y() - half, LOUPE_SIZE, LOUPE_SIZE)
        score = (not bounds.contains(r), sx * toward.x() + sy * toward.y())
        if best is None or score < best[0]:
            best = (score, r)
    r = best[1]
    dx = min(max(r.left(), bounds.left()), bounds.right() - r.width()) - r.left()
    dy = min(max(r.top(), bounds.top()), bounds.bottom() - r.height()) - r.top()
    return r.translated(dx, dy)


def paint_loupe(p: QPainter, rect: QRectF, pixmap: QPixmap, centre: QPointF, scale: float,
                lines: Sequence[Segment], active: int, background: QColor) -> None:
    """The pixels around ``centre`` (image px) magnified ``scale`` times in ``rect``, with the guides
    (image px) drawn through them and a ring on the endpoint being placed."""
    c = rect.center()

    def to_loupe(q: QPointF) -> QPointF:
        return c + (q - centre) * scale

    frame = QPainterPath()
    frame.addRoundedRect(rect, 4, 4)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.fillPath(frame, background)
    p.setClipPath(frame)
    p.save()
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.translate(c)
    p.scale(scale, scale)
    p.translate(-centre)
    p.drawPixmap(QPointF(0, 0), pixmap)
    p.restore()
    for i, (a, b) in enumerate(lines):
        line = QPainterPath(to_loupe(a))
        line.lineTo(to_loupe(b))
        _stroke(p, line, GUIDE_LIT if i == active else GUIDE, 1.2)
    ring = QPainterPath()
    ring.addEllipse(c, 5.5, 5.5)
    _stroke(p, ring, GUIDE_LIT, 1.2)
    p.setClipping(False)
    p.strokePath(frame, QPen(HALO, 3.0))
    p.strokePath(frame, QPen(LINE, 1.0))
    p.restore()


def paint_grid(p: QPainter, image: QRectF, width: int, height: int, clip: QRectF, dpr: float) -> None:
    """Lightroom's transform grid over ``image`` (its rectangle on screen, ``width`` x ``height`` image
    px): square cells, :data:`GRID_CELLS` along the long side, centred on the photo. Hairlines on
    device pixels, light over a dark shadow so they read on bright and dark film alike."""
    cell = max(width, height) / GRID_CELLS * image.width() / max(width, 1)  # screen px
    area = image.intersected(clip)
    if cell < 6 or area.isEmpty():
        return

    def lines(centre: float, lo: float, hi: float) -> list[float]:
        k0, k1 = math.ceil((lo - centre) / cell), math.floor((hi - centre) / cell)
        return [v for k in range(k0, k1 + 1) if lo + 0.5 < (v := centre + k * cell) < hi - 0.5]

    def snap(v: float) -> float:
        return (math.floor(v * dpr) + 0.5) / dpr

    xs = lines(image.center().x(), area.left(), area.right())
    ys = lines(image.center().y(), area.top(), area.bottom())
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    for colour, off in ((GRID_DARK, 1.0 / dpr), (GRID_LIGHT, 0.0)):
        p.setPen(QPen(colour, 0))  # cosmetic: one device pixel
        for x in xs:
            p.drawLine(QPointF(snap(x) + off, area.top()), QPointF(snap(x) + off, area.bottom()))
        for y in ys:
            p.drawLine(QPointF(area.left(), snap(y) + off), QPointF(area.right(), snap(y) + off))
    p.restore()
