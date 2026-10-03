"""Lightroom-style crop frame: dimmed surround, corner and edge handles, guide grids.

The frame is stored in normalised image coordinates so it survives a re-render at
another preview size; the drag geometry works in image pixels, where an aspect
ratio means what the user sees. Everything is painted in device pixels so lines
and handles stay crisp and the same size at any zoom.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem

Rect = tuple[float, float, float, float]  # (x0, y0, x1, y1), normalised

GRIDS = ("none", "thirds", "fine")
MIN_SIZE = 0.02  # of the image's shorter side
HANDLE_TOLERANCE = 8.0  # screen px
SHADE = QColor(0, 0, 0, 160)
LINE = QColor(236, 236, 236, 235)
OUTLINE = QColor(0, 0, 0, 120)

_SHAPES = {
    "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
    "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
    "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
    "move": Qt.CursorShape.OpenHandCursor,
}


def draw_cursor(paint: Callable[[QPainter], None], hot_x: int, hot_y: int, size: int = 32) -> QCursor:
    """A cursor painted in logical pixels at the screen's density, so it is sharp on HiDPI."""
    screen = QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    pix = QPixmap(round(size * dpr), round(size * dpr))
    pix.setDevicePixelRatio(dpr)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    paint(painter)
    painter.end()
    return QCursor(pix, hot_x, hot_y)


def outlined(painter: QPainter, path: QPainterPath, width: float = 1.6, fill: QColor | None = None) -> None:
    """Light stroke over a dark halo: readable on any part of a photo."""
    painter.strokePath(path, QPen(QColor(0, 0, 0, 200), width + 2.0, Qt.PenStyle.SolidLine,
                                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    if fill is not None:
        painter.fillPath(path, fill)
    painter.strokePath(path, QPen(QColor(242, 242, 242), width, Qt.PenStyle.SolidLine,
                                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))


def _paint_rotate(p: QPainter) -> None:
    cx, cy, r = 16.0, 22.0, 10.0
    path = QPainterPath()
    path.arcMoveTo(QRectF(cx - r, cy - r, 2 * r, 2 * r), 32)
    path.arcTo(QRectF(cx - r, cy - r, 2 * r, 2 * r), 32, 116)
    for deg, sign in ((32.0, 1.0), (148.0, -1.0)):
        a = math.radians(deg)
        tip = QPointF(cx + r * math.cos(a), cy - r * math.sin(a))
        d = QPointF(sign * math.sin(a), sign * math.cos(a))  # tangent leaving the arc at this end
        n = QPointF(-d.y(), d.x())
        apex = tip + d * 2.0
        path.moveTo(tip)
        path.lineTo(apex)
        path.moveTo(apex - d * 3.5 + n * 3.5)
        path.lineTo(apex)
        path.lineTo(apex - d * 3.5 - n * 3.5)
    outlined(p, path)


@lru_cache(maxsize=None)
def rotate_cursor() -> QCursor:
    return draw_cursor(_paint_rotate, 16, 16)


def cursor_for(part: str) -> QCursor:
    """Cursor for a hit-test result of :meth:`CropOverlay.hit`."""
    if part == "rotate":
        return rotate_cursor()
    return QCursor(_SHAPES.get(part, Qt.CursorShape.ArrowCursor))


def _clamp_into(r: QRectF, b: QRectF) -> QRectF:
    """Shrink about the centre (keeping the shape) until ``r`` fits ``b``, then slide it inside."""
    s = min(1.0, b.width() / max(r.width(), 1e-9), b.height() / max(r.height(), 1e-9))
    w, h = r.width() * s, r.height() * s
    c = r.center()
    x = min(max(c.x() - w / 2, b.left()), b.right() - w)
    y = min(max(c.y() - h / 2, b.top()), b.bottom() - h)
    return QRectF(x, y, w, h)


def _fit_aspect(r: QRectF, aspect: float) -> QRectF:
    """Largest rectangle of ``aspect`` (w/h) inside ``r``, sharing its centre."""
    if r.width() / max(r.height(), 1e-9) > aspect:
        w, h = r.height() * aspect, r.height()
    else:
        w, h = r.width(), r.width() / aspect
    c = r.center()
    return QRectF(c.x() - w / 2, c.y() - h / 2, w, h)


def _oriented(aspect: float, r: QRectF) -> float:
    """Lightroom applies a ratio in the crop's own orientation: 3:2 on a portrait crop is 2:3."""
    if abs(r.width() - r.height()) > 1e-6 and (r.width() < r.height()) != (aspect < 1.0):
        return 1.0 / aspect
    return aspect


def _arm(width: float, height: float) -> int:
    """Length in screen px of each arm of a corner bracket (and of the edge bars) on a frame this size."""
    return max(8, min(22, int(width) // 4, int(height) // 4))


class CropOverlay(QGraphicsItem):
    """Crop frame over the image item (placed at the scene origin, one unit per image pixel).

    The frame the user chose is kept apart from the one shown: bounds only shrink
    the shown frame, so rotating out and back (or any warp that grows the valid
    area again) restores it, as in Lightroom, even when the renders arrive late.
    """

    def __init__(self) -> None:
        super().__init__()
        self._w = 1.0
        self._h = 1.0
        self._rect = QRectF(0, 0, 1, 1)  # normalised, as shown: the chosen frame clamped into the bounds
        self._chosen = QRectF(0, 0, 1, 1)  # normalised, as last set by a drag, set_rect, an aspect or X
        self._bounds = QRectF(0, 0, 1, 1)  # normalised
        self._aspect: float | None = None  # image-pixel w/h, already oriented
        self._grid = "none"
        self._drag: tuple[str, QRectF, QPointF] | None = None
        self.setZValue(20)

    # ------------------------------------------------------------ state
    def set_image_size(self, width: float, height: float) -> None:
        if (width, height) != (self._w, self._h):
            self.prepareGeometryChange()
            self._w, self._h = float(max(width, 1)), float(max(height, 1))

    def rect(self) -> Rect:
        r = self._rect
        return (r.left(), r.top(), r.right(), r.bottom())

    def set_rect(self, rect: Rect | None) -> None:
        """Set the frame; ``None`` selects the whole allowed area. Bounds and a locked ratio still apply."""
        if rect is None:
            r = self._px(self._bounds)
        else:
            x0, y0, x1, y1 = rect
            r = self._px(QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized())
        if self._aspect is not None:
            self._aspect = _oriented(self._aspect, r)
            if abs(r.width() / max(r.height(), 1e-9) / self._aspect - 1.0) > 0.005:
                r = _fit_aspect(r, self._aspect)
        self._choose(r)

    def bounds(self) -> Rect:
        b = self._bounds
        return (b.left(), b.top(), b.right(), b.bottom())

    def set_bounds(self, rect: Rect | None) -> None:
        """Area the frame must stay inside (constrain to the valid image after a warp).

        The chosen frame is clamped, not the shown one, so wider bounds give back what narrower ones took.
        """
        if rect is None:
            self._bounds = QRectF(0, 0, 1, 1)
        else:
            x0, y0, x1, y1 = rect
            self._bounds = QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized().intersected(QRectF(0, 0, 1, 1))
        self._show()

    def aspect(self) -> float | None:
        return self._aspect

    def set_aspect(self, ratio: float | None) -> None:
        """Lock width/height (in image pixels) or unlock with ``None``; the frame is refitted inside itself."""
        if ratio is None or ratio <= 0:
            self._aspect = None
            return
        r = self._px(self._rect)
        self._aspect = _oriented(float(ratio), r)
        self._choose(_fit_aspect(r, self._aspect))

    def swap_orientation(self) -> None:
        """Portrait <-> landscape about the centre, like Lightroom's X."""
        r = self._px(self._rect)
        if self._aspect is not None:
            self._aspect = 1.0 / self._aspect
        c = r.center()
        swapped = QRectF(c.x() - r.height() / 2, c.y() - r.width() / 2, r.height(), r.width())
        self._choose(swapped)

    def set_grid(self, grid: str) -> None:
        grid = grid if grid in GRIDS else "none"
        if grid != self._grid:
            self._grid = grid
            self.update()

    # ------------------------------------------------------------ interaction (image pixels)
    def hit(self, pos: QPointF, zoom: float, tolerance: float = HANDLE_TOLERANCE) -> str:
        """Part under ``pos`` (image pixels) seen at ``zoom`` screen px per image px, ``tolerance`` in
        screen px: a handle ``"tl"``, ``"t"``, ``"tr"``, ``"r"``, ``"br"``, ``"b"``, ``"bl"``, ``"l"``;
        ``"move"`` inside; ``"rotate"`` outside."""
        r = self._px(self._rect)
        x, y = pos.x(), pos.y()
        tol = tolerance / zoom
        corner = tol * 1.6  # corners are the handles people reach for: give them more room
        # Along the edge band, the whole drawn bracket belongs to its corner.
        arm = _arm(r.width() * zoom, r.height() * zoom) / zoom
        if not (r.left() - corner <= x <= r.right() + corner and r.top() - corner <= y <= r.bottom() + corner):
            return "rotate"
        dl, dr, dt, db = abs(x - r.left()), abs(x - r.right()), abs(y - r.top()), abs(y - r.bottom())
        dx, dy = min(dl, dr), min(dt, db)
        h, v = ("l" if dl <= dr else "r"), ("t" if dt <= db else "b")
        if (dx <= corner and dy <= corner) or (dx <= tol and dy <= arm) or (dy <= tol and dx <= arm):
            return v + h
        if dx <= tol or dy <= tol:
            return v if dy <= tol else h
        return "move" if r.contains(pos) else "rotate"

    def begin_drag(self, part: str, pos: QPointF) -> None:
        self._drag = (part, self._px(self._rect), QPointF(pos))

    def drag_to(self, pos: QPointF) -> None:
        if self._drag is None:
            return
        part, start, origin = self._drag
        d = pos - origin
        bounds = self._px(self._bounds)
        if part == "move":
            r = _clamp_into(start.translated(d), bounds)
        elif self._aspect is None:
            r = self._resize_free(start, part, d, bounds)
        else:
            r = self._resize_locked(start, part, d, bounds)
        self._choose(r)

    def end_drag(self) -> None:
        self._drag = None

    def _min_size(self) -> float:
        return max(1.0, MIN_SIZE * min(self._w, self._h))

    def _resize_free(self, s: QRectF, part: str, d: QPointF, b: QRectF) -> QRectF:
        m = self._min_size()
        x0, y0, x1, y1 = s.left(), s.top(), s.right(), s.bottom()
        if "l" in part:
            x0 = min(max(x0 + d.x(), b.left()), x1 - m)
        if "r" in part:
            x1 = max(min(x1 + d.x(), b.right()), x0 + m)
        if "t" in part:
            y0 = min(max(y0 + d.y(), b.top()), y1 - m)
        if "b" in part:
            y1 = max(min(y1 + d.y(), b.bottom()), y0 + m)
        return QRectF(QPointF(x0, y0), QPointF(x1, y1))

    def _resize_locked(self, s: QRectF, part: str, d: QPointF, b: QRectF) -> QRectF:
        a = self._aspect
        min_w = self._min_size() * max(1.0, a)
        if len(part) == 2:
            # Corner: the opposite corner stays put; the size follows the
            # pointer projected onto the frame's diagonal.
            sx = 1.0 if "r" in part else -1.0
            sy = 1.0 if "b" in part else -1.0
            ax = s.left() if sx > 0 else s.right()
            ay = s.top() if sy > 0 else s.bottom()
            w = sx * ((s.right() if sx > 0 else s.left()) + d.x() - ax)
            h = sy * ((s.bottom() if sy > 0 else s.top()) + d.y() - ay)
            w = (w * a + h) / (a * a + 1.0) * a
            room_x = b.right() - ax if sx > 0 else ax - b.left()
            room_y = b.bottom() - ay if sy > 0 else ay - b.top()
            w = min(max(w, min_w), room_x, room_y * a)
            h = w / a
            return QRectF(ax if sx > 0 else ax - w, ay if sy > 0 else ay - h, w, h)
        c = s.center()
        if part in ("l", "r"):
            # Edge: the opposite edge stays put, the frame grows about its middle.
            ax = s.left() if part == "r" else s.right()
            w = (s.right() + d.x() - ax) if part == "r" else (ax - (s.left() + d.x()))
            room = b.right() - ax if part == "r" else ax - b.left()
            w = min(max(w, min_w), room, 2.0 * min(c.y() - b.top(), b.bottom() - c.y()) * a)
            h = w / a
            return QRectF(ax if part == "r" else ax - w, c.y() - h / 2, w, h)
        ay = s.top() if part == "b" else s.bottom()
        h = (s.bottom() + d.y() - ay) if part == "b" else (ay - (s.top() + d.y()))
        room = b.bottom() - ay if part == "b" else ay - b.top()
        h = min(max(h, min_w / a), room, 2.0 * min(c.x() - b.left(), b.right() - c.x()) / a)
        w = h * a
        return QRectF(c.x() - w / 2, ay if part == "b" else ay - h, w, h)

    # ------------------------------------------------------------ conversions
    def _px(self, r: QRectF) -> QRectF:
        return QRectF(r.left() * self._w, r.top() * self._h, r.width() * self._w, r.height() * self._h)

    def _norm(self, r: QRectF) -> QRectF:
        return QRectF(r.left() / self._w, r.top() / self._h, r.width() / self._w, r.height() / self._h)

    def _choose(self, r: QRectF) -> None:
        """Make ``r`` (image pixels) the chosen frame and show it inside the bounds."""
        self._chosen = self._norm(r)
        self._show()

    def _show(self) -> None:
        self._rect = self._norm(_clamp_into(self._px(self._chosen), self._px(self._bounds)))
        self.update()

    def crop_px(self) -> QRectF:
        """The frame in image pixels (scene coordinates of the view)."""
        return self._px(self._rect)

    # ------------------------------------------------------------ painting
    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._w, self._h)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        t = painter.worldTransform()
        image = t.mapRect(QRectF(0, 0, self._w, self._h))
        c = t.mapRect(self.crop_px())
        crop = QRect(round(c.left()), round(c.top()), max(1, round(c.width())), max(1, round(c.height())))
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        shade = QPainterPath()
        shade.addRect(image)
        shade.addRect(QRectF(crop))  # odd-even fill: everything but the frame
        painter.fillPath(shade, SHADE)
        self._paint_grid(painter, QRectF(crop))
        # Border, then the handles, with a dark halo so they read on bright film.
        line = QRectF(crop).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(OUTLINE, 3))
        painter.drawRect(line)
        painter.setPen(QPen(LINE, 1))
        painter.drawRect(line)
        handles = self._handles_path(crop)
        painter.strokePath(handles, QPen(QColor(0, 0, 0, 150), 2))
        painter.fillPath(handles, LINE)
        painter.restore()

    def _paint_grid(self, painter: QPainter, r: QRectF) -> None:
        if self._grid == "none" or r.width() < 12 or r.height() < 12:
            return
        if self._grid == "thirds":
            xs = [r.left() + r.width() * k / 3 for k in (1, 2)]
            ys = [r.top() + r.height() * k / 3 for k in (1, 2)]
            pens = [QPen(QColor(0, 0, 0, 70), 1), QPen(QColor(255, 255, 255, 135), 1)]
        else:
            # Square cells, about eight across the short side: a level for rotating.
            cell = min(r.width(), r.height()) / 8
            nx, ny = max(2, round(r.width() / cell)), max(2, round(r.height() / cell))
            xs = [r.left() + r.width() * k / nx for k in range(1, nx)]
            ys = [r.top() + r.height() * k / ny for k in range(1, ny)]
            pens = [QPen(QColor(255, 255, 255, 80), 1)]
        for i, pen in enumerate(pens):
            off = 1.0 if len(pens) == 2 and i == 0 else 0.0  # shadow one pixel right/below
            painter.setPen(pen)
            for x in xs:
                xx = math.floor(x) + 0.5 + off
                painter.drawLine(QPointF(xx, r.top() + 1), QPointF(xx, r.bottom() - 1))
            for y in ys:
                yy = math.floor(y) + 0.5 + off
                painter.drawLine(QPointF(r.left() + 1, yy), QPointF(r.right() - 1, yy))

    @staticmethod
    def _handles_path(crop: QRect) -> QPainterPath:
        """Corner brackets and edge bars, drawn just inside the frame."""
        x0, y0, w, h = crop.x(), crop.y(), crop.width(), crop.height()
        x1, y1 = x0 + w, y0 + h
        t = 4
        n = _arm(w, h)
        cx, cy = x0 + (w - n) // 2, y0 + (h - n) // 2
        rects = [
            (x0, y0, n, t), (x0, y0, t, n), (x1 - n, y0, n, t), (x1 - t, y0, t, n),
            (x0, y1 - t, n, t), (x0, y1 - n, t, n), (x1 - n, y1 - t, n, t), (x1 - t, y1 - n, t, n),
            (cx, y0, n, t), (cx, y1 - t, n, t), (x0, cy, t, n), (x1 - t, cy, t, n),
        ]
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)  # odd-even would punch the corner overlaps out
        for rect in rects:
            path.addRect(QRectF(*rect))
        return path.simplified()
