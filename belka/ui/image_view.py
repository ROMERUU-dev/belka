"""Zoomable image view with the develop tools: samplers, crop, straighten, Guided Upright, before/after.

Every coordinate crossing the API is normalised to the displayed image (0..1), so
it survives previews rendered at different sizes. Interaction follows Lightroom
Classic: a click toggles fit <-> 1:1 at the pointer, a drag pans when zoomed, the
crop frame rotates the photo when dragged from outside, Ctrl inside the crop tool
draws a straighten line, the guided tool draws up to four guides.
"""

from __future__ import annotations

import math
from functools import lru_cache

from PySide6.QtCore import QEvent, QLineF, QPoint, QPointF, QRectF, QSizeF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QContextMenuEvent,
    QCursor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QWidget

from belka.i18n import _, language
from belka.ui.crop_overlay import CropOverlay, Rect, cursor_for, draw_cursor, outlined
from belka.ui.guide_overlay import (
    END_TOLERANCE,
    MAX_GUIDES,
    MIN_LENGTH,
    Guide,
    GuideEditor,
    Hit,
    hit_test,
    loupe_rect,
    loupe_scale,
    paint_grid,
    paint_guides,
    paint_loupe,
)

TOOLS = ("none", "base", "neutral", "crop", "straighten", "guided")
FRAME_TOOLS = ("crop", "straighten", "guided")  # show the whole uncropped photo
# The eyedroppers sample the uncropped photo too (at the current zoom): a cropped
# "before" half beside it would sample other pixels.
UNCROPPED_TOOLS = FRAME_TOOLS + ("base", "neutral")
KEY_TOOLS = ("crop", "guided")  # own Enter and Esc wherever the window's focus is (see eventFilter)
COMPARE_MODES = ("off", "split", "side")
GRID_MODES = ("off", "grid")
ENTER_KEYS = (Qt.Key.Key_Return, Qt.Key.Key_Enter)
DELETE_KEYS = (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
BACKGROUND = QColor(30, 30, 30)
ZOOM_MIN, ZOOM_MAX = 0.02, 16.0
CROP_MARGIN = 40  # px around the fitted image while cropping: room for the handles and to rotate outside
SIDE_MARGIN = 12  # px gutter around each fitted side-by-side pane
CLICK_SLOP = 3  # px of movement that still counts as a click
DIVIDER_TOLERANCE = 6  # px


def _decimal(value: float, decimals: int = 1) -> str:
    text = f"{value:.{decimals}f}"
    return text.replace(".", ",") if language() == "es" else text


def _fit_into(size: QSizeF, target: QRectF) -> QRectF:
    """``size`` scaled to fit ``target``, centred (a before image of another shape is not stretched)."""
    if size.isEmpty():
        return QRectF(target)
    s = min(target.width() / size.width(), target.height() / size.height())
    w, h = size.width() * s, size.height() * s
    c = target.center()
    return QRectF(c.x() - w / 2, c.y() - h / 2, w, h)


def _paint_eyedropper(tint: QColor):
    def paint(p: QPainter) -> None:
        # Drawn along +x, then turned so the tip (the hot spot) is bottom-left.
        p.translate(3.0, 29.0)
        p.rotate(-45.0)
        tube = QPainterPath()
        tube.moveTo(0, 0)
        tube.lineTo(4.5, -2.2)
        tube.lineTo(19, -2.2)
        tube.lineTo(19, 2.2)
        tube.lineTo(4.5, 2.2)
        tube.closeSubpath()
        collar = QPainterPath()
        collar.addRoundedRect(QRectF(19, -4.4, 3.6, 8.8), 1.2, 1.2)
        bulb = QPainterPath()
        bulb.addRoundedRect(QRectF(22.6, -3.8, 10.4, 7.6), 3.8, 3.8)
        halo = QPen(QColor(0, 0, 0, 200), 3.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin)
        p.strokePath(tube.united(collar).united(bulb), halo)
        p.fillPath(tube, QColor(255, 255, 255, 70))
        p.fillPath(bulb, tint)
        line = QPen(QColor(242, 242, 242), 1.3)
        for part in (tube, bulb):
            p.strokePath(part, line)
        p.fillPath(collar, QColor(242, 242, 242))

    return paint


@lru_cache(maxsize=None)
def eyedropper_cursor(tint: str) -> QCursor:
    return draw_cursor(_paint_eyedropper(QColor(tint)), 3, 29)


def _paint_zoom(p: QPainter) -> None:
    handle = QPainterPath()
    handle.moveTo(18.8, 18.8)
    handle.lineTo(25.5, 25.5)
    outlined(p, handle, 2.6)
    lens = QPainterPath()
    lens.addEllipse(QPointF(13, 13), 7.5, 7.5)
    lens.moveTo(9.6, 13)
    lens.lineTo(16.4, 13)
    lens.moveTo(13, 9.6)
    lens.lineTo(13, 16.4)
    outlined(p, lens, 1.5)


@lru_cache(maxsize=None)
def zoom_cursor() -> QCursor:
    return draw_cursor(_paint_zoom, 13, 13)


def _label(p: QPainter, font: QFont, text: str, anchor: QPointF, right: bool = False) -> QRectF:
    """Small translucent pill; ``anchor`` is its top-left (top-right with ``right``)."""
    fm = QFontMetricsF(font)
    w, h = fm.horizontalAdvance(text) + 16, fm.height() + 6
    r = QRectF(anchor.x() - w if right else anchor.x(), anchor.y(), w, h)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(18, 18, 18, 195))
    p.drawRoundedRect(r, 3, 3)
    p.setFont(font)
    p.setPen(QColor(212, 212, 212))
    p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
    p.restore()
    return r


class _BeforePane(QWidget):
    """Left half of the side-by-side comparison, mirroring the view's zoom and scroll."""

    def __init__(self, view: ImageView) -> None:
        super().__init__(view)
        self._view = view
        self._press: QPointF | None = None
        self._last = QPointF()
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.hide()

    def _to_viewport(self, pos: QPointF) -> QPointF:
        return pos - QPointF((self.width() - self._view.viewport().width()) / 2, 0)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), BACKGROUND)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        view = self._view
        before = view._before
        if view.has_image() and not before.isNull():
            target = view._image_rect_vp().translated((self.width() - view.viewport().width()) / 2, 0)
            p.drawPixmap(_fit_into(QSizeF(before.size()), target), before, QRectF(before.rect()))
        p.fillRect(QRectF(self.width() - 1, 0, 1, self.height()), QColor(10, 10, 10))
        if view.has_image():
            _label(p, view._label_font(), _("Antes"), QPointF(12, 12))
        p.end()

    def mousePressEvent(self, event) -> None:
        # Not the right button: the context menu takes its release, and the pane would keep panning.
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._press = event.position()
            self._last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._press is not None:
            self._view._pan_by(event.position() - self._last)
            self._last = event.position()

    def mouseReleaseEvent(self, event) -> None:
        press, self._press = self._press, None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        if (press is not None and event.button() == Qt.MouseButton.LeftButton
                and (event.position() - press).manhattanLength() <= CLICK_SLOP):
            self._view.toggle_zoom(pos=self._to_viewport(event.position()))

    def wheelEvent(self, event) -> None:
        self._view._zoom_wheel(event.angleDelta().y(), self._to_viewport(event.position()))

    def contextMenuEvent(self, event) -> None:
        # The before image sits in the after image's rectangle: the same normalised point.
        self._view._context_menu(self._to_viewport(QPointF(event.pos())), event.globalPos())


class ImageView(QGraphicsView):
    regionSelected = Signal(str, tuple)  # tool, (x0, y0, x1, y1) normalised
    filesDropped = Signal(list)
    zoomChanged = Signal(float)
    cropChanged = Signal(tuple)  # the user moved, resized or swapped the frame
    cropCommitted = Signal(tuple)  # Enter, double-click inside the frame, or commit_crop()
    cropCancelled = Signal()  # Esc
    rotateStarted = Signal()  # press outside the frame: remember the angle the drag starts from
    angleDelta = Signal(float)  # degrees since the previous emission, positive = clockwise
    rotateFinished = Signal()  # rotation drag released or cancelled: one history step
    straightenLine = Signal(QPointF, QPointF)  # normalised image coordinates
    pixelHovered = Signal(float, float)  # normalised, (-1, -1) off the image
    viewportChanged = Signal(QRectF)  # visible part of the image, normalised
    guidesChanged = Signal(list)  # guided: [(x0, y0, x1, y1), ...] normalised, after each finished edit
    guideDragging = Signal(bool)  # guided: a guide is being drawn or moved (True at the press, False at the end)
    guideLimitReached = Signal()  # guided: a press to draw a fifth guide, which is ignored
    toolFinished = Signal(str)  # Enter in "guided"
    toolCancelled = Signal(str)  # Esc in "guided" (the crop tool keeps cropCancelled)
    contextMenuRequested = Signal(QPoint, float, float)  # global position, normalised image point or (-1, -1)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        # Zoom anchoring is done by hand (_zoom_at) so clicks, the wheel and the
        # navigator all keep the same point under the pointer.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # Labels and the split divider are fixed to the viewport, so scrolling
        # must repaint rather than blit.
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QBrush(BACKGROUND))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self.viewport().setMouseTracking(True)

        self._pixmap = QGraphicsPixmapItem()
        self._pixmap.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._pixmap)
        self._clipping = QGraphicsPixmapItem()
        self._clipping.setZValue(1)
        self._clipping.hide()
        self.scene().addItem(self._clipping)
        self._overlay = QGraphicsRectItem()
        self._overlay.setPen(QPen(QColor(120, 200, 255), 0, Qt.PenStyle.DashLine))
        self._overlay.setZValue(9)
        self._overlay.hide()
        self.scene().addItem(self._overlay)
        self._rubber = QGraphicsRectItem()
        self._rubber.setPen(QPen(QColor(255, 210, 60), 0, Qt.PenStyle.DashLine))
        self._rubber.setBrush(QBrush(QColor(255, 210, 60, 40)))
        self._rubber.setZValue(10)
        self._rubber.hide()
        self.scene().addItem(self._rubber)
        self._crop = CropOverlay()
        self._crop.hide()
        self.scene().addItem(self._crop)
        self._before_pane = _BeforePane(self)

        self._tool = "none"
        self._mode: str | None = "fit"  # "fit", "fill" or None for a fixed zoom
        self._compare = "off"
        self._before = QPixmap()
        self._split = 0.5  # divider, fraction of the viewport width
        self._gesture: str | None = None  # pan, click, split, crop, rotate, straighten, region, guide, limit, ignored
        self._button = Qt.MouseButton.NoButton  # the one that started the gesture, and ends it
        self._press_pos = QPointF()
        self._last_pos = QPointF()
        self._region_start = QPointF()
        self._angle = 0.0
        self._line: tuple[QPointF, QPointF] | None = None  # straighten, image pixels
        self._guides = GuideEditor()
        self._guide_hover: Hit | None = None
        self._grid = "off"
        self._hover: QPointF | None = None
        self._hovering = False
        self._zoom_key: tuple | None = None
        self._visible = QRectF()
        self._placeholder = _("Crea o abre un rollo, conecta la cámara y captura;\no arrastra aquí archivos RAW/TIFF/JPEG para importarlos.")
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            bar.valueChanged.connect(self._view_changed)

    # ------------------------------------------------------------ image
    def has_image(self) -> bool:
        return not self._pixmap.pixmap().isNull()

    def _image_size(self) -> tuple[int, int]:
        pix = self._pixmap.pixmap()
        return pix.width(), pix.height()

    def set_image(self, image: QImage | None) -> None:
        """Show ``image``. A fixed zoom survives an image of the same shape (next frame, or a
        sharper render of this one): the same part stays on screen at the same size."""
        old = self._pixmap.pixmap().size()
        centre = self._scene_pos(QPointF(self.viewport().rect().center())) if self.has_image() else None
        if image is None or image.isNull():
            self._cancel_gesture()
            self._pixmap.setPixmap(QPixmap())
            self.set_clipping_overlay(None)
            self._crop.hide()
            self._view_changed()
            return
        if self._gesture in ("crop", "straighten", "region") and image.size() != old:
            self._cancel_gesture()  # it holds pixel positions of the previous render
        self._pixmap.setPixmap(QPixmap.fromImage(image))
        w, h = self._image_size()
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        self._crop.set_image_size(w, h)
        self._crop.setVisible(self._tool == "crop")
        self._place_clipping()
        same_shape = centre is not None and abs(old.width() * h - w * old.height()) <= 0.01 * w * old.height()
        if self._mode is not None or not same_shape:
            self._refit()
            return
        if old.width() != w:
            self._set_scale(self.zoom() * old.width() / w)
            self.centerOn(QPointF(centre.x() * w / old.width(), centre.y() * h / old.height()))
        self._view_changed()

    def set_clipping_overlay(self, image: QImage | None) -> None:
        """Clipping warning (or any mask) drawn over the image, stretched to it."""
        if image is None or image.isNull():
            self._clipping.setPixmap(QPixmap())
            self._clipping.hide()
            return
        self._clipping.setPixmap(QPixmap.fromImage(image))
        self._place_clipping()
        self._clipping.show()

    def _place_clipping(self) -> None:
        mask = self._clipping.pixmap()
        if mask.isNull() or not self.has_image():
            return
        w, h = self._image_size()
        self._clipping.setTransform(QTransform.fromScale(w / mask.width(), h / mask.height()))

    def set_overlay(self, rect: Rect | None) -> None:
        if rect is None or not self.has_image():
            self._overlay.hide()
            return
        w, h = self._image_size()
        x0, y0, x1, y1 = rect
        self._overlay.setRect(QRectF(QPointF(x0 * w, y0 * h), QPointF(x1 * w, y1 * h)).normalized())
        self._overlay.show()

    # ------------------------------------------------------------ zoom and scroll
    def zoom(self) -> float:
        """Screen pixels per image pixel."""
        return self.transform().m11()

    def is_fit(self) -> bool:
        return self._mode == "fit"

    def fit(self) -> None:
        self._mode = "fit"
        self._fill_view(fill=False)

    def fill(self) -> None:
        """Zoom so the image covers the whole view (Lightroom's FILL)."""
        self._mode = "fill"
        self._fill_view(fill=True)

    def actual_size(self) -> None:
        self.zoom_to(1.0)

    def zoom_to(self, factor: float, pos: QPointF | None = None) -> None:
        """Absolute zoom keeping the image point under ``pos`` (viewport coordinates; default the centre)."""
        if not self.has_image():
            return
        self._mode = None
        self._zoom_at(factor, pos)

    def toggle_zoom(self, *, factor: float = 1.0, pos: QPointF | None = None) -> None:
        """Lightroom's Z / click: fit <-> ``factor`` at the pointer.

        Keyword-only so it can be connected straight to ``QAction.triggered``, whose
        ``checked`` would otherwise land in ``factor``.
        """
        if not self.has_image():
            return
        if self._mode != "fit":
            self.fit()
            return
        if pos is None:
            cursor = self.viewport().mapFromGlobal(QCursor.pos())
            pos = QPointF(cursor) if self.viewport().rect().contains(cursor) else None
        self.zoom_to(factor, pos)

    def center_on(self, x: float, y: float) -> None:
        """Centre the view on a normalised image point (navigator clicks)."""
        if not self.has_image():
            return
        w, h = self._image_size()
        self.centerOn(QPointF(x * w, y * h))
        self._view_changed()

    def visible_rect(self) -> QRectF:
        """Normalised part of the image inside the viewport (empty without an image)."""
        if not self.has_image():
            return QRectF()
        w, h = self._image_size()
        inv, _ok = self.viewportTransform().inverted()
        vis = inv.mapRect(QRectF(self.viewport().rect())).intersected(QRectF(0, 0, w, h))
        return QRectF(vis.x() / w, vis.y() / h, vis.width() / w, vis.height() / h)

    def _refit(self) -> None:
        if self._mode == "fill":
            self.fill()
        else:
            self.fit()

    def _fill_view(self, fill: bool) -> None:
        if not self.has_image():
            self._view_changed()
            return
        margin = 0
        if not fill:
            margin = CROP_MARGIN if self._tool in FRAME_TOOLS else SIDE_MARGIN if self._shown_compare() == "side" else 0
        w, h = self._image_size()
        vw = max(1, self.viewport().width() - 2 * margin)
        vh = max(1, self.viewport().height() - 2 * margin)
        scale = max(vw / w, vh / h) if fill else min(vw / w, vh / h)
        self._set_scale(scale)
        self.centerOn(QPointF(w / 2, h / 2))
        self._view_changed()

    def _set_scale(self, scale: float) -> None:
        scale = min(max(scale, ZOOM_MIN), ZOOM_MAX)
        self.setTransform(QTransform.fromScale(scale, scale))

    def _zoom_at(self, scale: float, anchor: QPointF | None) -> None:
        anchor = QPointF(self.viewport().rect().center()) if anchor is None else QPointF(anchor)
        target = self._scene_pos(anchor)
        self._set_scale(scale)
        # Scroll so the same image point is back under the anchor.
        d = self.viewportTransform().map(target) - anchor
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        h.setValue(h.value() + round(d.x()))
        v.setValue(v.value() + round(d.y()))
        self._view_changed()

    def _zoom_wheel(self, delta: int, pos: QPointF) -> None:
        if delta and self.has_image():
            self.zoom_to(self.zoom() * 2.0 ** (delta / 360.0), pos)

    def _pan_by(self, d: QPointF) -> None:
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        h.setValue(h.value() - round(d.x()))
        v.setValue(v.value() - round(d.y()))

    def _view_changed(self, *_args) -> None:
        key = (round(self.zoom(), 6), self._mode)
        if key != self._zoom_key:
            self._zoom_key = key
            self.zoomChanged.emit(self.zoom())
        visible = self.visible_rect()
        if visible != self._visible:
            self._visible = visible
            self.viewportChanged.emit(QRectF(visible))
        if self._shown_compare() == "side":
            self._before_pane.update()
        self.viewport().update()
        self._update_cursor()

    def wheelEvent(self, event) -> None:
        self._zoom_wheel(event.angleDelta().y(), event.position())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_panes()
        if self._mode is not None:
            self._refit()
        else:
            self._view_changed()

    # ------------------------------------------------------------ compare
    @property
    def compare_mode(self) -> str:
        """The requested mode; it is not shown while a frame tool is active (see :meth:`set_tool`)."""
        return self._compare

    def _shown_compare(self) -> str:
        """Comparison on screen. None while framing, as in Lightroom: the overlay needs the whole
        photo, and the before render is cropped, so its halves would not line up anyway."""
        return "off" if self._tool in UNCROPPED_TOOLS else self._compare

    def set_compare(self, mode: str, before: QImage | None = None) -> None:
        """Before/after: "off", "split" (draggable divider) or "side" (two synchronised panes).

        ``before`` may arrive later with another call; it is fitted into the
        current image's rectangle, so it should share its geometry.
        """
        mode = mode if mode in COMPARE_MODES else "off"
        self._before = QPixmap() if before is None or before.isNull() else QPixmap.fromImage(before)
        if mode != self._compare:
            self._compare = mode
            self._split = 0.5
            self._layout_panes()
            if self._mode is not None:
                self._refit()
        self._before_pane.update()
        self.viewport().update()

    def _layout_panes(self) -> None:
        half = self.width() // 2 if self._shown_compare() == "side" else 0
        if self.viewportMargins().left() != half:
            self.setViewportMargins(half, 0, 0, 0)
        self._before_pane.setGeometry(0, 0, half, self.height())
        self._before_pane.setVisible(half > 0)

    def _image_rect_vp(self) -> QRectF:
        w, h = self._image_size()
        return self.viewportTransform().mapRect(QRectF(0, 0, w, h))

    def _divider_x(self, x: float | None = None) -> float:
        """The divider (or ``x``) kept over the visible part of the image."""
        vis = self._image_rect_vp().intersected(QRectF(self.viewport().rect()))
        x = self._split * self.viewport().width() if x is None else x
        return min(max(x, vis.left()), vis.right())

    def _near_divider(self, pos: QPointF) -> bool:
        if self._shown_compare() != "split" or not self.has_image():
            return False
        vis = self._image_rect_vp().intersected(QRectF(self.viewport().rect()))
        return abs(pos.x() - self._divider_x()) <= DIVIDER_TOLERANCE and vis.top() <= pos.y() <= vis.bottom()

    def _label_font(self) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(11)
        return font

    # ------------------------------------------------------------ tools
    @property
    def tool(self) -> str:
        return self._tool

    def set_tool(self, tool: str) -> None:
        """Switch tool. Entering or leaving "crop"/"straighten"/"guided" hides or restores the comparison
        and fits the view: the photo changes geometry (uncropped <-> cropped), so a fixed zoom would land
        on another part of it, and the frame tools need its edges on screen with room around them."""
        self._cancel_gesture()
        framing = self._tool in FRAME_TOOLS
        uncropped = self._tool in UNCROPPED_TOOLS
        self._tool = tool if tool in TOOLS else "none"
        self._crop.setVisible(self._tool == "crop" and self.has_image())
        self._guides.selected = None
        self._guide_hover = None
        if self._tool in FRAME_TOOLS:
            self.setFocus()
        self._watch_keys()
        if (self._tool in FRAME_TOOLS) != framing:
            self._layout_panes()
            self.fit()
        elif (self._tool in UNCROPPED_TOOLS) != uncropped:
            self._layout_panes()
            self.viewport().update()
        self._update_cursor()

    def guides(self) -> list[Guide]:
        """Guided Upright lines, normalised to the displayed image."""
        return list(self._guides.guides)

    def set_guides(self, guides) -> None:
        """Show these guides (at most four ``(x0, y0, x1, y1)``, normalised to the displayed image), e.g.
        remapped onto a new render. Being normalised, they stay put across ``set_image``; a drag in
        progress carries on from its guide's new place. Emits no ``guidesChanged`` (only
        ``guideDragging(False)`` when the guide being dragged is no longer there)."""
        if not self._guides.set(guides):
            self.guideDragging.emit(False)  # the guide being dragged is gone
            self._gesture = None
        self._guide_hover = None
        self.viewport().update()

    @property
    def grid_mode(self) -> str:
        return self._grid

    def set_grid(self, mode: str) -> None:
        """Lightroom's transform grid over the photo, whatever the tool: "off" or "grid"."""
        mode = mode if mode in GRID_MODES else "off"
        if mode != self._grid:
            self._grid = mode
            self.viewport().update()

    def crop_rect(self) -> Rect:
        return self._crop.rect()

    def set_crop_rect(self, rect: Rect | None) -> None:
        """Replace the frame (normalised); ``None`` takes the whole allowed area.

        Only when the stored crop means something new: entering the tool, undo/redo, a 90° rotation or
        a flip. The view keeps the user's frame across ``set_image``; setting it after every render
        would throw their drag away.
        """
        self._crop.set_rect(rect)

    def crop_aspect(self) -> float | None:
        """Locked width/height in image pixels, oriented like the frame."""
        return self._crop.aspect()

    def set_crop_aspect(self, ratio: float | None) -> None:
        """Lock the frame to ``ratio`` (w/h, either orientation) or free it with ``None``."""
        self._crop.set_aspect(ratio)

    def set_crop_bounds(self, rect: Rect | None) -> None:
        """Keep the frame inside ``rect`` (constrain to image after straightening); ``None`` = whole image.

        Call it after every render while cropping. It never loses the user's frame: rotating out and
        back shrinks it and then gives it back.
        """
        self._crop.set_bounds(rect)

    def swap_crop_orientation(self) -> None:
        self._crop.swap_orientation()

    def commit_crop(self) -> None:
        """Hand the frame over with ``cropCommitted``, as Enter does (the window's Done button, R again)."""
        if self._tool == "crop" and self.has_image():
            self._cancel_gesture()
            self.cropCommitted.emit(self.crop_rect())

    # ------------------------------------------------------------ mouse
    def _scene_pos(self, pos: QPointF) -> QPointF:
        inv, _ok = self.viewportTransform().inverted()
        return inv.map(QPointF(pos))

    def _clamp(self, p: QPointF) -> QPointF:
        w, h = self._image_size()
        return QPointF(min(max(p.x(), 0.0), w), min(max(p.y(), 0.0), h))

    def _normalised(self, p: QPointF) -> QPointF:
        w, h = self._image_size()
        return QPointF(p.x() / w, p.y() / h)

    def _image_point(self, pos: QPointF) -> tuple[float, float]:
        """Normalised image point under ``pos`` (viewport), (-1, -1) off the image."""
        if self.has_image():
            p = self._scene_pos(pos)
            w, h = self._image_size()
            if 0 <= p.x() < w and 0 <= p.y() < h:
                return p.x() / w, p.y() / h
        return -1.0, -1.0

    def _pointer_angle(self, pos: QPointF) -> float:
        c = self.viewportTransform().map(self._crop.crop_px().center())
        return math.degrees(math.atan2(pos.y() - c.y(), pos.x() - c.x()))

    def mousePressEvent(self, event) -> None:
        button = event.button()
        self.setFocus(Qt.FocusReason.MouseFocusReason)  # the crop keys must not stay with a combo just used
        if self._gesture is not None:
            return  # one gesture at a time: another button must not hijack the drag
        if not self.has_image() or button not in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            super().mousePressEvent(event)
            return
        self._button = button
        pos = event.position()
        self._press_pos = QPointF(pos)
        self._last_pos = QPointF(pos)
        scene = self._scene_pos(pos)
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if button == Qt.MouseButton.MiddleButton:
            self._gesture = "pan"
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._near_divider(pos):
            self._gesture = "split"
        elif self._tool == "guided":
            self._press_guide(pos)
        elif self._tool == "crop" and not ctrl:
            part = self._crop.hit(scene, self.zoom())
            if part == "rotate":
                self._gesture = "rotate"
                self._angle = self._pointer_angle(pos)
                self._crop.set_grid("fine")
                self.rotateStarted.emit()
            else:
                self._gesture = "crop"
                self._crop.begin_drag(part, scene)
                self._crop.set_grid("thirds")
                if part == "move":
                    self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._tool in ("crop", "straighten"):
            self._gesture = "straighten"
            p = self._clamp(scene)
            self._line = (p, p)
        elif self._tool in ("base", "neutral"):
            self._gesture = "region"
            self._region_start = self._clamp(scene)
            self._rubber.setRect(QRectF(self._region_start, self._region_start))
            self._rubber.show()
        else:
            self._gesture = "click"

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        self._hover = QPointF(pos)
        self._emit_hover(pos)
        g = self._gesture
        if g is None:
            if self._tool == "guided":
                self._set_guide_hover(self._guide_hit(pos))
            self._update_cursor(event.modifiers())
            return
        scene = self._scene_pos(pos)
        if g == "click" and (pos - self._press_pos).manhattanLength() > CLICK_SLOP:
            g = self._gesture = "pan"
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        if g == "pan":
            self._pan_by(pos - self._last_pos)
        elif g == "split":
            self._split = self._divider_x(pos.x()) / max(1, self.viewport().width())
            self.viewport().update()
        elif g == "crop":
            self._crop.drag_to(scene)
            self.cropChanged.emit(self.crop_rect())
        elif g == "rotate":
            angle = self._pointer_angle(pos)
            delta = (angle - self._angle + 180.0) % 360.0 - 180.0
            self._angle = angle
            if delta:
                self.angleDelta.emit(delta)
        elif g == "straighten" and self._line is not None:
            self._line = (self._line[0], self._clamp(scene))
            self.viewport().update()
        elif g == "region":
            self._rubber.setRect(QRectF(self._region_start, self._clamp(scene)).normalized())
        elif g == "guide":
            self._guides.drag(self._guide_point(pos))
            self.viewport().update()
        elif g == "limit" and (pos - self._press_pos).manhattanLength() > CLICK_SLOP:
            self._gesture = "ignored"  # tell once per drag
            self.guideLimitReached.emit()
        self._last_pos = QPointF(pos)

    def mouseReleaseEvent(self, event) -> None:
        if self._gesture is None or event.button() != self._button:
            super().mouseReleaseEvent(event)
            return
        g, self._gesture = self._gesture, None
        if g == "click":
            self.toggle_zoom(pos=event.position())
        elif g == "crop":
            self._crop.end_drag()
            self._crop.set_grid("none")
        elif g == "rotate":
            self._crop.set_grid("none")
            self.rotateFinished.emit()
        elif g == "straighten":
            self._finish_straighten()
        elif g == "region":
            self._finish_region()
        elif g == "guide":
            self._finish_guide()
            self._set_guide_hover(self._guide_hit(event.position()))
        self._update_cursor(event.modifiers())

    def mouseDoubleClickEvent(self, event) -> None:
        if (self._tool == "crop" and self._gesture is None and self.has_image()
                and event.button() == Qt.MouseButton.LeftButton
                and self._crop.hit(self._scene_pos(event.position()), self.zoom()) == "move"):
            self.commit_crop()
            return
        # Qt delivers the second press of two quick clicks only as this event: without it a
        # second click would not zoom back out, nor a quick re-grab move a handle.
        self.mousePressEvent(event)

    def _finish_straighten(self) -> None:
        line, self._line = self._line, None
        self.viewport().update()
        if line is None or QLineF(*line).length() * self.zoom() < 6:
            return  # a click, not a line
        self.straightenLine.emit(self._normalised(line[0]), self._normalised(line[1]))

    def _finish_region(self) -> None:
        rect = self._rubber.rect()
        self._rubber.hide()
        w, h = self._image_size()
        if rect.width() < 4 or rect.height() < 4:
            # A click samples a small square around the point.
            size = max(w, h) * 0.01
            rect = QRectF(rect.center() - QPointF(size, size), rect.center() + QPointF(size, size)).intersected(
                QRectF(0, 0, w, h)
            )
        self.regionSelected.emit(self._tool, (rect.left() / w, rect.top() / h, rect.right() / w, rect.bottom() / h))

    # ------------------------------------------------------------ guided upright
    def _guide_point(self, pos: QPointF) -> tuple[float, float]:
        p = self._normalised(self._scene_pos(pos))
        return p.x(), p.y()

    def _guide_segments(self) -> list[tuple[QPointF, QPointF]]:
        """The guides in viewport pixels."""
        t = self.viewportTransform()
        w, h = self._image_size()
        return [(t.map(QPointF(x0 * w, y0 * h)), t.map(QPointF(x1 * w, y1 * h)))
                for x0, y0, x1, y1 in self._guides.guides]

    def _guide_hit(self, pos: QPointF | None) -> Hit | None:
        if pos is None or self._tool != "guided" or not self.has_image():
            return None
        return hit_test(self._guide_segments(), pos)

    def _set_guide_hover(self, hit: Hit | None) -> None:
        if hit != self._guide_hover:
            self._guide_hover = hit
            self.viewport().update()

    def _press_guide(self, pos: QPointF) -> None:
        """Grab a guide's endpoint or line, or start drawing a new one on the photo."""
        hit = self._guide_hit(pos)
        t = END_TOLERANCE
        on_image = self._image_rect_vp().adjusted(-t, -t, t, t).contains(pos)
        if (hit is not None or on_image) and self._guides.begin(hit, self._guide_point(pos)):
            self._gesture = "guide"
            self._guide_hover = None
            moving = hit is not None and hit[1] is None
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor if moving else Qt.CursorShape.CrossCursor)
            self.guideDragging.emit(True)
        else:
            self._guides.selected = None  # a click beside the guides deselects, as in Lightroom
            if on_image:
                self._gesture = "limit"  # a drag here would be a fifth guide
        self.viewport().update()

    def _finish_guide(self) -> None:
        if self._guides.creating:
            a, b = self._guide_segments()[self._guides.active()[0]]
            if QLineF(a, b).length() < MIN_LENGTH:
                self._guides.cancel()  # a click, not a guide
        if self._guides.dragging and self._guides.finish():
            self.guidesChanged.emit(self.guides())
        self.guideDragging.emit(False)
        self.viewport().update()

    def _remove_guide(self, index: int) -> None:
        self._guides.remove(index)
        self._guide_hover = None
        self.guidesChanged.emit(self.guides())
        self.viewport().update()
        self._update_cursor()

    # ------------------------------------------------------------ context menu
    def contextMenuEvent(self, event) -> None:
        if event.reason() == QContextMenuEvent.Reason.Keyboard:
            pos = self._hover if self._hover is not None else QPointF(self.viewport().rect().center())
            self._context_menu(pos, self.viewport().mapToGlobal(pos.toPoint()), keyboard=True)
        else:
            self._context_menu(QPointF(event.pos()), event.globalPos())
        event.accept()

    def _context_menu(self, pos: QPointF, global_pos: QPoint, keyboard: bool = False) -> None:
        """Right click (or the Menu key) at ``pos`` (viewport): it removes a guide under the pointer in
        the guided tool; anywhere else it asks the window for its menu. Never in the middle of a drag."""
        if self._gesture is not None:
            return
        hit = None if keyboard else self._guide_hit(pos)
        if hit is not None:
            self._remove_guide(hit[0])
            return
        x, y = self._image_point(pos)
        self.contextMenuRequested.emit(global_pos, x, y)

    def _cancel_gesture(self) -> None:
        g, self._gesture = self._gesture, None
        if g == "crop":
            self._crop.end_drag()
        if g == "rotate":
            self.rotateFinished.emit()
        if g == "guide":
            self._guides.cancel()
            self.guideDragging.emit(False)
        self._crop.set_grid("none")
        self._line = None
        self._rubber.hide()
        self.viewport().update()

    def _emit_hover(self, pos: QPointF) -> None:
        x, y = self._image_point(pos)
        if x >= 0:
            self._hovering = True
            self.pixelHovered.emit(x, y)
        elif self._hovering:
            self._hovering = False
            self.pixelHovered.emit(-1.0, -1.0)

    def viewportEvent(self, event) -> bool:
        if event.type() == QEvent.Type.Leave:
            self._hover = None
            self._set_guide_hover(None)
            if self._hovering:
                self._hovering = False
                self.pixelHovered.emit(-1.0, -1.0)
        return super().viewportEvent(event)

    def _update_cursor(self, modifiers=None) -> None:
        if self._gesture is not None:
            return  # the gesture chose its cursor
        vp = self.viewport()
        pos = self._hover
        if modifiers is None:
            modifiers = QGuiApplication.keyboardModifiers()
        if not self.has_image():
            vp.setCursor(Qt.CursorShape.ArrowCursor)
        elif pos is not None and self._near_divider(pos):
            vp.setCursor(Qt.CursorShape.SplitHCursor)
        elif self._tool == "crop":
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                vp.setCursor(Qt.CursorShape.CrossCursor)
            elif pos is not None:
                vp.setCursor(cursor_for(self._crop.hit(self._scene_pos(pos), self.zoom())))
        elif self._tool == "straighten":
            vp.setCursor(Qt.CursorShape.CrossCursor)
        elif self._tool == "guided":
            vp.setCursor(self._guided_cursor(pos))
        elif self._tool == "base":
            vp.setCursor(eyedropper_cursor("#e0954a"))  # film-base orange
        elif self._tool == "neutral":
            vp.setCursor(eyedropper_cursor("#a0a0a0"))
        elif self._mode == "fit":
            vp.setCursor(zoom_cursor())
        else:
            vp.setCursor(Qt.CursorShape.OpenHandCursor)

    def _guided_cursor(self, pos: QPointF | None) -> Qt.CursorShape:
        hit = self._guide_hit(pos)
        if hit is not None:
            return Qt.CursorShape.SizeAllCursor if hit[1] is not None else Qt.CursorShape.OpenHandCursor
        return Qt.CursorShape.CrossCursor if len(self._guides.guides) < MAX_GUIDES else Qt.CursorShape.ArrowCursor

    # ------------------------------------------------------------ keys
    @staticmethod
    def _bare(event) -> bool:
        mods = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier
        return not (event.modifiers() & mods)

    def _claims_key(self, event, focused: bool = True) -> bool:
        """Whether the view takes this key from the window's shortcuts: Enter/Esc (and X) of the crop
        and guided tools, Esc during a drag, and Delete for a selected guide while the view has focus."""
        key = event.key()
        if key == Qt.Key.Key_Escape and self._gesture is not None:
            return True
        if not self._bare(event):
            return False
        if self._tool == "crop":
            return key in (*ENTER_KEYS, Qt.Key.Key_Escape, Qt.Key.Key_X)
        if self._tool == "guided":
            return key in (*ENTER_KEYS, Qt.Key.Key_Escape) or (
                focused and key in DELETE_KEYS and self._gesture is None and self._guides.selected is not None)
        return False

    def _tool_key(self, event) -> bool:
        """Act on a key :meth:`_claims_key` takes; True when it was one."""
        if not self._claims_key(event):
            return False
        key = event.key()
        if self._tool == "crop" and self._bare(event):
            if key in ENTER_KEYS:
                self.commit_crop()
            elif key == Qt.Key.Key_Escape:
                self._cancel_gesture()
                self.cropCancelled.emit()
            else:
                self._crop.swap_orientation()
                self.cropChanged.emit(self.crop_rect())
        elif key == Qt.Key.Key_Escape and self._gesture is not None:
            self._cancel_gesture()  # in the guided tool the first Esc only drops the drag
        elif key in ENTER_KEYS:
            if self._gesture == "guide":  # done in mid-drag keeps the guide where it is, like the crop
                self._gesture = None
                self._finish_guide()
            self._cancel_gesture()
            self.toolFinished.emit(self._tool)
        elif key == Qt.Key.Key_Escape:
            self.toolCancelled.emit(self._tool)
        else:
            self._remove_guide(self._guides.selected)
        return True

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride and self._claims_key(event):
            event.accept()
            return True
        return super().event(event)

    def eventFilter(self, obj, event) -> bool:
        # Installed on the application only while a KEY_TOOL is on screen. Right after the
        # aspect combo or a button of the window was used it still has the focus, but
        # Enter/Esc (and X, which would flag the photo as rejected) belong to the tool, as
        # in Lightroom. Text fields keep their keys; Delete needs the view's own focus.
        if (event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
                and isinstance(obj, QWidget) and obj is not self and obj.window() is self.window()
                and not obj.testAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled)
                and self._claims_key(event, focused=False)):
            if event.type() == QEvent.Type.KeyPress:
                self._tool_key(event)
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def _watch_keys(self) -> None:
        app = QGuiApplication.instance()
        if self._tool in KEY_TOOLS and self.isVisible():
            app.installEventFilter(self)
        else:
            app.removeEventFilter(self)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._watch_keys()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._watch_keys()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if self._tool_key(event):
            return
        if key == Qt.Key.Key_Control:
            self._update_cursor(event.modifiers() | Qt.KeyboardModifier.ControlModifier)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Control:
            self._update_cursor(event.modifiers() & ~Qt.KeyboardModifier.ControlModifier)
        super().keyReleaseEvent(event)

    # ------------------------------------------------------------ painting
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.resetTransform()
        if not self.has_image():
            font = QFont(self.font())
            font.setPixelSize(12)
            painter.setFont(font)
            painter.setPen(QColor(140, 140, 140))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
        else:
            if self._shown_compare() == "split":
                self._paint_split(painter)  # the grid goes over both halves, under the divider
            elif self._grid == "grid":
                self._paint_grid(painter)
            if self._shown_compare() == "side":
                _label(painter, self._label_font(), _("Después"), QPointF(12, 12))
            if self._line is not None:
                self._paint_line(painter)
            if self._tool == "guided":
                self._paint_guides(painter)
        painter.restore()

    def _paint_grid(self, p: QPainter) -> None:
        w, h = self._image_size()
        paint_grid(p, self._image_rect_vp(), w, h, QRectF(self.viewport().rect()), self.viewport().devicePixelRatioF())

    def _paint_guides(self, p: QPainter) -> None:
        segments = self._guide_segments()
        active = self._guides.active()
        paint_guides(p, segments, self._guides.selected, active if active is not None else self._guide_hover)
        if active is None or active[1] is None:
            return
        # Loupe over the endpoint being placed, as Lightroom shows while dragging one.
        index, end = active
        bounds = QRectF(self.viewport().rect()).adjusted(8, 8, -8, -8)
        rect = loupe_rect(segments[index][end], segments[index][1 - end], bounds)
        w, h = self._image_size()
        lines = [(QPointF(x0 * w, y0 * h), QPointF(x1 * w, y1 * h)) for x0, y0, x1, y1 in self._guides.guides]
        paint_loupe(p, rect, self._pixmap.pixmap(), lines[index][end], loupe_scale(self.zoom()), lines, index,
                    BACKGROUND)

    def _paint_split(self, p: QPainter) -> None:
        image = self._image_rect_vp()
        vis = image.intersected(QRectF(self.viewport().rect()))
        if vis.isEmpty():
            return
        x = self._divider_x()
        if x > vis.left() and not self._before.isNull():
            p.save()
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.setClipRect(QRectF(vis.left(), vis.top(), x - vis.left(), vis.height()))
            p.drawPixmap(_fit_into(QSizeF(self._before.size()), image), self._before, QRectF(self._before.rect()))
            p.restore()
        if self._grid == "grid":
            self._paint_grid(p)
        xx = math.floor(x) + 0.5
        p.setPen(QPen(QColor(0, 0, 0, 110), 3))
        p.drawLine(QPointF(xx, vis.top()), QPointF(xx, vis.bottom()))
        p.setPen(QPen(QColor(236, 236, 236, 235), 1))
        p.drawLine(QPointF(xx, vis.top()), QPointF(xx, vis.bottom()))
        # Grip in the middle of the divider.
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(xx, vis.center().y())
        p.setPen(QPen(QColor(236, 236, 236, 235), 1))
        p.setBrush(QColor(26, 26, 26, 225))
        p.drawEllipse(c, 12, 12)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(236, 236, 236))
        for s in (-1.0, 1.0):
            tip = c + QPointF(s * 8.0, 0)
            p.drawPolygon(QPolygonF([tip, c + QPointF(s * 3.5, -4.0), c + QPointF(s * 3.5, 4.0)]))
        p.restore()
        font = self._label_font()
        top = vis.top() + 10
        before, after = _("Antes"), _("Después")
        fm = QFontMetricsF(font)
        if x - vis.left() > fm.horizontalAdvance(before) + 32:
            _label(p, font, before, QPointF(x - 10, top), right=True)
        if vis.right() - x > fm.horizontalAdvance(after) + 32:
            _label(p, font, after, QPointF(x + 10, top))

    def _paint_line(self, p: QPainter) -> None:
        a, b = self._line
        p0, p1 = self.viewportTransform().map(a), self.viewportTransform().map(b)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath(p0)
        path.lineTo(p1)
        outlined(p, path, 1.4)
        for q in (p0, p1):
            dot = QPainterPath()
            dot.addEllipse(q, 3.2, 3.2)
            outlined(p, dot, 1.0, QColor(242, 242, 242))
        if QLineF(p0, p1).length() < 12:
            return
        # Tilt from the nearest axis, in image pixels: what the straighten will undo.
        tilt = math.degrees(math.atan2(b.y() - a.y(), b.x() - a.x()))
        tilt = (tilt + 45.0) % 90.0 - 45.0
        r = self.viewport().rect()
        anchor = QPointF(min(p1.x() + 14, r.right() - 60), min(p1.y() + 12, r.bottom() - 26))
        _label(p, self._label_font(), f"{_decimal(abs(tilt))}°", anchor)

    # ------------------------------------------------------------ drag and drop
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        files = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if files:
            self.filesDropped.emit(files)
            event.acceptProposedAction()
