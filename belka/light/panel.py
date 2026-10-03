"""The light panel: part of a screen turned into a backlight for the film.

Shown full screen on whichever monitor the film sits on. Everything outside
the lit rectangle stays black so it cannot flare into the lens; guides, marks
and the HUD are drawn dim and disappear while the camera is exposing.

Keys: Space capture · arrows move (Shift ×5) · Ctrl+arrows resize ·
+/- intensity · G guides · M marks · H help · Esc close.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QScreen
from PySide6.QtWidgets import QWidget

from belka.i18n import _
from belka.light.geometry import Adapter, LightSettings

EDGE = 10  # px grabbed by the mouse around the rectangle's edges
HUD_COLOR = QColor(80, 80, 80)
MARK_COLOR = QColor(55, 55, 55)


def screen_key(screen: QScreen) -> str:
    parts = [screen.manufacturer(), screen.model(), screen.serialNumber() or screen.name()]
    return "|".join(p for p in parts if p) or screen.name()


def px_per_mm(screen: QScreen, correction: float = 1.0) -> float:
    size = screen.physicalSize()
    geo = screen.geometry()
    if size.width() <= 1:
        return 96 / 25.4 * correction
    return geo.width() / size.width() * correction


class LightPanel(QWidget):
    captureRequested = Signal()
    flatRequested = Signal()
    calibrateRequested = Signal()
    closed = Signal()
    settingsChanged = Signal()

    def __init__(self, settings: LightSettings, adapter: Adapter | None, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Belka · " + _("Panel de luz"))
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.settings = settings
        self.adapter = adapter
        self._color = QColor(255, 255, 255)
        self._capturing = False
        self._ruler = False
        self._status = ""
        self._thumb: QImage | None = None
        self._drag: tuple[str, QPointF, QRectF] | None = None
        self._buttons: dict[str, QRectF] = {}
        self._flat_armed = False

    # ------------------------------------------------------------ public API
    def show_on(self, screen: QScreen | None) -> None:
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if self.windowHandle() is None:
            self.create()
        if self.isVisible() and self.windowHandle().screen() is not screen:
            # On Wayland a visible fullscreen surface cannot be moved: only a
            # new toplevel gets set_fullscreen() for the chosen output.
            self.hide()
        self.windowHandle().setScreen(screen)
        if self.settings.fullscreen:
            if not self.isVisible():
                self.setGeometry(screen.geometry())  # placement hint for X11
            self.showFullScreen()
        else:
            self.showNormal()
            self.resize(int(screen.geometry().width() * 0.4), int(screen.geometry().height() * 0.4))
        self.bring_to_front()

    def bring_to_front(self) -> None:
        """raise_() is a no-op on Wayland; activation goes through xdg-activation."""
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def is_in_front(self) -> bool:
        return self.isVisible() and self.isActiveWindow()

    def set_adapter(self, adapter: Adapter | None) -> None:
        self.adapter = adapter
        self.settings.rect_mm = None
        self.update()

    def set_display_color(self, rgb: tuple[float, float, float]) -> None:
        self._color = QColor.fromRgbF(*[max(0.0, min(1.0, v)) for v in rgb])
        self.update()

    def set_capturing(self, capturing: bool) -> None:
        if capturing and not self._capturing:
            # The pointer is drawn by the compositor right under the film.
            self._saved_cursor = self.cursor()
            self.setCursor(Qt.CursorShape.BlankCursor)
            self._drag = None
        elif not capturing and self._capturing:
            self.setCursor(getattr(self, "_saved_cursor", Qt.CursorShape.ArrowCursor))
        self._capturing = capturing
        self.update()

    @property
    def capturing(self) -> bool:
        return self._capturing

    def set_status(self, text: str) -> None:
        self._status = text
        self.update()

    def set_thumbnail(self, image: QImage | None) -> None:
        self._thumb = image
        self.update()

    def show_ruler(self, show: bool) -> None:
        """Scale calibration: stretch the 100 mm bar with the arrows to match a ruler."""
        self._ruler = show
        self.bring_to_front()
        self.update()

    # ------------------------------------------------------------ geometry
    def _screen(self) -> QScreen:
        return self.screen() or QGuiApplication.primaryScreen()

    def ppm(self) -> float:
        screen = self._screen()
        return px_per_mm(screen, self.settings.scale_correction.get(screen_key(screen), 1.0))

    def light_rect(self) -> QRectF:
        full = QRectF(self.rect())
        if not self.settings.fullscreen:
            return full
        ppm = self.ppm()
        if self.settings.rect_mm is not None:
            x, y, w, h = self.settings.rect_mm
            rect = QRectF(x * ppm, y * ppm, min(w * ppm, full.width()), min(h * ppm, full.height()))
            # Saved on another (bigger) screen: keep it reachable on this one.
            rect.moveLeft(min(max(rect.left(), 0.0), full.width() - rect.width()))
            rect.moveTop(min(max(rect.top(), 0.0), full.height() - rect.height()))
            return rect
        if self.adapter is None or self.adapter.light_mm is None:
            return full
        w, h = self.adapter.light_mm
        rect = QRectF(0, 0, w * ppm, h * ppm)
        rect.moveCenter(full.center())
        return rect

    def _store_rect(self, rect: QRectF) -> None:
        ppm = self.ppm()
        self.settings.rect_mm = (
            round(rect.x() / ppm, 2), round(rect.y() / ppm, 2), round(rect.width() / ppm, 2), round(rect.height() / ppm, 2),
        )
        self.settingsChanged.emit()
        self.update()

    # ------------------------------------------------------------ painting
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        light = self.light_rect()
        painter.fillRect(light, self._color)
        self._buttons = {}  # refilled only when the HUD is actually drawn
        if not self._capturing:
            self._paint_guides(painter, light)
            if self._ruler:
                self._paint_ruler(painter)
            if self.settings.hud:
                self._paint_hud(painter, light)
        painter.end()

    def _paint_guides(self, painter: QPainter, light: QRectF) -> None:
        ppm = self.ppm()
        if self.settings.guides and self.adapter and self.adapter.frame_mm:
            fw, fh = self.adapter.frame_mm
            frame = QRectF(0, 0, fw * ppm, fh * ppm)
            frame.moveCenter(light.center())
            guide = QColor(self._color)
            guide = guide.darker(125)
            pen = QPen(guide, 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(frame)
        if self.settings.marks and light != QRectF(self.rect()):
            painter.setPen(QPen(MARK_COLOR, 2))
            gap, size = 2 * ppm, 6 * ppm
            for corner, dx, dy in (
                (light.topLeft(), -1, -1), (light.topRight(), 1, -1),
                (light.bottomLeft(), -1, 1), (light.bottomRight(), 1, 1),
            ):
                cx, cy = corner.x() + dx * gap, corner.y() + dy * gap
                painter.drawLine(QPointF(cx, cy), QPointF(cx - dx * size, cy))
                painter.drawLine(QPointF(cx, cy), QPointF(cx, cy - dy * size))

    def _paint_ruler(self, painter: QPainter) -> None:
        ppm = self.ppm()
        length = 100 * ppm
        x0 = (self.width() - length) / 2
        y = self.height() * 0.12
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.drawLine(QPointF(x0, y), QPointF(x0 + length, y))
        for i in range(11):
            tick = 14 if i % 5 == 0 else 8
            painter.drawLine(QPointF(x0 + i * 10 * ppm, y - tick), QPointF(x0 + i * 10 * ppm, y))
        painter.setFont(QFont(painter.font().family(), 11))
        painter.drawText(QPointF(x0, y + 24), _("Esta barra debe medir 100 mm. Ponle una regla encima:"))
        painter.drawText(QPointF(x0, y + 44), _("←/→ la acortan o alargan (Mayús: fino) · Enter: aceptar"))
        correction = self.settings.scale_correction.get(screen_key(self._screen()), 1.0)
        painter.drawText(QPointF(x0, y + 64), _("Corrección: {c:+.1%}").format(c=correction - 1.0))

    def _hud_lines(self, light: QRectF) -> list[str]:
        ppm = self.ppm()
        lines = [
            "Belka",
            _("Espacio: capturar · Flechas: mover (Mayús ×5) · Ctrl+flechas: tamaño"),
            _("+/−: intensidad · G: guías · M: marcas · H: ocultar ayuda · Esc: salir"),
            _("F: flat-field (sin película) · T: calibrar tinte (con película)"),
            _("Luz: {w:.0f}×{h:.0f} mm · intensidad {b:.0%}").format(
                w=light.width() / ppm, h=light.height() / ppm, b=self.settings.brightness),
        ]
        if self._status:
            lines.append(self._status)
        return lines

    def _hud_place(self, light: QRectF, width: float, height: float) -> QPointF | None:
        """Top-left corner for a HUD block that stays clear of the lit area."""
        full = QRectF(self.rect())
        gap, margin = 24, 18
        regions = {
            "left": QRectF(0, 0, light.left() - gap, full.height()),
            "right": QRectF(light.right() + gap, 0, full.right() - light.right() - gap, full.height()),
            "top": QRectF(0, 0, full.width(), light.top() - gap),
            "bottom": QRectF(0, light.bottom() + gap, full.width(), full.bottom() - light.bottom() - gap),
        }
        fits = [(k, r) for k, r in regions.items() if r.width() >= width + 2 * margin and r.height() >= height + 2 * margin]
        if not fits:
            return None
        side, r = max(fits, key=lambda kr: kr[1].width() * kr[1].height())
        x = r.right() - width - margin if side == "right" else r.left() + margin
        y = r.top() + margin if side == "top" else r.bottom() - height - margin
        return QPointF(x, y)

    def _paint_hud(self, painter: QPainter, light: QRectF) -> None:
        self._buttons = {}
        if QRectF(self.rect()) == light:
            return
        painter.setFont(QFont(painter.font().family(), 10))
        metrics = painter.fontMetrics()
        lines = self._hud_lines(light)
        line_h = metrics.height() + 2
        text_w = max(metrics.horizontalAdvance(t) for t in lines)
        text_h = line_h * len(lines)
        bw, bh = 120, 34
        thumb = None
        if self._thumb is not None and not self._thumb.isNull():
            thumb = self._thumb.scaledToWidth(220, Qt.TransformationMode.SmoothTransformation)
        # Drop the thumbnail, then the buttons, until the HUD fits beside the light.
        layouts = [(True, True), (False, True), (False, False)] if thumb else [(False, True), (False, False)]
        for with_thumb, with_buttons in layouts:
            height = text_h + (bh + 14 if with_buttons else 0) + (thumb.height() + 12 if with_thumb else 0)
            width = max(text_w, 2 * bw + 10 if with_buttons else 0, 220 if with_thumb else 0)
            origin = self._hud_place(light, width, height)
            if origin is not None:
                break
        else:
            return
        x, y = origin.x(), origin.y()
        if with_thumb:
            painter.setOpacity(0.55)
            painter.drawImage(QPointF(x, y), thumb)
            painter.setOpacity(1.0)
            y += thumb.height() + 12
        if with_buttons:
            for i, (key, label) in enumerate((("capture", _("● Capturar")), ("close", _("✕ Salir")))):
                rect = QRectF(x + i * (bw + 10), y, bw, bh)
                painter.setPen(QPen(HUD_COLOR, 1))
                painter.drawRoundedRect(rect, 6, 6)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
                self._buttons[key] = rect
            y += bh + 14
        painter.setPen(HUD_COLOR)
        for i, text in enumerate(lines):
            painter.drawText(QPointF(x, y + metrics.ascent() + i * line_h), text)

    # ------------------------------------------------------------ input
    def _hit(self, pos: QPointF) -> str | None:
        rect = self.light_rect()
        if rect == QRectF(self.rect()) or not self.settings.fullscreen:
            return None
        near_l = abs(pos.x() - rect.left()) < EDGE
        near_r = abs(pos.x() - rect.right()) < EDGE
        near_t = abs(pos.y() - rect.top()) < EDGE
        near_b = abs(pos.y() - rect.bottom()) < EDGE
        inside_x = rect.left() - EDGE < pos.x() < rect.right() + EDGE
        inside_y = rect.top() - EDGE < pos.y() < rect.bottom() + EDGE
        if not (inside_x and inside_y):
            return None
        edge = ("t" if near_t else "b" if near_b else "") + ("l" if near_l else "r" if near_r else "")
        if edge:
            return edge
        return "move" if rect.contains(pos) else None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._capturing:
            return  # nothing may move or change while the shutter is open
        if self.settings.hud and event.button() == Qt.MouseButton.LeftButton:
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    if key == "capture":
                        self.captureRequested.emit()
                    else:
                        self.close()
                    return
        hit = self._hit(pos)
        if hit and event.button() == Qt.MouseButton.LeftButton:
            self._drag = (hit, pos, QRectF(self.light_rect()))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._capturing:
            return  # keep the pointer blank for the whole exposure
        if self._drag is None:
            hit = self._hit(pos)
            cursors = {
                "move": Qt.CursorShape.SizeAllCursor, "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
                "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
                "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
                "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
            }
            self.setCursor(cursors.get(hit or "", Qt.CursorShape.ArrowCursor))
            return
        mode, start, rect = self._drag
        d = pos - start
        r = QRectF(rect)
        if mode == "move":
            r.translate(d)
        else:
            if "l" in mode:
                r.setLeft(min(r.left() + d.x(), r.right() - 20))
            if "r" in mode:
                r.setRight(max(r.right() + d.x(), r.left() + 20))
            if "t" in mode:
                r.setTop(min(r.top() + d.y(), r.bottom() - 20))
            if "b" in mode:
                r.setBottom(max(r.bottom() + d.y(), r.top() + 20))
        self._store_rect(r)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._drag = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        mods = event.modifiers()
        step_mm = 5.0 if mods & Qt.KeyboardModifier.ShiftModifier else 1.0
        moves = {
            Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0), Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1),
        }
        action_keys = (Qt.Key.Key_F, Qt.Key.Key_T, Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter)
        if event.isAutoRepeat() and key in action_keys:
            return  # holding F must not arm and fire the flat with the film in place
        if self._capturing and key != Qt.Key.Key_Escape:
            return  # the light must not change mid-capture
        if self._ruler:
            self._ruler_key(event)
            return
        if key != Qt.Key.Key_F:
            self._flat_armed = False
        if key in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.captureRequested.emit()
        elif key == Qt.Key.Key_F:
            if self._flat_armed:
                self._flat_armed = False
                self.flatRequested.emit()
            else:
                self._flat_armed = True
                self.set_status(_("Flat-field: retira la película del soporte y pulsa F otra vez."))
        elif key == Qt.Key.Key_T:
            self.calibrateRequested.emit()
        elif key == Qt.Key.Key_Escape:
            self.close()
        elif key in moves and self.settings.fullscreen:
            dx, dy = moves[key]
            r = QRectF(self.light_rect())
            px = step_mm * self.ppm()
            if mods & Qt.KeyboardModifier.ControlModifier:
                r.setWidth(max(r.width() + dx * px, 20))
                r.setHeight(max(r.height() - dy * px, 20))
            else:
                r.translate(dx * px, dy * px)
            self._store_rect(r)
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.settings.brightness = min(1.0, round(self.settings.brightness + 0.05, 2))
            self.settingsChanged.emit()
        elif key == Qt.Key.Key_Minus:
            self.settings.brightness = max(0.05, round(self.settings.brightness - 0.05, 2))
            self.settingsChanged.emit()
        elif key == Qt.Key.Key_G:
            self.settings.guides = not self.settings.guides
            self.settingsChanged.emit()
        elif key == Qt.Key.Key_M:
            self.settings.marks = not self.settings.marks
            self.settingsChanged.emit()
        elif key == Qt.Key.Key_H:
            self.settings.hud = not self.settings.hud
            self.settingsChanged.emit()
        else:
            super().keyPressEvent(event)
        self.update()

    def _ruler_key(self, event: QKeyEvent) -> None:
        key = event.key()
        sk = screen_key(self._screen())
        step = 0.001 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.005
        current = self.settings.scale_correction.get(sk, 1.0)
        if key == Qt.Key.Key_Right:
            self.settings.scale_correction[sk] = round(current + step, 4)
        elif key == Qt.Key.Key_Left:
            self.settings.scale_correction[sk] = round(max(0.5, current - step), 4)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            self._ruler = False
        self.settingsChanged.emit()
        self.update()

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
