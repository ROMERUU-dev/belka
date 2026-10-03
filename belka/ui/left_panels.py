"""Left column of the Develop module, laid out like Lightroom Classic.

Navigator, film-profile browser, Snapshots and History, each a collapsible
block. The panels only show state and emit requests: the main window owns the
session, the history store and the image view, and wires them together.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime

from PySide6.QtCore import QEvent, QModelIndex, QPersistentModelIndex, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from belka.core.film import FILM_TYPES, FilmProfile, ProfileLibrary
from belka.core.history import History, split_label
from belka.i18n import _
from belka.ui.icons import icon

BG = "#1e1e1e"
PANEL = "#262626"
TEXT = "#b8b8b8"
BRIGHT = "#ececec"
DIM = "#6e6e6e"
LINE = "#141414"
HOVER = "#2c2c2c"
SELECTED = "#4a4a4a"
ROW_HEIGHT = 20
NAVIGATOR_SOURCE = 640  # long side kept from each render: plenty for a ~300 px panel on HiDPI

ZOOM_MODES = ("fit", "100", "200")

_ID_ROLE = Qt.ItemDataRole.UserRole
_SEARCH_ROLE = Qt.ItemDataRole.UserRole + 1

_QSS = f"""
#PanelBlock {{ background: {BG}; }}
#PanelBody {{ background: {BG}; }}
QToolButton#HeaderButton {{
    color: #7c7c7c; background: transparent; border: none; border-radius: 3px;
    padding: 1px 4px; font-size: 11px;
}}
QToolButton#HeaderButton:hover {{ color: {BRIGHT}; background: #333333; }}
QToolButton#HeaderButton:checked {{ color: {BRIGHT}; }}
QToolButton#HeaderButton:disabled {{ color: #454545; }}
QToolButton#ZoomButton {{
    color: #7c7c7c; background: transparent; border: none; border-radius: 3px;
    padding: 1px 2px; font-size: 10px;
}}
QToolButton#ZoomButton:hover {{ color: {BRIGHT}; background: #333333; }}
QToolButton#ZoomButton:checked {{ color: {BRIGHT}; }}
QLineEdit {{
    background: #151515; color: {TEXT}; border: 1px solid #383838; border-radius: 3px;
    padding: 2px 4px; font-size: 11px; selection-background-color: #5c5c5c;
}}
QLineEdit:focus {{ border-color: #686868; }}
QTreeWidget, QListWidget {{ background: {BG}; color: {TEXT}; border: none; outline: 0; font-size: 11px; }}
QTreeWidget::item, QListWidget::item {{ height: {ROW_HEIGHT}px; border: none; }}
QListWidget::item {{ padding: 0 8px; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: {HOVER}; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: {SELECTED}; color: {BRIGHT}; }}
QTreeWidget::branch {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #3a3a3a; border-radius: 3px; min-height: 24px; margin: 1px; }}
QScrollBar::handle:vertical:hover {{ background: #555555; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QMenu {{ background: #2b2b2b; color: {TEXT}; border: 1px solid #3c3c3c; padding: 3px 0; font-size: 12px; }}
QMenu::item {{ padding: 4px 18px; }}
QMenu::item:selected {{ background: {SELECTED}; color: {BRIGHT}; }}
QMenu::separator {{ height: 1px; background: #3c3c3c; margin: 3px 0; }}
"""


def _type_label(film_type: str) -> str:
    labels = {"color_negative": _("Negativo color"), "bw_negative": _("Blanco y negro"), "slide": _("Diapositiva")}
    return labels.get(film_type, film_type)


def _fold(text: str) -> str:
    """Lower case without accents, so "generico" finds "Genérico"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _paint_triangle(painter: QPainter, center: QPointF, expanded: bool, color: QColor) -> None:
    """Lightroom's small filled disclosure triangle: ▾ open, ▸ closed."""
    x, y = center.x(), center.y()
    path = QPainterPath()
    if expanded:
        path.moveTo(x - 3.5, y - 1.75)
        path.lineTo(x + 3.5, y - 1.75)
        path.lineTo(x, y + 2.25)
    else:
        path.moveTo(x - 1.75, y - 3.5)
        path.lineTo(x - 1.75, y + 3.5)
        path.lineTo(x + 2.25, y)
    path.closeSubpath()
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawPath(path)
    painter.restore()


def _search_icon() -> QIcon:
    pix = QPixmap(28, 28)
    pix.setDevicePixelRatio(2.0)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(DIM), 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawEllipse(QRectF(2.5, 2.5, 7, 7))
    painter.drawLine(QPointF(8.6, 8.6), QPointF(11.5, 11.5))
    painter.end()
    return QIcon(pix)


def _header_button(icon_name: str, fallback: str, tooltip: str) -> QToolButton:
    """Small flat button for a block header; text if the icon set lacks the glyph."""
    button = QToolButton()
    button.setObjectName("HeaderButton")
    button.setToolTip(tooltip)
    button.setAutoRaise(True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    glyph = icon(icon_name, color="#9a9a9a", disabled="#454545", active=BRIGHT)
    if glyph.isNull():
        button.setText(fallback)
    else:
        button.setIcon(glyph)
        button.setIconSize(QSize(14, 14))
    return button


# ---------------------------------------------------------------- blocks

class _Header(QWidget):
    """Title bar of a block: triangle and title on the left, tools on the right."""

    clicked = Signal()

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.title = title
        self.expanded = True
        self.setFixedHeight(26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tools = QHBoxLayout(self)
        self.tools.setContentsMargins(0, 0, 6, 0)
        self.tools.setSpacing(0)
        self.tools.addStretch(1)
        self._font = QFont(self.font())
        self._font.setPixelSize(12)
        self._font.setWeight(QFont.Weight.DemiBold)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        r = self.rect()
        painter.fillRect(r, QColor(PANEL))
        painter.fillRect(QRect(r.left(), r.top(), r.width(), 1), QColor("#303030"))
        painter.fillRect(QRect(r.left(), r.bottom(), r.width(), 1), QColor(LINE))
        _paint_triangle(painter, QPointF(12, r.height() / 2), self.expanded, QColor("#8a8a8a"))
        right = r.right() - 6
        for i in range(self.tools.count()):
            widget = self.tools.itemAt(i).widget()
            if widget is not None and widget.isVisible():
                right = min(right, widget.geometry().left() - 4)
        painter.setFont(self._font)
        painter.setPen(QColor("#c9c9c9"))
        text_rect = QRect(24, 0, max(0, right - 24), r.height())
        elided = painter.fontMetrics().elidedText(self.title, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided)


class PanelBlock(QFrame):
    """Collapsible block: header (triangle, title, optional tools) over a body."""

    toggled = Signal(bool)

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("PanelBlock")
        self.setStyleSheet(_QSS)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = _Header(title)
        self.header.clicked.connect(lambda: self.set_expanded(not self.is_expanded()))
        self.body = QWidget()
        self.body.setObjectName("PanelBody")
        self.body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 7, 8, 9)
        self.body_layout.setSpacing(6)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        # Blocks keep their natural height; only one (History) soaks up the
        # spare column height, so a collapsed neighbour never stretches.
        self._expanded_policy = QSizePolicy.Policy.Fixed
        self.set_expanded(True)

    def add_header_widget(self, widget: QWidget) -> None:
        self.header.tools.addWidget(widget)

    def set_body_expanding(self, expanding: bool) -> None:
        """Let this block take the column's spare height while it is open."""
        self._expanded_policy = QSizePolicy.Policy.Expanding if expanding else QSizePolicy.Policy.Fixed
        self.set_expanded(self.is_expanded())

    def is_expanded(self) -> bool:
        return self.header.expanded

    def set_expanded(self, expanded: bool) -> None:
        changed = expanded != self.header.expanded
        self.header.expanded = expanded
        self.body.setVisible(expanded)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           self._expanded_policy if expanded else QSizePolicy.Policy.Fixed)
        self.header.update()
        if changed:
            self.toggled.emit(expanded)


class _List(QListWidget):
    """Flat list with a dim hint when empty."""

    def __init__(self, placeholder: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.placeholder = placeholder
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setUniformItemSizes(True)
        self.setMouseTracking(True)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.count() == 0 and self.placeholder:
            painter = QPainter(self.viewport())
            painter.setPen(QColor(DIM))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)

    def fit_rows(self, minimum: int, maximum: int) -> None:
        rows = min(max(self.count(), minimum), maximum)
        self.setFixedHeight(rows * ROW_HEIGHT + 2 * self.frameWidth())


# ---------------------------------------------------------------- navigator

class _NavigatorView(QWidget):
    """Thumbnail with the visible area outlined; click or drag to move it."""

    panRequested = Signal(float, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._scaled: QPixmap | None = None
        self._view: tuple[float, float, float, float] | None = None
        self._dragging = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(240, 160)

    def set_image(self, image: QImage | None) -> None:
        # Keep a small copy of our own: the caller's QImage may wrap a numpy
        # buffer that is freed once its render is superseded.
        if image is None or image.isNull():
            self._image = None
        elif max(image.width(), image.height()) > NAVIGATOR_SOURCE:
            self._image = image.scaled(QSize(NAVIGATOR_SOURCE, NAVIGATOR_SOURCE), Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
        else:
            self._image = image.copy()
        self._scaled = None
        self.setCursor(Qt.CursorShape.OpenHandCursor if self._image is not None else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_view_rect(self, rect: tuple[float, float, float, float] | None) -> None:
        if rect is not None:
            x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in rect)
            # The whole image on screen needs no outline, as in Lightroom; an
            # empty one is what the view reports when it has no image.
            whole = x1 - x0 >= 0.999 and y1 - y0 >= 0.999
            rect = None if whole or x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)
        self._view = rect
        self.update()

    def view_rect(self) -> tuple[float, float, float, float] | None:
        return self._view

    def image_rect(self) -> QRectF:
        """Where the thumbnail is drawn, letterboxed in the widget."""
        if self._image is None:
            return QRectF()
        area = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        iw, ih = self._image.width(), self._image.height()
        scale = min(area.width() / iw, area.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF(area.center().x() - w / 2, area.center().y() - h / 2, w, h)

    def resizeEvent(self, event) -> None:
        self._scaled = None
        # A 3:2 well like Lightroom's. A fixed height rather than
        # heightForWidth, which fixed-height blocks would centre in a gap.
        self.setFixedHeight(max(90, round(self._column_width() * 2 / 3)))
        super().resizeEvent(event)

    def _column_width(self) -> int:
        """Our width as if the enclosing scroll area showed no scroll bar.

        Taken from our own width, the well would shrink when the bar appears,
        the column would then fit and hide the bar, the well would grow back,
        and so on without end at some column heights.
        """
        node = self.parentWidget()
        while node is not None and not (isinstance(node, QScrollArea) and node.widget() is not None):
            node = node.parentWidget()
        if node is None:
            return self.width()
        # Against the content, not the viewport: the area resizes its content
        # before the viewport when the bar hides, so the viewport lags behind.
        return self.width() + node.maximumViewportSize().width() - node.widget().width()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171717"))
        if self._image is None:
            painter.setPen(QColor(DIM))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, _("Sin imagen"))
            return
        target = self.image_rect()
        if self._scaled is None:
            ratio = self.devicePixelRatioF()
            size = QSize(max(1, round(target.width() * ratio)), max(1, round(target.height() * ratio)))
            self._scaled = QPixmap.fromImage(self._image.scaled(
                size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._scaled.setDevicePixelRatio(ratio)
        painter.drawPixmap(target.topLeft(), self._scaled)
        if self._view is None:
            return
        x0, y0, x1, y1 = self._view
        box = QRectF(target.left() + x0 * target.width(), target.top() + y0 * target.height(),
                     (x1 - x0) * target.width(), (y1 - y0) * target.height())
        outside = QPainterPath()
        outside.addRect(target)
        inner = QPainterPath()
        inner.addRect(box)
        painter.fillPath(outside.subtracted(inner), QColor(0, 0, 0, 110))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3))
        painter.drawRect(box.adjusted(0.5, 0.5, -0.5, -0.5))
        painter.setPen(QPen(QColor(245, 245, 245), 1))
        painter.drawRect(box.adjusted(0.5, 0.5, -0.5, -0.5))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._image is not None:
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._pan_to(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._pan_to(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def _pan_to(self, pos: QPointF) -> None:
        target = self.image_rect()
        if target.isEmpty():
            return
        x = min(max((pos.x() - target.left()) / target.width(), 0.0), 1.0)
        y = min(max((pos.y() - target.top()) / target.height(), 0.0), 1.0)
        if self._view is not None:
            # Move the outline at once instead of waiting for the view's
            # echo, keeping it inside the image like the view will.
            x0, y0, x1, y1 = self._view
            w, h = x1 - x0, y1 - y0
            x0 = min(max(x - w / 2, 0.0), 1.0 - w)
            y0 = min(max(y - h / 2, 0.0), 1.0 - h)
            self._view = (x0, y0, x0 + w, y0 + h)
            x, y = x0 + w / 2, y0 + h / 2
            self.update()
        self.panRequested.emit(x, y)


class NavigatorPanel(PanelBlock):
    """Lightroom's Navigator: preview of the frame, visible area, zoom presets.

    ``panRequested(x, y)`` gives the new centre of the view in normalised
    image coordinates; ``zoomRequested`` one of "fit", "100", "200".
    """

    panRequested = Signal(float, float)
    zoomRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(_("Navegador"), parent)
        self.view = _NavigatorView()
        self.view.panRequested.connect(self.panRequested)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.addWidget(self.view)
        self.zoom_buttons: dict[str, QToolButton] = {}
        for mode, text in zip(ZOOM_MODES, (_("Ajustar"), _("100 %"), _("200 %"))):
            button = QToolButton()
            button.setObjectName("ZoomButton")
            button.setText(text)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, m=mode: self._zoom_clicked(m))
            self.add_header_widget(button)
            self.zoom_buttons[mode] = button
        self.zoom_buttons["fit"].setToolTip(_("Ver la imagen completa (Z alterna)"))
        self.set_zoom("fit")

    def set_image(self, image: QImage | None) -> None:
        self.view.set_image(image)

    def set_viewport(self, rect: QRectF | tuple[float, float, float, float] | None) -> None:
        """Visible area, normalised: ``ImageView.viewportChanged``'s QRectF or (x0, y0, x1, y1).

        None or the whole image hides the outline.
        """
        if isinstance(rect, QRectF):
            rect = (rect.left(), rect.top(), rect.right(), rect.bottom())
        self.view.set_view_rect(rect)

    def set_zoom(self, mode: str | None) -> None:
        """Highlight the active preset; any other zoom (wheel) highlights none."""
        for key, button in self.zoom_buttons.items():
            button.setChecked(key == mode)

    def _zoom_clicked(self, mode: str) -> None:
        self.set_zoom(mode)
        self.zoomRequested.emit(mode)


# ---------------------------------------------------------------- profiles

class _ProfileTree(QTreeWidget):
    """Tree with Lightroom-style triangles, full-row highlight and a height that follows its rows."""

    def __init__(self, placeholder: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.placeholder = placeholder
        self.setColumnCount(2)
        self.setHeaderHidden(True)
        self.setIndentation(14)
        self.setRootIsDecorated(True)
        self.setExpandsOnDoubleClick(False)
        self.setUniformRowHeights(True)
        self.setMouseTracking(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, header.ResizeMode.Stretch)
        header.setSectionResizeMode(1, header.ResizeMode.Fixed)
        header.resizeSection(1, 30)
        self._hover = QPersistentModelIndex()
        self.itemExpanded.connect(self.fit_rows)
        self.itemCollapsed.connect(self.fit_rows)

    def visible_rows(self, item: QTreeWidgetItem | None = None) -> int:
        item = item or self.invisibleRootItem()
        rows = 0
        for i in range(item.childCount()):
            child = item.child(i)
            if not child.isHidden():
                rows += 1 + (self.visible_rows(child) if child.isExpanded() else 0)
        return rows

    def fit_rows(self) -> None:
        """Grow with the open folders up to a limit, then scroll."""
        rows = min(max(self.visible_rows(), 3), 14)
        self.setFixedHeight(rows * ROW_HEIGHT + 2 * self.frameWidth())

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.visible_rows() == 0:
            painter = QPainter(self.viewport())
            painter.setPen(QColor(DIM))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        row = self.indexAt(event.position().toPoint()).siblingAtColumn(0)
        if row != QModelIndex(self._hover):
            self._hover = QPersistentModelIndex(row)
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = QPersistentModelIndex()
        self.viewport().update()
        super().leaveEvent(event)

    def drawBranches(self, painter: QPainter, rect: QRect, index: QModelIndex) -> None:
        # The stylesheet highlights only the cells; extend it under the indent.
        if self.selectionModel().isSelected(index):
            painter.fillRect(rect, QColor(SELECTED))
        elif index == QModelIndex(self._hover):
            painter.fillRect(rect, QColor(HOVER))
        if self.model().hasChildren(index):
            center = QPointF(rect.right() - 7, rect.center().y() + 0.5)
            _paint_triangle(painter, center, self.isExpanded(index), QColor("#8a8a8a"))


class ProfileBrowserPanel(PanelBlock):
    """Film profiles grouped by type and brand, the user's own first, with search.

    A single click on a profile emits ``profileChosen(id)``.
    """

    profileChosen = Signal(str)

    def __init__(self, library: ProfileLibrary, parent: QWidget | None = None):
        super().__init__(_("Perfiles de película"), parent)
        self.library = library
        self._current: str | None = None
        self.search = QLineEdit()
        self.search.setPlaceholderText(_("Buscar perfil"))
        self.search.setClearButtonEnabled(True)
        self.search.addAction(_search_icon(), QLineEdit.ActionPosition.LeadingPosition)
        self.search.textChanged.connect(self._apply_filter)
        self.tree = _ProfileTree(_("Ningún perfil coincide"))
        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.itemActivated.connect(self._on_activated)
        self._mine: QTreeWidgetItem | None = None
        self.body_layout.addWidget(self.search)
        self.body_layout.addWidget(self.tree)
        self.reload()

    def reload(self) -> None:
        """Rebuild the tree from the library (after saving or deleting a profile)."""
        self.tree.clear()
        profiles = self.library.all()
        mine = [p for p in profiles if not p.builtin]
        self._mine = self._group(self.tree.invisibleRootItem(), _("Mis perfiles"), bold=True) if mine else None
        for profile in mine:
            self._leaf(self._mine, profile)
        for film_type in FILM_TYPES:
            of_type = [p for p in profiles if p.builtin and p.type == film_type]
            if not of_type:
                continue
            group = self._group(self.tree.invisibleRootItem(), _type_label(film_type), bold=True)
            brands: dict[str, QTreeWidgetItem] = {}
            # Generic stocks sit directly under the type, before the brand folders.
            for profile in sorted(of_type, key=lambda p: bool(p.brand)):
                parent = group
                if profile.brand:
                    if profile.brand not in brands:
                        brands[profile.brand] = self._group(group, profile.brand)
                    parent = brands[profile.brand]
                self._leaf(parent, profile)
        self._apply_filter(self.search.text())

    def set_current(self, profile_id: str | None) -> None:
        """Select the frame's profile and open the folders leading to it, without emitting.

        Folders the user opened stay open while moving between frames, and the
        profile already shown changes nothing, so a folder the user closed
        stays closed when the caller repeats this after every edit.
        """
        item = self._find(profile_id)
        if profile_id == self._current and item is not None and item is self.tree.currentItem():
            return
        self._current = profile_id
        if item is None:
            self.tree.clearSelection()
            return
        node = item.parent()
        while node is not None:
            node.setExpanded(True)
            node = node.parent()
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)

    def current(self) -> str | None:
        return self._current

    def _group(self, parent: QTreeWidgetItem, title: str, bold: bool = False) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [title, ""])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        item.setForeground(1, QColor(DIM))
        font = QFont(self.tree.font())
        font.setPixelSize(11)
        if bold:
            font.setWeight(QFont.Weight.DemiBold)
            item.setForeground(0, QColor("#c9c9c9"))
        item.setFont(0, font)
        item.setFont(1, font)
        return item

    def _leaf(self, parent: QTreeWidgetItem, profile: FilmProfile) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [profile.name, ""])
        item.setData(0, _ID_ROLE, profile.id)
        words = [profile.name, profile.brand, profile.process, str(profile.iso or ""),
                 _type_label(profile.type), profile.id]
        item.setData(0, _SEARCH_ROLE, _fold(" ".join(words)))
        iso = f"ISO {profile.iso}" if profile.iso else ""
        details = " · ".join(v for v in (profile.brand, profile.process, iso) if v)
        item.setToolTip(0, "\n".join(v for v in (profile.name, details, profile.notes) if v))
        return item

    def _items(self, root: QTreeWidgetItem | None = None):
        root = root or self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            child = root.child(i)
            yield child
            yield from self._items(child)

    def _find(self, profile_id: str | None) -> QTreeWidgetItem | None:
        if not profile_id:
            return None
        return next((it for it in self._items() if it.data(0, _ID_ROLE) == profile_id), None)

    def _default_expansion(self) -> None:
        """Only the folders leading to the current profile, and the user's own, are open."""
        current = self._find(self._current)
        open_items = set()
        node = current.parent() if current is not None else None
        while node is not None:
            open_items.add(id(node))
            node = node.parent()
        for item in self._items():
            if item.childCount():
                item.setExpanded(id(item) in open_items or item is self._mine)
        self.tree.fit_rows()

    def _apply_filter(self, text: str) -> None:
        words = _fold(text).split()

        def visit(item: QTreeWidgetItem) -> int:
            if item.childCount() == 0:
                haystack = item.data(0, _SEARCH_ROLE) or ""
                match = all(w in haystack for w in words)
                item.setHidden(not match)
                return int(match)
            count = sum(visit(item.child(i)) for i in range(item.childCount()))
            item.setText(1, str(count) if count else "")
            item.setHidden(count == 0)
            return count

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            visit(root.child(i))
        if words:
            self.tree.expandAll()
            self.tree.fit_rows()
        else:
            self._default_expansion()

    def _on_clicked(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        profile_id = item.data(0, _ID_ROLE)
        if profile_id:
            self._current = profile_id
            self.profileChosen.emit(profile_id)
        else:
            item.setExpanded(not item.isExpanded())

    def _on_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        """Enter on a profile; the click of a double-click has already chosen it."""
        profile_id = item.data(0, _ID_ROLE)
        if profile_id and profile_id != self._current:
            self._on_clicked(item)


# ---------------------------------------------------------------- snapshots

class SnapshotsPanel(PanelBlock):
    """Named develop states of the frame (``Frame.snapshots``).

    '+' opens an inline name field (Enter or clicking elsewhere creates,
    Esc cancels); a click applies; the context menu, '−' or Supr deletes.
    Indices refer to the order given to :meth:`set_snapshots`.
    """

    snapshotCreate = Signal(str)
    snapshotApply = Signal(int)
    snapshotDelete = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(_("Instantáneas"), parent)
        self._editing = False
        self._rows: list[tuple[str, str]] | None = None  # (name, created) as last shown
        self.add_button = _header_button("plus", "+", _("Crear instantánea con los ajustes actuales"))
        self.add_button.clicked.connect(lambda: self.begin_create())
        self.remove_button = _header_button("minus", "−", _("Eliminar la instantánea seleccionada"))
        self.remove_button.clicked.connect(self._delete_selected)
        self.add_header_widget(self.add_button)
        self.add_header_widget(self.remove_button)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(_("Nombre de la instantánea"))
        self.name_edit.hide()
        self.name_edit.returnPressed.connect(self._commit)
        self.name_edit.installEventFilter(self)
        self.list = _List(_("Sin instantáneas"))
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._show_menu)
        self.list.itemClicked.connect(lambda item: self.snapshotApply.emit(item.data(_ID_ROLE)))
        self.list.itemSelectionChanged.connect(self._update_buttons)
        self.list.installEventFilter(self)
        # Rows run edge to edge like History's; only the name field is inset.
        self.body_layout.setContentsMargins(0, 4, 0, 6)
        edit_row = QHBoxLayout()
        edit_row.setContentsMargins(8, 0, 8, 2)
        edit_row.addWidget(self.name_edit)
        self.body_layout.addLayout(edit_row)
        self.body_layout.addWidget(self.list)
        self.set_snapshots([])

    def set_snapshots(self, snapshots: list[dict] | list[str]) -> None:
        """``Frame.snapshots`` as stored ({"name", "created", "settings"}) or just their names.

        The selection survives: the main window refreshes this list after
        every edit, including the one a click on a snapshot makes, and '−' or
        Supr must still find that snapshot selected.
        """
        rows = []
        for snap in snapshots:
            snap = snap if isinstance(snap, dict) else {"name": snap}
            rows.append((str(snap.get("name") or _("Sin nombre")), str(snap.get("created") or "")))
        if rows == self._rows:
            return
        selected = self.list.selectedItems()
        row = self.list.row(selected[0]) if selected else -1
        kept = self._rows[row] if row >= 0 else None
        self._rows = rows
        self.list.clear()
        for index, (name, created) in enumerate(rows):
            item = QListWidgetItem(name)
            item.setData(_ID_ROLE, index)
            item.setToolTip(f"{name}\n{created.replace('T', ' ')}" if created else name)
            self.list.addItem(item)
        if kept in rows:
            self.list.setCurrentRow(row if rows[row:row + 1] == [kept] else rows.index(kept))
        self.list.fit_rows(1, 8)
        self._update_buttons()

    def begin_create(self, name: str | None = None) -> None:
        """Show the name field with a date-and-time default, like Lightroom."""
        self.set_expanded(True)
        self._editing = True
        self.name_edit.setText(name or datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        self.name_edit.show()
        self.name_edit.selectAll()
        self.name_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def cancel_create(self) -> None:
        self._editing = False
        self.name_edit.hide()

    def _commit(self) -> None:
        if not self._editing:
            return
        name = self.name_edit.text().strip() or datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.cancel_create()
        self.snapshotCreate.emit(name)

    def eventFilter(self, obj, event) -> bool:
        # The window binds Esc (cancel tool) and may bind Enter; QLineEdit
        # does not claim those keys, so claim them here or the field never
        # sees them.
        if (obj is self.name_edit and event.type() == QEvent.Type.ShortcutOverride
                and event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter)):
            event.accept()
            return True
        # Leaving the field keeps the snapshot, as in Lightroom; a context menu
        # or switching windows only borrows the focus.
        if obj is self.name_edit and event.type() == QEvent.Type.FocusOut and event.reason() not in (
                Qt.FocusReason.PopupFocusReason, Qt.FocusReason.ActiveWindowFocusReason):
            self._commit()
        elif event.type() == QEvent.Type.KeyPress:
            if obj is self.name_edit and event.key() == Qt.Key.Key_Escape:
                self.cancel_create()
                return True
            if obj is self.list and event.key() == Qt.Key.Key_Delete:
                self._delete_selected()
                return True
        return super().eventFilter(obj, event)

    def context_menu(self, item: QListWidgetItem) -> QMenu:
        index = item.data(_ID_ROLE)
        menu = QMenu(self)
        menu.addAction(_("Aplicar"), lambda: self.snapshotApply.emit(index))
        menu.addSeparator()
        menu.addAction(_("Eliminar"), lambda: self.snapshotDelete.emit(index))
        return menu

    def _show_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is not None:
            self.list.setCurrentItem(item)
            self.context_menu(item).exec(self.list.viewport().mapToGlobal(pos))

    def _delete_selected(self) -> None:
        item = self.list.currentItem()
        if item is not None and item.isSelected():
            self.snapshotDelete.emit(item.data(_ID_ROLE))

    def _update_buttons(self) -> None:
        self.remove_button.setEnabled(bool(self.list.selectedItems()))


# ---------------------------------------------------------------- history

class _HistoryDelegate(QStyledItemDelegate):
    """Step name on the left, its value right-aligned; undone steps dimmed."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.current = -1

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(option.rect.width(), ROW_HEIGHT)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        painter.save()
        rect = option.rect
        step = index.data(_ID_ROLE)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if selected:
            painter.fillRect(rect, QColor(SELECTED))
        elif option.state & QStyle.StateFlag.State_MouseOver:
            painter.fillRect(rect, QColor(HOVER))
        name, value = split_label(index.data(Qt.ItemDataRole.DisplayRole) or "")
        undone = step > self.current
        painter.setFont(option.font)
        metrics = painter.fontMetrics()
        inner = rect.adjusted(11, 0, -10, 0)  # lines up with the Snapshots rows
        value_width = metrics.horizontalAdvance(value) if value else 0
        if value:
            painter.setPen(QColor(BRIGHT if selected else ("#555555" if undone else "#8c8c8c")))
            painter.drawText(inner, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, value)
        name_rect = inner.adjusted(0, 0, -(value_width + 10 if value else 0), 0)
        painter.setPen(QColor(BRIGHT if selected else (DIM if undone else TEXT)))
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         metrics.elidedText(name, Qt.TextElideMode.ElideRight, name_rect.width()))
        painter.restore()


class HistoryPanel(PanelBlock):
    """Lightroom's History: newest step on top, the current one highlighted.

    ``historyJump(i)`` uses the history's own order (0 = oldest step).
    """

    historyJump = Signal(int)
    historyClear = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(_("Historial"), parent)
        self.clear_button = _header_button("close", "×", _("Borrar el historial (conserva el estado actual)"))
        self.clear_button.clicked.connect(self.historyClear)
        self.add_header_widget(self.clear_button)
        self.list = _List(_("Sin pasos todavía"))
        self.delegate = _HistoryDelegate(self.list)
        self.list.setItemDelegate(self.delegate)
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.list.setMinimumHeight(6 * ROW_HEIGHT)
        self.list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.list.itemClicked.connect(lambda item: self.historyJump.emit(item.data(_ID_ROLE)))
        self.body_layout.setContentsMargins(0, 4, 0, 6)
        self.body_layout.addWidget(self.list)
        self.set_body_expanding(True)
        self.set_entries([], -1)

    def set_history(self, history: History | None) -> None:
        if history is None:
            self.set_entries([], -1)
        else:
            self.set_entries([e.label for e in history.entries()], history.index)

    def set_entries(self, labels: list[str], current: int) -> None:
        """Labels oldest first, as :meth:`History.entries` returns them."""
        self.delegate.current = current
        self.list.clear()
        current_item = None
        for step in range(len(labels) - 1, -1, -1):
            item = QListWidgetItem(labels[step])
            item.setData(_ID_ROLE, step)
            item.setToolTip(labels[step])
            self.list.addItem(item)
            if step == current:
                current_item = item
        if current_item is not None:
            self.list.setCurrentItem(current_item)
            self.list.scrollToItem(current_item)
        self.clear_button.setEnabled(len(labels) > 1)


# ---------------------------------------------------------------- column

class LeftColumn(QScrollArea):
    """The four panels stacked like Lightroom Classic; History takes the spare height."""

    def __init__(self, library: ProfileLibrary, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(220)
        self.setStyleSheet(f"QScrollArea {{ background: {BG}; border: none; }}" + _QSS)
        body = QWidget()
        body.setObjectName("PanelBlock")
        self.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.navigator = NavigatorPanel()
        self.profiles = ProfileBrowserPanel(library)
        self.snapshots = SnapshotsPanel()
        self.history = HistoryPanel()
        for panel in (self.navigator, self.profiles, self.snapshots):
            layout.addWidget(panel)
        layout.addWidget(self.history, 1)
        layout.addStretch(0)
