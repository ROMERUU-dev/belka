"""Thumbnails of the roll: the bottom filmstrip and the Library grid.

Both are the same list widget in two sizes; a delegate paints the frame
number, the star rating and the pick/reject flag the way Lightroom does.
"""

from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QListView,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
)

from belka.core.session import Frame

ROLE_ID = Qt.ItemDataRole.UserRole
ROLE_RATING = Qt.ItemDataRole.UserRole + 1
ROLE_FLAG = Qt.ItemDataRole.UserRole + 2
ROLE_LABEL = Qt.ItemDataRole.UserRole + 3

CELL_BG = QColor(42, 42, 44)
CELL_SELECTED = QColor(78, 78, 82)
CELL_CURRENT = QColor(214, 120, 40)
TEXT = QColor(170, 170, 170)
STAR_ON = QColor(225, 225, 225)
STAR_OFF = QColor(90, 90, 92)


class FrameDelegate(QStyledItemDelegate):
    def __init__(self, thumb: QSize, parent=None) -> None:
        super().__init__(parent)
        self.thumb = thumb

    def sizeHint(self, option, index) -> QSize:
        return QSize(self.thumb.width() + 16, self.thumb.height() + 34)

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(3, 3, -3, -3)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.fillRect(rect, CELL_SELECTED if selected else CELL_BG)
        if index.data(Qt.ItemDataRole.UserRole + 10):  # current frame
            painter.setPen(QPen(CELL_CURRENT, 2))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))
        icon: QIcon = index.data(Qt.ItemDataRole.DecorationRole)
        area = QRect(rect.left() + 5, rect.top() + 5, rect.width() - 10, rect.height() - 34)
        if icon is not None and not icon.isNull():
            pix = icon.pixmap(area.size())
            x = area.left() + (area.width() - pix.width()) // 2
            y = area.top() + (area.height() - pix.height()) // 2
            painter.drawPixmap(x, y, pix)
        flag = int(index.data(ROLE_FLAG) or 0)
        if flag == -1:
            # Rejected: dim the picture like Lightroom does.
            painter.fillRect(area, QColor(0, 0, 0, 140))
        font = QFont(painter.font())
        font.setPointSizeF(max(font.pointSizeF() - 1.5, 7.5))
        painter.setFont(font)
        painter.setPen(TEXT)
        text_rect = QRect(rect.left() + 6, rect.bottom() - 24, rect.width() - 12, 20)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, index.data(ROLE_LABEL) or "")
        rating = int(index.data(ROLE_RATING) or 0)
        stars = "".join("★" if i < rating else "·" for i in range(5))
        painter.setPen(STAR_ON if rating else STAR_OFF)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, stars)
        if flag:
            painter.setPen(QColor(235, 235, 235) if flag == 1 else QColor(220, 80, 70))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "⚑" if flag == 1 else "⊘")
        painter.restore()


class Filmstrip(QListWidget):
    frameSelected = Signal(str)
    frameActivated = Signal(str)  # double click: open in Develop
    selectionCountChanged = Signal(int)

    def __init__(self, thumb: QSize = QSize(132, 88), grid: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.thumb = thumb
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(grid)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setIconSize(thumb)
        self.setSpacing(2)
        self.setUniformItemSizes(True)
        self.setItemDelegate(FrameDelegate(thumb, self))
        if not grid:
            self.setFixedHeight(thumb.height() + 50)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setStyleSheet("QListWidget { background: #1b1b1c; border: none; }")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._current_id: str | None = None
        self.currentItemChanged.connect(self._on_current)
        self.itemDoubleClicked.connect(lambda item: self.frameActivated.emit(item.data(ROLE_ID)))
        self.itemSelectionChanged.connect(lambda: self.selectionCountChanged.emit(len(self.selectedItems())))

    def _placeholder(self) -> QIcon:
        pix = QPixmap(self.thumb)
        pix.fill(QColor(32, 32, 34))
        return QIcon(pix)

    def set_frames(self, frames: list[Frame]) -> None:
        self.blockSignals(True)
        self.clear()
        for frame in frames:
            self.add_frame(frame, select=False)
        self.blockSignals(False)

    def add_frame(self, frame: Frame, select: bool = True) -> None:
        item = QListWidgetItem(self._placeholder(), "")
        item.setData(ROLE_ID, frame.id)
        item.setData(ROLE_LABEL, frame.label)
        item.setData(ROLE_RATING, frame.rating)
        item.setData(ROLE_FLAG, frame.flag)
        item.setToolTip(frame.label)
        self.addItem(item)
        if select:
            self.setCurrentItem(item, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            self.scrollToItem(item)

    def update_frame(self, frame: Frame) -> None:
        item = self._item(frame.id)
        if item is not None:
            item.setData(ROLE_RATING, frame.rating)
            item.setData(ROLE_FLAG, frame.flag)
            item.setData(ROLE_LABEL, frame.label)

    def set_thumbnail(self, frame_id: str, image: QImage) -> None:
        item = self._item(frame_id)
        if item is None:
            return
        pix = QPixmap.fromImage(image).scaled(self.thumb, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
        item.setIcon(QIcon(pix))

    def remove_frame(self, frame_id: str) -> None:
        item = self._item(frame_id)
        if item is not None:
            self.takeItem(self.row(item))

    def _item(self, frame_id: str) -> QListWidgetItem | None:
        for i in range(self.count()):
            item = self.item(i)
            if item.data(ROLE_ID) == frame_id:
                return item
        return None

    def select_frame(self, frame_id: str, keep_selection: bool = False) -> None:
        """Make ``frame_id`` current. ``keep_selection`` leaves a selection that already
        holds it alone (the user's multi-selection for Sync or Delete)."""
        item = self._item(frame_id)
        if item is None:
            return
        if keep_selection and item.isSelected():
            self.setCurrentItem(item, QItemSelectionModel.SelectionFlag.NoUpdate)
        else:
            # ExtendedSelection would otherwise keep adding every frame the app
            # selects (captures, imports) to the selection Delete acts on.
            self.setCurrentItem(item, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        self.scrollToItem(item)

    def item_id_at(self, pos) -> str | None:
        item = self.itemAt(pos)
        return None if item is None else item.data(ROLE_ID)

    def mark_current(self, frame_id: str | None) -> None:
        for i in range(self.count()):
            item = self.item(i)
            item.setData(Qt.ItemDataRole.UserRole + 10, item.data(ROLE_ID) == frame_id)

    def selected_ids(self) -> list[str]:
        return [i.data(ROLE_ID) for i in self.selectedItems()]

    def _on_current(self, current: QListWidgetItem | None, _previous) -> None:
        # Ctrl/Shift+click only extends the selection; the photo being developed
        # stays the active one, as in Lightroom (Sync copies from it).
        modifiers = QApplication.keyboardModifiers()
        if current is not None and not modifiers & (Qt.KeyboardModifier.ControlModifier
                                                    | Qt.KeyboardModifier.ShiftModifier):
            self.frameSelected.emit(current.data(ROLE_ID))
