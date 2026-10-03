"""Collapsible panels in the manner of Lightroom Classic's right-hand column.

A :class:`Section` has a header with the on/off switch on the left, the
title on the right and a disclosure triangle; clicking the header folds it,
double-clicking the title (only the title) resets it, right-clicking offers
the panel menu (reset, expand/collapse all, solo mode). :class:`SectionList`
stacks them and keeps the collapse state.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractButton,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from belka.i18n import _
from belka.ui.widgets import ClickLabel

# Lightroom Classic greys. Panels sit on the darker ground, separated by it.
GROUND = "#1e1e1e"
PANEL = "#262626"
HEADER = "#2b2b2b"
TEXT = "#b8b8b8"
TEXT_BRIGHT = "#d6d6d6"
TEXT_DIM = "#7a7a7a"

PANEL_STYLE = f"""
QWidget#panelGround {{ background: {GROUND}; }}
QWidget#sectionBody {{ background: {PANEL}; }}
QLabel {{ color: {TEXT}; font-size: 11px; background: transparent; }}
QLabel:disabled {{ color: #5a5a5a; }}
QLabel#rowLabel {{ color: #a9a9a9; }}
QLabel#rowLabel:disabled {{ color: #565656; }}
QLabel#subHeading {{ color: {TEXT_BRIGHT}; font-size: 11px; }}
QLabel#note {{ color: {TEXT_DIM}; font-size: 10px; }}
QLabel#warning {{ color: #d9a441; font-size: 11px; }}
QLabel#fieldLabel {{ color: #a9a9a9; }}
ValueField {{
    background: transparent; color: #cfcfcf; border: 1px solid transparent; border-radius: 2px;
    font-size: 11px; padding: 0 2px; selection-background-color: #5a5a5a;
}}
ValueField:hover {{ border-color: #3a3a3a; }}
ValueField:focus {{ background: #161616; border-color: #6a6a6a; }}
ValueField:disabled {{ color: #565656; }}
QCheckBox {{ color: {TEXT}; font-size: 11px; spacing: 7px; background: transparent; }}
QCheckBox:disabled {{ color: #5a5a5a; }}
QCheckBox::indicator {{ width: 12px; height: 12px; }}
QPushButton {{
    background: #383838; color: #cfcfcf; border: 1px solid #191919; border-radius: 3px;
    padding: 3px 10px; font-size: 11px;
}}
QPushButton:hover {{ background: #434343; }}
QPushButton:pressed {{ background: #2c2c2c; }}
QPushButton:disabled {{ color: #5c5c5c; background: #2e2e2e; }}
QComboBox {{
    background: #1a1a1a; color: #cfcfcf; border: 1px solid #3a3a3a; border-radius: 2px;
    padding: 2px 6px; font-size: 11px; min-height: 16px;
}}
QComboBox:hover {{ border-color: #585858; }}
QComboBox:disabled {{ color: #5c5c5c; }}
QComboBox QAbstractItemView {{
    background: #2a2a2a; color: #cfcfcf; border: 1px solid #111111;
    selection-background-color: #4a4a4a; outline: none;
}}
QToolButton[segment="true"] {{
    color: #7e7e7e; border: none; background: transparent; padding: 2px 5px; font-size: 11px;
}}
QToolButton[segment="true"]:hover {{ color: #c8c8c8; }}
QToolButton[segment="true"]:checked {{ color: #f0f0f0; }}
QToolButton[segment="true"]:disabled {{ color: #4c4c4c; }}
QToolButton#iconButton {{
    border: 1px solid transparent; border-radius: 3px; padding: 2px; background: transparent;
    color: {TEXT}; font-size: 11px;
}}
QToolButton#iconButton:hover {{ background: #3a3a3a; border-color: #444444; }}
QToolButton#iconButton:pressed, QToolButton#iconButton:checked {{ background: #454545; }}
QToolButton#iconButton:disabled {{ color: #5a5a5a; }}
ColorSwatch {{ border: 1px solid #3c3c3c; border-radius: 2px; background: #1a1a1a; }}
QToolButton#uprightButton {{
    background: #333333; color: #c4c4c4; border: 1px solid #1a1a1a; padding: 3px 0;
    font-size: 11px;
}}
QToolButton#uprightButton:hover {{ background: #404040; color: #ececec; }}
QToolButton#uprightButton:pressed {{ background: #2a2a2a; }}
QToolButton#uprightButton:checked {{ background: #5e5e5e; color: #f4f4f4; }}
QToolButton#uprightButton:disabled {{ color: #5c5c5c; }}
QToolButton#textButton {{
    background: #333333; color: #c4c4c4; border: 1px solid #1a1a1a; border-radius: 2px;
    padding: 1px 9px; font-size: 11px;
}}
QToolButton#textButton:hover {{ background: #404040; color: #ececec; }}
QToolButton#textButton:pressed {{ background: #2a2a2a; }}
QToolButton#textButton:disabled {{ color: #5c5c5c; }}
QScrollBar:vertical {{ background: {GROUND}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #3e3e3e; border-radius: 3px; min-height: 30px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: #505050; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
"""


class PowerSwitch(QAbstractButton):
    """The small on/off switch at the left of a Lightroom panel header."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setFixedSize(22, 16)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(_("Activar o desactivar los ajustes de este panel"))

    def sizeHint(self) -> QSize:
        return QSize(22, 16)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(2.5, 4.5, 17, 8)
        on = self.isChecked()
        p.setPen(QPen(QColor("#8c8c8c" if on else "#575757"), 1.0))
        p.setBrush(QColor("#6e6e6e" if on else "#202020"))
        p.drawRoundedRect(track, 4, 4)
        knob_x = track.right() - 4 if on else track.left() + 4
        p.setPen(QPen(QColor(0, 0, 0, 150), 1.0))
        p.setBrush(QColor("#e8e8e8" if on else "#6a6a6a"))
        p.drawEllipse(QPointF(knob_x, track.center().y()), 4.6, 4.6)


class SectionHeader(QWidget):
    """Switch · title (right-aligned, as in Lightroom) · disclosure triangle."""

    clicked = Signal()
    doubleClicked = Signal()
    menuRequested = Signal(QPoint)

    HEIGHT = 30

    def __init__(self, title: str, switchable: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.expanded = True
        self.setFixedHeight(self.HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 24, 0)
        layout.setSpacing(6)
        self.switch = PowerSwitch() if switchable else None
        if self.switch is not None:
            layout.addWidget(self.switch)
        layout.addStretch(1)
        self.title = QLabel(title)
        self.title.setStyleSheet(f"color: {TEXT_BRIGHT}; font-size: 13px; background: transparent;")
        self.title.setToolTip(_("Clic: plegar o desplegar · Doble clic: restablecer"))
        layout.addWidget(self.title)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        # Only the title resets: two quick clicks elsewhere (the triangle, say)
        # just fold and unfold the panel.
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.title.geometry().contains(event.position().toPoint()):
            self.doubleClicked.emit()
        else:
            self.clicked.emit()

    def contextMenuEvent(self, event) -> None:
        self.menuRequested.emit(event.globalPos())

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(HEADER))
        p.setPen(QPen(QColor(255, 255, 255, 14), 1.0))
        p.drawLine(QPointF(0, 0.5), QPointF(self.width(), 0.5))
        x, y = self.width() - 13.0, self.height() / 2
        if self.expanded:
            tri = QPolygonF([QPointF(x - 4, y - 2), QPointF(x + 4, y - 2), QPointF(x, y + 3)])
        else:
            tri = QPolygonF([QPointF(x + 2.5, y - 4), QPointF(x + 2.5, y + 4), QPointF(x - 2.5, y)])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#9a9a9a"))
        p.drawPolygon(tri)


class SubHeading(QWidget):
    """A group title inside a panel ("Tono", "Presencia"); double-click resets the group."""

    doubleClicked = Signal()

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 2)
        layout.setSpacing(6)
        self.label = ClickLabel(text)
        self.label.setObjectName("subHeading")
        self.label.doubleClicked.connect(self.doubleClicked)
        self.label.shiftDoubleClicked.connect(self.doubleClicked)
        layout.addWidget(self.label)
        layout.addStretch(1)
        self._layout = layout

    def add_widget(self, widget: QWidget) -> None:
        """Put a control at the right end, like Lightroom's "Auto" beside "Tono"."""
        self._layout.addWidget(widget)


class Section(QWidget):
    """A collapsible panel: header plus a body that holds the controls."""

    toggled = Signal(bool)  # the power switch, from the user
    resetRequested = Signal()
    expandedChanged = Signal(bool)
    menuRequested = Signal(QPoint)

    def __init__(self, title: str, switchable: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.header = SectionHeader(title, switchable)
        self.body = QWidget()
        self.body.setObjectName("sectionBody")
        self.body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.content = QVBoxLayout(self.body)
        self.content.setContentsMargins(12, 4, 12, 10)
        self.content.setSpacing(2)
        outer.addWidget(self.header)
        outer.addWidget(self.body)
        # A double-click arrives after the click that already folded the panel;
        # remember the state before it so the double-click can put it back.
        self._before_click = True
        self.header.clicked.connect(self._on_click)
        self.header.doubleClicked.connect(self._on_double_click)
        self.header.menuRequested.connect(self.menuRequested)
        if self.header.switch is not None:
            self.header.switch.clicked.connect(lambda on: self.toggled.emit(bool(on)))

    def add(self, item: QWidget | QLayout) -> None:
        if isinstance(item, QLayout):
            self.content.addLayout(item)
        else:
            self.content.addWidget(item)

    def add_spacing(self, px: int) -> None:
        self.content.addSpacing(px)

    def is_expanded(self) -> bool:
        return self.header.expanded

    def set_expanded(self, on: bool) -> None:
        if on == self.header.expanded:
            return
        self.header.expanded = on
        self.body.setVisible(on)
        self.header.update()
        self.expandedChanged.emit(on)

    def is_active(self) -> bool:
        return self.header.switch is None or self.header.switch.isChecked()

    def set_active(self, on: bool) -> None:
        """Show the switch state without emitting ``toggled`` (loading a frame)."""
        if self.header.switch is not None:
            self.header.switch.setChecked(on)
        self.header.title.setStyleSheet(
            f"color: {TEXT_BRIGHT if on else TEXT_DIM}; font-size: 13px; background: transparent;")

    def _on_click(self) -> None:
        self._before_click = self.is_expanded()
        self.set_expanded(not self._before_click)

    def _on_double_click(self) -> None:
        self.set_expanded(self._before_click)
        self.resetRequested.emit()


class SectionList(QWidget):
    """The stacked sections, with Lightroom's solo mode and the panel menu."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("panelGround")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(1)
        self._sections: dict[str, Section] = {}
        self.solo = False

    def add(self, key: str, section: Section) -> Section:
        self._sections[key] = section
        self._layout.addWidget(section)
        section.expandedChanged.connect(lambda on, k=key: self._on_expanded(k, on))
        section.menuRequested.connect(lambda pos, k=key: self._menu(k, pos))
        return section

    def section(self, key: str) -> Section:
        return self._sections[key]

    def keys(self) -> list[str]:
        return list(self._sections)

    def add_stretch(self) -> None:
        self._layout.addStretch(1)

    def set_all_expanded(self, on: bool) -> None:
        for section in self._sections.values():
            section.set_expanded(on)

    def _on_expanded(self, key: str, on: bool) -> None:
        if on and self.solo:
            for other, section in self._sections.items():
                if other != key:
                    section.set_expanded(False)

    def _menu(self, key: str, pos: QPoint) -> None:
        menu = self.panel_menu(key)
        menu.exec(pos)
        menu.deleteLater()  # a new one is built on every right-click

    def panel_menu(self, key: str) -> QMenu:
        """Lightroom's right-click menu on a panel header."""
        menu = QMenu(self)
        section = self._sections[key]
        menu.addAction(_("Restablecer {panel}").format(panel=section.title), section.resetRequested.emit)
        menu.addSeparator()
        menu.addAction(_("Expandir todos los paneles"), lambda: self.set_all_expanded(True))
        menu.addAction(_("Contraer todos los paneles"), lambda: self.set_all_expanded(False))
        solo = QAction(_("Modo individual"), menu)
        solo.setCheckable(True)
        solo.setChecked(self.solo)
        solo.setToolTip(_("Al abrir un panel se cierran los demás"))
        solo.toggled.connect(self.set_solo)
        menu.addAction(solo)
        return menu

    def set_solo(self, on: bool) -> None:
        self.solo = on
        if on:
            open_keys = [k for k, s in self._sections.items() if s.is_expanded()]
            for k in open_keys[1:]:
                self._sections[k].set_expanded(False)

    def get_state(self) -> dict:
        return {"expanded": {k: s.is_expanded() for k, s in self._sections.items()}, "solo": self.solo}

    def set_state(self, state: dict) -> None:
        self.solo = bool(state.get("solo", False))
        for key, on in (state.get("expanded") or {}).items():
            if key in self._sections:
                # Without going through solo mode: the saved layout is already consistent.
                solo, self.solo = self.solo, False
                self._sections[key].set_expanded(bool(on))
                self.solo = solo
