"""Vertical icon toolbar beside the image, and the Lightroom-style top bar."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolBar, QToolButton, QWidget

from belka.i18n import _

BAR_STYLE = """
QToolBar { background: #232324; border: none; spacing: 2px; padding: 4px 2px; }
QToolBar::separator { background: #3a3a3c; height: 1px; margin: 4px 6px; }
QToolButton { border: none; border-radius: 4px; padding: 4px; color: #c8c8c8; }
QToolButton:hover { background: #38383a; }
QToolButton:checked { background: #4a3a2a; }
QToolButton:disabled { color: #5a5a5a; }
"""


class ToolStrip(QToolBar):
    """Tools with their icon and a tooltip naming the shortcut, like Lightroom's."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setOrientation(Qt.Orientation.Vertical)
        self.setMovable(False)
        self.setFloatable(False)
        self.setIconSize(QSize(20, 20))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setStyleSheet(BAR_STYLE)

    def add(self, action: QAction) -> None:
        keys = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        if keys:
            # The shortcut goes on the first line, after the tool's name.
            name, _sep, rest = (action.toolTip() or action.text().replace("&", "")).partition("\n")
            action.setToolTip(f"{name}  ({keys})" + (f"\n{rest}" if rest else ""))
        self.addAction(action)


class ModuleBar(QWidget):
    """Identity plate on the left, quick actions, and the module switcher on the right."""

    moduleChanged = Signal(str)

    def __init__(self, modules: list[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("moduleBar")
        self.setFixedHeight(44)
        self.setStyleSheet("""
            #moduleBar { background: #161617; border-bottom: 1px solid #2c2c2e; }
            #identity { color: #d8d8d8; font-size: 17px; font-weight: 300; letter-spacing: 1px; }
            #identitySub { color: #7a7a7a; font-size: 11px; }
            QPushButton[module="true"] { border: none; color: #7c7c7c; font-size: 15px; padding: 4px 10px; background: transparent; }
            QPushButton[module="true"]:hover { color: #c8c8c8; }
            QPushButton[module="true"]:checked { color: #f0f0f0; }
            QToolButton { border: none; border-radius: 4px; padding: 4px; color: #c8c8c8; }
            QToolButton:hover { background: #2e2e30; }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(6)
        identity = QLabel("Belka")
        identity.setObjectName("identity")
        self.subtitle = QLabel("")
        self.subtitle.setObjectName("identitySub")
        layout.addWidget(identity)
        layout.addSpacing(8)
        layout.addWidget(self.subtitle)
        layout.addSpacing(18)
        self._quick = QHBoxLayout()
        self._quick.setSpacing(2)
        layout.addLayout(self._quick)
        layout.addStretch(1)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for i, (key, title) in enumerate(modules):
            if i:
                sep = QLabel("|")
                sep.setStyleSheet("color: #4a4a4c; font-size: 15px;")
                layout.addWidget(sep)
            button = QPushButton(title)
            button.setProperty("module", True)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _c=False, k=key: self.moduleChanged.emit(k))
            self.group.addButton(button)
            self.buttons[key] = button
            layout.addWidget(button)

    def add_quick(self, action: QAction) -> QToolButton:
        button = QToolButton()
        button.setDefaultAction(action)
        button.setIconSize(QSize(20, 20))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        if action.shortcut().toString():
            button.setToolTip(f"{action.text().replace('&', '')}  ({action.shortcut().toString()})")
        self._quick.addWidget(button)
        return button

    def set_module(self, key: str) -> None:
        button = self.buttons.get(key)
        if button is not None:
            button.setChecked(True)

    def set_subtitle(self, text: str) -> None:
        self.subtitle.setText(text)
