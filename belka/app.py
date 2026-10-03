"""Application entry point."""

from __future__ import annotations

import signal
import sys

from PySide6.QtCore import QLibraryInfo, Qt, QTimer, QTranslator
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

from belka import APP_ID, APP_NAME, __version__, paths
from belka.i18n import language, set_language
from belka.settings import Settings


def dark_palette() -> QPalette:
    """Neutral dark grey: a coloured or bright UI would bias how the positive is judged."""
    p = QPalette()
    window, base, text = QColor(35, 35, 36), QColor(28, 28, 30), QColor(220, 220, 220)
    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, window)
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(50, 50, 52))
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, QColor(48, 48, 50))
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.ColorRole.Highlight, QColor(214, 120, 40))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(140, 140, 140))
    p.setColor(QPalette.ColorRole.Mid, QColor(70, 70, 72))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(110, 110, 110))
    return p


# Lightroom Classic's look: flat dark greys, thin scroll bars, quiet controls.
STYLE = """
QWidget { font-size: 12px; }
QToolTip { background: #2b2b2d; color: #e0e0e0; border: 1px solid #4a4a4c; padding: 4px 6px; }
QMenu { background: #262628; color: #d0d0d0; border: 1px solid #3a3a3c; }
QMenu::item { padding: 5px 22px 5px 22px; }
QMenu::item:selected { background: #3d3d40; }
QMenu::separator { height: 1px; background: #3a3a3c; margin: 4px 8px; }
QPushButton { background: #333335; color: #d6d6d6; border: 1px solid #424245; border-radius: 3px; padding: 4px 12px; }
QPushButton:hover { background: #3c3c3f; }
QPushButton:pressed, QPushButton:checked { background: #48484c; }
QPushButton:disabled { color: #6a6a6a; background: #2c2c2e; border-color: #353537; }
QComboBox { background: #2c2c2e; color: #d6d6d6; border: 1px solid #3e3e41; border-radius: 3px; padding: 3px 8px; }
QComboBox:hover { border-color: #55555a; }
QComboBox QAbstractItemView { background: #262628; color: #d6d6d6; selection-background-color: #3d3d40; border: 1px solid #3a3a3c; }
QLineEdit, QSpinBox, QDoubleSpinBox { background: #1e1e20; color: #e0e0e0; border: 1px solid #3a3a3c; border-radius: 3px; padding: 2px 4px; }
QCheckBox { color: #c8c8c8; }
QGroupBox { color: #9a9a9a; border: none; margin-top: 14px; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 0; }
QScrollBar::handle:vertical { background: #48484b; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #5c5c60; }
QScrollBar:horizontal { background: transparent; height: 9px; margin: 0; }
QScrollBar::handle:horizontal { background: #48484b; border-radius: 4px; min-width: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QStatusBar QLabel { color: #9a9a9a; padding: 0 8px; }
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--version" in argv:
        print(f"{APP_NAME} {__version__}")
        return 0
    settings = Settings()
    set_language(settings.get("language"))

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    # Qt's own buttons and dialogs (OK/Cancel, Yes/No, file dialogs) in the UI language.
    qt_translator = QTranslator(app)
    if qt_translator.load(f"qtbase_{language()}", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
        app.installTranslator(qt_translator)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setDesktopFileName(APP_ID)
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(STYLE)
    icon = paths.data_dir() / "icons" / "belka.svg"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    from belka.ui.main_window import MainWindow

    window = MainWindow(settings)
    window.show()
    # Logout, shutdown or `kill` must still release the camera, stop the idle
    # inhibitor and resume Night Light: route signals into a normal close, and
    # catch the paths that skip closeEvent with aboutToQuit.
    app.aboutToQuit.connect(window.shutdown)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_args: QTimer.singleShot(0, window.close))
    heartbeat = QTimer()  # lets the Python signal handlers run inside Qt's loop
    heartbeat.start(250)
    heartbeat.timeout.connect(lambda: None)
    return app.exec()
