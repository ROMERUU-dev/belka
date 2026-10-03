"""Line icons from belka/data/icons/*.svg, tinted to the current theme.

The SVGs draw with ``currentColor``; :func:`icon` substitutes one colour per
icon mode, so one file serves every state: normal, hover (a brighter tone of
the same colour, as Lightroom does), checked (accent) and disabled.

Each size is rendered from the vector rather than resampled, for the usual
UI sizes at every screen's device pixel ratio, so 14 px header buttons and
fractional HiDPI scales stay sharp. (A Python QIconEngine would render any
size on demand, but PySide cannot hand the engine's clone() result to C++,
which crashes when Qt detaches a shared icon.) A missing file gives an
empty QIcon, so callers can always fall back to their text.
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from belka import paths

ICON_DIR = paths.data_dir() / "icons"

# logical sizes the UI asks for; anything else is scaled from the nearest larger one
SIZES = (12, 14, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64)

_Mode, _State = QIcon.Mode, QIcon.State


@lru_cache(maxsize=None)
def _svg(name: str) -> str | None:
    path = ICON_DIR / f"{name}.svg"
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _renderer(svg: str, color: str) -> QSvgRenderer:
    return QSvgRenderer(QByteArray(svg.replace("currentColor", color).encode("utf-8")))


def _render(renderer: QSvgRenderer, side: int) -> QPixmap:
    pix = QPixmap(side, side)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    renderer.render(painter, QRectF(0, 0, side, side))
    painter.end()
    return pix


def _hover(color: str) -> str:
    """``color`` pushed 45 % towards white."""
    c = QColor(color)
    return QColor(*(round(v + (255 - v) * 0.45) for v in (c.red(), c.green(), c.blue()))).name()


def _device_sizes() -> list[int]:
    app = QGuiApplication.instance()
    scales = {1.0} | ({s.devicePixelRatio() for s in app.screens()} if app is not None else set())
    return sorted({round(size * scale) for size in SIZES for scale in scales})


@lru_cache(maxsize=256)
def icon(name: str, color: str = "#d8d8d8", disabled: str = "#6a6a6a", active: str = "#f0a050") -> QIcon:
    """``color`` normally, brighter on hover, ``active`` when checked, ``disabled`` greyed out."""
    svg = _svg(name)
    if svg is None:
        return QIcon()
    roles = [
        (color, [(_Mode.Normal, _State.Off)]),
        (_hover(color), [(_Mode.Active, _State.Off), (_Mode.Selected, _State.Off)]),
        (active, [(_Mode.Normal, _State.On), (_Mode.Active, _State.On), (_Mode.Selected, _State.On)]),
        (disabled, [(_Mode.Disabled, _State.Off), (_Mode.Disabled, _State.On)]),
    ]
    by_tint: dict[str, list[tuple[QIcon.Mode, QIcon.State]]] = {}
    for tint, pairs in roles:  # a colour shared by two roles is rendered once
        by_tint.setdefault(tint, []).extend(pairs)
    result = QIcon()
    sizes = _device_sizes()
    for tint, pairs in by_tint.items():
        renderer = _renderer(svg, tint)
        for side in sizes:
            pix = _render(renderer, side)
            for mode, state in pairs:
                result.addPixmap(pix, mode, state)
    return result


def pixmap(name: str, size: int, color: str = "#d8d8d8", dpr: float = 1.0) -> QPixmap:
    """One tinted pixmap for custom painting (``size`` in logical pixels); null if the icon is missing."""
    svg = _svg(name)
    if svg is None:
        return QPixmap()
    pix = _render(_renderer(svg, color), round(size * dpr))
    pix.setDevicePixelRatio(dpr)
    return pix


def available() -> list[str]:
    return sorted(p.stem for p in ICON_DIR.glob("*.svg"))
