import re

import numpy as np
import pytest
from PySide6.QtCore import QByteArray, QRectF, QSize
from PySide6.QtGui import QIcon, QImage
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from belka.ui import icons

# docs/arquitectura-v0.2.md, section "belka/ui/icons.py"
CONTRACT = """
crop straighten rotate_left rotate_right flip_horizontal flip_vertical perspective eyedropper_base eyedropper_wb
compare compare_split zoom_fit zoom_100 negative clipping grid_overlay undo redo reset eye eye_off camera shutter
light flat_field tint liveview focus_near focus_far autofocus import export new_roll open_roll library develop
capture star star_filled flag reject copy_settings paste_settings sync snapshot history histogram settings help
lock unlock chevron_down chevron_right plus minus close check upright_auto upright_level upright_vertical
upright_full aspect_lock info
""".split()


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _svg(name: str) -> str:
    return (icons.ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")


def _alpha(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return rows[:, : image.width() * 4].reshape(image.height(), image.width(), 4)[..., 3] / 255.0


def _rgb_of_ink(image: QImage) -> str:
    """Colour of the most opaque pixel: anti-aliased edges are premultiplied towards transparent."""
    alpha = _alpha(image)
    y, x = np.unravel_index(np.argmax(alpha), alpha.shape)
    return image.pixelColor(int(x), int(y)).name()


def test_contract_has_no_duplicates():
    assert len(CONTRACT) == len(set(CONTRACT)) == 63


@pytest.mark.parametrize("name", CONTRACT)
def test_every_contract_icon_is_a_valid_24px_line_svg(name):
    svg = _svg(name)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    assert renderer.isValid()
    assert renderer.viewBoxF() == QRectF(0, 0, 24, 24)
    root = svg[: svg.index(">")]
    for attribute in ('stroke="currentColor"', 'fill="none"', 'stroke-width="1.5"',
                      'stroke-linecap="round"', 'stroke-linejoin="round"'):
        assert attribute in root
    # one stroke weight for the whole set: nothing overrides or rescales the root's 1.5 px
    assert svg.count("stroke-width") == 1 and "transform" not in svg
    # tinting only works through currentColor: no hard-coded colours anywhere
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", svg)
    assert set(re.findall(r'\b(?:fill|stroke)="([^"]+)"', svg)) <= {"currentColor", "none"}


@pytest.mark.parametrize("name", CONTRACT)
def test_icon_renders_inside_the_2px_padding(name):
    ic = icons.icon(name)
    assert not ic.isNull()
    alpha = _alpha(ic.pixmap(QSize(48, 48)).toImage())
    assert alpha.max() > 0.9, "nothing drawn"
    border = alpha.copy()
    border[4:44, 4:44] = 0  # 2 px of the 24 px grid is 4 px at 48 px
    assert border.max() < 0.25


def test_available_lists_every_contract_icon():
    assert set(CONTRACT) <= set(icons.available())


def test_missing_icon_is_null_so_callers_fall_back_to_text():
    assert icons.icon("no_such_icon").isNull()
    assert icons.pixmap("no_such_icon", 16).isNull()


def test_modes_and_states_use_their_own_colours():
    ic = icons.icon("crop", color="#808080", disabled="#404040", active="#f0a050")
    size = QSize(20, 20)
    assert _rgb_of_ink(ic.pixmap(size, QIcon.Mode.Normal, QIcon.State.Off).toImage()) == "#808080"
    assert _rgb_of_ink(ic.pixmap(size, QIcon.Mode.Disabled, QIcon.State.Off).toImage()) == "#404040"
    assert _rgb_of_ink(ic.pixmap(size, QIcon.Mode.Normal, QIcon.State.On).toImage()) == "#f0a050"
    hover = _rgb_of_ink(ic.pixmap(size, QIcon.Mode.Active, QIcon.State.Off).toImage())
    assert hover == "#b9b9b9"  # 45 % of the way to white


def test_one_colour_for_two_roles_keeps_both():
    ic = icons.icon("flag", color="#ffffff", active="#ffffff")
    for mode, state in ((QIcon.Mode.Normal, QIcon.State.Off), (QIcon.Mode.Normal, QIcon.State.On)):
        assert _rgb_of_ink(ic.pixmap(QSize(20, 20), mode, state).toImage()) == "#ffffff"


@pytest.mark.parametrize("logical, dpr", [(14, 1.0), (16, 1.0), (20, 1.0), (20, 2.0), (16, 1.5)])
def test_sizes_are_rendered_not_resampled(logical, dpr):
    pix = icons.icon("grid_overlay").pixmap(QSize(logical, logical), dpr)
    assert pix.size() == QSize(round(logical * dpr), round(logical * dpr))
    assert pix.devicePixelRatio() == dpr
    direct = icons.pixmap("grid_overlay", logical, "#d8d8d8", dpr)
    assert np.array_equal(_alpha(pix.toImage()), _alpha(direct.toImage()))


def test_icon_survives_detach_on_a_shared_copy():
    shared = QIcon(icons.icon("star"))
    shared.addPixmap(icons.pixmap("star", 16))
    assert not shared.pixmap(QSize(20, 20)).isNull()


def test_pixmap_honours_device_pixel_ratio():
    pix = icons.pixmap("check", 20, "#ffffff", dpr=2.0)
    assert pix.size() == QSize(40, 40)
    assert pix.devicePixelRatio() == 2.0
