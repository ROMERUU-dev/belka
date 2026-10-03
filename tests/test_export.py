import numpy as np
import tifffile

from belka.core import synth
from belka.core.export import ExportOptions, export_frame
from belka.core.pipeline import DevelopSettings
from belka.core.rawio import save_linear_tiff
from belka.core.session import Session


def test_export_tiff16_and_jpeg(tmp_path, library):
    from PySide6.QtGui import QGuiApplication, QImage

    app = QGuiApplication.instance() or QGuiApplication([])
    s = Session.create(tmp_path, "export", "kodak-portra-400")
    raw, _ = synth.synthetic_negative(library.get("kodak-portra-400"))
    path = save_linear_tiff(s.raw_dir / "f.tif", raw)
    frame = s.add_frame([path])
    s.set_frame_settings(frame, DevelopSettings(profile_id="kodak-portra-400", rotation=90, crop=(0.1, 0.1, 0.9, 0.9)))
    written = export_frame(s, frame, library, ExportOptions())
    assert [p.suffix for p in written] == [".tif", ".jpg"]
    with tifffile.TiffFile(written[0]) as tif:
        page = tif.pages[0]
        data = page.asarray()
        assert data.dtype == np.uint16 and data.shape[2] == 3
        assert 34675 in page.tags  # ICC profile
        assert "kodak-portra-400" in page.description
    # Rotated 90 degrees: portrait.
    assert data.shape[0] > data.shape[1]
    jpg = QImage(str(written[1]))
    assert (jpg.width(), jpg.height()) == (data.shape[1], data.shape[0])
    assert Session.load(s.path).frames[0].exported
    assert app is not None


def test_export_with_accented_profile_name_and_outside_folder(tmp_path, library):
    """Regression: 'Genérico C-41' (the default film) made tifffile raise and left an 8-byte TIFF."""
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance() or QGuiApplication([])
    shared = tmp_path / "shared"
    paths = []
    for _ in range(2):  # two rolls with the same name exporting to one folder
        s = Session.create(tmp_path / "rolls", "Portra 400", "generic-c41")
        raw, _r = synth.synthetic_negative(library.get("generic-c41"))
        frame = s.add_frame([save_linear_tiff(s.raw_dir / "f.tif", raw)])
        paths += export_frame(s, frame, library, ExportOptions(folder=shared))
    assert len({p.name for p in paths}) == 4
    for p in paths:
        assert p.stat().st_size > 10_000
    with tifffile.TiffFile(paths[0]) as tif:
        assert "Gen\\u00e9rico" in tif.pages[0].description
    assert not list(shared.glob(".*part*"))
    assert app is not None


def test_export_size_is_the_size_of_the_cropped_picture(tmp_path, library):
    """Regression: '2048 px' shrank the whole capture before cropping, so a cropped
    frame came out at a fraction of the size asked for."""
    from belka.core.export import develop_full

    s = Session.create(tmp_path, "size", "kodak-portra-400")
    raw, _ = synth.synthetic_negative(library.get("kodak-portra-400"))
    frame = s.add_frame([save_linear_tiff(s.raw_dir / "f.tif", raw)])
    s.set_frame_settings(frame, DevelopSettings(profile_id="kodak-portra-400", crop=(0.2, 0.2, 0.8, 0.7)))
    full, _settings = develop_full(s, frame, library)
    side = max(full.shape[:2]) // 2
    small, _settings = develop_full(s, frame, library, max_side=side)
    assert max(small.shape[:2]) == side
    assert abs(small.shape[1] / small.shape[0] - full.shape[1] / full.shape[0]) < 0.02
