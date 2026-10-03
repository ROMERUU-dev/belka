import json

import numpy as np

from belka.core import synth
from belka.core.pipeline import DevelopSettings
from belka.core.rawio import load_linear, save_linear_tiff
from belka.core.session import FLAT_SINGLE, SESSION_FILE, Session, flat_role, load_frame, primary_file


def test_create_save_load(tmp_path):
    s = Session.create(tmp_path, "Portra / Oaxaca", "kodak-portra-400")
    assert s.path.name.endswith("Portra _ Oaxaca")
    assert (s.path / SESSION_FILE).exists()
    for d in (s.raw_dir, s.flat_dir, s.export_dir):
        assert d.is_dir()
    again = Session.load(s.path / SESSION_FILE)
    assert again.name == "Portra / Oaxaca"
    assert again.roll_defaults().profile_id == "kodak-portra-400"
    # Same name, same day: a new folder, not an overwrite.
    assert Session.create(tmp_path, "Portra / Oaxaca").path != s.path


def test_frames_settings_and_lock_base(tmp_path):
    s = Session.create(tmp_path, "r")
    raw = s.raw_dir / "a.tif"
    save_linear_tiff(raw, np.full((20, 30, 3), 0.5, np.float32))
    f1 = s.add_frame([raw])
    f2 = s.add_frame([raw])
    assert (f1.id, f2.id) == ("001", "002")
    assert f1.files == ["raw/a.tif"]
    s.set_frame_settings(f1, DevelopSettings(exposure=1.0, base=(0.9, 0.5, 0.3), rotation=90))
    assert s.settings_for(f1).exposure == 1.0
    assert s.settings_for(f2).exposure == 0.0
    s.roll_settings = DevelopSettings(base=(0.8, 0.6, 0.2)).to_json()
    s.lock_base = True
    assert s.settings_for(f1).base == (0.8, 0.6, 0.2)
    s.apply_to_all(DevelopSettings(profile_id="kodak-gold-200", contrast=1.3))
    assert s.settings_for(f1).rotation == 90, "geometry is kept per frame"
    assert s.settings_for(f2).contrast == 1.3
    assert s.film_profile == "kodak-gold-200"
    data = json.loads((s.path / SESSION_FILE).read_text())
    assert len(data["frames"]) == 2
    s.remove_frame(f2, delete_files=False)
    assert [f.id for f in Session.load(s.path).frames] == ["001"]
    assert s.next_id() == "002"


def test_import_copies_into_raw(tmp_path):
    src = tmp_path / "scan.tif"
    save_linear_tiff(src, np.full((10, 10, 3), 0.2, np.float32))
    s = Session.create(tmp_path / "rolls", "r")
    frames, errors = s.import_files([src])
    assert errors == []
    copied = s.resolve(frames[0].files[0])
    assert copied.parent == s.raw_dir and copied.exists()


def test_import_never_overwrites_a_kept_file(tmp_path):
    """Regression: removing the last frame but keeping its file, then importing,
    reused the id and overwrote the kept raw."""
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    save_linear_tiff(a, np.full((10, 10, 3), 0.2, np.float32))
    save_linear_tiff(b, np.full((10, 10, 3), 0.7, np.float32))
    s = Session.create(tmp_path / "rolls", "Keep")
    (fa,), _ = s.import_files([a])
    kept = s.resolve(fa.files[0])
    before = kept.read_bytes()
    s.remove_frame(fa, delete_files=False)
    (fb,), _ = s.import_files([b])
    assert kept.read_bytes() == before
    assert s.resolve(fb.files[0]) != kept


def test_partial_import_reports_errors_and_keeps_good_frames(tmp_path):
    good = tmp_path / "good.tif"
    save_linear_tiff(good, np.full((10, 10, 3), 0.2, np.float32))
    s = Session.create(tmp_path / "rolls", "r")
    frames, errors = s.import_files([good, tmp_path / "missing.nef"])
    assert len(frames) == 1 and len(errors) == 1 and "missing.nef" in errors[0]
    assert not any(p.name.endswith(".nef") for p in s.raw_dir.iterdir())


def test_flats_are_applied_per_light(tmp_path):
    s = Session.create(tmp_path, "r")
    yy, xx = np.mgrid[-1:1:60j, -1:1:90j]
    light = np.repeat((1 - 0.3 * (xx**2 + yy**2) / 2)[..., None], 3, -1).astype(np.float32) * 0.6
    s.store_flat(light, FLAT_SINGLE)  # shot with white light
    raw = s.raw_dir / "f.tif"
    save_linear_tiff(raw, light * 0.5)
    # Regression: the frame was shot after a tint calibration, with another
    # light colour; the flat (shape only) must still apply.
    frame = s.add_frame([raw], lights=[[0.61, 0.66, 1.0]])
    image = load_frame(s, frame)
    inner = image.rgb[5:-5, 5:-5, 1]
    assert inner.std() / inner.mean() < 0.03
    assert image.meta["flat_applied"]
    s.clear_flats()
    assert s.flats == {} and not any(s.flat_dir.glob("*.npy"))
    assert load_frame(s, frame).meta["flat_applied"] is False


def test_flats_from_older_rolls_keyed_by_colour_still_apply(tmp_path):
    s = Session.create(tmp_path, "r")
    flat = np.ones((8, 8, 3), np.float32) * 0.5
    np.save(s.flat_dir / "flat_156-167-255.npy", flat)
    np.save(s.flat_dir / "flat_000-167-000.npy", flat)
    s.flats = {"156-167-255": "flat/flat_156-167-255.npy", "000-167-000": "flat/flat_000-167-000.npy"}
    assert s.flat_path(FLAT_SINGLE).name == "flat_156-167-255.npy"
    assert s.flat_path(flat_role("rgb", 1)).name == "flat_000-167-000.npy"
    assert s.flat_path(flat_role("rgb", 0)) is None
    s.store_flat(np.ones((20, 20, 3), np.float32), FLAT_SINGLE)
    assert "156-167-255" not in s.flats and FLAT_SINGLE in s.flats


def test_rgb_sequential_frame_takes_one_channel_from_each_capture(tmp_path):
    s = Session.create(tmp_path, "r")
    files = []
    for i in range(3):
        img = np.zeros((16, 16, 3), np.float32)
        img[..., i] = 0.1 * (i + 1)
        img[..., (i + 1) % 3] = 0.9  # crosstalk that must be ignored
        path = s.raw_dir / f"x_{'RGB'[i]}.tif"
        save_linear_tiff(path, img)
        files.append(path)
    frame = s.add_frame(files, mode="rgb", lights=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    image = load_frame(s, frame)
    assert image.meta["rgb_sequential"]
    assert np.allclose(image.rgb.reshape(-1, 3).mean(0), [0.1, 0.2, 0.3], atol=1e-3)


def test_primary_file_prefers_raw():
    assert primary_file(["raw/a.jpg", "raw/a.nef"]) == "raw/a.nef"
    assert primary_file(["raw/a.tif"]) == "raw/a.tif"


def test_linear_tiff_round_trip(tmp_path, library):
    raw, _ = synth.synthetic_negative(library.get("generic-c41"))
    path = save_linear_tiff(tmp_path / "x.tif", raw, camera_matrix=np.eye(3), meta={"k": 1})
    back = load_linear(path)
    assert back.meta["k"] == 1
    assert np.allclose(back.camera_matrix, np.eye(3))
    assert np.abs(back.rgb - raw).max() < 1 / 30000

