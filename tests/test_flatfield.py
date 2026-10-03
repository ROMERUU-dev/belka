import numpy as np

from belka.core import flatfield


def _vignetted(h=400, w=600, strength=0.4):
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    fall = 1 - strength * (xx**2 + yy**2) / 2
    return np.repeat(fall[..., None], 3, axis=-1).astype(np.float32) * np.array([0.5, 0.8, 0.6], dtype=np.float32)


def test_flat_removes_vignetting():
    light = _vignetted()
    flat = flatfield.build_flat(light)
    assert flat.max() <= 1.05
    corrected = flatfield.apply_flat(light * 0.5, flat)
    for c in range(3):
        ch = corrected[20:-20, 20:-20, c]
        assert ch.std() / ch.mean() < 0.02


def test_flat_warnings():
    assert flatfield.flat_warnings(_vignetted()) == []
    clipped = np.ones((200, 300, 3), dtype=np.float32)
    assert any("saturado" in w for w in flatfield.flat_warnings(clipped))
    assert any("desigual" in w for w in flatfield.flat_warnings(_vignetted(strength=1.4)))


def test_resize_bilinear_shapes(tmp_path):
    small = np.random.default_rng(0).random((10, 15, 3)).astype(np.float32)
    big = flatfield.resize_bilinear(small, 97, 131)
    assert big.shape == (97, 131, 3)
    assert big.min() >= small.min() - 1e-6 and big.max() <= small.max() + 1e-6
    path = flatfield.save_flat(tmp_path / "f.npy", small)
    assert np.array_equal(flatfield.load_flat(path), small)
    assert flatfield.load_flat(tmp_path / "missing.npy") is None
