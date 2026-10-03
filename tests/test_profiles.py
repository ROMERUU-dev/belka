import json

import pytest

from belka.core.film import FILM_TYPES, FilmProfile, ProfileLibrary, slugify


def test_builtin_profiles_load_and_are_unique(library):
    profiles = library.all()
    ids = [p.id for p in profiles]
    assert len(ids) == len(set(ids))
    assert len(profiles) >= 40
    assert {p.type for p in profiles} == set(FILM_TYPES)
    for generic in ("generic-c41", "generic-bw", "generic-slide", "generic-ecn2"):
        assert generic in library


def test_profiles_are_physically_plausible(library):
    for p in library.all():
        assert all(0.3 <= g <= 2.5 for g in p.gamma), p.id
        if p.type == "color_negative":
            # The orange mask absorbs blue most and red least.
            assert p.base_density[0] < p.base_density[1] < p.base_density[2], p.id
        if p.is_bw:
            assert p.separation == 0.0, p.id


def test_unknown_profile_falls_back_to_generic(library):
    assert library.get("no-such-film").id == "generic-c41"
    assert library.get(None).id == "generic-c41"


def test_user_profile_round_trip_and_shadowing(tmp_path, library):
    lib = ProfileLibrary(user_dir=tmp_path)
    mine = FilmProfile(id="user-portra", name="Portra (D780)", gamma=(0.55, 0.6, 0.7), builtin=False)
    path = lib.save_user_profile(mine)
    assert json.loads(path.read_text())["gamma"] == [0.55, 0.6, 0.7]
    again = ProfileLibrary(user_dir=tmp_path)
    loaded = again.get("user-portra")
    assert loaded.gamma == (0.55, 0.6, 0.7)
    assert not loaded.builtin
    assert again.all()[0].type == "color_negative"
    again.delete_user_profile("user-portra")
    assert "user-portra" not in again


def test_broken_user_profile_is_skipped(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "invalid.json").write_text(json.dumps({"id": "x", "name": "x", "type": "nope"}))
    lib = ProfileLibrary(user_dir=tmp_path)
    assert "x" not in lib
    assert "generic-c41" in lib


def test_invalid_gamma_rejected():
    with pytest.raises(ValueError):
        FilmProfile.from_json({"id": "a", "name": "a", "gamma": [0.6, 0, 0.6]}, builtin=False)


def test_slugify():
    assert slugify("Portra 400 (mi D780)") == "portra-400-mi-d780"
    assert slugify("¡¿") == "perfil"
