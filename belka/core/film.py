"""Film stock profiles.

A profile describes how a film stock records light, in the terms the
inversion pipeline needs:

* ``gamma``: slope of each layer's characteristic curve (density per log10
  exposure) for R, G and B. Only the ratios matter for colour balance; the
  green value sets the overall contrast of the reconstruction.
* ``base_density``: approximate Status M density of the unexposed base (the
  orange mask on colour negatives). It is only a hint: the real base is always
  measured from the frame or the rebate.
* ``separation``: how strongly the density-domain unmixing matrix is applied
  to undo dye crosstalk (0 = off, 1 = the full camera matrix).
* ``paper_contrast`` / ``saturation``: defaults of the print stage.

The built-in values are read from published characteristic curves and are
approximate. They are a starting point: a user profile saved from a real roll
(``Guardar como perfil``) always wins.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from belka import paths

FILM_TYPES = ("color_negative", "bw_negative", "slide")


@dataclass
class FilmProfile:
    id: str
    name: str
    brand: str = ""
    type: str = "color_negative"
    process: str = "C-41"
    iso: int | None = None
    gamma: tuple[float, float, float] = (0.60, 0.62, 0.66)
    base_density: tuple[float, float, float] = (0.22, 0.62, 0.88)
    separation: float = 0.35
    paper_contrast: float = 1.0
    saturation: float = 1.0
    # B&W only: weights used to collapse the camera RGB to one channel.
    bw_mix: tuple[float, float, float] = (0.25, 0.5, 0.25)
    # Daylight film shot under tungsten (CineStill 800T, Vision3 500T...) or
    # the opposite: default temperature shift in the print stage.
    temperature: float = 0.0
    notes: str = ""
    builtin: bool = field(default=True, compare=False)

    @property
    def is_negative(self) -> bool:
        return self.type != "slide"

    @property
    def is_bw(self) -> bool:
        return self.type == "bw_negative"

    def relative_gamma(self) -> tuple[float, float, float]:
        g = self.gamma[1] or 1.0
        return (self.gamma[0] / g, 1.0, self.gamma[2] / g)

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("builtin", None)
        return data

    @classmethod
    def from_json(cls, data: dict, builtin: bool) -> "FilmProfile":
        known = {f for f in cls.__dataclass_fields__ if f != "builtin"}
        clean = {k: v for k, v in data.items() if k in known}
        for key in ("gamma", "base_density", "bw_mix"):
            if key in clean:
                clean[key] = tuple(float(v) for v in clean[key])
        profile = cls(**clean, builtin=builtin)
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.type not in FILM_TYPES:
            raise ValueError(f"{self.id}: tipo desconocido {self.type!r}")
        if len(self.gamma) != 3 or min(self.gamma) <= 0:
            raise ValueError(f"{self.id}: gamma debe tener 3 valores positivos")
        if len(self.base_density) != 3:
            raise ValueError(f"{self.id}: base_density debe tener 3 valores")
        if not 0.0 <= self.separation <= 1.5:
            raise ValueError(f"{self.id}: separation fuera de rango")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "perfil"


def film_name(profile: FilmProfile) -> str:
    """'Kodak Portra 400', without repeating a brand the name already carries ('CineStill 800T')."""
    if not profile.brand or profile.name.startswith(profile.brand):
        return profile.name
    return f"{profile.brand} {profile.name}"


class ProfileLibrary:
    """Built-in profiles plus the user's own, user ones shadowing built-ins."""

    def __init__(self, builtin_dir: Path | None = None, user_dir: Path | None = None):
        self.builtin_dir = builtin_dir or paths.data_dir() / "profiles"
        self.user_dir = user_dir or paths.user_profiles_dir()
        self._profiles: dict[str, FilmProfile] = {}
        self.reload()

    def reload(self) -> None:
        self._profiles.clear()
        for path in sorted(self.builtin_dir.glob("*.json")):
            for entry in _read_entries(path):
                profile = FilmProfile.from_json(entry, builtin=True)
                self._profiles[profile.id] = profile
        if self.user_dir.is_dir():
            for path in sorted(self.user_dir.glob("*.json")):
                try:
                    for entry in _read_entries(path):
                        profile = FilmProfile.from_json(entry, builtin=False)
                        self._profiles[profile.id] = profile
                except (ValueError, TypeError, json.JSONDecodeError):
                    # A hand-edited profile must not keep the app from starting.
                    continue

    def get(self, profile_id: str | None) -> FilmProfile:
        if profile_id and profile_id in self._profiles:
            return self._profiles[profile_id]
        return self._profiles.get("generic-c41") or next(iter(self._profiles.values()))

    def all(self) -> list[FilmProfile]:
        order = {t: i for i, t in enumerate(FILM_TYPES)}
        return sorted(
            self._profiles.values(),
            key=lambda p: (order.get(p.type, 9), not p.id.startswith("generic"), p.brand, p.name),
        )

    def __contains__(self, profile_id: str) -> bool:
        return profile_id in self._profiles

    def save_user_profile(self, profile: FilmProfile) -> Path:
        self.user_dir.mkdir(parents=True, exist_ok=True)
        profile.builtin = False
        path = self.user_dir / f"{profile.id}.json"
        path.write_text(json.dumps(profile.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._profiles[profile.id] = profile
        return path

    def delete_user_profile(self, profile_id: str) -> None:
        path = self.user_dir / f"{profile_id}.json"
        if path.exists():
            path.unlink()
        self.reload()


def _read_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "profiles" in data:
        return list(data["profiles"])
    if isinstance(data, list):
        return data
    return [data]
