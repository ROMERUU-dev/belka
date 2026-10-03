"""A roll of film being digitised: a folder with captures and a JSON index.

Layout::

    <rollo>/
        belka-rollo.json   index: film, roll-wide settings, frames
        raw/                 camera files as captured (never modified)
        flat/                flat-field maps (.npy), one per light colour
        positivos/           exported positives
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from belka.core import flatfield
from belka.core.pipeline import DevelopSettings
from belka.core.rawio import RAW_EXTENSIONS, LinearImage, load_linear

SESSION_FILE = "belka-rollo.json"
SESSION_VERSION = 1


def light_signature(rgb: tuple[float, float, float] | list[float]) -> str:
    return "-".join(f"{int(round(float(v) * 255)):03d}" for v in rgb)


FLAT_SINGLE = "single"


def flat_role(mode: str, part: int = 0) -> str:
    """Which flat a capture needs: one for white/tinted light, one per primary in RGB mode.

    Flats keep only the shape of the illumination (normalised per channel), so
    brightness or a recalibrated tint must not make them stop matching.
    """
    return f"rgb-{'RGB'[part]}" if mode == "rgb" else FLAT_SINGLE


def _legacy_role(key: str) -> str | None:
    """Role of a flat stored by older versions under its exact light colour."""
    try:
        values = [int(v) for v in key.split("-")]
    except ValueError:
        return None
    if len(values) != 3:
        return None
    lit = [i for i, v in enumerate(values) if v > 0]
    return f"rgb-{'RGB'[lit[0]]}" if len(lit) == 1 else FLAT_SINGLE


@dataclass
class Frame:
    id: str
    files: list[str]
    mode: str = "single"  # "single" or "rgb" (three captures: red, green, blue light)
    lights: list[list[float]] = field(default_factory=list)
    captured: str = ""
    settings: dict | None = None
    rating: int = 0  # 0..5 stars
    flag: int = 0  # 1 picked, -1 rejected, 0 none
    notes: str = ""
    exported: list[str] = field(default_factory=list)
    # Named develop states, like Lightroom snapshots: {"name", "created", "settings"}
    snapshots: list[dict] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict) -> "Frame":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def label(self) -> str:
        return f"#{self.id}" + (" RGB" if self.mode == "rgb" else "")


@dataclass
class Session:
    path: Path
    name: str
    film_profile: str = "generic-c41"
    roll_settings: dict = field(default_factory=dict)
    lock_base: bool = False
    flats: dict[str, str] = field(default_factory=dict)
    frames: list[Frame] = field(default_factory=list)
    camera: str = ""
    created: str = ""
    notes: str = ""

    # ------------------------------------------------------------ files
    @property
    def raw_dir(self) -> Path:
        return self.path / "raw"

    @property
    def flat_dir(self) -> Path:
        return self.path / "flat"

    @property
    def export_dir(self) -> Path:
        return self.path / "positivos"

    @property
    def index_file(self) -> Path:
        return self.path / SESSION_FILE

    @classmethod
    def create(cls, root: Path, name: str, film_profile: str = "generic-c41") -> "Session":
        safe = re.sub(r"[^\w\-. ]+", "_", name).strip() or "rollo"
        stamp = datetime.now().strftime("%Y-%m-%d")
        path = root / f"{stamp}_{safe}"
        n = 2
        while path.exists():
            path = root / f"{stamp}_{safe}-{n}"
            n += 1
        session = cls(
            path=path,
            name=name,
            film_profile=film_profile,
            roll_settings=DevelopSettings(profile_id=film_profile).to_json(),
            created=datetime.now().isoformat(timespec="seconds"),
        )
        for folder in (session.raw_dir, session.flat_dir, session.export_dir):
            folder.mkdir(parents=True, exist_ok=True)
        session.save()
        return session

    @classmethod
    def load(cls, path: Path) -> "Session":
        path = Path(path)
        if path.is_file():
            path = path.parent
        data = json.loads((path / SESSION_FILE).read_text(encoding="utf-8"))
        frames = [Frame.from_json(f) for f in data.get("frames", [])]
        session = cls(
            path=path,
            name=data.get("name", path.name),
            film_profile=data.get("film_profile", "generic-c41"),
            roll_settings=data.get("roll_settings") or {},
            lock_base=bool(data.get("lock_base", False)),
            flats=dict(data.get("flats", {})),
            frames=frames,
            camera=data.get("camera", ""),
            created=data.get("created", ""),
            notes=data.get("notes", ""),
        )
        for folder in (session.raw_dir, session.flat_dir, session.export_dir):
            folder.mkdir(parents=True, exist_ok=True)
        return session

    def save(self) -> None:
        data = {
            "version": SESSION_VERSION,
            "app": "Belka",
            "name": self.name,
            "film_profile": self.film_profile,
            "roll_settings": self.roll_settings,
            "lock_base": self.lock_base,
            "flats": self.flats,
            "camera": self.camera,
            "created": self.created,
            "notes": self.notes,
            "frames": [asdict(f) for f in self.frames],
        }
        self.path.mkdir(parents=True, exist_ok=True)
        # Write atomically: a crash mid-save must not lose the roll index.
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix=".rollo-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, self.index_file)

    # ------------------------------------------------------------ frames
    def next_id(self) -> str:
        used = [int(f.id) for f in self.frames if f.id.isdigit()]
        return f"{(max(used) + 1) if used else 1:03d}"

    def new_capture_basename(self, frame_id: str, suffix: str = "") -> str:
        safe = re.sub(r"[^\w\-]+", "_", self.name)[:40]
        return f"{safe}_{frame_id}{suffix}"

    def add_frame(self, files: list[Path], mode: str = "single", lights: list | None = None) -> Frame:
        rel = [str(Path(f).resolve().relative_to(self.path.resolve())) if _inside(f, self.path) else str(Path(f).resolve()) for f in files]
        frame = Frame(
            id=self.next_id(),
            files=rel,
            mode=mode,
            lights=[list(map(float, l)) for l in (lights or [])],
            captured=datetime.now().isoformat(timespec="seconds"),
        )
        self.frames.append(frame)
        self.save()
        return frame

    def import_files(self, files: list[Path], copy: bool = True) -> tuple[list[Frame], list[str]]:
        """Add files as frames. Returns the frames added and one message per failure."""
        added, errors = [], []
        for src in files:
            src = Path(src)
            frame_id = self.next_id()
            dest = src
            if copy:
                dest = self._unique_raw_path(self.new_capture_basename(frame_id), src.suffix.lower())
                created = False
                try:
                    # Exclusive create: never overwrite a raw file kept from a
                    # removed frame that happened to have the same id.
                    with open(src, "rb") as fin, open(dest, "xb") as fout:
                        created = True
                        shutil.copyfileobj(fin, fout, 1 << 20)
                    shutil.copystat(src, dest)
                except OSError as exc:
                    if created:
                        dest.unlink(missing_ok=True)  # a partial copy of ours
                    errors.append(f"{src.name}: {exc}")
                    continue
            added.append(self.add_frame([dest]))
        return added, errors

    def _unique_raw_path(self, stem: str, suffix: str) -> Path:
        dest = self.raw_dir / f"{stem}{suffix}"
        n = 2
        while dest.exists():
            dest = self.raw_dir / f"{stem}-{n}{suffix}"
            n += 1
        return dest

    def remove_frame(self, frame: Frame, delete_files: bool = False) -> list[Path]:
        """Take the frame out of the roll; with ``delete_files`` its captures go to the
        system trash (recoverable, as Lightroom's "Delete from Disk" does).

        The roll is saved first, so the catalogue never lists a file that was
        trashed. Returns the files that could not be trashed; they stay on disk.
        Files outside the roll folder (imported in place) are never touched.
        """
        self.frames = [f for f in self.frames if f.id != frame.id]
        self.save()
        kept: list[Path] = []
        if delete_files:
            for rel in frame.files:
                p = self.resolve(rel)
                if p.exists() and _inside(p, self.path) and not _move_to_trash(p):
                    kept.append(p)
        return kept

    def frame(self, frame_id: str) -> Frame | None:
        return next((f for f in self.frames if f.id == frame_id), None)

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.path / p

    # ------------------------------------------------------------ settings
    def roll_defaults(self) -> DevelopSettings:
        settings = DevelopSettings.from_json(self.roll_settings)
        if not self.roll_settings:
            settings.profile_id = self.film_profile
        return settings

    def settings_for(self, frame: Frame) -> DevelopSettings:
        roll = self.roll_defaults()
        settings = DevelopSettings.from_json(frame.settings) if frame.settings else roll.copy()
        if self.lock_base and roll.base is not None:
            settings.base = roll.base
        return settings

    def set_frame_settings(self, frame: Frame, settings: DevelopSettings) -> None:
        frame.settings = settings.to_json()
        self.save()

    def apply_to_all(self, settings: DevelopSettings, keep_geometry: bool = True) -> None:
        for frame in self.frames:
            current = self.settings_for(frame)
            new = settings.copy()
            if keep_geometry:
                new.crop, new.rotation, new.flip_h, new.flip_v = current.crop, current.rotation, current.flip_h, current.flip_v
            frame.settings = new.to_json()
        self.roll_settings = settings.copy(crop=None).to_json()
        self.film_profile = settings.profile_id
        self.save()

    # ------------------------------------------------------------ flats
    def store_flat(self, rgb: np.ndarray, role: str = FLAT_SINGLE) -> Path:
        path = self.flat_dir / f"flat_{role}.npy"
        flatfield.save_flat(path, flatfield.build_flat(rgb))
        # A new flat for a role replaces any older one, including ones stored
        # under an exact light colour by earlier versions.
        for key in [k for k in self.flats if k != role and _legacy_role(k) == role]:
            del self.flats[key]
        self.flats[role] = str(path.relative_to(self.path))
        self.save()
        return path

    def flat_path(self, role: str) -> Path | None:
        rel = self.flats.get(role)
        if rel is None:
            rel = next((v for k, v in self.flats.items() if _legacy_role(k) == role), None)
        return self.resolve(rel) if rel else None

    def flat_for(self, role: str) -> np.ndarray | None:
        path = self.flat_path(role)
        return flatfield.load_flat(path) if path else None

    def flat_roles(self) -> set[str]:
        return {k if k in (FLAT_SINGLE,) or k.startswith("rgb-") else (_legacy_role(k) or k) for k in self.flats}

    def clear_flats(self) -> None:
        for rel in self.flats.values():
            p = self.resolve(rel)
            if p.exists():
                p.unlink()
        self.flats.clear()
        self.save()


def _move_to_trash(path: Path) -> bool:
    from PySide6.QtCore import QFile

    # PySide returns a bare bool here (C++ also hands back the path in the trash).
    result = QFile.moveToTrash(str(path))
    ok = result[0] if isinstance(result, tuple) else result
    return bool(ok) and not path.exists()


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def primary_file(files: list[str]) -> str:
    """The camera raw when a frame was shot RAW+JPEG, else the first file."""
    for rel in files:
        if Path(rel).suffix.lower() in RAW_EXTENSIONS:
            return rel
    return files[0]


def load_frame(session: Session, frame: Frame, half_size: bool = True, max_side: int | None = None) -> LinearImage:
    """Linear camera RGB of a frame, flat-field corrected.

    An RGB-sequential frame is assembled from its three captures: the red
    channel of the red-lit shot, the green of the green-lit one and so on.
    """
    if frame.mode == "rgb" and len(frame.files) == 3:
        parts = []
        first: LinearImage | None = None
        flats_used = 0
        for idx, rel in enumerate(frame.files):
            image = load_linear(session.resolve(rel), half_size=half_size, max_side=max_side)
            flat = session.flat_for(flat_role("rgb", idx))
            flats_used += flat is not None
            rgb = flatfield.apply_flat(image.rgb, flat)
            parts.append(rgb[..., idx])
            if first is not None and image.rgb.shape[:2] != first.rgb.shape[:2]:
                raise ValueError("Las tres tomas RGB tienen tamaños distintos")
            first = first or image
        rgb = np.stack(parts, axis=-1)
        assert first is not None
        meta = {**first.meta, "rgb_sequential": True, "flat_applied": flats_used == 3}
        return LinearImage(rgb=rgb, camera_matrix=first.camera_matrix, meta=meta)

    image = load_linear(session.resolve(primary_file(frame.files)), half_size=half_size, max_side=max_side)
    flat = session.flat_for(FLAT_SINGLE)
    image.rgb = flatfield.apply_flat(image.rgb, flat)
    image.meta["flat_applied"] = flat is not None
    return image
