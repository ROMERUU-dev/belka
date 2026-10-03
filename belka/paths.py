"""Where Belka keeps its data, following the XDG layout."""

from __future__ import annotations

import os
import re
from pathlib import Path


def data_dir() -> Path:
    """Read-only data shipped with the package (profiles, presets, icons)."""
    return Path(__file__).resolve().parent / "data"


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "belka"


def user_profiles_dir() -> Path:
    return config_dir() / "profiles"


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "belka"


def pictures_dir() -> Path:
    """The user's Pictures folder (``Imágenes`` on a Spanish desktop)."""
    user_dirs = config_dir().parent / "user-dirs.dirs"
    if user_dirs.is_file():
        match = re.search(r'^XDG_PICTURES_DIR="([^"]+)"', user_dirs.read_text(encoding="utf-8"), re.M)
        if match:
            return Path(match.group(1).replace("$HOME", str(Path.home())))
    return Path.home() / "Pictures"


def default_sessions_dir() -> Path:
    return pictures_dir() / "Belka"
