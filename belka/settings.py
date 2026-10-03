"""User preferences, stored as JSON under ~/.config/belka."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from belka import paths


class Settings:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.config_dir() / "settings.json"
        self.data: dict = {}
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.data = {}

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".settings-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def recent_sessions(self) -> list[str]:
        return [p for p in self.get("recent_sessions", []) if Path(p, "belka-rollo.json").exists()]

    def add_recent_session(self, path: Path) -> None:
        items = [str(path)] + [p for p in self.get("recent_sessions", []) if p != str(path)]
        self.set("recent_sessions", items[:10])
