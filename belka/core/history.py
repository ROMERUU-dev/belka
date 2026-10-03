"""Per-frame develop history: undo, redo and Lightroom's History panel.

Every step keeps the complete :class:`DevelopSettings`, so undoing or jumping
to any step is just loading its settings; nothing is replayed. Commits of the
same control less than a second apart (wheel or arrow-key nudges, a slider
released and grabbed again) fold into one step, as in Lightroom, so the panel
lists decisions rather than every intermediate value. "Same control" means the
same step name and the same fields changed: a label alone can be ambiguous
once translated, and the fields alone cannot tell "Enderezar" from a
perspective "Rotar" that also moves the angle.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import fields
from typing import NamedTuple

from belka.core.pipeline import DevelopSettings
from belka.i18n import _

MAX_ENTRIES = 200
COALESCE_SECONDS = 1.0

# A trailing slider value: "Exposición +0,35", "Enderezar −1,5°", "Grano 30".
_VALUE = re.compile(r"\s+([+\-−±]?\d+(?:[.,]\d+)?(?:\s?(?:%|°|EV|px))?)$")

# Steps that pick a named option: everything after the colon is the value,
# digits included ("Película: Portra 400" is not "Película: Portra" set to 400).
# The formats are the producers' own, so the prefixes follow the language.
_CHOICE_FORMATS = ("Película: {name}", "Salida: {name}", "Instantánea: {name}", "Upright: {mode}")


def split_label(label: str) -> tuple[str, str]:
    """Name and value of a step label, for the History panel's two columns.

    ("Enfoque: Cantidad", "40") for "Enfoque: Cantidad 40", ("Película",
    "Portra 400") for "Película: Portra 400", ("Upright", "auto") for
    "Upright: auto"; (label, "") when it carries no value.
    """
    for fmt in _CHOICE_FORMATS:
        prefix = _(fmt).split("{", 1)[0]
        if label.startswith(prefix) and label[len(prefix):].strip():
            return prefix.rstrip(": "), label[len(prefix):].strip()
    match = _VALUE.search(label)
    if match is not None and match.start() > 0:
        return label[: match.start()], match.group(1)
    name, sep, value = label.partition(": ")
    if sep and name and value:
        return name, value
    return label, ""


def _changed_controls(a: DevelopSettings, b: DevelopSettings) -> frozenset[tuple[str, int | None]]:
    """The controls whose values differ between two settings.

    Items of fixed-length number tuples count on their own, so two HSL bands,
    two grading channels or two crop edges are different controls; tuples of
    points (curves) and None-or-tuple fields count as one.
    """
    changed: set[tuple[str, int | None]] = set()
    for f in fields(DevelopSettings):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if x == y:
            continue
        if (isinstance(x, tuple) and isinstance(y, tuple) and len(x) == len(y)
                and not any(isinstance(v, tuple) for v in x + y)):
            changed.update((f.name, i) for i, (u, v) in enumerate(zip(x, y)) if u != v)
        else:
            changed.add((f.name, None))
    return frozenset(changed)


class HistoryEntry(NamedTuple):
    """One step; unpacks as ``label, settings``. The settings are the history's own: read-only."""

    label: str
    settings: DevelopSettings

    @property
    def name(self) -> str:
        return split_label(self.label)[0]

    @property
    def value(self) -> str:
        return split_label(self.label)[1]


class History:
    """Linear history of one frame, oldest step first; ``index`` is the step on screen."""

    def __init__(self, settings: DevelopSettings | None = None, label: str = "", *,
                 clock: Callable[[], float] = time.monotonic, max_entries: int = MAX_ENTRIES):
        self._clock = clock
        # Room for the starting step and at least one edit.
        self._max = max(2, max_entries)
        self._entries: list[HistoryEntry] = []
        self._index = -1
        # Time of the last push while it may still absorb the next one; None
        # after undo or a jump, so the next edit branches off as a new step.
        self._last_push: float | None = None
        if settings is not None:
            self._entries.append(HistoryEntry(label, settings.copy()))
            self._index = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def index(self) -> int:
        return self._index

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._entries) - 1

    @property
    def current(self) -> DevelopSettings | None:
        return self._entries[self._index].settings.copy() if self._index >= 0 else None

    def entries(self) -> list[HistoryEntry]:
        return list(self._entries)

    def push(self, label: str, settings: DevelopSettings) -> bool:
        """Record an edit; returns False when it changes nothing.

        Steps after the current one (undone edits) are discarded, as in any
        editor. Beyond ``max_entries`` the oldest edits are forgotten, but
        never the starting step, which Before / After compares against.
        """
        if self._index >= 0 and self._entries[self._index].settings == settings:
            return False
        now = self._clock()
        entry = HistoryEntry(label, settings.copy())
        if self._folds_into_current(entry, now):
            if self._entries[self._index - 1].settings == entry.settings:
                # Nudged back to where it started: the step cancels out.
                del self._entries[self._index]
                self._index -= 1
                self._last_push = None
                return True
            self._entries[self._index] = entry
        else:
            del self._entries[self._index + 1:]
            self._entries.append(entry)
            del self._entries[1:1 + max(0, len(self._entries) - self._max)]
            self._index = len(self._entries) - 1
        self._last_push = now
        return True

    def _folds_into_current(self, entry: HistoryEntry, now: float) -> bool:
        # Never the starting step: undoing must always get back to it.
        if self._last_push is None or self._index < 1 or now - self._last_push > COALESCE_SECONDS:
            return False
        current = self._entries[self._index]
        # Only steps that set a control to a value fold; actions such as
        # "Rotar" or "Pegar ajustes" stay one step each.
        if not (current.value and entry.value) or current.name != entry.name:
            return False
        previous = self._entries[self._index - 1].settings
        return _changed_controls(previous, current.settings) == _changed_controls(current.settings, entry.settings)

    def undo(self) -> DevelopSettings | None:
        return self.jump(self._index - 1) if self.can_undo else None

    def redo(self) -> DevelopSettings | None:
        return self.jump(self._index + 1) if self.can_redo else None

    def jump(self, index: int) -> DevelopSettings:
        """Make step ``index`` (0 = oldest) current and return a copy of its settings."""
        if not 0 <= index < len(self._entries):
            raise IndexError(f"paso de historial fuera de rango: {index}")
        self._index = index
        self._last_push = None
        return self._entries[index].settings.copy()

    def clear(self, label: str | None = None) -> None:
        """Forget every step but the current one, which becomes the starting point."""
        if self._index < 0:
            return
        entry = self._entries[self._index]
        self._entries = [entry if label is None else entry._replace(label=label)]
        self._index = 0
        self._last_push = None


class HistoryStore:
    """One :class:`History` per frame id, kept for as long as the roll is open."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, max_entries: int = MAX_ENTRIES):
        self._clock = clock
        self._max = max_entries
        self._histories: dict[str, History] = {}

    def get(self, frame_id: str, initial: DevelopSettings | None = None, label: str = "") -> History:
        """The frame's history, created on first use and seeded with ``initial``.

        An existing history that is still empty is seeded too, so a frame first
        touched before its settings were known still gets its starting step.
        """
        history = self._histories.get(frame_id)
        if history is None or (len(history) == 0 and initial is not None):
            history = History(initial, label, clock=self._clock, max_entries=self._max)
            self._histories[frame_id] = history
        return history

    def __contains__(self, frame_id: str) -> bool:
        return frame_id in self._histories

    def discard(self, frame_id: str) -> None:
        self._histories.pop(frame_id, None)

    def clear(self) -> None:
        self._histories.clear()
