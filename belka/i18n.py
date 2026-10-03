"""Interface language. Spanish is the source; other languages map from it.

``_("texto")`` returns the translation for the active language, or the
Spanish text when there is none, so a missing entry never breaks the UI.
"""

from __future__ import annotations

import locale
import os

_language = "es"
_tables: dict[str, dict[str, str]] = {}


def detect_language() -> str:
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(var, "")
        if value:
            return "es" if value.lower().startswith("es") else "en"
    try:
        loc = locale.getlocale()[0] or ""
    except ValueError:
        loc = ""
    return "es" if loc.lower().startswith("es") else "en"


def set_language(code: str | None) -> str:
    global _language
    _language = code if code in ("es", "en") else detect_language()
    if _language != "es" and _language not in _tables:
        if _language == "en":
            from belka.translations import en

            _tables["en"] = en.STRINGS
    return _language


def language() -> str:
    return _language


def _(text: str) -> str:
    if _language == "es":
        return text
    return _tables.get(_language, {}).get(text, text)
