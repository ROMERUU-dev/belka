"""What Belka knows about each camera: libgphoto2 control names and connection tips.

The per-brand data lives in ``belka/data/cameras.json``, so supporting
another camera rarely needs code. Whether libgphoto2 can fire a model or show
its live view comes from libgphoto2's own model list, which is read from the
driver files on disk and never touches USB.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache

from belka import paths
from belka.i18n import _, language

# libgphoto2 driver states, as the dialog names them.
DRIVER_STATUS = {0: "production", 1: "testing", 2: "experimental", 3: "deprecated"}
_USB_PORTS = 4 | 32 | 64  # GP_PORT_USB | GP_PORT_USB_DISK_DIRECT | GP_PORT_USB_SCSI


@dataclass(frozen=True)
class FocusDrive:
    widgets: tuple[str, ...]
    kind: str  # "range" or "choice"
    units: str = ""  # range only: "steps" (motor steps), "size" (step size 1-3), "" = guess from the range


@dataclass(frozen=True)
class CameraProfile:
    brand: str  # brand id from cameras.json, "generic" when none matched
    name: str
    widgets: dict[str, tuple[str, ...]]
    focus: FocusDrive
    autofocus: tuple[str, ...]
    autofocus_release: float  # seconds to hold the AF "button" before releasing it; 0 = no release
    liveview: tuple[str, ...]
    usb_mode: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ModelSupport:
    model: str
    capture: bool
    preview: bool
    status: str  # a value of DRIVER_STATUS


def _text(value) -> str:
    """A cameras.json text: {"es": ..., "en": ...} or plain Spanish."""
    if isinstance(value, dict):
        return value.get(language()) or value.get("es", "")
    return _(value) if value else ""


@cache
def _catalog() -> dict:
    return json.loads((paths.data_dir() / "cameras.json").read_text(encoding="utf-8"))


def brand_for(model: str) -> dict | None:
    return next((b for b in _catalog()["brands"] if re.search(b["match"], model)), None)


def profile_for(model: str) -> CameraProfile:
    """Generic data refined by the brand, its matching series and the exact model."""
    generic = _catalog()["generic"]
    brand = brand_for(model)
    series = [s for s in brand.get("series", []) if re.search(s["match"], model)] if brand else []
    layers = [generic] + ([brand] if brand else []) + series
    if brand is not None and model in brand.get("models", {}):
        layers.append(brand["models"][model])
    # The most specific names go first; generic ones stay as a fallback for odd firmware.
    widgets = {
        key: tuple(dict.fromkeys(n for layer in reversed(layers) for n in layer.get("widgets", {}).get(key, [])))
        for key in generic["widgets"]
    }
    merged: dict = {}
    for layer in layers:
        merged.update(layer)
    focus = merged["focus"]
    name = next((s["name"] for s in series if "name" in s), brand["name"] if brand else generic["name"])
    return CameraProfile(
        brand=brand["id"] if brand else "generic",
        name=_text(name),
        widgets=widgets,
        focus=FocusDrive(tuple(focus.get("widgets", [])), focus.get("type", "range"), focus.get("units", "")),
        autofocus=tuple(merged["autofocus"].get("widgets", [])),
        autofocus_release=float(merged["autofocus"].get("release_after", 0.0)),
        liveview=tuple(merged.get("liveview", [])),
        usb_mode=_text(merged.get("usb_mode")),
        notes=tuple(_text(n) for layer in layers for n in layer.get("notes", [])),
    )


@cache
def _abilities() -> dict[str, ModelSupport]:
    """libgphoto2's static model list (USB still cameras only); empty without gphoto2."""
    try:
        import gphoto2 as gp

        abilities = gp.CameraAbilitiesList()
        abilities.load()
    except Exception:  # ImportError, or a broken libgphoto2 install
        return {}
    result = {}
    for i in range(abilities.count()):
        a = abilities[i]
        if a.device_type != gp.GP_DEVICE_STILL_CAMERA or not a.port & _USB_PORTS:
            continue
        result[a.model] = ModelSupport(
            model=a.model,
            capture=bool(a.operations & gp.GP_OPERATION_CAPTURE_IMAGE),
            preview=bool(a.operations & gp.GP_OPERATION_CAPTURE_PREVIEW),
            status=DRIVER_STATUS.get(a.status, "testing"),
        )
    return result


def support(model: str) -> ModelSupport | None:
    return _abilities().get(model)


def capture_models() -> list[ModelSupport]:
    """Every USB model libgphoto2 can fire remotely, sorted by name."""
    return sorted((m for m in _abilities().values() if m.capture), key=lambda m: m.model.lower())
