"""Light source settings: where on the screen to light, and with what colour.

Positions are kept in millimetres so a printed adapter (or a film holder)
lines up with the same lit area on any screen. ``QScreen.physicalSize`` comes
from the monitor's EDID, which is sometimes off by a few percent, hence the
per-screen scale correction measured with a ruler.

Colours are linear light (emission) in [0, 1]; the screen needs values
encoded with its ~2.2 gamma, done in :func:`display_value`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

from belka import paths

SCREEN_GAMMA = 2.2
MODES = ("white", "tint", "rgb")


@dataclass(frozen=True)
class Adapter:
    id: str
    name: str
    frame_mm: tuple[float, float] | None
    light_mm: tuple[float, float] | None
    marks: bool = False
    notes: str = ""


def load_adapters() -> list[Adapter]:
    data = json.loads((paths.data_dir() / "adapters.json").read_text(encoding="utf-8"))
    result = []
    for entry in data["adapters"]:
        result.append(
            Adapter(
                id=entry["id"],
                name=entry["name"],
                frame_mm=tuple(entry["frame_mm"]) if entry.get("frame_mm") else None,
                light_mm=tuple(entry["light_mm"]) if entry.get("light_mm") else None,
                marks=bool(entry.get("marks", False)),
                notes=entry.get("notes", ""),
            )
        )
    return result


def display_value(linear: float) -> float:
    return max(0.0, min(1.0, linear)) ** (1.0 / SCREEN_GAMMA)


@dataclass
class LightSettings:
    screen: str = ""
    adapter: str = "35mm"
    rect_mm: tuple[float, float, float, float] | None = None  # x, y, w, h from the top-left
    mode: str = "white"
    brightness: float = 1.0  # linear emission
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # linear emission ratios, max 1
    fullscreen: bool = True
    guides: bool = True
    marks: bool = True
    hud: bool = True
    settle_ms: int = 350
    scale_correction: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict | None) -> "LightSettings":
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        if clean.get("rect_mm") is not None:
            clean["rect_mm"] = tuple(float(v) for v in clean["rect_mm"])
        if clean.get("tint") is not None:
            clean["tint"] = tuple(float(v) for v in clean["tint"])
        if clean.get("mode") not in MODES:
            clean["mode"] = "white"
        return cls(**clean)

    # ------------------------------------------------------------ colours
    def emission(self, channel: int | None = None) -> tuple[float, float, float]:
        """Linear emission for a capture. ``channel`` picks R/G/B in RGB mode."""
        b = max(0.0, min(1.0, self.brightness))
        if self.mode == "white":
            base = (1.0, 1.0, 1.0)
        else:
            base = self.tint
        if self.mode == "rgb":
            idx = 0 if channel is None else channel
            base = tuple(base[i] if i == idx else 0.0 for i in range(3))
        return tuple(b * v for v in base)

    def display_rgb(self, channel: int | None = None) -> tuple[float, float, float]:
        return tuple(display_value(v) for v in self.emission(channel))

    def capture_lights(self) -> list[tuple[float, float, float]]:
        """Display colours of each capture in one frame (three in RGB mode)."""
        if self.mode == "rgb":
            return [self.display_rgb(c) for c in range(3)]
        return [self.display_rgb()]


def tint_for_base(base_camera_rgb, current_emission: tuple[float, float, float]) -> tuple[float, float, float]:
    """Light colour that makes the film base read neutral to the camera.

    The orange mask passes far more red than blue, so a white light wastes most
    of the sensor's blue range. Scaling each primary by the inverse of what the
    camera measured through the base balances the three channels; with no
    knowledge of the screen/camera crosstalk this is one step of a fixed-point
    iteration, so calibrating twice converges further.
    """
    base = [max(float(v), 1e-6) for v in base_camera_rgb]
    weakest = min(base)
    proposal = [current_emission[i] * weakest / base[i] for i in range(3)]
    peak = max(proposal)
    if peak <= 0:
        return (1.0, 1.0, 1.0)
    return tuple(round(v / peak, 4) for v in proposal)
