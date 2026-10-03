"""Sequencing light changes and camera captures.

A frame is one capture with white or tinted light, or three captures with
red, green and blue light. Before each exposure the panel switches colour and
the flow waits ``settle_ms`` so the LCD has fully changed (and its backlight
settled) before the shutter opens.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QTimer, Signal

from belka.camera.base import CaptureHint
from belka.camera.worker import CameraController
from belka.core import flatfield
from belka.core.pipeline import estimate_base
from belka.core.rawio import RAW_EXTENSIONS, load_linear
from belka.core.session import Session, flat_role
from belka.i18n import _
from belka.light.geometry import LightSettings, display_value, tint_for_base

_ids = itertools.count(1)


@dataclass
class Step:
    display: tuple[float, float, float]
    emission: tuple[float, float, float]


@dataclass
class Job:
    kind: str  # "frame", "flat" or "calibration"
    steps: list[Step]
    session: Session
    uses_panel: bool
    id: int = field(default_factory=lambda: next(_ids))
    files: list[list[Path]] = field(default_factory=list)


def pick_main_file(files: list[Path]) -> Path:
    for f in files:
        if f.suffix.lower() in RAW_EXTENSIONS:
            return f
    return files[0]


class CaptureFlow(QObject):
    frameCaptured = Signal(object, object)  # frame, the roll it was saved in
    flatsChanged = Signal()
    tintCalibrated = Signal(tuple)
    message = Signal(str)
    error = Signal(str)
    busyChanged = Signal(bool)

    def __init__(
        self,
        camera: CameraController,
        light: LightSettings,
        session: Callable[[], Session | None],
        show_color: Callable[[tuple[float, float, float]], None],
        light_visible: Callable[[], bool],
        set_capturing: Callable[[bool], None],
        parent: QObject | None = None,
        panel_ready: Callable[[], bool] | None = None,
    ):
        super().__init__(parent)
        self.camera = camera
        self.light = light
        self._session = session
        self._show_color = show_color
        self._light_visible = light_visible
        self._set_capturing = set_capturing
        # True once the light panel is the window actually on top. On Wayland
        # an app cannot raise its own window, only ask for activation, so the
        # first exposure waits for it instead of photographing the main window.
        self._panel_ready = panel_ready or (lambda: True)
        self._job: Job | None = None
        self._pending: list[Step] = []
        camera.worker.captured.connect(self._on_captured)
        camera.worker.capture_failed.connect(self._on_failed)

    @property
    def busy(self) -> bool:
        return self._job is not None

    # ------------------------------------------------------------ start
    def _steps(self) -> list[Step]:
        if not self._light_visible():
            # External light source: nothing to switch.
            return [Step(display=(1.0, 1.0, 1.0), emission=(1.0, 1.0, 1.0))]
        if self.light.mode == "rgb":
            return [Step(self.light.display_rgb(c), self.light.emission(c)) for c in range(3)]
        return [Step(self.light.display_rgb(), self.light.emission())]

    def _start(self, kind: str, steps: list[Step]) -> bool:
        if self._job is not None:
            self.message.emit(_("Espera: hay una captura en curso."))
            return False
        if not self.camera.connected:
            self.error.emit(_("Conecta una cámara primero."))
            return False
        session = self._session()
        if session is None:
            self.error.emit(_("Crea o abre un rollo primero."))
            return False
        # The job keeps its roll: opening another roll mid-capture must not
        # move the half-shot frame into it.
        self._job = Job(kind=kind, steps=steps, session=session, uses_panel=self._light_visible())
        self._pending = list(steps)
        self.busyChanged.emit(True)
        self._set_capturing(True)
        self._next()
        return True

    def cancel(self, reason: str = "") -> None:
        """Abort the current job; captures still in flight are ignored."""
        if self._job is None:
            return
        self._end()
        if reason:
            self.error.emit(reason)

    def capture_frame(self) -> bool:
        if self.light.mode == "rgb" and not self._light_visible():
            self.error.emit(_("El modo RGB secuencial necesita el panel de luz encendido."))
            return False
        return self._start("frame", self._steps())

    def capture_flat(self) -> bool:
        return self._start("flat", self._steps())

    def calibrate_tint(self) -> bool:
        if not self._light_visible():
            self.error.emit(_("Enciende el panel de luz para calibrar su tinte."))
            return False
        # Measure through the current tint so repeated calibrations converge
        # (the screen primaries leak into neighbouring camera channels).
        if self.light.mode == "white":
            emission = tuple(self.light.brightness for _ in range(3))
        else:
            emission = tuple(self.light.brightness * v for v in self.light.tint)
        return self._start("calibration", [Step(tuple(display_value(v) for v in emission), emission)])

    # ------------------------------------------------------------ sequence
    def _next(self) -> None:
        job = self._job
        if job is None:
            return
        if not self._pending:
            self._finish()
            return
        if job.uses_panel and not self._light_visible():
            self.cancel(_("Se apagó el panel de luz durante la captura: captura cancelada."))
            return
        step = self._pending.pop(0)
        self._show_color(step.display)
        session = job.session
        n = len(job.files)
        if job.kind == "frame":
            frame_id = session.next_id()
            suffix = "" if len(job.steps) == 1 else "_" + "RGB"[n]
            dest, basename = session.raw_dir, session.new_capture_basename(frame_id, suffix)
        elif job.kind == "flat":
            role = flat_role("rgb" if len(job.steps) == 3 else "single", n)
            dest, basename = session.flat_dir, f"flat_{role}"
        else:
            dest, basename = session.flat_dir, "calibracion_tinte"
        hint = CaptureHint(kind=job.kind, light=step.display, part=n, parts=len(job.steps))
        settle = self.light.settle_ms if job.uses_panel else 0
        token = job.id
        QTimer.singleShot(settle, lambda: self._fire(token, dest, basename, hint, 0))

    def _fire(self, token: int, dest: Path, basename: str, hint: CaptureHint, waited_ms: int) -> None:
        job = self._job
        if job is None or job.id != token:
            return
        if job.uses_panel:
            if not self._light_visible():
                self.cancel(_("Se apagó el panel de luz durante la captura: captura cancelada."))
                return
            if not self._panel_ready():
                if waited_ms >= 2500:
                    self.cancel(_("El panel de luz no quedó al frente, así que no se disparó. "
                                  "Captura desde el propio panel con Espacio."))
                    return
                QTimer.singleShot(100, lambda: self._fire(token, dest, basename, hint, waited_ms + 100))
                return
        self.camera.request_capture.emit(dest, basename, hint, token)

    def _on_captured(self, files: list, tag: object) -> None:
        if self._job is None or tag != self._job.id:
            return
        self._job.files.append([Path(f) for f in files])
        self._next()

    def _on_failed(self, message: str, tag: object) -> None:
        if self._job is None or tag != self._job.id:
            return
        self._end()
        self.error.emit(_("Falló la captura: {msg}").format(msg=message))

    def _end(self) -> None:
        self._job = None
        self._pending = []
        self._set_capturing(False)
        self._show_color(self.light.display_rgb())
        self.busyChanged.emit(False)

    def _finish(self) -> None:
        job = self._job
        assert job is not None
        session = job.session
        try:
            if job.kind == "frame":
                main = [pick_main_file(f) for f in job.files]
                displays = [list(s.display) for s in job.steps]
                mode = "rgb" if len(main) == 3 else "single"
                frame = session.add_frame(main, mode=mode, lights=displays)
                self._end()
                self.frameCaptured.emit(frame, session)
                return
            if job.kind == "flat":
                warnings = []
                mode = "rgb" if len(job.steps) == 3 else "single"
                for part, files in enumerate(job.files):
                    image = load_linear(pick_main_file(files), half_size=True)
                    warnings += flatfield.flat_warnings(image.rgb)
                    session.store_flat(image.rgb, flat_role(mode, part))
                self._end()
                self.flatsChanged.emit()
                if warnings:
                    self.error.emit(" ".join(warnings))
                else:
                    self.message.emit(_("Flat-field guardado."))
                return
            image = load_linear(pick_main_file(job.files[0]), half_size=True)
            base = estimate_base(image.rgb)
            tint = tint_for_base(base, job.steps[0].emission)
            self.light.tint = tint
            if self.light.mode == "white":
                # RGB mode keeps its mode: the tint sets the three primaries' levels.
                self.light.mode = "tint"
            self._end()
            self.tintCalibrated.emit(tint)
        except Exception as exc:
            self._end()
            self.error.emit(_("No se pudo procesar la captura: {msg}").format(msg=exc))
