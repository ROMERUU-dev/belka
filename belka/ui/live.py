"""Inverted live view off the GUI thread.

The camera sends 5–10 JPEG frames a second; decoding, analysing and
inverting each one on the GUI thread made the whole window lag. Here a
worker thread takes only the newest frame (older ones are dropped while it
is busy) and reuses the film analysis for a second.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage

from belka.core import pipeline as pl
from belka.core.film import FilmProfile
from belka.core.rawio import decode_jpeg_preview
from belka.i18n import _

LIVE_SIDE = 640  # plenty for framing and focus checks in the capture panel
ANALYSIS_SECONDS = 1.0


@dataclass
class LiveJob:
    data: bytes
    settings: pl.DevelopSettings
    profile: FilmProfile


class _LiveWorker(QObject):
    done = Signal(object, str)  # QImage, note

    def __init__(self) -> None:
        super().__init__()
        self._analysis: tuple[tuple, float, pl.Analysis] | None = None

    @Slot(object)
    def run(self, job: LiveJob) -> None:
        try:
            image, note = self._invert(job)
        except Exception as exc:  # a bad frame must not kill the thread
            image, note = QImage.fromData(job.data), _("Vista en vivo sin invertir: {msg}").format(msg=exc)
        self.done.emit(image, note)

    def _invert(self, job: LiveJob) -> tuple[QImage, str]:
        rgb = decode_jpeg_preview(job.data)
        if rgb is None:
            return QImage(), ""
        rgb = pl.downsample(rgb, LIVE_SIDE)
        if float(np.percentile(rgb, 99)) < 0.03:
            # Nothing backlit in view: inverting a near-black 8-bit JPEG only
            # amplifies compression noise into confetti.
            return QImage.fromData(job.data), _("Muy oscuro para invertir: ¿está encendida la luz?")
        s = job.settings
        key = (s.analysis_key(), job.profile.id, rgb.shape)
        now = time.monotonic()
        if self._analysis is None or self._analysis[0] != key or now - self._analysis[1] > ANALYSIS_SECONDS:
            self._analysis = (key, now, pl.analyze(rgb, s, job.profile))
        out = pl.render(pl.orient(rgb, s.rotation, s.flip_h, s.flip_v), self._analysis[2], s, job.profile)
        rgb8 = pl.to_uint8(out)
        h, w = rgb8.shape[:2]
        return QImage(rgb8.data, w, h, w * 3, QImage.Format.Format_RGB888).copy(), ""


class LiveInverter(QObject):
    """Feeds the worker the newest live frame only; ``ready`` brings the inverted image back."""

    ready = Signal(object, str)
    _submit = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.thread = QThread(self)
        self.thread.setObjectName("belka-live")
        self.worker = _LiveWorker()
        self.worker.moveToThread(self.thread)
        self._submit.connect(self.worker.run)
        self.worker.done.connect(self._on_done)
        self._busy = False
        self._pending: LiveJob | None = None
        self.thread.start()

    def submit(self, data: bytes, settings: pl.DevelopSettings, profile: FilmProfile) -> None:
        # The live inversion ignores the frame's base, crop and colour picks:
        # framing needs a stable, automatic picture.
        settings = settings.copy(base=None, crop=None, auto_balance=1.0, neutral=(0.0, 0.0, 0.0))
        self._pending = LiveJob(data, settings, profile)
        self._pump()

    def _pump(self) -> None:
        if self._busy or self._pending is None:
            return
        job, self._pending = self._pending, None
        self._busy = True
        self._submit.emit(job)

    def _on_done(self, image: QImage, note: str) -> None:
        self._busy = False
        if not image.isNull():
            self.ready.emit(image, note)
        self._pump()

    def clear(self) -> None:
        self._pending = None

    def shutdown(self) -> None:
        self._pending = None
        self.thread.quit()
        self.thread.wait(5000)
