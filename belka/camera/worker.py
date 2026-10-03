"""Runs a camera backend on its own thread.

libgphoto2 calls block for hundreds of milliseconds (a capture can take
seconds), so every call goes through this worker. The GUI talks to it only
through queued signals.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, QTimer, Signal, Slot

from belka.camera import gphoto
from belka.camera.base import CameraBackend, CameraError, CameraInfo, CaptureHint
from belka.i18n import _

# CameraInfo.backend → backend class. Only libgphoto2 today; a platform SDK
# (Windows, macOS) would register here.
BACKENDS: dict[str, Callable[[CameraInfo], CameraBackend]] = {"gphoto2": gphoto.GPhotoBackend}


def make_backend(info: CameraInfo) -> CameraBackend:
    try:
        return BACKENDS[info.backend](info)
    except KeyError:
        raise CameraError(_("Tipo de cámara desconocido: {kind}").format(kind=info.backend)) from None


class CameraWorker(QObject):
    opened = Signal(object, str)  # CameraInfo, summary
    closed = Signal()
    settings_ready = Signal(list)
    preview_frame = Signal(bytes)
    liveview_stopped = Signal()
    captured = Signal(list, object)  # list[Path], tag
    capture_failed = Signal(str, object)  # message, tag
    failed = Signal(str)
    busy_changed = Signal(bool)
    cameras_found = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self._backend: CameraBackend | None = None
        self._liveview = False
        self._timer: QTimer | None = None
        self._busy = False

    @Slot()
    def setup(self) -> None:
        self._timer = QTimer(self)
        self._timer.setInterval(70)
        self._timer.timeout.connect(self._tick)

    def _set_busy(self, busy: bool) -> None:
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    # ------------------------------------------------------------ slots
    @Slot()
    def detect(self) -> None:
        try:
            found = gphoto.detect()
        except CameraError as exc:  # libgphoto2 missing: say why instead of "no camera"
            self.failed.emit(str(exc))
            found = []
        self.cameras_found.emit(found)

    @Slot(object)
    def open_camera(self, info: CameraInfo) -> None:
        self.close_camera()
        self._set_busy(True)
        try:
            try:
                backend = make_backend(info)
                backend.open()
            except Exception as exc:  # CameraError, or anything libgphoto2 throws
                self.failed.emit(str(exc))
                return
            self._backend = backend
            # The backend may have re-resolved a stale USB port.
            self.opened.emit(backend.info, backend.summary())
            try:
                self.settings_ready.emit(backend.settings())
            except Exception as exc:
                # Not fatal: capture and live view work without the form.
                self.settings_ready.emit([])
                self.failed.emit(_("No se pudieron leer los ajustes de la cámara: {msg}").format(msg=exc))
        finally:
            self._set_busy(False)

    @Slot()
    def close_camera(self) -> None:
        self._liveview = False
        if self._timer:
            self._timer.stop()
        if self._backend is not None:
            try:
                self._backend.close()
            finally:
                self._backend = None
                self.closed.emit()

    @Slot(bool)
    def set_liveview(self, enabled: bool) -> None:
        self._liveview = enabled and self._backend is not None and "preview" in self._backend.capabilities
        if not self._timer:
            return
        if self._liveview:
            self._timer.start()
        else:
            self._timer.stop()
            if self._backend is not None:
                try:
                    self._backend.stop_preview()
                except CameraError:
                    pass

    @Slot()
    def _tick(self) -> None:
        if not self._liveview or self._backend is None or self._busy:
            return
        try:
            data = self._backend.preview()
        except Exception as exc:
            self._liveview = False
            self._timer.stop()
            self.liveview_stopped.emit()
            self.failed.emit(_("Vista en vivo detenida: {msg}").format(msg=exc))
            return
        if data:
            self.preview_frame.emit(data)

    @Slot(object, str, object, object)
    def capture(self, dest_dir: Path, basename: str, hint: CaptureHint, tag: object) -> None:
        if self._backend is None:
            self.capture_failed.emit(_("No hay cámara conectada"), tag)
            return
        self._set_busy(True)
        try:
            files = self._backend.capture(Path(dest_dir), basename, hint)
            self.captured.emit(files, tag)
        except Exception as exc:
            # Any failure must reach the capture flow, or it waits forever.
            self.capture_failed.emit(str(exc), tag)
        finally:
            self._set_busy(False)

    @Slot(str, object)
    def set_setting(self, name: str, value: object) -> None:
        if self._backend is None:
            return
        try:
            self._backend.set_setting(name, value)
            self.settings_ready.emit(self._backend.settings())
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot()
    def refresh_settings(self) -> None:
        if self._backend is None:
            return
        try:
            self.settings_ready.emit(self._backend.settings())
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot(int)
    def focus(self, steps: int) -> None:
        if self._backend is None:
            return
        try:
            self._backend.focus_step(steps)
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot()
    def autofocus(self) -> None:
        if self._backend is None:
            return
        try:
            self._backend.autofocus()
        except Exception as exc:
            self.failed.emit(str(exc))


class CameraController(QObject):
    """GUI-side handle: owns the thread and forwards requests as signals."""

    request_detect = Signal()
    request_open = Signal(object)
    request_close = Signal()
    request_liveview = Signal(bool)
    request_capture = Signal(object, str, object, object)
    request_setting = Signal(str, object)
    request_refresh = Signal()
    request_focus = Signal(int)
    request_autofocus = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.thread = QThread(self)
        self.thread.setObjectName("belka-camera")
        self.worker = CameraWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.setup)
        self.request_detect.connect(self.worker.detect)
        self.request_open.connect(self.worker.open_camera)
        self.request_close.connect(self.worker.close_camera)
        self.request_liveview.connect(self.worker.set_liveview)
        self.request_capture.connect(self.worker.capture)
        self.request_setting.connect(self.worker.set_setting)
        self.request_refresh.connect(self.worker.refresh_settings)
        self.request_focus.connect(self.worker.focus)
        self.request_autofocus.connect(self.worker.autofocus)
        self.worker.opened.connect(self._on_opened)
        self.worker.closed.connect(self._on_closed)
        self.worker.busy_changed.connect(self._on_busy)
        self.info: CameraInfo | None = None
        self.busy = False
        self.thread.start()

    @property
    def connected(self) -> bool:
        return self.info is not None

    def _on_opened(self, info: CameraInfo, _summary: str) -> None:
        self.info = info

    def _on_closed(self) -> None:
        self.info = None

    def _on_busy(self, busy: bool) -> None:
        self.busy = busy

    def shutdown(self) -> None:
        # Blocking, so the camera is released before the thread stops.
        QMetaObject.invokeMethod(self.worker, "close_camera", Qt.ConnectionType.BlockingQueuedConnection)
        self.thread.quit()
        self.thread.wait(5000)
