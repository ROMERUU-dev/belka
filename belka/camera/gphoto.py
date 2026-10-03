"""libgphoto2 backend: tethered capture, live view and settings over USB.

Works with any camera libgphoto2 can remote-control (Nikon, Canon, Sony,
Fujifilm, Panasonic, ...); the control names each brand uses come from
``belka/data/cameras.json`` (see ``belka.camera.models``). The Nikon D780
is model "Nikon DSC D780", USB 04b0:0446.
"""

from __future__ import annotations

import time
from pathlib import Path

from belka.camera import models
from belka.camera.base import (
    SETTING_LABELS,
    CameraBackend,
    CameraBusyError,
    CameraError,
    CameraInfo,
    CameraSetting,
    CaptureHint,
)
from belka.i18n import _
from belka.system import free_camera_from_desktop


def _gp():
    try:
        import gphoto2 as gp
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise CameraError("Falta el módulo gphoto2 de Python (pip install gphoto2)") from exc
    return gp


def detect() -> list[CameraInfo]:
    gp = _gp()
    try:
        found = gp.Camera.autodetect()
    except gp.GPhoto2Error:
        return []
    # Index access: iterating a CameraList is deprecated in python-gphoto2.
    return [CameraInfo(model=found.get_name(i), port=found.get_value(i), backend="gphoto2") for i in range(found.count())]


def _wrap(exc: Exception) -> CameraError:
    gp = _gp()
    code = getattr(exc, "code", None)
    if code == gp.GP_ERROR_IO_USB_CLAIM:
        return CameraBusyError(_(
            "Otro programa tiene la cámara (normalmente el escritorio la monta como disco) y Belka no pudo "
            "liberarla. Desmóntala en Archivos y pulsa ⟳."))
    if code == gp.GP_ERROR_CAMERA_BUSY:
        return CameraError(_("La cámara está ocupada; espera a que termine de guardar e inténtalo de nuevo."))
    if code in (gp.GP_ERROR_MODEL_NOT_FOUND, gp.GP_ERROR_IO_USB_FIND, gp.GP_ERROR_UNKNOWN_PORT):
        return CameraError(_("No se encontró la cámara. ¿Está encendida y conectada por USB? Pulsa ⟳ para buscarla de nuevo."))
    if code == gp.GP_ERROR_IO:
        return CameraError(_("Error de comunicación USB con la cámara. Revisa el cable o vuelve a conectarla."))
    return CameraError(str(exc))


class GPhotoBackend(CameraBackend):
    capabilities = frozenset({"preview", "settings", "focus"})

    def __init__(self, info: CameraInfo):
        super().__init__(info)
        self.profile = models.profile_for(info.model)
        support = models.support(info.model)
        if support is not None and not support.preview:
            self.capabilities = self.capabilities - {"preview"}
        self._camera = None
        self._names: dict[str, str] = {}
        self._liveview_widget = ""
        self._autofocus_widget = ""
        self._liveview = False

    # ------------------------------------------------------------ session
    def open(self) -> None:
        gp = _gp()
        error = self._try_open(gp)
        if error is not None and error.code in (gp.GP_ERROR_IO_USB_FIND, gp.GP_ERROR_UNKNOWN_PORT,
                                                gp.GP_ERROR_MODEL_NOT_FOUND):
            # Power-cycling or replugging the camera gives it a new USB
            # address: look it up again by model.
            same = [c for c in detect() if c.model == self.info.model]
            if len(same) == 1 and same[0].port != self.info.port:
                self.info = same[0]
                error = self._try_open(gp)
        if error is not None and error.code == gp.GP_ERROR_IO_USB_CLAIM:
            # GNOME mounts cameras as soon as they are plugged in (again after
            # a replug), which claims the USB interface. Unmount, retry once.
            freed = free_camera_from_desktop()
            time.sleep(0.8)
            error = self._try_open(gp)
            if error is not None and error.code == gp.GP_ERROR_IO_USB_CLAIM and not freed:
                raise CameraBusyError(_("Otro programa tiene la cámara. Cierra el visor de fotos o desmonta la "
                                        "cámara en Archivos y pulsa ⟳.")) from error
        if error is not None:
            raise _wrap(error) from error
        self._resolve_names()

    def _try_open(self, gp):
        try:
            self._open_once(gp)
            return None
        except gp.GPhoto2Error as exc:
            return exc

    def _open_once(self, gp) -> None:
        camera = gp.Camera()
        ports = gp.PortInfoList()
        ports.load()
        camera.set_port_info(ports[ports.lookup_path(self.info.port)])
        abilities = gp.CameraAbilitiesList()
        abilities.load()
        camera.set_abilities(abilities[abilities.lookup_model(self.info.model)])
        camera.init()
        self._camera = camera

    def close(self) -> None:
        if self._camera is None:
            return
        try:
            self.stop_preview()
            self._camera.exit()
        except Exception:
            pass
        self._camera = None

    def _cam(self):
        if self._camera is None:
            raise CameraError(_("La cámara no está conectada"))
        return self._camera

    def summary(self) -> str:
        try:
            return str(self._cam().get_summary())
        except Exception as exc:
            return f"{self.info.model}\n{exc}"

    # ------------------------------------------------------------ capture
    def capture(self, dest_dir: Path, basename: str, hint: CaptureHint | None = None) -> list[Path]:
        gp = _gp()
        cam = self._cam()
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CameraError(_("No se puede escribir en {path}: {msg}").format(path=dest_dir, msg=exc.strerror or exc)) from exc
        try:
            first = cam.capture(gp.GP_CAPTURE_IMAGE)
            files = [self._download(first.folder, first.name, dest_dir, basename)]
            # RAW+JPEG produces a second file, announced as an event. Stop at
            # "capture complete", after ~0.6 s of silence, or after 3 s.
            deadline = time.monotonic() + 3.0
            quiet = 0
            while time.monotonic() < deadline and quiet < 2:
                event, data = cam.wait_for_event(300)
                if event == gp.GP_EVENT_FILE_ADDED:
                    files.append(self._download(data.folder, data.name, dest_dir, basename))
                    quiet = 0
                elif event == gp.GP_EVENT_CAPTURE_COMPLETE:
                    break
                elif event == gp.GP_EVENT_TIMEOUT:
                    quiet += 1
        except gp.GPhoto2Error as exc:
            raise _wrap(exc) from exc
        except OSError as exc:
            raise CameraError(_("No se pudo guardar la foto en {path}: {msg}").format(path=dest_dir, msg=exc.strerror or exc)) from exc
        return files

    def _download(self, folder: str, name: str, dest_dir: Path, basename: str) -> Path:
        gp = _gp()
        suffix = Path(name).suffix.lower() or ".raw"
        target = dest_dir / f"{basename}{suffix}"
        n = 2
        while target.exists():
            target = dest_dir / f"{basename}-{n}{suffix}"
            n += 1
        cam_file = self._cam().file_get(folder, name, gp.GP_FILE_TYPE_NORMAL)
        cam_file.save(str(target))
        return target

    # ------------------------------------------------------------ live view
    def preview(self) -> bytes | None:
        gp = _gp()
        try:
            cam_file = self._cam().capture_preview()
        except gp.GPhoto2Error as exc:
            raise _wrap(exc) from exc
        self._liveview = True
        return bytes(cam_file.get_data_and_size())

    def stop_preview(self) -> None:
        if not self._liveview or self._camera is None:
            return
        self._liveview = False
        # Nikon and Canon leave live view when "viewfinder" is cleared, which
        # lowers the mirror and saves battery. Sony and Fujifilm have no such
        # control: live view just stops being polled.
        if not self._liveview_widget:
            return
        try:
            self._set_widget(self._liveview_widget, 0)
        except CameraError:
            pass

    # ------------------------------------------------------------ settings
    def _resolve_names(self) -> None:
        """Pick, once per connection, which of the brand's control names this camera has."""
        gp = _gp()
        self._names.clear()
        # Unresolved (config unreadable): try the brand's first names anyway.
        self._liveview_widget = next(iter(self.profile.liveview), "")
        self._autofocus_widget = next(iter(self.profile.autofocus), "")
        try:
            config = self._cam().get_config()
        except gp.GPhoto2Error:
            return
        for key, candidates in self.profile.widgets.items():
            name = _first_widget(gp, config, candidates)
            if name:
                self._names[key] = name
        self._liveview_widget = _first_widget(gp, config, self.profile.liveview)
        self._autofocus_widget = _first_widget(gp, config, self.profile.autofocus)

    def settings(self) -> list[CameraSetting]:
        gp = _gp()
        try:
            config = self._cam().get_config()
        except gp.GPhoto2Error as exc:
            raise _wrap(exc) from exc
        result = []
        for key, name in self._names.items():
            try:
                widget = config.get_child_by_name(name)
            except gp.GPhoto2Error:
                continue
            result.append(self._describe(key, widget))
        return result

    def _describe(self, key: str, widget) -> CameraSetting:
        gp = _gp()
        wtype = widget.get_type()
        value = widget.get_value()
        setting = CameraSetting(
            key=key,
            name=widget.get_name(),
            label=SETTING_LABELS.get(key, widget.get_label()),
            kind="text",
            value=value,
            readonly=bool(widget.get_readonly()),
        )
        if wtype in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU):
            setting.kind = "choice"
            setting.choices = [widget.get_choice(i) for i in range(widget.count_choices())]
        elif wtype == gp.GP_WIDGET_RANGE:
            setting.kind = "range"
            setting.range = tuple(widget.get_range())
        elif wtype == gp.GP_WIDGET_TOGGLE:
            setting.kind = "toggle"
        return setting

    def _set_widget(self, name: str, value: object) -> None:
        gp = _gp()
        cam = self._cam()
        try:
            config = cam.get_config()
            widget = config.get_child_by_name(name)
            widget.set_value(value)
            try:
                cam.set_single_config(name, widget)
            except (AttributeError, gp.GPhoto2Error):
                cam.set_config(config)
        except gp.GPhoto2Error as exc:
            raise _wrap(exc) from exc

    def set_setting(self, name: str, value: object) -> None:
        self._set_widget(name, value)

    # ------------------------------------------------------------ focus
    def focus_step(self, steps: int) -> None:
        """Move the lens focus; positive = towards infinity.

        Needs live view and an AF lens. Nikon takes motor steps, Sony a step
        size of 1-7, Canon a list of "Near/Far 1..3" choices.
        """
        gp = _gp()
        try:
            config = self._cam().get_config()
        except gp.GPhoto2Error as exc:
            raise _wrap(exc) from exc
        name = _first_widget(gp, config, self.profile.focus.widgets)
        if not name:
            raise CameraError(_("Esta cámara no expone el control de foco por USB"))
        widget = config.get_child_by_name(name)
        size = 1 if abs(steps) < 50 else (2 if abs(steps) < 300 else 3)
        if widget.get_type() == gp.GP_WIDGET_RANGE:
            lo, hi, _step = widget.get_range()
            units = self.profile.focus.units or ("steps" if hi - lo > 100 else "size")
            value = steps if units == "steps" else (size if steps > 0 else -size)
            self._set_widget(name, float(max(lo, min(hi, value))))
            return
        choices = [widget.get_choice(i) for i in range(widget.count_choices())]
        wanted = f"{'Far' if steps > 0 else 'Near'} {size}"
        match = next((c for c in choices if c.lower() == wanted.lower()), None)
        if match is None and len(choices) == 7:
            # libgphoto2 translates choice labels to the desktop language, so
            # fall back to Canon's fixed order: Near 1-3, None, Far 1-3.
            match = choices[3 + size] if steps > 0 else choices[size - 1]
        if match is None:
            raise CameraError(_("No se reconoce el control de foco de esta cámara"))
        self._set_widget(name, match)

    def autofocus(self) -> None:
        if not self._autofocus_widget:
            raise CameraError(_("Esta cámara no permite autoenfoque remoto"))
        self._set_widget(self._autofocus_widget, 1)
        if self.profile.autofocus_release:
            # Sony's AF is a half-pressed shutter button: hold it, then let go.
            time.sleep(self.profile.autofocus_release)
            self._set_widget(self._autofocus_widget, 0)


def _first_widget(gp, config, names) -> str:
    """The first of ``names`` this camera's configuration has, or ""."""
    for name in names:
        try:
            config.get_child_by_name(name)
        except gp.GPhoto2Error:
            continue
        return name
    return ""
