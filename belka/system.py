"""Small integrations with the Ubuntu/GNOME desktop."""

from __future__ import annotations

import shutil
import subprocess

from belka import APP_NAME

NIGHT_LIGHT_SCHEMA = "org.gnome.settings-daemon.plugins.color"


def free_camera_from_desktop() -> bool:
    """Unmount cameras GNOME auto-mounted through gvfs so libgphoto2 can claim them."""
    if not shutil.which("gio"):
        return False
    try:
        subprocess.run(["gio", "mount", "-s", "gphoto2"], check=False, timeout=10, capture_output=True)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def night_light_enabled() -> bool | None:
    """GNOME Night Light tints the screen, i.e. the light source. None = unknown."""
    if not shutil.which("gsettings"):
        return None
    try:
        out = subprocess.run(
            ["gsettings", "get", NIGHT_LIGHT_SCHEMA, "night-light-enabled"],
            check=False, timeout=5, capture_output=True, text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = out.stdout.strip()
    return {"true": True, "false": False}.get(value)


def _color_property(name: str, value: bool | None = None) -> bool | None:
    """Read or write a property of GNOME's colour daemon over D-Bus."""
    if not shutil.which("gdbus"):
        return None
    base = ["gdbus", "call", "--session", "--dest", "org.gnome.SettingsDaemon.Color",
            "--object-path", "/org/gnome/SettingsDaemon/Color", "--method"]
    if value is None:
        cmd = base + ["org.freedesktop.DBus.Properties.Get", "org.gnome.SettingsDaemon.Color", name]
    else:
        cmd = base + ["org.freedesktop.DBus.Properties.Set", "org.gnome.SettingsDaemon.Color", name,
                      f"<{'true' if value else 'false'}>"]
    try:
        out = subprocess.run(cmd, check=False, timeout=5, capture_output=True, text=True)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    if value is not None:
        return True
    return "true" in out.stdout


def pause_night_light() -> bool:
    """Pause Night Light the way GNOME's quick-settings toggle does.

    ``DisabledUntilTomorrow`` lives in the daemon, not in dconf: GNOME lifts it
    by itself the next day, so a crash or a logout with Belka open can never
    leave Night Light switched off for good.
    """
    return bool(_color_property("DisabledUntilTomorrow", True))


def resume_night_light() -> bool:
    return bool(_color_property("DisabledUntilTomorrow", False))


def night_light_paused() -> bool:
    return bool(_color_property("DisabledUntilTomorrow"))


class IdleInhibitor:
    """Keep the screen from dimming or blanking while it is the light source.

    The inhibitor runs ``cat`` on a pipe Belka holds open: if Belka exits
    in any way, even killed, the pipe closes, ``cat`` ends and the inhibition
    with it. (``--inhibit-only`` would wait forever and outlive a crash.)
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, reason: str = "Pantalla en uso como fuente de luz") -> bool:
        if self.active:
            return True
        cat = shutil.which("cat") or "/bin/cat"
        if shutil.which("gnome-session-inhibit"):
            cmd = ["gnome-session-inhibit", "--app-id", APP_NAME, "--reason", reason, "--inhibit", "idle", cat]
        elif shutil.which("systemd-inhibit"):
            cmd = ["systemd-inhibit", "--what=idle", f"--who={APP_NAME}", f"--why={reason}", cat]
        else:
            return False
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self._proc = None
            return False
        return True

    def stop(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        self._proc = None
