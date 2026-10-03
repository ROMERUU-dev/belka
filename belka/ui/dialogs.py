"""Dialogs: new roll, export, save profile, help."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from belka import paths
from belka.core.export import ExportOptions, export_frame
from belka.core.film import ProfileLibrary, film_name
from belka.core.session import Frame, Session
from belka.i18n import _


class FolderField(QWidget):
    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(str(path))
        browse = QPushButton(_("Elegir…"))
        browse.clicked.connect(self._browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(browse)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, _("Carpeta"), self.edit.text())
        if folder:
            self.edit.setText(folder)

    def path(self) -> Path:
        return Path(self.edit.text()).expanduser()


def profile_combo(library: ProfileLibrary, selected: str) -> QComboBox:
    combo = QComboBox()
    for p in library.all():
        combo.addItem(film_name(p), p.id)
    combo.setCurrentIndex(max(0, combo.findData(selected)))
    return combo


class NewRollDialog(QDialog):
    def __init__(self, library: ProfileLibrary, root: Path, profile_id: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Nuevo rollo"))
        form = QFormLayout(self)
        self.name = QLineEdit()
        self.name.setPlaceholderText(_("p. ej. Portra Oaxaca"))
        self.profile = profile_combo(library, profile_id)
        self.folder = FolderField(root)
        form.addRow(_("Nombre"), self.name)
        form.addRow(_("Película"), self.profile)
        form.addRow(_("Guardar en"), self.folder)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.resize(480, 0)

    def values(self) -> tuple[str, str, Path]:
        return self.name.text().strip() or _("Rollo"), self.profile.currentData(), self.folder.path()


class ExportDialog(QDialog):
    def __init__(self, session: Session, selected: int, total: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Exportar positivos"))
        form = QFormLayout(self)
        self.scope = QComboBox()
        if selected:
            self.scope.addItem(_("Seleccionados ({n})").format(n=selected), "selected")
        self.scope.addItem(_("Todo el rollo ({n})").format(n=total), "all")
        self.tiff = QCheckBox(_("TIFF 16 bits (máxima calidad, para editar)"))
        self.tiff.setChecked(True)
        self.jpeg = QCheckBox(_("JPEG"))
        self.jpeg.setChecked(True)
        self.quality = QSpinBox()
        self.quality.setRange(60, 100)
        self.quality.setValue(95)
        self.size = QComboBox()
        self.size.addItem(_("Resolución completa"), None)
        self.size.addItem("4000 px", 4000)
        self.size.addItem("2048 px", 2048)
        self.folder = FolderField(session.export_dir)
        form.addRow(_("Fotogramas"), self.scope)
        form.addRow(_("Formatos"), self.tiff)
        form.addRow("", self.jpeg)
        form.addRow(_("Calidad JPEG"), self.quality)
        form.addRow(_("Tamaño"), self.size)
        form.addRow(_("Carpeta"), self.folder)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.resize(520, 0)

    def options(self) -> tuple[str, ExportOptions]:
        formats = tuple(f for f, box in (("tiff16", self.tiff), ("jpeg", self.jpeg)) if box.isChecked())
        return self.scope.currentData(), ExportOptions(
            formats=formats or ("jpeg",), jpeg_quality=self.quality.value(), max_side=self.size.currentData(),
            folder=self.folder.path(),
        )


class ExportWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(list, list)  # written paths, errors

    def __init__(self, session: Session, frames: list[Frame], library: ProfileLibrary, options: ExportOptions):
        super().__init__()
        self.session, self.frames, self.library, self.options = session, frames, library, options
        self.cancelled = False

    @Slot()
    def run(self) -> None:
        written, errors = [], []
        for i, frame in enumerate(self.frames):
            if self.cancelled:
                break
            self.progress.emit(i, len(self.frames), frame.label)
            try:
                written += export_frame(self.session, frame, self.library, self.options)
            except Exception as exc:
                errors.append(f"{frame.label}: {exc}")
        self.finished.emit(written, errors)


class ExportRunner(QObject):
    progress = Signal(int, int, str)
    finished = Signal(list, list)

    def __init__(self, session: Session, frames: list[Frame], library: ProfileLibrary, options: ExportOptions, parent=None):
        super().__init__(parent)
        self.thread = QThread(self)
        self.worker = ExportWorker(session, frames, library, options)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress)
        self.worker.finished.connect(self._done)

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> None:
        self.worker.cancelled = True

    def _done(self, written: list, errors: list) -> None:
        self.thread.quit()
        self.thread.wait()
        self.finished.emit(written, errors)


class SaveProfileDialog(QDialog):
    def __init__(self, base_name: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Guardar perfil de película"))
        layout = QVBoxLayout(self)
        info = QLabel(_(
            "Guarda cómo responde esta película con tu cámara y tu luz: las pendientes por canal que "
            "midió el auto-balance, la separación, el contraste, la saturación y la temperatura. "
            "Úsalo después con auto-balance bajo para que todo el rollo quede consistente."
        ))
        info.setWordWrap(True)
        layout.addWidget(info)
        form = QFormLayout()
        self.name = QLineEdit(_("{name} (mi cámara)").format(name=base_name))
        form.addRow(_("Nombre"), self.name)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(460, 0)


HELP_ES = """
<h2>Cómo digitalizar con Belka</h2>
<h3>Montaje</h3>
<ol>
<li>Pon la pantalla horizontal (una laptop abierta a 180° o un monitor acostado) y la cámara encima en un
estativo o tripié, con un objetivo macro, perpendicular a la pantalla.</li>
<li>Coloca la película a <b>5–10 mm sobre la pantalla</b> con un difusor (acrílico opalino o papel
vegetal) entre ambas: así no se ve la rejilla de píxeles. Para eso servirá el adaptador impreso en 3D.</li>
<li>Brillo de la pantalla al máximo. Desactiva la <b>luz nocturna</b> de GNOME: tiñe la luz.</li>
<li>Usa velocidades de <b>1/30 s o más lentas</b>: algunas pantallas parpadean (PWM) y a velocidades
rápidas aparecen bandas.</li>
</ol>
<h3>Flujo</h3>
<ol>
<li><b>Archivo → Nuevo rollo</b> y elige la película.</li>
<li>Conecta la cámara por USB y pulsa <b>Conectar</b>. Si GNOME la abrió como disco, Belka la libera.</li>
<li><b>Encender luz</b>: en la pantalla elegida aparece la zona de luz del formato (35 mm, 120…).
Muévela con el ratón o las flechas para que quede bajo la película.</li>
<li>Opcional: <b>flat-field</b> (F dos veces, sin película) y <b>calibrar tinte</b> (T, con película):
la luz se vuelve azul-cian para que la máscara naranja no desperdicie el rango del sensor.</li>
<li>Activa la vista en vivo, enfoca con la ampliación 4–8× y captura con <b>Espacio</b>.</li>
<li>Ajusta el revelado a la derecha; <b>Aplicar al rollo</b> copia los ajustes y <b>Exportar</b> guarda
TIFF de 16 bits y JPEG.</li>
</ol>
<h3>Modo RGB secuencial</h3>
<p>Toma tres fotos por fotograma con luz roja, verde y azul puras y combina un canal de cada una. Separa
mejor los colores que la luz blanca (como un escáner), a cambio de tres disparos. La película y la cámara
no deben moverse entre tomas.</p>
<h3>Revelado (como Lightroom Classic)</h3>
<ul>
<li><b>Módulos</b> arriba a la derecha: Biblioteca (cuadrícula del rollo), Captura (cámara y luz) y Revelado.</li>
<li><b>Barra de herramientas</b> a la izquierda de la foto: recortar y enderezar, línea de nivel, Upright,
perspectiva, rotar y voltear, cuentagotas de base y de balance de blancos, antes/después, recorte de tonos y zoom.</li>
<li><b>Histograma</b>: arrastra en horizontal sobre sus zonas (negros, sombras, exposición, altas luces, blancos)
para ajustarlas. Los triángulos de las esquinas muestran el recorte de tonos.</li>
<li><b>Paneles</b> a la derecha: Perfil de película, Básico, Curva de tonos, HSL / Color, Gradación de color,
Detalle, Óptica, Transformar y Efectos. El interruptor de cada panel lo apaga; doble clic en su título o en un
deslizador lo restablece.</li>
<li>A la izquierda: Navegador, Perfiles de película, Instantáneas e Historial (clic en un paso para volver a él).</li>
<li><b>Recortar</b>: arrastra esquinas o lados; arrastra fuera del recuadro para girar; X cambia la orientación;
Enter aplica y Esc cancela. Con la perspectiva corregida, <i>Restringir recorte</i> evita esquinas vacías.</li>
</ul>
<h3>Atajos</h3>
<p><b>Módulos:</b> G Biblioteca · Ctrl+Alt+2 Captura · D Revelado<br>
<b>Herramientas:</b> R recortar · S enderezar con una línea · Mayús+U Upright · Mayús+T Upright guiado (reglas) ·
Ctrl+[ / Ctrl+] rotar · Mayús+H / Mayús+V voltear · B medir base · W balance de blancos · Ctrl+U tono automático ·
clic derecho: menú de la foto<br>
<b>Vista:</b> N negativo · \\ antes/después · Y lado a lado · J recorte de tonos · Ctrl+Alt+O cuadrícula ·
Z o Espacio 1:1 / ajustar · Ctrl+0 ajustar ·
Tab ocultar paneles · F6 tira · F7 / F8 paneles<br>
<b>Ajustes:</b> Ctrl+Z deshacer · Ctrl+Mayús+Z rehacer · Ctrl+Mayús+C / Ctrl+Mayús+V copiar / pegar ajustes ·
Ctrl+Alt+V ajustes del anterior · Ctrl+Mayús+S sincronizar con la selección · Ctrl+Mayús+R restablecer ·
Ctrl+N instantánea<br>
<b>Fotogramas:</b> ←/→ anterior / siguiente · 0–5 estrellas · P elegida · X rechazada · U sin marca ·
Supr quitar del rollo o eliminar del disco (a la papelera) · Ctrl+Retroceso eliminar las rechazadas<br>
<b>Rollo y captura:</b> Ctrl+Mayús+N nuevo rollo · Ctrl+O abrir · Ctrl+Mayús+I importar · Ctrl+Mayús+E exportar ·
L panel de luz · Espacio capturar · F1 esta guía</p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Guía rápida"))
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        from belka.i18n import language

        if language() == "en":
            from belka.translations.en import HELP_HTML

            browser.setHtml(HELP_HTML)
        else:
            browser.setHtml(HELP_ES)
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(640, 680)


def default_root() -> Path:
    return paths.default_sessions_dir()
