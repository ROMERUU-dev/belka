# Belka en otras plataformas — pronóstico (2026-10-02)

Seis instaladores: Linux, Windows y macOS, cada uno en x86-64 y ARM64. Disponibilidad de ruedas binarias
consultada en PyPI el 2026-10-02 (las versiones que usa Belka):

| Paquete | Linux x64 | Linux ARM64 | Windows x64 | Windows ARM64 | macOS Intel | macOS ARM |
|---|---|---|---|---|---|---|
| PySide6-Essentials 6.11.2 | sí | sí | sí | sí | sí | sí |
| numpy 2.5.3 | sí | sí | sí | sí | sí | sí |
| tifffile | sí (puro Python) | sí | sí | sí | sí | sí |
| opencv-python-headless 5.0.0.93 | sí | sí | sí | **no** | sí (macOS 14+) | sí |
| rawpy 0.27.1 (LibRaw) | sí | sí | sí | sí | **no** (desde 0.9) | sí |
| gphoto2 2.6.4 (libgphoto2) | sí | sí | **no** | **no** | sí (macOS 15+) | sí (macOS 14+) |

## Qué es portable y qué no

* **Portable tal cual:** revelado, inversión, perfiles, rollos, exportación, la interfaz Qt y el panel de luz
  (una ventana a pantalla completa en la pantalla elegida; la calibración en mm usa el tamaño físico que da Qt).
* **Específico de GNOME/Linux** (`belka/system.py`, `belka/camera/gphoto.py`): pausar la luz nocturna
  (gdbus), impedir la suspensión (`gnome-session-inhibit`), liberar la cámara montada por gvfs. Cada sistema
  necesita su equivalente: Windows `SetThreadExecutionState` y aviso de "Luz nocturna"; macOS `caffeinate` /
  IOPMAssertion, aviso de Night Shift/True Tone y liberar la cámara que retiene `ptpcamerad`.
* **Rutas:** hoy siguen XDG; pasar a `QStandardPaths` para %APPDATA% y ~/Library.
* **Control de la cámara en Windows: el obstáculo real.** libgphoto2 no tiene versión oficial para Windows;
  funciona con MSYS2 + libusb, pero exige cambiar el driver USB de la cámara con Zadig (rompe la importación de
  fotos de Windows), inaceptable para usuarios. Alternativas, de menos a más trabajo:
  1. **Carpeta vigilada:** el software de la marca (Nikon NX Tether, Canon EOS Utility, Sony Imaging Edge,
     Fujifilm X Acquire) dispara y guarda en una carpeta; Belka importa e invierte al instante y sigue
     manejando el panel de luz. Sirve también en Windows ARM.
  2. **PTP nativo por WPD** (Windows Portable Devices permite comandos PTP/MTP propios del fabricante): un
     backend propio sin drivers extra; hay que reimplementar por marca lo que hace libgphoto2.
  3. **SDK oficiales** (Nikon SDK, Canon EDSDK, Sony Camera Remote SDK, Fujifilm X SDK): registro con cada
     marca, licencias distintas, casi siempre solo x64.

## Pronóstico por instalador

| Instalador | Viabilidad | Trabajo | Notas |
|---|---|---|---|
| Linux x64 (.deb / AppImage) | segura | 1–2 días | hoy hay un script; falta el paquete |
| Linux ARM64 (.deb / AppImage) | alta | 2–3 días | todas las ruedas existen; probar en Raspberry Pi 5 (revelar será ~3× más lento) |
| macOS Apple Silicon (.dmg) | alta | 1–2 semanas | capa de sistema + firma y notarización (cuenta Apple Developer, 99 USD/año) |
| macOS Intel (.dmg) | media | 2–3 semanas | compilar rawpy/LibRaw en CI; solo Macs con macOS 14–15+ (2018–2020); Apple deja Intel tras macOS 26 |
| Windows x64 (instalador .exe/MSIX) | media | 1 semana con carpeta vigilada; 1–3 meses con control de cámara | sin certificado de firma, SmartScreen avisa |
| Windows ARM64 | baja–media | +2 semanas sobre x64 | sin OpenCV para ARM64 (compilarlo) o usar la versión x64 emulada (Windows 11 ARM la ejecuta); SDK de cámara solo x64 |

## Cómo se construirían

PyInstaller (o Briefcase) en GitHub Actions, un trabajo por plataforma en runners nativos: `ubuntu-24.04`,
`ubuntu-24.04-arm`, `windows-latest`, `windows-11-arm`, `macos-14` (ARM) y un runner Intel de macOS mientras
GitHub los siga ofreciendo. macOS no puede ser "universal2" porque rawpy no publica Intel.

Orden recomendado: Linux x64 y ARM64 → macOS ARM → Windows x64 con carpeta vigilada → Mac Intel y Windows ARM
como secundarios → control de cámara nativo en Windows.
