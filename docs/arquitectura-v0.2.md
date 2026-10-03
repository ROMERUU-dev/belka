# Belka 0.2 — arquitectura del revelado estilo Lightroom

Contrato entre los módulos de la versión 0.2. El modelo de datos ya está fijado en
`belka/core/pipeline.py` (`DevelopSettings`) y `belka/core/session.py` (`Frame.snapshots`,
`Frame.flag`): **no cambiar nombres ni significados de campos**; si un módulo necesita algo más,
documentarlo en su reporte.

## Cadena de procesamiento

```
NEF ─► rawio (lineal cámara) ─► flat-field ─► orient (rotación 90°, espejos)
    ─► transform.lens_correct (distorsión, viñeteo de lente)        [sección "lens"]
    ─► transform.warp (enderezar `angle` + perspectiva `persp_*`)   [sección "transform"]
    ─► recorte (`crop` en coordenadas de la imagen transformada; o el fotograma detectado)
    ─► pipeline.render (inversión por densidad → curva de papel → sRGB codificado)
    ─► adjust.apply_adjustments (todo lo "Lightroom")                [sección por sección]
    ─► pantalla / exportación
```

El análisis de la película (`pipeline.analyze`: base, niveles, fotograma automático) sigue sobre la
imagen orientada **sin** transformar; el fotograma detectado se lleva a coordenadas transformadas con
`transform.map_rect_from_source`.

Unidades de la interfaz: los deslizadores muestran −100…100 (estilo Lightroom) y guardan −1…1 en el
modelo, salvo donde el campo diga otra cosa (grados, píxeles, `persp_scale`).

## `belka/core/adjust.py` (nuevo)

```python
HSL_BANDS: list[tuple[str, float]]       # (nombre en español, centro del tono en OKLCh, grados)
def is_identity(s: DevelopSettings) -> bool
def apply_adjustments(rgb: np.ndarray, s: DevelopSettings, scale: float = 1.0, seed: int = 0) -> np.ndarray
def curve_lut(points, size: int = 1024) -> np.ndarray      # spline monótona por los puntos, [0,1]→[0,1]
def parametric_lut(s: DevelopSettings, size: int = 1024) -> np.ndarray
def clipping_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]   # (sombras recortadas, luces recortadas)
```

* Entrada/salida: `float32` H×W×3 sRGB codificado en [0, 1]. Nunca modifica la entrada.
* `scale` = ancho de la imagen procesada / ancho a resolución completa; los radios (`sharpen_radius`,
  claridad, textura, grano) se definen en píxeles de resolución completa y se escalan con él, para
  que vista previa y exportación coincidan.
* Orden: altas luces/sombras → textura/claridad → intensidad/saturación → curva paramétrica → curva
  RGB → curvas R/G/B → HSL → gradación de color → reducción de ruido → enfoque → viñeta post-recorte
  → grano.
* Secciones en `s.disabled` se omiten: "curve", "hsl", "grading", "detail", "effects".
* Color en OKLab/OKLCh (convertir desde sRGB lineal). Con `is_identity` verdadero devuelve la entrada
  tal cual.
* Rendimiento: 1800×1200 con todo activo < 250 ms; se puede usar OpenCV (`cv2`).

## `belka/core/transform.py` (nuevo)

```python
def is_identity(s) -> bool                      # considera s.disabled ("transform", "lens")
def homography(s, width: int, height: int) -> np.ndarray   # 3×3: píxel de SALIDA → píxel de ORIGEN
def lens_correct(img, s) -> np.ndarray          # distorsión radial k1 y viñeteo, en coordenadas de origen
def warp(img, s, with_mask: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]  # mismo tamaño
def map_points_to_source(pts_norm, s, w, h) -> np.ndarray
def map_points_from_source(pts_norm, s, w, h) -> np.ndarray
def map_rect_from_source(rect_norm, s, w, h, aspect="frame") -> tuple | None   # rectángulo más grande inscrito en el fotograma transformado (sin borde de película)
def largest_valid_rect(s, w, h, aspect: float | None = None) -> tuple[float, float, float, float]
def auto_upright(img_linear, mode: str) -> dict  # mode: "level", "vertical", "auto", "full" → {angle, persp_vertical, persp_horizontal}
def angle_from_line(p0, p1) -> float            # grados para dejar la línea horizontal o vertical (la más cercana)
```

* `angle` positivo gira en sentido horario la imagen mostrada. `persp_vertical` positivo corrige
  verticales que convergen hacia arriba (fotografía hecha desde abajo), como Lightroom.
* Fuera del área válida, `warp` pone negro y la máscara vale 0.

## `belka/ui/icons.py` y `belka/data/icons/*.svg`

`icon(name) -> QIcon`. SVG 24×24, trazo de 1.5 px, puntas redondas, `stroke="currentColor"`,
sin rellenos de color fijos. Nombres: crop, straighten, rotate_left, rotate_right, flip_horizontal,
flip_vertical, perspective, eyedropper_base, eyedropper_wb, compare, compare_split, zoom_fit,
zoom_100, negative, clipping, grid_overlay, undo, redo, reset, eye, eye_off, camera, shutter, light,
flat_field, tint, liveview, focus_near, focus_far, autofocus, import, export, new_roll, open_roll,
library, develop, capture, star, star_filled, flag, reject, copy_settings, paste_settings, sync,
snapshot, history, histogram, settings, help, lock, unlock, chevron_down, chevron_right, plus, minus,
close, check, upright_auto, upright_level, upright_vertical, upright_full, aspect_lock, info.

## `belka/ui/histogram.py` (nuevo)

```python
class InteractiveHistogram(QWidget):
    adjustRequested = Signal(str, float)   # (campo, delta): "black", "shadows", "exposure", "highlights", "white"
    dragFinished = Signal(str)             # campo, al soltar: cierra un paso del historial
    clippingToggled = Signal(str, bool)    # "shadows" | "highlights"
    def set_histogram(self, hist: np.ndarray | None)   # (3, 256) cuentas de sRGB 8 bits
    def set_values(self, values: dict[str, float])     # valores actuales en unidades del modelo
    def set_readout(self, text: str)                   # p. ej. "R 52 %  G 48 %  B 41 %"
    def set_info(self, text: str)                      # p. ej. "ISO 100  ·  f/8  ·  1/2 s"
    def set_clipping(self, shadows: bool, highlights: bool)
```

Zonas como Lightroom: negros 0–10 %, sombras 10–30 %, exposición 30–70 %, altas luces 70–90 %,
blancos 90–100 % del ancho. Al pasar el ratón se ilumina la zona y se muestra su nombre y valor;
arrastrar en horizontal emite `adjustRequested`. Triángulos de recorte en las esquinas superiores.

## `belka/ui/sections.py`, `belka/ui/curve_editor.py`, `belka/ui/develop_panel.py`

Panel derecho estilo Lightroom Classic: secciones plegables con interruptor de encendido y
restablecer (doble clic en el título), deslizadores con pista de color donde ayuda (temperatura,
tinte, tonos HSL). Secciones: Perfil de película (Belka) · Básico · Curva de tonos · HSL / Color ·
Gradación de color · Detalle · Óptica · Transformar · Efectos.

API pública que usa la ventana principal (mantenerla):

```python
class DevelopPanel(QScrollArea):
    settingsChanged = Signal(object)      # DevelopSettings en cada cambio (arrastres incluidos)
    editCommitted = Signal(str)           # etiqueta para el historial al soltar/confirmar ("Exposición +0,35")
    toolRequested = Signal(str)           # "base" | "neutral"
    uprightRequested = Signal(str)        # "off" | "auto" | "level" | "vertical" | "full"
    lockBaseChanged = Signal(bool)
    applyAllRequested = Signal(); saveProfileRequested = Signal(); resetRequested = Signal()
    settings: DevelopSettings (propiedad)
    def load(self, settings, lock_base: bool) -> None
    def apply_external(self, **changes) -> None
    def show_analysis(self, analysis) -> None
    def set_exposure_note(self, text: str) -> None
    def set_histogram(self, hist) -> None         # alimenta el fondo del editor de curvas
    def current_profile(self) -> FilmProfile
    def reload_profiles(self) -> None
```

## `belka/ui/image_view.py` (ampliado)

Herramientas: "none", "base", "neutral", "crop", "straighten". Recorte con asas, cuadrícula de
tercios, relación de aspecto bloqueable, giro arrastrando fuera del recuadro, línea de enderezar,
Enter confirma y Esc cancela. Comparar antes/después (división arrastrable o lado a lado). Capa de
recorte de tonos (sombras azules, luces rojas). Señales de posición del cursor (lectura RGB) y del
área visible (navegador).

## `belka/core/history.py` y `belka/ui/left_panels.py`

Historial por fotograma con deshacer/rehacer y fusión de pasos del mismo control dentro de 1 s;
paneles Navegador, Perfiles de película (buscador), Instantáneas e Historial.

## Integración (sesión principal)

Ventana con selector de módulos (Biblioteca · Captura · Revelado), barra de herramientas vertical
con íconos a la izquierda de la imagen, panel derecho con histograma + secciones, tira de fotogramas
con estrellas y banderas, atajos de Lightroom (R recortar, W balance de blancos, \ antes/después,
J recorte de tonos, Z zoom, Ctrl+Z, Ctrl+Mayús+C/V copiar/pegar ajustes), y el procesamiento por
etapas con caché en `belka/ui/processing.py`.

## Ampliación 0.2.1 (petición del usuario del 2026-10-02)

Modelo: `DevelopSettings.upright_mode` ("", "auto", "level", "vertical", "full", "guided") y
`DevelopSettings.upright_guides` (hasta 4 guías `(x0, y0, x1, y1)` normalizadas en la imagen orientada,
ANTES de la corrección de lente y el warp: las mismas coordenadas "de origen" que `map_points_to_source`).

### `belka/core/autotone.py` (nuevo)

```python
TONE_FIELDS = ("exposure", "contrast", "highlights", "shadows", "white", "black")
def auto_tone(geo, analysis, settings, profile, camera_matrix, rgb_sequential, fields=None) -> dict[str, float]
```

`geo`: cámara lineal ya orientada, transformada y recortada (cualquier tamaño; se reduce dentro). Devuelve
valores ABSOLUTOS en unidades del modelo para `fields` (por defecto todos), como el "Automático" de Tono de
Lightroom: exposición para un tono medio agradable, blancos/negros justo al borde del recorte, altas luces y
sombras para recuperar extremos, contraste moderado. Determinista y < 150 ms.

### `belka/core/transform.py` (añadido)

```python
def guided_upright(guides, s, w, h) -> dict   # {angle, persp_vertical, persp_horizontal}; {} si no hay guías válidas
```

1 guía: solo rotación. 2+ casi verticales: keystone vertical. 2+ casi horizontales: keystone horizontal.
Cada guía se clasifica por su orientación tras la transformación actual. Supone persp_rotate = persp_aspect = 0.

### `belka/ui/image_view.py` (añadido)

* Herramienta `"guided"`: trazar hasta 4 guías, mover sus extremos (con lupa), quitar una con clic derecho o
  Supr. `guidesChanged(list)` con `(x0, y0, x1, y1)` normalizadas en la imagen MOSTRADA; `set_guides(list)`.
* `set_grid(mode)`: "off" | "grid" (cuadrícula fina, como al mover un deslizador de Transformar en Lightroom).
* `contextMenuRequested(QPoint global, float x, float y)`: clic derecho que ninguna herramienta usa.

### `belka/ui/develop_panel.py` (añadido)

`autoToneRequested()` (botón "Auto" en Tono), `autoFieldRequested(str)` (Mayús+doble clic en un deslizador
de Tono), `uprightRequested("guided")`, botones Upright como grupo exclusivo que refleja `upright_mode`,
`gridToggled(bool)` ("Mostrar cuadrícula" en Transformar) y `transformDragging(bool)` al pulsar/soltar un
deslizador de Transformar u Óptica.

### Cámaras

Sin cámara simulada. `belka/data/cameras.json`: por marca, los nombres de los controles de libgphoto2 y
notas de conexión (modo USB que hay que elegir en la cámara); avisos por modelo.
