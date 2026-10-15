"""Portanegativos Belka 35 · v2 (retro): cuerpo en X con cuatro brazos que se
abren como una llave y se curvan hacia arriba, una capucha central con la
ventana escalonada y un riel con cajón para el difusor.

Cinco piezas, de abajo arriba (Z = 0 es la superficie de la pantalla):

* base: se apoya en la pantalla con la silueta del cuerpo. Tiene un riel abierto
  por un costado (+Y) donde entra el cajón del difusor, y cuatro pernos.
* cajón: placa con un aro elevado alrededor de la ventana. La hoja de difusor
  (papel de horno, cebolla o vegetal) se pone encima del aro y el
* aro: marco que baja por fuera del aro y la deja tensa, como un bastidor de
  bordado. El cajón entra y sale del riel sin desarmar nada.
* cuerpo: canal de 35 mm de punta a punta, 7 mm por encima del difusor. En cada
  extremo la tira corre entre dos brazos que suben en curva.
* capucha: el centro elevado con la ventana escalonada; aplana la tira con cuatro
  imanes de 6×2 mm.

La ventana deja ver la película de orilla a orilla (perforaciones y código DX
con luz) y los huecos entre fotogramas. La pantalla ilumina LIGHT_ALONG ×
LIGHT_ACROSS (preajuste «Portanegativos Belka 35», el mismo que la v1).

Uso: freecadcmd portanegativos_35_v2.py (con el AppImage: FreeCAD.AppImage freecadcmd …).
Escribe stl/*.stl orientadas para imprimir, portanegativos_35_v2.FCStd e img/.
"""

import math
import os

import FreeCAD as App
import MeshPart
import Part

V = App.Vector
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ película y luz
FILM_W = 35.0
CHANNEL_W = 35.4  # canal de la tira: película + holgura
CHANNEL_DEPTH = 0.4  # la capucha queda esto por encima de la película (0,13–0,15 mm)
WIN_ALONG, WIN_ACROSS = 42.0, 37.0  # ventana: fotograma de 36 mm + huecos, y todo el ancho de la tira
LIGHT_ALONG, LIGHT_ACROSS = 52.0, 46.0  # zona iluminada en la pantalla (preajuste de Belka)
MARK_GAP = 2.0  # Belka dibuja las marcas de esquina 2 mm fuera de la zona iluminada
BASE_WIN = (LIGHT_ALONG + 2 * MARK_GAP, LIGHT_ACROSS + 2 * MARK_GAP)

# ------------------------------------------------------------------ silueta (X a lo largo de la tira)
CENTRE = (76.0, 74.0)  # bloque central
OUTLINE_R = 4.0
ARM_ROOT_X, ARM_TIP_X = 38.0, 84.0  # cada brazo, del bloque central a la punta
ARM_INNER = 21.0  # los brazos dejan libre el paso de la tira (|y| < ARM_INNER)
ARM_ROOT_W, ARM_TIP_W = 16.0, 30.0  # el brazo se abre hacia la punta, como una llave
ARM_TIP_R = 7.0
ARM_BITE = (47.0, 9.0)  # mordida cóncava en el costado de cada brazo, junto al bloque: x del centro, radio
ARM_RISE = 11.0  # la punta del brazo queda tanto más arriba que el bloque central
DECK_W = 2 * ARM_INNER + 2.0  # entre los brazos, la tira corre sobre una cubierta

# ------------------------------------------------------------------ base y cajón
FLOOR_T = 2.0  # pantalla → cajón
SLOT_H = 2.9  # alto del riel
SLOT_CLEAR = 0.3  # holgura del cajón en el riel, por lado
BASE_H = FLOOR_T + SLOT_H
TRAY_PLATE = (68.0, 70.0)  # placa del cajón
TRAY_PLATE_T = 1.0
RING_WIN = (54.0, 48.0)  # ventana del aro: más que la zona iluminada
RING_WALL, RING_H = 1.6, 1.4
HOOP_GAP = 0.25  # entre aro y marco, por lado: dos veces el papel + ajuste
HOOP_OUT = (66.0, 68.0)
TAB = (24.0, 10.0)  # lengüeta para sacar el cajón (+Y)
PAPER = (66.0, 62.0)  # hoja sugerida

# ------------------------------------------------------------------ alturas
DIFFUSER_Z = FLOOR_T + TRAY_PLATE_T + RING_H  # el papel sobre el aro
FILM_ABOVE_DIFFUSER = 7.0  # difusor → película: desenfoca la textura del difusor
FILM_Z = DIFFUSER_Z + FILM_ABOVE_DIFFUSER
HOLDER_Z = BASE_H
HOLDER_TOP = FILM_Z + CHANNEL_DEPTH
TUNNEL_FLARE = 4.0  # la ventana se abre 45° hacia el difusor
LEAD_IN = (8.0, 40.0)  # boca de entrada de la tira: largo y ancho

# ------------------------------------------------------------------ capucha
CAP_INSET = 2.0
CAP_H = 9.0
CAP_STEPS, CAP_STEP = 4, 2.5  # escalones de la ventana, cada uno CAP_STEP más ancho por lado

# ------------------------------------------------------------------ uniones
PIN_D, PIN_CLEAR, PIN_INTO = 3.0, 0.35, 2.5
PINS = [(sx * 66.0, sy * (ARM_INNER + 8.0)) for sx in (-1, 1) for sy in (-1, 1)]
MAGNET_D, MAGNET_H = 6.2, 2.1  # imanes de 6×2 mm
MAGNETS = [(sx * 29.0, sy * 28.0) for sx in (-1, 1) for sy in (-1, 1)]
CAP_PINS = [(0.0, 29.0), (0.0, -29.0)]
CAP_PIN_D, CAP_PIN_H = 3.0, 2.0


def centred_box(w, d, h, z=0.0, x=0.0, y=0.0):
    return Part.makeBox(w, d, h, V(x - w / 2, y - d / 2, z))


def fuse_all(shapes):
    out = shapes[0]
    for s in shapes[1:]:
        out = out.fuse(s)
    return out.removeSplitter()


def vertical_fillet(shape, r, near=None):
    """Redondea las aristas verticales (todas, o las que pasan cerca de ``near``)."""
    edges = []
    for e in shape.Edges:
        a, b = (v.Point for v in e.Vertexes) if len(e.Vertexes) == 2 else (None, None)
        if a is None or abs(a.z - b.z) < 1e-6 or abs(a.x - b.x) > 1e-6 or abs(a.y - b.y) > 1e-6:
            continue
        if near is None or any(math.hypot(a.x - x, a.y - y) < 1.0 for x, y in near):
            edges.append(e)
    try:
        return shape.makeFillet(r, edges) if edges else shape
    except Part.OCCError:
        return shape


def rounded_box(w, d, h, r, z=0.0):
    return vertical_fillet(centred_box(w, d, h, z), r)


def arm_outer(x):
    """Borde de afuera del brazo: se abre hacia la punta en una curva."""
    t = (x - ARM_ROOT_X) / (ARM_TIP_X - ARM_ROOT_X)
    return ARM_INNER + ARM_ROOT_W + (ARM_TIP_W - ARM_ROOT_W) * max(t, 0.0) ** 1.6


def arm_top(x):
    """Lo alto del brazo: sube en curva del bloque central a la punta."""
    t = min(max((abs(x) - ARM_ROOT_X) / (ARM_TIP_X - ARM_ROOT_X), 0.0), 1.0)
    return HOLDER_TOP + ARM_RISE * t * t


def arm_plan(h, z):
    """Un brazo (+X, +Y), prisma de alto h desde z."""
    x0 = ARM_ROOT_X - 4.0  # entra en el bloque central
    xs = [x0 + (ARM_TIP_X - x0) * i / 24 for i in range(25)]
    pts = [V(x0, ARM_INNER, z)] + [V(x, ARM_INNER, z) for x in (ARM_TIP_X,)]
    pts += [V(x, arm_outer(x), z) for x in reversed(xs)]
    face = Part.Face(Part.makePolygon(pts + [pts[0]]))
    arm = face.extrude(V(0, 0, h))
    arm = vertical_fillet(arm, ARM_TIP_R, near=[(ARM_TIP_X, ARM_INNER), (ARM_TIP_X, arm_outer(ARM_TIP_X))])
    bite_x, bite_r = ARM_BITE
    return arm.cut(Part.makeCylinder(bite_r, h, V(bite_x, arm_outer(bite_x) + 0.55 * bite_r, z)))


def mirrored4(shape):
    out = [shape]
    for normal in (V(1, 0, 0), V(0, 1, 0)):
        out += [s.mirror(V(0, 0, 0), normal) for s in out]
    return out


def plan(h, z=0.0):
    """La silueta completa: bloque central, cubierta y los cuatro brazos."""
    parts = [rounded_box(*CENTRE, h, OUTLINE_R, z), centred_box(2 * ARM_TIP_X, DECK_W, h, z)]
    parts += mirrored4(arm_plan(h, z))
    return fuse_all(parts)


def smile_cutter():
    """Lo que queda por encima de la curva de los brazos (para quitarlo)."""
    xs = [ARM_ROOT_X - 6 + i * (ARM_TIP_X + 8 - ARM_ROOT_X + 6) / 60 for i in range(61)]
    top = HOLDER_TOP + ARM_RISE + 20
    pts = [V(x, -100, arm_top(x)) for x in xs] + [V(xs[-1], -100, top), V(xs[0], -100, top)]
    right = Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(V(0, 200, 0))
    return right.fuse(right.mirror(V(0, 0, 0), V(1, 0, 0)))


def frustum(w0, d0, w1, d1, z0, z1):
    """Prisma de sección rectangular que pasa de w0×d0 en z0 a w1×d1 en z1."""
    def rect(w, d, z):
        pts = [V(-w / 2, -d / 2, z), V(w / 2, -d / 2, z), V(w / 2, d / 2, z), V(-w / 2, d / 2, z)]
        return Part.makePolygon(pts + [pts[0]])
    return Part.makeLoft([rect(w0, d0, z0), rect(w1, d1, z1)], True)


def mouth(x0, x1, w0, w1, z, h):
    """Corte que ensancha el canal de w0 (en x0) a w1 (en x1): la boca de entrada."""
    def rect(x, w):
        pts = [V(x, -w / 2, z), V(x, w / 2, z), V(x, w / 2, z + h), V(x, -w / 2, z + h)]
        return Part.makePolygon(pts + [pts[0]])
    return Part.makeLoft([rect(x0, w0), rect(x1, w1)], True)


def cylinders(points, d, h, z):
    return fuse_all([Part.makeCylinder(d / 2, h, V(x, y, z)) for x, y in points])


def make_base():
    base = plan(BASE_H)
    base = base.cut(centred_box(*BASE_WIN, FLOOR_T))
    slot_w = TRAY_PLATE[0] + 2 * SLOT_CLEAR
    slot_y0 = -TRAY_PLATE[1] / 2 - SLOT_CLEAR  # tope del cajón; abierto hacia +Y
    slot_len = CENTRE[1] / 2 - slot_y0 + 1.0
    base = base.cut(Part.makeBox(slot_w, slot_len, SLOT_H + 0.01, V(-slot_w / 2, slot_y0, FLOOR_T)))
    return base.fuse(cylinders(PINS, PIN_D, PIN_INTO, BASE_H)).removeSplitter()


def make_tray():
    plate = rounded_box(*TRAY_PLATE, TRAY_PLATE_T, 3.0, FLOOR_T)
    # Desde el borde del aro (que se monta encima) hasta TAB[1] fuera de la placa.
    y0, y1 = HOOP_OUT[1] / 2 + 0.3, TRAY_PLATE[1] / 2 + TAB[1]
    tab = vertical_fillet(centred_box(TAB[0], y1 - y0, TRAY_PLATE_T + RING_H, FLOOR_T, y=(y0 + y1) / 2), 3.0)
    grip = fuse_all([centred_box(TAB[0] - 6, 0.8, 0.5, FLOOR_T + TRAY_PLATE_T + RING_H - 0.5,
                                 y=TRAY_PLATE[1] / 2 + 2.5 + 2.0 * i) for i in range(3)])
    ring_out = (RING_WIN[0] + 2 * RING_WALL, RING_WIN[1] + 2 * RING_WALL)
    ring = centred_box(*ring_out, RING_H, FLOOR_T + TRAY_PLATE_T).cut(
        centred_box(*RING_WIN, RING_H, FLOOR_T + TRAY_PLATE_T))
    tray = fuse_all([plate, tab, ring]).cut(grip)
    return tray.cut(centred_box(*RING_WIN, TRAY_PLATE_T, FLOOR_T))


def make_hoop():
    ring_out = (RING_WIN[0] + 2 * RING_WALL, RING_WIN[1] + 2 * RING_WALL)
    hoop = rounded_box(*HOOP_OUT, RING_H, 3.0, FLOOR_T + TRAY_PLATE_T)
    opening = (ring_out[0] + 2 * HOOP_GAP, ring_out[1] + 2 * HOOP_GAP)
    hoop = hoop.cut(centred_box(*opening, RING_H, FLOOR_T + TRAY_PLATE_T))
    # Dos muescas para hacer palanca al desarmarlo.
    pry = fuse_all([centred_box(6.0, 1.2, 0.6, FLOOR_T + TRAY_PLATE_T, x=sx * 20.0, y=HOOP_OUT[1] / 2 - 0.6)
                    for sx in (-1, 1)])
    return hoop.cut(pry)


def make_body():
    tall = HOLDER_TOP + ARM_RISE + 2 - HOLDER_Z
    arms = fuse_all(mirrored4(arm_plan(tall, HOLDER_Z))).cut(smile_cutter())
    deck = centred_box(2 * ARM_TIP_X, DECK_W, HOLDER_TOP - HOLDER_Z, HOLDER_Z)
    centre = rounded_box(*CENTRE, HOLDER_TOP - HOLDER_Z, OUTLINE_R, HOLDER_Z)
    body = fuse_all([centre, deck, arms])
    length = 2 * ARM_TIP_X + 2
    body = body.cut(centred_box(length, CHANNEL_W, CHANNEL_DEPTH, FILM_Z))
    for sx in (-1, 1):
        body = body.cut(mouth(sx * (ARM_TIP_X - LEAD_IN[0]), sx * (ARM_TIP_X + 0.5), CHANNEL_W, LEAD_IN[1],
                              FILM_Z, CHANNEL_DEPTH))
    body = body.cut(frustum(WIN_ALONG + 2 * TUNNEL_FLARE, WIN_ACROSS + 2 * TUNNEL_FLARE, WIN_ALONG, WIN_ACROSS,
                            HOLDER_Z, FILM_Z))
    body = body.cut(centred_box(WIN_ALONG, WIN_ACROSS, CHANNEL_DEPTH, FILM_Z))
    body = body.cut(cylinders(PINS, PIN_D + 2 * PIN_CLEAR, PIN_INTO + 0.3, HOLDER_Z))
    body = body.cut(cylinders(MAGNETS, MAGNET_D, MAGNET_H, HOLDER_TOP - MAGNET_H))
    return body.fuse(cylinders(CAP_PINS, CAP_PIN_D, CAP_PIN_H, HOLDER_TOP)).removeSplitter()


def make_cap():
    w, d = CENTRE[0] - 2 * CAP_INSET, CENTRE[1] - 2 * CAP_INSET
    cap = rounded_box(w, d, CAP_H, OUTLINE_R + 2, HOLDER_TOP)
    step_h = CAP_H / CAP_STEPS
    for i in range(CAP_STEPS):  # la ventana se abre en escalones hacia la cámara
        cap = cap.cut(centred_box(WIN_ALONG + 2 * i * CAP_STEP, WIN_ACROSS + 2 * i * CAP_STEP, step_h + 0.01,
                                  HOLDER_TOP + i * step_h))
    cap = cap.cut(cylinders(MAGNETS, MAGNET_D, MAGNET_H, HOLDER_TOP))
    cap = cap.cut(cylinders(CAP_PINS, CAP_PIN_D + 2 * PIN_CLEAR, CAP_PIN_H + 0.4, HOLDER_TOP))
    return cap.removeSplitter()


def check(parts):
    """Lo que el diseño promete, comprobado sobre la geometría."""
    base, tray, hoop, body, cap = (parts[k] for k in ("base", "cajon", "aro", "cuerpo", "capucha"))
    assert abs(FILM_Z - DIFFUSER_Z - FILM_ABOVE_DIFFUSER) < 1e-9
    assert WIN_ACROSS >= FILM_W + 2.0, "la ventana debe dejar ver la tira de orilla a orilla"
    assert LIGHT_ALONG >= WIN_ALONG + 2 * TUNNEL_FLARE and LIGHT_ACROSS >= WIN_ACROSS + 2 * TUNNEL_FLARE
    assert RING_WIN[0] > LIGHT_ALONG and RING_WIN[1] > LIGHT_ACROSS, "el aro no debe tapar la luz"
    assert FLOOR_T + TRAY_PLATE_T + RING_H + 0.3 < BASE_H, "el cajón con el papel debe caber en el riel"
    ring_out = (RING_WIN[0] + 2 * RING_WALL, RING_WIN[1] + 2 * RING_WALL)
    assert PAPER[0] > ring_out[0] + 6 and PAPER[1] > ring_out[1] + 6, "la hoja debe pasar el aro"
    assert PAPER[0] <= HOOP_OUT[0] and PAPER[1] <= HOOP_OUT[1]
    assert CENTRE[0] / 2 - CAP_INSET < ARM_ROOT_X, "la capucha no debe pisar los brazos"
    pairs = [("base/cajón", base, tray), ("base/aro", base, hoop), ("cajón/aro", tray, hoop),
             ("base/cuerpo", base, body), ("cajón/cuerpo", tray, body), ("aro/cuerpo", hoop, body),
             ("cuerpo/capucha", body, cap)]
    for name, a, b in pairs:
        overlap = a.common(b).Volume
        assert overlap < 1e-3, f"{name} se enciman {overlap:.3f} mm³"
    # El cajón sale por +Y sin chocar: barrerlo hacia afuera no toca la base ni el cuerpo.
    for dy in (10.0, 40.0, 80.0):
        moved = tray.copy()
        moved.translate(V(0, dy, 0))
        assert moved.common(base).Volume < 1e-3 and moved.common(body).Volume < 1e-3, "el cajón no sale"
    # La tira corre de punta a punta: nada sobre el canal salvo la capucha, en el centro.
    path = centred_box(2 * ARM_TIP_X + 4, CHANNEL_W - 0.01, 6.0, FILM_Z)
    assert body.common(path).Volume < 1e-3, "los brazos tapan el paso de la tira"
    # Los brazos suben en curva: la punta está ARM_RISE más alta que la raíz.
    tip = body.common(centred_box(4, 4, 40, 0, x=ARM_TIP_X - 6, y=ARM_INNER + 6)).BoundBox.ZMax
    assert tip > HOLDER_TOP + 0.8 * ARM_RISE, "los brazos no suben"
    column = centred_box(WIN_ALONG - 0.01, WIN_ACROSS - 0.01, FILM_Z - DIFFUSER_Z, DIFFUSER_Z)
    for name, shape in (("cajón", tray), ("aro", hoop), ("cuerpo", body)):
        assert shape.common(column).Volume < 1e-3, f"{name} tapa la luz"
    for name, shape in parts.items():
        bb = shape.BoundBox
        assert bb.XLength <= 220 and bb.YLength <= 220, f"{name} no cabe en la cama"
        assert shape.isValid(), f"{name} no es un sólido válido"


def print_pose(name, shape):
    """Cada pieza sobre su cara plana de abajo, sin soportes."""
    s = shape.copy()
    s.translate(V(0, 0, -s.BoundBox.ZMin))
    return s


def export(parts):
    os.makedirs(os.path.join(HERE, "stl"), exist_ok=True)
    for old in os.listdir(os.path.join(HERE, "stl")):
        if old.endswith(".stl") and old[:-4] not in parts:
            os.remove(os.path.join(HERE, "stl", old))
    doc = App.newDocument("portanegativos_35_v2")
    for name, shape in parts.items():
        doc.addObject("Part::Feature", name).Shape = shape
        mesh = MeshPart.meshFromShape(Shape=print_pose(name, shape), LinearDeflection=0.02,
                                      AngularDeflection=math.radians(10), Relative=False)
        # La caja de OCC es holgada en superficies curvas: se apoya la malla misma en la cama.
        mesh.translate(0, 0, -mesh.BoundBox.ZMin)
        mesh.write(os.path.join(HERE, "stl", f"{name}.stl"))
    doc.recompute()
    doc.saveAs(os.path.join(HERE, "portanegativos_35_v2.FCStd"))


def render(items, azim, elev, size=(1400, 900), margin=40):
    """Vista sombreada con z-buffer de [(forma, color)], proyección ortográfica."""
    import numpy as np
    import matplotlib.colors as mc

    a, e = math.radians(azim), math.radians(elev)
    d = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])  # hacia la cámara
    right = np.cross([0.0, 0.0, 1.0], d); right /= np.linalg.norm(right)
    up = np.cross(d, right)
    lights = [np.array(v) / np.linalg.norm(v) for v in ((0.4, -0.6, 0.9), (-0.7, 0.3, 0.5))]
    tris, cols = [], []
    for shape, color in items:
        verts, faces = shape.tessellate(0.15)
        vv = np.array([(p.x, p.y, p.z) for p in verts])
        ff = np.array(faces)
        t = vv[ff]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        n[(n @ d) < 0] *= -1  # dos caras: el teselado no siempre orienta las normales
        shade = 0.28 + 0.62 * np.clip(n @ lights[0], 0, 1) + 0.22 * np.clip(n @ lights[1], 0, 1)
        rgb = np.array(mc.to_rgb(color))
        tris.append(t)
        cols.append(np.clip(shade[:, None] * rgb[None, :] * 1.15, 0, 1))
    tris, cols = np.concatenate(tris), np.concatenate(cols)
    xs, ys, zs = tris @ right, tris @ up, tris @ d
    w, h = size
    scale = min((w - 2 * margin) / (xs.max() - xs.min()), (h - 2 * margin) / (ys.max() - ys.min()))
    px = (xs - xs.min()) * scale + (w - (xs.max() - xs.min()) * scale) / 2
    py = (ys.max() - ys) * scale + (h - (ys.max() - ys.min()) * scale) / 2
    image = np.ones((h, w, 3))
    depth = np.full((h, w), -np.inf)
    for (x0, x1, x2), (y0, y1, y2), (z0, z1, z2), c in zip(px, py, zs, cols):
        minx, maxx = int(max(min(x0, x1, x2), 0)), int(min(max(x0, x1, x2), w - 1)) + 1
        miny, maxy = int(max(min(y0, y1, y2), 0)), int(min(max(y0, y1, y2), h - 1)) + 1
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if maxx <= minx or maxy <= miny or abs(area) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(minx, maxx) + 0.5, np.arange(miny, maxy) + 0.5)
        w0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) / area
        w1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) / area
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        z = w0 * z0 + w1 * z1 + w2 * z2
        sub = depth[miny:maxy, minx:maxx]
        hit = inside & (z > sub)
        sub[hit] = z[hit]
        image[miny:maxy, minx:maxx][hit] = c
    return image


def previews(parts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(HERE, "img"), exist_ok=True)
    colors = {"base": "#4a565b", "cajon": "#9aa8ad", "aro": "#7d8b90", "cuerpo": "#30373a", "capucha": "#262b2d"}
    film = centred_box(240, FILM_W, 0.15, FILM_Z)

    def moved(shape, dz=0.0, dy=0.0):
        s = shape.copy()
        s.translate(V(0, dy, dz))
        return s

    assembled = [(moved(sh, dy=30.0 if n in ("cajon", "aro") else 0.0), colors[n]) for n, sh in parts.items()]
    assembled.append((film, "#d4772f"))
    plt.imsave(os.path.join(HERE, "img", "armado.png"), render(assembled, -62, 24))
    lift = {"base": 0, "cajon": 14, "aro": 24, "cuerpo": 38, "capucha": 60}
    exploded = [(moved(sh, dz=lift[n]), colors[n]) for n, sh in parts.items()]
    exploded.append((moved(centred_box(*PAPER, 0.05, DIFFUSER_Z), dz=lift["aro"] + 0.3), "#efe9d6"))
    plt.imsave(os.path.join(HERE, "img", "despiece.png"), render(exploded, -60, 26))
    plt.imsave(os.path.join(HERE, "img", "lado.png"),
               render([(sh, colors[n]) for n, sh in parts.items()] + [(film, "#d4772f")], -90, 8, size=(1400, 500)))

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=130)
    for name, shape in parts.items():
        for wire in shape.slice(V(1, 0, 0), 0.0):
            pts = wire.discretize(Distance=0.2)
            ax.fill([p.y for p in pts], [p.z for p in pts], color=colors[name], lw=0)
    ax.plot([-FILM_W / 2, FILM_W / 2], [FILM_Z + 0.07] * 2, color="#c86a2a", lw=2.5, label="película")
    ax.plot([-PAPER[1] / 2, PAPER[1] / 2], [DIFFUSER_Z + 0.03] * 2, color="#d9d2b8", lw=2, label="difusor")
    ax.axhline(0, color="#7fb7c9", lw=3, label="pantalla")
    ax.set_aspect("equal"); ax.set_xlabel("mm (a lo ancho de la tira)"); ax.set_ylabel("mm")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "img", "corte.png"), facecolor="white"); plt.close(fig)


def main():
    parts = {"base": make_base(), "cajon": make_tray(), "aro": make_hoop(), "cuerpo": make_body(),
             "capucha": make_cap()}
    check(parts)
    export(parts)
    previews(parts)
    for name, shape in parts.items():
        bb = print_pose(name, shape).BoundBox
        print(f"{name:8s} {bb.XLength:6.1f} × {bb.YLength:5.1f} × {bb.ZLength:4.1f} mm, {shape.Volume / 1000:5.1f} cm³")
    print(f"pantalla → difusor {DIFFUSER_Z:.1f} mm · difusor → película {FILM_ABOVE_DIFFUSER} mm · "
          f"luz en Belka {LIGHT_ALONG:.0f}×{LIGHT_ACROSS:.0f} mm")


main()
