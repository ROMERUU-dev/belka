"""Portanegativos Belka 35 · v2 (retro): cuerpo alargado con alas, bloque central
escalonado y un riel con cajón para el difusor.

Cinco piezas, de abajo arriba (Z = 0 es la superficie de la pantalla):

* base: se apoya en la pantalla con la silueta del cuerpo. Tiene un riel abierto
  por un costado (+Y) donde entra el cajón del difusor, y cuatro pernos.
* cajón: placa con un aro elevado alrededor de la ventana. La hoja de difusor
  (papel de horno, cebolla o vegetal) se pone encima del aro y el
* aro: marco que baja por fuera del aro y la deja tensa, como un bastidor de
  bordado. El cajón entra y sale del riel sin desarmar nada.
* cuerpo: canal de 35 mm de punta a punta, 7 mm por encima del difusor, con
  entradas abocinadas en las alas.
* bloque: el centro elevado con la ventana escalonada; aplana la tira con cuatro
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
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ------------------------------------------------------------------ película y luz
FILM_W = 35.0
CHANNEL_W = 35.4  # canal de la tira: película + holgura
CHANNEL_DEPTH = 0.4  # el bloque queda esto por encima de la película (0,13–0,15 mm)
WIN_ALONG, WIN_ACROSS = 42.0, 37.0  # ventana: fotograma de 36 mm + huecos, y todo el ancho de la tira
LIGHT_ALONG, LIGHT_ACROSS = 52.0, 46.0  # zona iluminada en la pantalla (preajuste de Belka)
MARK_GAP = 2.0  # Belka dibuja las marcas de esquina 2 mm fuera de la zona iluminada
BASE_WIN = (LIGHT_ALONG + 2 * MARK_GAP, LIGHT_ACROSS + 2 * MARK_GAP)

# ------------------------------------------------------------------ silueta (vista desde arriba; X a lo largo de la tira)
CENTRE = (76.0, 74.0)  # sección central
NECK_W = 54.0  # cintura entre el centro y las alas
WING = (30.0, 84.0)  # cada ala
WING_X = 67.0  # centro de cada ala
OUTLINE_R = 4.0
NOTCH_R = 7.0  # muesca de media luna en los costados de cada ala

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

# ------------------------------------------------------------------ bloque central
CAP_INSET = 2.0
CAP_H = 7.0
CAP_STEPS, CAP_STEP = 4, 2.5  # escalones de la ventana, cada uno CAP_STEP más ancho por lado

# ------------------------------------------------------------------ uniones
PIN_D, PIN_CLEAR, PIN_INTO = 3.0, 0.35, 2.5
PINS = [(sx * WING_X, sy * 22.0) for sx in (-1, 1) for sy in (-1, 1)]
MAGNET_D, MAGNET_H = 6.2, 2.1  # imanes de 6×2 mm
MAGNETS = [(sx * 29.0, sy * 28.0) for sx in (-1, 1) for sy in (-1, 1)]
CAP_PINS = [(0.0, 29.0), (0.0, -29.0)]
CAP_PIN_D, CAP_PIN_H = 3.0, 2.0
TEXT_H, TEXT_DEPTH = 5.0, 0.6
TEXT_Y = 26.0  # sobre cada ala, fuera del canal de la tira
ENGRAVED: list[str] = []


def centred_box(w, d, h, z=0.0, x=0.0, y=0.0):
    return Part.makeBox(w, d, h, V(x - w / 2, y - d / 2, z))


def fuse_all(shapes):
    out = shapes[0]
    for s in shapes[1:]:
        out = out.fuse(s)
    return out.removeSplitter()


def vertical_fillet(shape, r):
    edges = [e for e in shape.Edges
             if len(e.Vertexes) == 2 and abs(e.Vertexes[0].Point.z - e.Vertexes[1].Point.z) > 1e-6
             and abs(e.Vertexes[0].Point.x - e.Vertexes[1].Point.x) < 1e-6
             and abs(e.Vertexes[0].Point.y - e.Vertexes[1].Point.y) < 1e-6]
    try:
        return shape.makeFillet(r, edges)
    except Part.OCCError:
        return shape


def outline(h, z=0.0):
    """Silueta de hueso: centro, cintura y alas con muescas, redondeada."""
    parts = [centred_box(*CENTRE, h, z), centred_box(2 * WING_X, NECK_W, h, z)]
    parts += [centred_box(*WING, h, z, x=sx * WING_X) for sx in (-1, 1)]
    body = vertical_fillet(fuse_all(parts), OUTLINE_R)
    notches = [Part.makeCylinder(NOTCH_R, h, V(sx * WING_X, sy * WING[1] / 2, z)) for sx in (-1, 1) for sy in (-1, 1)]
    return body.cut(fuse_all(notches))


def rounded_box(w, d, h, r, z=0.0):
    return vertical_fillet(centred_box(w, d, h, z), r)


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


def engraving(text, x, y, z):
    """Texto grabado TEXT_DEPTH en la cara z, centrado en (x, y)."""
    try:
        wires = Part.makeWireString(text, FONT, TEXT_H, 0.0)
    except Exception as exc:
        print(f"sin texto grabado ({exc})")
        return None
    faces = []
    for char in wires:
        if not char:
            continue
        try:
            faces.append(Part.makeFace(char, "Part::FaceMakerBullseye"))
        except Exception:
            return None
    solid = fuse_all([f.extrude(V(0, 0, TEXT_DEPTH + 0.01)) for f in faces])
    bb = solid.BoundBox
    solid.translate(V(x - bb.Center.x, y - bb.Center.y, z - TEXT_DEPTH))
    return solid


def make_base():
    base = outline(BASE_H)
    base = base.cut(centred_box(*BASE_WIN, FLOOR_T))
    slot_w = TRAY_PLATE[0] + 2 * SLOT_CLEAR
    slot_y0 = -TRAY_PLATE[1] / 2 - SLOT_CLEAR  # tope del cajón; abierto hacia +Y
    slot_len = CENTRE[1] / 2 - slot_y0 + 1.0
    base = base.cut(Part.makeBox(slot_w, slot_len, SLOT_H + 0.01, V(-slot_w / 2, slot_y0, FLOOR_T)))
    pins = cylinders(PINS, PIN_D, PIN_INTO, BASE_H)
    return base.fuse(pins).removeSplitter()


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
    h = HOLDER_TOP - HOLDER_Z
    body = outline(h, HOLDER_Z)
    length = 2 * (WING_X + WING[0] / 2) + 2
    body = body.cut(centred_box(length, CHANNEL_W, CHANNEL_DEPTH, FILM_Z))
    end = WING_X + WING[0] / 2
    for sx in (-1, 1):
        body = body.cut(mouth(sx * (end - LEAD_IN[0]), sx * (end + 0.5), CHANNEL_W, LEAD_IN[1], FILM_Z, CHANNEL_DEPTH))
    body = body.cut(frustum(WIN_ALONG + 2 * TUNNEL_FLARE, WIN_ACROSS + 2 * TUNNEL_FLARE, WIN_ALONG, WIN_ACROSS,
                            HOLDER_Z, FILM_Z))
    body = body.cut(centred_box(WIN_ALONG, WIN_ACROSS, CHANNEL_DEPTH, FILM_Z))
    body = body.cut(cylinders(PINS, PIN_D + 2 * PIN_CLEAR, PIN_INTO + 0.3, HOLDER_Z))
    body = body.cut(cylinders(MAGNETS, MAGNET_D, MAGNET_H, HOLDER_TOP - MAGNET_H))
    for text, sx in (("BELKA", -1), ("35", 1)):
        mark = engraving(text, sx * WING_X, TEXT_Y, HOLDER_TOP)
        if mark is not None:
            body = body.cut(mark)
            ENGRAVED.append(text)
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
    base, tray, hoop, body, cap = (parts[k] for k in ("base", "cajon", "aro", "cuerpo", "bloque"))
    assert abs(FILM_Z - DIFFUSER_Z - FILM_ABOVE_DIFFUSER) < 1e-9
    assert WIN_ACROSS >= FILM_W + 2.0, "la ventana debe dejar ver la tira de orilla a orilla"
    assert LIGHT_ALONG >= WIN_ALONG + 2 * TUNNEL_FLARE and LIGHT_ACROSS >= WIN_ACROSS + 2 * TUNNEL_FLARE
    assert RING_WIN[0] > LIGHT_ALONG and RING_WIN[1] > LIGHT_ACROSS, "el aro no debe tapar la luz"
    assert FLOOR_T + TRAY_PLATE_T + RING_H + 0.3 < BASE_H, "el cajón con el papel debe caber en el riel"
    ring_out = (RING_WIN[0] + 2 * RING_WALL, RING_WIN[1] + 2 * RING_WALL)
    assert PAPER[0] > ring_out[0] + 6 and PAPER[1] > ring_out[1] + 6, "la hoja debe pasar el aro"
    assert PAPER[0] <= HOOP_OUT[0] and PAPER[1] <= HOOP_OUT[1]
    pairs = [("base/cajón", base, tray), ("base/aro", base, hoop), ("cajón/aro", tray, hoop),
             ("base/cuerpo", base, body), ("cajón/cuerpo", tray, body), ("aro/cuerpo", hoop, body),
             ("cuerpo/bloque", body, cap)]
    for name, a, b in pairs:
        overlap = a.common(b).Volume
        assert overlap < 1e-3, f"{name} se enciman {overlap:.3f} mm³"
    # El cajón sale por +Y sin chocar: barrerlo hacia afuera no toca la base ni el cuerpo.
    for dy in (10.0, 40.0, 80.0):
        moved = tray.copy()
        moved.translate(V(0, dy, 0))
        assert moved.common(base).Volume < 1e-3 and moved.common(body).Volume < 1e-3, "el cajón no sale"
    column = centred_box(WIN_ALONG - 0.01, WIN_ACROSS - 0.01, FILM_Z - DIFFUSER_Z, DIFFUSER_Z)
    for name, shape in (("cajón", tray), ("aro", hoop), ("cuerpo", body)):
        assert shape.common(column).Volume < 1e-3, f"{name} tapa la luz"
    if os.path.exists(FONT):
        assert ENGRAVED == ["BELKA", "35"], "el texto no se grabó"
    for name, shape in parts.items():
        bb = shape.BoundBox
        assert bb.XLength <= 220 and bb.YLength <= 220, f"{name} no cabe en la cama"
        assert shape.isValid(), f"{name} no es un sólido válido"


def print_pose(name, shape):
    """Cada pieza sobre su cara más grande, sin soportes."""
    s = shape.copy()
    s.translate(V(0, 0, -s.BoundBox.ZMin))
    return s


def export(parts):
    os.makedirs(os.path.join(HERE, "stl"), exist_ok=True)
    doc = App.newDocument("portanegativos_35_v2")
    for name, shape in parts.items():
        doc.addObject("Part::Feature", name).Shape = shape
        mesh = MeshPart.meshFromShape(Shape=print_pose(name, shape), LinearDeflection=0.02,
                                      AngularDeflection=math.radians(10), Relative=False)
        mesh.write(os.path.join(HERE, "stl", f"{name}.stl"))
    doc.recompute()
    doc.saveAs(os.path.join(HERE, "portanegativos_35_v2.FCStd"))


def previews(parts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    os.makedirs(os.path.join(HERE, "img"), exist_ok=True)
    colors = {"base": "#3d4a4f", "cajon": "#8a9aa0", "aro": "#6d7c82", "cuerpo": "#262b2d", "bloque": "#17191a"}

    def draw(ax, lift, slide=0.0, film=True):
        for name, shape in parts.items():
            s = shape.copy()
            s.translate(V(0, slide if name in ("cajon", "aro") else 0, lift[name]))
            verts, faces = s.tessellate(0.25)
            tris = [[(verts[i].x, verts[i].y, verts[i].z) for i in f] for f in faces]
            ax.add_collection3d(Poly3DCollection(tris, facecolor=colors[name], edgecolor="none", alpha=0.97))
        paper = centred_box(*PAPER, 0.05, DIFFUSER_Z + lift["cajon"])
        paper.translate(V(0, slide, 0))
        verts, faces = paper.tessellate(0.5)
        ax.add_collection3d(Poly3DCollection([[(verts[i].x, verts[i].y, verts[i].z) for i in f] for f in faces],
                                             facecolor="#f2efe4", edgecolor="none", alpha=0.9))
        if film:
            strip = centred_box(230, FILM_W, 0.15, FILM_Z + lift["cuerpo"])
            verts, faces = strip.tessellate(0.5)
            ax.add_collection3d(Poly3DCollection([[(verts[i].x, verts[i].y, verts[i].z) for i in f] for f in faces],
                                                 facecolor="#c86a2a", edgecolor="none", alpha=0.85))
        ax.set_xlim(-95, 95); ax.set_ylim(-75, 75); ax.set_zlim(0, 75)
        ax.set_box_aspect((190, 150, 75))
        ax.set_axis_off()

    fig = plt.figure(figsize=(10, 7), dpi=130)
    draw(fig.add_subplot(projection="3d"), {"base": 0, "cajon": 12, "aro": 22, "cuerpo": 34, "bloque": 50},
         slide=0.0, film=False)
    fig.axes[0].view_init(elev=26, azim=-60)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "img", "despiece.png"), facecolor="white")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6.5), dpi=130)
    draw(fig.add_subplot(projection="3d"), {k: 0 for k in parts}, slide=30.0)
    fig.axes[0].view_init(elev=32, azim=-62)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "img", "armado.png"), facecolor="white")
    plt.close(fig)

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
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "img", "corte.png"), facecolor="white")
    plt.close(fig)


def main():
    parts = {"base": make_base(), "cajon": make_tray(), "aro": make_hoop(), "cuerpo": make_body(),
             "bloque": make_cap()}
    check(parts)
    export(parts)
    previews(parts)
    for name, shape in parts.items():
        bb = print_pose(name, shape).BoundBox
        print(f"{name:8s} {bb.XLength:6.1f} × {bb.YLength:5.1f} × {bb.ZLength:4.1f} mm, {shape.Volume / 1000:5.1f} cm³")
    print(f"pantalla → difusor {DIFFUSER_Z:.1f} mm · difusor → película {FILM_ABOVE_DIFFUSER} mm · "
          f"luz en Belka {LIGHT_ALONG:.0f}×{LIGHT_ACROSS:.0f} mm")


main()
