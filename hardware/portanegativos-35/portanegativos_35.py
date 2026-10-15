"""Portanegativos Belka 35: tira de 35 mm sobre la pantalla, con difusor tensado debajo.

Cuatro piezas, de abajo arriba (Z = 0 es la superficie de la pantalla):

* base: marco que se apoya en la pantalla, con una ranura alrededor de la
  ventana y cuatro pernos.
* tensor: marco que baja sobre la base con un nervio que mete el difusor (papel
  de horno, cebolla o vegetal) en la ranura y lo deja tenso, como un bastidor.
* portanegativos: canal de 35 mm para la tira, 7 mm por encima del difusor, con
  una ventana que deja ver la película de orilla a orilla (perforaciones y
  código DX con luz) y los huecos entre fotogramas.
* tapa: aplana la tira con cuatro imanes de 6×2 mm.

La pantalla debe iluminar LIGHT_ALONG × LIGHT_ACROSS (preajuste «Portanegativos
Belka 35» del panel de luz); sus marcas de esquina quedan justo en el borde de la
ventana de la base.

Uso: freecadcmd portanegativos_35.py (o el AppImage: FreeCAD.AppImage freecadcmd …).
Escribe stl/*.stl orientadas para imprimir, portanegativos_35.FCStd y vistas en img/.
"""

import math
import os

import FreeCAD as App
import Mesh
import MeshPart
import Part

V = App.Vector
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ película y luz
FILM_W = 35.0
CHANNEL_W = 35.4  # canal de la tira: película + holgura
CHANNEL_DEPTH = 0.4  # la tapa queda esto por encima de la película (0,13–0,15 mm)
WIN_ALONG, WIN_ACROSS = 42.0, 37.0  # ventana: fotograma de 36 mm + huecos, y todo el ancho de la tira
LIGHT_ALONG, LIGHT_ACROSS = 52.0, 46.0  # zona iluminada en la pantalla (preajuste de Belka)
MARK_GAP = 2.0  # Belka dibuja las marcas de esquina 2 mm fuera de la zona iluminada

# ------------------------------------------------------------------ alturas
BASE_T = 3.0  # pantalla → difusor: desenfoca la rejilla de píxeles
CLAMP_T = 3.0
FILM_ABOVE_DIFFUSER = 7.0  # difusor → película: desenfoca la textura del difusor
LID_T = 3.0

# ------------------------------------------------------------------ marcos
FRAME_OUT = (110.0, 90.0)
FRAME_R = 5.0
BASE_WIN = (LIGHT_ALONG + 2 * MARK_GAP, LIGHT_ACROSS + 2 * MARK_GAP)  # las marcas se ven en sus esquinas
GROOVE_OFFSET, GROOVE_W, GROOVE_D = 4.0, 1.6, 1.2  # ranura de tensado, desde el borde de la ventana
RIB_W, RIB_H = 1.0, 0.9  # nervio del tensor (deja 0,3 mm para el papel)
PIN_D, PIN_CLEAR = 3.0, 0.35
# Fuera de una hoja de 80×70 mm: el papel no hay que perforarlo.
PINS = [(sx * 47.0, sy * 38.0) for sx in (-1, 1) for sy in (-1, 1)]
PAPER = (80.0, 70.0)
PIN_INTO_HOLDER = 2.5

# ------------------------------------------------------------------ portanegativos y tapa
HOLDER_OUT = (130.0, 90.0)
HOLDER_R = 8.0
TUNNEL_FLARE = 4.0  # la ventana se abre 45° hacia el difusor
LID_OUT = (80.0, 90.0)
MAGNET_D, MAGNET_H = 6.2, 2.1  # imanes de 6×2 mm
MAGNETS = [(sx * 32.0, sy * 32.0) for sx in (-1, 1) for sy in (-1, 1)]
LID_PINS = [(0.0, 32.0), (0.0, -32.0)]
LID_PIN_D, LID_PIN_H = 3.0, 2.0

DIFFUSER_Z = BASE_T
HOLDER_Z = BASE_T + CLAMP_T
FILM_Z = DIFFUSER_Z + FILM_ABOVE_DIFFUSER
HOLDER_TOP = FILM_Z + CHANNEL_DEPTH


def rounded_box(w, d, h, r, z=0.0):
    """Caja centrada en XY con las aristas verticales redondeadas."""
    box = Part.makeBox(w, d, h, V(-w / 2, -d / 2, z))
    edges = [e for e in box.Edges if abs(e.Vertexes[0].Point.z - e.Vertexes[1].Point.z) > 1e-6]
    return box.makeFillet(r, edges)


def centred_box(w, d, h, z=0.0, x=0.0, y=0.0):
    return Part.makeBox(w, d, h, V(x - w / 2, y - d / 2, z))


def rect_ring(inner_w, inner_d, width, h, z):
    outer = centred_box(inner_w + 2 * width, inner_d + 2 * width, h, z)
    return outer.cut(centred_box(inner_w, inner_d, h, z))


def frustum(w0, d0, w1, d1, z0, z1):
    """Prisma de sección rectangular que pasa de w0×d0 en z0 a w1×d1 en z1."""
    def rect(w, d, z):
        pts = [V(-w / 2, -d / 2, z), V(w / 2, -d / 2, z), V(w / 2, d / 2, z), V(-w / 2, d / 2, z)]
        return Part.makePolygon(pts + [pts[0]])
    return Part.makeLoft([rect(w0, d0, z0), rect(w1, d1, z1)], True)


def cylinders(points, d, h, z):
    shapes = [Part.makeCylinder(d / 2, h, V(x, y, z)) for x, y in points]
    out = shapes[0]
    for s in shapes[1:]:
        out = out.fuse(s)
    return out


def make_base():
    frame = rounded_box(*FRAME_OUT, BASE_T, FRAME_R)
    frame = frame.cut(centred_box(*BASE_WIN, BASE_T))
    groove = rect_ring(BASE_WIN[0] + 2 * GROOVE_OFFSET, BASE_WIN[1] + 2 * GROOVE_OFFSET, GROOVE_W,
                       GROOVE_D, BASE_T - GROOVE_D)
    frame = frame.cut(groove)
    pins = cylinders(PINS, PIN_D, CLAMP_T + PIN_INTO_HOLDER, BASE_T)
    return frame.fuse(pins).removeSplitter()


def make_clamp():
    frame = rounded_box(*FRAME_OUT, CLAMP_T, FRAME_R, BASE_T)
    frame = frame.cut(centred_box(*BASE_WIN, CLAMP_T, BASE_T))
    frame = frame.cut(cylinders(PINS, PIN_D + 2 * PIN_CLEAR, CLAMP_T, BASE_T))
    offset = GROOVE_OFFSET + (GROOVE_W - RIB_W) / 2
    rib = rect_ring(BASE_WIN[0] + 2 * offset, BASE_WIN[1] + 2 * offset, RIB_W, RIB_H, BASE_T - RIB_H)
    return frame.fuse(rib).removeSplitter()


def make_holder():
    h = HOLDER_TOP - HOLDER_Z
    body = rounded_box(*HOLDER_OUT, h, HOLDER_R, HOLDER_Z)
    channel = centred_box(HOLDER_OUT[0] + 2, CHANNEL_W, CHANNEL_DEPTH, FILM_Z)
    tunnel = frustum(WIN_ALONG + 2 * TUNNEL_FLARE, WIN_ACROSS + 2 * TUNNEL_FLARE, WIN_ALONG, WIN_ACROSS,
                     HOLDER_Z, FILM_Z)
    window = centred_box(WIN_ALONG, WIN_ACROSS, CHANNEL_DEPTH, FILM_Z)
    pin_holes = cylinders(PINS, PIN_D + 2 * PIN_CLEAR, PIN_INTO_HOLDER + 0.3, HOLDER_Z)
    pockets = cylinders(MAGNETS, MAGNET_D, MAGNET_H, HOLDER_TOP - MAGNET_H)
    body = body.cut(channel).cut(tunnel).cut(window).cut(pin_holes).cut(pockets)
    pins = cylinders(LID_PINS, LID_PIN_D, LID_PIN_H, HOLDER_TOP)
    return body.fuse(pins).removeSplitter()


def make_lid():
    lid = rounded_box(*LID_OUT, LID_T, HOLDER_R, HOLDER_TOP)
    # Bisel de 45° hacia la cámara: la tapa no proyecta sombra sobre la película.
    opening = frustum(WIN_ALONG, WIN_ACROSS, WIN_ALONG + 2 * LID_T, WIN_ACROSS + 2 * LID_T,
                      HOLDER_TOP, HOLDER_TOP + LID_T)
    lid = lid.cut(opening)
    lid = lid.cut(cylinders(MAGNETS, MAGNET_D, MAGNET_H, HOLDER_TOP))
    lid = lid.cut(cylinders(LID_PINS, LID_PIN_D + 2 * PIN_CLEAR, LID_PIN_H + 0.4, HOLDER_TOP))
    return lid.removeSplitter()


def check(parts):
    """Lo que el diseño promete, comprobado sobre la geometría."""
    base, clamp, holder, lid = (parts[k] for k in ("base", "tensor", "portanegativos", "tapa"))
    assert abs(FILM_Z - DIFFUSER_Z - FILM_ABOVE_DIFFUSER) < 1e-9
    assert WIN_ACROSS >= FILM_W + 2.0, "la ventana debe dejar ver la tira de orilla a orilla"
    assert LIGHT_ALONG >= WIN_ALONG + 2 * TUNNEL_FLARE and LIGHT_ACROSS >= WIN_ACROSS + 2 * TUNNEL_FLARE
    assert BASE_WIN[0] > LIGHT_ALONG and BASE_WIN[1] > LIGHT_ACROSS
    assert RIB_H < GROOVE_D and RIB_W < GROOVE_W
    groove_out = (BASE_WIN[0] + 2 * (GROOVE_OFFSET + GROOVE_W), BASE_WIN[1] + 2 * (GROOVE_OFFSET + GROOVE_W))
    assert PAPER[0] > groove_out[0] + 6 and PAPER[1] > groove_out[1] + 6, "la hoja debe pasar la ranura"
    assert all(abs(x) - PIN_D / 2 > PAPER[0] / 2 or abs(y) - PIN_D / 2 > PAPER[1] / 2 for x, y in PINS)
    for name, a, b in (("base/tensor", base, clamp), ("tensor/portanegativos", clamp, holder),
                       ("portanegativos/tapa", holder, lid), ("base/portanegativos", base, holder)):
        overlap = a.common(b).Volume
        assert overlap < 1e-3, f"{name} se enciman {overlap:.3f} mm³"
    # La luz llega a toda la ventana: nada sólido sobre la zona de la ventana entre difusor y película.
    column = centred_box(WIN_ALONG - 0.01, WIN_ACROSS - 0.01, FILM_Z - DIFFUSER_Z, DIFFUSER_Z)
    for name, shape in (("base", base), ("tensor", clamp), ("portanegativos", holder)):
        assert shape.common(column).Volume < 1e-3, f"{name} tapa la luz"
    for name, shape in parts.items():
        bb = shape.BoundBox
        assert bb.XLength <= 220 and bb.YLength <= 220, f"{name} no cabe en la cama"
        assert shape.isValid(), f"{name} no es un sólido válido"


def print_pose(name, shape):
    """Cada pieza sobre su cara más grande, sin soportes."""
    s = shape.copy()
    if name in ("tensor", "tapa"):  # nervio / bisel hacia arriba, cara plana en la cama
        s.rotate(V(0, 0, s.BoundBox.ZMin), V(1, 0, 0), 180)
    s.translate(V(0, 0, -s.BoundBox.ZMin))
    return s


def export(parts):
    os.makedirs(os.path.join(HERE, "stl"), exist_ok=True)
    doc = App.newDocument("portanegativos_35")
    for name, shape in parts.items():
        doc.addObject("Part::Feature", name).Shape = shape
        mesh = MeshPart.meshFromShape(Shape=print_pose(name, shape), LinearDeflection=0.02,
                                      AngularDeflection=math.radians(10), Relative=False)
        mesh.write(os.path.join(HERE, "stl", f"{name}.stl"))
    doc.recompute()
    doc.saveAs(os.path.join(HERE, "portanegativos_35.FCStd"))


def previews(parts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    os.makedirs(os.path.join(HERE, "img"), exist_ok=True)
    colors = {"base": "#3d4a4f", "tensor": "#56656b", "portanegativos": "#24292b", "tapa": "#1a1d1e"}
    lift = {"base": 0, "tensor": 12, "portanegativos": 28, "tapa": 44}
    fig = plt.figure(figsize=(9, 7), dpi=130)
    ax = fig.add_subplot(projection="3d")
    for name, shape in parts.items():
        s = shape.copy()
        s.translate(V(0, 0, lift[name]))
        verts, faces = s.tessellate(0.2)
        tris = [[(verts[i].x, verts[i].y, verts[i].z) for i in f] for f in faces]
        ax.add_collection3d(Poly3DCollection(tris, facecolor=colors[name], edgecolor="none", alpha=0.95))
    paper = centred_box(BASE_WIN[0] + 14, BASE_WIN[1] + 14, 0.05, DIFFUSER_Z + 6)
    verts, faces = paper.tessellate(0.5)
    ax.add_collection3d(Poly3DCollection([[(verts[i].x, verts[i].y, verts[i].z) for i in f] for f in faces],
                                         facecolor="#f2efe4", edgecolor="none", alpha=0.85))
    film = centred_box(110, FILM_W, 0.15, FILM_Z + 28 + 6)
    verts, faces = film.tessellate(0.5)
    ax.add_collection3d(Poly3DCollection([[(verts[i].x, verts[i].y, verts[i].z) for i in f] for f in faces],
                                         facecolor="#c86a2a", edgecolor="none", alpha=0.8))
    ax.set_xlim(-70, 70); ax.set_ylim(-60, 60); ax.set_zlim(0, 70)
    ax.set_box_aspect((140, 120, 70))
    ax.view_init(elev=28, azim=-58)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "img", "despiece.png"), facecolor="white")
    plt.close(fig)

    # Corte por el centro, a lo ancho de la tira: alturas de difusor, película y tapa.
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=130)
    for name, shape in parts.items():
        for wire in shape.slice(V(1, 0, 0), 0.0):
            pts = wire.discretize(Distance=0.2)
            ax.fill([p.y for p in pts], [p.z for p in pts], color=colors[name], lw=0)
    ax.plot([-FILM_W / 2, FILM_W / 2], [FILM_Z + 0.07] * 2, color="#c86a2a", lw=2.5, label="película")
    ax.plot([-BASE_WIN[1] / 2 - 8, BASE_WIN[1] / 2 + 8], [DIFFUSER_Z] * 2, color="#d9d2b8", lw=2, label="difusor")
    ax.axhline(0, color="#7fb7c9", lw=3, label="pantalla")
    ax.set_aspect("equal"); ax.set_xlabel("mm (a lo ancho de la tira)"); ax.set_ylabel("mm")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "img", "corte.png"), facecolor="white")
    plt.close(fig)


def main():
    parts = {"base": make_base(), "tensor": make_clamp(), "portanegativos": make_holder(), "tapa": make_lid()}
    check(parts)
    export(parts)
    previews(parts)
    for name, shape in parts.items():
        bb = print_pose(name, shape).BoundBox
        print(f"{name:15s} {bb.XLength:6.1f} × {bb.YLength:5.1f} × {bb.ZLength:4.1f} mm, {shape.Volume / 1000:5.1f} cm³")
    print(f"pantalla → difusor {DIFFUSER_Z} mm · difusor → película {FILM_ABOVE_DIFFUSER} mm · "
          f"luz en Belka {LIGHT_ALONG:.0f}×{LIGHT_ACROSS:.0f} mm")


main()
