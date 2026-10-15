"""Anillo inversor para montar al revés un objetivo con rosca de filtro de 52 mm en una cámara Nikon F.

La bayoneta la pone una tapa de cuerpo Nikon (BF-1B o similar): se le abre un
agujero de HOLE_D al centro y se le pega este anillo encima. El anillo tiene,
de abajo arriba:

* un tetón que entra justo en el agujero de la tapa y lo centra;
* una pestaña plana que se pega a la cara de la tapa;
* una rosca macho M52 × 0,75 (la de los filtros) donde se atornilla el
  objetivo por el frente.

También genera tres anillos de prueba cortos, con distinta holgura de rosca y
1, 2 o 3 muescas en la orilla para distinguirlos, para encontrar cuál entra
mejor en tu objetivo antes de imprimir el bueno.

Todo el anillo es un sólido de revolución cuyo radio exterior cambia con la
altura y el ángulo (tetón, pestaña, rosca), así que la malla se construye
directamente, cerrada y sin operaciones booleanas: las de una rosca de 0,75 mm
fallan en FreeCAD.

Uso: python3 anillo_inversor_52.py (necesita numpy; matplotlib para las
imágenes; también corre con el Python de FreeCAD: freecadcmd anillo_inversor_52.py).
Escribe stl/*.stl orientadas para imprimir e img/.
"""

import math
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ rosca de filtro
THREAD_D = 52.0  # M52 × 0,75: la rosca de filtro del objetivo
PITCH = 0.75
THREAD_LEN = 3.5  # ~4,5 vueltas: los filtros entran 3–4 mm
THREAD_CLEAR = 0.35  # se le quita al diámetro: la impresión sale un poco gorda (ajústalo con las pruebas)
TEST_CLEARS = (0.20, 0.35, 0.50)  # 1, 2 y 3 muescas
TEST_LEN = 2.0

# ------------------------------------------------------------------ tapa de cuerpo y anillo
HOLE_D = 40.0  # agujero que le abres a la tapa (sierra de copa o broca escalonada de 40 mm)
SPIGOT_H = 1.5  # el tetón entra esto en el agujero
FIT = 0.3  # holgura del tetón en el agujero, en el diámetro
FLANGE_D = 54.0  # pestaña que se pega a la cara de la tapa
FLANGE_T = 2.0
BORE_D = 37.0  # paso de luz: más que el lente frontal de un 50 mm (~35 mm)

# ------------------------------------------------------------------ malla
N_THETA = 360  # un grado: la hélice avanza 0,002 mm por paso
DZ = 0.04  # pasos de altura en la rosca: error de radio < 0,04 mm en los flancos

DEPTH = 0.6134 * PITCH  # profundidad del diente (rosca métrica exterior)
CREST = PITCH / 8
ROOT = CREST + 2 * DEPTH * math.tan(math.radians(30))  # ancho del diente en la raíz (flancos de 60°)


def thread_radius(theta, z, d_major, z0, z1):
    """Radio de la rosca (derecha) en el ángulo ``theta`` y la altura ``z``, con chaflán de 45° en las puntas."""
    r_major = d_major / 2
    r_minor = r_major - DEPTH
    u = np.mod((z - z0) - PITCH * theta / (2 * math.pi), PITCH)
    d = np.abs(u - PITCH / 2)
    flank = r_major - (d - CREST / 2) / (ROOT / 2 - CREST / 2) * DEPTH
    r = np.where(d <= CREST / 2, r_major, np.where(d >= ROOT / 2, r_minor, flank))
    return np.minimum(r, np.minimum(r_minor + (z - z0), r_minor + (z1 - z)))


def flange_radius(theta, notches):
    """La pestaña, con ``notches`` muescas de 6° (para distinguir las pruebas)."""
    r = np.full_like(theta, FLANGE_D / 2)
    for k in range(notches):
        centre = math.radians(270 + 14 * k)  # del lado de la cámara en las vistas
        r[np.abs(np.angle(np.exp(1j * (theta - centre)))) < math.radians(3)] = FLANGE_D / 2 - 1.5
    return r


def build(clear, thread_len, spigot=True, notches=0):
    """Malla cerrada del anillo (triángulos N×3×3), con la cara de abajo en Z = 0."""
    theta = np.linspace(0, 2 * math.pi, N_THETA, endpoint=False)
    z_flange, z_thread0 = (SPIGOT_H if spigot else 0.0), (SPIGOT_H if spigot else 0.0) + FLANGE_T
    z_top = z_thread0 + thread_len
    levels = []  # (z, radios): dos niveles a la misma altura dan una cara horizontal (escalón)
    if spigot:
        r_sp = np.full_like(theta, (HOLE_D - FIT) / 2)
        levels += [(0.0, r_sp), (z_flange, r_sp)]
    r_fl = flange_radius(theta, notches)
    levels += [(z_flange, r_fl), (z_thread0, r_fl)]
    for z in np.arange(z_thread0, z_top + 1e-9, DZ):
        levels.append((min(z, z_top), thread_radius(theta, min(z, z_top), THREAD_D - clear, z_thread0, z_top)))
    rings = [np.stack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z)], axis=1) for z, r in levels]
    bore = BORE_D / 2
    inner_bottom = np.stack([bore * np.cos(theta), bore * np.sin(theta), np.zeros_like(theta)], axis=1)
    inner_top = inner_bottom.copy()
    inner_top[:, 2] = z_top
    tris = []
    nxt = np.roll(np.arange(N_THETA), -1)

    def band(a, b, outward=True):
        """Cinta de cuadriláteros entre dos anillos de vértices (a abajo/adentro, b arriba/afuera)."""
        a0, a1, b0, b1 = a, a[nxt], b, b[nxt]
        if outward:
            tris.append(np.stack([a0, a1, b1], axis=1))
            tris.append(np.stack([a0, b1, b0], axis=1))
        else:
            tris.append(np.stack([a0, b1, a1], axis=1))
            tris.append(np.stack([a0, b0, b1], axis=1))

    for lo, hi in zip(rings, rings[1:]):
        band(lo, hi)  # cara exterior (y los escalones, que son cintas horizontales)
    band(inner_bottom, inner_top, outward=False)  # el paso de luz, con la normal hacia el eje
    band(inner_bottom, rings[0], outward=False)  # la cara de abajo, normal hacia -Z
    band(inner_top, rings[-1])  # la cara de arriba, normal hacia +Z
    return np.concatenate(tris)


def check_mesh(tris, name):
    """Cerrada (cada arista en dos triángulos), orientada hacia afuera y con el volumen esperado."""
    v = tris.reshape(-1, 3)
    keys = np.round(v / 1e-6).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_inverse=True)
    f = idx.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), f"{name}: la malla no está cerrada ({np.count_nonzero(counts != 2)} aristas sueltas)"
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
    assert volume > 0, f"{name}: normales hacia adentro"
    return volume


def write_stl(path, tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    data = np.concatenate([n[:, None, :], tris], axis=1).astype("<f4")
    with open(path, "wb") as fh:
        fh.write(b"Belka anillo inversor 52".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        rec = np.zeros(len(tris), dtype=[("d", "<f4", (12,)), ("attr", "<u2")])
        rec["d"] = data.reshape(len(tris), 12)
        fh.write(rec.tobytes())


def previews(tris):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(HERE, "img"), exist_ok=True)
    # Corte por el eje: el perfil del anillo a 0° (derecha) y a 180° (izquierda).
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=140)
    z0 = SPIGOT_H + FLANGE_T
    zs = np.arange(z0, z0 + THREAD_LEN + 1e-9, 0.005)
    for side, th in ((1, 0.0), (-1, math.pi)):
        rt = thread_radius(np.full_like(zs, th), zs, THREAD_D - THREAD_CLEAR, z0, z0 + THREAD_LEN)
        outline = [((HOLE_D - FIT) / 2, 0), ((HOLE_D - FIT) / 2, SPIGOT_H), (FLANGE_D / 2, SPIGOT_H),
                   (FLANGE_D / 2, z0)] + list(zip(rt, zs)) + [(BORE_D / 2, z0 + THREAD_LEN), (BORE_D / 2, 0)]
        ax.fill([side * r for r, _ in outline], [z for _, z in outline], color="#2b3133", lw=0)
    ax.axhline(SPIGOT_H, color="#aaa", lw=0.7, ls="--")
    ax.text(-27, SPIGOT_H - 1.1, "cara que se pega a la tapa", fontsize=8, color="#555")
    ax.text(14, z0 + THREAD_LEN + 0.6, "rosca M52 × 0,75 (aquí entra el objetivo)", fontsize=8, color="#555")
    ax.set_aspect("equal"); ax.set_xlabel("mm"); ax.set_ylabel("mm")
    ax.set_ylim(-1, z0 + THREAD_LEN + 2.5)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "img", "corte.png"), facecolor="white"); plt.close(fig)

    # Detalle de la rosca: un tramo del perfil, para ver los dientes.
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    rt = thread_radius(np.zeros_like(zs), zs, THREAD_D - THREAD_CLEAR, z0, z0 + THREAD_LEN)
    ax.fill(list(rt) + [BORE_D / 2 + 5, BORE_D / 2 + 5], list(zs) + [zs[-1], zs[0]], color="#2b3133", lw=0)
    ax.set_xlim(24.5, 26.3); ax.set_aspect("equal")
    ax.set_xlabel("radio (mm)"); ax.set_ylabel("altura (mm)"); ax.set_title("Dientes de la rosca (paso 0,75 mm)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "img", "rosca.png"), facecolor="white"); plt.close(fig)


def revolve(profile, n=180):
    """Malla de revolución (abierta en el eje no hace falta: perfiles que dan la vuelta) de [(r, z), ...] cerrado."""
    theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
    nxt = np.roll(np.arange(n), -1)
    rings = [np.stack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z)], axis=1) for r, z in profile]
    tris = []
    for a, b in zip(rings, rings[1:] + rings[:1]):
        tris.append(np.stack([a, a[nxt], b[nxt]], axis=1))
        tris.append(np.stack([a, b[nxt], b], axis=1))
    return np.concatenate(tris)


def render(items, azim, elev, size=(1400, 1000), margin=50):
    """Vista sombreada con z-buffer de [(triángulos, color)], proyección ortográfica.
    Devuelve la imagen y una función que lleva puntos 3D a píxeles (para rotular)."""
    import matplotlib.colors as mc

    a, e = math.radians(azim), math.radians(elev)
    d = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])  # hacia la cámara
    right = np.cross([0.0, 0.0, 1.0], d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    lights = [np.array(v) / np.linalg.norm(v) for v in ((0.5, -0.6, 0.8), (-0.6, 0.4, 0.4))]
    tris = np.concatenate([t for t, _ in items])
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n[(n @ d) < 0] *= -1
    shade = 0.25 + 0.65 * np.clip(n @ lights[0], 0, 1) + 0.25 * np.clip(n @ lights[1], 0, 1)
    base = np.concatenate([np.tile(mc.to_rgb(c), (len(t), 1)) for t, c in items])
    cols = np.clip(shade[:, None] * base * 1.15, 0, 1)
    xs, ys, zs = tris @ right, tris @ up, tris @ d
    w, h = size
    scale = min((w - 2 * margin) / np.ptp(xs), (h - 2 * margin) / np.ptp(ys))
    ox = (w - np.ptp(xs) * scale) / 2 - xs.min() * scale
    oy = (h - np.ptp(ys) * scale) / 2 + ys.max() * scale

    def project(p):
        p = np.asarray(p, dtype=float)
        return p @ right * scale + ox, oy - p @ up * scale

    px, py = xs * scale + ox, oy - ys * scale
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
    return image, project


def shifted(tris, dz=0.0, dx=0.0):
    out = tris.copy()
    out[..., 2] += dz
    out[..., 0] += dx
    return out


def body_cap():
    """Tapa de cuerpo aproximada (para la ilustración): disco con su faldón de bayoneta y el agujero de 40 mm."""
    r_hole, r_face, r_skirt = HOLE_D / 2, 29.0, 23.4
    profile = [(r_hole, 0.0), (r_face, 0.0), (r_face, 3.5), (r_skirt, 3.5), (r_skirt, 9.0),
               (r_skirt - 1.6, 9.0), (r_skirt - 1.6, 1.2), (r_hole, 1.2)]
    cap = revolve(profile)
    lugs = []
    for k in range(3):  # las tres orejetas de la bayoneta, dibujadas como bloques
        ang = math.radians(30 + 120 * k)
        c, s_ = math.cos(ang), math.sin(ang)
        box = np.array([[r_skirt - 0.2, -5, 7.0], [r_skirt + 2.2, -5, 7.0], [r_skirt + 2.2, 5, 7.0], [r_skirt - 0.2, 5, 7.0],
                        [r_skirt - 0.2, -5, 9.0], [r_skirt + 2.2, -5, 9.0], [r_skirt + 2.2, 5, 9.0], [r_skirt - 0.2, 5, 9.0]])
        rot = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        v = box @ rot.T
        f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7),
             (2, 7, 6), (3, 0, 4), (3, 4, 7)]
        lugs.append(np.array([[v[i] for i in tri] for tri in f]))
    return np.concatenate([cap] + lugs)


def lens_50():
    """Un 50 mm AF aproximado, al revés: el frente (rosca de filtro) arriba, la montura abajo."""
    profile = [(19.0, 0.0), (23.6, 0.0), (23.6, 4.0), (31.0, 6.0), (32.0, 8.0), (32.0, 30.0), (29.5, 33.0),
               (27.5, 36.0), (26.0, 39.0), (19.5, 39.0), (19.5, 37.0), (17.0, 2.0)]
    return revolve(profile)


def previews_3d(pieces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ring = pieces["anillo"]
    img, _ = render([(ring, "#6b7378")], -55, 24)
    plt.imsave(os.path.join(HERE, "img", "anillo.png"), img)

    tests = [(shifted(pieces[name], dx=dx), "#6b7378")
             for name, dx in (("prueba_rosca_20", -62.0), ("prueba_rosca_35", 0.0), ("prueba_rosca_50", 62.0))]
    img, project = render(tests, -90, 35, size=(1500, 650))
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=100)
    ax.imshow(img); ax.set_axis_off()
    for (label, dx) in (("0,20 mm · 1 muesca", -62.0), ("0,35 mm · 2 muescas", 0.0), ("0,50 mm · 3 muescas", 62.0)):
        x, y = project([dx, -36.0, 0.0])
        ax.text(x, y, label, ha="center", va="top", fontsize=15, color="#333")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(HERE, "img", "pruebas.png"), facecolor="white"); plt.close(fig)

    # Cómo va: la tapa en la cámara (arriba), el anillo pegado debajo y el objetivo atornillado al revés.
    cap = shifted(body_cap(), dz=0.0)
    cap[..., 2] *= -1  # el faldón de bayoneta hacia arriba, hacia la cámara
    cap = shifted(cap, dz=60.0)
    ring_up = ring.copy()
    ring_up[..., 2] *= -1
    ring_up = shifted(ring_up, dz=38.0)
    lens = shifted(lens_50(), dz=-34.0)
    items = [(cap, "#585d61"), (ring_up, "#2f3538"), (lens, "#7a8085")]
    img, project = render(items, -60, 18, size=(1300, 1300))
    fig, ax = plt.subplots(figsize=(19, 13), dpi=100)
    ax.imshow(img); ax.set_axis_off()
    ax.set_xlim(0, 1900); ax.set_ylim(1300, 0)
    for text, point in (("tapa de cuerpo Nikon\n(con agujero de 40 mm; va en la cámara)", (34, 0, 57)),
                        ("anillo impreso\n(pegado a la tapa)", (33, 0, 33)),
                        ("50 mm AF al revés\n(el frente se atornilla en el anillo)", (37, 0, -10))):
        x, y = project(point)
        ax.text(x + 20, y, text, fontsize=17, color="#222", va="center")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(HERE, "img", "montaje.png"), facecolor="white"); plt.close(fig)


def main():
    os.makedirs(os.path.join(HERE, "stl"), exist_ok=True)
    pieces = {"anillo": build(THREAD_CLEAR, THREAD_LEN)}
    for k, clear in enumerate(TEST_CLEARS, start=1):
        pieces[f"prueba_rosca_{round(clear * 100):02d}"] = build(clear, TEST_LEN, spigot=False, notches=k)
    for name, tris in pieces.items():
        volume = check_mesh(tris, name)
        write_stl(os.path.join(HERE, "stl", f"{name}.stl"), tris)
        r = np.hypot(tris[..., 0], tris[..., 1])
        print(f"{name:18s} Ø {2 * r.max():5.2f} × {tris[..., 2].max():4.2f} mm, {volume / 1000:5.2f} cm³, "
              f"{len(tris)} triángulos")
    # La rosca mide lo que debe: su diámetro mayor es el de M52 menos la holgura.
    z0 = SPIGOT_H + FLANGE_T
    zs = np.linspace(z0 + 1, z0 + THREAD_LEN - 1, 400)
    r_max = thread_radius(np.zeros_like(zs), zs, THREAD_D - THREAD_CLEAR, z0, z0 + THREAD_LEN).max()
    assert abs(2 * r_max - (THREAD_D - THREAD_CLEAR)) < 1e-6
    assert BORE_D < HOLE_D - FIT - 1.0, "el tetón queda sin pared"
    assert BORE_D < THREAD_D - 2 * DEPTH - 3.0, "la rosca queda sin pared"
    previews(pieces["anillo"])
    previews_3d(pieces)


main()
