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
        centre = math.radians(90 + 14 * k)
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


main()
